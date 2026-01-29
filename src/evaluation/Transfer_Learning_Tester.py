import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve, f1_score, \
    accuracy_score, roc_curve
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from pathlib import Path
import warnings
import optuna  # IMPORTANTE: Necesitas instalarlo (pip install optuna)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Para que no llene la consola de spam

# Configuración visual
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'DejaVu Sans'
rcParams['figure.titlesize'] = 14


class TransferLearningExperiment:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.data_dir = self.project_root / 'data' / 'experiments'

        # Directorios de salida
        self.results_dir = self.project_root / 'results'
        self.figures_dir = self.results_dir / 'figures' / 'transfer_learning'
        self.reports_dir = self.results_dir / 'reports' / 'transfer_learning'
        self.models_dir = self.project_root / 'models' / 'transfer'

        for d in [self.figures_dir, self.reports_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    # --- Definición de Modelos Internos ---
    class ImprovedMLP(nn.Module):
        def __init__(self, input_dim, hidden_dims, dropout_rate=0.3):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout_rate)
                ])
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    class EarlyStopping:
        def __init__(self, patience=10, delta=0.001):
            self.patience = patience
            self.delta = delta
            self.best_score = None
            self.counter = 0
            self.early_stop = False

        def __call__(self, val_score):
            if self.best_score is None:
                self.best_score = val_score
            elif val_score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = val_score
                self.counter = 0
            return self.early_stop

    class HybridEnsemble:
        def __init__(self, pytorch_model, xgb_model, device):
            self.pytorch_model = pytorch_model
            self.xgb_model = xgb_model
            self.device = device

        def predict_proba(self, X, weights=(0.5, 0.5)):
            if not isinstance(X, torch.Tensor):
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            else:
                X_tensor = X.to(self.device)

            self.pytorch_model.eval()
            with torch.no_grad():
                pt_prob = torch.sigmoid(self.pytorch_model(X_tensor)).cpu().numpy().flatten()

            xgb_prob = self.xgb_model.predict_proba(X)[:, 1]

            total = sum(weights)
            final_prob = (weights[0] / total * pt_prob + weights[1] / total * xgb_prob)
            return final_prob, pt_prob, xgb_prob

    # --- Selección de Datos ---
    def list_datasets(self):
        files = sorted(self.data_dir.rglob("*.csv"))
        datasets = [f for f in files if "optuna" not in f.name and "results" not in f.name]
        return datasets

    def select_datasets_ui(self):
        datasets = self.list_datasets()
        if not datasets:
            raise FileNotFoundError(f"No valid CSV datasets found in {self.data_dir}")

        print("\nAvailable Datasets:")
        for idx, path in enumerate(datasets):
            print(f"[{idx}] {path.parent.name}/{path.name}")

        def get_choice(prompt_text):
            while True:
                try:
                    choice = int(input(prompt_text))
                    if 0 <= choice < len(datasets):
                        return datasets[choice]
                except ValueError:
                    pass
                print("Invalid selection.")

        print("\n--- Configuration ---")
        train_path = get_choice("Select TRAINING dataset ID: ")
        test_path = get_choice("Select TESTING/VALIDATION dataset ID: ")

        train_name = f"{train_path.parent.name}_{train_path.stem.replace('dataset_', '').replace('80_20_', '')}"
        test_name = f"{test_path.parent.name}_{test_path.stem.replace('dataset_', '').replace('80_20_', '')}"
        exp_id = f"{train_name}_VS_{test_name}"

        return train_path, test_path, exp_id

    def load_and_align_data(self, train_path, test_path):
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        common_cols = list(set(df_train.columns).intersection(set(df_test.columns)))
        return df_train[common_cols], df_test[common_cols]

    def prepare_features(self, df):
        features_to_drop = ['immunogenicity', 'mut_is_binding_pos', 'bestWTMatchType_I',
                            'DAI_MixMHC_mbp', 'mutant_rank']
        X = df.select_dtypes(include=['float64', 'int64']).drop(columns=features_to_drop, errors='ignore')
        y = df['immunogenicity'].values
        return X, y

    # --- OPTUNA OPTIMIZATION ---
    def optimize_xgboost(self, X_train, y_train, X_val, y_val, n_trials=20):
        print(f"   > Optimizing XGBoost ({n_trials} trials)...")

        def objective(trial):
            param = {
                'verbosity': 0,
                'objective': 'binary:logistic',
                'booster': 'gbtree',
                'tree_method': 'hist',  # Más rápido
                'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'eta': trial.suggest_float('eta', 1e-3, 0.3, log=True),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300)
            }

            model = xgb.XGBClassifier(**param)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict_proba(X_val)[:, 1]

            # OPTIMIZANDO DIRECTAMENTE PARA AUC-PR
            return average_precision_score(y_val, preds)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        print(f"   ✓ Best XGB AP: {study.best_value:.4f}")
        return study.best_params

    def optimize_mlp(self, X_train, y_train, X_val, y_val, n_trials=20):
        print(f"   > Optimizing PyTorch MLP ({n_trials} trials)...")

        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        X_v = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        pos_weight = torch.tensor([len(y_train) / sum(y_train)] if sum(y_train) > 0 else [1.0]).to(self.device)

        def objective(trial):
            # Hiperparámetros
            n_layers = trial.suggest_int('n_layers', 1, 4)
            hidden_dims = [trial.suggest_int(f'n_units_l{i}', 32, 256) for i in range(n_layers)]
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

            # Modelo
            model = self.ImprovedMLP(X_train.shape[1], hidden_dims, dropout_rate).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            dataset = TensorDataset(X_t, y_t)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Entrenamiento corto para Optuna (max 20 epochs para ir rápido)
            best_ap_trial = 0
            for epoch in range(20):
                model.train()
                for x_b, y_b in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(x_b), y_b)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_probs = torch.sigmoid(model(X_v)).cpu().numpy().flatten()

                # Check metrics
                try:
                    ap = average_precision_score(y_val, val_probs)
                except:
                    ap = 0

                best_ap_trial = max(best_ap_trial, ap)

                # Pruning (parada anticipada si el trial va mal)
                trial.report(ap, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return best_ap_trial

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        print(f"   ✓ Best MLP AP: {study.best_value:.4f}")

        # Reconstruir config structure para el train final
        best_p = study.best_params
        final_config = {
            'n_layers': best_p['n_layers'],
            'lr': best_p['lr'],
            'dropout_rate': best_p['dropout_rate'],
            'batch_size': best_p['batch_size'],
            'weight_decay': 1e-4,  # Fijo
            'epochs': 100,  # Aumentado para train final
            'patience': 15
        }
        for i in range(best_p['n_layers']):
            final_config[f'hidden_dim_{i}'] = best_p[f'n_units_l{i}']

        return final_config

    # --- Training Final ---

    def train_final_mlp(self, X_train, y_train, X_val, y_val, config):
        hidden_dims = [config[f'hidden_dim_{i}'] for i in range(config['n_layers'])]
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        X_v = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

        model = self.ImprovedMLP(X_train.shape[1], hidden_dims, config['dropout_rate']).to(self.device)
        pos_weight = torch.tensor([len(y_train) / sum(y_train)] if sum(y_train) > 0 else [1.0]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])

        stopper = self.EarlyStopping(patience=config['patience'])
        best_state = None
        best_ap = 0

        for epoch in range(config['epochs']):
            model.train()
            for x_b, y_b in loader:
                optimizer.zero_grad()
                loss = criterion(model(x_b), y_b)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_probs = torch.sigmoid(model(X_v)).cpu().numpy().flatten()

            val_ap = average_precision_score(y_val, val_probs)
            if val_ap > best_ap:
                best_ap = val_ap
                best_state = model.state_dict().copy()

            if stopper(val_ap):
                break

        if best_state:
            model.load_state_dict(best_state)
        return model

    def generate_plots(self, val_probs, val_y, test_probs, test_y, exp_id):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # ROC
        fpr_v, tpr_v, _ = roc_curve(val_y, val_probs)
        fpr_t, tpr_t, _ = roc_curve(test_y, test_probs)
        axes[0].plot(fpr_v, tpr_v, label=f'Val AUC={roc_auc_score(val_y, val_probs):.3f}')
        axes[0].plot(fpr_t, tpr_t, label=f'Test AUC={roc_auc_score(test_y, test_probs):.3f}', color='red')
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0].set_title(f'ROC: {exp_id}')
        axes[0].legend()

        # PR
        p_v, r_v, _ = precision_recall_curve(val_y, val_probs)
        p_t, r_t, _ = precision_recall_curve(test_y, test_probs)
        axes[1].plot(r_v, p_v, label=f'Val AP={average_precision_score(val_y, val_probs):.3f}')
        axes[1].plot(r_t, p_t, label=f'Test AP={average_precision_score(test_y, test_probs):.3f}', color='red')
        axes[1].set_title(f'PR Curve: {exp_id}')
        axes[1].legend()

        plt.tight_layout()
        filename = self.figures_dir / f'comparison_{exp_id}.png'
        plt.savefig(filename, dpi=300)
        plt.close()

    def run(self):
        # 1. Interfaz y Carga
        train_path, test_path, exp_id = self.select_datasets_ui()
        df_train_full, df_test_full = self.load_and_align_data(train_path, test_path)

        X_train_raw, y_train_full = self.prepare_features(df_train_full)
        X_test_raw, y_test = self.prepare_features(df_test_full)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # Split para Optuna y Early Stopping
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        t_idx, v_idx = next(skf.split(X_train_scaled, y_train_full))
        X_train, y_train = X_train_scaled[t_idx], y_train_full[t_idx]
        X_val, y_val = X_train_scaled[v_idx], y_train_full[v_idx]

        print(f"\nData loaded. Optimization Target: AUC-PR (Average Precision)")
        print("-" * 60)

        # 2. Optuna Optimization
        print("Starting Hyperparameter Optimization (Optuna)...")
        xgb_best_params = self.optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=30)
        mlp_best_config = self.optimize_mlp(X_train, y_train, X_val, y_val, n_trials=30)
        print("-" * 60)

        # 3. Entrenamiento Final (Con los mejores params)
        print("Training Final Models with optimized parameters...")

        # MLP
        mlp_model = self.train_final_mlp(X_train, y_train, X_val, y_val, mlp_best_config)

        # XGBoost (Entrenamos con todo el train set o split, depende preferencia, aqui usamos split para validar ensemble)
        xgb_model = xgb.XGBClassifier(**xgb_best_params)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

        # 4. Optimización Pesos Ensemble
        ensemble = self.HybridEnsemble(mlp_model, xgb_model, self.device)
        print("Optimizing Ensemble Weights...")

        best_w, best_ap = (0.5, 0.5), 0
        best_val_probs = None
        weights_list = [(i / 10, 1 - i / 10) for i in range(11)]  # De 0.0/1.0 a 1.0/0.0

        for w in weights_list:
            probs, _, _ = ensemble.predict_proba(X_val, w)
            ap = average_precision_score(y_val, probs)
            if ap > best_ap:
                best_ap = ap
                best_w = w
                best_val_probs = probs

        print(f"Optimal Weights: MLP={best_w[0]}, XGB={best_w[1]} (Val AP: {best_ap:.4f})")

        # 5. Test Final
        print("\nEvaluate on Test Set...")
        final_probs, mlp_probs, xgb_probs = ensemble.predict_proba(X_test_scaled, best_w)

        metrics = {
            'AUC_ROC': roc_auc_score(y_test, final_probs),
            'AUC_PR': average_precision_score(y_test, final_probs),
        }

        # Threshold óptimo
        prec, rec, thresholds = precision_recall_curve(y_test, final_probs)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_thresh = thresholds[np.argmax(f1_scores)]
        preds = (final_probs >= best_thresh).astype(int)

        print("\n" + "=" * 40)
        print(f"FINAL RESULTS: {exp_id}")
        print("=" * 40)
        print(f"Ensemble AUC-PR: {metrics['AUC_PR']:.4f}")
        print(f"Ensemble ROC:    {metrics['AUC_ROC']:.4f}")
        print(f"F1-Score:        {f1_score(y_test, preds):.4f}")

        # Guardar todo
        self.generate_plots(best_val_probs, y_val, final_probs, y_test, exp_id)

        res_df = pd.DataFrame([{
            'experiment_id': exp_id,
            'auc_pr': metrics['AUC_PR'],
            'auc_roc': metrics['AUC_ROC'],
            'best_mlp_params': str(mlp_best_config),
            'best_xgb_params': str(xgb_best_params),
            'ensemble_weights': str(best_w)
        }])
        res_df.to_csv(self.reports_dir / f'results_{exp_id}.csv', index=False)

        print(f"\n✓ Experiment Complete. Results saved to {self.reports_dir}")


if __name__ == "__main__":
    exp = TransferLearningExperiment()
    exp.run()