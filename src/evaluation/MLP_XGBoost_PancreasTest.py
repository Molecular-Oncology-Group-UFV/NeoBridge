import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score, confusion_matrix, classification_report
import xgboost as xgb
import joblib
import warnings
import ast
import os
import optuna
from pathlib import Path
import sys

warnings.filterwarnings('ignore')


# ==============================================================================
# HELPER CLASSES
# ==============================================================================

class ImprovedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.3):
        super(ImprovedMLP, self).__init__()
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
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, val_auc):
        if self.best_score is None:
            self.best_score = val_auc
        elif val_auc < self.best_score + self.delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_auc
            self.epochs_no_improve = 0
        return self.early_stop


class HybridEnsemble:
    def __init__(self, pytorch_model, xgb_model, device):
        self.pytorch_model = pytorch_model
        self.xgb_model = xgb_model
        self.device = device

    def predict_proba(self, X):
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        else:
            X_tensor = X.to(self.device)

        self.pytorch_model.eval()
        with torch.no_grad():
            pytorch_logits = self.pytorch_model(X_tensor)
            pytorch_proba = torch.sigmoid(pytorch_logits).cpu().numpy()

        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]

        return pytorch_proba.flatten(), xgb_proba

    def ensemble_predict_proba(self, X, weights=(0.5, 0.5)):
        pytorch_proba, xgb_proba = self.predict_proba(X)
        total_weight = weights[0] + weights[1]
        normalized_weights = [w / total_weight for w in weights]

        ensemble_proba = (normalized_weights[0] * pytorch_proba +
                          normalized_weights[1] * xgb_proba)

        return ensemble_proba, pytorch_proba, xgb_proba


# ==============================================================================
# MAIN PIPELINE CLASS
# ==============================================================================

class PancreasGeneralizationPipeline:

    def __init__(self):
        # Path Configuration
        self.project_root = Path(__file__).resolve().parent.parent.parent

        # Inputs (Specific Experiment Data)
        self.data_dir = self.project_root / 'data' / 'experiments' / 'pancreas'
        self.train_path = self.data_dir / 'dataset_80_20_no_pancreas.csv'
        self.test_path = self.data_dir / 'dataset_only_pancreas.csv'

        # Outputs
        self.models_dir = self.project_root / 'models' / 'production' / 'pancreas_generalization'
        self.reports_dir = self.project_root / 'results' / 'reports' / 'Optuna_Results'
        self.optuna_csv_path = self.reports_dir / 'optuna_results_pancreas.csv'

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self):
        """Loads train (No Pancreas) and test (Only Pancreas) datasets."""
        if not self.train_path.exists() or not self.test_path.exists():
            raise FileNotFoundError(f"Datasets not found in {self.data_dir}")

        print(f"Loading training data (No Pancreas): {self.train_path}")
        df_train = pd.read_csv(self.train_path)

        print(f"Loading test data (Only Pancreas): {self.test_path}")
        df_test = pd.read_csv(self.test_path)

        # Align columns
        common_cols = list(set(df_train.columns).intersection(set(df_test.columns)))
        df_train = df_train[common_cols]
        df_test = df_test[common_cols]

        return df_train, df_test

    def prepare_features(self, df):
        """Extracts features and target."""
        X = df.select_dtypes(include=['float64', 'int64']).drop(
            columns=['immunogenicity', 'mut_is_binding_pos', 'bestWTMatchType_I', 'DAI_MixMHC_mbp', 'mutant_rank'],
            errors='ignore'
        )
        y = df['immunogenicity']
        return X, y

    def save_optuna_results(self, model_name, params, score, metric):
        """Saves optimization results to CSV."""
        record = {
            'model': model_name,
            'best_params': str(params),
            'result': score,
            'metric': metric,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        df = pd.DataFrame([record])
        if self.optuna_csv_path.exists():
            df.to_csv(self.optuna_csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(self.optuna_csv_path, index=False)
        print(f"✓ Results saved to {self.optuna_csv_path}")

    # --------------------------------------------------------------------------
    # Training Methods
    # --------------------------------------------------------------------------

    def train_pytorch_model(self, X_train, y_train, X_val, y_val, config, save_temp=False):
        hidden_dims = [config[f'hidden_dim_{i}'] for i in range(config['n_layers'])]

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

        model = ImprovedMLP(X_train.shape[1], hidden_dims, config['dropout_rate']).to(self.device)

        # Class balancing
        pos_weight = torch.tensor([8.0]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

        early_stopping = EarlyStopping(patience=config.get('patience', 15))
        best_auc = 0.0

        for epoch in range(config.get('epochs', 100)):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_labels, val_outputs = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = torch.sigmoid(model(inputs))
                    val_labels.extend(labels.cpu().numpy())
                    val_outputs.extend(outputs.cpu().numpy())

            try:
                val_auc = roc_auc_score(val_labels, val_outputs)
            except:
                val_auc = 0.0

            if val_auc > best_auc:
                best_auc = val_auc
                if save_temp:
                    torch.save(model.state_dict(), self.models_dir / 'temp_pytorch_pancreas.pth')

            if early_stopping(val_auc):
                break

        if save_temp and (self.models_dir / 'temp_pytorch_pancreas.pth').exists():
            model.load_state_dict(torch.load(self.models_dir / 'temp_pytorch_pancreas.pth', map_location=self.device))

        model.eval()
        return model, best_auc

    def train_xgboost_model(self, X_train, y_train, params):
        if params is None:
            params = {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 100}

        params.update({'random_state': 42, 'eval_metric': 'logloss', 'verbosity': 0})
        if 'scale_pos_weight' not in params:
            params['scale_pos_weight'] = 8.0

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        return model

    # --------------------------------------------------------------------------
    # Optimization Logic (Optuna)
    # --------------------------------------------------------------------------

    def optimize_pytorch(self, X_train_val, y_train_val, n_trials, metric, use_cv):
        print(f"\nOptimizing PyTorch ({n_trials} trials)...")

        def objective(trial):
            n_layers = trial.suggest_int('n_layers', 1, 4)
            hidden_dims = [trial.suggest_int(f'hidden_dim_{i}', 32, 512, log=True) for i in range(n_layers)]

            config = {
                'n_layers': n_layers,
                'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'epochs': 50,
                'patience': 10
            }
            for i, hdim in enumerate(hidden_dims): config[f'hidden_dim_{i}'] = hdim

            if use_cv:
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scores = []
                for t_idx, v_idx in skf.split(X_train_val, y_train_val):
                    model, best_auc = self.train_pytorch_model(
                        X_train_val[t_idx], y_train_val[t_idx],
                        X_train_val[v_idx], y_train_val[v_idx], config
                    )

                    if metric == "auc_pr":
                        model.eval()
                        with torch.no_grad():
                            probs = torch.sigmoid(model(torch.tensor(X_train_val[v_idx], dtype=torch.float32).to(
                                self.device))).cpu().numpy().flatten()
                        scores.append(average_precision_score(y_train_val[v_idx], probs))
                    else:
                        scores.append(best_auc)
                return np.mean(scores)
            else:
                t_idx, v_idx = next(
                    StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train_val, y_train_val))
                model, best_auc = self.train_pytorch_model(
                    X_train_val[t_idx], y_train_val[t_idx],
                    X_train_val[v_idx], y_train_val[v_idx], config
                )
                return best_auc

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        self.save_optuna_results('PyTorch', study.best_params, study.best_value, metric)
        return study.best_params

    def optimize_xgboost(self, X_train_val, y_train_val, n_trials, metric, use_cv):
        print(f"\nOptimizing XGBoost ({n_trials} trials)...")

        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 4.0, 12.0)
            }

            model = xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss', verbosity=0)

            scoring = 'average_precision' if metric == "auc_pr" else 'roc_auc'
            cv = 3 if use_cv else 5
            scores = cross_val_score(model, X_train_val, y_train_val, cv=cv, scoring=scoring, n_jobs=-1)
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        self.save_optuna_results('XGBoost', study.best_params, study.best_value, metric)
        return study.best_params

    def optimize_weights(self, pytorch_config, xgb_params, X_full, y_full, n_trials, metric, use_cv):
        print("\nOptimizing Ensemble Weights...")

        def objective(trial):
            w_pytorch = trial.suggest_float('weight_pytorch', 0.1, 0.9)
            w_xgb = 1.0 - w_pytorch

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []

            for t_idx, v_idx in skf.split(X_full, y_full):
                py_model, _ = self.train_pytorch_model(X_full[t_idx], y_full[t_idx], X_full[v_idx], y_full[v_idx],
                                                       pytorch_config)
                xgb_model = xgb.XGBClassifier(**xgb_params, random_state=42, eval_metric='logloss', verbosity=0)
                xgb_model.fit(X_full[t_idx], y_full[t_idx])

                ensemble = HybridEnsemble(py_model, xgb_model, self.device)
                probs, _, _ = ensemble.ensemble_predict_proba(X_full[v_idx], (w_pytorch, w_xgb))

                if metric == "auc_pr":
                    scores.append(average_precision_score(y_full[v_idx], probs))
                else:
                    scores.append(roc_auc_score(y_full[v_idx], probs))

            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_w = study.best_params['weight_pytorch']
        weights = (best_w, 1.0 - best_w)
        self.save_optuna_results('Ensemble_Weights', {'weights': weights}, study.best_value, metric)
        return weights

    # --------------------------------------------------------------------------
    # Main Execution
    # --------------------------------------------------------------------------

    def run(self, n_trials_pytorch, n_trials_xgb, n_trials_ensemble, metric, use_cv):
        # 1. Load Data
        df_train, df_test = self.load_data()

        print("\nPreparing Training Data (No Pancreas)...")
        X_train_raw, y_train = self.prepare_features(df_train)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)

        print("Preparing Test Data (Only Pancreas)...")
        X_test_raw, y_test_pancreas = self.prepare_features(df_test)
        X_test_pancreas_scaled = scaler.transform(X_test_raw)

        # 2. Optimization Phase
        print(f"\nStarting Optimization Phase (Metric: {metric.upper()})...")

        best_py_params = self.optimize_pytorch(X_train_scaled, y_train.values, n_trials_pytorch, metric, use_cv)
        best_py_params['epochs'] = 100
        best_py_params['patience'] = 15

        best_xgb_params = self.optimize_xgboost(X_train_scaled, y_train.values, n_trials_xgb, metric, use_cv)

        best_weights = self.optimize_weights(best_py_params, best_xgb_params, X_train_scaled, y_train.values,
                                             n_trials_ensemble, metric, use_cv)

        # 3. Final Training
        print("\nTraining final models on full 'No Pancreas' dataset...")

        t_idx, v_idx = next(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train_scaled, y_train))
        pytorch_model, _ = self.train_pytorch_model(
            X_train_scaled[t_idx], y_train.values[t_idx],
            X_train_scaled[v_idx], y_train.values[v_idx],
            best_py_params, save_temp=True
        )

        xgb_model = self.train_xgboost_model(X_train_scaled, y_train.values, best_xgb_params)

        # 4. Final Evaluation
        print("\n" + "=" * 60)
        print("GENERALIZATION TEST: PANCREAS CANCER COHORT")
        print("=" * 60)

        ensemble = HybridEnsemble(pytorch_model, xgb_model, self.device)
        probs, py_probs, xgb_probs = ensemble.ensemble_predict_proba(X_test_pancreas_scaled, best_weights)

        auc_ens = roc_auc_score(y_test_pancreas, probs)
        auc_py = roc_auc_score(y_test_pancreas, py_probs)
        auc_xgb = roc_auc_score(y_test_pancreas, xgb_probs)
        ap_ens = average_precision_score(y_test_pancreas, probs)

        print(f"PyTorch AUC (Pancreas): {auc_py:.4f}")
        print(f"XGBoost AUC (Pancreas): {auc_xgb:.4f}")
        print(f"Ensemble AUC (Pancreas): {auc_ens:.4f}")
        print(f"Ensemble AP (Pancreas):  {ap_ens:.4f}")
        print(f"Optimal Weights:         PyTorch={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}")

        # 5. Save Artifacts
        print("\nSaving final models...")
        torch.save(pytorch_model.state_dict(), self.models_dir / 'pytorch_model_pancreas.pth')
        xgb_model.save_model(self.models_dir / 'xgboost_model_pancreas.json')
        joblib.dump(ensemble, self.models_dir / 'hybrid_ensemble_pancreas.pkl')
        joblib.dump(scaler, self.models_dir / 'scaler_pancreas.pkl')

        config = {
            'pytorch_params': best_py_params,
            'xgb_params': best_xgb_params,
            'weights': best_weights,
            'metrics_pancreas': {'auc': auc_ens, 'ap': ap_ens}
        }
        joblib.dump(config, self.models_dir / 'optimal_config_pancreas.pkl')
        print(f"Done! Models saved to {self.models_dir}")


def main():
    print("PANCREAS CANCER GENERALIZATION TEST PIPELINE")

    metric = input("\nSelect metric (1: AUC-ROC, 2: AUC-PR) [2]: ") or "2"
    metric = "auc_pr" if metric == "2" else "auc_roc"

    cv_in = input("\nUse Cross-Validation for optimization? (y/n) [y]: ") or "y"
    use_cv = cv_in.lower() == "y"

    # Defaults
    n_py = 20 if use_cv else 30
    n_xgb = 20 if use_cv else 30
    n_ens = 15

    pipeline = PancreasGeneralizationPipeline()
    pipeline.run(n_py, n_xgb, n_ens, metric, use_cv)


if __name__ == "__main__":
    main()