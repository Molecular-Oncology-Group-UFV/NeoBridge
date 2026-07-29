import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve, precision_score, \
    recall_score, f1_score
import xgboost as xgb
import warnings
import optuna
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import os

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# CONFIGURACIÓN DE MATPLOTLIB (Estilo Paper)
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300
})


# ==============================================================================
# CLASES AUXILIARES
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


class HybridEnsemble:
    def __init__(self, pytorch_model, xgb_model, device):
        self.pytorch_model = pytorch_model
        self.xgb_model = xgb_model
        self.device = device

    def predict_proba(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device) if not isinstance(X, torch.Tensor) else X.to(
            self.device)
        self.pytorch_model.eval()
        with torch.no_grad():
            pytorch_proba = torch.sigmoid(self.pytorch_model(X_tensor)).cpu().numpy().flatten()
        X_numpy = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        xgb_proba = self.xgb_model.predict_proba(X_numpy)[:, 1]
        return pytorch_proba, xgb_proba

    def ensemble_predict_proba(self, X, weights=(0.1, 0.9)):
        p_prob, x_prob = self.predict_proba(X)
        w1, w2 = weights[0] / sum(weights), weights[1] / sum(weights)
        return (w1 * p_prob + w2 * x_prob)


# ==============================================================================
# MOTOR DE ABLACIÓN Y GRAFICADO
# ==============================================================================
class FeatureAblationEngine:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.data_path = self.project_root / 'data' / 'raw' / 'NeoPepBase.csv'
        self.reports_dir = self.project_root / 'results' / 'reports' / 'benchmarking'
        self.figures_dir = self.project_root / 'results' / 'figures' / 'benchmarking'
        self.delong_preds_path = self.project_root / 'results' / 'reports' / 'performance' / 'test_predictions_for_delong.csv'

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_base_data(self):
        df = pd.read_csv(self.data_path)
        features_to_drop = ['immunogenicity', 'mut_is_binding_pos', 'bestWTMatchType_I',
                            'DAI_MixMHC_mbp', 'mutant_rank', 'hydrophobicity_avg_diff']

        X_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=features_to_drop, errors='ignore')
        y = df['immunogenicity'].values
        return X_df, y

    def train_pytorch_fold(self, X_train, y_train, X_val, y_val, config):
        hidden_dims = [config[f'hidden_dim_{i}'] for i in range(config['n_layers'])]
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        X_v = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
        model = ImprovedMLP(X_train.shape[1], hidden_dims, config['dropout_rate']).to(self.device)

        pos_weight = torch.tensor([len(y_train) / sum(y_train) if sum(y_train) > 0 else 1.0]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

        best_auc = 0
        patience_counter = 0
        for epoch in range(30):
            model.train()
            for inputs, labels in loader:
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_probs = torch.sigmoid(model(X_v)).cpu().numpy().flatten()
            val_auc = roc_auc_score(y_val, val_probs)

            if val_auc > best_auc + 0.0001:
                best_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 5: break
        return best_auc, model

    def run_fast_optimization(self, X, y, model_type, n_trials=10):
        def objective(trial):
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            if model_type == 'PyTorch':
                n_layers = trial.suggest_int('n_layers', 1, 2)
                config = {'n_layers': n_layers, 'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                          'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.4), 'batch_size': 128,
                          'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)}
                for i in range(n_layers): config[f'hidden_dim_{i}'] = trial.suggest_int(f'hidden_dim_{i}', 32, 128)
            else:
                params = {'max_depth': trial.suggest_int('max_depth', 3, 6),
                          'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
                          'n_estimators': 100, 'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0),
                          'eval_metric': 'logloss', 'verbosity': 0, 'n_jobs': 1}

            for train_idx, val_idx in skf.split(X, y):
                X_t, X_v = X[train_idx], X[val_idx]
                y_t, y_v = y[train_idx], y[val_idx]
                if model_type == 'PyTorch':
                    score, _ = self.train_pytorch_fold(X_t, y_t, X_v, y_v, config)
                else:
                    model = xgb.XGBClassifier(**params)
                    model.fit(X_t, y_t)
                    score = roc_auc_score(y_v, model.predict_proba(X_v)[:, 1])
                scores.append(score)
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return study.best_params

    def run_experiment(self, experiment_name, X_df, y):
        print(f"\n[{experiment_name}] Features activas: {X_df.shape[1]}")
        X = X_df.values
        X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_dev = scaler.fit_transform(X_dev)
        X_test = scaler.transform(X_test)

        X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(X_dev, y_dev, test_size=0.1,
                                                                                  stratify=y_dev, random_state=42)

        best_py_params = self.run_fast_optimization(X_dev, y_dev, 'PyTorch', n_trials=10)
        best_xgb_params = self.run_fast_optimization(X_dev, y_dev, 'XGBoost', n_trials=15)

        # --- CORRECCIÓN AQUÍ: Inyectar el batch_size manualmente ---
        best_py_params['batch_size'] = 128
        # -----------------------------------------------------------

        _, pytorch_model = self.train_pytorch_fold(X_train_final, y_train_final, X_val_final, y_val_final,
                                                   best_py_params)
        xgb_model = xgb.XGBClassifier(**best_xgb_params)
        xgb_model.fit(X_train_final, y_train_final)

        ensemble = HybridEnsemble(pytorch_model, xgb_model, self.device)
        probs_test = ensemble.ensemble_predict_proba(X_test, weights=(0.1, 0.9))

        # -----------------------------------------------------------
        # NUEVO: Calcular AUC, AUPRC, Precision y Recall
        # -----------------------------------------------------------
        auc = roc_auc_score(y_test, probs_test)
        auprc = average_precision_score(y_test, probs_test)

        # Binarizar para precision y recall (usando threshold 0.5)
        preds = (probs_test >= 0.5).astype(int)
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)

        print(
            f"[{experiment_name}] DONE | AUC: {auc:.4f} | AUPRC: {auprc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")
        return probs_test, y_test, auc, auprc, precision, recall

    def plot_roc_curves(self, y_true, curves_data, title, filename, is_feature_ablation=False):
        plt.figure(figsize=(9, 7))
        colors = ['#084594', '#D95319', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#7f7f7f']

        for i, (name, probs) in enumerate(curves_data.items()):
            fpr, tpr, _ = roc_curve(y_true, probs)
            auc = roc_auc_score(y_true, probs)

            if 'Full' in name:
                plt.plot(fpr, tpr, linewidth=3.5, color='#084594', label=f"{name} (AUC = {auc:.3f})")
            else:
                plt.plot(fpr, tpr, linewidth=2.0, color=colors[i % len(colors)], label=f"{name} (AUC = {auc:.3f})",
                         alpha=0.85)

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Guess')
        plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold', pad=15)
        plt.legend(loc='lower right', frameon=True, edgecolor='black', fontsize=10 if is_feature_ablation else 12)
        plt.grid(True, linestyle='--', alpha=0.6)
        sns.despine()

        save_path = self.figures_dir / filename
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✓ Gráfico ROC guardado en: {save_path}")

    # -----------------------------------------------------------
    # NUEVA FUNCIÓN: Para graficar Curvas Precision-Recall (AUPRC)
    # -----------------------------------------------------------
    def plot_pr_curves(self, y_true, curves_data, title, filename, is_feature_ablation=False):
        plt.figure(figsize=(9, 7))
        colors = ['#084594', '#D95319', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#7f7f7f']

        for i, (name, probs) in enumerate(curves_data.items()):
            precision, recall, _ = precision_recall_curve(y_true, probs)
            auprc = average_precision_score(y_true, probs)

            if 'Full' in name:
                plt.plot(recall, precision, linewidth=3.5, color='#084594', label=f"{name} (AUPRC = {auprc:.3f})")
            else:
                plt.plot(recall, precision, linewidth=2.0, color=colors[i % len(colors)],
                         label=f"{name} (AUPRC = {auprc:.3f})", alpha=0.85)

        baseline = np.sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline (AP = {baseline:.3f})')

        plt.xlabel('Recall', fontsize=14, fontweight='bold')
        plt.ylabel('Precision', fontsize=14, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold', pad=15)
        plt.legend(loc='lower left', frameon=True, edgecolor='black', fontsize=10 if is_feature_ablation else 12)
        plt.grid(True, linestyle='--', alpha=0.6)
        sns.despine()

        save_path = self.figures_dir / filename
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✓ Gráfico PR guardado en: {save_path}")

    def execute_all_ablations(self):
        print("=" * 60)
        print("INICIANDO ESTUDIOS DE ABLACIÓN Y GRAFICADO (CELL GENOMICS)")
        print("=" * 60)

        metrics_records = []

        # 1. GENERAR FIGURA DE ABLACIÓN DE ARQUITECTURA
        try:
            df_preds = pd.read_csv(self.delong_preds_path)
            y_true_arch = df_preds['y_true']
            arch_curves = {
                'Full Hybrid Ensemble': df_preds['prob_ensemble'],
                'XGBoost Alone': df_preds['prob_xgb'],
                'PyTorch MLP Alone': df_preds['prob_mlp']
            }
            # Generar ROC y PR para Arquitectura
            self.plot_roc_curves(y_true_arch, arch_curves, 'Architecture Ablation Study (ROC)',
                                 'architecture_ablation_ROC.png')
            self.plot_pr_curves(y_true_arch, arch_curves, 'Architecture Ablation Study (PR)',
                                'architecture_ablation_PR.png')

            # Registrar métricas numéricas para reporte
            for name, probs in arch_curves.items():
                auc = roc_auc_score(y_true_arch, probs)
                auprc = average_precision_score(y_true_arch, probs)
                preds = (probs >= 0.5).astype(int)
                prec = precision_score(y_true_arch, preds, zero_division=0)
                rec = recall_score(y_true_arch, preds, zero_division=0)
                metrics_records.append(
                    {'Group': 'Architecture', 'Model': name, 'AUC': auc, 'AUPRC': auprc, 'Precision (t=0.5)': prec,
                     'Recall (t=0.5)': rec})

        except FileNotFoundError:
            print("⚠️ No se encontró el CSV del test de DeLong. Genera primero las predicciones principales.")

        # 2. DEFINIR LOS GRUPOS BIOLÓGICOS (Ablación de Features)
        X_df_base, y = self.load_base_data()
        all_cols = X_df_base.columns.tolist()
        dai_cols = [c for c in all_cols if 'DAI' in c.upper()]
        expr_cols = [c for c in all_cols if any(x in c.upper() for x in ['EXPRESSION', 'TCGA', 'GTEX', 'ABUNDANCE'])]
        tcr_cols = [c for c in all_cols if
                    any(x in c.upper() for x in ['TCR', 'HYDROPH', 'CHARGE', 'PMTNET', 'FOREIGNNESS'])]
        hla_cols = [c for c in all_cols if any(
            x in c.upper() for x in ['MHC', 'TAP', 'THALF', 'STAB', 'NETCHOP', 'PRIME']) and c not in dai_cols]

        experiments = {
            "Ablation: No DAI Features": X_df_base.drop(columns=dai_cols),
            "Baseline: DAI Only": X_df_base[dai_cols],
            "Ablation: No HLA Binding/Processing": X_df_base.drop(columns=hla_cols),
            "Ablation: No Tumor Expression": X_df_base.drop(columns=expr_cols),
            "Ablation: No TCR Recognition": X_df_base.drop(columns=tcr_cols)
        }

        results_dict = {}
        y_test_master = None

        # 3. EJECUTAR ENTRENAMIENTO PARA CADA GRUPO
        for exp_name, X_subset in experiments.items():
            probs, y_test, auc, auprc, prec, rec = self.run_experiment(exp_name, X_subset, y)
            results_dict[exp_name] = probs
            if y_test_master is None: y_test_master = y_test

            metrics_records.append(
                {'Group': 'Features', 'Model': exp_name, 'AUC': auc, 'AUPRC': auprc, 'Precision (t=0.5)': prec,
                 'Recall (t=0.5)': rec})

        # Cargar el full model para compararlo con todos los grupos eliminados
        try:
            df_preds = pd.read_csv(self.delong_preds_path)
            feature_curves = {'Full Model (All Features)': df_preds['prob_ensemble']}
        except FileNotFoundError:
            feature_curves = {}

        feature_curves.update(results_dict)

        if y_test_master is not None:
            # 4. GENERAR FIGURA DE ABLACIÓN DE GRUPOS DE FEATURES (ROC y PR)
            self.plot_roc_curves(y_test_master, feature_curves, 'Major Feature Groups Ablation Study (ROC)',
                                 'feature_ablation_ROC.png', is_feature_ablation=True)
            self.plot_pr_curves(y_test_master, feature_curves, 'Major Feature Groups Ablation Study (PR)',
                                'feature_ablation_PR.png', is_feature_ablation=True)

        # 5. GUARDAR TODAS LAS MÉTRICAS EN UN CSV PARA EL REBUTTAL
        df_metrics = pd.DataFrame(metrics_records)
        metrics_csv_path = self.reports_dir / 'ablation_metrics_rebuttal.csv'
        df_metrics.to_csv(metrics_csv_path, index=False)
        print(
            f"\n✓ CSV con todas las métricas (AUC, AUPRC, Precision, Recall) generado para el Rebuttal en: {metrics_csv_path}")


if __name__ == "__main__":
    engine = FeatureAblationEngine()
    engine.execute_all_ablations()