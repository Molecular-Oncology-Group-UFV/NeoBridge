import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from pathlib import Path
import optuna
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Figure configuration for publication
plt.style.use('seaborn-v0_8-whitegrid')
rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.titlesize': 18
})


# ==============================================================================
# MODEL DEFINITIONS
# ==============================================================================

class ImprovedMLP(nn.Module):
    """
    Multi-Layer Perceptron with Batch Normalization and Dropout for regularization.
    """

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


class HybridEnsemble:
    """
    Ensemble combining PyTorch MLP and XGBoost predictions via weighted averaging.
    """

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

        # Weighted average of probabilities
        total = sum(weights)
        final_prob = (weights[0] / total * pt_prob + weights[1] / total * xgb_prob)
        return final_prob, pt_prob, xgb_prob


# ==============================================================================
# GENERALIZATION BENCHMARK PIPELINE
# ==============================================================================

class GeneralizationBenchmark:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.data_dir = self.project_root / 'data' / 'experiments'

        # Output directories
        self.base_figs_dir = self.project_root / 'results' / 'figures' / 'generalization'
        self.base_reports_dir = self.project_root / 'results' / 'reports' / 'generalization'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cohort configuration mapping based on file structure
        self.experiments_config = {
            'melanoma': {
                'folder': 'melanoma',
                'train': 'dataset_80_20_no_melanoma.csv',
                'test': 'dataset_80_20_only_melanoma.csv',
                'title': 'Melanoma Cohort'
            },
            'pancreas': {
                'folder': 'pancreas',
                'train': 'dataset_80_20_no_pancreas.csv',
                'test': 'dataset_only_pancreas.csv',
                'title': 'Pancreatic Cancer Cohort'
            },
            'lung': {
                'folder': 'pulmon',
                'train': 'dataset_80_20_no_lung.csv',
                'test': 'dataset_only_lung.csv',
                'title': 'Lung Cancer Cohort'
            },
            'minor_cancers': {
                'folder': 'multiples_minoritarios',
                'train': 'dataset_80_20_major_cancers.csv',
                'test': 'dataset_80_20_minor_cancers.csv',
                'title': 'Minor Cancers Cohort'
            }
        }

    def load_data(self, config):
        folder = self.data_dir / config['folder']
        train_path = folder / config['train']
        test_path = folder / config['test']

        if not train_path.exists() or not test_path.exists():
            print(f"[-] Warning: Skipping {config['folder']}. Files not found.")
            return None, None

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        # Ensure feature alignment
        common = list(set(df_train.columns).intersection(set(df_test.columns)))
        return df_train[common], df_test[common]

    def prepare_features(self, df):
        """Extracts numerical features and target labels."""
        drop_cols = ['immunogenicity', 'mut_is_binding_pos', 'bestWTMatchType_I',
                     'DAI_MixMHC_mbp', 'mutant_rank']
        X = df.select_dtypes(include=['float64', 'int64']).drop(columns=drop_cols, errors='ignore').values
        y = df['immunogenicity'].values
        return X, y

    def fast_optimize_mlp(self, X_train, y_train, X_val, y_val):
        """
        Performs a rapid hyperparameter search (10 trials) for the MLP architecture.
        Optimizes for Average Precision on the validation set.
        """
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)

        def objective(trial):
            n_layers = trial.suggest_int('n_layers', 2, 3)
            dims = [trial.suggest_int(f'd{i}', 64, 256) for i in range(n_layers)]
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

            model = ImprovedMLP(X_train.shape[1], dims, 0.3).to(self.device)
            opt = optim.Adam(model.parameters(), lr=lr)
            crit = nn.BCEWithLogitsLoss()

            # Short training routine for rapid evaluation
            for _ in range(15):
                model.train()
                opt.zero_grad()
                crit(model(X_t), y_t).backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(torch.tensor(X_val, dtype=torch.float32).to(self.device))).cpu().numpy()
            return average_precision_score(y_val, probs)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)

        bp = study.best_params
        return {'n_layers': bp['n_layers'], 'dims': [bp[f'd{i}'] for i in range(bp['n_layers'])], 'lr': bp['lr']}

    def train_final_models(self, X_train, y_train, X_val, y_val, mlp_cfg):
        """Trains the final MLP (PyTorch) and XGBoost models."""

        # 1. Train MLP
        model_mlp = ImprovedMLP(X_train.shape[1], mlp_cfg['dims'], 0.3).to(self.device)
        optimizer = optim.Adam(model_mlp.parameters(), lr=mlp_cfg['lr'])
        # Weighted loss to handle potential class imbalance
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0]).to(self.device))

        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)

        # Full training routine (50 epochs)
        for _ in range(50):
            model_mlp.train()
            optimizer.zero_grad()
            loss = criterion(model_mlp(X_t), y_t)
            loss.backward()
            optimizer.step()

        # 2. Train XGBoost
        model_xgb = xgb.XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.05,
            scale_pos_weight=8.0, eval_metric='logloss', verbosity=0
        )
        model_xgb.fit(X_train, y_train)

        return model_mlp, model_xgb

    def save_plots(self, val_y, val_probs, test_y, test_probs, output_dir, cancer_title):
        """Generates and saves publication-quality ROC and Confusion Matrix plots."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # ==========================================
        # 1. ROC Curve Comparison (Publication Quality)
        # ==========================================
        fig, ax = plt.subplots(figsize=(8, 6.5))

        fpr_v, tpr_v, _ = roc_curve(val_y, val_probs)
        auc_v = roc_auc_score(val_y, val_probs)

        fpr_t, tpr_t, _ = roc_curve(test_y, test_probs)
        auc_t = roc_auc_score(test_y, test_probs)

        # Líneas
        ax.plot(fpr_v, tpr_v, linestyle='--', color='#1f77b4', linewidth=2,
                 label=f'Validation (Known Cancers) AUC={auc_v:.2f}')
        ax.plot(fpr_t, tpr_t, linestyle='-', color='#d62728', linewidth=3,
                 label=f'Generalization ({cancer_title}) AUC={auc_t:.2f}')
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--', alpha=0.7)

        # Textos y Etiquetas
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve: Generalization to {cancer_title}', pad=20)

        # Caja de estadísticas superior
        total_samples = len(test_y)
        pos_pct = (sum(test_y) / total_samples) * 100
        stats_text = f"Total Samples: {total_samples} | Positive: {pos_pct:.1f}%"

        # CORRECCIÓN ROC: Usamos coordenadas relativas al eje (ax.transAxes).
        # y=1.15 asegura que la caja esté siempre un 15% por encima del borde superior del gráfico,
        # totalmente separada del título.
        ax.text(0.5, 1.15, stats_text, transform=ax.transAxes,
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#cccccc'))

        ax.legend(loc='lower right', frameon=True, edgecolor='black')
        ax.grid(True, linestyle='--', alpha=0.6)
        sns.despine()

        # CORRECCIÓN ROC: Quitamos el tight_layout() normal que comprimía la figura.
        # Guardamos con bbox_inches='tight' que recalcula dinámicamente el espacio para que quepa la caja.
        plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

        # ==========================================
        # 2. Confusion Matrix (Publication Quality)
        # ==========================================
        prec, rec, thresholds = precision_recall_curve(test_y, test_probs)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_thresh = thresholds[np.argmax(f1_scores)]
        preds = (test_probs >= best_thresh).astype(int)

        cm = confusion_matrix(test_y, preds)

        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                         annot_kws={"size": 14})  # Números grandes

        ax.set_xlabel('Predicted', fontsize=14, labelpad=10)
        ax.set_ylabel('True', fontsize=14, labelpad=10)
        ax.set_xticklabels(['0', '1'], fontsize=12)
        ax.set_yticklabels(['0', '1'], fontsize=12, rotation=0)

        plt.title(f'Confusion Matrix ({cancer_title})\nThresh: {best_thresh:.3f}', fontsize=16, pad=15)

        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def run_all(self):
        print("=" * 60)
        print("STARTING GENERALIZATION BENCHMARK")
        print("=" * 60)

        summary = []

        for key, config in self.experiments_config.items():
            print(f"\n[+] Processing Cohort: {config['title']}")

            # Data Loading
            df_train_full, df_test = self.load_data(config)
            if df_train_full is None: continue

            X_train_raw, y_train_full = self.prepare_features(df_train_full)
            X_test_raw, y_test = self.prepare_features(df_test)

            # Standardization
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)

            # Internal Validation Split
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            t_idx, v_idx = next(skf.split(X_train_scaled, y_train_full))
            X_train, y_train = X_train_scaled[t_idx], y_train_full[t_idx]
            X_val, y_val = X_train_scaled[v_idx], y_train_full[v_idx]

            print("    Running hyperparameter optimization...")
            mlp_cfg = self.fast_optimize_mlp(X_train, y_train, X_val, y_val)
            mlp_model, xgb_model = self.train_final_models(X_train, y_train, X_val, y_val, mlp_cfg)

            # Ensemble Weight Optimization
            ensemble = HybridEnsemble(mlp_model, xgb_model, self.device)
            best_w, best_ap = (0.5, 0.5), 0
            for w in [(0.2, 0.8), (0.5, 0.5), (0.8, 0.2)]:
                probs, _, _ = ensemble.predict_proba(X_val, w)
                ap = average_precision_score(y_val, probs)
                if ap > best_ap: best_w = w

            print(f"    Optimal Ensemble Weights: MLP={best_w[0]}, XGB={best_w[1]}")

            # Final Predictions
            val_probs, _, _ = ensemble.predict_proba(X_val, best_w)
            test_probs, _, _ = ensemble.predict_proba(X_test_scaled, best_w)

            # Save Artifacts
            out_dir = self.base_figs_dir / key
            print(f"    Saving figures to: {out_dir}")
            self.save_plots(y_val, val_probs, y_test, test_probs, out_dir, config['title'])

            auc = roc_auc_score(y_test, test_probs)
            ap = average_precision_score(y_test, test_probs)
            summary.append({
                'Experiment': key,
                'AUC': auc,
                'AP': ap,
                'N_Test_Samples': len(y_test)
            })
            print(f"    ✓ Complete. AUC: {auc:.4f} | AP: {ap:.4f}")

        # Save Summary Report
        self.base_reports_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.base_reports_dir / 'generalization_summary.csv'
        pd.DataFrame(summary).to_csv(csv_path, index=False)

        print("\n" + "=" * 60)
        print(f"BENCHMARK COMPLETED. Summary saved to: {csv_path}")


if __name__ == "__main__":
    bench = GeneralizationBenchmark()
    bench.run_all()