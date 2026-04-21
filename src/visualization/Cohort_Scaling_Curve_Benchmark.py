import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
import joblib
import os
import optuna
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
from tqdm import tqdm
import time
from pathlib import Path
import sys

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Matplotlib configuration for publication style
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 10
rcParams['figure.titlesize'] = 14


# ==============================================================================
# MODEL CLASSES (Same as before)
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
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        else:
            X_tensor = X.to(self.device)
        self.pytorch_model.eval()
        with torch.no_grad():
            pytorch_proba = torch.sigmoid(self.pytorch_model(X_tensor)).cpu().numpy().flatten()
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        return pytorch_proba, xgb_proba

    def ensemble_predict_proba(self, X, weights=(0.5, 0.5)):
        p_prob, x_prob = self.predict_proba(X)
        total = sum(weights)
        ensemble_prob = (weights[0] / total * p_prob + weights[1] / total * x_prob)
        return ensemble_prob


# ==============================================================================
# COHORT SCALING BENCHMARK
# ==============================================================================

class CohortScalingBenchmark:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent.parent

        # IMPORTANTE: Asegúrate de que este archivo tiene la columna 'cohort'
        self.data_path = self.project_root / 'data' / 'raw' / 'NeoPepBase.csv'

        self.reports_dir = self.project_root / 'results' / 'reports' / 'cohort_scaling'
        self.figures_dir = self.project_root / 'results' / 'figures' / 'cohort_scaling'

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

    def load_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found at {self.data_path}")

        print(f"Loading data from {self.data_path.name}...")
        df = pd.read_csv(self.data_path)

        # Impute missing cohorts
        df['cohort'] = df['cohort'].fillna('CEDAR')

        # Features X
        drop_cols = ['immunogenicity', 'cohort', 'mutant_rank', 'mut_is_binding_pos', 'bestWTMatchType_I',
                     'DAI_MixMHC_mbp', 'Rank']
        X = df.select_dtypes(include=['float64', 'int64']).drop(columns=drop_cols, errors='ignore').values
        y = df['immunogenicity'].values
        cohorts = df['cohort'].values

        return X, y, cohorts, df['cohort'].unique()

    def train_models(self, X_train, y_train, X_val, y_val, params):
        # 1. Train PyTorch
        cfg_py = params['pytorch_params']
        hidden_dims = [cfg_py[f'hidden_dim_{i}'] for i in range(cfg_py['n_layers'])]

        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)

        model_py = ImprovedMLP(X_train.shape[1], hidden_dims, cfg_py['dropout_rate']).to(self.device)
        opt = optim.Adam(model_py.parameters(), lr=cfg_py['lr'], weight_decay=cfg_py['weight_decay'])
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0]).to(self.device))  # Fixed weight for speed

        # Quick training loop (no early stopping complex logic for scaling curves to save time)
        model_py.train()
        for _ in range(cfg_py['epochs']):  # Use fixed epochs from optimization
            opt.zero_grad()
            out = model_py(X_t)
            loss = crit(out, y_t)
            loss.backward()
            opt.step()

        # 2. Train XGBoost
        cfg_xgb = params['xgb_params']
        model_xgb = xgb.XGBClassifier(**cfg_xgb)
        model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        return model_py, model_xgb

    def get_default_params(self):
        # Default optimized parameters to save time (or load from file)
        # Puedes reemplazar esto cargando tu pickle anterior
        return {
            'pytorch_params': {
                'n_layers': 2, 'hidden_dim_0': 128, 'hidden_dim_1': 64,
                'lr': 0.001, 'dropout_rate': 0.3, 'batch_size': 64,
                'weight_decay': 1e-4, 'epochs': 40
            },
            'xgb_params': {
                'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.05,
                'scale_pos_weight': 8.0, 'eval_metric': 'logloss',
                'verbosity': 0, 'use_label_encoder': False
            },
            'ensemble_weights': (0.5, 0.5)
        }

    def calculate_cohort_scaling(self, X, y, cohorts, unique_cohorts, params, scaling_points):
        results = []

        # Filtramos puntos de escala que sean posibles
        max_data = len(y)
        scaling_points = [p for p in scaling_points if p < max_data]
        if max_data not in scaling_points: scaling_points.append(max_data)

        print(f"\nStarting Cohort Analysis on: {unique_cohorts}")

        for target_cohort in unique_cohorts:
            print(f"\n>>> Analyzing Target: {target_cohort}")

            # 1. SPLIT: Target Cohort vs The Rest
            is_target = (cohorts == target_cohort)

            X_test_cohort = X[is_target]
            y_test_cohort = y[is_target]

            X_train_pool = X[~is_target]
            y_train_pool = y[~is_target]

            # Skip if cohort is too small for meaningful testing
            if len(y_test_cohort) < 10:
                print(f"Skipping {target_cohort} (too small)")
                continue

            pool_size = len(y_train_pool)
            print(f"    Test Size: {len(y_test_cohort)} | Training Pool: {pool_size}")

            # Baseline (Random)
            baseline = y_test_cohort.mean()

            # 2. SCALING LOOP
            current_cohort_points = [p for p in scaling_points if p <= pool_size]
            if pool_size not in current_cohort_points: current_cohort_points.append(pool_size)

            for n_samples in tqdm(current_cohort_points, desc=f"Scaling {target_cohort}"):

                # Subsample training data
                # We use stratify to keep balance, but strictly from the pool (NOT target cohort)
                if n_samples < pool_size:
                    X_train, _, y_train, _ = train_test_split(
                        X_train_pool, y_train_pool, train_size=n_samples,
                        random_state=42, stratify=y_train_pool
                    )
                else:
                    X_train, y_train = X_train_pool, y_train_pool

                # Split for internal validation (for XGBoost eval_set)
                X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

                # IMPORTANT: Scale here to avoid leakage
                scaler = StandardScaler()
                X_t_scaled = scaler.fit_transform(X_t)
                X_v_scaled = scaler.transform(X_v)
                X_test_scaled = scaler.transform(X_test_cohort)

                # Train
                model_py, model_xgb = self.train_models(X_t_scaled, y_t, X_v_scaled, y_v, params)

                # Ensemble Predict
                ensemble = HybridEnsemble(model_py, model_xgb, self.device)
                probs = ensemble.ensemble_predict_proba(X_test_scaled, params['ensemble_weights'])

                # Metrics
                auc_pr = average_precision_score(y_test_cohort, probs)
                auc_roc = roc_auc_score(y_test_cohort, probs)

                results.append({
                    'target_cohort': target_cohort,
                    'train_size': n_samples,
                    'auc_pr': auc_pr,
                    'auc_roc': auc_roc,
                    'baseline_pr': baseline
                })

        return pd.DataFrame(results)

    def plot_cohort_scaling(self, df):
        # Plot AUC-PR
        plt.figure(figsize=(12, 8))

        cohorts = df['target_cohort'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(cohorts)))

        for i, cohort in enumerate(cohorts):
            subset = df[df['target_cohort'] == cohort]
            subset = subset.sort_values('train_size')

            plt.plot(subset['train_size'], subset['auc_pr'], 'o-',
                     label=f"{cohort} (PR)", color=colors[i], linewidth=2)

            # Plot baseline as dashed line same color
            baseline = subset['baseline_pr'].iloc[0]
            plt.axhline(y=baseline, color=colors[i], linestyle=':', alpha=0.5)

        plt.xlabel('Training Set Size (External Data)', fontweight='bold')
        plt.ylabel('AUC-PR on Target Cohort', fontweight='bold')
        plt.title('Cohort Generalization Scaling: Is it a Data Quantity Issue?', fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'cohort_scaling_auc_pr.png', dpi=300)
        plt.close()

        print(f"Plot saved to {self.figures_dir}")

    def run(self):
        print("=" * 80)
        print("COHORT SCALING ANALYSIS: DATA QUANTITY VS DISTRIBUTION SHIFT")
        print("=" * 80)

        # 1. Load
        X, y, cohorts, unique_cohorts = self.load_data()

        # 2. Config
        params = self.get_default_params()
        # Puntos de escala (ajusta según tu cantidad total de datos)
        scaling_points = [500, 1000, 2000, 4000, 8000, 12000, 15000, 20000]

        # 3. Calc
        results_df = self.calculate_cohort_scaling(X, y, cohorts, unique_cohorts, params, scaling_points)

        # 4. Save & Plot
        results_df.to_csv(self.reports_dir / 'cohort_scaling_results.csv', index=False)
        self.plot_cohort_scaling(results_df)

        # 5. Interpretation Helper
        print("\nINTERPRETATION GUIDE:")
        print(
            "1. RISING CURVE at the end: The model is 'Data Hungry'. Adding more external data will improve prediction on this cohort.")
        print(
            "2. FLAT CURVE (Plateau): The model has hit a 'Generalization Ceiling'. More of the SAME data type won't help. You need data specifically from this cohort or domain adaptation.")


if __name__ == "__main__":
    benchmark = CohortScalingBenchmark()
    benchmark.run()