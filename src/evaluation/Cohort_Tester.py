import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from pathlib import Path
import optuna
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Figure settings
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'DejaVu Sans'
rcParams['figure.titlesize'] = 14


# ==============================================================================
# MODEL ARCHITECTURES (Sin cambios)
# ==============================================================================

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
        if total == 0: total = 1e-6
        final_prob = (weights[0] / total * pt_prob + weights[1] / total * xgb_prob)
        return final_prob, pt_prob, xgb_prob


# ==============================================================================
# COHORT GENERALIZATION BENCHMARK
# ==============================================================================

class CohortGeneralizationBenchmark:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.data_path = self.project_root / 'data' / 'processed' / 'dataset_80_20_normalized_tcr_with_cohort_and_rank.csv'
        self.base_figs_dir = self.project_root / 'results' / 'figures' / 'cohort_generalization'
        self.base_reports_dir = self.project_root / 'results' / 'reports' / 'cohort_generalization'

        for d in [self.base_figs_dir, self.base_reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Hardware Acceleration: {self.device}")

    def load_and_preprocess_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {self.data_path}")

        print(f"[INFO] Loading dataset: {self.data_path.name}")
        df = pd.read_csv(self.data_path)
        df['cohort'] = df['cohort'].fillna('CEDAR')
        return df, df['cohort'].unique()

    def prepare_features(self, df):
        drop_cols = ['immunogenicity', 'cohort', 'mutant_rank', 'mut_is_binding_pos', 'bestWTMatchType_I',
                     'DAI_MixMHC_mbp']
        X = df.select_dtypes(include=['float64', 'int64']).drop(columns=drop_cols, errors='ignore').values
        y = df['immunogenicity'].values
        return X, y

    def fast_optimize_mlp(self, X_train, y_train, X_val, y_val, pos_weight):
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        weight_tensor = torch.tensor([pos_weight]).to(self.device)

        def objective(trial):
            n_layers = trial.suggest_int('n_layers', 2, 3)
            dims = [trial.suggest_int(f'd{i}', 64, 256) for i in range(n_layers)]
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

            model = ImprovedMLP(X_train.shape[1], dims, 0.3).to(self.device)
            opt = optim.Adam(model.parameters(), lr=lr)
            crit = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

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

    def optimize_weights_optuna(self, ensemble, X_val, y_val):
        def objective(trial):
            w_pytorch = trial.suggest_float('w_pytorch', 0.0, 1.0)
            probs, _, _ = ensemble.predict_proba(X_val, (w_pytorch, 1.0 - w_pytorch))
            return average_precision_score(y_val, probs)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30)
        best_w = study.best_params['w_pytorch']
        return (best_w, 1.0 - best_w)

    def train_final_models(self, X_train, y_train, X_val, y_val, mlp_cfg, pos_weight):
        model_mlp = ImprovedMLP(X_train.shape[1], mlp_cfg['dims'], 0.3).to(self.device)
        optimizer = optim.Adam(model_mlp.parameters(), lr=mlp_cfg['lr'])
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(self.device))
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)

        for _ in range(50):
            model_mlp.train()
            optimizer.zero_grad()
            loss = criterion(model_mlp(X_t), y_t)
            loss.backward()
            optimizer.step()

        model_xgb = xgb.XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.05,
            scale_pos_weight=pos_weight, eval_metric='logloss', verbosity=0
        )
        model_xgb.fit(X_train, y_train)

        return model_mlp, model_xgb

    # ==========================================================================
    # FUNCIONES DE RANKING CORREGIDAS (TTF = PRECISION @ K)
    # ==========================================================================

    def analyze_ranking_performance(self, y_true, y_probs, cohort_name):
        """
        Calcula TTF20, TTF50, TTF100 (Precisión en el corte K).
        """
        df = pd.DataFrame({'target': y_true, 'prob': y_probs})
        # Ordenar descendente por probabilidad (Ranking)
        df = df.sort_values(by='prob', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1

        total_positives = df['target'].sum()
        if total_positives == 0:
            # Caso borde: sin positivos en el test set
            return {'TTF20': 0, 'TTF50': 0, 'TTF100': 0, 'Total_Positives': 0}, df

        # Calcular acumulados
        df['cumsum_pos'] = df['target'].cumsum()
        df['precision_at_k'] = df['cumsum_pos'] / df['rank']
        df['recall_at_k'] = df['cumsum_pos'] / total_positives

        # Función auxiliar segura para obtener métrica en corte K
        def get_metric_at_k(dataframe, k, metric_col):
            if len(dataframe) >= k:
                return dataframe.loc[k - 1, metric_col]
            else:
                # Si la lista es menor que K, devolvemos el valor del último elemento
                return dataframe.iloc[-1][metric_col]

        # Extraer métricas TTF (Precision @ K)
        metrics = {
            'TTF20': get_metric_at_k(df, 20, 'precision_at_k'),
            'TTF50': get_metric_at_k(df, 50, 'precision_at_k'),
            'TTF100': get_metric_at_k(df, 100, 'precision_at_k'),
            'Total_Positives': total_positives
        }

        return metrics, df

    def plot_ranking_curves(self, df_rank, metrics, output_dir, cohort_name):
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot hasta rank 250 o longitud total
        max_rank = min(250, len(df_rank))
        data_plot = df_rank.iloc[:max_rank]

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Título global con tu métrica clave TTF20
        fig.suptitle(f'Ranking Performance: {cohort_name}\n(TTF20 [Prec@20]: {metrics["TTF20"]:.3f})', fontsize=14)

        # --- Gráfico Izquierdo: Precision at Rank K (TTF Analysis) ---
        ax1 = axes[0]
        ax1.plot(data_plot['rank'], data_plot['precision_at_k'], label='Model Precision', linewidth=2, color='#1f77b4')

        baseline = metrics['Total_Positives'] / len(df_rank)
        ax1.axhline(y=baseline, color='gray', linestyle='-.', label=f'Baseline ({baseline:.3f})')

        # Marcar el punto TTF20
        if len(data_plot) >= 20:
            ax1.axvline(x=20, color='red', linestyle='--', alpha=0.5)
            ax1.plot(20, metrics['TTF20'], 'ro', markersize=8, label=f'TTF20: {metrics["TTF20"]:.3f}')

        ax1.set_xlabel('Rank K')
        ax1.set_ylabel('Cumulative Precision')
        ax1.set_title('Precision at Rank K (TTF Focus)')
        ax1.set_ylim(0, 1.05)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- Gráfico Derecho: Recall at Rank K ---
        ax2 = axes[1]
        ax2.plot(data_plot['rank'], data_plot['recall_at_k'], label='Recall Accumulation', linewidth=2, color='#2ca02c')
        ax2.plot([0, max_rank], [0, max_rank / len(df_rank)], color='gray', linestyle='-.', label='Random Guessing')

        ax2.set_xlabel('Rank K')
        ax2.set_ylabel('Cumulative Recall')
        ax2.set_title('Recall at Rank K')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_dir / f'ranking_analysis_{cohort_name}.png', dpi=300)
        plt.close()

    # ==========================================================================

    def plot_confusion_matrix(self, y_true, y_probs, output_dir, cohort_name):
        output_dir.mkdir(parents=True, exist_ok=True)
        prec, rec, thresholds = precision_recall_curve(y_true, y_probs)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_thresh = thresholds[np.argmax(f1_scores)]

        preds = (y_probs >= best_thresh).astype(int)
        cm = confusion_matrix(y_true, preds)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix: {cohort_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(output_dir / f'confusion_matrix_{cohort_name}.png', bbox_inches='tight', dpi=300)
        plt.close()

    def plot_comparative_results(self, summary_data):
        if not summary_data: return
        df_res = pd.DataFrame(summary_data)

        csv_path = self.base_reports_dir / 'cohort_ranking_summary.csv'
        df_res.to_csv(csv_path, index=False)
        print(f"\n[INFO] Detailed Summary saved to: {csv_path}")

        # Comparativa visual de TTF20
        plt.figure(figsize=(12, 7))
        df_melt = df_res.melt(id_vars='Target_Cohort', value_vars=['TTF20', 'Baseline_PR'], var_name='Metric',
                              value_name='Score')

        sns.barplot(x='Target_Cohort', y='Score', hue='Metric', data=df_melt,
                    palette={'TTF20': '#d62728', 'Baseline_PR': '#7f7f7f'})
        plt.title('TTF20 (Precision@20) vs Baseline per Cohort')
        plt.ylabel('Score')
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.base_figs_dir / 'cohort_comparison_ttf20.png', dpi=300)
        plt.close()

    def run_benchmark(self):
        print("=" * 70)
        print("STARTING COHORT BENCHMARK (TTF20/50/100 Analysis)")
        print("=" * 70)

        df, cohorts = self.load_and_preprocess_data()
        summary = []

        for target_cohort in cohorts:
            print(f"\n>>> Processing Target Cohort: {target_cohort}")

            # 1. Data Split
            train_mask = df['cohort'] != target_cohort
            test_mask = df['cohort'] == target_cohort
            df_train_full = df[train_mask]
            df_test = df[test_mask]

            if len(df_test) < 10:
                continue

            # 2. Check Positive Ratios
            train_pos_ratio = df_train_full['immunogenicity'].mean()
            test_pos_ratio = df_test['immunogenicity'].mean()
            pos_weight = (1 - train_pos_ratio) / train_pos_ratio if train_pos_ratio > 0 else 1.0

            # 3. Features
            X_train_raw, y_train_full = self.prepare_features(df_train_full)
            X_test_raw, y_test = self.prepare_features(df_test)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)

            # 4. Train/Val Split
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            t_idx, v_idx = next(skf.split(X_train_scaled, y_train_full))
            X_train, y_train = X_train_scaled[t_idx], y_train_full[t_idx]
            X_val, y_val = X_train_scaled[v_idx], y_train_full[v_idx]

            print(f"    Train: {len(X_train)} | Test: {len(X_test_scaled)} ({test_pos_ratio:.1%} pos)")

            # 5. Train Models
            mlp_cfg = self.fast_optimize_mlp(X_train, y_train, X_val, y_val, pos_weight)
            mlp_model, xgb_model = self.train_final_models(X_train, y_train, X_val, y_val, mlp_cfg, pos_weight)

            # 6. Ensemble
            ensemble = HybridEnsemble(mlp_model, xgb_model, self.device)
            best_w = self.optimize_weights_optuna(ensemble, X_val, y_val)
            test_probs, _, _ = ensemble.predict_proba(X_test_scaled, best_w)

            # 7. TTF Analysis
            rank_metrics, df_rank = self.analyze_ranking_performance(y_test, test_probs, target_cohort)
            ap = average_precision_score(y_test, test_probs)

            # Save
            out_dir = self.base_figs_dir / str(target_cohort)
            self.plot_confusion_matrix(y_test, test_probs, out_dir, target_cohort)
            self.plot_ranking_curves(df_rank, rank_metrics, out_dir, target_cohort)

            summary.append({
                'Target_Cohort': target_cohort,
                'AUC_PR': ap,
                'Baseline_PR': test_pos_ratio,
                'TTF20': rank_metrics['TTF20'],
                'TTF50': rank_metrics['TTF50'],
                'TTF100': rank_metrics['TTF100'],
                'Total_Pos': rank_metrics['Total_Positives']
            })

            print(
                f"    ✓ TTF20: {rank_metrics['TTF20']:.3f} | TTF50: {rank_metrics['TTF50']:.3f} | TTF100: {rank_metrics['TTF100']:.3f}")

        self.plot_comparative_results(summary)
        print(f"\n[INFO] Benchmark Completed.")


if __name__ == "__main__":
    bench = CohortGeneralizationBenchmark()
    bench.run_benchmark()