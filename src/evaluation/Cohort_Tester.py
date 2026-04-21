import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, brier_score_loss
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from pathlib import Path
import optuna
import warnings
from sklearn.utils import resample
from scipy.stats import norm

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# CONFIGURACIÓN DE MATPLOTLIB
# ==============================================================================
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
    'legend.title_fontsize': 14,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})


# ==============================================================================
# MODEL ARCHITECTURES
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
        self.data_path = self.project_root / 'data' / 'raw' / 'NeoPepBase.csv'
        self.base_figs_dir = self.project_root / 'results' / 'figures' / 'cohort_generalization'
        self.base_reports_dir = self.project_root / 'results' / 'reports' / 'cohort_generalization'
        self.bench_figs_dir = self.project_root / 'results' / 'figures' / 'benchmarking'

        for d in [self.base_figs_dir, self.base_reports_dir, self.bench_figs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Hardware Acceleration: {self.device}")

    def load_and_preprocess_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {self.data_path}")
        df = pd.read_csv(self.data_path)
        df['cohort'] = df['cohort'].fillna('CEDAR')
        return df, df['cohort'].unique()

    def prepare_features(self, df):
        drop_cols = ['immunogenicity', 'cohort', 'mutant_rank', 'mut_is_binding_pos', 'bestWTMatchType_I',
                     'DAI_MixMHC_mbp']
        X = df.select_dtypes(include=['float64', 'int64']).drop(columns=drop_cols, errors='ignore').values
        y = df['immunogenicity'].values
        return X, y

    def fast_optimize_mlp(self, X_train, y_train, X_val, y_val, pos_weight, n_trials=10, dropout_rate=0.3):
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        weight_tensor = torch.tensor([pos_weight]).to(self.device)

        def objective(trial):
            n_layers = trial.suggest_int('n_layers', 2, 3)
            dims = [trial.suggest_int(f'd{i}', 32, 128) for i in range(n_layers)]
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

            model = ImprovedMLP(X_train.shape[1], dims, dropout_rate).to(self.device)
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
        study.optimize(objective, n_trials=n_trials)
        bp = study.best_params
        return {'n_layers': bp['n_layers'], 'dims': [bp[f'd{i}'] for i in range(bp['n_layers'])], 'lr': bp['lr']}

    def optimize_weights_optuna(self, ensemble, X_val, y_val, n_trials=30):
        def objective(trial):
            w_pytorch = trial.suggest_float('w_pytorch', 0.0, 1.0)
            probs, _, _ = ensemble.predict_proba(X_val, (w_pytorch, 1.0 - w_pytorch))
            return average_precision_score(y_val, probs)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        best_w = study.best_params['w_pytorch']
        return (best_w, 1.0 - best_w)

    def train_final_models(self, X_train, y_train, mlp_cfg, pos_weight):
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

    def train_final_models_regularized(self, X_train, y_train, mlp_cfg, pos_weight):
        model_mlp = ImprovedMLP(X_train.shape[1], mlp_cfg['dims'], dropout_rate=0.5).to(self.device)
        optimizer = optim.Adam(model_mlp.parameters(), lr=mlp_cfg['lr'], weight_decay=1e-4)
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
            n_estimators=100,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.5,
            scale_pos_weight=pos_weight,
            eval_metric='logloss',
            verbosity=0
        )
        model_xgb.fit(X_train, y_train)

        return model_mlp, model_xgb

    def calculate_extended_metrics(self, y_true, y_probs):
        if len(np.unique(y_true)) < 2:
            return {k: 0.0 for k in ['AUROC', 'AUPRC', 'F1', 'PPV', 'Sensitivity', 'Brier']}

        auroc = roc_auc_score(y_true, y_probs)
        auprc = average_precision_score(y_true, y_probs)
        brier = brier_score_loss(y_true, y_probs)

        prec, rec, thresholds = precision_recall_curve(y_true, y_probs)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_idx = np.argmax(f1_scores)

        return {
            'AUROC': auroc,
            'AUPRC': auprc,
            'F1': f1_scores[best_idx],
            'PPV': prec[best_idx],
            'Sensitivity': rec[best_idx],
            'Brier': brier
        }

    def analyze_ranking_performance(self, y_true, y_probs, cohort_name, n_bootstraps=5000):
        df = pd.DataFrame({'target': y_true, 'prob': y_probs})
        df = df.sort_values(by='prob', ascending=False).reset_index(drop=True)

        total_positives = df['target'].sum()
        baseline_prevalence = total_positives / len(y_true) if len(y_true) > 0 else 1e-6

        # Devuelve un array vacío para evitar errores si no hay positivos
        if total_positives == 0:
            return {'TTF50': 0, 'TTF50_lower': 0, 'TTF50_upper': 0, 'Enrichment': 0,
                    'Enrichment_lower': 0, 'Enrichment_upper': 0}, df, np.array([0])

        ttf50_point = df.head(50)['target'].mean() if len(df) >= 50 else df['target'].mean()
        enrichment_point = ttf50_point / baseline_prevalence

        print(f"      [~] Stratified Bootstrapping BCa CIs for {cohort_name} (n={n_bootstraps})...")
        boot_ttf50 = []
        indices = np.arange(len(y_true))

        # 1. Stratified Bootstrap
        for _ in range(n_bootstraps):
            boot_idx = resample(indices, replace=True, n_samples=len(y_true), stratify=y_true)
            boot_target = y_true[boot_idx]
            boot_prob = y_probs[boot_idx]

            sort_idx = np.argsort(boot_prob)[::-1]
            boot_target_sorted = boot_target[sort_idx]

            t50 = boot_target_sorted[:50].mean() if len(boot_target_sorted) >= 50 else boot_target_sorted.mean()
            boot_ttf50.append(t50)

        boot_ttf50 = np.array(boot_ttf50)
        boot_enrichment = boot_ttf50 / baseline_prevalence  # DISTRIBUCIÓN

        # 2. Jackknife O(N)
        N = len(y_true)
        jack_ttf50 = np.zeros(N)
        jack_enrichment = np.zeros(N)
        baseline_sum = np.sum(y_true)

        sort_idx_jack = np.argsort(y_probs)[::-1]
        y_true_sorted = y_true[sort_idx_jack]

        top_k_sum = np.sum(y_true_sorted[:50])
        val_k_plus_1 = y_true_sorted[50] if N > 50 else 0

        for i in range(N):
            new_N = N - 1
            new_baseline = (baseline_sum - y_true_sorted[i]) / new_N if new_N > 0 else 1e-6

            if i < 50:
                new_top_k_sum = top_k_sum - y_true_sorted[i] + val_k_plus_1
                j_ttf = new_top_k_sum / min(50, new_N)
            else:
                j_ttf = top_k_sum / min(50, new_N)

            jack_ttf50[i] = j_ttf
            jack_enrichment[i] = j_ttf / new_baseline

        def get_bca_ci(boot_stats, point_stat, jack_stats, alpha=0.05):
            B = len(boot_stats)
            prop_less = np.sum(boot_stats < point_stat) / B
            prop_less = max(min(prop_less, 1 - 1 / (2 * B)), 1 / (2 * B))
            z0 = norm.ppf(prop_less)

            jack_mean = np.mean(jack_stats)
            diffs = jack_mean - jack_stats
            num = np.sum(diffs ** 3)
            den = 6.0 * (np.sum(diffs ** 2)) ** 1.5
            a = num / den if den > 0 else 0.0

            z_alpha1 = norm.ppf(alpha / 2)
            z_alpha2 = norm.ppf(1 - alpha / 2)

            try:
                q1 = norm.cdf(z0 + (z0 + z_alpha1) / (1 - a * (z0 + z_alpha1)))
                q2 = norm.cdf(z0 + (z0 + z_alpha2) / (1 - a * (z0 + z_alpha2)))
            except:
                q1, q2 = alpha / 2, 1 - alpha / 2

            q1 = max(0.0, min(1.0, q1))
            q2 = max(0.0, min(1.0, q2))

            return np.percentile(boot_stats, q1 * 100), np.percentile(boot_stats, q2 * 100)

        ttf50_lower, ttf50_upper = get_bca_ci(boot_ttf50, ttf50_point, jack_ttf50)
        enr_lower, enr_upper = get_bca_ci(boot_enrichment, enrichment_point, jack_enrichment)

        metrics = {
            'TTF50': ttf50_point,
            'TTF50_lower': ttf50_lower,
            'TTF50_upper': ttf50_upper,
            'Enrichment': enrichment_point,
            'Enrichment_lower': enr_lower,
            'Enrichment_upper': enr_upper,
            'Baseline': baseline_prevalence
        }

        return metrics, df, boot_enrichment

    def plot_confusion_matrix(self, y_true, y_probs, output_dir, cohort_name):
        output_dir.mkdir(parents=True, exist_ok=True)
        prec, rec, thresholds = precision_recall_curve(y_true, y_probs)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_thresh = thresholds[np.argmax(f1_scores)]
        preds = (y_probs >= best_thresh).astype(int)
        cm = confusion_matrix(y_true, preds)

        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                         annot_kws={"size": 16, "weight": "bold"}, linewidths=1, linecolor='black')
        plt.title(f'Confusion Matrix: {cohort_name}\n(Threshold: {best_thresh:.3f})', pad=15)
        ax.set_xlabel('Predicted Label', labelpad=10)
        ax.set_ylabel('True Label', labelpad=10)
        ax.set_xticklabels(['0', '1'])
        ax.set_yticklabels(['0', '1'], rotation=0)
        plt.savefig(output_dir / f'confusion_matrix_{cohort_name}.png', bbox_inches='tight', dpi=300)
        plt.close()

    # ==========================================================================
    # NUEVA FUNCIÓN: 4 GRÁFICAS SEPARADAS (Bar, Swarm, Violin, Jitter)
    # ==========================================================================
    def plot_enrichment_visualizations(self, summary_data, boot_distributions):
        print("\n[INFO] Generando 4 figuras separadas con visualizaciones alternativas...")
        df_res = pd.DataFrame(summary_data)

        # 1. Preparación de etiquetas
        cohort_stats = {
            'CEDAR': '(N=8314, n⁺=3056)',
            'NCI': '(N=2084, n⁺=151)',
            'HiTIDE': '(N=1515, n⁺=48)',
            'TESLA': '(N=703, n⁺=72)'
        }

        cohort_order = df_res['Target_Cohort'].tolist()
        labels_dict = {c: f"{c}\n{cohort_stats.get(c, '')}" for c in cohort_order}

        # 2. Preparar datos para Seaborn (formato largo)
        rows_full = []
        rows_downsampled = []

        for cohort in cohort_order:
            dist = boot_distributions[cohort]
            label = labels_dict[cohort]

            # Todos los puntos para el violín y el jitter plot
            for val in dist:
                rows_full.append({'Cohort': label, 'Enrichment': val})

            # Downsampling severo para el Swarm plot (sino colapsa el motor de dibujo)
            np.random.seed(42)
            sample_size = min(300, len(dist))
            dist_down = np.random.choice(dist, size=sample_size, replace=False)
            for val in dist_down:
                rows_downsampled.append({'Cohort': label, 'Enrichment': val})

        df_full = pd.DataFrame(rows_full)
        df_swarm = pd.DataFrame(rows_downsampled)

        base_color = '#084594'
        scatter_color = '#404040'

        x_pos = np.arange(len(df_res))
        heights = df_res['Enrichment']
        lower_err = heights - df_res['Enrichment_lower']
        upper_err = df_res['Enrichment_upper'] - heights
        x_labels = [labels_dict[c] for c in cohort_order]

        # ---------------------------------------------------------------------
        # FIGURA 1: Gráfico de Barras (Original)
        # ---------------------------------------------------------------------
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        bars = ax1.bar(x_pos, heights, yerr=[lower_err, upper_err], capsize=8,
                       color=base_color, edgecolor='black', linewidth=1.5, alpha=0.85,
                       error_kw={'elinewidth': 2, 'markeredgewidth': 2})

        ax1.axhline(1.0, color='#d62728', linestyle='--', linewidth=2, label='Random Guess (1.0x)')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(x_labels, fontweight='bold', fontsize=11)
        ax1.set_ylabel('Fold Enrichment over Baseline (Top-50)', fontweight='bold', fontsize=14)
        ax1.set_title('Top-50 Enrichment Across Cohorts\n(with 95% BCa Confidence Intervals)', fontweight='bold',
                      fontsize=16, pad=15)
        ax1.legend(loc='upper right', frameon=True, edgecolor='black')
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        sns.despine(ax=ax1)

        for i, bar in enumerate(bars):
            y_upper = df_res['Enrichment_upper'].iloc[i]
            ax1.text(bar.get_x() + bar.get_width() / 2, y_upper + 0.5, f"{heights.iloc[i]:.1f}x",
                     ha='center', va='bottom', fontweight='bold', fontsize=12)

        plt.tight_layout()
        save_path_bar = self.base_figs_dir / 'Cohort_Enrichment_Bar.png'
        plt.savefig(save_path_bar, dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # ---------------------------------------------------------------------
        # FIGURA 2: Swarm Plot (con submuestreo)
        # ---------------------------------------------------------------------
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.swarmplot(data=df_swarm, x='Cohort', y='Enrichment', ax=ax2,
                      color=scatter_color, size=4, alpha=0.8)
        ax2.axhline(1.0, color='#d62728', linestyle='--', linewidth=2, label='Random Guess (1.0x)')
        ax2.set_ylabel('Fold Enrichment (Bootstraps)', fontweight='bold', fontsize=14)
        ax2.set_xlabel('')
        ax2.set_xticklabels(x_labels, fontweight='bold', fontsize=11)
        ax2.set_title('Top-50 Enrichment Distribution\nSwarm Plot (Random 300 bootstraps/cohort)', fontweight='bold',
                      fontsize=16, pad=15)
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        sns.despine(ax=ax2)
        ax2.legend(loc='upper right', frameon=True, edgecolor='black')

        for i, val in enumerate(heights):
            ax2.text(i, val + (val * 0.1), f"{val:.1f}x", ha='center', va='bottom',
                     fontweight='bold', color=base_color, fontsize=12)

        plt.tight_layout()
        save_path_swarm = self.base_figs_dir / 'Cohort_Enrichment_Swarm.png'
        plt.savefig(save_path_swarm, dpi=300, bbox_inches='tight')
        plt.close(fig2)

        # ---------------------------------------------------------------------
        # FIGURA 3: Violin Plot
        # ---------------------------------------------------------------------
        fig3, ax3 = plt.subplots(figsize=(8, 6))

        # AÑADIMOS cut=0 para que las colas no se extiendan más allá de los datos reales
        # AÑADIMOS inner=None para quitar cualquier línea gris automática
        sns.violinplot(data=df_full, x='Cohort', y='Enrichment', ax=ax3,
                       color=base_color, inner=None, cut=0, alpha=0.6, linewidth=1.5)

        # Nuestros bigotes negros reales (95% CI)
        ax3.errorbar(x=x_pos, y=heights, yerr=[lower_err, upper_err],
                     fmt='none', c='black', capsize=8, elinewidth=2, markeredgewidth=2)
        ax3.scatter(x_pos, heights, color='black', s=50, zorder=3)

        ax3.axhline(1.0, color='#d62728', linestyle='--', linewidth=2, label='Random Guess (1.0x)')
        ax3.set_ylabel('Fold Enrichment Density', fontweight='bold', fontsize=14)
        ax3.set_xlabel('')
        ax3.set_xticklabels(x_labels, fontweight='bold', fontsize=11)
        ax3.set_title('Top-50 Enrichment Density\nViolin Plot with Point Estimate & 95% CI', fontweight='bold',
                      fontsize=16, pad=15)

        # Etiquetas de texto (¡Las añadimos para que sea igual a los demás!)
        for i, val in enumerate(heights):
            ax3.text(i, df_res['Enrichment_upper'].iloc[i] + 0.5, f"{val:.1f}x",
                     ha='center', va='bottom', fontweight='bold', fontsize=12)

        ax3.grid(axis='y', linestyle='--', alpha=0.5)
        sns.despine(ax=ax3)
        ax3.legend(loc='upper right', frameon=True, edgecolor='black')

        plt.tight_layout()
        save_path_violin = self.base_figs_dir / 'Cohort_Enrichment_Violin.png'
        plt.savefig(save_path_violin, dpi=300, bbox_inches='tight')
        plt.close(fig3)

        # ---------------------------------------------------------------------
        # FIGURA 4: JITTER PLOT
        # ---------------------------------------------------------------------
        fig4, ax4 = plt.subplots(figsize=(8, 6))

        # Jitter plot usando todos los datos con alta transparencia
        sns.stripplot(data=df_full, x='Cohort', y='Enrichment', ax=ax4,
                      color=base_color, alpha=0.15, jitter=0.25, size=3, zorder=1)

        # Superponer la estimación puntual (media) y los intervalos CI encima del Jitter
        ax4.errorbar(x=x_pos, y=heights, yerr=[lower_err, upper_err],
                     fmt='none', c='black', capsize=15, elinewidth=2.5, markeredgewidth=2.5, zorder=3)
        ax4.scatter(x_pos, heights, color='white', edgecolor='black', s=80, zorder=4, label='Mean Estimate')

        ax4.axhline(1.0, color='#d62728', linestyle='--', linewidth=2, zorder=0, label='Random Guess (1.0x)')
        ax4.set_ylabel('Fold Enrichment (All Bootstraps)', fontweight='bold', fontsize=14)
        ax4.set_xlabel('')
        ax4.set_xticklabels(x_labels, fontweight='bold', fontsize=11)
        ax4.set_title('Top-50 Enrichment Distribution\nJitter Plot with Overlaid 95% CI', fontweight='bold',
                      fontsize=16, pad=15)

        # Etiquetas de texto
        for i, val in enumerate(heights):
            ax4.text(i, df_res['Enrichment_upper'].iloc[i] + 0.5, f"{val:.1f}x",
                     ha='center', va='bottom', fontweight='bold', fontsize=12)

        ax4.grid(axis='y', linestyle='--', alpha=0.5)
        sns.despine(ax=ax4)
        ax4.legend(loc='upper right', frameon=True, edgecolor='black')

        plt.tight_layout()
        save_path_jitter = self.base_figs_dir / 'Cohort_Enrichment_Jitter.png'
        plt.savefig(save_path_jitter, dpi=300, bbox_inches='tight')
        plt.close(fig4)

        print(f"✓ Guardadas 4 imágenes separadas en: {self.base_figs_dir}")

    def plot_neoranking_benchmark(self, summary_data, y_true_all, y_probs_all):
        if not summary_data: return
        df_res = pd.DataFrame(summary_data)

        my_mean_ttf50 = df_res['TTF50'].mean()
        my_mean_auprc = df_res['AUC_PR'].mean()

        nr_metrics = {'TTF': 0.250, 'AUPRC': 0.273}
        color_nr = '#999999'
        color_my = '#084594'

        def plot_individual_metric(metric_name, my_val, nr_val, title, filename, my_label):
            plt.figure(figsize=(6, 6))
            x = ['NeoRanking\n(Benchmark)', my_label]
            y = [nr_val, my_val]
            bars = plt.bar(x, y, color=[color_nr, color_my], edgecolor='black', width=0.6, linewidth=1)
            plt.axhline(nr_val, color='gray', linestyle=':', linewidth=1.5, zorder=0)
            plt.bar_label(bars, fmt='%.3f', padding=5, fontsize=14, fontweight='bold')
            plt.ylabel(metric_name)
            plt.title(title, pad=15)
            plt.ylim(0, max(y) * 1.25)
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            sns.despine()
            path = self.bench_figs_dir / filename
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()

        plot_individual_metric('AUPRC Score', my_mean_auprc, nr_metrics['AUPRC'],
                               'Overall Ranking Quality (AUPRC)', 'Benchmark_NeoRanking_AUPRC.png', 'NeoBridge\n(Mean)')
        plot_individual_metric('TTF Score', my_mean_ttf50, nr_metrics['TTF'],
                               'Ranking Precision: Top Fraction (TTIF)', 'Benchmark_NeoRanking_TTF.png',
                               'NeoBridge\n(Mean Top-50)')

        fig, ax = plt.subplots(figsize=(8, 6.5))
        prec, rec, _ = precision_recall_curve(y_true_all, y_probs_all)
        ax.plot(rec, prec, color=color_my, linewidth=3, label=f'NeoBridge (Mean AUPRC = {my_mean_auprc:.3f})')
        ax.axhline(nr_metrics['AUPRC'], color='#d62728', linestyle='--', linewidth=2.5,
                   label=f"NeoRanking Baseline (AUPRC = {nr_metrics['AUPRC']:.3f})")
        ax.set_xlabel('Recall (Sensitivity)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=16, fontweight='bold')
        ax.set_title('Precision-Recall Curve vs. NeoRanking', fontsize=18, fontweight='bold', pad=20)
        ax.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        sns.despine()
        plt.savefig(self.bench_figs_dir / 'Benchmark_NeoRanking_PR_Curve.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_size_matched_analysis(self, full_data_metrics, size_matched_metrics):
        print("\n[INFO] Generando visualización de Size-Matched Sensitivity...")

        df_sm = pd.DataFrame(size_matched_metrics)
        df_fd = pd.DataFrame(full_data_metrics)

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle('Size-Matched Sensitivity Analysis (N=3000)', y=1.05, fontweight='bold', fontsize=18)

        sns.boxplot(x='Target_Cohort', y='AUPRC', data=df_sm, ax=ax, color='lightblue', showfliers=False)
        sns.stripplot(x='Target_Cohort', y='AUPRC', data=df_sm, ax=ax, color='blue', alpha=0.4, jitter=True)

        for i, cohort in enumerate(df_fd['Target_Cohort'].unique()):
            val = df_fd[df_fd['Target_Cohort'] == cohort]['AUC_PR'].values[0]
            ax.plot(i, val, marker='*', color='red', markersize=15, label='Full Data' if i == 0 else "")
            ax.hlines(val, i - 0.4, i + 0.4, colors='red', linestyles='dashed', alpha=0.5)

        ax.set_title('Out-of-Sample AUPRC: Full Data vs N=3000', pad=15)
        ax.set_ylabel('AUPRC')

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        plt.tight_layout()
        save_path = self.base_figs_dir / 'Size_Matched_Sensitivity_Analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Guardada nueva gráfica: {save_path}")

    def plot_ensemble_weights_per_cohort(self, summary_data):
        print("\n[INFO] Generando visualización de Pesos del Ensemble por Cohorte...")
        df = pd.DataFrame(summary_data)

        fig, ax = plt.subplots(figsize=(9, 6))

        cohorts = df['Target_Cohort']
        # Convertimos a porcentajes
        mlp_weights = df['MLP_Weight'] * 100
        xgb_weights = df['XGB_Weight'] * 100

        x = np.arange(len(cohorts))
        width = 0.5

        # Colores consistentes con tu script de producción
        colors = ['#305F88', '#D95319']  # Azul para MLP, Naranja para XGBoost

        # Crear barras apiladas
        p1 = ax.bar(x, mlp_weights, width, label='Improved MLP', color=colors[0], edgecolor='black', linewidth=1)
        p2 = ax.bar(x, xgb_weights, width, bottom=mlp_weights, label='XGBoost', color=colors[1], edgecolor='black',
                    linewidth=1)

        # Añadir el texto del porcentaje dentro de las barras
        for i in range(len(cohorts)):
            if mlp_weights.iloc[i] > 3:  # Solo imprimir si el bloque es suficientemente grande
                ax.text(x[i], mlp_weights.iloc[i] / 2, f"{mlp_weights.iloc[i]:.1f}%",
                        ha='center', va='center', color='white', fontweight='bold', fontsize=12)
            if xgb_weights.iloc[i] > 3:
                ax.text(x[i], mlp_weights.iloc[i] + xgb_weights.iloc[i] / 2, f"{xgb_weights.iloc[i]:.1f}%",
                        ha='center', va='center', color='white', fontweight='bold', fontsize=12)

        # Línea base para comprobar tu hipótesis (10% que saca en producción)
        ax.axhline(10, color='#d62728', linestyle='--', linewidth=2, label='Production MLP Baseline (10%)')

        # Estilo del gráfico
        ax.set_ylabel('Model Contribution Weight (%)', fontweight='bold', fontsize=14)
        ax.set_title('Ensemble Weight Distribution Across Out-of-Sample Cohorts', fontweight='bold', fontsize=16,
                     pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(cohorts, fontweight='bold', fontsize=12)
        ax.set_ylim(0, 100)

        # Leyenda fuera del gráfico para que no tape los datos
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, edgecolor='black', fontsize=12)

        ax.grid(axis='y', linestyle='--', alpha=0.4)
        sns.despine()

        plt.tight_layout()
        save_path = self.base_figs_dir / 'Cohort_Ensemble_Weights_Contribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Guardada gráfica de contribución de pesos en: {save_path}")

    def run_benchmark(self):
        print("=" * 70)
        print("STARTING COHORT BENCHMARK & SIZE-MATCHED SENSITIVITY")
        print("=" * 70)

        df, cohorts = self.load_and_preprocess_data()
        summary_full_data = []
        summary_size_matched = []

        # Diccionario para almacenar las distribuciones de cada cohorte
        boot_distributions = {}

        all_y_true = []
        all_y_probs = []

        n_iterations = 250
        target_train_size = 3000

        for target_cohort in cohorts:
            print(f"\n>>> Processing Target Cohort: {target_cohort}")
            train_mask = df['cohort'] != target_cohort
            test_mask = df['cohort'] == target_cohort
            df_train_full = df[train_mask]
            df_test = df[test_mask]

            if len(df_test) < 10: continue

            train_pos_ratio = df_train_full['immunogenicity'].mean()
            test_pos_ratio = df_test['immunogenicity'].mean()
            pos_weight = (1 - train_pos_ratio) / train_pos_ratio if train_pos_ratio > 0 else 1.0

            X_train_raw, y_train_full = self.prepare_features(df_train_full)
            X_test_raw, y_test = self.prepare_features(df_test)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)

            # =================================================================
            # FIX: VALIDACIÓN OUT-OF-DISTRIBUTION (OOD) INTERNA
            # =================================================================
            # Extraemos una cohorte distinta a la target para usarla como validación.
            # Así obligamos a Optuna a asignar pesos basados en generalización OOD.
            available_cohorts = df_train_full['cohort'].unique()
            val_cohort_internal = available_cohorts[0]  # Cogemos una cohorte para validar

            print(f"    [!] Usando {val_cohort_internal} como set de validación OOD para Optuna.")

            # Creamos la máscara booleana manteniendo el orden del dataframe
            val_mask = (df_train_full['cohort'] == val_cohort_internal).values
            train_mask_internal = ~val_mask

            X_train = X_train_scaled[train_mask_internal]
            y_train = y_train_full[train_mask_internal]
            X_val = X_train_scaled[val_mask]
            y_val = y_train_full[val_mask]
            # =================================================================

            mlp_cfg = self.fast_optimize_mlp(X_train, y_train, X_val, y_val, pos_weight)
            mlp_model, xgb_model = self.train_final_models(X_train_scaled, y_train_full, mlp_cfg, pos_weight)

            ensemble = HybridEnsemble(mlp_model, xgb_model, self.device)
            best_w = self.optimize_weights_optuna(ensemble, X_val, y_val)

            # --- CÓDIGO MODIFICADO PARA COMPROBAR LA HIPÓTESIS ---
            # Capturamos las probabilidades individuales además de las del ensemble
            test_probs, pt_prob, xgb_prob = ensemble.predict_proba(X_test_scaled, best_w)

            # Calculamos AUPRC real (Out-of-Distribution) para cada uno
            ap_ensemble = average_precision_score(y_test, test_probs)
            ap_mlp = average_precision_score(y_test, pt_prob)
            ap_xgb = average_precision_score(y_test, xgb_prob)

            print(f"\n[METRICAS OOD AISLADAS - {target_cohort}]")
            print(f"  - MLP Solo (AUPRC):     {ap_mlp:.4f}")
            print(f"  - XGBoost Solo (AUPRC): {ap_xgb:.4f}")
            print(f"  - Ensemble (AUPRC):     {ap_ensemble:.4f}")
            # -----------------------------------------------------

            all_y_true.extend(y_test)
            all_y_probs.extend(test_probs)

            # EXTRAEMOS LA DISTRIBUCIÓN BOOTSTRAP (boot_enr)
            rank_metrics, df_rank, boot_enr = self.analyze_ranking_performance(y_test, test_probs, target_cohort)

            # Guardamos en el diccionario general
            boot_distributions[target_cohort] = boot_enr

            summary_full_data.append({
                'Target_Cohort': target_cohort,
                'AUC_PR': ap_ensemble,  # Usamos la métrica que ya calculamos arriba
                'Enrichment': rank_metrics.get('Enrichment', 0),
                'Enrichment_lower': rank_metrics.get('Enrichment_lower', 0),
                'Enrichment_upper': rank_metrics.get('Enrichment_upper', 0),
                'TTF50': rank_metrics.get('TTF50', 0),
                'MLP_Weight': best_w[0],  # Peso del PyTorch Model
                'XGB_Weight': best_w[1]  # Peso del XGBoost
            })

            print(f"    [!] Ejecutando Size-Matched Analysis (n={n_iterations}, N_train={target_train_size})...")

            pool_pos = df_train_full[df_train_full['immunogenicity'] == 1]
            pool_neg = df_train_full[df_train_full['immunogenicity'] == 0]

            n_pos_target = int(target_train_size * train_pos_ratio)
            n_neg_target = target_train_size - n_pos_target

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for i in range(n_iterations):
                sample_pos = pool_pos.sample(n=n_pos_target, replace=False)
                sample_neg = pool_neg.sample(n=n_neg_target, replace=False)
                df_matched = pd.concat([sample_pos, sample_neg]).sample(frac=1, random_state=i)

                X_match_raw, y_match = self.prepare_features(df_matched)

                scaler_matched = StandardScaler()
                X_match_scaled = scaler_matched.fit_transform(X_match_raw)
                X_test_matched_scaled = scaler_matched.transform(X_test_raw)

                t_idx_m, v_idx_m = next(skf.split(X_match_scaled, y_match))

                m_cfg = self.fast_optimize_mlp(
                    X_match_scaled[t_idx_m], y_match[t_idx_m],
                    X_match_scaled[v_idx_m], y_match[v_idx_m],
                    pos_weight, n_trials=3, dropout_rate=0.5
                )

                m_mlp, m_xgb = self.train_final_models_regularized(X_match_scaled, y_match, m_cfg, pos_weight)
                ens_match = HybridEnsemble(m_mlp, m_xgb, self.device)
                bw_match = best_w

                train_probs, _, _ = ens_match.predict_proba(X_match_scaled, bw_match)
                test_probs_m, _, _ = ens_match.predict_proba(X_test_matched_scaled, bw_match)

                in_metrics = self.calculate_extended_metrics(y_match, train_probs)
                out_metrics = self.calculate_extended_metrics(y_test, test_probs_m)

                res_dict = {'Target_Cohort': target_cohort, 'Iteration': i}
                for k, v in in_metrics.items(): res_dict[f'In_Sample_{k}'] = v
                for k, v in out_metrics.items(): res_dict[k] = v

                summary_size_matched.append(res_dict)

        self.plot_neoranking_benchmark(summary_full_data, np.array(all_y_true), np.array(all_y_probs))
        self.plot_size_matched_analysis(summary_full_data, summary_size_matched)

        # LLAMADA A LA FUNCIÓN QUE GENERA 4 ARCHIVOS SEPARADOS
        self.plot_enrichment_visualizations(summary_full_data, boot_distributions)

        self.plot_ensemble_weights_per_cohort(summary_full_data)

        pd.DataFrame(summary_size_matched).to_csv(self.base_reports_dir / 'size_matched_distributions.csv', index=False)
        print(f"\n[INFO] Benchmark Completed. CSV de distribuciones exportado.")



if __name__ == "__main__":
    bench = CohortGeneralizationBenchmark()
    bench.run_benchmark()