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

# ==============================================================================
# CONFIGURACIÓN DE MATPLOTLIB PARA PUBLICACIÓN CIENTÍFICA
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
        self.data_path = self.project_root / 'data' / 'processed' / 'dataset_80_20_normalized_tcr_with_cohort_and_rank.csv'
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
    # METRICS & PLOTTING
    # ==========================================================================

    def analyze_ranking_performance(self, y_true, y_probs, cohort_name):
        df = pd.DataFrame({'target': y_true, 'prob': y_probs})
        df = df.sort_values(by='prob', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1

        total_positives = df['target'].sum()
        if total_positives == 0:
            return {'TTF20': 0, 'TTF50': 0, 'TTF100': 0, 'FR': 0, 'Total_Positives': 0}, df

        df['cumsum_pos'] = df['target'].cumsum()
        df['precision_at_k'] = df['cumsum_pos'] / df['rank']
        df['recall_at_k'] = df['cumsum_pos'] / total_positives

        def get_metric_at_k(dataframe, k, metric_col):
            if len(dataframe) >= k:
                return dataframe.loc[k - 1, metric_col]
            else:
                return dataframe.iloc[-1][metric_col]

        pos_in_top_100 = df.iloc[:100]['target'].sum()
        fr_score = pos_in_top_100 / total_positives

        metrics = {
            'TTF20': get_metric_at_k(df, 20, 'precision_at_k'),
            'TTF50': get_metric_at_k(df, 50, 'precision_at_k'),
            'TTF100': get_metric_at_k(df, 100, 'precision_at_k'),
            'FR': fr_score,
            'Total_Positives': total_positives
        }

        return metrics, df

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

    def plot_ranking_curves(self, df_rank, metrics, output_dir, cohort_name):
        output_dir.mkdir(parents=True, exist_ok=True)
        max_rank = min(250, len(df_rank))
        data_plot = df_rank.iloc[:max_rank]

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle(f'Ranking Performance: {cohort_name}\n(TTF20: {metrics["TTF20"]:.3f})', y=1.05)

        # Plot 1
        axes[0].plot(data_plot['rank'], data_plot['precision_at_k'], label='Precision', color='#1f77b4', linewidth=2.5)
        axes[0].axhline(y=metrics['Total_Positives'] / len(df_rank), color='gray', linestyle='-.')
        axes[0].set_title('Precision at Rank K', pad=15)
        axes[0].set_xlabel('Rank')
        axes[0].grid(True, alpha=0.3)

        # Plot 2
        axes[1].plot(data_plot['rank'], data_plot['recall_at_k'], label='Recall', color='#2ca02c', linewidth=2.5)
        axes[1].set_title('Recall at Rank K', pad=15)
        axes[1].set_xlabel('Rank')
        axes[1].grid(True, alpha=0.3)

        plt.savefig(output_dir / f'ranking_analysis_{cohort_name}.png', bbox_inches='tight', dpi=300)
        plt.close()

    def plot_comparative_results(self, summary_data):
        if not summary_data: return
        df_res = pd.DataFrame(summary_data)

        csv_path = self.base_reports_dir / 'cohort_ranking_summary.csv'
        df_res.to_csv(csv_path, index=False)
        print(f"\n[INFO] Detailed Summary saved to: {csv_path}")

        # ======================================================================
        # FIGURA 6A: Generalización por Cohorte (Barplot)
        # ======================================================================
        fig, ax = plt.subplots(figsize=(12, 6))

        df_melt = df_res.melt(id_vars='Target_Cohort', value_vars=['TTF20', 'Baseline_PR'],
                              var_name='Metric', value_name='Score')

        df_melt['Metric'] = df_melt['Metric'].replace({
            'TTF20': 'Model Top-20 Fraction',
            'Baseline_PR': 'Random Baseline'
        })

        sns.barplot(
            x='Target_Cohort', y='Score', hue='Metric', data=df_melt,
            palette={'Model Top-20 Fraction': '#d62728', 'Random Baseline': '#999999'},
            edgecolor='black', linewidth=1, ax=ax
        )

        plt.title('Generalization by Cohort (Top-20 Fraction)', pad=30)
        plt.ylabel('Fraction of Positives (Precision)')
        plt.xlabel('Target Cohort')
        plt.ylim(0, 1.1)

        plt.xticks(rotation=45, ha='right')
        leg = plt.legend(title='', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)

        plt.grid(axis='y', linestyle='--', alpha=0.5)
        sns.despine()

        save_path_a = self.base_figs_dir / 'Figure_6A_Generalization_Barplot.png'
        plt.savefig(save_path_a, dpi=300, bbox_inches='tight', bbox_extra_artists=(leg,))
        plt.close()
        print(f"✓ Saved: {save_path_a}")

        # ======================================================================
        # FIGURA 6B: Tabla de Métricas
        # ======================================================================
        fig_height = len(df_res) * 0.8 + 2
        fig, ax2 = plt.subplots(figsize=(12, fig_height))
        ax2.axis('off')

        cols_table = ['Target_Cohort', 'Total_Pos', 'Baseline_PR', 'TTF20', 'TTF50']
        df_table = df_res[cols_table].copy()
        df_table.columns = ['Cohort', 'Positives (N)', 'Baseline', 'TTF20 (Top 20)', 'TTF50 (Top 50)']

        for col in ['Baseline', 'TTF20 (Top 20)', 'TTF50 (Top 50)']:
            df_table[col] = df_table[col].apply(lambda x: f"{x:.3f}")

        table = ax2.table(
            cellText=df_table.values,
            colLabels=df_table.columns,
            loc='center',
            cellLoc='center',
            bbox=[0, 0, 1, 1]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(13)

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('white')
            cell.set_linewidth(0)
            if row == 0:
                cell.set_text_props(weight='bold', color='black', fontsize=14)

        n_rows = len(df_res) + 1
        row_height = 1.0 / n_rows

        ax2.plot([0, 1], [1, 1], color='black', linewidth=2, transform=ax2.transAxes)
        y_header_bottom = 1.0 - row_height
        ax2.plot([0, 1], [y_header_bottom, y_header_bottom], color='black', linewidth=1, transform=ax2.transAxes)
        ax2.plot([0, 1], [0, 0], color='black', linewidth=2, transform=ax2.transAxes)

        plt.title('Detailed Ranking Metrics by Cohort', pad=15)

        save_path_b = self.base_figs_dir / 'Figure_6B_Metrics_Table.png'
        plt.savefig(save_path_b, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path_b}")

        # ======================================================================
        # FIGURA 6C: Model Enrichment Factor
        # ======================================================================
        plt.figure(figsize=(12, 6))

        df_res['Enrichment'] = df_res['TTF20'] / (df_res['Baseline_PR'] + 1e-6)

        sns.barplot(
            x='Target_Cohort', y='Enrichment', data=df_res,
            color='#2ca02c', edgecolor='black', alpha=0.9, linewidth=1
        )

        plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='No Improvement (Random)')

        plt.title('Model Enrichment Factor (TTF20 / Random Baseline)', pad=15)
        plt.ylabel('Fold Enrichment (x times better)')
        plt.xlabel('Target Cohort')

        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        sns.despine()

        for i, v in enumerate(df_res['Enrichment']):
            plt.text(i, v + (df_res['Enrichment'].max() * 0.02), f"{v:.1f}x",
                     ha='center', fontweight='bold', fontsize=12)

        save_path_c = self.base_figs_dir / 'Figure_6C_Enrichment_Factor.png'
        plt.savefig(save_path_c, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path_c}")

    def plot_neoranking_benchmark(self, summary_data, y_true_all, y_probs_all):
        """
        Genera figuras comparativas contra NeoRanking (Müller et al., 2023).
        """
        if not summary_data: return
        df_res = pd.DataFrame(summary_data)

        print("\n" + "=" * 60)
        print("GENERATING NEORANKING BENCHMARK FIGURES")
        print("=" * 60)

        # Calculamos la MEDIA exacta de las cohortes, como indicaste
        my_mean_ttf20 = df_res['TTF20'].mean()
        my_mean_auprc = df_res['AUC_PR'].mean()

        nr_metrics = {'TTF20': 0.250, 'AUPRC': 0.273}
        color_nr = '#999999'
        color_my = '#084594'

        def plot_individual_metric(metric_name, my_val, nr_val, title, filename):
            plt.figure(figsize=(6, 6))
            x = ['NeoRanking\n(Benchmark)', 'Hybrid Ensemble\n(Mean)']
            y = [nr_val, my_val]

            bars = plt.bar(x, y, color=[color_nr, color_my], edgecolor='black', width=0.6, linewidth=1)
            plt.axhline(nr_val, color='gray', linestyle=':', linewidth=1.5, zorder=0)

            plt.bar_label(bars, fmt='%.3f', padding=5, fontsize=14, fontweight='bold')

            plt.ylabel(metric_name)
            plt.title(title, pad=15)
            plt.ylim(0, max(y) * 1.25)

            plt.xticks(rotation=0)

            plt.grid(axis='y', linestyle='--', alpha=0.3)
            sns.despine()

            path = self.bench_figs_dir / filename
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: {path}")

        # Figura 1 y 2 (Gráficos de Barras originales actualizados a Media)
        plot_individual_metric('AUPRC Score', my_mean_auprc, nr_metrics['AUPRC'],
                               'Overall Ranking Quality (AUPRC)', 'Benchmark_NeoRanking_AUPRC.png')

        plot_individual_metric('TTF20 Score', my_mean_ttf20, nr_metrics['TTF20'],
                               'Ranking Precision: Top-20 Fraction (TTIF)', 'Benchmark_NeoRanking_TTF20.png')

        # ======================================================================
        # FIGURA 3: PRECISION-RECALL CURVE (LIMPIA)
        # ======================================================================
        fig, ax = plt.subplots(figsize=(8, 6.5))

        # Trazamos la forma de la curva con todas las predicciones
        prec, rec, _ = precision_recall_curve(y_true_all, y_probs_all)

        # Dibujar la curva PR de nuestro modelo (Etiquetada con la Media Real)
        ax.plot(rec, prec, color=color_my, linewidth=3,
                 label=f'XGBoost + MLP (Mean AUPRC = {my_mean_auprc:.3f})')

        # Línea de referencia de NeoRanking (AUPRC constante)
        ax.axhline(nr_metrics['AUPRC'], color='#d62728', linestyle='--', linewidth=2.5,
                   label=f"NeoRanking Baseline (AUPRC = {nr_metrics['AUPRC']:.3f})")

        # Formato Paper (Sin caja superior, directo y limpio)
        ax.set_xlabel('Recall (Sensitivity)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=16, fontweight='bold')
        ax.set_title('Precision-Recall Curve vs. NeoRanking', fontsize=18, fontweight='bold', pad=20)

        ax.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        sns.despine()

        save_path_pr = self.bench_figs_dir / 'Benchmark_NeoRanking_PR_Curve.png'
        plt.savefig(save_path_pr, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Figure 3 (PR Curve) saved: {save_path_pr}")

    def run_benchmark(self):
        print("=" * 70)
        print("STARTING COHORT BENCHMARK")
        print("=" * 70)

        df, cohorts = self.load_and_preprocess_data()
        summary = []

        # --- NUEVO: Listas para almacenar predicciones de todas las cohortes ---
        all_y_true = []
        all_y_probs = []

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

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            t_idx, v_idx = next(skf.split(X_train_scaled, y_train_full))
            X_train, y_train = X_train_scaled[t_idx], y_train_full[t_idx]
            X_val, y_val = X_train_scaled[v_idx], y_train_full[v_idx]

            mlp_cfg = self.fast_optimize_mlp(X_train, y_train, X_val, y_val, pos_weight)
            mlp_model, xgb_model = self.train_final_models(X_train, y_train, X_val, y_val, mlp_cfg, pos_weight)

            ensemble = HybridEnsemble(mlp_model, xgb_model, self.device)
            best_w = self.optimize_weights_optuna(ensemble, X_val, y_val)
            test_probs, _, _ = ensemble.predict_proba(X_test_scaled, best_w)

            # --- NUEVO: Guardamos los resultados para la curva PR global ---
            all_y_true.extend(y_test)
            all_y_probs.extend(test_probs)

            rank_metrics, df_rank = self.analyze_ranking_performance(y_test, test_probs, target_cohort)
            ap = average_precision_score(y_test, test_probs)

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
                'FR': rank_metrics['FR'],
                'Total_Pos': rank_metrics['Total_Positives']
            })
            print(f"    ✓ TTF20: {rank_metrics['TTF20']:.3f}")

        self.plot_comparative_results(summary)

        # --- NUEVO: Pasamos los resultados globales para crear la curva ---
        self.plot_neoranking_benchmark(summary, np.array(all_y_true), np.array(all_y_probs))

        print(f"\n[INFO] Benchmark Completed.")


if __name__ == "__main__":
    bench = CohortGeneralizationBenchmark()
    bench.run_benchmark()