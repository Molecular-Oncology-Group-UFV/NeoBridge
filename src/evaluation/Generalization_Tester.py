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

# Suppress warnings
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
# FAST HAMMING DISTANCE COMPUTATION
# ==============================================================================

def prepare_train_matrices(train_peptides):
    """Groups training peptides by length and converts them to byte arrays for fast matrix operations."""
    train_arrays_by_len = {}
    df_train_peps = pd.DataFrame({'peptide': train_peptides.unique()})
    df_train_peps['len'] = df_train_peps['peptide'].apply(len)

    for L, group in df_train_peps.groupby('len'):
        byte_arrays = [np.frombuffer(p.encode(), dtype=np.uint8) for p in group['peptide']]
        train_arrays_by_len[L] = np.vstack(byte_arrays)
    return train_arrays_by_len


def get_min_hamming_distance(test_pep, train_arrays_by_len):
    """Calculates the minimum Hamming distance against the entire training set (vectorized)."""
    L = len(test_pep)
    if L not in train_arrays_by_len:
        return 99  # 100% novel if no matching length exists

    test_arr = np.frombuffer(test_pep.encode(), dtype=np.uint8)
    train_mat = train_arrays_by_len[L]
    dists = np.sum(train_mat != test_arr, axis=1)
    return int(dists.min())


# ==============================================================================
# MODEL DEFINITIONS
# ==============================================================================

class ImprovedMLP(nn.Module):
    """Multi-Layer Perceptron with Batch Normalization and Dropout."""

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
    """Ensemble combining PyTorch MLP and XGBoost predictions."""

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


# ==============================================================================
# GENERALIZATION BENCHMARK PIPELINE
# ==============================================================================

class GeneralizationBenchmark:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.data_dir = self.project_root / 'data' / 'experiments'

        self.base_figs_dir = self.project_root / 'results' / 'figures' / 'generalization'
        self.base_reports_dir = self.project_root / 'results' / 'reports' / 'generalization'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.experiments_config = {
            'melanoma': {
                'folder': 'melanoma',
                'train': 'dataset_80_20_no_melanoma.csv',
                'test': 'dataset_80_20_only_melanoma.csv',
                'title': 'Melanoma'
            },
            'pancreas': {
                'folder': 'pancreas',
                'train': 'dataset_80_20_no_pancreas.csv',
                'test': 'dataset_only_pancreas.csv',
                'title': 'Pancreatic Cancer'
            },
            'lung': {
                'folder': 'pulmon',
                'train': 'dataset_80_20_no_lung.csv',
                'test': 'dataset_only_lung.csv',
                'title': 'Lung Cancer'
            },
            'minor_cancers': {
                'folder': 'multiples_minoritarios',
                'train': 'dataset_80_20_major_cancers.csv',
                'test': 'dataset_80_20_minor_cancers.csv',
                'title': 'Minor Cancers'
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
        """Performs rapid hyperparameter search for the MLP architecture."""
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)

        def objective(trial):
            n_layers = trial.suggest_int('n_layers', 2, 3)
            dims = [trial.suggest_int(f'd{i}', 64, 256) for i in range(n_layers)]
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

            model = ImprovedMLP(X_train.shape[1], dims, 0.3).to(self.device)
            opt = optim.Adam(model.parameters(), lr=lr)
            crit = nn.BCEWithLogitsLoss()

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
        """Trains final MLP and XGBoost models."""
        # 1. Train MLP
        model_mlp = ImprovedMLP(X_train.shape[1], mlp_cfg['dims'], 0.3).to(self.device)
        optimizer = optim.Adam(model_mlp.parameters(), lr=mlp_cfg['lr'])
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0]).to(self.device))

        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)

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

    def analyze_loco_homology_leakage(self, df_train, df_test, y_test, test_probs, cancer_title, folder_name):
        """Calculates sequence homology leakage and plots stratified ROC (Publication Quality)."""
        print(f"      [~] Auditing Sequence Homology Leakage for {cancer_title}...")

        # IMPORTANT: Ensure this matches your DataFrame's peptide sequence column
        PEPTIDE_COL = 'Peptide'

        # 1. Compute Hamming distances
        train_peptides = df_train[PEPTIDE_COL].dropna()
        train_matrices = prepare_train_matrices(train_peptides)

        df_test_eval = df_test.copy()
        df_test_eval['min_hamming'] = df_test_eval[PEPTIDE_COL].apply(
            lambda p: get_min_hamming_distance(str(p), train_matrices)
        )

        df_test_eval['is_novel'] = df_test_eval['min_hamming'] > 1

        # 2. Stratify outputs
        novel_mask = df_test_eval['is_novel'] == True
        leaked_mask = df_test_eval['is_novel'] == False

        y_novel, probs_novel = y_test[novel_mask], test_probs[novel_mask]
        y_leak, probs_leak = y_test[leaked_mask], test_probs[leaked_mask]

        n_all, n_nov, n_leak = len(y_test), novel_mask.sum(), leaked_mask.sum()

        auc_all = roc_auc_score(y_test, test_probs)
        auc_nov = roc_auc_score(y_novel, probs_novel) if len(np.unique(y_novel)) > 1 else 0
        auc_leak = roc_auc_score(y_leak, probs_leak) if n_leak > 0 and len(np.unique(y_leak)) > 1 else 0

        print(f"        -> Strictly Novel: {n_nov} ({(n_nov / n_all) * 100:.1f}%) | AUC: {auc_nov:.3f}")

        # 3. Plotting
        plt.style.use('default')
        sns.set_theme(style="ticks", context="paper", font_scale=1.2)
        fig, ax = plt.subplots(figsize=(8, 6.5))

        fpr_all, tpr_all, _ = roc_curve(y_test, test_probs)
        ax.plot(fpr_all, tpr_all, color='#555555', linestyle=':', linewidth=2.5, zorder=2,
                label=f'All Peptides\nAUC = {auc_all:.3f} (n={n_all})')

        if auc_nov > 0:
            fpr_nov, tpr_nov, _ = roc_curve(y_novel, probs_novel)
            ax.plot(fpr_nov, tpr_nov, color='#084594', linestyle='-', linewidth=3, zorder=4,
                    label=f'Strictly Novel (Hamming > 1)\nAUC = {auc_nov:.3f} (n={n_nov})')

        if auc_leak > 0:
            fpr_leak, tpr_leak, _ = roc_curve(y_leak, probs_leak)
            ax.plot(fpr_leak, tpr_leak, color='#cb181d', linestyle='--', linewidth=2, alpha=0.8, zorder=3,
                    label=f'Homologous (Hamming <= 1)\nAUC = {auc_leak:.3f} (n={n_leak})')

        ax.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title(f'{cancer_title} Hold-Out: Sequence Novelty Stratification', fontweight='bold', pad=15)
        ax.grid(True, linestyle='-', alpha=0.15, color='gray')

        legend = ax.legend(loc='lower right', fontsize=11, frameon=True, edgecolor='black', framealpha=0.9)
        legend.get_frame().set_linewidth(1.0)
        sns.despine()

        self.base_figs_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.base_figs_dir / f'Homology_Leakage_ROC_{folder_name}.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_size_matched_analysis(self, full_data_metrics, size_matched_metrics):
        """Generates out-of-sample AUPRC distribution with matched N sizes."""
        print("\n[INFO] Generating Size-Matched Sensitivity visualization...")

        df_sm = pd.DataFrame(size_matched_metrics)
        df_fd = pd.DataFrame(full_data_metrics)

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle('Size-Matched Sensitivity Analysis (Cancer Hold-Out)', y=1.05, fontweight='bold', fontsize=18)

        category_order = df_fd['Cancer_Holdout'].unique()

        sns.boxplot(x='Cancer_Holdout', y='AUPRC', data=df_sm, ax=ax, color='lightblue',
                    showfliers=False, order=category_order)
        sns.stripplot(x='Cancer_Holdout', y='AUPRC', data=df_sm, ax=ax, color='blue',
                      alpha=0.4, jitter=True, order=category_order)

        for i, cancer in enumerate(category_order):
            val = df_fd[df_fd['Cancer_Holdout'] == cancer]['AUPRC'].values[0]
            ax.plot(i, val, marker='*', color='red', markersize=18, markeredgecolor='black',
                    label='Full Data Baseline' if i == 0 else "")
            ax.hlines(val, i - 0.4, i + 0.4, colors='red', linestyles='dashed', alpha=0.6)

        ax.set_title('Out-of-Sample AUPRC: Full Data vs N-Matched', pad=15)
        ax.set_ylabel('AUPRC Score', fontweight='bold')
        ax.set_xlabel('Held-out Cancer Type', fontweight='bold', labelpad=15)

        plt.xticks(rotation=30, ha='right', fontsize=13)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best', frameon=True, edgecolor='black')

        plt.tight_layout()
        save_path = self.base_figs_dir / 'Cancer_Size_Matched_Analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_plots(self, val_y, val_probs, test_y, test_probs, output_dir, cancer_title):
        """Generates ROC and Confusion Matrix for general hold-out evaluations."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. ROC Curve Comparison
        fig, ax = plt.subplots(figsize=(8, 6.5))

        fpr_v, tpr_v, _ = roc_curve(val_y, val_probs)
        auc_v = roc_auc_score(val_y, val_probs)

        fpr_t, tpr_t, _ = roc_curve(test_y, test_probs)
        auc_t = roc_auc_score(test_y, test_probs)

        ax.plot(fpr_v, tpr_v, linestyle='--', color='#1f77b4', linewidth=2,
                label=f'Validation (Known Cancers) AUC={auc_v:.2f}')
        ax.plot(fpr_t, tpr_t, linestyle='-', color='#d62728', linewidth=3,
                label=f'Generalization ({cancer_title}) AUC={auc_t:.2f}')
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--', alpha=0.7)

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve: Generalization to {cancer_title}', pad=20)

        total_samples = len(test_y)
        pos_pct = (sum(test_y) / total_samples) * 100
        stats_text = f"Total Samples: {total_samples} | Positive: {pos_pct:.1f}%"

        ax.text(0.5, 1.15, stats_text, transform=ax.transAxes,
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#cccccc'))

        ax.legend(loc='lower right', frameon=True, edgecolor='black')
        ax.grid(True, linestyle='--', alpha=0.6)
        sns.despine()

        plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Confusion Matrix
        prec, rec, thresholds = precision_recall_curve(test_y, test_probs)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_thresh = thresholds[np.argmax(f1_scores)]
        preds = (test_probs >= best_thresh).astype(int)

        cm = confusion_matrix(test_y, preds)

        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 14})

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
        print("STARTING GENERALIZATION BENCHMARK WITH SIZE-MATCHING")
        print("=" * 60)

        summary_full_data = []
        summary_size_matched = []

        # 1. Determine N limit for Size-Matching
        min_train_size = float('inf')
        datasets_cache = {}

        for key, config in self.experiments_config.items():
            df_train, df_test = self.load_data(config)
            if df_train is not None:
                datasets_cache[key] = (df_train, df_test)
                if len(df_train) < min_train_size:
                    min_train_size = len(df_train)

        target_train_size = int(min_train_size * 0.95)
        n_iterations = 20
        print(f"[INFO] Optimal N for Size-Matching: {target_train_size} samples.")

        # 2. Main execution loop
        for key, config in self.experiments_config.items():
            if key not in datasets_cache: continue

            print(f"\n>>> Processing Hold-out Cancer: {config['title']}")
            df_train_full, df_test = datasets_cache[key]

            X_test_raw, y_test = self.prepare_features(df_test)
            X_train_raw_full, y_train_full = self.prepare_features(df_train_full)

            # --- PHASE A: FULL DATA BASELINE ---
            print("    [!] Computing Baseline Full Data...")
            scaler_full = StandardScaler()
            X_train_scaled_full = scaler_full.fit_transform(X_train_raw_full)
            X_test_scaled_full = scaler_full.transform(X_test_raw)

            skf_f = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            t_idx_f, v_idx_f = next(skf_f.split(X_train_scaled_full, y_train_full))

            mlp_cfg_full = self.fast_optimize_mlp(X_train_scaled_full[t_idx_f], y_train_full[t_idx_f],
                                                  X_train_scaled_full[v_idx_f], y_train_full[v_idx_f])

            mlp_full, xgb_full = self.train_final_models(X_train_scaled_full, y_train_full,
                                                         X_train_scaled_full[v_idx_f], y_train_full[v_idx_f],
                                                         mlp_cfg_full)

            ens_full = HybridEnsemble(mlp_full, xgb_full, self.device)
            test_probs_full, _, _ = ens_full.predict_proba(X_test_scaled_full)

            ap_full = average_precision_score(y_test, test_probs_full)
            print(f"        -> AUPRC Full Data: {ap_full:.4f}")

            # Execute Homology Leakage Audit and generate Stratified ROC plot
            self.analyze_loco_homology_leakage(df_train_full, df_test, y_test, test_probs_full, config['title'],
                                               config['folder'])

            summary_full_data.append({
                'Cancer_Holdout': config['title'],
                'AUPRC': ap_full
            })

            # --- PHASE B: SIZE-MATCHED ANALYSIS ---
            train_pos_ratio = df_train_full['immunogenicity'].mean()
            pool_pos = df_train_full[df_train_full['immunogenicity'] == 1]
            pool_neg = df_train_full[df_train_full['immunogenicity'] == 0]

            n_pos_target = int(target_train_size * train_pos_ratio)
            n_neg_target = target_train_size - n_pos_target

            print(f"    [!] Executing {n_iterations} Size-Matched iterations (N={target_train_size})...")

            for i in range(n_iterations):
                sample_pos = pool_pos.sample(n=n_pos_target, replace=False, random_state=i)
                sample_neg = pool_neg.sample(n=n_neg_target, replace=False, random_state=i + 100)
                df_matched = pd.concat([sample_pos, sample_neg]).sample(frac=1, random_state=i)

                X_match_raw, y_match = self.prepare_features(df_matched)

                scaler_sm = StandardScaler()
                X_match_scaled = scaler_sm.fit_transform(X_match_raw)
                X_test_scaled_sm = scaler_sm.transform(X_test_raw)

                skf_sm = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
                t_idx_m, v_idx_m = next(skf_sm.split(X_match_scaled, y_match))

                mlp_cfg_sm = self.fast_optimize_mlp(X_match_scaled[t_idx_m], y_match[t_idx_m],
                                                    X_match_scaled[v_idx_m], y_match[v_idx_m])

                mlp_sm, xgb_sm = self.train_final_models(X_match_scaled, y_match,
                                                         X_match_scaled[v_idx_m], y_match[v_idx_m], mlp_cfg_sm)

                ens_sm = HybridEnsemble(mlp_sm, xgb_sm, self.device)
                test_probs_sm, _, _ = ens_sm.predict_proba(X_test_scaled_sm)

                ap_out = average_precision_score(y_test, test_probs_sm)

                summary_size_matched.append({
                    'Cancer_Holdout': config['title'],
                    'Iteration': i,
                    'AUPRC': ap_out
                })

        # 3. Output Synthesis
        self.plot_size_matched_analysis(summary_full_data, summary_size_matched)

        self.base_reports_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_size_matched).to_csv(self.base_reports_dir / 'cancer_size_matched.csv', index=False)
        print("\n[INFO] Size-Matched Cancer Benchmark Completed")


if __name__ == "__main__":
    bench = GeneralizationBenchmark()
    bench.run_all()