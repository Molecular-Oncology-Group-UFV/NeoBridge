import os
import shutil
import subprocess
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold  # Reemplaza a train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, precision_recall_curve,
                             f1_score, accuracy_score, precision_score, recall_score, roc_curve)
import xgboost as xgb
import joblib
import warnings
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# CONFIGURACIÓN DE MATPLOTLIB PARA PUBLICACIÓN CIENTÍFICA
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300
})


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
            pytorch_proba = torch.sigmoid(pytorch_logits).cpu().numpy().flatten()

        if isinstance(X, torch.Tensor):
            X_numpy = X.cpu().numpy()
        else:
            X_numpy = X

        xgb_proba = self.xgb_model.predict_proba(X_numpy)[:, 1]

        return pytorch_proba, xgb_proba

    def ensemble_predict_proba(self, X, weights=(0.5, 0.5)):
        p_prob, x_prob = self.predict_proba(X)
        total = sum(weights)
        w1, w2 = weights[0] / total, weights[1] / total
        ensemble_proba = (w1 * p_prob + w2 * x_prob)
        return ensemble_proba, p_prob, x_prob


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

class HybridEnsemblePipeline:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent.parent

        # Rutas de datos diferenciadas
        self.raw_data_path = self.project_root / 'data' / 'raw' / 'NeoPepBase.csv'
        self.clustered_data_path = self.project_root / 'data' / 'raw' / 'NeoPepBase_clustered.csv'

        # NUEVAS carpetas para no sobrescribir tus resultados anteriores
        self.models_dir = self.project_root / 'models' / 'production' / 'hybrid_ensemble_clustered'
        self.reports_dir = self.project_root / 'results_mmseqs' / 'reports'
        self.figures_dir = self.project_root / 'results_mmseqs' / 'figures'

        for d in [self.models_dir, self.reports_dir, self.figures_dir]:
            d.mkdir(parents=True, exist_ok=True)

        for sub in ['optuna', 'performance', 'feature_importance', 'elispot_candidates', 'benchmarking', 'eda']:
            (self.reports_dir / sub).mkdir(exist_ok=True)
            (self.figures_dir / sub).mkdir(exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device selected: {self.device}")

    # --------------------------------------------------------------------------
    # NUEVO: CLUSTERING Y GRÁFICO PARA EL REVISOR
    # --------------------------------------------------------------------------

    def apply_mmseqs2_clustering(self, df, identity_threshold=0.8):
        print(f"\n--- INICIANDO CLUSTERING DE SECUENCIAS CON MMSEQS2 (Umbral: {identity_threshold * 100}%) ---")

        mmseqs_bin = shutil.which('mmseqs')
        if mmseqs_bin is None:
            raise FileNotFoundError("\n[ERROR FATAL] No se detecta el binario 'mmseqs' en el PATH de Python.")

        temp_dir = self.raw_data_path.parent / "mmseqs_temp"
        temp_dir.mkdir(exist_ok=True)
        fasta_path = temp_dir / "peptides.fasta"
        cluster_prefix = temp_dir / "clusterRes"

        unique_peptides = df['Peptide'].dropna().unique()
        with open(fasta_path, "w") as f:
            for i, seq in enumerate(unique_peptides):
                f.write(f">seq_{i}\n{seq}\n")

        try:
            print("  -> Ejecutando workflow easy-cluster...")
            clean_env = os.environ.copy()
            clean_env.pop("LD_LIBRARY_PATH", None)
            clean_env.pop("LD_PRELOAD", None)

            comando = [
                mmseqs_bin, 'easy-cluster', str(fasta_path), str(cluster_prefix),
                str(temp_dir), '--min-seq-id', str(identity_threshold), '-c', '0.8'
            ]

            subprocess.run(comando, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                           env=clean_env)

            tsv_path = str(cluster_prefix) + "_cluster.tsv"
            print("  -> Mapeando resultados...")
            clusters_df = pd.read_csv(tsv_path, sep='\t', header=None, names=['cluster_rep', 'member'])

            seq_dict = {f"seq_{i}": seq for i, seq in enumerate(unique_peptides)}
            clusters_df['peptide_seq'] = clusters_df['member'].map(seq_dict)
            seq_to_cluster = dict(zip(clusters_df['peptide_seq'], clusters_df['cluster_rep']))
            df['cluster_id'] = df['Peptide'].map(seq_to_cluster)

            df.to_csv(self.clustered_data_path, index=False)
            print(f"✓ Clustering finalizado. Se ha creado el nuevo archivo: {self.clustered_data_path.name}")

        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] El comando MMseqs2 falló con código {e.returncode}.")
            print(f"--- DETALLE DEL ERROR (stderr) ---\n{e.stderr}\n----------------------------------")
            raise
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

        return df

    def plot_clustering_validation(self, df):
        """Genera una figura para los revisores demostrando el clustering."""
        print("\nGenerando figura de validación del clustering (Reviewer Response)...")

        total_seqs = len(df)
        unique_clusters = df['cluster_id'].nunique()

        plt.figure(figsize=(8, 6))
        bars = plt.bar(['Total Peptides (Raw)', 'Unique Homology Clusters\n(<80% Identity)'],
                       [total_seqs, unique_clusters],
                       color=['#B0B0B0', '#084594'], edgecolor='black', linewidth=1.5, width=0.5)

        # Añadir las etiquetas con el número exacto
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + (total_seqs * 0.02),
                     f'{int(yval)}', ha='center', va='bottom', fontsize=16, fontweight='bold')

        plt.ylim(0, total_seqs * 1.15)
        plt.ylabel('Count', fontsize=16, fontweight='bold')
        plt.title('Sequence Redundancy Reduction (MMseqs2)', fontsize=18, fontweight='bold', pad=15)

        # Subtítulo explicativo explícito para el revisor
        plt.text(0.5, 0.93, 'Prospective Partitioning Applied: Sequence-Level Group Split',
                 transform=plt.gca().transAxes, ha='center', va='center', fontsize=12,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f9ff', edgecolor='#084594'))

        plt.xticks(fontsize=14, fontweight='bold')
        plt.yticks(fontsize=12)
        sns.despine()

        save_path = self.figures_dir / 'eda' / 'reviewer_clustering_validation.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Figura del revisor guardada en: {save_path}")

    def load_data(self):
        if self.clustered_data_path.exists():
            print(f"Loading pre-clustered data from {self.clustered_data_path.name}")
            df = pd.read_csv(self.clustered_data_path)
        else:
            if not self.raw_data_path.exists():
                raise FileNotFoundError(f"Data not found at {self.raw_data_path}")
            print(f"Loading raw data from {self.raw_data_path.name}")
            df = pd.read_csv(self.raw_data_path)
            df = self.apply_mmseqs2_clustering(df, identity_threshold=0.8)

        # Generar figura para el revisor ANTES de eliminar las variables
        self.plot_clustering_validation(df)

        features_to_drop = ['immunogenicity', 'mut_is_binding_pos', 'bestWTMatchType_I',
                            'DAI_MixMHC_mbp', 'mutant_rank', 'hydrophobicity_avg_diff', 'cluster_id']

        self.groups = df['cluster_id'].values
        X_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=features_to_drop, errors='ignore')
        X = X_df.values
        y = df['immunogenicity'].values
        self.feature_names = X_df.columns.tolist()

        return df, X, y

    def plot_feature_correlation(self, X, feature_names):
        print("\n" + "=" * 60)
        print("GENERATING FEATURE CORRELATION ANALYSIS")
        print("=" * 60)
        df_features = pd.DataFrame(X, columns=feature_names)
        corr_matrix = df_features.corr(method='pearson')
        fig_size = max(10, len(feature_names) * 0.4)
        plt.figure(figsize=(fig_size, fig_size * 0.8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .75}, annot=False, fmt=".2f")
        plt.title('Feature Correlation Matrix', fontsize=18, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(fontsize=11)
        plt.tight_layout()
        save_path_plot = self.figures_dir / 'eda' / 'feature_correlation_matrix.png'
        save_path_csv = self.reports_dir / 'eda' / 'feature_correlation_matrix.csv'
        plt.savefig(save_path_plot, dpi=300, bbox_inches='tight')
        plt.close()
        corr_matrix.to_csv(save_path_csv)
        print(f"✓ Correlation Matrix plot saved to: {save_path_plot}")

    # --------------------------------------------------------------------------
    # TRAINING HELPERS
    # --------------------------------------------------------------------------

    def train_pytorch_fold(self, X_train, y_train, X_val, y_val, config):
        hidden_dims = [config[f'hidden_dim_{i}'] for i in range(config['n_layers'])]
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        X_v = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
        model = ImprovedMLP(X_train.shape[1], hidden_dims, config['dropout_rate']).to(self.device)
        pos_weight_val = len(y_train) / sum(y_train) if sum(y_train) > 0 else 1.0
        pos_weight = torch.tensor([pos_weight_val]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

        best_auc = 0
        patience_counter = 0

        for epoch in range(config.get('epochs', 50)):
            model.train()
            for inputs, labels in loader:
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_probs = torch.sigmoid(model(X_v)).cpu().numpy().flatten()
            try:
                val_auc = roc_auc_score(y_val, val_probs)
            except:
                val_auc = 0.5
            if val_auc > best_auc + 0.0001:
                best_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= config.get('patience', 10):
                break

        return best_auc, model

    def run_optimization_cv(self, X, y, groups, model_type, n_trials, metric="auc_roc"):
        print(f"\nOptimizing {model_type} with 5-Fold Stratified Group CV ({n_trials} trials)...")

        def objective(trial):
            # Cambiado a Group Split
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            if model_type == 'PyTorch':
                n_layers = trial.suggest_int('n_layers', 1, 3)
                hidden_dims = [trial.suggest_int(f'hidden_dim_{i}', 32, 256, log=True) for i in range(n_layers)]
                config = {
                    'n_layers': n_layers, 'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                    'batch_size': trial.suggest_categorical('batch_size', [64, 128]),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
                    'epochs': 30, 'patience': 5
                }
                for i, h in enumerate(hidden_dims): config[f'hidden_dim_{i}'] = h
            elif model_type == 'XGBoost':
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'eval_metric': 'logloss', 'verbosity': 0, 'n_jobs': 1
                }

            # Pasar grupos al split
            for train_idx, val_idx in sgkf.split(X, y, groups):
                X_t, X_v = X[train_idx], X[val_idx]
                y_t, y_v = y[train_idx], y[val_idx]
                if model_type == 'PyTorch':
                    score, _ = self.train_pytorch_fold(X_t, y_t, X_v, y_v, config)
                    scores.append(score)
                elif model_type == 'XGBoost':
                    model = xgb.XGBClassifier(**params)
                    model.fit(X_t, y_t)
                    probs = model.predict_proba(X_v)[:, 1]
                    if metric == "auc_pr":
                        scores.append(average_precision_score(y_v, probs))
                    else:
                        scores.append(roc_auc_score(y_v, probs))
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        self.save_optimization_results(study, model_type, metric)
        return study.best_params

    def optimize_weights(self, pytorch_model, xgb_model, X_val, y_val, n_trials=20, metric="auc_roc"):
        print("\nOptimizing Ensemble Weights...")
        ensemble = HybridEnsemble(pytorch_model, xgb_model, self.device)

        def objective(trial):
            w_pytorch = trial.suggest_float('weight_pytorch', 0.1, 0.9)
            probs, _, _ = ensemble.ensemble_predict_proba(X_val, (w_pytorch, 1.0 - w_pytorch))
            if metric == "auc_pr": return average_precision_score(y_val, probs)
            return roc_auc_score(y_val, probs)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_w = study.best_params['weight_pytorch']
        return (best_w, 1.0 - best_w)

    def evaluate_performance(self, y_true, y_probs, set_name="test"):
        precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
        f2_scores = []
        for i in range(len(precision) - 1):
            p, r = precision[i], recall[i]
            f2 = (5 * p * r) / (4 * p + r) if (p + r) > 0 else 0
            f2_scores.append(f2)
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
        best_thresh_f2 = thresholds[np.argmax(f2_scores)]
        best_thresh_f1 = thresholds[np.argmax(f1_scores[:-1])]
        print(f"\n--- {set_name.upper()} THRESHOLDS ---")
        print(f"Optimal F2 Threshold: {best_thresh_f2:.4f}")
        print(f"Optimal F1 Threshold: {best_thresh_f1:.4f}")
        return best_thresh_f2, best_thresh_f1

    def plot_confusion_matrix(self, y_true, y_probs, threshold, set_name="final_test"):
        print(f"\nGenerating Publication-Quality Confusion Matrix ({set_name})...")
        y_pred = (y_probs >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 14})
        ax.set_xlabel('Predicted', fontsize=14, labelpad=10)
        ax.set_ylabel('True', fontsize=14, labelpad=10)
        ax.set_xticklabels(['0', '1'], fontsize=12)
        ax.set_yticklabels(['0', '1'], fontsize=12, rotation=0)
        plt.title(f'Confusion Matrix ({set_name})\nThresh: {threshold:.3f}', fontsize=16, pad=15)
        plt.tight_layout()
        save_path = self.figures_dir / 'performance' / f'confusion_matrix_paper.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Confusion Matrix saved to: {save_path}")

    def analyze_feature_importance(self, pytorch_model, xgb_model, X_sample, weights):
        print("\nGenerating Scientific Feature Importance Analysis...")
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_sample)
        xgb_imp = np.abs(shap_values).mean(0)
        pytorch_model.eval()
        X_t = torch.tensor(X_sample, dtype=torch.float32).to(self.device)
        X_t.requires_grad = True
        outputs = pytorch_model(X_t)
        outputs.backward(torch.ones_like(outputs))
        grads = X_t.grad.cpu().numpy()
        mlp_imp = np.abs(grads * X_sample).mean(0)
        xgb_imp /= xgb_imp.sum()
        mlp_imp /= mlp_imp.sum()
        ens_imp = weights[0] * mlp_imp + weights[1] * xgb_imp
        df_imp = pd.DataFrame({'Feature': self.feature_names, 'MLP': mlp_imp, 'XGBoost': xgb_imp, 'Ensemble': ens_imp})
        df_imp = df_imp.sort_values('Ensemble', ascending=False).head(20)
        df_imp.to_csv(self.reports_dir / 'feature_importance' / 'ensemble_importance.csv', index=False)
        df_melted = df_imp.melt(id_vars='Feature', var_name='Model', value_name='Importance')
        plt.figure(figsize=(10, 8))
        scientific_blues = {'MLP': '#9ecae1', 'XGBoost': '#4292c6', 'Ensemble': '#084594'}
        ax = sns.barplot(data=df_melted, x='Importance', y='Feature', hue='Model', palette=scientific_blues,
                         edgecolor='black', linewidth=0.5)
        ax.set_title('Feature Importance Comparison (Top 20)', fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Relative Importance (Normalized)', fontsize=14, fontweight='bold')
        ax.set_ylabel('', fontsize=14)
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        ax.grid(visible=False, axis='y')
        plt.legend(title='Model Component', title_fontsize=12, fontsize=11, loc='lower right', frameon=True,
                   framealpha=0.9, edgecolor='#cccccc')
        plt.yticks(fontsize=12)
        sns.despine(left=True, bottom=False)
        plt.tight_layout()
        save_path = self.figures_dir / 'feature_importance' / 'ensemble_feature_importance_scientific.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Gráfico científico guardado en: {save_path}")
        return np.corrcoef(mlp_imp, xgb_imp)[0, 1]

    def save_optimization_results(self, study, model_type, metric):
        df_results = study.trials_dataframe()
        df_results['model'] = model_type
        df_results['metric'] = metric
        results_path = self.reports_dir / 'optuna' / f'{model_type}_optimization_results.csv'
        df_results.to_csv(results_path, index=False)

    def plot_generalization_heatmap(self, y_val, probs_val, y_test, probs_test):
        print("\nGenerating Validation vs. Test Heatmap...")

        def get_metrics(y_true, probs):
            y_pred = (probs >= 0.5).astype(int)
            return {
                'AUC-ROC': roc_auc_score(y_true, probs),
                'AUC-PR': average_precision_score(y_true, probs),
                'F1-Score': f1_score(y_true, y_pred),
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred, zero_division=0),
                'Recall': recall_score(y_true, y_pred, zero_division=0)
            }

        val_metrics = get_metrics(y_val, probs_val)
        test_metrics = get_metrics(y_test, probs_test)
        df_metrics = pd.DataFrame([val_metrics, test_metrics], index=['Validation', 'Test']).T
        plt.figure(figsize=(6, 7))
        sns.heatmap(df_metrics, annot=True, fmt=".3f", cmap="Blues", cbar_kws={'label': 'Score Value', 'shrink': 0.8},
                    annot_kws={"size": 12, "weight": "bold"})
        plt.title('NeoBridge Performance (Clustered Data):\nValidation vs. Test Set', pad=20, fontsize=16,
                  fontweight='bold')
        plt.ylabel('Metric', fontsize=14, fontweight='bold')
        plt.xlabel('Dataset Split', fontsize=14, fontweight='bold')
        plt.yticks(rotation=0, fontsize=12)
        plt.xticks(fontsize=12)
        plt.tight_layout()
        save_path = self.figures_dir / 'performance' / 'validation_test_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Heatmap saved to: {save_path}")
        df_metrics.to_csv(self.reports_dir / 'performance' / 'validation_test_comparison.csv')

    def analyze_ranking_performance(self, df_test, y_true, y_probs):
        print("\n" + "=" * 60)
        print("RANKING & TOP-N METRICS ANALYSIS")
        print("=" * 60)
        total_positives = np.sum(y_true)
        total_samples = len(y_true)
        print(f"Total Immunogenic in Test Set: {total_positives} out of {total_samples}")
        results = df_test.copy()
        results['model_probability'] = y_probs
        results['true_label'] = y_true
        results = results.sort_values(by='model_probability', ascending=False).reset_index(drop=True)
        results['rank'] = np.arange(1, len(results) + 1)
        results['cumulative_positives'] = results['true_label'].cumsum()
        results['precision_at_k'] = results['cumulative_positives'] / results['rank']
        results['recall_at_k'] = results['cumulative_positives'] / total_positives
        auprc = average_precision_score(y_true, y_probs)
        ttif = results.iloc[19]['precision_at_k']
        recalls = {}
        for n in [20, 50, 100]:
            if n <= len(results):
                recalls[n] = results.iloc[n - 1]['recall_at_k']
            else:
                recalls[n] = results.iloc[-1]['recall_at_k']
        zoom_k = min(250, len(results))
        plot_data = results.head(zoom_k)
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        ax1 = axes[0]
        sns.lineplot(data=plot_data, x='rank', y='precision_at_k', ax=ax1, color='#1f77b4', linewidth=2.5,
                     label='Model Precision')
        ax1.scatter(20, ttif, color='red', s=100, zorder=5)
        ax1.axvline(x=20, color='red', linestyle='--', alpha=0.6, label='Rank 20 Cutoff')
        ax1.axhline(y=ttif, color='red', linestyle=':', alpha=0.6, label=f'TTF20 (Precision@20): {ttif:.3f}')
        baseline_precision = total_positives / total_samples
        ax1.axhline(y=baseline_precision, color='grey', linestyle='-.',
                    label=f'Baseline Prevalence ({baseline_precision:.3f})')
        ax1.set_title('Precision at Rank K (Focus on TTF20)', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Rank K (Number of top candidates selected)', fontsize=14)
        ax1.set_ylabel('Cumulative Precision (Fraction of Positives)', fontsize=14)
        ax1.set_xlim(1, zoom_k)
        ax1.set_ylim(0, 1.05)
        ax1.legend(loc='upper right', frameon=True, fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.tick_params(labelsize=12)
        ax2 = axes[1]
        sns.lineplot(data=plot_data, x='rank', y='recall_at_k', ax=ax2, color='#2ca02c', linewidth=2.5,
                     label='Model Recall Accumulation')
        colors_n = {20: 'red', 50: 'orange', 100: 'purple'}
        for n, color in colors_n.items():
            if n <= zoom_k:
                rec_val = recalls[n]
                ax2.scatter(n, rec_val, color=color, s=80, zorder=5)
                ax2.axvline(x=n, color=color, linestyle='--', alpha=0.5)
                if rec_val > 0: ax2.plot([1, n], [rec_val, rec_val], color=color, linestyle=':', alpha=0.5)
                ax2.text(n + 2, rec_val - (0.02 if rec_val > 0.1 else -0.01), f'Recall@{n}: {rec_val:.3f}', color=color,
                         fontweight='bold', fontsize=10)
        random_recall_at_zoom = zoom_k / total_samples
        ax2.plot([1, zoom_k], [1 / total_positives if total_positives > 0 else 0, random_recall_at_zoom], color='grey',
                 linestyle='-.', label='Random Guessing')
        ax2.set_title('Recall at Rank K (Cumulative Positives Found)', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Rank K', fontsize=14)
        ax2.set_ylabel('Cumulative Recall', fontsize=14)
        ax2.set_xlim(1, zoom_k)
        ax2.set_ylim(0, max(plot_data['recall_at_k'].max() * 1.1, 0.1))
        ax2.legend(loc='lower right', frameon=True, fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.tick_params(labelsize=12)
        plt.suptitle(f'Ranking Performance Analysis\n(AUPRC: {auprc:.4f} | TTF20: {ttif:.3f})', fontsize=18, y=1.02)
        plt.tight_layout()
        plot_path = self.figures_dir / 'performance' / 'ranking_metrics_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Ranking visualization saved to: {plot_path}")
        print("\n>>> Exporting Top 100 Candidates for ELISpot...")
        top_100_candidates = results.head(100).drop(columns=['cumulative_positives', 'precision_at_k', 'recall_at_k'])
        filename = f'top_100_elispot_candidates.csv'
        save_path = self.reports_dir / 'elispot_candidates' / filename
        top_100_candidates.to_csv(save_path, index=False)
        print(f"✓ Saved: {save_path}")

    def plot_ensemble_contribution(self, weights):
        print("\nGenerating Ensemble Contribution Figure...")
        w_nn, w_xgb = weights
        plt.style.use('seaborn-v0_8-white')
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        ax.set_position([0.1, 0.35, 0.8, 0.3])
        colors = ['#305F88', '#D95319']
        p1 = ax.barh(0, w_nn, height=0.6, color=colors[0], edgecolor='black', linewidth=1,
                     label='Improved MLP (PyTorch)')
        p2 = ax.barh(0, w_xgb, left=w_nn, height=0.6, color=colors[1], edgecolor='black', linewidth=1, label='XGBoost')
        if w_nn > 0.05: ax.text(w_nn / 2, 0, f"{w_nn * 100:.1f}%", ha='center', va='center', color='white',
                                fontweight='bold', fontsize=14)
        if w_xgb > 0.05: ax.text(w_nn + w_xgb / 2, 0, f"{w_xgb * 100:.1f}%", ha='center', va='center', color='white',
                                 fontweight='bold', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0', '25%', '50%', '75%', '100%'], fontsize=12)
        ax.set_xlabel('Model Contribution Weight ($w$)', fontsize=14, labelpad=10, fontweight='bold')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False, fontsize=12, handletextpad=0.5)
        fig.text(0.5, 0.92, 'Hybrid Ensemble Weight Distribution', ha='center', va='center', fontsize=18,
                 fontweight='bold', color='#333333')
        formula_latex = (
                    r"$P(y=1|x) = \mathbf{" + f"{w_nn:.3f}" + r"} \cdot \sigma(f_{MLP}(x)) + " r"\mathbf{" + f"{w_xgb:.3f}" + r"} \cdot P_{XGB}(x)$")
        fig.text(0.5, 0.15, formula_latex, ha='center', va='center', fontsize=14, color='#222222',
                 bbox=dict(facecolor='#f8f9fa', edgecolor='#cccccc', boxstyle='round,pad=0.6', linewidth=0.5))
        save_path = self.figures_dir / 'performance' / 'ensemble_contribution_formula.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Ensemble contribution figure saved to: {save_path}")

    def generate_immunemirror_benchmark(self, y_true, y_probs):
        print("\n" + "=" * 60)
        print("GENERATING SCIENTIFIC BENCHMARK FIGURES (Vs. ImmuneMirror)")
        print("=" * 60)
        my_auc = roc_auc_score(y_true, y_probs)
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        target_fpr = 1 - 0.70
        idx = np.argmin(np.abs(fpr - target_fpr))
        my_sens_at_spec70 = tpr[idx]
        my_spec_at_spec70 = 1 - fpr[idx]
        print(f"NeoBridge AUC: {my_auc:.3f}")
        print(f"At Specificity {my_spec_at_spec70:.3f}: Sensitivity = {my_sens_at_spec70:.3f}")
        im_metrics = {'AUC': 0.868, 'Sensitivity': 0.851, 'Specificity': 0.700}
        color_im = '#B0B0B0'
        color_my = '#084594'
        plt.figure(figsize=(6, 6))
        models = ['ImmuneMirror (RF)', 'NeoBridge']
        aucs = [im_metrics['AUC'], my_auc]
        bars = plt.bar(models, aucs, color=[color_im, color_my], edgecolor='black', linewidth=1, width=0.6)
        plt.axhline(im_metrics['AUC'], color='gray', linestyle=':', linewidth=1.5, zorder=0, alpha=0.7)
        plt.bar_label(bars, fmt='%.3f', padding=3, fontsize=14, fontweight='bold')
        plt.ylim(0, 1.15)
        plt.ylabel('AUC-ROC Score', fontsize=16, fontweight='bold')
        plt.title('Classification Power (Sequence Clustered)', fontsize=18, fontweight='bold', pad=15)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        sns.despine()
        save_path_auc = self.figures_dir / 'benchmarking' / 'benchmark_immunemirror_AUC.png'
        plt.savefig(save_path_auc, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Figure 1 saved: {save_path_auc}")
        plt.figure(figsize=(7, 6))
        labels = ['Sensitivity (Recall)', 'Specificity']
        x = np.arange(len(labels))
        width = 0.35
        im_values = [im_metrics['Sensitivity'], im_metrics['Specificity']]
        my_values = [my_sens_at_spec70, my_spec_at_spec70]
        rects1 = plt.bar(x - width / 2, im_values, width, label='ImmuneMirror', color=color_im, edgecolor='black',
                         linewidth=1)
        rects2 = plt.bar(x + width / 2, my_values, width, label='NeoBridge', color=color_my, edgecolor='black',
                         linewidth=1)
        plt.ylabel('Score', fontsize=16, fontweight='bold')
        plt.title('Performance at Fixed Specificity (70%)', fontsize=18, fontweight='bold', pad=15)
        plt.xticks(x, labels, fontsize=14, fontweight='normal')
        plt.yticks(fontsize=14)
        plt.ylim(0, 1.15)
        plt.legend(frameon=True, fancybox=False, edgecolor='black', loc='upper right', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        sns.despine()
        plt.bar_label(rects1, fmt='%.3f', padding=3, fontsize=12)
        plt.bar_label(rects2, fmt='%.3f', padding=3, fontsize=12, fontweight='bold')
        save_path_op = self.figures_dir / 'benchmarking' / 'benchmark_immunemirror_OperatingPoint.png'
        plt.savefig(save_path_op, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Figure 2 saved: {save_path_op}")
        fig, ax = plt.subplots(figsize=(8, 6.5))
        ax.plot(fpr, tpr, color=color_my, linewidth=3, label=f'NeoBridge (AUC = {my_auc:.3f})')
        im_fpr = 1 - im_metrics['Specificity']
        im_tpr = im_metrics['Sensitivity']
        ax.scatter([im_fpr], [im_tpr], color='#d62728', s=250, zorder=5, edgecolors='black', marker='*',
                   label=f"ImmuneMirror Op. Point\n(FPR: {im_fpr:.3f}, TPR: {im_tpr:.3f})")
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.7)
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=16, fontweight='bold')
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=16, fontweight='bold')
        ax.set_title('ROC Curve Comparison vs. ImmuneMirror', fontsize=18, fontweight='bold', pad=20)
        total_samples = 11188
        pos_pct = 23.0
        stats_text = f"Total Samples: {total_samples} | Positive: {pos_pct:.1f}%"
        ax.text(0.5, 1.15, stats_text, transform=ax.transAxes, ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#cccccc'))
        ax.legend(loc='lower right', frameon=True, edgecolor='black', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        sns.despine()
        save_path_roc = self.figures_dir / 'benchmarking' / 'benchmark_immunemirror_ROC_Curve.png'
        plt.savefig(save_path_roc, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Figure 3 (ROC Curve) saved: {save_path_roc}")

    # --------------------------------------------------------------------------
    # MAIN EXECUTION METHOD
    # --------------------------------------------------------------------------

    def run(self, n_trials_pytorch, n_trials_xgb, n_trials_ensemble, metric, use_cv):
        df, X, y = self.load_data()
        self.plot_feature_correlation(X, self.feature_names)

        # REEMPLAZO 1: Stratified Group Split para Train/Test
        print("\nRealizando partición Prospective (Sequence-Level Group Split: Dev / Test)...")
        sgkf_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        dev_idx, test_idx = next(sgkf_test.split(X, y, groups=self.groups))

        X_dev, X_test = X[dev_idx], X[test_idx]
        y_dev, y_test = y[dev_idx], y[test_idx]
        df_dev, df_test = df.iloc[dev_idx], df.iloc[test_idx]
        groups_dev = self.groups[dev_idx]

        print("Scaling features based on Development set statistics...")
        self.scaler = StandardScaler()
        X_dev = self.scaler.fit_transform(X_dev)
        X_test = self.scaler.transform(X_test)
        print(f"\nData Splits:")
        print(f"  Development (Opt): {X_dev.shape[0]} samples")
        print(f"  Test (Final Eval): {X_test.shape[0]} samples")

        best_py_params = self.run_optimization_cv(X_dev, y_dev, groups_dev, 'PyTorch', n_trials_pytorch, metric=metric)
        best_xgb_params = self.run_optimization_cv(X_dev, y_dev, groups_dev, 'XGBoost', n_trials_xgb, metric=metric)

        # REEMPLAZO 2: Stratified Group Split para Train/Validation
        print("\nRealizando partición Prospective (Train / Validation)...")
        sgkf_val = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
        train_idx_f, val_idx_f = next(sgkf_val.split(X_dev, y_dev, groups=groups_dev))

        X_train_final, X_val_final = X_dev[train_idx_f], X_dev[val_idx_f]
        y_train_final, y_val_final = y_dev[train_idx_f], y_dev[val_idx_f]

        print("\nTraining final models...")
        best_py_params['epochs'] = 100
        best_py_params['patience'] = 15
        _, pytorch_model = self.train_pytorch_fold(X_train_final, y_train_final, X_val_final, y_val_final,
                                                   best_py_params)

        final_xgb_params = best_xgb_params.copy()
        if 'n_estimators' in final_xgb_params: final_xgb_params.pop('n_estimators')
        xgb_model = xgb.XGBClassifier(**final_xgb_params, n_estimators=500)
        xgb_model.fit(X_train_final, y_train_final, eval_set=[(X_val_final, y_val_final)], verbose=False)

        best_weights = self.optimize_weights(pytorch_model, xgb_model, X_val_final, y_val_final, n_trials_ensemble,
                                             metric)
        self.plot_ensemble_contribution(best_weights)

        ensemble = HybridEnsemble(pytorch_model, xgb_model, self.device)
        probs_val, _, _ = ensemble.ensemble_predict_proba(X_val_final, weights=best_weights)
        probs_test, probs_mlp_test, probs_xgb_test = ensemble.ensemble_predict_proba(X_test, weights=best_weights)

        print("\nExportando predicciones para análisis estadístico (DeLong)...")
        df_preds = pd.DataFrame({
            'y_true': y_test,
            'prob_ensemble': probs_test,
            'prob_mlp': probs_mlp_test,
            'prob_xgb': probs_xgb_test
        })
        delong_path = self.reports_dir / 'performance' / 'test_predictions_for_delong.csv'
        df_preds.to_csv(delong_path, index=False)
        print(f"✓ Archivo guardado en: {delong_path}")

        self.plot_generalization_heatmap(y_val_final, probs_val, y_test, probs_test)
        final_auc = roc_auc_score(y_test, probs_test)
        final_pr = average_precision_score(y_test, probs_test)
        print(f"\nFinal AUC-ROC: {final_auc:.4f} | AUC-PR: {final_pr:.4f}")

        best_thresh_f2, best_thresh_f1 = self.evaluate_performance(y_test, probs_test, set_name="final_test")
        self.plot_confusion_matrix(y_test, probs_test, threshold=best_thresh_f1, set_name="final_test")
        self.analyze_feature_importance(pytorch_model, xgb_model, X_test[:200], best_weights)
        self.analyze_ranking_performance(df_test, y_test, probs_test)
        self.generate_immunemirror_benchmark(y_test, probs_test)

        print(f"\nSaving models to {self.models_dir}...")
        torch.save(pytorch_model.state_dict(), self.models_dir / 'pytorch_final.pth')
        xgb_model.save_model(self.models_dir / 'xgboost_final.json')
        joblib.dump(self.scaler, self.models_dir / 'scaler.pkl')
        print("Pipeline completed successfully.")


def main():
    print("NeoBridge ENSEMBLE PIPELINE (PROSPECTIVE SEQUENCE-SPLITTING MODEL)")
    print("=" * 60)

    metric_in = input("\nSelect metric (1: AUC-ROC, 2: AUC-PR) [Default: 2]: ") or "2"
    metric = "auc_pr" if metric_in == "2" else "auc_roc"

    cv_in = input("Use Cross-Validation? (y/n) [Default: y]: ") or "y"
    use_cv = cv_in.lower() == 'y'

    print("\n--- OPTUNA TRIALS CONFIGURATION ---")
    try:
        n_py = int(input("PyTorch trials [Default: 30]: ") or "30")
    except ValueError:
        n_py = 30
    try:
        n_xgb = int(input("XGBoost trials [Default: 30]: ") or "30")
    except ValueError:
        n_xgb = 30
    try:
        n_ens = int(input("Ensemble trials [Default: 20]: ") or "20")
    except ValueError:
        n_ens = 20

    print(f"\nConfiguration: Metric={metric}, PyTorch={n_py}, XGB={n_xgb}, Ensemble={n_ens}")
    print("Starting pipeline...\n")

    pipeline = HybridEnsemblePipeline()
    pipeline.run(n_trials_pytorch=n_py, n_trials_xgb=n_xgb, n_trials_ensemble=n_ens, metric=metric, use_cv=use_cv)


if __name__ == "__main__":
    main()