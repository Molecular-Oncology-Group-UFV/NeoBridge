import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, classification_report, precision_recall_curve,
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
    'font.size': 12,               # Tamaño base de fuente
    'axes.titlesize': 16,           # Título de los ejes
    'axes.labelsize': 14,           # Etiquetas de los ejes
    'xtick.labelsize': 11,          # Números del eje X
    'ytick.labelsize': 11,          # Números del eje Y
    'legend.fontsize': 11,           # Leyenda
    'figure.titlesize': 18,          # Título de la figura
    'figure.dpi': 300,               # Resolución por defecto
    'savefig.dpi': 300               # Resolución al guardar
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
        # Path Configuration
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.data_path = self.project_root / 'data' / 'processed' / 'dataset_80_20_normalized_tcr_with_cohort_and_rank.csv'

        # Output Directories
        self.models_dir = self.project_root / 'models' / 'production' / 'hybrid_ensemble'
        self.reports_dir = self.project_root / 'results' / 'reports'
        self.figures_dir = self.project_root / 'results' / 'figures'

        # Create directory structure
        for d in [self.models_dir, self.reports_dir, self.figures_dir]:
            d.mkdir(parents=True, exist_ok=True)

        for sub in ['optuna', 'performance', 'feature_importance', 'elispot_candidates', 'benchmarking']:
            (self.reports_dir / sub).mkdir(exist_ok=True)
            (self.figures_dir / sub).mkdir(exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device selected: {self.device}")

    def load_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found at {self.data_path}")

        print(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)

        # We keep the raw DataFrame to recover metadata later (peptide sequences, etc.)
        features_to_drop = ['immunogenicity', 'mut_is_binding_pos', 'bestWTMatchType_I',
                            'DAI_MixMHC_mbp', 'mutant_rank']

        # Select numeric features for training
        X_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=features_to_drop, errors='ignore')

        X = X_df.values
        y = df['immunogenicity'].values

        self.feature_names = X_df.columns.tolist()

        return df, X, y

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

    def run_optimization_cv(self, X, y, model_type, n_trials, metric="auc_roc"):
        print(f"\nOptimizing {model_type} with 5-Fold CV ({n_trials} trials)...")

        def objective(trial):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []

            if model_type == 'PyTorch':
                n_layers = trial.suggest_int('n_layers', 1, 3)
                hidden_dims = [trial.suggest_int(f'hidden_dim_{i}', 32, 256, log=True) for i in range(n_layers)]
                config = {
                    'n_layers': n_layers,
                    'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
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
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'eval_metric': 'logloss', 'verbosity': 0, 'n_jobs': 1
                }

            for train_idx, val_idx in skf.split(X, y):
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
        # F2/F1 Score Optimization
        precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
        f2_scores = []
        for i in range(len(precision) - 1):
            p, r = precision[i], recall[i]
            f2 = (5 * p * r) / (4 * p + r) if (p + r) > 0 else 0
            f2_scores.append(f2)

        f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)

        # Best Thresholds
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

        # Exactamente tu estilo original (sin bordes), pero con los números internos más grandes (size 14)
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                         annot_kws={"size": 14})

        # Títulos de los ejes como los tenías, con fuente 14
        ax.set_xlabel('Predicted', fontsize=14, labelpad=10)
        ax.set_ylabel('True', fontsize=14, labelpad=10)

        # Etiquetas limpias (0 y 1) con fuente 12.
        # Ponemos rotation=0 en el eje Y para que los números se lean rectos (en tu original salían tumbados)
        ax.set_xticklabels(['0', '1'], fontsize=12)
        ax.set_yticklabels(['0', '1'], fontsize=12, rotation=0)

        # Título idéntico al tuyo original pero más grande
        plt.title(f'Confusion Matrix ({set_name})\nThresh: {threshold:.3f}', fontsize=16, pad=15)

        plt.tight_layout()
        save_path = self.figures_dir / 'performance' / f'confusion_matrix_paper.png'

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Confusion Matrix saved to: {save_path}")


    def analyze_feature_importance(self, pytorch_model, xgb_model, X_sample, weights):
        print("\nGenerating Scientific Feature Importance Analysis...")

        # 1. Calcular Importancia XGBoost (SHAP)
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_sample)
        xgb_imp = np.abs(shap_values).mean(0)

        # 2. Calcular Importancia MLP (Input * Gradient)
        pytorch_model.eval()
        X_t = torch.tensor(X_sample, dtype=torch.float32).to(self.device)
        X_t.requires_grad = True
        outputs = pytorch_model(X_t)
        outputs.backward(torch.ones_like(outputs))

        # Calculamos la importancia absoluta
        grads = X_t.grad.cpu().numpy()
        mlp_imp = np.abs(grads * X_sample).mean(0)  # Input * Gradient method

        # 3. Normalizar para que sumen 1 (Importancia Relativa)
        xgb_imp /= xgb_imp.sum()
        mlp_imp /= mlp_imp.sum()

        # 4. Calcular Importancia del Ensemble ponderada
        ens_imp = weights[0] * mlp_imp + weights[1] * xgb_imp

        # 5. Crear DataFrame y procesar para visualización
        df_imp = pd.DataFrame({
            'Feature': self.feature_names,
            'MLP': mlp_imp,
            'XGBoost': xgb_imp,
            'Ensemble': ens_imp
        })

        # Ordenar por Ensemble y coger el Top 20 para no saturar la gráfica
        df_imp = df_imp.sort_values('Ensemble', ascending=False).head(20)

        # Guardar CSV crudo
        df_imp.to_csv(self.reports_dir / 'feature_importance' / 'ensemble_importance.csv', index=False)

        # Derretir (Melt) el DataFrame para poder usar Seaborn hue
        df_melted = df_imp.melt(id_vars='Feature', var_name='Model', value_name='Importance')

        # --- PLOTTING MEJORADO PARA PAPER ---
        plt.figure(figsize=(10, 8))

        # Definir paleta de azules profesionales
        scientific_blues = {
            'MLP': '#9ecae1',  # Azul cielo suave
            'XGBoost': '#4292c6',  # Azul corporativo medio
            'Ensemble': '#084594'  # Azul marino intenso
        }

        # Crear el gráfico de barras horizontal
        ax = sns.barplot(
            data=df_melted,
            x='Importance',
            y='Feature',
            hue='Model',
            palette=scientific_blues,
            edgecolor='black',
            linewidth=0.5
        )

        # Estilo "Paper" con tamaños de fuente actualizados
        ax.set_title('Feature Importance Comparison (Top 20)', fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Relative Importance (Normalized)', fontsize=14, fontweight='bold')
        ax.set_ylabel('', fontsize=14)

        # Rejilla vertical solamente
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        ax.grid(visible=False, axis='y')

        # Leyenda dentro del gráfico
        plt.legend(title='Model Component', title_fontsize=12, fontsize=11,
                   loc='lower right', frameon=True, framealpha=0.9, edgecolor='#cccccc')

        plt.yticks(fontsize=12)
        sns.despine(left=True, bottom=False)

        plt.tight_layout()

        # Guardar
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

    # --------------------------------------------------------------------------
    # HEATMAP GENERATION (TEST VS VALIDATION)
    # --------------------------------------------------------------------------

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

        # CAMBIO 1: Figura un poco más alta y ancha para que las celdas respiren
        plt.figure(figsize=(6, 7))

        # CAMBIO 2: Tamaño de fuente a 12 (antes 14)
        sns.heatmap(df_metrics, annot=True, fmt=".3f", cmap="Blues",
                    cbar_kws={'label': 'Score Value', 'shrink': 0.8},
                    annot_kws={"size": 12, "weight": "bold"})

        plt.title('General Model Performance:\nValidation vs. Test Set', pad=20, fontsize=16, fontweight='bold')
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

    # --------------------------------------------------------------------------
    # RANKING & TOP-N METRICS ANALYSIS
    # --------------------------------------------------------------------------

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

        print("Generating ranking metrics visualization plot...")
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
                if rec_val > 0:
                    ax2.plot([1, n], [rec_val, rec_val], color=color, linestyle=':', alpha=0.5)
                ax2.text(n + 2, rec_val - (0.02 if rec_val > 0.1 else -0.01), f'Recall@{n}: {rec_val:.3f}',
                         color=color, fontweight='bold', fontsize=10)
        random_recall_at_zoom = zoom_k / total_samples
        ax2.plot([1, zoom_k], [1 / total_positives if total_positives > 0 else 0, random_recall_at_zoom],
                 color='grey', linestyle='-.', label='Random Guessing')
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
        if w_nn > 0.05:
            ax.text(w_nn / 2, 0, f"{w_nn * 100:.1f}%", ha='center', va='center', color='white', fontweight='bold',
                    fontsize=14)
        if w_xgb > 0.05:
            ax.text(w_nn + w_xgb / 2, 0, f"{w_xgb * 100:.1f}%", ha='center', va='center', color='white',
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

    # ==========================================================================
    # NUEVA FUNCIÓN: Figura 8 (Benchmarking vs ImmuneMirror)
    # ==========================================================================
    def generate_immunemirror_benchmark(self, y_true, y_probs):
        """
        Genera TRES figuras independientes para el paper:
        1. Comparativa de AUC.
        2. Comparativa de Punto de Operación (Sensibilidad a Especificidad fija).
        3. Curva ROC con el punto de referencia de ImmuneMirror.
        """
        print("\n" + "=" * 60)
        print("GENERATING SCIENTIFIC BENCHMARK FIGURES (Vs. ImmuneMirror)")
        print("=" * 60)

        # --- CÁLCULO DE MÉTRICAS ---
        my_auc = roc_auc_score(y_true, y_probs)

        # Curva ROC y buscar el punto donde Especificidad es ~0.70
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        target_fpr = 1 - 0.70
        idx = np.argmin(np.abs(fpr - target_fpr))

        my_sens_at_spec70 = tpr[idx]
        my_spec_at_spec70 = 1 - fpr[idx]

        print(f"Hybrid Ensemble AUC: {my_auc:.3f}")
        print(f"At Specificity {my_spec_at_spec70:.3f}: Sensitivity = {my_sens_at_spec70:.3f}")

        # Datos de ImmuneMirror
        im_metrics = {'AUC': 0.868, 'Sensitivity': 0.851, 'Specificity': 0.700}

        # Colores
        color_im = '#B0B0B0'  # Gris suave
        color_my = '#084594'  # Azul Ensemble

        # ======================================================================
        # FIGURA 1: AUC COMPARISON (Archivo separado)
        # ======================================================================
        plt.figure(figsize=(6, 6))

        models = ['ImmuneMirror (RF)', 'XGBoost + MLP (Ours)']
        aucs = [im_metrics['AUC'], my_auc]

        bars = plt.bar(models, aucs, color=[color_im, color_my],
                       edgecolor='black', linewidth=1, width=0.6)

        plt.axhline(im_metrics['AUC'], color='gray', linestyle=':', linewidth=1.5, zorder=0, alpha=0.7)
        plt.bar_label(bars, fmt='%.3f', padding=3, fontsize=14, fontweight='bold')

        plt.ylim(0, 1.15)
        plt.ylabel('AUC-ROC Score', fontsize=16, fontweight='bold')
        plt.title('Classification Power (AUC-ROC)', fontsize=18, fontweight='bold', pad=15)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        sns.despine()

        save_path_auc = self.figures_dir / 'benchmarking' / 'benchmark_immunemirror_AUC.png'
        plt.savefig(save_path_auc, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Figure 1 saved: {save_path_auc}")

        # ======================================================================
        # FIGURA 2: OPERATING POINT COMPARISON (Archivo separado)
        # ======================================================================
        plt.figure(figsize=(7, 6))

        labels = ['Sensitivity (Recall)', 'Specificity']
        x = np.arange(len(labels))
        width = 0.35

        im_values = [im_metrics['Sensitivity'], im_metrics['Specificity']]
        my_values = [my_sens_at_spec70, my_spec_at_spec70]

        rects1 = plt.bar(x - width / 2, im_values, width, label='ImmuneMirror',
                         color=color_im, edgecolor='black', linewidth=1)
        rects2 = plt.bar(x + width / 2, my_values, width, label='XGBoost + MLP',
                         color=color_my, edgecolor='black', linewidth=1)

        plt.ylabel('Score', fontsize=16, fontweight='bold')
        plt.title('Performance at Fixed Specificity (70%)', fontsize=18, fontweight='bold', pad=15)

        # --- AQUÍ QUITAMOS LA NEGRITA DEL EJE X ---
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

        # ======================================================================
        # FIGURA 3: ROC CURVE COMPARISON (NUEVA)
        # ======================================================================
        fig, ax = plt.subplots(figsize=(8, 6.5))

        # Dibujar la curva ROC de tu modelo
        ax.plot(fpr, tpr, color=color_my, linewidth=3,
                label=f'XGBoost + MLP (AUC = {my_auc:.3f})')

        # Dibujar el punto operativo de ImmuneMirror
        im_fpr = 1 - im_metrics['Specificity']  # FPR = 1 - Specificity
        im_tpr = im_metrics['Sensitivity']  # TPR = Sensitivity

        # Estrella un poco más grande (s=250) para que destaque bien
        ax.scatter([im_fpr], [im_tpr], color='#d62728', s=250, zorder=5, edgecolors='black', marker='*',
                   label=f"ImmuneMirror Op. Point\n(FPR: {im_fpr:.3f}, TPR: {im_tpr:.3f})")

        # Línea base aleatoria
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.7)

        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=16, fontweight='bold')
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=16, fontweight='bold')
        ax.set_title('ROC Curve Comparison vs. ImmuneMirror', fontsize=18, fontweight='bold', pad=20)

        # --- CAJA DE ESTADÍSTICAS HARDCODEADA CORRECTAMENTE ---
        total_samples = 11188
        pos_pct = 23.0
        stats_text = f"Total Samples: {total_samples} | Positive: {pos_pct:.1f}%"

        # Posicionamiento relativo (1.15) para que no pise el título
        ax.text(0.5, 1.15, stats_text, transform=ax.transAxes,
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#cccccc'))

        ax.legend(loc='lower right', frameon=True, edgecolor='black', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        sns.despine()

        save_path_roc = self.figures_dir / 'benchmarking' / 'benchmark_immunemirror_ROC_Curve.png'
        # Usamos bbox_inches='tight' para que el guardado respete la caja flotante superior
        plt.savefig(save_path_roc, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Figure 3 (ROC Curve) saved: {save_path_roc}")

    # --------------------------------------------------------------------------
    # MAIN EXECUTION METHOD
    # --------------------------------------------------------------------------

    def run(self, n_trials_pytorch, n_trials_xgb, n_trials_ensemble, metric, use_cv):
        # 1. Load Data
        df, X, y = self.load_data()

        # 2. Strict Hold-out Test Set (20%)
        # Pass full DF to split to keep metadata
        X_dev, X_test, y_dev, y_test, df_dev, df_test = train_test_split(
            X, y, df, test_size=0.20, random_state=42, stratify=y
        )

        # --- CORRECCIÓN DE FUGA DE DATOS (StandardScaler) ---
        print("Scaling features based on Development set statistics...")
        self.scaler = StandardScaler()
        # 1. Ajustamos (fit) SOLO con los datos de desarrollo
        X_dev = self.scaler.fit_transform(X_dev)
        # 2. Transformamos el test con las medias/std del desarrollo (sin mirar el test)
        X_test = self.scaler.transform(X_test)
        # ----------------------------------------------------

        print(f"\nData Splits:")
        print(f"  Development (Opt): {X_dev.shape[0]} samples")
        print(f"  Test (Final Eval): {X_test.shape[0]} samples")

        # 3. Hyperparameter Optimization
        best_py_params = self.run_optimization_cv(X_dev, y_dev, 'PyTorch', n_trials_pytorch, metric=metric)
        best_xgb_params = self.run_optimization_cv(X_dev, y_dev, 'XGBoost', n_trials_xgb, metric=metric)

        # 4. Final Model Training (Internal Split for Ensemble Weights)
        X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
            X_dev, y_dev, test_size=0.1, stratify=y_dev, random_state=42
        )

        print("\nTraining final models...")
        best_py_params['epochs'] = 100;
        best_py_params['patience'] = 15
        _, pytorch_model = self.train_pytorch_fold(X_train_final, y_train_final, X_val_final, y_val_final,
                                                   best_py_params)

        final_xgb_params = best_xgb_params.copy()
        if 'n_estimators' in final_xgb_params: final_xgb_params.pop('n_estimators')
        xgb_model = xgb.XGBClassifier(**final_xgb_params, n_estimators=500)
        xgb_model.fit(X_train_final, y_train_final, eval_set=[(X_val_final, y_val_final)], verbose=False)

        # 5. Optimize Weights
        best_weights = self.optimize_weights(pytorch_model, xgb_model, X_val_final, y_val_final, n_trials_ensemble,
                                             metric)

        self.plot_ensemble_contribution(best_weights)

        # 6. Final Evaluation
        ensemble = HybridEnsemble(pytorch_model, xgb_model, self.device)

        # --- GENERATE PREDICTIONS FOR VALIDATION AND TEST FOR HEATMAP ---
        probs_val, _, _ = ensemble.ensemble_predict_proba(X_val_final, weights=best_weights)
        probs_test, _, _ = ensemble.ensemble_predict_proba(X_test, weights=best_weights)

        self.plot_generalization_heatmap(y_val_final, probs_val, y_test, probs_test)

        final_auc = roc_auc_score(y_test, probs_test)
        final_pr = average_precision_score(y_test, probs_test)
        print(f"\nFinal AUC-ROC: {final_auc:.4f} | AUC-PR: {final_pr:.4f}")

        best_thresh_f2, best_thresh_f1 = self.evaluate_performance(y_test, probs_test, set_name="final_test")

        # Generamos la matriz de confusión utilizando el threshold que optimiza F1
        self.plot_confusion_matrix(y_test, probs_test, threshold=best_thresh_f1, set_name="final_test")

        self.analyze_feature_importance(pytorch_model, xgb_model, X_test[:200], best_weights)

        # 7. Analyze Ranking & Export Top 100
        self.analyze_ranking_performance(df_test, y_test, probs_test)

        # 8. BENCHMARKING FIGURE (NEW)
        self.generate_immunemirror_benchmark(y_test, probs_test)

        # 9. Save Models
        print(f"\nSaving models to {self.models_dir}...")
        torch.save(pytorch_model.state_dict(), self.models_dir / 'pytorch_final.pth')
        xgb_model.save_model(self.models_dir / 'xgboost_final.json')
        joblib.dump(self.scaler, self.models_dir / 'scaler.pkl')
        print("Pipeline completed successfully.")


def main():
    print("HYBRID ENSEMBLE PIPELINE (GENERAL MODEL - PRODUCTION)")
    print("=" * 60)

    # 1. Configurar Métrica
    metric_in = input("\nSelect metric (1: AUC-ROC, 2: AUC-PR) [Default: 2]: ") or "2"
    metric = "auc_pr" if metric_in == "2" else "auc_roc"

    # 2. Configurar CV
    cv_in = input("Use Cross-Validation? (y/n) [Default: y]: ") or "y"
    use_cv = cv_in.lower() == 'y'

    # 3. Configurar Trials (Optuna)
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