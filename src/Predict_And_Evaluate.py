import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

from matplotlib import rcParams

warnings.filterwarnings('ignore')

# ==============================================================================
# MATPLOTLIB CONFIGURATION FOR SCIENTIFIC PUBLICATION
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


# ==============================================================================
# 1. ARCHITECTURES
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
        final_prob = (weights[0] / total * pt_prob + weights[1] / total * xgb_prob)
        return final_prob


# ==============================================================================
# 2. DISCOVERY PIPELINE
# ==============================================================================

class DiscoveryPipeline:
    def __init__(self, model_dir, reports_dir, figures_dir):
        self.model_dir = Path(model_dir)
        self.reports_dir = Path(reports_dir)
        self.figures_dir = Path(figures_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Inference device: {self.device}")

    def load_artifacts(self, input_dim):
        print("[INFO] Loading pre-trained models and scaler...")
        self.scaler = joblib.load(self.model_dir / 'scaler.pkl')

        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(self.model_dir / 'xgboost_final.json')

        # 1. Cargamos PRIMERO los pesos (los "muebles") en memoria
        state_dict_path = self.model_dir / 'pytorch_final.pth'
        state_dict = torch.load(state_dict_path, map_location=self.device)

        # 2. Hacemos "Ingeniería inversa" para averiguar las dimensiones ocultas
        auto_mlp_dims = []
        for key, tensor in state_dict.items():
            # Buscamos las capas Lineales (tienen 2 dimensiones en sus pesos)
            if 'weight' in key and len(tensor.shape) == 2:
                out_dim, in_dim = tensor.shape
                # Ignoramos la última capa porque siempre sale a 1 (predicción final)
                if out_dim != 1:
                    auto_mlp_dims.append(out_dim)

        print(f"[INFO] Auto-detected PyTorch hidden layers: {auto_mlp_dims}")

        # 3. Construimos el modelo con las dimensiones detectadas automáticamente
        self.mlp_model = ImprovedMLP(input_dim, hidden_dims=auto_mlp_dims).to(self.device)

        # 4. Ahora sí, metemos los pesos
        self.mlp_model.load_state_dict(state_dict)

        self.ensemble = HybridEnsemble(self.mlp_model, self.xgb_model, self.device)
        print("[INFO] AI models loaded successfully.")

    def preprocess_data(self, df):
        drop_cols = ['immunogenicity', 'mut_is_binding_pos', 'bestWTMatchType_I',
                     'DAI_MixMHC_mbp', 'mutant_rank', 'Peptide', 'cohort', 'patient',
                     'Cancer_Type', 'gene', 'wt_seq', 'mutant_best_alleles']

        X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        X_df = X_df.select_dtypes(include=['float64', 'int64'])

        print(f"[INFO] Features extracted for prediction: {X_df.shape[1]}")
        X_scaled = self.scaler.transform(X_df.values)
        return X_scaled

    def run_screening(self, dataset_path, threshold=0.5):
        print(f"\n[INFO] Reading peptide database: {dataset_path}")
        df = pd.read_csv(dataset_path)

        dummy_drop = ['immunogenicity', 'mut_is_binding_pos', 'bestWTMatchType_I', 'DAI_MixMHC_mbp', 'mutant_rank',
                      'Peptide', 'cohort', 'patient', 'Cancer_Type', 'gene', 'wt_seq', 'mutant_best_alleles']
        input_dim = \
        df.drop(columns=[c for c in dummy_drop if c in df.columns]).select_dtypes(include=['float64', 'int64']).shape[1]

        self.load_artifacts(input_dim)
        X_scaled = self.preprocess_data(df)

        print("\n[INFO] Executing predictions on all candidates...")
        final_probs = self.ensemble.predict_proba(X_scaled, weights=(0.5, 0.5))

        df['Immunogenicity_Probability'] = final_probs

        # Sort the FULL dataset descending by probability
        df = df.sort_values(by='Immunogenicity_Probability', ascending=False).reset_index(drop=True)

        self._save_results(df, threshold)
        self._generate_discovery_figures(df, threshold)

    def _save_results(self, df, threshold):
        full_out = self.reports_dir / 'Full_Dataset_Predictions.csv'
        df.to_csv(full_out, index=False)

        top100_out = self.reports_dir / 'Top_100_Vaccine_Candidates.csv'
        df.head(100).to_csv(top100_out, index=False)

        total = len(df)
        positives = (df['Immunogenicity_Probability'] >= threshold).sum()

        print("-" * 60)
        print("SCREENING REPORT")
        print("-" * 60)
        print(f"Total peptides analyzed: {total:,}")
        print(f"Viable peptides (Probability >= {threshold}): {positives:,} ({(positives / total) * 100:.2f}%)")
        print(f"Highest probability: {df['Immunogenicity_Probability'].max():.4f}")
        print(f"Lowest probability: {df['Immunogenicity_Probability'].min():.4f}")
        print("\nFiles generated:")
        print(f"  - {full_out.name}")
        print(f"  - {top100_out.name}")

    def _generate_discovery_figures(self, df, threshold):
        print("\n[INFO] Generating publication-ready figures...")

        # Figure 1: Probability Distribution (Using the FULL 1.3M dataset)
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.histplot(df['Immunogenicity_Probability'], bins=50, kde=True, color='#084594', edgecolor='black', ax=ax1)
        ax1.axvline(threshold, color='#d73027', linestyle='--', linewidth=2.5, label=f'Threshold ({threshold})')

        pos_count = (df['Immunogenicity_Probability'] >= threshold).sum()
        ax1.fill_betweenx([0, ax1.get_ylim()[1]], threshold, 1, color='#d73027', alpha=0.1)

        ax1.text(threshold + 0.02, ax1.get_ylim()[1] * 0.85, f'{pos_count:,}\nPositives',
                 color='#a50f15', fontweight='bold', fontsize=12)

        ax1.set_title('Immunogenicity Probability Distribution', pad=20)
        ax1.set_xlabel('Probability', labelpad=10)
        ax1.set_ylabel('Count', labelpad=10)

        ax1.grid(axis='x', linestyle='--', alpha=0.7)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.legend(frameon=True, edgecolor='black')

        fig1_path = self.figures_dir / 'Probability_Distribution.png'
        plt.savefig(fig1_path)
        plt.close(fig1)

        # Figure 2: Top N Peptides
        top_n = 15

        # FIX: Deduplicate ONLY for the plot so texts don't overlap,
        # keeping the occurrence with the highest probability (since it's already sorted)
        if 'Peptide' in df.columns and 'gene' in df.columns:
            plot_df = df.drop_duplicates(subset=['Peptide', 'gene'], keep='first')
        else:
            plot_df = df.copy()

        top_df = plot_df.head(top_n).sort_values(by='Immunogenicity_Probability', ascending=True)

        fig2, ax2 = plt.subplots(figsize=(10, 8))
        if 'Peptide' in top_df.columns:
            y_labels = top_df['Peptide'].astype(str) + " (" + top_df['gene'].fillna('Unknown').astype(str) + ")"
        else:
            y_labels = [f"Candidate #{i}" for i in top_df.index]

        bars = ax2.barh(y_labels, top_df['Immunogenicity_Probability'], color='#2171b5', edgecolor='black',
                        linewidth=1.5)

        ax2.set_title(f'Top {top_n} Most Promising Peptides', pad=20)
        ax2.set_xlabel('Immunogenicity Probability', labelpad=10)
        ax2.set_xlim(0, 1.1)

        ax2.grid(axis='x', linestyle='--', alpha=0.7)
        ax2.grid(axis='y', visible=False)

        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 0.015, bar.get_y() + bar.get_height() / 2, f'{width:.3f}',
                     va='center', fontsize=12)

        for spine in ax2.spines.values():
            spine.set_linewidth(1.2)

        fig2_path = self.figures_dir / 'Top_Candidates.png'
        plt.savefig(fig2_path)
        plt.close(fig2)

        print(f"[INFO] Figures saved in: {self.figures_dir}")


# ==============================================================================
# EXECUTION
# ==============================================================================
if __name__ == "__main__":
    DATASET_PATH = "data/processed/dataset_pancreas_completo.csv"
    MODELS_DIR = "models/production/hybrid_ensemble"
    REPORTS_DIR = "results/reports/predictions_discovery_reports"
    FIGURES_DIR = "predictions_discovery"

    pipeline = DiscoveryPipeline(model_dir=MODELS_DIR,
                                 reports_dir=REPORTS_DIR,
                                 figures_dir=FIGURES_DIR)
    pipeline.run_screening(dataset_path=DATASET_PATH, threshold=0.5)