import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.calibration import calibration_curve
from matplotlib import rcParams
from pathlib import Path

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
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300
})


class CalibrationAnalyzer:
    def __init__(self):
        # Rutas de archivos basadas en la estructura de tu proyecto
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.preds_path = self.project_root / 'results' / 'reports' / 'performance' / 'test_predictions_for_delong.csv'
        self.reports_dir = self.project_root / 'results' / 'reports' / 'benchmarking'
        self.figures_dir = self.project_root / 'results' / 'figures' / 'benchmarking'

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def calculate_ece(self, y_true, y_prob, n_bins=10):
        """Calcula el Expected Calibration Error (ECE)"""
        bins = np.linspace(0., 1., n_bins + 1)
        binids = np.digitize(y_prob, bins) - 1
        ece = 0.0

        for i in range(n_bins):
            bin_mask = (binids == i)
            if np.sum(bin_mask) > 0:
                acc = np.mean(y_true[bin_mask])
                conf = np.mean(y_prob[bin_mask])
                weight = np.sum(bin_mask) / len(y_prob)
                ece += np.abs(acc - conf) * weight
        return ece

    def analyze_calibration(self):
        print("=" * 60)
        print("ANALIZANDO CALIBRACIÓN DE PROBABILIDADES (REVISOR 2)")
        print("=" * 60)

        try:
            df = pd.read_csv(self.preds_path)
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo en {self.preds_path}")
            return

        y_true = df['y_true'].values

        models = {
            'XGBoost Alone': df['prob_xgb'].values,
            'PyTorch MLP Alone': df['prob_mlp'].values,
            'Full Hybrid Ensemble': df['prob_ensemble'].values
        }

        results = []

        # 1. Calcular métricas para cada modelo
        for name, probs in models.items():
            auc = roc_auc_score(y_true, probs)
            brier = brier_score_loss(y_true, probs)
            lloss = log_loss(y_true, probs)
            ece = self.calculate_ece(y_true, probs)

            results.append({
                'Model': name,
                'AUC-ROC (Ranking)': auc,
                'Brier Score ↓': brier,
                'Log Loss ↓': lloss,
                'ECE ↓': ece
            })

        # Mostrar tabla en consola y guardar en CSV
        df_results = pd.DataFrame(results)
        print("\n--- RESULTADOS DE CALIBRACIÓN (Menor es mejor en Brier, Log Loss y ECE) ---")
        print(df_results.to_string(index=False))

        csv_path = self.reports_dir / 'calibration_metrics_comparison.csv'
        df_results.to_csv(csv_path, index=False)
        print(f"\n✓ Tabla guardada en: {csv_path}")

        # 2. Generar Reliability Diagram Comparativo
        self.plot_comparative_reliability(y_true, models)

    def plot_comparative_reliability(self, y_true, models):
        plt.figure(figsize=(9, 7))

        # Colores consistentes con la otra gráfica
        colors = {
            'Full Hybrid Ensemble': '#084594',  # Azul oscuro
            'XGBoost Alone': '#D95319',  # Naranja
            'PyTorch MLP Alone': '#2ca02c'  # Verde
        }

        # Línea de calibración perfecta
        plt.plot([0, 1], [0, 1], 'k:', label='Perfectly Calibrated', linewidth=2)

        for name, probs in models.items():
            # Extraer puntos de calibración (10 deciles)
            fraction_of_positives, mean_predicted_value = calibration_curve(y_true, probs, n_bins=10,
                                                                            strategy='uniform')

            linewidth = 3.5 if 'Full' in name else 2.0
            marker = 's' if 'Full' in name else 'o'
            markersize = 9 if 'Full' in name else 7

            plt.plot(mean_predicted_value, fraction_of_positives, marker=marker,
                     linewidth=linewidth, markersize=markersize, color=colors[name],
                     label=f"{name}")

        plt.xlabel('Mean Predicted Probability', fontsize=14, fontweight='bold')
        plt.ylabel('Fraction of Positives (True Probability)', fontsize=14, fontweight='bold')
        plt.title('Comparative Reliability Diagram', fontsize=16, fontweight='bold', pad=15)

        plt.legend(loc='lower right', frameon=True, edgecolor='black', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        sns.despine()

        # Guardar gráfico
        plot_path = self.figures_dir / 'comparative_reliability_diagram.png'
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"✓ Gráfico comparativo guardado en: {plot_path}")


if __name__ == "__main__":
    analyzer = CalibrationAnalyzer()
    analyzer.analyze_calibration()