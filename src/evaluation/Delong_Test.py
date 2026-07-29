import pandas as pd
import numpy as np
from MLstatkit import Delong_test
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

def calculate_ece(y_true, y_prob, n_bins=10):
    """Calcula el Expected Calibration Error (ECE)"""
    bin_edges = np.linspace(0., 1., n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if np.sum(bin_mask) > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            ece += np.abs(bin_acc - bin_conf) * np.sum(bin_mask) / len(y_true)
    return ece

def run_statistical_comparison(predictions_path):
    print("=" * 70)
    print("EJECUTANDO TEST ESTADÍSTICO DE DeLONG Y GENERANDO TABLA")
    print("=" * 70)

    # 1. Cargar las predicciones exportadas por el pipeline
    try:
        df = pd.read_csv(predictions_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {predictions_path}")
        return

    y_true = df['y_true'].values
    prob_ensemble = df['prob_ensemble'].values
    prob_xgb = df['prob_xgb'].values
    prob_mlp = df['prob_mlp'].values

    # 2. Ejecutar tests de DeLong tomando XGBoost como "Referencia"
    # Comparamos Ensemble vs XGBoost
    res_xgb_ens = Delong_test(y_true, prob_xgb, prob_ensemble)
    p_value_ensemble = res_xgb_ens[1]

    # Comparamos MLP vs XGBoost
    res_xgb_mlp = Delong_test(y_true, prob_xgb, prob_mlp)
    p_value_mlp = res_xgb_mlp[1]

    # 3. Calcular todas las métricas y estructurar la tabla
    models = {
        "XGBoost (Aislado)": prob_xgb,
        "PyTorch MLP (Aislada)": prob_mlp,
        "NeoBridge (Ensemble Híbrido)": prob_ensemble
    }

    table_data = []

    for model_name, y_prob in models.items():
        # Calcular métricas
        auc = roc_auc_score(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        logloss = log_loss(y_true, y_prob)
        ece = calculate_ece(y_true, y_prob)

        # Asignar el p-value correspondiente
        if "XGBoost" in model_name:
            p_val_str = "Referencia"
        elif "MLP" in model_name:
            p_val_str = f"{p_value_mlp:.4e}"
        else:
            # Poner en negrita si es significativo para que destaque en markdown
            if p_value_ensemble < 0.05:
                p_val_str = f"**{p_value_ensemble:.4e}**"
            else:
                p_val_str = f"{p_value_ensemble:.4e}"

        # Guardar la fila
        table_data.append({
            "Modelo Predictivo": model_name,
            "AUC-ROC": f"{auc:.3f}",
            "DeLong p-value (vs XGBoost)": p_val_str,
            "Brier Score": f"{brier:.3f}",
            "Log Loss": f"{logloss:.3f}",
            "ECE": f"{ece:.3f}"
        })

    # 4. Crear DataFrame y exportar resultados
    results_df = pd.DataFrame(table_data)

    print("\n--- RESULTADOS (TABLA SUPLEMENTARIA) ---")
    print(results_df.to_markdown(index=False))

    # Guardar en CSV
    output_path = "results/reports/performance/delong_calibration_table.csv"
    try:
        results_df.to_csv(output_path, index=False)
        print(f"\n✅ Tabla guardada correctamente en: {output_path}")
    except Exception as e:
        print(f"\n⚠️ No se pudo guardar el CSV: {e}")

    print("-" * 70)

if __name__ == "__main__":
    preds_path = "results/reports/performance/test_predictions_for_delong.csv"
    run_statistical_comparison(preds_path)