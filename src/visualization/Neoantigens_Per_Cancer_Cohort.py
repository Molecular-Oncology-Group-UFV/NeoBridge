import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from matplotlib import rcParams

# ==============================================================================
# CONFIGURACIÓN DE MATPLOTLIB (Mantenemos tu estilo de publicación)
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


def analyze_cohort_enrichment(predictions_path, output_dir, top_n=1000):
    print(f"[INFO] Cargando las predicciones desde: {predictions_path}")
    df = pd.read_csv(predictions_path)

    # Asegurarnos de que el dataset está ordenado por probabilidad (por si acaso)
    df = df.sort_values(by='Immunogenicity_Probability', ascending=False).reset_index(drop=True)

    # Filtrar solo las filas que tienen una cohorte asignada (ignorar NaNs)
    df_valid = df.dropna(subset=['cohort'])

    # 1. Calcular proporciones en el DATASET COMPLETO (Background)
    total_counts = df_valid['cohort'].value_counts()
    total_props = (total_counts / total_counts.sum()) * 100

    # 2. Calcular proporciones en el TOP N
    top_df = df_valid.head(top_n)
    top_counts = top_df['cohort'].value_counts()
    top_props = (top_counts / top_counts.sum()) * 100

    print("\n" + "=" * 50)
    print(f"REPORTE DE COHORTES (TOP {top_n})")
    print("=" * 50)
    for cohort in total_counts.index:
        bg_pct = total_props.get(cohort, 0)
        top_pct = top_props.get(cohort, 0)
        enrichment = top_pct / bg_pct if bg_pct > 0 else 0
        print(f"[{cohort}]")
        print(f"  - Dataset global: {total_counts.get(cohort, 0):,} péptidos ({bg_pct:.1f}%)")
        print(f"  - En el Top {top_n}: {top_counts.get(cohort, 0):,} péptidos ({top_pct:.1f}%)")
        print(f"  - Enriquecimiento: {enrichment:.2f}x")

    # 3. Preparar datos para la figura
    plot_data = []
    for cohort in total_counts.index:
        plot_data.append({'Cohorte': cohort, 'Porcentaje': total_props.get(cohort, 0), 'Grupo': 'Dataset Global'})
        plot_data.append(
            {'Cohorte': cohort, 'Porcentaje': top_props.get(cohort, 0), 'Grupo': f'Top {top_n} Inmunogénicos'})

    df_plot = pd.DataFrame(plot_data)

    # ==============================================================================
    # CREACIÓN DE LA FIGURA
    # ==============================================================================
    print("\n[INFO] Generando figura comparativa...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Gráfico de barras agrupadas usando seaborn
    sns.barplot(
        data=df_plot,
        x='Cohorte',
        y='Porcentaje',
        hue='Grupo',
        palette=['#878787', '#d73027'],  # Gris para el global, Rojo/Destacado para el Top
        edgecolor='black',
        linewidth=1.5,
        ax=ax
    )

    # Personalización del gráfico
    ax.set_title(f'Distribución de Cohortes: Global vs Top {top_n}', pad=20)
    ax.set_ylabel('Proporción de Péptidos (%)', labelpad=10)
    ax.set_xlabel('Cohorte de Origen', labelpad=10)
    ax.set_ylim(0, 100)  # Mantener el eje Y de 0 a 100% para evitar malinterpretaciones visuales

    # Añadir los porcentajes encima de las barras
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=11, fontweight='bold')

    # Ajustes de la leyenda
    ax.legend(title='', frameon=True, edgecolor='black', loc='upper right')

    # Bordes más gruesos para estilo científico
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    # Guardar la figura
    out_file = Path(output_dir) / f'Cohort_Enrichment_Top{top_n}.png'
    plt.savefig(out_file)
    plt.close(fig)
    print(f"[INFO] Figura guardada con éxito en: {out_file}")


if __name__ == '__main__':
    # Rutas (Ajustalas según la estructura de tus carpetas)
    PREDICTIONS_CSV = "results/reports/predictions_discovery_reports/Full_Dataset_Predictions.csv"
    OUTPUT_DIR = "predictions_discovery"

    analyze_cohort_enrichment(
        predictions_path=PREDICTIONS_CSV,
        output_dir=OUTPUT_DIR,
        top_n=1000
    )