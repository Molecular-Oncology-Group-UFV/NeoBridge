import pandas as pd
from pathlib import Path


def generate_hla_distribution_table():
    # 1. Definir rutas (asumiendo tu estructura de carpetas)
    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / 'data' / 'raw' / 'NeoPepBase.csv'
    reports_dir = project_root / 'results' / 'reports' / 'supplementary'
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 2. Cargar datos
    print("Cargando NeoPepBase...")
    df = pd.read_csv(data_path)

    # ¡IMPORTANTE! Cambia esto por el nombre real de tu columna de HLA
    hla_col = 'mutant_best_alleles'

    # (Opcional) Si la tabla es solo del Training set y tienes una columna de split
    # df = df[df['dataset_split'] == 'train']

    # 3. Calcular distribución (Frecuencia absoluta y Porcentaje)
    hla_counts = df[hla_col].value_counts().reset_index()
    hla_counts.columns = ['HLA Allele', 'Count']

    total_samples = hla_counts['Count'].sum()
    hla_counts['Percentage (%)'] = (hla_counts['Count'] / total_samples * 100).round(2)

    # 4. Exportar a CSV y Markdown
    csv_path = reports_dir / 'Supplementary_Table_HLA_Distribution.csv'
    md_path = reports_dir / 'Supplementary_Table_HLA_Distribution.md'

    hla_counts.to_csv(csv_path, index=False)

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Supplementary Table X: HLA Allele Distribution in the Training Set\n\n")
        f.write(hla_counts.to_markdown(index=False))

    print(f"✓ Tabla generada exitosamente. Total de alelos únicos: {len(hla_counts)}")
    print(f"✓ Guardado en: {csv_path}")


if __name__ == "__main__":
    generate_hla_distribution_table()