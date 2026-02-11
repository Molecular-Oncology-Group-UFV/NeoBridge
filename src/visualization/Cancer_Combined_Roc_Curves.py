import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Configuración básica de estilo limpio
plt.style.use('seaborn-v0_8-white')
warnings.filterwarnings('ignore')


class SplitFigureCombiner:
    def __init__(self):
        # Definir la raíz del proyecto
        self.project_root = Path(__file__).resolve().parent.parent.parent

        # Rutas a los Datasets
        self.data_paths = {
            'Melanoma': 'data/experiments/melanoma/dataset_80_20_only_melanoma.csv',
            'Lung': 'data/experiments/pulmon/dataset_only_lung.csv',
            'Pancreas': 'data/experiments/pancreas/dataset_only_pancreas.csv',
            'Minor Cancers': 'data/experiments/multiples_minoritarios/dataset_80_20_minor_cancers.csv'
        }

        # Rutas a las Imágenes ROC existentes
        self.image_paths = {
            'Melanoma': 'results/figures/generalization/melanoma/roc_curve.png',
            'Lung': 'results/figures/generalization/lung/roc_curve.png',
            'Pancreas': 'results/figures/generalization/pancreas/roc_curve.png',
            'Minor Cancers': 'results/figures/generalization/minor_cancers/roc_curve.png'
        }

        self.output_dir = self.project_root / 'results/figures/generalization'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_dataset_stats(self, cohort):
        """Obtiene estadísticas básicas del CSV."""
        path = self.project_root / self.data_paths[cohort]
        if not path.exists(): return 0, 0.0, None

        df = pd.read_csv(path)
        n_total = len(df)
        n_pos = df['immunogenicity'].sum()
        pos_ratio = (n_pos / n_total) * 100

        dist_data = None
        if cohort == 'Minor Cancers':
            col = 'Cancer_Type' if 'Cancer_Type' in df.columns else 'cancer_type'
            if col in df.columns:
                # Calcular conteos
                counts = df[col].value_counts()

                # --- CORRECCIÓN: Eliminar 'Healthy' si existe ---
                if 'Healthy' in counts.index:
                    counts = counts.drop('Healthy')
                    # print(f"['Minor Cancers'] Excluded 'Healthy' category from top list.")

                # Top 6 de los restantes
                dist_data = counts.head(6)
                dist_data = dist_data.iloc[::-1]

        return n_total, pos_ratio, dist_data

    # --- NUEVA FUNCIÓN PARA EL GRÁFICO INDEPENDIENTE ---
    def create_standalone_composition_plot(self, dist_data, filename):
        """Genera un gráfico de barras independiente y limpio para la composición."""
        print(f"Generando figura de composición: {filename}...")

        # Tamaño similar a un panel de la figura principal (aprox 8x6 o 8x7)
        fig, ax = plt.subplots(figsize=(8, 7))

        y_pos = np.arange(len(dist_data))
        values = dist_data.values
        labels = dist_data.index

        # Barras con colores suaves
        bars = ax.barh(y_pos, values, color=plt.cm.Pastel1.colors[:len(values)],
                       edgecolor='#666666', height=0.6)

        # Configuración de ejes para un look super limpio
        ax.set_yticks(y_pos)
        # Etiquetas del eje Y más grandes y legibles
        ax.set_yticklabels(labels, fontsize=12, fontweight='bold', color='#333333')
        ax.set_xticks([])  # Sin eje X

        # Eliminar los bordes del gráfico (spines)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Título principal
        ax.set_title('Minor Cancers Cohort Composition (Top 6)', fontsize=16, fontweight='bold', pad=20, color='black')

        # Números al final de las barras
        max_val = values.max()
        for bar in bars:
            width = bar.get_width()
            # Texto un poco separado de la barra
            ax.text(width + (max_val * 0.02), bar.get_y() + bar.get_height() / 2,
                    str(int(width)), va='center', ha='left',
                    fontsize=12, color='black', fontweight='bold')

        # Ajustar límite X para que quepan los números
        ax.set_xlim(0, max_val * 1.15)

        save_path = self.output_dir / filename
        # bbox_inches='tight' es clave para eliminar márgenes blancos extra
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Cerrar la figura para liberar memoria
        print(f"✅ Figura de composición guardada independientemente: {save_path}")

    def create_two_panel_figure(self, cohorts_list, letters_list, filename):
        """Genera una figura de 1 fila y 2 columnas con las curvas ROC."""
        print(f"Generando figura compuesta: {filename}...")

        # Figura más ancha que alta (16x7)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'wspace': 0.15})
        axes = axes.flatten()

        for i, cohort in enumerate(cohorts_list):
            ax = axes[i]

            # 1. Cargar imagen base
            img_path = self.project_root / self.image_paths[cohort]
            if img_path.exists():
                img = mpimg.imread(img_path)
                ax.imshow(img)
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, "Image Missing", ha='center')
                ax.axis('off')
                continue

            # 2. Datos (solo necesitamos n_total y pos_ratio aquí)
            n_total, pos_ratio, _ = self.get_dataset_stats(cohort)

            # 3. ETIQUETA LETRA (A/B o C/D)
            ax.text(-0.05, 1.02, letters_list[i], transform=ax.transAxes,
                    fontsize=24, fontweight='bold', va='bottom', color='black')

            # 4. CAJA DE ESTADÍSTICAS (Posición superior centrada)
            stats_text = f"Total Samples: {n_total} | Positive: {pos_ratio:.1f}%"

            ax.text(0.5, 1.03, stats_text, transform=ax.transAxes,
                    fontsize=12,
                    fontweight='bold',
                    ha='center', va='bottom',
                    color='#333333',
                    bbox=dict(facecolor='white', edgecolor='#dddddd', boxstyle='round,pad=0.4', alpha=1.0))

            # --- ELIMINADO: Ya no se inserta el gráfico pequeño aquí ---

        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Figura compuesta guardada: {save_path}")

    def run(self):
        # Figura 1: Melanoma y Pulmón
        self.create_two_panel_figure(
            ['Melanoma', 'Lung'],
            ['A', 'B'],
            'figure_generalization_part1.png'
        )

        # Figura 2: Páncreas y Minoritarios (Ahora limpia sin el inset)
        self.create_two_panel_figure(
            ['Pancreas', 'Minor Cancers'],
            ['C', 'D'],
            'figure_generalization_part2.png'
        )

        # --- NUEVO PASO: Generar el gráfico de composición aparte ---
        # 1. Obtener los datos específicos de Minor Cancers
        _, _, dist_data_minor = self.get_dataset_stats('Minor Cancers')

        if dist_data_minor is not None:
            # 2. Llamar a la función dedicada para crear la figura sola
            self.create_standalone_composition_plot(
                dist_data_minor,
                'figure_minor_cancers_composition_standalone.png'
            )


if __name__ == "__main__":
    combiner = SplitFigureCombiner()
    combiner.run()