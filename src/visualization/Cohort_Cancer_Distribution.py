import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from pathlib import Path
import matplotlib.ticker as ticker
import sys

class CohortVisualizer:

    def __init__(self):
        # Path configuration
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.data_path = self.project_root / 'data' / 'raw' / 'NeoPepBase.csv'

        self.output_dir = self.project_root / 'results' / 'figures' / 'cancer_cohort_figures'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.report_dir = self.project_root / 'results' / 'reports' / 'cohorts'
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self._setup_style()

    def _setup_style(self):
        """Sets up matplotlib style for scientific publication."""
        plt.style.use('seaborn-v0_8-ticks')
        # Fuentes preferidas en publicaciones científicas
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

        # Tamaños base ampliados
        rcParams['font.size'] = 14
        rcParams['axes.titlesize'] = 16
        rcParams['axes.labelsize'] = 16
        rcParams['axes.titleweight'] = 'bold'
        rcParams['axes.labelweight'] = 'bold' # Añadido para que los labels de los ejes sean bold por defecto
        rcParams['xtick.labelsize'] = 14
        rcParams['ytick.labelsize'] = 14
        rcParams['legend.fontsize'] = 12
        rcParams['legend.title_fontsize'] = 14 # Añadido para el título de las leyendas
        rcParams['figure.titlesize'] = 18

        # Espaciados por defecto (para evitar hardcodear labelpad y pad)
        rcParams['axes.labelpad'] = 10
        rcParams['axes.titlepad'] = 15

        # Configuración para exportación de alta calidad
        rcParams['figure.facecolor'] = 'white'
        rcParams['axes.facecolor'] = 'white'
        rcParams['pdf.fonttype'] = 42
        rcParams['ps.fonttype'] = 42

    def load_data(self) -> pd.DataFrame:
        """Loads and cleans the dataset."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {self.data_path}")

        print(f"Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)

        # Cohort cleaning: Replace NaN/Empty/Unknown with CEDAR
        df['cohort_clean'] = df['cohort'].fillna('CEDAR')
        df['cohort_clean'] = df['cohort_clean'].replace(['', 'Unknown'], 'CEDAR')

        return df

    def plot_global_immunogenicity(self, df: pd.DataFrame):
        """
        Figure Global: Global distribution of Positive vs Negative samples.
        """
        print("Generating Figure Global: Immunogenicity Distribution...")

        if 'immunogenicity' not in df.columns:
            print("Error: 'immunogenicity' column not found.")
            return

        # Prepare data
        counts = df['immunogenicity'].value_counts().sort_index()  # 0 then 1
        total = len(df)
        percentages = (counts / total) * 100

        # Setup figure
        fig, ax = plt.subplots(1, 1, figsize=(6, 7))

        # Colors
        colors = ['#305F88', '#4FA890']
        labels = ['Negative', 'Positive']

        # Plot bars
        bars = ax.bar(range(len(counts)), counts.values,
                      color=colors, edgecolor='#333333', linewidth=0.7, width=0.6)

        # Labels and Title
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Number of Records')
        ax.set_title('Global Immunogenicity Distribution')

        # Y-axis formatter
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

        # Annotations (Se mantiene color negro para contraste)
        for bar, count, pct in zip(bars, counts.values, percentages.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + (total * 0.01),
                    f"{count:,}\n({pct:.1f}%)",
                    ha='center', va='bottom', color='black')

        ax.set_ylim(0, max(counts.values) * 1.15)
        ax.grid(axis='y', alpha=0.15, linestyle='--', color='gray')
        sns.despine(bottom=False, left=False, top=True, right=True)

        plt.tight_layout()
        save_path = self.output_dir / 'figure_global_immunogenicity.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_cancer_distribution(self, df: pd.DataFrame):
        """Figure 1: Distribution by cancer type (Scientific Style)."""
        print("Generating Figure 1: Cancer Distribution...")
        cancer_counts = df['Cancer_Type'].value_counts()

        # CAMBIO 1: Aumentamos el ancho de la figura de 12 a 16 para separar las barras
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))

        bar_color = '#305F88'

        bars = ax.bar(range(len(cancer_counts)), cancer_counts.values,
                      color=bar_color, edgecolor='#333333', linewidth=0.7, width=0.75)

        ax.set_xlabel('Cancer Type')
        ax.set_ylabel('Number of Records')
        ax.set_title('Global Distribution of Cancer Types')

        max_y = max(cancer_counts.values)

        for bar, count in zip(bars, cancer_counts.values):
            height = bar.get_height()
            # CAMBIO 2: Añadimos rotation=90 para que los números suban en vertical y no choquen
            ax.text(bar.get_x() + bar.get_width() / 2, height + (max_y * 0.02),
                    f'{count:,}', ha='center', va='bottom', color='black',
                    rotation=90)

        ax.set_xticks(range(len(cancer_counts)))
        ax.set_xticklabels(cancer_counts.index, rotation=45, ha='right')

        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

        # CAMBIO 3: Damos un 25% extra de margen superior (en vez de 10%) para que quepan los textos rotados
        ax.set_ylim(0, max_y * 1.25)

        ax.grid(axis='y', alpha=0.2, linestyle='--', color='gray')
        sns.despine()

        plt.tight_layout()
        save_path = self.output_dir / 'figure1_cancer_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_cohort_distribution(self, df: pd.DataFrame):
        """Figure 2: Stacked Distribution by Cohort + CSV Export."""
        print("Generating Figure 2: Cohort Distribution (Stacked)...")

        if 'immunogenicity' not in df.columns:
            print("Error: 'immunogenicity' column not found.")
            return

        # Prepare Data
        cohort_stats = df.groupby('cohort_clean')['immunogenicity'].value_counts().unstack(fill_value=0)
        cohort_stats = cohort_stats.sort_index(axis=1)
        cohort_stats['Total'] = cohort_stats.sum(axis=1)
        cohort_stats = cohort_stats.sort_values('Total', ascending=True)

        # --- CSV EXPORT BLOCK ---
        rename_map = {}
        if 0 in cohort_stats.columns: rename_map[0] = 'Negative'
        if 1 in cohort_stats.columns: rename_map[1] = 'Positive'

        csv_data = cohort_stats.rename(columns=rename_map)
        if 'Negative' not in csv_data.columns: csv_data['Negative'] = 0
        if 'Positive' not in csv_data.columns: csv_data['Positive'] = 0
        csv_data = csv_data[['Negative', 'Positive', 'Total']]

        csv_path = self.report_dir / 'cohort_stats.csv'
        csv_data.to_csv(csv_path)
        print(f"Saved cohort statistics to: {csv_path}")
        # ------------------------

        plot_data = cohort_stats.drop(columns='Total')

        fig, ax = plt.subplots(1, 1, figsize=(11, 7))

        colors = ['#305F88', '#4FA890']

        plot_data.plot(kind='barh', stacked=True, ax=ax, color=colors,
                       edgecolor='#333333', linewidth=0.7, width=0.75)

        ax.set_ylabel('Cohort')
        ax.set_xlabel('Number of Records')
        ax.set_title('Immunogenicity Distribution by Cohort')

        ax.legend(title='Immunogenicity', loc='lower right', frameon=True, framealpha=0.9, edgecolor='white')

        label_threshold = 200
        for c in ax.containers:
            labels = [f'{int(v.get_width())}' if v.get_width() > label_threshold else '' for v in c]
            ax.bar_label(c, labels=labels, label_type='center', color='white')

        max_x = 0
        for idx, (cohort, row) in enumerate(cohort_stats.iterrows()):
            total = row['Total']
            max_x = max(max_x, total)
            ax.text(total + (total * 0.015), idx,
                    f'{int(total)}',
                    ha='left', va='center', color='black')

        ax.grid(axis='x', alpha=0.2, linestyle='--', color='gray')
        ax.set_xlim(0, max_x * 1.1)
        sns.despine(left=False, bottom=False, right=True, top=False)

        plt.tight_layout()
        save_path = self.output_dir / 'figure2_cohort_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_heatmap(self, df: pd.DataFrame):
        """Figure 3: Scientific Heatmap (Filtered < 30 cases & No Unknown)."""
        print("Generating Figure 3: Scientific Heatmap (Filtered)...")

        # 1. Data Preparation
        df['cohort_clean'] = df['cohort_clean'].replace('Unknown', 'CEDAR')

        cancer_counts = df['Cancer_Type'].value_counts()
        valid_cancers = cancer_counts[cancer_counts >= 30].index.tolist()
        if 'Unknown' in valid_cancers:
            valid_cancers.remove('Unknown')

        df_filtered = df[df['Cancer_Type'].isin(valid_cancers)]
        print(f"Heatmap filtered: Keeping {len(valid_cancers)} cancer types (>= 30 cases, excluding Unknown).")

        if len(df_filtered) == 0:
            print("Warning: No valid cancer types found. Skipping heatmap.")
            return

        cross_tab = pd.crosstab(df_filtered['cohort_clean'], df_filtered['Cancer_Type'], normalize='index') * 100
        cross_tab = cross_tab.T

        cohort_order = df['cohort_clean'].value_counts().index
        cancer_order = valid_cancers
        cross_tab = cross_tab.reindex(index=cancer_order, columns=cohort_order)

        n_rows, n_cols = cross_tab.shape

        # CAMBIO 1: Hacemos que el multiplicador de anchura sea mucho mayor (2.5)
        # Esto estirará la figura horizontalmente dando espacio a los porcentajes.
        figsize_w = n_cols * 2.5 + 4
        figsize_h = n_rows * 0.6 + 2

        fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
        cmap = 'magma_r'

        # CAMBIO 2: ELIMINADO el parámetro 'square=True'
        # Esto permite que las celdas adopten una forma rectangular más natural
        sns.heatmap(cross_tab, cmap=cmap, ax=ax,
                    linewidths=0.5, linecolor='white',
                    cbar_kws={'label': 'Prevalence (%)', 'shrink': 0.7, 'aspect': 25},
                    vmin=0, vmax=100)

        threshold_to_show = 2.0
        threshold_text_color = 40.0

        for y in range(cross_tab.shape[0]):
            for x in range(cross_tab.shape[1]):
                val = cross_tab.iloc[y, x]
                if pd.isna(val): continue

                if val >= threshold_to_show:
                    text_color = 'white' if val > threshold_text_color else 'black'
                    ax.text(x + 0.5, y + 0.5, f'{val:.1f}%',
                            ha='center', va='center', color=text_color)

        ax.set_xlabel('Cohort')
        ax.set_ylabel('Cancer Type (Known & >= 30 Cases)')
        ax.set_title('Cancer Type Prevalence by Cohort')

        plt.yticks(rotation=0)
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        save_path = self.output_dir / 'figure3_heatmap_scientific.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_stacked_bars(self, df: pd.DataFrame):
        """Figure 3b: Stacked bar chart (Cancer Types)."""
        print("Generating Figure 3b: Stacked Bars...")
        cancer_order = df['Cancer_Type'].value_counts().index
        cohort_order = df['cohort_clean'].value_counts().index
        stack_data = pd.crosstab(df['cohort_clean'], df['Cancer_Type']).reindex(cohort_order)[cancer_order]

        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        colors = sns.color_palette("husl", len(stack_data.columns))

        bottom_vals = np.zeros(len(stack_data))
        for i, cancer_type in enumerate(stack_data.columns):
            values = stack_data[cancer_type].values
            ax.bar(range(len(stack_data)), values, bottom=bottom_vals,
                   color=colors[i], edgecolor='white', linewidth=0.5,
                   label=cancer_type)
            bottom_vals += values

        ax.set_xlabel('Cohort')
        ax.set_ylabel('Number of Records')
        ax.set_title('Cancer Type Distribution by Cohort')
        ax.set_xticks(range(len(stack_data)))
        ax.set_xticklabels(stack_data.index, rotation=45, ha='right')

        ax.grid(axis='y', alpha=0.2, linestyle='--', color='gray')
        sns.despine()

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1],
                  title='Cancer Type',
                  bbox_to_anchor=(1.01, 1),
                  loc='upper left',
                  borderaxespad=0.,
                  frameon=True)

        plt.tight_layout()
        save_path = self.output_dir / 'figure3b_stacked_bars.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def run_all(self):
        try:
            df = self.load_data()
            self.plot_global_immunogenicity(df)
            self.plot_cancer_distribution(df)
            self.plot_cohort_distribution(df)
            self.plot_heatmap(df)
            self.plot_stacked_bars(df)
            print(f"\nAll figures saved to: {self.output_dir}")
            print(f"Cohort report saved to: {self.report_dir}")
        except Exception as e:
            print(f"Error during execution: {e}")


if __name__ == "__main__":
    CohortVisualizer().run_all()