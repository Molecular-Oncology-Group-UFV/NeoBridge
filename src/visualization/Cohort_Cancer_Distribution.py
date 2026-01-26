import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from pathlib import Path
import sys


class CohortVisualizer:

    def __init__(self):
        # Path configuration
        # Project root is 2 levels up from src/visualization
        self.project_root = Path(__file__).resolve().parent.parent.parent

        # Input path (Processed data)
        self.data_path = self.project_root / 'data' / 'processed' / 'dataset_80_20_normalized_tcr_with_cohort.csv'

        # Output path (Figures)
        self.output_dir = self.project_root / 'results' / 'figures' / 'cancer_cohort_figures'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Style configuration
        self._setup_style()

    def _setup_style(self):
        """Sets up matplotlib style for scientific publication."""
        plt.style.use('seaborn-v0_8-whitegrid')
        rcParams['font.family'] = 'DejaVu Sans'
        rcParams['font.size'] = 10
        rcParams['axes.labelsize'] = 11
        rcParams['axes.titlesize'] = 12
        rcParams['xtick.labelsize'] = 9
        rcParams['ytick.labelsize'] = 9
        rcParams['legend.fontsize'] = 9
        rcParams['figure.titlesize'] = 12

    def load_data(self) -> pd.DataFrame:
        """Loads and cleans the dataset."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {self.data_path}")

        print(f"Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)

        # Cohort cleaning
        df['cohort_clean'] = df['cohort'].fillna('Unknown')
        df['cohort_clean'] = df['cohort_clean'].replace('', 'Unknown')

        return df

    def plot_cancer_distribution(self, df: pd.DataFrame):
        """Figure 1: Distribution by cancer type."""
        print("Generating Figure 1: Cancer Distribution...")
        cancer_counts = df['Cancer_Type'].value_counts()

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(cancer_counts)))

        bars = ax.bar(range(len(cancer_counts)), cancer_counts.values,
                      color=colors, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Cancer Type', fontweight='bold')
        ax.set_ylabel('Number of Records', fontweight='bold')
        ax.set_title('Cancer Type Distribution in Dataset', fontweight='bold', pad=15)

        # Annotations
        for i, (bar, count) in enumerate(zip(bars, cancer_counts.values)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(cancer_counts.values) * 0.01,
                    str(count), ha='center', va='bottom', fontsize=8)

        ax.set_xticks(range(len(cancer_counts)))
        ax.set_xticklabels(cancer_counts.index, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        save_path = self.output_dir / 'figure1_cancer_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_cohort_distribution(self, df: pd.DataFrame):
        """Figure 2: Distribution by Cohort."""
        print("Generating Figure 2: Cohort Distribution...")
        cohort_counts = df['cohort_clean'].value_counts()
        cohort_counts_sorted = cohort_counts.sort_values()

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Specific color for 'Unknown'
        colors = ['#1f77b4'] * len(cohort_counts_sorted)
        if 'Unknown' in cohort_counts_sorted.index:
            idx = list(cohort_counts_sorted.index).index('Unknown')
            colors[idx] = '#ff7f0e'  # Orange for unknown

        bars = ax.barh(range(len(cohort_counts_sorted)), cohort_counts_sorted.values,
                       color=colors, edgecolor='black', linewidth=0.5)

        ax.set_ylabel('Cohort', fontweight='bold')
        ax.set_xlabel('Number of Records', fontweight='bold')
        ax.set_title('Record Distribution by Cohort', fontweight='bold', pad=15)

        for i, (bar, count) in enumerate(zip(bars, cohort_counts_sorted.values)):
            ax.text(bar.get_width() + max(cohort_counts_sorted.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                    str(count), ha='left', va='center', fontsize=8)

        ax.set_yticks(range(len(cohort_counts_sorted)))
        ax.set_yticklabels(cohort_counts_sorted.index)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        plt.tight_layout()
        save_path = self.output_dir / 'figure2_cohort_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_dot_plot(self, df: pd.DataFrame):
        """Figure 3: Dot Plot (Balloon Plot)."""
        print("Generating Figure 3: Dot Plot...")

        # Prepare data
        cross_tab = pd.crosstab(df['cohort_clean'], df['Cancer_Type'], normalize='index') * 100

        cancer_order = df['Cancer_Type'].value_counts().index
        cohort_order = df['cohort_clean'].value_counts().index
        cross_tab = cross_tab.reindex(cohort_order)[cancer_order]

        plot_data = cross_tab.reset_index().melt(id_vars='cohort_clean', var_name='Cancer_Type',
                                                 value_name='Percentage')
        plot_data = plot_data[plot_data['Percentage'] > 0]

        fig, ax = plt.subplots(figsize=(14, 6))

        scatter = ax.scatter(
            x=plot_data['Cancer_Type'],
            y=plot_data['cohort_clean'],
            s=plot_data['Percentage'] * 30,
            c=plot_data['Percentage'],
            cmap='YlOrRd',
            edgecolors='black',
            linewidth=0.5,
            alpha=0.9,
            vmin=0, vmax=100
        )

        ax.set_axisbelow(True)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlabel('Cancer Type', fontweight='bold', fontsize=11)
        ax.set_ylabel('Cohort', fontweight='bold', fontsize=11)
        ax.set_title('Cancer Type Prevalence by Cohort', fontweight='bold', fontsize=13, pad=15)
        ax.set_ylim(-0.5, len(cohort_order) - 0.5)

        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)

        cbar = plt.colorbar(scatter, pad=0.02)
        cbar.set_label('Percentage (%)', fontweight='bold', rotation=270, labelpad=15)

        for i, row in plot_data.iterrows():
            if row['Percentage'] > 5.0:
                ax.text(
                    row['Cancer_Type'],
                    row['cohort_clean'],
                    f"{row['Percentage']:.0f}",
                    ha='center', va='center',
                    fontsize=8,
                    fontweight='bold',
                    color='white' if row['Percentage'] > 50 else 'black'
                )

        plt.tight_layout()
        save_path = self.output_dir / 'figure3_dotplot_clean.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_stacked_bars(self, df: pd.DataFrame):
        """Figure 3b: Stacked bar chart."""
        print("Generating Figure 3b: Stacked Bars...")

        cancer_order = df['Cancer_Type'].value_counts().index
        cohort_order = df['cohort_clean'].value_counts().index

        stack_data = pd.crosstab(df['cohort_clean'], df['Cancer_Type'])
        stack_data = stack_data.reindex(cohort_order)
        stack_data = stack_data[cancer_order]

        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        n_cancers = len(stack_data.columns)
        colors = plt.cm.tab20c(np.linspace(0, 1, n_cancers))

        bottom_vals = np.zeros(len(stack_data))
        for i, cancer_type in enumerate(stack_data.columns):
            values = stack_data[cancer_type].values
            ax.bar(range(len(stack_data)), values, bottom=bottom_vals,
                   color=colors[i], edgecolor='white', linewidth=0.5,
                   label=cancer_type if i < 15 else None)
            bottom_vals += values

        ax.set_xlabel('Cohort', fontweight='bold')
        ax.set_ylabel('Number of Records', fontweight='bold')
        ax.set_title('Cancer Type Distribution by Cohort', fontweight='bold', pad=15)
        ax.set_xticks(range(len(stack_data)))
        ax.set_xticklabels(stack_data.index, rotation=45, ha='right')

        if n_cancers <= 15:
            ax.legend(title='Cancer Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Separate legend if there are many types
            self._save_separate_legend(stack_data.columns.tolist(), colors)

        plt.tight_layout()
        save_path = self.output_dir / 'figure3b_stacked_bars.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _save_separate_legend(self, labels, colors):
        """Saves the legend to a separate file if it is too large."""
        fig_legend, ax_legend = plt.subplots(figsize=(8, 10))
        ax_legend.axis('off')
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], edgecolor='black')
                           for i in range(len(labels))]
        ax_legend.legend(legend_elements, labels,
                         title='Cancer Type', loc='center', ncol=2)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cancer_type_legend.png', dpi=300, bbox_inches='tight')
        plt.close()

    def run_all(self):
        """Executes all visualizations."""
        try:
            df = self.load_data()
            self.plot_cancer_distribution(df)
            self.plot_cohort_distribution(df)
            self.plot_dot_plot(df)
            self.plot_stacked_bars(df)
            print(f"\nAll figures saved to: {self.output_dir}")
        except Exception as e:
            print(f"Error during execution: {e}")


def main():
    visualizer = CohortVisualizer()
    visualizer.run_all()


if __name__ == "__main__":
    main()