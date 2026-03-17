import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from upsetplot import UpSet
from Bio import SeqIO
import requests
import time

# Silenciar los FutureWarnings internos de la librería upsetplot
warnings.filterwarnings("ignore", category=FutureWarning, module="upsetplot")

# Configuración del estilo de las gráficas
plt.style.use('seaborn-v0_8-whitegrid')


# ==========================================================
# FUNCIONES AUXILIARES DE UNIPROT Y ANOTACIÓN CLÍNICA
# ==========================================================
def get_uniprot_location_by_gene(gene_name):
    if pd.isna(gene_name):
        return "Unknown"

    url = f"https://rest.uniprot.org/uniprotkb/search?query=(gene_exact:{gene_name})+AND+(organism_id:9606)&format=tsv&fields=cc_subcellular_location&size=1"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            if len(lines) > 1:
                return lines[1]
    except Exception as e:
        print(f"  [Warning] Failed to fetch data for {gene_name}: {e}")

    time.sleep(0.2)
    return "Unknown"


def classify_location(loc_string):
    loc_string = str(loc_string).lower()
    if 'unknown' in loc_string or loc_string.strip() == '':
        return 'Unknown'
    elif 'membrane' in loc_string or 'cell surface' in loc_string:
        return 'Membrane (Surface)'
    elif 'secreted' in loc_string or 'extracellular' in loc_string:
        return 'Secreted'
    elif 'nucleus' in loc_string:
        return 'Nucleus'
    elif 'mitochondrion' in loc_string or 'mitochondrial' in loc_string:
        return 'Mitochondrion'
    else:
        return 'Cytoplasm / Other Intracellular'


def classify_therapy(row):
    if row.get('Validation_Score', 0) >= 2 and row.get('Homology_Passed', False) == True:
        if row['Simplified_Location'] == 'Membrane (Surface)':
            return 'Advanced Immunotherapy (CAR-T / BiTEs)'
        else:
            return 'HLA-I Restricted Vaccine Target'
    return 'Low Confidence / Discard'


# ==========================================================
# FUNCIONES DE PLOTEO
# ==========================================================
def plot_individual_evidence(df, column, title, xlabel, filename):
    plt.figure(figsize=(7, 5))
    plot_data = df[column].map({True: 'Detected', False: 'Not Detected'})
    color_map = {'Detected': '#2ca02c', 'Not Detected': '#d62728'}
    ax = sns.countplot(x=plot_data, hue=plot_data, palette=color_map,
                       order=['Detected', 'Not Detected'], legend=False)
    plt.title(title, fontweight='bold', pad=15, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Number of Peptides (Top 100)', fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"predictions_discovery/{filename}", dpi=300)
    plt.close()


def plot_homology_filter(df, column, filename):
    plt.figure(figsize=(8, 6))
    counts = df[column].value_counts()
    labels = []
    sizes = []
    colors = []
    if True in counts:
        labels.append('Passed (Tumor-Specific)')
        sizes.append(counts[True])
        colors.append('#2ca02c')
    if False in counts:
        labels.append('Failed (Self-Peptide)')
        sizes.append(counts[False])
        colors.append('#d62728')

    wedges, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                       startangle=90, pctdistance=0.85,
                                       wedgeprops=dict(width=0.3, edgecolor='w'))
    plt.setp(autotexts, size=12, weight="bold", color="white")
    plt.setp(texts, size=12, weight="bold")
    plt.title('Validation 6: Central Tolerance (Homology Filter)', fontweight='bold', pad=20, fontsize=16)
    plt.text(0, 0, f"Total\n{len(df)}", ha='center', va='center', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"predictions_discovery/{filename}", dpi=300)
    plt.close()


def plot_subcellular_distribution(df, filename):
    plt.figure(figsize=(9, 7))
    plot_data = df[df['Simplified_Location'] != 'Unknown']['Simplified_Location']

    if len(plot_data) > 0:
        counts = plot_data.value_counts()
        colors = sns.color_palette("Set2")[0:len(counts)]

        wedges, texts, autotexts = plt.pie(counts, labels=counts.index, colors=colors, autopct='%1.1f%%',
                                           startangle=140, shadow=False, wedgeprops=dict(edgecolor='w', linewidth=1.5))

        plt.setp(autotexts, size=10, weight="bold", color="white")
        plt.setp(texts, size=11)
        plt.title('Subcellular Distribution of Neoantigen Source Proteins', fontweight='bold', pad=20, fontsize=16)
    else:
        plt.text(0.5, 0.5, "No Subcellular Data Available", ha='center', va='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(f"predictions_discovery/{filename}", dpi=300)
    plt.close()


def plot_therapeutic_map(df, filename):
    plt.figure(figsize=(10, 6))
    high_conf_df = df[df['Therapeutic_Potential'] != 'Low Confidence / Discard']

    if len(high_conf_df) > 0:
        ax = sns.countplot(data=high_conf_df, y='Validation_Score', hue='Therapeutic_Potential',
                           palette={'Advanced Immunotherapy (CAR-T / BiTEs)': '#ff7f0e',
                                    'HLA-I Restricted Vaccine Target': '#1f77b4'})

        plt.title('Therapeutic Opportunity Map (High Confidence Candidates)', fontweight='bold', pad=20, fontsize=16)
        plt.xlabel('Number of Neoantigens', fontsize=12)
        plt.ylabel('Validation Score (Evidence Level)', fontsize=12)
        plt.legend(title='Clinical Application', loc='lower right', frameon=True)

        for p in ax.patches:
            width = p.get_width()
            if width > 0:
                ax.annotate(f'{int(width)}',
                            (width + 0.5, p.get_y() + p.get_height() / 2.),
                            ha='left', va='center', fontweight='bold', fontsize=11)
    else:
        plt.text(0.5, 0.5, "No high-confidence candidates found", ha='center', va='center')

    plt.tight_layout()
    plt.savefig(f"predictions_discovery/{filename}", dpi=300)
    plt.close()


def run_multiomic_validation():
    print("=" * 70)
    print("🧬 STARTING FULL MULTI-OMIC VALIDATION & CLINICAL ANNOTATION")
    print("=" * 70)

    # Rutas a los ficheros
    preds_path = "predictions_discovery/Top_100_Vaccine_Candidates.csv"
    rna_path = "data/raw/mRNA_RSEM_UQ_log2_Tumor.cct"
    prot_path = "data/raw/proteomics_gene_level_MD_abundance_tumor.cct"
    maf_path = "data/raw/Mutation_results.maf"
    tcga_rna_path = "data/raw/PAAD.rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.data.txt"
    tcga_maf_path = "data/raw/TCGA_PAAD_Mutation.maf"
    fasta_path = "data/raw/sequence_proteins.fasta"

    # ==========================================================
    # 1. LOAD PREDICTIONS
    # ==========================================================
    try:
        df_preds = pd.read_csv(preds_path)
        print(f"[INFO] Predictions loaded: {len(df_preds)} candidate peptides.")
    except FileNotFoundError:
        print(f"[ERROR] File not found: {preds_path}.")
        return

    # ==========================================================
    # 1.5 ANOTACIÓN CLÍNICA: UNIPROT SUBCELLULAR LOCATION
    # ==========================================================
    print("[INFO] Fetching Subcellular Location from UniProtKB using Gene Names...")
    if 'gene' in df_preds.columns:
        unique_genes = df_preds['gene'].dropna().unique()
        loc_dict = {}
        for i, gene in enumerate(unique_genes):
            if i % 10 == 0:
                print(f"  -> Processing {i}/{len(unique_genes)} genes...")
            loc_dict[gene] = get_uniprot_location_by_gene(gene)

        df_preds['Raw_Location'] = df_preds['gene'].map(loc_dict)
        df_preds['Simplified_Location'] = df_preds['Raw_Location'].apply(classify_location)
        print("[INFO] Subcellular location mapping complete.")
    else:
        print(f"[WARNING] Column 'gene' not found. Skipping UniProt annotation.")
        df_preds['Simplified_Location'] = 'Unknown'

    # ==========================================================
    # 2. VALIDATION: AUTOTOLERANCE (HOMOLOGY FILTER)
    # ==========================================================
    print("[INFO] Cross-referencing with healthy human proteome (Homology Filter)...")
    try:
        healthy_proteome = []
        for record in SeqIO.parse(fasta_path, "fasta"):
            healthy_proteome.append(str(record.seq))
        healthy_proteome_str = "|".join(healthy_proteome)

        peptide_col_name = 'Peptide'
        if peptide_col_name in df_preds.columns:
            df_preds['Homology_Passed'] = df_preds[peptide_col_name].apply(lambda x: x not in healthy_proteome_str)
            passed_count = df_preds['Homology_Passed'].sum()
            print(f"[INFO] Homology check complete. {passed_count}/{len(df_preds)} peptides are truly tumor-specific.")
        else:
            print(f"[WARNING] Column '{peptide_col_name}' not found.")
            df_preds['Homology_Passed'] = True
    except Exception as e:
        print(f"[ERROR] Failed to process FASTA: {e}")
        df_preds['Homology_Passed'] = True

    # ==========================================================
    # 3. VALIDATION: CPTAC COHORT (RNA, Protein, Mutation)
    # ==========================================================
    print("[INFO] Cross-referencing with original CPTAC multi-omic data...")
    try:
        df_rna = pd.read_csv(rna_path, sep='\t', index_col=0)
        rna_means = df_rna.mean(axis=1).reset_index()
        rna_means.columns = ['gene', 'RNA_Expression_Mean']
        df_preds = pd.merge(df_preds, rna_means, on='gene', how='left')
        df_preds['RNA_Detected'] = df_preds['RNA_Expression_Mean'] > 0
    except:
        df_preds['RNA_Detected'] = False

    try:
        df_prot = pd.read_csv(prot_path, sep='\t', index_col=0)
        prot_means = df_prot.mean(axis=1).reset_index()
        prot_means.columns = ['gene', 'Protein_Abundance_Mean']
        df_preds = pd.merge(df_preds, prot_means, on='gene', how='left')
        df_preds['Protein_Detected'] = df_preds['Protein_Abundance_Mean'].notna()
    except:
        df_preds['Protein_Detected'] = False

    # ---> AQUÍ ESTÁ EL CAMBIO PARA OBTENER LOS PACIENTES DE CPTAC <---
    try:
        df_maf = pd.read_csv(maf_path, sep='\t', comment='#', low_memory=False)

        # 1. Contamos el número de pacientes únicos
        recurrence = df_maf.groupby('Hugo_Symbol')['Tumor_Sample_Barcode'].nunique().reset_index()
        recurrence.columns = ['gene', 'Mutation_Patient_Count']

        # 2. Agrupamos los IDs de los pacientes separados por comas
        patients_list = df_maf.groupby('Hugo_Symbol')['Tumor_Sample_Barcode'].unique().apply(
            lambda x: ', '.join(x)).reset_index()
        patients_list.columns = ['gene', 'CPTAC_Patients_List']

        # Hacemos merge de ambas columnas a nuestro DataFrame principal
        df_preds = pd.merge(df_preds, recurrence, on='gene', how='left')
        df_preds = pd.merge(df_preds, patients_list, on='gene', how='left')

        df_preds['Mutation_Patient_Count'] = df_preds['Mutation_Patient_Count'].fillna(0).astype(int)
        df_preds['Mutation_Detected'] = df_preds['Mutation_Patient_Count'] > 0
    except:
        df_preds['Mutation_Detected'] = False
        df_preds['CPTAC_Patients_List'] = np.nan

    # Scoring principal (Máximo 4)
    df_preds['Validation_Score'] = (
            (df_preds['RNA_Detected']).astype(int) +
            (df_preds['Protein_Detected']).astype(int) +
            (df_preds['Mutation_Detected']).astype(int) +
            (df_preds['Homology_Passed']).astype(int)
    )

    df_preds['Therapeutic_Potential'] = df_preds.apply(classify_therapy, axis=1)
    df_preds = df_preds.sort_values(by=['Validation_Score', 'Homology_Passed'], ascending=[False, False])

    # ==========================================================
    # 4. VALIDATION: TCGA COHORT (RNA, Mutation)
    # ==========================================================
    print("[INFO] Processing independent TCGA cohort data...")
    try:
        df_tcga_rna = pd.read_csv(tcga_rna_path, sep='\t', index_col=0, low_memory=False)
        df_tcga_rna = df_tcga_rna.iloc[1:].apply(pd.to_numeric, errors='coerce')
        df_tcga_rna.index = [str(idx).split('|')[0] for idx in df_tcga_rna.index]
        tcga_rna_means = df_tcga_rna.mean(axis=1).reset_index()
        tcga_rna_means.columns = ['gene', 'TCGA_RNA_Expression_Mean']
        df_preds = pd.merge(df_preds, tcga_rna_means, on='gene', how='left')
        df_preds['TCGA_RNA_Detected'] = df_preds['TCGA_RNA_Expression_Mean'] > 0
    except:
        df_preds['TCGA_RNA_Detected'] = False

    try:
        df_tcga_maf = pd.read_csv(tcga_maf_path, sep='\t', comment='#', low_memory=False, encoding='latin1')

        tcga_rec = df_tcga_maf.groupby('Hugo_Symbol')['Tumor_Sample_Barcode'].nunique().reset_index()
        tcga_rec.columns = ['gene', 'TCGA_Mutation_Patient_Count']

        tcga_patients_list = df_tcga_maf.groupby('Hugo_Symbol')['Tumor_Sample_Barcode'].unique().apply(
            lambda x: ', '.join(x)).reset_index()
        tcga_patients_list.columns = ['gene', 'TCGA_Patients_List']

        df_preds = pd.merge(df_preds, tcga_rec, on='gene', how='left')
        df_preds = pd.merge(df_preds, tcga_patients_list, on='gene', how='left')

        df_preds['TCGA_Mutation_Patient_Count'] = df_preds['TCGA_Mutation_Patient_Count'].fillna(0).astype(int)
        df_preds['TCGA_Mutation_Detected'] = df_preds['TCGA_Mutation_Patient_Count'] > 0
    except:
        df_preds['TCGA_Mutation_Detected'] = False
        df_preds['TCGA_Patients_List'] = np.nan

    # Guardado del fichero maestro FINAL
    df_preds.to_csv("results/reports/predictions_discovery_reports/Top100_Validated_Master_Clinical.csv", index=False)

    # ==========================================================
    # 4.5 GENERATE TARGET LIST FOR CLUSTER (NUEVO)
    # ==========================================================
    print("[INFO] Generating Target List for Cluster Validation...")
    try:
        if 'Peptide' in df_preds.columns:
            # Filtrar solo los neoantígenos detectados en CPTAC
            target_df = df_preds[df_preds['Mutation_Detected'] == True].copy()

            # Eliminar duplicados para tener un diccionario limpio de Péptido - Gen
            target_df = target_df.drop_duplicates(subset=['gene', 'Peptide'])

            # Formatear la tabla final
            cols_to_keep = ['gene', 'Peptide', 'CPTAC_Patients_List']
            target_export = target_df[cols_to_keep].rename(columns={
                'gene': 'Gen',
                'Peptide': 'Peptido_Mutado',
                'CPTAC_Patients_List': 'Pacientes_CPTAC'
            })

            # Guardar el .txt separado por "|" para fácil lectura en bash/terminal
            target_path = "predictions_discovery/Target_List_Cluster.txt"
            target_export.to_csv(target_path, sep='|', index=False)
            print(f"[INFO] Target List saved successfully at {target_path}")
        else:
            print("[WARNING] 'Peptide' column not found. Skipping Target List generation.")
    except Exception as e:
        print(f"[ERROR] Failed to generate Target List: {e}")

    # ==========================================================
    # 5. GENERATE ALL PLOTS
    # ==========================================================
    print("\n[INFO] Generating all validation and clinical plots...")

    try:
        # Figuras de Evidencia Básica
        plot_individual_evidence(df_preds, 'RNA_Detected', 'Validation 1: RNA-Seq Expression', 'Gene Expression Status',
                                 'Evidence_1_RNA.png')
        plot_individual_evidence(df_preds, 'Protein_Detected', 'Validation 2: Proteomics', 'Protein Status',
                                 'Evidence_2_Proteomics.png')
        plot_individual_evidence(df_preds, 'Mutation_Detected', 'Validation 3: MAF', 'Mutation Cohort',
                                 'Evidence_3_Genomics.png')

        # Figura Autotolerancia
        plot_homology_filter(df_preds, 'Homology_Passed', 'Evidence_10_Homology_Filter.png')

        # NUEVAS FIGURAS CLÍNICAS (UniProt)
        plot_subcellular_distribution(df_preds, 'Clinical_1_Subcellular_Distribution.png')
        plot_therapeutic_map(df_preds, 'Clinical_2_Therapeutic_Map.png')

        # Figura Global Score
        plt.figure(figsize=(8, 5))
        ax = sns.countplot(data=df_preds, x='Validation_Score', hue='Validation_Score', palette='magma', legend=False)
        plt.title('Global Summary: Multi-Omic Evidence Level', fontweight='bold', fontsize=16)
        plt.xlabel('Validation Score (0 to 4 validation layers)', fontsize=12)
        plt.ylabel('Number of Peptides', fontsize=12)
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                        va='bottom')
        plt.tight_layout()
        plt.savefig("predictions_discovery/Evidence_4_Global.png", dpi=300)
        plt.close()

        # UpSet Plot
        upset_cols = ['RNA_Detected', 'Protein_Detected', 'Mutation_Detected', 'Homology_Passed', 'TCGA_RNA_Detected']
        upset_names = ['CPTAC RNA', 'CPTAC Protein', 'CPTAC Mutation', 'Safety (No-Self)', 'TCGA RNA']
        df_upset = df_preds[upset_cols].copy()
        df_upset.columns = upset_names
        df_upset = df_upset.set_index(upset_names)
        fig9 = plt.figure(figsize=(10, 6))
        upset = UpSet(df_upset, subset_size='count', show_counts=True, sort_by='cardinality', facecolor='#3182bd',
                      element_size=40)
        upset.plot(fig=fig9)
        plt.suptitle('Intersection of Multi-Omic & Safety Evidence', fontweight='bold', fontsize=14, y=1.05)
        plt.savefig("predictions_discovery/Evidence_9_UpSet_Plot.png", dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"\n[ERROR] Failed to generate figures: {e}")

    print("\n" + "=" * 70)
    print("✅ PROCESS COMPLETED SUCCESSFULLY.")
    print("=" * 70)


if __name__ == "__main__":
    run_multiomic_validation()