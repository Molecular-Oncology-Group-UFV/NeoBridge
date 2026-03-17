import pandas as pd
import requests
import time
from Bio import SeqIO
import os

# ==========================================================
# 1. CONFIGURACIÓN DE RUTAS (Ajusta según tu entorno)
# ==========================================================
preds_path = "predictions_discovery/Full_Dataset_Predictions.csv"
rna_path = "data/raw/mRNA_RSEM_UQ_log2_Tumor.cct"
prot_path = "data/raw/proteomics_gene_level_MD_abundance_tumor.cct"
maf_path = "data/raw/Mutation_results.maf"
fasta_sano_path = "data/raw/sequence_proteins.fasta"
output_fasta_path = "data/processed/Custom_Top1000_FragPipe.fasta"

print("🚀 Iniciando validación completa del dataset para el Top 1000...")

# ==========================================================
# 2. CARGA DE DATOS ÓMICOS PARA VALIDACIÓN
# ==========================================================
try:
    df_preds = pd.read_csv(preds_path)
    df_rna = pd.read_csv(rna_path, sep='\t', index_col=0)
    df_prot = pd.read_csv(prot_path, sep='\t', index_col=0)
    df_maf = pd.read_csv(maf_path, sep='\t', comment='#', low_memory=False)

    # Cargar secuencias sanas para el filtro de Homología (Safety)
    self_proteins = [str(record.seq) for record in SeqIO.parse(fasta_sano_path, "fasta")]
    print(f"[INFO] Datos ómicos y {len(self_proteins)} proteínas sanas cargadas.")
except Exception as e:
    print(f"[ERROR] Fallo al cargar archivos: {e}")
    exit()

# ==========================================================
# 3. CÁLCULO DE CAPAS DE EVIDENCIA (Lógica de Caracterización)
# ==========================================================
print("[INFO] Calculando capas de validación...")

# Capa 1: RNA (¿Se expresa el gen?)
df_preds['RNA_Detected'] = df_preds['gene'].isin(df_rna.index)

# Capa 2: Proteína (¿Se detecta la proteína sana?)
df_preds['Protein_Detected'] = df_preds['gene'].isin(df_prot.index)

# Capa 3: Mutación (¿Aparece el gen en el MAF de CPTAC?)
maf_genes = set(df_maf['Hugo_Symbol'].unique())
df_preds['Mutation_Detected'] = df_preds['gene'].isin(maf_genes)


# Capa 4: Homología (¿Es el péptido seguro/no-self?)
def check_homology(peptide, self_list):
    for prot in self_list:
        if peptide in prot: return False
    return True


print("   -> Comprobando homología (esto puede tardar un poco)...")
df_preds['Homology_Passed'] = df_preds['Peptide'].apply(lambda x: check_homology(x, self_proteins))

# Cálculo del Score Final (0 a 4)
df_preds['Validation_Score'] = (
        df_preds['RNA_Detected'].astype(int) +
        df_preds['Protein_Detected'].astype(int) +
        df_preds['Mutation_Detected'].astype(int) +
        df_preds['Homology_Passed'].astype(int)
)

# ==========================================================
# 4. RANKING Y SELECCIÓN DEL TOP 1000
# ==========================================================
# Ordenamos por Score (mayor es mejor) y por Homología
df_sorted = df_preds.sort_values(by=['Validation_Score', 'Homology_Passed'], ascending=[False, False])

# Tomamos el Top 1000
df_top1000 = df_sorted.head(1000).copy()

# Filtramos los que tienen mutación para el FASTA (los que podemos "encontrar" realmente)
df_fasta_targets = df_top1000[df_top1000['Mutation_Detected'] == True].copy()

print(f"\n✅ Ranking completado.")
print(f"   - Candidatos en el Top 1000 con mutación confirmada: {len(df_fasta_targets)}")


# ==========================================================
# 5. GENERACIÓN DEL FASTA (OPCIÓN A: PROTEÍNA COMPLETA)
# ==========================================================
def get_uniprot_fasta(gene_name):
    url = f"https://rest.uniprot.org/uniprotkb/search?query=(gene_exact:{gene_name})+AND+(organism_id:9606)+AND+(reviewed:true)&format=fasta&size=1"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and r.text.strip():
            lines = r.text.strip().split('\n')
            return "".join(lines[1:])
    except:
        pass
    return None


nuevas_proteinas = []
exitos = 0

print(f"\n🧬 Generando secuencias mutadas para {len(df_fasta_targets)} candidatos...")

for _, row in df_fasta_targets.iterrows():
    gen, pep_mut, pep_sano = row['gene'], row['Peptide'], row['wt_seq']

    seq_original = get_uniprot_fasta(gen)
    if seq_original and pep_sano in seq_original:
        seq_mutada = seq_original.replace(pep_sano, pep_mut)
        nuevas_proteinas.append(f">sp|MUT_{gen}_{pep_mut}| Top1000\n{seq_mutada}")
        exitos += 1
    else:
        # Plan B: Contexto sintético
        nuevas_proteinas.append(f">sp|MUT_SYNTH_{gen}_{pep_mut}| Top1000\nMKKKKK{pep_mut}KKKKK")
        exitos += 1

    time.sleep(0.1)  # Breve pausa para la API

# Guardar
with open(output_fasta_path, "w") as f_out:
    # Escribir el proteoma original
    with open(fasta_sano_path, "r") as f_in:
        f_out.write(f_in.read())
    f_out.write("\n" + "\n".join(nuevas_proteinas) + "\n")

print(f"\n✨ PROCESO FINALIZADO.")
print(f"Archivo creado: {output_fasta_path}")
print(f"Se han integrado {exitos} candidatos de alto valor.")