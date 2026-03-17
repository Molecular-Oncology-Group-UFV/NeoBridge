import pandas as pd
from Bio import SeqIO
import requests
import time

# Rutas de tus archivos
fasta_sano_path = "data/raw/sequence_proteins.fasta"
target_list_path = "data/interim/proteomal/Target_List_Cluster.txt"
csv_maestro_path = "predictions_discovery/Top100_Validated_Master_Clinical.csv"
output_fasta_path = "data/processed/Custom_FragPipe_Proteome.fasta"

print("Descargando el proteoma revisado desde UniProt (Swiss-Prot)...")


def get_uniprot_fasta_canonical(gene_name):
    """Descarga la secuencia FASTA revisada (Swiss-Prot) garantizando la canónica."""
    # ¡CLAVE! Añadimos (reviewed:true) para no recibir isoformas raras o fragmentos
    url = f"https://rest.uniprot.org/uniprotkb/search?query=(gene_exact:{gene_name})+AND+(organism_id:9606)+AND+(reviewed:true)&format=fasta&size=1"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200 and response.text.strip():
            lines = response.text.strip().split('\n')
            header = lines[0]
            secuencia = "".join(lines[1:])
            return header, secuencia
    except Exception as e:
        pass
    return None, None


# 1. Leer tu lista maestra de neoantígenos y el CSV maestro
df_targets = pd.read_csv(target_list_path, sep='|')
try:
    df_maestro = pd.read_csv(csv_maestro_path)
    dict_wt = dict(zip(df_maestro['Peptide'], df_maestro['wt_seq']))
except FileNotFoundError:
    print(f"ADVERTENCIA: No se encuentra {csv_maestro_path}.")
    dict_wt = {}

nuevas_proteinas = []
exitos = 0

print("\nConectando a UniProt para mutar las proteínas completas...")

# Iteramos sobre nuestra lista del clúster
for index, row in df_targets.iterrows():
    gen = row['Gen']
    peptido_mutado = row['Peptido_Mutado']
    peptido_sano = dict_wt.get(peptido_mutado, None)

    if not peptido_sano:
        continue

    print(f"  -> Procesando {gen}...")
    cabecera, seq_original = get_uniprot_fasta_canonical(gen)

    if seq_original:
        if peptido_sano in seq_original:
            seq_mutada = seq_original.replace(peptido_sano, peptido_mutado)
            nueva_cabecera = f">sp|MUT_{gen}_{peptido_mutado}| Proteina mutada"
            nuevas_proteinas.append(f"{nueva_cabecera}\n{seq_mutada}")
            exitos += 1
            print(f"     [OK] Mutación insertada en la proteína completa de {gen}")
        else:
            # Plan B: Si la mutación cae justo en una isoforma, forzamos un contexto para la Tripsina.
            # Le añadimos Lysinas (K) para que la tripsina sepa por dónde cortar.
            print(f"     [AVISO] {peptido_sano} no está en la canónica. Creando secuencia sintética digerible...")
            nueva_cabecera = f">sp|MUT_SINTETICA_{gen}_{peptido_mutado}| Proteina mutada sintetica"
            seq_sintetica = f"MKKKKK{peptido_mutado}KKKKK"
            nuevas_proteinas.append(f"{nueva_cabecera}\n{seq_sintetica}")
            exitos += 1
    else:
        print(f"     [ERROR] No se pudo descargar {gen}.")

    time.sleep(0.3)

# 3. Guardar el archivo FASTA final
print("\nGuardando archivo FASTA final para FragPipe...")
with open(output_fasta_path, "w") as f_out:
    # Escribimos el proteoma original que tenías
    with open(fasta_sano_path, "r") as f_in:
        for line in f_in:
            f_out.write(line)

    f_out.write('\n')

    # Añadimos las nuevas proteínas mutadas completas
    for prot in nuevas_proteinas:
        f_out.write(prot + "\n")

print(f"\n¡Éxito! Diccionario creado: {output_fasta_path}")
print(f"Se han añadido {exitos} proteínas al proteoma para buscar en FragPipe.")