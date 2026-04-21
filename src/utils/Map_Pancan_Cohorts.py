import pandas as pd

# 1. Definimos las rutas de los archivos
ruta_dataset = 'data/processed/dataset_pancreas_completo.csv'
ruta_mapeo = 'data/interim/mapeo_pacientes.tsv'

# Por seguridad, te recomiendo guardarlo en un archivo nuevo,
# pero si quieres puedes poner la misma ruta para sobreescribirlo.
ruta_salida = 'data/processed/dataset_pancreas_completo_con_cohorte.csv'

print("Cargando los datos...")

# 2. Cargamos el dataset principal
df_pancreas = pd.read_csv(ruta_dataset)

# 3. Cargamos el mapeo TSV
# Le indicamos sep='\t' porque está separado por tabulaciones.
# Le indicamos header=None y names=[...] porque nuestro TSV no tiene fila de títulos.
df_mapeo = pd.read_csv(ruta_mapeo, sep='\t', header=None, names=['patient', 'cohort_map'])

print("Cruzando la información...")

# 4. Convertimos el mapeo en un diccionario { 'DO221541': 'PACA-CA', ... }
diccionario_mapeo = df_mapeo.set_index('patient')['cohort_map'].to_dict()

# 5. Rellenamos la columna 'cohort' que ya existe en el CSV principal.
# Usamos map() para buscar el paciente y colocar su cohorte.
# .fillna() asegura que si un paciente no está en el TSV, no borremos lo que ya tuviera en la columna.
df_pancreas['cohort'] = df_pancreas['patient'].map(diccionario_mapeo).fillna(df_pancreas['cohort'])

print("Guardando el archivo actualizado...")

# 6. Guardamos el resultado en un nuevo CSV (sin exportar el índice de las filas)
df_pancreas.to_csv(ruta_salida, index=False)

print(f"¡Listo! Tienes el nuevo dataset guardado en: {ruta_salida}")