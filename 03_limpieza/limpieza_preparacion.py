# ============================================================
# NOTEBOOK 03 — Limpieza y Preparación del Dataset
# Proyecto: AI Financial Life Coach
# ============================================================

import pandas as pd
import numpy as np
import os

RAW_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
CLEAN_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean')
os.makedirs(CLEAN_DIR, exist_ok=True)

print("=" * 60)
print("LIMPIEZA Y PREPARACIÓN DEL DATASET")
print("=" * 60)


# ============================================================
# CARGA DE DATASETS RAW
# ============================================================

df_ahorro    = pd.read_csv(f"{RAW_DIR}/eurostat_tasa_ahorro.csv")
df_desempleo = pd.read_csv(f"{RAW_DIR}/eurostat_desempleo_25_34.csv")
df_ipc       = pd.read_csv(f"{RAW_DIR}/eurostat_ipc.csv")
df_euribor   = pd.read_csv(f"{RAW_DIR}/scraping_euribor_wiki.csv")
df_morosidad = pd.read_csv(f"{RAW_DIR}/scraping_morosidad_bde.csv")

print(f"\nDatasets cargados:")
print(f"  Tasa ahorro:    {df_ahorro.shape}")
print(f"  Desempleo:      {df_desempleo.shape}")
print(f"  IPC:            {df_ipc.shape}")
print(f"  Euríbor:        {df_euribor.shape}")
print(f"  Morosidad:      {df_morosidad.shape}")


# ============================================================
# 1. ELIMINACIÓN DE DUPLICADOS
# ============================================================

print("\n[1/5] Eliminación de duplicados...")

for name, df in [('ahorro', df_ahorro), ('desempleo', df_desempleo),
                  ('ipc', df_ipc), ('euribor', df_euribor),
                  ('morosidad', df_morosidad)]:
    n_dup = df.duplicated().sum()
    print(f"   {name}: {n_dup} duplicados encontrados")
    if n_dup > 0:
        df.drop_duplicates(inplace=True)

print("   ✅ Duplicados eliminados")


# ============================================================
# 2. TRATAMIENTO DE VALORES FALTANTES
# ============================================================

print("\n[2/5] Tratamiento de valores faltantes...")

for name, df in [('ahorro', df_ahorro), ('desempleo', df_desempleo),
                  ('ipc', df_ipc), ('euribor', df_euribor),
                  ('morosidad', df_morosidad)]:
    nulls = df.isnull().sum().sum()
    print(f"   {name}: {nulls} valores faltantes")

# Imputar si hubiera nulos en variables numéricas
for df in [df_ahorro, df_desempleo, df_ipc, df_euribor, df_morosidad]:
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].interpolate(), inplace=True)
            print(f"   ⚠ Imputados valores faltantes en {col} por interpolación")

print("   ✅ Valores faltantes tratados")


# ============================================================
# 3. ESTANDARIZACIÓN DE FORMATOS Y VARIABLES
# ============================================================

print("\n[3/5] Estandarización de formatos...")

# Dataset ahorro: extraer año y trimestre del campo periodo
df_ahorro['anio']       = df_ahorro['periodo'].str[:4].astype(int)
df_ahorro['trimestre']  = df_ahorro['periodo'].str[5:7]
df_ahorro['tasa_ahorro'] = pd.to_numeric(df_ahorro['tasa_ahorro'], errors='coerce')

# Dataset IPC: extraer año y mes
df_ipc['anio'] = df_ipc['periodo'].str[:4].astype(int)
df_ipc['mes']  = df_ipc['periodo'].str[5:7].astype(int)
df_ipc['ipc']  = pd.to_numeric(df_ipc['ipc'], errors='coerce')

# Dataset euribor: estandarizar columna según lo que haya llegado
if 'euribor_12m_media' in df_euribor.columns:
    df_euribor.rename(columns={'euribor_12m_media': 'euribor_pct'}, inplace=True)
elif 'Euríbor 12 meses' in df_euribor.columns:
    df_euribor.rename(columns={'Euríbor 12 meses': 'euribor_pct'}, inplace=True)
else:
    num_col = df_euribor.select_dtypes(include=[np.number]).columns[0]
    df_euribor.rename(columns={num_col: 'euribor_pct'}, inplace=True)

df_euribor['anio'] = pd.to_numeric(df_euribor['anio'], errors='coerce').astype('Int64')

# Dataset morosidad: asegurar tipos correctos
df_morosidad['tasa_morosidad_pct']      = pd.to_numeric(df_morosidad['tasa_morosidad_pct'], errors='coerce')
df_morosidad['credito_dudoso_MM_eur']   = pd.to_numeric(df_morosidad['credito_dudoso_MM_eur'], errors='coerce')

print("   ✅ Formatos estandarizados")


# ============================================================
# 4. CONSTRUCCIÓN DEL DATASET ANUAL CONSOLIDADO
# ============================================================

print("\n[4/5] Construcción del dataset consolidado anual...")

# Agregar ahorro por año (media de trimestres)
ahorro_anual = (df_ahorro.groupby('anio')['tasa_ahorro']
                .mean().round(2).reset_index()
                .rename(columns={'tasa_ahorro': 'tasa_ahorro_pct'}))

# IPC medio anual
ipc_anual = (df_ipc.groupby('anio')['ipc']
             .mean().round(2).reset_index()
             .rename(columns={'ipc': 'ipc_variacion_anual_pct'}))

# Euríbor ya es anual
euribor_anual = df_euribor[['anio', 'euribor_pct']].copy()
euribor_anual['anio'] = euribor_anual['anio'].astype(int)

# Desempleo ya es anual
desempleo_anual = df_desempleo[['anio', 'tasa_desempleo']].copy()

# Morosidad: media anual
morosidad_anual = (df_morosidad.groupby('anio')['tasa_morosidad_pct']
                   .mean().round(2).reset_index())

# Merge progresivo
df_final = ahorro_anual.copy()
df_final = df_final.merge(desempleo_anual,  on='anio', how='outer')
df_final = df_final.merge(ipc_anual,        on='anio', how='outer')
df_final = df_final.merge(euribor_anual,    on='anio', how='outer')
df_final = df_final.merge(morosidad_anual,  on='anio', how='outer')

# Filtrar años con datos relevantes
df_final = df_final[df_final['anio'] >= 2008].sort_values('anio').reset_index(drop=True)

# Añadir metadatos
df_final['pais']   = 'España'
df_final['fuente'] = 'Eurostat + Banco de España'

print(f"   ✅ Dataset consolidado: {df_final.shape[0]} filas x {df_final.shape[1]} columnas")
print(f"\n   Columnas: {list(df_final.columns)}")
print(f"\n   Primeros registros:")
print(df_final.head(5).to_string(index=False))


# ============================================================
# 5. VALIDACIÓN FINAL Y GUARDADO
# ============================================================

print("\n[5/5] Validación final y guardado...")

# Verificar rango de valores
print("\n   Estadísticas descriptivas del dataset final:")
print(df_final.describe().round(2).to_string())

# Verificar valores negativos en tasas (admisibles solo en euribor)
for col in ['tasa_ahorro_pct', 'tasa_desempleo', 'tasa_morosidad_pct']:
    if col in df_final.columns:
        neg = (df_final[col] < 0).sum()
        if neg > 0:
            print(f"   ⚠ {col}: {neg} valores negativos (revisar)")

# Guardar
df_final.to_csv(f"{CLEAN_DIR}/dataset_final.csv", index=False)

# Guardar también los datasets individuales limpios
df_ahorro.to_csv(f"{CLEAN_DIR}/ahorro_clean.csv",       index=False)
df_morosidad.to_csv(f"{CLEAN_DIR}/morosidad_clean.csv", index=False)
df_euribor.to_csv(f"{CLEAN_DIR}/euribor_clean.csv",     index=False)

print(f"\n   ✅ Archivos guardados en: {CLEAN_DIR}")
print("=" * 60)
print("LIMPIEZA COMPLETADA")
print("=" * 60)
print(f"\nDataset final: {df_final.shape[0]} observaciones x {df_final.shape[1]} variables")
print(f"Periodo: {df_final['anio'].min()} — {df_final['anio'].max()}")
print(f"Variables financieras: tasa_ahorro_pct, tasa_desempleo, ")
print(f"                       ipc_variacion_anual_pct, euribor_pct, tasa_morosidad_pct")
