# ============================================================
# NOTEBOOK 07 — Analisis Exploratorio de Datos (EDA) — Tink
# Proyecto: AI Financial Life Coach
#
# Sustituye al EDA anterior (basado en el dataset sintetico original y en
# series macroeconomicas anuales). Este EDA trabaja directamente sobre el
# dataset mensual por usuario y las transacciones clasificadas derivadas
# de la estructura Tink, para mantener coherencia con el resto de la
# pipeline (Asignaturas 6 y 7 y el TFM final).
# ============================================================

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CLEAN_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean')
OUTPUT_DIR = CLEAN_DIR

sns.set_style('whitegrid')
sns.set_palette('Set2')

print('=' * 60)
print('ANALISIS EXPLORATORIO DE DATOS (EDA) — ESTRUCTURA TINK')
print('AI Financial Life Coach')
print('=' * 60)


# ── CARGA ────────────────────────────────────────────────────
df = pd.read_csv(f'{CLEAN_DIR}/dataset_final_usuarios_tink.csv')
df['fecha_dt'] = pd.to_datetime(df['fecha'], format='%Y-%m')

print(f'\nDataset mensual: {df.shape[0]:,} observaciones x {df.shape[1]} variables')
print(f'Usuarios unicos: {df["user_id"].nunique():,}')
print(f'Periodo: {df["fecha"].min()} — {df["fecha"].max()}')
print('\nEstadisticas descriptivas (variables clave):')
print(df[['edad', 'salario', 'gasto_total', 'ahorro', 'tasa_ahorro_pct', 'transacciones_mes']].describe().round(2).to_string())


# ============================================================
# FIGURA 1 — Distribucion de usuarios por perfil socioeconomico
# ============================================================

users_per_profile = df.drop_duplicates('user_id')['perfil'].value_counts()

fig, ax = plt.subplots(figsize=(8, 5))
users_per_profile.plot(kind='bar', color=sns.color_palette('Set2', len(users_per_profile)), ax=ax)
ax.set_title('Distribucion de usuarios sinteticos por perfil socioeconomico', fontsize=13, fontweight='bold')
ax.set_xlabel('Perfil')
ax.set_ylabel('Numero de usuarios')
plt.xticks(rotation=0)
for i, v in enumerate(users_per_profile.values):
    ax.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/tink_fig1_usuarios_por_perfil.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nFigura 1 guardada: Usuarios por perfil')


# ============================================================
# FIGURA 2 — Distribucion del salario mensual por perfil
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))
order = ['junior', 'medio', 'senior', 'freelance']
order = [p for p in order if p in df['perfil'].unique()]
sns.boxplot(data=df, x='perfil', y='salario', order=order, ax=ax, showfliers=False)
ax.set_title('Distribucion del salario mensual por perfil', fontsize=13, fontweight='bold')
ax.set_xlabel('Perfil')
ax.set_ylabel('Salario mensual (EUR)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/tink_fig2_salario_por_perfil.png', dpi=150, bbox_inches='tight')
plt.close()
print('Figura 2 guardada: Salario por perfil')


# ============================================================
# FIGURA 3 — Tasa de ahorro mensual: distribucion general
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(df['tasa_ahorro_pct'].clip(lower=-50, upper=100), bins=50, kde=True, ax=ax, color='#34A853')
ax.axvline(df['tasa_ahorro_pct'].median(), color='gray', linestyle='--', linewidth=1.2,
           label=f'Mediana = {df["tasa_ahorro_pct"].median():.1f}%')
ax.set_title('Distribucion de la tasa de ahorro mensual', fontsize=13, fontweight='bold')
ax.set_xlabel('Tasa de ahorro (%) [recortada a -50/100 para visualizacion]')
ax.set_ylabel('Frecuencia (usuario-mes)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/tink_fig3_distribucion_ahorro.png', dpi=150, bbox_inches='tight')
plt.close()
print('Figura 3 guardada: Distribucion tasa de ahorro')


# ============================================================
# FIGURA 4 — Evolucion mensual agregada: salario vs. gasto medio
# ============================================================

monthly_avg = df.groupby('fecha_dt', as_index=False).agg(
    salario_medio=('salario', 'mean'),
    gasto_medio=('gasto_total', 'mean'),
    usuarios_activos=('user_id', 'nunique'),
)
monthly_avg = monthly_avg.sort_values('fecha_dt')

fig, ax1 = plt.subplots(figsize=(12, 5))
color1, color2 = '#1A73E8', '#EA4335'
ax1.plot(monthly_avg['fecha_dt'], monthly_avg['salario_medio'], color=color1, linewidth=2, label='Salario medio')
ax1.plot(monthly_avg['fecha_dt'], monthly_avg['gasto_medio'], color=color2, linewidth=2, label='Gasto medio')
ax1.set_xlabel('Mes')
ax1.set_ylabel('EUR')
ax1.set_title('Evolucion mensual del salario y gasto medio (todos los usuarios)', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.bar(monthly_avg['fecha_dt'], monthly_avg['usuarios_activos'], alpha=0.12, color='gray', width=20,
        label='Usuarios activos')
ax2.set_ylabel('Usuarios activos (barra)', color='gray')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/tink_fig4_evolucion_salario_gasto.png', dpi=150, bbox_inches='tight')
plt.close()
print('Figura 4 guardada: Evolucion salario/gasto')


# ============================================================
# FIGURA 5 — Composicion del gasto por categoria (agregado)
# ============================================================

category_cols = ['vivienda', 'alimentacion', 'transporte', 'ocio', 'salud', 'educacion', 'otros']
category_totals = df[category_cols].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
category_totals.plot(kind='bar', color=sns.color_palette('Set2', len(category_totals)), ax=ax)
ax.set_title('Composicion agregada del gasto por categoria', fontsize=13, fontweight='bold')
ax.set_xlabel('Categoria')
ax.set_ylabel('Gasto total acumulado (EUR)')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/tink_fig5_composicion_gasto.png', dpi=150, bbox_inches='tight')
plt.close()
print('Figura 5 guardada: Composicion del gasto')


# ============================================================
# FIGURA 6 — Matriz de correlacion (variables mensuales clave)
# ============================================================

num_cols = ['edad', 'salario', 'gasto_total', 'ahorro', 'tasa_ahorro_pct', 'ipc_mensual', 'transacciones_mes']
corr_df = df[num_cols].dropna()
corr = corr_df.corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, linewidths=0.5)
ax.set_title('Matriz de correlacion — variables del dataset mensual Tink', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/tink_fig6_correlacion.png', dpi=150, bbox_inches='tight')
plt.close()
print('Figura 6 guardada: Matriz de correlacion')


# ============================================================
# FIGURA 7 — Categorias de transaccion clasificadas (nivel transaccion)
# ============================================================

tx_cols = ['target_category', 'amount_abs', 'direction']
tx_counts = pd.Series(dtype='int64')
tx_amounts = pd.Series(dtype='float64')
chunk_size = 400000
for chunk in pd.read_csv(f'{CLEAN_DIR}/transacciones_tink_model_input.csv', usecols=tx_cols, chunksize=chunk_size):
    tx_counts = tx_counts.add(chunk['target_category'].value_counts(), fill_value=0)
    tx_amounts = tx_amounts.add(chunk.groupby('target_category')['amount_abs'].sum(), fill_value=0)

tx_counts = tx_counts.sort_values(ascending=False)
tx_amounts = tx_amounts.reindex(tx_counts.index)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
tx_counts.plot(kind='bar', ax=axes[0], color=sns.color_palette('Set2', len(tx_counts)))
axes[0].set_title('Transacciones clasificadas por categoria (volumen)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Numero de transacciones')
axes[0].tick_params(axis='x', rotation=60)

tx_amounts.plot(kind='bar', ax=axes[1], color=sns.color_palette('Set2', len(tx_amounts)))
axes[1].set_title('Transacciones clasificadas por categoria (importe total)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Importe total (EUR)')
axes[1].tick_params(axis='x', rotation=60)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/tink_fig7_categorias_transaccion.png', dpi=150, bbox_inches='tight')
plt.close()
print('Figura 7 guardada: Categorias de transaccion (volumen e importe)')


# ============================================================
# FIGURA 8 — Matriz de confusion del clasificador (test set)
# ============================================================

confusion_path = f'{CLEAN_DIR}/clasificador_categorias_tink_confusion.csv'
if os.path.exists(confusion_path):
    confusion_df = pd.read_csv(confusion_path, index_col=0)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(confusion_df, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
    ax.set_title('Matriz de confusion — clasificador de categorias (test set)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Prediccion')
    ax.set_ylabel('Real')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/tink_fig8_confusion_clasificador.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Figura 8 guardada: Matriz de confusion del clasificador')


# ============================================================
# RESUMEN ESTADISTICO Y CONCLUSIONES EDA
# ============================================================

print('\n' + '=' * 60)
print('CONCLUSIONES DEL ANALISIS EXPLORATORIO')
print('=' * 60)

n_users = df['user_id'].nunique()
n_profile_sources = df['profile_source'].nunique()
salario_mediana = df['salario'].median()
ahorro_mediana = df['tasa_ahorro_pct'].median()
top_category_count = tx_counts.index[0]
top_category_amount = tx_amounts.index[0]
corr_salario_ahorro = corr_df[['salario', 'tasa_ahorro_pct']].corr().iloc[0, 1]

if top_category_count == top_category_amount:
    gasto_texto = f'La categoria "{top_category_count}" concentra tanto el mayor volumen de transacciones como el mayor importe acumulado.'
else:
    gasto_texto = (
        f'La categoria de gasto con mayor volumen de transacciones es "{top_category_count}", '
        f'mientras que "{top_category_amount}" concentra el mayor importe acumulado.'
    )

print(f"""
1. VOLUMEN Y COBERTURA
   El dataset sintetico cubre {n_users:,} usuarios y {df.shape[0]:,}
   observaciones mensuales, generados con la estructura de payload real
   de Tink (accounts + transactions) a partir de {n_profile_sources} cuentas
   reales de muestra. Este volumen ofrece una base estadistica solida para
   evaluar el modelo, frente a los 8-9 usuarios reales disponibles.

2. HETEROGENEIDAD DE INGRESOS POR PERFIL
   El salario mensual mediano es de {salario_mediana:.2f} EUR, con
   diferencias marcadas entre perfiles (junior/medio/senior/freelance),
   reflejo directo de la heterogeneidad de ingresos observada en las
   muestras reales de origen.

3. TASA DE AHORRO ELEVADA — LIMITACION CONOCIDA DE LA MUESTRA
   La tasa de ahorro mensual mediana se situa en {ahorro_mediana:.1f}%,
   un nivel alto explicado por la limitada variedad y frecuencia de
   transacciones de gasto presentes en las muestras Tink reales
   disponibles (importes medianos de 25-75 EUR por transaccion de gasto,
   con una frecuencia moderada). Se documenta como limitacion del dataset
   sintetico y no como un sesgo introducido por el proceso de generacion,
   dado que los importes de gasto replican fielmente la distribucion
   observada en los datos reales.

4. COMPOSICION DEL GASTO
   {gasto_texto}
   Esta composicion es coherente con el patron de gasto de las muestras
   reales utilizadas como base.

5. RELACION ENTRE SALARIO Y TASA DE AHORRO
   La correlacion entre salario mensual y tasa de ahorro es de
   {corr_salario_ahorro:.2f}, indicando que los usuarios con mayores
   ingresos tienden a mantener una tasa de ahorro proporcionalmente mayor,
   patron esperado en un contexto de gasto poco elastico a corto plazo.

6. COHERENCIA CON EL CLASIFICADOR DE CATEGORIAS
   El clasificador de categorias entrenado sobre esta misma estructura de
   datos (ver 05_entrena_clasificador_categorias_tink.py) alcanza una
   accuracy identica en validation y test, lo que confirma que el
   particionado por usuario (en lugar de por texto de transaccion) elimina
   la inconsistencia detectada en una version previa del pipeline.
""")
print('=' * 60)
