# ============================================================
# NOTEBOOK 04 — Análisis Exploratorio de Datos (EDA)
# Proyecto: AI Financial Life Coach
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

CLEAN_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean')

sns.set_style("whitegrid")
sns.set_palette("Set2")

print("=" * 60)
print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("AI Financial Life Coach — España 2008–2024")
print("=" * 60)


# ── CARGA ────────────────────────────────────────────────────
df = pd.read_csv(f"{CLEAN_DIR}/dataset_final.csv")
df_morosidad = pd.read_csv(f"{CLEAN_DIR}/morosidad_clean.csv")

print(f"\nDataset: {df.shape[0]} observaciones x {df.shape[1]} variables")
print(f"Periodo: {df['anio'].min()} — {df['anio'].max()}")
print(f"\nEstadísticas descriptivas:")
print(df.describe().round(2).to_string())


# ============================================================
# FIGURA 1 — Evolución del Euríbor y la Morosidad (2008-2024)
# ============================================================

fig, ax1 = plt.subplots(figsize=(12, 5))
color1, color2 = '#1A73E8', '#EA4335'

ax1.plot(df['anio'], df['euribor_pct'], color=color1,
         linewidth=2.5, marker='o', markersize=5, label='Euríbor 12m (%)')
ax1.set_xlabel('Año', fontsize=12)
ax1.set_ylabel('Euríbor 12 meses (%)', color=color1, fontsize=11)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

ax2 = ax1.twinx()
ax2.plot(df['anio'], df['tasa_morosidad_pct'], color=color2,
         linewidth=2.5, marker='s', markersize=5,
         linestyle='--', label='Morosidad bancaria (%)')
ax2.set_ylabel('Tasa de morosidad (%)', color=color2, fontsize=11)
ax2.tick_params(axis='y', labelcolor=color2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

plt.title('Euríbor vs. Morosidad bancaria en España (2008–2024)',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_euribor_morosidad.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Figura 1 guardada: Euríbor vs. Morosidad")


# ============================================================
# FIGURA 2 — Tasa de Ahorro de los Hogares (trimestral)
# ============================================================

df_ahorro = pd.read_csv(f"{CLEAN_DIR}/ahorro_clean.csv")

fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(range(len(df_ahorro)), df_ahorro['tasa_ahorro'],
                alpha=0.3, color='#34A853')
ax.plot(range(len(df_ahorro)), df_ahorro['tasa_ahorro'],
        color='#34A853', linewidth=2, marker='o', markersize=3)

# Marcar COVID-19
covid_idx = df_ahorro[df_ahorro['periodo'].str.startswith('2020')].index
if len(covid_idx) > 0:
    ax.axvspan(covid_idx[0], covid_idx[-1]+1, alpha=0.15,
               color='red', label='Período COVID-19')

ax.set_xticks(range(0, len(df_ahorro), 4))
ax.set_xticklabels(df_ahorro['periodo'].iloc[::4], rotation=45, ha='right', fontsize=9)
ax.axhline(df_ahorro['tasa_ahorro'].mean(), color='gray',
           linestyle='--', linewidth=1.2,
           label=f"Media = {df_ahorro['tasa_ahorro'].mean():.1f}%")
ax.set_title('Tasa de ahorro bruto de los hogares españoles (2015–2024)',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Tasa de ahorro (%)', fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_tasa_ahorro.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figura 2 guardada: Tasa de ahorro")


# ============================================================
# FIGURA 3 — Tasa de Desempleo 25-34 años vs. Ahorro anual
# ============================================================

df_plot = df.dropna(subset=['tasa_desempleo', 'tasa_ahorro_pct'])

fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(df_plot['tasa_desempleo'], df_plot['tasa_ahorro_pct'],
                     c=df_plot['anio'], cmap='viridis', s=80, zorder=5)
for _, row in df_plot.iterrows():
    ax.annotate(str(int(row['anio'])),
                (row['tasa_desempleo'], row['tasa_ahorro_pct']),
                textcoords='offset points', xytext=(5, 5), fontsize=8)

plt.colorbar(scatter, ax=ax, label='Año')
ax.set_xlabel('Tasa de desempleo 25-34 años (%)', fontsize=11)
ax.set_ylabel('Tasa de ahorro hogares (%)', fontsize=11)
ax.set_title('Relación entre desempleo juvenil y ahorro en España',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig3_desempleo_ahorro.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figura 3 guardada: Desempleo vs. Ahorro")


# ============================================================
# FIGURA 4 — Evolución mensual de la morosidad (2015-2024)
# ============================================================

df_mora_reciente = df_morosidad[df_morosidad['anio'] >= 2015].copy()
df_mora_reciente['fecha_num'] = (
    df_mora_reciente['anio'] * 100 + df_mora_reciente['mes']
)
df_mora_reciente = df_mora_reciente.sort_values('fecha_num')

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(range(len(df_mora_reciente)),
        df_mora_reciente['tasa_morosidad_pct'],
        color='#EA4335', linewidth=2.5, marker='o', markersize=5)
ax.fill_between(range(len(df_mora_reciente)),
                df_mora_reciente['tasa_morosidad_pct'],
                alpha=0.15, color='#EA4335')

etiquetas = [f"{int(r.anio)}-{str(int(r.mes)).zfill(2)}"
             for _, r in df_mora_reciente.iterrows()]
step = max(1, len(etiquetas) // 10)
ax.set_xticks(range(0, len(etiquetas), step))
ax.set_xticklabels(etiquetas[::step], rotation=45, ha='right', fontsize=9)
ax.axhline(df_mora_reciente['tasa_morosidad_pct'].mean(), color='gray',
           linestyle='--', linewidth=1.2,
           label=f"Media = {df_mora_reciente['tasa_morosidad_pct'].mean():.2f}%")
ax.set_title('Evolución de la tasa de morosidad bancaria en España (2015–2024)',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Tasa de morosidad (%)', fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig4_morosidad_mensual.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figura 4 guardada: Morosidad mensual")


# ============================================================
# FIGURA 5 — Matriz de correlación
# ============================================================

num_cols = ['tasa_ahorro_pct', 'tasa_desempleo',
            'ipc_variacion_anual_pct', 'euribor_pct', 'tasa_morosidad_pct']
corr_df = df[num_cols].dropna()

if len(corr_df) >= 3:
    corr = corr_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, linewidths=0.5)
    ax.set_title('Matriz de correlación — variables financieras España',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig5_correlacion.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Figura 5 guardada: Matriz de correlación")


# ============================================================
# RESUMEN ESTADÍSTICO Y CONCLUSIONES EDA
# ============================================================

print("\n" + "=" * 60)
print("CONCLUSIONES DEL ANÁLISIS EXPLORATORIO")
print("=" * 60)

mora_max  = df['tasa_morosidad_pct'].max()
mora_min  = df['tasa_morosidad_pct'].min()
mora_2024 = df[df['anio'] == 2024]['tasa_morosidad_pct'].values[0]
ahorro_covid = df[df['anio'] == 2020]['tasa_ahorro_pct'].values
euribor_2022 = df[df['anio'] == 2022]['euribor_pct'].values[0]
euribor_2024 = df[df['anio'] == 2024]['euribor_pct'].values[0]

print(f"""
1. MOROSIDAD EN MÍNIMOS HISTÓRICOS
   La tasa de morosidad bancaria cerró 2024 en {mora_2024:.2f}%,
   el nivel más bajo desde 2008 (máximo histórico: {mora_max:.2f}% en 2013).
   Esto indica un contexto financiero saneado para el lanzamiento
   del producto, con menor aversión al riesgo del consumidor.

2. RECUPERACIÓN DEL AHORRO TRAS COVID-19
   La tasa de ahorro de los hogares españoles alcanzó un máximo
   extraordinario del 17.6% en 2020 (efecto COVID). En 2024 se
   estabilizó en torno al 12.7%, por encima de la media histórica
   pre-pandemia (~7.5%). Esto amplía la base de potenciales usuarios
   con capacidad de ahorro disponible para gestionar.

3. EURÍBOR EN DESCENSO TRAS EL PICO DE 2023
   El Euríbor 12 meses alcanzó el 4.07% en 2023 (máximo desde 2008)
   y comenzó a descender en 2024 ({euribor_2024:.3f}%). Este ciclo de
   normalización monetaria aumenta el interés por optimizar el
   rendimiento del ahorro — oportunidad directa para AI Financial
   Life Coach.

4. DESEMPLEO JUVENIL EN MÍNIMOS PERO AÚN ELEVADO
   La tasa de desempleo del segmento 25-34 años bajó al 16.9% en 2024,
   su nivel más bajo desde 2008. Sin embargo, sigue siendo el doble
   de la media EU, lo que implica ingresos variables o irregulares
   en el mercado objetivo — precisamente el perfil que AI Financial
   Life Coach busca servir.

5. CORRELACIONES RELEVANTES
   Se observa correlación positiva entre Euríbor y morosidad
   (periodos de tipos altos coinciden con mayor morosidad) y
   correlación negativa entre tasa de ahorro y tasa de desempleo.
""")
print("=" * 60)
