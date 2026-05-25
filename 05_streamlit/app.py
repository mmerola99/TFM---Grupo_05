# ============================================================
# NOTEBOOK 05 — App Streamlit: AI Financial Life Coach
# Exploración interactiva del dataset financiero España
# ============================================================
# Per avviare: streamlit run app.py --server.port 8080

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── CONFIGURACIÓN DE PÁGINA ──────────────────────────────────
st.set_page_config(
    page_title="AI Financial Life Coach — Dashboard",
    page_icon="💰",
    layout="wide"
)

sns.set_style("whitegrid")

# ── CARGA DE DATOS ───────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
CLEAN_DIR  = os.path.join(BASE_DIR, '..', 'data', 'clean')

@st.cache_data
def load_data():
    df          = pd.read_csv(f"{CLEAN_DIR}/dataset_final.csv")
    df_ahorro   = pd.read_csv(f"{CLEAN_DIR}/ahorro_clean.csv")
    df_morosidad= pd.read_csv(f"{CLEAN_DIR}/morosidad_clean.csv")
    return df, df_ahorro, df_morosidad

df, df_ahorro, df_morosidad = load_data()

# ── HEADER ───────────────────────────────────────────────────
st.title("💰 AI Financial Life Coach")
st.subheader("Dashboard de exploración — Contexto financiero España")
st.markdown(
    "Esta aplicación forma parte del TFM del Máster en Big Data & Business Intelligence (Next Educación). "
    "Permite explorar los indicadores macroeconómicos del mercado objetivo de AI Financial Life Coach: "
    "España, segmento 25–40 años."
)
st.divider()

# ── MÉTRICAS PRINCIPALES ─────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

ultimo = df.iloc[-1]

with col1:
    st.metric(
        label="Tasa de ahorro hogares (2024)",
        value=f"{ultimo['tasa_ahorro_pct']:.1f}%",
        delta=f"+{ultimo['tasa_ahorro_pct'] - df.iloc[-3]['tasa_ahorro_pct']:.1f}pp vs 2022"
    )
with col2:
    st.metric(
        label="Morosidad bancaria (2024)",
        value=f"{ultimo['tasa_morosidad_pct']:.2f}%",
        delta=f"{ultimo['tasa_morosidad_pct'] - df[df['anio']==2023]['tasa_morosidad_pct'].values[0]:.2f}pp vs 2023",
        delta_color="inverse"
    )
with col3:
    st.metric(
        label="Euríbor 12m (2024)",
        value=f"{ultimo['euribor_pct']:.3f}%",
        delta=f"{ultimo['euribor_pct'] - df[df['anio']==2023]['euribor_pct'].values[0]:.3f}pp vs 2023",
        delta_color="inverse"
    )
with col4:
    st.metric(
        label="Desempleo 25-34 años (2024)",
        value=f"{ultimo['tasa_desempleo']:.1f}%",
        delta=f"{ultimo['tasa_desempleo'] - df[df['anio']==2023]['tasa_desempleo'].values[0]:.1f}pp vs 2023",
        delta_color="inverse"
    )

st.divider()

# ── SIDEBAR: FILTROS ─────────────────────────────────────────
st.sidebar.header("⚙️ Filtros")

anio_min, anio_max = st.sidebar.slider(
    "Período de análisis",
    min_value=int(df['anio'].min()),
    max_value=int(df['anio'].max()),
    value=(2015, 2024)
)

variables = st.sidebar.multiselect(
    "Variables a mostrar",
    options=['euribor_pct', 'tasa_morosidad_pct',
             'tasa_ahorro_pct', 'tasa_desempleo'],
    default=['euribor_pct', 'tasa_morosidad_pct']
)

# Filtrar por año
df_filtrado = df[(df['anio'] >= anio_min) & (df['anio'] <= anio_max)]

# ── GRÁFICO 1: EVOLUCIÓN DE VARIABLES SELECCIONADAS ─────────
st.subheader("📈 Evolución de indicadores financieros")

if variables:
    fig1, ax = plt.subplots(figsize=(12, 5))
    colores = ['#1A73E8', '#EA4335', '#34A853', '#FBBC04']
    for i, var in enumerate(variables):
        datos = df_filtrado[['anio', var]].dropna()
        ax.plot(datos['anio'], datos[var],
                marker='o', linewidth=2.5, markersize=5,
                color=colores[i % len(colores)], label=var)
    ax.set_xlabel("Año")
    ax.set_ylabel("Valor (%)")
    ax.legend()
    ax.set_title(f"Indicadores seleccionados ({anio_min}–{anio_max})")
    st.pyplot(fig1)
    plt.close()
else:
    st.info("Selecciona al menos una variable en el panel lateral.")

st.divider()

# ── GRÁFICO 2: TASA DE AHORRO TRIMESTRAL ────────────────────
st.subheader("💼 Tasa de ahorro bruto de los hogares (trimestral)")

df_ahorro_filtrado = df_ahorro[
    df_ahorro['anio'] >= anio_min
].copy() if 'anio' in df_ahorro.columns else df_ahorro.copy()

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.fill_between(range(len(df_ahorro_filtrado)),
                 df_ahorro_filtrado['tasa_ahorro'],
                 alpha=0.3, color='#34A853')
ax2.plot(range(len(df_ahorro_filtrado)),
         df_ahorro_filtrado['tasa_ahorro'],
         color='#34A853', linewidth=2)
step = max(1, len(df_ahorro_filtrado) // 8)
ax2.set_xticks(range(0, len(df_ahorro_filtrado), step))
ax2.set_xticklabels(
    df_ahorro_filtrado['periodo'].iloc[::step],
    rotation=45, ha='right', fontsize=9
)
ax2.axhline(df_ahorro_filtrado['tasa_ahorro'].mean(),
            color='gray', linestyle='--', linewidth=1.2,
            label=f"Media = {df_ahorro_filtrado['tasa_ahorro'].mean():.1f}%")
ax2.set_ylabel("Tasa de ahorro (%)")
ax2.legend()
ax2.set_title("Tasa de ahorro bruto de los hogares españoles (Fuente: Eurostat)")
st.pyplot(fig2)
plt.close()

st.divider()

# ── GRÁFICO 3: MOROSIDAD MENSUAL ─────────────────────────────
st.subheader("🏦 Morosidad bancaria mensual (Banco de España)")

df_mora_filtrado = df_morosidad[df_morosidad['anio'] >= anio_min].copy()

fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(range(len(df_mora_filtrado)),
         df_mora_filtrado['tasa_morosidad_pct'],
         color='#EA4335', linewidth=2.5, marker='o', markersize=4)
ax3.fill_between(range(len(df_mora_filtrado)),
                 df_mora_filtrado['tasa_morosidad_pct'],
                 alpha=0.15, color='#EA4335')
etiquetas = [f"{int(r.anio)}-{str(int(r.mes)).zfill(2)}"
             for _, r in df_mora_filtrado.iterrows()]
step = max(1, len(etiquetas) // 10)
ax3.set_xticks(range(0, len(etiquetas), step))
ax3.set_xticklabels(etiquetas[::step], rotation=45, ha='right', fontsize=9)
ax3.set_ylabel("Tasa de morosidad (%)")
ax3.set_title("Evolución mensual de la morosidad bancaria en España")
st.pyplot(fig3)
plt.close()

st.divider()

# ── DATOS CRUDOS ─────────────────────────────────────────────
with st.expander("📊 Ver dataset consolidado completo"):
    st.dataframe(df_filtrado, use_container_width=True)
    st.download_button(
        label="⬇️ Descargar CSV",
        data=df_filtrado.to_csv(index=False).encode('utf-8'),
        file_name="dataset_ai_financial_life_coach.csv",
        mime="text/csv"
    )

# ── FOOTER ───────────────────────────────────────────────────
st.divider()
st.caption(
    "**Fuentes:** Eurostat Statistics API (nasq_10_ki, une_rt_a, prc_hicp_manr) · "
    "Banco de España — Boletín Estadístico Cap.4 | "
    "**Proyecto:** TFM — AI Financial Life Coach · Big Data & Business Intelligence · Next Educación"
)
