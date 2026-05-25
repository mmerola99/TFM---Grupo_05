# ============================================================
# app.py — AI Financial Life Coach
# Dashboard predictivo interactivo
# Lanzar: streamlit run app.py --server.port 8080
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.set_page_config(
    page_title="AI Financial Life Coach",
    page_icon="💰",
    layout="wide"
)

sns.set_style("whitegrid")

BASE_DIR  = os.path.dirname(__file__)
CLEAN_DIR = os.path.join(BASE_DIR, '..', 'data', 'clean')
RAW_DIR   = os.path.join(BASE_DIR, '..', 'data', 'raw')

# ── CARGA DE DATOS Y MODELOS ─────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(f"{RAW_DIR}/dataset_sintetico_usuarios.csv")
    ahorro_medio = pd.read_csv(f"{CLEAN_DIR}/ahorro_medio_mensual.csv")
    with open(f"{CLEAN_DIR}/metricas_modelos.json") as f:
        metricas = json.load(f)
    return df, ahorro_medio, metricas

@st.cache_resource
def train_models(df):
    # Modelo 1 — Regresión Lineal
    features = ['salario', 'vivienda', 'alimentacion', 'transporte',
                'ocio', 'salud', 'educacion', 'otros', 'edad', 'ipc_mensual']
    le = LabelEncoder()
    df['perfil_encoded'] = le.fit_transform(df['perfil'])
    features_clf = features + ['perfil_encoded']

    scaler1 = StandardScaler()
    X1 = scaler1.fit_transform(df[features])
    lr = LinearRegression()
    lr.fit(X1, df['ahorro'])

    scaler2 = StandardScaler()
    X2 = scaler2.fit_transform(df[features_clf])
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf.fit(X2, df['perfil_ahorro'])

    # Modelo 3 — Serie temporal
    ahorro_med = df.groupby('fecha')['ahorro'].mean().reset_index()
    ahorro_med['fecha_dt'] = pd.to_datetime(ahorro_med['fecha'])
    ahorro_med = ahorro_med.sort_values('fecha_dt').reset_index(drop=True)
    ahorro_med['t'] = range(len(ahorro_med))
    ahorro_med['sin_mes'] = np.sin(2*np.pi*ahorro_med['fecha_dt'].dt.month/12)
    ahorro_med['cos_mes'] = np.cos(2*np.pi*ahorro_med['fecha_dt'].dt.month/12)

    ts = LinearRegression()
    ts.fit(ahorro_med[['t','sin_mes','cos_mes']], ahorro_med['ahorro'])

    return lr, scaler1, clf, scaler2, le, ts, ahorro_med

df, ahorro_medio, metricas = load_data()
lr_model, scaler1, clf_model, scaler2, le, ts_model, ahorro_med = train_models(df)

# ── HEADER ───────────────────────────────────────────────────
st.title("💰 AI Financial Life Coach")
st.subheader("Plataforma de gestión financiera personal — España 2023-2026")
st.markdown(
    "Dashboard predictivo desarrollado como parte del TFM del Máster en "
    "Big Data & Business Intelligence (Next Educación). Simula el comportamiento "
    "financiero de usuarios españoles de 25-40 años y predice su capacidad de ahorro."
)
st.divider()

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Simulador Personal",
    "📊 Análisis del Dataset",
    "🤖 Rendimiento de Modelos",
    "📈 Proyección Temporal"
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — SIMULADOR PERSONAL
# ══════════════════════════════════════════════════════════════
with tab1:
    st.header("🔮 Simulador de perfil financiero personal")
    st.markdown(
        "Introduce tu perfil financiero y el modelo predirá tu ahorro mensual "
        "estimado y clasificará tu perfil de ahorrador."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tu perfil")
        edad        = st.slider("Edad", 25, 40, 30)
        salario     = st.number_input("Salario mensual neto (€)", 1100, 6000, 2385, step=50)
        perfil_tipo = st.selectbox("Tipo de empleo",
                                   ['medio', 'junior', 'senior', 'freelance'])
        ipc         = st.number_input("IPC mensual estimado (%)", 0.1, 1.0, 0.25, step=0.05)

    with col2:
        st.subheader("Tus gastos mensuales (€)")
        vivienda      = st.number_input("Vivienda (alquiler/hipoteca)", 300, 2000, 700, step=50)
        alimentacion  = st.number_input("Alimentación", 100, 800, 350, step=25)
        transporte    = st.number_input("Transporte", 50, 500, 240, step=25)
        ocio          = st.number_input("Ocio y restauración", 30, 400, 190, step=10)
        salud         = st.number_input("Salud", 0, 300, 95, step=10)
        educacion     = st.number_input("Educación / formación", 0, 300, 70, step=10)
        otros         = st.number_input("Otros gastos", 0, 400, 145, step=10)

    gasto_total = vivienda + alimentacion + transporte + ocio + salud + educacion + otros
    ahorro_calc = salario - gasto_total
    tasa_calc   = (ahorro_calc / salario * 100) if salario > 0 else 0

    st.divider()
    st.subheader("📊 Resultados de la predicción")

    # Predicción Modelo 1
    perfil_enc = le.transform([perfil_tipo])[0]
    X_sim1 = scaler1.transform([[salario, vivienda, alimentacion, transporte,
                                  ocio, salud, educacion, otros, edad, ipc]])
    ahorro_pred = lr_model.predict(X_sim1)[0]

    # Predicción Modelo 2
    X_sim2 = scaler2.transform([[salario, vivienda, alimentacion, transporte,
                                  ocio, salud, educacion, otros, edad, ipc,
                                  perfil_enc]])
    perfil_pred = clf_model.predict(X_sim2)[0]
    proba_pred  = clf_model.predict_proba(X_sim2)[0]

    col3, col4, col5 = st.columns(3)

    with col3:
        st.metric("Ahorro predicho (Modelo 1)",
                  f"{ahorro_pred:.2f} €",
                  delta=f"{ahorro_pred - ahorro_calc:.2f}€ vs cálculo manual")

    with col4:
        color_perfil = {
            'buen_ahorrador':       '🟢',
            'ahorro_moderado':      '🟡',
            'ahorro_insuficiente':  '🔴'
        }
        etiqueta = {
            'buen_ahorrador':       'Buen ahorrador',
            'ahorro_moderado':      'Ahorro moderado',
            'ahorro_insuficiente':  'Ahorro insuficiente'
        }
        st.metric("Perfil de ahorro (Modelo 2)",
                  f"{color_perfil.get(perfil_pred,'⚪')} {etiqueta.get(perfil_pred, perfil_pred)}")

    with col5:
        st.metric("Tasa de ahorro calculada",
                  f"{tasa_calc:.1f}%",
                  delta=f"Media España: 12.0%",
                  delta_color="off")

    # Recomendación
    st.divider()
    st.subheader("💡 Recomendación de AI Financial Life Coach")

    if perfil_pred == 'buen_ahorrador':
        st.success(
            f"✅ Tu perfil financiero es **sólido**. Con un ahorro estimado de "
            f"**{ahorro_pred:.0f}€/mes** ({tasa_calc:.1f}% de tus ingresos), "
            f"estás por encima de la media española del 12%. "
            f"Considera destinar parte de tu ahorro a productos de inversión "
            f"como fondos indexados o depósitos a plazo fijo."
        )
    elif perfil_pred == 'ahorro_moderado':
        st.warning(
            f"⚠️ Tu ahorro de **{ahorro_pred:.0f}€/mes** ({tasa_calc:.1f}%) es moderado. "
            f"La principal oportunidad de mejora está en **vivienda** "
            f"({vivienda/salario*100:.0f}% de tu salario) y **ocio** "
            f"({ocio/salario*100:.0f}%). Reducir el gasto en ocio un 10% "
            f"ahorraría {ocio*0.1:.0f}€/mes adicionales."
        )
    else:
        st.error(
            f"🔴 Tu ahorro de **{ahorro_pred:.0f}€/mes** ({tasa_calc:.1f}%) "
            f"está por debajo del mínimo recomendado del 5%. "
            f"Tu gasto en vivienda ({vivienda/salario*100:.0f}% del salario) "
            f"supera el umbral recomendado del 30%. "
            f"Prioriza reducir gastos variables como ocio y otros."
        )

    # Gráfico distribución gastos
    st.divider()
    st.subheader("📊 Distribución de tus gastos")

    gastos_dict = {
        'Vivienda': vivienda, 'Alimentación': alimentacion,
        'Transporte': transporte, 'Ocio': ocio,
        'Salud': salud, 'Educación': educacion, 'Otros': otros
    }
    fig_pie, ax_pie = plt.subplots(figsize=(7, 4))
    colores = ['#1A73E8','#34A853','#FBBC04','#EA4335','#9C27B0','#FF5722','#607D8B']
    wedges, texts, autotexts = ax_pie.pie(
        gastos_dict.values(),
        labels=gastos_dict.keys(),
        colors=colores,
        autopct='%1.1f%%',
        startangle=90
    )
    ax_pie.set_title(f'Distribución de gastos — Total: {gasto_total:.0f}€/mes')
    st.pyplot(fig_pie)
    plt.close()


# ══════════════════════════════════════════════════════════════
# TAB 2 — ANÁLISIS DEL DATASET
# ══════════════════════════════════════════════════════════════
with tab2:
    st.header("📊 Análisis del dataset sintético")
    st.markdown(
        f"Dataset de **{df['user_id'].nunique()} usuarios** españoles con "
        f"**{len(df):,} registros** mensuales (enero 2023 – mayo 2026). "
        f"Calibrado sobre datos reales INE/Eurostat 2025."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Salario medio", f"{df['salario'].mean():.0f} €/mes")
    with col2:
        st.metric("Ahorro medio", f"{df['ahorro'].mean():.0f} €/mes")
    with col3:
        st.metric("Tasa ahorro media", f"{df['tasa_ahorro_pct'].mean():.1f}%")
    with col4:
        st.metric("Usuarios analizados", f"{df['user_id'].nunique():,}")

    st.divider()

    # Filtros
    st.sidebar.header("⚙️ Filtros del dataset")
    perfiles_sel = st.sidebar.multiselect(
        "Perfil laboral",
        options=df['perfil'].unique(),
        default=df['perfil'].unique()
    )
    edad_rango = st.sidebar.slider("Rango de edad", 25, 40, (25, 40))
    fecha_sel  = st.sidebar.select_slider(
        "Período",
        options=sorted(df['fecha'].unique()),
        value=(df['fecha'].min(), df['fecha'].max())
    )

    df_f = df[
        (df['perfil'].isin(perfiles_sel)) &
        (df['edad'] >= edad_rango[0]) &
        (df['edad'] <= edad_rango[1]) &
        (df['fecha'] >= fecha_sel[0]) &
        (df['fecha'] <= fecha_sel[1])
    ]

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Distribución del ahorro mensual")
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        ax1.hist(df_f['ahorro'], bins=40, color='#1A73E8', alpha=0.8, edgecolor='white')
        ax1.axvline(df_f['ahorro'].mean(), color='#EA4335', linestyle='--',
                    linewidth=1.8, label=f"Media: {df_f['ahorro'].mean():.0f}€")
        ax1.set_xlabel("Ahorro mensual (€)")
        ax1.set_ylabel("Frecuencia")
        ax1.set_title("Distribución del ahorro mensual")
        ax1.legend()
        st.pyplot(fig1)
        plt.close()

    with col_b:
        st.subheader("Perfil de ahorro por tipo de empleo")
        perfil_counts = df_f.groupby(['perfil', 'perfil_ahorro']).size().unstack(fill_value=0)
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        perfil_counts.plot(kind='bar', ax=ax2, color=['#EA4335','#34A853','#FBBC04'],
                           alpha=0.85, edgecolor='white')
        ax2.set_xlabel("Tipo de empleo")
        ax2.set_ylabel("Número de registros")
        ax2.set_title("Perfil de ahorro por tipo de empleo")
        ax2.legend(title="Perfil ahorro", fontsize=8)
        plt.xticks(rotation=0)
        st.pyplot(fig2)
        plt.close()

    st.subheader("Evolución del ahorro medio mensual")
    ahorro_evol = df_f.groupby('fecha')['ahorro'].mean().reset_index()
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(range(len(ahorro_evol)), ahorro_evol['ahorro'],
             color='#1A73E8', linewidth=2, marker='o', markersize=3)
    step = max(1, len(ahorro_evol) // 10)
    ax3.set_xticks(range(0, len(ahorro_evol), step))
    ax3.set_xticklabels(ahorro_evol['fecha'].iloc[::step], rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel("Ahorro medio (€)")
    ax3.set_title("Evolución del ahorro medio mensual — usuarios filtrados")
    st.pyplot(fig3)
    plt.close()

    with st.expander("📋 Ver datos del dataset"):
        st.dataframe(df_f.head(100), use_container_width=True)
        st.download_button(
            "⬇️ Descargar CSV",
            df_f.to_csv(index=False).encode('utf-8'),
            "dataset_filtrado.csv", "text/csv"
        )


# ══════════════════════════════════════════════════════════════
# TAB 3 — RENDIMIENTO DE MODELOS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.header("🤖 Rendimiento de los modelos predictivos")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Modelo 1 — Regresión Lineal")
        st.markdown("**Predicción del ahorro mensual (€)**")
        st.metric("R² (varianza explicada)", f"{metricas['modelo1_r2']:.4f}")
        st.metric("MAE (error medio absoluto)", f"{metricas['modelo1_mae']:.2f} €")
        st.markdown(
            "Un R² cercano a 1.0 indica que el modelo explica prácticamente "
            "toda la variabilidad del ahorro a partir de las variables de entrada. "
            "El MAE indica el error promedio en euros de las predicciones."
        )
        img_path = f"{CLEAN_DIR}/modelo1_regresion_lineal.png"
        if os.path.exists(img_path):
            st.image(img_path, caption="Ahorro real vs. predicho")

        img_coef = f"{CLEAN_DIR}/modelo1_coeficientes.png"
        if os.path.exists(img_coef):
            st.image(img_coef, caption="Importancia de variables")

    with col2:
        st.subheader("Modelo 2 — Regresión Logística")
        st.markdown("**Clasificación del perfil de ahorro**")
        st.metric("Accuracy", f"{metricas['modelo2_accuracy']:.4f}")
        st.markdown(
            "Clasifica cada usuario en tres perfiles: buen ahorrador, "
            "ahorro moderado o ahorro insuficiente. "
            "Esta clasificación es la base del sistema de recomendaciones "
            "personalizadas de AI Financial Life Coach."
        )
        img_cm = f"{CLEAN_DIR}/modelo2_confusion.png"
        if os.path.exists(img_cm):
            st.image(img_cm, caption="Matriz de confusión")


# ══════════════════════════════════════════════════════════════
# TAB 4 — PROYECCIÓN TEMPORAL
# ══════════════════════════════════════════════════════════════
with tab4:
    st.header("📈 Proyección temporal del ahorro (Modelo 3)")
    st.markdown(
        "Proyección del ahorro medio mensual para los próximos 6 meses "
        "(junio–noviembre 2026) basada en la tendencia histórica del dataset."
    )

    img_proj = f"{CLEAN_DIR}/modelo3_proyeccion.png"
    if os.path.exists(img_proj):
        st.image(img_proj, use_container_width=True)

    st.subheader("Valores proyectados")
    pred_df = pd.DataFrame(metricas['modelo3_predicciones'])
    pred_df.columns = ['Mes', 'Ahorro medio proyectado (€)']

    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(pred_df, use_container_width=True)
    with col2:
        st.markdown("""
        **Interpretación:**

        La proyección muestra una ligera tendencia descendente del ahorro medio
        en el segundo semestre de 2026, coherente con las previsiones de
        CaixaBank Research (mayo 2026) que anticipan:

        - Descenso gradual de la tasa de ahorro desde el 12% actual
        - Euríbor estabilizándose en torno al 2,2%
        - Desempleo por debajo del 10% a finales de 2026

        Este contexto implica que los usuarios de AI Financial Life Coach
        necesitarán cada vez más apoyo para mantener sus metas de ahorro
        ante el aumento del consumo.
        """)

st.divider()
st.caption(
    "**Fuentes:** Dataset sintético calibrado sobre INE 2025, Eurostat 2025, "
    "CaixaBank Research 2026 | "
    "**Proyecto:** TFM — AI Financial Life Coach · "
    "Big Data & Business Intelligence · Next Educación"
)
