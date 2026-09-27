# ============================================================
# modelos_predictivos.py
# Modelos de Machine Learning — AI Financial Life Coach
#
# Modelo 1: Regresión Lineal   — predicción del ahorro mensual
# Modelo 2: Regresión Logística — clasificación del perfil de ahorro
# Modelo 3: Serie temporal      — proyección del ahorro agregado (6 meses)
# Modelo 1-bis: Proyección individual — ahorro del mes siguiente por usuario
#
# ------------------------------------------------------------
# NOTA METODOLÓGICA (revisión de esta versión)
# ------------------------------------------------------------
# Los Modelos 1 y 2 predicen ahora el mes t a partir de información
# disponible ANTES de ese mes (rezagos y medias móviles de los meses
# t-1, t-2, t-3), en lugar de utilizar como predictores las propias
# componentes del mismo mes que definen algebraicamente el objetivo
# (ahorro = salario - gasto_total). Esta reformulación temporal evita
# la fuga de información (data leakage) y obliga al modelo a aprender
# un patrón de comportamiento financiero real, no una identidad
# contable. La partición train/test se realiza además por usuario
# (GroupShuffleSplit), de modo que los meses de un mismo usuario no
# se repartan simultáneamente entre entrenamiento y prueba.
#
# La arquitectura se organiza en funciones reutilizables para que el
# re-entrenamiento periódico (Fase 2: incorporación de datos reales
# procedentes de Tink, una vez transformados al mismo esquema de 17
# columnas mediante el pipeline de preparación correspondiente) pueda
# invocarse sin cambios estructurales, concatenando el dataset
# sintético y el dataset real antes de llamar a build_lag_features().
# ============================================================

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_absolute_error, r2_score,
                              classification_report, accuracy_score,
                              confusion_matrix, ConfusionMatrixDisplay)

sns.set_style("whitegrid")
sns.set_palette("Set2")

DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
CLEAN_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean')
os.makedirs(CLEAN_DIR, exist_ok=True)

RANDOM_STATE = 42
CATEGORIAS = ['vivienda', 'alimentacion', 'transporte', 'ocio',
              'salud', 'educacion', 'otros']

PROFILE_ENCODING = {'junior': 0, 'medio': 1, 'senior': 2, 'freelance': 3}


# ============================================================
# INGESTA Y ARMONIZACIÓN DE ESQUEMA
# ============================================================

def cargar_dataset(path):
    """Carga un dataset mensual por usuario con el esquema estándar de
    17 columnas (sintético o derivado de Tink vía prepara_usuarios_tink.py)
    y homogeneiza tipos para que ambas fuentes sean intercambiables."""
    df = pd.read_csv(path)

    if 'perfil_encoded' not in df.columns:
        df['perfil_encoded'] = df['perfil'].map(PROFILE_ENCODING)

    df['fecha_dt'] = pd.to_datetime(df['fecha'], format='%Y-%m')
    df = df.sort_values(['user_id', 'fecha_dt']).reset_index(drop=True)
    return df


# ============================================================
# INGENIERÍA DE VARIABLES TEMPORALES (anti-leakage)
# ============================================================

def build_lag_features(df):
    """Construye, para cada usuario y cada mes t, un vector de
    predictores basado exclusivamente en los meses t-1, t-2 y t-3.

    El mes t conserva únicamente su variable objetivo (ahorro,
    perfil_ahorro) y las variables verdaderamente exógenas o estáticas
    que no participan de la identidad ahorro = salario - gasto_total:
    edad, perfil_encoded e ipc (rezagado un mes, ya que el IPC de un
    mes se publica con retraso y no está disponible al inicio de éste).

    De este modo ningún predictor del mes t puede reconstruir
    algebraicamente el ahorro de ese mismo mes.
    """
    df = df.copy()
    g = df.groupby('user_id')

    # IPC rezagado (variable macro exógena, no depende del usuario)
    df['ipc_lag1'] = g['ipc_mensual'].shift(1)

    # Ahorro: 3 rezagos individuales + media móvil de los 3 meses previos
    for lag in (1, 2, 3):
        df[f'ahorro_lag{lag}'] = g['ahorro'].shift(lag)
    df['ahorro_roll3'] = g['ahorro'].transform(lambda s: s.shift(1).rolling(3).mean())

    # Salario y gasto total: nivel reciente + tendencia (media móvil)
    df['salario_lag1'] = g['salario'].shift(1)
    df['salario_roll3'] = g['salario'].transform(lambda s: s.shift(1).rolling(3).mean())
    df['gasto_total_lag1'] = g['gasto_total'].shift(1)
    df['gasto_total_roll3'] = g['gasto_total'].transform(lambda s: s.shift(1).rolling(3).mean())

    # Categorías de gasto: patrón reciente (un rezago es suficiente para
    # capturar el mix de consumo sin disparar la dimensionalidad)
    for cat in CATEGORIAS:
        df[f'{cat}_lag1'] = g[cat].shift(1)

    feature_cols = (
        ['edad', 'perfil_encoded', 'ipc_lag1',
         'ahorro_lag1', 'ahorro_lag2', 'ahorro_lag3', 'ahorro_roll3',
         'salario_lag1', 'salario_roll3',
         'gasto_total_lag1', 'gasto_total_roll3']
        + [f'{cat}_lag1' for cat in CATEGORIAS]
    )

    # Los primeros 3 meses de cada usuario no tienen histórico suficiente
    df_model = df.dropna(subset=feature_cols).reset_index(drop=True)

    return df_model, feature_cols


# ============================================================
# PARTICIÓN TRAIN/TEST AGRUPADA POR USUARIO
# ============================================================

def split_por_usuario(df, test_size=0.2, random_state=RANDOM_STATE):
    """Reparte los usuarios (no las filas) entre train y test, de modo
    que ningún usuario aparezca simultáneamente en ambos conjuntos."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                  random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df['user_id']))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


# ============================================================
# INTERPRETABILIDAD (XAI)
# ============================================================

def explicacion_local(coef, intercept, x_std, feature_names, top_n=6):
    """Descompone una predicción individual en la contribución de cada
    variable: contribución_i = coeficiente_i * valor_estandarizado_i.
    La suma de las contribuciones más el intercepto reproduce la
    predicción del modelo, lo que permite explicar 'por qué' el modelo
    ha emitido ese valor concreto para ese usuario/mes (interpretabilidad
    local), complementando el gráfico de coeficientes globales."""
    contribuciones = coef * x_std
    orden = np.argsort(-np.abs(contribuciones))[:top_n]
    detalle = pd.DataFrame({
        'variable': np.array(feature_names)[orden],
        'contribucion': contribuciones[orden],
    })
    prediccion_reconstruida = contribuciones.sum() + intercept
    return detalle, prediccion_reconstruida


def graficar_explicacion_local(detalle, titulo, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    colores = ['#EA4335' if c < 0 else '#34A853' for c in detalle['contribucion']]
    ax.barh(detalle['variable'], detalle['contribucion'], color=colores, alpha=0.9)
    ax.axvline(0, color='gray', linewidth=0.8)
    ax.set_title(titulo, fontweight='bold')
    ax.set_xlabel('Contribución a la predicción (€)')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


print("=" * 60)
print("MODELOS PREDICTIVOS — AI Financial Life Coach")
print("=" * 60)

# ── CARGA DE DATOS ───────────────────────────────────────────
df_raw = cargar_dataset(f"{DATA_DIR}/dataset_sintetico_usuarios.csv")
print(f"\nDataset: {df_raw.shape[0]:,} registros | {df_raw['user_id'].nunique()} usuarios")
print(f"Período: {df_raw['fecha'].min()} — {df_raw['fecha'].max()}")

df_model, feature_cols = build_lag_features(df_raw)
print(f"Registros aptos para modelado tras construir rezagos: {df_model.shape[0]:,} "
      f"(se descartan los 3 primeros meses de cada usuario por falta de histórico)")


# ============================================================
# MODELO 1 — REGRESIÓN LINEAL (reformulación temporal)
# Objetivo: predecir el ahorro del mes t a partir del histórico
#           t-1, t-2, t-3 del propio usuario (sin usar componentes
#           del propio mes t)
# ============================================================

print("\n" + "=" * 60)
print("MODELO 1 — Regresión Lineal: predicción del ahorro mensual")
print("(a partir del histórico de los 3 meses anteriores del usuario)")
print("=" * 60)

target_lr = 'ahorro'

train_df, test_df = split_por_usuario(df_model, test_size=0.2)

X_train, y_train = train_df[feature_cols], train_df[target_lr]
X_test, y_test = test_df[feature_cols], test_df[target_lr]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr_model = LinearRegression()
lr_model.fit(X_train_s, y_train)
y_pred_lr = lr_model.predict(X_test_s)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"\n  Usuarios train: {train_df['user_id'].nunique():,} | "
      f"Usuarios test: {test_df['user_id'].nunique():,}")
print(f"  Filas  train: {len(X_train):,} | Filas  test: {len(X_test):,}")
print(f"  MAE:  {mae_lr:.2f} € (error medio absoluto)")
print(f"  R²:   {r2_lr:.4f} (varianza explicada)")

# Gráfico 1: Real vs Predicho
fig, ax = plt.subplots(figsize=(8, 5))
rng = np.random.default_rng(RANDOM_STATE)
muestra = rng.choice(len(y_test), min(500, len(y_test)), replace=False)
ax.scatter(y_test.iloc[muestra], y_pred_lr[muestra],
           alpha=0.4, s=20, color='#1A73E8')
lim = [min(y_test.min(), y_pred_lr.min()),
       max(y_test.max(), y_pred_lr.max())]
ax.plot(lim, lim, 'r--', linewidth=1.5, label='Predicción perfecta')
ax.set_xlabel('Ahorro real (€)')
ax.set_ylabel('Ahorro predicho (€)')
ax.set_title('Modelo 1 — Regresión Lineal: Ahorro real vs. predicho',
             fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f"{CLEAN_DIR}/modelo1_regresion_lineal.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Figura guardada: modelo1_regresion_lineal.png")

# Gráfico 2: Importancia global de variables (coeficientes estandarizados)
coef_df = pd.DataFrame({
    'variable': feature_cols,
    'coeficiente': lr_model.coef_
}).sort_values('coeficiente', key=abs, ascending=True)

fig, ax = plt.subplots(figsize=(8, 7))
colors = ['#EA4335' if c < 0 else '#34A853' for c in coef_df['coeficiente']]
ax.barh(coef_df['variable'], coef_df['coeficiente'], color=colors, alpha=0.85)
ax.axvline(0, color='gray', linewidth=0.8)
ax.set_title('Modelo 1 — Importancia global de variables (coeficientes estandarizados)',
             fontweight='bold')
ax.set_xlabel('Coeficiente (impacto en el ahorro)')
plt.tight_layout()
plt.savefig(f"{CLEAN_DIR}/modelo1_coeficientes.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Figura guardada: modelo1_coeficientes.png")

# Gráfico 3: interpretabilidad LOCAL — se explica una predicción concreta
# (el primer caso del conjunto de test) descomponiendo la contribución
# de cada variable, en lugar de limitarse a la importancia global.
detalle_local, pred_reconstruida = explicacion_local(
    lr_model.coef_, lr_model.intercept_, X_test_s[0], feature_cols)
usuario_ejemplo = int(test_df.iloc[0]['user_id'])
mes_ejemplo = test_df.iloc[0]['fecha']
graficar_explicacion_local(
    detalle_local,
    f'Modelo 1 — Explicación local (usuario {usuario_ejemplo}, {mes_ejemplo})',
    f"{CLEAN_DIR}/modelo1_explicacion_local_ejemplo.png")
print(f"  Figura guardada: modelo1_explicacion_local_ejemplo.png "
      f"(predicción reconstruida: {pred_reconstruida:.2f} € "
      f"vs. modelo: {y_pred_lr[0]:.2f} €)")


# ============================================================
# MODELO 1-BIS — PROYECCIÓN INDIVIDUAL POR USUARIO
# Objetivo: usando el Modelo 1 ya entrenado, generar para cada
# usuario una predicción del ahorro de "el mes siguiente al último
# mes disponible en sus datos", a partir de su propio histórico
# reciente. Esta tabla es la que alimenta la página de KPIs por
# usuario (una fila = un usuario = su previsión individual).
# ============================================================

print("\n" + "=" * 60)
print("MODELO 1-BIS — Proyección individual del ahorro (por usuario)")
print("=" * 60)

ultimo_mes_por_usuario = (
    df_model.sort_values(['user_id', 'fecha_dt'])
    .groupby('user_id')
    .tail(1)
    .reset_index(drop=True)
)
# Estas filas usan como predictores los rezagos de los 3 últimos meses
# reales de cada usuario, y devuelven una predicción para "el mes
# siguiente" (fuera de la muestra, sin ahorro real todavía observado).
X_usuarios = ultimo_mes_por_usuario[feature_cols]
X_usuarios_s = scaler.transform(X_usuarios)
ahorro_predicho_usuario = lr_model.predict(X_usuarios_s)

tabla_kpi_usuarios = pd.DataFrame({
    'user_id': ultimo_mes_por_usuario['user_id'],
    'ultimo_mes_disponible': ultimo_mes_por_usuario['fecha'],
    'perfil': ultimo_mes_por_usuario['perfil'],
    'ahorro_ultimo_mes': ultimo_mes_por_usuario['ahorro'],
    'ahorro_predicho_mes_siguiente': ahorro_predicho_usuario.round(2),
})
tabla_kpi_usuarios['variacion_esperada_pct'] = (
    (tabla_kpi_usuarios['ahorro_predicho_mes_siguiente']
     - tabla_kpi_usuarios['ahorro_ultimo_mes'])
    / tabla_kpi_usuarios['ahorro_ultimo_mes'].replace(0, np.nan) * 100
).round(2)

tabla_kpi_usuarios.to_csv(f"{CLEAN_DIR}/kpi_usuarios_proyeccion.csv", index=False)
print(f"  Tabla de KPIs por usuario guardada: kpi_usuarios_proyeccion.csv "
      f"({len(tabla_kpi_usuarios):,} usuarios)")

# Ejemplo ilustrativo: explicación local para un usuario cualquiera de
# la tabla, pensada para la página individual del usuario en la app.
idx_ejemplo = 0
detalle_usuario, _ = explicacion_local(
    lr_model.coef_, lr_model.intercept_, X_usuarios_s[idx_ejemplo], feature_cols)
uid_ejemplo = int(tabla_kpi_usuarios.iloc[idx_ejemplo]['user_id'])
graficar_explicacion_local(
    detalle_usuario,
    f'Proyección individual — Explicación local (usuario {uid_ejemplo})',
    f"{CLEAN_DIR}/modelo1bis_explicacion_usuario_ejemplo.png")
print(f"  Figura guardada: modelo1bis_explicacion_usuario_ejemplo.png")


# ============================================================
# MODELO 2 — REGRESIÓN LOGÍSTICA (reformulación temporal)
# Objetivo: clasificar el perfil de ahorro del mes t a partir del
#           histórico t-1, t-2, t-3 del propio usuario
# Clases: buen_ahorrador / ahorro_moderado / ahorro_insuficiente
# ============================================================

print("\n" + "=" * 60)
print("MODELO 2 — Regresión Logística: clasificación del perfil de ahorro")
print("(a partir del histórico de los 3 meses anteriores del usuario)")
print("=" * 60)

target_clf = 'perfil_ahorro'

# Se reutiliza la misma partición por usuario que en el Modelo 1 para
# mantener coherencia entre ambos modelos (mismos usuarios en test).
train_df2, test_df2 = train_df, test_df

X2_train, y2_train = train_df2[feature_cols], train_df2[target_clf]
X2_test, y2_test = test_df2[feature_cols], test_df2[target_clf]

scaler2 = StandardScaler()
X2_train_s = scaler2.fit_transform(X2_train)
X2_test_s = scaler2.transform(X2_test)

clf_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                class_weight='balanced')
clf_model.fit(X2_train_s, y2_train)
y2_pred = clf_model.predict(X2_test_s)

acc = accuracy_score(y2_test, y2_pred)
print(f"\n  Filas train: {len(X2_train):,} | Filas test: {len(X2_test):,}")
print(f"  Accuracy: {acc:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y2_test, y2_pred))

# Gráfico 4: Matriz de confusión
fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y2_test, y2_pred, labels=clf_model.classes_)
disp = ConfusionMatrixDisplay(cm, display_labels=clf_model.classes_)
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Modelo 2 — Matriz de confusión (clasificación perfil ahorro)',
             fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CLEAN_DIR}/modelo2_confusion.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Figura guardada: modelo2_confusion.png")

# Gráfico 5: interpretabilidad global — coeficientes por clase (one-vs-rest)
fig, axes = plt.subplots(1, len(clf_model.classes_), figsize=(6 * len(clf_model.classes_), 6),
                          sharey=True)
for ax_i, clase in zip(np.atleast_1d(axes), clf_model.classes_):
    idx_clase = list(clf_model.classes_).index(clase)
    coefs_clase = clf_model.coef_[idx_clase]
    orden = np.argsort(coefs_clase)
    ax_i.barh(np.array(feature_cols)[orden], coefs_clase[orden],
              color=['#EA4335' if c < 0 else '#34A853' for c in coefs_clase[orden]],
              alpha=0.85)
    ax_i.axvline(0, color='gray', linewidth=0.8)
    ax_i.set_title(f'Clase: {clase}', fontweight='bold')
plt.suptitle('Modelo 2 — Importancia global de variables por clase', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CLEAN_DIR}/modelo2_coeficientes_por_clase.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Figura guardada: modelo2_coeficientes_por_clase.png")


# ============================================================
# MODELO 3 — SERIE TEMPORAL (agregada, sin cambios de fondo)
# Objetivo: proyectar el ahorro medio de la población de usuarios
#           para los próximos 6 meses.
# Método: regresión lineal sobre tendencia + estacionalidad
#         (simplificado, sin Prophet para evitar dependencias Stan)
#
# Nota: este modelo trabaja sobre el ahorro medio AGREGADO de todos
# los usuarios por mes, no sobre el ahorro individual de cada usuario
# en un mismo mes, por lo que no está sujeto al problema de fuga de
# información de los Modelos 1 y 2 (no existe una identidad contable
# entre el ahorro medio de un mes y su propia serie histórica). La
# vista por usuario individual la aporta el Modelo 1-bis.
# ============================================================

print("\n" + "=" * 60)
print("MODELO 3 — Serie Temporal: proyección del ahorro medio (6 meses)")
print("=" * 60)

ahorro_medio = (df_raw.groupby('fecha')['ahorro']
                .mean()
                .reset_index()
                .rename(columns={'ahorro': 'ahorro_medio'}))
ahorro_medio['fecha_dt'] = pd.to_datetime(ahorro_medio['fecha'])
ahorro_medio = ahorro_medio.sort_values('fecha_dt').reset_index(drop=True)
ahorro_medio['t'] = range(len(ahorro_medio))
ahorro_medio['mes_num'] = ahorro_medio['fecha_dt'].dt.month
ahorro_medio['sin_mes'] = np.sin(2 * np.pi * ahorro_medio['mes_num'] / 12)
ahorro_medio['cos_mes'] = np.cos(2 * np.pi * ahorro_medio['mes_num'] / 12)

X_ts = ahorro_medio[['t', 'sin_mes', 'cos_mes']]
y_ts = ahorro_medio['ahorro_medio']

ts_model = LinearRegression()
ts_model.fit(X_ts, y_ts)

ultima_fecha = ahorro_medio['fecha_dt'].max()
fechas_futuras = pd.date_range(ultima_fecha + pd.DateOffset(months=1), periods=6, freq='MS')
t_futuro = range(len(ahorro_medio), len(ahorro_medio) + 6)
meses_futuros = fechas_futuras.month

X_futuro = pd.DataFrame({
    't': list(t_futuro),
    'sin_mes': np.sin(2 * np.pi * meses_futuros / 12),
    'cos_mes': np.cos(2 * np.pi * meses_futuros / 12),
})

predicciones = ts_model.predict(X_futuro)

df_pred = pd.DataFrame({
    'fecha': [f.strftime('%Y-%m') for f in fechas_futuras],
    'ahorro_predicho': predicciones.round(2),
    'tipo': 'prediccion'
})

print(f"\n  Proyección ahorro medio mensual "
      f"({fechas_futuras[0].strftime('%b %Y')}–{fechas_futuras[-1].strftime('%b %Y')}):")
for _, row in df_pred.iterrows():
    print(f"  {row['fecha']}: {row['ahorro_predicho']:.2f} €")

# Gráfico 6: Serie histórica + proyección
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(ahorro_medio['fecha_dt'], ahorro_medio['ahorro_medio'],
        color='#1A73E8', linewidth=2, marker='o', markersize=3,
        label='Ahorro medio histórico')
y_fitted = ts_model.predict(X_ts)
ax.plot(ahorro_medio['fecha_dt'], y_fitted,
        color='#34A853', linewidth=1.5, linestyle='--', label='Tendencia del modelo')
ax.plot(fechas_futuras, predicciones,
        color='#EA4335', linewidth=2.5, marker='s', markersize=6,
        linestyle='--', label='Proyección 6 meses')
ax.axvspan(fechas_futuras[0], fechas_futuras[-1],
           alpha=0.08, color='#EA4335', label='Período proyectado')
ax.axvline(ultima_fecha, color='gray', linestyle=':', linewidth=1.5)
ax.annotate(f'Último dato\n({ultima_fecha.strftime("%b %Y")})',
            xy=(ultima_fecha, ahorro_medio['ahorro_medio'].iloc[-1]),
            xytext=(15, 10), textcoords='offset points', fontsize=8, color='gray')
ax.set_title('Modelo 3 — Proyección del ahorro medio mensual (6 meses)', fontweight='bold')
ax.set_xlabel('Fecha')
ax.set_ylabel('Ahorro medio (€)')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{CLEAN_DIR}/modelo3_proyeccion.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Figura guardada: modelo3_proyeccion.png")


# ── GUARDAR MÉTRICAS ─────────────────────────────────────────
metricas = {
    'modelo1_mae': round(mae_lr, 2),
    'modelo1_r2': round(r2_lr, 4),
    'modelo1_n_features': len(feature_cols),
    'modelo1_features': feature_cols,
    'modelo2_accuracy': round(acc, 4),
    'modelo2_clases': list(clf_model.classes_),
    'modelo3_predicciones': df_pred[['fecha', 'ahorro_predicho']].to_dict('records'),
    'particion': 'GroupShuffleSplit por user_id (20% usuarios en test)',
    'usuarios_train': int(train_df['user_id'].nunique()),
    'usuarios_test': int(test_df['user_id'].nunique()),
}
with open(f"{CLEAN_DIR}/metricas_modelos.json", 'w') as f:
    json.dump(metricas, f, indent=2)

# ── GUARDAR DATASETS ─────────────────────────────────────────
df_model.to_csv(f"{CLEAN_DIR}/dataset_final_usuarios.csv", index=False)
ahorro_medio.to_csv(f"{CLEAN_DIR}/ahorro_medio_mensual.csv", index=False)

print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"  Modelo 1 — Regresión Lineal (histórico t-1..t-3):")
print(f"    MAE = {mae_lr:.2f}€ | R² = {r2_lr:.4f}")
print(f"  Modelo 1-bis — Proyección individual: "
      f"{len(tabla_kpi_usuarios):,} usuarios")
print(f"  Modelo 2 — Regresión Logística (histórico t-1..t-3):")
print(f"    Accuracy = {acc:.4f}")
print(f"  Modelo 3 — Proyección temporal agregada:")
print(f"    Ahorro proyectado {df_pred.iloc[0]['fecha']}: {predicciones[0]:.2f}€")
print(f"    Ahorro proyectado {df_pred.iloc[-1]['fecha']}: {predicciones[-1]:.2f}€")
print(f"\n  Archivos guardados en: {CLEAN_DIR}")
