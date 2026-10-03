# ============================================================
# modelos_predictivos.py
# Modelos de Machine Learning — AI Financial Life Coach
#
# Modelo 1: Regresión lineal regularizada (Ridge / Lasso) — ahorro mensual
# Modelo 2: Regresión logística — clasificación binaria del perfil de ahorro
# Modelo avanzado (rúbrica 4.3): ensemble HistGradientBoosting, contraste
#           de los Modelos 1 y 2 frente a un modelo no lineal
# Modelo 3: Serie temporal — proyección del ahorro agregado (6 meses),
#           con backtest frente a pronósticos ingenuos
# Modelo 1-bis: Proyección individual — ahorro del mes siguiente por usuario
#
# ------------------------------------------------------------
# NOTA METODOLÓGICA
# ------------------------------------------------------------
# Los Modelos 1 y 2 predicen el mes t a partir de información
# disponible ANTES de ese mes (rezagos y medias móviles de los meses
# t-1, t-2, t-3), en lugar de utilizar como predictores las propias
# componentes del mismo mes que definen algebraicamente el objetivo
# (ahorro = salario - gasto_total). Esta reformulación temporal evita
# la fuga de información (data leakage).
#
# Validación: partición train/test por usuario (GroupShuffleSplit,
# 20% de usuarios en test) y, dentro del conjunto de entrenamiento,
# ajuste de hiperparámetros mediante GridSearchCV con validación
# cruzada agrupada por usuario (GroupKFold, 5 particiones): ningún
# usuario aparece a la vez en los pliegues de entrenamiento y
# validación, ni en train y test. Cada modelo se compara además con
# pronósticos ingenuos (baselines) sobre el mismo conjunto de test.
#
# El preprocesamiento (codificación de 'perfil', evaluación de
# asimetría y transformación logarítmica, escalado) está documentado
# y justificado empíricamente en preprocesamiento.py (Asignatura 7,
# apartado 4.1); este script importa y reutiliza esas mismas
# funciones para garantizar que el modelado usa exactamente el
# preprocesamiento allí descrito, sin lógica duplicada.
# ============================================================

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                   Ridge, Lasso)
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_absolute_error, r2_score,
                              classification_report, accuracy_score,
                              confusion_matrix, ConfusionMatrixDisplay)

from preprocesamiento import (codificar_perfil_onehot, aplicar_log_transform,
                               decidir_variables_a_transformar)

sns.set_style("whitegrid")
sns.set_palette("Set2")

DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean')
CLEAN_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean')
os.makedirs(CLEAN_DIR, exist_ok=True)

RANDOM_STATE = 42
CATEGORIAS = ['vivienda', 'alimentacion', 'transporte', 'ocio',
              'salud', 'educacion', 'otros']

# Codificación de 'perfil': one-hot encoding (ver preprocesamiento.py),
# no ordinal — 'perfil' es una categórica nominal sin orden natural.
# Columnas dummy resultantes (junior es la categoría base, omitida por
# drop_first=True en codificar_perfil_onehot).
PERFIL_ONEHOT_COLS = ['perfil_medio', 'perfil_senior', 'perfil_freelance']

# Umbral de perfil_ahorro: se lee del JSON generado por
# 06_prepara_usuarios_tink.py (mediana empírica de este dataset) para
# mantener coherencia con el target del Modelo 2. Fallback a un umbral
# absoluto clásico (15%) si el archivo no existe.
#
# Esquema binario (2 clases: ahorro_insuficiente / ahorro_adecuado) por
# mediana, en lugar del esquema original de terciles (3 clases): se
# verifico empiricamente que los terciles dejaban la clase central en
# una franja demasiado estrecha (~4 puntos porcentuales) para ser
# separable con las features disponibles (recall 0.02 en esa clase).
# El esquema binario elimina ese problema estructural -- un unico
# limite de decision en vez de dos -- y mejora el F1 macro de 0.43 a
# 0.72 (ver docstring de compute_saving_profile_threshold() en
# 06_prepara_usuarios_tink.py para el historial completo de esta
# decision, incluyendo las alternativas de 3 clases evaluadas y
# descartadas).
_THRESHOLDS_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'clean', 'perfil_ahorro_thresholds_tink.json'
)
if os.path.exists(_THRESHOLDS_PATH):
    with open(_THRESHOLDS_PATH) as _f:
        _thresholds = json.load(_f)
    UMBRAL_MEDIANA_AHORRO = _thresholds['umbral_mediana_pct']
    print(f"Umbral perfil_ahorro cargado de {_THRESHOLDS_PATH}: "
          f"mediana={UMBRAL_MEDIANA_AHORRO}% (esquema binario)")
else:
    UMBRAL_MEDIANA_AHORRO = 15.0
    print("Umbral perfil_ahorro: no se encontro JSON de umbral relativo, "
          f"usando umbral absoluto clasico ({UMBRAL_MEDIANA_AHORRO}%)")


# ============================================================
# INGESTA Y ARMONIZACIÓN DE ESQUEMA
# ============================================================

def cargar_dataset(path):
    """Carga un dataset mensual por usuario con el esquema estándar de
    17 columnas (sintético o derivado de Tink vía prepara_usuarios_tink.py)
    y homogeneiza tipos para que ambas fuentes sean intercambiables.

    La codificación de 'perfil' se realiza mediante one-hot encoding
    (ver preprocesamiento.codificar_perfil_onehot): es una variable
    categórica nominal sin orden natural entre sus categorías, por lo
    que una codificación ordinal induciría una relación de magnitud
    inexistente en los datos."""
    df = pd.read_csv(path)
    df, _ = codificar_perfil_onehot(df)

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
    edad, perfil (codificado one-hot) e ipc (rezagado un mes, ya que
    el IPC de un mes se publica con retraso y no está disponible al
    inicio de éste).

    De este modo ningún predictor del mes t puede reconstruir
    algebraicamente el ahorro de ese mismo mes.

    'salud' y 'educacion' se excluyen de las features del modelo (no
    de CATEGORIAS en general, que sigue usándose para la capa
    prescriptiva): tienen desviación estándar 0 en la totalidad del
    dataset (639.433 filas), es decir, ninguna transacción de estas
    categorías está presente en los datos derivados de Tink —
    consistente con la limitación ya documentada en
    sgbd/docs/tink_mapping.md sobre la categorización de
    transacciones aún pendiente. Incluir una columna constante como
    predictor no aporta información y solo añade ruido a la matriz
    de diseño.

    Salario y gasto_total (y sus rezagos/medias móviles) se incluyen
    en su versión log1p: preprocesamiento.py evalúa su asimetría
    (skew salario=1.06, skew gasto_total=13.01) y, superando el
    umbral de |skew|>1, confirma empíricamente (comparación de MAE/R²
    del Modelo 1) que la transformación logarítmica mejora el ajuste
    frente a las variables originales. 'ahorro' se deja sin
    transformar: contiene valores negativos y es la variable
    objetivo del Modelo 1, donde se prioriza la interpretabilidad de
    los coeficientes en euros.
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

    # Transformación logarítmica (justificada empíricamente en
    # preprocesamiento.py) de salario y gasto_total, aplicada también
    # a sus rezagos/medias móviles ya construidos arriba.
    _, variables_a_transformar = decidir_variables_a_transformar(df)
    df, _ = aplicar_log_transform(df, variables_a_transformar)

    # Categorías de gasto: patrón reciente (un rezago es suficiente para
    # capturar el mix de consumo sin disparar la dimensionalidad)
    for cat in CATEGORIAS:
        df[f'{cat}_lag1'] = g[cat].shift(1)

    categorias_con_varianza = [c for c in CATEGORIAS if c not in ('salud', 'educacion')]

    feature_cols = (
        ['edad'] + PERFIL_ONEHOT_COLS +
        ['ipc_lag1',
         'ahorro_lag1', 'ahorro_lag2', 'ahorro_lag3', 'ahorro_roll3',
         'salario_lag1_log', 'salario_roll3_log',
         'gasto_total_lag1_log', 'gasto_total_roll3_log']
        + [f'{cat}_lag1' for cat in categorias_con_varianza]
    )

    return df, feature_cols


def construir_fila_futura(df_raw):
    """Construye, para cada usuario, una fila 'virtual' correspondiente
    al mes inmediatamente posterior al último mes disponible en sus
    datos (t = T+1), con todas las variables de resultado del propio
    mes (ahorro, salario, gasto por categoría, tasa de ahorro, perfil
    de ahorro, ipc) puestas a NaN, ya que ese mes todavía no ha
    ocurrido.

    Esta fila se concatena con el histórico real y se pasa por
    build_lag_features(): al calcular los rezagos con groupby+shift,
    la fila virtual hereda automáticamente como 'lag1' los valores
    REALES del último mes conocido (T), como 'lag2' los de T-1, etc.
    De esta forma la proyección a T+1 usa exactamente los mismos 3
    meses de histórico que el modelo usó en entrenamiento, y no
    reproduce una estimación del propio mes T (que ya es conocido).
    """
    ultima_fila = (df_raw.sort_values(['user_id', 'fecha_dt'])
                   .groupby('user_id').tail(1).copy())
    ultima_fila['fecha_dt'] = ultima_fila['fecha_dt'] + pd.DateOffset(months=1)
    ultima_fila['fecha'] = ultima_fila['fecha_dt'].dt.strftime('%Y-%m')

    columnas_resultado = (['ahorro', 'salario', 'gasto_total',
                            'tasa_ahorro_pct', 'perfil_ahorro', 'ipc_mensual']
                          + CATEGORIAS)
    for col in columnas_resultado:
        ultima_fila[col] = np.nan

    ultima_fila['es_pronostico'] = True
    return ultima_fila


def separar_reales_y_pronostico(df_con_lags, feature_cols):
    """A partir del dataframe con rezagos ya calculados (histórico real
    + fila virtual T+1 por usuario), separa:
      - df_entrenable: meses reales con objetivo conocido y con
        histórico suficiente (se descartan los 3 primeros meses de
        cada usuario), listo para entrenar/evaluar.
      - df_pronostico: la fila T+1 de cada usuario, lista para generar
        la proyección individual (Modelo 1-bis), una vez el modelo
        esté entrenado.
    """
    es_pronostico = df_con_lags.get('es_pronostico', False)
    if not isinstance(es_pronostico, pd.Series):
        es_pronostico = pd.Series(False, index=df_con_lags.index)
    es_pronostico = es_pronostico.fillna(False).astype(bool)

    df_pronostico = df_con_lags.loc[es_pronostico].dropna(subset=feature_cols).reset_index(drop=True)
    df_entrenable = (df_con_lags.loc[~es_pronostico]
                      .dropna(subset=feature_cols)
                      .reset_index(drop=True))
    return df_entrenable, df_pronostico


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


def build_chained_savings_index(df):
    """Construye una serie mensual de ahorro medio robusta a los cambios
    de composicion de la poblacion activa mes a mes.

    Motivo: el numero de usuarios activos varia entre ~5.000 y ~30.000
    segun el mes (verificado empiricamente), porque cada uno de los 8
    perfiles reales de origen Tink cubre una ventana temporal distinta
    (desde 2 hasta 52 meses), y los usuarios sinteticos heredan la
    ventana de su perfil de origen. La media transversal simple
    (groupby('fecha')['ahorro'].mean()) confunde dos efectos distintos:
    el cambio real de comportamiento financiero mes a mes, y el cambio
    de QUIEN esta siendo promediado ese mes. Esto produce saltos
    artificiales en la serie (verificado: un salto de 2.783€ a 4.653€
    coincide exactamente con la entrada de un nuevo bloque de ~5.000
    usuarios, no con un cambio de comportamiento real).

    Un panel balanceado (mismos usuarios en todos los meses) no es
    viable aqui: la interseccion de las ventanas temporales de los 8
    perfiles de origen esta vacia (no hay un solo mes cubierto por
    todos). En su lugar se usa un indice encadenado (chain-linked
    index), la misma tecnica empleada en economia para indices con
    cesta cambiante (p. ej. 'same-store sales' en retail, o indices de
    precios con productos que entran y salen de la cesta): para cada
    par de meses consecutivos (t-1, t) se calcula la tasa de variacion
    SOLO sobre los usuarios presentes en AMBOS meses (el subconjunto
    estable de esa transicion concreta), y esa tasa se encadena a
    partir del nivel real del primer mes. El resultado esta en las
    mismas unidades (€) que la serie original y sigue usando el 100%
    de los usuarios disponibles en cada transicion (verificado: ningun
    par de meses consecutivos tiene cero usuarios en comun; minimo
    5.066), pero ya no arrastra saltos causados por la entrada o
    salida de cohortes completas."""
    df = df.sort_values('fecha')
    months = sorted(df['fecha'].unique())
    users_by_month = {m: set(df.loc[df['fecha'] == m, 'user_id']) for m in months}
    ahorro_indexed = df.set_index(['fecha', 'user_id'])['ahorro']

    nivel = float(df.loc[df['fecha'] == months[0], 'ahorro'].mean())
    filas = [{'fecha': months[0], 'ahorro_medio_encadenado': round(nivel, 2),
              'n_usuarios_comunes': None}]

    for i in range(1, len(months)):
        mes_prev, mes_actual = months[i - 1], months[i]
        comunes = list(users_by_month[mes_prev] & users_by_month[mes_actual])
        media_prev = ahorro_indexed.loc[mes_prev].loc[comunes].mean()
        media_actual = ahorro_indexed.loc[mes_actual].loc[comunes].mean()
        tasa_variacion = (media_actual / media_prev) if media_prev else 1.0
        nivel = nivel * tasa_variacion
        filas.append({
            'fecha': mes_actual,
            'ahorro_medio_encadenado': round(nivel, 2),
            'n_usuarios_comunes': len(comunes),
        })

    return pd.DataFrame(filas)


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
df_raw = cargar_dataset(f"{DATA_DIR}/dataset_final_usuarios_tink.csv")
print(f"\nDataset: {df_raw.shape[0]:,} registros | {df_raw['user_id'].nunique()} usuarios")
print(f"Período: {df_raw['fecha'].min()} — {df_raw['fecha'].max()}")

fila_futura = construir_fila_futura(df_raw)
df_extendido = pd.concat([df_raw, fila_futura], ignore_index=True, sort=False)
df_con_lags, feature_cols = build_lag_features(df_extendido)
df_model, df_pronostico_base = separar_reales_y_pronostico(df_con_lags, feature_cols)

print(f"Registros aptos para modelado tras construir rezagos: {df_model.shape[0]:,} "
      f"(se descartan los 3 primeros meses de cada usuario por falta de histórico)")
print(f"Filas de proyección a T+1 construidas: {df_pronostico_base.shape[0]:,} usuarios")


# ============================================================
# AJUSTE DE HIPERPARÁMETROS CON VALIDACIÓN CRUZADA AGRUPADA
# ============================================================

N_FOLDS_CV = 5

# Paralelismo de la búsqueda de hiperparámetros. Por defecto 1 (secuencial):
# con n_jobs=-1 joblib lanza un proceso por núcleo y cada uno carga su propia
# copia de los ~420.000 registros de entrenamiento, lo que agota la memoria en
# equipos con muchos núcleos y poca RAM libre. Los resultados son idénticos con
# cualquier valor; solo cambia el tiempo de ejecución. Puede aumentarse con la
# variable de entorno AFILC_N_JOBS (p. ej. AFILC_N_JOBS=2).
N_JOBS = int(os.environ.get('AFILC_N_JOBS', '1'))


def ajustar_hiperparametros(candidatos, X, y, groups, scoring):
    """Selecciona modelo e hiperparámetros mediante GridSearchCV con
    validación cruzada agrupada por usuario (GroupKFold).

    candidatos: dict {nombre: (estimador, rejilla_de_parametros)}.
    Los estimadores lineales se envuelven en un Pipeline con
    StandardScaler, de modo que el escalador se ajusta dentro de cada
    pliegue y no usa estadísticos de los pliegues de validación.

    Se usa GroupKFold (y no un KFold simple) porque cada usuario
    aporta varios meses: con un KFold simple, meses del mismo usuario
    caerían a la vez en entrenamiento y validación, y la puntuación de
    validación (y por tanto la elección de hiperparámetros) quedaría
    sesgada al alza.

    Devuelve (nombre_ganador, mejores_parametros, tabla_cv), donde
    tabla_cv recoge la puntuación media y la desviación típica entre
    pliegues de cada combinación evaluada.
    """
    gkf = GroupKFold(n_splits=N_FOLDS_CV)
    filas, mejor = [], None
    for nombre, (estimador, rejilla) in candidatos.items():
        n_jobs = 1 if 'HGB' in nombre else N_JOBS  # HGB ya paraleliza internamente
        gs = GridSearchCV(estimador, rejilla, cv=gkf, scoring=scoring,
                          n_jobs=n_jobs, refit=False)
        gs.fit(X, y, groups=groups)
        res = gs.cv_results_
        for params, media, std in zip(res['params'], res['mean_test_score'],
                                      res['std_test_score']):
            filas.append({'modelo': nombre,
                          'parametros': {k.replace('m__', ''): v for k, v in params.items()},
                          'cv_media': float(media), 'cv_std': float(std)})
            print(f"    {nombre:<10} {str({k.replace('m__', ''): v for k, v in params.items()}):<55}"
                  f" cv={media:.4f} ± {std:.4f}")
        if mejor is None or gs.best_score_ > mejor[2]:
            mejor = (nombre, {k.replace('m__', ''): v for k, v in gs.best_params_.items()},
                     gs.best_score_)
    return mejor[0], mejor[1], pd.DataFrame(filas)


# ============================================================
# MODELO 1 — REGRESIÓN LINEAL (reformulación temporal)
# Objetivo: predecir el ahorro del mes t a partir del histórico
#           t-1, t-2, t-3 del propio usuario (sin usar componentes
#           del propio mes t)
# ============================================================

print("\n" + "=" * 60)
print("MODELO 1 — Regresión lineal regularizada: predicción del ahorro mensual")
print("(a partir del histórico de los 3 meses anteriores del usuario)")
print("=" * 60)

target_lr = 'ahorro'

train_df, test_df = split_por_usuario(df_model, test_size=0.2)

X_train, y_train = train_df[feature_cols], train_df[target_lr]
X_test, y_test = test_df[feature_cols], test_df[target_lr]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"\n  Usuarios train: {train_df['user_id'].nunique():,} | "
      f"Usuarios test: {test_df['user_id'].nunique():,}")
print(f"  Filas  train: {len(X_train):,} | Filas  test: {len(X_test):,}")

# Ajuste de hiperparámetros: se comparan dos regularizaciones lineales
# (Ridge, L2; Lasso, L1) con GroupKFold sobre el conjunto de
# entrenamiento. Ambas conservan la interpretabilidad lineal que exige
# la capa prescriptiva (descomposición coeficiente × valor
# estandarizado, explicacion_local()). Lasso se incluye además porque,
# al poder anular coeficientes, podría resolver la colinealidad entre
# salario_lag1_log y salario_roll3_log (r=0,96) descartando una de las
# dos variables; el resultado se documenta más abajo.
print("\n  Ajuste de hiperparámetros (GridSearchCV + GroupKFold, métrica: MAE):")
candidatos_m1 = {
    'Ridge': (Pipeline([('sc', StandardScaler()), ('m', Ridge())]),
              {'m__alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000]}),
    'Lasso': (Pipeline([('sc', StandardScaler()), ('m', Lasso(max_iter=5000))]),
              {'m__alpha': [0.1, 1, 5, 10]}),
}
nombre_m1, params_m1, cv_m1 = ajustar_hiperparametros(
    candidatos_m1, X_train, y_train, train_df['user_id'], 'neg_mean_absolute_error')
print(f"  Seleccionado: {nombre_m1} {params_m1} "
      f"(MAE CV = {-cv_m1['cv_media'].max():.2f} €)")

ClaseM1 = Ridge if nombre_m1 == 'Ridge' else Lasso
lr_model = ClaseM1(alpha=params_m1['alpha'],
                   **({'max_iter': 5000} if nombre_m1 == 'Lasso' else {}))
lr_model.fit(X_train_s, y_train)
y_pred_lr = lr_model.predict(X_test_s)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"  MAE test:  {mae_lr:.2f} € (error medio absoluto)")
print(f"  R² test:   {r2_lr:.4f} (varianza explicada)")

# Baselines (pronósticos ingenuos sin modelo) sobre el mismo test:
# imprescindibles para valorar cuánto aporta realmente el modelo.
baselines_m1 = {
    'Persistencia (ahorro del mes anterior)': test_df['ahorro_lag1'],
    'Media móvil de 3 meses (ahorro_roll3)': test_df['ahorro_roll3'],
    'Media del conjunto de entrenamiento': np.full(len(test_df), y_train.mean()),
}
baselines_m1_out = {}
print("\n  Baselines sobre el mismo conjunto de test:")
for nombre_b, pred_b in baselines_m1.items():
    mae_b, r2_b = mean_absolute_error(y_test, pred_b), r2_score(y_test, pred_b)
    baselines_m1_out[nombre_b] = {'mae': round(float(mae_b), 2), 'r2': round(float(r2_b), 4)}
    print(f"    {nombre_b:<42} MAE {mae_b:8.2f} €  R² {r2_b:.4f}")

# Colinealidad salario_lag1_log / salario_roll3_log (r=0,96): se
# comparan los coeficientes de OLS sin regularizar y del modelo
# seleccionado, y se mide su estabilidad entre los 5 pliegues de la
# validación cruzada agrupada. Resultado documentado en la memoria:
# la regularización elegida por validación cruzada apenas modifica los
# coeficientes (la validación cruzada optimiza el error de predicción,
# no la interpretabilidad), pero su signo y magnitud son estables entre
# pliegues: no se trata de inestabilidad numérica, sino de un efecto
# condicional (de supresión) de dos variables casi redundantes.
vars_salario = ['salario_lag1_log', 'salario_roll3_log']
idx_sal = [feature_cols.index(v) for v in vars_salario]
ols_ref = LinearRegression().fit(X_train_s, y_train)
coefs_pliegues = []
for tr_idx, _ in GroupKFold(n_splits=N_FOLDS_CV).split(X_train, y_train, train_df['user_id']):
    sc_f = StandardScaler().fit(X_train.iloc[tr_idx])
    m_f = ClaseM1(alpha=params_m1['alpha'],
                  **({'max_iter': 5000} if nombre_m1 == 'Lasso' else {}))
    m_f.fit(sc_f.transform(X_train.iloc[tr_idx]), y_train.iloc[tr_idx])
    coefs_pliegues.append(m_f.coef_[idx_sal])
coefs_pliegues = np.array(coefs_pliegues)
estabilidad_coef = {}
print("\n  Colinealidad — coeficientes de las variables de salario:")
for j, v in enumerate(vars_salario):
    estabilidad_coef[v] = {
        'ols': round(float(ols_ref.coef_[idx_sal[j]]), 1),
        'modelo_seleccionado': round(float(lr_model.coef_[idx_sal[j]]), 1),
        'media_pliegues': round(float(coefs_pliegues[:, j].mean()), 1),
        'std_pliegues': round(float(coefs_pliegues[:, j].std()), 1),
    }
    print(f"    {v:<20} OLS {estabilidad_coef[v]['ols']:>9.1f} | "
          f"{nombre_m1} {estabilidad_coef[v]['modelo_seleccionado']:>9.1f} | "
          f"pliegues {estabilidad_coef[v]['media_pliegues']:.1f} ± {estabilidad_coef[v]['std_pliegues']:.1f}")
corr_sal = float(train_df[vars_salario].corr().iloc[0, 1])
corr_roll_ahorro = float(train_df[['salario_roll3_log', 'ahorro']].corr().iloc[0, 1])

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
ax.set_title(f'Modelo 1 — Regresión lineal ({nombre_m1}): ahorro real vs. predicho',
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
ax.set_title(f'Modelo 1 ({nombre_m1}) — Importancia global de variables\n'
             '(coeficientes estandarizados)', fontweight='bold')
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
# usuario una predicción genuina de "fuera de muestra" del ahorro
# del mes siguiente al último mes disponible en sus datos (T+1),
# a partir de los 3 meses reales más recientes de su histórico.
# Esta tabla es la que alimenta la página de KPIs por usuario (una
# fila = un usuario = su previsión individual).
# ============================================================

print("\n" + "=" * 60)
print("MODELO 1-BIS — Proyección individual del ahorro (por usuario, mes T+1)")
print("=" * 60)

X_usuarios = df_pronostico_base[feature_cols]
X_usuarios_s = scaler.transform(X_usuarios)
ahorro_predicho_usuario = lr_model.predict(X_usuarios_s)

# Salario estimado del mes T+1: a falta de un modelo específico de
# ingresos, se emplea como aproximación el salario del último mes real
# conocido (salario_lag1 de la fila de pronóstico). Se documenta como
# limitación: una línea de trabajo futura es sustituir esta
# aproximación por un modelo de ingresos propio.
salario_proxy = df_pronostico_base['salario_lag1']
tasa_predicha = (ahorro_predicho_usuario / salario_proxy.replace(0, np.nan) * 100).round(2)


def clasificar_saving_profile(tasa):
    """Umbral de perfil de ahorro (esquema binario). Se lee, si existe, de
    perfil_ahorro_thresholds_tink.json (mediana empirica calculada por
    06_prepara_usuarios_tink.py sobre este mismo dataset), para mantener
    coherencia con el target ya presente en los datos que entrena el
    Modelo 2. Si el archivo no existe (p. ej. al trabajar con el dataset
    sintetico original), se recurre al umbral absoluto clasico (15%)
    como fallback. Ver docstring de compute_saving_profile_threshold()
    en 06_prepara_usuarios_tink.py para la justificacion metodologica
    completa (incluye el historial de las versiones anteriores: umbral
    absoluto -> terciles -> binario por mediana)."""
    if pd.isna(tasa):
        return np.nan
    if tasa >= UMBRAL_MEDIANA_AHORRO:
        return 'ahorro_adecuado'
    return 'ahorro_insuficiente'


perfil_predicho = tasa_predicha.apply(clasificar_saving_profile)

tabla_kpi_usuarios = pd.DataFrame({
    'user_id': df_pronostico_base['user_id'],
    'ultimo_mes_real': df_pronostico_base['fecha_dt'] - pd.DateOffset(months=1),
    'mes_proyectado': df_pronostico_base['fecha'],
    'perfil': df_pronostico_base['perfil'],
    'ahorro_ultimo_mes_real': df_pronostico_base['ahorro_lag1'],
    'salario_proxy_mes_proyectado': salario_proxy.round(2),
    'ahorro_predicho_mes_proyectado': ahorro_predicho_usuario.round(2),
    'tasa_ahorro_predicha_pct': tasa_predicha,
    'perfil_ahorro_predicho': perfil_predicho,
})
tabla_kpi_usuarios['ultimo_mes_real'] = tabla_kpi_usuarios['ultimo_mes_real'].dt.strftime('%Y-%m')
tabla_kpi_usuarios['variacion_esperada_pct'] = (
    (tabla_kpi_usuarios['ahorro_predicho_mes_proyectado']
     - tabla_kpi_usuarios['ahorro_ultimo_mes_real'])
    / tabla_kpi_usuarios['ahorro_ultimo_mes_real'].replace(0, np.nan) * 100
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
# CAPA PRESCRIPTIVA — RECOMENDACIONES DE AJUSTE DE GASTO
#
# A partir de la proyección individual del Modelo 1-bis, esta capa
# traduce el diagnóstico predictivo ("¿cuánto ahorrará el usuario el
# mes que viene?") en una recomendación concreta y accionable ("¿qué
# categoría de gasto debería ajustar, y en qué medida, para mejorar su
# posición?"), cerrando el ciclo input → proceso (modelos predictivos)
# → output (recomendación prescriptiva) declarado en el objetivo
# general del proyecto.
#
# Diseño elegido: un motor de reglas transparente y trazable, en
# lugar de un modelo de optimización o de un nuevo modelo de caja
# negra, por dos motivos: (a) coherencia con el compromiso de
# interpretabilidad asumido para el proyecto, que exige poder explicar
# el motivo de cada recomendación en términos que un usuario final
# pueda entender; y (b) reutilización directa de la descomposición de
# contribuciones ya calculada para el Modelo 1 (interpretabilidad
# local), evitando introducir un componente adicional no explicable.
#
# Reglas:
#  1. Cada categoría de gasto se clasifica en un nivel de flexibilidad
#     (rígida / semi-flexible / flexible), siguiendo la distinción
#     habitual entre gasto esencial y discrecional en las encuestas de
#     presupuestos familiares utilizadas como referencia en este
#     trabajo: vivienda, salud y educación se consideran gastos
#     rígidos (no se recomienda ajustarlos); alimentación y transporte,
#     semi-flexibles; ocio y otros, flexibles.
#  2. El objetivo de ahorro no es un valor arbitrario: se reutiliza el
#     mismo umbral (mediana) ya definido para el perfil de ahorro
#     (esquema binario). Si la tasa de ahorro proyectada para el mes
#     siguiente sitúa al usuario en 'ahorro_insuficiente', se le
#     propone alcanzar ese umbral.
#  3. Para cerrar la brecha entre el ahorro proyectado y el ahorro
#     objetivo, se identifican las categorías flexibles y
#     semi-flexibles cuya contribución local (Modelo 1) más penaliza
#     el ahorro, y se recomienda una reducción de gasto acotada (máximo
#     20 % en categorías flexibles, 10 % en semi-flexibles, sobre el
#     gasto del último mes real) hasta cerrar la brecha o agotar las
#     categorías disponibles. El límite evita recomendaciones
#     irreales (p. ej. eliminar por completo una categoría).
# ============================================================

print("\n" + "=" * 60)
print("CAPA PRESCRIPTIVA — Recomendaciones de ajuste de gasto")
print("=" * 60)

FLEXIBILIDAD_CATEGORIAS = {
    'vivienda': 'rigida', 'salud': 'rigida', 'educacion': 'rigida',
    'alimentacion': 'semi_flexible', 'transporte': 'semi_flexible',
    'ocio': 'flexible', 'otros': 'flexible',
}
TOPE_REDUCCION = {'flexible': 0.20, 'semi_flexible': 0.10, 'rigida': 0.0}
CATEGORIAS_AJUSTABLES = [c for c, f in FLEXIBILIDAD_CATEGORIAS.items() if f != 'rigida']


def generar_recomendacion(fila_pronostico, x_std, coef, intercept, feature_names,
                           tasa_predicha, perfil_predicho):
    if perfil_predicho == 'ahorro_adecuado' or pd.isna(perfil_predicho):
        return {'requiere_ajuste': False, 'motivo': 'objetivo_ya_alcanzado', 'ajustes': []}

    objetivo_pct = UMBRAL_MEDIANA_AHORRO
    salario_proxy_fila = fila_pronostico['salario_lag1']
    ahorro_objetivo = objetivo_pct / 100 * salario_proxy_fila
    ahorro_predicho_fila = tasa_predicha / 100 * salario_proxy_fila
    brecha = round(ahorro_objetivo - ahorro_predicho_fila, 2)
    if brecha <= 0:
        return {'requiere_ajuste': False, 'motivo': 'objetivo_ya_alcanzado', 'ajustes': []}

    contribuciones = pd.Series(coef * x_std, index=feature_names)
    orden_categorias = sorted(
        CATEGORIAS_AJUSTABLES,
        key=lambda c: contribuciones.get(f'{c}_lag1', 0.0)
    )  # más negativo (penaliza más el ahorro) primero

    ajustes = []
    restante = brecha
    for cat in orden_categorias:
        if restante <= 0:
            break
        gasto_actual = fila_pronostico[f'{cat}_lag1']
        if pd.isna(gasto_actual) or gasto_actual <= 0:
            continue
        tope = TOPE_REDUCCION[FLEXIBILIDAD_CATEGORIAS[cat]] * gasto_actual
        reduccion = round(min(tope, restante), 2)
        if reduccion <= 0:
            continue
        ajustes.append({
            'categoria': cat,
            'gasto_actual_ultimo_mes': round(gasto_actual, 2),
            'reduccion_sugerida_eur': reduccion,
            'reduccion_sugerida_pct': round(reduccion / gasto_actual * 100, 1),
        })
        restante = round(restante - reduccion, 2)

    ahorro_estimado_tras_ajuste = round(ahorro_predicho_fila + (brecha - restante), 2)
    tasa_estimada_tras_ajuste = round(
        ahorro_estimado_tras_ajuste / salario_proxy_fila * 100, 2
    ) if salario_proxy_fila else np.nan

    return {
        'requiere_ajuste': True,
        'motivo': f'proyeccion_por_debajo_de_{objetivo_pct:.0f}pct',
        'objetivo_pct': objetivo_pct,
        'brecha_ahorro_eur': brecha,
        'brecha_cubierta_eur': round(brecha - restante, 2),
        'brecha_cubierta_pct': round((brecha - restante) / brecha * 100, 1) if brecha else 0.0,
        'ahorro_estimado_tras_ajuste': ahorro_estimado_tras_ajuste,
        'tasa_ahorro_estimada_tras_ajuste': tasa_estimada_tras_ajuste,
        'ajustes': ajustes,
    }


recomendaciones = []
for i in range(len(df_pronostico_base)):
    fila = df_pronostico_base.iloc[i]
    resultado = generar_recomendacion(
        fila, X_usuarios_s[i], lr_model.coef_, lr_model.intercept_, feature_cols,
        tasa_predicha.iloc[i], perfil_predicho.iloc[i])

    top_ajuste = resultado['ajustes'][0] if resultado['ajustes'] else None
    recomendaciones.append({
        'user_id': int(fila['user_id']),
        'perfil_ahorro_predicho': perfil_predicho.iloc[i],
        'requiere_ajuste': resultado['requiere_ajuste'],
        'brecha_ahorro_eur': resultado.get('brecha_ahorro_eur', 0.0),
        'brecha_cubierta_pct': resultado.get('brecha_cubierta_pct', np.nan),
        'tasa_ahorro_estimada_tras_ajuste': resultado.get('tasa_ahorro_estimada_tras_ajuste', np.nan),
        'categoria_principal_sugerida': top_ajuste['categoria'] if top_ajuste else None,
        'reduccion_principal_eur': top_ajuste['reduccion_sugerida_eur'] if top_ajuste else 0.0,
        'reduccion_principal_pct': top_ajuste['reduccion_sugerida_pct'] if top_ajuste else 0.0,
        'n_categorias_afectadas': len(resultado['ajustes']),
    })

tabla_recomendaciones = pd.DataFrame(recomendaciones)
tabla_recomendaciones.to_csv(f"{CLEAN_DIR}/recomendaciones_prescriptivas.csv", index=False)

n_con_ajuste = int(tabla_recomendaciones['requiere_ajuste'].sum())
pct_con_ajuste = round(n_con_ajuste / len(tabla_recomendaciones) * 100, 1)
print(f"\n  Usuarios con recomendación de ajuste: {n_con_ajuste:,} de "
      f"{len(tabla_recomendaciones):,} ({pct_con_ajuste}%)")
if n_con_ajuste:
    cobertura_media = tabla_recomendaciones.loc[
        tabla_recomendaciones['requiere_ajuste'], 'brecha_cubierta_pct'].mean()
    print(f"  Cobertura media de la brecha de ahorro tras el ajuste sugerido: "
          f"{cobertura_media:.1f}%")
    print(f"\n  Categoría principal sugerida (frecuencia):")
    print(tabla_recomendaciones.loc[tabla_recomendaciones['requiere_ajuste'],
                                      'categoria_principal_sugerida']
          .value_counts().to_string())

# Gráfico 7: frecuencia de la categoría principal sugerida
fig, ax = plt.subplots(figsize=(7, 4))
frecuencia = (tabla_recomendaciones.loc[tabla_recomendaciones['requiere_ajuste'],
                                          'categoria_principal_sugerida']
              .value_counts())
if len(frecuencia):
    ax.bar(frecuencia.index, frecuencia.values, color='#1A73E8', alpha=0.85)
ax.set_title('Capa prescriptiva — Categoría principal sugerida para el ajuste',
             fontweight='bold')
ax.set_ylabel('Nº de usuarios')
plt.tight_layout()
plt.savefig(f"{CLEAN_DIR}/prescriptivo_categoria_sugerida.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Figura guardada: prescriptivo_categoria_sugerida.png")

# Ejemplo ilustrativo completo (para un usuario con ajuste sugerido, si existe)
usuarios_con_ajuste = tabla_recomendaciones.index[tabla_recomendaciones['requiere_ajuste']]
if len(usuarios_con_ajuste):
    idx_demo = usuarios_con_ajuste[0]
    fila_demo = df_pronostico_base.iloc[idx_demo]
    resultado_demo = generar_recomendacion(
        fila_demo, X_usuarios_s[idx_demo], lr_model.coef_, lr_model.intercept_, feature_cols,
        tasa_predicha.iloc[idx_demo], perfil_predicho.iloc[idx_demo])
    print(f"\n  Ejemplo — usuario {int(fila_demo['user_id'])}:")
    print(f"    Tasa de ahorro proyectada: {tasa_predicha.iloc[idx_demo]:.2f}% "
          f"(perfil: {perfil_predicho.iloc[idx_demo]})")
    print(f"    Objetivo: {resultado_demo['objetivo_pct']:.0f}% "
          f"| Brecha: {resultado_demo['brecha_ahorro_eur']:.2f}€")
    for ajuste in resultado_demo['ajustes']:
        print(f"    - Reducir '{ajuste['categoria']}' en "
              f"{ajuste['reduccion_sugerida_eur']:.2f}€ "
              f"({ajuste['reduccion_sugerida_pct']:.1f}%)")
    print(f"    Tasa de ahorro estimada tras el ajuste: "
          f"{resultado_demo['tasa_ahorro_estimada_tras_ajuste']:.2f}%")


# ============================================================
# MODELO 2 — REGRESIÓN LOGÍSTICA (reformulación temporal)
# Objetivo: clasificar el perfil de ahorro del mes t a partir del
#           histórico t-1, t-2, t-3 del propio usuario
# Clases: ahorro_adecuado / ahorro_insuficiente (esquema binario por
# mediana; ver docstring de compute_saving_profile_threshold() en
# 06_prepara_usuarios_tink.py para el historial de esta decision)
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

# Ajuste de hiperparámetros: fuerza de regularización C de la
# regresión logística (L2, ponderación de clases balanceada),
# seleccionada con GridSearchCV + GroupKFold optimizando F1 macro.
print(f"\n  Filas train: {len(X2_train):,} | Filas test: {len(X2_test):,}")
print("\n  Ajuste de hiperparámetros (GridSearchCV + GroupKFold, métrica: F1 macro):")
candidatos_m2 = {
    'Logistica': (Pipeline([('sc', StandardScaler()),
                            ('m', LogisticRegression(max_iter=2000, class_weight='balanced'))]),
                  {'m__C': [0.001, 0.01, 0.1, 1, 10, 100]}),
}
_, params_m2, cv_m2 = ajustar_hiperparametros(
    candidatos_m2, X2_train, y2_train, train_df2['user_id'], 'f1_macro')
print(f"  C seleccionado: {params_m2['C']} (F1 macro CV = {cv_m2['cv_media'].max():.4f})")

clf_model = LogisticRegression(C=params_m2['C'], max_iter=2000, class_weight='balanced')
clf_model.fit(X2_train_s, y2_train)
y2_pred = clf_model.predict(X2_test_s)

acc = accuracy_score(y2_test, y2_pred)
f1_macro_m2 = f1_score(y2_test, y2_pred, average='macro')
print(f"  Accuracy test: {acc:.4f} | F1 macro test: {f1_macro_m2:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y2_test, y2_pred))

# Baselines de clasificación sobre el mismo test
clase_mayoritaria = y2_train.value_counts().idxmax()
pred_mayoritaria = np.full(len(y2_test), clase_mayoritaria)
# La clase del mes anterior se toma del histórico completo (df_raw) y
# se une por usuario y mes (train_df/test_df tienen índices reiniciados).
_orden = df_raw.sort_values(['user_id', 'fecha_dt'])
_prev = _orden.assign(clase_previa=_orden.groupby('user_id')['perfil_ahorro'].shift(1))[
    ['user_id', 'fecha', 'clase_previa']]
_test_prev = test_df2[['user_id', 'fecha']].merge(_prev, on=['user_id', 'fecha'], how='left')
_mask_prev = _test_prev['clase_previa'].notna().values
baselines_m2_out = {
    'Clase mayoritaria': {
        'accuracy': round(float(accuracy_score(y2_test, pred_mayoritaria)), 4),
        'f1_macro': round(float(f1_score(y2_test, pred_mayoritaria, average='macro')), 4)},
    'Persistencia (clase del mes anterior)': {
        'accuracy': round(float(accuracy_score(y2_test.values[_mask_prev],
                                               _test_prev['clase_previa'].values[_mask_prev])), 4),
        'f1_macro': round(float(f1_score(y2_test.values[_mask_prev],
                                         _test_prev['clase_previa'].values[_mask_prev],
                                         average='macro')), 4)},
}
print("  Baselines sobre el mismo conjunto de test:")
for nombre_b, met in baselines_m2_out.items():
    print(f"    {nombre_b:<40} accuracy {met['accuracy']:.4f}  F1 macro {met['f1_macro']:.4f}")

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

# Gráfico 5: interpretabilidad global de coeficientes.
# En clasificación BINARIA, scikit-learn devuelve un único vector de
# coeficientes (coef_.shape == (1, n_features)): el modelo estima
# directamente log-odds(clf_model.classes_[1]) frente a la clase de
# referencia classes_[0], por lo que no hay "un gráfico por clase"
# como en el caso multiclase (one-vs-rest) de versiones anteriores.
# Un coeficiente positivo aumenta la probabilidad de classes_[1].
if clf_model.coef_.shape[0] == 1:
    clase_positiva = clf_model.classes_[1]
    clase_referencia = clf_model.classes_[0]
    coefs = clf_model.coef_[0]
    orden = np.argsort(coefs)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(np.array(feature_cols)[orden], coefs[orden],
            color=['#EA4335' if c < 0 else '#34A853' for c in coefs[orden]],
            alpha=0.85)
    ax.axvline(0, color='gray', linewidth=0.8)
    ax.set_title(f'Modelo 2 — Importancia global de variables\n'
                 f'(positivo = mayor probabilidad de "{clase_positiva}", '
                 f'referencia: "{clase_referencia}")', fontweight='bold')
    ax.set_xlabel('Coeficiente (log-odds estandarizado)')
    plt.tight_layout()
    plt.savefig(f"{CLEAN_DIR}/modelo2_coeficientes_por_clase.png", dpi=150, bbox_inches='tight')
    plt.close()
else:
    # Multiclase (one-vs-rest): un panel de coeficientes por clase.
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
# MODELO AVANZADO (rúbrica 4.3) — ENSEMBLE: HistGradientBoosting
# Objetivo: contrastar los Modelos 1 y 2 (lineales, interpretables)
#           con un ensemble de árboles por gradient boosting, capaz de
#           capturar no linealidades e interacciones, sobre exactamente
#           las mismas variables, partición y validación cruzada.
# Valor añadido: cuantifica el coste en precisión de exigir
#           interpretabilidad lineal (requisito de la capa prescriptiva)
#           y comprueba si el techo de rendimiento del Modelo 2 se debe
#           al algoritmo o a la información disponible en las variables.
# ============================================================

print("\n" + "=" * 60)
print("MODELO AVANZADO — Ensemble HistGradientBoosting (regresión y clasificación)")
print("=" * 60)

# Nota: con >10.000 filas, HistGradientBoosting activa por defecto la
# parada temprana (early stopping) sobre un 10% de validación interna,
# por lo que el número de iteraciones lo decide el propio algoritmo; la
# rejilla se centra en la tasa de aprendizaje y la complejidad de cada
# árbol (número máximo de hojas).
rejilla_hgb = {'learning_rate': [0.05, 0.1], 'max_leaf_nodes': [15, 31, 63]}

print("\n  Regresión — ajuste de hiperparámetros (GroupKFold, MAE):")
_, params_hgb_r, cv_hgb_r = ajustar_hiperparametros(
    {'HGB_reg': (HistGradientBoostingRegressor(random_state=RANDOM_STATE), rejilla_hgb)},
    X_train, y_train, train_df['user_id'], 'neg_mean_absolute_error')
hgb_reg = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params_hgb_r)
hgb_reg.fit(X_train, y_train)
y_pred_hgb = hgb_reg.predict(X_test)
mae_hgb, r2_hgb = mean_absolute_error(y_test, y_pred_hgb), r2_score(y_test, y_pred_hgb)
print(f"  Seleccionado {params_hgb_r} | MAE test {mae_hgb:.2f} € | R² test {r2_hgb:.4f} "
      f"(Modelo 1: MAE {mae_lr:.2f} €, R² {r2_lr:.4f})")

print("\n  Clasificación — ajuste de hiperparámetros (GroupKFold, F1 macro):")
_, params_hgb_c, cv_hgb_c = ajustar_hiperparametros(
    {'HGB_clf': (HistGradientBoostingClassifier(random_state=RANDOM_STATE,
                                                class_weight='balanced'), rejilla_hgb)},
    X2_train, y2_train, train_df2['user_id'], 'f1_macro')
hgb_clf = HistGradientBoostingClassifier(random_state=RANDOM_STATE, class_weight='balanced',
                                         **params_hgb_c)
hgb_clf.fit(X2_train, y2_train)
y2_pred_hgb = hgb_clf.predict(X2_test)
acc_hgb = accuracy_score(y2_test, y2_pred_hgb)
f1_hgb = f1_score(y2_test, y2_pred_hgb, average='macro')
print(f"  Seleccionado {params_hgb_c} | accuracy test {acc_hgb:.4f} | F1 macro test {f1_hgb:.4f} "
      f"(Modelo 2: accuracy {acc:.4f}, F1 macro {f1_macro_m2:.4f})")

# Gráfico comparativo: baselines vs. modelo lineal vs. ensemble
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
etiquetas_r = ['Media\nentrenamiento', 'Persistencia\n(mes anterior)', 'Media móvil\n3 meses',
               f'Modelo 1\n({nombre_m1})', 'Ensemble\n(HGB)']
valores_r = [baselines_m1_out['Media del conjunto de entrenamiento']['mae'],
             baselines_m1_out['Persistencia (ahorro del mes anterior)']['mae'],
             baselines_m1_out['Media móvil de 3 meses (ahorro_roll3)']['mae'],
             mae_lr, mae_hgb]
colores_r = ['#BDBDBD', '#BDBDBD', '#BDBDBD', '#1A73E8', '#34A853']
barras = axes[0].bar(etiquetas_r, valores_r, color=colores_r)
axes[0].bar_label(barras, fmt='%.0f €', padding=3)
axes[0].set_title('Regresión del ahorro — MAE en test (menor es mejor)', fontweight='bold')
axes[0].set_ylabel('MAE (€)')
axes[0].set_ylim(0, max(valores_r) * 1.15)

etiquetas_c = ['Clase\nmayoritaria', 'Persistencia\n(clase mes anterior)',
               'Modelo 2\n(Logística)', 'Ensemble\n(HGB)']
valores_c = [baselines_m2_out['Clase mayoritaria']['f1_macro'],
             baselines_m2_out['Persistencia (clase del mes anterior)']['f1_macro'],
             f1_macro_m2, f1_hgb]
barras = axes[1].bar(etiquetas_c, valores_c, color=['#BDBDBD', '#BDBDBD', '#1A73E8', '#34A853'])
axes[1].bar_label(barras, fmt='%.3f', padding=3)
axes[1].set_title('Clasificación del perfil de ahorro — F1 macro en test (mayor es mejor)',
                  fontweight='bold')
axes[1].set_ylabel('F1 macro')
axes[1].set_ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{CLEAN_DIR}/modelo_avanzado_comparacion.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Figura guardada: modelo_avanzado_comparacion.png")


# ============================================================
# MODELO 3 — SERIE TEMPORAL (índice encadenado por composición estable)
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
#
# CORRECCIÓN METODOLÓGICA (revisión de esta versión): la media
# transversal simple por mes está distorsionada por el cambio de
# composición de la población activa (entre ~5.000 y ~30.000 usuarios
# según el mes, por las distintas ventanas temporales de los 8
# perfiles reales de origen). Se sustituye por un índice encadenado
# (ver docstring de build_chained_savings_index) que aísla la dinámica
# temporal real del efecto de composición. Se conserva también la
# media simple (ahorro_medio_naive) únicamente a título comparativo,
# para documentar visualmente el efecto de la corrección.
# ============================================================

print("\n" + "=" * 60)
print("MODELO 3 — Serie Temporal: proyección del ahorro medio (6 meses)")
print("=" * 60)

ahorro_medio_naive = (df_raw.groupby('fecha')['ahorro']
                       .mean()
                       .reset_index()
                       .rename(columns={'ahorro': 'ahorro_medio_naive'}))

ahorro_medio = build_chained_savings_index(df_raw)
ahorro_medio = ahorro_medio.merge(ahorro_medio_naive, on='fecha', how='left')
ahorro_medio = ahorro_medio.rename(columns={'ahorro_medio_encadenado': 'ahorro_medio'})
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
r2_ts_insample = ts_model.score(X_ts, y_ts)

# Validación del Modelo 3 — backtest: se reentrena el mismo modelo
# excluyendo los 6 últimos meses observados y se comparan sus
# predicciones para esos meses con tres pronósticos ingenuos. Es la
# forma estándar de validar un modelo de series temporales (validación
# fuera de muestra respetando el orden temporal).
H_BACKTEST = 6
ts_bt = LinearRegression().fit(X_ts.iloc[:-H_BACKTEST], y_ts.iloc[:-H_BACKTEST])
y_bt_real = y_ts.iloc[-H_BACKTEST:].values
pronosticos_bt = {
    'Modelo 3 (tendencia + estacionalidad)': ts_bt.predict(X_ts.iloc[-H_BACKTEST:]),
    'Ingenuo: último valor observado': np.full(H_BACKTEST, y_ts.iloc[-H_BACKTEST - 1]),
    'Ingenuo: media histórica': np.full(H_BACKTEST, y_ts.iloc[:-H_BACKTEST].mean()),
    'Ingenuo estacional: mismo mes del año anterior': y_ts.iloc[-H_BACKTEST - 12:-12].values,
}
backtest_m3 = {k: round(float(mean_absolute_error(y_bt_real, v)), 2)
               for k, v in pronosticos_bt.items()}
print(f"\n  R² del ajuste dentro de muestra: {r2_ts_insample:.3f}")
print(f"  Variabilidad de la serie: desviación típica {y_ts.std():.2f} € "
      f"sobre un nivel medio de {y_ts.mean():.2f} €")
print(f"  Backtest (últimos {H_BACKTEST} meses, MAE):")
for k, v in backtest_m3.items():
    print(f"    {k:<48} {v:7.2f} €")

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

# Gráfico 6-bis: comparación media naive (distorsionada por composición)
# vs. índice encadenado (corregido), para documentar visualmente el
# efecto de la corrección metodológica en la memoria.
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(ahorro_medio['fecha_dt'], ahorro_medio['ahorro_medio_naive'],
        color='#9AA0A6', linewidth=1.5, linestyle=':', marker='o', markersize=3,
        label='Media simple (sin corregir, distorsionada por composición)')
ax.plot(ahorro_medio['fecha_dt'], ahorro_medio['ahorro_medio'],
        color='#1A73E8', linewidth=2, marker='o', markersize=3,
        label='Índice encadenado (corregido)')
ax.set_title('Modelo 3 — Efecto de la corrección por composición de la muestra',
              fontweight='bold')
ax.set_xlabel('Fecha')
ax.set_ylabel('Ahorro medio (€)')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{CLEAN_DIR}/modelo3_comparacion_naive_vs_encadenado.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Figura guardada: modelo3_comparacion_naive_vs_encadenado.png")


# ── GUARDAR MÉTRICAS ─────────────────────────────────────────
metricas = {
    'modelo1_mae': round(mae_lr, 2),
    'modelo1_r2': round(r2_lr, 4),
    'modelo1_n_features': len(feature_cols),
    'modelo1_features': feature_cols,
    'validacion': (f'GroupShuffleSplit por user_id (20% usuarios en test); ajuste de '
                   f'hiperparametros con GridSearchCV + GroupKFold ({N_FOLDS_CV} pliegues) '
                   f'sobre el conjunto de entrenamiento'),
    'modelo1_modelo_seleccionado': nombre_m1,
    'modelo1_hiperparametros': params_m1,
    'modelo1_cv_resultados': cv_m1.assign(cv_mae=-cv_m1['cv_media'])[
        ['modelo', 'parametros', 'cv_mae', 'cv_std']].round(2).to_dict('records'),
    'modelo1_baselines_test': baselines_m1_out,
    'modelo1_colinealidad': {
        'corr_salario_lag1_log_vs_salario_roll3_log': round(corr_sal, 3),
        'corr_salario_roll3_log_vs_ahorro': round(corr_roll_ahorro, 3),
        'coeficientes': estabilidad_coef,
    },
    'modelo1_nota_colinealidad': (
        'salario_lag1_log y salario_roll3_log estan fuertemente correlacionadas '
        f'(r={corr_sal:.2f}). salario_roll3_log recibe un coeficiente negativo pese '
        f'a correlacionar positivamente con el ahorro (r={corr_roll_ahorro:+.2f}). '
        'La regularizacion seleccionada por validacion cruzada (Ridge o Lasso) apenas '
        'modifica los coeficientes respecto a OLS, porque la validacion cruzada '
        'optimiza el error de prediccion, no la interpretabilidad; y Lasso no anula '
        'ninguna de las dos variables. El coeficiente es estable entre pliegues de '
        'la validacion cruzada, por lo que no es inestabilidad numerica sino un '
        'efecto condicional (de supresion): mide el efecto de salario_roll3_log '
        'manteniendo constantes el salario del mes anterior y los rezagos de ahorro, '
        'no su efecto marginal. Los coeficientes de ambas variables de salario deben '
        'interpretarse conjuntamente.'
    ),
    'modelo2_accuracy': round(acc, 4),
    'modelo2_f1_macro': round(float(f1_macro_m2), 4),
    'modelo2_C_seleccionado': params_m2['C'],
    'modelo2_cv_resultados': cv_m2[['parametros', 'cv_media', 'cv_std']].round(4).to_dict('records'),
    'modelo2_baselines_test': baselines_m2_out,
    'modelo2_clases': list(clf_model.classes_),
    'modelo_avanzado_ensemble': {
        'algoritmo': 'HistGradientBoosting (gradient boosting de arboles)',
        'regresion': {'hiperparametros': params_hgb_r, 'mae_test': round(float(mae_hgb), 2),
                      'r2_test': round(float(r2_hgb), 4),
                      'cv_resultados': cv_hgb_r.assign(cv_mae=-cv_hgb_r['cv_media'])[
                          ['parametros', 'cv_mae', 'cv_std']].round(2).to_dict('records')},
        'clasificacion': {'hiperparametros': params_hgb_c, 'accuracy_test': round(float(acc_hgb), 4),
                          'f1_macro_test': round(float(f1_hgb), 4),
                          'cv_resultados': cv_hgb_c[['parametros', 'cv_media', 'cv_std']]
                          .round(4).to_dict('records')},
    },
    'modelo2_esquema_clases': (
        'Esquema binario (ahorro_adecuado / ahorro_insuficiente) por '
        'mediana, sustituyendo el esquema anterior de terciles (3 '
        'clases). Historial: (1) umbral absoluto fijo -> colapsaba a una '
        'sola clase; (2) terciles -> resolvia el colapso pero la clase '
        'central quedaba en una franja de solo ~4 puntos porcentuales, '
        'estadisticamente no separable (recall 0.02); (3) se evaluo '
        'tambien un esquema de 3 clases por igual anchura (8/20/72%), '
        'que mejoraba el recall de la clase central (0.65) pero hacia '
        "colapsar la precision de 'ahorro_insuficiente' (0.52 -> 0.20, "
        'F1 0.61 -> 0.32); (4) esquema binario por mediana (adoptado): '
        'un unico limite de decision elimina el problema estructural de '
        'una clase atrapada entre dos fronteras, sin el trade-off de '
        'precision observado en la alternativa de 3 clases. F1 macro '
        'final: 0.72 (vs. 0.43 con terciles; ambas cifras no son '
        'directamente comparables, al cambiar el numero de clases: el F1 '
        'macro de referencia por azar es ~0.33 con 3 clases y ~0.50 con 2). '
        'Ver docstring de '
        'compute_saving_profile_threshold() en 06_prepara_usuarios_tink.py '
        'para el detalle completo de las 4 iteraciones.'
    ),
    'modelo3_predicciones': df_pred[['fecha', 'ahorro_predicho']].to_dict('records'),
    'modelo3_metodo': (
        'Indice encadenado (chain-linked) sobre usuarios comunes entre '
        'meses consecutivos, en lugar de media transversal simple. '
        'Corrige la distorsion por composicion cambiante de la muestra '
        '(entre ~5.000 y ~30.000 usuarios activos segun el mes, por las '
        'distintas ventanas temporales de los 8 perfiles reales de '
        'origen Tink). Ver docstring de build_chained_savings_index().'
    ),
    'modelo3_r2_insample': round(float(r2_ts_insample), 3),
    'modelo3_std_serie': round(float(y_ts.std()), 2),
    'modelo3_nivel_medio_serie': round(float(y_ts.mean()), 2),
    'modelo3_backtest_mae_6meses': backtest_m3,
    'modelo3_usuarios_comunes_min': int(ahorro_medio['n_usuarios_comunes'].dropna().min()),
    'modelo3_usuarios_comunes_mediana': int(ahorro_medio['n_usuarios_comunes'].dropna().median()),
    'particion': 'GroupShuffleSplit por user_id (20% usuarios en test)',
    'usuarios_train': int(train_df['user_id'].nunique()),
    'usuarios_test': int(test_df['user_id'].nunique()),
    'prescriptivo_usuarios_con_ajuste': n_con_ajuste,
    'prescriptivo_pct_usuarios_con_ajuste': pct_con_ajuste,
    'prescriptivo_cobertura_media_brecha_pct': (
        round(float(cobertura_media), 1) if n_con_ajuste else None
    ),
}
with open(f"{CLEAN_DIR}/metricas_modelos.json", 'w') as f:
    json.dump(metricas, f, indent=2)

# ── GUARDAR DATASETS ─────────────────────────────────────────
df_model.to_csv(f"{CLEAN_DIR}/dataset_final_usuarios.csv", index=False)
ahorro_medio.to_csv(f"{CLEAN_DIR}/ahorro_medio_mensual.csv", index=False)

print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"  Modelo 1 — {nombre_m1} {params_m1} (histórico t-1..t-3):")
print(f"    MAE = {mae_lr:.2f}€ | R² = {r2_lr:.4f}")
print(f"  Modelo avanzado — Ensemble HGB: MAE = {mae_hgb:.2f}€ | R² = {r2_hgb:.4f} | "
      f"F1 macro clasificación = {f1_hgb:.4f}")
print(f"  Modelo 1-bis — Proyección individual: "
      f"{len(tabla_kpi_usuarios):,} usuarios")
print(f"  Capa prescriptiva — Usuarios con recomendación de ajuste: "
      f"{n_con_ajuste:,} ({pct_con_ajuste}%)")
print(f"  Modelo 2 — Regresión Logística (histórico t-1..t-3):")
print(f"    Accuracy = {acc:.4f} | F1 macro = {f1_macro_m2:.4f} | C = {params_m2['C']}")
print(f"  Modelo 3 — Proyección temporal agregada:")
print(f"    Ahorro proyectado {df_pred.iloc[0]['fecha']}: {predicciones[0]:.2f}€")
print(f"    Ahorro proyectado {df_pred.iloc[-1]['fecha']}: {predicciones[-1]:.2f}€")
print(f"\n  Archivos guardados en: {CLEAN_DIR}")
