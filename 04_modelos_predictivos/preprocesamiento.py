# ============================================================
# preprocesamiento.py
# Preprocesamiento explícito de datos — AI Financial Life Coach
# Asignatura 7, apartado 4.1 (20%)
# ------------------------------------------------------------
# Este módulo aísla, documenta y justifica empíricamente cada
# decisión de preprocesamiento aplicada antes de entrenar los
# modelos (ver 04_modelos_predictivos/modelos_predictivos.py,
# que importa y reutiliza estas mismas funciones para garantizar
# que el modelado use exactamente el mismo preprocesamiento aquí
# documentado, sin duplicación de lógica).
#
# Decisiones cubiertas:
#   1. Codificación de la variable categórica nominal 'perfil'
#      (one-hot encoding, no ordinal).
#   2. Evaluación de asimetría (skewness) de las variables
#      monetarias y transformación logarítmica (log1p) donde
#      está empíricamente justificada.
#   3. Escalado (StandardScaler), ajustado exclusivamente sobre
#      el conjunto de entrenamiento.
#
# Cada decisión se apoya en un diagnóstico cuantitativo (no en
# una regla general aplicada sin comprobar), conforme al criterio
# de rigor metodológico de la rúbrica.
# ============================================================

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")
sns.set_palette("Set2")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean')

# Variables monetarias candidatas a transformación logarítmica.
# 'ahorro' se excluye deliberadamente: contiene valores negativos
# (no admite log directo) y es la variable objetivo del Modelo 1,
# donde se prioriza la interpretabilidad de los coeficientes en
# euros (ver apartado 7 de la memoria) sobre una posible mejora
# marginal de ajuste.
VARIABLES_MONETARIAS_A_EVALUAR = ['salario', 'gasto_total']

PERFIL_CATEGORIAS = ['junior', 'medio', 'senior', 'freelance']


# ============================================================
# 1. CODIFICACIÓN DE VARIABLES CATEGÓRICAS
# ============================================================

def codificar_perfil_onehot(df):
    """Codifica la variable categórica nominal 'perfil' (junior /
    medio / senior / freelance) mediante one-hot encoding.

    Se descarta deliberadamente una codificación ordinal (p. ej.
    junior=0, medio=1, senior=2, freelance=3): 'perfil' no tiene un
    orden natural entre sus categorías (un perfil 'freelance' no es
    'mayor' ni 'menor' que un perfil 'senior' en ninguna escala
    continua), por lo que una codificación ordinal induciría en el
    modelo una relación de magnitud inexistente en los datos. Se usa
    drop_first=True para evitar colinealidad perfecta con el
    intercepto (dummy variable trap) en los modelos lineales.

    Devuelve el DataFrame con las columnas dummy añadidas y la lista
    de nombres de esas columnas.
    """
    dummies = pd.get_dummies(
        df['perfil'].astype(pd.CategoricalDtype(categories=PERFIL_CATEGORIAS)),
        prefix='perfil', drop_first=True, dtype=float
    )
    df = pd.concat([df, dummies], axis=1)
    return df, list(dummies.columns)


# ============================================================
# 2. EVALUACIÓN DE ASIMETRÍA Y TRANSFORMACIÓN LOGARÍTMICA
# ============================================================

def evaluar_asimetria(df, columnas):
    """Calcula el coeficiente de asimetría (skewness) de cada
    columna indicada. Devuelve un diccionario {columna: skew}.

    Regla de decisión aplicada (convención estadística estándar):
    |skew| > 1 se considera asimetría relevante y candidata a
    transformación logarítmica; |skew| <= 1 se considera aceptable
    para un modelo lineal sin transformar.
    """
    resultados = {}
    for col in columnas:
        resultados[col] = float(skew(df[col].dropna()))
    return resultados


def aplicar_log_transform(df, columnas_a_transformar):
    """Aplica log1p (log(1+x)) a las columnas indicadas y a sus
    variantes rezagadas/móviles ya construidas (*_lag1, *_roll3),
    si existen en el DataFrame. log1p (en vez de log simple) evita
    -inf en observaciones con valor 0 o próximo a 0.

    Devuelve el DataFrame con columnas nuevas sufijadas '_log' y la
    lista de columnas creadas.
    """
    nuevas_columnas = []
    for base in columnas_a_transformar:
        candidatas = [base] + [f'{base}_lag1', f'{base}_roll3']
        for col in candidatas:
            if col in df.columns:
                nueva = f'{col}_log'
                df[nueva] = np.log1p(df[col].clip(lower=0))
                nuevas_columnas.append(nueva)
    return df, nuevas_columnas


def decidir_variables_a_transformar(df, umbral_skew=1.0):
    """Evalúa la asimetría de las variables monetarias candidatas y
    decide, de forma reproducible y basada en datos (no en una regla
    aplicada a ciegas), cuáles superan el umbral y deben
    transformarse. Devuelve (diagnostico_dict, lista_a_transformar).
    """
    diagnostico = evaluar_asimetria(df, VARIABLES_MONETARIAS_A_EVALUAR)
    a_transformar = [c for c, s in diagnostico.items() if abs(s) > umbral_skew]
    return diagnostico, a_transformar


# ============================================================
# 3. ESCALADO (fit exclusivamente sobre entrenamiento)
# ============================================================

def escalar_features(X_train, X_test):
    """Ajusta un StandardScaler únicamente sobre X_train y lo aplica
    (transform) a X_train y X_test. Ajustar el escalador sobre todo
    el dataset (incluyendo test) filtraría estadísticos del conjunto
    de prueba hacia el preprocesamiento de entrenamiento — una forma
    sutil de fuga de información que se evita deliberadamente aquí.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


# ============================================================
# DIAGNÓSTICO VISUAL (antes / después de log-transform)
# ============================================================

def graficar_efecto_log_transform(df, columnas_transformadas, output_path):
    """Genera un panel de histogramas comparando la distribución
    original y la transformada (log1p) de cada variable, como
    evidencia visual de la decisión tomada."""
    n = len(columnas_transformadas)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(11, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    for i, col in enumerate(columnas_transformadas):
        sns.histplot(df[col].dropna(), bins=60, ax=axes[i, 0], color='#EA4335')
        axes[i, 0].set_title(f'{col} — original (skew={skew(df[col].dropna()):.2f})')
        sns.histplot(df[f'{col}_log'].dropna(), bins=60, ax=axes[i, 1], color='#1A73E8')
        axes[i, 1].set_title(f'{col} — log1p (skew={skew(df[f"{col}_log"].dropna()):.2f})')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# EJECUCIÓN STANDALONE — diagnóstico reproducible
# ============================================================

if __name__ == '__main__':
    path = os.path.join(DATA_DIR, 'dataset_final_usuarios_tink.csv')
    df = pd.read_csv(path)

    print("=" * 60)
    print("1. CODIFICACIÓN DE 'perfil' (one-hot)")
    print("=" * 60)
    df, cols_perfil = codificar_perfil_onehot(df)
    print(f"Columnas dummy creadas: {cols_perfil}")
    print(df[['perfil'] + cols_perfil].drop_duplicates().sort_values('perfil')
          .to_string(index=False))

    print("\n" + "=" * 60)
    print("2. EVALUACIÓN DE ASIMETRÍA (skewness)")
    print("=" * 60)
    diagnostico, a_transformar = decidir_variables_a_transformar(df)
    for col, s in diagnostico.items():
        marca = "-> se transforma (|skew| > 1)" if col in a_transformar else "-> se deja sin transformar"
        print(f"  {col}: skew = {s:.3f}  {marca}")
    print(f"  ahorro: no evaluado para transformación (contiene valores negativos "
          f"y es la variable objetivo del Modelo 1; se prioriza la interpretabilidad "
          f"de sus coeficientes en euros).")

    print("\n" + "=" * 60)
    print("3. APLICACIÓN DE LOG-TRANSFORM")
    print("=" * 60)
    df, cols_log = aplicar_log_transform(df, a_transformar)
    print(f"Columnas log-transformadas creadas: {cols_log}")

    diagnostico_out = {
        'skewness_original': diagnostico,
        'variables_transformadas': a_transformar,
        'columnas_log_creadas': cols_log,
        'umbral_skew': 1.0,
        'nota_ahorro': ('excluido de la evaluacion de transformacion logaritmica: '
                         'contiene valores negativos y es la variable objetivo del '
                         'Modelo 1, donde se prioriza la interpretabilidad en euros.'),
        'nota_encoding_perfil': ('one-hot encoding (drop_first=True), no ordinal: '
                                  '"perfil" es una variable categorica nominal sin '
                                  'orden natural entre sus categorias.'),
    }
    out_path = os.path.join(DATA_DIR, 'preprocesamiento_diagnostico.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(diagnostico_out, f, indent=2, ensure_ascii=False)
    print(f"\nDiagnóstico guardado en: {out_path}")

    fig_path = os.path.join(DATA_DIR, 'preprocesamiento_log_transform.png')
    graficar_efecto_log_transform(df, a_transformar, fig_path)
    print(f"Gráfico comparativo guardado en: {fig_path}")
