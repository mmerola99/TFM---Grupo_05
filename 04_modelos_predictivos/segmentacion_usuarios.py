# ============================================================
# segmentacion_usuarios.py
# Segmentación no supervisada de usuarios — AI Financial Life Coach
# Asignatura 7, apartado 4.4 (5%): Análisis No Supervisado / Segmentación
# ------------------------------------------------------------
# Técnica: K-Means sobre un perfil comportamental agregado por
# usuario (no por fila usuario-mes), para obtener un segmento
# ESTABLE por persona, útil para el producto (p. ej. adaptar el
# tono y las recomendaciones de la capa prescriptiva según el
# segmento del usuario, en vez de tratar a todos por igual).
#
# Se justifica el número de clusters con el método del codo
# (inercia) y el coeficiente de silueta, se evalúan los resultados
# cuantitativamente (silueta, tamaño de cada segmento) y se
# interpretan cualitativamente relacionándolos con el TFM.
# ============================================================

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from preprocesamiento import codificar_perfil_onehot

sns.set_style("whitegrid")
sns.set_palette("Set2")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean')
RANDOM_STATE = 42

CATEGORIAS = ['vivienda', 'alimentacion', 'transporte', 'ocio', 'otros']


# ============================================================
# 1. AGREGACIÓN A NIVEL DE USUARIO
# ============================================================

def construir_perfil_usuario(df):
    """Agrega el dataset mensual (fila = usuario-mes) a un perfil
    comportamental ESTABLE por usuario (fila = usuario).

    Se agrega en vez de clusterizar usuario-mes directamente por una
    razón de negocio, no solo técnica: un segmento tiene valor para
    el producto (adaptar recomendaciones, tono, prioridades) cuando
    identifica un tipo de persona estable, no un estado pasajero de
    un mes concreto. Clusterizar usuario-mes haría que el mismo
    usuario "saltara" de segmento mes a mes por ruido, lo cual no es
    interpretable ni accionable.

    Variables del perfil (todas numéricas, escala mensual media por
    usuario a lo largo de todo su histórico disponible):
      - ingreso_medio: salario medio mensual.
      - ahorro_medio: ahorro medio mensual (€).
      - tasa_ahorro_media: tasa de ahorro media (%).
      - volatilidad_ahorro: desviación típica del ahorro mensual,
        normalizada por el ingreso medio (coeficiente de variación),
        para que sea comparable entre usuarios de distinto nivel de
        renta (una desviación típica de 500€ es "mucha" volatilidad
        para quien gana 1000€/mes y "poca" para quien gana 8000€/mes).
      - pct_gasto_<categoria>: composición media del gasto por
        categoría, como fracción del gasto total (perfil de consumo,
        no solo de ahorro).
      - edad: edad del usuario (constante en el histórico disponible).
    """
    g = df.groupby('user_id')

    perfil = pd.DataFrame({
        'ingreso_medio': g['salario'].mean(),
        'ahorro_medio': g['ahorro'].mean(),
        'tasa_ahorro_media': g['tasa_ahorro_pct'].mean(),
        'ahorro_std': g['ahorro'].std().fillna(0),
        'edad': g['edad'].first(),
        'perfil': g['perfil'].first(),
        'n_meses_observados': g['ahorro'].count(),
    })

    # Volatilidad normalizada (coeficiente de variación del ahorro
    # respecto al ingreso medio del propio usuario).
    perfil['volatilidad_ahorro'] = (perfil['ahorro_std'] /
                                     perfil['ingreso_medio'].replace(0, np.nan)).fillna(0)

    gasto_total_medio = g['gasto_total'].mean()
    for cat in CATEGORIAS:
        perfil[f'pct_gasto_{cat}'] = (g[cat].mean() / gasto_total_medio.replace(0, np.nan)).fillna(0)

    perfil = perfil.reset_index()
    return perfil


# ============================================================
# 2. SELECCIÓN DEL NÚMERO DE CLUSTERS (codo + silueta)
# ============================================================

def evaluar_k(X_scaled, k_range=range(2, 9)):
    """Calcula inercia (método del codo) y coeficiente de silueta
    para cada k en el rango indicado. La silueta se calcula sobre
    una submuestra (máx. 20.000 puntos) por coste computacional en
    un dataset de 40.000 usuarios, sin afectar la asignación final
    de clusters (que sí usa todos los usuarios)."""
    rng = np.random.default_rng(RANDOM_STATE)
    n_muestra = min(20_000, X_scaled.shape[0])
    idx_muestra = rng.choice(X_scaled.shape[0], n_muestra, replace=False)

    resultados = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled[idx_muestra], labels[idx_muestra])
        resultados.append({'k': k, 'inercia': km.inertia_, 'silueta': sil})
        print(f"  k={k}: inercia={km.inertia_:,.0f}  silueta={sil:.4f}")
    return pd.DataFrame(resultados)


def graficar_evaluacion_k(df_eval, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(df_eval['k'], df_eval['inercia'], marker='o', color='#1A73E8')
    axes[0].set_title('Método del codo', fontweight='bold')
    axes[0].set_xlabel('Número de clusters (k)')
    axes[0].set_ylabel('Inercia (suma de distancias al cuadrado)')

    axes[1].plot(df_eval['k'], df_eval['silueta'], marker='o', color='#34A853')
    axes[1].set_title('Coeficiente de silueta', fontweight='bold')
    axes[1].set_xlabel('Número de clusters (k)')
    axes[1].set_ylabel('Silueta media')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# 3. CLUSTERING FINAL E INTERPRETACIÓN
# ============================================================

def interpretar_clusters(perfil, feature_cols):
    """Devuelve una tabla con el perfil medio de cada cluster (para
    interpretación cualitativa) y el tamaño de cada segmento."""
    cols_resumen = list(dict.fromkeys(feature_cols + ['ahorro_medio', 'ingreso_medio',
                                                        'tasa_ahorro_media']))
    resumen = perfil.groupby('cluster')[cols_resumen].mean()
    resumen['n_usuarios'] = perfil.groupby('cluster').size()
    resumen['pct_usuarios'] = (resumen['n_usuarios'] / len(perfil) * 100).round(1)
    return resumen.round(2)


def graficar_clusters_pca(X_scaled, labels, output_path):
    """Proyección a 2D vía PCA únicamente para visualización (el
    clustering en sí se realiza en el espacio original de features,
    no en el espacio PCA)."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    var_explicada = pca.explained_variance_ratio_.sum()

    fig, ax = plt.subplots(figsize=(8, 6))
    rng = np.random.default_rng(RANDOM_STATE)
    n_muestra = min(8000, len(coords))
    idx = rng.choice(len(coords), n_muestra, replace=False)
    scatter = ax.scatter(coords[idx, 0], coords[idx, 1], c=labels[idx],
                          cmap='Set2', alpha=0.5, s=12)
    ax.set_title(f'Segmentación de usuarios — proyección PCA\n'
                 f'({var_explicada*100:.1f}% de varianza explicada en 2 componentes)',
                 fontweight='bold')
    ax.set_xlabel('Componente principal 1')
    ax.set_ylabel('Componente principal 2')
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", loc='best')
    ax.add_artist(legend1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def graficar_perfil_clusters(resumen, feature_cols, output_path):
    """Heatmap del perfil medio (estandarizado por columna, para
    comparabilidad visual) de cada cluster."""
    datos = resumen[feature_cols].copy()
    datos_z = (datos - datos.mean()) / datos.std()

    fig, ax = plt.subplots(figsize=(10, 0.6 * len(resumen) + 2))
    sns.heatmap(datos_z, annot=datos.round(2), fmt='g', cmap='RdYlGn',
                center=0, cbar_kws={'label': 'Desviación respecto a la media (z-score)'}, ax=ax)
    ax.set_title('Perfil medio por segmento (valores reales anotados, color = z-score)',
                 fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Cluster')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("SEGMENTACIÓN DE USUARIOS — K-Means")
    print("=" * 60)

    path = os.path.join(DATA_DIR, 'dataset_final_usuarios_tink.csv')
    df = pd.read_csv(path)
    df, cols_perfil = codificar_perfil_onehot(df)

    perfil = construir_perfil_usuario(df)
    perfil = pd.concat([perfil, df.groupby('user_id')[cols_perfil].first().reset_index(drop=True)], axis=1)
    print(f"\nPerfiles de usuario construidos: {len(perfil):,}")

    feature_cols = (['ingreso_medio', 'tasa_ahorro_media', 'volatilidad_ahorro', 'edad']
                     + [f'pct_gasto_{c}' for c in CATEGORIAS] + cols_perfil)

    X = perfil[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n--- Evaluación del número de clusters (k=2..8) ---")
    df_eval = evaluar_k(X_scaled)
    graficar_evaluacion_k(df_eval, os.path.join(DATA_DIR, 'segmentacion_evaluacion_k.png'))

    # Selección de k: se prioriza la silueta más alta entre k=3..6
    # (se excluye k=2, que en la práctica solo separa ingresos
    # altos/bajos y aporta poco valor de segmentación para el
    # producto; ver discusión en la memoria).
    candidatos = df_eval[df_eval['k'].between(3, 6)]
    k_elegido = int(candidatos.loc[candidatos['silueta'].idxmax(), 'k'])
    print(f"\nK elegido: {k_elegido} (silueta más alta en el rango 3-6, "
          f"evitando k=2 que solo separa por nivel de ingreso)")

    km_final = KMeans(n_clusters=k_elegido, random_state=RANDOM_STATE, n_init=10)
    perfil['cluster'] = km_final.fit_predict(X_scaled)

    sil_final = silhouette_score(
        X_scaled[np.random.default_rng(RANDOM_STATE).choice(len(X_scaled), 20000, replace=False)],
        perfil['cluster'].values[np.random.default_rng(RANDOM_STATE).choice(len(X_scaled), 20000, replace=False)]
    )
    print(f"Silueta del modelo final (k={k_elegido}): {sil_final:.4f}")

    resumen = interpretar_clusters(perfil, feature_cols)
    print("\n--- Perfil medio por cluster ---")
    print(resumen.to_string())

    graficar_clusters_pca(X_scaled, perfil['cluster'].values,
                           os.path.join(DATA_DIR, 'segmentacion_clusters_pca.png'))
    graficar_perfil_clusters(resumen, ['ingreso_medio', 'tasa_ahorro_media',
                                        'volatilidad_ahorro', 'edad'] +
                              [f'pct_gasto_{c}' for c in CATEGORIAS],
                              os.path.join(DATA_DIR, 'segmentacion_perfil_clusters.png'))

    # Guardar asignaciones y resumen
    perfil[['user_id', 'cluster'] + feature_cols + ['ahorro_medio', 'ingreso_medio',
                                                      'tasa_ahorro_media']].to_csv(
        os.path.join(DATA_DIR, 'metrics', 'segmentacion_usuarios.csv'), index=False)

    resumen_out = resumen.reset_index().to_dict(orient='records')
    diagnostico = {
        'metodo': 'K-Means sobre perfil comportamental agregado por usuario',
        'k_elegido': k_elegido,
        'criterio_seleccion_k': ('silueta maxima en rango k=3..6; se excluye k=2 por separar '
                                  'trivialmente solo por nivel de ingreso'),
        'silueta_k_elegido': round(float(sil_final), 4),
        'evaluacion_k': df_eval.to_dict(orient='records'),
        'n_usuarios_segmentados': int(len(perfil)),
        'resumen_por_cluster': resumen_out,
    }
    with open(os.path.join(DATA_DIR, 'segmentacion_diagnostico.json'), 'w', encoding='utf-8') as f:
        json.dump(diagnostico, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nArchivos guardados en: {DATA_DIR}")
    print("  - segmentacion_evaluacion_k.png")
    print("  - segmentacion_clusters_pca.png")
    print("  - segmentacion_perfil_clusters.png")
    print("  - segmentacion_diagnostico.json")
    print("  - metrics/segmentacion_usuarios.csv")
