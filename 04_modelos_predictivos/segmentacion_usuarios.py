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

import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn.decomposition import PCA

from preprocesamiento import codificar_perfil_onehot

sns.set_style("whitegrid")
sns.set_palette("Set2")

# Por defecto, silhouette_score intenta reservar hasta 1 GiB de
# memoria de una sola vez para la matriz de distancias por pares
# (relevante con una submuestra de 20.000 usuarios). En equipos con
# poca RAM libre esa reserva puede fallar (MemoryError). Se reduce
# el tamaño de bloque interno de scikit-learn: el resultado numerico
# es identico (se calcula por partes en vez de en un unico bloque),
# solo cambia el consumo de memoria pico.
sklearn.set_config(working_memory=64)

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
      - edad, perfil: variables descriptivas. NO se usan para formar
        los segmentos (ver FEATURES_CLUSTERING), solo para describirlos.
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

    # Tasa de ahorro media winsorizada (percentiles 1 y 99): K-Means
    # minimiza distancias euclídeas al cuadrado y es muy sensible a
    # valores extremos; la tasa media presenta una cola izquierda muy
    # larga (mínimo cercano a -1.000%, usuarios con gasto muy superior
    # a su ingreso) que, sin acotar, dominaría la formación de clústeres.
    lo, hi = perfil['tasa_ahorro_media'].quantile([0.01, 0.99])
    perfil['tasa_ahorro_media_w'] = perfil['tasa_ahorro_media'].clip(lo, hi)

    perfil = perfil.reset_index()
    return perfil


# Variables con las que se forman los segmentos: exclusivamente
# comportamentales (cuánto ahorra el usuario, cuán estable es su
# ahorro y cómo reparte su gasto). Se excluyen deliberadamente las
# variables demográficas (edad, perfil laboral) y el nivel de ingreso,
# para que los segmentos describan COMPORTAMIENTO financiero y no
# reproduzcan una variable ya conocida. Se evaluó también la variante
# con gasto absoluto por categoría (en euros, log1p): obtenía siluetas
# similares pero reproducía en mayor medida el nivel de ingreso y el
# perfil laboral (NMI con perfil más alto), por lo que se prefiere la
# composición relativa del gasto.
FEATURES_CLUSTERING = (['tasa_ahorro_media_w', 'volatilidad_ahorro']
                       + [f'pct_gasto_{c}' for c in CATEGORIAS])


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
    # Columnas sin variación entre segmentos (p. ej. perfil_freelance,
    # 0,3% de los usuarios) tienen desviación típica 0: z-score = 0.
    datos_z = ((datos - datos.mean()) / datos.std().replace(0, np.nan)).fillna(0)

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
# 4. PERSONALIZACIÓN DE RECOMENDACIONES POR SEGMENTO
# ============================================================

def analizar_recomendaciones_por_segmento(perfil, path_recomendaciones, path_kpi):
    """Relaciona los segmentos con la capa prescriptiva del TFM.

    La capa prescriptiva de modelos_predictivos.py usa un objetivo de
    ahorro ÚNICO para todos los usuarios (la mediana global de la tasa
    de ahorro). Aquí se evalúa una alternativa personalizada: tomar
    como objetivo de cada usuario la mediana de la tasa de ahorro de
    SU segmento (\"usuarios con un comportamiento como el tuyo ahorran
    X%\"). Se compara, por segmento, cuántos usuarios necesitarían un
    ajuste y qué brecha (en puntos porcentuales) tendrían que cerrar
    con cada criterio.

    Requiere haber ejecutado antes modelos_predictivos.py (genera
    kpi_usuarios_proyeccion.csv y recomendaciones_prescriptivas.csv).
    Devuelve None si esos archivos no existen.
    """
    if not (os.path.exists(path_recomendaciones) and os.path.exists(path_kpi)):
        return None
    kpi = pd.read_csv(path_kpi)[['user_id', 'tasa_ahorro_predicha_pct']]
    rec = pd.read_csv(path_recomendaciones)[['user_id', 'requiere_ajuste',
                                             'categoria_principal_sugerida']]
    d = (perfil[['user_id', 'cluster', 'tasa_ahorro_media']]
         .merge(kpi, on='user_id').merge(rec, on='user_id', how='left'))

    with open(os.path.join(DATA_DIR, 'perfil_ahorro_thresholds_tink.json')) as f:
        objetivo_global = json.load(f)['umbral_mediana_pct']
    objetivo_segmento = perfil.groupby('cluster')['tasa_ahorro_media'].median()
    d['objetivo_personalizado'] = d['cluster'].map(objetivo_segmento)

    d['brecha_global_pp'] = (objetivo_global - d['tasa_ahorro_predicha_pct']).clip(lower=0)
    d['brecha_personal_pp'] = (d['objetivo_personalizado'] - d['tasa_ahorro_predicha_pct']).clip(lower=0)

    filas = []
    for c, g in d.groupby('cluster'):
        con_ajuste = g[g['requiere_ajuste'] == True]
        cat_top = (con_ajuste['categoria_principal_sugerida'].value_counts(normalize=True)
                   if len(con_ajuste) else pd.Series(dtype=float))
        filas.append({
            'cluster': int(c),
            'usuarios_con_proyeccion': int(len(g)),
            'objetivo_personalizado_pct': round(float(objetivo_segmento[c]), 1),
            'pct_ajuste_objetivo_global': round(float((g['brecha_global_pp'] > 0).mean() * 100), 1),
            'pct_ajuste_objetivo_personalizado': round(float((g['brecha_personal_pp'] > 0).mean() * 100), 1),
            'brecha_mediana_global_pp': round(float(g.loc[g['brecha_global_pp'] > 0, 'brecha_global_pp'].median()), 1)
                if (g['brecha_global_pp'] > 0).any() else 0.0,
            'brecha_mediana_personal_pp': round(float(g.loc[g['brecha_personal_pp'] > 0, 'brecha_personal_pp'].median()), 1)
                if (g['brecha_personal_pp'] > 0).any() else 0.0,
            'categoria_mas_sugerida': (cat_top.index[0] if len(cat_top) else None),
            'pct_categoria_mas_sugerida': (round(float(cat_top.iloc[0] * 100), 1) if len(cat_top) else None),
        })
    tabla = pd.DataFrame(filas)
    return tabla, objetivo_global


def graficar_recomendaciones_por_segmento(tabla, objetivo_global, output_path, sin_proyeccion=()):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(tabla))
    w = 0.38
    b1 = ax.bar(x - w / 2, tabla['pct_ajuste_objetivo_global'], w, color='#BDBDBD',
                label=f'Objetivo global único ({objetivo_global:.1f}%)')
    b2 = ax.bar(x + w / 2, tabla['pct_ajuste_objetivo_personalizado'], w, color='#1A73E8',
                label='Objetivo personalizado (mediana del segmento)')
    ax.bar_label(b1, fmt='%.0f%%', padding=2, fontsize=8)
    ax.bar_label(b2, fmt='%.0f%%', padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Segmento {c}\n(obj. {o:.0f}%)' for c, o in
                        zip(tabla['cluster'], tabla['objetivo_personalizado_pct'])])
    ax.set_ylabel('% de usuarios que requieren ajuste')
    ax.set_ylim(0, 110)
    titulo = 'Recomendaciones por segmento: objetivo global vs. personalizado'
    if len(sin_proyeccion):
        titulo += (f"\n(segmento(s) {', '.join(map(str, sin_proyeccion))} sin proyección: "
                   "historial insuficiente para el Modelo 1-bis)")
    ax.set_title(titulo, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
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
    perfil = perfil.merge(df.groupby('user_id')[cols_perfil + ['profile_source']].first()
                          .reset_index(), on='user_id')
    print(f"\nPerfiles de usuario construidos: {len(perfil):,}")
    print(f"Variables de segmentación (solo comportamentales): {FEATURES_CLUSTERING}")

    X = perfil[FEATURES_CLUSTERING].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n--- Evaluación del número de clusters (k=2..8) ---")
    df_eval = evaluar_k(X_scaled)
    graficar_evaluacion_k(df_eval, os.path.join(DATA_DIR, 'segmentacion_evaluacion_k.png'))

    # Selección de k: máximo GLOBAL del coeficiente de silueta en todo
    # el rango evaluado (k=2..8), sin restringir el rango a priori.
    k_elegido = int(df_eval.loc[df_eval['silueta'].idxmax(), 'k'])
    print(f"\nK elegido: {k_elegido} (máximo global de la silueta en k=2..8)")

    km_final = KMeans(n_clusters=k_elegido, random_state=RANDOM_STATE, n_init=10)
    perfil['cluster'] = km_final.fit_predict(X_scaled)

    idx_sil = np.random.default_rng(RANDOM_STATE).choice(len(X_scaled), 20000, replace=False)
    sil_final = silhouette_score(X_scaled[idx_sil], perfil['cluster'].values[idx_sil])
    print(f"Silueta del modelo final (k={k_elegido}): {sil_final:.4f}")

    # Diagnóstico de dependencia respecto al generador sintético: en qué
    # medida los segmentos reproducen los 8 perfiles reales de origen
    # (profile_source) y el perfil laboral, que NO se usaron como input.
    nmi_origen = normalized_mutual_info_score(perfil['profile_source'], perfil['cluster'])
    nmi_perfil = normalized_mutual_info_score(perfil['perfil'], perfil['cluster'])
    print(f"NMI segmentos vs. perfil de origen Tink: {nmi_origen:.3f} | "
          f"vs. perfil laboral: {nmi_perfil:.3f}")
    tabla_origen = pd.crosstab(perfil['profile_source'], perfil['cluster'])
    print("\n--- Usuarios por perfil de origen y segmento ---")
    print(tabla_origen.to_string())

    descriptivas = (['ingreso_medio', 'ahorro_medio', 'tasa_ahorro_media', 'edad',
                     'n_meses_observados'] + cols_perfil)
    resumen = interpretar_clusters(perfil, FEATURES_CLUSTERING + descriptivas)
    print("\n--- Perfil medio por cluster ---")
    print(resumen.to_string())

    graficar_clusters_pca(X_scaled, perfil['cluster'].values,
                           os.path.join(DATA_DIR, 'segmentacion_clusters_pca.png'))
    graficar_perfil_clusters(resumen, FEATURES_CLUSTERING + ['ingreso_medio'] + cols_perfil,
                              os.path.join(DATA_DIR, 'segmentacion_perfil_clusters.png'))

    # Relación con la capa prescriptiva (personalización por segmento)
    resultado_rec = analizar_recomendaciones_por_segmento(
        perfil, os.path.join(DATA_DIR, 'recomendaciones_prescriptivas.csv'),
        os.path.join(DATA_DIR, 'kpi_usuarios_proyeccion.csv'))
    tabla_rec = None
    if resultado_rec is not None:
        tabla_rec, objetivo_global = resultado_rec
        print("\n--- Recomendaciones por segmento (objetivo global vs. personalizado) ---")
        print(tabla_rec.to_string(index=False))
        sin_proy = sorted(set(perfil['cluster']) - set(tabla_rec['cluster']))
        if sin_proy:
            print(f"  Segmentos sin ninguna proyección individual (historial inferior a 3 meses): {sin_proy}")
        graficar_recomendaciones_por_segmento(
            tabla_rec, objetivo_global, os.path.join(DATA_DIR, 'segmentacion_recomendaciones.png'),
            sin_proyeccion=sin_proy)
    else:
        print("\n(Se omite el análisis de recomendaciones por segmento: ejecutar antes "
              "modelos_predictivos.py)")

    # Guardar asignaciones y resumen (sin columnas duplicadas)
    cols_salida = list(dict.fromkeys(['user_id', 'cluster', 'profile_source'] +
                                     FEATURES_CLUSTERING + descriptivas))
    perfil[cols_salida].to_csv(os.path.join(DATA_DIR, 'metrics', 'segmentacion_usuarios.csv'),
                               index=False)

    diagnostico = {
        'metodo': 'K-Means sobre perfil comportamental agregado por usuario',
        'features_clustering': FEATURES_CLUSTERING,
        'features_descriptivas_no_usadas_en_clustering': descriptivas,
        'k_elegido': k_elegido,
        'criterio_seleccion_k': 'maximo global del coeficiente de silueta en k=2..8',
        'silueta_k_elegido': round(float(sil_final), 4),
        'evaluacion_k': df_eval.to_dict(orient='records'),
        'n_usuarios_segmentados': int(len(perfil)),
        'nmi_vs_perfil_origen_tink': round(float(nmi_origen), 3),
        'nmi_vs_perfil_laboral': round(float(nmi_perfil), 3),
        'usuarios_por_origen_y_segmento': {str(k): {str(c): int(v) for c, v in fila.items()}
                                           for k, fila in tabla_origen.iterrows()},
        'resumen_por_cluster': resumen.reset_index().to_dict(orient='records'),
        'recomendaciones_por_segmento': (tabla_rec.to_dict(orient='records')
                                         if tabla_rec is not None else None),
    }
    with open(os.path.join(DATA_DIR, 'segmentacion_diagnostico.json'), 'w', encoding='utf-8') as f:
        json.dump(diagnostico, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nArchivos guardados en: {DATA_DIR}")
    for nombre in ['segmentacion_evaluacion_k.png', 'segmentacion_clusters_pca.png',
                   'segmentacion_perfil_clusters.png', 'segmentacion_recomendaciones.png',
                   'segmentacion_diagnostico.json', 'metrics/segmentacion_usuarios.csv']:
        print(f"  - {nombre}")
