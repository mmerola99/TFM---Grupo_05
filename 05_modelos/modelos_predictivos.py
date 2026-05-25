# ============================================================
# modelos_predictivos.py
# Modelos de Machine Learning — AI Financial Life Coach
#
# Modelo 1: Regresión Lineal — predicción del ahorro mensual
# Modelo 2: Regresión Logística — clasificación perfil ahorro
# Modelo 3: Serie temporal — proyección ahorro 6 meses
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from sklearn.model_selection import train_test_split
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

print("=" * 60)
print("MODELOS PREDICTIVOS — AI Financial Life Coach")
print("=" * 60)

# ── CARGA DE DATOS ───────────────────────────────────────────
df = pd.read_csv(f"{DATA_DIR}/dataset_sintetico_usuarios.csv")
print(f"\nDataset: {df.shape[0]:,} registros | {df['user_id'].nunique()} usuarios")
print(f"Período: {df['fecha'].min()} — {df['fecha'].max()}")


# ============================================================
# MODELO 1 — REGRESIÓN LINEAL
# Objetivo: predecir el ahorro mensual de un usuario
# Variables predictoras: salario, gasto_total, edad, ipc_mensual
# Variable objetivo: ahorro (€)
# ============================================================

print("\n" + "="*60)
print("MODELO 1 — Regresión Lineal: Predicción del ahorro mensual")
print("="*60)

features_lr = ['salario', 'vivienda', 'alimentacion', 'transporte',
               'ocio', 'salud', 'educacion', 'otros', 'edad', 'ipc_mensual']
target_lr   = 'ahorro'

X = df[features_lr]
y = df[target_lr]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

lr_model = LinearRegression()
lr_model.fit(X_train_s, y_train)
y_pred_lr = lr_model.predict(X_test_s)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr  = r2_score(y_test, y_pred_lr)

print(f"\n  Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"  MAE:  {mae_lr:.2f} € (error medio absoluto)")
print(f"  R²:   {r2_lr:.4f} (varianza explicada)")

# Gráfico 1: Real vs Predicho
fig, ax = plt.subplots(figsize=(8, 5))
muestra = np.random.choice(len(y_test), min(500, len(y_test)), replace=False)
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
print("  ✅ Figura guardada: modelo1_regresion_lineal.png")

# Gráfico 2: Importancia de variables (coeficientes)
coef_df = pd.DataFrame({
    'variable':    features_lr,
    'coeficiente': lr_model.coef_
}).sort_values('coeficiente', key=abs, ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#EA4335' if c < 0 else '#34A853' for c in coef_df['coeficiente']]
ax.barh(coef_df['variable'], coef_df['coeficiente'], color=colors, alpha=0.85)
ax.axvline(0, color='gray', linewidth=0.8)
ax.set_title('Modelo 1 — Importancia de variables (coeficientes estandarizados)',
             fontweight='bold')
ax.set_xlabel('Coeficiente (impacto en el ahorro)')
plt.tight_layout()
plt.savefig(f"{CLEAN_DIR}/modelo1_coeficientes.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Figura guardada: modelo1_coeficientes.png")


# ============================================================
# MODELO 2 — REGRESIÓN LOGÍSTICA
# Objetivo: clasificar el perfil de ahorro del usuario
# Clases: buen_ahorrador / ahorro_moderado / ahorro_insuficiente
# ============================================================

print("\n" + "="*60)
print("MODELO 2 — Regresión Logística: Clasificación perfil ahorro")
print("="*60)

# Codificar variable categórica perfil
le = LabelEncoder()
df['perfil_encoded'] = le.fit_transform(df['perfil'])

features_clf = ['salario', 'vivienda', 'alimentacion', 'transporte',
                'ocio', 'salud', 'educacion', 'otros',
                'edad', 'ipc_mensual', 'perfil_encoded']
target_clf   = 'perfil_ahorro'

X2 = df[features_clf]
y2 = df[target_clf]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2)

scaler2 = StandardScaler()
X2_train_s = scaler2.fit_transform(X2_train)
X2_test_s  = scaler2.transform(X2_test)

clf_model = LogisticRegression(max_iter=1000, random_state=42,
                                class_weight='balanced')
clf_model.fit(X2_train_s, y2_train)
y2_pred = clf_model.predict(X2_test_s)

acc = accuracy_score(y2_test, y2_pred)
print(f"\n  Train: {len(X2_train):,} | Test: {len(X2_test):,}")
print(f"  Accuracy: {acc:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y2_test, y2_pred))

# Gráfico 3: Matriz de confusión
fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y2_test, y2_pred, labels=clf_model.classes_)
disp = ConfusionMatrixDisplay(cm, display_labels=clf_model.classes_)
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Modelo 2 — Matriz de confusión (clasificación perfil ahorro)',
             fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CLEAN_DIR}/modelo2_confusion.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Figura guardada: modelo2_confusion.png")


# ============================================================
# MODELO 3 — SERIE TEMPORAL
# Objetivo: proyectar el ahorro medio de los usuarios
#           para los próximos 6 meses (junio–noviembre 2026)
# Método: regresión lineal sobre tendencia + estacionalidad
#         (simplificado, sin Prophet para evitar dependencias Stan)
# ============================================================

print("\n" + "="*60)
print("MODELO 3 — Serie Temporal: Proyección ahorro 6 meses")
print("="*60)

# Calcular ahorro medio mensual de todos los usuarios
ahorro_medio = (df.groupby('fecha')['ahorro']
                .mean()
                .reset_index()
                .rename(columns={'ahorro': 'ahorro_medio'}))
ahorro_medio['fecha_dt'] = pd.to_datetime(ahorro_medio['fecha'])
ahorro_medio = ahorro_medio.sort_values('fecha_dt').reset_index(drop=True)
ahorro_medio['t'] = range(len(ahorro_medio))
ahorro_medio['mes_num'] = ahorro_medio['fecha_dt'].dt.month

# Añadir variables estacionales (seno/coseno del mes)
ahorro_medio['sin_mes'] = np.sin(2 * np.pi * ahorro_medio['mes_num'] / 12)
ahorro_medio['cos_mes'] = np.cos(2 * np.pi * ahorro_medio['mes_num'] / 12)

# Entrenar sobre datos históricos (hasta mayo 2026)
X_ts = ahorro_medio[['t', 'sin_mes', 'cos_mes']]
y_ts = ahorro_medio['ahorro_medio']

ts_model = LinearRegression()
ts_model.fit(X_ts, y_ts)

# Proyectar 6 meses futuros: junio–noviembre 2026
fechas_futuras = pd.date_range('2026-06-01', periods=6, freq='MS')
t_futuro = range(len(ahorro_medio), len(ahorro_medio) + 6)
meses_futuros = fechas_futuras.month

X_futuro = pd.DataFrame({
    't':       list(t_futuro),
    'sin_mes': np.sin(2 * np.pi * meses_futuros / 12),
    'cos_mes': np.cos(2 * np.pi * meses_futuros / 12),
})

predicciones = ts_model.predict(X_futuro)

df_pred = pd.DataFrame({
    'fecha':          [f.strftime('%Y-%m') for f in fechas_futuras],
    'ahorro_predicho': predicciones.round(2),
    'tipo':           'prediccion'
})

print(f"\n  Proyección ahorro medio mensual (Jun–Nov 2026):")
for _, row in df_pred.iterrows():
    print(f"  {row['fecha']}: {row['ahorro_predicho']:.2f} €")

# Gráfico 4: Serie histórica + proyección
fig, ax = plt.subplots(figsize=(12, 5))

# Datos históricos
ax.plot(ahorro_medio['fecha_dt'],
        ahorro_medio['ahorro_medio'],
        color='#1A73E8', linewidth=2, marker='o', markersize=3,
        label='Ahorro medio histórico')

# Ajuste del modelo sobre histórico
y_fitted = ts_model.predict(X_ts)
ax.plot(ahorro_medio['fecha_dt'], y_fitted,
        color='#34A853', linewidth=1.5, linestyle='--',
        label='Tendencia del modelo')

# Proyección futura
ax.plot(fechas_futuras, predicciones,
        color='#EA4335', linewidth=2.5, marker='s', markersize=6,
        linestyle='--', label='Proyección Jun–Nov 2026')

# Zona de proyección sombreada
ax.axvspan(fechas_futuras[0], fechas_futuras[-1],
           alpha=0.08, color='#EA4335', label='Período proyectado')
ax.axvline(pd.Timestamp('2026-05-01'), color='gray',
           linestyle=':', linewidth=1.5)
ax.annotate('Hoy\n(mayo 2026)',
            xy=(pd.Timestamp('2026-05-01'), ahorro_medio['ahorro_medio'].iloc[-1]),
            xytext=(15, 10), textcoords='offset points',
            fontsize=8, color='gray')

ax.set_title('Modelo 3 — Proyección del ahorro medio mensual (Jun–Nov 2026)',
             fontweight='bold')
ax.set_xlabel('Fecha')
ax.set_ylabel('Ahorro medio (€)')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{CLEAN_DIR}/modelo3_proyeccion.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Figura guardada: modelo3_proyeccion.png")

# ── GUARDAR MÉTRICAS ─────────────────────────────────────────
metricas = {
    'modelo1_mae':      round(mae_lr, 2),
    'modelo1_r2':       round(r2_lr, 4),
    'modelo2_accuracy': round(acc, 4),
    'modelo3_predicciones': df_pred[['fecha','ahorro_predicho']].to_dict('records'),
    'clases_modelo2':   list(clf_model.classes_),
}
with open(f"{CLEAN_DIR}/metricas_modelos.json", 'w') as f:
    json.dump(metricas, f, indent=2)

# ── GUARDAR DATASET LIMPIO ───────────────────────────────────
df.to_csv(f"{CLEAN_DIR}/dataset_final_usuarios.csv", index=False)
ahorro_medio.to_csv(f"{CLEAN_DIR}/ahorro_medio_mensual.csv", index=False)

print("\n" + "="*60)
print("RESUMEN FINAL")
print("="*60)
print(f"  Modelo 1 — Regresión Lineal:")
print(f"    MAE = {mae_lr:.2f}€ | R² = {r2_lr:.4f}")
print(f"  Modelo 2 — Regresión Logística:")
print(f"    Accuracy = {acc:.4f}")
print(f"  Modelo 3 — Proyección temporal:")
print(f"    Ahorro proyectado Jun 2026: {predicciones[0]:.2f}€")
print(f"    Ahorro proyectado Nov 2026: {predicciones[-1]:.2f}€")
print(f"\n  Archivos guardados en: {CLEAN_DIR}")
