"""
06b_calibra_gasto_estructural.py
================================

Imputacion de gastos estructurales no capturados por el sandbox de Tink.

Motivacion
----------
Los perfiles de prueba del sandbox de Tink solo registran una parte del gasto
real (sobre todo pagos con tarjeta): en el dataset generado por el script 06
el 93,6 % de los meses no tiene gasto en vivienda y no hay ningun gasto en
salud ni en educacion. Como la tasa de ahorro se calcula como
(ingresos - gasto observado) / ingresos, el resultado (mediana del 88 %) es
irrealista: la tasa de ahorro de los hogares espanoles fue del 12 % de la renta
disponible en 2025 (INE).

Metodo
------
1. Se asigna a cada usuario una tasa de ahorro objetivo estable, centrada en el
   12 % (INE, 2025) y diferenciada por perfil profesional, con dispersion
   individual y algunos casos negativos.
2. A partir de su salario medio se obtiene su gasto mensual objetivo y se
   reparte entre categorias con la estructura del gasto de los hogares de la
   Encuesta de Presupuestos Familiares 2024 (INE).
3. En cada categoria se calcula la parte que falta respecto al gasto ya
   observado en esa categoria y se escala para que el total imputado no supere
   lo que falta para alcanzar el gasto objetivo (nunca se reduce el gasto
   observado). La
   vivienda es un importe fijo mensual; el resto lleva una pequena variacion
   mensual.
4. Se recalculan gasto_total, ahorro, tasa_ahorro_pct y perfil_ahorro, y se
   actualiza el umbral (mediana) usado por los Modelos 2 y la capa prescriptiva.

El proceso es determinista (semilla fija por usuario) y reproducible.
Entrada y salida: data/clean/dataset_final_usuarios_tink.csv (la version
original se guarda en data/clean/_v1_sin_calibrar/).
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
RAW = ROOT / "data" / "raw"
BACKUP = CLEAN / "_v1_sin_calibrar"
DATASET = CLEAN / "dataset_final_usuarios_tink.csv"
THRESHOLDS = CLEAN / "perfil_ahorro_thresholds_tink.json"
REPORT = CLEAN / "calibracion_gasto_report.json"

CATS = ["vivienda", "alimentacion", "transporte", "ocio", "salud", "educacion", "otros"]

# Estructura del gasto de los hogares, EPF 2024 (INE), agrupada en las
# categorias del proyecto (% del gasto total):
#   vivienda     = vivienda, agua, electricidad, gas (32,4) + muebles y hogar (3,7)
#   alimentacion = alimentos y bebidas no alcoholicas (15,8) + restaurantes (9,9)
#                  + bebidas alcoholicas y tabaco (1,0)
#   transporte   = transporte (11,4)
#   ocio         = ocio, deporte y cultura (5,0)
#   salud        = sanidad (4,0)
#   educacion    = ensenanza (1,6)
#   otros        = vestido y calzado (4,2) + informacion y comunicaciones (3,2)
#                  + seguros y servicios financieros (3,7) + cuidado personal y otros (3,6)
EPF_2024 = {"vivienda": 36.1, "alimentacion": 26.7, "transporte": 11.4, "ocio": 5.0,
            "salud": 4.0, "educacion": 1.6, "otros": 14.7}
W = {k: v / sum(EPF_2024.values()) for k, v in EPF_2024.items()}

# Tasa de ahorro objetivo por perfil (%). Media ponderada ~12 % (INE 2025).
TASA_OBJETIVO = {"junior": 7.0, "medio": 11.0, "senior": 16.0, "freelance": 10.0}
DISPERSION_USUARIO = 8.0      # desviacion tipica entre usuarios (p.p.)
LIMITES_TASA = (-25.0, 45.0)  # recorte de valores extremos
VARIACION_MENSUAL = 0.06      # variacion mensual del gasto imputado no fijo


def rng_usuario(user_id):
    seed = int(hashlib.md5(f"afilc-{user_id}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def main():
    BACKUP.mkdir(exist_ok=True)
    if not (BACKUP / DATASET.name).exists():
        shutil.copy2(DATASET, BACKUP / DATASET.name)
        if THRESHOLDS.exists():
            shutil.copy2(THRESHOLDS, BACKUP / THRESHOLDS.name)
    df = pd.read_csv(BACKUP / DATASET.name)  # siempre parte de la version sin calibrar
    antes = {"tasa_mediana": float(df.tasa_ahorro_pct.median()),
             "tasa_agregada": float(df.ahorro.sum() / df.salario.sum() * 100),
             "gasto_mediano": float(df.gasto_total.median())}

    obs_medio = df.groupby("user_id")[CATS].mean()
    sal_medio = df.groupby("user_id")["salario"].mean()
    perfil = df.groupby("user_id")["perfil"].first()

    imput_base = {}
    tasa_obj = {}
    for uid in obs_medio.index:
        rng = rng_usuario(uid)
        t = np.clip(rng.normal(TASA_OBJETIVO.get(perfil[uid], 12.0), DISPERSION_USUARIO), *LIMITES_TASA)
        tasa_obj[uid] = t
        gasto_obj = sal_medio[uid] * (1 - t / 100)
        # deficit de cada categoria frente a la estructura EPF; el total imputado
        # se limita a lo que falta para alcanzar el gasto objetivo
        deficit = {c: max(0.0, gasto_obj * W[c] - obs_medio.at[uid, c]) for c in CATS}
        falta = max(0.0, gasto_obj - obs_medio.loc[uid].sum())
        total_def = sum(deficit.values())
        escala = min(1.0, falta / total_def) if total_def > 0 else 0.0
        imput_base[uid] = {c: deficit[c] * escala for c in CATS}
    base = pd.DataFrame(imput_base).T  # importe mensual imputado por usuario y categoria

    # variacion mensual (determinista) salvo vivienda, que es un importe fijo
    rng = np.random.default_rng(2026)
    imp = base.loc[df.user_id].reset_index(drop=True)
    for c in CATS:
        if c != "vivienda":
            imp[c] = imp[c] * np.clip(1 + rng.normal(0, VARIACION_MENSUAL, len(df)), 0.7, 1.3)
    for c in CATS:
        df[c] = (df[c] + imp[c].values).round(2)

    df["gasto_total"] = df[CATS].sum(axis=1).round(2)
    df["ahorro"] = (df.salario - df.gasto_total).round(2)
    df["tasa_ahorro_pct"] = np.where(df.salario > 0, (df.ahorro / df.salario * 100).round(2), 0.0)
    umbral = round(float(df.tasa_ahorro_pct.median()), 2)
    df["perfil_ahorro"] = np.where(df.tasa_ahorro_pct >= umbral, "ahorro_adecuado", "ahorro_insuficiente")
    df.to_csv(DATASET, index=False)
    raw_copy = RAW / "dataset_sintetico_usuarios_tink.csv"
    if raw_copy.parent.exists():
        df.to_csv(raw_copy, index=False)

    with open(THRESHOLDS, "w") as f:
        json.dump({
            "esquema": "binario", "umbral_mediana_pct": umbral,
            "clases": ["ahorro_insuficiente", "ahorro_adecuado"],
            "metodo": "mediana empirica de tasa_ahorro_pct tras la calibracion del gasto estructural (06b)",
        }, f, indent=2)

    despues = {"tasa_mediana": float(df.tasa_ahorro_pct.median()),
               "tasa_agregada": float(df.ahorro.sum() / df.salario.sum() * 100),
               "gasto_mediano": float(df.gasto_total.median()),
               "estructura_gasto_pct": (df[CATS].sum() / df.gasto_total.sum() * 100).round(1).to_dict(),
               "tasa_agregada_por_perfil": (df.groupby("perfil").ahorro.sum() / df.groupby("perfil").salario.sum() * 100).round(1).to_dict(),
               "pct_meses_ahorro_negativo": float((df.ahorro < 0).mean() * 100)}
    json.dump({"antes": antes, "despues": despues, "umbral_mediana_pct": umbral,
               "referencias": ["INE, Encuesta de Presupuestos Familiares 2024",
                               "INE, Cuentas trimestrales no financieras de los sectores institucionales (tasa de ahorro de los hogares 2025: 12 %)"]},
              open(REPORT, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({"antes": antes, "despues": despues, "umbral": umbral}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
