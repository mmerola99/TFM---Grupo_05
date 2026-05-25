# AI Financial Life Coach — Obtención y Preparación de Datos

**Asignatura 5: Obtención de Datos para el TFM**
**Máster en Big Data & Business Intelligence — Next Educación**

**Equipo:** Carlos Alfonso Cuaya Xinto · Cinthya Solis Meza · Jorge Manuel Caceres Mondragon · Marco Merola · Miguel Angel Lozano Torres

---

## Descripción del proyecto

Este repositorio contiene el código y los datos utilizados para la obtención, limpieza y análisis exploratorio del contexto financiero del mercado objetivo de **AI Financial Life Coach**: jóvenes profesionales de 25-40 años en España.

Los datos han sido recopilados mediante dos vías obligatorias:
- **API REST (Eurostat):** tasas de ahorro, desempleo y precios al consumo para España
- **Web Scraping (Banco de España / Wikipedia):** Euríbor histórico y morosidad bancaria

---

## Estructura del repositorio

```
tfm_ai_financial_life_coach/
├── README.md
├── 01_scraping/
│   └── scraping_bde.py          ← Web scraping: Euríbor + Morosidad BdE
├── 02_api/
│   └── api_eurostat.py          ← API Eurostat: Ahorro, Desempleo, IPC
├── 03_limpieza/
│   └── limpieza_preparacion.py  ← Limpieza, estandarización y dataset final
├── 04_eda/
│   └── analisis_exploratorio.py ← EDA completo con 5 visualizaciones
├── 05_streamlit/
│   └── app.py                   ← Dashboard interactivo Streamlit
└── data/
    ├── raw/                     ← Datos originales sin procesar
    └── clean/                   ← Dataset final limpio + gráficos EDA
```

---

## Cómo ejecutar el proyecto

### 1. Instalar dependencias

```bash
pip install requests beautifulsoup4 pandas numpy matplotlib seaborn streamlit
```

### 2. Ejecutar los notebooks en orden

```bash
# Paso 1: Obtención via API
python 02_api/api_eurostat.py

# Paso 2: Web Scraping
python 01_scraping/scraping_bde.py

# Paso 3: Limpieza y preparación
python 03_limpieza/limpieza_preparacion.py

# Paso 4: Análisis exploratorio (genera gráficos en data/clean/)
python 04_eda/analisis_exploratorio.py

# Paso 5: App Streamlit (desde la carpeta 05_streamlit)
cd 05_streamlit
streamlit run app.py --server.port 8080
```

---

## Fuentes de datos

| Fuente | Método | Dataset | Indicador |
|--------|--------|---------|-----------|
| Eurostat | API REST | nasq_10_ki | Tasa de ahorro bruto hogares España (%) |
| Eurostat | API REST | une_rt_a | Tasa desempleo 25-34 años España (%) |
| Eurostat | API REST | prc_hicp_manr | IPC variación anual España (%) |
| Banco de España | Web Scraping | Boletín Estadístico Cap.4 | Morosidad bancaria mensual (%) |
| Wikipedia / BdE | Web Scraping | Euríbor histórico | Euríbor 12 meses media anual (%) |

---

## Variables del dataset final (data/clean/dataset_final.csv)

| Variable | Descripción | Unidad |
|----------|-------------|--------|
| anio | Año de referencia | 2008–2024 |
| tasa_ahorro_pct | Tasa de ahorro bruto hogares | % |
| tasa_desempleo | Tasa de desempleo 25-34 años | % |
| ipc_variacion_anual_pct | Variación anual del IPC | % |
| euribor_pct | Euríbor 12 meses media anual | % |
| tasa_morosidad_pct | Morosidad bancaria | % |

---

## Hallazgos principales del EDA

1. La morosidad bancaria cerró 2024 en el 3,32% — mínimo desde 2008
2. La tasa de ahorro se estabilizó en el 12,7% en 2024, por encima de la media histórica pre-pandemia
3. El Euríbor inició su descenso en 2024 tras el pico del 4,07% en 2023
4. El desempleo del segmento 25-34 años bajó al 16,9% — su mínimo histórico
5. Correlación positiva entre Euríbor y morosidad; negativa entre ahorro y desempleo

---

## Relación con el TFM

Los datos obtenidos constituyen la base del análisis de mercado del TFM:
**"AI Financial Life Coach — Plataforma de gestión financiera personal con ML para el mercado español"**

El contexto macroeconómico analizado (ahorro, euríbor, morosidad, desempleo juvenil) justifica la oportunidad de negocio y alimenta las variables de entrada del modelo predictivo de la plataforma.
