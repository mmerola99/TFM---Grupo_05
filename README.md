# AI Financial Life Coach — Obtención de Datos para el TFM

**Asignatura 5: Obtención de Datos para el TFM**  
**Máster en Big Data y Business Intelligence — Next Educación**

**Equipo:** Carlos Alfonso Cuaya Xinto · Cinthya Solis Meza · Jorge Manuel Caceres Mondragon · Marco Merola · Miguel Angel Lozano Torres

---

## Descripción del proyecto

Este repositorio contiene el código y los datos utilizados para la obtención, limpieza, análisis exploratorio y modelado predictivo del contexto financiero del mercado objetivo de **AI Financial Life Coach**: jóvenes profesionales de 25-40 años en España.

Los datos han sido recopilados mediante dos vías obligatorias (API y web scraping) y complementados con un dataset sintético de 1.000 usuarios calibrado sobre estadísticas reales del INE y Eurostat 2025.

---

## Estructura del repositorio

| Carpeta / Archivo | Descripción |
|---|---|
| `01_scraping/scraping_bde.py` | Web scraping: Euríbor histórico + Morosidad BdE |
| `02_api/api_eurostat.py` | API Eurostat: Tasa de ahorro, Desempleo, IPC |
| `02_api/genera_dataset_sintetico.py` | Generación dataset sintético 41.000 registros |
| `03_limpieza/limpieza_preparacion.py` | Limpieza, estandarización y dataset consolidado |
| `04_eda/analisis_exploratorio.py` | Análisis exploratorio con 5 visualizaciones |
| `05_modelos/modelos_predictivos.py` | 3 modelos de ML (Regresión Lineal, Logística, Serie temporal) |
| `05_streamlit/app.py` | Dashboard predictivo interactivo con simulador |
| `data/raw/` | CSV de datos originales sin procesar |
| `data/clean/` | Dataset limpio + gráficos PNG + métricas JSON |             ← Dataset limpio + gráficos + métricas
---

## Cómo ejecutar el proyecto

### 1. Instalar dependencias

```bash
pip install requests beautifulsoup4 pandas numpy matplotlib seaborn streamlit scikit-learn
```

### 2. Ejecutar los scripts en orden

```bash
# Paso 1: Extracción via API
python 02_api/api_eurostat.py

# Paso 2: Web Scraping
python 01_scraping/scraping_bde.py

# Paso 3: Limpieza y preparación
python 03_limpieza/limpieza_preparacion.py

# Paso 4: Generación del dataset sintético
python 02_api/genera_dataset_sintetico.py

# Paso 5: Análisis exploratorio (genera gráficos en data/clean/)
python 04_eda/analisis_exploratorio.py

# Paso 6: Modelos predictivos
python 05_modelos/modelos_predictivos.py

# Paso 7: App Streamlit
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
| Generado por el grupo | Dataset sintético | INE + Eurostat 2025 | 41.000 registros de 1.000 usuarios |

---

## Modelos predictivos

| Modelo | Objetivo | Resultado |
|--------|----------|-----------|
| Regresión Lineal | Predecir ahorro mensual (€) | R²=1,0000 / MAE=0€ |
| Regresión Logística | Clasificar perfil de ahorro | Accuracy=96,3% / Recall=100% |
| Serie temporal | Proyectar ahorro 6 meses | 370€ (jun 2026) → 354€ (nov 2026) |

---

## Relación con el TFM

Los datos y modelos obtenidos constituyen la base del TFM:  
**"AI Financial Life Coach — Plataforma de gestión financiera personal con ML para el mercado español"**