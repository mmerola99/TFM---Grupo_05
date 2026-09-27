# Resumen del Pipeline en `scripts_mejorados`

## 1. Que representa esta carpeta

La carpeta `scripts_mejorados/` concentra el flujo Tink actual del proyecto en el orden en que debe ejecutarse. La idea central es trabajar con transacciones bancarias individuales como unidad base, enriquecerlas con reglas de negocio y convertirlas en un dataset apto para entrenamiento.

Este flujo cubre tres necesidades:

1. generar o replicar datos tipo Tink,
2. construir una base transaccional util para analitica y ML,
3. entrenar un clasificador de categorias y, si hace falta, derivar despues a una vista mensual por usuario.

---

## 2. Resumen de cada script

### `scripts_mejorados/01_genera_payloads_tink_sinteticos.py`

Es el punto de entrada del pipeline. Genera payloads sinteticos con estructura tipo Tink a partir de muestras existentes.

Su funcion es:

1. leer ejemplos base de cuentas y transacciones,
2. construir perfiles sinteticos consistentes,
3. generar JSON por usuario con cuentas y movimientos,
4. crear un `manifest.json` que actua como indice de todo lo producido.

Entradas principales:

1. muestras dentro de `accounts/`.

Salidas principales:

1. `accounts/synthetic/manifest.json`,
2. archivos JSON sinteticos dentro de `accounts/synthetic/`.

Valor dentro del flujo:

1. permite probar el pipeline completo sin depender de una extraccion real desde banco o agregador,
2. deja una fuente estructurada y reproducible para las capas posteriores.

---

### `scripts_mejorados/02_prepara_transacciones_tink.py`

Es la capa 1 del flujo. Convierte los payloads tipo Tink en una tabla transaccional canonica con una fila por transaccion.

Su funcion es:

1. leer `manifest.json`,
2. cargar cuentas y movimientos asociados a cada usuario,
3. aplanar la estructura JSON,
4. combinar metadatos de cuenta con cada movimiento,
5. calcular campos basicos de fecha, importe y direccion.

Campos relevantes que consolida:

1. identificadores de usuario y cuenta,
2. `booked_date`, año, mes y dia,
3. `amount_signed` y `amount_abs`,
4. direccion `credit/debit`,
5. descripciones originales del movimiento.

Salidas principales:

1. `data/raw/transacciones_tink.csv`,
2. `data/clean/transacciones_tink.csv`.

Valor dentro del flujo:

1. define la base canonica de trabajo,
2. estandariza la informacion para que ya pueda consumirse desde analitica, reglas o modelos.

---

### `scripts_mejorados/03_enriquecer_transacciones_tink.py`

Es la capa 2 del flujo. Enriquece las transacciones con semantica de negocio y primeras etiquetas heuristicas.

Su funcion es:

1. normalizar el texto de las descripciones,
2. detectar merchants y aliases frecuentes,
3. inferir categorias generales de gasto o ingreso,
4. asignar una fuente de etiqueta y un nivel de confianza,
5. calcular señales agregadas por merchant.

Campos relevantes que genera:

1. `description_normalized`,
2. `merchant_normalized`,
3. `category_general`,
4. `label_source`,
5. `label_confidence`,
6. `merchant_tx_count`, `merchant_active_months`, `merchant_total_amount`,
7. `merchant_recurring_candidate`.

Salidas principales:

1. `data/raw/transacciones_tink_enriched.csv`,
2. `data/clean/transacciones_tink_enriched.csv`.

Valor dentro del flujo:

1. transforma texto bancario poco interpretable en señales utilizables,
2. crea la primera capa de etiquetado que luego alimenta el entrenamiento.

---

### `scripts_mejorados/04_prepara_dataset_entrenamiento_tink.py`

Es la capa 3 del flujo. Toma la salida enriquecida y decide que filas sirven para entrenar y cuales deben revisarse.

Su funcion es:

1. filtrar las transacciones con etiquetas suficientemente confiables,
2. excluir `uncategorized` y casos de baja confianza,
3. crear variables derivadas utiles para modelado,
4. calcular soporte por clase,
5. generar splits de `train`, `validation` y `test`,
6. enviar el resto a una cola de revision manual.

Salidas principales:

1. `data/raw/transacciones_tink_model_input.csv`,
2. `data/clean/transacciones_tink_model_input.csv`,
3. `data/raw/transacciones_tink_label_review.csv`.

Puntos importantes del estado actual:

1. el split ya no se hace fila por fila, sino por grupos semanticos,
2. esto reduce la fuga de informacion entre train, validation y test,
3. para clases con muy pocos grupos disponibles se usa un reparto mas cuidadoso para no colapsar toda la evaluacion en una sola clase.

Valor dentro del flujo:

1. separa datos entrenables de datos dudosos,
2. deja una base mucho mas honesta para evaluar el modelo.

---

### `scripts_mejorados/05_entrena_clasificador_categorias_tink.py`

Es la capa 5 del flujo. Entrena el clasificador de categorias sobre el dataset preparado en la capa anterior.

Su funcion es:

1. leer `transacciones_tink_model_input.csv`,
2. filtrar clases con soporte minimo,
3. construir features de texto, numericas, categoricas y booleanas,
4. entrenar de forma incremental por chunks,
5. evaluar en validation y test,
6. guardar artefactos de salida reutilizables.

Caracteristicas tecnicas actuales:

1. usa entrenamiento incremental para poder procesar el dataset completo,
2. usa hashing para texto y categoricas,
3. escala numericas de forma incremental,
4. serializa modelo, metricas y predicciones finales.

Salidas principales:

1. `data/clean/metricas_clasificador_categorias_tink.json`,
2. `data/clean/predicciones_clasificador_categorias_tink.csv`,
3. `data/clean/clasificador_categorias_tink_confusion.csv`,
4. `05_modelos/artifacts/clasificador_categorias_tink.pkl`.

Punto importante del estado actual:

1. la evaluacion perfecta anterior desaparecio tras corregir el split de la capa 4,
2. las metricas actuales reflejan mejor la dificultad real del dato sintetico,
3. el baseline ya sirve para medir de forma mas realista la calidad del pipeline y sus etiquetas heuristicas.

---

### `scripts_mejorados/06_prepara_usuarios_tink.py`

Es un script derivado, no un paso obligatorio del flujo principal. Convierte transacciones Tink en una vista mensual por usuario.

Su funcion es:

1. agrupar movimientos por usuario y por mes,
2. agregar ingresos y gastos por categorias,
3. producir una tabla mas cercana al enfoque clasico de usuario-mes,
4. servir de puente entre el pipeline transaccional y una salida agregada.

Salidas principales:

1. `data/raw/dataset_sintetico_usuarios_tink.csv`,
2. `data/clean/dataset_final_usuarios_tink.csv`.

Valor dentro del flujo:

1. permite volver a una logica de analisis agregado por usuario,
2. facilita comparaciones con el enfoque anterior si se necesitan,
3. no es necesario para entrenar el clasificador transaccional.

---

## 3. Como se integra todo

La integracion del pipeline queda asi:

1. `01_genera_payloads_tink_sinteticos.py` crea la materia prima con estructura tipo Tink,
2. `02_prepara_transacciones_tink.py` transforma esa materia prima en una tabla transaccional canonica,
3. `03_enriquecer_transacciones_tink.py` convierte cada transaccion en una observacion con semantica de negocio,
4. `04_prepara_dataset_entrenamiento_tink.py` separa las filas utiles para entrenamiento de las filas que requieren revision,
5. `05_entrena_clasificador_categorias_tink.py` aprende a predecir categorias sobre nuevas transacciones,
6. `06_prepara_usuarios_tink.py` solo entra si despues quieres una salida mensual agregada por usuario.

Visto como cadena de valor, el flujo hace esto:

1. genera datos bancarios estructurados,
2. los convierte en una base transaccional limpia,
3. les añade reglas y etiquetas iniciales,
4. prepara un dataset entrenable con evaluacion controlada,
5. entrena un modelo reutilizable,
6. opcionalmente deriva la informacion a una vista agregada por usuario.

---

## 4. Orden recomendado de ejecucion

Si quieres ejecutar el pipeline principal, el orden recomendado es:

1. `scripts_mejorados/01_genera_payloads_tink_sinteticos.py`,
2. `scripts_mejorados/02_prepara_transacciones_tink.py`,
3. `scripts_mejorados/03_enriquecer_transacciones_tink.py`,
4. `scripts_mejorados/04_prepara_dataset_entrenamiento_tink.py`,
5. `scripts_mejorados/05_entrena_clasificador_categorias_tink.py`.

Si ademas necesitas una salida agregada por usuario, entonces al final ejecutas:

6. `scripts_mejorados/06_prepara_usuarios_tink.py`.

---

## 5. Resumen final

La carpeta `scripts_mejorados/` ya define un pipeline completo y coherente.

Empieza generando o replicando datos tipo Tink, los convierte en transacciones canonicas, las enriquece con reglas de negocio, prepara un dataset entrenable y finalmente entrena un clasificador de categorias. Como salida opcional, tambien puede reconstruir una vista mensual por usuario.

La idea central del flujo es que la unidad principal de trabajo ya no es el usuario agregado, sino la transaccion bancaria individual. Desde ahi se construyen tanto el modelo transaccional como, si hace falta, la capa agregada posterior.