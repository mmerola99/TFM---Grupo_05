# Pipeline Tink Paso a Paso

## Objetivo general

Este flujo incorpora una cadena de procesamiento orientada a un escenario de Open Banking con Tink.

La propuesta consiste en:

1. generar o recibir payloads con estructura parecida a Tink,
2. convertirlos en una tabla transaccional canónica,
3. enriquecer esas transacciones con señales de negocio,
4. preparar datos aptos para entrenamiento,
5. entrenar un clasificador de categorías sobre transacciones.

## Visión de negocio

### Enfoque

El flujo parte de movimientos bancarios individuales. Eso permite:

1. trabajar con una granularidad realista,
2. construir categorías de gasto e ingreso sobre texto y contexto de cada transacción,
3. entrenar modelos reutilizables para nuevas transacciones,
4. mantener una separación clara entre datos base, reglas heurísticas y modelos.

---

## Flujo principal

## Paso 1. Generación de payloads Tink sintéticos

### Script

`02_api/genera_payloads_tink_sinteticos.py`

### Explicación de negocio

Este paso crea usuarios sintéticos con cuentas y transacciones que imitan la forma de respuesta de Tink. Sirve para probar el pipeline completo aunque todavía no se disponga de una extracción masiva real desde un banco o desde un agregador.

Su valor de negocio es que permite desarrollar y validar todo el producto de datos sin depender al 100% del acceso a producción o a un sandbox externo en cada iteración.

### Explicación técnica

El script:

1. lee muestras base desde la carpeta `accounts/`,
2. detecta payloads de cuentas y de transacciones,
3. construye perfiles sintéticos a partir de esas muestras,
4. genera nuevos usuarios manteniendo una estructura compatible con Tink,
5. escribe archivos JSON por usuario y un `manifest.json` que actúa como índice de todo lo generado.

Puntos técnicos importantes:

1. usa semilla aleatoria para reproducibilidad,
2. genera IBANs y ids estables,
3. conserva señales como idioma, tipos de cuenta, balances, descripciones y fechas,
4. hace checkpoints periódicos para no perder progreso en corridas grandes.

### Entrada principal

- Muestras JSON en `accounts/`

### Salida principal

- Payloads sintéticos en `accounts/synthetic/`
- Índice en `accounts/synthetic/manifest.json`

---

## Paso 2. Extracción canónica de transacciones

### Script

`03_limpieza/prepara_transacciones_tink.py`

### Explicación de negocio

Este es el primer paso que convierte los payloads en una tabla utilizable para analítica y ML. Aquí se define la base operativa del sistema: una fila por transacción.

Desde negocio, esto permite organizar la información a partir de eventos financieros reales: cada pago, cada ingreso y cada retirada.

### Explicación técnica

El script:

1. lee `accounts/synthetic/manifest.json`,
2. carga las cuentas de cada usuario,
3. carga todos sus archivos de transacciones,
4. aplana la estructura JSON a una tabla tabular,
5. combina metadatos de cuenta con cada transacción,
6. calcula campos derivados básicos.

Campos clave generados o consolidados:

1. `user_id` y `user_label`,
2. `account_id`, `account_type`, `account_currency`, balances e IBAN,
3. `booked_date`, año, mes, día y día de la semana,
4. `amount_signed`, `amount_abs`, dirección `credit/debit`,
5. descripciones originales y visibles,
6. estado de la transacción y metadatos del proveedor.

### Entrada principal

- `accounts/synthetic/manifest.json`
- Payloads de cuentas y transacciones referenciados en el manifest

### Salida principal

- `data/raw/transacciones_tink.csv`
- `data/clean/transacciones_tink.csv`

---

## Paso 3. Enriquecimiento semántico de transacciones

### Script

`03_limpieza/enriquecer_transacciones_tink.py`

### Explicación de negocio

La respuesta cruda de Tink no trae necesariamente una categorización útil para el caso de uso final. Este paso añade una primera capa de interpretación del movimiento.

Desde negocio, aquí se empieza a responder preguntas como:

1. ¿esto parece salario, compra, alquiler o suscripción?,
2. ¿qué merchant o entidad aparece realmente detrás del texto?,
3. ¿hay patrones de recurrencia que luego puedan servir para ahorro, presupuesto o alertas?

### Explicación técnica

El script:

1. normaliza el texto de las descripciones,
2. detecta aliases de merchants frecuentes,
3. aplica reglas heurísticas por texto y por dirección `credit/debit`,
4. asigna una taxonomía general estable,
5. añade features agregadas por merchant y usuario.

Salidas semánticas importantes:

1. `description_normalized`,
2. `merchant_normalized`,
3. `category_general`,
4. `label_source`,
5. `label_confidence`,
6. `merchant_tx_count`, `merchant_active_months`, `merchant_total_amount`,
7. `merchant_recurring_candidate`.

La taxonomía actual incluye categorías como:

1. `income_salary`,
2. `income_interest`,
3. `income_transfer`,
4. `housing`,
5. `groceries`,
6. `dining`,
7. `transport`,
8. `subscriptions`,
9. `shopping`,
10. `insurance`,
11. `cash_withdrawal`,
12. `uncategorized`.

### Entrada principal

- `data/raw/transacciones_tink.csv`

### Salida principal

- `data/raw/transacciones_tink_enriched.csv`
- `data/clean/transacciones_tink_enriched.csv`

---

## Paso 4. Preparación del dataset de entrenamiento

### Script

`03_limpieza/prepara_dataset_entrenamiento_tink.py`

### Explicación de negocio

No todas las etiquetas heurísticas tienen la misma calidad. Este paso separa lo que ya es suficientemente confiable para entrenar de lo que todavía conviene revisar manualmente.

Desde negocio, esto es importante porque evita entrenar un modelo final con ruido innecesario. Las reglas se utilizan como punto de partida para acelerar el etiquetado inicial.

### Explicación técnica

El script:

1. lee el dataset enriquecido de capa 2,
2. convierte columnas numéricas y completa valores faltantes básicos,
3. crea variables derivadas útiles para modelado,
4. filtra filas con etiquetas aceptables,
5. envía el resto a una cola de revisión,
6. asigna splits `train`, `validation` y `test` de forma reproducible.

Criterios principales:

1. las filas `uncategorized` no entran al entrenamiento,
2. las filas con confianza por debajo del umbral tampoco,
3. se marca el motivo de revisión con `review_reason`,
4. se calcula el soporte por clase con `target_class_count`,
5. se identifica si una clase es rara con `target_is_rare`.

### Entrada principal

- `data/raw/transacciones_tink_enriched.csv`

### Salida principal

- `data/raw/transacciones_tink_model_input.csv`
- `data/clean/transacciones_tink_model_input.csv`
- `data/raw/transacciones_tink_label_review.csv`

---

## Paso 5. Entrenamiento del clasificador de categorías

### Script

`05_modelos/entrena_clasificador_categorias_tink.py`

### Explicación de negocio

Aquí se convierte la capa heurística en una capacidad predictiva reutilizable. El objetivo es que, una vez entrenado el modelo, nuevas transacciones puedan clasificarse automáticamente sin depender solo de reglas fijas.

Desde negocio, esto acerca el pipeline a una app real: categorización automática, análisis de gasto, presupuestos, ahorro asistido y recomendaciones personalizadas.

### Explicación técnica

El script:

1. lee el dataset de entrenamiento de capa 3,
2. filtra clases con soporte mínimo,
3. prepara un pipeline de ML con texto y variables tabulares,
4. entrena una regresión logística multiclase balanceada,
5. evalúa sobre validation y test,
6. guarda métricas, predicciones, matriz de confusión y modelo serializado.

Features usadas actualmente:

1. texto normalizado de descripción con TF-IDF,
2. variables numéricas como importe y recurrencia,
3. variables categóricas como merchant, weekday, currency y tipo de cuenta,
4. flags booleanos convertidos a `0/1`.

Artefactos generados:

1. JSON de métricas,
2. CSV de predicciones,
3. CSV de matriz de confusión,
4. modelo serializado en `pickle`.

### Entrada principal

- `data/clean/transacciones_tink_model_input.csv`

### Salida principal

- `data/clean/metricas_clasificador_categorias_tink.json`
- `data/clean/predicciones_clasificador_categorias_tink.csv`
- `data/clean/clasificador_categorias_tink_confusion.csv`
- `05_modelos/artifacts/clasificador_categorias_tink.pkl`

---

## Resumen de arquitectura

### Fuente base

`02_api/genera_payloads_tink_sinteticos.py`

### Ruta canónica recomendada

1. `03_limpieza/prepara_transacciones_tink.py`
2. `03_limpieza/enriquecer_transacciones_tink.py`
3. `03_limpieza/prepara_dataset_entrenamiento_tink.py`
4. `05_modelos/entrena_clasificador_categorias_tink.py`

---

## Qué aporta cada capa al negocio

1. Payloads sintéticos: permiten desarrollar sin bloqueo externo.
2. Capa transaccional canónica: define la verdad base realista.
3. Capa enriquecida: convierte texto bancario en señales útiles.
4. Capa de entrenamiento: separa etiquetas confiables de casos dudosos.
5. Modelo de categorías: automatiza la clasificación para nuevas transacciones.

---

## Limitaciones actuales

1. Las categorías iniciales nacen de reglas heurísticas, no de etiquetas humanas completas.
2. El sample validado hasta ahora es sintético y relativamente pequeño.
3. Algunas clases siguen cayendo en `uncategorized` y requieren ampliación de reglas o revisión manual.
4. Las métricas altas en muestras pequeñas pueden ser optimistas por repetición de merchants y textos.

---

## Orden recomendado de uso

Si el objetivo es construir el producto alrededor de Tink, el orden recomendado es:

1. generar payloads o recibir datos tipo Tink,
2. extraer la tabla transaccional canónica,
3. enriquecer merchants y categorías,
4. preparar el dataset entrenable,
5. entrenar el clasificador,
6. después, explotar esas salidas en reporting, analítica avanzada o servicios de inferencia.