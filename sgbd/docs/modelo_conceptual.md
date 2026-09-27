# Modelo de datos — AI Financial Life Coach (Grupo 05)

Este documento resume el modelo conceptual y físico implementado en este
repositorio, de forma autocontenida (sin necesidad de consultar el PDF
entregado para poder reproducir o auditar el esquema).

> Esta versión incorpora las correcciones solicitadas tras la revisión
> docente de la Asignatura 6 (nota: 8,55/10). El detalle completo de cada
> corrección, debilidad por debilidad, está en
> [`CORRECCIONES_APLICADAS.md`](./CORRECCIONES_APLICADAS.md).

## 1. Diagrama Entidad-Relación (Datasets A, B y C — PostgreSQL)

![Diagrama E/R](./diagrama_er_corregido.png)

**Corrección de cardinalidad (debilidad docente n.º 1):** en notación
mín-máx, la etiqueta `(min,max)` escrita junto a una entidad describe la
participación de *esa* entidad en la relación. El diagrama anterior tenía
las etiquetas intercambiadas en las 7 relaciones 1:N del modelo — `(1,1)`
junto a la entidad propietaria y `(0,N)` junto a la dependiente, el sentido
contrario al que exige la propia prosa del trabajo. El diagrama regenerado
corrige las 7 relaciones (Reporta, Pertenece_a, Posee, Define, Recibe,
Genera, Registra): `(0,N)` junto a la entidad "uno"/propietaria, `(1,1)`
junto a la entidad dependiente.

## 2. Diagrama relacional físico (con claves ajenas y reglas de borrado)

![Diagrama relacional físico](./diagrama_relacional_fisico.png)

## 3. Entidades y claves

Todas las entidades del modelo son **entidades fuertes**: cada una tiene una
clave primaria propia e independiente (identificador autoincremental). Cinco
de ellas mantienen además una **dependencia existencial** hacia otra entidad
(participación total, cardinalidad (1,1)):

| Entidad | Clave primaria | Depende existencialmente de | Regla de borrado |
|---|---|---|---|
| `paises` | `pais_id` | — | — |
| `indicadores_macro` | `indicador_id` | — | — |
| `observaciones_macro` | `observacion_id` | `indicadores_macro`, `paises` | RESTRICT |
| `observaciones_sinteticas` | `registro_id` | — (entidad aislada, Dataset A.2) | — |
| `usuarios` | `usuario_id` | `paises` (opcional) | RESTRICT |
| `cuentas` | `cuenta_id` | `usuarios` | CASCADE |
| `transacciones` | `transaccion_id` | `cuentas` | RESTRICT |
| `objetivos_financieros` | `objetivo_id` | `usuarios` | CASCADE |
| `recomendaciones` | `recomendacion_id` | `usuarios` | CASCADE |
| `interacciones` | `interaccion_id` | `usuarios` | CASCADE |

`observaciones_macro` resuelve además la única relación N:M del modelo
(`paises` ↔ `indicadores_macro`), incorporando los atributos propios de esa
relación (`periodo`, `valor`, `fuente`, `fecha_extraccion`) y una restricción
`UNIQUE (indicador_id, pais_id, periodo)` que evita duplicados temporales.

`observaciones_sinteticas` (Dataset A.2) replica fielmente las 17 columnas
originales del generador (`02_api/genera_dataset_sintetico.py`) más el
identificador sustituto `registro_id` añadido como clave primaria del modelo
físico: **17 + 1 = 18 columnas en total** (corrección de la debilidad
docente n.º 6, que señalaba que esta distinción no quedaba clara).

## 4. Reglas de integridad principales

- **CASCADE** se aplica cuando la entidad dependiente pierde todo su sentido
  sin la entidad propietaria (p. ej., las cuentas de un usuario eliminado, o
  su historial de interacciones con el asistente).
- **RESTRICT** se aplica cuando existe un requisito de conservación de
  historial que prevalece sobre la simplicidad del borrado (p. ej., las
  transacciones no pueden desaparecer arrastradas por el borrado de una
  cuenta sin una decisión explícita, por su valor de auditoría). Esta regla
  se mantiene sin cambios como salvaguarda frente a un borrado físico
  accidental — nunca fue, ni pretende ser, el mecanismo de supresión a
  disposición del usuario final (ver más abajo).
- El dominio de `transacciones.categoria` consta de diez valores cerrados:
  los siete alineados con las categorías de gasto del dataset sintético
  (`vivienda`, `alimentacion`, `transporte`, `ocio`, `salud`, `educacion`,
  `otros`) más tres adicionales propios de un libro de movimientos real que
  no existen en el dataset sintético por no ser gasto (`ingresos`, `ahorro`,
  `transferencia`).
- `observaciones_sinteticas` reproduce mediante restricciones `CHECK` las
  verificaciones de coherencia interna que ya validamos en la
  asignatura de obtención de datos: `gasto_total` como suma de las siete
  categorías, `ahorro = salario - gasto_total`, `tasa_ahorro_pct =
  ahorro / salario * 100`, y consistencia de `perfil_ahorro` respecto a los
  umbrales documentados. Los dominios y rangos de columna (`perfil` en
  `{junior, medio, senior, freelance}`, `edad` entre 25 y 40 años, `salario`
  entre 1.134 € y 6.800 €, y los rangos del resto de columnas numéricas) se
  han verificado y alineado con la tabla de variables del trabajo de la
  Asignatura 5 "Obtención de Datos para el TFM", para evitar cualquier
  divergencia entre nuestros propios entregables.
- `interacciones.salida_modelo` es de tipo `JSONB`, con una restricción
  `CHECK (salida_modelo ? 'modelo')` que garantiza que todo documento
  declare qué modelo lo generó, sin imponer una estructura fija al resto del
  contenido (que varía según el modelo de ML, ver apartado 6).

El detalle completo (tipos de datos, todas las restricciones, índices y la
vista `v_saldo_usuario`) está en [`../sql/ddl/001_schema.sql`](../sql/ddl/001_schema.sql).

### 4.1 Derecho de supresión: anonimización (corrección de la debilidad docente n.º 2)

La combinación `CASCADE`/`RESTRICT` descrita arriba protege correctamente el
historial de auditoría, pero por sí sola no resolvía la historia de usuario
declarada en el trabajo ("el usuario debe poder ejercer su derecho de
supresión conservando el historial financiero"): con solo esas dos reglas,
el `DELETE` físico de un usuario con transacciones simplemente fallaba
siempre, sin ninguna vía alternativa para el usuario final.

Se añade una función `coach.anonimizar_usuario(usuario_id)` como mecanismo
de supresión efectivo:

- Sustituye `email` y `nombre` por valores anonimizados y registra
  `eliminado_en = now()` (nueva columna en `usuarios`).
- **No** toca `cuentas` ni `transacciones`: el historial financiero se
  conserva íntegro, tal y como exige el requisito de auditoría.
- Rechaza con una excepción explícita una segunda anonimización del mismo
  usuario.
- El `ON DELETE RESTRICT` de `cuentas → transacciones` sigue intacto y
  sigue bloqueando el borrado físico accidental; ambos mecanismos coexisten
  con responsabilidades distintas y ya no se contradicen.

### 4.2 Sincronización de `cuentas.saldo_actual` (corrección de la debilidad docente n.º 4)

`saldo_actual` sigue siendo una columna desnormalizada por rendimiento
(evita recalcular `SUM(importe)` en cada consulta de saldo), pero su
consistencia con `transacciones` ya no depende de la disciplina de la capa
de aplicación: un trigger `trg_actualizar_saldo_cuenta`, ejecutado `AFTER
INSERT OR UPDATE OR DELETE` sobre `transacciones`, recalcula
`saldo_actual` mediante `SUM(importe)` completo sobre la cuenta afectada
(y, si una transacción cambia de cuenta, sobre ambas cuentas implicadas).
El recómputo completo (no incremental) se elige deliberadamente por
simplicidad y robustez a la escala del MVP.

## 5. Dataset C — interacciones con el asistente (JSONB sobre PostgreSQL)

El Dataset C se modela conceptualmente mediante un esquema de documento (JSON
anotado), no mediante un desglose exhaustivo en E/R, por la naturaleza
variable de su estructura según el modelo de machine learning que genera
cada interacción: forzar un E/R clásico exigiría una entidad con decenas de
atributos mayoritariamente nulos, o una jerarquía de subentidades a rediseñar
con cada nuevo modelo (apartado 3.1 del documento entregado).

**Persistimos `interacciones` en el mismo motor PostgreSQL** que el resto
del modelo, usando una columna `JSONB` para la parte de estructura variable
(`salida_modelo`) y columnas normales para los campos comunes a toda
interacción (`usuario_id`, `tipo_interaccion`, `timestamp`,
`entrada_usuario`, `respuesta_mostrada`). Los tres documentos de ejemplo (uno
por cada modelo de ML del TFM: regresión lineal, regresión logística, serie
temporal) están cargados en [`../sql/seed/002_seed.sql`](../sql/seed/002_seed.sql)
y son consultables con `Consulta 7` en
[`../sql/queries/consultas_representativas.sql`](../sql/queries/consultas_representativas.sql).

## 6. Justificación tecnológica (resumen)

| Dataset | Tecnología | Motivo principal |
|---|---|---|
| A.1, A.2, B, C | PostgreSQL 16 (columna `JSONB` para C) | Tipado numérico estricto, integridad referencial declarativa siempre activa, ACID completo, y una única tecnología que reduce la complejidad operativa del MVP frente a mantener dos motores de persistencia distintos |

El razonamiento comparativo completo (PostgreSQL vs. MySQL/MariaDB vs.
SQLite para A/B; MongoDB vs. PostgreSQL+JSONB para C, con la decisión final
revisada a favor de JSONB) está desarrollado en el documento PDF entregado,
apartados 4.2 y 4.3. La revisión docente señaló que ese texto matiza
insuficientemente la comparación (agrupa MySQL/MariaDB sin versiones, y
compara conjuntamente los Datasets A y B pese a tener patrones de acceso muy
distintos, uno analítico y otro OLTP) — corrección pendiente de aplicar
directamente en ese documento (ver
[`CORRECCIONES_APLICADAS.md`](./CORRECCIONES_APLICADAS.md), debilidad n.º 5),
fuera del alcance de esta carpeta `sgbd/`.

## 7. Evolución futura y arquitectura de producción

La arquitectura de producción prevista (Google Cloud Platform, integración
regulada con Tink como agregador PSD2, uso de Gemini para clasificación NLP
de transacciones) está documentada como referencia, sin implementar, en
[`alineacion_jira_asignatura3.md`](./alineacion_jira_asignatura3.md) y
[`tink_mapping.md`](./tink_mapping.md). Ninguna de las dos introduce una
dependencia externa en este MVP, que permanece ejecutable íntegramente en
local con `docker compose up -d`.

## 8. Fuentes

- Grupo 05 (C. Solis Meza, J. Cáceres Mondragón, M. Merola, M. Á. Lozano
  Torres). Fuentes y Obtención de Datos — Asignatura 5, Máster en Big Data &
  Business Intelligence, Next Educación. Trabajo académico del grupo.
  Repositorio: https://github.com/mmerola99/TFM---Grupo_05
- Grupo 05 (C. Solis Meza, J. Cáceres Mondragón, M. Merola, M. Á. Lozano
  Torres, C. A. Cuaya Xinto). Backlog ágil y metodología Scrum — Asignatura
  3, Máster en Big Data & Business Intelligence, Next Educación. Trabajo
  académico del grupo.
- Nuestro repositorio técnico:
  https://github.com/mmerola99/TFM---Grupo_05/tree/main/sgbd
