# Correcciones aplicadas tras la revisión docente

Este documento resume, debilidad por debilidad, las seis observaciones
recibidas en la evaluación de la Asignatura 6 (nota: 8,55/10) y el estado de
cada corrección en esta versión del SGBD. El texto íntegro de la revisión
docente está reproducido aquí solo en la columna "Observación", para que
este fichero sea autocontenido.

| # | Observación docente | Estado | Dónde se resuelve |
|---|---|---|---|
| 1 | "La notación de cardinalidades del diagrama E/R está invertida... (1,1) aparece junto a USUARIO y (0,N) junto a CUENTA... el mismo problema se repite en otras relaciones." | **Corregida** | [`diagrama_er_corregido.png`](./diagrama_er_corregido.png) |
| 2 | "La política de borrado no resuelve la historia de usuario declarada... la combinación CASCADE–RESTRICT provoca que el borrado del usuario sea rechazado mientras existan transacciones." | **Corregida** | `coach.anonimizar_usuario()` en [`../sql/ddl/001_schema.sql`](../sql/ddl/001_schema.sql); demostrada en [`../sql/queries/consulta5_automatica.sql`](../sql/queries/consulta5_automatica.sql) |
| 3 | "La tasa de aceptación de recomendaciones no mide la conversión a premium... no existe una relación documental suficiente entre aceptar una recomendación y contratar el plan premium." | **Corregida** | Consulta 4 (4.a / 4.b) en [`../sql/queries/consultas_representativas.sql`](../sql/queries/consultas_representativas.sql) |
| 4 | "La desnormalización de saldo_actual queda incompletamente resuelta... no se documenta cómo se garantiza su sincronización con las transacciones." | **Corregida** | `trg_actualizar_saldo_cuenta` en [`../sql/ddl/001_schema.sql`](../sql/ddl/001_schema.sql) |
| 5 | "La comparación tecnológica contiene afirmaciones poco matizadas... se agrupan MySQL y MariaDB sin concretar versiones... los Datasets A y B se comparan conjuntamente pese a tener patrones muy diferentes." | **Pendiente de aplicar en el documento PDF principal** | Apartados 4.2–4.3 del documento entregado (fuera del alcance de esta carpeta `sgbd/`, que no contiene ese texto) |
| 6 | "Hay pequeñas inconsistencias de detalle. El Dataset A.2 se describe como un conjunto de 18 columnas, pero la enumeración de campos de origen no deja clara la diferencia entre las columnas originales y el identificador sustituto." | **Corregida** | Comentario de `observaciones_sinteticas` en [`../sql/ddl/001_schema.sql`](../sql/ddl/001_schema.sql) y nota del diagrama E/R |

## Detalle de cada corrección

### 1. Cardinalidad invertida en el diagrama E/R

En notación mín-máx, la etiqueta `(min,max)` escrita junto a una entidad
describe la participación de **esa** entidad en la relación (cuántas veces
puede aparecer una instancia suya del lado de la relación). El diagrama
anterior escribía `(1,1)` junto a la entidad "uno"/propietaria (p. ej.
USUARIO, PAIS, CUENTA como propietaria de Registra) y `(0,N)` junto a la
entidad dependiente (CUENTA como dependiente de Posee, TRANSACCION,
OBSERVACION_MACRO, OBJETIVO_FINANCIERO, RECOMENDACION, INTERACCIONES) — el
sentido contrario al que exige la prosa del propio documento ("cada cuenta
pertenece a un usuario, un usuario puede tener varias cuentas"). Esta
inversión afectaba a las 7 relaciones 1:N del modelo (Reporta, Pertenece_a,
Posee, Define, Recibe, Genera, Registra), no solo a la señalada como
ejemplo.

Se ha regenerado el diagrama completo
([`diagrama_er_corregido.png`](./diagrama_er_corregido.png)) con las
etiquetas en el sentido correcto: `(0,N)` junto a la entidad propietaria,
`(1,1)` junto a la entidad dependiente, en las 7 relaciones. El propio
diagrama incluye ahora una nota explicando la corrección aplicada.

### 2. Contradicción CASCADE–RESTRICT y derecho de supresión

El esquema físico ya usaba correctamente `ON DELETE RESTRICT` entre
`cuentas` y `transacciones` como salvaguarda contra un borrado físico
accidental que arrastraría el historial financiero. El problema señalado es
distinto: esa misma regla, sin ningún mecanismo alternativo, dejaba al
usuario sin ninguna vía real para ejercer su derecho de supresión (RGPD)
— el `DELETE` simplemente fallaba siempre que existieran transacciones, que
es el caso normal.

La solución no es relajar `RESTRICT` (seguiría comprometiendo la integridad
del histórico de auditoría), sino añadir una vía de supresión que sí
resuelve la historia de usuario: **anonimización** en lugar de borrado
físico.

- Nueva columna `usuarios.eliminado_en TIMESTAMP`.
- Nueva función `coach.anonimizar_usuario(p_usuario_id)`: sustituye
  `email`/`nombre` por valores anonimizados, marca `eliminado_en = now()`, y
  rechaza con una excepción explícita una segunda anonimización del mismo
  usuario.
- `cuentas` y `transacciones` del usuario permanecen intactas — el
  historial financiero (y su valor de auditoría) se conserva íntegro, que
  es exactamente lo que la Consulta 5 pretendía demostrar.
- El `ON DELETE RESTRICT` original se mantiene sin cambios como salvaguarda
  frente a un borrado físico accidental o no autorizado; ya no es, ni
  pretendía ser, el mecanismo de supresión del usuario final.

Verificado extremo a extremo en
[`../sql/queries/consulta5_automatica.sql`](../sql/queries/consulta5_automatica.sql):
el borrado físico sigue bloqueado (parte A), la anonimización tiene éxito y
conserva cuentas/transacciones (parte B), y una segunda anonimización se
rechaza de forma controlada (parte C).

### 3. Aceptación de recomendaciones ≠ conversión a premium

La Consulta 4 original solo calculaba `recomendaciones aceptadas /
recomendaciones emitidas` y la presentaba como aproximación al objetivo de
negocio de conversión a premium (6 %), sin ninguna relación documental
entre ambos eventos — un usuario puede aceptar recomendaciones sin ser
premium, o ser premium sin haber aceptado ninguna.

Se han separado explícitamente dos métricas:

- **4.a** — tasa de aceptación de recomendaciones (la métrica original, sin
  cambios en su cálculo, pero ya no presentada como proxy de conversión).
- **4.b** (nueva) — de los usuarios que han aceptado al menos una
  recomendación, qué porcentaje tiene efectivamente el plan premium hoy.
  Es una aproximación explícita (no prueba causalidad ni orden temporal),
  pero permite contrastar ambas métricas en vez de confundirlas.

Los datos de prueba (`002_seed.sql`) se han ampliado con una recomendación
aceptada por un usuario en plan `free`, precisamente para que ambas
métricas *diverjan* en el ejemplo (verificado: 66,7 % de aceptación frente a
50,0 % de conversión aproximada) y la distinción quede demostrada, no solo
enunciada.

### 4. Sincronización de `cuentas.saldo_actual`

El esquema original mantenía `saldo_actual` como columna desnormalizada por
rendimiento, pero no documentaba ni implementaba ningún mecanismo que
garantizara su consistencia con la suma real de `transacciones` — la
columna dependía enteramente de que la aplicación la actualizara con
disciplina en cada inserción, lo cual el propio diagrama físico no reflejaba
en ninguna parte.

Se ha añadido un trigger a nivel de base de datos que hace imposible la
divergencia, independientemente de la capa de aplicación:

- `coach.fn_actualizar_saldo_cuenta()` recalcula `saldo_actual` mediante
  `SUM(importe)` sobre todas las transacciones de la cuenta afectada
  (recómputo completo, no incremental, deliberadamente simple y robusto
  para la escala del MVP).
- `trg_actualizar_saldo_cuenta`, disparado `AFTER INSERT OR UPDATE OR
  DELETE` sobre `transacciones`, invoca la función anterior — incluyendo el
  caso de una transacción reasignada de cuenta (`UPDATE` que cambia
  `cuenta_id`), que recalcula ambas cuentas implicadas.

Los datos de prueba (`002_seed.sql`) se han adaptado para reflejar un libro
de movimientos real (saldo inicial + movimientos) en vez de un valor final
`saldo_actual` fijado a mano, y se ha verificado que el trigger recomputa
exactamente los mismos saldos finales que documentaba la versión anterior.

### 5. Comparación tecnológica poco matizada

Esta observación se refiere al texto de los apartados 4.2–4.3 del documento
PDF principal del trabajo (agrupación de MySQL/MariaDB sin versiones,
comparación conjunta de los Datasets A y B pese a patrones de acceso
distintos), no a ningún fichero de esta carpeta `sgbd/`. No se puede aplicar
ni verificar esta corrección desde el repositorio técnico: queda pendiente
de revisión directa en el documento entregado.

### 6. Inconsistencia de detalle: 18 columnas del Dataset A.2

El comentario del esquema afirmaba "18 columnas" sin distinguir las 17
columnas originales del generador sintético (`02_api/genera_dataset_sintetico.py`)
del identificador sustituto `registro_id` añadido en el modelo físico como
clave primaria. Se ha corregido el comentario de `observaciones_sinteticas`
en `001_schema.sql` para decir explícitamente "17 variables originales + 1
identificador sustituto (`registro_id`) = 18 columnas en total", y la misma
aclaración se repite en la nota del diagrama E/R regenerado.
