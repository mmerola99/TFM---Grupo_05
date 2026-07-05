# Open Banking (Tink) — mapeo de referencia para evolución futura

**Estado: documentación de referencia, no implementada.** Este fichero no
forma parte del MVP entregado ni introduce ninguna dependencia del repositorio
hacia servicios externos. Se documenta aquí para dejar constancia de cómo se
integraría en un sprint futuro del TFM una fuente de datos reales vía
Open Banking (PSD2), y para que el mapeo de campos no se improvise en el
momento de hacerlo.

## 1. Origen

Dos salidas de la API de Tink (sandbox "demo bank", usuario de España) sobre
sus endpoints `list accounts` y `list transactions`, compartidas por el grupo
el 2/07/2026. Son datos sintéticos del propio sandbox de Tink, no datos de
usuarios reales.

## 2. Lo que confirma una decisión de diseño ya tomada

Tink representa los importes como `unscaledValue` + `scale` (equivalente a un
`BigDecimal`: valor real = `unscaledValue × 10^(-scale)`), nunca como número
de coma flotante. Es la misma motivación que llevó a elegir `NUMERIC` en
PostgreSQL para todas las columnas monetarias del esquema actual
(`sql/ddl/001_schema.sql`): en un dominio financiero, el redondeo de coma
flotante no es aceptable. No requiere ningún cambio; se documenta como
validación externa de una decisión ya justificada en el trabajo.

## 3. Lo que NO está disponible en estas dos salidas

El endpoint `list transactions` consultado **no incluye categoría de
transacción**: cada movimiento trae únicamente `types.type = "DEFAULT"` y una
descripción genérica (`"Payment"`). La categorización real de Tink se sirve
mediante un endpoint de enrichment/categorización distinto, que no se ha
consultado todavía. Por tanto, **no existe todavía una base real para mapear
las diez categorías actuales de `transacciones.categoria`** contra un dominio
de categorías de Tink — ese mapeo queda pendiente de una futura consulta a
dicho endpoint.

## 4. Mapeo de campos (accounts)

| Campo Tink | Campo propio equivalente | Nota |
|---|---|---|
| `accounts[].id` | `cuentas.cuenta_id` (lógico) | Tink usa un hash como id; nuestro esquema usa `SERIAL`. Requeriría una columna adicional `id_externo` si se integrara. |
| `accounts[].type` (`CHECKING`, `SAVINGS`) | `cuentas.tipo_cuenta` (`corriente`, `ahorro`) | Alineado. Tink define también `CREDIT_CARD` y otros tipos no vistos en esta muestra, que ya tienen equivalente en nuestro dominio (`tarjeta_credito`). |
| `balances.booked.amount` | `cuentas.saldo_actual` | Requiere convertir `unscaledValue`/`scale` a `NUMERIC` en la capa de aplicación. |
| `identifiers.iban.iban` | *(no modelado)* | Campo no presente en el esquema actual; a añadir como columna `cuentas.iban` si se integra una fuente real. |
| `financialInstitutionId` | *(no modelado)* | En un escenario multi-banco real, ameritaría una entidad `ENTIDAD_FINANCIERA` propia en lugar de un identificador suelto. |
| `customerSegment`, `dates.lastRefreshed` | *(no modelado)* | Metadatos operativos de la integración, no de negocio; no se contempla incorporarlos al modelo conceptual. |

## 5. Mapeo de campos (transactions)

| Campo Tink | Campo propio equivalente | Nota |
|---|---|---|
| `transactions[].accountId` | `transacciones.cuenta_id` | Directo. |
| `amount.value` (`unscaledValue`/`scale`) + `amount.currencyCode` | `transacciones.importe` | Conversión a `NUMERIC`; `cuentas.moneda` ya existe para la divisa. |
| `dates.booked` | `transacciones.fecha` | Directo. |
| `descriptions.display` | `transacciones.descripcion` | Directo. |
| `types.type` / categoría (endpoint de enrichment, pendiente) | `transacciones.categoria` | **Pendiente**: sin datos de categorización real todavía, no se puede proponer un mapeo fiable contra el dominio cerrado de diez valores. |
| `status` (`BOOKED`, ...) | *(no modelado)* | Nuestro esquema no distingue transacciones pendientes/reservadas de confirmadas; a valorar si se integrara una fuente en tiempo real. |

## 6. Siguiente paso si se decide avanzar

1. Consultar el endpoint de categorización/enrichment de Tink sobre el mismo
   usuario de prueba y documentar aquí las categorías reales devueltas.
2. Definir el mapeo categoría-Tink → dominio propio (o ampliar el dominio
   propio si aparecen categorías sin equivalente razonable).
3. Evaluar si conviene ampliar el dataset sintético con una distribución más
   realista de descripciones de transacción, sin depender de una llamada en
   vivo a la API en el proceso de generación (para no comprometer la
   reproducibilidad local del repositorio).

Ninguno de estos pasos se ha ejecutado en este MVP.
