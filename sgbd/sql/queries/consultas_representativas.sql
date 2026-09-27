-- =====================================================================
-- AI Financial Life Coach — Grupo 05
-- Las siete consultas representativas documentadas en el apartado 6
-- del trabajo. Ejecutar tras 001_schema.sql y 002_seed.sql.
--
-- Esta versión incorpora la corrección de la Consulta 4 solicitada tras
-- la revisión docente (ver docs/CORRECCIONES_APLICADAS.md, debilidad 3):
-- la tasa de aceptación de recomendaciones y la conversión a premium se
-- presentan ahora como dos métricas explícitamente distintas, con una
-- consulta adicional que aproxima la relación real entre ambas.
-- =====================================================================

SET search_path TO coach, public;

-- =====================================================================
-- Consulta 1 — Serie temporal de un indicador macroeconómico por país
-- =====================================================================
SELECT p.nombre AS pais,
       i.nombre AS indicador,
       o.periodo,
       o.valor,
       i.unidad
FROM coach.observaciones_macro o
JOIN coach.indicadores_macro i ON i.indicador_id = o.indicador_id
JOIN coach.paises p ON p.pais_id = o.pais_id
WHERE i.codigo_eurostat = 'prc_hicp_manr'
  AND p.codigo_iso2 = 'ES'
ORDER BY o.periodo;


-- =====================================================================
-- Consulta 2 — Saldo agregado y movimientos recientes de un usuario
-- =====================================================================
-- Saldo total del usuario (via vista; saldo_actual de cada cuenta ahora
-- mantenido automáticamente por trg_actualizar_saldo_cuenta)
SELECT usuario_id, nombre, saldo_total
FROM coach.v_saldo_usuario
WHERE usuario_id = 1;

-- Últimos 10 movimientos
SELECT t.fecha, t.importe, t.categoria,
       t.descripcion, c.tipo_cuenta
FROM coach.transacciones t
JOIN coach.cuentas c ON c.cuenta_id = t.cuenta_id
WHERE c.usuario_id = 1
ORDER BY t.fecha DESC
LIMIT 10;


-- =====================================================================
-- Consulta 3 — Perfil de gasto por categoría y mes
-- =====================================================================
SELECT c.usuario_id,
       t.categoria,
       DATE_TRUNC('month', t.fecha) AS mes,
       SUM(t.importe) AS total_categoria,
       COUNT(*) AS num_transacciones
FROM coach.transacciones t
JOIN coach.cuentas c ON c.cuenta_id = t.cuenta_id
WHERE c.usuario_id = 1
GROUP BY c.usuario_id, t.categoria, DATE_TRUNC('month', t.fecha)
ORDER BY mes, total_categoria;


-- =====================================================================
-- Consulta 4 — Historial de recomendaciones, tasa de aceptación y
-- aproximación a la conversión a premium (CORREGIDA)
--
-- NOTA METODOLÓGICA (corrección de la debilidad docente): la tasa de
-- aceptación mide cuántas recomendaciones emitidas fueron aceptadas por
-- el usuario. NO mide cuántos usuarios contratan el plan premium: un
-- usuario puede aceptar recomendaciones sin ser premium (ver el
-- usuario 2 en el seed, plan 'free', con una recomendación aceptada), o
-- ser premium sin haber aceptado ninguna. Se presentan ambas métricas
-- por separado y se añade una tercera consulta que aproxima la relación
-- real entre aceptar recomendaciones y convertir a premium, hasta que
-- se documente una relación causal explícita entre ambos eventos.
-- =====================================================================

-- Historial de un usuario concreto
SELECT fecha_emision, tipo, estado,
       confianza_modelo, contenido
FROM coach.recomendaciones
WHERE usuario_id = 1
ORDER BY fecha_emision DESC;

-- 4.a — Tasa de aceptación de recomendaciones (NO es la tasa de
-- conversión a premium; ver nota metodológica más arriba)
SELECT ROUND(
    COUNT(*) FILTER (WHERE estado = 'aceptada')::NUMERIC
    / NULLIF(COUNT(*), 0) * 100, 1
) AS tasa_aceptacion_pct
FROM coach.recomendaciones;

-- 4.b — Aproximación a la conversión real a premium: de los usuarios
-- que han aceptado al menos una recomendación, qué porcentaje tiene
-- efectivamente el plan premium hoy. Es una aproximación (no prueba
-- causalidad ni orden temporal aceptación->conversión), pero permite
-- contrastar la tasa de aceptación con un indicador ligado al negocio.
SELECT
    COUNT(*) FILTER (WHERE u.plan = 'premium') AS usuarios_premium,
    COUNT(*) AS usuarios_con_alguna_aceptacion,
    ROUND(
        COUNT(*) FILTER (WHERE u.plan = 'premium')::NUMERIC
        / NULLIF(COUNT(*), 0) * 100, 1
    ) AS conversion_aproximada_pct
FROM coach.usuarios u
WHERE EXISTS (
    SELECT 1 FROM coach.recomendaciones r
    WHERE r.usuario_id = u.usuario_id AND r.estado = 'aceptada'
);


-- =====================================================================
-- Consulta 5 — Verificación de integridad referencial ante borrado
-- de usuario (secuencia de validación completa)
--
-- NOTA: los marcadores <usuario_id> y <cuenta_id> de más abajo son
-- intencionados (reproducen la secuencia manual descrita en el
-- apartado 6 del trabajo, paso a paso con RETURNING). Si se ejecuta
-- este fichero completo de una sola vez, estas líneas fallarán con un
-- error de sintaxis "at or near <" — es el comportamiento esperado.
-- Para una comprobación automática de un solo golpe, usar en su lugar
-- consulta5_automatica.sql, que hace exactamente lo mismo sin marcadores
-- y además demuestra la corrección de la debilidad docente nº 2
-- (derecho de supresión mediante anonimización).
-- =====================================================================
-- 1. Crear usuario de prueba
INSERT INTO coach.usuarios (email, nombre, plan)
VALUES ('test.borrado@example.com', 'Test Borrado', 'free')
RETURNING usuario_id;                              -- anotar el usuario_id devuelto, p.ej. 3

-- 2. Asociar una cuenta y una transacción
--    (sustituir <usuario_id> por el valor devuelto en el paso anterior)
INSERT INTO coach.cuentas (usuario_id, tipo_cuenta)
VALUES (<usuario_id>, 'corriente')
RETURNING cuenta_id;                                -- anotar el cuenta_id devuelto, p.ej. 4

--    (sustituir <cuenta_id> por el valor devuelto en el paso anterior)
--    Importe en positivo: con el trigger de saldo activo, una cuenta
--    'corriente' recién creada (saldo 0) no admite un cargo que la
--    deje en descubierto (chk_saldo_no_negativo).
INSERT INTO coach.transacciones (cuenta_id, importe, categoria)
VALUES (<cuenta_id>, 50.00, 'ingresos');

-- 3. Intentar borrar físicamente el usuario (debe fallar por ON DELETE
--    RESTRICT: esto sigue siendo así deliberadamente, es la salvaguarda
--    frente a un borrado físico accidental, NO el mecanismo de
--    supresión a disposición del usuario final — ver paso 4)
DELETE FROM coach.usuarios WHERE usuario_id = <usuario_id>;

-- RESULTADO ESPERADO:
-- ERROR: update or delete on table "cuentas" violates foreign key
-- constraint "transacciones_cuenta_id_fkey" on table "transacciones"
-- La operación se bloquea por ON DELETE RESTRICT en cuentas->transacciones,
-- evitando la eliminación silenciosa del historial de auditoría.

-- 4. La vía correcta para que el usuario ejerza su derecho de
--    supresión es la anonimización, que SÍ conserva el historial
--    financiero íntegro (corrección de la debilidad docente nº 2):
SELECT coach.anonimizar_usuario(<usuario_id>);

-- Comprobación: los datos personales quedan anonimizados, pero cuentas
-- y transacciones del usuario siguen intactas.
SELECT usuario_id, email, nombre, eliminado_en
FROM coach.usuarios
WHERE usuario_id = <usuario_id>;


-- =====================================================================
-- Consulta 6 — Distribución del perfil de ahorro por tipo de empleo
-- (dataset sintético, Dataset A.2)
-- =====================================================================
SELECT perfil,
       perfil_ahorro,
       COUNT(*) AS observaciones,
       ROUND(AVG(tasa_ahorro_pct), 2) AS tasa_media,
       ROUND(AVG(salario), 0) AS salario_medio
FROM coach.observaciones_sinteticas
GROUP BY perfil, perfil_ahorro
ORDER BY perfil, perfil_ahorro;


-- =====================================================================
-- Consulta 7 — Historial de interacciones de un usuario con el modelo
-- que las generó (Dataset C, columna JSONB)
-- =====================================================================
SELECT usuario_id,
       tipo_interaccion,
       timestamp,
       salida_modelo ->> 'modelo' AS modelo_origen,
       salida_modelo
FROM coach.interacciones
WHERE usuario_id = 1
ORDER BY timestamp DESC;

-- Variante analítica: cuántas interacciones ha generado cada modelo
-- (agregación directamente sobre el contenido del JSONB)
SELECT salida_modelo ->> 'modelo' AS modelo_origen,
       COUNT(*) AS num_interacciones
FROM coach.interacciones
GROUP BY salida_modelo ->> 'modelo'
ORDER BY num_interacciones DESC;
--
