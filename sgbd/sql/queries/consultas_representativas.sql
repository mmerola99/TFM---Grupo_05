-- =====================================================================
-- AI Financial Life Coach — Grupo 05
-- Las seis consultas representativas documentadas en el apartado 6
-- del trabajo. Ejecutar tras 001_schema.sql y 002_seed.sql.
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
-- Saldo total del usuario (via vista)
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
-- Consulta 4 — Historial de recomendaciones y tasa de aceptación
-- =====================================================================
-- Historial de un usuario concreto
SELECT fecha_emision, tipo, estado,
       confianza_modelo, contenido
FROM coach.recomendaciones
WHERE usuario_id = 1
ORDER BY fecha_emision DESC;

-- Tasa de aceptación global (proxy de conversión)
SELECT ROUND(
    COUNT(*) FILTER (WHERE estado = 'aceptada')::NUMERIC
    / NULLIF(COUNT(*), 0) * 100, 1
) AS tasa_aceptacion_pct
FROM coach.recomendaciones;


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
-- consulta5_automatica.sql, que hace exactamente lo mismo sin marcadores.
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
INSERT INTO coach.transacciones (cuenta_id, importe, categoria)
VALUES (<cuenta_id>, -10.00, 'ocio');

-- 3. Intentar borrar el usuario (debe fallar por ON DELETE RESTRICT)
DELETE FROM coach.usuarios WHERE usuario_id = <usuario_id>;

-- RESULTADO ESPERADO:
-- ERROR: update or delete on table "cuentas" violates foreign key
-- constraint "transacciones_cuenta_id_fkey" on table "transacciones"
-- La operación se bloquea por ON DELETE RESTRICT en cuentas->transacciones,
-- evitando la eliminación silenciosa del historial de auditoría.


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
