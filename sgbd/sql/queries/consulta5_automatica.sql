-- =====================================================================
-- Consulta 5 — Verificación de integridad referencial (versión automática)
-- Hace exactamente lo mismo que consultas_representativas.sql (Consulta 5)
-- pero sin necesidad de copiar manualmente los ids devueltos por RETURNING.
-- Pensada para ejecutarse de un solo golpe desde psql o VS Code.
-- =====================================================================

SET search_path TO coach, public;

DO $$
DECLARE
    v_usuario_id  INTEGER;
    v_cuenta_id   INTEGER;
BEGIN
    -- 1. Crear usuario de prueba
    INSERT INTO coach.usuarios (email, nombre, plan)
    VALUES ('test.borrado@example.com', 'Test Borrado', 'free')
    RETURNING usuario_id INTO v_usuario_id;

    -- 2. Asociar una cuenta y una transacción
    INSERT INTO coach.cuentas (usuario_id, tipo_cuenta)
    VALUES (v_usuario_id, 'corriente')
    RETURNING cuenta_id INTO v_cuenta_id;

    INSERT INTO coach.transacciones (cuenta_id, importe, categoria)
    VALUES (v_cuenta_id, -10.00, 'ocio');

    RAISE NOTICE 'Usuario de prueba creado con usuario_id = %, cuenta_id = %', v_usuario_id, v_cuenta_id;

    -- 3. Intentar borrar el usuario -> debe fallar por ON DELETE RESTRICT
    --    en cuentas -> transacciones
    DELETE FROM coach.usuarios WHERE usuario_id = v_usuario_id;

    -- Si llegamos aquí, la restricción NO se activó (comportamiento inesperado)
    RAISE EXCEPTION 'ERROR DE VALIDACION: se ha podido borrar el usuario, la regla ON DELETE RESTRICT no está activa';

EXCEPTION
    WHEN foreign_key_violation THEN
        RAISE NOTICE 'OK: el borrado se ha bloqueado correctamente por ON DELETE RESTRICT (cuentas -> transacciones)';
END $$;
