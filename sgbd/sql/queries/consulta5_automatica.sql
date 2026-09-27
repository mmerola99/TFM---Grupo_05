-- =====================================================================
-- Consulta 5 — Verificación de integridad referencial y derecho de
-- supresión (versión automática, sin marcadores manuales)
--
-- Corrección aplicada (debilidad docente nº 2): además de comprobar que
-- ON DELETE RESTRICT sigue bloqueando correctamente el borrado físico
-- (salvaguarda frente a un DELETE accidental), esta versión demuestra
-- también que coach.anonimizar_usuario() SÍ resuelve la historia de
-- usuario original: el usuario puede ejercer su derecho de supresión y
-- el historial financiero (cuentas y transacciones) permanece intacto.
-- =====================================================================

SET search_path TO coach, public;

DO $$
DECLARE
    v_usuario_id  INTEGER;
    v_cuenta_id   INTEGER;
    v_email       VARCHAR(150);
    v_nombre      VARCHAR(150);
    v_num_cuentas INTEGER;
BEGIN
    -- 1. Crear usuario de prueba
    INSERT INTO coach.usuarios (email, nombre, plan)
    VALUES ('test.borrado@example.com', 'Test Borrado', 'free')
    RETURNING usuario_id INTO v_usuario_id;

    -- 2. Asociar una cuenta y una transacción
    INSERT INTO coach.cuentas (usuario_id, tipo_cuenta)
    VALUES (v_usuario_id, 'corriente')
    RETURNING cuenta_id INTO v_cuenta_id;

    -- Importe en positivo: con el trigger de sincronización de saldo ya
    -- activo, una cuenta 'corriente' recién creada (saldo 0) no admite
    -- una transacción negativa que la dejaría en descubierto
    -- (chk_saldo_no_negativo), así que probamos con un ingreso.
    INSERT INTO coach.transacciones (cuenta_id, importe, categoria)
    VALUES (v_cuenta_id, 50.00, 'ingresos');

    RAISE NOTICE 'Usuario de prueba creado con usuario_id = %, cuenta_id = %', v_usuario_id, v_cuenta_id;

    -- =================================================================
    -- PARTE A: el borrado físico sigue bloqueado (salvaguarda deliberada,
    -- no es el mecanismo de supresión para el usuario final)
    -- =================================================================
    BEGIN
        DELETE FROM coach.usuarios WHERE usuario_id = v_usuario_id;
        RAISE EXCEPTION 'ERROR DE VALIDACION: se ha podido borrar el usuario, la regla ON DELETE RESTRICT no está activa';
    EXCEPTION
        WHEN foreign_key_violation THEN
            RAISE NOTICE 'OK (parte A): el borrado FISICO se ha bloqueado correctamente por ON DELETE RESTRICT (cuentas -> transacciones).';
    END;

    -- =================================================================
    -- PARTE B: la vía correcta para el derecho de supresión es la
    -- anonimización — debe tener éxito y conservar cuentas/transacciones
    -- =================================================================
    PERFORM coach.anonimizar_usuario(v_usuario_id);

    SELECT email, nombre INTO v_email, v_nombre
    FROM coach.usuarios WHERE usuario_id = v_usuario_id;

    SELECT COUNT(*) INTO v_num_cuentas
    FROM coach.cuentas WHERE usuario_id = v_usuario_id;

    IF v_email LIKE 'eliminado+%' AND v_nombre = 'Usuario eliminado' AND v_num_cuentas = 1 THEN
        RAISE NOTICE 'OK (parte B): usuario anonimizado correctamente (email=%, nombre=%) conservando % cuenta(s) y su historial de transacciones.',
            v_email, v_nombre, v_num_cuentas;
    ELSE
        RAISE EXCEPTION 'ERROR DE VALIDACION: la anonimizacion no se aplico como se esperaba (email=%, nombre=%, cuentas=%)',
            v_email, v_nombre, v_num_cuentas;
    END IF;

    -- =================================================================
    -- PARTE C: anonimizar dos veces debe fallar de forma controlada
    -- =================================================================
    BEGIN
        PERFORM coach.anonimizar_usuario(v_usuario_id);
        RAISE EXCEPTION 'ERROR DE VALIDACION: se ha podido anonimizar dos veces al mismo usuario';
    EXCEPTION
        WHEN OTHERS THEN
            IF SQLERRM LIKE '%ya había sido anonimizado%' THEN
                RAISE NOTICE 'OK (parte C): una segunda anonimizacion del mismo usuario se rechaza correctamente.';
            ELSE
                RAISE;
            END IF;
    END;

END $$;