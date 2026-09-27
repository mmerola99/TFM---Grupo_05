-- =====================================================================
-- AI Financial Life Coach — Grupo 05
-- DDL del esquema físico (Datasets A.1, A.2 y B) sobre PostgreSQL 16
-- Corresponde al modelo E/R documentado en el apartado 3.2 del trabajo
-- (version corregida: todas las entidades son entidades fuertes con
-- clave primaria propia; la dependencia existencial se materializa
-- mediante FK obligatoria + regla de borrado, no mediante clave parcial).
--
-- ESTA VERSIÓN incorpora las correcciones solicitadas tras la evaluación
-- docente de la entrega anterior (ver docs/CORRECCIONES_APLICADAS.md):
--   1. Cardinalidades del diagrama E/R corregidas (docs/diagrama_er_corregido.png)
--   2. Mecanismo de anonimización para el derecho de supresión del usuario
--      (función coach.anonimizar_usuario), sin alterar las reglas CASCADE/
--      RESTRICT que protegen la integridad del historial financiero
--   3. Consulta 4 separa explícitamente aceptación de recomendaciones y
--      conversión a premium (ver sql/queries/consultas_representativas.sql)
--   4. Trigger que mantiene cuentas.saldo_actual sincronizado con
--      transacciones, documentando el mecanismo antes ausente
--   6. Comentario de observaciones_sinteticas corregido (17+1=18 columnas,
--      no 18 en el script de origen)
-- =====================================================================

CREATE SCHEMA IF NOT EXISTS coach;
SET search_path TO coach, public;

-- ---------------------------------------------------------------------
-- PAIS (entidad fuerte)
-- ---------------------------------------------------------------------
CREATE TABLE coach.paises (
    pais_id      SERIAL PRIMARY KEY,
    codigo_iso2  CHAR(2) NOT NULL UNIQUE,
    nombre       VARCHAR(100) NOT NULL
);

-- ---------------------------------------------------------------------
-- INDICADOR_MACRO (entidad fuerte)
-- ---------------------------------------------------------------------
CREATE TABLE coach.indicadores_macro (
    indicador_id     SERIAL PRIMARY KEY,
    codigo_eurostat  VARCHAR(30) NOT NULL UNIQUE,
    nombre           VARCHAR(150) NOT NULL,
    unidad           VARCHAR(50) NOT NULL,
    descripcion      TEXT
);

-- ---------------------------------------------------------------------
-- OBSERVACION_MACRO (entidad fuerte; resuelve la relación N:M
-- PAIS <-> INDICADOR_MACRO, apartado 3.4 punto 4)
-- ---------------------------------------------------------------------
CREATE TABLE coach.observaciones_macro (
    observacion_id   SERIAL PRIMARY KEY,
    indicador_id     INTEGER NOT NULL REFERENCES coach.indicadores_macro(indicador_id)
                        ON DELETE RESTRICT,
    pais_id          INTEGER NOT NULL REFERENCES coach.paises(pais_id)
                        ON DELETE RESTRICT,
    periodo          VARCHAR(10) NOT NULL,          -- p.ej. '2024', '2024-Q3'
    valor            NUMERIC(12,4) NOT NULL,
    fuente           VARCHAR(100) NOT NULL,
    fecha_extraccion TIMESTAMP NOT NULL DEFAULT now(),
    CONSTRAINT uq_observacion_macro UNIQUE (indicador_id, pais_id, periodo)
);

-- ---------------------------------------------------------------------
-- OBSERVACION_SINTETICA (entidad fuerte y aislada — Dataset A.2)
-- Replica fielmente las 17 columnas originales de
-- 02_api/genera_dataset_sintetico.py (user_id, fecha, edad, perfil,
-- salario, vivienda, alimentacion, transporte, ocio, salud, educacion,
-- otros, gasto_total, ahorro, tasa_ahorro_pct, perfil_ahorro,
-- ipc_mensual), más el identificador sustituto registro_id añadido en
-- este modelo físico (17 + 1 = 18 columnas en total). Se deja explícito
-- para no reproducir la ambigüedad "18 vs 19" señalada en la revisión
-- docente de la Asignatura 6.
-- Dominios y rangos verificados contra el trabajo de la Asignatura 5
-- "Obtención de Datos para el TFM" (tabla de variables, apartado 4.4).
-- Sin FK hacia USUARIO: user_id es un identificador de simulación, no
-- una referencia a un usuario real (apartado 3.2 / 3.3).
-- ---------------------------------------------------------------------
CREATE TABLE coach.observaciones_sinteticas (
    registro_id      SERIAL PRIMARY KEY,
    user_id          INTEGER NOT NULL,
    fecha            DATE NOT NULL,
    edad             SMALLINT NOT NULL CHECK (edad BETWEEN 25 AND 40),
    perfil           VARCHAR(20) NOT NULL CHECK (perfil IN
                        ('junior', 'medio', 'senior', 'freelance')),
    salario          NUMERIC(10,2) NOT NULL CHECK (salario BETWEEN 1134.00 AND 6800.00), -- SMI 2025 — salario máximo observado
    vivienda         NUMERIC(10,2) NOT NULL CHECK (vivienda BETWEEN 280.00 AND 2100.00),
    alimentacion     NUMERIC(10,2) NOT NULL CHECK (alimentacion BETWEEN 95.00 AND 850.00),
    transporte       NUMERIC(10,2) NOT NULL CHECK (transporte BETWEEN 50.00 AND 520.00),
    ocio             NUMERIC(10,2) NOT NULL CHECK (ocio BETWEEN 28.00 AND 420.00),
    salud            NUMERIC(10,2) NOT NULL CHECK (salud BETWEEN 10.00 AND 310.00),
    educacion        NUMERIC(10,2) NOT NULL CHECK (educacion BETWEEN 8.00 AND 290.00),
    otros            NUMERIC(10,2) NOT NULL CHECK (otros BETWEEN 15.00 AND 410.00),
    gasto_total      NUMERIC(10,2) NOT NULL CHECK (gasto_total BETWEEN 870.00 AND 5600.00),
    ahorro           NUMERIC(10,2) NOT NULL CHECK (ahorro BETWEEN -190.00 AND 2480.00),
    tasa_ahorro_pct  NUMERIC(6,2) NOT NULL CHECK (tasa_ahorro_pct BETWEEN -9.80 AND 44.60),
    perfil_ahorro    VARCHAR(20) NOT NULL CHECK (perfil_ahorro IN
                        ('buen_ahorrador', 'ahorro_moderado', 'ahorro_insuficiente')),
    ipc_mensual      NUMERIC(6,3) NOT NULL CHECK (ipc_mensual BETWEEN 0.19 AND 0.60),

    -- Coherencia gasto_total = suma de las siete categorías (tolerancia 1 céntimo)
    CONSTRAINT chk_gasto_total CHECK (
        ABS(gasto_total - (vivienda + alimentacion + transporte + ocio + salud + educacion + otros)) <= 0.01
    ),
    -- Coherencia ahorro = salario - gasto_total (tolerancia 1 céntimo, apartado 2.2)
    CONSTRAINT chk_ahorro CHECK (
        ABS(ahorro - (salario - gasto_total)) <= 0.01
    ),
    -- Coherencia tasa_ahorro_pct = ahorro / salario * 100 (tolerancia 0,01 p.p., apartado 2.2)
    CONSTRAINT chk_tasa_ahorro CHECK (
        ABS(tasa_ahorro_pct - (ahorro / salario * 100)) <= 0.01
    ),
    -- Coherencia perfil_ahorro respecto a los umbrales documentados (apartado 2.2)
    CONSTRAINT chk_perfil_ahorro_umbral CHECK (
        (perfil_ahorro = 'buen_ahorrador'      AND tasa_ahorro_pct >= 15) OR
        (perfil_ahorro = 'ahorro_moderado'     AND tasa_ahorro_pct >= 5 AND tasa_ahorro_pct < 15) OR
        (perfil_ahorro = 'ahorro_insuficiente' AND tasa_ahorro_pct < 5)
    )
);

-- ---------------------------------------------------------------------
-- USUARIO (entidad fuerte)
-- Añadimos eliminado_en para soportar el derecho de supresión mediante
-- anonimización (ver función coach.anonimizar_usuario más abajo), en
-- lugar de un DELETE físico que entra en conflicto con la conservación
-- del historial financiero exigida por la Consulta 5.
-- ---------------------------------------------------------------------
CREATE TABLE coach.usuarios (
    usuario_id   SERIAL PRIMARY KEY,
    email        VARCHAR(150) NOT NULL UNIQUE,
    nombre       VARCHAR(150) NOT NULL,
    fecha_alta   TIMESTAMP NOT NULL DEFAULT now(),
    plan         VARCHAR(20) NOT NULL DEFAULT 'free' CHECK (plan IN ('free', 'premium')),
    pais_id      INTEGER REFERENCES coach.paises(pais_id) ON DELETE RESTRICT,
    eliminado_en TIMESTAMP
);

-- ---------------------------------------------------------------------
-- CUENTA (entidad fuerte, dependencia existencial de USUARIO — (1,1))
-- saldo_actual es una columna desnormalizada por rendimiento (evita
-- recalcular SUM(transacciones.importe) en cada consulta de saldo);
-- su sincronización queda garantizada por el trigger
-- trg_actualizar_saldo_cuenta definido más abajo, no por disciplina de
-- aplicación.
-- ---------------------------------------------------------------------
CREATE TABLE coach.cuentas (
    cuenta_id      SERIAL PRIMARY KEY,
    usuario_id     INTEGER NOT NULL REFERENCES coach.usuarios(usuario_id)
                     ON DELETE CASCADE,
    tipo_cuenta    VARCHAR(20) NOT NULL CHECK (tipo_cuenta IN
                     ('corriente', 'ahorro', 'tarjeta_credito', 'inversion')),
    moneda         CHAR(3) NOT NULL DEFAULT 'EUR',
    saldo_actual   NUMERIC(12,2) NOT NULL DEFAULT 0,
    fecha_apertura TIMESTAMP NOT NULL DEFAULT now(),
    -- El saldo no puede ser negativo salvo que sea una tarjeta de crédito (apartado 5)
    CONSTRAINT chk_saldo_no_negativo CHECK (
        saldo_actual >= 0 OR tipo_cuenta = 'tarjeta_credito'
    )
);

-- ---------------------------------------------------------------------
-- TRANSACCION (entidad fuerte, dependencia existencial de CUENTA — (1,1))
-- Dominio de categoria: 10 valores — 7 alineados con Dataset A.2 + 3
-- propios de un libro de movimientos real (apartado 2.3, corregido).
-- ---------------------------------------------------------------------
CREATE TABLE coach.transacciones (
    transaccion_id SERIAL PRIMARY KEY,
    cuenta_id      INTEGER NOT NULL REFERENCES coach.cuentas(cuenta_id)
                     ON DELETE RESTRICT,
    fecha          TIMESTAMP NOT NULL DEFAULT now(),
    importe        NUMERIC(12,2) NOT NULL CHECK (importe <> 0),
    categoria      VARCHAR(20) NOT NULL CHECK (categoria IN (
                     'vivienda', 'alimentacion', 'transporte', 'ocio',
                     'salud', 'educacion', 'otros',
                     'ingresos', 'ahorro', 'transferencia'
                   )),
    descripcion    VARCHAR(255)
);

-- ---------------------------------------------------------------------
-- OBJETIVO_FINANCIERO (entidad fuerte, dependencia existencial de USUARIO)
-- ---------------------------------------------------------------------
CREATE TABLE coach.objetivos_financieros (
    objetivo_id      SERIAL PRIMARY KEY,
    usuario_id       INTEGER NOT NULL REFERENCES coach.usuarios(usuario_id)
                       ON DELETE CASCADE,
    descripcion      VARCHAR(255) NOT NULL,
    importe_objetivo NUMERIC(12,2) NOT NULL CHECK (importe_objetivo > 0),
    fecha_limite     DATE,
    estado           VARCHAR(20) NOT NULL DEFAULT 'pendiente' CHECK (estado IN
                       ('pendiente', 'en_progreso', 'cumplido', 'cancelado'))
);

-- ---------------------------------------------------------------------
-- RECOMENDACION (entidad fuerte, dependencia existencial de USUARIO)
-- ---------------------------------------------------------------------
CREATE TABLE coach.recomendaciones (
    recomendacion_id SERIAL PRIMARY KEY,
    usuario_id       INTEGER NOT NULL REFERENCES coach.usuarios(usuario_id)
                       ON DELETE CASCADE,
    tipo             VARCHAR(50) NOT NULL,
    contenido        TEXT NOT NULL,
    fecha_emision    TIMESTAMP NOT NULL DEFAULT now(),
    estado           VARCHAR(20) NOT NULL DEFAULT 'pendiente' CHECK (estado IN
                       ('pendiente', 'aceptada', 'rechazada')),
    confianza_modelo NUMERIC(5,4) CHECK (confianza_modelo BETWEEN 0 AND 1)
);

-- ---------------------------------------------------------------------
-- INTERACCIONES (entidad fuerte, dependencia existencial de USUARIO)
-- Dataset C — interacciones con el asistente y salidas de los tres
-- modelos de ML. Estructura semiestructurada (apartado 2.4): los campos
-- comunes (usuario_id, tipo_interaccion, timestamp) se modelan como
-- columnas, y la parte variable según el modelo que genera cada
-- interacción se persiste en una columna JSONB con validación mínima
-- mediante restricción CHECK, en lugar de en una colección MongoDB
-- separada (decisión revisada respecto al diseño inicial documentado
-- en versiones anteriores del trabajo; ver apartado 4.3).
-- ---------------------------------------------------------------------
CREATE TABLE coach.interacciones (
    interaccion_id    SERIAL PRIMARY KEY,
    usuario_id        INTEGER NOT NULL REFERENCES coach.usuarios(usuario_id)
                        ON DELETE CASCADE,
    tipo_interaccion  VARCHAR(30) NOT NULL CHECK (tipo_interaccion IN (
                        'prediccion_ahorro', 'clasificacion_perfil', 'proyeccion_temporal'
                      )),
    timestamp         TIMESTAMP NOT NULL DEFAULT now(),
    entrada_usuario   TEXT,
    respuesta_mostrada TEXT NOT NULL,
    salida_modelo     JSONB NOT NULL,
    -- Validación mínima de esquema: el documento JSON debe declarar
    -- qué modelo lo generó (equivalente físico del $jsonSchema
    -- documentado inicialmente para MongoDB)
    CONSTRAINT chk_salida_modelo_tiene_modelo CHECK (salida_modelo ? 'modelo')
);

-- =====================================================================
-- ÍNDICES (apartado 5 y explicaciones de las Consultas 1-7, apartado 6)
-- =====================================================================

-- Consulta 1: series temporales por indicador y país
-- (la propia restricción UNIQUE ya crea el índice compuesto necesario)

-- Apartado 5 / Consulta 6: filtros interactivos de Streamlit sobre
-- observaciones_sinteticas — dos índices con propósitos distintos
CREATE INDEX idx_observaciones_sinteticas_perfil_fecha
    ON coach.observaciones_sinteticas (perfil, fecha);
CREATE INDEX idx_observaciones_sinteticas_perfil_ahorro
    ON coach.observaciones_sinteticas (perfil, perfil_ahorro);

-- Apartado 5: índices sobre claves ajenas y columnas de fecha del Dataset B
CREATE INDEX idx_cuentas_usuario
    ON coach.cuentas (usuario_id);
CREATE INDEX idx_transacciones_cuenta_fecha
    ON coach.transacciones (cuenta_id, fecha DESC);
CREATE INDEX idx_recomendaciones_usuario_fecha
    ON coach.recomendaciones (usuario_id, fecha_emision DESC);

-- Consulta 3: agregación por categoría
CREATE INDEX idx_transacciones_categoria
    ON coach.transacciones (categoria);

-- Consulta 7 (Dataset C): filtrado por usuario y orden cronológico, más
-- índice GIN para consultas dentro del contenido JSONB de salida_modelo
CREATE INDEX idx_interacciones_usuario_timestamp
    ON coach.interacciones (usuario_id, timestamp DESC);
CREATE INDEX idx_interacciones_salida_modelo_gin
    ON coach.interacciones USING GIN (salida_modelo);

-- Buenas prácticas adicionales no citadas explícitamente en el documento,
-- pero recomendables para el resto de claves ajenas del Dataset B
CREATE INDEX idx_objetivos_usuario
    ON coach.objetivos_financieros (usuario_id);
CREATE INDEX idx_usuarios_pais
    ON coach.usuarios (pais_id);

-- =====================================================================
-- VISTA (apartado 5): saldo agregado por usuario
-- =====================================================================
CREATE OR REPLACE VIEW coach.v_saldo_usuario AS
SELECT u.usuario_id,
       u.nombre,
       COALESCE(SUM(c.saldo_actual), 0) AS saldo_total
FROM coach.usuarios u
LEFT JOIN coach.cuentas c ON c.usuario_id = u.usuario_id
GROUP BY u.usuario_id, u.nombre;

-- =====================================================================
-- CORRECCIÓN 4 (debilidad docente): sincronización de cuentas.saldo_actual
--
-- Antes: saldo_actual se escribía manualmente en el INSERT/UPDATE de
-- cuentas, sin ningún mecanismo que lo mantuviera coherente con el
-- histórico real de transacciones (podían coexistir dos
-- representaciones contradictorias del saldo, tal como señaló la
-- revisión docente).
--
-- Ahora: cada INSERT, UPDATE o DELETE sobre transacciones dispara un
-- recálculo completo de saldo_actual para la(s) cuenta(s) afectada(s),
-- a partir de SUM(transacciones.importe). saldo_actual sigue siendo una
-- columna desnormalizada (por rendimiento de lectura, apartado 5), pero
-- su valor es ahora siempre una función determinista del histórico de
-- transacciones, nunca un dato introducido de forma independiente.
-- =====================================================================
CREATE OR REPLACE FUNCTION coach.fn_actualizar_saldo_cuenta()
RETURNS TRIGGER AS $$
DECLARE
    v_cuenta_id INTEGER;
BEGIN
    IF TG_OP = 'DELETE' THEN
        v_cuenta_id := OLD.cuenta_id;
    ELSE
        v_cuenta_id := NEW.cuenta_id;
    END IF;

    UPDATE coach.cuentas
    SET saldo_actual = COALESCE(
        (SELECT SUM(importe) FROM coach.transacciones WHERE cuenta_id = v_cuenta_id),
        0
    )
    WHERE cuenta_id = v_cuenta_id;

    -- Si un UPDATE reasigna la transacción a otra cuenta, recalculamos
    -- también la cuenta de origen para no dejarla con un saldo obsoleto.
    IF TG_OP = 'UPDATE' AND OLD.cuenta_id IS DISTINCT FROM NEW.cuenta_id THEN
        UPDATE coach.cuentas
        SET saldo_actual = COALESCE(
            (SELECT SUM(importe) FROM coach.transacciones WHERE cuenta_id = OLD.cuenta_id),
            0
        )
        WHERE cuenta_id = OLD.cuenta_id;
    END IF;

    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_actualizar_saldo_cuenta
    AFTER INSERT OR UPDATE OR DELETE ON coach.transacciones
    FOR EACH ROW EXECUTE FUNCTION coach.fn_actualizar_saldo_cuenta();

-- =====================================================================
-- CORRECCIÓN 2 (debilidad docente): derecho de supresión del usuario
-- conservando el historial financiero
--
-- Antes: la única vía documentada para "eliminar" un usuario era un
-- DELETE físico sobre coach.usuarios, que la Consulta 5 demostraba
-- correctamente bloqueado por ON DELETE RESTRICT (cuentas ->
-- transacciones) en cuanto existía algún movimiento. Esto protegía el
-- historial, pero dejaba sin resolver la propia historia de usuario:
-- el usuario no podía ejercer su derecho de supresión de ninguna forma.
--
-- Ahora: coach.anonimizar_usuario() implementa el derecho de supresión
-- mediante anonimización (borrado lógico de datos personales), en línea
-- con la práctica habitual bajo RGPD cuando existe una obligación legal
-- de conservar registros financieros: se sobrescriben los datos
-- identificativos (email, nombre) y se marca eliminado_en, sin borrar
-- ni las cuentas ni las transacciones ni el resto del historial. Las
-- reglas CASCADE/RESTRICT existentes se mantienen sin cambios: siguen
-- siendo la salvaguarda correcta para un DELETE físico completo (por
-- ejemplo, purgar una cuenta de prueba sin datos reales), mientras que
-- anonimizar_usuario() es la vía correcta para el caso de uso real de
-- "un usuario ejerce su derecho de supresión".
-- =====================================================================
CREATE OR REPLACE FUNCTION coach.anonimizar_usuario(p_usuario_id INTEGER)
RETURNS VOID AS $$
BEGIN
    UPDATE coach.usuarios
    SET email        = 'eliminado+' || p_usuario_id || '@anon.local',
        nombre       = 'Usuario eliminado',
        eliminado_en = now()
    WHERE usuario_id = p_usuario_id
      AND eliminado_en IS NULL;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Usuario % no existe o ya había sido anonimizado previamente', p_usuario_id;
    END IF;
END;
$$ LANGUAGE plpgsql;
