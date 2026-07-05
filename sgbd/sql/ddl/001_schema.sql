-- =====================================================================
-- AI Financial Life Coach — Grupo 05
-- DDL del esquema físico (Datasets A.1, A.2 y B) sobre PostgreSQL 16
-- Corresponde al modelo E/R documentado en el apartado 3.2 del trabajo
-- (version corregida: todas las entidades son entidades fuertes con
-- clave primaria propia; la dependencia existencial se materializa
-- mediante FK obligatoria + regla de borrado, no mediante clave parcial).
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
-- Replica fielmente las 18 columnas de 02_api/genera_dataset_sintetico.py.
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
-- ---------------------------------------------------------------------
CREATE TABLE coach.usuarios (
    usuario_id   SERIAL PRIMARY KEY,
    email        VARCHAR(150) NOT NULL UNIQUE,
    nombre       VARCHAR(150) NOT NULL,
    fecha_alta   TIMESTAMP NOT NULL DEFAULT now(),
    plan         VARCHAR(20) NOT NULL DEFAULT 'free' CHECK (plan IN ('free', 'premium')),
    pais_id      INTEGER REFERENCES coach.paises(pais_id) ON DELETE RESTRICT
);

-- ---------------------------------------------------------------------
-- CUENTA (entidad fuerte, dependencia existencial de USUARIO — (1,1))
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

-- =====================================================================
-- ÍNDICES (apartado 5 y explicaciones de las Consultas 1-6, apartado 6)
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
