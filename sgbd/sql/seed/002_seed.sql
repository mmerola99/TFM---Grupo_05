-- =====================================================================
-- AI Financial Life Coach — Grupo 05
-- Datos de ejemplo (seed) para validar el esquema completo en local,
-- sin necesidad de conectarse a las API/fuentes externas (apartado 5).
-- =====================================================================

SET search_path TO coach, public;

-- ---------------------------------------------------------------------
-- PAIS
-- ---------------------------------------------------------------------
INSERT INTO coach.paises (codigo_iso2, nombre) VALUES
    ('ES', 'España');

-- ---------------------------------------------------------------------
-- INDICADOR_MACRO (códigos reales citados en el apartado 2.1)
-- ---------------------------------------------------------------------
INSERT INTO coach.indicadores_macro (codigo_eurostat, nombre, unidad, descripcion) VALUES
    ('nasq_10_ki',      'Tasa de ahorro bruto de los hogares', '%',      'Eurostat, frecuencia trimestral (2015-2025)'),
    ('une_rt_a',         'Tasa de desempleo 25-34 años',        '%',      'Eurostat, frecuencia anual (2010-2024)'),
    ('prc_hicp_manr',    'Variación anual del IPC',             '%',      'Eurostat, frecuencia anual (2019-2024)'),
    ('bde_morosidad',    'Tasa de morosidad bancaria',          '%',      'Banco de España, Boletín Estadístico cap. 4 (web scraping, 2008-2024)'),
    ('wiki_euribor_12m', 'Euríbor a 12 meses (media anual)',    'p.p.',   'Wikipedia, tabla histórica (web scraping, 2008-2024)');

-- ---------------------------------------------------------------------
-- OBSERVACION_MACRO (resuelve la relación N:M PAIS <-> INDICADOR_MACRO)
-- ---------------------------------------------------------------------
INSERT INTO coach.observaciones_macro (indicador_id, pais_id, periodo, valor, fuente, fecha_extraccion) VALUES
    ((SELECT indicador_id FROM coach.indicadores_macro WHERE codigo_eurostat = 'nasq_10_ki'),
     (SELECT pais_id FROM coach.paises WHERE codigo_iso2 = 'ES'), '2024', 12.70, 'Eurostat API', now()),
    ((SELECT indicador_id FROM coach.indicadores_macro WHERE codigo_eurostat = 'une_rt_a'),
     (SELECT pais_id FROM coach.paises WHERE codigo_iso2 = 'ES'), '2024', 16.90, 'Eurostat API', now()),
    ((SELECT indicador_id FROM coach.indicadores_macro WHERE codigo_eurostat = 'prc_hicp_manr'),
     (SELECT pais_id FROM coach.paises WHERE codigo_iso2 = 'ES'), '2022',  8.30, 'Eurostat API', now()),
    ((SELECT indicador_id FROM coach.indicadores_macro WHERE codigo_eurostat = 'prc_hicp_manr'),
     (SELECT pais_id FROM coach.paises WHERE codigo_iso2 = 'ES'), '2023',  3.50, 'Eurostat API', now()),
    ((SELECT indicador_id FROM coach.indicadores_macro WHERE codigo_eurostat = 'prc_hicp_manr'),
     (SELECT pais_id FROM coach.paises WHERE codigo_iso2 = 'ES'), '2024',  2.80, 'Eurostat API', now()),
    ((SELECT indicador_id FROM coach.indicadores_macro WHERE codigo_eurostat = 'wiki_euribor_12m'),
     (SELECT pais_id FROM coach.paises WHERE codigo_iso2 = 'ES'), '2026',  2.20, 'Wikipedia (scraping)', now()),
    ((SELECT indicador_id FROM coach.indicadores_macro WHERE codigo_eurostat = 'bde_morosidad'),
     (SELECT pais_id FROM coach.paises WHERE codigo_iso2 = 'ES'), '2024',  3.60, 'Banco de España (scraping)', now());

-- ---------------------------------------------------------------------
-- OBSERVACION_SINTETICA (3 filas de ejemplo, una por cada perfil_ahorro,
-- construidas para cumplir exactamente las restricciones CHECK de
-- coherencia interna y de rango definidas en el DDL, estas últimas
-- verificadas contra la tabla de variables del trabajo de la
-- Asignatura 5 "Obtención de Datos para el TFM")
-- ---------------------------------------------------------------------
INSERT INTO coach.observaciones_sinteticas
    (user_id, fecha, edad, perfil, salario, vivienda, alimentacion, transporte, ocio, salud, educacion, otros,
     gasto_total, ahorro, tasa_ahorro_pct, perfil_ahorro, ipc_mensual)
VALUES
    -- buen_ahorrador (tasa 30%)
    (1, '2023-01-01', 29, 'medio', 2500.00, 750.00, 400.00, 200.00, 150.00, 100.00, 50.00, 100.00,
     1750.00, 750.00, 30.00, 'buen_ahorrador', 0.200),
    -- ahorro_moderado (tasa 10%)
    (2, '2023-01-01', 34, 'freelance', 1800.00, 600.00, 400.00, 200.00, 150.00, 120.00, 80.00, 70.00,
     1620.00, 180.00, 10.00, 'ahorro_moderado', 0.220),
    -- ahorro_insuficiente (tasa ~3,08%)
    (3, '2023-01-01', 27, 'junior', 1300.00, 500.00, 350.00, 150.00, 100.00, 80.00, 40.00, 40.00,
     1260.00, 40.00, 3.08, 'ahorro_insuficiente', 0.250);

-- ---------------------------------------------------------------------
-- USUARIO
-- ---------------------------------------------------------------------
INSERT INTO coach.usuarios (email, nombre, plan, pais_id) VALUES
    ('ana.garcia@example.com',   'Ana García',   'premium', (SELECT pais_id FROM coach.paises WHERE codigo_iso2 = 'ES')),
    ('luis.martinez@example.com','Luis Martínez','free',    (SELECT pais_id FROM coach.paises WHERE codigo_iso2 = 'ES'));

-- ---------------------------------------------------------------------
-- CUENTA
-- ---------------------------------------------------------------------
INSERT INTO coach.cuentas (usuario_id, tipo_cuenta, moneda, saldo_actual) VALUES
    (1, 'corriente', 'EUR', 2450.00),
    (1, 'ahorro',    'EUR', 8200.00),
    (2, 'corriente', 'EUR', 640.00);

-- ---------------------------------------------------------------------
-- TRANSACCION (cubre las 10 categorías del dominio corregido)
-- ---------------------------------------------------------------------
INSERT INTO coach.transacciones (cuenta_id, fecha, importe, categoria, descripcion) VALUES
    (1, now() - interval '30 days', -750.00, 'vivienda',     'Alquiler mensual'),
    (1, now() - interval '28 days', -180.00, 'alimentacion', 'Supermercado'),
    (1, now() - interval '20 days',  -35.00, 'ocio',         'Cine y cena'),
    (1, now() - interval '15 days',  -60.00, 'transporte',   'Abono transporte'),
    (1, now() - interval '10 days', 2200.00, 'ingresos',     'Nómina mensual'),
    (1, now() - interval '9 days',  -300.00, 'ahorro',       'Transferencia a cuenta de ahorro'),
    (2, now() - interval '9 days',   300.00, 'transferencia','Recepción desde cuenta corriente'),
    (3, now() - interval '5 days',   -45.00, 'salud',        'Farmacia'),
    (3, now() - interval '2 days',  -120.00, 'educacion',    'Curso online');

-- ---------------------------------------------------------------------
-- OBJETIVO_FINANCIERO
-- ---------------------------------------------------------------------
INSERT INTO coach.objetivos_financieros (usuario_id, descripcion, importe_objetivo, fecha_limite, estado) VALUES
    (1, 'Fondo de emergencia (3 meses de gastos)', 5000.00, '2026-12-31', 'en_progreso'),
    (2, 'Ahorro para vacaciones',                   800.00, '2026-09-01', 'pendiente');

-- ---------------------------------------------------------------------
-- RECOMENDACION
-- ---------------------------------------------------------------------
INSERT INTO coach.recomendaciones (usuario_id, tipo, contenido, estado, confianza_modelo) VALUES
    (1, 'ahorro_sugerido', 'Podrías incrementar tu tasa de ahorro un 5% reduciendo gasto en ocio.', 'aceptada', 0.9200),
    (2, 'alerta_gasto',    'Tu gasto en transporte ha subido un 20% respecto al mes anterior.',      'pendiente', 0.8100);
