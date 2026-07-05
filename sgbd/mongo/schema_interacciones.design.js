// =====================================================================
// AI Financial Life Coach — Grupo 05
// Esquema de documento de referencia para el Dataset C (interacciones
// con el asistente y salidas de los modelos), apartado 3.5 del trabajo.
//
// IMPORTANTE: este fichero es documentación de diseño, NO forma parte
// del MVP ejecutable entregado (apartado 5.2 del trabajo). No hay
// contenedor MongoDB en docker-compose.yml. Se incluye aquí para dejar
// constancia del esquema que se implementará físicamente en el sprint
// posterior a la construcción de la interfaz conversacional (apartado 7).
// =====================================================================

// ---------------------------------------------------------------------
// 1. Ejemplos de documento por tipo de interacción (JSON anotado)
// ---------------------------------------------------------------------

// Ejemplo A — salida del modelo de Regresión Lineal (predicción de ahorro)
const ejemplo_regresion_lineal = {
  usuario_id: 1,
  tipo_interaccion: "prediccion_ahorro",   // dominio cerrado, ver $jsonSchema abajo
  timestamp: "2026-07-01T10:15:00Z",
  entrada_usuario: "¿Cuánto podré ahorrar el próximo mes?",
  respuesta_mostrada: "Se estima un ahorro de 420 EUR el próximo mes.",
  salida_modelo: {
    modelo: "regresion_lineal",
    prediccion_ahorro_eur: 420.35,
    r2: 1.0,
    mae_eur: 0,
    features_utilizadas: ["salario", "gasto_total", "tasa_ahorro_pct_historica"],
  },
};

// Ejemplo B — salida del modelo de Regresión Logística (clasificación de perfil)
const ejemplo_regresion_logistica = {
  usuario_id: 1,
  tipo_interaccion: "clasificacion_perfil",
  timestamp: "2026-07-01T10:16:00Z",
  entrada_usuario: null,                    // generado automáticamente, sin pregunta explícita
  respuesta_mostrada: "Tu perfil actual es: ahorro_moderado.",
  salida_modelo: {
    modelo: "regresion_logistica",
    perfil_predicho: "ahorro_moderado",
    accuracy: 0.963,
    recall_ahorro_insuficiente: 1.0,
    probabilidades: {
      buen_ahorrador: 0.18,
      ahorro_moderado: 0.71,
      ahorro_insuficiente: 0.11,
    },
  },
};

// Ejemplo C — salida del modelo de serie temporal (proyección a 6 meses)
const ejemplo_serie_temporal = {
  usuario_id: 1,
  tipo_interaccion: "proyeccion_temporal",
  timestamp: "2026-07-01T10:17:00Z",
  entrada_usuario: "Muéstrame mi proyección de ahorro a 6 meses",
  respuesta_mostrada: "Proyección de ahorro para los próximos 6 meses adjunta.",
  salida_modelo: {
    modelo: "serie_temporal",
    horizonte_meses: 6,
    proyeccion_eur: [420, 435, 440, 455, 460, 470],
  },
};

// ---------------------------------------------------------------------
// 2. Esquema de validación declarativo ($jsonSchema) — apartado 3.5
//    Aplicar con db.createCollection() o db.runCommand({collMod: ...})
//    en el momento de la incorporación física (apartado 7).
// ---------------------------------------------------------------------
const interacciones_jsonSchema = {
  $jsonSchema: {
    bsonType: "object",
    required: ["usuario_id", "tipo_interaccion", "timestamp", "salida_modelo"],
    properties: {
      usuario_id: {
        bsonType: "int",
        description: "Referencia lógica a coach.usuarios(usuario_id) en PostgreSQL. " +
                      "No es una FK real: MongoDB no impone integridad referencial " +
                      "entre bases distintas; la consistencia se gestiona a nivel de aplicación.",
      },
      tipo_interaccion: {
        bsonType: "string",
        enum: ["prediccion_ahorro", "clasificacion_perfil", "proyeccion_temporal"],
        description: "Dominio cerrado, uno por cada modelo de ML del TFM (apartado 2.4).",
      },
      timestamp: {
        bsonType: "date",
        description: "Fecha y hora de la interacción, en UTC.",
      },
      entrada_usuario: {
        bsonType: ["string", "null"],
        description: "Texto introducido por el usuario; puede ser null si la interacción " +
                      "no fue iniciada explícitamente por el usuario.",
      },
      respuesta_mostrada: {
        bsonType: "string",
      },
      salida_modelo: {
        bsonType: "object",
        required: ["modelo"],
        properties: {
          modelo: {
            bsonType: "string",
            enum: ["regresion_lineal", "regresion_logistica", "serie_temporal"],
          },
          // El resto de campos de salida_modelo varía deliberadamente según
          // 'modelo' (apartado 2.4 y 3.1) y no se restringe aquí para no
          // perder la flexibilidad estructural que justifica el uso de MongoDB.
        },
      },
    },
  },
};

// ---------------------------------------------------------------------
// 3. Comando de referencia para crear la colección con validación
//    (a ejecutar en el sprint de incorporación física, no en este MVP)
// ---------------------------------------------------------------------
//
// db.createCollection("interacciones", {
//   validator: interacciones_jsonSchema,
//   validationLevel: "strict",
//   validationAction: "error",
// });
//
// db.interacciones.createIndex({ usuario_id: 1, timestamp: -1 });
// db.interacciones.createIndex({ tipo_interaccion: 1 });

module.exports = {
  ejemplo_regresion_lineal,
  ejemplo_regresion_logistica,
  ejemplo_serie_temporal,
  interacciones_jsonSchema,
};
