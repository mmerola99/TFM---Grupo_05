# AI Financial Life Coach — Grupo 05 — MVP SGBD

**Asignatura:** Sistemas de Gestión de Bases de Datos (SGBD) — Máster en Big
Data & Business Intelligence, Next Educación
**Grupo 05:** C. Solis Meza, J. Cáceres Mondragón, M. Merola, M. Á. Lozano Torres
**Repositorio:** esta carpeta (`sgbd/`) forma parte del monorepo del TFM
[TFM---Grupo_05](https://github.com/mmerola99/TFM---Grupo_05); el resto del
repositorio corresponde a la Asignatura 5 (obtención de datos).

## 1. Qué contiene esta entrega

En este trabajo implementamos físicamente los Datasets A (A.1 y A.2) y B sobre
PostgreSQL 16, y documentamos —sin implementarlo en este MVP— el diseño del
Dataset C sobre MongoDB. Esta carpeta contiene todo lo necesario para levantar
y evaluar esa implementación en local, sin ninguna dependencia externa:

- La implementación completa del modelo físico (DDL, índices, restricciones)
- Los datos de ejemplo (seed) y las seis consultas representativas documentadas
- La documentación del modelado de datos, en `docs/`
- Este README, con instrucciones paso a paso para reproducir el entorno

Antes de entregar este trabajo, probamos el conjunto completo de extremo a
extremo (esquema, seed y las seis consultas) en un entorno limpio: no se
producen errores de sintaxis, todas las restricciones `CHECK` se satisfacen
con los datos de ejemplo, y la Consulta 5 confirma el comportamiento
`ON DELETE RESTRICT` esperado. Los pasos siguientes reproducen exactamente esa
misma verificación.

## 2. Estructura del repositorio

```
.
├── docker-compose.yml
├── docs/
│   ├── modelo_conceptual.md            # entidades, claves, reglas de borrado, dominios, tecnología
│   ├── diagrama_er_corregido.png       # diagrama Entidad-Relación
│   └── tink_mapping.md                 # mapeo de referencia Open Banking (Tink) — evolución futura, no implementado
├── sql/
│   ├── ddl/001_schema.sql              # esquema completo (tablas, CHECK, FK, índices, vista)
│   ├── seed/002_seed.sql               # datos de ejemplo, listos para las 6 consultas
│   └── queries/
│       ├── consultas_representativas.sql   # las 6 consultas del apartado 6, comentadas
│       └── consulta5_automatica.sql        # Consulta 5 sin marcadores manuales
└── mongo/
    └── schema_interacciones.design.js  # esquema de referencia del Dataset C (solo diseño)
```

Para una síntesis del modelo de datos (entidades, claves, reglas de borrado,
dominios y elección tecnológica) sin necesidad de abrir el documento PDF
entregado, puede consultarse directamente
[`docs/modelo_conceptual.md`](./docs/modelo_conceptual.md).

## 3. Requisitos previos

Antes de empezar, es necesario tener instalado:

| Herramienta | Uso | Enlace |
|---|---|---|
| Docker Desktop | Ejecuta PostgreSQL en un contenedor aislado, sin instalación nativa | https://www.docker.com/products/docker-desktop/ |
| Visual Studio Code | Editor para abrir el repositorio y ejecutar las consultas | https://code.visualstudio.com/ |
| Extensión Docker (Microsoft) | Gestiona el contenedor desde VS Code | Marketplace de VS Code |
| Extensión PostgreSQL (Chris Kolkman) o SQLTools | Conecta VS Code a la base de datos y ejecuta ficheros `.sql` | Marketplace de VS Code |

Detallamos la instalación de cada uno en la sección 4. Si ya se dispone de
Docker Desktop y VS Code, puede pasarse directamente a la sección 5.

## 4. Instalación de las herramientas (una sola vez)

### 4.1 Docker Desktop (Windows)

1. Descargar "Docker Desktop for Windows" desde
   **https://www.docker.com/products/docker-desktop/**.
2. Ejecutar el instalador. Si se solicita, mantener activada la opción
   **"Use WSL 2 instead of Hyper-V"** (backend recomendado).
3. Reiniciar el equipo si el instalador lo solicita.
4. Abrir Docker Desktop y esperar a que el icono de la ballena en la barra de
   tareas indique que está en marcha ("Docker Desktop is running"). La
   primera apertura puede tardar varios minutos.

### 4.2 Visual Studio Code

1. Descargar desde **https://code.visualstudio.com/** → "Download for Windows".
2. Ejecutar el instalador con las opciones por defecto.

### 4.3 Extensiones de VS Code

Desde el icono de extensiones en la barra lateral (o `Ctrl+Shift+X`), instalar:

- **Docker** (de Microsoft)
- **PostgreSQL** (de Chris Kolkman), o alternativamente **SQLTools** junto con
  el driver **SQLTools PostgreSQL/Redshift Driver**

## 5. Levantar el MVP

1. Clonar o descomprimir este repositorio en una carpeta local, por ejemplo
   `C:\TFM\sgbd\`.
2. Abrir esa carpeta en VS Code (`Archivo → Abrir carpeta...`).
3. Abrir una terminal integrada (`` Ctrl+ñ `` o `Terminal → Nueva terminal`)
   y ejecutar:

   ```powershell
   docker compose up -d
   ```

   Este comando descarga la imagen `postgres:16-alpine`, crea el contenedor
   `afilc_postgres`, y ejecuta automáticamente `sql/ddl/001_schema.sql` y
   `sql/seed/002_seed.sql` en ese orden, mediante el mecanismo
   `docker-entrypoint-initdb.d` de la imagen oficial de PostgreSQL. La primera
   ejecución tarda entre 30 segundos y 2 minutos, según la conexión; las
   siguientes son casi instantáneas.

4. Verificar que el contenedor está operativo:

   ```powershell
   docker compose ps
   ```

   **Resultado esperado:** una fila con el nombre `afilc_postgres` y estado
   `healthy`. Si el estado es `starting`, esperar unos segundos y repetir el
   comando.

## 6. Conectar VS Code a la base de datos

Con la extensión PostgreSQL instalada:

1. Abrir el icono del elefante de PostgreSQL en la barra lateral → **"+"**
   para añadir una nueva conexión.
2. Completar los parámetros de conexión exactamente como siguen:

   | Parámetro | Valor |
   |---|---|
   | Host | `localhost` |
   | Port | `5432` |
   | Usuario | `afilc_user` |
   | Contraseña | `afilc_pass` |
   | Base de datos | `afilc` |
   | Tipo de conexión | Standard Connection (sin SSL) |

3. Al finalizar, la conexión debería mostrar el esquema `coach` con las nueve
   tablas y la vista `v_saldo_usuario`.
4. Antes de ejecutar cualquier consulta, seleccionar esta conexión como activa
   desde la barra de estado inferior de VS Code (donde se indica
   "Select Postgres Server" o el nombre de la conexión ya seleccionada).

## 7. Ejecutar las seis consultas representativas (apartado 6 del trabajo)

Abrir `sql/queries/consultas_representativas.sql`. Para cada consulta:
seleccionar el bloque de código correspondiente y ejecutar **"Run Query"**
(botón sobre la selección, o clic derecho → Run Query).

| Consulta | Resultado esperado |
|---|---|
| 1 — Serie temporal IPC | 3 filas (periodos 2022, 2023, 2024) con `pais`, `indicador`, `periodo`, `valor` |
| 2 — Saldo y movimientos | Una fila con el saldo total del usuario 1, seguida de sus últimos movimientos |
| 3 — Gasto por categoría y mes | Filas agrupadas por `categoria` y `mes` para el usuario 1 |
| 4 — Recomendaciones y tasa de aceptación | Historial de recomendaciones, seguido de `tasa_aceptacion_pct` |
| 6 — Perfil de ahorro por empleo | 3 filas (una por cada perfil laboral presente en el seed) |

**Consulta 5** requiere una atención particular, porque contiene marcadores
intencionados (`<usuario_id>`, `<cuenta_id>`) que reproducen la secuencia
manual descrita en el trabajo:

- **Paso a paso:** ejecutar el primer `INSERT ... RETURNING`, anotar el
  `usuario_id` devuelto, sustituirlo donde aparece `<usuario_id>`, repetir con
  `<cuenta_id>`, y finalmente ejecutar el `DELETE`. El resultado esperado es un
  error de violación de clave foránea, que demuestra que `ON DELETE RESTRICT`
  bloquea correctamente el borrado.
- **Automática:** ejecutar en su lugar
  `sql/queries/consulta5_automatica.sql`, que reproduce la misma secuencia sin
  marcadores manuales y finaliza con el mensaje
  `NOTICE: OK: el borrado se ha bloqueado correctamente por ON DELETE RESTRICT (cuentas -> transacciones)`.

## 8. Detener o reiniciar el entorno

```powershell
docker compose down          # detiene el contenedor, conserva los datos
docker compose down -v       # detiene el contenedor y elimina también el volumen (reinicio completo)
```

## 9. Alternativa sin Docker (PostgreSQL instalado de forma nativa)

Si se prefiere no usar Docker:

1. Descargar el instalador de PostgreSQL 16 para Windows desde
   **https://www.postgresql.org/download/windows/** (instalador de EDB) y
   ejecutarlo. Anotar la contraseña definida para el usuario `postgres`.
2. Abrir **pgAdmin** (se instala junto con PostgreSQL) o la terminal `psql`.
3. Crear la base de datos y el usuario:
   ```sql
   CREATE DATABASE afilc;
   CREATE USER afilc_user WITH PASSWORD 'afilc_pass';
   GRANT ALL PRIVILEGES ON DATABASE afilc TO afilc_user;
   ALTER DATABASE afilc OWNER TO afilc_user;
   ```
4. Ejecutar el DDL y el seed, en este orden:
   ```powershell
   psql -h localhost -U afilc_user -d afilc -f sql\ddl\001_schema.sql
   psql -h localhost -U afilc_user -d afilc -f sql\seed\002_seed.sql
   ```
5. Ejecutar las consultas de la sección 7 con `psql` o pgAdmin, en lugar de la
   extensión de VS Code.

## 10. Sobre el Dataset C (MongoDB)

`mongo/schema_interacciones.design.js` es documentación de diseño: **no**
forma parte del código a ejecutar en este MVP, decisión que justificamos en
el apartado 5.2 del trabajo. El fichero contiene tres documentos de ejemplo
(uno por cada modelo de machine learning del TFM) y el esquema de validación
`$jsonSchema` completo, listos para aplicarse cuando incorporemos físicamente
el contenedor MongoDB, en el sprint posterior a la construcción de la interfaz
conversacional (apartado 7 del trabajo).

## 11. Resolución de problemas frecuentes

| Síntoma | Causa probable | Solución |
|---|---|---|
| `docker compose up -d` no responde o el contenedor no arranca | Docker Desktop no está iniciado, o el motor WSL2 no ha arrancado | Abrir Docker Desktop y esperar a que indique "running" antes de repetir el comando |
| VS Code muestra "No PostgreSQL Server or Database selected" al ejecutar una consulta | El fichero `.sql` no tiene una conexión activa asociada | Seleccionar la conexión desde la barra de estado inferior antes de ejecutar |
| Error `syntax error at or near "<"` al ejecutar todo el fichero de consultas de una vez | Es el comportamiento esperado en la Consulta 5 (ver sección 7) | Ejecutar `consulta5_automatica.sql` en su lugar, o sustituir los marcadores manualmente |
| `docker compose ps` muestra el contenedor en estado `starting` de forma prolongada | Puede haberse producido un conflicto con el puerto 5432 si ya hay otro PostgreSQL escuchando en local | Detener el otro servicio PostgreSQL, o cambiar el puerto expuesto en `docker-compose.yml` |
