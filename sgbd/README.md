# AI Financial Life Coach — Grupo 05 — MVP SGBD

Repositorio técnico correspondiente al trabajo de Sistemas de Gestión de Bases
de Datos (SGBD). Implementa físicamente los Datasets A (A.1 y A.2) y B sobre
PostgreSQL 16. El Dataset C (MongoDB) se documenta en `mongo/` pero **no se
ejecuta** en este MVP (ver `docs/modelo_conceptual.md`, apartado 4).

Para una síntesis del modelo de datos (entidades, claves, reglas de borrado,
dominios y elección tecnológica) sin necesidad de abrir el documento PDF
entregado, ver [`docs/modelo_conceptual.md`](./docs/modelo_conceptual.md).

## Estructura del repositorio

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

Todo el conjunto se ha probado end-to-end (DDL + seed + las 6 consultas) antes
de esta entrega: no hay errores de sintaxis, todas las restricciones CHECK se
satisfacen con los datos de ejemplo, y la Consulta 5 confirma el
comportamiento `ON DELETE RESTRICT` esperado.

---

## Opción recomendada: Docker Desktop + Visual Studio Code

Es el camino más simple porque no hay que instalar PostgreSQL en tu máquina:
todo corre dentro de un contenedor aislado y reproducible.

### 1. Instalar Docker Desktop (Windows)

1. Ve a **https://www.docker.com/products/docker-desktop/** y descarga
   "Docker Desktop for Windows".
2. Ejecuta el instalador. Si te pregunta, deja activada la opción **"Use WSL 2
   instead of Hyper-V"** (es el backend recomendado).
3. Reinicia el ordenador si el instalador lo pide.
4. Abre Docker Desktop y espera a que el icono de la ballena en la barra de
   tareas indique que está en marcha ("Docker Desktop is running").

### 2. Instalar Visual Studio Code (si no lo tienes ya)

1. Ve a **https://code.visualstudio.com/** → botón "Download for Windows".
2. Ejecuta el instalador con las opciones por defecto.

### 3. Extensiones de VS Code necesarias

Abre VS Code → icono de extensiones en la barra lateral (o `Ctrl+Shift+X`) e
instala:

- **Docker** (de Microsoft) — para gestionar contenedores desde VS Code.
- **PostgreSQL** (de Chris Kolkman, o alternativamente **SQLTools** + el
  driver **SQLTools PostgreSQL/Redshift Driver**) — para conectarte a la base
  de datos y ejecutar `.sql` directamente desde el editor.

### 4. Levantar el MVP

1. Descomprime este repositorio en una carpeta, por ejemplo
   `C:\Lab-pruebas\afilc-sgbd\`.
2. Abre esa carpeta en VS Code (`Archivo → Abrir carpeta...`).
3. Abre una terminal integrada en VS Code (`` Ctrl+ñ `` o `Terminal → Nueva
   terminal`) y ejecuta:

   ```powershell
   docker compose up -d
   ```

   Esto descarga la imagen `postgres:16-alpine`, crea el contenedor
   `afilc_postgres` y ejecuta automáticamente `001_schema.sql` y
   `002_seed.sql` la primera vez que se crea el volumen de datos (mecanismo
   `docker-entrypoint-initdb.d` de la imagen oficial de PostgreSQL).

4. Verifica que el contenedor está sano:

   ```powershell
   docker compose ps
   ```

   Debe aparecer `afilc_postgres` con estado `healthy`.

### 5. Conectar VS Code a la base de datos

Con la extensión **PostgreSQL** instalada:

1. Icono del elefante de PostgreSQL en la barra lateral → **"+"** para añadir
   una conexión.
2. Parámetros de conexión:
   - Host: `localhost`
   - Port: `5432`
   - Usuario: `afilc_user`
   - Contraseña: `afilc_pass`
   - Base de datos: `afilc`
3. Una vez conectado, abre cualquier fichero `.sql` del repositorio, selecciona
   el texto de la consulta que quieras ejecutar y usa **"Run Query"** (botón
   que aparece sobre la selección, o clic derecho → Run Query).

### 6. Ejecutar las consultas del apartado 6

- Abre `sql/queries/consultas_representativas.sql` y ejecuta las Consultas 1,
  2, 3, 4 y 6 seleccionando cada bloque y pulsando "Run Query".
- Para la **Consulta 5**, tienes dos opciones:
  - **Paso a paso** (igual que en el documento): ejecuta el primer `INSERT`
    con `RETURNING`, copia el `usuario_id` devuelto, sustitúyelo donde pone
    `<usuario_id>`, repite con `<cuenta_id>`, y finalmente ejecuta el
    `DELETE` — verás el error de `foreign key violation` esperado.
  - **De un solo golpe**: ejecuta directamente
    `sql/queries/consulta5_automatica.sql`, que hace lo mismo automáticamente
    y muestra `NOTICE: OK: el borrado se ha bloqueado correctamente...`.

### 7. Parar / limpiar el entorno

```powershell
docker compose down          # para el contenedor, conserva los datos
docker compose down -v       # para el contenedor y borra también el volumen (reinicio limpio)
```

---

## Opción alternativa: sin Docker (PostgreSQL instalado localmente)

Si prefieres no usar Docker:

1. Descarga el instalador de PostgreSQL 16 para Windows desde
   **https://www.postgresql.org/download/windows/** (enlace al instalador de
   EDB) y ejecútalo. Anota la contraseña que definas para el usuario
   `postgres` durante la instalación.
2. Abre **pgAdmin** (se instala junto con PostgreSQL) o la terminal `psql`
   incluida.
3. Crea la base de datos y el usuario:
   ```sql
   CREATE DATABASE afilc;
   CREATE USER afilc_user WITH PASSWORD 'afilc_pass';
   GRANT ALL PRIVILEGES ON DATABASE afilc TO afilc_user;
   ```
4. Ejecuta el DDL y el seed, en este orden, contra la base `afilc`:
   ```powershell
   psql -h localhost -U afilc_user -d afilc -f sql\ddl\001_schema.sql
   psql -h localhost -U afilc_user -d afilc -f sql\seed\002_seed.sql
   ```
5. Ejecuta las consultas igual que en el paso 6 de la opción con Docker,
   usando `psql` o pgAdmin en lugar de la extensión de VS Code.

---

## Sobre el Dataset C (MongoDB)

`mongo/schema_interacciones.design.js` es documentación de diseño, **no**
código para ejecutar en este MVP (así se justifica en el apartado 5.2 del
trabajo). Contiene tres documentos de ejemplo (uno por modelo de ML) y el
esquema de validación `$jsonSchema` completo, listos para aplicarse cuando se
incorpore físicamente el contenedor MongoDB en el sprint posterior a la
construcción de la interfaz conversacional (apartado 7).
