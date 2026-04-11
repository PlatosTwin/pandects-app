# Pandects

Pandects is an open-source M&A agreement research platform. This repository contains the public web application, Flask API, documentation site, ETL pipelines, email templates, and deployment helpers that power [pandects.org](https://pandects.org).

This repo is open to outside contributions, but not every subsystem is equally accessible. The intended contributor path is a public-safe local mode that does not require your own MariaDB snapshot or real credentials for Resend, Cloudflare R2, Zitadel, Fly, or other private infrastructure.

## Repo layout

- `frontend/`: Vite + React client, SSR/prerender server, shared frontend types
- `backend/`: Flask API, auth/session logic, MCP routes, OpenAPI source
- `docs/`: Docusaurus guides and API reference site
- `etl/`: Dagster pipelines and ML/data processing code
- `emails/`: existing React Email templates plus maintainer-only delivery tooling
- `bulk/`: maintainer-only bulk export and R2 sync scripts
- `db/`: maintainer-only Fly deployment config for the main MariaDB instance
- `branding/`: Shared brand links, tokens, and logo assets
- `examples/`: public notebooks and API usage examples
- `pg/`: maintainer notes for Fly Postgres auth DB access

## Supported contribution paths

### Fully contributor-friendly

- `frontend/`
- `docs/`
- most documentation changes at the repo root

### Contributor-friendly in public-safe local mode

- `backend/` when using `SKIP_MAIN_DB_REFLECTION=1`
- auth-related backend work that can use the default local sqlite auth DB
- tests, type checks, docs, and UI work that do not need live private integrations

### Partially public, operationally maintainer-gated

- `etl/`
- `bulk/`
- `db/`
- credential-backed integrations such as Resend, Cloudflare R2, Zitadel, Fly, and private MariaDB access

These areas are still open-source code, but many workflows assume private infrastructure that outside contributors will not have.

For `etl/`, `bulk/`, `db/`, `pg/`, and credential-backed parts of `emails/`, contributors should treat the code as readable and patchable, but not assume the operational commands are runnable without maintainer coordination.

## Prerequisites

- Python 3.11
- Node.js 24.x
- npm 10+
- `caffeinate` if you want to use the repo's preferred long-running command style on macOS

Optional maintainer tooling:

- Fly CLI
- WireGuard
- MariaDB client tools
- `mydumper` / `myloader`

## Public-safe local mode quickstart

This is the default onboarding path for outside contributors.

### 1. Clone and inspect environment examples

```bash
git clone https://github.com/PlatosTwin/pandects-app.git
cd pandects-app
```

Copy only the env files you actually need:

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local
```

You do not need real secrets for routine frontend, docs, or most backend test work.

### 2. Backend safe mode

```bash
python3 -m venv backend/venv
backend/venv/bin/python -m pip install -r backend/requirements.txt
SKIP_MAIN_DB_REFLECTION=1 backend/venv/bin/python -m unittest discover backend/tests -v
```

For local API work that should not touch the main MariaDB schema:

```bash
SKIP_MAIN_DB_REFLECTION=1 FLASK_APP=backend.app backend/venv/bin/flask run --port 5113
```

Notes:

- The backend loads `backend/.env` automatically.
- Auth data defaults to a local sqlite database unless you explicitly point it elsewhere.
- Features that require the real main MariaDB schema or third-party auth/email services are maintainer-only.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Default local frontend URL: `http://localhost:8080`

You can point the frontend at:

- a local backend via `VITE_API_BASE_URL=http://localhost:5113`
- the public API if you are doing purely UI work that does not need unsafe mutations or private auth flows

### 4. Docs

```bash
cd docs
npm install
npm run build
```

For local docs development:

```bash
cd docs
npm run dev:clean
```

Default local docs URL: `http://localhost:3001`

## Root helper commands

A small contributor-focused `Makefile` is provided:

- `make help`
- `make backend-test`
- `make dev-backend-safe`
- `make frontend-test`
- `make frontend-typecheck`
- `make dev-frontend`
- `make docs-build`
- `make etl-typecheck`

`dev-all.sh` still exists, but it is a maintainer-oriented full-stack script. It assumes a much richer local environment than most outside contributors will have.

## Environment files

Each subsystem that reads env vars now has a colocated example file where that makes sense:

- `backend/.env.example`
- `frontend/.env.example`
- `etl/.env.example`
- `bulk/.env.example`
- `emails/.env.example`
- `db/.env.example`

The high-level variable matrix lives in [ENVIRONMENT.md](ENVIRONMENT.md).

## Maintainer-only infrastructure

These systems are part of the production or private-maintainer setup and are not required for normal public contributions:

- main MariaDB data on the maintainer machine
- Resend template syncing and live email delivery
- Cloudflare R2 bulk dump publishing
- Zitadel-backed auth and MCP identity configuration
- Fly deployment and Fly-internal networking
- Fly Postgres / WireGuard access described in `pg/README.md`

The documentation should label these clearly instead of treating them as default onboarding steps.

## Contributing

Start with:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [SECURITY.md](SECURITY.md)
- [SUPPORT.md](SUPPORT.md)

When in doubt, prefer work that is fully reproducible in public-safe local mode and call out any maintainer-only assumptions in your PR.

## License

This project is licensed under the GNU GPLv3. See [LICENSE.md](LICENSE.md).
