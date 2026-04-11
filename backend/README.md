# Backend

## Purpose

`backend/` contains the Flask API, auth/session logic, MCP routes, OpenAPI generation, and the backend-side schemas and services used by Pandects.

## What outside contributors can do here

- run backend unit tests in public-safe mode
- work on route contracts, schemas, auth flows that use the local sqlite auth DB, and documentation
- improve error handling, tests, and internal refactors that do not require the live main MariaDB schema

## Required tools

- Python 3.11
- a local virtualenv at `backend/venv`

## Local setup

```bash
python3 -m venv backend/venv
backend/venv/bin/python -m pip install -r backend/requirements.txt
cp backend/.env.example backend/.env
```

## Public-safe local commands

Run tests:

```bash
caffeinate -i env SKIP_MAIN_DB_REFLECTION=1 backend/venv/bin/python -m unittest discover backend/tests -v
```

Run the backend without touching the main MariaDB reflection path:

```bash
caffeinate -i env SKIP_MAIN_DB_REFLECTION=1 FLASK_APP=backend.app backend/venv/bin/flask run --port 5113
```

Generate the OpenAPI file when needed:

```bash
caffeinate -i env SKIP_MAIN_DB_REFLECTION=1 backend/venv/bin/python -m backend.app openapi
```

## Environment variables

See:

- `backend/.env.example`
- root `ENVIRONMENT.md`

Most outside contributors only need:

- `PUBLIC_API_BASE_URL`
- `PUBLIC_FRONTEND_BASE_URL`
- `SKIP_MAIN_DB_REFLECTION=1`
- optionally `AUTH_SESSION_TRANSPORT=bearer`

The auth DB defaults to local sqlite when `AUTH_DATABASE_URI` is not set.

## Maintainer-only dependencies and quirks

- The main application data lives in a MariaDB database that outside contributors will not have.
- Real OAuth, Resend, Turnstile, MCP identity, and Zitadel integration require private credentials.
- Live main DB inspection is maintainer-only and intentionally not part of the public contributor workflow.

## Related docs

- root [README.md](../README.md)
- root [ENVIRONMENT.md](../ENVIRONMENT.md)
- [pg/README.md](../pg/README.md) for auth Postgres maintainer notes
