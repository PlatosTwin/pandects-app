# ETL

## Purpose

`etl/` contains the Dagster jobs, assets, ML code, and data processing utilities used to ingest and enrich M&A agreements.

## What outside contributors can do here

- review and improve the code
- add tests and type-checking coverage
- improve docs and developer ergonomics
- make carefully scoped reviewed changes that do not require the maintainer's private data environment

This directory is partially public but operationally maintainer-gated.

## Required tools

- Python 3.11
- ETL virtualenv at `etl/.venv`
- Dagster tooling for local pipeline work

## Public-safe checks

Type check the ETL source:

```bash
caffeinate -i etl/.venv/bin/basedpyright etl/src
```

## Environment variables

See:

- `etl/.env.example`
- root `ENVIRONMENT.md`

Real ETL execution generally requires private MariaDB access and provider credentials such as `OPENAI_API_KEY` and `VOYAGE_API_KEY`.

## Maintainer-only dependencies and quirks

- many ETL assets assume the real main MariaDB dataset
- some utilities read local env files directly for operational workflows
- bulk backfills and repair flows are not part of normal outside-contributor onboarding
- running Dagster jobs against real data is maintainer-only

## Architecture notes

- canonical Dagster definitions live in `etl/src/etl/defs/jobs.py`
- `etl/src/etl/definitions.py` re-exports the canonical `defs`
- default run config lives in `etl/configs/pipeline_config.yaml`

## Related docs

- root [README.md](../README.md)
- root [ENVIRONMENT.md](../ENVIRONMENT.md)
- [backend/README.md](../backend/README.md)
