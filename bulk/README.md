# Bulk

## Purpose

`bulk/` contains maintainer-oriented scripts for creating database export artifacts and publishing or restoring bulk data snapshots.

## What outside contributors can do here

- inspect the code
- improve documentation and safety checks
- review scripts for clarity or portability
- propose narrowly scoped reviewed patches

Outside contributors should not assume they can run anything in this directory.

## Runtime status

This directory is maintainer-only for actual execution.

## Public schema docs

The tables in the public dump are documented automatically:

- `public_tables.txt` — the dump allowlist, shared by `push_to_r2.sh` and the docs generator.
- `schema_docs/table_docs.yml` — hand-written table/column descriptions.
- `schema_docs/generate_schema_docs.py` — introspects the live MariaDB schema, merges the descriptions, and regenerates two checked-in artifacts: `schema_docs/pandects.dbml` (published to [dbdocs.io](https://dbdocs.io/nmbogdan/Pandects) by `.github/workflows/publish-dbdocs.yml`) and `docs/docs/guides/bulk-data-schema.md` (deployed with the docs site).

`push_to_r2.sh` runs the generator before every dump and aborts if a dumped table or column lacks a description — so a schema change cannot reach the public dump undocumented. If the generated files change, commit and push them; CI updates dbdocs.io and the docs site.

To run it standalone:

```bash
bulk/.venv/bin/python3 bulk/schema_docs/generate_schema_docs.py
```

## Environment variables

See:

- `bulk/.env.example`
- root `ENVIRONMENT.md`

This directory expects private MariaDB and Cloudflare R2 credentials for real use.

## Maintainer-only dependencies and quirks

- requires access to the source MariaDB data
- requires R2 credentials for upload
- publishes public artifacts and should be treated as an operational workflow, not a casual contributor entrypoint

## Related docs

- root [README.md](../README.md)
- root [ENVIRONMENT.md](../ENVIRONMENT.md)
