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
