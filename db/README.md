# DB

## Purpose

`db/` contains Fly deployment configuration for the main MariaDB service and related operational env expectations.

## What outside contributors can do here

- review deployment config
- improve docs
- make carefully scoped config changes with maintainer coordination
- propose narrowly scoped reviewed patches

Outside contributors should not expect to run or administer this service locally.

## Runtime status

This directory is maintainer-only for actual execution and deployment.

## Section text search migration

`section_text_search.sql` is the canonical serving-table definition. Build a
monthly replacement as an unindexed shadow first (omit the FULLTEXT key while
loading), run `etl.utils.section_text_search_backfill` with the shadow table name,
add the FULLTEXT key, and run the hash-aware `--drift-only` catch-up from the
beginning until a complete pass reports zero batches. Validate exact row,
agreement, version, and `source_xml_sha256` coverage before an atomic
`RENAME TABLE` swap. Keep the previous table temporarily for rollback.

After the local swap, publish a manifest-v2 R2 snapshot and follow the release
ordering documented in `bulk/README.md`. The production-only Postgres database
is not involved; section search is part of the main MariaDB serving snapshot.

## Environment variables

See:

- `db/.env.example`
- root `ENVIRONMENT.md`

## Maintainer-only dependencies and quirks

- private database credentials are required
- deployment assumes maintainer Fly access
- this directory is not part of the public-safe onboarding flow

## Related docs

- root [README.md](../README.md)
- root [ENVIRONMENT.md](../ENVIRONMENT.md)
