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

The public download and private production restore use separate table sets:

- `public_tables.txt` — public SQL dump tables, shared by `push_to_r2.sh` and the docs generator.
- `private_restore_tables.txt` — internal serving tables added only to the private logical backup used for production restore.
- `schema_docs/table_docs.yml` — hand-written table/column descriptions.
- `schema_docs/generate_schema_docs.py` — introspects the live MariaDB schema, merges the descriptions, and regenerates two checked-in artifacts: `schema_docs/pandects.dbml` (published to [dbdocs.io](https://dbdocs.io/nmbogdan/Pandects) by `.github/workflows/publish-dbdocs.yml`) and `docs/docs/guides/bulk-data-schema.md` (deployed with the docs site).

`push_to_r2.sh` runs the generator before every dump and aborts if a dumped table or column lacks a description — so a schema change cannot reach the public dump undocumented. If the generated files change, commit and push them; CI updates dbdocs.io and the docs site.

The promoted private manifest records both table sets and exact row counts for
every restored table. Before upload, `push_to_r2.sh` requires the private search
table to match the current searchable section set (including source-content
hashes), use the expected default InnoDB FULLTEXT settings, and have its required
FULLTEXT index. `restore_from_r2.py` verifies that every non-empty table has schema
and data artifacts before resetting production, then validates all row counts,
FULLTEXT settings, and source/hash coverage after loading.

Release order matters: build, converge, validate, and swap the local serving table;
publish and verify a manifest-v2 snapshot; deploy the matching restore image; restore
and verify production MariaDB; only then deploy backend and ETL code that reflects
the new serving table.

Pause section-writing ETL for the final convergence pass, integrity gate, and
logical dump. The manifest records pre-dump counts and the restore rechecks them,
so an intervening write fails closed, but quiescing avoids publishing an unusable
snapshot in the first place.

`MYLOADER_THREADS` defaults to `2`, sized for the current single-vCPU,
memory-constrained production VM. Override it only after benchmarking the
restore on the target machine.

`restore_prod.sh` is the one-shot production restore: it records the
`pandects-db` machine size, scales it to `performance-2x` / 4GB, runs
`restore_from_r2.py` on a temporary `pandects-bulk` machine (with
`MYLOADER_THREADS=4` for the larger VM), verifies row counts, the FULLTEXT
index, and a boolean-mode smoke query against the live database, and scales
`pandects-db` back to its recorded size on exit — on failure too. If the
restore is still running when the script is interrupted or times out, it
leaves the restore machine and the scaled-up DB alone and prints how to finish
by hand. Deploy the restore image first whenever the restore script changed:

```bash
cd bulk && fly deploy --app pandects-bulk
MARIADB_PASSWORD=... bash bulk/restore_prod.sh
```

Before resetting production, `restore_from_r2.py` probes `myloader --help` and passes `--optimize-keys AFTER_IMPORT_PER_TABLE`, `--innodb-optimize-keys`, or no key-optimization flag depending on what the installed build advertises; it aborts before the drop if myloader cannot run.

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
