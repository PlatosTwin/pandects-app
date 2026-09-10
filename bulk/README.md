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
by hand. Expect it to run for hours: the 2026-09 restore took 4h13m, most
of it building the `section_text_search` FULLTEXT index during load,
because the image's mydumper 0.10.0 cannot defer key creation. Deploy the
restore image first whenever the restore script changed:

```bash
cd bulk && fly deploy --app pandects-bulk
MARIADB_PASSWORD=... bash bulk/restore_prod.sh
```

Before resetting production, `restore_from_r2.py` probes `myloader --help` and passes `--optimize-keys AFTER_IMPORT_PER_TABLE`, `--innodb-optimize-keys`, or no key-optimization flag depending on what the installed build advertises; it aborts before the drop if myloader cannot run.

To run it standalone:

```bash
bulk/.venv/bin/python3 bulk/schema_docs/generate_schema_docs.py
```

## Changelog

Dataset, schema, and API changes are documented per dump release (design:
`changelog/DESIGN.md`):

- `changelog/changelog.yml` — source of truth. Any change that alters the
  public dataset or its meaning must append an entry under `unreleased:` in
  the same commit. Only `unreleased:` is hand-edited; `releases:` is stamped
  by `push_to_r2.sh`.
- `changelog/render_changelog.py` — validates the yml and regenerates
  `changelog/CHANGELOG.md` and `docs/docs/guides/changelog.md` (run
  `... render` standalone; `push_to_r2.sh` runs it at release time).

`push_to_r2.sh` gates every dump: it aborts if the schema fingerprint changed
since the last release without a `schema` entry (same philosophy as the
docs-coverage gate), and warns for confirmation when row counts move
anomalously without a `data` entry. After the dump artifacts upload
successfully it rolls `unreleased:` into a stamped release and publishes
`dumps/changelog.json` next to the dump (also served by `GET /v1/changelog`
and summarized in the MCP `get_server_capabilities` changelog section).
Commit the rewritten changelog files after each push — the next push diffs
against the committed state.

Failure recovery:

- Push fails **before** step 3b (dump or upload failed): the repo is
  untouched; fix the cause and re-run the whole script.
- Push fails **at step 3c** (roll-up done, changelog upload failed): do not
  revert the repo — the release is already stamped for a published dump.
  Re-render and upload by hand:

  ```bash
  bulk/.venv/bin/python3 bulk/changelog/render_changelog.py render --json-out /tmp/changelog.json
  # then upload /tmp/changelog.json to dumps/changelog_<ts>.json and copy it
  # over dumps/changelog.json (both public-read). <ts> is the dump timestamp —
  # it's recorded as "changelog_key" in dumps/latest.json.
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
