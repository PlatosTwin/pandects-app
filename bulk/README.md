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
