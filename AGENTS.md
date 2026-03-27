# Project Instructions

## Instruction Priority
- Follow instructions in this order:
1. System/developer/runtime instructions
2. This root `AGENTS.md` (repo-wide)
3. Directory-local `AGENTS.md` files (for files in that subtree, e.g. `frontend/AGENTS.md`)
4. `.cursor/rules/*.mdc` rules
- If two instructions conflict at the same level, prefer the more specific and actionable one.

## Working Scope
- Default to maintained source and configuration files: `backend/`, `frontend/`, `etl/`, repo docs/source files, and tracked instruction files.
- Treat generated or derived artifacts as out of scope unless the task is explicitly about regenerating them. This includes build outputs, generated API docs, generated OpenAPI copies, and similar checked-in derivatives.
- Do not edit notebooks, local scratch files, or large ETL backup/backfill datasets unless the user asks for that exact work.

## Code Style
- If you're unsure about a function or schema, clarify your uncertainty instead of coding in a fallback to handle missing values or unknown types. Fallbacks should be unnecessary in most cases.
- If asked to fix a bug, don't just produce a band-aid solution; understand the root cause and fix it.
- Code with an eye toward elegance.
- Remove stale comments and dead branches when they are truly obsolete.
- Add comments sparingly and only when they explain non-obvious intent, invariants, or operational constraints.

## Validation
- Use the narrowest relevant validation for the files you touch.
- Do not use `uv` for validation in this repo unless the user explicitly asks for it.
- For backend Python changes, run targeted `unittest` modules from the repo root with the repo's backend interpreter, for example: `caffeinate -i backend/venv/bin/python -m unittest backend.tests.test_auth -v`
- For ETL Python changes, run targeted tests or `basedpyright` for the touched area with the ETL environment, for example: `caffeinate -i etl/.venv/bin/python -m unittest etl.tests.test_resources -v` or `caffeinate -i etl/.venv/bin/basedpyright etl/src/etl/defs/i_tx_metadata_asset.py`
- For material frontend changes in `frontend/`, run `npm test` and `npm run typecheck`.

## Shell and Database Access
- When you need live database access, do not stop at "the sandbox cannot access the DB". Request elevated shell access immediately and run the query yourself.
- Main application SQL is MariaDB SQL. Auth data is a separate database bind and may be SQLite locally or Postgres in production; do not assume the auth DB uses MariaDB.
- Prefer running shell commands with `caffeinate -i`, including tests, one-off scripts, and DB clients.
- Prefer direct virtualenv/tool paths over activation hand-waving when you can name the exact binary, for example `backend/venv/bin/python`, `etl/.venv/bin/python`, or `etl/.venv/bin/basedpyright`.
- If backend imports fail because main-db reflection is not needed for the task, set `SKIP_MAIN_DB_REFLECTION=1` for that command instead of treating import-time reflection as a blocker.
- If a task depends on schema inspection or SQL verification, do the inspection directly instead of guessing. Include the exact command you ran in your notes to the user when it matters.
- Safe MariaDB command patterns to prefer:
  - `caffeinate -i mysql -h "$MARIADB_HOST" -u "$MARIADB_USER" -p"$MARIADB_PASSWORD" "$MARIADB_DATABASE"`
  - `caffeinate -i mysql -h "$MARIADB_HOST" -u "$MARIADB_USER" -p"$MARIADB_PASSWORD" "$MARIADB_DATABASE" -e "SHOW TABLES LIKE 'agreements';"`
  - `caffeinate -i mysql -h "$MARIADB_HOST" -u "$MARIADB_USER" -p"$MARIADB_PASSWORD" "$MARIADB_DATABASE" -e "DESCRIBE agreements;"`
- Required main DB env vars are `MARIADB_USER`, `MARIADB_PASSWORD`, `MARIADB_HOST`, and `MARIADB_DATABASE`. `backend/.env.example` shows the expected names. If those vars are missing locally, say that clearly instead of claiming DB access is impossible in principle.

## Common Workflows
- Full local dev stack (from repo root): `./dev-all.sh`
- Backend local run (from repo root): `FLASK_APP=backend.app flask run`
- ETL pipelines with repo config:
  - `dagster job execute -f etl/src/etl/defs/jobs.py -j cleanup_pipeline -c etl/configs/pipeline_config.yaml`
  - `dagster job execute -f etl/src/etl/defs/jobs.py -j xml_fresh_pipeline -c etl/configs/pipeline_config.yaml`
  - `dagster job execute -f etl/src/etl/defs/jobs.py -j xml_repair_cycle_pipeline -c etl/configs/pipeline_config.yaml`
- ETL local Launchpad/dev server (from `etl/`): `dg dev`
- Frontend email template render/sync (from `frontend/`):
  - `npm run render:emails`
  - `npm run render:emails:sync`
- Docs app (from `docs/`):
  - `npm run dev:clean`
  - `npm run build`
  - `npm run clean-api && npm run gen-api`

## Cursor Rules
- Read `.cursor/rules/` before making substantial changes.
- Treat the Cursor rules as repo-specific supplements. If a rule duplicates or conflicts with higher-priority instructions, follow the higher-priority instruction and keep the repo-specific intent in mind.
