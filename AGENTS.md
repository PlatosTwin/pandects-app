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
- For backend Python changes, activate `backend/venv` and run targeted `unittest` modules from the repo root, for example: `python3 -m unittest backend.tests.test_auth -v`
- For ETL Python changes, use the ETL environment and run targeted tests or `basedpyright` for the touched area.
- For material frontend changes in `frontend/`, run `npm test` and `npm run typecheck`.

## Common Workflows
- Backend local run (from repo root): `FLASK_APP=backend.app flask run`
- ETL pipelines with repo config:
  - `dagster job execute -f etl/src/etl/defs/jobs.py -j cleanup_pipeline -c etl/configs/pipeline_config.yaml`
  - `dagster job execute -f etl/src/etl/defs/jobs.py -j xml_fresh_pipeline -c etl/configs/pipeline_config.yaml`
  - `dagster job execute -f etl/src/etl/defs/jobs.py -j xml_repair_cycle_pipeline -c etl/configs/pipeline_config.yaml`
- Frontend email template render/sync (from `frontend/`):
  - `npm run render:emails`
  - `npm run render:emails:sync`
- Docs app (from `docs/`):
  - `npm run dev:clean`
  - `npm run build`

## Cursor Rules
- Read `.cursor/rules/` before making substantial changes.
- Treat the Cursor rules as repo-specific supplements. If a rule duplicates or conflicts with higher-priority instructions, follow the higher-priority instruction and keep the repo-specific intent in mind.
