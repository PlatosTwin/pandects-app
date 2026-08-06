# Contributing to Pandects

Thanks for contributing. This repository is open source, but some workflows depend on private infrastructure that only the maintainer has access to. The goal for outside contributors is to work in the documented public-safe local mode unless a maintainer explicitly asks for something else.

## Good contribution targets

- frontend UX, accessibility, and tests
- docs, guides, API examples, and onboarding
- backend tests and backend work that can run with `SKIP_MAIN_DB_REFLECTION=1`
- refactors and bug fixes that do not require the private MariaDB dataset or real third-party credentials

## Maintainer-gated areas

The following are open-source code, but many operational workflows are not reproducible for outside contributors:

- live main MariaDB access
- Cloudflare R2 publishing
- Resend template delivery
- Zitadel configuration and real auth credentials
- Fly deployment and Fly-private networking
- some ETL and bulk sync operations

If your change touches one of these areas, keep the change scoped, document assumptions clearly, and avoid representing private infrastructure as part of the normal contributor setup.

## Setup

Use the repo root [README.md](README.md) for the main onboarding flow. The short version:

1. Use Python 3.11 and Node.js 24.
2. Copy only the `.env.example` files you need.
3. Prefer the public-safe local workflows:
   - `make backend-test`
   - `make dev-backend-safe`
   - `make frontend-test`
   - `make frontend-typecheck`
   - `make docs-build`

## Code knowledge graph

`graphify-out/` is a committed map of this repo's symbols and their relationships. It is optional, but it is the fastest way to orient yourself in an unfamiliar subsystem — it resolves calls, imports, and inheritance across directories that a text search will miss.

```bash
pip install graphifyy
graphify explain "McpToolSpec"               # a symbol, its file:line, and every edge
graphify affected "PipelineConfig" --depth 2 # what depends on this — run before refactoring shared code
```

`graphify-out/GRAPH_REPORT.md` is the readable summary. Two caveats: edges are tagged `EXTRACTED` (explicit in source) or `INFERRED` (model-reasoned), so confirm inferred ones against the code, and the graph describes structure rather than runtime behavior.

`graphify-out/` is generated — never hand-edit it, and leave it out of your PR. CI refreshes it on `main` once your change merges. If you want your local copy to stay current as you commit, run `graphify hook install` once; that also registers the merge driver that keeps `graph.json` from producing conflict markers.

## Branches and pull requests

1. Fork the repository.
2. Create a focused branch.
3. Keep the change tight and explain any product or operational assumptions.
4. Run the narrowest relevant checks before opening the PR.
5. Open a pull request against `main`.

Useful PR content:

- what changed
- why it changed
- how you tested it
- whether the change is public-safe or maintainer-gated

## Validation expectations

Run the narrowest relevant checks for what you touched.

- Backend: `caffeinate -i backend/venv/bin/python -m unittest discover backend/tests -v`
- Frontend: `caffeinate -i npm test` and `caffeinate -i npm run typecheck` from `frontend/`
- Docs: `caffeinate -i npm run build` from `docs/`
- ETL: use the documented `basedpyright` or targeted ETL test commands when the change is genuinely ETL-scoped

If a workflow requires private data or credentials you do not have, say that explicitly in the PR rather than guessing.

## Documentation expectations

Please update docs when behavior, setup, or env vars change. In this repo that usually means one or more of:

- root `README.md`
- subsystem `README.md`
- `.env.example` files
- `ENVIRONMENT.md`

Do not add contributor-facing documentation under git-ignored paths.

### Dataset changelog

Public dataset, API, and MCP changes are announced to consumers through a per-release changelog (design and rationale: `bulk/changelog/DESIGN.md`).

**When an entry is required.** Your change alters the public dataset or its meaning if it touches any of:

- `bulk/public_tables.txt` (tables entering or leaving the public dump)
- `bulk/schema_docs/table_docs.yml` or the schema of any allowlisted table
- data in allowlisted tables (migrations, backfills, value fixes)
- `backend/schemas/public_api.py` or REST route behavior consumers rely on
- the `backend/mcp/` tool surface (tools added/removed, arguments, output shapes)

If so, append an entry under `unreleased:` in `bulk/changelog/changelog.yml` **in the same commit**.

**Entry template:**

```yaml
- type: schema        # schema | data | api | mcp | docs | pipeline
  severity: notable   # see below
  summary: "One-line, consumer-facing description."
  details: >           # optional longer prose
  tables: [agreements] # optional; must appear in bulk/public_tables.txt
  migration: null      # optional guidance/SQL for consumers adapting to the change
  refs: ["<commit or PR>"]
```

**Severity semantics** (pick the worst that applies):

- `breaking` — a consumer query that worked against the previous dump can return wrong or empty results (column rename/drop, semantic change of values)
- `notable` — results change but existing queries still run (data corrections, new defaults)
- `minor` — additive or cosmetic

Write the summary for a data consumer, not a code reviewer: say what changed in the published data or API, not which function was edited.

**Before you commit — checklist:**

1. Append the entry under `unreleased:` (top of `bulk/changelog/changelog.yml`).
2. Regenerate the rendered files and commit them together with the yml:

   ```bash
   python3 bulk/changelog/render_changelog.py render
   ```

   This needs only PyYAML (`pip install pyyaml`), no database access, and rewrites `bulk/changelog/CHANGELOG.md` and `docs/docs/guides/changelog.md`.
3. Hand-edit **only** `unreleased:`. Never edit `releases:` (stamped by the maintainer's release script), `bulk/changelog/CHANGELOG.md`, or `docs/docs/guides/changelog.md` (both generated).

**How this is enforced.** CI fails the PR if a schema/API-surface path changes without a `bulk/changelog/changelog.yml` edit (`.github/workflows/changelog-guard.yml`), and a backend test (`backend/tests/test_changelog.py`) fails if the rendered files don't match the yml — that failure means you skipped step 2. At the next bulk release the maintainer's push script rolls `unreleased:` into a stamped release and publishes it as `dumps/changelog.json`, `GET /v1/changelog`, and the docs-site changelog page.

## Security issues

Do not open a public issue for a suspected security problem. Follow [SECURITY.md](SECURITY.md).
