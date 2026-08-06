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

Any change that alters the public dataset or its meaning — `bulk/public_tables.txt`, `bulk/schema_docs/table_docs.yml`, DB migrations or data fixes affecting allowlisted tables, `backend/schemas/public_api.py`, or the `backend/mcp/` tool surface — must append an entry to `unreleased:` in `bulk/changelog/changelog.yml` in the same commit (CI enforces this for the schema-bearing paths). Entry template:

```yaml
- type: schema        # schema | data | api | mcp | docs | pipeline
  severity: notable   # breaking | notable | minor
  summary: "One-line, consumer-facing description."
  tables: [agreements]   # optional
  refs: ["<commit or PR>"]
```

See `bulk/changelog/DESIGN.md` for the full format and release flow.

## Security issues

Do not open a public issue for a suspected security problem. Follow [SECURITY.md](SECURITY.md).
