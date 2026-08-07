# Changelog feature — design

Status: implemented (phases 1–2; phase 3 deferred). This doc specifies a dataset/schema changelog that ships
alongside the monthly bulk DB exports, is reachable via the API and MCP, and is wired into
every LLM-instruction surface so that big changes to schema, data, or the API/MCP surface
get documented as a matter of course.

## 1. Problem

The monthly `bulk/push_to_r2.sh` push publishes a new public dump, but nothing records *what
changed* between dumps. Consumers (humans re-importing `latest.sql.gz`, agents using the API
or MCP) have no way to learn that a column was renamed, 135 SPAC acquirer rows were
corrected, or a table was added — except by diffing two multi-GB dumps. The schema-docs gate
in `push_to_r2.sh` guarantees every column is *described*, but not that changes are
*announced*.

Motivating example: the 2026-08-05 SPAC acquirer=Pubco data fix (135 rows) will silently
change query results for anyone comparing acquirer names across dump versions. That is
exactly the entry a changelog should carry.

## 2. What exists today (integration points)

| Surface | Where | Relevance |
|---|---|---|
| Bulk push pipeline | `bulk/push_to_r2.sh` | Runs schema-docs gate (step 0), dumps, uploads to R2 `pandects-bulk`, writes per-dump `*.manifest.json` + `dumps/latest.json` pointers |
| Schema docs | `bulk/schema_docs/generate_schema_docs.py` → `pandects.dbml`, `docs/docs/guides/bulk-data-schema.md` | Already regenerated before every dump; its output is a ready-made schema fingerprint |
| Dump provenance | `X-Pandects-Dump-Hash` header + `dump_version` body field on API responses (see `docs/docs/guides/getting-started.md`) | Every API response is already keyed to a dump sha256 — the changelog should be keyed the same way |
| Dumps API | `GET /v1/dumps` in `backend/routes/reference_data.py` (`DumpEntrySchema`, R2-listing + TTL cache in `backend/routes/deps.py`) | Pattern to reuse for serving the changelog |
| MCP | `_server_capabilities_payload()` in `backend/mcp/tools/__init__.py:1113`; `initialize` instructions string in `backend/mcp/routes/__init__.py:178` | Where agents learn what this server offers |
| LLM docs | `docs/scripts/generate-llms-docs.js` → `llms.txt`, `llms-full.txt`, per-guide/per-endpoint md (guides are an explicit list, `GUIDES`) | Changelog guide must be added to the list |
| Contributor/agent instructions | `CONTRIBUTING.md`, `bulk/README.md`, local (gitignored) `CLAUDE.md` | Where the "write a changelog entry" rule must live |

## 3. Design overview

One committed, machine-readable source of truth; everything else is rendered or served from it.

```
bulk/changelog/changelog.yml        (committed; humans/agents append "unreleased" entries)
        │
        ├─ bulk/changelog/render_changelog.py   (validate + render; run by push_to_r2.sh step 0b)
        │       ├─ bulk/changelog/CHANGELOG.md          (committed; human-readable, GitHub-browsable)
        │       ├─ docs/docs/guides/changelog.md        (committed; docs site + llms.txt)
        │       └─ /tmp/.../changelog.json              (uploaded to R2, not committed)
        │
        ├─ R2: dumps/changelog.json  +  dumps/changelog_<ts>.json   (public, next to the dumps)
        │       └─ manifest.json / latest.json gain "changelog_url"
        │
        ├─ API: GET /v1/changelog  (thin cached reader of R2 changelog.json; /v1/dumps gains changelog_url)
        │
        └─ MCP: "changelog" section in get_server_capabilities + mention in initialize instructions
```

Release semantics follow Keep-a-Changelog adapted to data releases: entries accumulate under
`unreleased:` between pushes; at push time the script rolls them into a release stamped with
the dump timestamp, dump sha256, and auto-computed stats.

## 4. Source of truth: `bulk/changelog/changelog.yml`

```yaml
# Hand-edited section. Append an entry here in the same PR/commit as the change it describes.
unreleased:
  - type: data          # schema | data | api | mcp | docs | pipeline
    severity: notable   # breaking | notable | minor
    summary: "Fixed SPAC acquirer misattribution: 135 agreements had Pubco recorded as acquirer"
    details: >
      Offline acquirer extraction now disambiguates the SPAC counterparty from the
      surviving Pubco entity. Affects acquirer_name on 135 xar_agreements rows.
    tables: [xar_agreements]
    migration: null      # optional: guidance/SQL for consumers adapting to the change
    refs: ["0da4871"]    # commits, PR numbers, issue links

# Machine-managed section. push_to_r2.sh appends here; do not hand-edit.
releases:
  - version: "2026-09-01"
    released: "2026-09-01T04:12:33Z"
    dump_sha256: "3a7f9b2c…"
    dump_key: "dumps/public_2026-09-01_04-12-33.sql.gz"
    schema_fingerprint: "sha256 of bulk/schema_docs/pandects.dbml at release time"
    stats:
      row_counts: {xar_agreements: 5183, sections: 812445}   # per allowlisted table
    changes:
      - …entries rolled in from unreleased…
```

Rules:

- **Humans/agents only ever touch `unreleased`.** Everything under `releases` is stamped by
  the push script, so authorship stays trivial ("append one YAML mapping") and the stamped
  fields can't drift from the artifacts.
- `type` scope is deliberately wider than the dump: `api`/`mcp` entries let API-surface
  changes (new endpoint, renamed field, new MCP tool) ride the same log, since the same
  agents consume both. `version` for those is still the dump release they ship inside —
  acceptable coarseness; the `refs` field carries the precise commit.
- `severity: breaking` means a consumer query that worked against the previous dump can
  return wrong/empty results (column rename/drop, semantic change of values). `notable` =
  results change but queries still run (the SPAC fix). `minor` = additive/cosmetic.
- Validation (in `render_changelog.py`, also exercised by a backend unit test): known keys
  only, enum membership, non-empty summary, `tables` ⊆ `bulk/public_tables.txt`,
  releases sorted newest-first, `dump_sha256` unique.

## 5. Release flow: `push_to_r2.sh` changes

New **step 0b**, right after `generate_schema_docs.py` (reusing its fresh `pandects.dbml`):

1. Validate `changelog.yml`.
2. Compute `schema_fingerprint = sha256(pandects.dbml)` and compare with the newest
   release's fingerprint.
3. **Gate:** if the fingerprint changed and `unreleased` contains no `type: schema` entry →
   abort with the same style of message as the existing docs-coverage gate ("schema changed
   vs last release; describe it in bulk/changelog/changelog.yml"). This is the enforcement
   backstop that doesn't depend on anyone remembering instructions.
4. If `unreleased` is empty and the schema is unchanged, do **not** block: auto-generate a
   minimal entry `{type: data, severity: minor, summary: "Routine data refresh"}`. Routine
   monthly pushes must never be blocked on ceremony; the stats block still shows growth.
5. **Soft data-change check (warn; hard-fails only when stdin is not a tty):** compute per-table row counts (one
   `SELECT COUNT(*)` per allowlisted table; reused later for the release's `stats` block)
   and compare against the newest release's `stats.row_counts`. If any table's delta is
   anomalous versus that table's *typical* per-release delta (median of the deltas across
   prior releases, falling back to a flat percentage threshold until enough history
   accumulates) and `unreleased` contains no `type: data` entry, print a prominent warning
   listing the tables and deltas, and require interactive confirmation to proceed. This
   exists because data-only fixes are applied via SQL directly to the DB and touch no repo
   files — the CI path guard (§9 layer 2) never fires and the schema fingerprint is
   unchanged, so without this check they'd rely on instructions alone. Known residual gap:
   same-cardinality value rewrites (e.g. the SPAC acquirer fix, 135 rows updated in place)
   don't move row counts and remain covered only by layer 1; full data diffing was
   considered and rejected as too heavy against multi-GB dumps.

After the dump artifacts (dump, checksums, manifests, `latest.*` pointers) have uploaded
**successfully** — never before, so a failed upload cannot leave the repo recording a
phantom release:

6. Roll `unreleased` into a new `releases[0]` stamped with `version` (dump date; same-day
   repushes get a `.2`/`.3` suffix so versions stay unique and `?since=` filtering never
   hides a release), `released`, `dump_sha256`, `dump_key`, `schema_fingerprint`, and the
   per-table `row_counts` computed in step 5.
7. Render `CHANGELOG.md`, `docs/docs/guides/changelog.md`, and `changelog.json`.
8. Upload `changelog.json` twice: versioned `dumps/changelog_<ts>.json` and the pointer
   `dumps/changelog.json` (public-read, same `latest.*` pattern as dumps).
9. The per-dump manifest and `dumps/latest.json` carry
   `"changelog_url": "https://bulk.pandects.org/dumps/changelog.json"` and
   `"changelog_key": "dumps/changelog_<ts>.json"` (the key is deterministic, so the
   manifest can reference it even though the changelog uploads afterwards).
10. Extend the existing end-of-run "schema docs changed, commit and push" warning to also
   cover `bulk/changelog/changelog.yml`, `bulk/changelog/CHANGELOG.md`, and
   `docs/docs/guides/changelog.md` — the rolled-up yml must land on main so the next push
   diffs against it.

Ordering note: the roll-up mutates committed files mid-push (the existing model for
`pandects.dbml`), but only after the dump is published, so every stamped release describes
a real artifact. If the changelog upload itself fails after the roll-up, re-render and
upload by hand rather than reverting the repo (recovery steps in `bulk/README.md`).

## 6. Distribution

### 6.1 Static artifact on R2 (canonical)

`https://bulk.pandects.org/dumps/changelog.json` is the canonical machine-readable
changelog: it lives next to the dumps it describes, survives backend outages, is
CDN-cacheable, and needs zero backend code. Anyone who can download `latest.sql.gz` can
download the changelog with the same tooling. This alone satisfies "sits alongside the bulk
db exports."

### 6.2 API: yes — but thin, and reading from R2

Best-practice reasoning: the backend deploys continuously from main, while prod data
refreshes ~monthly. If `/v1/changelog` served the *repo's* `changelog.yml`, it could
describe releases newer than the dump prod is actually serving. Serving the *published R2*
`changelog.json` keeps the endpoint truthful relative to published artifacts, and the
`released`/`dump_sha256` fields let clients align it with the `X-Pandects-Dump-Hash` they
receive on every response.

- `GET /v1/changelog` — new blueprint in `backend/routes/reference_data.py`, cloning the
  `/v1/dumps` pattern: fetch `dumps/changelog.json` from R2, TTL cache in
  `backend/routes/deps.py` (same `_dumps_cache` idiom), return as JSON. Query params:
  - `since=<version>` — releases newer than a version (agents catching up),
  - `dump_sha256=<hash>` — the single release matching a provenance hash (closes the loop
    with the response header).
- `GET /v1/dumps` — add `changelog_url` to `DumpEntrySchema`
  (`backend/schemas/public_api.py:342`) populated from the manifest.
- New marshmallow schemas (`ChangelogReleaseSchema`, `ChangelogChangeSchema`) so the
  endpoint lands in `openapi.yaml` → the per-endpoint llms md
  (`/llms/pandects/getChangelog.md`) is generated for free by `generate-llms-docs.js`.

Deploy caveat (see prod-serving-schema landmine): the endpoint depends only on R2, not on
DB tables, so it cannot hit the missing-table class of prod 500s — but still verify the
live route after deploy.

### 6.3 Frontend (optional, later)

The page that lists dumps via `/v1/dumps` can link `changelog_url` per dump. No custom UI
needed initially; the docs-site changelog page is the human surface.

## 7. MCP integration

1. **`get_server_capabilities`**: add a `changelog` section to
   `_server_capabilities_payload()` (`backend/mcp/tools/__init__.py:1113`) and to
   `_CAPABILITIES_SECTIONS_ALL`/defaults + output schema
   (`backend/mcp/tools/output_schemas.py:1206`):

   ```json
   "changelog": {
     "latest_version": "2026-09-01",
     "latest_released": "2026-09-01T04:12:33Z",
     "breaking_changes_in_latest": false,
     "url": "https://bulk.pandects.org/dumps/changelog.json",
     "api_route": "/v1/changelog",
     "note": "Dataset and schema change history. Correlate a response's X-Pandects-Dump-Hash / dump_version.hash with releases[].dump_sha256 to know exactly which changes your data includes."
   }
   ```

   Populated from the same cached R2 read as `/v1/changelog` (graceful `null`s if R2 is
   unreachable — capabilities must never 500).

2. **`initialize` instructions string** (`backend/mcp/routes/__init__.py:178`): append one
   sentence: *"Dataset and schema changes are documented in the changelog (see the
   changelog section of get_server_capabilities); check it when results seem to have
   changed across sessions."*

3. **Phase 3 (optional): `get_changelog` tool.** MCP-only clients can't always fetch
   arbitrary URLs, so a tool wrapping the cached R2 read (args: `since`, `dump_sha256`,
   `limit`) has real value. Deferred because the tool count is already large and the
   capabilities section covers discovery; promote it if `submit_feedback` reports agents
   asking for it.

## 8. Docs site + llms.txt

- `docs/docs/guides/changelog.md` — **generated** by `render_changelog.py` (like
  `bulk-data-schema.md` is generated by the schema-docs generator), committed, with
  frontmatter (`title: Changelog`, `description: Dataset, schema, and API change history
  per bulk release.`).
- Add `readGuide("guides/changelog.md")` to the `GUIDES` list in
  `docs/scripts/generate-llms-docs.js` — this puts it into `llms.txt`, `llms-full.txt`,
  and `/llms/guides/`.
- `docs/docs/guides/getting-started.md` — extend the existing "Data provenance" section
  with two sentences linking the changelog and explaining the
  `dump_sha256 ↔ X-Pandects-Dump-Hash` correlation.

## 9. Making sure entries actually get written (LLM-instruction + enforcement layers)

Instructions get changes documented only if backed by gates. Three layers, outermost is hard:

**Layer 1 — instructions (all agent surfaces):**

- Local `CLAUDE.md` (gitignored — edit locally, never commit): add a standing rule:
  *"Any change that alters the public dataset or its meaning — `bulk/public_tables.txt`,
  `bulk/schema_docs/table_docs.yml`, DB migrations affecting allowlisted tables, bulk data
  fixes, `backend/schemas/public_api.py`, `backend/mcp/tools/` surface — must append an
  entry to `unreleased:` in `bulk/changelog/changelog.yml` in the same commit."*
- `CONTRIBUTING.md`: same rule, phrased for contributors, with a copy-pasteable entry
  template.
- `bulk/README.md`: new "Changelog" section documenting the workflow and the push gate
  (mirrors the existing "Public schema docs" section).
- A short comment header inside `changelog.yml` itself (agents editing nearby files see it).

**Layer 2 — CI path guard (soft gate):**

GitHub Actions job on push/PR, triggered by `paths:` matching the schema/data/API-surface
files above, failing when the diff does not also touch `bulk/changelog/changelog.yml`.
There is no branch protection and etl/** has no CI, so this is advisory red-X — but the
existing push-until-green workflow (`/push` skill iterates on Actions failures) makes a red
check effective in practice. Also add a plain pytest (`backend/tests/test_changelog.py`)
validating yml well-formedness on every CI run, in the spirit of
`backend/tests/test_r2_allowlist.py`.

**Layer 3 — push-time gates:** the hard schema-fingerprint check plus the soft row-count
check, both in §5. Even if layers 1–2 are bypassed, an undocumented *schema* change cannot
reach the public dump (hard abort — identical philosophy to the existing docs-coverage
gate), and an undocumented *data* change that moves row counts anomalously gets flagged at
the terminal for confirmation. Layer 3 matters most for data fixes applied via SQL directly
to the DB, which touch no repo files and therefore never trigger layer 2. The one class
nothing mechanical catches is a same-cardinality value rewrite; that remains layer 1's job.

## 10. Alternatives considered

- **Derive the changelog from git log.** Rejected: consumer-meaningful ≠ commit-meaningful;
  data-only fixes applied via SQL never appear in git; requires an LLM summarization step
  at push time that produces unreviewed prose.
- **Store the changelog in a DB table.** Rejected: the changelog describes the DB *across*
  versions; storing it inside the artifact it describes is circular, and rebuild-style ETL
  (cf. the taxonomy PK incident) makes in-DB state fragile.
- **Serve `/v1/changelog` from the committed yml.** Rejected: code deploys are decoupled
  from data pushes; the API would describe unpublished releases (§6.2).
- **GitHub Releases.** Rejected: dumps aren't git artifacts; audience includes agents that
  only see the API/MCP/R2.

## 11. Implementation checklist

**Phase 1 — ship with the next monthly push (the September push documents the SPAC fix):**
1. `bulk/changelog/changelog.yml` seeded with the SPAC-fix entry under `unreleased` and one
   retroactive release entry for the currently published dump (fields backfilled from
   `dumps/latest.json`).
2. `bulk/changelog/render_changelog.py` (validate, roll-up, render md/json).
3. `push_to_r2.sh` step 0b gates (hard schema-fingerprint check + soft row-count check) +
   post-dump roll-up/upload/manifest fields (§5).
4. `bulk/README.md` + `CONTRIBUTING.md` + local `CLAUDE.md` rule text (§9 layer 1).
5. `backend/tests/test_changelog.py` yml validation test.

**Phase 2 — serving surfaces (independent deploy, any time after phase 1's first push):**
6. `/v1/changelog` route + `changelog_url` on `/v1/dumps` + schemas/OpenAPI (§6.2).
7. MCP capabilities `changelog` section + instructions-string sentence (§7.1–7.2).
8. `docs/docs/guides/changelog.md` generation + `GUIDES` list entry + getting-started
   provenance paragraph (§8).
9. CI path-guard workflow (§9 layer 2).

**Phase 3 — optional:**
10. `get_changelog` MCP tool.
11. Auto-generated schema-diff summaries (parse dbml diff into draft entries).
12. Frontend dump-page changelog links.
