---
id: technical-details
title: Technical Details
description: High-level technical reference for The Pandects MCP server and the tool surface it exposes.
sidebar_position: 3
---

# Technical Details

The Pandects MCP is a remote MCP server at `https://api.pandects.org/mcp`. All corpus tools are read-only; the single write surface is `submit_feedback`, which stores agent experience feedback for the maintainer.

It is designed for human users working through LLM clients, but this page describes the technical surface available behind the scenes.

## High-Level Capability Areas

The current MCP surface covers:

- Agreement discovery and retrieval
- Section search and section listing within one agreement
- Section-level retrieval
- Concept-to-taxonomy mapping and focused snippet extraction
- Tax clause retrieval
- Filter and taxonomy lookup
- Counsel, NAICS, summary, trend, and server-introspection reference data
- An agent feedback channel (`submit_feedback`)

## Tool Groups

### Agreement Discovery And Retrieval

- `search_agreements`
- `list_agreements`
- `get_agreement`
- `get_agreements_batch`

Use these when the client needs to find the right agreement first or fetch one or more agreements directly. `get_agreements_batch` fetches metadata for up to 25 known agreement UUIDs in one call, closing the N× `get_agreement` gap for a known set.

### Section Research

- `search_sections`
- `list_agreement_sections`
- `list_agreement_sections_batch`
- `get_section`
- `get_section_snippet`
- `get_section_snippets_batch`
- `get_sections_batch`

Use these when the client needs to search clause language across the corpus, navigate sections inside one agreement, inspect a known section directly, or extract a shorter plain-text excerpt from one section. `list_agreement_sections_batch`, `get_section_snippets_batch`, and `get_sections_batch` accept a list of agreement or section UUIDs and return results in a single call, reducing round-trips for multi-agreement workflows. `get_sections_batch` returns full section XML (capped by default at 10 000 characters per section); `get_section_snippets_batch` returns focused plain-text excerpts and is the right choice when the full XML is not needed.

### Discovery Helpers

- `suggest_clause_families`

Use this when the client knows the business or legal concept but does not know the right taxonomy `standard_id` yet. The tool returns ranked clause-family candidates with their taxonomy paths and matched terms.
Each match also reports whether it is a canonical fit, a proxy, or a broader semantic match.
The response additionally rolls the per-match verdicts up into a top-level `coverage` (`canonical` / `proxy` / `weak` / `none`) and a human-readable `coverage_note`. When `coverage` is `weak` or `none`, the concept is not cleanly represented and the returned nodes may be adjacent — or even opposite — concepts (for example, "go-shop" only surfaces the distinct "no-shop" node), so clients should verify them in section text rather than filtering on them blindly. Pass `taxonomy=tax_clauses` to map tax concepts and receive dotted `tax.*` ids for `search_tax_clauses`.

### Tax Clause Research

- `search_tax_clauses`
- `get_agreement_tax_clauses`
- `get_section_tax_clauses`

Use these when the task is specifically about extracted tax-module clauses rather than the full agreement or section text. `search_tax_clauses` is the corpus-wide entry point for the tax taxonomy: it mirrors `search_sections` but filters clause-level tax precedents by `tax_standard_id` (dotted `tax.*` ids from `suggest_clause_families(taxonomy='tax_clauses')` or `get_tax_clause_taxonomy`) plus the standard M&A filters. It excludes clauses inside representations & warranties by default; pass `include_rep_warranty=true` to include them. `get_agreement_tax_clauses` and `get_section_tax_clauses` remain the right tools once you already hold a specific agreement or section UUID.

### Research Bootstrap

- `list_filter_options`
- `get_clause_taxonomy`
- `get_tax_clause_taxonomy`

Use these when the client needs valid structured inputs before searching.

### Reference And Context

- `get_counsel_catalog`
- `get_naics_catalog`
- `get_agreements_summary`
- `get_agreement_trends`
- `get_server_capabilities`
- `get_server_metrics`

Use these when the client needs canonical lookup data, corpus-level context, or MCP introspection metadata.

### Feedback

- `submit_feedback`

The one write tool on the server. Agents (or users, through their client) use it to report bugs, workflow friction, missing capabilities, data-quality issues, documentation gaps, and praise for what worked well, during or at the end of a research session. Submissions are persisted with the submitting account, OAuth client, and scope set, and are read by the maintainer; there is no read-back surface.

Inputs: required `summary` (≤ 200 chars) and `detail` (≤ 5000 chars); optional `category` (`bug`, `friction`, `missing_capability`, `data_quality`, `docs_gap`, `praise`, `other` — default `other`), `severity` (`low`/`medium`/`high`), `tool_name` (validated against the live tool registry), `suggestions` (≤ 2000 chars), and `context` (a small JSON object, ≤ 2000 chars serialized — typically the arguments that triggered the issue). Whitespace-only text, an unknown `tool_name`, or an oversized `context` is rejected with an agent-visible `-32602` explaining the fix. Submissions are limited to 20 per account per rolling hour; hitting the limit returns an agent-visible message asking for consolidated reports. The response acknowledges with a `feedback_id` and `recorded_at`.

## Current Tool List

The current MCP tools are:

- `search_agreements`
- `search_sections`
- `search_tax_clauses`
- `list_agreements`
- `list_agreement_sections`
- `list_agreement_sections_batch`
- `get_agreement`
- `get_agreements_batch`
- `get_section`
- `get_section_snippet`
- `get_section_snippets_batch`
- `get_sections_batch`
- `get_agreement_tax_clauses`
- `get_section_tax_clauses`
- `list_filter_options`
- `suggest_clause_families`
- `get_server_metrics`
- `get_server_capabilities`
- `get_clause_taxonomy`
- `get_tax_clause_taxonomy`
- `get_counsel_catalog`
- `get_naics_catalog`
- `get_agreements_summary`
- `get_agreement_trends`
- `submit_feedback`

## Design Notes

- Every corpus tool is read-only; `submit_feedback` is the only tool that writes, and it writes exclusively to the feedback store
- Clients should typically choose the right tools automatically
- `search_agreements` is the discovery-oriented agreement search; accepts a `standard_id` list to filter to agreements that contain at least one section tagged with any of the given taxonomy ids
- `list_agreements` is the exact-filter, cursor-based agreement listing surface; also accepts `standard_id` for taxonomy-based agreement filtering, and reports dropped ids under `interpretation.unrecognized_standard_ids`. Unlike `search_agreements`, which always emits a four-key `interpretation` block, `list_agreements` emits `interpretation` **only when ids were dropped**, and it carries exactly `unrecognized_standard_ids` and `notes` (`additionalProperties: false`) — so read it defensively and do not expect `taxonomy_filters` there
- `list_agreements` rows are rich (full deal metadata); `search_agreements` rows are slim (`agreement_uuid`, `year`, `target`, `acquirer`, `filing_date`, `url`, `verified`, `section_count`, `target_counsel`, `acquirer_counsel`). Use `list_agreements` when the caller needs full metadata and expects to paginate deeply or export; use `search_agreements` for exploratory discovery. A full `list_agreements` row at `page_size=100` produces roughly 100k characters — pass `fields` (see below) or keep `page_size <= 25` when reading results in-context
- Both `search_agreements` and `list_agreements` accept an optional `fields: [string]` array (validated against that tool's own row keys) that projects each result row down to only the requested keys; `agreement_uuid` is always included regardless. Omitting `fields` keeps the full row for back-compat. `agreement_uuid` is the only key both tools' output schemas require — `search_agreements`' `verified` field was previously required on every row, but a projected row can omit it, so the schema now only guarantees `agreement_uuid`. Example: `fields: ["agreement_uuid", "url"]` for a bulk-export scan that only needs identifiers and EDGAR links
- Both `search_agreements` and `list_agreements` rows include `target_counsel`/`acquirer_counsel` (canonical firm names, ordered by position, one array per side), fetched with one batched query per page joining `agreement_counsel`/`counsel`. This closes a gap where filtering by `any_counsel` matched a firm on either side but gave no way to tell which side matched. The batched query is skipped when `fields` is provided and does not request either counsel key
- `list_agreements` accepts `sort_by` (`agreement_uuid` — default, the original behavior — `year`, `target`, `acquirer`, `filing_date`) and `sort_dir` (`asc`/`desc`). The `cursor` is a versioned, opaque composite keyset token that also embeds the `sort_by`/`sort_dir` it was issued under; a cursor from a call with different sort settings, or a cursor from before sort parity shipped, is rejected with an agent-visible `-32602` error rather than being silently misapplied or crashing — restart pagination without a cursor when changing sort settings. NULLs in the sort column always sort last, regardless of `sort_dir`, since MariaDB has no `NULLS LAST` keyword; ties (including the all-null block) break by `agreement_uuid` ascending so a full paginated scan stays gap- and duplicate-free
- `list_agreement_sections` is an MCP convenience tool for within-agreement navigation
- `search_sections` is a clause-language retrieval surface, not a normalized document-facts surface; it has **no free-text/keyword parameter** — it searches by clause-family `standard_id` and the structured M&A filters only. Use `suggest_clause_families` (or `get_clause_taxonomy`) to translate a plain-English concept into a `standard_id`, then filter by it. There is no keyword-search fallback to add: the `section_text_search` table in the main DB holds plain text for 10 477 of 967 121 sections (1.1%), roughly half of which are governing-law clauses, and it carries no FULLTEXT index — only B-tree keys on `section_uuid`, `agreement_uuid`, and `xml_version`. Exposing it would answer phrase queries from a narrow, unrepresentative slice of the corpus and return confident false negatives for everything else. If a concept has no taxonomy node, that should be reported as a coverage gap rather than approximated
- `search_sections` exposes `count_mode` and returns `count_metadata` plus `interpretation` so clients can tell when totals are exact versus estimated and when taxonomy is acting as a proxy
- `search_agreements` returns exact totals today and also includes `count_metadata` plus `interpretation`
- `standard_id` values are opaque 16-character hex node IDs obtained from `get_clause_taxonomy` or `suggest_clause_families` — they are not dotted decimal numbers. Unrecognized IDs are ignored (they silently match nothing) and are echoed back under `interpretation.unrecognized_standard_ids` — alongside `interpretation.taxonomy_filters` on `search_sections`/`search_agreements`, which emit the full interpretation block; `list_agreements` reports only the dropped ids and `notes`. This lets clients detect a mistyped or stale ID rather than mistaking an empty result for "no matches"
- `suggest_clause_families` exists to bridge plain-English concepts to taxonomy ids and now reports fit/confidence metadata so clients can distinguish canonical matches from broader proxies, plus a roll-up `coverage`/`coverage_note` so a weak or unrepresented concept is not silently treated as covered. Node **labels are not unique**: 69 of the 172 taxonomy nodes share a label with another node (the reps taxonomy carries one copy under Buyer/Parent and another under Company/Seller), and the duplicates score identically, so `label` alone cannot break the tie. Every match therefore carries `qualified_label` — the label plus the distinguishing ancestor in brackets when it collides, unchanged otherwise — and a `label_is_ambiguous` flag. `qualified_label` is unique across the taxonomy: where no ancestor separates the nodes (one branch repeats the same label at L1, L2 and L3) it falls back to the `standard_id`. Select nodes by `standard_id`; use `qualified_label` when naming one
- The tax taxonomy is a **separate** namespace from the clause taxonomy: tax nodes are dotted `tax.*` ids and are searched corpus-wide through `search_tax_clauses` via `tax_standard_id` (not the 16-hex `standard_id` used by `search_sections`/`search_agreements`). Passing a `tax.*` id to `standard_id` is ignored and echoed under `interpretation.unrecognized_standard_ids`; the reverse — a 16-hex clause-family id passed to `tax_standard_id` — is echoed under `interpretation.unrecognized_tax_standard_ids` on `search_tax_clauses`, `get_agreement_tax_clauses`, and `get_section_tax_clauses`, with an extra note when the ignored id has the 16-hex clause-family shape. The block is present only when ids were dropped, so its absence means every id was applied. Use `suggest_clause_families(taxonomy='tax_clauses')` to obtain valid tax ids
- `get_section_snippet` is a focused reading aid, not a replacement for `get_section` or a canonical extracted-facts surface
- `get_section_snippets_batch` and `list_agreement_sections_batch` accept arrays of UUIDs and collapse multiple single-item calls into one round-trip; use them when a workflow would otherwise fan out across many agreements or sections
- `get_sections_batch` fetches full section XML for up to 10 sections in one call; XML is capped at `max_xml_chars` per section (default 10 000, range 500–20 000) to prevent context overload; when a section is truncated the result includes `xml_truncated: true`; pass `max_xml_chars: null` only if uncapped XML is explicitly needed
- The batch retrieval tools distinguish a missing record from an empty one, so a partial result cannot be read as a measured absence. `get_sections_batch` omits a `section_uuid` that matches no section and names it under `unresolved_section_uuids`, so `returned_count` being lower than the number of distinct UUIDs passed is explained in-band rather than left for the caller to detect by diffing. `list_agreement_sections_batch` does the same for agreement UUIDs under `unresolved_agreement_uuids`: an unknown or non-retrievable agreement is dropped from `results` instead of being returned as a `section_count: 0` entry, which previously asserted that a real agreement had no sections. `get_section_snippets_batch` reports the same field. The ids appear in two places on each response — top level and inside `interpretation` — so either read works. As with the taxonomy `unrecognized_*` blocks, the field is present only when something was dropped, so its absence means every UUID resolved. Repeated UUIDs are deduplicated before the lookup, so a uuid passed twice yields one result and is counted once
- `list_agreement_sections_batch` caps each agreement at `max_sections_per_agreement` sections (default 200, range 1–1000). Agreements run to 530 sections, so 20 of them uncapped is roughly 6 400 rows in a single response — the one batch tool that bounded only its input count and not its payload. Each entry carries three distinct counts: `total_agreement_sections` (every section in the agreement, ignoring any `standard_id` filter), `matched_section_count` (sections matching the filter, before the cap), and `section_count` (what the response actually contains, after the cap) — plus `sections_truncated`; page through a single long agreement with `list_agreement_sections` instead of raising the cap
- `list_agreement_sections` returns not-found for an agreement UUID with no sections in the search index, matching `get_agreement` for the same UUID. The index currently holds exactly the retrievable agreements and every one of them has at least one section, so an empty unfiltered listing can only mean a bad UUID; returning an empty page for it claimed an absence that had never been measured. An empty `results` page now unambiguously means the `standard_id` filter matched nothing. Note the section tools test membership of that index rather than re-applying the eligibility gate directly — the two coincide on the current corpus, but they are maintained by separate ETL steps
- `search_sections` results include `filing_date` and `transaction_price_total` inline on every result without needing to request them via `metadata`. Requesting `transaction_price_total` in `metadata` no longer repeats it inside the metadata block on the MCP contract, since the value is already top-level; the web contract is unchanged. If it was the **only** field requested, the now-empty `metadata` object is dropped from the result entirely rather than returned empty, so index it defensively
- `search_sections` accepts `include_snippet: true` (with optional `snippet_focus_terms` and `snippet_max_chars`, default 400, range 120–1200) to return a plain-text `snippet` plus `matched_terms`, `source_length`, and `monetary_values` on every result. This collapses the usual `search_sections` → `get_section_snippets_batch` pair into one call for the common "find the clause and read what it says" workflow; the section text is already loaded to serve the search, so the excerpt costs no extra query. It stays off by default, and is far smaller than `include_xml`
- `search_sections` reports `page_unique_agreement_count` — distinct agreements **within the returned page**, not across the whole filtered set. It sits next to a corpus-wide `total_count`, so the scope is named explicitly to prevent reading "241 sections across 3 agreements" off a 3-result page
- `get_section_snippet`, `get_section_snippets_batch`, `get_sections_batch`, and `search_sections` with `include_snippet: true` all include a `monetary_values` list — dollar amounts and value expressions extracted from the section text — so clients can surface deal economics without parsing XML. Extraction requires the digit run to end in a digit and unit suffixes (`B`/`M`/`T`/`K`, `bn`, `mm`, `million`, …) to end on a word boundary, so trailing punctuation and the first letter of the following word are not absorbed into the amount
- `search_sections`, `search_agreements`, and `list_agreements` all accept `filed_after` and `filed_before` (ISO 8601 date strings, `YYYY-MM-DD`) for sub-year filing-date precision; `year`/`year_min`/`year_max` filter on the agreement year; `filed_after`/`filed_before` filter on the exact filing date. `list_agreements` previously advertised and validated these five fields (`any_counsel`, `year_min`, `year_max`, `filed_after`, `filed_before`) without ever applying `filed_after`/`filed_before` in the query — the filter was silently a no-op. Both the bug and the schema/handler drift that let it through the advertised-schema build are fixed: `list_agreements`' advertised input schema is now built from the exact schema instance its handler validates against, instead of a narrower base schema patched with JSON-schema-only overrides that a field not already present would silently drop
- `get_agreement` returns metadata only by default (the response reports `xml_included: false`) to keep the payload small; pass `include_xml: true` to also return the agreement XML, which preserves the existing redaction and full-text access behavior (`focus_section_uuid`/`neighbor_sections` only take effect when `include_xml` is true and the response is **redacted**). A caller holding `agreements:read_fulltext` therefore gets the whole document and those two arguments do nothing — and agreement XML is large: 91.1% of retrievable agreements exceed 200 000 characters (9 021 of 9 902), the mean is ~345 000 and the largest is 2 400 873, so `include_xml: true` will exhaust a typical context window. To read specific clauses from a known agreement, use `search_sections` with `agreement_uuid` and `include_snippet: true`, or `get_sections_batch` (which caps XML per section), rather than fetching the full body
- `get_agreements_batch` fetches metadata for up to 25 known agreement UUIDs in one call, using the same column list as `get_agreement` plus `target_industry_label`/`acquirer_industry_label` and the `target_counsel`/`acquirer_counsel` echo. Metadata only — there is no `include_xml` option. A UUID that matches no retrievable agreement is omitted from `results` and named under `unresolved_agreement_uuids`, following the same pattern as `get_sections_batch`/`list_agreement_sections_batch`: do not read a shorter `results` list as an absence without checking that field
- `target_industry_label`/`acquirer_industry_label` (on `list_agreements`, `get_agreement`, `get_agreements_batch`, and `search_sections` per-result `metadata`) are now always present when the row itself is present, `null` rather than omitted when the underlying NAICS code is absent or does not resolve in the catalog. A real export previously found the key missing on 9 of 326 rows, which broke strict downstream schema validation expecting the key always present
- `verified` (on `search_agreements` rows) is a metadata human-verification flag — true when the agreement's deal metadata has been reviewed and confirmed by hand — distinct from the automated XML-structure verifier that runs during ingestion (the `xml.status` column). It also gates visibility for otherwise-gated filings: a gated agreement only appears in results when `verified` is true, though an ungated agreement can appear with either value
- `get_agreements_summary` reports `agreements` and `sections` at the scope the retrieval tools can actually return, and `pages` at ingestion scope. `agreements` is counted over the same expressions every listing builds (the join to the latest verified XML row, plus the public-eligibility gate), so it equals an unfiltered `search_agreements` `total_count`. `sections` counts the same search index `search_sections` paginates, so it equals an unfiltered `search_sections` total. `pages` is the ingestion-wide total from the `summary_data` rollup and is **not** retrievable-scoped: the per-page table that would let it be filtered to the retrievable set holds full page content, is ETL-only, and is not present in the serving schema, so there is no served source for a retrievable page count. Treat `pages` as a corpus-ingestion size, the same figure the website shows; do not divide it by `agreements` for a per-retrievable-document average, since the two are different scopes. `agreements` and `sections` are read live rather than from the rollup because that rollup omits the latest-verified-XML join and so counts agreements no tool here can return — on the current corpus 10 784 rollup agreements against 9 902 retrievable, an 8.2% gap
- The website and the MCP server publish deliberately different corpus counts, and this is not a bug. The site's summary cards report the ingestion rollup (`summary_data`, 10 784 agreements); `get_agreements_summary` reports the retrievable universe (9 902). The 882-agreement difference is agreements that have been ingested but have no latest verified XML, so no Pandects surface — web index, REST API, or MCP — can return their text. `sections` and `pages` figures shown on the site come from the same rollup and are scoped to it. When quoting a corpus size, say which measure you mean
- `list_filter_options` returns more than the value lists: `retrieval_parameter_map` translates each plural catalog key into the singular argument name the retrieval tools expect (`target_industries` → `target_industry`), and `industry_labels` decodes the NAICS codes in the industry lists. A handful of in-use codes (for example `37`, `67`, `459`, `513`, `677`, `922`) have no entry in the NAICS reference tables and so appear in the filter lists without a label
- Industry filters match on NAICS **codes**, not descriptions. `target_industry: ["511"]` works; `target_industry: ["Technology"]` matches nothing and is not reported as an unrecognized value, so it looks like a real empty result. Use `list_filter_options` (which returns the in-use codes and `industry_labels`) or `get_naics_catalog` to resolve a description to a code before filtering
- `search_sections` returns a top-level `standard_id_labels` map resolving the `standard_id` values on the page to their taxonomy labels, so a client does not need a follow-up `get_clause_taxonomy` call just to name the nodes it matched. It resolves what the taxonomy tables contain: a small number of ids present on sections have no taxonomy row and are simply absent from the map, so look ids up in it rather than assuming every id on the page appears
- `monetary_values` is capped at 20 distinct values per section. When the cap drops any, `monetary_values_truncated` is `true` — read it before treating the list as every amount in the clause
- `get_agreement_trends` accepts a `sections` selector (`ownership`, `target_industries`, `pairings`, `naics_catalog`) and defaults to `[ownership, target_industries]`; the large `pairings` (every industry-by-industry cell) and `naics_catalog` (full NAICS hierarchy) sections are opt-in, and the response echoes `sections_returned`. It also accepts `year_min`/`year_max` to slice the per-year arrays (ownership mix/deal size, target-industry distributions) to a window instead of returning every year. `ownership.buyer_type_matrix` and `industries.pairings` come from summary tables that are pre-aggregated across the whole corpus and carry no year dimension, so the window cannot apply to them. The echoed `year_filter` states this in-band: alongside `year_min`/`year_max` it lists `applied_to` (the dotted paths the window narrowed) and `not_applied_to` (paths still covering all years), plus a `note` explaining why. Read `not_applied_to` before quoting any figure from those sections as a windowed number
- Agreement industry fields are stored as bare NAICS codes (e.g. `"334"`). `get_agreement`, `list_agreements`, `get_agreements_batch`, and `search_sections` (in per-result `metadata`) add a decoded `target_industry_label` / `acquirer_industry_label` sibling, so clients no longer need a separate `get_naics_catalog` round-trip to read an industry; see the null-consistency note above for when the code does not resolve
- The corpus is SEC-filed, predominantly public-target M&A exhibits sourced from EDGAR. Private-deal language is largely absent, so a language-frequency study built on this corpus inherits that skew; `target_type` is the closest available filter for public/private, not an exact proxy for deal population. The same underlying deal can also appear as multiple filings — both parties may file, and amendments or closing re-filings add further copies — and there is currently no deal-level dedupe key, so a filing count is not a deal count
- For corpus-scale text analysis, filter with `list_agreements` using `fields: ["agreement_uuid", "url", ...]` and fetch full agreement bodies from the returned EDGAR `url` out-of-band. Do not pull full texts through MCP responses at scale: a full-row `list_agreements` page at `page_size=100` already runs roughly 100k characters before any XML is attached, and `get_agreement`/`get_sections_batch` XML is larger still
- `get_agreement_tax_clauses` and `get_section_tax_clauses` return an `extraction_status` (`found` / `no_tax_clauses` / `not_extracted`) plus an `extraction_note` so an empty `clauses` list is unambiguous: `no_tax_clauses` means clause extraction ran and found no tax-module clauses (a real absence), while `not_extracted` means no clauses of any module exist for the record, so the empty result is uninformative rather than a confirmed absence
- `get_agreement_tax_clauses` and `get_section_tax_clauses` are page-paginated (`page`, `page_size`, default 25, max 200) and return the standard flat pagination block (`total_count`, `total_pages`, `has_next`, …). Both also accept `tax_standard_id` (dotted `tax.*` ids, parent ids expand to descendants) to return only clauses assigned to specific taxonomy nodes. This matters because most extracted tax-module clauses carry an empty `standard_ids` list — an unfiltered agreement fetch can return several hundred clauses of full text. `extraction_status` always describes the record as a whole, never the `tax_standard_id` filter, so a filter that matches nothing still reports `found` rather than implying extraction never ran
- `get_server_capabilities` is the main machine-readable semantics surface; it includes auth guidance, field inventory, concept notes, and negative guidance about when not to use a tool
- Feedback encouragement is deliberate and layered: the `initialize` instructions ask agents to report friction as it happens and to file a session-end wrap-up, `get_server_capabilities` carries a `server.feedback_tool`/`server.feedback_note` pair plus a "report friction or gaps you encountered" workflow, and the `submit_feedback` tool description spells out what a useful report contains (tool, arguments, expected vs actual — and praise for what worked). Any authenticated principal can submit; no extra scope is required
- The server exposes a small set of MCP resources (`pandects://capabilities`, `pandects://auth-help`, `pandects://tools-manifest`) that mirror `get_server_capabilities` for clients that prefer the `resources/read` primitive over calling a tool. `get_server_capabilities` reports `server.resources_supported: true` accordingly; a test pins that declaration to what `resources/list` actually serves, since the two had drifted apart
- The server exposes curated MCP prompts (`compare_agreements`, `clause_family_survey`, `deal_trend_brief`) as research templates; they orchestrate the primitive retrieval tools rather than introducing new functionality. Because they encode a tool sequence, they have to be revisited whenever the tools change: `clause_family_survey` previously taught the `search_sections` → `get_section_snippet` round-trip that `include_snippet` removed, and `compare_agreements` routed general "quantified clause positions" to the tax-only tools. Both now match the current tool surface

## Transport

- `POST /mcp` is the primary JSON-RPC endpoint. It supports content negotiation: clients that advertise `Accept: text/event-stream` receive an SSE-framed response; clients that prefer `application/json` receive a plain JSON body. This matches the Streamable HTTP behaviour required by Claude Code.
- `GET /mcp` returns an SSE retry probe for clients that opportunistically open a server-to-client stream.
- `DELETE /mcp` is accepted as an authenticated session-termination signal and returns `204`.
- `initialize` responses carry an `Mcp-Session-Id` header. The server is stateless, so the id is informational — clients are not required to echo it, but Claude Code does.
- Every response carries an `MCP-Protocol-Version` header echoing the negotiated protocol version.
- Advertised server capabilities: `tools`, `resources` (listChanged=false, subscribe=false), `prompts` (listChanged=false), and `logging` (`logging/setLevel` is accepted as a no-op).

### Tool error codes

`tools/call` failures use distinct JSON-RPC error codes so a client can tell a fixable request from a server-side fault. The codes are identical across the JSON and SSE response paths:

- `-32602` — invalid tool arguments (schema/validation failure or a malformed identifier). The offending field(s) and reason are folded into the top-level error `message` (e.g. `Invalid tool arguments: metadata: Must be one of: ...`), not just the `data` payload, so clients that surface only `message` can still self-correct. The full per-field marshmallow error tree remains available under `data`. Argument rejections raised by a tool handler rather than by schema validation carry their reason too (e.g. `Tool request was invalid. Invalid section_uuid: <value>`); this applies only to messages the handlers explicitly author, so an unexpected `400` raised elsewhere in the stack stays generic and never leaks internal detail.
- `-32002` — the tool ran but the requested resource was not found (e.g. an unknown agreement or section UUID); the error `data` is `{"category": "not_found", "status_code": 404}`. Verify the identifier with a search or list tool and retry — this is not a transient error.
- `-32001` — unauthenticated: no bearer token, or one that is invalid or expired. Distinct from `-32003`, which means the token is valid but lacks the scope the tool requires
- `-32003` — missing scope / authorization
- `-32603` — the tool result violated its advertised output schema
- `-32004` — a genuine tool failure (server-side)

In `get_server_metrics`, a not-found tool result is counted under its own `not_found` error category (not the generic `http_exception`), since a bad identifier is a client mistake rather than a server fault. Note that metrics counters are per-process: in a multi-worker deployment each `get_server_metrics` call reports only the worker that served it, not a server-wide total.

### Progress notifications

When a `tools/call` request includes `params._meta.progressToken` **and** the client advertises `Accept: text/event-stream`, the server returns a multi-event SSE stream:

1. `notifications/progress` with `progress=0`, `total=1`, and a `Starting <tool>` message
2. `notifications/progress` with `progress=1`, `total=1`, and a `<tool> complete` message
3. The final `tools/call` JSON-RPC result (or error)

This keeps intermediary proxies and client UIs aware of in-flight work on long calls. Clients that do not set a progress token, or do not accept SSE, receive the usual single-response behaviour.

### OAuth discovery and Dynamic Client Registration

The server is protected by an embedded OAuth authorization server whose issuer lives under `/v1/auth/oauth`. To make OAuth discovery work with clients that implement RFC 8414 strictly (including Claude Code), authorization-server metadata is exposed at three locations:

- `GET /.well-known/oauth-authorization-server` — host-root fallback
- `GET /.well-known/oauth-authorization-server/v1/auth/oauth` — RFC 8414 host-root + issuer-path form
- `GET /v1/auth/oauth/.well-known/oauth-authorization-server` — issuer-prefixed form (original)

`GET /.well-known/openid-configuration` is also exposed at the host root for OIDC-leaning clients.

The metadata document advertises `registration_endpoint` (`/v1/auth/oauth/register`), so compliant clients can self-register via Dynamic Client Registration (RFC 7591) without a manual out-of-band step. Only public, PKCE (S256), authorization-code clients are supported.

## Authentication

- MCP uses normal Pandects account login
- MCP does not use Pandects API keys
- `codex mcp add` only registers the server; in Codex, `codex mcp login <name>` starts OAuth
- Auth failures return structured remediation metadata so clients can distinguish missing-token, expired-token, unverified-account, and unlinked-subject cases

## Related Pages

- [Using MCP](./using)
- [Setup](./setup)
