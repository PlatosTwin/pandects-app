# Span taxonomy: concept labels on text spans, with extracted values

Status: design draft, 2026-10-20. Nothing here is built. A companion plan, `docs/plans/smart-reader.md` (live TOC, defined terms and cross-references; [PR #186](https://github.com/PlatosTwin/pandects-app/pull/186)), shares the text layer described in [section 6](#6-span-annotation-model).

## 1. Recommendation in brief

1. **Replace section labels with concept instances anchored to text spans.** A *provision instance* is one occurrence of a concept in one agreement (for example, "the Company Termination Fee in the Grace merger agreement"). It has one or more spans, each with a role (amount, trigger, definition), and a typed set of extracted values. Each value cites the span it came from. Section-level labels become a derived view.
2. **Anchor spans to a canonical text layer projected from the verified XML. Do not anchor to the XML string or to section UUIDs.** Store each span as a position (offsets plus a layer hash) and a quote (the exact text plus 64 characters of context on each side). The pilot re-anchored 99.5% of 2,000 spans across real XML rebuilds this way. Raw offsets alone would have silently pointed 13% of them at the wrong text.
3. **Author the taxonomy as versioned YAML in git**, in two discipline packs (`ma`, `tax`) with readable, stable concept IDs, and load it with explicit DDL that has primary keys. This retires both the out-of-band seeding and the primary-key landmine.
4. **Extract in three stages:** cheap, high-recall candidate detection (structure, lexical cues, embeddings, the defined-term graph); then span classification by a small model over a short list of candidate concepts; then value extraction by a stronger model, with required evidence quotes and deterministic validators. A full corpus pass for 10 concepts costs roughly $450–$900 with Claude Sonnet 5.5 doing the value step at batch rates. Gold pre-labeling with Opus is tens of dollars.
5. **Start with a 50-agreement pilot on three concepts** (termination fee, reverse termination fee, outside date), scored against a human-reviewed gold set. Grow to 10 concepts only after it clears the bar in [section 9](#9-evaluation-plan).

## 2. The problem

Today's taxonomy answers "what is this section about?" from its title alone. Users want two things it cannot answer:

- **Where exactly is the provision?** One section often holds several provisions. "Effect of Termination; Termination Fees" carries a company fee, a reverse fee, expense reimbursement and an exclusive-remedy clause. One provision can also span several sections: the fee amount sits in 8.3, its triggers point at 8.1(c) and 8.1(e), and "Termination Fee" is defined in Article I. A section label can only say the topic appears somewhere inside.
- **What does it say?** Users want the fee amount, who pays it, when, on which triggers, and how large it is relative to equity value, and they want the numbers to be aggregable across the corpus. The only value extraction today is a request-time money regex in the MCP server that returns raw strings with no role. The deferred MCP aggregation work (see "MCP aggregation deferred" in the project notes) is blocked on exactly this missing piece.

A rebuild of the taxonomy is already due (the project notes say it will be redone from scratch). This plan makes that rebuild the moment to change the unit of labeling.

## 3. Current state

Facts below come from the code and from read-only queries on the local `pdx` database on 2026-10-20.

### 3.1 Section taxonomy

- **Tables.** `taxonomy_l1/l2/l3` (live: 12 / 61 / 99 nodes) hold `standard_id char(16)` (opaque hex), `label` and `parent_id`. The `*_new` copies hold the 202-node redesign (12 / 62 / 140), also in `etl/src/etl/models/taxonomy/deepseek/taxonomy_enrichment.yaml`. That redesign adds a description, canonical terms, example phrases and exclusion cues for each node, loaded into `taxonomy_node_meta` and `taxonomy_node_embedding` (voyage-4-large).
- **No DDL in the repo.** The tables are seeded out of band. Rebuilds drop their primary keys, so `backend/models/main_db.py:660-683` pins a mapper primary key to keep the backend booting. This is the "taxonomy PK landmine".
- **Labeling** (`etl/src/etl/domain/h_taxonomy.py:116-310`, `etl/src/etl/defs/h_taxonomy_asset.py`):
  - The prompt sends titles only: "using ONLY the provided titles/context (no full section text)", the section, article and neighbor titles, and the whole flattened taxonomy.
  - It asks for the lowest applicable level, one primary category plus up to three more. Model `gpt-5.4-mini` with high reasoning effort, OpenAI Batch API, 5 sections per request.
  - Output is a JSON array in `sections.section_standard_id_gold_label`. The same string is written into the XML `<section standardId=…>` attribute, overwriting that XML version in place.
  - No confidence and no cost or token tracking.
- **Coverage and freshness.** 505,035 of 966,534 section rows carry labels. They were produced by a mix of models: gpt-5-mini 368,891, gpt-5.4-mini 69,081, gpt-5.1 65,822, gpt-5.4 1,241. Re-tagged sections lose their labels until the next rebuild (changelog entry). `section_select.py` itself calls the old labels stale.
- **Research path.** An offline DeepSeek-V4-Pro run with a hybrid retriever (top-50 candidates, body text included) was validated but not wired in.
- **External write-up.** Praxis1909/background-provision-classification records the lessons:
  - Accuracy falls as the number of classes shown grows.
  - Retrieval recall@50 is 99.3%.
  - Micro-F1 is 0.836, but only against a GPT-generated reference, not human gold.
  - Confusion about which party a representation belongs to caused a large share of errors.
  - Give each discipline (M&A, tax, compensation) its own taxonomy.

### 3.2 Tax taxonomy and clauses

- **Taxonomy.** `tax_clause_taxonomy_l1/l2/l3` has 9 / 45 / 96 nodes with readable dotted IDs (`tax.1.3`). This is a better ID style than the hex IDs.
- **Clauses.** `clauses` (module `tax`) holds 28,166 clauses over roughly 250 agreements. Each comes from a rule-based splitter (`etl/src/etl/domain/tax_module.py:342-408`), which tries, in order, the whole section, a title match, top-level `(a)/(b)` enumeration, then paragraph grouping. `tax_clause_assignments` holds 11,432 labels on 7,262 clauses. The pipeline is off by default (`enable_tax_taxonomy: false`).
- **The offsets are the first span model in the codebase, and they are fragile:**
  - `start_char/end_char` index the raw `sections.xml_content` XML string, tags included. The DBML says "within the section text", which is wrong.
  - `clause_text` is tag-stripped text joined with `\n\n`, so it cannot be sliced back out of any stored text.
  - Rebuilds delete every clause for the agreement and reinsert (`db_utils.py:653-723`), so assignments do not survive an XML change.
  - Clause boundaries are only as good as the enumeration heuristics. They never cross a section.

### 3.3 XML, sections and coordinate systems

- **XML shape.** The verified XML (`xml.xml`, keyed by `agreement_uuid`, `version`) nests `<article>` → `<section>` → leaf `<text>` and `<definition>` elements, one per source line fragment. `<page>` and `<pageUUID>` markers are interleaved (`etl/src/etl/domain/f_xml.py:698-900`).
- **Section IDs are not stable.** A section UUID is uuid5 of the agreement, normalized titles and article/section order (`f_xml.py:40-54`). A heading repair or an inserted section changes it, and changes every later section's order. In the pilot below, 5.9% of sections in rebuilt agreements got new UUIDs. In one agreement, 69 of 84 did.
- **Three text coordinate systems already exist**, and none is canonical: the XML string (tax clause offsets), `section_text_search.normalized_text` (case-folded NFKC text) and `clause_text`.
- **Corpus size.** 9,873 agreements have a latest verified XML, with 964,202 sections and 2.38 billion characters of normalized text, about 600 million tokens (roughly 60,000 per agreement).

### 3.4 How labels reach users

- **API.** `GET /v1/taxonomy`, `/v1/taxonomy/tax-clauses` and their `/counts` siblings; `GET /v1/sections?standard_id=`, `/v1/search/agreements?standard_id=`, `GET /v1/tax-clauses?tax_standard_id=`, plus agreement and section tax-clause endpoints. These expose `start_char`/`end_char`, but nothing consumes them. Filters run as an `EXISTS` on `latest_sections_search_standard_ids` and `tax_clause_assignments` (`backend/models/main_db.py:860, 950`).
- **MCP.** `get_clause_taxonomy`, `get_tax_clause_taxonomy`, `search_tax_clauses` and `suggest_clause_families`. The last ranks with string similarity plus a hard-coded synonym table (`backend/mcp/tools/shared.py:314, 443-481`), a sign that opaque hex IDs and label-only nodes are hard for agents to use.
- **Frontend.** The Search taxonomy filter (`NestedCheckboxFilter`, `SearchQueryBar.tsx:174-188`), the prerendered Taxonomy page (`pages/Taxonomy.tsx`) and section-type chips in the reader (`AgreementReader.tsx:160, 245-260`). The reader highlights only at section level. No frontend code uses span offsets.
- **Bulk dump.** The dump includes `sections`, `clauses`, both taxonomies, `tax_clause_assignments` and the search denormalizations (`bulk/public_tables.txt`). `push_to_r2.sh` fails on any allowlisted table or column without docs in `bulk/schema_docs/table_docs.yml`, and on any schema-fingerprint change without a changelog `schema` entry.

### 3.5 Why `monetary_values` cannot be aggregated

`_MONEY_RE` and `_extract_monetary_values_with_truncation` (`backend/mcp/tools/shared.py:663-698`) run at request time over a whole section's plain text. They return at most 20 distinct raw strings. The design rules out aggregation, not just the implementation:

1. **No role.** A termination fee, an indemnity cap, a basket, a per-share price, a debt threshold in a definition and an example amount all come back as the same kind of string.
2. **No normalization.** "$1.5 billion", "$1,500,000,000" and "1.5 billion dollars" stay three values. De-duplication is by lowercase string, not by amount.
3. **Wrong unit.** It works per section, not per provision. One section can hold three fees, and one fee's amount can sit in a different section from its triggers. The 20-value cap truncates long sections.
4. **Dollars only.** Euro, sterling and percentage-based amounts ("3.5% of the Equity Value") are missed.
5. **No offsets**, so a value cannot be checked against its text.
6. **History of silent corruption.** The code comment records regex defects that "corrupted ~11% of extracted values across the corpus" before they were fixed.

The design below fixes each point: values hang off a concept instance (role), are normalized into typed fields, can be expressed as a percentage of a basis, and carry evidence offsets.

## 4. Design principles

- **Standoff, not inline.** Annotations live in tables and point into text. The XML is never mutated to hold labels. Today's in-place `standardId` write into a versioned XML is a pattern to stop.
- **One canonical text layer per XML version**, shared with the smart reader.
- **Concepts are about what a provision does, not where it sits.** Navigation (the browse tree) and concepts (what spans point at) are separate structures.
- **Every extracted value carries evidence**: a quote that must appear verbatim in the text. A value with no evidence is not stored.
- **Small decision sets.** Never ask a model to choose among 200 classes. Retrieve a short list first (the Praxis lesson).
- **Gold is human-reviewed.** Model pre-labels are suggestions until a reviewer accepts them. A reference produced by another model is not gold.
- **Everything is versioned:** taxonomy release, value-schema version, prompt hash and model for every run, with cost recorded.

## 5. Taxonomy structure

### 5.1 Two layers: concepts and navigation

| Layer | What it is | Used by |
|---|---|---|
| **Concept** | A provision or mechanic that a span can carry: `ma.termination.fee.target`, `ma.mae.carveout`, `tax.election.338h10`. Leaves and inner nodes can both be concepts. | Annotations, values, search filters, aggregation |
| **Navigation tree** | Article-like grouping for browsing (Termination › Fees and Expenses). Each concept has one primary navigation parent and may have extra "also listed under" links. | Taxonomy page, filter UI, section-topic view |

Keeping the layers apart solves two problems in today's tree:
- A concept like the 338(h)(10) election belongs to tax but also matters under M&A "Tax Matters". It is listed in both trees but is one concept, so it is counted once.
- Representations no longer need duplicate subtrees per party. "Compliance with Laws" is one concept, with `party` as an attribute of the instance.

Duplicated per-party subtrees were the single largest error source in the Praxis write-up. Moving `party` to the instance changes the model's job from "pick one of two near-identical classes" to "read which party makes the representation", which comes from the article title and is easy to check with a rule.

### 5.2 IDs and versioning

- **Concept ID** is a readable, dotted, lowercase slug, namespaced by discipline pack (`ma.`, `tax.`, later `comp.` for executive compensation and benefits). It is stable forever: never reused, never renamed. A label can change freely. A concept that has to be split or merged is deprecated with `replaced_by: [...]`.
- **Taxonomy release** follows `YYYY.N` (for example `2027.1`). Every annotation and value records the release and the value-schema version it was produced under. Adding concepts or fields is a minor change. A split, merge or change of field meaning bumps the schema version and ships a migration map.
- **Crosswalks**, stored as rows rather than in code, map to:
  - the legacy section IDs (16-hex, both the live 99-leaf and the 140-leaf trees);
  - legacy tax IDs (`tax.N.N.N`);
  - DealOS KG concept IDs (`cra:4`, `pcra:38`, `pcor:9`), each with a relation: exact, broader, narrower or related.

  The legacy crosswalk keeps old API filters working through a deprecation window.

### 5.3 Source of truth and loading

- **YAML in git**, under `etl/taxonomy/<pack>/*.yaml`, one file per navigation branch. Each node carries:
  - id, label, kind (`provision`, `component`, `definition`, `modifier`) and navigation parent;
  - description, canonical terms, example phrases and exclusion cues (carried over from `taxonomy_enrichment.yaml`, which is good material);
  - an optional `value_schema`;
  - crosswalks;
  - `introduced_in` and `deprecated_in`.
- **A loader** builds the tables from the YAML, using DDL checked into `etl/sql/migrations/`, primary keys included. It upserts by ID and never drops. A CI check validates the YAML: unique IDs, parents that exist, no cycles, value schemas that compile, and no reused IDs across releases (checked against the previous release file).
- **This removes the landmine:** no table is ever recreated, and the backend's mapper-PK workaround can go once the old tables are retired.

### 5.4 How tax fits

`tax` is its own pack, authored with tax practice in mind, as both the Praxis write-up and the current 9-L1 tax tree argue. It shares everything else: the span model, the instance and value tables, the extraction pipeline and the evaluation harness. Concepts that live at the seam are owned by one pack and linked from the other's navigation tree:

| Concept | Owner | Also listed under |
|---|---|---|
| 338(h)(10) / 336(e) / 754 elections | `tax` | `ma` › Covenants › Tax Matters |
| Purchase price allocation | `tax` | `ma` › Consideration |
| Tax representations | `tax` | `ma` › Representations (with `party`) |
| Tax indemnity, straddle allocation | `tax` | `ma` › Indemnification |
| Transfer taxes | `tax` | `ma` › Fees and Expenses |
| Intended tax treatment (368(a), 351) | `tax` | `ma` › Deal Structure |

The existing tax L2/L3 nodes mostly map one to one onto `tax.` concepts. Their IDs already have the right shape (`tax.1.3` becomes `tax.election`, and `tax.1.3.x` becomes `tax.election.338h10`).

### 5.5 Granularity and the first concept list

There are two tiers of concepts:
- **Value concepts** carry a value schema and get the full three-stage extraction. These are the "smart" labels.
- **Label concepts** carry spans but no values. They give coverage for search and navigation.

v1 starts with the 10 value concepts in [section 7](#7-value-schemas-first-10-concepts). Coverage grows toward the roughly 140-leaf breadth of the current redesign, reusing its nodes as label concepts where they are already provision-shaped. Candidates for the next wave of value concepts:

- go-shop and window-shop period;
- no-shop fiduciary-out mechanics;
- matching rights (days, number of rounds);
- regulatory efforts standard ("hell or high water", divestiture caps);
- the HSR/CFIUS condition;
- specific performance;
- financing cooperation;
- the purchase price adjustment (working-capital collar);
- the earnout (metric, period, cap);
- escrow and holdback;
- the R&W insurance retention;
- non-compete length;
- the D&O tail (years, premium cap);
- governing law and forum;
- the interim operating covenant thresholds.

DealOS KG is the vocabulary reference. Its `moving_parts` (obligor, conduct, outcome, timing, exceptions) are a good generic skeleton for value schemas, and its hierarchy (for example, the MAE family: core concept, carve-outs ×7, disproportionate-impact exception, measurement period) has already done the decomposition work. Its corpus is 8 mostly private purchase agreements, though, while Pandects is mostly public mergers on EDGAR. Crosswalk to it; do not adopt it wholesale. Its prose should not be copied into Pandects.

## 6. Span annotation model

### 6.1 The canonical text layer

For each `(agreement_uuid, xml_version)`, the text layer is a deterministic projection of the verified XML:
- walk `<body>` in document order;
- emit the text of every `<text>` element, unstripped, followed by `\n`. A `<definition>` is a container of `<text>` elements (it never holds text of its own), so it is not a leaf; its `<text>` children are tagged with the defined term;
- skip `<page>` and `<pageUUID>`.

Other parts:
- **Offsets** are Unicode code-point indices (Python `str` indices). The API also returns UTF-16 offsets for the browser, or the frontend maps between them.
- **`layer_hash`** is the sha256 of the layer text.
- **Section and paragraph ranges** come out of the same walk: the start and end of each `<section uuid>`, and each smart-reader subsection node such as `8.3(b)(i)`. They are stored as a structure index on the layer.

Front matter, table of contents, signature pages and back matter are separate zones of the same layer. The value pilot found that exhibits carry values the body depends on, so back matter should be in v1 if the XML holds it reliably ([open question 3](#15-open-questions-for-you)). The projection also drops page-footer leftovers (a page number followed by a GUID line) and records repeated blocks as an XML quality signal.

**Reconciling with the smart reader's anchor contract** ([PR #186](https://github.com/PlatosTwin/pandects-app/pull/186), section 6.3). The reader index anchors to (section UUID or region, text-node ordinal, offset in UTF-16 code units over the entity-decoded node text), and it is regenerated for every XML version. The two schemes fit together:
- **The node text must be identical.** The layer is the concatenation of exactly the per-node strings the reader uses, with the same normalization as `normalizeXmlText` and no extra stripping (the pilot script's `.strip()` must go), joined by `\n`. A (node ordinal, offset) pair and a layer offset then convert into each other with a prefix-sum table, which is stored in the structure index.
- **Spans keep quote selectors; the reader index does not need them.** The reader index is derived and rebuilt with each XML version, so section-UUID churn does not hurt it. Concept spans are curated and reviewed, so they must survive rebuilds, which needs the quote selectors.
- **The API serves both encodings** for spans: layer code points for Python and SQL, and (node, UTF-16 offset) for the browser.
- **One shared fixture test** pins the conversion. The reader plan already proposes a pytest and vitest fixture with curly quotes, `&amp;`, `§` and a non-BMP character.

**Shared with the smart reader.** The reader's live TOC, definitions and cross-references index the same leaves. One projection function, one hash and one structure index should serve both plans. A definition found by the reader then has the same coordinates as a concept span, and a cross-reference like "Section 8.1(c)" resolves to a range in the layer that a termination-fee trigger value can point to.

### 6.2 Selectors: how a span is stored

Each span stores both of the following, following the W3C Web Annotation selector model:

- **Position:** `layer_hash`, `start`, `end`.
- **Quote:** `exact` (the text itself), `prefix` and `suffix` (64 characters each).
- **Structure hint:** `section_uuid` and the subsection path (`8.3(b)(i)`), used for display and as a re-anchoring hint. It is never the identity.

### 6.3 Re-anchoring after an XML rebuild

When a new XML version is verified, every span on the previous version is re-anchored, in this order:

1. **Same `layer_hash`:** offsets carry over unchanged.
2. **Exact quote occurs once** in the new layer: move to it.
3. **Several matches:** pick the match whose prefix or suffix matches.
4. **Still ambiguous:** pick the match nearest the old offset scaled by the change in length.
5. **No exact match:** fuzzy match (for example diff-match-patch, threshold tuned on the pilot). The span is marked `anchor_status = fuzzy` and its values go to the review queue.
6. **Nothing found:** mark `orphaned`, keep the row, and exclude it from serving.

**Pilot result.** 50 version pairs, from 40 agreements with more than one XML version (read-only dump; script at `docs/plans/span-taxonomy/anchor_pilot.py`), 2,000 sampled sentence-length spans:

| Measure | Result |
|---|---|
| Version pairs with identical text layers | 32 / 50 (64%) |
| Sections whose UUID survived the rebuild | 4,523 / 4,808 (94.1%) |
| Span still correct at its **raw offset** | 1,739 / 2,000 (87.0%) |
| Span still correct at its **section-relative offset** | 1,855 / 2,000 (92.8%) |
| Re-anchored by **unique exact quote** | 1,701 (85.1%) |
| …plus prefix/suffix context | +144 (7.2%) |
| …plus nearest-to-scaled-offset tiebreak | +146 (7.3%), 1 of them uncertain |
| **Re-anchored in total** | **1,991 / 2,000 (99.55%)** |
| Lost: text actually changed (needs fuzzy match or review) | 8 (0.4%) |

What the pilot shows:
- Offsets alone would silently corrupt about one span in eight after a rebuild.
- Section-relative offsets are no fix: in one agreement, 69 of 84 section UUIDs changed.
- Quote plus context plus a position hint is enough, and it fails loudly (lost, not wrong) when the text itself changed.
- Many quotes recur within an agreement (boilerplate repeated across articles), so the context and position steps are not optional.

### 6.4 Instances, spans, roles, overlap

```
provision_instance   one concept occurring once in one agreement
  ├── span (role = primary)        "Parent shall pay ... the Termination Fee"
  ├── span (role = amount)         "...an amount equal to $135,000,000..."
  ├── span (role = trigger) ×3     "terminated pursuant to Section 8.1(d)..."
  ├── span (role = definition)     "Termination Fee" means ...  (Article I)
  └── values {payor, payee, amounts[], triggers[], ...}
        each value → evidence span ids
```

- **Several spans per instance** cover provisions that cross sections and provisions that are not contiguous.
- **Several instances per section** cover sections that hold several provisions.
- **Spans may overlap and nest freely.** The MAE definition span contains seven carve-out spans, each its own instance of `ma.mae.carveout`, linked to the parent instance by `parent_instance_id`. Nesting is a relation between instances, not a constraint on offsets.
- **Boundaries:**
  - A span is the tightest contiguous text that carries its role, usually a sentence, an enumerated item or a defined-term definition.
  - Boundaries snap to whitespace or punctuation, never mid-word.
  - The primary span of an instance should be the operative sentence, not the whole section. A section-level fallback is allowed only with `role = section_fallback`, and is counted separately in evaluation.
- **Instance identity.** Each instance has a permanent UUID. Extraction runs propose instances. A new run matches each proposal to an existing instance (same agreement and concept, overlapping primary span) before creating a new one, so reviews and links survive reruns.
- **Status and provenance.** Each instance has a status (`predicted`, `accepted`, `corrected`, `rejected`) and records the source: run ID, model, prompt hash, or `human:<reviewer>`. Serving shows predicted and accepted rows with a status flag. Aggregations can be restricted to accepted rows, or to rows above a confidence threshold.

## 7. Value schemas (first 10 concepts)

### 7.1 Shared value types

| Type | Fields |
|---|---|
| `Money` | `amount` (decimal), `currency` (ISO 4217), `basis` (`fixed`, `pct_of_equity_value`, `pct_of_enterprise_value`, `pct_of_purchase_price`, `formula`, `other`), `pct` (number or null), `raw` (verbatim text) |
| `Duration` | `value`, `unit` (`days`, `business_days`, `months`, `years`), `anchor` (what the clock runs from, e.g. `signing`, `closing`, `termination`, `statute_of_limitations`), `offset_days` (for "60 days after the statute of limitations") |
| `DateRule` | `date` (ISO, if fixed), `rule` (Duration from an anchor, if relative) |
| `PartyRole` | normalized role (`target`, `buyer`, `parent`, `merger_sub`, `seller`, `sponsor`, `guarantor`, `either`, `each`), plus the agreement's own defined term (`"Parent"`, `"Purchaser"`) |
| `XRef` | the raw reference (`"Section 8.1(c)"`) plus the resolved range in the text layer (from the smart reader's cross-reference index), plus the concept instance it lands on, if any |
| `Evidence` | a list of span IDs. Required on every field that is not null. |

Every field has a three-way absence: `null` (not extracted), `not_stated` (the agreement is silent) or `not_applicable`. Aggregation must tell "no cap" apart from "we didn't find a cap". At instance level, a concept that the agreement affirmatively lacks (no financing condition) is an instance with `finding = absent` and a `scope_reviewed` list of ranges. It has evidence in place of a span.

**Derived fields** are computed in code, never by the model, and flagged `derived: true` with their inputs recorded. The main example is `pct_of_equity_value_computed = fee.amount / deal.equity_value`. Equity value is often not stated in the agreement; it can be derived from per-share price × fully diluted shares, or taken from `tx_metadata` (`transaction_price_total`), which has known errors (see the SPAC acquirer note). So the derived percentage records which source it used, and aggregation can exclude low-trust sources.

**Enums versus open fields.** An enum is a promise that the value set is known and stable. It pays off only when people will group or filter by the field. Each field gets one of four shapes:

| Shape | Use when | Examples |
|---|---|---|
| **Closed enum** | The set is fixed by law, by arithmetic or by a structural fact, and nothing realistic falls outside it | currency (ISO 4217), Duration `unit`, outside-date `effect`, basket `type`, `election` (338(h)(10), 336(e), 338(g) are the only elections the Code offers), remedy exclusivity, the absence state |
| **Coded with verbatim text** | Users will aggregate on it, but drafting has a long tail and the pilot already needed new values | trigger `kind`, MAE carve-out `categories[]`, survival `item`, indemnity `applies_to` / `excludes`, extension `condition`, tier `label`, reverse-fee `kind`, Money `basis`, outside-date `breach_bar`, `efforts_standard` |
| **Number or structure** | The variation is quantitative or relational, so an enum would bucket away the information | tail `threshold_pct` and `months`, Durations, payor as a list of PartyRoles plus `liability` (`joint_and_several` or `several`) instead of a `joint` enum value, `deemed_as` as a link |
| **Open text** | People read the field more than they count it, or the variation is in the conditions | `applies_when`, trigger `conditions`, `rwi_interaction`, the MAE `standard` phrase, election `conditions` |

Rules for the coded shape:
- The field is stored as `{code, text}`. `text` is the shortest verbatim phrase that justifies the code, always kept, so a later recode never needs another model pass.
- `other` is allowed, and an `other` value must carry `text`. No `other_text` enum values.
- The extractor may return `proposed:<snake_case>` for a value that fits nothing. Proposals and `other` texts are reviewed each release. When a cluster reaches about 2% of a field's instances, it becomes a code, which is a minor release with a crosswalk.
- An `other` rate above about 5% for a field fails the release check. That is the signal that the enum is wrong, not the extractor.
- Code lists live in the taxonomy YAML next to the concept, each value with a one-line definition and a positive and negative example. The prompts and validators read the same file.
- Aggregations always show the `other` and `proposed` counts beside the coded buckets.

Ordinal codes are worth the effort where a scale is real. `efforts_standard` (`hell_or_high_water` > `reasonable_best` > `commercially_reasonable` > `other`) and `breach_bar` (`contributed` > `proximate_cause` > `principal_cause`) both carry an `ordinal` so queries can ask for "at least as strict as".

In evaluation, closed enums and codes are scored exactly. Open text is scored by whether its evidence span covers the gold span, not by string match.

### 7.2 The concepts

**1. `ma.termination.fee.target`: target termination fee** (DealOS `cra:stub-termination-fees`)
- `payor`, `payee`: lists of PartyRole, with `liability` (`joint_and_several` or `several`) when more than one party pays.
- `tiers[]`: each tier has `label` (`standard`, `go_shop`, `window_shop`, `superior_proposal`, `other`), `amount` (Money), and `applies_when` (text plus an XRef).
- `triggers[]`, each with:
  - `kind`: one of `superior_proposal_termination`, `adverse_recommendation_change`, `recommendation_failure`, `breach_of_no_shop`, `vote_failure_with_tail`, `outside_date_with_tail`, `breach_with_tail`, `other`;
  - `termination_right` (XRef);
  - `tail`, set per trigger: `months`, `threshold_pct` (often 50% rather than the 15–20% in the definition), and `requires` (`definitive_agreement`, `consummation`, `either`);
  - `remedy`, set per trigger: `exclusive`, `election_fee_or_damages`, or `non_exclusive`, with carve-outs (`fraud`, `willful_breach`);
  - `deemed_as`: a link to another trigger;
  - `conditions`: text for compound conditions.
- `payment_timing`: Duration plus anchor, or `concurrent_with_termination`.
- `expense_reimbursement_credit`: whether expenses are credited against the fee.
- `pay_once_only`: whether the fee is payable on only one occasion.
- `pct_of_equity_value_computed`: derived.

**2. `ma.termination.fee.reverse`: reverse termination fee** (DealOS `cra:stub-reverse-termination-fee`)

Everything in concept 1, with an added trigger kind `failure_to_close`, plus:
- `kind`: `financing`, `regulatory`, `general`, or `two_tier`;
- `guarantee`: `limited_guarantee`, `equity_commitment`, or `none`, plus the guarantor;
- `liability_cap`: the formula for the cap on buyer liability, as Money or text ("fee plus reimbursement obligations"), or `none` (DealOS `limits_amount_under` → `cra:124`);
- `specific_performance_interaction`: `exclusive_alternative`, `both_available`, `conditional`, or `not_stated`.

**3. `ma.termination.expense_reimbursement`: expense reimbursement on termination**
- `payor`, `payee`;
- `cap`: Money;
- `triggers[]`, as in concept 1;
- `credited_against_fee`.

Three sibling label concepts keep other reimbursements out of this one: `ma.financing.cooperation_costs`, `ma.termination.collection_costs` and `ma.regulatory.filing_fee_allocation`.

**4. `ma.termination.outside_date`: outside date** (DealOS `cra:4`, `cra:114`)
- `initial`: DateRule.
- `effect`: `termination_right` or `automatic_termination`.
- `extensions[]`: each has `length` (Duration or text, when relative to a variable such as a notice period), `new_date` (DateRule), `condition` (`regulatory_pending`, `marketing_period`, `sec_review`, `notice_period`, `other`), `exercise` (`automatic`, `either_party`, `buyer`, `target`, `mutual`), `count`.
- `breach_bar`: coded (`none`, `principal_cause`, `proximate_cause`, `contributed`, `other`) with verbatim text and an ordinal. This is the standard by which a breaching party loses the right to terminate.
- `days_from_signing_computed`: derived.

**5. `ma.mae.definition`: material adverse effect definition** (DealOS `cra:82`)
- `defined_term`;
- `party` (whose MAE: `target`, `buyer`, or `both`);
- `prongs`: `business_condition`, `ability_to_consummate`, `prospects_included`;
- `standard` ("would reasonably be expected to");
- `carveouts[]`: child instances of `ma.mae.carveout`.

**`ma.mae.carveout`: MAE carve-out** (label concept with a few values)
- `categories[]` (one clause often bundles several): `general_economic`, `financial_markets`, `industry`, `change_in_law`, `change_in_gaap`, `political_war_terrorism`, `pandemic`, `natural_disaster`, `announcement_or_pendency`, `compliance_with_agreement`, `failure_to_meet_projections`, `stock_price_or_volume`, `buyer_action_or_request`, `other`;
- `disproportionate_effect_exception`: true or false.

**6. `ma.financing.structure`: financing condition and related terms**
- `financing_condition`: true or false. True is rare in public deals; its absence is the common finding, recorded as `finding = absent` with the scope reviewed (see pilot finding 8).
- `committed_financing`: `debt`, `equity`, or `none`, plus `sources[]`.
- `efforts_standard`: coded with an ordinal (see the enum policy in 7.1).
- `marketing_period`: Duration, with blackout dates.
- `financing_cooperation`: present or absent.
- `rtf_link`: the instance ID of the financing reverse termination fee.

**7. `ma.indemn.cap`: indemnification cap** (DealOS `pcra:35`)
- `caps[]`: each has `applies_to[]` and `excludes[]`, both drawn from `general_reps`, `fundamental_reps`, `tax`, `covenants`, `specific_indemnity`, `fraud`, `willful_breach`; `amount` (Money; often `pct_of_purchase_price`); `indemnitor`. Buyer-side and seller-side caps are separate entries.
- `escrow_sole_recourse`: true or false.
- `rwi_interaction`: text.

**8. `ma.indemn.basket`: basket / deductible** (DealOS `pcra:38`)
- `type`: `deductible`, `tipping`, or `none`;
- `amount`: Money;
- `de_minimis`: a Money threshold per claim; whether claims below it count toward the basket; and whether the threshold decides if a breach exists at all;
- `applies_to[]`;
- `indemnitor`.

**9. `ma.indemn.survival`: survival periods** (DealOS survival architecture)
- `periods[]`: each has `item` (`all_reps`, `general_reps`, `fundamental_reps`, `tax_reps`, `covenants_pre`, `covenants_post`, `other`) and `period` (a Duration, `indefinite`, or `statute_of_limitations_plus` with an `offset_days`).
- `no_survival`: the public-deal norm, where representations do not survive closing.

**10. `tax.election.338h10`: 338(h)(10) / 336(e) election** (DealOS `pcor:9`)
- `election`: `338h10`, `336e`, `338g`, `prohibited`, or `none`;
- `decides`: `buyer`, `seller`, `joint`, or `mandatory_joint`, plus `conditions` (text);
- `seller_must_join`;
- `allocation`: who prepares it, the deadline, the dispute mechanism, and the Section 1060 reference;
- `tax_cost_shift`: `payer`, `payee`, `scope` (`incremental_federal`, `incremental_state_local`, `all`) and a formula reference. The direction is explicit: in SkyWest, the sellers pay the buyer;
- `deemed_sale_tax_borne_by`.

**Exclusions.** Schemas cover what is stated in the agreement. Deal metadata (equity value, signing date) is joined in, never extracted again. Amendments that change a fee are out of scope for v1; flag them as an open question.

### 7.3 Pilot findings on the schemas

A subagent annotated six agreements against a draft (v0) of these schemas, working on the text layer. Offsets were computed in code, and a validator checked every quote against the text.

| Agreement | Kind | Instances | Spans | Multi-section instances |
|---|---|---|---|---|
| Atlantic Southeast Airlines / SkyWest | private stock purchase, 338(h)(10) | 11 | 14 | 0 |
| Cerner / Oracle (2021) | public tender offer and merger | 6 | 20 | 2 |
| W. R. Grace / Standard Industries (2021) | public merger | 7 | 27 | 4 |
| Conexant / Synaptics | private membership-interest purchase | 18 | 28 | 2 |
| Kadmon / Sanofi (2021) | public merger | 4 | 11 | 1 |
| Danka / Konica Minolta | private stock purchase | 9 | 19 | 3 |
| **Total** | | **55** | **119** | **12** |

- 12 sections held more than one concept.
- 27 carve-out ranges were nested inside the 6 MAE definition spans.
- Exactly 2 span pairs overlapped, both in Danka: one clause is both a fee trigger and the outside date.
- A careful pass took about 15,000–25,000 model tokens per agreement for these concepts. That is consistent with the stage 3 estimate in [section 8.3](#83-cost-per-corpus-pass).

The annotation files themselves were lost: the scratch folder was cleaned up while the run was still finishing. The counts and findings here come from the subagent's report, written before that. The pilot is cheap to redo on these six agreements in M1.

**What the pilot showed**, and the changes already folded into [section 7.2](#72-the-concepts):

1. **Provisions cross sections routinely.** 12 of 55 instances span more than one section, including every fee and reverse-fee instance in the public deals. Values also sit in other sections as defined terms: in Synaptics the indemnity cap is the "Indemnification Escrow Amount", and "Fundamental Representations" are defined in the closing conditions. Value evidence therefore needs a `resolves` link from the usage to the definition span. The worst instance took 9 spans (the Grace termination fee: amount, three triggers, two exclusivity statements, a tail, a deeming rule). The multi-span instance model is required, not optional.
2. **Triggers are not independent enum values.**
   - Tail windows differ by termination ground (Grace: 6 months after an End Date termination, 12 months after a vote failure or breach).
   - Remedy exclusivity differs by trigger (Grace: the fee is exclusive for a recommendation change, but on the tail trigger Parent may elect the fee *or* willful-breach damages).
   - One trigger can be deemed to be another (a vote failure is treated as a recommendation-change termination).

   So `tail`, `remedy` and a `deemed_as` link now live on each trigger, plus a free-text `conditions` field for compound conditions such as "primarily attributable to". A new kind, `recommendation_failure`, covers a failure to reaffirm or reject, as in Cerner, which is not a change of recommendation.
3. **Exhibits hold values the body cites.** Missing from the body-only layer:
   - Grace's per-share merger consideration and "Marketing Period" (both defined in Exhibit A);
   - Cerner's tender-offer conditions, which are where any financing condition would be (Exhibit B).

   Body-only text is not enough for value concepts. [Open question 3](#15-open-questions-for-you) now recommends exhibits in v1.
4. **Equity value is never stated as one number**, in any of the six agreements.
   - Cerner's fee works out to about 3.4% and Kadmon's to about 3.65%, using basic share counts from the capitalization representation.
   - Grace's cannot be computed without Exhibit A.
   - Danka's fee is 2.08% of purchase price, a different basis.

   A `ma.deal.value` concept (price per share, share basis, stated purchase price) should feed the derived percentage. This confirms that `pct_of_equity_value_computed` is a derived field, computed in code, with its source recorded ([open question 5](#15-open-questions-for-you)).
5. **Duplicated definitions were a projection bug, not an XML defect** (corrected 2026-10-21). The first projection treated `<definition>` as a leaf, so it emitted each definition's whole text, `<pageUUID>` values included, and then each of its `<text>` lines again. With the fix, a 400-agreement random sample has no duplicated definitions. 25 of the 400 have 2–10% repeated long lines, and those checked are genuine mirrored representations (sections 3.16 and 4.16 for the two parties). The re-anchoring numbers in 6.3 were measured on the buggy layer, so they are, if anything, conservative.
6. **"Expense reimbursement" conflated four things:** fee reimbursement on termination, reimbursement of financing-cooperation costs (payable whether or not the deal closes), costs of collecting an unpaid fee, and a split of HSR filing fees. They are now separate concepts: `ma.termination.expense_reimbursement`, `ma.financing.cooperation_costs`, `ma.termination.collection_costs`, `ma.regulatory.filing_fee_allocation`.
7. **Booleans hide structure.** These booleans became structured fields:
   - the reverse fee's cap on buyer liability: a formula, not a yes/no (Grace: the fee *plus* uncapped reimbursements);
   - `exclusive_remedy`: an enum with an election option;
   - the outside-date breach bar: a causation standard ("principal cause");
   - the 338(h)(10) `gross_up`: a `tax_cost_shift` with a payer. In SkyWest the gross-up runs the other way: sellers pay the buyer's incremental state tax as a condition of the buyer joining the election.
8. **Absence is a finding.** "No financing condition" and "representations do not survive" have no positive span. They come from reading the conditions article and finding nothing, or from a non-survival clause. Instances can now carry `finding = absent`, recording the scope reviewed (for example, the conditions article and Exhibit B), so "absent" is kept apart from "not extracted".
9. **Enumerations needed more values:**
   - outside date: `effect = termination_right` or `automatic_termination` (SkyWest terminates automatically);
   - survival: an `indefinite` value;
   - MAE carve-outs: `categories[]` rather than one category, since one clause often bundles economic, political and regulatory changes;
   - indemnity cap: `applies_to` plus `excludes` lists ("all reps, but not covenants or the tax indemnity");
   - de minimis: a flag for when the threshold decides whether a breach exists at all, not just whether it counts toward the basket;
   - reverse fee: a `failure_to_close` trigger. Financing failure is never named as a trigger; it shows up as the buyer's failure to close;
   - payor: joint payors (Danka: Seller and Parent), now a list of parties rather than an enum value;
   - fee amounts: alternative amounts, as distinct from tiers;
   - Money `basis`: `pct_of_market_cap`;
   - MAE carve-outs: `stockholder_litigation`, `credit_rating` and `disclosure_schedule_items`, plus room for exceptions to carve-outs and alias defined terms;
   - survival: a fixed calendar date;
   - outside date: a second, conditional long-stop date (Danka), and breach bars that differ by party;
   - indemnity: a `materiality_scrape` field and a tax-indemnity cap tier;
   - 338(h)(10): which entities the election covers (Synaptics elects for one subsidiary only).
10. **Overlap stays modest at this grain, and nesting is common.** Only 2 span pairs overlapped. Nesting appeared 27 times (carve-outs inside MAE definitions). Both are supported by the model, but neither is the hard part. Cross-section instances and value resolution through defined terms are.

The annotations reached the role vocabulary through use. In the first three agreements alone, 15 roles appeared; the common ones were `primary`, `trigger`, `extension`, `exclusive_remedy`, `carve_out`, `cap`, `guarantee` and `sources`. Roles become a small controlled list per concept in the YAML, not free text.

### 7.4 M1 pilot results (2026-10-21)

**Setup.** 50 verified agreements, stratified as 36 public-target and 14 private-target deals across four eras (before 2008, 2008–15, 2016–20, 2021 on). Three concepts: target fee, reverse fee, outside date. Everything ran as subagents in one Claude Code session, with no API calls:
- **Stage 1 (code):** keeps sections whose title, article or text matches fee, termination or outside-date patterns, plus definitions of matching terms. Median 9 sections and 46K characters per agreement, about 19% of the body.
- **Extractors, two arms:** Sonnet and Haiku, each seeing only the stage 1 candidates.
- **Gold:** Opus, independently, searching the full body.
- **Adjudication:** Opus judged every field where gold and the Sonnet arm disagreed, and fixed gold where it was wrong.

Code and numbers are in `docs/plans/span-taxonomy/pilot/`.

| Measure | Sonnet | Haiku |
|---|---|---|
| Presence, 150 agreement × concept calls | 150/150 | 149/150 (one capped expense reimbursement read as a fee) |
| Quotes found verbatim in the text | 99.4% of 1,351 | 99.4% of 1,176 |
| Target fee amount | 33/34 | 31/34 |
| Outside date: initial date / effect / number of extensions | 49/49, 49/49, 47/49 | 48/49, 49/49, 46/49 |
| Target fee triggers: exact set / set F1 | 18/34 / 0.93 | 16/34 / 0.89 |
| Reverse fee triggers: exact set / set F1 | 14/21 / 0.84 | 12/21 / 0.74 |
| Reverse fee `kind` / `payor` | 16/21, 14/21 | 13/21, 19/21 |
| Remedy (target / reverse) | 29/34, 18/21 | 28/34, 18/21 |
| Extension conditions / breach bar | 42/49, 41/49 | 40/49, 40/49 |
| Operative spans found (target / reverse / outside date) | 60/64, 34/35, 50/50 | 51/64, 33/35, 49/50 |
| Trigger and remedy spans found | 94/238, 38/85 | 69/238, 28/85 |
| Subagent tokens for the 50 agreements | 1.50M | 2.45M |

Stage 1 retrieval kept 612 of 628 gold spans (97.5%). Its misses were 5 fee definitions in a separate definitions section and 11 peripheral spans.

**Adjudication.** 56 disputed fields were judged:
- 30 times gold was right;
- once the extractor was right;
- twice neither was;
- 23 times the schema was ambiguous: both readings were defensible because a code or rule was undefined.

Gold was corrected in 4 places. So the gold labeler is reliable on what the text says, and **most of the remaining disagreement comes from the schema, not from the models.**

**Against the acceptance bar** ([section 9](#9-evaluation-plan)):
- **Passes:**
  - presence detection (Sonnet: precision and recall 1.0 on all three concepts);
  - the values people aggregate on: fee amount, outside date, effect and number of extensions;
  - quote validity.
- **Fails:**
  - the coded multi-value fields: triggers, reverse-fee kind, remedy, extension conditions and breach bar;
  - span recall for trigger and remedy roles. Extractors marked the clause that imposes the fee but seldom the termination rights it refers to, because the schema never asked for one span per trigger.
- **Model choice:** Haiku was less accurate on most coded fields (reverse-fee payor was the exception), yet used more tokens than Sonnet, because it read more. A small model should be measured on the narrow stage 2 job (does this passage belong to the concept?), not on full extraction.

**Schema v2: rules from the adjudication.**
1. **Breach bar becomes two fields.**
   - `conduct`: `breach`, `failure_to_perform`, `any_action_or_inaction`, `willful_breach`.
   - `causation`, an ordinal: `principal_or_primary_cause`, `cause_or_result`, `no_causation_required`, `none`.

   The one enum could not hold the most common wording, "has been the cause of, or resulted in", and labelers split it between `principal_cause` and `other`.
2. **PartyRole records the side.** `side` is `target_side` or `buyer_side`, plus `entity` (`parent`, `merger_sub`, `sponsor`, …) and the defined term. "Parent" was coded `buyer` in 8 gold files and `parent` in 26.
3. **One remedy rule:**
   - `exclusive` when the text says the fee is the sole or exclusive remedy, or that termination releases all liability except the fee;
   - `non_exclusive` only when the text says the fee is in addition to other remedies, or cumulative;
   - otherwise `not_stated`, with the carve-outs listed.

   Two adjudicators gave opposite rules for the silent case.
4. **Reverse-fee `kind` gets definitions** and a `reciprocal` code, for a fee both sides owe on mirrored triggers.
5. **Trigger rules:**
   - a tender offer that expires with its minimum condition unmet counts as an approval failure (rename `vote_failure_*` to `approval_failure_*`);
   - one termination right that covers several events emits every matching code;
   - a fee payable on a recommendation change without termination gets its own code;
   - each trigger needs its own `trigger` span on the termination right.
6. **Amounts are always a list.** An amount defined outside the body is an entry with `amount: "not_stated"`, never an empty list.
7. **"Or such later date as the parties agree" is not an extension.**

**No exhibits has a cost.** In 3 of 50 agreements, the fee amount or the outside date is defined outside the body, so those values are `not_stated`. In two the definitions sit in Exhibit A. In the third, the agreement's own defined-terms article is missing from the body, which looks like a page-classification error worth checking. If that rate holds, about 6% of agreements lose a headline value. A defined-terms exhibit is part of the operative agreement, so the cheapest fix is to admit exhibits titled as definitions while keeping other exhibits out.

**Equity value and fee percentages.** Equity value was found for 32 of 34 deals with a target fee: 4 from the agreement, 28 from web sources with URLs, and 2 from `tx_metadata`. `tx_metadata` agreed within 5% in 25 of 32. On the 27 deals with an equity-basis value:
- The target fee was a median **3.5%** of equity value, and 1.4–4.9% in 26 of them.
- The outlier, Anywhere Real Estate at 12.5%, is a heavily levered target, where the fee is about 2% of enterprise value.
- In one more deal (OM Group), the lookup matched a different deal: a carve-out sale instead of the parent merger.

Two rules follow. Store the percentage against both equity value and enterprise value, with their sources. Flag any fee above 8% of equity value for review.

**Next step.** Detection, retrieval and the core values meet the bar, and the failures trace to the seven schema rules above rather than to model limits. So M2 should start with schema v2. Its 50-agreement dev split re-checks these three concepts under v2 before the other seven concepts are scored.

### 7.5 M2 pilot results (2026-10-21)

**Setup.** 150 new verified agreements, none from M1, stratified as 75 public and 75 private targets across the same four eras (18 or 19 per cell). Every third agreement went to dev (50); the other 100 are the sealed test split. All 10 concepts of [7.2](#72-the-concepts), as subagents with no API calls:
- **Stage 1 (code):** four concept groups (termination, deal risk, indemnification, tax). Long definitions sections are trimmed to relevant terms. Median 19 sections and 67K characters per agreement, about 30% of the body.
- **Extractor:** Sonnet, seeing only the stage 1 candidates.
- **Gold:** Opus, independently, searching the full body.
- **Adjudication:** Opus judged every field where gold and Sonnet disagreed, and fixed gold where it was wrong.

The dev split ran under schema v2. Its disputes became 14 clarifications (schema v2.1, `pilot/schema_v2_1.md`), which governed the test labels and all adjudication. Stage 1 was also fixed on dev (inline MAE definitions) before test candidates were built; nothing was changed after the test split was labeled.

One test agreement (GovDelivery) was dropped: its XML has text but no sections, so the text layer projected an empty body. The test numbers below cover 99 agreements.

**Test split, Sonnet against adjudicated gold.**

| Concept | Present / 99 | Precision | Recall | Key fields (exact match) |
|---|---|---|---|---|
| Target fee | 53 | 1.00 | 0.98 | amounts 49/52, payor 51/52, triggers 40/52 (set F1 0.95), remedy 45/52 |
| Reverse fee | 27 | 0.93 | 0.96 | amounts 25/26, kind 24/26, triggers 22/26 (set F1 0.95), remedy 26/26 |
| Expense reimbursement | 26 | 1.00 | 0.92 | cap 23/24, payor 24/24 |
| Outside date | 96 | 1.00 | 0.98 | initial date 92/94, effect 91/94, extensions 92/94, breach-bar causation 84/94 |
| MAE definition | 99 | 1.00 | 0.96 | carve-out categories 72/95 (set F1 0.96), ability-to-consummate prong 89/95, disproportionate-effect exception 90/95 |
| Financing structure | 77 | 0.98 | **0.70** | committed sources 40/54, financing condition 51/54, efforts 53/54 |
| Indemnity cap | 42 | 1.00 | 0.98 | cap amounts **25/41**, escrow sole recourse 33/41 |
| Indemnity basket | 35 | 1.00 | 0.94 | baskets 30/33, de minimis 32/33 |
| Survival | 97 | 1.00 | 0.99 | general reps **75/96**, fundamental reps 90/96 |
| 338(h)(10) / 338(g) | 15 | 1.00 | 1.00 | election 15/15, who decides 11/15, tax cost payer 11/15 |

- Quotes were found verbatim in 99.8% of 4,808 (99.3% of 2,360 on dev).
- False positives: 3 instances in 99 agreements (0.03 per agreement).
- Span recall by concept was 0.73–0.94, precision 0.85–1.00.
- Cost: about 60K Sonnet subagent tokens per agreement for all 10 concepts, and a similar amount of Opus for gold.

**Adjudication.** 210 disputed test fields were judged:
- 138 times gold was right;
- 34 times the two said the same thing in different encodings;
- 23 times the schema was ambiguous;
- 13 times the extractor was right, and twice neither was.

So gold was corrected in 15 of the 2,082 compared fields. Schema ambiguity fell from 41% of disputes in M1 to 11% here, so schema v2 and v2.1 did their job.

**Where the extractor's errors come from.** Of 140 test fields the extractor got wrong:

| Cause | Fields |
|---|---|
| Retrieval gap: the text was not in the stage 1 candidates | 83 |
| Wrong code | 17 |
| Wrong value | 16 |
| A v2.1 rule not applied | 12 |
| Text was in the candidates but missed | 7 |
| Over-inclusion | 5 |

**Retrieval is now the bottleneck.** Stage 1 kept 94.1% of gold spans on test (89.9% on dev, before the inline-MAE fix), and the misses explain most of the failing rows above:
- **Financing (50 of 118 missed spans).** v2.1 rule 3 makes a "sufficient funds" or "financial capability" rep enough for financing to be present. Stage 1 does not look for those titles, so most of the 23 missed financing instances never reached the extractor.
- **Indemnity limits (21).** Caps and baskets often sit in sections titled "Limitations", "Limits on Claims", "Maximum liability" or "Minimum Claims", which no pattern matched. That explains 6 of the 16 wrong cap amounts; the rest are coding errors (percentage-of-price caps, special-indemnity caps) spread across v2.1 rules.
- **A trimming bug.** The rule that trims long definitions sections also fired on 10 sections that merely contain five or more inline definitions, including three termination sections whose trigger lists were cut. It should apply only to sections titled as definitions.
- **Small misses:** "Section 4.05Financing." (no space, so `\bfinancing` fails); numbered titles with no words (one Dutch agreement kept 1 of 72 sections); MAE definitions in an annex outside the body.

Survival general reps (75/96) is not a real failure. In 19 of its 21 disputes both sides set `no_survival: true`, and one also added a redundant `all_reps` period of kind `none` or `until_closing`; adjudicators called these equivalent. The fix is a normalization rule (rule 6 below), not a better model.

**Against the acceptance bar** ([section 9](#9-evaluation-plan)), on test:
- **Passes on detection** (precision and recall ≥ 0.95, at most 0.1 false positives per agreement): target fee, outside date, MAE definition, indemnity cap, survival and 338.
- **Narrowly fails detection:** reverse fee (precision 0.93: a ticking fee and a merger-of-equals fee were counted as reverse fees), basket (recall 0.94) and expense reimbursement (recall 0.92). These rest on two or three agreements each.
- **Fails:** financing (recall 0.70, almost all retrieval).
- **Key fields at ≥ 0.97:** outside-date initial date, the 338 election, reverse-fee remedy and payor, and expense payor pass. Target-fee payor (0.98) passes and amount (0.94) is close. Fee triggers (set F1 0.95, exact set 0.77) and cap amounts (0.61) fail.

**Schema v2.2: the open rules.** The 23 ambiguous disputes reduce to these:
1. A general "remedies are cumulative" or "no further liability" clause that does not mention the fee does not set the fee's `remedy`; it stays `not_stated`. (Adjudicators ruled this way every time, but the rule is unwritten.)
2. When a 338 election is prohibited, `decides` is `not_applicable`.
3. Reverse-fee `kind` gets `mixed` for one fee with financing, regulatory and breach triggers.
4. `breach_bar` becomes a list, so an outside date with two bars records both. "Proximately caused" or "contributed" maps to `cause_or_result`.
5. `none` means the provision exists without that element; `not_stated` means the element exists but is defined outside the text. This settles de minimis, breach bar and survival.
6. Survival merges `none` and `until_closing`, and when `no_survival` is true the reps are not listed again in `periods`.
7. A buyer deposit kept as liquidated damages is a reverse fee (`kind: deposit`). A "ticking" payment owed while the deal is pending is not a fee.
8. A fee of a fixed amount plus capped expenses is one fee amount plus an expense reimbursement with `credited_against_fee: false`.
9. In a de-SPAC, the trust account with a minimum-cash condition is `cash_on_hand`, and a PIPE is committed equity.
10. A fee trigger keyed to a defined term outside the text is `other`, with the term recorded.

**Equity value and fee percentages (test).** Equity value was found for 49 of the 50 deals with a target fee: 11 from the agreement and 38 from web sources with URLs. On the 45 deals with a USD fee and a USD equity value:
- The target fee was a median **3.5%** of equity value, with the 10th to 90th percentile at 2.8–4.1%.
- None was above 8%, and the full range was 1.3–5.0%.

That matches M1 (median 3.5%) on an independent sample, so the 8% review flag stands.

**Next step.** The extractor is no longer the main problem; stage 1 is. Before a corpus pass:
1. Fix stage 1: add the funds-rep and indemnity-limit title patterns, restrict trimming to definitions sections, and fall back to text patterns when titles are bare numbers.
2. Admit defined-terms exhibits and annexes into the text layer, as [7.4](#74-m1-pilot-results-2026-10-21) proposed. Several MAE definitions and outside dates sat there.
3. Write the ten v2.2 rules.
4. Re-run stage 1 and the extractor on the test split and re-score against the same gold, which isolates the retrieval gain. Some of these fixes were found by reading test errors, so that score is optimistic. The pass/fail call for M3 should come from a fresh holdout of about 50 agreements, gold-labeled the same way.

Only concepts that then pass the bar go into the M3 corpus pass.

### 7.6 M2 holdout results (2026-10-21)

**Setup.** Steps 1–3 of the M2 next step were done first, then a fresh holdout was drawn and labeled. Step 4 (re-scoring the old test split) was skipped in favor of the holdout, which gives the unbiased number.
- **Sample:** 50 new verified agreements, none from M1 or M2: 25 public and 25 private targets across the four eras (7 per era-a cell, 6 per other cell).
- **Stage 1 (`prepare_v3.py`):**
  - adds funds-rep and financial-capability titles and text, and finds "Financing" glued to a section number;
  - adds indemnity-limit titles (limitations, maximum liability, minimum claims, cap, threshold, escrow);
  - matches a number-only title on the section's first line;
  - trims only sections titled as definitions.
- **Text layer (`text_layer.project(exhibits=True)`):**
  - admits defined-terms exhibits and annexes from back matter (12 of 50 agreements gained one);
  - splits body text that sits outside any section into synthetic sections, so a GovDelivery-type agreement is no longer empty.
- **Candidates:** median 21 sections and 85K characters, about 30% of the text.
- **Schema v2.2 (`pilot/schema_v2_2.md`):** v2.1 plus 13 clarifications (the ten in [7.5](#75-m2-pilot-results-2026-10-21) and three more).
- **Roles:** the same as M2. Sonnet extracts from candidates; Opus writes gold from the full text; Opus adjudicates every disputed field and fixes gold where it is wrong.

**One defect found mid-run, scored both ways.** In an admitted exhibit, only the line that opens a definition ("X" means …) carried the defined term. Its continuation lines, such as MAE carve-out clauses, did not, so definitions trimming dropped them.
- The fix keeps each continuation line with its term. It changes candidates only: the full text, and so gold, is identical.
- The 9 affected agreements were re-extracted. Both arms are scored below against the same adjudicated gold.
- The "as run" arm is the honest holdout number. The fixed arm shows what the fix buys.

**Holdout, Sonnet against adjudicated gold (as run).**

| Concept | Present / 50 | Precision | Recall | Key fields (exact match) |
|---|---|---|---|---|
| Target fee | 26 | 1.00 | 1.00 | amounts 26/26, payor 26/26, remedy 26/26, triggers 12/26 (set F1 0.92) |
| Reverse fee | 13 | 1.00 | 1.00 | amounts 12/13, kind 12/13, triggers 9/13 (set F1 0.87), remedy 13/13 |
| Expense reimbursement | 15 | 1.00 | 1.00 | cap 15/15, payor 15/15 |
| Outside date | 49 | 1.00 | 0.98 | initial date 48/48, effect 47/48, extensions 46/48, breach-bar causation 44/48 |
| MAE definition | 50 | 1.00 | 1.00 | carve-out categories 34/50 (set F1 0.91; **0.95 with the fix**), ability-to-consummate prong 45/50, disproportionate-effect exception 47/50 (49/50 with the fix) |
| Financing structure | 39 | 0.97 | **0.95** | committed sources 31/37, financing condition 34/37, efforts 34/37 |
| Indemnity cap | 23 | 1.00 | 1.00 | cap amounts **16/23** (set F1 0.84), escrow sole recourse 21/23 |
| Indemnity basket | 22 | 1.00 | 1.00 | baskets 21/22, de minimis 21/22 |
| Survival | 50 | 1.00 | 1.00 | general reps 48/50, fundamental reps 50/50 |
| 338(h)(10) / 338(g) | 5 | 1.00 | 1.00 | election 5/5, who decides 5/5, tax cost payer 4/5 |

- **Detection:** one false positive in 50 agreements (0.02 per agreement) and three misses, all before adjudication. Detection did not change in adjudication.
- **Financing recall:** 0.949, two misses in 39. It sits at the 0.95 bar, against 0.70 on the M2 test split.
- **Fixed arm:** identical except one reverse fee is missed. bfcd12e3 is an interest payment owed on termination. Adjudication called it a reverse fee, but the re-run read it as a ticking payment under v2.2 rule 7, which leaves this case open.
- **Quotes:** 99.7% of 2,608 were found verbatim.
- **Span recall:** 0.76–1.00 by concept, precision 0.89–1.00. As in M2, the misses are mostly extra supporting spans gold quoted (cure periods, definitions), not the operative clause.

**Adjudication.** 77 disputed items were judged as 82 field verdicts. Two items had fields that came out differently.
- Gold was right 60 times.
- The two said the same thing 4 times.
- The schema was ambiguous 8 times.
- The extractor was right 7 times, and neither was 3 times.

Gold was corrected in 10 fields. Ambiguity was 10% of verdicts, about the same as M2 (11%).

**Where the extractor's errors come from.** Of 63 fields the extractor got wrong:

| Cause | Fields | M2 test share |
|---|---|---|
| Retrieval gap | 26 (41%) | 59% |
| Text in candidates but missed | 13 | 5% |
| Wrong code | 12 | 12% |
| A v2.1/v2.2 rule not applied | 5 | 9% |
| Over-inclusion | 4 | 4% |
| Wrong value | 3 | 11% |

Stage 1 kept 96.8% of gold spans (94.1% on the M2 test split). Financing went from 50 of 118 missed spans to 12 of 34. The remaining retrieval gaps are a different kind from M2's. They are no longer missed sections but missed **definitions of terms the operative clause uses**:
- **Target-fee triggers (12/26).** In 7 of 14 disputes the fee is triggered by an "Adverse Recommendation Change". Its definition, which includes failing to recommend (v2.2 rule 12), sits in the no-shop covenant, and that section was not a candidate. Two more triggers are keyed to a "Triggering Event" defined elsewhere.
- **Cap amounts (16/23).** Three caps are "the Purchase Price" or "the Merger Consideration". The extractor coded these `other` or `formula`; adjudicators coded 100% of price, a rule that is not yet written. Three more are an escrow or holdback amount, or an enterprise value, defined in a section stage 1 did not keep.
- **MAE carve-outs.** 8 of 22 MAE disputes were retrieval gaps, mostly the cut-off exhibit definitions that the fix above repairs. 4 were ambiguous carve-out codes ("performance of this Agreement", "calamity").

Two other gaps remain. On 43a2d63d, the defined-terms exhibit is stored as whole-page text blobs, with several definitions on one line under an "A-1 EXHIBIT A DEFINITIONS" page header. The exhibit detector does not recognize it, so the outside date and MAE there are invisible to both gold and the extractor. On ac729f07, the long stop date appears only in a section title.

**Against the acceptance bar** ([section 9](#9-evaluation-plan)), on a holdout nothing was tuned on:
- **Passes on detection** (precision and recall ≥ 0.95, at most 0.1 false positives per agreement): all ten concepts. Financing recall is 0.949, which rounds to 0.95 but is two misses short of clear. The reverse fee, 338 and expense reimbursement rest on 13, 5 and 15 positives, too few to separate 0.95 from 1.00.
- **Key fields at ≥ 0.97:**
  - **Pass:** target-fee amount, payor and remedy; reverse-fee payor and remedy; expense cap and payor; the outside-date initial date; fundamental-rep survival; the 338 election.
  - **Fail:** fee triggers (set F1 0.92 and 0.87), cap amounts (0.84), MAE carve-outs (0.91, or 0.95 with the fix), breach-bar causation (0.92) and financing committed sources (0.93).

**Schema v2.3 candidates** from the adjudicators:
1. A cap equal to "the Purchase Price" or "the Merger Consideration", including per seller, is `pct_of_purchase_price` with pct 100.
2. A financing condition that protects only the target is not `financing_condition`. Whether `false` may be inferred when the conditions list has none (v2.1 rule 2 now forces `not_stated`) needs a ruling.
3. An interest payment owed on termination is a reverse fee. Rule 7's ticking payment is owed only while the deal is pending.
4. Plain "contributed to" in a breach bar is `cause_or_result`. A bar on terminating while a specific-performance action is pending is not a breach bar.
5. Terminating on delivery of a notice of intent to change the recommendation is its own trigger (`other`, with the term recorded).
6. Partial non-survival (general reps lapse, fundamental reps survive) sets `no_survival: true` with the surviving reps in `periods`.
7. Placement of a few MAE carve-outs: "general conditions affecting the industry" is `general_economic` unless industry changes are carved out separately; "performance of this Agreement" and "calamity or crisis" need a code.

**Next step.** The pass/fail call for M3:
- **Ready for the M3 corpus pass:** detection for all ten concepts, plus the passing key fields listed above.
- **Gated behind the next round:** fee triggers and cap amounts. Their remaining failures are one retrieval gap and one unwritten rule, not model limits.

That next round:
1. **Definition resolution in stage 1:** when a candidate section uses a defined term (Adverse Recommendation Change, Triggering Event, Indemnity Escrow Amount, Enterprise Value), also keep the section or definition that defines it.
2. **Exhibit detector:** handle page-blob exhibits (a page-number prefix such as "A-1" and several definitions on one line).
3. **Schema v2.3:** write the seven rules above.
4. **Re-check on this holdout:** for fee triggers, cap amounts and MAE carve-outs only. It is no longer blind for these fields, so the result is reported as a regression check, not a new pass/fail.

### 7.7 Definition-resolution re-check (2026-10-21)

**What changed.**
- **Definition resolution (`prepare_v4.py`):**
  - When a candidate section uses a relevant defined term whose definition is not already a candidate, stage 1 adds the defining paragraph, marked `groups=resolved`. Examples of such terms: Adverse Recommendation Change, Triggering Event, Indemnity Escrow Amount, Enterprise Value.
  - Definitions are found in three forms: definition leaves, inline `(… "X")` definitions, and `"X" means` / `"X" shall be deemed to have occurred` sentences.
  - One pass only.
  - All 10 definitions that [7.6](#76-m2-holdout-results-2026-10-21) found missing now reach the extractor.
  - Candidates grow by a median 7%, to 33% of the text (median 4.5 resolved sections per agreement).
- **Page-blob exhibits (`text_layer.project(page_blobs=True)`):**
  - An exhibit stored one page per `<text>` is admitted when its first page carries a definitions heading ("A-1 EXHIBIT A DEFINITIONS …").
  - A line that defines several terms records them all.
  - Counting definitions instead of requiring the heading would also have admitted a transition services agreement and a plan of arrangement, so it is not used.
  - On the holdout this changes one agreement (43a2d63d), whose Outside Date and MAE definitions now reach both gold and the extractor.
- **Schema v2.3 (`pilot/schema_v2_3.md`):** v2.2 plus 14 clarifications. They cover the seven candidates in [7.6](#76-m2-holdout-results-2026-10-21), plus:
  - how resolved definitions are used;
  - caps equal to a defined amount, and caps summed from several escrows;
  - share purchases on termination;
  - one MAE entry per defined term;
  - `buyer_breach` for reverse-fee breaches;
  - "proposal pending, no tail" triggers.

**How it was measured.**
- **Scope:** the same 50 holdout agreements, four concepts only: target fee, reverse fee, indemnity cap and MAE definition.
- **Gold:** the adjudicated gold from 7.6 had been deleted with the run's scratch files, so Opus wrote fresh gold for these concepts under v2.3, from the full text.
- **Arms:** two Sonnet arms ran on the same schema and were scored against the same gold:
  - **v3:** the 7.6 candidates (with the exhibit continuation fix);
  - **v4:** definition resolution.
- **Adjudication:** Opus judged every disputed field in either arm and corrected gold in 9 fields.
- **Caveats:**
  - The holdout is not blind for these fields: the v2.3 rules and the resolution patterns were written from its 7.6 errors. Read the numbers as "did the fixes work", not as a fresh pass/fail.
  - A network outage killed four extraction batches midway; the four missing agreements were re-run.

**Results, against adjudicated gold.**

| Field | v3 candidates | v4 definition resolution |
|---|---|---|
| Target-fee triggers (exact set) | 14/26 (set F1 0.92) | **20/26 (0.96)** |
| Reverse-fee triggers (exact set) | 9/13 (0.85) | **12/14 (0.93)** |
| Cap amounts (exact set) | 16/23 (0.86) | **19/23 (0.92)** |
| MAE carve-out categories (exact set) | 34/49 (0.95) | **38/49 (0.97)** |
| MAE ability-to-consummate prong | 44/49 | 46/49 |
| Target-fee amount, payor, remedy | 26/26 each | 26/26 each |
| Detection false positives / misses | 1 / 2 | 3 / 1 |
| Disputed fields after adjudication | 49 | 36 |

- **Retrieval gaps:** 17 extractor errors in the v3 arm, 3 in v4. Definition resolution did what it was for.
- **Detection.** v4's three false positives come from two agreements:
  - b5eb61e9 calls a $200,000 expense reimbursement a "Buyer/Seller Termination Fee", which yields one target-fee and one reverse-fee false positive.
  - 43a2d63d is an R&W-insurance-only deal that v4 coded as having a cap.
  - Both are coding errors on text the extractor saw, not retrieval. The run averages 0.06 false positives per agreement, inside the 0.1 bar.
- **Missed rules are now the largest error source:** 14 of v4's 29 extractor errors (v3: 16 of 46), against 3 retrieval gaps. The rules missed most often:
  - **v2.2 rule 12:** a recommendation-change definition that includes failing to recommend needs both codes. This accounts for 4 of the 6 remaining target-trigger disputes.
  - **v2.3 rule 11:** MAE carve-out placement ("labor" or "financial" inside a general-conditions list, "calamity").
  - **v2.3 rule 2:** a cap equal to the price.

  The extractor saw the definition in each case and did not apply the rule.

**Against the acceptance bar.**
- **Pass:** MAE carve-out categories now pass by set F1 (0.97).
- **Close:** target-fee triggers (0.96) and reverse-fee triggers (0.93) are close.
- **Fail:** cap amounts (0.92) still fail.

What remains is rule application, so the next step is not more retrieval.

**Next step.**
1. **Deterministic post-checks for the rules that keep being missed:**
   - when the resolved recommendation-change definition mentions failing to make, include or reaffirm the recommendation, add `recommendation_failure`;
   - when a cap's text is "the Purchase Price" or "the Merger Consideration", set `pct_of_purchase_price` 100;
   - when a fee clause's amount is defined as out-of-pocket expenses, move it to expense reimbursement.

   These are code, not model calls, so they can be tested exactly.
2. **Judge the post-checks on a fresh 50-agreement holdout** that has been read by no one, which gives the M3 call. (This re-check's per-agreement outputs were deleted with its scratch files, so the post-checks cannot be measured on them; [7.8](#78-m3-holdout-2026-10-21) runs them on the fresh holdout instead.)
3. **Before M3,** settle the open questions the adjudicators raised:
   - whether a refund of a prior bidder's fee paid as liquidated damages is a reverse fee;
   - whether a cap is present when indemnification lives in an exhibit outside the text, or in an R&W-insurance-only deal;
   - whether a price stated as a sum of parts fills the cap amount.

### 7.8 M3 holdout (2026-10-21)

**What was fixed before the run.**
- **Schema v2.4 (`pilot/schema_v2_4.md`):** v2.3 plus the rulings on the open questions above:
  - a refund of a prior bidder's fee is not a reverse fee (notes only);
  - indemnification governed by a document outside the text makes the cap and basket `absent`, with a note;
  - an R&W-insurance-only deal has no cap;
  - a price stated only as components leaves the cap amount `not_stated`.
- **Post-checks (`pilot/postchecks.py`, 14 tests in `pilot/test_postchecks.py`):** the three rules from 7.7, as code that reads only the candidates the extractor saw. The tests are the clauses the extractor got wrong in 7.7, plus the wording each check must leave alone.
- **Stage 1 fix:** a term defined at the end of a list ("(iii) …, a 'Change of Recommendation'") now brings the earlier list items and the lead-in line. Before, the "(ii) fail to include the Company Recommendation" item never reached the extractor.

**How it was measured.**
- **Sample:** 50 agreements none of the earlier rounds used (seed `m3holdout`), drawn the same way as the 7.6 holdout.
- **Blindness:** the schema, post-checks and stage 1 were frozen before any gold or extraction existed. The only look at the new sample was a count of how often the recommendation check matched its candidates (9 of 50), and of how many candidate files the list fix changed (10, all slightly).
- **Labels:** Opus wrote gold for all ten concepts from the full text. Sonnet extracted from the candidates; the candidates were a median 36% of the text.
- **Adjudication:** Opus judged every disputed field (66) and corrected gold in 3 agreements.
- **Arms:** Sonnet as extracted, and the same output after the post-checks.

**Results, against adjudicated gold.**
- **Detection:** 9 of 10 concepts have precision and recall of at least 0.95. Financing does not: recall 0.83 (7 misses).
- **False positives:** 1 in 50 agreements (an MAE definition), 0.02 per agreement.
- **Quotes:** every non-empty quote is verbatim. The scorer's 99.1% counts 21 empty quote strings as misses.
- **Stage 1 retrieval:** 97.1% of gold spans fall in a candidate section. 17 of the 30 misses are financing.

| Field | Exact | Set F1 |
|---|---|---|
| Target-fee amount, payor | 25/25 each | |
| Target-fee triggers | 19/25 (20/25 after post-checks) | 0.97 |
| Target-fee remedy | 24/25 | |
| Reverse-fee triggers | 11/15 | 0.86 |
| Reverse-fee kind | 13/15 | |
| Expense-reimbursement cap | 19/19 | |
| Outside-date initial date | 45/47 | |
| Outside-date effect | 47/47 | |
| MAE carve-out categories | 34/45 | 0.96 |
| MAE ability-to-consummate prong | 43/45 | |
| Cap amounts | 16/22 | 0.90 |
| Basket type and amount | 19/21 | 0.93 |
| Survival, general and fundamental reps | 48/49 each | |

**What went wrong.**
- **Verdicts:** gold right 56, extractor right 2, both wrong 1, schema ambiguous 7.
- **Extractor errors (57):**
  - retrieval gaps 21;
  - wrong codes 13;
  - missed rules 8;
  - over-inclusion 5;
  - wrong values 5;
  - missed in candidates 5.
- **Retrieval is the largest cause again, mostly financing (9 of its 16 errors):**
  - funds representations that stage 1 did not select;
  - SPAC trust accounts and buyer deposits;
  - a lenders'-consent closing condition.
- **Financing also over-includes:** 4 cases list uncommitted capital raises as committed financing.
- **Missed rules:** 8 across ten concepts, against 14 for four concepts in 7.7. Sonnet applied the recommendation-failure, cap-equals-price and expense-fee rules on its own in all but one case. The post-checks changed one agreement (a7d85da2) and fixed its trigger set; they made no wrong change.
- **Reverse-fee errors are mostly wrong codes:** auction-loss terminations (`superior_proposal_termination`), reciprocal fees, and drop-dead-date triggers.

**Against the acceptance bar.**
- **Meets it on this holdout:**
  - target fee (detection, amount, payor; triggers at set F1 0.97; remedy 24/25);
  - expense reimbursement;
  - survival.
- **Close:**
  - outside date: detection and effect are perfect; the initial date is 45/47, both misses retrieval gaps;
  - MAE definition: carve-outs 0.96, precision 0.978.
- **Does not meet it:**
  - reverse fee (triggers 0.86);
  - indemnity cap (amounts 0.90);
  - basket (0.93);
  - financing (recall 0.83).
- **Too few cases to call:** 338 elections (4 present).

**Open schema questions from adjudication.**
- Whether a later outside date that applies only when regulatory conditions are unmet is an extension.
- Whether an automatic extension while a specific-performance action is pending is an extension.
- Whether two mirrored cure-period extensions are one entry or two.
- Whether carve-outs that sit in an MAE closing condition, not the definition, count.
- Whether an undefined inline MAE closing condition is a definition.
- Codes for the buyer's-identity and cyber-attack carve-outs.
- How to code `escrow_sole_recourse` when the escrow is the only recourse for general reps but not fundamental reps, or is used "initially".

**Next step.**
1. **M3 scope:** serve target fee, expense reimbursement and survival, plus the outside date and the MAE definition's presence and prongs. Hold the other coded fields until they pass.
2. **Financing retrieval:** select funds representations, trust-account and deposit provisions, and financing-related closing conditions in stage 1. Add a rule that a capital raise without a commitment is not committed financing.
3. **Settle the open questions above in v2.5,** then re-measure reverse fee, cap and basket on a fresh holdout. Done in 7.9.

### 7.9 Financing-retrieval round (2026-10-21)

Steps 2 and 3 of the 7.8 next step, measured on four concepts: financing, reverse fee, cap and basket.

**What was fixed before the run.**
- **Stage 1 (`pilot/prepare_v5.py`):** `prepare_v4` plus the selections that 7.8's financing misses called for:
  - funds representations worded "the funds necessary", "adequate (cash) resources", "access to sufficient" or "financing sufficient";
  - SPAC trust accounts, PIPE and subscription investors, and buyer deposits (including those named in the recitals);
  - financing conditions and covenants: lenders' consent, financing letters, cooperation with the buyer's banks or lenders;
  - a buyer's guarantee of its own or an affiliate's obligations;
  - every section of an indemnification article, whatever its title;
  - a definition whose opening words name a relevant term, even when the layer did not parse the term.
- **Its effect on the M3 sample (not blind, since the fixes came from those misses):** stage 1 gold-span recall for all ten concepts went from 96.1% to 98.5%. Financing went from 97 to 111 of 116 spans, cap from 106 to 111 of 114, reverse fee from 84 to 87 of 87. The candidates grew from 36.1% to 38.2% of the text.
- **Schema v2.5 (`pilot/schema_v2_5.md`):** v2.4 plus 12 rulings, covering the 7.8 open questions and the financing over-inclusion:
  - committed financing only when committed or in hand (an uncommitted raise goes in notes);
  - funds-representation phrasings;
  - a lenders'-consent condition makes `financing_condition` true;
  - reverse-fee triggers for auction losses, outside dates and reciprocal liquidated damages;
  - per-seller caps;
  - `escrow_sole_recourse` when the escrow is the sole recourse only for general reps, or only "first";
  - deductible only with "only in excess" wording;
  - the three outside-date extension questions;
  - MAE carve-outs from the definition only, and capitalized versus passing MAE mentions;
  - codes for the buyer's-identity and cyber-attack carve-outs;
  - an election that needs the other party's consent.

**How it was measured.**
- **Sample:** 50 fresh agreements (seed `m4holdout`), drawn the same way as 7.6 and 7.8. None appears in M1, M2, the M2 holdout or the M3 holdout. The corpus had drifted since M3: re-running the M3 query reproduced 49 of its 50 agreements, so the exclusion used the M3 manifest itself.
- **Blindness:** stage 1 and v2.5 were frozen before any gold or extraction existed.
- **Labels:** Opus wrote gold for the four concepts from the full text. Sonnet extracted from the candidates, which were a median 35% of the text.
- **Adjudication:** Opus judged every disputed field (32 in 20 agreements) and corrected gold in 1 field (a financing-cooperation covenant gold had missed).
- **Post-checks:** they changed nothing on this sample, so the post-checked arm is identical to the raw one.

**Results, against adjudicated gold.**
- **False positives:** none in 200 concept decisions.
- **Quotes:** every non-empty quote is verbatim (two empty strings count as misses in the scorer's 99.7%).
- **Stage 1 retrieval:** 96.5% of gold spans fall in a candidate section (386 of 400). 8 of the 14 misses are financing.
- **Cap and basket detection:** the one miss for each is the same agreement (4c30a132), whose indemnification section stage 1 did not select.

| Concept | Present | Precision | Recall | M3 recall |
|---|---|---|---|---|
| Financing | 42 | 1.00 | **0.93** | 0.83 |
| Reverse fee | 16 | 1.00 | 1.00 | 1.00 |
| Cap | 22 | 1.00 | 0.955 | 1.00 |
| Basket | 20 | 1.00 | 0.95 | 1.00 |

| Field | Exact | Set F1 | M3 |
|---|---|---|---|
| Financing committed types | 35/39 | 0.95 | 27/35 (0.88) |
| Financing efforts, condition | 38/39, 37/39 | | 33/35 each |
| Financing cooperation, marketing period | 39/39 each | | |
| Reverse-fee triggers | 13/16 | 0.95 | 11/15 (0.86) |
| Reverse-fee kind, payor | 16/16 each | | 13/15 kind |
| Reverse-fee amounts | 15/16 (16/16 on the percentage) | 0.95 | 13/15 |
| Cap amounts | 14/21 (18/21 on the percentage) | 0.78 (0.94) | 16/22 (0.90) |
| Cap escrow sole recourse | 21/21 | | 18/22 |
| Basket type and amount | 14/19 (17/19 on the percentage) | 0.78 (0.91) | 19/21 (0.93) |
| De minimis | 18/19 | | 21/21 |

"On the percentage" compares a percentage-of-price amount by its percentage alone (`score_v2.PCT_FIRST`). It matters because of the main open question below. The scorer also now treats percentages 1 and 1.0 as equal; that changed no M3 number.

**What went wrong.**
- **Verdicts:** gold right 21, extractor right 1, schema ambiguous 9, equivalent 1.
- **Extractor errors (21):**
  - retrieval gaps 7;
  - wrong codes 5;
  - missed rules 4;
  - wrong values 3;
  - missed in candidates 2.
- **Financing retrieval is better but not closed.** The three missed financing agreements, and most of its 8 missed spans, use wordings the new patterns do not cover:
  - "access to … the financial resources to consummate the Merger";
  - "will have available funds, or immediately available capacity under committed credit facilities";
  - a covenant "to maintain sufficient liquid accounts or borrowing capacity";
  - a recital naming a sponsor's "Cash Equity Capitalization".
- **Financing over-inclusion is gone.** None of the 4 committed-type errors lists an uncommitted raise. Two miss a funds representation's `cash_on_hand`, one codes a defined "Equity Funds" as equity, and one is ambiguous (below).
- **Cap and basket amounts fall mostly to one unsettled question.** In 6 of the 9 ambiguous verdicts, both sides agree on the percentage of a single stated price, and the extractor also fills in the dollar figure it computes. v2.3 rule 2 allows that only for a cap equal to 100% of the price. Compared on the percentage, cap amounts are 18/21 and baskets 17/19. The rest are real errors: a missed second cap (two escrows summed), a per-seller tax cap, and "in excess of (but not including)" coded as tipping instead of a deductible.

**Against the acceptance bar.**
- **Meets it on this holdout:** reverse fee (detection 1.00; triggers set F1 0.95; kind, payor and amounts on the percentage all exact).
- **Still short even if computed amounts are allowed:** cap amounts (0.94) and basket (0.91), compared on the percentage.
- **Does not meet it:**
  - financing detection (recall 0.93; all three misses are retrieval gaps);
  - cap amounts (0.78 as scored);
  - basket (0.78 as scored).
- **Improved from 7.8:**
  - financing recall 0.83 → 0.93;
  - committed types 0.88 → 0.95;
  - reverse-fee triggers 0.86 → 0.95.

**Open schema questions from adjudication.**
- **Computed amounts.** May a fee, cap or basket stated as a percentage of a price given as one number also carry the computed dollar amount? This is 6 verdicts.
- **Words against numerals:** "three percent (6.0%)".
- **Commitments named only in a funds representation:** does "sufficient cash, or commitments from the Guarantors", with no commitment letter, add `equity`?
- **A "cumulative remedies" clause scoped to the fee's section:** does it count as language about the fee?
- **Two indemnitors with different de minimis thresholds:** `de_minimis` holds one value.
- **Kind of a reverse fee:** every route needs a regulatory condition, but one route is a buyer breach. Is that `regulatory` or `mixed`?

**Next step.**
1. **Settle computed amounts in v2.6.** The simplest rule is that `amount` holds only a figure the text states, with the percentage kept, so that scoring and serving compare percentages. Then re-score this holdout. No new labels are needed, since both sides already agree on the percentage.
2. **Add the four financing wordings above to stage 1,** and check them against the M3 and M4 samples. That check is not blind, so the next fresh holdout gives the real number.
3. **Serve reverse fee** alongside the 7.8 set. Hold financing, cap and basket until the next holdout clears the bar.


### 7.10 Schema v2.6 and the holdout re-score (2026-10-22)

Steps 1 and 2 of the 7.9 next step. No new labels and no new extraction: the 7.9 holdout is re-scored under the new rule.

**What changed.**
- **Schema v2.6 (`pilot/schema_v2_6.md`):** v2.5 plus 2 rulings.
  - `amount` holds only a dollar figure the text states. A fee, cap, basket or de minimis stated as a percentage of a price keeps `basis` and `pct`, with `amount: "not_stated"`, even when the price is one stated number. A cap equal to a stated price keeps that number (v2.3 rule 2).
  - Three more funds-representation wordings ("access to … the financial resources", "available funds, or immediately available capacity under committed credit facilities", "maintain sufficient liquid accounts or borrowing capacity").
- **Post-check (`postchecks.py`, v2.6 rule 1):** a percentage-of-price amount whose dollar figure does not appear in the candidates becomes `"not_stated"`. It changed 15 amounts in 5 agreements; 5 new tests cover it.
- **Gold:** the same rule, checked against the full text, found 5 computed amounts in adjudicated gold (2 agreements, where the extractor had agreed). They were set to `"not_stated"`. These are corrections by rule, not new judgments.
- **Stage 1 (`pilot/prepare_v6.py`):** the four financing wordings from 7.9, plus nearby forms: a closing "not conditioned upon" financing, keeping or amending a credit agreement, and alternative financing.

**Results, post-checked arm against corrected gold.**

| Field | v2.5 (7.9) | v2.6 | Bar |
|---|---|---|---|
| Cap amounts | 14/21, F1 0.78 | 17/21, F1 **0.91** | 0.90 |
| Basket type and amount | 14/19, F1 0.78 | 17/19, F1 **0.91** | 0.93 |
| Reverse-fee amounts | 15/16, F1 0.95 | 16/16, F1 1.00 | |

- **Cap amounts now meet the bar,** by a small margin. The 4 remaining cap misses are real errors or other open questions: a missed second cap (two escrows summed), a per-seller tax cap, "three percent (6.0%)", and a 100% cap of the "Closing Purchase Price" where the extractor kept a stated $56,000,000 and gold left it blank.
- **Basket still fails** (0.91 against 0.93). Its 2 misses are a deductible coded as tipping ("in excess of (but not including)") and a $1,000,000 basket gold codes `other` that the extractor read as a $500,000 deductible.
- **The raw extractor arm drops** (cap 0.75, basket 0.74), because it followed v2.5 and computed amounts. The v2.6 text has not yet been run through the extractor; on the next holdout both the instruction and the post-check apply.
- **Stage 1, not blind:** on this holdout, v6 puts 394 of 400 gold spans in the candidates (v5: 386), recovering all 8 financing misses, and the candidates grow from 39.6% to 40.5% of the text. On the M3 holdout it changes nothing (1,024 of 1,031 either way). The numbers are in `pilot/m4_holdout_v26_results.json`.

**Schema v2.7 (`pilot/schema_v2_7.md`):** v2.6 plus 5 rulings, settling the rest of the 7.9 open questions before the next run:
- words control when words and numerals disagree;
- a funds representation that mentions commitments stays `cash_on_hand` unless the commitment itself (a letter, subscription agreement or funding guarantee) is in the text;
- a cumulative-remedies clause scoped to the fee's own section, with no exception for the fee, makes the remedy `non_exclusive`;
- `de_minimis` holds the target side's threshold, with others in notes;
- reverse-fee `kind` follows the triggers, so a buyer-breach route makes it `mixed` even when every route also needs a regulatory condition.

The first four agree with adjudicated gold on this holdout, so re-scoring it under them would be circular. None was re-scored.

**Next holdout, inputs frozen (2026-10-22).** 50 fresh agreements (seed `m5holdout`, 25 public and 25 private, all four eras), with stage 1 from `prepare_v6.py` already built. The candidates are a median 36.7% of the text. M3 and M4 are excluded by their manifests. M1, M2 and the M2 holdout are excluded by re-running their seeded queries, which may differ slightly from the originals because the corpus has drifted. No gold or extraction exists yet, so the run is still blind.

**Next step.**
1. **Run that holdout** on financing, cap and basket (reverse fee optional) with schema v2.7 and the post-checks: Opus gold from the full text, Sonnet from the candidates, Opus adjudication. It gives the blind financing-recall number and the first extractor run under v2.6 and v2.7.
2. **Serve cap amounts** with the reverse fee if that run confirms them. Hold financing and basket until they clear the bar.

### 7.11 M5 holdout (2026-10-22)

Step 1 of the 7.10 next step: a blind run of stage 1 v6, schema v2.7 and the post-checks on financing, cap and basket.

**How it was measured.**
- **Sample:** the 50 agreements frozen in 7.10 (seed `m5holdout`). Stage 1 and the schema were fixed before any gold or extraction existed.
- **Labels:** Opus wrote gold for the three concepts from the full text. Sonnet extracted from the candidates, a median 36.7% of the text. Both ran as 10 batches of 5 agreements.
- **Adjudication:** Opus judged all 22 disputed fields (21 agreements). Gold was right 18 times, the extractor once (a de-SPAC trust account, so financing is present), and 3 were schema questions.
- **Post-checks:** one change (v2.3 rule 2 on one cap); the v2.6 stated-amount check changed nothing, because the extractor no longer computes dollar figures.

**Results, against adjudicated gold.**
- **False positives:** none in 150 concept decisions.
- **Quotes:** all 488 verbatim.
- **Stage 1 retrieval:** 96.1% of gold spans in a candidate section (244 of 254). 8 of the 10 misses are financing.

| Concept or field | Result | Bar | 7.9 / 7.10 |
|---|---|---|---|
| Financing detection | precision 1.00, recall **0.86** (30/35) | 0.95 | 0.93 |
| Financing committed types | 25/30, set F1 0.94 | | 0.95 |
| Financing efforts, marketing period | 30/30 each | | |
| Financing condition, cooperation | 29/30 each | | |
| Cap detection | 19/19, no false positives | 0.95 | 0.955 |
| Cap amounts | 13/19, set F1 **0.77** | 0.90 | 0.91 (re-scored) |
| Cap escrow sole recourse | 16/19 | | 21/21 |
| Basket detection | 16/16, no false positives | 0.95 | 0.95 |
| Basket type and amount | 16/16, set F1 **1.00** | 0.93 | 0.91 (re-scored) |
| De minimis | 16/16 | | 18/19 |

**Against the bar.**
- **Basket meets it:** detection and every field exact on 16 baskets.
- **Financing fails on retrieval again.** All 5 missed agreements are stage 1 gaps, as in 7.9, but with new wordings: "on hand cash in an amount sufficient", "cash and cash equivalents on hand or readily available", "immediately available funding through cash on hand or under its credit facilities", a recital naming "a binding commitment of its financing banks", and a cooperation covenant that sits outside every financing-titled section. Adding phrases after each holdout has not converged: each fresh sample brings new ones.
- **Cap amounts fall back to 0.77,** from 0.91 on the 7.10 re-score. The misses are not computed amounts (v2.6 settled those). They are coding errors on caps built from other amounts: an escrow percentage coded as a formula with no percentage, two general-rep caps merged into one, a "Maximum Individual Amount" whose definition stage 1 did not pull in, a price less one class's share coded as 100% of the price, a sum of escrow and earn-out set-off coded as one percentage, and the cash price actually received coded as a formula instead of 100% of the price (v2.3 rule 2). The 7.10 number was a re-score of a sample the rules came from, so this is the first blind measure of v2.6.
- **Escrow sole recourse** (16/19): one "claims go to the holdback first" case coded true despite v2.5 rule 6, one escrow that only partly supports the indemnity left `not_stated`, and one set-off-only case (below).

**Extractor errors (18):** retrieval gaps 6, missed rules 5, wrong values 3, missed in candidates 3, wrong code 1.

**Open schema questions from adjudication.**
- A parent's guarantee of all the buyer's payment obligations: is it committed financing (`other`), given v2.7 rule 2's "guarantee of the funding"?
- An equity rollover (target holders contribute shares to the buyer): `equity` or `other`?
- `escrow_sole_recourse` when there is no escrow and the only recovery is set-off against a seller note or deferred payments.

**Next step.**
1. **Stop adding financing phrases.** Select by structure instead: every section of the buyer's representations article and of the covenants that name financing, plus recitals. That costs candidate share; measure it on M3, M4 and M5 before the next blind run.
2. **Cap amounts:** add post-checks for the patterns above where they are mechanical (a cap that names an escrow or holdback amount takes that amount's basis and percentage; separate caps for policy-covered and uncovered claims stay separate), and pull in definitions of terms used in a cap ("Maximum Individual Amount").
3. **Serve basket** with the reverse fee. Hold financing and cap amounts.

### 7.12 Structural financing retrieval and schema v2.8 (2026-10-22)

Steps 1 and 2 of the 7.11 next step, with no LLM calls: stage 1 changes measured on the three labeled holdouts, and schema rulings for the next blind run.

**Stage 1 v7 (`pilot/prepare_v7.py`).** On top of v6, it adds whole sections by where they sit, not by phrase:
- the preamble and recitals (everything before the first article);
- a section of any representations article whose title or body speaks of having sufficient cash, funds, funding or capital resources;
- any section outside a definitions section that uses the word "financing";
- definitions of cap and limit terms ("Maximum Individual Amount"), which a cap often names without stating.

| Sample | Gold spans in candidates, v6 → v7 | Median candidate share, v6 → v7 |
|---|---|---|
| M3 holdout | 1,024 → 1,025 of 1,031 | 38.2% → 40.9% |
| M4 holdout | 394 → 395 of 400 | 35.2% → 38.3% |
| M5 holdout | 244 → **252** of 254 (financing misses 8 → 0) | 36.7% → 42.0% |

This is not blind for M5, whose misses prompted the change, but the rules name structure, not the M5 wordings, and they also recover a financing span on M3 and a reverse-fee span on M4. The price is about 3 to 5 points of candidate text. The remaining misses are 2 financing spans in one M3 tender-offer agreement (an offer condition and a miscellaneous undertaking), 4 MAE spans (M3), and 7 cap and basket spans across M4 and M5. The numbers are in `pilot/stage1_v7_check.json`.

**Cap amounts.** The 7.11 cap errors are judgment calls, not patterns a deterministic post-check can fix safely, so no post-check was added. Instead, v2.8 states each one as a rule.

**Schema v2.8 (`pilot/schema_v2_8.md`):** v2.7 plus 8 rulings.
- The three 7.11 questions: a parent guarantee is committed financing `other`; an equity rollover is `other`; recovery only by set-off is `escrow_sole_recourse` true.
- Five cap rules from the 7.11 errors: a cap equal to an escrow or holdback defined as a percentage of the price takes that percentage; one entry per route (policy-covered versus uncovered, per seller versus aggregate); the price actually received is 100%; the price less a part, or an escrow plus a set-off, is a formula; claims paid "first" from a holdback make `escrow_sole_recourse` false.

Re-scoring M5 under v2.8 would be circular, and it would need a new extraction anyway, so it was not done.

**Next step.**
1. **A fresh blind holdout** with `prepare_v7.py` and schema v2.8, on financing and cap, which are the two still short of the bar. A run of the same size as 7.11 costs about 3.6M subagent tokens; financing and cap only, on 50 agreements, would cost somewhat less.
2. **Serve basket** with the reverse fee.

### 7.13 Model-chosen stage 1 (2026-10-23)

Stage 1 had been pattern rules, and every miss since M3 was a rule that did not fire. This test asks whether a small model, using judgment, picks the right sections instead.

**How it ran.**
- **Picker:** Claude Haiku, as subagents, on all 150 labeled agreements (M3, M4 and M5 holdouts).
- **What it reads:** an outline of each agreement (`pilot/router.py outline`): every section's article and title, its first 200 characters, and the opening words of each lettered or numbered paragraph. That is a median 44,000 characters, about 12% of the full text.
- **What it returns:** for each of the ten concepts, the sections a later model must read, favoring recall. Definitions sections are not offered: their relevant terms are kept, and the terms the picked sections use are resolved, exactly as in v7, so the test isolates the choice of operative sections.
- **Scoring:** the same as stage 1 throughout. A gold span counts when it falls in a candidate section. Nothing new was labeled.
- **Leakage:** the picker's brief names typical funds-representation titles ("Funding", "Capital Resources", "Sufficient Funds"), some of which came from the M5 misses, so M5 is not fully blind for it either.

**Results.**

| Sample | Gold spans: picker / v7 / both together | Instances with any span reachable: picker / v7 / both together | Median candidate share: picker / v7 / both together |
|---|---|---|---|
| M3 (10 concepts) | 0.975 / 0.994 / **0.997** | 288 / 288 / **289** of 289 | 24.6% / 40.9% / 42.9% |
| M4 (4 concepts) | 0.970 / 0.988 / **1.000** | 99 / 99 / **100** of 100 | 22.5% / 38.3% / 40.5% |
| M5 (3 concepts) | 0.961 / 0.992 / **0.996** | 68 / 70 / 70 of 70 | 23.8% / 42.0% / 42.3% |

- **The picker finds the provision almost always** (455 of 459 instances), from 40% less text than the rules. What it loses is the second and third span of an instance, such as a fee trigger in another section, a financing covenant beside the funds representation, or an outside-date extension. Its missed spans (26, 12 and 10 per sample) are mostly financing (22 in all), reverse fee (10) and outside date (4).
- **The rules (v7) reach more spans** but read 16 to 18 more points of text. Their misses are a different set: MAE definitions and caps outside the selected sections.
- **The two together miss 4 of 1,685 gold spans,** for 0.3 to 2.2 points more text than v7 alone. The union is the best of the three, because the picker and the rules miss different things.
- **Cost:** the outline is about 11,000 tokens per agreement. Run as subagents here, the picker used about 4.4M tokens for 150 agreements, about 30,000 per agreement including instructions and tool overhead. Through the API with the brief cached, it would cost close to the outline alone.

**What this means for the design.** Judgment in retrieval helps most as a second opinion, not a replacement. The plan's stage 1 (8.1) already called for more than patterns: embedding retrieval and the defined-term graph. A small model reading the outline is a cheap stand-in for both, and it catches drafting no pattern anticipated. Stage 2 and 3 still make every decision about what a span means and what it says.

**Next step.**
1. **Use both for stage 1:** the v7 rules plus the picker's sections, at about 42% of the text.
2. **Run the next blind holdout with that stage 1 and schema v2.8,** on financing and cap. This time the picker's brief carries no titles from any labeled sample.

### 7.14 M6 holdout: rules plus picker, schema v2.8 (2026-10-23)

**Setup.**
- **Sample:** 50 fresh agreements (seed `m6holdout`): 25 public and 25 private, all four eras. The M3, M4 and M5 manifests are excluded.
- **Stage 1:** the v7 rules plus the Haiku picker's sections (`router.py union`), at a median 44.2% of the text. The picker's brief named no section titles from any labeled sample, so the run is blind for both halves.
- **Concepts:** financing structure and indemnification cap, with schema v2.8 and the post-checks.
- **Roles:** the same as 7.11. Sonnet extracts from candidates, Opus writes gold from the full text, and Opus adjudicates all 14 disputes.

**Results against the bars.**

| Measure | Bar | M6 | M5 (7.11) |
|---|---|---|---|
| Financing detection, precision / recall | 0.95 / 0.95 | **0.971 / 1.000** (34 of 34, 1 false positive) | 1.000 / 0.857 |
| Cap detection, precision / recall | 0.95 / 0.95 | **1.000 / 1.000** (20 of 20) | 1.000 / 1.000 |
| Cap amounts, set F1 | 0.90 | **0.946** (18 of 20 exact) | 0.794 |
| Escrow sole recourse | 0.97 | **20 of 20** | 16 of 19 |
| Committed financing types, set F1 | 0.97 | 0.931 (27 of 34 exact) | 0.943 |
| Financing cooperation, efforts, condition, marketing period | 0.97 | 33, 33, 34 and 34 of 34 | 29, 30, 29 and 30 of 30 |

- **Financing detection meets the bar on a blind sample for the first time.** Recall went from 0.86 to 1.00. Every M5 miss was retrieval, and the union stage 1 reached every provision here. The one false positive is a de-SPAC paid in stock (1d5864c6), which the adjudicator rated ambiguous: v2.2 rule 9's two sentences point different ways when the trust funds only redemptions.
- **Cap amounts meet the bar** at 0.946, and escrow sole recourse is perfect. The v2.8 rulings on formula caps, escrow percentages and set-off settled the M5 errors. The two remaining misses are two separate formula caps the extractor missed (a0302d25) and the holdback amounts below (67e84aa3).
- **Committed financing types are the one field below the bar.** Four of the seven misses drop `cash_on_hand` from a funds representation that counts the buyer's own cash beside its debt ("together with any available cash", "cash on hand and marketable securities"). The rule is in the schema (v2.5 rule 2, v2.7 rule 2), so this is rule application, not a missing rule. Two misses are retrieval: a definition naming a signed sale-leaseback as the financing (24a44fbc), and an adviser's cash contribution named only in the funds representation (0e814534). One is ambiguous: an affiliate's limited guarantee whose terms are not in the text (9cd0ec1b).
- **Gold quote validity** was 1.0 (534 quotes) and extractor quote validity 1.0 (430).

**Adjudication.** 14 disputes: gold correct 9, extractor correct 2, ambiguous 3. Gold was fixed twice: one funds representation gains `cash_on_hand` (7b1695ec), and a borrowing to fund a cash distribution at closing makes financing present (78106203). The third ambiguous item is a sponsor financing facility for after closing (f577915e): the schema does not say whether cooperation must be with acquisition financing.

**Stage 1.** Against adjudicated gold:

| | Gold spans reached | Median candidate share |
|---|---|---|
| Picker only | 256 of 269 (0.952) | 20.5% |
| v7 rules only | 263 of 269 (0.978) | 40.8% |
| Both (as run) | **268 of 269 (0.996)** | 44.2% |

The picker and the rules again miss different spans, and together they miss one. Two of the extractor's errors were rated retrieval gaps, both involving definitions. In 67e84aa3 the cap runs through holdback amounts defined as `“General Indemnity Holdback Amount:” $37,700,000`. The resolver does not recognize that form (a colon, no "means"), so the figures never reached the candidates. In 24a44fbc the candidates mention the VICI agreement, but not the definition saying it was signed with the merger agreement.

**A post-check bug, fixed.** The v2.3 rule 2 post-check (a cap of "the Purchase Price" is 100% of the price) fired twice on this run and was wrong both times. It rewrote "the Purchase Price, less the aggregate amount of all Buyer Claims" (a formula, by v2.8 rule 7) and "35% of the portion of the Merger Consideration that has been paid" to 100%. With the post-check as run, cap amounts were 0.904. It now leaves alone any cap the model already coded as a formula or a percentage, and any price followed by "less" or "minus" (two new tests). Re-run over the saved M3 to M6 extractions, the fixed check never fires. The old one fired once more, in M5 (c40f0c61), and was wrong there too: M5 cap amounts with the fixed post-check are 0.794, not 0.774. The table uses the fixed post-check.

**What this means.** With the combined stage 1 and schema v2.8, both concepts meet the detection bar and the cap meets every field bar. The only shortfall is committed financing types, at 0.931 against 0.97, and most of it is the extractor dropping `cash_on_hand` from funds representations that name it.

**Next step.**
1. **A post-check for `cash_on_hand`:** when a quoted funds representation counts the buyer's cash ("cash on hand", "available cash", "together with ... cash"), add `cash_on_hand` to committed financing. That would recover four of the seven misses.
2. **Teach the resolver the colon form of definition** (`“Term:” value`), which hid the cap figures in 67e84aa3.
3. **Then extend the union stage 1 and v2.8 to the other eight concepts** on one more blind holdout, to confirm the retrieval change holds across them.

### 7.15 Two fixes from M6, re-scored (2026-10-23)

These are the first two next steps from 7.14. The M6 numbers below are not blind: both fixes were written after seeing M6's errors.

**The fixes.**
- **A `cash_on_hand` post-check** (v2.5 rule 3, in `postchecks.py`). When a present financing instance has no `cash_on_hand`, the check looks in the candidates for a sentence that does four things: says the funds are sufficient, counts the buyer's own cash ("together with any available cash", "cash on hand", "marketable securities", "available funds"), names the buyer, and ties the money to the price or the deal. If it finds one, it adds `cash_on_hand` with that sentence as the quote. It skips contingency covenants ("In the event ... becomes unavailable", "provided that", substitute financing) and paying-agent deposits. Five tests cover it.
  - It does not catch a funds representation with no cash words at all ("Parent has available to it all financing necessary", 6fa82f1a). A wider version that did catch those broke as many earlier answers as it fixed.
- **Colon-form definitions** (`prepare_v7.py`). Definition resolution now also reads `“General Indemnity Holdback Amount:” $37,700,000` and `“Special Indemnity Holdback Amount”: means $18,850,000`. Five tests cover it. This changes the candidates of one agreement across all four holdouts (67e84aa3, by 1,778 characters), and that agreement was re-extracted.
  - The text layer also gives both of those leaves the wrong term ("Excluded License", "Lump Sum Bonus Payments"). The resolver now reads the term from the text, so this did not matter here. The text layer itself was not changed.

**M6 re-scored**, against the same adjudicated gold:

| Measure | Bar | 7.14 | Now |
|---|---|---|---|
| Committed financing types, set F1 | 0.97 | 0.931 (27 of 34 exact) | 0.962 (30 of 34) |
| Cap amounts, set F1 | 0.90 | 0.946 (18 of 20) | **0.987** (19 of 20) |
| Financing and cap detection | 0.95 | met | met (unchanged) |

- **Committed types are still just short** of 0.97. The four left are the cash-words-free funds representation above (6fa82f1a), two retrieval gaps (24a44fbc, 0e814534) and one ambiguous guarantee (9cd0ec1b).
- **The cap's last miss** is two separate formula caps the extractor did not code (a0302d25).
- **The re-extracted 67e84aa3 now has both holdback figures.**

**The post-check on the earlier holdouts.** Run over the saved M3, M4 and M5 extractions, it fires 8 times. Against the gold as first labeled, 5 fires agreed and 3 did not. The 3 that did not are worded like the M6 sentences that the M6 gold codes as `cash_on_hand`:
- two say the financing proceeds, "together with" cash, are sufficient (M4 cd25cee3, 5c8bc13b);
- one says the buyer "will at the Closing have sufficient immediately available funds, in cash" (M3 d1bc88a8).

Their gold was written under v2.4 and v2.5, before v2.7 rule 2 settled that a funds representation is `cash_on_hand` even when it mentions commitments.

**Adjudication of the three (Opus, under v2.8).** All three need `cash_on_hand`, and gold was fixed. Each now gets one added entry marked `"adjudicated": true`, and the originals are kept in the gitignored `out/gold_pre_coh/`.
- d1bc88a8 is a plain funds representation (v2.1 rule 3). The lender's commitment letter adds `debt` beside it, not in place of it (v2.7 rule 2).
- cd25cee3 counts the company's cash and cash equivalents with the financing proceeds.
- 5c8bc13b counts "any cash, marketable securities, available lines of credit or other sources of immediately available funds".

Against the corrected gold, no fire on any holdout disagrees with gold. Committed financing types, exact matches before and after the post-check:

| Holdout | Before | After |
|---|---|---|
| M3 | 27 of 35 | 28 of 35 |
| M4 | 33 of 39 | 36 of 39 |
| M5 | 25 of 30 | 27 of 30 |
| M6 | 27 of 34 | 30 of 34 |

- **M4 baseline.** The before figure fell from 35 to 33 because two extractions that matched the old gold no longer do. The committed-type figures in `m4_holdout_results.json` (7.9) predate this fix and were not rewritten.
- **M3 d1bc88a8** still misses after the post-check: the extractor coded the lender commitment `other` instead of `debt`.

**Next step.**
1. **Keep the `cash_on_hand` post-check.** It agrees with adjudicated gold on all 11 fires across four holdouts.
2. **Extend the union stage 1, v2.8 and the post-checks to the other eight concepts** on one more blind holdout.

### 7.16 M7 holdout: all ten concepts, rules plus picker, schema v2.8 (2026-10-23)

**Setup.**
- **Sample:** 50 fresh agreements (seed `m7holdout`): 25 public and 25 private, all four eras. The M3 to M6 manifests are excluded by uuid, and M1, M2 and the M2 holdout by their seeded queries. The query is saved as the run's `sample.sql`.
- **Stage 1:** the v7 rules plus the Haiku picker's sections for all ten concepts (`router.py union`), at a median 46.5% of the text. The picker's brief carries no section titles from any labeled sample.
- **Schema and post-checks:** v2.8 and the post-checks as of 7.15, frozen before any gold or extraction existed. The run is blind.
- **Roles:** Opus writes gold for all ten concepts from the full text. Sonnet extracts from the candidates. Opus adjudicates every disputed field (50 items, 51 verdicts).

**Results against the bar** (9.2: precision and recall ≥ 0.95, key fields ≥ 0.97, at most 0.1 false positives per agreement), after adjudication:

| Concept | Detection P / R | Fields (exact, or set F1) | Bar |
|---|---|---|---|
| Target fee | 1.00 / 1.00 (25) | amount F1 0.96 (24/25); payor, remedy 25/25; triggers F1 0.96 (17/25) | close: triggers, amount |
| Reverse fee | 1.00 / 0.94 (16 of 17) | amounts, kind, payor, remedy 16/16; triggers F1 0.99 (15/16) | close: one miss, rated ambiguous |
| Expense reimbursement | 1.00 / 1.00 (8) | cap, payor 8/8 | **met** |
| Outside date | 1.00 / 1.00 (49) | initial date 47/49; effect 49/49; extensions 48/49; breach bar F1 0.99 | close: initial date |
| MAE definition | 1.00 / 1.00 (50) | carve-outs F1 0.97 (41/50); ability prong 49/50; prospects, disproportionate 48/50; number of definitions 47/50 | close: secondary fields |
| Financing | 0.97 / 0.97 (36) | committed types F1 0.97 (32/36); cooperation, efforts, condition 35/36; marketing period 36/36 | close: committed types 0.967 |
| Indemnification cap | 1.00 / 1.00 (19) | amounts 19/19; escrow sole recourse 16/19 | fails: escrow sole recourse |
| Basket | 1.00 / 1.00 (18) | baskets, de minimis 18/18 | **met** |
| Survival | 1.00 / 1.00 (50) | general reps, no-survival 50/50; fundamental reps 48/50 | close: fundamental reps |
| 338 election | 1.00 / 1.00 (3) | election, decides 3/3; tax cost payer 0/3 | fails on 3 cases |

- **Detection holds across all ten concepts.** There is one false positive in 50 agreements (0.02 per agreement), and nine concepts meet both detection bars. The reverse fee's one miss (b05519ff) points to an outside "Contingency Letter Agreement" whose terms are not in the text, and the adjudicator rated it ambiguous. Financing, the concept that failed every earlier holdout on recall, is at 0.97. The miss is a lender-consent closing condition the extractor did not treat as a financing condition (v2.5 rule 3); the false positive is a stock-for-stock deal (v2.2 rule 9).
- **Two concepts meet every bar** (expense reimbursement and basket), and six more are within one or two fields of it. Cap amounts are 19/19, after failing every holdout from M2 to M5.
- **The clear failures are narrow.**
  - **Escrow sole recourse (16/19):** one escrow is only "the first source of payment" (v2.5 rule 6, v2.8 rule 8), and one deal has no escrow at all. The third is ambiguous: an escrow that is "collateral" for a seller who is liable directly.
  - **338 tax cost payer (0/3):** all three elections are prohibited, and gold and extractor differ on how to code who bears the cost of a prohibited election. Three cases are too few to judge the concept on.
- **Quotes:** 99.6% of 2,497 extractor quotes are verbatim.

**Post-checks.** They changed 6 agreements, and all 6 changes turned a wrong field right. The recommendation-failure check (v2.2 rule 12) fired 4 times, lifting target-fee triggers from 13 to 17 of 25 exact. The `cash_on_hand` check (7.15) fired twice, lifting committed financing types from 30 to 32 of 36.

**Adjudication.** 51 verdicts on 50 disputed items:

| Verdict | Count |
|---|---|
| Gold correct | 33 |
| Schema ambiguous | 10 (20%) |
| Extractor correct | 5 |
| Equivalent | 2 |
| Both wrong | 1 |

Gold was fixed in 6 agreements. Of the extractor's 34 errors:

| Cause | Count |
|---|---|
| Missed a schema rule | 12 |
| Wrong code | 6 |
| Wrong value | 6 |
| Retrieval gap | 5 |
| Missed text it had | 5 |

The recurring rule misses:
- A fee for breaching the covenant to call the stockholder meeting is `breach_no_tail` (v2.2 rule 11). There were 3 misses, coded `other`.
- "Material Adverse Change", defined separately, is its own MAE entry (v2.3 rule 10). There were 2 misses.

The ten ambiguous items are open schema questions:
- whether "near-term or long-term financial condition" is a prospects prong;
- which of two party-specific outside dates is the initial one;
- whether a general "legal conditions" carve-out adds `change_in_law`;
- the basis for a fee set as 3% of consideration plus option value;
- a mixed debt-and-equity commitment;
- a buyer covenant to transfer cash;
- a carve-out that fits two codes;
- the collateral escrow above;
- the reverse-fee miss above;
- whether the general frustration clause is a breach bar on the outside date.

**Stage 1.** Against adjudicated gold:

| | Gold spans reached | Median candidate share |
|---|---|---|
| Picker only | 870 of 894 (0.973) | 26.1% |
| v7 rules only | 886 of 894 (0.991) | 42.9% |
| Both (as run) | **893 of 894 (0.999)** | 46.5% |

As in 7.13 and 7.14, the two miss different spans, and together they miss one. Five extractor errors were still retrieval gaps, all inside definitions rather than whole sections:
- **Definitions not resolved:** a "Drop Dead Date" definition (00b0aa8c) and an "Excepted Representations" list (0224bd6d), which the operative clauses use.
- **Definitions cut short in the text layer:** two MAE definitions that stop mid-clause (cff15b20; af3dd60c, printed "MaterialAdverse Effect" without a space) and a Triggering Event definition cut at a page break (e446bd34).

**Cost.** Subagent tokens for the 50 agreements:

| Role | Tokens | Per agreement |
|---|---|---|
| Haiku picker | 1.5M | about 30K |
| Sonnet extraction | 2.8M | about 56K |
| Opus gold | 2.7M | |
| Opus adjudication | 0.6M | |

**What this means.** On a blind sample, the combined stage 1, schema v2.8 and post-checks meet the detection bar for nine of ten concepts and come close on the tenth. Two concepts meet every bar. What remains is a short list of field-level rules and five definition gaps, not retrieval of whole provisions. Compared with the last blind ten-concept run (M3, 7.8):
- financing recall went from 0.83 to 0.97;
- reverse-fee triggers from F1 0.86 to 0.99;
- cap amounts from 0.90 to 1.00;
- stage 1 span recall from 97.1% to 99.9%.

**Next step.**
1. **A post-check for v2.2 rule 11:** a fee on termination for breaching the meeting covenant is `breach_no_tail`.
2. **Schema v2.9:** rule on the ten ambiguous items, plus who bears the cost of a prohibited 338 election.
3. **Definition gaps:**
   - resolve "Drop Dead Date"-style date terms and lists of representations by name;
   - fix the text layer where a definition stops mid-clause.
4. **Then decide on M3 scope:** which concepts go to a corpus pass, given that two meet every bar and six are within a field or two of it.

### 7.17 Fixes from M7, re-scored (2026-10-23)

These are the first three next steps from 7.16. The M7 numbers below are **not blind**: every fix was written after seeing M7's errors, and several v2.9 rulings settle disputes in the direction one arm had already taken.

**The fixes.**
- **A post-check for v2.2 rule 11** (`postchecks.py`). A target-fee or reimbursement trigger coded `other` that is a termination for breaching the meeting covenant "by failing to call or hold" the stockholder meeting, with no tail, becomes `breach_no_tail`. Six tests cover it. Reverse fees are left alone because they use buyer-side codes.
  - Run over the saved extractions of all five holdouts, it fires 4 times, and gold agrees each time.
  - A first, looser version also fired on a meeting-date reference, a failure to hold the meeting by a date, and a reverse fee. Those three cases are now negative tests.
- **Definition gaps** (`prepare_v7.py`, also used by `router.py`):
  - **More terms:** date and representation terms are kept and resolved ("Drop Dead Date", "Longstop Date", "Excepted Representations"), and "MaterialAdverse Effect" printed without its space counts as an MAE term.
  - **Run-on definitions:** a kept definition that runs on into the next paragraphs brings them, up to the next quoted term. This covers a page number or "Table of Contents" line, and a definition the text layer split.
  - **Effect:** all five M7 gaps now reach the candidates. The change touches 14 of 50 M7 agreements and 16 of 50 M6 agreements, adding a median of about 1,200 characters. The median candidate share moves from 46.5% to 46.6%. Five new tests cover it.
  - **Not fixed:** the text layer itself still gives those leaves wrong terms.
- **Schema v2.9** (`pilot/schema_v2_9.md`): eleven rulings, one for each M7 question and one for who pays the cost of a prohibited 338 election.
  - **Outside date:** a general clause is a breach bar only if it speaks of terminating. With two party-specific outside dates, the earlier one is `initial`.
  - **MAE:** a time qualifier on financial condition is not `prospects`. "Legal conditions" does not add `change_in_law`. An interim-covenant carve-out is `compliance_with_agreement`.
  - **Fees:** a fee on consideration plus option value is `pct_of_equity_value`. A remedy fixed only in an outside letter does not make a reverse fee present.
  - **Cap:** an escrow that is only collateral means `escrow_sole_recourse` is false.
  - **Financing:** a mixed debt-and-equity commitment is two entries. An affiliate's covenant to fund is `other`.
  - **338 election:** a prohibited election has no cost shift unless the agreement assigns the tax on elections the buyer may still make.

**Re-score.**
- **Gold:** v2.9 changed gold in five agreements (backups in the gitignored `out/gold_pre_v29/`).
- **Re-extraction:** Sonnet re-extracted 20 agreements under v2.9: the 14 whose candidates changed, plus the 6 whose disputes v2.9 decides. The re-extraction raised 7 new disagreements, and Opus adjudicated them. Gold was right on 4, the extractor on 2 (gold fixed), and 1 is a new open question: the efforts standard in a de-SPAC with no debt.

| Measure | Bar | 7.16 | Old extraction, v2.9 gold | After re-extraction |
|---|---|---|---|---|
| Reverse fee recall | 0.95 | 0.94 | **1.00** | **1.00** |
| Target-fee triggers, set F1 | 0.97 | 0.961 (17/25) | 0.982 (20/25) | **0.982** (21/25) |
| Target-fee amounts, set F1 | 0.97 | 0.964 | 0.964 | **1.00** |
| Committed financing types, set F1 | 0.97 | 0.967 | 0.975 | **0.992** (35/36) |
| MAE carve-outs, set F1 | 0.97 | 0.971 | 0.972 | **0.995** (45/50) |
| MAE prospects, disproportionate exception | 0.97 | 48/50 each | 48/50 each | **50/50 each** |
| Outside-date initial date | 0.97 | 47/49 | 48/49 | **49/49** |
| 338 tax cost payer | 0.97 | 0/3 | 0/3 | **3/3** |
| Escrow sole recourse | 0.97 | 16/19 | 16/19 | 17/19 |
| Reverse-fee triggers, set F1 | 0.97 | 0.988 (15/16) | 0.988 | 0.953 (13/16) |
| Outside-date breach bar, set F1 | 0.97 | 0.989 | 0.989 | 0.967 |
| Financing efforts | 0.97 | 35/36 | 35/36 | 34/36 |
| Basket de minimis | 0.97 | 18/18 | 18/18 | 17/18 |

- **Detection now meets the bar for all ten concepts.** The reverse fee's one miss is settled by v2.9 rule 10. Financing stays at 0.97 for both precision and recall.
- **Four concepts now meet every bar:** target fee, expense reimbursement, survival and the 338 election (on 3 cases).
- **The rulings and post-check alone did a share of this:** see the "old extraction" column.
- **Re-extraction lifted most fields further, but not all.** Four fields fell, all in re-extracted agreements:
  - reverse-fee triggers: mirrored tail codes in be399153, and an extra `other` in 9ac16a58;
  - outside-date breach bar: "through" coded `no_causation_required` in 1509f5f1;
  - financing efforts: the new de-SPAC question;
  - de minimis: a per-claim threshold in cff15b20.

  These look like run-to-run differences rather than effects of the fixes: none of these fields was the target of a fix. On a sample of this size, one such difference moves a field across the 0.97 bar. Basket dropped out of "meets every bar" for this reason alone. The variance itself was not measured.
- **MAE number of definitions** (47/50) is also still below 0.97: separately defined "Material Adverse Change" terms the extractor folds into one entry.
- **Escrow sole recourse still fails** (17/19). Its two misses are in agreements that were not re-extracted, where the v2.8 extractor missed rules v2.5 rule 6 and v2.8 rule 8 already state.

**What this means.** The fixes closed what they targeted: the reverse-fee detection miss, target-fee triggers, the MAE fields, the outside-date initial date and the 338 cost payer. What is left near the bar is mostly run-to-run noise in single fields, plus escrow sole recourse.

**Next step.**
1. **Measure run-to-run variance directly:** extract the same 50 M7 agreements a second time and count the fields that differ. This shows which fields can reliably clear 0.97 on 50 agreements, and whether two runs plus a tie-break is worth its cost.
2. **A post-check for escrow sole recourse:** the extractor still misses "first source of payment" and set-off-without-escrow wording that the rules already cover.
3. **Rule on the de-SPAC efforts standard** (v2.10), then a fresh blind holdout with v2.9 to confirm the gains without the bias above.

### 7.18 Run-to-run variance on M7 (2026-10-23)

This is step 1 from 7.17.

**Setup.**
- Sonnet extracted all 50 M7 agreements a second time ("run 2"), under v2.9 and from the same candidates, followed by the same post-checks.
- Run 2 is compared with run 1, the extraction scored in 7.17. Run 1 holds 30 agreements extracted under v2.8 and 20 re-extracted under v2.9.
- The 20 agreements extracted under v2.9 in both runs show variance under identical conditions. On the other 30, the run difference also includes the schema change.
- Gold is the adjudicated v2.9 gold from 7.17. It was not changed for this run, and the run-2 disagreements were not adjudicated.
- One run-2 extractor listed other agents' run-2 outputs after writing its own five files. Its own extractions were finished by then, so they could not have been affected.
- Script: `pilot/compare_runs.py`. Numbers: `pilot/m7_variance_results.json`.

**How much the runs differ.**
- **Detection:** the runs disagree on none of the 500 agreement-concept decisions. Run 2's detection matches run 1's on every concept: all at 1.00 except financing at 0.97, the same as 7.17.
- **Field values:** 28 of 1,109 compared values differ (2.5%), spread over 23 agreements.
  - On the 20 agreements under identical conditions: 9 of 445 (2.0%).
  - On the 30 where the schema also changed: 19 of 664 (2.9%).
  - So most of the variation is sampling. Little of it comes from the schema version.
- **Where the differences fall:** 12 of the 28 are in trigger lists and extension counts. Target-fee triggers account for 6, reverse-fee triggers for 3, and the number of outside-date extensions for 3. The other 16 are single flips across 12 fields, such as an efforts standard, a cooperation flag, or one MAE carve-out.
- **Which run is right:** in every differing value, exactly one run matches gold (run 1 in 17, run 2 in 11). Separately, 18 field errors appear in both runs with the same value. Those are systematic, not noise.

**Which fields clear 0.97.** Each field's score is the lower of its two runs. Scalar fields use exact match and set fields use set F1, as in 7.16 and 7.17.

| Result | Fields (n present in gold) |
|---|---|
| **Clears in both runs** (12) | committed financing types (36), marketing period (36), cooperation (36, 35/36 in each run), general-rep survival (50), no survival (50), MAE ability to consummate, prospects, disproportionate exception and carve-outs (50 each), outside-date effect, initial date and extension conditions (49 each) |
| **Clears in both runs, too few cases to judge** (15) | target-fee amounts, payor, remedy and triggers (25); reverse-fee amounts, payor, kind and remedy (16); cap amounts (19); baskets (18); expense reimbursement cap and payor (8); the three 338 fields (3) |
| **Clears in one run only** (6) | financing condition (0.972 / 0.944), fundamental-rep survival (0.98 / 0.96), MAE number of definitions (0.94 / 0.98), outside-date breach bar (0.967 / 0.989), number of extensions (0.98 / 0.959), de minimis (0.944 / 1.00, n 18) |
| **Fails in both runs** (3) | financing efforts (34/36 in each run), escrow sole recourse (17/19 and 18/19), reverse-fee triggers (0.953 / 0.943) |

- **Below 34 cases, a single miss puts a field under 0.97,** so a perfect score there says little. Target-fee triggers also clear only on set F1. Their exact-match accuracy is 21/25 and 17/25, and they differ between the runs in 6 of 25 agreements, more than any other field.
- **No field is shown to be at 0.97 or better with 95% confidence on 50 agreements.**
  - A field with no errors on 50 cases has an exact one-sided lower bound of 0.942.
  - Showing 0.97 takes about 99 cases with no errors, 157 with one, or 208 with two.
  - So "clears" in the table means the field met the bar in both runs, not that it is proven to.

**What this means.**
- **Noise is real but small:** about 2% of field values change from run to run, and none of the detection calls do. Per field, though, one flip is enough to cross 0.97 on 50 agreements. That accounts for the six fields that clear in one run only, and for the four that fell in 7.17.
- **Two runs plus a tie-break would fix only the varying errors.** At best that is about 28 values (2.5%), for a full second extraction (about 60K Sonnet tokens per agreement). The 18 systematic errors would remain, and they need rules or post-checks.
  - A cheaper version reruns only the concepts whose fields are known to vary: fee triggers and outside-date extensions. Even then, run 1 was right in 17 of the 28 differences and run 2 in only 11, so a third run would not reliably pick the right answer.
  - Not worth it at this error rate.
- **The bar needs more cases than one holdout provides.** Confirming 0.97 per field needs about 100 agreements with the concept present: about two more 50-agreement holdouts for the common fields, and many more for fees, caps, baskets and 338.

**Next steps.**
1. **Target the systematic errors:**
   - an escrow-sole-recourse post-check (step 2 of 7.17);
   - a ruling on the de-SPAC efforts standard;
   - checks on reverse-fee trigger coding, where both runs miss the same mirrored and `other` codes.
2. **Pool holdouts for the bar:** score each field across M7 and the next fresh blind holdouts together, rather than per 50-agreement holdout, until it has about 100 cases.

### 7.19 Schema v2.10 and post-checks for the shared errors (2026-10-23)

This is step 1 from 7.18. It covers the three fields that failed in both runs. Like 7.17, it is **not blind**: every rule and check was written after seeing these errors. The checks are also audited on the M3 to M6 extractions, which they were not written from.

**Schema v2.10** (`pilot/schema_v2_10.md`), four rulings:
1. **A termination while a fee trigger was available** is that trigger, never `other`. This covers a termination on another ground (the outside date, a failed vote, an expired offer) at a time the payee "had the right to terminate" on a fee trigger. Gold had coded this three ways: folded in (M3 28803f36, M7 9ac16a58) and `other` (M7 47e96297).
2. **v2.2 rules 11 and 12 apply to the reverse fee.**
   - The buyer board's failure to make or include its recommendation is `recommendation_failure`, beside `buyer_recommendation_change`, not `buyer_approval_failure`.
   - A breach of the buyer's meeting covenant is `buyer_breach`.
3. **Set-off without an escrow** is `not_stated`, unless the text says set-off does not limit other remedies. In that case it is false.
4. **Which efforts standard.** The buyer's covenant on its debt financing comes first. Otherwise it is the covenant on the financing the buyer must obtain: an equity commitment, or in a de-SPAC the PIPE or transaction financing.
   - This settles the de-SPAC question from 7.17. It follows what gold already did in 13 records across the five holdouts.
   - The one exception, M7 3b86d6d4, changes to `reasonable_best`, the value both runs gave.

**Gold changes.** Two M7 records changed: 3b86d6d4 (efforts) and 47e96297 (target fee: the `other` trigger was dropped). Backups are in the gitignored `out/gold_pre_v210/`.

**Post-checks** (`postchecks.py`, 21 new tests, 58 in all):
- **Escrow "first, then sellers"** (v2.5 rule 6, v2.8 rule 8). An indemnity paid "first" from an escrow, holdback, policy or note and then by the sellers, or "the first source of payment", makes `escrow_sole_recourse` false.
  - **Left alone:** purchase-price adjustments, chains from one escrow to another or to a policy, and an adverbial "then".
  - On gold for the five holdouts, the check matches 6 of the 22 records gold codes false, and it fires on none of the 30 gold codes true or the 50 it leaves `not_stated`. The set-off check fires on one more false record, and on nothing else. The other false records rest on wording a pattern cannot read reliably.
- **Non-exclusive set-off** (v2.10 rule 3): changes a silent `escrow_sole_recourse` to false.
- **Had-right triggers** (v2.10 rule 1): drops an `other` trigger whose text says the payee "had the right to" or "could have" terminated, when other triggers remain.
- **Buyer recommendation failure** (v2.10 rule 2):
  - recodes a reverse-fee `buyer_approval_failure` that quotes the board's failure to include or make its recommendation;
  - adds `recommendation_failure` when the termination clause a `buyer_recommendation_change` trigger cites also covers that failure.
- **The meeting-covenant check (v2.2 rule 11) now also runs on reverse fees,** coding `buyer_breach`.

**Audit.** Across the saved extractions of all five holdouts, the new checks fire 10 times (3 on M3, 7 on M7), and each fire leaves its field matching gold. The details are in `pilot/m7_v210_results.json`.

**Re-score** (lower of the two runs, as in 7.18):

| Field | Bar | 7.18, run 1 / run 2 | v2.10 + post-checks |
|---|---|---|---|
| Escrow sole recourse | 0.97 | 17/19, 18/19 (fails both) | **19/19, 19/19** (too few cases to judge) |
| Financing efforts | 0.97 | 34/36, 34/36 (fails both) | **35/36, 35/36** (clears) |
| Reverse-fee triggers, set F1 | 0.97 | 0.953, 0.943 (fails both) | 0.977, 0.966 (one run below) |
| Target-fee triggers, set F1 | 0.97 | 0.982, 0.972 | 0.986, 0.975 |

- **No field now fails in both runs:**
  - 13 fields clear in both runs;
  - 16 clear on too few cases to judge;
  - 7 clear in one run only.
- **Run-to-run differences:** 27 of 1,109 values differ, against 28 before.
- **Detection is unchanged:** every concept is at 1.00, except financing at 0.97.

**What is left.**
- **Reverse-fee triggers** fail in run 2 on variation between runs, not on a shared error:
  - run 2 adds `financing_failure` in cbc1bf49 and `other` in 04d3f8ef;
  - run 1 codes a mirrored tail as `buyer_breach` in be399153.
- **Efforts:** the remaining misses are de-SPAC PIPE covenants that one run left `not_stated` (450faad1 in run 2, ebd2b75e in run 1). Rule 4 now tells the extractor to code them, but neither run saw v2.10. A blind run is needed to show whether the extractor follows it.

**Next steps.**
1. **A fresh blind holdout with v2.10 and these post-checks.** Score it alone, and pooled with M7, so that the common fields approach 100 cases.
2. **Watch the fields that vary most from run to run:** fee trigger lists and the number of outside-date extensions. A single run will keep moving these across 0.97.

### 7.20 M8 holdout: blind, schema v2.10, pooled with M7 (2026-10-23)

This is step 1 from 7.19.

**Setup.**
- **Sample:** 50 fresh agreements, drawn with the seed `'m8holdout'` and the same strata as M3 to M7. M1 to M7 are excluded (`.claude/span-taxonomy-runs/m8holdout/sample.sql`).
- **Stage 1:** the union of the v7 rules and the Haiku picker (7.14).
- **Extraction:** Sonnet, one run, under schema v2.10, followed by the post-checks.
- **Gold:** Opus read the full text and wrote gold without seeing the candidates or the extraction. Opus then adjudicated every disagreement under v2.10.
- **Blind:** nothing was changed after seeing M8's results.
- **Non-US agreements:** three are drafted under non-US law: English (51dd99bb), German (5eb36afc) and Ontario (f7dc72eb). They are tagged `holdout_non_us` in the manifest. The headline numbers below are the 47 US agreements, and all 50 are in `pilot/m8_holdout_results.json`. M7 has no non-US agreements.
- **Pooled numbers:** these add M7 run 2, the all-v2.9 extraction from 7.18, with the v2.10 post-checks and v2.10 gold. That gives 97 US agreements.
- **New code:** `pilot/pool_holdouts.py` sums counts across holdouts; `score_v2.py` now also returns the raw set-field counts.

**Stage 1.**
- It reaches 894 of 898 gold spans (0.996), from a median of 44% of the text.
- The v7 rules alone would show 41% of the text.
- Of the 4 misses, 2 are target-fee definitions, 1 is an MAE definition and 1 is a cap.

**Adjudication.** 55 verdicts:

| Verdict | Count |
|---|---|
| Gold right | 29 |
| Extractor right (gold fixed) | 12 |
| Both wrong (gold fixed) | 5 |
| Schema ambiguous | 8 |
| Equivalent | 1 |

Of the 34 extractor errors:
- 12 are rules it had but missed;
- 10 are wrong codes;
- 6 are text it was shown but missed;
- 4 are text stage 1 did not reach;
- 2 are wrong values.

**Detection** (precision / recall; bar 0.95 and at most 0.1 false positives per agreement):

| Concept | M8, 47 US | Pooled, 97 US |
|---|---|---|
| Target fee | 1.00 / 1.00 | 1.00 / 1.00 |
| Reverse fee | 1.00 / 1.00 | 1.00 / 1.00 |
| Expense reimbursement | 0.909 / 1.00 | 0.947 / 1.00 |
| Outside date | 1.00 / 1.00 | 1.00 / 1.00 |
| MAE definition | 1.00 / 0.979 | 1.00 / 0.990 |
| Financing | 1.00 / 1.00 | 0.986 / 0.986 |
| Indemnification cap | 1.00 / 1.00 | 1.00 / 1.00 |
| Basket | 1.00 / 1.00 | 1.00 / 1.00 |
| Survival | 1.00 / 1.00 | 1.00 / 1.00 |
| 338 election | 0.778 / 1.00 | 0.833 / 1.00 |

- **Eight of ten concepts meet the detection bar, on M8 and pooled.** The false-positive rate is under 0.1 per agreement for every concept.
- **Expense reimbursement misses on one false positive.** A "reimbursement" of a fee the buyer had advanced to an earlier bidder (e3d6f6e3) is coded present. v2.4 rule 1 already excludes it.
- **The 338 election misses on two false positives, both schema-ambiguous:**
  - a remark that tax on a "deemed asset sale under IRC Section 338" is for the buyer's account (dff2b928);
  - a tax representation about past 336(e) elections (e2b03c9a).
- **The one MAE miss is also schema-ambiguous:** an undefined lowercase "material adverse change" closing condition (8183fd32).

**Fields** (exact match for scalar fields, set F1 for set fields; bar 0.97):
- **Pooled, 29 of 36 fields meet the bar.** The seven that do not:
  - outside-date extension conditions, set F1 0.923 (95 cases);
  - the number of extensions, 0.958 (95);
  - target-fee triggers, set F1 0.969 (47);
  - cap amounts, set F1 0.968 (36);
  - reverse-fee kind and remedy, 27/28 each;
  - the 338 decider, 9/10.
- **Fields that newly meet the bar across 97 agreements include:**
  - escrow sole recourse (36/36);
  - financing efforts and cooperation (71/73 each);
  - financing condition (71/73);
  - committed financing types (set F1 0.979);
  - target-fee amounts (set F1 0.99);
  - reverse-fee triggers (set F1 0.973).
- **On M8 alone, the extra misses are single errors on small counts.** For example, reverse-fee kind is 11 of 12 and the target-fee remedy is 21 of 22.
- **At 95% confidence, only "no survival" is shown at 0.97** (97/97, lower bound 0.970). The common fields reach about 95 to 98 cases, close to the roughly 100 that 7.18 called for. Their lower bounds still sit at 0.94 to 0.97, because each has one or two errors.

**What the errors are.** Most are single codings the rules already cover, not new kinds of provision:
- **MAE carve-outs (9 records):** the general-economic and calamity rulings in v2.3 rule 11, and industry wording.
- **Target-fee triggers (7):**
  - a meeting-covenant breach cited only by section number ("breach of Sections 6.3 or 6.11"), which the v2.2 rule 11 post-check cannot read;
  - a tender offer expiring unaccepted, which is `approval_failure_with_tail`;
  - one recommendation-change definition stage 1 did not reach.
- **Outside-date extensions:** a marketing-period extension coded `other` (6bae3995), and one condition stage 1 did not reach.
- **Cap amounts:** the per-seller "portion of the price received" entry that v2.8 rule 5 requires was dropped in two agreements.
- **The schema-ambiguous items need rulings:**
  - whether a passing 338 or 336(e) mention, with no election in this deal, makes the 338 concept present;
  - an undefined inline MAC condition;
  - party sides in a de-SPAC where the operating company issues the shares;
  - a force-majeure catch-all carve-out;
  - transaction litigation in a private deal;
  - `joint` against `mandatory_joint` for a shared-intent 338 clause;
  - an unexplained $25.3M "reimbursement".

**Cost.** One Sonnet run is about 62K tokens per agreement for ten concepts. The Haiku picker adds about 30K, read from outlines.

**Next steps.**
1. **Schema v2.11 rulings** on the seven ambiguous items above.
2. **Two narrow post-checks:**
   - resolve a section-number meeting-covenant breach to `breach_no_tail` (or `buyer_breach`) when the cited section is the meeting covenant;
   - recode a tender offer expiring without acceptance as `approval_failure_with_tail` when a tail applies.
3. **Treat duration units as equal in scoring.** "1 year" and "12 months" should match; the scorer now counts them as a difference (1c996793).
4. **Keep pooling.** Showing 0.97 at 95% confidence takes about 157 cases with one error, or 208 with two (7.18). Two more 50-agreement holdouts would bring the common fields there.

### 7.21 Schema v2.11, two post-checks and M7 and M8 re-scored (2026-10-24)

This is steps 1 to 3 from 7.20. It is not blind: the rulings and checks were written after seeing M8.

**Schema v2.11** (`pilot/schema_v2_11.md`) has ten rulings:
1. **338 presence.** The concept is present when the agreement sets a term for an election in this deal: it permits, requires, prohibits or conditions one, or assigns its tax cost. A representation about past elections is absent.
2. **Undefined MAE.** An undefined lowercase MAE or MAC used in a condition, representation or termination right is present. This amends v2.5 rule 10.
3. **De-SPAC sides.** The SPAC is `buyer_side` and the operating business `target_side`, whichever entity issues the shares.
4. **"Political conditions."** These add `political_war_terrorism` beside `general_economic`. This amends v2.3 rule 11 to match what gold already does in 22 definitions in M7 and M8.
5. **Force-majeure catch-alls.** They add no code beyond the listed ones.
6. **Transaction litigation.** It is `stockholder_litigation` whoever brings it.
7. **`joint` and `mandatory_joint`.** These are now defined. An obligation to take steps to secure the election's treatment is `mandatory_joint`.
8. **Breach triggers that cite covenants by number.** They are coded by what each cited section is: the meeting covenant is `breach_no_tail` (`buyer_breach` on a reverse fee), and the no-shop is `breach_of_no_shop`.
9. **A tender offer lapsing unaccepted.** It is `approval_failure_with_tail` when a tail applies.
10. **A fixed "reimbursement".** A fixed sum payable on termination and not tied to named costs is a fee amount.

**Gold changes.**
- **M8:** six records changed.
  - dff2b928: the 338 concept is now present (rule 1).
  - 0d53ab07: `other` was removed from a force-majeure carve-out (rule 5).
  - c1f822d9: the decider is now `mandatory_joint` (rule 7).
  - a1c23200, fa1ab1ce and ff72f700: `breach_no_tail` was added (rule 8). Before this, M8 gold coded the same pattern three ways, and these three were the ones without it.
- **Rulings that went against the extractor:** e2b03c9a (338 absent), e3d6f6e3 (the $25.3M stays a fee amount) and 8183fd32 (MAE present).
- **M7:** no disagreement falls under the new rulings, so its gold is unchanged.
- **Backup:** the M8 gold before these changes is in `out/gold_pre_v211`.

**Post-checks** (`pilot/postchecks.py`; 14 new tests in `test_postchecks.py`, 77 in all with `test_prepare_v7.py` and the new `test_score_v2.py`):
- **Cited covenants (rule 8).** The check resolves each cited section number to that section's title. It reads the titles from the full text, because stage 1 does not always keep the cited section; in 0522c1f4 the no-shop section was not a candidate.
- **Offer lapse (rule 9).** The check reads the clause that a tail trigger cites.
- **Fires on all six holdouts:**
  - M8: eight agreements, and 7 of them now match gold's target-fee triggers exactly. The eighth, fa1ab1ce, still lacks `recommendation_failure`, an error these checks do not address.
  - M3: two fires, 8588a483 and 8c9ecefd. Both breach triggers cite the stockholder-meeting covenant beside the no-shop. M3 gold predates rule 8 and codes only the no-shop, so these fires follow the rule but disagree with M3's gold. M3 is not re-scored.
  - M4 to M7: none. One M4 reverse-fee fire on a mirrored no-shop trigger was ruled out by limiting the reverse-fee path to `other` triggers.

**Scorer.** `score_v2.duration` counts years as months, so "1 year" equals "12 months". This changes one record (M8 1c996793, survival).

**Results.**
- **M7:** its post-checked outputs do not change, and both runs score as they did in 7.19.
- **M8 (47 US):** fields meeting the bar go from 27 to 29 of 36. Detection still meets the bar for 8 of 10 concepts.
- **Pooled (97 US):**

| | 7.20 | v2.11 |
|---|---|---|
| Fields meeting 0.97 | 29 of 36 | 31 of 36 |
| Target-fee triggers | set F1 0.969 (33/47 exact) | 0.983 (38/47 exact) |
| 338 decider | 9/10 | 11/11 |
| 338 detection precision | 0.833 | 0.917 |
| Expense reimbursement precision | 0.947 | 0.947 |
| Fields shown at 0.97 with 95% confidence | no survival | no survival, general reps (97/97 each) |

- **Detection still fails for two concepts,** each on one false positive. Both are now extractor errors under written rules:
  - the 338 election, on the 336(e) representation in e2b03c9a (rule 1);
  - expense reimbursement, on the advanced-fee refund in e3d6f6e3 (v2.4 rule 1 and rule 10).
- **Five fields still fail:**
  - outside-date extension conditions, set F1 0.923 (95 cases);
  - the number of extensions, 0.958 (95);
  - cap amounts, set F1 0.968 (36);
  - reverse-fee kind and remedy, 27/28 each.
  These are the extension and cap errors 7.20 described: a marketing-period extension coded `other`, a condition stage 1 did not reach, and the dropped per-seller cap entry.

**What this does and does not show.** The two post-checks are narrow, and on the six holdouts they fire only where a written rule says they should. All six M8 gold changes moved gold toward what the extractor or a post-check produced; the three rulings that went the other way kept gold as it was. So the pooled gain is partly a matter of definition: it shows the rules are now consistent, not that extraction improved. Only a fresh blind holdout under v2.11 measures that.

**Next steps.**
1. **A blind M9 holdout under v2.11.** Pool it with M7 and M8 to reach about 150 US agreements, which is near the 157 cases needed to show 0.97 with one error.
2. **Extension conditions and counts.** Rule a marketing-period extension explicitly. Check that stage 1 keeps the outside-date section's extension provisos.
3. **Cap amounts.** Add a post-check that restores the per-seller "portion of the price received" entry when the text has one and the extraction dropped it (v2.8 rule 5).

### 7.22 M9 holdout: blind, schema v2.11, pooled with M7 and M8 (2026-10-24)

This is step 1 from 7.21.

**Setup.**
- **Sample:** 50 fresh agreements, drawn with the seed `'m9holdout'` and the same strata as M3 to M8. M1 to M8 are excluded (`.claude/span-taxonomy-runs/m9holdout/sample.sql`).
- **Stage 1:** the union of the v7 rules and the Haiku picker (7.14).
- **Extraction:** Sonnet, one run, under schema v2.11, followed by the post-checks.
- **Gold:** Opus read the full text and wrote gold without seeing the candidates or the extraction. Opus then adjudicated every disagreement under v2.11.
- **Blind:** schema v2.11 and the post-checks were frozen before the sample was drawn, and nothing was changed after seeing M9's results.
- **Non-US agreements:** five are drafted under non-US law: English (44a0670c), British Columbia (acbdeb56), German (b6f62e1d), Alberta (8e3b203c) and Irish (cba1f9a5). They are tagged `holdout_non_us`. The headline numbers are the 45 US agreements; all 50 are in `pilot/m9_holdout_results.json`.
- **Pooled numbers:** these add M7 run 2 and M8 (7.21), for 142 US agreements.

**Stage 1.**
- It reaches 956 of 962 gold spans (0.994), from a median of 48% of the text. The v7 rules alone would show 43.5%.
- The 6 misses are 3 outside-date spans, 2 MAE spans and 1 financing span.

**Post-checks.** Three fired, all the v2.5 rule 3 `cash_on_hand` check. The two v2.11 checks did not fire; the extractor coded both patterns itself where they occurred.

**Adjudication.** 44 verdicts:

| Verdict | Count |
|---|---|
| Gold right | 30 |
| Extractor right (gold fixed) | 10 |
| Schema ambiguous | 3 |
| Equivalent | 1 |

Of the 30 extractor errors:
- 12 are rules it had but missed;
- 5 are wrong codes;
- 5 are text stage 1 did not reach;
- 5 are text it was shown but missed;
- 3 are wrong values.

**Detection** (precision / recall; bar 0.95 and at most 0.1 false positives per agreement):

| Concept | M9, 45 US | Pooled, 142 US |
|---|---|---|
| Target fee | 1.00 / 1.00 | 1.00 / 1.00 |
| Reverse fee | 1.00 / 1.00 | 1.00 / 1.00 |
| Expense reimbursement | 1.00 / 1.00 | 0.967 / 1.00 |
| Outside date | 1.00 / 1.00 | 1.00 / 1.00 |
| MAE definition | 1.00 / 1.00 | 1.00 / 0.993 |
| Financing | 0.973 / 1.00 | 0.982 / 0.991 |
| Indemnification cap | 1.00 / 1.00 | 1.00 / 1.00 |
| Basket | 1.00 / 1.00 | 1.00 / 1.00 |
| Survival | 1.00 / 1.00 | 1.00 / 1.00 |
| 338 election | 1.00 / 1.00 | 0.938 / 1.00 |

- **On M9, all ten concepts meet the detection bar.** This is the first blind holdout where they all do. The one false positive is financing in a stock-for-stock de-SPAC with no buyer cash obligation (68d7e2e2), which v2.2 rule 9 already excludes.
- **Pooled, nine of ten meet it.** The 338 election misses on a single false positive (15 of 16), the 336(e) representation in M8 (e2b03c9a). Expense reimbursement now meets the bar (0.967).

**Fields** (exact match for scalar fields, set F1 for set fields; bar 0.97):
- **M9 alone, 28 of 36 meet the bar** (26 against raw gold, before adjudication). The eight that fail are mostly single errors on small counts:
  - committed financing types, set F1 0.925 (36 cases);
  - escrow sole recourse, 12/14;
  - fundamental-rep survival and the MAE disproportionate-effect exception, 43/45 each;
  - the expense-reimbursement cap, 10/11, and the target-fee remedy, 21/22;
  - outside-date extension conditions (set F1 0.902) and the number of extensions (41/43).
- **Pooled over 142 US agreements, 31 of 36 meet the bar**, the same count as M7 and M8 alone, but a different five fail:
  - outside-date extension conditions, set F1 0.915 (138 cases);
  - the number of extensions, 0.957 (138);
  - committed financing types, set F1 0.963 (109);
  - escrow sole recourse, 48/50;
  - the expense-reimbursement cap, 28/29.
- **Newly meeting the bar when pooled:** cap amounts (set F1 0.978, 50 cases) and reverse-fee kind and remedy (41/42 each).
- **Shown at 0.97 with 95% confidence:** general-rep survival and "no survival" (142/142 each, lower bound 0.979), and the outside-date effect (138/138, 0.979). The fields with one or two errors in 109 to 141 cases now have lower bounds of 0.94 to 0.97.

**What the errors are.**
- **Committed financing (6 records):**
  - in four, the extractor dropped `cash_on_hand` from a funds representation that counts available cash beside the debt or equity ("when combined with other funds available to Parent"), which v2.7 rule 2 covers;
  - in two, it dropped a parent guarantee coded `other` (v2.8 rule 1), one of them outside the candidates;
  - in one, it coded commitment letters that were still to be delivered.
  The v2.5 rule 3 post-check catches only a funds representation with no other source.
- **MAE carve-outs (7 records):** industry wording, a transaction-cost carve-out and a credit-rating carve-out. They pass as a set (F1 0.98).
- **Outside-date extensions:**
  - an HSR extension whose condition stage 1 did not reach (39857b40);
  - one extension split in two (8d343a41);
  - a specific-performance extension the extractor missed (0e7cbf22).
  This field fails on every holdout.
- **Scoring artifact:** a5fad085 fundamental-rep survival, `offset_days` null against 0, was ruled equivalent but scores as an error.
- **Schema-ambiguous items:**
  - a survival period that is the later of a duration and a fixed date (b6f62e1d);
  - one that is the earlier of the statute of limitations and six years (389ccf69);
  - whether a lowercase MAE on the buyer's ability to close adds a second definition beside a defined MAE (5fa2b324).

**Cost.** About 60K Sonnet tokens per agreement for the ten concepts, plus about 27K for the Haiku picker reading outlines.

**Next steps.**
1. **A post-check for funds representations that combine sources.** When a funds representation counts "available cash", "cash on hand" or "other funds" beside the commitments, add `cash_on_hand` (v2.7 rule 2).
2. **Rulings on the three ambiguous items,** and treat a null `offset_days` as 0 in scoring.
3. **Outside-date extensions.** Rule how to count a single extension measured by several delays. Check that stage 1 keeps the conditions an extension cites.
4. **Keep pooling.** At 142 agreements, a common field with one error has a lower bound of about 0.967, just short of 0.97. Another 50-agreement holdout takes the common fields to about 190 cases: enough to show 0.97 with one error (157 needed), not with two (208).

### 7.23 Schema v2.12, the combined-sources post-check, M7 to M9 re-scored (2026-10-25)

This is steps 1 to 3 from 7.22. It is not blind: the ruling and the check were written after seeing M9.

**Schema v2.12** (`pilot/schema_v2_12.md`), four rulings:
1. **Funds representations that combine sources.** A representation that counts the buyer's own funds beside its commitments adds `cash_on_hand` next to `debt` or `equity` (v2.7 rule 2). Examples: "when combined with other funds available to Parent", "the Buyer will have sufficient funds", "after giving effect to the Equity Financing, Parent will have sufficient financial resources".
2. **Survival periods that are the earlier or later of two measures.** `kind` is the first measure named, and the same period records the other. This is how gold already coded M8 677f6d1e and M9 b6f62e1d.
3. **An undefined MAE beside a defined one.** An undefined lowercase MAE used operatively for a party with no defined MAE of its own adds an entry for that party. M9 gold does this in four agreements.
4. **One extension measured by several delays.** It is one entry, with condition `other` when the delays differ.

**Post-check.** The v2.5 rule 3 check (`pilot/postchecks.py`) already added `cash_on_hand` beside other sources. It missed these cases only because its wording patterns did not cover them.
- **What changed:** it now also reads "other funds available", "cash-on-hand", and the buyer as the subject of "has", "have" or "will have sufficient funds, cash or financial resources".
- **Fires, compared with gold on M3 to M9:**
  - the first draft added seven fires, three of them wrong: a covenant not to reduce the financing (M3 b0d8da74), a default representation (M4 dbdace06), and another entity's funds (M9 abe6927f);
  - requiring the buyer as the direct subject removed all three;
  - the four fires that remain all match gold: M8 1f6673b9 and M9 6693cf04, 7d951084 and c815853e.
- **Tests:** seven new tests in `test_postchecks.py`, 84 in all.

**Gold.** One record changed. 389ccf69 (M9) gets the six-year duration on its three "earlier of" survival periods (rule 2). The change does not favor the extractor: it coded the fundamental period as a 6-year duration, so it is still wrong. The other three rulings keep gold as it was, and in two of them the extractor is wrong:
- b6f62e1d is missing the date;
- 5fa2b324 is missing the buyer-side entry.
The M9 gold before these changes is in `out/gold_pre_v212`.

**Scorer.** `score_v2.period` counts a null `offset_days` as 0. This fixes the a5fad085 record that 7.22 ruled equivalent, and one fundamental-reps record in each M7 run.

**Results.**
- **M9 (45 US):** fields meeting the bar go from 28 to 29 of 36.
  - Fundamental-rep survival passes (44/45).
  - Committed financing types rise to set F1 0.954 (33/36 exact) but still fail. The three left are two parent guarantees coded `other` (one of them outside the candidates) and commitment letters not yet delivered (a1cd2f32).
- **M8:** committed financing rises to set F1 0.983.
- **M7:** both runs now meet the bar for 33 of 36 fields.
- **Pooled (142 US):**

| | 7.22 | v2.12 |
|---|---|---|
| Fields meeting 0.97 | 31 of 36 | 32 of 36 |
| Committed financing types | set F1 0.963 | 0.974 |
| Fundamental-rep survival | 138/142 | 140/142 |
| Concepts meeting the detection bar | 9 of 10 | 9 of 10 |

- **Four fields still fail:**
  - outside-date extension conditions, set F1 0.915 (138 cases);
  - the number of extensions, 132/138;
  - escrow sole recourse, 48/50;
  - the expense-reimbursement cap, 28/29.
- **Detection:** the 338 election still misses on M8's one false positive.
- **Fields shown at 0.97 with 95% confidence:** still general-rep survival, "no survival" and the outside-date effect.

**What this does and does not show.** The check fires only where gold has `cash_on_hand`, on all seven holdouts. But the wording was chosen after seeing M9's misses, so the gain on M9 is not a blind measurement.

**Next steps.**
1. **Outside-date extensions,** the one field that fails on every holdout. The misses recorded in 7.20 and 7.22 include two conditions stage 1 did not reach, the split extension that rule 4 now covers, a marketing-period extension coded `other`, and a missed specific-performance extension. Check that stage 1 keeps every section the outside-date clause cites, as 7.21 did for the cited covenants.
2. **Another blind holdout under v2.12,** to measure the check and rulings blind and to take the common fields past the 157 cases needed to show 0.97 with one error.

### 7.24 Outside-date extensions: cited sections, schema v2.13, M7 to M9 re-scored (2026-10-25)

This is step 1 from 7.23. It is not blind: the stage 1 change, the rulings and the checks were all written after seeing the misses.

**The misses.** Under v2.12, nine agreements across the M7 runs, M8 and M9 had a wrong extension count or condition:
- **Split counts (3):** one grant recorded as two entries. These are M7 run 1 cbc1bf49 ("up to two times", split by date), M7 run 2 04d3f8ef (alternative routes) and M9 8d343a41 (one extension measured by several delays).
- **Missed specific-performance extensions (2):** M7 run 2 4c95f1e9 and M9 0e7cbf22. Both sit in the specific-performance section.
- **Conditions cited only by number, with the cited section outside the candidates (2):** M9 39857b40, whose §7.1(b) is the HSR condition, and M8 92491b2d, whose §§7.4 and 8.4 are titled "Regulatory Consents and Approvals".
- **A marketing period coded `other` (1):** M8 6bae3995. Its gold also recorded a bar on terminating as an extension.
- **A missed financial-information extension (1):** M8 1f6673b9. No rule addresses it.

**Stage 1.** `prepare_v7.outside_date_citations` finds each sentence that extends the outside date and keeps every section it cites by number, as 7.21 did for cited covenants. Candidates change for six M8 agreements and two M9 agreements, and for none in M7. The outside date was re-extracted for those eight (Sonnet, `briefs/extract_od.md`, this concept only), with the other nine concepts kept from the v2.12 run. M6 gains two cited sections too, but it was left alone: its candidates also differ in 16 places from older stage 1 drift.

**Schema v2.13** (`pilot/schema_v2_13.md`), six rulings:
1. **Counting.** Each extension right as drafted is one entry. "Up to two (2) times" is one entry with `count` 2. Alternative routes to one extension are one entry, coded `other` when the routes differ in kind. Separately drafted grants are separate entries.
2. **Specific-performance extensions** count wherever they sit (v2.5 rule 8).
3. **Conditions cited by number** are coded by what the cited condition is. An antitrust, HSR or regulatory condition is `regulatory_pending`.
4. **Marketing or financing periods** are `marketing_period`.
5. **A bar on terminating is not an extension** (v2.5 rule 8, v2.3 rule 12).
6. **One test led by regulatory approvals** is `regulatory_pending`, even when it also lists the S-4, a related injunction or a separation condition. This was written during the run. The first re-extraction of M8 0522c1f4 read rule 1's `other` as covering such a test and downgraded a condition that the v2.12 run and gold both coded `regulatory_pending`. After rule 6, a second extraction coded it as gold does.

**Post-checks** (`pilot/postchecks.py`, one per rule 1 to 4):
- **Rule 1** merges entries whose quotes overlap: one inside the other, or one's end the other's start by 20 or more characters. The count is the larger of the two. Across M3 to M9 it fires three times, all correct: cbc1bf49, 04d3f8ef and 8d343a41.
- **Rule 2** adds the specific-performance extension when no entry mentions specific performance or a pending action. It fires on 4c95f1e9, 0e7cbf22 and M9 74ac81e0. On 74ac81e0, gold had missed the extension and was fixed.
- **Rule 3** reads the titles and first 300 characters of cited clauses, and fires only when the extension's condition text names a condition. Its first draft fired wrongly on a cited termination clause that mentions HSR further down (fb9dae77) and on "regulatory body" in a market-MAC clause (1f6673b9). Narrowing it fixed both, at the cost of one correct fire (89ba8925). On the new candidates it does not fire on M7 to M9: with the cited sections in hand, the extractor codes both cited-condition cases correctly itself.
- **Rule 4** recodes `other` to `marketing_period` when the text names a marketing or financing period. It fires on 6bae3995.
- **Tests:** 12 new in `test_postchecks.py` and 2 in `test_prepare_v7.py`, 99 in all.

**Gold.** Two records changed, both marked adjudicated. The pre-change gold is in `out/gold_pre_v213`.
- M8 6bae3995 loses the bar on terminating (rule 5).
- M9 74ac81e0 gains the specific-performance extension it missed (rule 2).

**Results.**

| | v2.12 (7.23) | v2.13 |
|---|---|---|
| Pooled fields meeting 0.97 (142 US) | 32 of 36 | 34 of 36 |
| Extension conditions | set F1 0.915, 132/138 exact | 1.000, 138/138 |
| Number of extensions | 132/138 | 137/138 |
| M7 run 1 / run 2 fields | 33 / 33 | 33 / 34 |
| M8 (47 US) fields | 29 | 31 |
| M9 (45 US) fields | 29 | 31 |
| Fields shown at 0.97 with 95% confidence | 3 | 4 (adds extension conditions) |

- **The one extension miss left** is 1f6673b9, the financial-information extension.
- **Still failing in the pool:** escrow sole recourse (48/50) and the expense-reimbursement cap (28/29).
- **Detection:** unchanged. The 338 election still misses on M8's one false positive.

**What this does and does not show.** Every miss now has a rule, a check or a stage 1 fix behind it, and the checks fire nowhere else on M3 to M9. But all of it was written against these misses, so 138/138 is a fit to M7 to M9, not a measurement. Two other caveats apply:
- Only the outside date was re-extracted, and only for the eight agreements whose candidates changed (plus 0522c1f4 again), so this checks the new candidates rather than replacing a full run.
- Rule 6 was written to undo a regression the re-extraction caused, which is the clearest sign the v2.13 wording can mislead the extractor.

**Next steps.**
1. **A blind holdout under v2.13** with the new stage 1. It would measure the extension fields blind and take the common fields past the 157 cases needed to show 0.97 with one error.
2. **Escrow sole recourse and the expense-reimbursement cap,** the two fields still failing, each on one or two cases.

### 7.25 M10 holdout: blind, schema v2.13, pooled with M7 to M9 (2026-10-29)

This is step 1 from 7.24.

**Setup.**
- **Sample:** 50 fresh agreements, drawn with the seed `'m10holdout'` and the same strata as M3 to M9.
- **Exclusions:** the M3 to M9 manifests had been deleted as scratch, so the draw excludes every agreement uuid that appears in the earlier pilot sessions' transcripts (772, a superset of M1 to M9) plus the current M1 draw. With the 50 M10 agreements added, it is saved as `pilot/holdout_uuids_used.txt`, so the next draw does not depend on scratch.
- **US only:** unlike M8 and M9, non-US agreements were screened out at the draw by their governing-law clause. Two were replaced by the next agreement in the same stratum: 70eaaf56 (English law) and 297cb7a8 (British Columbia). All 50 count toward the headline.
- **Stage 1:** the v7 rules, including `outside_date_citations` (7.24), in union with the Haiku picker.
- **Extraction:** Sonnet, one run, under schema v2.13, followed by every post-check in `postchecks.py`.
- **Gold:** Opus read the full text and wrote gold without seeing the candidates or the extraction. Opus then adjudicated every disagreement under v2.13.
- **Blind:** v2.13, stage 1 and the post-checks were frozen before the draw, and nothing was changed after seeing M10.
- **Pooled numbers:** these add M10 to the 142-agreement pool after v2.13 (7.24), for 192 US agreements. The M7 to M9 extractions and gold were deleted as scratch, so they enter from their saved counts in `m9_v213_results.json`. Detection and scalar fields pool exactly. Set F1 there was saved without the matches behind it, so a pooled set F1 is the mean of the two inputs weighted by cases (`pool_holdouts.py`, marked `set_f1_approx`). M10's own counts are saved in full.

**Stage 1.**
- It reaches 926 of 939 gold spans (0.986), from a median of 42% of the text. The v7 rules alone would show 38%.
- The 13 misses are 5 MAE spans, 3 cap spans, 2 outside-date spans, 2 financing spans and 1 target-fee span. Recall is lower than on M8 (0.996) and M9 (0.994).

**Post-checks.** Six fired, four of them correctly:
- the merge of a split extension (v2.13 rule 1, b4b56b47);
- a condition cited by number (v2.13 rule 3, 050572a3);
- a recommendation-failure trigger read from a definition (v2.2 rule 12, 2d3cf3b5);
- the v2.8 rule 8 escrow check, three times. It was right once (84d63ad6) and wrong twice (e1533237, ec5c1541). In both wrong fires the "first from the escrow, then from the sellers" sentence governs fundamental-representation claims only, while general-representation claims stay limited to the escrow. v2.5 rule 6 makes that true.

**Adjudication.** 50 disputed items, 55 verdicts:

| Verdict | Count |
|---|---|
| Gold right | 39 |
| Extractor right (gold fixed) | 12 |
| Both wrong (gold fixed) | 1 |
| Schema ambiguous | 3 |

Of the 40 extractor errors:
- 14 are rules it had but missed;
- 13 are wrong codes;
- 8 are text stage 1 did not reach;
- 3 are wrong values;
- 2 are text it was shown but missed.

Two adjudicators ruled the same pattern differently: an MAE defined under two synonyms ("Material Adverse Effect" and "Material Adverse Change"). v2.3 rule 10 gives each defined term an entry, and both gold files already did, so the 96016511 verdict is recorded as gold right.

**Detection** (precision / recall; bar 0.95 and at most 0.1 false positives per agreement):

| Concept | M10, 50 US | Pooled, 192 US |
|---|---|---|
| Target fee | 1.00 / 1.00 | 1.00 / 1.00 |
| Reverse fee | 1.00 / 1.00 | 1.00 / 1.00 |
| Expense reimbursement | 1.00 / 1.00 | 0.973 / 1.00 |
| Outside date | 1.00 / 1.00 | 1.00 / 1.00 |
| MAE definition | 1.00 / 0.979 | 1.00 / 0.990 |
| Financing | 1.00 / 0.971 | 0.986 / 0.986 |
| Indemnification cap | 1.00 / 1.00 | 1.00 / 1.00 |
| Basket | 1.00 / 1.00 | 1.00 / 1.00 |
| Survival | 1.00 / 1.00 | 1.00 / 1.00 |
| 338 election | 1.00 / 1.00 | 0.958 / 1.00 |

- **All ten concepts meet the detection bar, on M10 and pooled.** This is the first time the pool does: the 338 election reaches 23 of 24 (0.958), with M8's 336(e) representation still its only false positive.
- **The two misses on M10:**
  - an undefined lowercase MAE in sections stage 1 did not reach (a3f48f73);
  - a parent covenant to cause the buyer to pay, coded financing under v2.8 rule 1 and v2.9 rule 9 (b9a3b058). The adjudicator called it borderline.
- **Against raw gold the extractor had two 338 false positives.** Both turned out to be gold errors: a prohibition on any Section 338 election (eaed6407 and 53ce24a6), which v2.5 rule 12 codes as present.

**Fields** (exact match for scalar fields, set F1 for set fields; bar 0.97):
- **M10 alone, 30 of 36 meet the bar** (27 against raw gold, 26 before the post-checks). The six that fail:
  - target-fee triggers, set F1 0.959 (14/21 exact);
  - committed financing types, set F1 0.959 (29/33);
  - cap amounts, set F1 0.923 (21/24);
  - escrow sole recourse, 20/24;
  - the number of MAE definitions, 45/48;
  - the target-fee remedy, 20/21.
- **The extension fields hold blind.** Extension conditions are 47/47 and the number of extensions 46/47. The one miss is an alternative later date that applies only if a permit is not issued, which v2.5 rule 8 counts as an extension (0d7cae08).
- **The expense-reimbursement cap is 7/7.**
- **Pooled over 192 US agreements, 33 of 36 meet the bar.** Three fail:
  - cap amounts, set F1 about 0.96 (68/74 exact);
  - escrow sole recourse, 68/74;
  - the target-fee remedy, 87/90.
- **Newly meeting the bar in the pool:** the expense-reimbursement cap (35/36, 0.972).
- **Shown at 0.97 with 95% confidence: five fields.** They are general-rep survival and "no survival" (192/192 each, lower bound 0.985), the outside-date effect and extension conditions (185/185, 0.984), and the initial outside date (184/185, 0.975). The number of extensions (183/185) has a lower bound of 0.966.

**What the errors are.**
- **Target-fee triggers (7 records):** four are a missing `recommendation_failure`. The recommendation-change definition includes failing to include or reaffirm the recommendation (v2.2 rule 12).
  - In three (fdaa819a, 0e2268be, ef0a3783), that definition sat outside the candidates.
  - In one (bcf14416), it was in them.
  - The other three: entering into an acquisition agreement coded as a recommendation change (8f1ab49b), a mirror-drafted trigger read literally (d69fbb6c), and one schema-ambiguous item.
- **Committed financing (4 records):** all four drop a parent or sponsor guarantee of the buyer's obligations, which v2.8 rule 1 codes `other`. One (8f1ab49b) also drops `cash_on_hand` from a funds representation conditioned on the commitments being funded.
- **Escrow sole recourse (4 records):**
  - the two wrong post-check fires;
  - an escrow that only backs the seller's direct liability, which is false under v2.9 rule 5 (7c620a27);
  - a holdback plus an earn-out set-off with no stated order, which is schema-ambiguous (0f3367d4).
- **Cap amounts (3):**
  - an escrow amount defined outside the candidates (96016511);
  - a sum-of-parts price coded `formula` (c01291d8);
  - a per-seller "ratable share" coded as the whole escrow (e1533237).
- **Number of MAE definitions (3):**
  - the synonym pair above (96016511);
  - an extra generic entry beside two party-specific ones (18b8e081);
  - an undefined buyer-side MAE outside the candidates (496ce18b).
- **MAE carve-outs (13 records):** they pass as a set (F1 0.978). As before, most are industry wording that v2.3 rule 11 codes `general_economic` only.
- **Schema-ambiguous items:**
  - a holdback with an earn-out set-off and no order between them (0f3367d4);
  - a tail fee on an outside-date termination that applies only if the vote failed, in an agreement with no separate vote-failure right (d6016227);
  - a de-SPAC condition that the SPAC's own deal close first, when that deal supplies the cash (a90b6e3f).
- **Gold quality:** gold was wrong in 13 of 55 verdicts. Eight of the fixes add something gold missed: two extensions, three MAE entries (two buyer-side, one DOE), two 338 prohibitions and a marketing period.

**Cost.** About 72K Sonnet tokens per agreement for the ten concepts, up from about 60K, plus about 31K for the Haiku picker. Opus spent about 73K per agreement on gold and 0.74M in all on adjudication.

**What this does and does not show.** M10 is the first blind run under v2.13. The extension fields that 7.24 fitted to M7 to M9 hold on new agreements, and the pool now meets the detection bar for every concept. Two caveats apply:
- Every pooled set F1 is approximate, because the M7 to M9 matches were not kept. Exact-match counts, detection and scalar fields are exact.
- M10 alone is weaker than M9 on fee triggers, committed financing and caps. About a third of those errors are text stage 1 did not reach, and the rest are written rules applied inconsistently. Neither is a new kind of provision.

**Next steps.**
1. **Escrow sole recourse.** Keep the v2.8 rule 8 check from firing on a sentence that governs only fundamental-representation, fraud or tax claims.
2. **Recommendation-failure definitions outside the candidates.** Read them from the full text, as the cited-covenant check does (7.21).
3. **Guarantees as committed financing.** Add `other` when a parent or sponsor guarantees the buyer's payment obligations (v2.8 rule 1).
4. **Rulings** on the three ambiguous items and on MAE synonyms.
5. **Keep pooling, and keep raw counts.** The common fields now reach about 185 to 192 cases. That is enough to show 0.97 with one error (157 needed), not with two (208).

### 7.26 Escrow scope, split definitions, schema v2.14, M10 re-scored (2026-10-29)

This is steps 1, 2 and 4 from 7.25, and the second step 7.24 set: the two fields still failing in the pool. It is not blind: every change was written after seeing M10's misses.

**The two failing fields from 7.24.**
- **Escrow sole recourse** still fails, and M10 shows a clear cause. Two of its four misses are v2.8 rule 8 fires on the wrong sentence (below).
- **The expense-reimbursement cap** now passes in the pool (35/36, 0.972), and M10 has it right 7 times out of 7. Its one miss is M9's (10/11). M9's inputs were deleted, so no cause can be checked, and nothing was changed.

**Post-checks** (`pilot/postchecks.py`):
- **Escrow order for some claims only** (v2.5 rule 6). The v2.8 rule 8 check now skips a "first from the escrow, then from the sellers" sentence in two cases:
  - it covers only Fundamental Representations and does not also name the general ones;
  - its last source is "the breaching" holder, meaning a holder's own breach.
  On M10 this removes both wrong fires (e1533237, ec5c1541). The ec5c1541 sentence on its holders' own breaches fired once the first sentence was skipped, which is what led to the second exclusion. The right fire on 84d63ad6 still fires.
- **Recommendation-change definitions** (v2.2 rule 12). The check now rejoins a sentence the text layer split across lines: a line that starts in lowercase, or one that follows an unclosed quotation mark. When the candidates hold no such definition, it reads the full text. A quote that would straddle the join is cut back to one line. On M10 it adds the three triggers gold has: 0e2268be and bcf14416 (split lines) and fdaa819a (definition outside the candidates). It adds none that gold lacks.
- **Tests:** seven new tests in `test_postchecks.py`. Each fails without its change (checked by disabling it). There are 108 tests in all, with `test_pool_holdouts.py`.

**Not done:**
- **A guarantee post-check.** All four committed-financing misses on M10 drop a parent or sponsor guarantee (v2.8 rule 1). But "guarantee" or "guarantor" also appears in seven M10 agreements whose gold correctly has no `other`. M3 to M9 are not on disk to test a narrower pattern against, so no check was written.
- **The other two escrow misses:**
  - an escrow that only backs the seller's direct liability, coded `not_stated` (7c620a27, v2.9 rule 5);
  - a holdback and earn-out set-off (0f3367d4), now ruled below.

**Schema v2.14** (`pilot/schema_v2_14.md`), five rulings:
1. **MAE synonyms.** A definition of two synonymous terms gets an entry for each (v2.3 rule 10).
2. **Escrow order for some claims only.** It does not decide `escrow_sole_recourse`; the general representations do. This is the post-check above.
3. **Holdback and earn-out set-off with no order.** With no direct claim against the sellers this is true (v2.8 rule 3). v2.8 rule 8 needs a stated order.
4. **An outside-date tail fee that also needs a failed vote.** It is `outside_date_with_tail` alone when no termination right exists for the failed vote.
5. **A de-SPAC condition that the funding deal close first.** It is a financing condition.

All five keep M10 gold as it was, so gold is unchanged. Under rule 3 the extractor's `false` on 0f3367d4 is now an error.

**Results.**

| | M10, 7.25 | M10, v2.14 | Pool, 7.25 | Pool, v2.14 |
|---|---|---|---|---|
| Fields meeting 0.97 | 30 of 36 | 31 of 36 | 33 of 36 | 33 of 36 |
| Escrow sole recourse | 20/24 | 22/24 | 68/74 | 70/74 |
| Target-fee triggers | set F1 0.959, 14/21 exact | 0.973, 17/21 | about 0.980, 73/90 | about 0.983, 76/90 |
| Concepts meeting the detection bar | 10 | 10 | 10 | 10 |
| Fields shown at 0.97 with 95% confidence | 0 | 0 | 5 | 5 |

- **Target-fee triggers now pass on M10.**
- **Still failing on M10:** committed financing types (0.959), cap amounts (0.923), escrow sole recourse (22/24), the number of MAE definitions (45/48) and the target-fee remedy (20/21).
- **Still failing in the pool:**
  - escrow sole recourse (70/74, 0.946);
  - cap amounts (set F1 about 0.96);
  - the target-fee remedy (87/90).

**What this does and does not show.** The two check changes fire only where M10 gold agrees, and the tests pin each case. Three things are weaker than in earlier rounds:
- **M7 to M9 could not be re-run.** Their extractions were deleted. So the pool mixes M7 to M9 under the v2.13 checks with M10 under v2.14, and the new checks were not tested for wrong fires on M3 to M9, as 7.21 and 7.23 were.
- **The gains are a fit to M10, not a measurement.** That includes 2 of the 4 escrow cases and 3 of the 7 trigger cases.
- **Escrow sole recourse is now the field most likely to fail blind.** It has two M10 misses no check covers, plus the two unexplained M7 to M9 misses.

**Next steps.**
1. **Keep the per-agreement JSON of the next holdout** (or at least `score_v2`'s raw counts per holdout), so later checks can be tested across holdouts and pooled set F1 is exact.
2. **A blind M11 under v2.14,** drawn excluding `pilot/holdout_uuids_used.txt`. It would measure the escrow and trigger changes blind.
3. **Committed financing guarantees.** Rule which guarantees count under v2.8 rule 1: a parent or sponsor guarantee of the buyer's payment obligations, not a guarantee of a merger subsidiary by its own parent. Then test a post-check on M11.

### 7.27 M11 holdout: blind, schema v2.14, pooled with M7 to M10 (2026-10-29)

This is step 2 from 7.26.

**Setup.**
- **Sample:** 50 fresh agreements, drawn with the seed `'m11holdout'` and the same strata as M3 to M10, excluding the 823 uuids in `pilot/holdout_uuids_used.txt`. The M11 uuids are now added to it (873 in all).
- **US only:** non-US agreements were screened out at the draw, as in M10. One was replaced by the next agreement in its stratum: 46bfb0aa (British Columbia). All 50 count toward the headline.
- **Stage 1, extraction, gold and adjudication:** as in 7.25, under schema v2.14 with every post-check in `postchecks.py`, including the two 7.26 changes. Nothing was changed after the draw.
- **Data kept.** Unlike M3 to M10, the per-agreement gold, extractions, post-checked outputs, verdicts and briefs are kept in `.claude/span-taxonomy-runs/m11holdout/`. M11's own set F1 is exact. Pooled set F1 is still approximate, because M7 to M10 enter from the saved totals in `m10_v214_results.json`.

**Stage 1.**
- It reaches 961 of 971 gold spans (0.990), from a median of 42% of the text. The v7 rules alone would show 38% and reach 956 (0.985); the picker alone would show 21% and reach 927 (0.955).
- Eight of the 10 misses are MAE spans, mostly an undefined buyer-side "material adverse effect" in a buyer representation or bring-down condition. The other two are one outside-date span and one 338 span.

**Post-checks.** One fired, and it was wrong. The v2.13 rule 4 check recoded an extension as `financing_pending` on e20827af. That extension runs when every condition is met before the Termination Date and has nothing to do with a marketing or financing period, so gold's `other` stands. This costs the extension-conditions field its only M11 miss.
- The two 7.26 changes did not fire. The escrow check saw no "first from the escrow" sentence, and every recommendation-failure trigger gold has, the extractor already had.

**Adjudication.** 41 disputed items, 51 fields:

| Verdict | Count |
|---|---|
| Gold right | 31 |
| Extractor right (gold fixed) | 7 |
| Both wrong | 0 |
| Schema ambiguous | 3 |

Of the 31 extractor errors:
- 13 are rules it had but missed;
- 10 are wrong codes;
- 5 are text stage 1 did not reach;
- 2 are text it was shown but missed;
- 1 is a wrong value.

**Detection** (precision / recall; bar 0.95 and at most 0.1 false positives per agreement):

| Concept | M11, 50 US | Pooled, 242 US |
|---|---|---|
| Target fee | 0.963 / 1.00 | 0.991 / 1.00 |
| Reverse fee | 1.00 / 0.950 | 1.00 / 0.987 |
| Expense reimbursement | 1.00 / 1.00 | 0.978 / 1.00 |
| Outside date | 1.00 / 1.00 | 1.00 / 1.00 |
| MAE definition | 1.00 / 1.00 | 1.00 / 0.992 |
| Financing | 1.00 / 1.00 | 0.989 / 0.989 |
| Indemnification cap | 1.00 / 1.00 | 1.00 / 1.00 |
| Basket | 1.00 / 1.00 | 1.00 / 1.00 |
| Survival | 1.00 / 1.00 | 1.00 / 1.00 |
| 338 election | 1.00 / 1.00 | 0.962 / 1.00 |

- **All ten concepts meet the detection bar, on M11 and pooled.**
- **Both M11 detection errors come from one agreement** (76955f6e). In this merger of equals both companies' shares convert into a new holding company's stock. The holding company is a subsidiary of one of them, so v2.1 rule 11 makes that one the buyer. The extractor put the parties on the other sides, so its fee is a target fee where gold has a reverse fee.

**Fields** (exact match for scalar fields, set F1 for set fields; bar 0.97):
- **M11 alone, 26 of 36 meet the bar** (27 before the post-checks; the difference is the wrong fire above). Most of the ten that fail miss once or twice. Those that miss more:
  - **the number of MAE definitions, 40/50.** Five misses are an undefined buyer-side MAE that v2.12 rule 3 adds as an entry, all in text stage 1 did not reach; one of them also drops a separately defined synonym. Four are an extra entry for a single generic definition, which v2.3 rule 10 counts once per defined term. One is schema-ambiguous (below).
  - **committed financing types, set F1 0.955 (33/37):** an affiliate's covenant to fund the buyer (v2.9 rule 9), unnamed commitment letters with draw-down conditions (debt, v2.7 rule 2), a shareholder subscription for the buyer's stock (equity), and a guarantee of the buyer's payment obligations (v2.8 rule 1).
  - **the reverse-fee remedy, 17/19:** a general no-further-liability clause read as the fee's remedy (v2.2 rule 1), and one schema-ambiguous item.
- **What held blind:**
  - target-fee triggers, set F1 0.996 (25/26 exact);
  - escrow sole recourse, 20/21. Its one miss is a seller who stays directly liable up to the cap beyond the escrow, which is false under v2.9 rule 5 (4b76d9e6). That is the same pattern as the M10 miss 7c620a27, which no check covers.
  - cap amounts, 21/21;
  - the expense-reimbursement cap, 9/9;
  - the number of extensions, 46/47.
- **Pooled over 242 US agreements, 30 of 36 meet the bar**, down from 33 of 36 over 192. Six fail:
  - the number of MAE definitions, 224/239 (new);
  - committed financing types, set F1 about 0.968 (new);
  - the reverse-fee remedy, 74/77 (new);
  - cap amounts, set F1 about 0.969;
  - escrow sole recourse, 90/95;
  - the target-fee remedy, 112/116.
- **Shown at 0.97 with 95% confidence: eight fields**, up from five:
  - general-rep survival and "no survival", 241/241 each (lower bound 0.988);
  - the outside-date effect, 232/232 (0.987);
  - extension conditions and the initial outside date, 231/232 each (0.980);
  - the target-fee payor, 116/116 (0.975);
  - fundamental-rep survival, 239/241 (0.974);
  - the marketing period, 178/179 (0.974).
  The number of extensions (229/232) has a lower bound of 0.967.

**Schema-ambiguous items:**
- **An MAE defined for the target only but applied to the buyer** (42deaef0). In this de-SPAC the capitalized term is defined for the Company, but a representation and the buyer bring-down apply it "on the Parent Parties". v2.3 rule 10 gives one entry per defined term; the purpose of v2.12 rule 3 suggests a buyer entry too.
- **"Commercially reasonable best efforts"** (a76cee7f), which mixes two listed financing efforts standards.
- **A sole-remedy sentence that cites only the target fee's section** (704fa6c1), while its wording ("amounts payable by Parent ... as applicable") makes sense only if the reverse fee is included.

**Gold quality.** Gold was wrong in 7 of 41 verdicts. Five of the fixes add something gold missed: two extensions while a specific-performance action is pending (v2.5 rule 8), an affiliate funding covenant, a financing-cooperation covenant and a commodity-price MAE carve-out. The other two remove an `industry` carve-out code and a target-fee trigger.

**Cost.** About 75K Sonnet tokens per agreement for the ten concepts, plus about 30K for the Haiku picker. Opus spent about 71K per agreement on gold and 0.69M in all on adjudication.

**What this does and does not show.**
- **The 7.26 fits mostly hold blind.** Target-fee triggers, escrow sole recourse, cap amounts and the expense-reimbursement cap all meet the bar on M11. The two 7.26 checks did not fire at all, though, so M11 does not test them, for better or worse.
- **The pool's drop from 33 to 30 fields is mostly the MAE definition count.** It is the field most exposed to stage 1: an undefined buyer-side MAE sits in a buyer representation or condition that stage 1 does not keep. M10 missed it three times; M11 misses it ten times.
- **The v2.13 rule 4 post-check fired wrongly for the first time.** It is the only post-check fire on M11.

**Next steps.**
1. **Buyer-side MAE retrieval.** Keep the sections where the buyer representations or the target's closing conditions use "material adverse effect" or a capitalized MAE term, as `outside_date_citations` does for the outside date (7.24). M11's ten misses give a test set.
2. **One entry per defined term.** Restate v2.3 rule 10 in the extractor brief with the two common cases: a generic definition used by both parties, and party-specific terms built on it. Rule on the de-SPAC case above.
3. **Narrow the v2.13 rule 4 check** to extensions whose condition names a marketing period, financing or debt, and add the e20827af sentence as a test.
4. **Committed financing.** The four M11 misses are four different rules, so a single post-check would not cover them. Rule on unnamed commitment letters and shareholder subscriptions.
5. **Keep pooling, and keep the per-agreement data.** M11 is the first holdout kept in full, so later checks can be tested against it for wrong fires.

### 7.28 Buyer-side MAE retrieval and the rule 4 check, M11 re-scored (2026-10-29)

This is steps 1 and 3 from 7.27. It is not blind: both changes were written after seeing M11's misses, and M11 is the only holdout on disk to test them against.

**Stage 1** (`prepare_v7.mae_sections`). Two kinds of section are now kept:
- a section of the buyer's representations, or a closing-conditions section, that uses "material adverse effect" (or "change") in lowercase. The lowercase form marks an undefined MAE, the case v2.12 rule 3 adds as a buyer-side entry. The buyer's representations are found by an article or title naming Parent, Purchaser, Buyer, the acquirer or the merger subsidiary.
- a section outside the definitions that defines the MAE inline with a qualifier before "means" ('a "Material Adverse Effect" with respect to any Person means'), which the definition patterns did not read (326a76e2).

**Post-check** (`postchecks.py`). The v2.13 rule 4 check now looks for a marketing or financing period only in an extension's quotes. On e20827af it had matched the extractor's own paraphrase ("e.g., during the financing period"), and the clause names no financing period.

**Tests:** four for `mae_sections` in `test_prepare_v7.py` (a buyer representation, a closing condition, an inline definition, and target-side and defined-term sections left alone) and one for the rule 4 check. There are 113 tests in all.

**Re-run on M11.**
- Stage 1 changed the candidates of 12 agreements. Sonnet re-extracted the MAE definition for those 12 from the new candidates (`briefs/extract_mae.md`, into `out/sonnet_mae`). The instances were spliced into `out/sonnet` (the 7.27 extraction is kept as `sonnet_pre_728`), and the post-checks were re-run.
- The re-extraction raised two new disagreements, adjudicated by Opus. On 85487997 the extractor was right: the buyer's litigation representation uses an undefined MAE on its ability to close, which adds a buyer-side entry, so gold was fixed. On 326a76e2 gold was right: a disposition made under the agreement's own terms is `compliance_with_agreement`.
- The "before" numbers below are the 7.27 extraction scored against the current gold, so both columns use the same gold.

**Results.**

| | Before | After |
|---|---|---|
| Stage 1, gold spans reached | 961 of 972 (0.989) | 969 of 972 (0.997) |
| Median candidate share (union) | 42.3% | 43.2% |
| Post-check fires on M11 | 1 (wrong) | 0 |
| M11, fields meeting 0.97 | 26 of 36 | 28 of 36 |
| M11, number of MAE definitions | 39/50 | 43/50 |
| M11, extension conditions | 46/47 | 47/47 |
| Pool, fields meeting 0.97 | 30 of 36 | 30 of 36 |
| Pool, number of MAE definitions | 223/239 | 227/239 |
| Pool, extension conditions | 231/232 | 232/232 |
| Pool, fields shown at 0.97 with 95% confidence | 8 | 8 |

- Stage 1 now misses one MAE span, in a seller representation (c42461f3), plus one outside-date and one 338 span.
- The MAE ability-to-consummate prong reaches the bar on M11 (49/50), because the buyer entries the re-extraction now finds have the right prongs.
- **The number of MAE definitions still fails** (0.950 pooled). Seven M11 misses are left, and none is retrieval:
  - **three buyer-side entries the extractor saw and left out** (a709ba58, 3774f6db, e3d37e9c). It read v2.12 rule 3 as applying only to closing conditions, because the rule names a bring-down condition as its typical case; one of its reports says so. All three uses are in buyer representations.
  - **three extra entries for one generic definition** (64aaa139, 704fa6c1, 76955f6e), which v2.3 rule 10 counts once per defined term;
  - **one schema-ambiguous item** (42deaef0, 7.27).

**What this does and does not show.**
- **Retrieval is no longer the cause of the MAE count misses.** What is left is two written rules the extractor reads differently from gold.
- **Stage 1 grows by about one point of text.** The new sections are buyer representations and conditions, which are short.
- **The rule 4 change only removes a fire.** It cannot add one, so it cannot make a new wrong fire. It does drop a right fire whenever the only evidence for a marketing or financing period is in the extractor's paraphrase. The 7.24 test case quotes the clause and still fires.
- **Both changes are fitted to M11,** and no other holdout's data survives to check them for wrong fires or noise. The next blind holdout measures them.

**Next steps.**
1. **Restate v2.12 rule 3** to say that an undefined MAE used operatively in the buyer's representations counts, not only in a condition. Restate v2.3 rule 10 with the generic-definition case (7.27 step 2). Together these would cover six of the seven remaining misses.
2. **A blind M12** under those rulings and this stage 1, keeping its per-agreement data.

### 7.29 M12 holdout: blind, schema v2.15 and the 7.28 stage 1, pooled with M7 to M11 (2026-10-29)

This is the two steps from 7.28: schema v2.15 restates the two MAE counting rules, and a blind M12 tests them with the 7.28 stage 1.

**Schema v2.15** (`pilot/schema_v2_15.md`) is v2.14 plus two restated rules, with no new codes:
1. **An undefined buyer-side MAE** (v2.12 rule 3) counts wherever it is used operatively, including in the buyer's own representations, not only in a closing condition.
2. **Count defined terms, not parties** (v2.3 rule 10). A term that applies to any party is one entry, `both`. Party-specific terms built on a generic definition get one entry each, and the generic term gets none unless it is used on its own. A term defined for one party but also applied to the other stays one entry, `both`.

**Setup.**
- **Sample:** 50 fresh agreements, drawn with the seed `'m12holdout'` and the same strata as M3 to M11, excluding the 873 uuids in `pilot/holdout_uuids_used.txt`. The M12 uuids are now added to it (923 in all).
- **US only:** three were replaced by the next agreement in their stratum: 254d775e (Belgium), da442c51 (English law) and 34e1642b (Finland). All 50 count toward the headline.
- **Stage 1:** `prepare_v7` with the 7.28 `mae_sections`, unioned with the Haiku picker. **Extraction, gold and adjudication:** as in 7.27, under v2.15 with every post-check in `postchecks.py`. Nothing was changed after the draw.
- **Data kept** in `.claude/span-taxonomy-runs/m12holdout/`. Pooled set F1 is approximate, because M7 to M11 enter from the saved totals in `m11_s728_results.json`.

**Stage 1.**
- It reaches 1,037 of 1,048 gold spans (0.990), from a median of 47% of the text. The v7 rules alone would show 41% and reach 1,023 (0.976); the picker alone would show 26% and reach 1,001 (0.955).
- Eight of the 11 misses are MAE spans. The adjudicated ones are a generic MAE definition at the end of the agreement (775de59e), a defined "Excluded Matter" that the carve-outs name (41b903c1), a target-side term with no "Material Adverse" in it, "Material Impact" (daef34e9), and an undefined MAE in a representation by a third party, the target's investment adviser (d39418dd). The other three misses are one span each for the expense reimbursement, the cap and the target fee.
- Two outside-date fields also failed on retrieval: a date set in an appendix (daef34e9) and an "End Time" definition (63f75a60).

**Post-checks.** Four fired, all the v2.2 rule 12 check adding a recommendation-failure trigger, and all four agree with gold. They lift target-fee triggers from set F1 0.966 to 0.983 (17/23 exact to 21/23).

**Adjudication.** 36 disputed items, split into 39 verdicts:

| Verdict | Count |
|---|---|
| Gold right | 30 |
| Extractor right (gold fixed) | 5 |
| Both wrong | 1 |
| Schema ambiguous | 3 |

Of the 31 extractor errors:
- 9 are wrong codes;
- 8 are rules it had but missed;
- 6 are text stage 1 did not reach;
- 6 are wrong values;
- 2 are text it was shown but missed.

**Detection** (precision / recall; bar 0.95 and at most 0.1 false positives per agreement):

| Concept | M12, 50 US | Pooled, 292 US |
|---|---|---|
| Target fee | 1.00 / 1.00 | 0.993 / 1.00 |
| Reverse fee | 0.923 / 1.00 | 0.989 / 0.989 |
| Expense reimbursement | 1.00 / 1.00 | 0.983 / 1.00 |
| Outside date | 1.00 / 0.978 | 1.00 / 0.996 |
| MAE definition | 1.00 / 1.00 | 1.00 / 0.993 |
| Financing | 1.00 / 1.00 | 0.991 / 0.991 |
| Indemnification cap | 1.00 / 1.00 | 1.00 / 1.00 |
| Basket | 1.00 / 1.00 | 1.00 / 1.00 |
| Survival | 1.00 / 1.00 | 1.00 / 1.00 |
| 338 election | 1.00 / 1.00 | 0.971 / 1.00 |

- **Nine of ten concepts meet the detection bar on M12, and all ten pooled.**
- **The reverse-fee false positive is the same rule as M11's** (7989f883). In this merger of equals both companies' shares convert into a new parent's stock, so v2.1 rule 11 makes both of them target-side. The extractor coded the mutual fee as a target fee plus a reciprocal reverse fee, and that one agreement also costs the target-fee payor and the expense-reimbursement payor a miss each.
- The outside-date miss (48be3ebb) is a closing deadline of five business days after signing, defined in the definitions article, which the extractor did not treat as an outside date.

**Fields** (exact match for scalar fields, set F1 for set fields; bar 0.97):
- **M12 alone, 25 of 36 meet the bar** (24 before the post-checks). Of the 11 that fail, eight miss once, and those are mostly fields with 12 to 23 cases, where one miss is below 0.97. The others:
  - **baskets, set F1 0.864 (18/21):** a second "tipping" threshold kept only in notes (ee21c8a7), a hybrid basket coded as a deductible (d8160e98), and one schema-ambiguous item (below);
  - **the initial outside date and the number of extensions, 42/44 each:** the initial date misses are the two retrieval gaps above. The extension misses are the End Time gap (63f75a60, which also costs extension conditions their one miss) and a specific-performance tolling clause the extractor read as an extension (18152259).
- **The two restated MAE rules hold blind.** The number of MAE definitions is 48/50, against 40/50 for M11 under v2.14, and both misses are retrieval (daef34e9, d39418dd). No miss comes from applying either rule.
- **What else held blind:** target-fee triggers (set F1 0.983), cap amounts (0.976), committed financing types (0.970) and the expense-reimbursement cap (12/12).
- **Pooled over 292 US agreements, 29 of 36 meet the bar**, against 30 of 36 over 242. Seven fail:
  - committed financing types, set F1 about 0.968;
  - baskets, set F1 about 0.961 (new);
  - escrow sole recourse, 112/118;
  - the number of MAE definitions, 275/289;
  - the expense-reimbursement payor, 55/57 (new);
  - the reverse-fee remedy, 85/89;
  - the target-fee remedy, 134/139.
  Cap amounts now meet the bar (set F1 about 0.970).
- **Shown at 0.97 with 95% confidence: nine fields**, up from eight:
  - general-rep survival and "no survival", 290/290 each (lower bound 0.990);
  - the outside-date effect, 276/276 (0.989);
  - extension conditions, 275/276 (0.983);
  - the marketing period, 215/216, and fundamental-rep survival, 288/290 (0.978 each);
  - the basket de minimis, 108/108, and the MAE prospects prong, 286/289 (0.973 each);
  - the initial outside date, 273/276 (0.972).
  The target-fee payor drops out (138/139 after 7989f883). The MAE ability prong and disproportionate exception (285/289 each) have a lower bound of 0.969.

**Schema-ambiguous items:**
- **A cap or basket set as a percentage of the R&W policy retention** (13bb23ab, two items). The two arms agree on the amount and differ only on the basis, `other` with a percentage or `formula`; no basis code covers a retention.
- **A diligence-access covenant that names the buyer's financing sources** (ee21c8a7). It is unclear whether this counts as financing cooperation.

**Gold quality.** Gold was wrong in 6 of 39 verdicts. Three fixes add something gold missed: a 338 prohibition, a second automatic extension for a delayed stockholder meeting, and a target-fee trigger for a breach of the recommendation covenant (v2.11 rule 8). The other three recode a formula cap's amount to `not_stated`, a breach bar's causation to `cause_or_result`, and an MAE carve-out for a joint transition committee to `buyer_action_or_request`.

**Cost.** About 77K Sonnet tokens per agreement for the ten concepts, plus about 29K for the Haiku picker. Opus spent about 74K per agreement on gold and 0.79M in all on adjudication.

**What this does and does not show.**
- **The v2.15 restatement worked on its first blind test:** the MAE count went from 40/50 to 48/50, and the two misses left are retrieval. The pooled MAE count still fails (0.952), because M11's misses stay in the pool.
- **The 7.28 stage 1 change did not add much text.** The union share rose from 43% to 47% of the text, partly because M12's agreements differ. It still missed four MAE spans of kinds the 7.28 change does not cover.
- **The pool's 29 of 36 is within one field of 7.27's 30.** Two fields newly fail, each on a few misses: the expense-reimbursement payor (one agreement) and baskets (three misses). Cap amounts clear.
- **Small fields swing with one agreement.** One merger of equals, 7989f883, costs M12 one detection bar and three field bars.

**Next steps.**
1. **Merger-of-equals party sides.** v2.1 rule 11 has now caused the only reverse-fee detection error in both M11 and M12. A post-check could recode fee sides when both companies' shares convert into a new parent's stock.
2. **MAE retrieval, the remaining kinds.** Keep a defined term that an MAE definition or its carve-outs name ("Excluded Matter"), and a target-side term built on "material" and "adverse" without "Material Adverse" ("Material Impact").
3. **Rule on the retention basis** for caps and baskets, and on financing-source diligence access.
4. **Keep pooling.** With M11 and M12 kept in full, new post-checks can be tested against 100 agreements for wrong fires.

### 7.30 Merger-of-equals party sides and the remaining MAE retrieval, M11 and M12 re-scored (2026-10-29)

This is steps 1 and 2 from 7.29. It is not blind: both changes were written after seeing M11's and M12's misses, and those are the only holdouts on disk to test them against.

**Post-check** (`postchecks.check_double_conversion`, v2.1 rule 11). It reads the whole agreement for two or more parties whose shares each "shall be converted into" shares of one issuer, and for a sentence saying the issuer "is a ... Subsidiary of" or is "owned by" one of them. It only recodes:
- payors and payees coded `both` that name a converting party become `target_side`, or `buyer_side` for the party that owns the issuer, and the issuer itself is `buyer_side`;
- with no owner, a reverse fee whose payors are all target-side, beside a target fee, is dropped as a fee between two targets (7989f883);
- with an owner, a target fee the owner pays becomes a reverse fee of kind `general` (v2.1 rule 12), with an adverse-recommendation trigger renamed to its buyer-side code (76955f6e).

Over the 100 agreements on disk it finds three such structures: the two above, and 248e37e8 (Parent owned by Allscripts), which has no fee and is left unchanged. It fires on nothing else.

**Stage 1** (`prepare_v7.mae_definition_leaves`):
- an MAE defined inline in an interpretation or definitions section is kept ('any reference to a "Material Adverse Effect" with respect to any entity ... means', 775de59e). Interpretation sections had been skipped as definitions, and the definition patterns do not read this form.
- "Material Impact" counts as an MAE term (daef34e9).
- the definitions of terms an MAE definition names are kept with it, so a carve-out list defined as an "Excluded Matter" now comes along (41b903c1).

All three texts are now in the candidates. The definitions it adds are short: the median candidate share moves from 43.0% to 43.3% on M11 and from 46.7% to 46.8% on M12.

**Tests:** four for the post-check (a jointly owned parent, a parent one party owns, an ordinary merger and sides already coded) and four for the new retrieval. There are 121 tests in all.

**Re-run.**
- Stage 1 changed the candidates of 38 M11 and 30 M12 agreements, mostly by adding a definition an MAE definition names. Sonnet re-extracted the MAE definition for those 68 from the new candidates (`briefs/extract_mae_730.md`, into `out/sonnet_mae730`), the instances were spliced into `out/sonnet` (the 7.29 state is kept as `sonnet_pre_730` and `sonnet_pc_pre_730`), and the post-checks were re-run.
- **Schema caveat for M11.** The M11 brief names schema v2.14, but several M11 re-extractors, and the M11 adjudicator, also applied v2.15, which restates the MAE counting rules. M11's MAE gain below is partly the v2.15 restatement, not retrieval alone.
- 17 new disagreements were adjudicated by Opus. On M12, gold was right on 9 and one is schema-ambiguous (the joint transition committee of 217ec25c again). On M11, gold was right on 5 and the extractor on 2: a buyer-side entry under v2.15 rule 1 (2b464afb) and a `labor` carve-out for officer departures (e2f8f4cb). Gold was fixed for both.
- The "before" numbers below are the 7.29 state scored against the current gold, so both columns use the same gold.

**Results.**

| | Before | After |
|---|---|---|
| M11, detection bars met | 10 of 10 | 10 of 10 (reverse-fee recall 0.95 to 1.00, target-fee precision 0.963 to 1.00) |
| M11, fields meeting 0.97 | 28 of 36 | 27 of 36 |
| M11, number of MAE definitions | 42/50 | 46/50 |
| M12, detection bars met | 9 of 10 | 10 of 10 |
| M12, fields meeting 0.97 | 25 of 36 | 27 of 36 |
| M12, number of MAE definitions | 48/50 | 45/50 |
| M12, MAE carve-outs (set F1) | 0.973 | 0.986 |
| Pool (292 US), detection bars met | 10 of 10 | 10 of 10 (target and reverse fee both 1.00 / 1.00) |
| Pool, fields meeting 0.97 | 29 of 36 | 30 of 36 |
| Pool, number of MAE definitions | 274/289 | 275/289 |
| Pool, fields shown at 0.97 with 95% confidence | 9 | 11 |

- **The post-check removes every party-side error it was aimed at.** The reverse-fee and target-fee detection errors on 7989f883 and 76955f6e are gone, and so are the payor misses they caused. Pooled, the expense-reimbursement payor (56/57) meets the bar again, and the target-fee payor (139/139) and the MAE disproportionate exception (286/289) are newly shown at 0.97. The price is one miss: the reverse fee moved on 76955f6e has no amount, because the line stating $14,500,000 is in no candidates file (`termination.fee.reverse.amounts`, 19/20 on M11).
- **The MAE retrieval delivers the text, but the count does not follow.** On M11 the count rises from 42 to 46: five fixes, one new miss (09ae7c9e, which drops an undefined capitalized buyer term). On M12 it falls from 48 to 45. All three new M12 misses are extractor errors on text the extractor was shown: a third entry for a generic definition (775de59e, against v2.15 rule 1), and two buyer-side terms left out (14a59d95, c3c50c51). daef34e9 still misses: "Material Impact" is now in the candidates, but the extractor did not treat it as an MAE term. The pool gains one.
- **MAE carve-outs improve on M12** (set F1 0.973 to 0.986), mostly from the "Excluded Matter" list and the Section 9.2 definition.

**What this does and does not show.**
- **Run-to-run variation is as large as the retrieval effect on the MAE count.** 68 agreements were re-extracted for one concept, and the count moved by +4 on one holdout and −3 on the other, on text that was largely unchanged. 7.18 measured 2.5% of field values changing between runs; the MAE count is one of the fields most exposed to it, because one extra or missing entry is a miss.
- **The post-check is narrow.** It fires only on a double conversion into one issuer, which the 100 agreements show three times. It does not cover a merger of equals in which one company merges into the other.
- **Both changes are fitted to M11 and M12.** The next blind holdout measures them.

**Next steps.**
1. **"Material Impact" and similar terms in the brief.** Name in the extractor brief that a defined term meaning a material adverse effect counts whatever its words.
2. **A blind M13** under v2.15, this stage 1 and the new post-check, keeping its per-agreement data.
3. **Settle the joint-committee carve-out** (217ec25c), now schema-ambiguous twice.

### 7.31 M13 holdout: blind, schema v2.15 and the 7.30 stage 1 and post-check, pooled with M7 to M12 (2026-10-30)

This is step 2 from 7.30: a blind M13 under v2.15, the 7.30 stage 1 and the new merger-of-equals post-check. Step 1, naming "Material Impact"-style terms in the extractor brief, was not done first, so the brief is the same as M12's.

**Setup.**
- **Sample:** 50 fresh agreements, drawn with the seed `'m13holdout'` and the same strata as M3 to M12, excluding the 923 uuids in `pilot/holdout_uuids_used.txt`. The M13 uuids are now added to it (973 in all).
- **US only:** two were replaced by the next agreement in their stratum: d01eb166 (England and Wales) and 874b53ad (England). All 50 count toward the headline.
- **Stage 1:** `prepare_v7` with the 7.28 `mae_sections` and the 7.30 `mae_definition_leaves`, unioned with the Haiku picker. **Extraction, gold and adjudication:** as in 7.29, under v2.15 with every post-check in `postchecks.py`, including the 7.30 `check_double_conversion`. Nothing was changed after the draw.
- **Data kept** in `.claude/span-taxonomy-runs/m13holdout/`. Pooled set F1 is approximate, because M7 to M12 enter from the saved totals in `m12_s730_results.json`.

**How the run went.** The run was split across two accounts. The first stopped at its usage limit after gold and 39 of the 50 extractions; several extraction batches had failed on rate limits. The other 11 were re-extracted in four new batches with the same brief. Before scoring, every extraction and gold file was checked: ten concepts, the right agreement id, and every quote verbatim on one line of its input. All passed.
- A dead extractor had left a half-written helper script for b8faf5bc in `out/sonnet`. It was deleted unused, and b8faf5bc was extracted from scratch.
- **Blindness:** one first-account extraction batch read part of another agreement's Sonnet extraction (5ed13dbd), and one rerun batch looked at the first five lines of another Sonnet file (00247e5c). Neither saw gold, so the comparison with gold stays blind.

**Stage 1.**
- At the section level the union reaches 1,074 of 1,081 gold spans (0.994), from a median of 45% of the text. The v7 rules alone reach 1,065 (0.985) from 40%; the picker alone reaches 1,009 (0.933) from 23%.
- The section count overstates what the extractor saw. Adjudication found seven retrieval gaps, most of them definition lines trimmed out of a definitions section that was kept: two fee amounts in a0e138bb, the raised deposit in 9dae91bb, a per-defect de minimis in 899f2e0f, and MAE definitions or prongs in 967bef44, a621ea6f and 2c3e8ffb.

**Post-checks.** Four fired, and two were wrong:
- **v2.2 rule 12 on 0a91852c** added a recommendation-failure trigger to termination rights that never use the defined "Adverse Recommendation Change".
- **v2.5 rule 3 on de5d8138** added `cash_on_hand` from a sufficiency representation that counts the target's cash, not the buyer's.
- The other two agree with gold: v2.5 rule 3 on af7f15bd and v2.13 rule 3 on f5f1289e.
- The merger-of-equals check did not fire. M13 has no double conversion into one issuer.

**Adjudication.** 38 disputed items, split into 40 verdicts:

| Verdict | Count |
|---|---|
| Gold right | 29 |
| Extractor right (gold fixed) | 7 |
| Equivalent | 2 |
| Schema ambiguous | 2 |

Of the 29 extractor errors:
- 12 are rules it had but missed;
- 7 are text stage 1 did not reach;
- 5 are wrong codes, two of them the post-check fires above;
- 5 are text it was shown but missed.

**Detection** (precision / recall; bar 0.95 and at most 0.1 false positives per agreement):

| Concept | M13, 50 US | Pooled, 342 US |
|---|---|---|
| Target fee | 1.00 / 1.00 | 1.00 / 1.00 |
| Reverse fee | 1.00 / 1.00 | 1.00 / 1.00 |
| Expense reimbursement | 1.00 / 1.00 | 0.985 / 1.00 |
| Outside date | 1.00 / 1.00 | 1.00 / 0.997 |
| MAE definition | 1.00 / 1.00 | 1.00 / 0.994 |
| Financing | 0.974 / 1.00 | 0.988 / 0.992 |
| Indemnification cap | 1.00 / 1.00 | 1.00 / 1.00 |
| Basket | 1.00 / 1.00 | 1.00 / 1.00 |
| Survival | 1.00 / 1.00 | 1.00 / 1.00 |
| 338 election | 1.00 / 1.00 | 0.975 / 1.00 |

- **All ten concepts meet the detection bar on M13 and pooled.** The one detection error is financing found present in an all-stock reverse merger (d11c65b6), where the PIPE funds the combined company and v2.2 rule 9 says that is not acquisition financing.

**Fields** (exact match for scalar fields, set F1 for set fields; bar 0.97):
- **M13 alone, 24 of 36 meet the bar** (23 before the post-checks). Twelve fail. Most are fields with 16 to 25 cases, where one miss is below 0.97: the fee amounts, the reverse-fee remedy and triggers, the basket de minimis, escrow sole recourse.
- **The MAE definition is the weak concept.** Five of its fields fail on M13: carve-outs (set F1 0.946), prospects (43/49), ability to consummate (44/49), the disproportionate exception (45/49), and the number of definitions (46/49). The misses are concentrated, not spread: nine agreements account for all of them.
  - **A definition missed as a whole** drives most of the scalar misses, since one miss costs four fields. Three are retrieval (967bef44, a621ea6f, 2c3e8ffb). Two are definitions inside a representation section that was in the candidates, which the extractor took to be undefined (aed38d35 in §3.1(a), e4c33baa in §4.1). 10eaf6ad's definition is cut off by a missing page.
  - **Carve-out codes** miss on single clauses: commodity prices (`other`, 00247e5c), "national calamity" (`natural_disaster` under v2.3 rule 11, c1031a42), and a government shutdown (`other`, 383339d9).
  - **The count** misses on 7cb4111d, where "Company" and "Parent Material Adverse Effect" are used "as defined in Section 9.03" but only a generic definition exists, and on a621ea6f (below).
- **Committed financing types (set F1 0.960)** miss on limited guarantees and parent funding covenants coded `other` under v2.8 rule 1 and v2.9 rule 9 (623d58f2, 49999287, 3590e99a, 383339d9), plus the wrong post-check fire on de5d8138.
- **What held blind:** every outside-date field (extension conditions and the number of extensions at 47/48 each), the cap amounts (set F1 0.986), both fee payor sides, the survival fields, and target-fee triggers (set F1 0.987).
- **Pooled over 342 US agreements, 31 of 36 meet the bar**, against 30 of 36 over 292. The target-fee remedy now clears (159/164). Five fail:
  - committed financing types, set F1 about 0.967;
  - baskets, set F1 about 0.967;
  - escrow sole recourse, 131/138;
  - the number of MAE definitions, 321/338;
  - the reverse-fee remedy, 101/106.
- **Shown at 0.97 with 95% confidence: nine fields**, down from eleven:
  - general-rep survival and "no survival", 340/340 each (lower bound 0.991);
  - the outside-date effect, 324/324 (0.991);
  - fundamental-rep survival, 338/340 (0.982), and the target-fee payor, 164/164 (0.982);
  - the marketing period, 252/253 (0.981);
  - extension conditions, 322/324 (0.981);
  - the initial outside date, 321/324 (0.976);
  - the reverse-fee payor, 106/106 (0.972, new).
  Three drop out: the basket de minimis (126/127, 0.963), the MAE disproportionate exception (331/338, 0.961) and the MAE prospects prong (329/338, 0.954), all on M13's MAE and retrieval misses.

**Schema-ambiguous items:**
- **One MAE term defined in two sentences, one per party, with different prongs** (a621ea6f). v2.15 rule 2 counts defined terms, but here one term carries two definitions. Gold keeps two entries.
- **An extension keyed to an annex condition that is not in the text** (a0e138bb). The schema does not say whether to infer the kind of a cited condition the text does not contain.

**Gold quality.** Gold was wrong in 7 of 40 verdicts. Three fixes add or recode under written rules: a specific-performance extension (v2.13 rule 2, c1031a42), a regulatory-led extension test (v2.13 rule 6, 00247e5c), and a 338(g) clause with no cost shift coded `none` (v2.2 rule 5, ee9cf4fa). Two remove carve-out codes (`other` for an example of consummation effects, and for "political or social conditions"). One removes a breach bar that is not a breach test, and one sets a financing condition to false where the text says so expressly.

**Cost.** About 76K Sonnet tokens per agreement for the ten concepts (measured on the 11 re-extracted agreements), plus about 31K for the Haiku picker. Opus spent at least 73K per agreement on gold and 0.81M in all on adjudication. The logs are incomplete: one gold batch and most first-account extraction batches were not recorded.

**What this does and does not show.**
- **Detection holds blind:** all ten concepts meet the bar on M13, and pooled over 342 agreements, for the second holdout running.
- **The pool gained a field (31 of 36) but lost two confidence bounds.** M13 is the MAE definition's worst blind run since M11. Its misses are whole definitions missed, by retrieval or by the extractor, not wrong rulings. The 7.30 MAE retrieval did not reach the three retrieval cases here.
- **Two post-checks fired wrongly for the first time in many runs.** v2.2 rule 12 and v2.5 rule 3 each need a narrower trigger.
- **The merger-of-equals check is untested blind:** M13 has no case for it.

**Next steps.**
1. **MAE definitions inside representation sections.** Tell the extractor that a defined MAE may sit inside a representation, as in aed38d35 and e4c33baa, and keep definition lines that a kept definitions section trims (967bef44, a621ea6f).
2. **Narrow the two post-checks that fired wrongly.** v2.2 rule 12 should require the defined recommendation-change term in the termination right; v2.5 rule 3 should require that the cash counted is the buyer's.
3. **Rule on one MAE term defined per party, and on cited conditions outside the text.**
4. **Committed financing `other`:** limited guarantees and parent funding covenants are missed again. A post-check could add `other` when a limited guarantee covers payment obligations beyond the reverse fee.

### 7.32 Fixes from M13: two post-checks narrowed, stage 1, schema v2.16, M11 to M13 re-scored (2026-10-30)

This is steps 1 to 3 from 7.31, as one fix round before the corpus pass (M3). It is not blind: every change was written after seeing M13's misses. The test for wrong fires is the 150 agreements of M11 to M13, the holdouts whose per-agreement data is kept.

**Post-checks** (`pilot/postchecks.py`):
- **v2.2 rule 12** adds `recommendation_failure` only when the adverse-recommendation trigger uses the defined term whose definition includes the failure. A termination right that lists the actions itself ("shall have withdrawn or modified ... its recommendation") does not take in the definition's failure limb (M13 0a91852c).
- **v2.5 rule 3** leaves alone a sufficiency representation that counts the target's cash ("together with available cash of the Company", M13 de5d8138).
- **Audit on M11 to M13:** each change removes its one wrong fire and keeps every correct one (four rule 12 fires on M12, one rule 3 fire on M13).
- **Tests:** the recommendation-failure tests now quote the defined term in the trigger, as real termination rights do, plus one new test for each change.

**Stage 1** (`pilot/prepare_v7.py`), four additions:
1. Fee amounts defined under other names are resolved: "Company Termination Compensation", "Parent Termination Payment", "Deposit" (a0e138bb, 9dae91bb).
2. An abbreviated MAE term counts as an MAE term ("Snacks Business MAE", 967bef44).
3. A definition in the form 'will be deemed to have a "Material Adverse Effect" on' counts as an inline definition (a621ea6f).
4. When the agreement uses no capitalized MAE term at all (a heading such as "No Material Adverse Change." aside), the paragraphs of any party's representations that use the undefined lowercase one are kept (`undefined_mae_leaves`, 2c3e8ffb). Only those paragraphs are kept, not their sections: keeping whole sections took one agreement from 43% to 74% of its text for two gold quotes.

On M11 to M13, gold quotes found verbatim in the candidates rise from 5,988 to 6,026 of 6,074, and none are lost. Twenty agreements change (3, 7 and 10), and the median share of the text rises from 44.7% to 44.8%. A fifth change was tried and reverted: letting an MAE definition run on across a page break into the next section. It changed eight more agreements and gained no gold quote; the 10eaf6ad definition it was aimed at starts on a page missing from the text.

**Schema v2.16** (`pilot/schema_v2_16.md`), four rulings:
1. **Where an MAE definition sits.** It may be inside a representation, an interpretation section or an exhibit. Read every candidate section before treating a capitalized MAE term as undefined (aed38d35, e4c33baa).
2. **An MAE term in any words:** "Material Impact", "Snacks Business MAE", or a "deemed to have" sentence.
3. **One MAE term defined separately for each party,** in two sentences with their own tests, is one entry per party (a621ea6f).
4. **An extension keyed to a condition the text does not contain** is `other`, without guessing what the condition is (a0e138bb).

**Re-extraction.** The MAE definition was re-extracted (Sonnet, `briefs/extract_732.md`, v2.16) for all 50 M13 agreements and for the 10 M11 and M12 agreements whose candidates changed. For a0e138bb and 9dae91bb the target and reverse fees were re-extracted too. Every output was checked: the right concepts, the agreement id, and every quote verbatim on one candidate line. The post-checks were then rerun on all 150 agreements. The 7.31 state is kept as `sonnet_pre_732`, `sonnet_pc_pre_732` and `candidates_pre_732`.

**Adjudication.** The re-extraction left 11 disputes that gold had not ruled on (10 on M13, 1 on M12, none on M11). There are 12 verdicts: gold was right on 10, and the extractor on 2. Two gold records changed:
- bf19d9e0: one MAE term defined in two sentences now gets one entry per party (v2.16 rule 3);
- 967bef44: a carve-out for announcement effects "including any employee departures" adds `labor`, as M11 and M12 gold codes it.

**Results**, before and after against the same final gold:

| | Before | After |
|---|---|---|
| M11, fields meeting 0.97 | 27 of 36 | 27 of 36 |
| M12, fields meeting 0.97 | 27 of 36 | 27 of 36 |
| M13, fields meeting 0.97 | 24 of 36 | 29 of 36 |
| M13, MAE prospects / ability / disproportionate exception | 43 / 44 / 45 of 49 | 48 / 49 / 48 of 49 |
| M13, MAE carve-outs (set F1) | 0.944 | 0.978 |
| M13, fee amounts (target / reverse, set F1) | 0.966 / 0.889 | 1.000 / 1.000 |
| Pool (342 US), detection bars met | 10 of 10 | 10 of 10 |
| Pool, fields meeting 0.97 | 31 of 36 | 31 of 36 |
| Pool, fields shown at 0.97 with 95% confidence | 9 | 12 |

- **The three new fields shown at 0.97 are all MAE:** the prospects prong, the ability-to-consummate prong and the disproportionate exception, 334/338 each (lower bound 0.973).
- **Two M13 post-check errors are gone:** committed financing types rise to set F1 0.968 and target-fee triggers to 0.992.
- **One field gets worse:** the reverse-fee kind on a0e138bb. Its fee is now visible, and the extractor coded it `other` where v2.1 rule 12 makes it `reciprocal` (15/16 on M13).
- **The MAE count is still the weakest pooled field** (321/338). M13's three misses left: 10eaf6ad, whose definition starts on a missing page; and 7cb4111d and d70d2259, where the extractor coded party terms built on one generic definition ("Company Material Adverse Effect (as defined in Section 9.03)") as a single `both` entry. v2.15 rule 2 gives each party term its own entry.
- **Still failing pooled:** committed financing types (set F1 0.968), baskets (0.967), escrow sole recourse (131/138), the MAE count (321/338) and the reverse-fee remedy (101/106).

**Cost.** About 1.42M Sonnet tokens for the 60 re-extractions (24K per agreement for the MAE alone) and 0.26M Opus tokens for adjudication.

**What this does and does not show.**
- **The stage 1 and brief changes fix what M13 missed, on M13.** That is a fit, not a measurement. On M11 and M12 they change no field.
- **Only M13 was re-extracted in full,** so part of its gain may be run-to-run variation (7.30 measured shifts of up to four in the MAE count on largely unchanged text). The scalar MAE fields moved by three to five of 49, in the direction the retrieval fix predicts, on the agreements it targeted.
- **The two narrowed post-checks** were checked against every fire on the 150 kept agreements. They could still fire wrongly on wordings these do not contain.

**Next steps.**
1. **M3, the corpus pass.** The eligible corpus is 11,780 agreements with a verified latest XML (about 5% non-US by the holdout draws). At the pilot's rates, a pass for all ten concepts is roughly 0.9B Sonnet tokens plus 0.37B Haiku tokens for the picker. A production prompt that caches the schema once should cost less per agreement, but that has not been measured. Fields that still fail pooled should be held back or flagged in serving.
2. **Party terms built on one generic MAE definition** (7cb4111d, d70d2259): restate v2.15 rule 2 in the brief, as v2.15 did for the counting rules.

## 8. Extraction pipeline

### 8.1 Stages

```
verified XML ─▶ [0] text layer + structure index (shared with smart reader:
                    subsections, defined terms, xrefs)
             ─▶ [1] candidate detection   (no LLM; high recall)
             ─▶ [2] span classification   (small model; ≤20 candidate concepts per window)
             ─▶ [3] value extraction      (strong model; per instance; JSON schema)
             ─▶ [4] validation & review   (deterministic checks, review queue)
             ─▶ serving tables, re-anchored on each new XML version
```

**Stage 1: candidate detection.** Goal: recall of at least 0.98 per concept, measured on gold. The signals combine:
- heading and structure cues (article and section titles; definitions articles);
- lexical cue sets per concept, authored in the YAML (for example `termination fee`, `break-up fee`, a `$` amount within 200 characters of `terminat`);
- paragraph-level embedding retrieval against each concept's enriched text (voyage-4-large is already in use for taxonomy nodes);
- the defined-term graph from the smart reader: if a term like "Company Termination Fee" is defined, its definition and every paragraph that uses it become candidates.

The output is a set of `(paragraph range, candidate concepts)` windows. Rules decide only what to look at, never what a span means.

**Stage 2: span classification.**
- **Input:** one window (a paragraph plus its section heading and one paragraph of context each side) and a short list of at most 20 candidate concepts, each with its description and exclusion cues.
- **Output:** concept IDs, each with role-tagged quotes. The model returns exact quoted text, not offsets; code finds each quote in the layer. A quote that cannot be found fails the span. Models count characters badly, and quote-matching gives a free hallucination check.
- **Model:** a small, cheap one (Claude Haiku 5.5 or gpt-5.4-mini), with a stronger fallback for windows where the small model abstains or its confidence is low.

**Stage 3: value extraction.**
- **Input:** each instance's spans, the resolved definitions of every defined term used in them, the resolved targets of every cross-reference (so "terminated pursuant to Section 8.1(e)" arrives with 8.1(e)'s text), and deal metadata.
- **Output:** strict structured output against the concept's JSON schema, with an evidence quote for every non-null field.
- **Model:** a stronger one (Claude Sonnet 5.5 as the default, Opus 5.5 for gold pre-labeling and disputes).

**Stage 4: validation.** All checks are deterministic:
- every evidence quote is found in the layer and lies inside or adjacent to the instance's spans;
- Money parses, and `amount` matches the number in `raw`;
- percentages fall within plausible bands (a target fee of 0.5–10% of equity value; a basket of 0.1–5% of purchase price);
- the outside date falls after signing;
- trigger cross-references resolve to a termination-right instance;
- the payor differs from the payee.

Failures and low-confidence instances go to a review queue. Validator thresholds are tuned to send under 10% of instances to review.

### 8.2 LLM vs. trained model

| Stage | v1 | Later |
|---|---|---|
| 1 Candidates | Rules plus embeddings | Same; add a distilled paragraph classifier once around 5,000 accepted instances exist |
| 2 Spans | Small LLM | Distill into a token or span tagger (e.g. a DeBERTa- or ModernBERT-class encoder fine-tuned on accepted spans). Near-zero marginal cost per ingest, and the LLM stays as a fallback for low-confidence windows |
| 3 Values | Strong LLM | Stays an LLM. Values need reading defined terms and cross-references together, which is where encoders are weak |

The ML taxonomy classifier was abandoned for section labels. That does not argue against distillation here, for two reasons: the label set per window is small, and the training data will be human-accepted spans, not GPT references.

### 8.3 Cost per corpus pass

Basis: 9,873 agreements, about 600 million tokens of body text. Prices are Claude list prices as of 2026-10 per million tokens: Haiku 5.5 $0.10 in / $0.50 out; Sonnet 5.5 $2 / $10; Opus 5.5 $4 / $20. Batch is 50% off. gpt-5.4-mini prices were not checked; the current pipeline uses it, so price it before choosing.

| Approach | Input tokens | Rough cost (batch) |
|---|---|---|
| Naive: whole agreement to Sonnet 5.5, once per concept family (×4) | 2.4B | ≈ $2,400 + output |
| Naive: whole agreement to Haiku 5.5 (×4) | 2.4B | ≈ $120 + output, but recall on long documents is unproven |
| **Recommended: stage 2** on candidate windows (≈15% of text for 10 concepts, plus prompt overhead) | ≈ 120M | Haiku ≈ $6–15; Sonnet ≈ $120–200 |
| **Recommended: stage 3** at about 10 instances per agreement × about 3,000 tokens (spans, definitions, cross-references), plus ≈ 4,000 output tokens per agreement including reasoning | ≈ 300M in, 40–80M out | Sonnet ≈ $300 + $200–400 |
| **Recommended total**, 10 concepts, Sonnet for values | | **≈ $450–900** per full pass |
| Gold pre-labeling: 150 agreements, whole text, Opus 5.5 | ≈ 9M in | ≈ $20–60 |
| Incremental ingest (≈ 100 new agreements per month) | | under $10 per month |

Prompt caching of the shared system prompt and schemas trims stage 2 and 3 further. Track spend per run (`extraction_run.cost_usd`), something today's batch tables do not do.

## 9. Evaluation plan

### 9.1 Gold set

- **Pilot gold:** 50 agreements, chosen for spread on target type (public or private), deal type, era (2000–2026) and size, with known hard cases:
  - two-tier and go-shop fees;
  - regulatory reverse fees;
  - asset deals with no fee;
  - private deals with full indemnity packages;
  - 338(h)(10) deals with a gross-up.
- **v1 gold:** 150 agreements, split into 50 dev and 100 test. The test split stays sealed and is only scored.
- **Process:** Opus 5.5 pre-labels the whole agreement for the pilot concepts. You review and correct each instance in a simple local review UI: accept, edit span, edit value, add a missed instance, reject. Per the project notes, review time is the scarce resource, so the UI shows one instance per screen, with the evidence highlighted.
- **Second pass:** a second, independent pass by a different model family flags disagreements for adjudication. On roughly 20 agreements, a second human pass measures agreement between annotators. If humans agree on a span less than about 0.85 of the time, the concept definition is the problem, not the model.
- **Negative coverage:** every gold agreement is reviewed for *all* pilot concepts, so a missing instance is a real negative.

### 9.2 Metrics

| Level | Metric | Match rule |
|---|---|---|
| Instance detection | Precision, recall and F1 per concept | Same concept, primary spans overlap with IoU ≥ 0.5 |
| Span boundaries | Token-level F1 on matched instances; exact-boundary rate | Whitespace tokens |
| Values | Accuracy per field | Enums and booleans exact; Money exact after normalization (amount and currency), pct ±0.01 pt; dates exact; lists by set F1 |
| End to end | Share of gold instances fully correct (detected, every key field right) | Key fields per concept: fee amount, payor, payee, trigger kinds |
| Negatives | False-positive instances per agreement | Agreements and concepts with no gold instance |
| Calibration | Reliability of `confidence` | Bucketed by confidence |
| Anchoring | Orphan and fuzzy rate per rebuild | All spans on rebuilt agreements |

**Acceptance bar before a corpus pass:** per concept, instance recall ≥ 0.95, precision ≥ 0.95, key-field accuracy ≥ 0.97, and at most 0.1 false-positive instances per agreement. Aggregations publish only from concepts that pass, and show sample size and coverage next to every distribution.

### 9.3 Ongoing checks

- A monthly spot-check of about 20 random new instances, in line with the monthly page-gold validation you already do.
- Drift alerts when the share of agreements with a given concept moves more than X points between ingest months.
- The orphan rate after each XML rebuild batch.

## 10. Storage

All new tables get explicit DDL in `etl/sql/migrations/`, with primary keys and foreign keys, and are created by the ETL, never seeded by hand.

| Table | Key | Purpose | Public dump? |
|---|---|---|---|
| `taxonomy_release` | `release_id` | `discipline`, `version`, `created_at`, notes | yes |
| `concept` | `concept_id` | label, discipline, kind, `nav_parent_id`, `value_schema_version`, description, terms, examples, exclusion cues, `introduced_in`, `deprecated_in`, `replaced_by` | yes |
| `concept_nav_link` | (`concept_id`, `nav_id`) | "also listed under" links | yes |
| `concept_crosswalk` | (`concept_id`, `scheme`, `external_id`) | legacy section and tax IDs, DealOS KG; `relation` | yes |
| `text_layer` | (`agreement_uuid`, `xml_version`) | `layer_hash`, `length`, `text`, structure index (JSON) | text: private serving table, like `section_text_search`; hash and length public |
| `provision_instance` | `instance_id` | `agreement_uuid`, `concept_id`, `party`, `parent_instance_id`, `status`, `confidence`, `values_json`, `schema_version`, `taxonomy_release`, `run_id`, `created_at`, `updated_at` | yes |
| `provision_span` | `span_id` | `instance_id`, `role`, `agreement_uuid`, `xml_version`, `layer_hash`, `start_char`, `end_char`, `quote`, `prefix`, `suffix`, `section_uuid`, `struct_path`, `anchor_status` | yes |
| `provision_value` | (`instance_id`, `field_path`, `ordinal`) | flattened, typed values for SQL: `value_num`, `value_text`, `value_date`, `currency`, `unit`, `basis`, `derived`, `evidence_span_ids` | yes |
| `extraction_run` | `run_id` | stage, model, prompt hash, schema and taxonomy versions, counts, tokens, `cost_usd`, timestamps | no (ETL only) |
| `annotation_review` | `review_id` | instance, reviewer, action, before/after JSON, timestamp | no |
| `section_concepts` (derived) | (`section_uuid`, `concept_id`) | section-topic view, with coverage share; replaces `latest_sections_search_standard_ids` | yes |

**`provision_value` exists so aggregation stays SQL-first.** A termination-fee distribution by year becomes one join, which matches the habit of answering data questions with one `./bin/mariadb-read` query:

```sql
SELECT YEAR(l.filing_date) yr, COUNT(*) n,
       AVG(v.value_num) avg_pct
FROM provision_instance i
JOIN provision_value v ON v.instance_id = i.instance_id
  AND v.field_path = 'pct_of_equity_value_computed'
JOIN latest_sections_search l ...      -- or an agreements-level view
WHERE i.concept_id = 'ma.termination.fee.target' AND i.status IN ('accepted','predicted')
GROUP BY yr;
```

**Redaction.** The API already redacts some sections for anonymous callers, and the reader-index plan filters anchors inside redacted sections. Provision endpoints must do the same: for anonymous callers, drop `quote`, `prefix` and `suffix` on spans inside redacted sections, and drop values whose only evidence sits there. Aggregates are unaffected.

**Serving-schema caution.** Production `pdx` is restored only from the allowlists (`bulk/public_tables.txt`, `bulk/private_restore_tables.txt`), so every serving table must be on one of them. Per the project notes, a query that passes locally can still fail with a 500 in production if a table is missing. The deploy checklist therefore includes "the table exists in prod, and a live call returns rows".

## 11. API, MCP, bulk and frontend changes

**REST** (new; old endpoints kept through a deprecation window):
- `GET /v1/concepts`: the tree for each pack, with the navigation and concept layers, crosswalks and value schemas.
- `GET /v1/concepts/{id}/counts`.
- `GET /v1/agreements/{uuid}/provisions?concept=`: instances with spans, values and evidence.
- `GET /v1/provisions?concept=…&value.<field>[gte|lte|eq]=…`: search by value, across agreements, with deal filters ("reverse fees above 6% in sponsor deals since 2020").
- `GET /v1/concepts/{id}/distribution?field=…&group_by=year|industry|target_type`: returns `n`, coverage, quantiles and histogram buckets. It always reports the denominator: agreements in scope, agreements with an instance, and agreements reviewed.
- Existing `standard_id` and `tax_standard_id` filters keep working through `concept_crosswalk` plus `section_concepts`. Hex IDs are accepted, and the response says which concept they mapped to.

**MCP:**
- `get_concept_taxonomy` replaces both taxonomy tools, with readable IDs, so `suggest_clause_families` can mostly fall back to plain search over concept descriptions.
- `search_provisions` and `get_provision_distribution` are new. `get_provision_distribution` is the aggregation tool that was deferred, now built on clean per-provision values.
- In section payloads, typed provision values appear next to `monetary_values`. The regex field is relabeled as raw strings, not usable as data. It goes away once concept coverage is broad enough.

**Bulk dump:** add the new public tables to `bulk/public_tables.txt` and document every column in `bulk/schema_docs/table_docs.yml` (the push fails otherwise), with a new group "Provisions (span taxonomy)". Add a changelog `schema` entry. Fix the wrong `clauses.start_char` docs now, independently.

**Frontend:**
- Search gets a concept filter plus value-range facets for value concepts.
- Result snippets become the span, not the section.
- The reader highlights spans and shows values on hover, using the smart reader's overlay layer. The Taxonomy page renders the concept layer, with schema fields and example spans.

## 12. Migration from section-level labels

1. **Crosswalk first.** Map each of the 99 live leaves and 140 redesign leaves to concepts, marking each mapping exact, broader or narrower. This is mostly mechanical: the redesign leaves are already provision-shaped. Map the tax L3 leaves the same way.
2. **Do not import the old labels as gold.** The titles-only labels from mixed models are a weak prior. Stage 1 can use them as one candidate signal, nothing more.
3. **Tax clauses become spans.** Convert each clause's XML-string offsets into text-layer offsets: walk the section XML and map string positions to layer positions (deterministic, testable). Each `tax_clause_assignments` row becomes a `provision_instance` with `source = legacy_tax_v1` and `status = predicted`. They are re-reviewed through the same queue, starting with the about 250 agreements already covered.
4. **Section topics come from spans.** `section_concepts` is derived: a section carries a concept if one of that concept's primary spans overlaps it. Label concepts with no span model yet get a body-text section labeling pass (the DeepSeek retrieve-first research path, which sees body text) under the new concept IDs. Titles-only labeling retires.
5. **Stop writing `standardId` into the XML.** The reader reads concepts from the API instead.
6. **Retire the old tables** (`taxonomy_l*`, `taxonomy_l*_new`, `latest_sections_search_standard_ids`, `sections.section_standard_id*`) only after one prod push cycle with both sets of tables present and the crosswalked filters checked. The changelog announces it.
7. **Prod cadence.** Prod is refreshed from local about monthly, so each migration step lands with the monthly push, not as a separate deploy.

## 13. How it feeds search, aggregation and the smart reader

- **Search:**
  - Filter by concept, which is more precise than section topics, and by value.
  - Rank and snippet on the span.
  - Compare the same concept across agreements, which generalizes `TaxClauseCompare` into "compare provisions".
- **Aggregation:**
  - Termination-fee and reverse-fee distributions (percentage of equity value) by year, sector, buyer type and counsel.
  - Outside-date length, MAE carve-out prevalence (for example, the rise of pandemic carve-outs after 2020), and basket and cap levels in private deals.
  - Each is published with n, coverage and status filters. `get_agreement_trends` gains concept-based sections.
- **Smart reader**, in both directions:
  - The reader supplies the text layer, the subsection structure, defined terms and resolved cross-references. Stage 1 uses them as candidate signals and stage 3 as context.
  - This plan supplies concept overlays (chips at paragraph level, "jump to Termination Fee", values on hover) and typed links: a fee trigger that points to 8.1(e) becomes a link from the fee instance to the termination-right instance.
  - Both plans already choose standoff over rewriting the XML. Still to settle together: one projection function with the node-text normalization, the (node, UTF-16) ↔ layer conversion fixture, and the subsection path syntax (`8.3(b)(i)`).

## 14. Milestones

| # | Scope | Exit criterion |
|---|---|---|
| **M0: Decisions** (about 1 week) | Answer the [open questions](#15-open-questions-for-you); agree the text-layer spec with the smart-reader plan; YAML skeleton for `ma` and `tax` with the first 10 value concepts; DDL drafted | Spec merged as docs; loader validates the YAML in CI |
| **M1: Pilot** (run 2026-10-21, [7.4](#74-m1-pilot-results-2026-10-21)) | Text layer; stages 1–3 for **3 concepts** (target fee, reverse fee, outside date) on **50 agreements**; Opus gold with model adjudication; metrics report | Met for detection, retrieval and core values. Coded fields diagnosed as schema gaps, and schema v2 written. Cost: about 30K Sonnet tokens per agreement for 3 concepts |
| **M2: v1 concepts** (run 2026-10-21, [7.5](#75-m2-pilot-results-2026-10-21)) | All 10 value concepts; 150-agreement gold (50 dev / 100 test); review UI hardened; stage 2 model choice settled by measurement | Detection bar met on test for 6 of 10 concepts; financing fails on retrieval. After stage 1 fixes, defined-terms exhibits and schema v2.2, a fresh 50-agreement holdout ([7.6](#76-m2-holdout-results-2026-10-21)) meets the detection bar for all 10 (financing at 0.95). Fee triggers and cap amounts failed on missed definitions and one unwritten rule. Definition resolution and schema v2.3 ([7.7](#77-definition-resolution-re-check-2026-10-21)) cut retrieval gaps from 17 to 3 and lift target-fee triggers to set F1 0.96 and MAE carve-outs to 0.97; cap amounts (0.92) still fail, now on rule application. A blind 50-agreement holdout with schema v2.4 and post-checks ([7.8](#78-m3-holdout-2026-10-21)) meets the bar for target fee, expense reimbursement and survival, is close for outside date and MAE, and fails for reverse fee, cap, basket and financing (recall 0.83, mostly retrieval). Stage 1 financing patterns and schema v2.5, on a further fresh holdout ([7.9](#79-financing-retrieval-round-2026-10-21)), bring the reverse fee to the bar and financing recall to 0.93; cap and basket amounts still fail, mostly on whether a percentage of the price may also carry its computed dollar figure. Schema v2.6 ([7.10](#710-schema-v26-and-the-holdout-re-score-2026-10-22)) keeps only stated figures; re-scored, cap amounts reach F1 0.91 (bar 0.90) and basket 0.91 (bar 0.93). A blind 50-agreement run of v2.7 ([7.11](#711-m5-holdout-2026-10-22)) meets the bar for basket (16/16) but not for financing (recall 0.86, all retrieval) or cap amounts (0.77). Structural financing retrieval and schema v2.8 ([7.12](#712-structural-financing-retrieval-and-schema-v28-2026-10-22)) recover every M5 financing span (not blind) and settle the cap errors as rules, ready for the next blind run. A Haiku section picker reading only an outline ([7.13](#713-model-chosen-stage-1-2026-10-23)) reaches 455 of 459 provisions from 24% of the text; combined with the v7 rules it misses 4 of 1,685 gold spans. A blind 50-agreement run with that combined stage 1 and schema v2.8 ([7.14](#714-m6-holdout-rules-plus-picker-schema-v28-2026-10-23)) meets the detection bar for financing (recall 1.00) and cap, and the cap field bars (amounts 0.95); committed financing types (0.93) still fall short, mostly on `cash_on_hand`. A `cash_on_hand` post-check and colon-form definitions ([7.15](#715-two-fixes-from-m6-re-scored-2026-10-23), not blind) lift them to 0.96 and cap amounts to 0.99; adjudication then confirmed every post-check fire on M3 to M5. A blind 50-agreement run on all ten concepts with that stage 1, v2.8 and the post-checks ([7.16](#716-m7-holdout-all-ten-concepts-rules-plus-picker-schema-v28-2026-10-23)) meets the detection bar for nine of ten (reverse fee 0.94, one ambiguous miss), and every bar for expense reimbursement and basket. Six more are within a field or two, and escrow sole recourse and the 338 tax cost payer fail. A meeting-covenant post-check, definition fixes and schema v2.9 ([7.17](#717-fixes-from-m7-re-scored-2026-10-23), not blind) bring detection to the bar for all ten. Target fee, expense reimbursement, survival and 338 then meet every bar; escrow sole recourse still fails, and run-to-run variation moves single fields across 0.97. A second extraction of all 50 ([7.18](#718-run-to-run-variance-on-m7-2026-10-23)) changes no detection call and 2.5% of field values; 12 fields clear 0.97 in both runs, 15 more clear on too few cases to judge, 6 clear in one run only, and 3 fail in both. Proving 0.97 per field needs about 100 cases. Schema v2.10 and four post-checks for the shared errors ([7.19](#719-schema-v210-and-post-checks-for-the-shared-errors-2026-10-23), not blind) leave no field failing in both runs; escrow sole recourse reaches 19/19 and financing efforts 35/36, and reverse-fee triggers clear in one run. A blind 50-agreement holdout under v2.10 ([7.20](#720-m8-holdout-blind-schema-v210-pooled-with-m7-2026-10-23)), pooled with M7 on 97 US agreements, meets the detection bar for eight of ten concepts (expense reimbursement and the 338 election miss on single false positives) and the field bar for 29 of 36 fields; only "no survival" is shown at 0.97 with 95% confidence. Schema v2.11 and two post-checks for cited-covenant breaches and lapsed tender offers ([7.21](#721-schema-v211-two-post-checks-and-m7-and-m8-re-scored-2026-10-24), not blind) bring the pooled field count to 31 of 36, with target-fee triggers at 0.983 and the 338 decider at 11/11; both detection failures are now extractor errors under written rules. A blind 50-agreement holdout under v2.11 ([7.22](#722-m9-holdout-blind-schema-v211-pooled-with-m7-and-m8-2026-10-24)) meets the detection bar for all ten concepts on its 45 US agreements, the first blind run to do so, and 28 of 36 fields. Pooled over 142 US agreements, 9 of 10 concepts and 31 of 36 fields meet the bar; outside-date extensions, committed financing types, escrow sole recourse and the expense-reimbursement cap fail, and three fields are shown at 0.97 with 95% confidence. Schema v2.12 and a widened funds-representation post-check ([7.23](#723-schema-v212-the-combined-sources-post-check-m7-to-m9-re-scored-2026-10-25), not blind) bring the pool to 32 of 36 fields, with committed financing types at 0.974; outside-date extensions, escrow sole recourse and the expense-reimbursement cap still fail. Keeping the sections the outside-date clause cites, schema v2.13 and four extension post-checks ([7.24](#724-outside-date-extensions-cited-sections-schema-v213-m7-to-m9-re-scored-2026-10-25), not blind) bring the pool to 34 of 36 fields, with extension conditions at 138/138 and the number of extensions at 137/138; escrow sole recourse and the expense-reimbursement cap still fail. A blind 50-agreement holdout under v2.13 ([7.25](#725-m10-holdout-blind-schema-v213-pooled-with-m7-to-m9-2026-10-29)), all US, meets the detection bar for all ten concepts and the field bar for 30 of 36 fields; extension conditions hold blind at 47/47. Pooled over 192 US agreements, all ten concepts and 33 of 36 fields meet the bar, five fields are shown at 0.97 with 95% confidence, and cap amounts, escrow sole recourse and the target-fee remedy fail. Schema v2.14 and two post-check fixes for escrow scope and split recommendation-change definitions ([7.26](#726-escrow-scope-split-definitions-schema-v214-m10-re-scored-2026-10-29), not blind) bring M10 to 31 of 36 fields and pooled escrow sole recourse to 70/74; the pool stays at 33 of 36. A blind 50-agreement holdout under v2.14 ([7.27](#727-m11-holdout-blind-schema-v214-pooled-with-m7-to-m10-2026-10-29)), all US, meets the detection bar for all ten concepts and the field bar for 26 of 36 fields; target-fee triggers (0.996), escrow sole recourse (20/21) and cap amounts (21/21) hold blind. Pooled over 242 US agreements, all ten concepts and 30 of 36 fields meet the bar and eight fields are shown at 0.97 with 95% confidence; the number of MAE definitions (224/239, mostly undefined buyer-side MAEs outside stage 1) and committed financing types newly fail. Keeping buyer representations and closing conditions that use an undefined MAE, and narrowing the v2.13 rule 4 check to quoted text ([7.28](#728-buyer-side-mae-retrieval-and-the-rule-4-check-m11-re-scored-2026-10-29), not blind), lift stage 1 to 969 of 972 spans and M11 to 28 of 36 fields; the pool stays at 30 of 36, and the number of MAE definitions (227/239) now fails on rule application, not retrieval. Schema v2.15, restating the two MAE counting rules, and a blind 50-agreement holdout under it ([7.29](#729-m12-holdout-blind-schema-v215-and-the-728-stage-1-pooled-with-m7-to-m11-2026-10-29)), all US, bring the MAE count to 48/50 with both misses in retrieval; M12 meets the detection bar for nine of ten concepts (one merger-of-equals reverse-fee false positive) and 25 of 36 fields. Pooled over 292 US agreements, all ten concepts and 29 of 36 fields meet the bar and nine fields are shown at 0.97 with 95% confidence. A post-check for two-sided holding-company mergers and retrieval of the remaining MAE definition forms ([7.30](#730-merger-of-equals-party-sides-and-the-remaining-mae-retrieval-m11-and-m12-re-scored-2026-10-29), not blind) clear every party-side error and bring the pool to 30 of 36 fields with 11 shown at 0.97; the MAE count moves by about as much from run-to-run variation as from retrieval (pool 275/289).. A blind 50-agreement M13 under v2.15 with that stage 1 and post-check ([7.31](#731-m13-holdout-blind-schema-v215-and-the-730-stage-1-and-post-check-pooled-with-m7-to-m12-2026-10-30)) meets the detection bar for all ten concepts on M13 and over 342 pooled US agreements; the pool meets 31 of 36 fields, with 9 shown at 0.97. M13's misses center on whole MAE definitions missed, and two post-checks fired wrongly. Narrowing those two post-checks, four stage 1 additions and schema v2.16 ([7.32](#732-fixes-from-m13-two-post-checks-narrowed-stage-1-schema-v216-m11-to-m13-re-scored-2026-10-30), not blind) bring M13 to 29 of 36 fields and the pool to 12 fields shown at 0.97, the three new ones all MAE fields; the pool still meets 31 of 36. Not done in this run: the review UI and a stage 2 model comparison. Cost: about 60K to 77K Sonnet tokens per agreement for 10 concepts |
| **M3: Corpus and serving** (2–3 weeks) | Full pass for the passing concepts; serving tables, REST and MCP endpoints, bulk docs and changelog; reader span highlighting behind a flag | Live calls in prod return instances; distributions show n and coverage |
| **M4: Tax and breadth** | Migrate tax clauses to spans; label concepts toward roughly 140-leaf coverage; derived `section_concepts`; deprecate old filters | Old taxonomy tables unused for one push cycle |
| **M5: Distill and maintain** | Span tagger trained on accepted spans; incremental per-ingest runs; monthly checks | Stage 2 LLM spend cut by at least 80% with no recall loss on test |

## 15. Open questions for you

**Decisions, 2026-10-21:**
- **Q3, text layer scope:** no exhibits. The layer is the body only. Values that live only in an exhibit are recorded as `not_stated` with a note. *Revised after M2:* defined-terms exhibits and annexes in back matter are admitted, since several MAE definitions and outside dates live there ([7.6](#76-m2-holdout-results-2026-10-21)). Other exhibits stay out.
- **Q5, equity value:** derive it from the agreement when price per share and share count are stated. Otherwise look up the announced equity value online, from the press release or a filing, with the source URL recorded. `tx_metadata` is a last resort.
- **Q6, model vendor:** for M1 and M2, extraction and gold labeling run as subagents inside a Claude Code session, with no separate API calls. Cost is measured in subagent tokens. The vendor choice for a full corpus pass is decided later.
- **Q7, review budget:** gold is written by an independent model labeler and adjudicated by a second model, not reviewed by hand. Model-written gold is reported separately so your monthly spot check can target it.

1. **Concept granularity for the first wave.** Is the list of 10 right? In particular: one `ma.financing.structure` concept or separate ones, and should go-shop or matching rights replace expense reimbursement in v1?
2. **Party as an attribute.** Are you comfortable collapsing the duplicate Company/Buyer representation subtrees into one concept with a `party` attribute? It is a visible change on the Taxonomy page.
3. **Scope of the text layer.** The pilot argues for including exhibits from v1: per-share consideration, tender-offer conditions and the "Marketing Period" definition were all in exhibits ([section 7.3](#73-pilot-findings-on-the-schemas)). That needs the exhibits to be in the verified XML (back matter) reliably. Is that true today, or does it need its own XML work?
4. **Unreviewed rows in public data.** Serve `predicted` instances with a status flag, or only `accepted` ones? The flag is cheaper and more useful; accepted-only is safer for aggregates.
5. **Equity-value source for percentage fields.** Derive it from the agreement (price × fully diluted shares, when stated), or use `tx_metadata.transaction_price_total`, or both, with the source recorded?
6. **Model vendor.** The pipeline runs on OpenAI batch today. Keep gpt-5.4-mini for stage 2 and pick by measurement for stage 3, or standardize on one vendor?
7. **Your review budget.** M1 needs about 50 agreements × 3 concepts reviewed (about 150–250 instances); M2 needs about 1,500. Is that realistic, or should a second reviewer or a stricter model-adjudication step carry part of it?
8. **DealOS KG.** Is it fine to store crosswalk IDs to it in the public dump? Its prose is not copied either way.
9. **Amendments.** Later amendments sometimes change fees and outside dates. Out of scope for v1, or tracked as a `supersedes` link between agreements?

## Appendix A. Pilot artifacts

- `docs/plans/span-taxonomy/anchor_pilot.py`: the text-layer projection and re-anchoring measurement. It reads an XML dump, writes nothing, and is not wired into the ETL.
- `docs/plans/span-taxonomy/pilot/`: the M1 pilot code. `text_layer.py` (projection), `prepare.py` (stage 1 retrieval and reading views), `schema_m1.md` (the extraction schema as run), `score.py` and `report.py` (scoring), `m1_results.json` (the numbers in 7.4). For M2: `prepare_v2.py` (four-group stage 1), `schema_v2.md` and `schema_v2_1.md` (the schemas as run on dev and test), `score_v2.py` and `report_v2.py` (scoring), `m2_results.json` (the numbers in 7.5). For the M2 holdout: `prepare_v3.py` (stage 1 fixes), `text_layer.project(exhibits=True)` (defined-terms exhibits and section-less body text), `schema_v2_2.md` (the schema as run), and `m2_holdout_results.json` (the numbers in 7.6, both arms). For the definition-resolution re-check: `prepare_v4.py` (definition resolution), `text_layer.project(page_blobs=True)` (page-blob exhibits), `schema_v2_3.md`, `score_v2.py --concepts` (score a subset of concepts) and `m2_defres_results.json` (the numbers in 7.7, both arms). For the M3 holdout: `schema_v2_4.md`, `postchecks.py` and `test_postchecks.py`, `report_m3.py`, and `m3_holdout_results.json` (the numbers in 7.8, both arms). For the financing-retrieval round: `prepare_v5.py` (stage 1 financing and indemnification selection), `schema_v2_5.md`, `report_m3.py D C1,C2,...` (a subset of concepts), `score_v2.PCT_FIRST` (the percentage-only comparison) and `m4_holdout_results.json` (the numbers in 7.9). For v2.6 and v2.7: `prepare_v6.py` (more financing wordings; `--rebuild` re-runs stage 1 from saved layers), `schema_v2_6.md`, `schema_v2_7.md`, the v2.6 post-check in `postchecks.py`, and `m4_holdout_v26_results.json` (the numbers in 7.10). For the M5 holdout: `m5_holdout_results.json` (the numbers in 7.11). For 7.12: `prepare_v7.py` (structural financing selection and cap-term definitions), `schema_v2_8.md`, and `stage1_v7_check.json`. For 7.13: `router.py` (outline, picked candidates, the union with v7, and scoring) and `router_check.json`. For 7.14: the fixed v2.3 rule 2 post-check and `m6_holdout_results.json`. For 7.15: the v2.5 rule 3 post-check, colon-form definitions in `prepare_v7.py` with `test_prepare_v7.py`, and `m6_rescore_results.json`. For 7.16: `m7_holdout_results.json`. For 7.17: `schema_v2_9.md`, the v2.2 rule 11 post-check, run-on definitions and the new terms in `prepare_v7.py` (tests in `test_postchecks.py` and `test_prepare_v7.py`), and `m7_rescore_results.json`. For 7.18: `compare_runs.py` (two runs against each other and gold) and `m7_variance_results.json`. For 7.19: `schema_v2_10.md`, the escrow, set-off, had-right and buyer recommendation-failure post-checks (tests in `test_postchecks.py`), and `m7_v210_results.json`. For 7.20: `pool_holdouts.py` (counts summed across holdouts, with exact lower bounds) and `m8_holdout_results.json`. For 7.21: `schema_v2_11.md`, the cited-covenant and offer-lapse post-checks (tests in `test_postchecks.py`), `score_v2.duration` with `test_score_v2.py`, and `m8_v211_results.json`. For 7.22: `m9_holdout_results.json`. For 7.23: `schema_v2_12.md`, the widened v2.5 rule 3 post-check (tests in `test_postchecks.py`), the `offset_days` change in `score_v2.py` (test in `test_score_v2.py`), and `m9_v212_results.json`. For 7.24: `prepare_v7.outside_date_citations` (tests in `test_prepare_v7.py`), `schema_v2_13.md`, the four extension post-checks (tests in `test_postchecks.py`), and `m9_v213_results.json`. For 7.25: saved-result inputs to `pool_holdouts.py` (`FILE.json#KEY`, tests in `test_pool_holdouts.py`) and `m10_holdout_results.json`. For 7.26: `schema_v2_14.md`, the narrowed escrow check and the split-line and full-text recommendation-failure check (tests in `test_postchecks.py`), and `m10_v214_results.json`. For 7.27: `m11_holdout_results.json`, with the per-agreement data kept in `.claude/span-taxonomy-runs/m11holdout/`. For 7.28: `prepare_v7.mae_sections` (tests in `test_prepare_v7.py`), the narrowed v2.13 rule 4 check (test in `test_postchecks.py`), and `m11_s728_results.json`. For 7.29: `schema_v2_15.md` and `m12_holdout_results.json`, with the per-agreement data kept in `.claude/span-taxonomy-runs/m12holdout/`. For 7.30: `postchecks.check_double_conversion` and `prepare_v7.mae_definition_leaves` (tests in `test_postchecks.py` and `test_prepare_v7.py`), and `m12_s730_results.json`. For 7.31: `m13_holdout_results.json` and the M13 uuids in `holdout_uuids_used.txt`. For 7.32: the narrowed checks in `postchecks.py`, `prepare_v7.undefined_mae_leaves` and the new terms (tests in `test_postchecks.py` and `test_prepare_v7.py`), `schema_v2_16.md`, and `m13_s732_results.json`.
- The pilots used read-only queries on the local database. No taxonomy LLM step was run over the corpus. Extraction, gold and adjudication ran as subagents in a Claude Code session, with no separate API calls (decision Q6). The per-agreement JSON (extractions, gold, verdicts) was scratch and was deleted after scoring; `prepare.py` with the sample SQL in `pilot/README.md` rebuilds the inputs.
