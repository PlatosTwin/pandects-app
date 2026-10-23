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
| **M2: v1 concepts** (run 2026-10-21, [7.5](#75-m2-pilot-results-2026-10-21)) | All 10 value concepts; 150-agreement gold (50 dev / 100 test); review UI hardened; stage 2 model choice settled by measurement | Detection bar met on test for 6 of 10 concepts; financing fails on retrieval. After stage 1 fixes, defined-terms exhibits and schema v2.2, a fresh 50-agreement holdout ([7.6](#76-m2-holdout-results-2026-10-21)) meets the detection bar for all 10 (financing at 0.95). Fee triggers and cap amounts failed on missed definitions and one unwritten rule. Definition resolution and schema v2.3 ([7.7](#77-definition-resolution-re-check-2026-10-21)) cut retrieval gaps from 17 to 3 and lift target-fee triggers to set F1 0.96 and MAE carve-outs to 0.97; cap amounts (0.92) still fail, now on rule application. A blind 50-agreement holdout with schema v2.4 and post-checks ([7.8](#78-m3-holdout-2026-10-21)) meets the bar for target fee, expense reimbursement and survival, is close for outside date and MAE, and fails for reverse fee, cap, basket and financing (recall 0.83, mostly retrieval). Stage 1 financing patterns and schema v2.5, on a further fresh holdout ([7.9](#79-financing-retrieval-round-2026-10-21)), bring the reverse fee to the bar and financing recall to 0.93; cap and basket amounts still fail, mostly on whether a percentage of the price may also carry its computed dollar figure. Schema v2.6 ([7.10](#710-schema-v26-and-the-holdout-re-score-2026-10-22)) keeps only stated figures; re-scored, cap amounts reach F1 0.91 (bar 0.90) and basket 0.91 (bar 0.93). Not done in this run: the review UI and a stage 2 model comparison. Cost: about 60K Sonnet tokens per agreement for 10 concepts |
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
- `docs/plans/span-taxonomy/pilot/`: the M1 pilot code. `text_layer.py` (projection), `prepare.py` (stage 1 retrieval and reading views), `schema_m1.md` (the extraction schema as run), `score.py` and `report.py` (scoring), `m1_results.json` (the numbers in 7.4). For M2: `prepare_v2.py` (four-group stage 1), `schema_v2.md` and `schema_v2_1.md` (the schemas as run on dev and test), `score_v2.py` and `report_v2.py` (scoring), `m2_results.json` (the numbers in 7.5). For the M2 holdout: `prepare_v3.py` (stage 1 fixes), `text_layer.project(exhibits=True)` (defined-terms exhibits and section-less body text), `schema_v2_2.md` (the schema as run), and `m2_holdout_results.json` (the numbers in 7.6, both arms). For the definition-resolution re-check: `prepare_v4.py` (definition resolution), `text_layer.project(page_blobs=True)` (page-blob exhibits), `schema_v2_3.md`, `score_v2.py --concepts` (score a subset of concepts) and `m2_defres_results.json` (the numbers in 7.7, both arms). For the M3 holdout: `schema_v2_4.md`, `postchecks.py` and `test_postchecks.py`, `report_m3.py`, and `m3_holdout_results.json` (the numbers in 7.8, both arms). For the financing-retrieval round: `prepare_v5.py` (stage 1 financing and indemnification selection), `schema_v2_5.md`, `report_m3.py D C1,C2,...` (a subset of concepts), `score_v2.PCT_FIRST` (the percentage-only comparison) and `m4_holdout_results.json` (the numbers in 7.9). For v2.6 and v2.7: `prepare_v6.py` (more financing wordings; `--rebuild` re-runs stage 1 from saved layers), `schema_v2_6.md`, `schema_v2_7.md`, the v2.6 post-check in `postchecks.py`, and `m4_holdout_v26_results.json` (the numbers in 7.10).
- The pilots used read-only queries on the local database. No taxonomy LLM step was run over the corpus. Extraction, gold and adjudication ran as subagents in a Claude Code session, with no separate API calls (decision Q6). The per-agreement JSON (extractions, gold, verdicts) was scratch and was deleted after scoring; `prepare.py` with the sample SQL in `pilot/README.md` rebuilds the inputs.
