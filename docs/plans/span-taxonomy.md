# Span taxonomy: concept labels on text spans, with extracted values

Status: design draft, 2026-10-20. Nothing here is built. A companion plan, `docs/plans/smart-reader.md` (live TOC, defined terms, cross-references), is being written in parallel, and the two share the text layer described in [section 6](#6-span-annotation-model).

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
- emit the string value of every `<text>` and `<definition>` leaf, stripped, followed by `\n`;
- skip `<page>` and `<pageUUID>`.

Other parts:
- **Offsets** are Unicode code-point indices (Python `str` indices). The API also returns UTF-16 offsets for the browser, or the frontend maps between them.
- **`layer_hash`** is the sha256 of the layer text.
- **Section and paragraph ranges** come out of the same walk: the start and end of each `<section uuid>`, and each smart-reader subsection node such as `8.3(b)(i)`. They are stored as a structure index on the layer.

Front matter, table of contents, signature pages and back matter can be added later as separate zones of the same layer. v1 covers the body only. Exhibits often matter (forms of guarantee, debt commitment letters), so this is an open question ([section 15](#15-open-questions-for-you)).

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

Every field has a three-way absence: `null` (not extracted), `not_stated` (the agreement is silent) or `not_applicable`. Aggregation must tell "no cap" apart from "we didn't find a cap".

**Derived fields** are computed in code, never by the model, and flagged `derived: true` with their inputs recorded. The main example is `pct_of_equity_value_computed = fee.amount / deal.equity_value`. Equity value is often not stated in the agreement; it can be derived from per-share price × fully diluted shares, or taken from `tx_metadata` (`transaction_price_total`), which has known errors (see the SPAC acquirer note). So the derived percentage records which source it used, and aggregation can exclude low-trust sources.

### 7.2 The concepts

**1. `ma.termination.fee.target`: target termination fee** (DealOS `cra:stub-termination-fees`)
- `payor`, `payee`: PartyRole.
- `tiers[]`: each tier has `label` (`standard`, `go_shop`, `window_shop`, `superior_proposal`, `other`), `amount` (Money), and `applies_when` (text plus an XRef).
- `triggers[]`, each with:
  - `kind`: one of `superior_proposal_termination`, `adverse_recommendation_change`, `breach_of_no_shop`, `vote_failure_with_tail`, `outside_date_with_tail`, `breach_with_tail`, `other`;
  - `termination_right` (XRef);
  - `tail`: `months`, `threshold_pct` (often 50% rather than the 15–20% in the definition), and `requires` (`definitive_agreement`, `consummation`, `either`).
- `payment_timing`: Duration plus anchor.
- `exclusive_remedy`: true or false, with the carve-outs (`fraud`, `willful_breach`).
- `expense_reimbursement_credit`: whether expenses are credited against the fee.
- `pay_once_only`: whether the fee is payable on only one occasion.
- `pct_of_equity_value_computed`: derived.

**2. `ma.termination.fee.reverse`: reverse termination fee** (DealOS `cra:stub-reverse-termination-fee`)

Everything in concept 1, plus:
- `kind`: `financing`, `regulatory`, `general`, or `two_tier`;
- `guarantee`: `limited_guarantee`, `equity_commitment`, or `none`, plus the guarantor;
- `liability_cap`: whether the fee caps buyer liability (DealOS `limits_amount_under` → `cra:124`);
- `specific_performance_interaction`: `exclusive_alternative`, `both_available`, `conditional`, or `not_stated`.

**3. `ma.termination.expense_reimbursement`: expense reimbursement**
- `payor`, `payee`;
- `cap`: Money;
- `triggers[]`, as in concept 1;
- `credited_against_fee`.

**4. `ma.termination.outside_date`: outside date** (DealOS `cra:4`, `cra:114`)
- `initial`: DateRule.
- `extensions[]`: each has `length` (Duration), `new_date` (DateRule), `condition` (`regulatory_pending`, `marketing_period`, `sec_review`, `other`), `exercise` (`automatic`, `either_party`, `buyer`, `target`), `count`.
- `breach_bar`: whether a party whose breach caused the delay may still terminate.
- `days_from_signing_computed`: derived.

**5. `ma.mae.definition`: material adverse effect definition** (DealOS `cra:82`)
- `defined_term`;
- `party` (whose MAE: `target`, `buyer`, or `both`);
- `prongs`: `business_condition`, `ability_to_consummate`, `prospects_included`;
- `standard` ("would reasonably be expected to");
- `carveouts[]`: child instances of `ma.mae.carveout`.

**`ma.mae.carveout`: MAE carve-out** (label concept with a few values)
- `category`: `general_economic`, `financial_markets`, `industry`, `change_in_law`, `change_in_gaap`, `political_war_terrorism`, `pandemic`, `natural_disaster`, `announcement_or_pendency`, `compliance_with_agreement`, `failure_to_meet_projections`, `stock_price_or_volume`, `buyer_action_or_request`, `other`;
- `disproportionate_effect_exception`: true or false.

**6. `ma.financing.structure`: financing condition and related terms**
- `financing_condition`: true or false. True is rare in public deals; its absence is the common finding.
- `committed_financing`: `debt`, `equity`, or `none`, plus `sources[]`.
- `efforts_standard`.
- `marketing_period`: Duration, with blackout dates.
- `financing_cooperation`: present or absent.
- `rtf_link`: the instance ID of the financing reverse termination fee.

**7. `ma.indemn.cap`: indemnification cap** (DealOS `pcra:35`)
- `caps[]`: each has `applies_to` (`general_reps`, `fundamental_reps`, `tax`, `covenants`, `specific_indemnity`, `all`), `amount` (Money; often `pct_of_purchase_price`), `indemnitor`, `excludes` (`fraud`, `willful_breach`).
- `escrow_sole_recourse`: true or false.
- `rwi_interaction`: text.

**8. `ma.indemn.basket`: basket / deductible** (DealOS `pcra:38`)
- `type`: `deductible`, `tipping`, or `none`;
- `amount`: Money;
- `de_minimis`: a Money threshold per claim, and whether claims below it count toward the basket;
- `applies_to[]`;
- `indemnitor`.

**9. `ma.indemn.survival`: survival periods** (DealOS survival architecture)
- `periods[]`: each has `item` (`general_reps`, `fundamental_reps`, `tax_reps`, `covenants_pre`, `covenants_post`, `other`) and `period` (a Duration, or `statute_of_limitations_plus` with an `offset_days`).
- `no_survival`: the public-deal norm, where representations do not survive closing.

**10. `tax.election.338h10`: 338(h)(10) / 336(e) election** (DealOS `pcor:9`)
- `election`: `338h10`, `336e`, `338g`, `prohibited`, or `none`;
- `decides`: `buyer`, `seller`, or `joint`;
- `seller_must_join`;
- `allocation`: who prepares it, the deadline, the dispute mechanism, and the Section 1060 reference;
- `gross_up`: true or false, with a formula reference;
- `deemed_sale_tax_borne_by`.

**Exclusions.** Schemas cover what is stated in the agreement. Deal metadata (equity value, signing date) is joined in, never extracted again. Amendments that change a fee are out of scope for v1; flag them as an open question.

### 7.3 Pilot findings on the schemas

<!-- PILOT_FINDINGS -->

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
  - Shared decisions to settle with that plan: the text-layer projection (one function), the subsection path syntax (`8.3(b)(i)`), and whether paragraph IDs are written into the XML or kept standoff (this plan prefers standoff).

## 14. Milestones

| # | Scope | Exit criterion |
|---|---|---|
| **M0: Decisions** (about 1 week) | Answer the [open questions](#15-open-questions-for-you); agree the text-layer spec with the smart-reader plan; YAML skeleton for `ma` and `tax` with the first 10 value concepts; DDL drafted | Spec merged as docs; loader validates the YAML in CI |
| **M1: Pilot** (2–3 weeks) | Text layer and re-anchoring job; stages 1–4 for **3 concepts** (target fee, reverse fee, outside date) on **50 agreements**; Opus pre-labels plus your review of the 50 for gold; metrics report | Acceptance bar met on the pilot concepts, or a written diagnosis of why not; cost per agreement measured |
| **M2: v1 concepts** (3–4 weeks) | All 10 value concepts; 150-agreement gold (50 dev / 100 test); review UI hardened; stage 2 model choice settled by measurement | Bar met per concept on the sealed test split |
| **M3: Corpus and serving** (2–3 weeks) | Full pass for the passing concepts; serving tables, REST and MCP endpoints, bulk docs and changelog; reader span highlighting behind a flag | Live calls in prod return instances; distributions show n and coverage |
| **M4: Tax and breadth** | Migrate tax clauses to spans; label concepts toward roughly 140-leaf coverage; derived `section_concepts`; deprecate old filters | Old taxonomy tables unused for one push cycle |
| **M5: Distill and maintain** | Span tagger trained on accepted spans; incremental per-ingest runs; monthly checks | Stage 2 LLM spend cut by at least 80% with no recall loss on test |

## 15. Open questions for you

1. **Concept granularity for the first wave.** Is the list of 10 right? In particular: one `ma.financing.structure` concept or separate ones, and should go-shop or matching rights replace expense reimbursement in v1?
2. **Party as an attribute.** Are you comfortable collapsing the duplicate Company/Buyer representation subtrees into one concept with a `party` attribute? It is a visible change on the Taxonomy page.
3. **Scope of the text layer.** Body only for v1, or include exhibits (guarantees, commitment letters, forms of escrow agreement) from the start? Reverse-fee guarantees often live there.
4. **Unreviewed rows in public data.** Serve `predicted` instances with a status flag, or only `accepted` ones? The flag is cheaper and more useful; accepted-only is safer for aggregates.
5. **Equity-value source for percentage fields.** Derive it from the agreement (price × fully diluted shares, when stated), or use `tx_metadata.transaction_price_total`, or both, with the source recorded?
6. **Model vendor.** The pipeline runs on OpenAI batch today. Keep gpt-5.4-mini for stage 2 and pick by measurement for stage 3, or standardize on one vendor?
7. **Your review budget.** M1 needs about 50 agreements × 3 concepts reviewed (about 150–250 instances); M2 needs about 1,500. Is that realistic, or should a second reviewer or a stricter model-adjudication step carry part of it?
8. **DealOS KG.** Is it fine to store crosswalk IDs to it in the public dump? Its prose is not copied either way.
9. **Amendments.** Later amendments sometimes change fees and outside dates. Out of scope for v1, or tracked as a `supersedes` link between agreements?

## Appendix A. Pilot artifacts

- `docs/plans/span-taxonomy/anchor_pilot.py`: the text-layer projection and re-anchoring measurement. It reads an XML dump, writes nothing, and is not wired into the ETL.
- The pilot used read-only queries on the local database. No taxonomy LLM step was run over the corpus.
