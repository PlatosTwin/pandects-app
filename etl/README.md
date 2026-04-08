## ETL Overview

Current ETL execution is split across three explicit Dagster jobs:

1. `regular_ingest`
   - `01-01_regular_ingest_staging_asset`
   - `02-01_regular_ingest_pre_processing_asset`
   - `03-01_regular_ingest_tagging_asset`
   - `04-01_regular_ingest_build_xml`
   - `04-02_regular_ingest_verify_xml`
   - `05-01_regular_ingest_ai_repair_enqueue_asset`
   - `05-02_regular_ingest_ai_repair_poll_asset`
   - `05-03_regular_ingest_reconcile_tags`
   - `05-04_regular_ingest_post_repair_build_xml`
   - `05-05_regular_ingest_post_repair_verify_xml`
   - `06-01_regular_ingest_sections_from_fresh_xml`
   - `06-02_regular_ingest_sections_from_repair_xml`
   - `07-01_regular_ingest_taxonomy_llm_asset`
   - `08-03_regular_ingest_tax_module_asset`
   - `09_regular_ingest_taxonomy_gold_backfill_asset`
   - `10-01_regular_ingest_tx_metadata_offline_asset`
   - `10-02_regular_ingest_tx_metadata_web_search_asset`
2. `xml_fresh_pipeline`
   - `04-01_build_xml`
   - `04-02_verify_xml`
   - `06-01_sections_from_fresh_xml`
   - `08-01_tax_module_from_fresh_xml`
3. `xml_repair_cycle_pipeline`
   - `05-01_ai_repair_enqueue_asset`
   - `05-02_ai_repair_poll_asset`
   - `05-03_reconcile_tags`
   - `05-04_post_repair_build_xml`
   - `05-05_post_repair_verify_xml`
   - `06-02_sections_from_repair_xml`
   - `08-02_tax_module_from_repair_xml`

Core Dagster definitions live in `etl/src/etl/defs/jobs.py`. `etl/src/etl/definitions.py` now re-exports that one canonical `defs` object.

## Run With Config

Default run config lives in `etl/configs/pipeline_config.yaml`.

```bash
dagster job execute -f etl/src/etl/defs/jobs.py -j regular_ingest -c etl/configs/pipeline_config.yaml
dagster job execute -f etl/src/etl/defs/jobs.py -j xml_fresh_pipeline -c etl/configs/pipeline_config.yaml
dagster job execute -f etl/src/etl/defs/jobs.py -j xml_repair_cycle_pipeline -c etl/configs/pipeline_config.yaml
```

With `dg dev`, open the target asset or job in Launchpad and override config there.

## Stage Order

1. Stage agreements
   - `src/etl/defs/a_staging_asset.py`
   - generic/manual: `01_staging_asset`
   - run-scoped: `01-01_regular_ingest_staging_asset`
2. Pre-process pages
   - `src/etl/defs/b_pre_processing_asset.py`
   - generic/manual: `02_pre_processing_asset`
   - run-scoped: `02-01_regular_ingest_pre_processing_asset`
3. Tag body pages
   - `src/etl/defs/c_tagging_asset.py`
   - generic/manual: `03_tagging_asset`
   - run-scoped: `03-01_regular_ingest_tagging_asset`
4. Fresh XML lane
   - `04-01_build_xml`
   - `04-02_verify_xml`
   - run-scoped: `04-01_regular_ingest_build_xml`, `04-02_regular_ingest_verify_xml`
5. Repair XML lane
   - `05-01_ai_repair_enqueue_asset`
   - `05-02_ai_repair_poll_asset`
   - `05-03_reconcile_tags`
   - `05-04_post_repair_build_xml`
   - `05-05_post_repair_verify_xml`
   - run-scoped: `05-01_regular_ingest_ai_repair_enqueue_asset` through `05-05_regular_ingest_post_repair_verify_xml`
6. Sections
   - generic/manual: `06_sections_asset`
   - fresh XML path: `06-01_sections_from_fresh_xml`
   - repair XML path: `06-02_sections_from_repair_xml`
   - run-scoped: `06-01_regular_ingest_sections_from_fresh_xml`, `06-02_regular_ingest_sections_from_repair_xml`
7. Taxonomy
   - generic/manual: `07_taxonomy_asset`
   - run-scoped LLM path: `07-01_regular_ingest_taxonomy_llm_asset`
8. Tax module
   - generic/manual: `08_tax_module_asset`
   - fresh XML path: `08-01_tax_module_from_fresh_xml`
   - repair XML path: `08-02_tax_module_from_repair_xml`
   - run-scoped: `08-03_regular_ingest_tax_module_asset`
9. Taxonomy gold backfill
   - run-scoped only: `09_regular_ingest_taxonomy_gold_backfill_asset`
10. Transaction metadata
   - generic/manual: `10_tx_metadata_asset`
   - run-scoped: `10-01_regular_ingest_tx_metadata_offline_asset`, `10-02_regular_ingest_tx_metadata_web_search_asset`
11. Section embeddings
   - generic/manual: `11_embed_sections`
99. Maintenance
   - isolated/manual: `99_gating`

## Generic And Manual Assets

The following assets remain intentionally registered even though they are not part of the three explicit jobs:

- `01_staging_asset`
- `02_pre_processing_asset`
- `03_tagging_asset`
- `06_sections_asset`
- `07_taxonomy_asset`
- `08_tax_module_asset`
- `10_tx_metadata_asset`
- `11_embed_sections`
- `99_gating`

These are manual or generic entrypoints for maintenance, backfills, or ad hoc Launchpad runs. They are not dead assets.

## Fresh XML vs Repair XML

Fresh XML assets (`04-01`, `04-02`) operate on agreements that are not yet marked as AI-repair attempted:
- build or refresh when no latest XML exists, or body-tagged pages are newer than latest XML
- verify only latest rows with `status IS NULL` and `COALESCE(ai_repair_attempted, 0) = 0`

Repair XML assets (`05-04`, `05-05`) operate on agreements already in the repair cycle:
- rebuild only when latest XML is `invalid`, `ai_repair_attempted = 1`, and tags were updated since XML creation
- verify only latest rows with `status IS NULL` and `ai_repair_attempted = 1`

## AI Repair Selection

`05-01_ai_repair_enqueue_asset` currently:
- targets agreements whose latest XML is `invalid`
- limits to a hardcoded list of repair-eligible XML reason codes
- includes all unresolved spans with no entity-type or confidence-threshold filter
- orders agreements by fewest affected pages first

Batch size for selection is `xml_agreement_batch_size`.

## Sections Upload Behavior

Sections are inserted only from latest verified XML versions:
- `06-01_sections_from_fresh_xml` consumes verified UUIDs from `04-02_verify_xml`
- `06-02_sections_from_repair_xml` consumes verified UUIDs from `05-05_post_repair_verify_xml`

`06_sections_asset` remains the generic manual sections run.

## Gating And Summary Refresh

`99_gating` (`src/etl/defs/gating_asset.py`) is the isolated manual maintenance asset that applies gating and refreshes summary data.

Stage math and stage selectors share canonical SQL in `src/etl/utils/pipeline_state_sql.py`. The same predicates drive:
- stage and color classification in `agreement_status_summary`
- persisted gating flags
- queue selection in pre-processing, tagging, XML, and AI-repair enqueue

Post-run refresh (`run_post_asset_refresh`) is automatic only for stage-ending assets:
- `01_staging_asset`
- `02_pre_processing_asset`
- `03_tagging_asset`
- `06_sections_asset`
- `06-01_sections_from_fresh_xml`
- `06-02_sections_from_repair_xml`
- `07_taxonomy_asset`
- `10_tx_metadata_asset`
- `11_embed_sections`

## Key Config Knobs

Defined in `src/etl/defs/resources.py` and loaded from `etl/configs/pipeline_config.yaml`:

- `pre_processing_mode`: `from_scratch` | `cleanup`
- `queue_run_mode`: `single_batch` | `drain`
- `resume_openai_batches`
- `pre_processing_agreement_batch_size`
- `tagging_agreement_batch_size`
- `xml_agreement_batch_size`
- `taxonomy_agreement_batch_size`
- `taxonomy_mode`: `llm` | `ml` | `gold_backfill`
- `taxonomy_section_title_regex`
- `taxonomy_llm_model`
- `taxonomy_llm_sections_per_request`
- `tax_module_agreement_batch_size`
- `tax_module_llm_model`
- `tax_module_llm_clauses_per_request`
- `tx_metadata_agreement_batch_size`
- `tx_metadata_mode`: `offline` | `web_search`
- `embed_target`: `agreement` | `section`
- `embed_focus_section`
- `embed_agreement_batch_size`
- `embed_focus_section_batch_size`
- `refresh`

Run-scoped XML, repair, and downstream enrichment assets accept upstream-scoped batches and are intended to preserve that run scope rather than drain the global queue.
