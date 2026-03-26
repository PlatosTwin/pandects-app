## ETL Overview

Current ETL execution is split across three jobs (there is no all-in-one `etl_pipeline` job):

1. `cleanup_pipeline`
   - `1_staging_asset` -> `2_pre_processing_asset` -> `3_tagging_asset`
2. `xml_fresh_pipeline`
   - `4-1_build_xml` -> `4-2_verify_xml` -> `6-1_sections_from_fresh_xml`
3. `xml_repair_cycle_pipeline`
   - `5-1_ai_repair_enqueue_asset` -> `5-2_ai_repair_poll_asset` -> `5-3_reconcile_tags`
   - `5-4_post_repair_build_xml` -> `5-5_post_repair_verify_xml` -> `6-2_sections_from_repair_xml`

Core definitions live in `etl/src/etl/defs/jobs.py`.

## Run With Config

Default run config lives in `etl/configs/pipeline_config.yaml`.

```bash
dagster job execute -f etl/src/etl/defs/jobs.py -j cleanup_pipeline -c etl/configs/pipeline_config.yaml
dagster job execute -f etl/src/etl/defs/jobs.py -j xml_fresh_pipeline -c etl/configs/pipeline_config.yaml
dagster job execute -f etl/src/etl/defs/jobs.py -j xml_repair_cycle_pipeline -c etl/configs/pipeline_config.yaml
```

With `dg dev`, open the target asset/job in Launchpad and override config there.

## Pipeline Flow

1. Stage agreements: `src/etl/defs/a_staging_asset.py`
2. Pre-process pages: `src/etl/defs/b_pre_processing_asset.py`
3. Tag body pages: `src/etl/defs/c_tagging_asset.py`
4. Fresh XML lane:
   - build XML (`src/etl/defs/f_xml_asset.py`, `4-1_build_xml`)
   - verify XML (`src/etl/defs/f_xml_asset.py`, `4-2_verify_xml`)
   - upload sections for verified XML only (`src/etl/defs/g_sections_asset.py`, `6-1_sections_from_fresh_xml`)
5. Repair lane (for invalid XML):
   - enqueue LLM repair for selected invalid XML agreements (`src/etl/defs/d_ai_repair_asset.py`, `5-1_ai_repair_enqueue_asset`)
   - poll results (`src/etl/defs/d_ai_repair_asset.py`, `5-2_ai_repair_poll_asset`)
   - reconcile repaired tags (`src/etl/defs/e_reconcile_tags.py`, `5-3_reconcile_tags`)
   - rebuild XML on repaired agreements (`src/etl/defs/f_xml_repair_cycle_asset.py`, `5-4_post_repair_build_xml`)
   - re-verify rebuilt XML (`src/etl/defs/f_xml_repair_cycle_asset.py`, `5-5_post_repair_verify_xml`)
   - upload sections for re-verified XML only (`src/etl/defs/g_sections_asset.py`, `6-2_sections_from_repair_xml`)

Optional downstream assets:
- Taxonomy: `src/etl/defs/h_taxonomy_asset.py` (`7_taxonomy_asset`)
- Transaction metadata: `src/etl/defs/i_tx_metadata_asset.py` (`8_tx_metadata_asset`)

## Fresh XML vs Repair XML

Fresh XML assets (`4-1`, `4-2`) operate on agreements that are not yet marked as AI-repair attempted:
- build/refresh when no latest XML exists, or body-tagged pages are newer than latest XML
- verify only latest rows with `status IS NULL` and `COALESCE(ai_repair_attempted, 0) = 0`

Repair XML assets (`5-4`, `5-5`) operate on agreements already in the repair cycle:
- rebuild only when latest XML is `invalid`, `ai_repair_attempted = 1`, and tags were updated since XML creation
- verify only latest rows with `status IS NULL` and `ai_repair_attempted = 1`

## AI Repair Selection

`5-1_ai_repair_enqueue_asset` currently:
- targets agreements whose latest XML is `invalid`
- limits to a hardcoded list of repair-eligible XML reason codes
- includes all unresolved spans (no entity-type filter, no confidence-threshold filter)
- orders agreements by fewest affected pages first:
  - `ORDER BY COUNT(DISTINCT p.page_uuid) ASC, p.agreement_uuid`

Batch size for selection is `xml_agreement_batch_size`.

## Sections Upload Behavior

Sections are inserted only from latest verified XML versions:
- `6-1_sections_from_fresh_xml` consumes verified UUIDs from `4-2_verify_xml`
- `6-2_sections_from_repair_xml` consumes verified UUIDs from `5-5_post_repair_verify_xml`

`6_sections_asset` still exists as a backward-compatible generic sections run.

## Gating + Summary Refresh

`z_gating` (`src/etl/defs/z_gating_asset.py`) applies gating and refreshes summary data.

Stage math and stage selectors are now intended to share one canonical SQL source
in `src/etl/utils/pipeline_state_sql.py`. The same predicates drive:
- stage/color classification in `agreement_status_summary`,
- persisted gating flags,
- queue selection in pre-processing/tagging/XML/AI-repair enqueue.

Post-run refresh (`run_post_asset_refresh`) is currently automatic only for stage-ending assets:
- `1_staging_asset`
- `2_pre_processing_asset`
- `3_tagging_asset`
- `6_sections_asset`
- `6-1_sections_from_fresh_xml`
- `6-2_sections_from_repair_xml`

## Key Config Knobs

Defined in `src/etl/defs/resources.py` and loaded from `etl/configs/pipeline_config.yaml`:

- `pre_processing_mode`: `from_scratch` | `cleanup`
- `queue_run_mode`: `single_batch` | `drain` for queue-draining assets (`pre_processing`, `tagging`, fresh `xml`, generic `sections`, `taxonomy`)
- `resume_openai_batches`: resume compatible in-flight OpenAI batches for XML verify and AI-repair assets
- `pre_processing_agreement_batch_size`
- `tagging_agreement_batch_size`
- `xml_agreement_batch_size`
- `taxonomy_agreement_batch_size`
- `taxonomy_mode`: `llm` | `ml` | `gold_backfill`
- `taxonomy_section_title_regex`
- `taxonomy_llm_model`
- `taxonomy_llm_sections_per_request`
- `tx_metadata_agreement_batch_size`
- `tx_metadata_mode`: `offline` | `web_search`
- Staging controls:
  - `staging_days_to_fetch`
  - `staging_rate_limit_max_requests`
  - `staging_rate_limit_window_seconds`
  - `staging_max_workers`
  - `staging_use_keyword_filter`

Run-scoped XML/repair assets do not use `queue_run_mode`; they accept exactly one upstream batch and fail fast when that invariant is violated. `tx_metadata_asset` likewise requires `queue_run_mode=single_batch`.

`refresh` remains present in config schema for compatibility, but post-asset refresh routing is currently controlled by asset name in `src/etl/utils/post_asset_refresh.py`.
