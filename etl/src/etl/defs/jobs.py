# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
import dagster as dg
from etl.defs.a_staging_asset import regular_ingest_staging_asset, staging_asset
from etl.defs.b_pre_processing_asset import pre_processing_asset, regular_ingest_pre_processing_asset
from etl.defs.c_tagging_asset import (
    ingestion_cleanup_a_tagging_asset,
    regular_ingest_tagging_asset,
    tagging_asset,
)
from etl.defs.d_ai_repair_asset import (
    ai_repair_enqueue_asset,
    ai_repair_poll_asset,
    ingestion_cleanup_a_ai_repair_enqueue_asset,
    ingestion_cleanup_a_ai_repair_poll_asset,
    ingestion_cleanup_b_ai_repair_enqueue_asset,
    ingestion_cleanup_b_ai_repair_poll_asset,
    regular_ingest_ai_repair_enqueue_asset,
    regular_ingest_ai_repair_poll_asset,
)
from etl.defs.e_reconcile_tags import (
    ingestion_cleanup_a_reconcile_tags,
    ingestion_cleanup_b_reconcile_tags,
    reconcile_tags,
    regular_ingest_reconcile_tags,
)
from etl.defs.f_xml_asset import (
    ingestion_cleanup_a_xml_asset,
    ingestion_cleanup_a_xml_verify_asset,
    ingestion_cleanup_c_xml_asset,
    ingestion_cleanup_c_xml_verify_asset,
    regular_ingest_xml_asset,
    regular_ingest_xml_verify_asset,
    xml_asset,
    xml_verify_asset,
)
from etl.defs.f_xml_repair_cycle_asset import (
    ingestion_cleanup_a_post_repair_build_xml_asset,
    ingestion_cleanup_a_post_repair_verify_xml_asset,
    ingestion_cleanup_b_post_repair_build_xml_asset,
    ingestion_cleanup_b_post_repair_verify_xml_asset,
    post_repair_build_xml_asset,
    post_repair_verify_xml_asset,
    regular_ingest_post_repair_build_xml_asset,
    regular_ingest_post_repair_verify_xml_asset,
)
from etl.defs.g_sections_asset import (
    ingestion_cleanup_a_sections_from_fresh_xml_asset,
    ingestion_cleanup_a_sections_from_repair_xml_asset,
    ingestion_cleanup_b_sections_from_repair_xml_asset,
    ingestion_cleanup_c_sections_asset,
    regular_ingest_sections_from_fresh_xml_asset,
    regular_ingest_sections_from_repair_xml_asset,
    sections_asset,
    sections_from_fresh_xml_asset,
    sections_from_repair_xml_asset,
)
from etl.defs.h_taxonomy_asset import (
    ingestion_cleanup_a_taxonomy_gold_backfill_asset,
    ingestion_cleanup_a_taxonomy_llm_asset,
    ingestion_cleanup_b_taxonomy_gold_backfill_asset,
    ingestion_cleanup_b_taxonomy_llm_asset,
    ingestion_cleanup_c_taxonomy_gold_backfill_asset,
    ingestion_cleanup_c_taxonomy_llm_asset,
    regular_ingest_taxonomy_gold_backfill_asset,
    regular_ingest_taxonomy_llm_asset,
    taxonomy_asset,
)
from etl.defs.i_tx_metadata_asset import (
    ingestion_cleanup_a_tx_metadata_offline_asset,
    ingestion_cleanup_a_tx_metadata_web_search_asset,
    ingestion_cleanup_b_tx_metadata_offline_asset,
    ingestion_cleanup_b_tx_metadata_web_search_asset,
    ingestion_cleanup_c_tx_metadata_offline_asset,
    ingestion_cleanup_c_tx_metadata_web_search_asset,
    regular_ingest_tx_metadata_offline_asset,
    regular_ingest_tx_metadata_web_search_asset,
    tx_metadata_asset,
)
from etl.defs.j_embed_sections_asset import embed_sections_asset
from etl.defs.k_tax_module_asset import (
    ingestion_cleanup_a_tax_module_asset,
    ingestion_cleanup_b_tax_module_asset,
    ingestion_cleanup_c_tax_module_asset,
    regular_ingest_tax_module_asset,
    tax_module_asset,
    tax_module_from_fresh_xml_asset,
    tax_module_from_repair_xml_asset,
)
from etl.defs.gating_asset import gating_asset
from etl.defs.resources import get_resources
from etl.utils.logical_job_runs import MANAGED_LOGICAL_JOB_NAMES, mark_logical_run_failed

base_resources = get_resources()


@dg.failure_hook(required_resource_keys={"db"})
def _managed_logical_run_failure_hook(context: dg.HookContext) -> None:
    if context.job_name not in MANAGED_LOGICAL_JOB_NAMES:
        return
    mark_logical_run_failed(
        db=context.resources.db,
        job_name=context.job_name,
        stage_name=context.step_key,
        dagster_run_id=context.run_id,
    )

xml_fresh_pipeline = dg.define_asset_job(
    name="xml_fresh_pipeline",
    selection=dg.AssetSelection.assets(
        xml_asset,
        xml_verify_asset,
        sections_from_fresh_xml_asset,
        tax_module_from_fresh_xml_asset,
    ),
)

xml_repair_cycle_pipeline = dg.define_asset_job(
    name="xml_repair_cycle_pipeline",
    selection=dg.AssetSelection.assets(
        ai_repair_enqueue_asset,
        ai_repair_poll_asset,
        reconcile_tags,
        post_repair_build_xml_asset,
        post_repair_verify_xml_asset,
        sections_from_repair_xml_asset,
        tax_module_from_repair_xml_asset,
    ),
)

regular_ingest = dg.define_asset_job(
    name="regular_ingest",
    hooks={_managed_logical_run_failure_hook},
    selection=dg.AssetSelection.assets(
        regular_ingest_staging_asset,
        regular_ingest_pre_processing_asset,
        regular_ingest_tagging_asset,
        regular_ingest_xml_asset,
        regular_ingest_xml_verify_asset,
        regular_ingest_sections_from_fresh_xml_asset,
        regular_ingest_ai_repair_enqueue_asset,
        regular_ingest_ai_repair_poll_asset,
        regular_ingest_reconcile_tags,
        regular_ingest_post_repair_build_xml_asset,
        regular_ingest_post_repair_verify_xml_asset,
        regular_ingest_sections_from_repair_xml_asset,
        regular_ingest_taxonomy_llm_asset,
        regular_ingest_tax_module_asset,
        regular_ingest_taxonomy_gold_backfill_asset,
        regular_ingest_tx_metadata_offline_asset,
        regular_ingest_tx_metadata_web_search_asset,
    ),
)

ingestion_cleanup_a = dg.define_asset_job(
    name="ingestion_cleanup_a",
    description="Base cleanup job; resumes with page tagging previously gated agreements.",
    hooks={_managed_logical_run_failure_hook},
    selection=dg.AssetSelection.assets(
        ingestion_cleanup_a_tagging_asset,
        ingestion_cleanup_a_xml_asset,
        ingestion_cleanup_a_xml_verify_asset,
        ingestion_cleanup_a_sections_from_fresh_xml_asset,
        ingestion_cleanup_a_ai_repair_enqueue_asset,
        ingestion_cleanup_a_ai_repair_poll_asset,
        ingestion_cleanup_a_reconcile_tags,
        ingestion_cleanup_a_post_repair_build_xml_asset,
        ingestion_cleanup_a_post_repair_verify_xml_asset,
        ingestion_cleanup_a_sections_from_repair_xml_asset,
        ingestion_cleanup_a_taxonomy_llm_asset,
        ingestion_cleanup_a_tax_module_asset,
        ingestion_cleanup_a_taxonomy_gold_backfill_asset,
        ingestion_cleanup_a_tx_metadata_offline_asset,
        ingestion_cleanup_a_tx_metadata_web_search_asset,
    ),
)

ingestion_cleanup_b = dg.define_asset_job(
    name="ingestion_cleanup_b",
    description="Invalid XML cleanup job; starts with attempting to clear invalid XML.",
    hooks={_managed_logical_run_failure_hook},
    selection=dg.AssetSelection.assets(
        ingestion_cleanup_b_ai_repair_enqueue_asset,
        ingestion_cleanup_b_ai_repair_poll_asset,
        ingestion_cleanup_b_reconcile_tags,
        ingestion_cleanup_b_post_repair_build_xml_asset,
        ingestion_cleanup_b_post_repair_verify_xml_asset,
        ingestion_cleanup_b_sections_from_repair_xml_asset,
        ingestion_cleanup_b_taxonomy_llm_asset,
        ingestion_cleanup_b_tax_module_asset,
        ingestion_cleanup_b_taxonomy_gold_backfill_asset,
        ingestion_cleanup_b_tx_metadata_offline_asset,
        ingestion_cleanup_b_tx_metadata_web_search_asset,
    ),
)

ingestion_cleanup_c = dg.define_asset_job(
    name="ingestion_cleanup_c",
    description="Fresh XML cleanup job; resumes by building XML from fresh-XML eligible agreements.",
    hooks={_managed_logical_run_failure_hook},
    selection=dg.AssetSelection.assets(
        ingestion_cleanup_c_xml_asset,
        ingestion_cleanup_c_xml_verify_asset,
        ingestion_cleanup_c_sections_asset,
        ingestion_cleanup_c_taxonomy_llm_asset,
        ingestion_cleanup_c_tax_module_asset,
        ingestion_cleanup_c_taxonomy_gold_backfill_asset,
        ingestion_cleanup_c_tx_metadata_offline_asset,
        ingestion_cleanup_c_tx_metadata_web_search_asset,
    ),
)

defs = dg.Definitions(
    assets=[
        staging_asset,
        regular_ingest_staging_asset,
        pre_processing_asset,
        regular_ingest_pre_processing_asset,
        tagging_asset,
        regular_ingest_tagging_asset,
        ingestion_cleanup_a_tagging_asset,
        ai_repair_enqueue_asset,
        regular_ingest_ai_repair_enqueue_asset,
        ingestion_cleanup_a_ai_repair_enqueue_asset,
        ingestion_cleanup_b_ai_repair_enqueue_asset,
        ai_repair_poll_asset,
        regular_ingest_ai_repair_poll_asset,
        ingestion_cleanup_a_ai_repair_poll_asset,
        ingestion_cleanup_b_ai_repair_poll_asset,
        reconcile_tags,
        regular_ingest_reconcile_tags,
        ingestion_cleanup_a_reconcile_tags,
        ingestion_cleanup_b_reconcile_tags,
        xml_asset,
        regular_ingest_xml_asset,
        ingestion_cleanup_a_xml_asset,
        ingestion_cleanup_c_xml_asset,
        xml_verify_asset,
        regular_ingest_xml_verify_asset,
        ingestion_cleanup_a_xml_verify_asset,
        ingestion_cleanup_c_xml_verify_asset,
        post_repair_build_xml_asset,
        regular_ingest_post_repair_build_xml_asset,
        ingestion_cleanup_a_post_repair_build_xml_asset,
        ingestion_cleanup_b_post_repair_build_xml_asset,
        post_repair_verify_xml_asset,
        regular_ingest_post_repair_verify_xml_asset,
        ingestion_cleanup_a_post_repair_verify_xml_asset,
        ingestion_cleanup_b_post_repair_verify_xml_asset,
        sections_asset,
        sections_from_fresh_xml_asset,
        regular_ingest_sections_from_fresh_xml_asset,
        ingestion_cleanup_a_sections_from_fresh_xml_asset,
        sections_from_repair_xml_asset,
        regular_ingest_sections_from_repair_xml_asset,
        ingestion_cleanup_a_sections_from_repair_xml_asset,
        ingestion_cleanup_b_sections_from_repair_xml_asset,
        ingestion_cleanup_c_sections_asset,
        taxonomy_asset,
        regular_ingest_taxonomy_llm_asset,
        ingestion_cleanup_a_taxonomy_llm_asset,
        ingestion_cleanup_b_taxonomy_llm_asset,
        ingestion_cleanup_c_taxonomy_llm_asset,
        tax_module_asset,
        tax_module_from_fresh_xml_asset,
        tax_module_from_repair_xml_asset,
        regular_ingest_tax_module_asset,
        ingestion_cleanup_a_tax_module_asset,
        ingestion_cleanup_b_tax_module_asset,
        ingestion_cleanup_c_tax_module_asset,
        regular_ingest_taxonomy_gold_backfill_asset,
        ingestion_cleanup_a_taxonomy_gold_backfill_asset,
        ingestion_cleanup_b_taxonomy_gold_backfill_asset,
        ingestion_cleanup_c_taxonomy_gold_backfill_asset,
        tx_metadata_asset,
        regular_ingest_tx_metadata_offline_asset,
        regular_ingest_tx_metadata_web_search_asset,
        ingestion_cleanup_a_tx_metadata_offline_asset,
        ingestion_cleanup_a_tx_metadata_web_search_asset,
        ingestion_cleanup_b_tx_metadata_offline_asset,
        ingestion_cleanup_b_tx_metadata_web_search_asset,
        ingestion_cleanup_c_tx_metadata_offline_asset,
        ingestion_cleanup_c_tx_metadata_web_search_asset,
        embed_sections_asset,
        gating_asset,
    ],
    jobs=[
        xml_fresh_pipeline,
        xml_repair_cycle_pipeline,
        regular_ingest,
        ingestion_cleanup_a,
        ingestion_cleanup_b,
        ingestion_cleanup_c,
    ],
    resources=base_resources,
)
