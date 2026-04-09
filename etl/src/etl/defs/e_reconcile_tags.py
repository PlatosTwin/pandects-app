# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
from typing import List

import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import bindparam, text

from etl.defs.d_ai_repair_asset import (
    ai_repair_poll_asset,
    ingestion_cleanup_a_ai_repair_poll_asset,
    ingestion_cleanup_b_ai_repair_poll_asset,
    regular_ingest_ai_repair_poll_asset,
)
from etl.defs.resources import DBResource, PipelineConfig
from etl.utils.logical_job_runs import mark_logical_run_stage_completed, should_skip_managed_stage
from etl.utils.post_asset_refresh import run_post_asset_refresh


def _reconcile_tags_for_requests(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    polled_request_ids: List[str],
    *,
    log_prefix: str,
) -> List[str]:
    """
    Merge full-page AI outputs into corrected tagged text and update
    pdx.tagged_outputs.tagged_text_corrected.
    Existing corrected text is overwritten when new AI-repair output differs.
    Excerpt rulings are ignored in the XML-repair flow.
    """
    engine = db.get_engine()
    schema = db.database
    pages_table = f"{schema}.pages"
    xml_table = f"{schema}.xml"
    tagged_outputs_table = f"{schema}.tagged_outputs"
    ai_repair_requests_table = f"{schema}.ai_repair_requests"
    ai_repair_full_pages_table = f"{schema}.ai_repair_full_pages"

    target_request_ids = sorted(set(polled_request_ids))
    if not target_request_ids:
        context.log.info("%s: no upstream successful full-page request IDs from poll.", log_prefix)
        run_post_asset_refresh(context, db, pipeline_config)
        return []

    with engine.begin() as conn:
        full_rows = (
            conn.execute(
                text(
                    f"""
                    SELECT
                        p.agreement_uuid,
                        p.page_uuid,
                        f.tagged_text
                    FROM {ai_repair_requests_table} r
                    JOIN {pages_table} p
                        ON p.page_uuid = r.page_uuid
                    JOIN {xml_table} x
                        ON x.agreement_uuid = p.agreement_uuid
                       AND x.latest = 1
                    JOIN {ai_repair_full_pages_table} f
                        ON f.request_id = r.request_id
                    WHERE r.request_id IN :rids
                      AND r.status = 'completed'
                      AND r.mode = 'full'
                      AND CAST(SUBSTRING_INDEX(r.request_id, '::', -1) AS UNSIGNED) = x.version
                    ORDER BY p.agreement_uuid, p.page_uuid
                    """
                ).bindparams(bindparam("rids", expanding=True)),
                {"rids": tuple(target_request_ids)},
            )
            .mappings()
            .fetchall()
        )

        if not full_rows:
            context.log.info(
                "%s: no completed full-page outputs for upstream agreements on latest XML version.",
                log_prefix,
            )
            run_post_asset_refresh(context, db, pipeline_config)
            return []

        update_corrected = text(
            f"""
            UPDATE {tagged_outputs_table}
            SET tagged_text_corrected = :txt
            WHERE page_uuid = :pid
              AND NOT (tagged_text_corrected <=> :txt)
            """
        )
        pages_considered = 0
        pages_updated = 0
        updated_agreements: set[str] = set()
        for row in full_rows:
            pages_considered += 1
            agreement_uuid = str(row["agreement_uuid"])
            page_uuid = str(row["page_uuid"])
            tagged_text = str(row["tagged_text"] or "")
            result = conn.execute(update_corrected, {"pid": page_uuid, "txt": tagged_text})
            if int(result.rowcount or 0) > 0:
                pages_updated += 1
                updated_agreements.add(agreement_uuid)

        context.log.info(
            "Batch statistics: full-pages considered=%s, updated=%s, unchanged=%s",
            pages_considered,
            pages_updated,
            pages_considered - pages_updated,
        )

    run_post_asset_refresh(context, db, pipeline_config)
    return sorted(updated_agreements)


@dg.asset(
    name="05-03_reconcile_tags",
    ins={"polled_request_ids": dg.AssetIn(key=ai_repair_poll_asset.key)},
)
def reconcile_tags(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    polled_request_ids: List[str],
) -> List[str]:
    return _reconcile_tags_for_requests(
        context,
        db,
        pipeline_config,
        polled_request_ids,
        log_prefix="reconcile_tags",
    )


@dg.asset(
    name="05-03_regular_ingest_reconcile_tags",
    ins={"polled_request_ids": dg.AssetIn(key=regular_ingest_ai_repair_poll_asset.key)},
)
def regular_ingest_reconcile_tags(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    polled_request_ids: List[str],
) -> List[str]:
    updated_agreements = _reconcile_tags_for_requests(
        context,
        db,
        pipeline_config,
        polled_request_ids,
        log_prefix="regular_ingest_reconcile_tags",
    )
    mark_logical_run_stage_completed(
        db=db,
        job_name="regular_ingest",
        stage_name="regular_ingest_reconcile_tags",
    )
    return updated_agreements


@dg.asset(
    name="05-08_ingestion_cleanup_a_reconcile_tags",
    ins={"polled_request_ids": dg.AssetIn(key=ingestion_cleanup_a_ai_repair_poll_asset.key)},
)
def ingestion_cleanup_a_reconcile_tags(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    polled_request_ids: List[str],
) -> List[str]:
    updated_agreements = _reconcile_tags_for_requests(
        context,
        db,
        pipeline_config,
        polled_request_ids,
        log_prefix="ingestion_cleanup_a_reconcile_tags",
    )
    mark_logical_run_stage_completed(
        db=db,
        job_name="ingestion_cleanup_a",
        stage_name="ingestion_cleanup_a_reconcile_tags",
    )
    return updated_agreements


@dg.asset(
    name="05-12_ingestion_cleanup_b_reconcile_tags",
    ins={"polled_request_ids": dg.AssetIn(key=ingestion_cleanup_b_ai_repair_poll_asset.key)},
)
def ingestion_cleanup_b_reconcile_tags(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    polled_request_ids: List[str],
) -> List[str]:
    should_skip, current_stage = should_skip_managed_stage(
        db=db,
        job_name="ingestion_cleanup_b",
        stage_name="ingestion_cleanup_b_reconcile_tags",
    )
    if should_skip:
        context.log.info(
            "ingestion_cleanup_b_reconcile_tags: skipping because logical run already reached %s.",
            current_stage,
        )
        return []
    updated_agreements = _reconcile_tags_for_requests(
        context,
        db,
        pipeline_config,
        polled_request_ids,
        log_prefix="ingestion_cleanup_b_reconcile_tags",
    )
    mark_logical_run_stage_completed(
        db=db,
        job_name="ingestion_cleanup_b",
        stage_name="ingestion_cleanup_b_reconcile_tags",
    )
    return updated_agreements
