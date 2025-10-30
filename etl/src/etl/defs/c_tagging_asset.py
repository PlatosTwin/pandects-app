"""Apply NER tagging to processed pages and persist outputs."""

from typing import Any, Dict, List

import dagster as dg
from sqlalchemy import text

from etl.defs.b_pre_processing_asset import pre_processing_asset
from etl.defs.resources import DBResource, PipelineConfig, TaggingModel
from etl.domain.tagging import tag
from etl.utils.db_utils import upsert_tags


@dg.asset(deps=[pre_processing_asset], name="3_tagging_asset")
def tagging_asset(
    context,
    db: DBResource,
    tagging_model: TaggingModel,
    pipeline_config: PipelineConfig,
) -> None:
    """Apply NER tagging to processed pages.

    In cleanup mode, processes only existing unprocessed pages.

    Args:
        context: Dagster execution context.
        db: Database resource for connection.
        tagging_model: Model for page tagging.
        pipeline_config: Pipeline configuration for mode.
    """
    inference_model = tagging_model.model()

    # batching controls
    page_bs_tag = context.run.tags.get("page_batch_size")
    run_scope_tag = context.run.tags.get("run_scope")
    batch_size: int = (
        int(page_bs_tag) if page_bs_tag else pipeline_config.page_batch_size
    )
    is_batched: bool = (
        run_scope_tag == "batched"
        if run_scope_tag is not None
        else pipeline_config.is_batched()
    )

    last_uuid: str = ""
    engine = db.get_engine()
    mode_tag = context.run.tags.get("pipeline_mode")
    is_cleanup = (
        (mode_tag == "cleanup")
        if mode_tag is not None
        else pipeline_config.is_cleanup_mode()
    )

    # Override mode from job context if available
    if hasattr(context, "job_def") and hasattr(context.job_def, "config"):
        job_config = context.job_def.config
        if hasattr(job_config, "mode"):
            is_cleanup = job_config.mode.value == "cleanup"

    context.log.info(
        f"Running tagging in {'CLEANUP' if is_cleanup else 'FROM_SCRATCH'} mode"
    )

    ran_batches = 0
    while True:
        with engine.begin() as conn:
            # Fetch batch of body pages that are missing tags
            result = conn.execute(
                text(
                    """
                    SELECT
                        p.page_uuid,
                        p.processed_page_content
                    FROM
                        pdx.pages p
                    LEFT JOIN pdx.tagged_outputs t
                        ON t.page_uuid = p.page_uuid
                    WHERE
                        p.page_uuid > :last_uuid
                        AND p.source_page_type = 'body'
                        AND p.processed_page_content IS NOT NULL
                        AND t.page_uuid IS NULL
                    ORDER BY
                        p.page_uuid ASC
                    LIMIT
                        :batch_size
                """
                ),
                {"last_uuid": last_uuid, "batch_size": batch_size},
            )
            rows_mapping = result.mappings().fetchall()

            if not rows_mapping:
                break

            # Apply tagging to pages
            rows: List[Dict[str, Any]] = [dict(r) for r in rows_mapping]
            tagged_pages = tag(rows, inference_model, context)

            try:
                upsert_tags(tagged_pages, conn)
                context.log.info(f"Successfully tagged {len(tagged_pages)} pages")
            except Exception as e:
                context.log.error(f"Error upserting tags: {e}")
                raise RuntimeError(e)

            last_uuid = rows[-1]["page_uuid"]

        ran_batches += 1
        if is_batched:
            break
