"""Apply NER tagging to processed pages and persist outputs."""

from typing import Any, Dict, List

import dagster as dg
from sqlalchemy import text

from etl.defs.b_pre_processing_asset import pre_processing_asset
from etl.defs.resources import DBResource, PipelineConfig, TaggingModel
from etl.domain.c_tagging import tag
from etl.utils.db_utils import upsert_tags
from etl.utils.run_config import get_int_tag, is_batched, is_cleanup_mode


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
    page_bs_tag = get_int_tag(context, "tagging_page_batch_size")
    batch_size = page_bs_tag if page_bs_tag is not None else pipeline_config.tagging_page_batch_size
    batched = is_batched(context, pipeline_config)

    last_uuid: str = ""
    engine = db.get_engine()
    is_cleanup = is_cleanup_mode(context, pipeline_config)

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
                        AND coalesce(p.gold_label, p.source_page_type) = 'body'
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
