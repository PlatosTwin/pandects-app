"""Apply NER tagging to processed pages and persist outputs."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false

from typing import cast

import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import text

from etl.defs.b_pre_processing_asset import pre_processing_asset
from etl.defs.resources import DBResource, PipelineConfig, TaggingModel
from etl.domain.c_tagging import (
    TaggingModelProtocol,
    TaggingRow,
    ContextProtocol as TaggingContext,
    tag,
)
from etl.domain.z_gating import apply_pages_gating
from etl.utils.db_utils import upsert_tags
from etl.utils.post_asset_refresh import run_post_asset_refresh
from etl.utils.run_config import is_batched, is_cleanup_mode


@dg.asset(deps=[pre_processing_asset], name="3_tagging_asset")
def tagging_asset(
    context: AssetExecutionContext,
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
    inference_model = cast(
        TaggingModelProtocol, cast(object, tagging_model.model())
    )

    # batching controls
    agreement_batch_size = pipeline_config.tagging_agreement_batch_size
    batched = is_batched(context, pipeline_config)

    last_uuid: str = ""
    engine = db.get_engine()
    schema = db.database
    pages_table = f"{schema}.pages"
    tagged_outputs_table = f"{schema}.tagged_outputs"
    is_cleanup = is_cleanup_mode(context, pipeline_config)

    context.log.info(
        f"Running tagging in {'CLEANUP' if is_cleanup else 'FROM_SCRATCH'} mode"
    )

    with engine.begin() as conn:
        _ = apply_pages_gating(conn, db.database)

    while True:
        with engine.begin() as conn:
            # Single query: agreements + body pages missing tags via CTE
            rows_mapping = (
                conn.execute(
                    text(
                        f"""
                        WITH agreement_batch AS (
                            SELECT p.agreement_uuid
                            FROM {pages_table} p
                            LEFT JOIN {tagged_outputs_table} t ON t.page_uuid = p.page_uuid
                            WHERE p.agreement_uuid > :last_uuid
                              AND coalesce(p.gold_label, p.source_page_type) = 'body'
                              AND p.processed_page_content IS NOT NULL
                              AND t.page_uuid IS NULL
                              AND p.gated = 0
                            GROUP BY p.agreement_uuid
                            ORDER BY p.agreement_uuid ASC
                            LIMIT :batch_size
                        )
                        SELECT
                            p.agreement_uuid,
                            p.page_uuid,
                            p.processed_page_content
                        FROM {pages_table} p
                        LEFT JOIN {tagged_outputs_table} t ON t.page_uuid = p.page_uuid
                        WHERE p.agreement_uuid IN (SELECT agreement_uuid FROM agreement_batch)
                          AND coalesce(p.gold_label, p.source_page_type) = 'body'
                          AND p.processed_page_content IS NOT NULL
                          AND t.page_uuid IS NULL
                        ORDER BY p.agreement_uuid ASC, p.page_order ASC, p.page_uuid ASC
                        """
                    ),
                    {"last_uuid": last_uuid, "batch_size": agreement_batch_size},
                )
                .mappings()
                .fetchall()
            )

            if not rows_mapping:
                break

            # Apply tagging to pages
            rows: list[TaggingRow] = [
                cast(TaggingRow, cast(object, dict(r))) for r in rows_mapping
            ]
            tagged_pages = tag(
                rows, inference_model, cast(TaggingContext, cast(object, context))
            )

            try:
                upsert_tags(tagged_pages, db.database, conn)
                context.log.info(f"Successfully tagged {len(tagged_pages)} pages")
            except Exception as e:
                context.log.error(f"Error upserting tags: {e}")
                raise RuntimeError(e)

            last_uuid = max(r["agreement_uuid"] for r in rows_mapping)

        if batched:
            break

    run_post_asset_refresh(context, db, pipeline_config)
