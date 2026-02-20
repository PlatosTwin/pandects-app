"""Apply NER tagging to processed pages and persist outputs."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false

from typing import cast

import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import bindparam, text

from etl.defs.b_pre_processing_asset import pre_processing_asset
from etl.defs.resources import DBResource, PipelineConfig, TaggingModel
from etl.domain.c_tagging import (
    TaggingModelProtocol,
    TaggingRow,
    ContextProtocol as TaggingContext,
    tag,
)
from etl.utils.db_utils import upsert_tags
from etl.utils.post_asset_refresh import run_post_asset_refresh
from etl.utils.pipeline_state_sql import canonical_tagging_queue_sql
from etl.utils.run_config import is_batched


@dg.asset(deps=[pre_processing_asset], name="3_tagging_asset")
def tagging_asset(
    context: AssetExecutionContext,
    db: DBResource,
    tagging_model: TaggingModel,
    pipeline_config: PipelineConfig,
) -> None:
    """Apply NER tagging to processed pages.

    Args:
        context: Dagster execution context.
        db: Database resource for connection.
        tagging_model: Model for page tagging.
        pipeline_config: Pipeline configuration.
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
    context.log.info("Running tagging")

    while True:
        with engine.begin() as conn:
            agreement_uuids = (
                conn.execute(
                    text(canonical_tagging_queue_sql(schema)),
                    {"last_uuid": last_uuid, "batch_size": agreement_batch_size},
                )
                .scalars()
                .all()
            )
            if not agreement_uuids:
                break

            rows_mapping = (
                conn.execute(
                    text(
                        f"""
                        SELECT
                            p.agreement_uuid,
                            p.page_uuid,
                            p.processed_page_content
                        FROM {pages_table} p
                        LEFT JOIN {tagged_outputs_table} t
                            ON t.page_uuid = p.page_uuid
                        WHERE p.agreement_uuid IN :uuids
                          AND COALESCE(p.gold_label, p.source_page_type) = 'body'
                          AND p.processed_page_content IS NOT NULL
                          AND t.page_uuid IS NULL
                        ORDER BY p.agreement_uuid ASC, p.page_order ASC, p.page_uuid ASC
                        """
                    ).bindparams(bindparam("uuids", expanding=True)),
                    {"uuids": agreement_uuids},
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

            last_uuid = str(agreement_uuids[-1])

        if batched:
            break

    run_post_asset_refresh(context, db, pipeline_config)
