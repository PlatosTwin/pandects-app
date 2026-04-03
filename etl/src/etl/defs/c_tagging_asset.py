"""Apply NER tagging to processed pages and persist outputs."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false

from typing import cast

import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import bindparam, text

from etl.defs.b_pre_processing_asset import (
    pre_processing_asset,
    regular_ingest_pre_processing_asset,
)
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
from etl.utils.run_config import runs_single_batch


def _run_tagging_for_agreements(
    context: AssetExecutionContext,
    *,
    db: DBResource,
    tagging_model: TaggingModel,
    pipeline_config: PipelineConfig,
    target_agreement_uuids: list[str] | None,
    log_prefix: str,
) -> list[str]:
    inference_model = cast(
        TaggingModelProtocol, cast(object, tagging_model.model())
    )

    agreement_batch_size = pipeline_config.tagging_agreement_batch_size
    single_batch_run = runs_single_batch(context, pipeline_config)

    last_uuid = ""
    engine = db.get_engine()
    schema = db.database
    pages_table = f"{schema}.pages"
    tagged_outputs_table = f"{schema}.tagged_outputs"
    scoped_uuids = sorted(set(target_agreement_uuids or []))
    processed_agreement_uuids: set[str] = set()

    while True:
        with engine.begin() as conn:
            if scoped_uuids:
                agreement_uuids = (
                    conn.execute(
                        text(canonical_tagging_queue_sql(schema, scoped=True)).bindparams(
                            bindparam("auuids", expanding=True)
                        ),
                        {
                            "auuids": tuple(scoped_uuids),
                            "batch_size": max(agreement_batch_size, len(scoped_uuids)),
                        },
                    )
                    .scalars()
                    .all()
                )
            else:
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
                if scoped_uuids:
                    break
                last_uuid = str(agreement_uuids[-1])
                continue

            rows: list[TaggingRow] = [
                cast(TaggingRow, cast(object, dict(r))) for r in rows_mapping
            ]
            tagged_pages = tag(
                rows, inference_model, cast(TaggingContext, cast(object, context))
            )

            try:
                upsert_tags(tagged_pages, db.database, conn)
                processed_agreement_uuids.update(
                    str(row["agreement_uuid"])
                    for row in rows_mapping
                    if row.get("agreement_uuid") is not None
                )
                context.log.info("%s: successfully tagged %s pages", log_prefix, len(tagged_pages))
            except Exception as e:
                context.log.error(f"{log_prefix}: error upserting tags: {e}")
                raise RuntimeError(e)

            if scoped_uuids:
                break

            last_uuid = str(agreement_uuids[-1])

        if single_batch_run:
            break

    return sorted(processed_agreement_uuids)


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
    context.log.info("Running tagging")
    processed_agreement_uuids = _run_tagging_for_agreements(
        context,
        db=db,
        tagging_model=tagging_model,
        pipeline_config=pipeline_config,
        target_agreement_uuids=None,
        log_prefix="tagging_asset",
    )
    context.log.info("tagging_asset: processed %s agreements", len(processed_agreement_uuids))
    run_post_asset_refresh(context, db, pipeline_config)


@dg.asset(
    name="3-1_regular_ingest_tagging_asset",
    ins={"pre_processed_agreement_uuids": dg.AssetIn(key=regular_ingest_pre_processing_asset.key)},
)
def regular_ingest_tagging_asset(
    context: AssetExecutionContext,
    db: DBResource,
    tagging_model: TaggingModel,
    pipeline_config: PipelineConfig,
    pre_processed_agreement_uuids: list[str],
) -> list[str]:
    processed_agreement_uuids = _run_tagging_for_agreements(
        context,
        db=db,
        tagging_model=tagging_model,
        pipeline_config=pipeline_config,
        target_agreement_uuids=pre_processed_agreement_uuids,
        log_prefix="regular_ingest_tagging_asset",
    )
    run_post_asset_refresh(context, db, pipeline_config)
    return processed_agreement_uuids
