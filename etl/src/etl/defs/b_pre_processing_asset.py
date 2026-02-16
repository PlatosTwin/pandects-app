"""Split agreements into pages, classify page types, and format text.

Respects CLEANUP vs FROM_SCRATCH modes via `PipelineConfig`.
Writes pages to `pdx.pages`; FROM_SCRATCH inserts, CLEANUP updates.
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false

from typing import cast

import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import text

from etl.defs.a_staging_asset import staging_asset
from etl.defs.resources import ClassifierModel, DBResource, PipelineConfig
from etl.domain.b_pre_processing import (
    AgreementRow,
    CleanupRow,
    ContextProtocol as PreProcessContext,
    cleanup,
    pre_process,
)
from etl.domain.z_gating import apply_agreement_gating
from etl.utils.db_utils import upsert_pages
from etl.utils.post_asset_refresh import run_post_asset_refresh
from etl.utils.run_config import is_cleanup_mode


@dg.asset(deps=[staging_asset], name="2_pre_processing_asset")
def pre_processing_asset(
    context: AssetExecutionContext,
    db: DBResource,
    classifier_model: ClassifierModel,
    pipeline_config: PipelineConfig,
) -> None:
    """Split agreements into pages, classify page types, and process HTML into formatted text.

    In cleanup mode, processes only existing unprocessed agreements.

    Args:
        context: Dagster execution context.
        db: Database resource for connection.
        classifier_model: Model for page classification.
        pipeline_config: Pipeline configuration for mode.
    """
    last_uuid: str = ""
    engine = db.get_engine()
    inference_model = classifier_model.model()

    is_cleanup = is_cleanup_mode(context, pipeline_config)

    batch_size = pipeline_config.pre_processing_agreement_batch_size
    schema = db.database
    agreements_table = f"{schema}.agreements"
    pages_table = f"{schema}.pages"

    with engine.begin() as conn:
        _ = apply_agreement_gating(conn, db.database)
    
    # FROM_SCRATCH mode
    # Just like CLEANUP mode, but we split the agreement into pages first
    # If we're in CLEANUP mode, that means we've already split the agreement into pages
    if not is_cleanup:
        context.log.info("Running pre-processing in FROM_SCRATCH mode")

        while True:
            with engine.begin() as conn:
                # Fetch agreements that do not yet have any pages (idempotent)
                result = conn.execute(
                    text(
                        f"""
                    SELECT
                        a.agreement_uuid,
                        a.url
                    FROM
                        {agreements_table} a
                    LEFT JOIN {pages_table} p ON p.agreement_uuid = a.agreement_uuid
                    WHERE
                        a.agreement_uuid > :last_uuid
                        AND p.page_uuid IS NULL
                        AND a.gated = 0
                        AND (a.paginated IS NULL OR a.paginated = TRUE)
                    ORDER BY
                        a.agreement_uuid ASC
                    LIMIT
                        :batch_size
                    """
                    ),
                    {"last_uuid": last_uuid, "batch_size": batch_size},
                )
                rows = result.fetchall()

                if not rows:
                    break

                # Split agreements into pages
                agreements: list[AgreementRow] = [
                    {"agreement_uuid": r[0], "url": r[1]} for r in rows
                ]

                # Process (tag and format) agreements
                staged_pages, pagination_statuses = pre_process(
                    cast(PreProcessContext, cast(object, context)),
                    agreements,
                    inference_model,
                )

                if pagination_statuses:
                    _ = conn.execute(
                        text(
                            f"""
                        UPDATE {agreements_table}
                        SET paginated = :paginated
                        WHERE agreement_uuid = :agreement_uuid
                          AND NOT (paginated <=> :paginated)
                        """
                        ),
                        [
                            {
                                "agreement_uuid": agreement_uuid,
                                "paginated": paginated,
                            }
                            for agreement_uuid, paginated in pagination_statuses.items()
                        ],
                    )

                if staged_pages:
                    try:
                        upsert_pages(
                            staged_pages, "insert", db.database, conn
                        )
                        agreement_count = len(
                            {p.agreement_uuid for p in staged_pages if p.agreement_uuid}
                        )
                        context.log.info(
                            f"Successfully processed {len(staged_pages)} pages from {agreement_count} agreements"
                        )
                    except Exception as e:
                        context.log.error(f"Error upserting pages: {e}")
                        raise RuntimeError(e)

                last_uuid = rows[-1][0]

    # CLEANUP mode
    # Just like FROM_SCRATCH mode, but we fetch pages that we've already split
    else:
        context.log.info("Running pre-processing in CLEANUP mode")

        totals_dict = {"agreements": 0, "pages": 0}

        # Since we've already split the agreements into pages, we can just fetch the pages
        while True:
            with engine.begin() as conn:
                # Fetch pages that are missing derived outputs (processed text or page type).
                # Use UNION to split OR for index-friendly execution.
                result = conn.execute(
                    text(
                        f"""
                    WITH
                    batch_a AS (
                        SELECT DISTINCT agreement_uuid
                        FROM {pages_table}
                        WHERE agreement_uuid > :last_uuid AND processed_page_content IS NULL
                    ),
                    batch_b AS (
                        SELECT DISTINCT agreement_uuid
                        FROM {pages_table}
                        WHERE agreement_uuid > :last_uuid AND source_page_type IS NULL
                    ),
                    agreement_batch AS (
                        (SELECT agreement_uuid FROM batch_a)
                        UNION
                        (SELECT agreement_uuid FROM batch_b)
                        ORDER BY 1
                        LIMIT :batch_size
                    )
                    SELECT
                        p.agreement_uuid,
                        p.page_uuid,
                        p.raw_page_content,
                        p.page_order,
                        p.source_is_txt,
                        p.source_is_html
                    FROM {pages_table} p
                    WHERE p.agreement_uuid IN (SELECT agreement_uuid FROM agreement_batch)
                      AND (p.processed_page_content IS NULL OR p.source_page_type IS NULL)
                    ORDER BY p.page_order, p.page_uuid
                    """
                    ),
                    {"last_uuid": last_uuid, "batch_size": batch_size},
                )
                rows = result.fetchall()
                if not rows:
                    break
                last_uuid = max(r[0] for r in rows)

                # Process existing pages
                pages = [
                    cast(
                        CleanupRow,
                        cast(
                            object,
                            {
                                "agreement_uuid": cast(str, r[0]),
                                "page_uuid": cast(str | None, r[1]),
                                "content": cast(str, r[2]),
                                "page_order": cast(int, r[3]),
                                "is_txt": cast(bool, r[4]),
                                "is_html": cast(bool, r[5]),
                            },
                        ),
                    )
                    for r in rows
                ]
                staged_pages = cleanup(pages, inference_model)

                if staged_pages:
                    try:
                        upsert_pages(
                            staged_pages, "update", db.database, conn
                        )
                        
                        num_agr = len(set(p["agreement_uuid"] for p in pages))
                        num_pages = len(staged_pages)
                        totals_dict["agreements"] += num_agr
                        totals_dict["pages"] += num_pages

                        context.log.info(
                            f"Successfully updated {num_pages} pages from {num_agr} agreements. Total pages: {totals_dict['pages']}. Total agreements: {totals_dict['agreements']}"
                        )
                        
                    except Exception as e:
                        context.log.error(f"Error upserting pages: {e}")
                        raise RuntimeError(e)

    run_post_asset_refresh(context, db, pipeline_config)
