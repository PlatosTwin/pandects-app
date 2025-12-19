"""Split agreements into pages, classify page types, and format text.

Respects CLEANUP vs FROM_SCRATCH modes via context tags or `PipelineConfig`.
Writes pages to `pdx.pages`; FROM_SCRATCH inserts, CLEANUP updates.
"""

from typing import Any, Dict, List

import dagster as dg
from sqlalchemy import text

from etl.defs.a_staging_asset import staging_asset
from etl.defs.resources import ClassifierModel, DBResource, PipelineConfig
from etl.domain.b_pre_processing import cleanup, pre_process
from etl.utils.db_utils import upsert_pages
from etl.utils.run_config import is_cleanup_mode


@dg.asset(deps=[staging_asset], name="2_pre_processing_asset")
def pre_processing_asset(
    context,
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

    batch_size: int = 5  # Process pages from batch_size agreements at a time
    
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
                        """
                    SELECT
                        agreement_uuid,
                        url
                    FROM
                        pdx.agreements
                    WHERE
                        agreement_uuid > :last_uuid
                        AND NOT EXISTS (
                            SELECT 1 FROM pdx.pages p
                            WHERE p.agreement_uuid = pdx.agreements.agreement_uuid
                        )
                    ORDER BY
                        agreement_uuid ASC
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
                agreements: List[Dict[str, str]] = [
                    {"agreement_uuid": r[0], "url": r[1]} for r in rows
                ]

                # Process (tag and format) agreements
                staged_pages = pre_process(context, agreements, inference_model)

                if staged_pages:
                    try:
                        upsert_pages(staged_pages, operation_type="insert", conn=conn)
                        context.log.info(
                            f"Successfully processed {len(staged_pages)} pages from {len(agreements)} agreements"
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
                # Fetch pages that are missing derived outputs (processed text or page type)
                result = conn.execute(
                    text(
                        """
                    WITH agreement_batch AS (
                        SELECT
                            distinct agreement_uuid
                        FROM
                            pdx.pages
                        WHERE
                            agreement_uuid > :last_uuid
                            AND (processed_page_content IS NULL OR source_page_type IS NULL)
                        ORDER BY
                            agreement_uuid ASC
                        LIMIT :batch_size
                    )
                    SELECT
                        p.agreement_uuid,
                        p.page_uuid,
                        p.raw_page_content,
                        p.source_is_txt,
                        p.source_is_html
                    FROM
                        pdx.pages AS p
                    WHERE
                        p.agreement_uuid in (SELECT agreement_uuid FROM agreement_batch)
                        AND (p.processed_page_content IS NULL OR p.source_page_type IS NULL)
                    ORDER BY
                        p.page_order,
                        p.page_uuid;
                    """
                    ),
                    {"last_uuid": last_uuid, "batch_size": batch_size},
                )
                rows = result.fetchall()
                if not rows:
                    break
                last_uuid = max(r[0] for r in rows)

                # Process existing pages
                pages: List[Dict[str, Any]] = [
                    {
                        "agreement_uuid": r[0],
                        "page_uuid": r[1],
                        "content": r[2],
                        "is_txt": r[3],
                        "is_html": r[4],
                    }
                    for r in rows
                ]
                staged_pages = cleanup(pages, inference_model)

                if staged_pages:
                    try:
                        upsert_pages(staged_pages, operation_type="update", conn=conn)
                        
                        num_agr = len(set(p["agreement_uuid"] for p in pages))
                        num_pages = len(staged_pages)
                        totals_dict["agreements"] += num_agr
                        totals_dict["pages"] += num_pages

                        context.log.info(
                            f"Successfully updated {num_pages} pages from {num_agr} agreements. "
                            f"Total pages: {totals_dict['pages']}. Total agreements: {totals_dict['agreements']}"
                        )
                        
                    except Exception as e:
                        context.log.error(f"Error upserting pages: {e}")
                        raise RuntimeError(e)
