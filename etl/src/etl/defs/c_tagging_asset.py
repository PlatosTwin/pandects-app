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
from etl.utils.db_utils import upsert_tags
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
    is_cleanup = is_cleanup_mode(context, pipeline_config)

    context.log.info(
        f"Running tagging in {'CLEANUP' if is_cleanup else 'FROM_SCRATCH'} mode"
    )

    ran_batches = 0
    while True:
        with engine.begin() as conn:
            # Fetch batch of agreements with at least one body page missing tags
            agreement_uuids = (
                conn.execute(
                    text(
                        """
                        WITH uncertain_pages AS (
                            SELECT DISTINCT
                                agreement_uuid
                            FROM (
                                SELECT
                                    agreement_uuid,
                                    page_order,
                                    page_type_prob_front_matter,
                                    page_type_prob_toc,
                                    page_type_prob_body,
                                    page_type_prob_sig,
                                    page_type_prob_back_matter,
                                    gold_label,
                                    MIN(
                                        CASE
                                            WHEN source_page_type = 'sig'
                                            AND page_type_prob_sig >= 0.95
                                            THEN page_order
                                        END
                                    ) OVER (PARTITION BY agreement_uuid) AS sig_cutoff_page
                                FROM
                                    pdx.pages
                            ) sub
                            WHERE
                                (
                                    page_type_prob_front_matter BETWEEN 0.3 AND 0.7
                                    OR page_type_prob_toc BETWEEN 0.3 AND 0.7
                                    OR page_type_prob_body BETWEEN 0.3 AND 0.7
                                    OR page_type_prob_sig BETWEEN 0.3 AND 0.7
                                    OR page_type_prob_back_matter BETWEEN 0.3 AND 0.7
                                )
                                AND (
                                    page_order <= sig_cutoff_page
                                    OR sig_cutoff_page IS NULL
                                )
                                AND gold_label IS NULL
                        ),
                        out_of_order_agreements AS (
                            WITH PageRanks AS (
                                SELECT
                                    agreement_uuid,
                                    page_order,
                                    CASE
                                        WHEN source_page_type = 'front_matter' THEN 1
                                        WHEN source_page_type = 'toc' THEN 2
                                        WHEN source_page_type = 'body' THEN 3
                                        WHEN source_page_type = 'sig' THEN 4
                                        WHEN source_page_type = 'back_matter' THEN 5
                                        ELSE 99
                                    END AS type_rank
                                FROM
                                    pdx.pages
                                WHERE
                                    gold_label IS NULL
                            ),
                            RankedPages AS (
                                SELECT
                                    agreement_uuid,
                                    page_order,
                                    type_rank,
                                    MAX(type_rank) OVER (
                                        PARTITION BY agreement_uuid
                                        ORDER BY page_order
                                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                                    ) AS max_prev_type_rank
                                FROM
                                    PageRanks
                            )
                            SELECT DISTINCT
                                agreement_uuid
                            FROM
                                RankedPages
                            WHERE
                                max_prev_type_rank IS NOT NULL
                                AND type_rank < max_prev_type_rank
                        )
                        SELECT
                            p.agreement_uuid
                        FROM
                            pdx.pages p
                        LEFT JOIN pdx.tagged_outputs t
                            ON t.page_uuid = p.page_uuid
                        WHERE
                            p.agreement_uuid > :last_uuid
                            AND coalesce(p.gold_label, p.source_page_type) = 'body'
                            AND p.processed_page_content IS NOT NULL
                            AND t.page_uuid IS NULL
                            AND NOT EXISTS (
                                SELECT 1
                                FROM uncertain_pages u
                                WHERE u.agreement_uuid = p.agreement_uuid
                            )
                            AND NOT EXISTS (
                                SELECT 1
                                FROM out_of_order_agreements o
                                WHERE o.agreement_uuid = p.agreement_uuid
                            )
                        GROUP BY
                            p.agreement_uuid
                        ORDER BY
                            p.agreement_uuid ASC
                        LIMIT
                            :batch_size
                    """
                    ),
                    {"last_uuid": last_uuid, "batch_size": agreement_batch_size},
                )
                .scalars()
                .all()
            )

            if not agreement_uuids:
                break

            # Fetch body pages missing tags for those agreements
            rows_mapping = (
                conn.execute(
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
                            p.agreement_uuid IN :uuids
                            AND coalesce(p.gold_label, p.source_page_type) = 'body'
                            AND p.processed_page_content IS NOT NULL
                            AND t.page_uuid IS NULL
                        ORDER BY
                            p.agreement_uuid ASC,
                            p.page_order ASC,
                            p.page_uuid ASC
                    """
                    ),
                    {"uuids": tuple(agreement_uuids)},
                )
                .mappings()
                .fetchall()
            )

            if not rows_mapping:
                last_uuid = agreement_uuids[-1]
                continue

            # Apply tagging to pages
            rows: list[TaggingRow] = [
                cast(TaggingRow, cast(object, dict(r))) for r in rows_mapping
            ]
            tagged_pages = tag(
                rows, inference_model, cast(TaggingContext, cast(object, context))
            )

            try:
                upsert_tags(tagged_pages, conn)
                context.log.info(f"Successfully tagged {len(tagged_pages)} pages")
            except Exception as e:
                context.log.error(f"Error upserting tags: {e}")
                raise RuntimeError(e)

            last_uuid = agreement_uuids[-1]

        ran_batches += 1
        if batched:
            break
