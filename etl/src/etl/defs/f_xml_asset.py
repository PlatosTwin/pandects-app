"""Assemble tagged sections into XML documents for each agreement."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false

import pandas as pd

import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import text

from etl.defs.c_tagging_asset import tagging_asset
from etl.defs.e_reconcile_tags import reconcile_tags
from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.f_xml import count_article_tags, generate_xml
from etl.utils.db_utils import upsert_xml
from etl.utils.run_config import is_batched, is_cleanup_mode
from etl.utils.summary_data import refresh_summary_data


@dg.asset(deps=[tagging_asset, reconcile_tags], name="6_xml_asset")
def xml_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> None:
    """
    Assemble tagged sections into XML documents.

    Re-creates XML for agreements where tagged_outputs have been updated since the last XML creation.
    Maintains version numbers and tracks creation dates.
    
    Args:
        context: Dagster execution context.
        db: Database resource for connection.
        pipeline_config: Pipeline configuration for mode.
    """
    # batching controls
    agreement_batch_size = pipeline_config.xml_agreement_batch_size
    batched = is_batched(context, pipeline_config)

    engine = db.get_engine()
    is_cleanup = is_cleanup_mode(context, pipeline_config)

    context.log.info(
        f"Running XML generation in {'CLEANUP' if is_cleanup else 'FROM_SCRATCH'} mode"
    )

    last_uuid = ''
    while True:
        with engine.begin() as conn:
            # Fetch batch of agreements where:
            # 1. Either no XML exists yet, OR
            # 2. tagged_outputs have been updated after the last XML creation date
            agreement_uuids = (
                conn.execute(
                    text(
                        """
                        SELECT DISTINCT
                            a.agreement_uuid
                        FROM
                            pdx.agreements a
                        JOIN pdx.pages p
                            ON p.agreement_uuid = a.agreement_uuid
                        LEFT JOIN pdx.tagged_outputs t
                            ON t.page_uuid = p.page_uuid
                        LEFT JOIN pdx.xml x
                            ON x.agreement_uuid = a.agreement_uuid
                        WHERE a.agreement_uuid > :last_uuid
                        AND p.source_page_type = 'body'
                        AND COALESCE(t.tagged_text_gold, t.tagged_text_corrected, t.tagged_text) IS NOT NULL
                        AND NOT EXISTS (
                            SELECT 1 
                            FROM pdx.pages p_err
                            JOIN pdx.tagged_outputs t_err
                                ON t_err.page_uuid = p_err.page_uuid
                            WHERE t_err.label_error = 1
                            AND p_err.agreement_uuid = a.agreement_uuid
                        )
                        AND (
                            x.agreement_uuid IS NULL
                            OR t.updated_date > x.created_date
                        )
                        GROUP BY a.agreement_uuid
                        HAVING
                        -- ensure that there is at least one body page and that all body pages are tagged
                        SUM(CASE WHEN p.source_page_type = 'body' THEN 1 ELSE 0 END) > 0
                        AND SUM(
                            CASE WHEN p.source_page_type = 'body'
                                AND COALESCE(t.tagged_text_corrected, t.tagged_text) IS NOT NULL
                                THEN 1 ELSE 0 END
                        ) = SUM(CASE WHEN p.source_page_type = 'body' THEN 1 ELSE 0 END)
                        ORDER BY a.agreement_uuid
                        LIMIT :limit;
                """
                    ),
                    {"limit": agreement_batch_size, "last_uuid": last_uuid},
                )
                .scalars()
                .all()
            )

            # If none left, we're done
            if not agreement_uuids:
                break

            # Fetch every page and its tagged output for those agreements
            rows = (
                conn.execute(
                    text(
                        """
                    SELECT
                    p.agreement_uuid,
                    p.page_uuid,
                    p.page_order,
                    coalesce(p.gold_label, p.source_page_type) as source_page_type,
                    coalesce(tgo.tagged_text_gold, tgo.tagged_text_corrected, tgo.tagged_text, p.processed_page_content) as tagged_output,
                    url,
                    acquirer,
                    target,
                    filing_date,
                    source_is_txt,
                    source_is_html
                    FROM pdx.pages p
                    JOIN pdx.agreements a on p.agreement_uuid = a.agreement_uuid
                    LEFT JOIN pdx.tagged_outputs tgo
                    ON p.page_uuid = tgo.page_uuid
                    WHERE p.agreement_uuid IN :uuids
                    ORDER BY p.agreement_uuid, p.page_order
                """
                    ),
                    {"uuids": tuple(agreement_uuids)},
                )
                .mappings()
                .fetchall()
            )

            df = pd.DataFrame(rows)
            min_article_tags = 5
            body_rows = df[df["source_page_type"] == "body"]
            article_counts = (
                body_rows.groupby("agreement_uuid")["tagged_output"]
                .apply(lambda series: count_article_tags("".join(series.to_list())))  # pyright: ignore[reportUnknownLambdaType]
            )
            mask = article_counts >= min_article_tags
            filtered = article_counts[mask]
            eligible_uuids: list[str] = list(filtered.index)  # pyright: ignore[reportAttributeAccessIssue]
            if not eligible_uuids:
                context.log.info(
                    "Skipping batch: no agreements with at least 5 <article> tags."
                )
                last_uuid = agreement_uuids[-1]
                continue
            skipped_count = len(agreement_uuids) - len(eligible_uuids)
            if skipped_count > 0:
                context.log.info(
                    f"Skipping {skipped_count} agreements with fewer than 5 <article> tags."
                )
            df = df[df["agreement_uuid"].isin(eligible_uuids)]
            # Determine version: new agreements get v1, updated pages increment version
            existing_versions = conn.execute(
                text("""
                    SELECT agreement_uuid, MAX(version) as max_version
                    FROM pdx.xml
                    WHERE agreement_uuid IN :uuids
                    GROUP BY agreement_uuid
                """),
                {"uuids": tuple(eligible_uuids)},
            ).mappings().fetchall()
            
            version_map = {row["agreement_uuid"]: row["max_version"] + 1 for row in existing_versions}
            
            # Generate XML from tagged pages
            xml = generate_xml(df, version_map)

            try:
                upsert_xml(xml, conn)
                context.log.info(
                    f"Successfully generated XML for {len(eligible_uuids)} agreements"
                )
            except Exception as e:
                context.log.error(f"Error upserting XML: {e}")
                raise RuntimeError(e)
            
            last_uuid = agreement_uuids[-1]
        if batched:
            break

    refresh_summary_data(context, db)
