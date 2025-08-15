"""XML generation asset for assembling tagged sections into XML documents."""

from etl.defs.resources import DBResource, PipelineConfig
from sqlalchemy import text
from etl.domain.xml import generate_xml
import dagster as dg
from etl.defs.tagging_asset import tagging_asset
from etl.utils.db_utils import upsert_xml
import pandas as pd
from typing import List


@dg.asset(deps=[tagging_asset])
def xml_asset(context, db: DBResource, pipeline_config: PipelineConfig) -> None:
    """
    Assemble tagged sections into XML documents.

    In cleanup mode, processes only existing unprocessed agreements.
    
    Args:
        context: Dagster execution context.
        db: Database resource for connection.
        pipeline_config: Pipeline configuration for mode.
    """
    agreement_batch_size: int = 10
    engine = db.get_engine()
    is_cleanup = pipeline_config.is_cleanup_mode()

    # Override mode from job context if available
    if hasattr(context, "job_def") and hasattr(context.job_def, "config"):
        job_config = context.job_def.config
        if hasattr(job_config, "mode"):
            is_cleanup = job_config.mode.value == "cleanup"

    context.log.info(
        f"Running XML generation in {'CLEANUP' if is_cleanup else 'FROM_SCRATCH'} mode"
    )

    while True:
        with engine.begin() as conn:
            last_uuid = ''
            
            # Fetch batch of agreements where all pages have been tagged
            agreement_uuids: List[str] = (
                conn.execute(
                    text(
                        """
                    SELECT
                        a.agreement_uuid
                    FROM
                        pdx.agreements a
                    JOIN
                        pdx.pages p
                        ON a.agreement_uuid = p.agreement_uuid
                    WHERE
                        a.processed = 0
                        AND a.agreement_uuid > :last_uuid
                    GROUP BY
                        a.agreement_uuid
                    HAVING
                        MIN(case when p.source_page_type = 'body' then p.processed end) = 1
                    ORDER BY
                        a.agreement_uuid
                    LIMIT
                        :limit;
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
                    coalesce(tgo.tagged_text, p.processed_page_content) as tagged_output,
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
                    ORDER BY p.agreement_uuid, p.page_uuid
                """
                    ),
                    {"uuids": tuple(agreement_uuids)},
                )
                .mappings()
                .fetchall()
            )

            df = pd.DataFrame(rows)
            # Generate XML from tagged pages
            xml = generate_xml(df)

            try:
                upsert_xml(xml, conn)
                context.log.info(
                    f"Successfully generated XML for {len(agreement_uuids)} agreements"
                )
            except Exception as e:
                context.log.error(f"Error upserting XML: {e}")
                raise RuntimeError(e)
            
            last_uuid = agreement_uuids[-1]
