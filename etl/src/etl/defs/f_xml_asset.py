"""Assemble tagged sections into XML documents for each agreement."""

import pandas as pd

import dagster as dg
from sqlalchemy import text

from etl.defs.c_tagging_asset import tagging_asset
from etl.defs.e_reconcile_tags import reconcile_tags
from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.xml import generate_xml
from etl.utils.db_utils import upsert_xml


@dg.asset(deps=[tagging_asset, reconcile_tags], name="6_xml_asset")
def xml_asset(context, db: DBResource, pipeline_config: PipelineConfig) -> None:
    """
    Assemble tagged sections into XML documents.

    In cleanup mode, processes only existing unprocessed agreements.
    
    Args:
        context: Dagster execution context.
        db: Database resource for connection.
        pipeline_config: Pipeline configuration for mode.
    """
    # batching controls
    ag_bs_tag = context.run.tags.get("agreement_batch_size")
    run_scope_tag = context.run.tags.get("run_scope")
    agreement_batch_size: int = int(ag_bs_tag) if ag_bs_tag else pipeline_config.agreement_batch_size
    is_batched: bool = (
        run_scope_tag == "batched"
        if run_scope_tag is not None
        else pipeline_config.is_batched()
    )

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

    ran_batches = 0
    while True:
        with engine.begin() as conn:
            last_uuid = ''
            
            # Fetch batch of agreements where all body pages have been tagged and XML not yet created
            agreement_uuids = (
                conn.execute(
                    text(
                        """
                    SELECT
                        a.agreement_uuid
                    FROM
                        pdx.agreements a
                    JOIN pdx.pages p
                        ON p.agreement_uuid = a.agreement_uuid
                    LEFT JOIN pdx.tagged_outputs t
                        ON t.page_uuid = p.page_uuid
                    WHERE a.agreement_uuid > :last_uuid
                      AND NOT EXISTS (
                        SELECT 1 FROM pdx.xml x WHERE x.agreement_uuid = a.agreement_uuid
                      )
                    GROUP BY a.agreement_uuid
                    HAVING
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
                    p.source_page_type,
                    coalesce(tgo.tagged_text_corrected, tgo.tagged_text, p.processed_page_content) as tagged_output,
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
        ran_batches += 1
        if is_batched:
            break
