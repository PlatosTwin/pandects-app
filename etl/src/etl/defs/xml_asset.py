from etl.defs.resources import DBResource, PipelineConfig
from sqlalchemy import text
from etl.domain.xml import generate_xml
import dagster as dg
from etl.defs.tagging_asset import tagging_asset
from etl.utils.db_utils import upsert_xml
import pandas as pd


@dg.asset(deps=[tagging_asset])
def xml_asset(context, db: DBResource, pipeline_config: PipelineConfig):
    """
    Pulls tagged sections and assembles them into XML.

    In cleanup mode, processes only existing unprocessed agreements.

    Args:
        context (dg.AssetExecutionContext): Dagster context.
        db (DBResource): Database resource for connection.
        pipeline_config (PipelineConfig): Pipeline configuration for mode.
    Returns:
        None
    """
    agreement_batch_size: int = 10
    engine = db.get_engine()
    is_cleanup = pipeline_config.is_cleanup_mode()

    # Check if we're in a job context and get the mode from there
    if hasattr(context, "job_def") and hasattr(context.job_def, "config"):
        job_config = context.job_def.config
        if hasattr(job_config, "mode"):
            is_cleanup = job_config.mode.value == "cleanup"

    context.log.info(
        f"Running XML generation in {'CLEANUP' if is_cleanup else 'FROM_SCRATCH'} mode"
    )

    with engine.begin() as conn:
        while True:
            # fetch batch of staged (not tagged) pages
            # provided that all pages in the agreement have been tagged
            # TODO: alerting if there are partially tagged agreements
            agreement_uuids = (
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
                    GROUP BY
                        a.agreement_uuid
                    HAVING
                        MIN(p.processed) = 1
                    ORDER BY
                        a.agreement_uuid
                    LIMIT
                        :limit;
                """
                    ),
                    {"limit": agreement_batch_size},
                )
                .scalars()
                .all()
            )

            # if none left, we're done
            if not agreement_uuids:
                break

            # 2) fetch every page (and its tagged_output) for those agreements
            rows = (
                conn.execute(
                    text(
                        """
                    SELECT
                    p.agreement_uuid,
                    p.page_uuid,
                    tgo.tagged_output
                    FROM pdx.pages p
                    LEFT JOIN pdx.tagged_output tgo
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
            # tag pages
            xml = generate_xml(df)

            try:
                upsert_xml(xml, conn)
                context.log.info(
                    f"Successfully generated XML for {len(agreement_uuids)} agreements"
                )
            except Exception as e:
                context.log.error(f"Error upserting XML: {e}")
                raise RuntimeError(e)
