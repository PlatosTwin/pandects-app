from etl.defs.resources import DBResource, ClassifierModel, PipelineConfig
from sqlalchemy import text
from etl.domain.pre_processing import pre_process, cleanup
import dagster as dg
from etl.defs.staging_asset import staging_asset
from etl.utils.db_utils import upsert_pages


@dg.asset(deps=[staging_asset])
def pre_processing_asset(
    context,
    db: DBResource,
    classifier_model: ClassifierModel,
    pipeline_config: PipelineConfig,
) -> None:
    """
    Pulls staged agreements, splits agreements into pages, classifies page type,
    and processes HTML into formatted text, in preparation for LLM tagging in next stage.

    In cleanup mode, processes only existing unprocessed agreements.

    Args:
        context (dg.AssetExecutionContext): Dagster context.
        db (DBResource): Database resource for connection.
        classifier_model (ClassifierModel): Model for page classification.
        pipeline_config (PipelineConfig): Pipeline configuration for mode.
    Returns:
        None
    """
    last_uuid: str = ""
    engine = db.get_engine()
    is_cleanup = pipeline_config.is_cleanup_mode()

    # Check if we're in a job context and get the mode from there
    if hasattr(context, "job_def") and hasattr(context.job_def, "config"):
        job_config = context.job_def.config
        if hasattr(job_config, "mode"):
            is_cleanup = job_config.mode.value == "cleanup"

    if not is_cleanup:
        context.log.info(f"Running pre-processing in 'FROM_SCRATCH' mode")
        
        batch_size: int = 15  # batch_size agreements at a time

        while True:
            with engine.begin() as conn:
                # fetch batch of staged (not processed) AGREEMENTS
                result = conn.execute(
                    text(
                        """
                    SELECT agreement_uuid, url
                    FROM pdx.agreements
                    WHERE agreement_uuid > :last_uuid
                    AND processed = 0
                    ORDER BY agreement_uuid ASC
                    LIMIT :batch_size
                    """
                    ),
                    {"last_uuid": last_uuid, "batch_size": batch_size},
                )
                rows = result.fetchall()

                if not rows:
                    break

                # split agreements into pages, and process the pages
                agreements = [{"agreement_uuid": r[0], "url": r[1]} for r in rows]
                inference_model = classifier_model.model()
                staged_pages = pre_process(agreements, inference_model)

                if staged_pages:
                    try:
                        upsert_pages(staged_pages, operation_type="insert", conn=conn)
                        context.log.info(
                            f"Successfully PROCESSED ANEW {len(staged_pages)} pages from {len(agreements)} agreements"
                        )
                    except Exception as e:
                        context.log.error(f"Error upserting pages: {e}")
                        raise RuntimeError(e)

                last_uuid = rows[-1][0]
    else:
        context.log.info(f"Running pre-processing in 'CLEANUP' mode")
        batch_size: int = 5  # pages from batch_size agreements at a time

        while True:
            with engine.begin() as conn:
                # fetch batch of staged (not processed) PAGES
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
                        p.processed = 0
                        AND p.agreement_uuid in (SELECT agreement_uuid FROM agreement_batch)
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

                # split agreements into pages, and process the pages
                pages = [
                    {
                        "agreement_uuid": r[0],
                        "page_uuid": r[1],
                        "content": r[2],
                        "is_txt": r[3],
                        "is_html": r[4],
                    }
                    for r in rows
                ]
                inference_model = classifier_model.model()
                staged_pages = cleanup(pages, inference_model)

                if staged_pages:
                    try:
                        upsert_pages(staged_pages, operation_type="update", conn=conn)
                        context.log.info(
                            f"Successfully UPDATED {len(staged_pages)} pages from {len(set(p['agreement_uuid'] for p in pages))} agreements"
                        )
                    except Exception as e:
                        context.log.error(f"Error upserting pages: {e}")
                        raise RuntimeError(e)

                last_uuid = rows[-1][0]
