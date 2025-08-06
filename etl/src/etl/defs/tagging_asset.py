from etl.defs.resources import DBResource, TaggingModel, PipelineConfig
from sqlalchemy import text
from etl.domain.tagging import tag
import dagster as dg
from etl.defs.staging_asset import staging_asset
from etl.utils.db_utils import upsert_tags


@dg.asset(deps=[staging_asset])
def tagging_asset(
    context,
    db: DBResource,
    tagging_model: TaggingModel,
    pipeline_config: PipelineConfig,
):
    """
    Pulls staged pages and runs them through the tagging model.

    In cleanup mode, processes only existing unprocessed pages.

    Args:
        context (dg.AssetExecutionContext): Dagster context.
        db (DBResource): Database resource for connection.
        tagging_model (TaggingModel): Model for page tagging.
        pipeline_config (PipelineConfig): Pipeline configuration for mode.
    Returns:
        None
    """
    inference_model = tagging_model.model()

    batch_size: int = 250
    last_uuid: str = ""
    engine = db.get_engine()
    is_cleanup = pipeline_config.is_cleanup_mode()

    # Check if we're in a job context and get the mode from there
    if hasattr(context, "job_def") and hasattr(context.job_def, "config"):
        job_config = context.job_def.config
        if hasattr(job_config, "mode"):
            is_cleanup = job_config.mode.value == "cleanup"

    context.log.info(
        f"Running tagging in {'CLEANUP' if is_cleanup else 'FROM_SCRATCH'} mode"
    )

    with engine.begin() as conn:
        while True:
            # fetch batch of staged (not tagged) pages
            result = conn.execute(
                text(
                    """
                    SELECT page_uuid, processed_page_content
                    FROM pdx.pages
                    WHERE page_uuid > :last_uuid
                    AND processed = 0
                    AND source_page_type = 'body'
                    ORDER BY page_uuid ASC
                    LIMIT :batch_size
                """
                ),
                {"last_uuid": last_uuid, "batch_size": batch_size},
            )
            rows = result.mappings().fetchall()

            if not rows:
                break

            # tag pages
            tagged_pages = tag(rows, inference_model)

            try:
                upsert_tags(tagged_pages, conn)
                context.log.info(f"Successfully tagged {len(tagged_pages)} pages")
            except Exception as e:
                context.log.error(f"Error upserting tags: {e}")
                raise RuntimeError(e)

            last_uuid = rows[-1]["page_uuid"]
