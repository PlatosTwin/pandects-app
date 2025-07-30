from etl.defs.resources import DBResource, ClassifierModel
from sqlalchemy import text
from etl.domain.pre_processing import pre_process
import dagster as dg
from etl.defs.staging_asset import staging_asset
from etl.utils.db_utils import upsert_pages


@dg.asset(deps=[staging_asset])
def pre_processing_asset(db: DBResource, classifier_model: ClassifierModel) -> None:
    """
    Pulls staged agreements, splits agreements into pages, classifies page type,
    and processes HTML into formatted text, in preparation for LLM tagging in next stage.
    Args:
        db (DBResource): Database resource for connection.
        classifier_model (ClassifierModel): Model for page classification.
    Returns:
        None
    """
    batch_size: int = 15  # batch_size agreements at a time
    last_uuid: str = ""
    engine = db.get_engine()

    with engine.begin() as conn:
        while True:
            # fetch batch of staged (not processed) agreements
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
            staged_pages = pre_process(agreements, classifier_model)
            
            if staged_pages:
                try:
                    upsert_pages(staged_pages, conn)
                except Exception as e:
                    print(f"Error upserting pages: {e}")
                    raise RuntimeError(e)

            last_uuid = rows[-1][0]
