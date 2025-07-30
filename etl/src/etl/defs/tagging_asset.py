from etl.defs.resources import DBResource, TaggingModel
from sqlalchemy import text
from etl.domain.tagging import tag
import dagster as dg
from etl.defs.staging_asset import staging_asset
from etl.utils.db_utils import upsert_tags


@dg.asset(deps=[staging_asset])
def tagging_asset(db: DBResource, tagging_model: TaggingModel):
    """
    Pulls staged pages and runs them through the tagging model.
    Args:
        db (DBResource): Database resource for connection.
        tagging_model (TaggingModel): Model for page tagging.
    Returns:
        None
    """
    tagging_model = tagging_model.model()
    
    batch_size: int = 250
    last_uuid: str = ""
    engine = db.get_engine()

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
            tagged_pages = tag(rows, tagging_model)

            try:
                upsert_tags(tagged_pages, conn)
            except Exception as e:
                print(f"Error upserting tags: {e}")
                raise RuntimeError(e)

            last_uuid = rows[-1]['page_uuid']
