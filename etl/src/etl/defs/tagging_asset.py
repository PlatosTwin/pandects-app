from etl.defs.resources import DBResource, TaggingModel
from sqlalchemy import text
from etl.domain.tagging import tag
import dagster as dg
from etl.defs.staging_asset import staging_asset


@dg.asset(deps=[staging_asset])
def tagging_asset(db: DBResource, tagging_model: TaggingModel):
    """ """
    pass
    # batch_size: int = 15
    # last_uuid: str = ""
    # engine = db.get_engine()

    # with engine.begin() as conn:
    #     while True:
    #         # fetch batch of staged (not processed) agreements
    #         result = conn.execute(
    #             text(
    #                 """
    #                 SELECT agreement_uuid, url
    #                 FROM pdx.agreements
    #                 WHERE agreement_uuid > :last_uuid
    #                 AND processed = 0
    #                 ORDER BY agreement_uuid ASC
    #                 LIMIT :batch_size
    #             """
    #             ),
    #             {"last_uuid": last_uuid, "batch_size": batch_size},
    #         )
    #         rows = result.fetchall()
    #         if not rows:
    #             break

    #         agreements = [{"agreement_uuid": r[0], "url": r[1]} for r in rows]
    #         staged_pages = pre_process(agreements, classifier_model)
    #         try:
    #             upsert_pages(staged_pages, conn)
    #         except Exception as e:
    #             print(f"Error upserting pages: {e}")
    #             break

    #         last_uuid = rows[-1][0]
