from etl.defs.resources import DBResource
from sqlalchemy import text
from etl.domain.xml import generate_xml
import dagster as dg
from etl.defs.tagging_asset import tagging_asset
from etl.utils.db_utils import upsert_xml
import pandas as pd


@dg.asset(deps=[tagging_asset])
def xml_asset(db: DBResource):
    """
    Pulls tagged sections and assembles them into XML.
    Args:
        db (DBResource): Database resource for connection.
    Returns:
        None
    """
    agreement_batch_size: int = 10
    engine = db.get_engine()

    with engine.begin() as conn:
        while True:
            # fetch batch of staged (not tagged) pages
            agreement_uuids = (
                conn.execute(
                    text(
                        """
                    SELECT agreement_uuid
                    FROM pdx.agreements
                    WHERE processed = 0
                    ORDER BY agreement_uuid
                    LIMIT :limit
                """
                    ),
                    {"limit": agreement_batch_size},
                )
                .scalars()
                .all()
            )

            # if none left, we’re done
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
            except Exception as e:
                print(f"Error upserting tags: {e}")
                raise RuntimeError(e)
