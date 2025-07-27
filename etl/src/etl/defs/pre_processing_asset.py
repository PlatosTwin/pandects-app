from etl.defs.resources import DBResource, ClassifierModel
from sqlalchemy import text
from etl.domain.pre_processing import pre_process
from datetime import datetime
import dagster as dg
from etl.defs.staging_asset import staging_asset


def upsert_pages(staged_pages, conn):
    rows = []
    count = len(staged_pages)
    for page in staged_pages:

        rows.append(
            {
                "agreement_uuid": page.agreement_uuid,
                "page_uuid": page.page_uuid,
                "page_order": page.page_order,
                "raw_page_content": page.raw_page_content,
                "processed_page_content": page.processed_page_content,
                "source_is_txt": page.source_is_txt,
                "source_is_html": page.source_is_html,
                "source_page_type": page.source_page_type,
                "page_type_prob_front_matter": page.page_type_prob_front_matter,
                "page_type_prob_toc": page.page_type_prob_toc,
                "page_type_prob_body": page.page_type_prob_body,
            }
        )

    upsert_sql = text(
        """
        INSERT INTO pdx.pages (
            agreement_uuid,
            page_uuid,
            page_order,
            raw_page_content,
            processed_page_content,
            source_is_txt,
            source_is_html,
            source_page_type,
            page_type_prob_front_matter,
            page_type_prob_toc,
            page_type_prob_body,
        ) VALUES (
            :agreement_uuid,
            :page_uuid,
            :page_order,
            :raw_page_content,
            :processed_page_content,
            :source_is_txt,
            :source_is_html,
            :source_page_type,
            :page_type_prob_front_matter,
            :page_type_prob_toc,
            :page_type_prob_body,
        )
        ON DUPLICATE KEY UPDATE
            agreement_uuid              = VALUES(agreement_uuid),
            page_uuid                   = VALUES(page_uuid),
            page_order                  = VALUES(page_order),
            raw_page_content            = VALUES(raw_page_content),
            processed_page_content      = VALUES(processed_page_content),
            source_is_txt               = VALUES(source_is_txt),
            source_is_html              = VALUES(source_is_html),
            source_page_type            = VALUES(source_page_type),
            page_type_prob_front_matter = VALUES(page_type_prob_front_matter),
            page_type_prob_toc          = VALUES(page_type_prob_toc),
            page_type_prob_body         = VALUES(page_type_prob_body),
    """
    )

    # execute in batches of 250
    for i in range(0, count, 250):
        batch = rows[i : i + 250]
        conn.execute(upsert_sql, batch)


@dg.asset(deps=[staging_asset])
def pre_processing_asset(db: DBResource, classifier_model: ClassifierModel):
    """
    Pulls staged agreements, splits agreements into pages, classifies page type,
    and processes HTML into formatted text, in preparation for LLM tagging in next stage.
    """
    batch_size = 15
    last_uuid = ""

    engine = db.get_engine()

    with engine.begin() as conn:
        while True:
            # fetch batch of staged (not processed) agreements
            result = conn.execute(
                f"""
                    select
                        agreement_uuid,
                        url
                    from
                        pdx.agreements
                    where
                        agreement_uuid > '{last_uuid}'
                        and not processed
                    order by
                        agreement_uuid asc
                    limit
                        {batch_size}
                    """
            )
            rows = result.fetchall()
            if not rows:
                break

            # process pages from batch
            staged_pages = pre_process(rows, classifier_model)

            # upsert pages from batch
            upsert_pages(staged_pages, conn)

            last_uuid = rows[-1]["agreement_uuid"]
