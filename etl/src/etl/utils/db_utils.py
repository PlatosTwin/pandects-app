from sqlalchemy.engine import Connection
from sqlalchemy import text
from typing import Sequence

def upsert_pages(staged_pages: Sequence, conn: Connection) -> None:
    """
    Upserts a batch of PageMetadata objects into the pdx.pages table.
    Args:
        staged_pages (Sequence): List of PageMetadata or dicts.
        conn (Connection): SQLAlchemy connection.
    """
    rows = []
    for page in staged_pages:
        rows.append({
            "agreement_uuid": getattr(page, "agreement_uuid", page["agreement_uuid"]),
            "page_uuid": getattr(page, "page_uuid", page["page_uuid"]),
            "page_order": getattr(page, "page_order", page["page_order"]),
            "raw_page_content": getattr(page, "raw_page_content", page["raw_page_content"]),
            "processed_page_content": getattr(page, "processed_page_content", page["processed_page_content"]),
            "source_is_txt": getattr(page, "source_is_txt", page["source_is_txt"]),
            "source_is_html": getattr(page, "source_is_html", page["source_is_html"]),
            "source_page_type": getattr(page, "source_page_type", page["source_page_type"]),
            "page_type_prob_front_matter": getattr(page, "page_type_prob_front_matter", page["page_type_prob_front_matter"]),
            "page_type_prob_toc": getattr(page, "page_type_prob_toc", page["page_type_prob_toc"]),
            "page_type_prob_body": getattr(page, "page_type_prob_body", page["page_type_prob_body"]),
        })
    upsert_sql = text("""
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
            page_type_prob_body
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
            :page_type_prob_body
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
            page_type_prob_body         = VALUES(page_type_prob_body)
    """)
    # execute in batches of 250
    for i in range(0, len(rows), 250):
        batch = rows[i : i + 250]
        conn.execute(upsert_sql, batch)
