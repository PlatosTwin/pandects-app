from sqlalchemy.engine import Connection
from sqlalchemy import text
from typing import Sequence
import json


def upsert_agreements(staged_agreements: Sequence, conn: Connection) -> None:
    upsert_sql = text(
        """
            INSERT INTO pdx.agreements (
              agreement_uuid,
              url,
              target,
              acquirer,
              filing_date,
              transaction_price,
              transaction_type,
              transaction_consideration,
              target_type
            ) VALUES (
              :agreement_uuid,
              :url,
              :target,
              :acquirer,
              :filing_date,
              :transaction_price,
              :transaction_type,
              :transaction_consideration,
              :target_type
            )
            ON DUPLICATE KEY UPDATE
              url                      = VALUES(url),
              target                   = VALUES(target),
              acquirer                 = VALUES(acquirer),
              filing_date              = VALUES(filing_date),
              transaction_price        = VALUES(transaction_price),
              transaction_type         = VALUES(transaction_type),
              transaction_consideration = VALUES(transaction_consideration),
              target_type              = VALUES(target_type)
        """
    )

    count = len(staged_agreements)
    rows = []
    for f in staged_agreements:
        rows.append(
            {
                "agreement_uuid": f.agreement_uuid,
                "url": f.url,
                "target": f.target,
                "acquirer": f.acquirer,
                "filing_date": f.filing_date,
                "transaction_type": f.transaction_type,
                "transaction_price": f.transaction_price,
                "transaction_consideration": f.transaction_consideration,
                "target_type": f.target_type,
            }
        )

    # execute in batches of 250
    for i in range(0, count, 250):
        batch = rows[i : i + 250]
        conn.execute(upsert_sql, batch)


def upsert_pages(staged_pages: Sequence, operation_type: str, conn: Connection) -> None:
    """
    Upserts a batch of PageMetadata objects into the pdx.pages table.
    Args:
        staged_pages (Sequence): List of PageMetadata or dicts.
        conn (Connection): SQLAlchemy connection.
    """
    insert_cols = [
        "agreement_uuid",
        "page_order",
        "raw_page_content",
        "processed_page_content",
        "source_is_txt",
        "source_is_html",
        "source_page_type",
        "page_type_prob_front_matter",
        "page_type_prob_toc",
        "page_type_prob_body",
        "page_type_prob_sig",
        "page_type_prob_back_matter",
    ]

    update_cols = [
        "page_uuid",
        "processed_page_content",
        "source_page_type",
        "page_type_prob_front_matter",
        "page_type_prob_toc",
        "page_type_prob_body",
        "page_type_prob_sig",
        "page_type_prob_back_matter",
    ]

    if operation_type == "insert":
        operation_cols = insert_cols

        cols = ",\n    ".join(insert_cols)
        value_placeholders = ",\n    ".join(f":{c}" for c in insert_cols)
        update_clause = ",\n    ".join(
            f"{c}=VALUES({c})" for c in insert_cols if c != "page_uuid"
        )

        upsert_sql = text(
            f"""
        INSERT INTO pdx.pages (
            {cols}
        ) VALUES (
            {value_placeholders}
        )
        ON DUPLICATE KEY UPDATE
            {update_clause}
        """
        )

    elif operation_type == "update":
        operation_cols = update_cols

        # build a SET clause for all cols except the pk
        cols = ",\n    ".join(update_cols)
        set_clause = ",\n    ".join(
            f"{c} = :{c}" for c in update_cols if c != "page_uuid"
        )

        upsert_sql = text(
            f"""
        UPDATE pdx.pages
        SET
            {set_clause}
        WHERE page_uuid = :page_uuid
        """
        )

    else:
        raise RuntimeError(
            f"Unknown value provided for 'operation_type': {operation_type}"
        )

    rows = [
        {col: getattr(page, col) for col in operation_cols} for page in staged_pages
    ]

    # execute in batches of 250
    for i in range(0, len(rows), 250):
        batch = rows[i : i + 250]
        conn.execute(upsert_sql, batch)


def upsert_tags(staged_tags: Sequence, conn: Connection) -> None:
    """
    Upserts a batch of TagData objects into the pdx.tagged_outputs table.
    Args:
        staged_tags (Sequence): List of TagData or dicts with keys
            page_uuid, tagged_text, low_count, spans, chars.
        conn (Connection): SQLAlchemy connection.
    """
    upsert_sql_tags = text(
        """
        INSERT INTO pdx.tagged_outputs (
            page_uuid,
            tagged_text,
            low_count,
            spans,
            chars
        ) VALUES (
            :page_uuid,
            :tagged_text,
            :low_count,
            :spans,
            :chars
        )
        ON DUPLICATE KEY UPDATE
            tagged_text = VALUES(tagged_text),
            low_count   = VALUES(low_count),
            spans       = VALUES(spans),
            chars       = VALUES(chars)
        """
    )
    update_sql_pages = text(
        """
    UPDATE pdx.pages
       SET processed = 1
     WHERE page_uuid  = :page_uuid
    """
    )

    rows_tags = []
    rows_pages = []
    for tag in staged_tags:
        rows_tags.append(
            {
                "page_uuid": tag.page_uuid,
                "tagged_text": tag.tagged_text,
                "low_count": tag.low_count,
                "spans": json.dumps(tag.spans),
                "chars": json.dumps(tag.chars),
            }
        )

        rows_pages.append({"page_uuid": tag.page_uuid})

    # execute in batches of 250
    for i in range(0, len(rows_tags), 250):
        batch_tags = rows_tags[i : i + 250]
        conn.execute(upsert_sql_tags, batch_tags)

        # set pages as processed
        batch_pages = rows_pages[i : i + 250]
        conn.execute(update_sql_pages, batch_pages)


def upsert_xml(staged_xml: Sequence, conn: Connection) -> None:
    """
    Upserts a batch of XML objects into the pdx.xml table.
    Args:
        staged_xml (Sequence): List of XML or dicts with keys
            agreement_uuid, xml
        conn (Connection): SQLAlchemy connection.
    """
    upsert_sql_xml = text(
        """
        INSERT INTO pdx.xml (
            agreement_uuid,
            xml
        ) VALUES (
            :agreement_uuid,
            :xml
        )
        ON DUPLICATE KEY UPDATE
            xml = VALUES(xml)
        """
    )
    update_sql_agreements = text(
        """
    UPDATE pdx.agreements
       SET processed = 1
     WHERE agreement_uuid  = :agreement_uuid
    """
    )

    rows_xmls = []
    rows_agreements = []
    for xml in staged_xml:
        rows_xmls.append(
            {
                "agreement_uuid": xml.agreement_uuid,
                "xml": xml.xml,
            }
        )

        rows_agreements.append({"agreement_uuid": xml.agreement_uuid})

    # execute in batches of 250
    for i in range(0, len(rows_xmls), 250):
        batch_tags = rows_xmls[i : i + 250]
        conn.execute(upsert_sql_xml, batch_tags)

        # set agreements as processed
        batch_pages = rows_agreements[i : i + 250]
        conn.execute(update_sql_agreements, batch_pages)
