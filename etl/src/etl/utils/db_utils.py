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
              filing_date
            ) VALUES (
              :agreement_uuid,
              :url,
              :target,
              :acquirer,
              :filing_date
            )
            ON DUPLICATE KEY UPDATE
              url                      = VALUES(url),
              target                   = VALUES(target),
              acquirer                 = VALUES(acquirer),
              filing_date              = VALUES(filing_date)
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
                "filing_date": f.filing_date
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
        "postprocess_modified",
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
        "postprocess_modified",
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
            page_uuid, tagged_text, low_count, spans, tokens.
        conn (Connection): SQLAlchemy connection.
    """
    upsert_sql_tags = text(
        """
        INSERT INTO pdx.tagged_outputs (
            page_uuid,
            tagged_text,
            low_count,
            spans,
            tokens
        ) VALUES (
            :page_uuid,
            :tagged_text,
            :low_count,
            :spans,
            :tokens
        )
        ON DUPLICATE KEY UPDATE
            tagged_text = VALUES(tagged_text),
            low_count   = VALUES(low_count),
            spans       = VALUES(spans),
            tokens       = VALUES(tokens)
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
                "tokens": json.dumps(tag.tokens),
            }
        )

        rows_pages.append({"page_uuid": tag.page_uuid})

    # execute in batches of 250
    for i in range(0, len(rows_tags), 250):
        batch_tags = rows_tags[i : i + 250]
        conn.execute(upsert_sql_tags, batch_tags)


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
        """
    )

    rows_xmls = []
    for xml in staged_xml:
        rows_xmls.append(
            {
                "agreement_uuid": xml.agreement_uuid,
                "xml": xml.xml,
            }
        )

        # agreements list no longer needed for processed flag updates

    upsert_sql_xml = text("CALL pdx.upsert_xml(:uuid, :xml)")
    # No processed flag writes; idempotency achieved via existence checks downstream

    # rows_xmls: either list[dict] with keys {"agreement_uuid","xml"} or list[tuple]
    # rows_agreements: either list[dict] with key {"agreement_uuid"} or list[str]/list[tuple]

    for i in range(0, len(rows_xmls), 250):
        batch_tags = rows_xmls[i : i + 250]
        # batch_agreements removed; we do not update agreements.processed

        # normalize
        tag_params = (
            [{"uuid": r["agreement_uuid"], "xml": r["xml"]} for r in batch_tags]
            if isinstance(batch_tags[0], dict)
            else [{"uuid": u, "xml": x} for (u, x) in batch_tags]
        )
        # agr_params removed

        # one transaction per outer context; no begin_nested()
        for p in tag_params:
            # CALL per row; no result sets are produced by the proc, so just close
            res = conn.execute(upsert_sql_xml, p)
            res.close()

        # no agreement processed flag updates


def upsert_sections(staged_sections: Sequence, conn: Connection) -> None:
    """
    Upsert section rows into pdx.sections.

    Each item in staged_sections must be a dict with keys:
      agreement_uuid, section_uuid, article_title, article_title_normed,
      section_title, section_title_normed, xml_content
    """
    upsert_sql = text(
        """
        INSERT INTO pdx.sections (
            agreement_uuid,
            section_uuid,
            article_title,
            article_title_normed,
            article_order,
            section_title,
            section_title_normed,
            section_order,
            xml_content
        ) VALUES (
            :agreement_uuid,
            :section_uuid,
            :article_title,
            :article_title_normed,
            :article_order,
            :section_title,
            :section_title_normed,
            :section_order,
            :xml_content
        )
        ON DUPLICATE KEY UPDATE
            article_title = VALUES(article_title),
            article_title_normed = VALUES(article_title_normed),
            article_order = VALUES(article_order),
            section_title = VALUES(section_title),
            section_title_normed = VALUES(section_title_normed),
            section_order = VALUES(section_order),
            xml_content = VALUES(xml_content)
        """
    )

    rows = list(staged_sections)
    for i in range(0, len(rows), 250):
        batch = rows[i : i + 250]
        conn.execute(upsert_sql, batch)
