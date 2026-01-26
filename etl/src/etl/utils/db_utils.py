from sqlalchemy.engine import Connection
from sqlalchemy import text
from collections.abc import Sequence, Mapping
from typing import Protocol
import json


class AgreementRow(Protocol):
    agreement_uuid: str
    url: str
    filing_date: object | None
    # Optional fields (DMA corpus flow)
    target: str | None
    acquirer: str | None
    announce_date: object | None  # Deal announcement date
    # Optional fields (SEC index flow)
    prob_filing: float | None
    filing_company_name: str | None
    filing_company_cik: str | None
    form_type: str | None
    exhibit_type: str | None  # "2" or "10"
    secondary_filing_url: str | None  # URL of duplicate filing if detected


class PageRow(Protocol):
    agreement_uuid: str | None
    page_uuid: str | None
    page_order: int | None
    raw_page_content: str | None
    processed_page_content: str | None
    source_is_txt: bool | None
    source_is_html: bool | None
    source_page_type: str | None
    page_type_prob_front_matter: float | None
    page_type_prob_toc: float | None
    page_type_prob_body: float | None
    page_type_prob_sig: float | None
    page_type_prob_back_matter: float | None
    postprocess_modified: bool | None


class TagRow(Protocol):
    page_uuid: str
    tagged_text: str
    low_count: int
    spans: list[dict[str, object]]
    tokens: list[dict[str, object]]


class XmlRow(Protocol):
    agreement_uuid: str
    xml: str
    version: int


def upsert_agreements(staged_agreements: Sequence[AgreementRow], conn: Connection) -> None:
    upsert_sql = text(
        """
            INSERT INTO pdx.agreements (
              agreement_uuid,
              url,
              filing_date,
              target,
              acquirer,
              announce_date,
              prob_filing,
              filing_company_name,
              filing_company_cik,
              form_type,
              exhibit_type,
              secondary_filing_url,
              source,
              ingested_date
            ) VALUES (
              :agreement_uuid,
              :url,
              :filing_date,
              :target,
              :acquirer,
              :announce_date,
              :prob_filing,
              :filing_company_name,
              :filing_company_cik,
              :form_type,
              :exhibit_type,
              :secondary_filing_url,
              :source,
              DEFAULT
            )
            ON DUPLICATE KEY UPDATE
              url                      = COALESCE(VALUES(url), url),
              filing_date              = COALESCE(VALUES(filing_date), filing_date),
              target                   = COALESCE(VALUES(target), target),
              acquirer                 = COALESCE(VALUES(acquirer), acquirer),
              announce_date            = COALESCE(VALUES(announce_date), announce_date),
              prob_filing              = COALESCE(VALUES(prob_filing), prob_filing),
              filing_company_name      = COALESCE(VALUES(filing_company_name), filing_company_name),
              filing_company_cik       = COALESCE(VALUES(filing_company_cik), filing_company_cik),
              form_type                = COALESCE(VALUES(form_type), form_type),
              exhibit_type             = COALESCE(VALUES(exhibit_type), exhibit_type),
              secondary_filing_url     = COALESCE(VALUES(secondary_filing_url), secondary_filing_url),
              source                   = COALESCE(VALUES(source), source)
        """
    )

    count = len(staged_agreements)
    # HARDCODED: source value is set to "edgar" for all agreements during staging
    # To change this value, update the "source" field below
    rows: list[dict[str, object]] = [
        {
            "agreement_uuid": f.agreement_uuid,
            "url": f.url,
            "filing_date": f.filing_date,
            "target": f.target,
            "acquirer": f.acquirer,
            "announce_date": f.announce_date,
            "prob_filing": f.prob_filing,
            "filing_company_name": f.filing_company_name,
            "filing_company_cik": f.filing_company_cik,
            "form_type": f.form_type,
            "exhibit_type": f.exhibit_type,
            "secondary_filing_url": f.secondary_filing_url,
            "source": "edgar",  # HARDCODED: Change this value to update the source
        }
        for f in staged_agreements
    ]

    # execute in batches of 250
    for i in range(0, count, 250):
        batch = rows[i : i + 250]
        _ = conn.execute(upsert_sql, batch)


def upsert_pages(staged_pages: Sequence[PageRow], operation_type: str, conn: Connection) -> None:
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
        update_clause = ",\n    ".join(f"{c}=VALUES({c})" for c in insert_cols)

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

    rows: list[dict[str, object]] = [
        {col: getattr(page, col) for col in operation_cols} for page in staged_pages
    ]

    # execute in batches of 250
    for i in range(0, len(rows), 250):
        batch = rows[i : i + 250]
        _ = conn.execute(upsert_sql, batch)


def upsert_tags(staged_tags: Sequence[TagRow], conn: Connection) -> None:
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
            tokens,
            created_date
        ) VALUES (
            :page_uuid,
            :tagged_text,
            :low_count,
            :spans,
            :tokens,
            DEFAULT
        )
        ON DUPLICATE KEY UPDATE
            tagged_text   = VALUES(tagged_text),
            low_count     = VALUES(low_count),
            spans         = VALUES(spans),
            tokens        = VALUES(tokens)
        """
    )

    rows_tags: list[dict[str, object]] = [
        {
            "page_uuid": tag.page_uuid,
            "tagged_text": tag.tagged_text,
            "low_count": tag.low_count,
            "spans": json.dumps(tag.spans),
            "tokens": json.dumps(tag.tokens),
        }
        for tag in staged_tags
    ]

    # execute in batches of 250
    for i in range(0, len(rows_tags), 250):
        batch_tags = rows_tags[i : i + 250]
        _ = conn.execute(upsert_sql_tags, batch_tags)


def upsert_xml(staged_xml: Sequence[XmlRow], conn: Connection) -> None:
    """
    Upserts a batch of XML objects into the pdx.xml table.
    
    Uses the version provided in staged_xml (no auto-increment).
    Only the xml_asset should create new versions; taxonomy and other
    downstream assets preserve the version when updating.
    created_date is set automatically by the database (UTC_TIMESTAMP()).
    
    Args:
        staged_xml (Sequence): List of XML objects with keys
            agreement_uuid, xml, version
        conn (Connection): SQLAlchemy connection.
    """
    upsert_sql_xml = text(
        """
        INSERT INTO pdx.xml (
            agreement_uuid,
            xml,
            version
        ) VALUES (
            :agreement_uuid,
            :xml,
            :version
        )
        ON DUPLICATE KEY UPDATE
            xml = VALUES(xml)
        """
    )

    rows_xmls: list[dict[str, object]] = []
    for xml in staged_xml:
        rows_xmls.append(
            {
                "agreement_uuid": xml.agreement_uuid,
                "xml": xml.xml,
                "version": xml.version,
            }
        )

    for i in range(0, len(rows_xmls), 250):
        batch_tags = rows_xmls[i : i + 250]
        _ = conn.execute(upsert_sql_xml, batch_tags)


def upsert_sections(staged_sections: Sequence[Mapping[str, object]], conn: Connection) -> None:
    """
    Upsert section rows into pdx.sections.

    Each item in staged_sections must be a dict with keys:
      agreement_uuid, section_uuid, article_title, article_title_normed,
      section_title, section_title_normed, xml_content, xml_version
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
            xml_content,
            xml_version
        ) VALUES (
            :agreement_uuid,
            :section_uuid,
            :article_title,
            :article_title_normed,
            :article_order,
            :section_title,
            :section_title_normed,
            :section_order,
            :xml_content,
            :xml_version
        )
        ON DUPLICATE KEY UPDATE
            article_title = VALUES(article_title),
            article_title_normed = VALUES(article_title_normed),
            article_order = VALUES(article_order),
            section_title = VALUES(section_title),
            section_title_normed = VALUES(section_title_normed),
            section_order = VALUES(section_order),
            xml_content = VALUES(xml_content),
            xml_version = VALUES(xml_version)
        """
    )

    rows: list[Mapping[str, object]] = list(staged_sections)
    for i in range(0, len(rows), 250):
        batch = rows[i : i + 250]
        _ = conn.execute(upsert_sql, batch)
