from sqlalchemy.engine import Connection
from sqlalchemy import bindparam, text
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
    exhibit_type: str | None  # "2", "10", or "99"
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
    review_flag: bool | None
    validation_priority: float | None


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


def upsert_agreements(
    staged_agreements: Sequence[AgreementRow],
    schema: str,
    conn: Connection,
) -> None:
    agreements_table = f"{schema}.agreements"
    insert_sql = text(
        f"""
            INSERT INTO {agreements_table} (
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
        """
    )
    update_sql = text(
        f"""
        UPDATE {agreements_table}
        SET
            url = COALESCE(:url, url),
            filing_date = COALESCE(:filing_date, filing_date),
            target = COALESCE(:target, target),
            acquirer = COALESCE(:acquirer, acquirer),
            announce_date = COALESCE(:announce_date, announce_date),
            prob_filing = COALESCE(:prob_filing, prob_filing),
            filing_company_name = COALESCE(:filing_company_name, filing_company_name),
            filing_company_cik = COALESCE(:filing_company_cik, filing_company_cik),
            form_type = COALESCE(:form_type, form_type),
            exhibit_type = COALESCE(:exhibit_type, exhibit_type),
            secondary_filing_url = COALESCE(:secondary_filing_url, secondary_filing_url),
            source = COALESCE(:source, source)
        WHERE agreement_uuid = :agreement_uuid
          AND (
            NOT (url <=> COALESCE(:url, url))
            OR NOT (filing_date <=> COALESCE(:filing_date, filing_date))
            OR NOT (target <=> COALESCE(:target, target))
            OR NOT (acquirer <=> COALESCE(:acquirer, acquirer))
            OR NOT (announce_date <=> COALESCE(:announce_date, announce_date))
            OR NOT (prob_filing <=> COALESCE(:prob_filing, prob_filing))
            OR NOT (filing_company_name <=> COALESCE(:filing_company_name, filing_company_name))
            OR NOT (filing_company_cik <=> COALESCE(:filing_company_cik, filing_company_cik))
            OR NOT (form_type <=> COALESCE(:form_type, form_type))
            OR NOT (exhibit_type <=> COALESCE(:exhibit_type, exhibit_type))
            OR NOT (secondary_filing_url <=> COALESCE(:secondary_filing_url, secondary_filing_url))
            OR NOT (source <=> COALESCE(:source, source))
          )
        """
    )
    select_existing_sql = text(
        f"""
        SELECT agreement_uuid
        FROM {agreements_table}
        WHERE agreement_uuid IN :uuids
        """
    ).bindparams(bindparam("uuids", expanding=True))

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
        if not batch:
            continue
        deduped_batch_by_uuid: dict[object, dict[str, object]] = {}
        for row in batch:
            deduped_batch_by_uuid[row["agreement_uuid"]] = row
        batch = list(deduped_batch_by_uuid.values())
        uuids = [row["agreement_uuid"] for row in batch]
        existing_uuids = set(
            conn.execute(select_existing_sql, {"uuids": uuids}).scalars().all()
        )
        rows_insert = [row for row in batch if row["agreement_uuid"] not in existing_uuids]
        rows_update = [row for row in batch if row["agreement_uuid"] in existing_uuids]
        if rows_insert:
            _ = conn.execute(insert_sql, rows_insert)
        if rows_update:
            _ = conn.execute(update_sql, rows_update)


def upsert_pages(
    staged_pages: Sequence[PageRow],
    operation_type: str,
    schema: str,
    conn: Connection,
) -> None:
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
        "review_flag",
        "validation_priority",
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
        "review_flag",
        "validation_priority",
    ]

    pages_table = f"{schema}.pages"

    if operation_type == "insert":
        operation_cols = insert_cols

        cols = ",\n    ".join(insert_cols)
        value_placeholders = ",\n    ".join(f":{c}" for c in insert_cols)
        insert_sql = text(
            f"""
        INSERT INTO {pages_table} (
            {cols}
        ) VALUES (
            {value_placeholders}
        )
        """
        )
        update_sql = text(
            f"""
            UPDATE {pages_table}
            SET
                raw_page_content = :raw_page_content,
                processed_page_content = :processed_page_content,
                source_is_txt = :source_is_txt,
                source_is_html = :source_is_html,
                source_page_type = :source_page_type,
                page_type_prob_front_matter = :page_type_prob_front_matter,
                page_type_prob_toc = :page_type_prob_toc,
                page_type_prob_body = :page_type_prob_body,
                page_type_prob_sig = :page_type_prob_sig,
                page_type_prob_back_matter = :page_type_prob_back_matter,
                postprocess_modified = :postprocess_modified,
                review_flag = :review_flag,
                validation_priority = :validation_priority
            WHERE agreement_uuid = :agreement_uuid
              AND page_order = :page_order
              AND (
                NOT (raw_page_content <=> :raw_page_content)
                OR NOT (processed_page_content <=> :processed_page_content)
                OR NOT (source_is_txt <=> :source_is_txt)
                OR NOT (source_is_html <=> :source_is_html)
                OR NOT (source_page_type <=> :source_page_type)
                OR NOT (page_type_prob_front_matter <=> :page_type_prob_front_matter)
                OR NOT (page_type_prob_toc <=> :page_type_prob_toc)
                OR NOT (page_type_prob_body <=> :page_type_prob_body)
                OR NOT (page_type_prob_sig <=> :page_type_prob_sig)
                OR NOT (page_type_prob_back_matter <=> :page_type_prob_back_matter)
                OR NOT (postprocess_modified <=> :postprocess_modified)
                OR NOT (review_flag <=> :review_flag)
                OR NOT (validation_priority <=> :validation_priority)
              )
            """
        )
        select_existing_sql = text(
            f"""
            SELECT agreement_uuid, page_order
            FROM {pages_table}
            WHERE agreement_uuid IN :agreement_uuids
            """
        ).bindparams(bindparam("agreement_uuids", expanding=True))

        rows: list[dict[str, object]] = [
            {col: getattr(page, col) for col in operation_cols} for page in staged_pages
        ]
        for i in range(0, len(rows), 250):
            batch = rows[i : i + 250]
            if not batch:
                continue
            deduped_batch_by_key: dict[tuple[object, object], dict[str, object]] = {}
            for row in batch:
                deduped_batch_by_key[(row["agreement_uuid"], row["page_order"])] = row
            batch = list(deduped_batch_by_key.values())
            agreement_uuids = sorted(
                {
                    str(row["agreement_uuid"])
                    for row in batch
                    if row["agreement_uuid"] is not None
                }
            )
            if not agreement_uuids:
                continue
            existing_rows = conn.execute(
                select_existing_sql,
                {"agreement_uuids": agreement_uuids},
            ).mappings().fetchall()
            existing_keys = {
                (row["agreement_uuid"], row["page_order"]) for row in existing_rows
            }
            rows_insert = [
                row
                for row in batch
                if (row["agreement_uuid"], row["page_order"]) not in existing_keys
            ]
            rows_update = [
                row
                for row in batch
                if (row["agreement_uuid"], row["page_order"]) in existing_keys
            ]
            if rows_insert:
                _ = conn.execute(insert_sql, rows_insert)
            if rows_update:
                _ = conn.execute(update_sql, rows_update)
        return

    elif operation_type == "update":
        operation_cols = update_cols

        # build a SET clause for all cols except the pk
        set_clause = ",\n    ".join(
            f"{c} = :{c}" for c in update_cols if c != "page_uuid"
        )
        where_changed = " OR ".join(
            f"NOT ({c} <=> :{c})" for c in update_cols if c != "page_uuid"
        )

        update_sql = text(
            f"""
        UPDATE {pages_table}
        SET
            {set_clause}
        WHERE page_uuid = :page_uuid
          AND ({where_changed})
        """
        )
        rows = [
            {col: getattr(page, col) for col in operation_cols} for page in staged_pages
        ]
        for i in range(0, len(rows), 250):
            batch = rows[i : i + 250]
            if not batch:
                continue
            _ = conn.execute(update_sql, batch)
        return

    raise RuntimeError(
        f"Unknown value provided for 'operation_type': {operation_type}"
    )


def upsert_tags(
    staged_tags: Sequence[TagRow],
    schema: str,
    conn: Connection,
) -> None:
    """
    Upserts a batch of TagData objects into the pdx.tagged_outputs table.
    Args:
        staged_tags (Sequence): List of TagData or dicts with keys
            page_uuid, tagged_text, low_count, spans, tokens.
        conn (Connection): SQLAlchemy connection.
    """
    tagged_outputs_table = f"{schema}.tagged_outputs"
    insert_sql_tags = text(
        f"""
        INSERT INTO {tagged_outputs_table} (
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
        """
    )
    update_sql_tags = text(
        f"""
        UPDATE {tagged_outputs_table}
        SET
            tagged_text = :tagged_text,
            low_count = :low_count,
            spans = :spans,
            tokens = :tokens
        WHERE page_uuid = :page_uuid
          AND (
            NOT (tagged_text <=> :tagged_text)
            OR NOT (low_count <=> :low_count)
            OR NOT (spans <=> :spans)
            OR NOT (tokens <=> :tokens)
          )
        """
    )
    select_existing_sql = text(
        f"""
        SELECT page_uuid
        FROM {tagged_outputs_table}
        WHERE page_uuid IN :pids
        """
    ).bindparams(bindparam("pids", expanding=True))

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
        if not batch_tags:
            continue
        deduped_batch_by_pid: dict[object, dict[str, object]] = {}
        for row in batch_tags:
            deduped_batch_by_pid[row["page_uuid"]] = row
        batch_tags = list(deduped_batch_by_pid.values())
        page_uuids = [str(row["page_uuid"]) for row in batch_tags]
        existing_page_uuids = set(
            conn.execute(select_existing_sql, {"pids": page_uuids}).scalars().all()
        )
        rows_insert = [
            row for row in batch_tags if row["page_uuid"] not in existing_page_uuids
        ]
        rows_update = [
            row for row in batch_tags if row["page_uuid"] in existing_page_uuids
        ]
        if rows_insert:
            _ = conn.execute(insert_sql_tags, rows_insert)
        if rows_update:
            _ = conn.execute(update_sql_tags, rows_update)


def upsert_xml(
    staged_xml: Sequence[XmlRow],
    schema: str,
    conn: Connection,
) -> None:
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
    xml_table = f"{schema}.xml"
    upsert_sql_xml = text(
        f"""
        INSERT INTO {xml_table} (
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


def upsert_sections(
    staged_sections: Sequence[Mapping[str, object]],
    schema: str,
    conn: Connection,
) -> None:
    """
    Upsert section rows into pdx.sections.

    Each item in staged_sections must be a dict with keys:
      agreement_uuid, section_uuid, article_title, article_title_normed,
      section_title, section_title_normed, xml_content, xml_version
    """
    sections_table = f"{schema}.sections"
    upsert_sql = text(
        f"""
        INSERT INTO {sections_table} (
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
