"""Assemble tagged sections into XML documents and verify XML tree structure."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false

import io
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple

import pandas as pd

import dagster as dg
from dagster import AssetExecutionContext
from openai import OpenAI
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection

from etl.defs.c_tagging_asset import tagging_asset
from etl.defs.e_reconcile_tags import reconcile_tags
from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.f_xml import generate_xml
from etl.domain.z_gating import apply_tagged_outputs_gating
from etl.utils.db_utils import upsert_xml
from etl.utils.post_asset_refresh import run_post_asset_refresh
from etl.utils.run_config import is_batched, is_cleanup_mode


TERMINAL_BATCH_STATUSES = ("completed", "failed", "cancelled", "expired")
TAG_TREE_SKIP_TAGS = {"text", "page", "definition", "pageUUID"}
XML_VERIFY_INSTRUCTIONS = (
    "Validate an agreement XML tag tree and return JSON with key `status` only. "
    "Apply these hard rules exactly: "
    "1) Empty articles are allowed, but at most one article may have zero sections. "
    "2) Within each article, sections must start at 1 and be strictly sequential with no gaps. "
    "3) Every section title must begin with a valid section number prefix in one of these forms: "
    "`Section A.B ...`, `SECTION A.B ...`, or `A.B ...` where A and B are integers; "
    "titles that do not match this are invalid. "
    "4) In `<body>`, the first structural child must be an `<article>`, and the first article must be Article I/1. "
    "If any hard rule fails, return status='invalid'. "
    "If all hard rules pass and structure is coherent, return status='verified'. "
    "If still unclear, return status=null."
)
ARTICLE_NUMBER_RE = re.compile(r"^\s*ARTICLE\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)
SECTION_NUMBER_RE = re.compile(
    r"^\s*(?:SECTION\s+)?(?P<article>\d+)\s*\.\s*(?P<section>\d+)",
    re.IGNORECASE,
)
ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _oai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for xml_verify_asset.")
    return OpenAI(api_key=api_key)


def _ensure_xml_verify_batches_table(conn: Connection, schema: str) -> None:
    _ = conn.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.xml_verify_batches (
                batch_id VARCHAR(128) PRIMARY KEY,
                created_at DATETIME NOT NULL DEFAULT UTC_TIMESTAMP(),
                status VARCHAR(32) NOT NULL,
                input_file_id VARCHAR(128) NULL,
                output_file_id VARCHAR(128) NULL,
                error_file_id VARCHAR(128) NULL,
                completion_window VARCHAR(16) NOT NULL,
                request_total INT NOT NULL,
                pulled TINYINT(1) NOT NULL DEFAULT 0,
                pulled_at DATETIME NULL
            )
            """
        )
    )


def _fetch_unpulled_xml_verify_batch(conn: Connection, schema: str) -> Dict[str, Any] | None:
    row = conn.execute(
        text(
            f"""
            SELECT
                batch_id,
                status,
                input_file_id,
                output_file_id,
                error_file_id,
                completion_window,
                request_total
            FROM {schema}.xml_verify_batches
            WHERE pulled = 0
            ORDER BY created_at ASC
            LIMIT 1
            """
        )
    ).mappings().first()
    if row is None:
        return None
    return dict(row)


def _upsert_xml_verify_batch_row(
    conn: Connection,
    schema: str,
    *,
    batch: Any,
    completion_window: str,
    request_total: int,
) -> None:
    _ = conn.execute(
        text(
            f"""
            INSERT INTO {schema}.xml_verify_batches (
                batch_id,
                created_at,
                status,
                input_file_id,
                output_file_id,
                error_file_id,
                completion_window,
                request_total,
                pulled
            )
            VALUES (
                :batch_id,
                UTC_TIMESTAMP(),
                :status,
                :input_file_id,
                :output_file_id,
                :error_file_id,
                :completion_window,
                :request_total,
                0
            )
            ON DUPLICATE KEY UPDATE
                status = VALUES(status),
                input_file_id = VALUES(input_file_id),
                output_file_id = VALUES(output_file_id),
                error_file_id = VALUES(error_file_id),
                completion_window = VALUES(completion_window),
                request_total = VALUES(request_total)
            """
        ),
        {
            "batch_id": batch.id,
            "status": batch.status,
            "input_file_id": getattr(batch, "input_file_id", None),
            "output_file_id": getattr(batch, "output_file_id", None),
            "error_file_id": getattr(batch, "error_file_id", None),
            "completion_window": completion_window,
            "request_total": request_total,
        },
    )


def _mark_xml_verify_batch_pulled(conn: Connection, schema: str, batch_id: str) -> None:
    _ = conn.execute(
        text(
            f"""
            UPDATE {schema}.xml_verify_batches
            SET pulled = 1, pulled_at = UTC_TIMESTAMP()
            WHERE batch_id = :batch_id
            """
        ),
        {"batch_id": batch_id},
    )


def _extract_output_text_from_batch_body(body: Dict[str, Any]) -> str:
    output = body.get("output")
    if not isinstance(output, list):
        raise ValueError(f"Expected body.output to be a list, got {type(output).__name__}")
    msg_blocks = [o for o in output if o.get("type") == "message"]
    if not msg_blocks:
        raise ValueError("No assistant message block in output.")
    contents = msg_blocks[0].get("content")
    if not isinstance(contents, list):
        raise ValueError(f"Expected message content to be a list, got {type(contents).__name__}")
    text_items = [c for c in contents if isinstance(c, dict) and "text" in c]
    if not text_items:
        raise ValueError("Assistant message has no text content.")
    raw_text = text_items[0]["text"]
    if not isinstance(raw_text, str):
        raise ValueError(f"Expected text to be a string, got {type(raw_text).__name__}")
    return raw_text


def _read_file_text(resp: Any) -> str:
    text_attr = getattr(resp, "text", None)
    if callable(text_attr):
        out_text = text_attr()
    elif isinstance(text_attr, str):
        out_text = text_attr
    else:
        content_attr = getattr(resp, "content", None)
        if isinstance(content_attr, bytes):
            out_text = content_attr.decode("utf-8")
        else:
            read_attr = getattr(resp, "read", None)
            if not callable(read_attr):
                raise TypeError("Batch output content has no text/content/read interface.")
            raw_bytes = read_attr()
            if not isinstance(raw_bytes, bytes):
                raise TypeError("Batch output read() did not return bytes.")
            out_text = raw_bytes.decode("utf-8")
    if not isinstance(out_text, str):
        raise TypeError("Batch output text is not a string.")
    return out_text


def _poll_batch_until_terminal(
    context: AssetExecutionContext,
    client: OpenAI,
    batch_id: str,
) -> Any:
    base_sleep_seconds = 5
    backoff_level = 0
    no_update_polls = 0
    last_progress_snapshot: Tuple[Any, ...] | None = None
    max_sleep_seconds = 30 * 60

    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status in TERMINAL_BATCH_STATUSES:
            return batch

        rc = getattr(batch, "request_counts", None)
        if rc is not None:
            completed = getattr(rc, "completed", 0) or 0
            failed = getattr(rc, "failed", 0) or 0
            progress_snapshot = (batch.status, completed, failed)
        else:
            progress_snapshot = (batch.status,)

        if progress_snapshot == last_progress_snapshot:
            no_update_polls += 1
        else:
            if backoff_level > 0:
                prev_sleep = min(
                    base_sleep_seconds * (2**backoff_level),
                    max_sleep_seconds,
                )
                context.log.info(
                    f"xml_verify_asset: backoff reset: interval {prev_sleep}s -> {base_sleep_seconds}s"
                )
            no_update_polls = 0
            backoff_level = 0
            last_progress_snapshot = progress_snapshot

        if no_update_polls >= 10:
            prev_sleep = min(
                base_sleep_seconds * (2**backoff_level),
                max_sleep_seconds,
            )
            backoff_level += 1
            no_update_polls = 0
            new_sleep = min(
                base_sleep_seconds * (2**backoff_level),
                max_sleep_seconds,
            )
            if new_sleep > prev_sleep:
                context.log.info(
                    f"xml_verify_asset: backoff increased: interval {prev_sleep}s -> {new_sleep}s"
                )

        sleep_seconds = min(base_sleep_seconds * (2**backoff_level), max_sleep_seconds)
        context.log.info(
            f"xml_verify_asset: batch {batch_id} status={batch.status}; sleeping {sleep_seconds}s"
        )
        time.sleep(sleep_seconds)


def _roman_to_int(value: str) -> int | None:
    roman = value.strip().upper()
    if not roman:
        return None
    total = 0
    prev = 0
    for char in reversed(roman):
        if char not in ROMAN_VALUES:
            return None
        current = ROMAN_VALUES[char]
        if current < prev:
            total -= current
        else:
            total += current
            prev = current
    return total


def _extract_article_number(title: str) -> int | None:
    match = ARTICLE_NUMBER_RE.match(title)
    if not match:
        return None
    token = match.group(1)
    if token.isdigit():
        return int(token)
    return _roman_to_int(token)


def _extract_article_number_from_elem(article_elem: ET.Element) -> int | None:
    title = article_elem.attrib.get("title", "")
    parsed = _extract_article_number(title)
    if parsed is not None:
        return parsed
    order = article_elem.attrib.get("order")
    if order and order.isdigit():
        return int(order)
    return None


def _extract_section_numbers(title: str) -> Tuple[int, int] | None:
    match = SECTION_NUMBER_RE.match(title)
    if not match:
        return None
    suffix = title[match.end() :].lstrip()
    if suffix.startswith(".") and len(suffix) > 1 and suffix[1].isdigit():
        return None
    article_num = int(match.group("article"))
    section_num = int(match.group("section"))
    return article_num, section_num


def _find_hard_rule_violations(root: ET.Element) -> List[str]:
    violations: List[str] = []
    body = root.find("body")
    if body is None:
        return ["Missing <body>."]

    body_children = [child for child in list(body) if child.tag not in TAG_TREE_SKIP_TAGS]
    if not body_children:
        return ["<body> has no structural children."]

    if body_children[0].tag != "article":
        violations.append("<body> must start with <article>.")
    if any(child.tag != "article" for child in body_children):
        violations.append("<body> contains non-article structural elements.")

    articles = [child for child in body_children if child.tag == "article"]
    if not articles:
        violations.append("<body> has no articles.")
        return violations

    first_article_num = _extract_article_number_from_elem(articles[0])
    if first_article_num != 1:
        violations.append("First body article is not Article I/1.")

    empty_article_count = 0
    for article_elem in articles:
        article_title = article_elem.attrib.get("title", "")
        article_num = _extract_article_number_from_elem(article_elem)
        section_children = [c for c in list(article_elem) if c.tag == "section"]
        if not section_children:
            empty_article_count += 1
            continue

        expected_section_num = 1
        for section_elem in section_children:
            section_title = section_elem.attrib.get("title", "")
            parsed_numbers = _extract_section_numbers(section_title)
            if parsed_numbers is None:
                violations.append(
                    f"Section title is not a valid numbered section: {section_title!r} in article {article_title!r}."
                )
                continue

            section_article_num, section_num = parsed_numbers
            if article_num is not None and section_article_num != article_num:
                violations.append(
                    f"Section {section_article_num}.{section_num} does not match article {article_num} ({article_title!r})."
                )
            if section_num != expected_section_num:
                violations.append(
                    f"Non-sequential section number in article {article_title!r}: expected {expected_section_num}, found {section_num}."
                )
                expected_section_num = section_num + 1
            else:
                expected_section_num += 1

    if empty_article_count > 1:
        violations.append(
            f"Too many empty articles: found {empty_article_count}, maximum allowed is 1."
        )
    return violations


def _render_tag_tree_from_root(root: ET.Element) -> str:
    lines: List[str] = []

    def _walk(elem: ET.Element, indent: int) -> None:
        attrs = []
        for key in ("title", "order", "standardId"):
            if key in elem.attrib:
                attrs.append(f'{key}="{elem.attrib[key]}"')
        attr_str = " " + " ".join(attrs) if attrs else ""
        lines.append("    " * indent + f"{elem.tag}{attr_str}")
        for child in elem:
            if child.tag not in TAG_TREE_SKIP_TAGS:
                _walk(child, indent + 1)

    _walk(root, indent=0)
    return "\n".join(lines)


def _build_xml_verify_batch_request_body(
    *,
    custom_id: str,
    tag_tree: str,
    model: str,
) -> Dict[str, Any]:
    schema: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {
                "anyOf": [
                    {"type": "string", "enum": ["verified", "invalid"]},
                    {"type": "null"},
                ]
            }
        },
        "required": ["status"],
    }
    body: Dict[str, Any] = {
        "model": model,
        "instructions": XML_VERIFY_INSTRUCTIONS,
        "input": f"XML tag tree:\n{tag_tree}",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "xml_tree_validation",
                "strict": True,
                "schema": schema,
            }
        },
    }
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def _parse_xml_verify_response_text(raw_text: str) -> str | None:
    obj = json.loads(raw_text)
    if not isinstance(obj, dict):
        raise ValueError("Response JSON is not an object.")
    if "status" not in obj:
        raise ValueError("Missing required key: status")

    status = obj["status"]
    if status == "null":
        return None
    if status is None:
        return None
    if status not in ("verified", "invalid"):
        raise ValueError(f"Invalid status: {status}")
    return str(status)


def _parse_custom_id(custom_id: str) -> Tuple[str, int]:
    parts = custom_id.split("|")
    if len(parts) != 2:
        raise ValueError(f"Unexpected custom_id format: {custom_id}")
    agreement_uuid = parts[0]
    version = int(parts[1])
    return agreement_uuid, version


def _apply_xml_verify_batch_output(
    context: AssetExecutionContext,
    engine: Any,
    client: OpenAI,
    xml_table: str,
    batch: Any,
) -> Tuple[int, int]:
    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        context.log.warning("xml_verify_asset: batch has no output_file_id.")
        return 0, 0

    out_content = client.files.content(output_file_id)
    out_text = _read_file_text(out_content)

    update_q = text(
        f"""
        UPDATE {xml_table}
        SET status = :status,
            status_source = 'asset'
        WHERE agreement_uuid = :agreement_uuid
          AND version = :version
          AND (NOT (status <=> :status) OR NOT (status_source <=> 'asset'))
        """
    )

    updated = 0
    parse_errors = 0
    lines = [line for line in out_text.strip().splitlines() if line.strip()]
    with engine.begin() as conn:
        for line_str in lines:
            try:
                raw = json.loads(line_str)
                custom_id = raw.get("custom_id")
                resp = raw.get("response")
                if not custom_id or not resp:
                    raise ValueError("Missing custom_id or response.")
                agreement_uuid, version = _parse_custom_id(str(custom_id))

                status_code = resp.get("status_code")
                if status_code not in (200, 201, 202):
                    raise ValueError(f"Unexpected status_code={status_code}")
                body = resp.get("body")
                if not isinstance(body, dict):
                    raise ValueError("Missing response body.")

                raw_text = _extract_output_text_from_batch_body(body)
                parsed_status = _parse_xml_verify_response_text(raw_text)

                result = conn.execute(
                    update_q,
                    {
                        "agreement_uuid": agreement_uuid,
                        "version": version,
                        "status": parsed_status,
                    },
                )
                updated += int(result.rowcount or 0)
            except Exception as e:
                parse_errors += 1
                context.log.warning(f"xml_verify_asset: parse/apply error: {e}")
    return updated, parse_errors


@dg.asset(deps=[tagging_asset, reconcile_tags], name="6-1_build_xml")
def xml_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> None:
    """
    Assemble tagged sections into XML documents.

    Re-creates XML for agreements where tagged_outputs have been updated since the last XML creation.
    Maintains version numbers and tracks creation dates.
    
    Args:
        context: Dagster execution context.
        db: Database resource for connection.
        pipeline_config: Pipeline configuration for mode.
    """
    # batching controls
    agreement_batch_size = pipeline_config.xml_agreement_batch_size
    batched = is_batched(context, pipeline_config)

    engine = db.get_engine()
    schema = db.database
    agreements_table = f"{schema}.agreements"
    pages_table = f"{schema}.pages"
    tagged_outputs_table = f"{schema}.tagged_outputs"
    xml_table = f"{schema}.xml"
    is_cleanup = is_cleanup_mode(context, pipeline_config)

    context.log.info(
        f"Running XML generation in {'CLEANUP' if is_cleanup else 'FROM_SCRATCH'} mode"
    )

    with engine.begin() as conn:
        _ = apply_tagged_outputs_gating(conn, db.database)

    last_uuid = ''
    while True:
        with engine.begin() as conn:
            # Fetch batch of agreements where:
            # 1. Either no XML exists yet, OR
            # 2. tagged_outputs have been updated after the last XML creation date
            agreement_uuids = (
                conn.execute(
                    text(
                        f"""
                        SELECT DISTINCT
                            a.agreement_uuid
                        FROM
                            {agreements_table} a
                        JOIN {pages_table} p
                            ON p.agreement_uuid = a.agreement_uuid
                        LEFT JOIN {tagged_outputs_table} t
                            ON t.page_uuid = p.page_uuid
                        LEFT JOIN {xml_table} x
                            ON x.agreement_uuid = a.agreement_uuid
                            AND x.latest = 1
                        LEFT JOIN (
                            SELECT DISTINCT p2.agreement_uuid
                            FROM {tagged_outputs_table} t2
                            JOIN {pages_table} p2
                                ON t2.page_uuid = p2.page_uuid
                            WHERE t2.gated = 1
                        ) gated
                            ON gated.agreement_uuid = a.agreement_uuid
                        WHERE a.agreement_uuid > :last_uuid
                        AND p.source_page_type = 'body'
                        AND COALESCE(t.tagged_text_gold, t.tagged_text_corrected, t.tagged_text) IS NOT NULL
                        AND gated.agreement_uuid IS NULL
                        AND (
                            x.agreement_uuid IS NULL
                            OR EXISTS (
                                SELECT 1
                                FROM {pages_table} p_upd
                                JOIN {tagged_outputs_table} t_upd
                                    ON t_upd.page_uuid = p_upd.page_uuid
                                WHERE p_upd.agreement_uuid = a.agreement_uuid
                                AND p_upd.source_page_type = 'body'
                                AND t_upd.updated_date > x.created_date
                            )
                        )
                        GROUP BY a.agreement_uuid
                        HAVING
                        -- ensure that there is at least one body page and that all body pages are tagged
                        SUM(CASE WHEN p.source_page_type = 'body' THEN 1 ELSE 0 END) > 0
                        AND SUM(
                            CASE WHEN p.source_page_type = 'body'
                                AND COALESCE(t.tagged_text_corrected, t.tagged_text) IS NOT NULL
                                THEN 1 ELSE 0 END
                        ) = SUM(CASE WHEN p.source_page_type = 'body' THEN 1 ELSE 0 END)
                        ORDER BY a.agreement_uuid
                        LIMIT :limit;
                """
                    ),
                    {"limit": agreement_batch_size, "last_uuid": last_uuid},
                )
                .scalars()
                .all()
            )

            # If none left, we're done
            if not agreement_uuids:
                break

            # Fetch every page and its tagged output for those agreements
            rows = (
                conn.execute(
                    text(
                        f"""
                    SELECT
                    p.agreement_uuid,
                    p.page_uuid,
                    p.page_order,
                    coalesce(p.gold_label, p.source_page_type) as source_page_type,
                    coalesce(tgo.tagged_text_gold, tgo.tagged_text_corrected, tgo.tagged_text, p.processed_page_content) as tagged_output,
                    url,
                    acquirer,
                    target,
                    filing_date,
                    source_is_txt,
                    source_is_html
                    FROM {pages_table} p
                    JOIN {agreements_table} a on p.agreement_uuid = a.agreement_uuid
                    LEFT JOIN {tagged_outputs_table} tgo
                    ON p.page_uuid = tgo.page_uuid
                    WHERE p.agreement_uuid IN :uuids
                    ORDER BY p.agreement_uuid, p.page_order
                """
                    ),
                    {"uuids": tuple(agreement_uuids)},
                )
                .mappings()
                .fetchall()
            )

            df = pd.DataFrame(rows)
            # Determine version: new agreements get v1, updated pages increment version
            existing_versions = conn.execute(
                text(f"""
                    SELECT agreement_uuid, MAX(version) as max_version
                    FROM {xml_table}
                    WHERE agreement_uuid IN :uuids
                    GROUP BY agreement_uuid
                """),
                {"uuids": tuple(agreement_uuids)},
            ).mappings().fetchall()
            
            version_map = {row["agreement_uuid"]: row["max_version"] + 1 for row in existing_versions}
            
            # Generate XML from tagged pages
            xml, xml_generation_failures = generate_xml(df, version_map)
            for failure in xml_generation_failures:
                context.log.warning(
                    "Skipping XML generation due to parse error for agreement_uuid=%s: %s",
                    failure.agreement_uuid,
                    failure.error,
                )

            if not xml:
                context.log.warning(
                    "Skipping XML upsert for this batch because all %s agreements failed XML parsing",
                    len(agreement_uuids),
                )
                last_uuid = agreement_uuids[-1]
                if batched:
                    break
                continue

            generated_agreement_uuids = [item.agreement_uuid for item in xml]

            try:
                upsert_xml(xml, db.database, conn)
                _ = conn.execute(
                    text(
                        f"""
                        UPDATE {xml_table} x
                        JOIN (
                            SELECT agreement_uuid, MAX(version) AS max_version
                            FROM {xml_table}
                            WHERE agreement_uuid IN :uuids
                            GROUP BY agreement_uuid
                        ) m ON x.agreement_uuid = m.agreement_uuid
                        SET x.latest = CASE
                            WHEN x.version = m.max_version THEN 1
                            ELSE 0
                        END
                        WHERE x.agreement_uuid IN :uuids
                        """
                    ).bindparams(bindparam("uuids", expanding=True)),
                    {"uuids": generated_agreement_uuids},
                )
                context.log.info(
                    f"Successfully generated XML for {len(generated_agreement_uuids)} agreements"
                )
            except Exception as e:
                context.log.error(f"Error upserting XML: {e}")
                raise RuntimeError(e)
            
            last_uuid = agreement_uuids[-1]
        if batched:
            break

    run_post_asset_refresh(context, db, pipeline_config)


@dg.asset(deps=[xml_asset], name="4-2_verify_xml")
def xml_verify_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> None:
    agreement_batch_size = pipeline_config.xml_verify_batch_size
    engine = db.get_engine()
    schema = db.database
    xml_table = f"{schema}.xml"
    client = _oai_client()

    with engine.begin() as conn:
        _ensure_xml_verify_batches_table(conn, schema)
        existing_batch = _fetch_unpulled_xml_verify_batch(conn, schema)

    if existing_batch is not None:
        existing_batch_id = str(existing_batch["batch_id"])
        context.log.info(f"xml_verify_asset: resuming existing batch {existing_batch_id}.")
        batch = _poll_batch_until_terminal(context, client, existing_batch_id)
        request_total = int(existing_batch["request_total"])
        with engine.begin() as conn:
            _upsert_xml_verify_batch_row(
                conn,
                schema,
                batch=batch,
                completion_window=str(existing_batch["completion_window"]),
                request_total=request_total,
            )

        if batch.status != "completed":
            context.log.warning(
                f"xml_verify_asset: batch {batch.id} ended with status={batch.status}; no status updates applied."
            )
            with engine.begin() as conn:
                _mark_xml_verify_batch_pulled(conn, schema, batch.id)
        else:
            updated, parse_errors = _apply_xml_verify_batch_output(
                context=context,
                engine=engine,
                client=client,
                xml_table=xml_table,
                batch=batch,
            )
            with engine.begin() as conn:
                _mark_xml_verify_batch_pulled(conn, schema, batch.id)
            context.log.info(
                f"xml_verify_asset: resumed batch {batch.id} completed; updated={updated}, parse_errors={parse_errors}"
            )

        run_post_asset_refresh(context, db, pipeline_config)
        return

    select_q = text(
        f"""
        SELECT agreement_uuid, version, xml
        FROM {xml_table}
        WHERE status IS NULL
          AND gated = 0
          AND latest = 1
        ORDER BY agreement_uuid ASC
        LIMIT :lim
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(select_q, {"lim": agreement_batch_size}).mappings().fetchall()
    if not rows:
        context.log.info("xml_verify_asset: no XML rows with status IS NULL and latest=1.")
        run_post_asset_refresh(context, db, pipeline_config)
        return

    lines: List[Dict[str, Any]] = []
    hard_invalid_rows: List[Dict[str, Any]] = []
    for row in rows:
        agreement_uuid = str(row["agreement_uuid"])
        version = int(row["version"])
        xml_text = row["xml"]
        try:
            root = ET.fromstring(str(xml_text))
        except Exception as e:
            hard_invalid_rows.append(
                {
                    "agreement_uuid": agreement_uuid,
                    "version": version,
                    "reason": f"XML parse failure: {e}",
                }
            )
            continue

        hard_rule_violations = _find_hard_rule_violations(root)
        if hard_rule_violations:
            hard_invalid_rows.append(
                {
                    "agreement_uuid": agreement_uuid,
                    "version": version,
                    "reason": "; ".join(hard_rule_violations[:3]),
                }
            )
            continue

        try:
            tag_tree = _render_tag_tree_from_root(root)
        except Exception as e:
            hard_invalid_rows.append(
                {
                    "agreement_uuid": agreement_uuid,
                    "version": version,
                    "reason": f"Tag tree render failure: {e}",
                }
            )
            continue

        custom_id = f"{agreement_uuid}|{version}"
        lines.append(
            _build_xml_verify_batch_request_body(
                custom_id=custom_id,
                tag_tree=tag_tree,
                model="gpt-5-mini",
            )
        )

    hard_invalid_updated = 0
    if hard_invalid_rows:
        hard_invalidate_q = text(
            f"""
            UPDATE {xml_table}
            SET status = 'invalid',
                status_source = 'asset'
            WHERE agreement_uuid = :agreement_uuid
              AND version = :version
              AND (NOT (status <=> 'invalid') OR NOT (status_source <=> 'asset'))
            """
        )
        with engine.begin() as conn:
            for row in hard_invalid_rows:
                result = conn.execute(
                    hard_invalidate_q,
                    {
                        "agreement_uuid": row["agreement_uuid"],
                        "version": row["version"],
                    },
                )
                hard_invalid_updated += int(result.rowcount or 0)

        sample_reasons = ", ".join(
            f"{r['agreement_uuid']}@v{r['version']}: {r['reason']}"
            for r in hard_invalid_rows[:3]
        )
        context.log.info(
            "xml_verify_asset: hard-rule invalidated %s XML rows before LLM. samples=%s",
            len(hard_invalid_rows),
            sample_reasons,
        )

    if not lines:
        context.log.info(
            "xml_verify_asset: no LLM submissions required after hard-rule checks; hard_invalid_updated=%s",
            hard_invalid_updated,
        )
        run_post_asset_refresh(context, db, pipeline_config)
        return

    jsonl_buf = io.StringIO()
    for line in lines:
        _ = jsonl_buf.write(json.dumps(line, ensure_ascii=False) + "\n")
    jsonl_bytes = io.BytesIO(jsonl_buf.getvalue().encode("utf-8"))
    jsonl_bytes.name = "xml_verify_requests.jsonl"

    input_file = client.files.create(purpose="batch", file=jsonl_bytes)
    completion_window = "24h"
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/responses",
        completion_window=completion_window,
    )
    with engine.begin() as conn:
        _upsert_xml_verify_batch_row(
            conn,
            schema,
            batch=batch,
            completion_window=completion_window,
            request_total=len(lines),
        )
    context.log.info(
        f"xml_verify_asset: created batch {batch.id} with {len(lines)} requests; polling until complete."
    )

    final_batch = _poll_batch_until_terminal(context, client, batch.id)
    with engine.begin() as conn:
        _upsert_xml_verify_batch_row(
            conn,
            schema,
            batch=final_batch,
            completion_window=completion_window,
            request_total=len(lines),
        )

    if final_batch.status != "completed":
        context.log.warning(
            f"xml_verify_asset: batch {final_batch.id} ended with status={final_batch.status}; no status updates applied."
        )
        with engine.begin() as conn:
            _mark_xml_verify_batch_pulled(conn, schema, final_batch.id)
        run_post_asset_refresh(context, db, pipeline_config)
        return

    updated, parse_errors = _apply_xml_verify_batch_output(
        context=context,
        engine=engine,
        client=client,
        xml_table=xml_table,
        batch=final_batch,
    )
    with engine.begin() as conn:
        _mark_xml_verify_batch_pulled(conn, schema, final_batch.id)
    context.log.info(
        f"xml_verify_asset: batch {final_batch.id} completed; updated={updated}, parse_errors={parse_errors}, hard_invalid_updated={hard_invalid_updated}"
    )

    run_post_asset_refresh(context, db, pipeline_config)
