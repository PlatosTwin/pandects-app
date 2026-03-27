"""Assemble tagged sections into XML documents and verify XML tree structure."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false

import io
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd

import dagster as dg
from dagster import AssetExecutionContext
from openai import OpenAI
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection

from etl.defs.c_tagging_asset import tagging_asset
from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.f_xml import generate_xml
from etl.utils.db_utils import upsert_xml
from etl.utils.batch_keys import agreement_version_batch_key
from etl.utils.openai_batch import (
    extract_output_text_from_batch_body,
    poll_batch_until_terminal,
    read_openai_file_text,
)
from etl.utils.post_asset_refresh import run_post_asset_refresh
from etl.utils.latest_sections_search import refresh_latest_sections_search
from etl.utils.pipeline_state_sql import (
    canonical_fresh_xml_build_queue_sql,
    canonical_fresh_xml_verify_queue_sql,
)
from etl.utils.run_config import runs_single_batch
from etl.utils.schema_guards import assert_tables_exist


TAG_TREE_SKIP_TAGS = {"text", "page", "definition", "pageUUID"}
XML_REASON_XML_PARSE_FAILURE = "xml_parse_failure"
XML_REASON_TAG_TREE_RENDER_FAILURE = "tag_tree_render_failure"
XML_REASON_LLM_INVALID = "llm_invalid"
XML_REASON_MISSING_BODY = "missing_body"
XML_REASON_BODY_HAS_NO_STRUCTURAL_CHILDREN = "body_has_no_structural_children"
XML_REASON_BODY_STARTS_NON_ARTICLE = "body_starts_non_article"
XML_REASON_BODY_CONTAINS_NON_ARTICLE_CHILDREN = "body_contains_non_article_children"
XML_REASON_BODY_HAS_NO_ARTICLES = "body_has_no_articles"
XML_REASON_TOO_FEW_ARTICLES = "too_few_articles"
XML_REASON_FIRST_ARTICLE_NOT_ONE = "first_article_not_one"
XML_REASON_SECTION_TITLE_INVALID_NUMBERING = "section_title_invalid_numbering"
XML_REASON_SECTION_ARTICLE_MISMATCH = "section_article_mismatch"
XML_REASON_SECTION_NON_SEQUENTIAL = "section_non_sequential"
XML_REASON_TOO_MANY_EMPTY_ARTICLES = "too_many_empty_articles"
XML_VERIFY_BATCH_SCOPE_DEFAULT = "default"
XML_VERIFY_BATCH_SCOPE_REPAIR = "repair"
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


@dataclass(frozen=True)
class XMLHardRuleViolation:
    reason_code: str
    reason_detail: str
    page_uuids: Tuple[str, ...]


def _oai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for xml_verify_asset.")
    return OpenAI(api_key=api_key)


def _fetch_unpulled_xml_verify_batch(
    conn: Connection,
    schema: str,
    batch_scope: str,
    batch_key: str | None = None,
) -> Dict[str, Any] | None:
    if batch_key is None:
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
                    request_total,
                    batch_scope,
                    batch_key
                FROM {schema}.xml_verify_batches
                WHERE pulled = 0
                  AND batch_scope = :batch_scope
                ORDER BY created_at ASC
                LIMIT 1
                """
            ),
            {"batch_scope": batch_scope},
        ).mappings().first()
    else:
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
                    request_total,
                    batch_scope,
                    batch_key
                FROM {schema}.xml_verify_batches
                WHERE pulled = 0
                  AND batch_scope = :batch_scope
                  AND batch_key = :batch_key
                ORDER BY created_at ASC
                LIMIT 1
                """
            ),
            {"batch_scope": batch_scope, "batch_key": batch_key},
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
    batch_scope: str,
    batch_key: str | None,
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
                batch_scope,
                batch_key,
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
                :batch_scope,
                :batch_key,
                0
            )
            ON DUPLICATE KEY UPDATE
                status = VALUES(status),
                input_file_id = VALUES(input_file_id),
                output_file_id = VALUES(output_file_id),
                error_file_id = VALUES(error_file_id),
                completion_window = VALUES(completion_window),
                request_total = VALUES(request_total),
                batch_scope = VALUES(batch_scope),
                batch_key = VALUES(batch_key)
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
            "batch_scope": batch_scope,
            "batch_key": batch_key,
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


def _load_xml_verify_batch_agreement_uuids(
    client: OpenAI,
    batch_row: Dict[str, Any],
) -> List[str]:
    input_file_id = batch_row.get("input_file_id")
    if not input_file_id:
        raise ValueError("xml verify batch row has no input_file_id")

    raw_text = read_openai_file_text(client.files.content(str(input_file_id))).strip()
    if not raw_text:
        raise ValueError(f"xml verify batch input file {input_file_id} is empty")

    agreement_uuids: set[str] = set()
    for line in raw_text.splitlines():
        payload = json.loads(line)
        custom_id = payload.get("custom_id")
        if not isinstance(custom_id, str):
            raise ValueError("xml verify batch input line is missing string custom_id")
        agreement_uuid, _ = _parse_custom_id(custom_id)
        agreement_uuids.add(agreement_uuid)

    if not agreement_uuids:
        raise ValueError("xml verify batch input file contained no agreement targets")
    return sorted(agreement_uuids)


def _resume_xml_verify_batch(
    context: AssetExecutionContext,
    engine: Any,
    db: DBResource,
    pipeline_config: PipelineConfig,
    client: OpenAI,
    *,
    schema: str,
    xml_table: str,
    batch_scope: str,
    batch_row: Dict[str, Any],
    agreement_uuids: List[str],
    log_prefix: str,
    hard_invalid_updated: int,
) -> List[str]:
    batch_id = str(batch_row["batch_id"])
    batch_key = batch_row.get("batch_key")
    context.log.info(
        "%s: resuming unpulled batch %s for %s agreements.",
        log_prefix,
        batch_id,
        len(agreement_uuids),
    )
    batch = poll_batch_until_terminal(
        context,
        client,
        batch_id,
        log_prefix=log_prefix,
    )
    request_total = int(batch_row["request_total"])
    with engine.begin() as conn:
        _upsert_xml_verify_batch_row(
            conn,
            schema,
            batch=batch,
            completion_window=str(batch_row["completion_window"]),
            request_total=request_total,
            batch_scope=batch_scope,
            batch_key=str(batch_key) if batch_key is not None else None,
        )

    if batch.status != "completed":
        context.log.warning(
            "%s: resumed batch %s ended with status=%s; no status updates applied.",
            log_prefix,
            batch.id,
            batch.status,
        )
        with engine.begin() as conn:
            _mark_xml_verify_batch_pulled(conn, schema, str(batch.id))
        run_post_asset_refresh(context, db, pipeline_config)
        return agreement_uuids

    updated, parse_errors = _apply_xml_verify_batch_output(
        context=context,
        engine=engine,
        client=client,
        xml_table=xml_table,
        xml_status_reasons_table=f"{schema}.xml_status_reasons",
        batch=batch,
        log_prefix=log_prefix,
    )
    with engine.begin() as conn:
        _mark_xml_verify_batch_pulled(conn, schema, str(batch.id))
    context.log.info(
        "%s: resumed batch %s completed; updated=%s, parse_errors=%s, hard_invalid_updated=%s",
        log_prefix,
        batch.id,
        updated,
        parse_errors,
        hard_invalid_updated,
    )

    run_post_asset_refresh(context, db, pipeline_config)
    return agreement_uuids


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


def _collect_page_uuids(root: ET.Element) -> Tuple[str, ...]:
    attr_uuid = (
        root.attrib.get("pageUUID")
        or root.attrib.get("page_uuid")
        or root.attrib.get("sourcePageUUID")
    )
    if attr_uuid is not None:
        attr_uuid = attr_uuid.strip()
        if attr_uuid:
            return (attr_uuid,)
    page_uuids: List[str] = []
    seen: set[str] = set()
    for node in root.iter("pageUUID"):
        text_val = (node.text or "").strip()
        if not text_val:
            continue
        if text_val in seen:
            continue
        seen.add(text_val)
        page_uuids.append(text_val)
    return tuple(page_uuids)


def find_hard_rule_violations(root: ET.Element) -> List[XMLHardRuleViolation]:
    violations: List[XMLHardRuleViolation] = []
    body = root.find("body")
    if body is None:
        return [
            XMLHardRuleViolation(
                reason_code=XML_REASON_MISSING_BODY,
                reason_detail="Missing <body>.",
                page_uuids=(),
            )
        ]

    body_children = [child for child in list(body) if child.tag not in TAG_TREE_SKIP_TAGS]
    if not body_children:
        return [
            XMLHardRuleViolation(
                reason_code=XML_REASON_BODY_HAS_NO_STRUCTURAL_CHILDREN,
                reason_detail="<body> has no structural children.",
                page_uuids=(),
            )
        ]

    if body_children[0].tag != "article":
        violations.append(
            XMLHardRuleViolation(
                reason_code=XML_REASON_BODY_STARTS_NON_ARTICLE,
                reason_detail="<body> must start with <article>.",
                page_uuids=_collect_page_uuids(body_children[0]),
            )
        )
    if any(child.tag != "article" for child in body_children):
        violations.append(
            XMLHardRuleViolation(
                reason_code=XML_REASON_BODY_CONTAINS_NON_ARTICLE_CHILDREN,
                reason_detail="<body> contains non-article structural elements.",
                page_uuids=_collect_page_uuids(body),
            )
        )

    articles = [child for child in body_children if child.tag == "article"]
    if not articles:
        violations.append(
            XMLHardRuleViolation(
                reason_code=XML_REASON_BODY_HAS_NO_ARTICLES,
                reason_detail="<body> has no articles.",
                page_uuids=_collect_page_uuids(body),
            )
        )
        return violations

    if len(articles) < 5:
        violations.append(
            XMLHardRuleViolation(
                reason_code=XML_REASON_TOO_FEW_ARTICLES,
                reason_detail=f"Too few articles: found {len(articles)}, minimum required is 5.",
                page_uuids=_collect_page_uuids(body),
            )
        )

    first_article_num = _extract_article_number_from_elem(articles[0])
    if first_article_num != 1:
        violations.append(
            XMLHardRuleViolation(
                reason_code=XML_REASON_FIRST_ARTICLE_NOT_ONE,
                reason_detail="First body article is not Article I/1.",
                page_uuids=_collect_page_uuids(articles[0]),
            )
        )

    empty_article_count = 0
    empty_article_page_uuids: List[str] = []
    for article_elem in articles:
        article_title = article_elem.attrib.get("title", "")
        article_num = _extract_article_number_from_elem(article_elem)
        section_children = [c for c in list(article_elem) if c.tag == "section"]
        if not section_children:
            empty_article_count += 1
            for page_uuid in _collect_page_uuids(article_elem):
                if page_uuid not in empty_article_page_uuids:
                    empty_article_page_uuids.append(page_uuid)
            continue

        expected_section_num = 1
        for section_elem in section_children:
            section_title = section_elem.attrib.get("title", "")
            section_page_uuids = _collect_page_uuids(section_elem)
            parsed_numbers = _extract_section_numbers(section_title)
            if parsed_numbers is None:
                violations.append(
                    XMLHardRuleViolation(
                        reason_code=XML_REASON_SECTION_TITLE_INVALID_NUMBERING,
                        reason_detail=f"Section title is not a valid numbered section: {section_title!r} in article {article_title!r}.",
                        page_uuids=section_page_uuids,
                    )
                )
                continue

            section_article_num, section_num = parsed_numbers
            if article_num is not None and section_article_num != article_num:
                violations.append(
                    XMLHardRuleViolation(
                        reason_code=XML_REASON_SECTION_ARTICLE_MISMATCH,
                        reason_detail=f"Section {section_article_num}.{section_num} does not match article {article_num} ({article_title!r}).",
                        page_uuids=section_page_uuids,
                    )
                )
            if section_num != expected_section_num:
                violations.append(
                    XMLHardRuleViolation(
                        reason_code=XML_REASON_SECTION_NON_SEQUENTIAL,
                        reason_detail=f"Non-sequential section number in article {article_title!r}: expected {expected_section_num}, found {section_num}.",
                        page_uuids=section_page_uuids,
                    )
                )
                expected_section_num = section_num + 1
            else:
                expected_section_num += 1

    if empty_article_count > 1:
        violations.append(
            XMLHardRuleViolation(
                reason_code=XML_REASON_TOO_MANY_EMPTY_ARTICLES,
                reason_detail=f"Too many empty articles: found {empty_article_count}, maximum allowed is 1.",
                page_uuids=tuple(empty_article_page_uuids),
            )
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


def _dedupe_reason_rows(reason_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()
    for row in reason_rows:
        reason_code = str(row["reason_code"])
        reason_detail = None if row.get("reason_detail") is None else str(row["reason_detail"])
        page_uuid = None if row.get("page_uuid") is None else str(row["page_uuid"])
        key = (reason_code, reason_detail or "", page_uuid or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            {
                "reason_code": reason_code,
                "reason_detail": reason_detail,
                "page_uuid": page_uuid,
            }
        )
    return deduped


def _replace_xml_status_reasons(
    conn: Connection,
    xml_status_reasons_table: str,
    *,
    agreement_uuid: str,
    version: int,
    reason_rows: List[Dict[str, Any]],
) -> None:
    _ = conn.execute(
        text(
            f"""
            DELETE FROM {xml_status_reasons_table}
            WHERE agreement_uuid = :agreement_uuid
              AND xml_version = :version
            """
        ),
        {"agreement_uuid": agreement_uuid, "version": version},
    )

    deduped_rows = _dedupe_reason_rows(reason_rows)
    if not deduped_rows:
        return

    insert_q = text(
        f"""
        INSERT INTO {xml_status_reasons_table} (
            agreement_uuid,
            xml_version,
            reason_code,
            reason_detail,
            page_uuid,
            created_at
        )
        VALUES (
            :agreement_uuid,
            :version,
            :reason_code,
            :reason_detail,
            :page_uuid,
            UTC_TIMESTAMP()
        )
        """
    )
    for row in deduped_rows:
        _ = conn.execute(
            insert_q,
            {
                "agreement_uuid": agreement_uuid,
                "version": version,
                "reason_code": row["reason_code"],
                "reason_detail": row["reason_detail"],
                "page_uuid": row["page_uuid"],
            },
        )


def _set_xml_status_with_reasons(
    conn: Connection,
    xml_table: str,
    xml_status_reasons_table: str,
    *,
    agreement_uuid: str,
    version: int,
    status: str | None,
    reason_rows: List[Dict[str, Any]],
) -> int:
    deduped_rows = _dedupe_reason_rows(reason_rows)
    primary_reason_code = deduped_rows[0]["reason_code"] if status == "invalid" and deduped_rows else None
    primary_reason_detail = deduped_rows[0]["reason_detail"] if status == "invalid" and deduped_rows else None
    result = conn.execute(
        text(
            f"""
            UPDATE {xml_table}
            SET status = :status,
                status_source = 'asset',
                status_reason_code = :reason_code,
                status_reason_detail = :reason_detail
            WHERE agreement_uuid = :agreement_uuid
              AND version = :version
              AND (
                NOT (status <=> :status)
                OR NOT (status_source <=> 'asset')
                OR NOT (status_reason_code <=> :reason_code)
                OR NOT (status_reason_detail <=> :reason_detail)
              )
            """
        ),
        {
            "agreement_uuid": agreement_uuid,
            "version": version,
            "status": status,
            "reason_code": primary_reason_code,
            "reason_detail": primary_reason_detail,
        },
    )
    _replace_xml_status_reasons(
        conn,
        xml_status_reasons_table,
        agreement_uuid=agreement_uuid,
        version=version,
        reason_rows=deduped_rows if status == "invalid" else [],
    )
    return int(result.rowcount or 0)


def _apply_xml_verify_batch_output(
    context: AssetExecutionContext,
    engine: Any,
    client: OpenAI,
    xml_table: str,
    xml_status_reasons_table: str,
    batch: Any,
    *,
    log_prefix: str = "xml_verify_asset",
) -> Tuple[int, int]:
    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        context.log.warning("%s: batch has no output_file_id.", log_prefix)
        return 0, 0

    out_content = client.files.content(output_file_id)
    out_text = read_openai_file_text(out_content)

    updated = 0
    parse_errors = 0
    lines = [line for line in out_text.strip().splitlines() if line.strip()]
    context.log.info(
        "%s: applying batch output for responses_total=%s.",
        log_prefix,
        len(lines),
    )
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

                raw_text = extract_output_text_from_batch_body(body)
                parsed_status = _parse_xml_verify_response_text(raw_text)
                reason_rows: List[Dict[str, Any]] = []
                if parsed_status == "invalid":
                    reason_rows.append(
                        {
                            "reason_code": XML_REASON_LLM_INVALID,
                            "reason_detail": None,
                            "page_uuid": None,
                        }
                    )

                updated += _set_xml_status_with_reasons(
                    conn,
                    xml_table,
                    xml_status_reasons_table,
                    agreement_uuid=agreement_uuid,
                    version=version,
                    status=parsed_status,
                    reason_rows=reason_rows,
                )
            except Exception as e:
                parse_errors += 1
                context.log.warning("%s: parse/apply error: %s", log_prefix, e)
    context.log.info(
        "%s: applied batch output responses=%s/%s, updated=%s, parse_errors=%s",
        log_prefix,
        len(lines),
        len(lines),
        updated,
        parse_errors,
    )
    return updated, parse_errors


@dg.asset(deps=[tagging_asset], name="4-1_build_xml")
def xml_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> List[str]:
    """
    Assemble tagged sections into XML documents.

    Re-creates XML for agreements where tagged_outputs have been updated since the last XML creation.
    Maintains version numbers and tracks creation dates.
    
    Args:
        context: Dagster execution context.
        db: Database resource for connection.
        pipeline_config: Pipeline configuration.
    """
    # batching controls
    agreement_batch_size = pipeline_config.xml_agreement_batch_size
    single_batch_run = runs_single_batch(context, pipeline_config)

    engine = db.get_engine()
    schema = db.database
    agreements_table = f"{schema}.agreements"
    pages_table = f"{schema}.pages"
    tagged_outputs_table = f"{schema}.tagged_outputs"
    xml_table = f"{schema}.xml"
    context.log.info("Running XML generation")

    if pipeline_config.resume_openai_batches:
        with engine.begin() as conn:
            stranded_verify_batch = _fetch_unpulled_xml_verify_batch(
                conn,
                schema,
                batch_scope=XML_VERIFY_BATCH_SCOPE_DEFAULT,
            )
        if stranded_verify_batch is not None:
            context.log.info(
                "xml_asset: deferring new XML generation because unpulled verify batch %s is waiting to resume.",
                stranded_verify_batch["batch_id"],
            )
            run_post_asset_refresh(context, db, pipeline_config)
            return []

    last_uuid = ''
    built_agreement_uuids: List[str] = []
    while True:
        with engine.begin() as conn:
            agreement_uuids = (
                conn.execute(
                    text(canonical_fresh_xml_build_queue_sql(schema)),
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
                    case
                        when coalesce(p.gold_label, p.source_page_type) = 'body' then
                            coalesce(
                                tgo.tagged_text_gold,
                                tgo.tagged_text_corrected,
                                tgo.tagged_text,
                                p.processed_page_content
                            )
                        else p.processed_page_content
                    end as tagged_output,
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
                if single_batch_run:
                    break
                continue

            generated_agreement_uuids = [str(item.agreement_uuid) for item in xml]
            built_agreement_uuids.extend(generated_agreement_uuids)

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
                refreshed = refresh_latest_sections_search(
                    conn,
                    db.database,
                    generated_agreement_uuids,
                )
                context.log.info(
                    "Successfully generated XML for %s agreements; refreshed latest_sections_search rows=%s",
                    len(generated_agreement_uuids),
                    refreshed,
                )
            except Exception as e:
                context.log.error(f"Error upserting XML: {e}")
                raise RuntimeError(e)
            
            last_uuid = agreement_uuids[-1]
        if single_batch_run:
            break

    run_post_asset_refresh(context, db, pipeline_config)
    return sorted(set(built_agreement_uuids))


@dg.asset(
    name="4-2_verify_xml",
    ins={"built_xml_agreement_uuids": dg.AssetIn(key=xml_asset.key)},
)
def xml_verify_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    built_xml_agreement_uuids: List[str],
) -> List[str]:
    agreement_batch_size = pipeline_config.xml_agreement_batch_size
    resume_openai_batches = pipeline_config.resume_openai_batches
    target_agreement_uuids = sorted(set(built_xml_agreement_uuids))

    engine = db.get_engine()
    schema = db.database
    xml_table = f"{schema}.xml"
    client = _oai_client()

    with engine.begin() as conn:
        assert_tables_exist(conn, schema=schema, table_names=("xml_verify_batches", "xml_status_reasons"))

    if resume_openai_batches:
        with engine.begin() as conn:
            stranded_batch = _fetch_unpulled_xml_verify_batch(
                conn,
                schema,
                batch_scope=XML_VERIFY_BATCH_SCOPE_DEFAULT,
            )
        if stranded_batch is not None:
            try:
                stranded_agreement_uuids = _load_xml_verify_batch_agreement_uuids(
                    client,
                    stranded_batch,
                )
            except Exception as e:
                context.log.warning(
                    "xml_verify_asset: failed to load agreement scope for unpulled batch %s: %s",
                    stranded_batch["batch_id"],
                    e,
                )
            else:
                target_scope = set(target_agreement_uuids)
                stranded_scope = set(stranded_agreement_uuids)
                if not target_scope or target_scope != stranded_scope:
                    return _resume_xml_verify_batch(
                        context,
                        engine,
                        db,
                        pipeline_config,
                        client,
                        schema=schema,
                        xml_table=xml_table,
                        batch_scope=XML_VERIFY_BATCH_SCOPE_DEFAULT,
                        batch_row=stranded_batch,
                        agreement_uuids=stranded_agreement_uuids,
                        log_prefix="xml_verify_asset",
                        hard_invalid_updated=0,
                    )

    if not target_agreement_uuids:
        context.log.info("xml_verify_asset: no upstream agreements from xml_asset.")
        run_post_asset_refresh(context, db, pipeline_config)
        return []
    if len(target_agreement_uuids) > agreement_batch_size:
        raise ValueError(
            "xml_verify_asset received more upstream agreements than xml_agreement_batch_size; "
            + "run-scoped XML verification accepts at most one upstream XML batch."
        )

    queue_q = text(canonical_fresh_xml_verify_queue_sql(schema, scoped=True)).bindparams(
        bindparam("auuids", expanding=True)
    )
    with engine.begin() as conn:
        eligible_uuids = conn.execute(
            queue_q,
            {"lim": agreement_batch_size, "auuids": target_agreement_uuids},
        ).scalars().all()
    if not eligible_uuids:
        context.log.info(
            "xml_verify_asset: no upstream-selected XML rows with status IS NULL, latest=1, and ai_repair_attempted=0."
        )
        run_post_asset_refresh(context, db, pipeline_config)
        return []

    select_q = text(
        f"""
        SELECT agreement_uuid, version, xml
        FROM {xml_table}
        WHERE agreement_uuid IN :auuids
          AND latest = 1
        ORDER BY agreement_uuid ASC
        """
    ).bindparams(bindparam("auuids", expanding=True))
    with engine.begin() as conn:
        rows = conn.execute(
            select_q,
            {"auuids": tuple(eligible_uuids)},
        ).mappings().fetchall()

    selected_for_verify = [str(row["agreement_uuid"]) for row in rows]

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
                    "reason_rows": [
                        {
                            "reason_code": XML_REASON_XML_PARSE_FAILURE,
                            "reason_detail": f"XML parse failure: {e}",
                            "page_uuid": None,
                        }
                    ],
                }
            )
            continue

        hard_rule_violations = find_hard_rule_violations(root)
        if hard_rule_violations:
            reason_rows: List[Dict[str, Any]] = []
            for violation in hard_rule_violations:
                if violation.page_uuids:
                    for page_uuid in violation.page_uuids:
                        reason_rows.append(
                            {
                                "reason_code": violation.reason_code,
                                "reason_detail": violation.reason_detail,
                                "page_uuid": page_uuid,
                            }
                        )
                else:
                    reason_rows.append(
                        {
                            "reason_code": violation.reason_code,
                            "reason_detail": violation.reason_detail,
                            "page_uuid": None,
                        }
                    )
            hard_invalid_rows.append(
                {
                    "agreement_uuid": agreement_uuid,
                    "version": version,
                    "reason_rows": reason_rows,
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
                    "reason_rows": [
                        {
                            "reason_code": XML_REASON_TAG_TREE_RENDER_FAILURE,
                            "reason_detail": f"Tag tree render failure: {e}",
                            "page_uuid": None,
                        }
                    ],
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
        xml_status_reasons_table = f"{schema}.xml_status_reasons"
        with engine.begin() as conn:
            for row in hard_invalid_rows:
                hard_invalid_updated += _set_xml_status_with_reasons(
                    conn,
                    xml_table,
                    xml_status_reasons_table,
                    agreement_uuid=str(row["agreement_uuid"]),
                    version=int(row["version"]),
                    status="invalid",
                    reason_rows=list(row["reason_rows"]),
                )

        sample_reasons = ", ".join(
            f"{r['agreement_uuid']}@v{r['version']}[{r['reason_rows'][0]['reason_code']}]: {r['reason_rows'][0]['reason_detail']}"
            for r in hard_invalid_rows[:3]
        )
        context.log.info(
            "xml_verify_asset: hard-rule invalidated %s XML rows before LLM. samples=%s",
            len(hard_invalid_rows),
            sample_reasons,
        )
        reason_counts: Dict[str, int] = {}
        for row in hard_invalid_rows:
            reason_codes = {
                str(reason_row["reason_code"])
                for reason_row in list(row["reason_rows"])
            }
            for reason_code in reason_codes:
                reason_counts[reason_code] = reason_counts.get(reason_code, 0) + 1
        context.log.info(
            "xml_verify_asset: hard-rule reason counts=%s",
            reason_counts,
        )

    if not lines:
        context.log.info(
            "xml_verify_asset: no LLM submissions required after hard-rule checks; hard_invalid_updated=%s",
            hard_invalid_updated,
        )
        run_post_asset_refresh(context, db, pipeline_config)
        return selected_for_verify

    llm_targets = sorted({_parse_custom_id(str(line["custom_id"])) for line in lines})
    if not llm_targets:
        raise ValueError("xml_verify_asset: no (agreement_uuid, version) targets derived from LLM lines.")
    verify_batch_key = agreement_version_batch_key(llm_targets)
    context.log.info(
        "xml_verify_asset: selected agreements=%s, llm_requests=%s, hard_invalid=%s",
        len(selected_for_verify),
        len(lines),
        len(hard_invalid_rows),
    )

    if resume_openai_batches:
        with engine.begin() as conn:
            existing_batch = _fetch_unpulled_xml_verify_batch(
                conn,
                schema,
                batch_scope=XML_VERIFY_BATCH_SCOPE_DEFAULT,
                batch_key=verify_batch_key,
            )
        if existing_batch is not None:
            context.log.info(
                "xml_verify_asset: resuming matching unpulled batch %s for batch_key=%s.",
                existing_batch["batch_id"],
                verify_batch_key[:12],
            )
            return _resume_xml_verify_batch(
                context,
                engine,
                db,
                pipeline_config,
                client,
                schema=schema,
                xml_table=xml_table,
                batch_scope=XML_VERIFY_BATCH_SCOPE_DEFAULT,
                batch_row=existing_batch,
                agreement_uuids=selected_for_verify,
                log_prefix="xml_verify_asset",
                hard_invalid_updated=hard_invalid_updated,
            )

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
            batch_scope=XML_VERIFY_BATCH_SCOPE_DEFAULT,
            batch_key=verify_batch_key,
        )
    context.log.info(
        f"xml_verify_asset: created batch {batch.id} with {len(lines)} requests; polling until complete."
    )

    final_batch = poll_batch_until_terminal(
        context,
        client,
        batch.id,
        log_prefix="xml_verify_asset",
    )
    with engine.begin() as conn:
        _upsert_xml_verify_batch_row(
            conn,
            schema,
            batch=final_batch,
            completion_window=completion_window,
            request_total=len(lines),
            batch_scope=XML_VERIFY_BATCH_SCOPE_DEFAULT,
            batch_key=verify_batch_key,
        )

    if final_batch.status != "completed":
        context.log.warning(
            f"xml_verify_asset: batch {final_batch.id} ended with status={final_batch.status}; no status updates applied."
        )
        with engine.begin() as conn:
            _mark_xml_verify_batch_pulled(conn, schema, final_batch.id)
        run_post_asset_refresh(context, db, pipeline_config)
        return selected_for_verify

    updated, parse_errors = _apply_xml_verify_batch_output(
        context=context,
        engine=engine,
        client=client,
        xml_table=xml_table,
        xml_status_reasons_table=f"{schema}.xml_status_reasons",
        batch=final_batch,
        log_prefix="xml_verify_asset",
    )
    with engine.begin() as conn:
        _mark_xml_verify_batch_pulled(conn, schema, final_batch.id)
    context.log.info(
        f"xml_verify_asset: batch {final_batch.id} completed; updated={updated}, parse_errors={parse_errors}, hard_invalid_updated={hard_invalid_updated}"
    )

    run_post_asset_refresh(context, db, pipeline_config)
    return selected_for_verify
