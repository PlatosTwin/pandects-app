"""Assemble tagged sections into XML documents and verify XML tree structure."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false

import io
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import pandas as pd

import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection

from etl.defs.c_tagging_asset import (
    ingestion_cleanup_a_tagging_asset,
    regular_ingest_tagging_asset,
    tagging_asset,
)
from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.f_xml import generate_xml
from etl.domain.xml_tag_repairs import (
    SectionGap,
    XMLTagRepairStats,
    apply_body_start_tag_repairs,
    insert_missing_section_heading_tags,
    normalize_no_space_section_prefixes,
    section_title_is_whole_number_article_heading,
    section_title_starts_with_same_article_decimal,
    split_leading_page_number_section_tags,
    split_combined_omitted_heading_tags,
    split_combined_missing_section_tags,
    unwrap_bare_number_section_tags,
    unwrap_inline_section_reference_tags,
    unwrap_standalone_section_label_tags,
    wrap_untagged_article_headings_before_first_sections,
)
from etl.utils.db_utils import upsert_xml
from etl.utils.batch_keys import agreement_version_batch_key
from etl.utils.logical_job_runs import (
    build_logical_batch_key,
    load_active_logical_run,
    load_active_scope_for_job,
    mark_logical_run_stage_completed,
    should_skip_managed_stage,
    start_or_resume_logical_run,
)
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

if TYPE_CHECKING:
    from openai import OpenAI


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
XML_REASON_ARTICLES_OUT_OF_ORDER = "articles_out_of_order"
XML_VERIFY_BATCH_SCOPE_DEFAULT = "default"
XML_VERIFY_BATCH_SCOPE_REPAIR = "repair"
XML_VERIFY_INSTRUCTIONS = (
    "Validate an agreement XML tag tree and return JSON with key `status` only. "
    "Apply these hard rules exactly: "
    "1) Empty articles are allowed, but at most one article may have zero sections, "
    "excluding articles whose titles mark them as omitted, deleted, reserved, or ***. "
    "2) Within each article, sections must start at 1 and be strictly sequential with no gaps, "
    "unless the XML's own tableOfContents shows the same local section-number jump or the "
    "expected/skipped section number is absent from the body text and titles. "
    "3) Every section title must begin with a valid section number prefix in one of these forms: "
    "`Section A.B ...`, `Sections A.B ...`, `SECTION A.B ...`, `A.B ...`, or `3A.B ...`; "
    "titles that do not match this are invalid. "
    "4) In `<body>`, the first structural child must be an `<article>`, and the first article must be Article I/1. "
    "If any hard rule fails, return status='invalid'. "
    "If all hard rules pass and structure is coherent, return status='verified'. "
    "If still unclear, return status=null."
)
ARTICLE_NUMBER_RE = re.compile(
    r"^\s*(?:ARTICLE\s+([IVXLCDM]+|\d+)|(?:SECTION\s+)?(\d+))(?=\b|[\s.:)\-])",
    re.IGNORECASE,
)
ARTICLE_A_NUMBER_RE = re.compile(
    r"^\s*(?:ARTICLE\s+(?P<article_number>[IVXLCDM]+|\d+)A|(?:SECTION\s+)?(?P<numeric_number>\d+)A)(?=\b|[\s.:)\-])",
    re.IGNORECASE,
)
# Handles separator variants of A-suffix articles: III-A, III.A, V(B), 3 C.
# The space-letter form (3 C) requires a word boundary after the letter so it
# doesn't greedily match the first letter of a following word (e.g. "3 CONDITIONS").
ARTICLE_A_VARIANT_RE = re.compile(
    r"^\s*ARTICLE\s+(?P<number>[IVXLCDM]+|\d+)(?:[-.](?P<dash>[A-Z])|\((?P<paren>[A-Z])\)|[ \t]+(?P<space>[A-Z]))(?=\b|[\s.:)\-]|$)",
    re.IGNORECASE,
)
SECTION_NUMBER_RE = re.compile(
    r"^\s*(?:SECTIONS?\s+)?(?P<article>\d+)A?\s*\.\s*(?P<section>\d+)",
    re.IGNORECASE,
)
TOC_SECTION_NUMBER_RE = re.compile(
    r"(?<!\d)(?:SECTIONS?\s+)?(?P<article>\d+)A?\s*\.\s*(?P<section>\d+)(?!\s*\.\s*\d)",
    re.IGNORECASE,
)
SECTION_NONSEQ_DETAIL_RE = re.compile(
    r"Non-sequential section number in article (?P<title>.+?): expected (?P<expected>\d+), found (?P<found>\d+)\."
)
TAGGED_SECTION_TAG_RE = re.compile(r"<section>(.*?)</section>", re.IGNORECASE | re.DOTALL)
TAGGED_STRUCTURAL_TAG_RE = re.compile(
    r"<(?P<tag>article|section)>(?P<title>.*?)</(?P=tag)>",
    re.IGNORECASE | re.DOTALL,
)
BARE_ARTICLE_MARKER_RE = re.compile(
    r"^\s*ARTICLE\s+(?P<number>[IVXLCDM]+|\d+)\.?\s*$",
    re.IGNORECASE,
)
SPLIT_SECTION_NUMBER_DIGIT_RE = re.compile(
    (
        r"^\s*(?P<label>(?:SECTION|Section)\s+)?(?P<article>\d+)(?P<dot>\s*\.\s*)"
        r"(?P<head>\d+)(?P<gap>\s+)(?P<tail>\d+)(?P<after>\s+.+)$"
    ),
    re.DOTALL,
)
SPLIT_ARTICLE_NUMBER_DIGIT_RE = re.compile(
    (
        r"^\s*(?P<label>(?:SECTION|Section)\s+)(?P<head>\d+)(?P<gap>\s+)"
        r"(?P<tail>\d+)(?P<dot>\s*\.\s*)(?P<section>\d+)(?P<after>(?:\s+.+|$))$"
    ),
    re.DOTALL,
)
OMITTED_EMPTY_ARTICLE_RE = re.compile(
    r"(?:\b(?:intentionally\s+)?(?:omitted|deleted|reserved)\b|\*{3,})",
    re.IGNORECASE,
)
ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


@dataclass(frozen=True)
class XMLHardRuleViolation:
    reason_code: str
    reason_detail: str
    page_uuids: Tuple[str, ...]


def _oai_client() -> "OpenAI":
    from openai import OpenAI

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
    client: "OpenAI",
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
    client: "OpenAI",
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


def _fetch_latest_verified_agreement_uuids(
    conn: Connection,
    *,
    xml_table: str,
    agreement_uuids: List[str],
) -> List[str]:
    if not agreement_uuids:
        return []
    rows = conn.execute(
        text(
            f"""
            SELECT agreement_uuid
            FROM {xml_table}
            WHERE agreement_uuid IN :auuids
              AND latest = 1
              AND status = 'verified'
            ORDER BY agreement_uuid ASC
            """
        ).bindparams(bindparam("auuids", expanding=True)),
        {"auuids": tuple(sorted(set(agreement_uuids)))},
    ).scalars().all()
    return [str(row) for row in rows]


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
    if match is None:
        a_match = ARTICLE_A_NUMBER_RE.match(title)
        if a_match is None:
            return None
        token = a_match.group("article_number") or a_match.group("numeric_number")
        if token is None:
            return None
        if token.isdigit():
            return int(token)
        return _roman_to_int(token)
    else:
        token = match.group(1) or match.group(2)
    if token is None:
        return None
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


def _article_sort_key(title: str) -> float | None:
    """Return a sortable float for an article title.

    Plain articles (I, II, 3) map to their integer value.
    A-suffix articles map to base + 0.5 — this covers all four forms:
      - no separator:  IIIA, 3A        (ARTICLE_A_NUMBER_RE)
      - hyphen/dot:    III-A, III.A    (ARTICLE_A_VARIANT_RE)
      - parenthesised: V(B), V(C)      (ARTICLE_A_VARIANT_RE)
      - space-letter:  3 C, 3 D        (ARTICLE_A_VARIANT_RE)
    Returns None when the title cannot be parsed.
    """
    a_match = ARTICLE_A_NUMBER_RE.match(title)
    if a_match is not None:
        token = a_match.group("article_number") or a_match.group("numeric_number")
        if token is None:
            return None
        base = int(token) if token.isdigit() else _roman_to_int(token)
        if base is None:
            return None
        return base + 0.5
    v_match = ARTICLE_A_VARIANT_RE.match(title)
    if v_match is not None:
        token = v_match.group("number")
        base = int(token) if token.isdigit() else _roman_to_int(token)
        if base is None:
            return None
        return base + 0.5
    match = ARTICLE_NUMBER_RE.match(title)
    if match is not None:
        token = match.group(1) or match.group(2)
        if token is None:
            return None
        n = int(token) if token.isdigit() else _roman_to_int(token)
        if n is None:
            return None
        return float(n)
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


def _article_uses_local_one_section_numbering(section_children: List[ET.Element]) -> bool:
    if not section_children:
        return False
    for expected_section_num, section_elem in enumerate(section_children, start=1):
        parsed_numbers = _extract_section_numbers(section_elem.attrib.get("title", ""))
        if parsed_numbers != (1, expected_section_num):
            return False
    return True


def _article_title_marks_intentionally_empty(title: str) -> bool:
    return OMITTED_EMPTY_ARTICLE_RE.search(title) is not None


def _bare_article_marker_number(title: str) -> int | None:
    match = BARE_ARTICLE_MARKER_RE.match(title)
    if match is None:
        return None
    token = match.group("number")
    if token.isdigit():
        return int(token)
    return _roman_to_int(token)


def _article_has_substantive_non_section_content(article_elem: ET.Element) -> bool:
    for node in article_elem.iter():
        if node.tag in {"page", "pageUUID"}:
            continue
        text_value = (node.text or "").strip()
        if text_value:
            return True
        tail_value = (node.tail or "").strip()
        if tail_value:
            return True
    return False


def _article_text_starts_with_section_heading(
    article_elem: ET.Element,
    *,
    article_num: int | None,
) -> bool:
    if article_num is None:
        return False
    heading_pattern = re.compile(
        rf"^\s*(?:SECTIONS?\s+)?{article_num}\s*\.\s*\d+\s*\.?(?:\s+|$)",
        re.IGNORECASE,
    )
    for node in article_elem.iter():
        if node.tag in {"page", "pageUUID", "section"}:
            continue
        text_value = (node.text or "").strip()
        if text_value and heading_pattern.match(text_value):
            return True
        tail_value = (node.tail or "").strip()
        if tail_value and heading_pattern.match(tail_value):
            return True
    return False


def _iter_nonempty_text_nodes(root: ET.Element) -> List[str]:
    values: List[str] = []
    for node in root.iter():
        if node.tag == "pageUUID":
            continue
        text_value = (node.text or "").strip()
        if text_value:
            values.append(text_value)
        tail_value = (node.tail or "").strip()
        if tail_value:
            values.append(tail_value)
    return values


def extract_toc_section_sequences(root: ET.Element) -> Dict[int, List[int]]:
    toc = root.find("tableOfContents")
    if toc is None:
        return {}
    sequences: Dict[int, List[int]] = {}
    toc_text = "\n".join(_iter_nonempty_text_nodes(toc))
    for match in TOC_SECTION_NUMBER_RE.finditer(toc_text):
        article_num = int(match.group("article"))
        section_num = int(match.group("section"))
        sequences.setdefault(article_num, []).append(section_num)
    return sequences


def extract_body_section_sequences(root: ET.Element) -> Dict[int, List[int]]:
    body = root.find("body")
    if body is None:
        return {}
    sequences: Dict[int, List[int]] = {}
    for article_elem in [child for child in list(body) if child.tag == "article"]:
        article_num = _extract_article_number_from_elem(article_elem)
        if article_num is None:
            continue
        article_sequence: List[int] = []
        for section_elem in [child for child in list(article_elem) if child.tag == "section"]:
            section_title = section_elem.attrib.get("title", "")
            parsed_numbers = _extract_section_numbers(section_title)
            if parsed_numbers is None:
                continue
            section_article_num, section_num = parsed_numbers
            if section_article_num != article_num:
                continue
            article_sequence.append(section_num)
        if article_sequence:
            sequences[article_num] = article_sequence
    return sequences


def extract_body_page_uuid_article_map(root: ET.Element) -> Dict[str, Tuple[int, ...]]:
    body = root.find("body")
    if body is None:
        return {}
    page_articles: Dict[str, List[int]] = {}
    for article_elem in [child for child in list(body) if child.tag == "article"]:
        article_num = _extract_article_number_from_elem(article_elem)
        if article_num is None:
            continue
        article_page_uuids = _merge_page_uuid_targets(
            _collect_page_uuids(article_elem),
            *[_collect_page_uuids(section_elem) for section_elem in list(article_elem) if section_elem.tag == "section"],
        )
        for page_uuid in article_page_uuids:
            article_nums = page_articles.setdefault(page_uuid, [])
            if article_num not in article_nums:
                article_nums.append(article_num)
    return {
        page_uuid: tuple(article_nums)
        for page_uuid, article_nums in page_articles.items()
    }


def toc_consistent_article_numbers(root: ET.Element) -> set[int]:
    toc_sequences = extract_toc_section_sequences(root)
    body_sequences = extract_body_section_sequences(root)
    consistent_articles: set[int] = set()
    for article_num, body_sequence in body_sequences.items():
        toc_sequence = toc_sequences.get(article_num)
        if toc_sequence is None:
            continue
        if toc_sequence == body_sequence:
            consistent_articles.add(article_num)
    return consistent_articles


def toc_has_matching_local_gap(
    toc_sequences: Dict[int, List[int]],
    *,
    article_num: int | None,
    expected_section_num: int,
    found_section_num: int,
) -> bool:
    if article_num is None or found_section_num <= expected_section_num:
        return False
    toc_sequence = toc_sequences.get(article_num)
    if not toc_sequence:
        return False
    skipped = set(range(expected_section_num, found_section_num))
    if any(section_num in skipped for section_num in toc_sequence):
        return False
    if expected_section_num == 1:
        return found_section_num in toc_sequence
    previous_section_num = expected_section_num - 1
    return any(
        previous == previous_section_num and current == found_section_num
        for previous, current in zip(toc_sequence, toc_sequence[1:])
    )


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


def _section_title_has_additional_numbering(title: str) -> bool:
    match = SECTION_NUMBER_RE.match(title)
    if match is None:
        return False
    suffix = title[match.end() :]
    return re.search(r"\b(?:SECTIONS?\s+)?\d+\s*\.\s*\d+\b", suffix, re.IGNORECASE) is not None


def _merge_page_uuid_targets(*page_uuid_groups: Tuple[str, ...]) -> Tuple[str, ...]:
    merged: List[str] = []
    seen: set[str] = set()
    for group in page_uuid_groups:
        for page_uuid in group:
            if page_uuid in seen:
                continue
            seen.add(page_uuid)
            merged.append(page_uuid)
    return tuple(merged)


def _section_title_mentions_number(
    title: str,
    *,
    article_num: int,
    section_num: int,
) -> bool:
    match = SECTION_NUMBER_RE.match(title)
    search_text = title[match.end():] if match is not None else title
    return _section_number_search_pattern(article_num, section_num).search(search_text) is not None


def _section_number_search_pattern(article_num: int, section_num: int) -> re.Pattern[str]:
    pattern = rf"(?<![\d.])(?:SECTIONS?\s+)?{article_num}\s*\.\s*0*{section_num}"
    pattern += r"(?!\s*\.\s*\d)(?=\b|[\s.;:,\)\]\-])"
    return re.compile(pattern, re.IGNORECASE)


def _section_body_mentions_number(
    section_elem: ET.Element,
    *,
    article_num: int,
    section_num: int,
) -> bool:
    pattern = _section_number_search_pattern(article_num, section_num)
    for node in section_elem.iter():
        if node.tag in {"page", "pageUUID"}:
            continue
        text_value = (node.text or "").strip()
        if text_value and pattern.search(text_value):
            return True
    return False


def _section_body_number_mention_page_uuids(
    section_elem: ET.Element,
    *,
    article_num: int,
    section_num: int,
) -> Tuple[str, ...]:
    pattern = _section_number_search_pattern(article_num, section_num)
    page_uuids: List[str] = []

    def add_page_uuids(values: Tuple[str, ...]) -> None:
        for page_uuid in values:
            if page_uuid not in page_uuids:
                page_uuids.append(page_uuid)

    for idx, child in enumerate(list(section_elem)):
        if child.tag in {"page", "pageUUID"}:
            continue
        mention_found = False
        for node in child.iter():
            if node.tag in {"page", "pageUUID"}:
                continue
            text_value = (node.text or "").strip()
            if text_value and pattern.search(text_value):
                mention_found = True
                break
        tail_value = (child.tail or "").strip()
        if tail_value and pattern.search(tail_value):
            mention_found = True
        if mention_found:
            add_page_uuids(_nearest_page_uuid_for_child(section_elem, idx))

    return tuple(page_uuids)


def _text_starts_with_article_heading(text_value: str, article_num: int) -> bool:
    if _extract_article_number(text_value) != article_num:
        return False
    return _extract_section_numbers(text_value) is None


def _nearest_page_uuid_for_child(parent: ET.Element, child_index: int) -> Tuple[str, ...]:
    children = list(parent)
    child_page_uuids = _collect_page_uuids(children[child_index])
    if child_page_uuids:
        return child_page_uuids

    for following in children[child_index + 1 :]:
        if following.tag == "pageUUID":
            page_uuid = (following.text or "").strip()
            if page_uuid:
                return (page_uuid,)

    for previous in reversed(children[:child_index]):
        if previous.tag == "pageUUID":
            page_uuid = (previous.text or "").strip()
            if page_uuid:
                return (page_uuid,)

    return _collect_page_uuids(parent)


def _article_heading_page_uuids_in_section(
    section_elem: ET.Element,
    *,
    article_num: int,
) -> Tuple[str, ...]:
    page_uuids: List[str] = []
    for idx, child in enumerate(list(section_elem)):
        if child.tag in {"page", "pageUUID"}:
            continue
        text_value = (child.text or "").strip()
        if not text_value or not _text_starts_with_article_heading(text_value, article_num):
            continue
        for page_uuid in _nearest_page_uuid_for_child(section_elem, idx):
            if page_uuid not in page_uuids:
                page_uuids.append(page_uuid)
    return tuple(page_uuids)


def _toc_marker_page_uuids_in_section(section_elem: ET.Element) -> Tuple[str, ...]:
    page_uuids: List[str] = []
    for idx, child in enumerate(list(section_elem)):
        if child.tag in {"page", "pageUUID"}:
            continue
        text_value = (child.text or "").strip()
        if not text_value or re.search(r"\btable\s+of\s+contents\b", text_value, re.IGNORECASE) is None:
            continue
        for page_uuid in _nearest_page_uuid_for_child(section_elem, idx):
            if page_uuid not in page_uuids:
                page_uuids.append(page_uuid)
    return tuple(page_uuids)


def _target_section_article_mismatch_page_uuids(
    *,
    previous_section_elem: ET.Element | None,
    section_article_num: int,
    section_page_uuids: Tuple[str, ...],
) -> Tuple[str, ...]:
    if previous_section_elem is None:
        return section_page_uuids
    previous_article_heading_page_uuids = _article_heading_page_uuids_in_section(
        previous_section_elem,
        article_num=section_article_num,
    )
    previous_toc_marker_page_uuids = _toc_marker_page_uuids_in_section(previous_section_elem)
    return _merge_page_uuid_targets(
        previous_article_heading_page_uuids,
        previous_toc_marker_page_uuids,
        section_page_uuids,
    )


def _iter_body_number_search_values(root: ET.Element) -> List[str]:
    body = root.find("body")
    if body is None:
        return []
    values: List[str] = []
    for node in body.iter():
        if node.tag == "pageUUID":
            continue
        title = node.attrib.get("title")
        if title:
            values.append(title)
        text_value = (node.text or "").strip()
        if text_value:
            values.append(text_value)
        tail_value = (node.tail or "").strip()
        if tail_value:
            values.append(tail_value)
    return values


def _body_mentions_section_number(
    root: ET.Element,
    *,
    article_num: int,
    section_num: int,
) -> bool:
    pattern = _section_number_search_pattern(article_num, section_num)
    return any(pattern.search(value) is not None for value in _iter_body_number_search_values(root))


def _expected_or_skipped_section_numbers_absent_from_agreement(
    root: ET.Element,
    *,
    article_num: int | None,
    expected_section_num: int,
    found_section_num: int,
) -> bool:
    if article_num is None:
        return False
    if found_section_num <= expected_section_num:
        return not _body_mentions_section_number(
            root,
            article_num=article_num,
            section_num=expected_section_num,
        )
    return all(
        not _body_mentions_section_number(
            root,
            article_num=article_num,
            section_num=section_num,
        )
        for section_num in range(expected_section_num, found_section_num)
    )


def _target_section_non_sequential_page_uuids(
    *,
    previous_section_elem: ET.Element | None,
    current_section_elem: ET.Element,
    section_title: str,
    section_article_num: int,
    expected_section_num: int,
    found_section_num: int,
    current_page_uuids: Tuple[str, ...],
    previous_page_uuids: Tuple[str, ...],
) -> Tuple[str, ...]:
    """
    Target the page most likely to contain the root cause of a numbering gap.

    When numbering jumps forward, the preceding section is often the malformed one
    that absorbed the missing heading during tagging/OCR. For duplicates or
    backward jumps, the current section is still the better target.
    """
    if _section_title_has_additional_numbering(section_title) or _section_title_mentions_number(
        section_title,
        article_num=section_article_num,
        section_num=expected_section_num,
    ):
        return current_page_uuids
    if found_section_num > expected_section_num and previous_page_uuids:
        previous_mention_page_uuids = (
            _section_body_number_mention_page_uuids(
                previous_section_elem,
                article_num=section_article_num,
                section_num=expected_section_num,
            )
            if previous_section_elem is not None
            else ()
        )
        previous_mentions_expected = bool(previous_mention_page_uuids)
        current_mentions_expected = _section_body_mentions_number(
            current_section_elem,
            article_num=section_article_num,
            section_num=expected_section_num,
        )
        if previous_mentions_expected and not current_mentions_expected:
            return previous_mention_page_uuids
        if previous_mentions_expected and current_mentions_expected:
            return _merge_page_uuid_targets(previous_mention_page_uuids, current_page_uuids)
        if current_mentions_expected:
            return current_page_uuids
        return _merge_page_uuid_targets(previous_page_uuids, current_page_uuids)
    return current_page_uuids


def find_hard_rule_violations(root: ET.Element) -> List[XMLHardRuleViolation]:
    violations: List[XMLHardRuleViolation] = []
    toc_sequences = extract_toc_section_sequences(root)
    toc_consistent_articles = toc_consistent_article_numbers(root)
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

    prev_key: float | None = None
    for article_elem in articles:
        article_title = article_elem.attrib.get("title", "")
        key = _article_sort_key(article_title)
        if key is None:
            continue
        # Allow multiple A-suffix sub-articles at the same base (e.g. III-A, III-B
        # or V(B), V(C)) — their keys are equal non-integers (base + 0.5).
        # Still flag duplicate integer keys, which indicate truly repeated articles.
        is_a_suffix = key != float(int(key))
        out_of_order = prev_key is not None and (
            key < prev_key or (key == prev_key and not is_a_suffix)
        )
        if out_of_order:
            violations.append(
                XMLHardRuleViolation(
                    reason_code=XML_REASON_ARTICLES_OUT_OF_ORDER,
                    reason_detail=f"Article out of order: {article_title[:200]!r} (key {key}) follows key {prev_key}.",
                    page_uuids=_collect_page_uuids(article_elem),
                )
            )
        else:
            prev_key = key

    empty_article_count = 0
    empty_article_page_uuids: List[str] = []
    for article_elem in articles:
        article_title = article_elem.attrib.get("title", "")
        article_num = _extract_article_number_from_elem(article_elem)
        section_children = [c for c in list(article_elem) if c.tag == "section"]
        if not section_children:
            if _article_title_marks_intentionally_empty(article_title):
                continue
            if (
                _article_has_substantive_non_section_content(article_elem)
                and not _article_text_starts_with_section_heading(
                    article_elem,
                    article_num=article_num,
                )
            ):
                continue
            empty_article_count += 1
            for page_uuid in _collect_page_uuids(article_elem):
                if page_uuid not in empty_article_page_uuids:
                    empty_article_page_uuids.append(page_uuid)
            continue

        expected_section_num = 1
        previous_section_elem: ET.Element | None = None
        previous_section_page_uuids: Tuple[str, ...] = ()
        uses_local_one_section_numbering = _article_uses_local_one_section_numbering(
            section_children
        )
        for section_elem in section_children:
            section_title = section_elem.attrib.get("title", "")
            section_page_uuids = _collect_page_uuids(section_elem)
            parsed_numbers = _extract_section_numbers(section_title)
            if parsed_numbers is None:
                violations.append(
                    XMLHardRuleViolation(
                        reason_code=XML_REASON_SECTION_TITLE_INVALID_NUMBERING,
                        reason_detail=f"Section title is not a valid numbered section: {section_title[:200]!r} in article {article_title[:200]!r}.",
                        page_uuids=section_page_uuids,
                    )
                )
                continue

            section_article_num, section_num = parsed_numbers
            local_one_numbering_mismatch = (
                article_num is not None
                and article_num != 1
                and section_article_num == 1
                and uses_local_one_section_numbering
            )
            if (
                article_num is not None
                and section_article_num != article_num
                and not local_one_numbering_mismatch
            ):
                mismatch_page_uuids = _target_section_article_mismatch_page_uuids(
                    previous_section_elem=previous_section_elem,
                    section_article_num=section_article_num,
                    section_page_uuids=section_page_uuids,
                )
                violations.append(
                    XMLHardRuleViolation(
                        reason_code=XML_REASON_SECTION_ARTICLE_MISMATCH,
                        reason_detail=f"Section {section_article_num}.{section_num} does not match article {article_num} ({article_title[:200]!r}).",
                        page_uuids=mismatch_page_uuids,
                    )
                )
            if section_num != expected_section_num:
                toc_suppresses_nonseq = (
                    article_num in toc_consistent_articles
                    or toc_has_matching_local_gap(
                        toc_sequences,
                        article_num=article_num,
                        expected_section_num=expected_section_num,
                        found_section_num=section_num,
                    )
                )
                agreement_absence_suppresses_nonseq = _expected_or_skipped_section_numbers_absent_from_agreement(
                    root,
                    article_num=article_num,
                    expected_section_num=expected_section_num,
                    found_section_num=section_num,
                )
                if not toc_suppresses_nonseq and not agreement_absence_suppresses_nonseq:
                    violations.append(
                        XMLHardRuleViolation(
                            reason_code=XML_REASON_SECTION_NON_SEQUENTIAL,
                            reason_detail=f"Non-sequential section number in article {article_title[:200]!r}: expected {expected_section_num}, found {section_num}.",
                            page_uuids=_target_section_non_sequential_page_uuids(
                                previous_section_elem=previous_section_elem,
                                current_section_elem=section_elem,
                                section_title=section_title,
                                section_article_num=section_article_num,
                                expected_section_num=expected_section_num,
                                found_section_num=section_num,
                                current_page_uuids=section_page_uuids,
                                previous_page_uuids=previous_section_page_uuids,
                            ),
                        )
                    )
                expected_section_num = section_num + 1
            else:
                expected_section_num += 1
            previous_section_elem = section_elem
            previous_section_page_uuids = section_page_uuids

    if empty_article_count > 1:
        violations.append(
            XMLHardRuleViolation(
                reason_code=XML_REASON_TOO_MANY_EMPTY_ARTICLES,
                reason_detail=f"Too many empty articles: found {empty_article_count}, maximum allowed is 1.",
                page_uuids=tuple(empty_article_page_uuids),
            )
        )
    return violations


def _section_gaps_from_violations(
    violations: List[XMLHardRuleViolation],
) -> List[SectionGap]:
    gaps: List[SectionGap] = []
    for violation in violations:
        if violation.reason_code != XML_REASON_SECTION_NON_SEQUENTIAL:
            continue
        match = SECTION_NONSEQ_DETAIL_RE.search(violation.reason_detail)
        if not match:
            continue
        article_num = _extract_article_number(match.group("title").strip("'"))
        if article_num is None:
            continue
        gaps.append(
            SectionGap(
                article_num=article_num,
                expected=int(match.group("expected")),
                found=int(match.group("found")),
            )
        )
    return gaps


def _hard_rule_result_for_df(
    df: pd.DataFrame,
) -> Tuple[Counter[str], List[XMLHardRuleViolation]]:
    xml_rows, generation_failures = generate_xml(df)
    counts: Counter[str] = Counter()
    violations: List[XMLHardRuleViolation] = []
    if generation_failures:
        counts["xml_generation_failure"] += len(generation_failures)
    for xml_row in xml_rows:
        try:
            root = ET.fromstring(xml_row.xml)
        except ET.ParseError:
            counts[XML_REASON_XML_PARSE_FAILURE] += 1
            continue
        row_violations = find_hard_rule_violations(root)
        violations.extend(row_violations)
        counts.update(violation.reason_code for violation in row_violations)
    return counts, violations


def _safe_hard_rule_improvement(
    before_counts: Counter[str],
    after_counts: Counter[str],
) -> bool:
    if not before_counts:
        return False
    if set(after_counts) - set(before_counts):
        return False
    if any(after_counts[reason] > before_counts[reason] for reason in after_counts):
        return False
    return sum(after_counts.values()) < sum(before_counts.values())


def _source_page_type(row: pd.Series) -> object:
    gold_label = row.get("gold_label")
    if isinstance(gold_label, str) and gold_label.strip():
        return gold_label
    return row.get("source_page_type")


def _apply_tag_repair_to_body_rows(
    df: pd.DataFrame,
    repair_name: str,
    gaps: List[SectionGap],
) -> Tuple[pd.DataFrame, XMLTagRepairStats]:
    out = df.copy()
    total_stats = XMLTagRepairStats()
    for idx, row in out.iterrows():
        if _source_page_type(row) != "body":
            continue
        tagged_output = row.get("tagged_output")
        if not isinstance(tagged_output, str) or not tagged_output.strip():
            continue
        if repair_name == "split_omitted":
            repaired, stats = split_combined_omitted_heading_tags(tagged_output)
        elif repair_name == "split_combined":
            repaired, stats = split_combined_missing_section_tags(tagged_output, gaps)
        elif repair_name == "body_start":
            repaired, stats = apply_body_start_tag_repairs(tagged_output)
        elif repair_name == "unwrap_bare_number_sections":
            repaired, stats = unwrap_bare_number_section_tags(tagged_output)
        elif repair_name == "unwrap_standalone_section_labels":
            repaired, stats = unwrap_standalone_section_label_tags(tagged_output)
        elif repair_name == "normalize_no_space_section_prefixes":
            repaired, stats = normalize_no_space_section_prefixes(tagged_output)
        elif repair_name == "split_leading_page_number_sections":
            repaired, stats = split_leading_page_number_section_tags(tagged_output)
        elif repair_name == "unwrap_inline_section_references":
            repaired, stats = unwrap_inline_section_reference_tags(tagged_output)
        elif repair_name == "insert_missing":
            repaired, stats = insert_missing_section_heading_tags(tagged_output, gaps)
        elif repair_name == "wrap_untagged_article":
            repaired, stats = wrap_untagged_article_headings_before_first_sections(tagged_output)
        else:
            raise ValueError(f"Unknown XML tag repair: {repair_name}")
        total_stats.update(stats)
        if repaired != tagged_output:
            out.at[idx, "tagged_output"] = repaired
    return out, total_stats


def _apply_cross_row_article_heading_repairs(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, XMLTagRepairStats]:
    out = df.copy()
    stats = XMLTagRepairStats()
    body_rows = out[out.apply(_source_page_type, axis=1) == "body"]
    if body_rows.empty:
        return out, stats
    if "page_order" in body_rows.columns:
        body_rows = body_rows.sort_values(["page_order", "page_uuid"], kind="stable")

    section_positions: list[tuple[object, int, int, str]] = []
    for idx, row in body_rows.iterrows():
        tagged_output = row.get("tagged_output")
        if not isinstance(tagged_output, str) or not tagged_output.strip():
            continue
        for match in TAGGED_SECTION_TAG_RE.finditer(tagged_output):
            section_positions.append((idx, match.start(), match.end(), match.group(1).strip()))

    replacements_by_idx: dict[object, dict[tuple[int, int], str]] = {}
    for current, following in zip(section_positions, section_positions[1:]):
        idx, start, end, title = current
        _next_idx, _next_start, _next_end, next_title = following
        stats.attempts["promote_cross_row_whole_number_sections_to_articles"] += 1
        article_num = section_title_is_whole_number_article_heading(title)
        if article_num is None:
            continue
        if not section_title_starts_with_same_article_decimal(next_title, article_num):
            continue
        replacements_by_idx.setdefault(idx, {})[(start, end)] = f"<article>{title}</article>"
        stats.applied["promote_cross_row_whole_number_sections_to_articles"] += 1

    for idx, replacements in replacements_by_idx.items():
        tagged_output = out.at[idx, "tagged_output"]
        if not isinstance(tagged_output, str):
            continue
        pieces: list[str] = []
        last = 0
        for (start, end), replacement in sorted(replacements.items()):
            pieces.append(tagged_output[last:start])
            pieces.append(replacement)
            last = end
        pieces.append(tagged_output[last:])
        out.at[idx, "tagged_output"] = "".join(pieces)

    return out, stats


def _heading_article_context_number(tag_name: str, title: str) -> int | None:
    if tag_name.lower() == "article":
        return _extract_article_number(title)
    article_num = section_title_is_whole_number_article_heading(title)
    if article_num is not None:
        return int(article_num)
    section_numbers = _extract_section_numbers(title)
    if section_numbers is not None:
        return section_numbers[0]
    return None


def _apply_article_sequence_heading_repairs(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, XMLTagRepairStats]:
    out = df.copy()
    stats = XMLTagRepairStats()
    body_rows = out[out.apply(_source_page_type, axis=1) == "body"]
    if body_rows.empty:
        return out, stats
    if "page_order" in body_rows.columns:
        body_rows = body_rows.sort_values(["page_order", "page_uuid"], kind="stable")

    heading_positions: list[tuple[object, int, int, str, str, int | None]] = []
    for idx, row in body_rows.iterrows():
        tagged_output = row.get("tagged_output")
        if not isinstance(tagged_output, str) or not tagged_output.strip():
            continue
        for match in TAGGED_STRUCTURAL_TAG_RE.finditer(tagged_output):
            tag_name = match.group("tag").lower()
            title = match.group("title").strip()
            heading_positions.append(
                (
                    idx,
                    match.start(),
                    match.end(),
                    tag_name,
                    title,
                    _heading_article_context_number(tag_name, title),
                )
            )

    replacements_by_idx: dict[object, dict[tuple[int, int], str]] = {}
    for pos, heading in enumerate(heading_positions):
        idx, start, end, tag_name, title, _article_context_num = heading
        if tag_name != "section":
            continue
        stats.attempts["promote_sequence_whole_number_sections_to_articles"] += 1
        article_num_text = section_title_is_whole_number_article_heading(title)
        if article_num_text is None:
            continue
        article_num = int(article_num_text)
        previous_article_num = (
            heading_positions[pos - 1][5]
            if pos > 0
            else None
        )
        next_article_num = (
            heading_positions[pos + 1][5]
            if pos + 1 < len(heading_positions)
            else None
        )
        follows_previous_article = previous_article_num == article_num - 1
        precedes_next_article = next_article_num == article_num + 1
        starts_sequence = previous_article_num is None and article_num == 1 and precedes_next_article
        follows_article_sequence_gap = (
            follows_previous_article
            and (next_article_num is None or next_article_num > article_num)
        )
        if not (precedes_next_article or starts_sequence or follows_article_sequence_gap):
            continue
        replacements_by_idx.setdefault(idx, {})[(start, end)] = f"<article>{title}</article>"
        stats.applied["promote_sequence_whole_number_sections_to_articles"] += 1

    for idx, replacements in replacements_by_idx.items():
        tagged_output = out.at[idx, "tagged_output"]
        if not isinstance(tagged_output, str):
            continue
        pieces: list[str] = []
        last = 0
        for (start, end), replacement in sorted(replacements.items()):
            pieces.append(tagged_output[last:start])
            pieces.append(replacement)
            last = end
        pieces.append(tagged_output[last:])
        out.at[idx, "tagged_output"] = "".join(pieces)

    return out, stats


def _split_section_digit_title_candidates(
    title: str,
    *,
    article_num: int,
) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []
    section_match = SPLIT_SECTION_NUMBER_DIGIT_RE.match(title)
    if section_match is not None and int(section_match.group("article")) == article_num:
        section_num = int(section_match.group("head") + section_match.group("tail"))
        candidates.append(
            (
                section_num,
                title[: section_match.start("gap")] + title[section_match.end("gap") :],
            )
        )

    article_match = SPLIT_ARTICLE_NUMBER_DIGIT_RE.match(title)
    if article_match is not None and int(article_match.group("head") + article_match.group("tail")) == article_num:
        candidates.append(
            (
                int(article_match.group("section")),
                title[: article_match.start("gap")] + title[article_match.end("gap") :],
            )
        )
    return candidates


def _apply_split_section_number_digit_repairs(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, XMLTagRepairStats]:
    out = df.copy()
    stats = XMLTagRepairStats()
    body_rows = out[out.apply(_source_page_type, axis=1) == "body"]
    if body_rows.empty:
        return out, stats
    if "page_order" in body_rows.columns:
        body_rows = body_rows.sort_values(["page_order", "page_uuid"], kind="stable")

    section_positions: list[tuple[object, int, int, str, int | None, int | None]] = []
    current_article_num: int | None = None
    for idx, row in body_rows.iterrows():
        tagged_output = row.get("tagged_output")
        if not isinstance(tagged_output, str) or not tagged_output.strip():
            continue
        for match in TAGGED_STRUCTURAL_TAG_RE.finditer(tagged_output):
            tag_name = match.group("tag").lower()
            title = match.group("title").strip()
            if tag_name == "article":
                current_article_num = _extract_article_number(title)
                continue
            parsed_section = _extract_section_numbers(title)
            parsed_section_num = (
                parsed_section[1]
                if parsed_section is not None and parsed_section[0] == current_article_num
                else None
            )
            section_positions.append(
                (
                    idx,
                    match.start(),
                    match.end(),
                    title,
                    current_article_num,
                    parsed_section_num,
                )
            )

    replacements_by_idx: dict[object, dict[tuple[int, int], str]] = {}
    for pos, section_position in enumerate(section_positions):
        idx, start, end, title, article_num, _parsed_section_num = section_position
        stats.attempts["join_split_section_number_digits"] += 1
        if article_num is None:
            continue

        previous_section_num: int | None = None
        for previous in reversed(section_positions[:pos]):
            if previous[4] != article_num:
                break
            if previous[5] is not None:
                previous_section_num = previous[5]
                break

        next_section_num: int | None = None
        for following in section_positions[pos + 1 :]:
            if following[4] != article_num:
                break
            if following[5] is not None:
                next_section_num = following[5]
                break

        if previous_section_num is None or next_section_num is None:
            continue
        for repaired_section_num, repaired_title in _split_section_digit_title_candidates(
            title,
            article_num=article_num,
        ):
            previous_supports_repair = previous_section_num == repaired_section_num - 1
            next_supports_repair = next_section_num == repaired_section_num + 1
            prior_sequence_allows_repair = previous_section_num < repaired_section_num
            following_sequence_allows_repair = next_section_num > repaired_section_num
            if not (
                (previous_supports_repair and following_sequence_allows_repair)
                or (next_supports_repair and prior_sequence_allows_repair)
            ):
                continue
            replacements_by_idx.setdefault(idx, {})[(start, end)] = (
                f"<section>{repaired_title}</section>"
            )
            stats.applied["join_split_section_number_digits"] += 1
            break

    for idx, replacements in replacements_by_idx.items():
        tagged_output = out.at[idx, "tagged_output"]
        if not isinstance(tagged_output, str):
            continue
        pieces: list[str] = []
        last = 0
        for (start, end), replacement in sorted(replacements.items()):
            pieces.append(tagged_output[last:start])
            pieces.append(replacement)
            last = end
        pieces.append(tagged_output[last:])
        out.at[idx, "tagged_output"] = "".join(pieces)

    return out, stats


def _merge_article_title_parts(left_title: str, right_title: str) -> str:
    left = left_title.strip()
    right = right_title.strip()
    if left.endswith("."):
        return f"{left} {right}"
    return f"{left}. {right}"


def _apply_split_article_title_repairs(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, XMLTagRepairStats]:
    out = df.copy()
    stats = XMLTagRepairStats()
    body_rows = out[out.apply(_source_page_type, axis=1) == "body"]
    if body_rows.empty:
        return out, stats
    if "page_order" in body_rows.columns:
        body_rows = body_rows.sort_values(["page_order", "page_uuid"], kind="stable")

    heading_positions: list[tuple[object, int, int, str, str]] = []
    for idx, row in body_rows.iterrows():
        tagged_output = row.get("tagged_output")
        if not isinstance(tagged_output, str) or not tagged_output.strip():
            continue
        for match in TAGGED_STRUCTURAL_TAG_RE.finditer(tagged_output):
            heading_positions.append(
                (
                    idx,
                    match.start(),
                    match.end(),
                    match.group("tag").lower(),
                    match.group("title").strip(),
                )
            )

    replacements_by_idx: dict[object, dict[tuple[int, int], str]] = {}
    for pos, heading in enumerate(heading_positions[:-2]):
        idx, start, end, tag_name, title = heading
        stats.attempts["merge_split_article_title_tags"] += 1
        if tag_name != "article":
            continue
        article_num = _bare_article_marker_number(title)
        if article_num is None:
            continue
        next_idx, next_start, next_end, next_tag_name, next_title = heading_positions[pos + 1]
        if next_tag_name != "article":
            continue
        if _extract_article_number(next_title) is not None:
            continue
        following_tag_name = heading_positions[pos + 2][3]
        following_title = heading_positions[pos + 2][4]
        if following_tag_name != "section":
            continue
        following_section_numbers = _extract_section_numbers(following_title)
        if following_section_numbers is None or following_section_numbers[0] != article_num:
            continue

        replacements_by_idx.setdefault(idx, {})[(start, end)] = (
            f"<article>{_merge_article_title_parts(title, next_title)}</article>"
        )
        replacements_by_idx.setdefault(next_idx, {})[(next_start, next_end)] = ""
        stats.applied["merge_split_article_title_tags"] += 1

    for idx, replacements in replacements_by_idx.items():
        tagged_output = out.at[idx, "tagged_output"]
        if not isinstance(tagged_output, str):
            continue
        pieces: list[str] = []
        last = 0
        for (start, end), replacement in sorted(replacements.items()):
            pieces.append(tagged_output[last:start])
            pieces.append(replacement)
            last = end
        pieces.append(tagged_output[last:])
        out.at[idx, "tagged_output"] = "".join(pieces)

    return out, stats


def _apply_safe_xml_tag_repairs_to_df(
    df: pd.DataFrame,
    *,
    context: AssetExecutionContext | None = None,
    log_prefix: str = "xml_asset",
) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    accepted_counts: Counter[str] = Counter()
    rejected_counts: Counter[str] = Counter()
    out, direct_text_stats = _apply_tag_repair_to_body_rows(
        out,
        "normalize_no_space_section_prefixes",
        [],
    )

    for agreement_uuid in out["agreement_uuid"].drop_duplicates().tolist():
        agreement_mask = out["agreement_uuid"] == agreement_uuid
        current_df = out.loc[agreement_mask].copy()
        current_counts, current_violations = _hard_rule_result_for_df(current_df)
        if not current_counts:
            continue

        for repair_name in (
            "split_omitted",
            "split_combined",
            "body_start",
            "unwrap_bare_number_sections",
            "unwrap_standalone_section_labels",
            "split_leading_page_number_sections",
            "unwrap_inline_section_references",
            "article_sequence",
            "join_split_digits",
            "split_article_title",
            "cross_row_article",
            "wrap_untagged_article",
            "insert_missing",
        ):
            gaps = _section_gaps_from_violations(current_violations)
            if repair_name in {"split_combined", "insert_missing"} and not gaps:
                continue
            if repair_name == "cross_row_article":
                candidate_df, stats = _apply_cross_row_article_heading_repairs(current_df)
                if stats.applied:
                    candidate_df, followup_stats = _apply_tag_repair_to_body_rows(
                        candidate_df,
                        "wrap_untagged_article",
                        gaps,
                    )
                    stats.update(followup_stats)
                    candidate_df, followup_stats = _apply_tag_repair_to_body_rows(
                        candidate_df,
                        "split_omitted",
                        gaps,
                    )
                    stats.update(followup_stats)
            elif repair_name == "article_sequence":
                candidate_df, stats = _apply_article_sequence_heading_repairs(current_df)
                if stats.applied:
                    candidate_df, followup_stats = _apply_tag_repair_to_body_rows(
                        candidate_df,
                        "wrap_untagged_article",
                        gaps,
                    )
                    stats.update(followup_stats)
                    candidate_df, followup_stats = _apply_tag_repair_to_body_rows(
                        candidate_df,
                        "split_omitted",
                        gaps,
                    )
                    stats.update(followup_stats)
            elif repair_name == "join_split_digits":
                candidate_df, stats = _apply_split_section_number_digit_repairs(current_df)
            elif repair_name == "split_article_title":
                candidate_df, stats = _apply_split_article_title_repairs(current_df)
            else:
                candidate_df, stats = _apply_tag_repair_to_body_rows(
                    current_df,
                    repair_name,
                    gaps,
                )
            if not stats.applied:
                continue
            candidate_counts, candidate_violations = _hard_rule_result_for_df(candidate_df)
            if _safe_hard_rule_improvement(current_counts, candidate_counts):
                current_df = candidate_df
                current_counts = candidate_counts
                current_violations = candidate_violations
                accepted_counts[repair_name] += 1
            else:
                rejected_counts[repair_name] += 1

        out.loc[agreement_mask, "tagged_output"] = current_df["tagged_output"]

    if context is not None and (direct_text_stats.applied or accepted_counts or rejected_counts):
        context.log.info(
            "%s: xml tag repairs direct_text=%s accepted=%s rejected=%s",
            log_prefix,
            dict(direct_text_stats.applied),
            dict(accepted_counts),
            dict(rejected_counts),
        )
    return out


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


def _build_xml_verify_toc_context(root: ET.Element) -> str | None:
    toc_sequences = extract_toc_section_sequences(root)
    if not toc_sequences:
        return None
    lines: List[str] = []
    for article_num in sorted(toc_sequences):
        sequence = toc_sequences[article_num]
        if not sequence:
            continue
        numbering = ", ".join(f"{article_num}.{section_num}" for section_num in sequence)
        lines.append(f"Article {article_num}: {numbering}")
    if not lines:
        return None
    return "XML tableOfContents numbering:\n" + "\n".join(lines)


def _build_xml_verify_batch_request_body(
    *,
    custom_id: str,
    tag_tree: str,
    model: str,
    toc_context: str | None = None,
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
    input_text = f"XML tag tree:\n{tag_tree}"
    if toc_context is not None and toc_context.strip():
        input_text += f"\n\n{toc_context.strip()}"
    body: Dict[str, Any] = {
        "model": model,
        "instructions": XML_VERIFY_INSTRUCTIONS,
        "input": input_text,
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


def _normalized_reason_row_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    reason_code = str(row["reason_code"])
    reason_detail = None if row.get("reason_detail") is None else str(row["reason_detail"])
    page_uuid = None if row.get("page_uuid") is None else str(row["page_uuid"])
    return (reason_code, reason_detail or "", page_uuid or "")


def _normalized_reason_row_keys(reason_rows: List[Dict[str, Any]]) -> Tuple[Tuple[str, str, str], ...]:
    deduped_rows = _dedupe_reason_rows(reason_rows)
    return tuple(sorted(_normalized_reason_row_key(row) for row in deduped_rows))


def _fetch_existing_reason_rows(
    conn: Connection,
    xml_status_reasons_table: str,
    *,
    agreement_uuid: str,
    version: int,
) -> List[Dict[str, Any]]:
    rows = conn.execute(
        text(
            f"""
            SELECT reason_code, reason_detail, page_uuid
            FROM {xml_status_reasons_table}
            WHERE agreement_uuid = :agreement_uuid
              AND xml_version = :version
            """
        ),
        {"agreement_uuid": agreement_uuid, "version": version},
    ).mappings().fetchall()
    return [
        {
            "reason_code": str(row["reason_code"]),
            "reason_detail": None if row.get("reason_detail") is None else str(row["reason_detail"]),
            "page_uuid": None if row.get("page_uuid") is None else str(row["page_uuid"]),
        }
        for row in rows
    ]


def _reason_rows_changed(
    existing_reason_rows: List[Dict[str, Any]],
    new_reason_rows: List[Dict[str, Any]],
) -> bool:
    return _normalized_reason_row_keys(existing_reason_rows) != _normalized_reason_row_keys(new_reason_rows)


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
                "reason_detail": row["reason_detail"][:16000] if row["reason_detail"] is not None else None,
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
    existing_reason_rows = _fetch_existing_reason_rows(
        conn,
        xml_status_reasons_table,
        agreement_uuid=agreement_uuid,
        version=version,
    )
    primary_reason_code = deduped_rows[0]["reason_code"] if status == "invalid" and deduped_rows else None
    _raw_detail = deduped_rows[0]["reason_detail"] if status == "invalid" and deduped_rows else None
    primary_reason_detail = _raw_detail[:16000] if _raw_detail is not None else None
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
    if status == "invalid" and _reason_rows_changed(existing_reason_rows, deduped_rows):
        _ = conn.execute(
            text(
                f"""
                UPDATE {xml_table}
                SET ai_repair_attempted = 0
                WHERE agreement_uuid = :agreement_uuid
                  AND version = :version
                  AND ai_repair_attempted = 1
                """
            ),
            {"agreement_uuid": agreement_uuid, "version": version},
        )
    return int(result.rowcount or 0)


def _xml_llm_verification_enabled(pipeline_config: PipelineConfig) -> bool:
    raw_value = getattr(pipeline_config, "xml_enable_llm_verification", None)
    if raw_value is None:
        return True
    return bool(raw_value)


def _mark_xml_rows_verified(
    conn: Connection,
    xml_table: str,
    xml_status_reasons_table: str,
    *,
    rows: List[Tuple[str, int]],
) -> int:
    updated = 0
    for agreement_uuid, version in rows:
        updated += _set_xml_status_with_reasons(
            conn,
            xml_table,
            xml_status_reasons_table,
            agreement_uuid=agreement_uuid,
            version=version,
            status="verified",
            reason_rows=[],
        )
    return updated


def _apply_xml_verify_batch_output(
    context: AssetExecutionContext,
    engine: Any,
    client: "OpenAI",
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


def _run_xml_build_for_agreements(
    context: AssetExecutionContext,
    *,
    db: DBResource,
    pipeline_config: PipelineConfig,
    target_agreement_uuids: list[str] | None,
    log_prefix: str,
) -> List[str]:
    agreement_batch_size = pipeline_config.xml_agreement_batch_size
    single_batch_run = runs_single_batch(context, pipeline_config)

    engine = db.get_engine()
    schema = db.database
    agreements_table = f"{schema}.agreements"
    pages_table = f"{schema}.pages"
    tagged_outputs_table = f"{schema}.tagged_outputs"
    xml_table = f"{schema}.xml"
    explicit_scope = target_agreement_uuids is not None
    scoped_uuids = sorted(set(target_agreement_uuids or []))
    if explicit_scope and not scoped_uuids:
        context.log.info("%s: explicit empty scope; no XML build work to run.", log_prefix)
        return []

    if pipeline_config.resume_openai_batches:
        with engine.begin() as conn:
            stranded_verify_batch = _fetch_unpulled_xml_verify_batch(
                conn,
                schema,
                batch_scope=XML_VERIFY_BATCH_SCOPE_DEFAULT,
            )
        if stranded_verify_batch is not None:
            context.log.info(
                "%s: deferring new XML generation because unpulled verify batch %s is waiting to resume.",
                log_prefix,
                stranded_verify_batch["batch_id"],
            )
            return []

    last_uuid = ""
    built_agreement_uuids: List[str] = []
    while True:
        with engine.begin() as conn:
            if scoped_uuids:
                agreement_uuids = (
                    conn.execute(
                        text(canonical_fresh_xml_build_queue_sql(schema, scoped=True)).bindparams(
                            bindparam("auuids", expanding=True)
                        ),
                        {"limit": max(agreement_batch_size, len(scoped_uuids)), "auuids": tuple(scoped_uuids)},
                    )
                    .scalars()
                    .all()
                )
            else:
                agreement_uuids = (
                    conn.execute(
                        text(canonical_fresh_xml_build_queue_sql(schema)),
                        {"limit": agreement_batch_size, "last_uuid": last_uuid},
                    )
                    .scalars()
                    .all()
                )

            if not agreement_uuids:
                break

            rows = (
                conn.execute(
                    text(
                        f"""
                    SELECT
                    p.agreement_uuid,
                    p.page_uuid,
                    p.page_order,
                    p.raw_page_content,
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
            df = _apply_safe_xml_tag_repairs_to_df(
                df,
                context=context,
                log_prefix=log_prefix,
            )
            xml, xml_generation_failures = generate_xml(df, version_map)
            for failure in xml_generation_failures:
                context.log.warning(
                    "%s: skipping XML generation due to parse error for agreement_uuid=%s: %s",
                    log_prefix,
                    failure.agreement_uuid,
                    failure.error,
                )

            if not xml:
                if scoped_uuids:
                    break
                last_uuid = agreement_uuids[-1]
                if single_batch_run:
                    break
                continue

            generated_agreement_uuids = [str(item.agreement_uuid) for item in xml]
            built_agreement_uuids.extend(generated_agreement_uuids)

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
                "%s: generated XML for %s agreements; refreshed latest_sections_search rows=%s",
                log_prefix,
                len(generated_agreement_uuids),
                refreshed,
            )

            if scoped_uuids:
                break

            last_uuid = agreement_uuids[-1]

        if single_batch_run:
            break

    return sorted(set(built_agreement_uuids))


@dg.asset(deps=[tagging_asset], name="04-01_build_xml")
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
    context.log.info("Running XML generation")
    built_agreement_uuids = _run_xml_build_for_agreements(
        context,
        db=db,
        pipeline_config=pipeline_config,
        target_agreement_uuids=None,
        log_prefix="xml_asset",
    )
    run_post_asset_refresh(context, db, pipeline_config)
    return built_agreement_uuids


def _select_cleanup_c_scope(
    context: AssetExecutionContext,
    *,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> List[str]:
    agreement_batch_size = pipeline_config.xml_agreement_batch_size
    single_batch_run = runs_single_batch(context, pipeline_config)
    engine = db.get_engine()
    schema = db.database
    last_uuid = ""
    selected_agreement_uuids: List[str] = []

    while True:
        with engine.begin() as conn:
            agreement_uuids = [
                str(row)
                for row in conn.execute(
                    text(canonical_fresh_xml_build_queue_sql(schema)),
                    {"limit": agreement_batch_size, "last_uuid": last_uuid},
                ).scalars().all()
            ]

        if not agreement_uuids:
            break

        selected_agreement_uuids.extend(agreement_uuids)
        last_uuid = agreement_uuids[-1]

        if single_batch_run:
            break

    return sorted(set(selected_agreement_uuids))


@dg.asset(
    name="04-01_regular_ingest_build_xml",
    ins={"tagged_agreement_uuids": dg.AssetIn(key=regular_ingest_tagging_asset.key)},
)
def regular_ingest_xml_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    tagged_agreement_uuids: List[str],
) -> List[str]:
    scope_uuids = load_active_scope_for_job(
        context,
        db=db,
        job_name="regular_ingest",
        fallback_agreement_uuids=tagged_agreement_uuids,
    )
    should_skip, current_stage = should_skip_managed_stage(
        db=db,
        job_name="regular_ingest",
        stage_name="regular_ingest_build_xml",
    )
    if should_skip:
        context.log.info(
            "regular_ingest_xml_asset: skipping because logical run already reached %s.",
            current_stage,
        )
        return scope_uuids
    built_agreement_uuids = _run_xml_build_for_agreements(
        context,
        db=db,
        pipeline_config=pipeline_config,
        target_agreement_uuids=scope_uuids,
        log_prefix="regular_ingest_xml_asset",
    )
    run_post_asset_refresh(context, db, pipeline_config)
    mark_logical_run_stage_completed(
        db=db,
        job_name="regular_ingest",
        stage_name="regular_ingest_build_xml",
    )
    return built_agreement_uuids


@dg.asset(
    name="04-03_ingestion_cleanup_a_build_xml",
    ins={"tagged_agreement_uuids": dg.AssetIn(key=ingestion_cleanup_a_tagging_asset.key)},
)
def ingestion_cleanup_a_xml_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    tagged_agreement_uuids: List[str],
) -> List[str]:
    scope_uuids = load_active_scope_for_job(
        context,
        db=db,
        job_name="ingestion_cleanup_a",
        fallback_agreement_uuids=tagged_agreement_uuids,
    )
    should_skip, current_stage = should_skip_managed_stage(
        db=db,
        job_name="ingestion_cleanup_a",
        stage_name="ingestion_cleanup_a_build_xml",
    )
    if should_skip:
        context.log.info(
            "ingestion_cleanup_a_xml_asset: skipping because logical run already reached %s.",
            current_stage,
        )
        return scope_uuids
    built_agreement_uuids = _run_xml_build_for_agreements(
        context,
        db=db,
        pipeline_config=pipeline_config,
        target_agreement_uuids=scope_uuids,
        log_prefix="ingestion_cleanup_a_xml_asset",
    )
    run_post_asset_refresh(context, db, pipeline_config)
    mark_logical_run_stage_completed(
        db=db,
        job_name="ingestion_cleanup_a",
        stage_name="ingestion_cleanup_a_build_xml",
    )
    return built_agreement_uuids


@dg.asset(name="04-05_ingestion_cleanup_c_build_xml")
def ingestion_cleanup_c_xml_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> List[str]:
    selected_scope = _select_cleanup_c_scope(
        context,
        db=db,
        pipeline_config=pipeline_config,
    )
    logical_run = start_or_resume_logical_run(
        context,
        db=db,
        pipeline_config=pipeline_config,
        job_name="ingestion_cleanup_c",
        initial_stage="ingestion_cleanup_c_build_xml",
        selected_agreement_uuids=selected_scope,
    )
    scope_uuids = logical_run.agreement_uuids if logical_run is not None else []
    should_skip, current_stage = should_skip_managed_stage(
        db=db,
        job_name="ingestion_cleanup_c",
        stage_name="ingestion_cleanup_c_build_xml",
    )
    if should_skip:
        context.log.info(
            "ingestion_cleanup_c_xml_asset: skipping because logical run already reached %s.",
            current_stage,
        )
        return []
    built_agreement_uuids = _run_xml_build_for_agreements(
        context,
        db=db,
        pipeline_config=pipeline_config,
        target_agreement_uuids=scope_uuids,
        log_prefix="ingestion_cleanup_c_xml_asset",
    )
    run_post_asset_refresh(context, db, pipeline_config)
    mark_logical_run_stage_completed(
        db=db,
        job_name="ingestion_cleanup_c",
        stage_name="ingestion_cleanup_c_build_xml",
    )
    return built_agreement_uuids


@dg.asset(
    name="04-02_verify_xml",
    ins={"built_xml_agreement_uuids": dg.AssetIn(key=xml_asset.key)},
)
def xml_verify_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    built_xml_agreement_uuids: List[str],
) -> List[str]:
    agreement_batch_size = pipeline_config.xml_agreement_batch_size
    llm_verification_enabled = _xml_llm_verification_enabled(pipeline_config)
    resume_openai_batches = (
        pipeline_config.resume_openai_batches and llm_verification_enabled
    )
    target_agreement_uuids = sorted(set(built_xml_agreement_uuids))

    engine = db.get_engine()
    schema = db.database
    xml_table = f"{schema}.xml"
    client = _oai_client() if llm_verification_enabled else None

    with engine.begin() as conn:
        assert_tables_exist(conn, schema=schema, table_names=("xml_verify_batches", "xml_status_reasons"))

    if resume_openai_batches:
        assert client is not None
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
    direct_verify_rows: List[Tuple[str, int]] = []
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

        if not llm_verification_enabled:
            direct_verify_rows.append((agreement_uuid, version))
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
                model="gpt-5.4-mini",
                toc_context=_build_xml_verify_toc_context(root),
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

    if not llm_verification_enabled:
        direct_verified_updated = 0
        if direct_verify_rows:
            with engine.begin() as conn:
                direct_verified_updated = _mark_xml_rows_verified(
                    conn,
                    xml_table,
                    f"{schema}.xml_status_reasons",
                    rows=direct_verify_rows,
                )
        context.log.info(
            "xml_verify_asset: bypassed LLM verification by config; directly verified=%s, hard_invalid_updated=%s",
            direct_verified_updated,
            hard_invalid_updated,
        )
        run_post_asset_refresh(context, db, pipeline_config)
        return selected_for_verify

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
    assert client is not None

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


def _run_scoped_fresh_xml_verify(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    *,
    target_scope: List[str],
    job_name: str,
    stage_name: str,
    log_prefix: str,
    empty_scope_message: str,
    no_eligible_message: str,
    request_filename_prefix: str,
    mark_stage_on_empty_scope: bool = False,
    mark_stage_on_no_eligible: bool = False,
) -> List[str]:
    agreement_batch_size = pipeline_config.xml_agreement_batch_size
    llm_verification_enabled = _xml_llm_verification_enabled(pipeline_config)
    resume_openai_batches = (
        pipeline_config.resume_openai_batches and llm_verification_enabled
    )
    target_agreement_uuids = sorted(set(target_scope))

    engine = db.get_engine()
    schema = db.database
    xml_table = f"{schema}.xml"
    client = _oai_client() if llm_verification_enabled else None

    with engine.begin() as conn:
        assert_tables_exist(conn, schema=schema, table_names=("xml_verify_batches", "xml_status_reasons"))

    should_skip, current_stage = should_skip_managed_stage(
        db=db,
        job_name=job_name,
        stage_name=stage_name,
    )
    if should_skip:
        context.log.info("%s: skipping because logical run already reached %s.", log_prefix, current_stage)
        return []

    if not target_agreement_uuids:
        context.log.info(empty_scope_message)
        run_post_asset_refresh(context, db, pipeline_config)
        if mark_stage_on_empty_scope:
            mark_logical_run_stage_completed(db=db, job_name=job_name, stage_name=stage_name)
        return []

    queue_q = text(canonical_fresh_xml_verify_queue_sql(schema, scoped=True)).bindparams(
        bindparam("auuids", expanding=True)
    )
    with engine.begin() as conn:
        eligible_uuids = [
            str(row)
            for row in conn.execute(
                queue_q,
                {"lim": max(agreement_batch_size, len(target_agreement_uuids)), "auuids": target_agreement_uuids},
            ).scalars().all()
        ]
    if not eligible_uuids:
        context.log.info(no_eligible_message)
        run_post_asset_refresh(context, db, pipeline_config)
        if mark_stage_on_no_eligible:
            mark_logical_run_stage_completed(db=db, job_name=job_name, stage_name=stage_name)
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

    verified_agreement_uuids: set[str] = set()
    for start in range(0, len(eligible_uuids), agreement_batch_size):
        chunk_uuids = eligible_uuids[start : start + agreement_batch_size]
        with engine.begin() as conn:
            rows = conn.execute(select_q, {"auuids": tuple(chunk_uuids)}).mappings().fetchall()

        selected_for_verify = [str(row["agreement_uuid"]) for row in rows]
        lines: List[Dict[str, Any]] = []
        direct_verify_rows: List[Tuple[str, int]] = []
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

            if not llm_verification_enabled:
                direct_verify_rows.append((agreement_uuid, version))
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
                    model="gpt-5.4-mini",
                    toc_context=_build_xml_verify_toc_context(root),
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

        if not llm_verification_enabled:
            if direct_verify_rows:
                with engine.begin() as conn:
                    _ = _mark_xml_rows_verified(
                        conn,
                        xml_table,
                        f"{schema}.xml_status_reasons",
                        rows=direct_verify_rows,
                    )
            with engine.begin() as conn:
                verified_agreement_uuids.update(
                    _fetch_latest_verified_agreement_uuids(
                        conn,
                        xml_table=xml_table,
                        agreement_uuids=selected_for_verify,
                    )
                )
            continue

        if not lines:
            with engine.begin() as conn:
                verified_agreement_uuids.update(
                    _fetch_latest_verified_agreement_uuids(
                        conn,
                        xml_table=xml_table,
                        agreement_uuids=selected_for_verify,
                    )
                )
            continue

        llm_targets = sorted({_parse_custom_id(str(line["custom_id"])) for line in lines})
        if not llm_targets:
            raise ValueError(f"{log_prefix}: no (agreement_uuid, version) targets derived from LLM lines.")
        active_run = load_active_logical_run(db=db, job_name=job_name)
        verify_batch_key = build_logical_batch_key(
            logical_run_id=None if active_run is None else str(active_run["logical_run_id"]),
            stage_name=stage_name,
            default_key=agreement_version_batch_key(llm_targets),
        )
        assert client is not None

        if resume_openai_batches:
            with engine.begin() as conn:
                existing_batch = _fetch_unpulled_xml_verify_batch(
                    conn,
                    schema,
                    batch_scope=XML_VERIFY_BATCH_SCOPE_DEFAULT,
                    batch_key=verify_batch_key,
                )
            if existing_batch is not None:
                _ = _resume_xml_verify_batch(
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
                    log_prefix=log_prefix,
                    hard_invalid_updated=hard_invalid_updated,
                )
                with engine.begin() as conn:
                    verified_agreement_uuids.update(
                        _fetch_latest_verified_agreement_uuids(
                            conn,
                            xml_table=xml_table,
                            agreement_uuids=selected_for_verify,
                        )
                    )
                continue

        jsonl_buf = io.StringIO()
        for line in lines:
            _ = jsonl_buf.write(json.dumps(line, ensure_ascii=False) + "\n")
        jsonl_bytes = io.BytesIO(jsonl_buf.getvalue().encode("utf-8"))
        jsonl_bytes.name = f"{request_filename_prefix}_{start}.jsonl"

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

        final_batch = poll_batch_until_terminal(context, client, batch.id, log_prefix=log_prefix)
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

        if final_batch.status == "completed":
            _ = _apply_xml_verify_batch_output(
                context=context,
                engine=engine,
                client=client,
                xml_table=xml_table,
                xml_status_reasons_table=f"{schema}.xml_status_reasons",
                batch=final_batch,
                log_prefix=log_prefix,
            )
        else:
            context.log.warning(
                "%s: batch %s ended with status=%s; no status updates applied.",
                log_prefix,
                final_batch.id,
                final_batch.status,
            )
        with engine.begin() as conn:
            _mark_xml_verify_batch_pulled(conn, schema, final_batch.id)
        with engine.begin() as conn:
            verified_agreement_uuids.update(
                _fetch_latest_verified_agreement_uuids(
                    conn,
                    xml_table=xml_table,
                    agreement_uuids=selected_for_verify,
                )
            )

    run_post_asset_refresh(context, db, pipeline_config)
    mark_logical_run_stage_completed(db=db, job_name=job_name, stage_name=stage_name)
    return sorted(verified_agreement_uuids)


@dg.asset(
    name="04-02_regular_ingest_verify_xml",
    ins={"built_xml_agreement_uuids": dg.AssetIn(key=regular_ingest_xml_asset.key)},
)
def regular_ingest_xml_verify_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    built_xml_agreement_uuids: List[str],
) -> List[str]:
    return _run_scoped_fresh_xml_verify(
        context,
        db,
        pipeline_config,
        target_scope=load_active_scope_for_job(
            context,
            db=db,
            job_name="regular_ingest",
            fallback_agreement_uuids=built_xml_agreement_uuids,
        ),
        job_name="regular_ingest",
        stage_name="regular_ingest_verify_xml",
        log_prefix="regular_ingest_xml_verify_asset",
        empty_scope_message="regular_ingest_xml_verify_asset: no upstream agreements from regular_ingest_xml_asset.",
        no_eligible_message=(
            "regular_ingest_xml_verify_asset: no upstream-selected XML rows with status IS NULL, latest=1, and ai_repair_attempted=0."
        ),
        request_filename_prefix="regular_ingest_xml_verify_requests",
        mark_stage_on_no_eligible=True,
    )


@dg.asset(
    name="04-04_ingestion_cleanup_a_verify_xml",
    ins={"built_xml_agreement_uuids": dg.AssetIn(key=ingestion_cleanup_a_xml_asset.key)},
)
def ingestion_cleanup_a_xml_verify_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    built_xml_agreement_uuids: List[str],
) -> List[str]:
    return _run_scoped_fresh_xml_verify(
        context,
        db,
        pipeline_config,
        target_scope=load_active_scope_for_job(
            context,
            db=db,
            job_name="ingestion_cleanup_a",
            fallback_agreement_uuids=built_xml_agreement_uuids,
        ),
        job_name="ingestion_cleanup_a",
        stage_name="ingestion_cleanup_a_verify_xml",
        log_prefix="ingestion_cleanup_a_xml_verify_asset",
        empty_scope_message="ingestion_cleanup_a_xml_verify_asset: no upstream agreements from ingestion_cleanup_a_xml_asset.",
        no_eligible_message=(
            "ingestion_cleanup_a_xml_verify_asset: no upstream-selected XML rows with status IS NULL, latest=1, and ai_repair_attempted=0."
        ),
        request_filename_prefix="ingestion_cleanup_a_xml_verify_requests",
        mark_stage_on_no_eligible=True,
    )


@dg.asset(
    name="04-06_ingestion_cleanup_c_verify_xml",
    ins={"built_xml_agreement_uuids": dg.AssetIn(key=ingestion_cleanup_c_xml_asset.key)},
)
def ingestion_cleanup_c_xml_verify_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    built_xml_agreement_uuids: List[str],
) -> List[str]:
    return _run_scoped_fresh_xml_verify(
        context,
        db,
        pipeline_config,
        target_scope=load_active_scope_for_job(
            context,
            db=db,
            job_name="ingestion_cleanup_c",
            fallback_agreement_uuids=built_xml_agreement_uuids,
        ),
        job_name="ingestion_cleanup_c",
        stage_name="ingestion_cleanup_c_verify_xml",
        log_prefix="ingestion_cleanup_c_xml_verify_asset",
        empty_scope_message="ingestion_cleanup_c_xml_verify_asset: no agreements in the active cleanup scope.",
        no_eligible_message=(
            "ingestion_cleanup_c_xml_verify_asset: no active-scope XML rows with status IS NULL, latest=1, and ai_repair_attempted=0."
        ),
        request_filename_prefix="ingestion_cleanup_c_xml_verify_requests",
        mark_stage_on_empty_scope=True,
        mark_stage_on_no_eligible=True,
    )
