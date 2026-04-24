"""
AI Repair assets:
- ai_repair_enqueue_asset: enqueues full-page AI retagging for XML-targeted pages
- ai_repair_poll_asset: polls batches, downloads outputs, and persists results

Required tables (managed via migrations):
- pdx.ai_repair_batches     (batch-level tracking)
- pdx.ai_repair_requests    (one row per JSONL line / custom_id)
- pdx.ai_repair_rulings     (excerpt-mode rulings at page-level coords)
- pdx.ai_repair_full_pages  (full-page tagged_text outputs)
- pdx.xml_status_reasons    (invalid XML reason rows with page_uuid targets)
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false

import io
import json
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Set, Optional

import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import text, bindparam
from sqlalchemy.engine import Connection
import os
import time

from etl.defs.resources import DBResource, PipelineConfig
from etl.defs.resources import AIRepairAttemptPriority
from etl.defs.f_xml_asset import (
    XML_REASON_BODY_STARTS_NON_ARTICLE,
    XML_REASON_FIRST_ARTICLE_NOT_ONE,
    XML_REASON_SECTION_ARTICLE_MISMATCH,
    XML_REASON_SECTION_NON_SEQUENTIAL,
    XML_REASON_SECTION_TITLE_INVALID_NUMBERING,
    XML_REASON_TOO_MANY_EMPTY_ARTICLES,
    extract_body_page_uuid_article_map,
    ingestion_cleanup_a_xml_verify_asset,
    extract_toc_section_sequences,
    regular_ingest_xml_verify_asset,
)
from etl.utils.post_asset_refresh import run_post_asset_refresh
from etl.utils.pipeline_state_sql import canonical_ai_repair_enqueue_queue_sql
from etl.utils.batch_keys import agreement_batch_key
from etl.utils.openai_batch import (
    extract_output_text_from_batch_body,
    read_openai_file_text,
)
from etl.utils.logical_job_runs import (
    build_logical_batch_scope_tag,
    load_active_logical_run,
    load_active_scope_for_job,
    mark_logical_run_stage_completed,
    should_skip_managed_stage,
    start_or_resume_logical_run,
)
from etl.utils.schema_guards import assert_tables_exist
from etl.domain.d_ai_repair import (
    RepairDecision,
    build_jsonl_lines_for_page,
    build_jsonl_line_for_source_text_verdict,
)

if TYPE_CHECKING:
    from openai import OpenAI


def _oai_client() -> "OpenAI":
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for ai_repair assets.")
    return OpenAI(api_key=api_key)


AI_REPAIR_ELIGIBLE_XML_REASON_CODES: Tuple[str, ...] = (
    XML_REASON_BODY_STARTS_NON_ARTICLE,
    XML_REASON_FIRST_ARTICLE_NOT_ONE,
    XML_REASON_SECTION_TITLE_INVALID_NUMBERING,
    XML_REASON_SECTION_ARTICLE_MISMATCH,
    XML_REASON_SECTION_NON_SEQUENTIAL,
    XML_REASON_TOO_MANY_EMPTY_ARTICLES,
)
AI_REPAIR_SOURCE_VERDICT_XML_REASON_CODES: Tuple[str, ...] = (
    XML_REASON_SECTION_NON_SEQUENTIAL,
)

AI_REPAIR_FIRST_PASS_MODEL = "gpt-5-mini"
AI_REPAIR_RETRY_MODEL = "gpt-5.1"
AI_REPAIR_SOURCE_VERDICT_MODEL = "gpt-5.1"
XML_REASON_LLM_SOURCE_TEXT_BYPASS = "llm_source_text_hard_rule_bypass"
SOURCE_TEXT_BYPASS_CONFIDENCE_THRESHOLD = 0.90
SOURCE_VERDICT_MAX_FLAGGED_PAGES = 4
SOURCE_VERDICT_MAX_CONTEXT_PAGES = 9
_ALLOWED_FULL_MODE_TAGS: Tuple[str, ...] = (
    "<article>",
    "</article>",
    "<section>",
    "</section>",
    "<page>",
    "</page>",
)

_AI_REPAIR_TERMINAL_BATCH_STATUSES = ("completed", "failed", "cancelled", "expired")
_AI_REPAIR_FAILED_BATCH_STATUSES = ("failed", "cancelled", "expired")


def _full_request_id(page_uuid: str, xml_version: int) -> str:
    return f"{page_uuid}::full::{int(xml_version)}"


def _source_request_id(agreement_uuid: str, xml_version: int) -> str:
    return f"{agreement_uuid}::source::{int(xml_version)}"


def _parse_source_request_id(request_id: str) -> Tuple[str, int]:
    parts = request_id.split("::")
    if len(parts) != 3 or parts[1] != "source":
        raise ValueError(f"Invalid source verdict request_id: {request_id!r}")
    return parts[0], int(parts[2])


def _repair_model_for_attempted(ai_repair_attempted: int) -> str:
    if ai_repair_attempted in (0, 1):
        return AI_REPAIR_FIRST_PASS_MODEL
    raise ValueError(f"Invalid ai_repair_attempted value: {ai_repair_attempted!r}")


def _repair_model_for_candidate(
    ai_repair_attempted: int,
    *,
    has_completed_requests: bool,
) -> str:
    if has_completed_requests:
        return AI_REPAIR_RETRY_MODEL
    return _repair_model_for_attempted(ai_repair_attempted)


def _fetch_candidate_page_reason_codes(
    conn: Connection,
    schema: str,
    *,
    page_uuids: List[str],
) -> Dict[str, Set[str]]:
    if not page_uuids:
        return {}
    rows = conn.execute(
        text(
            f"""
            SELECT
                r.page_uuid,
                r.reason_code
            FROM {schema}.xml_status_reasons r
            JOIN {schema}.xml x
                ON x.agreement_uuid = r.agreement_uuid
               AND x.version = r.xml_version
            WHERE x.latest = 1
              AND x.status = 'invalid'
              AND r.page_uuid IN :page_uuids
            """
        ).bindparams(bindparam("page_uuids", expanding=True)),
        {"page_uuids": page_uuids},
    ).mappings().fetchall()
    reason_codes_by_page_uuid: Dict[str, Set[str]] = {}
    for row in rows:
        page_uuid = str(row["page_uuid"])
        reason_code = str(row["reason_code"])
        reason_codes_by_page_uuid.setdefault(page_uuid, set()).add(reason_code)
    return reason_codes_by_page_uuid


def _fetch_latest_xml_text_by_agreement(
    conn: Connection,
    schema: str,
    *,
    agreement_uuids: List[str],
) -> Dict[str, str]:
    if not agreement_uuids:
        return {}
    rows = conn.execute(
        text(
            f"""
            SELECT agreement_uuid, xml
            FROM {schema}.xml
            WHERE latest = 1
              AND agreement_uuid IN :agreement_uuids
            """
        ).bindparams(bindparam("agreement_uuids", expanding=True)),
        {"agreement_uuids": agreement_uuids},
    ).mappings().fetchall()
    return {
        str(row["agreement_uuid"]): str(row.get("xml") or "")
        for row in rows
    }


def _build_toc_context_for_page(page_uuid: str, xml_text: str) -> str | None:
    if not xml_text.strip():
        return None
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    article_nums = extract_body_page_uuid_article_map(root).get(page_uuid)
    if article_nums is None or len(article_nums) != 1:
        return None
    article_num = article_nums[0]
    toc_sections = extract_toc_section_sequences(root).get(article_num)
    if not toc_sections:
        return None
    toc_numbers = ", ".join(f"{article_num}.{section_num}" for section_num in toc_sections)
    return f"Article {article_num} TOC section numbering: {toc_numbers}"


def _fetch_candidates(
    conn: Connection,
    schema: str,
    agreement_limit: int | None,
    target_agreement_uuids: list[str] | None = None,
    page_budget: int | None = None,
    attempt_priority: AIRepairAttemptPriority = AIRepairAttemptPriority.NOT_ATTEMPTED_FIRST,
    exclude_in_flight: bool = True,
) -> List[Dict[str, Any]]:
    """
    Pull page-level AI-repair targets derived from invalid latest XML reasons.
    Targets come from xml_status_reasons.page_uuid rows and are ordered by
    configured attempted/not-attempted priority, then by whether the
    agreement has only section_non_sequential reasons, then by fewest
    unresolved target pages. When page_budget is set, selected agreements are
    capped by total unresolved target pages instead of agreement count.
    """
    pages_table = f"{schema}.pages"
    ai_repair_requests_table = f"{schema}.ai_repair_requests"
    if not AI_REPAIR_ELIGIBLE_XML_REASON_CODES:
        raise ValueError("AI repair eligible XML reason codes must not be empty.")

    blocked_statuses: Tuple[str, ...]
    if exclude_in_flight:
        blocked_statuses = ("completed", "queued", "running")
    else:
        blocked_statuses = ("completed",)

    scoped_uuids = sorted(set(target_agreement_uuids or []))
    queue_sql = text(
        canonical_ai_repair_enqueue_queue_sql(schema, scoped=bool(scoped_uuids))
    ).bindparams(bindparam("reason_codes", expanding=True))
    params: Dict[str, Any] = {
        "reason_codes": list(AI_REPAIR_ELIGIBLE_XML_REASON_CODES),
    }
    if scoped_uuids:
        queue_sql = queue_sql.bindparams(bindparam("agreement_uuids", expanding=True))
        params["agreement_uuids"] = scoped_uuids
    invalid_rows = conn.execute(queue_sql, params).mappings().fetchall()
    if not invalid_rows:
        return []

    xml_version_by_agreement: Dict[str, int] = {}
    ai_repair_attempted_by_agreement: Dict[str, int] = {}
    page_uuids_by_agreement: Dict[str, List[str]] = {}
    reason_codes_by_agreement: Dict[str, set[str]] = {}
    request_ids: List[str] = []

    for row in invalid_rows:
        agreement_uuid = str(row["agreement_uuid"])
        xml_version = int(row["xml_version"])
        reason_code = str(row["reason_code"])
        attempted_raw = row.get("ai_repair_attempted")
        attempted = int(attempted_raw or 0)
        if attempted not in (0, 1):
            raise ValueError(
                f"Invalid ai_repair_attempted value for agreement {agreement_uuid}: {attempted_raw!r}."
            )
        page_uuid_raw = row.get("page_uuid")
        if page_uuid_raw is None:
            continue
        page_uuid = str(page_uuid_raw).strip()
        if not page_uuid:
            continue

        payload_version = xml_version_by_agreement.get(agreement_uuid)
        if payload_version is None:
            xml_version_by_agreement[agreement_uuid] = xml_version
            payload_version = xml_version
        elif payload_version != xml_version:
            raise ValueError(
                f"Multiple latest xml versions for agreement {agreement_uuid}: {payload_version} and {xml_version}."
            )
        payload_attempted = ai_repair_attempted_by_agreement.get(agreement_uuid)
        if payload_attempted is None:
            ai_repair_attempted_by_agreement[agreement_uuid] = attempted
        elif payload_attempted != attempted:
            raise ValueError(
                f"Inconsistent ai_repair_attempted values for agreement {agreement_uuid}: {payload_attempted} and {attempted}."
            )
        reason_codes_by_agreement.setdefault(agreement_uuid, set()).add(reason_code)

        page_uuids = page_uuids_by_agreement.setdefault(agreement_uuid, [])
        if page_uuid in page_uuids:
            continue
        page_uuids.append(page_uuid)
        req_id = _full_request_id(page_uuid, xml_version)
        request_ids.append(req_id)

    if not request_ids:
        return []

    existing_request_ids = set(
        str(v)
        for v in conn.execute(
            text(
                f"""
                SELECT request_id
                FROM {ai_repair_requests_table}
                WHERE request_id IN :rids
                  AND status IN :statuses
                """
            ).bindparams(
                bindparam("rids", expanding=True),
                bindparam("statuses", expanding=True),
            ),
            {"rids": request_ids, "statuses": list(blocked_statuses)},
        ).scalars().all()
    )
    completed_request_ids = set(
        str(v)
        for v in conn.execute(
            text(
                f"""
                SELECT request_id
                FROM {ai_repair_requests_table}
                WHERE request_id IN :rids
                  AND status = 'completed'
                """
            ).bindparams(bindparam("rids", expanding=True)),
            {"rids": request_ids},
        ).scalars().all()
    )

    unresolved_by_agreement: Dict[str, Tuple[int, int, int, int, List[str]]] = {}
    for agreement_uuid, page_uuids in page_uuids_by_agreement.items():
        attempted = int(ai_repair_attempted_by_agreement[agreement_uuid])
        xml_version = int(xml_version_by_agreement[agreement_uuid])
        reason_codes = reason_codes_by_agreement.get(agreement_uuid, set())
        section_non_sequential_only = int(
            bool(reason_codes)
            and reason_codes == {XML_REASON_SECTION_NON_SEQUENTIAL}
        )
        has_completed_requests = any(
            _full_request_id(page_uuid, xml_version) in completed_request_ids
            for page_uuid in page_uuids
        )
        unresolved_page_uuids: List[str] = []
        for page_uuid in page_uuids:
            req_id = _full_request_id(page_uuid, xml_version)
            if req_id in existing_request_ids:
                continue
            unresolved_page_uuids.append(page_uuid)
        if not unresolved_page_uuids:
            continue
        unresolved_by_agreement[agreement_uuid] = (
            attempted,
            section_non_sequential_only,
            int(has_completed_requests),
            xml_version,
            unresolved_page_uuids,
        )

    if not unresolved_by_agreement:
        return []

    attempted_rank = {
        AIRepairAttemptPriority.NOT_ATTEMPTED_FIRST: {0: 0, 1: 1},
        AIRepairAttemptPriority.ATTEMPTED_FIRST: {1: 0, 0: 1},
    }[attempt_priority]

    ranked_agreements = sorted(
        unresolved_by_agreement.items(),
        key=lambda item: (
            attempted_rank[item[1][0]],
            0 if item[1][1] == 1 else 1,
            len(item[1][4]),
            item[0],
        ),
    )
    if page_budget is not None and page_budget > 0:
        selected_ranked_agreements: List[Tuple[str, Tuple[int, int, int, int, List[str]]]] = []
        selected_page_total = 0
        for agreement_uuid, payload in ranked_agreements:
            unresolved_page_count = len(payload[4])
            if selected_page_total + unresolved_page_count > page_budget:
                break
            selected_ranked_agreements.append((agreement_uuid, payload))
            selected_page_total += unresolved_page_count
        ranked_agreements = selected_ranked_agreements
    elif agreement_limit is not None:
        ranked_agreements = ranked_agreements[:agreement_limit]

    selected_page_rows: List[Dict[str, Any]] = []
    page_uuid_to_target: Dict[str, Tuple[str, int, int, int]] = {}
    for agreement_uuid, payload in ranked_agreements:
        attempted, _section_non_sequential_only, has_completed_requests, xml_version, unresolved_page_uuids = payload
        for page_uuid in unresolved_page_uuids:
            page_uuid_to_target[page_uuid] = (
                agreement_uuid,
                attempted,
                has_completed_requests,
                xml_version,
            )

    if not page_uuid_to_target:
        return []

    page_rows = conn.execute(
        text(
            f"""
            SELECT
                p.page_uuid,
                p.agreement_uuid,
                p.page_order,
                p.processed_page_content AS text
            FROM {pages_table} p
            WHERE p.page_uuid IN :pids
            ORDER BY p.agreement_uuid, p.page_order, p.page_uuid
            """
        ).bindparams(bindparam("pids", expanding=True)),
        {"pids": list(page_uuid_to_target.keys())},
    ).mappings().fetchall()

    for row in page_rows:
        page_uuid = str(row["page_uuid"])
        target = page_uuid_to_target.get(page_uuid)
        if target is None:
            continue
        agreement_uuid, attempted, has_completed_requests, xml_version = target
        selected_page_rows.append(
            {
                "page_uuid": page_uuid,
                "agreement_uuid": agreement_uuid,
                "text": str(row.get("text") or ""),
                "ai_repair_attempted": attempted,
                "has_completed_requests": has_completed_requests,
                "xml_version": xml_version,
                "page_order": int(row.get("page_order") or 0),
            }
        )

    selected_page_rows.sort(
        key=lambda row: (
            row["agreement_uuid"],
            int(row["page_order"]),
            row["page_uuid"],
        )
    )
    return selected_page_rows


def _fetch_source_verdict_candidates(
    conn: Connection,
    schema: str,
    agreement_limit: int | None,
    target_agreement_uuids: list[str] | None = None,
    page_budget: int | None = None,
    attempt_priority: AIRepairAttemptPriority = AIRepairAttemptPriority.NOT_ATTEMPTED_FIRST,
    exclude_in_flight: bool = True,
) -> List[Dict[str, Any]]:
    """
    Pull agreement-level source-text verdict targets.

    These are deliberately narrower than tag repair targets: currently only
    pure section_non_sequential failures with concrete page UUIDs are eligible.
    The verdict request gets the flagged page(s) plus neighboring pages so the
    model can decide whether the hard failure is a real source-text defect.
    """
    if not AI_REPAIR_SOURCE_VERDICT_XML_REASON_CODES:
        raise ValueError("AI repair source verdict reason codes must not be empty.")

    ai_repair_requests_table = f"{schema}.ai_repair_requests"
    ai_repair_source_verdicts_table = f"{schema}.ai_repair_source_verdicts"
    pages_table = f"{schema}.pages"
    tagged_outputs_table = f"{schema}.tagged_outputs"
    xml_status_reasons_table = f"{schema}.xml_status_reasons"

    scoped_uuids = sorted(set(target_agreement_uuids or []))
    queue_sql = text(
        canonical_ai_repair_enqueue_queue_sql(schema, scoped=bool(scoped_uuids))
    ).bindparams(bindparam("reason_codes", expanding=True))
    params: Dict[str, Any] = {
        "reason_codes": list(AI_REPAIR_SOURCE_VERDICT_XML_REASON_CODES),
    }
    if scoped_uuids:
        queue_sql = queue_sql.bindparams(bindparam("agreement_uuids", expanding=True))
        params["agreement_uuids"] = scoped_uuids
    invalid_rows = conn.execute(queue_sql, params).mappings().fetchall()
    if not invalid_rows:
        return []

    xml_version_by_agreement: Dict[str, int] = {}
    attempted_by_agreement: Dict[str, int] = {}
    target_page_uuids_by_agreement: Dict[str, List[str]] = {}
    request_ids: List[str] = []
    for row in invalid_rows:
        agreement_uuid = str(row["agreement_uuid"])
        xml_version = int(row["xml_version"])
        attempted = int(row.get("ai_repair_attempted") or 0)
        page_uuid = str(row.get("page_uuid") or "").strip()
        if not page_uuid:
            continue
        existing_version = xml_version_by_agreement.get(agreement_uuid)
        if existing_version is None:
            xml_version_by_agreement[agreement_uuid] = xml_version
            attempted_by_agreement[agreement_uuid] = attempted
            request_ids.append(_source_request_id(agreement_uuid, xml_version))
        elif existing_version != xml_version:
            raise ValueError(
                f"Multiple latest xml versions for agreement {agreement_uuid}: {existing_version} and {xml_version}."
            )
        target_pages = target_page_uuids_by_agreement.setdefault(agreement_uuid, [])
        if page_uuid not in target_pages:
            target_pages.append(page_uuid)

    if not request_ids:
        return []

    blocked_statuses = ("completed", "queued", "running") if exclude_in_flight else ("completed",)
    existing_request_ids = set(
        str(v)
        for v in conn.execute(
            text(
                f"""
                SELECT request_id
                FROM {ai_repair_requests_table}
                WHERE request_id IN :rids
                  AND status IN :statuses
                """
            ).bindparams(
                bindparam("rids", expanding=True),
                bindparam("statuses", expanding=True),
            ),
            {"rids": request_ids, "statuses": list(blocked_statuses)},
        ).scalars().all()
    )
    existing_verdict_request_ids = set(
        str(v)
        for v in conn.execute(
            text(
                f"""
                SELECT request_id
                FROM {ai_repair_source_verdicts_table}
                WHERE request_id IN :rids
                """
            ).bindparams(bindparam("rids", expanding=True)),
            {"rids": request_ids},
        ).scalars().all()
    )

    candidate_agreement_uuids = [
        agreement_uuid
        for agreement_uuid in sorted(target_page_uuids_by_agreement)
        if _source_request_id(agreement_uuid, xml_version_by_agreement[agreement_uuid])
        not in existing_request_ids
        and _source_request_id(agreement_uuid, xml_version_by_agreement[agreement_uuid])
        not in existing_verdict_request_ids
    ]
    if not candidate_agreement_uuids:
        return []

    reason_rows = (
        conn.execute(
            text(
                f"""
                SELECT
                    agreement_uuid,
                    xml_version,
                    reason_code,
                    reason_detail,
                    page_uuid
                FROM {xml_status_reasons_table}
                WHERE agreement_uuid IN :auuids
                ORDER BY agreement_uuid, id
                """
            ).bindparams(bindparam("auuids", expanding=True)),
            {"auuids": candidate_agreement_uuids},
        )
        .mappings()
        .fetchall()
    )
    reasons_by_agreement: Dict[str, List[Dict[str, Any]]] = {}
    reason_codes_by_agreement: Dict[str, Set[str]] = {}
    for row in reason_rows:
        agreement_uuid = str(row["agreement_uuid"])
        xml_version = int(row["xml_version"])
        if xml_version != xml_version_by_agreement.get(agreement_uuid):
            continue
        reason_code = str(row["reason_code"])
        page_uuid = None if row.get("page_uuid") is None else str(row["page_uuid"])
        reasons_by_agreement.setdefault(agreement_uuid, []).append(
            {
                "reason_code": reason_code,
                "reason_detail": None if row.get("reason_detail") is None else str(row["reason_detail"]),
                "page_uuid": page_uuid,
            }
        )
        reason_codes_by_agreement.setdefault(agreement_uuid, set()).add(reason_code)

    eligible_agreement_uuids: List[str] = []
    eligible_page_uuids: Set[str] = set()
    for agreement_uuid in candidate_agreement_uuids:
        reason_codes = reason_codes_by_agreement.get(agreement_uuid, set())
        if reason_codes != set(AI_REPAIR_SOURCE_VERDICT_XML_REASON_CODES):
            continue
        target_page_uuids = target_page_uuids_by_agreement.get(agreement_uuid, [])
        if not target_page_uuids or len(target_page_uuids) > SOURCE_VERDICT_MAX_FLAGGED_PAGES:
            continue
        rows_for_agreement = reasons_by_agreement.get(agreement_uuid, [])
        if any(row.get("page_uuid") is None for row in rows_for_agreement):
            continue
        eligible_agreement_uuids.append(agreement_uuid)
        eligible_page_uuids.update(target_page_uuids)

    if not eligible_agreement_uuids:
        return []

    target_page_rows = (
        conn.execute(
            text(
                f"""
                SELECT agreement_uuid, page_uuid, page_order
                FROM {pages_table}
                WHERE page_uuid IN :pids
                """
            ).bindparams(bindparam("pids", expanding=True)),
            {"pids": sorted(eligible_page_uuids)},
        )
        .mappings()
        .fetchall()
    )
    target_orders_by_agreement: Dict[str, Set[int]] = {}
    for row in target_page_rows:
        agreement_uuid = str(row["agreement_uuid"])
        if agreement_uuid not in eligible_agreement_uuids:
            continue
        target_orders_by_agreement.setdefault(agreement_uuid, set()).add(int(row["page_order"]))

    eligible_agreement_uuids = [
        agreement_uuid
        for agreement_uuid in eligible_agreement_uuids
        if target_orders_by_agreement.get(agreement_uuid)
    ]
    if not eligible_agreement_uuids:
        return []

    ranked_agreements = sorted(
        eligible_agreement_uuids,
        key=lambda agreement_uuid: (
            {
                AIRepairAttemptPriority.NOT_ATTEMPTED_FIRST: {0: 0, 1: 1},
                AIRepairAttemptPriority.ATTEMPTED_FIRST: {1: 0, 0: 1},
            }[attempt_priority][int(attempted_by_agreement[agreement_uuid])],
            len(target_page_uuids_by_agreement[agreement_uuid]),
            agreement_uuid,
        ),
    )
    if page_budget is not None and page_budget > 0:
        selected: List[str] = []
        selected_page_total = 0
        for agreement_uuid in ranked_agreements:
            page_count = len(target_page_uuids_by_agreement[agreement_uuid])
            if selected_page_total + page_count > page_budget:
                break
            selected.append(agreement_uuid)
            selected_page_total += page_count
        ranked_agreements = selected
    elif agreement_limit is not None:
        ranked_agreements = ranked_agreements[:agreement_limit]

    if not ranked_agreements:
        return []

    context_orders_by_agreement: Dict[str, Set[int]] = {}
    for agreement_uuid in ranked_agreements:
        context_orders: Set[int] = set()
        for page_order in target_orders_by_agreement[agreement_uuid]:
            context_orders.update({page_order - 1, page_order, page_order + 1})
        if len(context_orders) > SOURCE_VERDICT_MAX_CONTEXT_PAGES:
            continue
        context_orders_by_agreement[agreement_uuid] = {
            page_order for page_order in context_orders if page_order > 0
        }

    if not context_orders_by_agreement:
        return []

    context_page_rows = (
        conn.execute(
            text(
                f"""
                SELECT
                    p.agreement_uuid,
                    p.page_uuid,
                    p.page_order,
                    p.processed_page_content AS source_text,
                    COALESCE(
                        tgo.tagged_text_corrected,
                        tgo.tagged_text_gold,
                        tgo.tagged_text,
                        p.processed_page_content
                    ) AS tagged_text
                FROM {pages_table} p
                LEFT JOIN {tagged_outputs_table} tgo
                    ON tgo.page_uuid = p.page_uuid
                WHERE p.agreement_uuid IN :auuids
                ORDER BY p.agreement_uuid, p.page_order
                """
            ).bindparams(bindparam("auuids", expanding=True)),
            {"auuids": sorted(context_orders_by_agreement)},
        )
        .mappings()
        .fetchall()
    )

    page_contexts_by_agreement: Dict[str, List[Dict[str, Any]]] = {}
    target_page_sets = {
        agreement_uuid: set(target_page_uuids_by_agreement[agreement_uuid])
        for agreement_uuid in context_orders_by_agreement
    }
    for row in context_page_rows:
        agreement_uuid = str(row["agreement_uuid"])
        page_order = int(row["page_order"])
        if page_order not in context_orders_by_agreement.get(agreement_uuid, set()):
            continue
        page_uuid = str(row["page_uuid"])
        page_contexts_by_agreement.setdefault(agreement_uuid, []).append(
            {
                "page_uuid": page_uuid,
                "page_order": page_order,
                "is_flagged_page": page_uuid in target_page_sets[agreement_uuid],
                "source_text": str(row.get("source_text") or ""),
                "tagged_text": str(row.get("tagged_text") or ""),
            }
        )

    source_candidates: List[Dict[str, Any]] = []
    for agreement_uuid in ranked_agreements:
        page_contexts = page_contexts_by_agreement.get(agreement_uuid, [])
        if not page_contexts:
            continue
        target_page_uuids = target_page_uuids_by_agreement[agreement_uuid]
        source_candidates.append(
            {
                "agreement_uuid": agreement_uuid,
                "xml_version": int(xml_version_by_agreement[agreement_uuid]),
                "ai_repair_attempted": int(attempted_by_agreement[agreement_uuid]),
                "anchor_page_uuid": target_page_uuids[0],
                "target_page_uuids": target_page_uuids,
                "reason_rows": reasons_by_agreement[agreement_uuid],
                "page_contexts": page_contexts,
            }
        )
    return source_candidates


def _fetch_open_ai_repair_batch(
    conn: Connection,
    schema: str,
    *,
    batch_key: str | None = None,
) -> Dict[str, Any] | None:
    ai_repair_batches_table = f"{schema}.ai_repair_batches"
    if batch_key is None:
        q = text(
            f"""
            SELECT
                batch_id,
                status,
                completion_window,
                request_total,
                batch_key
            FROM {ai_repair_batches_table}
            WHERE status NOT IN :terminal_statuses
            ORDER BY created_at ASC
            LIMIT 1
            """
        ).bindparams(bindparam("terminal_statuses", expanding=True))
        row = conn.execute(q, {"terminal_statuses": list(_AI_REPAIR_TERMINAL_BATCH_STATUSES)}).mappings().first()
    else:
        q = text(
            f"""
            SELECT
                batch_id,
                status,
                completion_window,
                request_total,
                batch_key
            FROM {ai_repair_batches_table}
            WHERE status NOT IN :terminal_statuses
              AND batch_key = :batch_key
            ORDER BY created_at ASC
            LIMIT 1
            """
        ).bindparams(bindparam("terminal_statuses", expanding=True))
        row = conn.execute(
            q,
            {
                "terminal_statuses": list(_AI_REPAIR_TERMINAL_BATCH_STATUSES),
                "batch_key": batch_key,
            },
        ).mappings().first()

    if row is None:
        return None
    return dict(row)


def _fetch_open_ai_repair_batch_for_scope(
    conn: Connection,
    schema: str,
    *,
    scoped_agreement_uuids: list[str],
) -> Dict[str, Any] | None:
    matching_batch: Dict[str, Any] | None = None
    for model in (AI_REPAIR_FIRST_PASS_MODEL, AI_REPAIR_RETRY_MODEL, AI_REPAIR_SOURCE_VERDICT_MODEL):
        batch_key = _ai_repair_batch_key(
            agreement_uuids=scoped_agreement_uuids,
            model=model,
        )
        batch_row = _fetch_open_ai_repair_batch(conn, schema, batch_key=batch_key)
        if batch_row is None:
            continue
        if matching_batch is None or str(batch_row["batch_id"]) < str(matching_batch["batch_id"]):
            matching_batch = batch_row
    return matching_batch


def _ai_repair_batch_key(
    *,
    agreement_uuids: list[str],
    model: str,
    scope_tag: str | None = None,
) -> str:
    key_parts = [*agreement_uuids]
    if scope_tag:
        key_parts.append(f"scope:{scope_tag}")
    key_parts.append(f"model:{model}")
    return agreement_batch_key(key_parts)


def _fetch_batch_agreement_uuids(
    conn: Connection,
    schema: str,
    batch_id: str,
) -> List[str]:
    ai_repair_requests_table = f"{schema}.ai_repair_requests"
    pages_table = f"{schema}.pages"
    q = text(
        f"""
        SELECT DISTINCT p.agreement_uuid
        FROM {ai_repair_requests_table} r
        JOIN {pages_table} p
            ON p.page_uuid = r.page_uuid
        WHERE r.batch_id = :batch_id
        ORDER BY p.agreement_uuid
        """
    )
    return [str(v) for v in conn.execute(q, {"batch_id": batch_id}).scalars().all()]


def _insert_batch_row(
    conn: Connection,
    schema: str,
    batch: Any,
    completion_window: str,
    request_total: int,
    batch_key: str,
) -> None:
    ai_repair_batches_table = f"{schema}.ai_repair_batches"
    q = text(
        f"""
        INSERT INTO {ai_repair_batches_table}
            (batch_id, created_at, status, input_file_id, output_file_id, error_file_id,
             completion_window, request_total, request_failed, batch_key)
        VALUES
            (:batch_id, UTC_TIMESTAMP(), :status, :input_file_id, :output_file_id, :error_file_id,
             :cw, :rt, 0, :batch_key)
        ON DUPLICATE KEY UPDATE
            status = VALUES(status),
            input_file_id = VALUES(input_file_id),
            output_file_id = VALUES(output_file_id),
            error_file_id  = VALUES(error_file_id),
            completion_window = VALUES(completion_window),
            request_total  = VALUES(request_total),
            batch_key = VALUES(batch_key)
        """
    )
    _ = conn.execute(
        q,
        {
            "batch_id": batch.id,
            "status": batch.status,
            "input_file_id": getattr(batch, "input_file_id", None),
            "output_file_id": getattr(batch, "output_file_id", None),
            "error_file_id": getattr(batch, "error_file_id", None),
            "cw": completion_window,
            "rt": request_total,
            "batch_key": batch_key,
        },
    )


def _insert_requests(
    conn: Connection,
    schema: str,
    batch_id: str,
    lines_meta: List[Dict[str, Any]],
) -> None:
    """
    lines_meta: emitted by build_jsonl_lines_for_page(), one dict per custom_id:
        {request_id, page_uuid, mode, excerpt_start, excerpt_end}
    
    Only inserts new requests or updates requests with terminal statuses.
    Does not overwrite requests that are already queued or running.
    """
    ai_repair_requests_table = f"{schema}.ai_repair_requests"
    q = text(
        f"""
        INSERT INTO {ai_repair_requests_table}
            (request_id, batch_id, page_uuid, mode, excerpt_start, excerpt_end, created_at, status)
        VALUES
            (:rid, :bid, :pid, :mode, :xs, :xe, UTC_TIMESTAMP(), 'queued')
        ON DUPLICATE KEY UPDATE
            batch_id = CASE
                WHEN status IN ('queued', 'running') THEN batch_id
                ELSE VALUES(batch_id)
            END,
            status = CASE
                WHEN status IN ('queued', 'running') THEN status
                ELSE 'queued'
            END
        """
    )
    for m in lines_meta:
        _ = conn.execute(
            q,
            {
                "rid": m["request_id"],
                "bid": batch_id,
                "pid": m["page_uuid"],
                "mode": m["mode"],
                "xs": m["excerpt_start"],
                "xe": m["excerpt_end"],
            },
        )


def _mark_completed(conn: Connection, schema: str, request_ids: Set[str]) -> None:
    if not request_ids:
        return
    ai_repair_requests_table = f"{schema}.ai_repair_requests"
    ai_repair_processed_spans_table = f"{schema}.ai_repair_processed_spans"
    q = text(f"UPDATE {ai_repair_requests_table} SET status = 'completed' WHERE request_id IN :ids").bindparams(
        bindparam("ids", expanding=True)
    )
    _ = conn.execute(q, {"ids": list(request_ids)})

    # Also mark processed spans as completed
    q_spans = text(
        f"UPDATE {ai_repair_processed_spans_table} SET status = 'completed' WHERE request_id IN :ids"
    ).bindparams(bindparam("ids", expanding=True))
    _ = conn.execute(q_spans, {"ids": list(request_ids)})


def _mark_xml_ai_repair_attempted(
    conn: Connection,
    schema: str,
    agreement_uuids: Set[str],
) -> int:
    """
    Mark latest XML rows as having entered the AI-repair cycle.
    """
    if not agreement_uuids:
        return 0
    xml_table = f"{schema}.xml"
    q = text(
        f"""
        UPDATE {xml_table}
        SET ai_repair_attempted = 1
        WHERE latest = 1
          AND agreement_uuid IN :auuids
          AND NOT (ai_repair_attempted <=> 1)
        """
    ).bindparams(bindparam("auuids", expanding=True))
    result = conn.execute(q, {"auuids": sorted(agreement_uuids)})
    return int(result.rowcount or 0)


def _enqueue_ai_repair_for_agreements(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    *,
    target_agreement_uuids: list[str] | None,
    batch_scope_tag: str | None = None,
    log_prefix: str,
) -> List[str]:
    """
    Enqueue full-page AI retagging for XML-invalid target pages (by pageUUID).
    Target pages are derived from hard-rule XML violations for the latest invalid XML.
    """
    engine = db.get_engine()
    batch_completion_window = "24h"
    should_exit_after_tx = False
    resume_openai_batches = pipeline_config.resume_openai_batches
    explicit_scope = target_agreement_uuids is not None
    scoped_uuids = sorted(set(target_agreement_uuids or []))
    if explicit_scope and not scoped_uuids:
        context.log.info("%s: explicit empty scope; no AI repair work to enqueue.", log_prefix)
        run_post_asset_refresh(context, db, pipeline_config)
        return []

    enqueued_agreement_uuids: Set[str] = set()

    if target_agreement_uuids is not None and not scoped_uuids:
        context.log.info("%s: no upstream agreements to enqueue.", log_prefix)
        return []

    with engine.begin() as conn:
        assert_tables_exist(
            conn,
            schema=db.database,
            table_names=(
                "ai_repair_batches",
                "ai_repair_requests",
                "ai_repair_rulings",
                "ai_repair_full_pages",
                "ai_repair_source_verdicts",
                "ai_repair_processed_spans",
                "xml_status_reasons",
            ),
        )

        # 1) fetch candidate pages needing AI repair
        batch_size = pipeline_config.xml_agreement_batch_size
        page_budget = int(getattr(pipeline_config, "ai_repair_page_budget", 0) or 0)
        agreement_limit = (
            None
            if page_budget > 0
            else (max(batch_size, len(scoped_uuids)) if scoped_uuids else batch_size)
        )
        candidate_agreement_by_request_id: Dict[str, str] = {}

        preselected_candidates: List[Dict[str, Any]] | None = None
        if resume_openai_batches:
            matched_open_batch: Dict[str, Any] | None
            if scoped_uuids:
                matched_open_batch = _fetch_open_ai_repair_batch_for_scope(
                    conn,
                    db.database,
                    scoped_agreement_uuids=scoped_uuids,
                )
            elif batch_scope_tag is None:
                matched_open_batch = _fetch_open_ai_repair_batch(
                    conn,
                    db.database,
                )
            else:
                preselected_candidates = _fetch_candidates(
                    conn,
                    db.database,
                    agreement_limit=agreement_limit,
                    target_agreement_uuids=None,
                    page_budget=page_budget if page_budget > 0 else None,
                    attempt_priority=pipeline_config.ai_repair_attempt_priority,
                    exclude_in_flight=False,
                )
                matched_open_batch = None
                candidate_agreement_uuids_by_model: Dict[str, list[str]] = {}
                for model in (AI_REPAIR_FIRST_PASS_MODEL, AI_REPAIR_RETRY_MODEL):
                    model_agreement_uuids = sorted(
                        {
                            str(row["agreement_uuid"])
                            for row in preselected_candidates
                            if _repair_model_for_candidate(
                                int(row["ai_repair_attempted"]),
                                has_completed_requests=bool(int(row["has_completed_requests"])),
                            )
                            == model
                        }
                    )
                    if model_agreement_uuids:
                        candidate_agreement_uuids_by_model[model] = model_agreement_uuids

                for model, model_agreement_uuids in candidate_agreement_uuids_by_model.items():
                    batch_key = _ai_repair_batch_key(
                        agreement_uuids=model_agreement_uuids,
                        model=model,
                        scope_tag=batch_scope_tag,
                    )
                    batch_row = _fetch_open_ai_repair_batch(conn, db.database, batch_key=batch_key)
                    if batch_row is None:
                        continue
                    if matched_open_batch is None or str(batch_row["batch_id"]) < str(matched_open_batch["batch_id"]):
                        matched_open_batch = batch_row

            if matched_open_batch is not None:
                resumed_batch_id = str(matched_open_batch["batch_id"])
                resumed_agreement_uuids = _fetch_batch_agreement_uuids(
                    conn,
                    db.database,
                    resumed_batch_id,
                )
                if resumed_agreement_uuids:
                    enqueued_agreement_uuids.update(resumed_agreement_uuids)
                    context.log.info(
                        "%s: resuming stranded batch %s for %s agreements.",
                        log_prefix,
                        resumed_batch_id,
                        len(resumed_agreement_uuids),
                    )
                    should_exit_after_tx = True

        if should_exit_after_tx:
            pass
        else:
            candidates = (
                preselected_candidates
                if preselected_candidates is not None and batch_scope_tag is None
                else None
            )
            if candidates is None:
                candidates = _fetch_candidates(
                    conn,
                    db.database,
                    agreement_limit=agreement_limit,
                    target_agreement_uuids=scoped_uuids or None,
                    page_budget=page_budget if page_budget > 0 else None,
                    attempt_priority=pipeline_config.ai_repair_attempt_priority,
                )
            source_candidates = _fetch_source_verdict_candidates(
                conn,
                db.database,
                agreement_limit=agreement_limit,
                target_agreement_uuids=scoped_uuids or None,
                page_budget=page_budget if page_budget > 0 else None,
                attempt_priority=pipeline_config.ai_repair_attempt_priority,
            )
            if not candidates and not source_candidates:
                context.log.info("%s: no candidates.", log_prefix)
                should_exit_after_tx = True
            else:
                candidate_attempted_by_agreement = {
                    str(r["agreement_uuid"]): int(r["ai_repair_attempted"])
                    for r in [*candidates, *source_candidates]
                }
                candidate_agreement_count = len(candidate_attempted_by_agreement)
                candidate_already_attempted = sum(
                    1 for attempted in candidate_attempted_by_agreement.values() if attempted == 1
                )
                context.log.info(
                    "%s: selected agreements total=%s unattempted=%s already_attempted=%s target_pages=%s",
                    log_prefix,
                    candidate_agreement_count,
                    candidate_agreement_count - candidate_already_attempted,
                    candidate_already_attempted,
                    len(candidates),
                )
                context.log.info(
                    "%s: selected source-verdict agreements=%s",
                    log_prefix,
                    len(source_candidates),
                )
                context.log.info(
                    "%s: attempt priority=%s",
                    log_prefix,
                    pipeline_config.ai_repair_attempt_priority.value,
                )

            # 2) build full-page JSONL for targeted pages
            request_lines_by_model: Dict[str, List[Dict[str, Any]]] = {}
            lines_meta_by_model: Dict[str, List[Dict[str, Any]]] = {}
            request_count_by_model: Dict[str, int] = {}
            candidate_page_uuids = [str(row["page_uuid"]) for row in candidates]
            candidate_reason_codes = _fetch_candidate_page_reason_codes(
                conn,
                db.database,
                page_uuids=candidate_page_uuids,
            )
            candidate_xml_text_by_agreement = _fetch_latest_xml_text_by_agreement(
                conn,
                db.database,
                agreement_uuids=sorted(
                    {str(row["agreement_uuid"]) for row in candidates}
                ),
            )

            for row in candidates:
                page_uuid = str(row["page_uuid"])
                agreement_uuid = str(row["agreement_uuid"])
                text = str(row["text"])
                attempted = int(row["ai_repair_attempted"])
                has_completed_requests = bool(int(row["has_completed_requests"]))
                xml_version = int(row["xml_version"])
                page_reason_codes = candidate_reason_codes.get(page_uuid, set())
                toc_context: str | None = None
                if page_reason_codes == {XML_REASON_SECTION_NON_SEQUENTIAL}:
                    toc_context = _build_toc_context_for_page(
                        page_uuid,
                        candidate_xml_text_by_agreement.get(agreement_uuid, ""),
                    )
                model = _repair_model_for_candidate(
                    attempted,
                    has_completed_requests=has_completed_requests,
                )
                request_count_by_model[model] = request_count_by_model.get(model, 0) + 1
                full_decision = RepairDecision(
                    mode="full",
                    windows=[(0, len(text))],
                    token_map=[],
                )
                batch_lines, metas = build_jsonl_lines_for_page(
                    page_uuid=page_uuid,
                    text=text,
                    decision=full_decision,
                    model=model,
                    uncertain_spans=[],
                    xml_version=xml_version,
                    toc_context=toc_context,
                )
                request_lines_by_model.setdefault(model, []).extend(batch_lines)
                lines_meta_by_model.setdefault(model, []).extend(metas)
                for meta in metas:
                    candidate_agreement_by_request_id[str(meta["request_id"])] = agreement_uuid

            for row in source_candidates:
                agreement_uuid = str(row["agreement_uuid"])
                xml_version = int(row["xml_version"])
                line, meta = build_jsonl_line_for_source_text_verdict(
                    agreement_uuid=agreement_uuid,
                    xml_version=xml_version,
                    anchor_page_uuid=str(row["anchor_page_uuid"]),
                    model=AI_REPAIR_SOURCE_VERDICT_MODEL,
                    reason_rows=list(row["reason_rows"]),
                    page_contexts=list(row["page_contexts"]),
                )
                request_lines_by_model.setdefault(AI_REPAIR_SOURCE_VERDICT_MODEL, []).append(line)
                lines_meta_by_model.setdefault(AI_REPAIR_SOURCE_VERDICT_MODEL, []).append(meta)
                request_count_by_model[AI_REPAIR_SOURCE_VERDICT_MODEL] = (
                    request_count_by_model.get(AI_REPAIR_SOURCE_VERDICT_MODEL, 0) + 1
                )
                candidate_agreement_by_request_id[str(meta["request_id"])] = agreement_uuid

            all_lines_meta = [
                meta
                for metas in lines_meta_by_model.values()
                for meta in metas
            ]
            if not all_lines_meta:
                context.log.info("%s: nothing to enqueue.", log_prefix)
                should_exit_after_tx = True
            else:
                client = _oai_client()
                llm_agreement_uuids = sorted(
                    {
                        str(candidate_agreement_by_request_id[str(meta["request_id"])])
                        for meta in all_lines_meta
                    }
                )
                if not llm_agreement_uuids:
                    raise ValueError("ai_repair_enqueue_asset: failed to derive agreement UUIDs for enqueued lines.")
                context.log.info(
                    "%s: prepared requests by model=%s",
                    log_prefix,
                    dict(sorted(request_count_by_model.items())),
                )

                for model, model_lines in sorted(request_lines_by_model.items()):
                    model_metas = lines_meta_by_model[model]
                    model_batch_key = _ai_repair_batch_key(
                        agreement_uuids=llm_agreement_uuids,
                        model=model,
                        scope_tag=batch_scope_tag,
                    )
                    jsonl_full_buf = io.StringIO()
                    for line in model_lines:
                        _ = jsonl_full_buf.write(json.dumps(line, ensure_ascii=False) + "\n")
                    jsonl_bytes = io.BytesIO(jsonl_full_buf.getvalue().encode("utf-8"))
                    jsonl_bytes.name = f"ai_repair_requests_full_{model}.jsonl"
                    in_file = client.files.create(purpose="batch", file=jsonl_bytes)
                    batch = client.batches.create(
                        input_file_id=in_file.id,
                        endpoint="/v1/responses",
                        completion_window=batch_completion_window,
                    )

                    request_total = len(model_metas)
                    _insert_batch_row(
                        conn,
                        db.database,
                        batch,
                        batch_completion_window,
                        request_total,
                        model_batch_key,
                    )
                    _insert_requests(conn, db.database, batch.id, model_metas)
                    context.log.info(
                        "Enqueued OpenAI Batch %s (full, model=%s) with %s requests; input_file_id=%s",
                        batch.id,
                        model,
                        request_total,
                        in_file.id,
                    )

                    for meta in model_metas:
                        request_id = str(meta["request_id"])
                        if request_id not in candidate_agreement_by_request_id:
                            raise ValueError(
                                f"Missing agreement mapping for request_id={request_id} while marking XML AI-repair attempts."
                            )
                        enqueued_agreement_uuids.add(candidate_agreement_by_request_id[request_id])
                marked_rows = _mark_xml_ai_repair_attempted(
                    conn, db.database, enqueued_agreement_uuids
                )
                context.log.info(
                    "%s: marked ai_repair_attempted=1 on %s latest XML rows for %s agreements.",
                    log_prefix,
                    marked_rows,
                    len(enqueued_agreement_uuids),
                )

    if should_exit_after_tx:
        run_post_asset_refresh(context, db, pipeline_config)
        return sorted(enqueued_agreement_uuids)

    run_post_asset_refresh(context, db, pipeline_config)
    return sorted(enqueued_agreement_uuids)


def _select_ai_repair_scope(
    _context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> list[str]:
    engine = db.get_engine()
    schema = db.database
    with engine.begin() as conn:
        assert_tables_exist(
            conn,
            schema=schema,
            table_names=(
                "ai_repair_batches",
                "ai_repair_requests",
                "ai_repair_rulings",
                "ai_repair_full_pages",
                "ai_repair_source_verdicts",
                "ai_repair_processed_spans",
                "xml_status_reasons",
            ),
        )
        candidates = _fetch_candidates(
            conn,
            schema,
            agreement_limit=None if int(getattr(pipeline_config, "ai_repair_page_budget", 0) or 0) > 0 else pipeline_config.xml_agreement_batch_size,
            target_agreement_uuids=None,
            page_budget=int(getattr(pipeline_config, "ai_repair_page_budget", 0) or 0) or None,
            attempt_priority=pipeline_config.ai_repair_attempt_priority,
        )
        source_candidates = _fetch_source_verdict_candidates(
            conn,
            schema,
            agreement_limit=None if int(getattr(pipeline_config, "ai_repair_page_budget", 0) or 0) > 0 else pipeline_config.xml_agreement_batch_size,
            target_agreement_uuids=None,
            page_budget=int(getattr(pipeline_config, "ai_repair_page_budget", 0) or 0) or None,
            attempt_priority=pipeline_config.ai_repair_attempt_priority,
        )
    return sorted(
        {
            str(row["agreement_uuid"])
            for row in [*candidates, *source_candidates]
            if row.get("agreement_uuid")
        }
    )


@dg.asset(name="05-01_ai_repair_enqueue_asset")
def ai_repair_enqueue_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> List[str]:
    return _enqueue_ai_repair_for_agreements(
        context,
        db,
        pipeline_config,
        target_agreement_uuids=None,
        log_prefix="ai_repair_enqueue_asset",
    )


@dg.asset(
    name="05-01_regular_ingest_ai_repair_enqueue_asset",
    ins={"verified_agreement_uuids": dg.AssetIn(key=regular_ingest_xml_verify_asset.key)},
)
def regular_ingest_ai_repair_enqueue_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    verified_agreement_uuids: List[str],
) -> List[str]:
    should_skip, current_stage = should_skip_managed_stage(
        db=db,
        job_name="regular_ingest",
        stage_name="regular_ingest_ai_repair_enqueue",
    )
    if should_skip:
        context.log.info(
            "regular_ingest_ai_repair_enqueue_asset: skipping because logical run already reached %s.",
            current_stage,
        )
        return []
    scope_uuids = load_active_scope_for_job(
        context,
        db=db,
        job_name="regular_ingest",
        fallback_agreement_uuids=verified_agreement_uuids,
    )
    active_run = load_active_logical_run(db=db, job_name="regular_ingest")
    enqueued_agreement_uuids = _enqueue_ai_repair_for_agreements(
        context,
        db,
        pipeline_config,
        target_agreement_uuids=scope_uuids,
        batch_scope_tag=build_logical_batch_scope_tag(
            "regular_ingest",
            "regular_ingest_ai_repair_enqueue",
            None if active_run is None else str(active_run["logical_run_id"]),
        ),
        log_prefix="regular_ingest_ai_repair_enqueue_asset",
    )
    mark_logical_run_stage_completed(
        db=db,
        job_name="regular_ingest",
        stage_name="regular_ingest_ai_repair_enqueue",
    )
    return enqueued_agreement_uuids


@dg.asset(
    name="05-06_ingestion_cleanup_a_ai_repair_enqueue_asset",
    ins={"verified_agreement_uuids": dg.AssetIn(key=ingestion_cleanup_a_xml_verify_asset.key)},
)
def ingestion_cleanup_a_ai_repair_enqueue_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    verified_agreement_uuids: List[str],
) -> List[str]:
    should_skip, current_stage = should_skip_managed_stage(
        db=db,
        job_name="ingestion_cleanup_a",
        stage_name="ingestion_cleanup_a_ai_repair_enqueue",
    )
    if should_skip:
        context.log.info(
            "ingestion_cleanup_a_ai_repair_enqueue_asset: skipping because logical run already reached %s.",
            current_stage,
        )
        return []
    scope_uuids = load_active_scope_for_job(
        context,
        db=db,
        job_name="ingestion_cleanup_a",
        fallback_agreement_uuids=verified_agreement_uuids,
    )
    active_run = load_active_logical_run(db=db, job_name="ingestion_cleanup_a")
    enqueued_agreement_uuids = _enqueue_ai_repair_for_agreements(
        context,
        db,
        pipeline_config,
        target_agreement_uuids=scope_uuids,
        batch_scope_tag=build_logical_batch_scope_tag(
            "ingestion_cleanup_a",
            "ingestion_cleanup_a_ai_repair_enqueue",
            None if active_run is None else str(active_run["logical_run_id"]),
        ),
        log_prefix="ingestion_cleanup_a_ai_repair_enqueue_asset",
    )
    mark_logical_run_stage_completed(
        db=db,
        job_name="ingestion_cleanup_a",
        stage_name="ingestion_cleanup_a_ai_repair_enqueue",
    )
    return enqueued_agreement_uuids


@dg.asset(name="05-10_ingestion_cleanup_b_ai_repair_enqueue_asset")
def ingestion_cleanup_b_ai_repair_enqueue_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> List[str]:
    selected_scope = _select_ai_repair_scope(context, db, pipeline_config)
    logical_run = start_or_resume_logical_run(
        context,
        db=db,
        pipeline_config=pipeline_config,
        job_name="ingestion_cleanup_b",
        initial_stage="ingestion_cleanup_b_ai_repair_enqueue",
        selected_agreement_uuids=selected_scope,
    )
    scope_uuids = logical_run.agreement_uuids if logical_run is not None else []
    should_skip, current_stage = should_skip_managed_stage(
        db=db,
        job_name="ingestion_cleanup_b",
        stage_name="ingestion_cleanup_b_ai_repair_enqueue",
    )
    if should_skip:
        context.log.info(
            "ingestion_cleanup_b_ai_repair_enqueue_asset: skipping because logical run already reached %s.",
            current_stage,
        )
        return []
    enqueued_agreement_uuids = _enqueue_ai_repair_for_agreements(
        context,
        db,
        pipeline_config,
        target_agreement_uuids=scope_uuids,
        batch_scope_tag=build_logical_batch_scope_tag(
            "ingestion_cleanup_b",
            "ingestion_cleanup_b_ai_repair_enqueue",
            None if logical_run is None else logical_run.logical_run_id,
        ),
        log_prefix="ingestion_cleanup_b_ai_repair_enqueue_asset",
    )
    mark_logical_run_stage_completed(
        db=db,
        job_name="ingestion_cleanup_b",
        stage_name="ingestion_cleanup_b_ai_repair_enqueue",
    )
    return enqueued_agreement_uuids


def _request_counts(batch: Any) -> Tuple[int, int, int]:
    rc = batch.request_counts
    if rc is None:
        raise ValueError(f"Batch {batch.id} is missing request_counts.")
    total = rc.total
    failed = rc.failed
    completed = rc.completed
    if not isinstance(total, int) or not isinstance(failed, int) or not isinstance(completed, int):
        raise ValueError(f"Batch {batch.id} request_counts fields must be integers.")
    return total, failed, completed


def _bulk_update_status(
    conn: Connection, schema: str, request_ids: Set[str], status: str
) -> None:
    if not request_ids:
        return
    ai_repair_requests_table = f"{schema}.ai_repair_requests"
    ai_repair_processed_spans_table = f"{schema}.ai_repair_processed_spans"
    q = text(
        f"UPDATE {ai_repair_requests_table} SET status = :st WHERE request_id IN :ids"
    ).bindparams(bindparam("ids", expanding=True))
    _ = conn.execute(q, {"st": status, "ids": list(request_ids)})

    # Also update processed spans status
    q_spans = text(
        f"UPDATE {ai_repair_processed_spans_table} SET status = :st WHERE request_id IN :ids"
    ).bindparams(bindparam("ids", expanding=True))
    _ = conn.execute(q_spans, {"st": status, "ids": list(request_ids)})


def _extract_usage(body: Dict[str, Any]) -> Dict[str, Any]:
    usage = body["usage"]
    if not isinstance(usage, dict):
        raise ValueError("Expected usage to be an object.")
    required = ("input_tokens", "output_tokens", "total_tokens")
    for key in required:
        if key not in usage:
            raise ValueError(f"Missing usage field: {key}")
        if not isinstance(usage[key], int):
            raise ValueError(f"usage.{key} must be an integer.")
    details_keys = ("input_tokens_details", "output_tokens_details")
    for key in details_keys:
        if key in usage and not isinstance(usage[key], dict):
            raise ValueError(f"usage.{key} must be an object when present.")
    return usage


def _parse_full_page_tag_spans(raw: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """Return (request_id, spans) strictly or raise."""
    rid = raw["custom_id"]
    resp = raw["response"]
    sc = resp["status_code"]
    if sc not in (200, 201, 202):
        raise ValueError(f"Non-success status_code: {sc}")
    body = resp["body"]
    raw_text = extract_output_text_from_batch_body(body)
    obj = json.loads(raw_text)
    if not isinstance(obj, dict) or "spans" not in obj or "warnings" not in obj:
        raise ValueError("Missing 'spans' or 'warnings' in full-page output.")
    spans = obj["spans"]
    if not isinstance(spans, list):
        raise ValueError("'spans' must be a list.")
    parsed_spans: List[Dict[str, Any]] = []
    for span in spans:
        if not isinstance(span, dict):
            raise ValueError("span is not an object")
        start_char = span["start_char"]
        end_char = span["end_char"]
        label = span["label"]
        selected_text = span["selected_text"]
        if not isinstance(start_char, int) or not isinstance(end_char, int):
            raise ValueError("start_char/end_char must be integers")
        if label not in ("article", "section", "page"):
            raise ValueError(f"invalid span label: {label}")
        if not isinstance(selected_text, str):
            raise ValueError("selected_text must be a string")
        parsed_spans.append(
            {
                "start_char": start_char,
                "end_char": end_char,
                "label": label,
                "selected_text": selected_text,
            }
        )
    return rid, parsed_spans


def _parse_source_text_verdict(raw: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    rid = raw["custom_id"]
    resp = raw["response"]
    sc = resp["status_code"]
    if sc not in (200, 201, 202):
        raise ValueError(f"Non-success status_code: {sc}")
    body = resp["body"]
    raw_text = extract_output_text_from_batch_body(body)
    obj = json.loads(raw_text)
    if not isinstance(obj, dict):
        raise ValueError("source-text verdict output must be an object.")
    required = (
        "verdict",
        "source_text_is_correct",
        "flagged_pages_are_causal",
        "saw_full_problem_text",
        "confidence",
        "problem_summary",
        "evidence",
        "missing_or_duplicate_section_numbers",
        "warnings",
    )
    for key in required:
        if key not in obj:
            raise ValueError(f"Missing source-text verdict field: {key}")
    verdict = obj["verdict"]
    if verdict not in (
        "source_text_hard_rule_exception",
        "tagging_error_or_missing_tag",
        "insufficient_context",
    ):
        raise ValueError(f"invalid source-text verdict: {verdict}")
    for key in ("source_text_is_correct", "flagged_pages_are_causal", "saw_full_problem_text"):
        if not isinstance(obj[key], bool):
            raise ValueError(f"{key} must be boolean.")
    confidence = obj["confidence"]
    if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
        raise ValueError("confidence must be a number between 0 and 1.")
    if not isinstance(obj["problem_summary"], str):
        raise ValueError("problem_summary must be a string.")
    if not isinstance(obj["evidence"], list):
        raise ValueError("evidence must be a list.")
    for evidence in obj["evidence"]:
        if not isinstance(evidence, dict):
            raise ValueError("evidence entries must be objects.")
        for key in ("page_uuid", "quote", "interpretation"):
            if not isinstance(evidence.get(key), str):
                raise ValueError(f"evidence.{key} must be a string.")
    if not isinstance(obj["missing_or_duplicate_section_numbers"], list) or not all(
        isinstance(value, str) for value in obj["missing_or_duplicate_section_numbers"]
    ):
        raise ValueError("missing_or_duplicate_section_numbers must be a string list.")
    if not isinstance(obj["warnings"], list) or not all(isinstance(value, str) for value in obj["warnings"]):
        raise ValueError("warnings must be a string list.")
    return rid, {
        "verdict": str(verdict),
        "source_text_is_correct": bool(obj["source_text_is_correct"]),
        "flagged_pages_are_causal": bool(obj["flagged_pages_are_causal"]),
        "saw_full_problem_text": bool(obj["saw_full_problem_text"]),
        "confidence": float(confidence),
        "problem_summary": str(obj["problem_summary"]),
        "evidence": obj["evidence"],
        "missing_or_duplicate_section_numbers": obj["missing_or_duplicate_section_numbers"],
        "warnings": obj["warnings"],
    }


def _choose_best_alignment_candidate(
    candidates: List[Tuple[int, int]],
    *,
    approx_start: int,
    approx_end: int,
) -> Tuple[int, int]:
    if not candidates:
        raise ValueError("No alignment candidates available.")
    ranked = sorted(
        candidates,
        key=lambda item: (
            abs(item[0] - approx_start),
            abs(item[1] - approx_end),
            item[0],
            item[1],
        ),
    )
    best = ranked[0]
    if len(ranked) > 1:
        best_score = (
            abs(best[0] - approx_start),
            abs(best[1] - approx_end),
        )
        second = ranked[1]
        second_score = (
            abs(second[0] - approx_start),
            abs(second[1] - approx_end),
        )
        if second_score == best_score:
            raise ValueError("Span alignment is ambiguous near the provided offsets.")
    return best


def _find_nearby_exact_alignment(
    source_text: str,
    selected_text: str,
    *,
    approx_start: int,
    approx_end: int,
    radius: int = 128,
) -> Tuple[int, int] | None:
    if not selected_text:
        return None
    lo = max(0, approx_start - radius)
    hi = min(len(source_text), approx_end + radius)
    candidates: List[Tuple[int, int]] = []
    search_from = lo
    while search_from <= hi:
        idx = source_text.find(selected_text, search_from, hi)
        if idx == -1:
            break
        candidates.append((idx, idx + len(selected_text)))
        search_from = idx + 1
    if not candidates:
        return None
    return _choose_best_alignment_candidate(
        candidates,
        approx_start=approx_start,
        approx_end=approx_end,
    )


def _find_nearby_whitespace_insensitive_alignment(
    source_text: str,
    selected_text: str,
    *,
    approx_start: int,
    approx_end: int,
    radius: int = 128,
) -> Tuple[int, int] | None:
    selected_compact_chars = [ch for ch in selected_text if not ch.isspace()]
    if not selected_compact_chars:
        return None
    selected_compact = "".join(selected_compact_chars)

    lo = max(0, approx_start - radius)
    hi = min(len(source_text), approx_end + radius)
    window = source_text[lo:hi]
    compact_chars: List[str] = []
    compact_to_source: List[int] = []
    for idx, ch in enumerate(window):
        if ch.isspace():
            continue
        compact_chars.append(ch)
        compact_to_source.append(lo + idx)
    compact_window = "".join(compact_chars)
    if not compact_window:
        return None

    candidates: List[Tuple[int, int]] = []
    search_from = 0
    while search_from <= len(compact_window):
        idx = compact_window.find(selected_compact, search_from)
        if idx == -1:
            break
        source_start = compact_to_source[idx]
        source_end = compact_to_source[idx + len(selected_compact) - 1] + 1
        candidates.append((source_start, source_end))
        search_from = idx + 1
    if not candidates:
        return None
    return _choose_best_alignment_candidate(
        candidates,
        approx_start=approx_start,
        approx_end=approx_end,
    )


def _align_span_to_source(
    source_text: str,
    *,
    start_char: int,
    end_char: int,
    selected_text: str,
) -> Tuple[int, int, str]:
    if end_char <= start_char:
        raise ValueError("span offsets are out of bounds.")
    approx_start = min(max(start_char, 0), len(source_text))
    approx_end = min(max(end_char, 0), len(source_text))

    if start_char >= 0 and end_char <= len(source_text):
        direct_text = source_text[start_char:end_char]
    else:
        direct_text = ""
    if direct_text == selected_text:
        return start_char, end_char, direct_text

    exact_alignment = _find_nearby_exact_alignment(
        source_text,
        selected_text,
        approx_start=approx_start,
        approx_end=approx_end,
    )
    if exact_alignment is not None:
        aligned_start, aligned_end = exact_alignment
        return aligned_start, aligned_end, source_text[aligned_start:aligned_end]

    whitespace_alignment = _find_nearby_whitespace_insensitive_alignment(
        source_text,
        selected_text,
        approx_start=approx_start,
        approx_end=approx_end,
    )
    if whitespace_alignment is not None:
        aligned_start, aligned_end = whitespace_alignment
        return aligned_start, aligned_end, source_text[aligned_start:aligned_end]

    raise ValueError("selected_text could not be aligned to source text near the provided offsets.")


def _validate_full_page_tag_spans(
    source_text: str,
    spans: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    validated_spans: List[Dict[str, Any]] = []
    for span in spans:
        start_char = int(span["start_char"])
        end_char = int(span["end_char"])
        label = str(span["label"])
        selected_text = str(span["selected_text"])
        aligned_start, aligned_end, aligned_text = _align_span_to_source(
            source_text,
            start_char=start_char,
            end_char=end_char,
            selected_text=selected_text,
        )
        validated_spans.append(
            {
                "start_char": aligned_start,
                "end_char": aligned_end,
                "label": label,
                "selected_text": aligned_text,
            }
        )

    validated_spans.sort(key=lambda span: (int(span["start_char"]), int(span["end_char"])))
    previous_end = -1
    for span in validated_spans:
        start_char = int(span["start_char"])
        end_char = int(span["end_char"])
        if start_char < previous_end:
            raise ValueError("spans overlap or nest.")
        previous_end = end_char
    return validated_spans


def _apply_full_page_tag_spans(
    source_text: str,
    spans: List[Dict[str, Any]],
) -> str:
    validated_spans = _validate_full_page_tag_spans(source_text, spans)
    pieces: List[str] = []
    cursor = 0
    for span in validated_spans:
        start_char = int(span["start_char"])
        end_char = int(span["end_char"])
        label = str(span["label"])
        pieces.append(source_text[cursor:start_char])
        pieces.append(f"<{label}>")
        pieces.append(source_text[start_char:end_char])
        pieces.append(f"</{label}>")
        cursor = end_char
    pieces.append(source_text[cursor:])
    tagged_text = "".join(pieces)
    reconstructed_source = tagged_text
    for tag in _ALLOWED_FULL_MODE_TAGS:
        reconstructed_source = reconstructed_source.replace(tag, "")
    if reconstructed_source != source_text:
        raise ValueError("tag insertion produced non-source-preserving output.")
    return tagged_text


def _parse_excerpt_rulings(raw: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """Return (request_id, rulings) strictly or raise."""
    rid = raw["custom_id"]
    resp = raw["response"]
    sc = resp["status_code"]
    if sc not in (200, 201, 202):
        raise ValueError(f"Non-success status_code: {sc}")
    body = resp["body"]
    raw_text = extract_output_text_from_batch_body(body)
    obj = json.loads(raw_text)
    if not isinstance(obj, dict) or "rulings" not in obj or "warnings" not in obj:
        raise ValueError("Missing 'rulings' or 'warnings' in excerpt output.")
    rulings = obj["rulings"]
    if not isinstance(rulings, list):
        raise ValueError("'rulings' must be a list.")
    out: List[Dict[str, Any]] = []
    for r in rulings:
        if not isinstance(r, dict):
            raise ValueError("ruling is not an object")
        s = r["start_char"]
        e = r["end_char"]
        lab = r["label"]
        if not isinstance(s, int) or not isinstance(e, int):
            raise ValueError("start_char/end_char must be integers")
        if lab not in ("article", "section", "page", "none"):
            raise ValueError(f"invalid ruling label: {lab}")
        out.append({"start_char": s, "end_char": e, "label": lab})
    return rid, out


def _source_reason_rows_are_bypass_eligible(
    reason_rows: List[Dict[str, Any]],
) -> Tuple[bool, Set[str]]:
    if not reason_rows:
        return False, set()
    reason_codes = {str(row["reason_code"]) for row in reason_rows}
    if reason_codes != set(AI_REPAIR_SOURCE_VERDICT_XML_REASON_CODES):
        return False, set()
    page_uuids: Set[str] = set()
    for row in reason_rows:
        page_uuid = row.get("page_uuid")
        if page_uuid is None or not str(page_uuid).strip():
            return False, set()
        page_uuids.add(str(page_uuid))
    return bool(page_uuids), page_uuids


def apply_source_text_bypasses_for_agreements(
    conn: Connection,
    schema: str,
    agreement_uuids: List[str],
) -> int:
    """
    Mark latest XML verified when a high-confidence source-text verdict covers
    the current hard-rule failure exactly enough to bypass normal validation.
    """
    target_agreement_uuids = sorted(set(agreement_uuids))
    if not target_agreement_uuids:
        return 0

    xml_table = f"{schema}.xml"
    xml_status_reasons_table = f"{schema}.xml_status_reasons"
    ai_repair_source_verdicts_table = f"{schema}.ai_repair_source_verdicts"

    xml_rows = (
        conn.execute(
            text(
                f"""
                SELECT agreement_uuid, version
                FROM {xml_table}
                WHERE latest = 1
                  AND status = 'invalid'
                  AND agreement_uuid IN :auuids
                """
            ).bindparams(bindparam("auuids", expanding=True)),
            {"auuids": target_agreement_uuids},
        )
        .mappings()
        .fetchall()
    )
    if not xml_rows:
        return 0

    current_versions = {str(row["agreement_uuid"]): int(row["version"]) for row in xml_rows}
    reason_rows = (
        conn.execute(
            text(
                f"""
                SELECT agreement_uuid, xml_version, reason_code, reason_detail, page_uuid
                FROM {xml_status_reasons_table}
                WHERE agreement_uuid IN :auuids
                ORDER BY agreement_uuid, id
                """
            ).bindparams(bindparam("auuids", expanding=True)),
            {"auuids": sorted(current_versions)},
        )
        .mappings()
        .fetchall()
    )
    current_reasons_by_agreement: Dict[str, List[Dict[str, Any]]] = {}
    for row in reason_rows:
        agreement_uuid = str(row["agreement_uuid"])
        if int(row["xml_version"]) != current_versions.get(agreement_uuid):
            continue
        current_reasons_by_agreement.setdefault(agreement_uuid, []).append(
            {
                "reason_code": str(row["reason_code"]),
                "reason_detail": None if row.get("reason_detail") is None else str(row["reason_detail"]),
                "page_uuid": None if row.get("page_uuid") is None else str(row["page_uuid"]),
            }
        )

    verdict_rows = (
        conn.execute(
            text(
                f"""
                SELECT
                    request_id,
                    agreement_uuid,
                    xml_version,
                    verdict,
                    source_text_is_correct,
                    flagged_pages_are_causal,
                    saw_full_problem_text,
                    confidence,
                    problem_summary,
                    target_page_uuids_json
                FROM {ai_repair_source_verdicts_table}
                WHERE agreement_uuid IN :auuids
                  AND verdict = 'source_text_hard_rule_exception'
                  AND source_text_is_correct = 1
                  AND flagged_pages_are_causal = 1
                  AND saw_full_problem_text = 1
                  AND confidence >= :confidence
                ORDER BY agreement_uuid, created_at DESC
                """
            ).bindparams(bindparam("auuids", expanding=True)),
            {
                "auuids": sorted(current_versions),
                "confidence": SOURCE_TEXT_BYPASS_CONFIDENCE_THRESHOLD,
            },
        )
        .mappings()
        .fetchall()
    )
    verdicts_by_agreement: Dict[str, List[Dict[str, Any]]] = {}
    for row in verdict_rows:
        verdicts_by_agreement.setdefault(str(row["agreement_uuid"]), []).append(dict(row))

    update_verified = text(
        f"""
        UPDATE {xml_table}
        SET status = 'verified',
            status_source = 'asset',
            status_reason_code = :reason_code,
            status_reason_detail = :reason_detail
        WHERE agreement_uuid = :agreement_uuid
          AND version = :version
          AND latest = 1
          AND status = 'invalid'
        """
    )
    delete_reasons = text(
        f"""
        DELETE FROM {xml_status_reasons_table}
        WHERE agreement_uuid = :agreement_uuid
          AND xml_version = :version
        """
    )

    bypassed = 0
    for agreement_uuid, version in current_versions.items():
        eligible, current_page_uuids = _source_reason_rows_are_bypass_eligible(
            current_reasons_by_agreement.get(agreement_uuid, [])
        )
        if not eligible:
            continue
        selected_verdict: Dict[str, Any] | None = None
        for verdict_row in verdicts_by_agreement.get(agreement_uuid, []):
            try:
                verdict_page_uuids = {
                    str(page_uuid)
                    for page_uuid in json.loads(str(verdict_row["target_page_uuids_json"] or "[]"))
                }
            except json.JSONDecodeError:
                continue
            if current_page_uuids.issubset(verdict_page_uuids):
                selected_verdict = verdict_row
                break
        if selected_verdict is None:
            continue

        reason_detail = (
            f"LLM source-text bypass via {selected_verdict['request_id']}: "
            f"{str(selected_verdict['problem_summary'])[:900]}"
        )
        result = conn.execute(
            update_verified,
            {
                "agreement_uuid": agreement_uuid,
                "version": version,
                "reason_code": XML_REASON_LLM_SOURCE_TEXT_BYPASS,
                "reason_detail": reason_detail,
            },
        )
        if int(result.rowcount or 0) > 0:
            _ = conn.execute(
                delete_reasons,
                {"agreement_uuid": agreement_uuid, "version": version},
            )
            bypassed += 1
    return bypassed


def _poll_ai_repair_batches(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    enqueued_agreement_uuids: List[str],
    *,
    log_prefix: str,
) -> List[str]:
    """
    Poll terminal batches, read output/error JSONL, and persist parsed entities strictly.

    Status handling:
      - Parsed OK → status = 'completed'
      - HTTP success but parse failed → status = 'parse_error'
      - Error-file entries → status = 'failed'
      - No output/no error → status = 'completed_no_output'

    Returns:
      - request_ids for full-page outputs and source-text verdicts successfully parsed in this poll run.
    """
    engine = db.get_engine()
    schema = db.database
    ai_repair_batches_table = f"{schema}.ai_repair_batches"
    ai_repair_requests_table = f"{schema}.ai_repair_requests"
    ai_repair_processed_spans_table = f"{schema}.ai_repair_processed_spans"
    ai_repair_full_pages_table = f"{schema}.ai_repair_full_pages"
    ai_repair_rulings_table = f"{schema}.ai_repair_rulings"
    ai_repair_source_verdicts_table = f"{schema}.ai_repair_source_verdicts"
    xml_status_reasons_table = f"{schema}.xml_status_reasons"
    pages_table = f"{schema}.pages"
    client = _oai_client()
    target_agreement_uuids = sorted(set(enqueued_agreement_uuids))

    if not target_agreement_uuids:
        context.log.info("%s: no upstream agreements from enqueue.", log_prefix)
        run_post_asset_refresh(context, db, pipeline_config)
        return []

    successful_request_ids: Set[str] = set()
    terminal_failed_batches: List[Tuple[str, str, int, int]] = []

    base_sleep_seconds = 5
    backoff_level = 0
    no_update_polls = 0
    last_progress_snapshot: Optional[Tuple[Tuple[str, str, int, int], ...]] = None

    while True:
        running_progress: List[Dict[str, Any]] = []
        with engine.begin() as conn:
            rows = (
                conn.execute(
                    text(
                        f"""
                        SELECT DISTINCT b.batch_id
                        FROM {ai_repair_batches_table} b
                        JOIN {ai_repair_requests_table} r
                            ON r.batch_id = b.batch_id
                        JOIN {pages_table} p
                            ON p.page_uuid = r.page_uuid
                        WHERE b.status NOT IN ('completed','failed','cancelled','expired')
                          AND p.agreement_uuid IN :auuids
                        ORDER BY b.created_at ASC
                        LIMIT 20
                        """
                    ).bindparams(bindparam("auuids", expanding=True)),
                    {"auuids": target_agreement_uuids},
                )
                .mappings()
                .fetchall()
            )
            if not rows:
                context.log.info("%s: no batches to poll.", log_prefix)
                break

            upd_batch = text(
                f"""
                UPDATE {ai_repair_batches_table}
                SET status=:st, output_file_id=:of, error_file_id=:ef,
                    request_total=:rt, request_failed=:rf
                WHERE batch_id=:bid
                """
            )
            select_req = text(
                f"SELECT request_id, page_uuid, mode, excerpt_start FROM {ai_repair_requests_table} WHERE batch_id = :bid"
            )
            mark_running = text(
                f"UPDATE {ai_repair_requests_table} SET status='running' WHERE batch_id=:bid AND status='queued'"
            )
            mark_running_spans = text(
                f"UPDATE {ai_repair_processed_spans_table} SET status='running' WHERE batch_id=:bid AND status='queued'"
            )

            for r in rows:
                bid = r["batch_id"]
                b = client.batches.retrieve(bid)

                # Update batch row with fresh metadata
                total, failed, completed = _request_counts(b)

                _ = conn.execute(
                    upd_batch,
                    {
                        "st": b.status,
                        "of": getattr(b, "output_file_id", None),
                        "ef": getattr(b, "error_file_id", None),
                        "rt": total,
                        "rf": failed,
                        "bid": bid,
                    },
                )

                if b.status not in ("completed", "failed", "cancelled", "expired"):
                    _ = conn.execute(mark_running, {"bid": bid})
                    _ = conn.execute(mark_running_spans, {"bid": bid})
                    done = completed + failed
                    pct = int((done / total) * 100) if total else 0
                    running_progress.append(
                        {
                            "bid": bid,
                            "status": b.status,
                            "total": total,
                            "failed": failed,
                            "completed": completed,
                            "pct": pct,
                        }
                    )
                    continue

                # Terminal batch: fetch request metadata
                req_info = {
                    row.request_id: (row.page_uuid, row.mode, row.excerpt_start)
                    for row in conn.execute(select_req, {"bid": bid}).mappings().fetchall()
                }
                req_ids_all = set(req_info.keys())
                req_page_uuids = sorted({str(meta[0]) for meta in req_info.values()})
                page_text_by_uuid: Dict[str, str] = {}
                if req_page_uuids:
                    page_rows = (
                        conn.execute(
                            text(
                                f"""
                                SELECT page_uuid, processed_page_content AS text
                                FROM {pages_table}
                                WHERE page_uuid IN :pids
                                """
                            ).bindparams(bindparam("pids", expanding=True)),
                            {"pids": req_page_uuids},
                        )
                        .mappings()
                        .fetchall()
                    )
                    page_text_by_uuid = {
                        str(row["page_uuid"]): str(row.get("text") or "")
                        for row in page_rows
                    }

                batch_failed_terminal = b.status in _AI_REPAIR_FAILED_BATCH_STATUSES
                if batch_failed_terminal:
                    terminal_failed_batches.append((str(bid), str(b.status), total, failed))

                success_ids: Set[str] = set()
                http_success_ids: Set[str] = set()
                failed_ids: Set[str] = set()
                parse_error_ids: Set[str] = set()
                usage_by_request: Dict[str, Dict[str, int]] = {}

                # Parsed excerpt rulings and full pages
                parsed_rulings: List[Tuple[str, str, List[Dict[str, Any]]]] = []  # (rid, page_uuid, rulings)
                parsed_full_pages: List[Tuple[str, str, str]] = []  # (rid, page_uuid, tagged_text)
                parsed_source_verdicts: List[Tuple[str, str, int, Dict[str, Any]]] = []  # (rid, agreement_uuid, version, verdict)

                # Process output JSONL (success lines)
                ofid = getattr(b, "output_file_id", None)
                if ofid:
                    out_text = read_openai_file_text(client.files.content(ofid)).strip()
                    if out_text:
                        for line in out_text.splitlines():
                            raw = json.loads(line)
                            rid = raw["custom_id"]
                            resp = raw["response"]
                            sc = resp["status_code"]
                            if sc in (200, 201, 202):
                                http_success_ids.add(rid)
                            try:
                                if sc in (200, 201, 202):
                                    usage_by_request[rid] = _extract_usage(resp["body"])
                                pid, mode, xs = req_info[rid]
                                if mode == "full":
                                    rid2, spans = _parse_full_page_tag_spans(raw)
                                    source_text = page_text_by_uuid.get(str(pid))
                                    if source_text is None:
                                        raise ValueError(
                                            f"Missing page text for page_uuid={pid}."
                                        )
                                    tagged_text = _apply_full_page_tag_spans(
                                        source_text, spans
                                    )
                                    parsed_full_pages.append((rid2, pid, tagged_text))
                                    success_ids.add(rid2)
                                elif mode == "excerpt":
                                    rid2, rulings = _parse_excerpt_rulings(raw)
                                    parsed_rulings.append((rid2, pid, rulings))
                                    success_ids.add(rid2)
                                elif mode == "source":
                                    rid2, verdict = _parse_source_text_verdict(raw)
                                    agreement_uuid, xml_version = _parse_source_request_id(rid2)
                                    parsed_source_verdicts.append(
                                        (rid2, agreement_uuid, xml_version, verdict)
                                    )
                                    success_ids.add(rid2)
                                else:
                                    raise ValueError(f"Unexpected request mode {mode!r} for {rid}.")
                            except (ValueError, KeyError, TypeError) as e:
                                # Parse errors: malformed JSON, missing fields, type mismatches
                                context.log.warning(
                                    f"Batch {bid}: parse error for request {rid}: {e}"
                                )
                                parse_error_ids.add(rid)
                            except Exception as e:
                                # Unexpected errors: log and mark as parse error
                                context.log.error(
                                    f"Batch {bid}: unexpected error parsing request {rid}: {e}",
                                    exc_info=True,
                                )
                                parse_error_ids.add(rid)
                    else:
                        context.log.warning(f"Batch {bid} has empty output content.")

                # Process error JSONL (explicit failures)
                efid = getattr(b, "error_file_id", None)
                if efid:
                    err_text = read_openai_file_text(client.files.content(efid)).strip()
                    if err_text:
                        for line in err_text.splitlines():
                            err = json.loads(line)
                            rid = err["custom_id"]
                            failed_ids.add(rid)

                # Persist parsed data

                if usage_by_request:
                    upd_usage = text(
                        f"UPDATE {ai_repair_requests_table} SET token_usage = :usage WHERE request_id = :rid"
                    )
                    for rid, usage in usage_by_request.items():
                        _ = conn.execute(
                            upd_usage, {"rid": rid, "usage": json.dumps(usage)}
                        )

                # Full-page tagged_text
                if parsed_full_pages:
                    ins_full = text(
                        f"""
                        INSERT INTO {ai_repair_full_pages_table} (request_id, page_uuid, tagged_text, batch_id)
                        VALUES (:rid, :pid, :txt, :bid)
                        ON DUPLICATE KEY UPDATE tagged_text = VALUES(tagged_text), batch_id = VALUES(batch_id)
                        """
                    )
                    for rid2, pid, txt in parsed_full_pages:
                        _ = conn.execute(
                            ins_full,
                            {"rid": rid2, "pid": pid, "txt": txt, "bid": bid},
                        )

                # Excerpt rulings (convert to page-level coords using excerpt_start)
                if parsed_rulings:
                    ins_r = text(
                        f"""
                        INSERT INTO {ai_repair_rulings_table} (request_id, page_uuid, start_char, end_char, label, batch_id)
                        VALUES (:rid, :pid, :s, :e, :lab, :bid)
                        ON DUPLICATE KEY UPDATE label = VALUES(label), batch_id = VALUES(batch_id)
                        """
                    )
                    for rid2, pid, rulings in parsed_rulings:
                        _, _, xs = req_info[rid2]
                        if not isinstance(xs, int):
                            raise ValueError(f"Missing excerpt_start for request {rid2}.")
                        base = xs
                        for r in rulings:
                            s_adj = int(r["start_char"]) + base
                            e_adj = int(r["end_char"]) + base
                            _ = conn.execute(
                                ins_r,
                                {"rid": rid2, "pid": pid, "s": s_adj, "e": e_adj, "lab": r["label"], "bid": bid},
                            )

                if parsed_source_verdicts:
                    source_reason_rows_by_request: Dict[str, List[Dict[str, Any]]] = {}
                    select_source_reasons = text(
                        f"""
                        SELECT reason_code, reason_detail, page_uuid
                        FROM {xml_status_reasons_table}
                        WHERE agreement_uuid = :agreement_uuid
                          AND xml_version = :xml_version
                        ORDER BY id
                        """
                    )
                    for rid2, agreement_uuid, xml_version, _verdict in parsed_source_verdicts:
                        source_reason_rows = [
                            {
                                "reason_code": str(reason_row["reason_code"]),
                                "reason_detail": None if reason_row.get("reason_detail") is None else str(reason_row["reason_detail"]),
                                "page_uuid": None if reason_row.get("page_uuid") is None else str(reason_row["page_uuid"]),
                            }
                            for reason_row in conn.execute(
                                select_source_reasons,
                                {
                                    "agreement_uuid": agreement_uuid,
                                    "xml_version": xml_version,
                                },
                            ).mappings().fetchall()
                        ]
                        source_reason_rows_by_request[rid2] = source_reason_rows

                    ins_source = text(
                        f"""
                        INSERT INTO {ai_repair_source_verdicts_table} (
                            request_id,
                            agreement_uuid,
                            xml_version,
                            verdict,
                            source_text_is_correct,
                            flagged_pages_are_causal,
                            saw_full_problem_text,
                            confidence,
                            problem_summary,
                            reason_rows_json,
                            target_page_uuids_json,
                            evidence_json,
                            missing_or_duplicate_section_numbers_json,
                            warnings_json,
                            batch_id,
                            created_at
                        )
                        VALUES (
                            :rid,
                            :agreement_uuid,
                            :xml_version,
                            :verdict,
                            :source_text_is_correct,
                            :flagged_pages_are_causal,
                            :saw_full_problem_text,
                            :confidence,
                            :problem_summary,
                            :reason_rows_json,
                            :target_page_uuids_json,
                            :evidence_json,
                            :missing_or_duplicate_section_numbers_json,
                            :warnings_json,
                            :bid,
                            UTC_TIMESTAMP()
                        )
                        ON DUPLICATE KEY UPDATE
                            verdict = VALUES(verdict),
                            source_text_is_correct = VALUES(source_text_is_correct),
                            flagged_pages_are_causal = VALUES(flagged_pages_are_causal),
                            saw_full_problem_text = VALUES(saw_full_problem_text),
                            confidence = VALUES(confidence),
                            problem_summary = VALUES(problem_summary),
                            reason_rows_json = VALUES(reason_rows_json),
                            target_page_uuids_json = VALUES(target_page_uuids_json),
                            evidence_json = VALUES(evidence_json),
                            missing_or_duplicate_section_numbers_json = VALUES(missing_or_duplicate_section_numbers_json),
                            warnings_json = VALUES(warnings_json),
                            batch_id = VALUES(batch_id)
                        """
                    )
                    for rid2, agreement_uuid, xml_version, verdict in parsed_source_verdicts:
                        source_reason_rows = source_reason_rows_by_request.get(rid2, [])
                        target_page_uuids = sorted(
                            {
                                str(row["page_uuid"])
                                for row in source_reason_rows
                                if row.get("page_uuid") is not None
                            }
                        )
                        _ = conn.execute(
                            ins_source,
                            {
                                "rid": rid2,
                                "agreement_uuid": agreement_uuid,
                                "xml_version": xml_version,
                                "verdict": verdict["verdict"],
                                "source_text_is_correct": int(bool(verdict["source_text_is_correct"])),
                                "flagged_pages_are_causal": int(bool(verdict["flagged_pages_are_causal"])),
                                "saw_full_problem_text": int(bool(verdict["saw_full_problem_text"])),
                                "confidence": float(verdict["confidence"]),
                                "problem_summary": verdict["problem_summary"],
                                "reason_rows_json": json.dumps(source_reason_rows, ensure_ascii=False),
                                "target_page_uuids_json": json.dumps(target_page_uuids, ensure_ascii=False),
                                "evidence_json": json.dumps(verdict["evidence"], ensure_ascii=False),
                                "missing_or_duplicate_section_numbers_json": json.dumps(
                                    verdict["missing_or_duplicate_section_numbers"],
                                    ensure_ascii=False,
                                ),
                                "warnings_json": json.dumps(verdict["warnings"], ensure_ascii=False),
                                "bid": bid,
                            },
                        )

                parsed_full_ids = set(rid for rid, _, _ in parsed_full_pages)
                parsed_source_ids = set(rid for rid, _, _, _ in parsed_source_verdicts)
                parsed_ids = parsed_full_ids | parsed_source_ids | set(rid for rid, _, _ in parsed_rulings)

                # Mark requests completed that produced outputs (either kind)
                _mark_completed(conn, db.database, parsed_ids)
                successful_request_ids.update(parsed_full_ids | parsed_source_ids)

                # Completed with HTTP success but no parsed record (and not failed/parse_error)
                no_output_ids = (
                    (http_success_ids - parsed_ids) - failed_ids - parse_error_ids
                )
                _bulk_update_status(conn, db.database, no_output_ids, "completed_no_output")

                # Parse errors on HTTP-success lines
                _bulk_update_status(
                    conn, db.database, parse_error_ids - failed_ids - parsed_ids, "parse_error"
                )

                # Explicit failures
                _bulk_update_status(conn, db.database, failed_ids, "failed")

                # Leftovers on completed batches produced no output/error payloads.
                # Leftovers on expired/failed/cancelled batches never returned and
                # should be retried on a future enqueue.
                leftover_ids = req_ids_all - http_success_ids - failed_ids - parse_error_ids
                leftover_status = "failed" if batch_failed_terminal else "completed_no_output"
                _bulk_update_status(conn, db.database, leftover_ids, leftover_status)

                # Summary
                context.log.info(
                    "Batch %s: success=%s failed=%s parse_error=%s no_output=%s leftover=%s"
                    % (
                        bid,
                        len(success_ids),
                        len(failed_ids),
                        len(parse_error_ids),
                        len(no_output_ids),
                        len(leftover_ids),
                    )
                )

        # Emit progress summary for running batches outside of transaction
        if running_progress:
            context.log.info(f"{len(running_progress)} batches running.")
            for p in running_progress:
                context.log.info(
                    f"Batch {p['bid']}: status={p['status']}, {(p['completed'] + p['failed'])}/{p['total']} done ({p['pct']}%), failed={p['failed']}"
                )

        # If nothing is running, exit; otherwise, wait and poll again with backoff
        if not running_progress:
            break

        progress_snapshot = tuple(
            sorted(
                (p["bid"], p["status"], p["completed"], p["failed"]) for p in running_progress
            )
        )
        if progress_snapshot == last_progress_snapshot:
            no_update_polls += 1
        else:
            if backoff_level > 0:
                prev_sleep = base_sleep_seconds * (2**backoff_level)
                context.log.info(
                    f"Backoff reset: interval {prev_sleep}s -> {base_sleep_seconds}s"
                )
            no_update_polls = 0
            backoff_level = 0
            last_progress_snapshot = progress_snapshot

        max_sleep_seconds = 30 * 60
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
                context.log.info(f"Backoff increased: interval {prev_sleep}s -> {new_sleep}s")

        time.sleep(min(base_sleep_seconds * (2**backoff_level), max_sleep_seconds))

    with engine.begin() as conn:
        unresolved_requests = int(
            conn.execute(
                text(
                    f"""
                    SELECT COUNT(*)
                    FROM {ai_repair_requests_table} r
                    JOIN {pages_table} p
                        ON p.page_uuid = r.page_uuid
                    WHERE p.agreement_uuid IN :auuids
                      AND r.status IN ('queued', 'running')
                    """
                ).bindparams(bindparam("auuids", expanding=True)),
                {"auuids": target_agreement_uuids},
            ).scalar_one()
        )
    if unresolved_requests > 0:
        raise RuntimeError(
            f"ai_repair_poll_asset: {unresolved_requests} queued/running requests remain unresolved for this run."
        )
    if terminal_failed_batches:
        batch_summaries = ", ".join(
            f"{batch_id}[{status} total={total} failed={failed}]"
            for batch_id, status, total, failed in terminal_failed_batches
        )
        raise RuntimeError(
            "ai_repair_poll_asset: terminal failed/cancelled/expired batches detected; "
            + f"stopping run. batches={batch_summaries}"
        )

    run_post_asset_refresh(context, db, pipeline_config)
    return sorted(successful_request_ids)


@dg.asset(
    name="05-02_ai_repair_poll_asset",
    ins={"enqueued_agreement_uuids": dg.AssetIn(key=ai_repair_enqueue_asset.key)},
)
def ai_repair_poll_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    enqueued_agreement_uuids: List[str],
) -> List[str]:
    return _poll_ai_repair_batches(
        context,
        db,
        pipeline_config,
        enqueued_agreement_uuids,
        log_prefix="ai_repair_poll_asset",
    )


@dg.asset(
    name="05-02_regular_ingest_ai_repair_poll_asset",
    ins={"enqueued_agreement_uuids": dg.AssetIn(key=regular_ingest_ai_repair_enqueue_asset.key)},
)
def regular_ingest_ai_repair_poll_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    enqueued_agreement_uuids: List[str],
) -> List[str]:
    should_skip, current_stage = should_skip_managed_stage(
        db=db,
        job_name="regular_ingest",
        stage_name="regular_ingest_ai_repair_poll",
    )
    if should_skip:
        context.log.info(
            "regular_ingest_ai_repair_poll_asset: skipping because logical run already reached %s.",
            current_stage,
        )
        return []
    successful_request_ids = _poll_ai_repair_batches(
        context,
        db,
        pipeline_config,
        load_active_scope_for_job(
            context,
            db=db,
            job_name="regular_ingest",
            fallback_agreement_uuids=enqueued_agreement_uuids,
        ),
        log_prefix="regular_ingest_ai_repair_poll_asset",
    )
    mark_logical_run_stage_completed(
        db=db,
        job_name="regular_ingest",
        stage_name="regular_ingest_ai_repair_poll",
    )
    return successful_request_ids


@dg.asset(
    name="05-07_ingestion_cleanup_a_ai_repair_poll_asset",
    ins={"enqueued_agreement_uuids": dg.AssetIn(key=ingestion_cleanup_a_ai_repair_enqueue_asset.key)},
)
def ingestion_cleanup_a_ai_repair_poll_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    enqueued_agreement_uuids: List[str],
) -> List[str]:
    should_skip, current_stage = should_skip_managed_stage(
        db=db,
        job_name="ingestion_cleanup_a",
        stage_name="ingestion_cleanup_a_ai_repair_poll",
    )
    if should_skip:
        context.log.info(
            "ingestion_cleanup_a_ai_repair_poll_asset: skipping because logical run already reached %s.",
            current_stage,
        )
        return []
    successful_request_ids = _poll_ai_repair_batches(
        context,
        db,
        pipeline_config,
        load_active_scope_for_job(
            context,
            db=db,
            job_name="ingestion_cleanup_a",
            fallback_agreement_uuids=enqueued_agreement_uuids,
        ),
        log_prefix="ingestion_cleanup_a_ai_repair_poll_asset",
    )
    mark_logical_run_stage_completed(
        db=db,
        job_name="ingestion_cleanup_a",
        stage_name="ingestion_cleanup_a_ai_repair_poll",
    )
    return successful_request_ids


@dg.asset(
    name="05-11_ingestion_cleanup_b_ai_repair_poll_asset",
    ins={"enqueued_agreement_uuids": dg.AssetIn(key=ingestion_cleanup_b_ai_repair_enqueue_asset.key)},
)
def ingestion_cleanup_b_ai_repair_poll_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    enqueued_agreement_uuids: List[str],
) -> List[str]:
    should_skip, current_stage = should_skip_managed_stage(
        db=db,
        job_name="ingestion_cleanup_b",
        stage_name="ingestion_cleanup_b_ai_repair_poll",
    )
    if should_skip:
        context.log.info(
            "ingestion_cleanup_b_ai_repair_poll_asset: skipping because logical run already reached %s.",
            current_stage,
        )
        return []
    successful_request_ids = _poll_ai_repair_batches(
        context,
        db,
        pipeline_config,
        load_active_scope_for_job(
            context,
            db=db,
            job_name="ingestion_cleanup_b",
            fallback_agreement_uuids=enqueued_agreement_uuids,
        ),
        log_prefix="ingestion_cleanup_b_ai_repair_poll_asset",
    )
    mark_logical_run_stage_completed(
        db=db,
        job_name="ingestion_cleanup_b",
        stage_name="ingestion_cleanup_b_ai_repair_poll",
    )
    return successful_request_ids
