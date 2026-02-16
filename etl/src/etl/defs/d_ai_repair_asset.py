"""
AI Repair assets:
- ai_repair_enqueue_asset: decides full page vs excerpt and enqueues OpenAI Batch jobs
- ai_repair_poll_asset: polls batches, downloads outputs, and persists results

Tables created if missing (MariaDB):
- pdx.ai_repair_batches     (batch-level tracking)
- pdx.ai_repair_requests    (one row per JSONL line / custom_id)
- pdx.ai_repair_rulings     (excerpt-mode rulings at page-level coords)
- pdx.ai_repair_full_pages  (full-page tagged_text outputs)
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false

import io
import json
from typing import Any, Dict, List, Tuple, Set, Optional

import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import text, bindparam
from sqlalchemy.engine import Connection
from openai import OpenAI
import os
import time

from etl.defs.resources import DBResource, PipelineConfig, AiRepairMode
from etl.defs.f_xml_asset import (
    XML_REASON_BODY_STARTS_NON_ARTICLE,
    XML_REASON_FIRST_ARTICLE_NOT_ONE,
    XML_REASON_LLM_INVALID,
    XML_REASON_SECTION_ARTICLE_MISMATCH,
    XML_REASON_SECTION_NON_SEQUENTIAL,
    XML_REASON_SECTION_TITLE_INVALID_NUMBERING,
    XML_REASON_XML_PARSE_FAILURE,
    xml_verify_asset,
)
from etl.utils.run_config import is_batched
from etl.utils.post_asset_refresh import run_post_asset_refresh
from etl.domain.d_ai_repair import (
    UncertainSpan,
    RepairDecision,
    decide_repair_windows,
    filter_uncertain_spans,
    build_jsonl_lines_for_page,
)


# If you prefer, you can plumb this via resources; this is fine too:
def _oai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for ai_repair assets.")
    return OpenAI(api_key=api_key)


DDL_CREATE = [
    # batches: one row per batch create
    """
    CREATE TABLE IF NOT EXISTS pdx.ai_repair_batches (
        batch_id          VARCHAR(128) PRIMARY KEY,
        created_at        DATETIME NOT NULL DEFAULT UTC_TIMESTAMP(),
        status            VARCHAR(32) NOT NULL,
        input_file_id     VARCHAR(128) NULL,
        output_file_id    VARCHAR(128) NULL,
        error_file_id     VARCHAR(128) NULL,
        completion_window VARCHAR(16) NOT NULL,
        request_total     INT DEFAULT 0,
        request_failed    INT DEFAULT 0
    )
    """,
    # requests: one row per custom_id (JSONL line). We also store offsets if excerpt mode.
    """
    CREATE TABLE IF NOT EXISTS pdx.ai_repair_requests (
        request_id   VARCHAR(256) PRIMARY KEY,  -- custom_id
        batch_id     VARCHAR(128) NOT NULL,
        page_uuid    CHAR(36) NOT NULL,
        mode         ENUM('full','excerpt') NOT NULL,
        excerpt_start INT NULL,
        excerpt_end   INT NULL,
        created_at   DATETIME NOT NULL DEFAULT UTC_TIMESTAMP(),
        status       VARCHAR(32) NOT NULL,
        token_usage  JSON NULL
    )
    """,
    
    # excerpt rulings: one row per ruled span (page-level coords)
    """
    CREATE TABLE IF NOT EXISTS pdx.ai_repair_rulings (
        request_id   VARCHAR(256) NOT NULL,
        page_uuid    CHAR(36) NOT NULL,
        start_char   INT NOT NULL,
        end_char     INT NOT NULL,
        label        ENUM('article','section','page','none') NOT NULL,
        batch_id     VARCHAR(128) NOT NULL,
        PRIMARY KEY (request_id, start_char, end_char)
    )
    """,
    # full-page tagged outputs
    """
    CREATE TABLE IF NOT EXISTS pdx.ai_repair_full_pages (
        request_id   VARCHAR(256) PRIMARY KEY,
        page_uuid    CHAR(36) NOT NULL,
        tagged_text  LONGTEXT NOT NULL,
        batch_id     VARCHAR(128) NOT NULL
    )
    """,
    # processed spans: tracks which spans have been sent to repair LLM
    """
    CREATE TABLE IF NOT EXISTS pdx.ai_repair_processed_spans (
        page_uuid           CHAR(36) NOT NULL,
        entity              VARCHAR(32) NOT NULL,
        start_char          INT NOT NULL,
        end_char            INT NOT NULL,
        entity_focus       VARCHAR(32) NOT NULL,
        confidence_threshold DECIMAL(5,4) NOT NULL,
        request_id          VARCHAR(256) NOT NULL,
        batch_id            VARCHAR(128) NOT NULL,
        status              VARCHAR(32) NOT NULL,
        created_at          DATETIME NOT NULL DEFAULT UTC_TIMESTAMP(),
        PRIMARY KEY (page_uuid, entity, start_char, end_char, entity_focus, confidence_threshold)
    )
    """,
]

AI_REPAIR_ELIGIBLE_XML_REASON_CODES: Tuple[str, ...] = (
    XML_REASON_XML_PARSE_FAILURE,
    XML_REASON_LLM_INVALID,
    XML_REASON_BODY_STARTS_NON_ARTICLE,
    XML_REASON_FIRST_ARTICLE_NOT_ONE,
    XML_REASON_SECTION_TITLE_INVALID_NUMBERING,
    XML_REASON_SECTION_ARTICLE_MISMATCH,
    XML_REASON_SECTION_NON_SEQUENTIAL,
)


def _ensure_tables(conn: Connection, schema: str) -> None:
    for ddl in DDL_CREATE:
        ddl_with_schema = ddl.replace("pdx.", f"{schema}.")
        _ = conn.execute(text(ddl_with_schema))


def _fetch_candidates(
    conn: Connection,
    schema: str,
    agreement_limit: int,
    entity_focus: str,
    confidence_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Pull pages from agreements whose latest XML is invalid for a repair-eligible reason,
    with uncertain spans that match entity_focus and confidence_threshold and have
    at least one unprocessed span.
    """
    pages_table = f"{schema}.pages"
    tagged_outputs_table = f"{schema}.tagged_outputs"
    ai_repair_processed_spans_table = f"{schema}.ai_repair_processed_spans"
    xml_table = f"{schema}.xml"
    if not AI_REPAIR_ELIGIBLE_XML_REASON_CODES:
        raise ValueError("AI repair eligible XML reason codes must not be empty.")

    # Find agreements that have pages with unprocessed spans matching the criteria
    agreements_q = text(
        f"""
        SELECT
            p.agreement_uuid
        FROM
            {pages_table} p
            JOIN {xml_table} x
                ON x.agreement_uuid = p.agreement_uuid
               AND x.latest = 1
            JOIN {tagged_outputs_table} t USING (page_uuid)
            CROSS JOIN JSON_TABLE(
                t.spans,
                '$[*]' COLUMNS (
                    entity VARCHAR(255) PATH '$.entity',
                    start_char INT PATH '$.start_char',
                    end_char INT PATH '$.end_char',
                    avg_confidence DOUBLE PATH '$.avg_confidence'
                )
            ) AS jt
        WHERE
            x.status = 'invalid'
            AND x.status_reason_code IN :reason_codes
            AND
            CAST(jt.entity AS CHAR) COLLATE utf8mb4_unicode_ci = CAST(:ef AS CHAR) COLLATE utf8mb4_unicode_ci
            AND jt.avg_confidence < :ct
            AND NOT EXISTS (
                SELECT 1
                FROM {ai_repair_processed_spans_table} ps
                WHERE ps.page_uuid = p.page_uuid
                  AND ps.entity = CAST(jt.entity AS CHAR) COLLATE utf8mb4_unicode_ci
                  AND ps.start_char = jt.start_char
                  AND ps.end_char = jt.end_char
                  AND ps.entity_focus = CAST(:ef AS CHAR) COLLATE utf8mb4_unicode_ci
                  AND ps.status IN ('completed', 'queued', 'running')
            )
        GROUP BY
            p.agreement_uuid
        ORDER BY
            COUNT(DISTINCT p.page_uuid) ASC,
            p.agreement_uuid
        LIMIT
            :lim
        """
    ).bindparams(bindparam("reason_codes", expanding=True))
    agreement_uuids = conn.execute(
        agreements_q,
        {
            "lim": agreement_limit,
            "reason_codes": list(AI_REPAIR_ELIGIBLE_XML_REASON_CODES),
            "ef": entity_focus,
            "ct": confidence_threshold,
        },
    ).scalars().all()
    if not agreement_uuids:
        return []

    # Fetch all pages from those agreements that have matching spans
    pages_q = text(
        f"""
        SELECT DISTINCT
            p.page_uuid,
            p.agreement_uuid,
            p.processed_page_content AS text,
            t.spans AS spans
        FROM
            {pages_table} p
            JOIN {tagged_outputs_table} t USING (page_uuid)
            CROSS JOIN JSON_TABLE(
                t.spans,
                '$[*]' COLUMNS (
                    entity VARCHAR(255) PATH '$.entity',
                    start_char INT PATH '$.start_char',
                    end_char INT PATH '$.end_char',
                    avg_confidence DOUBLE PATH '$.avg_confidence'
                )
            ) AS jt
        WHERE
            p.agreement_uuid IN :uuids
            AND CAST(jt.entity AS CHAR) COLLATE utf8mb4_unicode_ci = CAST(:ef AS CHAR) COLLATE utf8mb4_unicode_ci
            AND jt.avg_confidence < :ct
            AND NOT EXISTS (
                SELECT 1
                FROM {ai_repair_processed_spans_table} ps
                WHERE ps.page_uuid = p.page_uuid
                  AND ps.entity = CAST(jt.entity AS CHAR) COLLATE utf8mb4_unicode_ci
                  AND ps.start_char = jt.start_char
                  AND ps.end_char = jt.end_char
                  AND ps.entity_focus = CAST(:ef AS CHAR) COLLATE utf8mb4_unicode_ci
                  AND ps.status IN ('completed', 'queued', 'running')
            )
        ORDER BY
            p.agreement_uuid,
            p.page_order,
            p.page_uuid
        """
    )
    rows = conn.execute(
        pages_q,
        {
            "uuids": tuple(agreement_uuids),
            "ef": entity_focus,
            "ct": confidence_threshold,
        },
    ).mappings().fetchall()
    return [dict(r) for r in rows]


def _fetch_processed_spans_batch(
    conn: Connection,
    schema: str,
    page_uuids: List[str],
    entity_focus: str,
) -> Dict[str, Set[Tuple[str, int, int]]]:
    """
    Batch-fetch processed spans for multiple pages.
    Returns dict: page_uuid -> set of (entity, start_char, end_char).
    """
    if not page_uuids:
        return {}
    ai_repair_processed_spans_table = f"{schema}.ai_repair_processed_spans"
    q = text(
        f"""
        SELECT page_uuid, entity, start_char, end_char
        FROM {ai_repair_processed_spans_table}
        WHERE page_uuid IN :pids
          AND entity_focus = :ef
          AND status IN ('completed', 'queued', 'running')
        """
    ).bindparams(bindparam("pids", expanding=True))
    rows = conn.execute(q, {"pids": page_uuids, "ef": entity_focus}).mappings().fetchall()
    out: Dict[str, Set[Tuple[str, int, int]]] = {}
    for r in rows:
        pid = r["page_uuid"]
        key = (r["entity"], r["start_char"], r["end_char"])
        out.setdefault(pid, set()).add(key)
    return out


def _filter_already_processed_spans(
    spans: List[UncertainSpan],
    processed_set: Set[Tuple[str, int, int]],
) -> List[UncertainSpan]:
    """
    Filter out spans that are in processed_set (completed, queued, or running).
    """
    if not spans:
        return []
    return [
        span
        for span in spans
        if (span.entity, span.start_char, span.end_char) not in processed_set
    ]


def _parse_uncertain_spans(spans_json: str) -> List[UncertainSpan]:
    spans_raw = json.loads(spans_json)
    if not isinstance(spans_raw, list):
        raise ValueError("Spans JSON must decode to a list.")

    spans: List[UncertainSpan] = []
    for s in spans_raw:
        if not isinstance(s, dict):
            raise ValueError("Span entries must be objects.")
        entity = s["entity"]
        start_char = s["start_char"]
        end_char = s["end_char"]
        avg_confidence = s["avg_confidence"]

        if not isinstance(entity, str):
            raise ValueError("Span entity must be a string.")
        if not isinstance(start_char, int) or not isinstance(end_char, int):
            raise ValueError("Span start/end must be integers.")
        if end_char < start_char:
            raise ValueError("Span end_char must be >= start_char.")
        if not isinstance(avg_confidence, (int, float)):
            raise ValueError("Span avg_confidence must be numeric.")

        spans.append(
            UncertainSpan(
                entity=entity,
                start_char=start_char,
                end_char=end_char,
                avg_confidence=float(avg_confidence),
            )
        )
    return spans


def _insert_batch_row(
    conn: Connection,
    schema: str,
    batch: Any,
    completion_window: str,
    request_total: int,
) -> None:
    ai_repair_batches_table = f"{schema}.ai_repair_batches"
    q = text(
        f"""
        INSERT INTO {ai_repair_batches_table}
            (batch_id, created_at, status, input_file_id, output_file_id, error_file_id,
             completion_window, request_total, request_failed)
        VALUES
            (:batch_id, UTC_TIMESTAMP(), :status, :input_file_id, :output_file_id, :error_file_id,
             :cw, :rt, 0)
        ON DUPLICATE KEY UPDATE
            status = VALUES(status),
            input_file_id = VALUES(input_file_id),
            output_file_id = VALUES(output_file_id),
            error_file_id  = VALUES(error_file_id),
            completion_window = VALUES(completion_window),
            request_total  = VALUES(request_total)
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


def _insert_processed_spans(
    conn: Connection,
    schema: str,
    batch_id: str,
    request_id: str,
    page_uuid: str,
    spans: List[UncertainSpan],
    entity_focus: str,
    confidence_threshold: float,
) -> None:
    """
    Record which spans are being processed in this request.
    """
    if not spans:
        return

    ai_repair_processed_spans_table = f"{schema}.ai_repair_processed_spans"
    q = text(
        f"""
        INSERT INTO {ai_repair_processed_spans_table}
            (page_uuid, entity, start_char, end_char, entity_focus, confidence_threshold,
             request_id, batch_id, status, created_at)
        VALUES
            (:pid, :entity, :start, :end, :ef, :ct, :rid, :bid, 'queued', UTC_TIMESTAMP())
        ON DUPLICATE KEY UPDATE
            request_id = VALUES(request_id),
            batch_id = VALUES(batch_id),
            status = CASE
                WHEN status IN ('queued', 'running') THEN status
                ELSE VALUES(status)
            END
        """
    )
    for span in spans:
        _ = conn.execute(
            q,
            {
                "pid": page_uuid,
                "entity": span.entity,
                "start": span.start_char,
                "end": span.end_char,
                "ef": entity_focus,
                "ct": confidence_threshold,
                "rid": request_id,
                "bid": batch_id,
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


@dg.asset(deps=[xml_verify_asset], name="5-1_ai_repair_enqueue_asset")
def ai_repair_enqueue_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> None:
    """
    Decide full-page vs. surgical excerpt using uncertainty spans, then enqueue OpenAI Batch job(s).

    Strategy:
      - If uncertainty coverage is diffuse (wide breadth / many clusters), send FULL page.
      - Else, send merged excerpt windows with before/after context.
      - Requests are batched into a single .jsonl, uploaded, and created as one Batch job.

    Tuning knobs in domain decide_repair_windows().
    """
    engine = db.get_engine()
    client = _oai_client()

    batch_completion_window = "24h"
    full_page_model = "gpt-5"
    excerpt_model = "gpt-5-mini"

    with engine.begin() as conn:
        _ensure_tables(conn, db.database)

        # 1) fetch candidate pages needing AI repair
        batch_size = pipeline_config.ai_repair_agreement_batch_size
        ai_repair_mode = pipeline_config.ai_repair_mode
        entity_focus = pipeline_config.ai_repair_entity_focus.value
        confidence_threshold = pipeline_config.ai_repair_confidence_threshold
        batched = is_batched(context, pipeline_config)
        send_full_pages = ai_repair_mode == AiRepairMode.ALL

        if not batched:
            context.log.warning("ai_repair_enqueue_asset runs only in batched mode; skipping.")
            run_post_asset_refresh(context, db, pipeline_config, conn=conn)
            return

        candidates = _fetch_candidates(
            conn,
            db.database,
            agreement_limit=batch_size,
            entity_focus=entity_focus,
            confidence_threshold=confidence_threshold,
        )
        if not candidates:
            context.log.info("ai_repair_enqueue_asset: no candidates.")
            run_post_asset_refresh(context, db, pipeline_config, conn=conn)
            return

        # 2) batch-fetch processed spans for all candidates (one query instead of per-page)
        candidate_page_uuids = [r["page_uuid"] for r in candidates]
        processed_by_page = _fetch_processed_spans_batch(
            conn, db.database, candidate_page_uuids, entity_focus
        )

        # 3) build JSONL in-memory, split by mode to keep batch models consistent
        jsonl_full_buf = io.StringIO()
        jsonl_excerpt_buf = io.StringIO()
        lines_meta_full: List[Dict[str, Any]] = []
        lines_meta_excerpt: List[Dict[str, Any]] = []
        # Track which spans are included in each request: request_id -> List[UncertainSpan]
        spans_by_request: Dict[str, List[UncertainSpan]] = {}
        deferred_full_pages = 0

        for row in candidates:
            page_uuid = row["page_uuid"]
            text = row["text"]
            spans = _parse_uncertain_spans(row["spans"])
            focused_spans = filter_uncertain_spans(
                spans,
                entity_focus=entity_focus,
                confidence_threshold=confidence_threshold,
            )
            if not focused_spans:
                continue

            # Filter out spans that have already been processed with these parameters
            processed_set = processed_by_page.get(page_uuid, set())
            unprocessed_spans = _filter_already_processed_spans(
                focused_spans,
                processed_set,
            )
            if not unprocessed_spans:
                continue

            decision: RepairDecision = decide_repair_windows(
                text=text,
                uncertain_spans=unprocessed_spans,
            )

            if decision.mode == "full":
                if not send_full_pages:
                    deferred_full_pages += 1
                    continue
                batch_lines, metas = build_jsonl_lines_for_page(
                    page_uuid=page_uuid,
                    text=text,
                    decision=decision,
                    model=full_page_model,
                    uncertain_spans=unprocessed_spans,
                )
                for line in batch_lines:
                    _ = jsonl_full_buf.write(json.dumps(line, ensure_ascii=False) + "\n")
                lines_meta_full.extend(metas)
                # For full-page mode, all unprocessed_spans are included
                for meta in metas:
                    spans_by_request[meta["request_id"]] = unprocessed_spans
            elif decision.mode == "excerpt":
                batch_lines, metas = build_jsonl_lines_for_page(
                    page_uuid=page_uuid,
                    text=text,
                    decision=decision,
                    model=excerpt_model,
                    uncertain_spans=unprocessed_spans,
                )
                for line in batch_lines:
                    _ = jsonl_excerpt_buf.write(
                        json.dumps(line, ensure_ascii=False) + "\n"
                    )
                lines_meta_excerpt.extend(metas)
                # For excerpt mode, track which spans intersect with each window
                for meta in metas:
                    cs = meta["excerpt_start"]
                    ce = meta["excerpt_end"]
                    # Find spans that intersect with this excerpt window
                    intersecting_spans = [
                        span
                        for span in unprocessed_spans
                        if span.start_char < ce and span.end_char > cs
                    ]
                    spans_by_request[meta["request_id"]] = intersecting_spans
            else:
                raise ValueError(f"Unexpected repair decision mode: {decision.mode!r}")

        if not lines_meta_full and not lines_meta_excerpt:
            if deferred_full_pages > 0 and not send_full_pages:
                context.log.info(
                    f"ai_repair_enqueue_asset: deferred {deferred_full_pages} full-page candidates in EXCERPT mode."
                )
            else:
                context.log.info("ai_repair_enqueue_asset: nothing to enqueue.")
            run_post_asset_refresh(context, db, pipeline_config, conn=conn)
            return

        def _enqueue_batch(jsonl_buf: io.StringIO, lines_meta: List[Dict[str, Any]], label: str) -> None:
            if not lines_meta:
                return
            jsonl_bytes = io.BytesIO(jsonl_buf.getvalue().encode("utf-8"))
            jsonl_bytes.name = f"ai_repair_requests_{label}.jsonl"

            in_file = client.files.create(purpose="batch", file=jsonl_bytes)
            batch = client.batches.create(
                input_file_id=in_file.id,
                endpoint="/v1/responses",
                completion_window=batch_completion_window,
            )

            request_total = len(lines_meta)
            _insert_batch_row(
                conn, db.database, batch, batch_completion_window, request_total
            )
            _insert_requests(conn, db.database, batch.id, lines_meta)
            
            # Record which spans are being processed in each request
            for meta in lines_meta:
                request_id = meta["request_id"]
                page_uuid = meta["page_uuid"]
                if request_id not in spans_by_request:
                    raise ValueError(
                        f"Missing span tracking for request {request_id}, page {page_uuid}. "
                        + "This indicates a bug in the span tracking logic."
                    )
                spans_for_request = spans_by_request[request_id]
                _insert_processed_spans(
                    conn,
                    db.database,
                    batch.id,
                    request_id,
                    page_uuid,
                    spans_for_request,
                    entity_focus,
                    confidence_threshold,
                )

            context.log.info(
                f"Enqueued OpenAI Batch {batch.id} ({label}) with {request_total} requests; input_file_id={in_file.id}"
            )

        # 3) upload JSONL + create Batch per mode
        _enqueue_batch(jsonl_full_buf, lines_meta_full, "full")
        _enqueue_batch(jsonl_excerpt_buf, lines_meta_excerpt, "excerpt")

        if deferred_full_pages > 0 and not send_full_pages:
            context.log.info(
                f"ai_repair_enqueue_asset: deferred {deferred_full_pages} full-page candidates in EXCERPT mode."
            )

    run_post_asset_refresh(context, db, pipeline_config)


def _read_file_text(client: OpenAI, file_id: str) -> str:
    """Minimal reader: rely on SDK's .text or .content→utf-8 bytes."""
    resp = client.files.content(file_id)
    if hasattr(resp, "text"):
        t = resp.text
        text_val = t() if callable(t) else t
        if not isinstance(text_val, str):
            raise ValueError("Expected text response from OpenAI file content.")
        return text_val
    return resp.content.decode("utf-8")


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


def _extract_message_text(body: Dict[str, Any]) -> str:
    """Pull the assistant message first text block from body.output."""
    output = body["output"]
    if not isinstance(output, list):
        raise ValueError(f"Expected body.output to be a list, got {type(output).__name__}")
    msg_blocks = [o for o in output if o.get("type") == "message"]
    if not msg_blocks:
        raise ValueError("No assistant message block in output.")
    contents = msg_blocks[0]["content"]
    if not isinstance(contents, list):
        raise ValueError(f"Expected message content to be a list, got {type(contents).__name__}")
    text_items = [c for c in contents if isinstance(c, dict) and "text" in c]
    if not text_items:
        raise ValueError("Assistant message has no text content.")
    raw_text = text_items[0]["text"]
    if not isinstance(raw_text, str):
        raise ValueError(f"Expected text to be a string, got {type(raw_text).__name__}")
    return raw_text


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


def _parse_full_page_tagged_text(raw: Dict[str, Any]) -> Tuple[str, str]:
    """Return (request_id, tagged_text) strictly or raise."""
    rid = raw["custom_id"]
    resp = raw["response"]
    sc = resp["status_code"]
    if sc not in (200, 201, 202):
        raise ValueError(f"Non-success status_code: {sc}")
    body = resp["body"]
    raw_text = _extract_message_text(body)
    obj = json.loads(raw_text)
    if not isinstance(obj, dict) or "tagged_text" not in obj or "warnings" not in obj:
        raise ValueError("Missing 'tagged_text' or 'warnings' in full-page output.")
    return rid, obj["tagged_text"]


def _parse_excerpt_rulings(raw: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """Return (request_id, rulings) strictly or raise."""
    rid = raw["custom_id"]
    resp = raw["response"]
    sc = resp["status_code"]
    if sc not in (200, 201, 202):
        raise ValueError(f"Non-success status_code: {sc}")
    body = resp["body"]
    raw_text = _extract_message_text(body)
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


@dg.asset(deps=[ai_repair_enqueue_asset], name="5-2_ai_repair_poll_asset")
def ai_repair_poll_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> None:
    """
    Poll terminal batches, read output/error JSONL, persist parsed entities strictly.

    Status handling:
      - Parsed OK → persisted via _persist_results (assumed to set success status)
      - HTTP success but parse failed → status = 'parse_error'
      - Error-file entries → status = 'failed'
      - No output/no error → status = 'completed_no_output'
    """
    engine = db.get_engine()
    schema = db.database
    ai_repair_batches_table = f"{schema}.ai_repair_batches"
    ai_repair_requests_table = f"{schema}.ai_repair_requests"
    ai_repair_processed_spans_table = f"{schema}.ai_repair_processed_spans"
    ai_repair_full_pages_table = f"{schema}.ai_repair_full_pages"
    ai_repair_rulings_table = f"{schema}.ai_repair_rulings"
    client = _oai_client()

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
                        SELECT batch_id
                        FROM {ai_repair_batches_table}
                        WHERE status NOT IN ('completed','failed','cancelled','expired')
                        ORDER BY created_at ASC
                        LIMIT 20
                        """
                    )
                )
                .mappings()
                .fetchall()
            )
            if not rows:
                context.log.info("ai_repair_poll_asset: no batches to poll.")
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

                success_ids: Set[str] = set()
                http_success_ids: Set[str] = set()
                failed_ids: Set[str] = set()
                parse_error_ids: Set[str] = set()
                usage_by_request: Dict[str, Dict[str, int]] = {}

                # Parsed excerpt rulings and full pages
                parsed_rulings: List[Tuple[str, str, List[Dict[str, Any]]]] = []  # (rid, page_uuid, rulings)
                parsed_full_pages: List[Tuple[str, str, str]] = []  # (rid, page_uuid, tagged_text)

                # Process output JSONL (success lines)
                ofid = getattr(b, "output_file_id", None)
                if ofid:
                    out_text = _read_file_text(client, ofid).strip()
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
                                    rid2, tagged_text = _parse_full_page_tagged_text(raw)
                                    parsed_full_pages.append((rid2, pid, tagged_text))
                                    success_ids.add(rid2)
                                elif mode == "excerpt":
                                    rid2, rulings = _parse_excerpt_rulings(raw)
                                    parsed_rulings.append((rid2, pid, rulings))
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
                    err_text = _read_file_text(client, efid).strip()
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

                parsed_ids = set(rid for rid, _, _ in parsed_full_pages) | set(rid for rid, _, _ in parsed_rulings)

                # Mark requests completed that produced outputs (either kind)
                _mark_completed(conn, db.database, parsed_ids)

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

                # Leftovers: neither output nor error line → completed_no_output
                leftover_ids = req_ids_all - http_success_ids - failed_ids - parse_error_ids
                _bulk_update_status(conn, db.database, leftover_ids, "completed_no_output")

                # Summary
                context.log.info(
                    f"Batch {bid}: success={len(success_ids)} failed={len(failed_ids)} parse_error={len(parse_error_ids)} no_output={len(no_output_ids)}"
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

    run_post_asset_refresh(context, db, pipeline_config)
