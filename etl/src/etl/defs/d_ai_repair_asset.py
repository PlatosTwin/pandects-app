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

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Set, Optional

import dagster as dg
from sqlalchemy import text, bindparam
from sqlalchemy.engine import Connection
from openai import OpenAI
import os
import time

from etl.defs.resources import DBResource, PipelineConfig
from etl.utils.run_config import is_batched
from etl.domain.d_ai_repair import (
    UncertainSpan,
    RepairDecision,
    decide_repair_windows,
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
        created_at        DATETIME NOT NULL,
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
        created_at   DATETIME NOT NULL,
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
]


def _ensure_tables(conn: Connection) -> None:
    for ddl in DDL_CREATE:
        _ = conn.execute(text(ddl))


def _fetch_candidates(conn: Connection, agreement_limit: int) -> List[Dict[str, Any]]:
    """
    Pull pages with uncertain spans for the first N agreements that need AI repair.
    """
    agreements_q = text(
        """
        SELECT
            p.agreement_uuid
        FROM
            pdx.pages p
            JOIN pdx.tagged_outputs t USING (page_uuid)
        WHERE
            JSON_LENGTH(t.spans) > 0
            AND NOT EXISTS (
                SELECT
                    1
                FROM
                    pdx.ai_repair_requests r
                WHERE
                    r.page_uuid = p.page_uuid
                    and status not in ('completed', 'queued')
            )
        GROUP BY
            p.agreement_uuid
        ORDER BY
            p.agreement_uuid
        LIMIT
            :lim
        """
    )
    agreement_uuids = conn.execute(
        agreements_q, {"lim": agreement_limit}
    ).scalars().all()
    if not agreement_uuids:
        return []

    pages_q = text(
        """
        SELECT
            p.page_uuid,
            p.agreement_uuid,
            p.processed_page_content AS text,
            t.spans AS spans
        FROM
            pdx.pages p
            JOIN pdx.tagged_outputs t USING (page_uuid)
        WHERE
            p.agreement_uuid IN :uuids
            AND JSON_LENGTH(t.spans) > 0
            AND NOT EXISTS (
                SELECT
                    1
                FROM
                    pdx.ai_repair_requests r
                WHERE
                    r.page_uuid = p.page_uuid
                    and status not in ('completed', 'queued')
            )
        ORDER BY
            p.agreement_uuid,
            p.page_order,
            p.page_uuid
        """
    )
    rows = conn.execute(pages_q, {"uuids": tuple(agreement_uuids)}).mappings().fetchall()
    return [dict(r) for r in rows]


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
    conn: Connection, batch: Any, completion_window: str, request_total: int
) -> None:
    q = text(
        """
        INSERT INTO pdx.ai_repair_batches
            (batch_id, created_at, status, input_file_id, output_file_id, error_file_id,
             completion_window, request_total, request_failed)
        VALUES
            (:batch_id, :created_at, :status, :input_file_id, :output_file_id, :error_file_id,
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
            "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
            "status": batch.status,
            "input_file_id": getattr(batch, "input_file_id", None),
            "output_file_id": getattr(batch, "output_file_id", None),
            "error_file_id": getattr(batch, "error_file_id", None),
            "cw": completion_window,
            "rt": request_total,
        },
    )


def _insert_requests(
    conn: Connection, batch_id: str, lines_meta: List[Dict[str, Any]]
) -> None:
    """
    lines_meta: emitted by build_jsonl_lines_for_page(), one dict per custom_id:
        {request_id, page_uuid, mode, excerpt_start, excerpt_end}
    """
    q = text(
        """
        INSERT INTO pdx.ai_repair_requests
            (request_id, batch_id, page_uuid, mode, excerpt_start, excerpt_end, created_at, status)
        VALUES
            (:rid, :bid, :pid, :mode, :xs, :xe, :ts, 'queued')
        ON DUPLICATE KEY UPDATE
            batch_id = VALUES(batch_id),
            status   = 'queued'
        """
    )
    now = datetime.now(timezone.utc).replace(tzinfo=None)
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
                "ts": now,
            },
        )


def _mark_completed(conn: Connection, request_ids: Set[str]) -> None:
    if not request_ids:
        return
    q = text("UPDATE pdx.ai_repair_requests SET status = 'completed' WHERE request_id IN :ids").bindparams(
        bindparam("ids", expanding=True)
    )
    _ = conn.execute(q, {"ids": list(request_ids)})


@dg.asset(deps=[], name="4-1_ai_repair_enqueue_asset")
def ai_repair_enqueue_asset(
    context: dg.AssetExecutionContext,
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
        _ensure_tables(conn)

        # 1) fetch candidate pages needing AI repair
        batch_size = pipeline_config.ai_repair_agreement_batch_size
        batched = is_batched(context, pipeline_config)

        if not batched:
            context.log.warning("ai_repair_enqueue_asset runs only in batched mode; skipping.")
            return

        candidates = _fetch_candidates(conn, agreement_limit=batch_size)
        if not candidates:
            context.log.info("ai_repair_enqueue_asset: no candidates.")
            return

        # 2) build JSONL in-memory, split by mode to keep batch models consistent
        jsonl_full_buf = io.StringIO()
        jsonl_excerpt_buf = io.StringIO()
        lines_meta_full: List[Dict[str, Any]] = []
        lines_meta_excerpt: List[Dict[str, Any]] = []

        for row in candidates:
            page_uuid = row["page_uuid"]
            text = row["text"]
            spans = _parse_uncertain_spans(row["spans"])

            decision: RepairDecision = decide_repair_windows(
                text=text,
                uncertain_spans=spans,
            )

            if decision.mode == "full":
                batch_lines, metas = build_jsonl_lines_for_page(
                    page_uuid=page_uuid,
                    text=text,
                    decision=decision,
                    model=full_page_model,
                    uncertain_spans=spans,
                )
                for line in batch_lines:
                    _ = jsonl_full_buf.write(json.dumps(line, ensure_ascii=False) + "\n")
                lines_meta_full.extend(metas)
            elif decision.mode == "excerpt":
                batch_lines, metas = build_jsonl_lines_for_page(
                    page_uuid=page_uuid,
                    text=text,
                    decision=decision,
                    model=excerpt_model,
                    uncertain_spans=spans,
                )
                for line in batch_lines:
                    _ = jsonl_excerpt_buf.write(
                        json.dumps(line, ensure_ascii=False) + "\n"
                    )
                lines_meta_excerpt.extend(metas)
            else:
                raise ValueError(f"Unexpected repair decision mode: {decision.mode!r}")

        if not lines_meta_full and not lines_meta_excerpt:
            context.log.info("ai_repair_enqueue_asset: nothing to enqueue.")
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
            _insert_batch_row(conn, batch, batch_completion_window, request_total)
            _insert_requests(conn, batch.id, lines_meta)

            context.log.info(
                f"Enqueued OpenAI Batch {batch.id} ({label}) with {request_total} requests; input_file_id={in_file.id}"
            )

        # 3) upload JSONL + create Batch per mode
        _enqueue_batch(jsonl_full_buf, lines_meta_full, "full")
        _enqueue_batch(jsonl_excerpt_buf, lines_meta_excerpt, "excerpt")


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


def _bulk_update_status(conn: Connection, request_ids: Set[str], status: str) -> None:
    if not request_ids:
        return
    q = text(
        "UPDATE pdx.ai_repair_requests SET status = :st WHERE request_id IN :ids"
    ).bindparams(bindparam("ids", expanding=True))
    _ = conn.execute(q, {"st": status, "ids": list(request_ids)})


def _extract_message_text(body: Dict[str, Any]) -> str:
    """Pull the assistant message first text block from body.output."""
    output = body["output"]
    msg_blocks = [o for o in output if o.get("type") == "message"]
    if not msg_blocks:
        raise ValueError("No assistant message block in output.")
    contents = msg_blocks[0]["content"]
    text_items = [c for c in contents if isinstance(c, dict) and "text" in c]
    if not text_items:
        raise ValueError("Assistant message has no text content.")
    raw_text = text_items[0]["text"]
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


@dg.asset(deps=[ai_repair_enqueue_asset], name="4-2_ai_repair_poll_asset")
def ai_repair_poll_asset(context: dg.AssetExecutionContext, db: DBResource) -> None:
    """
    Poll terminal batches, read output/error JSONL, persist parsed entities strictly.

    Status handling:
      - Parsed OK → persisted via _persist_results (assumed to set success status)
      - HTTP success but parse failed → status = 'parse_error'
      - Error-file entries → status = 'failed'
      - No output/no error → status = 'completed_no_output'
    """
    engine = db.get_engine()
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
                        """
                        SELECT batch_id
                        FROM pdx.ai_repair_batches
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
                return

            upd_batch = text(
                """
                UPDATE pdx.ai_repair_batches
                SET status=:st, output_file_id=:of, error_file_id=:ef,
                    request_total=:rt, request_failed=:rf
                WHERE batch_id=:bid
                """
            )
            select_req = text(
                "SELECT request_id, page_uuid, mode, excerpt_start FROM pdx.ai_repair_requests WHERE batch_id = :bid"
            )
            mark_running = text(
                "UPDATE pdx.ai_repair_requests SET status='running' WHERE batch_id=:bid AND status='queued'"
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
                            except Exception:
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
                        "UPDATE pdx.ai_repair_requests SET token_usage = :usage WHERE request_id = :rid"
                    )
                    for rid, usage in usage_by_request.items():
                        _ = conn.execute(
                            upd_usage, {"rid": rid, "usage": json.dumps(usage)}
                        )

                # Full-page tagged_text
                if parsed_full_pages:
                    ins_full = text(
                        """
                        INSERT INTO pdx.ai_repair_full_pages (request_id, page_uuid, tagged_text, batch_id)
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
                        """
                        INSERT INTO pdx.ai_repair_rulings (request_id, page_uuid, start_char, end_char, label, batch_id)
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
                _mark_completed(conn, parsed_ids)

                # Completed with HTTP success but no parsed record (and not failed/parse_error)
                no_output_ids = (
                    (http_success_ids - parsed_ids) - failed_ids - parse_error_ids
                )
                _bulk_update_status(conn, no_output_ids, "completed_no_output")

                # Parse errors on HTTP-success lines
                _bulk_update_status(
                    conn, parse_error_ids - failed_ids - parsed_ids, "parse_error"
                )

                # Explicit failures
                _bulk_update_status(conn, failed_ids, "failed")

                # Leftovers: neither output nor error line → completed_no_output
                leftover_ids = req_ids_all - http_success_ids - failed_ids - parse_error_ids
                _bulk_update_status(conn, leftover_ids, "completed_no_output")

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
