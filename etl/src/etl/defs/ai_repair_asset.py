"""
AI Repair assets:
- ai_repair_enqueue_asset: decides full page vs excerpt and enqueues OpenAI Batch jobs
- ai_repair_poll_asset: polls batches, downloads outputs, and persists results

Tables created if missing (MariaDB):
- pdx.ai_repair_batches   (batch-level tracking)
- pdx.ai_repair_requests  (one row per JSONL line / custom_id)
- pdx.ai_repair_results   (final structured entities from model)
"""

from __future__ import annotations

import io
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple, Optional, Set

import dagster as dg
from sqlalchemy import text, bindparam
from openai import OpenAI
from dotenv import load_dotenv
import os

from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.ai_repair import (
    UncertainSpan,
    RepairDecision,
    decide_repair_windows,
    build_jsonl_lines_for_page,
    parse_batch_output_line,
)

load_dotenv()


# If you prefer, you can plumb this via resources; this is fine too:
def _oai_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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
        status       VARCHAR(32) NOT NULL
    )
    """,
    # results: entities produced by AI
    """
    CREATE TABLE IF NOT EXISTS pdx.ai_repair_results (
        request_id   VARCHAR(256) NOT NULL,
        page_uuid    CHAR(36) NOT NULL,
        label        ENUM('article','section','page') NOT NULL,
        start_char   INT NOT NULL,
        end_char     INT NOT NULL,
        title        TEXT NULL,
        batch_id     VARCHAR(128) NOT NULL,
        PRIMARY KEY (request_id, start_char, end_char)
    )
    """,
]


def _ensure_tables(conn) -> None:
    for ddl in DDL_CREATE:
        conn.execute(text(ddl))


def _fetch_candidates(conn, batch_limit: int) -> List[Dict[str, Any]]:
    """
    Pull pages that have uncertain spans and haven't yet been queued for AI repair.
    Adapt the WHERE clause to your schema—assumes you persisted `spans` JSON in pdx.tagged_outputs.
    """
    # NOTE: You may want to restrict to body pages or add thresholds in SQL.
    q = text(
        """
        SELECT
            p.page_uuid,
            p.processed_page_content AS text,
            t.spans AS spans  -- expect a JSON array of spans
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
                    and status = 'completed'
            )
        ORDER BY
            p.page_uuid
        LIMIT
            :lim
        """
    )
    rows = conn.execute(q, {"lim": batch_limit}).mappings().fetchall()
    return [dict(r) for r in rows]


def _insert_batch_row(conn, batch, completion_window: str, request_total: int) -> None:
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
    conn.execute(
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


def _insert_requests(conn, batch_id: str, lines_meta: List[Dict[str, Any]]) -> None:
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
        conn.execute(
            q,
            {
                "rid": m["request_id"],
                "bid": batch_id,
                "pid": m["page_uuid"],
                "mode": m["mode"],
                "xs": m.get("excerpt_start"),
                "xe": m.get("excerpt_end"),
                "ts": now,
            },
        )


def _persist_results(conn, batch_id: str, parsed: List[Dict[str, Any]]) -> None:
    """
    parsed: list of dicts from parse_batch_output_line(), each with fields:
      request_id, page_uuid, entities=[{label,start_char,end_char,title?}, ...]
    """
    if not parsed:
        return

    # Pull mode & excerpt_start for this batch once (avoid N queries)
    req_info = {
        row.request_id: (row.mode, row.excerpt_start)
        for row in conn.execute(
            text(
                "SELECT request_id, mode, excerpt_start FROM pdx.ai_repair_requests WHERE batch_id = :bid"
            ),
            {"bid": batch_id},
        ).mappings()
    }

    ins = text(
        """
        INSERT INTO pdx.ai_repair_results
            (request_id, page_uuid, label, start_char, end_char, title, batch_id)
        VALUES
            (:rid, :pid, :label, :s, :e, :title, :bid)
        ON DUPLICATE KEY UPDATE
            title = VALUES(title)
        """
    )
    upd_req = text(
        "UPDATE pdx.ai_repair_requests SET status = :st WHERE request_id = :rid"
    )

    for item in parsed:
        rid = item["request_id"]
        pid = item["page_uuid"]
        mode, xs = req_info.get(rid, (None, None))
        base = int(xs) if (mode == "excerpt" and xs is not None) else 0

        for ent in item["entities"]:
            s_adj = int(ent["start_char"]) + base
            e_adj = int(ent["end_char"]) + base
            conn.execute(
                ins,
                {
                    "rid": rid,
                    "pid": pid,
                    "label": ent["label"],
                    "s": s_adj,
                    "e": e_adj,
                    "title": ent.get("title"),
                    "bid": batch_id,
                },
            )
        # Mark request completed even when 0 entities (valid empty output)
        conn.execute(upd_req, {"st": "completed", "rid": rid})


@dg.asset(deps=[], name="ai_repair_enqueue_asset")
def ai_repair_enqueue_asset(
    context,
    db: DBResource,
    pipeline_config: PipelineConfig,  # currently unused; keep for symmetry
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

    batch_completion_window = "24h"  # Batch SLA window; keeps cost down. :contentReference[oaicite:1]{index=1}
    model_name = "gpt-5"  # Cheap + capable for headings extraction. See docs. :contentReference[oaicite:2]{index=2}

    with engine.begin() as conn:
        _ensure_tables(conn)

        # 1) fetch candidate pages needing AI repair
        candidates = _fetch_candidates(conn, batch_limit=50)  # tune as desired
        if not candidates:
            context.log.info("ai_repair_enqueue_asset: no candidates.")
            return

        # 2) build JSONL in-memory
        jsonl_buf = io.StringIO()
        lines_meta: List[Dict[str, Any]] = []
        total_lines = 0

        for row in candidates:
            page_uuid = row["page_uuid"]
            text = row["text"]
            try:
                spans_raw = json.loads(row["spans"])
            except Exception:
                # tolerate already-loaded JSON; or skip bad rows
                spans_raw = row["spans"]

            spans = [
                UncertainSpan(
                    entity=s.get("entity", "o"),
                    start_token=int(s["start_token"]),
                    end_token=int(s["end_token"]),
                    avg_confidence=float(s.get("avg_confidence", 0.0)),
                )
                for s in spans_raw
            ]

            decision: RepairDecision = decide_repair_windows(
                text=text,
                uncertain_spans=spans,
                # You can override thresholds per your corpus by passing kwargs here
            )

            batch_lines, metas = build_jsonl_lines_for_page(
                page_uuid=page_uuid,
                text=text,
                decision=decision,
                model=model_name,
            )
            for line in batch_lines:
                jsonl_buf.write(json.dumps(line, ensure_ascii=False) + "\n")

            lines_meta.extend(metas)
            total_lines += len(batch_lines)

        if total_lines == 0:
            context.log.info("ai_repair_enqueue_asset: nothing to enqueue.")
            return

        # 3) upload JSONL + create Batch
        jsonl_bytes = io.BytesIO(jsonl_buf.getvalue().encode("utf-8"))
        jsonl_bytes.name = "ai_repair_requests.jsonl"

        in_file = client.files.create(purpose="batch", file=jsonl_bytes)
        batch = client.batches.create(
            input_file_id=in_file.id,
            endpoint="/v1/responses",  # matches the body.url in each JSONL line
            completion_window=batch_completion_window,
        )  # :contentReference[oaicite:3]{index=3}

        _insert_batch_row(conn, batch, batch_completion_window, total_lines)
        _insert_requests(conn, batch.id, lines_meta)

        context.log.info(
            f"Enqueued OpenAI Batch {batch.id} with {total_lines} requests; input_file_id={in_file.id}"
        )


ALLOWED_LABELS = {"article", "section", "page"}


def _read_file_text(client, file_id: str) -> str:
    """Minimal reader: rely on SDK's .text or .content→utf-8 bytes."""
    resp = client.files.content(file_id)
    if hasattr(resp, "text"):
        t = resp.text
        return t() if callable(t) else t
    return resp.content.decode("utf-8")


def _bulk_update_status(conn, request_ids: Set[str], status: str) -> None:
    if not request_ids:
        return
    q = text(
        "UPDATE pdx.ai_repair_requests SET status = :st WHERE request_id IN :ids"
    ).bindparams(bindparam("ids", expanding=True))
    conn.execute(q, {"st": status, "ids": list(request_ids)})


def _extract_entities_from_body(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Strictly extract entities from response.body:
      - Find the *assistant* message in body["output"] (type == "message")
      - Take the first content item with a "text" field (the JSON string)
      - json.loads it and return obj["entities"] (must exist and be a list)
    """
    output = body["output"]  # KeyError → parse_error
    # Usually index 1, but we scan for "message" to be explicit
    msg_blocks = [o for o in output if o.get("type") == "message"]
    if not msg_blocks:
        raise ValueError("No assistant message block in output.")
    contents = msg_blocks[0]["content"]  # KeyError → parse_error
    # Pick the first content item that carries the text JSON
    text_items = [c for c in contents if isinstance(c, dict) and "text" in c]
    if not text_items:
        raise ValueError("Assistant message has no text content.")
    raw_text = text_items[0]["text"]
    obj = json.loads(raw_text)  # any JSON error → parse_error
    if not isinstance(obj, dict):
        raise ValueError("Top-level response JSON is not an object.")
    if "entities" not in obj or "warnings" not in obj:
        raise ValueError("Missing required keys 'entities' or 'warnings'.")
    ents = obj["entities"]
    if not isinstance(ents, list):
        raise ValueError("'entities' must be a list.")
    # Strict validation; no coercion
    for e in ents:
        if not isinstance(e, dict):
            raise ValueError("Entity is not an object.")
        label = e.get("label")
        sc = e.get("start_char")
        ec = e.get("end_char")
        title = e.get("title")
        if label not in ALLOWED_LABELS:
            raise ValueError(f"Invalid label: {label!r}")
        if not isinstance(sc, int) or not isinstance(ec, int):
            raise ValueError("start_char/end_char must be integers.")
        if not isinstance(title, str):
            raise ValueError("title must be a string.")
    return ents


def _parse_success_line_strict(raw: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Return (request_id, entities) or raise for any deviation.
    """
    rid = raw["custom_id"]  # KeyError → parse_error
    resp = raw["response"]  # KeyError → parse_error
    sc = resp.get("status_code")
    if sc not in (200, 201, 202):
        raise ValueError(f"Non-success status_code: {sc}")
    body = resp["body"]  # KeyError → parse_error
    entities = _extract_entities_from_body(body)
    return rid, entities


@dg.asset(deps=[ai_repair_enqueue_asset], name="ai_repair_poll_asset")
def ai_repair_poll_asset(context, db: DBResource) -> None:
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
            "SELECT request_id, page_uuid FROM pdx.ai_repair_requests WHERE batch_id = :bid"
        )
        mark_running = text(
            "UPDATE pdx.ai_repair_requests SET status='running' WHERE batch_id=:bid AND status='queued'"
        )

        for r in rows:
            bid = r["batch_id"]
            b = client.batches.retrieve(bid)

            # Update batch row with fresh metadata
            rc = getattr(b, "request_counts", None)
            conn.execute(
                upd_batch,
                {
                    "st": b.status,
                    "of": getattr(b, "output_file_id", None),
                    "ef": getattr(b, "error_file_id", None),
                    "rt": getattr(rc, "total", 0) or 0,
                    "rf": getattr(rc, "failed", 0) or 0,
                    "bid": bid,
                },
            )

            if b.status not in ("completed", "failed", "cancelled", "expired"):
                conn.execute(mark_running, {"bid": bid})
                context.log.info(f"Batch {bid} status={b.status}")
                continue

            # Terminal batch: fetch request metadata
            req_meta = {
                row.request_id: row.page_uuid
                for row in conn.execute(select_req, {"bid": bid})
            }
            req_ids_all = set(req_meta.keys())

            success_ids: Set[str] = set()
            http_success_ids: Set[str] = set()
            failed_ids: Set[str] = set()
            parse_error_ids: Set[str] = set()

            parsed_rows: List[Dict[str, Any]] = []

            # Process output JSONL (success lines)
            ofid = getattr(b, "output_file_id", None)
            if ofid:
                out_text = _read_file_text(client, ofid).strip()
                if out_text:
                    for line in out_text.splitlines():
                        raw = json.loads(line)
                        rid = raw.get("custom_id")
                        if not rid:
                            # Skip malformed line; we don't try to salvage
                            continue
                        # Mark HTTP success if present
                        resp = raw.get("response") or {}
                        sc = resp.get("status_code")
                        if sc in (200, 201, 202) or ("body" in resp):
                            http_success_ids.add(rid)
                        try:
                            rid2, ents = _parse_success_line_strict(raw)
                            page_uuid = req_meta.get(rid2)
                            parsed_rows.append(
                                {
                                    "request_id": rid2,
                                    "page_uuid": page_uuid,
                                    "entities": ents,
                                }
                            )
                            success_ids.add(rid2)
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
                        rid = err.get("custom_id")
                        if rid:
                            failed_ids.add(rid)

            # Persist strictly parsed results (expects your existing implementation)
            _persist_results(conn, bid, parsed_rows)

            parsed_ids = {p["request_id"] for p in parsed_rows}

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
            entity_count = sum(len(r["entities"]) for r in parsed_rows)
            context.log.info(
                f"Batch {bid}: parsed {len(parsed_rows)} requests; "
                f"entities={entity_count} success={len(success_ids)} "
                f"failed={len(failed_ids)} parse_error={len(parse_error_ids)} "
                f"no_output={len(no_output_ids)}"
            )
