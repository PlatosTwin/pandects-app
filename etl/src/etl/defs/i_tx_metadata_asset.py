"""Batch transaction metadata enrichment via OpenAI web search.

This asset selects earliest agreements missing metadata, submits one OpenAI Batch
to the Responses API with the web_search tool, waits for completion, parses
outputs, and updates `pdx.agreements` accordingly, setting metadata=1 for
successfully parsed rows (even if values are unknown/null). Rows with OpenAI
errors are left untouched.

Runs in batched mode only; in non-batched mode it logs a warning and exits.
Default batch size is 50 agreements (overridable via run tag `agreement_batch_size`).
"""

from __future__ import annotations

import io
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import dagster as dg
from sqlalchemy import text
from openai import OpenAI
from dotenv import load_dotenv

from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.i_tx_metadata import (
    build_jsonl_lines_for_agreements,
    parse_metadata_line,
)


load_dotenv()


def _oai_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _read_file_text(client: OpenAI, file_id: str) -> str:
    resp = client.files.content(file_id)
    if hasattr(resp, "text"):
        t = resp.text
        return t() if callable(t) else (t if isinstance(t, str) else str(t))
    return resp.content.decode("utf-8")


def _map_consideration(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    if val == "all_cash":
        return "cash"
    if val == "all_stock":
        return "stock"
    if val == "mixed":
        return "mixed"
    return None  # unknown → NULL


def _map_public_flag_to_type(flag: Optional[bool]) -> Optional[str]:
    if flag is None:
        return None
    return "public" if bool(flag) else "private"


@dg.asset(deps=[], name="9_tx_metadata_asset")
def tx_metadata_asset(
    context,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> None:
    # Enforce batched-only mode
    run_scope_tag = context.run.tags.get("run_scope")
    is_batched: bool = (
        run_scope_tag == "batched"
        if run_scope_tag is not None
        else pipeline_config.is_batched()
    )
    if not is_batched:
        context.log.warning("tx_metadata_asset runs only in batched mode; skipping.")
        return

    # Batch size with default 50
    bs_tag = context.run.tags.get("agreement_batch_size")
    try:
        batch_size = int(bs_tag) if bs_tag else 50
    except Exception:
        batch_size = 50

    engine = db.get_engine()

    # Select earliest agreements without metadata
    select_q = text(
        """
        SELECT agreement_uuid, target, acquirer, filing_date
        FROM pdx.agreements
        WHERE COALESCE(metadata, 0) = 0
        ORDER BY (filing_date IS NULL) ASC, filing_date ASC, agreement_uuid ASC
        LIMIT :lim
        """
    )

    with engine.begin() as conn:
        rows = conn.execute(select_q, {"lim": batch_size}).mappings().fetchall()
        agreements = [dict(r) for r in rows]

    if not agreements:
        context.log.info("tx_metadata_asset: no agreements need metadata.")
        return

    context.log.info(f"tx_metadata_asset: selected {len(agreements)} agreements for enrichment")

    # Build JSONL
    model_name = "gpt-5-mini"
    lines = build_jsonl_lines_for_agreements(agreements, model=model_name)
    buf = io.StringIO()
    for line in lines:
        buf.write(json.dumps(line, ensure_ascii=False) + "\n")

    client = _oai_client()
    jsonl_bytes = io.BytesIO(buf.getvalue().encode("utf-8"))
    jsonl_bytes.name = "tx_metadata_requests.jsonl"

    in_file = client.files.create(purpose="batch", file=jsonl_bytes)
    batch = client.batches.create(
        input_file_id=in_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )

    context.log.info(f"tx_metadata_asset: created batch {batch.id} with {len(lines)} requests; input_file_id={in_file.id}")

    # Poll until terminal
    while True:
        b = client.batches.retrieve(batch.id)
        rc = getattr(b, "request_counts", None)
        total = (getattr(rc, "total", 0) or 0) if rc is not None else 0
        failed = (getattr(rc, "failed", 0) or 0) if rc is not None else 0
        completed = getattr(rc, "completed", None) if rc is not None else None
        if completed is None and rc is not None:
            completed = getattr(rc, "succeeded", 0) or 0
        done = (completed or 0) + (failed or 0)
        pct = int((done / total) * 100) if total else 0
        context.log.info(
            f"Batch {b.id}: status={b.status}, {(completed or 0)}/{total} completed, failed={failed} ({pct}%)"
        )

        if b.status in ("completed", "failed", "cancelled", "expired"):
            batch = b
            break
        time.sleep(5)

    # Read outputs and errors
    success_data: List[Tuple[str, Dict[str, Any]]] = []  # (agreement_uuid, parsed_obj)
    http_success_ids: set[str] = set()
    error_ids: set[str] = set()

    ofid = getattr(batch, "output_file_id", None)
    if ofid:
        out_text = _read_file_text(client, ofid).strip()
        if out_text:
            for line in out_text.splitlines():
                raw = json.loads(line)
                rid = raw.get("custom_id")
                if not rid:
                    continue
                resp = raw.get("response") or {}
                sc = resp.get("status_code")
                if sc in (200, 201, 202) or ("body" in resp):
                    http_success_ids.add(rid)
                try:
                    agr_id, obj = parse_metadata_line(raw)
                    success_data.append((agr_id, obj))
                except Exception:
                    # parsing failure → not a success; do not mark metadata
                    pass

    efid = getattr(batch, "error_file_id", None)
    if efid:
        err_text = _read_file_text(client, efid).strip()
        if err_text:
            for line in err_text.splitlines():
                err = json.loads(line)
                rid = err.get("custom_id")
                if rid:
                    error_ids.add(rid)

    # Persist: only for parsed successes
    update_q = text(
        """
        UPDATE pdx.agreements
        SET 
            transaction_consideration = :consideration,
            transaction_price_cash = :price_cash,
            transaction_price_stock = :price_stock,
            transaction_price_total = :price_total,
            target_type = :target_type,
            acquirer_type = :acquirer_type,
            target_pe = :target_pe,
            acquirer_pe = :acquirer_pe,
            sources = :sources,
            metadata = 1
        WHERE agreement_uuid = :uuid
        """
    )

    updated = 0
    skipped_due_to_error = 0
    param_rows: List[Dict[str, Any]] = []
    for uuid, obj in success_data:
        if uuid in error_ids:
            skipped_due_to_error += 1
            continue

        consideration = _map_consideration(obj.get("consideration_type"))
        price = obj.get("purchase_price") or {}
        cash = price.get("cash") if isinstance(price, dict) else None
        stock = price.get("stock") if isinstance(price, dict) else None
        price_cash = None if cash is None else float(cash)
        price_stock = None if stock is None else float(stock)
        price_total = None
        if price_cash is not None or price_stock is not None:
            price_total = (price_cash or 0.0) + (price_stock or 0.0)

        target_type = _map_public_flag_to_type(obj.get("target_public"))
        acquirer_type = _map_public_flag_to_type(obj.get("acquirer_public"))
        target_pe = obj.get("target_pe")
        acquirer_pe = obj.get("acquirer_pe")
        sources = obj.get("sources")

        param_rows.append(
            {
                "consideration": consideration,
                "price_cash": price_cash,
                "price_stock": price_stock,
                "price_total": price_total,
                "target_type": target_type,
                "acquirer_type": acquirer_type,
                "target_pe": target_pe,
                "acquirer_pe": acquirer_pe,
                "sources": sources,
                "uuid": uuid,
            }
        )

    if param_rows:
        with engine.begin() as conn:
            for params in param_rows:
                conn.execute(update_q, params)
                updated += 1

    attempted = len(lines)
    parsed_ok = len(success_data)
    http_success = len(http_success_ids)
    context.log.info(
        f"tx_metadata_asset: attempted={attempted}, http_success={http_success}, parsed={parsed_ok}, updated={updated}, errors={len(error_ids)}, skipped_due_to_error={skipped_due_to_error}"
    )


