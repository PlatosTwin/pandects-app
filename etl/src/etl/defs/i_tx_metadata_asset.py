"""Transaction metadata enrichment: offline (document-only) or web-search mode.

Offline mode: selects agreements missing target, acquirer, or deal_type; sends
front_matter + first two body pages to gpt-5-mini via OpenAI Batch API; updates
target, acquirer, deal_type only.

Web-search mode: selects agreements with metadata=0 and target/acquirer set;
uses Responses API with web_search; updates web-sourced fields only (not target,
acquirer, deal_type) and sets metadata=1.

Runs in batched mode only. Mode and batch size via PipelineConfig.
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false

import io
import json
import os
import time
from typing import Any, Dict, List, Tuple

import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import bindparam, text
from openai import OpenAI

from etl.defs.resources import DBResource, PipelineConfig, TxMetadataMode
from etl.domain.i_tx_metadata import (
    build_offline_tx_metadata_request_body,
    build_offline_update_params,
    build_tx_metadata_request_body_web_search_only,
    build_tx_metadata_update_params_web_search_only,
    parse_offline_tx_metadata_response_text,
    parse_tx_metadata_response_text_web_search,
)
from etl.domain.z_gating import apply_gating
from etl.utils.run_config import is_batched
from etl.utils.summary_data import refresh_summary_data


def _oai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for tx_metadata_asset.")
    return OpenAI(api_key=api_key)


def _extract_output_text_from_batch_body(body: Dict[str, Any]) -> str:
    """Pull the assistant message first text block from batch response body.output."""
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


@dg.asset(deps=[], name="9_tx_metadata_asset")
def tx_metadata_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> None:
    if not is_batched(context, pipeline_config):
        context.log.warning("tx_metadata_asset runs only in batched mode; skipping.")
        with db.get_engine().begin() as conn:
            _ = apply_gating(conn, db.database)
        refresh_summary_data(context, db)
        return

    mode = pipeline_config.tx_metadata_mode
    batch_size = pipeline_config.tx_metadata_agreement_batch_size
    engine = db.get_engine()
    schema = db.database
    agreements_table = f"{schema}.agreements"
    pages_table = f"{schema}.pages"
    tagged_outputs_table = f"{schema}.tagged_outputs"

    if mode == TxMetadataMode.OFFLINE:
        _run_offline_mode(
            context, db, engine, schema,
            agreements_table, pages_table, tagged_outputs_table,
            batch_size,
        )
    else:
        _run_web_search_mode(
            context, db, engine, schema,
            agreements_table, batch_size,
        )

    with engine.begin() as conn:
        _ = apply_gating(conn, db.database)
    refresh_summary_data(context, db)


def _run_offline_mode(
    context: AssetExecutionContext,
    db: DBResource,
    engine: Any,
    schema: str,
    agreements_table: str,
    pages_table: str,
    tagged_outputs_table: str,
    batch_size: int,
) -> None:
    """Offline: select agreements missing target/acquirer/deal_type; Batch API; UPDATE three columns."""
    # Select agreements where target IS NULL OR acquirer IS NULL OR deal_type IS NULL
    select_q = text(
        f"""
        SELECT agreement_uuid
        FROM {agreements_table}
        WHERE target IS NULL OR acquirer IS NULL OR deal_type IS NULL
        ORDER BY agreement_uuid ASC
        LIMIT :lim
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(select_q, {"lim": batch_size}).mappings().fetchall()
    agreement_uuids = [r["agreement_uuid"] for r in rows]
    if not agreement_uuids:
        context.log.info("tx_metadata_asset (offline): no agreements need target/acquirer/deal_type.")
        return

    # Fetch for each agreement: all front_matter + first 2 body pages (by page_order)
    # Use same text source as XML: coalesce(tagged_text_gold, tagged_text_corrected, tagged_text, processed_page_content)
    pages_q = text(
        f"""
        WITH ordered AS (
            SELECT
                p.agreement_uuid,
                p.page_order,
                coalesce(p.gold_label, p.source_page_type) AS source_page_type,
                coalesce(t.tagged_text_gold, t.tagged_text_corrected, t.tagged_text, p.processed_page_content) AS page_text,
                ROW_NUMBER() OVER (
                    PARTITION BY p.agreement_uuid, coalesce(p.gold_label, p.source_page_type)
                    ORDER BY p.page_order
                ) AS rn
            FROM {pages_table} p
            LEFT JOIN {tagged_outputs_table} t ON t.page_uuid = p.page_uuid
            WHERE p.agreement_uuid IN :uuids
              AND coalesce(p.gold_label, p.source_page_type) IN ('front_matter', 'body')
        )
        SELECT agreement_uuid, page_order, page_text
        FROM ordered
        WHERE source_page_type = 'front_matter'
           OR (source_page_type = 'body' AND rn <= 2)
        ORDER BY agreement_uuid, page_order
        """
    ).bindparams(bindparam("uuids", expanding=True))
    with engine.begin() as conn:
        page_rows = conn.execute(pages_q, {"uuids": tuple(agreement_uuids)}).mappings().fetchall()

    # Group by agreement_uuid and concatenate text (order preserved by ORDER BY above)
    by_agr: Dict[str, List[str]] = {}
    for r in page_rows:
        agr_uuid = r["agreement_uuid"]
        text_val = r["page_text"] or ""
        by_agr.setdefault(agr_uuid, []).append(text_val)
    agreement_texts = {
        agr_uuid: "\n\n".join(texts)
        for agr_uuid, texts in by_agr.items()
    }

    # Build JSONL for Batch API: one line per agreement
    lines: List[Dict[str, Any]] = []
    for agr_uuid in agreement_uuids:
        concat = agreement_texts.get(agr_uuid) or ""
        if not concat.strip():
            context.log.warning(f"tx_metadata_asset (offline): no page text for {agr_uuid}; skipping.")
            continue
        line = build_offline_tx_metadata_request_body(agr_uuid, concat, model="gpt-5-mini")
        lines.append(line)

    if not lines:
        context.log.info("tx_metadata_asset (offline): no agreements with page text to send.")
        return

    client = _oai_client()
    jsonl_buf = io.StringIO()
    for line in lines:
        jsonl_buf.write(json.dumps(line, ensure_ascii=False) + "\n")
    jsonl_bytes = io.BytesIO(jsonl_buf.getvalue().encode("utf-8"))
    jsonl_bytes.name = "tx_metadata_offline_requests.jsonl"
    in_file = client.files.create(purpose="batch", file=jsonl_bytes)
    completion_window = "24h"
    batch = client.batches.create(
        input_file_id=in_file.id,
        endpoint="/v1/responses",
        completion_window=completion_window,
    )
    context.log.info(
        f"tx_metadata_asset (offline): created batch {batch.id} with {len(lines)} requests; polling until complete."
    )

    # Poll until terminal with exponential backoff (same pattern as ai_repair_poll_asset)
    base_sleep_seconds = 5
    backoff_level = 0
    no_update_polls = 0
    last_progress_snapshot: Tuple[Any, ...] | None = None
    max_sleep_seconds = 30 * 60

    while True:
        b = client.batches.retrieve(batch.id)
        if b.status in ("completed", "failed", "cancelled", "expired"):
            break

        rc = getattr(b, "request_counts", None)
        if rc is not None:
            total = getattr(rc, "total", 0) or 0
            completed = getattr(rc, "completed", 0) or 0
            failed = getattr(rc, "failed", 0) or 0
            progress_snapshot = (b.status, completed, failed)
        else:
            progress_snapshot = (b.status,)

        if progress_snapshot == last_progress_snapshot:
            no_update_polls += 1
        else:
            if backoff_level > 0:
                prev_sleep = min(
                    base_sleep_seconds * (2**backoff_level),
                    max_sleep_seconds,
                )
                context.log.info(
                    f"tx_metadata_asset (offline): backoff reset: interval {prev_sleep}s -> {base_sleep_seconds}s"
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
                    f"tx_metadata_asset (offline): backoff increased: interval {prev_sleep}s -> {new_sleep}s"
                )

        sleep_seconds = min(base_sleep_seconds * (2**backoff_level), max_sleep_seconds)
        context.log.info(
            f"tx_metadata_asset (offline): batch {batch.id} status={b.status}; sleeping {sleep_seconds}s"
        )
        time.sleep(sleep_seconds)

    if b.status != "completed":
        context.log.warning(
            f"tx_metadata_asset (offline): batch {batch.id} ended with status={b.status}; not applying updates."
        )
        return

    ofid = getattr(b, "output_file_id", None)
    if not ofid:
        context.log.warning("tx_metadata_asset (offline): batch has no output_file_id.")
        return

    out_content = client.files.content(ofid)
    if hasattr(out_content, "text"):
        t = out_content.text
        out_text = t() if callable(t) else t
    else:
        out_text = out_content.content.decode("utf-8") if hasattr(out_content, "content") else out_content.decode("utf-8")
    if not isinstance(out_text, str):
        out_text = out_content.read().decode("utf-8")

    update_offline_q = text(
        f"""
        UPDATE {agreements_table}
        SET
            target = COALESCE(target, :target),
            acquirer = COALESCE(acquirer, :acquirer),
            deal_type = COALESCE(deal_type, :deal_type)
        WHERE agreement_uuid = :uuid
        """
    )
    updated = 0
    parse_errors = 0
    for line_str in out_text.strip().splitlines():
        if not line_str.strip():
            continue
        raw = json.loads(line_str)
        rid = raw.get("custom_id")
        resp = raw.get("response")
        if not rid or not resp:
            continue
        sc = resp.get("status_code")
        if sc not in (200, 201, 202):
            parse_errors += 1
            continue
        body = resp.get("body")
        if not body:
            parse_errors += 1
            continue
        try:
            raw_text = _extract_output_text_from_batch_body(body)
            parsed = parse_offline_tx_metadata_response_text(raw_text)
            params = build_offline_update_params(agreement_uuid=rid, parsed=parsed)
            with engine.begin() as conn:
                conn.execute(update_offline_q, params)
            updated += 1
        except Exception as e:
            parse_errors += 1
            context.log.warning(f"tx_metadata_asset (offline): parse error for {rid}: {e}")

    context.log.info(
        f"tx_metadata_asset (offline): batch {batch.id} completed; updated={updated}, parse_errors={parse_errors}"
    )


def _run_web_search_mode(
    context: AssetExecutionContext,
    db: DBResource,
    engine: Any,
    schema: str,
    agreements_table: str,
    batch_size: int,
) -> None:
    """Web-search: select agreements with metadata=0 and target/acquirer set; sync API; UPDATE web columns only."""
    select_q = text(
        f"""
        SELECT agreement_uuid, target, acquirer, filing_date
        FROM {agreements_table}
        WHERE COALESCE(metadata, 0) = 0
          AND target IS NOT NULL AND acquirer IS NOT NULL
        ORDER BY (filing_date IS NULL) ASC, filing_date ASC, agreement_uuid ASC
        LIMIT :lim
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(select_q, {"lim": batch_size}).mappings().fetchall()
    agreements = [dict(r) for r in rows]
    if not agreements:
        context.log.info("tx_metadata_asset (web_search): no agreements need web metadata.")
        return

    context.log.info(f"tx_metadata_asset (web_search): selected {len(agreements)} agreements for enrichment")
    model_name = "gpt-5"
    client = _oai_client()

    success_data: List[Tuple[str, Dict[str, Any]]] = []
    attempted = 0
    parse_errors = 0
    for agreement in agreements:
        attempted += 1
        agr_uuid = agreement["agreement_uuid"]
        body = build_tx_metadata_request_body_web_search_only(agreement, model=model_name)
        try:
            resp = client.responses.create(**body)  # type: ignore[arg-type]
            raw_text = getattr(resp, "output_text", None) or ""
            if not isinstance(raw_text, str):
                raise TypeError("output_text is not a string.")
            obj = parse_tx_metadata_response_text_web_search(raw_text)
            success_data.append((agr_uuid, obj))
        except Exception as e:
            parse_errors += 1
            context.log.warning(f"tx_metadata_asset (web_search): failed for {agr_uuid}: {e}")

    update_web_q = text(
        f"""
        UPDATE {agreements_table}
        SET
            transaction_consideration = :consideration,
            transaction_price_cash = :price_cash,
            transaction_price_stock = :price_stock,
            transaction_price_assets = :price_assets,
            transaction_price_total = :price_total,
            target_type = :target_type,
            acquirer_type = :acquirer_type,
            target_pe = :target_pe,
            acquirer_pe = :acquirer_pe,
            target_industry = :target_industry,
            acquirer_industry = :acquirer_industry,
            announce_date = :announce_date,
            close_date = :close_date,
            deal_status = :deal_status,
            attitude = :attitude,
            purpose = :purpose,
            metadata_sources = :metadata_sources,
            metadata = 1
        WHERE agreement_uuid = :uuid
        """
    )
    updated = 0
    skipped_due_to_error = 0
    for uuid, obj in success_data:
        try:
            params = build_tx_metadata_update_params_web_search_only(
                agreement_uuid=uuid, tx_metadata_obj=obj
            )
            with engine.begin() as conn:
                conn.execute(update_web_q, params)
            updated += 1
        except Exception as e:
            skipped_due_to_error += 1
            context.log.warning(f"tx_metadata_asset (web_search): invalid params for {uuid}: {e}")

    context.log.info(
        f"tx_metadata_asset (web_search): attempted={attempted}, parsed={len(success_data)}, "
        f"updated={updated}, parse_errors={parse_errors}, skipped_due_to_error={skipped_due_to_error}"
    )
