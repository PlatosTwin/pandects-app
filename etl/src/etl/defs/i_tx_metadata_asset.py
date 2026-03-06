"""Transaction metadata enrichment: offline (document-only) or web-search mode.

Offline mode: selects agreements missing target, acquirer, or deal_type; sends
front_matter + first two body pages to gpt-5-mini via OpenAI Batch API; updates
target, acquirer, deal_type only. Offline batch state is persisted so interrupted
runs can resume polling/apply work on the next invocation.

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
from sqlalchemy import text
from sqlalchemy.engine import Connection
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
from etl.utils.openai_batch import (
    extract_output_text_from_batch_body,
    poll_batch_until_terminal,
    read_openai_file_text,
)
from etl.utils.latest_sections_search import refresh_latest_sections_search
from etl.utils.post_asset_refresh import run_post_asset_refresh, run_pre_asset_gating
from etl.utils.run_config import is_batched
from etl.utils.schema_guards import assert_tables_exist

MAX_TX_METADATA_WEB_FAILURES = 3
MAX_TX_METADATA_FAILURE_PAYLOAD_CHARS = 20_000


def _oai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for tx_metadata_asset.")
    return OpenAI(api_key=api_key)


def _fetch_unapplied_offline_batch(conn: Connection, schema: str) -> Dict[str, Any] | None:
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
            FROM {schema}.tx_metadata_offline_batches
            WHERE applied = 0
            ORDER BY created_at ASC
            LIMIT 1
            """
        )
    ).mappings().first()
    if row is None:
        return None
    return dict(row)


def _upsert_offline_batch_row(
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
            INSERT INTO {schema}.tx_metadata_offline_batches (
                batch_id,
                created_at,
                status,
                input_file_id,
                output_file_id,
                error_file_id,
                completion_window,
                request_total,
                applied
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


def _mark_offline_batch_applied(conn: Connection, schema: str, batch_id: str) -> None:
    _ = conn.execute(
        text(
            f"""
            UPDATE {schema}.tx_metadata_offline_batches
            SET applied = 1, applied_at = UTC_TIMESTAMP()
            WHERE batch_id = :batch_id
            """
        ),
        {"batch_id": batch_id},
    )


def _truncate_failure_payload(raw_payload: str | None) -> str | None:
    if raw_payload is None:
        return None
    if len(raw_payload) <= MAX_TX_METADATA_FAILURE_PAYLOAD_CHARS:
        return raw_payload
    return (
        raw_payload[:MAX_TX_METADATA_FAILURE_PAYLOAD_CHARS]
        + f"\n...[truncated to {MAX_TX_METADATA_FAILURE_PAYLOAD_CHARS} chars]"
    )


def _record_web_failure(
    conn: Connection,
    schema: str,
    *,
    agreement_uuid: str,
    error_stage: str,
    failure_reason: str,
    raw_payload: str | None,
    max_failures: int,
) -> None:
    _ = conn.execute(
        text(
            f"""
            INSERT INTO {schema}.tx_metadata_web_failures (
                agreement_uuid,
                failure_count,
                quarantined,
                last_error_stage,
                last_failure_reason,
                last_raw_payload,
                first_failure_at,
                last_failure_at
            )
            VALUES (
                :agreement_uuid,
                1,
                CASE WHEN 1 >= :max_failures THEN 1 ELSE 0 END,
                :error_stage,
                :failure_reason,
                :last_raw_payload,
                UTC_TIMESTAMP(),
                UTC_TIMESTAMP()
            )
            ON DUPLICATE KEY UPDATE
                failure_count = failure_count + 1,
                quarantined = CASE WHEN failure_count + 1 >= :max_failures THEN 1 ELSE quarantined END,
                last_error_stage = VALUES(last_error_stage),
                last_failure_reason = VALUES(last_failure_reason),
                last_raw_payload = VALUES(last_raw_payload),
                last_failure_at = UTC_TIMESTAMP()
            """
        ),
        {
            "agreement_uuid": agreement_uuid,
            "max_failures": max_failures,
            "error_stage": error_stage,
            "failure_reason": failure_reason,
            "last_raw_payload": _truncate_failure_payload(raw_payload),
        },
    )


def _clear_web_failure(conn: Connection, schema: str, *, agreement_uuid: str) -> None:
    _ = conn.execute(
        text(
            f"""
            DELETE FROM {schema}.tx_metadata_web_failures
            WHERE agreement_uuid = :agreement_uuid
            """
        ),
        {"agreement_uuid": agreement_uuid},
    )


def _apply_offline_batch_output(
    context: AssetExecutionContext,
    engine: Any,
    client: OpenAI,
    schema: str,
    agreements_table: str,
    batch: Any,
) -> Tuple[int, int, list[str]]:
    ofid = getattr(batch, "output_file_id", None)
    if not ofid:
        context.log.warning("tx_metadata_asset (offline): batch has no output_file_id.")
        return 0, 0, []

    out_content = client.files.content(ofid)
    out_text = read_openai_file_text(out_content)

    update_offline_q = text(
        f"""
        UPDATE {agreements_table}
        SET
            target = COALESCE(target, :target),
            acquirer = COALESCE(acquirer, :acquirer),
            deal_type = COALESCE(deal_type, :deal_type)
        WHERE agreement_uuid = :uuid
          AND (
            NOT (target <=> COALESCE(target, :target))
            OR NOT (acquirer <=> COALESCE(acquirer, :acquirer))
            OR NOT (deal_type <=> COALESCE(deal_type, :deal_type))
          )
        """
    )
    updated = 0
    parse_errors = 0
    refreshed_uuids: list[str] = []
    for line_str in out_text.strip().splitlines():
        if not line_str.strip():
            continue
        rid = "unknown"
        try:
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
            raw_text = extract_output_text_from_batch_body(body)
            parsed = parse_offline_tx_metadata_response_text(raw_text)
            params = build_offline_update_params(agreement_uuid=rid, parsed=parsed)
        except (TypeError, ValueError, KeyError) as e:
            parse_errors += 1
            context.log.warning(f"tx_metadata_asset (offline): parse error for {rid}: {e}")
            continue

        with engine.begin() as conn:
            result = conn.execute(update_offline_q, params)
            if int(result.rowcount or 0) > 0:
                _ = refresh_latest_sections_search(conn, schema, [str(rid)])
                refreshed_uuids.append(str(rid))
        updated += 1
    return updated, parse_errors, refreshed_uuids


@dg.asset(deps=[], name="8_tx_metadata_asset")
def tx_metadata_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> None:
    run_pre_asset_gating(context, db)

    if not is_batched(context, pipeline_config):
        context.log.warning("tx_metadata_asset runs only in batched mode; skipping.")
        run_post_asset_refresh(context, db, pipeline_config)
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
            context, engine,
            schema,
            agreements_table, pages_table, tagged_outputs_table,
            batch_size,
        )
    else:
        _run_web_search_mode(
            context, engine,
            schema,
            agreements_table, batch_size,
        )

    run_post_asset_refresh(context, db, pipeline_config)


def _run_offline_mode(
    context: AssetExecutionContext,
    engine: Any,
    schema: str,
    agreements_table: str,
    pages_table: str,
    tagged_outputs_table: str,
    batch_size: int,
) -> None:
    """Offline mode with resumable batch polling and idempotent updates."""
    client = _oai_client()
    with engine.begin() as conn:
        assert_tables_exist(
            conn,
            schema=schema,
            table_names=("tx_metadata_offline_batches",),
        )
        existing_batch = _fetch_unapplied_offline_batch(conn, schema)

    if existing_batch is not None:
        existing_batch_id = existing_batch["batch_id"]
        context.log.info(
            f"tx_metadata_asset (offline): resuming existing batch {existing_batch_id}."
        )
        batch = poll_batch_until_terminal(
            context,
            client,
            existing_batch_id,
            log_prefix="tx_metadata_asset (offline)",
        )
        request_total = int(existing_batch["request_total"])
        with engine.begin() as conn:
            _upsert_offline_batch_row(
                conn,
                schema,
                batch=batch,
                completion_window=str(existing_batch["completion_window"]),
                request_total=request_total,
            )

        if batch.status != "completed":
            context.log.warning(
                f"tx_metadata_asset (offline): batch {batch.id} ended with status={batch.status}; not applying updates."
            )
            with engine.begin() as conn:
                _mark_offline_batch_applied(conn, schema, batch.id)
            return

        updated, parse_errors, refreshed_uuids = _apply_offline_batch_output(
            context=context,
            engine=engine,
            client=client,
            schema=schema,
            agreements_table=agreements_table,
            batch=batch,
        )
        with engine.begin() as conn:
            _mark_offline_batch_applied(conn, schema, batch.id)
        context.log.info(
            "tx_metadata_asset (offline): resumed batch %s completed; updated=%s, parse_errors=%s, refreshed_latest_sections_search=%s",
            batch.id,
            updated,
            parse_errors,
            len(refreshed_uuids),
        )
        return

    # Select page payloads for agreements that are actually runnable in offline mode:
    # target/acquirer/deal_type missing, not gated, and has non-empty text in
    # (all front_matter + first 2 body pages). Then cap at batch_size agreements.
    pages_q = text(
        f"""
        WITH candidate_pages AS (
            SELECT
                a.agreement_uuid,
                p.page_order,
                coalesce(p.gold_label, p.source_page_type) AS source_page_type,
                coalesce(t.tagged_text_gold, t.tagged_text_corrected, t.tagged_text, p.processed_page_content) AS page_text,
                ROW_NUMBER() OVER (
                    PARTITION BY a.agreement_uuid, coalesce(p.gold_label, p.source_page_type)
                    ORDER BY p.page_order
                ) AS rn
            FROM {agreements_table} a
            JOIN {pages_table} p
                ON p.agreement_uuid = a.agreement_uuid
            LEFT JOIN {tagged_outputs_table} t ON t.page_uuid = p.page_uuid
            WHERE
                (a.target IS NULL OR a.acquirer IS NULL OR a.deal_type IS NULL)
                AND (a.paginated = True OR a.paginated IS NULL)
                AND a.gated = 0
              AND coalesce(p.gold_label, p.source_page_type) IN ('front_matter', 'body')
        ),
        selected_pages AS (
            SELECT agreement_uuid, page_order, page_text
            FROM candidate_pages
            WHERE source_page_type = 'front_matter'
               OR (source_page_type = 'body' AND rn <= 2)
        ),
        selected_agreements AS (
            SELECT agreement_uuid
            FROM selected_pages
            GROUP BY agreement_uuid
            HAVING SUM(
                CASE
                    WHEN TRIM(COALESCE(page_text, '')) <> '' THEN 1
                    ELSE 0
                END
            ) > 0
            ORDER BY agreement_uuid ASC
            LIMIT :lim
        )
        SELECT sp.agreement_uuid, sp.page_order, sp.page_text
        FROM selected_pages sp
        JOIN selected_agreements sa
            ON sa.agreement_uuid = sp.agreement_uuid
        ORDER BY sp.agreement_uuid ASC, sp.page_order ASC
        """
    )
    with engine.begin() as conn:
        page_rows = conn.execute(pages_q, {"lim": batch_size}).mappings().fetchall()
    if not page_rows:
        context.log.info("tx_metadata_asset (offline): no runnable agreements need target/acquirer/deal_type.")
        return

    # Group by agreement_uuid and concatenate text (order preserved by ORDER BY above)
    by_agr: Dict[str, List[str]] = {}
    for r in page_rows:
        agr_uuid = r["agreement_uuid"]
        text_val = r["page_text"] or ""
        by_agr.setdefault(agr_uuid, []).append(text_val)
    agreement_uuids = list(by_agr.keys())
    context.log.info(
        "tx_metadata_asset (offline): selected %s runnable agreements (batch_size=%s).",
        len(agreement_uuids),
        batch_size,
    )
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

    jsonl_buf = io.StringIO()
    for line in lines:
        _ = jsonl_buf.write(json.dumps(line, ensure_ascii=False) + "\n")
    jsonl_bytes = io.BytesIO(jsonl_buf.getvalue().encode("utf-8"))
    jsonl_bytes.name = "tx_metadata_offline_requests.jsonl"
    in_file = client.files.create(purpose="batch", file=jsonl_bytes)
    completion_window = "24h"
    batch = client.batches.create(
        input_file_id=in_file.id,
        endpoint="/v1/responses",
        completion_window=completion_window,
    )
    with engine.begin() as conn:
        _upsert_offline_batch_row(
            conn,
            schema,
            batch=batch,
            completion_window=completion_window,
            request_total=len(lines),
        )
    context.log.info(
        f"tx_metadata_asset (offline): created batch {batch.id} with {len(lines)} requests; polling until complete."
    )

    final_batch = poll_batch_until_terminal(
        context,
        client,
        batch.id,
        log_prefix="tx_metadata_asset (offline)",
    )
    with engine.begin() as conn:
        _upsert_offline_batch_row(
            conn,
            schema,
            batch=final_batch,
            completion_window=completion_window,
            request_total=len(lines),
        )

    if final_batch.status != "completed":
        context.log.warning(
            f"tx_metadata_asset (offline): batch {final_batch.id} ended with status={final_batch.status}; not applying updates."
        )
        with engine.begin() as conn:
            _mark_offline_batch_applied(conn, schema, final_batch.id)
        return

    updated, parse_errors, refreshed_uuids = _apply_offline_batch_output(
        context=context,
        engine=engine,
        client=client,
        schema=schema,
        agreements_table=agreements_table,
        batch=final_batch,
    )
    with engine.begin() as conn:
        _mark_offline_batch_applied(conn, schema, final_batch.id)

    context.log.info(
        "tx_metadata_asset (offline): batch %s completed; updated=%s, parse_errors=%s, refreshed_latest_sections_search=%s",
        final_batch.id,
        updated,
        parse_errors,
        len(refreshed_uuids),
    )


def _run_web_search_mode(
    context: AssetExecutionContext,
    engine: Any,
    schema: str,
    agreements_table: str,
    batch_size: int,
) -> None:
    """Web-search: select agreements needing metadata with names or URL context; sync API; update web columns."""
    with engine.begin() as conn:
        assert_tables_exist(
            conn,
            schema=schema,
            table_names=("tx_metadata_web_failures",),
        )

    max_failures = MAX_TX_METADATA_WEB_FAILURES

    select_q = text(
        f"""
        SELECT a.agreement_uuid, a.target, a.acquirer, a.filing_date, a.url
        FROM {agreements_table} a
        LEFT JOIN {schema}.tx_metadata_web_failures wf
          ON wf.agreement_uuid = a.agreement_uuid
        WHERE COALESCE(a.metadata, 0) = 0
          AND (
            (
              a.target IS NOT NULL AND TRIM(a.target) <> ''
              AND a.acquirer IS NOT NULL AND TRIM(a.acquirer) <> ''
            )
            OR (a.url IS NOT NULL AND TRIM(a.url) <> '')
          )
          AND COALESCE(wf.quarantined, 0) = 0
          AND COALESCE(wf.failure_count, 0) < :max_failures
        ORDER BY (filing_date IS NULL) ASC, filing_date ASC, agreement_uuid ASC
        LIMIT :lim
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(
            select_q,
            {
                "lim": batch_size,
                "max_failures": max_failures,
            },
        ).mappings().fetchall()
    agreements = [dict(r) for r in rows]
    if not agreements:
        context.log.info("tx_metadata_asset (web_search): no agreements need web metadata.")
        return

    context.log.info(f"tx_metadata_asset (web_search): selected {len(agreements)} agreements for enrichment")
    model_name = "gpt-5.1"
    client = _oai_client()

    class _WebSearchRequestError(RuntimeError):
        def __init__(self, message: str, *, raw_payload: str | None = None) -> None:
            super().__init__(message)
            self.raw_payload = raw_payload

    def _request_web_search_payload(
        agreement_row: Dict[str, Any],
        *,
        max_attempts: int = 3,
    ) -> tuple[Dict[str, Any], str]:
        agr_uuid_local = str(agreement_row["agreement_uuid"])
        body = build_tx_metadata_request_body_web_search_only(agreement_row, model=model_name)
        last_error: Exception | None = None
        last_raw_payload: str | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = client.responses.create(**body)  # type: ignore[arg-type]
                raw_text = getattr(resp, "output_text", None) or ""
                if not isinstance(raw_text, str):
                    raise TypeError("output_text is not a string.")
                last_raw_payload = raw_text
                return parse_tx_metadata_response_text_web_search(raw_text), raw_text
            except Exception as exc:
                last_error = exc
                if attempt == max_attempts:
                    break
                sleep_seconds = 2 ** (attempt - 1)
                context.log.warning(
                    "tx_metadata_asset (web_search): attempt %s/%s failed for %s: %s; retrying in %ss",
                    attempt,
                    max_attempts,
                    agr_uuid_local,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
        if last_error is None:
            raise RuntimeError("Unexpected empty web-search failure state.")
        raise _WebSearchRequestError(
            f"web-search response failed after {max_attempts} attempts: {last_error}",
            raw_payload=last_raw_payload,
        ) from last_error

    success_data: List[Tuple[str, Dict[str, Any], Any, str]] = []
    attempted = 0
    parse_errors = 0
    for agreement in agreements:
        attempted += 1
        agr_uuid = agreement["agreement_uuid"]
        try:
            obj, raw_payload = _request_web_search_payload(agreement)
            success_data.append((agr_uuid, obj, agreement.get("filing_date"), raw_payload))
        except _WebSearchRequestError as e:
            parse_errors += 1
            context.log.warning(f"tx_metadata_asset (web_search): failed for {agr_uuid}: {e}")
            with engine.begin() as conn:
                _record_web_failure(
                    conn,
                    schema,
                    agreement_uuid=str(agr_uuid),
                    error_stage="request_or_parse",
                    failure_reason=str(e),
                    raw_payload=e.raw_payload,
                    max_failures=max_failures,
                )

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
            metadata_uncited_fields = :metadata_uncited_fields,
            metadata = 1
        WHERE agreement_uuid = :uuid
          AND (
            NOT (transaction_consideration <=> :consideration)
            OR NOT (transaction_price_cash <=> :price_cash)
            OR NOT (transaction_price_stock <=> :price_stock)
            OR NOT (transaction_price_assets <=> :price_assets)
            OR NOT (transaction_price_total <=> :price_total)
            OR NOT (target_type <=> :target_type)
            OR NOT (acquirer_type <=> :acquirer_type)
            OR NOT (target_pe <=> :target_pe)
            OR NOT (acquirer_pe <=> :acquirer_pe)
            OR NOT (target_industry <=> :target_industry)
            OR NOT (acquirer_industry <=> :acquirer_industry)
            OR NOT (announce_date <=> :announce_date)
            OR NOT (close_date <=> :close_date)
            OR NOT (deal_status <=> :deal_status)
            OR NOT (attitude <=> :attitude)
            OR NOT (purpose <=> :purpose)
            OR NOT (metadata_sources <=> :metadata_sources)
            OR NOT (metadata_uncited_fields <=> :metadata_uncited_fields)
            OR NOT (metadata <=> 1)
          )
        """
    )
    updated = 0
    skipped_due_to_error = 0
    refreshed_uuids: list[str] = []
    for uuid, obj, filing_date, raw_payload in success_data:
        try:
            params = build_tx_metadata_update_params_web_search_only(
                agreement_uuid=uuid,
                tx_metadata_obj=obj,
                filing_date=filing_date,
                pending_max_age_years=3,
            )
        except (TypeError, ValueError, KeyError) as e:
            skipped_due_to_error += 1
            context.log.warning(f"tx_metadata_asset (web_search): invalid params for {uuid}: {e}")
            with engine.begin() as conn:
                _record_web_failure(
                    conn,
                    schema,
                    agreement_uuid=str(uuid),
                    error_stage="validation",
                    failure_reason=str(e),
                    raw_payload=raw_payload,
                    max_failures=max_failures,
                )
            continue

        with engine.begin() as conn:
            result = conn.execute(update_web_q, params)
            _clear_web_failure(conn, schema, agreement_uuid=str(uuid))
            if int(result.rowcount or 0) > 0:
                _ = refresh_latest_sections_search(conn, schema, [str(uuid)])
                refreshed_uuids.append(str(uuid))
                updated += 1

    context.log.info(
        "tx_metadata_asset (web_search): attempted=%s, parsed=%s, updated=%s, parse_errors=%s, skipped_due_to_error=%s, refreshed_latest_sections_search=%s",
        attempted,
        len(success_data),
        updated,
        parse_errors,
        skipped_due_to_error,
        len(refreshed_uuids),
    )
