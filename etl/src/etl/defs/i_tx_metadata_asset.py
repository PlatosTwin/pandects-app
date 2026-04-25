"""Transaction metadata enrichment: offline (document-only) or web-search mode.

Offline mode can run two independent document-only subflows:
- metadata extraction for target, acquirer, and deal_type
- counsel extraction from the counsel section for target/acquirer counsel

Each subflow persists its own batch state so interrupted runs can resume polling
and applying without blocking the other.

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
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection

from etl.domain.counsel import (
    canonicalize_counsel_name,
    format_counsel_display_name,
    split_counsel_names,
)
from etl.defs.resources import DBResource, PipelineConfig, TxMetadataMode
from etl.defs.h_taxonomy_asset import (
    ingestion_cleanup_a_taxonomy_gold_backfill_asset,
    ingestion_cleanup_b_taxonomy_gold_backfill_asset,
    ingestion_cleanup_c_taxonomy_gold_backfill_asset,
    regular_ingest_taxonomy_gold_backfill_asset,
)
from etl.domain.i_tx_metadata import (
    build_offline_counsel_request_body,
    build_offline_counsel_update_params,
    build_offline_tx_metadata_request_body,
    build_web_search_retry_context,
    build_offline_update_params,
    build_tx_metadata_request_body_web_search_only,
    parse_offline_counsel_response_text,
    build_tx_metadata_update_params_web_search_only,
    merge_retry_web_search_response,
    parse_offline_tx_metadata_response_text,
    parse_tx_metadata_response_text_web_search,
)
from etl.utils.openai_batch import (
    extract_output_text_from_batch_body,
    poll_batch_until_terminal,
    read_openai_file_text,
)
from etl.utils.batch_keys import agreement_batch_key
from etl.utils.latest_sections_search import refresh_latest_sections_search
from etl.utils.logical_job_runs import (
    build_logical_batch_key,
    load_active_logical_run,
    load_active_scope_for_job,
    mark_logical_run_stage_completed,
    should_skip_managed_stage,
)
from etl.utils.post_asset_refresh import run_post_asset_refresh, run_pre_asset_gating
from etl.utils.run_config import ensure_single_batch_run
from etl.utils.schema_guards import assert_tables_exist

if TYPE_CHECKING:
    from openai import OpenAI

MAX_TX_METADATA_FAILURE_PAYLOAD_CHARS = 20_000
WEB_SEARCH_MAX_WORKERS = 30
WEB_SEARCH_COMMIT_BATCH_SIZE = 10
OFFLINE_METADATA_BATCH_KIND = "metadata"
OFFLINE_COUNSEL_BATCH_KIND = "counsel"
COUNSEL_SECTION_STANDARD_ID = "d75eeddb4839a607"


def _oai_client() -> "OpenAI":
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for tx_metadata_asset.")
    return OpenAI(api_key=api_key)


def _has_text_sql(column_sql: str) -> str:
    return f"{column_sql} IS NOT NULL AND TRIM({column_sql}) <> ''"


def _web_search_missing_core_metadata_sql(*, alias: str = "a") -> str:
    consideration_sql = f"{alias}.transaction_consideration"
    price_total_sql = f"{alias}.transaction_price_total"
    price_cash_sql = f"{alias}.transaction_price_cash"
    price_stock_sql = f"{alias}.transaction_price_stock"
    price_assets_sql = f"{alias}.transaction_price_assets"
    target_type_sql = f"{alias}.target_type"
    acquirer_type_sql = f"{alias}.acquirer_type"

    has_consideration = _has_text_sql(consideration_sql)
    has_total = _has_text_sql(price_total_sql)
    has_cash = _has_text_sql(price_cash_sql)
    has_stock = _has_text_sql(price_stock_sql)
    has_assets = _has_text_sql(price_assets_sql)
    has_target_type = _has_text_sql(target_type_sql)
    has_acquirer_type = _has_text_sql(acquirer_type_sql)

    mixed_component_count = (
        f"(CASE WHEN {has_cash} THEN 1 ELSE 0 END + "
        f"CASE WHEN {has_stock} THEN 1 ELSE 0 END + "
        f"CASE WHEN {has_assets} THEN 1 ELSE 0 END)"
    )

    return (
        f"NOT ({has_consideration})\n"
        f"            OR NOT ({has_target_type})\n"
        f"            OR NOT ({has_acquirer_type})\n"
        f"            OR (\n"
        f"                COALESCE({consideration_sql}, '') = 'cash'\n"
        f"                AND (\n"
        f"                    NOT ({has_cash})\n"
        f"                    OR NOT ({has_total})\n"
        f"                )\n"
        f"            )\n"
        f"            OR (\n"
        f"                COALESCE({consideration_sql}, '') = 'stock'\n"
        f"                AND (\n"
        f"                    NOT ({has_stock})\n"
        f"                    OR NOT ({has_total})\n"
        f"                )\n"
        f"            )\n"
        f"            OR (\n"
        f"                COALESCE({consideration_sql}, '') = 'mixed'\n"
        f"                AND (\n"
        f"                    {mixed_component_count} < 2\n"
        f"                    OR NOT ({has_total})\n"
        f"                )\n"
        f"            )"
    )


def _select_recent_pending_agreement_uuids_for_web_search(
    conn: Connection,
    schema: str,
    *,
    lookback_years: int = 1,
) -> list[str]:
    if lookback_years < 0:
        raise ValueError("lookback_years must be >= 0.")
    rows = conn.execute(
        text(
            f"""
            SELECT a.agreement_uuid
            FROM {schema}.agreements a
            WHERE COALESCE(a.gated, 0) = 0
              AND COALESCE(a.deal_status, '') = 'pending'
              AND a.filing_date IS NOT NULL
              AND a.filing_date >= DATE_SUB(CURDATE(), INTERVAL :lookback_years YEAR)
            ORDER BY a.filing_date DESC, a.agreement_uuid ASC
            """
        ),
        {"lookback_years": lookback_years},
    ).scalars().all()
    return [str(row) for row in rows if row is not None]


def _web_search_model_for_agreement(agreement_row: Dict[str, Any]) -> str:
    if int(agreement_row.get("initial_metadata_pass", 1) or 0) == 1:
        return "gpt-5.4-mini"
    return "gpt-5.4"


def _fetch_unapplied_offline_batch(
    conn: Connection,
    schema: str,
    *,
    batch_kind: str,
    batch_key: str | None = None,
) -> Dict[str, Any] | None:
    batch_key_clause = ""
    params: dict[str, object] = {"batch_kind": batch_kind}
    if batch_key is not None:
        batch_key_clause = "AND batch_key = :batch_key"
        params["batch_key"] = batch_key
    row = conn.execute(
        text(
            f"""
            SELECT
                batch_kind,
                batch_id,
                status,
                input_file_id,
                output_file_id,
                error_file_id,
                completion_window,
                request_total,
                batch_key
            FROM {schema}.tx_metadata_offline_batches
            WHERE applied = 0
              AND batch_kind = :batch_kind
              {batch_key_clause}
            ORDER BY created_at ASC
            LIMIT 1
            """
        ),
        params,
    ).mappings().first()
    if row is None:
        return None
    return dict(row)


def _upsert_offline_batch_row(
    conn: Connection,
    schema: str,
    *,
    batch_kind: str,
    batch: Any,
    completion_window: str,
    request_total: int,
    batch_key: str,
) -> None:
    _ = conn.execute(
        text(
            f"""
            INSERT INTO {schema}.tx_metadata_offline_batches (
                batch_kind,
                batch_id,
                created_at,
                status,
                input_file_id,
                output_file_id,
                error_file_id,
                completion_window,
                request_total,
                batch_key,
                applied
            )
            VALUES (
                :batch_kind,
                :batch_id,
                UTC_TIMESTAMP(),
                :status,
                :input_file_id,
                :output_file_id,
                :error_file_id,
                :completion_window,
                :request_total,
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
                batch_key = VALUES(batch_key)
            """
        ),
        {
            "batch_kind": batch_kind,
            "batch_id": batch.id,
            "status": batch.status,
            "input_file_id": getattr(batch, "input_file_id", None),
            "output_file_id": getattr(batch, "output_file_id", None),
            "error_file_id": getattr(batch, "error_file_id", None),
            "completion_window": completion_window,
            "request_total": request_total,
            "batch_key": batch_key,
        },
    )

def _mark_offline_batch_applied(
    conn: Connection,
    schema: str,
    *,
    batch_kind: str,
    batch_id: str,
) -> None:
    _ = conn.execute(
        text(
            f"""
            UPDATE {schema}.tx_metadata_offline_batches
            SET applied = 1, applied_at = UTC_TIMESTAMP()
            WHERE batch_id = :batch_id
              AND batch_kind = :batch_kind
            """
        ),
        {"batch_id": batch_id, "batch_kind": batch_kind},
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
                0,
                :error_stage,
                :failure_reason,
                :last_raw_payload,
                UTC_TIMESTAMP(),
                UTC_TIMESTAMP()
            )
            ON DUPLICATE KEY UPDATE
                failure_count = failure_count + 1,
                quarantined = 0,
                last_error_stage = VALUES(last_error_stage),
                last_failure_reason = VALUES(last_failure_reason),
                last_raw_payload = VALUES(last_raw_payload),
                last_failure_at = UTC_TIMESTAMP()
            """
        ),
        {
            "agreement_uuid": agreement_uuid,
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


def _chunk_agreements(
    agreements: list[Dict[str, Any]],
    *,
    chunk_size: int,
) -> list[list[Dict[str, Any]]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    return [
        agreements[index:index + chunk_size]
        for index in range(0, len(agreements), chunk_size)
    ]


def _persist_web_search_successes(
    *,
    context: AssetExecutionContext,
    engine: Any,
    schema: str,
    update_web_q: Any,
    successes: list[tuple[Dict[str, Any], Dict[str, Any], Any, str, Dict[str, int], int]],
    failed_uuid_by_stage: Dict[str, List[str]],
    token_totals: Dict[str, int],
    refreshed_uuids: list[str],
) -> tuple[int, int, int]:
    updated = 0
    validation_failures = 0
    refreshed = 0
    for agreement_row, obj, filing_date, raw_payload, response_usage, search_count in successes:
        uuid = str(agreement_row["agreement_uuid"])
        try:
            merged_obj = merge_retry_web_search_response(
                agreement=agreement_row,
                tx_metadata_obj=obj,
            )
            params = build_tx_metadata_update_params_web_search_only(
                agreement_uuid=uuid,
                tx_metadata_obj=merged_obj,
                response_usage=response_usage,
                search_count=search_count,
                filing_date=filing_date,
                pending_max_age_years=3,
            )
        except (TypeError, ValueError, KeyError) as e:
            validation_failures += 1
            failed_uuid_by_stage["validation"].append(str(uuid))
            context.log.warning(f"tx_metadata_asset (web_search): invalid params for {uuid}: {e}")
            with engine.begin() as conn:
                _record_web_failure(
                    conn,
                    schema,
                    agreement_uuid=str(uuid),
                    error_stage="validation",
                    failure_reason=str(e),
                    raw_payload=raw_payload,
                )
            continue

        with engine.begin() as conn:
            result = conn.execute(update_web_q, params)
            _clear_web_failure(conn, schema, agreement_uuid=str(uuid))
            if int(result.rowcount or 0) > 0:
                _ = refresh_latest_sections_search(conn, schema, [str(uuid)])
                refreshed_uuids.append(str(uuid))
                updated += 1
                refreshed += 1
            for key, value in response_usage.items():
                token_totals[key] += value

    return updated, validation_failures, refreshed


def _extract_response_usage(resp: Any) -> Dict[str, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        raise ValueError("web-search response missing usage.")
    if hasattr(usage, "model_dump"):
        usage_obj = usage.model_dump()
    elif isinstance(usage, dict):
        usage_obj = usage
    else:
        raise TypeError("web-search response usage must be a dict-like object.")

    extracted: Dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage_obj.get(key)
        if not isinstance(value, int):
            raise TypeError(f"web-search response usage.{key} must be an integer.")
        extracted[key] = value
    return extracted


def _extract_web_search_count(resp: Any) -> int:
    output = getattr(resp, "output", None)
    if output is None:
        return 0
    if not isinstance(output, list):
        raise TypeError("web-search response output must be a list when present.")

    search_count = 0
    for item in output:
        item_type = getattr(item, "type", None)
        if item_type != "web_search_call":
            continue
        action = getattr(item, "action", None)
        if getattr(action, "type", None) == "search":
            search_count += 1
    return search_count


def _apply_offline_batch_output(
    context: AssetExecutionContext,
    engine: Any,
    client: "OpenAI",
    schema: str,
    agreements_table: str,
    batch: Any,
    *,
    update_sql: Any | None = None,
    parse_response_text: Any | None = None,
    build_update_params: Any | None = None,
    log_prefix: str = "tx_metadata_asset (offline)",
 ) -> Tuple[int, int, list[str], list[str]]:
    ofid = getattr(batch, "output_file_id", None)
    if not ofid:
        context.log.warning("%s: batch has no output_file_id.", log_prefix)
        return 0, 0, [], []

    out_content = client.files.content(ofid)
    out_text = read_openai_file_text(out_content)
    resolved_update_sql = (
        _metadata_offline_update_sql(agreements_table)
        if update_sql is None
        else update_sql
    )
    resolved_parse_response_text = (
        parse_offline_tx_metadata_response_text
        if parse_response_text is None
        else parse_response_text
    )
    resolved_build_update_params = (
        build_offline_update_params
        if build_update_params is None
        else build_update_params
    )
    updated = 0
    parse_errors = 0
    refreshed_uuids: list[str] = []
    processed_uuids: list[str] = []
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
            parsed = resolved_parse_response_text(raw_text)
            params = resolved_build_update_params(agreement_uuid=rid, parsed=parsed)
        except (TypeError, ValueError, KeyError) as e:
            parse_errors += 1
            context.log.warning("%s: parse error for %s: %s", log_prefix, rid, e)
            continue
        processed_uuids.append(str(rid))

        with engine.begin() as conn:
            result = conn.execute(resolved_update_sql, params)
            if int(result.rowcount or 0) > 0:
                _ = refresh_latest_sections_search(conn, schema, [str(rid)])
                refreshed_uuids.append(str(rid))
                updated += 1
    return updated, parse_errors, refreshed_uuids, processed_uuids


def _metadata_offline_update_sql(agreements_table: str) -> Any:
    return text(
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


def _counsel_offline_update_sql(agreements_table: str) -> Any:
    return text(
        f"""
        UPDATE {agreements_table}
        SET
            target_counsel = COALESCE(target_counsel, :target_counsel),
            acquirer_counsel = COALESCE(acquirer_counsel, :acquirer_counsel)
        WHERE agreement_uuid = :uuid
          AND (
            NOT (target_counsel <=> COALESCE(target_counsel, :target_counsel))
            OR NOT (acquirer_counsel <=> COALESCE(acquirer_counsel, :acquirer_counsel))
          )
        """
    )


def _sync_counsel_mappings(
    *,
    context: AssetExecutionContext,
    engine: Any,
    schema: str,
    agreements_table: str,
    agreement_uuids: Sequence[str] | None = None,
    log_prefix: str = "tx_metadata_asset (offline counsel)",
) -> None:
    counsel_table = f"{schema}.counsel"
    agreement_counsel_table = f"{schema}.agreement_counsel"
    target_uuids = tuple(sorted({agreement_uuid for agreement_uuid in (agreement_uuids or []) if agreement_uuid}))
    if agreement_uuids is not None and not target_uuids:
        context.log.info("%s: no agreement counsel rows needed syncing.", log_prefix)
        return
    sync_sql = (
        text(
            f"""
            SELECT
                agreement_uuid,
                target_counsel,
                acquirer_counsel
            FROM {agreements_table}
            WHERE agreement_uuid IN :agreement_uuids
              AND (
                    (target_counsel IS NOT NULL AND TRIM(target_counsel) <> '')
                    OR (acquirer_counsel IS NOT NULL AND TRIM(acquirer_counsel) <> '')
              )
            ORDER BY agreement_uuid ASC
            """
        ).bindparams(bindparam("agreement_uuids", expanding=True))
        if agreement_uuids is not None
        else text(
            f"""
            SELECT
                agreement_uuid,
                target_counsel,
                acquirer_counsel
            FROM {agreements_table}
            WHERE (target_counsel IS NOT NULL AND TRIM(target_counsel) <> '')
               OR (acquirer_counsel IS NOT NULL AND TRIM(acquirer_counsel) <> '')
            ORDER BY agreement_uuid ASC
            """
        )
    )
    load_counsel_sql = text(
        f"""
        SELECT counsel_id, canonical_name, canonical_name_normalized
        FROM {counsel_table}
        """
    )
    insert_counsel_sql = text(
        f"""
        INSERT INTO {counsel_table} (
            canonical_name,
            canonical_name_normalized
        ) VALUES (
            :canonical_name,
            :canonical_name_normalized
        )
        """
    )
    select_counsel_id_sql = text(
        f"""
        SELECT counsel_id
        FROM {counsel_table}
        WHERE canonical_name_normalized = :canonical_name_normalized
        LIMIT 1
        """
    )
    delete_mapping_sql = text(
        f"DELETE FROM {agreement_counsel_table} WHERE agreement_uuid = :agreement_uuid"
    )
    insert_mapping_sql = text(
        f"""
        INSERT INTO {agreement_counsel_table} (
            agreement_uuid,
            side,
            position,
            raw_name,
            counsel_id
        ) VALUES (
            :agreement_uuid,
            :side,
            :position,
            :raw_name,
            :counsel_id
        )
        """
    )

    with engine.begin() as conn:
        assert_tables_exist(conn, schema=schema, table_names=("counsel", "agreement_counsel"))
        agreements = conn.execute(
            sync_sql,
            {"agreement_uuids": target_uuids} if agreement_uuids is not None else {},
        ).mappings().fetchall()
        counsel_rows = conn.execute(load_counsel_sql).mappings().fetchall()
        counsel_by_key = {
            str(row["canonical_name_normalized"]): int(row["counsel_id"])
            for row in counsel_rows
            if row.get("canonical_name_normalized") is not None and row.get("counsel_id") is not None
        }

        mapping_rows: list[dict[str, object]] = []
        synced_agreements = 0
        for agreement in agreements:
            agreement_uuid = str(agreement["agreement_uuid"])
            conn.execute(delete_mapping_sql, {"agreement_uuid": agreement_uuid})
            synced_agreements += 1

            for side, raw_value in (
                ("target", agreement.get("target_counsel")),
                ("acquirer", agreement.get("acquirer_counsel")),
            ):
                seen_keys: set[str] = set()
                position = 0
                for raw_name in split_counsel_names(raw_value):
                    canonical_name_normalized = canonicalize_counsel_name(raw_name)
                    canonical_name = format_counsel_display_name(raw_name)
                    if canonical_name_normalized is None or canonical_name is None:
                        continue
                    if canonical_name_normalized in seen_keys:
                        continue
                    seen_keys.add(canonical_name_normalized)
                    position += 1

                    counsel_id = counsel_by_key.get(canonical_name_normalized)
                    if counsel_id is None:
                        result = conn.execute(
                            insert_counsel_sql,
                            {
                                "canonical_name": canonical_name,
                                "canonical_name_normalized": canonical_name_normalized,
                            },
                        )
                        inserted_id = getattr(result, "lastrowid", None)
                        if inserted_id is None:
                            inserted_row = conn.execute(
                                select_counsel_id_sql,
                                {
                                    "canonical_name_normalized": canonical_name_normalized,
                                },
                            ).mappings().first()
                            if inserted_row is None or inserted_row.get("counsel_id") is None:
                                raise RuntimeError(
                                    f"Unable to resolve counsel_id for {canonical_name_normalized!r}."
                                )
                            counsel_id = int(inserted_row["counsel_id"])
                        else:
                            counsel_id = int(inserted_id)
                        counsel_by_key[canonical_name_normalized] = counsel_id

                    mapping_rows.append(
                        {
                            "agreement_uuid": agreement_uuid,
                            "side": side,
                            "position": position,
                            "raw_name": raw_name,
                            "counsel_id": counsel_id,
                        }
                    )

        if mapping_rows:
            conn.execute(insert_mapping_sql, mapping_rows)
    context.log.info(
        "%s: synced %s agreement counsel rows across %s agreements.",
        log_prefix,
        len(mapping_rows),
        synced_agreements,
    )


@dg.asset(deps=[], name="10_tx_metadata_asset")
def tx_metadata_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> None:
    """Manual transaction-metadata entrypoint outside the explicit ingest jobs."""
    run_pre_asset_gating(context, db)

    ensure_single_batch_run(context, pipeline_config, asset_name="tx_metadata_asset")

    mode = pipeline_config.tx_metadata_mode
    batch_size = pipeline_config.tx_metadata_agreement_batch_size
    engine = db.get_engine()
    schema = db.database
    agreements_table = f"{schema}.agreements"
    pages_table = f"{schema}.pages"
    tagged_outputs_table = f"{schema}.tagged_outputs"

    if mode == TxMetadataMode.OFFLINE:
        processed_agreement_uuids = _run_offline_mode(
            context, engine,
            schema,
            agreements_table, pages_table, tagged_outputs_table,
            batch_size,
            log_prefix="tx_metadata_asset (offline)",
        )
        context.log.info("tx_metadata_asset (offline): processed %s agreements", len(processed_agreement_uuids))
    else:
        summary = _run_web_search_mode(
            context, engine,
            schema,
            agreements_table, batch_size,
            log_prefix="tx_metadata_asset (web_search)",
        )
        context.add_output_metadata(
            {
                "total_web_searches": summary["total_searches"],
                "web_searches_by_agreement": dg.MetadataValue.json(
                    summary["searches_by_agreement"]
                ),
            }
        )

    run_post_asset_refresh(context, db, pipeline_config)


def _maybe_skip_managed_tx_metadata_stage(
    context: AssetExecutionContext,
    *,
    db: DBResource,
    job_name: str,
    stage_name: str,
    log_prefix: str,
    skip_if_completed: bool,
) -> bool:
    if not skip_if_completed:
        return False
    should_skip, current_stage = should_skip_managed_stage(
        db=db,
        job_name=job_name,
        stage_name=stage_name,
    )
    if not should_skip:
        return False
    context.log.info(
        "%s: skipping because logical run already reached %s.",
        log_prefix,
        current_stage,
    )
    return True


def _run_managed_tx_metadata_offline_asset(
    context: AssetExecutionContext,
    *,
    db: DBResource,
    pipeline_config: PipelineConfig,
    agreement_uuids: list[str],
    asset_name: str,
    job_name: str,
    stage_name: str,
    log_prefix: str,
    skip_if_completed: bool = False,
) -> list[str]:
    ensure_single_batch_run(context, pipeline_config, asset_name=asset_name)
    if _maybe_skip_managed_tx_metadata_stage(
        context,
        db=db,
        job_name=job_name,
        stage_name=stage_name,
        log_prefix=log_prefix,
        skip_if_completed=skip_if_completed,
    ):
        return []

    scope_uuids = load_active_scope_for_job(
        context,
        db=db,
        job_name=job_name,
        fallback_agreement_uuids=agreement_uuids,
    )
    if not scope_uuids:
        mark_logical_run_stage_completed(
            db=db,
            job_name=job_name,
            stage_name=stage_name,
        )
        return []

    run_pre_asset_gating(context, db)
    active_run = load_active_logical_run(db=db, job_name=job_name)
    engine = db.get_engine()
    schema = db.database
    agreements_table = f"{schema}.agreements"
    pages_table = f"{schema}.pages"
    tagged_outputs_table = f"{schema}.tagged_outputs"
    processed_agreement_uuids = _run_offline_mode(
        context,
        engine,
        schema,
        agreements_table,
        pages_table,
        tagged_outputs_table,
        pipeline_config.tx_metadata_agreement_batch_size,
        target_agreement_uuids=scope_uuids,
        batch_key_override=build_logical_batch_key(
            logical_run_id=None if active_run is None else str(active_run["logical_run_id"]),
            stage_name=stage_name,
            default_key=agreement_batch_key(scope_uuids) if scope_uuids else None,
        ),
        log_prefix=log_prefix,
    )
    run_post_asset_refresh(context, db, pipeline_config)
    mark_logical_run_stage_completed(
        db=db,
        job_name=job_name,
        stage_name=stage_name,
    )
    return processed_agreement_uuids


def _run_managed_tx_metadata_web_search_asset(
    context: AssetExecutionContext,
    *,
    db: DBResource,
    pipeline_config: PipelineConfig,
    agreement_uuids: list[str],
    asset_name: str,
    job_name: str,
    stage_name: str,
    log_prefix: str,
    skip_if_completed: bool = False,
) -> list[str]:
    ensure_single_batch_run(context, pipeline_config, asset_name=asset_name)
    if _maybe_skip_managed_tx_metadata_stage(
        context,
        db=db,
        job_name=job_name,
        stage_name=stage_name,
        log_prefix=log_prefix,
        skip_if_completed=skip_if_completed,
    ):
        return []

    scope_uuids = load_active_scope_for_job(
        context,
        db=db,
        job_name=job_name,
        fallback_agreement_uuids=agreement_uuids,
    )

    forced_verification_uuids: list[str] = []
    if job_name == "regular_ingest":
        with db.get_engine().begin() as conn:
            forced_verification_uuids = _select_recent_pending_agreement_uuids_for_web_search(
                conn,
                db.database,
            )
        if forced_verification_uuids:
            context.log.info(
                "%s: adding %s recent pending agreements for end-of-job web-search verification.",
                log_prefix,
                len(forced_verification_uuids),
            )

    target_scope_uuids = sorted({*scope_uuids, *forced_verification_uuids})
    if not target_scope_uuids:
        mark_logical_run_stage_completed(
            db=db,
            job_name=job_name,
            stage_name=stage_name,
            complete_run=True,
        )
        return []

    run_pre_asset_gating(context, db)
    engine = db.get_engine()
    schema = db.database
    agreements_table = f"{schema}.agreements"
    summary = _run_web_search_mode(
        context,
        engine,
        schema,
        agreements_table,
        pipeline_config.tx_metadata_agreement_batch_size,
        target_agreement_uuids=target_scope_uuids,
        force_include_agreement_uuids=forced_verification_uuids,
        log_prefix=log_prefix,
    )
    context.add_output_metadata(
        {
            "total_web_searches": summary["total_searches"],
            "web_searches_by_agreement": dg.MetadataValue.json(summary["searches_by_agreement"]),
        }
    )
    run_post_asset_refresh(context, db, pipeline_config)
    mark_logical_run_stage_completed(
        db=db,
        job_name=job_name,
        stage_name=stage_name,
        complete_run=True,
    )
    processed_uuids = summary["processed_uuids"]
    if not isinstance(processed_uuids, list):
        raise TypeError(f"{asset_name} expected summary['processed_uuids'] to be a list.")
    return sorted({str(agreement_uuid) for agreement_uuid in processed_uuids if agreement_uuid})


@dg.asset(
    name="10-01_regular_ingest_tx_metadata_offline_asset",
    ins={"agreement_uuids": dg.AssetIn(key=regular_ingest_taxonomy_gold_backfill_asset.key)},
)
def regular_ingest_tx_metadata_offline_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    agreement_uuids: list[str],
) -> list[str]:
    return _run_managed_tx_metadata_offline_asset(
        context,
        db=db,
        pipeline_config=pipeline_config,
        agreement_uuids=agreement_uuids,
        asset_name="regular_ingest_tx_metadata_offline_asset",
        job_name="regular_ingest",
        stage_name="regular_ingest_tx_metadata_offline",
        log_prefix="regular_ingest_tx_metadata_offline_asset",
        skip_if_completed=True,
    )


@dg.asset(
    name="10-02_regular_ingest_tx_metadata_web_search_asset",
    ins={"agreement_uuids": dg.AssetIn(key=regular_ingest_tx_metadata_offline_asset.key)},
)
def regular_ingest_tx_metadata_web_search_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    agreement_uuids: list[str],
) -> list[str]:
    return _run_managed_tx_metadata_web_search_asset(
        context,
        db=db,
        pipeline_config=pipeline_config,
        agreement_uuids=agreement_uuids,
        asset_name="regular_ingest_tx_metadata_web_search_asset",
        job_name="regular_ingest",
        stage_name="regular_ingest_tx_metadata_web_search",
        log_prefix="regular_ingest_tx_metadata_web_search_asset",
        skip_if_completed=True,
    )


@dg.asset(
    name="10-03_ingestion_cleanup_a_tx_metadata_offline_asset",
    ins={"agreement_uuids": dg.AssetIn(key=ingestion_cleanup_a_taxonomy_gold_backfill_asset.key)},
)
def ingestion_cleanup_a_tx_metadata_offline_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    agreement_uuids: list[str],
) -> list[str]:
    return _run_managed_tx_metadata_offline_asset(
        context,
        db=db,
        pipeline_config=pipeline_config,
        agreement_uuids=agreement_uuids,
        asset_name="ingestion_cleanup_a_tx_metadata_offline_asset",
        job_name="ingestion_cleanup_a",
        stage_name="ingestion_cleanup_a_tx_metadata_offline",
        log_prefix="ingestion_cleanup_a_tx_metadata_offline_asset",
        skip_if_completed=True,
    )


@dg.asset(
    name="10-04_ingestion_cleanup_a_tx_metadata_web_search_asset",
    ins={"agreement_uuids": dg.AssetIn(key=ingestion_cleanup_a_tx_metadata_offline_asset.key)},
)
def ingestion_cleanup_a_tx_metadata_web_search_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    agreement_uuids: list[str],
) -> list[str]:
    return _run_managed_tx_metadata_web_search_asset(
        context,
        db=db,
        pipeline_config=pipeline_config,
        agreement_uuids=agreement_uuids,
        asset_name="ingestion_cleanup_a_tx_metadata_web_search_asset",
        job_name="ingestion_cleanup_a",
        stage_name="ingestion_cleanup_a_tx_metadata_web_search",
        log_prefix="ingestion_cleanup_a_tx_metadata_web_search_asset",
        skip_if_completed=True,
    )


@dg.asset(
    name="10-05_ingestion_cleanup_b_tx_metadata_offline_asset",
    ins={"agreement_uuids": dg.AssetIn(key=ingestion_cleanup_b_taxonomy_gold_backfill_asset.key)},
)
def ingestion_cleanup_b_tx_metadata_offline_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    agreement_uuids: list[str],
) -> list[str]:
    return _run_managed_tx_metadata_offline_asset(
        context,
        db=db,
        pipeline_config=pipeline_config,
        agreement_uuids=agreement_uuids,
        asset_name="ingestion_cleanup_b_tx_metadata_offline_asset",
        job_name="ingestion_cleanup_b",
        stage_name="ingestion_cleanup_b_tx_metadata_offline",
        log_prefix="ingestion_cleanup_b_tx_metadata_offline_asset",
        skip_if_completed=True,
    )


@dg.asset(
    name="10-06_ingestion_cleanup_b_tx_metadata_web_search_asset",
    ins={"agreement_uuids": dg.AssetIn(key=ingestion_cleanup_b_tx_metadata_offline_asset.key)},
)
def ingestion_cleanup_b_tx_metadata_web_search_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    agreement_uuids: list[str],
) -> list[str]:
    return _run_managed_tx_metadata_web_search_asset(
        context,
        db=db,
        pipeline_config=pipeline_config,
        agreement_uuids=agreement_uuids,
        asset_name="ingestion_cleanup_b_tx_metadata_web_search_asset",
        job_name="ingestion_cleanup_b",
        stage_name="ingestion_cleanup_b_tx_metadata_web_search",
        log_prefix="ingestion_cleanup_b_tx_metadata_web_search_asset",
        skip_if_completed=True,
    )


@dg.asset(
    name="10-07_ingestion_cleanup_c_tx_metadata_offline_asset",
    ins={"agreement_uuids": dg.AssetIn(key=ingestion_cleanup_c_taxonomy_gold_backfill_asset.key)},
)
def ingestion_cleanup_c_tx_metadata_offline_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    agreement_uuids: list[str],
) -> list[str]:
    return _run_managed_tx_metadata_offline_asset(
        context,
        db=db,
        pipeline_config=pipeline_config,
        agreement_uuids=agreement_uuids,
        asset_name="ingestion_cleanup_c_tx_metadata_offline_asset",
        job_name="ingestion_cleanup_c",
        stage_name="ingestion_cleanup_c_tx_metadata_offline",
        log_prefix="ingestion_cleanup_c_tx_metadata_offline_asset",
        skip_if_completed=True,
    )


@dg.asset(
    name="10-08_ingestion_cleanup_c_tx_metadata_web_search_asset",
    ins={"agreement_uuids": dg.AssetIn(key=ingestion_cleanup_c_tx_metadata_offline_asset.key)},
)
def ingestion_cleanup_c_tx_metadata_web_search_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    agreement_uuids: list[str],
) -> list[str]:
    return _run_managed_tx_metadata_web_search_asset(
        context,
        db=db,
        pipeline_config=pipeline_config,
        agreement_uuids=agreement_uuids,
        asset_name="ingestion_cleanup_c_tx_metadata_web_search_asset",
        job_name="ingestion_cleanup_c",
        stage_name="ingestion_cleanup_c_tx_metadata_web_search",
        log_prefix="ingestion_cleanup_c_tx_metadata_web_search_asset",
        skip_if_completed=True,
    )


def _run_offline_mode(
    context: AssetExecutionContext,
    engine: Any,
    schema: str,
    agreements_table: str,
    pages_table: str,
    tagged_outputs_table: str,
    batch_size: int,
    target_agreement_uuids: list[str] | None = None,
    batch_key_override: str | None = None,
    log_prefix: str = "tx_metadata_asset (offline)",
) -> list[str]:
    """Offline mode with resumable batch polling and idempotent updates."""
    explicit_scope = target_agreement_uuids is not None
    scoped_uuids = sorted({str(agreement_uuid) for agreement_uuid in (target_agreement_uuids or []) if agreement_uuid})
    if explicit_scope and not scoped_uuids:
        context.log.info("%s: explicit empty scope; no offline metadata work to run.", log_prefix)
        return []
    client = _oai_client()
    batch_key = batch_key_override or (agreement_batch_key(scoped_uuids) if scoped_uuids else "global")
    processed_agreement_uuids: set[str] = set()
    with engine.begin() as conn:
        assert_tables_exist(
            conn,
            schema=schema,
            table_names=("tx_metadata_offline_batches",),
        )
    metadata_batch = _prepare_offline_metadata_batch(
        context=context,
        engine=engine,
        client=client,
        schema=schema,
        agreements_table=agreements_table,
        pages_table=pages_table,
        tagged_outputs_table=tagged_outputs_table,
        batch_size=batch_size,
        target_agreement_uuids=scoped_uuids if explicit_scope else None,
        batch_key=batch_key,
        log_prefix=f"{log_prefix} metadata",
    )
    initial_results: dict[str, dict[str, Any]] = {}
    if metadata_batch is None:
        counsel_batch = _prepare_offline_counsel_batch(
            context=context,
            engine=engine,
            client=client,
            schema=schema,
            agreements_table=agreements_table,
            batch_size=batch_size,
            target_agreement_uuids=scoped_uuids if explicit_scope else None,
            batch_key=batch_key,
            log_prefix=f"{log_prefix} counsel",
        )
        initial_results = _process_offline_batches(
            context=context,
            engine=engine,
            schema=schema,
            agreements_table=agreements_table,
            pending_batches=[counsel_batch] if counsel_batch is not None else [],
        )
    else:
        with ThreadPoolExecutor(max_workers=2) as executor:
            metadata_future = executor.submit(
                _process_offline_batches,
                context=context,
                engine=engine,
                schema=schema,
                agreements_table=agreements_table,
                pending_batches=[metadata_batch],
            )
            counsel_batch = _prepare_offline_counsel_batch(
                context=context,
                engine=engine,
                client=client,
                schema=schema,
                agreements_table=agreements_table,
                batch_size=batch_size,
                target_agreement_uuids=scoped_uuids if explicit_scope else None,
                batch_key=batch_key,
                log_prefix=f"{log_prefix} counsel",
            )
            counsel_future = None
            if counsel_batch is not None:
                counsel_future = executor.submit(
                    _process_offline_batches,
                    context=context,
                    engine=engine,
                    schema=schema,
                    agreements_table=agreements_table,
                    pending_batches=[counsel_batch],
                )
            initial_results.update(metadata_future.result())
            if counsel_future is not None:
                initial_results.update(counsel_future.result())
    for result in initial_results.values():
        raw_processed_uuids = result.get("processed_uuids")
        if isinstance(raw_processed_uuids, list):
            processed_agreement_uuids.update(str(agreement_uuid) for agreement_uuid in raw_processed_uuids if agreement_uuid)

    _sync_counsel_mappings(
        context=context,
        engine=engine,
        schema=schema,
        agreements_table=agreements_table,
        agreement_uuids=_collect_counsel_sync_uuids(initial_results),
        log_prefix=f"{log_prefix} counsel",
    )

    metadata_result = initial_results.get(OFFLINE_METADATA_BATCH_KIND)
    metadata_updated = int(metadata_result["updated"]) if metadata_result is not None else 0
    if metadata_updated <= 0:
        return sorted(processed_agreement_uuids)

    context.log.info(
        "%s: metadata filled %s agreements; rechecking counsel candidates now that target/acquirer may be available.",
        log_prefix,
        metadata_updated,
    )
    follow_up_counsel_batch = _prepare_offline_counsel_batch(
        context=context,
        engine=engine,
        client=client,
        schema=schema,
        agreements_table=agreements_table,
        batch_size=batch_size,
        target_agreement_uuids=scoped_uuids if explicit_scope else None,
        batch_key=batch_key,
        log_prefix=f"{log_prefix} counsel",
    )
    if follow_up_counsel_batch is None:
        return sorted(processed_agreement_uuids)
    follow_up_results = _process_offline_batches(
        context=context,
        engine=engine,
        schema=schema,
        agreements_table=agreements_table,
        pending_batches=[follow_up_counsel_batch],
    )
    _sync_counsel_mappings(
        context=context,
        engine=engine,
        schema=schema,
        agreements_table=agreements_table,
        agreement_uuids=_collect_counsel_sync_uuids(follow_up_results),
        log_prefix=f"{log_prefix} counsel",
    )
    for result in follow_up_results.values():
        raw_processed_uuids = result.get("processed_uuids")
        if isinstance(raw_processed_uuids, list):
            processed_agreement_uuids.update(str(agreement_uuid) for agreement_uuid in raw_processed_uuids if agreement_uuid)
    return sorted(processed_agreement_uuids)


def _collect_counsel_sync_uuids(results: dict[str, dict[str, Any]]) -> list[str]:
    counsel_result = results.get(OFFLINE_COUNSEL_BATCH_KIND)
    if counsel_result is None:
        return []
    processed_uuids = counsel_result.get("processed_uuids")
    if not isinstance(processed_uuids, list):
        return []
    return sorted({str(agreement_uuid) for agreement_uuid in processed_uuids if agreement_uuid})


def _prepare_offline_metadata_batch(
    *,
    context: AssetExecutionContext,
    engine: Any,
    client: "OpenAI",
    schema: str,
    agreements_table: str,
    pages_table: str,
    tagged_outputs_table: str,
    batch_size: int,
    target_agreement_uuids: list[str] | None,
    batch_key: str,
    log_prefix: str,
) -> dict[str, Any] | None:
    existing_batch = _load_existing_offline_batch(
        engine,
        schema=schema,
        batch_kind=OFFLINE_METADATA_BATCH_KIND,
        batch_key=batch_key,
    )
    if existing_batch is not None:
        return {
            "batch_kind": OFFLINE_METADATA_BATCH_KIND,
            "batch_row": existing_batch,
            "update_sql": _metadata_offline_update_sql(agreements_table),
            "parse_response_text": parse_offline_tx_metadata_response_text,
            "build_update_params": build_offline_update_params,
            "log_prefix": log_prefix,
        }

    lines = _build_offline_metadata_lines(
        context=context,
        engine=engine,
        agreements_table=agreements_table,
        pages_table=pages_table,
        tagged_outputs_table=tagged_outputs_table,
        batch_size=batch_size,
        target_agreement_uuids=target_agreement_uuids,
        log_prefix=log_prefix,
    )
    if not lines:
        context.log.info("%s: no runnable agreements need target/acquirer/deal_type.", log_prefix)
        return None
    batch_row = _create_offline_batch(
        context=context,
        engine=engine,
        client=client,
        schema=schema,
        batch_kind=OFFLINE_METADATA_BATCH_KIND,
        lines=lines,
        request_filename="tx_metadata_offline_metadata_requests.jsonl",
        log_prefix=log_prefix,
        batch_key=batch_key,
    )
    return {
        "batch_kind": OFFLINE_METADATA_BATCH_KIND,
        "batch_row": batch_row,
        "update_sql": _metadata_offline_update_sql(agreements_table),
        "parse_response_text": parse_offline_tx_metadata_response_text,
        "build_update_params": build_offline_update_params,
        "log_prefix": log_prefix,
    }


def _prepare_offline_counsel_batch(
    *,
    context: AssetExecutionContext,
    engine: Any,
    client: "OpenAI",
    schema: str,
    agreements_table: str,
    batch_size: int,
    target_agreement_uuids: list[str] | None,
    batch_key: str,
    log_prefix: str,
) -> dict[str, Any] | None:
    existing_batch = _load_existing_offline_batch(
        engine,
        schema=schema,
        batch_kind=OFFLINE_COUNSEL_BATCH_KIND,
        batch_key=batch_key,
    )
    if existing_batch is not None:
        return {
            "batch_kind": OFFLINE_COUNSEL_BATCH_KIND,
            "batch_row": existing_batch,
            "update_sql": _counsel_offline_update_sql(agreements_table),
            "parse_response_text": parse_offline_counsel_response_text,
            "build_update_params": build_offline_counsel_update_params,
            "log_prefix": log_prefix,
        }

    context.log.info("%s: selecting agreements and assembling counsel section text.", log_prefix)
    lines = _build_offline_counsel_lines(
        context=context,
        engine=engine,
        schema=schema,
        agreements_table=agreements_table,
        batch_size=batch_size,
        target_agreement_uuids=target_agreement_uuids,
        log_prefix=log_prefix,
    )
    context.log.info("%s: assembled %s counsel requests.", log_prefix, len(lines))
    if not lines:
        context.log.info("%s: no runnable agreements need counsel extraction.", log_prefix)
        return None
    batch_row = _create_offline_batch(
        context=context,
        engine=engine,
        client=client,
        schema=schema,
        batch_kind=OFFLINE_COUNSEL_BATCH_KIND,
        lines=lines,
        request_filename="tx_metadata_offline_counsel_requests.jsonl",
        log_prefix=log_prefix,
        batch_key=batch_key,
    )
    return {
        "batch_kind": OFFLINE_COUNSEL_BATCH_KIND,
        "batch_row": batch_row,
        "update_sql": _counsel_offline_update_sql(agreements_table),
        "parse_response_text": parse_offline_counsel_response_text,
        "build_update_params": build_offline_counsel_update_params,
        "log_prefix": log_prefix,
    }


def _load_existing_offline_batch(
    engine: Any,
    *,
    schema: str,
    batch_kind: str,
    batch_key: str,
) -> Dict[str, Any] | None:
    with engine.begin() as conn:
        return _fetch_unapplied_offline_batch(
            conn,
            schema,
            batch_kind=batch_kind,
            batch_key=batch_key,
        )


def _resume_and_apply_offline_batch(
    *,
    context: AssetExecutionContext,
    engine: Any,
    client: "OpenAI",
    schema: str,
    agreements_table: str,
    batch_kind: str,
    batch_row: Dict[str, Any],
    update_sql: Any,
    parse_response_text: Any,
    build_update_params: Any,
    log_prefix: str,
) -> dict[str, Any]:
    batch = poll_batch_until_terminal(
        context,
        client,
        str(batch_row["batch_id"]),
        log_prefix=log_prefix,
    )
    with engine.begin() as conn:
        _upsert_offline_batch_row(
            conn,
            schema,
            batch_kind=batch_kind,
            batch=batch,
            completion_window=str(batch_row["completion_window"]),
            request_total=int(batch_row["request_total"]),
            batch_key=str(batch_row["batch_key"]),
        )
    if batch.status != "completed":
        context.log.warning("%s: batch %s ended with status=%s; not applying updates.", log_prefix, batch.id, batch.status)
        with engine.begin() as conn:
            _mark_offline_batch_applied(conn, schema, batch_kind=batch_kind, batch_id=str(batch.id))
        return {
            "batch_id": str(batch.id),
            "batch_kind": batch_kind,
            "status": str(batch.status),
            "updated": 0,
            "parse_errors": 0,
            "refreshed_uuids": [],
            "processed_uuids": [],
        }
    updated, parse_errors, refreshed_uuids, processed_uuids = _apply_offline_batch_output(
        context=context,
        engine=engine,
        client=client,
        schema=schema,
        agreements_table=agreements_table,
        batch=batch,
        update_sql=update_sql,
        parse_response_text=parse_response_text,
        build_update_params=build_update_params,
        log_prefix=log_prefix,
    )
    with engine.begin() as conn:
        _mark_offline_batch_applied(conn, schema, batch_kind=batch_kind, batch_id=str(batch.id))
    context.log.info(
        "%s: resumed batch %s completed; updated=%s, parse_errors=%s, refreshed_latest_sections_search=%s",
        log_prefix,
        batch.id,
        updated,
        parse_errors,
        len(refreshed_uuids),
    )
    return {
        "batch_id": str(batch.id),
        "batch_kind": batch_kind,
        "status": str(batch.status),
        "updated": updated,
        "parse_errors": parse_errors,
        "refreshed_uuids": refreshed_uuids,
        "processed_uuids": processed_uuids,
    }


def _create_offline_batch(
    *,
    context: AssetExecutionContext,
    engine: Any,
    client: "OpenAI",
    schema: str,
    batch_kind: str,
    lines: List[Dict[str, Any]],
    request_filename: str,
    log_prefix: str,
    batch_key: str,
) -> dict[str, Any]:
    jsonl_buf = io.StringIO()
    for line in lines:
        _ = jsonl_buf.write(json.dumps(line, ensure_ascii=False) + "\n")
    jsonl_bytes = io.BytesIO(jsonl_buf.getvalue().encode("utf-8"))
    jsonl_bytes.name = request_filename
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
            batch_kind=batch_kind,
            batch=batch,
            completion_window=completion_window,
            request_total=len(lines),
            batch_key=batch_key,
        )
    context.log.info("%s: created batch %s with %s requests.", log_prefix, batch.id, len(lines))
    return {
        "batch_id": batch.id,
        "completion_window": completion_window,
        "request_total": len(lines),
        "batch_key": batch_key,
    }


def _process_offline_batches(
    *,
    context: AssetExecutionContext,
    engine: Any,
    schema: str,
    agreements_table: str,
    pending_batches: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not pending_batches:
        return {}

    def _run_single_batch(batch_spec: dict[str, Any]) -> dict[str, Any]:
        return _resume_and_apply_offline_batch(
            context=context,
            engine=engine,
            client=_oai_client(),
            schema=schema,
            agreements_table=agreements_table,
            batch_kind=str(batch_spec["batch_kind"]),
            batch_row=batch_spec["batch_row"],
            update_sql=batch_spec["update_sql"],
            parse_response_text=batch_spec["parse_response_text"],
            build_update_params=batch_spec["build_update_params"],
            log_prefix=str(batch_spec["log_prefix"]),
        )

    if len(pending_batches) == 1:
        batch_spec = pending_batches[0]
        return {
            str(batch_spec["batch_kind"]): _run_single_batch(batch_spec),
        }

    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(len(pending_batches), 2)) as executor:
        future_to_kind = {
            executor.submit(_run_single_batch, batch_spec): str(batch_spec["batch_kind"])
            for batch_spec in pending_batches
        }
        for future, batch_kind in future_to_kind.items():
            results[batch_kind] = future.result()
    return results


def _build_offline_metadata_lines(
    *,
    context: AssetExecutionContext,
    engine: Any,
    agreements_table: str,
    pages_table: str,
    tagged_outputs_table: str,
    batch_size: int,
    target_agreement_uuids: list[str] | None = None,
    log_prefix: str = "tx_metadata_asset (offline metadata)",
) -> List[Dict[str, Any]]:
    if target_agreement_uuids is not None and not target_agreement_uuids:
        return []
    query_limit = max(batch_size, len(set(target_agreement_uuids))) if target_agreement_uuids else batch_size
    scope_clause = "AND a.agreement_uuid IN :agreement_uuids" if target_agreement_uuids else ""
    pages_q = text(
        f"""
        WITH candidate_pages AS (
            SELECT
                a.agreement_uuid,
                a.filing_date,
                CASE
                    WHEN (a.target IS NULL OR TRIM(a.target) = '')
                     AND (a.acquirer IS NULL OR TRIM(a.acquirer) = '')
                     AND (a.deal_type IS NULL OR TRIM(a.deal_type) = '')
                    THEN 1
                    ELSE 0
                END AS missing_all_metadata_fields,
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
                {scope_clause}
                AND coalesce(p.gold_label, p.source_page_type) IN ('front_matter', 'body')
        ),
        selected_pages AS (
            SELECT agreement_uuid, filing_date, missing_all_metadata_fields, page_order, page_text
            FROM candidate_pages
            WHERE source_page_type = 'front_matter'
               OR (source_page_type = 'body' AND rn <= 2)
        ),
        selected_agreements AS (
            SELECT
                agreement_uuid,
                MAX(missing_all_metadata_fields) AS missing_all_metadata_fields,
                MAX(filing_date) AS filing_date
            FROM selected_pages
            GROUP BY agreement_uuid
            HAVING SUM(
                CASE
                    WHEN TRIM(COALESCE(page_text, '')) <> '' THEN 1
                    ELSE 0
                END
            ) > 0
            ORDER BY
                missing_all_metadata_fields DESC,
                (filing_date IS NULL) ASC,
                filing_date ASC,
                agreement_uuid ASC
            LIMIT :lim
        )
        SELECT sp.agreement_uuid, sp.page_order, sp.page_text
        FROM selected_pages sp
        JOIN selected_agreements sa
            ON sa.agreement_uuid = sp.agreement_uuid
        ORDER BY
            sa.missing_all_metadata_fields DESC,
            (sa.filing_date IS NULL) ASC,
            sa.filing_date ASC,
            sp.agreement_uuid ASC,
            sp.page_order ASC
        """
    )
    if target_agreement_uuids:
        pages_q = pages_q.bindparams(bindparam("agreement_uuids", expanding=True))
    with engine.begin() as conn:
        params: dict[str, object] = {"lim": query_limit}
        if target_agreement_uuids:
            params["agreement_uuids"] = tuple(sorted(set(target_agreement_uuids)))
        page_rows = conn.execute(pages_q, params).mappings().fetchall()
    by_agr: Dict[str, List[str]] = {}
    for row in page_rows:
        agreement_uuid = str(row["agreement_uuid"])
        page_text = str(row["page_text"] or "")
        by_agr.setdefault(agreement_uuid, []).append(page_text)
    lines: List[Dict[str, Any]] = []
    for agreement_uuid, texts in by_agr.items():
        concat = "\n\n".join(texts).strip()
        if not concat:
            context.log.warning("%s: no page text for %s; skipping.", log_prefix, agreement_uuid)
            continue
        lines.append(build_offline_tx_metadata_request_body(agreement_uuid, concat, model="gpt-5.4-mini"))
    return lines


def _build_offline_counsel_lines(
    *,
    context: AssetExecutionContext,
    engine: Any,
    schema: str,
    agreements_table: str,
    batch_size: int,
    target_agreement_uuids: list[str] | None = None,
    log_prefix: str = "tx_metadata_asset (offline counsel)",
) -> List[Dict[str, Any]]:
    if target_agreement_uuids is not None and not target_agreement_uuids:
        return []
    query_limit = max(batch_size, len(set(target_agreement_uuids))) if target_agreement_uuids else batch_size
    pages_table = f"{schema}.pages"
    tagged_outputs_table = f"{schema}.tagged_outputs"
    standard_ids_table = f"{schema}.latest_sections_search_standard_ids"
    sections_table = f"{schema}.sections"
    xml_table = f"{schema}.xml"
    scope_clause = "AND a.agreement_uuid IN :agreement_uuids" if target_agreement_uuids else ""
    candidate_q = text(
        f"""
        SELECT
            a.agreement_uuid,
            a.target,
            a.acquirer,
            a.filing_date,
            CASE
                WHEN (a.target_counsel IS NULL OR TRIM(a.target_counsel) = '')
                 AND (a.acquirer_counsel IS NULL OR TRIM(a.acquirer_counsel) = '')
                THEN 1
                ELSE 0
            END AS missing_all_counsel_fields
        FROM {agreements_table} a
        JOIN {xml_table} x
          ON x.agreement_uuid = a.agreement_uuid
         AND x.latest = 1
         AND x.status = 'verified'
        WHERE (a.target_counsel IS NULL OR a.acquirer_counsel IS NULL)
          {scope_clause}
          AND a.target IS NOT NULL
          AND TRIM(a.target) <> ''
          AND a.acquirer IS NOT NULL
          AND TRIM(a.acquirer) <> ''
          AND EXISTS (
                SELECT 1
                FROM {standard_ids_table} lssi
                WHERE lssi.agreement_uuid = a.agreement_uuid
                  AND lssi.standard_id = :counsel_standard_id
          )
          AND NOT EXISTS (
                SELECT 1
                FROM {pages_table} p
                JOIN {tagged_outputs_table} t
                  ON t.page_uuid = p.page_uuid
                WHERE p.agreement_uuid = a.agreement_uuid
                  AND COALESCE(p.gold_label, p.source_page_type) = 'body'
                  AND x.created_date IS NOT NULL
                  AND t.updated_date > x.created_date
          )
        ORDER BY
            missing_all_counsel_fields DESC,
            (a.filing_date IS NULL) ASC,
            a.filing_date ASC,
            a.agreement_uuid ASC
        LIMIT :lim
        """
    )
    if target_agreement_uuids:
        candidate_q = candidate_q.bindparams(bindparam("agreement_uuids", expanding=True))
    section_q = (
        text(
            f"""
            SELECT
                lssi.agreement_uuid,
                lssi.section_uuid,
                s.xml_content
            FROM {standard_ids_table} lssi
            JOIN {sections_table} s
              ON s.section_uuid = lssi.section_uuid
             AND s.agreement_uuid = lssi.agreement_uuid
            JOIN {xml_table} x
              ON x.agreement_uuid = s.agreement_uuid
             AND x.version = s.xml_version
             AND x.latest = 1
             AND x.status = 'verified'
            WHERE lssi.standard_id = :counsel_standard_id
              AND lssi.agreement_uuid IN :agreement_uuids
            ORDER BY
                lssi.agreement_uuid ASC,
                lssi.section_uuid ASC
            """
        ).bindparams(bindparam("agreement_uuids", expanding=True))
    )
    with engine.begin() as conn:
        candidate_rows = conn.execute(
            candidate_q,
            {
                "counsel_standard_id": COUNSEL_SECTION_STANDARD_ID,
                "lim": query_limit,
                **(
                    {"agreement_uuids": tuple(sorted(set(target_agreement_uuids)))}
                    if target_agreement_uuids
                    else {}
                ),
            },
        ).mappings().fetchall()
        agreement_order = [str(row["agreement_uuid"]) for row in candidate_rows]
        if not agreement_order:
            return []
        rows = conn.execute(
            section_q,
            {
                "counsel_standard_id": COUNSEL_SECTION_STANDARD_ID,
                "agreement_uuids": agreement_order,
            },
        ).mappings().fetchall()
    grouped_texts: Dict[str, List[str]] = {}
    target_by_agreement = {
        str(row["agreement_uuid"]): str(row["target"])
        for row in candidate_rows
    }
    acquirer_by_agreement = {
        str(row["agreement_uuid"]): str(row["acquirer"])
        for row in candidate_rows
    }
    for row in rows:
        agreement_uuid = str(row["agreement_uuid"])
        grouped_texts.setdefault(agreement_uuid, []).append(str(row["xml_content"] or ""))
    lines: List[Dict[str, Any]] = []
    for agreement_uuid in agreement_order:
        section_texts = grouped_texts.get(agreement_uuid, [])
        joined_text = "\n\n".join(section_texts).strip()
        if not joined_text:
            context.log.warning("%s: no counsel section text for %s; skipping.", log_prefix, agreement_uuid)
            continue
        lines.append(
            build_offline_counsel_request_body(
                agreement_uuid,
                section_text=joined_text,
                target_name=target_by_agreement[agreement_uuid],
                acquirer_name=acquirer_by_agreement[agreement_uuid],
                model="gpt-5.4-mini",
            )
        )
    return lines


def _run_web_search_mode(
    context: AssetExecutionContext,
    engine: Any,
    schema: str,
    agreements_table: str,
    batch_size: int,
    target_agreement_uuids: list[str] | None = None,
    include_all_scoped_agreements: bool = False,
    force_include_agreement_uuids: list[str] | None = None,
    log_prefix: str = "tx_metadata_asset (web_search)",
) -> Dict[str, Any]:
    """Web-search: select agreements needing metadata with names or URL context; sync API; update web columns."""
    if target_agreement_uuids is not None and not target_agreement_uuids:
        context.log.info("%s: explicit empty scope; no web-search work to run.", log_prefix)
        return {"total_searches": 0, "searches_by_agreement": {}, "processed_uuids": []}
    forced_uuids = tuple(sorted({str(agreement_uuid) for agreement_uuid in (force_include_agreement_uuids or []) if agreement_uuid}))
    scoped_uuids = tuple(sorted({str(agreement_uuid) for agreement_uuid in (target_agreement_uuids or []) if agreement_uuid}))
    minimum_query_size = len(set(scoped_uuids) | set(forced_uuids))
    query_limit = max(batch_size, minimum_query_size) if minimum_query_size else batch_size
    with engine.begin() as conn:
        assert_tables_exist(
            conn,
            schema=schema,
            table_names=("tx_metadata_web_failures",),
        )

    scope_clause = "AND a.agreement_uuid IN :agreement_uuids" if scoped_uuids else ""
    apply_default_candidate_filter = not (include_all_scoped_agreements and target_agreement_uuids)
    forced_candidate_clause = " OR a.agreement_uuid IN :force_include_agreement_uuids" if forced_uuids else ""
    candidate_filter_clause = (
        f"""
        (
            COALESCE(a.metadata, 0) = 0
            OR (
                COALESCE(a.metadata, 0) = 1
                AND (
                    {_web_search_missing_core_metadata_sql(alias='a')}
                )
            )
            {forced_candidate_clause}
        )
        """
        if apply_default_candidate_filter
        else f"1 = 1{' OR a.agreement_uuid IN :force_include_agreement_uuids' if forced_uuids else ''}"
    )
    select_q = text(
        f"""
        SELECT
            a.agreement_uuid,
            a.target,
            a.acquirer,
            a.filing_date,
            a.url,
            a.transaction_consideration,
            a.transaction_price_cash,
            a.transaction_price_stock,
            a.transaction_price_assets,
            a.transaction_price_total,
            a.target_type,
            a.acquirer_type,
            a.target_pe,
            a.acquirer_pe,
            a.target_industry,
            a.acquirer_industry,
            a.announce_date,
            a.close_date,
            a.deal_status,
            a.attitude,
            a.purpose,
            a.metadata_sources,
            a.metadata_uncited_fields,
            COALESCE(wf.failure_count, 0) AS failure_count,
            CASE WHEN COALESCE(a.metadata, 0) = 0 THEN 1 ELSE 0 END AS initial_metadata_pass
        FROM {agreements_table} a
        LEFT JOIN {schema}.tx_metadata_web_failures wf
          ON wf.agreement_uuid = a.agreement_uuid
        WHERE {candidate_filter_clause}
          AND (
            (
              a.target IS NOT NULL AND TRIM(a.target) <> ''
              AND a.acquirer IS NOT NULL AND TRIM(a.acquirer) <> ''
            )
            OR (a.url IS NOT NULL AND TRIM(a.url) <> '')
          )
          {scope_clause}
        ORDER BY
            initial_metadata_pass DESC,
            COALESCE(wf.failure_count, 0) ASC,
            (filing_date IS NULL) ASC,
            filing_date ASC,
            agreement_uuid ASC
        LIMIT :lim
        """
    )
    if scoped_uuids:
        select_q = select_q.bindparams(bindparam("agreement_uuids", expanding=True))
    if forced_uuids:
        select_q = select_q.bindparams(bindparam("force_include_agreement_uuids", expanding=True))
    with engine.begin() as conn:
        params: dict[str, object] = {"lim": query_limit}
        if scoped_uuids:
            params["agreement_uuids"] = scoped_uuids
        if forced_uuids:
            params["force_include_agreement_uuids"] = forced_uuids
        rows = conn.execute(select_q, params).mappings().fetchall()
    agreements = [dict(r) for r in rows]
    if not agreements:
        context.log.info("%s: no agreements need web metadata.", log_prefix)
        return {"total_searches": 0, "searches_by_agreement": {}, "processed_uuids": []}
    processed_uuids = sorted({str(agreement["agreement_uuid"]) for agreement in agreements if agreement.get("agreement_uuid")})

    context.log.info(
        "%s: selected %s agreements for enrichment (max_workers=%s, commit_batch_size=%s)",
        log_prefix,
        len(agreements),
        WEB_SEARCH_MAX_WORKERS,
        WEB_SEARCH_COMMIT_BATCH_SIZE,
    )
    client = _oai_client()

    class _WebSearchRequestError(RuntimeError):
        def __init__(self, message: str, *, raw_payload: str | None = None) -> None:
            super().__init__(message)
            self.raw_payload = raw_payload

    def _request_web_search_payload(
        agreement_row: Dict[str, Any],
        *,
        model_name: str,
        max_attempts: int = 3,
    ) -> tuple[Dict[str, Any], str, Dict[str, int], int]:
        agr_uuid_local = str(agreement_row["agreement_uuid"])
        try:
            body = build_tx_metadata_request_body_web_search_only(
                agreement_row,
                model=model_name,
                retry_context=build_web_search_retry_context(agreement_row),
            )
        except Exception as exc:
            raise _WebSearchRequestError(
                f"failed to build web-search request: {exc}",
                raw_payload=None,
            ) from exc
        last_error: Exception | None = None
        last_raw_payload: str | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = client.responses.create(**body)  # type: ignore[arg-type]
                raw_text = getattr(resp, "output_text", None) or ""
                if not isinstance(raw_text, str):
                    raise TypeError("output_text is not a string.")
                usage = _extract_response_usage(resp)
                search_count = _extract_web_search_count(resp)
                last_raw_payload = raw_text
                return (
                    parse_tx_metadata_response_text_web_search(raw_text),
                    raw_text,
                    usage,
                    search_count,
                )
            except Exception as exc:
                last_error = exc
                if attempt == max_attempts:
                    break
                sleep_seconds = 2 ** (attempt - 1)
                context.log.warning(
                    "%s: attempt %s/%s failed for %s: %s; retrying in %ss",
                    log_prefix,
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

    attempted = 0
    parsed = 0
    parse_errors = 0
    failed_uuid_by_stage: Dict[str, List[str]] = {"request_or_parse": [], "validation": []}
    token_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    total_searches = 0
    searches_by_agreement: Dict[str, int] = {}
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
    agreement_groups = [
        (
            model_name,
            [agreement for agreement in agreements if _web_search_model_for_agreement(agreement) == model_name],
        )
        for model_name in ("gpt-5.4-mini", "gpt-5.4")
    ]
    agreement_groups = [(model_name, group) for model_name, group in agreement_groups if group]
    total_chunks = sum(
        len(_chunk_agreements(group, chunk_size=WEB_SEARCH_COMMIT_BATCH_SIZE))
        for _, group in agreement_groups
    )
    completed_chunk_count = 0
    context.log.info(
        "tx_metadata_asset (web_search): selected by model gpt-5.4-mini=%s gpt-5.4=%s",
        sum(1 for agreement in agreements if _web_search_model_for_agreement(agreement) == "gpt-5.4-mini"),
        sum(1 for agreement in agreements if _web_search_model_for_agreement(agreement) == "gpt-5.4"),
    )

    for model_name, queued_agreements in agreement_groups:
        group_start_chunk = completed_chunk_count
        next_agreement_index = 0
        completed_success_buffer: list[tuple[Dict[str, Any], Dict[str, Any], Any, str, Dict[str, int], int]] = []
        active_futures: Dict[Future[tuple[Dict[str, Any], str, Dict[str, int], int]], Dict[str, Any]] = {}

        def _submit_available_work(executor: ThreadPoolExecutor) -> None:
            nonlocal next_agreement_index
            while (
                next_agreement_index < len(queued_agreements)
                and len(active_futures) < WEB_SEARCH_MAX_WORKERS
            ):
                agreement = queued_agreements[next_agreement_index]
                next_agreement_index += 1
                if (next_agreement_index - 1) % WEB_SEARCH_COMMIT_BATCH_SIZE == 0:
                    chunk_offset = (next_agreement_index - 1) // WEB_SEARCH_COMMIT_BATCH_SIZE
                    chunk_number = group_start_chunk + chunk_offset + 1
                    chunk_size = min(
                        WEB_SEARCH_COMMIT_BATCH_SIZE,
                        len(queued_agreements) - (next_agreement_index - 1),
                    )
                    context.log.info(
                        "tx_metadata_asset (web_search): starting chunk %s/%s with %s agreements using model %s",
                        chunk_number,
                        total_chunks,
                        chunk_size,
                        model_name,
                    )
                future = executor.submit(
                    _request_web_search_payload,
                    agreement,
                    model_name=model_name,
                )
                active_futures[future] = agreement

        with ThreadPoolExecutor(max_workers=WEB_SEARCH_MAX_WORKERS) as executor:
            _submit_available_work(executor)
            while active_futures:
                done, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    agreement = active_futures.pop(future)
                    attempted += 1
                    agr_uuid = str(agreement["agreement_uuid"])
                    try:
                        obj, raw_payload, response_usage, search_count = future.result()
                        completed_success_buffer.append(
                            (
                                agreement,
                                obj,
                                agreement.get("filing_date"),
                                raw_payload,
                                response_usage,
                                search_count,
                            )
                        )
                        parsed += 1
                        searches_by_agreement[agr_uuid] = search_count
                        total_searches += search_count
                    except _WebSearchRequestError as e:
                        parse_errors += 1
                        failed_uuid_by_stage["request_or_parse"].append(agr_uuid)
                        context.log.warning(f"tx_metadata_asset (web_search): failed for {agr_uuid}: {e}")
                        with engine.begin() as conn:
                            _record_web_failure(
                                conn,
                                schema,
                                agreement_uuid=agr_uuid,
                                error_stage="request_or_parse",
                                failure_reason=str(e),
                                raw_payload=e.raw_payload,
                            )

                while len(completed_success_buffer) >= WEB_SEARCH_COMMIT_BATCH_SIZE:
                    batch_successes = completed_success_buffer[:WEB_SEARCH_COMMIT_BATCH_SIZE]
                    del completed_success_buffer[:WEB_SEARCH_COMMIT_BATCH_SIZE]
                    chunk_number = completed_chunk_count + 1
                    batch_updated, batch_validation_failures, batch_refreshed = _persist_web_search_successes(
                        context=context,
                        engine=engine,
                        schema=schema,
                        update_web_q=update_web_q,
                        successes=batch_successes,
                        failed_uuid_by_stage=failed_uuid_by_stage,
                        token_totals=token_totals,
                        refreshed_uuids=refreshed_uuids,
                    )
                    updated += batch_updated
                    skipped_due_to_error += batch_validation_failures
                    completed_chunk_count += 1
                    context.log.info(
                        "tx_metadata_asset (web_search): finished chunk %s/%s attempted=%s request_failures=%s validation_failures=%s updated=%s refreshed_latest_sections_search=%s model=%s",
                        chunk_number,
                        total_chunks,
                        len(batch_successes),
                        0,
                        batch_validation_failures,
                        batch_updated,
                        batch_refreshed,
                        model_name,
                    )

                _submit_available_work(executor)

        if completed_success_buffer:
            chunk_number = completed_chunk_count + 1
            batch_updated, batch_validation_failures, batch_refreshed = _persist_web_search_successes(
                context=context,
                engine=engine,
                schema=schema,
                update_web_q=update_web_q,
                successes=completed_success_buffer,
                failed_uuid_by_stage=failed_uuid_by_stage,
                token_totals=token_totals,
                refreshed_uuids=refreshed_uuids,
            )
            updated += batch_updated
            skipped_due_to_error += batch_validation_failures
            completed_chunk_count += 1
            context.log.info(
                "tx_metadata_asset (web_search): finished chunk %s/%s attempted=%s request_failures=%s validation_failures=%s updated=%s refreshed_latest_sections_search=%s model=%s",
                chunk_number,
                total_chunks,
                len(completed_success_buffer),
                0,
                batch_validation_failures,
                batch_updated,
                batch_refreshed,
                model_name,
            )

    if failed_uuid_by_stage["request_or_parse"] or failed_uuid_by_stage["validation"]:
        context.log.warning(
            "tx_metadata_asset (web_search): failed agreements by stage request_or_parse=%s validation=%s",
            failed_uuid_by_stage["request_or_parse"],
            failed_uuid_by_stage["validation"],
        )

    context.log.info(
        "tx_metadata_asset (web_search): attempted=%s, parsed=%s, updated=%s, parse_errors=%s, skipped_due_to_error=%s, refreshed_latest_sections_search=%s, request_or_parse_failures=%s, validation_failures=%s, total_searches=%s, token_totals=%s",
        attempted,
        parsed,
        updated,
        parse_errors,
        skipped_due_to_error,
        len(refreshed_uuids),
        len(failed_uuid_by_stage["request_or_parse"]),
        len(failed_uuid_by_stage["validation"]),
        total_searches,
        token_totals,
    )
    return {
        "total_searches": total_searches,
        "searches_by_agreement": searches_by_agreement,
        "processed_uuids": processed_uuids,
    }
