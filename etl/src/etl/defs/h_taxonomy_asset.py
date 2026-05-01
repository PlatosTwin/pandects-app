# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false

import io
import json
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection

from etl.defs.g_sections_asset import sections_asset
from etl.defs.g_sections_asset import (
    ingestion_cleanup_a_sections_from_fresh_xml_asset,
    ingestion_cleanup_a_sections_from_repair_xml_asset,
    ingestion_cleanup_b_sections_from_repair_xml_asset,
    ingestion_cleanup_c_sections_asset,
    ingestion_cleanup_d_sections_asset,
    regular_ingest_sections_from_fresh_xml_asset,
    regular_ingest_sections_from_repair_xml_asset,
)
from etl.defs.resources import DBResource, PipelineConfig, TaxonomyModel, TaxonomyMode
from etl.domain.f_xml import XMLData
from etl.domain.h_taxonomy import (
    ContextProtocol as TaxonomyContext,
    TaxonomyLLMRow,
    TaxonomyPredictor,
    TaxonomyRow,
    apply_standard_ids_to_xml,
    build_taxonomy_llm_request_body,
    build_taxonomy_prompt_payload,
    parse_taxonomy_llm_response_text,
    predict_taxonomy,
    serialize_taxonomy_labels,
)
from etl.utils.batch_keys import agreement_batch_key
from etl.utils.db_utils import upsert_xml
from etl.utils.latest_sections_search import refresh_latest_sections_search
from etl.utils.logical_job_runs import (
    build_logical_batch_key,
    load_active_logical_run,
    load_active_scope_for_job,
    mark_logical_run_stage_completed,
    should_skip_managed_stage,
)
from etl.utils.openai_batch import (
    extract_output_text_from_batch_body,
    poll_batch_until_terminal,
    read_openai_file_text,
)
from etl.utils.post_asset_refresh import run_post_asset_refresh
from etl.utils.run_config import runs_single_batch
from etl.utils.schema_guards import assert_tables_exist

if TYPE_CHECKING:
    from openai import OpenAI


TAXONOMY_LLM_BATCHES_TABLE = "taxonomy_llm_batches"
TAXONOMY_LLM_REQUEST_FILENAME = "taxonomy_llm_requests.jsonl"
DEFAULT_TAXONOMY_LLM_SECTIONS_PER_REQUEST = 5
TAXONOMY_LLM_COMPLETION_WINDOW = "24h"


def _oai_client() -> "OpenAI":
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for taxonomy_asset llm mode.")
    return OpenAI(api_key=api_key)


def _normalized_payload(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None
    cleaned = raw_value.strip()
    if cleaned == "":
        return None
    return cleaned


def _normalized_nonempty_payload(raw_value: str | None) -> str | None:
    cleaned = _normalized_payload(raw_value)
    if cleaned in {None, "[]"}:
        return None
    return cleaned


def _prediction_missing_clause(column_name: str) -> str:
    return f"({column_name} IS NULL OR TRIM({column_name}) = '')"


def _prediction_clause_for_mode(mode: TaxonomyMode) -> str:
    if mode == TaxonomyMode.LLM:
        return _prediction_missing_clause("s.section_standard_id_gold_label")
    if mode == TaxonomyMode.ML:
        return (
            _prediction_missing_clause("s.section_standard_id_gold_label")
            + " AND "
            + _prediction_missing_clause("s.section_standard_id")
        )
    return (
        "s.section_standard_id_gold_label IS NOT NULL "
        "AND TRIM(s.section_standard_id_gold_label) <> '' "
        "AND TRIM(s.section_standard_id_gold_label) <> '[]'"
    )


def _taxonomy_regex_clause(regex: str | None) -> str:
    if not regex:
        return ""
    return " AND s.section_title_normed REGEXP :section_title_regex"


def _llm_title_eligibility_clause() -> str:
    return (
        "CHAR_LENGTH(TRIM(COALESCE(s.article_title, ''))) >= 3 "
        "AND CHAR_LENGTH(TRIM(COALESCE(s.section_title, ''))) >= 3 "
        "AND LOWER(COALESCE(s.section_title, '')) NOT LIKE '%[reserved]%' "
        "AND LOWER(COALESCE(s.section_title, '')) NOT LIKE '%[omitted]%' "
        "AND LOWER(COALESCE(s.section_title, '')) NOT LIKE '%[intentionally deleted]%' "
        "AND LOWER(COALESCE(s.section_title, '')) NOT LIKE '%[deleted]%'"
    )


def _fetch_taxonomy_json(conn: Connection, schema: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        text(
            f"""
            SELECT
                l1.standard_id AS l1_standard_id,
                l1.label AS l1_label,
                l2.standard_id AS l2_standard_id,
                l2.label AS l2_label,
                l3.standard_id AS l3_standard_id,
                l3.label AS l3_label
            FROM {schema}.taxonomy_l1 l1
            LEFT JOIN {schema}.taxonomy_l2 l2
                ON l1.standard_id = l2.parent_id
            LEFT JOIN {schema}.taxonomy_l3 l3
                ON l2.standard_id = l3.parent_id
            ORDER BY l1.standard_id, l2.standard_id, l3.standard_id
            """
        )
    ).mappings()
    return [dict(row) for row in rows]


def _fetch_unapplied_taxonomy_llm_batch(
    conn: Connection,
    schema: str,
    *,
    batch_key: str | None = None,
) -> dict[str, Any] | None:
    params: dict[str, object] = {}
    batch_key_clause = ""
    if batch_key is not None:
        batch_key_clause = "AND batch_key = :batch_key"
        params["batch_key"] = batch_key
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
                model_name,
                batch_key
            FROM {schema}.{TAXONOMY_LLM_BATCHES_TABLE}
            WHERE applied = 0
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


def _upsert_taxonomy_llm_batch_row(
    conn: Connection,
    schema: str,
    *,
    batch: Any,
    completion_window: str,
    request_total: int,
    model_name: str,
    batch_key: str,
) -> None:
    _ = conn.execute(
        text(
            f"""
            INSERT INTO {schema}.{TAXONOMY_LLM_BATCHES_TABLE} (
                batch_id,
                created_at,
                status,
                input_file_id,
                output_file_id,
                error_file_id,
                completion_window,
                request_total,
                model_name,
                batch_key,
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
                :model_name,
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
                model_name = VALUES(model_name),
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
            "model_name": model_name,
            "batch_key": batch_key,
        },
    )


def _mark_taxonomy_llm_batch_applied(
    conn: Connection,
    schema: str,
    *,
    batch_id: str,
) -> None:
    _ = conn.execute(
        text(
            f"""
            UPDATE {schema}.{TAXONOMY_LLM_BATCHES_TABLE}
            SET applied = 1, applied_at = UTC_TIMESTAMP()
            WHERE batch_id = :batch_id
            """
        ),
        {"batch_id": batch_id},
    )


def _select_agreement_batch(
    conn: Connection,
    *,
    sections_table: str,
    xml_table: str,
    mode: TaxonomyMode,
    last_uuid: str,
    agreement_batch_size: int,
    section_title_regex: str | None,
    scoped_uuids: list[str] | None = None,
) -> list[str]:
    params: dict[str, object] = {"last": last_uuid, "lim": agreement_batch_size}
    regex_clause = ""
    scope_clause = ""
    if mode in {TaxonomyMode.LLM, TaxonomyMode.ML} and section_title_regex:
        regex_clause = _taxonomy_regex_clause(section_title_regex)
        params["section_title_regex"] = section_title_regex
    llm_title_clause = ""
    if mode == TaxonomyMode.LLM:
        llm_title_clause = f"AND {_llm_title_eligibility_clause()}"
    if scoped_uuids:
        scope_clause = "AND s.agreement_uuid IN :agreement_uuids"
        params["agreement_uuids"] = tuple(scoped_uuids)
    query = text(
        f"""
        SELECT DISTINCT s.agreement_uuid
        FROM {sections_table} s
        JOIN {xml_table} x
          ON x.agreement_uuid = s.agreement_uuid
         AND x.version = s.xml_version
        WHERE s.agreement_uuid > :last
          {scope_clause}
          AND x.latest = 1
          AND x.status = 'verified'
          AND {_prediction_clause_for_mode(mode)}
          {llm_title_clause}
          {regex_clause}
        ORDER BY s.agreement_uuid
        LIMIT :lim
        """
    )
    if scoped_uuids:
        query = query.bindparams(bindparam("agreement_uuids", expanding=True))
    rows = conn.execute(
        query,
        params,
    ).mappings().fetchall()
    return [str(row["agreement_uuid"]) for row in rows]


def _fetch_prediction_rows(
    conn: Connection,
    *,
    sections_table: str,
    xml_table: str,
    agreement_uuids: list[str],
    mode: TaxonomyMode,
    section_title_regex: str | None,
) -> list[dict[str, Any]]:
    if not agreement_uuids:
        return []
    params: dict[str, object] = {"agreements": tuple(agreement_uuids)}
    regex_clause = ""
    if section_title_regex:
        regex_clause = _taxonomy_regex_clause(section_title_regex)
        params["section_title_regex"] = section_title_regex
    llm_title_clause = ""
    if mode == TaxonomyMode.LLM:
        llm_title_clause = f"AND {_llm_title_eligibility_clause()}"
    query = text(
        f"""
        SELECT
            s.section_uuid,
            s.agreement_uuid,
            s.article_title,
            s.section_title,
            s.article_title_normed,
            s.section_title_normed,
            s.article_order,
            s.section_order,
            s.xml_content,
            s.section_standard_id_gold_label
        FROM {sections_table} s
        JOIN {xml_table} x
          ON x.agreement_uuid = s.agreement_uuid
         AND x.version = s.xml_version
        WHERE s.agreement_uuid IN :agreements
          AND x.latest = 1
          AND x.status = 'verified'
          AND {_prediction_clause_for_mode(mode)}
          {llm_title_clause}
          {regex_clause}
        ORDER BY
            s.agreement_uuid,
            COALESCE(s.article_order, -1),
            COALESCE(s.section_order, -1),
            s.section_uuid
        """
    ).bindparams(bindparam("agreements", expanding=True))
    rows = conn.execute(query, params).mappings().fetchall()
    return [dict(row) for row in rows]


def _fetch_gold_rows(
    conn: Connection,
    *,
    sections_table: str,
    xml_table: str,
    agreement_uuids: list[str],
) -> list[dict[str, Any]]:
    if not agreement_uuids:
        return []
    rows = conn.execute(
        text(
            f"""
            SELECT
                s.section_uuid,
                s.agreement_uuid,
                s.section_standard_id_gold_label
            FROM {sections_table} s
            JOIN {xml_table} x
              ON x.agreement_uuid = s.agreement_uuid
             AND x.version = s.xml_version
            WHERE s.agreement_uuid IN :agreements
              AND x.latest = 1
              AND x.status = 'verified'
              AND {_prediction_clause_for_mode(TaxonomyMode.GOLD_BACKFILL)}
            ORDER BY
                s.agreement_uuid,
                COALESCE(s.article_order, -1),
                COALESCE(s.section_order, -1),
                s.section_uuid
            """
        ).bindparams(bindparam("agreements", expanding=True)),
        {"agreements": tuple(agreement_uuids)},
    ).mappings().fetchall()
    return [dict(row) for row in rows]


def _build_llm_rows(prediction_rows: list[dict[str, Any]]) -> list[TaxonomyLLMRow]:
    built_rows: list[TaxonomyLLMRow] = []
    by_agreement: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        by_agreement[str(row["agreement_uuid"])].append(row)

    for agreement_uuid in sorted(by_agreement):
        rows = by_agreement[agreement_uuid]
        for idx, row in enumerate(rows):
            prev_row = rows[idx - 1] if idx > 0 else None
            next_row = rows[idx + 1] if idx + 1 < len(rows) else None
            built_rows.append(
                {
                    "section_uuid": str(row["section_uuid"]),
                    "agreement_uuid": agreement_uuid,
                    "article_title": cast(str | None, row.get("article_title")),
                    "section_title": cast(str | None, row.get("section_title")),
                    "prev_article_title": cast(
                        str | None,
                        None if prev_row is None else prev_row.get("article_title"),
                    ),
                    "prev_section_title": cast(
                        str | None,
                        None if prev_row is None else prev_row.get("section_title"),
                    ),
                    "next_article_title": cast(
                        str | None,
                        None if next_row is None else next_row.get("article_title"),
                    ),
                    "next_section_title": cast(
                        str | None,
                        None if next_row is None else next_row.get("section_title"),
                    ),
                }
            )
    return built_rows


def _apply_xml_updates_for_agreements(
    conn: Connection,
    *,
    db: DBResource,
    agreement_uuids: list[str],
    section_mapping_by_agreement: dict[str, dict[str, str]],
) -> int:
    if not agreement_uuids:
        return 0
    xml_rows = conn.execute(
        text(
            f"""
            SELECT m.agreement_uuid, m.xml, m.version
            FROM {db.database}.xml m
            WHERE m.agreement_uuid IN :agreements
              AND m.latest = 1
              AND m.status = 'verified'
            """
        ).bindparams(bindparam("agreements", expanding=True)),
        {"agreements": tuple(agreement_uuids)},
    ).mappings().fetchall()

    staged_xml: list[XMLData] = []
    for row in xml_rows:
        agreement_uuid = str(row["agreement_uuid"])
        mapping = section_mapping_by_agreement.get(agreement_uuid, {})
        if not mapping:
            continue
        new_xml = apply_standard_ids_to_xml(str(row["xml"]), mapping)
        staged_xml.append(
            XMLData(
                agreement_uuid=agreement_uuid,
                xml=new_xml,
                version=int(row.get("version", 1) or 1),
            )
        )
    if staged_xml:
        upsert_xml(staged_xml, db.database, conn)
    return len(staged_xml)


def _apply_ml_predictions(
    context: AssetExecutionContext,
    conn: Connection,
    *,
    db: DBResource,
    sections_table: str,
    prediction_rows: list[dict[str, Any]],
    model: TaxonomyPredictor,
) -> tuple[int, int, list[str]]:
    rows: list[TaxonomyRow] = [cast(TaxonomyRow, cast(object, row)) for row in prediction_rows]
    sec_idx, preds = predict_taxonomy(
        rows,
        model,
        cast(TaxonomyContext, cast(object, context)),
    )
    updates: list[dict[str, object]] = []
    mapping_by_agreement: dict[str, dict[str, str]] = defaultdict(dict)
    for meta, pred in zip(sec_idx, preds):
        inferred_label = cast(str, cast(object, pred.get("label")))
        payload = serialize_taxonomy_labels([inferred_label])
        alt_probs = pred.get("alt_probs") or [0.0, 0.0, 0.0]
        updates.append(
            {
                "section_uuid": meta["section_uuid"],
                "label": payload,
                "a": float(alt_probs[0]) if len(alt_probs) > 0 else 0.0,
                "b": float(alt_probs[1]) if len(alt_probs) > 1 else 0.0,
                "c": float(alt_probs[2]) if len(alt_probs) > 2 else 0.0,
            }
        )
        mapping_by_agreement[meta["agreement_uuid"]][meta["section_uuid"]] = payload

    updated_rows = 0
    if updates:
        update_sql = text(
            f"""
            UPDATE {sections_table}
            SET section_standard_id = :label,
                alt_label_a_prob = :a,
                alt_label_b_prob = :b,
                alt_label_c_prob = :c
            WHERE section_uuid = :section_uuid
            """
        )
        for start in range(0, len(updates), 250):
            result = conn.execute(update_sql, updates[start : start + 250])
            updated_rows += int(result.rowcount or 0)

    agreement_uuids = sorted(mapping_by_agreement)
    xml_updated = _apply_xml_updates_for_agreements(
        conn,
        db=db,
        agreement_uuids=agreement_uuids,
        section_mapping_by_agreement=mapping_by_agreement,
    )
    refreshed = refresh_latest_sections_search(conn, db.database, agreement_uuids)
    context.log.info(
        (
            "taxonomy_asset (ml): updated %s sections across %s agreements; "
            "upserted %s XMLs; refreshed latest_sections_search rows=%s"
        ),
        updated_rows,
        len(agreement_uuids),
        xml_updated,
        refreshed,
    )
    return updated_rows, xml_updated, agreement_uuids


def _apply_gold_backfill(
    context: AssetExecutionContext,
    conn: Connection,
    *,
    db: DBResource,
    gold_rows: list[dict[str, Any]],
) -> tuple[int, list[str]]:
    mapping_by_agreement: dict[str, dict[str, str]] = defaultdict(dict)
    for row in gold_rows:
        payload = _normalized_nonempty_payload(
            cast(str | None, row.get("section_standard_id_gold_label"))
        )
        if payload is None:
            continue
        agreement_uuid = str(row["agreement_uuid"])
        mapping_by_agreement[agreement_uuid][str(row["section_uuid"])] = payload

    agreement_uuids = sorted(mapping_by_agreement)
    xml_updated = _apply_xml_updates_for_agreements(
        conn,
        db=db,
        agreement_uuids=agreement_uuids,
        section_mapping_by_agreement=mapping_by_agreement,
    )
    refreshed = refresh_latest_sections_search(conn, db.database, agreement_uuids)
    context.log.info(
        (
            "taxonomy_asset (gold_backfill): applied gold labels across %s agreements; "
            "upserted %s XMLs; refreshed latest_sections_search rows=%s"
        ),
        len(agreement_uuids),
        xml_updated,
        refreshed,
    )
    return xml_updated, agreement_uuids


def _create_taxonomy_llm_lines(
    *,
    prediction_rows: list[dict[str, Any]],
    taxonomy_json: list[dict[str, Any]],
    model_name: str,
    batch_key: str,
    sections_per_request: int,
) -> list[dict[str, Any]]:
    llm_rows = _build_llm_rows(prediction_rows)
    lines: list[dict[str, Any]] = []
    for batch_idx, start in enumerate(range(0, len(llm_rows), sections_per_request)):
        chunk = llm_rows[start : start + sections_per_request]
        lines.append(
            build_taxonomy_llm_request_body(
                custom_id=f"{batch_key}:{batch_idx}",
                section_payloads=[build_taxonomy_prompt_payload(row) for row in chunk],
                taxonomy_json=taxonomy_json,
                model=model_name,
            )
        )
    return lines


def _apply_taxonomy_llm_batch_output(
    context: AssetExecutionContext,
    *,
    engine: Any,
    client: "OpenAI",
    db: DBResource,
    sections_table: str,
    batch: Any,
    model_name: str,
    log_prefix: str,
) -> tuple[int, int, list[str]]:
    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        context.log.warning("%s: batch %s has no output_file_id.", log_prefix, batch.id)
        return 0, 0, []

    out_content = client.files.content(output_file_id)
    out_text = read_openai_file_text(out_content)
    payload_by_section_uuid: dict[str, str] = {}
    parse_errors = 0

    for line_str in out_text.strip().splitlines():
        if not line_str.strip():
            continue
        custom_id = "unknown"
        try:
            raw = json.loads(line_str)
            custom_id = str(raw.get("custom_id") or "unknown")
            response = raw.get("response")
            if not isinstance(response, dict):
                raise ValueError("Missing response object.")
            status_code = response.get("status_code")
            if status_code not in (200, 201, 202):
                raise ValueError(f"Unexpected status_code={status_code!r}.")
            body = response.get("body")
            if not isinstance(body, dict):
                raise ValueError("Missing response body.")
            parsed = parse_taxonomy_llm_response_text(
                extract_output_text_from_batch_body(body)
            )
            for section_uuid, categories in parsed.items():
                payload_by_section_uuid[section_uuid] = serialize_taxonomy_labels(categories)
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            parse_errors += 1
            context.log.warning(
                "%s: parse error for %s: %s",
                log_prefix,
                custom_id,
                exc,
            )

    if not payload_by_section_uuid:
        return 0, parse_errors, []

    section_uuids = sorted(payload_by_section_uuid)
    with engine.begin() as conn:
        section_rows = conn.execute(
            text(
                f"""
                SELECT section_uuid, agreement_uuid
                FROM {sections_table}
                WHERE section_uuid IN :section_uuids
                """
            ).bindparams(bindparam("section_uuids", expanding=True)),
            {"section_uuids": tuple(section_uuids)},
        ).mappings().fetchall()

        mapping_by_agreement: dict[str, dict[str, str]] = defaultdict(dict)
        updates: list[dict[str, str]] = []
        for row in section_rows:
            section_uuid = str(row["section_uuid"])
            agreement_uuid = str(row["agreement_uuid"])
            payload = payload_by_section_uuid[section_uuid]
            updates.append(
                {
                    "section_uuid": section_uuid,
                    "gold_label_payload": payload,
                    "model_name": model_name,
                }
            )
            mapping_by_agreement[agreement_uuid][section_uuid] = payload

        if updates:
            update_sql = text(
                f"""
                UPDATE {sections_table}
                SET section_standard_id_gold_label = :gold_label_payload,
                    gold_label_model = :model_name
                WHERE section_uuid = :section_uuid
                """
            )
            for start in range(0, len(updates), 250):
                _ = conn.execute(update_sql, updates[start : start + 250])

        agreement_uuids = sorted(mapping_by_agreement)
        xml_updated = _apply_xml_updates_for_agreements(
            conn,
            db=db,
            agreement_uuids=agreement_uuids,
            section_mapping_by_agreement=mapping_by_agreement,
        )
        refreshed = refresh_latest_sections_search(conn, db.database, agreement_uuids)

    context.log.info(
        ("%s: applied %s section labels across %s agreements; upserted %s XMLs; "
         + "refreshed latest_sections_search rows=%s"),
        log_prefix,
        len(payload_by_section_uuid),
        len(agreement_uuids),
        xml_updated,
        refreshed,
    )
    return len(payload_by_section_uuid), parse_errors, agreement_uuids


def _resume_and_apply_taxonomy_llm_batch(
    context: AssetExecutionContext,
    *,
    engine: Any,
    client: "OpenAI",
    db: DBResource,
    sections_table: str,
    batch_row: dict[str, Any],
    log_prefix: str,
) -> None:
    batch = poll_batch_until_terminal(
        context,
        client,
        str(batch_row["batch_id"]),
        log_prefix=log_prefix,
    )
    with engine.begin() as conn:
        _upsert_taxonomy_llm_batch_row(
            conn,
            db.database,
            batch=batch,
            completion_window=str(batch_row["completion_window"]),
            request_total=int(batch_row["request_total"]),
            model_name=str(batch_row["model_name"]),
            batch_key=str(batch_row["batch_key"]),
        )
    if batch.status == "completed":
        _ = _apply_taxonomy_llm_batch_output(
            context,
            engine=engine,
            client=client,
            db=db,
            sections_table=sections_table,
            batch=batch,
            model_name=str(batch_row["model_name"]),
            log_prefix=log_prefix,
        )
    else:
        context.log.warning(
            "%s: batch %s ended with status=%s; no labels applied.",
            log_prefix,
            batch.id,
            batch.status,
        )
    with engine.begin() as conn:
        _mark_taxonomy_llm_batch_applied(conn, db.database, batch_id=str(batch.id))


def _create_and_apply_taxonomy_llm_batch(
    context: AssetExecutionContext,
    *,
    engine: Any,
    client: "OpenAI",
    db: DBResource,
    sections_table: str,
    prediction_rows: list[dict[str, Any]],
    model_name: str,
    sections_per_request: int,
    batch_key_override: str | None = None,
    log_prefix: str,
) -> None:
    agreement_uuids = sorted({str(row["agreement_uuid"]) for row in prediction_rows})
    batch_key = batch_key_override or agreement_batch_key(agreement_uuids)
    with engine.begin() as conn:
        taxonomy_json = _fetch_taxonomy_json(conn, db.database)
    lines = _create_taxonomy_llm_lines(
        prediction_rows=prediction_rows,
        taxonomy_json=taxonomy_json,
        model_name=model_name,
        batch_key=batch_key,
        sections_per_request=sections_per_request,
    )
    if not lines:
        context.log.info("%s: no request lines to submit.", log_prefix)
        return

    jsonl_buf = io.StringIO()
    for line in lines:
        _ = jsonl_buf.write(json.dumps(line, ensure_ascii=False) + "\n")
    jsonl_bytes = io.BytesIO(jsonl_buf.getvalue().encode("utf-8"))
    jsonl_bytes.name = TAXONOMY_LLM_REQUEST_FILENAME

    input_file = client.files.create(purpose="batch", file=jsonl_bytes)
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/responses",
        completion_window=TAXONOMY_LLM_COMPLETION_WINDOW,
    )
    with engine.begin() as conn:
        _upsert_taxonomy_llm_batch_row(
            conn,
            db.database,
            batch=batch,
            completion_window=TAXONOMY_LLM_COMPLETION_WINDOW,
            request_total=len(lines),
            model_name=model_name,
            batch_key=batch_key,
        )
    context.log.info(
        "%s: created batch %s with %s requests for %s agreements.",
        log_prefix,
        batch.id,
        len(lines),
        len(agreement_uuids),
    )
    _resume_and_apply_taxonomy_llm_batch(
        context,
        engine=engine,
        client=client,
        db=db,
        sections_table=sections_table,
        batch_row={
            "batch_id": batch.id,
            "completion_window": TAXONOMY_LLM_COMPLETION_WINDOW,
            "request_total": len(lines),
            "model_name": model_name,
            "batch_key": batch_key,
        },
        log_prefix=log_prefix,
    )


def _run_taxonomy_mode(
    context: AssetExecutionContext,
    *,
    db: DBResource,
    taxonomy_model: TaxonomyModel,
    pipeline_config: PipelineConfig,
    mode: TaxonomyMode,
    target_agreement_uuids: list[str] | None,
    batch_key_override: str | None = None,
    log_prefix: str,
) -> list[str]:
    agreement_batch_size = pipeline_config.taxonomy_agreement_batch_size
    llm_sections_per_request = pipeline_config.taxonomy_llm_sections_per_request
    single_batch_run = runs_single_batch(context, pipeline_config)
    section_title_regex = pipeline_config.taxonomy_section_title_regex

    if llm_sections_per_request <= 0:
        raise ValueError("taxonomy_llm_sections_per_request must be > 0.")

    engine = db.get_engine()
    schema = db.database
    sections_table = f"{schema}.sections"
    xml_table = f"{schema}.xml"
    last_uuid = ""
    explicit_scope = target_agreement_uuids is not None
    scoped_uuids = sorted(set(target_agreement_uuids or []))
    if explicit_scope and not scoped_uuids:
        context.log.info("%s: explicit empty scope; no taxonomy work to run.", log_prefix)
        run_post_asset_refresh(context, db, pipeline_config)
        return []
    scoped_batch_key = batch_key_override or (agreement_batch_key(scoped_uuids) if scoped_uuids else None)
    processed_agreement_uuids: set[str] = set()

    if mode == TaxonomyMode.LLM:
        with engine.begin() as conn:
            assert_tables_exist(conn, schema=schema, table_names=(TAXONOMY_LLM_BATCHES_TABLE,))

    ml_model: TaxonomyPredictor | None = None
    if mode == TaxonomyMode.ML:
        ml_model = cast(TaxonomyPredictor, cast(object, taxonomy_model.model()))

    while True:
        if mode == TaxonomyMode.LLM and pipeline_config.resume_openai_batches:
            with engine.begin() as conn:
                existing_batch = _fetch_unapplied_taxonomy_llm_batch(
                    conn,
                    schema,
                    batch_key=scoped_batch_key if scoped_uuids else None,
                )
            if existing_batch is not None:
                _resume_and_apply_taxonomy_llm_batch(
                    context,
                    engine=engine,
                    client=_oai_client(),
                    db=db,
                    sections_table=sections_table,
                    batch_row=existing_batch,
                    log_prefix=log_prefix,
                )
                if scoped_uuids or single_batch_run:
                    break
                continue

        with engine.begin() as conn:
            agreement_uuids = _select_agreement_batch(
                conn,
                sections_table=sections_table,
                xml_table=xml_table,
                mode=mode,
                last_uuid=last_uuid,
                agreement_batch_size=max(agreement_batch_size, len(scoped_uuids)) if scoped_uuids else agreement_batch_size,
                section_title_regex=section_title_regex,
                scoped_uuids=scoped_uuids or None,
            )
            if not agreement_uuids:
                break

            if mode == TaxonomyMode.ML:
                prediction_rows = _fetch_prediction_rows(
                    conn,
                    sections_table=sections_table,
                    xml_table=xml_table,
                    agreement_uuids=agreement_uuids,
                    mode=mode,
                    section_title_regex=section_title_regex,
                )
                if prediction_rows:
                    if ml_model is None:
                        raise RuntimeError("Taxonomy model was not initialized for ml mode.")
                    _ = _apply_ml_predictions(
                        context,
                        conn,
                        db=db,
                        sections_table=sections_table,
                        prediction_rows=prediction_rows,
                        model=ml_model,
                    )
                    processed_agreement_uuids.update(agreement_uuids)
            elif mode == TaxonomyMode.GOLD_BACKFILL:
                gold_rows = _fetch_gold_rows(
                    conn,
                    sections_table=sections_table,
                    xml_table=xml_table,
                    agreement_uuids=agreement_uuids,
                )
                if gold_rows:
                    _ = _apply_gold_backfill(
                        context,
                        conn,
                        db=db,
                        gold_rows=gold_rows,
                    )
                    processed_agreement_uuids.update(agreement_uuids)
            else:
                prediction_rows = _fetch_prediction_rows(
                    conn,
                    sections_table=sections_table,
                    xml_table=xml_table,
                    agreement_uuids=agreement_uuids,
                    mode=mode,
                    section_title_regex=section_title_regex,
                )
                if prediction_rows:
                    _create_and_apply_taxonomy_llm_batch(
                        context,
                        engine=engine,
                        client=_oai_client(),
                        db=db,
                        sections_table=sections_table,
                        prediction_rows=prediction_rows,
                        model_name=pipeline_config.taxonomy_llm_model,
                        sections_per_request=llm_sections_per_request,
                        batch_key_override=scoped_batch_key,
                        log_prefix=log_prefix,
                    )
                    processed_agreement_uuids.update(agreement_uuids)

            last_uuid = agreement_uuids[-1]

        if scoped_uuids or single_batch_run:
            break

    run_post_asset_refresh(context, db, pipeline_config)
    context.log.info("%s: processed %s agreements in mode=%s", log_prefix, len(processed_agreement_uuids), mode.value)
    return sorted(processed_agreement_uuids)


@dg.asset(deps=[sections_asset], name="07_taxonomy_asset")
def taxonomy_asset(
    context: AssetExecutionContext,
    db: DBResource,
    taxonomy_model: TaxonomyModel,
    pipeline_config: PipelineConfig,
) -> None:
    """Manual taxonomy entrypoint for generic sections runs."""
    processed_agreement_uuids = _run_taxonomy_mode(
        context,
        db=db,
        taxonomy_model=taxonomy_model,
        pipeline_config=pipeline_config,
        mode=pipeline_config.taxonomy_mode,
        target_agreement_uuids=None,
        log_prefix="taxonomy_asset",
    )
    context.log.info("taxonomy_asset: processed %s agreements", len(processed_agreement_uuids))


def _run_managed_taxonomy_asset(
    context: AssetExecutionContext,
    *,
    db: DBResource,
    taxonomy_model: TaxonomyModel,
    pipeline_config: PipelineConfig,
    job_name: str,
    stage_name: str,
    fallback_agreement_uuids: list[str],
    mode: TaxonomyMode,
    log_prefix: str,
    skip_if_completed: bool = False,
) -> list[str]:
    if skip_if_completed:
        should_skip, current_stage = should_skip_managed_stage(
            db=db,
            job_name=job_name,
            stage_name=stage_name,
        )
        if should_skip:
            context.log.info(
                "%s: skipping because logical run already reached %s.",
                log_prefix,
                current_stage,
            )
            return []

    scope_uuids = load_active_scope_for_job(
        context,
        db=db,
        job_name=job_name,
        fallback_agreement_uuids=fallback_agreement_uuids,
    )
    if mode == TaxonomyMode.LLM and not getattr(pipeline_config, "enable_section_taxonomy", False):
        context.log.info(
            "%s: skipping section taxonomy because enable_section_taxonomy=false.",
            log_prefix,
        )
        mark_logical_run_stage_completed(
            db=db,
            job_name=job_name,
            stage_name=stage_name,
        )
        return scope_uuids

    active_run = load_active_logical_run(db=db, job_name=job_name) if mode == TaxonomyMode.LLM else None
    batch_key_override = None
    if mode == TaxonomyMode.LLM:
        batch_key_override = build_logical_batch_key(
            logical_run_id=None if active_run is None else str(active_run["logical_run_id"]),
            stage_name=stage_name,
            default_key=agreement_batch_key(scope_uuids) if scope_uuids else None,
        )

    processed_agreement_uuids = _run_taxonomy_mode(
        context,
        db=db,
        taxonomy_model=taxonomy_model,
        pipeline_config=pipeline_config,
        mode=mode,
        target_agreement_uuids=scope_uuids,
        batch_key_override=batch_key_override,
        log_prefix=log_prefix,
    )
    mark_logical_run_stage_completed(
        db=db,
        job_name=job_name,
        stage_name=stage_name,
    )
    return processed_agreement_uuids


@dg.asset(
    name="07-01_regular_ingest_taxonomy_llm_asset",
    ins={
        "fresh_section_agreement_uuids": dg.AssetIn(key=regular_ingest_sections_from_fresh_xml_asset.key),
        "repair_section_agreement_uuids": dg.AssetIn(key=regular_ingest_sections_from_repair_xml_asset.key),
    },
)
def regular_ingest_taxonomy_llm_asset(
    context: AssetExecutionContext,
    db: DBResource,
    taxonomy_model: TaxonomyModel,
    pipeline_config: PipelineConfig,
    fresh_section_agreement_uuids: list[str],
    repair_section_agreement_uuids: list[str],
) -> list[str]:
    return _run_managed_taxonomy_asset(
        context,
        db=db,
        taxonomy_model=taxonomy_model,
        pipeline_config=pipeline_config,
        job_name="regular_ingest",
        stage_name="regular_ingest_taxonomy_llm",
        fallback_agreement_uuids=sorted(set(fresh_section_agreement_uuids) | set(repair_section_agreement_uuids)),
        mode=TaxonomyMode.LLM,
        log_prefix="regular_ingest_taxonomy_llm_asset",
        skip_if_completed=True,
    )


@dg.asset(
    name="09_regular_ingest_taxonomy_gold_backfill_asset",
    ins={"section_agreement_uuids": dg.AssetIn(key=dg.AssetKey("08-03_regular_ingest_tax_module_asset"))},
)
def regular_ingest_taxonomy_gold_backfill_asset(
    context: AssetExecutionContext,
    db: DBResource,
    taxonomy_model: TaxonomyModel,
    pipeline_config: PipelineConfig,
    section_agreement_uuids: list[str],
) -> list[str]:
    return _run_managed_taxonomy_asset(
        context,
        db=db,
        taxonomy_model=taxonomy_model,
        pipeline_config=pipeline_config,
        job_name="regular_ingest",
        stage_name="regular_ingest_taxonomy_gold_backfill",
        fallback_agreement_uuids=section_agreement_uuids,
        mode=TaxonomyMode.GOLD_BACKFILL,
        log_prefix="regular_ingest_taxonomy_gold_backfill_asset",
        skip_if_completed=True,
    )


@dg.asset(
    name="07-02_ingestion_cleanup_a_taxonomy_llm_asset",
    ins={
        "fresh_section_agreement_uuids": dg.AssetIn(key=ingestion_cleanup_a_sections_from_fresh_xml_asset.key),
        "repair_section_agreement_uuids": dg.AssetIn(key=ingestion_cleanup_a_sections_from_repair_xml_asset.key),
    },
)
def ingestion_cleanup_a_taxonomy_llm_asset(
    context: AssetExecutionContext,
    db: DBResource,
    taxonomy_model: TaxonomyModel,
    pipeline_config: PipelineConfig,
    fresh_section_agreement_uuids: list[str],
    repair_section_agreement_uuids: list[str],
) -> list[str]:
    return _run_managed_taxonomy_asset(
        context,
        db=db,
        taxonomy_model=taxonomy_model,
        pipeline_config=pipeline_config,
        job_name="ingestion_cleanup_a",
        stage_name="ingestion_cleanup_a_taxonomy_llm",
        fallback_agreement_uuids=sorted(set(fresh_section_agreement_uuids) | set(repair_section_agreement_uuids)),
        mode=TaxonomyMode.LLM,
        log_prefix="ingestion_cleanup_a_taxonomy_llm_asset",
        skip_if_completed=True,
    )


@dg.asset(
    name="07-03_ingestion_cleanup_b_taxonomy_llm_asset",
    ins={
        "repair_section_agreement_uuids": dg.AssetIn(key=ingestion_cleanup_b_sections_from_repair_xml_asset.key),
    },
)
def ingestion_cleanup_b_taxonomy_llm_asset(
    context: AssetExecutionContext,
    db: DBResource,
    taxonomy_model: TaxonomyModel,
    pipeline_config: PipelineConfig,
    repair_section_agreement_uuids: list[str],
) -> list[str]:
    return _run_managed_taxonomy_asset(
        context,
        db=db,
        taxonomy_model=taxonomy_model,
        pipeline_config=pipeline_config,
        job_name="ingestion_cleanup_b",
        stage_name="ingestion_cleanup_b_taxonomy_llm",
        fallback_agreement_uuids=sorted(set(repair_section_agreement_uuids)),
        mode=TaxonomyMode.LLM,
        log_prefix="ingestion_cleanup_b_taxonomy_llm_asset",
        skip_if_completed=True,
    )


@dg.asset(
    name="07-04_ingestion_cleanup_c_taxonomy_llm_asset",
    ins={"section_agreement_uuids": dg.AssetIn(key=ingestion_cleanup_c_sections_asset.key)},
)
def ingestion_cleanup_c_taxonomy_llm_asset(
    context: AssetExecutionContext,
    db: DBResource,
    taxonomy_model: TaxonomyModel,
    pipeline_config: PipelineConfig,
    section_agreement_uuids: list[str],
) -> list[str]:
    return _run_managed_taxonomy_asset(
        context,
        db=db,
        taxonomy_model=taxonomy_model,
        pipeline_config=pipeline_config,
        job_name="ingestion_cleanup_c",
        stage_name="ingestion_cleanup_c_taxonomy_llm",
        fallback_agreement_uuids=sorted(set(section_agreement_uuids)),
        mode=TaxonomyMode.LLM,
        log_prefix="ingestion_cleanup_c_taxonomy_llm_asset",
        skip_if_completed=True,
    )


@dg.asset(
    name="07-05_ingestion_cleanup_d_taxonomy_llm_asset",
    ins={"section_agreement_uuids": dg.AssetIn(key=ingestion_cleanup_d_sections_asset.key)},
)
def ingestion_cleanup_d_taxonomy_llm_asset(
    context: AssetExecutionContext,
    db: DBResource,
    taxonomy_model: TaxonomyModel,
    pipeline_config: PipelineConfig,
    section_agreement_uuids: list[str],
) -> list[str]:
    return _run_managed_taxonomy_asset(
        context,
        db=db,
        taxonomy_model=taxonomy_model,
        pipeline_config=pipeline_config,
        job_name="ingestion_cleanup_d",
        stage_name="ingestion_cleanup_d_taxonomy_llm",
        fallback_agreement_uuids=sorted(set(section_agreement_uuids)),
        mode=TaxonomyMode.LLM,
        log_prefix="ingestion_cleanup_d_taxonomy_llm_asset",
        skip_if_completed=True,
    )


@dg.asset(
    name="09-01_ingestion_cleanup_a_taxonomy_gold_backfill_asset",
    ins={"section_agreement_uuids": dg.AssetIn(key=dg.AssetKey("08-04_ingestion_cleanup_a_tax_module_asset"))},
)
def ingestion_cleanup_a_taxonomy_gold_backfill_asset(
    context: AssetExecutionContext,
    db: DBResource,
    taxonomy_model: TaxonomyModel,
    pipeline_config: PipelineConfig,
    section_agreement_uuids: list[str],
) -> list[str]:
    return _run_managed_taxonomy_asset(
        context,
        db=db,
        taxonomy_model=taxonomy_model,
        pipeline_config=pipeline_config,
        job_name="ingestion_cleanup_a",
        stage_name="ingestion_cleanup_a_taxonomy_gold_backfill",
        fallback_agreement_uuids=section_agreement_uuids,
        mode=TaxonomyMode.GOLD_BACKFILL,
        log_prefix="ingestion_cleanup_a_taxonomy_gold_backfill_asset",
        skip_if_completed=True,
    )


@dg.asset(
    name="09-02_ingestion_cleanup_b_taxonomy_gold_backfill_asset",
    ins={"section_agreement_uuids": dg.AssetIn(key=dg.AssetKey("08-05_ingestion_cleanup_b_tax_module_asset"))},
)
def ingestion_cleanup_b_taxonomy_gold_backfill_asset(
    context: AssetExecutionContext,
    db: DBResource,
    taxonomy_model: TaxonomyModel,
    pipeline_config: PipelineConfig,
    section_agreement_uuids: list[str],
) -> list[str]:
    return _run_managed_taxonomy_asset(
        context,
        db=db,
        taxonomy_model=taxonomy_model,
        pipeline_config=pipeline_config,
        job_name="ingestion_cleanup_b",
        stage_name="ingestion_cleanup_b_taxonomy_gold_backfill",
        fallback_agreement_uuids=section_agreement_uuids,
        mode=TaxonomyMode.GOLD_BACKFILL,
        log_prefix="ingestion_cleanup_b_taxonomy_gold_backfill_asset",
        skip_if_completed=True,
    )


@dg.asset(
    name="09-03_ingestion_cleanup_c_taxonomy_gold_backfill_asset",
    ins={"section_agreement_uuids": dg.AssetIn(key=dg.AssetKey("08-06_ingestion_cleanup_c_tax_module_asset"))},
)
def ingestion_cleanup_c_taxonomy_gold_backfill_asset(
    context: AssetExecutionContext,
    db: DBResource,
    taxonomy_model: TaxonomyModel,
    pipeline_config: PipelineConfig,
    section_agreement_uuids: list[str],
) -> list[str]:
    return _run_managed_taxonomy_asset(
        context,
        db=db,
        taxonomy_model=taxonomy_model,
        pipeline_config=pipeline_config,
        job_name="ingestion_cleanup_c",
        stage_name="ingestion_cleanup_c_taxonomy_gold_backfill",
        fallback_agreement_uuids=section_agreement_uuids,
        mode=TaxonomyMode.GOLD_BACKFILL,
        log_prefix="ingestion_cleanup_c_taxonomy_gold_backfill_asset",
        skip_if_completed=True,
    )


@dg.asset(
    name="09-04_ingestion_cleanup_d_taxonomy_gold_backfill_asset",
    ins={"section_agreement_uuids": dg.AssetIn(key=dg.AssetKey("08-07_ingestion_cleanup_d_tax_module_asset"))},
)
def ingestion_cleanup_d_taxonomy_gold_backfill_asset(
    context: AssetExecutionContext,
    db: DBResource,
    taxonomy_model: TaxonomyModel,
    pipeline_config: PipelineConfig,
    section_agreement_uuids: list[str],
) -> list[str]:
    return _run_managed_taxonomy_asset(
        context,
        db=db,
        taxonomy_model=taxonomy_model,
        pipeline_config=pipeline_config,
        job_name="ingestion_cleanup_d",
        stage_name="ingestion_cleanup_d_taxonomy_gold_backfill",
        fallback_agreement_uuids=section_agreement_uuids,
        mode=TaxonomyMode.GOLD_BACKFILL,
        log_prefix="ingestion_cleanup_d_taxonomy_gold_backfill_asset",
        skip_if_completed=True,
    )
