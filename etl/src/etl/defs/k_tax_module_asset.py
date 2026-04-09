# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false

import io
import json
import os
from typing import Any, cast

import dagster as dg
from dagster import AssetExecutionContext
from openai import OpenAI
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection

from etl.defs.g_sections_asset import (
    sections_asset,
    sections_from_fresh_xml_asset,
    sections_from_repair_xml_asset,
)
from etl.defs.h_taxonomy_asset import (
    ingestion_cleanup_a_taxonomy_llm_asset,
    ingestion_cleanup_b_taxonomy_llm_asset,
    regular_ingest_taxonomy_llm_asset,
)
from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.tax_module import (
    TaxAssignmentRecord,
    TaxClauseRecord,
    TaxModuleLLMRow,
    TaxSectionRow,
    build_tax_clause_llm_request_body,
    build_tax_clause_prompt_payload,
    extract_tax_clauses,
    is_tax_related_section,
    parse_tax_clause_llm_response_text,
)
from etl.utils.batch_keys import agreement_batch_key
from etl.utils.db_utils import replace_module_clauses, upsert_tax_clause_assignments
from etl.utils.logical_job_runs import (
    build_logical_batch_key,
    load_active_logical_run,
    load_active_scope_for_job,
    mark_logical_run_stage_completed,
)
from etl.utils.openai_batch import (
    extract_output_text_from_batch_body,
    poll_batch_until_terminal,
    read_openai_file_text,
)
from etl.utils.post_asset_refresh import run_post_asset_refresh
from etl.utils.run_config import runs_single_batch
from etl.utils.schema_guards import assert_tables_exist


CLAUSES_TABLE = "clauses"
TAX_CLAUSE_ASSIGNMENTS_TABLE = "tax_clause_assignments"
TAX_CLAUSE_TAXONOMY_L1_TABLE = "tax_clause_taxonomy_l1"
TAX_CLAUSE_TAXONOMY_L2_TABLE = "tax_clause_taxonomy_l2"
TAX_CLAUSE_TAXONOMY_L3_TABLE = "tax_clause_taxonomy_l3"
TAX_MODULE_LLM_BATCHES_TABLE = "tax_module_llm_batches"
TAX_MODULE_LLM_REQUEST_FILENAME = "tax_module_llm_requests.jsonl"
TAX_MODULE_LLM_COMPLETION_WINDOW = "24h"


def _oai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for tax_module_asset.")
    return OpenAI(api_key=api_key)


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
            FROM {schema}.{TAX_CLAUSE_TAXONOMY_L1_TABLE} l1
            LEFT JOIN {schema}.{TAX_CLAUSE_TAXONOMY_L2_TABLE} l2
                ON l1.standard_id = l2.parent_id
            LEFT JOIN {schema}.{TAX_CLAUSE_TAXONOMY_L3_TABLE} l3
                ON l2.standard_id = l3.parent_id
            ORDER BY l1.standard_id, l2.standard_id, l3.standard_id
            """
        )
    ).mappings().fetchall()
    return [dict(row) for row in rows]


def _fetch_tax_section_standard_ids(conn: Connection, schema: str) -> set[str]:
    query = """
        SELECT standard_id
        FROM {table_name}
        WHERE LOWER(COALESCE(label, '')) LIKE '%tax%'
    """
    standard_ids: set[str] = set()
    for table_name in (
        TAX_CLAUSE_TAXONOMY_L1_TABLE.replace("tax_clause_", ""),
        TAX_CLAUSE_TAXONOMY_L2_TABLE.replace("tax_clause_", ""),
        TAX_CLAUSE_TAXONOMY_L3_TABLE.replace("tax_clause_", ""),
    ):
        rows = conn.execute(text(query.format(table_name=f"{schema}.{table_name}"))).scalars().all()
        for standard_id in rows:
            if isinstance(standard_id, str) and standard_id.strip():
                standard_ids.add(standard_id.strip())
    return standard_ids


def _tax_candidate_sql(
    *,
    section_ids: set[str],
    standard_id_column_sql: str,
) -> tuple[str, dict[str, object]]:
    conditions = [
        "LOWER(COALESCE(s.section_title_normed, '')) LIKE '%tax%'",
        "LOWER(COALESCE(s.article_title_normed, '')) LIKE '%tax%'",
    ]
    params: dict[str, object] = {}
    if section_ids:
        for idx, standard_id in enumerate(sorted(section_ids)):
            key = f"sid_like_{idx}"
            conditions.append(f"{standard_id_column_sql} LIKE :{key}")
            params[key] = f'%"{standard_id}"%'
    return "(" + " OR ".join(conditions) + ")", params


def _select_agreement_batch(
    conn: Connection,
    *,
    schema: str,
    agreement_batch_size: int,
    last_uuid: str,
    tax_section_standard_ids: set[str],
    scoped_uuids: list[str] | None,
) -> list[str]:
    candidate_sql, params = _tax_candidate_sql(
        section_ids=tax_section_standard_ids,
        standard_id_column_sql="COALESCE(s.section_standard_id_gold_label, s.section_standard_id, '')",
    )
    query_params: dict[str, object] = {
        "last_uuid": last_uuid,
        "lim": agreement_batch_size,
        **params,
    }
    scope_clause = ""
    if scoped_uuids:
        scope_clause = "AND s.agreement_uuid IN :agreement_uuids"
        query_params["agreement_uuids"] = tuple(scoped_uuids)
    query = text(
        f"""
        SELECT DISTINCT s.agreement_uuid
        FROM {schema}.sections s
        JOIN {schema}.xml x
          ON x.agreement_uuid = s.agreement_uuid
         AND x.version = s.xml_version
        LEFT JOIN {schema}.{CLAUSES_TABLE} c
          ON c.section_uuid = s.section_uuid
         AND c.module = 'tax'
         AND c.xml_version = s.xml_version
        WHERE s.agreement_uuid > :last_uuid
          {scope_clause}
          AND x.latest = 1
          AND x.status = 'verified'
          AND {candidate_sql}
          AND c.clause_uuid IS NULL
        ORDER BY s.agreement_uuid
        LIMIT :lim
        """
    )
    if scoped_uuids:
        query = query.bindparams(bindparam("agreement_uuids", expanding=True))
    rows = conn.execute(query, query_params).scalars().all()
    return [str(row) for row in rows]


def _fetch_sections_for_agreements(
    conn: Connection,
    *,
    schema: str,
    agreement_uuids: list[str],
) -> list[TaxSectionRow]:
    if not agreement_uuids:
        return []
    rows = conn.execute(
        text(
            f"""
            SELECT
                s.agreement_uuid,
                s.section_uuid,
                s.article_title,
                s.article_title_normed,
                s.section_title,
                s.section_title_normed,
                s.xml_content,
                s.xml_version,
                s.section_standard_id,
                s.section_standard_id_gold_label
            FROM {schema}.sections s
            JOIN {schema}.xml x
              ON x.agreement_uuid = s.agreement_uuid
             AND x.version = s.xml_version
            WHERE s.agreement_uuid IN :agreement_uuids
              AND x.latest = 1
              AND x.status = 'verified'
            ORDER BY
                s.agreement_uuid,
                COALESCE(s.article_order, -1),
                COALESCE(s.section_order, -1),
                s.section_uuid
            """
        ).bindparams(bindparam("agreement_uuids", expanding=True)),
        {"agreement_uuids": tuple(agreement_uuids)},
    ).mappings().fetchall()
    return [cast(TaxSectionRow, cast(object, dict(row))) for row in rows]


def _build_clause_rows(
    *,
    section_rows: list[TaxSectionRow],
    tax_section_standard_ids: set[str],
) -> list[TaxClauseRecord]:
    clauses: list[TaxClauseRecord] = []
    for row in section_rows:
        if not is_tax_related_section(row, tax_standard_ids=tax_section_standard_ids):
            continue
        clauses.extend(extract_tax_clauses(row))
    return clauses


def _insert_clauses_for_agreements(
    conn: Connection,
    *,
    schema: str,
    agreement_uuids: list[str],
    clause_rows: list[TaxClauseRecord],
) -> None:
    replace_module_clauses(
        cast(list[dict[str, object]], cast(object, clause_rows)),
        agreement_uuids=agreement_uuids,
        module="tax",
        schema=schema,
        conn=conn,
    )


def _llm_rows_from_clauses(
    clause_rows: list[TaxClauseRecord],
    *,
    section_rows_by_uuid: dict[str, TaxSectionRow],
) -> list[TaxModuleLLMRow]:
    out: list[TaxModuleLLMRow] = []
    for clause in clause_rows:
        section_row = section_rows_by_uuid[clause["section_uuid"]]
        out.append(
            {
                "clause_uuid": clause["clause_uuid"],
                "agreement_uuid": clause["agreement_uuid"],
                "section_uuid": clause["section_uuid"],
                "article_title": section_row.get("article_title"),
                "section_title": section_row.get("section_title"),
                "clause_text": clause["clause_text"],
                "anchor_label": clause["anchor_label"],
                "context_type": clause["context_type"],
            }
        )
    return out


def _create_llm_lines(
    *,
    clause_rows: list[TaxClauseRecord],
    section_rows_by_uuid: dict[str, TaxSectionRow],
    taxonomy_json: list[dict[str, Any]],
    model_name: str,
    clauses_per_request: int,
    batch_key: str,
) -> list[dict[str, Any]]:
    llm_rows = _llm_rows_from_clauses(clause_rows, section_rows_by_uuid=section_rows_by_uuid)
    lines: list[dict[str, Any]] = []
    for batch_idx, start in enumerate(range(0, len(llm_rows), clauses_per_request)):
        chunk = llm_rows[start : start + clauses_per_request]
        lines.append(
            build_tax_clause_llm_request_body(
                custom_id=f"{batch_key}:{batch_idx}",
                clause_payloads=[build_tax_clause_prompt_payload(row) for row in chunk],
                taxonomy_json=taxonomy_json,
                model=model_name,
            )
        )
    return lines


def _fetch_unapplied_tax_module_batch(
    conn: Connection,
    schema: str,
    *,
    batch_key: str | None = None,
) -> dict[str, Any] | None:
    batch_key_clause = ""
    params: dict[str, object] = {}
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
            FROM {schema}.{TAX_MODULE_LLM_BATCHES_TABLE}
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


def _upsert_tax_module_batch_row(
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
            INSERT INTO {schema}.{TAX_MODULE_LLM_BATCHES_TABLE} (
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
            ) VALUES (
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


def _mark_tax_module_batch_applied(conn: Connection, schema: str, *, batch_id: str) -> None:
    _ = conn.execute(
        text(
            f"""
            UPDATE {schema}.{TAX_MODULE_LLM_BATCHES_TABLE}
            SET applied = 1, applied_at = UTC_TIMESTAMP()
            WHERE batch_id = :batch_id
            """
        ),
        {"batch_id": batch_id},
    )


def _apply_tax_module_batch_output(
    context: AssetExecutionContext,
    *,
    engine: Any,
    client: OpenAI,
    db: DBResource,
    batch: Any,
    model_name: str,
) -> tuple[int, int]:
    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        context.log.warning("tax_module_asset: batch %s has no output_file_id.", batch.id)
        return 0, 0

    out_text = read_openai_file_text(client.files.content(output_file_id))
    payload_by_clause_uuid: dict[str, list[str]] = {}
    parse_errors = 0
    for line_str in out_text.strip().splitlines():
        if not line_str.strip():
            continue
        try:
            raw = json.loads(line_str)
            response = raw.get("response")
            if not isinstance(response, dict):
                raise ValueError("Missing response.")
            body = response.get("body")
            if not isinstance(body, dict):
                raise ValueError("Missing body.")
            parsed = parse_tax_clause_llm_response_text(extract_output_text_from_batch_body(body))
            payload_by_clause_uuid.update(parsed)
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            parse_errors += 1
            context.log.warning("tax_module_asset: parse error for batch line: %s", exc)

    if not payload_by_clause_uuid:
        return 0, parse_errors

    assignments: list[TaxAssignmentRecord] = []
    for clause_uuid, categories in payload_by_clause_uuid.items():
        seen: set[str] = set()
        for category in categories:
            cleaned = category.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            assignments.append(
                {
                    "clause_uuid": clause_uuid,
                    "standard_id": cleaned,
                    "is_gold_label": 1,
                    "model_name": model_name,
                }
            )

    with engine.begin() as conn:
        if assignments:
            upsert_tax_clause_assignments(
                cast(list[dict[str, object]], cast(object, assignments)),
                schema=db.database,
                conn=conn,
            )
    return len(payload_by_clause_uuid), parse_errors


def _resume_and_apply_tax_module_batch(
    context: AssetExecutionContext,
    *,
    engine: Any,
    client: OpenAI,
    db: DBResource,
    batch_row: dict[str, Any],
) -> None:
    batch = poll_batch_until_terminal(
        context,
        client,
        str(batch_row["batch_id"]),
        log_prefix="tax_module_asset",
    )
    with engine.begin() as conn:
        _upsert_tax_module_batch_row(
            conn,
            db.database,
            batch=batch,
            completion_window=str(batch_row["completion_window"]),
            request_total=int(batch_row["request_total"]),
            model_name=str(batch_row["model_name"]),
            batch_key=str(batch_row["batch_key"]),
        )
    if batch.status == "completed":
        _ = _apply_tax_module_batch_output(
            context,
            engine=engine,
            client=client,
            db=db,
            batch=batch,
            model_name=str(batch_row["model_name"]),
        )
    else:
        context.log.warning(
            "tax_module_asset: batch %s ended with status=%s; no assignments applied.",
            batch.id,
            batch.status,
        )
    with engine.begin() as conn:
        _mark_tax_module_batch_applied(conn, db.database, batch_id=str(batch.id))


def _run_tax_module_for_agreements(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    *,
    target_agreement_uuids: list[str] | None,
    batch_key_override: str | None = None,
    log_prefix: str,
) -> list[str]:
    agreement_batch_size = pipeline_config.tax_module_agreement_batch_size
    clauses_per_request = pipeline_config.tax_module_llm_clauses_per_request
    if clauses_per_request <= 0:
        raise ValueError("tax_module_llm_clauses_per_request must be > 0.")

    single_batch_run = runs_single_batch(context, pipeline_config)
    engine = db.get_engine()
    schema = db.database
    last_uuid = ""
    processed_agreement_uuids: list[str] = []
    explicit_scope = target_agreement_uuids is not None
    scoped_uuids = sorted(set(target_agreement_uuids or []))
    if explicit_scope and not scoped_uuids:
        context.log.info("%s: explicit empty scope; no tax-module work to run.", log_prefix)
        run_post_asset_refresh(context, db, pipeline_config)
        return []

    with engine.begin() as conn:
        assert_tables_exist(
            conn,
            schema=schema,
            table_names=(
                CLAUSES_TABLE,
                TAX_CLAUSE_ASSIGNMENTS_TABLE,
                TAX_CLAUSE_TAXONOMY_L1_TABLE,
                TAX_CLAUSE_TAXONOMY_L2_TABLE,
                TAX_CLAUSE_TAXONOMY_L3_TABLE,
                TAX_MODULE_LLM_BATCHES_TABLE,
            ),
        )

    scoped_batch_key = batch_key_override or (agreement_batch_key(scoped_uuids) if scoped_uuids else None)
    while True:
        if pipeline_config.resume_openai_batches:
            with engine.begin() as conn:
                existing_batch = _fetch_unapplied_tax_module_batch(
                    conn,
                    schema,
                    batch_key=scoped_batch_key,
                )
            if existing_batch is not None:
                context.log.info(
                    "%s: resuming unapplied batch %s.",
                    log_prefix,
                    existing_batch["batch_id"],
                )
                _resume_and_apply_tax_module_batch(
                    context,
                    engine=engine,
                    client=_oai_client(),
                    db=db,
                    batch_row=existing_batch,
                )
                if single_batch_run:
                    break
                continue

        with engine.begin() as conn:
            tax_section_standard_ids = _fetch_tax_section_standard_ids(conn, schema)
            agreement_uuids = _select_agreement_batch(
                conn,
                schema=schema,
                agreement_batch_size=max(agreement_batch_size, len(scoped_uuids) or agreement_batch_size),
                last_uuid=last_uuid,
                tax_section_standard_ids=tax_section_standard_ids,
                scoped_uuids=scoped_uuids or None,
            )
            if not agreement_uuids:
                break

            section_rows = _fetch_sections_for_agreements(conn, schema=schema, agreement_uuids=agreement_uuids)
            clause_rows = _build_clause_rows(
                section_rows=section_rows,
                tax_section_standard_ids=tax_section_standard_ids,
            )
            _insert_clauses_for_agreements(
                conn,
                schema=schema,
                agreement_uuids=agreement_uuids,
                clause_rows=clause_rows,
            )
            processed_agreement_uuids.extend(agreement_uuids)

            if clause_rows:
                section_rows_by_uuid = {row["section_uuid"]: row for row in section_rows}
                taxonomy_json = _fetch_taxonomy_json(conn, schema)
                batch_key = scoped_batch_key or agreement_batch_key(agreement_uuids)
                lines = _create_llm_lines(
                    clause_rows=clause_rows,
                    section_rows_by_uuid=section_rows_by_uuid,
                    taxonomy_json=taxonomy_json,
                    model_name=pipeline_config.tax_module_llm_model,
                    clauses_per_request=clauses_per_request,
                    batch_key=batch_key,
                )
                if lines:
                    jsonl_buf = io.StringIO()
                    for line in lines:
                        _ = jsonl_buf.write(json.dumps(line, ensure_ascii=False) + "\n")
                    jsonl_bytes = io.BytesIO(jsonl_buf.getvalue().encode("utf-8"))
                    jsonl_bytes.name = TAX_MODULE_LLM_REQUEST_FILENAME
                    client = _oai_client()
                    input_file = client.files.create(purpose="batch", file=jsonl_bytes)
                    batch = client.batches.create(
                        input_file_id=input_file.id,
                        endpoint="/v1/responses",
                        completion_window=TAX_MODULE_LLM_COMPLETION_WINDOW,
                    )
                    _upsert_tax_module_batch_row(
                        conn,
                        schema,
                        batch=batch,
                        completion_window=TAX_MODULE_LLM_COMPLETION_WINDOW,
                        request_total=len(lines),
                        model_name=pipeline_config.tax_module_llm_model,
                        batch_key=batch_key,
                    )
                    context.log.info(
                        "%s: created batch %s with %s requests for %s agreements.",
                        log_prefix,
                        batch.id,
                        len(lines),
                        len(agreement_uuids),
                    )
                    batch_row = {
                        "batch_id": batch.id,
                        "completion_window": TAX_MODULE_LLM_COMPLETION_WINDOW,
                        "request_total": len(lines),
                        "model_name": pipeline_config.tax_module_llm_model,
                        "batch_key": batch_key,
                    }
                else:
                    batch_row = None
            else:
                context.log.info("%s: no tax clauses extracted for %s agreements.", log_prefix, len(agreement_uuids))
                batch_row = None

            last_uuid = agreement_uuids[-1]

        if batch_row is not None:
            _resume_and_apply_tax_module_batch(
                context,
                engine=engine,
                client=_oai_client(),
                db=db,
                batch_row=batch_row,
            )

        if scoped_uuids:
            break
        if single_batch_run:
            break

    run_post_asset_refresh(context, db, pipeline_config)
    return sorted(set(processed_agreement_uuids))


@dg.asset(deps=[sections_asset], name="08_tax_module_asset")
def tax_module_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> list[str]:
    """Manual tax-module entrypoint for generic sections runs."""
    return _run_tax_module_for_agreements(
        context,
        db,
        pipeline_config,
        target_agreement_uuids=None,
        log_prefix="tax_module_asset",
    )


@dg.asset(
    name="08-01_tax_module_from_fresh_xml",
    ins={"section_agreement_uuids": dg.AssetIn(key=sections_from_fresh_xml_asset.key)},
)
def tax_module_from_fresh_xml_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    section_agreement_uuids: list[str],
) -> list[str]:
    return _run_tax_module_for_agreements(
        context,
        db,
        pipeline_config,
        target_agreement_uuids=section_agreement_uuids,
        log_prefix="tax_module_from_fresh_xml_asset",
    )


@dg.asset(
    name="08-02_tax_module_from_repair_xml",
    ins={"section_agreement_uuids": dg.AssetIn(key=sections_from_repair_xml_asset.key)},
)
def tax_module_from_repair_xml_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    section_agreement_uuids: list[str],
) -> list[str]:
    return _run_tax_module_for_agreements(
        context,
        db,
        pipeline_config,
        target_agreement_uuids=section_agreement_uuids,
        log_prefix="tax_module_from_repair_xml_asset",
    )


@dg.asset(
    name="08-03_regular_ingest_tax_module_asset",
    ins={"section_agreement_uuids": dg.AssetIn(key=regular_ingest_taxonomy_llm_asset.key)},
)
def regular_ingest_tax_module_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    section_agreement_uuids: list[str],
) -> list[str]:
    scope_uuids = load_active_scope_for_job(
        context,
        db=db,
        job_name="regular_ingest",
        fallback_agreement_uuids=section_agreement_uuids,
    )
    active_run = load_active_logical_run(db=db, job_name="regular_ingest")
    processed_agreement_uuids = _run_tax_module_for_agreements(
        context,
        db,
        pipeline_config,
        target_agreement_uuids=scope_uuids,
        batch_key_override=build_logical_batch_key(
            logical_run_id=None if active_run is None else str(active_run["logical_run_id"]),
            stage_name="regular_ingest_tax_module",
            default_key=agreement_batch_key(scope_uuids) if scope_uuids else None,
        ),
        log_prefix="regular_ingest_tax_module_asset",
    )
    mark_logical_run_stage_completed(
        db=db,
        job_name="regular_ingest",
        stage_name="regular_ingest_tax_module",
    )
    return processed_agreement_uuids


@dg.asset(
    name="08-04_ingestion_cleanup_a_tax_module_asset",
    ins={"section_agreement_uuids": dg.AssetIn(key=ingestion_cleanup_a_taxonomy_llm_asset.key)},
)
def ingestion_cleanup_a_tax_module_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    section_agreement_uuids: list[str],
) -> list[str]:
    scope_uuids = load_active_scope_for_job(
        context,
        db=db,
        job_name="ingestion_cleanup_a",
        fallback_agreement_uuids=section_agreement_uuids,
    )
    active_run = load_active_logical_run(db=db, job_name="ingestion_cleanup_a")
    processed_agreement_uuids = _run_tax_module_for_agreements(
        context,
        db,
        pipeline_config,
        target_agreement_uuids=scope_uuids,
        batch_key_override=build_logical_batch_key(
            logical_run_id=None if active_run is None else str(active_run["logical_run_id"]),
            stage_name="ingestion_cleanup_a_tax_module",
            default_key=agreement_batch_key(scope_uuids) if scope_uuids else None,
        ),
        log_prefix="ingestion_cleanup_a_tax_module_asset",
    )
    mark_logical_run_stage_completed(
        db=db,
        job_name="ingestion_cleanup_a",
        stage_name="ingestion_cleanup_a_tax_module",
    )
    return processed_agreement_uuids


@dg.asset(
    name="08-05_ingestion_cleanup_b_tax_module_asset",
    ins={"section_agreement_uuids": dg.AssetIn(key=ingestion_cleanup_b_taxonomy_llm_asset.key)},
)
def ingestion_cleanup_b_tax_module_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    section_agreement_uuids: list[str],
) -> list[str]:
    scope_uuids = load_active_scope_for_job(
        context,
        db=db,
        job_name="ingestion_cleanup_b",
        fallback_agreement_uuids=section_agreement_uuids,
    )
    active_run = load_active_logical_run(db=db, job_name="ingestion_cleanup_b")
    processed_agreement_uuids = _run_tax_module_for_agreements(
        context,
        db,
        pipeline_config,
        target_agreement_uuids=scope_uuids,
        batch_key_override=build_logical_batch_key(
            logical_run_id=None if active_run is None else str(active_run["logical_run_id"]),
            stage_name="ingestion_cleanup_b_tax_module",
            default_key=agreement_batch_key(scope_uuids) if scope_uuids else None,
        ),
        log_prefix="ingestion_cleanup_b_tax_module_asset",
    )
    mark_logical_run_stage_completed(
        db=db,
        job_name="ingestion_cleanup_b",
        stage_name="ingestion_cleanup_b_tax_module",
    )
    return processed_agreement_uuids
