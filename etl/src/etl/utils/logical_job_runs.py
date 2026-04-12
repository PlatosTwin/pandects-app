"""Persisted logical run state for resumable ETL jobs."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from dagster import AssetExecutionContext
from sqlalchemy import bindparam, text

from etl.defs.resources import DBResource, PipelineConfig
from etl.utils.schema_guards import assert_tables_exist

LOGICAL_RUN_STATUS_RUNNING = "RUNNING"
LOGICAL_RUN_STATUS_COMPLETED = "COMPLETED"
LOGICAL_RUN_STATUS_FAILED = "FAILED"
LOGICAL_RUN_STATUS_ABANDONED = "ABANDONED"

MANAGED_LOGICAL_JOB_NAMES = {
    "regular_ingest",
    "ingestion_cleanup_a",
    "ingestion_cleanup_b",
    "ingestion_cleanup_c",
}

MANAGED_JOB_STAGE_SEQUENCE: dict[str, tuple[str, ...]] = {
    "regular_ingest": (
        "regular_ingest_staging",
        "regular_ingest_pre_processing",
        "regular_ingest_tagging",
        "regular_ingest_build_xml",
        "regular_ingest_verify_xml",
        "regular_ingest_ai_repair_enqueue",
        "regular_ingest_ai_repair_poll",
        "regular_ingest_reconcile_tags",
        "regular_ingest_post_repair_build_xml",
        "regular_ingest_post_repair_verify_xml",
        "regular_ingest_sections_from_fresh_xml",
        "regular_ingest_sections_from_repair_xml",
        "regular_ingest_taxonomy_llm",
        "regular_ingest_tax_module",
        "regular_ingest_taxonomy_gold_backfill",
        "regular_ingest_tx_metadata_offline",
        "regular_ingest_tx_metadata_web_search",
    ),
    "ingestion_cleanup_a": (
        "ingestion_cleanup_a_tagging",
        "ingestion_cleanup_a_build_xml",
        "ingestion_cleanup_a_verify_xml",
        "ingestion_cleanup_a_sections_from_fresh_xml",
        "ingestion_cleanup_a_ai_repair_enqueue",
        "ingestion_cleanup_a_ai_repair_poll",
        "ingestion_cleanup_a_reconcile_tags",
        "ingestion_cleanup_a_post_repair_build_xml",
        "ingestion_cleanup_a_post_repair_verify_xml",
        "ingestion_cleanup_a_sections_from_repair_xml",
        "ingestion_cleanup_a_taxonomy_llm",
        "ingestion_cleanup_a_tax_module",
        "ingestion_cleanup_a_taxonomy_gold_backfill",
        "ingestion_cleanup_a_tx_metadata_offline",
        "ingestion_cleanup_a_tx_metadata_web_search",
    ),
    "ingestion_cleanup_b": (
        "ingestion_cleanup_b_ai_repair_enqueue",
        "ingestion_cleanup_b_ai_repair_poll",
        "ingestion_cleanup_b_reconcile_tags",
        "ingestion_cleanup_b_post_repair_build_xml",
        "ingestion_cleanup_b_post_repair_verify_xml",
        "ingestion_cleanup_b_sections_from_repair_xml",
        "ingestion_cleanup_b_taxonomy_llm",
        "ingestion_cleanup_b_tax_module",
        "ingestion_cleanup_b_taxonomy_gold_backfill",
        "ingestion_cleanup_b_tx_metadata_offline",
        "ingestion_cleanup_b_tx_metadata_web_search",
    ),
    "ingestion_cleanup_c": (
        "ingestion_cleanup_c_build_xml",
        "ingestion_cleanup_c_verify_xml",
        "ingestion_cleanup_c_sections",
        "ingestion_cleanup_c_taxonomy_llm",
        "ingestion_cleanup_c_tax_module",
        "ingestion_cleanup_c_taxonomy_gold_backfill",
        "ingestion_cleanup_c_tx_metadata_offline",
        "ingestion_cleanup_c_tx_metadata_web_search",
    ),
}


def normalize_managed_stage_name(stage_name: str | None) -> str | None:
    if not stage_name:
        return None
    tokens = [token for token in str(stage_name).split("_") if token]
    while tokens and tokens[0].isdigit():
        _ = tokens.pop(0)
    if not tokens:
        return None
    normalized = "_".join(tokens)
    if normalized.endswith("_asset"):
        normalized = normalized[: -len("_asset")]
    return normalized or None


def _stage_index(job_name: str, stage_name: str | None) -> int:
    stage_sequence = MANAGED_JOB_STAGE_SEQUENCE.get(job_name)
    normalized = normalize_managed_stage_name(stage_name)
    if stage_sequence is None or normalized not in stage_sequence:
        return -1
    return stage_sequence.index(normalized)


@dataclass(frozen=True)
class LogicalJobRun:
    logical_run_id: str
    job_name: str
    status: str
    current_stage: str | None
    completed_stages: list[str]
    agreement_uuids: list[str]
    resumed_existing: bool


def _is_sqlite_bind(bind: Any) -> bool:
    return str(getattr(getattr(bind, "dialect", None), "name", "") or "").lower() == "sqlite"


def _job_runs_table(schema: str, bind: Any | None = None) -> str:
    if bind is not None and _is_sqlite_bind(bind):
        return "pipeline_job_runs"
    return f"{schema}.pipeline_job_runs"


def _job_run_agreements_table(schema: str, bind: Any | None = None) -> str:
    if bind is not None and _is_sqlite_bind(bind):
        return "pipeline_job_run_agreements"
    return f"{schema}.pipeline_job_run_agreements"


def _job_runs_has_column(conn: Any, *, schema: str, column_name: str) -> bool:
    table_name = _job_runs_table(schema, conn)
    if _is_sqlite_bind(conn):
        rows = conn.execute(text(f"PRAGMA table_info({table_name})")).mappings().all()
        return any(str(row.get("name") or "") == column_name for row in rows)
    row = conn.execute(
        text(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = :schema
              AND table_name = :table_name
              AND column_name = :column_name
            LIMIT 1
            """
        ),
        {
            "schema": schema,
            "table_name": table_name.split(".")[-1],
            "column_name": column_name,
        },
    ).first()
    return row is not None


def _job_runs_completed_stages_expr(conn: Any, *, schema: str) -> str:
    if _job_runs_has_column(conn, schema=schema, column_name="completed_stages_json"):
        return "completed_stages_json"
    return "NULL AS completed_stages_json"


def _stage_sort_key(job_name: str, stage_name: str) -> tuple[int, str]:
    return (_stage_index(job_name, stage_name), stage_name)


def _infer_completed_stages_from_current_stage(job_name: str, current_stage: str | None) -> list[str]:
    normalized_stage = normalize_managed_stage_name(current_stage)
    stage_sequence = MANAGED_JOB_STAGE_SEQUENCE.get(job_name)
    if normalized_stage is None or stage_sequence is None or normalized_stage not in stage_sequence:
        return []
    stage_idx = stage_sequence.index(normalized_stage)
    return list(stage_sequence[: stage_idx + 1])


def _decode_completed_stages(
    *,
    job_name: str,
    raw_value: Any,
    current_stage: str | None,
) -> list[str]:
    if isinstance(raw_value, str) and raw_value.strip():
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            normalized = {
                stage_name
                for item in parsed
                for stage_name in [normalize_managed_stage_name(str(item))]
                if stage_name
            }
            return sorted(normalized, key=lambda stage_name: _stage_sort_key(job_name, stage_name))
    return _infer_completed_stages_from_current_stage(job_name, current_stage)


def _encode_completed_stages(job_name: str, completed_stages: set[str]) -> str:
    normalized: set[str] = set()
    for stage_name in completed_stages:
        normalized_stage = normalize_managed_stage_name(stage_name)
        if normalized_stage:
            normalized.add(normalized_stage)
    return json.dumps(sorted(normalized, key=lambda stage_name: _stage_sort_key(job_name, stage_name)))


def _highest_completed_stage(job_name: str, completed_stages: set[str]) -> str | None:
    if not completed_stages:
        return None
    return sorted(completed_stages, key=lambda stage_name: _stage_sort_key(job_name, stage_name))[-1]


def _hydrate_logical_run_row(job_name: str, row: dict[str, Any]) -> dict[str, Any]:
    hydrated = dict(row)
    current_stage_raw = hydrated.get("current_stage")
    current_stage = normalize_managed_stage_name(str(current_stage_raw) if current_stage_raw is not None else None)
    hydrated["current_stage"] = current_stage
    hydrated["completed_stages"] = _decode_completed_stages(
        job_name=job_name,
        raw_value=hydrated.get("completed_stages_json"),
        current_stage=current_stage,
    )
    return hydrated


def _pipeline_config_payload(pipeline_config: PipelineConfig) -> str | None:
    dump_fn = getattr(pipeline_config, "model_dump", None)
    if callable(dump_fn):
        return json.dumps(dump_fn(mode="json"), sort_keys=True)
    return None


def _current_dagster_run_id(context: AssetExecutionContext) -> str | None:
    run_id = getattr(context, "run_id", None)
    return str(run_id) if run_id else None


def _insert_agreement_scope(
    conn: Any,
    *,
    schema: str,
    logical_run_id: str,
    agreement_uuids: list[str],
) -> None:
    table_name = _job_run_agreements_table(schema, conn)
    if _is_sqlite_bind(conn):
        insert_sql = text(
            f"""
            INSERT OR IGNORE INTO {table_name} (
                logical_run_id,
                agreement_uuid,
                scope_position
            ) VALUES (
                :logical_run_id,
                :agreement_uuid,
                :scope_position
            )
            """
        )
    else:
        insert_sql = text(
            f"""
            INSERT INTO {table_name} (
                logical_run_id,
                agreement_uuid,
                scope_position
            ) VALUES (
                :logical_run_id,
                :agreement_uuid,
                :scope_position
            )
            ON DUPLICATE KEY UPDATE
                scope_position = VALUES(scope_position)
            """
        )

    for idx, agreement_uuid in enumerate(agreement_uuids):
        _ = conn.execute(
            insert_sql,
            {
                "logical_run_id": logical_run_id,
                "agreement_uuid": agreement_uuid,
                "scope_position": idx,
            },
        )


def assert_logical_job_run_tables_exist(conn: Any, *, schema: str) -> None:
    if not hasattr(conn, "execute"):
        return

    if _is_sqlite_bind(conn):
        expected_names = ("pipeline_job_runs", "pipeline_job_run_agreements")
        existing = set(
            conn.execute(
                text(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                      AND name IN :table_names
                    """
                ).bindparams(bindparam("table_names", expanding=True)),
                {"table_names": list(expected_names)},
            ).scalars().all()
        )
        missing = [name for name in expected_names if name not in existing]
        if missing:
            missing_csv = ", ".join(missing)
            raise RuntimeError(
                f"Missing required table(s) in schema '{schema}': {missing_csv}. "
                + "Create these tables via migrations before running this asset."
            )
        return

    assert_tables_exist(
        conn,
        schema=schema,
        table_names=("pipeline_job_runs", "pipeline_job_run_agreements"),
    )


def fetch_active_logical_run(conn: Any, *, schema: str, job_name: str) -> dict[str, Any] | None:
    completed_stages_expr = _job_runs_completed_stages_expr(conn, schema=schema)
    row = conn.execute(
        text(
            f"""
            SELECT logical_run_id, job_name, status, current_stage, {completed_stages_expr}
            FROM {_job_runs_table(schema, conn)}
            WHERE job_name = :job_name
              AND status = :status
            ORDER BY started_at DESC, logical_run_id DESC
            LIMIT 1
            """
        ),
        {
            "job_name": job_name,
            "status": LOGICAL_RUN_STATUS_RUNNING,
        },
    ).mappings().first()
    if row is None:
        return None
    return _hydrate_logical_run_row(job_name, dict(row))


def fetch_resumable_logical_run(conn: Any, *, schema: str, job_name: str) -> dict[str, Any] | None:
    completed_stages_expr = _job_runs_completed_stages_expr(conn, schema=schema)
    rows = [
        _hydrate_logical_run_row(job_name, dict(row))
        for row in conn.execute(
        text(
            f"""
            SELECT logical_run_id, job_name, status, current_stage, started_at, {completed_stages_expr}
            FROM {_job_runs_table(schema, conn)}
            WHERE job_name = :job_name
              AND status IN :resumable_statuses
            """
        ).bindparams(bindparam("resumable_statuses", expanding=True)),
        {
            "job_name": job_name,
            "resumable_statuses": [LOGICAL_RUN_STATUS_RUNNING, LOGICAL_RUN_STATUS_FAILED],
        },
    ).mappings().all()
    ]
    if not rows:
        return None
    rows.sort(
        key=lambda row: (
            max(
                [_stage_index(job_name, row.get("current_stage"))]
                + [_stage_index(job_name, stage_name) for stage_name in row.get("completed_stages", [])]
            ),
            str(row.get("started_at") or ""),
            str(row.get("logical_run_id") or ""),
        ),
        reverse=True,
    )
    return rows[0]


def load_active_logical_run(*, db: DBResource, job_name: str) -> dict[str, Any] | None:
    engine = db.get_engine()
    schema = db.database
    with engine.begin() as conn:
        assert_logical_job_run_tables_exist(conn, schema=schema)
        return fetch_active_logical_run(conn, schema=schema, job_name=job_name)


def load_logical_run_scope(conn: Any, *, schema: str, logical_run_id: str) -> list[str]:
    return [
        str(row)
        for row in conn.execute(
            text(
                f"""
                SELECT agreement_uuid
                FROM {_job_run_agreements_table(schema, conn)}
                WHERE logical_run_id = :logical_run_id
                ORDER BY scope_position ASC, agreement_uuid ASC
                """
            ),
            {"logical_run_id": logical_run_id},
        ).scalars().all()
    ]


def _abandon_active_runs(
    conn: Any,
    *,
    schema: str,
    job_name: str,
) -> None:
    _ = conn.execute(
        text(
            f"""
            UPDATE {_job_runs_table(schema, conn)}
            SET status = :abandoned_status,
                finished_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE job_name = :job_name
              AND status IN :unfinished_statuses
            """
        ).bindparams(bindparam("unfinished_statuses", expanding=True)),
        {
            "job_name": job_name,
            "unfinished_statuses": [LOGICAL_RUN_STATUS_RUNNING, LOGICAL_RUN_STATUS_FAILED],
            "abandoned_status": LOGICAL_RUN_STATUS_ABANDONED,
        },
    )


def start_or_resume_logical_run(
    context: AssetExecutionContext,
    *,
    db: DBResource,
    pipeline_config: PipelineConfig,
    job_name: str,
    initial_stage: str,
    selected_agreement_uuids: list[str],
) -> LogicalJobRun | None:
    if job_name not in MANAGED_LOGICAL_JOB_NAMES:
        raise ValueError(f"Unsupported logical job name: {job_name}")

    scope = sorted({str(agreement_uuid) for agreement_uuid in selected_agreement_uuids if agreement_uuid})
    engine = db.get_engine()
    schema = db.database

    with engine.begin() as conn:
        assert_logical_job_run_tables_exist(conn, schema=schema)
        existing_run = None
        if pipeline_config.resume_logical_runs and not pipeline_config.force_new_logical_run:
            existing_run = fetch_resumable_logical_run(conn, schema=schema, job_name=job_name)
        if existing_run is not None:
            logical_run_id = str(existing_run["logical_run_id"])
            resumed_scope = load_logical_run_scope(conn, schema=schema, logical_run_id=logical_run_id)
            _ = conn.execute(
                text(
                    f"""
                    UPDATE {_job_runs_table(schema, conn)}
                    SET dagster_run_id = :dagster_run_id,
                        status = :running_status,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE logical_run_id = :logical_run_id
                    """
                ),
                {
                    "logical_run_id": logical_run_id,
                    "dagster_run_id": _current_dagster_run_id(context),
                    "running_status": LOGICAL_RUN_STATUS_RUNNING,
                },
            )
            return LogicalJobRun(
                logical_run_id=logical_run_id,
                job_name=job_name,
                status=LOGICAL_RUN_STATUS_RUNNING,
                current_stage=str(existing_run["current_stage"]) if existing_run.get("current_stage") is not None else None,
                completed_stages=list(existing_run.get("completed_stages", [])),
                agreement_uuids=resumed_scope,
                resumed_existing=True,
            )

        if pipeline_config.force_new_logical_run:
            _abandon_active_runs(conn, schema=schema, job_name=job_name)

        if not scope:
            return None

        logical_run_id = str(uuid4())
        insert_columns = [
            "logical_run_id",
            "job_name",
            "status",
            "current_stage",
            "dagster_run_id",
            "config_json",
            "scope_size",
            "started_at",
            "updated_at",
            "finished_at",
        ]
        insert_values = [
            ":logical_run_id",
            ":job_name",
            ":status",
            ":current_stage",
            ":dagster_run_id",
            ":config_json",
            ":scope_size",
            "CURRENT_TIMESTAMP",
            "CURRENT_TIMESTAMP",
            "NULL",
        ]
        params: dict[str, Any] = {
            "logical_run_id": logical_run_id,
            "job_name": job_name,
            "status": LOGICAL_RUN_STATUS_RUNNING,
            "current_stage": initial_stage,
            "dagster_run_id": _current_dagster_run_id(context),
            "config_json": _pipeline_config_payload(pipeline_config),
            "scope_size": len(scope),
        }
        if _job_runs_has_column(conn, schema=schema, column_name="completed_stages_json"):
            insert_columns.insert(4, "completed_stages_json")
            insert_values.insert(4, ":completed_stages_json")
            params["completed_stages_json"] = json.dumps([])
        _ = conn.execute(
            text(
                f"""
                INSERT INTO {_job_runs_table(schema, conn)} (
                    {", ".join(insert_columns)}
                ) VALUES (
                    {", ".join(insert_values)}
                )
                """
            ),
            params,
        )
        _insert_agreement_scope(
            conn,
            schema=schema,
            logical_run_id=logical_run_id,
            agreement_uuids=scope,
        )
        return LogicalJobRun(
            logical_run_id=logical_run_id,
            job_name=job_name,
            status=LOGICAL_RUN_STATUS_RUNNING,
            current_stage=initial_stage,
            completed_stages=[],
            agreement_uuids=scope,
            resumed_existing=False,
        )


def load_active_scope_for_job(
    context: AssetExecutionContext,
    *,
    db: DBResource,
    job_name: str,
    fallback_agreement_uuids: list[str],
) -> list[str]:
    fallback_scope = sorted({str(agreement_uuid) for agreement_uuid in fallback_agreement_uuids if agreement_uuid})
    engine = db.get_engine()
    schema = db.database
    with engine.begin() as conn:
        assert_logical_job_run_tables_exist(conn, schema=schema)
        active_run = fetch_active_logical_run(conn, schema=schema, job_name=job_name)
        if active_run is None:
            return fallback_scope
        logical_run_id = str(active_run["logical_run_id"])
        _ = conn.execute(
            text(
                f"""
                UPDATE {_job_runs_table(schema, conn)}
                SET dagster_run_id = :dagster_run_id,
                    updated_at = CURRENT_TIMESTAMP
                WHERE logical_run_id = :logical_run_id
                """
            ),
            {
                "logical_run_id": logical_run_id,
                "dagster_run_id": _current_dagster_run_id(context),
            },
        )
        return load_logical_run_scope(conn, schema=schema, logical_run_id=logical_run_id)


def build_logical_batch_scope_tag(job_name: str, stage_name: str, logical_run_id: str | None) -> str | None:
    if not logical_run_id:
        return None
    return f"{job_name}:{stage_name}:{logical_run_id}"


def build_logical_batch_key(*, logical_run_id: str | None, stage_name: str, default_key: str | None) -> str | None:
    if logical_run_id is None:
        return default_key
    if default_key:
        return f"{logical_run_id}:{stage_name}:{default_key}"
    return f"{logical_run_id}:{stage_name}"


def mark_logical_run_failed(
    *,
    db: DBResource,
    job_name: str,
    stage_name: str | None,
    dagster_run_id: str | None = None,
) -> None:
    engine = db.get_engine()
    schema = db.database
    with engine.begin() as conn:
        assert_logical_job_run_tables_exist(conn, schema=schema)
        logical_run_id: str | None = None
        if dagster_run_id:
            row = conn.execute(
                text(
                    f"""
                    SELECT logical_run_id
                    FROM {_job_runs_table(schema, conn)}
                    WHERE job_name = :job_name
                      AND status = :running_status
                      AND dagster_run_id = :dagster_run_id
                    ORDER BY started_at DESC, logical_run_id DESC
                    LIMIT 1
                    """
                ),
                {
                    "job_name": job_name,
                    "running_status": LOGICAL_RUN_STATUS_RUNNING,
                    "dagster_run_id": dagster_run_id,
                },
            ).scalar()
            if row is not None:
                logical_run_id = str(row)

        if logical_run_id is None:
            active_run = fetch_active_logical_run(conn, schema=schema, job_name=job_name)
            if active_run is None:
                return
            logical_run_id = str(active_run["logical_run_id"])

        normalized_stage_name = normalize_managed_stage_name(stage_name)
        _ = conn.execute(
            text(
                f"""
                UPDATE {_job_runs_table(schema, conn)}
                SET current_stage = :current_stage,
                    status = :failed_status,
                    updated_at = CURRENT_TIMESTAMP,
                    finished_at = CURRENT_TIMESTAMP
                WHERE logical_run_id = :logical_run_id
                """
            ),
            {
                "logical_run_id": logical_run_id,
                "current_stage": normalized_stage_name,
                "failed_status": LOGICAL_RUN_STATUS_FAILED,
            },
        )


def should_skip_managed_stage(
    *,
    db: DBResource,
    job_name: str,
    stage_name: str,
) -> tuple[bool, str | None]:
    stage_sequence = MANAGED_JOB_STAGE_SEQUENCE.get(job_name)
    if stage_sequence is None or stage_name not in stage_sequence:
        return False, None

    active_run = load_active_logical_run(db=db, job_name=job_name)
    if active_run is None:
        return False, None

    current_stage = active_run.get("current_stage")
    completed_stages = {
        stage
        for stage in active_run.get("completed_stages", [])
        if isinstance(stage, str) and stage in stage_sequence
    }
    if stage_name in completed_stages:
        return True, current_stage

    # Backward compatibility for logical runs recorded before completed_stages_json
    # was populated consistently. If a legacy run has no explicit completed-stage
    # history but has already advanced to a later stage, skip any earlier stage.
    current_stage_index = _stage_index(job_name, str(current_stage) if current_stage is not None else None)
    target_stage_index = _stage_index(job_name, stage_name)
    if not completed_stages and current_stage_index > target_stage_index >= 0:
        return True, current_stage

    return False, current_stage


def mark_logical_run_stage_completed(
    *,
    db: DBResource,
    job_name: str,
    stage_name: str,
    complete_run: bool = False,
) -> None:
    engine = db.get_engine()
    schema = db.database
    with engine.begin() as conn:
        assert_logical_job_run_tables_exist(conn, schema=schema)
        active_run = fetch_active_logical_run(conn, schema=schema, job_name=job_name)
        if active_run is None:
            return
        normalized_stage_name = normalize_managed_stage_name(stage_name)
        completed_stages = {
            stage
            for stage in active_run.get("completed_stages", [])
            if isinstance(stage, str)
        }
        if normalized_stage_name is not None:
            completed_stages.add(normalized_stage_name)
        next_stage_name = _highest_completed_stage(job_name, completed_stages) or normalized_stage_name
        params: dict[str, Any] = {
            "logical_run_id": str(active_run["logical_run_id"]),
            "current_stage": next_stage_name,
        }
        set_clauses = [
            "current_stage = :current_stage",
            "updated_at = CURRENT_TIMESTAMP",
        ]
        if complete_run:
            set_clauses.extend(
                [
                    "status = :completed_status",
                    "finished_at = CURRENT_TIMESTAMP",
                ]
            )
        if _job_runs_has_column(conn, schema=schema, column_name="completed_stages_json"):
            params["completed_stages_json"] = _encode_completed_stages(job_name, completed_stages)
            set_clauses.insert(1, "completed_stages_json = :completed_stages_json")
        if complete_run:
            params["completed_status"] = LOGICAL_RUN_STATUS_COMPLETED
        _ = conn.execute(
            text(
                f"""
                UPDATE {_job_runs_table(schema, conn)}
                SET {", ".join(set_clauses)}
                WHERE logical_run_id = :logical_run_id
                """
            ),
            params,
        )
