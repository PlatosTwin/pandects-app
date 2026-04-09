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
}

MANAGED_JOB_STAGE_SEQUENCE: dict[str, tuple[str, ...]] = {
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
}


@dataclass(frozen=True)
class LogicalJobRun:
    logical_run_id: str
    job_name: str
    status: str
    current_stage: str | None
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
    row = conn.execute(
        text(
            f"""
            SELECT logical_run_id, job_name, status, current_stage
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
    return dict(row)


def fetch_resumable_logical_run(conn: Any, *, schema: str, job_name: str) -> dict[str, Any] | None:
    row = conn.execute(
        text(
            f"""
            SELECT logical_run_id, job_name, status, current_stage
            FROM {_job_runs_table(schema, conn)}
            WHERE job_name = :job_name
              AND status IN :resumable_statuses
            ORDER BY started_at DESC, logical_run_id DESC
            LIMIT 1
            """
        ).bindparams(bindparam("resumable_statuses", expanding=True)),
        {
            "job_name": job_name,
            "resumable_statuses": [LOGICAL_RUN_STATUS_RUNNING, LOGICAL_RUN_STATUS_FAILED],
        },
    ).mappings().first()
    if row is None:
        return None
    return dict(row)


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
                agreement_uuids=resumed_scope,
                resumed_existing=True,
            )

        if pipeline_config.force_new_logical_run:
            _abandon_active_runs(conn, schema=schema, job_name=job_name)

        if not scope:
            return None

        logical_run_id = str(uuid4())
        _ = conn.execute(
            text(
                f"""
                INSERT INTO {_job_runs_table(schema, conn)} (
                    logical_run_id,
                    job_name,
                    status,
                    current_stage,
                    dagster_run_id,
                    config_json,
                    scope_size,
                    started_at,
                    updated_at,
                    finished_at
                ) VALUES (
                    :logical_run_id,
                    :job_name,
                    :status,
                    :current_stage,
                    :dagster_run_id,
                    :config_json,
                    :scope_size,
                    CURRENT_TIMESTAMP,
                    CURRENT_TIMESTAMP,
                    NULL
                )
                """
            ),
            {
                "logical_run_id": logical_run_id,
                "job_name": job_name,
                "status": LOGICAL_RUN_STATUS_RUNNING,
                "current_stage": initial_stage,
                "dagster_run_id": _current_dagster_run_id(context),
                "config_json": _pipeline_config_payload(pipeline_config),
                "scope_size": len(scope),
            },
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
                "current_stage": stage_name,
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

    current_stage_raw = active_run.get("current_stage")
    current_stage = str(current_stage_raw) if current_stage_raw is not None else None
    if current_stage not in stage_sequence:
        return False, current_stage

    current_idx = stage_sequence.index(current_stage)
    stage_idx = stage_sequence.index(stage_name)
    return current_idx > stage_idx, current_stage


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
        params: dict[str, Any] = {
            "logical_run_id": str(active_run["logical_run_id"]),
            "current_stage": stage_name,
        }
        if complete_run:
            _ = conn.execute(
                text(
                    f"""
                    UPDATE {_job_runs_table(schema, conn)}
                    SET current_stage = :current_stage,
                        status = :completed_status,
                        updated_at = CURRENT_TIMESTAMP,
                        finished_at = CURRENT_TIMESTAMP
                    WHERE logical_run_id = :logical_run_id
                    """
                ),
                {
                    **params,
                    "completed_status": LOGICAL_RUN_STATUS_COMPLETED,
                },
            )
        else:
            _ = conn.execute(
                text(
                    f"""
                    UPDATE {_job_runs_table(schema, conn)}
                    SET current_stage = :current_stage,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE logical_run_id = :logical_run_id
                    """
                ),
                params,
            )
