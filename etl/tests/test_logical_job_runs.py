"""Unit tests for persisted logical ingest job runs."""
# pyright: reportAny=false

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import cast

from dagster import AssetExecutionContext
from sqlalchemy import create_engine, text

from etl.defs.resources import DBResource, PipelineConfig
from etl.utils.logical_job_runs import (
    LOGICAL_RUN_STATUS_ABANDONED,
    LOGICAL_RUN_STATUS_COMPLETED,
    LOGICAL_RUN_STATUS_RUNNING,
    load_active_logical_run,
    load_active_scope_for_job,
    mark_logical_run_stage_completed,
    start_or_resume_logical_run,
)


class _FakeDB:
    def __init__(self) -> None:
        self.database = "pdx"
        self._engine = create_engine("sqlite:///:memory:")
        with self._engine.begin() as conn:
            _ = conn.execute(
                text(
                    """
                    CREATE TABLE pipeline_job_runs (
                        logical_run_id TEXT PRIMARY KEY,
                        job_name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        current_stage TEXT NULL,
                        dagster_run_id TEXT NULL,
                        config_json TEXT NULL,
                        scope_size INTEGER NOT NULL,
                        started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        finished_at TEXT NULL
                    )
                    """
                )
            )
            _ = conn.execute(
                text(
                    """
                    CREATE TABLE pipeline_job_run_agreements (
                        logical_run_id TEXT NOT NULL,
                        agreement_uuid TEXT NOT NULL,
                        scope_position INTEGER NOT NULL,
                        PRIMARY KEY (logical_run_id, agreement_uuid)
                    )
                    """
                )
            )

    def get_engine(self):  # type: ignore[reportUnknownParameterType]
        return self._engine


def _pipeline_config(*, resume_logical_runs: bool = True, force_new_logical_run: bool = False) -> PipelineConfig:
    return cast(
        PipelineConfig,
        cast(
            object,
            SimpleNamespace(
                resume_logical_runs=resume_logical_runs,
                force_new_logical_run=force_new_logical_run,
            ),
        ),
    )


class LogicalJobRunTests(unittest.TestCase):
    def test_start_creates_new_logical_run_with_persisted_scope(self) -> None:
        db = _FakeDB()
        context = SimpleNamespace(run_id="dagster-1")

        logical_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_a",
            initial_stage="ingestion_cleanup_a_tagging",
            selected_agreement_uuids=["agreement-2", "agreement-1", "agreement-1"],
        )

        self.assertIsNotNone(logical_run)
        assert logical_run is not None
        self.assertFalse(logical_run.resumed_existing)
        self.assertEqual(logical_run.agreement_uuids, ["agreement-1", "agreement-2"])

        active_run = load_active_logical_run(
            db=cast(DBResource, cast(object, db)),
            job_name="ingestion_cleanup_a",
        )
        self.assertIsNotNone(active_run)
        assert active_run is not None
        self.assertEqual(active_run["status"], LOGICAL_RUN_STATUS_RUNNING)

        persisted_scope = load_active_scope_for_job(
            cast(AssetExecutionContext, cast(object, context)),
            db=cast(DBResource, cast(object, db)),
            job_name="ingestion_cleanup_a",
            fallback_agreement_uuids=["drifted-agreement"],
        )
        self.assertEqual(persisted_scope, ["agreement-1", "agreement-2"])

    def test_start_resumes_existing_run_instead_of_recomputing_scope(self) -> None:
        db = _FakeDB()
        first_context = SimpleNamespace(run_id="dagster-1")
        second_context = SimpleNamespace(run_id="dagster-2")

        first_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, first_context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_b",
            initial_stage="ingestion_cleanup_b_ai_repair_enqueue",
            selected_agreement_uuids=["agreement-1", "agreement-2"],
        )
        resumed_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, second_context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_b",
            initial_stage="ingestion_cleanup_b_ai_repair_enqueue",
            selected_agreement_uuids=["agreement-3"],
        )

        self.assertIsNotNone(first_run)
        self.assertIsNotNone(resumed_run)
        assert first_run is not None
        assert resumed_run is not None
        self.assertTrue(resumed_run.resumed_existing)
        self.assertEqual(resumed_run.logical_run_id, first_run.logical_run_id)
        self.assertEqual(resumed_run.agreement_uuids, ["agreement-1", "agreement-2"])

    def test_force_new_logical_run_abandons_old_run_and_creates_new_one(self) -> None:
        db = _FakeDB()
        context = SimpleNamespace(run_id="dagster-1")

        first_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="regular_ingest",
            initial_stage="regular_ingest_pre_processing",
            selected_agreement_uuids=["agreement-1"],
        )
        second_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(force_new_logical_run=True),
            job_name="regular_ingest",
            initial_stage="regular_ingest_pre_processing",
            selected_agreement_uuids=["agreement-2"],
        )

        self.assertIsNotNone(first_run)
        self.assertIsNotNone(second_run)
        assert first_run is not None
        assert second_run is not None
        self.assertNotEqual(first_run.logical_run_id, second_run.logical_run_id)
        self.assertEqual(second_run.agreement_uuids, ["agreement-2"])

        with db.get_engine().begin() as conn:
            statuses = conn.execute(
                text(
                    """
                    SELECT logical_run_id, status
                    FROM pipeline_job_runs
                    WHERE job_name = :job_name
                    ORDER BY logical_run_id ASC
                    """
                ),
                {"job_name": "regular_ingest"},
            ).mappings().all()

        status_by_run_id = {str(row["logical_run_id"]): str(row["status"]) for row in statuses}
        self.assertEqual(status_by_run_id[first_run.logical_run_id], LOGICAL_RUN_STATUS_ABANDONED)
        self.assertEqual(status_by_run_id[second_run.logical_run_id], LOGICAL_RUN_STATUS_RUNNING)

    def test_mark_completed_updates_stage_and_final_status(self) -> None:
        db = _FakeDB()
        context = SimpleNamespace(run_id="dagster-1")

        logical_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_a",
            initial_stage="ingestion_cleanup_a_tagging",
            selected_agreement_uuids=["agreement-1"],
        )
        self.assertIsNotNone(logical_run)

        mark_logical_run_stage_completed(
            db=cast(DBResource, cast(object, db)),
            job_name="ingestion_cleanup_a",
            stage_name="ingestion_cleanup_a_tx_metadata_web_search",
            complete_run=True,
        )

        active_run = load_active_logical_run(
            db=cast(DBResource, cast(object, db)),
            job_name="ingestion_cleanup_a",
        )
        self.assertIsNone(active_run)

        with db.get_engine().begin() as conn:
            final_row = conn.execute(
                text(
                    """
                    SELECT status, current_stage, finished_at
                    FROM pipeline_job_runs
                    WHERE job_name = :job_name
                    """
                ),
                {"job_name": "ingestion_cleanup_a"},
            ).mappings().first()

        self.assertIsNotNone(final_row)
        assert final_row is not None
        self.assertEqual(final_row["status"], LOGICAL_RUN_STATUS_COMPLETED)
        self.assertEqual(final_row["current_stage"], "ingestion_cleanup_a_tx_metadata_web_search")
        self.assertIsNotNone(final_row["finished_at"])


if __name__ == "__main__":
    _ = unittest.main()
