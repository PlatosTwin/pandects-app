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
    LOGICAL_RUN_STATUS_FAILED,
    LOGICAL_RUN_STATUS_RUNNING,
    MANAGED_JOB_STAGE_SEQUENCE,
    load_active_logical_run,
    load_active_scope_for_job,
    mark_logical_run_failed,
    mark_logical_run_stage_completed,
    normalize_managed_stage_name,
    should_skip_managed_stage,
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
                        completed_stages_json TEXT NULL,
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


class _FakeDagsterRun:
    def __init__(self, *, is_finished: bool, run_id: str | None = None, root_run_id: str | None = None) -> None:
        self.is_finished = is_finished
        self.run_id = run_id
        self.root_run_id = root_run_id


class _FakeInstance:
    """Minimal stand-in for DagsterInstance liveness lookups."""

    def __init__(
        self,
        *,
        finished_run_ids: set[str],
        alive_run_ids: set[str],
        root_run_ids: dict[str, str] | None = None,
    ) -> None:
        self._finished_run_ids = finished_run_ids
        self._alive_run_ids = alive_run_ids
        self._root_run_ids = root_run_ids or {}

    def get_run_by_id(self, run_id: str):  # type: ignore[reportUnknownParameterType]
        root_run_id = self._root_run_ids.get(run_id)
        if run_id in self._finished_run_ids:
            return _FakeDagsterRun(is_finished=True, run_id=run_id, root_run_id=root_run_id)
        if run_id in self._alive_run_ids:
            return _FakeDagsterRun(is_finished=False, run_id=run_id, root_run_id=root_run_id)
        return None


class _FakeLog:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def warning(self, message: str, *args: object) -> None:
        self.warnings.append(message % args if args else message)

    def info(self, message: str, *args: object) -> None:
        self.infos.append(message % args if args else message)


def _reexecution_context(
    run_id: str,
    *,
    parent_run_id: str,
    root_run_id: str,
    instance: _FakeInstance | None = None,
    from_failure: bool = True,
) -> SimpleNamespace:
    tags = {"dagster/parent_run_id": parent_run_id, "dagster/root_run_id": root_run_id}
    if from_failure:
        tags["dagster/is_resume_retry"] = "true"
    return SimpleNamespace(
        run_id=run_id,
        run=SimpleNamespace(run_id=run_id, parent_run_id=parent_run_id, root_run_id=root_run_id, tags=tags),
        instance=instance,
        log=_FakeLog(),
    )


def _owner_for_run(db: "_FakeDB", logical_run_id: str) -> str:
    with db.get_engine().begin() as conn:
        row = conn.execute(
            text("SELECT dagster_run_id FROM pipeline_job_runs WHERE logical_run_id = :id"),
            {"id": logical_run_id},
        ).scalar()
    return str(row)


def _status_for_run(db: "_FakeDB", logical_run_id: str) -> str:
    with db.get_engine().begin() as conn:
        row = conn.execute(
            text("SELECT status FROM pipeline_job_runs WHERE logical_run_id = :id"),
            {"id": logical_run_id},
        ).scalar()
    return str(row)


class LogicalJobRunTests(unittest.TestCase):
    def test_orphaned_running_run_reconciled_to_failed_when_worker_dead(self) -> None:
        db = _FakeDB()
        first_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, SimpleNamespace(run_id="dagster-1"))),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_a",
            initial_stage="ingestion_cleanup_a_tagging",
            selected_agreement_uuids=["agreement-1"],
        )
        assert first_run is not None

        # Worker for dagster-1 is dead; with resume disabled a fresh run is created,
        # and the orphaned RUNNING row must no longer look active.
        log = _FakeLog()
        second_context = SimpleNamespace(
            run_id="dagster-2",
            instance=_FakeInstance(finished_run_ids={"dagster-1"}, alive_run_ids=set()),
            log=log,
        )
        second_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, second_context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(resume_logical_runs=False),
            job_name="ingestion_cleanup_a",
            initial_stage="ingestion_cleanup_a_tagging",
            selected_agreement_uuids=["agreement-9"],
        )
        assert second_run is not None

        self.assertFalse(second_run.resumed_existing)
        self.assertNotEqual(second_run.logical_run_id, first_run.logical_run_id)
        self.assertEqual(_status_for_run(db, first_run.logical_run_id), LOGICAL_RUN_STATUS_FAILED)
        self.assertEqual(_status_for_run(db, second_run.logical_run_id), LOGICAL_RUN_STATUS_RUNNING)
        self.assertTrue(any("reconciled orphaned logical run" in line for line in log.warnings))

        # The reconciled (now active) run is the new one only.
        active_run = load_active_logical_run(
            db=cast(DBResource, cast(object, db)),
            job_name="ingestion_cleanup_a",
        )
        assert active_run is not None
        self.assertEqual(active_run["logical_run_id"], second_run.logical_run_id)

    def test_running_run_with_live_worker_is_not_reconciled(self) -> None:
        db = _FakeDB()
        first_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, SimpleNamespace(run_id="dagster-1"))),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_a",
            initial_stage="ingestion_cleanup_a_tagging",
            selected_agreement_uuids=["agreement-1"],
        )
        assert first_run is not None

        second_context = SimpleNamespace(
            run_id="dagster-2",
            instance=_FakeInstance(finished_run_ids=set(), alive_run_ids={"dagster-1"}),
            log=_FakeLog(),
        )
        _ = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, second_context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(resume_logical_runs=False),
            job_name="ingestion_cleanup_a",
            initial_stage="ingestion_cleanup_a_tagging",
            selected_agreement_uuids=["agreement-9"],
        )

        # A live previous worker must never be demoted.
        self.assertEqual(_status_for_run(db, first_run.logical_run_id), LOGICAL_RUN_STATUS_RUNNING)

    def test_resume_with_dead_worker_still_resumes_and_logs_banner(self) -> None:
        db = _FakeDB()
        first_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, SimpleNamespace(run_id="dagster-1"))),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_a",
            initial_stage="ingestion_cleanup_a_tagging",
            selected_agreement_uuids=["agreement-1", "agreement-2"],
        )
        assert first_run is not None

        log = _FakeLog()
        second_context = SimpleNamespace(
            run_id="dagster-2",
            instance=_FakeInstance(finished_run_ids={"dagster-1"}, alive_run_ids=set()),
            log=log,
        )
        resumed_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, second_context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_a",
            initial_stage="ingestion_cleanup_a_tagging",
            selected_agreement_uuids=["agreement-9"],
        )
        assert resumed_run is not None

        self.assertTrue(resumed_run.resumed_existing)
        self.assertEqual(resumed_run.logical_run_id, first_run.logical_run_id)
        self.assertEqual(resumed_run.agreement_uuids, ["agreement-1", "agreement-2"])
        self.assertEqual(_status_for_run(db, first_run.logical_run_id), LOGICAL_RUN_STATUS_RUNNING)
        self.assertTrue(any("RESUMING existing logical run" in line for line in log.warnings))
        self.assertTrue(any("no longer alive" in line for line in log.warnings))

    def test_resume_within_same_dagster_run_does_not_warn_about_concurrent_run(self) -> None:
        db = _FakeDB()
        first_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, SimpleNamespace(run_id="dagster-1"))),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_a",
            initial_stage="ingestion_cleanup_a_tagging",
            selected_agreement_uuids=["agreement-1", "agreement-2"],
        )
        assert first_run is not None

        # Step one of the new Dagster run resumes the failed logical run and
        # re-points it at "dagster-2".
        first_step_log = _FakeLog()
        first_step_context = SimpleNamespace(
            run_id="dagster-2",
            instance=_FakeInstance(finished_run_ids={"dagster-1"}, alive_run_ids=set()),
            log=first_step_log,
        )
        resumed_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, first_step_context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_a",
            initial_stage="ingestion_cleanup_a_tagging",
            selected_agreement_uuids=["agreement-9"],
        )
        assert resumed_run is not None
        self.assertTrue(resumed_run.resumed_existing)
        self.assertTrue(any("RESUMING existing logical run" in line for line in first_step_log.warnings))

        # A later step of the same Dagster run sees its own run id as the
        # logical run's worker; that is not a concurrent run.
        second_step_log = _FakeLog()
        second_step_context = SimpleNamespace(
            run_id="dagster-2",
            instance=_FakeInstance(finished_run_ids={"dagster-1"}, alive_run_ids={"dagster-2"}),
            log=second_step_log,
        )
        same_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, second_step_context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_a",
            initial_stage="ingestion_cleanup_a_tagging",
            selected_agreement_uuids=["agreement-9"],
        )
        assert same_run is not None
        self.assertEqual(same_run.logical_run_id, first_run.logical_run_id)
        self.assertEqual(same_run.agreement_uuids, ["agreement-1", "agreement-2"])
        self.assertEqual(second_step_log.warnings, [])
        self.assertTrue(
            any("already started or resumed earlier in this Dagster run" in line for line in second_step_log.infos)
        )
        self.assertFalse(any("concurrent" in line for line in second_step_log.infos))

    def test_later_step_of_fresh_dagster_run_does_not_warn(self) -> None:
        db = _FakeDB()
        first_step_context = SimpleNamespace(
            run_id="dagster-1",
            instance=_FakeInstance(finished_run_ids=set(), alive_run_ids={"dagster-1"}),
            log=_FakeLog(),
        )
        first_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, first_step_context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_a",
            initial_stage="ingestion_cleanup_a_tagging",
            selected_agreement_uuids=["agreement-1"],
        )
        assert first_run is not None
        self.assertFalse(first_run.resumed_existing)

        second_step_log = _FakeLog()
        second_step_context = SimpleNamespace(
            run_id="dagster-1",
            instance=_FakeInstance(finished_run_ids=set(), alive_run_ids={"dagster-1"}),
            log=second_step_log,
        )
        same_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, second_step_context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_a",
            initial_stage="ingestion_cleanup_a_tagging",
            selected_agreement_uuids=["agreement-1"],
        )
        assert same_run is not None
        self.assertEqual(same_run.logical_run_id, first_run.logical_run_id)
        self.assertEqual(second_step_log.warnings, [])
        self.assertTrue(
            any("already started or resumed earlier in this Dagster run" in line for line in second_step_log.infos)
        )

    def test_failed_run_from_same_dagster_run_still_logs_full_banner(self) -> None:
        db = _FakeDB()
        first_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, SimpleNamespace(run_id="dagster-1"))),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_a",
            initial_stage="ingestion_cleanup_a_tagging",
            selected_agreement_uuids=["agreement-1"],
        )
        assert first_run is not None
        mark_logical_run_failed(
            db=cast(DBResource, cast(object, db)),
            job_name="ingestion_cleanup_a",
            stage_name="ingestion_cleanup_a_tagging",
            dagster_run_id="dagster-1",
        )
        self.assertEqual(_status_for_run(db, first_run.logical_run_id), LOGICAL_RUN_STATUS_FAILED)

        # A later step of the same Dagster run picks the FAILED row back up;
        # the operator should see the full banner, but not a concurrent-run warning.
        log = _FakeLog()
        later_step_context = SimpleNamespace(
            run_id="dagster-1",
            instance=_FakeInstance(finished_run_ids=set(), alive_run_ids={"dagster-1"}),
            log=log,
        )
        resumed_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, later_step_context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_a",
            initial_stage="ingestion_cleanup_a_tagging",
            selected_agreement_uuids=["agreement-1"],
        )
        assert resumed_run is not None
        self.assertTrue(any("RESUMING existing logical run" in line for line in log.warnings))
        self.assertTrue(any("prior status: FAILED" in line for line in log.warnings))
        self.assertTrue(any("this same Dagster run" in line for line in log.warnings))
        self.assertFalse(any("concurrent" in line for line in log.warnings))


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
        self.assertEqual(active_run["completed_stages"], [])

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

    def test_start_creates_new_cleanup_c_run_with_sections_stage(self) -> None:
        db = _FakeDB()
        context = SimpleNamespace(run_id="dagster-c-1")

        logical_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_c",
            initial_stage="ingestion_cleanup_c_build_xml",
            selected_agreement_uuids=["agreement-2", "agreement-1"],
        )

        self.assertIsNotNone(logical_run)
        assert logical_run is not None
        self.assertEqual(logical_run.agreement_uuids, ["agreement-1", "agreement-2"])

        persisted_scope = load_active_scope_for_job(
            cast(AssetExecutionContext, cast(object, context)),
            db=cast(DBResource, cast(object, db)),
            job_name="ingestion_cleanup_c",
            fallback_agreement_uuids=[],
        )
        self.assertEqual(persisted_scope, ["agreement-1", "agreement-2"])

    def test_force_new_logical_run_abandons_old_run_and_creates_new_one(self) -> None:
        db = _FakeDB()
        first_context = SimpleNamespace(run_id="dagster-1")
        context = SimpleNamespace(run_id="dagster-2")

        first_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, first_context)),
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

    def test_force_new_logical_run_reuses_run_started_earlier_in_same_dagster_run(self) -> None:
        db = _FakeDB()
        old_context = SimpleNamespace(run_id="dagster-old")
        context = SimpleNamespace(run_id="dagster-1")

        old_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, old_context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="regular_ingest",
            initial_stage="regular_ingest_staging",
            selected_agreement_uuids=["agreement-old"],
        )
        assert old_run is not None

        staging_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(force_new_logical_run=True),
            job_name="regular_ingest",
            initial_stage="regular_ingest_staging",
            selected_agreement_uuids=["agreement-1", "agreement-2"],
        )
        assert staging_run is not None
        mark_logical_run_stage_completed(
            db=cast(DBResource, cast(object, db)),
            job_name="regular_ingest",
            stage_name="regular_ingest_staging",
        )

        pre_processing_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(force_new_logical_run=True),
            job_name="regular_ingest",
            initial_stage="regular_ingest_pre_processing",
            selected_agreement_uuids=["agreement-1", "agreement-2", "agreement-3"],
        )

        assert pre_processing_run is not None
        self.assertNotEqual(staging_run.logical_run_id, old_run.logical_run_id)
        self.assertEqual(pre_processing_run.logical_run_id, staging_run.logical_run_id)
        self.assertEqual(pre_processing_run.agreement_uuids, ["agreement-1", "agreement-2"])
        self.assertEqual(pre_processing_run.completed_stages, ["regular_ingest_staging"])
        self.assertEqual(_status_for_run(db, old_run.logical_run_id), LOGICAL_RUN_STATUS_ABANDONED)
        self.assertEqual(_status_for_run(db, staging_run.logical_run_id), LOGICAL_RUN_STATUS_RUNNING)
        with db.get_engine().begin() as conn:
            run_count = conn.execute(text("SELECT COUNT(*) FROM pipeline_job_runs")).scalar()
        self.assertEqual(run_count, 2)

    def test_resume_disabled_still_continues_run_started_earlier_in_same_dagster_run(self) -> None:
        db = _FakeDB()
        context = SimpleNamespace(run_id="dagster-1")

        first_step_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(resume_logical_runs=False),
            job_name="regular_ingest",
            initial_stage="regular_ingest_staging",
            selected_agreement_uuids=["agreement-1"],
        )
        later_step_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(resume_logical_runs=False),
            job_name="regular_ingest",
            initial_stage="regular_ingest_pre_processing",
            selected_agreement_uuids=["agreement-1"],
        )

        assert first_step_run is not None
        assert later_step_run is not None
        self.assertEqual(later_step_run.logical_run_id, first_step_run.logical_run_id)
        with db.get_engine().begin() as conn:
            run_count = conn.execute(text("SELECT COUNT(*) FROM pipeline_job_runs")).scalar()
        self.assertEqual(run_count, 1)

    def test_force_new_reexecution_from_failure_continues_parent_logical_run(self) -> None:
        db = _FakeDB()
        original_context = SimpleNamespace(run_id="dagster-1")

        original_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, original_context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(force_new_logical_run=True),
            job_name="regular_ingest",
            initial_stage="regular_ingest_staging",
            selected_agreement_uuids=["agreement-1", "agreement-2"],
        )
        assert original_run is not None
        mark_logical_run_stage_completed(
            db=cast(DBResource, cast(object, db)),
            job_name="regular_ingest",
            stage_name="regular_ingest_staging",
        )
        mark_logical_run_failed(
            db=cast(DBResource, cast(object, db)),
            job_name="regular_ingest",
            stage_name="regular_ingest_pre_processing",
            dagster_run_id="dagster-1",
        )

        retry_context = _reexecution_context(
            "dagster-2",
            parent_run_id="dagster-1",
            root_run_id="dagster-1",
            instance=_FakeInstance(finished_run_ids={"dagster-1"}, alive_run_ids={"dagster-2"}),
        )
        retried_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, retry_context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(force_new_logical_run=True),
            job_name="regular_ingest",
            initial_stage="regular_ingest_pre_processing",
            selected_agreement_uuids=["agreement-3"],
        )

        assert retried_run is not None
        self.assertEqual(retried_run.logical_run_id, original_run.logical_run_id)
        self.assertEqual(retried_run.completed_stages, ["regular_ingest_staging"])
        self.assertEqual(retried_run.agreement_uuids, ["agreement-1", "agreement-2"])
        self.assertEqual(_status_for_run(db, original_run.logical_run_id), LOGICAL_RUN_STATUS_RUNNING)
        self.assertEqual(_owner_for_run(db, original_run.logical_run_id), "dagster-2")
        warnings = cast(_FakeLog, retry_context.log).warnings
        self.assertEqual(len(warnings), 1)
        self.assertIn("re-executes the run that owned it", warnings[0])
        with db.get_engine().begin() as conn:
            run_count = conn.execute(text("SELECT COUNT(*) FROM pipeline_job_runs")).scalar()
        self.assertEqual(run_count, 1)

    def test_force_new_sibling_reexecution_continues_logical_run_by_root(self) -> None:
        db = _FakeDB()
        first_retry_context = _reexecution_context("dagster-2", parent_run_id="dagster-1", root_run_id="dagster-1")

        # The first retry left the logical run FAILED and owned by itself.
        logical_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, first_retry_context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(force_new_logical_run=True),
            job_name="regular_ingest",
            initial_stage="regular_ingest_pre_processing",
            selected_agreement_uuids=["agreement-1"],
        )
        assert logical_run is not None
        mark_logical_run_failed(
            db=cast(DBResource, cast(object, db)),
            job_name="regular_ingest",
            stage_name="regular_ingest_tagging",
            dagster_run_id="dagster-2",
        )

        # A second retry launched from the original run shares only the root.
        second_retry_context = _reexecution_context(
            "dagster-3",
            parent_run_id="dagster-1",
            root_run_id="dagster-1",
            instance=_FakeInstance(
                finished_run_ids={"dagster-1", "dagster-2"},
                alive_run_ids={"dagster-3"},
                root_run_ids={"dagster-2": "dagster-1", "dagster-3": "dagster-1"},
            ),
        )
        retried_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, second_retry_context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(force_new_logical_run=True),
            job_name="regular_ingest",
            initial_stage="regular_ingest_tagging",
            selected_agreement_uuids=["agreement-9"],
        )

        assert retried_run is not None
        self.assertEqual(retried_run.logical_run_id, logical_run.logical_run_id)
        self.assertEqual(_status_for_run(db, logical_run.logical_run_id), LOGICAL_RUN_STATUS_RUNNING)

    def test_force_new_reexecute_all_steps_starts_fresh(self) -> None:
        db = _FakeDB()
        original_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, SimpleNamespace(run_id="dagster-1"))),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(force_new_logical_run=True),
            job_name="regular_ingest",
            initial_stage="regular_ingest_staging",
            selected_agreement_uuids=["agreement-1"],
        )
        assert original_run is not None
        mark_logical_run_stage_completed(
            db=cast(DBResource, cast(object, db)),
            job_name="regular_ingest",
            stage_name="regular_ingest_staging",
        )
        mark_logical_run_failed(
            db=cast(DBResource, cast(object, db)),
            job_name="regular_ingest",
            stage_name="regular_ingest_tagging",
            dagster_run_id="dagster-1",
        )

        rerun_context = _reexecution_context(
            "dagster-2",
            parent_run_id="dagster-1",
            root_run_id="dagster-1",
            instance=_FakeInstance(finished_run_ids={"dagster-1"}, alive_run_ids={"dagster-2"}),
            from_failure=False,
        )
        rerun = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, rerun_context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(force_new_logical_run=True),
            job_name="regular_ingest",
            initial_stage="regular_ingest_staging",
            selected_agreement_uuids=["agreement-2"],
        )

        assert rerun is not None
        self.assertNotEqual(rerun.logical_run_id, original_run.logical_run_id)
        self.assertEqual(rerun.completed_stages, [])
        self.assertEqual(_status_for_run(db, original_run.logical_run_id), LOGICAL_RUN_STATUS_ABANDONED)
        should_skip, _ = should_skip_managed_stage(
            db=cast(DBResource, cast(object, db)),
            job_name="regular_ingest",
            stage_name="regular_ingest_staging",
        )
        self.assertFalse(should_skip)

    def test_reexecution_from_failure_does_not_take_over_live_sibling_retry(self) -> None:
        db = _FakeDB()
        sibling_context = _reexecution_context("dagster-2", parent_run_id="dagster-1", root_run_id="dagster-1")
        sibling_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, sibling_context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(force_new_logical_run=True),
            job_name="regular_ingest",
            initial_stage="regular_ingest_pre_processing",
            selected_agreement_uuids=["agreement-1"],
        )
        assert sibling_run is not None

        retry_context = _reexecution_context(
            "dagster-3",
            parent_run_id="dagster-1",
            root_run_id="dagster-1",
            instance=_FakeInstance(
                finished_run_ids={"dagster-1"},
                alive_run_ids={"dagster-2", "dagster-3"},
                root_run_ids={"dagster-2": "dagster-1", "dagster-3": "dagster-1"},
            ),
        )
        retried_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, retry_context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(resume_logical_runs=False),
            job_name="regular_ingest",
            initial_stage="regular_ingest_pre_processing",
            selected_agreement_uuids=["agreement-2"],
        )

        assert retried_run is not None
        self.assertNotEqual(retried_run.logical_run_id, sibling_run.logical_run_id)
        self.assertEqual(_owner_for_run(db, sibling_run.logical_run_id), "dagster-2")

    def test_active_scope_revives_failed_run_for_reexecution_from_failure(self) -> None:
        db = _FakeDB()
        original_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, SimpleNamespace(run_id="dagster-1"))),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(force_new_logical_run=True),
            job_name="regular_ingest",
            initial_stage="regular_ingest_staging",
            selected_agreement_uuids=["agreement-1", "agreement-2"],
        )
        assert original_run is not None
        for stage_name in ("regular_ingest_staging", "regular_ingest_pre_processing"):
            mark_logical_run_stage_completed(
                db=cast(DBResource, cast(object, db)),
                job_name="regular_ingest",
                stage_name=stage_name,
            )
        mark_logical_run_failed(
            db=cast(DBResource, cast(object, db)),
            job_name="regular_ingest",
            stage_name="regular_ingest_tagging",
            dagster_run_id="dagster-1",
        )

        # The retry starts at tagging, past the steps that call start_or_resume.
        retry_context = _reexecution_context(
            "dagster-2",
            parent_run_id="dagster-1",
            root_run_id="dagster-1",
            instance=_FakeInstance(finished_run_ids={"dagster-1"}, alive_run_ids={"dagster-2"}),
        )
        scope = load_active_scope_for_job(
            cast(AssetExecutionContext, cast(object, retry_context)),
            db=cast(DBResource, cast(object, db)),
            job_name="regular_ingest",
            fallback_agreement_uuids=[],
        )

        self.assertEqual(scope, ["agreement-1", "agreement-2"])
        self.assertEqual(_status_for_run(db, original_run.logical_run_id), LOGICAL_RUN_STATUS_RUNNING)
        self.assertEqual(_owner_for_run(db, original_run.logical_run_id), "dagster-2")
        with db.get_engine().begin() as conn:
            finished_at = conn.execute(
                text("SELECT finished_at FROM pipeline_job_runs WHERE logical_run_id = :id"),
                {"id": original_run.logical_run_id},
            ).scalar()
        self.assertIsNone(finished_at)
        should_skip, _ = should_skip_managed_stage(
            db=cast(DBResource, cast(object, db)),
            job_name="regular_ingest",
            stage_name="regular_ingest_pre_processing",
        )
        self.assertTrue(should_skip)
        mark_logical_run_stage_completed(
            db=cast(DBResource, cast(object, db)),
            job_name="regular_ingest",
            stage_name="regular_ingest_tagging",
        )
        active_run = load_active_logical_run(db=cast(DBResource, cast(object, db)), job_name="regular_ingest")
        assert active_run is not None
        self.assertEqual(active_run["logical_run_id"], original_run.logical_run_id)
        self.assertIn("regular_ingest_tagging", active_run["completed_stages"])

    def test_inherited_asset_resume_retry_tag_does_not_count_as_resume(self) -> None:
        db = _FakeDB()
        original_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, SimpleNamespace(run_id="dagster-1"))),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(force_new_logical_run=True),
            job_name="ingestion_cleanup_a",
            initial_stage="ingestion_cleanup_a_tagging",
            selected_agreement_uuids=["agreement-1"],
        )
        assert original_run is not None
        mark_logical_run_failed(
            db=cast(DBResource, cast(object, db)),
            job_name="ingestion_cleanup_a",
            stage_name="ingestion_cleanup_a_tagging",
            dagster_run_id="dagster-1",
        )

        # A re-execute-all of a from-asset-failure retry inherits that retry's tag.
        rerun_context = _reexecution_context(
            "dagster-3",
            parent_run_id="dagster-1",
            root_run_id="dagster-1",
            instance=_FakeInstance(finished_run_ids={"dagster-1"}, alive_run_ids={"dagster-3"}),
            from_failure=False,
        )
        rerun_context.run.tags["dagster/is_asset_resume_retry"] = "true"
        rerun = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, rerun_context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(force_new_logical_run=True),
            job_name="ingestion_cleanup_a",
            initial_stage="ingestion_cleanup_a_tagging",
            selected_agreement_uuids=["agreement-2"],
        )

        assert rerun is not None
        self.assertNotEqual(rerun.logical_run_id, original_run.logical_run_id)
        self.assertEqual(_status_for_run(db, original_run.logical_run_id), LOGICAL_RUN_STATUS_ABANDONED)

    def test_mark_failed_does_not_fail_logical_run_owned_by_live_dagster_run(self) -> None:
        db = _FakeDB()
        owner_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, SimpleNamespace(run_id="dagster-1"))),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_c",
            initial_stage="ingestion_cleanup_c_sections",
            selected_agreement_uuids=["agreement-1"],
        )
        assert owner_run is not None

        mark_logical_run_failed(
            db=cast(DBResource, cast(object, db)),
            job_name="ingestion_cleanup_c",
            stage_name="ingestion_cleanup_c_sections",
            dagster_run_id="dagster-2",
            instance=_FakeInstance(finished_run_ids=set(), alive_run_ids={"dagster-1", "dagster-2"}),
        )

        self.assertEqual(_status_for_run(db, owner_run.logical_run_id), LOGICAL_RUN_STATUS_RUNNING)

    def test_force_new_reexecution_does_not_adopt_unrelated_failed_run(self) -> None:
        db = _FakeDB()
        unrelated_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, SimpleNamespace(run_id="dagster-other"))),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="regular_ingest",
            initial_stage="regular_ingest_staging",
            selected_agreement_uuids=["agreement-1"],
        )
        assert unrelated_run is not None
        mark_logical_run_failed(
            db=cast(DBResource, cast(object, db)),
            job_name="regular_ingest",
            stage_name="regular_ingest_staging",
            dagster_run_id="dagster-other",
        )

        retry_context = _reexecution_context(
            "dagster-2",
            parent_run_id="dagster-1",
            root_run_id="dagster-1",
            instance=_FakeInstance(
                finished_run_ids={"dagster-other", "dagster-1"},
                alive_run_ids={"dagster-2"},
                root_run_ids={"dagster-other": "dagster-other"},
            ),
        )
        retried_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, retry_context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(force_new_logical_run=True),
            job_name="regular_ingest",
            initial_stage="regular_ingest_staging",
            selected_agreement_uuids=["agreement-2"],
        )

        assert retried_run is not None
        self.assertNotEqual(retried_run.logical_run_id, unrelated_run.logical_run_id)
        self.assertEqual(_status_for_run(db, unrelated_run.logical_run_id), LOGICAL_RUN_STATUS_ABANDONED)

    def test_active_scope_does_not_take_over_run_owned_by_live_dagster_run(self) -> None:
        db = _FakeDB()
        owner_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, SimpleNamespace(run_id="dagster-1"))),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_c",
            initial_stage="ingestion_cleanup_c_sections",
            selected_agreement_uuids=["agreement-1", "agreement-2"],
        )
        assert owner_run is not None

        log = _FakeLog()
        other_context = SimpleNamespace(
            run_id="dagster-2",
            instance=_FakeInstance(finished_run_ids=set(), alive_run_ids={"dagster-1", "dagster-2"}),
            log=log,
        )
        scope = load_active_scope_for_job(
            cast(AssetExecutionContext, cast(object, other_context)),
            db=cast(DBResource, cast(object, db)),
            job_name="ingestion_cleanup_c",
            fallback_agreement_uuids=["agreement-9"],
        )

        self.assertEqual(scope, ["agreement-9"])
        self.assertEqual(_owner_for_run(db, owner_run.logical_run_id), "dagster-1")
        self.assertEqual(len(log.warnings), 1)

    def test_active_scope_adopts_run_whose_dagster_run_is_gone(self) -> None:
        db = _FakeDB()
        owner_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, SimpleNamespace(run_id="dagster-1"))),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_c",
            initial_stage="ingestion_cleanup_c_sections",
            selected_agreement_uuids=["agreement-1", "agreement-2"],
        )
        assert owner_run is not None

        other_context = SimpleNamespace(
            run_id="dagster-2",
            instance=_FakeInstance(finished_run_ids={"dagster-1"}, alive_run_ids={"dagster-2"}),
            log=_FakeLog(),
        )
        scope = load_active_scope_for_job(
            cast(AssetExecutionContext, cast(object, other_context)),
            db=cast(DBResource, cast(object, db)),
            job_name="ingestion_cleanup_c",
            fallback_agreement_uuids=["agreement-9"],
        )

        self.assertEqual(scope, ["agreement-1", "agreement-2"])
        self.assertEqual(_owner_for_run(db, owner_run.logical_run_id), "dagster-2")

    def test_start_resumes_failed_run_instead_of_starting_fresh(self) -> None:
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
        assert first_run is not None

        mark_logical_run_failed(
            db=cast(DBResource, cast(object, db)),
            job_name="ingestion_cleanup_b",
            stage_name="ingestion_cleanup_b_post_repair_verify_xml",
            dagster_run_id="dagster-1",
        )

        resumed_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, second_context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_b",
            initial_stage="ingestion_cleanup_b_ai_repair_enqueue",
            selected_agreement_uuids=["agreement-3"],
        )

        self.assertIsNotNone(resumed_run)
        assert resumed_run is not None
        self.assertTrue(resumed_run.resumed_existing)
        self.assertEqual(resumed_run.logical_run_id, first_run.logical_run_id)
        self.assertEqual(resumed_run.agreement_uuids, ["agreement-1", "agreement-2"])
        self.assertEqual(resumed_run.status, LOGICAL_RUN_STATUS_RUNNING)

        active_run = load_active_logical_run(
            db=cast(DBResource, cast(object, db)),
            job_name="ingestion_cleanup_b",
        )
        self.assertIsNotNone(active_run)
        assert active_run is not None
        self.assertEqual(active_run["logical_run_id"], first_run.logical_run_id)
        self.assertEqual(active_run["status"], LOGICAL_RUN_STATUS_RUNNING)

    def test_regular_ingest_staging_is_tracked_as_completed_stage(self) -> None:
        db = _FakeDB()
        context = SimpleNamespace(run_id="dagster-staging-1")

        logical_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="regular_ingest",
            initial_stage="regular_ingest_staging",
            selected_agreement_uuids=["agreement-1"],
        )

        self.assertIsNotNone(logical_run)
        assert logical_run is not None
        self.assertEqual(MANAGED_JOB_STAGE_SEQUENCE["regular_ingest"][0], "regular_ingest_staging")

        mark_logical_run_stage_completed(
            db=cast(DBResource, cast(object, db)),
            job_name="regular_ingest",
            stage_name="regular_ingest_staging",
        )

        should_skip, current_stage = should_skip_managed_stage(
            db=cast(DBResource, cast(object, db)),
            job_name="regular_ingest",
            stage_name="regular_ingest_staging",
        )
        self.assertTrue(should_skip)
        self.assertEqual(current_stage, "regular_ingest_staging")

    def test_force_new_logical_run_abandons_failed_run_and_creates_new_one(self) -> None:
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
        assert first_run is not None

        mark_logical_run_failed(
            db=cast(DBResource, cast(object, db)),
            job_name="regular_ingest",
            stage_name="regular_ingest_post_repair_verify_xml",
            dagster_run_id="dagster-1",
        )

        second_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(force_new_logical_run=True),
            job_name="regular_ingest",
            initial_stage="regular_ingest_pre_processing",
            selected_agreement_uuids=["agreement-2"],
        )

        self.assertIsNotNone(second_run)
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
                    SELECT status, current_stage, finished_at, completed_stages_json
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
        self.assertEqual(final_row["completed_stages_json"], '["ingestion_cleanup_a_tx_metadata_web_search"]')

    def test_mark_failed_updates_status_and_stage_for_matching_dagster_run(self) -> None:
        db = _FakeDB()
        context = SimpleNamespace(run_id="dagster-1")

        logical_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_b",
            initial_stage="ingestion_cleanup_b_ai_repair_enqueue",
            selected_agreement_uuids=["agreement-1"],
        )
        self.assertIsNotNone(logical_run)

        mark_logical_run_failed(
            db=cast(DBResource, cast(object, db)),
            job_name="ingestion_cleanup_b",
            stage_name="ingestion_cleanup_b_post_repair_verify_xml",
            dagster_run_id="dagster-1",
        )

        active_run = load_active_logical_run(
            db=cast(DBResource, cast(object, db)),
            job_name="ingestion_cleanup_b",
        )
        self.assertIsNone(active_run)

        with db.get_engine().begin() as conn:
            final_row = conn.execute(
                text(
                    """
                    SELECT status, current_stage, finished_at
                    , completed_stages_json
                    FROM pipeline_job_runs
                    WHERE job_name = :job_name
                    """
                ),
                {"job_name": "ingestion_cleanup_b"},
            ).mappings().first()

        self.assertIsNotNone(final_row)
        assert final_row is not None
        self.assertEqual(final_row["status"], LOGICAL_RUN_STATUS_FAILED)
        self.assertEqual(final_row["current_stage"], "ingestion_cleanup_b_post_repair_verify_xml")
        self.assertIsNotNone(final_row["finished_at"])
        self.assertEqual(final_row["completed_stages_json"], "[]")

    def test_normalize_managed_stage_name_handles_step_keys_and_asset_suffixes(self) -> None:
        self.assertEqual(
            normalize_managed_stage_name("05_14_ingestion_cleanup_b_post_repair_verify_xml"),
            "ingestion_cleanup_b_post_repair_verify_xml",
        )
        self.assertEqual(
            normalize_managed_stage_name("10_06_ingestion_cleanup_b_tx_metadata_web_search_asset"),
            "ingestion_cleanup_b_tx_metadata_web_search",
        )
        self.assertEqual(
            normalize_managed_stage_name("09_regular_ingest_taxonomy_gold_backfill_asset"),
            "regular_ingest_taxonomy_gold_backfill",
        )

    def test_managed_stage_sequences_include_xml_build_stages(self) -> None:
        self.assertIn("regular_ingest_build_xml", MANAGED_JOB_STAGE_SEQUENCE["regular_ingest"])
        self.assertIn("ingestion_cleanup_a_build_xml", MANAGED_JOB_STAGE_SEQUENCE["ingestion_cleanup_a"])
        self.assertIn("regular_ingest_verify_xml", MANAGED_JOB_STAGE_SEQUENCE["regular_ingest"])
        self.assertIn("ingestion_cleanup_a_verify_xml", MANAGED_JOB_STAGE_SEQUENCE["ingestion_cleanup_a"])
        self.assertEqual(MANAGED_JOB_STAGE_SEQUENCE["ingestion_cleanup_d"][0], "ingestion_cleanup_d_sections")

    def test_mark_completed_does_not_regress_managed_stage_progress(self) -> None:
        db = _FakeDB()
        context = SimpleNamespace(run_id="dagster-1")

        logical_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_b",
            initial_stage="ingestion_cleanup_b_ai_repair_enqueue",
            selected_agreement_uuids=["agreement-1"],
        )
        self.assertIsNotNone(logical_run)

        mark_logical_run_stage_completed(
            db=cast(DBResource, cast(object, db)),
            job_name="ingestion_cleanup_b",
            stage_name="ingestion_cleanup_b_post_repair_verify_xml",
        )
        mark_logical_run_stage_completed(
            db=cast(DBResource, cast(object, db)),
            job_name="ingestion_cleanup_b",
            stage_name="ingestion_cleanup_b_ai_repair_enqueue",
        )

        active_run = load_active_logical_run(
            db=cast(DBResource, cast(object, db)),
            job_name="ingestion_cleanup_b",
        )
        self.assertIsNotNone(active_run)
        assert active_run is not None
        self.assertEqual(active_run["current_stage"], "ingestion_cleanup_b_post_repair_verify_xml")
        self.assertEqual(
            active_run["completed_stages"],
            ["ingestion_cleanup_b_ai_repair_enqueue", "ingestion_cleanup_b_post_repair_verify_xml"],
        )

    def test_should_skip_managed_stage_for_earlier_cleanup_b_stage_after_failure(self) -> None:
        db = _FakeDB()
        context = SimpleNamespace(run_id="dagster-1")

        logical_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_b",
            initial_stage="ingestion_cleanup_b_ai_repair_enqueue",
            selected_agreement_uuids=["agreement-1"],
        )
        self.assertIsNotNone(logical_run)

        mark_logical_run_stage_completed(
            db=cast(DBResource, cast(object, db)),
            job_name="ingestion_cleanup_b",
            stage_name="ingestion_cleanup_b_ai_repair_enqueue",
        )
        mark_logical_run_failed(
            db=cast(DBResource, cast(object, db)),
            job_name="ingestion_cleanup_b",
            stage_name="ingestion_cleanup_b_post_repair_verify_xml",
            dagster_run_id="dagster-1",
        )
        _ = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, SimpleNamespace(run_id="dagster-2"))),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="ingestion_cleanup_b",
            initial_stage="ingestion_cleanup_b_ai_repair_enqueue",
            selected_agreement_uuids=["agreement-2"],
        )

        should_skip_enqueue, current_stage_enqueue = should_skip_managed_stage(
            db=cast(DBResource, cast(object, db)),
            job_name="ingestion_cleanup_b",
            stage_name="ingestion_cleanup_b_ai_repair_enqueue",
        )
        self.assertTrue(should_skip_enqueue)
        self.assertEqual(current_stage_enqueue, "ingestion_cleanup_b_post_repair_verify_xml")

        should_skip_verify, current_stage_verify = should_skip_managed_stage(
            db=cast(DBResource, cast(object, db)),
            job_name="ingestion_cleanup_b",
            stage_name="ingestion_cleanup_b_post_repair_verify_xml",
        )
        self.assertFalse(should_skip_verify)
        self.assertEqual(current_stage_verify, "ingestion_cleanup_b_post_repair_verify_xml")

    def test_should_not_skip_uncompleted_parallel_branch_stage(self) -> None:
        db = _FakeDB()
        context = SimpleNamespace(run_id="dagster-1")

        logical_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="regular_ingest",
            initial_stage="regular_ingest_pre_processing",
            selected_agreement_uuids=["agreement-1"],
        )
        self.assertIsNotNone(logical_run)

        mark_logical_run_stage_completed(
            db=cast(DBResource, cast(object, db)),
            job_name="regular_ingest",
            stage_name="regular_ingest_sections_from_fresh_xml",
        )

        should_skip_fresh, current_stage_fresh = should_skip_managed_stage(
            db=cast(DBResource, cast(object, db)),
            job_name="regular_ingest",
            stage_name="regular_ingest_sections_from_fresh_xml",
        )
        self.assertTrue(should_skip_fresh)
        self.assertEqual(current_stage_fresh, "regular_ingest_sections_from_fresh_xml")

        should_skip_repair, current_stage_repair = should_skip_managed_stage(
            db=cast(DBResource, cast(object, db)),
            job_name="regular_ingest",
            stage_name="regular_ingest_sections_from_repair_xml",
        )
        self.assertFalse(should_skip_repair)
        self.assertEqual(current_stage_repair, "regular_ingest_sections_from_fresh_xml")

    def test_should_skip_earlier_stage_for_legacy_run_without_completed_stage_history(self) -> None:
        db = _FakeDB()
        context = SimpleNamespace(run_id="dagster-legacy-1")

        logical_run = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, context)),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="regular_ingest",
            initial_stage="regular_ingest_pre_processing",
            selected_agreement_uuids=["agreement-1"],
        )
        self.assertIsNotNone(logical_run)

        mark_logical_run_failed(
            db=cast(DBResource, cast(object, db)),
            job_name="regular_ingest",
            stage_name="regular_ingest_tx_metadata_offline",
            dagster_run_id="dagster-legacy-1",
        )
        _ = start_or_resume_logical_run(
            cast(AssetExecutionContext, cast(object, SimpleNamespace(run_id="dagster-legacy-2"))),
            db=cast(DBResource, cast(object, db)),
            pipeline_config=_pipeline_config(),
            job_name="regular_ingest",
            initial_stage="regular_ingest_staging",
            selected_agreement_uuids=[],
        )

        should_skip_staging, current_stage_staging = should_skip_managed_stage(
            db=cast(DBResource, cast(object, db)),
            job_name="regular_ingest",
            stage_name="regular_ingest_staging",
        )
        self.assertTrue(should_skip_staging)
        self.assertEqual(current_stage_staging, "regular_ingest_tx_metadata_offline")

        should_skip_current, current_stage_current = should_skip_managed_stage(
            db=cast(DBResource, cast(object, db)),
            job_name="regular_ingest",
            stage_name="regular_ingest_tx_metadata_offline",
        )
        self.assertFalse(should_skip_current)
        self.assertEqual(current_stage_current, "regular_ingest_tx_metadata_offline")


if __name__ == "__main__":
    _ = unittest.main()
