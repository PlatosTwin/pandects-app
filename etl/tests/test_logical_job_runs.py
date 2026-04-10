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


if __name__ == "__main__":
    _ = unittest.main()
