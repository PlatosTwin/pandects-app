"""Scoped regular_ingest resume guards."""
# pyright: reportAny=false

import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import ANY, patch

from dagster import AssetExecutionContext

from etl.defs.a_staging_asset import regular_ingest_staging_asset
from etl.defs.b_pre_processing_asset import regular_ingest_pre_processing_asset
from etl.defs.c_tagging_asset import regular_ingest_tagging_asset
from etl.defs.f_xml_asset import regular_ingest_xml_asset, regular_ingest_xml_verify_asset
from etl.defs.f_xml_repair_cycle_asset import (
    regular_ingest_post_repair_build_xml_asset,
    regular_ingest_post_repair_verify_xml_asset,
)
from etl.defs.resources import DBResource, PipelineConfig


class _FakeLog:
    def info(self, *_args: object, **_kwargs: object) -> None:
        return None

    def warning(self, *_args: object, **_kwargs: object) -> None:
        return None


class _FakeResult:
    def __init__(self, *, rows: list[dict[str, object]] | None = None) -> None:
        self._rows = rows or []

    class _Scalars:
        def __init__(self, values: list[object]) -> None:
            self._values = values

        def all(self) -> list[object]:
            return self._values

    def mappings(self) -> "_FakeResult":
        return self

    def fetchall(self) -> list[dict[str, object]]:
        return self._rows

    def scalars(self) -> "_FakeResult._Scalars":
        values = [next(iter(row.values()), None) for row in self._rows]
        return _FakeResult._Scalars(values)


class _FakeConn:
    def execute(self, statement: object, _params: dict[str, object] | None = None) -> _FakeResult:
        sql = str(statement)
        if "AND status = 'verified'" in sql:
            return _FakeResult(rows=[])
        if "latest ai_repair_attempted XML rows" in sql:
            return _FakeResult(rows=[])
        if "latest=1, and ai_repair_attempted=0" in sql:
            return _FakeResult(rows=[])
        if "canonical_post_repair_build_queue_sql" in sql:
            return _FakeResult(rows=[])
        return _FakeResult(rows=[])


class _FakeBeginContext:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def __enter__(self) -> _FakeConn:
        return self._conn

    def __exit__(self, *_exc: object) -> None:
        return None


class _FakeEngine:
    def __init__(self) -> None:
        self._conn = _FakeConn()

    def begin(self) -> _FakeBeginContext:
        return _FakeBeginContext(self._conn)


class _FakeDB:
    def __init__(self) -> None:
        self.database = "pdx"
        self._engine = _FakeEngine()

    def get_engine(self) -> _FakeEngine:
        return self._engine


def _fallback_scope(
    _context: object,
    *,
    db: object,
    job_name: str,
    fallback_agreement_uuids: list[str],
) -> list[str]:
    _ = db
    _ = job_name
    return list(fallback_agreement_uuids)


class RegularIngestScopeResumeTests(unittest.TestCase):
    def test_regular_ingest_staging_skips_when_stage_already_completed(self) -> None:
        db = _FakeDB()
        context = SimpleNamespace(log=_FakeLog())
        pipeline_config = PipelineConfig()

        with (
            patch(
                "etl.defs.a_staging_asset.start_or_resume_logical_run",
                return_value=SimpleNamespace(agreement_uuids=["agreement-1"], resumed_existing=True),
            ),
            patch(
                "etl.defs.a_staging_asset.should_skip_managed_stage",
                return_value=(True, "regular_ingest_tx_metadata_offline"),
            ),
            patch(
                "etl.defs.a_staging_asset._run_staging",
                side_effect=AssertionError("regular_ingest_staging_asset should not restage a resumed run"),
            ),
        ):
            decorated_fn = getattr(cast(object, regular_ingest_staging_asset.op.compute_fn), "decorated_fn")
            result = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                pipeline_config,
            )

        self.assertEqual(result, ["agreement-1"])

    def test_regular_ingest_pre_processing_skips_pre_gating_for_empty_scope(self) -> None:
        db = _FakeDB()
        context = SimpleNamespace(log=_FakeLog())
        pipeline_config = PipelineConfig()
        classifier_model = cast(object, SimpleNamespace())
        review_model = cast(object, SimpleNamespace())

        with (
            patch("etl.defs.b_pre_processing_asset.is_pre_processing_cleanup_mode", return_value=False),
            patch("etl.defs.b_pre_processing_asset.start_or_resume_logical_run", return_value=None),
            patch("etl.defs.b_pre_processing_asset.should_skip_managed_stage", return_value=(False, None)),
            patch(
                "etl.defs.b_pre_processing_asset.run_pre_asset_gating",
                side_effect=AssertionError("regular_ingest_pre_processing_asset should not gate an empty staged scope"),
            ),
            patch("etl.defs.b_pre_processing_asset.mark_logical_run_stage_completed", return_value=None) as mark_stage,
        ):
            decorated_fn = getattr(cast(object, regular_ingest_pre_processing_asset.op.compute_fn), "decorated_fn")
            result = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                classifier_model,
                review_model,
                pipeline_config,
                [],
            )

        self.assertEqual(result, [])
        mark_stage.assert_called_once_with(
            db=ANY,
            job_name="regular_ingest",
            stage_name="regular_ingest_pre_processing",
        )

    def test_regular_ingest_pre_processing_skips_completed_stage(self) -> None:
        db = _FakeDB()
        context = SimpleNamespace(log=_FakeLog())
        pipeline_config = PipelineConfig()
        classifier_model = cast(object, SimpleNamespace())
        review_model = cast(object, SimpleNamespace())

        with (
            patch("etl.defs.b_pre_processing_asset.is_pre_processing_cleanup_mode", return_value=False),
            patch(
                "etl.defs.b_pre_processing_asset.start_or_resume_logical_run",
                return_value=SimpleNamespace(agreement_uuids=["agreement-1"], resumed_existing=True),
            ),
            patch(
                "etl.defs.b_pre_processing_asset.should_skip_managed_stage",
                return_value=(True, "regular_ingest_tx_metadata_offline"),
            ),
            patch(
                "etl.defs.b_pre_processing_asset._run_pre_processing_from_scratch",
                side_effect=AssertionError("regular_ingest_pre_processing_asset should skip completed stage work"),
            ),
        ):
            decorated_fn = getattr(cast(object, regular_ingest_pre_processing_asset.op.compute_fn), "decorated_fn")
            result = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                classifier_model,
                review_model,
                pipeline_config,
                ["agreement-1"],
            )

        self.assertEqual(result, ["agreement-1"])

    def test_regular_ingest_tagging_skips_completed_stage(self) -> None:
        db = _FakeDB()
        context = SimpleNamespace(log=_FakeLog())
        pipeline_config = PipelineConfig()
        tagging_model = cast(object, SimpleNamespace())

        with (
            patch("etl.defs.c_tagging_asset.load_active_scope_for_job", return_value=["agreement-1"]),
            patch(
                "etl.defs.c_tagging_asset.should_skip_managed_stage",
                return_value=(True, "regular_ingest_tx_metadata_offline"),
            ),
            patch(
                "etl.defs.c_tagging_asset._run_tagging_for_agreements",
                side_effect=AssertionError("regular_ingest_tagging_asset should skip completed stage work"),
            ),
        ):
            decorated_fn = getattr(cast(object, regular_ingest_tagging_asset.op.compute_fn), "decorated_fn")
            result = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                tagging_model,
                pipeline_config,
                ["agreement-1"],
            )

        self.assertEqual(result, ["agreement-1"])

    def test_regular_ingest_xml_build_skips_completed_stage(self) -> None:
        db = _FakeDB()
        context = SimpleNamespace(log=_FakeLog())
        pipeline_config = PipelineConfig()

        with (
            patch("etl.defs.f_xml_asset.load_active_scope_for_job", return_value=["agreement-1"]),
            patch(
                "etl.defs.f_xml_asset.should_skip_managed_stage",
                return_value=(True, "regular_ingest_tx_metadata_offline"),
            ),
            patch(
                "etl.defs.f_xml_asset._run_xml_build_for_agreements",
                side_effect=AssertionError("regular_ingest_xml_asset should skip completed stage work"),
            ),
        ):
            decorated_fn = getattr(cast(object, regular_ingest_xml_asset.op.compute_fn), "decorated_fn")
            result = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                pipeline_config,
                ["agreement-1"],
            )

        self.assertEqual(result, ["agreement-1"])

    def test_regular_ingest_xml_verify_does_not_resume_unrelated_stranded_batch(self) -> None:
        db = _FakeDB()
        context = SimpleNamespace(log=_FakeLog())
        pipeline_config = cast(
            PipelineConfig,
            cast(object, SimpleNamespace(xml_agreement_batch_size=10, resume_openai_batches=True)),
        )

        with (
            patch("etl.defs.f_xml_asset._oai_client", return_value=SimpleNamespace()),
            patch("etl.defs.f_xml_asset.assert_tables_exist", return_value=None),
            patch(
                "etl.defs.f_xml_asset.load_active_scope_for_job",
                side_effect=_fallback_scope,
            ),
            patch("etl.defs.f_xml_asset.load_active_logical_run", return_value=None),
            patch("etl.defs.f_xml_asset.should_skip_managed_stage", return_value=(False, None)),
            patch("etl.defs.f_xml_asset.mark_logical_run_stage_completed", return_value=None),
            patch(
                "etl.defs.f_xml_asset._fetch_unpulled_xml_verify_batch",
                side_effect=AssertionError("regular_ingest_xml_verify_asset should not inspect unrelated stranded batches"),
            ),
            patch("etl.defs.f_xml_asset.run_post_asset_refresh", return_value=None),
        ):
            decorated_fn = getattr(cast(object, regular_ingest_xml_verify_asset.op.compute_fn), "decorated_fn")
            result = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                pipeline_config,
                ["agreement-1"],
            )

        self.assertEqual(result, [])

    def test_regular_ingest_post_repair_build_does_not_defer_for_unrelated_verify_batch(self) -> None:
        db = _FakeDB()
        context = SimpleNamespace(log=_FakeLog())
        pipeline_config = cast(
            PipelineConfig,
            cast(object, SimpleNamespace(xml_agreement_batch_size=10, resume_openai_batches=True)),
        )

        with (
            patch(
                "etl.defs.f_xml_repair_cycle_asset.load_active_scope_for_job",
                side_effect=_fallback_scope,
            ),
            patch("etl.defs.f_xml_repair_cycle_asset.mark_logical_run_stage_completed", return_value=None),
            patch(
                "etl.defs.f_xml_repair_cycle_asset._fetch_unpulled_xml_verify_batch",
                side_effect=AssertionError("regular_ingest_post_repair_build_xml_asset should not be blocked by unrelated verify batches"),
            ),
            patch("etl.defs.f_xml_repair_cycle_asset.run_post_asset_refresh", return_value=None),
        ):
            decorated_fn = getattr(cast(object, regular_ingest_post_repair_build_xml_asset.op.compute_fn), "decorated_fn")
            result = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                pipeline_config,
                ["agreement-1"],
            )

        self.assertEqual(result, [])

    def test_regular_ingest_post_repair_verify_does_not_resume_unrelated_stranded_batch(self) -> None:
        db = _FakeDB()
        context = SimpleNamespace(log=_FakeLog())
        pipeline_config = cast(
            PipelineConfig,
            cast(object, SimpleNamespace(xml_agreement_batch_size=10, resume_openai_batches=True)),
        )

        with (
            patch("etl.defs.f_xml_repair_cycle_asset._oai_client", return_value=SimpleNamespace()),
            patch("etl.defs.f_xml_repair_cycle_asset.assert_tables_exist", return_value=None),
            patch(
                "etl.defs.f_xml_repair_cycle_asset.load_active_scope_for_job",
                side_effect=_fallback_scope,
            ),
            patch("etl.defs.f_xml_repair_cycle_asset.load_active_logical_run", return_value=None),
            patch("etl.defs.f_xml_repair_cycle_asset.mark_logical_run_stage_completed", return_value=None),
            patch(
                "etl.defs.f_xml_repair_cycle_asset._fetch_unpulled_xml_verify_batch",
                side_effect=AssertionError("regular_ingest_post_repair_verify_xml_asset should not inspect unrelated stranded batches"),
            ),
            patch("etl.defs.f_xml_repair_cycle_asset.run_post_asset_refresh", return_value=None),
        ):
            decorated_fn = getattr(cast(object, regular_ingest_post_repair_verify_xml_asset.op.compute_fn), "decorated_fn")
            result = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                pipeline_config,
                ["agreement-1"],
            )

        self.assertEqual(result, [])


if __name__ == "__main__":
    _ = unittest.main()
