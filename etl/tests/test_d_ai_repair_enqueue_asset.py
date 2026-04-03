# pyright: reportAny=false, reportPrivateUsage=false
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import ANY, patch

from dagster import AssetExecutionContext

from etl.defs.d_ai_repair_asset import ai_repair_enqueue_asset, regular_ingest_ai_repair_enqueue_asset
from etl.defs.resources import (
    AIRepairAttemptPriority,
    DBResource,
    PipelineConfig,
)


class _FakeBeginContext:
    def __enter__(self) -> "_FakeConn":
        return _FakeConn()

    def __exit__(self, *_exc: object) -> None:
        return None


class _FakeEngine:
    def begin(self) -> _FakeBeginContext:
        return _FakeBeginContext()


class _FakeConn:
    pass


class _FakeLog:
    def info(self, *_args: object, **_kwargs: object) -> None:
        return None

    def warning(self, *_args: object, **_kwargs: object) -> None:
        return None

    def error(self, *_args: object, **_kwargs: object) -> None:
        return None


class _FakeDB:
    def __init__(self) -> None:
        self.database = "pdx"
        self._engine = _FakeEngine()

    def get_engine(self) -> _FakeEngine:
        return self._engine


class AIRepairEnqueueAssetTests(unittest.TestCase):
    def test_resume_prefers_oldest_stranded_batch_before_new_candidates(self) -> None:
        context = SimpleNamespace(log=_FakeLog())
        db = _FakeDB()
        pipeline_config = cast(
            PipelineConfig,
            cast(
                object,
                SimpleNamespace(
                    xml_agreement_batch_size=25,
                    resume_openai_batches=True,
                    ai_repair_attempt_priority=AIRepairAttemptPriority.NOT_ATTEMPTED_FIRST,
                ),
            ),
        )

        with (
            patch(
                "etl.defs.d_ai_repair_asset._oai_client",
                side_effect=AssertionError("resume path should not create an OpenAI client"),
            ),
            patch("etl.defs.d_ai_repair_asset.assert_tables_exist"),
            patch(
                "etl.defs.d_ai_repair_asset._fetch_open_ai_repair_batch",
                return_value={"batch_id": "batch-old"},
            ) as fetch_open_batch,
            patch(
                "etl.defs.d_ai_repair_asset._fetch_batch_agreement_uuids",
                return_value=["agreement-old"],
            ) as fetch_batch_agreements,
            patch("etl.defs.d_ai_repair_asset._fetch_candidates") as fetch_candidates,
            patch("etl.defs.d_ai_repair_asset.run_post_asset_refresh"),
        ):
            enqueue_fn = getattr(ai_repair_enqueue_asset.op.compute_fn, "decorated_fn")
            result = enqueue_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                pipeline_config,
            )

        self.assertEqual(result, ["agreement-old"])
        fetch_open_batch.assert_called_once()
        fetch_batch_agreements.assert_called_once()
        fetch_candidates.assert_not_called()

    def test_regular_ingest_resume_only_checks_scope_specific_batch_keys(self) -> None:
        context = SimpleNamespace(log=_FakeLog())
        db = _FakeDB()
        pipeline_config = cast(
            PipelineConfig,
            cast(
                object,
                SimpleNamespace(
                    xml_agreement_batch_size=25,
                    resume_openai_batches=True,
                    ai_repair_attempt_priority=AIRepairAttemptPriority.NOT_ATTEMPTED_FIRST,
                ),
            ),
        )

        with (
            patch(
                "etl.defs.d_ai_repair_asset._oai_client",
                side_effect=AssertionError("resume path should not create an OpenAI client"),
            ),
            patch("etl.defs.d_ai_repair_asset.assert_tables_exist"),
            patch(
                "etl.defs.d_ai_repair_asset._fetch_open_ai_repair_batch_for_scope",
                return_value={"batch_id": "batch-scope"},
            ) as fetch_scoped_batch,
            patch(
                "etl.defs.d_ai_repair_asset._fetch_open_ai_repair_batch",
                side_effect=AssertionError("regular_ingest should not inspect unrelated global batches"),
            ),
            patch(
                "etl.defs.d_ai_repair_asset._fetch_batch_agreement_uuids",
                return_value=["agreement-1"],
            ) as fetch_batch_agreements,
            patch("etl.defs.d_ai_repair_asset._fetch_candidates") as fetch_candidates,
            patch("etl.defs.d_ai_repair_asset.run_post_asset_refresh"),
        ):
            enqueue_fn = getattr(regular_ingest_ai_repair_enqueue_asset.op.compute_fn, "decorated_fn")
            result = enqueue_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                pipeline_config,
                ["agreement-1"],
            )

        self.assertEqual(result, ["agreement-1"])
        fetch_scoped_batch.assert_called_once_with(
            ANY,
            "pdx",
            scoped_agreement_uuids=["agreement-1"],
        )
        fetch_batch_agreements.assert_called_once()
        fetch_candidates.assert_not_called()


if __name__ == "__main__":
    _ = unittest.main()
