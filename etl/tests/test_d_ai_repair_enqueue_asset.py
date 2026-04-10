# pyright: reportAny=false, reportPrivateUsage=false
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import ANY, patch

from dagster import AssetExecutionContext

from etl.defs.d_ai_repair_asset import (
    _enqueue_ai_repair_for_agreements,
    ai_repair_enqueue_asset,
    regular_ingest_ai_repair_poll_asset,
    regular_ingest_ai_repair_enqueue_asset,
)
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


class AIRepairEnqueueAssetTests(unittest.TestCase):
    def test_enqueue_attaches_toc_only_for_pure_section_non_sequential_candidates(self) -> None:
        context = SimpleNamespace(log=_FakeLog())
        db = _FakeDB()
        pipeline_config = cast(
            PipelineConfig,
            cast(
                object,
                SimpleNamespace(
                    xml_agreement_batch_size=25,
                    resume_openai_batches=False,
                    ai_repair_attempt_priority=AIRepairAttemptPriority.NOT_ATTEMPTED_FIRST,
                ),
            ),
        )

        build_calls: list[dict[str, object]] = []

        class _FakeFiles:
            def create(self, *, purpose: str, file: object) -> object:
                _ = purpose
                _ = file
                return SimpleNamespace(id="file-1")

        class _FakeBatches:
            def create(self, *, input_file_id: str, endpoint: str, completion_window: str) -> object:
                _ = input_file_id
                _ = endpoint
                _ = completion_window
                return SimpleNamespace(id="batch-1", status="queued")

        fake_client = SimpleNamespace(files=_FakeFiles(), batches=_FakeBatches())

        def _fake_build_jsonl_lines_for_page(**kwargs: object) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
            build_calls.append(dict(kwargs))
            page_uuid = cast(str, kwargs["page_uuid"])
            xml_version = cast(int, kwargs["xml_version"])
            return (
                [{"custom_id": f"{page_uuid}::full::{xml_version}", "body": {"input": []}}],
                [{"request_id": f"{page_uuid}::full::{xml_version}", "page_uuid": page_uuid, "mode": "full", "excerpt_start": 0, "excerpt_end": 1}],
            )

        candidates = [
            {
                "page_uuid": "page-gap",
                "agreement_uuid": "agreement-1",
                "text": "5.22 Customers and Suppliers.",
                "ai_repair_attempted": 0,
                "has_completed_requests": 0,
                "xml_version": 4,
                "page_order": 1,
            },
            {
                "page_uuid": "page-mixed",
                "agreement_uuid": "agreement-1",
                "text": "5.23 Accounts Receivable and Payable; Loans.",
                "ai_repair_attempted": 0,
                "has_completed_requests": 0,
                "xml_version": 4,
                "page_order": 2,
            },
        ]

        with (
            patch("etl.defs.d_ai_repair_asset.assert_tables_exist"),
            patch("etl.defs.d_ai_repair_asset._fetch_candidates", return_value=candidates),
            patch(
                "etl.defs.d_ai_repair_asset._fetch_candidate_page_reason_codes",
                return_value={
                    "page-gap": {"section_non_sequential"},
                    "page-mixed": {"section_non_sequential", "section_article_mismatch"},
                },
            ),
            patch(
                "etl.defs.d_ai_repair_asset._fetch_latest_xml_text_by_agreement",
                return_value={"agreement-1": "<document />"},
            ),
            patch(
                "etl.defs.d_ai_repair_asset._build_toc_context_for_page",
                return_value="Article 5 TOC section numbering: 5.20, 5.21, 5.22, 5.23",
            ) as build_toc_context,
            patch("etl.defs.d_ai_repair_asset.build_jsonl_lines_for_page", side_effect=_fake_build_jsonl_lines_for_page),
            patch("etl.defs.d_ai_repair_asset._oai_client", return_value=fake_client),
            patch("etl.defs.d_ai_repair_asset._insert_batch_row"),
            patch("etl.defs.d_ai_repair_asset._insert_requests"),
            patch("etl.defs.d_ai_repair_asset._mark_xml_ai_repair_attempted", return_value=1),
            patch("etl.defs.d_ai_repair_asset.run_post_asset_refresh"),
        ):
            result = _enqueue_ai_repair_for_agreements(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                pipeline_config,
                target_agreement_uuids=None,
                log_prefix="ai_repair_enqueue_asset",
            )

        self.assertEqual(result, ["agreement-1"])
        self.assertEqual(len(build_calls), 2)
        self.assertEqual(build_calls[0]["toc_context"], "Article 5 TOC section numbering: 5.20, 5.21, 5.22, 5.23")
        self.assertIsNone(build_calls[1]["toc_context"])
        self.assertEqual(build_toc_context.call_count, 1)

    def test_enqueue_omits_toc_when_numbering_candidate_has_no_parseable_toc(self) -> None:
        context = SimpleNamespace(log=_FakeLog())
        db = _FakeDB()
        pipeline_config = cast(
            PipelineConfig,
            cast(
                object,
                SimpleNamespace(
                    xml_agreement_batch_size=25,
                    resume_openai_batches=False,
                    ai_repair_attempt_priority=AIRepairAttemptPriority.NOT_ATTEMPTED_FIRST,
                ),
            ),
        )

        build_calls: list[dict[str, object]] = []

        class _FakeFiles:
            def create(self, *, purpose: str, file: object) -> object:
                _ = purpose
                _ = file
                return SimpleNamespace(id="file-1")

        class _FakeBatches:
            def create(self, *, input_file_id: str, endpoint: str, completion_window: str) -> object:
                _ = input_file_id
                _ = endpoint
                _ = completion_window
                return SimpleNamespace(id="batch-1", status="queued")

        fake_client = SimpleNamespace(files=_FakeFiles(), batches=_FakeBatches())

        def _fake_build_jsonl_lines_for_page(**kwargs: object) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
            build_calls.append(dict(kwargs))
            page_uuid = cast(str, kwargs["page_uuid"])
            xml_version = cast(int, kwargs["xml_version"])
            return (
                [{"custom_id": f"{page_uuid}::full::{xml_version}", "body": {"input": []}}],
                [{"request_id": f"{page_uuid}::full::{xml_version}", "page_uuid": page_uuid, "mode": "full", "excerpt_start": 0, "excerpt_end": 1}],
            )

        candidates = [
            {
                "page_uuid": "page-gap",
                "agreement_uuid": "agreement-1",
                "text": "5.22 Customers and Suppliers.",
                "ai_repair_attempted": 0,
                "has_completed_requests": 0,
                "xml_version": 4,
                "page_order": 1,
            },
        ]

        with (
            patch("etl.defs.d_ai_repair_asset.assert_tables_exist"),
            patch("etl.defs.d_ai_repair_asset._fetch_candidates", return_value=candidates),
            patch(
                "etl.defs.d_ai_repair_asset._fetch_candidate_page_reason_codes",
                return_value={"page-gap": {"section_non_sequential"}},
            ),
            patch(
                "etl.defs.d_ai_repair_asset._fetch_latest_xml_text_by_agreement",
                return_value={"agreement-1": "<document />"},
            ),
            patch("etl.defs.d_ai_repair_asset._build_toc_context_for_page", return_value=None),
            patch("etl.defs.d_ai_repair_asset.build_jsonl_lines_for_page", side_effect=_fake_build_jsonl_lines_for_page),
            patch("etl.defs.d_ai_repair_asset._oai_client", return_value=fake_client),
            patch("etl.defs.d_ai_repair_asset._insert_batch_row"),
            patch("etl.defs.d_ai_repair_asset._insert_requests"),
            patch("etl.defs.d_ai_repair_asset._mark_xml_ai_repair_attempted", return_value=1),
            patch("etl.defs.d_ai_repair_asset.run_post_asset_refresh"),
        ):
            _ = _enqueue_ai_repair_for_agreements(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                pipeline_config,
                target_agreement_uuids=None,
                log_prefix="ai_repair_enqueue_asset",
            )

        self.assertEqual(len(build_calls), 1)
        self.assertIsNone(build_calls[0]["toc_context"])

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
            patch("etl.defs.d_ai_repair_asset.should_skip_managed_stage", return_value=(False, None)),
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
            patch("etl.defs.d_ai_repair_asset.should_skip_managed_stage", return_value=(False, None)),
            patch(
                "etl.defs.d_ai_repair_asset._oai_client",
                side_effect=AssertionError("resume path should not create an OpenAI client"),
            ),
            patch("etl.defs.d_ai_repair_asset.assert_tables_exist"),
            patch(
                "etl.defs.d_ai_repair_asset.load_active_scope_for_job",
                side_effect=_fallback_scope,
            ),
            patch("etl.defs.d_ai_repair_asset.load_active_logical_run", return_value=None),
            patch("etl.defs.d_ai_repair_asset.mark_logical_run_stage_completed", return_value=None),
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

    def test_regular_ingest_enqueue_noops_for_explicit_empty_scope(self) -> None:
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
            patch("etl.defs.d_ai_repair_asset.should_skip_managed_stage", return_value=(False, None)),
            patch("etl.defs.d_ai_repair_asset.load_active_scope_for_job", return_value=[]),
            patch("etl.defs.d_ai_repair_asset.load_active_logical_run", return_value=None),
            patch("etl.defs.d_ai_repair_asset.mark_logical_run_stage_completed", return_value=None),
            patch(
                "etl.defs.d_ai_repair_asset._fetch_open_ai_repair_batch_for_scope",
                side_effect=AssertionError("empty scoped run should not inspect repair batches"),
            ),
            patch(
                "etl.defs.d_ai_repair_asset._fetch_candidates",
                side_effect=AssertionError("empty scoped run should not select repair candidates"),
            ),
            patch("etl.defs.d_ai_repair_asset.run_post_asset_refresh", return_value=None),
        ):
            enqueue_fn = getattr(regular_ingest_ai_repair_enqueue_asset.op.compute_fn, "decorated_fn")
            result = enqueue_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                pipeline_config,
                [],
            )

        self.assertEqual(result, [])

    def test_regular_ingest_enqueue_uses_full_explicit_scope_limit(self) -> None:
        context = SimpleNamespace(log=_FakeLog())
        db = _FakeDB()
        pipeline_config = cast(
            PipelineConfig,
            cast(
                object,
                SimpleNamespace(
                    xml_agreement_batch_size=10,
                    resume_openai_batches=False,
                    ai_repair_attempt_priority=AIRepairAttemptPriority.NOT_ATTEMPTED_FIRST,
                ),
            ),
        )
        scoped_uuids = [f"agreement-{idx:02d}" for idx in range(12)]

        with (
            patch("etl.defs.d_ai_repair_asset.assert_tables_exist"),
            patch(
                "etl.defs.d_ai_repair_asset._fetch_candidates",
                return_value=[],
            ) as fetch_candidates,
            patch("etl.defs.d_ai_repair_asset.run_post_asset_refresh", return_value=None),
        ):
            result = _enqueue_ai_repair_for_agreements(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                pipeline_config,
                target_agreement_uuids=scoped_uuids,
                log_prefix="regular_ingest_ai_repair_enqueue_asset",
            )

        self.assertEqual(result, [])
        fetch_candidates.assert_called_once_with(
            ANY,
            "pdx",
            agreement_limit=12,
            target_agreement_uuids=scoped_uuids,
            page_budget=None,
            attempt_priority=AIRepairAttemptPriority.NOT_ATTEMPTED_FIRST,
        )

    def test_regular_ingest_enqueue_skips_when_stage_already_completed(self) -> None:
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
                "etl.defs.d_ai_repair_asset.should_skip_managed_stage",
                return_value=(True, "regular_ingest_ai_repair_poll"),
            ),
            patch(
                "etl.defs.d_ai_repair_asset._enqueue_ai_repair_for_agreements",
                side_effect=AssertionError("regular_ingest enqueue should skip when already completed"),
            ),
        ):
            enqueue_fn = getattr(regular_ingest_ai_repair_enqueue_asset.op.compute_fn, "decorated_fn")
            result = enqueue_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                pipeline_config,
                ["agreement-1"],
            )

        self.assertEqual(result, [])

    def test_regular_ingest_poll_skips_when_stage_already_completed(self) -> None:
        context = SimpleNamespace(log=_FakeLog())
        db = _FakeDB()
        pipeline_config = cast(PipelineConfig, cast(object, SimpleNamespace()))

        with (
            patch(
                "etl.defs.d_ai_repair_asset.should_skip_managed_stage",
                return_value=(True, "regular_ingest_reconcile_tags"),
            ),
            patch(
                "etl.defs.d_ai_repair_asset._poll_ai_repair_batches",
                side_effect=AssertionError("regular_ingest poll should skip when already completed"),
            ),
        ):
            poll_fn = getattr(regular_ingest_ai_repair_poll_asset.op.compute_fn, "decorated_fn")
            result = poll_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                pipeline_config,
                ["agreement-1"],
            )

        self.assertEqual(result, [])


if __name__ == "__main__":
    _ = unittest.main()
