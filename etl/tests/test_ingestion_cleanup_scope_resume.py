"""Scoped ingestion cleanup resume and batch-key guards."""
# pyright: reportAny=false

import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import ANY, patch

from dagster import AssetExecutionContext

from etl.defs.d_ai_repair_asset import (
    ingestion_cleanup_a_ai_repair_enqueue_asset,
    ingestion_cleanup_b_ai_repair_enqueue_asset,
)
from etl.defs.f_xml_asset import ingestion_cleanup_a_xml_verify_asset
from etl.defs.f_xml_repair_cycle_asset import (
    ingestion_cleanup_a_post_repair_build_xml_asset,
    ingestion_cleanup_a_post_repair_verify_xml_asset,
)
from etl.defs.h_taxonomy_asset import ingestion_cleanup_a_taxonomy_llm_asset
from etl.defs.k_tax_module_asset import ingestion_cleanup_a_tax_module_asset
from etl.defs.resources import (
    AIRepairAttemptPriority,
    DBResource,
    PipelineConfig,
    QueueRunMode,
    TaxonomyMode,
)
from etl.utils.batch_keys import agreement_batch_key


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

    def first(self) -> dict[str, object] | None:
        return self._rows[0] if self._rows else None

    def scalars(self) -> "_FakeResult._Scalars":
        values = [next(iter(row.values()), None) for row in self._rows]
        return _FakeResult._Scalars(values)


class _FakeConn:
    def execute(self, statement: object, _params: object | None = None) -> _FakeResult:
        sql = str(statement)
        if "AND status = 'verified'" in sql:
            return _FakeResult(rows=[])
        if "latest ai_repair_attempted XML rows" in sql:
            return _FakeResult(rows=[])
        if "latest=1, and ai_repair_attempted=0" in sql:
            return _FakeResult(rows=[])
        if "SELECT DISTINCT s.agreement_uuid" in sql:
            return _FakeResult(rows=[{"agreement_uuid": "agreement-1"}])
        if "FROM pdx.sections s" in sql and "section_uuid" in sql:
            return _FakeResult(
                rows=[
                    {
                        "agreement_uuid": "agreement-1",
                        "section_uuid": "section-1",
                        "article_title": "ARTICLE I",
                        "article_title_normed": "tax matters",
                        "section_title": "Section 1.1 Taxes",
                        "section_title_normed": "taxes",
                        "xml_content": "<section>Tax text</section>",
                        "xml_version": 2,
                        "section_standard_id": None,
                        "section_standard_id_gold_label": None,
                    }
                ]
            )
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


class _FakeBatchFiles:
    def create(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(id="file-1")


class _FakeBatchAPI:
    def create(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            id="batch-1",
            status="in_progress",
            input_file_id="file-1",
            output_file_id=None,
            error_file_id=None,
        )


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


class IngestionCleanupScopeResumeTests(unittest.TestCase):
    def test_cleanup_a_xml_verify_does_not_resume_unrelated_stranded_batch(self) -> None:
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
            patch("etl.defs.f_xml_asset.mark_logical_run_stage_completed", return_value=None),
            patch(
                "etl.defs.f_xml_asset._fetch_unpulled_xml_verify_batch",
                side_effect=AssertionError("cleanup verify should not inspect unrelated stranded batches"),
            ),
            patch("etl.defs.f_xml_asset.run_post_asset_refresh", return_value=None),
        ):
            decorated_fn = getattr(cast(object, ingestion_cleanup_a_xml_verify_asset.op.compute_fn), "decorated_fn")
            result = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                pipeline_config,
                ["agreement-1"],
            )

        self.assertEqual(result, [])

    def test_cleanup_a_ai_repair_enqueue_uses_scope_specific_batch_keys(self) -> None:
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
            patch("etl.defs.d_ai_repair_asset._oai_client", side_effect=AssertionError("resume path should not create an OpenAI client")),
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
                side_effect=AssertionError("cleanup enqueue should not inspect unrelated global batches"),
            ),
            patch(
                "etl.defs.d_ai_repair_asset._fetch_batch_agreement_uuids",
                return_value=["agreement-1"],
            ),
            patch("etl.defs.d_ai_repair_asset._fetch_candidates"),
            patch("etl.defs.d_ai_repair_asset.run_post_asset_refresh"),
        ):
            enqueue_fn = getattr(cast(object, ingestion_cleanup_a_ai_repair_enqueue_asset.op.compute_fn), "decorated_fn")
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

    def test_cleanup_b_ai_repair_enqueue_does_not_resume_unrelated_global_batches(self) -> None:
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
            patch("etl.defs.d_ai_repair_asset._oai_client", side_effect=AssertionError("resume path should not create an OpenAI client")),
            patch("etl.defs.d_ai_repair_asset.assert_tables_exist"),
            patch("etl.defs.d_ai_repair_asset.should_skip_managed_stage", return_value=(False, None)),
            patch(
                "etl.defs.d_ai_repair_asset.start_or_resume_logical_run",
                return_value=SimpleNamespace(
                    logical_run_id="logical-run-1",
                    agreement_uuids=["agreement-1"],
                    resumed_existing=False,
                ),
            ),
            patch("etl.defs.d_ai_repair_asset.mark_logical_run_stage_completed", return_value=None),
            patch(
                "etl.defs.d_ai_repair_asset._fetch_candidates",
                side_effect=[
                    [
                        {
                            "agreement_uuid": "agreement-1",
                            "page_uuid": "page-1",
                            "text": "Body",
                            "ai_repair_attempted": 0,
                            "has_completed_requests": 0,
                            "xml_version": 1,
                        }
                    ],
                    [],
                ],
            ) as fetch_candidates,
            patch(
                "etl.defs.d_ai_repair_asset._fetch_open_ai_repair_batch",
                return_value=None,
            ) as fetch_batch,
            patch(
                "etl.defs.d_ai_repair_asset._fetch_batch_agreement_uuids",
                side_effect=AssertionError("cleanup_b should not resume unrelated global batches"),
            ),
            patch("etl.defs.d_ai_repair_asset.run_post_asset_refresh"),
        ):
            enqueue_fn = getattr(cast(object, ingestion_cleanup_b_ai_repair_enqueue_asset.op.compute_fn), "decorated_fn")
            result = enqueue_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                pipeline_config,
            )

        self.assertEqual(result, [])
        self.assertEqual(fetch_candidates.call_args_list[0].kwargs.get("exclude_in_flight", False), False)
        self.assertIsNone(fetch_candidates.call_args_list[1].kwargs.get("exclude_in_flight"))
        self.assertEqual(fetch_batch.call_count, 2)
        self.assertTrue(all("batch_key" in call.kwargs for call in fetch_batch.call_args_list))

    def test_cleanup_b_ai_repair_enqueue_skips_when_failed_run_already_reached_later_stage(self) -> None:
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
            patch("etl.defs.d_ai_repair_asset._select_ai_repair_scope", return_value=["agreement-1"]),
            patch(
                "etl.defs.d_ai_repair_asset.start_or_resume_logical_run",
                return_value=SimpleNamespace(
                    logical_run_id="logical-run-1",
                    agreement_uuids=["agreement-1"],
                    resumed_existing=True,
                ),
            ),
            patch(
                "etl.defs.d_ai_repair_asset.should_skip_managed_stage",
                return_value=(True, "ingestion_cleanup_b_post_repair_verify_xml"),
            ),
            patch(
                "etl.defs.d_ai_repair_asset._enqueue_ai_repair_for_agreements",
                side_effect=AssertionError("cleanup_b enqueue should skip when the logical run already reached a later stage"),
            ),
        ):
            enqueue_fn = getattr(cast(object, ingestion_cleanup_b_ai_repair_enqueue_asset.op.compute_fn), "decorated_fn")
            result = enqueue_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                pipeline_config,
            )

        self.assertEqual(result, [])

    def test_cleanup_a_post_repair_build_does_not_defer_for_unrelated_verify_batch(self) -> None:
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
                side_effect=AssertionError("cleanup post-repair build should not inspect unrelated verify batches"),
            ),
            patch("etl.defs.f_xml_repair_cycle_asset.run_post_asset_refresh", return_value=None),
        ):
            decorated_fn = getattr(cast(object, ingestion_cleanup_a_post_repair_build_xml_asset.op.compute_fn), "decorated_fn")
            result = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                pipeline_config,
                ["agreement-1"],
            )

        self.assertEqual(result, [])

    def test_cleanup_a_post_repair_verify_does_not_resume_unrelated_stranded_batch(self) -> None:
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
                side_effect=AssertionError("cleanup post-repair verify should not inspect unrelated stranded batches"),
            ),
            patch("etl.defs.f_xml_repair_cycle_asset.run_post_asset_refresh", return_value=None),
        ):
            decorated_fn = getattr(cast(object, ingestion_cleanup_a_post_repair_verify_xml_asset.op.compute_fn), "decorated_fn")
            result = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                pipeline_config,
                ["agreement-1"],
            )

        self.assertEqual(result, [])

    def test_cleanup_a_taxonomy_uses_full_scope_batch_key_for_create_and_resume(self) -> None:
        context = SimpleNamespace(log=_FakeLog())
        db = SimpleNamespace(database="pdx", get_engine=lambda: _FakeEngine())
        taxonomy_model = SimpleNamespace(model=lambda: object())
        pipeline_config = SimpleNamespace(
            taxonomy_agreement_batch_size=10,
            taxonomy_mode=TaxonomyMode.LLM,
            taxonomy_section_title_regex=None,
            taxonomy_llm_model="gpt-5-mini",
            taxonomy_llm_sections_per_request=1,
            queue_run_mode=QueueRunMode.SINGLE_BATCH,
            resume_openai_batches=True,
        )
        expected_scope_key = agreement_batch_key(["agreement-1", "agreement-2"])

        with (
            patch("etl.defs.h_taxonomy_asset.assert_tables_exist", return_value=None),
            patch("etl.defs.h_taxonomy_asset.should_skip_managed_stage", return_value=(False, None)),
            patch(
                "etl.defs.h_taxonomy_asset.load_active_scope_for_job",
                side_effect=_fallback_scope,
            ),
            patch("etl.defs.h_taxonomy_asset.load_active_logical_run", return_value=None),
            patch("etl.defs.h_taxonomy_asset.mark_logical_run_stage_completed", return_value=None),
            patch("etl.defs.h_taxonomy_asset._fetch_unapplied_taxonomy_llm_batch", return_value=None) as fetch_batch,
            patch("etl.defs.h_taxonomy_asset._create_and_apply_taxonomy_llm_batch", return_value=None) as create_batch,
            patch("etl.defs.h_taxonomy_asset._oai_client", return_value=object()),
            patch("etl.defs.h_taxonomy_asset.run_post_asset_refresh", return_value=None),
        ):
            decorated_fn = getattr(cast(object, ingestion_cleanup_a_taxonomy_llm_asset.op.compute_fn), "decorated_fn")
            _ = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                db=cast(object, db),
                taxonomy_model=cast(object, taxonomy_model),
                pipeline_config=cast(object, pipeline_config),
                fresh_section_agreement_uuids=["agreement-1"],
                repair_section_agreement_uuids=["agreement-2"],
            )

        self.assertEqual(fetch_batch.call_args.kwargs["batch_key"], expected_scope_key)
        self.assertEqual(create_batch.call_args.kwargs["batch_key_override"], expected_scope_key)

    def test_cleanup_a_tax_module_uses_scope_batch_key_for_resume_and_create(self) -> None:
        context = SimpleNamespace(log=_FakeLog())
        db = _FakeDB()
        pipeline_config = cast(
            PipelineConfig,
            cast(
                object,
                SimpleNamespace(
                    tax_module_agreement_batch_size=10,
                    tax_module_llm_clauses_per_request=1,
                    tax_module_llm_model="gpt-5-mini",
                    resume_openai_batches=True,
                    queue_run_mode="SINGLE_BATCH",
                ),
            ),
        )
        expected_batch_key = agreement_batch_key(["agreement-1", "agreement-2"])

        with (
            patch("etl.defs.k_tax_module_asset.assert_tables_exist", return_value=None),
            patch("etl.defs.k_tax_module_asset.runs_single_batch", return_value=True),
            patch("etl.defs.k_tax_module_asset.should_skip_managed_stage", return_value=(False, None)),
            patch(
                "etl.defs.k_tax_module_asset.load_active_scope_for_job",
                side_effect=_fallback_scope,
            ),
            patch("etl.defs.k_tax_module_asset.load_active_logical_run", return_value=None),
            patch("etl.defs.k_tax_module_asset.mark_logical_run_stage_completed", return_value=None),
            patch("etl.defs.k_tax_module_asset._fetch_unapplied_tax_module_batch", return_value=None) as fetch_batch,
            patch("etl.defs.k_tax_module_asset._fetch_tax_section_standard_ids", return_value={"tax"}),
            patch(
                "etl.defs.k_tax_module_asset.extract_tax_clauses",
                return_value=[
                    {
                        "clause_uuid": "clause-1",
                        "agreement_uuid": "agreement-1",
                        "section_uuid": "section-1",
                        "module": "tax",
                        "clause_text": "Tax clause",
                        "anchor_label": "tax",
                        "context_type": "body",
                        "xml_version": 2,
                    }
                ],
            ),
            patch("etl.defs.k_tax_module_asset.replace_module_clauses", return_value=None),
            patch("etl.defs.k_tax_module_asset._fetch_taxonomy_json", return_value=[]),
            patch("etl.defs.k_tax_module_asset._create_llm_lines", return_value=[{"custom_id": "line-1"}]) as create_lines,
            patch(
                "etl.defs.k_tax_module_asset._oai_client",
                return_value=SimpleNamespace(files=_FakeBatchFiles(), batches=_FakeBatchAPI()),
            ),
            patch("etl.defs.k_tax_module_asset._resume_and_apply_tax_module_batch", return_value=None),
            patch("etl.defs.k_tax_module_asset.run_post_asset_refresh", return_value=None),
        ):
            decorated_fn = getattr(cast(object, ingestion_cleanup_a_tax_module_asset.op.compute_fn), "decorated_fn")
            _ = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                pipeline_config,
                ["agreement-1", "agreement-2"],
            )

        self.assertEqual(fetch_batch.call_args.kwargs["batch_key"], expected_batch_key)
        self.assertEqual(create_lines.call_args.kwargs["batch_key"], expected_batch_key)


if __name__ == "__main__":
    _ = unittest.main()
