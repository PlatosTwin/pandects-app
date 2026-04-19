"""Scoped regular_ingest tax-module resume guards."""
# pyright: reportAny=false, reportPrivateUsage=false

import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from dagster import AssetExecutionContext

from etl.defs.k_tax_module_asset import _tax_candidate_sql, regular_ingest_tax_module_asset
from etl.defs.resources import DBResource, PipelineConfig
from etl.utils.batch_keys import agreement_batch_key


class _FakeResult:
    def __init__(self, *, rows: list[dict[str, object]] | None = None) -> None:
        self._rows = rows or []

    def mappings(self) -> "_FakeResult":
        return self

    def fetchall(self) -> list[dict[str, object]]:
        return self._rows

    def first(self) -> dict[str, object] | None:
        return self._rows[0] if self._rows else None

    def scalars(self) -> "_FakeScalars":
        return _FakeScalars([next(iter(row.values()), None) for row in self._rows])


class _FakeScalars:
    def __init__(self, values: list[object]) -> None:
        self._values = values

    def all(self) -> list[object]:
        return self._values


class _FakeConn:
    def execute(self, statement: object, _params: object | None = None) -> _FakeResult:
        sql = str(statement)
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


class _FakeLog:
    def info(self, *_args: object, **_kwargs: object) -> None:
        return None

    def warning(self, *_args: object, **_kwargs: object) -> None:
        return None


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


class TaxModuleAssetTests(unittest.TestCase):
    def test_tax_candidate_sql_includes_tax_code_signals(self) -> None:
        candidate_sql, params = _tax_candidate_sql(
            section_ids=set(),
            standard_id_column_sql="COALESCE(s.section_standard_id_gold_label, s.section_standard_id, '')",
        )

        self.assertIn("s.xml_content", candidate_sql)
        self.assertIn("s.section_title", candidate_sql)
        self.assertIn("%338%", params.values())
        self.assertIn("%firpta%", params.values())
        self.assertIn("%purchase price allocation%", params.values())
        self.assertIn("%golden parachute%", params.values())
        self.assertNotIn("%credit%", params.values())
        self.assertNotIn("%ric%", params.values())
        self.assertNotIn("%vat%", params.values())

    def test_regular_ingest_tax_module_uses_scope_batch_key_for_resume_and_create(self) -> None:
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
            decorated_fn = getattr(cast(object, regular_ingest_tax_module_asset.op.compute_fn), "decorated_fn")
            _ = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                pipeline_config,
                ["agreement-1", "agreement-2"],
            )

        self.assertEqual(fetch_batch.call_args.kwargs["batch_key"], expected_batch_key)
        self.assertEqual(create_lines.call_args.kwargs["batch_key"], expected_batch_key)

    def test_regular_ingest_tax_module_noops_for_explicit_empty_scope(self) -> None:
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

        with (
            patch("etl.defs.k_tax_module_asset.should_skip_managed_stage", return_value=(False, None)),
            patch("etl.defs.k_tax_module_asset.load_active_scope_for_job", return_value=[]),
            patch("etl.defs.k_tax_module_asset.load_active_logical_run", return_value=None),
            patch("etl.defs.k_tax_module_asset.mark_logical_run_stage_completed", return_value=None),
            patch(
                "etl.defs.k_tax_module_asset._fetch_unapplied_tax_module_batch",
                side_effect=AssertionError("empty scoped run should not inspect tax-module batches"),
            ),
            patch(
                "etl.defs.k_tax_module_asset._create_llm_lines",
                side_effect=AssertionError("empty scoped run should not create tax-module requests"),
            ),
            patch("etl.defs.k_tax_module_asset.run_post_asset_refresh", return_value=None),
        ):
            decorated_fn = getattr(cast(object, regular_ingest_tax_module_asset.op.compute_fn), "decorated_fn")
            result = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                pipeline_config,
                [],
            )

        self.assertEqual(result, [])


if __name__ == "__main__":
    _ = unittest.main()
