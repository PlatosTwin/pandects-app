# pyright: reportAny=false, reportPrivateUsage=false
import json
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from dagster import AssetExecutionContext

import etl.defs.h_taxonomy_asset as taxonomy_asset_module
from etl.defs.h_taxonomy_asset import regular_ingest_taxonomy_llm_asset, taxonomy_asset
from etl.defs.resources import QueueRunMode, TaxonomyMode
from etl.utils.batch_keys import agreement_batch_key


class _FakeResult:
    def __init__(self, *, rowcount: int = 0, rows: list[dict[str, object]] | None = None) -> None:
        self.rowcount = rowcount
        self._rows = rows or []

    def mappings(self) -> "_FakeResult":
        return self

    def fetchall(self) -> list[dict[str, object]]:
        return self._rows

    def first(self) -> dict[str, object] | None:
        return self._rows[0] if self._rows else None


class _FakeBeginContext:
    def __init__(self, conn: object) -> None:
        self._conn = conn

    def __enter__(self) -> object:
        return self._conn

    def __exit__(self, *_exc: object) -> None:
        return None


class _FakeConn:
    def __init__(self, sec_rows: list[dict[str, object]]) -> None:
        self._sec_rows = sec_rows
        self.executed_sql: list[str] = []
        self.update_payloads: list[dict[str, object]] = []
        self.gold_update_payloads: list[dict[str, object]] = []

    def execute(self, statement: object, params: object | None = None) -> _FakeResult:
        sql = str(statement)
        self.executed_sql.append(sql)

        if "SELECT DISTINCT s.agreement_uuid" in sql:
            return _FakeResult(rows=[{"agreement_uuid": "agreement-1"}])

        if "SELECT\n            s.section_uuid," in sql or "SELECT\n                s.section_uuid," in sql:
            return _FakeResult(rows=self._sec_rows)

        if "SELECT section_uuid, agreement_uuid" in sql and "WHERE section_uuid IN" in sql:
            return _FakeResult(
                rows=[
                    {
                        "section_uuid": row["section_uuid"],
                        "agreement_uuid": row["agreement_uuid"],
                    }
                    for row in self._sec_rows
                ]
            )

        if "UPDATE pdx.sections" in sql and "section_standard_id_gold_label" in sql:
            if isinstance(params, list):
                for payload in cast(list[object], params):
                    if isinstance(payload, dict):
                        self.gold_update_payloads.append(cast(dict[str, object], payload))
            return _FakeResult(rowcount=1)

        if "UPDATE pdx.sections" in sql and "section_standard_id =" in sql:
            if isinstance(params, list):
                for payload in cast(list[object], params):
                    if isinstance(payload, dict):
                        self.update_payloads.append(cast(dict[str, object], payload))
            return _FakeResult(rowcount=1)

        if "SELECT m.agreement_uuid, m.xml, m.version" in sql:
            return _FakeResult(
                rows=[
                    {
                        "agreement_uuid": "agreement-1",
                        "xml": "<document />",
                        "version": 2,
                    }
                ]
            )

        if "INSERT INTO pdx.taxonomy_llm_batches" in sql:
            return _FakeResult(rowcount=1)

        if "UPDATE pdx.taxonomy_llm_batches" in sql:
            return _FakeResult(rowcount=1)

        raise AssertionError(f"Unexpected SQL in test: {sql}")


class _FakeEngine:
    def __init__(self, conn: object) -> None:
        self._conn = conn

    def begin(self) -> _FakeBeginContext:
        return _FakeBeginContext(self._conn)


class _FakeLog:
    def info(self, *args: object, **kwargs: object) -> None:
        _ = args
        _ = kwargs

    def warning(self, *args: object, **kwargs: object) -> None:
        _ = args
        _ = kwargs


class _FakeBatchFiles:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text

    def create(self, *, purpose: str, file: object) -> SimpleNamespace:
        _ = purpose
        _ = file
        return SimpleNamespace(id="file-1")

    def content(self, _file_id: str) -> SimpleNamespace:
        return SimpleNamespace(text=self.output_text)


class _FakeBatchClient:
    def __init__(self, output_text: str) -> None:
        def _create_batch(**_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                id="batch-1",
                status="in_progress",
                input_file_id="file-1",
                output_file_id="file-out-1",
                error_file_id=None,
            )

        self.files = _FakeBatchFiles(output_text)
        self.batches = SimpleNamespace(create=_create_batch)


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


class TaxonomyAssetTests(unittest.TestCase):
    def test_ml_mode_serializes_label_array_and_applies_regex_filter(self) -> None:
        conn = _FakeConn(
            sec_rows=[
                {
                    "section_uuid": "section-1",
                    "agreement_uuid": "agreement-1",
                    "article_title": "ARTICLE I",
                    "section_title": "Section 1",
                    "article_title_normed": "article i",
                    "section_title_normed": "governing law",
                    "xml_content": "<section>Body</section>",
                    "section_standard_id_gold_label": None,
                }
            ]
        )
        context = SimpleNamespace(log=_FakeLog())
        db = SimpleNamespace(database="pdx", get_engine=lambda: _FakeEngine(conn))
        taxonomy_model = SimpleNamespace(model=lambda: object())
        pipeline_config = SimpleNamespace(
            taxonomy_agreement_batch_size=10,
            taxonomy_mode=TaxonomyMode.ML,
            taxonomy_section_title_regex="governing",
            taxonomy_llm_model="gpt-5.4-mini",
            taxonomy_llm_sections_per_request=5,
            queue_run_mode=QueueRunMode.SINGLE_BATCH,
            resume_openai_batches=True,
        )

        with (
            patch(
                "etl.defs.h_taxonomy_asset.predict_taxonomy",
                return_value=(
                    [{"section_uuid": "section-1", "agreement_uuid": "agreement-1"}],
                    [{"label": "governing_law", "alt_probs": [0.9, 0.05, 0.05]}],
                ),
            ),
            patch(
                "etl.defs.h_taxonomy_asset.apply_standard_ids_to_xml",
                return_value="<document standardId='[\"governing_law\"]' />",
            ) as apply_standard_ids_to_xml,
            patch("etl.defs.h_taxonomy_asset.upsert_xml"),
            patch("etl.defs.h_taxonomy_asset.refresh_latest_sections_search"),
            patch("etl.defs.h_taxonomy_asset.run_post_asset_refresh", return_value=None),
        ):
            decorated_fn = getattr(cast(object, taxonomy_asset.op.compute_fn), "decorated_fn")
            _ = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                db=cast(object, db),
                taxonomy_model=cast(object, taxonomy_model),
                pipeline_config=cast(object, pipeline_config),
            )

        self.assertEqual(conn.update_payloads[0]["label"], '["governing_law"]')
        apply_standard_ids_to_xml.assert_called_once_with(
            "<document />",
            {"section-1": '["governing_law"]'},
        )
        self.assertTrue(any("REGEXP :section_title_regex" in sql for sql in conn.executed_sql))

    def test_gold_backfill_ignores_regex_and_does_not_update_sections(self) -> None:
        conn = _FakeConn(
            sec_rows=[
                {
                    "section_uuid": "section-1",
                    "agreement_uuid": "agreement-1",
                    "section_standard_id_gold_label": '["gold_governing_law"]',
                }
            ]
        )
        context = SimpleNamespace(log=_FakeLog())
        db = SimpleNamespace(database="pdx", get_engine=lambda: _FakeEngine(conn))
        taxonomy_model = SimpleNamespace(
            model=lambda: (_ for _ in ()).throw(
                AssertionError("taxonomy model should not be loaded in gold_backfill mode")
            )
        )
        pipeline_config = SimpleNamespace(
            taxonomy_agreement_batch_size=10,
            taxonomy_mode=TaxonomyMode.GOLD_BACKFILL,
            taxonomy_section_title_regex="governing",
            taxonomy_llm_model="gpt-5.4-mini",
            taxonomy_llm_sections_per_request=5,
            queue_run_mode=QueueRunMode.SINGLE_BATCH,
            resume_openai_batches=True,
        )

        with (
            patch(
                "etl.defs.h_taxonomy_asset.apply_standard_ids_to_xml",
                return_value="<document standardId='[\"gold_governing_law\"]' />",
            ) as apply_standard_ids_to_xml,
            patch("etl.defs.h_taxonomy_asset.upsert_xml"),
            patch("etl.defs.h_taxonomy_asset.refresh_latest_sections_search"),
            patch("etl.defs.h_taxonomy_asset.run_post_asset_refresh", return_value=None),
        ):
            decorated_fn = getattr(cast(object, taxonomy_asset.op.compute_fn), "decorated_fn")
            _ = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                db=cast(object, db),
                taxonomy_model=cast(object, taxonomy_model),
                pipeline_config=cast(object, pipeline_config),
            )

        apply_standard_ids_to_xml.assert_called_once_with(
            "<document />",
            {"section-1": '["gold_governing_law"]'},
        )
        self.assertFalse(conn.update_payloads)
        self.assertFalse(any("REGEXP :section_title_regex" in sql for sql in conn.executed_sql))

    def test_llm_mode_creates_batch_and_applies_json_array_gold_labels(self) -> None:
        conn = _FakeConn(
            sec_rows=[
                {
                    "section_uuid": "section-1",
                    "agreement_uuid": "agreement-1",
                    "article_title": "ARTICLE I",
                    "section_title": "Section 1",
                    "article_title_normed": "general provisions",
                    "section_title_normed": "governing law",
                    "article_order": 1,
                    "section_order": 1,
                    "xml_content": "<section>Body</section>",
                    "section_standard_id_gold_label": None,
                }
            ]
        )
        context = SimpleNamespace(log=_FakeLog())
        db = SimpleNamespace(database="pdx", get_engine=lambda: _FakeEngine(conn))
        taxonomy_model = SimpleNamespace(model=lambda: object())
        pipeline_config = SimpleNamespace(
            taxonomy_agreement_batch_size=10,
            taxonomy_mode=TaxonomyMode.LLM,
            taxonomy_section_title_regex=None,
            taxonomy_llm_model="gpt-5.4-mini",
            taxonomy_llm_sections_per_request=1,
            queue_run_mode=QueueRunMode.SINGLE_BATCH,
            resume_openai_batches=False,
        )
        output_text = json.dumps(
            {
                "custom_id": "ignored",
                "response": {
                    "status_code": 200,
                    "body": {
                        "output": [
                            {
                                "type": "message",
                                "content": [
                                    {
                                        "text": json.dumps(
                                            {
                                                "assignments": [
                                                    {
                                                        "section_uuid": "section-1",
                                                        "categories": ["law", "venue"],
                                                    }
                                                ]
                                            }
                                        )
                                    }
                                ],
                            }
                        ]
                    },
                },
            }
        )

        with (
            patch("etl.defs.h_taxonomy_asset.assert_tables_exist", return_value=None),
            patch(
                "etl.defs.h_taxonomy_asset._fetch_taxonomy_json",
                return_value=[
                    {
                        "l1_standard_id": "l1",
                        "l1_label": "Forum",
                        "l2_standard_id": "l2",
                        "l2_label": "Law",
                        "l3_standard_id": "l3",
                        "l3_label": "Governing Law",
                    }
                ],
            ),
            patch(
                "etl.defs.h_taxonomy_asset._oai_client",
                return_value=_FakeBatchClient(output_text),
            ),
            patch(
                "etl.defs.h_taxonomy_asset._create_taxonomy_llm_lines",
                wraps=taxonomy_asset_module._create_taxonomy_llm_lines,
            ) as create_lines,
            patch(
                "etl.defs.h_taxonomy_asset.poll_batch_until_terminal",
                return_value=SimpleNamespace(
                    id="batch-1",
                    status="completed",
                    input_file_id="file-1",
                    output_file_id="file-out-1",
                    error_file_id=None,
                ),
            ),
            patch(
                "etl.defs.h_taxonomy_asset.apply_standard_ids_to_xml",
                return_value="<document standardId='[\"law\", \"venue\"]' />",
            ) as apply_standard_ids_to_xml,
            patch("etl.defs.h_taxonomy_asset.upsert_xml"),
            patch("etl.defs.h_taxonomy_asset.refresh_latest_sections_search"),
            patch("etl.defs.h_taxonomy_asset.run_post_asset_refresh", return_value=None),
        ):
            decorated_fn = getattr(cast(object, taxonomy_asset.op.compute_fn), "decorated_fn")
            _ = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                db=cast(object, db),
                taxonomy_model=cast(object, taxonomy_model),
                pipeline_config=cast(object, pipeline_config),
            )

        self.assertEqual(create_lines.call_args.kwargs["sections_per_request"], 1)
        self.assertEqual(conn.gold_update_payloads[0]["gold_label_payload"], '["law", "venue"]')
        self.assertEqual(conn.gold_update_payloads[0]["model_name"], "gpt-5.4-mini")
        apply_standard_ids_to_xml.assert_called_once_with(
            "<document />",
            {"section-1": '["law", "venue"]'},
        )
        self.assertTrue(any("INSERT INTO pdx.taxonomy_llm_batches" in sql for sql in conn.executed_sql))
        self.assertGreaterEqual(
            sum(
                "CHAR_LENGTH(TRIM(COALESCE(s.article_title, ''))) >= 3" in sql
                and "CHAR_LENGTH(TRIM(COALESCE(s.section_title, ''))) >= 3" in sql
                and "NOT LIKE '%[reserved]%'" in sql
                and "NOT LIKE '%[omitted]%'" in sql
                and "NOT LIKE '%[intentionally deleted]%'" in sql
                and "NOT LIKE '%[deleted]%'" in sql
                for sql in conn.executed_sql
            ),
            2,
        )

    def test_llm_prompt_payload_uses_raw_titles(self) -> None:
        lines = taxonomy_asset_module._create_taxonomy_llm_lines(
            prediction_rows=[
                {
                    "section_uuid": "section-1",
                    "agreement_uuid": "agreement-1",
                    "article_title": "ARTICLE I Conditions",
                    "section_title": "Section 12.1 Section 14",
                    "article_title_normed": "general provisions",
                    "section_title_normed": "governing law",
                }
            ],
            taxonomy_json=[],
            model_name="gpt-5.4-mini",
            batch_key="batch-key",
            sections_per_request=1,
        )

        prompt_text = cast(str, lines[0]["body"]["input"][0]["content"])
        self.assertIn("Article title: I Conditions.", prompt_text)
        self.assertIn("Section title: 12.1 Section 14.", prompt_text)
        self.assertNotIn("general provisions", prompt_text)

    def test_llm_mode_resumes_existing_batch_before_selecting_new_work(self) -> None:
        conn = _FakeConn(sec_rows=[])
        context = SimpleNamespace(log=_FakeLog())
        db = SimpleNamespace(database="pdx", get_engine=lambda: _FakeEngine(conn))
        taxonomy_model = SimpleNamespace(model=lambda: object())
        pipeline_config = SimpleNamespace(
            taxonomy_agreement_batch_size=10,
            taxonomy_mode=TaxonomyMode.LLM,
            taxonomy_section_title_regex=None,
            taxonomy_llm_model="gpt-5.4-mini",
            taxonomy_llm_sections_per_request=5,
            queue_run_mode=QueueRunMode.SINGLE_BATCH,
            resume_openai_batches=True,
        )

        with (
            patch("etl.defs.h_taxonomy_asset.assert_tables_exist", return_value=None),
            patch(
                "etl.defs.h_taxonomy_asset._fetch_unapplied_taxonomy_llm_batch",
                return_value={
                    "batch_id": "batch-1",
                    "completion_window": "24h",
                    "request_total": 1,
                    "model_name": "gpt-5.4-mini",
                    "batch_key": "key-1",
                },
            ),
            patch("etl.defs.h_taxonomy_asset._resume_and_apply_taxonomy_llm_batch") as resume_batch,
            patch(
                "etl.defs.h_taxonomy_asset._select_agreement_batch",
                side_effect=AssertionError("should not select new work before resuming existing batch"),
            ),
            patch("etl.defs.h_taxonomy_asset._oai_client", return_value=object()),
            patch("etl.defs.h_taxonomy_asset.run_post_asset_refresh", return_value=None),
        ):
            decorated_fn = getattr(cast(object, taxonomy_asset.op.compute_fn), "decorated_fn")
            _ = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                db=cast(object, db),
                taxonomy_model=cast(object, taxonomy_model),
                pipeline_config=cast(object, pipeline_config),
            )

        resume_batch.assert_called_once()

    def test_regular_ingest_llm_mode_uses_full_scope_batch_key_for_create_and_resume(self) -> None:
        conn = _FakeConn(
            sec_rows=[
                {
                    "section_uuid": "section-1",
                    "agreement_uuid": "agreement-1",
                    "article_title": "ARTICLE I",
                    "section_title": "Section 1",
                    "article_title_normed": "general provisions",
                    "section_title_normed": "governing law",
                    "article_order": 1,
                    "section_order": 1,
                    "xml_content": "<section>Body</section>",
                    "section_standard_id_gold_label": None,
                }
            ]
        )
        context = SimpleNamespace(log=_FakeLog())
        db = SimpleNamespace(database="pdx", get_engine=lambda: _FakeEngine(conn))
        taxonomy_model = SimpleNamespace(model=lambda: object())
        pipeline_config = SimpleNamespace(
            taxonomy_agreement_batch_size=10,
            taxonomy_mode=TaxonomyMode.LLM,
            taxonomy_section_title_regex=None,
            taxonomy_llm_model="gpt-5.4-mini",
            taxonomy_llm_sections_per_request=1,
            enable_section_taxonomy=True,
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
            patch(
                "etl.defs.h_taxonomy_asset._fetch_unapplied_taxonomy_llm_batch",
                return_value=None,
            ) as fetch_batch,
            patch(
                "etl.defs.h_taxonomy_asset._create_and_apply_taxonomy_llm_batch",
                return_value=None,
            ) as create_batch,
            patch("etl.defs.h_taxonomy_asset._oai_client", return_value=object()),
            patch("etl.defs.h_taxonomy_asset.run_post_asset_refresh", return_value=None),
        ):
            decorated_fn = getattr(cast(object, regular_ingest_taxonomy_llm_asset.op.compute_fn), "decorated_fn")
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

    def test_regular_ingest_llm_mode_noops_for_explicit_empty_scope(self) -> None:
        conn = _FakeConn(sec_rows=[])
        context = SimpleNamespace(log=_FakeLog())
        db = SimpleNamespace(database="pdx", get_engine=lambda: _FakeEngine(conn))
        taxonomy_model = SimpleNamespace(model=lambda: object())
        pipeline_config = SimpleNamespace(
            taxonomy_agreement_batch_size=10,
            taxonomy_mode=TaxonomyMode.LLM,
            taxonomy_section_title_regex=None,
            taxonomy_llm_model="gpt-5.4-mini",
            taxonomy_llm_sections_per_request=1,
            queue_run_mode=QueueRunMode.SINGLE_BATCH,
            resume_openai_batches=True,
        )

        with (
            patch("etl.defs.h_taxonomy_asset.should_skip_managed_stage", return_value=(False, None)),
            patch("etl.defs.h_taxonomy_asset.load_active_scope_for_job", return_value=[]),
            patch("etl.defs.h_taxonomy_asset.load_active_logical_run", return_value=None),
            patch("etl.defs.h_taxonomy_asset.mark_logical_run_stage_completed", return_value=None),
            patch(
                "etl.defs.h_taxonomy_asset._fetch_unapplied_taxonomy_llm_batch",
                side_effect=AssertionError("empty scoped run should not inspect taxonomy batches"),
            ),
            patch(
                "etl.defs.h_taxonomy_asset._create_and_apply_taxonomy_llm_batch",
                side_effect=AssertionError("empty scoped run should not create taxonomy batches"),
            ),
            patch("etl.defs.h_taxonomy_asset.run_post_asset_refresh", return_value=None),
        ):
            decorated_fn = getattr(cast(object, regular_ingest_taxonomy_llm_asset.op.compute_fn), "decorated_fn")
            result = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                db=cast(object, db),
                taxonomy_model=cast(object, taxonomy_model),
                pipeline_config=cast(object, pipeline_config),
                fresh_section_agreement_uuids=[],
                repair_section_agreement_uuids=[],
            )

        self.assertEqual(result, [])

    def test_regular_ingest_llm_mode_skips_when_section_taxonomy_disabled(self) -> None:
        context = SimpleNamespace(log=_FakeLog())
        db = SimpleNamespace(database="pdx", get_engine=lambda: _FakeEngine(_FakeConn(sec_rows=[])))
        taxonomy_model = SimpleNamespace(model=lambda: object())
        pipeline_config = SimpleNamespace(
            taxonomy_agreement_batch_size=10,
            taxonomy_mode=TaxonomyMode.LLM,
            taxonomy_section_title_regex=None,
            taxonomy_llm_model="gpt-5.4-mini",
            taxonomy_llm_sections_per_request=1,
            enable_section_taxonomy=False,
            queue_run_mode=QueueRunMode.SINGLE_BATCH,
            resume_openai_batches=True,
        )

        with (
            patch("etl.defs.h_taxonomy_asset.should_skip_managed_stage", return_value=(False, None)),
            patch("etl.defs.h_taxonomy_asset.load_active_scope_for_job", return_value=["agreement-1"]),
            patch("etl.defs.h_taxonomy_asset.mark_logical_run_stage_completed", return_value=None) as mark_completed,
            patch(
                "etl.defs.h_taxonomy_asset._fetch_unapplied_taxonomy_llm_batch",
                side_effect=AssertionError("disabled taxonomy should not inspect taxonomy batches"),
            ),
            patch(
                "etl.defs.h_taxonomy_asset._create_and_apply_taxonomy_llm_batch",
                side_effect=AssertionError("disabled taxonomy should not create taxonomy batches"),
            ),
        ):
            decorated_fn = getattr(cast(object, regular_ingest_taxonomy_llm_asset.op.compute_fn), "decorated_fn")
            result = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                db=cast(object, db),
                taxonomy_model=cast(object, taxonomy_model),
                pipeline_config=cast(object, pipeline_config),
                fresh_section_agreement_uuids=["agreement-1"],
                repair_section_agreement_uuids=[],
            )

        self.assertEqual(result, ["agreement-1"])
        mark_completed.assert_called_once_with(
            db=db,
            job_name="regular_ingest",
            stage_name="regular_ingest_taxonomy_llm",
        )


if __name__ == "__main__":
    _ = unittest.main()
