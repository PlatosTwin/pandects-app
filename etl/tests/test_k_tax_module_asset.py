# pyright: reportAny=false
import json
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from dagster import AssetExecutionContext, build_op_context

from etl.defs.k_tax_module_asset import (
    _apply_tax_module_batch_output,
    _build_clause_rows,
    tax_module_from_fresh_xml_asset,
)


class _FakeFileContent:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeFiles:
    def __init__(self, text: str) -> None:
        self._text = text

    def content(self, _file_id: str) -> _FakeFileContent:
        return _FakeFileContent(self._text)


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.files = _FakeFiles(text)


class _FakeBeginContext:
    def __init__(self, conn: object) -> None:
        self._conn = conn

    def __enter__(self) -> object:
        return self._conn

    def __exit__(self, *_exc: object) -> None:
        return None


class _FakeEngine:
    def __init__(self, conn: object) -> None:
        self._conn = conn

    def begin(self) -> _FakeBeginContext:
        return _FakeBeginContext(self._conn)


class _FakeConn:
    pass


class _FakeLog:
    def info(self, *args: object, **kwargs: object) -> None:
        _ = args
        _ = kwargs

    def warning(self, *args: object, **kwargs: object) -> None:
        _ = args
        _ = kwargs


class TaxModuleAssetTests(unittest.TestCase):
    def test_build_clause_rows_filters_to_tax_sections(self) -> None:
        clauses = _build_clause_rows(
            section_rows=[
                {
                    "agreement_uuid": "agreement-1",
                    "section_uuid": "section-tax",
                    "article_title": "ARTICLE VI COVENANTS",
                    "article_title_normed": "covenants",
                    "section_title": "Tax Matters",
                    "section_title_normed": "tax matters",
                    "xml_content": "<text>(a) Parent pays transfer taxes.</text>",
                    "xml_version": 1,
                    "section_standard_id": None,
                    "section_standard_id_gold_label": None,
                },
                {
                    "agreement_uuid": "agreement-1",
                    "section_uuid": "section-non-tax",
                    "article_title": "ARTICLE VI COVENANTS",
                    "article_title_normed": "covenants",
                    "section_title": "Expenses",
                    "section_title_normed": "expenses",
                    "xml_content": "<text>The parties split expenses.</text>",
                    "xml_version": 1,
                    "section_standard_id": None,
                    "section_standard_id_gold_label": None,
                },
            ],
            tax_section_standard_ids=set(),
        )

        self.assertEqual(len(clauses), 1)
        self.assertEqual(clauses[0]["section_uuid"], "section-tax")

    def test_apply_tax_module_batch_output_upserts_assignments(self) -> None:
        response_body = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {
                                    "assignments": [
                                        {
                                            "clause_uuid": "clause-1",
                                            "categories": ["tax_transfer", "tax_transfer"],
                                        }
                                    ]
                                }
                            ),
                        }
                    ],
                }
            ]
        }
        batch_output_text = json.dumps(
            {
                "response": {
                    "body": response_body,
                }
            }
        )
        fake_conn = _FakeConn()
        fake_engine = _FakeEngine(fake_conn)
        db = SimpleNamespace(database="pdx")
        context = SimpleNamespace(log=_FakeLog())

        with patch(
            "etl.defs.k_tax_module_asset.upsert_tax_clause_assignments"
        ) as upsert_assignments:
            applied_count, parse_errors = _apply_tax_module_batch_output(
                cast(AssetExecutionContext, cast(object, context)),
                engine=fake_engine,
                client=_FakeClient(batch_output_text),
                db=cast(object, db),
                batch=SimpleNamespace(output_file_id="file-1"),
                model_name="gpt-5-mini",
            )

        self.assertEqual(applied_count, 1)
        self.assertEqual(parse_errors, 0)
        upsert_assignments.assert_called_once()
        assignments = upsert_assignments.call_args.args[0]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0]["standard_id"], "tax_transfer")

    def test_tax_module_from_fresh_xml_asset_preserves_scope(self) -> None:
        context = build_op_context()
        db = SimpleNamespace()
        pipeline_config = SimpleNamespace()

        with patch(
            "etl.defs.k_tax_module_asset._run_tax_module_for_agreements", return_value=[]
        ) as runner:
            tax_module_from_fresh_xml_asset(
                context=cast(AssetExecutionContext, cast(object, context)),
                db=cast(object, db),
                pipeline_config=cast(object, pipeline_config),
                section_agreement_uuids=["agreement-1"],
            )

        runner.assert_called_once_with(
            cast(AssetExecutionContext, cast(object, context)),
            cast(object, db),
            cast(object, pipeline_config),
            target_agreement_uuids=["agreement-1"],
            log_prefix="tax_module_from_fresh_xml_asset",
        )


if __name__ == "__main__":
    unittest.main()
