# pyright: reportAny=false, reportPrivateUsage=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportUnknownMemberType=false
import json
import unittest
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from dagster import AssetExecutionContext
from openai import OpenAI
from etl.defs.resources import DBResource, PipelineConfig
from etl.defs.f_xml_asset import (
    XML_REASON_BODY_STARTS_NON_ARTICLE,
    XML_REASON_SECTION_NON_SEQUENTIAL,
    XML_REASON_SECTION_TITLE_INVALID_NUMBERING,
    XML_REASON_LLM_INVALID,
    XML_REASON_TOO_FEW_ARTICLES,
    _reason_rows_changed,
    _apply_xml_verify_batch_output,
    find_hard_rule_violations,
    xml_verify_asset,
)


class XMLVerifyAssetTests(unittest.TestCase):
    def test_find_hard_rule_violations_uses_section_page_uuid_attribute(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="ARTICLE III">
                  <section title="AND" pageUUID="page-22">
                    <text>Body text</text>
                    <pageUUID>page-99</pageUUID>
                  </section>
                </article>
              </body>
            </document>
            """
        )
        violations = find_hard_rule_violations(root)
        target = next(
            v for v in violations if v.reason_code == XML_REASON_SECTION_TITLE_INVALID_NUMBERING
        )
        self.assertEqual(target.page_uuids, ("page-22",))

    def test_find_hard_rule_violations_rejects_fewer_than_five_articles(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="ARTICLE I"><section title="Section 1.1 First" pageUUID="page-1" /></article>
                <article title="ARTICLE II"><section title="Section 2.1 Second" pageUUID="page-2" /></article>
                <article title="ARTICLE III"><section title="Section 3.1 Third" pageUUID="page-3" /></article>
                <article title="ARTICLE IV"><section title="Section 4.1 Fourth" pageUUID="page-4" /></article>
              </body>
            </document>
            """
        )

        violations = find_hard_rule_violations(root)

        target = next(v for v in violations if v.reason_code == XML_REASON_TOO_FEW_ARTICLES)
        self.assertEqual(
            target.reason_detail,
            "Too few articles: found 4, minimum required is 5.",
        )

    def test_find_hard_rule_violations_allows_five_articles(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="ARTICLE I"><section title="Section 1.1 First" pageUUID="page-1" /></article>
                <article title="ARTICLE II"><section title="Section 2.1 Second" pageUUID="page-2" /></article>
                <article title="ARTICLE III"><section title="Section 3.1 Third" pageUUID="page-3" /></article>
                <article title="ARTICLE IV"><section title="Section 4.1 Fourth" pageUUID="page-4" /></article>
                <article title="ARTICLE V"><section title="Section 5.1 Fifth" pageUUID="page-5" /></article>
              </body>
            </document>
            """
        )

        violations = find_hard_rule_violations(root)

        self.assertFalse(any(v.reason_code == XML_REASON_TOO_FEW_ARTICLES for v in violations))

    def test_find_hard_rule_violations_targets_previous_section_on_forward_gap(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="ARTICLE I">
                  <section title="Section 1.1 First" pageUUID="page-1" />
                  <section title="Section 1.2 Second" pageUUID="page-2" />
                  <section title="Section 1.4 Intentionally Deleted. Section 1.5 Closing." pageUUID="page-3">
                    <pageUUID>page-4</pageUUID>
                  </section>
                  <section title="Section 1.6 Transfer Taxes" pageUUID="page-4" />
                </article>
                <article title="ARTICLE II"><section title="Section 2.1 Second" pageUUID="page-5" /></article>
                <article title="ARTICLE III"><section title="Section 3.1 Third" pageUUID="page-6" /></article>
                <article title="ARTICLE IV"><section title="Section 4.1 Fourth" pageUUID="page-7" /></article>
                <article title="ARTICLE V"><section title="Section 5.1 Fifth" pageUUID="page-8" /></article>
              </body>
            </document>
            """
        )

        violations = find_hard_rule_violations(root)

        target = next(v for v in violations if v.reason_code == XML_REASON_SECTION_NON_SEQUENTIAL)
        self.assertEqual(target.page_uuids, ("page-3",))

    def test_find_hard_rule_violations_targets_previous_section_when_it_mentions_missing_heading(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="ARTICLE I">
                  <section title="Section 1.1 First" pageUUID="page-1" />
                  <section title="Section 1.2 Second" pageUUID="page-2">
                    <text>Section 1.3 Interim Covenants.</text>
                  </section>
                  <section title="Section 1.4 Closing" pageUUID="page-3" />
                </article>
                <article title="ARTICLE II"><section title="Section 2.1 Second" pageUUID="page-5" /></article>
                <article title="ARTICLE III"><section title="Section 3.1 Third" pageUUID="page-6" /></article>
                <article title="ARTICLE IV"><section title="Section 4.1 Fourth" pageUUID="page-7" /></article>
                <article title="ARTICLE V"><section title="Section 5.1 Fifth" pageUUID="page-8" /></article>
              </body>
            </document>
            """
        )

        violations = find_hard_rule_violations(root)

        target = next(v for v in violations if v.reason_code == XML_REASON_SECTION_NON_SEQUENTIAL)
        self.assertEqual(target.page_uuids, ("page-2",))

    def test_find_hard_rule_violations_targets_both_adjacent_pages_when_forward_gap_is_ambiguous(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="ARTICLE I">
                  <section title="Section 1.1 First" pageUUID="page-1" />
                  <section title="Section 1.2 Second" pageUUID="page-2" />
                  <section title="Section 1.4 Closing" pageUUID="page-3" />
                </article>
                <article title="ARTICLE II"><section title="Section 2.1 Second" pageUUID="page-5" /></article>
                <article title="ARTICLE III"><section title="Section 3.1 Third" pageUUID="page-6" /></article>
                <article title="ARTICLE IV"><section title="Section 4.1 Fourth" pageUUID="page-7" /></article>
                <article title="ARTICLE V"><section title="Section 5.1 Fifth" pageUUID="page-8" /></article>
              </body>
            </document>
            """
        )

        violations = find_hard_rule_violations(root)

        target = next(v for v in violations if v.reason_code == XML_REASON_SECTION_NON_SEQUENTIAL)
        self.assertEqual(target.page_uuids, ("page-2", "page-3"))

    def test_reason_rows_changed_ignores_order(self) -> None:
        existing = [
            {"reason_code": "section_non_sequential", "reason_detail": "gap", "page_uuid": "page-2"},
            {"reason_code": "section_article_mismatch", "reason_detail": "mismatch", "page_uuid": "page-3"},
        ]
        new = [
            {"reason_code": "section_article_mismatch", "reason_detail": "mismatch", "page_uuid": "page-3"},
            {"reason_code": "section_non_sequential", "reason_detail": "gap", "page_uuid": "page-2"},
        ]

        self.assertFalse(_reason_rows_changed(existing, new))

    def test_reason_rows_changed_detects_page_target_change(self) -> None:
        existing = [
            {"reason_code": "section_non_sequential", "reason_detail": "gap", "page_uuid": "page-2"},
        ]
        new = [
            {"reason_code": "section_non_sequential", "reason_detail": "gap", "page_uuid": "page-3"},
        ]

        self.assertTrue(_reason_rows_changed(existing, new))

    def test_apply_xml_verify_batch_output_sets_status_source_to_asset(self) -> None:
        class _FakeContent:
            def __init__(self, text_value: str) -> None:
                self._text_value = text_value

            def text(self) -> str:
                return self._text_value

        class _FakeFiles:
            def __init__(self, text_value: str) -> None:
                self._text_value = text_value

            def content(self, _file_id: str) -> _FakeContent:
                return _FakeContent(self._text_value)

        class _FakeClient:
            def __init__(self, text_value: str) -> None:
                self.files = _FakeFiles(text_value)

        class _FakeResult:
            def __init__(
                self,
                rowcount: int,
                rows: list[dict[str, object]] | None = None,
            ) -> None:
                self.rowcount = rowcount
                self._rows = rows or []

            class _Mappings:
                def __init__(self, rows: list[dict[str, object]]) -> None:
                    self._rows = rows

                def fetchall(self) -> list[dict[str, object]]:
                    return self._rows

            def mappings(self):
                return _FakeResult._Mappings(self._rows)

        class _FakeConn:
            def __init__(self) -> None:
                self.executed: list[tuple[str, dict[str, object]]] = []

            def execute(self, statement: object, params: dict[str, object]) -> _FakeResult:
                sql = str(statement)
                self.executed.append((sql, params))
                if "SELECT reason_code, reason_detail, page_uuid" in sql:
                    return _FakeResult(0, rows=[])
                return _FakeResult(1)

        class _BeginContext:
            def __init__(self, conn: _FakeConn) -> None:
                self._conn = conn

            def __enter__(self) -> _FakeConn:
                return self._conn

            def __exit__(self, *_exc: object) -> None:
                return None

        class _FakeEngine:
            def __init__(self, conn: _FakeConn) -> None:
                self._conn = conn

            def begin(self) -> _BeginContext:
                return _BeginContext(self._conn)

        response_payload = {
            "custom_id": "agreement-1|3",
            "response": {
                "status_code": 200,
                "body": {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"text": '{"status":"verified"}'}],
                        }
                    ]
                },
            },
        }
        out_text = json.dumps(response_payload)
        conn = _FakeConn()
        engine = _FakeEngine(conn)
        client = _FakeClient(out_text)

        class _FakeLog:
            def info(self, *_args: object, **_kwargs: object) -> None:
                return None

            def warning(self, *_args: object, **_kwargs: object) -> None:
                return None

        context = SimpleNamespace(
            log=_FakeLog()
        )
        batch = SimpleNamespace(output_file_id="file-1")

        updated, parse_errors = _apply_xml_verify_batch_output(
            context=cast(AssetExecutionContext, cast(object, context)),
            engine=engine,
            client=cast(OpenAI, cast(object, client)),
            xml_table="pdx.xml",
            xml_status_reasons_table="pdx.xml_status_reasons",
            batch=batch,
        )

        self.assertEqual(updated, 1)
        self.assertEqual(parse_errors, 0)
        self.assertEqual(len(conn.executed), 3)
        select_sql, _ = conn.executed[0]
        self.assertIn("SELECT reason_code, reason_detail, page_uuid", select_sql)
        executed_sql, params = conn.executed[1]
        self.assertIn("status_source = 'asset'", executed_sql)
        self.assertIn("status_reason_code = :reason_code", executed_sql)
        self.assertIn("status_reason_detail = :reason_detail", executed_sql)
        self.assertEqual(params["agreement_uuid"], "agreement-1")
        self.assertEqual(params["version"], 3)
        self.assertEqual(params["status"], "verified")
        self.assertIsNone(params["reason_code"])
        self.assertIsNone(params["reason_detail"])
        delete_sql, _ = conn.executed[2]
        self.assertIn("DELETE FROM pdx.xml_status_reasons", delete_sql)

    def test_apply_xml_verify_batch_output_sets_llm_invalid_reason_code(self) -> None:
        class _FakeContent:
            def __init__(self, text_value: str) -> None:
                self._text_value = text_value

            def text(self) -> str:
                return self._text_value

        class _FakeFiles:
            def __init__(self, text_value: str) -> None:
                self._text_value = text_value

            def content(self, _file_id: str) -> _FakeContent:
                return _FakeContent(self._text_value)

        class _FakeClient:
            def __init__(self, text_value: str) -> None:
                self.files = _FakeFiles(text_value)

        class _FakeResult:
            def __init__(
                self,
                rowcount: int,
                rows: list[dict[str, object]] | None = None,
            ) -> None:
                self.rowcount = rowcount
                self._rows = rows or []

            class _Mappings:
                def __init__(self, rows: list[dict[str, object]]) -> None:
                    self._rows = rows

                def fetchall(self) -> list[dict[str, object]]:
                    return self._rows

            def mappings(self):
                return _FakeResult._Mappings(self._rows)

        class _FakeConn:
            def __init__(self) -> None:
                self.executed: list[tuple[str, dict[str, object]]] = []

            def execute(self, statement: object, params: dict[str, object]) -> _FakeResult:
                sql = str(statement)
                self.executed.append((sql, params))
                if "SELECT reason_code, reason_detail, page_uuid" in sql:
                    return _FakeResult(0, rows=[])
                return _FakeResult(1)

        class _BeginContext:
            def __init__(self, conn: _FakeConn) -> None:
                self._conn = conn

            def __enter__(self) -> _FakeConn:
                return self._conn

            def __exit__(self, *_exc: object) -> None:
                return None

        class _FakeEngine:
            def __init__(self, conn: _FakeConn) -> None:
                self._conn = conn

            def begin(self) -> _BeginContext:
                return _BeginContext(self._conn)

        response_payload = {
            "custom_id": "agreement-2|4",
            "response": {
                "status_code": 200,
                "body": {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"text": '{"status":"invalid"}'}],
                        }
                    ]
                },
            },
        }
        out_text = json.dumps(response_payload)
        conn = _FakeConn()
        engine = _FakeEngine(conn)
        client = _FakeClient(out_text)

        class _FakeLog:
            def info(self, *_args: object, **_kwargs: object) -> None:
                return None

            def warning(self, *_args: object, **_kwargs: object) -> None:
                return None

        context = SimpleNamespace(log=_FakeLog())
        batch = SimpleNamespace(output_file_id="file-2")

        updated, parse_errors = _apply_xml_verify_batch_output(
            context=cast(AssetExecutionContext, cast(object, context)),
            engine=engine,
            client=cast(OpenAI, cast(object, client)),
            xml_table="pdx.xml",
            xml_status_reasons_table="pdx.xml_status_reasons",
            batch=batch,
        )

        self.assertEqual(updated, 1)
        self.assertEqual(parse_errors, 0)
        self.assertEqual(len(conn.executed), 5)
        select_sql, _ = conn.executed[0]
        self.assertIn("SELECT reason_code, reason_detail, page_uuid", select_sql)
        _, params = conn.executed[1]
        self.assertEqual(params["agreement_uuid"], "agreement-2")
        self.assertEqual(params["version"], 4)
        self.assertEqual(params["status"], "invalid")
        self.assertEqual(params["reason_code"], XML_REASON_LLM_INVALID)
        self.assertIsNone(params["reason_detail"])
        insert_sql, insert_params = conn.executed[3]
        self.assertIn("INSERT INTO pdx.xml_status_reasons", insert_sql)
        self.assertEqual(insert_params["reason_code"], XML_REASON_LLM_INVALID)
        reset_sql, _ = conn.executed[4]
        self.assertIn("SET ai_repair_attempted = 0", reset_sql)

    def test_xml_verify_asset_hard_invalid_sets_status_source_to_asset(self) -> None:
        class _FakeLog:
            def info(self, *_args: object, **_kwargs: object) -> None:
                return None

            def warning(self, *_args: object, **_kwargs: object) -> None:
                return None

        class _Result:
            def __init__(
                self,
                *,
                rows: list[dict[str, object]] | None = None,
                rowcount: int = 0,
            ) -> None:
                self._rows = rows or []
                self.rowcount = rowcount

            class _Scalars:
                def __init__(self, values: list[object]) -> None:
                    self._values = values

                def all(self) -> list[object]:
                    return self._values

            def mappings(self) -> "_Result":
                return self

            def fetchall(self) -> list[dict[str, object]]:
                return self._rows

            def scalars(self) -> "_Result._Scalars":
                values: list[object] = []
                for row in self._rows:
                    first = next(iter(row.values()), None)
                    values.append(first)
                return _Result._Scalars(values)

        class _FakeConn:
            def __init__(self) -> None:
                self.executed: list[tuple[str, dict[str, object]]] = []

            def execute(
                self,
                statement: object,
                params: dict[str, object] | None = None,
            ) -> _Result:
                sql = str(statement)
                query_params = params or {}
                self.executed.append((sql, query_params))
                if "FROM state_components" in sql and "latest_xml_status IS NULL" in sql:
                    return _Result(rows=[{"agreement_uuid": "agreement-hard-invalid"}])
                if "SELECT agreement_uuid, version, xml" in sql:
                    return _Result(
                        rows=[
                            {
                                "agreement_uuid": "agreement-hard-invalid",
                                "version": 1,
                                "xml": "<document><body><section title='bad'/></body></document>",
                            }
                        ]
                    )
                return _Result(rowcount=1)

        class _BeginContext:
            def __init__(self, conn: _FakeConn) -> None:
                self._conn = conn

            def __enter__(self) -> _FakeConn:
                return self._conn

            def __exit__(self, *_exc: object) -> None:
                return None

        class _FakeEngine:
            def __init__(self, conn: _FakeConn) -> None:
                self._conn = conn

            def begin(self) -> _BeginContext:
                return _BeginContext(self._conn)

        conn = _FakeConn()
        engine = _FakeEngine(conn)
        db = SimpleNamespace(database="pdx", get_engine=lambda: engine)
        pipeline_config = SimpleNamespace(
            xml_agreement_batch_size=10,
            resume_openai_batches=True,
        )
        context = SimpleNamespace(log=_FakeLog())

        with (
            patch("etl.defs.f_xml_asset._oai_client", return_value=SimpleNamespace()),
            patch("etl.defs.f_xml_asset.assert_tables_exist", return_value=None),
            patch("etl.defs.f_xml_asset._fetch_unpulled_xml_verify_batch", return_value=None),
            patch("etl.defs.f_xml_asset.run_post_asset_refresh", return_value=None),
        ):
            _ = xml_verify_asset.node_def.compute_fn.decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                cast(PipelineConfig, cast(object, pipeline_config)),
                ["agreement-hard-invalid"],
            )

        hard_invalid_update_sql, hard_invalid_params = next(
            (sql, params)
            for sql, params in conn.executed
            if "status_reason_code = :reason_code" in sql
        )
        self.assertIn("status_source = 'asset'", hard_invalid_update_sql)
        self.assertIn("status_source <=> 'asset'", hard_invalid_update_sql)
        self.assertIn("status_reason_code = :reason_code", hard_invalid_update_sql)
        self.assertIn("status_reason_detail = :reason_detail", hard_invalid_update_sql)
        self.assertEqual(
            hard_invalid_params["reason_code"], XML_REASON_BODY_STARTS_NON_ARTICLE
        )
        self.assertIn(
            "<body> must start with <article>.",
            str(hard_invalid_params["reason_detail"]),
        )
        self.assertTrue(
            any("INSERT INTO pdx.xml_status_reasons" in sql for sql, _ in conn.executed)
        )


if __name__ == "__main__":
    _ = unittest.main()
