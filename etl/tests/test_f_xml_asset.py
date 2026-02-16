# pyright: reportAny=false, reportPrivateUsage=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportUnknownMemberType=false
import json
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from dagster import AssetExecutionContext
from openai import OpenAI
from etl.defs.resources import DBResource, PipelineConfig
from etl.defs.f_xml_asset import (
    XML_REASON_BODY_STARTS_NON_ARTICLE,
    XML_REASON_LLM_INVALID,
    _apply_xml_verify_batch_output,
    xml_verify_asset,
)


class XMLVerifyAssetTests(unittest.TestCase):
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
            def __init__(self, rowcount: int) -> None:
                self.rowcount = rowcount

        class _FakeConn:
            def __init__(self) -> None:
                self.executed: list[tuple[str, dict[str, object]]] = []

            def execute(self, statement: object, params: dict[str, object]) -> _FakeResult:
                self.executed.append((str(statement), params))
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
            batch=batch,
        )

        self.assertEqual(updated, 1)
        self.assertEqual(parse_errors, 0)
        self.assertEqual(len(conn.executed), 1)
        executed_sql, params = conn.executed[0]
        self.assertIn("status_source = 'asset'", executed_sql)
        self.assertIn("status_reason_code = :reason_code", executed_sql)
        self.assertIn("status_reason_detail = :reason_detail", executed_sql)
        self.assertEqual(params["agreement_uuid"], "agreement-1")
        self.assertEqual(params["version"], 3)
        self.assertEqual(params["status"], "verified")
        self.assertIsNone(params["reason_code"])
        self.assertIsNone(params["reason_detail"])

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
            def __init__(self, rowcount: int) -> None:
                self.rowcount = rowcount

        class _FakeConn:
            def __init__(self) -> None:
                self.executed: list[tuple[str, dict[str, object]]] = []

            def execute(self, statement: object, params: dict[str, object]) -> _FakeResult:
                self.executed.append((str(statement), params))
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
            def warning(self, *_args: object, **_kwargs: object) -> None:
                return None

        context = SimpleNamespace(log=_FakeLog())
        batch = SimpleNamespace(output_file_id="file-2")

        updated, parse_errors = _apply_xml_verify_batch_output(
            context=cast(AssetExecutionContext, cast(object, context)),
            engine=engine,
            client=cast(OpenAI, cast(object, client)),
            xml_table="pdx.xml",
            batch=batch,
        )

        self.assertEqual(updated, 1)
        self.assertEqual(parse_errors, 0)
        self.assertEqual(len(conn.executed), 1)
        _, params = conn.executed[0]
        self.assertEqual(params["agreement_uuid"], "agreement-2")
        self.assertEqual(params["version"], 4)
        self.assertEqual(params["status"], "invalid")
        self.assertEqual(params["reason_code"], XML_REASON_LLM_INVALID)
        self.assertIsNone(params["reason_detail"])

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

            def mappings(self) -> "_Result":
                return self

            def fetchall(self) -> list[dict[str, object]]:
                return self._rows

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
        pipeline_config = SimpleNamespace(xml_verify_batch_size=10)
        context = SimpleNamespace(log=_FakeLog())

        with (
            patch("etl.defs.f_xml_asset._oai_client", return_value=SimpleNamespace()),
            patch("etl.defs.f_xml_asset._ensure_xml_verify_batches_table", return_value=None),
            patch("etl.defs.f_xml_asset._fetch_unpulled_xml_verify_batch", return_value=None),
            patch("etl.defs.f_xml_asset.run_post_asset_refresh", return_value=None),
        ):
            _ = xml_verify_asset.node_def.compute_fn.decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                cast(PipelineConfig, cast(object, pipeline_config)),
            )

        hard_invalid_update_sql, hard_invalid_params = next(
            (sql, params) for sql, params in conn.executed if "SET status = 'invalid'" in sql
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


if __name__ == "__main__":
    _ = unittest.main()
