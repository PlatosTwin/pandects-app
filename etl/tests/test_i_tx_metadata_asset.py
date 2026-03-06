# pyright: reportAny=false, reportPrivateUsage=false
import json
import unittest
from datetime import date
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from dagster import AssetExecutionContext
from openai import OpenAI

from etl.defs.i_tx_metadata_asset import (
    _apply_offline_batch_output,
    _run_web_search_mode,
)


class _FakeResult:
    def __init__(
        self,
        *,
        rowcount: int = 0,
        rows: list[dict[str, object]] | None = None,
        scalar_rows: list[object] | None = None,
    ) -> None:
        self.rowcount = rowcount
        self._rows = rows or []
        self._scalar_rows = scalar_rows or []

    def mappings(self) -> "_FakeResult":
        return self

    def fetchall(self) -> list[dict[str, object]]:
        return self._rows

    def scalars(self) -> "_FakeResult":
        return self

    def all(self) -> list[object]:
        return self._scalar_rows


class _FakeBeginContext:
    def __init__(self, conn: object) -> None:
        self._conn = conn

    def __enter__(self) -> object:
        return self._conn

    def __exit__(self, *_exc: object) -> None:
        return None


class _FakeOfflineConn:
    def __init__(self, rowcount: int, *, fail_on_execute: bool = False) -> None:
        self.rowcount = rowcount
        self.fail_on_execute = fail_on_execute
        self.executed_sql: list[str] = []

    def execute(self, statement: object, _params: dict[str, object]) -> _FakeResult:
        if self.fail_on_execute:
            raise RuntimeError("database write failed")
        self.executed_sql.append(str(statement))
        return _FakeResult(rowcount=self.rowcount)


class _FakeOfflineEngine:
    def __init__(self, conn: _FakeOfflineConn) -> None:
        self._conn = conn

    def begin(self) -> _FakeBeginContext:
        return _FakeBeginContext(self._conn)


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


class _FakeOfflineClient:
    def __init__(self, text_value: str) -> None:
        self.files = _FakeFiles(text_value)


class _FakeWebConn:
    def __init__(
        self,
        *,
        fail_on_update: bool = False,
        select_rows: list[dict[str, object]] | None = None,
    ) -> None:
        self.executed_sql: list[str] = []
        self.last_update_params: dict[str, object] | None = None
        self.last_select_params: dict[str, object] | None = None
        self.fail_on_update = fail_on_update
        self.failure_upserts = 0
        self.failure_clears = 0
        today_iso = date.today().isoformat()
        self.select_rows = select_rows or [
            {
                "agreement_uuid": "agreement-1",
                "target": "Target A",
                "acquirer": "Acquirer A",
                "filing_date": today_iso,
                "url": "https://www.sec.gov/Archives/edgar/data/123/abc.htm",
            }
        ]

    def execute(self, statement: object, params: dict[str, object]) -> _FakeResult:
        sql = str(statement)
        self.executed_sql.append(sql)
        if "FROM information_schema.tables" in sql:
            return _FakeResult(scalar_rows=["tx_metadata_web_failures"])
        if "SELECT a.agreement_uuid, a.target, a.acquirer, a.filing_date, a.url" in sql:
            self.last_select_params = dict(params)
            return _FakeResult(rows=self.select_rows)
        if "INSERT INTO pdx.tx_metadata_web_failures" in sql:
            self.failure_upserts += 1
            return _FakeResult(rowcount=1)
        if "DELETE FROM pdx.tx_metadata_web_failures" in sql:
            self.failure_clears += 1
            return _FakeResult(rowcount=1)
        if "UPDATE pdx.agreements" in sql:
            if self.fail_on_update:
                raise RuntimeError("database update failed")
            self.last_update_params = dict(params)
            return _FakeResult(rowcount=1)
        raise AssertionError(f"Unexpected SQL in test: {sql}")


class _FakeWebEngine:
    def __init__(self, conn: _FakeWebConn) -> None:
        self._conn = conn

    def begin(self) -> _FakeBeginContext:
        return _FakeBeginContext(self._conn)


class _FakeResponsesClient:
    def __init__(self, *, response_text: str) -> None:
        self._response_text = response_text

    def create(self, **_kwargs: object) -> object:
        return SimpleNamespace(output_text=self._response_text)


class _FlakyResponsesClient:
    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self.calls = 0

    def create(self, **_kwargs: object) -> object:
        if self.calls >= len(self._responses):
            raise RuntimeError("no more fake responses configured")
        value = self._responses[self.calls]
        self.calls += 1
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(output_text=value)


class _FakeWebClient:
    def __init__(self, *, response_text: str) -> None:
        self.responses = _FakeResponsesClient(response_text=response_text)


class _FakeLog:
    def info(self, *args: object, **kwargs: object) -> None:
        _ = args
        _ = kwargs

    def warning(self, *args: object, **kwargs: object) -> None:
        _ = args
        _ = kwargs


class TxMetadataProjectionRefreshTests(unittest.TestCase):
    def _valid_web_search_payload(self) -> str:
        return json.dumps(
            {
                "consideration_type": "all_cash",
                "purchase_price": {"cash": 100.0, "stock": 0.0, "assets": 0.0},
                "target_public": True,
                "acquirer_public": False,
                "target_pe": None,
                "acquirer_pe": None,
                "target_industry": "311",
                "acquirer_industry": "52",
                "announce_date": "2024-01-01",
                "close_date": None,
                "deal_status": "pending",
                "attitude": "friendly",
                "purpose": "strategic",
                "metadata_sources": {
                    "citations": [
                        {
                            "url": "https://example.com/deal",
                            "fields": [
                                "consideration_type",
                                "purchase_price.cash",
                                "purchase_price.stock",
                                "purchase_price.assets",
                                "target_public",
                                "acquirer_public",
                                "target_industry",
                                "acquirer_industry",
                                "announce_date",
                                "deal_status",
                                "attitude",
                                "purpose",
                            ],
                        }
                    ],
                    "notes": None,
                },
            }
        )

    def test_apply_offline_batch_output_refreshes_projection_for_updated_agreement(self) -> None:
        response_payload = {
            "custom_id": "agreement-1",
            "response": {
                "status_code": 200,
                "body": {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"text": '{"target":"Target A","acquirer":"Acquirer A","deal_type":"merger"}'}],
                        }
                    ]
                },
            },
        }
        client = _FakeOfflineClient(json.dumps(response_payload))
        conn = _FakeOfflineConn(rowcount=1)
        engine = _FakeOfflineEngine(conn)
        context = SimpleNamespace(log=_FakeLog())

        with patch("etl.defs.i_tx_metadata_asset.refresh_latest_sections_search") as refresh:
            updated, parse_errors, refreshed_uuids = _apply_offline_batch_output(
                context=cast(AssetExecutionContext, cast(object, context)),
                engine=engine,
                client=cast(OpenAI, cast(object, client)),
                schema="pdx",
                agreements_table="pdx.agreements",
                batch=SimpleNamespace(output_file_id="file-1"),
            )

        self.assertEqual(updated, 1)
        self.assertEqual(parse_errors, 0)
        self.assertEqual(refreshed_uuids, ["agreement-1"])
        refresh.assert_called_once_with(conn, "pdx", ["agreement-1"])

    def test_apply_offline_batch_output_raises_on_database_error(self) -> None:
        response_payload = {
            "custom_id": "agreement-1",
            "response": {
                "status_code": 200,
                "body": {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"text": '{"target":"Target A","acquirer":"Acquirer A","deal_type":"merger"}'}],
                        }
                    ]
                },
            },
        }
        client = _FakeOfflineClient(json.dumps(response_payload))
        conn = _FakeOfflineConn(rowcount=1, fail_on_execute=True)
        engine = _FakeOfflineEngine(conn)
        context = SimpleNamespace(log=_FakeLog())

        with self.assertRaisesRegex(RuntimeError, "database write failed"):
            _ = _apply_offline_batch_output(
                context=cast(AssetExecutionContext, cast(object, context)),
                engine=engine,
                client=cast(OpenAI, cast(object, client)),
                schema="pdx",
                agreements_table="pdx.agreements",
                batch=SimpleNamespace(output_file_id="file-1"),
            )

    def test_run_web_search_mode_refreshes_projection_for_updated_agreement(self) -> None:
        conn = _FakeWebConn()
        engine = _FakeWebEngine(conn)
        context = SimpleNamespace(log=_FakeLog())

        with (
            patch(
                "etl.defs.i_tx_metadata_asset._oai_client",
                return_value=_FakeWebClient(response_text=self._valid_web_search_payload()),
            ),
            patch("etl.defs.i_tx_metadata_asset.refresh_latest_sections_search") as refresh,
        ):
            _run_web_search_mode(
                context=cast(AssetExecutionContext, cast(object, context)),
                engine=engine,
                schema="pdx",
                agreements_table="pdx.agreements",
                batch_size=10,
            )

        refresh.assert_called_once_with(conn, "pdx", ["agreement-1"])
        self.assertEqual(conn.last_select_params, {"lim": 10, "max_failures": 3})

    def test_run_web_search_mode_raises_on_database_update_error(self) -> None:
        conn = _FakeWebConn(fail_on_update=True)
        engine = _FakeWebEngine(conn)
        context = SimpleNamespace(log=_FakeLog())

        with patch(
            "etl.defs.i_tx_metadata_asset._oai_client",
            return_value=_FakeWebClient(response_text=self._valid_web_search_payload()),
        ):
            with self.assertRaisesRegex(RuntimeError, "database update failed"):
                _run_web_search_mode(
                    context=cast(AssetExecutionContext, cast(object, context)),
                    engine=engine,
                    schema="pdx",
                    agreements_table="pdx.agreements",
                    batch_size=10,
                )

    def test_run_web_search_mode_retries_transient_response_error(self) -> None:
        conn = _FakeWebConn()
        engine = _FakeWebEngine(conn)
        context = SimpleNamespace(log=_FakeLog())
        flaky_responses = _FlakyResponsesClient(
            responses=[
                RuntimeError("transient web-search failure"),
                self._valid_web_search_payload(),
            ]
        )
        flaky_client = SimpleNamespace(responses=flaky_responses)

        with (
            patch("etl.defs.i_tx_metadata_asset._oai_client", return_value=flaky_client),
            patch("etl.defs.i_tx_metadata_asset.time.sleep", return_value=None),
            patch("etl.defs.i_tx_metadata_asset.refresh_latest_sections_search") as refresh,
        ):
            _run_web_search_mode(
                context=cast(AssetExecutionContext, cast(object, context)),
                engine=engine,
                schema="pdx",
                agreements_table="pdx.agreements",
                batch_size=10,
            )

        self.assertEqual(flaky_responses.calls, 2)
        refresh.assert_called_once_with(conn, "pdx", ["agreement-1"])
        self.assertEqual(conn.failure_upserts, 0)
        self.assertGreaterEqual(conn.failure_clears, 1)

    def test_run_web_search_mode_processes_url_only_agreement(self) -> None:
        conn = _FakeWebConn(
            select_rows=[
                {
                    "agreement_uuid": "agreement-url-only",
                    "target": None,
                    "acquirer": None,
                    "filing_date": date.today().isoformat(),
                    "url": "https://www.sec.gov/Archives/edgar/data/999/only-url.htm",
                }
            ]
        )
        engine = _FakeWebEngine(conn)
        context = SimpleNamespace(log=_FakeLog())

        with (
            patch(
                "etl.defs.i_tx_metadata_asset._oai_client",
                return_value=_FakeWebClient(response_text=self._valid_web_search_payload()),
            ),
            patch("etl.defs.i_tx_metadata_asset.refresh_latest_sections_search") as refresh,
        ):
            _run_web_search_mode(
                context=cast(AssetExecutionContext, cast(object, context)),
                engine=engine,
                schema="pdx",
                agreements_table="pdx.agreements",
                batch_size=10,
            )

        refresh.assert_called_once_with(conn, "pdx", ["agreement-url-only"])
        self.assertEqual(conn.failure_upserts, 0)
        self.assertGreaterEqual(conn.failure_clears, 1)

    def test_run_web_search_mode_nulls_old_pending_filing(self) -> None:
        conn = _FakeWebConn(
            select_rows=[
                {
                    "agreement_uuid": "agreement-old",
                    "target": "Target A",
                    "acquirer": "Acquirer A",
                    "filing_date": "2000-01-01",
                    "url": "https://www.sec.gov/Archives/edgar/data/999/old.htm",
                }
            ]
        )
        engine = _FakeWebEngine(conn)
        context = SimpleNamespace(log=_FakeLog())

        with (
            patch(
                "etl.defs.i_tx_metadata_asset._oai_client",
                return_value=_FakeWebClient(response_text=self._valid_web_search_payload()),
            ),
            patch("etl.defs.i_tx_metadata_asset.refresh_latest_sections_search") as refresh,
        ):
            _run_web_search_mode(
                context=cast(AssetExecutionContext, cast(object, context)),
                engine=engine,
                schema="pdx",
                agreements_table="pdx.agreements",
                batch_size=10,
            )

        refresh.assert_called_once_with(conn, "pdx", ["agreement-old"])
        self.assertIsNotNone(conn.last_update_params)
        assert conn.last_update_params is not None
        self.assertIsNone(conn.last_update_params["deal_status"])
        self.assertEqual(conn.failure_upserts, 0)
        self.assertGreaterEqual(conn.failure_clears, 1)

    def test_run_web_search_mode_records_request_failure(self) -> None:
        conn = _FakeWebConn()
        engine = _FakeWebEngine(conn)
        context = SimpleNamespace(log=_FakeLog())
        flaky_responses = _FlakyResponsesClient(
            responses=[
                RuntimeError("transient web-search failure"),
                RuntimeError("still failing"),
                RuntimeError("final failure"),
            ]
        )
        flaky_client = SimpleNamespace(responses=flaky_responses)

        with (
            patch("etl.defs.i_tx_metadata_asset._oai_client", return_value=flaky_client),
            patch("etl.defs.i_tx_metadata_asset.time.sleep", return_value=None),
        ):
            _run_web_search_mode(
                context=cast(AssetExecutionContext, cast(object, context)),
                engine=engine,
                schema="pdx",
                agreements_table="pdx.agreements",
                batch_size=10,
            )

        self.assertEqual(conn.failure_upserts, 1)
        self.assertEqual(conn.failure_clears, 0)


if __name__ == "__main__":
    _ = unittest.main()
