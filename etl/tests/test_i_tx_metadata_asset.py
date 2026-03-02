# pyright: reportAny=false, reportPrivateUsage=false
import json
import unittest
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
    def __init__(self, *, rowcount: int = 0, rows: list[dict[str, object]] | None = None) -> None:
        self.rowcount = rowcount
        self._rows = rows or []

    def mappings(self) -> "_FakeResult":
        return self

    def fetchall(self) -> list[dict[str, object]]:
        return self._rows


class _FakeBeginContext:
    def __init__(self, conn: object) -> None:
        self._conn = conn

    def __enter__(self) -> object:
        return self._conn

    def __exit__(self, *_exc: object) -> None:
        return None


class _FakeOfflineConn:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount
        self.executed_sql: list[str] = []

    def execute(self, statement: object, _params: dict[str, object]) -> _FakeResult:
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
    def __init__(self) -> None:
        self.executed_sql: list[str] = []
        self.last_update_params: dict[str, object] | None = None

    def execute(self, statement: object, params: dict[str, object]) -> _FakeResult:
        sql = str(statement)
        self.executed_sql.append(sql)
        if "SELECT agreement_uuid, target, acquirer, filing_date" in sql:
            return _FakeResult(
                rows=[
                    {
                        "agreement_uuid": "agreement-1",
                        "target": "Target A",
                        "acquirer": "Acquirer A",
                        "filing_date": "2024-01-01",
                    }
                ]
            )
        if "UPDATE pdx.agreements" in sql:
            self.last_update_params = dict(params)
            return _FakeResult(rowcount=1)
        raise AssertionError(f"Unexpected SQL in test: {sql}")


class _FakeWebEngine:
    def __init__(self, conn: _FakeWebConn) -> None:
        self._conn = conn

    def begin(self) -> _FakeBeginContext:
        return _FakeBeginContext(self._conn)


class _FakeResponsesClient:
    def create(self, **_kwargs: object) -> object:
        return SimpleNamespace(output_text='{"transaction_consideration":"cash"}')


class _FakeWebClient:
    def __init__(self) -> None:
        self.responses = _FakeResponsesClient()


class _FakeLog:
    def info(self, *args: object, **kwargs: object) -> None:
        _ = args
        _ = kwargs

    def warning(self, *args: object, **kwargs: object) -> None:
        _ = args
        _ = kwargs


class TxMetadataProjectionRefreshTests(unittest.TestCase):
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

    def test_run_web_search_mode_refreshes_projection_for_updated_agreement(self) -> None:
        conn = _FakeWebConn()
        engine = _FakeWebEngine(conn)
        context = SimpleNamespace(log=_FakeLog())

        with (
            patch("etl.defs.i_tx_metadata_asset._oai_client", return_value=_FakeWebClient()),
            patch(
                "etl.defs.i_tx_metadata_asset.parse_tx_metadata_response_text_web_search",
                return_value={"transaction_consideration": "cash"},
            ),
            patch(
                "etl.defs.i_tx_metadata_asset.build_tx_metadata_update_params_web_search_only",
                return_value={
                    "uuid": "agreement-1",
                    "consideration": "cash",
                    "price_cash": None,
                    "price_stock": None,
                    "price_assets": None,
                    "price_total": None,
                    "target_type": None,
                    "acquirer_type": None,
                    "target_pe": None,
                    "acquirer_pe": None,
                    "target_industry": None,
                    "acquirer_industry": None,
                    "announce_date": None,
                    "close_date": None,
                    "deal_status": None,
                    "attitude": None,
                    "purpose": None,
                    "metadata_sources": None,
                },
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


if __name__ == "__main__":
    _ = unittest.main()
