# pyright: reportAny=false, reportPrivateUsage=false
import json
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from dagster import AssetExecutionContext

from etl.defs.d_ai_repair_asset import ai_repair_poll_asset
from etl.defs.resources import DBResource, PipelineConfig


class _FakeResult:
    def __init__(
        self,
        *,
        rows: list[object] | None = None,
        scalar_one_value: int | None = None,
    ) -> None:
        self._rows = rows or []
        self._scalar_one_value = scalar_one_value

    def mappings(self) -> "_FakeResult":
        return self

    def fetchall(self) -> list[object]:
        return self._rows

    def scalar_one(self) -> int:
        if self._scalar_one_value is None:
            raise AssertionError("scalar_one() was called without scalar value.")
        return self._scalar_one_value


class _FakeConn:
    def __init__(self) -> None:
        self._list_batches_calls = 0
        self.rulings_insert_params: list[dict[str, object]] = []
        self.usage_update_params: list[dict[str, object]] = []

    def execute(
        self, statement: object, params: dict[str, object] | None = None
    ) -> _FakeResult:
        sql = str(statement)
        safe_params = params or {}

        if "SELECT DISTINCT b.batch_id" in sql:
            self._list_batches_calls += 1
            if self._list_batches_calls == 1:
                return _FakeResult(rows=[{"batch_id": "batch-1"}])
            return _FakeResult(rows=[])
        if "UPDATE pdx.ai_repair_batches" in sql:
            return _FakeResult()
        if "FROM pdx.ai_repair_requests WHERE batch_id = :bid" in sql:
            return _FakeResult(
                rows=[
                    SimpleNamespace(
                        request_id="rid-1",
                        page_uuid="page-1",
                        mode="excerpt",
                        excerpt_start=0,
                    ),
                    SimpleNamespace(
                        request_id="rid-2",
                        page_uuid="page-2",
                        mode="excerpt",
                        excerpt_start=0,
                    ),
                ]
            )
        if "SELECT page_uuid, processed_page_content AS text" in sql:
            return _FakeResult(
                rows=[
                    {"page_uuid": "page-1", "text": "alpha"},
                    {"page_uuid": "page-2", "text": "beta"},
                ]
            )
        if "UPDATE pdx.ai_repair_requests SET token_usage = :usage" in sql:
            self.usage_update_params.append(dict(safe_params))
            return _FakeResult()
        if "INSERT INTO pdx.ai_repair_rulings" in sql:
            self.rulings_insert_params.append(dict(safe_params))
            return _FakeResult()
        if "SELECT COUNT(*)" in sql and "r.status IN ('queued', 'running')" in sql:
            return _FakeResult(scalar_one_value=0)
        raise AssertionError(f"Unexpected SQL in test: {sql}")


class _FakeBeginContext:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def __enter__(self) -> _FakeConn:
        return self._conn

    def __exit__(self, *_exc: object) -> None:
        return None


class _FakeEngine:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def begin(self) -> _FakeBeginContext:
        return _FakeBeginContext(self._conn)


class _FakeContent:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def text(self) -> str:
        return self._payload


class _FakeFiles:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def content(self, _file_id: str) -> _FakeContent:
        return _FakeContent(self._payload)


class _FakeBatches:
    def retrieve(self, _batch_id: str) -> object:
        return SimpleNamespace(
            status="expired",
            output_file_id="output-file-1",
            error_file_id=None,
            request_counts=SimpleNamespace(total=2, failed=1, completed=1),
        )


class _FakeClient:
    def __init__(self, output_payload: str) -> None:
        self.batches = _FakeBatches()
        self.files = _FakeFiles(output_payload)


class _FakeLog:
    def info(self, *_args: object, **_kwargs: object) -> None:
        return None

    def warning(self, *_args: object, **_kwargs: object) -> None:
        return None

    def error(self, *_args: object, **_kwargs: object) -> None:
        return None


class _FakeDB:
    def __init__(self, conn: _FakeConn) -> None:
        self.database = "pdx"
        self._engine = _FakeEngine(conn)

    def get_engine(self) -> _FakeEngine:
        return self._engine


class AIRepairPollAssetTests(unittest.TestCase):
    def test_expired_batch_keeps_parsed_results_and_only_fails_leftovers(self) -> None:
        output_line = {
            "custom_id": "rid-1",
            "response": {
                "status_code": 200,
                "body": {
                    "usage": {
                        "input_tokens": 7,
                        "output_tokens": 3,
                        "total_tokens": 10,
                    },
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "text": json.dumps(
                                        {
                                            "rulings": [
                                                {
                                                    "start_char": 0,
                                                    "end_char": 5,
                                                    "label": "page",
                                                }
                                            ],
                                            "warnings": [],
                                        }
                                    )
                                }
                            ],
                        }
                    ],
                },
            },
        }
        conn = _FakeConn()
        db = _FakeDB(conn)
        context = SimpleNamespace(log=_FakeLog())
        pipeline_config = cast(PipelineConfig, cast(object, SimpleNamespace()))
        status_updates: list[tuple[set[str], str]] = []
        completed_updates: list[set[str]] = []

        def _capture_bulk_status(
            _conn: object, _schema: str, request_ids: set[str], status: str
        ) -> None:
            status_updates.append((set(request_ids), status))

        def _capture_mark_completed(
            _conn: object, _schema: str, request_ids: set[str]
        ) -> None:
            completed_updates.append(set(request_ids))

        with (
            patch(
                "etl.defs.d_ai_repair_asset._oai_client",
                return_value=_FakeClient(json.dumps(output_line)),
            ),
            patch(
                "etl.defs.d_ai_repair_asset._bulk_update_status",
                side_effect=_capture_bulk_status,
            ),
            patch(
                "etl.defs.d_ai_repair_asset._mark_completed",
                side_effect=_capture_mark_completed,
            ),
            patch("etl.defs.d_ai_repair_asset.run_post_asset_refresh"),
        ):
            poll_fn = getattr(ai_repair_poll_asset.op.compute_fn, "decorated_fn")
            with self.assertRaises(RuntimeError):
                _ = poll_fn(
                    cast(AssetExecutionContext, cast(object, context)),
                    cast(DBResource, cast(object, db)),
                    pipeline_config,
                    ["agreement-1"],
                )

        self.assertIn({"rid-1"}, completed_updates)
        self.assertIn(({"rid-2"}, "failed"), status_updates)
        self.assertNotIn(({"rid-1"}, "failed"), status_updates)
        self.assertEqual(
            conn.rulings_insert_params,
            [
                {
                    "rid": "rid-1",
                    "pid": "page-1",
                    "s": 0,
                    "e": 5,
                    "lab": "page",
                    "bid": "batch-1",
                }
            ],
        )


if __name__ == "__main__":
    _ = unittest.main()
