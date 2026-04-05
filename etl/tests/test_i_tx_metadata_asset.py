# pyright: reportAny=false, reportPrivateUsage=false, reportAttributeAccessIssue=false, reportUnknownMemberType=false, reportUnknownVariableType=false
import json
import threading
import unittest
from datetime import date
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from dagster import AssetExecutionContext
from openai import OpenAI

from etl.defs.resources import PipelineConfig, QueueRunMode
from etl.defs.i_tx_metadata_asset import (
    _apply_offline_batch_output,
    _build_offline_counsel_lines,
    _build_offline_metadata_lines,
    _counsel_offline_update_sql,
    _run_offline_mode,
    _run_web_search_mode,
    _web_search_missing_core_metadata_sql,
    tx_metadata_asset,
)
from etl.domain.i_tx_metadata import (
    build_offline_counsel_update_params,
    parse_offline_counsel_response_text,
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
        self.update_params_history: list[dict[str, object]] = []
        self.last_select_params: dict[str, object] | None = None
        self.last_select_rows: list[dict[str, object]] | None = None
        self.fail_on_update = fail_on_update
        self.failure_upserts = 0
        self.failure_clears = 0
        self.update_calls = 0
        today_iso = date.today().isoformat()
        self.select_rows = select_rows or [
            {
                "agreement_uuid": "agreement-1",
                "target": "Target A",
                "acquirer": "Acquirer A",
                "filing_date": today_iso,
                "url": "https://www.sec.gov/Archives/edgar/data/123/abc.htm",
                "failure_count": 0,
                "initial_metadata_pass": 1,
            }
        ]

    def execute(self, statement: object, params: dict[str, object]) -> _FakeResult:
        sql = str(statement)
        self.executed_sql.append(sql)
        if "FROM information_schema.tables" in sql:
            return _FakeResult(scalar_rows=["tx_metadata_web_failures"])
        if "SELECT" in sql and "FROM pdx.agreements a" in sql:
            self.last_select_params = dict(params)
            self.last_select_rows = self.select_rows
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
            self.update_params_history.append(dict(params))
            self.update_calls += 1
            return _FakeResult(rowcount=1)
        raise AssertionError(f"Unexpected SQL in test: {sql}")


class _FakeWebEngine:
    def __init__(self, conn: _FakeWebConn) -> None:
        self._conn = conn

    def begin(self) -> _FakeBeginContext:
        return _FakeBeginContext(self._conn)


class _FakeCounselSelectionConn:
    def __init__(
        self,
        *,
        candidate_rows: list[dict[str, object]],
        section_rows: list[dict[str, object]],
    ) -> None:
        self.candidate_rows = candidate_rows
        self.section_rows = section_rows
        self.executed_sql: list[str] = []

    def execute(self, statement: object, _params: dict[str, object]) -> _FakeResult:
        sql = str(statement)
        self.executed_sql.append(sql)
        if "FROM pdx.latest_sections_search_standard_ids lssi" in sql and "s.xml_content" in sql:
            return _FakeResult(rows=self.section_rows)
        return _FakeResult(rows=self.candidate_rows)


class _FakeCounselSelectionEngine:
    def __init__(self, conn: _FakeCounselSelectionConn) -> None:
        self._conn = conn

    def begin(self) -> _FakeBeginContext:
        return _FakeBeginContext(self._conn)


class _FakeMetadataSelectionConn:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.executed_sql: list[str] = []

    def execute(self, statement: object, _params: dict[str, object]) -> _FakeResult:
        self.executed_sql.append(str(statement))
        return _FakeResult(rows=self.rows)


class _FakeMetadataSelectionEngine:
    def __init__(self, conn: _FakeMetadataSelectionConn) -> None:
        self._conn = conn

    def begin(self) -> _FakeBeginContext:
        return _FakeBeginContext(self._conn)


class _NoopConn:
    pass


class _NoopEngine:
    def begin(self) -> _FakeBeginContext:
        return _FakeBeginContext(_NoopConn())


def _fake_web_search_output(search_count: int) -> list[object]:
    return [
        SimpleNamespace(
            type="web_search_call",
            action=SimpleNamespace(type="search"),
        )
        for _ in range(search_count)
    ]


class _FakeResponsesClient:
    def __init__(
        self,
        *,
        response_text: str,
        usage: dict[str, int] | None = None,
        search_count: int = 1,
    ) -> None:
        self._response_text = response_text
        self._usage = usage or {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        self._search_count = search_count
        self.models: list[object] = []
        self.inputs: list[object] = []

    def create(self, **_kwargs: object) -> object:
        self.models.append(_kwargs.get("model"))
        self.inputs.append(_kwargs.get("input"))
        return SimpleNamespace(
            output_text=self._response_text,
            usage=self._usage,
            output=_fake_web_search_output(self._search_count),
        )


class _RoutingResponsesClient:
    def __init__(
        self,
        responses_by_input_fragment: dict[str, object],
        *,
        usage: dict[str, int] | None = None,
        search_count: int = 1,
    ) -> None:
        self._responses_by_input_fragment = responses_by_input_fragment
        self._usage = usage or {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        self._search_count = search_count
        self.models: list[object] = []
        self.inputs: list[object] = []

    def create(self, **kwargs: object) -> object:
        self.models.append(kwargs.get("model"))
        input_text = kwargs.get("input")
        self.inputs.append(input_text)
        if not isinstance(input_text, str):
            raise TypeError("input must be a string in fake client.")
        for fragment, response in self._responses_by_input_fragment.items():
            if fragment not in input_text:
                continue
            if isinstance(response, Exception):
                raise response
            return SimpleNamespace(
                output_text=response,
                usage=self._usage,
                output=_fake_web_search_output(self._search_count),
            )
        raise RuntimeError(f"no fake response configured for input: {input_text}")


class _FlakyResponsesClient:
    def __init__(self, responses: list[object], *, search_count: int = 1) -> None:
        self._responses = responses
        self.calls = 0
        self._search_count = search_count
        self._lock = threading.Lock()

    def create(self, **_kwargs: object) -> object:
        with self._lock:
            if self.calls >= len(self._responses):
                raise RuntimeError("no more fake responses configured")
            value = self._responses[self.calls]
            self.calls += 1
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(
            output_text=value,
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            output=_fake_web_search_output(self._search_count),
        )


class _FakeWebClient:
    def __init__(
        self,
        *,
        response_text: str,
        usage: dict[str, int] | None = None,
        search_count: int = 1,
    ) -> None:
        self.responses = _FakeResponsesClient(
            response_text=response_text,
            usage=usage,
            search_count=search_count,
        )


class _RoutingWebClient:
    def __init__(
        self,
        responses_by_input_fragment: dict[str, object],
        *,
        usage: dict[str, int] | None = None,
        search_count: int = 1,
    ) -> None:
        self.responses = _RoutingResponsesClient(
            responses_by_input_fragment=responses_by_input_fragment,
            usage=usage,
            search_count=search_count,
        )


class _FakeLog:
    def __init__(self) -> None:
        self.info_calls: list[tuple[object, ...]] = []
        self.warning_calls: list[tuple[object, ...]] = []

    def info(self, *args: object, **kwargs: object) -> None:
        self.info_calls.append(args)
        _ = kwargs

    def warning(self, *args: object, **kwargs: object) -> None:
        self.warning_calls.append(args)
        _ = kwargs


class TxMetadataProjectionRefreshTests(unittest.TestCase):
    def _agreement_row(
        self,
        agreement_uuid: str,
        *,
        failure_count: int = 0,
        initial_metadata_pass: int = 1,
        url: str | None = None,
        **overrides: object,
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "agreement_uuid": agreement_uuid,
            "target": f"Target {agreement_uuid}",
            "acquirer": f"Acquirer {agreement_uuid}",
            "filing_date": date.today().isoformat(),
            "url": url or f"https://www.sec.gov/Archives/edgar/data/{agreement_uuid}.htm",
            "transaction_consideration": "cash" if initial_metadata_pass == 0 else None,
            "transaction_price_cash": 100.0 if initial_metadata_pass == 0 else None,
            "transaction_price_stock": None,
            "transaction_price_assets": None,
            "transaction_price_total": 100.0 if initial_metadata_pass == 0 else None,
            "target_type": "public" if initial_metadata_pass == 0 else None,
            "acquirer_type": "private" if initial_metadata_pass == 0 else None,
            "target_pe": None,
            "acquirer_pe": None,
            "target_industry": "311" if initial_metadata_pass == 0 else None,
            "acquirer_industry": "52" if initial_metadata_pass == 0 else None,
            "announce_date": "2024-01-01" if initial_metadata_pass == 0 else None,
            "close_date": None,
            "deal_status": "complete" if initial_metadata_pass == 0 else None,
            "attitude": "friendly" if initial_metadata_pass == 0 else None,
            "purpose": "strategic" if initial_metadata_pass == 0 else None,
            "metadata_sources": json.dumps(
                {
                    "metadata_sources": {
                        "citations": [
                            self._citation("consideration_type"),
                            self._citation("purchase_price.cash"),
                            self._citation("target_public"),
                            self._citation("acquirer_public"),
                            self._citation("target_industry"),
                            self._citation("acquirer_industry"),
                            self._citation("announce_date"),
                            self._citation("deal_status"),
                            self._citation("attitude"),
                            self._citation("purpose"),
                        ] if initial_metadata_pass == 0 else [],
                        "notes": None,
                    },
                    "metadata_run_stats": {"token_usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}, "search_count": 1},
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ) if initial_metadata_pass == 0 else None,
            "metadata_uncited_fields": json.dumps(["close_date"], ensure_ascii=False, separators=(",", ":")) if initial_metadata_pass == 0 else None,
            "failure_count": failure_count,
            "initial_metadata_pass": initial_metadata_pass,
        }
        row.update(overrides)
        return row

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
                        self._citation("consideration_type"),
                        self._citation("purchase_price.cash"),
                        self._citation("purchase_price.stock"),
                        self._citation("purchase_price.assets"),
                        self._citation("target_public"),
                        self._citation("acquirer_public"),
                        self._citation("target_industry"),
                        self._citation("acquirer_industry"),
                        self._citation("announce_date"),
                        self._citation("deal_status"),
                        self._citation("attitude"),
                        self._citation("purpose"),
                    ],
                    "notes": None,
                },
            }
        )

    def _citation(self, field: str) -> dict[str, object]:
        return {
            "field": field,
            "url": "https://example.com/deal",
            "source_type": "company_press_release",
            "published_at": "2024-01-01",
            "locator": "press release",
            "excerpt": f"Support for {field}.",
        }

    def test_web_search_missing_core_metadata_sql_is_consideration_aware(self) -> None:
        sql = _web_search_missing_core_metadata_sql(alias="a")

        self.assertIn("COALESCE(a.transaction_consideration, '') = 'cash'", sql)
        self.assertIn("NOT (a.transaction_price_cash IS NOT NULL AND TRIM(a.transaction_price_cash) <> '')", sql)
        self.assertIn("COALESCE(a.transaction_consideration, '') = 'stock'", sql)
        self.assertIn("NOT (a.transaction_price_stock IS NOT NULL AND TRIM(a.transaction_price_stock) <> '')", sql)
        self.assertIn("COALESCE(a.transaction_consideration, '') = 'mixed'", sql)
        self.assertIn("CASE WHEN a.transaction_price_cash IS NOT NULL AND TRIM(a.transaction_price_cash) <> '' THEN 1 ELSE 0 END", sql)
        self.assertIn("NOT (a.target_type IS NOT NULL AND TRIM(a.target_type) <> '')", sql)
        self.assertIn("NOT (a.acquirer_type IS NOT NULL AND TRIM(a.acquirer_type) <> '')", sql)

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
            updated, parse_errors, refreshed_uuids, processed_uuids = _apply_offline_batch_output(
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
        self.assertEqual(processed_uuids, ["agreement-1"])
        refresh.assert_called_once_with(conn, "pdx", ["agreement-1"])

    def test_tx_metadata_asset_fails_fast_when_queue_run_mode_is_not_single_batch(self) -> None:
        context = SimpleNamespace(log=_FakeLog())
        db = SimpleNamespace()
        pipeline_config = PipelineConfig(queue_run_mode=QueueRunMode.DRAIN)

        with patch("etl.defs.i_tx_metadata_asset.run_pre_asset_gating", return_value=None):
            with self.assertRaisesRegex(
                ValueError,
                "tx_metadata_asset requires pipeline_config.queue_run_mode='single_batch'",
            ):
                _ = tx_metadata_asset.node_def.compute_fn.decorated_fn(
                    cast(AssetExecutionContext, cast(object, context)),
                    cast(object, db),
                    pipeline_config,
                )

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

    def test_apply_offline_batch_output_supports_counsel_updates(self) -> None:
        response_payload = {
            "custom_id": "agreement-1",
            "response": {
                "status_code": 200,
                "body": {
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "text": (
                                        '{"target_counsel":"Wachtell, Lipton, Rosen & Katz",'
                                        '"acquirer_counsel":"Kirkland & Ellis LLP"}'
                                    )
                                }
                            ],
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
            updated, parse_errors, refreshed_uuids, processed_uuids = _apply_offline_batch_output(
                context=cast(AssetExecutionContext, cast(object, context)),
                engine=engine,
                client=cast(OpenAI, cast(object, client)),
                schema="pdx",
                agreements_table="pdx.agreements",
                batch=SimpleNamespace(output_file_id="file-1"),
                update_sql=_counsel_offline_update_sql("pdx.agreements"),
                parse_response_text=parse_offline_counsel_response_text,
                build_update_params=build_offline_counsel_update_params,
                log_prefix="tx_metadata_asset (offline counsel)",
            )

        self.assertEqual(updated, 1)
        self.assertEqual(parse_errors, 0)
        self.assertEqual(refreshed_uuids, ["agreement-1"])
        self.assertEqual(processed_uuids, ["agreement-1"])
        refresh.assert_called_once_with(conn, "pdx", ["agreement-1"])

    def test_apply_offline_batch_output_does_not_count_noop_update_as_updated(self) -> None:
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
        conn = _FakeOfflineConn(rowcount=0)
        engine = _FakeOfflineEngine(conn)
        context = SimpleNamespace(log=_FakeLog())

        with patch("etl.defs.i_tx_metadata_asset.refresh_latest_sections_search") as refresh:
            updated, parse_errors, refreshed_uuids, processed_uuids = _apply_offline_batch_output(
                context=cast(AssetExecutionContext, cast(object, context)),
                engine=engine,
                client=cast(OpenAI, cast(object, client)),
                schema="pdx",
                agreements_table="pdx.agreements",
                batch=SimpleNamespace(output_file_id="file-1"),
            )

        self.assertEqual(updated, 0)
        self.assertEqual(parse_errors, 0)
        self.assertEqual(refreshed_uuids, [])
        self.assertEqual(processed_uuids, ["agreement-1"])
        refresh.assert_not_called()

    def test_run_offline_mode_starts_metadata_processing_while_counsel_is_preparing(self) -> None:
        context = SimpleNamespace(log=_FakeLog())
        engine = _NoopEngine()
        creation_order: list[str] = []
        metadata_started = threading.Event()

        def fake_create_offline_batch(*, batch_kind: str, **_kwargs: object) -> dict[str, object]:
            creation_order.append(batch_kind)
            return {
                "batch_id": f"batch-{batch_kind}",
                "completion_window": "24h",
                "request_total": 1,
            }

        def fake_resume_and_apply(*, batch_kind: str, **_kwargs: object) -> dict[str, object]:
            if batch_kind == "metadata":
                metadata_started.set()
            return {
                "batch_id": f"batch-{batch_kind}",
                "batch_kind": batch_kind,
                "status": "completed",
                "updated": 0,
                "parse_errors": 0,
                "refreshed_uuids": [],
                "processed_uuids": [f"agreement-{batch_kind}"],
            }

        def fake_build_counsel_lines(**_kwargs: object) -> list[dict[str, object]]:
            if not metadata_started.wait(0.5):
                raise AssertionError("expected metadata processing to start before counsel prep finished")
            return [{"custom_id": "agreement-c"}]

        with (
            patch("etl.defs.i_tx_metadata_asset._oai_client", return_value=SimpleNamespace()),
            patch("etl.defs.i_tx_metadata_asset.assert_tables_exist"),
            patch("etl.defs.i_tx_metadata_asset._load_existing_offline_batch", side_effect=[None, None]),
            patch("etl.defs.i_tx_metadata_asset._build_offline_metadata_lines", return_value=[{"custom_id": "agreement-m"}]),
            patch("etl.defs.i_tx_metadata_asset._build_offline_counsel_lines", side_effect=fake_build_counsel_lines),
            patch("etl.defs.i_tx_metadata_asset._create_offline_batch", side_effect=fake_create_offline_batch),
            patch("etl.defs.i_tx_metadata_asset._resume_and_apply_offline_batch", side_effect=fake_resume_and_apply),
            patch("etl.defs.i_tx_metadata_asset._sync_counsel_mappings") as sync_counsel,
        ):
            processed_uuids = _run_offline_mode(
                context=cast(AssetExecutionContext, cast(object, context)),
                engine=engine,
                schema="pdx",
                agreements_table="pdx.agreements",
                pages_table="pdx.pages",
                tagged_outputs_table="pdx.tagged_outputs",
                batch_size=10,
            )

        self.assertEqual(creation_order, ["metadata", "counsel"])
        self.assertEqual(sync_counsel.call_count, 1)
        self.assertEqual(processed_uuids, ["agreement-counsel", "agreement-metadata"])

    def test_run_offline_mode_rechecks_counsel_after_metadata_updates(self) -> None:
        context = SimpleNamespace(log=_FakeLog())
        engine = _NoopEngine()
        created_batch_kinds: list[str] = []

        def fake_create_offline_batch(*, batch_kind: str, **_kwargs: object) -> dict[str, object]:
            created_batch_kinds.append(batch_kind)
            return {
                "batch_id": f"batch-{batch_kind}-{len(created_batch_kinds)}",
                "completion_window": "24h",
                "request_total": 1,
            }

        def fake_resume_and_apply(*, batch_kind: str, **_kwargs: object) -> dict[str, object]:
            updated = 2 if batch_kind == "metadata" else 1
            return {
                "batch_id": f"batch-{batch_kind}",
                "batch_kind": batch_kind,
                "status": "completed",
                "updated": updated,
                "parse_errors": 0,
                "refreshed_uuids": [],
                "processed_uuids": [f"agreement-{batch_kind}"],
            }

        with (
            patch("etl.defs.i_tx_metadata_asset._oai_client", return_value=SimpleNamespace()),
            patch("etl.defs.i_tx_metadata_asset.assert_tables_exist"),
            patch("etl.defs.i_tx_metadata_asset._load_existing_offline_batch", side_effect=[None, None, None]),
            patch("etl.defs.i_tx_metadata_asset._build_offline_metadata_lines", return_value=[{"custom_id": "agreement-m"}]),
            patch(
                "etl.defs.i_tx_metadata_asset._build_offline_counsel_lines",
                side_effect=[[], [{"custom_id": "agreement-c"}]],
            ) as build_counsel_lines,
            patch("etl.defs.i_tx_metadata_asset._create_offline_batch", side_effect=fake_create_offline_batch),
            patch("etl.defs.i_tx_metadata_asset._resume_and_apply_offline_batch", side_effect=fake_resume_and_apply),
            patch("etl.defs.i_tx_metadata_asset._sync_counsel_mappings") as sync_counsel,
        ):
            processed_uuids = _run_offline_mode(
                context=cast(AssetExecutionContext, cast(object, context)),
                engine=engine,
                schema="pdx",
                agreements_table="pdx.agreements",
                pages_table="pdx.pages",
                tagged_outputs_table="pdx.tagged_outputs",
                batch_size=10,
            )

        self.assertEqual(created_batch_kinds, ["metadata", "counsel"])
        self.assertEqual(build_counsel_lines.call_count, 2)
        self.assertEqual(sync_counsel.call_count, 2)
        self.assertEqual(processed_uuids, ["agreement-counsel", "agreement-metadata"])

    def test_build_offline_counsel_lines_uses_matching_sections(self) -> None:
        conn = _FakeCounselSelectionConn(
            candidate_rows=[
                {
                    "agreement_uuid": "agreement-1",
                    "target": "Target A",
                    "acquirer": "Acquirer A",
                    "filing_date": date(2024, 1, 1),
                    "missing_all_counsel_fields": 1,
                }
            ],
            section_rows=[
                {
                    "agreement_uuid": "agreement-1",
                    "section_uuid": "section-1",
                    "xml_content": "<section>Target counsel is Wachtell, Lipton, Rosen &amp; Katz.</section>",
                },
                {
                    "agreement_uuid": "agreement-1",
                    "section_uuid": "section-2",
                    "xml_content": "<section>Acquirer counsel is Kirkland &amp; Ellis LLP.</section>",
                },
            ],
        )
        engine = _FakeCounselSelectionEngine(conn)
        context = SimpleNamespace(log=_FakeLog())

        lines = _build_offline_counsel_lines(
            context=cast(AssetExecutionContext, cast(object, context)),
            engine=engine,
            schema="pdx",
            agreements_table="pdx.agreements",
            batch_size=10,
        )

        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["custom_id"], "agreement-1")
        self.assertIn("Target name: Target A", str(lines[0]["body"]["input"][0]["content"]))
        self.assertTrue(
            any(
                "FROM pdx.latest_sections_search_standard_ids lssi" in sql
                and "lssi.standard_id = :counsel_standard_id" in sql
                for sql in conn.executed_sql
            )
        )

    def test_build_offline_counsel_lines_prioritizes_missing_all_fields_then_oldest_filing_date(self) -> None:
        conn = _FakeCounselSelectionConn(
            candidate_rows=[
                {
                    "agreement_uuid": "agreement-older-missing-all",
                    "target": "Target A",
                    "acquirer": "Acquirer A",
                    "filing_date": date(2023, 1, 1),
                    "missing_all_counsel_fields": 1,
                },
                {
                    "agreement_uuid": "agreement-newer-missing-all",
                    "target": "Target B",
                    "acquirer": "Acquirer B",
                    "filing_date": date(2024, 1, 1),
                    "missing_all_counsel_fields": 1,
                },
                {
                    "agreement_uuid": "agreement-oldest-partial",
                    "target": "Target C",
                    "acquirer": "Acquirer C",
                    "filing_date": date(2022, 1, 1),
                    "missing_all_counsel_fields": 0,
                },
            ],
            section_rows=[
                {
                    "agreement_uuid": "agreement-older-missing-all",
                    "section_uuid": "section-1",
                    "xml_content": "<section>Target counsel is A.</section>",
                },
                {
                    "agreement_uuid": "agreement-newer-missing-all",
                    "section_uuid": "section-1",
                    "xml_content": "<section>Target counsel is B.</section>",
                },
                {
                    "agreement_uuid": "agreement-oldest-partial",
                    "section_uuid": "section-1",
                    "xml_content": "<section>Target counsel is C.</section>",
                },
            ],
        )
        engine = _FakeCounselSelectionEngine(conn)
        context = SimpleNamespace(log=_FakeLog())

        lines = _build_offline_counsel_lines(
            context=cast(AssetExecutionContext, cast(object, context)),
            engine=engine,
            schema="pdx",
            agreements_table="pdx.agreements",
            batch_size=3,
        )

        self.assertEqual(
            [str(line["custom_id"]) for line in lines],
            [
                "agreement-older-missing-all",
                "agreement-newer-missing-all",
                "agreement-oldest-partial",
            ],
        )
        self.assertTrue(
            any("missing_all_counsel_fields DESC" in sql and "filing_date ASC" in sql for sql in conn.executed_sql)
        )

    def test_build_offline_metadata_lines_prioritizes_missing_all_fields_then_oldest_filing_date(self) -> None:
        conn = _FakeMetadataSelectionConn(
            [
                {
                    "agreement_uuid": "agreement-older-missing-all",
                    "page_order": 1,
                    "page_text": "Older missing all front matter",
                },
                {
                    "agreement_uuid": "agreement-newer-missing-all",
                    "page_order": 1,
                    "page_text": "Newer missing all front matter",
                },
                {
                    "agreement_uuid": "agreement-oldest-partial",
                    "page_order": 1,
                    "page_text": "Oldest partial front matter",
                },
            ]
        )
        engine = _FakeMetadataSelectionEngine(conn)
        context = SimpleNamespace(log=_FakeLog())

        lines = _build_offline_metadata_lines(
            context=cast(AssetExecutionContext, cast(object, context)),
            engine=engine,
            agreements_table="pdx.agreements",
            pages_table="pdx.pages",
            tagged_outputs_table="pdx.tagged_outputs",
            batch_size=3,
        )

        self.assertEqual(
            [str(line["custom_id"]) for line in lines],
            [
                "agreement-older-missing-all",
                "agreement-newer-missing-all",
                "agreement-oldest-partial",
            ],
        )
        self.assertTrue(
            any("missing_all_metadata_fields DESC" in sql and "filing_date ASC" in sql for sql in conn.executed_sql)
        )

    def test_build_offline_counsel_update_params_only_emits_raw_counsel_fields(self) -> None:
        self.assertEqual(
            build_offline_counsel_update_params(
                agreement_uuid="agreement-1",
                parsed={
                    "target_counsel": "Wachtell, Lipton, Rosen & Katz",
                    "acquirer_counsel": "Kirkland & Ellis LLP",
                },
            ),
            {
                "uuid": "agreement-1",
                "target_counsel": "Wachtell, Lipton, Rosen & Katz",
                "acquirer_counsel": "Kirkland & Ellis LLP",
            },
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
            summary = _run_web_search_mode(
                context=cast(AssetExecutionContext, cast(object, context)),
                engine=engine,
                schema="pdx",
                agreements_table="pdx.agreements",
                batch_size=10,
            )

        refresh.assert_called_once_with(conn, "pdx", ["agreement-1"])
        self.assertEqual(conn.last_select_params, {"lim": 10})
        assert conn.last_update_params is not None
        metadata_payload = json.loads(cast(str, conn.last_update_params["metadata_sources"]))
        self.assertEqual(
            metadata_payload["metadata_run_stats"]["token_usage"],
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        self.assertEqual(metadata_payload["metadata_run_stats"]["search_count"], 1)
        self.assertEqual(conn.update_calls, 1)
        self.assertEqual(summary["processed_uuids"], ["agreement-1"])
        select_sql = next(sql for sql in conn.executed_sql if "FROM pdx.agreements a" in sql)
        self.assertIn("COALESCE(a.metadata, 0) = 0", select_sql)
        self.assertIn("COALESCE(a.metadata, 0) = 1", select_sql)
        self.assertIn("initial_metadata_pass DESC", select_sql)

    def test_run_web_search_mode_uses_gpt_5_1_for_requeues_without_mixing_chunk_models(self) -> None:
        conn = _FakeWebConn(
            select_rows=[
                self._agreement_row("agreement-initial", initial_metadata_pass=1),
                self._agreement_row("agreement-requeue", initial_metadata_pass=0),
            ]
        )
        engine = _FakeWebEngine(conn)
        fake_log = _FakeLog()
        context = SimpleNamespace(log=fake_log)
        client = _RoutingWebClient(
            responses_by_input_fragment={
                "agreement-initial.htm": self._valid_web_search_payload(),
                "agreement-requeue.htm": self._valid_web_search_payload(),
            }
        )

        with (
            patch("etl.defs.i_tx_metadata_asset._oai_client", return_value=client),
            patch("etl.defs.i_tx_metadata_asset.refresh_latest_sections_search"),
        ):
            _ = _run_web_search_mode(
                context=cast(AssetExecutionContext, cast(object, context)),
                engine=engine,
                schema="pdx",
                agreements_table="pdx.agreements",
                batch_size=2,
            )

        self.assertEqual(client.responses.models, ["gpt-5-mini", "gpt-5.1"])
        self.assertTrue(
            any(call[0] == "tx_metadata_asset (web_search): starting chunk %s/%s with %s agreements using model %s" and call[4] == "gpt-5-mini" for call in fake_log.info_calls)
        )
        self.assertTrue(
            any(call[0] == "tx_metadata_asset (web_search): starting chunk %s/%s with %s agreements using model %s" and call[4] == "gpt-5.1" for call in fake_log.info_calls)
        )

    def test_run_web_search_mode_targets_retry_prompt_to_unresolved_fields(self) -> None:
        conn = _FakeWebConn(
            select_rows=[
                self._agreement_row("agreement-requeue", initial_metadata_pass=0),
            ]
        )
        engine = _FakeWebEngine(conn)
        context = SimpleNamespace(log=_FakeLog())
        client = _FakeWebClient(response_text=self._valid_web_search_payload())

        with (
            patch("etl.defs.i_tx_metadata_asset._oai_client", return_value=client),
            patch("etl.defs.i_tx_metadata_asset.refresh_latest_sections_search"),
        ):
            _ = _run_web_search_mode(
                context=cast(AssetExecutionContext, cast(object, context)),
                engine=engine,
                schema="pdx",
                agreements_table="pdx.agreements",
                batch_size=1,
            )

        self.assertEqual(client.responses.models, ["gpt-5.1"])
        self.assertEqual(len(client.responses.inputs), 1)
        retry_input = cast(str, client.responses.inputs[0])
        self.assertIn("focus_fields: close_date, deal_status", retry_input)
        self.assertIn("known_trusted_metadata:", retry_input)
        self.assertIn("consideration_type: all_cash", retry_input)

    def test_run_web_search_mode_preserves_locked_fields_when_retry_returns_nulls(self) -> None:
        conn = _FakeWebConn(
            select_rows=[
                self._agreement_row("agreement-requeue", initial_metadata_pass=0),
            ]
        )
        engine = _FakeWebEngine(conn)
        context = SimpleNamespace(log=_FakeLog())
        retry_payload = json.dumps(
            {
                "consideration_type": "all_cash",
                "purchase_price": {"cash": None, "stock": None, "assets": None},
                "target_public": None,
                "acquirer_public": None,
                "target_pe": None,
                "acquirer_pe": None,
                "target_industry": None,
                "acquirer_industry": None,
                "announce_date": "2024-01-01",
                "close_date": "2024-02-01",
                "deal_status": "complete",
                "attitude": None,
                "purpose": None,
                "metadata_sources": {
                    "citations": [
                        self._citation("close_date"),
                        self._citation("deal_status"),
                    ],
                    "notes": "Retry focused on completion date only.",
                },
            }
        )

        with (
            patch("etl.defs.i_tx_metadata_asset._oai_client", return_value=_FakeWebClient(response_text=retry_payload)),
            patch("etl.defs.i_tx_metadata_asset.refresh_latest_sections_search"),
        ):
            _ = _run_web_search_mode(
                context=cast(AssetExecutionContext, cast(object, context)),
                engine=engine,
                schema="pdx",
                agreements_table="pdx.agreements",
                batch_size=1,
            )

        assert conn.last_update_params is not None
        self.assertEqual(conn.last_update_params["price_cash"], 100.0)
        self.assertEqual(conn.last_update_params["target_type"], "public")
        self.assertEqual(conn.last_update_params["acquirer_type"], "private")
        self.assertEqual(conn.last_update_params["close_date"], "2024-02-01")

    def test_run_web_search_mode_records_retry_context_build_failure_without_crashing_run(self) -> None:
        conn = _FakeWebConn(
            select_rows=[
                self._agreement_row(
                    "agreement-bad-requeue",
                    initial_metadata_pass=0,
                    metadata_uncited_fields={"bad": "shape"},
                ),
            ]
        )
        engine = _FakeWebEngine(conn)
        fake_log = _FakeLog()
        context = SimpleNamespace(log=fake_log)

        with patch(
            "etl.defs.i_tx_metadata_asset._oai_client",
            return_value=_FakeWebClient(response_text=self._valid_web_search_payload()),
        ):
            summary = _run_web_search_mode(
                context=cast(AssetExecutionContext, cast(object, context)),
                engine=engine,
                schema="pdx",
                agreements_table="pdx.agreements",
                batch_size=1,
            )

        self.assertEqual(conn.update_calls, 0)
        self.assertEqual(conn.failure_upserts, 1)
        self.assertEqual(summary["total_searches"], 0)
        self.assertTrue(
            any("failed for agreement-bad-requeue" in str(call[0]) for call in fake_log.warning_calls)
        )

    def test_run_web_search_mode_raises_on_database_update_error(self) -> None:
        conn = _FakeWebConn(fail_on_update=True)
        engine = _FakeWebEngine(conn)
        context = SimpleNamespace(log=_FakeLog())

        with patch(
            "etl.defs.i_tx_metadata_asset._oai_client",
            return_value=_FakeWebClient(response_text=self._valid_web_search_payload()),
        ):
            with self.assertRaisesRegex(RuntimeError, "database update failed"):
                _ = _run_web_search_mode(
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
            _ = _run_web_search_mode(
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
                    "failure_count": 0,
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
            _ = _run_web_search_mode(
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
                    "failure_count": 0,
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
            _ = _run_web_search_mode(
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
            _ = _run_web_search_mode(
                context=cast(AssetExecutionContext, cast(object, context)),
                engine=engine,
                schema="pdx",
                agreements_table="pdx.agreements",
                batch_size=10,
            )

        self.assertEqual(conn.failure_upserts, 1)
        self.assertEqual(conn.failure_clears, 0)

    def test_run_web_search_mode_checkpoints_each_chunk_of_ten(self) -> None:
        select_rows = [
            self._agreement_row(f"agreement-{index:02d}")
            for index in range(12)
        ]
        conn = _FakeWebConn(select_rows=select_rows)
        engine = _FakeWebEngine(conn)
        fake_log = _FakeLog()
        context = SimpleNamespace(log=fake_log)

        with (
            patch(
                "etl.defs.i_tx_metadata_asset._oai_client",
                return_value=_FakeWebClient(response_text=self._valid_web_search_payload()),
            ),
            patch("etl.defs.i_tx_metadata_asset.refresh_latest_sections_search"),
        ):
            _ = _run_web_search_mode(
                context=cast(AssetExecutionContext, cast(object, context)),
                engine=engine,
                schema="pdx",
                agreements_table="pdx.agreements",
                batch_size=12,
            )

        self.assertEqual(conn.update_calls, 12)
        first_checkpoint = {cast(str, params["uuid"]) for params in conn.update_params_history[:10]}
        second_checkpoint = {cast(str, params["uuid"]) for params in conn.update_params_history[10:]}
        self.assertEqual(len(first_checkpoint), 10)
        self.assertEqual(len(second_checkpoint), 2)
        self.assertEqual(
            first_checkpoint | second_checkpoint,
            {f"agreement-{index:02d}" for index in range(12)},
        )
        self.assertTrue(
            any(
                call[:4] == (
                    "tx_metadata_asset (web_search): finished chunk %s/%s attempted=%s request_failures=%s validation_failures=%s updated=%s refreshed_latest_sections_search=%s model=%s",
                    1,
                    2,
                    10,
                )
                for call in fake_log.info_calls
            )
        )
        self.assertTrue(
            any(
                call[:4] == (
                    "tx_metadata_asset (web_search): finished chunk %s/%s attempted=%s request_failures=%s validation_failures=%s updated=%s refreshed_latest_sections_search=%s model=%s",
                    2,
                    2,
                    2,
                )
                for call in fake_log.info_calls
            )
        )

    def test_run_web_search_mode_commits_first_chunk_before_later_chunk_failure(self) -> None:
        select_rows = [
            self._agreement_row(f"agreement-{index:02d}")
            for index in range(12)
        ]
        conn = _FakeWebConn(select_rows=select_rows)
        engine = _FakeWebEngine(conn)
        context = SimpleNamespace(log=_FakeLog())
        later_chunk_failures: dict[str, object] = {
            "agreement-10.htm": RuntimeError("later chunk failed"),
            "agreement-11.htm": RuntimeError("later chunk failed"),
        }
        responses_by_input_fragment: dict[str, object] = {
            f"agreement-{index:02d}.htm": self._valid_web_search_payload()
            for index in range(10)
        }
        responses_by_input_fragment.update(later_chunk_failures)
        flaky_client = _RoutingWebClient(
            responses_by_input_fragment=responses_by_input_fragment
        )

        with (
            patch("etl.defs.i_tx_metadata_asset._oai_client", return_value=flaky_client),
            patch("etl.defs.i_tx_metadata_asset.time.sleep", return_value=None),
            patch("etl.defs.i_tx_metadata_asset.refresh_latest_sections_search"),
        ):
            _ = _run_web_search_mode(
                context=cast(AssetExecutionContext, cast(object, context)),
                engine=engine,
                schema="pdx",
                agreements_table="pdx.agreements",
                batch_size=12,
            )

        self.assertEqual(conn.update_calls, 10)
        self.assertEqual(conn.failure_upserts, 2)

    def test_run_web_search_mode_mixed_outcomes_track_metrics(self) -> None:
        select_rows = [
            self._agreement_row("agreement-success"),
            self._agreement_row("agreement-request-failure"),
            self._agreement_row("agreement-validation-failure"),
        ]
        conn = _FakeWebConn(select_rows=select_rows)
        engine = _FakeWebEngine(conn)
        context = SimpleNamespace(log=_FakeLog())
        invalid_payload = json.dumps(
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
                "metadata_sources": {"citations": [], "notes": None},
            }
        )
        flaky_client = _RoutingWebClient(
            responses_by_input_fragment={
                "agreement-success.htm": self._valid_web_search_payload(),
                "agreement-request-failure.htm": RuntimeError("request failed"),
                "agreement-validation-failure.htm": invalid_payload,
            },
            search_count=2,
        )

        with (
            patch("etl.defs.i_tx_metadata_asset._oai_client", return_value=flaky_client),
            patch("etl.defs.i_tx_metadata_asset.time.sleep", return_value=None),
            patch("etl.defs.i_tx_metadata_asset.refresh_latest_sections_search"),
        ):
            summary = _run_web_search_mode(
                context=cast(AssetExecutionContext, cast(object, context)),
                engine=engine,
                schema="pdx",
                agreements_table="pdx.agreements",
                batch_size=3,
            )

        self.assertEqual(conn.update_calls, 1)
        self.assertEqual(conn.failure_upserts, 2)
        self.assertEqual(summary["total_searches"], 4)
        self.assertEqual(summary["searches_by_agreement"]["agreement-success"], 2)
        self.assertEqual(summary["searches_by_agreement"]["agreement-validation-failure"], 2)

    def test_run_web_search_mode_prioritizes_lower_failure_count(self) -> None:
        conn = _FakeWebConn(
            select_rows=[
                {
                    "agreement_uuid": "agreement-clean",
                    "target": "Target A",
                    "acquirer": "Acquirer A",
                    "filing_date": date.today().isoformat(),
                    "url": "https://www.sec.gov/Archives/edgar/data/123/clean.htm",
                    "failure_count": 0,
                },
                {
                    "agreement_uuid": "agreement-retry",
                    "target": "Target B",
                    "acquirer": "Acquirer B",
                    "filing_date": date.today().isoformat(),
                    "url": "https://www.sec.gov/Archives/edgar/data/123/retry.htm",
                    "failure_count": 4,
                },
            ]
        )
        engine = _FakeWebEngine(conn)
        context = SimpleNamespace(log=_FakeLog())
        with (
            patch(
                "etl.defs.i_tx_metadata_asset._oai_client",
                return_value=_FakeWebClient(response_text=self._valid_web_search_payload()),
            ),
            patch("etl.defs.i_tx_metadata_asset.refresh_latest_sections_search"),
        ):
            _ = _run_web_search_mode(
                context=cast(AssetExecutionContext, cast(object, context)),
                engine=engine,
                schema="pdx",
                agreements_table="pdx.agreements",
                batch_size=2,
            )

        self.assertIsNotNone(conn.last_select_rows)
        assert conn.last_select_rows is not None
        self.assertEqual(conn.last_select_rows[0]["agreement_uuid"], "agreement-clean")
        self.assertEqual(conn.last_select_rows[1]["agreement_uuid"], "agreement-retry")

    def test_run_web_search_mode_logs_failed_uuid_summary(self) -> None:
        conn = _FakeWebConn()
        engine = _FakeWebEngine(conn)
        fake_log = _FakeLog()
        context = SimpleNamespace(log=fake_log)
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
            _ = _run_web_search_mode(
                context=cast(AssetExecutionContext, cast(object, context)),
                engine=engine,
                schema="pdx",
                agreements_table="pdx.agreements",
                batch_size=10,
            )

        self.assertTrue(
            any("failed agreements by stage" in str(call[0]) for call in fake_log.warning_calls)
        )


if __name__ == "__main__":
    _ = unittest.main()
