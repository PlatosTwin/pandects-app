# pyright: reportAny=false, reportPrivateUsage=false
import datetime
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from dagster import AssetExecutionContext
from requests.exceptions import HTTPError
from requests.models import Response

from etl.defs.a_staging_asset import (
    _PersistedAgreement,
    _build_duplicate_resolutions,
    staging_asset,
)
from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.a_staging import (
    ExhibitSignature,
    SecDailyIndexUnavailable,
    _compute_minhash,
    parse_index_file,
)


class _FakeLog:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, msg: str) -> None:
        self.messages.append(msg)

    def error(self, msg: str) -> None:
        self.messages.append(msg)


class _ParseContext:
    def __init__(self) -> None:
        self.log = _FakeLog()


class _SelectResult:
    def __init__(self, value: datetime.datetime | None) -> None:
        self._value = value

    def scalar_one_or_none(self) -> datetime.datetime | None:
        return self._value


class _InsertResult:
    def __init__(self, lastrowid: int) -> None:
        self.lastrowid = lastrowid


class _ExecuteResult:
    def __init__(self, rowcount: int = 1) -> None:
        self.rowcount = rowcount


class _FakeConn:
    def __init__(self, last_run: datetime.datetime) -> None:
        self._last_run = last_run
        self.progress_updates: list[dict[str, object]] = []
        self.success_update: dict[str, object] | None = None
        self.started_payload: dict[str, object] | None = None
        self.failed = False

    def execute(self, statement: object, params: dict[str, object] | None = None) -> object:
        sql = str(statement)
        if "SELECT last_pulled_to" in sql:
            return _SelectResult(self._last_run)
        if "INSERT INTO pdx.pipeline_runs" in sql and params is not None:
            self.started_payload = params
            return _InsertResult(lastrowid=99)
        if "status = 'SUCCEEDED'" in sql and params is not None:
            self.success_update = params
            return _ExecuteResult()
        if "status = 'FAILED'" in sql:
            self.failed = True
            return _ExecuteResult()
        if "SET last_pulled_to = :pulled_to" in sql and params is not None:
            self.progress_updates.append(params)
            return _ExecuteResult()
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


class StagingAssetTests(unittest.TestCase):
    def test_parse_index_file_raises_when_daily_index_returns_404(self) -> None:
        response = cast(Response, cast(object, SimpleNamespace(status_code=404)))
        error = HTTPError(response=response)

        def _missing_index(*_args: object, **_kwargs: object) -> Response:
            raise error

        context = _ParseContext()

        with self.assertRaises(SecDailyIndexUnavailable):
            _ = parse_index_file(
                index_url="https://www.sec.gov/Archives/edgar/daily-index/2026/QTR1/form.20260314.idx",
                user_agent="test-agent",
                context=context,
                rate_limited_get=_missing_index,
            )

    def test_parse_index_file_raises_when_daily_index_returns_403(self) -> None:
        response = cast(Response, cast(object, SimpleNamespace(status_code=403)))
        error = HTTPError(response=response)

        def _forbidden_index(*_args: object, **_kwargs: object) -> Response:
            raise error

        context = _ParseContext()

        with self.assertRaises(SecDailyIndexUnavailable):
            _ = parse_index_file(
                index_url="https://www.sec.gov/Archives/edgar/daily-index/2026/QTR1/form.20260314.idx",
                user_agent="test-agent",
                context=context,
                rate_limited_get=_forbidden_index,
            )

    def test_parse_index_file_reraises_non_403_or_404_http_error(self) -> None:
        response = cast(Response, cast(object, SimpleNamespace(status_code=500)))
        error = HTTPError(response=response)

        def _server_error(*_args: object, **_kwargs: object) -> Response:
            raise error

        context = _ParseContext()

        with self.assertRaises(HTTPError):
            _ = parse_index_file(
                index_url="https://www.sec.gov/Archives/edgar/daily-index/2026/QTR1/form.20260314.idx",
                user_agent="test-agent",
                context=context,
                rate_limited_get=_server_error,
            )

    def test_staging_asset_treats_missing_historical_index_as_empty_day(self) -> None:
        conn = _FakeConn(last_run=datetime.datetime(2026, 3, 10, 17, 30))
        engine = _FakeEngine(conn)
        context = SimpleNamespace(log=_FakeLog())
        db = SimpleNamespace(database="pdx", get_engine=lambda: engine)
        pipeline_config = PipelineConfig(staging_target_date="2026-03-14")
        unavailable = SecDailyIndexUnavailable(
            "https://www.sec.gov/Archives/edgar/daily-index/2026/QTR1/form.20260312.idx"
        )

        with (
            patch("etl.defs.a_staging_asset.ExhibitClassifier.load", return_value=object()),
            patch(
                "etl.defs.a_staging_asset._latest_sec_index_date_available",
                return_value=datetime.date(2026, 3, 13),
            ),
            patch(
                "etl.defs.a_staging_asset._today_eastern",
                return_value=datetime.date(2026, 3, 15),
            ),
            patch(
                "etl.defs.a_staging_asset.fetch_new_filings_sec_index",
                side_effect=[[], unavailable, [], []],
            ) as fetch_new_filings,
            patch(
                "etl.defs.a_staging_asset._reconcile_cross_day_duplicates",
                return_value=(0, set()),
            ),
            patch("etl.defs.a_staging_asset.run_post_asset_refresh", return_value=None),
        ):
            decorated_fn = getattr(cast(object, staging_asset.op.compute_fn), "decorated_fn")
            result = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                db=cast(DBResource, cast(object, db)),
                pipeline_config=pipeline_config,
            )

        self.assertEqual(result, 0)
        self.assertEqual(fetch_new_filings.call_count, 4)
        self.assertEqual(len(conn.progress_updates), 4)
        self.assertFalse(conn.failed)
        self.assertIsNotNone(conn.success_update)
        success_update = cast(dict[str, object], conn.success_update)
        self.assertEqual(
            success_update["pulled_to"],
            datetime.datetime(2026, 3, 14),
        )
        self.assertEqual(success_update["count"], 0)

    def test_staging_asset_stops_before_unavailable_day_past_buffer(self) -> None:
        conn = _FakeConn(last_run=datetime.datetime(2026, 3, 10, 17, 30))
        engine = _FakeEngine(conn)
        context = SimpleNamespace(log=_FakeLog())
        db = SimpleNamespace(database="pdx", get_engine=lambda: engine)
        pipeline_config = PipelineConfig(staging_target_date="2026-03-17")
        unavailable = SecDailyIndexUnavailable(
            "https://www.sec.gov/Archives/edgar/daily-index/2026/QTR1/form.20260314.idx"
        )

        with (
            patch("etl.defs.a_staging_asset.ExhibitClassifier.load", return_value=object()),
            patch(
                "etl.defs.a_staging_asset._latest_sec_index_date_available",
                return_value=datetime.date(2026, 3, 13),
            ),
            patch(
                "etl.defs.a_staging_asset._today_eastern",
                return_value=datetime.date(2026, 3, 18),
            ),
            patch(
                "etl.defs.a_staging_asset.fetch_new_filings_sec_index",
                side_effect=[[], [], [], unavailable],
            ) as fetch_new_filings,
            patch(
                "etl.defs.a_staging_asset._reconcile_cross_day_duplicates",
                return_value=(0, set()),
            ),
            patch("etl.defs.a_staging_asset.run_post_asset_refresh", return_value=None),
        ):
            decorated_fn = getattr(cast(object, staging_asset.op.compute_fn), "decorated_fn")
            result = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                db=cast(DBResource, cast(object, db)),
                pipeline_config=pipeline_config,
            )

        self.assertEqual(result, 0)
        self.assertEqual(fetch_new_filings.call_count, 4)
        self.assertEqual(len(conn.progress_updates), 3)
        self.assertFalse(conn.failed)
        self.assertIsNotNone(conn.success_update)
        success_update = cast(dict[str, object], conn.success_update)
        self.assertEqual(
            success_update["pulled_to"],
            datetime.datetime(2026, 3, 13),
        )
        self.assertEqual(success_update["count"], 0)

    def test_build_duplicate_resolutions_collapses_cross_day_duplicate(self) -> None:
        early = _PersistedAgreement(
            agreement_uuid="early",
            url="https://example.com/early.htm",
            filing_date=datetime.date(2021, 7, 12),
            ingested_date=datetime.datetime(2026, 2, 28, 23, 2, 34),
            secondary_filing_url=None,
        )
        late = _PersistedAgreement(
            agreement_uuid="late",
            url="https://example.com/late.htm",
            filing_date=datetime.date(2021, 7, 13),
            ingested_date=datetime.datetime(2026, 2, 28, 23, 4, 49),
            secondary_filing_url=None,
        )
        signatures = {
            early.url: ExhibitSignature(
                page_count=120,
                auto_status_verified=True,
                content_fingerprint="same-doc",
                minhash=_compute_minhash("agreement and plan of merger " * 40),
            ),
            late.url: ExhibitSignature(
                page_count=120,
                auto_status_verified=True,
                content_fingerprint="same-doc",
                minhash=_compute_minhash("agreement and plan of merger " * 40),
            ),
        }

        resolutions = _build_duplicate_resolutions(
            ingested_agreements=[late],
            candidate_agreements=[early, late],
            signatures_by_url=signatures,
        )

        self.assertEqual(len(resolutions), 1)
        self.assertEqual(resolutions[0].survivor.agreement_uuid, "early")
        self.assertEqual(resolutions[0].loser.agreement_uuid, "late")


if __name__ == "__main__":
    _ = unittest.main()
