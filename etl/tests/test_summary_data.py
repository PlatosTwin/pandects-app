# pyright: reportAny=false, reportPrivateUsage=false
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

from etl.defs.resources import DBResource
from etl.utils.summary_data import IN_FLIGHT_RUN_STATUSES, _get_other_in_flight_run_ids, refresh_summary_data


class SummaryDataTests(unittest.TestCase):
    def test_get_other_in_flight_run_ids_excludes_current_run(self) -> None:
        context = Mock()
        context.run_id = "run-current"
        context.instance = Mock()
        context.instance.get_runs.return_value = [
            SimpleNamespace(run_id="run-current"),
            SimpleNamespace(run_id="run-other"),
        ]

        other_run_ids = _get_other_in_flight_run_ids(context)

        self.assertEqual(other_run_ids, ["run-other"])
        called_filter = context.instance.get_runs.call_args.kwargs["filters"]
        self.assertEqual(tuple(called_filter.statuses), IN_FLIGHT_RUN_STATUSES)

    def test_refresh_summary_data_skips_when_other_runs_in_flight(self) -> None:
        context = Mock()
        context.run_id = "run-current"
        context.instance = Mock()
        context.instance.get_runs.return_value = [
            SimpleNamespace(run_id="run-current"),
            SimpleNamespace(run_id="run-other"),
        ]
        context.log = Mock()

        db = Mock()
        db.get_engine = Mock(side_effect=AssertionError("refresh should have been skipped"))

        refresh_summary_data(context, db)

        db.get_engine.assert_not_called()
        context.log.info.assert_called_once()
        self.assertIn(
            "Skipping summary_data refresh",
            context.log.info.call_args.args[0],
        )

    def test_refresh_summary_data_updates_deal_type_summary(self) -> None:
        class _ScalarResult:
            def __init__(self, value: object):
                self._value = value

            def scalar(self) -> object:
                return self._value

        class _FakeConn:
            def __init__(self) -> None:
                self.executed_sql: list[str] = []

            def execute(self, statement: object, *_args: object, **_kwargs: object) -> _ScalarResult:
                sql = str(statement)
                self.executed_sql.append(sql)
                if "GET_LOCK" in sql:
                    return _ScalarResult(1)
                return _ScalarResult(None)

        class _BeginContext:
            def __init__(self, conn: _FakeConn):
                self._conn = conn

            def __enter__(self) -> _FakeConn:
                return self._conn

            def __exit__(self, *_exc: object) -> None:
                return None

        class _FakeEngine:
            def __init__(self, conn: _FakeConn):
                self._conn = conn

            def begin(self) -> _BeginContext:
                return _BeginContext(self._conn)

        conn = _FakeConn()
        engine = _FakeEngine(conn)
        db = SimpleNamespace(database="pdx", get_engine=lambda: engine)

        refresh_summary_data(None, cast(DBResource, cast(object, db)))

        self.assertTrue(
            any("TRUNCATE TABLE pdx.agreement_deal_type_summary" in sql for sql in conn.executed_sql)
        )
        summary_insert_sql = next(
            sql
            for sql in conn.executed_sql
            if "INSERT INTO pdx.summary_data" in sql
        )
        self.assertIn("FROM pdx.agreements AS a", summary_insert_sql)
        deal_type_insert_sql = next(
            sql
            for sql in conn.executed_sql
            if "INSERT INTO pdx.agreement_deal_type_summary" in sql
        )
        self.assertIn("WITH eligible_xml AS", deal_type_insert_sql)
        self.assertIn("COALESCE(a.deal_type, 'unknown')", deal_type_insert_sql)
        self.assertIn("COUNT(*) AS `count`", deal_type_insert_sql)


if __name__ == "__main__":
    _ = unittest.main()
