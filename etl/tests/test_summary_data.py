# pyright: reportAny=false, reportPrivateUsage=false
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

from etl.defs.resources import DBResource
from etl.utils.summary_data import refresh_summary_data


class SummaryDataTests(unittest.TestCase):
    def test_refresh_summary_data_does_not_check_other_runs(self) -> None:
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

        context = Mock()
        context.instance = Mock()
        context.instance.get_runs = Mock(side_effect=AssertionError("should not inspect other runs"))
        context.log = Mock()

        refresh_summary_data(context, cast(DBResource, cast(object, db)))
        context.instance.get_runs.assert_not_called()

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
        self.assertTrue(
            any("TRUNCATE TABLE pdx.agreement_overview_summary" in sql for sql in conn.executed_sql)
        )
        summary_insert_sql = next(
            sql
            for sql in conn.executed_sql
            if "INSERT INTO pdx.summary_data" in sql
        )
        self.assertIn("FROM pdx.agreements AS a", summary_insert_sql)
        self.assertIn("COALESCE(LOWER(a.status), '') <> 'invalid'", summary_insert_sql)
        self.assertIn(
            "NOT (COALESCE(a.gated, 0) = 1 AND COALESCE(a.verified, 0) = 0)",
            summary_insert_sql,
        )
        status_insert_sql = next(
            sql
            for sql in conn.executed_sql
            if "INSERT INTO pdx.agreement_status_summary" in sql
        )
        self.assertIn("COALESCE(LOWER(a.status), '') <> 'invalid'", status_insert_sql)
        self.assertIn(
            "NOT (COALESCE(a.gated, 0) = 1 AND COALESCE(a.verified, 0) = 0)",
            status_insert_sql,
        )
        deal_type_insert_sql = next(
            sql
            for sql in conn.executed_sql
            if "INSERT INTO pdx.agreement_deal_type_summary" in sql
        )
        self.assertIn("FROM tmp_xml_eligible x", deal_type_insert_sql)
        self.assertIn("COALESCE(a.deal_type, 'unknown')", deal_type_insert_sql)
        self.assertIn("COUNT(*) AS `count`", deal_type_insert_sql)
        self.assertIn("COALESCE(LOWER(a.status), '') <> 'invalid'", deal_type_insert_sql)
        self.assertIn(
            "NOT (COALESCE(a.gated, 0) = 1 AND COALESCE(a.verified, 0) = 0)",
            deal_type_insert_sql,
        )
        overview_insert_sql = next(
            sql
            for sql in conn.executed_sql
            if "INSERT INTO pdx.agreement_overview_summary" in sql
        )
        self.assertIn("COALESCE(a.metadata, 0) = 1", overview_insert_sql)
        self.assertIn("section_standard_id_gold_label", overview_insert_sql)
        self.assertIn("JOIN tmp_xml_latest x", overview_insert_sql)
        self.assertIn("TRIM(a3.filing_date) REGEXP '^[0-9]{4}-[0-9]{2}-[0-9]{2}'", overview_insert_sql)
        self.assertIn("SUBSTRING(TRIM(a3.filing_date), 1, 10)", overview_insert_sql)
        self.assertNotIn("a3.filing_date != ''", overview_insert_sql)
        self.assertIn(
            "NOT (COALESCE(a.gated, 0) = 1 AND COALESCE(a.verified, 0) = 0)",
            overview_insert_sql,
        )

    def test_refresh_summary_data_uses_canonical_stage_sql(self) -> None:
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

        stage_sql = next(
            sql for sql in conn.executed_sql if "CREATE TEMPORARY TABLE tmp_stage_state AS" in sql
        )
        self.assertIn("WITH page_counts AS", stage_sql)
        self.assertIn("WHEN latest_xml_status IS NULL THEN 'red'", stage_sql)
        self.assertIn("WHEN latest_xml_status = 'invalid' THEN 'red'", stage_sql)
        self.assertIn("WHEN body_page_count = 0 THEN 'red'", stage_sql)


if __name__ == "__main__":
    _ = unittest.main()
