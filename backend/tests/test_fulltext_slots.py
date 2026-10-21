import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("SKIP_MAIN_DB_REFLECTION", "1")

from backend.core import fulltext_slots  # noqa: E402
from backend.core.fulltext_slots import FulltextSearchBusy, fulltext_slot  # noqa: E402


class _FakeConn:
    """Answers the named-lock statements from a dict of held locks."""

    def __init__(self, held: set[str], connection_id: int = 7) -> None:
        self.held = held
        self.connection_id = connection_id
        self.mine: set[str] = set()
        self.statements: list[str] = []

    def __enter__(self) -> "_FakeConn":
        return self

    def __exit__(self, *_exc: object) -> None:
        # Closing the connection drops its locks, as MariaDB does.
        self.held -= self.mine

    def _take(self, name: str) -> bool:
        if name in self.held:
            return False
        self.held.add(name)
        self.mine.add(name)
        return True

    def execute(self, statement: object, params: dict[str, object] | None = None):
        sql = str(statement)
        params = params or {}
        self.statements.append(sql)
        if sql.startswith("SELECT CASE"):
            value = next(
                (name for _key, name in sorted(params.items()) if self._take(str(name))),
                None,
            )
        elif "CONNECTION_ID" in sql:
            value = self.connection_id
        elif "RELEASE_LOCK" in sql:
            name = str(params["n"])
            self.held.discard(name)
            self.mine.discard(name)
            value = 1
        elif "GET_LOCK" in sql:
            value = 1 if self._take(str(params["n"])) else 0
        else:
            raise AssertionError(sql)
        return SimpleNamespace(scalar=lambda: value)


class _FakeEngine:
    def __init__(self, held: set[str], dialect: str = "mariadb") -> None:
        self.dialect = SimpleNamespace(name=dialect)
        self.held = held
        self.conns: list[_FakeConn] = []

    def connect(self) -> _FakeConn:
        conn = _FakeConn(self.held)
        self.conns.append(conn)
        return conn


class FulltextSlotTests(unittest.TestCase):
    def test_sqlite_is_not_limited(self):
        engine = _FakeEngine(set(), dialect="sqlite")
        with fulltext_slot(engine):
            pass
        self.assertEqual(engine.conns, [])

    def test_takes_a_free_slot_and_releases_it(self):
        held: set[str] = set()
        engine = _FakeEngine(held)
        with fulltext_slot(engine):
            self.assertEqual(held, {"pandects:default:fulltext:0"})
        self.assertEqual(held, set())

    def test_second_search_takes_the_other_slot(self):
        held = {"pandects:default:fulltext:0"}
        engine = _FakeEngine(held)
        with fulltext_slot(engine):
            self.assertIn("pandects:default:fulltext:1", held)
        self.assertEqual(held, {"pandects:default:fulltext:0"})

    def test_releases_slot_when_search_raises(self):
        held: set[str] = set()
        engine = _FakeEngine(held)
        with self.assertRaises(RuntimeError):
            with fulltext_slot(engine):
                raise RuntimeError("search failed")
        self.assertEqual(held, set())

    def test_all_slots_taken_answers_busy(self):
        held = {f"pandects:default:fulltext:{i}" for i in range(fulltext_slots.FULLTEXT_SEARCH_SLOTS)}
        engine = _FakeEngine(set(held))
        with self.assertRaises(FulltextSearchBusy) as caught:
            with fulltext_slot(engine):
                self.fail("ran without a slot")
        self.assertEqual(caught.exception.code, 503)
        self.assertIn(("Retry-After", "5"), caught.exception.get_headers())
        # It queued on one slot (GET_LOCK with a wait) before giving up.
        self.assertTrue(any("GET_LOCK(:n, :w)" in sql for sql in engine.conns[0].statements))

    def test_no_wait_configured_fails_fast(self):
        held = {f"pandects:default:fulltext:{i}" for i in range(fulltext_slots.FULLTEXT_SEARCH_SLOTS)}
        engine = _FakeEngine(set(held))
        with patch.object(fulltext_slots, "FULLTEXT_SLOT_WAIT_SECONDS", 0):
            with self.assertRaises(FulltextSearchBusy):
                with fulltext_slot(engine):
                    self.fail("ran without a slot")
        self.assertFalse(any("CONNECTION_ID" in sql for sql in engine.conns[0].statements))


if __name__ == "__main__":
    unittest.main()
