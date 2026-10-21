import unittest

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from etl.utils.db_utils import delete_stale_sections


def _engine() -> Engine:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        _ = conn.execute(text("ATTACH DATABASE ':memory:' AS pdx"))
        _ = conn.execute(
            text("CREATE TABLE pdx.sections (section_uuid TEXT, agreement_uuid TEXT, xml_version INTEGER)")
        )
        _ = conn.execute(
            text("CREATE TABLE pdx.clauses (clause_uuid TEXT, agreement_uuid TEXT, section_uuid TEXT)")
        )
        _ = conn.execute(text("CREATE TABLE pdx.tax_clause_assignments (clause_uuid TEXT)"))
    return engine


class DeleteStaleSectionsTests(unittest.TestCase):
    def test_deletes_only_older_versions_of_rebuilt_agreements(self) -> None:
        engine = _engine()
        with engine.begin() as conn:
            _ = conn.execute(
                text("INSERT INTO pdx.sections VALUES (:s, :a, :v)"),
                [
                    {"s": "a-kept", "a": "a", "v": 2},
                    {"s": "a-dropped", "a": "a", "v": 1},
                    {"s": "a-null", "a": "a", "v": None},
                    {"s": "b-old", "a": "b", "v": 1},
                ],
            )
            _ = conn.execute(
                text("INSERT INTO pdx.clauses VALUES (:c, :a, :s)"),
                [
                    {"c": "c-kept", "a": "a", "s": "a-kept"},
                    {"c": "c-dropped", "a": "a", "s": "a-dropped"},
                ],
            )
            _ = conn.execute(
                text("INSERT INTO pdx.tax_clause_assignments VALUES (:c)"),
                [{"c": "c-kept"}, {"c": "c-dropped"}],
            )

            deleted = delete_stale_sections({"a": 2}, "pdx", conn)

            sections = conn.execute(text("SELECT section_uuid FROM pdx.sections ORDER BY 1")).scalars().all()
            clauses = conn.execute(text("SELECT clause_uuid FROM pdx.clauses")).scalars().all()
            assignments = conn.execute(text("SELECT clause_uuid FROM pdx.tax_clause_assignments")).scalars().all()

        self.assertEqual(deleted, 2)
        # Agreement b was not rebuilt in this batch, so its rows are untouched.
        self.assertEqual(sections, ["a-kept", "b-old"])
        self.assertEqual(clauses, ["c-kept"])
        self.assertEqual(assignments, ["c-kept"])

    def test_no_agreements_is_a_noop(self) -> None:
        engine = _engine()
        with engine.begin() as conn:
            self.assertEqual(delete_stale_sections({}, "pdx", conn), 0)


if __name__ == "__main__":
    _ = unittest.main()
