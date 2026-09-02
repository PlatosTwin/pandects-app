# pyright: reportAny=false
import argparse
import contextlib
import hashlib
import io
import sqlite3
import unittest
from typing import cast

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.pool import StaticPool

from etl.utils.section_text_search import (
    _drifted_agreements_sql,
    _source_hash_diff_expression,
    prune_section_text_search,
    refresh_section_text_search,
    select_section_text_backfill_agreements,
    select_section_text_drift_details,
    select_section_text_drifted_agreements,
)
from etl.utils.section_text_search_backfill import parse_args, run_backfill


def _register_sha256(raw_connection: sqlite3.Connection) -> None:
    raw_connection.create_function(
        "sha256",
        1,
        lambda value: hashlib.sha256(value.encode("utf-8")).digest(),
    )


_SCHEMA_SQL = (
    """
    CREATE TABLE xml (
        agreement_uuid TEXT NOT NULL,
        version INTEGER NOT NULL,
        status TEXT,
        latest INTEGER NOT NULL,
        PRIMARY KEY (agreement_uuid, version)
    )
    """,
    """
    CREATE TABLE sections (
        section_uuid TEXT PRIMARY KEY,
        agreement_uuid TEXT NOT NULL,
        xml_version INTEGER,
        xml_content TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE latest_sections_search (
        section_uuid TEXT PRIMARY KEY,
        agreement_uuid TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE section_text_search (
        section_uuid TEXT PRIMARY KEY,
        agreement_uuid TEXT NOT NULL,
        xml_version INTEGER,
        source_xml_sha256 BLOB,
        normalized_text TEXT NOT NULL,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
)


class SectionTextSearchBackfillCliTests(unittest.TestCase):
    def test_defaults_use_the_benchmarked_batch_sizes(self) -> None:
        args = parse_args([])

        self.assertEqual(args.agreement_batch_size, 100)
        self.assertEqual(args.write_batch_size, 1000)


class SectionTextSearchBackfillRunTests(unittest.TestCase):
    engine: Engine | None = None

    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            poolclass=StaticPool,
        )
        event.listen(
            self.engine,
            "connect",
            lambda raw_connection, _record: _register_sha256(raw_connection),
        )
        with self.engine.begin() as conn:
            for statement in _SCHEMA_SQL:
                _ = conn.execute(text(statement))
            _ = conn.execute(
                text(
                    """
                    INSERT INTO xml (agreement_uuid, version, status, latest) VALUES
                        ('a1', 1, 'verified', 0),
                        ('a1', 2, 'verified', 1),
                        ('a2', 1, 'verified', 1)
                    """
                )
            )
            _ = conn.execute(
                text(
                    """
                    INSERT INTO sections (
                        section_uuid, agreement_uuid, xml_version, xml_content
                    ) VALUES
                        ('a1-current', 'a1', 2, '<text>Current.</text>'),
                        ('a1-superseded', 'a1', 1, '<text>Superseded.</text>'),
                        ('a2-current', 'a2', 1, '<text>Missing text row.</text>')
                    """
                )
            )

    def tearDown(self) -> None:
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None

    @staticmethod
    def _drift_only_args() -> argparse.Namespace:
        return argparse.Namespace(
            schema="",
            target_table="section_text_search",
            agreement_batch_size=1,
            write_batch_size=1000,
            drift_only=True,
            after_agreement_uuid="",
            max_batches=None,
        )

    def _run(self) -> tuple[str, SystemExit | None]:
        assert self.engine is not None
        output = io.StringIO()
        exit_error: SystemExit | None = None
        with contextlib.redirect_stdout(output):
            try:
                run_backfill(self.engine, self._drift_only_args())
            except SystemExit as error:
                exit_error = error
        return output.getvalue(), exit_error

    def test_drift_only_pass_converges_when_every_drifted_section_is_materializable(
        self,
    ) -> None:
        assert self.engine is not None
        with self.engine.begin() as conn:
            _ = conn.execute(
                text(
                    """
                    INSERT INTO latest_sections_search (section_uuid, agreement_uuid)
                    VALUES ('a1-current', 'a1'), ('a2-current', 'a2')
                    """
                )
            )

        first_output, first_exit = self._run()
        second_output, second_exit = self._run()

        self.assertIsNone(first_exit)
        self.assertIsNone(second_exit)
        self.assertIn(
            "Committed batch 1: agreements=1, sections=1, unresolved_agreements=0, "
            "resume_after=a1",
            first_output,
        )
        self.assertIn("Done: batches=2, agreements=2, sections=2", first_output)
        self.assertIn("Done: batches=0, agreements=0, sections=0", second_output)

    def test_drift_only_pass_reports_unmaterializable_sections_and_exits_nonzero(
        self,
    ) -> None:
        assert self.engine is not None
        with self.engine.begin() as conn:
            _ = conn.execute(
                text(
                    """
                    INSERT INTO latest_sections_search (section_uuid, agreement_uuid)
                    VALUES
                        ('a1-current', 'a1'),
                        ('a1-superseded', 'a1'),
                        ('a2-current', 'a2')
                    """
                )
            )
            _ = conn.execute(
                text(
                    """
                    INSERT INTO sections (
                        section_uuid, agreement_uuid, xml_version, xml_content
                    ) VALUES ('a2-unlisted', 'a2', 1, '<text>Eligible but unlisted.</text>')
                    """
                )
            )

        output, exit_error = self._run()

        self.assertIsNotNone(exit_error)
        assert exit_error is not None
        self.assertIn(
            "2 section(s) across 2 agreement(s) stay drifted after refresh",
            str(exit_error.code),
        )
        self.assertIn(
            "Unresolved drift: agreement=a1 section=a1-superseded reason=missing text row",
            output,
        )
        self.assertIn(
            "Unresolved drift: agreement=a2 section=a2-unlisted "
            "reason=text row has no latest_sections_search row",
            output,
        )
        self.assertIn(
            "Committed batch 1: agreements=0, sections=1, unresolved_agreements=1, "
            "resume_after=a1",
            output,
        )
        self.assertIn(
            "Committed batch 2: agreements=0, sections=2, unresolved_agreements=1, "
            "resume_after=a2",
            output,
        )
        self.assertIn("Done: batches=2, agreements=0, sections=3", output)
        with self.engine.begin() as conn:
            rows = conn.execute(
                text("SELECT section_uuid FROM section_text_search ORDER BY section_uuid")
            ).scalars().all()
        self.assertEqual(rows, ["a1-current", "a2-current", "a2-unlisted"])


class SectionTextSearchRefreshTests(unittest.TestCase):
    engine: Engine | None = None
    conn: Connection | None = None

    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        self.conn = self.engine.connect()
        _register_sha256(
            cast(sqlite3.Connection, self.conn.connection.driver_connection)
        )
        for statement in _SCHEMA_SQL:
            _ = self.conn.execute(text(statement))

    def tearDown(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None

    def test_refresh_materializes_all_current_eligible_sections_in_small_batches(self) -> None:
        assert self.conn is not None
        _ = self.conn.execute(
            text(
                """
                INSERT INTO xml (agreement_uuid, version, status, latest) VALUES
                    ('a1', 1, 'verified', 0),
                    ('a1', 2, 'verified', 1),
                    ('a2', 1, NULL, 1),
                    ('a3', 1, 'invalid', 1)
                """
            )
        )
        _ = self.conn.execute(
            text(
                """
                INSERT INTO sections (
                    section_uuid, agreement_uuid, xml_version, xml_content
                ) VALUES
                    ('a1-old', 'a1', 1, '<text>Old language.</text>'),
                    ('a1-new-1', 'a1', 2, '<text>Buyer &amp; Seller.</text>'),
                    ('a1-new-2', 'a1', 2, '<text>Second café clause.</text>'),
                    ('a2-new', 'a2', 1, '<text>Null status remains eligible.</text>'),
                    ('a3-invalid', 'a3', 1, '<text>Must not be searchable.</text>')
                """
            )
        )

        refreshed = refresh_section_text_search(
            self.conn,
            "",
            ["a3", "a1", "a2", "a1"],
            write_batch_size=1,
        )

        self.assertEqual(refreshed, 3)
        rows = self.conn.execute(
            text(
                """
                SELECT section_uuid, agreement_uuid, xml_version,
                       source_xml_sha256, normalized_text
                FROM section_text_search
                ORDER BY section_uuid
                """
            )
        ).fetchall()
        self.assertEqual(
            rows,
            [
                (
                    "a1-new-1",
                    "a1",
                    2,
                    hashlib.sha256(b"<text>Buyer &amp; Seller.</text>").digest(),
                    "buyer & seller.",
                ),
                (
                    "a1-new-2",
                    "a1",
                    2,
                    hashlib.sha256(
                        "<text>Second café clause.</text>".encode("utf-8")
                    ).digest(),
                    "second café clause.",
                ),
                (
                    "a2-new",
                    "a2",
                    1,
                    hashlib.sha256(
                        b"<text>Null status remains eligible.</text>"
                    ).digest(),
                    "null status remains eligible.",
                ),
            ],
        )

    def test_refresh_updates_changed_text_and_deletes_stale_rows(self) -> None:
        assert self.conn is not None
        _ = self.conn.execute(
            text(
                """
                INSERT INTO xml (agreement_uuid, version, status, latest)
                VALUES ('a1', 2, 'verified', 1)
                """
            )
        )
        _ = self.conn.execute(
            text(
                """
                INSERT INTO sections (
                    section_uuid, agreement_uuid, xml_version, xml_content
                ) VALUES ('current', 'a1', 2, '<text>Updated text.</text>')
                """
            )
        )
        _ = self.conn.execute(
            text(
                """
                INSERT INTO section_text_search (
                    section_uuid, agreement_uuid, xml_version, normalized_text
                ) VALUES
                    ('current', 'a1', 1, 'old text.'),
                    ('stale', 'a1', 1, 'stale text.'),
                    ('other', 'a2', 1, 'other agreement.')
                """
            )
        )

        first_count = refresh_section_text_search(self.conn, "", ["a1"])
        _ = self.conn.execute(
            text(
                """
                UPDATE sections
                SET xml_content = '<text>Same version changed.</text>'
                WHERE section_uuid = 'current'
                """
            )
        )
        second_count = refresh_section_text_search(self.conn, "", ["a1"])

        self.assertEqual(first_count, 1)
        self.assertEqual(second_count, 1)
        rows = self.conn.execute(
            text(
                """
                SELECT section_uuid, agreement_uuid, xml_version,
                       source_xml_sha256, normalized_text
                FROM section_text_search
                ORDER BY section_uuid
                """
            )
        ).fetchall()
        self.assertEqual(
            rows,
            [
                (
                    "current",
                    "a1",
                    2,
                    hashlib.sha256(b"<text>Same version changed.</text>").digest(),
                    "same version changed.",
                ),
                ("other", "a2", 1, None, "other agreement."),
            ],
        )

    def test_refresh_fails_fast_when_text_table_is_missing_columns(self) -> None:
        assert self.conn is not None
        _ = self.conn.execute(text("DROP TABLE section_text_search"))
        _ = self.conn.execute(
            text(
                """
                CREATE TABLE section_text_search (
                    section_uuid TEXT PRIMARY KEY,
                    agreement_uuid TEXT NOT NULL
                )
                """
            )
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "missing required columns: normalized_text, source_xml_sha256, xml_version",
        ):
            _ = refresh_section_text_search(self.conn, "", ["a1"])

    def test_refresh_rejects_invalid_batch_sizes(self) -> None:
        assert self.conn is not None
        with self.assertRaisesRegex(ValueError, "write_batch_size"):
            _ = refresh_section_text_search(
                self.conn,
                "",
                ["a1"],
                write_batch_size=0,
            )
        with self.assertRaisesRegex(ValueError, "agreement_query_batch_size"):
            _ = refresh_section_text_search(
                self.conn,
                "",
                ["a1"],
                agreement_query_batch_size=0,
            )

    def test_backfill_agreement_selection_uses_a_keyset_cursor(self) -> None:
        assert self.conn is not None
        _ = self.conn.execute(
            text(
                """
                INSERT INTO latest_sections_search (section_uuid, agreement_uuid) VALUES
                    ('s1', 'a1'), ('s2', 'a1'), ('s3', 'a2'), ('s4', 'a3')
                """
            )
        )

        agreement_uuids = select_section_text_backfill_agreements(
            self.conn,
            "",
            after_agreement_uuid="a1",
            limit=2,
        )

        self.assertEqual(agreement_uuids, ["a2", "a3"])

    def test_drift_selection_finds_missing_wrong_version_and_stale_rows(self) -> None:
        assert self.conn is not None
        _ = self.conn.execute(
            text(
                """
                INSERT INTO latest_sections_search (section_uuid, agreement_uuid) VALUES
                    ('current-a1', 'a1'), ('current-a2', 'a2'),
                    ('current-a3', 'a3'), ('current-a5', 'a5')
                """
            )
        )
        _ = self.conn.execute(
            text(
                """
                INSERT INTO sections (
                    section_uuid, agreement_uuid, xml_version, xml_content
                ) VALUES
                    ('current-a1', 'a1', 1, '<text>Current.</text>'),
                    ('current-a2', 'a2', 2, '<text>Changed.</text>'),
                    ('current-a3', 'a3', 1, '<text>Missing.</text>'),
                    ('current-a5', 'a5', 1, '<text>Same version, changed XML.</text>')
                """
            )
        )
        _ = self.conn.execute(
            text(
                """
                INSERT INTO section_text_search (
                    section_uuid, agreement_uuid, xml_version,
                    source_xml_sha256, normalized_text
                ) VALUES
                    ('current-a1', 'a1', 1, sha256('<text>Current.</text>'), 'current.'),
                    ('current-a2', 'a2', 1, sha256('<text>Changed.</text>'), 'old.'),
                    ('stale-a4', 'a4', 1, sha256('<text>Stale.</text>'), 'stale.'),
                    ('current-a5', 'a5', 1, sha256('<text>Original XML.</text>'), 'old.')
                """
            )
        )

        agreement_uuids = select_section_text_drifted_agreements(
            self.conn,
            "",
            table_name="section_text_search",
            after_agreement_uuid="a1",
            limit=4,
        )

        self.assertEqual(agreement_uuids, ["a2", "a3", "a4", "a5"])

    def test_drift_selection_pages_each_branch_before_the_union(self) -> None:
        sql = _drifted_agreements_sql(
            "mysql",
            latest_table="pdx.latest_sections_search",
            sections_table="pdx.sections",
            target_table="pdx.section_text_search",
        )

        branches = sql.split("UNION")
        self.assertEqual(len(branches), 2)
        for branch in branches:
            self.assertIn("agreement_uuid > :after_agreement_uuid", branch)
            self.assertIn("LIMIT :limit", branch)
        self.assertEqual(sql.count("LIMIT :limit"), 3)
        self.assertEqual(sql.count("ORDER BY"), 3)
        self.assertLess(
            branches[0].index("LIMIT :limit"),
            branches[0].index(") missing_or_stale"),
        )
        self.assertLess(
            branches[1].index("LIMIT :limit"),
            branches[1].index(") orphaned"),
        )

    def test_drift_details_name_each_drifted_section_and_reason(self) -> None:
        assert self.conn is not None
        _ = self.conn.execute(
            text(
                """
                INSERT INTO latest_sections_search (section_uuid, agreement_uuid) VALUES
                    ('current-a1', 'a1'), ('current-a2', 'a2'),
                    ('current-a3', 'a3'), ('current-a5', 'a5'), ('current-a6', 'a6')
                """
            )
        )
        _ = self.conn.execute(
            text(
                """
                INSERT INTO sections (
                    section_uuid, agreement_uuid, xml_version, xml_content
                ) VALUES
                    ('current-a1', 'a1', 1, '<text>Current.</text>'),
                    ('current-a2', 'a2', 2, '<text>Changed.</text>'),
                    ('current-a3', 'a3', 1, '<text>Missing.</text>'),
                    ('current-a5', 'a5', 1, '<text>Same version, changed XML.</text>'),
                    ('current-a6', 'a6', 1, '<text>Not requested.</text>')
                """
            )
        )
        _ = self.conn.execute(
            text(
                """
                INSERT INTO section_text_search (
                    section_uuid, agreement_uuid, xml_version,
                    source_xml_sha256, normalized_text
                ) VALUES
                    ('current-a1', 'a1', 1, sha256('<text>Current.</text>'), 'current.'),
                    ('current-a2', 'a2', 1, sha256('<text>Changed.</text>'), 'old.'),
                    ('stale-a4', 'a4', 1, sha256('<text>Stale.</text>'), 'stale.'),
                    ('current-a5', 'a5', 1, sha256('<text>Original XML.</text>'), 'old.')
                """
            )
        )

        details = select_section_text_drift_details(
            self.conn,
            "",
            table_name="section_text_search",
            agreement_uuids=["a5", "a1", "a2", "a3", "a4", "a1"],
        )

        self.assertEqual(
            details,
            [
                {
                    "section_uuid": "current-a2",
                    "agreement_uuid": "a2",
                    "reason": "xml_version differs from sections",
                },
                {
                    "section_uuid": "current-a3",
                    "agreement_uuid": "a3",
                    "reason": "missing text row",
                },
                {
                    "section_uuid": "stale-a4",
                    "agreement_uuid": "a4",
                    "reason": "text row has no latest_sections_search row",
                },
                {
                    "section_uuid": "current-a5",
                    "agreement_uuid": "a5",
                    "reason": "source_xml_sha256 differs from sections.xml_content",
                },
            ],
        )

    def test_prune_deletes_only_text_rows_that_left_the_current_section_set(self) -> None:
        assert self.conn is not None
        _ = self.conn.execute(
            text(
                """
                INSERT INTO latest_sections_search (section_uuid, agreement_uuid)
                VALUES ('kept-a1', 'a1'), ('kept-a2', 'a2')
                """
            )
        )
        _ = self.conn.execute(
            text(
                """
                INSERT INTO section_text_search (
                    section_uuid, agreement_uuid, xml_version, normalized_text
                ) VALUES
                    ('kept-a1', 'a1', 1, 'kept.'),
                    ('gone-a1', 'a1', 1, 'gone.'),
                    ('kept-a2', 'a2', 1, 'kept.'),
                    ('gone-a3', 'a3', 1, 'other agreement, untouched.')
                """
            )
        )

        pruned = prune_section_text_search(self.conn, "", ["a1", "a2", "", "a1"])

        self.assertEqual(pruned, 1)
        rows = self.conn.execute(
            text("SELECT section_uuid FROM section_text_search ORDER BY section_uuid")
        ).scalars().all()
        self.assertEqual(rows, ["gone-a3", "kept-a1", "kept-a2"])

    def test_prune_is_a_no_op_without_agreements_or_without_the_table(self) -> None:
        assert self.conn is not None
        self.assertEqual(prune_section_text_search(self.conn, "", []), 0)
        _ = self.conn.execute(text("DROP TABLE section_text_search"))
        self.assertEqual(prune_section_text_search(self.conn, "", ["a1"]), 0)

    def test_mariadb_drift_expression_hashes_source_xml_in_database(self) -> None:
        self.assertEqual(
            _source_hash_diff_expression("mysql"),
            "t.source_xml_sha256 IS NULL "
            "OR t.source_xml_sha256 <> UNHEX(SHA2(s.xml_content, 256))",
        )


if __name__ == "__main__":
    _ = unittest.main()
