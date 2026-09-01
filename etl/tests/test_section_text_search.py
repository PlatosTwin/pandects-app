# pyright: reportAny=false
import hashlib
import sqlite3
import unittest
from typing import cast

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine

from etl.utils.section_text_search import (
    _source_hash_diff_expression,
    refresh_section_text_search,
    select_section_text_backfill_agreements,
    select_section_text_drifted_agreements,
)
from etl.utils.section_text_search_backfill import parse_args


class SectionTextSearchBackfillCliTests(unittest.TestCase):
    def test_defaults_use_the_benchmarked_batch_sizes(self) -> None:
        args = parse_args([])

        self.assertEqual(args.agreement_batch_size, 100)
        self.assertEqual(args.write_batch_size, 1000)


class SectionTextSearchRefreshTests(unittest.TestCase):
    engine: Engine | None = None
    conn: Connection | None = None

    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        self.conn = self.engine.connect()
        raw_connection = cast(
            sqlite3.Connection,
            self.conn.connection.driver_connection,
        )
        raw_connection.create_function(
            "sha256",
            1,
            lambda value: hashlib.sha256(value.encode("utf-8")).digest(),
        )
        _ = self.conn.execute(
            text(
                """
                CREATE TABLE xml (
                    agreement_uuid TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    status TEXT,
                    latest INTEGER NOT NULL,
                    PRIMARY KEY (agreement_uuid, version)
                )
                """
            )
        )
        _ = self.conn.execute(
            text(
                """
                CREATE TABLE sections (
                    section_uuid TEXT PRIMARY KEY,
                    agreement_uuid TEXT NOT NULL,
                    xml_version INTEGER,
                    xml_content TEXT NOT NULL
                )
                """
            )
        )
        _ = self.conn.execute(
            text(
                """
                CREATE TABLE latest_sections_search (
                    section_uuid TEXT PRIMARY KEY,
                    agreement_uuid TEXT NOT NULL
                )
                """
            )
        )
        _ = self.conn.execute(
            text(
                """
                CREATE TABLE section_text_search (
                    section_uuid TEXT PRIMARY KEY,
                    agreement_uuid TEXT NOT NULL,
                    xml_version INTEGER,
                    source_xml_sha256 BLOB,
                    normalized_text TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )

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

    def test_mariadb_drift_expression_hashes_source_xml_in_database(self) -> None:
        self.assertEqual(
            _source_hash_diff_expression("mysql"),
            "t.source_xml_sha256 IS NULL "
            "OR t.source_xml_sha256 <> UNHEX(SHA2(s.xml_content, 256))",
        )


if __name__ == "__main__":
    _ = unittest.main()
