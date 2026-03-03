# pyright: reportAny=false
import unittest

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine

from etl.utils.latest_sections_search import refresh_latest_sections_search


class LatestSectionsSearchRefreshTests(unittest.TestCase):
    engine: Engine | None = None
    conn: Connection | None = None

    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        self.conn = self.engine.connect()
        _ = self.conn.execute(
            text(
                """
            CREATE TABLE agreements (
                agreement_uuid TEXT PRIMARY KEY,
                filing_date TEXT,
                prob_filing REAL,
                filing_company_name TEXT,
                filing_company_cik TEXT,
                form_type TEXT,
                exhibit_type TEXT,
                target TEXT,
                acquirer TEXT,
                transaction_price_total REAL,
                transaction_price_stock REAL,
                transaction_price_cash REAL,
                transaction_price_assets REAL,
                transaction_consideration TEXT,
                target_type TEXT,
                acquirer_type TEXT,
                target_industry TEXT,
                acquirer_industry TEXT,
                announce_date TEXT,
                close_date TEXT,
                deal_status TEXT,
                attitude TEXT,
                deal_type TEXT,
                purpose TEXT,
                target_pe INTEGER,
                acquirer_pe INTEGER,
                verified INTEGER,
                url TEXT
            )
            """
            )
        )
        _ = self.conn.execute(
            text(
                """
            CREATE TABLE xml (
                agreement_uuid TEXT NOT NULL,
                version INTEGER NOT NULL,
                status TEXT,
                latest INTEGER NOT NULL
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
                article_title TEXT,
                section_title TEXT,
                section_standard_id TEXT,
                section_standard_id_gold_label TEXT,
                xml_version INTEGER
            )
            """
            )
        )
        _ = self.conn.execute(
            text(
                """
            CREATE TABLE latest_sections_search (
                section_uuid TEXT PRIMARY KEY,
                agreement_uuid TEXT NOT NULL,
                filing_date TEXT,
                prob_filing REAL,
                filing_company_name TEXT,
                filing_company_cik TEXT,
                form_type TEXT,
                exhibit_type TEXT,
                target TEXT,
                acquirer TEXT,
                transaction_price_total REAL,
                transaction_price_stock REAL,
                transaction_price_cash REAL,
                transaction_price_assets REAL,
                transaction_consideration TEXT,
                target_type TEXT,
                acquirer_type TEXT,
                target_industry TEXT,
                acquirer_industry TEXT,
                announce_date TEXT,
                close_date TEXT,
                deal_status TEXT,
                attitude TEXT,
                deal_type TEXT,
                purpose TEXT,
                target_pe INTEGER,
                acquirer_pe INTEGER,
                verified INTEGER,
                url TEXT,
                section_standard_ids TEXT,
                article_title TEXT,
                section_title TEXT
            )
            """
            )
        )
        _ = self.conn.execute(
            text(
                """
            CREATE TABLE latest_sections_search_standard_ids (
                standard_id TEXT NOT NULL,
                section_uuid TEXT NOT NULL,
                agreement_uuid TEXT NOT NULL,
                PRIMARY KEY (standard_id, section_uuid)
            )
            """
            )
        )
        _ = self.conn.execute(
            text(
                """
            INSERT INTO agreements (
                agreement_uuid, filing_date, target, acquirer, transaction_consideration,
                target_type, acquirer_type, target_industry, acquirer_industry,
                deal_status, attitude, deal_type, purpose, target_pe, acquirer_pe,
                verified, url
            ) VALUES (
                'a1', '2020-01-01', 'Target A', 'Acquirer A', 'cash',
                'public', 'public', 'tech', 'tech',
                'complete', 'friendly', 'merger', 'strategic', 0, 0,
                1, 'http://example.com/a1'
            )
            """
            )
        )

    def tearDown(self) -> None:
        if self.conn is not None:
            self.conn.close()
        if self.engine is not None:
            self.engine.dispose()

    def test_refresh_keeps_only_sections_from_latest_eligible_xml(self) -> None:
        assert self.conn is not None
        _ = self.conn.execute(
            text("INSERT INTO xml (agreement_uuid, version, status, latest) VALUES (:agreement_uuid, :version, :status, :latest)"),
            [
                {"agreement_uuid": "a1", "version": 1, "status": "verified", "latest": 0},
                {"agreement_uuid": "a1", "version": 2, "status": "verified", "latest": 1},
            ],
        )
        _ = self.conn.execute(
            text(
                """
            INSERT INTO sections (
                section_uuid, agreement_uuid, article_title, section_title,
                section_standard_id, section_standard_id_gold_label, xml_version
            ) VALUES (
                :section_uuid, :agreement_uuid, :article_title, :section_title,
                :section_standard_id, :section_standard_id_gold_label, :xml_version
            )
            """
            ),
            [
                {
                    "section_uuid": "stale-section",
                    "agreement_uuid": "a1",
                    "article_title": "ARTICLE I",
                    "section_title": "Old",
                    "section_standard_id": "[\"old\"]",
                    "section_standard_id_gold_label": None,
                    "xml_version": 1,
                },
                {
                    "section_uuid": "fresh-section",
                    "agreement_uuid": "a1",
                    "article_title": "ARTICLE I",
                    "section_title": "New",
                    "section_standard_id": "[\"fallback\"]",
                    "section_standard_id_gold_label": "[\"gold\"]",
                    "xml_version": 2,
                },
            ],
        )
        _ = self.conn.execute(
            text(
                """
            INSERT INTO latest_sections_search (
                section_uuid, agreement_uuid, filing_date, target, acquirer, section_standard_ids
            ) VALUES ('stale-section', 'a1', '2020-01-01', 'Target A', 'Acquirer A', '["old"]')
            """
            )
        )

        inserted = refresh_latest_sections_search(self.conn, "", ["a1"])

        self.assertEqual(inserted, 1)
        rows = self.conn.execute(
            text(
                """
            SELECT section_uuid, section_standard_ids
            FROM latest_sections_search
            ORDER BY section_uuid
            """
            )
        ).fetchall()
        self.assertEqual(rows, [("fresh-section", '["gold"]')])
        standard_id_rows = self.conn.execute(
            text(
                """
            SELECT standard_id, section_uuid, agreement_uuid
            FROM latest_sections_search_standard_ids
            ORDER BY standard_id, section_uuid
            """
            )
        ).fetchall()
        self.assertEqual(standard_id_rows, [("gold", "fresh-section", "a1")])

    def test_refresh_populates_standard_id_mapping_for_multi_label_sections(self) -> None:
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
                section_uuid, agreement_uuid, article_title, section_title,
                section_standard_id, section_standard_id_gold_label, xml_version
            ) VALUES (
                'multi-section', 'a1', 'ARTICLE I', 'Multi',
                '["fallback"]', '["governing_law","other","governing_law"]', 2
            )
            """
            )
        )

        inserted = refresh_latest_sections_search(self.conn, "", ["a1"])

        self.assertEqual(inserted, 1)
        standard_id_rows = self.conn.execute(
            text(
                """
            SELECT standard_id, section_uuid, agreement_uuid
            FROM latest_sections_search_standard_ids
            ORDER BY standard_id
            """
            )
        ).fetchall()
        self.assertEqual(
            standard_id_rows,
            [
                ("governing_law", "multi-section", "a1"),
                ("other", "multi-section", "a1"),
            ],
        )

    def test_refresh_deletes_rows_when_latest_xml_is_invalid(self) -> None:
        assert self.conn is not None
        _ = self.conn.execute(
            text("INSERT INTO xml (agreement_uuid, version, status, latest) VALUES (:agreement_uuid, :version, :status, :latest)"),
            [
                {"agreement_uuid": "a1", "version": 1, "status": "verified", "latest": 0},
                {"agreement_uuid": "a1", "version": 2, "status": "invalid", "latest": 1},
            ],
        )
        _ = self.conn.execute(
            text(
                """
            INSERT INTO sections (
                section_uuid, agreement_uuid, article_title, section_title,
                section_standard_id, section_standard_id_gold_label, xml_version
            ) VALUES ('stale-section', 'a1', 'ARTICLE I', 'Old', '["old"]', NULL, 1)
            """
            )
        )
        _ = self.conn.execute(
            text(
                """
            INSERT INTO latest_sections_search (
                section_uuid, agreement_uuid, filing_date, target, acquirer, section_standard_ids
            ) VALUES ('stale-section', 'a1', '2020-01-01', 'Target A', 'Acquirer A', '["old"]')
            """
            )
        )
        _ = self.conn.execute(
            text(
                """
            INSERT INTO latest_sections_search_standard_ids (
                standard_id, section_uuid, agreement_uuid
            ) VALUES ('old', 'stale-section', 'a1')
            """
            )
        )

        inserted = refresh_latest_sections_search(self.conn, "", ["a1"])

        self.assertEqual(inserted, 0)
        remaining = self.conn.execute(text("SELECT COUNT(*) FROM latest_sections_search")).fetchone()
        self.assertEqual(remaining, (0,))
        remaining_standard_ids = self.conn.execute(
            text("SELECT COUNT(*) FROM latest_sections_search_standard_ids")
        ).fetchone()
        self.assertEqual(remaining_standard_ids, (0,))


if __name__ == "__main__":
    _ = unittest.main()
