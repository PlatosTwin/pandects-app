import base64
import json
import os
import tempfile
import unittest
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import text
from sqlalchemy.dialects import mysql as mysql_dialect


def _set_default_env(main_db_uri: str, auth_db_uri: str) -> None:
    os.environ["SKIP_MAIN_DB_REFLECTION"] = "1"
    os.environ["MAIN_DATABASE_URI"] = main_db_uri
    os.environ["MAIN_DB_SCHEMA"] = ""
    os.environ.setdefault("MARIADB_USER", "root")
    os.environ.setdefault("MARIADB_PASSWORD", "password")
    os.environ.setdefault("MARIADB_HOST", "127.0.0.1")
    os.environ.setdefault("MARIADB_DATABASE", "pdx")
    os.environ.setdefault("AUTH_SECRET_KEY", "test-auth-secret")
    os.environ.setdefault("PUBLIC_API_BASE_URL", "http://localhost:5000")
    os.environ.setdefault("PUBLIC_FRONTEND_BASE_URL", "http://localhost:8080")
    os.environ.setdefault("GOOGLE_OAUTH_CLIENT_ID", "test-google-client-id")
    os.environ.setdefault("GOOGLE_OAUTH_CLIENT_SECRET", "test-google-client-secret")
    os.environ.setdefault("AUTH_SESSION_TRANSPORT", "bearer")
    os.environ["AUTH_DATABASE_URI"] = auth_db_uri


class MainRoutesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        main_db = tempfile.NamedTemporaryFile(prefix="pandects_main_", suffix=".sqlite", delete=False)
        main_db.close()
        auth_db = tempfile.NamedTemporaryFile(prefix="pandects_auth_", suffix=".sqlite", delete=False)
        auth_db.close()
        _set_default_env(f"sqlite:///{main_db.name}", f"sqlite:///{auth_db.name}")

        import backend.app as app_module

        cls.app_module = app_module
        sqlite_uri = f"sqlite:///{main_db.name}"
        cls.app = cls.app_module.create_test_app(
            config_overrides={
                "MAIN_DB_SCHEMA": "",
                "SQLALCHEMY_DATABASE_URI": sqlite_uri,
                "SQLALCHEMY_BINDS": {"auth": f"sqlite:///{auth_db.name}"},
            }
        )
        with cls.app.app_context():
            engine = cls.app_module.db.engine
            cls.app_module.metadata.create_all(engine)
            cls.app_module.db.create_all(bind_key="auth")

            with engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO agreements (agreement_uuid, filing_date, target, acquirer, verified, url, deal_type, "
                        "target_counsel, acquirer_counsel, "
                        "transaction_price_total, transaction_price_stock, transaction_price_cash, transaction_price_assets, "
                        "transaction_consideration, target_type, acquirer_type, target_industry, acquirer_industry, "
                        "deal_status, attitude, purpose, target_pe, acquirer_pe) "
                        "VALUES "
                        "('a1', '2020-01-01', 'Target A', 'Acquirer A', 1, 'http://example.com/a1', 'merger', "
                        "'Wilson Sonsini Goodrich & Rosati, P.C.; Goodwin Procter LLP', 'Wiggin and Dana LLP', "
                        "'50000000', NULL, '50000000', NULL, 'cash', 'public', 'public', 'tech', 'tech', 'complete', 'friendly', 'strategic', 0, 0), "
                        "('a2', '2021-02-01', 'Target B', 'Acquirer B', 0, 'http://example.com/a2', 'stock_acquisition', "
                        "'Wilson Sonsini Goodrich & Rosati Professional Corporation', 'Wiggin & Dana, LLP', "
                        "'150000000', '150000000', NULL, NULL, 'stock', 'private', 'public', 'healthcare', 'tech', 'pending', 'hostile', 'financial', 1, 0), "
                        "('a3', '2022-03-01', 'Target C', 'Acquirer C', 1, 'http://example.com/a3', 'asset_purchase', "
                        "'Wachtell, Lipton, Rosen & Katz', 'Skadden, Arps, Slate, Meagher & Flom LLP', "
                        "'300000000', NULL, NULL, '300000000', 'assets', 'private', 'private', 'energy', 'industrial', 'cancelled', 'friendly', 'strategic', 0, 1), "
                        "('a4', '2023-04-01', 'Target D', 'Acquirer D', 1, 'http://example.com/a4', 'merger', "
                        "'Sullivan & Cromwell LLP', 'Simpson Thacher & Bartlett LLP', "
                        "'12000000000', NULL, '12000000000', NULL, 'cash', 'public', 'public', 'finance', 'finance', 'pending', 'friendly', 'strategic', 0, 0)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO xml (agreement_uuid, xml, version, status, latest) VALUES "
                        "('a1', '<document><article>"
                        "<section uuid=\"00000000-0000-0000-0000-000000000002\"><text>STALE</text></section>"
                        "</article></document>', 1, NULL, 0), "
                        "('a1', '<document><article>"
                        "<section uuid=\"00000000-0000-0000-0000-000000000001\"><text>KEEP</text></section>"
                        "<section uuid=\"00000000-0000-0000-0000-000000000003\"><text>HIDE</text></section>"
                        "</article></document>', 2, NULL, 1), "
                        "('a2', '<document><article><section uuid=\"00000000-0000-0000-0000-000000000011\"><text>A2</text></section></article></document>', 1, 'verified', 1), "
                        "('a3', '<document><article><section uuid=\"00000000-0000-0000-0000-000000000021\"><text>A3</text></section></article></document>', 1, 'verified', 1), "
                        "('a4', '<document><article><section uuid=\"00000000-0000-0000-0000-000000000041\"><text>OLD VERIFIED</text></section></article></document>', 1, 'verified', 0), "
                        "('a4', '<document><article><section uuid=\"00000000-0000-0000-0000-000000000042\"><text>LATEST INVALID</text></section></article></document>', 2, 'invalid', 1)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO counsel (counsel_id, canonical_name, canonical_name_normalized) VALUES "
                        "(1, 'Wilson Sonsini Goodrich & Rosati', 'wilson sonsini goodrich rosati'), "
                        "(2, 'Goodwin Procter', 'goodwin procter'), "
                        "(3, 'Wiggin & Dana', 'wiggin dana'), "
                        "(4, 'Skadden, Arps, Slate, Meagher & Flom', 'skadden arps slate meagher flom'), "
                        "(5, 'Wachtell, Lipton, Rosen & Katz', 'wachtell lipton rosen katz'), "
                        "(6, 'Sullivan & Cromwell', 'sullivan cromwell'), "
                        "(7, 'Simpson Thacher & Bartlett', 'simpson thacher bartlett')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreement_counsel (agreement_uuid, side, position, raw_name, counsel_id) VALUES "
                        "('a1', 'target', 1, 'Wilson Sonsini Goodrich & Rosati, P.C.', 1), "
                        "('a1', 'target', 2, 'Goodwin Procter LLP', 2), "
                        "('a1', 'acquirer', 1, 'Wiggin and Dana LLP', 3), "
                        "('a2', 'target', 1, 'Wilson Sonsini Goodrich & Rosati Professional Corporation', 1), "
                        "('a2', 'acquirer', 1, 'Wiggin & Dana, LLP', 3), "
                        "('a3', 'target', 1, 'Wachtell, Lipton, Rosen & Katz', 5), "
                        "('a3', 'acquirer', 1, 'Skadden, Arps, Slate, Meagher & Flom LLP', 4), "
                        "('a4', 'target', 1, 'Sullivan & Cromwell LLP', 6), "
                        "('a4', 'acquirer', 1, 'Simpson Thacher & Bartlett LLP', 7)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO sections (agreement_uuid, section_uuid, article_title, section_title, "
                        "xml_content, section_standard_id, xml_version) VALUES "
                        "('a1', '00000000-0000-0000-0000-000000000001', "
                        "'ARTICLE I', 'Section 1', '<section>TEXT</section>', '[\"s1\"]', 2), "
                        "('a1', '00000000-0000-0000-0000-000000000002', "
                        "'ARTICLE I', 'Old Section', '<section>STALE</section>', '[\"s-old\"]', 1), "
                        "('a4', '00000000-0000-0000-0000-000000000041', "
                        "'ARTICLE I', 'Old Verified Section', '<section>OLD VERIFIED</section>', '[\"s4-old\"]', 1)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO latest_sections_search ("
                        "section_uuid, agreement_uuid, filing_date, prob_filing, filing_company_name, "
                        "filing_company_cik, form_type, exhibit_type, target, acquirer, "
                        "transaction_price_total, transaction_price_stock, transaction_price_cash, "
                        "transaction_price_assets, transaction_consideration, target_type, acquirer_type, "
                        "target_industry, acquirer_industry, announce_date, close_date, deal_status, "
                        "attitude, deal_type, purpose, target_pe, acquirer_pe, verified, url, "
                        "section_standard_ids, article_title, section_title"
                        ") VALUES ("
                        "'00000000-0000-0000-0000-000000000001', 'a1', '2020-01-01', NULL, NULL, NULL, "
                        "NULL, NULL, 'Target A', 'Acquirer A', '50000000', NULL, '50000000', NULL, 'cash', 'public', "
                        "'public', 'tech', 'tech', NULL, NULL, 'complete', 'friendly', 'merger', "
                        "'strategic', 0, 0, 1, 'http://example.com/a1', '[\"s1\"]', 'ARTICLE I', 'Section 1'"
                        ")"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO latest_sections_search_standard_ids (standard_id, section_uuid, agreement_uuid) "
                        "VALUES ('s1', '00000000-0000-0000-0000-000000000001', 'a1')"
                    )
                )
                conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS agreement_deal_type_summary ("
                        "year INTEGER NOT NULL, deal_type TEXT NOT NULL, count INTEGER NOT NULL)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS agreement_overview_summary ("
                        "singleton_key INTEGER NOT NULL PRIMARY KEY, "
                        "metadata_covered_agreements INTEGER NULL, "
                        "metadata_coverage_pct REAL NULL, "
                        "taxonomy_covered_sections INTEGER NULL, "
                        "taxonomy_coverage_pct REAL NULL, "
                        "latest_filing_date TEXT NULL)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS agreement_ownership_mix_summary ("
                        "year INTEGER NOT NULL, "
                        "target_bucket TEXT NOT NULL, "
                        "deal_count INTEGER NOT NULL, "
                        "total_transaction_value REAL NOT NULL)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS agreement_ownership_deal_size_summary ("
                        "year INTEGER NOT NULL, "
                        "target_bucket TEXT NOT NULL, "
                        "deal_count INTEGER NOT NULL, "
                        "p25_transaction_value REAL NULL, "
                        "median_transaction_value REAL NULL, "
                        "p75_transaction_value REAL NULL)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS agreement_buyer_type_matrix_summary ("
                        "target_bucket TEXT NOT NULL, "
                        "buyer_bucket TEXT NOT NULL, "
                        "deal_count INTEGER NOT NULL, "
                        "median_transaction_value REAL NULL)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS agreement_target_industry_summary ("
                        "year INTEGER NOT NULL, "
                        "industry TEXT NOT NULL, "
                        "deal_count INTEGER NOT NULL, "
                        "total_transaction_value REAL NOT NULL)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS agreement_industry_pairing_summary ("
                        "target_industry TEXT NOT NULL, "
                        "acquirer_industry TEXT NOT NULL, "
                        "deal_count INTEGER NOT NULL, "
                        "total_transaction_value REAL NOT NULL)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreement_deal_type_summary (year, deal_type, count) VALUES "
                        "(2020, 'merger', 1), "
                        "(2021, 'stock_acquisition', 2)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreement_overview_summary "
                        "(singleton_key, metadata_covered_agreements, metadata_coverage_pct, taxonomy_covered_sections, taxonomy_coverage_pct, latest_filing_date) VALUES "
                        "(1, 123, 61.5, 4567, 87.2, '2023-04-01')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreement_ownership_mix_summary (year, target_bucket, deal_count, total_transaction_value) VALUES "
                        "(2020, 'public', 1, 50000000), "
                        "(2021, 'private', 1, 150000000), "
                        "(2022, 'private', 1, 300000000)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreement_ownership_deal_size_summary "
                        "(year, target_bucket, deal_count, p25_transaction_value, median_transaction_value, p75_transaction_value) VALUES "
                        "(2020, 'public', 1, 50000000, 50000000, 50000000), "
                        "(2021, 'private', 1, 150000000, 150000000, 150000000), "
                        "(2022, 'private', 1, 300000000, 300000000, 300000000)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreement_buyer_type_matrix_summary "
                        "(target_bucket, buyer_bucket, deal_count, median_transaction_value) VALUES "
                        "('public', 'public_buyer', 1, 50000000), "
                        "('private', 'public_buyer', 1, 150000000), "
                        "('private', 'private_equity', 1, 300000000)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreement_target_industry_summary "
                        "(year, industry, deal_count, total_transaction_value) VALUES "
                        "(2020, 'tech', 1, 50000000), "
                        "(2021, 'healthcare', 1, 150000000), "
                        "(2022, 'energy', 1, 300000000)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreement_industry_pairing_summary "
                        "(target_industry, acquirer_industry, deal_count, total_transaction_value) VALUES "
                        "('energy', 'industrial', 1, 300000000), "
                        "('healthcare', 'tech', 1, 150000000), "
                        "('tech', 'tech', 1, 50000000)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO naics_sectors (super_sector, sector_group, sector_desc, sector_code) VALUES "
                        "('Goods-Producing Industries', 'Natural Resources and Mining', "
                        "'Agriculture, Forestry, Fishing and Hunting', 11), "
                        "('Goods-Producing Industries', 'Natural Resources and Mining', "
                        "'Mining, Quarrying, and Oil and Gas Extraction', 21)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO naics_sub_sectors (sub_sector_desc, sub_sector_code, sector_code) VALUES "
                        "('Crop Production', 111, 11), "
                        "('Animal Production', 112, 11), "
                        "('Oil and Gas Extraction', 211, 21)"
                    )
                )

    def test_agreements_index_pagination(self):
        client = self.app.test_client()
        res = client.get("/v1/agreements-index?page=1&page_size=2")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("total_count"), 3)
        self.assertEqual(body.get("total_pages"), 2)
        self.assertEqual(len(body.get("results", [])), 2)

    def test_agreements_bulk_cursor_pagination(self):
        client = self.app.test_client()
        res = client.get("/v1/agreements?page_size=2")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(len(body.get("results", [])), 2)
        self.assertTrue(body.get("has_next"))
        cursor = body.get("next_cursor")
        self.assertIsInstance(cursor, str)
        padded = str(cursor) + ("=" * (-len(str(cursor)) % 4))
        decoded = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
        self.assertEqual(decoded.get("agreement_uuid"), "a2")

        res_next = client.get(f"/v1/agreements?page_size=2&cursor={cursor}")
        self.assertEqual(res_next.status_code, 200)
        body_next = res_next.get_json()
        results_next = body_next.get("results", [])
        self.assertEqual(len(results_next), 1)
        self.assertEqual(results_next[0].get("agreement_uuid"), "a3")
        self.assertFalse(body_next.get("has_next"))
        self.assertIsNone(body_next.get("next_cursor"))

    def test_agreements_bulk_include_xml_forbidden_for_anonymous(self):
        client = self.app.test_client()
        res = client.get("/v1/agreements?include_xml=true")
        self.assertEqual(res.status_code, 403)

    def test_agreements_bulk_include_xml_with_api_key(self):
        client = self.app.test_client()
        with self.app.app_context():
            user = self.app_module.AuthUser()
            user.id = "00000000-0000-0000-0000-0000000000a1"
            user.email = "bulk-api@example.com"
            user.password_hash = "not-used"
            user.email_verified_at = datetime.now(timezone.utc).replace(tzinfo=None)
            self.app_module.db.session.add(user)
            self.app_module.db.session.commit()
            _, api_key = self.app_module._create_api_key(user_id=user.id, name="bulk")

        res = client.get(
            "/v1/agreements?include_xml=true&page_size=1",
            headers={"X-API-Key": api_key},
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        results = body.get("results", [])
        self.assertEqual(len(results), 1)
        self.assertIn("xml", results[0])
        self.assertIn("<document>", results[0]["xml"])

    def test_agreements_bulk_filters(self):
        client = self.app.test_client()
        res = client.get(
            "/v1/agreements"
            "?year=2021"
            "&deal_type=stock_acquisition"
            "&transaction_price_total=100M%20-%20250M"
            "&transaction_price_stock=100M%20-%20250M"
            "&target_pe=true"
            "&acquirer_pe=false"
            "&transaction_consideration=stock"
            "&target_type=private"
            "&acquirer_type=public"
            "&target_industry=healthcare"
            "&acquirer_industry=tech"
            "&deal_status=pending"
            "&attitude=hostile"
            "&purpose=financial"
            "&page_size=10"
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        results = body.get("results", [])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get("agreement_uuid"), "a2")

    def test_search_filters_include_enabled_metadata_fields(self):
        client = self.app.test_client()
        res = client.get(
            "/v1/sections"
            "?transaction_price_total=0%20-%20100M"
            "&transaction_price_cash=0%20-%20100M"
            "&transaction_consideration=cash"
            "&target_type=public"
            "&acquirer_type=public"
            "&target_industry=tech"
            "&acquirer_industry=tech"
            "&deal_status=complete"
            "&attitude=friendly"
            "&purpose=strategic"
            "&target_pe=false"
            "&acquirer_pe=false"
            "&page=1&page_size=10"
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("total_count"), 1)
        results = body.get("results", [])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get("section_uuid"), "00000000-0000-0000-0000-000000000001")

    def test_agreements_bulk_year_field_uses_filing_date_prefix(self):
        try:
            with self.app.app_context():
                engine = self.app_module.db.engine
                with engine.begin() as conn:
                    conn.execute(
                        text(
                            "INSERT INTO agreements (agreement_uuid, filing_date, target, acquirer, verified, url) "
                            "VALUES ('a_time', '2024-01-15T12:30:45', 'Target Time', 'Acquirer Time', 1, 'http://example.com/a_time')"
                        )
                    )
                    conn.execute(
                        text(
                            "INSERT INTO xml (agreement_uuid, xml, version, status, latest) VALUES "
                            "('a_time', '<document><article><section uuid=\"00000000-0000-0000-0000-000000000051\"><text>TIME</text></section></article></document>', 1, 'verified', 1)"
                        )
                    )

            client = self.app.test_client()
            res = client.get("/v1/agreements?agreement_uuid=a_time&page_size=10")
            self.assertEqual(res.status_code, 200)
            body = res.get_json()
            results = body.get("results", [])
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].get("year"), 2024)
        finally:
            with self.app.app_context():
                engine = self.app_module.db.engine
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM xml WHERE agreement_uuid = 'a_time'"))
                    conn.execute(text("DELETE FROM agreements WHERE agreement_uuid = 'a_time'"))

    def test_agreements_bulk_section_uuid_filter(self):
        client = self.app.test_client()
        res = client.get(
            "/v1/agreements?section_uuid=00000000-0000-0000-0000-000000000001&page_size=10"
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        results = body.get("results", [])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get("agreement_uuid"), "a1")

    def test_agreements_bulk_section_uuid_filter_excludes_stale_section(self):
        client = self.app.test_client()
        res = client.get(
            "/v1/agreements?section_uuid=00000000-0000-0000-0000-000000000002&page_size=10"
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("results", []), [])

    def test_agreements_bulk_section_uuid_filter_excludes_old_verified_when_latest_invalid(self):
        client = self.app.test_client()
        res = client.get(
            "/v1/agreements?section_uuid=00000000-0000-0000-0000-000000000041&page_size=10"
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("results", []), [])

    def test_agreements_bulk_rejects_standard_id_filter(self):
        client = self.app.test_client()
        res = client.get("/v1/agreements?standard_id=s1")
        self.assertEqual(res.status_code, 400)

    def test_search_basic(self):
        client = self.app.test_client()
        res = client.get("/v1/sections?year=2020&page=1&page_size=10")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("total_count"), 1)
        self.assertFalse(body.get("total_count_is_approximate"))
        self.assertEqual(len(body.get("results", [])), 1)
        self.assertIn("access", body)

    def test_search_by_standard_id(self):
        client = self.app.test_client()
        res = client.get("/v1/sections?standard_id=s1&page=1&page_size=10")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("total_count"), 1)
        self.assertFalse(body.get("total_count_is_approximate"))
        self.assertEqual(len(body.get("results", [])), 1)

    def test_search_by_standard_id_matches_multi_label_sections(self):
        with self.app.app_context():
            engine = self.app_module.db.engine
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO sections (agreement_uuid, section_uuid, article_title, section_title, "
                        "xml_content, section_standard_id, section_standard_id_gold_label, xml_version) VALUES "
                        "('a2', '00000000-0000-0000-0000-000000000004', 'ARTICLE I', 'Section Multi', "
                        "'<section>MULTI</section>', '[\"fallback\"]', '[\"multi\",\"other\"]', 1)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO latest_sections_search ("
                        "section_uuid, agreement_uuid, filing_date, prob_filing, filing_company_name, "
                        "filing_company_cik, form_type, exhibit_type, target, acquirer, "
                        "transaction_price_total, transaction_price_stock, transaction_price_cash, "
                        "transaction_price_assets, transaction_consideration, target_type, acquirer_type, "
                        "target_industry, acquirer_industry, announce_date, close_date, deal_status, "
                        "attitude, deal_type, purpose, target_pe, acquirer_pe, verified, url, "
                        "section_standard_ids, article_title, section_title"
                        ") VALUES ("
                        "'00000000-0000-0000-0000-000000000004', 'a2', '2021-02-01', NULL, NULL, NULL, "
                        "NULL, NULL, 'Target B', 'Acquirer B', NULL, NULL, NULL, NULL, 'stock', 'private', "
                        "'public', 'healthcare', 'tech', NULL, NULL, 'pending', 'hostile', 'stock_acquisition', "
                        "'financial', 1, 0, 0, 'http://example.com/a2', '[\"multi\",\"other\"]', "
                        "'ARTICLE I', 'Section Multi'"
                        ")"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO latest_sections_search_standard_ids (standard_id, section_uuid, agreement_uuid) "
                        "VALUES "
                        "('multi', '00000000-0000-0000-0000-000000000004', 'a2'), "
                        "('other', '00000000-0000-0000-0000-000000000004', 'a2')"
                    )
                )

        client = self.app.test_client()
        res = client.get("/v1/sections?standard_id=multi&page=1&page_size=10")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("total_count"), 1)
        results = body.get("results", [])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get("section_uuid"), "00000000-0000-0000-0000-000000000004")

    def test_search_marks_total_count_as_approximate_when_more_pages_exist(self):
        with self.app.app_context():
            engine = self.app_module.db.engine
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO agreements (agreement_uuid, filing_date, target, acquirer, verified, url, deal_type, "
                        "transaction_consideration, target_type, acquirer_type, target_industry, acquirer_industry, "
                        "deal_status, attitude, purpose, target_pe, acquirer_pe) "
                        "VALUES "
                        "('a5', '2024-01-01', 'Target E', 'Acquirer E', 1, 'http://example.com/a5', 'merger', "
                        "'cash', 'public', 'public', 'tech', 'tech', 'complete', 'friendly', 'strategic', 0, 0)"
                    )
                )
                for idx in range(10):
                    section_uuid = f"00000000-0000-0000-0000-0000000001{idx:02d}"
                    conn.execute(
                        text(
                            "INSERT INTO sections (agreement_uuid, section_uuid, article_title, section_title, "
                            "xml_content, section_standard_id, xml_version) VALUES "
                            "(:agreement_uuid, :section_uuid, 'ARTICLE I', :section_title, :xml_content, '[\"s5\"]', 1)"
                        ),
                        {
                            "agreement_uuid": "a5",
                            "section_uuid": section_uuid,
                            "section_title": f"Section {idx + 10}",
                            "xml_content": f"<section>EXTRA {idx}</section>",
                        },
                    )
                    conn.execute(
                        text(
                            "INSERT INTO latest_sections_search ("
                            "section_uuid, agreement_uuid, filing_date, prob_filing, filing_company_name, "
                            "filing_company_cik, form_type, exhibit_type, target, acquirer, "
                            "transaction_price_total, transaction_price_stock, transaction_price_cash, "
                            "transaction_price_assets, transaction_consideration, target_type, acquirer_type, "
                            "target_industry, acquirer_industry, announce_date, close_date, deal_status, "
                            "attitude, deal_type, purpose, target_pe, acquirer_pe, verified, url, "
                            "section_standard_ids, article_title, section_title"
                            ") VALUES ("
                            ":section_uuid, 'a5', '2024-01-01', NULL, NULL, NULL, "
                            "NULL, NULL, 'Target E', 'Acquirer E', NULL, NULL, NULL, NULL, 'cash', 'public', "
                            "'public', 'tech', 'tech', NULL, NULL, 'complete', 'friendly', 'merger', "
                            "'strategic', 0, 0, 1, 'http://example.com/a5', '[\"s5\"]', 'ARTICLE I', :section_title"
                            ")"
                        ),
                        {
                            "section_uuid": section_uuid,
                            "section_title": f"Section {idx + 10}",
                        },
                    )

        client = self.app.test_client()
        res = client.get("/v1/sections?year=2024&page=1&page_size=5")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(len(body.get("results", [])), 5)
        self.assertTrue(body.get("has_next"))
        self.assertTrue(body.get("total_count_is_approximate"))
        self.assertGreaterEqual(body.get("total_count"), 6)
        self.assertGreaterEqual(body.get("total_pages"), 2)

    def test_search_total_count_metadata_uses_table_estimate_for_unfiltered_search(self):
        with (
            patch.object(self.app_module, "_estimated_query_row_count", return_value=None),
            patch.object(
                self.app_module,
                "_estimated_latest_sections_search_table_rows",
                return_value=488296,
            ),
        ):
            total_count, is_approximate = self.app_module._search_total_count_metadata(
                query=object(),
                page=6,
                page_size=25,
                item_count=25,
                has_next=True,
                has_filters=False,
            )

        self.assertEqual(total_count, 488296)
        self.assertTrue(is_approximate)

    def test_search_excludes_stale_section_versions(self):
        client = self.app.test_client()
        res = client.get("/v1/sections?standard_id=s-old&page=1&page_size=10")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("total_count"), 0)
        self.assertEqual(body.get("results", []), [])

    def test_search_excludes_old_verified_when_latest_xml_is_invalid(self):
        client = self.app.test_client()
        res = client.get("/v1/sections?standard_id=s4-old&page=1&page_size=10")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("total_count"), 0)
        self.assertEqual(body.get("results", []), [])

    def test_search_with_requested_metadata(self):
        client = self.app.test_client()
        res = client.get("/v1/sections?year=2020&metadata=deal_type&page=1&page_size=10")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("total_count"), 1)
        results = body.get("results", [])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get("metadata"), {"deal_type": "merger"})

    def test_search_with_invalid_metadata_field(self):
        client = self.app.test_client()
        res = client.get("/v1/sections?metadata=not_a_field&page=1&page_size=10")
        self.assertEqual(res.status_code, 422)

    def test_agreements_index_year_query_matches_timestamp_like_filing_date(self):
        try:
            with self.app.app_context():
                engine = self.app_module.db.engine
                with engine.begin() as conn:
                    conn.execute(
                        text(
                            "INSERT INTO agreements (agreement_uuid, filing_date, target, acquirer, verified, url) "
                            "VALUES ('a_index_time', '2024-06-30 08:15:00', 'Target Index', 'Acquirer Index', 1, 'http://example.com/a_index_time')"
                        )
                    )
                    conn.execute(
                        text(
                            "INSERT INTO xml (agreement_uuid, xml, version, status, latest) VALUES "
                            "('a_index_time', '<document><article><section uuid=\"00000000-0000-0000-0000-000000000061\"><text>INDEX</text></section></article></document>', 1, 'verified', 1)"
                        )
                    )

            client = self.app.test_client()
            res = client.get("/v1/agreements-index?query=2024&page=1&page_size=10")
            self.assertEqual(res.status_code, 200)
            body = res.get_json()
            results = body.get("results", [])
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].get("agreement_uuid"), "a_index_time")
            self.assertEqual(results[0].get("year"), 2024)
        finally:
            with self.app.app_context():
                engine = self.app_module.db.engine
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM xml WHERE agreement_uuid = 'a_index_time'"))
                    conn.execute(text("DELETE FROM agreements WHERE agreement_uuid = 'a_index_time'"))

    def test_get_section_by_uuid(self):
        client = self.app.test_client()
        section_uuid = "00000000-0000-0000-0000-000000000001"
        res = client.get(f"/v1/sections/{section_uuid}")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("agreement_uuid"), "a1")
        self.assertEqual(body.get("section_uuid"), section_uuid)
        self.assertEqual(body.get("section_standard_id"), ["s1"])
        self.assertNotIn("articleStandardId", body)

    def test_get_section_by_uuid_rejects_stale_version(self):
        client = self.app.test_client()
        res = client.get("/v1/sections/00000000-0000-0000-0000-000000000002")
        self.assertEqual(res.status_code, 404)

    def test_get_section_by_uuid_rejects_old_verified_when_latest_xml_is_invalid(self):
        client = self.app.test_client()
        res = client.get("/v1/sections/00000000-0000-0000-0000-000000000041")
        self.assertEqual(res.status_code, 404)

    def test_get_agreement_rejects_old_verified_when_latest_xml_is_invalid(self):
        client = self.app.test_client()
        res = client.get("/v1/agreements/a4")
        self.assertEqual(res.status_code, 404)

    def test_agreement_redaction_for_anonymous(self):
        client = self.app.test_client()
        res = client.get(
            "/v1/agreements/a1?focus_section_uuid=00000000-0000-0000-0000-000000000001&neighbor_sections=0"
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertTrue(body.get("is_redacted"))
        xml = body.get("xml", "")
        self.assertIn("[REDACTED]", xml)

    def test_public_agreement_endpoints_exclude_gated_unverified_agreements(self):
        with self.app.app_context():
            engine = self.app_module.db.engine
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO agreements (agreement_uuid, filing_date, target, acquirer, verified, gated, url) "
                        "VALUES ('a_gated_hidden', '2025-06-01', 'Target Hidden', 'Acquirer Hidden', 0, 1, 'http://example.com/a_gated_hidden')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO xml (agreement_uuid, xml, version, status, latest) VALUES "
                        "('a_gated_hidden', '<document><article><section uuid=\"00000000-0000-0000-0000-000000000071\"><text>HIDDEN</text></section></article></document>', 1, 'verified', 1)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO sections (agreement_uuid, section_uuid, article_title, section_title, "
                        "xml_content, section_standard_id, xml_version) VALUES "
                        "('a_gated_hidden', '00000000-0000-0000-0000-000000000071', 'ARTICLE I', 'Hidden Section', '<section>HIDDEN</section>', '[\"s-hidden\"]', 1)"
                    )
                )

        try:
            client = self.app.test_client()

            list_res = client.get("/v1/agreements?agreement_uuid=a_gated_hidden&page_size=10")
            self.assertEqual(list_res.status_code, 200)
            self.assertEqual(list_res.get_json().get("results", []), [])

            detail_res = client.get("/v1/agreements/a_gated_hidden")
            self.assertEqual(detail_res.status_code, 404)

            index_res = client.get("/v1/agreements-index?query=Target Hidden&page=1&page_size=10")
            self.assertEqual(index_res.status_code, 200)
            self.assertEqual(index_res.get_json().get("results", []), [])

            filters_res = client.get("/v1/filter-options")
            self.assertEqual(filters_res.status_code, 200)
            body = filters_res.get_json()
            self.assertNotIn("Target Hidden", body.get("targets", []))
            self.assertNotIn("Acquirer Hidden", body.get("acquirers", []))
        finally:
            with self.app.app_context():
                engine = self.app_module.db.engine
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM sections WHERE agreement_uuid = 'a_gated_hidden'"))
                    conn.execute(text("DELETE FROM xml WHERE agreement_uuid = 'a_gated_hidden'"))
                    conn.execute(text("DELETE FROM agreements WHERE agreement_uuid = 'a_gated_hidden'"))
                self.app_module._filter_options_cache["payload"] = None
                self.app_module._filter_options_cache["ts"] = 0

    def test_agreements_deal_types_summary(self):
        client = self.app.test_client()
        res = client.get("/v1/agreements-deal-types-summary")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(
            body.get("years"),
            [
                {"year": 2020, "deal_type": "merger", "count": 1},
                {"year": 2021, "deal_type": "stock_acquisition", "count": 2},
            ],
        )

    def test_counsel_leaderboards_aggregate_variants_and_multi_firm_entries(self):
        client = self.app.test_client()
        res = client.get("/v1/counsel-leaderboards")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()

        buy_side = body.get("buy_side", {})
        sell_side = body.get("sell_side", {})

        buy_top_by_count = buy_side.get("top_by_count", [])
        self.assertEqual(buy_top_by_count[0]["counsel"], "Wiggin & Dana")
        self.assertEqual(buy_top_by_count[0]["deal_count"], 2)
        self.assertEqual(buy_top_by_count[0]["total_transaction_value"], 200000000.0)

        buy_top_by_value = buy_side.get("top_by_value", [])
        self.assertEqual(
            buy_top_by_value[0],
            {
                "counsel": "Skadden, Arps, Slate, Meagher & Flom",
                "deal_count": 1,
                "total_transaction_value": 300000000.0,
                "years": [
                    {
                        "year": 2022,
                        "deal_count": 1,
                        "total_transaction_value": 300000000.0,
                    }
                ],
            },
        )

        sell_top_by_count = sell_side.get("top_by_count", [])
        self.assertEqual(sell_top_by_count[0]["counsel"], "Wilson Sonsini Goodrich & Rosati")
        self.assertEqual(sell_top_by_count[0]["deal_count"], 2)
        self.assertEqual(sell_top_by_count[0]["total_transaction_value"], 200000000.0)
        self.assertEqual(
            sell_top_by_count[0]["years"],
            [
                {
                    "year": 2020,
                    "deal_count": 1,
                    "total_transaction_value": 50000000.0,
                },
                {
                    "year": 2021,
                    "deal_count": 1,
                    "total_transaction_value": 150000000.0,
                },
            ],
        )

        self.assertTrue(
            any(
                row["counsel"] == "Goodwin Procter"
                and row["deal_count"] == 1
                and row["total_transaction_value"] == 50000000.0
                for row in sell_top_by_count
            )
        )

    def test_agreements_status_summary_includes_overview_metrics(self):
        with self.app.app_context():
            engine = self.app_module.db.engine
            with engine.begin() as conn:
                conn.execute(text("DROP TABLE IF EXISTS agreement_overview_summary"))
                conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS agreement_overview_summary ("
                        "singleton_key INTEGER NOT NULL PRIMARY KEY, "
                        "metadata_covered_agreements INTEGER NULL, "
                        "metadata_coverage_pct REAL NULL, "
                        "taxonomy_covered_sections INTEGER NULL, "
                        "taxonomy_coverage_pct REAL NULL, "
                        "latest_filing_date TEXT NULL)"
                    )
                )
                conn.execute(text("DELETE FROM agreement_overview_summary"))
                conn.execute(
                    text(
                        "INSERT INTO agreement_overview_summary "
                        "(singleton_key, metadata_covered_agreements, metadata_coverage_pct, taxonomy_covered_sections, taxonomy_coverage_pct, latest_filing_date) VALUES "
                        "(1, 123, 61.5, 4567, 87.2, '2023-04-01')"
                    )
                )

        client = self.app.test_client()
        res = client.get("/v1/agreements-status-summary")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("latest_filing_date"), "2023-04-01")
        self.assertEqual(body.get("metadata_covered_agreements"), 123)
        self.assertEqual(body.get("metadata_coverage_pct"), 61.5)
        self.assertEqual(body.get("taxonomy_covered_sections"), 4567)
        self.assertEqual(body.get("taxonomy_coverage_pct"), 87.2)

    def test_agreements_status_summary_excludes_gated_unverified_from_latest_filing_date(self):
        with self.app.app_context():
            engine = self.app_module.db.engine
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS agreement_status_summary ("
                        "year INTEGER NOT NULL, color TEXT NOT NULL, current_stage TEXT NOT NULL, count INTEGER NOT NULL)"
                    )
                )
                conn.execute(text("DROP TABLE IF EXISTS agreement_overview_summary"))
                conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS agreement_overview_summary ("
                        "singleton_key INTEGER NOT NULL PRIMARY KEY, "
                        "metadata_covered_agreements INTEGER NULL, "
                        "metadata_coverage_pct REAL NULL, "
                        "taxonomy_covered_sections INTEGER NULL, "
                        "taxonomy_coverage_pct REAL NULL, "
                        "latest_filing_date TEXT NULL)"
                    )
                )
                conn.execute(text("DELETE FROM agreement_status_summary"))
                conn.execute(text("DELETE FROM agreement_overview_summary"))
                conn.execute(
                    text(
                        "INSERT INTO agreement_status_summary (year, color, current_stage, count) VALUES "
                        "(2023, 'green', 'processed', 1)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreement_overview_summary "
                        "(singleton_key, metadata_covered_agreements, metadata_coverage_pct, taxonomy_covered_sections, taxonomy_coverage_pct, latest_filing_date) VALUES "
                        "(1, 20, 50.0, 300, 75.0, '2023-04-01')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreements (agreement_uuid, filing_date, target, acquirer, verified, gated, url) "
                        "VALUES ('a_gated_pending', '2025-06-01', 'Target Pending', 'Acquirer Pending', 0, 1, 'http://example.com/a_gated_pending')"
                    )
                )

        try:
            client = self.app.test_client()
            res = client.get("/v1/agreements-status-summary")
            self.assertEqual(res.status_code, 200)
            body = res.get_json()
            self.assertEqual(body.get("latest_filing_date"), "2023-04-01")
            self.assertEqual(body.get("metadata_covered_agreements"), 20)
            self.assertEqual(body.get("metadata_coverage_pct"), 50.0)
            self.assertEqual(body.get("taxonomy_covered_sections"), 300)
            self.assertEqual(body.get("taxonomy_coverage_pct"), 75.0)
        finally:
            with self.app.app_context():
                engine = self.app_module.db.engine
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM agreement_status_summary"))
                    conn.execute(text("DELETE FROM agreement_overview_summary"))
                    conn.execute(text("DELETE FROM agreements WHERE agreement_uuid = 'a_gated_pending'"))

    def test_mysql_agreement_year_expr_avoids_str_to_date(self):
        mysql_bind = SimpleNamespace(dialect=SimpleNamespace(name="mysql"))
        with (
            self.app.app_context(),
            patch.object(self.app_module.db.session, "get_bind", return_value=mysql_bind),
        ):
            compiled = str(
                self.app_module._agreement_year_expr().compile(
                    dialect=mysql_dialect.dialect(),
                    compile_kwargs={"literal_binds": True},
                )
            )

        compiled_sql = compiled.lower()
        self.assertIn("substr(", compiled_sql)
        self.assertNotIn("str_to_date", compiled_sql)

    def test_year_from_filing_date_value_accepts_date_objects(self):
        self.assertEqual(
            self.app_module._year_from_filing_date_value(datetime(2024, 1, 1, 12, 0, 0)),
            2024,
        )
        self.assertEqual(self.app_module._year_from_filing_date_value(date(2023, 6, 1)), 2023)

    def test_naics_hierarchy(self):
        client = self.app.test_client()
        res = client.get("/v1/naics")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(
            body,
            {
                "sectors": [
                    {
                        "sector_code": "11",
                        "sector_desc": "Agriculture, Forestry, Fishing and Hunting",
                        "sector_group": "Natural Resources and Mining",
                        "super_sector": "Goods-Producing Industries",
                        "sub_sectors": [
                            {
                                "sub_sector_code": "111",
                                "sub_sector_desc": "Crop Production",
                            },
                            {
                                "sub_sector_code": "112",
                                "sub_sector_desc": "Animal Production",
                            },
                        ],
                    },
                    {
                        "sector_code": "21",
                        "sector_desc": "Mining, Quarrying, and Oil and Gas Extraction",
                        "sector_group": "Natural Resources and Mining",
                        "super_sector": "Goods-Producing Industries",
                        "sub_sectors": [
                            {
                                "sub_sector_code": "211",
                                "sub_sector_desc": "Oil and Gas Extraction",
                            }
                        ],
                    },
                ]
            },
        )

    def test_counsel_reference_list(self):
        client = self.app.test_client()
        res = client.get("/v1/counsel")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(
            body,
            {
                "counsel": [
                    {
                        "counsel_id": 2,
                        "canonical_name": "Goodwin Procter",
                    },
                    {
                        "counsel_id": 7,
                        "canonical_name": "Simpson Thacher & Bartlett",
                    },
                    {
                        "counsel_id": 4,
                        "canonical_name": "Skadden, Arps, Slate, Meagher & Flom",
                    },
                    {
                        "counsel_id": 6,
                        "canonical_name": "Sullivan & Cromwell",
                    },
                    {
                        "counsel_id": 5,
                        "canonical_name": "Wachtell, Lipton, Rosen & Katz",
                    },
                    {
                        "counsel_id": 3,
                        "canonical_name": "Wiggin & Dana",
                    },
                    {
                        "counsel_id": 1,
                        "canonical_name": "Wilson Sonsini Goodrich & Rosati",
                    },
                ]
            },
        )

    def test_agreement_trends_summary(self):
        client = self.app.test_client()
        res = client.get("/v1/agreement-trends")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertIsInstance(body, dict)

        ownership = body.get("ownership", {})
        mix_by_year = ownership.get("mix_by_year", [])
        self.assertEqual(
            mix_by_year,
            [
                {
                    "year": 2020,
                    "public_deal_count": 1,
                    "private_deal_count": 0,
                    "public_total_transaction_value": 50000000.0,
                    "private_total_transaction_value": 0.0,
                },
                {
                    "year": 2021,
                    "public_deal_count": 0,
                    "private_deal_count": 1,
                    "public_total_transaction_value": 0.0,
                    "private_total_transaction_value": 150000000.0,
                },
                {
                    "year": 2022,
                    "public_deal_count": 0,
                    "private_deal_count": 1,
                    "public_total_transaction_value": 0.0,
                    "private_total_transaction_value": 300000000.0,
                },
            ],
        )

        buyer_matrix = ownership.get("buyer_type_matrix", [])
        matrix_by_key = {
            (row["target_bucket"], row["buyer_bucket"]): row for row in buyer_matrix
        }
        self.assertEqual(matrix_by_key[("public", "public_buyer")]["deal_count"], 1)
        self.assertEqual(matrix_by_key[("private", "public_buyer")]["deal_count"], 1)
        self.assertEqual(matrix_by_key[("private", "private_equity")]["deal_count"], 1)

        industries = body.get("industries", {})
        by_year = industries.get("target_industries_by_year", [])
        self.assertIn(
            {
                "year": 2020,
                "industry": "tech",
                "deal_count": 1,
                "total_transaction_value": 50000000.0,
            },
            by_year,
        )
        self.assertIn(
            {
                "target_industry": "healthcare",
                "acquirer_industry": "tech",
                "deal_count": 1,
                "total_transaction_value": 150000000.0,
            },
            industries.get("pairings", []),
        )


if __name__ == "__main__":
    unittest.main()
