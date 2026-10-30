import base64
import gzip
import json
import os
import tempfile
import unittest
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sqlalchemy import text
from sqlalchemy.dialects import mysql as mysql_dialect

from backend.core.caching import clear_all_ttl_caches


def _set_default_env(main_db_uri: str, auth_db_uri: str) -> None:
    os.environ["SKIP_MAIN_DB_REFLECTION"] = "1"
    os.environ["TEMPORARY_ACCESS_LOCKDOWN"] = "0"
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
                        "target_counsel, acquirer_counsel, target_industry, acquirer_industry, announce_date, close_date, deal_status, "
                        "attitude, deal_type, purpose, target_pe, acquirer_pe, verified, url, "
                        "section_standard_ids, article_title, section_title"
                        ") VALUES ("
                        "'00000000-0000-0000-0000-000000000001', 'a1', '2020-01-01', NULL, NULL, NULL, "
                        "NULL, NULL, 'Target A', 'Acquirer A', '50000000', NULL, '50000000', NULL, 'cash', 'public', "
                        "'public', 'Wilson Sonsini Goodrich & Rosati, P.C.; Goodwin Procter LLP', 'Wiggin and Dana LLP', 'tech', 'tech', NULL, NULL, 'complete', 'friendly', 'merger', "
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
                        "INSERT INTO tax_clause_taxonomy_l1 (standard_id, label) VALUES "
                        "('tax_root', 'Tax')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO tax_clause_taxonomy_l2 (standard_id, label, parent_id) VALUES "
                        "('tax_operational', 'Operational Tax Matters', 'tax_root')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO tax_clause_taxonomy_l3 (standard_id, label, parent_id) VALUES "
                        "('tax_transfer', 'Transfer Taxes', 'tax_operational'), "
                        "('tax_treatment', 'Tax Treatment of the Transaction', 'tax_operational')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO clauses ("
                        "clause_uuid, agreement_uuid, section_uuid, xml_version, module, clause_order, "
                        "anchor_label, start_char, end_char, clause_text, source_method, context_type"
                        ") VALUES ("
                        "'clause-a1-1', 'a1', '00000000-0000-0000-0000-000000000001', 2, 'tax', 1, "
                        "'(a)', 0, 55, 'Parent shall bear all transfer taxes.', 'enumerated_split', 'operative'"
                        "), ("
                        "'clause-a1-2', 'a1', '00000000-0000-0000-0000-000000000001', 2, 'tax', 2, "
                        "'(b)', 56, 120, 'Company and Parent intend the merger to qualify as tax-free.', 'enumerated_split', 'rep_warranty'"
                        ")"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO tax_clause_assignments (clause_uuid, standard_id, is_gold_label, model_name, assigned_at) VALUES "
                        "('clause-a1-1', 'tax_transfer', 1, 'gpt-5-mini', '2026-04-02T00:00:00Z'), "
                        "('clause-a1-2', 'tax_treatment', 1, 'gpt-5-mini', '2026-04-02T00:00:00Z')"
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
                        "CREATE TABLE IF NOT EXISTS agreement_filter_option_summary ("
                        "field_name TEXT NOT NULL, "
                        "option_value TEXT NOT NULL, "
                        "agreement_count INTEGER NOT NULL)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS agreement_metadata_field_coverage_summary ("
                        "field_name TEXT NOT NULL PRIMARY KEY, "
                        "label TEXT NOT NULL, "
                        "ingested_eligible_agreements INTEGER NOT NULL, "
                        "ingested_covered_agreements INTEGER NOT NULL, "
                        "ingested_coverage_pct REAL NULL, "
                        "processed_eligible_agreements INTEGER NOT NULL, "
                        "processed_covered_agreements INTEGER NOT NULL, "
                        "processed_coverage_pct REAL NULL, "
                        "note TEXT NOT NULL)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS agreement_counsel_leaderboard_summary ("
                        "side TEXT NOT NULL, "
                        "counsel_key TEXT NOT NULL, "
                        "counsel TEXT NOT NULL, "
                        "year INTEGER NOT NULL, "
                        "deal_count INTEGER NOT NULL, "
                        "total_transaction_value REAL NOT NULL)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS agreement_index_summary ("
                        "agreement_uuid TEXT NOT NULL PRIMARY KEY, "
                        "year INTEGER NULL, "
                        "target TEXT NULL, "
                        "acquirer TEXT NULL, "
                        "verified INTEGER NOT NULL)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS summary_data ("
                        "count_agreements INTEGER NOT NULL, "
                        "count_sections INTEGER NOT NULL, "
                        "count_pages INTEGER NOT NULL)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO summary_data (count_agreements, count_sections, count_pages) "
                        "VALUES (4, 3, 12)"
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
                        "(2020, '111', 1, 50000000), "
                        "(2021, '21', 1, 150000000), "
                        "(2022, '211', 1, 300000000)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreement_industry_pairing_summary "
                        "(target_industry, acquirer_industry, deal_count, total_transaction_value) VALUES "
                        "('211', '21', 1, 300000000), "
                        "('21', '111', 1, 150000000), "
                        "('111', '111', 1, 50000000)"
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

    def setUp(self) -> None:
        # Reset every shared mutable state on backend.app that a request
        # path could touch — otherwise a test that exercises the login
        # throttle or API-key last-used touch leaves residue for the next
        # test and produces order-dependent failures.
        self.app_module._rate_limit_state.clear()
        self.app_module._endpoint_rate_limit_state.clear()
        self.app_module._account_login_failure_state.clear()
        self.app_module._search_count_cache.clear()
        self.app_module._api_key_last_used_touch_state.clear()
        self._restore_base_dataset()

    def _restore_base_dataset(self) -> None:
        with self.app.app_context():
            engine = self.app_module.db.engine
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM tax_clause_assignments"))
                conn.execute(text("DELETE FROM clauses"))
                conn.execute(text("DELETE FROM tax_clause_taxonomy_l3"))
                conn.execute(text("DELETE FROM tax_clause_taxonomy_l2"))
                conn.execute(text("DELETE FROM tax_clause_taxonomy_l1"))
                conn.execute(text("DELETE FROM latest_sections_search_standard_ids"))
                conn.execute(text("DELETE FROM latest_sections_search"))
                conn.execute(text("DELETE FROM sections"))
                conn.execute(text("DELETE FROM agreement_counsel"))
                conn.execute(text("DELETE FROM counsel"))
                conn.execute(text("DELETE FROM reader_index"))
                conn.execute(text("DELETE FROM xml"))
                conn.execute(text("DELETE FROM agreements"))
                conn.execute(text("DELETE FROM agreement_filter_option_summary"))
                conn.execute(text("DELETE FROM agreement_metadata_field_coverage_summary"))
                conn.execute(text("DELETE FROM agreement_counsel_leaderboard_summary"))
                conn.execute(text("DELETE FROM agreement_index_summary"))
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
                        "target_counsel, acquirer_counsel, target_industry, acquirer_industry, announce_date, close_date, deal_status, "
                        "attitude, deal_type, purpose, target_pe, acquirer_pe, verified, url, "
                        "section_standard_ids, article_title, section_title"
                        ") VALUES ("
                        "'00000000-0000-0000-0000-000000000001', 'a1', '2020-01-01', NULL, NULL, NULL, "
                        "NULL, NULL, 'Target A', 'Acquirer A', '50000000', NULL, '50000000', NULL, 'cash', 'public', "
                        "'public', 'Wilson Sonsini Goodrich & Rosati, P.C.; Goodwin Procter LLP', 'Wiggin and Dana LLP', 'tech', 'tech', NULL, NULL, 'complete', 'friendly', 'merger', "
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
                conn.execute(text("DROP TABLE IF EXISTS agreement_status_summary"))
                conn.execute(text("DROP TABLE IF EXISTS agreement_overview_summary"))
                conn.execute(
                    text(
                        "INSERT INTO tax_clause_taxonomy_l1 (standard_id, label) VALUES "
                        "('tax_root', 'Tax')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO tax_clause_taxonomy_l2 (standard_id, label, parent_id) VALUES "
                        "('tax_operational', 'Operational Tax Matters', 'tax_root')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO tax_clause_taxonomy_l3 (standard_id, label, parent_id) VALUES "
                        "('tax_transfer', 'Transfer Taxes', 'tax_operational'), "
                        "('tax_treatment', 'Tax Treatment of the Transaction', 'tax_operational')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO clauses ("
                        "clause_uuid, agreement_uuid, section_uuid, xml_version, module, clause_order, "
                        "anchor_label, start_char, end_char, clause_text, source_method, context_type"
                        ") VALUES ("
                        "'clause-a1-1', 'a1', '00000000-0000-0000-0000-000000000001', 2, 'tax', 1, "
                        "'(a)', 0, 55, 'Parent shall bear all transfer taxes.', 'enumerated_split', 'operative'"
                        "), ("
                        "'clause-a1-2', 'a1', '00000000-0000-0000-0000-000000000001', 2, 'tax', 2, "
                        "'(b)', 56, 120, 'Company and Parent intend the merger to qualify as tax-free.', 'enumerated_split', 'rep_warranty'"
                        ")"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO tax_clause_assignments (clause_uuid, standard_id, is_gold_label, model_name, assigned_at) VALUES "
                        "('clause-a1-1', 'tax_transfer', 1, 'gpt-5-mini', '2026-04-02T00:00:00Z'), "
                        "('clause-a1-2', 'tax_treatment', 1, 'gpt-5-mini', '2026-04-02T00:00:00Z')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreement_filter_option_summary (field_name, option_value, agreement_count) VALUES "
                        "('targets', 'Target A', 1), "
                        "('targets', 'Target B', 1), "
                        "('targets', 'Target C', 1), "
                        "('acquirers', 'Acquirer A', 1), "
                        "('acquirers', 'Acquirer B', 1), "
                        "('acquirers', 'Acquirer C', 1), "
                        "('target_counsels', 'Goodwin Procter', 1), "
                        "('target_counsels', 'Wachtell, Lipton, Rosen & Katz', 1), "
                        "('target_counsels', 'Wilson Sonsini Goodrich & Rosati', 2), "
                        "('acquirer_counsels', 'Skadden, Arps, Slate, Meagher & Flom', 1), "
                        "('acquirer_counsels', 'Wiggin & Dana', 2), "
                        "('target_industries', 'energy', 1), "
                        "('target_industries', 'healthcare', 1), "
                        "('target_industries', 'tech', 1), "
                        "('acquirer_industries', 'industrial', 1), "
                        "('acquirer_industries', 'tech', 2)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreement_metadata_field_coverage_summary ("
                        "field_name, label, ingested_eligible_agreements, ingested_covered_agreements, "
                        "ingested_coverage_pct, processed_eligible_agreements, processed_covered_agreements, "
                        "processed_coverage_pct, note"
                        ") VALUES "
                        "('deal_type', 'Deal type', 3, 3, 100.0, 2, 2, 100.0, 'Expected for all eligible agreements.'), "
                        "('target_industry', 'Target industry', 3, 3, 100.0, 2, 2, 100.0, 'Optional in sourcing, but counted when present.')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreement_counsel_leaderboard_summary ("
                        "side, counsel_key, counsel, year, deal_count, total_transaction_value"
                        ") VALUES "
                        "('buy_side', 'wiggin dana', 'Wiggin & Dana', 2020, 1, 50000000), "
                        "('buy_side', 'wiggin dana', 'Wiggin & Dana', 2021, 1, 150000000), "
                        "('buy_side', 'skadden arps slate meagher flom', 'Skadden, Arps, Slate, Meagher & Flom', 2022, 1, 300000000), "
                        "('sell_side', 'wilson sonsini goodrich rosati', 'Wilson Sonsini Goodrich & Rosati', 2020, 1, 50000000), "
                        "('sell_side', 'wilson sonsini goodrich rosati', 'Wilson Sonsini Goodrich & Rosati', 2021, 1, 150000000), "
                        "('sell_side', 'goodwin procter', 'Goodwin Procter', 2020, 1, 50000000), "
                        "('sell_side', 'wachtell lipton rosen katz', 'Wachtell, Lipton, Rosen & Katz', 2022, 1, 300000000)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreement_index_summary (agreement_uuid, year, target, acquirer, verified) VALUES "
                        "('a2', 2021, 'Target B', 'Acquirer B', 0), "
                        "('a3', 2022, 'Target C', 'Acquirer C', 1)"
                    )
                )
            clear_all_ttl_caches()

    def test_agreements_index_pagination(self):
        client = self.app.test_client()
        res = client.get("/v1/agreements-index?page=1&page_size=2")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("total_count"), 2)
        self.assertEqual(body.get("total_pages"), 1)
        self.assertEqual(len(body.get("results", [])), 2)

    def test_agreements_index_includes_deal_terms(self):
        client = self.app.test_client()
        res = client.get("/v1/agreements-index?page=1&page_size=10&sort_by=year&sort_dir=asc")
        self.assertEqual(res.status_code, 200)
        rows = {row["agreement_uuid"]: row for row in res.get_json()["results"]}
        self.assertEqual(rows["a2"]["consideration_type"], "stock")
        self.assertEqual(rows["a2"]["total_consideration"], 150000000.0)
        self.assertEqual(rows["a3"]["consideration_type"], "assets")
        self.assertNotIn("target_industry", rows["a2"])

    def test_oversized_page_size_clamps_to_tier_max(self):
        client = self.app.test_client()
        with self.app.app_context():
            user = self.app_module.AuthUser()
            user.id = "00000000-0000-0000-0000-0000000000c1"
            user.email = "page-size-api@example.com"
            user.password_hash = "not-used"
            user.email_verified_at = datetime.now(timezone.utc).replace(tzinfo=None)
            self.app_module.db.session.add(user)
            self.app_module.db.session.commit()
            _, api_key = self.app_module._create_api_key(user_id=user.id, name="page-size")

        for path in ("/v1/agreements-index", "/v1/search/agreements", "/v1/sections"):
            anonymous = client.get(f"{path}?page=1&page_size=10000")
            self.assertEqual(anonymous.status_code, 200, path)
            self.assertEqual(anonymous.get_json()["page_size"], 10, path)
            keyed = client.get(f"{path}?page=1&page_size=10000", headers={"X-API-Key": api_key})
            self.assertEqual(keyed.status_code, 200, path)
            self.assertEqual(keyed.get_json()["page_size"], 100, path)

    def test_agreements_summary_includes_latest_filing_date(self):
        clear_all_ttl_caches()
        with self.app.app_context():
            engine = self.app_module.db.engine
            with engine.begin() as conn:
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
        res = client.get("/v1/agreements-summary")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("agreements"), 4)
        self.assertEqual(body.get("sections"), 3)
        self.assertEqual(body.get("pages"), 12)
        self.assertEqual(body.get("latest_filing_date"), "2023-04-01")

    def test_agreements_summary_includes_processed_agreements(self):
        clear_all_ttl_caches()
        client = self.app.test_client()
        res = client.get("/v1/agreements-summary")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        # Every eligible agreement ingested vs. the browsable rows in agreement_index_summary.
        self.assertEqual(body.get("agreements"), 4)
        self.assertEqual(body.get("processed_agreements"), 2)

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
            "&target_counsel=Wilson%20Sonsini%20Goodrich%20%26%20Rosati"
            "&acquirer_counsel=Wiggin%20%26%20Dana"
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
            "&target_counsel=Wilson%20Sonsini%20Goodrich%20%26%20Rosati"
            "&acquirer_counsel=Wiggin%20%26%20Dana"
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

    def test_search_agreements_returns_aggregated_matches(self):
        client = self.app.test_client()
        res = client.get("/v1/search/agreements?standard_id=s1&page=1&page_size=10")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("total_count"), 1)
        results = body.get("results", [])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get("agreement_uuid"), "a1")
        self.assertEqual(results[0].get("match_count"), 1)
        self.assertEqual(
            results[0].get("matched_sections", [])[0].get("section_uuid"),
            "00000000-0000-0000-0000-000000000001",
        )

    def test_search_agreements_section_uuid_filter(self):
        client = self.app.test_client()
        res = client.get(
            "/v1/search/agreements?section_uuid=00000000-0000-0000-0000-000000000001&page=1&page_size=10"
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        results = body.get("results", [])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get("agreement_uuid"), "a1")

    def test_list_agreement_sections_index(self):
        client = self.app.test_client()
        res = client.get("/v1/agreements/a1/sections")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("agreement_uuid"), "a1")
        results = body.get("results", [])
        self.assertEqual(results[0].get("section_uuid"), "00000000-0000-0000-0000-000000000001")
        self.assertEqual(results[0].get("standard_id"), ["s1"])

    def test_search_basic(self):
        client = self.app.test_client()
        res = client.get("/v1/sections?year=2020&page=1&page_size=10")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("total_count"), 1)
        self.assertFalse(body.get("total_count_is_approximate"))
        self.assertEqual(len(body.get("results", [])), 1)
        self.assertIn("access", body)
        self.assertEqual(
            body.get("results", [])[0].get("xml"),
            "<section>TEXT</section>",
        )

    def test_sections_include_xml_allowed_for_anonymous(self):
        client = self.app.test_client()
        res = client.get("/v1/sections?include_xml=true")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        results = body.get("results", [])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get("xml"), "<section>TEXT</section>")

    def test_sections_default_includes_xml_for_logged_in_user(self):
        client = self.app.test_client()
        with self.app.app_context():
            user = self.app_module.AuthUser()
            user.id = "00000000-0000-0000-0000-0000000000b2"
            user.email = "sections-user@example.com"
            user.password_hash = "not-used"
            user.email_verified_at = datetime.now(timezone.utc).replace(tzinfo=None)
            self.app_module.db.session.add(user)
            self.app_module.db.session.commit()
            with self.app.test_request_context():
                session_token = self.app_module._issue_session_token(user_id=user.id)

        res = client.get(
            "/v1/sections?page_size=1",
            headers={"Authorization": f"Bearer {session_token}"},
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        results = body.get("results", [])
        self.assertEqual(len(results), 1)
        self.assertIn("xml", results[0])
        self.assertEqual(results[0]["xml"], "<section>TEXT</section>")

    def test_sections_include_xml_with_api_key(self):
        client = self.app.test_client()
        with self.app.app_context():
            user = self.app_module.AuthUser()
            user.id = "00000000-0000-0000-0000-0000000000b1"
            user.email = "sections-api@example.com"
            user.password_hash = "not-used"
            user.email_verified_at = datetime.now(timezone.utc).replace(tzinfo=None)
            self.app_module.db.session.add(user)
            self.app_module.db.session.commit()
            _, api_key = self.app_module._create_api_key(user_id=user.id, name="sections")

        res = client.get(
            "/v1/sections?include_xml=true&page_size=1",
            headers={"X-API-Key": api_key},
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        results = body.get("results", [])
        self.assertEqual(len(results), 1)
        self.assertIn("xml", results[0])
        self.assertEqual(results[0]["xml"], "<section>TEXT</section>")

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

    # -- hot-path query guards ----------------------------------------------------------
    #
    # Production runs these searches against a 192MB buffer pool, so a plan that reads
    # latest_sections_search (1.1M rows, 2.5GB) once per row or per matching section
    # passes locally in a second and times out in production at the 55s statement cap.
    # These tests pin the shape and number of statements the hot search paths issue.

    def _capture_statements(self, path: str) -> tuple[object, list[str]]:
        from sqlalchemy import event

        statements: list[str] = []

        def _record(_conn, _cursor, statement, _params, _context, _executemany):
            statements.append(" ".join(str(statement).split()))

        with self.app.app_context():
            engine = self.app_module.db.engine
        event.listen(engine, "before_cursor_execute", _record)
        try:
            res = self.app.test_client().get(path)
        finally:
            event.remove(engine, "before_cursor_execute", _record)
        self.assertEqual(res.status_code, 200, res.get_json())
        return res.get_json(), statements

    def test_hot_search_paths_stay_within_statement_budget(self):
        # Every statement crosses regions on a machine away from the database
        # (~70ms each), so an added round trip on these paths is a regression.
        # TTL caches are off in tests; with them on, the ineligible-agreement
        # probe drops out too (see the next test).
        budgets = {
            "/v1/sections?page=1&page_size=25": 4,
            "/v1/sections?standard_id=s1&page=1&page_size=25": 7,
            "/v1/search/agreements?page=1&page_size=25": 4,
            "/v1/search/agreements?standard_id=s1&page=1&page_size=25": 8,
        }
        for path, budget in budgets.items():
            with self.subTest(path=path):
                _, statements = self._capture_statements(path)
                self.assertLessEqual(len(statements), budget, "\n".join(statements))

    def test_ineligible_agreement_probe_is_cached_between_section_searches(self):
        self.app.config["TTL_CACHE_IN_TESTS"] = True
        clear_all_ttl_caches()
        try:
            path = "/v1/sections?page=1&page_size=25"
            _, first = self._capture_statements(path)
            _, second = self._capture_statements(path)
        finally:
            self.app.config.pop("TTL_CACHE_IN_TESTS", None)
            clear_all_ttl_caches()
        self.assertTrue(any("gated" in s for s in first))
        self.assertFalse(any("gated" in s for s in second), "\n".join(second))

    def test_clause_only_section_count_reads_only_the_mapping(self):
        # Counting through latest_sections_search visits every one of its rows (a
        # correlated EXISTS per row, or a scan against a materialized IN list).
        body, statements = self._capture_statements(
            "/v1/sections?standard_id=s1&page=1&page_size=25"
        )
        counts = [s for s in statements if "count(" in s.lower()]
        self.assertTrue(counts)
        for statement in counts:
            self.assertRegex(statement, r"FROM (\w+\.)?latest_sections_search_standard_ids ")
            self.assertNotIn("latest_sections_search.", statement)
        self.assertEqual(body["total_count"], 1)

    def test_clause_only_section_count_is_distinct_and_exact_mode_counts_agreements(self):
        self.test_search_by_standard_id_matches_multi_label_sections()
        client = self.app.test_client()
        res = client.get("/v1/sections?standard_id=multi&standard_id=other&page=1&page_size=10")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["total_count"], 1)
        res = client.get(
            "/v1/sections?standard_id=multi&standard_id=other&count_mode=exact&page=1&page_size=10"
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body["total_count"], 1)
        self.assertEqual(body["total_agreement_count"], 1)

    def test_clause_deal_search_by_year_reads_only_the_mapping(self):
        body, statements = self._capture_statements(
            "/v1/search/agreements?standard_id=s1&page=1&page_size=25"
        )
        aggregate = [s for s in statements if "match_count" in s]
        self.assertTrue(aggregate)
        for statement in aggregate:
            self.assertRegex(statement, r"FROM (\w+\.)?latest_sections_search_standard_ids ")
            self.assertNotIn("latest_sections_search.", statement)
        by_target = self.app.test_client().get(
            "/v1/search/agreements?standard_id=s1&sort_by=target&page=1&page_size=25"
        ).get_json()
        self.assertEqual(body["total_count"], by_target["total_count"])
        self.assertEqual(
            {r["agreement_uuid"]: r["match_count"] for r in body["results"]},
            {r["agreement_uuid"]: r["match_count"] for r in by_target["results"]},
        )

    def test_search_returns_exact_total_count_for_filtered_first_page(self):
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
        self.assertFalse(body.get("total_count_is_approximate"))
        self.assertEqual(body.get("total_count"), 10)
        self.assertEqual(body.get("total_pages"), 2)

    def _seed_industry_agreements(self, *, industry: str, count: int, filing_dates=None):
        with self.app.app_context():
            engine = self.app_module.db.engine
            with engine.begin() as conn:
                for idx in range(count):
                    uuid = f"{industry}-{idx:02d}"
                    filing_date = (
                        filing_dates[idx] if filing_dates is not None else "2020-01-01"
                    )
                    conn.execute(
                        text(
                            "INSERT INTO agreements (agreement_uuid, filing_date, target, "
                            "acquirer, verified, url, target_industry) VALUES "
                            "(:u, :filing_date, :t, :a, 1, :url, :industry)"
                        ),
                        {
                            "u": uuid,
                            "filing_date": filing_date,
                            "t": f"Target {idx}",
                            "a": f"Acquirer {idx}",
                            "url": f"http://example.com/{uuid}",
                            "industry": industry,
                        },
                    )
                    conn.execute(
                        text(
                            "INSERT INTO xml (agreement_uuid, xml, version, status, latest) "
                            "VALUES (:u, '<document><article><section uuid=\"s\">"
                            "<text>Z</text></section></article></document>', 1, 'verified', 1)"
                        ),
                        {"u": uuid},
                    )

    def test_search_agreements_total_count_is_exact_and_stable_on_every_page(self):
        # The agreements grid must show the true total on every page, never the
        # "p to q of q+1" lower bound that past count rewrites regressed into.
        self._seed_industry_agreements(industry="zcount", count=10)
        client = self.app.test_client()

        seen_totals: list[int | None] = []
        for page in (1, 2, 3, 4):
            res = client.get(
                f"/v1/search/agreements?target_industry=zcount&page={page}&page_size=3"
            )
            self.assertEqual(res.status_code, 200)
            body = res.get_json()
            seen_totals.append(body.get("total_count"))
            self.assertFalse(body.get("total_count_is_approximate"))

        self.assertEqual(seen_totals, [10, 10, 10, 10])
        # Page 2 (items 4-6, more to come) must not collapse to upper-bound + 1.
        self.assertNotEqual(seen_totals[1], (2 - 1) * 3 + 3 + 1)

    def test_search_agreements_count_is_cached_across_pages(self):
        import backend.search_counts as search_counts

        self._seed_industry_agreements(industry="zcache", count=6)
        client = self.app.test_client()

        original = search_counts.exact_query_count
        with patch.object(
            search_counts, "exact_query_count", side_effect=original
        ) as spy:
            first = client.get(
                "/v1/search/agreements?target_industry=zcache&page=1&page_size=2"
            )
            second = client.get(
                "/v1/search/agreements?target_industry=zcache&page=2&page_size=2"
            )

        self.assertEqual(first.get_json().get("total_count"), 6)
        self.assertEqual(second.get_json().get("total_count"), 6)
        # One COUNT for both pages: the second page reuses the cached total.
        self.assertEqual(spy.call_count, 1)

    def test_search_agreements_count_mode_exact_bypasses_cache(self):
        import backend.search_counts as search_counts

        self._seed_industry_agreements(industry="zexact", count=4)
        client = self.app.test_client()

        original = search_counts.exact_query_count
        with patch.object(
            search_counts, "exact_query_count", side_effect=original
        ) as spy:
            client.get(
                "/v1/search/agreements"
                "?target_industry=zexact&page=1&page_size=2&count_mode=exact"
            )
            client.get(
                "/v1/search/agreements"
                "?target_industry=zexact&page=2&page_size=2&count_mode=exact"
            )

        # count_mode=exact is authoritative: it never serves a cached total.
        self.assertEqual(spy.call_count, 2)

    def test_search_agreements_date_range_filters(self):
        self._seed_industry_agreements(
            industry="zdate",
            count=3,
            filing_dates=["2010-01-01", "2011-01-01", "2012-01-01"],
        )
        client = self.app.test_client()

        def total(query: str) -> int:
            res = client.get(f"/v1/search/agreements?target_industry=zdate&{query}")
            self.assertEqual(res.status_code, 200)
            return res.get_json().get("total_count")

        self.assertEqual(total("year_min=2011"), 2)
        self.assertEqual(total("year_max=2010"), 1)
        self.assertEqual(total("filed_after=2011-06-01"), 1)
        self.assertEqual(total("filed_before=2011-06-01"), 2)
        self.assertEqual(total("year_min=2011&year_max=2011"), 1)

    def _sections_total_count(self, **kwargs: object) -> tuple[int, bool]:
        from backend.services.sections_service import sections_total_count_metadata

        total_count, is_approximate, _method = sections_total_count_metadata(
            self.app_module._build_sections_service_deps(),
            count_mode="auto",
            **kwargs,
        )
        return total_count, is_approximate

    def test_search_total_count_metadata_uses_table_estimate_for_unfiltered_search(self):
        with (
            patch.object(self.app_module, "_estimated_query_row_count", return_value=None),
            patch.object(
                self.app_module,
                "_estimated_latest_sections_search_table_rows",
                return_value=488296,
            ),
        ):
            total_count, is_approximate = self._sections_total_count(
                query=object(),
                page=6,
                page_size=25,
                item_count=25,
                has_next=True,
                has_filters=False,
            )

        self.assertEqual(total_count, 488296)
        self.assertTrue(is_approximate)

    def test_search_total_count_metadata_uses_query_estimate_for_filtered_search_beyond_page_one(self):
        with patch.object(self.app_module, "_estimated_query_row_count", return_value=240):
            total_count, is_approximate = self._sections_total_count(
                query=object(),
                page=6,
                page_size=25,
                item_count=25,
                has_next=True,
                has_filters=True,
            )

        self.assertEqual(total_count, 240)
        self.assertTrue(is_approximate)

    def test_search_total_count_metadata_uses_exact_count_for_filtered_first_page(self):
        query = MagicMock()
        ordered_query = MagicMock()
        query.order_by.return_value = ordered_query
        ordered_query.count.return_value = 37

        total_count, is_approximate = self._sections_total_count(
            query=query,
            page=1,
            page_size=25,
            item_count=25,
            has_next=True,
            has_filters=True,
        )

        query.order_by.assert_called_once_with(None)
        ordered_query.count.assert_called_once_with()
        self.assertEqual(total_count, 37)
        self.assertFalse(is_approximate)

    def test_search_total_count_metadata_uses_exact_count_when_estimate_is_too_small(self):
        query = MagicMock()
        ordered_query = MagicMock()
        query.order_by.return_value = ordered_query
        ordered_query.count.return_value = 163
        with patch.object(self.app_module, "_estimated_query_row_count", return_value=120):
            total_count, is_approximate = self._sections_total_count(
                query=query,
                page=6,
                page_size=25,
                item_count=25,
                has_next=True,
                has_filters=True,
            )

        self.assertEqual(total_count, 163)
        self.assertFalse(is_approximate)

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

    def test_sections_search_boots_and_serves_non_text_queries_without_text_index_table(self):
        with patch.object(self.app_module, "SectionTextSearch", None):
            app = self.app_module.create_test_app(
                config_overrides={
                    "MAIN_DB_SCHEMA": "",
                    "SQLALCHEMY_DATABASE_URI": self.app.config["SQLALCHEMY_DATABASE_URI"],
                    "SQLALCHEMY_BINDS": {
                        "auth": self.app.config["SQLALCHEMY_BINDS"]["auth"]["url"]
                    },
                }
            )
        client = app.test_client()

        res = client.get("/v1/sections?year=2020&page=1&page_size=10")
        self.assertEqual(res.status_code, 200)
        self.assertGreater(len(res.get_json()["results"]), 0)

        res = client.get("/v1/sections?text_query=material+adverse+effect&page=1&page_size=10")
        self.assertEqual(res.status_code, 422)
        self.assertEqual(
            res.get_json(),
            {
                "error": "validation_error",
                "message": "Section text search is not available on this server yet.",
            },
        )

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
                    conn.execute(
                        text(
                            "INSERT INTO agreement_index_summary (agreement_uuid, year, target, acquirer, verified) "
                            "VALUES ('a_index_time', 2024, 'Target Index', 'Acquirer Index', 1)"
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
                    conn.execute(text("DELETE FROM agreement_index_summary WHERE agreement_uuid = 'a_index_time'"))
                    conn.execute(text("DELETE FROM xml WHERE agreement_uuid = 'a_index_time'"))
                    conn.execute(text("DELETE FROM agreements WHERE agreement_uuid = 'a_index_time'"))

    def test_agreements_index_excludes_latest_xml_pending_verification(self):
        try:
            with self.app.app_context():
                engine = self.app_module.db.engine
                with engine.begin() as conn:
                    conn.execute(
                        text(
                            "INSERT INTO agreements (agreement_uuid, filing_date, target, acquirer, verified, url) "
                            "VALUES ('a_index_pending_xml', '2024-07-01', 'Target Pending XML', 'Acquirer Pending XML', 1, 'http://example.com/a_index_pending_xml')"
                        )
                    )
                    conn.execute(
                        text(
                            "INSERT INTO xml (agreement_uuid, xml, version, status, latest) VALUES "
                            "('a_index_pending_xml', '<document><article><section uuid=\"00000000-0000-0000-0000-000000000062\"><text>PENDING</text></section></article></document>', 1, NULL, 1)"
                        )
                    )

            client = self.app.test_client()
            res = client.get("/v1/agreements-index?query=Target Pending XML&page=1&page_size=10")
            self.assertEqual(res.status_code, 200)
            body = res.get_json()
            self.assertEqual(body.get("results", []), [])
            self.assertEqual(body.get("total_count"), 0)
        finally:
            with self.app.app_context():
                engine = self.app_module.db.engine
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM xml WHERE agreement_uuid = 'a_index_pending_xml'"))
                    conn.execute(text("DELETE FROM agreements WHERE agreement_uuid = 'a_index_pending_xml'"))

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

    _KEEP = "00000000-0000-0000-0000-000000000001"
    _HIDE = "00000000-0000-0000-0000-000000000003"

    def _seed_reader_index(self, *, xml_version: int = 2) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": 1,
            "enricher": "0.4.0",
            "agreement_uuid": "a1",
            "xml_version": xml_version,
            "outline": [
                {
                    "id": "s1.2",
                    "kind": "section",
                    "label": "1.2",
                    "title": "Definitions",
                    "anchor": {"s": self._HIDE},
                    "children": [
                        {"id": "s1.2(a)", "kind": "sub", "label": "1.2(a)", "title": "HIDE", "anchor": {"s": self._HIDE, "t": 0}}
                    ],
                }
            ],
            "terms": [
                {
                    "id": 0,
                    "term": "Company",
                    "decls": [{"anchor": {"s": self._HIDE, "t": 0}, "span": [0, 4], "kind": "paren", "in": "Section 1.2"}],
                    "confidence": "high",
                }
            ],
            "uses": {f"{self._KEEP}/0": [[0, 4, 0]], f"{self._HIDE}/0": [[0, 4, 0]]},
            "xrefs": [
                {"at": {"s": self._KEEP, "t": 0}, "span": [0, 4], "text": "Section 1.2", "status": "resolved", "target": "s1.2"},
                {"at": {"s": self._HIDE, "t": 0}, "span": [0, 4], "text": "Section 1.1", "status": "unresolved"},
            ],
            "stats": {"n_terms": 1},
        }
        raw = json.dumps(payload).encode("utf-8")
        with self.app.app_context():
            with self.app_module.db.engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO reader_index (agreement_uuid, xml_version, enricher_version, payload_gz) "
                        "VALUES ('a1', :v, '0.4.0', :p)"
                    ),
                    {"v": xml_version, "p": gzip.compress(raw)},
                )
        return payload

    def _api_key_headers(self) -> dict[str, str]:
        with self.app.app_context():
            user = self.app_module.AuthUser()
            user.id = "00000000-0000-0000-0000-0000000000b7"
            user.email = "reader-index@example.com"
            user.password_hash = "not-used"
            user.email_verified_at = datetime.now(timezone.utc).replace(tzinfo=None)
            self.app_module.db.session.add(user)
            self.app_module.db.session.commit()
            _, api_key = self.app_module._create_api_key(user_id=user.id, name="reader-index")
        return {"X-API-Key": api_key}

    def test_reader_index_404_without_a_row_for_the_served_version(self):
        client = self.app.test_client()
        self.assertEqual(client.get("/v1/agreements/a1/reader-index").status_code, 404)
        # a row for an older XML version is not served
        _ = self._seed_reader_index(xml_version=1)
        self.assertEqual(client.get("/v1/agreements/a1/reader-index").status_code, 404)
        self.assertEqual(client.get("/v1/agreements/missing/reader-index").status_code, 404)

    def test_reader_index_full_for_authenticated_callers(self):
        payload = self._seed_reader_index()
        client = self.app.test_client()
        headers = self._api_key_headers()
        res = client.get("/v1/agreements/a1/reader-index", headers=headers)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), payload)
        self.assertEqual(res.headers["Cache-Control"], "private, no-cache")
        etag = res.headers["ETag"]
        self.assertEqual(etag, 'W/"2-0.4.0"')

        gz = client.get("/v1/agreements/a1/reader-index", headers={**headers, "Accept-Encoding": "gzip"})
        self.assertEqual(gz.headers.get("Content-Encoding"), "gzip")
        self.assertEqual(json.loads(gzip.decompress(gz.get_data())), payload)

        cached = client.get("/v1/agreements/a1/reader-index", headers={**headers, "If-None-Match": etag})
        self.assertEqual(cached.status_code, 304)

    def test_reader_index_redacts_for_anonymous_callers(self):
        _ = self._seed_reader_index()
        client = self.app.test_client()
        res = client.get(f"/v1/agreements/a1/reader-index?focus_section_uuid={self._KEEP}&neighbor_sections=0")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertTrue(body["is_redacted"])
        self.assertEqual(body["terms"][0]["decls"], [{"redacted": True, "label": "Section 1.2"}])
        self.assertEqual(list(body["uses"]), [f"{self._KEEP}/0"])
        self.assertEqual([x["text"] for x in body["xrefs"]], ["Section 1.2"])
        self.assertEqual(body["outline"][0]["children"][0]["title"], "")
        self.assertNotIn("HIDE", res.get_data(as_text=True))
        # the redacted variant has its own validator
        self.assertNotEqual(res.headers["ETag"], 'W/"2-0.4.0"')

        wide = client.get(f"/v1/agreements/a1/reader-index?focus_section_uuid={self._KEEP}&neighbor_sections=1")
        self.assertEqual(len(wide.get_json()["xrefs"]), 2)
        self.assertEqual(client.get("/v1/agreements/a1/reader-index?neighbor_sections=9").status_code, 400)

    def test_public_agreement_endpoints_exclude_gated_unverified_agreements(self):
        with self.app.app_context():
            engine = self.app_module.db.engine
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO agreements (agreement_uuid, filing_date, target, acquirer, verified, gated, url, target_counsel, acquirer_counsel) "
                        "VALUES ('a_gated_hidden', '2025-06-01', 'Target Hidden', 'Acquirer Hidden', 0, 1, 'http://example.com/a_gated_hidden', 'Target Hidden Counsel LLP', 'Acquirer Hidden Counsel LLP')"
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
                conn.execute(
                    text(
                        "INSERT INTO clauses ("
                        "clause_uuid, agreement_uuid, section_uuid, xml_version, module, clause_order, "
                        "anchor_label, start_char, end_char, clause_text, source_method, context_type"
                        ") VALUES ("
                        "'clause-hidden-1', 'a_gated_hidden', '00000000-0000-0000-0000-000000000071', 1, 'tax', 1, "
                        "'(a)', 0, 12, 'Hidden tax clause', 'enumerated_split', 'operative'"
                        ")"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO tax_clause_assignments (clause_uuid, standard_id, is_gold_label, model_name, assigned_at) VALUES "
                        "('clause-hidden-1', 'tax_transfer', 1, 'gpt-5-mini', '2026-04-02T00:00:00Z')"
                    )
                )

        try:
            client = self.app.test_client()

            list_res = client.get("/v1/agreements?agreement_uuid=a_gated_hidden&page_size=10")
            self.assertEqual(list_res.status_code, 200)
            self.assertEqual(list_res.get_json().get("results", []), [])

            detail_res = client.get("/v1/agreements/a_gated_hidden")
            self.assertEqual(detail_res.status_code, 404)

            agreement_tax_res = client.get("/v1/agreements/a_gated_hidden/tax-clauses")
            self.assertEqual(agreement_tax_res.status_code, 200)
            self.assertEqual(agreement_tax_res.get_json(), {"clauses": []})

            section_tax_res = client.get(
                "/v1/sections/00000000-0000-0000-0000-000000000071/tax-clauses"
            )
            self.assertEqual(section_tax_res.status_code, 200)
            self.assertEqual(section_tax_res.get_json(), {"clauses": []})

            tax_search_res = client.get("/v1/tax-clauses?agreement_uuid=a_gated_hidden&page=1&page_size=10")
            self.assertEqual(tax_search_res.status_code, 200)
            self.assertEqual(tax_search_res.get_json().get("results", []), [])

            index_res = client.get("/v1/agreements-index?query=Target Hidden&page=1&page_size=10")
            self.assertEqual(index_res.status_code, 200)
            self.assertEqual(index_res.get_json().get("results", []), [])

            filters_res = client.get("/v1/filter-options")
            self.assertEqual(filters_res.status_code, 200)
            body = filters_res.get_json()
            self.assertNotIn("Target Hidden", body.get("targets", []))
            self.assertNotIn("Acquirer Hidden", body.get("acquirers", []))
            self.assertNotIn("Target Hidden Counsel LLP", body.get("target_counsels", []))
            self.assertNotIn("Acquirer Hidden Counsel LLP", body.get("acquirer_counsels", []))
        finally:
            with self.app.app_context():
                engine = self.app_module.db.engine
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM tax_clause_assignments WHERE clause_uuid = 'clause-hidden-1'"))
                    conn.execute(text("DELETE FROM clauses WHERE agreement_uuid = 'a_gated_hidden'"))
                    conn.execute(text("DELETE FROM sections WHERE agreement_uuid = 'a_gated_hidden'"))
                    conn.execute(text("DELETE FROM xml WHERE agreement_uuid = 'a_gated_hidden'"))
                    conn.execute(text("DELETE FROM agreements WHERE agreement_uuid = 'a_gated_hidden'"))
                clear_all_ttl_caches()

    def test_section_search_excludes_gated_unverified_rows_in_search_table(self):
        # latest_sections_search should never hold these rows; the search still drops
        # them if it does, while gated-but-verified agreements stay visible.
        hidden_section = "00000000-0000-0000-0000-000000000081"
        visible_section = "00000000-0000-0000-0000-000000000082"
        seeded = (
            ("a_lss_gated_hidden", hidden_section, 0),
            ("a_lss_gated_verified", visible_section, 1),
        )
        with self.app.app_context():
            engine = self.app_module.db.engine
            with engine.begin() as conn:
                for agreement_uuid, section_uuid, verified in seeded:
                    conn.execute(
                        text(
                            "INSERT INTO agreements (agreement_uuid, filing_date, target, acquirer, "
                            "verified, gated, url) VALUES "
                            "(:a, '2025-06-01', 'Target Gate', 'Acquirer Gate', :v, 1, 'http://example.com/gate')"
                        ),
                        {"a": agreement_uuid, "v": verified},
                    )
                    conn.execute(
                        text(
                            "INSERT INTO sections (agreement_uuid, section_uuid, article_title, section_title, "
                            "xml_content, section_standard_id, xml_version) VALUES "
                            "(:a, :s, 'ARTICLE I', 'Gate Section', '<section>GATE</section>', '[\"s-gate\"]', 1)"
                        ),
                        {"a": agreement_uuid, "s": section_uuid},
                    )
                    conn.execute(
                        text(
                            "INSERT INTO latest_sections_search ("
                            "section_uuid, agreement_uuid, filing_date, target, acquirer, verified, url, "
                            "section_standard_ids, article_title, section_title"
                            ") VALUES ("
                            ":s, :a, '2025-06-01', 'Target Gate', 'Acquirer Gate', :v, "
                            "'http://example.com/gate', '[\"s-gate\"]', 'ARTICLE I', 'Gate Section')"
                        ),
                        {"a": agreement_uuid, "s": section_uuid, "v": verified},
                    )
                    conn.execute(
                        text(
                            "INSERT INTO latest_sections_search_standard_ids "
                            "(standard_id, section_uuid, agreement_uuid) VALUES ('s-gate', :s, :a)"
                        ),
                        {"a": agreement_uuid, "s": section_uuid},
                    )

        try:
            client = self.app.test_client()
            res = client.get("/v1/sections?target=Target Gate&page=1&page_size=10")
            self.assertEqual(res.status_code, 200)
            body = res.get_json()
            returned = [row["section_uuid"] for row in body.get("results", [])]
            self.assertEqual(returned, [visible_section])
            self.assertEqual(body.get("total_count"), 1)
            # The clause-only paths count from the mapping; they must drop it too.
            for count_mode in ("auto", "exact"):
                with self.subTest(count_mode=count_mode):
                    res = client.get(
                        f"/v1/sections?standard_id=s-gate&count_mode={count_mode}&page=1&page_size=10"
                    )
                    self.assertEqual(res.status_code, 200)
                    body = res.get_json()
                    self.assertEqual(
                        [row["section_uuid"] for row in body["results"]], [visible_section]
                    )
                    self.assertEqual(body["total_count"], 1)
            res = client.get("/v1/search/agreements?standard_id=s-gate&page=1&page_size=10")
            self.assertEqual(res.status_code, 200)
            body = res.get_json()
            self.assertEqual(
                [row["agreement_uuid"] for row in body["results"]], ["a_lss_gated_verified"]
            )
            self.assertEqual(body["total_count"], 1)
        finally:
            with self.app.app_context():
                engine = self.app_module.db.engine
                with engine.begin() as conn:
                    for agreement_uuid, _, _ in seeded:
                        params = {"a": agreement_uuid}
                        conn.execute(
                            text("DELETE FROM latest_sections_search_standard_ids WHERE agreement_uuid = :a"),
                            params,
                        )
                        conn.execute(text("DELETE FROM latest_sections_search WHERE agreement_uuid = :a"), params)
                        conn.execute(text("DELETE FROM sections WHERE agreement_uuid = :a"), params)
                        conn.execute(text("DELETE FROM agreements WHERE agreement_uuid = :a"), params)

    def test_filter_options_include_counsel_lists(self):
        clear_all_ttl_caches()
        client = self.app.test_client()
        res = client.get("/v1/filter-options")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertIn(
            "Wilson Sonsini Goodrich & Rosati",
            body.get("target_counsels", []),
        )
        self.assertIn("Goodwin Procter", body.get("target_counsels", []))
        self.assertIn("Wiggin & Dana", body.get("acquirer_counsels", []))
        self.assertIn("clause_types", body)
        self.assertIsInstance(body.get("clause_types"), dict)

    def test_filter_options_can_limit_fields(self):
        clear_all_ttl_caches()
        client = self.app.test_client()
        res = client.get(
            "/v1/filter-options?fields=target_counsels&fields=acquirer_industries"
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(
            set(body.keys()),
            {"target_counsels", "acquirer_industries", "industry_labels"},
        )
        # "tech" is not a NAICS code, so it gets no label.
        self.assertEqual(body["industry_labels"], {})
        self.assertIn("Goodwin Procter", body.get("target_counsels", []))
        self.assertIn("tech", body.get("acquirer_industries", []))

    def test_filter_options_can_request_clause_types(self):
        clear_all_ttl_caches()
        client = self.app.test_client()
        res = client.get("/v1/filter-options?fields=clause_types")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(set(body.keys()), {"clause_types"})
        self.assertIsInstance(body.get("clause_types"), dict)

    def test_filter_option_values_endpoint_searches_targets(self):
        client = self.app.test_client()
        res = client.get("/v1/filter-options/target?query=Target%20A&limit=5")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertIn("Target A", body.get("options", []))
        self.assertLessEqual(len(body.get("options", [])), 5)
        hidden_res = client.get("/v1/filter-options/target?query=Target%20Hidden")
        self.assertEqual(hidden_res.status_code, 200)
        self.assertEqual(hidden_res.get_json().get("options", []), [])

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
        self.assertEqual(
            buy_side.get("annual", [])[0],
            {
                "year": 2022,
                "top_by_count": [
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
                    }
                ],
                "top_by_value": [
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
                    }
                ],
            },
        )

    def test_agreements_status_summary_includes_overview_metrics(self):
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
                conn.execute(text("DELETE FROM agreements"))
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
                        "(1, 123, 61.5, 4567, 87.2, '2023-04-01')"
                    )
                )
                conn.execute(text("DELETE FROM agreement_metadata_field_coverage_summary"))
                conn.execute(
                    text(
                        "INSERT INTO agreement_metadata_field_coverage_summary ("
                        "field_name, label, ingested_eligible_agreements, ingested_covered_agreements, "
                        "ingested_coverage_pct, processed_eligible_agreements, processed_covered_agreements, "
                        "processed_coverage_pct, note"
                        ") VALUES "
                        "('transaction_price_cash', 'Cash price', 2, 1, 50.0, 1, 1, 100.0, 'Only applies to cash or mixed deals.'), "
                        "('transaction_price_stock', 'Stock price', 1, 1, 100.0, 1, 1, 100.0, 'Only applies to stock or mixed deals.'), "
                        "('transaction_price_assets', 'Asset price', 0, 0, NULL, 0, 0, NULL, 'Shown only against mixed deals; null can still be valid when no asset component exists.'), "
                        "('target_counsel', 'Target counsel', 3, 3, 100.0, 2, 2, 100.0, 'Optional in sourcing, but counted when present.'), "
                        "('acquirer_counsel', 'Acquirer counsel', 3, 2, 66.7, 2, 1, 50.0, 'Optional in sourcing, but counted when present.'), "
                        "('target_pe', 'Target PE', 3, 1, 33.3, 2, 1, 50.0, 'Optional in sourcing, but counted when present.'), "
                        "('purpose', 'Purpose', 3, 1, 33.3, 2, 1, 50.0, 'Optional in sourcing, but counted when present.')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreements ("
                        "agreement_uuid, filing_date, target, acquirer, verified, gated, url, "
                        "transaction_consideration, transaction_price_total, transaction_price_cash, "
                        "transaction_price_stock, transaction_price_assets, target_type, acquirer_type, "
                        "target_counsel, acquirer_counsel, target_pe, acquirer_pe, "
                        "target_industry, acquirer_industry, announce_date, "
                        "close_date, deal_status, attitude, deal_type, purpose"
                        ") VALUES "
                        "('cash_complete', '2023-04-01', 'Target Cash', 'Acquirer Cash', 1, 0, 'http://example.com/cash_complete', "
                        "'cash', '100', '100', NULL, NULL, 'public', 'private', "
                        "'Target Counsel LLP', 'Acquirer Counsel LLP', 1, 0, '52', '51', "
                        "'2023-01-01', '2023-03-01', 'complete', 'friendly', 'merger', 'strategic'), "
                        "('stock_missing_optional', '2023-04-02', 'Target Stock', 'Acquirer Stock', 1, 0, 'http://example.com/stock_missing_optional', "
                        "'stock', '250', NULL, '250', NULL, 'private', 'public', "
                        "'Target Counsel LLP', NULL, NULL, NULL, NULL, '54', "
                        "'2023-02-01', NULL, 'pending', NULL, 'stock_acquisition', NULL), "
                        "('cash_ingested_only', '2023-04-03', 'Target Pending', 'Acquirer Pending', 0, 0, 'http://example.com/cash_ingested_only', "
                        "'cash', '175', NULL, NULL, NULL, 'public', 'private', "
                        "'Target Counsel LLP', 'Acquirer Counsel LLP', NULL, NULL, '31', '32', "
                        "'2023-02-15', NULL, 'pending', NULL, 'merger', NULL)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO xml (agreement_uuid, xml, version, status, latest) VALUES "
                        "('cash_complete', '<document><article><section uuid=\"10000000-0000-0000-0000-000000000001\"><text>CASH</text></section></article></document>', 1, 'verified', 1), "
                        "('stock_missing_optional', '<document><article><section uuid=\"10000000-0000-0000-0000-000000000002\"><text>STOCK</text></section></article></document>', 1, 'verified', 1)"
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
        self.assertEqual(
            [row["field"] for row in body.get("metadata_field_coverage", [])],
            [
                "transaction_price_cash",
                "transaction_price_stock",
                "transaction_price_assets",
                "target_counsel",
                "acquirer_counsel",
                "target_pe",
                "purpose",
            ],
        )
        metadata_field_coverage = {
            row["field"]: row for row in body.get("metadata_field_coverage", [])
        }
        self.assertEqual(
            metadata_field_coverage["transaction_price_cash"]["ingested_eligible_agreements"], 2
        )
        self.assertEqual(
            metadata_field_coverage["transaction_price_cash"]["ingested_covered_agreements"], 1
        )
        self.assertEqual(metadata_field_coverage["transaction_price_cash"]["ingested_coverage_pct"], 50.0)
        self.assertEqual(
            metadata_field_coverage["transaction_price_cash"]["processed_eligible_agreements"], 1
        )
        self.assertEqual(
            metadata_field_coverage["transaction_price_cash"]["processed_covered_agreements"], 1
        )
        self.assertEqual(metadata_field_coverage["transaction_price_cash"]["processed_coverage_pct"], 100.0)
        self.assertEqual(
            metadata_field_coverage["transaction_price_stock"]["ingested_eligible_agreements"], 1
        )
        self.assertEqual(
            metadata_field_coverage["transaction_price_stock"]["ingested_covered_agreements"], 1
        )
        self.assertEqual(metadata_field_coverage["transaction_price_stock"]["ingested_coverage_pct"], 100.0)
        self.assertEqual(
            metadata_field_coverage["transaction_price_stock"]["processed_eligible_agreements"], 1
        )
        self.assertEqual(
            metadata_field_coverage["transaction_price_stock"]["processed_covered_agreements"], 1
        )
        self.assertEqual(metadata_field_coverage["transaction_price_stock"]["processed_coverage_pct"], 100.0)
        self.assertEqual(
            metadata_field_coverage["transaction_price_assets"]["ingested_eligible_agreements"], 0
        )
        self.assertEqual(
            metadata_field_coverage["transaction_price_assets"]["ingested_covered_agreements"], 0
        )
        self.assertIsNone(metadata_field_coverage["transaction_price_assets"]["ingested_coverage_pct"])
        self.assertEqual(
            metadata_field_coverage["transaction_price_assets"]["processed_eligible_agreements"], 0
        )
        self.assertEqual(
            metadata_field_coverage["transaction_price_assets"]["processed_covered_agreements"], 0
        )
        self.assertIsNone(metadata_field_coverage["transaction_price_assets"]["processed_coverage_pct"])
        self.assertEqual(metadata_field_coverage["target_counsel"]["ingested_covered_agreements"], 3)
        self.assertEqual(metadata_field_coverage["target_counsel"]["ingested_coverage_pct"], 100.0)
        self.assertEqual(metadata_field_coverage["target_counsel"]["processed_covered_agreements"], 2)
        self.assertEqual(metadata_field_coverage["target_counsel"]["processed_coverage_pct"], 100.0)
        self.assertEqual(metadata_field_coverage["acquirer_counsel"]["ingested_covered_agreements"], 2)
        self.assertEqual(metadata_field_coverage["acquirer_counsel"]["ingested_coverage_pct"], 66.7)
        self.assertEqual(metadata_field_coverage["acquirer_counsel"]["processed_covered_agreements"], 1)
        self.assertEqual(metadata_field_coverage["acquirer_counsel"]["processed_coverage_pct"], 50.0)
        self.assertEqual(metadata_field_coverage["target_pe"]["processed_covered_agreements"], 1)
        self.assertEqual(metadata_field_coverage["purpose"]["processed_covered_agreements"], 1)

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
                    conn.execute(text("DELETE FROM agreement_metadata_field_coverage_summary"))
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

    def test_tax_clause_taxonomy_hierarchy(self):
        client = self.app.test_client()
        res = client.get("/v1/taxonomy/tax-clauses")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(
            body,
            {
                "Tax": {
                    "id": "tax_root",
                    "children": {
                        "Operational Tax Matters": {
                            "id": "tax_operational",
                            "children": {
                                "Tax Treatment of the Transaction": {"id": "tax_treatment"},
                                "Transfer Taxes": {"id": "tax_transfer"},
                            },
                        }
                    },
                }
            },
        )

    def test_tax_clauses_search_default_excludes_rep_warranty(self):
        client = self.app.test_client()
        res = client.get("/v1/tax-clauses?page=1&page_size=10")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        results = body.get("results", [])
        clause_uuids = [r["clause_uuid"] for r in results]
        self.assertIn("clause-a1-1", clause_uuids)
        self.assertNotIn("clause-a1-2", clause_uuids)
        first = results[0]
        self.assertEqual(first["agreement_uuid"], "a1")
        self.assertEqual(first["target"], "Target A")
        self.assertEqual(first["acquirer"], "Acquirer A")
        self.assertEqual(first["year"], 2020)
        self.assertEqual(first["context_type"], "operative")
        self.assertEqual(first["tax_standard_ids"], ["tax_transfer"])

    def test_tax_clauses_search_include_rep_warranty(self):
        client = self.app.test_client()
        res = client.get("/v1/tax-clauses?include_rep_warranty=true&page=1&page_size=10")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        results = body.get("results", [])
        clause_uuids = {r["clause_uuid"] for r in results}
        self.assertIn("clause-a1-1", clause_uuids)
        self.assertIn("clause-a1-2", clause_uuids)

    def test_tax_clauses_search_by_tax_standard_id_expands_parent(self):
        client = self.app.test_client()
        res = client.get(
            "/v1/tax-clauses?tax_standard_id=tax_operational&include_rep_warranty=true&page=1&page_size=10"
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        clause_uuids = {r["clause_uuid"] for r in body.get("results", [])}
        self.assertEqual(clause_uuids, {"clause-a1-1", "clause-a1-2"})

    def test_tax_clauses_search_filters_by_deal_metadata(self):
        client = self.app.test_client()
        res = client.get("/v1/tax-clauses?year=2020&target=Target%20A&page=1&page_size=10")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("total_count"), 1)
        res_no_match = client.get("/v1/tax-clauses?year=2099&page=1&page_size=10")
        self.assertEqual(res_no_match.status_code, 200)
        self.assertEqual(res_no_match.get_json().get("total_count"), 0)

    def test_tax_clauses_search_excludes_non_latest_xml_versions(self):
        with self.app.app_context():
            engine = self.app_module.db.engine
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO clauses ("
                        "clause_uuid, agreement_uuid, section_uuid, xml_version, module, clause_order, "
                        "anchor_label, start_char, end_char, clause_text, source_method, context_type"
                        ") VALUES ("
                        "'clause-a4-stale', 'a4', '00000000-0000-0000-0000-000000000041', 1, 'tax', 1, "
                        "'(a)', 0, 18, 'Stale verified clause', 'enumerated_split', 'operative'"
                        ")"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO tax_clause_assignments (clause_uuid, standard_id, is_gold_label, model_name, assigned_at) VALUES "
                        "('clause-a4-stale', 'tax_transfer', 1, 'gpt-5-mini', '2026-04-02T00:00:00Z')"
                    )
                )

        client = self.app.test_client()
        res = client.get("/v1/tax-clauses?agreement_uuid=a4&page=1&page_size=10")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        clause_uuids = {row["clause_uuid"] for row in body.get("results", [])}
        self.assertNotIn("clause-a4-stale", clause_uuids)

    def test_agreement_tax_clauses_endpoint(self):
        client = self.app.test_client()
        res = client.get("/v1/agreements/a1/tax-clauses")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(
            body,
            {
                "clauses": [
                    {
                        "clause_uuid": "clause-a1-1",
                        "agreement_uuid": "a1",
                        "section_uuid": "00000000-0000-0000-0000-000000000001",
                        "article_title": "ARTICLE I",
                        "section_title": "Section 1",
                        "anchor_label": "(a)",
                        "start_char": 0,
                        "end_char": 55,
                        "clause_text": "Parent shall bear all transfer taxes.",
                        "context_type": "operative",
                        "standard_ids": ["tax_transfer"],
                        "standard_id_labels": {"tax_transfer": "Transfer Taxes"},
                    },
                    {
                        "clause_uuid": "clause-a1-2",
                        "agreement_uuid": "a1",
                        "section_uuid": "00000000-0000-0000-0000-000000000001",
                        "article_title": "ARTICLE I",
                        "section_title": "Section 1",
                        "anchor_label": "(b)",
                        "start_char": 56,
                        "end_char": 120,
                        "clause_text": "Company and Parent intend the merger to qualify as tax-free.",
                        "context_type": "rep_warranty",
                        "standard_ids": ["tax_treatment"],
                        "standard_id_labels": {
                            "tax_treatment": "Tax Treatment of the Transaction"
                        },
                    },
                ]
            },
        )

    def test_section_tax_clauses_endpoint(self):
        client = self.app.test_client()
        res = client.get("/v1/sections/00000000-0000-0000-0000-000000000001/tax-clauses")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(len(body.get("clauses", [])), 2)
        self.assertEqual(body["clauses"][0]["clause_uuid"], "clause-a1-1")
        self.assertEqual(body["clauses"][1]["standard_ids"], ["tax_treatment"])

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
        self.assertEqual(
            ownership.get("deal_size_by_year", []),
            [
                {
                    "year": 2020,
                    "public_deal_count": 1,
                    "private_deal_count": 0,
                    "public_p25_transaction_value": 50000000.0,
                    "public_median_transaction_value": 50000000.0,
                    "public_p75_transaction_value": 50000000.0,
                    "private_p25_transaction_value": None,
                    "private_median_transaction_value": None,
                    "private_p75_transaction_value": None,
                },
                {
                    "year": 2021,
                    "public_deal_count": 0,
                    "private_deal_count": 1,
                    "public_p25_transaction_value": None,
                    "public_median_transaction_value": None,
                    "public_p75_transaction_value": None,
                    "private_p25_transaction_value": 150000000.0,
                    "private_median_transaction_value": 150000000.0,
                    "private_p75_transaction_value": 150000000.0,
                },
                {
                    "year": 2022,
                    "public_deal_count": 0,
                    "private_deal_count": 1,
                    "public_p25_transaction_value": None,
                    "public_median_transaction_value": None,
                    "public_p75_transaction_value": None,
                    "private_p25_transaction_value": 300000000.0,
                    "private_median_transaction_value": 300000000.0,
                    "private_p75_transaction_value": 300000000.0,
                },
            ],
        )

        industries = body.get("industries", {})
        by_year = industries.get("target_industries_by_year", [])
        self.assertIn(
            {
                "year": 2020,
                "industry": "Crop Production",
                "deal_count": 1,
                "total_transaction_value": 50000000.0,
            },
            by_year,
        )
        self.assertIn(
            {
                "target_industry": "Mining, Quarrying, and Oil and Gas Extraction",
                "acquirer_industry": "Crop Production",
                "deal_count": 1,
                "total_transaction_value": 150000000.0,
            },
            industries.get("pairings", []),
        )


    # -- Unified agreement filter chain ------------------------------------------------

    def _agreement_uuids(self, path: str) -> list[str]:
        res = self.app.test_client().get(path)
        self.assertEqual(res.status_code, 200, res.get_json())
        return sorted(item["agreement_uuid"] for item in res.get_json().get("results", []))

    def test_list_agreements_accepts_year_and_filing_date_ranges(self):
        # a4's latest XML is invalid, so it never appears; a1/a2/a3 filed 2020/2021/2022.
        self.assertEqual(self._agreement_uuids("/v1/agreements?year_min=2021&page_size=10"), ["a2", "a3"])
        self.assertEqual(self._agreement_uuids("/v1/agreements?year_max=2020&page_size=10"), ["a1"])
        # filed_after is inclusive and filed_before exclusive, as on /v1/sections.
        self.assertEqual(
            self._agreement_uuids(
                "/v1/agreements?filed_after=2021-02-01&filed_before=2022-03-01&page_size=10"
            ),
            ["a2"],
        )
        res = self.app.test_client().get("/v1/agreements?filed_after=2021/02/01")
        self.assertEqual(res.status_code, 422)

    def test_agreement_filters_agree_across_list_and_search_routes(self):
        cases = [
            "year=2021",
            "year_min=2021&year_max=2022",
            "filed_after=2020-06-01",
            "target_type=private",
            "deal_status=complete&deal_status=pending",
            "target_pe=true",
            "transaction_price_total=100M - 250M",
            "target_counsel=Wilson Sonsini Goodrich %26 Rosati",
            "target_industry=tech&acquirer_industry=tech",
            "transaction_consideration=assets",
        ]
        for case in cases:
            with self.subTest(case=case):
                listed = self._agreement_uuids(f"/v1/agreements?{case}&page_size=50")
                searched = self._agreement_uuids(f"/v1/search/agreements?{case}&page_size=50")
                self.assertEqual(listed, searched)
                self.assertTrue(listed, f"case {case!r} should match at least one agreement")

    def test_unknown_price_bucket_is_rejected_with_valid_values(self):
        from backend.filtering import TRANSACTION_PRICE_BUCKETS

        client = self.app.test_client()
        paths = ("/v1/sections", "/v1/search/agreements", "/v1/agreements", "/v1/tax-clauses")
        fields_ = (
            "transaction_price_total",
            "transaction_price_stock",
            "transaction_price_cash",
            "transaction_price_assets",
        )
        for path in paths:
            for field in fields_:
                with self.subTest(path=path, field=field):
                    res = client.get(f"{path}?{field}=100M - 250M&{field}=1B - 5B&page_size=10")
                    self.assertEqual(res.status_code, 422)
                    body = res.get_json()
                    self.assertEqual(body["error"], "validation_error")
                    message = body["message"]
                    self.assertTrue(message.startswith(f"{field}[1]: "), message)
                    self.assertIn("'1B - 5B'", message)
                    self.assertNotIn("'100M - 250M'", message)
                    for bucket in TRANSACTION_PRICE_BUCKETS:
                        self.assertIn(bucket, message)
            with self.subTest(path=path, valid=True):
                res = client.get(f"{path}?transaction_price_total=1B - 2B&page_size=10")
                self.assertEqual(res.status_code, 200, res.get_json())

    def test_agreement_filters_agree_between_sections_and_agreement_search(self):
        # Only a1 has a latest section in the search index, so the section route and the
        # clause-scoped agreement search must both keep or both drop it.
        for case, expected in (("year=2020", ["a1"]), ("year=2021", []), ("target_pe=false", ["a1"])):
            with self.subTest(case=case):
                res = self.app.test_client().get(f"/v1/sections?{case}&page_size=50")
                self.assertEqual(res.status_code, 200)
                section_agreements = sorted(
                    {item["agreement_uuid"] for item in res.get_json().get("results", [])}
                )
                searched = self._agreement_uuids(
                    f"/v1/search/agreements?{case}&standard_id=s1&page_size=50"
                )
                self.assertEqual(section_agreements, expected)
                self.assertEqual(searched, expected)

    def test_agreement_filter_predicates_are_empty_without_filters(self):
        from backend.filtering import agreement_filter_predicates, has_agreement_filters

        with self.app.app_context():
            agreements = self.app_module.Agreements
            empty = agreement_filter_predicates(
                agreements,
                {"year": [], "target": [], "year_min": None, "filed_after": "  "},
                agreement_counsel=self.app_module.AgreementCounsel,
                counsel=self.app_module.Counsel,
            )
            self.assertEqual(empty, [])
            self.assertFalse(has_agreement_filters({"year": [], "filed_before": None}))
            self.assertTrue(has_agreement_filters({"year_min": 2020}))
            self.assertTrue(has_agreement_filters({"target_pe": ["false"]}))
            predicates = agreement_filter_predicates(
                agreements,
                {"year": [2020, 2021], "filed_after": "2020-01-01", "target_pe": ["true"]},
                agreement_counsel=self.app_module.AgreementCounsel,
                counsel=self.app_module.Counsel,
            )
            self.assertEqual(len(predicates), 3)

    # -- /v1/search/agreements -----------------------------------------------------------

    def test_search_agreements_rejects_metadata(self):
        res = self.app.test_client().get("/v1/search/agreements?metadata=deal_type")
        self.assertEqual(res.status_code, 422)
        self.assertIn("metadata", json.dumps(res.get_json()))

    def test_search_agreements_direct_path_reports_dump_version(self):
        dump = {"hash": "abc123", "dump_ts": "2026-10-01T00:00:00Z"}
        with patch("backend.routes.agreements.g", SimpleNamespace(dump_version=dump)):
            client = self.app.test_client()
            with_dump = client.get("/v1/search/agreements?year=2020").get_json()
            without_dump = client.get("/v1/search/agreements?year=2020&include_dump=false").get_json()
        self.assertEqual(with_dump.get("dump_version"), dump)
        self.assertNotIn("dump_version", without_dump)
        self.assertIn("count_metadata", with_dump)

    def _seed_text_search_rows(self) -> None:
        with self.app.app_context():
            with self.app_module.db.engine.begin() as conn:
                conn.execute(
                    text(
                        "UPDATE sections SET xml_content = '<section><text>Parent shall pay the "
                        "Reverse Termination Fee upon any termination.</text></section>', "
                        "article_order = 1, section_order = 1 "
                        "WHERE section_uuid = '00000000-0000-0000-0000-000000000001'"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreements (agreement_uuid, filing_date, target, acquirer, verified, gated, url) "
                        "VALUES ('a_text_gated', '2021-06-01', 'Gated Target', 'Gated Acquirer', 0, 1, 'http://example.com/g')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO xml (agreement_uuid, xml, version, status, latest) VALUES "
                        "('a_text_gated', '<document/>', 1, 'verified', 1)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO sections (agreement_uuid, section_uuid, article_title, section_title, "
                        "xml_content, section_standard_id, xml_version, article_order, section_order) VALUES "
                        "('a2', '00000000-0000-0000-0000-000000000011', 'ARTICLE VIII', 'Termination Fee', "
                        "'<section><text>The Company shall pay a termination fee.</text></section>', '[\"s1\"]', 1, 8, 2), "
                        "('a_text_gated', '00000000-0000-0000-0000-000000000081', 'ARTICLE VIII', 'Fee', "
                        "'<section><text>Reverse termination fee.</text></section>', '[\"s1\"]', 1, 8, 1)"
                    )
                )
                for agreement_uuid, section_uuid, filing_date, title in (
                    ("a2", "00000000-0000-0000-0000-000000000011", "2021-02-01", "Termination Fee"),
                    ("a_text_gated", "00000000-0000-0000-0000-000000000081", "2021-06-01", "Fee"),
                ):
                    conn.execute(
                        text(
                            "INSERT INTO latest_sections_search (section_uuid, agreement_uuid, filing_date, "
                            "target, acquirer, verified, url, section_standard_ids, article_title, section_title) "
                            "VALUES (:section_uuid, :agreement_uuid, :filing_date, 'T', 'A', 0, 'http://x', "
                            "'[\"s1\"]', 'ARTICLE VIII', :title)"
                        ),
                        {
                            "section_uuid": section_uuid,
                            "agreement_uuid": agreement_uuid,
                            "filing_date": filing_date,
                            "title": title,
                        },
                    )
                conn.execute(
                    text(
                        "INSERT INTO latest_sections_search_standard_ids (standard_id, section_uuid, agreement_uuid) "
                        "VALUES ('s1', '00000000-0000-0000-0000-000000000011', 'a2'), "
                        "('s1', '00000000-0000-0000-0000-000000000081', 'a_text_gated')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO section_text_search (section_uuid, agreement_uuid, xml_version, "
                        "normalized_text, updated_at) VALUES "
                        "('00000000-0000-0000-0000-000000000001', 'a1', 2, "
                        "'parent shall pay the reverse termination fee upon any termination', '2026-10-01'), "
                        "('00000000-0000-0000-0000-000000000011', 'a2', 1, "
                        "'the company shall pay a termination fee', '2026-10-01'), "
                        "('00000000-0000-0000-0000-000000000081', 'a_text_gated', 1, "
                        "'reverse termination fee', '2026-10-01')"
                    )
                )

    def _clear_text_search_rows(self) -> None:
        with self.app.app_context():
            with self.app_module.db.engine.begin() as conn:
                conn.execute(text("DELETE FROM section_text_search"))
                conn.execute(text("DELETE FROM xml WHERE agreement_uuid = 'a_text_gated'"))
                conn.execute(text("DELETE FROM agreements WHERE agreement_uuid = 'a_text_gated'"))

    def _sqlite_text_search(self):
        """Stand in for MariaDB FULLTEXT and POSIX REGEXP on the SQLite test database."""
        import contextlib
        import re

        from sqlalchemy import and_, event, or_

        def fake_candidate(column, *, text_query: str, match_mode: str):
            terms = [term for term in re.split(r"\W+", text_query.lower()) if term]
            likes = [column.like(f"%{term}%") for term in terms]
            return or_(*likes) if match_mode == "any_terms" else and_(*likes)

        def regexp(pattern: str, value: str | None) -> bool:
            if value is None:
                return False
            translated = (
                pattern.replace("[[:<:]]", r"\b").replace("[[:>:]]", r"\b").replace("[:alnum:]", "a-zA-Z0-9")
            )
            return re.search(translated, value, re.IGNORECASE) is not None

        def on_connect(dbapi_connection, _record) -> None:
            dbapi_connection.create_function("regexp", 2, regexp)

        @contextlib.contextmanager
        def active():
            from backend.services.sections_service import reset_text_plan_cache

            reset_text_plan_cache()
            with self.app.app_context():
                engine = self.app_module.db.engine
                event.listen(engine, "connect", on_connect)
                engine.dispose()
            try:
                with patch(
                    "backend.services.agreements_service.build_section_text_candidate_expression",
                    fake_candidate,
                ):
                    yield
            finally:
                with self.app.app_context():
                    engine = self.app_module.db.engine
                    event.remove(engine, "connect", on_connect)
                    engine.dispose()
                reset_text_plan_cache()

        return active()

    def test_search_agreements_applies_text_query(self):
        self._seed_text_search_rows()
        try:
            with self._sqlite_text_search():
                client = self.app.test_client()

                phrase = client.get(
                    "/v1/search/agreements?text_query=reverse+termination+fee&page_size=10"
                )
                self.assertEqual(phrase.status_code, 200, phrase.get_json())
                body = phrase.get_json()
                # a_text_gated matches too, but is gated and unverified.
                self.assertEqual([r["agreement_uuid"] for r in body["results"]], ["a1"])
                self.assertEqual(body["total_count"], 1)
                self.assertFalse(body["total_count_is_approximate"])
                self.assertEqual(body["count_metadata"]["method"], "query_count")
                first = body["results"][0]
                self.assertEqual(first["match_count"], 1)
                preview = first["matched_sections"][0]
                self.assertEqual(preview["section_uuid"], "00000000-0000-0000-0000-000000000001")
                self.assertEqual(preview["standard_id"], ["s1"])
                self.assertIn("Reverse Termination Fee", preview["snippet"])
                self.assertEqual(preview["matched_terms"], ["Reverse Termination Fee"])

                all_terms = client.get(
                    "/v1/search/agreements?text_query=termination+fee&text_match_mode=all_terms"
                    "&sort_by=year&sort_direction=asc&page_size=10"
                )
                self.assertEqual(all_terms.status_code, 200, all_terms.get_json())
                self.assertEqual(
                    [r["agreement_uuid"] for r in all_terms.get_json()["results"]], ["a1", "a2"]
                )

                filtered = client.get(
                    "/v1/search/agreements?text_query=termination+fee&text_match_mode=all_terms"
                    "&year=2021&standard_id=s1&page_size=10"
                )
                self.assertEqual(
                    [r["agreement_uuid"] for r in filtered.get_json()["results"]], ["a2"]
                )

                paged = client.get(
                    "/v1/search/agreements?text_query=termination+fee&text_match_mode=all_terms"
                    "&sort_by=year&sort_direction=asc&page=2&page_size=1"
                )
                paged_body = paged.get_json()
                self.assertEqual([r["agreement_uuid"] for r in paged_body["results"]], ["a2"])
                self.assertEqual(paged_body["total_count"], 2)

                none = client.get("/v1/search/agreements?text_query=indemnification&page_size=10")
                self.assertEqual(none.get_json()["results"], [])
                self.assertEqual(none.get_json()["total_count"], 0)
        finally:
            self._clear_text_search_rows()

    def test_search_agreements_text_query_too_expensive_is_a_422(self):
        from sqlalchemy.exc import OperationalError

        timeout = OperationalError(
            "SELECT 1", {}, Exception(1969, "Query execution was interrupted (max_statement_time exceeded)")
        )
        with patch(
            "backend.services.agreements_service._search_agreements_by_text", side_effect=timeout
        ):
            res = self.app.test_client().get("/v1/search/agreements?text_query=material+adverse+effect")
        self.assertEqual(res.status_code, 422)
        self.assertIn("text_query", json.dumps(res.get_json()).lower())

    def test_search_agreements_text_count_falls_back_to_lower_bound(self):
        self._seed_text_search_rows()
        try:
            with self._sqlite_text_search(), patch(
                "backend.services.agreements_service.budgeted_exact_count", return_value=None
            ):
                res = self.app.test_client().get(
                    "/v1/search/agreements?text_query=termination+fee&text_match_mode=all_terms"
                    "&page=1&page_size=1"
                )
            body = res.get_json()
            self.assertEqual(res.status_code, 200, body)
            self.assertTrue(body["total_count_is_approximate"])
            self.assertEqual(body["count_metadata"]["method"], "filtered_lower_bound")
            self.assertEqual(body["total_count"], 2)
            self.assertTrue(body["has_next"])
        finally:
            self._clear_text_search_rows()

    def test_text_search_answers_503_when_fulltext_slots_are_busy(self):
        # Keyword searches share a server-wide cap on concurrent FULLTEXT queries
        # (prod MariaDB was OOM-killed by several at once); a full cap is a retryable 503.
        from backend.core.fulltext_slots import FULLTEXT_BUSY_MESSAGE, FulltextSearchBusy

        import contextlib

        @contextlib.contextmanager
        def busy(_engine):
            raise FulltextSearchBusy()
            yield

        client = self.app.test_client()
        with (
            patch("backend.services.agreements_service.fulltext_slot", busy),
            patch("backend.services.sections_service.fulltext_slot", busy),
        ):
            for path in (
                "/v1/search/agreements?text_query=material+adverse+effect",
                "/v1/sections?text_query=material+adverse+effect",
            ):
                res = client.get(path)
                self.assertEqual(res.status_code, 503, (path, res.get_json()))
                self.assertEqual(res.headers.get("Retry-After"), "5")
                self.assertIn(FULLTEXT_BUSY_MESSAGE, json.dumps(res.get_json()))
            # Searches without a keyword never take a slot.
            self.assertEqual(client.get("/v1/search/agreements?page_size=1").status_code, 200)
            self.assertEqual(client.get("/v1/sections?page_size=1").status_code, 200)

    # -- Public eligibility on section reads ---------------------------------------------

    def test_section_search_and_get_section_exclude_gated_unverified_agreements(self):
        hidden_section = "00000000-0000-0000-0000-000000000091"
        with self.app.app_context():
            with self.app_module.db.engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO agreements (agreement_uuid, filing_date, target, acquirer, verified, gated, url) "
                        "VALUES ('a_sec_gated', '2020-05-01', 'Hidden T', 'Hidden A', 0, 1, 'http://example.com/h')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO xml (agreement_uuid, xml, version, status, latest) VALUES "
                        "('a_sec_gated', '<document/>', 1, 'verified', 1)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO sections (agreement_uuid, section_uuid, article_title, section_title, "
                        "xml_content, section_standard_id, xml_version) VALUES "
                        f"('a_sec_gated', '{hidden_section}', 'ARTICLE I', 'Hidden', '<section>H</section>', '[\"s1\"]', 1)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO latest_sections_search (section_uuid, agreement_uuid, filing_date, "
                        "target, acquirer, verified, url, section_standard_ids, article_title, section_title) "
                        f"VALUES ('{hidden_section}', 'a_sec_gated', '2020-05-01', 'Hidden T', 'Hidden A', 0, "
                        "'http://example.com/h', '[\"s1\"]', 'ARTICLE I', 'Hidden')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO latest_sections_search_standard_ids (standard_id, section_uuid, agreement_uuid) "
                        f"VALUES ('s1', '{hidden_section}', 'a_sec_gated')"
                    )
                )
        try:
            client = self.app.test_client()
            for path in ("/v1/sections?year=2020&page_size=50", "/v1/sections?standard_id=s1&page_size=50"):
                with self.subTest(path=path):
                    res = client.get(path)
                    self.assertEqual(res.status_code, 200)
                    body = res.get_json()
                    uuids = [item["section_uuid"] for item in body["results"]]
                    self.assertEqual(uuids, ["00000000-0000-0000-0000-000000000001"])
                    self.assertEqual(body["total_count"], 1)
            self.assertEqual(client.get(f"/v1/sections/{hidden_section}").status_code, 404)
        finally:
            with self.app.app_context():
                with self.app_module.db.engine.begin() as conn:
                    conn.execute(text("DELETE FROM xml WHERE agreement_uuid = 'a_sec_gated'"))
                    conn.execute(text("DELETE FROM agreements WHERE agreement_uuid = 'a_sec_gated'"))

    # -- Response and parameter normalization --------------------------------------------

    def test_agreements_index_accepts_sort_direction_and_legacy_sort_dir(self):
        client = self.app.test_client()

        def order(query: str) -> list[str]:
            res = client.get(f"/v1/agreements-index?sort_by=year&page_size=10&{query}")
            self.assertEqual(res.status_code, 200, res.get_json())
            return [item["agreement_uuid"] for item in res.get_json()["results"]]

        self.assertEqual(order("sort_direction=asc"), ["a2", "a3"])
        self.assertEqual(order("sort_dir=asc"), ["a2", "a3"])
        self.assertEqual(order(""), ["a3", "a2"])
        # sort_direction wins when both are sent.
        self.assertEqual(order("sort_direction=desc&sort_dir=asc"), ["a3", "a2"])
        # Same 400 this route has always returned for an invalid sort_dir.
        self.assertEqual(client.get("/v1/agreements-index?sort_direction=up").status_code, 400)

    def test_get_agreement_and_get_section_carry_canonical_ids(self):
        client = self.app.test_client()
        agreement = client.get("/v1/agreements/a1").get_json()
        self.assertEqual(agreement["agreement_uuid"], "a1")
        section = client.get("/v1/sections/00000000-0000-0000-0000-000000000001").get_json()
        self.assertEqual(section["standard_id"], ["s1"])
        self.assertEqual(section["section_standard_id"], section["standard_id"])

    def test_tax_clauses_report_dump_version(self):
        dump = {"hash": "abc123", "dump_ts": "2026-10-01T00:00:00Z"}
        with patch("backend.routes.tax_clauses.g", SimpleNamespace(dump_version=dump)):
            client = self.app.test_client()
            with_dump = client.get("/v1/tax-clauses?page=1&page_size=10")
            without_dump = client.get("/v1/tax-clauses?page=1&page_size=10&include_dump=false")
        self.assertEqual(with_dump.status_code, 200, with_dump.get_json())
        self.assertEqual(with_dump.get_json().get("dump_version"), dump)
        self.assertNotIn("dump_version", without_dump.get_json())

if __name__ == "__main__":
    unittest.main()
