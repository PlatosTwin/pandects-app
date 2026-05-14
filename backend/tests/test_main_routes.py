import base64
import json
import os
import tempfile
import unittest
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
        self.app_module._rate_limit_state.clear()
        self.app_module._endpoint_rate_limit_state.clear()
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
            self.app_module._filter_options_cache["payload"] = None
            self.app_module._filter_options_cache["ts"] = 0

    def test_agreements_index_pagination(self):
        client = self.app.test_client()
        res = client.get("/v1/agreements-index?page=1&page_size=2")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("total_count"), 2)
        self.assertEqual(body.get("total_pages"), 1)
        self.assertEqual(len(body.get("results", [])), 2)

    def test_agreements_summary_includes_latest_filing_date(self):
        self.app_module._agreements_summary_cache["payload"] = None
        self.app_module._agreements_summary_cache["ts"] = 0
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
        self.assertEqual(body.get("count_metadata", {}).get("method"), "query_count")
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

    def test_search_total_count_metadata_legacy_shim_forwards_auto_mode(self):
        with patch.object(
            self.app_module,
            "_svc_sections_total_count_metadata",
            return_value=(123, True, "table_estimate"),
        ) as metadata_mock:
            total_count, is_approximate = self.app_module._search_total_count_metadata(
                query=object(),
                page=2,
                page_size=25,
                item_count=25,
                has_next=True,
                has_filters=False,
            )

        metadata_mock.assert_called_once()
        self.assertEqual(metadata_mock.call_args.kwargs["count_mode"], "auto")
        self.assertEqual(total_count, 123)
        self.assertTrue(is_approximate)

    def test_search_total_count_metadata_uses_query_estimate_for_filtered_search_beyond_page_one(self):
        with patch.object(self.app_module, "_estimated_query_row_count", return_value=240):
            total_count, is_approximate = self.app_module._search_total_count_metadata(
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

        total_count, is_approximate = self.app_module._search_total_count_metadata(
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
        ordered_query.count.return_value = 212
        with patch.object(self.app_module, "_estimated_query_row_count", return_value=120):
            total_count, is_approximate = self.app_module._search_total_count_metadata(
                query=query,
                page=6,
                page_size=25,
                item_count=25,
                has_next=True,
                has_filters=True,
            )

        query.order_by.assert_called_once_with(None)
        ordered_query.count.assert_called_once_with()
        self.assertEqual(total_count, 212)
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
                self.app_module._filter_options_cache["payload"] = None
                self.app_module._filter_options_cache["ts"] = 0

    def test_filter_options_include_counsel_lists(self):
        self.app_module._filter_options_cache["payload"] = None
        self.app_module._filter_options_cache["ts"] = 0
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
        self.app_module._filter_options_cache["payload"] = None
        self.app_module._filter_options_cache["ts"] = 0
        client = self.app.test_client()
        res = client.get(
            "/v1/filter-options?fields=target_counsels&fields=acquirer_industries"
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(
            set(body.keys()),
            {"target_counsels", "acquirer_industries"},
        )
        self.assertIn("Goodwin Procter", body.get("target_counsels", []))
        self.assertIn("tech", body.get("acquirer_industries", []))

    def test_filter_options_can_request_clause_types(self):
        self.app_module._filter_options_cache["payload"] = None
        self.app_module._filter_options_cache["ts"] = 0
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
        self.assertIn("count_metadata", body)
        self.assertNotEqual(body.get("count_metadata", {}).get("method"), "filtered_lower_bound")
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


if __name__ == "__main__":
    unittest.main()
