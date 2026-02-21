import os
import tempfile
import unittest

from sqlalchemy import text


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
                        "INSERT INTO agreements (agreement_uuid, filing_date, target, acquirer, verified, url, deal_type) "
                        "VALUES "
                        "('a1', '2020-01-01', 'Target A', 'Acquirer A', 1, 'http://example.com/a1', 'merger'), "
                        "('a2', '2021-02-01', 'Target B', 'Acquirer B', 0, 'http://example.com/a2', NULL), "
                        "('a3', '2022-03-01', 'Target C', 'Acquirer C', 1, 'http://example.com/a3', NULL)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO xml (agreement_uuid, xml, version, status) VALUES "
                        "('a1', '<document><article>"
                        "<section uuid=\"00000000-0000-0000-0000-000000000001\"><text>KEEP</text></section>"
                        "<section uuid=\"00000000-0000-0000-0000-000000000002\"><text>HIDE</text></section>"
                        "</article></document>', 1, NULL)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO sections (agreement_uuid, section_uuid, article_title, section_title, "
                        "xml_content, section_standard_id) VALUES "
                        "('a1', '00000000-0000-0000-0000-000000000001', "
                        "'ARTICLE I', 'Section 1', '<section>TEXT</section>', '[\"s1\"]')"
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
                        "INSERT INTO agreement_deal_type_summary (year, deal_type, count) VALUES "
                        "(2020, 'merger', 1), "
                        "(2021, 'stock_acquisition', 2)"
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
        self.assertEqual(body.get("total_count"), 1)
        self.assertEqual(body.get("total_pages"), 1)
        self.assertEqual(len(body.get("results", [])), 1)

    def test_search_basic(self):
        client = self.app.test_client()
        res = client.get("/v1/search?year=2020&page=1&page_size=10")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("total_count"), 1)
        self.assertEqual(len(body.get("results", [])), 1)
        self.assertIn("access", body)

    def test_search_by_standard_id(self):
        client = self.app.test_client()
        res = client.get("/v1/search?standard_id=s1&page=1&page_size=10")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("total_count"), 1)
        self.assertEqual(len(body.get("results", [])), 1)

    def test_search_with_requested_metadata(self):
        client = self.app.test_client()
        res = client.get("/v1/search?year=2020&metadata=deal_type&page=1&page_size=10")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("total_count"), 1)
        results = body.get("results", [])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get("metadata"), {"deal_type": "merger"})

    def test_search_with_invalid_metadata_field(self):
        client = self.app.test_client()
        res = client.get("/v1/search?metadata=not_a_field&page=1&page_size=10")
        self.assertEqual(res.status_code, 422)

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


if __name__ == "__main__":
    unittest.main()
