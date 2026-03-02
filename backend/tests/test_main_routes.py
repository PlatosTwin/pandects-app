import base64
import json
import os
import tempfile
import unittest
from datetime import datetime

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
                        "INSERT INTO agreements (agreement_uuid, filing_date, target, acquirer, verified, url, deal_type, "
                        "transaction_consideration, target_type, acquirer_type, target_industry, acquirer_industry, "
                        "deal_status, attitude, purpose, target_pe, acquirer_pe) "
                        "VALUES "
                        "('a1', '2020-01-01', 'Target A', 'Acquirer A', 1, 'http://example.com/a1', 'merger', "
                        "'cash', 'public', 'public', 'tech', 'tech', 'complete', 'friendly', 'strategic', 0, 0), "
                        "('a2', '2021-02-01', 'Target B', 'Acquirer B', 0, 'http://example.com/a2', 'stock_acquisition', "
                        "'stock', 'private', 'public', 'healthcare', 'tech', 'pending', 'hostile', 'financial', 1, 0), "
                        "('a3', '2022-03-01', 'Target C', 'Acquirer C', 1, 'http://example.com/a3', 'asset_purchase', "
                        "'assets', 'private', 'private', 'energy', 'industrial', 'cancelled', 'friendly', 'strategic', 0, 1), "
                        "('a4', '2023-04-01', 'Target D', 'Acquirer D', 1, 'http://example.com/a4', 'merger', "
                        "'cash', 'public', 'public', 'finance', 'finance', 'pending', 'friendly', 'strategic', 0, 0)"
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
            user.email_verified_at = datetime.utcnow()
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

    def test_search_excludes_stale_section_versions(self):
        client = self.app.test_client()
        res = client.get("/v1/search?standard_id=s-old&page=1&page_size=10")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("total_count"), 0)
        self.assertEqual(body.get("results", []), [])

    def test_search_excludes_old_verified_when_latest_xml_is_invalid(self):
        client = self.app.test_client()
        res = client.get("/v1/search?standard_id=s4-old&page=1&page_size=10")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("total_count"), 0)
        self.assertEqual(body.get("results", []), [])

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
