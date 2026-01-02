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
    os.environ.setdefault("MARIADB_DATABASE", "mna")
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
                        "INSERT INTO agreements (uuid, year, target, acquirer, verified, url) "
                        "VALUES "
                        "('a1', 2020, 'Target A', 'Acquirer A', 1, 'http://example.com/a1'), "
                        "('a2', 2021, 'Target B', 'Acquirer B', 0, 'http://example.com/a2'), "
                        "('a3', 2022, 'Target C', 'Acquirer C', 1, 'http://example.com/a3')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO xml (agreement_uuid, xml) VALUES "
                        "('a1', '<document><article>"
                        "<section uuid=\"00000000-0000-0000-0000-000000000001\"><text>KEEP</text></section>"
                        "<section uuid=\"00000000-0000-0000-0000-000000000002\"><text>HIDE</text></section>"
                        "</article></document>')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO sections (agreement_uuid, section_uuid, article_title, section_title, "
                        "xml_content, article_standard_id, section_standard_id) VALUES "
                        "('a1', '00000000-0000-0000-0000-000000000001', "
                        "'ARTICLE I', 'Section 1', '<section>TEXT</section>', 'a1', 's1')"
                    )
                )

    def test_agreements_index_pagination(self):
        client = self.app.test_client()
        res = client.get("/api/agreements-index?page=1&pageSize=2")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("totalCount"), 3)
        self.assertEqual(body.get("totalPages"), 2)
        self.assertEqual(len(body.get("results", [])), 2)

    def test_search_basic(self):
        client = self.app.test_client()
        res = client.get("/api/search?year=2020&page=1&pageSize=10")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get("totalCount"), 1)
        self.assertEqual(len(body.get("results", [])), 1)
        self.assertIn("access", body)

    def test_agreement_redaction_for_anonymous(self):
        client = self.app.test_client()
        res = client.get(
            "/api/agreements/a1?focusSectionUuid=00000000-0000-0000-0000-000000000001&neighborSections=0"
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertTrue(body.get("isRedacted"))
        xml = body.get("xml", "")
        self.assertIn("[REDACTED]", xml)


if __name__ == "__main__":
    unittest.main()
