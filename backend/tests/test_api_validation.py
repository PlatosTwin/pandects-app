import os
import tempfile
import time
import unittest


def _set_default_env() -> None:
    os.environ.setdefault("SKIP_MAIN_DB_REFLECTION", "1")
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
    os.environ["TURNSTILE_ENABLED"] = "0"
    os.environ.pop("TURNSTILE_SITE_KEY", None)
    os.environ.pop("TURNSTILE_SECRET_KEY", None)


_set_default_env()

_AUTH_DB_TEMP = tempfile.NamedTemporaryFile(prefix="pandects_auth_", suffix=".sqlite", delete=False)
_AUTH_DB_TEMP.close()
os.environ.setdefault("AUTH_DATABASE_URI", f"sqlite:///{_AUTH_DB_TEMP.name}")


from backend.app import create_test_app, db  # noqa: E402
import backend.app as backend_app  # noqa: E402


class ApiValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_test_app(
            config_overrides={
                "SQLALCHEMY_BINDS": {"auth": f"sqlite:///{_AUTH_DB_TEMP.name}"},
            }
        )
        with cls.app.app_context():
            db.create_all(bind_key="auth")

    def test_zitadel_start_rejects_unsupported_provider(self):
        client = self.app.test_client()
        res = client.get("/v1/auth/zitadel/start?provider=github")
        self.assertEqual(res.status_code, 400)
        body = res.get_json()
        self.assertIsInstance(body, dict)
        self.assertEqual(body.get("error"), "Bad Request")

    def test_zitadel_start_rejects_unsupported_prompt(self):
        client = self.app.test_client()
        res = client.get("/v1/auth/zitadel/start?provider=email&prompt=consent")
        self.assertEqual(res.status_code, 400)
        body = res.get_json()
        self.assertIsInstance(body, dict)
        self.assertEqual(body.get("error"), "Bad Request")

    def test_legacy_register_route_is_not_registered(self):
        client = self.app.test_client()
        res = client.post("/v1/auth/register", json={})
        self.assertEqual(res.status_code, 404)

    def test_dumps_cache_returns_cached_payload(self):
        payload: list[dict[str, object]] = [{"timestamp": "2025-01-01"}]
        backend_app._dumps_cache["payload"] = payload
        backend_app._dumps_cache["ts"] = time.time()

        client = self.app.test_client()
        res = client.get("/v1/dumps")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body, payload)


if __name__ == "__main__":
    unittest.main()
