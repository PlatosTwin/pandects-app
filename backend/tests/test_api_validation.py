import os
import tempfile
import time
import unittest


def _set_default_env() -> None:
    os.environ.setdefault("SKIP_MAIN_DB_REFLECTION", "1")
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

    def test_register_missing_body_returns_validation_error(self):
        client = self.app.test_client()
        res = client.post("/api/auth/register", json={})
        self.assertEqual(res.status_code, 400)
        body = res.get_json()
        self.assertIsInstance(body, dict)
        self.assertEqual(body.get("error"), "validation_error")

    def test_register_requires_legal(self):
        client = self.app.test_client()
        res = client.post(
            "/api/auth/register",
            json={"email": "x@example.com", "password": "password123"},
        )
        self.assertEqual(res.status_code, 412)
        body = res.get_json()
        self.assertIsInstance(body, dict)
        self.assertEqual(body.get("error"), "legal_required")

    def test_register_requires_captcha_when_enabled(self):
        os.environ["TURNSTILE_ENABLED"] = "1"
        os.environ["TURNSTILE_SITE_KEY"] = "test-site-key"
        os.environ["TURNSTILE_SECRET_KEY"] = "test-secret-key"
        client = self.app.test_client()

        try:
            res = client.post(
                "/api/auth/register",
                json={
                    "email": "captcha@example.com",
                    "password": "password123",
                    "legal": {
                        "checkedAtMs": 1700000000000,
                        "docs": ["tos", "privacy", "license"],
                    },
                },
            )
            self.assertEqual(res.status_code, 412)
            body = res.get_json()
            self.assertIsInstance(body, dict)
            self.assertEqual(body.get("error"), "captcha_required")
        finally:
            os.environ["TURNSTILE_ENABLED"] = "0"
            os.environ.pop("TURNSTILE_SITE_KEY", None)
            os.environ.pop("TURNSTILE_SECRET_KEY", None)

    def test_dumps_cache_returns_cached_payload(self):
        payload = [{"timestamp": "2025-01-01"}]
        backend_app._dumps_cache["payload"] = payload
        backend_app._dumps_cache["ts"] = time.time()

        client = self.app.test_client()
        res = client.get("/api/dumps")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body, payload)


if __name__ == "__main__":
    unittest.main()
