import os
import tempfile
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


_set_default_env()

_AUTH_DB_TEMP = tempfile.NamedTemporaryFile(prefix="pandects_auth_", suffix=".sqlite", delete=False)
_AUTH_DB_TEMP.close()
os.environ.setdefault("AUTH_DATABASE_URI", f"sqlite:///{_AUTH_DB_TEMP.name}")


from backend.app import create_test_app, db  # noqa: E402
import backend.app as backend_app  # noqa: E402


class PaginationAndAccessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_test_app(
            config_overrides={
                "SQLALCHEMY_BINDS": {"auth": f"sqlite:///{_AUTH_DB_TEMP.name}"},
            }
        )
        with cls.app.app_context():
            db.create_all(bind_key="auth")

    def test_pagination_metadata_basic(self):
        meta = backend_app._pagination_metadata(total_count=101, page=1, page_size=25)
        self.assertEqual(meta["totalPages"], 5)
        self.assertTrue(meta["hasNext"])
        self.assertFalse(meta["hasPrev"])
        self.assertEqual(meta["nextNum"], 2)
        self.assertIsNone(meta["prevNum"])

    def test_pagination_metadata_empty(self):
        meta = backend_app._pagination_metadata(total_count=0, page=1, page_size=25)
        self.assertEqual(meta["totalPages"], 0)
        self.assertFalse(meta["hasNext"])
        self.assertFalse(meta["hasPrev"])

    def test_csrf_required_cookie_transport(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        with self.app.test_request_context("/api/auth/login", method="POST"):
            self.assertTrue(backend_app._csrf_required("/api/auth/login"))

    def test_csrf_not_required_bearer_transport(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        with self.app.test_request_context("/api/auth/login", method="POST"):
            self.assertFalse(backend_app._csrf_required("/api/auth/login"))


if __name__ == "__main__":
    unittest.main()
