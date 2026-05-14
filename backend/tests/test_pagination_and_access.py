import os
import tempfile
import unittest
import importlib
import warnings
from unittest.mock import patch
from sqlalchemy.exc import SAWarning


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
        self.assertEqual(meta["total_pages"], 5)
        self.assertTrue(meta["has_next"])
        self.assertFalse(meta["has_prev"])
        self.assertEqual(meta["next_num"], 2)
        self.assertIsNone(meta["prev_num"])

    def test_pagination_metadata_empty(self):
        meta = backend_app._pagination_metadata(total_count=0, page=1, page_size=25)
        self.assertEqual(meta["total_pages"], 0)
        self.assertFalse(meta["has_next"])
        self.assertFalse(meta["has_prev"])

    def test_search_gets_have_higher_rate_limits_for_pagination(self):
        ctx = backend_app.AccessContext(tier="anonymous")
        with self.app.test_request_context("/v1/sections?page=42", method="GET"):
            self.assertEqual(backend_app._rate_limit_for_request(ctx, 60), 300)
        with self.app.test_request_context("/v1/search/agreements?page=42", method="GET"):
            self.assertEqual(backend_app._rate_limit_for_request(ctx, 60), 300)
        with self.app.test_request_context("/v1/tax-clauses?page=42", method="GET"):
            self.assertEqual(backend_app._rate_limit_for_request(ctx, 60), 300)

    def test_non_search_rate_limits_stay_unchanged(self):
        ctx = backend_app.AccessContext(tier="anonymous")
        with self.app.test_request_context("/v1/auth/me", method="GET"):
            self.assertEqual(backend_app._rate_limit_for_request(ctx, 60), 60)
        with self.app.test_request_context("/v1/auth/flag-inaccurate", method="POST"):
            self.assertEqual(backend_app._rate_limit_for_request(ctx, 60), 60)

    def test_csrf_required_cookie_transport(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        with self.app.test_request_context("/v1/auth/logout", method="POST"):
            self.assertTrue(backend_app._csrf_required("/v1/auth/logout"))

    def test_csrf_not_required_bearer_transport(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        with self.app.test_request_context("/v1/auth/logout", method="POST"):
            self.assertFalse(backend_app._csrf_required("/v1/auth/logout"))

    def test_safe_next_path_rejects_malformed_values(self):
        self.assertEqual(backend_app._safe_next_path("/account"), "/account")
        self.assertEqual(backend_app._safe_next_path("/account?x=1"), "/account?x=1")
        self.assertIsNone(backend_app._safe_next_path("https://evil.test/account"))
        self.assertIsNone(backend_app._safe_next_path("//evil.test/account"))
        self.assertIsNone(backend_app._safe_next_path("/account#frag"))
        self.assertIsNone(backend_app._safe_next_path("/account\\evil"))
        self.assertIsNone(backend_app._safe_next_path("/account\nx"))

    def test_rate_limit_pruning_caps_state_growth(self):
        old_max = backend_app._RATE_LIMIT_MAX_KEYS
        old_interval = backend_app._RATE_LIMIT_PRUNE_INTERVAL_SECONDS
        old_last_prune = backend_app._rate_limit_last_prune_at
        try:
            backend_app._RATE_LIMIT_MAX_KEYS = 3
            backend_app._RATE_LIMIT_PRUNE_INTERVAL_SECONDS = 0.0
            backend_app._rate_limit_last_prune_at = 0.0
            backend_app._rate_limit_state.clear()
            backend_app._endpoint_rate_limit_state.clear()

            now = 1000.0
            for idx in range(10):
                backend_app._rate_limit_state[f"r{idx}"] = {"ts": now + idx, "count": 1}
                backend_app._endpoint_rate_limit_state[f"e{idx}"] = {"ts": now + idx, "count": 1}

            backend_app._prune_rate_limit_state(now + 10.0)

            self.assertLessEqual(len(backend_app._rate_limit_state), 3)
            self.assertLessEqual(len(backend_app._endpoint_rate_limit_state), 3)
        finally:
            backend_app._RATE_LIMIT_MAX_KEYS = old_max
            backend_app._RATE_LIMIT_PRUNE_INTERVAL_SECONDS = old_interval
            backend_app._rate_limit_last_prune_at = old_last_prune
            backend_app._rate_limit_state.clear()
            backend_app._endpoint_rate_limit_state.clear()

    def test_endpoint_rate_limit_state_is_bounded_under_high_cardinality_auth_requests(self):
        old_max = backend_app._RATE_LIMIT_MAX_KEYS
        old_interval = backend_app._RATE_LIMIT_PRUNE_INTERVAL_SECONDS
        old_last_prune = backend_app._rate_limit_last_prune_at
        try:
            backend_app._RATE_LIMIT_MAX_KEYS = 8
            backend_app._RATE_LIMIT_PRUNE_INTERVAL_SECONDS = 0.0
            backend_app._rate_limit_last_prune_at = 0.0
            backend_app._rate_limit_state.clear()
            backend_app._endpoint_rate_limit_state.clear()

            for idx in range(60):
                with patch("backend.app.time.time", return_value=1000.0 + idx):
                    with patch("backend.app._request_ip_address", return_value=f"10.0.0.{idx}"):
                        with self.app.test_request_context("/v1/auth/flag-inaccurate", method="POST"):
                            backend_app._check_endpoint_rate_limit()

            # Pruning runs before insertion, so transient max+1 is expected.
            self.assertLessEqual(len(backend_app._endpoint_rate_limit_state), 9)
            self.assertLessEqual(len(backend_app._rate_limit_state), 9)
            backend_app._prune_rate_limit_state(2000.0)
            self.assertLessEqual(len(backend_app._endpoint_rate_limit_state), 8)
            self.assertLessEqual(len(backend_app._rate_limit_state), 8)
        finally:
            backend_app._RATE_LIMIT_MAX_KEYS = old_max
            backend_app._RATE_LIMIT_PRUNE_INTERVAL_SECONDS = old_interval
            backend_app._rate_limit_last_prune_at = old_last_prune
            backend_app._rate_limit_state.clear()
            backend_app._endpoint_rate_limit_state.clear()

    def test_reflection_toggle_defaults_enabled_and_can_be_disabled(self):
        def _reload_backend_app_safely():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        r"This declarative base already contains a class "
                        r"with the same class name and module name as backend\.app\..*"
                    ),
                    category=SAWarning,
                )
                return importlib.reload(backend_app)

        with patch.dict(
            os.environ,
            {
                "SKIP_MAIN_DB_REFLECTION": "1",
                "ENABLE_MAIN_DB_REFLECTION": "1",
            },
            clear=False,
        ):
            mod = _reload_backend_app_safely()
            self.assertTrue(mod._ENABLE_MAIN_DB_REFLECTION)

        with patch.dict(
            os.environ,
            {
                "SKIP_MAIN_DB_REFLECTION": "1",
                "ENABLE_MAIN_DB_REFLECTION": "0",
            },
            clear=False,
        ):
            mod = _reload_backend_app_safely()
            self.assertFalse(mod._ENABLE_MAIN_DB_REFLECTION)

        with patch.dict(
            os.environ,
            {
                "SKIP_MAIN_DB_REFLECTION": "1",
                "ENABLE_MAIN_DB_REFLECTION": "1",
            },
            clear=False,
        ):
            mod = _reload_backend_app_safely()
            self.assertTrue(mod._ENABLE_MAIN_DB_REFLECTION)


if __name__ == "__main__":
    unittest.main()
