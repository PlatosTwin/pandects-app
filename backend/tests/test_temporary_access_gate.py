import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from sqlalchemy.exc import OperationalError


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

_AUTH_DB_TEMP = tempfile.NamedTemporaryFile(prefix="pandects_gate_", suffix=".sqlite", delete=False)
_AUTH_DB_TEMP.close()
os.environ.setdefault("AUTH_DATABASE_URI", f"sqlite:///{_AUTH_DB_TEMP.name}")


from backend.app import create_test_app, db  # noqa: E402
import backend.app as backend_app  # noqa: E402


# The gate reads TEMPORARY_ACCESS_LOCKDOWN per-request, so each test forces the
# value it needs and patch.dict restores the suite-wide default afterwards.
_LOCKDOWN_ON = {"TEMPORARY_ACCESS_LOCKDOWN": "1"}


class TemporaryAccessGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_test_app(
            config_overrides={
                "SQLALCHEMY_BINDS": {"auth": f"sqlite:///{_AUTH_DB_TEMP.name}"},
            }
        )
        with cls.app.app_context():
            db.create_all(bind_key="auth")

    def _seed_user(self, *, user_id: str, email: str, created_at: datetime) -> None:
        with self.app.app_context():
            user = backend_app.AuthUser()
            user.id = user_id
            user.email = email
            user.password_hash = "not-used"
            user.email_verified_at = created_at
            user.created_at = created_at
            db.session.merge(user)
            db.session.commit()

    @patch.dict(os.environ, _LOCKDOWN_ON)
    def test_anonymous_data_api_blocked(self):
        client = self.app.test_client()
        res = client.get("/v1/agreements")
        self.assertEqual(res.status_code, 403)
        self.assertIn("temporarily disabled", res.get_json().get("message", ""))

    @patch.dict(os.environ, _LOCKDOWN_ON)
    def test_password_signup_blocked(self):
        client = self.app.test_client()
        res = client.post("/v1/auth/signup/password", json={"email": "x@y.z", "password": "pw"})
        self.assertEqual(res.status_code, 403)
        self.assertIn("New user creation is temporarily disabled", res.get_json().get("message", ""))

    @patch.dict(os.environ, _LOCKDOWN_ON)
    def test_auth_endpoints_are_not_gated(self):
        # /v1/auth/* must stay reachable so existing users can still sign in;
        # anonymous /me reaches the handler and returns 401, not the gate's 403.
        client = self.app.test_client()
        res = client.get("/v1/auth/me")
        self.assertEqual(res.status_code, 401)

    @patch.dict(os.environ, _LOCKDOWN_ON)
    def test_healthz_is_not_gated(self):
        res = self.app.test_client().get("/healthz")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_data(as_text=True), "ok")

    @patch.dict(os.environ, _LOCKDOWN_ON)
    def test_unknown_bearer_token_still_blocked(self):
        client = self.app.test_client()
        res = client.get("/v1/agreements", headers={"Authorization": "Bearer not-a-session"})
        self.assertEqual(res.status_code, 403)

    @patch.dict(os.environ, _LOCKDOWN_ON)
    def test_auth_db_error_answers_503_not_lockdown_403(self):
        # A signed-in user whose session lookup fails (auth Postgres down) must not
        # be told the data API is disabled for them.
        outage = OperationalError("SELECT auth_sessions", {}, Exception("connection refused"))
        client = self.app.test_client()
        with patch("backend.auth.session_runtime.session_token_hash", side_effect=outage):
            res = client.get("/v1/agreements", headers={"Authorization": "Bearer some-session"})
        self.assertEqual(res.status_code, 503)
        self.assertIn("Auth backend is unavailable", res.get_json().get("message", ""))

    @patch.dict(os.environ, _LOCKDOWN_ON)
    def test_account_age_lookup_error_answers_503(self):
        # The session and user resolve; only the gate's own account-age read fails.
        user_id = "00000000-0000-0000-0000-0000000000e3"
        self._seed_user(
            user_id=user_id,
            email="existing-outage@example.com",
            created_at=backend_app._utc_now() - timedelta(days=3),
        )
        outage = OperationalError("SELECT auth_users", {}, Exception("connection refused"))
        tiers: list[str] = []
        real_predates = backend_app._access_account_predates_today

        def predates_during_outage(ctx):
            tiers.append(ctx.tier)
            with patch("backend.auth.access.db.session.get", side_effect=outage):
                return real_predates(ctx)

        client = self.app.test_client()
        with (
            patch("backend.auth.access.load_session_token", return_value=user_id),
            patch(
                "backend.auth.access.access_account_predates_today",
                side_effect=predates_during_outage,
            ),
        ):
            res = client.get("/v1/agreements", headers={"Authorization": "Bearer some-session"})
        self.assertEqual(tiers, ["user"])
        self.assertEqual(res.status_code, 503)

    @patch.dict(os.environ, _LOCKDOWN_ON)
    def test_auth_db_error_without_credentials_still_403(self):
        # Nothing to look up, so an outage must not turn the lockdown 403 into a retry.
        outage = OperationalError("SELECT auth_users", {}, Exception("connection refused"))
        client = self.app.test_client()
        with (
            patch("backend.auth.session_runtime.session_token_hash", side_effect=outage),
            patch("backend.auth.access.db.session.get", side_effect=outage),
        ):
            res = client.get("/v1/agreements")
            self.assertEqual(res.status_code, 403)
            res = client.get("/v1/agreements", headers={"Authorization": "Bearer "})
            self.assertEqual(res.status_code, 403)

    def test_predates_today_helper(self):
        now = backend_app._utc_now()
        yesterday = now - timedelta(days=1)
        self._seed_user(
            user_id="00000000-0000-0000-0000-0000000000e1",
            email="existing@example.com",
            created_at=yesterday,
        )
        self._seed_user(
            user_id="00000000-0000-0000-0000-0000000000e2",
            email="newtoday@example.com",
            created_at=now,
        )
        with self.app.app_context():
            existing_ctx = backend_app.AccessContext(
                tier="user", user_id="00000000-0000-0000-0000-0000000000e1"
            )
            today_ctx = backend_app.AccessContext(
                tier="user", user_id="00000000-0000-0000-0000-0000000000e2"
            )
            anon_ctx = backend_app.AccessContext(tier="anonymous")
            self.assertTrue(backend_app._access_account_predates_today(existing_ctx))
            self.assertFalse(backend_app._access_account_predates_today(today_ctx))
            self.assertFalse(backend_app._access_account_predates_today(anon_ctx))


if __name__ == "__main__":
    unittest.main()
