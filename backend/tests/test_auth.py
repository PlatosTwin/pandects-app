import os
import tempfile
import unittest
from sqlalchemy import text


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
    os.environ["TURNSTILE_ENABLED"] = "0"
    os.environ.pop("TURNSTILE_SITE_KEY", None)
    os.environ.pop("TURNSTILE_SECRET_KEY", None)


_set_default_env()

_AUTH_DB_TEMP = tempfile.NamedTemporaryFile(prefix="pandects_auth_", suffix=".sqlite", delete=False)
_AUTH_DB_TEMP.close()
os.environ.setdefault("AUTH_DATABASE_URI", f"sqlite:///{_AUTH_DB_TEMP.name}")


from backend.app import app, db, ApiKey, AuthUser  # noqa: E402
import backend.app as backend_app  # noqa: E402


class AuthFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app.testing = True
        with app.app_context():
            db.create_all(bind_key="auth")

    def setUp(self) -> None:
        with app.app_context():
            engine = db.engines["auth"]
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM auth_sessions"))
                conn.execute(text("DELETE FROM auth_password_reset_tokens"))
                conn.execute(text("DELETE FROM api_request_events"))
                conn.execute(text("DELETE FROM api_usage_hourly"))
                conn.execute(text("DELETE FROM api_usage_daily_ips"))
                conn.execute(text("DELETE FROM api_usage_daily"))
                conn.execute(text("DELETE FROM api_keys"))
                conn.execute(text("DELETE FROM legal_acceptances"))
                conn.execute(text("DELETE FROM auth_signon_events"))
                conn.execute(text("DELETE FROM auth_users"))
        backend_app._rate_limit_state.clear()
        backend_app._endpoint_rate_limit_state.clear()

    def _csrf_cookie_value(self, client) -> str:
        cookie = client.get_cookie("pdcts_csrf")
        if cookie is None:
            self.fail("Expected pdcts_csrf cookie to be set")
        return cookie.value

    def _set_google_nonce_cookie(self, client, nonce: str = "test-google-nonce") -> str:
        client.set_cookie(
            "pdcts_google_nonce",
            nonce,
            path="/api/auth/google/credential",
        )
        return nonce

    def test_cookie_transport_register_login_csrf_and_logout(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        client = app.test_client()

        res = client.get("/api/auth/csrf")
        self.assertEqual(res.status_code, 200)
        csrf = self._csrf_cookie_value(client)
        checked_at_ms = 1700000000000

        res = client.post(
            "/api/auth/register",
            json={
                "email": "a@example.com",
                "password": "password123",
                "legal": {
                    "checkedAtMs": checked_at_ms,
                    "docs": ["tos", "privacy", "license"],
                },
            },
        )
        self.assertEqual(res.status_code, 403)

        res = client.post(
            "/api/auth/register",
            json={
                "email": "a@example.com",
                "password": "password123",
                "legal": {
                    "checkedAtMs": checked_at_ms,
                    "docs": ["tos", "privacy", "license"],
                },
            },
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(res.status_code, 201)
        set_cookie = "\n".join(res.headers.getlist("Set-Cookie"))
        if "pdcts_session=" in set_cookie:
            self.assertIn("Expires=Thu, 01 Jan 1970", set_cookie)

        with app.app_context():
            user = AuthUser.query.filter_by(email="a@example.com").first()
            self.assertIsNotNone(user)
            token = backend_app._issue_email_verification_token(
                user_id=user.id, email=user.email
            )

        res = client.post("/api/auth/email/verify", json={"token": token})
        self.assertEqual(res.status_code, 200)

        res = client.get("/api/auth/csrf")
        csrf = self._csrf_cookie_value(client)

        res = client.post(
            "/api/auth/login",
            json={"email": "a@example.com", "password": "password123"},
        )
        self.assertEqual(res.status_code, 403)

        res = client.post(
            "/api/auth/login",
            json={"email": "a@example.com", "password": "password123"},
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(res.status_code, 200)
        set_cookie = "\n".join(res.headers.getlist("Set-Cookie"))
        self.assertIn("pdcts_session=", set_cookie)
        self.assertIn("HttpOnly", set_cookie)

        res = client.get("/api/auth/me")
        self.assertEqual(res.status_code, 200)

        res = client.post("/api/auth/api-keys", json={"name": "x"})
        self.assertEqual(res.status_code, 403)

        csrf = self._csrf_cookie_value(client)
        res = client.post(
            "/api/auth/api-keys",
            json={"name": "x"},
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(res.status_code, 200)

        res = client.post("/api/auth/logout")
        self.assertEqual(res.status_code, 403)
        res = client.post("/api/auth/logout", headers={"X-CSRF-Token": csrf})
        self.assertEqual(res.status_code, 200)
        set_cookie = "\n".join(res.headers.getlist("Set-Cookie"))
        self.assertIn("pdcts_session=;", set_cookie)

        res = client.get("/api/auth/me")
        self.assertEqual(res.status_code, 401)

    def test_register_and_login_record_signon_events(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = app.test_client()

        res = client.post(
            "/api/auth/register",
            json={
                "email": "events@example.com",
                "password": "password123",
                "legal": {
                    "checkedAtMs": 1700000000000,
                    "docs": ["tos", "privacy", "license"],
                },
            },
        )
        self.assertEqual(res.status_code, 201)

        with app.app_context():
            user = AuthUser.query.filter_by(email="events@example.com").first()
            self.assertIsNotNone(user)
            token = backend_app._issue_email_verification_token(
                user_id=user.id, email=user.email
            )

        res = client.post("/api/auth/login", json={"email": "events@example.com", "password": "password123"})
        self.assertEqual(res.status_code, 403)

        res = client.post("/api/auth/email/verify", json={"token": token})
        self.assertEqual(res.status_code, 200)

        res = client.post(
            "/api/auth/login",
            json={"email": "events@example.com", "password": "password123"},
        )
        self.assertEqual(res.status_code, 200)

        with app.app_context():
            engine = db.engines["auth"]
            with engine.begin() as conn:
                rows = conn.execute(
                    text("SELECT provider, action FROM auth_signon_events")
                ).fetchall()
            self.assertEqual(len(rows), 2)
            self.assertIn(("email", "register"), rows)
            self.assertIn(("email", "login"), rows)

    def test_logout_revokes_bearer_session(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = app.test_client()

        res = client.post(
            "/api/auth/register",
            json={
                "email": "bearer@example.com",
                "password": "password123",
                "legal": {
                    "checkedAtMs": 1700000000000,
                    "docs": ["tos", "privacy", "license"],
                },
            },
        )
        self.assertEqual(res.status_code, 201)

        with app.app_context():
            user = AuthUser.query.filter_by(email="bearer@example.com").first()
            self.assertIsNotNone(user)
            token = backend_app._issue_email_verification_token(
                user_id=user.id, email=user.email
            )

        res = client.post("/api/auth/email/verify", json={"token": token})
        self.assertEqual(res.status_code, 200)

        res = client.post(
            "/api/auth/login",
            json={"email": "bearer@example.com", "password": "password123"},
        )
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertIsInstance(payload, dict)
        session_token = payload.get("sessionToken")
        self.assertIsInstance(session_token, str)

        res = client.get(
            "/api/auth/me", headers={"Authorization": f"Bearer {session_token}"}
        )
        self.assertEqual(res.status_code, 200)

        res = client.post(
            "/api/auth/logout", headers={"Authorization": f"Bearer {session_token}"}
        )
        self.assertEqual(res.status_code, 200)

        res = client.get(
            "/api/auth/me", headers={"Authorization": f"Bearer {session_token}"}
        )
        self.assertEqual(res.status_code, 401)

    def test_password_reset_flow(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = app.test_client()

        res = client.post(
            "/api/auth/register",
            json={
                "email": "reset@example.com",
                "password": "password123",
                "legal": {
                    "checkedAtMs": 1700000000000,
                    "docs": ["tos", "privacy", "license"],
                },
            },
        )
        self.assertEqual(res.status_code, 201)

        with app.app_context():
            user = AuthUser.query.filter_by(email="reset@example.com").first()
            self.assertIsNotNone(user)
            token = backend_app._issue_email_verification_token(
                user_id=user.id, email=user.email
            )

        res = client.post("/api/auth/email/verify", json={"token": token})
        self.assertEqual(res.status_code, 200)

        captured: dict[str, str] = {}
        original_send = backend_app._send_password_reset_email
        backend_app._send_password_reset_email = lambda *, to_email, token: captured.setdefault(
            "token", token
        )
        try:
            res = client.post("/api/auth/password/forgot", json={"email": "reset@example.com"})
            self.assertEqual(res.status_code, 200)
            reset_token = captured.get("token")
            self.assertIsInstance(reset_token, str)

            res = client.post(
                "/api/auth/password/reset",
                json={"token": reset_token, "password": "newpassword123"},
            )
            self.assertEqual(res.status_code, 200)
        finally:
            backend_app._send_password_reset_email = original_send

        res = client.post(
            "/api/auth/login",
            json={"email": "reset@example.com", "password": "newpassword123"},
        )
        self.assertEqual(res.status_code, 200)

    def test_google_credential_requires_legal_for_new_users_and_logs_events(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = app.test_client()
        self._set_google_nonce_cookie(client)

        original_verify = backend_app._google_verify_id_token
        backend_app._google_verify_id_token = lambda _token, expected_nonce=None: "google-new@example.com"

        try:
            res = client.post("/api/auth/google/credential", json={"credential": "fake"})
            self.assertEqual(res.status_code, 412)
            body = res.get_json()
            self.assertIsInstance(body, dict)
            self.assertEqual(body.get("error"), "legal_required")

            res = client.post(
                "/api/auth/google/credential",
                json={
                    "credential": "fake",
                    "legal": {
                        "checkedAtMs": 1700000000000,
                        "docs": ["tos", "privacy", "license"],
                    },
                },
            )
            self.assertEqual(res.status_code, 200)

            with app.app_context():
                engine = db.engines["auth"]
                with engine.begin() as conn:
                    rows = conn.execute(
                        text(
                            "SELECT provider, action "
                            "FROM auth_signon_events "
                            "WHERE provider = 'google'"
                        )
                    ).fetchall()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0], ("google", "register"))
        finally:
            backend_app._google_verify_id_token = original_verify

    def test_google_credential_requires_nonce_cookie(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = app.test_client()

        res = client.post("/api/auth/google/credential", json={"credential": "fake"})
        self.assertEqual(res.status_code, 400)
        body = res.get_json()
        self.assertIsInstance(body, dict)
        self.assertIn("Missing Google nonce", body.get("message", ""))

    def test_google_credential_passes_nonce_from_cookie(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = app.test_client()
        expected = self._set_google_nonce_cookie(client)

        captured: dict[str, str | None] = {"nonce": None}
        original_verify = backend_app._google_verify_id_token
        backend_app._google_verify_id_token = (
            lambda _token, expected_nonce=None: captured.__setitem__("nonce", expected_nonce) or "google-new@example.com"
        )
        try:
            res = client.post("/api/auth/google/credential", json={"credential": "fake"})
            self.assertEqual(res.status_code, 412)
            self.assertEqual(captured["nonce"], expected)
        finally:
            backend_app._google_verify_id_token = original_verify

    def test_register_requires_captcha_when_enabled(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["TURNSTILE_ENABLED"] = "1"
        os.environ["TURNSTILE_SITE_KEY"] = "test-site-key"
        os.environ["TURNSTILE_SECRET_KEY"] = "test-secret-key"
        client = app.test_client()

        original_verify = backend_app._verify_turnstile_token
        backend_app._verify_turnstile_token = lambda *, token: None
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

            res = client.post(
                "/api/auth/register",
                json={
                    "email": "captcha@example.com",
                    "password": "password123",
                    "captchaToken": "ok",
                    "legal": {
                        "checkedAtMs": 1700000000000,
                        "docs": ["tos", "privacy", "license"],
                    },
                },
            )
            self.assertEqual(res.status_code, 201)
        finally:
            backend_app._verify_turnstile_token = original_verify
            os.environ.pop("TURNSTILE_ENABLED", None)
            os.environ.pop("TURNSTILE_SITE_KEY", None)
            os.environ.pop("TURNSTILE_SECRET_KEY", None)

    def test_bearer_transport_issues_session_tokens(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = app.test_client()

        res = client.post(
            "/api/auth/register",
            json={
                "email": "b@example.com",
                "password": "password123",
                "legal": {
                    "checkedAtMs": 1700000000000,
                    "docs": ["tos", "privacy", "license"],
                },
            },
        )
        self.assertEqual(res.status_code, 201)
        res = client.post("/api/auth/login", json={"email": "b@example.com", "password": "password123"})
        self.assertEqual(res.status_code, 403)

        with app.app_context():
            user = AuthUser.query.filter_by(email="b@example.com").first()
            self.assertIsNotNone(user)
            verify = backend_app._issue_email_verification_token(user_id=user.id, email=user.email)

        res = client.post("/api/auth/email/verify", json={"token": verify})
        self.assertEqual(res.status_code, 200)

        res = client.post("/api/auth/login", json={"email": "b@example.com", "password": "password123"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsInstance(data, dict)
        token = data.get("sessionToken")
        self.assertIsInstance(token, str)
        self.assertGreater(len(token), 10)

        res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(res.status_code, 200)

    def test_google_credential_invalid_token_is_401_without_network(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = app.test_client()
        self._set_google_nonce_cookie(client)

        import jwt

        class DummyJwkClient:
            def get_signing_key_from_jwt(self, _token: str):
                raise jwt.exceptions.InvalidTokenError("bad token")

        backend_app._google_jwk_client = DummyJwkClient()

        res = client.post("/api/auth/google/credential", json={"credential": "nope"})
        self.assertEqual(res.status_code, 401)

    def test_google_credential_jwks_outage_returns_503(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = app.test_client()
        self._set_google_nonce_cookie(client)

        import jwt

        class DummyJwkClient:
            def get_signing_key_from_jwt(self, _token: str):
                raise jwt.exceptions.PyJWKClientError("jwks unavailable")

        backend_app._google_jwk_client = DummyJwkClient()

        res = client.post("/api/auth/google/credential", json={"credential": "nope"})
        self.assertEqual(res.status_code, 503)

    def test_cors_allows_credentials_for_localhost(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        client = app.test_client()

        res = client.get("/api/auth/csrf", headers={"Origin": "http://localhost:8080"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("Access-Control-Allow-Credentials"), "true")

    def test_delete_account_requires_confirmation_and_revokes_keys(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        client = app.test_client()

        res = client.get("/api/auth/csrf")
        csrf = self._csrf_cookie_value(client)

        res = client.post(
            "/api/auth/register",
            json={
                "email": "delete-me@example.com",
                "password": "password123",
                "legal": {
                    "checkedAtMs": 1700000000000,
                    "docs": ["tos", "privacy", "license"],
                },
            },
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(res.status_code, 201)

        with app.app_context():
            user = AuthUser.query.filter_by(email="delete-me@example.com").first()
            self.assertIsNotNone(user)
            verify = backend_app._issue_email_verification_token(user_id=user.id, email=user.email)

        res = client.post("/api/auth/email/verify", json={"token": verify})
        self.assertEqual(res.status_code, 200)

        res = client.get("/api/auth/csrf")
        csrf = self._csrf_cookie_value(client)
        res = client.post(
            "/api/auth/login",
            json={"email": "delete-me@example.com", "password": "password123"},
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(res.status_code, 200)

        csrf = self._csrf_cookie_value(client)
        res = client.post(
            "/api/auth/account/delete",
            json={"confirm": "nope"},
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(res.status_code, 400)

        res = client.post(
            "/api/auth/account/delete",
            json={"confirm": "Delete"},
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(res.status_code, 200)

        res = client.get("/api/auth/me")
        self.assertEqual(res.status_code, 401)

    def test_api_key_whitespace_is_ignored_for_last_used(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = app.test_client()

        res = client.post(
            "/api/auth/register",
            json={
                "email": "keyuser@example.com",
                "password": "password123",
                "legal": {
                    "checkedAtMs": 1700000000000,
                    "docs": ["tos", "privacy", "license"],
                },
            },
        )
        self.assertEqual(res.status_code, 201)
        with app.app_context():
            user = AuthUser.query.filter_by(email="keyuser@example.com").first()
            self.assertIsNotNone(user)
            verify = backend_app._issue_email_verification_token(user_id=user.id, email=user.email)

        res = client.post("/api/auth/email/verify", json={"token": verify})
        self.assertEqual(res.status_code, 200)

        res = client.post("/api/auth/login", json={"email": "keyuser@example.com", "password": "password123"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsInstance(data, dict)
        token = data.get("sessionToken")
        self.assertIsInstance(token, str)

        res = client.post(
            "/api/auth/api-keys",
            json={"name": "x"},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(res.status_code, 200)
        created = res.get_json()
        self.assertIsInstance(created, dict)
        api_key = created.get("apiKeyPlaintext")
        self.assertIsInstance(api_key, str)

        with app.app_context():
            key_row = ApiKey.query.filter_by(prefix=api_key[:18]).first()
            self.assertIsNotNone(key_row)
            self.assertIsNone(key_row.last_used_at)

        res = client.get("/api/auth/me", headers={"X-API-Key": f"  {api_key}  "})
        self.assertEqual(res.status_code, 401)

        with app.app_context():
            key_row = ApiKey.query.filter_by(prefix=api_key[:18]).first()
            self.assertIsNotNone(key_row)
            self.assertIsNotNone(key_row.last_used_at)


if __name__ == "__main__":
    unittest.main()
