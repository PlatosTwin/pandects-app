import os
import tempfile
import unittest
from unittest.mock import patch
from sqlalchemy import text
from werkzeug.exceptions import ServiceUnavailable


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
    os.environ["TURNSTILE_ENABLED"] = "0"
    os.environ.pop("TURNSTILE_SITE_KEY", None)
    os.environ.pop("TURNSTILE_SECRET_KEY", None)


_set_default_env()

_AUTH_DB_TEMP = tempfile.NamedTemporaryFile(prefix="pandects_auth_", suffix=".sqlite", delete=False)
_AUTH_DB_TEMP.close()
os.environ.setdefault("AUTH_DATABASE_URI", f"sqlite:///{_AUTH_DB_TEMP.name}")


from backend.app import create_test_app, db, ApiKey, AuthUser  # noqa: E402
import backend.app as backend_app  # noqa: E402


class AuthFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_test_app(
            config_overrides={
                "SQLALCHEMY_BINDS": {"auth": f"sqlite:///{_AUTH_DB_TEMP.name}"},
            }
        )
        with cls.app.app_context():
            db.create_all(bind_key="auth")

    def setUp(self) -> None:
        with self.app.app_context():
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
        backend_app._api_key_last_used_touch_state.clear()

    def _csrf_cookie_value(self, client) -> str:
        cookie = client.get_cookie("pdcts_csrf")
        if cookie is None:
            self.fail("Expected pdcts_csrf cookie to be set")
        return cookie.value

    def _set_google_nonce_cookie(self, client, nonce: str = "test-google-nonce") -> str:
        client.set_cookie(
            "pdcts_google_nonce",
            nonce,
            path="/v1/auth/google/credential",
        )
        return nonce

    def _require_user(self, email: str) -> AuthUser:
        user = AuthUser.query.filter_by(email=email).first()
        if user is None:
            self.fail(f"Expected test user with email={email}")
        return user

    def test_cookie_transport_register_login_csrf_and_logout(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        client = self.app.test_client()

        res = client.get("/v1/auth/csrf")
        self.assertEqual(res.status_code, 200)
        csrf = self._csrf_cookie_value(client)
        checked_at_ms = 1700000000000

        res = client.post(
            "/v1/auth/register",
            json={
                "email": "a@example.com",
                "password": "password123",
                "legal": {
                    "checked_at_ms": checked_at_ms,
                    "docs": ["tos", "privacy", "license"],
                },
            },
        )
        self.assertEqual(res.status_code, 403)

        res = client.post(
            "/v1/auth/register",
            json={
                "email": "a@example.com",
                "password": "password123",
                "legal": {
                    "checked_at_ms": checked_at_ms,
                    "docs": ["tos", "privacy", "license"],
                },
            },
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(res.status_code, 201)
        set_cookie = "\n".join(res.headers.getlist("Set-Cookie"))
        if "pdcts_session=" in set_cookie:
            self.assertIn("Expires=Thu, 01 Jan 1970", set_cookie)

        with self.app.app_context():
            user = self._require_user("a@example.com")
            token = backend_app._issue_email_verification_token(
                user_id=user.id, email=user.email
            )

        res = client.post("/v1/auth/email/verify", json={"token": token})
        self.assertEqual(res.status_code, 200)

        res = client.get("/v1/auth/csrf")
        csrf = self._csrf_cookie_value(client)

        res = client.post(
            "/v1/auth/login",
            json={"email": "a@example.com", "password": "password123"},
        )
        self.assertEqual(res.status_code, 403)

        res = client.post(
            "/v1/auth/login",
            json={"email": "a@example.com", "password": "password123"},
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(res.status_code, 200)
        set_cookie = "\n".join(res.headers.getlist("Set-Cookie"))
        self.assertIn("pdcts_session=", set_cookie)
        self.assertIn("HttpOnly", set_cookie)

        res = client.get("/v1/auth/me")
        self.assertEqual(res.status_code, 200)

        res = client.post("/v1/auth/api-keys", json={"name": "x"})
        self.assertEqual(res.status_code, 403)

        csrf = self._csrf_cookie_value(client)
        res = client.post(
            "/v1/auth/api-keys",
            json={"name": "x"},
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(res.status_code, 200)

        res = client.post("/v1/auth/logout")
        self.assertEqual(res.status_code, 403)
        res = client.post("/v1/auth/logout", headers={"X-CSRF-Token": csrf})
        self.assertEqual(res.status_code, 200)
        set_cookie = "\n".join(res.headers.getlist("Set-Cookie"))
        self.assertIn("pdcts_session=;", set_cookie)

        res = client.get("/v1/auth/me")
        self.assertEqual(res.status_code, 401)

    def test_register_and_login_record_signon_events(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()

        res = client.post(
            "/v1/auth/register",
            json={
                "email": "events@example.com",
                "password": "password123",
                "legal": {
                    "checked_at_ms": 1700000000000,
                    "docs": ["tos", "privacy", "license"],
                },
            },
        )
        self.assertEqual(res.status_code, 201)

        with self.app.app_context():
            user = self._require_user("events@example.com")
            token = backend_app._issue_email_verification_token(
                user_id=user.id, email=user.email
            )

        res = client.post("/v1/auth/login", json={"email": "events@example.com", "password": "password123"})
        self.assertEqual(res.status_code, 403)

        res = client.post("/v1/auth/email/verify", json={"token": token})
        self.assertEqual(res.status_code, 200)

        res = client.post(
            "/v1/auth/login",
            json={"email": "events@example.com", "password": "password123"},
        )
        self.assertEqual(res.status_code, 200)

        with self.app.app_context():
            engine = db.engines["auth"]
            with engine.begin() as conn:
                rows = conn.execute(
                    text("SELECT provider, action FROM auth_signon_events")
                ).fetchall()
            self.assertEqual(len(rows), 2)
            self.assertIn(("email", "register"), rows)
            self.assertIn(("email", "login"), rows)

    def test_register_persists_account_when_verification_email_delivery_fails(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()

        original_send = backend_app._send_email_verification_email

        def _fail_send(*, to_email: str, token: str) -> None:
            raise ServiceUnavailable(description="Email delivery failed.")

        backend_app._send_email_verification_email = _fail_send
        try:
            res = client.post(
                "/v1/auth/register",
                json={
                    "email": "delivery-fail@example.com",
                    "password": "password123",
                    "legal": {
                        "checked_at_ms": 1700000000000,
                        "docs": ["tos", "privacy", "license"],
                    },
                },
            )
        finally:
            backend_app._send_email_verification_email = original_send

        self.assertEqual(res.status_code, 201)
        body = res.get_json()
        self.assertIsInstance(body, dict)
        self.assertEqual(body.get("status"), "verification_required")

        with self.app.app_context():
            user = self._require_user("delivery-fail@example.com")
            self.assertIsNone(user.email_verified_at)
            engine = db.engines["auth"]
            with engine.begin() as conn:
                legal_count = conn.execute(
                    text("SELECT COUNT(*) FROM legal_acceptances WHERE user_id = :user_id"),
                    {"user_id": user.id},
                ).scalar_one()
                signon_count = conn.execute(
                    text(
                        "SELECT COUNT(*) FROM auth_signon_events "
                        "WHERE user_id = :user_id AND provider = 'email' AND action = 'register'"
                    ),
                    {"user_id": user.id},
                ).scalar_one()
            self.assertEqual(legal_count, len(backend_app._LEGAL_DOCS))
            self.assertEqual(signon_count, 1)

    def test_register_existing_user_still_succeeds_when_verification_email_delivery_fails(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        with self.app.app_context():
            existing = AuthUser()
            existing.email = "existing-unverified@example.com"
            existing.password_hash = backend_app.generate_password_hash("password123")
            existing.email_verified_at = None
            db.session.add(existing)
            db.session.commit()
            existing_id = existing.id

        original_send = backend_app._send_email_verification_email

        def _fail_send(*, to_email: str, token: str) -> None:
            raise ServiceUnavailable(description="Email delivery failed.")

        backend_app._send_email_verification_email = _fail_send
        try:
            res = client.post(
                "/v1/auth/register",
                json={
                    "email": "existing-unverified@example.com",
                    "password": "password123",
                    "legal": {
                        "checked_at_ms": 1700000000000,
                        "docs": ["tos", "privacy", "license"],
                    },
                },
            )
        finally:
            backend_app._send_email_verification_email = original_send

        self.assertEqual(res.status_code, 201)
        body = res.get_json()
        self.assertIsInstance(body, dict)
        user_payload = body.get("user")
        self.assertIsInstance(user_payload, dict)
        self.assertEqual(user_payload.get("id"), existing_id)

    def test_logout_revokes_bearer_session(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()

        res = client.post(
            "/v1/auth/register",
            json={
                "email": "bearer@example.com",
                "password": "password123",
                "legal": {
                    "checked_at_ms": 1700000000000,
                    "docs": ["tos", "privacy", "license"],
                },
            },
        )
        self.assertEqual(res.status_code, 201)

        with self.app.app_context():
            user = self._require_user("bearer@example.com")
            token = backend_app._issue_email_verification_token(
                user_id=user.id, email=user.email
            )

        res = client.post("/v1/auth/email/verify", json={"token": token})
        self.assertEqual(res.status_code, 200)

        res = client.post(
            "/v1/auth/login",
            json={"email": "bearer@example.com", "password": "password123"},
        )
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertIsInstance(payload, dict)
        session_token = payload.get("session_token")
        self.assertIsInstance(session_token, str)

        res = client.get(
            "/v1/auth/me", headers={"Authorization": f"Bearer {session_token}"}
        )
        self.assertEqual(res.status_code, 200)

        res = client.post(
            "/v1/auth/logout", headers={"Authorization": f"Bearer {session_token}"}
        )
        self.assertEqual(res.status_code, 200)

        res = client.get(
            "/v1/auth/me", headers={"Authorization": f"Bearer {session_token}"}
        )
        self.assertEqual(res.status_code, 401)

    def test_flag_inaccurate_requires_auth_and_validates(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()

        res = client.post(
            "/v1/auth/flag-inaccurate",
            json={
                "source": "search_result",
                "agreement_uuid": "a1",
                "section_uuid": "s1",
                "message": "Typo in summary.",
                "request_follow_up": True,
                "issue_types": ["Incorrect metadata"],
            },
        )
        self.assertEqual(res.status_code, 401)

        res = client.post(
            "/v1/auth/register",
            json={
                "email": "flag@example.com",
                "password": "password123",
                "legal": {
                    "checked_at_ms": 1700000000000,
                    "docs": ["tos", "privacy", "license"],
                },
            },
        )
        self.assertEqual(res.status_code, 201)
        with self.app.app_context():
            user = self._require_user("flag@example.com")
            token = backend_app._issue_email_verification_token(
                user_id=user.id, email=user.email
            )
        res = client.post("/v1/auth/email/verify", json={"token": token})
        self.assertEqual(res.status_code, 200)
        res = client.post(
            "/v1/auth/login",
            json={"email": "flag@example.com", "password": "password123"},
        )
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertIsInstance(payload, dict)
        session_token = payload.get("session_token")
        self.assertIsInstance(session_token, str)
        headers = {"Authorization": f"Bearer {session_token}"}

        res = client.post(
            "/v1/auth/flag-inaccurate",
            headers=headers,
            json={
                "source": "search_result",
                "agreement_uuid": "a1",
                "section_uuid": "s1",
                "message": "Metadata mismatch in clause excerpt.",
                "request_follow_up": False,
                "issue_types": ["Incorrect metadata", "Incorrect tagging (Article/Section)"],
            },
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), {"status": "ok"})

        res = client.post(
            "/v1/auth/flag-inaccurate",
            headers=headers,
            json={
                "source": "agreement_view",
                "agreement_uuid": "a2",
                "request_follow_up": True,
                "issue_types": ["Incorrect taxonomy class"],
            },
        )
        self.assertEqual(res.status_code, 200)

        res = client.post(
            "/v1/auth/flag-inaccurate",
            headers=headers,
            json={
                "source": "bad",
                "agreement_uuid": "a1",
                "section_uuid": "s1",
                "message": "Bad source",
                "request_follow_up": False,
                "issue_types": ["Something else"],
            },
        )
        self.assertEqual(res.status_code, 400)

        res = client.post(
            "/v1/auth/flag-inaccurate",
            headers=headers,
            json={
                "source": "search_result",
                "agreement_uuid": "a1",
                "issue_types": ["Corrupted formatting"],
            },
        )
        self.assertEqual(res.status_code, 400)

        res = client.post(
            "/v1/auth/flag-inaccurate",
            headers=headers,
            json={
                "source": "agreement_view",
                "agreement_uuid": "a1",
                "issue_types": ["Incorrect tagging (Article/Section)"],
            },
        )
        self.assertEqual(res.status_code, 200)

        res = client.post(
            "/v1/auth/flag-inaccurate",
            headers=headers,
            json={
                "source": "agreement_view",
                "agreement_uuid": "a1",
                "message": "Invalid issue type.",
                "issue_types": ["Not a real option"],
            },
        )
        self.assertEqual(res.status_code, 400)

        res = client.post(
            "/v1/auth/flag-inaccurate",
            headers=headers,
            json={
                "source": "agreement_view",
                "agreement_uuid": "a1",
                "message": "Missing issue types.",
            },
        )
        self.assertEqual(res.status_code, 400)

        res = client.post(
            "/v1/auth/flag-inaccurate",
            headers=headers,
            json={
                "source": "agreement_view",
                "agreement_uuid": "a1",
                "issue_types": [],
            },
        )
        self.assertEqual(res.status_code, 400)

    def test_password_reset_flow(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()

        res = client.post(
            "/v1/auth/register",
            json={
                "email": "reset@example.com",
                "password": "password123",
                "legal": {
                    "checked_at_ms": 1700000000000,
                    "docs": ["tos", "privacy", "license"],
                },
            },
        )
        self.assertEqual(res.status_code, 201)

        with self.app.app_context():
            user = self._require_user("reset@example.com")
            token = backend_app._issue_email_verification_token(
                user_id=user.id, email=user.email
            )

        res = client.post("/v1/auth/email/verify", json={"token": token})
        self.assertEqual(res.status_code, 200)

        captured: dict[str, str] = {}
        original_send = backend_app._send_password_reset_email
        backend_app._send_password_reset_email = lambda *, to_email, token: captured.setdefault(
            "token", token
        )
        try:
            res = client.post("/v1/auth/password/forgot", json={"email": "reset@example.com"})
            self.assertEqual(res.status_code, 200)
            reset_token = captured.get("token")
            self.assertIsInstance(reset_token, str)

            res = client.post(
                "/v1/auth/password/reset",
                json={"token": reset_token, "password": "newpassword123"},
            )
            self.assertEqual(res.status_code, 200)
        finally:
            backend_app._send_password_reset_email = original_send

        res = client.post(
            "/v1/auth/login",
            json={"email": "reset@example.com", "password": "newpassword123"},
        )
        self.assertEqual(res.status_code, 200)

    def test_resend_verification_returns_sent_when_delivery_fails(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        with self.app.app_context():
            user = AuthUser()
            user.email = "resend-fail@example.com"
            user.password_hash = backend_app.generate_password_hash("password123")
            user.email_verified_at = None
            db.session.add(user)
            db.session.commit()

        original_send = backend_app._send_email_verification_email

        def _fail_send(*, to_email: str, token: str) -> None:
            raise ServiceUnavailable(description="Email delivery failed.")

        backend_app._send_email_verification_email = _fail_send
        try:
            res = client.post("/v1/auth/email/resend", json={"email": "resend-fail@example.com"})
        finally:
            backend_app._send_email_verification_email = original_send

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), {"status": "sent"})

    def test_password_forgot_returns_sent_when_delivery_fails(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        with self.app.app_context():
            user = AuthUser()
            user.email = "forgot-fail@example.com"
            user.password_hash = backend_app.generate_password_hash("password123")
            user.email_verified_at = None
            db.session.add(user)
            db.session.commit()

        original_send = backend_app._send_password_reset_email

        def _fail_send(*, to_email: str, token: str) -> None:
            raise ServiceUnavailable(description="Email delivery failed.")

        backend_app._send_password_reset_email = _fail_send
        try:
            res = client.post("/v1/auth/password/forgot", json={"email": "forgot-fail@example.com"})
        finally:
            backend_app._send_password_reset_email = original_send

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), {"status": "sent"})

    def test_google_credential_requires_legal_for_new_users_and_logs_events(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        self._set_google_nonce_cookie(client)

        original_verify = backend_app._google_verify_id_token
        backend_app._google_verify_id_token = lambda _token, expected_nonce=None: "google-new@example.com"

        try:
            res = client.post("/v1/auth/google/credential", json={"credential": "fake"})
            self.assertEqual(res.status_code, 412)
            body = res.get_json()
            self.assertIsInstance(body, dict)
            self.assertEqual(body.get("error"), "legal_required")

            res = client.post(
                "/v1/auth/google/credential",
                json={
                    "credential": "fake",
                    "legal": {
                        "checked_at_ms": 1700000000000,
                        "docs": ["tos", "privacy", "license"],
                    },
                },
            )
            self.assertEqual(res.status_code, 200)

            with self.app.app_context():
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
        client = self.app.test_client()

        res = client.post("/v1/auth/google/credential", json={"credential": "fake"})
        self.assertEqual(res.status_code, 400)
        body = res.get_json()
        self.assertIsInstance(body, dict)
        self.assertIn("Missing Google nonce", body.get("message", ""))

    def test_google_credential_passes_nonce_from_cookie(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        expected = self._set_google_nonce_cookie(client)

        captured: dict[str, str | None] = {"nonce": None}
        original_verify = backend_app._google_verify_id_token
        backend_app._google_verify_id_token = (
            lambda _token, expected_nonce=None: captured.__setitem__("nonce", expected_nonce) or "google-new@example.com"
        )
        try:
            res = client.post("/v1/auth/google/credential", json={"credential": "fake"})
            self.assertEqual(res.status_code, 412)
            self.assertEqual(captured["nonce"], expected)
        finally:
            backend_app._google_verify_id_token = original_verify

    def test_register_requires_captcha_when_enabled(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["TURNSTILE_ENABLED"] = "1"
        os.environ["TURNSTILE_SITE_KEY"] = "test-site-key"
        os.environ["TURNSTILE_SECRET_KEY"] = "test-secret-key"
        client = self.app.test_client()

        original_verify = backend_app._verify_turnstile_token
        backend_app._verify_turnstile_token = lambda *, token: None
        try:
            res = client.post(
                "/v1/auth/register",
                json={
                    "email": "captcha@example.com",
                    "password": "password123",
                    "legal": {
                        "checked_at_ms": 1700000000000,
                        "docs": ["tos", "privacy", "license"],
                    },
                },
            )
            self.assertEqual(res.status_code, 412)
            body = res.get_json()
            self.assertIsInstance(body, dict)
            self.assertEqual(body.get("error"), "captcha_required")

            res = client.post(
                "/v1/auth/register",
                json={
                    "email": "captcha@example.com",
                    "password": "password123",
                    "captcha_token": "ok",
                    "legal": {
                        "checked_at_ms": 1700000000000,
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
        client = self.app.test_client()

        res = client.post(
            "/v1/auth/register",
            json={
                "email": "b@example.com",
                "password": "password123",
                "legal": {
                    "checked_at_ms": 1700000000000,
                    "docs": ["tos", "privacy", "license"],
                },
            },
        )
        self.assertEqual(res.status_code, 201)
        res = client.post("/v1/auth/login", json={"email": "b@example.com", "password": "password123"})
        self.assertEqual(res.status_code, 403)

        with self.app.app_context():
            user = self._require_user("b@example.com")
            verify = backend_app._issue_email_verification_token(user_id=user.id, email=user.email)

        res = client.post("/v1/auth/email/verify", json={"token": verify})
        self.assertEqual(res.status_code, 200)

        res = client.post("/v1/auth/login", json={"email": "b@example.com", "password": "password123"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsInstance(data, dict)
        token = data.get("session_token")
        self.assertIsInstance(token, str)
        self.assertGreater(len(token), 10)

        res = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(res.status_code, 200)

    def test_google_credential_invalid_token_is_401_without_network(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        self._set_google_nonce_cookie(client)

        import jwt

        class DummyJwkClient:
            def get_signing_key_from_jwt(self, _token: str):
                raise jwt.exceptions.InvalidTokenError("bad token")

        backend_app._google_jwk_client = DummyJwkClient()

        res = client.post("/v1/auth/google/credential", json={"credential": "nope"})
        self.assertEqual(res.status_code, 401)

    def test_google_credential_jwks_outage_returns_503(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        self._set_google_nonce_cookie(client)

        import jwt

        class DummyJwkClient:
            def get_signing_key_from_jwt(self, _token: str):
                raise jwt.exceptions.PyJWKClientError("jwks unavailable")

        backend_app._google_jwk_client = DummyJwkClient()

        res = client.post("/v1/auth/google/credential", json={"credential": "nope"})
        self.assertEqual(res.status_code, 503)

    def test_cors_allows_credentials_for_localhost(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        client = self.app.test_client()

        origin = "http://localhost:8080"
        res = client.get("/v1/auth/csrf", headers={"Origin": origin})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("Access-Control-Allow-Credentials"), "true")
        self.assertEqual(res.headers.get("Access-Control-Allow-Origin"), origin)

    def test_cors_allows_docs_origins(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        client = self.app.test_client()

        for origin in (
            "http://localhost:3001",
            "https://docs.pandects.org",
        ):
            with self.subTest(origin=origin):
                res = client.get("/v1/auth/csrf", headers={"Origin": origin})
                self.assertEqual(res.status_code, 200)
                self.assertEqual(res.headers.get("Access-Control-Allow-Credentials"), "true")
                self.assertEqual(res.headers.get("Access-Control-Allow-Origin"), origin)

    def test_delete_account_requires_confirmation_and_revokes_keys(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        client = self.app.test_client()

        res = client.get("/v1/auth/csrf")
        csrf = self._csrf_cookie_value(client)

        res = client.post(
            "/v1/auth/register",
            json={
                "email": "delete-me@example.com",
                "password": "password123",
                "legal": {
                    "checked_at_ms": 1700000000000,
                    "docs": ["tos", "privacy", "license"],
                },
            },
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(res.status_code, 201)

        with self.app.app_context():
            user = AuthUser.query.filter_by(email="delete-me@example.com").first()
            if user is None:
                self.fail("Expected user to be created for delete-account flow.")
            verify = backend_app._issue_email_verification_token(user_id=user.id, email=user.email)

        res = client.post("/v1/auth/email/verify", json={"token": verify})
        self.assertEqual(res.status_code, 200)

        res = client.get("/v1/auth/csrf")
        csrf = self._csrf_cookie_value(client)
        res = client.post(
            "/v1/auth/login",
            json={"email": "delete-me@example.com", "password": "password123"},
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(res.status_code, 200)

        csrf = self._csrf_cookie_value(client)
        res = client.post(
            "/v1/auth/account/delete",
            json={"confirm": "nope"},
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(res.status_code, 400)

        res = client.post(
            "/v1/auth/account/delete",
            json={"confirm": "Delete"},
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(res.status_code, 200)

        res = client.get("/v1/auth/me")
        self.assertEqual(res.status_code, 401)

    def test_api_key_whitespace_is_ignored_for_last_used(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()

        res = client.post(
            "/v1/auth/register",
            json={
                "email": "keyuser@example.com",
                "password": "password123",
                "legal": {
                    "checked_at_ms": 1700000000000,
                    "docs": ["tos", "privacy", "license"],
                },
            },
        )
        self.assertEqual(res.status_code, 201)
        with self.app.app_context():
            user = AuthUser.query.filter_by(email="keyuser@example.com").first()
            if user is None:
                self.fail("Expected user to be created for api-key flow.")
            verify = backend_app._issue_email_verification_token(user_id=user.id, email=user.email)

        res = client.post("/v1/auth/email/verify", json={"token": verify})
        self.assertEqual(res.status_code, 200)

        res = client.post("/v1/auth/login", json={"email": "keyuser@example.com", "password": "password123"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsInstance(data, dict)
        token = data.get("session_token")
        self.assertIsInstance(token, str)

        res = client.post(
            "/v1/auth/api-keys",
            json={"name": "x"},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(res.status_code, 200)
        created = res.get_json()
        self.assertIsInstance(created, dict)
        api_key = created.get("api_key_plaintext")
        self.assertIsInstance(api_key, str)

        with self.app.app_context():
            key_row = ApiKey.query.filter_by(prefix=api_key[:18]).first()
            if key_row is None:
                self.fail("Expected api key row to exist before usage update.")
            self.assertIsNone(key_row.last_used_at)

        res = client.get("/v1/auth/me", headers={"X-API-Key": f"  {api_key}  "})
        self.assertEqual(res.status_code, 401)

        with self.app.app_context():
            key_row = ApiKey.query.filter_by(prefix=api_key[:18]).first()
            if key_row is None:
                self.fail("Expected api key row to exist after usage update.")
            self.assertIsNotNone(key_row.last_used_at)

    def test_api_key_last_used_updates_are_throttled(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()

        res = client.post(
            "/v1/auth/register",
            json={
                "email": "keyuser-throttle@example.com",
                "password": "password123",
                "legal": {
                    "checked_at_ms": 1700000000000,
                    "docs": ["tos", "privacy", "license"],
                },
            },
        )
        self.assertEqual(res.status_code, 201)
        with self.app.app_context():
            user = AuthUser.query.filter_by(email="keyuser-throttle@example.com").first()
            if user is None:
                self.fail("Expected user to be created for throttled api-key flow.")
            verify = backend_app._issue_email_verification_token(user_id=user.id, email=user.email)

        res = client.post("/v1/auth/email/verify", json={"token": verify})
        self.assertEqual(res.status_code, 200)

        res = client.post(
            "/v1/auth/login",
            json={"email": "keyuser-throttle@example.com", "password": "password123"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsInstance(data, dict)
        token = data.get("session_token")
        self.assertIsInstance(token, str)

        res = client.post(
            "/v1/auth/api-keys",
            json={"name": "x"},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(res.status_code, 200)
        created = res.get_json()
        self.assertIsInstance(created, dict)
        api_key = created.get("api_key_plaintext")
        self.assertIsInstance(api_key, str)

        old_touch_seconds = backend_app._API_KEY_LAST_USED_TOUCH_SECONDS
        try:
            backend_app._API_KEY_LAST_USED_TOUCH_SECONDS = 300
            backend_app._api_key_last_used_touch_state.clear()

            with patch("backend.app.time.time", return_value=1000.0):
                res = client.get("/v1/auth/me", headers={"X-API-Key": api_key})
                self.assertEqual(res.status_code, 401)
            with self.app.app_context():
                key_row = ApiKey.query.filter_by(prefix=api_key[:18]).first()
                if key_row is None:
                    self.fail("Expected api key row to exist after first touch.")
                first_touched_at = key_row.last_used_at
                self.assertIsNotNone(first_touched_at)

            with patch("backend.app.time.time", return_value=1001.0):
                res = client.get("/v1/auth/me", headers={"X-API-Key": api_key})
                self.assertEqual(res.status_code, 401)
            with self.app.app_context():
                key_row = ApiKey.query.filter_by(prefix=api_key[:18]).first()
                if key_row is None:
                    self.fail("Expected api key row to exist after second touch.")
                self.assertEqual(key_row.last_used_at, first_touched_at)

            with patch("backend.app.time.time", return_value=1401.0):
                res = client.get("/v1/auth/me", headers={"X-API-Key": api_key})
                self.assertEqual(res.status_code, 401)
            with self.app.app_context():
                key_row = ApiKey.query.filter_by(prefix=api_key[:18]).first()
                if key_row is None:
                    self.fail("Expected api key row to exist after third touch.")
                self.assertNotEqual(key_row.last_used_at, first_touched_at)
        finally:
            backend_app._API_KEY_LAST_USED_TOUCH_SECONDS = old_touch_seconds
            backend_app._api_key_last_used_touch_state.clear()


if __name__ == "__main__":
    unittest.main()
