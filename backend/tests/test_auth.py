import os
import tempfile
import unittest
import hashlib
import hmac
import json
import time
from datetime import date, timedelta
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch
from sqlalchemy import text


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
    os.environ.setdefault("MCP_ZITADEL_CLIENT_ID", "test-zitadel-client-id")
    os.environ.setdefault("MCP_OIDC_ISSUER", "https://pandects-test-zitadel.example.com")
    os.environ.setdefault("MCP_OIDC_AUDIENCE", "https://api.pandects.org/mcp")
    os.environ.pop("AUTH_ZITADEL_PROJECT_ID", None)
    os.environ.pop("AUTH_ZITADEL_PROJECT_NAME", None)
    os.environ["TURNSTILE_ENABLED"] = "0"
    os.environ.pop("TURNSTILE_SITE_KEY", None)
    os.environ.pop("TURNSTILE_SECRET_KEY", None)


_set_default_env()

_AUTH_DB_TEMP = tempfile.NamedTemporaryFile(prefix="pandects_auth_", suffix=".sqlite", delete=False)
_AUTH_DB_TEMP.close()
os.environ.setdefault("AUTH_DATABASE_URI", f"sqlite:///{_AUTH_DB_TEMP.name}")


from backend.app import create_test_app, db, ApiKey, ApiUsageDaily, AuthExternalSubject, AuthUser  # noqa: E402
import backend.app as backend_app  # noqa: E402
import backend.routes.auth as auth_routes  # noqa: E402
from werkzeug.exceptions import Conflict, NotFound, Unauthorized  # noqa: E402


def _make_api_usage_daily(*, api_key_id: str, day: date, count: int) -> object:
    return ApiUsageDaily(api_key_id=api_key_id, day=day, count=count)  # pyright: ignore[reportCallIssue]


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
                conn.execute(text("DELETE FROM auth_external_subjects"))
                conn.execute(text("DELETE FROM auth_signon_events"))
                conn.execute(text("DELETE FROM auth_users"))
        backend_app._rate_limit_state.clear()
        backend_app._endpoint_rate_limit_state.clear()
        backend_app._api_key_last_used_touch_state.clear()
        auth_routes._ZITADEL_API_TOKEN_CACHE.clear()

    def _csrf_cookie_value(self, client) -> str:
        cookie = client.get_cookie("pdcts_csrf")
        if cookie is None:
            self.fail("Expected pdcts_csrf cookie to be set")
        return cookie.value

    def _require_user(self, email: str) -> AuthUser:
        user = AuthUser.query.filter_by(email=email).first()
        if user is None:
            self.fail(f"Expected test user with email={email}")
        return user

    def _create_local_user(
        self,
        *,
        email: str,
        password: str = "password123",
        verified: bool = True,
        legal: bool = True,
    ) -> str:
        with self.app.app_context():
            user = AuthUser()
            user.email = email
            user.password_hash = backend_app.generate_password_hash(password)
            user.email_verified_at = backend_app._utc_now() if verified else None
            db.session.add(user)
            db.session.flush()
            if legal:
                checked_at = backend_app._utc_now()
                for doc, meta in backend_app._LEGAL_DOCS.items():
                    db.session.add(
                        backend_app.LegalAcceptance(
                            user_id=user.id,
                            document=doc,
                            version=meta["version"],
                            document_hash=meta["sha256"],
                            checked_at=checked_at,
                            submitted_at=checked_at,
                            ip_address=None,
                            user_agent=None,
                        )
                    )
            db.session.commit()
            return user.id

    def _issue_bearer_session(self, *, email: str) -> str:
        with self.app.app_context():
            user = self._require_user(email)
            with self.app.test_request_context("/v1/auth/zitadel/complete", method="POST"):
                return backend_app._issue_session_token(user.id)

    def _set_cookie_session(self, client, *, email: str) -> str:
        token = self._issue_bearer_session(email=email)
        client.set_cookie("pdcts_session", token, path="/")
        return token

    def _zitadel_signature_headers(self, payload: dict[str, object]) -> dict[str, str]:
        os.environ["AUTH_ZITADEL_NOTIFICATION_SIGNING_KEY"] = "test-zitadel-notification-signing-key"
        raw = json.dumps(payload).encode("utf-8")
        timestamp = str(int(time.time()))
        digest = hmac.new(
            b"test-zitadel-notification-signing-key",
            timestamp.encode("utf-8") + b"." + raw,
            hashlib.sha256,
        ).hexdigest()
        return {
            "Content-Type": "application/json",
            "Zitadel-Signature": f"t={timestamp},v1={digest}",
        }

    def test_cookie_transport_register_login_csrf_and_logout(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        client = self.app.test_client()

        res = client.get("/v1/auth/csrf")
        self.assertEqual(res.status_code, 200)
        csrf = self._csrf_cookie_value(client)
        res = client.post("/v1/auth/register", json={})
        self.assertEqual(res.status_code, 404)
        res = client.post("/v1/auth/login", json={})
        self.assertEqual(res.status_code, 404)

        self._create_local_user(email="a@example.com")
        self._set_cookie_session(client, email="a@example.com")

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
        res = client.post("/v1/auth/register", json={})
        self.assertEqual(res.status_code, 404)
        res = client.post("/v1/auth/login", json={})
        self.assertEqual(res.status_code, 404)

    def test_register_persists_account_when_verification_email_delivery_fails(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        res = client.post("/v1/auth/register", json={})
        self.assertEqual(res.status_code, 404)

    def test_register_existing_user_still_succeeds_when_verification_email_delivery_fails(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        self._create_local_user(email="existing-unverified@example.com", verified=False, legal=False)
        res = client.post("/v1/auth/register", json={})
        self.assertEqual(res.status_code, 404)

    def test_logout_revokes_bearer_session(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        self._create_local_user(email="bearer@example.com")
        session_token = self._issue_bearer_session(email="bearer@example.com")

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

        self._create_local_user(email="flag@example.com")
        headers = {"Authorization": f"Bearer {self._issue_bearer_session(email='flag@example.com')}"}

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

    def test_external_subject_link_and_list(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["MCP_IDENTITY_PROVIDER"] = "zitadel"
        client = self.app.test_client()
        self._create_local_user(email="link@example.com")
        headers = {"Authorization": f"Bearer {self._issue_bearer_session(email='link@example.com')}"}

        original_authenticate = backend_app._authenticate_external_identity

        def _fake_authenticate_external_identity(
            *, access_token: str, provider_name: str | None = None
        ):
            self.assertEqual(access_token, "zitadel-access-token")
            self.assertEqual(provider_name, "zitadel")
            return type(
                "ExternalIdentityStub",
                (),
                {
                    "issuer": "https://pandects-test-zitadel.example.com",
                    "subject": "zitadel-user-123",
                },
            )()

        backend_app._authenticate_external_identity = _fake_authenticate_external_identity
        try:
            res = client.post(
                "/v1/auth/external-subjects",
                headers=headers,
                json={
                    "provider": "zitadel",
                    "access_token": "zitadel-access-token",
                },
            )
            self.assertEqual(res.status_code, 201)
            link_payload = res.get_json()
            self.assertEqual(link_payload["status"], "linked")
            self.assertEqual(link_payload["link"]["provider"], "zitadel")
            self.assertEqual(
                link_payload["link"]["issuer"],
                "https://pandects-test-zitadel.example.com",
            )
            self.assertEqual(link_payload["link"]["subject"], "zitadel-user-123")

            res = client.post(
                "/v1/auth/external-subjects",
                headers=headers,
                json={
                    "provider": "zitadel",
                    "access_token": "zitadel-access-token",
                },
            )
            self.assertEqual(res.status_code, 200)
            link_payload = res.get_json()
            self.assertEqual(link_payload["status"], "already_linked")

            res = client.get("/v1/auth/external-subjects", headers=headers)
            self.assertEqual(res.status_code, 200)
            list_payload = res.get_json()
            self.assertEqual(len(list_payload["links"]), 1)
            self.assertEqual(
                list_payload["links"][0]["issuer"],
                "https://pandects-test-zitadel.example.com",
            )

            link_id = list_payload["links"][0]["id"]
            res = client.delete(f"/v1/auth/external-subjects/{link_id}", headers=headers)
            self.assertEqual(res.status_code, 200)
            unlink_payload = res.get_json()
            self.assertEqual(unlink_payload["status"], "unlinked")

            res = client.get("/v1/auth/external-subjects", headers=headers)
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.get_json()["links"], [])
        finally:
            backend_app._authenticate_external_identity = original_authenticate

        with self.app.app_context():
            rows = AuthExternalSubject.query.all()
            self.assertEqual(rows, [])

    def test_external_subject_link_conflicts_when_identity_is_owned_by_another_user(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["MCP_IDENTITY_PROVIDER"] = "zitadel"
        client = self.app.test_client()
        for email in ("first-link@example.com", "second-link@example.com"):
            self._create_local_user(email=email)

        with self.app.app_context():
            first = self._require_user("first-link@example.com")
            second = self._require_user("second-link@example.com")
            db.session.add(
                AuthExternalSubject(
                    user_id=first.id,
                    issuer="https://pandects-test-zitadel.example.com",
                    subject="zitadel-user-123",
                )
            )
            db.session.commit()

        headers = {
            "Authorization": f"Bearer {self._issue_bearer_session(email='second-link@example.com')}"
        }

        original_authenticate = backend_app._authenticate_external_identity

        def _fake_authenticate_external_identity(
            *, access_token: str, provider_name: str | None = None
        ):
            self.assertEqual(access_token, "zitadel-access-token")
            self.assertEqual(provider_name, "zitadel")
            return type(
                "ExternalIdentityStub",
                (),
                {
                    "issuer": "https://pandects-test-zitadel.example.com",
                    "subject": "zitadel-user-123",
                },
            )()

        backend_app._authenticate_external_identity = _fake_authenticate_external_identity
        try:
            res = client.post(
                "/v1/auth/external-subjects",
                headers=headers,
                json={
                    "provider": "zitadel",
                    "access_token": "zitadel-access-token",
                },
            )
        finally:
            backend_app._authenticate_external_identity = original_authenticate

        self.assertEqual(res.status_code, 409)

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

    def test_zitadel_oauth_start_and_complete_links_identity(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["MCP_IDENTITY_PROVIDER"] = "zitadel"
        os.environ["MCP_ZITADEL_CLIENT_ID"] = "test-zitadel-client-id"
        os.environ["MCP_OIDC_ISSUER"] = "https://pandects-test-zitadel.example.com"
        os.environ["MCP_OIDC_AUTHORIZATION_ENDPOINT"] = (
            "https://pandects-test-zitadel.example.com/oauth/v2/authorize"
        )
        os.environ["MCP_OIDC_TOKEN_ENDPOINT"] = (
            "https://pandects-test-zitadel.example.com/oauth/v2/token"
        )
        os.environ["MCP_OIDC_AUDIENCE"] = "https://api.pandects.org/mcp"
        os.environ["MCP_ZITADEL_RESOURCE"] = "https://api.pandects.org/mcp"
        os.environ["MCP_ZITADEL_AUDIENCE"] = "https://api.pandects.org/mcp"
        client = self.app.test_client()
        self._create_local_user(email="zitadel-oauth@example.com")
        headers = {
            "Authorization": f"Bearer {self._issue_bearer_session(email='zitadel-oauth@example.com')}"
        }

        original_oauth_fetch_json = backend_app._oauth_fetch_json
        original_authenticate = backend_app._authenticate_external_identity

        def _fake_oauth_fetch_json(url: str, *, data: dict[str, str] | None = None):
            self.assertEqual(url, "https://pandects-test-zitadel.example.com/oauth/v2/token")
            self.assertIsInstance(data, dict)
            assert data is not None
            self.assertEqual(data["grant_type"], "authorization_code")
            self.assertEqual(data["client_id"], "test-zitadel-client-id")
            self.assertEqual(data["code"], "auth-code-123")
            self.assertEqual(
                data["redirect_uri"],
                "http://localhost:8080/auth/zitadel/callback",
            )
            self.assertTrue(data["code_verifier"])
            return {"access_token": "zitadel-access-token"}

        def _fake_authenticate_external_identity(
            *, access_token: str, provider_name: str | None = None
        ):
            self.assertEqual(access_token, "zitadel-access-token")
            self.assertEqual(provider_name, "zitadel")
            return type(
                "ExternalIdentityStub",
                (),
                {
                    "issuer": "https://pandects-test-zitadel.example.com",
                    "subject": "zitadel-user-456",
                },
            )()

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        backend_app._authenticate_external_identity = _fake_authenticate_external_identity
        try:
            res = client.get(
                "/v1/auth/external-subjects/zitadel/start?next=/account",
                headers=headers,
            )
            self.assertEqual(res.status_code, 200)
            start_payload = res.get_json()
            self.assertIsInstance(start_payload, dict)
            authorize_url = start_payload.get("authorize_url")
            self.assertIsInstance(authorize_url, str)

            parsed = urlparse(authorize_url)
            self.assertEqual(
                f"{parsed.scheme}://{parsed.netloc}{parsed.path}",
                "https://pandects-test-zitadel.example.com/oauth/v2/authorize",
            )
            query = parse_qs(parsed.query)
            state = query.get("state", [None])[0]
            self.assertIsInstance(state, str)
            self.assertEqual(query.get("client_id"), ["test-zitadel-client-id"])
            self.assertEqual(
                query.get("redirect_uri"),
                ["http://localhost:8080/auth/zitadel/callback"],
            )

            res = client.post(
                "/v1/auth/external-subjects/zitadel/complete",
                headers=headers,
                json={"code": "auth-code-123", "state": state},
            )
            self.assertEqual(res.status_code, 201)
            complete_payload = res.get_json()
            self.assertEqual(complete_payload["status"], "linked")
            self.assertEqual(complete_payload["return_to"], "/account")
            self.assertEqual(
                complete_payload["link"]["issuer"],
                "https://pandects-test-zitadel.example.com",
            )
            self.assertEqual(complete_payload["link"]["subject"], "zitadel-user-456")
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json
            backend_app._authenticate_external_identity = original_authenticate

        with self.app.app_context():
            rows = AuthExternalSubject.query.all()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].issuer, "https://pandects-test-zitadel.example.com")
            self.assertEqual(rows[0].subject, "zitadel-user-456")

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

    def test_zitadel_website_auth_auto_links_existing_verified_user(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["MCP_IDENTITY_PROVIDER"] = "zitadel"
        os.environ["MCP_ZITADEL_CLIENT_ID"] = "test-zitadel-client-id"
        os.environ["MCP_OIDC_ISSUER"] = "https://pandects-test-zitadel.example.com"
        os.environ["MCP_OIDC_AUTHORIZATION_ENDPOINT"] = (
            "https://pandects-test-zitadel.example.com/oauth/v2/authorize"
        )
        os.environ["MCP_OIDC_TOKEN_ENDPOINT"] = (
            "https://pandects-test-zitadel.example.com/oauth/v2/token"
        )
        client = self.app.test_client()

        with self.app.app_context():
            existing = AuthUser()
            existing.email = "existing-zitadel@example.com"
            existing.password_hash = backend_app.generate_password_hash("password123")
            existing.email_verified_at = backend_app._utc_now()
            db.session.add(existing)
            db.session.flush()
            checked_at = backend_app._utc_now()
            for doc, meta in backend_app._LEGAL_DOCS.items():
                db.session.add(
                    backend_app.LegalAcceptance(
                        user_id=existing.id,
                        document=doc,
                        version=meta["version"],
                        document_hash=meta["sha256"],
                        checked_at=checked_at,
                        submitted_at=checked_at,
                        ip_address=None,
                        user_agent=None,
                    )
                )
            db.session.commit()
            existing_user_id = existing.id

        original_oauth_fetch_json = backend_app._oauth_fetch_json
        original_authenticate = backend_app._authenticate_external_identity

        def _fake_oauth_fetch_json(url: str, *, data: dict[str, str] | None = None):
            self.assertEqual(url, "https://pandects-test-zitadel.example.com/oauth/v2/token")
            self.assertIsInstance(data, dict)
            assert data is not None
            self.assertEqual(data["grant_type"], "authorization_code")
            self.assertEqual(
                data["redirect_uri"],
                "http://localhost:8080/auth/zitadel/callback",
            )
            return {"access_token": "zitadel-website-access-token"}

        def _fake_authenticate_external_identity(
            *, access_token: str, provider_name: str | None = None
        ):
            self.assertEqual(access_token, "zitadel-website-access-token")
            self.assertEqual(provider_name, "zitadel")
            return type(
                "ExternalIdentityStub",
                (),
                {
                    "issuer": "https://pandects-test-zitadel.example.com",
                    "subject": "zitadel-website-user-123",
                    "claims": {
                        "email": "existing-zitadel@example.com",
                        "email_verified": True,
                    },
                },
            )()

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        backend_app._authenticate_external_identity = _fake_authenticate_external_identity
        try:
            res = client.get(
                "/v1/auth/zitadel/start?next=/search&provider=google&prompt=select_account"
            )
            self.assertEqual(res.status_code, 200)
            start_payload = res.get_json()
            self.assertIsInstance(start_payload, dict)
            authorize_url = start_payload.get("authorize_url")
            self.assertIsInstance(authorize_url, str)

            parsed = urlparse(authorize_url)
            query = parse_qs(parsed.query)
            state = query.get("state", [None])[0]
            self.assertIsInstance(state, str)
            self.assertEqual(query.get("prompt"), ["select_account"])

            res = client.post(
                "/v1/auth/zitadel/complete",
                json={"code": "auth-code-website-123", "state": state},
            )
            self.assertEqual(res.status_code, 200)
            payload = res.get_json()
            self.assertEqual(payload["status"], "authenticated")
            self.assertEqual(payload["next_path"], "/search")
            session_token = payload.get("session_token")
            self.assertIsInstance(session_token, str)
            me = client.get(
                "/v1/auth/me", headers={"Authorization": f"Bearer {session_token}"}
            )
            self.assertEqual(me.status_code, 200)
            self.assertEqual(me.get_json()["user"]["email"], "existing-zitadel@example.com")
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json
            backend_app._authenticate_external_identity = original_authenticate

        with self.app.app_context():
            rows = AuthExternalSubject.query.all()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].user_id, existing_user_id)
            engine = db.engines["auth"]
            with engine.begin() as conn:
                signons = conn.execute(
                    text("SELECT provider, action FROM auth_signon_events")
                ).fetchall()
            self.assertEqual(signons, [("zitadel", "login")])

    def test_zitadel_google_intent_auth_auto_links_existing_verified_user(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["MCP_IDENTITY_PROVIDER"] = "zitadel"
        os.environ["AUTH_ZITADEL_API_TOKEN"] = "test-zitadel-api-token"
        os.environ["AUTH_ZITADEL_GOOGLE_IDP_ID"] = "google-idp-123"
        client = self.app.test_client()

        existing_user_id = self._create_local_user(email="existing-google@example.com")
        original_oauth_fetch_json = backend_app._oauth_fetch_json

        sent_verification = {"called": False}

        def _fake_oauth_fetch_json(
            url: str,
            *,
            data: dict[str, str] | None = None,
            json_body: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            method: str | None = None,
        ):
            self.assertEqual(headers, {"Authorization": "Bearer test-zitadel-api-token"})
            self.assertIsNone(data)
            if url == "https://pandects-test-zitadel.example.com/v2/idp_intents":
                self.assertEqual(method, "POST")
                self.assertEqual(
                    json_body,
                    {
                        "idpId": "google-idp-123",
                        "urls": {
                            "successUrl": "http://localhost:8080/auth/zitadel/callback",
                            "failureUrl": "http://localhost:8080/auth/zitadel/callback",
                        },
                    },
                )
                return {"authUrl": "https://accounts.google.com/o/oauth2/v2/auth?client_id=test"}
            if url == "https://pandects-test-zitadel.example.com/v2/users/zitadel-user-google-123":
                self.assertEqual(method, "GET")
                self.assertIsNone(json_body)
                return {
                    "user": {
                        "human": {
                            "profile": {
                                "displayName": "Existing Google",
                                "givenName": "Existing",
                                "familyName": "Google",
                            },
                            "email": {
                                "email": "existing-google@example.com",
                                "isVerified": True,
                            },
                        }
                    }
                }
            self.fail(f"Unexpected ZITADEL API request: {url}")

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        try:
            res = client.get("/v1/auth/zitadel/google/start?next=/search")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(
                res.get_json()["authorize_url"],
                "https://accounts.google.com/o/oauth2/v2/auth?client_id=test",
            )

            res = client.post(
                "/v1/auth/zitadel/complete",
                json={
                    "intent_id": "intent-123",
                    "intent_token": "intent-token-123",
                    "user_id": "zitadel-user-google-123",
                },
            )
            self.assertEqual(res.status_code, 200)
            payload = res.get_json()
            self.assertEqual(payload["status"], "authenticated")
            self.assertEqual(payload["next_path"], "/search")
            session_token = payload.get("session_token")
            self.assertIsInstance(session_token, str)
            me = client.get(
                "/v1/auth/me", headers={"Authorization": f"Bearer {session_token}"}
            )
            self.assertEqual(me.status_code, 200)
            self.assertEqual(me.get_json()["user"]["email"], "existing-google@example.com")
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json

        with self.app.app_context():
            rows = AuthExternalSubject.query.all()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].user_id, existing_user_id)
            self.assertEqual(rows[0].issuer, "https://pandects-test-zitadel.example.com")
            self.assertEqual(rows[0].subject, "zitadel-user-google-123")
            engine = db.engines["auth"]
            with engine.begin() as conn:
                signons = conn.execute(
                    text("SELECT provider, action FROM auth_signon_events")
                ).fetchall()
            self.assertEqual(signons, [("zitadel", "login")])

    def test_zitadel_google_intent_callback_uses_existing_linked_subject_without_retrieval(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["MCP_IDENTITY_PROVIDER"] = "zitadel"
        os.environ["AUTH_ZITADEL_API_TOKEN"] = "test-zitadel-api-token"
        os.environ["AUTH_ZITADEL_GOOGLE_IDP_ID"] = "google-idp-123"
        client = self.app.test_client()

        existing_user_id = self._create_local_user(email="linked-google@example.com")
        with self.app.app_context():
            db.session.add(
                AuthExternalSubject(
                    user_id=existing_user_id,
                    issuer="https://pandects-test-zitadel.example.com",
                    subject="zitadel-user-linked-123",
                )
            )
            db.session.commit()

        original_oauth_fetch_json = backend_app._oauth_fetch_json

        def _fake_oauth_fetch_json(
            url: str,
            *,
            data: dict[str, str] | None = None,
            json_body: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            method: str | None = None,
        ):
            if url == "https://pandects-test-zitadel.example.com/v2/idp_intents":
                return {"authUrl": "https://accounts.google.com/o/oauth2/v2/auth?client_id=test"}
            self.fail("Existing linked ZITADEL user should not retrieve the idp intent payload.")

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        try:
            start = client.get("/v1/auth/zitadel/google/start?next=/account")
            self.assertEqual(start.status_code, 200)
            res = client.post(
                "/v1/auth/zitadel/complete",
                json={
                    "intent_id": "intent-123",
                    "intent_token": "intent-token-123",
                    "user_id": "zitadel-user-linked-123",
                },
            )
            self.assertEqual(res.status_code, 200)
            payload = res.get_json()
            self.assertEqual(payload["status"], "authenticated")
            session_token = payload.get("session_token")
            self.assertIsInstance(session_token, str)
            me = client.get(
                "/v1/auth/me", headers={"Authorization": f"Bearer {session_token}"}
            )
            self.assertEqual(me.status_code, 200)
            self.assertEqual(me.get_json()["user"]["email"], "linked-google@example.com")
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json

    def test_zitadel_google_intent_callback_reuses_existing_zitadel_user_after_conflict(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["MCP_IDENTITY_PROVIDER"] = "zitadel"
        os.environ["AUTH_ZITADEL_API_TOKEN"] = "test-zitadel-api-token"
        os.environ["AUTH_ZITADEL_GOOGLE_IDP_ID"] = "google-idp-123"
        client = self.app.test_client()

        existing_user_id = self._create_local_user(email="conflict-google@example.com")
        original_oauth_fetch_json = backend_app._oauth_fetch_json

        def _fake_oauth_fetch_json(
            url: str,
            *,
            data: dict[str, str] | None = None,
            json_body: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            method: str | None = None,
        ):
            self.assertEqual(headers, {"Authorization": "Bearer test-zitadel-api-token"})
            self.assertIsNone(data)
            if url == "https://pandects-test-zitadel.example.com/v2/idp_intents":
                return {"authUrl": "https://accounts.google.com/o/oauth2/v2/auth?client_id=test"}
            if url == "https://pandects-test-zitadel.example.com/v2/idp_intents/intent-123":
                self.assertEqual(method, "POST")
                self.assertEqual(json_body, {"idpIntentToken": "intent-token-123"})
                return {
                    "idpInformation": {
                        "idpId": "google-idp-123",
                        "rawInformation": {
                            "User": {
                                "sub": "google-sub-123",
                                "email": "conflict-google@example.com",
                                "email_verified": True,
                                "name": "Conflict Google",
                            }
                        },
                    }
                }
            if url == "https://pandects-test-zitadel.example.com/v2/users/human":
                raise Conflict(description="User already exists")
            if url == "https://pandects-test-zitadel.example.com/v2/users":
                self.assertEqual(method, "POST")
                self.assertEqual(
                    json_body,
                    {
                        "pagination": {"limit": 1},
                        "queries": [
                            {
                                "loginNameQuery": {
                                    "loginName": "conflict-google@example.com",
                                    "method": "TEXT_QUERY_METHOD_EQUALS",
                                }
                            }
                        ],
                    },
                )
                return {"result": [{"userId": "zitadel-existing-conflict-user"}]}
            if url == "https://pandects-test-zitadel.example.com/v2/users/zitadel-existing-conflict-user":
                self.assertEqual(method, "GET")
                return {
                    "user": {
                        "human": {
                            "profile": {
                                "displayName": "Conflict Google",
                            },
                            "email": {
                                "email": "conflict-google@example.com",
                                "isVerified": True,
                            },
                        }
                    }
                }
            self.fail(f"Unexpected ZITADEL API request: {url}")

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        try:
            start = client.get("/v1/auth/zitadel/google/start?next=/account")
            self.assertEqual(start.status_code, 200)
            res = client.post(
                "/v1/auth/zitadel/complete",
                json={
                    "intent_id": "intent-123",
                    "intent_token": "intent-token-123",
                },
            )
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json

        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload["status"], "authenticated")
        self.assertEqual(payload["user"]["email"], "conflict-google@example.com")

        with self.app.app_context():
            link = AuthExternalSubject.query.filter_by(
                issuer="https://pandects-test-zitadel.example.com",
                subject="zitadel-existing-conflict-user",
            ).first()
            self.assertIsNotNone(link)
            assert link is not None
            self.assertEqual(link.user_id, existing_user_id)

    def test_zitadel_google_start_mints_api_token_from_private_key_jwt(self):
        os.environ["MCP_OIDC_ISSUER"] = "https://pandects-test-zitadel.example.com"
        os.environ["MCP_OIDC_TOKEN_ENDPOINT"] = (
            "https://pandects-test-zitadel.example.com/oauth/v2/token"
        )
        os.environ["AUTH_ZITADEL_API_TOKEN"] = ""
        os.environ["AUTH_ZITADEL_API_CLIENT_ID"] = "service-client-id"
        os.environ["AUTH_ZITADEL_API_KEY_ID"] = "service-key-id"
        os.environ["AUTH_ZITADEL_API_PRIVATE_KEY"] = (
            "-----BEGIN RSA PRIVATE KEY-----\\n"
            "MIIEowIBAAKCAQEAsgennGq2EFP2OIe0JX5yZ9imwIh/FVW40GnvfPzh6KmeAHuP\\n"
            "v1yZ49pNkgA60UqdoWULGyjy/oK+DDg2bPsKSrBgldvRPUHdKxr0a3dQp8C9i8UX\\n"
            "EMMR2CFEPP+063MBKcbdJNKRfiXIO/2DmjQOYVSlmrMoUuXJz1/0YmFsMj7A5Bps\\n"
            "ic5X5J5jSEFpoAeMMZqoCupYjRLioL7RNv+JsPjUVb9h5CQ6ssC4jIpMQHsidTej\\n"
            "iYQlQQeFL3zmU81BzQJ4lM7qdKjFXrkK/0qHe4N5KEmSjxNGc9OkEUhzWwVP25fE\\n"
            "YW6JkqvLTK2ZSDd7T0FeDWbPGEvKM0SgwOERbwIDAQABAoIBAEI7AMr7FAJdCgOb\\n"
            "0JQGR4+ElPyZixWnz1qRnovUFEMXHvW1AhRB4epXY3ZGaZtE9AF/8rLG+CdhAkzP\\n"
            "eMfwWLLSjQwTO/NbFmmb3IRCXhoaZSmjR+Jvf6r1LUq4IllZbnxZRBoX0BKrAaP3\\n"
            "u0bZyOPCtt0ne3/jhsGERAl5juPdv7MUjbUMEp4VVCZcJxoYXOF4P6AY1vcYdLL0\\n"
            "Pg80ngp6q4mx/8ns6Kzbt4rdex1nCji6GIgmxa9gwielJe3YMiNvTKvOBN2ZdqRc\\n"
            "gjGTiruKKExtMg7reNptWeLHPwozOL8xwnjvaL9q5XasOHB8LYbwppgVS71Z98hw\\n"
            "mqN50KECgYEA3h7U/bogsb1K+Gzd3lCogA+EM2BBY8kufyv+wauzFiLLU7JifOAH\\n"
            "eR7bjWQMVi2m2iw24FT+P0an5swxh7FL0QCx2gkt3LYGZH+KrWjvZ4DhX5ZoG2dG\\n"
            "eMIEVhSrvujzTqClnGydhFeBgbqMcC6EZ5OmVR8qwjaZ+aQtoGofed8CgYEAzS82\\n"
            "np8xFK9tWZPJS2zx4R3SyEjr2c/705zO6wU/sPqwBF32fmPatHnnU87XzJxAV3MY\\n"
            "FIvA8QMNZfTtCtWjo6SUEfBpQqBmwcnX9JYuI8d3zBLo56P6SrKoe6D6ODc0Zz5i\\n"
            "dPSzxVR+r/39F4l5z2BxYuhgcfKXlZY3HAQYenECgYAXDbgphXHzQKRRWGtGsbRr\\n"
            "ZjDgbDMdOjo7NMPCMiHqQD4+N5uFPnNIHO3IpQOqxh41MrWXXvrsclbm23agkMQ/\\n"
            "swTCjoVWDQZo09v+149RfMzncOLpRTTJP8nXbVnN/LuUA5RswdEvdS2Z21TMJ+fS\\n"
            "ID75QrzbX3Nnt6SMq7cMnQKBgC4NG7AsQILJALzrG3GvSPZikC1dmHmxYW7UMeiz\\n"
            "q+DX0uuX/zvMw4hgF9hKg0qsAxDdhxkNaMdvDPHGL/GPk4Ol64m/MJDAmW+DEtIV\\n"
            "ZtOm8C9ASz+6IPHk+UWOErrNQRiu+sAPL83pMenkEorW0x0FI020o/jPHtB2/3Vt\\n"
            "QeUxAoGBANIj2oXYRWTR+xTYRr/ZS7ZQX0mQcKlFQ4Hqx91QNXi3nKqSL8jixxVp\\n"
            "2bJTZxUcoB/E2mtzsgVN9lAiHo3wAziDJ8NMNPmb0RcSXxYrEddav8vNxGhDibyr\\n"
            "PVjZ5he+nSAUrEl1esf5mvtrvPgTQbUnWwwQvz+7wqcnUU8ssQ/M\\n"
            "-----END RSA PRIVATE KEY-----\\n"
        )
        os.environ["AUTH_ZITADEL_GOOGLE_IDP_ID"] = "google-idp-123"
        client = self.app.test_client()
        original_oauth_fetch_json = backend_app._oauth_fetch_json

        def _fake_oauth_fetch_json(
            url: str,
            *,
            data: dict[str, str] | None = None,
            json_body: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            method: str | None = None,
        ):
            if url == "https://pandects-test-zitadel.example.com/oauth/v2/token":
                self.assertEqual(data["grant_type"], "client_credentials")
                self.assertEqual(data["scope"], "urn:zitadel:iam:org:project:id:zitadel:aud")
                self.assertEqual(
                    data["client_assertion_type"],
                    "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
                )
                self.assertIsInstance(data.get("client_assertion"), str)
                self.assertIsNone(json_body)
                self.assertIsNone(headers)
                self.assertIsNone(method)
                return {"access_token": "service-api-token", "expires_in": 300}
            if url == "https://pandects-test-zitadel.example.com/v2/idp_intents":
                self.assertIsNone(data)
                self.assertEqual(method, "POST")
                self.assertEqual(headers, {"Authorization": "Bearer service-api-token"})
                self.assertEqual(json_body["idpId"], "google-idp-123")
                return {"authUrl": "https://accounts.google.com/o/oauth2/v2/auth?client_id=test"}
            self.fail(f"Unexpected ZITADEL request: {url}")

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        try:
            res = client.get("/v1/auth/zitadel/google/start?next=/search")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(
                res.get_json()["authorize_url"],
                "https://accounts.google.com/o/oauth2/v2/auth?client_id=test",
            )
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json

    def test_zitadel_website_auth_new_user_requires_legal_then_finalizes(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["MCP_IDENTITY_PROVIDER"] = "zitadel"
        os.environ["MCP_ZITADEL_CLIENT_ID"] = "test-zitadel-client-id"
        os.environ["MCP_OIDC_ISSUER"] = "https://pandects-test-zitadel.example.com"
        os.environ["MCP_OIDC_AUTHORIZATION_ENDPOINT"] = (
            "https://pandects-test-zitadel.example.com/oauth/v2/authorize"
        )
        os.environ["MCP_OIDC_TOKEN_ENDPOINT"] = (
            "https://pandects-test-zitadel.example.com/oauth/v2/token"
        )
        client = self.app.test_client()

        original_oauth_fetch_json = backend_app._oauth_fetch_json
        original_authenticate = backend_app._authenticate_external_identity

        def _fake_oauth_fetch_json(url: str, *, data: dict[str, str] | None = None):
            self.assertEqual(url, "https://pandects-test-zitadel.example.com/oauth/v2/token")
            return {"access_token": "zitadel-website-access-token-2"}

        def _fake_authenticate_external_identity(
            *, access_token: str, provider_name: str | None = None
        ):
            self.assertEqual(access_token, "zitadel-website-access-token-2")
            self.assertEqual(provider_name, "zitadel")
            return type(
                "ExternalIdentityStub",
                (),
                {
                    "issuer": "https://pandects-test-zitadel.example.com",
                    "subject": "zitadel-website-user-456",
                    "claims": {
                        "email": "new-zitadel-user@example.com",
                        "email_verified": True,
                    },
                },
            )()

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        backend_app._authenticate_external_identity = _fake_authenticate_external_identity
        try:
            res = client.get("/v1/auth/zitadel/start?next=/account&provider=email&prompt=create")
            self.assertEqual(res.status_code, 200)
            query = parse_qs(urlparse(res.get_json()["authorize_url"]).query)
            state = query.get("state", [None])[0]
            self.assertIsInstance(state, str)
            self.assertEqual(query.get("prompt"), ["create"])

            res = client.post(
                "/v1/auth/zitadel/complete",
                json={"code": "auth-code-website-456", "state": state},
            )
            self.assertEqual(res.status_code, 200)
            payload = res.get_json()
            self.assertEqual(payload["status"], "legal_required")
            self.assertEqual(payload["user"]["email"], "new-zitadel-user@example.com")

            finalize = client.post(
                "/v1/auth/zitadel/finalize",
                json={
                    "legal": {
                        "checked_at_ms": 1700000000000,
                        "docs": ["tos", "privacy", "license"],
                    }
                },
            )
            self.assertEqual(finalize.status_code, 200)
            finalize_payload = finalize.get_json()
            self.assertEqual(finalize_payload["status"], "authenticated")
            session_token = finalize_payload.get("session_token")
            self.assertIsInstance(session_token, str)
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json
            backend_app._authenticate_external_identity = original_authenticate

    def test_password_login_auto_links_existing_verified_user_by_email(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()

        with self.app.app_context():
            existing = AuthUser()
            existing.email = "existing-password@example.com"
            existing.password_hash = backend_app.generate_password_hash("password123")
            existing.email_verified_at = backend_app._utc_now()
            db.session.add(existing)
            db.session.flush()
            checked_at = backend_app._utc_now()
            for doc, meta in backend_app._LEGAL_DOCS.items():
                db.session.add(
                    backend_app.LegalAcceptance(
                        user_id=existing.id,
                        document=doc,
                        version=meta["version"],
                        document_hash=meta["sha256"],
                        checked_at=checked_at,
                        submitted_at=checked_at,
                        ip_address=None,
                        user_agent=None,
                    )
                )
            db.session.commit()
            existing_user_id = existing.id

        original_oauth_fetch_json = backend_app._oauth_fetch_json

        def _fake_oauth_fetch_json(
            url: str,
            *,
            data: dict[str, str] | None = None,
            json_body: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            method: str | None = None,
        ):
            self.assertEqual(headers, {"Authorization": "Bearer test-zitadel-api-token"})
            self.assertIsNone(data)
            self.assertEqual(url, "https://pandects-test-zitadel.example.com/v2/sessions")
            self.assertEqual(method, "POST")
            self.assertEqual(
                json_body,
                {"checks": {"user": {"loginName": "existing-password@example.com"}}},
            )
            return {"sessionId": "session-123"}

        def _fake_oauth_fetch_json_second(
            url: str,
            *,
            data: dict[str, str] | None = None,
            json_body: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            method: str | None = None,
        ):
            self.assertEqual(headers, {"Authorization": "Bearer test-zitadel-api-token"})
            self.assertIsNone(data)
            if url == "https://pandects-test-zitadel.example.com/v2/sessions":
                self.assertEqual(method, "POST")
                return {"sessionId": "session-123"}
            if url == "https://pandects-test-zitadel.example.com/v2/sessions/session-123":
                if method == "PATCH":
                    self.assertEqual(
                        json_body,
                        {"checks": {"password": {"password": "password123"}}},
                    )
                    return {"sessionToken": "zitadel-session-token"}
                self.assertEqual(method, "GET")
                return {
                    "session": {
                        "factors": {
                            "user": {
                                "id": "zitadel-user-abc",
                                "loginName": "existing-password@example.com",
                                "displayName": "Existing Password User",
                            },
                            "password": {"verifiedAt": "2026-04-04T00:00:00Z"},
                        }
                    }
                }
            if url == "https://pandects-test-zitadel.example.com/v2/users/zitadel-user-abc":
                self.assertEqual(method, "GET")
                return {
                    "user": {
                        "human": {
                            "email": {
                                "email": "existing-password@example.com",
                                "isVerified": True,
                            },
                            "profile": {
                                "displayName": "Existing Password User",
                            },
                        }
                    }
                }
            self.fail(f"Unexpected URL: {url}")

        os.environ["AUTH_ZITADEL_API_TOKEN"] = "test-zitadel-api-token"
        backend_app._oauth_fetch_json = _fake_oauth_fetch_json_second
        try:
            res = client.post(
                "/v1/auth/login/password",
                json={
                    "email": "existing-password@example.com",
                    "password": "password123",
                    "next": "/search",
                },
            )
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json

        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload["status"], "authenticated")
        self.assertEqual(payload["next_path"], "/search")
        self.assertEqual(payload["user"]["id"], existing_user_id)
        self.assertIsInstance(payload.get("session_token"), str)

        with self.app.app_context():
            link = AuthExternalSubject.query.filter_by(
                issuer="https://pandects-test-zitadel.example.com",
                subject="zitadel-user-abc",
            ).first()
            self.assertIsNotNone(link)
            assert link is not None
            self.assertEqual(link.user_id, existing_user_id)

    def test_password_login_migrates_existing_local_password_user_when_zitadel_user_missing(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["AUTH_ZITADEL_API_TOKEN"] = "test-zitadel-api-token"
        os.environ["AUTH_ZITADEL_PROJECT_ID"] = "pandects-project-123"
        client = self.app.test_client()
        existing_user_id = self._create_local_user(
            email="legacy-password@example.com",
            password="password123",
            verified=True,
            legal=True,
        )

        original_oauth_fetch_json = backend_app._oauth_fetch_json

        def _fake_oauth_fetch_json(
            url: str,
            *,
            data: dict[str, str] | None = None,
            json_body: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            method: str | None = None,
        ):
            self.assertEqual(headers, {"Authorization": "Bearer test-zitadel-api-token"})
            self.assertIsNone(data)
            if url == "https://pandects-test-zitadel.example.com/v2/sessions":
                raise Unauthorized(description="Invalid credentials.")
            if url == "https://pandects-test-zitadel.example.com/v2/users/human":
                self.assertEqual(method, "POST")
                self.assertIsInstance(json_body, dict)
                assert json_body is not None
                self.assertEqual(json_body["username"], "legacy-password@example.com")
                return {"userId": "zitadel-user-migrated"}
            if url == "https://pandects-test-zitadel.example.com/management/v1/users/grants/_search":
                self.assertEqual(method, "POST")
                return {"result": []}
            if url == "https://pandects-test-zitadel.example.com/management/v1/users/grants":
                self.assertEqual(method, "POST")
                return {"id": "grant-123"}
            self.fail(f"Unexpected URL: {url}")

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        try:
            res = client.post(
                "/v1/auth/login/password",
                json={
                    "email": "legacy-password@example.com",
                    "password": "password123",
                },
            )
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json
            os.environ.pop("AUTH_ZITADEL_PROJECT_ID", None)

        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload["status"], "authenticated")
        self.assertEqual(payload["user"]["id"], existing_user_id)

        with self.app.app_context():
            link = AuthExternalSubject.query.filter_by(
                issuer="https://pandects-test-zitadel.example.com",
                subject="zitadel-user-migrated",
            ).first()
            self.assertIsNotNone(link)

    def test_password_signup_requires_legal_then_verifies_then_finalizes(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["AUTH_ZITADEL_API_TOKEN"] = "test-zitadel-api-token"
        client = self.app.test_client()

        original_oauth_fetch_json = backend_app._oauth_fetch_json
        sent_verification = {"called": False}

        def _fake_oauth_fetch_json(
            url: str,
            *,
            data: dict[str, str] | None = None,
            json_body: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            method: str | None = None,
        ):
            self.assertEqual(headers, {"Authorization": "Bearer test-zitadel-api-token"})
            self.assertIsNone(data)
            if url == "https://pandects-test-zitadel.example.com/v2/users/zitadel-new-user/email/send":
                self.assertEqual(method, "POST")
                self.assertEqual(
                    json_body,
                    {
                        "sendCode": {
                            "urlTemplate": "http://localhost:8080/verify-email?user_id={{.UserID}}&code={{.Code}}&org_id={{.OrgID}}"
                        }
                    },
                )
                sent_verification["called"] = True
                return {"details": {}}
            self.assertEqual(url, "https://pandects-test-zitadel.example.com/v2/users/human")
            self.assertEqual(method, "POST")
            self.assertIsInstance(json_body, dict)
            assert json_body is not None
            self.assertEqual(json_body["username"], "new-password-user@example.com")
            self.assertEqual(json_body["profile"]["givenName"], "New")
            self.assertEqual(json_body["profile"]["familyName"], "User")
            self.assertEqual(json_body["email"]["email"], "new-password-user@example.com")
            self.assertIn("/verify-email", json_body["email"]["sendCode"]["urlTemplate"])
            return {"userId": "zitadel-new-user"}

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        try:
            res = client.post(
                "/v1/auth/signup/password",
                json={
                    "email": "new-password-user@example.com",
                    "password": "Secr3tP4ss!",
                    "first_name": "New",
                    "last_name": "User",
                    "next": "/account",
                },
            )
            self.assertEqual(res.status_code, 200)
            payload = res.get_json()
            self.assertEqual(payload["status"], "legal_required")

            finalize = client.post(
                "/v1/auth/zitadel/finalize",
                json={
                    "legal": {
                        "checked_at_ms": 1_715_000_000_000,
                        "docs": ["tos", "privacy", "license"],
                    }
                },
            )
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json

        self.assertEqual(finalize.status_code, 200)
        finalize_payload = finalize.get_json()
        self.assertEqual(finalize_payload["status"], "verification_required")
        self.assertTrue(sent_verification["called"])

    def test_password_signup_assigns_default_zitadel_project_roles_when_project_is_configured(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["AUTH_ZITADEL_API_TOKEN"] = "test-zitadel-api-token"
        os.environ["AUTH_ZITADEL_PROJECT_ID"] = "pandects-project-123"
        client = self.app.test_client()

        original_oauth_fetch_json = backend_app._oauth_fetch_json
        grant_calls: list[tuple[str, str | None, dict[str, object] | None]] = []

        def _fake_oauth_fetch_json(
            url: str,
            *,
            data: dict[str, str] | None = None,
            json_body: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            method: str | None = None,
        ):
            self.assertEqual(headers, {"Authorization": "Bearer test-zitadel-api-token"})
            self.assertIsNone(data)
            if url == "https://pandects-test-zitadel.example.com/v2/users/human":
                return {"userId": "zitadel-role-grant-user"}
            if url == "https://pandects-test-zitadel.example.com/management/v1/users/grants/_search":
                grant_calls.append((url, method, json_body))
                return {"result": []}
            if url == "https://pandects-test-zitadel.example.com/management/v1/users/grants":
                grant_calls.append((url, method, json_body))
                return {"id": "grant-123"}
            if url == "https://pandects-test-zitadel.example.com/v2/users/zitadel-role-grant-user/email/send":
                return {"details": {}}
            self.fail(f"Unexpected URL: {url}")

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        try:
            res = client.post(
                "/v1/auth/signup/password",
                json={
                    "email": "role-grant@example.com",
                    "password": "Secr3tP4ss!",
                    "next": "/account",
                },
            )
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json
            os.environ.pop("AUTH_ZITADEL_PROJECT_ID", None)

        self.assertEqual(res.status_code, 200)
        signup_payload = res.get_json()
        self.assertEqual(signup_payload["status"], "legal_required")
        self.assertEqual(len(grant_calls), 2)
        self.assertEqual(grant_calls[0][0], "https://pandects-test-zitadel.example.com/management/v1/users/grants/_search")
        self.assertEqual(grant_calls[0][1], "POST")
        self.assertEqual(
            grant_calls[0][2],
            {
                "queries": [
                    {"user_id_query": {"user_id": "zitadel-role-grant-user"}},
                    {"project_id_query": {"project_id": "pandects-project-123"}},
                ]
            },
        )
        self.assertEqual(grant_calls[1][0], "https://pandects-test-zitadel.example.com/management/v1/users/grants")
        self.assertEqual(grant_calls[1][1], "POST")
        self.assertEqual(
            grant_calls[1][2],
            {
                "userId": "zitadel-role-grant-user",
                "projectId": "pandects-project-123",
                "roleKeys": [
                    "agreements_read",
                    "agreements_read_fulltext",
                    "agreements_search",
                    "sections_search",
                ],
            },
        )

    def test_password_signup_resumes_incomplete_account_instead_of_conflicting(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        with self.app.app_context():
            user_id = self._create_local_user(
                email="resume-signup@example.com",
                password="Secr3tP4ss!",
                verified=True,
                legal=False,
            )
            db.session.add(
                AuthExternalSubject(
                    user_id=user_id,
                    issuer="https://pandects-test-zitadel.example.com",
                    subject="zitadel-resume-user",
                )
            )
            db.session.commit()

        original_oauth_fetch_json = backend_app._oauth_fetch_json

        def _fail_oauth_fetch_json(*args, **kwargs):
            self.fail("Signup resume should not create or fetch a remote auth user.")

        backend_app._oauth_fetch_json = _fail_oauth_fetch_json
        try:
            res = client.post(
                "/v1/auth/signup/password",
                json={
                    "email": "resume-signup@example.com",
                    "password": "Secr3tP4ss!",
                    "first_name": "Resume",
                    "last_name": "Signup",
                    "next": "/account",
                },
            )
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json

        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload["status"], "legal_required")
        self.assertEqual(payload["user"]["email"], "resume-signup@example.com")

    def test_password_signup_pending_no_legal_and_unverified_returns_legal_required(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["AUTH_ZITADEL_API_TOKEN"] = "test-zitadel-api-token"
        client = self.app.test_client()
        with self.app.app_context():
            user_id = self._create_local_user(
                email="pending-no-legal@example.com",
                password="Secr3tP4ss!",
                verified=False,
                legal=False,
            )
            db.session.add(
                AuthExternalSubject(
                    user_id=user_id,
                    issuer="https://pandects-test-zitadel.example.com",
                    subject="zitadel-pending-no-legal-user",
                )
            )
            db.session.commit()

        original_oauth_fetch_json = backend_app._oauth_fetch_json

        def _fake_oauth_fetch_json(
            url: str,
            *,
            data: dict[str, str] | None = None,
            json_body: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            method: str | None = None,
        ):
            self.assertEqual(headers, {"Authorization": "Bearer test-zitadel-api-token"})
            self.assertIsNone(data)
            self.assertEqual(
                url,
                "https://pandects-test-zitadel.example.com/v2/users/zitadel-pending-no-legal-user",
            )
            self.assertEqual(method, "GET")
            return {
                "user": {
                    "human": {
                        "email": {
                            "email": "pending-no-legal@example.com",
                            "isVerified": False,
                        },
                        "profile": {
                            "displayName": "Pending No Legal",
                        },
                    }
                }
            }

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        try:
            res = client.post(
                "/v1/auth/signup/password",
                json={
                    "email": "pending-no-legal@example.com",
                    "password": "DifferentP4ss!",
                    "next": "/account",
                },
            )
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json

        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload["status"], "legal_required")
        self.assertEqual(payload["user"]["email"], "pending-no-legal@example.com")

    def test_password_signup_recreates_missing_remote_user_for_pending_account(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["AUTH_ZITADEL_API_TOKEN"] = "test-zitadel-api-token"
        client = self.app.test_client()
        with self.app.app_context():
            user_id = self._create_local_user(
                email="pending-missing-remote@example.com",
                password="OldP4ssword!",
                verified=False,
                legal=False,
            )
            db.session.add(
                AuthExternalSubject(
                    user_id=user_id,
                    issuer="https://pandects-test-zitadel.example.com",
                    subject="zitadel-stale-pending-user",
                )
            )
            db.session.commit()

        original_oauth_fetch_json = backend_app._oauth_fetch_json

        def _fake_oauth_fetch_json(
            url: str,
            *,
            data: dict[str, str] | None = None,
            json_body: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            method: str | None = None,
        ):
            self.assertEqual(headers, {"Authorization": "Bearer test-zitadel-api-token"})
            self.assertIsNone(data)
            if url == "https://pandects-test-zitadel.example.com/v2/users/zitadel-stale-pending-user":
                self.assertEqual(method, "GET")
                raise NotFound(description="Not found")
            if url == "https://pandects-test-zitadel.example.com/v2/users/human":
                self.assertEqual(method, "POST")
                self.assertEqual(json_body["username"], "pending-missing-remote@example.com")
                self.assertEqual(json_body["password"]["password"], "NewP4ssword!")
                return {"userId": "zitadel-recreated-pending-user"}
            self.fail(f"Unexpected URL: {url}")

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        try:
            res = client.post(
                "/v1/auth/signup/password",
                json={
                    "email": "pending-missing-remote@example.com",
                    "password": "NewP4ssword!",
                    "next": "/account",
                },
            )
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json

        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload["status"], "legal_required")
        self.assertEqual(payload["user"]["email"], "pending-missing-remote@example.com")

        with self.app.app_context():
            link = AuthExternalSubject.query.filter_by(
                user_id=user_id,
                issuer="https://pandects-test-zitadel.example.com",
            ).one()
            self.assertEqual(link.subject, "zitadel-recreated-pending-user")

    def test_password_signup_existing_unverified_user_resends_verification(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["AUTH_ZITADEL_API_TOKEN"] = "test-zitadel-api-token"
        client = self.app.test_client()
        with self.app.app_context():
            user_id = self._create_local_user(
                email="verify-again@example.com",
                password="Secr3tP4ss!",
                verified=False,
                legal=True,
            )
            db.session.add(
                AuthExternalSubject(
                    user_id=user_id,
                    issuer="https://pandects-test-zitadel.example.com",
                    subject="zitadel-verify-again-user",
                )
            )
            db.session.commit()

        original_oauth_fetch_json = backend_app._oauth_fetch_json

        def _fake_oauth_fetch_json(
            url: str,
            *,
            data: dict[str, str] | None = None,
            json_body: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            method: str | None = None,
        ):
            self.assertEqual(headers, {"Authorization": "Bearer test-zitadel-api-token"})
            self.assertIsNone(data)
            if url == "https://pandects-test-zitadel.example.com/v2/users/zitadel-verify-again-user":
                self.assertEqual(method, "GET")
                return {
                    "user": {
                        "human": {
                            "email": {
                                "email": "verify-again@example.com",
                                "isVerified": False,
                            },
                            "profile": {
                                "displayName": "Verify Again",
                            },
                        }
                    }
                }
            self.assertEqual(
                url,
                "https://pandects-test-zitadel.example.com/v2/users/zitadel-verify-again-user/email/send",
            )
            self.assertEqual(method, "POST")
            self.assertEqual(
                json_body,
                {
                    "sendCode": {
                        "urlTemplate": "http://localhost:8080/verify-email?user_id={{.UserID}}&code={{.Code}}&org_id={{.OrgID}}"
                    }
                },
            )
            return {"details": {}}

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        try:
            res = client.post(
                "/v1/auth/signup/password",
                json={
                    "email": "verify-again@example.com",
                    "password": "Secr3tP4ss!",
                    "next": "/account",
                },
            )
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json

        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload["status"], "verification_required")
        self.assertEqual(payload["user"]["email"], "verify-again@example.com")

    def test_password_reset_request_migrates_local_user_and_requests_reset_link(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["AUTH_ZITADEL_API_TOKEN"] = "test-zitadel-api-token"
        os.environ["AUTH_ZITADEL_PROJECT_ID"] = "pandects-project-123"
        client = self.app.test_client()
        self._create_local_user(
            email="reset-me@example.com",
            password="password123",
            verified=True,
            legal=True,
        )

        original_oauth_fetch_json = backend_app._oauth_fetch_json
        seen_reset = {"called": False}

        def _fake_oauth_fetch_json(
            url: str,
            *,
            data: dict[str, str] | None = None,
            json_body: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            method: str | None = None,
        ):
            self.assertEqual(headers, {"Authorization": "Bearer test-zitadel-api-token"})
            self.assertIsNone(data)
            if url == "https://pandects-test-zitadel.example.com/v2/users/human":
                return {"userId": "zitadel-reset-user"}
            if url == "https://pandects-test-zitadel.example.com/management/v1/users/grants/_search":
                self.assertEqual(method, "POST")
                return {"result": []}
            if url == "https://pandects-test-zitadel.example.com/management/v1/users/grants":
                self.assertEqual(method, "POST")
                return {"id": "grant-123"}
            if url == "https://pandects-test-zitadel.example.com/v2/users/zitadel-reset-user/password_reset":
                self.assertEqual(method, "POST")
                self.assertIsInstance(json_body, dict)
                assert json_body is not None
                self.assertIn("/reset-password/confirm", json_body["sendLink"]["urlTemplate"])
                seen_reset["called"] = True
                return {"details": {}}
            self.fail(f"Unexpected URL: {url}")

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        try:
            res = client.post(
                "/v1/auth/password-reset/request",
                json={"email": "reset-me@example.com"},
            )
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json
            os.environ.pop("AUTH_ZITADEL_PROJECT_ID", None)

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), {"status": "requested"})
        self.assertTrue(seen_reset["called"])

    def test_password_login_requires_verified_email(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["AUTH_ZITADEL_API_TOKEN"] = "test-zitadel-api-token"
        client = self.app.test_client()

        original_oauth_fetch_json = backend_app._oauth_fetch_json

        def _fake_oauth_fetch_json(
            url: str,
            *,
            data: dict[str, str] | None = None,
            json_body: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            method: str | None = None,
        ):
            self.assertEqual(headers, {"Authorization": "Bearer test-zitadel-api-token"})
            self.assertIsNone(data)
            if url == "https://pandects-test-zitadel.example.com/v2/sessions":
                self.assertEqual(method, "POST")
                return {"sessionId": "zitadel-session"}
            if url == "https://pandects-test-zitadel.example.com/v2/sessions/zitadel-session":
                if method == "PATCH":
                    return {"details": {}}
                self.assertEqual(method, "GET")
                return {
                    "session": {
                        "factors": {
                            "user": {
                                "id": "zitadel-unverified-user",
                                "displayName": "Unverified User",
                            },
                            "password": {"verifiedAt": "2026-01-01T00:00:00Z"},
                        }
                    }
                }
            if url == "https://pandects-test-zitadel.example.com/v2/users/zitadel-unverified-user":
                self.assertEqual(method, "GET")
                return {
                    "user": {
                        "human": {
                            "email": {
                                "email": "unverified@example.com",
                                "isVerified": False,
                            },
                            "profile": {
                                "displayName": "Unverified User",
                            },
                        }
                    }
                }
            self.fail(f"Unexpected URL: {url}")

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        try:
            res = client.post(
                "/v1/auth/login/password",
                json={
                    "email": "unverified@example.com",
                    "password": "Secr3tP4ss!",
                    "next": "/account",
                },
            )
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json

        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.get_json()["message"], "Verify your email before signing in.")

    def test_email_verify_resend_resends_for_unverified_user(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["AUTH_ZITADEL_API_TOKEN"] = "test-zitadel-api-token"
        client = self.app.test_client()
        with self.app.app_context():
            user_id = self._create_local_user(
                email="resend-verify@example.com",
                password="Secr3tP4ss!",
                verified=False,
                legal=True,
            )
            db.session.add(
                AuthExternalSubject(
                    user_id=user_id,
                    issuer="https://pandects-test-zitadel.example.com",
                    subject="zitadel-resend-verify-user",
                )
            )
            db.session.commit()

        original_oauth_fetch_json = backend_app._oauth_fetch_json

        def _fake_oauth_fetch_json(
            url: str,
            *,
            data: dict[str, str] | None = None,
            json_body: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            method: str | None = None,
        ):
            self.assertEqual(headers, {"Authorization": "Bearer test-zitadel-api-token"})
            self.assertIsNone(data)
            if url == "https://pandects-test-zitadel.example.com/v2/users/zitadel-resend-verify-user":
                self.assertEqual(method, "GET")
                return {
                    "user": {
                        "human": {
                            "email": {
                                "email": "resend-verify@example.com",
                                "isVerified": False,
                            },
                            "profile": {"displayName": "Resend Verify"},
                        }
                    }
                }
            if (
                url
                == "https://pandects-test-zitadel.example.com/v2/users/zitadel-resend-verify-user/email/send"
            ):
                self.assertEqual(method, "POST")
                self.assertEqual(
                    json_body,
                    {
                        "sendCode": {
                            "urlTemplate": "http://localhost:8080/verify-email?user_id={{.UserID}}&code={{.Code}}&org_id={{.OrgID}}"
                        }
                    },
                )
                return {"details": {}}
            self.fail(f"Unexpected URL: {url}")

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        try:
            res = client.post(
                "/v1/auth/email/verify/resend",
                json={"email": "resend-verify@example.com"},
            )
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json

        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload["status"], "verification_required")
        self.assertEqual(payload["user"]["email"], "resend-verify@example.com")

    def test_zitadel_email_notification_verification_uses_pandects_email_sender(self):
        client = self.app.test_client()
        payload = {
            "contextInfo": {"recipientEmailAddress": "verify-target@example.com"},
            "templateData": {"url": "http://localhost:8080/verify-email?user_id=u&code=c"},
            "args": {"code": "17UA42"},
        }

        with patch.object(auth_routes, "send_pandects_auth_email") as send_email:
            res = client.post(
                "/v1/auth/zitadel/notifications/email",
                data=json.dumps(payload),
                headers=self._zitadel_signature_headers(payload),
            )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), {"status": "sent"})
        send_email.assert_called_once_with(
            notification_type="verify-email",
            to_email="verify-target@example.com",
            action_url="http://localhost:8080/verify-email?user_id=u&code=c",
            code="17UA42",
        )

    def test_zitadel_email_notification_reset_uses_pandects_email_sender(self):
        client = self.app.test_client()
        payload = {
            "contextInfo": {"recipientEmailAddress": "reset-target@example.com"},
            "templateData": {"url": "http://localhost:8080/reset-password/confirm?user_id=u&code=c"},
        }

        with patch.object(auth_routes, "send_pandects_auth_email") as send_email:
            res = client.post(
                "/v1/auth/zitadel/notifications/email",
                data=json.dumps(payload),
                headers=self._zitadel_signature_headers(payload),
            )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), {"status": "sent"})
        send_email.assert_called_once_with(
            notification_type="reset-password",
            to_email="reset-target@example.com",
            action_url="http://localhost:8080/reset-password/confirm?user_id=u&code=c",
            code=None,
        )

    def test_zitadel_email_notification_rejects_invalid_signature(self):
        client = self.app.test_client()
        payload = {
            "contextInfo": {"recipientEmailAddress": "verify-target@example.com"},
            "templateData": {"url": "http://localhost:8080/verify-email?user_id=u&code=c"},
        }

        res = client.post(
            "/v1/auth/zitadel/notifications/email",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json", "Zitadel-Signature": "t=1,v1=bad"},
        )

        self.assertEqual(res.status_code, 401)

    def test_password_login_requires_terms_and_email_verification_for_pending_signup(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        with self.app.app_context():
            user_id = self._create_local_user(
                email="pending-login@example.com",
                password="Secr3tP4ss!",
                verified=False,
                legal=False,
            )
            db.session.add(
                AuthExternalSubject(
                    user_id=user_id,
                    issuer="https://pandects-test-zitadel.example.com",
                    subject="zitadel-pending-login-user",
                )
            )
            db.session.commit()

        original_oauth_fetch_json = backend_app._oauth_fetch_json

        def _fail_oauth_fetch_json(*args, **kwargs):
            self.fail("Pending signup login block should not call remote auth.")

        backend_app._oauth_fetch_json = _fail_oauth_fetch_json
        try:
            res = client.post(
                "/v1/auth/login/password",
                json={
                    "email": "pending-login@example.com",
                    "password": "Secr3tP4ss!",
                    "next": "/account",
                },
            )
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json

        self.assertEqual(res.status_code, 400)
        self.assertEqual(
            res.get_json()["message"],
            "You need to accept the terms and verify your email before signing in.",
        )

    def test_password_reset_confirm_updates_password_via_zitadel(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["AUTH_ZITADEL_API_TOKEN"] = "test-zitadel-api-token"
        client = self.app.test_client()

        original_oauth_fetch_json = backend_app._oauth_fetch_json

        def _fake_oauth_fetch_json(
            url: str,
            *,
            data: dict[str, str] | None = None,
            json_body: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            method: str | None = None,
        ):
            self.assertEqual(headers, {"Authorization": "Bearer test-zitadel-api-token"})
            self.assertIsNone(data)
            self.assertEqual(url, "https://pandects-test-zitadel.example.com/v2/users/zitadel-reset-user/password")
            self.assertEqual(method, "POST")
            self.assertEqual(
                json_body,
                {
                    "verificationCode": {"code": "reset-code-123"},
                    "newPassword": {
                        "password": "N3wPassw0rd!",
                        "changeRequired": False,
                    },
                },
            )
            return {"details": {}}

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        try:
            res = client.post(
                "/v1/auth/password-reset/confirm",
                json={
                    "user_id": "zitadel-reset-user",
                    "code": "reset-code-123",
                    "password": "N3wPassw0rd!",
                },
            )
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), {"status": "updated"})

    def test_zitadel_website_auth_repeat_login_reuses_linked_user(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["MCP_IDENTITY_PROVIDER"] = "zitadel"
        os.environ["MCP_ZITADEL_CLIENT_ID"] = "test-zitadel-client-id"
        os.environ["MCP_OIDC_ISSUER"] = "https://pandects-test-zitadel.example.com"
        os.environ["MCP_OIDC_AUTHORIZATION_ENDPOINT"] = (
            "https://pandects-test-zitadel.example.com/oauth/v2/authorize"
        )
        os.environ["MCP_OIDC_TOKEN_ENDPOINT"] = (
            "https://pandects-test-zitadel.example.com/oauth/v2/token"
        )
        client = self.app.test_client()

        original_oauth_fetch_json = backend_app._oauth_fetch_json
        original_authenticate = backend_app._authenticate_external_identity
        token_counter = {"value": 0}

        def _fake_oauth_fetch_json(url: str, *, data: dict[str, str] | None = None):
            token_counter["value"] += 1
            return {"access_token": f"zitadel-repeat-token-{token_counter['value']}"}

        def _fake_authenticate_external_identity(
            *, access_token: str, provider_name: str | None = None
        ):
            self.assertEqual(provider_name, "zitadel")
            return type(
                "ExternalIdentityStub",
                (),
                {
                    "issuer": "https://pandects-test-zitadel.example.com",
                    "subject": "zitadel-repeat-user",
                    "claims": {
                        "email": "repeat-zitadel@example.com",
                        "email_verified": True,
                    },
                },
            )()

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        backend_app._authenticate_external_identity = _fake_authenticate_external_identity
        try:
            for _ in range(2):
                res = client.get("/v1/auth/zitadel/start?next=/account&provider=email")
                self.assertEqual(res.status_code, 200)
                query = parse_qs(urlparse(res.get_json()["authorize_url"]).query)
                state = query.get("state", [None])[0]
                self.assertIsInstance(state, str)
                res = client.post(
                    "/v1/auth/zitadel/complete",
                    json={"code": "repeat-code", "state": state},
                )
                if res.get_json()["status"] == "legal_required":
                    res = client.post(
                        "/v1/auth/zitadel/finalize",
                        json={
                            "legal": {
                                "checked_at_ms": 1700000000000,
                                "docs": ["tos", "privacy", "license"],
                            }
                        },
                    )
                self.assertEqual(res.status_code, 200)
                self.assertEqual(res.get_json()["status"], "authenticated")
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json
            backend_app._authenticate_external_identity = original_authenticate

        with self.app.app_context():
            users = AuthUser.query.filter_by(email="repeat-zitadel@example.com").all()
            rows = AuthExternalSubject.query.all()
            self.assertEqual(len(users), 1)
            self.assertEqual(len(rows), 1)

    def test_legacy_password_reset_routes_are_disabled(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        forgot = client.post("/v1/auth/password/forgot", json={"email": "reset@example.com"})
        self.assertEqual(forgot.status_code, 404)

        reset = client.post(
            "/v1/auth/password/reset",
            json={"token": "unused", "password": "newpassword123"},
        )
        self.assertEqual(reset.status_code, 404)

    def test_legacy_email_verification_routes_are_disabled(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()

        resend = client.post("/v1/auth/email/resend", json={"email": "resend-fail@example.com"})
        self.assertEqual(resend.status_code, 404)

        verify = client.post("/v1/auth/email/verify", json={"token": "unused"})
        self.assertEqual(verify.status_code, 404)

    def test_legacy_google_credential_route_is_not_registered(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        res = client.post("/v1/auth/google/credential", json={"credential": "fake"})
        self.assertEqual(res.status_code, 404)

    def test_register_requires_captcha_when_enabled(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["TURNSTILE_ENABLED"] = "1"
        os.environ["TURNSTILE_SITE_KEY"] = "test-site-key"
        os.environ["TURNSTILE_SECRET_KEY"] = "test-secret-key"
        client = self.app.test_client()
        try:
            res = client.post("/v1/auth/register", json={})
            self.assertEqual(res.status_code, 404)
        finally:
            os.environ.pop("TURNSTILE_ENABLED", None)
            os.environ.pop("TURNSTILE_SITE_KEY", None)
            os.environ.pop("TURNSTILE_SECRET_KEY", None)

    def test_bearer_transport_issues_session_tokens(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        self._create_local_user(email="b@example.com")
        token = self._issue_bearer_session(email="b@example.com")
        self.assertIsInstance(token, str)
        self.assertGreater(len(token), 10)

        res = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(res.status_code, 200)

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
        self._create_local_user(email="delete-me@example.com")
        self._set_cookie_session(client, email="delete-me@example.com")

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

    def test_delete_account_removes_linked_zitadel_user(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        os.environ["AUTH_ZITADEL_API_TOKEN"] = "test-zitadel-api-token"
        client = self.app.test_client()

        user_id = self._create_local_user(email="delete-zitadel@example.com")
        with self.app.app_context():
            db.session.add(
                AuthExternalSubject(
                    user_id=user_id,
                    issuer="https://pandects-test-zitadel.example.com",
                    subject="zitadel-delete-user",
                )
            )
            db.session.commit()

        client.get("/v1/auth/csrf")
        self._set_cookie_session(client, email="delete-zitadel@example.com")
        csrf = self._csrf_cookie_value(client)

        original_oauth_fetch_json = backend_app._oauth_fetch_json

        def _fake_oauth_fetch_json(
            url: str,
            *,
            data: dict[str, str] | None = None,
            json_body: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            method: str | None = None,
        ):
            self.assertEqual(headers, {"Authorization": "Bearer test-zitadel-api-token"})
            self.assertIsNone(data)
            self.assertIsNone(json_body)
            self.assertEqual(method, "DELETE")
            self.assertEqual(
                url,
                "https://pandects-test-zitadel.example.com/v2/users/zitadel-delete-user",
            )
            return {"details": {}}

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        try:
            res = client.post(
                "/v1/auth/account/delete",
                json={"confirm": "Delete"},
                headers={"X-CSRF-Token": csrf},
            )
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json

        self.assertEqual(res.status_code, 200)
        with self.app.app_context():
            deleted_user = db.session.get(AuthUser, user_id)
            self.assertIsNotNone(deleted_user)
            assert deleted_user is not None
            self.assertNotEqual(deleted_user.email, "delete-zitadel@example.com")
            self.assertTrue(deleted_user.email.endswith("@deleted.invalid"))
            links = AuthExternalSubject.query.filter_by(user_id=user_id).all()
            self.assertEqual(links, [])

    def test_api_key_whitespace_is_ignored_for_last_used(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        self._create_local_user(email="keyuser@example.com")
        token = self._issue_bearer_session(email="keyuser@example.com")

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

    def test_permanent_api_key_delete_tombstones_key_and_retains_usage_rows(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        self._create_local_user(email="perm-del-key@example.com")
        token = self._issue_bearer_session(email="perm-del-key@example.com")

        res = client.post(
            "/v1/auth/api-keys",
            json={"name": "purge-me"},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertIsInstance(body, dict)
        key_id = body.get("api_key", {}).get("id")
        self.assertIsInstance(key_id, str)
        plaintext = body.get("api_key_plaintext")
        self.assertIsInstance(plaintext, str)

        with self.app.app_context():
            db.session.add(_make_api_usage_daily(api_key_id=key_id, day=date.today(), count=3))
            db.session.commit()

        res = client.delete(
            f"/v1/auth/api-keys/{key_id}/permanent",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), {"status": "deleted"})

        with self.app.app_context():
            key_row = db.session.get(ApiKey, key_id)
            self.assertIsNotNone(key_row)
            if key_row is None:
                self.fail("Expected tombstoned api key row to remain for retained usage.")
            self.assertIsNotNone(key_row.deleted_at)
            self.assertIsNotNone(key_row.revoked_at)
            self.assertIsNone(key_row.name)
            self.assertEqual(ApiUsageDaily.query.filter_by(api_key_id=key_id).count(), 1)
            self.assertIsNone(backend_app._lookup_api_key(plaintext))

        res = client.get("/v1/auth/api-keys", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(res.status_code, 200)
        listed_ids = {item["id"] for item in res.get_json()["keys"]}
        self.assertNotIn(key_id, listed_ids)

        res = client.get("/v1/auth/usage?period=1w", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(res.status_code, 200)
        usage_body = res.get_json()
        self.assertEqual(usage_body["total"], 3)
        self.assertEqual(usage_body["by_day"], [{"day": date.today().isoformat(), "count": 3}])

        res = client.get(
            f"/v1/auth/usage?period=1w&api_key_id={key_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(res.status_code, 404)

    def test_api_key_last_used_updates_are_throttled(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        self._create_local_user(email="keyuser-throttle@example.com")
        token = self._issue_bearer_session(email="keyuser-throttle@example.com")

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

    def test_api_key_touch_state_is_bounded_under_high_cardinality_keys(self):
        old_touch_seconds = backend_app._API_KEY_LAST_USED_TOUCH_SECONDS
        old_max_keys = backend_app._API_KEY_LAST_USED_MAX_KEYS
        try:
            backend_app._API_KEY_LAST_USED_TOUCH_SECONDS = 300
            backend_app._API_KEY_LAST_USED_MAX_KEYS = 5
            backend_app._api_key_last_used_touch_state.clear()

            for idx in range(40):
                with patch("backend.app.time.time", return_value=2000.0 + idx):
                    should_touch = backend_app._should_touch_api_key_last_used(f"key-{idx}")
                self.assertTrue(should_touch)

            # Pruning runs before insertion, so transient max+1 is expected.
            self.assertLessEqual(len(backend_app._api_key_last_used_touch_state), 6)
            retained_keys = set(backend_app._api_key_last_used_touch_state.keys())
            self.assertIn("key-39", retained_keys)
            self.assertNotIn("key-0", retained_keys)
        finally:
            backend_app._API_KEY_LAST_USED_TOUCH_SECONDS = old_touch_seconds
            backend_app._API_KEY_LAST_USED_MAX_KEYS = old_max_keys
            backend_app._api_key_last_used_touch_state.clear()

    def test_usage_endpoint_supports_period_and_api_key_filters(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        self._create_local_user(email="usage-filters@example.com")
        token = self._issue_bearer_session(email="usage-filters@example.com")

        res = client.post(
            "/v1/auth/api-keys",
            json={"name": "first key"},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(res.status_code, 200)
        first_payload = res.get_json()
        self.assertIsInstance(first_payload, dict)
        first_api_key = first_payload.get("api_key")
        self.assertIsInstance(first_api_key, dict)
        first_key_id = first_api_key.get("id")
        self.assertIsInstance(first_key_id, str)

        res = client.post(
            "/v1/auth/api-keys",
            json={"name": "second key"},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(res.status_code, 200)
        second_payload = res.get_json()
        self.assertIsInstance(second_payload, dict)
        second_api_key = second_payload.get("api_key")
        self.assertIsInstance(second_api_key, dict)
        second_key_id = second_api_key.get("id")
        self.assertIsInstance(second_key_id, str)

        today = backend_app._utc_today()
        with self.app.app_context():
            db.session.add_all(
                [
                    _make_api_usage_daily(api_key_id=first_key_id, day=today, count=4),
                    _make_api_usage_daily(api_key_id=second_key_id, day=today, count=3),
                    _make_api_usage_daily(api_key_id=first_key_id, day=today - timedelta(days=8), count=10),
                ]
            )
            db.session.commit()

        res = client.get("/v1/auth/usage?period=1w", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(res.status_code, 200)
        weekly_payload = res.get_json()
        self.assertIsInstance(weekly_payload, dict)
        self.assertEqual(weekly_payload.get("total"), 7)
        weekly_by_day = weekly_payload.get("by_day")
        self.assertIsInstance(weekly_by_day, list)
        self.assertEqual(len(weekly_by_day), 1)
        self.assertEqual(weekly_by_day[0].get("day"), today.isoformat())
        self.assertEqual(weekly_by_day[0].get("count"), 7)

        res = client.get(
            f"/v1/auth/usage?period=1w&api_key_id={first_key_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(res.status_code, 200)
        weekly_first_key = res.get_json()
        self.assertIsInstance(weekly_first_key, dict)
        self.assertEqual(weekly_first_key.get("total"), 4)
        weekly_first_by_day = weekly_first_key.get("by_day")
        self.assertIsInstance(weekly_first_by_day, list)
        self.assertEqual(len(weekly_first_by_day), 1)
        self.assertEqual(weekly_first_by_day[0].get("day"), today.isoformat())
        self.assertEqual(weekly_first_by_day[0].get("count"), 4)

        res = client.get(
            f"/v1/auth/usage?period=all&api_key_id={first_key_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(res.status_code, 200)
        all_first_key = res.get_json()
        self.assertIsInstance(all_first_key, dict)
        self.assertEqual(all_first_key.get("total"), 14)
        all_by_day = all_first_key.get("by_day")
        self.assertIsInstance(all_by_day, list)
        self.assertEqual(len(all_by_day), 2)
        self.assertEqual(all_by_day[0].get("day"), (today - timedelta(days=8)).isoformat())
        self.assertEqual(all_by_day[0].get("count"), 10)
        self.assertEqual(all_by_day[1].get("day"), today.isoformat())
        self.assertEqual(all_by_day[1].get("count"), 4)

        res = client.get("/v1/auth/usage?period=90d", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(res.status_code, 400)

        res = client.get(
            "/v1/auth/usage?api_key_id=00000000-0000-0000-0000-000000000000",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
