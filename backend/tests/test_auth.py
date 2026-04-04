import os
import tempfile
import unittest
from datetime import date, timedelta
from urllib.parse import parse_qs, urlparse
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
    os.environ.setdefault("MCP_ZITADEL_CLIENT_ID", "test-zitadel-client-id")
    os.environ.setdefault("MCP_OIDC_ISSUER", "https://pandects-test-zitadel.example.com")
    os.environ.setdefault("MCP_OIDC_AUDIENCE", "https://api.pandects.org/mcp")
    os.environ["TURNSTILE_ENABLED"] = "0"
    os.environ.pop("TURNSTILE_SITE_KEY", None)
    os.environ.pop("TURNSTILE_SECRET_KEY", None)


_set_default_env()

_AUTH_DB_TEMP = tempfile.NamedTemporaryFile(prefix="pandects_auth_", suffix=".sqlite", delete=False)
_AUTH_DB_TEMP.close()
os.environ.setdefault("AUTH_DATABASE_URI", f"sqlite:///{_AUTH_DB_TEMP.name}")


from backend.app import create_test_app, db, ApiKey, ApiUsageDaily, AuthExternalSubject, AuthUser  # noqa: E402
import backend.app as backend_app  # noqa: E402


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

    def test_cookie_transport_register_login_csrf_and_logout(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        client = self.app.test_client()

        res = client.get("/v1/auth/csrf")
        self.assertEqual(res.status_code, 200)
        csrf = self._csrf_cookie_value(client)
        res = client.post("/v1/auth/register", json={})
        self.assertEqual(res.status_code, 403)
        res = client.post("/v1/auth/register", json={}, headers={"X-CSRF-Token": csrf})
        self.assertEqual(res.status_code, 404)
        res = client.post("/v1/auth/login", json={})
        self.assertEqual(res.status_code, 403)
        res = client.post("/v1/auth/login", json={}, headers={"X-CSRF-Token": csrf})
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

        original_google_fetch_json = backend_app._google_fetch_json
        original_authenticate = backend_app._authenticate_external_identity

        def _fake_google_fetch_json(url: str, *, data: dict[str, str] | None = None):
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

        backend_app._google_fetch_json = _fake_google_fetch_json
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
            backend_app._google_fetch_json = original_google_fetch_json
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

        original_google_fetch_json = backend_app._google_fetch_json
        original_authenticate = backend_app._authenticate_external_identity

        def _fake_google_fetch_json(url: str, *, data: dict[str, str] | None = None):
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

        backend_app._google_fetch_json = _fake_google_fetch_json
        backend_app._authenticate_external_identity = _fake_authenticate_external_identity
        try:
            res = client.get("/v1/auth/zitadel/start?next=/search&provider=google")
            self.assertEqual(res.status_code, 200)
            start_payload = res.get_json()
            self.assertIsInstance(start_payload, dict)
            authorize_url = start_payload.get("authorize_url")
            self.assertIsInstance(authorize_url, str)

            parsed = urlparse(authorize_url)
            query = parse_qs(parsed.query)
            state = query.get("state", [None])[0]
            self.assertIsInstance(state, str)

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
            backend_app._google_fetch_json = original_google_fetch_json
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

        original_google_fetch_json = backend_app._google_fetch_json
        original_authenticate = backend_app._authenticate_external_identity

        def _fake_google_fetch_json(url: str, *, data: dict[str, str] | None = None):
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

        backend_app._google_fetch_json = _fake_google_fetch_json
        backend_app._authenticate_external_identity = _fake_authenticate_external_identity
        try:
            res = client.get("/v1/auth/zitadel/start?next=/account&provider=email")
            self.assertEqual(res.status_code, 200)
            query = parse_qs(urlparse(res.get_json()["authorize_url"]).query)
            state = query.get("state", [None])[0]
            self.assertIsInstance(state, str)

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
            backend_app._google_fetch_json = original_google_fetch_json
            backend_app._authenticate_external_identity = original_authenticate

        with self.app.app_context():
            user = self._require_user("new-zitadel-user@example.com")
            self.assertIsNotNone(user.email_verified_at)
            rows = AuthExternalSubject.query.all()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].user_id, user.id)
            engine = db.engines["auth"]
            with engine.begin() as conn:
                legal_count = conn.execute(
                    text("SELECT COUNT(*) FROM legal_acceptances WHERE user_id = :user_id"),
                    {"user_id": user.id},
                ).scalar_one()
                signons = conn.execute(
                    text("SELECT provider, action FROM auth_signon_events")
                ).fetchall()
            self.assertEqual(legal_count, len(backend_app._LEGAL_DOCS))
            self.assertEqual(signons, [("zitadel", "register")])

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

        original_google_fetch_json = backend_app._google_fetch_json
        original_authenticate = backend_app._authenticate_external_identity
        token_counter = {"value": 0}

        def _fake_google_fetch_json(url: str, *, data: dict[str, str] | None = None):
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

        backend_app._google_fetch_json = _fake_google_fetch_json
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
            backend_app._google_fetch_json = original_google_fetch_json
            backend_app._authenticate_external_identity = original_authenticate

        with self.app.app_context():
            users = AuthUser.query.filter_by(email="repeat-zitadel@example.com").all()
            rows = AuthExternalSubject.query.all()
            self.assertEqual(len(users), 1)
            self.assertEqual(len(rows), 1)

    def test_password_reset_flow(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        self._create_local_user(email="reset@example.com")

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

        with self.app.app_context():
            user = self._require_user("reset@example.com")
            self.assertTrue(
                backend_app.check_password_hash(user.password_hash, "newpassword123")
            )

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
            _ = (to_email, token)
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
            _ = (to_email, token)
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
        res = client.post("/v1/auth/google/credential", json={"credential": "fake"})
        self.assertEqual(res.status_code, 404)

    def test_google_credential_requires_nonce_cookie(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()

        res = client.post("/v1/auth/google/credential", json={"credential": "fake"})
        self.assertEqual(res.status_code, 404)

    def test_google_credential_passes_nonce_from_cookie(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        self._set_google_nonce_cookie(client)
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

    def test_google_credential_invalid_token_is_401_without_network(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        self._set_google_nonce_cookie(client)
        res = client.post("/v1/auth/google/credential", json={"credential": "nope"})
        self.assertEqual(res.status_code, 404)

    def test_google_credential_jwks_outage_returns_503(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        self._set_google_nonce_cookie(client)
        res = client.post("/v1/auth/google/credential", json={"credential": "nope"})
        self.assertEqual(res.status_code, 404)

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
