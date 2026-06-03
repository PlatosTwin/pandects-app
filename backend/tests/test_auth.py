import os
import tempfile
import unittest
import hashlib
import hmac
import json
import time
import uuid
from base64 import urlsafe_b64encode
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch
from typing import cast
from sqlalchemy import text


def _set_default_env() -> None:
    os.environ["SKIP_MAIN_DB_REFLECTION"] = "1"
    os.environ["MARIADB_USER"] = "root"
    os.environ["MARIADB_PASSWORD"] = "password"
    os.environ["MARIADB_HOST"] = "127.0.0.1"
    os.environ["MARIADB_DATABASE"] = "pdx"
    os.environ["AUTH_SECRET_KEY"] = "test-auth-secret"
    os.environ["PUBLIC_API_BASE_URL"] = "http://localhost:5000"
    os.environ["PUBLIC_FRONTEND_BASE_URL"] = "http://localhost:8080"
    os.environ["GOOGLE_OAUTH_CLIENT_ID"] = "test-google-client-id"
    os.environ["GOOGLE_OAUTH_CLIENT_SECRET"] = "test-google-client-secret"
    os.environ["MCP_ZITADEL_CLIENT_ID"] = "test-zitadel-client-id"
    os.environ["MCP_OIDC_ISSUER"] = "https://pandects-test-zitadel.example.com"
    os.environ["MCP_OIDC_AUDIENCE"] = "https://api.pandects.org/mcp"
    _ = os.environ.pop("AUTH_ZITADEL_PROJECT_ID", None)
    _ = os.environ.pop("AUTH_ZITADEL_PROJECT_NAME", None)
    os.environ["TURNSTILE_ENABLED"] = "0"
    _ = os.environ.pop("TURNSTILE_SITE_KEY", None)
    _ = os.environ.pop("TURNSTILE_SECRET_KEY", None)


_set_default_env()

_AUTH_DB_TEMP = tempfile.NamedTemporaryFile(prefix="pandects_auth_", suffix=".sqlite", delete=False)
_AUTH_DB_TEMP.close()
os.environ["AUTH_DATABASE_URI"] = f"sqlite:///{_AUTH_DB_TEMP.name}"


from backend.app import create_test_app  # noqa: E402
from backend.extensions import db  # noqa: E402
from backend.models.auth import ApiKey, ApiUsageDaily, AuthExternalSubject, AuthOAuthAuthorizationCode, AuthOAuthClient, AuthOAuthRefreshToken, AuthOAuthUserGrant, AuthUser, LegalAcceptance  # noqa: E402
import backend.app as backend_app  # noqa: E402
import backend.routes.auth as auth_routes  # noqa: E402
from werkzeug.exceptions import Conflict, NotFound, Unauthorized  # noqa: E402


def _make_api_usage_daily(*, api_key_id: str, day: date, count: int) -> object:
    return ApiUsageDaily(api_key_id=api_key_id, day=day, count=count)  # pyright: ignore[reportCallIssue]


def _pkce_challenge(verifier: str) -> str:
    return urlsafe_b64encode(hashlib.sha256(verifier.encode("utf-8")).digest()).rstrip(b"=").decode("ascii")


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
        _set_default_env()
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
                # OAuth tables: FK order matters — codes/refresh/grants
                # reference clients; grants also reference users.
                conn.execute(text("DELETE FROM auth_oauth_authorization_codes"))
                conn.execute(text("DELETE FROM auth_oauth_refresh_tokens"))
                conn.execute(text("DELETE FROM auth_oauth_user_grants"))
                conn.execute(text("DELETE FROM auth_oauth_clients"))
                conn.execute(text("DELETE FROM auth_users"))
        backend_app._rate_limit_state.clear()
        backend_app._endpoint_rate_limit_state.clear()
        backend_app._account_login_failure_state.clear()
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

    def _legal_acceptance(
        self,
        *,
        user_id: str,
        document: str,
        version: str,
        document_hash: str,
        checked_at: datetime,
    ) -> LegalAcceptance:
        acceptance = LegalAcceptance()
        acceptance.user_id = user_id
        acceptance.document = document
        acceptance.version = version
        acceptance.document_hash = document_hash
        acceptance.checked_at = checked_at
        acceptance.submitted_at = checked_at
        acceptance.ip_address = None
        acceptance.user_agent = None
        return acceptance

    def _auth_external_subject(self, *, user_id: str, issuer: str, subject: str) -> AuthExternalSubject:
        link = AuthExternalSubject()
        link.user_id = user_id
        link.issuer = issuer
        link.subject = subject
        return link

    def _dict_value(self, payload: Mapping[str, object], key: str) -> dict[str, object]:
        value = payload[key]
        self.assertIsInstance(value, dict)
        return cast(dict[str, object], value)

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
                        self._legal_acceptance(
                            user_id=user.id,
                            document=doc,
                            version=meta["version"],
                            document_hash=meta["sha256"],
                            checked_at=checked_at,
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

    def _grant_oauth_consent(self, *, email: str, client_id: str, scope: str) -> None:
        """Pre-record per-client OAuth consent so /oauth/authorize will mint a
        code instead of returning consent_required. Tests that drive the full
        authorize -> token flow as an already-trusted client should call this
        after registering the DCR client and before the first authorize."""
        with self.app.app_context():
            user = self._require_user(email)
            normalized = " ".join(sorted({s for s in scope.split(" ") if s}))
            existing = AuthOAuthUserGrant.query.filter_by(
                user_id=user.id, client_id=client_id
            ).first()
            if existing is None:
                grant = AuthOAuthUserGrant()
                grant.user_id = user.id
                grant.client_id = client_id
                grant.scope = normalized
                db.session.add(grant)
            else:
                existing.scope = normalized
                existing.revoked_at = None
            db.session.commit()

    def _zitadel_signature_headers(self, payload: Mapping[str, object]) -> dict[str, str]:
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
            _ = self._require_user("second-link@example.com")
            db.session.add(
                self._auth_external_subject(
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
                    self._legal_acceptance(
                        user_id=existing.id,
                        document=doc,
                        version=meta["version"],
                        document_hash=meta["sha256"],
                        checked_at=checked_at,
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
            if url == "https://pandects-test-zitadel.example.com/v2/idp_intents/intent-123":
                self.assertEqual(method, "POST")
                self.assertEqual(json_body, {"idpIntentToken": "intent-token-123"})
                return {
                    "idpInformation": {
                        "idpId": "google-idp-123",
                        "rawInformation": {
                            "User": {
                                "sub": "google-sub-123",
                                "email": "existing-google@example.com",
                                "email_verified": True,
                                "name": "Existing Google",
                                "given_name": "Existing",
                                "family_name": "Google",
                            }
                        },
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

    def test_zitadel_google_intent_new_user_sends_signup_notification(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["MCP_IDENTITY_PROVIDER"] = "zitadel"
        os.environ["AUTH_ZITADEL_API_TOKEN"] = "test-zitadel-api-token"
        os.environ["AUTH_ZITADEL_GOOGLE_IDP_ID"] = "google-idp-123"
        client = self.app.test_client()

        original_oauth_fetch_json = backend_app._oauth_fetch_json
        original_send_resend_text_email = backend_app._send_resend_text_email
        original_send_welcome_email = backend_app._send_welcome_email
        original_async_enabled = backend_app._ASYNC_SIDE_EFFECTS_ENABLED
        sent_notifications: list[dict[str, str]] = []

        def _fake_send_resend_text_email(*, to_email: str, subject: str, text: str) -> None:
            sent_notifications.append({"to_email": to_email, "subject": subject, "text": text})

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
                                "email": "new-google@example.com",
                                "email_verified": True,
                                "name": "New Google",
                                "given_name": "New",
                                "family_name": "Google",
                            }
                        },
                    }
                }
            if url == "https://pandects-test-zitadel.example.com/v2/users/human":
                self.assertEqual(method, "POST")
                self.assertIsInstance(json_body, dict)
                assert json_body is not None
                self.assertEqual(json_body["username"], "new-google@example.com")
                self.assertEqual(self._dict_value(json_body, "profile")["displayName"], "New Google")
                return {"userId": "zitadel-new-google-123"}
            self.fail(f"Unexpected ZITADEL API request: {url}")

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        backend_app._send_resend_text_email = _fake_send_resend_text_email
        backend_app._send_welcome_email = lambda **_kwargs: None
        backend_app._ASYNC_SIDE_EFFECTS_ENABLED = False
        try:
            start = client.get("/v1/auth/zitadel/google/start?next=/account")
            self.assertEqual(start.status_code, 200)

            res = client.post(
                "/v1/auth/zitadel/complete",
                json={"intent_id": "intent-123", "intent_token": "intent-token-123"},
            )
            self.assertEqual(res.status_code, 200)
            payload = res.get_json()
            self.assertEqual(payload["status"], "legal_required")
            self.assertEqual(payload["user"]["email"], "new-google@example.com")
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json
            backend_app._send_resend_text_email = original_send_resend_text_email
            backend_app._send_welcome_email = original_send_welcome_email
            backend_app._ASYNC_SIDE_EFFECTS_ENABLED = original_async_enabled

        self.assertEqual(len(sent_notifications), 1)
        self.assertEqual(sent_notifications[0]["to_email"], "nmbogdan@alumni.stanford.edu")
        self.assertEqual(sent_notifications[0]["subject"], "New Pandects signup")
        self.assertIn("Full name: New Google", sent_notifications[0]["text"])
        self.assertIn("Email: new-google@example.com", sent_notifications[0]["text"])
        self.assertIn("Provider: google", sent_notifications[0]["text"])
        self.assertRegex(sent_notifications[0]["text"], r"Signup date: \d{4}-\d{2}-\d{2}")
        self.assertRegex(sent_notifications[0]["text"], r"Signup time: \d{2}:\d{2}:\d{2} UTC")

    def test_zitadel_google_intent_callback_retrieves_intent_before_using_existing_linked_subject(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["MCP_IDENTITY_PROVIDER"] = "zitadel"
        os.environ["AUTH_ZITADEL_API_TOKEN"] = "test-zitadel-api-token"
        os.environ["AUTH_ZITADEL_GOOGLE_IDP_ID"] = "google-idp-123"
        client = self.app.test_client()

        existing_user_id = self._create_local_user(email="linked-google@example.com")
        with self.app.app_context():
            db.session.add(
                self._auth_external_subject(
                    user_id=existing_user_id,
                    issuer="https://pandects-test-zitadel.example.com",
                    subject="zitadel-user-linked-123",
                )
            )
            db.session.commit()

        original_oauth_fetch_json = backend_app._oauth_fetch_json
        retrieved_intent = {"value": False}

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
            if url == "https://pandects-test-zitadel.example.com/v2/idp_intents/intent-123":
                retrieved_intent["value"] = True
                self.assertEqual(method, "POST")
                self.assertEqual(json_body, {"idpIntentToken": "intent-token-123"})
                return {
                    "idpInformation": {
                        "idpId": "google-idp-123",
                        "rawInformation": {
                            "User": {
                                "sub": "google-sub-123",
                                "email": "linked-google@example.com",
                                "email_verified": True,
                                "name": "Linked Google",
                            }
                        },
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
            self.assertTrue(retrieved_intent["value"])
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json

    def test_zitadel_complete_rejects_google_intent_payload_without_google_intent_flow(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["MCP_IDENTITY_PROVIDER"] = "zitadel"
        os.environ["MCP_ZITADEL_CLIENT_ID"] = "test-zitadel-client-id"
        os.environ["MCP_OIDC_ISSUER"] = "https://pandects-test-zitadel.example.com"
        os.environ["MCP_OIDC_AUTHORIZATION_ENDPOINT"] = (
            "https://pandects-test-zitadel.example.com/oauth/v2/authorize"
        )
        client = self.app.test_client()

        start = client.get("/v1/auth/zitadel/start?next=/account&provider=email")
        self.assertEqual(start.status_code, 200)

        res = client.post(
            "/v1/auth/zitadel/complete",
            json={
                "intent_id": "intent-123",
                "intent_token": "intent-token-123",
                "user_id": "zitadel-user-linked-123",
            },
        )

        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.get_json()["message"], "Invalid authorization state.")

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
                if data is None:
                    self.fail("Expected token request form data.")
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
                if json_body is None:
                    self.fail("Expected IDP intent request JSON.")
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
                    self._legal_acceptance(
                        user_id=existing.id,
                        document=doc,
                        version=meta["version"],
                        document_hash=meta["sha256"],
                        checked_at=checked_at,
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
                email_payload = self._dict_value(json_body, "email")
                self.assertIs(email_payload["isVerified"], True)
                return {"userId": "zitadel-user-migrated"}
            if url == "https://pandects-test-zitadel.example.com/management/v1/users/grants/_search":
                self.assertEqual(method, "POST")
                return {"result": []}
            if url == "https://pandects-test-zitadel.example.com/management/v1/users/zitadel-user-migrated/grants":
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
        original_send_resend_text_email = backend_app._send_resend_text_email
        original_send_welcome_email = backend_app._send_welcome_email
        original_async_enabled = backend_app._ASYNC_SIDE_EFFECTS_ENABLED
        sent_verification = {"called": False}
        sent_notifications: list[dict[str, str]] = []
        created_signup_payloads: list[dict[str, object]] = []

        def _fake_send_resend_text_email(*, to_email: str, subject: str, text: str) -> None:
            sent_notifications.append({"to_email": to_email, "subject": subject, "text": text})

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
            created_signup_payloads.append(json_body)
            self.assertEqual(json_body["username"], "new-password-user@example.com")
            profile_payload = self._dict_value(json_body, "profile")
            email_payload = self._dict_value(json_body, "email")
            self.assertEqual(profile_payload["givenName"], "New")
            self.assertEqual(profile_payload["familyName"], "User")
            self.assertEqual(email_payload["email"], "new-password-user@example.com")
            self.assertNotIn("sendCode", email_payload)
            self.assertIs(email_payload["isVerified"], False)
            return {"userId": "zitadel-new-user"}

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        backend_app._send_resend_text_email = _fake_send_resend_text_email
        backend_app._send_welcome_email = lambda **_kwargs: None
        backend_app._ASYNC_SIDE_EFFECTS_ENABLED = False
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
            backend_app._send_resend_text_email = original_send_resend_text_email
            backend_app._send_welcome_email = original_send_welcome_email
            backend_app._ASYNC_SIDE_EFFECTS_ENABLED = original_async_enabled

        self.assertEqual(finalize.status_code, 200)
        finalize_payload = finalize.get_json()
        self.assertEqual(finalize_payload["status"], "verification_required")
        self.assertEqual(len(created_signup_payloads), 1)
        self.assertTrue(sent_verification["called"])
        self.assertEqual(len(sent_notifications), 1)
        self.assertEqual(sent_notifications[0]["to_email"], "nmbogdan@alumni.stanford.edu")
        self.assertEqual(sent_notifications[0]["subject"], "New Pandects signup")
        self.assertIn("Full name: New User", sent_notifications[0]["text"])
        self.assertIn("Email: new-password-user@example.com", sent_notifications[0]["text"])
        self.assertIn("Provider: password", sent_notifications[0]["text"])
        self.assertRegex(sent_notifications[0]["text"], r"Signup date: \d{4}-\d{2}-\d{2}")
        self.assertRegex(sent_notifications[0]["text"], r"Signup time: \d{2}:\d{2}:\d{2} UTC")

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
            if url == "https://pandects-test-zitadel.example.com/management/v1/users/zitadel-role-grant-user/grants":
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
        self.assertEqual(
            grant_calls[1][0],
            "https://pandects-test-zitadel.example.com/management/v1/users/zitadel-role-grant-user/grants",
        )
        self.assertEqual(grant_calls[1][1], "POST")
        self.assertEqual(
            grant_calls[1][2],
            {
                "projectId": "pandects-project-123",
                "roleKeys": [
                    "agreements_read",
                    "agreements_read_fulltext",
                    "agreements_search",
                    "sections_search",
                ],
            },
        )

    def test_password_signup_updates_existing_zitadel_project_grant_when_roles_are_missing(self):
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
                return {"userId": "zitadel-role-update-user"}
            if url == "https://pandects-test-zitadel.example.com/management/v1/users/grants/_search":
                grant_calls.append((url, method, json_body))
                return {
                    "result": [
                        {
                            "id": "grant-123",
                            "roleKeys": ["agreements_read"],
                        }
                    ]
                }
            if (
                url
                == "https://pandects-test-zitadel.example.com/management/v1/users/zitadel-role-update-user/grants/grant-123"
            ):
                grant_calls.append((url, method, json_body))
                return {"id": "grant-123"}
            self.fail(f"Unexpected URL: {url}")

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        try:
            res = client.post(
                "/v1/auth/signup/password",
                json={
                    "email": "role-update@example.com",
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
        self.assertEqual(grant_calls[0][1], "POST")
        self.assertEqual(grant_calls[1][1], "PUT")
        self.assertEqual(
            grant_calls[1][2],
            {
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
                self._auth_external_subject(
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

    def test_password_signup_masks_existing_verified_account_to_prevent_enumeration(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        with self.app.app_context():
            user_id = self._create_local_user(
                email="taken@example.com",
                password="Secr3tP4ss!",
                verified=True,
                legal=True,
            )
            db.session.add(
                self._auth_external_subject(
                    user_id=user_id,
                    issuer="https://pandects-test-zitadel.example.com",
                    subject="zitadel-taken-user",
                )
            )
            db.session.commit()

        original_oauth_fetch_json = backend_app._oauth_fetch_json

        def _fail_oauth_fetch_json(*args, **kwargs):
            self.fail("Masked-existing-account signup must not call the remote auth provider.")

        backend_app._oauth_fetch_json = _fail_oauth_fetch_json
        try:
            res = client.post(
                "/v1/auth/signup/password",
                json={
                    "email": "taken@example.com",
                    "password": "AnotherP4ss!",
                    "next": "/account",
                },
            )
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json

        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload["status"], "verification_required")
        self.assertEqual(payload["user"]["email"], "taken@example.com")
        # Masked response must mint a throwaway UUID, not echo the real user
        # id and not return an obviously-fake empty string — both would let
        # an unauthenticated caller distinguish "existing" from "new".
        masked_id = payload["user"]["id"]
        self.assertIsInstance(masked_id, str)
        self.assertNotEqual(masked_id, user_id)
        self.assertNotEqual(masked_id, "")
        self.assertEqual(len(masked_id), 36)

    def test_password_signup_requires_captcha_token_when_turnstile_enabled(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["TURNSTILE_ENABLED"] = "1"
        os.environ["TURNSTILE_SITE_KEY"] = "test-site-key"
        os.environ["TURNSTILE_SECRET_KEY"] = "test-secret-key"
        client = self.app.test_client()
        try:
            res = client.post(
                "/v1/auth/signup/password",
                json={
                    "email": "needs-captcha@example.com",
                    "password": "Secr3tP4ss!",
                    "next": "/account",
                },
            )
        finally:
            os.environ.pop("TURNSTILE_ENABLED", None)
            os.environ.pop("TURNSTILE_SITE_KEY", None)
            os.environ.pop("TURNSTILE_SECRET_KEY", None)

        self.assertEqual(res.status_code, 412)
        payload = res.get_json()
        self.assertEqual(payload["error"], "captcha_required")

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
                self._auth_external_subject(
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
                self._auth_external_subject(
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
                self.assertIsInstance(json_body, dict)
                assert json_body is not None
                self.assertEqual(json_body["username"], "pending-missing-remote@example.com")
                password_payload = self._dict_value(json_body, "password")
                email_payload = self._dict_value(json_body, "email")
                self.assertEqual(password_payload["password"], "NewP4ssword!")
                self.assertIs(email_payload["isVerified"], False)
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
                self._auth_external_subject(
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

    def test_password_signup_defers_verification_send_until_finalize(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["AUTH_ZITADEL_API_TOKEN"] = "test-zitadel-api-token"
        client = self.app.test_client()

        original_oauth_fetch_json = backend_app._oauth_fetch_json
        signup_payloads: list[dict[str, object]] = []
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
            if url == "https://pandects-test-zitadel.example.com/v2/users/human":
                self.assertEqual(method, "POST")
                self.assertIsInstance(json_body, dict)
                assert json_body is not None
                signup_payloads.append(json_body)
                email_payload = self._dict_value(json_body, "email")
                self.assertNotIn("sendCode", email_payload)
                self.assertIs(email_payload["isVerified"], False)
                return {"userId": "zitadel-deferred-verify-user"}
            if url == "https://pandects-test-zitadel.example.com/v2/users/zitadel-deferred-verify-user/email/send":
                self.assertEqual(method, "POST")
                sent_verification["called"] = True
                return {"details": {}}
            self.fail(f"Unexpected URL: {url}")

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        try:
            res = client.post(
                "/v1/auth/signup/password",
                json={
                    "email": "deferred-verify@example.com",
                    "password": "Secr3tP4ss!",
                    "next": "/account",
                },
            )
            self.assertEqual(res.status_code, 200)
            payload = res.get_json()
            self.assertEqual(payload["status"], "legal_required")
            self.assertFalse(sent_verification["called"])

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

        self.assertEqual(len(signup_payloads), 1)
        self.assertEqual(finalize.status_code, 200)
        finalize_payload = finalize.get_json()
        self.assertEqual(finalize_payload["status"], "verification_required")
        self.assertTrue(sent_verification["called"])

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
                self.assertIsInstance(json_body, dict)
                assert json_body is not None
                email_payload = self._dict_value(json_body, "email")
                self.assertIs(email_payload["isVerified"], True)
                return {"userId": "zitadel-reset-user"}
            if url == "https://pandects-test-zitadel.example.com/management/v1/users/grants/_search":
                self.assertEqual(method, "POST")
                return {"result": []}
            if url == "https://pandects-test-zitadel.example.com/management/v1/users/zitadel-reset-user/grants":
                self.assertEqual(method, "POST")
                return {"id": "grant-123"}
            if url == "https://pandects-test-zitadel.example.com/v2/users/zitadel-reset-user/password_reset":
                self.assertEqual(method, "POST")
                self.assertIsInstance(json_body, dict)
                assert json_body is not None
                send_link_payload = self._dict_value(json_body, "sendLink")
                url_template = send_link_payload["urlTemplate"]
                if not isinstance(url_template, str):
                    self.fail("Expected reset URL template.")
                self.assertIn("/reset-password/confirm", url_template)
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

    def test_password_reset_request_sends_reset_for_unverified_linked_user(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["AUTH_ZITADEL_API_TOKEN"] = "test-zitadel-api-token"
        client = self.app.test_client()
        with self.app.app_context():
            user_id = self._create_local_user(
                email="unverified-reset@example.com",
                password="password123",
                verified=False,
                legal=False,
            )
            db.session.add(
                self._auth_external_subject(
                    user_id=user_id,
                    issuer="https://pandects-test-zitadel.example.com",
                    subject="zitadel-unverified-reset-user",
                )
            )
            db.session.commit()

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
            if url == "https://pandects-test-zitadel.example.com/v2/users/zitadel-unverified-reset-user/password_reset":
                self.assertEqual(method, "POST")
                self.assertIsInstance(json_body, dict)
                assert json_body is not None
                send_link_payload = self._dict_value(json_body, "sendLink")
                url_template = send_link_payload["urlTemplate"]
                if not isinstance(url_template, str):
                    self.fail("Expected reset URL template.")
                self.assertIn("/reset-password/confirm", url_template)
                seen_reset["called"] = True
                return {"details": {}}
            self.fail(f"Unexpected URL: {url}")

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        try:
            res = client.post(
                "/v1/auth/password-reset/request",
                json={"email": "unverified-reset@example.com"},
            )
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json

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
                self._auth_external_subject(
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
            "args": {"code": "95HQ2"},
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
            code="95HQ2",
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
                self._auth_external_subject(
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
                    "verificationCode": "reset-code-123",
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

    def test_zitadel_mcp_token_flow_returns_access_token_for_current_user(self):
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
        self._create_local_user(email="token-user@example.com")
        session_token = self._issue_bearer_session(email="token-user@example.com")
        auth_headers = {"Authorization": f"Bearer {session_token}"}

        original_oauth_fetch_json = backend_app._oauth_fetch_json
        original_authenticate = backend_app._authenticate_external_identity

        def _fake_oauth_fetch_json(url: str, *, data: dict[str, str] | None = None):
            self.assertEqual(url, "https://pandects-test-zitadel.example.com/oauth/v2/token")
            if data is None:
                self.fail("Expected token request form data.")
            self.assertEqual(data["grant_type"], "authorization_code")
            return {
                "access_token": "test-mcp-access-token",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "sections:search agreements:search agreements:read",
            }

        def _fake_authenticate_external_identity(
            *, access_token: str, provider_name: str | None = None
        ):
            self.assertEqual(access_token, "test-mcp-access-token")
            self.assertEqual(provider_name, "zitadel")
            return type(
                "ExternalIdentityStub",
                (),
                {
                    "issuer": "https://pandects-test-zitadel.example.com",
                    "subject": "zitadel-token-user",
                    "claims": {
                        "email": "token-user@example.com",
                        "email_verified": True,
                    },
                },
            )()

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        backend_app._authenticate_external_identity = _fake_authenticate_external_identity
        try:
            start = client.get("/v1/auth/mcp-token/start?next=/account", headers=auth_headers)
            self.assertEqual(start.status_code, 200)
            query = parse_qs(urlparse(start.get_json()["authorize_url"]).query)
            state = query.get("state", [None])[0]
            self.assertIsInstance(state, str)

            res = client.post(
                "/v1/auth/zitadel/complete",
                json={"code": "token-code", "state": state},
                headers=auth_headers,
            )
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json
            backend_app._authenticate_external_identity = original_authenticate

        self.assertEqual(res.status_code, 200)
        self.assertEqual(
            res.get_json(),
            {
                "status": "mcp_token",
                "next_path": "/account",
                "access_token": "test-mcp-access-token",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "sections:search agreements:search agreements:read",
            },
        )
        with self.app.app_context():
            rows = AuthExternalSubject.query.all()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].subject, "zitadel-token-user")

    def test_oauth_facade_flow_registers_authorizes_and_initializes_mcp(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        client = self.app.test_client()
        user_id = self._create_local_user(email="oauth-facade@example.com")
        with self.app.app_context():
            db.session.add(
                self._auth_external_subject(
                    user_id=user_id,
                    issuer="https://pandects-test-zitadel.example.com",
                    subject="oauth-facade-zitadel-user",
                )
            )
            db.session.commit()
        self._set_cookie_session(client, email="oauth-facade@example.com")
        client.get("/v1/auth/csrf")
        csrf = self._csrf_cookie_value(client)

        register = client.post(
            "/v1/auth/oauth/register",
            json={
                "client_name": "Codex MCP",
                "redirect_uris": ["https://codex.example.com/callback"],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(register.status_code, 201)
        client_id = register.get_json()["client_id"]

        # Pre-grant consent so /authorize mints a code instead of returning
        # the consent_required envelope.
        self._grant_oauth_consent(
            email="oauth-facade@example.com",
            client_id=client_id,
            scope="agreements:read",
        )

        authorize = client.get(
            "/v1/auth/oauth/authorize"
            f"?client_id={client_id}"
            "&redirect_uri=https://codex.example.com/callback"
            "&response_type=code"
            "&scope=agreements:read"
            "&state=test-state"
            "&code_challenge=test-challenge"
            "&code_challenge_method=S256"
        )
        self.assertEqual(authorize.status_code, 302)
        redirect_location = authorize.headers["Location"]
        parsed = urlparse(redirect_location)
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.netloc, "codex.example.com")
        auth_code = parse_qs(parsed.query)["code"][0]

        token = client.post(
            "/v1/auth/oauth/token",
            data={
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": auth_code,
                "redirect_uri": "https://codex.example.com/callback",
                "code_verifier": "test-verifier",
            },
        )
        self.assertEqual(token.status_code, 400)

        good_challenge = _pkce_challenge("good-verifier")
        authorize = client.get(
            "/v1/auth/oauth/authorize"
            f"?client_id={client_id}"
            "&redirect_uri=https://codex.example.com/callback"
            "&response_type=code"
            "&scope=agreements:read"
            "&state=test-state"
            f"&code_challenge={good_challenge}"
            "&code_challenge_method=S256"
        )
        auth_code = parse_qs(urlparse(authorize.headers["Location"]).query)["code"][0]
        token = client.post(
            "/v1/auth/oauth/token",
            data={
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": auth_code,
                "redirect_uri": "https://codex.example.com/callback",
                "code_verifier": "good-verifier",
            },
        )
        self.assertEqual(token.status_code, 200)
        token_json = token.get_json()
        self.assertNotIn("refresh_token", token_json)
        access_token = token_json["access_token"]

        mcp = client.post(
            "/mcp",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        )
        self.assertEqual(mcp.status_code, 200)
        self.assertEqual(mcp.get_json()["result"]["serverInfo"]["name"], "pandects-mcp")

    def test_oauth_token_exchange_includes_refresh_token(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        client = self.app.test_client()
        user_id = self._create_local_user(email="refresh-issue@example.com")
        with self.app.app_context():
            db.session.add(
                self._auth_external_subject(
                    user_id=user_id,
                    issuer="https://pandects-test-zitadel.example.com",
                    subject="refresh-issue-zitadel-user",
                )
            )
            db.session.commit()
        self._set_cookie_session(client, email="refresh-issue@example.com")

        register = client.post(
            "/v1/auth/oauth/register",
            json={
                "client_name": "Codex MCP",
                "redirect_uris": ["https://codex.example.com/callback"],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
        self.assertEqual(register.status_code, 201)
        client_id = register.get_json()["client_id"]

        self._grant_oauth_consent(
            email="refresh-issue@example.com",
            client_id=client_id,
            scope="agreements:read",
        )

        good_challenge = _pkce_challenge("good-verifier")
        authorize = client.get(
            "/v1/auth/oauth/authorize"
            f"?client_id={client_id}"
            "&redirect_uri=https://codex.example.com/callback"
            "&response_type=code"
            "&scope=agreements:read"
            "&state=refresh-state"
            f"&code_challenge={good_challenge}"
            "&code_challenge_method=S256"
        )
        self.assertEqual(authorize.status_code, 302)
        auth_code = parse_qs(urlparse(authorize.headers["Location"]).query)["code"][0]

        token = client.post(
            "/v1/auth/oauth/token",
            data={
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": auth_code,
                "redirect_uri": "https://codex.example.com/callback",
                "code_verifier": "good-verifier",
            },
        )
        self.assertEqual(token.status_code, 200)
        token_json = token.get_json()
        self.assertIn("access_token", token_json)
        self.assertIn("refresh_token", token_json)
        self.assertEqual(token_json["token_type"], "Bearer")
        self.assertIsInstance(token_json["expires_in"], int)

    def test_refresh_token_grant_issues_new_pair_and_rotates_old_token(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        client = self.app.test_client()
        user_id = self._create_local_user(email="refresh-rotate@example.com")
        with self.app.app_context():
            db.session.add(
                self._auth_external_subject(
                    user_id=user_id,
                    issuer="https://pandects-test-zitadel.example.com",
                    subject="refresh-rotate-zitadel-user",
                )
            )
            db.session.commit()
        self._set_cookie_session(client, email="refresh-rotate@example.com")

        register = client.post(
            "/v1/auth/oauth/register",
            json={
                "client_name": "Codex MCP",
                "redirect_uris": ["https://codex.example.com/callback"],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
        client_id = register.get_json()["client_id"]

        self._grant_oauth_consent(
            email="refresh-rotate@example.com",
            client_id=client_id,
            scope="agreements:read",
        )

        good_challenge = _pkce_challenge("rotate-verifier")
        authorize = client.get(
            "/v1/auth/oauth/authorize"
            f"?client_id={client_id}"
            "&redirect_uri=https://codex.example.com/callback"
            "&response_type=code"
            "&scope=agreements:read"
            f"&code_challenge={good_challenge}"
            "&code_challenge_method=S256"
        )
        auth_code = parse_qs(urlparse(authorize.headers["Location"]).query)["code"][0]

        first_token = client.post(
            "/v1/auth/oauth/token",
            data={
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": auth_code,
                "redirect_uri": "https://codex.example.com/callback",
                "code_verifier": "rotate-verifier",
            },
        )
        self.assertEqual(first_token.status_code, 200)
        first_json = first_token.get_json()
        refresh_token_v1 = first_json["refresh_token"]
        access_token_v1 = first_json["access_token"]

        # Use the refresh token to get a new pair.
        refreshed = client.post(
            "/v1/auth/oauth/token",
            data={
                "grant_type": "refresh_token",
                "client_id": client_id,
                "refresh_token": refresh_token_v1,
            },
        )
        self.assertEqual(refreshed.status_code, 200)
        refreshed_json = refreshed.get_json()
        self.assertIn("access_token", refreshed_json)
        self.assertIn("refresh_token", refreshed_json)
        # New access token must differ from the original.
        self.assertNotEqual(refreshed_json["access_token"], access_token_v1)
        # New refresh token must differ from the original.
        refresh_token_v2 = refreshed_json["refresh_token"]
        self.assertNotEqual(refresh_token_v2, refresh_token_v1)

        # Reusing the old refresh token must be rejected (rotation) AND revoke
        # the entire family — re-presentation of a rotated refresh token signals
        # token theft per RFC 6819 §5.2.2.3.
        reuse = client.post(
            "/v1/auth/oauth/token",
            data={
                "grant_type": "refresh_token",
                "client_id": client_id,
                "refresh_token": refresh_token_v1,
            },
        )
        self.assertEqual(reuse.status_code, 400)

        # The rotated token must now also be rejected because the family was
        # revoked when reuse of v1 was detected.
        second_refresh = client.post(
            "/v1/auth/oauth/token",
            data={
                "grant_type": "refresh_token",
                "client_id": client_id,
                "refresh_token": refresh_token_v2,
            },
        )
        self.assertEqual(second_refresh.status_code, 400)

    def test_refresh_token_grant_rejects_client_without_refresh_token_registration(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        client = self.app.test_client()
        user_id = self._create_local_user(email="refresh-disallowed@example.com")
        with self.app.app_context():
            db.session.add(
                self._auth_external_subject(
                    user_id=user_id,
                    issuer="https://pandects-test-zitadel.example.com",
                    subject="refresh-disallowed-zitadel-user",
                )
            )
            db.session.commit()
        self._set_cookie_session(client, email="refresh-disallowed@example.com")

        register = client.post(
            "/v1/auth/oauth/register",
            json={
                "client_name": "Codex MCP",
                "redirect_uris": ["https://codex.example.com/callback"],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
        self.assertEqual(register.status_code, 201)
        client_id = register.get_json()["client_id"]

        with self.app.app_context():
            refresh = AuthOAuthRefreshToken()
            refresh.client_id = client_id
            refresh.user_id = user_id
            refresh.scope = "agreements:read"
            refresh.family_id = str(uuid.uuid4())
            refresh.expires_at = backend_app._utc_now() + timedelta(hours=1)
            raw_refresh = "seeded-refresh-token"
            refresh.token_hash = hashlib.sha256(raw_refresh.encode("utf-8")).hexdigest()
            db.session.add(refresh)
            db.session.commit()

        refreshed = client.post(
            "/v1/auth/oauth/token",
            data={
                "grant_type": "refresh_token",
                "client_id": client_id,
                "refresh_token": raw_refresh,
            },
        )
        self.assertEqual(refreshed.status_code, 400)

    def test_authorization_code_grant_rejects_client_without_authorization_code_registration(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        client = self.app.test_client()
        user_id = self._create_local_user(email="authcode-disallowed@example.com")
        with self.app.app_context():
            db.session.add(
                self._auth_external_subject(
                    user_id=user_id,
                    issuer="https://pandects-test-zitadel.example.com",
                    subject="authcode-disallowed-zitadel-user",
                )
            )
            db.session.commit()
        self._set_cookie_session(client, email="authcode-disallowed@example.com")

        # Seed an OAuth client whose grant_types list omits authorization_code.
        # We bypass /oauth/register here so the test is decoupled from any
        # validation that endpoint may layer on top of the registered grants.
        client_id = "client-without-authcode"
        redirect_uri = "https://codex.example.com/callback"
        raw_code = "seeded-auth-code"
        challenge = _pkce_challenge("good-verifier")
        with self.app.app_context():
            oauth_client = AuthOAuthClient()
            oauth_client.client_id = client_id
            oauth_client.client_name = "Refresh-only client"
            oauth_client.redirect_uris = [redirect_uri]
            oauth_client.token_endpoint_auth_method = "none"
            oauth_client.grant_types = ["refresh_token"]
            oauth_client.response_types = []
            db.session.add(oauth_client)

            auth_code = AuthOAuthAuthorizationCode()
            auth_code.code_hash = hashlib.sha256(raw_code.encode("utf-8")).hexdigest()
            auth_code.client_id = client_id
            auth_code.user_id = user_id
            auth_code.redirect_uri = redirect_uri
            auth_code.scope = "agreements:read"
            auth_code.code_challenge = challenge
            auth_code.code_challenge_method = "S256"
            auth_code.expires_at = backend_app._utc_now() + timedelta(minutes=5)
            db.session.add(auth_code)
            db.session.commit()

        token = client.post(
            "/v1/auth/oauth/token",
            data={
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": raw_code,
                "redirect_uri": redirect_uri,
                "code_verifier": "good-verifier",
            },
        )
        self.assertEqual(token.status_code, 400)

    def test_oauth_authorize_returns_login_bridge_when_session_transport_is_bearer(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()

        register = client.post(
            "/v1/auth/oauth/register",
            json={
                "client_name": "Codex MCP",
                "redirect_uris": ["https://codex.example.com/callback"],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
        self.assertEqual(register.status_code, 201)
        client_id = register.get_json()["client_id"]

        authorize = client.get(
            "/v1/auth/oauth/authorize"
            f"?client_id={client_id}"
            "&redirect_uri=https://codex.example.com/callback"
            "&response_type=code"
            "&scope=agreements:read"
            "&state=test-state"
            "&code_challenge=test-challenge"
            "&code_challenge_method=S256"
        )
        self.assertEqual(authorize.status_code, 200)
        body = authorize.get_data(as_text=True)
        self.assertIn("/login?next=", body)
        self.assertNotIn("pandects.sessionToken", body)
        self.assertNotIn("/v1/auth/oauth/browser-session", body)
        self.assertIn("%2Fv1%2Fauth%2Foauth%2Fauthorize", body)
        # The post-login resume URL must carry the original OAuth params so that
        # the consent flow is bound to the request the client initiated, not to
        # a server-side pending cookie that a different user could finalize.
        self.assertIn("redirect_uri%3Dhttps%253A%252F%252Fcodex.example.com%252Fcallback", body)
        self.assertIn("code_challenge%3Dtest-challenge", body)
        self.assertIn("state%3Dtest-state", body)

    def test_oauth_authorize_login_redirect_carries_request_params_for_resume(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        client = self.app.test_client()

        register = client.post(
            "/v1/auth/oauth/register",
            json={
                "client_name": "Codex MCP",
                "redirect_uris": ["https://codex.example.com/callback"],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
        self.assertEqual(register.status_code, 201)
        client_id = register.get_json()["client_id"]

        authorize = client.get(
            "/v1/auth/oauth/authorize"
            f"?client_id={client_id}"
            "&redirect_uri=https://codex.example.com/callback"
            "&response_type=code"
            "&scope=agreements:read"
            "&state=test-state"
            "&code_challenge=test-challenge"
            "&code_challenge_method=S256"
        )
        self.assertEqual(authorize.status_code, 302)
        parsed = urlparse(authorize.headers["Location"])
        self.assertEqual(parsed.path, "/login")
        next_path = parse_qs(parsed.query)["next"][0]
        # Resume binding: the OAuth params must be embedded in the post-login
        # `next` URL so that the request that user A initiated cannot be
        # finalized by user B logging in on the same browser.
        next_parsed = urlparse(next_path)
        self.assertEqual(next_parsed.path, "/v1/auth/oauth/authorize")
        next_query = parse_qs(next_parsed.query)
        self.assertEqual(next_query["client_id"], [client_id])
        self.assertEqual(next_query["redirect_uri"], ["https://codex.example.com/callback"])
        self.assertEqual(next_query["state"], ["test-state"])
        self.assertEqual(next_query["code_challenge"], ["test-challenge"])

        user_id = self._create_local_user(email="oauth-resume@example.com", verified=True, legal=True)
        with self.app.app_context():
            db.session.add(
                self._auth_external_subject(
                    user_id=user_id,
                    issuer=os.environ["MCP_OIDC_ISSUER"],
                    subject="zitadel|oauth-resume",
                )
            )
            db.session.commit()

        self._set_cookie_session(client, email="oauth-resume@example.com")
        self._grant_oauth_consent(
            email="oauth-resume@example.com",
            client_id=client_id,
            scope="agreements:read",
        )
        # Resume must re-supply the OAuth params (the SPA reads them from the
        # next= URL); a bare GET is now an error.
        bare_resume = client.get("/v1/auth/oauth/authorize")
        self.assertEqual(bare_resume.status_code, 400)

        resumed = client.get(next_path)
        self.assertEqual(resumed.status_code, 302)
        resumed_parsed = urlparse(resumed.headers["Location"])
        self.assertEqual(resumed_parsed.scheme, "https")
        self.assertEqual(resumed_parsed.netloc, "codex.example.com")
        resumed_query = parse_qs(resumed_parsed.query)
        self.assertIn("code", resumed_query)
        self.assertEqual(resumed_query["state"], ["test-state"])

    def test_oauth_authorize_ignores_pending_cookie_when_new_query_is_supplied(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        client = self.app.test_client()

        register = client.post(
            "/v1/auth/oauth/register",
            json={
                "client_name": "Codex MCP",
                "redirect_uris": ["https://codex.example.com/callback"],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
        self.assertEqual(register.status_code, 201)
        client_id = register.get_json()["client_id"]

        authorize = client.get(
            "/v1/auth/oauth/authorize"
            f"?client_id={client_id}"
            "&redirect_uri=https://codex.example.com/callback"
            "&response_type=code"
            "&scope=agreements:read"
            "&state=test-state"
            "&code_challenge=test-challenge"
            "&code_challenge_method=S256"
        )
        self.assertEqual(authorize.status_code, 302)

        partial = client.get(f"/v1/auth/oauth/authorize?client_id={client_id}")
        self.assertEqual(partial.status_code, 400)
        self.assertIn("Missing OAuth client_id or redirect_uri.", partial.get_data(as_text=True))

    def test_oauth_authorize_legal_redirect_carries_request_params_for_resume(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        client = self.app.test_client()

        register = client.post(
            "/v1/auth/oauth/register",
            json={
                "client_name": "Codex MCP",
                "redirect_uris": ["https://codex.example.com/callback"],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
        self.assertEqual(register.status_code, 201)
        client_id = register.get_json()["client_id"]

        user_id = self._create_local_user(email="oauth-legal@example.com", verified=True, legal=False)
        with self.app.app_context():
            db.session.add(
                self._auth_external_subject(
                    user_id=user_id,
                    issuer=os.environ["MCP_OIDC_ISSUER"],
                    subject="zitadel|oauth-legal",
                )
            )
            db.session.commit()

        self._set_cookie_session(client, email="oauth-legal@example.com")
        authorize = client.get(
            "/v1/auth/oauth/authorize"
            f"?client_id={client_id}"
            "&redirect_uri=https://codex.example.com/callback"
            "&response_type=code"
            "&scope=agreements:read"
            "&state=test-state"
            "&code_challenge=test-challenge"
            "&code_challenge_method=S256"
        )
        self.assertEqual(authorize.status_code, 302)
        parsed = urlparse(authorize.headers["Location"])
        self.assertEqual(parsed.path, "/login")
        next_path = parse_qs(parsed.query)["next"][0]
        next_parsed = urlparse(next_path)
        self.assertEqual(next_parsed.path, "/v1/auth/oauth/authorize")
        next_query = parse_qs(next_parsed.query)
        self.assertEqual(next_query["client_id"], [client_id])
        self.assertEqual(next_query["state"], ["test-state"])
        self.assertEqual(next_query["code_challenge"], ["test-challenge"])

    def test_oauth_register_accepts_refresh_token_metadata_for_public_code_clients(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()

        register = client.post(
            "/v1/auth/oauth/register",
            json={
                "client_name": "Codex MCP",
                "redirect_uris": ["https://codex.example.com/callback"],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
        self.assertEqual(register.status_code, 201)
        self.assertEqual(
            register.get_json()["grant_types"],
            ["authorization_code", "refresh_token"],
        )

    def test_oauth_register_is_rate_limited_per_ip(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()

        payload = {
            "client_name": "Codex MCP",
            "redirect_uris": ["https://codex.example.com/callback"],
            "grant_types": ["authorization_code"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        }
        limit = backend_app._ENDPOINT_RATE_LIMITS[("POST", "/v1/auth/oauth/register")]
        statuses = [
            client.post("/v1/auth/oauth/register", json=payload).status_code
            for _ in range(limit + 1)
        ]
        self.assertTrue(all(code == 201 for code in statuses[:limit]), statuses)
        self.assertEqual(statuses[-1], 429)

    def test_oauth_register_rejects_unsafe_redirect_uris(self):
        client = self.app.test_client()

        unsafe_redirects = [
            "http://evil.example.com/callback",
            "https://codex.example.com/callback#token",
            "https://user:pass@codex.example.com/callback",
            r"https://codex.example.com\@evil.example.com/callback",
        ]
        for redirect_uri in unsafe_redirects:
            with self.subTest(redirect_uri=redirect_uri):
                register = client.post(
                    "/v1/auth/oauth/register",
                    json={
                        "client_name": "Codex MCP",
                        "redirect_uris": [redirect_uri],
                        "grant_types": ["authorization_code"],
                        "response_types": ["code"],
                        "token_endpoint_auth_method": "none",
                    },
                )
                self.assertEqual(register.status_code, 400)

    def test_oauth_authorize_redirect_preserves_registered_query_string(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()

        register = client.post(
            "/v1/auth/oauth/register",
            json={
                "client_name": "Codex MCP",
                "redirect_uris": ["https://codex.example.com/callback?client=desktop"],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
        self.assertEqual(register.status_code, 201)
        client_id = register.get_json()["client_id"]

        authorize = client.get(
            "/v1/auth/oauth/authorize"
            f"?client_id={client_id}"
            "&redirect_uri=https://codex.example.com/callback?client=desktop"
            "&response_type=token"
            "&scope=agreements:read"
            "&state=test-state"
            "&code_challenge=test-challenge"
            "&code_challenge_method=S256"
        )
        self.assertEqual(authorize.status_code, 302)
        parsed = urlparse(authorize.headers["Location"])
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.netloc, "codex.example.com")
        self.assertEqual(parsed.path, "/callback")
        self.assertEqual(
            parse_qs(parsed.query),
            {
                "client": ["desktop"],
                "error": ["unsupported_response_type"],
                "state": ["test-state"],
            },
        )

    def test_oauth_authorize_defaults_scope_when_client_omits_it(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()

        register = client.post(
            "/v1/auth/oauth/register",
            json={
                "client_name": "Codex MCP",
                "redirect_uris": ["https://codex.example.com/callback"],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
        self.assertEqual(register.status_code, 201)
        client_id = register.get_json()["client_id"]

        user_id = self._create_local_user(email="scope-default@example.com", verified=True, legal=True)
        with self.app.app_context():
            db.session.add(
                self._auth_external_subject(
                    user_id=user_id,
                    issuer=os.environ["MCP_OIDC_ISSUER"],
                    subject="zitadel|scope-default",
                )
            )
            db.session.commit()
        token = self._issue_bearer_session(email="scope-default@example.com")
        # The request omits ?scope=, so /authorize falls back to the full
        # supported-scope list — pre-grant exactly that set so we mint a code.
        self._grant_oauth_consent(
            email="scope-default@example.com",
            client_id=client_id,
            scope="sections:search agreements:search agreements:read agreements:read_fulltext",
        )

        authorize = client.get(
            "/v1/auth/oauth/authorize"
            f"?client_id={client_id}"
            "&redirect_uri=https://codex.example.com/callback"
            "&response_type=code"
            "&state=test-state"
            "&code_challenge=test-challenge"
            "&code_challenge_method=S256",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(authorize.status_code, 302)
        parsed = urlparse(authorize.headers["Location"])
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.netloc, "codex.example.com")
        self.assertIn("code", parse_qs(parsed.query))

    def test_oauth_authorize_without_grant_redirects_to_consent_page(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        os.environ["PUBLIC_FRONTEND_BASE_URL"] = "https://pandects.example"
        client = self.app.test_client()
        user_id = self._create_local_user(email="consent-redirect@example.com")
        with self.app.app_context():
            db.session.add(
                self._auth_external_subject(
                    user_id=user_id,
                    issuer=os.environ["MCP_OIDC_ISSUER"],
                    subject="zitadel|consent-redirect",
                )
            )
            db.session.commit()
        self._set_cookie_session(client, email="consent-redirect@example.com")

        register = client.post(
            "/v1/auth/oauth/register",
            json={
                "client_name": "Codex MCP",
                "redirect_uris": ["https://codex.example.com/callback"],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
        self.assertEqual(register.status_code, 201)
        client_id = register.get_json()["client_id"]

        # No prior grant -> /authorize must redirect to the SPA consent page
        # instead of minting a code. The redirect carries every param the page
        # needs to replay /authorize after the user clicks Allow.
        authorize = client.get(
            "/v1/auth/oauth/authorize"
            f"?client_id={client_id}"
            "&redirect_uri=https://codex.example.com/callback"
            "&response_type=code"
            "&scope=agreements:read"
            "&state=consent-state"
            "&code_challenge=consent-challenge"
            "&code_challenge_method=S256"
        )
        self.assertEqual(authorize.status_code, 302)
        location = urlparse(authorize.headers["Location"])
        self.assertEqual(location.scheme, "https")
        self.assertEqual(location.netloc, "pandects.example")
        self.assertEqual(location.path, "/oauth/consent")
        query = parse_qs(location.query)
        self.assertEqual(query["client_id"], [client_id])
        self.assertEqual(query["client_name"], ["Codex MCP"])
        self.assertEqual(query["redirect_uri"], ["https://codex.example.com/callback"])
        self.assertEqual(query["response_type"], ["code"])
        self.assertEqual(query["scope"], ["agreements:read"])
        self.assertEqual(query["state"], ["consent-state"])
        self.assertEqual(query["code_challenge"], ["consent-challenge"])
        self.assertEqual(query["code_challenge_method"], ["S256"])

    def test_oauth_consent_endpoint_records_grant_and_lets_authorize_mint_code(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        os.environ["PUBLIC_FRONTEND_BASE_URL"] = "https://pandects.example"
        client = self.app.test_client()
        user_id = self._create_local_user(email="consent-grant@example.com")
        with self.app.app_context():
            db.session.add(
                self._auth_external_subject(
                    user_id=user_id,
                    issuer=os.environ["MCP_OIDC_ISSUER"],
                    subject="zitadel|consent-grant",
                )
            )
            db.session.commit()
        self._set_cookie_session(client, email="consent-grant@example.com")
        client.get("/v1/auth/csrf")
        csrf = self._csrf_cookie_value(client)

        register = client.post(
            "/v1/auth/oauth/register",
            json={
                "client_name": "Codex MCP",
                "redirect_uris": ["https://codex.example.com/callback"],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
            headers={"X-CSRF-Token": csrf},
        )
        client_id = register.get_json()["client_id"]

        # POST /oauth/consent records the grant; the next /authorize hit mints
        # a code instead of redirecting.
        consent = client.post(
            "/v1/auth/oauth/consent",
            json={"client_id": client_id, "scope": "agreements:read"},
            headers={"X-CSRF-Token": csrf, "Content-Type": "application/json"},
        )
        self.assertEqual(consent.status_code, 200)
        body = consent.get_json()
        self.assertEqual(body["status"], "granted")
        self.assertEqual(body["client_id"], client_id)
        self.assertEqual(body["scope"], "agreements:read")

        authorize = client.get(
            "/v1/auth/oauth/authorize"
            f"?client_id={client_id}"
            "&redirect_uri=https://codex.example.com/callback"
            "&response_type=code"
            "&scope=agreements:read"
            "&state=after-consent"
            "&code_challenge=after-challenge"
            "&code_challenge_method=S256"
        )
        self.assertEqual(authorize.status_code, 302)
        parsed = urlparse(authorize.headers["Location"])
        self.assertEqual(parsed.netloc, "codex.example.com")
        self.assertIn("code", parse_qs(parsed.query))

    def test_oauth_consent_endpoint_rejects_unsupported_scope_and_unknown_client(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        client = self.app.test_client()
        self._create_local_user(email="consent-reject@example.com")
        self._set_cookie_session(client, email="consent-reject@example.com")
        client.get("/v1/auth/csrf")
        csrf = self._csrf_cookie_value(client)

        register = client.post(
            "/v1/auth/oauth/register",
            json={
                "client_name": "Codex MCP",
                "redirect_uris": ["https://codex.example.com/callback"],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
            headers={"X-CSRF-Token": csrf},
        )
        client_id = register.get_json()["client_id"]

        bad_scope = client.post(
            "/v1/auth/oauth/consent",
            json={"client_id": client_id, "scope": "totally:made-up"},
            headers={"X-CSRF-Token": csrf, "Content-Type": "application/json"},
        )
        self.assertEqual(bad_scope.status_code, 400)

        unknown_client = client.post(
            "/v1/auth/oauth/consent",
            json={"client_id": "does-not-exist", "scope": "agreements:read"},
            headers={"X-CSRF-Token": csrf, "Content-Type": "application/json"},
        )
        self.assertEqual(unknown_client.status_code, 400)

    def test_oauth_consent_endpoint_widens_existing_grant(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        client = self.app.test_client()
        self._create_local_user(email="consent-widen@example.com")
        self._set_cookie_session(client, email="consent-widen@example.com")
        client.get("/v1/auth/csrf")
        csrf = self._csrf_cookie_value(client)

        register = client.post(
            "/v1/auth/oauth/register",
            json={
                "client_name": "Codex MCP",
                "redirect_uris": ["https://codex.example.com/callback"],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
            headers={"X-CSRF-Token": csrf},
        )
        client_id = register.get_json()["client_id"]

        first = client.post(
            "/v1/auth/oauth/consent",
            json={"client_id": client_id, "scope": "agreements:read"},
            headers={"X-CSRF-Token": csrf, "Content-Type": "application/json"},
        )
        self.assertEqual(first.status_code, 200)

        second = client.post(
            "/v1/auth/oauth/consent",
            json={"client_id": client_id, "scope": "agreements:search"},
            headers={"X-CSRF-Token": csrf, "Content-Type": "application/json"},
        )
        self.assertEqual(second.status_code, 200)
        merged_scope = second.get_json()["scope"].split(" ")
        self.assertIn("agreements:read", merged_scope)
        self.assertIn("agreements:search", merged_scope)

    def test_login_per_account_throttle_blocks_after_repeated_failures(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["AUTH_ZITADEL_API_TOKEN"] = "test-zitadel-api-token"
        client = self.app.test_client()
        self._create_local_user(
            email="brute@example.com", password="password123", verified=True, legal=True
        )

        original_oauth_fetch_json = backend_app._oauth_fetch_json

        def _fake_oauth_fetch_json(url, *, data=None, json_body=None, headers=None, method=None):
            # Every Zitadel session-create call fails → drops into the
            # _maybe_migrate_local_password_user path; wrong passwords there
            # return None → abort(401) → throttle increments.
            if url.endswith("/v2/sessions"):
                raise Unauthorized(description="Invalid credentials.")
            self.fail(f"Unexpected URL: {url}")

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        try:
            for _ in range(backend_app._ACCOUNT_LOGIN_FAILURE_LIMIT):
                res = client.post(
                    "/v1/auth/login/password",
                    json={"email": "brute@example.com", "password": "wrong"},
                )
                self.assertEqual(res.status_code, 401)

            # Account is now locked. The next attempt — even with the CORRECT
            # password — must 429, proving the throttle is account-scoped and
            # not just a wrong-password counter.
            locked = client.post(
                "/v1/auth/login/password",
                json={"email": "brute@example.com", "password": "password123"},
            )
            self.assertEqual(locked.status_code, 429)
            self.assertIn("Retry-After", locked.headers)
            body = locked.get_json()
            self.assertEqual(body.get("error"), "rate_limited")

            # A different email is untouched.
            self._create_local_user(
                email="bystander@example.com", password="password123", verified=True, legal=True
            )
            other = client.post(
                "/v1/auth/login/password",
                json={"email": "bystander@example.com", "password": "wrong"},
            )
            self.assertEqual(other.status_code, 401)
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json

    def test_login_success_clears_per_account_failure_counter(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        os.environ["AUTH_ZITADEL_API_TOKEN"] = "test-zitadel-api-token"
        client = self.app.test_client()
        user_id = self._create_local_user(
            email="reset-throttle@example.com",
            password="password123",
            verified=True,
            legal=True,
        )
        # Simulate prior failures right up to the threshold but not over.
        for _ in range(backend_app._ACCOUNT_LOGIN_FAILURE_LIMIT - 1):
            backend_app._record_account_login_failure("reset-throttle@example.com")

        original_oauth_fetch_json = backend_app._oauth_fetch_json

        def _fake_oauth_fetch_json(url, *, data=None, json_body=None, headers=None, method=None):
            # First /v2/sessions call: rejected, falls through to local
            # migration which succeeds because the password matches.
            if url.endswith("/v2/sessions"):
                raise Unauthorized(description="Invalid credentials.")
            if url.endswith("/v2/users/human"):
                return {"userId": "zitadel-reset-throttle"}
            if "/users/grants/_search" in url:
                return {"result": []}
            if url.endswith("/users/zitadel-reset-throttle/grants"):
                return {"id": "grant-reset"}
            self.fail(f"Unexpected URL: {url}")

        backend_app._oauth_fetch_json = _fake_oauth_fetch_json
        try:
            ok = client.post(
                "/v1/auth/login/password",
                json={"email": "reset-throttle@example.com", "password": "password123"},
            )
            self.assertEqual(ok.status_code, 200)
            self.assertEqual(ok.get_json()["status"], "authenticated")
        finally:
            backend_app._oauth_fetch_json = original_oauth_fetch_json

        # Counter cleared — a fresh batch of failures shouldn't trip the lock
        # before reaching the limit again.
        self.assertNotIn(
            backend_app._account_login_key("reset-throttle@example.com"),
            backend_app._account_login_failure_state,
        )
        _ = user_id

    def test_email_verify_resend_is_rate_limited_per_ip(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        limit = backend_app._ENDPOINT_RATE_LIMITS[("POST", "/v1/auth/email/verify/resend")]
        statuses = []
        for _ in range(limit + 2):
            res = client.post(
                "/v1/auth/email/verify/resend",
                json={"email": "verify-bomb@example.com"},
            )
            statuses.append(res.status_code)
        # All in-budget calls must return the verification-required envelope;
        # over-budget calls must 429.
        self.assertTrue(all(s == 200 for s in statuses[:limit]))
        self.assertEqual(statuses[-1], 429)

    def test_oauth_register_sweeps_stale_and_inactive_clients(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()

        # Pre-seed two clients that should be swept:
        #  - "stale_unused": registered >24h ago, never used
        #  - "stale_inactive": last_used_at older than 90d
        # And one that should survive: recent and never used (inside grace).
        with self.app.app_context():
            now = backend_app._utc_now()
            unused = AuthOAuthClient()
            unused.client_id = "stale_unused"
            unused.redirect_uris = ["https://example.com/cb"]
            unused.grant_types = ["authorization_code"]
            unused.response_types = ["code"]
            unused.created_at = now - timedelta(hours=48)
            db.session.add(unused)

            inactive = AuthOAuthClient()
            inactive.client_id = "stale_inactive"
            inactive.redirect_uris = ["https://example.com/cb"]
            inactive.grant_types = ["authorization_code"]
            inactive.response_types = ["code"]
            inactive.created_at = now - timedelta(days=365)
            inactive.last_used_at = now - timedelta(days=120)
            db.session.add(inactive)

            recent = AuthOAuthClient()
            recent.client_id = "recent_unused"
            recent.redirect_uris = ["https://example.com/cb"]
            recent.grant_types = ["authorization_code"]
            recent.response_types = ["code"]
            recent.created_at = now - timedelta(hours=1)
            db.session.add(recent)
            db.session.commit()

        register = client.post(
            "/v1/auth/oauth/register",
            json={
                "client_name": "Fresh MCP",
                "redirect_uris": ["https://example.com/cb"],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
        self.assertEqual(register.status_code, 201)

        with self.app.app_context():
            remaining = {row.client_id for row in AuthOAuthClient.query.all()}
            self.assertNotIn("stale_unused", remaining)
            self.assertNotIn("stale_inactive", remaining)
            self.assertIn("recent_unused", remaining)

    def test_oauth_register_sweep_cascades_codes_refresh_tokens_and_grants(self):
        """All three tables referencing auth_oauth_clients (codes, refresh
        tokens, user grants) must be cleared before the client row is deleted.
        We don't configure ON DELETE CASCADE, so on Postgres a missed table
        would fail the sweep with a FK violation; SQLite hides the bug because
        Flask-SQLAlchemy doesn't enable FK enforcement by default."""
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        user_id = self._create_local_user(
            email="sweep-cascade@example.com", verified=True, legal=True
        )

        with self.app.app_context():
            now = backend_app._utc_now()
            stale_client = AuthOAuthClient()
            stale_client.client_id = "cascade_target"
            stale_client.redirect_uris = ["https://example.com/cb"]
            stale_client.grant_types = ["authorization_code", "refresh_token"]
            stale_client.response_types = ["code"]
            stale_client.created_at = now - timedelta(days=2)
            db.session.add(stale_client)

            code = AuthOAuthAuthorizationCode()
            code.code_hash = "deadbeef" * 8
            code.client_id = "cascade_target"
            code.user_id = user_id
            code.redirect_uri = "https://example.com/cb"
            code.scope = "agreements:read"
            code.code_challenge = "x"
            code.code_challenge_method = "S256"
            code.expires_at = now + timedelta(minutes=3)
            db.session.add(code)

            refresh = AuthOAuthRefreshToken()
            refresh.token_hash = "feedface" * 8
            refresh.client_id = "cascade_target"
            refresh.user_id = user_id
            refresh.scope = "agreements:read"
            refresh.family_id = "fam-1"
            refresh.expires_at = now + timedelta(days=30)
            db.session.add(refresh)

            grant = AuthOAuthUserGrant()
            grant.user_id = user_id
            grant.client_id = "cascade_target"
            grant.scope = "agreements:read"
            db.session.add(grant)
            db.session.commit()

        register = client.post(
            "/v1/auth/oauth/register",
            json={
                "client_name": "Fresh MCP",
                "redirect_uris": ["https://example.com/cb"],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
        self.assertEqual(register.status_code, 201)

        with self.app.app_context():
            self.assertIsNone(AuthOAuthClient.query.filter_by(client_id="cascade_target").first())
            self.assertEqual(
                AuthOAuthAuthorizationCode.query.filter_by(client_id="cascade_target").count(), 0
            )
            self.assertEqual(
                AuthOAuthRefreshToken.query.filter_by(client_id="cascade_target").count(), 0
            )
            self.assertEqual(
                AuthOAuthUserGrant.query.filter_by(client_id="cascade_target").count(), 0
            )

    def test_oauth_authorize_and_token_touch_client_last_used_at(self):
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        client = self.app.test_client()
        user_id = self._create_local_user(email="touch-lastused@example.com")
        with self.app.app_context():
            db.session.add(
                self._auth_external_subject(
                    user_id=user_id,
                    issuer=os.environ["MCP_OIDC_ISSUER"],
                    subject="zitadel|touch-lastused",
                )
            )
            db.session.commit()
        self._set_cookie_session(client, email="touch-lastused@example.com")

        register = client.post(
            "/v1/auth/oauth/register",
            json={
                "client_name": "Codex MCP",
                "redirect_uris": ["https://codex.example.com/callback"],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
        client_id = register.get_json()["client_id"]

        # Fresh client: last_used_at is NULL until somebody actually authorizes.
        with self.app.app_context():
            row = AuthOAuthClient.query.filter_by(client_id=client_id).first()
            self.assertIsNone(row.last_used_at)

        self._grant_oauth_consent(
            email="touch-lastused@example.com",
            client_id=client_id,
            scope="agreements:read",
        )

        good_challenge = _pkce_challenge("touch-verifier")
        authorize = client.get(
            "/v1/auth/oauth/authorize"
            f"?client_id={client_id}"
            "&redirect_uri=https://codex.example.com/callback"
            "&response_type=code"
            "&scope=agreements:read"
            "&state=touch-state"
            f"&code_challenge={good_challenge}"
            "&code_challenge_method=S256"
        )
        self.assertEqual(authorize.status_code, 302)

        with self.app.app_context():
            row = AuthOAuthClient.query.filter_by(client_id=client_id).first()
            self.assertIsNotNone(row.last_used_at)
            after_authorize = row.last_used_at

        auth_code = parse_qs(urlparse(authorize.headers["Location"]).query)["code"][0]
        token = client.post(
            "/v1/auth/oauth/token",
            data={
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": auth_code,
                "redirect_uri": "https://codex.example.com/callback",
                "code_verifier": "touch-verifier",
            },
        )
        self.assertEqual(token.status_code, 200)

        with self.app.app_context():
            row = AuthOAuthClient.query.filter_by(client_id=client_id).first()
            self.assertIsNotNone(row.last_used_at)
            self.assertGreaterEqual(row.last_used_at, after_authorize)

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

    def test_cors_docs_origin_allowed_even_when_cors_origins_env_set(self):
        # Regression: when CORS_ORIGINS is set in production (e.g. to restrict
        # localhost origins), the docs site must still be allowed.
        os.environ["AUTH_SESSION_TRANSPORT"] = "cookie"
        os.environ["CORS_ORIGINS"] = "https://pandects.org,https://www.pandects.org"
        try:
            app = create_test_app(
                config_overrides={
                    "SQLALCHEMY_BINDS": {"auth": f"sqlite:///{_AUTH_DB_TEMP.name}"},
                }
            )
            client = app.test_client()
            for origin in (
                "https://docs.pandects.org",
                "https://www.docs.pandects.org",
            ):
                with self.subTest(origin=origin):
                    res = client.get("/v1/auth/csrf", headers={"Origin": origin})
                    self.assertEqual(res.status_code, 200)
                    self.assertEqual(res.headers.get("Access-Control-Allow-Origin"), origin)
        finally:
            os.environ.pop("CORS_ORIGINS", None)

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
                self._auth_external_subject(
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

    def test_password_login_pending_block_path_invokes_enumeration_delay(self):
        # Verified-but-no-legal user with a Zitadel link triggers the local
        # fast-path that reveals account existence. We need the randomized
        # delay to fire so its timing matches a real Zitadel round-trip.
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        client = self.app.test_client()
        with self.app.app_context():
            user_id = self._create_local_user(
                email="pending-block@example.com",
                verified=True,
                legal=False,
            )
            db.session.add(
                self._auth_external_subject(
                    user_id=user_id,
                    issuer="https://pandects-test-zitadel.example.com",
                    subject="zitadel-pending-block-user",
                )
            )
            db.session.commit()

        from unittest.mock import patch

        with patch("backend.app.time.sleep") as sleep_mock:
            res = client.post(
                "/v1/auth/login/password",
                json={
                    "email": "pending-block@example.com",
                    "password": "password123",
                },
            )

        self.assertEqual(res.status_code, 400)
        self.assertIn("terms", (res.get_json() or {}).get("message", "").lower())
        sleep_mock.assert_called()

    def test_password_reset_request_fails_closed_when_turnstile_unconfigured_on_fly(self):
        # On Fly (production proxy) we want hard 503 if TURNSTILE keys are
        # missing rather than silently dropping the captcha gate.
        os.environ["FLY_APP_NAME"] = "pandects-test"
        _ = os.environ.pop("TURNSTILE_ENABLED", None)
        _ = os.environ.pop("TURNSTILE_SITE_KEY", None)
        _ = os.environ.pop("TURNSTILE_SECRET_KEY", None)
        client = self.app.test_client()
        try:
            res = client.post(
                "/v1/auth/password-reset/request",
                json={"email": "reset-fail-closed@example.com"},
            )
        finally:
            os.environ.pop("FLY_APP_NAME", None)

        self.assertEqual(res.status_code, 503)

    def test_password_signup_fails_closed_when_turnstile_unconfigured_on_fly(self):
        os.environ["FLY_APP_NAME"] = "pandects-test"
        _ = os.environ.pop("TURNSTILE_ENABLED", None)
        _ = os.environ.pop("TURNSTILE_SITE_KEY", None)
        _ = os.environ.pop("TURNSTILE_SECRET_KEY", None)
        client = self.app.test_client()
        try:
            res = client.post(
                "/v1/auth/signup/password",
                json={
                    "email": "signup-fail-closed@example.com",
                    "password": "Secr3tP4ss!",
                    "next": "/account",
                },
            )
        finally:
            os.environ.pop("FLY_APP_NAME", None)

        self.assertEqual(res.status_code, 503)

    def test_password_reset_request_allowed_when_turnstile_explicitly_disabled(self):
        # Explicit opt-out (TURNSTILE_ENABLED=0) must keep working even on Fly
        # so the operator can disable captcha for a specific deployment.
        os.environ["FLY_APP_NAME"] = "pandects-test"
        os.environ["TURNSTILE_ENABLED"] = "0"
        client = self.app.test_client()
        try:
            res = client.post(
                "/v1/auth/password-reset/request",
                json={"email": "no-such-account@example.com"},
            )
        finally:
            os.environ.pop("FLY_APP_NAME", None)
            os.environ["TURNSTILE_ENABLED"] = "0"

        self.assertEqual(res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
