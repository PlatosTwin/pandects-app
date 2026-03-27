import os
import tempfile
import unittest
from datetime import datetime, timezone

import jwt
from sqlalchemy import text


def _set_default_env(main_db_uri: str, auth_db_uri: str) -> None:
    os.environ["SKIP_MAIN_DB_REFLECTION"] = "1"
    os.environ["MAIN_DATABASE_URI"] = main_db_uri
    os.environ["MAIN_DB_SCHEMA"] = ""
    os.environ.setdefault("MARIADB_USER", "root")
    os.environ.setdefault("MARIADB_PASSWORD", "password")
    os.environ.setdefault("MARIADB_HOST", "127.0.0.1")
    os.environ.setdefault("MARIADB_DATABASE", "pdx")
    os.environ.setdefault("AUTH_SECRET_KEY", "test-auth-secret")
    os.environ.setdefault("PUBLIC_API_BASE_URL", "http://localhost:5000")
    os.environ.setdefault("PUBLIC_FRONTEND_BASE_URL", "http://localhost:8080")
    os.environ.setdefault("AUTH_SESSION_TRANSPORT", "bearer")
    os.environ["AUTH_DATABASE_URI"] = auth_db_uri
    os.environ["MCP_OIDC_ISSUER"] = "https://issuer.example.com"
    os.environ["MCP_OIDC_AUDIENCE"] = "pandects-mcp"
    os.environ["MCP_OIDC_JWKS_URL"] = "https://issuer.example.com/jwks"
    os.environ["MCP_OIDC_SIGNING_ALGORITHMS"] = "HS256"
    os.environ["MCP_OIDC_AUTHORIZATION_SERVER_URL"] = "https://issuer.example.com"


class McpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        main_db = tempfile.NamedTemporaryFile(prefix="pandects_mcp_main_", suffix=".sqlite", delete=False)
        main_db.close()
        auth_db = tempfile.NamedTemporaryFile(prefix="pandects_mcp_auth_", suffix=".sqlite", delete=False)
        auth_db.close()
        _set_default_env(f"sqlite:///{main_db.name}", f"sqlite:///{auth_db.name}")

        import backend.app as app_module
        import backend.auth.mcp_runtime as mcp_runtime

        cls.app_module = app_module
        cls.mcp_runtime = mcp_runtime
        sqlite_uri = f"sqlite:///{main_db.name}"
        cls.app = cls.app_module.create_test_app(
            config_overrides={
                "MAIN_DB_SCHEMA": "",
                "SQLALCHEMY_DATABASE_URI": sqlite_uri,
                "SQLALCHEMY_BINDS": {"auth": f"sqlite:///{auth_db.name}"},
            }
        )
        with cls.app.app_context():
            engine = cls.app_module.db.engine
            cls.app_module.metadata.create_all(engine)
            cls.app_module.db.create_all(bind_key="auth")
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO agreements (agreement_uuid, filing_date, target, acquirer, verified, url) "
                        "VALUES ('a1', '2020-01-01', 'Target A', 'Acquirer A', 1, 'http://example.com/a1')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO xml (agreement_uuid, xml, version, status, latest) VALUES "
                        "('a1', '<document><article>"
                        "<section uuid=\"00000000-0000-0000-0000-000000000001\"><text>KEEP</text></section>"
                        "<section uuid=\"00000000-0000-0000-0000-000000000002\"><text>HIDE</text></section>"
                        "</article></document>', 1, 'verified', 1)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO sections (agreement_uuid, section_uuid, article_title, section_title, "
                        "xml_content, section_standard_id, xml_version) VALUES "
                        "('a1', '00000000-0000-0000-0000-000000000001', "
                        "'ARTICLE I', 'Section 1', '<section>TEXT</section>', '[\"s1\"]', 1)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO latest_sections_search ("
                        "section_uuid, agreement_uuid, filing_date, prob_filing, filing_company_name, "
                        "filing_company_cik, form_type, exhibit_type, target, acquirer, "
                        "transaction_price_total, transaction_price_stock, transaction_price_cash, "
                        "transaction_price_assets, transaction_consideration, target_type, acquirer_type, "
                        "target_industry, acquirer_industry, announce_date, close_date, deal_status, "
                        "attitude, deal_type, purpose, target_pe, acquirer_pe, verified, url, "
                        "section_standard_ids, article_title, section_title"
                        ") VALUES ("
                        "'00000000-0000-0000-0000-000000000001', 'a1', '2020-01-01', NULL, NULL, NULL, "
                        "NULL, NULL, 'Target A', 'Acquirer A', NULL, NULL, NULL, NULL, 'cash', 'public', "
                        "'public', 'tech', 'tech', NULL, NULL, 'complete', 'friendly', 'merger', "
                        "'strategic', 0, 0, 1, 'http://example.com/a1', '[\"s1\"]', 'ARTICLE I', 'Section 1'"
                        ")"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO latest_sections_search_standard_ids (standard_id, section_uuid, agreement_uuid) "
                        "VALUES ('s1', '00000000-0000-0000-0000-000000000001', 'a1')"
                    )
                )

            user = cls.app_module.AuthUser()
            user.id = "00000000-0000-0000-0000-0000000000f1"
            user.email = "mcp@example.com"
            user.password_hash = None
            user.email_verified_at = datetime.now(timezone.utc).replace(tzinfo=None)
            cls.app_module.db.session.add(user)
            cls.app_module.db.session.flush()

            subject = cls.app_module.AuthExternalSubject()
            subject.user_id = user.id
            subject.issuer = "https://issuer.example.com"
            subject.subject = "sub-123"
            cls.app_module.db.session.add(subject)
            cls.app_module.db.session.commit()

        class DummyJwkClient:
            def get_signing_key_from_jwt(self, _token: str):
                return type("SigningKey", (), {"key": "test-signing-secret"})()

        cls.mcp_runtime._mcp_jwk_client = DummyJwkClient()
        cls.mcp_runtime._mcp_identity_provider = None

    def _bearer(self, scope: str = "sections:search agreements:search agreements:read") -> str:
        return self._bearer_for_subject(subject="sub-123", scope=scope)

    def _bearer_for_subject(
        self,
        *,
        subject: str,
        scope: str = "sections:search agreements:search agreements:read",
    ) -> str:
        token = jwt.encode(
            {
                "iss": "https://issuer.example.com",
                "sub": subject,
                "aud": "pandects-mcp",
                "scope": scope,
            },
            "test-signing-secret",
            algorithm="HS256",
        )
        return f"Bearer {token}"

    def test_protected_resource_metadata(self):
        client = self.app.test_client()
        res = client.get("/.well-known/oauth-protected-resource")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload["resource"], "http://localhost:5000/mcp")
        self.assertIn("https://issuer.example.com", payload["authorization_servers"])

    def test_mcp_requires_bearer_token(self):
        client = self.app.test_client()
        res = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        self.assertEqual(res.status_code, 401)
        self.assertIn("WWW-Authenticate", res.headers)

    def test_initialize_and_tools_list(self):
        client = self.app.test_client()
        res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer()},
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        )
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload["result"]["serverInfo"]["name"], "pandects-mcp")

        res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer()},
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        )
        self.assertEqual(res.status_code, 200)
        tools = res.get_json()["result"]["tools"]
        self.assertEqual([tool["name"] for tool in tools], ["search_sections", "list_agreements", "get_agreement"])

    def test_search_sections_tool(self):
        client = self.app.test_client()
        res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer(scope="sections:search")},
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "search_sections", "arguments": {"page_size": 10}},
            },
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        structured = body["result"]["structuredContent"]
        self.assertEqual(len(structured["results"]), 1)
        self.assertEqual(structured["access"]["tier"], "mcp")

    def test_get_agreement_redacts_without_fulltext_scope(self):
        client = self.app.test_client()
        res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer(scope="agreements:read")},
            json={
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "get_agreement",
                    "arguments": {
                        "agreement_uuid": "a1",
                        "focus_section_uuid": "00000000-0000-0000-0000-000000000001",
                        "neighbor_sections": 0,
                    },
                },
            },
        )
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()["result"]["structuredContent"]
        self.assertTrue(payload["is_redacted"])
        self.assertIn("[REDACTED]", payload["xml"])

    def test_get_agreement_fulltext_requires_scope(self):
        client = self.app.test_client()
        res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer(scope="agreements:read agreements:read_fulltext")},
            json={
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "get_agreement",
                    "arguments": {"agreement_uuid": "a1"},
                },
            },
        )
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()["result"]["structuredContent"]
        self.assertNotIn("is_redacted", payload)
        self.assertIn("<text>KEEP</text>", payload["xml"])

    def test_missing_scope_is_403(self):
        client = self.app.test_client()
        res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer(scope="sections:search")},
            json={
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {
                    "name": "get_agreement",
                    "arguments": {"agreement_uuid": "a1"},
                },
            },
        )
        self.assertEqual(res.status_code, 403)

    def test_unlinked_subject_is_401(self):
        client = self.app.test_client()
        res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer_for_subject(subject="sub-missing")},
            json={"jsonrpc": "2.0", "id": 7, "method": "initialize"},
        )
        self.assertEqual(res.status_code, 401)
        self.assertIn("WWW-Authenticate", res.headers)

    def test_normalized_external_identity_supports_scope_and_scp_claims(self):
        normalized_from_scope = self.mcp_runtime._normalize_external_identity(
            {
                "iss": "https://issuer.example.com/",
                "sub": "sub-123",
                "aud": "pandects-mcp",
                "scope": "agreements:read agreements:read_fulltext",
            }
        )
        self.assertEqual(normalized_from_scope.issuer, "https://issuer.example.com")
        self.assertEqual(
            normalized_from_scope.scopes,
            frozenset({"agreements:read", "agreements:read_fulltext"}),
        )
        self.assertEqual(normalized_from_scope.audiences, frozenset({"pandects-mcp"}))

        normalized_from_scp = self.mcp_runtime._normalize_external_identity(
            {
                "iss": "https://issuer.example.com",
                "sub": "sub-123",
                "aud": ["pandects-mcp", "other-audience"],
                "scp": ["sections:search", "agreements:search"],
            }
        )
        self.assertEqual(
            normalized_from_scp.scopes,
            frozenset({"sections:search", "agreements:search"}),
        )
        self.assertEqual(
            normalized_from_scp.audiences,
            frozenset({"pandects-mcp", "other-audience"}),
        )

    def test_identity_provider_name_defaults_to_oidc(self):
        previous = os.environ.pop("MCP_IDENTITY_PROVIDER", None)
        try:
            self.assertEqual(self.mcp_runtime.mcp_identity_provider_name(), "oidc")
        finally:
            if previous is not None:
                os.environ["MCP_IDENTITY_PROVIDER"] = previous

    def test_identity_provider_can_be_selected_by_env(self):
        runtime = self.mcp_runtime
        previous = os.environ.get("MCP_IDENTITY_PROVIDER")

        class StubProvider(runtime.McpIdentityProvider):
            def authenticate_access_token(self, token: str) -> runtime.ExternalIdentity:
                self.seen_token = token
                return runtime.ExternalIdentity(
                    issuer="https://stub-issuer.example.com",
                    subject="stub-subject",
                    scopes=frozenset({"agreements:read"}),
                    audiences=frozenset({"pandects-mcp"}),
                    claims={"sub": "stub-subject"},
                )

        runtime.register_mcp_identity_provider("stub", StubProvider)
        runtime._mcp_identity_provider = None
        os.environ["MCP_IDENTITY_PROVIDER"] = "stub"
        try:
            provider = runtime._identity_provider()
            self.assertIsInstance(provider, StubProvider)
        finally:
            runtime._mcp_identity_provider = None
            if previous is None:
                os.environ.pop("MCP_IDENTITY_PROVIDER", None)
            else:
                os.environ["MCP_IDENTITY_PROVIDER"] = previous

    def test_unknown_identity_provider_raises_runtime_error(self):
        runtime = self.mcp_runtime
        previous = os.environ.get("MCP_IDENTITY_PROVIDER")
        runtime._mcp_identity_provider = None
        os.environ["MCP_IDENTITY_PROVIDER"] = "missing-provider"
        try:
            with self.assertRaises(RuntimeError):
                runtime._identity_provider()
        finally:
            runtime._mcp_identity_provider = None
            if previous is None:
                os.environ.pop("MCP_IDENTITY_PROVIDER", None)
            else:
                os.environ["MCP_IDENTITY_PROVIDER"] = previous


if __name__ == "__main__":
    unittest.main()
