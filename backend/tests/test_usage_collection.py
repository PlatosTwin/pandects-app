"""Tests for the usage-data collection paths added for the analytics
dashboard: page views, MCP/web hourly rollups, logout signon events, signup
attribution, coarse geo, and raw-event retention."""

import os
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError


def _set_default_env() -> None:
    os.environ["SKIP_MAIN_DB_REFLECTION"] = "1"
    os.environ["TEMPORARY_ACCESS_LOCKDOWN"] = "0"
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
    os.environ["TURNSTILE_ENABLED"] = "0"
    # Exercise the synchronous (unbuffered) rollup path by default so tests
    # observe writes without racing a flush thread.
    os.environ["USAGE_LOG_BUFFER_ENABLED"] = "0"


_set_default_env()

_AUTH_DB_TEMP = tempfile.NamedTemporaryFile(
    prefix="pandects_usage_", suffix=".sqlite", delete=False
)
_AUTH_DB_TEMP.close()
os.environ["AUTH_DATABASE_URI"] = f"sqlite:///{_AUTH_DB_TEMP.name}"


from backend.app import create_test_app  # noqa: E402
from backend.extensions import db  # noqa: E402
from backend.models import (  # noqa: E402
    ApiKey,
    ApiRequestEvent,
    AuthSignonEvent,
    AuthSignupAttribution,
    AuthUser,
    McpUsageHourly,
    PageView,
    WebUsageHourly,
)
from backend.auth.legal_runtime import record_signon_event  # noqa: E402
from backend.auth.mcp_oauth_runtime import access_token_claims  # noqa: E402
from backend.auth.mcp_runtime import McpPrincipal, _claim_client_id  # noqa: E402
from backend.auth.session_runtime import AccessContext, issue_session_token  # noqa: E402
import backend.services.usage as usage_module  # noqa: E402
from backend.services.usage import (  # noqa: E402
    LATENCY_BUCKET_BOUNDS_MS,
    HourlyRollupAggregate,
    HourlyRollupBuffer,
    UsageBuffer,
    _commit_rollup_pending,
    _init_latency_buckets,
    record_mcp_tool_usage,
    record_web_session_usage,
    request_country,
)
import backend.app as backend_app  # noqa: E402


class UsageCollectionTests(unittest.TestCase):
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
                conn.execute(text("DELETE FROM page_views"))
                conn.execute(text("DELETE FROM mcp_usage_hourly"))
                conn.execute(text("DELETE FROM web_usage_hourly"))
                conn.execute(text("DELETE FROM api_request_events"))
                conn.execute(text("DELETE FROM auth_signup_attributions"))
                conn.execute(text("DELETE FROM auth_signon_events"))
                conn.execute(text("DELETE FROM auth_sessions"))
                conn.execute(text("DELETE FROM api_keys"))
                conn.execute(text("DELETE FROM auth_users"))
        backend_app._rate_limit_state.clear()
        backend_app._endpoint_rate_limit_state.clear()

    def _create_user(self, *, verified: bool = True) -> str:
        with self.app.app_context():
            user = AuthUser()
            user.email = f"user-{uuid.uuid4().hex[:10]}@example.com"
            if verified:
                user.email_verified_at = datetime.utcnow()
            db.session.add(user)
            db.session.commit()
            return user.id

    # ── page views ─────────────────────────────────────────────────────

    def test_page_view_recorded_anonymously(self) -> None:
        client = self.app.test_client()
        res = client.post(
            "/v1/page-views",
            json={"path": "/search?q=mae", "referrer": "https://news.example.com/x"},
            headers={"Fly-Region": "iad"},
        )
        self.assertEqual(res.status_code, 204)
        with self.app.app_context():
            rows = PageView.query.all()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].path, "/search?q=mae")
            self.assertIsNone(rows[0].user_id)
            self.assertEqual(rows[0].referrer, "https://news.example.com/x")
            self.assertEqual(rows[0].country, "iad")

    def test_page_view_rejects_bad_paths_silently(self) -> None:
        client = self.app.test_client()
        for payload in (
            {"path": "https://evil.example.com"},
            {"path": "//protocol-relative"},
            {"path": 42},
            {},
            None,
        ):
            res = client.post("/v1/page-views", json=payload)
            self.assertEqual(res.status_code, 204)
        with self.app.app_context():
            self.assertEqual(PageView.query.count(), 0)

    def test_page_view_is_csrf_exempt_with_session_cookie(self) -> None:
        client = self.app.test_client()
        client.set_cookie("pdcts_session", "some-session-token")
        res = client.post("/v1/page-views", json={"path": "/dashboard"})
        self.assertEqual(res.status_code, 204)

    def test_page_view_strips_controls_and_requires_http_referrer(self) -> None:
        client = self.app.test_client()
        res = client.post(
            "/v1/page-views",
            json={"path": "/a\x00b\x1fc", "referrer": "javascript:alert(1)"},
        )
        self.assertEqual(res.status_code, 204)
        with self.app.app_context():
            rows = PageView.query.all()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].path, "/abc")
            self.assertIsNone(rows[0].referrer)

    def test_page_view_allowed_while_temporary_lockdown_enabled(self) -> None:
        # The anonymous-visitor pipeline is the point of this endpoint; it
        # must stay open when the temporary access gate 403s the data API.
        os.environ["TEMPORARY_ACCESS_LOCKDOWN"] = "1"
        try:
            client = self.app.test_client()
            res = client.post("/v1/page-views", json={"path": "/pricing"})
            self.assertEqual(res.status_code, 204)
            # Control: other anonymous data-API requests are gated.
            gated = client.get("/v1/not-a-real-route")
            self.assertEqual(gated.status_code, 403)
        finally:
            os.environ["TEMPORARY_ACCESS_LOCKDOWN"] = "0"
        with self.app.app_context():
            self.assertEqual(PageView.query.count(), 1)

    # ── MCP tool-call rollups ───────────────────────────────────────────

    def test_mcp_tool_usage_rolls_up_by_user_client_tool_status(self) -> None:
        user_id = self._create_user()
        with self.app.test_request_context("/mcp", method="POST"):
            record_mcp_tool_usage(
                user_id=user_id,
                client_id="client-abc",
                tool_name="search_sections",
                outcome="ok",
                error_category=None,
                latency_ms=120,
                request_bytes=300,
            )
            record_mcp_tool_usage(
                user_id=user_id,
                client_id="client-abc",
                tool_name="search_sections",
                outcome="ok",
                error_category=None,
                latency_ms=80,
                request_bytes=200,
            )
            record_mcp_tool_usage(
                user_id=user_id,
                client_id=None,
                tool_name="search_sections",
                outcome="error",
                error_category="validation",
                latency_ms=5,
            )
        with self.app.app_context():
            rows = McpUsageHourly.query.order_by(McpUsageHourly.status).all()
            self.assertEqual(len(rows), 2)
            ok_row = next(r for r in rows if r.status == "ok")
            self.assertEqual(ok_row.count, 2)
            self.assertEqual(ok_row.client_id, "client-abc")
            self.assertEqual(ok_row.total_ms, 200)
            self.assertEqual(ok_row.max_ms, 120)
            self.assertEqual(ok_row.request_bytes, 500)
            self.assertEqual(sum(ok_row.latency_buckets), 2)
            err_row = next(r for r in rows if r.status == "validation")
            self.assertEqual(err_row.count, 1)
            self.assertEqual(err_row.client_id, "")

    def test_mcp_tools_call_dispatch_persists_usage_row(self) -> None:
        """End-to-end wiring: an authenticated POST /mcp tools/call must leave
        a mcp_usage_hourly row via the g.mcp_principal stash, even when the
        tool name is unknown."""
        user_id = self._create_user()
        principal = McpPrincipal(
            access_context=AccessContext(tier="mcp", user_id=user_id),
            scopes=frozenset(),
            issuer="https://test-issuer",
            subject=user_id,
            user_id=user_id,
            client_id="client-e2e",
        )
        client = self.app.test_client()
        with patch(
            "backend.mcp.routes.authenticate_mcp_request", return_value=principal
        ):
            res = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "definitely_not_a_tool", "arguments": {}},
                },
            )
        self.assertEqual(res.status_code, 200)
        with self.app.app_context():
            rows = McpUsageHourly.query.all()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].user_id, user_id)
            self.assertEqual(rows[0].client_id, "client-e2e")
            self.assertEqual(rows[0].tool_name, "definitely_not_a_tool")
            self.assertEqual(rows[0].status, "unknown_tool")
            self.assertEqual(rows[0].count, 1)

    # ── web session rollups ─────────────────────────────────────────────

    def test_after_request_hook_records_web_usage_end_to_end(self) -> None:
        """The web rollup must be wired into the real after_request hook, not
        just callable: a session-authenticated /v1 request (here a 404) must
        leave a web_usage_hourly row."""
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        user_id = self._create_user()
        with self.app.test_request_context("/"):
            token = issue_session_token(user_id)
        client = self.app.test_client()
        res = client.get(
            "/v1/route-that-does-not-exist",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(res.status_code, 404)
        with self.app.app_context():
            rows = WebUsageHourly.query.all()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].user_id, user_id)
            self.assertEqual(rows[0].route, "/v1/route-that-does-not-exist")
            self.assertEqual(rows[0].status_class, 4)

    def test_web_session_usage_recorded_for_user_tier(self) -> None:
        user_id = self._create_user()
        with self.app.test_request_context("/v1/agreements", method="GET"):
            from flask import Response as FlaskResponse

            ctx = AccessContext(tier="user", user_id=user_id)
            record_web_session_usage(
                ctx=ctx,
                response=FlaskResponse("[]", status=200),
                auth_is_mocked=lambda: False,
            )
        with self.app.app_context():
            rows = WebUsageHourly.query.all()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].user_id, user_id)
            self.assertEqual(rows[0].route, "/v1/agreements")
            self.assertEqual(rows[0].method, "GET")
            self.assertEqual(rows[0].status_class, 2)
            self.assertEqual(rows[0].count, 1)

    def test_web_session_usage_skips_other_tiers_and_paths(self) -> None:
        user_id = self._create_user()
        from flask import Response as FlaskResponse

        cases = (
            ("/v1/agreements", AccessContext(tier="api_key", api_key_id="k1")),
            ("/v1/agreements", AccessContext(tier="anonymous")),
            ("/v1/auth/me", AccessContext(tier="user", user_id=user_id)),
            ("/v1/page-views", AccessContext(tier="user", user_id=user_id)),
            ("/healthz", AccessContext(tier="user", user_id=user_id)),
        )
        for path, ctx in cases:
            with self.app.test_request_context(path, method="GET"):
                record_web_session_usage(
                    ctx=ctx,
                    response=FlaskResponse("", status=200),
                    auth_is_mocked=lambda: False,
                )
        with self.app.app_context():
            self.assertEqual(WebUsageHourly.query.count(), 0)

    # ── rollup buffer ───────────────────────────────────────────────────

    def test_hourly_rollup_buffer_flushes_and_upserts(self) -> None:
        user_id = self._create_user()
        buffer = HourlyRollupBuffer(
            app=self.app,
            db=db,
            model=McpUsageHourly,
            key_columns=("user_id", "client_id", "hour", "tool_name", "status"),
            latency_bucket_bounds=LATENCY_BUCKET_BOUNDS_MS,
            flush_interval_seconds=3600,
            max_pending_events=10_000,
            thread_name="test-rollup-buffer",
        )
        try:
            hour = datetime(2026, 7, 19, 10, 0, 0)
            key = (user_id, "c1", hour, "get_agreement", "ok")
            buffer.record(key=key, latency_ms=40, request_bytes=10)
            buffer.record(key=key, latency_ms=60, request_bytes=20)
            buffer.flush()
            buffer.record(key=key, latency_ms=100, request_bytes=5)
            buffer.flush()
        finally:
            buffer.stop()
        with self.app.app_context():
            rows = McpUsageHourly.query.all()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].count, 3)
            self.assertEqual(rows[0].total_ms, 200)
            self.assertEqual(rows[0].max_ms, 100)
            self.assertEqual(rows[0].request_bytes, 35)
            self.assertEqual(sum(rows[0].latency_buckets), 3)

    def test_rollup_commit_retries_after_concurrent_insert_race(self) -> None:
        """If another worker wins the INSERT race, the losing flush must
        retry and merge instead of dropping its snapshot."""
        user_id = self._create_user()
        hour = datetime(2026, 7, 19, 11, 0, 0)
        key = (user_id, "c1", hour, "get_agreement", "ok")
        buckets = _init_latency_buckets(LATENCY_BUCKET_BOUNDS_MS)
        buckets[0] = 1
        pending = {
            key: HourlyRollupAggregate(
                count=1,
                total_ms=10,
                max_ms=10,
                buckets=buckets,
                request_bytes=0,
                response_bytes=0,
            )
        }
        real_flush = usage_module._flush_rollup_rows
        calls = {"n": 0}

        def racing_flush(**kwargs) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise IntegrityError("INSERT", None, Exception("duplicate key"))
            real_flush(**kwargs)

        with self.app.app_context():
            with patch.object(usage_module, "_flush_rollup_rows", racing_flush):
                _commit_rollup_pending(
                    db=db,
                    model=McpUsageHourly,
                    key_columns=(
                        "user_id",
                        "client_id",
                        "hour",
                        "tool_name",
                        "status",
                    ),
                    latency_bucket_bounds=LATENCY_BUCKET_BOUNDS_MS,
                    pending=pending,
                )
            rows = McpUsageHourly.query.all()
            self.assertEqual(calls["n"], 2)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].count, 1)

    # ── logout signon events ────────────────────────────────────────────

    def test_logout_records_signon_event(self) -> None:
        os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
        user_id = self._create_user()
        from backend.auth.session_runtime import issue_session_token

        with self.app.test_request_context("/"):
            token = issue_session_token(user_id)
        client = self.app.test_client()
        res = client.post(
            "/v1/auth/logout", headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(res.status_code, 200)
        with self.app.app_context():
            events = AuthSignonEvent.query.filter_by(action="logout").all()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].user_id, user_id)
            self.assertEqual(events[0].provider, "session")

    # ── signup attribution ──────────────────────────────────────────────

    def test_register_signon_event_captures_attribution_cookie(self) -> None:
        user_id = self._create_user()
        cookie_value = (
            "%7B%22s%22%3A%22newsletter%22%2C%22m%22%3A%22email%22%2C"
            "%22r%22%3A%22https%3A%2F%2Fnews.example.com%2Fpost%22%2C"
            "%22l%22%3A%22%2Fpricing%22%7D"
        )
        with self.app.test_request_context(
            "/v1/auth/register",
            method="POST",
            headers={"Cookie": f"pdcts_attr={cookie_value}"},
        ):
            record_signon_event(user_id=user_id, provider="zitadel", action="register")
            db.session.commit()
        with self.app.app_context():
            rows = AuthSignupAttribution.query.all()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].user_id, user_id)
            self.assertEqual(rows[0].utm_source, "newsletter")
            self.assertEqual(rows[0].utm_medium, "email")
            self.assertEqual(rows[0].referrer, "https://news.example.com/post")
            self.assertEqual(rows[0].landing_path, "/pricing")

    def test_register_without_attribution_cookie_records_nothing(self) -> None:
        user_id = self._create_user()
        with self.app.test_request_context("/v1/auth/register", method="POST"):
            record_signon_event(user_id=user_id, provider="zitadel", action="register")
            db.session.commit()
        with self.app.app_context():
            self.assertEqual(AuthSignupAttribution.query.count(), 0)

    def test_login_action_does_not_capture_attribution(self) -> None:
        user_id = self._create_user()
        with self.app.test_request_context(
            "/v1/auth/login",
            method="POST",
            headers={"Cookie": "pdcts_attr=%7B%22s%22%3A%22x%22%7D"},
        ):
            record_signon_event(user_id=user_id, provider="zitadel", action="login")
            db.session.commit()
        with self.app.app_context():
            self.assertEqual(AuthSignupAttribution.query.count(), 0)

    # ── coarse geo ──────────────────────────────────────────────────────

    def test_request_country_prefers_cdn_header_over_fly_region(self) -> None:
        with self.app.test_request_context(
            "/", headers={"CF-IPCountry": "de", "Fly-Region": "fra"}
        ):
            self.assertEqual(request_country(), "DE")
        with self.app.test_request_context("/", headers={"Fly-Region": "FRA"}):
            self.assertEqual(request_country(), "fra")
        with self.app.test_request_context("/", headers={"CF-IPCountry": "XX"}):
            self.assertIsNone(request_country())
        with self.app.test_request_context("/"):
            self.assertIsNone(request_country())

    # ── raw event retention ─────────────────────────────────────────────

    def test_usage_buffer_prunes_old_request_events(self) -> None:
        user_id = self._create_user()
        with self.app.app_context():
            key = ApiKey()
            key.user_id = user_id
            key.prefix = "pdcts_prunetest"
            key.key_hash = "hash"
            db.session.add(key)
            db.session.commit()
            key_id = key.id
            old_event = ApiRequestEvent()
            old_event.api_key_id = key_id
            old_event.occurred_at = datetime.utcnow() - timedelta(days=400)
            old_event.route = "/v1/agreements"
            old_event.method = "GET"
            old_event.status_code = 200
            old_event.status_class = 2
            old_event.latency_ms = 10
            fresh_event = ApiRequestEvent()
            fresh_event.api_key_id = key_id
            fresh_event.occurred_at = datetime.utcnow()
            fresh_event.route = "/v1/agreements"
            fresh_event.method = "GET"
            fresh_event.status_code = 200
            fresh_event.status_class = 2
            fresh_event.latency_ms = 10
            db.session.add_all([old_event, fresh_event])
            db.session.commit()

        from backend.models import ApiUsageDaily, ApiUsageDailyIp, ApiUsageHourly

        buffer = UsageBuffer(
            app=self.app,
            db=db,
            ApiUsageDaily=ApiUsageDaily,
            ApiUsageHourly=ApiUsageHourly,
            ApiUsageDailyIp=ApiUsageDailyIp,
            ApiRequestEvent=ApiRequestEvent,
            latency_bucket_bounds=LATENCY_BUCKET_BOUNDS_MS,
            flush_interval_seconds=3600,
            max_pending_events=10_000,
            request_event_retention_days=180,
        )
        try:
            buffer._last_prune = 0.0  # prune clock starts at boot; force due
            with self.app.app_context():
                buffer._prune_request_events_if_due()
            with self.app.app_context():
                remaining = ApiRequestEvent.query.all()
                self.assertEqual(len(remaining), 1)
                self.assertGreater(
                    remaining[0].occurred_at,
                    datetime.utcnow() - timedelta(days=2),
                )
        finally:
            buffer.stop()

    # ── access-token client claim ───────────────────────────────────────

    def test_access_token_claims_include_client_id(self) -> None:
        claims = access_token_claims(
            subject="user-1",
            audience="https://api.pandects.org/mcp",
            scope="agreements:read",
            token_id="jti-1",
            client_id="client-xyz",
        )
        self.assertEqual(claims.get("client_id"), "client-xyz")
        self.assertEqual(_claim_client_id(claims), "client-xyz")
        legacy = access_token_claims(
            subject="user-1",
            audience="https://api.pandects.org/mcp",
            scope="agreements:read",
            token_id="jti-2",
        )
        self.assertNotIn("client_id", legacy)
        self.assertIsNone(_claim_client_id(legacy))
        self.assertEqual(_claim_client_id({"azp": "zitadel-app"}), "zitadel-app")


if __name__ == "__main__":
    unittest.main()
