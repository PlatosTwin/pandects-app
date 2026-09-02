import os
import tempfile
import time
import unittest


def _set_default_env() -> None:
    os.environ.setdefault("SKIP_MAIN_DB_REFLECTION", "1")
    os.environ.setdefault("TEMPORARY_ACCESS_LOCKDOWN", "0")
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
    os.environ["TURNSTILE_ENABLED"] = "0"
    os.environ.pop("TURNSTILE_SITE_KEY", None)
    os.environ.pop("TURNSTILE_SECRET_KEY", None)


_set_default_env()

_AUTH_DB_TEMP = tempfile.NamedTemporaryFile(prefix="pandects_auth_", suffix=".sqlite", delete=False)
_AUTH_DB_TEMP.close()
os.environ.setdefault("AUTH_DATABASE_URI", f"sqlite:///{_AUTH_DB_TEMP.name}")


from backend.app import create_test_app, db  # noqa: E402
import backend.app as backend_app  # noqa: E402


class ApiValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_test_app(
            config_overrides={
                "SQLALCHEMY_BINDS": {"auth": f"sqlite:///{_AUTH_DB_TEMP.name}"},
            }
        )
        with cls.app.app_context():
            db.create_all(bind_key="auth")

    def test_zitadel_start_rejects_unsupported_provider(self):
        client = self.app.test_client()
        res = client.get("/v1/auth/zitadel/start?provider=github")
        self.assertEqual(res.status_code, 400)
        body = res.get_json()
        self.assertIsInstance(body, dict)
        self.assertEqual(body.get("error"), "Bad Request")

    def test_zitadel_start_rejects_unsupported_prompt(self):
        client = self.app.test_client()
        res = client.get("/v1/auth/zitadel/start?provider=email&prompt=consent")
        self.assertEqual(res.status_code, 400)
        body = res.get_json()
        self.assertIsInstance(body, dict)
        self.assertEqual(body.get("error"), "Bad Request")

    def test_legacy_register_route_is_not_registered(self):
        client = self.app.test_client()
        res = client.post("/v1/auth/register", json={})
        self.assertEqual(res.status_code, 404)

    def test_dumps_cache_returns_cached_payload(self):
        payload: list[dict[str, object]] = [{"timestamp": "2025-01-01"}]
        backend_app._dumps_cache["payload"] = payload
        backend_app._dumps_cache["ts"] = time.time()

        client = self.app.test_client()
        res = client.get("/v1/dumps")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body, payload)

    def test_changelog_returns_empty_when_unavailable(self):
        # No R2 client and no cache in the test app => graceful empty payload.
        backend_app._changelog_cache["payload"] = None
        backend_app._changelog_cache["ts"] = 0.0

        client = self.app.test_client()
        res = client.get("/v1/changelog")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(
            res.get_json(),
            {"latest_version": None, "latest_released": None, "releases": []},
        )

    def test_changelog_serves_and_filters_cached_payload(self):
        releases = [
            {
                "version": "2026-08-01",
                "released": "2026-08-01T00:00:00Z",
                "dump_sha256": "b" * 64,
                "dump_key": "dumps/public_2026-08-01.sql.gz",
                "changes": [{"type": "data", "severity": "notable", "summary": "fix"}],
            },
            {
                "version": "2026-07-19",
                "released": "2026-07-20T06:32:18Z",
                "dump_sha256": "a" * 64,
                "dump_key": "dumps/public_2026-07-19.sql.gz",
                "changes": [{"type": "docs", "severity": "minor", "summary": "baseline"}],
            },
        ]
        backend_app._changelog_cache["payload"] = {
            "latest_version": "2026-08-01",
            "latest_released": "2026-08-01T00:00:00Z",
            "releases": releases,
        }
        backend_app._changelog_cache["ts"] = time.time()
        try:
            client = self.app.test_client()

            res = client.get("/v1/changelog")
            self.assertEqual(res.status_code, 200)
            body = res.get_json()
            self.assertEqual(body["latest_version"], "2026-08-01")
            self.assertEqual([r["version"] for r in body["releases"]], ["2026-08-01", "2026-07-19"])

            res = client.get("/v1/changelog?since=2026-07-19")
            self.assertEqual([r["version"] for r in res.get_json()["releases"]], ["2026-08-01"])

            res = client.get(f"/v1/changelog?dump_sha256={'a' * 64}")
            body = res.get_json()
            self.assertEqual([r["version"] for r in body["releases"]], ["2026-07-19"])
            # latest_* still describe the newest published release when filtered.
            self.assertEqual(body["latest_version"], "2026-08-01")
        finally:
            backend_app._changelog_cache["payload"] = None
            backend_app._changelog_cache["ts"] = 0.0

    def test_changelog_sanitizes_malformed_stats_instead_of_500(self):
        # The payload comes from an R2 object; a corrupted row_counts value
        # must be dropped by the route, not crash marshmallow at dump time.
        backend_app._changelog_cache["payload"] = {
            "latest_version": "2026-08-01",
            "latest_released": "2026-08-01T00:00:00Z",
            "releases": [
                {
                    "version": "2026-08-01",
                    "released": "2026-08-01T00:00:00Z",
                    "dump_sha256": "b" * 64,
                    "dump_key": "dumps/public_2026-08-01.sql.gz",
                    "stats": {"row_counts": {"agreements": "notanint", "sections": 5}},
                    "changes": [{"type": "data", "severity": "minor", "summary": "x"}],
                },
                {
                    "version": "2026-07-19",
                    "released": "2026-07-20T06:32:18Z",
                    "dump_sha256": "a" * 64,
                    "dump_key": "dumps/public_2026-07-19.sql.gz",
                    "stats": "notadict",
                    "changes": [{"type": "docs", "severity": "minor", "summary": "y"}],
                },
                {
                    "version": "2026-07-01",
                    "released": "2026-07-01T00:00:00Z",
                    "dump_sha256": "c" * 64,
                    "dump_key": "dumps/public_2026-07-01.sql.gz",
                    # Non-iterables in list-typed fields raise in marshmallow's
                    # List._serialize; the sanitizer must drop them.
                    "changes": [
                        {"type": "data", "severity": "minor", "summary": "z", "tables": 7, "refs": 9},
                        "notadict",
                    ],
                },
                {
                    "version": "2026-06-01",
                    "released": "2026-06-01T00:00:00Z",
                    "dump_sha256": "d" * 64,
                    "dump_key": "dumps/public_2026-06-01.sql.gz",
                    "changes": 42,
                },
            ],
        }
        backend_app._changelog_cache["ts"] = time.time()
        try:
            res = self.app.test_client().get("/v1/changelog")
            self.assertEqual(res.status_code, 200)
            releases = res.get_json()["releases"]
            self.assertEqual(releases[0]["stats"], {"row_counts": {"sections": 5}})
            self.assertIsNone(releases[1]["stats"])
            self.assertEqual(
                releases[2]["changes"],
                [{"type": "data", "severity": "minor", "summary": "z"}],
            )
            self.assertEqual(releases[3]["changes"], [])
        finally:
            backend_app._changelog_cache["payload"] = None
            backend_app._changelog_cache["ts"] = 0.0

    def test_changelog_serves_stale_payload_when_refresh_fails(self):
        # An expired TTL plus a failed refresh (no R2 client in tests) must
        # serve the stale copy, not an affirmative "no releases" lie — and the
        # negative-cache window must do the same.
        payload: dict[str, object] = {
            "latest_version": "2026-08-01",
            "latest_released": "2026-08-01T00:00:00Z",
            "releases": [
                {
                    "version": "2026-08-01",
                    "released": "2026-08-01T00:00:00Z",
                    "dump_sha256": "b" * 64,
                    "dump_key": "dumps/public_2026-08-01.sql.gz",
                    "changes": [{"type": "data", "severity": "minor", "summary": "x"}],
                }
            ],
        }
        backend_app._changelog_cache["payload"] = payload
        backend_app._changelog_cache["ts"] = 0.0  # long-expired TTL
        backend_app._changelog_cache["fail_ts"] = 0.0
        try:
            client = self.app.test_client()
            res = client.get("/v1/changelog")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.get_json()["latest_version"], "2026-08-01")
            # The failed refresh armed the negative cache; the stale copy must
            # still be served inside that window.
            self.assertGreater(backend_app._changelog_cache["fail_ts"], 0.0)
            res = client.get("/v1/changelog")
            self.assertEqual(res.get_json()["latest_version"], "2026-08-01")
        finally:
            backend_app._changelog_cache["payload"] = None
            backend_app._changelog_cache["ts"] = 0.0
            backend_app._changelog_cache["fail_ts"] = 0.0


if __name__ == "__main__":
    unittest.main()
