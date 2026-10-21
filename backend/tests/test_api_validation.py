import os
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast
from unittest.mock import patch


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
from backend.core.caching import TESTING_OPT_IN_CONFIG_KEY, clear_all_ttl_caches  # noqa: E402
from backend.routes import reference_data  # noqa: E402

# The cached payload builders ignore their deps argument for keying.
_ANY_DEPS = cast(Any, None)


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
        self.assertEqual(body.get("error"), "bad_request")

    def test_zitadel_start_rejects_unsupported_prompt(self):
        client = self.app.test_client()
        res = client.get("/v1/auth/zitadel/start?provider=email&prompt=consent")
        self.assertEqual(res.status_code, 400)
        body = res.get_json()
        self.assertIsInstance(body, dict)
        self.assertEqual(body.get("error"), "bad_request")

    def test_legacy_register_route_is_not_registered(self):
        client = self.app.test_client()
        res = client.post("/v1/auth/register", json={})
        self.assertEqual(res.status_code, 404)
        self.assertEqual((res.get_json() or {}).get("error"), "not_found")

    def test_generic_errors_use_machine_codes_in_envelope(self):
        client = self.app.test_client()
        cases = [
            (client.get("/v1/does-not-exist"), 404, "not_found"),
            (client.post("/v1/taxonomy"), 405, "method_not_allowed"),
            (client.get("/v1/agreements?page_size=abc"), 422, "validation_error"),
        ]
        for res, status, code in cases:
            body = res.get_json()
            self.assertEqual(res.status_code, status)
            self.assertIsInstance(body, dict)
            self.assertEqual(set(body), {"error", "message"})
            self.assertEqual(body["error"], code)
            self.assertTrue(body["message"])
        validation_body = cases[2][0].get_json()
        self.assertEqual(validation_body["message"], "page_size: Not a valid integer.")

    def test_error_code_for_status_covers_unlisted_statuses(self):
        from backend.core.errors import error_code_for_status

        self.assertEqual(error_code_for_status(429), "rate_limited")
        self.assertEqual(error_code_for_status(500), "internal_error")
        self.assertEqual(error_code_for_status(410), "gone")
        self.assertEqual(error_code_for_status(418), "im_a_teapot")

    @contextmanager
    def _ttl_caching(self) -> Iterator[None]:
        """Turn the testing bypass off so seeded `ttl_cached` entries serve."""
        clear_all_ttl_caches()
        self.app.config[TESTING_OPT_IN_CONFIG_KEY] = True
        try:
            yield
        finally:
            self.app.config[TESTING_OPT_IN_CONFIG_KEY] = False
            clear_all_ttl_caches()

    def test_dumps_cache_returns_cached_payload(self):
        payload: list[dict[str, object]] = [{"timestamp": "2025-01-01"}]
        with self._ttl_caching():
            reference_data.dump_list_cached.cache_set(payload, _ANY_DEPS)
            client = self.app.test_client()
            res = client.get("/v1/dumps")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.get_json(), payload)
            self.assertEqual(res.headers.get("Cache-Control"), None)

    def test_changelog_returns_empty_when_unavailable(self):
        # No R2 client and no cache in the test app => graceful empty payload.
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
        with self._ttl_caching():
            reference_data.changelog_payload_cached.cache_set(
                {
                    "latest_version": "2026-08-01",
                    "latest_released": "2026-08-01T00:00:00Z",
                    "releases": releases,
                },
                _ANY_DEPS,
            )
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

    def test_changelog_sanitizes_malformed_stats_instead_of_500(self):
        # The payload comes from an R2 object; a corrupted row_counts value
        # must be dropped by the route, not crash marshmallow at dump time.
        malformed: dict[str, object] = {
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
        with self._ttl_caching():
            reference_data.changelog_payload_cached.cache_set(malformed, _ANY_DEPS)
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
        ttl = reference_data.CHANGELOG_CACHE_TTL_SECONDS
        fail_ttl = reference_data.CHANGELOG_FAIL_TTL_SECONDS
        with self._ttl_caching():
            with patch("backend.core.caching.time.time", return_value=1000.0):
                reference_data.changelog_payload_cached.cache_set(payload, _ANY_DEPS)
            client = self.app.test_client()
            expired_at = 1000.0 + ttl + 1
            with patch("backend.core.caching.time.time", return_value=expired_at):
                res = client.get("/v1/changelog")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.get_json()["latest_version"], "2026-08-01")
            # The failed refresh armed the negative cache; the stale copy must
            # still be served inside that window, without another R2 read.
            with (
                patch("backend.core.caching.time.time", return_value=expired_at + fail_ttl - 1),
                patch.object(
                    reference_data.changelog_payload_cached,
                    "__wrapped__",
                    side_effect=AssertionError("refetched inside the negative-cache window"),
                ),
            ):
                res = client.get("/v1/changelog")
            self.assertEqual(res.get_json()["latest_version"], "2026-08-01")


if __name__ == "__main__":
    unittest.main()
