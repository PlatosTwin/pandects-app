import unittest
from unittest.mock import patch

from flask import Flask

from backend.core.caching import clear_all_ttl_caches, ttl_cached


class TtlCachedTests(unittest.TestCase):
    def test_caches_until_ttl_expires(self) -> None:
        calls: list[int] = []

        @ttl_cached(10)
        def payload() -> dict[str, int]:
            calls.append(1)
            return {"n": len(calls)}

        with patch("backend.core.caching.time.time", return_value=100.0):
            self.assertEqual(payload(), {"n": 1})
            self.assertEqual(payload(), {"n": 1})
        with patch("backend.core.caching.time.time", return_value=109.9):
            self.assertEqual(payload(), {"n": 1})
        with patch("backend.core.caching.time.time", return_value=110.0):
            self.assertEqual(payload(), {"n": 2})
        self.assertEqual(len(calls), 2)

    def test_keyed_variants_are_independent_and_bounded(self) -> None:
        calls: list[str] = []

        @ttl_cached(60, key=lambda field, *, limit: (field, limit), max_keys=2)
        def values(field: str, *, limit: int) -> list[str]:
            calls.append(field)
            return [field] * limit

        with patch("backend.core.caching.time.time", return_value=1.0):
            self.assertEqual(values("a", limit=1), ["a"])
            self.assertEqual(values("a", limit=1), ["a"])
        with patch("backend.core.caching.time.time", return_value=2.0):
            self.assertEqual(values("b", limit=2), ["b", "b"])
        with patch("backend.core.caching.time.time", return_value=3.0):
            _ = values("c", limit=1)
        self.assertEqual(calls, ["a", "b", "c"])
        # Oldest entry ("a") was evicted to stay within max_keys.
        self.assertEqual(values.cache_len(), 2)
        with patch("backend.core.caching.time.time", return_value=4.0):
            _ = values("a", limit=1)
        self.assertEqual(calls, ["a", "b", "c", "a"])

    def test_cache_clear_and_cache_set(self) -> None:
        calls: list[int] = []

        @ttl_cached(60)
        def payload() -> int:
            calls.append(1)
            return len(calls)

        self.assertEqual(payload(), 1)
        payload.cache_clear()
        self.assertEqual(payload(), 2)
        payload.cache_set(99)
        self.assertEqual(payload(), 99)
        clear_all_ttl_caches()
        self.assertEqual(payload(), 3)

    def test_bypassed_under_testing_unless_opted_in(self) -> None:
        calls: list[int] = []

        @ttl_cached(60)
        def payload() -> int:
            calls.append(1)
            return len(calls)

        app = Flask(__name__)
        app.testing = True
        with app.app_context():
            self.assertEqual(payload(), 1)
            self.assertEqual(payload(), 2)
            self.assertEqual(payload.cache_len(), 0)
            app.config["TTL_CACHE_IN_TESTS"] = True
            self.assertEqual(payload(), 3)
            self.assertEqual(payload(), 3)

    def test_retry_none_after_serves_stale_value_and_backs_off(self) -> None:
        results: list[str | None] = ["v1", None, None, "v2"]
        calls: list[int] = []

        @ttl_cached(10, retry_none_after=5)
        def fetch() -> str | None:
            calls.append(1)
            return results[len(calls) - 1]

        with patch("backend.core.caching.time.time", return_value=0.0):
            self.assertEqual(fetch(), "v1")
        # Expired; the upstream fails, so the stale value is served.
        with patch("backend.core.caching.time.time", return_value=20.0):
            self.assertEqual(fetch(), "v1")
        # Within the failure back-off: no new upstream call.
        with patch("backend.core.caching.time.time", return_value=24.0):
            self.assertEqual(fetch(), "v1")
        self.assertEqual(len(calls), 2)
        with patch("backend.core.caching.time.time", return_value=25.0):
            self.assertEqual(fetch(), "v1")
        with patch("backend.core.caching.time.time", return_value=31.0):
            self.assertEqual(fetch(), "v2")
        self.assertEqual(len(calls), 4)

    def test_retry_none_after_without_prior_value_returns_none(self) -> None:
        calls: list[int] = []

        @ttl_cached(10, retry_none_after=5)
        def fetch() -> str | None:
            calls.append(1)
            return None

        with patch("backend.core.caching.time.time", return_value=0.0):
            self.assertIsNone(fetch())
            self.assertIsNone(fetch())
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
