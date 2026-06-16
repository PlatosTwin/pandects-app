import unittest
from threading import Lock
from unittest.mock import MagicMock

from backend.search_counts import (
    build_search_count_cache_key,
    cached_exact_query_count,
    exact_query_count,
)


def _to_int(value: object, *, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _fake_query(count_value: int) -> MagicMock:
    query = MagicMock()
    ordered = MagicMock()
    query.order_by.return_value = ordered
    ordered.count.return_value = count_value
    return query


class BuildSearchCountCacheKeyTests(unittest.TestCase):
    def test_pagination_and_sort_and_count_mode_do_not_change_key(self):
        # The total depends only on the filter signature, so page/sort/count_mode
        # must map to the same cache entry (that is what lets pages share a count).
        page_one = {
            "page": 1,
            "page_size": 10,
            "sort_by": "year",
            "sort_direction": "asc",
            "count_mode": "auto",
            "include_dump": True,
            "target": ["Acme"],
        }
        deep_page = {
            "page": 9,
            "page_size": 50,
            "sort_by": "target",
            "sort_direction": "desc",
            "count_mode": "exact",
            "include_dump": False,
            "target": ["Acme"],
        }
        self.assertEqual(
            build_search_count_cache_key("sections", page_one),
            build_search_count_cache_key("sections", deep_page),
        )

    def test_changing_a_filter_changes_the_key(self):
        self.assertNotEqual(
            build_search_count_cache_key("sections", {"target": ["Acme"]}),
            build_search_count_cache_key("sections", {"target": ["Other"]}),
        )

    def test_list_filter_order_does_not_change_key(self):
        self.assertEqual(
            build_search_count_cache_key("sections", {"target": ["A", "B"]}),
            build_search_count_cache_key("sections", {"target": ["B", "A"]}),
        )

    def test_scope_separates_otherwise_identical_filters(self):
        args = {"target": ["Acme"]}
        self.assertNotEqual(
            build_search_count_cache_key("agreements_search:direct", args),
            build_search_count_cache_key("agreements_search:clause", args),
        )


class CachedExactQueryCountTests(unittest.TestCase):
    def _call(
        self,
        query: object,
        cache: dict[str, tuple[float, int]],
        cache_key: str | None,
        now,
        ttl: float = 120.0,
    ) -> int:
        return cached_exact_query_count(
            query=query,
            cache_key=cache_key,
            cache=cache,
            lock=Lock(),
            ttl_seconds=ttl,
            max_keys=100,
            to_int=_to_int,
            now=now,
        )

    def test_hit_within_ttl_reuses_value_without_recounting(self):
        query = _fake_query(42)
        cache: dict[str, tuple[float, int]] = {}
        clock = [1000.0]

        self.assertEqual(self._call(query, cache, "k", lambda: clock[0]), 42)
        # A recount would now observe a different value; the cache must not.
        query.order_by.return_value.count.return_value = 999
        clock[0] = 1050.0
        self.assertEqual(self._call(query, cache, "k", lambda: clock[0]), 42)
        self.assertEqual(query.order_by.return_value.count.call_count, 1)

    def test_value_is_recomputed_after_ttl_expiry(self):
        query = _fake_query(42)
        cache: dict[str, tuple[float, int]] = {}
        clock = [1000.0]

        self.assertEqual(self._call(query, cache, "k", lambda: clock[0], ttl=10.0), 42)
        query.order_by.return_value.count.return_value = 99
        clock[0] = 1020.0
        self.assertEqual(self._call(query, cache, "k", lambda: clock[0], ttl=10.0), 99)

    def test_none_cache_key_always_counts_fresh(self):
        query = _fake_query(7)
        cache: dict[str, tuple[float, int]] = {}

        self.assertEqual(self._call(query, cache, None, lambda: 0.0), 7)
        self.assertEqual(cache, {})
        query.order_by.return_value.count.return_value = 8
        self.assertEqual(self._call(query, cache, None, lambda: 0.0), 8)

    def test_exact_query_count_drops_order_by(self):
        query = _fake_query(5)
        self.assertEqual(exact_query_count(query, _to_int), 5)
        query.order_by.assert_called_once_with(None)


if __name__ == "__main__":
    unittest.main()
