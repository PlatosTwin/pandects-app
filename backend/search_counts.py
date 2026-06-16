"""Shared helpers for paginated-search result counts.

The number of rows a search matches depends only on its filter signature, never
on which page or sort order is being viewed. These helpers memoize the exact
`COUNT(*)` per filter signature so paging through one search runs a single count
instead of one per page, while every page reports the identical exact total.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Callable, Protocol, cast


class ToIntProtocol(Protocol):
    def __call__(self, value: object, *, default: int = 0) -> int: ...


class _StatementQuery(Protocol):
    def order_by(self, *clauses: object) -> "_StatementQuery": ...

    def count(self) -> object: ...


# Argument keys that never change which rows match a search. They must be kept
# out of the cache key so that paging/sorting reuses one cached count. This is
# an exclude-list on purpose: every other parsed arg participates in the key, so
# a filter added later is captured automatically. Dropping a real filter from
# the key is the only mistake that can return a wrong count; excluding a true
# non-filter can at worst cost a redundant cache miss.
_NON_FILTER_ARG_KEYS = frozenset(
    {"page", "page_size", "sort_by", "sort_direction", "count_mode", "include_dump"}
)


def exact_query_count(query: object, to_int: ToIntProtocol) -> int:
    """Run an exact `COUNT(*)` for a query, dropping any ORDER BY first."""
    return to_int(cast(_StatementQuery, query).order_by(None).count())


def build_search_count_cache_key(scope: str, parsed_args: Mapping[str, object]) -> str:
    """Derive a stable cache key from a search's filter signature.

    `scope` distinguishes endpoints (and query shapes within an endpoint). List
    values are sorted because `IN (...)` filters are order-independent, which
    lifts the cache hit rate without affecting which rows match.
    """
    filter_items: list[tuple[str, object]] = []
    for key in sorted(parsed_args):
        if key in _NON_FILTER_ARG_KEYS:
            continue
        value = parsed_args[key]
        if isinstance(value, (list, tuple)):
            value = sorted(str(item) for item in value)
        filter_items.append((key, value))
    raw = json.dumps([scope, filter_items], sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def cached_exact_query_count(
    *,
    query: object,
    cache_key: str | None,
    cache: dict[str, tuple[float, int]],
    lock: Any,
    ttl_seconds: float,
    max_keys: int,
    to_int: ToIntProtocol,
    now: Callable[[], float],
) -> int:
    """Exact `COUNT(*)` for `query`, memoized per `cache_key` for `ttl_seconds`.

    `cache_key=None` bypasses the cache and always runs a fresh count (used when
    a caller explicitly requests an authoritative, uncached total).
    """
    if cache_key is None:
        return exact_query_count(query, to_int)

    current = now()
    with lock:
        entry = cache.get(cache_key)
        if entry is not None and (current - entry[0]) < ttl_seconds:
            return entry[1]

    # Run the COUNT outside the lock: it can hit the database and must not
    # serialize unrelated count requests behind one slow query. A concurrent
    # miss only costs a redundant COUNT, never a wrong value.
    value = exact_query_count(query, to_int)

    with lock:
        _prune_count_cache(cache, max_keys, ttl_seconds, current)
        cache[cache_key] = (current, value)
    return value


def _prune_count_cache(
    cache: dict[str, tuple[float, int]],
    max_keys: int,
    ttl_seconds: float,
    now_ts: float,
) -> None:
    expired = [key for key, (ts, _value) in cache.items() if (now_ts - ts) >= ttl_seconds]
    for key in expired:
        del cache[key]
    overflow = len(cache) - max_keys
    if overflow > 0:
        oldest = sorted(cache.items(), key=lambda item: item[1][0])[:overflow]
        for key, _entry in oldest:
            del cache[key]


__all__ = [
    "ToIntProtocol",
    "exact_query_count",
    "build_search_count_cache_key",
    "cached_exact_query_count",
]
