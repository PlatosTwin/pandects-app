"""In-process TTL memoization for read-mostly payloads.

``ttl_cached`` replaces the hand-rolled ``{"ts": ..., "payload": ...}`` dict +
``Lock`` pattern. One decorated function owns its store, its lock, and its
expiry rule::

    @ttl_cached(21600)
    def taxonomy_payload() -> dict[str, object]: ...

    @ttl_cached(60, key=lambda field, *, limit: (field, limit), max_keys=512)
    def filter_option_values(field: str, *, limit: int) -> list[str]: ...

Rules:

- Thread-safe. The lock guards the store only; the wrapped function runs
  outside it, so a slow rebuild never serializes unrelated readers. Two
  concurrent misses may both compute; the later write wins. Callers must be
  idempotent (every payload builder here is a read).
- Bypassed under ``TESTING``: when the current Flask app has
  ``testing = True`` the function runs on every call and nothing is stored, so
  tests never see a payload cached by an earlier test. A test that exercises
  caching itself sets ``app.config["TTL_CACHE_IN_TESTS"] = True``.
- ``.cache_clear()`` drops every entry; ``clear_all_ttl_caches()`` clears every
  cache this module has created.
- ``retry_none_after`` handles flaky upstreams: a ``None`` result counts as a
  failure, keeps serving the last good value (or ``None``), and suppresses
  retries for that many seconds instead of re-blocking every call on the
  upstream timeout.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from functools import update_wrapper
from threading import Lock
from typing import Any, Generic, ParamSpec, TypeVar, cast
from weakref import WeakSet

from flask import current_app, has_app_context

P = ParamSpec("P")
R = TypeVar("R")

TESTING_OPT_IN_CONFIG_KEY = "TTL_CACHE_IN_TESTS"


def ttl_caching_active() -> bool:
    """False inside a testing app unless the test opted in."""
    if not has_app_context():
        return True
    if not current_app.testing:
        return True
    return bool(current_app.config.get(TESTING_OPT_IN_CONFIG_KEY, False))


@dataclass
class _Entry(Generic[R]):
    value: R
    stored_at: float
    failed_at: float | None = None


def _default_key(*args: object, **kwargs: object) -> Hashable:
    if not kwargs:
        return args
    return (args, tuple(sorted(kwargs.items())))


class TtlCachedFunction(Generic[P, R]):
    """A function memoized per key for ``ttl_seconds``. See module docstring."""

    def __init__(
        self,
        func: Callable[P, R],
        *,
        ttl_seconds: float,
        key: Callable[P, Hashable] | None,
        max_keys: int | None,
        retry_none_after: float | None,
    ) -> None:
        self.__wrapped__: Callable[P, R] = func
        self.ttl_seconds: float = float(ttl_seconds)
        self.max_keys: int | None = max_keys
        self.retry_none_after: float | None = retry_none_after
        self._key: Callable[..., Hashable] = (
            cast(Callable[..., Hashable], key) if key is not None else _default_key
        )
        self._entries: dict[Hashable, _Entry[R]] = {}
        self._lock: Lock = Lock()
        _ = update_wrapper(self, func)
        _REGISTRY.add(self)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        if not ttl_caching_active():
            return self.__wrapped__(*args, **kwargs)
        cache_key = self._key(*args, **kwargs)
        now = time.time()
        with self._lock:
            entry = self._entries.get(cache_key)
            if entry is not None:
                if entry.failed_at is not None:
                    if self.retry_none_after is not None and (
                        now - entry.failed_at < self.retry_none_after
                    ):
                        return entry.value
                elif now - entry.stored_at < self.ttl_seconds:
                    return entry.value

        value = self.__wrapped__(*args, **kwargs)

        with self._lock:
            if value is None and self.retry_none_after is not None:
                previous = self._entries.get(cache_key)
                stale = previous.value if previous is not None else cast(R, None)
                stored_at = previous.stored_at if previous is not None else 0.0
                self._entries[cache_key] = _Entry(stale, stored_at, failed_at=now)
                return stale
            self._prune_locked(now)
            self._entries[cache_key] = _Entry(value, now)
        return value

    def _prune_locked(self, now: float) -> None:
        if self.max_keys is None or len(self._entries) < self.max_keys:
            return
        expired = [
            key
            for key, entry in self._entries.items()
            if entry.failed_at is None and now - entry.stored_at >= self.ttl_seconds
        ]
        for key in expired:
            del self._entries[key]
        overflow = len(self._entries) - self.max_keys + 1
        if overflow > 0:
            oldest = sorted(self._entries.items(), key=lambda item: item[1].stored_at)
            for key, _entry in oldest[:overflow]:
                del self._entries[key]

    def cache_clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def cache_set(self, value: R, *args: P.args, **kwargs: P.kwargs) -> None:
        """Seed the entry for these arguments as fresh (tests, warmups)."""
        cache_key = self._key(*args, **kwargs)
        with self._lock:
            self._entries[cache_key] = _Entry(value, time.time())

    def cache_len(self) -> int:
        with self._lock:
            return len(self._entries)


_REGISTRY: WeakSet[TtlCachedFunction[Any, Any]] = WeakSet()


def ttl_cached(
    seconds: float,
    *,
    key: Callable[P, Hashable] | None = None,
    max_keys: int | None = None,
    retry_none_after: float | None = None,
) -> Callable[[Callable[P, R]], TtlCachedFunction[P, R]]:
    """Memoize a function's result per ``key(*args, **kwargs)`` for ``seconds``.

    ``key`` defaults to the positional and keyword arguments themselves (which
    must then be hashable); a zero-argument function has a single slot.
    ``max_keys`` bounds a keyed cache: expired entries go first, then the
    oldest. See the module docstring for ``retry_none_after`` and the testing
    bypass.
    """

    def decorator(func: Callable[P, R]) -> TtlCachedFunction[P, R]:
        return TtlCachedFunction(
            func,
            ttl_seconds=seconds,
            key=key,
            max_keys=max_keys,
            retry_none_after=retry_none_after,
        )

    return decorator


def clear_all_ttl_caches() -> None:
    for cached in list(_REGISTRY):
        cached.cache_clear()


__all__ = [
    "TESTING_OPT_IN_CONFIG_KEY",
    "TtlCachedFunction",
    "clear_all_ttl_caches",
    "ttl_cached",
    "ttl_caching_active",
]
