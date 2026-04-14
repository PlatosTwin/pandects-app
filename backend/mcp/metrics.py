from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from threading import Lock


_DEFAULT_LATENCY_BUCKETS_MS = (50, 100, 250, 500, 1000, 2500, 5000)


def _init_buckets(bounds: tuple[int, ...]) -> list[int]:
    return [0] * (len(bounds) + 1)


def _bucket_index(bounds: tuple[int, ...], latency_ms: int) -> int:
    for idx, bound in enumerate(bounds):
        if latency_ms <= bound:
            return idx
    return len(bounds)


@dataclass
class ToolMetricAggregate:
    calls: int
    errors: int
    total_latency_ms: int
    max_latency_ms: int
    latency_buckets: list[int]
    error_categories: dict[str, int]


class McpMetricsRegistry:
    def __init__(self, *, latency_buckets_ms: tuple[int, ...] = _DEFAULT_LATENCY_BUCKETS_MS) -> None:
        self._latency_buckets_ms = latency_buckets_ms
        self._lock = Lock()
        self._tool_metrics: dict[str, ToolMetricAggregate] = {}
        self._auth_failures: dict[int, int] = defaultdict(int)

    def record_tool_call(
        self,
        *,
        tool_name: str,
        latency_ms: int,
        outcome: str,
        error_category: str | None = None,
    ) -> None:
        with self._lock:
            aggregate = self._tool_metrics.get(tool_name)
            if aggregate is None:
                aggregate = ToolMetricAggregate(
                    calls=0,
                    errors=0,
                    total_latency_ms=0,
                    max_latency_ms=0,
                    latency_buckets=_init_buckets(self._latency_buckets_ms),
                    error_categories={},
                )
                self._tool_metrics[tool_name] = aggregate
            aggregate.calls += 1
            aggregate.total_latency_ms += latency_ms
            aggregate.max_latency_ms = max(aggregate.max_latency_ms, latency_ms)
            aggregate.latency_buckets[_bucket_index(self._latency_buckets_ms, latency_ms)] += 1
            if outcome != "ok":
                aggregate.errors += 1
                if error_category is not None:
                    aggregate.error_categories[error_category] = aggregate.error_categories.get(error_category, 0) + 1

    def record_auth_failure(self, *, status_code: int) -> None:
        with self._lock:
            self._auth_failures[status_code] += 1

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            tool_metrics = {
                tool_name: {
                    "calls": aggregate.calls,
                    "errors": aggregate.errors,
                    "avg_latency_ms": (aggregate.total_latency_ms / aggregate.calls) if aggregate.calls else 0.0,
                    "max_latency_ms": aggregate.max_latency_ms,
                    "latency_buckets": aggregate.latency_buckets[:],
                    "error_categories": dict(sorted(aggregate.error_categories.items())),
                }
                for tool_name, aggregate in sorted(self._tool_metrics.items())
            }
            auth_failures = {
                str(status_code): count
                for status_code, count in sorted(self._auth_failures.items())
            }
        return {
            "latency_bucket_bounds_ms": list(self._latency_buckets_ms),
            "tool_calls": tool_metrics,
            "auth_failures": auth_failures,
        }

    def reset(self) -> None:
        with self._lock:
            self._tool_metrics.clear()
            self._auth_failures.clear()


_metrics_registry = McpMetricsRegistry()


def get_mcp_metrics_registry() -> McpMetricsRegistry:
    return _metrics_registry


__all__ = ["McpMetricsRegistry", "get_mcp_metrics_registry"]
