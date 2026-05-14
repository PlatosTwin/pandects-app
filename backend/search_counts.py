from __future__ import annotations

from typing import Any, Callable, Protocol, cast

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError


class SearchCountDeps(Protocol):
    @property
    def db(self) -> Any: ...

    @property
    def _SEARCH_EXPLAIN_ESTIMATE_ENABLED(self) -> bool: ...

    @property
    def _to_int(self) -> "_ToIntProtocol": ...


class _ToIntProtocol(Protocol):
    def __call__(self, value: object, *, default: int = 0) -> int: ...


class _CompilableStatement(Protocol):
    def compile(
        self,
        *,
        dialect: object,
        compile_kwargs: dict[str, object],
    ) -> object: ...


class StatementQuery(Protocol):
    def order_by(self, *clauses: object) -> "StatementQuery": ...

    def count(self) -> object: ...

    @property
    def statement(self) -> _CompilableStatement: ...


def exact_query_count(deps: SearchCountDeps, query: object) -> int:
    return deps._to_int(cast(StatementQuery, query).order_by(None).count())


def estimated_query_row_count(deps: SearchCountDeps, query: object) -> int | None:
    """Ask the database for an approximate row count when exact counts are too expensive."""
    if not deps._SEARCH_EXPLAIN_ESTIMATE_ENABLED:
        return None
    db = getattr(deps, "db")
    bind = db.session.get_bind()
    if bind.dialect.name == "sqlite":
        return None
    try:
        typed_query = cast(StatementQuery, query)
        selectable = typed_query.order_by(None).statement
        compiled = selectable.compile(
            dialect=bind.dialect,
            compile_kwargs={"literal_binds": True},
        )
        explain_rows = (
            db.session.execute(text(f"EXPLAIN {compiled}"))
            .mappings()
            .all()
        )
    except SQLAlchemyError:
        return None

    max_rows = 0
    for explain_row in explain_rows:
        row_estimate = deps._to_int(explain_row.get("rows"))
        if row_estimate > max_rows:
            max_rows = row_estimate
    return max_rows if max_rows > 0 else None


def search_total_count_metadata(
    deps: SearchCountDeps,
    *,
    query: object,
    page: int,
    page_size: int,
    item_count: int,
    has_next: bool,
    has_filters: bool,
    count_mode: str,
    estimated_query_row_count_fn: Callable[[object], int | None],
    estimated_table_rows_fn: Callable[[], int | None] | None = None,
) -> tuple[int, bool, str]:
    """Return honest count metadata without current-page lower-bound approximations."""
    if count_mode == "exact":
        return exact_query_count(deps, query), False, "query_count"

    if has_filters:
        if page <= 1:
            return exact_query_count(deps, query), False, "query_count"

        exact_terminal_total = ((page - 1) * page_size) + item_count
        if not has_next:
            return exact_terminal_total, False, "page_terminal_count"

        minimum_observed_total = exact_terminal_total + 1
        estimate = estimated_query_row_count_fn(query)
        if estimate is not None and estimate >= minimum_observed_total:
            return estimate, True, "query_estimate"

        return exact_query_count(deps, query), False, "query_count"

    table_rows = estimated_table_rows_fn() if estimated_table_rows_fn is not None else None
    if table_rows is not None:
        return max(item_count, table_rows), True, "table_estimate"

    minimum_observed_total = ((page - 1) * page_size) + item_count + (1 if has_next else 0)
    estimate = estimated_query_row_count_fn(query)
    if estimate is not None and estimate >= minimum_observed_total:
        return estimate, True, "query_estimate"

    return exact_query_count(deps, query), False, "query_count"


def count_metadata_payload(
    *,
    total_count_is_approximate: bool,
    count_method: str,
    exact_count_requested: bool,
) -> dict[str, object]:
    planning_reliability = "high"
    if total_count_is_approximate:
        planning_reliability = (
            "medium"
            if count_method in {"query_estimate", "table_estimate"}
            else "low"
        )
    return {
        "mode": "estimated" if total_count_is_approximate else "exact",
        "method": count_method,
        "planning_reliability": planning_reliability,
        "exact_count_requested": exact_count_requested,
    }
