from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy import Float, and_, cast as sql_cast, or_, select
from sqlalchemy.sql.elements import ColumnElement


_TRANSACTION_PRICE_BUCKETS: dict[str, tuple[float, float | None]] = {
    "0 - 100M": (0.0, 100_000_000.0),
    "100M - 250M": (100_000_000.0, 250_000_000.0),
    "250M - 500M": (250_000_000.0, 500_000_000.0),
    "500M - 750M": (500_000_000.0, 750_000_000.0),
    "750M - 1B": (750_000_000.0, 1_000_000_000.0),
    "1B - 5B": (1_000_000_000.0, 5_000_000_000.0),
    "5B - 10B": (5_000_000_000.0, 10_000_000_000.0),
    "10B - 20B": (10_000_000_000.0, 20_000_000_000.0),
    "20B+": (20_000_000_000.0, None),
}


def build_transaction_price_bucket_filter(
    column: Any,
    selected_buckets: Iterable[str],
) -> ColumnElement[bool] | None:
    predicates: list[ColumnElement[bool]] = []
    numeric_column = sql_cast(column, Float)

    for bucket in selected_buckets:
        bounds = _TRANSACTION_PRICE_BUCKETS.get(bucket)
        if bounds is None:
            continue
        lower_bound, upper_bound = bounds
        if upper_bound is None:
            predicates.append(numeric_column >= lower_bound)
        else:
            predicates.append(
                and_(numeric_column >= lower_bound, numeric_column < upper_bound)
            )

    if not predicates:
        return None

    return or_(*predicates)


def build_canonical_counsel_agreement_uuid_subquery(
    *,
    side: str,
    canonical_names: Iterable[str],
    agreement_counsel: Any,
    counsel: Any,
) -> Any | None:
    selected_names = [value for value in canonical_names if value]
    if not selected_names:
        return None

    return (
        select(agreement_counsel.agreement_uuid)
        .join(counsel, counsel.counsel_id == agreement_counsel.counsel_id)
        .where(
            agreement_counsel.side == side,
            counsel.canonical_name.in_(selected_names),
        )
        .distinct()
    )
