from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal
from typing import Any

from sqlalchemy import Float, Integer, Numeric, and_, cast as sql_cast, or_, select
from sqlalchemy.sql.elements import ColumnElement


_TRANSACTION_PRICE_BUCKETS: dict[str, tuple[Decimal, Decimal | None]] = {
    "0 - 100M": (Decimal("0"), Decimal("100000000")),
    "100M - 250M": (Decimal("100000000"), Decimal("250000000")),
    "250M - 500M": (Decimal("250000000"), Decimal("500000000")),
    "500M - 750M": (Decimal("500000000"), Decimal("750000000")),
    "750M - 1B": (Decimal("750000000"), Decimal("1000000000")),
    "1B - 2B": (Decimal("1000000000"), Decimal("2000000000")),
    "2B - 5B": (Decimal("2000000000"), Decimal("5000000000")),
    "5B - 10B": (Decimal("5000000000"), Decimal("10000000000")),
    "10B - 20B": (Decimal("10000000000"), Decimal("20000000000")),
    "20B+": (Decimal("20000000000"), None),
}


_NUMERIC_SQL_TYPES = (Float, Integer, Numeric)


def build_transaction_price_bucket_filter(
    column: Any,
    selected_buckets: Iterable[str],
) -> ColumnElement[bool] | None:
    predicates: list[ColumnElement[bool]] = []
    column_type = getattr(column, "type", None)
    numeric_column = (
        column
        if isinstance(column_type, _NUMERIC_SQL_TYPES)
        else sql_cast(column, Float)
    )

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


def build_any_counsel_agreement_uuid_subquery(
    *,
    canonical_names: Iterable[str],
    agreement_counsel: Any,
    counsel: Any,
) -> Any | None:
    """Match agreements where the firm appears on either side (target or acquirer)."""
    selected_names = [value for value in canonical_names if value]
    if not selected_names:
        return None

    return (
        select(agreement_counsel.agreement_uuid)
        .join(counsel, counsel.counsel_id == agreement_counsel.counsel_id)
        .where(counsel.canonical_name.in_(selected_names))
        .distinct()
    )
