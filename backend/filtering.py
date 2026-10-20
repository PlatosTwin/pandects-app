"""Shared SQL filter builders for agreement-level metadata.

Every search surface (REST and MCP) filters on the same agreement metadata --
filing-date windows, price buckets, deal attributes, counsel, PE flags. Those
columns exist under identical names on both the ``agreements`` table and the
denormalized ``latest_sections_search`` table, so a single implementation,
``apply_agreement_filters``, serves both: callers pass whichever model the
query is built on as ``cols``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Any, Protocol, cast

from sqlalchemy import Float, Integer, Numeric, and_, cast as sql_cast, func, or_, select, true
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


class _PublicEligibleAgreementModel(Protocol):
    __table__: Any
    agreement_uuid: Any
    verified: Any


def agreement_is_public_eligible_expr(
    agreements: _PublicEligibleAgreementModel,
) -> ColumnElement[bool]:
    """Agreement rows the public surfaces may return: not gated, or gated but verified.

    Use when the ``agreements`` table is already part of the query.
    """
    gated_col = agreements.__table__.c.get("gated")
    if gated_col is None:
        return true()
    return or_(
        func.coalesce(gated_col, 0) != 1,
        func.coalesce(agreements.verified, 0) == 1,
    )


def agreement_uuid_is_public_eligible_expr(
    agreement_uuid_column: Any,
    agreements: _PublicEligibleAgreementModel,
) -> ColumnElement[bool]:
    """The same eligibility rule for a query keyed by agreement_uuid on another table.

    Section-level queries read the denormalized ``latest_sections_search`` and never
    join ``agreements``; joining it only to apply the gate would cost a primary-key
    lookup per candidate row. The gated set is small, so an uncorrelated
    ``NOT IN (gated agreements)`` is evaluated once and probed per row instead.
    """
    gated_col = agreements.__table__.c.get("gated")
    if gated_col is None:
        return true()
    gated_agreement_uuids = (
        select(agreements.agreement_uuid)
        .where(
            func.coalesce(gated_col, 0) == 1,
            func.coalesce(agreements.verified, 0) != 1,
        )
        .correlate(None)
    )
    return cast(ColumnElement[bool], agreement_uuid_column.notin_(gated_agreement_uuids))


def pe_flag_values(values: Iterable[object]) -> list[int]:
    """Map the public ``"true"``/``"false"`` PE filter values to stored 1/0 flags."""
    mapped: list[int] = []
    for value in values:
        if value == "true":
            mapped.append(1)
        elif value == "false":
            mapped.append(0)
    return mapped


# Exact-match list filters, keyed by both the request argument and the column name.
AGREEMENT_LIST_FILTER_FIELDS: tuple[str, ...] = (
    "target",
    "acquirer",
    "transaction_consideration",
    "target_type",
    "acquirer_type",
    "target_industry",
    "acquirer_industry",
    "deal_status",
    "attitude",
    "deal_type",
    "purpose",
)
AGREEMENT_PRICE_FILTER_FIELDS: tuple[str, ...] = (
    "transaction_price_total",
    "transaction_price_stock",
    "transaction_price_cash",
    "transaction_price_assets",
)
AGREEMENT_PE_FILTER_FIELDS: tuple[str, ...] = ("target_pe", "acquirer_pe")
AGREEMENT_RANGE_FILTER_FIELDS: tuple[str, ...] = (
    "year_min",
    "year_max",
    "filed_after",
    "filed_before",
)


def _list_arg(args: Mapping[str, object], key: str) -> list[Any]:
    value = args.get(key)
    if isinstance(value, (list, tuple)):
        return list(cast(Iterable[Any], value))
    return []


def _str_arg(args: Mapping[str, object], key: str) -> str | None:
    value = args.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def agreement_filter_predicates(
    cols: Any,
    args: Mapping[str, object],
    *,
    agreement_counsel: Any,
    counsel: Any,
) -> list[ColumnElement[bool]]:
    """Predicates for every agreement-metadata filter present in ``args``.

    ``cols`` is the model the query is built on -- ``Agreements`` or
    ``LatestSectionsSearch`` -- since both expose these columns by the same names.
    Keys a route's schema does not define are simply absent from ``args`` and
    contribute nothing, so one implementation serves every surface:

    - ``year``: any of the listed filing years.
    - ``year_min`` / ``year_max``: inclusive filing-year bounds.
    - ``filed_after`` (inclusive) / ``filed_before`` (exclusive): ISO dates.
    - list filters (``target``, ``deal_type``, ...): exact match on any value.
    - price buckets: any of the named buckets (unknown bucket names are ignored).
    - ``target_counsel`` / ``acquirer_counsel``: canonical firm on that side;
      ``any_counsel``: canonical firm on either side.
    - ``target_pe`` / ``acquirer_pe``: ``"true"``/``"false"``.
    - ``agreement_uuid``: one agreement.

    Section-scoped filters (``section_uuid``, ``standard_id``) depend on the table and
    stay with the caller.
    """
    predicates: list[ColumnElement[bool]] = []

    years = [year for year in _list_arg(args, "year") if isinstance(year, int)]
    if years:
        predicates.append(
            or_(
                *(
                    and_(
                        cols.filing_date >= f"{year:04d}-01-01",
                        cols.filing_date < f"{year + 1:04d}-01-01",
                    )
                    for year in years
                )
            )
        )

    year_min = args.get("year_min")
    if isinstance(year_min, int):
        predicates.append(cols.filing_date >= f"{year_min:04d}-01-01")
    year_max = args.get("year_max")
    if isinstance(year_max, int):
        predicates.append(cols.filing_date < f"{year_max + 1:04d}-01-01")
    filed_after = _str_arg(args, "filed_after")
    if filed_after is not None:
        predicates.append(cols.filing_date >= filed_after)
    filed_before = _str_arg(args, "filed_before")
    if filed_before is not None:
        predicates.append(cols.filing_date < filed_before)

    for field_name in ("target", "acquirer"):
        values = _list_arg(args, field_name)
        if values:
            predicates.append(getattr(cols, field_name).in_(values))

    for field_name in AGREEMENT_PRICE_FILTER_FIELDS:
        bucket_filter = build_transaction_price_bucket_filter(
            getattr(cols, field_name),
            [value for value in _list_arg(args, field_name) if isinstance(value, str)],
        )
        if bucket_filter is not None:
            predicates.append(bucket_filter)

    for field_name in ("transaction_consideration", "target_type", "acquirer_type"):
        values = _list_arg(args, field_name)
        if values:
            predicates.append(getattr(cols, field_name).in_(values))

    for side in ("target", "acquirer"):
        counsel_subquery = build_canonical_counsel_agreement_uuid_subquery(
            side=side,
            canonical_names=_list_arg(args, f"{side}_counsel"),
            agreement_counsel=agreement_counsel,
            counsel=counsel,
        )
        if counsel_subquery is not None:
            predicates.append(cols.agreement_uuid.in_(counsel_subquery))
    any_counsel_subquery = build_any_counsel_agreement_uuid_subquery(
        canonical_names=_list_arg(args, "any_counsel"),
        agreement_counsel=agreement_counsel,
        counsel=counsel,
    )
    if any_counsel_subquery is not None:
        predicates.append(cols.agreement_uuid.in_(any_counsel_subquery))

    for field_name in (
        "target_industry",
        "acquirer_industry",
        "deal_status",
        "attitude",
        "deal_type",
        "purpose",
    ):
        values = _list_arg(args, field_name)
        if values:
            predicates.append(getattr(cols, field_name).in_(values))

    for field_name in AGREEMENT_PE_FILTER_FIELDS:
        flags = pe_flag_values(_list_arg(args, field_name))
        if flags:
            predicates.append(getattr(cols, field_name).in_(flags))

    agreement_uuid = _str_arg(args, "agreement_uuid")
    if agreement_uuid is not None:
        predicates.append(cols.agreement_uuid == agreement_uuid)

    return predicates


def apply_agreement_filters(
    query: Any,
    cols: Any,
    args: Mapping[str, object],
    *,
    agreement_counsel: Any,
    counsel: Any,
) -> Any:
    """Apply every agreement-metadata filter in ``args`` to ``query``.

    See ``agreement_filter_predicates`` for the semantics of each argument.
    """
    for predicate in agreement_filter_predicates(
        cols,
        args,
        agreement_counsel=agreement_counsel,
        counsel=counsel,
    ):
        query = query.filter(predicate)
    return query


def has_agreement_filters(args: Mapping[str, object]) -> bool:
    """True when ``args`` carries any filter ``apply_agreement_filters`` would apply."""
    list_fields = (
        "year",
        *AGREEMENT_LIST_FILTER_FIELDS,
        *AGREEMENT_PRICE_FILTER_FIELDS,
        *AGREEMENT_PE_FILTER_FIELDS,
        "target_counsel",
        "acquirer_counsel",
        "any_counsel",
    )
    for field_name in list_fields:
        if _list_arg(args, field_name):
            return True
    if isinstance(args.get("year_min"), int) or isinstance(args.get("year_max"), int):
        return True
    return any(
        _str_arg(args, key) is not None
        for key in ("filed_after", "filed_before", "agreement_uuid")
    )


__all__ = [
    "AGREEMENT_LIST_FILTER_FIELDS",
    "AGREEMENT_PE_FILTER_FIELDS",
    "AGREEMENT_PRICE_FILTER_FIELDS",
    "AGREEMENT_RANGE_FILTER_FIELDS",
    "agreement_filter_predicates",
    "agreement_is_public_eligible_expr",
    "agreement_uuid_is_public_eligible_expr",
    "apply_agreement_filters",
    "build_any_counsel_agreement_uuid_subquery",
    "build_canonical_counsel_agreement_uuid_subquery",
    "build_transaction_price_bucket_filter",
    "has_agreement_filters",
    "pe_flag_values",
]
