from __future__ import annotations

import threading
import time
from typing import Any, Protocol, cast

from marshmallow import ValidationError
from sqlalchemy import and_, asc, desc, distinct, or_, text
from sqlalchemy.dialects.mysql import match as mysql_match
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.sql.elements import ColumnElement

from backend.filtering import (
    build_canonical_counsel_agreement_uuid_subquery,
    build_transaction_price_bucket_filter,
)
from backend.routes.deps import AccessContextProtocol, SectionsServiceDeps
from backend.search_counts import build_search_count_cache_key
from backend.schemas.sections import SectionsArgsPayload
from backend.text_search import (
    TEXT_SEARCH_MAX_STATEMENT_SECONDS,
    compile_boolean_text_query,
    compile_phrase_text_pattern,
)


SECTION_TEXT_SEARCH_UNAVAILABLE_MESSAGE = "Section text search is not available on this server yet."
TEXT_QUERY_TOO_EXPENSIVE_MESSAGE = (
    "text_query matched too many sections to rank within the time limit. "
    "Use a longer or more distinctive phrase; extra words shrink the candidate set. "
    "Metadata filters do not help here — the full-text scan runs before them."
)
_STATEMENT_TIMEOUT_ERROR_CODE = 1969
# HA_ERR_OUT_OF_MEM. InnoDB raises it when a FULLTEXT query's result cache
# exceeds innodb_ft_result_cache_limit -- the bound that stops a dense phrase
# from exhausting the server's memory. Verified against MariaDB 11.8.2:
# "ERROR 128 (HY000): Table handler out of memory".
_FTS_RESULT_CACHE_ERROR_CODE = 128

# Phrase search has two viable plans with opposite failure modes. The FULLTEXT
# plan narrows by the candidate index first, so its cost scales with how many
# sections contain the phrase's words -- fast for rare phrases, unusable for
# common ones ("material adverse effect" matches ~16% of the corpus). The
# date-ordered plan walks latest_sections_search in sort order and applies the
# phrase REGEXP per row, so it stops early exactly when the phrase is common and
# degrades when it is rare. Neither wins everywhere and the candidate count that
# would let us choose up front costs more than the query it optimizes, so the
# first request for a filter signature races them: try one under a short budget,
# fall back to the other, then remember the winner for subsequent pages.
TEXT_PLAN_FULLTEXT = "fulltext_candidates"
TEXT_PLAN_DATE_SCAN = "date_ordered_scan"
# The date-ordered plan wins fast and loses slowly: when it suits the phrase it
# answers in well under a second, and when it does not it grinds. A tight first
# budget therefore cuts a wrong guess short, while the FULLTEXT fallback still
# gets the room it needs for the rare phrases it is good at.
_TEXT_PLAN_FIRST_ATTEMPT_SECONDS = 2
_TEXT_COUNT_BUDGET_SECONDS = 6
_TEXT_PLAN_CACHE_TTL_SECONDS = 900.0
_TEXT_PLAN_CACHE_MAX_KEYS = 512
_text_plan_cache: dict[str, tuple[float, str]] = {}
_text_plan_cache_lock = threading.Lock()
# Counts that already blew the budget once. Re-running them on every page of the
# same search burns the budget again to reach the same verdict.
_text_count_unaffordable: dict[str, float] = {}


def _remembered_text_plan(cache_key: str | None) -> str | None:
    if cache_key is None:
        return None
    now = time.time()
    with _text_plan_cache_lock:
        entry = _text_plan_cache.get(cache_key)
        if entry is None or (now - entry[0]) >= _TEXT_PLAN_CACHE_TTL_SECONDS:
            return None
        return entry[1]


def _remember_text_plan(cache_key: str | None, plan: str) -> None:
    if cache_key is None:
        return
    now = time.time()
    with _text_plan_cache_lock:
        expired = [
            key
            for key, (stamp, _plan) in _text_plan_cache.items()
            if (now - stamp) >= _TEXT_PLAN_CACHE_TTL_SECONDS
        ]
        for key in expired:
            del _text_plan_cache[key]
        overflow = len(_text_plan_cache) - _TEXT_PLAN_CACHE_MAX_KEYS
        if overflow > 0:
            oldest = sorted(_text_plan_cache.items(), key=lambda item: item[1][0])
            for key, _entry in oldest[:overflow]:
                del _text_plan_cache[key]
        _text_plan_cache[cache_key] = (now, plan)


def reset_text_plan_cache() -> None:
    """Drop every remembered plan choice. Test seam; not used at runtime."""
    with _text_plan_cache_lock:
        _text_plan_cache.clear()
        _text_count_unaffordable.clear()


def _count_known_unaffordable(cache_key: str | None) -> bool:
    if cache_key is None:
        return False
    now = time.time()
    with _text_plan_cache_lock:
        stamp = _text_count_unaffordable.get(cache_key)
        if stamp is None or (now - stamp) >= _TEXT_PLAN_CACHE_TTL_SECONDS:
            return False
        return True


def _remember_count_unaffordable(cache_key: str | None) -> None:
    if cache_key is None:
        return
    now = time.time()
    with _text_plan_cache_lock:
        expired = [
            key
            for key, stamp in _text_count_unaffordable.items()
            if (now - stamp) >= _TEXT_PLAN_CACHE_TTL_SECONDS
        ]
        for key in expired:
            del _text_count_unaffordable[key]
        overflow = len(_text_count_unaffordable) - _TEXT_PLAN_CACHE_MAX_KEYS
        if overflow > 0:
            oldest = sorted(_text_count_unaffordable.items(), key=lambda item: item[1])
            for key, _stamp in oldest[:overflow]:
                del _text_count_unaffordable[key]
        _text_count_unaffordable[cache_key] = now


def _error_code(exc: SQLAlchemyError) -> object | None:
    orig_args = cast(tuple[object, ...], getattr(getattr(exc, "orig", None), "args", ()))
    return orig_args[0] if orig_args else None


def is_statement_timeout_error(exc: SQLAlchemyError) -> bool:
    """True when MariaDB aborted the statement for exceeding max_statement_time."""
    return _error_code(exc) == _STATEMENT_TIMEOUT_ERROR_CODE


def is_fts_result_cache_error(exc: SQLAlchemyError) -> bool:
    """True when a FULLTEXT query outgrew innodb_ft_result_cache_limit."""
    return _error_code(exc) == _FTS_RESULT_CACHE_ERROR_CODE


def is_text_plan_exhausted_error(exc: SQLAlchemyError) -> bool:
    """True when a text query failed in a way another plan or a bound can absorb.

    Both outcomes mean "this plan cannot serve this phrase here": one ran out of
    time, the other out of memory. Neither is a fault the caller can act on, so
    both fall through to the other plan and then to a degraded count.
    """
    return is_statement_timeout_error(exc) or is_fts_result_cache_error(exc)


def _join_section_text_search(
    query: Any,
    *,
    latest: Any,
    section_text_search: Any,
    text_query: str,
) -> Any:
    if not text_query.strip():
        raise ValueError("text_query must contain searchable text.")
    if section_text_search is None:
        raise ValidationError({"text_query": [SECTION_TEXT_SEARCH_UNAVAILABLE_MESSAGE]})
    return query.execution_options(
        max_statement_time=TEXT_SEARCH_MAX_STATEMENT_SECONDS
    ).join(
        section_text_search,
        section_text_search.section_uuid == latest.section_uuid,
    )


def build_section_text_match_expression(
    normalized_text_column: object,
    *,
    text_query: str,
    match_mode: str,
) -> ColumnElement[bool]:
    """Build a parameterized MariaDB FULLTEXT predicate for normalized section text."""
    compiled_query = compile_boolean_text_query(text_query, match_mode)
    candidate_match = cast(
        ColumnElement[bool],
        mysql_match(
            cast(Any, normalized_text_column),
            against=compiled_query,
            in_boolean_mode=True,
        ),
    )
    if match_mode != "phrase":
        return candidate_match
    phrase_pattern = compile_phrase_text_pattern(text_query)
    phrase_match = cast(
        ColumnElement[bool],
        cast(Any, normalized_text_column).op("REGEXP")(phrase_pattern),
    )
    return and_(candidate_match, phrase_match)


def build_section_text_candidate_expression(
    normalized_text_column: object,
    *,
    text_query: str,
    match_mode: str,
) -> ColumnElement[bool]:
    """Build the indexed candidate predicate used for result and count planning."""
    return cast(
        ColumnElement[bool],
        mysql_match(
            cast(Any, normalized_text_column),
            against=compile_boolean_text_query(text_query, match_mode),
            in_boolean_mode=True,
        ),
    )


def apply_section_text_join(
    query: Any,
    *,
    latest: Any,
    section_text_search: Any,
    text_query: str | None,
) -> Any:
    """Join the private text index without applying any text predicate.

    Both phrase plans need the join (the REGEXP reads normalized_text); they
    differ only in whether the FULLTEXT candidate predicate is also applied.
    """
    if text_query is None:
        return query
    return _join_section_text_search(
        query,
        latest=latest,
        section_text_search=section_text_search,
        text_query=text_query,
    )


class _CompilableStatement(Protocol):
    def compile(
        self,
        *,
        dialect: object,
        compile_kwargs: dict[str, object],
    ) -> object:
        ...


class _StatementQuery(Protocol):
    def order_by(self, *clauses: object) -> "_StatementQuery":
        ...

    def count(self) -> object:
        ...

    @property
    def statement(self) -> _CompilableStatement:
        ...


def estimated_query_row_count(deps: SectionsServiceDeps, query: object) -> int | None:
    """Ask the database for an approximate row count when exact counts are too expensive."""
    if not deps._SEARCH_EXPLAIN_ESTIMATE_ENABLED:
        return None
    db = deps.db
    to_int = deps._to_int
    bind = db.session.get_bind()
    if bind.dialect.name == "sqlite":
        return None
    try:
        typed_query = cast(_StatementQuery, query)
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
        row_estimate = to_int(explain_row.get("rows"))
        if row_estimate > max_rows:
            max_rows = row_estimate
    return max_rows if max_rows > 0 else None


def estimated_latest_sections_search_table_rows(deps: SectionsServiceDeps) -> int | None:
    """Read MariaDB's table estimate for the unfiltered search corpus."""
    db = deps.db
    bind = db.session.get_bind()
    if bind.dialect.name == "sqlite":
        return None
    try:
        row = (
            db.session.execute(
                text(
                    """
                    SELECT TABLE_ROWS
                    FROM information_schema.TABLES
                    WHERE TABLE_SCHEMA = DATABASE()
                      AND TABLE_NAME = 'latest_sections_search'
                    """
                )
            )
            .mappings()
            .first()
        )
    except SQLAlchemyError:
        return None
    if row is None:
        return None
    return deps._to_int(row.get("TABLE_ROWS"))


def sections_total_count_metadata(
    deps: SectionsServiceDeps,
    *,
    query: object,
    page: int,
    page_size: int,
    item_count: int,
    has_next: bool,
    has_filters: bool,
    count_mode: str,
    count_cache_key: str | None = None,
    prefer_estimate: bool = False,
) -> tuple[int, bool, str]:
    """Return `(total_count, is_approximate, method)` without forcing exact counts unless requested.

    Filtered searches may use a database-planner estimate when it remains consistent
    with the rows already observed. Unfiltered searches can fall back to the table-level
    estimate because the endpoint already reads from a denormalized latest-sections table.

    Exact counts (explicit `count_mode=exact`, ordinary filtered first pages, and
    estimate fallbacks) are memoized per filter signature via `count_cache_key` so
    paging one search re-counts once instead of once per page. Text search can set
    `prefer_estimate` to avoid a first-page count scan. `count_cache_key=None`
    always counts fresh.
    """
    estimated_query_row_count_fn = deps._estimated_query_row_count
    estimated_table_rows_fn = deps._estimated_latest_sections_search_table_rows
    if count_mode == "exact":
        exact_total = deps._cached_exact_query_count(query, cache_key=count_cache_key)
        return exact_total, False, "query_count"

    if has_filters:
        if page <= 1 and not prefer_estimate:
            exact_total = deps._cached_exact_query_count(query, cache_key=count_cache_key)
            return exact_total, False, "query_count"

        observed_total = ((page - 1) * page_size) + item_count
        minimum_total = observed_total + (1 if has_next else 0)
        estimate = estimated_query_row_count_fn(query)
        if estimate is not None and estimate >= minimum_total:
            return estimate, True, "table_estimate"

        # Never present a page-derived lower bound as an approximate corpus total.
        # If the optimizer cannot supply a credible estimate, use the memoized exact
        # count instead.
        exact_total = deps._cached_exact_query_count(query, cache_key=count_cache_key)
        return exact_total, False, "query_count"

    table_rows = estimated_table_rows_fn()
    if table_rows is None:
        table_rows = deps._cached_exact_query_count(query, cache_key=count_cache_key)
        total_count = max(item_count, table_rows)
        return total_count, False, "query_count"
    total_count = max(item_count, table_rows)
    return total_count, True, "table_estimate"


def _sections_count_metadata_payload(
    *,
    total_count_is_approximate: bool,
    count_method: str,
    exact_count_requested: bool,
) -> dict[str, object]:
    planning_reliability = "high"
    if total_count_is_approximate:
        planning_reliability = (
            "medium"
            if count_method in {"table_estimate", "fulltext_candidate_count"}
            else "low"
        )
    return {
        "mode": "estimated" if total_count_is_approximate else "exact",
        "method": count_method,
        "planning_reliability": planning_reliability,
        "exact_count_requested": exact_count_requested,
    }


def _sections_interpretation_payload(
    *,
    parsed_args: SectionsArgsPayload,
    standard_ids_expanded: bool,
    total_count_is_approximate: bool,
    count_method: str,
) -> dict[str, object]:
    applied_filters: list[dict[str, str]] = []
    for field_name in (
        "year",
        "target",
        "acquirer",
        "transaction_price_total",
        "transaction_price_stock",
        "transaction_price_cash",
        "transaction_price_assets",
        "transaction_consideration",
        "target_type",
        "acquirer_type",
        "target_counsel",
        "acquirer_counsel",
        "target_industry",
        "acquirer_industry",
        "deal_status",
        "attitude",
        "deal_type",
        "purpose",
        "target_pe",
        "acquirer_pe",
        "agreement_uuid",
        "section_uuid",
    ):
        raw_value = parsed_args[field_name]
        if isinstance(raw_value, list):
            if not raw_value:
                continue
        elif raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
            continue
        representation = "first_class_section_field" if field_name == "section_uuid" else "first_class_agreement_field"
        applied_filters.append(
            {
                "field": field_name,
                "representation": representation,
                "match_kind": "exact_metadata_filter",
            }
        )

    for range_field in ("year_min", "year_max", "filed_after", "filed_before"):
        if parsed_args[range_field] is not None:
            applied_filters.append(
                {
                    "field": range_field,
                    "representation": "first_class_agreement_field",
                    "match_kind": "range_metadata_filter",
                }
            )

    taxonomy_filters = [
        {
            "standard_id": standard_id,
            "match_mode": "expanded_descendants" if standard_ids_expanded else "exact_node",
        }
        for standard_id in parsed_args["standard_id"]
        if standard_id
    ]

    notes: list[str] = []
    text_query = parsed_args["text_query"]
    if text_query is not None and text_query.strip():
        applied_filters.append(
            {
                "field": "text_query",
                "representation": "derived_from_text",
                "match_kind": f'{parsed_args["text_match_mode"]}_full_text',
            }
        )
        notes.append(
            "Text search is case-insensitive and literal; a trailing `*` performs word-prefix matching."
        )
    if taxonomy_filters:
        notes.append("Taxonomy filters reflect clause-family assignments and may act as proxies for broader legal concepts.")
    if total_count_is_approximate:
        if count_method == "fulltext_candidate_count":
            notes.append(
                "The text count is a stable upper bound from indexed all-term candidates; "
                "use count_mode=exact when phrase-filtered pagination certainty matters."
            )
        else:
            notes.append("Counts are approximate under the current mode; use count_mode=exact when pagination certainty matters.")
    elif count_method == "query_count":
        notes.append("Counts were computed exactly from the current filtered query.")

    return {
        "applied_filters": applied_filters,
        "taxonomy_filters": taxonomy_filters,
        "heuristics_used": [],
        "notes": notes,
    }


def _count_with_timeout_fallback(
    deps: SectionsServiceDeps,
    *,
    query: Any,
    cache_key: str | None,
) -> tuple[int | None, bool]:
    """Exact COUNT(*) when it fits the budget, otherwise an optimizer estimate.

    A common phrase can match a sixth of the corpus, where an exact count costs
    far more than the page it accompanies. Returns (count, is_exact); the count
    is None only when the estimate is also unavailable.
    """
    if _count_known_unaffordable(cache_key):
        return None, False
    try:
        budgeted = query.execution_options(
            max_statement_time=_TEXT_COUNT_BUDGET_SECONDS
        )
        return deps._cached_exact_query_count(budgeted, cache_key=cache_key), True
    except OperationalError as exc:
        if not is_text_plan_exhausted_error(exc):
            raise
        deps.db.session.rollback()
    _remember_count_unaffordable(cache_key)
    # EXPLAIN reports rows=1 for a FULLTEXT plan, so this is None for exactly the
    # phrases that need it most; callers fall back to a page-derived lower bound.
    return estimated_query_row_count(deps, query), False


def _fetch_phrase_page(
    deps: SectionsServiceDeps,
    *,
    fulltext_query: Any,
    date_scan_query: Any,
    cache_key: str | None,
    offset: int,
    limit: int,
) -> tuple[list[object], Any]:
    """Serve a phrase page under whichever plan can actually run it.

    Returns the rows and the query that produced them, so the count path runs
    against the same shape the page came from.
    """
    remembered = _remembered_text_plan(cache_key)
    ordered_plans: list[tuple[str, Any]] = [
        (TEXT_PLAN_DATE_SCAN, date_scan_query),
        (TEXT_PLAN_FULLTEXT, fulltext_query),
    ]
    if remembered == TEXT_PLAN_FULLTEXT:
        ordered_plans.reverse()
    (first_plan, first_query), (second_plan, second_query) = ordered_plans

    if remembered is not None:
        rows = cast(list[object], first_query.offset(offset).limit(limit).all())
        return rows, first_query

    try:
        budgeted = first_query.execution_options(
            max_statement_time=_TEXT_PLAN_FIRST_ATTEMPT_SECONDS
        )
        rows = cast(list[object], budgeted.offset(offset).limit(limit).all())
    except OperationalError as exc:
        if not is_text_plan_exhausted_error(exc):
            raise
        # The abort leaves the session unusable; the fallback needs a clean one.
        deps.db.session.rollback()
    else:
        _remember_text_plan(cache_key, first_plan)
        return rows, first_query

    rows = cast(list[object], second_query.offset(offset).limit(limit).all())
    _remember_text_plan(cache_key, second_plan)
    return rows, second_query


def run_sections(
    deps: SectionsServiceDeps,
    *,
    ctx: AccessContextProtocol,
    parsed_args: SectionsArgsPayload,
    hydrate_xml: bool = True,
) -> dict[str, object]:
    try:
        return _run_sections(deps, ctx=ctx, parsed_args=parsed_args, hydrate_xml=hydrate_xml)
    except OperationalError as exc:
        text_query = parsed_args["text_query"]
        if not (text_query and text_query.strip() and is_text_plan_exhausted_error(exc)):
            raise
        deps.db.session.rollback()
        raise ValidationError({"text_query": [TEXT_QUERY_TOO_EXPENSIVE_MESSAGE]}) from exc


def _run_sections(
    deps: SectionsServiceDeps,
    *,
    ctx: AccessContextProtocol,
    parsed_args: SectionsArgsPayload,
    hydrate_xml: bool,
) -> dict[str, object]:
    db = deps.db
    agreement_counsel = deps.AgreementCounsel
    counsel = deps.Counsel
    latest = deps.LatestSectionsSearch
    section_text_search = deps.SectionTextSearch
    sections = deps.Sections
    row_mapping_as_dict = deps._row_mapping_as_dict
    pagination_metadata = deps._pagination_metadata
    dedupe_preserve_order = deps._dedupe_preserve_order
    expand_taxonomy_cached = deps._expand_taxonomy_standard_ids_cached
    standard_id_filter_expr = deps._standard_id_filter_expr
    parse_standard_ids = deps._parse_section_standard_ids
    year_from_filing_date = deps._year_from_filing_date_value

    years = parsed_args["year"]
    year_min = parsed_args["year_min"]
    year_max = parsed_args["year_max"]
    filed_after = parsed_args["filed_after"]
    filed_before = parsed_args["filed_before"]
    targets = parsed_args["target"]
    acquirers = parsed_args["acquirer"]
    standard_ids = parsed_args["standard_id"]
    transaction_price_totals = parsed_args["transaction_price_total"]
    transaction_price_stocks = parsed_args["transaction_price_stock"]
    transaction_price_cashes = parsed_args["transaction_price_cash"]
    transaction_price_assets = parsed_args["transaction_price_assets"]
    transaction_considerations = parsed_args["transaction_consideration"]
    target_types = parsed_args["target_type"]
    acquirer_types = parsed_args["acquirer_type"]
    target_counsels = parsed_args["target_counsel"]
    acquirer_counsels = parsed_args["acquirer_counsel"]
    target_industries = parsed_args["target_industry"]
    acquirer_industries = parsed_args["acquirer_industry"]
    deal_statuses = parsed_args["deal_status"]
    attitudes = parsed_args["attitude"]
    deal_types = parsed_args["deal_type"]
    purposes = parsed_args["purpose"]
    target_pes = parsed_args["target_pe"]
    acquirer_pes = parsed_args["acquirer_pe"]
    requested_metadata_fields = dedupe_preserve_order(parsed_args["metadata"])
    agreement_uuid = parsed_args["agreement_uuid"]
    section_uuid = parsed_args["section_uuid"]
    text_query = parsed_args["text_query"]
    text_match_mode = parsed_args["text_match_mode"]
    count_mode = parsed_args["count_mode"]
    sort_by = parsed_args["sort_by"]
    sort_direction = parsed_args["sort_direction"]
    page = parsed_args["page"]
    page_size = parsed_args["page_size"]
    include_xml = hydrate_xml

    if page < 1:
        page = 1
    max_page_size = 100 if ctx.is_authenticated else 10
    if page_size < 1 or page_size > max_page_size:
        page_size = min(25, max_page_size)

    # Build the ID-only query first so filters, sort order, and count estimation all share
    # the same search surface before we hydrate the selected rows.
    q = db.session.query(latest.section_uuid.label("section_uuid"))
    q = apply_section_text_join(
        q,
        latest=latest,
        section_text_search=section_text_search,
        text_query=text_query,
    )

    if years:
        year_filters = tuple(
            and_(
                latest.filing_date >= f"{year:04d}-01-01",
                latest.filing_date < f"{year + 1:04d}-01-01",
            )
            for year in years
        )
        q = q.filter(or_(*year_filters))

    if year_min is not None:
        q = q.filter(latest.filing_date >= f"{year_min:04d}-01-01")
    if year_max is not None:
        q = q.filter(latest.filing_date < f"{year_max + 1:04d}-01-01")
    if filed_after:
        q = q.filter(latest.filing_date >= filed_after)
    if filed_before:
        q = q.filter(latest.filing_date < filed_before)

    if targets:
        q = q.filter(latest.target.in_(targets))
    if acquirers:
        q = q.filter(latest.acquirer.in_(acquirers))
    transaction_price_total_filter = build_transaction_price_bucket_filter(
        latest.transaction_price_total,
        transaction_price_totals,
    )
    if transaction_price_total_filter is not None:
        q = q.filter(transaction_price_total_filter)
    transaction_price_stock_filter = build_transaction_price_bucket_filter(
        latest.transaction_price_stock,
        transaction_price_stocks,
    )
    if transaction_price_stock_filter is not None:
        q = q.filter(transaction_price_stock_filter)
    transaction_price_cash_filter = build_transaction_price_bucket_filter(
        latest.transaction_price_cash,
        transaction_price_cashes,
    )
    if transaction_price_cash_filter is not None:
        q = q.filter(transaction_price_cash_filter)
    transaction_price_assets_filter = build_transaction_price_bucket_filter(
        latest.transaction_price_assets,
        transaction_price_assets,
    )
    if transaction_price_assets_filter is not None:
        q = q.filter(transaction_price_assets_filter)

    standard_ids_expanded = False
    if standard_ids:
        standard_ids_key = tuple(sorted({value for value in standard_ids if value}))
        expanded_standard_ids = list(expand_taxonomy_cached(standard_ids_key))
        if expanded_standard_ids:
            standard_ids_expanded = set(expanded_standard_ids) != set(standard_ids_key)
            q = q.filter(standard_id_filter_expr(expanded_standard_ids))

    if target_types:
        q = q.filter(latest.target_type.in_(target_types))
    if transaction_considerations:
        q = q.filter(latest.transaction_consideration.in_(transaction_considerations))
    if acquirer_types:
        q = q.filter(latest.acquirer_type.in_(acquirer_types))
    target_counsel_subquery = build_canonical_counsel_agreement_uuid_subquery(
        side="target",
        canonical_names=target_counsels,
        agreement_counsel=agreement_counsel,
        counsel=counsel,
    )
    if target_counsel_subquery is not None:
        q = q.filter(latest.agreement_uuid.in_(target_counsel_subquery))
    acquirer_counsel_subquery = build_canonical_counsel_agreement_uuid_subquery(
        side="acquirer",
        canonical_names=acquirer_counsels,
        agreement_counsel=agreement_counsel,
        counsel=counsel,
    )
    if acquirer_counsel_subquery is not None:
        q = q.filter(latest.agreement_uuid.in_(acquirer_counsel_subquery))
    if target_industries:
        q = q.filter(latest.target_industry.in_(target_industries))
    if acquirer_industries:
        q = q.filter(latest.acquirer_industry.in_(acquirer_industries))
    if deal_statuses:
        q = q.filter(latest.deal_status.in_(deal_statuses))
    if attitudes:
        q = q.filter(latest.attitude.in_(attitudes))
    if deal_types:
        q = q.filter(latest.deal_type.in_(deal_types))
    if purposes:
        q = q.filter(latest.purpose.in_(purposes))

    if target_pes:
        db_target_pes: list[int] = []
        for pe in target_pes:
            if pe == "true":
                db_target_pes.append(1)
            elif pe == "false":
                db_target_pes.append(0)
        if db_target_pes:
            q = q.filter(latest.target_pe.in_(db_target_pes))

    if acquirer_pes:
        db_acquirer_pes: list[int] = []
        for pe in acquirer_pes:
            if pe == "true":
                db_acquirer_pes.append(1)
            elif pe == "false":
                db_acquirer_pes.append(0)
        if db_acquirer_pes:
            q = q.filter(latest.acquirer_pe.in_(db_acquirer_pes))

    if agreement_uuid and agreement_uuid.strip():
        q = q.filter(latest.agreement_uuid == agreement_uuid.strip())

    if section_uuid and section_uuid.strip():
        q = q.filter(latest.section_uuid == section_uuid.strip())

    text_active = bool(text_query is not None and text_query.strip())
    phrase_mode = text_active and text_match_mode == "phrase"

    # Filters are applied; branch the two phrase plans off this shared base.
    base_query = q
    if text_active:
        candidate_query = base_query.filter(
            build_section_text_candidate_expression(
                section_text_search.normalized_text,
                text_query=cast(str, text_query),
                match_mode=text_match_mode,
            )
        )
    else:
        candidate_query = base_query
    text_count_query = candidate_query

    date_scan_query: Any = None
    if phrase_mode:
        phrase_expression = cast(Any, section_text_search.normalized_text).op("REGEXP")(
            compile_phrase_text_pattern(cast(str, text_query))
        )
        # Both plans select the same rows: the REGEXP is the authoritative phrase
        # filter, and the FULLTEXT predicate only narrows candidates ahead of it.
        # Dropping it lets the optimizer drive from the sort index instead.
        q = candidate_query.filter(phrase_expression)
        date_scan_query = base_query.filter(phrase_expression)
    else:
        q = candidate_query

    descending = sort_direction == "desc"
    if sort_by == "year":
        primary_sort = latest.filing_date
    elif sort_by == "target":
        primary_sort = latest.target
    else:
        primary_sort = latest.acquirer

    def apply_sort(query: Any) -> Any:
        if descending:
            return query.order_by(desc(primary_sort), desc(latest.section_uuid))
        return query.order_by(asc(primary_sort), asc(latest.section_uuid))

    q = apply_sort(q)

    count_cache_key = build_search_count_cache_key("sections", parsed_args)
    offset = (page - 1) * page_size
    if phrase_mode and date_scan_query is not None:
        page_rows, q = _fetch_phrase_page(
            deps,
            fulltext_query=q,
            date_scan_query=apply_sort(date_scan_query),
            cache_key=count_cache_key,
            offset=offset,
            limit=page_size + 1,
        )
    else:
        page_rows = cast(list[object], q.offset(offset).limit(page_size + 1).all())
    has_next = len(page_rows) > page_size
    item_rows = page_rows[:page_size]
    item_count = len(item_rows)
    has_filters = any(
        (
            years,
            year_min is not None,
            year_max is not None,
            filed_after,
            filed_before,
            targets,
            acquirers,
            standard_ids,
            transaction_price_totals,
            transaction_price_stocks,
            transaction_price_cashes,
            transaction_price_assets,
            transaction_considerations,
            target_types,
            acquirer_types,
            target_counsels,
            acquirer_counsels,
            target_industries,
            acquirer_industries,
            deal_statuses,
            attitudes,
            deal_types,
            purposes,
            target_pes,
            acquirer_pes,
            agreement_uuid and agreement_uuid.strip(),
            section_uuid and section_uuid.strip(),
            text_query and text_query.strip(),
        )
    )
    total_agreement_count: int | None = None
    page_lower_bound = ((page - 1) * page_size) + item_count
    # With only a lower bound, still report more rows than this page when another
    # page exists, so total_pages cannot contradict has_next.
    unknown_total_floor = page_lower_bound + 1 if has_next else page_lower_bound
    if count_mode == "exact":
        counted, counted_exactly = _count_with_timeout_fallback(
            deps, query=q, cache_key=count_cache_key
        )
        if counted_exactly:
            total_count = cast(int, counted)
            total_agreement_count = deps._cached_exact_query_count(
                q.order_by(None).with_entities(distinct(latest.agreement_uuid)),
                cache_key=f"{count_cache_key}:agreements",
            )
            total_count_is_approximate = False
            count_method = "query_count"
        else:
            # An exact total was requested but is not affordable for this phrase.
            # Report the estimate honestly rather than failing the whole search.
            total_count = counted if counted is not None else unknown_total_floor
            total_count_is_approximate = True
            count_method = "table_estimate" if counted is not None else "filtered_lower_bound"
    elif text_query is not None and text_query.strip():
        if not has_next and (page == 1 or item_count > 0):
            total_count = page_lower_bound
            total_count_is_approximate = False
            count_method = "query_count"
        else:
            counted, counted_exactly = _count_with_timeout_fallback(
                deps,
                query=text_count_query,
                cache_key=f"{count_cache_key}:fulltext-candidates",
            )
            if counted is None:
                total_count = unknown_total_floor
                total_count_is_approximate = True
                count_method = "filtered_lower_bound"
            else:
                total_count = counted
                total_count_is_approximate = (
                    text_match_mode == "phrase" or not counted_exactly
                )
                if not counted_exactly:
                    count_method = "table_estimate"
                elif total_count_is_approximate:
                    count_method = "fulltext_candidate_count"
                else:
                    count_method = "query_count"
    else:
        total_count, total_count_is_approximate, count_method = sections_total_count_metadata(
            deps,
            query=q,
            page=page,
            page_size=page_size,
            item_count=item_count,
            has_next=has_next,
            has_filters=has_filters,
            count_mode=count_mode,
            count_cache_key=count_cache_key,
        )

    section_uuids = [
        section_id
        for item_row in item_rows
        for section_id in [row_mapping_as_dict(item_row).get("section_uuid")]
        if isinstance(section_id, str)
    ]

    metadata_column_by_field = {
        "filing_date": latest.filing_date,
        "prob_filing": latest.prob_filing,
        "filing_company_name": latest.filing_company_name,
        "filing_company_cik": latest.filing_company_cik,
        "form_type": latest.form_type,
        "exhibit_type": latest.exhibit_type,
        "transaction_price_total": latest.transaction_price_total,
        "transaction_price_stock": latest.transaction_price_stock,
        "transaction_price_cash": latest.transaction_price_cash,
        "transaction_price_assets": latest.transaction_price_assets,
        "transaction_consideration": latest.transaction_consideration,
        "target_type": latest.target_type,
        "acquirer_type": latest.acquirer_type,
        "target_industry": latest.target_industry,
        "acquirer_industry": latest.acquirer_industry,
        "announce_date": latest.announce_date,
        "close_date": latest.close_date,
        "deal_status": latest.deal_status,
        "attitude": latest.attitude,
        "deal_type": latest.deal_type,
        "purpose": latest.purpose,
        "target_pe": latest.target_pe,
        "acquirer_pe": latest.acquirer_pe,
        "url": latest.url,
    }

    details_by_uuid: dict[str, dict[str, object]] = {}
    if section_uuids:
        detail_columns = [
            latest.section_uuid.label("section_uuid"),
            latest.agreement_uuid.label("agreement_uuid"),
            latest.section_standard_ids.label("section_standard_ids"),
            latest.article_title.label("article_title"),
            latest.section_title.label("section_title"),
            latest.acquirer.label("acquirer"),
            latest.target.label("target"),
            latest.filing_date.label("filing_date"),
            latest.transaction_price_total.label("transaction_price_total"),
            latest.verified.label("verified"),
        ]
        if include_xml:
            detail_columns.append(sections.xml_content.label("xml_content"))
        for field_name in requested_metadata_fields:
            detail_columns.append(
                metadata_column_by_field[field_name].label(field_name)
            )
        section_rows = cast(
            list[object],
            db.session.query(*detail_columns)
            .select_from(sections)
            .join(
                latest,
                sections.section_uuid == latest.section_uuid,
            )
            .filter(sections.section_uuid.in_(section_uuids))
            .all(),
        )
        for row in section_rows:
            row_map = row_mapping_as_dict(row)
            row_section_uuid = row_map.get("section_uuid")
            if isinstance(row_section_uuid, str):
                details_by_uuid[row_section_uuid] = row_map

    meta = pagination_metadata(
        total_count=total_count,
        page=page,
        page_size=page_size,
        has_next_override=has_next,
        total_count_is_approximate=total_count_is_approximate,
    )

    results: list[dict[str, object]] = []
    for section_uuid_value in section_uuids:
        detail_row = details_by_uuid.get(section_uuid_value)
        if detail_row is None:
            raise RuntimeError(
                f"Section UUID {section_uuid_value} missing from detail lookup."
            )
        result_payload = {
            "id": section_uuid_value,
            "agreement_uuid": detail_row.get("agreement_uuid"),
            "section_uuid": section_uuid_value,
            "standard_id": parse_standard_ids(
                detail_row.get("section_standard_ids")
            ),
            "article_title": detail_row.get("article_title"),
            "section_title": detail_row.get("section_title"),
            "acquirer": detail_row.get("acquirer"),
            "target": detail_row.get("target"),
            "filing_date": detail_row.get("filing_date"),
            "transaction_price_total": detail_row.get("transaction_price_total"),
            "year": year_from_filing_date(detail_row.get("filing_date")),
            "verified": (
                bool(detail_row.get("verified"))
                if detail_row.get("verified") is not None
                else False
            ),
        }
        if include_xml:
            result_payload["xml"] = detail_row.get("xml_content")
        if requested_metadata_fields:
            result_payload["metadata"] = {
                field_name: detail_row.get(field_name)
                for field_name in requested_metadata_fields
            }
        results.append(result_payload)

    unique_agreement_count = len({
        str(r["agreement_uuid"])
        for r in results
        if r.get("agreement_uuid") is not None
    })
    response: dict[str, object] = {
        "results": results,
        "unique_agreement_count": unique_agreement_count,
        "access": {
            "tier": ctx.tier,
            "message": None
            if ctx.is_authenticated
            else "Limited mode: sign in to unlock full pagination and use the MCP server.",
        },
        "count_metadata": _sections_count_metadata_payload(
            total_count_is_approximate=total_count_is_approximate,
            count_method=count_method,
            exact_count_requested=count_mode == "exact",
        ),
        "interpretation": _sections_interpretation_payload(
            parsed_args=parsed_args,
            standard_ids_expanded=standard_ids_expanded,
            total_count_is_approximate=total_count_is_approximate,
            count_method=count_method,
        ),
        **meta,
    }
    if total_agreement_count is not None:
        response["total_agreement_count"] = total_agreement_count
    return response


__all__ = [
    "SECTION_TEXT_SEARCH_UNAVAILABLE_MESSAGE",
    "TEXT_PLAN_DATE_SCAN",
    "TEXT_PLAN_FULLTEXT",
    "TEXT_QUERY_TOO_EXPENSIVE_MESSAGE",
    "apply_section_text_join",
    "build_section_text_candidate_expression",
    "build_section_text_match_expression",
    "estimated_latest_sections_search_table_rows",
    "estimated_query_row_count",
    "is_statement_timeout_error",
    "reset_text_plan_cache",
    "run_sections",
    "sections_total_count_metadata",
]
