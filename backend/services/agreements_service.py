"""Query orchestration for the agreements domain.

Routes in ``backend/routes/agreements`` and the MCP handlers both build their
payloads here, so the agreement metadata projection, the search plans, and the
summary/trend payloads exist exactly once.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Any, cast

from marshmallow import ValidationError
from sqlalchemy import and_, asc, bindparam, desc, false, func, text
from sqlalchemy.exc import DatabaseError, SQLAlchemyError

from backend.core.pagination import (
    AUTHENTICATED_MAX_PAGE_SIZE,
    clamp_page_size,
    search_max_page_size,
)
from backend.filtering import agreement_is_public_eligible_expr, apply_agreement_filters
from backend.routes.deps import AccessContextProtocol, AgreementsDeps
from backend.schemas.public_api import AgreementsBulkArgsPayload
from backend.schemas.sections import SectionsArgsPayload
from backend.search_counts import build_search_count_cache_key
from backend.services import taxonomy_service
from backend.services.counsel_leaderboards import build_counsel_leaderboards_from_summary_rows
from backend.services.sections_service import (
    SECTION_TEXT_SEARCH_UNAVAILABLE_MESSAGE,
    TEXT_QUERY_TOO_EXPENSIVE_MESSAGE,
    budgeted_exact_count,
    build_section_text_candidate_expression,
    fetch_with_text_plan_race,
    is_text_plan_exhausted_error,
)
from backend.services.reference_labels import industry_labels_for
from backend.services.snippets import (
    extract_text_from_xml,
    strip_xml_snippet,
    text_match_patterns,
    text_match_snippet,
)
from backend.summary_specs import METADATA_FIELD_COVERAGE_CONFIG
from backend.text_search import TEXT_SEARCH_MAX_STATEMENT_SECONDS, compile_phrase_text_pattern


# ---------------------------------------------------------------------------
# Agreement metadata projection
# ---------------------------------------------------------------------------

# The agreement metadata row every agreement-level payload (REST and MCP) returns.
# ``year`` is derived from ``filing_date``; every other name is an ``agreements`` column.
AGREEMENT_METADATA_COLUMNS: tuple[str, ...] = (
    "agreement_uuid",
    "year",
    "target",
    "acquirer",
    "filing_date",
    "prob_filing",
    "filing_company_name",
    "filing_company_cik",
    "form_type",
    "exhibit_type",
    "transaction_price_total",
    "transaction_price_stock",
    "transaction_price_cash",
    "transaction_price_assets",
    "transaction_consideration",
    "target_type",
    "acquirer_type",
    "target_industry",
    "acquirer_industry",
    "announce_date",
    "close_date",
    "deal_status",
    "attitude",
    "deal_type",
    "purpose",
    "target_pe",
    "acquirer_pe",
    "url",
)

PayloadBuilder = Callable[[AgreementsDeps], dict[str, object]]

# How many matched sections an agreement search result previews.
MATCHED_SECTION_PREVIEW_LIMIT = 3
_MATCHED_SECTION_SNIPPET_CHARS = 220


def agreement_metadata_columns(deps: AgreementsDeps) -> list[Any]:
    """Labeled select columns for ``AGREEMENT_METADATA_COLUMNS``."""
    agreements = deps.Agreements
    columns: list[Any] = []
    for name in AGREEMENT_METADATA_COLUMNS:
        if name == "year":
            columns.append(deps._agreement_year_expr().label("year"))
        else:
            columns.append(getattr(agreements, name).label(name))
    return columns


def row_to_agreement_metadata(
    row_map: Mapping[str, object],
    *,
    convert: Callable[[object], object] | None = None,
) -> dict[str, object]:
    """Project a row selected with ``agreement_metadata_columns`` into a payload dict.

    ``convert`` maps each value (MCP passes its JSON coercion); REST passes nothing and
    lets the response schema serialize.
    """
    if convert is None:
        return {name: row_map.get(name) for name in AGREEMENT_METADATA_COLUMNS}
    return {name: convert(row_map.get(name)) for name in AGREEMENT_METADATA_COLUMNS}


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def to_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float, Decimal, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_industry_label(raw_value: object, *, label_by_code: dict[str, str]) -> str:
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return ""
    return label_by_code.get(raw_text, raw_text)


def _iso_date_or_none(value: object) -> object:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if value is not None:
        return str(value)
    return None


def _search_access(ctx: AccessContextProtocol) -> dict[str, object]:
    return {
        "tier": ctx.tier,
        "message": None
        if ctx.is_authenticated
        else "Sign in to view full agreement text and unlock higher page sizes.",
    }


def _count_metadata(*, approximate: bool, method: str, exact_requested: bool) -> dict[str, object]:
    return {
        "mode": "estimated" if approximate else "exact",
        "method": method,
        "planning_reliability": "medium" if approximate else "high",
        "exact_count_requested": exact_requested,
    }


def _args_mapping(parsed_args: object) -> dict[str, object]:
    return cast(dict[str, object], parsed_args)


# ---------------------------------------------------------------------------
# GET /v1/agreements (keyset listing)
# ---------------------------------------------------------------------------


def list_agreements(
    deps: AgreementsDeps,
    *,
    ctx: AccessContextProtocol,
    parsed_args: AgreementsBulkArgsPayload,
) -> dict[str, object]:
    include_xml = parsed_args["include_xml"]
    page_size = clamp_page_size(parsed_args["page_size"], max_page_size=AUTHENTICATED_MAX_PAGE_SIZE)
    after_agreement_uuid = deps._decode_agreements_cursor(parsed_args["cursor"])

    agreements = deps.Agreements
    xml = deps.XML
    sections = deps.Sections
    db = deps.db
    q = (
        db.session.query(*agreement_metadata_columns(deps))
        .join(xml, deps._agreement_latest_xml_join_condition())
        .filter(agreement_is_public_eligible_expr(agreements))
    )
    if include_xml:
        q = q.add_columns(xml.xml.label("xml"))
    q = apply_agreement_filters(
        q,
        agreements,
        _args_mapping(parsed_args),
        agreement_counsel=deps.AgreementCounsel,
        counsel=deps.Counsel,
    )

    section_uuid = parsed_args["section_uuid"]
    if section_uuid and section_uuid.strip():
        section_exists = (
            db.session.query(sections.section_uuid)
            .filter(
                sections.agreement_uuid == agreements.agreement_uuid,
                sections.section_uuid == section_uuid.strip(),
                sections.xml_version == xml.version,
            )
            .exists()
        )
        q = q.filter(section_exists)

    if after_agreement_uuid:
        q = q.filter(agreements.agreement_uuid > after_agreement_uuid)

    rows = cast(
        list[object],
        q.order_by(asc(agreements.agreement_uuid)).limit(page_size + 1).all(),
    )
    has_next = len(rows) > page_size
    page_rows = rows[:page_size]

    results: list[dict[str, object]] = []
    for row in page_rows:
        row_map = deps._row_mapping_as_dict(row)
        payload = row_to_agreement_metadata(row_map)
        if include_xml:
            payload["xml"] = row_map.get("xml")
        results.append(payload)

    next_cursor: str | None = None
    if has_next:
        last_agreement_uuid = deps._row_mapping_as_dict(page_rows[-1]).get("agreement_uuid")
        if not isinstance(last_agreement_uuid, str) or not last_agreement_uuid:
            raise RuntimeError("Agreements list query returned a row without agreement_uuid.")
        next_cursor = deps._encode_agreements_cursor(last_agreement_uuid)

    return {
        "results": results,
        "access": {
            "tier": ctx.tier,
            "message": None
            if ctx.is_authenticated
            else "XML access requires authentication. Use include_xml=true with a signed-in user or API key.",
        },
        "page_size": page_size,
        "returned_count": len(results),
        "has_next": has_next,
        "next_cursor": next_cursor,
    }


# ---------------------------------------------------------------------------
# GET /v1/search/agreements
# ---------------------------------------------------------------------------

METADATA_NOT_SUPPORTED_MESSAGE = (
    "metadata is not supported on /v1/search/agreements: every result already carries "
    "the full agreement metadata row."
)


def search_agreements(
    deps: AgreementsDeps,
    *,
    ctx: AccessContextProtocol,
    parsed_args: SectionsArgsPayload,
) -> dict[str, object]:
    """One row per agreement, optionally scoped by section-level criteria.

    Three plans, chosen by which section-level criteria are present:

    - none: agreement metadata filters only, read from ``agreements``;
    - ``standard_id`` / ``section_uuid``: agreements with at least one latest section
      matching them, with up to three matched-section previews;
    - ``text_query`` (alone or combined with the above): agreements with at least one
      latest section whose text matches, with previews focused on the match.
    """
    if parsed_args["metadata"]:
        raise ValidationError({"metadata": [METADATA_NOT_SUPPORTED_MESSAGE]})

    page = max(parsed_args["page"], 1)
    page_size = clamp_page_size(
        parsed_args["page_size"],
        max_page_size=search_max_page_size(is_authenticated=ctx.is_authenticated),
    )
    text_query = parsed_args["text_query"]
    if text_query is not None and text_query.strip():
        try:
            return _search_agreements_by_text(
                deps, ctx=ctx, parsed_args=parsed_args, page=page, page_size=page_size
            )
        except DatabaseError as exc:
            if not is_text_plan_exhausted_error(exc):
                raise
            deps.db.session.rollback()
            raise ValidationError({"text_query": [TEXT_QUERY_TOO_EXPENSIVE_MESSAGE]}) from exc

    section_uuid = parsed_args["section_uuid"]
    if parsed_args["standard_id"] or (section_uuid and section_uuid.strip()):
        return _search_agreements_by_clause(
            deps, ctx=ctx, parsed_args=parsed_args, page=page, page_size=page_size
        )
    return _search_agreements_direct(
        deps, ctx=ctx, parsed_args=parsed_args, page=page, page_size=page_size
    )


def _agreement_sort_column(deps: AgreementsDeps, sort_by: str) -> Any:
    agreements = deps.Agreements
    if sort_by == "year":
        return agreements.filing_date
    if sort_by == "target":
        return agreements.target
    return agreements.acquirer


def _ordered(query: Any, *, primary: Any, tiebreak: Any, descending: bool) -> Any:
    if descending:
        return query.order_by(desc(primary), desc(tiebreak))
    return query.order_by(asc(primary), asc(tiebreak))


def _search_agreements_direct(
    deps: AgreementsDeps,
    *,
    ctx: AccessContextProtocol,
    parsed_args: SectionsArgsPayload,
    page: int,
    page_size: int,
) -> dict[str, object]:
    agreements = deps.Agreements
    direct_query = (
        deps.db.session.query(*agreement_metadata_columns(deps))
        .join(deps.XML, deps._agreement_latest_xml_join_condition())
        .filter(agreement_is_public_eligible_expr(agreements))
    )
    direct_query = apply_agreement_filters(
        direct_query,
        agreements,
        _args_mapping(parsed_args),
        agreement_counsel=deps.AgreementCounsel,
        counsel=deps.Counsel,
    )
    direct_query = _ordered(
        direct_query,
        primary=_agreement_sort_column(deps, parsed_args["sort_by"]),
        tiebreak=agreements.agreement_uuid,
        descending=parsed_args["sort_direction"] == "desc",
    )

    page_rows_raw = cast(
        list[object],
        direct_query.offset((page - 1) * page_size).limit(page_size + 1).all(),
    )
    has_next = len(page_rows_raw) > page_size
    page_rows = page_rows_raw[:page_size]
    count_mode = parsed_args["count_mode"]
    count_cache_key = (
        None
        if count_mode == "exact"
        else build_search_count_cache_key("agreements_search:direct", parsed_args)
    )
    total_count = deps._cached_exact_query_count(direct_query, cache_key=count_cache_key)
    meta = deps._pagination_metadata(
        total_count=total_count,
        page=page,
        page_size=page_size,
        has_next_override=has_next,
        total_count_is_approximate=False,
    )
    results: list[dict[str, object]] = []
    for row in page_rows:
        payload = row_to_agreement_metadata(deps._row_mapping_as_dict(row))
        payload["match_count"] = 0
        payload["matched_sections"] = []
        results.append(payload)
    return {
        "results": results,
        "access": _search_access(ctx),
        "count_metadata": _count_metadata(
            approximate=False, method="query_count", exact_requested=count_mode == "exact"
        ),
        **meta,
    }


def _expanded_standard_ids(deps: AgreementsDeps, standard_ids: list[str]) -> list[str]:
    if not standard_ids:
        return []
    standard_ids_key = tuple(sorted({value for value in standard_ids if value}))
    return list(deps._expand_taxonomy_standard_ids_cached(standard_ids_key))


def _build_agreement_search_match_query(
    deps: AgreementsDeps,
    *,
    parsed_args: SectionsArgsPayload,
) -> Any:
    """Latest sections that satisfy the section-level and agreement-level filters."""
    latest = deps.LatestSectionsSearch
    agreements = deps.Agreements
    q = (
        deps.db.session.query(
            latest.section_uuid.label("section_uuid"),
            latest.agreement_uuid.label("agreement_uuid"),
        )
        .join(agreements, agreements.agreement_uuid == latest.agreement_uuid)
        .filter(agreement_is_public_eligible_expr(agreements))
    )
    q = apply_agreement_filters(
        q,
        latest,
        _args_mapping(parsed_args),
        agreement_counsel=deps.AgreementCounsel,
        counsel=deps.Counsel,
    )
    expanded = _expanded_standard_ids(deps, parsed_args["standard_id"])
    if expanded:
        q = q.filter(deps._standard_id_filter_expr(expanded))
    section_uuid = parsed_args["section_uuid"]
    if section_uuid and section_uuid.strip():
        q = q.filter(latest.section_uuid == section_uuid.strip())
    return q


def _search_agreements_by_clause(
    deps: AgreementsDeps,
    *,
    ctx: AccessContextProtocol,
    parsed_args: SectionsArgsPayload,
    page: int,
    page_size: int,
) -> dict[str, object]:
    latest = deps.LatestSectionsSearch
    sections = deps.Sections
    db = deps.db
    match_query = _build_agreement_search_match_query(deps, parsed_args=parsed_args)
    aggregated_query = match_query.with_entities(
        latest.agreement_uuid.label("agreement_uuid"),
        func.count().label("match_count"),
        func.min(latest.filing_date).label("sort_filing_date"),
        func.min(latest.target).label("sort_target"),
        func.min(latest.acquirer).label("sort_acquirer"),
    ).group_by(latest.agreement_uuid)

    sort_by = parsed_args["sort_by"]
    sort_label = {"year": "sort_filing_date", "target": "sort_target"}.get(sort_by, "sort_acquirer")
    aggregated_query = _ordered(
        aggregated_query,
        primary=text(sort_label),
        tiebreak=text("agreement_uuid"),
        descending=parsed_args["sort_direction"] == "desc",
    )

    page_rows_raw = cast(
        list[object],
        aggregated_query.offset((page - 1) * page_size).limit(page_size + 1).all(),
    )
    has_next = len(page_rows_raw) > page_size
    page_rows = page_rows_raw[:page_size]
    count_mode = parsed_args["count_mode"]
    count_cache_key = (
        None
        if count_mode == "exact"
        else build_search_count_cache_key("agreements_search:clause", parsed_args)
    )
    total_count = deps._cached_exact_query_count(aggregated_query, cache_key=count_cache_key)
    meta = deps._pagination_metadata(
        total_count=total_count,
        page=page,
        page_size=page_size,
        has_next_override=has_next,
        total_count_is_approximate=False,
    )

    agreement_ids: list[str] = []
    match_count_by_agreement: dict[str, int] = {}
    for row in page_rows:
        row_map = deps._row_mapping_as_dict(row)
        agreement_id = row_map.get("agreement_uuid")
        if not isinstance(agreement_id, str):
            continue
        agreement_ids.append(agreement_id)
        match_count_by_agreement[agreement_id] = deps._to_int(row_map.get("match_count"))

    matched_sections_by_agreement: dict[str, list[dict[str, object]]] = defaultdict(list)
    if agreement_ids:
        preview_rank = func.row_number().over(
            partition_by=latest.agreement_uuid,
            order_by=[
                asc(sections.article_order),
                asc(sections.section_order),
                asc(latest.section_uuid),
            ],
        ).label("preview_rank")
        preview_subquery = (
            match_query.join(sections, sections.section_uuid == latest.section_uuid)
            .with_entities(
                latest.agreement_uuid.label("agreement_uuid"),
                latest.section_uuid.label("section_uuid"),
                latest.article_title.label("article_title"),
                latest.section_title.label("section_title"),
                latest.section_standard_ids.label("section_standard_ids"),
                sections.xml_content.label("xml_content"),
                sections.article_order.label("article_order"),
                sections.section_order.label("section_order"),
                preview_rank,
            )
            .filter(latest.agreement_uuid.in_(agreement_ids))
            .subquery()
        )
        preview_rows = cast(
            list[object],
            db.session.query(preview_subquery)
            .filter(preview_subquery.c.preview_rank <= MATCHED_SECTION_PREVIEW_LIMIT)
            .order_by(
                asc(preview_subquery.c.agreement_uuid),
                asc(preview_subquery.c.article_order),
                asc(preview_subquery.c.section_order),
                asc(preview_subquery.c.section_uuid),
            )
            .all(),
        )
        for row in preview_rows:
            row_map = deps._row_mapping_as_dict(row)
            agreement_id = row_map.get("agreement_uuid")
            if not isinstance(agreement_id, str):
                continue
            matched_sections_by_agreement[agreement_id].append(
                {
                    "section_uuid": row_map.get("section_uuid"),
                    "article_title": row_map.get("article_title"),
                    "section_title": row_map.get("section_title"),
                    "standard_id": deps._parse_section_standard_ids(
                        row_map.get("section_standard_ids")
                    ),
                    "snippet": strip_xml_snippet(row_map.get("xml_content")),
                }
            )

    results = _agreement_results_for_ids(
        deps,
        agreement_ids=agreement_ids,
        match_count_by_agreement=match_count_by_agreement,
        matched_sections_by_agreement=matched_sections_by_agreement,
    )
    return {
        "results": results,
        "access": _search_access(ctx),
        "count_metadata": _count_metadata(
            approximate=False, method="query_count", exact_requested=count_mode == "exact"
        ),
        **meta,
    }


def _agreement_results_for_ids(
    deps: AgreementsDeps,
    *,
    agreement_ids: list[str],
    match_count_by_agreement: Mapping[str, int],
    matched_sections_by_agreement: Mapping[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    """Hydrate a page of agreement ids into result rows, preserving page order."""
    if not agreement_ids:
        return []
    agreements = deps.Agreements
    detail_rows = cast(
        list[object],
        deps.db.session.query(*agreement_metadata_columns(deps))
        .filter(agreements.agreement_uuid.in_(agreement_ids))
        .all(),
    )
    details_by_id: dict[str, dict[str, object]] = {}
    for row in detail_rows:
        row_map = deps._row_mapping_as_dict(row)
        agreement_id = row_map.get("agreement_uuid")
        if isinstance(agreement_id, str):
            details_by_id[agreement_id] = row_map

    results: list[dict[str, object]] = []
    for agreement_id in agreement_ids:
        detail_row = details_by_id.get(agreement_id)
        if detail_row is None:
            continue
        payload = row_to_agreement_metadata(detail_row)
        payload["match_count"] = match_count_by_agreement.get(agreement_id, 0)
        payload["matched_sections"] = list(
            matched_sections_by_agreement.get(agreement_id, [])
        )[:MATCHED_SECTION_PREVIEW_LIMIT]
        results.append(payload)
    return results


# --- text_query plan --------------------------------------------------------
#
# An agreement matches when at least one of its latest sections matches the text
# (and the standard_id / section_uuid criteria, when given). Section text is
# matched exactly as /v1/sections matches it.
#
# Like phrase search on sections, two plans select the same agreements:
#
# - FULLTEXT candidates: the indexed MATCH narrows sections first, matching
#   sections are grouped per agreement, and agreements are sorted. Fast for rare
#   text, unusable when the words are common (the index intersection alone can run
#   for a minute on "material adverse effect").
# - sort-ordered scan (phrase only): walk agreements in sort order and probe each
#   one's sections with the phrase REGEXP, stopping at the first hit. Fast exactly
#   when the phrase is common, a full scan when it is rare.
#
# The first request for a filter signature races them under a short budget and
# remembers the winner (see sections_service.fetch_with_text_plan_race).


def _section_level_predicates(
    deps: AgreementsDeps, *, parsed_args: SectionsArgsPayload
) -> list[Any]:
    latest = deps.LatestSectionsSearch
    predicates: list[Any] = []
    expanded = _expanded_standard_ids(deps, parsed_args["standard_id"])
    if expanded:
        predicates.append(deps._standard_id_filter_expr(expanded))
    section_uuid = parsed_args["section_uuid"]
    if section_uuid and section_uuid.strip():
        predicates.append(latest.section_uuid == section_uuid.strip())
    return predicates


def _text_section_query(
    deps: AgreementsDeps,
    *,
    columns: list[Any],
    section_predicates: list[Any],
    text_predicates: list[Any],
) -> Any:
    """Latest sections joined to their text index, filtered to the matching ones."""
    section_text_search = deps.SectionTextSearch
    latest = deps.LatestSectionsSearch
    query = (
        deps.db.session.query(*columns)
        .select_from(section_text_search)
        .join(latest, latest.section_uuid == section_text_search.section_uuid)
    )
    for predicate in (*section_predicates, *text_predicates):
        query = query.filter(predicate)
    return query


def _agreement_level_query(
    deps: AgreementsDeps, *, columns: list[Any], parsed_args: SectionsArgsPayload
) -> Any:
    agreements = deps.Agreements
    query = deps.db.session.query(*columns).filter(agreement_is_public_eligible_expr(agreements))
    return apply_agreement_filters(
        query,
        agreements,
        _args_mapping(parsed_args),
        agreement_counsel=deps.AgreementCounsel,
        counsel=deps.Counsel,
    )


def _search_agreements_by_text(
    deps: AgreementsDeps,
    *,
    ctx: AccessContextProtocol,
    parsed_args: SectionsArgsPayload,
    page: int,
    page_size: int,
) -> dict[str, object]:
    section_text_search = deps.SectionTextSearch
    if section_text_search is None:
        raise ValidationError({"text_query": [SECTION_TEXT_SEARCH_UNAVAILABLE_MESSAGE]})
    agreements = deps.Agreements
    text_query = cast(str, parsed_args["text_query"]).strip()
    match_mode = parsed_args["text_match_mode"]
    phrase_mode = match_mode == "phrase"

    candidate_predicate = build_section_text_candidate_expression(
        section_text_search.normalized_text,
        text_query=text_query,
        match_mode=match_mode,
    )
    phrase_predicate: Any = None
    if phrase_mode:
        phrase_predicate = section_text_search.normalized_text.op("REGEXP")(
            compile_phrase_text_pattern(text_query)
        )
    section_predicates = _section_level_predicates(deps, parsed_args=parsed_args)
    fulltext_text_predicates = [candidate_predicate] + (
        [phrase_predicate] if phrase_predicate is not None else []
    )

    sort_column = _agreement_sort_column(deps, parsed_args["sort_by"])
    descending = parsed_args["sort_direction"] == "desc"

    # FULLTEXT plan: matching sections grouped per agreement, then agreement filters.
    matches = (
        _text_section_query(
            deps,
            columns=[section_text_search.agreement_uuid.label("agreement_uuid")],
            section_predicates=section_predicates,
            text_predicates=fulltext_text_predicates,
        )
        .group_by(section_text_search.agreement_uuid)
        .subquery()
    )
    fulltext_query = _ordered(
        _agreement_level_query(
            deps, columns=[agreements.agreement_uuid.label("agreement_uuid")], parsed_args=parsed_args
        ).join(matches, matches.c.agreement_uuid == agreements.agreement_uuid),
        primary=sort_column,
        tiebreak=agreements.agreement_uuid,
        descending=descending,
    ).execution_options(max_statement_time=TEXT_SEARCH_MAX_STATEMENT_SECONDS)

    plan_cache_key = build_search_count_cache_key("agreements_search:text", parsed_args)
    offset = (page - 1) * page_size
    if phrase_mode:
        # Sort-ordered plan: a scalar `(SELECT ... LIMIT 1) IS NOT NULL` probe. An
        # EXISTS here is rewritten into a semi-join that materializes every matching
        # section before the first agreement is read, which is the full scan this plan
        # exists to avoid.
        first_match = (
            _text_section_query(
                deps,
                columns=[section_text_search.section_uuid],
                section_predicates=section_predicates,
                text_predicates=[phrase_predicate],
            )
            .filter(section_text_search.agreement_uuid == agreements.agreement_uuid)
            .limit(1)
            .correlate(agreements)
            .scalar_subquery()
        )
        scan_query = _ordered(
            _agreement_level_query(
                deps, columns=[agreements.agreement_uuid.label("agreement_uuid")], parsed_args=parsed_args
            ).filter(first_match.isnot(None)),
            primary=sort_column,
            tiebreak=agreements.agreement_uuid,
            descending=descending,
        ).execution_options(max_statement_time=TEXT_SEARCH_MAX_STATEMENT_SECONDS)
        page_rows_raw, _ = fetch_with_text_plan_race(
            deps.db,
            fulltext_query=fulltext_query,
            date_scan_query=scan_query,
            cache_key=plan_cache_key,
            offset=offset,
            limit=page_size + 1,
        )
    else:
        page_rows_raw = cast(list[object], fulltext_query.offset(offset).limit(page_size + 1).all())

    has_next = len(page_rows_raw) > page_size
    page_rows = page_rows_raw[:page_size]
    agreement_ids = [
        agreement_id
        for row in page_rows
        for agreement_id in [deps._row_mapping_as_dict(row).get("agreement_uuid")]
        if isinstance(agreement_id, str)
    ]

    count_mode = parsed_args["count_mode"]
    item_count = len(agreement_ids)
    page_lower_bound = offset + item_count
    if not has_next and (page == 1 or item_count > 0):
        # The whole result set ends on this page, so its size is known exactly.
        total_count = page_lower_bound
        approximate = False
        count_method = "query_count"
    else:
        # Counting always uses the FULLTEXT shape (see sections_service: the scan
        # plan has no candidate predicate, so counting it would REGEXP the corpus).
        counted = budgeted_exact_count(
            deps.db,
            cached_exact_query_count=deps._cached_exact_query_count,
            query=fulltext_query,
            cache_key=None if count_mode == "exact" else f"{plan_cache_key}:count",
        )
        if counted is not None:
            total_count = counted
            approximate = False
            count_method = "query_count"
        else:
            # Never show fewer pages than has_next implies.
            total_count = page_lower_bound + 1 if has_next else page_lower_bound
            approximate = True
            count_method = "filtered_lower_bound"

    match_patterns = text_match_patterns(text_query, match_mode)
    match_count_by_agreement, matched_sections_by_agreement = _text_matched_sections(
        deps,
        agreement_ids=agreement_ids,
        section_predicates=section_predicates,
        # Previews and counts run per page of agreements. For a phrase, the REGEXP
        # alone is exact and avoids re-running the FULLTEXT intersection.
        text_predicates=[phrase_predicate] if phrase_mode else [candidate_predicate],
        match_patterns=match_patterns,
    )
    results = _agreement_results_for_ids(
        deps,
        agreement_ids=agreement_ids,
        match_count_by_agreement=match_count_by_agreement,
        matched_sections_by_agreement=matched_sections_by_agreement,
    )
    meta = deps._pagination_metadata(
        total_count=total_count,
        page=page,
        page_size=page_size,
        has_next_override=has_next,
        total_count_is_approximate=approximate,
    )
    return {
        "results": results,
        "access": _search_access(ctx),
        "count_metadata": _count_metadata(
            approximate=approximate,
            method=count_method,
            exact_requested=count_mode == "exact",
        ),
        **meta,
    }


def _text_matched_sections(
    deps: AgreementsDeps,
    *,
    agreement_ids: list[str],
    section_predicates: list[Any],
    text_predicates: list[Any],
    match_patterns: list[Any],
) -> tuple[dict[str, int], dict[str, list[dict[str, object]]]]:
    """Per-agreement match counts and the first matches in document order."""
    match_counts: dict[str, int] = {}
    previews: dict[str, list[dict[str, object]]] = defaultdict(list)
    if not agreement_ids:
        return match_counts, previews
    latest = deps.LatestSectionsSearch
    sections = deps.Sections
    section_text_search = deps.SectionTextSearch
    rows = cast(
        list[object],
        _text_section_query(
            deps,
            columns=[
                latest.agreement_uuid.label("agreement_uuid"),
                latest.section_uuid.label("section_uuid"),
                sections.article_order.label("article_order"),
                sections.section_order.label("section_order"),
            ],
            section_predicates=section_predicates,
            text_predicates=text_predicates,
        )
        .join(sections, sections.section_uuid == latest.section_uuid)
        .filter(section_text_search.agreement_uuid.in_(agreement_ids))
        .execution_options(max_statement_time=TEXT_SEARCH_MAX_STATEMENT_SECONDS)
        .all(),
    )
    ordered_by_agreement: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for row in rows:
        row_map = deps._row_mapping_as_dict(row)
        agreement_id = row_map.get("agreement_uuid")
        section_uuid = row_map.get("section_uuid")
        if not isinstance(agreement_id, str) or not isinstance(section_uuid, str):
            continue
        ordered_by_agreement[agreement_id].append(
            (
                deps._to_int(row_map.get("article_order")),
                deps._to_int(row_map.get("section_order")),
                section_uuid,
            )
        )
    preview_uuids: list[str] = []
    for agreement_id, entries in ordered_by_agreement.items():
        match_counts[agreement_id] = len(entries)
        preview_uuids.extend(
            section_uuid for _a, _s, section_uuid in sorted(entries)[:MATCHED_SECTION_PREVIEW_LIMIT]
        )
    if not preview_uuids:
        return match_counts, previews

    detail_rows = cast(
        list[object],
        deps.db.session.query(
            latest.agreement_uuid.label("agreement_uuid"),
            latest.section_uuid.label("section_uuid"),
            latest.article_title.label("article_title"),
            latest.section_title.label("section_title"),
            latest.section_standard_ids.label("section_standard_ids"),
            sections.xml_content.label("xml_content"),
        )
        .join(sections, sections.section_uuid == latest.section_uuid)
        .filter(latest.section_uuid.in_(preview_uuids))
        .all(),
    )
    details_by_section: dict[str, dict[str, object]] = {}
    for row in detail_rows:
        row_map = deps._row_mapping_as_dict(row)
        section_uuid = row_map.get("section_uuid")
        if isinstance(section_uuid, str):
            details_by_section[section_uuid] = row_map
    for section_uuid in preview_uuids:
        detail = details_by_section.get(section_uuid)
        if detail is None:
            continue
        agreement_id = cast(str, detail.get("agreement_uuid"))
        snippet, matched_terms = text_match_snippet(
            extract_text_from_xml(detail.get("xml_content")),
            patterns=match_patterns,
            max_chars=_MATCHED_SECTION_SNIPPET_CHARS,
        )
        previews[agreement_id].append(
            {
                "section_uuid": section_uuid,
                "article_title": detail.get("article_title"),
                "section_title": detail.get("section_title"),
                "standard_id": deps._parse_section_standard_ids(detail.get("section_standard_ids")),
                "snippet": snippet or None,
                "matched_terms": matched_terms,
            }
        )
    return match_counts, previews


# ---------------------------------------------------------------------------
# Single-agreement and single-section reads
# ---------------------------------------------------------------------------


def get_agreement_row(
    deps: AgreementsDeps, agreement_uuid: str, *, include_xml: bool
) -> dict[str, object] | None:
    """Metadata (and optionally XML) for one publicly eligible agreement."""
    agreements = deps.Agreements
    xml = deps.XML
    columns = agreement_metadata_columns(deps)
    # The XML join is kept either way so agreements without a latest XML row 404.
    if include_xml:
        columns.append(xml.xml.label("xml"))
    row = (
        deps.db.session.query(*columns)
        .join(xml, deps._agreement_latest_xml_join_condition())
        .filter(
            agreements.agreement_uuid == agreement_uuid,
            agreement_is_public_eligible_expr(agreements),
        )
        .first()
    )
    if row is None:
        return None
    return deps._row_mapping_as_dict(cast(object, row))


def get_agreement(
    deps: AgreementsDeps,
    *,
    ctx: AccessContextProtocol,
    agreement_uuid: str,
    focus_section_uuid: str | None,
    neighbor_sections: int,
) -> dict[str, object] | None:
    row_map = get_agreement_row(deps, agreement_uuid, include_xml=True)
    if row_map is None:
        return None
    xml_content_obj = row_map.get("xml")
    xml_content = xml_content_obj if isinstance(xml_content_obj, str) else ""
    payload = row_to_agreement_metadata(row_map)
    if not ctx.is_authenticated:
        payload["xml"] = deps._redact_agreement_xml(
            xml_content,
            focus_section_uuid=focus_section_uuid,
            neighbor_sections=neighbor_sections,
        )
        payload["is_redacted"] = True
        return payload
    payload["xml"] = xml_content
    return payload


def list_agreement_sections(deps: AgreementsDeps, agreement_uuid: str) -> dict[str, object] | None:
    agreements = deps.Agreements
    sections = deps.Sections
    xml = deps.XML
    db = deps.db
    agreement_exists = (
        db.session.query(agreements.agreement_uuid)
        .join(xml, deps._agreement_latest_xml_join_condition())
        .filter(
            agreements.agreement_uuid == agreement_uuid,
            agreement_is_public_eligible_expr(agreements),
        )
        .first()
    )
    if agreement_exists is None:
        return None

    rows = cast(
        list[object],
        db.session.query(
            sections.section_uuid.label("section_uuid"),
            sections.article_title.label("article_title"),
            sections.section_title.label("section_title"),
            sections.article_order.label("article_order"),
            sections.section_order.label("section_order"),
            deps._coalesced_section_standard_ids().label("section_standard_ids"),
        )
        .join(
            xml,
            and_(
                sections.agreement_uuid == xml.agreement_uuid,
                sections.xml_version == xml.version,
                xml.latest == 1,
            ),
        )
        .filter(sections.agreement_uuid == agreement_uuid)
        .order_by(
            asc(sections.article_order),
            asc(sections.section_order),
            asc(sections.section_uuid),
        )
        .all(),
    )
    return {
        "agreement_uuid": agreement_uuid,
        "results": [
            {
                "section_uuid": row_map.get("section_uuid"),
                "article_title": row_map.get("article_title"),
                "section_title": row_map.get("section_title"),
                "article_order": row_map.get("article_order"),
                "section_order": row_map.get("section_order"),
                "standard_id": deps._parse_section_standard_ids(row_map.get("section_standard_ids")),
            }
            for row_map in (deps._row_mapping_as_dict(row) for row in rows)
        ],
    }


def get_section(deps: AgreementsDeps, section_uuid: str) -> dict[str, object] | None:
    """One latest section of a publicly eligible agreement.

    Returns both ``standard_id`` (the name every other section payload uses) and the
    legacy ``section_standard_id`` alias.
    """
    sections = deps.Sections
    agreements = deps.Agreements
    row = (
        deps.db.session.query(
            sections.agreement_uuid.label("agreement_uuid"),
            sections.section_uuid.label("section_uuid"),
            deps._coalesced_section_standard_ids().label("section_standard_ids"),
            sections.xml_content.label("xml_content"),
            sections.article_title.label("article_title"),
            sections.section_title.label("section_title"),
        )
        .join(deps.XML, deps._section_latest_xml_join_condition())
        .join(agreements, agreements.agreement_uuid == sections.agreement_uuid)
        .filter(
            sections.section_uuid == section_uuid,
            agreement_is_public_eligible_expr(agreements),
        )
        .first()
    )
    if row is None:
        return None
    row_map = deps._row_mapping_as_dict(cast(object, row))
    standard_ids = deps._parse_section_standard_ids(row_map.get("section_standard_ids"))
    return {
        "agreement_uuid": row_map.get("agreement_uuid"),
        "section_uuid": row_map.get("section_uuid"),
        "standard_id": standard_ids,
        "section_standard_id": list(standard_ids),
        "xml": row_map.get("xml_content"),
        "article_title": row_map.get("article_title"),
        "section_title": row_map.get("section_title"),
    }


# ---------------------------------------------------------------------------
# Tax-module clauses for one agreement or section
# ---------------------------------------------------------------------------


def _tax_clause_id_query(
    deps: AgreementsDeps,
    *,
    agreement_uuid: str | None = None,
    section_uuid: str | None = None,
    tax_standard_id: list[str] | None = None,
) -> Any:
    """Ordered, distinct tax-module clause ids matching the filters.

    Kept separate from the row fetch so callers can count and paginate on whole
    clauses. Paginating the joined row set instead would slice mid-clause and
    silently drop taxonomy assignments from the boundary clause.
    """
    agreements = deps.Agreements
    clauses = deps.Clauses
    agreement_cols = agreements.__table__.c
    clause_cols = clauses.__table__.c

    query = (
        deps.db.session.query(clause_cols["clause_uuid"].label("clause_uuid"))
        .join(agreements, agreement_cols["agreement_uuid"] == clause_cols["agreement_uuid"])
        .filter(
            clause_cols["module"] == "tax",
            agreement_is_public_eligible_expr(agreements),
        )
    )
    if agreement_uuid is not None:
        query = query.filter(clause_cols["agreement_uuid"] == agreement_uuid)
    if section_uuid is not None:
        query = query.filter(clause_cols["section_uuid"] == section_uuid)
    if tax_standard_id:
        expanded = list(
            deps._expand_tax_clause_taxonomy_standard_ids_cached(
                tuple(sorted({value for value in tax_standard_id if value}))
            )
        )
        if not expanded:
            return query.filter(false())
        query = query.filter(deps._tax_clause_standard_id_filter_expr(expanded))
    return query.order_by(
        asc(clause_cols["agreement_uuid"]),
        asc(clause_cols["section_uuid"]),
        asc(clause_cols["clause_order"]),
    )


def tax_clause_count(
    deps: AgreementsDeps,
    *,
    agreement_uuid: str | None = None,
    section_uuid: str | None = None,
    tax_standard_id: list[str] | None = None,
) -> int:
    id_query = _tax_clause_id_query(
        deps,
        agreement_uuid=agreement_uuid,
        section_uuid=section_uuid,
        tax_standard_id=tax_standard_id,
    )
    return cast(int, id_query.count())


def tax_clause_rows(
    deps: AgreementsDeps,
    *,
    agreement_uuid: str | None = None,
    section_uuid: str | None = None,
    tax_standard_id: list[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict[str, object]]:
    agreements = deps.Agreements
    clauses = deps.Clauses
    assignments = deps.TaxClauseAssignment
    sections = deps.Sections
    agreement_cols = agreements.__table__.c
    clause_cols = clauses.__table__.c
    section_cols = sections.__table__.c

    page_clause_uuids: list[str] | None = None
    if limit is not None or offset or tax_standard_id:
        id_query = _tax_clause_id_query(
            deps,
            agreement_uuid=agreement_uuid,
            section_uuid=section_uuid,
            tax_standard_id=tax_standard_id,
        )
        if offset:
            id_query = id_query.offset(offset)
        if limit is not None:
            id_query = id_query.limit(limit)
        page_clause_uuids = [str(row.clause_uuid) for row in id_query.all()]
        if not page_clause_uuids:
            return []

    query = (
        deps.db.session.query(
            clause_cols["clause_uuid"].label("clause_uuid"),
            clause_cols["agreement_uuid"].label("agreement_uuid"),
            clause_cols["section_uuid"].label("section_uuid"),
            section_cols["article_title"].label("article_title"),
            section_cols["section_title"].label("section_title"),
            clause_cols["anchor_label"].label("anchor_label"),
            clause_cols["start_char"].label("start_char"),
            clause_cols["end_char"].label("end_char"),
            clause_cols["clause_text"].label("clause_text"),
            clause_cols["context_type"].label("context_type"),
            assignments.standard_id.label("standard_id"),
        )
        .join(agreements, agreement_cols["agreement_uuid"] == clause_cols["agreement_uuid"])
        .join(
            sections,
            and_(
                section_cols["section_uuid"] == clause_cols["section_uuid"],
                section_cols["agreement_uuid"] == clause_cols["agreement_uuid"],
                section_cols["xml_version"] == clause_cols["xml_version"],
            ),
        )
        .join(deps.XML, deps._section_latest_xml_join_condition())
        .outerjoin(assignments, assignments.clause_uuid == clause_cols["clause_uuid"])
        .filter(
            clause_cols["module"] == "tax",
            agreement_is_public_eligible_expr(agreements),
        )
    )
    if agreement_uuid is not None:
        query = query.filter(clause_cols["agreement_uuid"] == agreement_uuid)
    if section_uuid is not None:
        query = query.filter(clause_cols["section_uuid"] == section_uuid)
    if page_clause_uuids is not None:
        query = query.filter(clause_cols["clause_uuid"].in_(page_clause_uuids))

    rows = query.order_by(
        asc(clause_cols["agreement_uuid"]),
        asc(clause_cols["section_uuid"]),
        asc(clause_cols["clause_order"]),
        asc(assignments.standard_id),
    ).all()

    grouped: dict[str, dict[str, object]] = {}
    for row in rows:
        row_map = deps._row_mapping_as_dict(cast(object, row))
        clause_uuid = str(row_map["clause_uuid"])
        grouped_row = grouped.get(clause_uuid)
        if grouped_row is None:
            grouped_row = {
                "clause_uuid": clause_uuid,
                "agreement_uuid": row_map.get("agreement_uuid"),
                "section_uuid": row_map.get("section_uuid"),
                "article_title": row_map.get("article_title"),
                "section_title": row_map.get("section_title"),
                "anchor_label": row_map.get("anchor_label"),
                "start_char": row_map.get("start_char"),
                "end_char": row_map.get("end_char"),
                "clause_text": row_map.get("clause_text"),
                "context_type": row_map.get("context_type"),
                "standard_ids": [],
            }
            grouped[clause_uuid] = grouped_row
        standard_id = row_map.get("standard_id")
        if isinstance(standard_id, str) and standard_id:
            standard_ids = cast(list[str], grouped_row["standard_ids"])
            if standard_id not in standard_ids:
                standard_ids.append(standard_id)
    return list(grouped.values())


# ---------------------------------------------------------------------------
# GET /v1/agreements-index
# ---------------------------------------------------------------------------


def agreements_index(
    deps: AgreementsDeps,
    *,
    ctx: AccessContextProtocol,
    page: int,
    page_size: int,
    sort_by: str,
    sort_direction: str,
    query: str,
) -> dict[str, object]:
    page = max(page, 1)
    page_size = clamp_page_size(
        page_size,
        max_page_size=search_max_page_size(is_authenticated=ctx.is_authenticated),
    )
    db = deps.db
    sort_column = {"year": "year", "target": "target", "acquirer": "acquirer"}.get(sort_by, "year")
    sort_direction_sql = "DESC" if sort_direction.lower() == "desc" else "ASC"
    where_clauses = ["1 = 1"]
    params: dict[str, object] = {}
    if query:
        if query.isdigit():
            where_clauses.append("year = :year_value")
            params["year_value"] = int(query)
        else:
            where_clauses.append("(target LIKE :query_like OR acquirer LIKE :query_like)")
            params["query_like"] = f"{query}%"
    where_sql = " AND ".join(where_clauses)
    total_count = deps._to_int(
        cast(
            object,
            db.session.execute(
                text(
                    f"""
                    SELECT COUNT(agreement_uuid) AS total_count
                    FROM {deps._schema_prefix()}agreement_index_summary
                    WHERE {where_sql}
                    """
                ),
                params,
            ).scalar(),
        )
    )
    items = (
        db.session.execute(
            text(
                f"""
                SELECT agreement_uuid, year, target, acquirer, verified
                FROM {deps._schema_prefix()}agreement_index_summary
                WHERE {where_sql}
                ORDER BY {sort_column} {sort_direction_sql}, agreement_uuid ASC
                LIMIT :limit_value OFFSET :offset_value
                """
            ),
            {**params, "limit_value": page_size, "offset_value": (page - 1) * page_size},
        )
        .mappings()
        .all()
    )
    meta = deps._pagination_metadata(total_count=total_count, page=page, page_size=page_size)

    # The summary table carries only the sortable columns; deal terms for the
    # visible page come from the agreements table by primary key.
    page_uuids = [
        str(uuid)
        for uuid in (deps._row_mapping_as_dict(cast(object, row)).get("agreement_uuid") for row in items)
        if uuid
    ]
    deal_terms: dict[str, tuple[object, object]] = {}
    if page_uuids:
        agreements = deps.Agreements
        for term_row in (
            db.session.query(
                agreements.agreement_uuid,
                agreements.transaction_consideration,
                agreements.transaction_price_total,
            )
            .filter(agreements.agreement_uuid.in_(page_uuids))
            .all()
        ):
            deal_terms[str(term_row[0])] = (term_row[1], term_row[2])

    results: list[dict[str, object]] = []
    for row in items:
        row_map = deps._row_mapping_as_dict(cast(object, row))
        verified_value = row_map.get("verified")
        consideration_type, total_consideration = deal_terms.get(
            str(row_map.get("agreement_uuid")), (None, None)
        )
        results.append(
            {
                "agreement_uuid": row_map.get("agreement_uuid"),
                "year": row_map.get("year"),
                "target": row_map.get("target"),
                "acquirer": row_map.get("acquirer"),
                "consideration_type": consideration_type,
                "total_consideration": to_float_or_none(total_consideration),
                "verified": bool(verified_value) if verified_value is not None else False,
            }
        )
    return {"results": results, **meta}


# ---------------------------------------------------------------------------
# Corpus summaries (cached by the routes)
# ---------------------------------------------------------------------------

_METADATA_FIELD_COVERAGE_DISPLAY_ORDER = {
    config["field"]: index for index, config in enumerate(METADATA_FIELD_COVERAGE_CONFIG)
}


def metadata_field_coverage_sort_key(row_dict: dict[str, object]) -> tuple[int, str]:
    field_name = row_dict.get("field_name")
    normalized_field_name = field_name if isinstance(field_name, str) else ""
    return (
        _METADATA_FIELD_COVERAGE_DISPLAY_ORDER.get(
            normalized_field_name,
            len(_METADATA_FIELD_COVERAGE_DISPLAY_ORDER),
        ),
        normalized_field_name,
    )


def metadata_field_coverage(deps: AgreementsDeps) -> list[dict[str, object]]:
    rows = (
        deps.db.session.execute(
            text(
                f"""
                SELECT
                    field_name,
                    label,
                    ingested_eligible_agreements,
                    ingested_covered_agreements,
                    ingested_coverage_pct,
                    processed_eligible_agreements,
                    processed_covered_agreements,
                    processed_coverage_pct,
                    note
                FROM {deps._schema_prefix()}agreement_metadata_field_coverage_summary
                """
            )
        )
        .mappings()
        .all()
    )
    sorted_rows = sorted(
        (deps._row_mapping_as_dict(cast(object, row)) for row in rows),
        key=metadata_field_coverage_sort_key,
    )
    return [
        {
            "field": row_dict.get("field_name"),
            "label": row_dict.get("label"),
            "ingested_eligible_agreements": deps._to_int(row_dict.get("ingested_eligible_agreements")),
            "ingested_covered_agreements": deps._to_int(row_dict.get("ingested_covered_agreements")),
            "ingested_coverage_pct": to_float_or_none(row_dict.get("ingested_coverage_pct")),
            "processed_eligible_agreements": deps._to_int(row_dict.get("processed_eligible_agreements")),
            "processed_covered_agreements": deps._to_int(row_dict.get("processed_covered_agreements")),
            "processed_coverage_pct": to_float_or_none(row_dict.get("processed_coverage_pct")),
            "note": row_dict.get("note"),
        }
        for row_dict in sorted_rows
    ]


def agreements_status_summary_payload(deps: AgreementsDeps) -> dict[str, object]:
    db = deps.db
    overview_row = (
        db.session.execute(
            text(
                f"""
                SELECT
                    metadata_covered_agreements,
                    metadata_coverage_pct,
                    taxonomy_covered_sections,
                    taxonomy_coverage_pct,
                    latest_filing_date
                FROM {deps._schema_prefix()}agreement_overview_summary
                LIMIT 1
                """
            )
        )
        .mappings()
        .first()
    )
    overview = (
        deps._row_mapping_as_dict(cast(object, overview_row)) if overview_row is not None else {}
    )
    rows = (
        db.session.execute(
            text(
                f"""
                SELECT year, color, current_stage, count
                FROM {deps._schema_prefix()}agreement_status_summary
                WHERE year IS NOT NULL
                ORDER BY year ASC, current_stage ASC, color ASC
                """
            )
        )
        .mappings()
        .all()
    )
    years: list[dict[str, object]] = []
    for row in rows:
        row_dict = deps._row_mapping_as_dict(cast(object, row))
        years.append(
            {
                "year": deps._to_int(row_dict.get("year")),
                "color": row_dict.get("color"),
                "current_stage": row_dict.get("current_stage"),
                "count": deps._to_int(row_dict.get("count")),
            }
        )
    return {
        "years": years,
        "latest_filing_date": _iso_date_or_none(overview.get("latest_filing_date")),
        "metadata_covered_agreements": deps._to_int(overview.get("metadata_covered_agreements")),
        "metadata_coverage_pct": to_float_or_none(overview.get("metadata_coverage_pct")),
        "taxonomy_covered_sections": deps._to_int(overview.get("taxonomy_covered_sections")),
        "taxonomy_coverage_pct": to_float_or_none(overview.get("taxonomy_coverage_pct")),
        "metadata_field_coverage": metadata_field_coverage(deps),
    }


def agreements_deal_types_summary_payload(deps: AgreementsDeps) -> dict[str, object]:
    rows = (
        deps.db.session.execute(
            text(
                f"""
                SELECT year, deal_type, `count`
                FROM {deps._schema_prefix()}agreement_deal_type_summary
                WHERE year IS NOT NULL
                ORDER BY year ASC, deal_type ASC
                """
            )
        )
        .mappings()
        .all()
    )
    years: list[dict[str, object]] = []
    for row in rows:
        row_dict = deps._row_mapping_as_dict(cast(object, row))
        years.append(
            {
                "year": deps._to_int(row_dict.get("year")),
                "deal_type": str(row_dict.get("deal_type") or "unknown"),
                "count": deps._to_int(row_dict.get("count")),
            }
        )
    return {"years": years}


def agreements_corpus_summary_payload(deps: AgreementsDeps) -> dict[str, object]:
    db = deps.db
    row = db.session.execute(
        text(
            f"""
            SELECT
              COALESCE(SUM(count_agreements), 0) AS agreements,
              COALESCE(SUM(count_sections), 0) AS sections,
              COALESCE(SUM(count_pages), 0) AS pages
            FROM {deps._schema_prefix()}summary_data
            """
        )
    ).mappings().first()
    try:
        overview_row = db.session.execute(
            text(
                f"""
                SELECT latest_filing_date
                FROM {deps._schema_prefix()}agreement_overview_summary
                LIMIT 1
                """
            )
        ).mappings().first()
    except SQLAlchemyError:
        overview_row = None
    row_dict = deps._row_mapping_as_dict(cast(object, row)) if row is not None else {}
    overview = (
        deps._row_mapping_as_dict(cast(object, overview_row)) if overview_row is not None else {}
    )
    return {
        "agreements": deps._to_int(row_dict.get("agreements")),
        "sections": deps._to_int(row_dict.get("sections")),
        "pages": deps._to_int(row_dict.get("pages")),
        "latest_filing_date": _iso_date_or_none(overview.get("latest_filing_date")),
    }


def counsel_leaderboards_payload(deps: AgreementsDeps) -> dict[str, object]:
    rows = (
        deps.db.session.execute(
            text(
                f"""
                SELECT side, counsel_key, counsel, year, deal_count, total_transaction_value
                FROM {deps._schema_prefix()}agreement_counsel_leaderboard_summary a
                ORDER BY year ASC, side ASC, counsel ASC
                """
            )
        )
        .mappings()
        .all()
    )
    return build_counsel_leaderboards_from_summary_rows(
        [deps._row_mapping_as_dict(cast(object, row)) for row in rows]
    )


# ---------------------------------------------------------------------------
# Agreement trends (REST /v1/agreement-trends and MCP get_agreement_trends)
# ---------------------------------------------------------------------------


def naics_label_by_code(deps: AgreementsDeps) -> dict[str, str]:
    schema_prefix = deps._schema_prefix()
    label_by_code: dict[str, str] = {}
    for table, code_col, desc_col in (
        ("naics_sectors", "sector_code", "sector_desc"),
        ("naics_sub_sectors", "sub_sector_code", "sub_sector_desc"),
    ):
        rows = deps.db.session.execute(
            text(f"SELECT {code_col}, {desc_col} FROM {schema_prefix}{table}")
        ).mappings().all()
        for row in rows:
            code = row.get(code_col)
            label = row.get(desc_col)
            if code is None or not isinstance(label, str):
                continue
            label_by_code[str(code)] = label
    return label_by_code


def ownership_trends(deps: AgreementsDeps) -> dict[str, list[dict[str, object]]]:
    """Public/private target mix and deal size by year, plus the buyer-type matrix."""
    db = deps.db
    schema_prefix = deps._schema_prefix()
    mix_rows = db.session.execute(
        text(
            f"""
            SELECT year, target_bucket, deal_count, total_transaction_value
            FROM {schema_prefix}agreement_ownership_mix_summary
            ORDER BY year ASC, target_bucket ASC
            """
        )
    ).mappings().all()
    deal_size_rows = db.session.execute(
        text(
            f"""
            SELECT year, target_bucket, deal_count, p25_transaction_value, median_transaction_value, p75_transaction_value
            FROM {schema_prefix}agreement_ownership_deal_size_summary
            ORDER BY year ASC, target_bucket ASC
            """
        )
    ).mappings().all()
    buyer_matrix_rows = db.session.execute(
        text(
            f"""
            SELECT target_bucket, buyer_bucket, deal_count, median_transaction_value
            FROM {schema_prefix}agreement_buyer_type_matrix_summary
            ORDER BY target_bucket ASC, buyer_bucket ASC
            """
        )
    ).mappings().all()

    mix_by_year: dict[int, dict[str, object]] = {}
    for row in mix_rows:
        year = deps._to_int(row.get("year"))
        year_row = mix_by_year.setdefault(
            year,
            {
                "year": year,
                "public_deal_count": 0,
                "private_deal_count": 0,
                "public_total_transaction_value": 0.0,
                "private_total_transaction_value": 0.0,
            },
        )
        bucket = str(row.get("target_bucket") or "")
        if bucket in ("public", "private"):
            year_row[f"{bucket}_deal_count"] = deps._to_int(row.get("deal_count"))
            year_row[f"{bucket}_total_transaction_value"] = (
                to_float_or_none(row.get("total_transaction_value")) or 0.0
            )

    deal_size_by_year: dict[int, dict[str, object]] = {}
    for row in deal_size_rows:
        year = deps._to_int(row.get("year"))
        year_row = deal_size_by_year.setdefault(
            year,
            {
                "year": year,
                "public_deal_count": 0,
                "private_deal_count": 0,
                "public_p25_transaction_value": None,
                "public_median_transaction_value": None,
                "public_p75_transaction_value": None,
                "private_p25_transaction_value": None,
                "private_median_transaction_value": None,
                "private_p75_transaction_value": None,
            },
        )
        bucket = str(row.get("target_bucket") or "")
        if bucket in ("public", "private"):
            year_row[f"{bucket}_deal_count"] = deps._to_int(row.get("deal_count"))
            for stat in ("p25", "median", "p75"):
                year_row[f"{bucket}_{stat}_transaction_value"] = to_float_or_none(
                    row.get(f"{stat}_transaction_value")
                )

    buyer_matrix_lookup = {
        (str(row.get("target_bucket") or ""), str(row.get("buyer_bucket") or "")): row
        for row in buyer_matrix_rows
    }
    buyer_matrix: list[dict[str, object]] = []
    for target_bucket in ("public", "private"):
        for buyer_bucket in ("public_buyer", "private_strategic", "private_equity", "other"):
            row = buyer_matrix_lookup.get((target_bucket, buyer_bucket))
            buyer_matrix.append(
                {
                    "target_bucket": target_bucket,
                    "buyer_bucket": buyer_bucket,
                    "deal_count": deps._to_int(row.get("deal_count")) if row else 0,
                    "median_transaction_value": (
                        to_float_or_none(row.get("median_transaction_value")) if row else None
                    ),
                }
            )

    return {
        "mix_by_year": [mix_by_year[year] for year in sorted(mix_by_year)],
        "deal_size_by_year": [deal_size_by_year[year] for year in sorted(deal_size_by_year)],
        "buyer_type_matrix": buyer_matrix,
    }


def target_industries_by_year(
    deps: AgreementsDeps, *, label_by_code: dict[str, str]
) -> list[dict[str, object]]:
    rows = deps.db.session.execute(
        text(
            f"""
            SELECT year, industry, deal_count, total_transaction_value
            FROM {deps._schema_prefix()}agreement_target_industry_summary
            ORDER BY year ASC, industry ASC
            """
        )
    ).mappings().all()
    return [
        {
            "year": deps._to_int(row.get("year")),
            "industry": normalize_industry_label(row.get("industry"), label_by_code=label_by_code),
            "deal_count": deps._to_int(row.get("deal_count")),
            "total_transaction_value": to_float_or_none(row.get("total_transaction_value")) or 0.0,
        }
        for row in rows
    ]


def industry_pairings(
    deps: AgreementsDeps, *, label_by_code: dict[str, str]
) -> list[dict[str, object]]:
    rows = deps.db.session.execute(
        text(
            f"""
            SELECT target_industry, acquirer_industry, deal_count, total_transaction_value
            FROM {deps._schema_prefix()}agreement_industry_pairing_summary
            ORDER BY deal_count DESC, total_transaction_value DESC, target_industry ASC, acquirer_industry ASC
            """
        )
    ).mappings().all()
    return [
        {
            "target_industry": normalize_industry_label(
                row.get("target_industry"), label_by_code=label_by_code
            ),
            "acquirer_industry": normalize_industry_label(
                row.get("acquirer_industry"), label_by_code=label_by_code
            ),
            "deal_count": deps._to_int(row.get("deal_count")),
            "total_transaction_value": to_float_or_none(row.get("total_transaction_value")) or 0.0,
        }
        for row in rows
    ]


def agreement_trends_payload(deps: AgreementsDeps) -> dict[str, object]:
    label_by_code = naics_label_by_code(deps)
    return {
        "ownership": ownership_trends(deps),
        "industries": {
            "target_industries_by_year": target_industries_by_year(deps, label_by_code=label_by_code),
            "pairings": industry_pairings(deps, label_by_code=label_by_code),
        },
    }


# ---------------------------------------------------------------------------
# Filter options
# ---------------------------------------------------------------------------

FILTER_OPTION_FIELDS: tuple[str, ...] = (
    "targets",
    "acquirers",
    "target_counsels",
    "acquirer_counsels",
    "target_industries",
    "acquirer_industries",
    "clause_types",
)


def clause_types_tree(deps: AgreementsDeps) -> dict[str, object]:
    """Deprecated ``clause_types`` filter option: the ``GET /v1/taxonomy`` tree, verbatim."""
    del deps
    return taxonomy_service.taxonomy_tree("clauses")


def filter_options_payload(deps: AgreementsDeps, requested_fields: tuple[str, ...]) -> dict[str, object]:
    summary_fields = tuple(field for field in requested_fields if field != "clause_types")
    payload: dict[str, object] = {}
    if summary_fields:
        rows = (
            deps.db.session.execute(
                text(
                    f"""
                    SELECT field_name, option_value
                    FROM {deps._schema_prefix()}agreement_filter_option_summary
                    WHERE field_name IN :field_names
                    ORDER BY field_name ASC, option_value ASC
                    """
                ).bindparams(bindparam("field_names", expanding=True)),
                {"field_names": list(summary_fields)},
            )
            .mappings()
            .all()
        )
        for field_name in summary_fields:
            payload[field_name] = []
        for row in rows:
            row_dict = deps._row_mapping_as_dict(cast(object, row))
            field_name = row_dict.get("field_name")
            option_value = row_dict.get("option_value")
            if isinstance(field_name, str) and isinstance(option_value, str):
                field_values = payload.setdefault(field_name, [])
                if isinstance(field_values, list):
                    cast(list[str], field_values).append(option_value)
    industry_fields = [
        field for field in ("target_industries", "acquirer_industries") if field in payload
    ]
    if industry_fields:
        payload["industry_labels"] = industry_labels_for(
            sorted(
                {
                    value
                    for field in industry_fields
                    for value in cast(list[str], payload[field])
                }
            )
        )
    if "clause_types" in requested_fields:
        payload["clause_types"] = clause_types_tree(deps)
    return payload


def filter_option_values(
    deps: AgreementsDeps, *, field_name: str, query: str, limit: int
) -> list[str]:
    """Prefix-matched option values for the ``target``/``acquirer`` typeahead."""
    params: dict[str, object] = {
        "field_name": "targets" if field_name == "target" else "acquirers",
        "limit_value": limit,
    }
    where_sql = "field_name = :field_name"
    if query:
        where_sql += " AND option_value LIKE :query_like"
        params["query_like"] = f"{query}%"
    rows = deps.db.session.execute(
        text(
            f"""
            SELECT option_value
            FROM {deps._schema_prefix()}agreement_filter_option_summary
            WHERE {where_sql}
            ORDER BY option_value ASC
            LIMIT :limit_value
            """
        ),
        params,
    ).all()
    return [cast(str, row[0]) for row in rows if isinstance(row[0], str) and row[0].strip()]


__all__ = [
    "AGREEMENT_METADATA_COLUMNS",
    "FILTER_OPTION_FIELDS",
    "MATCHED_SECTION_PREVIEW_LIMIT",
    "METADATA_NOT_SUPPORTED_MESSAGE",
    "PayloadBuilder",
    "agreement_metadata_columns",
    "agreement_trends_payload",
    "agreements_corpus_summary_payload",
    "agreements_deal_types_summary_payload",
    "agreements_index",
    "agreements_status_summary_payload",
    "clause_types_tree",
    "counsel_leaderboards_payload",
    "filter_option_values",
    "filter_options_payload",
    "get_agreement",
    "get_agreement_row",
    "get_section",
    "industry_pairings",
    "list_agreement_sections",
    "list_agreements",
    "metadata_field_coverage",
    "naics_label_by_code",
    "normalize_industry_label",
    "ownership_trends",
    "row_to_agreement_metadata",
    "search_agreements",
    "target_industries_by_year",
    "tax_clause_count",
    "tax_clause_rows",
    "to_float_or_none",
]
