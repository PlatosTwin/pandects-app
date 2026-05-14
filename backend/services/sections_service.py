from __future__ import annotations

from typing import cast

from sqlalchemy import text, and_, or_, asc, desc
from sqlalchemy.exc import SQLAlchemyError

from backend.filtering import (
    build_canonical_counsel_agreement_uuid_subquery,
    build_transaction_price_bucket_filter,
)
from backend.routes.deps import AccessContextProtocol, SectionsServiceDeps
from backend.search_counts import (
    count_metadata_payload,
    estimated_query_row_count as _shared_estimated_query_row_count,
    search_total_count_metadata,
)
from backend.schemas.sections import SectionsArgsPayload


def estimated_query_row_count(deps: SectionsServiceDeps, query: object) -> int | None:
    return _shared_estimated_query_row_count(deps, query)


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
) -> tuple[int, bool, str]:
    """Return `(total_count, is_approximate, method)` without fake lower bounds."""
    return search_total_count_metadata(
        deps,
        query=query,
        page=page,
        page_size=page_size,
        item_count=item_count,
        has_next=has_next,
        has_filters=has_filters,
        count_mode=count_mode,
        estimated_query_row_count_fn=deps._estimated_query_row_count,
        estimated_table_rows_fn=deps._estimated_latest_sections_search_table_rows,
    )


def _sections_count_metadata_payload(
    *,
    total_count_is_approximate: bool,
    count_method: str,
    exact_count_requested: bool,
) -> dict[str, object]:
    return count_metadata_payload(
        total_count_is_approximate=total_count_is_approximate,
        count_method=count_method,
        exact_count_requested=exact_count_requested,
    )


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
    if taxonomy_filters:
        notes.append("Taxonomy filters reflect clause-family assignments and may act as proxies for broader legal concepts.")
    if total_count_is_approximate:
        notes.append("Counts are approximate under the current mode; use count_mode=exact when pagination certainty matters.")
    elif count_method == "query_count":
        notes.append("Counts were computed exactly from the current filtered query.")

    return {
        "applied_filters": applied_filters,
        "taxonomy_filters": taxonomy_filters,
        "heuristics_used": [],
        "notes": notes,
    }


def run_sections(
    deps: SectionsServiceDeps,
    *,
    ctx: AccessContextProtocol,
    parsed_args: SectionsArgsPayload,
) -> dict[str, object]:
    db = deps.db
    agreement_counsel = deps.AgreementCounsel
    counsel = deps.Counsel
    latest = deps.LatestSectionsSearch
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
    count_mode = parsed_args["count_mode"]
    sort_by = parsed_args["sort_by"]
    sort_direction = parsed_args["sort_direction"]
    page = parsed_args["page"]
    page_size = parsed_args["page_size"]
    include_xml = True

    if page < 1:
        page = 1
    max_page_size = 100 if ctx.is_authenticated else 10
    if page_size < 1 or page_size > max_page_size:
        page_size = min(25, max_page_size)

    # Build the ID-only query first so filters, sort order, and count estimation all share
    # the same search surface before we hydrate the selected rows.
    q = db.session.query(latest.section_uuid.label("section_uuid"))

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

    descending = sort_direction == "desc"
    if sort_by == "year":
        primary_sort = latest.filing_date
    elif sort_by == "target":
        primary_sort = latest.target
    else:
        primary_sort = latest.acquirer
    if descending:
        q = q.order_by(desc(primary_sort), desc(latest.section_uuid))
    else:
        q = q.order_by(asc(primary_sort), asc(latest.section_uuid))

    offset = (page - 1) * page_size
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
        )
    )
    total_count, total_count_is_approximate, count_method = sections_total_count_metadata(
        deps,
        query=q,
        page=page,
        page_size=page_size,
        item_count=item_count,
        has_next=has_next,
        has_filters=has_filters,
        count_mode=count_mode,
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
    return {
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


__all__ = [
    "estimated_latest_sections_search_table_rows",
    "estimated_query_row_count",
    "run_sections",
    "sections_total_count_metadata",
]
