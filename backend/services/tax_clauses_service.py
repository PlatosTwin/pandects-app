from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

from sqlalchemy import and_, asc, desc

from backend.core.pagination import clamp_page_size, search_max_page_size
from backend.filtering import agreement_is_public_eligible_expr, apply_agreement_filters
from backend.models.main_db import xml_latest_ok_filter
from backend.routes.deps import (
    AccessContextProtocol,
    CachedExactQueryCountProtocol,
    PaginationMetadataProtocol,
    RowMappingAsDictProtocol,
    ToIntProtocol,
)
from backend.search_counts import build_search_count_cache_key
from backend.schemas.tax_clauses import TaxClausesArgsPayload


@dataclass(frozen=True)
class TaxClausesServiceDeps:
    db: Any
    AgreementCounsel: Any
    Agreements: Any
    Clauses: Any
    Counsel: Any
    TaxClauseAssignment: Any
    TaxClauseTaxonomyL1: Any
    TaxClauseTaxonomyL2: Any
    TaxClauseTaxonomyL3: Any
    XML: Any
    _to_int: ToIntProtocol
    _cached_exact_query_count: CachedExactQueryCountProtocol
    _row_mapping_as_dict: RowMappingAsDictProtocol
    _pagination_metadata: PaginationMetadataProtocol
    _expand_tax_clause_taxonomy_standard_ids_cached: Callable[
        [tuple[str, ...]], tuple[str, ...]
    ]
    _tax_clause_standard_id_filter_expr: Callable[[list[str]], object]
    _year_from_filing_date_value: Callable[[object], int | None]


class _Query(Protocol):
    def filter(self, *args: object) -> "_Query": ...
    def order_by(self, *args: object) -> "_Query": ...
    def offset(self, n: int) -> "_Query": ...
    def limit(self, n: int) -> "_Query": ...
    def all(self) -> list[object]: ...
    def count(self) -> int: ...


def run_tax_clauses(
    deps: TaxClausesServiceDeps,
    *,
    ctx: AccessContextProtocol,
    parsed_args: TaxClausesArgsPayload,
) -> dict[str, object]:
    db = deps.db
    agreement_counsel = deps.AgreementCounsel
    counsel = deps.Counsel
    agreements = deps.Agreements
    clauses = deps.Clauses
    tax_clause_assignment = deps.TaxClauseAssignment
    xml = deps.XML
    row_mapping_as_dict = deps._row_mapping_as_dict
    pagination_metadata = deps._pagination_metadata
    expand_cached = deps._expand_tax_clause_taxonomy_standard_ids_cached
    tax_standard_id_filter_expr = deps._tax_clause_standard_id_filter_expr
    year_from_filing_date = deps._year_from_filing_date_value

    tax_standard_ids = parsed_args["tax_standard_id"]
    section_uuid = parsed_args["section_uuid"]
    clause_uuid = parsed_args["clause_uuid"]
    include_rep_warranty = parsed_args["include_rep_warranty"]
    count_mode = parsed_args["count_mode"]
    sort_by = parsed_args["sort_by"]
    sort_direction = parsed_args["sort_direction"]
    page = parsed_args["page"]
    page_size = parsed_args["page_size"]

    if page < 1:
        page = 1
    page_size = clamp_page_size(
        page_size,
        max_page_size=search_max_page_size(is_authenticated=ctx.is_authenticated),
    )

    q = cast(
        _Query,
        db.session.query(clauses.clause_uuid.label("clause_uuid"))
        .join(agreements, clauses.agreement_uuid == agreements.agreement_uuid)
        .join(
            xml,
            and_(
                clauses.agreement_uuid == xml.agreement_uuid,
                clauses.xml_version == xml.version,
                xml_latest_ok_filter(),
            ),
        )
        .filter(
            clauses.module == "tax",
            agreement_is_public_eligible_expr(agreements),
        ),
    )

    if not include_rep_warranty:
        q = q.filter(clauses.context_type != "rep_warranty")

    q = cast(
        _Query,
        apply_agreement_filters(
            q,
            agreements,
            cast(dict[str, object], cast(object, parsed_args)),
            agreement_counsel=agreement_counsel,
            counsel=counsel,
        ),
    )

    if tax_standard_ids:
        key = tuple(sorted({value for value in tax_standard_ids if value}))
        expanded = list(expand_cached(key))
        if expanded:
            q = q.filter(tax_standard_id_filter_expr(expanded))

    if section_uuid and section_uuid.strip():
        q = q.filter(clauses.section_uuid == section_uuid.strip())
    if clause_uuid and clause_uuid.strip():
        q = q.filter(clauses.clause_uuid == clause_uuid.strip())

    descending = sort_direction == "desc"
    if sort_by == "year":
        primary_sort = agreements.filing_date
    elif sort_by == "target":
        primary_sort = agreements.target
    else:
        primary_sort = agreements.acquirer
    if descending:
        q = q.order_by(
            desc(primary_sort),
            desc(clauses.agreement_uuid),
            asc(clauses.section_uuid),
            asc(clauses.clause_order),
        )
    else:
        q = q.order_by(
            asc(primary_sort),
            asc(clauses.agreement_uuid),
            asc(clauses.section_uuid),
            asc(clauses.clause_order),
        )

    offset = (page - 1) * page_size
    page_rows = q.offset(offset).limit(page_size + 1).all()
    has_next = len(page_rows) > page_size
    item_rows = page_rows[:page_size]
    item_count = len(item_rows)

    clause_uuids: list[str] = []
    for row in item_rows:
        value = row_mapping_as_dict(row).get("clause_uuid")
        if isinstance(value, str):
            clause_uuids.append(value)

    count_cache_key = (
        None
        if count_mode == "exact"
        else build_search_count_cache_key("tax_clauses", parsed_args)
    )
    total_count = deps._cached_exact_query_count(q, cache_key=count_cache_key)
    total_count_is_approximate = False
    count_method = "query_count"

    details_by_uuid: dict[str, dict[str, object]] = {}
    assignments_by_uuid: dict[str, list[str]] = {}
    if clause_uuids:
        detail_rows = (
            db.session.query(
                clauses.clause_uuid.label("clause_uuid"),
                clauses.agreement_uuid.label("agreement_uuid"),
                clauses.section_uuid.label("section_uuid"),
                clauses.clause_text.label("clause_text"),
                clauses.anchor_label.label("anchor_label"),
                clauses.context_type.label("context_type"),
                clauses.source_method.label("source_method"),
                agreements.filing_date.label("filing_date"),
                agreements.target.label("target"),
                agreements.acquirer.label("acquirer"),
                agreements.verified.label("verified"),
                agreements.transaction_price_total.label("transaction_price_total"),
                agreements.transaction_consideration.label("transaction_consideration"),
                agreements.deal_status.label("deal_status"),
                agreements.deal_type.label("deal_type"),
                agreements.target_counsel.label("target_counsel"),
                agreements.acquirer_counsel.label("acquirer_counsel"),
            )
            .join(agreements, clauses.agreement_uuid == agreements.agreement_uuid)
            .filter(clauses.clause_uuid.in_(clause_uuids))
            .all()
        )
        for row in detail_rows:
            row_map = row_mapping_as_dict(row)
            row_clause_uuid = row_map.get("clause_uuid")
            if isinstance(row_clause_uuid, str):
                details_by_uuid[row_clause_uuid] = row_map

        assignment_rows = cast(
            list[tuple[object, object]],
            db.session.query(
                tax_clause_assignment.clause_uuid,
                tax_clause_assignment.standard_id,
            )
            .filter(tax_clause_assignment.clause_uuid.in_(clause_uuids))
            .all(),
        )
        for clause_uuid_value, standard_id_value in assignment_rows:
            if isinstance(clause_uuid_value, str) and isinstance(standard_id_value, str):
                assignments_by_uuid.setdefault(clause_uuid_value, []).append(standard_id_value)

    meta = pagination_metadata(
        total_count=total_count,
        page=page,
        page_size=page_size,
        has_next_override=has_next,
        total_count_is_approximate=total_count_is_approximate,
    )

    results: list[dict[str, object]] = []
    for clause_uuid_value in clause_uuids:
        detail = details_by_uuid.get(clause_uuid_value)
        if detail is None:
            raise RuntimeError(
                f"Clause UUID {clause_uuid_value} missing from detail lookup."
            )
        results.append(
            {
                "id": clause_uuid_value,
                "clause_uuid": clause_uuid_value,
                "agreement_uuid": detail.get("agreement_uuid"),
                "section_uuid": detail.get("section_uuid"),
                "clause_text": detail.get("clause_text"),
                "anchor_label": detail.get("anchor_label"),
                "context_type": detail.get("context_type"),
                "source_method": detail.get("source_method"),
                "tax_standard_ids": sorted(set(assignments_by_uuid.get(clause_uuid_value, []))),
                "year": year_from_filing_date(detail.get("filing_date")),
                "target": detail.get("target"),
                "acquirer": detail.get("acquirer"),
                "verified": (
                    bool(detail.get("verified"))
                    if detail.get("verified") is not None
                    else False
                ),
                "transaction_price_total": detail.get("transaction_price_total"),
                "transaction_consideration": detail.get("transaction_consideration"),
                "deal_status": detail.get("deal_status"),
                "deal_type": detail.get("deal_type"),
                "target_counsel": detail.get("target_counsel"),
                "acquirer_counsel": detail.get("acquirer_counsel"),
            }
        )

    return {
        "results": results,
        "access": {
            "tier": ctx.tier,
            "message": None
            if ctx.is_authenticated
            else "Limited mode: sign in to view clause text, unlock full pagination, and use the MCP server.",
        },
        "count_metadata": {
            "mode": "estimated" if total_count_is_approximate else "exact",
            "method": count_method,
            "planning_reliability": "low" if total_count_is_approximate else "high",
            "exact_count_requested": count_mode == "exact",
        },
        **meta,
    }


__all__ = [
    "TaxClausesServiceDeps",
    "run_tax_clauses",
]
