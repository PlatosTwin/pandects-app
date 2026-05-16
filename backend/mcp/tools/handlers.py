from __future__ import annotations

import re
from datetime import date
from typing import Any, cast

from flask import abort
from marshmallow import Schema, fields as ma_fields, validate
from sqlalchemy import and_, asc, desc, func, or_, text

from backend.auth.mcp_runtime import McpPrincipal
from backend.filtering import (
    build_any_counsel_agreement_uuid_subquery,
    build_canonical_counsel_agreement_uuid_subquery,
    build_transaction_price_bucket_filter,
)
from backend.mcp.tools.args_schemas import (
    McpAgreementArgsSchema,
    McpBatchAgreementSectionsArgsSchema,
    McpFilterOptionsArgsSchema,
    McpListAgreementSectionsArgsSchema,
    McpSearchAgreementsExtraArgsSchema,
    McpSectionArgsSchema,
)
from backend.mcp.tools.constants import (
    _FILTER_OPTIONS_FIELDS,
    _TRANSACTION_PRICE_BUCKET_OPTIONS,
)
from backend.mcp.tools.dispatch import McpToolResult, _require_scope, _validate_payload
from backend.mcp.tools.schema_utils import (
    _filter_option_metadata,
    _merge_schema_instances,
    _schema_from_fields,
)
from backend.mcp.tools.shared import (
    _agreement_filter_interpretation,
    _agreement_trends_payload,
    _agreements_summary_payload,
    _build_taxonomy_tree,
    _count_metadata_payload,
    _counsel_payload,
    _extract_monetary_values,
    _extract_text_from_xml,
    _focused_snippet,
    _json_compatible_value,
    _naics_payload,
    _normalized_page,
    _normalized_page_size,
    _ranked_taxonomy_matches,
)
from backend.routes.agreements import _agreement_is_public_eligible_expr, _tax_clause_rows
from backend.routes.deps import AgreementsDeps, ReferenceDataDeps, SectionsServiceDeps
from backend.schemas.public_api import (
    AgreementArgsPayload,
    AgreementArgsSchema,
    AgreementsBulkArgsPayload,
    AgreementsBulkArgsSchema,
    AgreementsIndexArgsSchema,
)
from backend.schemas.sections import SectionsArgsPayload, SectionsArgsSchema
from backend.services.sections_service import run_sections


def _list_agreements(
    deps: AgreementsDeps,
    *,
    principal: McpPrincipal,
    payload: dict[str, object],
) -> McpToolResult:
    _require_scope(principal, "agreements:search")
    parsed_args = cast(
        AgreementsBulkArgsPayload,
        cast(object, _validate_payload(
            _merge_schema_instances(AgreementsBulkArgsSchema(), McpSearchAgreementsExtraArgsSchema()),
            payload,
        )),
    )
    any_counsel_values = cast(list[str], payload.get("any_counsel", []))
    include_xml = parsed_args["include_xml"]
    if include_xml:
        _require_scope(principal, "agreements:read_fulltext")

    page_size = _normalized_page_size(parsed_args["page_size"])
    after_agreement_uuid = deps._decode_agreements_cursor(parsed_args["cursor"])
    agreements = deps.Agreements
    agreement_counsel = deps.AgreementCounsel
    counsel = deps.Counsel
    xml = deps.XML
    sections = deps.Sections
    latest_sections = deps.LatestSectionsSearch
    db = deps.db
    year_expr = deps._agreement_year_expr().label("year")
    section_count_subquery = (
        db.session.query(func.count(latest_sections.section_uuid))
        .filter(latest_sections.agreement_uuid == agreements.agreement_uuid)
        .correlate(agreements)
        .scalar_subquery()
    ).label("section_count")
    item_columns = [
        agreements.agreement_uuid.label("agreement_uuid"),
        year_expr,
        agreements.target.label("target"),
        agreements.acquirer.label("acquirer"),
        agreements.filing_date.label("filing_date"),
        agreements.prob_filing.label("prob_filing"),
        agreements.filing_company_name.label("filing_company_name"),
        agreements.filing_company_cik.label("filing_company_cik"),
        agreements.form_type.label("form_type"),
        agreements.exhibit_type.label("exhibit_type"),
        agreements.transaction_price_total.label("transaction_price_total"),
        agreements.transaction_price_stock.label("transaction_price_stock"),
        agreements.transaction_price_cash.label("transaction_price_cash"),
        agreements.transaction_price_assets.label("transaction_price_assets"),
        agreements.transaction_consideration.label("transaction_consideration"),
        agreements.target_type.label("target_type"),
        agreements.acquirer_type.label("acquirer_type"),
        agreements.target_industry.label("target_industry"),
        agreements.acquirer_industry.label("acquirer_industry"),
        agreements.announce_date.label("announce_date"),
        agreements.close_date.label("close_date"),
        agreements.deal_status.label("deal_status"),
        agreements.attitude.label("attitude"),
        agreements.deal_type.label("deal_type"),
        agreements.purpose.label("purpose"),
        agreements.target_pe.label("target_pe"),
        agreements.acquirer_pe.label("acquirer_pe"),
        agreements.url.label("url"),
        section_count_subquery,
    ]
    q = (
        db.session.query(*item_columns)
        .join(xml, deps._agreement_latest_xml_join_condition())
        .filter(_agreement_is_public_eligible_expr(agreements))
    )

    if include_xml:
        q = q.add_columns(xml.xml.label("xml"))

    years = parsed_args["year"]
    if years:
        year_filters = tuple(
            and_(
                agreements.filing_date >= f"{year:04d}-01-01",
                agreements.filing_date < f"{year + 1:04d}-01-01",
            )
            for year in years
        )
        q = q.filter(or_(*year_filters))

    year_min_val = cast(int | None, payload.get("year_min"))
    year_max_val = cast(int | None, payload.get("year_max"))
    if year_min_val is not None:
        q = q.filter(agreements.filing_date >= f"{year_min_val:04d}-01-01")
    if year_max_val is not None:
        q = q.filter(agreements.filing_date < f"{year_max_val + 1:04d}-01-01")

    list_filters = (
        ("target", agreements.target),
        ("acquirer", agreements.acquirer),
        ("transaction_consideration", agreements.transaction_consideration),
        ("target_type", agreements.target_type),
        ("acquirer_type", agreements.acquirer_type),
        ("target_industry", agreements.target_industry),
        ("acquirer_industry", agreements.acquirer_industry),
        ("deal_status", agreements.deal_status),
        ("attitude", agreements.attitude),
        ("deal_type", agreements.deal_type),
        ("purpose", agreements.purpose),
    )
    for key, column in list_filters:
        values = parsed_args[key]
        if values:
            q = q.filter(column.in_(values))

    transaction_price_total_filter = build_transaction_price_bucket_filter(
        agreements.transaction_price_total,
        parsed_args["transaction_price_total"],
    )
    if transaction_price_total_filter is not None:
        q = q.filter(transaction_price_total_filter)
    transaction_price_stock_filter = build_transaction_price_bucket_filter(
        agreements.transaction_price_stock,
        parsed_args["transaction_price_stock"],
    )
    if transaction_price_stock_filter is not None:
        q = q.filter(transaction_price_stock_filter)
    transaction_price_cash_filter = build_transaction_price_bucket_filter(
        agreements.transaction_price_cash,
        parsed_args["transaction_price_cash"],
    )
    if transaction_price_cash_filter is not None:
        q = q.filter(transaction_price_cash_filter)
    transaction_price_assets_filter = build_transaction_price_bucket_filter(
        agreements.transaction_price_assets,
        parsed_args["transaction_price_assets"],
    )
    if transaction_price_assets_filter is not None:
        q = q.filter(transaction_price_assets_filter)

    for key, column in (("target_pe", agreements.target_pe), ("acquirer_pe", agreements.acquirer_pe)):
        values = parsed_args[key]
        if not values:
            continue
        db_values: list[int] = []
        for value in values:
            if value == "true":
                db_values.append(1)
            elif value == "false":
                db_values.append(0)
        if db_values:
            q = q.filter(column.in_(db_values))

    agreement_uuid = parsed_args["agreement_uuid"]
    if agreement_uuid and agreement_uuid.strip():
        q = q.filter(agreements.agreement_uuid == agreement_uuid.strip())

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

    standard_ids = [v for v in cast(list[str], parsed_args["standard_id"]) if v]
    if standard_ids:
        standard_ids_key = tuple(sorted(set(standard_ids)))
        expanded = list(deps._expand_taxonomy_standard_ids_cached(standard_ids_key))
        if expanded:
            q = q.filter(deps._standard_id_agreement_filter_expr(agreements.agreement_uuid, expanded))

    target_counsel_subquery = build_canonical_counsel_agreement_uuid_subquery(
        side="target",
        canonical_names=cast(list[str], parsed_args["target_counsel"]),
        agreement_counsel=agreement_counsel,
        counsel=counsel,
    )
    if target_counsel_subquery is not None:
        q = q.filter(agreements.agreement_uuid.in_(target_counsel_subquery))
    acquirer_counsel_subquery = build_canonical_counsel_agreement_uuid_subquery(
        side="acquirer",
        canonical_names=cast(list[str], parsed_args["acquirer_counsel"]),
        agreement_counsel=agreement_counsel,
        counsel=counsel,
    )
    if acquirer_counsel_subquery is not None:
        q = q.filter(agreements.agreement_uuid.in_(acquirer_counsel_subquery))
    any_counsel_subquery = build_any_counsel_agreement_uuid_subquery(
        canonical_names=any_counsel_values,
        agreement_counsel=agreement_counsel,
        counsel=counsel,
    )
    if any_counsel_subquery is not None:
        q = q.filter(agreements.agreement_uuid.in_(any_counsel_subquery))

    if after_agreement_uuid:
        q = q.filter(agreements.agreement_uuid > after_agreement_uuid)

    rows = cast(list[object], q.order_by(asc(agreements.agreement_uuid)).limit(page_size + 1).all())
    has_next = len(rows) > page_size
    page_rows = rows[:page_size]
    results: list[dict[str, object]] = []
    for row in page_rows:
        row_map = deps._row_mapping_as_dict(row)
        item = {
            "agreement_uuid": _json_compatible_value(row_map.get("agreement_uuid")),
            "year": _json_compatible_value(row_map.get("year")),
            "target": _json_compatible_value(row_map.get("target")),
            "acquirer": _json_compatible_value(row_map.get("acquirer")),
            "filing_date": _json_compatible_value(row_map.get("filing_date")),
            "prob_filing": _json_compatible_value(row_map.get("prob_filing")),
            "filing_company_name": _json_compatible_value(row_map.get("filing_company_name")),
            "filing_company_cik": _json_compatible_value(row_map.get("filing_company_cik")),
            "form_type": _json_compatible_value(row_map.get("form_type")),
            "exhibit_type": _json_compatible_value(row_map.get("exhibit_type")),
            "transaction_price_total": _json_compatible_value(row_map.get("transaction_price_total")),
            "transaction_price_stock": _json_compatible_value(row_map.get("transaction_price_stock")),
            "transaction_price_cash": _json_compatible_value(row_map.get("transaction_price_cash")),
            "transaction_price_assets": _json_compatible_value(row_map.get("transaction_price_assets")),
            "transaction_consideration": _json_compatible_value(row_map.get("transaction_consideration")),
            "target_type": _json_compatible_value(row_map.get("target_type")),
            "acquirer_type": _json_compatible_value(row_map.get("acquirer_type")),
            "target_industry": _json_compatible_value(row_map.get("target_industry")),
            "acquirer_industry": _json_compatible_value(row_map.get("acquirer_industry")),
            "announce_date": _json_compatible_value(row_map.get("announce_date")),
            "close_date": _json_compatible_value(row_map.get("close_date")),
            "deal_status": _json_compatible_value(row_map.get("deal_status")),
            "attitude": _json_compatible_value(row_map.get("attitude")),
            "deal_type": _json_compatible_value(row_map.get("deal_type")),
            "purpose": _json_compatible_value(row_map.get("purpose")),
            "target_pe": _json_compatible_value(row_map.get("target_pe")),
            "acquirer_pe": _json_compatible_value(row_map.get("acquirer_pe")),
            "url": _json_compatible_value(row_map.get("url")),
            "section_count": _json_compatible_value(row_map.get("section_count")),
        }
        if include_xml:
            item["xml"] = _json_compatible_value(row_map.get("xml"))
        results.append(item)

    next_cursor: str | None = None
    if has_next:
        last_row = deps._row_mapping_as_dict(page_rows[-1])
        last_agreement_uuid = last_row.get("agreement_uuid")
        if not isinstance(last_agreement_uuid, str) or not last_agreement_uuid:
            raise RuntimeError("Agreements list query returned a row without agreement_uuid.")
        next_cursor = deps._encode_agreements_cursor(last_agreement_uuid)

    response = {
        "results": results,
        "access": {"tier": principal.access_context.tier, "message": None},
        "page_size": page_size,
        "returned_count": len(results),
        "has_next": has_next,
        "next_cursor": next_cursor,
    }
    return McpToolResult(
        text=f"Returned {len(results)} agreement(s).",
        structured_content=response,
    )


def _search_agreements(
    deps: AgreementsDeps,
    *,
    principal: McpPrincipal,
    payload: dict[str, object],
) -> McpToolResult:
    _require_scope(principal, "agreements:search")
    parsed_args = _validate_payload(
        _merge_schema_instances(AgreementsIndexArgsSchema(), AgreementsBulkArgsSchema(), McpSearchAgreementsExtraArgsSchema()),
        payload,
    )
    page = _normalized_page(cast(int, parsed_args["page"]))
    page_size = _normalized_page_size(cast(int, parsed_args["page_size"]))
    sort_by = cast(str, parsed_args["sort_by"])
    sort_dir = cast(str, parsed_args["sort_dir"])
    query = cast(str, parsed_args["query"]).strip()

    agreements = deps.Agreements
    agreement_counsel = deps.AgreementCounsel
    counsel = deps.Counsel
    xml = deps.XML
    sections = deps.Sections
    latest_sections = deps.LatestSectionsSearch
    db = deps.db
    year_expr = deps._agreement_year_expr()
    sort_map = {"year": year_expr, "target": agreements.target, "acquirer": agreements.acquirer}
    sort_column = sort_map.get(sort_by, year_expr)
    order_by = sort_column.desc() if sort_dir == "desc" else sort_column.asc()
    section_count_subquery = (
        db.session.query(func.count(latest_sections.section_uuid))
        .filter(latest_sections.agreement_uuid == agreements.agreement_uuid)
        .correlate(agreements)
        .scalar_subquery()
    ).label("section_count")

    q = (
        db.session.query(
            agreements.agreement_uuid,
            year_expr.label("year"),
            agreements.target,
            agreements.acquirer,
            agreements.filing_date,
            agreements.url,
            agreements.verified,
            section_count_subquery,
        )
        .join(xml, deps._agreement_latest_xml_join_condition())
        .filter(_agreement_is_public_eligible_expr(agreements))
    )
    count_q = (
        db.session.query(func.count(agreements.agreement_uuid))
        .select_from(agreements)
        .join(xml, deps._agreement_latest_xml_join_condition())
        .filter(_agreement_is_public_eligible_expr(agreements))
    )

    years = cast(list[int], parsed_args["year"])
    if years:
        year_filters = tuple(
            and_(
                agreements.filing_date >= f"{year:04d}-01-01",
                agreements.filing_date < f"{year + 1:04d}-01-01",
            )
            for year in years
        )
        year_clause = or_(*year_filters)
        q = q.filter(year_clause)
        count_q = count_q.filter(year_clause)

    year_min = cast(int | None, parsed_args.get("year_min"))
    year_max = cast(int | None, parsed_args.get("year_max"))
    if year_min is not None:
        clause = agreements.filing_date >= f"{year_min:04d}-01-01"
        q = q.filter(clause)
        count_q = count_q.filter(clause)
    if year_max is not None:
        clause = agreements.filing_date < f"{year_max + 1:04d}-01-01"
        q = q.filter(clause)
        count_q = count_q.filter(clause)

    filed_after = cast(str | None, parsed_args.get("filed_after"))
    filed_before = cast(str | None, parsed_args.get("filed_before"))
    if filed_after:
        clause = agreements.filing_date >= filed_after
        q = q.filter(clause)
        count_q = count_q.filter(clause)
    if filed_before:
        clause = agreements.filing_date < filed_before
        q = q.filter(clause)
        count_q = count_q.filter(clause)

    list_filters = (
        ("target", agreements.target),
        ("acquirer", agreements.acquirer),
        ("transaction_consideration", agreements.transaction_consideration),
        ("target_type", agreements.target_type),
        ("acquirer_type", agreements.acquirer_type),
        ("target_industry", agreements.target_industry),
        ("acquirer_industry", agreements.acquirer_industry),
        ("deal_status", agreements.deal_status),
        ("attitude", agreements.attitude),
        ("deal_type", agreements.deal_type),
        ("purpose", agreements.purpose),
    )
    for key, column in list_filters:
        values = cast(list[str], parsed_args[key])
        if values:
            clause = column.in_(values)
            q = q.filter(clause)
            count_q = count_q.filter(clause)

    transaction_price_total_filter = build_transaction_price_bucket_filter(
        agreements.transaction_price_total,
        cast(list[str], parsed_args["transaction_price_total"]),
    )
    if transaction_price_total_filter is not None:
        q = q.filter(transaction_price_total_filter)
        count_q = count_q.filter(transaction_price_total_filter)
    transaction_price_stock_filter = build_transaction_price_bucket_filter(
        agreements.transaction_price_stock,
        cast(list[str], parsed_args["transaction_price_stock"]),
    )
    if transaction_price_stock_filter is not None:
        q = q.filter(transaction_price_stock_filter)
        count_q = count_q.filter(transaction_price_stock_filter)
    transaction_price_cash_filter = build_transaction_price_bucket_filter(
        agreements.transaction_price_cash,
        cast(list[str], parsed_args["transaction_price_cash"]),
    )
    if transaction_price_cash_filter is not None:
        q = q.filter(transaction_price_cash_filter)
        count_q = count_q.filter(transaction_price_cash_filter)
    transaction_price_assets_filter = build_transaction_price_bucket_filter(
        agreements.transaction_price_assets,
        cast(list[str], parsed_args["transaction_price_assets"]),
    )
    if transaction_price_assets_filter is not None:
        q = q.filter(transaction_price_assets_filter)
        count_q = count_q.filter(transaction_price_assets_filter)

    for key, column in (("target_pe", agreements.target_pe), ("acquirer_pe", agreements.acquirer_pe)):
        values = cast(list[str], parsed_args[key])
        if not values:
            continue
        db_values: list[int] = []
        for value in values:
            if value == "true":
                db_values.append(1)
            elif value == "false":
                db_values.append(0)
        if db_values:
            clause = column.in_(db_values)
            q = q.filter(clause)
            count_q = count_q.filter(clause)

    target_counsel_subquery = build_canonical_counsel_agreement_uuid_subquery(
        side="target",
        canonical_names=cast(list[str], parsed_args["target_counsel"]),
        agreement_counsel=agreement_counsel,
        counsel=counsel,
    )
    if target_counsel_subquery is not None:
        clause = agreements.agreement_uuid.in_(target_counsel_subquery)
        q = q.filter(clause)
        count_q = count_q.filter(clause)
    acquirer_counsel_subquery = build_canonical_counsel_agreement_uuid_subquery(
        side="acquirer",
        canonical_names=cast(list[str], parsed_args["acquirer_counsel"]),
        agreement_counsel=agreement_counsel,
        counsel=counsel,
    )
    if acquirer_counsel_subquery is not None:
        clause = agreements.agreement_uuid.in_(acquirer_counsel_subquery)
        q = q.filter(clause)
        count_q = count_q.filter(clause)

    any_counsel_subquery = build_any_counsel_agreement_uuid_subquery(
        canonical_names=cast(list[str], parsed_args["any_counsel"]),
        agreement_counsel=agreement_counsel,
        counsel=counsel,
    )
    if any_counsel_subquery is not None:
        clause = agreements.agreement_uuid.in_(any_counsel_subquery)
        q = q.filter(clause)
        count_q = count_q.filter(clause)

    agreement_uuid = cast(str | None, parsed_args["agreement_uuid"])
    if agreement_uuid and agreement_uuid.strip():
        clause = agreements.agreement_uuid == agreement_uuid.strip()
        q = q.filter(clause)
        count_q = count_q.filter(clause)

    section_uuid = cast(str | None, parsed_args["section_uuid"])
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
        count_q = count_q.filter(section_exists)

    standard_ids = [v for v in cast(list[str], parsed_args["standard_id"]) if v]
    if standard_ids:
        standard_ids_key = tuple(sorted(set(standard_ids)))
        expanded = list(deps._expand_taxonomy_standard_ids_cached(standard_ids_key))
        if expanded:
            std_id_clause = deps._standard_id_agreement_filter_expr(agreements.agreement_uuid, expanded)
            q = q.filter(std_id_clause)
            count_q = count_q.filter(std_id_clause)

    if query:
        if query.isdigit():
            year_value = int(query)
            q = q.filter(year_expr == year_value)
            count_q = count_q.filter(year_expr == year_value)
        else:
            like = f"{query}%"
            filters = or_(agreements.target.ilike(like), agreements.acquirer.ilike(like))
            q = q.filter(filters)
            count_q = count_q.filter(filters)

    q = q.order_by(order_by, agreements.agreement_uuid)
    total_count = deps._to_int(cast(object, count_q.scalar()))
    offset = (page - 1) * page_size
    items = cast(list[object], q.offset(offset).limit(page_size).all())
    meta = deps._pagination_metadata(total_count=total_count, page=page, page_size=page_size)

    results: list[dict[str, object]] = []
    for row in items:
        row_map = deps._row_mapping_as_dict(cast(object, row))
        verified_value = row_map.get("verified")
        results.append(
            {
                "agreement_uuid": _json_compatible_value(row_map.get("agreement_uuid")),
                "year": _json_compatible_value(row_map.get("year")),
                "target": _json_compatible_value(row_map.get("target")),
                "acquirer": _json_compatible_value(row_map.get("acquirer")),
                "filing_date": _json_compatible_value(row_map.get("filing_date")),
                "url": _json_compatible_value(row_map.get("url")),
                "verified": bool(verified_value) if verified_value is not None else False,
                "section_count": _json_compatible_value(row_map.get("section_count")),
            }
        )

    response = {
        "results": results,
        "returned_count": len(results),
        "count_metadata": _count_metadata_payload(
            mode="exact",
            method="query_count",
            planning_reliability="high",
            exact_count_requested=False,
        ),
        "interpretation": _agreement_filter_interpretation(
            cast(AgreementsBulkArgsPayload, cast(object, parsed_args)),
            query=query,
        ),
        **meta,
    }
    return McpToolResult(
        text=f"Found {len(results)} agreement(s) on page {page}.",
        structured_content=response,
    )


def _get_agreement(
    deps: AgreementsDeps,
    *,
    principal: McpPrincipal,
    payload: dict[str, object],
) -> McpToolResult:
    _require_scope(principal, "agreements:read")
    parsed_args = cast(
        AgreementArgsPayload,
        cast(object, _validate_payload(McpAgreementArgsSchema(), payload)),
    )
    agreement_uuid = cast(str, payload["agreement_uuid"]).strip()
    focus_section_uuid = parsed_args.get("focus_section_uuid")
    if focus_section_uuid is not None:
        focus_section_uuid = focus_section_uuid.strip()
        if not deps._SECTION_ID_RE.match(focus_section_uuid):
            abort(400, description="Invalid focus_section_uuid.")
    neighbor_sections_int = parsed_args["neighbor_sections"]
    allow_fulltext = "agreements:read_fulltext" in principal.scopes

    agreements = deps.Agreements
    xml = deps.XML
    db = deps.db
    year_expr = deps._agreement_year_expr().label("year")
    row = (
        db.session.query(
            year_expr,
            agreements.target,
            agreements.acquirer,
            agreements.filing_date,
            agreements.prob_filing,
            agreements.filing_company_name,
            agreements.filing_company_cik,
            agreements.form_type,
            agreements.exhibit_type,
            agreements.transaction_price_total,
            agreements.transaction_price_stock,
            agreements.transaction_price_cash,
            agreements.transaction_price_assets,
            agreements.transaction_consideration,
            agreements.target_type,
            agreements.acquirer_type,
            agreements.target_industry,
            agreements.acquirer_industry,
            agreements.announce_date,
            agreements.close_date,
            agreements.deal_status,
            agreements.attitude,
            agreements.deal_type,
            agreements.purpose,
            agreements.target_pe,
            agreements.acquirer_pe,
            agreements.url,
            xml.xml,
        )
        .join(xml, deps._agreement_latest_xml_join_condition())
        .filter(agreements.agreement_uuid == agreement_uuid)
        .first()
    )
    if row is None:
        abort(404)

    row_map = deps._row_mapping_as_dict(cast(object, row))
    xml_content_obj = row_map.get("xml")
    xml_content = xml_content_obj if isinstance(xml_content_obj, str) else ""
    response = {
        "year": _json_compatible_value(row_map.get("year")),
        "target": _json_compatible_value(row_map.get("target")),
        "acquirer": _json_compatible_value(row_map.get("acquirer")),
        "filing_date": _json_compatible_value(row_map.get("filing_date")),
        "prob_filing": _json_compatible_value(row_map.get("prob_filing")),
        "filing_company_name": _json_compatible_value(row_map.get("filing_company_name")),
        "filing_company_cik": _json_compatible_value(row_map.get("filing_company_cik")),
        "form_type": _json_compatible_value(row_map.get("form_type")),
        "exhibit_type": _json_compatible_value(row_map.get("exhibit_type")),
        "transaction_price_total": _json_compatible_value(row_map.get("transaction_price_total")),
        "transaction_price_stock": _json_compatible_value(row_map.get("transaction_price_stock")),
        "transaction_price_cash": _json_compatible_value(row_map.get("transaction_price_cash")),
        "transaction_price_assets": _json_compatible_value(row_map.get("transaction_price_assets")),
        "transaction_consideration": _json_compatible_value(row_map.get("transaction_consideration")),
        "target_type": _json_compatible_value(row_map.get("target_type")),
        "acquirer_type": _json_compatible_value(row_map.get("acquirer_type")),
        "target_industry": _json_compatible_value(row_map.get("target_industry")),
        "acquirer_industry": _json_compatible_value(row_map.get("acquirer_industry")),
        "announce_date": _json_compatible_value(row_map.get("announce_date")),
        "close_date": _json_compatible_value(row_map.get("close_date")),
        "deal_status": _json_compatible_value(row_map.get("deal_status")),
        "attitude": _json_compatible_value(row_map.get("attitude")),
        "deal_type": _json_compatible_value(row_map.get("deal_type")),
        "purpose": _json_compatible_value(row_map.get("purpose")),
        "target_pe": _json_compatible_value(row_map.get("target_pe")),
        "acquirer_pe": _json_compatible_value(row_map.get("acquirer_pe")),
        "url": _json_compatible_value(row_map.get("url")),
    }
    if allow_fulltext:
        response["xml"] = xml_content
        response["is_redacted"] = False
    else:
        response["xml"] = deps._redact_agreement_xml(
            xml_content,
            focus_section_uuid=focus_section_uuid,
            neighbor_sections=neighbor_sections_int,
        )
        response["is_redacted"] = True
    return McpToolResult(
        text=f"Fetched agreement {agreement_uuid}.",
        structured_content=response,
    )


def _get_section(
    deps: AgreementsDeps,
    *,
    principal: McpPrincipal,
    payload: dict[str, object],
) -> McpToolResult:
    _require_scope(principal, "sections:search")
    parsed_args = _validate_payload(McpSectionArgsSchema(), payload)
    section_uuid = cast(str, parsed_args["section_uuid"]).strip()
    if not deps._SECTION_ID_RE.match(section_uuid):
        abort(400, description="Invalid section_uuid.")

    sections = deps.Sections
    xml = deps.XML
    row = (
        deps.db.session.query(
            sections.agreement_uuid.label("agreement_uuid"),
            sections.section_uuid.label("section_uuid"),
            deps._coalesced_section_standard_ids().label("section_standard_ids"),
            sections.xml_content.label("xml_content"),
            sections.article_title.label("article_title"),
            sections.section_title.label("section_title"),
        )
        .join(xml, deps._section_latest_xml_join_condition())
        .filter(sections.section_uuid == section_uuid)
        .first()
    )
    if row is None:
        abort(404)

    row_map = deps._row_mapping_as_dict(cast(object, row))
    response = {
        "agreement_uuid": row_map.get("agreement_uuid"),
        "section_uuid": row_map.get("section_uuid"),
        "standard_id": deps._parse_section_standard_ids(row_map.get("section_standard_ids")),
        "xml": row_map.get("xml_content"),
        "article_title": row_map.get("article_title"),
        "section_title": row_map.get("section_title"),
    }
    return McpToolResult(
        text=f"Fetched section {section_uuid}.",
        structured_content=response,
    )


def _suggest_clause_families(
    deps: ReferenceDataDeps,
    *,
    principal: McpPrincipal,
    payload: dict[str, object],
) -> McpToolResult:
    _require_scope(principal, "sections:search")
    parsed_args = _validate_payload(
        _schema_from_fields(
            "McpSuggestClauseFamiliesArgs",
            {
                "concept": ma_fields.Str(required=True, validate=validate.Length(min=1)),
                "taxonomy": ma_fields.Str(load_default="clauses", validate=validate.OneOf(["clauses", "tax_clauses"])),
                "top_k": ma_fields.Int(load_default=5, validate=validate.Range(min=1, max=10)),
            },
        ),
        payload,
    )
    concept = cast(str, parsed_args["concept"]).strip()
    taxonomy = cast(str, parsed_args["taxonomy"])
    top_k = cast(int, parsed_args["top_k"])
    matches = _ranked_taxonomy_matches(deps=deps, concept=concept, taxonomy=taxonomy, top_k=top_k)
    response = {
        "concept": concept,
        "taxonomy": taxonomy,
        "matches": matches,
        "returned_count": len(matches),
    }
    return McpToolResult(
        text=f"Suggested {len(matches)} clause family match(es) for '{concept}'.",
        structured_content=response,
    )


def _get_section_snippet(
    deps: AgreementsDeps,
    *,
    principal: McpPrincipal,
    payload: dict[str, object],
) -> McpToolResult:
    _require_scope(principal, "sections:search")
    parsed_args = _validate_payload(
        _schema_from_fields(
            "McpSectionSnippetArgs",
            {
                "section_uuid": ma_fields.Str(required=True, validate=validate.Length(min=1)),
                "focus_terms": ma_fields.List(ma_fields.Str(), load_default=[]),
                "max_chars": ma_fields.Int(load_default=400, validate=validate.Range(min=120, max=1200)),
            },
        ),
        payload,
    )
    section_uuid = cast(str, parsed_args["section_uuid"]).strip()
    if not deps._SECTION_ID_RE.match(section_uuid):
        abort(400, description="Invalid section_uuid.")
    row = (
        deps.db.session.query(
            deps.Sections.agreement_uuid.label("agreement_uuid"),
            deps.Sections.section_uuid.label("section_uuid"),
            deps.Sections.article_title.label("article_title"),
            deps.Sections.section_title.label("section_title"),
            deps._coalesced_section_standard_ids().label("section_standard_ids"),
            deps.Sections.xml_content.label("xml_content"),
        )
        .filter(deps.Sections.section_uuid == section_uuid)
        .first()
    )
    if row is None:
        abort(404)

    row_map = deps._row_mapping_as_dict(cast(object, row))
    xml_text = _extract_text_from_xml(row_map.get("xml_content"))
    focus_terms = [term for term in cast(list[str], parsed_args["focus_terms"]) if term.strip()]
    snippet, matched_terms = _focused_snippet(
        xml_text,
        focus_terms=focus_terms,
        max_chars=cast(int, parsed_args["max_chars"]),
    )
    response = {
        "agreement_uuid": row_map.get("agreement_uuid"),
        "section_uuid": section_uuid,
        "standard_id": deps._parse_section_standard_ids(row_map.get("section_standard_ids")),
        "article_title": row_map.get("article_title"),
        "section_title": row_map.get("section_title"),
        "snippet": snippet,
        "matched_terms": matched_terms,
        "source_length": len(xml_text),
        "monetary_values": _extract_monetary_values(xml_text),
    }
    return McpToolResult(
        text=f"Returned a focused snippet for section {section_uuid}.",
        structured_content=response,
    )


def _get_section_snippets_batch(
    deps: AgreementsDeps,
    *,
    principal: McpPrincipal,
    payload: dict[str, object],
) -> McpToolResult:
    _require_scope(principal, "sections:search")
    parsed_args = _validate_payload(
        _schema_from_fields(
            "McpBatchSectionSnippetsArgs",
            {
                "section_uuids": ma_fields.List(
                    ma_fields.Str(),
                    required=True,
                    validate=validate.Length(min=1, max=20),
                ),
                "focus_terms": ma_fields.List(ma_fields.Str(), load_default=[]),
                "max_chars": ma_fields.Int(load_default=400, validate=validate.Range(min=120, max=1200)),
            },
        ),
        payload,
    )
    section_uuids = [s.strip() for s in cast(list[str], parsed_args["section_uuids"]) if s.strip()]
    if not section_uuids:
        abort(400, description="No valid section_uuids provided.")
    for suuid in section_uuids:
        if not deps._SECTION_ID_RE.match(suuid):
            abort(400, description=f"Invalid section_uuid: {suuid}")
    focus_terms = [t for t in cast(list[str], parsed_args["focus_terms"]) if t.strip()]
    max_chars = cast(int, parsed_args["max_chars"])

    rows = cast(
        list[object],
        deps.db.session.query(
            deps.Sections.agreement_uuid.label("agreement_uuid"),
            deps.Sections.section_uuid.label("section_uuid"),
            deps.Sections.article_title.label("article_title"),
            deps.Sections.section_title.label("section_title"),
            deps._coalesced_section_standard_ids().label("section_standard_ids"),
            deps.Sections.xml_content.label("xml_content"),
        )
        .filter(deps.Sections.section_uuid.in_(section_uuids))
        .all(),
    )
    row_by_uuid: dict[str, dict[str, object]] = {}
    for row in rows:
        row_map = deps._row_mapping_as_dict(cast(object, row))
        suuid = row_map.get("section_uuid")
        if isinstance(suuid, str):
            row_by_uuid[suuid] = row_map

    results: list[dict[str, object]] = []
    for suuid in section_uuids:
        row_map = row_by_uuid.get(suuid)
        if row_map is None:
            continue
        xml_text = _extract_text_from_xml(row_map.get("xml_content"))
        snippet, matched_terms = _focused_snippet(xml_text, focus_terms=focus_terms, max_chars=max_chars)
        results.append(
            {
                "agreement_uuid": row_map.get("agreement_uuid"),
                "section_uuid": suuid,
                "standard_id": deps._parse_section_standard_ids(row_map.get("section_standard_ids")),
                "article_title": row_map.get("article_title"),
                "section_title": row_map.get("section_title"),
                "snippet": snippet,
                "matched_terms": matched_terms,
                "source_length": len(xml_text),
                "monetary_values": _extract_monetary_values(xml_text),
            }
        )

    response: dict[str, object] = {
        "results": results,
        "returned_count": len(results),
    }
    return McpToolResult(
        text=f"Returned snippets for {len(results)} section(s).",
        structured_content=response,
    )


def _get_sections_batch(
    deps: AgreementsDeps,
    *,
    principal: McpPrincipal,
    payload: dict[str, object],
) -> McpToolResult:
    _require_scope(principal, "sections:search")
    parsed_args = _validate_payload(
        _schema_from_fields(
            "McpBatchSectionsArgs",
            {
                "section_uuids": ma_fields.List(
                    ma_fields.Str(),
                    required=True,
                    validate=validate.Length(min=1, max=10),
                ),
                "max_xml_chars": ma_fields.Int(load_default=10000, allow_none=True, validate=validate.Range(min=500, max=20000)),
            },
        ),
        payload,
    )
    section_uuids = [s.strip() for s in cast(list[str], parsed_args["section_uuids"]) if s.strip()]
    if not section_uuids:
        abort(400, description="No valid section_uuids provided.")
    for suuid in section_uuids:
        if not deps._SECTION_ID_RE.match(suuid):
            abort(400, description=f"Invalid section_uuid: {suuid}")

    sections = deps.Sections
    xml = deps.XML
    agreements = deps.Agreements
    db = deps.db
    year_expr = deps._agreement_year_expr().label("year")

    rows = cast(
        list[object],
        db.session.query(
            sections.agreement_uuid.label("agreement_uuid"),
            sections.section_uuid.label("section_uuid"),
            deps._coalesced_section_standard_ids().label("section_standard_ids"),
            sections.article_title.label("article_title"),
            sections.section_title.label("section_title"),
            sections.xml_content.label("xml_content"),
            year_expr,
            agreements.target.label("target"),
            agreements.acquirer.label("acquirer"),
            agreements.filing_date.label("filing_date"),
            agreements.transaction_price_total.label("transaction_price_total"),
        )
        .join(xml, deps._section_latest_xml_join_condition())
        .join(agreements, agreements.agreement_uuid == sections.agreement_uuid)
        .filter(sections.section_uuid.in_(section_uuids))
        .all(),
    )

    max_xml_chars = cast(int | None, parsed_args["max_xml_chars"])

    row_by_uuid: dict[str, dict[str, object]] = {}
    for row in rows:
        row_map = deps._row_mapping_as_dict(cast(object, row))
        suuid = row_map.get("section_uuid")
        if isinstance(suuid, str):
            row_by_uuid[suuid] = row_map

    results: list[dict[str, object]] = []
    for suuid in section_uuids:
        row_map = row_by_uuid.get(suuid)
        if row_map is None:
            continue
        xml_content = row_map.get("xml_content")
        xml_text = _extract_text_from_xml(xml_content)
        xml_truncated = False
        if max_xml_chars is not None and isinstance(xml_content, str) and len(xml_content) > max_xml_chars:
            xml_content = xml_content[:max_xml_chars]
            xml_truncated = True
        results.append(
            {
                "agreement_uuid": row_map.get("agreement_uuid"),
                "section_uuid": suuid,
                "standard_id": deps._parse_section_standard_ids(row_map.get("section_standard_ids")),
                "article_title": row_map.get("article_title"),
                "section_title": row_map.get("section_title"),
                "xml": xml_content,
                "xml_truncated": xml_truncated,
                "target": row_map.get("target"),
                "acquirer": row_map.get("acquirer"),
                "year": _json_compatible_value(row_map.get("year")),
                "filing_date": _json_compatible_value(row_map.get("filing_date")),
                "transaction_price_total": _json_compatible_value(row_map.get("transaction_price_total")),
                "monetary_values": _extract_monetary_values(xml_text),
            }
        )

    response: dict[str, object] = {
        "results": results,
        "returned_count": len(results),
    }
    return McpToolResult(
        text=f"Returned {len(results)} full section(s).",
        structured_content=response,
    )


def _list_agreement_sections(
    deps: SectionsServiceDeps,
    *,
    principal: McpPrincipal,
    payload: dict[str, object],
) -> McpToolResult:
    _require_scope(principal, "sections:search")
    parsed_args = _validate_payload(McpListAgreementSectionsArgsSchema(), payload)
    agreement_uuid = cast(str, parsed_args["agreement_uuid"]).strip()
    if agreement_uuid == "":
        abort(400, description="Invalid agreement_uuid.")

    page = _normalized_page(cast(int, parsed_args["page"]))
    page_size = _normalized_page_size(cast(int, parsed_args["page_size"]))
    standard_ids = [value for value in cast(list[str], parsed_args["standard_id"]) if value]
    include_standard_ids = cast(bool, parsed_args["include_standard_ids"])
    sort_by = cast(str, parsed_args["sort_by"])
    sort_direction = cast(str, parsed_args["sort_direction"])

    latest = deps.LatestSectionsSearch
    db = deps.db
    q = db.session.query(
        latest.section_uuid.label("section_uuid"),
        latest.agreement_uuid.label("agreement_uuid"),
        latest.section_standard_ids.label("section_standard_ids"),
        latest.article_title.label("article_title"),
        latest.section_title.label("section_title"),
        latest.target.label("target"),
        latest.acquirer.label("acquirer"),
        latest.filing_date.label("filing_date"),
        latest.verified.label("verified"),
    ).filter(latest.agreement_uuid == agreement_uuid)

    if standard_ids:
        standard_ids_key = tuple(sorted(set(standard_ids)))
        expanded_standard_ids = list(deps._expand_taxonomy_standard_ids_cached(standard_ids_key))
        if expanded_standard_ids:
            q = q.filter(deps._standard_id_filter_expr(expanded_standard_ids))

    if sort_by == "document_order":
        sections_model = deps.Sections
        q = q.join(sections_model, sections_model.section_uuid == latest.section_uuid)
        if sort_direction == "desc":
            q = q.order_by(desc(sections_model.article_order), desc(sections_model.section_order), desc(latest.section_uuid))
        else:
            q = q.order_by(asc(sections_model.article_order), asc(sections_model.section_order), asc(latest.section_uuid))
    else:
        sort_column_map = {
            "article_title": latest.article_title,
            "section_title": latest.section_title,
            "section_uuid": latest.section_uuid,
        }
        primary_sort = sort_column_map[sort_by]
        if sort_direction == "desc":
            q = q.order_by(desc(primary_sort), desc(latest.section_uuid))
        else:
            q = q.order_by(asc(primary_sort), asc(latest.section_uuid))

    total_agreement_sections = deps._to_int(
        cast(
            object,
            deps.db.session.query(func.count(latest.section_uuid))
            .filter(latest.agreement_uuid == agreement_uuid)
            .scalar(),
        )
    )
    total_count = deps._to_int(cast(object, q.order_by(None).count()))
    offset = (page - 1) * page_size
    rows = cast(list[object], q.offset(offset).limit(page_size).all())
    meta = deps._pagination_metadata(total_count=total_count, page=page, page_size=page_size)

    results: list[dict[str, object]] = []
    for row in rows:
        row_map = deps._row_mapping_as_dict(cast(object, row))
        entry: dict[str, object] = {
            "id": row_map.get("section_uuid"),
            "agreement_uuid": row_map.get("agreement_uuid"),
            "section_uuid": row_map.get("section_uuid"),
            "article_title": row_map.get("article_title"),
            "section_title": row_map.get("section_title"),
            "target": row_map.get("target"),
            "acquirer": row_map.get("acquirer"),
            "year": deps._year_from_filing_date_value(row_map.get("filing_date")),
            "verified": bool(row_map.get("verified")) if row_map.get("verified") is not None else False,
        }
        if include_standard_ids:
            entry["standard_id"] = deps._parse_section_standard_ids(row_map.get("section_standard_ids"))
        results.append(entry)

    response = {
        "agreement_uuid": agreement_uuid,
        "total_agreement_sections": total_agreement_sections,
        "results": results,
        "returned_count": len(results),
        **meta,
    }
    return McpToolResult(
        text=f"Returned {len(results)} section(s) for agreement {agreement_uuid}.",
        structured_content=response,
    )


def _list_agreement_sections_batch(
    deps: SectionsServiceDeps,
    *,
    principal: McpPrincipal,
    payload: dict[str, object],
) -> McpToolResult:
    _require_scope(principal, "sections:search")
    parsed_args = _validate_payload(McpBatchAgreementSectionsArgsSchema(), payload)
    agreement_uuids = [u.strip() for u in cast(list[str], parsed_args["agreement_uuids"]) if u.strip()]
    if not agreement_uuids:
        abort(400, description="No valid agreement_uuids provided.")

    standard_ids = [value for value in cast(list[str], parsed_args["standard_id"]) if value]
    include_standard_ids = cast(bool, parsed_args["include_standard_ids"])
    sort_by = cast(str, parsed_args["sort_by"])
    sort_direction = cast(str, parsed_args["sort_direction"])

    latest = deps.LatestSectionsSearch
    db = deps.db
    q = db.session.query(
        latest.section_uuid.label("section_uuid"),
        latest.agreement_uuid.label("agreement_uuid"),
        latest.section_standard_ids.label("section_standard_ids"),
        latest.article_title.label("article_title"),
        latest.section_title.label("section_title"),
        latest.target.label("target"),
        latest.acquirer.label("acquirer"),
        latest.filing_date.label("filing_date"),
        latest.verified.label("verified"),
    ).filter(latest.agreement_uuid.in_(agreement_uuids))

    if standard_ids:
        standard_ids_key = tuple(sorted(set(standard_ids)))
        expanded_standard_ids = list(deps._expand_taxonomy_standard_ids_cached(standard_ids_key))
        if expanded_standard_ids:
            q = q.filter(deps._standard_id_filter_expr(expanded_standard_ids))

    if sort_by == "document_order":
        sections_model = deps.Sections
        q = q.join(sections_model, sections_model.section_uuid == latest.section_uuid)
        if sort_direction == "desc":
            q = q.order_by(desc(latest.agreement_uuid), desc(sections_model.article_order), desc(sections_model.section_order))
        else:
            q = q.order_by(asc(latest.agreement_uuid), asc(sections_model.article_order), asc(sections_model.section_order))
    else:
        sort_column_map = {
            "article_title": latest.article_title,
            "section_title": latest.section_title,
            "section_uuid": latest.section_uuid,
        }
        primary_sort = sort_column_map[sort_by]
        if sort_direction == "desc":
            q = q.order_by(desc(latest.agreement_uuid), desc(primary_sort), desc(latest.section_uuid))
        else:
            q = q.order_by(asc(latest.agreement_uuid), asc(primary_sort), asc(latest.section_uuid))

    rows = cast(list[object], q.all())

    # unfiltered total sections per agreement (independent of standard_id filter)
    unfiltered_rows = cast(
        list[object],
        deps.db.session.query(
            latest.agreement_uuid.label("agreement_uuid"),
            func.count(latest.section_uuid).label("total_count"),
        )
        .filter(latest.agreement_uuid.in_(agreement_uuids))
        .group_by(latest.agreement_uuid)
        .all(),
    )
    unfiltered_counts: dict[str, int] = {
        cast(str, deps._row_mapping_as_dict(cast(object, r)).get("agreement_uuid")): deps._to_int(
            cast(object, deps._row_mapping_as_dict(cast(object, r)).get("total_count"))
        )
        for r in unfiltered_rows
    }

    grouped: dict[str, list[dict[str, object]]] = {uuid: [] for uuid in agreement_uuids}
    for row in rows:
        row_map = deps._row_mapping_as_dict(cast(object, row))
        uuid = row_map.get("agreement_uuid")
        if isinstance(uuid, str) and uuid in grouped:
            entry: dict[str, object] = {
                "id": row_map.get("section_uuid"),
                "section_uuid": row_map.get("section_uuid"),
                "agreement_uuid": uuid,
                "article_title": row_map.get("article_title"),
                "section_title": row_map.get("section_title"),
                "target": row_map.get("target"),
                "acquirer": row_map.get("acquirer"),
                "year": deps._year_from_filing_date_value(row_map.get("filing_date")),
                "verified": bool(row_map.get("verified")) if row_map.get("verified") is not None else False,
            }
            if include_standard_ids:
                entry["standard_id"] = deps._parse_section_standard_ids(row_map.get("section_standard_ids"))
            grouped[uuid].append(entry)

    per_agreement = [
        {
            "agreement_uuid": uuid,
            "total_agreement_sections": unfiltered_counts.get(uuid, 0),
            "sections": grouped[uuid],
            "section_count": len(grouped[uuid]),
        }
        for uuid in agreement_uuids
    ]
    total_sections = sum(len(grouped[uuid]) for uuid in agreement_uuids)
    response = {
        "results": per_agreement,
        "returned_agreement_count": len(agreement_uuids),
        "total_section_count": total_sections,
    }
    return McpToolResult(
        text=f"Returned sections for {len(agreement_uuids)} agreement(s), {total_sections} section(s) total.",
        structured_content=response,
    )


def _search_sections(
    deps: SectionsServiceDeps,
    *,
    principal: McpPrincipal,
    payload: dict[str, object],
    agreements_deps: AgreementsDeps,
) -> McpToolResult:
    _require_scope(principal, "sections:search")
    parsed_args = cast(
        SectionsArgsPayload,
        cast(object, _validate_payload(SectionsArgsSchema(), payload)),
    )
    include_xml = parsed_args["include_xml"]
    response = run_sections(deps, ctx=principal.access_context, parsed_args=parsed_args)
    results = cast(list[dict[str, object]], response.get("results", []))

    if not include_xml:
        for item in results:
            item.pop("xml", None)

    all_standard_ids: set[str] = set()
    for item in results:
        for sid in cast(list[str], item.get("standard_id", [])):
            if isinstance(sid, str):
                all_standard_ids.add(sid)
    if all_standard_ids:
        db = deps.db
        label_rows = cast(
            list[tuple[object, object]],
            db.session.query(
                cast(Any, agreements_deps.TaxonomyL1).standard_id,
                cast(Any, agreements_deps.TaxonomyL1).label,
            )
            .filter(cast(Any, agreements_deps.TaxonomyL1).standard_id.in_(all_standard_ids))
            .all()
            + db.session.query(
                cast(Any, agreements_deps.TaxonomyL2).standard_id,
                cast(Any, agreements_deps.TaxonomyL2).label,
            )
            .filter(cast(Any, agreements_deps.TaxonomyL2).standard_id.in_(all_standard_ids))
            .all()
            + db.session.query(
                cast(Any, agreements_deps.TaxonomyL3).standard_id,
                cast(Any, agreements_deps.TaxonomyL3).label,
            )
            .filter(cast(Any, agreements_deps.TaxonomyL3).standard_id.in_(all_standard_ids))
            .all(),
        )
        standard_id_labels: dict[str, str] = {
            sid: label
            for sid, label in label_rows
            if isinstance(sid, str) and isinstance(label, str)
        }
        response["standard_id_labels"] = standard_id_labels

    count = len(results)
    return McpToolResult(
        text=f"Returned {count} section(s).",
        structured_content=response,
    )


def _get_agreement_tax_clauses(
    deps: AgreementsDeps,
    *,
    principal: McpPrincipal,
    payload: dict[str, object],
) -> McpToolResult:
    _require_scope(principal, "agreements:read")
    agreement_uuid = cast(str, _validate_payload(McpAgreementArgsSchema(), payload)["agreement_uuid"]).strip()
    if agreement_uuid == "":
        abort(400, description="Invalid agreement_uuid.")
    clauses = _tax_clause_rows(deps, agreement_uuid=agreement_uuid)
    response = {"agreement_uuid": agreement_uuid, "clauses": clauses, "returned_count": len(clauses)}
    return McpToolResult(
        text=f"Returned {len(clauses)} tax clause(s) for agreement {agreement_uuid}.",
        structured_content=response,
    )


def _get_section_tax_clauses(
    deps: AgreementsDeps,
    *,
    principal: McpPrincipal,
    payload: dict[str, object],
) -> McpToolResult:
    _require_scope(principal, "agreements:read")
    section_uuid = cast(str, _validate_payload(McpSectionArgsSchema(), payload)["section_uuid"]).strip()
    if not deps._SECTION_ID_RE.match(section_uuid):
        abort(400, description="Invalid section_uuid.")
    clauses = _tax_clause_rows(deps, section_uuid=section_uuid)
    response = {"section_uuid": section_uuid, "clauses": clauses, "returned_count": len(clauses)}
    return McpToolResult(
        text=f"Returned {len(clauses)} tax clause(s) for section {section_uuid}.",
        structured_content=response,
    )


def _list_filter_options(
    deps: AgreementsDeps,
    *,
    principal: McpPrincipal,
    payload: dict[str, object],
) -> McpToolResult:
    _require_scope(principal, "sections:search")
    requested_fields = cast(list[str], _validate_payload(McpFilterOptionsArgsSchema(), payload)["fields"])
    selected_fields = tuple(requested_fields) if requested_fields else _FILTER_OPTIONS_FIELDS
    db = deps.db
    agreements = deps.Agreements
    schema_prefix = deps._schema_prefix
    xml_eligible = (
        "EXISTS ("
        "  SELECT 1 FROM {t}xml x "
        "  WHERE x.agreement_uuid = a.agreement_uuid "
        "    AND (x.status IS NULL OR x.status = 'verified')"
        ")"
    ).format(t=schema_prefix())
    has_sections = (
        "EXISTS ("
        "  SELECT 1 FROM {t}sections s "
        "  WHERE s.agreement_uuid = a.agreement_uuid"
        ")"
    ).format(t=schema_prefix())
    is_public_eligible = (
        "NOT (COALESCE(a.gated, 0) = 1 AND COALESCE(a.verified, 0) = 0)"
        if "gated" in agreements.__table__.c
        else "1 = 1"
    )

    payload_out: dict[str, object] = {}
    if "targets" in selected_fields:
        payload_out["targets"] = [
            cast(str, row[0])
            for row in db.session.execute(
                text(
                    f"""
                    SELECT DISTINCT a.target
                    FROM {schema_prefix()}agreements a
                    WHERE a.target IS NOT NULL
                      AND a.target <> ''
                      AND {is_public_eligible}
                      AND {has_sections}
                      AND {xml_eligible}
                    ORDER BY a.target
                    """
                )
            ).fetchall()
        ]
    if "acquirers" in selected_fields:
        payload_out["acquirers"] = [
            cast(str, row[0])
            for row in db.session.execute(
                text(
                    f"""
                    SELECT DISTINCT a.acquirer
                    FROM {schema_prefix()}agreements a
                    WHERE a.acquirer IS NOT NULL
                      AND a.acquirer <> ''
                      AND {is_public_eligible}
                      AND {has_sections}
                      AND {xml_eligible}
                    ORDER BY a.acquirer
                    """
                )
            ).fetchall()
        ]
    if "transaction_price_totals" in selected_fields:
        payload_out["transaction_price_totals"] = list(_TRANSACTION_PRICE_BUCKET_OPTIONS)
    if "transaction_price_stocks" in selected_fields:
        payload_out["transaction_price_stocks"] = list(_TRANSACTION_PRICE_BUCKET_OPTIONS)
    if "transaction_price_cashes" in selected_fields:
        payload_out["transaction_price_cashes"] = list(_TRANSACTION_PRICE_BUCKET_OPTIONS)
    if "transaction_price_assets" in selected_fields:
        payload_out["transaction_price_assets"] = list(_TRANSACTION_PRICE_BUCKET_OPTIONS)
    if "transaction_considerations" in selected_fields:
        payload_out["transaction_considerations"] = [
            cast(str, row[0])
            for row in db.session.execute(
                text(
                    f"""
                    SELECT DISTINCT a.transaction_consideration
                    FROM {schema_prefix()}agreements a
                    WHERE a.transaction_consideration IS NOT NULL
                      AND a.transaction_consideration <> ''
                      AND {is_public_eligible}
                      AND {has_sections}
                      AND {xml_eligible}
                    ORDER BY a.transaction_consideration
                    """
                )
            ).fetchall()
        ]
    if "target_types" in selected_fields:
        payload_out["target_types"] = [
            cast(str, row[0])
            for row in db.session.execute(
                text(
                    f"""
                    SELECT DISTINCT a.target_type
                    FROM {schema_prefix()}agreements a
                    WHERE a.target_type IS NOT NULL
                      AND a.target_type <> ''
                      AND {is_public_eligible}
                      AND {has_sections}
                      AND {xml_eligible}
                    ORDER BY a.target_type
                    """
                )
            ).fetchall()
        ]
    if "acquirer_types" in selected_fields:
        payload_out["acquirer_types"] = [
            cast(str, row[0])
            for row in db.session.execute(
                text(
                    f"""
                    SELECT DISTINCT a.acquirer_type
                    FROM {schema_prefix()}agreements a
                    WHERE a.acquirer_type IS NOT NULL
                      AND a.acquirer_type <> ''
                      AND {is_public_eligible}
                      AND {has_sections}
                      AND {xml_eligible}
                    ORDER BY a.acquirer_type
                    """
                )
            ).fetchall()
        ]
    if "target_counsels" in selected_fields:
        payload_out["target_counsels"] = [
            cast(str, row[0])
            for row in db.session.execute(
                text(
                    f"""
                    SELECT DISTINCT c.canonical_name
                    FROM {schema_prefix()}agreement_counsel ac
                    JOIN {schema_prefix()}counsel c
                      ON c.counsel_id = ac.counsel_id
                    JOIN {schema_prefix()}agreements a
                      ON a.agreement_uuid = ac.agreement_uuid
                    WHERE ac.side = 'target'
                      AND c.canonical_name IS NOT NULL
                      AND c.canonical_name <> ''
                      AND {is_public_eligible}
                      AND {has_sections}
                      AND {xml_eligible}
                    ORDER BY c.canonical_name
                    """
                )
            ).fetchall()
        ]
    if "acquirer_counsels" in selected_fields:
        payload_out["acquirer_counsels"] = [
            cast(str, row[0])
            for row in db.session.execute(
                text(
                    f"""
                    SELECT DISTINCT c.canonical_name
                    FROM {schema_prefix()}agreement_counsel ac
                    JOIN {schema_prefix()}counsel c
                      ON c.counsel_id = ac.counsel_id
                    JOIN {schema_prefix()}agreements a
                      ON a.agreement_uuid = ac.agreement_uuid
                    WHERE ac.side = 'acquirer'
                      AND c.canonical_name IS NOT NULL
                      AND c.canonical_name <> ''
                      AND {is_public_eligible}
                      AND {has_sections}
                      AND {xml_eligible}
                    ORDER BY c.canonical_name
                    """
                )
            ).fetchall()
        ]
    if "target_industries" in selected_fields:
        payload_out["target_industries"] = [
            cast(str, row[0])
            for row in db.session.execute(
                text(
                    f"""
                    SELECT DISTINCT a.target_industry
                    FROM {schema_prefix()}agreements a
                    WHERE a.target_industry IS NOT NULL
                      AND a.target_industry <> ''
                      AND {is_public_eligible}
                      AND {has_sections}
                      AND {xml_eligible}
                    ORDER BY a.target_industry
                    """
                )
            ).fetchall()
        ]
    if "acquirer_industries" in selected_fields:
        payload_out["acquirer_industries"] = [
            cast(str, row[0])
            for row in db.session.execute(
                text(
                    f"""
                    SELECT DISTINCT a.acquirer_industry
                    FROM {schema_prefix()}agreements a
                    WHERE a.acquirer_industry IS NOT NULL
                      AND a.acquirer_industry <> ''
                      AND {is_public_eligible}
                      AND {has_sections}
                      AND {xml_eligible}
                    ORDER BY a.acquirer_industry
                    """
                )
            ).fetchall()
        ]
    if "deal_statuses" in selected_fields:
        payload_out["deal_statuses"] = [
            cast(str, row[0])
            for row in db.session.execute(
                text(
                    f"""
                    SELECT DISTINCT a.deal_status
                    FROM {schema_prefix()}agreements a
                    WHERE a.deal_status IS NOT NULL
                      AND a.deal_status <> ''
                      AND {is_public_eligible}
                      AND {has_sections}
                      AND {xml_eligible}
                    ORDER BY a.deal_status
                    """
                )
            ).fetchall()
        ]
    if "attitudes" in selected_fields:
        payload_out["attitudes"] = [
            cast(str, row[0])
            for row in db.session.execute(
                text(
                    f"""
                    SELECT DISTINCT a.attitude
                    FROM {schema_prefix()}agreements a
                    WHERE a.attitude IS NOT NULL
                      AND a.attitude <> ''
                      AND {is_public_eligible}
                      AND {has_sections}
                      AND {xml_eligible}
                    ORDER BY a.attitude
                    """
                )
            ).fetchall()
        ]
    if "deal_types" in selected_fields:
        payload_out["deal_types"] = [
            cast(str, row[0])
            for row in db.session.execute(
                text(
                    f"""
                    SELECT DISTINCT a.deal_type
                    FROM {schema_prefix()}agreements a
                    WHERE a.deal_type IS NOT NULL
                      AND a.deal_type <> ''
                      AND {is_public_eligible}
                      AND {has_sections}
                      AND {xml_eligible}
                    ORDER BY a.deal_type
                    """
                )
            ).fetchall()
        ]
    if "purposes" in selected_fields:
        payload_out["purposes"] = [
            cast(str, row[0])
            for row in db.session.execute(
                text(
                    f"""
                    SELECT DISTINCT a.purpose
                    FROM {schema_prefix()}agreements a
                    WHERE a.purpose IS NOT NULL
                      AND a.purpose <> ''
                      AND {is_public_eligible}
                      AND {has_sections}
                      AND {xml_eligible}
                    ORDER BY a.purpose
                    """
                )
            ).fetchall()
        ]
    if "target_pes" in selected_fields:
        payload_out["target_pes"] = ["true", "false"]
    if "acquirer_pes" in selected_fields:
        payload_out["acquirer_pes"] = ["true", "false"]

    industry_fields_present = {"target_industries", "acquirer_industries"} & set(selected_fields)
    if industry_fields_present:
        all_industry_codes: set[str] = set()
        for _field_key in industry_fields_present:
            all_industry_codes.update(cast(list[str], payload_out.get(_field_key, [])))
        if all_industry_codes:
            sector_rows = db.session.execute(
                text(f"SELECT CAST(sector_code AS CHAR), sector_desc FROM {schema_prefix()}naics_sectors")
            ).fetchall()
            sub_sector_rows = db.session.execute(
                text(f"SELECT CAST(sub_sector_code AS CHAR), sub_sector_desc FROM {schema_prefix()}naics_sub_sectors")
            ).fetchall()
            industry_labels: dict[str, str] = {}
            for _code, _desc in sector_rows:
                if str(_code) in all_industry_codes and _desc:
                    industry_labels[str(_code)] = str(_desc)
            for _code, _desc in sub_sector_rows:
                if str(_code) in all_industry_codes and _desc:
                    industry_labels[str(_code)] = str(_desc)
            payload_out["industry_labels"] = industry_labels

    filter_metadata = _filter_option_metadata()
    response = {
        "fields": list(selected_fields),
        "retrieval_parameter_map": {
            "targets": "target",
            "acquirers": "acquirer",
            "transaction_price_totals": "transaction_price_total",
            "transaction_price_stocks": "transaction_price_stock",
            "transaction_price_cashes": "transaction_price_cash",
            "transaction_price_assets": "transaction_price_assets",
            "transaction_considerations": "transaction_consideration",
            "target_types": "target_type",
            "acquirer_types": "acquirer_type",
            "target_counsels": "target_counsel",
            "acquirer_counsels": "acquirer_counsel",
            "target_industries": "target_industry",
            "acquirer_industries": "acquirer_industry",
            "deal_statuses": "deal_status",
            "attitudes": "attitude",
            "deal_types": "deal_type",
            "purposes": "purpose",
            "target_pes": "target_pe",
            "acquirer_pes": "acquirer_pe",
        },
        "field_metadata": {
            field_name: filter_metadata[field_name]
            for field_name in selected_fields
            if field_name in filter_metadata
        },
        **payload_out,
    }
    return McpToolResult(
        text=f"Returned {len(selected_fields)} filter option group(s).",
        structured_content=response,
    )


def _get_clause_taxonomy(
    deps: ReferenceDataDeps,
    *,
    principal: McpPrincipal,
) -> McpToolResult:
    _require_scope(principal, "sections:search")
    response = _build_taxonomy_tree(
        l1_model=deps.TaxonomyL1,
        l2_model=deps.TaxonomyL2,
        l3_model=deps.TaxonomyL3,
        deps=deps,
    )
    return McpToolResult(
        text=f"Returned {len(response)} top-level clause taxonomy node(s).",
        structured_content=response,
    )


def _get_tax_clause_taxonomy(
    deps: ReferenceDataDeps,
    *,
    principal: McpPrincipal,
) -> McpToolResult:
    _require_scope(principal, "sections:search")
    response = _build_taxonomy_tree(
        l1_model=deps.TaxClauseTaxonomyL1,
        l2_model=deps.TaxClauseTaxonomyL2,
        l3_model=deps.TaxClauseTaxonomyL3,
        deps=deps,
    )
    return McpToolResult(
        text=f"Returned {len(response)} top-level tax clause taxonomy node(s).",
        structured_content=response,
    )


def _get_counsel_catalog(
    deps: ReferenceDataDeps,
    *,
    principal: McpPrincipal,
    payload: dict[str, object],
) -> McpToolResult:
    _require_scope(principal, "sections:search")
    query = payload.get("query")
    limit = payload.get("limit")
    response = _counsel_payload(
        deps,
        query=query if isinstance(query, str) else None,
        limit=limit if isinstance(limit, int) else None,
    )
    counsel_rows = cast(list[object], response.get("counsel", []))
    return McpToolResult(
        text=f"Returned {len(counsel_rows)} counsel entr{'' if len(counsel_rows) == 1 else 'ies'}.",
        structured_content=response,
    )


def _get_naics_catalog(
    deps: ReferenceDataDeps,
    *,
    principal: McpPrincipal,
) -> McpToolResult:
    _require_scope(principal, "sections:search")
    response = _naics_payload(deps)
    sector_rows = cast(list[object], response.get("sectors", []))
    return McpToolResult(
        text=f"Returned {len(sector_rows)} NAICS sector(s).",
        structured_content=response,
    )


def _get_agreements_summary(
    deps: AgreementsDeps,
    *,
    principal: McpPrincipal,
) -> McpToolResult:
    _require_scope(principal, "agreements:search")
    response = _agreements_summary_payload(deps)
    return McpToolResult(
        text="Returned corpus summary metrics.",
        structured_content=response,
    )


def _get_agreement_trends(
    deps: AgreementsDeps,
    *,
    principal: McpPrincipal,
    reference_data_deps: ReferenceDataDeps,
) -> McpToolResult:
    _require_scope(principal, "agreements:search")
    response = _agreement_trends_payload(deps, reference_data_deps=reference_data_deps)
    return McpToolResult(
        text="Returned agreement trend analytics.",
        structured_content=response,
    )

