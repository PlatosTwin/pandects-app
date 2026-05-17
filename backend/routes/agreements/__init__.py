from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from threading import Lock
from typing import cast

from flask import Flask, Response, abort, current_app, g, jsonify, request
from flask.views import MethodView
from flask_smorest import Blueprint
from sqlalchemy import and_, asc, bindparam, desc, func, or_, text
from sqlalchemy.exc import SQLAlchemyError

from backend.services.counsel_leaderboards import build_counsel_leaderboards_from_summary_rows
from backend.filtering import (
    build_canonical_counsel_agreement_uuid_subquery,
    build_transaction_price_bucket_filter,
)
from backend.routes.agreements.helpers import (
    _agreement_is_public_eligible_expr,
    _apply_agreement_metadata_filters,
    _build_agreement_search_match_query,
    _cacheable_json_response,
    _metadata_field_coverage_sort_key,
    _normalize_industry_label,
    _strip_xml_snippet,
    _tax_clause_rows,
    _to_float_or_none,
)
from backend.routes.deps import AgreementsDeps
from backend.schemas.public_api import (
    AgreementArgsPayload,
    AgreementArgsSchema,
    AgreementSearchResponseSchema,
    AgreementSectionIndexResponseSchema,
    AgreementResponseSchema,
    AgreementsBulkArgsPayload,
    AgreementsBulkArgsSchema,
    AgreementsIndexArgsSchema,
    AgreementsListResponseSchema,
    SectionResponseSchema,
    TaxClauseListResponseSchema,
)
from backend.schemas.sections import SectionsArgsPayload, SectionsArgsSchema

__all__ = [
    "_agreement_is_public_eligible_expr",
    "_normalize_industry_label",
    "_tax_clause_rows",
    "_to_float_or_none",
    "register_agreements_routes",
]



def register_agreements_routes(
    target_app: Flask,
    *,
    deps: AgreementsDeps,
) -> tuple[Blueprint, Blueprint, Blueprint]:
    agreement_trends_cache: dict[str, object] = {"ts": 0.0, "payload": None}
    agreement_trends_lock = Lock()
    counsel_leaderboards_cache: dict[str, object] = {"ts": 0.0, "payload": None}
    counsel_leaderboards_lock = Lock()
    agreement_status_summary_cache: dict[str, object] = {"ts": 0.0, "payload": None}
    agreement_status_summary_lock = Lock()
    agreement_deal_types_summary_cache: dict[str, object] = {"ts": 0.0, "payload": None}
    agreement_deal_types_summary_lock = Lock()

    agreements_blp = Blueprint(
        "agreements",
        "agreements",
        url_prefix="/v1/agreements",
        description="Retrieve full text for a given agreement",
    )
    sections_blp = Blueprint(
        "sections",
        "sections",
        url_prefix="/v1/sections",
        description="Retrieve full text for a given section",
    )
    agreement_search_blp = Blueprint(
        "agreements_search",
        "agreements_search",
        url_prefix="/v1/search/agreements",
        description="Search agreements and return one result per deal with matched section previews.",
    )

    def get_metadata_field_coverage() -> list[dict[str, object]]:
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
            (
                deps._row_mapping_as_dict(cast(object, row))
                for row in rows
            ),
            key=_metadata_field_coverage_sort_key,
        )
        return [
            {
                "field": row_dict.get("field_name"),
                "label": row_dict.get("label"),
                "ingested_eligible_agreements": deps._to_int(
                    cast(object, row_dict.get("ingested_eligible_agreements"))
                ),
                "ingested_covered_agreements": deps._to_int(
                    cast(object, row_dict.get("ingested_covered_agreements"))
                ),
                "ingested_coverage_pct": _to_float_or_none(
                    row_dict.get("ingested_coverage_pct")
                ),
                "processed_eligible_agreements": deps._to_int(
                    cast(object, row_dict.get("processed_eligible_agreements"))
                ),
                "processed_covered_agreements": deps._to_int(
                    cast(object, row_dict.get("processed_covered_agreements"))
                ),
                "processed_coverage_pct": _to_float_or_none(
                    row_dict.get("processed_coverage_pct")
                ),
                "note": row_dict.get("note"),
            }
            for row_dict in sorted_rows
        ]

    @agreements_blp.route("")
    class AgreementsListResource(MethodView):
        @agreements_blp.doc(
            operationId="listAgreements",
            summary="List agreements with keyset pagination",
            description=(
                "Lists eligible agreements using a base64 cursor. Supports the same agreement-level "
                "filters as `/v1/sections` except clause-type taxonomy filtering."
            ),
        )
        @agreements_blp.arguments(AgreementsBulkArgsSchema, location="query")
        @agreements_blp.response(200, AgreementsListResponseSchema)
        def get(self, args: dict[str, object]) -> dict[str, object]:
            ctx = deps._current_access_context()
            parsed_args = cast(AgreementsBulkArgsPayload, cast(object, args))

            if "standard_id" in request.args:
                abort(400, description="The standard_id filter is not supported on /v1/agreements.")

            include_xml = parsed_args["include_xml"]
            if include_xml and not ctx.is_authenticated:
                abort(403, description="Authentication required when include_xml=true.")

            page_size = parsed_args["page_size"]
            if page_size < 1 or page_size > 100:
                page_size = 25

            after_agreement_uuid = deps._decode_agreements_cursor(parsed_args["cursor"])

            years = parsed_args["year"]
            targets = parsed_args["target"]
            acquirers = parsed_args["acquirer"]
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
            agreement_uuid = parsed_args["agreement_uuid"]
            section_uuid = parsed_args["section_uuid"]

            agreements = deps.Agreements
            agreement_counsel = deps.AgreementCounsel
            counsel = deps.Counsel
            xml = deps.XML
            sections = deps.Sections
            db = deps.db
            year_expr = deps._agreement_year_expr().label("year")
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
            ]
            q = (
                db.session.query(*item_columns)
                .join(xml, deps._agreement_latest_xml_join_condition())
                .filter(_agreement_is_public_eligible_expr(agreements))
            )

            if include_xml:
                q = q.add_columns(xml.xml.label("xml"))

            if years:
                year_filters = tuple(
                    and_(
                        agreements.filing_date >= f"{year:04d}-01-01",
                        agreements.filing_date < f"{year + 1:04d}-01-01",
                    )
                    for year in years
                )
                q = q.filter(or_(*year_filters))

            if targets:
                q = q.filter(agreements.target.in_(targets))
            if acquirers:
                q = q.filter(agreements.acquirer.in_(acquirers))
            transaction_price_total_filter = build_transaction_price_bucket_filter(
                agreements.transaction_price_total,
                transaction_price_totals,
            )
            if transaction_price_total_filter is not None:
                q = q.filter(transaction_price_total_filter)
            transaction_price_stock_filter = build_transaction_price_bucket_filter(
                agreements.transaction_price_stock,
                transaction_price_stocks,
            )
            if transaction_price_stock_filter is not None:
                q = q.filter(transaction_price_stock_filter)
            transaction_price_cash_filter = build_transaction_price_bucket_filter(
                agreements.transaction_price_cash,
                transaction_price_cashes,
            )
            if transaction_price_cash_filter is not None:
                q = q.filter(transaction_price_cash_filter)
            transaction_price_assets_filter = build_transaction_price_bucket_filter(
                agreements.transaction_price_assets,
                transaction_price_assets,
            )
            if transaction_price_assets_filter is not None:
                q = q.filter(transaction_price_assets_filter)
            if transaction_considerations:
                q = q.filter(agreements.transaction_consideration.in_(transaction_considerations))
            if target_types:
                q = q.filter(agreements.target_type.in_(target_types))
            if acquirer_types:
                q = q.filter(agreements.acquirer_type.in_(acquirer_types))
            target_counsel_subquery = build_canonical_counsel_agreement_uuid_subquery(
                side="target",
                canonical_names=target_counsels,
                agreement_counsel=agreement_counsel,
                counsel=counsel,
            )
            if target_counsel_subquery is not None:
                q = q.filter(agreements.agreement_uuid.in_(target_counsel_subquery))
            acquirer_counsel_subquery = build_canonical_counsel_agreement_uuid_subquery(
                side="acquirer",
                canonical_names=acquirer_counsels,
                agreement_counsel=agreement_counsel,
                counsel=counsel,
            )
            if acquirer_counsel_subquery is not None:
                q = q.filter(agreements.agreement_uuid.in_(acquirer_counsel_subquery))
            if target_industries:
                q = q.filter(agreements.target_industry.in_(target_industries))
            if acquirer_industries:
                q = q.filter(agreements.acquirer_industry.in_(acquirer_industries))
            if deal_statuses:
                q = q.filter(agreements.deal_status.in_(deal_statuses))
            if attitudes:
                q = q.filter(agreements.attitude.in_(attitudes))
            if deal_types:
                q = q.filter(agreements.deal_type.in_(deal_types))
            if purposes:
                q = q.filter(agreements.purpose.in_(purposes))

            if target_pes:
                db_target_pes: list[int] = []
                for pe in target_pes:
                    if pe == "true":
                        db_target_pes.append(1)
                    elif pe == "false":
                        db_target_pes.append(0)
                if db_target_pes:
                    q = q.filter(agreements.target_pe.in_(db_target_pes))

            if acquirer_pes:
                db_acquirer_pes: list[int] = []
                for pe in acquirer_pes:
                    if pe == "true":
                        db_acquirer_pes.append(1)
                    elif pe == "false":
                        db_acquirer_pes.append(0)
                if db_acquirer_pes:
                    q = q.filter(agreements.acquirer_pe.in_(db_acquirer_pes))

            if agreement_uuid and agreement_uuid.strip():
                q = q.filter(agreements.agreement_uuid == agreement_uuid.strip())

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
                q.order_by(asc(agreements.agreement_uuid))
                .limit(page_size + 1)
                .all(),
            )
            has_next = len(rows) > page_size
            page_rows = rows[:page_size]

            results: list[dict[str, object]] = []
            for row in page_rows:
                row_map = deps._row_mapping_as_dict(row)
                payload = {
                    "agreement_uuid": row_map.get("agreement_uuid"),
                    "year": row_map.get("year"),
                    "target": row_map.get("target"),
                    "acquirer": row_map.get("acquirer"),
                    "filing_date": row_map.get("filing_date"),
                    "prob_filing": row_map.get("prob_filing"),
                    "filing_company_name": row_map.get("filing_company_name"),
                    "filing_company_cik": row_map.get("filing_company_cik"),
                    "form_type": row_map.get("form_type"),
                    "exhibit_type": row_map.get("exhibit_type"),
                    "transaction_price_total": row_map.get("transaction_price_total"),
                    "transaction_price_stock": row_map.get("transaction_price_stock"),
                    "transaction_price_cash": row_map.get("transaction_price_cash"),
                    "transaction_price_assets": row_map.get("transaction_price_assets"),
                    "transaction_consideration": row_map.get("transaction_consideration"),
                    "target_type": row_map.get("target_type"),
                    "acquirer_type": row_map.get("acquirer_type"),
                    "target_industry": row_map.get("target_industry"),
                    "acquirer_industry": row_map.get("acquirer_industry"),
                    "announce_date": row_map.get("announce_date"),
                    "close_date": row_map.get("close_date"),
                    "deal_status": row_map.get("deal_status"),
                    "attitude": row_map.get("attitude"),
                    "deal_type": row_map.get("deal_type"),
                    "purpose": row_map.get("purpose"),
                    "target_pe": row_map.get("target_pe"),
                    "acquirer_pe": row_map.get("acquirer_pe"),
                    "url": row_map.get("url"),
                }
                if include_xml:
                    payload["xml"] = row_map.get("xml")
                results.append(payload)

            next_cursor: str | None = None
            if has_next:
                last_row = deps._row_mapping_as_dict(page_rows[-1])
                last_agreement_uuid = last_row.get("agreement_uuid")
                if not isinstance(last_agreement_uuid, str) or not last_agreement_uuid:
                    raise RuntimeError("Agreements list query returned a row without agreement_uuid.")
                next_cursor = deps._encode_agreements_cursor(last_agreement_uuid)

            response_payload: dict[str, object] = {
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
            dump_version = getattr(g, "dump_version", None)
            if dump_version is not None and bool(args.get("include_dump", True)):
                response_payload["dump_version"] = dump_version
            return response_payload

    @agreement_search_blp.route("")
    class AgreementSearchResource(MethodView):
        @agreement_search_blp.doc(
            operationId="searchAgreements",
            summary="Search agreements with matched section previews",
            description=(
                "Searches agreements with the same structured filters as `/v1/sections`, "
                "including taxonomy IDs, and returns one row per agreement."
            ),
        )
        @agreement_search_blp.arguments(SectionsArgsSchema, location="query")
        @agreement_search_blp.response(200, AgreementSearchResponseSchema)
        def get(self, args: dict[str, object]) -> dict[str, object]:
            ctx = deps._current_access_context()
            parsed_args = cast(SectionsArgsPayload, cast(object, args))
            sections = deps.Sections
            agreements = deps.Agreements
            latest = deps.LatestSectionsSearch
            xml = deps.XML
            db = deps.db

            page = parsed_args["page"]
            page_size = parsed_args["page_size"]
            if page < 1:
                page = 1
            max_page_size = 100 if ctx.is_authenticated else 10
            if page_size < 1 or page_size > max_page_size:
                page_size = min(25, max_page_size)

            clause_context_requested = bool(
                parsed_args["standard_id"] or (
                    parsed_args["section_uuid"] and parsed_args["section_uuid"].strip()
                )
            )

            if not clause_context_requested:
                direct_query = (
                    db.session.query(
                        agreements.agreement_uuid.label("agreement_uuid"),
                        deps._agreement_year_expr().label("year"),
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
                    )
                    .join(xml, deps._agreement_latest_xml_join_condition())
                    .filter(_agreement_is_public_eligible_expr(agreements))
                )
                direct_query = _apply_agreement_metadata_filters(
                    direct_query,
                    db=db,
                    agreements=agreements,
                    agreement_counsel=deps.AgreementCounsel,
                    counsel=deps.Counsel,
                    sections=sections,
                    parsed_args=parsed_args,
                )

                sort_by = parsed_args["sort_by"]
                descending = parsed_args["sort_direction"] == "desc"
                if sort_by == "year":
                    primary_sort = agreements.filing_date
                elif sort_by == "target":
                    primary_sort = agreements.target
                else:
                    primary_sort = agreements.acquirer
                if descending:
                    direct_query = direct_query.order_by(
                        desc(primary_sort),
                        desc(agreements.agreement_uuid),
                    )
                else:
                    direct_query = direct_query.order_by(
                        asc(primary_sort),
                        asc(agreements.agreement_uuid),
                    )

                page_rows_raw = cast(
                    list[object],
                    direct_query.offset((page - 1) * page_size).limit(page_size + 1).all(),
                )
                has_next = len(page_rows_raw) > page_size
                page_rows = page_rows_raw[:page_size]
                total_count = deps._to_int(direct_query.order_by(None).count())
                total_count_is_approximate = False

                meta = deps._pagination_metadata(
                    total_count=total_count,
                    page=page,
                    page_size=page_size,
                    has_next_override=has_next,
                    total_count_is_approximate=total_count_is_approximate,
                )

                results = []
                for row in page_rows:
                    row_map = deps._row_mapping_as_dict(cast(object, row))
                    results.append(
                        {
                            "agreement_uuid": row_map.get("agreement_uuid"),
                            "year": row_map.get("year"),
                            "target": row_map.get("target"),
                            "acquirer": row_map.get("acquirer"),
                            "filing_date": row_map.get("filing_date"),
                            "prob_filing": row_map.get("prob_filing"),
                            "filing_company_name": row_map.get("filing_company_name"),
                            "filing_company_cik": row_map.get("filing_company_cik"),
                            "form_type": row_map.get("form_type"),
                            "exhibit_type": row_map.get("exhibit_type"),
                            "transaction_price_total": row_map.get("transaction_price_total"),
                            "transaction_price_stock": row_map.get("transaction_price_stock"),
                            "transaction_price_cash": row_map.get("transaction_price_cash"),
                            "transaction_price_assets": row_map.get("transaction_price_assets"),
                            "transaction_consideration": row_map.get("transaction_consideration"),
                            "target_type": row_map.get("target_type"),
                            "acquirer_type": row_map.get("acquirer_type"),
                            "target_industry": row_map.get("target_industry"),
                            "acquirer_industry": row_map.get("acquirer_industry"),
                            "announce_date": row_map.get("announce_date"),
                            "close_date": row_map.get("close_date"),
                            "deal_status": row_map.get("deal_status"),
                            "attitude": row_map.get("attitude"),
                            "deal_type": row_map.get("deal_type"),
                            "purpose": row_map.get("purpose"),
                            "target_pe": row_map.get("target_pe"),
                            "acquirer_pe": row_map.get("acquirer_pe"),
                            "url": row_map.get("url"),
                            "match_count": 0,
                            "matched_sections": [],
                        }
                    )

                return {
                    "results": results,
                    "access": {
                        "tier": ctx.tier,
                        "message": None
                        if ctx.is_authenticated
                        else "Sign in to view full agreement text and unlock higher page sizes.",
                    },
                    **meta,
                }

            match_query, _ = _build_agreement_search_match_query(
                deps,
                parsed_args=parsed_args,
            )
            aggregated_query = (
                match_query.with_entities(
                    latest.agreement_uuid.label("agreement_uuid"),
                    func.count().label("match_count"),
                    func.min(latest.filing_date).label("sort_filing_date"),
                    func.min(latest.target).label("sort_target"),
                    func.min(latest.acquirer).label("sort_acquirer"),
                )
                .group_by(latest.agreement_uuid)
            )

            sort_by = parsed_args["sort_by"]
            descending = parsed_args["sort_direction"] == "desc"
            if sort_by == "year":
                sort_column = text("sort_filing_date")
            elif sort_by == "target":
                sort_column = text("sort_target")
            else:
                sort_column = text("sort_acquirer")
            if descending:
                aggregated_query = aggregated_query.order_by(
                    desc(sort_column),
                    desc(text("agreement_uuid")),
                )
            else:
                aggregated_query = aggregated_query.order_by(
                    asc(sort_column),
                    asc(text("agreement_uuid")),
                )

            page_rows_raw = cast(
                list[object],
                aggregated_query.offset((page - 1) * page_size).limit(page_size + 1).all(),
            )
            has_next = len(page_rows_raw) > page_size
            page_rows = page_rows_raw[:page_size]
            total_count = deps._to_int(aggregated_query.order_by(None).count())
            total_count_is_approximate = False

            meta = deps._pagination_metadata(
                total_count=total_count,
                page=page,
                page_size=page_size,
                has_next_override=has_next,
                total_count_is_approximate=total_count_is_approximate,
            )
            agreement_ids: list[str] = []
            match_count_by_agreement: dict[str, int] = {}
            for row in page_rows:
                row_map = deps._row_mapping_as_dict(cast(object, row))
                agreement_id = row_map.get("agreement_uuid")
                if not isinstance(agreement_id, str):
                    continue
                agreement_ids.append(agreement_id)
                match_count_by_agreement[agreement_id] = deps._to_int(
                    row_map.get("match_count")
                )

            agreement_details_by_id: dict[str, dict[str, object]] = {}
            if agreement_ids:
                detail_rows = cast(
                    list[object],
                    db.session.query(
                        agreements.agreement_uuid.label("agreement_uuid"),
                        deps._agreement_year_expr().label("year"),
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
                    )
                    .filter(agreements.agreement_uuid.in_(agreement_ids))
                    .all(),
                )
                for row in detail_rows:
                    row_map = deps._row_mapping_as_dict(cast(object, row))
                    agreement_id = row_map.get("agreement_uuid")
                    if isinstance(agreement_id, str):
                        agreement_details_by_id[agreement_id] = row_map

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
                    match_query.join(
                        sections,
                        sections.section_uuid == latest.section_uuid,
                    )
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
                    .filter(preview_subquery.c.preview_rank <= 3)
                    .order_by(
                        asc(preview_subquery.c.agreement_uuid),
                        asc(preview_subquery.c.article_order),
                        asc(preview_subquery.c.section_order),
                        asc(preview_subquery.c.section_uuid),
                    )
                    .all(),
                )
                for row in preview_rows:
                    row_map = deps._row_mapping_as_dict(cast(object, row))
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
                            "snippet": _strip_xml_snippet(row_map.get("xml_content")),
                        }
                    )

            results: list[dict[str, object]] = []
            for agreement_id in agreement_ids:
                detail_row = agreement_details_by_id.get(agreement_id)
                if detail_row is None:
                    continue
                results.append(
                    {
                        "agreement_uuid": agreement_id,
                        "year": detail_row.get("year"),
                        "target": detail_row.get("target"),
                        "acquirer": detail_row.get("acquirer"),
                        "filing_date": detail_row.get("filing_date"),
                        "prob_filing": detail_row.get("prob_filing"),
                        "filing_company_name": detail_row.get("filing_company_name"),
                        "filing_company_cik": detail_row.get("filing_company_cik"),
                        "form_type": detail_row.get("form_type"),
                        "exhibit_type": detail_row.get("exhibit_type"),
                        "transaction_price_total": detail_row.get("transaction_price_total"),
                        "transaction_price_stock": detail_row.get("transaction_price_stock"),
                        "transaction_price_cash": detail_row.get("transaction_price_cash"),
                        "transaction_price_assets": detail_row.get("transaction_price_assets"),
                        "transaction_consideration": detail_row.get("transaction_consideration"),
                        "target_type": detail_row.get("target_type"),
                        "acquirer_type": detail_row.get("acquirer_type"),
                        "target_industry": detail_row.get("target_industry"),
                        "acquirer_industry": detail_row.get("acquirer_industry"),
                        "announce_date": detail_row.get("announce_date"),
                        "close_date": detail_row.get("close_date"),
                        "deal_status": detail_row.get("deal_status"),
                        "attitude": detail_row.get("attitude"),
                        "deal_type": detail_row.get("deal_type"),
                        "purpose": detail_row.get("purpose"),
                        "target_pe": detail_row.get("target_pe"),
                        "acquirer_pe": detail_row.get("acquirer_pe"),
                        "url": detail_row.get("url"),
                        "match_count": match_count_by_agreement.get(agreement_id, 0),
                        "matched_sections": matched_sections_by_agreement.get(
                            agreement_id,
                            [],
                        )[:3],
                    }
                )

            search_payload: dict[str, object] = {
                "results": results,
                "access": {
                    "tier": ctx.tier,
                    "message": None
                    if ctx.is_authenticated
                    else "Sign in to view full agreement text and unlock higher page sizes.",
                },
                **meta,
            }
            dump_version = getattr(g, "dump_version", None)
            if dump_version is not None and bool(args.get("include_dump", True)):
                search_payload["dump_version"] = dump_version
            return search_payload

    @agreements_blp.route("/<string:agreement_uuid>")
    class AgreementResource(MethodView):
        @agreements_blp.doc(
            operationId="getAgreement",
            summary="Retrieve agreement text by UUID",
            description="Returns agreement metadata and XML content. For anonymous callers, XML can be redacted based on `focus_section_uuid` and `neighbor_sections`.",
        )
        @agreements_blp.arguments(AgreementArgsSchema, location="query")
        @agreements_blp.response(200, AgreementResponseSchema)
        def get(self, args: dict[str, object], agreement_uuid: str) -> dict[str, object]:
            ctx = deps._current_access_context()
            parsed_args = cast(AgreementArgsPayload, cast(object, args))
            focus_section_uuid = parsed_args.get("focus_section_uuid")
            if focus_section_uuid is not None:
                focus_section_uuid = focus_section_uuid.strip()
                if not deps._SECTION_ID_RE.match(focus_section_uuid):
                    abort(400, description="Invalid focus_section_uuid.")
            neighbor_sections_int = parsed_args["neighbor_sections"]

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
                .filter(
                    agreements.agreement_uuid == agreement_uuid,
                    _agreement_is_public_eligible_expr(agreements),
                )
                .first()
            )

            if row is None:
                abort(404)

            row_map = deps._row_mapping_as_dict(cast(object, row))
            xml_content_obj = row_map.get("xml")
            xml_content = xml_content_obj if isinstance(xml_content_obj, str) else ""
            payload = {
                "year": row_map.get("year"),
                "target": row_map.get("target"),
                "acquirer": row_map.get("acquirer"),
                "filing_date": row_map.get("filing_date"),
                "prob_filing": row_map.get("prob_filing"),
                "filing_company_name": row_map.get("filing_company_name"),
                "filing_company_cik": row_map.get("filing_company_cik"),
                "form_type": row_map.get("form_type"),
                "exhibit_type": row_map.get("exhibit_type"),
                "transaction_price_total": row_map.get("transaction_price_total"),
                "transaction_price_stock": row_map.get("transaction_price_stock"),
                "transaction_price_cash": row_map.get("transaction_price_cash"),
                "transaction_price_assets": row_map.get("transaction_price_assets"),
                "transaction_consideration": row_map.get("transaction_consideration"),
                "target_type": row_map.get("target_type"),
                "acquirer_type": row_map.get("acquirer_type"),
                "target_industry": row_map.get("target_industry"),
                "acquirer_industry": row_map.get("acquirer_industry"),
                "announce_date": row_map.get("announce_date"),
                "close_date": row_map.get("close_date"),
                "deal_status": row_map.get("deal_status"),
                "attitude": row_map.get("attitude"),
                "deal_type": row_map.get("deal_type"),
                "purpose": row_map.get("purpose"),
                "target_pe": row_map.get("target_pe"),
                "acquirer_pe": row_map.get("acquirer_pe"),
                "url": row_map.get("url"),
            }
            if not ctx.is_authenticated:
                redacted_xml = deps._redact_agreement_xml(
                    xml_content,
                    focus_section_uuid=focus_section_uuid,
                    neighbor_sections=neighbor_sections_int,
                )
                payload["xml"] = redacted_xml
                payload["is_redacted"] = True
                return payload
            payload["xml"] = xml_content
            return payload

    @agreements_blp.route("/<string:agreement_uuid>/sections")
    class AgreementSectionsIndexResource(MethodView):
        @agreements_blp.doc(
            operationId="listAgreementSections",
            summary="List ordered agreement sections",
            description=(
                "Returns ordered section metadata for one agreement, including section UUIDs "
                "and taxonomy IDs, for in-document navigation."
            ),
        )
        @agreements_blp.response(200, AgreementSectionIndexResponseSchema)
        def get(self, agreement_uuid: str) -> dict[str, object]:
            agreements = deps.Agreements
            sections = deps.Sections
            xml = deps.XML
            db = deps.db

            agreement_exists = (
                db.session.query(agreements.agreement_uuid)
                .join(xml, deps._agreement_latest_xml_join_condition())
                .filter(
                    agreements.agreement_uuid == agreement_uuid,
                    _agreement_is_public_eligible_expr(agreements),
                )
                .first()
            )
            if agreement_exists is None:
                abort(404)

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
                        "standard_id": deps._parse_section_standard_ids(
                            row_map.get("section_standard_ids")
                        ),
                    }
                    for row_map in (
                        deps._row_mapping_as_dict(cast(object, row)) for row in rows
                    )
                ],
            }

    @sections_blp.route("/<string:section_uuid>")
    class SectionResource(MethodView):
        @sections_blp.doc(
            operationId="getSection",
            summary="Retrieve section text by UUID",
            description="Returns one section payload including taxonomy IDs and XML content.",
        )
        @sections_blp.response(200, SectionResponseSchema)
        def get(self, section_uuid: str) -> dict[str, object]:
            section_uuid = section_uuid.strip()
            if not deps._SECTION_ID_RE.match(section_uuid):
                abort(400, description="Invalid section_uuid.")

            sections = deps.Sections
            xml = deps.XML
            db = deps.db
            section_cols = sections.__table__.c
            section_standard_ids_expr = deps._coalesced_section_standard_ids().label(
                "section_standard_ids"
            )
            row = (
                db.session.query(
                    section_cols["agreement_uuid"].label("agreement_uuid"),
                    section_cols["section_uuid"].label("section_uuid"),
                    section_standard_ids_expr,
                    section_cols["xml_content"].label("xml_content"),
                    section_cols["article_title"].label("article_title"),
                    section_cols["section_title"].label("section_title"),
                )
                .join(
                    xml,
                    deps._section_latest_xml_join_condition(),
                )
                .filter(section_cols["section_uuid"] == section_uuid)
                .first()
            )

            if row is None:
                abort(404)

            (
                agreement_uuid,
                section_uuid_value,
                section_standard_ids_raw,
                xml_content,
                article_title,
                section_title,
            ) = cast(
                tuple[object, object, object, object, object, object],
                cast(object, row),
            )

            section_standard_ids = deps._parse_section_standard_ids(section_standard_ids_raw)

            return {
                "agreement_uuid": agreement_uuid,
                "section_uuid": section_uuid_value,
                "section_standard_id": section_standard_ids,
                "xml": xml_content,
                "article_title": article_title,
                "section_title": section_title,
            }

    @agreements_blp.route("/<string:agreement_uuid>/tax-clauses")
    class AgreementTaxClausesResource(MethodView):
        @agreements_blp.doc(
            operationId="getAgreementTaxClauses",
            summary="Retrieve tax clauses for an agreement",
            description="Returns extracted tax-module clauses for the latest verified XML of an agreement.",
        )
        @agreements_blp.response(200, TaxClauseListResponseSchema)
        def get(self, agreement_uuid: str) -> dict[str, object]:
            agreement_uuid = agreement_uuid.strip()
            if agreement_uuid == "":
                abort(400, description="Invalid agreement_uuid.")
            return {"clauses": _tax_clause_rows(deps, agreement_uuid=agreement_uuid)}

    @sections_blp.route("/<string:section_uuid>/tax-clauses")
    class SectionTaxClausesResource(MethodView):
        @sections_blp.doc(
            operationId="getSectionTaxClauses",
            summary="Retrieve tax clauses for a section",
            description="Returns extracted tax-module clauses for a specific latest section.",
        )
        @sections_blp.response(200, TaxClauseListResponseSchema)
        def get(self, section_uuid: str) -> dict[str, object]:
            section_uuid = section_uuid.strip()
            if not deps._SECTION_ID_RE.match(section_uuid):
                abort(400, description="Invalid section_uuid.")
            return {"clauses": _tax_clause_rows(deps, section_uuid=section_uuid)}

    def get_agreements_index() -> dict[str, object]:
        ctx = deps._current_access_context()
        args = deps._load_query(AgreementsIndexArgsSchema())
        page = cast(int, args["page"])
        page_size = cast(int, args["page_size"])
        sort_by = str(args["sort_by"] or "year")
        sort_dir = str(args["sort_dir"] or "desc")
        query = str(args.get("query") or "").strip()

        if page < 1:
            page = 1

        max_page_size = 100 if ctx.is_authenticated else 10
        if page_size < 1 or page_size > max_page_size:
            page_size = min(25, max_page_size)

        db = deps.db
        sort_direction = sort_dir.lower()
        sort_column_map = {
            "year": "year",
            "target": "target",
            "acquirer": "acquirer",
        }
        sort_column = sort_column_map.get(sort_by, "year")
        sort_direction_sql = "DESC" if sort_direction == "desc" else "ASC"
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
        offset = (page - 1) * page_size
        page_params = {
            **params,
            "limit_value": page_size,
            "offset_value": offset,
        }
        items = (
            db.session.execute(
                text(
                    f"""
                    SELECT
                        agreement_uuid,
                        year,
                        target,
                        acquirer,
                        verified
                    FROM {deps._schema_prefix()}agreement_index_summary
                    WHERE {where_sql}
                    ORDER BY {sort_column} {sort_direction_sql}, agreement_uuid ASC
                    LIMIT :limit_value OFFSET :offset_value
                    """
                ),
                page_params,
            )
            .mappings()
            .all()
        )
        meta = deps._pagination_metadata(total_count=total_count, page=page, page_size=page_size)

        results: list[dict[str, object]] = []
        for row in items:
            row_map = deps._row_mapping_as_dict(cast(object, row))
            verified_value = row_map.get("verified")
            results.append(
                {
                    "agreement_uuid": row_map.get("agreement_uuid"),
                    "year": row_map.get("year"),
                    "target": row_map.get("target"),
                    "acquirer": row_map.get("acquirer"),
                    "consideration_type": None,
                    "total_consideration": None,
                    "target_industry": None,
                    "acquirer_industry": None,
                    "verified": bool(verified_value) if verified_value is not None else False,
                }
            )

        return {"results": results, **meta}

    def get_agreements_status_summary() -> tuple[Response, int] | Response:
        cache_enabled = not current_app.testing
        now = deps.time.time()
        if cache_enabled:
            with agreement_status_summary_lock:
                cached_payload = agreement_status_summary_cache["payload"]
                cached_ts = cast(float, agreement_status_summary_cache["ts"])
                cache_is_valid = cached_payload is not None and (
                    now - cached_ts < deps._AGREEMENTS_SUMMARY_TTL_SECONDS
                )
            if cache_is_valid and isinstance(cached_payload, dict):
                return _cacheable_json_response(
                    cached_payload,
                    max_age=deps._AGREEMENTS_SUMMARY_TTL_SECONDS,
                )

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
        overview_row_dict = (
            deps._row_mapping_as_dict(cast(object, overview_row))
            if overview_row is not None
            else {}
        )
        latest_filing_date = overview_row_dict.get("latest_filing_date")
        if isinstance(latest_filing_date, (date, datetime)):
            latest_filing_date = latest_filing_date.isoformat()
        elif latest_filing_date is not None:
            latest_filing_date = str(latest_filing_date)
        rows = (
            db.session.execute(
                text(
                    f"""
                    SELECT
                        year,
                        color,
                        current_stage,
                        count
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
                    "year": deps._to_int(cast(object, row_dict.get("year"))),
                    "color": row_dict.get("color"),
                    "current_stage": row_dict.get("current_stage"),
                    "count": deps._to_int(cast(object, row_dict.get("count"))),
                }
            )
        payload = {
            "years": years,
            "latest_filing_date": latest_filing_date,
            "metadata_covered_agreements": deps._to_int(
                cast(object, overview_row_dict.get("metadata_covered_agreements"))
            ),
            "metadata_coverage_pct": _to_float_or_none(
                overview_row_dict.get("metadata_coverage_pct")
            ),
            "taxonomy_covered_sections": deps._to_int(
                cast(object, overview_row_dict.get("taxonomy_covered_sections"))
            ),
            "taxonomy_coverage_pct": _to_float_or_none(
                overview_row_dict.get("taxonomy_coverage_pct")
            ),
            "metadata_field_coverage": get_metadata_field_coverage(),
        }
        if cache_enabled:
            with agreement_status_summary_lock:
                agreement_status_summary_cache["payload"] = payload
                agreement_status_summary_cache["ts"] = now
        return _cacheable_json_response(
            payload,
            max_age=deps._AGREEMENTS_SUMMARY_TTL_SECONDS,
        )

    def get_agreements_deal_types_summary() -> tuple[Response, int] | Response:
        cache_enabled = not current_app.testing
        now = deps.time.time()
        if cache_enabled:
            with agreement_deal_types_summary_lock:
                cached_payload = agreement_deal_types_summary_cache["payload"]
                cached_ts = cast(float, agreement_deal_types_summary_cache["ts"])
                cache_is_valid = cached_payload is not None and (
                    now - cached_ts < deps._AGREEMENTS_SUMMARY_TTL_SECONDS
                )
            if cache_is_valid and isinstance(cached_payload, dict):
                return _cacheable_json_response(
                    cached_payload,
                    max_age=deps._AGREEMENTS_SUMMARY_TTL_SECONDS,
                )

        db = deps.db
        rows = (
            db.session.execute(
                text(
                    f"""
                    SELECT
                        year,
                        deal_type,
                        `count`
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
                    "year": deps._to_int(cast(object, row_dict.get("year"))),
                    "deal_type": str(row_dict.get("deal_type") or "unknown"),
                    "count": deps._to_int(cast(object, row_dict.get("count"))),
                }
            )
        payload = {"years": years}
        if cache_enabled:
            with agreement_deal_types_summary_lock:
                agreement_deal_types_summary_cache["payload"] = payload
                agreement_deal_types_summary_cache["ts"] = now
        return _cacheable_json_response(
            payload,
            max_age=deps._AGREEMENTS_SUMMARY_TTL_SECONDS,
        )

    def get_agreements_summary() -> tuple[Response, int] | Response:
        now = deps.time.time()
        with deps._agreements_summary_lock:
            cached_payload = deps._agreements_summary_cache["payload"]
            cached_ts = deps._agreements_summary_cache["ts"]
            cache_is_valid = cached_payload is not None and (
                now - cached_ts < deps._AGREEMENTS_SUMMARY_TTL_SECONDS
            )
        if cache_is_valid and cached_payload is not None:
            return _cacheable_json_response(
                cast(dict[str, object], cached_payload),
                max_age=deps._AGREEMENTS_SUMMARY_TTL_SECONDS,
            )

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
        overview_row_dict = (
            deps._row_mapping_as_dict(cast(object, overview_row))
            if overview_row is not None
            else {}
        )
        latest_filing_date = overview_row_dict.get("latest_filing_date")
        if isinstance(latest_filing_date, (date, datetime)):
            latest_filing_date = latest_filing_date.isoformat()
        elif latest_filing_date is not None:
            latest_filing_date = str(latest_filing_date)
        payload: dict[str, object] = {
            "agreements": deps._to_int(cast(object, row_dict.get("agreements"))),
            "sections": deps._to_int(cast(object, row_dict.get("sections"))),
            "pages": deps._to_int(cast(object, row_dict.get("pages"))),
            "latest_filing_date": latest_filing_date,
        }
        with deps._agreements_summary_lock:
            deps._agreements_summary_cache["payload"] = payload
            deps._agreements_summary_cache["ts"] = now

        return _cacheable_json_response(
            payload,
            max_age=deps._AGREEMENTS_SUMMARY_TTL_SECONDS,
        )

    def get_counsel_leaderboards() -> tuple[Response, int] | Response:
        cache_enabled = not current_app.testing
        now = deps.time.time()
        if cache_enabled:
            with counsel_leaderboards_lock:
                cached_payload = counsel_leaderboards_cache["payload"]
                cached_ts = cast(float, counsel_leaderboards_cache["ts"])
                cache_is_valid = cached_payload is not None and (
                    now - cached_ts < deps._AGREEMENTS_SUMMARY_TTL_SECONDS
                )
            if cache_is_valid and isinstance(cached_payload, dict):
                return _cacheable_json_response(
                    cached_payload,
                    max_age=deps._AGREEMENTS_SUMMARY_TTL_SECONDS,
                )

        db = deps.db
        rows = (
            db.session.execute(
                text(
                    f"""
                    SELECT
                        side,
                        counsel_key,
                        counsel,
                        year,
                        deal_count,
                        total_transaction_value
                    FROM {deps._schema_prefix()}agreement_counsel_leaderboard_summary a
                    ORDER BY year ASC, side ASC, counsel ASC
                    """
                )
            )
            .mappings()
            .all()
        )

        payload = build_counsel_leaderboards_from_summary_rows(
            [
                deps._row_mapping_as_dict(cast(object, row))
                for row in rows
            ]
        )
        if cache_enabled:
            with counsel_leaderboards_lock:
                counsel_leaderboards_cache["payload"] = payload
                counsel_leaderboards_cache["ts"] = now
        return _cacheable_json_response(
            payload,
            max_age=deps._AGREEMENTS_SUMMARY_TTL_SECONDS,
        )

    def get_agreement_trends() -> tuple[Response, int] | Response:
        cache_enabled = not current_app.testing
        now = deps.time.time()
        if cache_enabled:
            with agreement_trends_lock:
                cached_payload = agreement_trends_cache["payload"]
                cached_ts = cast(float, agreement_trends_cache["ts"])
                cache_is_valid = cached_payload is not None and (
                    now - cached_ts < deps._AGREEMENTS_SUMMARY_TTL_SECONDS
                )
            if cache_is_valid and isinstance(cached_payload, dict):
                return _cacheable_json_response(
                    cached_payload,
                    max_age=deps._AGREEMENTS_SUMMARY_TTL_SECONDS,
                )

        db = deps.db
        schema_prefix = deps._schema_prefix()
        ownership_mix_rows = db.session.execute(
            text(
                f"""
                SELECT year, target_bucket, deal_count, total_transaction_value
                FROM {schema_prefix}agreement_ownership_mix_summary
                ORDER BY year ASC, target_bucket ASC
                """
            )
        ).mappings().all()
        ownership_deal_size_rows = db.session.execute(
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
        target_industry_rows = db.session.execute(
            text(
                f"""
                SELECT year, industry, deal_count, total_transaction_value
                FROM {schema_prefix}agreement_target_industry_summary
                ORDER BY year ASC, industry ASC
                """
            )
        ).mappings().all()
        industry_pairing_rows = db.session.execute(
            text(
                f"""
                SELECT target_industry, acquirer_industry, deal_count, total_transaction_value
                FROM {schema_prefix}agreement_industry_pairing_summary
                ORDER BY deal_count DESC, total_transaction_value DESC, target_industry ASC, acquirer_industry ASC
                """
            )
        ).mappings().all()
        naics_sector_rows = db.session.execute(
            text(
                f"""
                SELECT sector_code, sector_desc
                FROM {schema_prefix}naics_sectors
                """
            )
        ).mappings().all()
        naics_sub_sector_rows = db.session.execute(
            text(
                f"""
                SELECT sub_sector_code, sub_sector_desc
                FROM {schema_prefix}naics_sub_sectors
                """
            )
        ).mappings().all()
        naics_label_by_code: dict[str, str] = {}
        for row in naics_sector_rows:
            sector_code = row.get("sector_code")
            sector_desc = row.get("sector_desc")
            if sector_code is None or not isinstance(sector_desc, str):
                continue
            naics_label_by_code[str(sector_code)] = sector_desc
        for row in naics_sub_sector_rows:
            sub_sector_code = row.get("sub_sector_code")
            sub_sector_desc = row.get("sub_sector_desc")
            if sub_sector_code is None or not isinstance(sub_sector_desc, str):
                continue
            naics_label_by_code[str(sub_sector_code)] = sub_sector_desc

        ownership_mix_by_year: dict[int, dict[str, object]] = {}
        for row in ownership_mix_rows:
            year = deps._to_int(row.get("year"))
            year_row = ownership_mix_by_year.setdefault(
                year,
                {
                    "year": year,
                    "public_deal_count": 0,
                    "private_deal_count": 0,
                    "public_total_transaction_value": 0.0,
                    "private_total_transaction_value": 0.0,
                },
            )
            target_bucket = str(row.get("target_bucket") or "")
            if target_bucket == "public":
                year_row["public_deal_count"] = deps._to_int(row.get("deal_count"))
                year_row["public_total_transaction_value"] = _to_float_or_none(
                    row.get("total_transaction_value")
                ) or 0.0
            elif target_bucket == "private":
                year_row["private_deal_count"] = deps._to_int(row.get("deal_count"))
                year_row["private_total_transaction_value"] = _to_float_or_none(
                    row.get("total_transaction_value")
                ) or 0.0

        ownership_deal_size_by_year: dict[int, dict[str, object]] = {}
        for row in ownership_deal_size_rows:
            year = deps._to_int(row.get("year"))
            year_row = ownership_deal_size_by_year.setdefault(
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
            target_bucket = str(row.get("target_bucket") or "")
            if target_bucket == "public":
                year_row["public_deal_count"] = deps._to_int(row.get("deal_count"))
                year_row["public_p25_transaction_value"] = _to_float_or_none(
                    row.get("p25_transaction_value")
                )
                year_row["public_median_transaction_value"] = _to_float_or_none(
                    row.get("median_transaction_value")
                )
                year_row["public_p75_transaction_value"] = _to_float_or_none(
                    row.get("p75_transaction_value")
                )
            elif target_bucket == "private":
                year_row["private_deal_count"] = deps._to_int(row.get("deal_count"))
                year_row["private_p25_transaction_value"] = _to_float_or_none(
                    row.get("p25_transaction_value")
                )
                year_row["private_median_transaction_value"] = _to_float_or_none(
                    row.get("median_transaction_value")
                )
                year_row["private_p75_transaction_value"] = _to_float_or_none(
                    row.get("p75_transaction_value")
                )

        buyer_matrix_lookup = {
            (
                str(row.get("target_bucket") or ""),
                str(row.get("buyer_bucket") or ""),
            ): row
            for row in buyer_matrix_rows
        }
        buyer_matrix = []
        for target_bucket in ("public", "private"):
            for buyer_bucket in (
                "public_buyer",
                "private_strategic",
                "private_equity",
                "other",
            ):
                row = buyer_matrix_lookup.get((target_bucket, buyer_bucket))
                buyer_matrix.append(
                    {
                        "target_bucket": target_bucket,
                        "buyer_bucket": buyer_bucket,
                        "deal_count": deps._to_int(row.get("deal_count")) if row else 0,
                        "median_transaction_value": (
                            _to_float_or_none(row.get("median_transaction_value"))
                            if row
                            else None
                        ),
                    }
                )

        payload: dict[str, object] = {
            "ownership": {
                "mix_by_year": [
                    ownership_mix_by_year[year]
                    for year in sorted(ownership_mix_by_year.keys())
                ],
                "deal_size_by_year": [
                    ownership_deal_size_by_year[year]
                    for year in sorted(ownership_deal_size_by_year.keys())
                ],
                "buyer_type_matrix": buyer_matrix,
            },
            "industries": {
                "target_industries_by_year": [
                    {
                        "year": deps._to_int(row.get("year")),
                        "industry": _normalize_industry_label(
                            row.get("industry"),
                            label_by_code=naics_label_by_code,
                        ),
                        "deal_count": deps._to_int(row.get("deal_count")),
                        "total_transaction_value": _to_float_or_none(
                            row.get("total_transaction_value")
                        ) or 0.0,
                    }
                    for row in target_industry_rows
                ],
                "pairings": [
                    {
                        "target_industry": _normalize_industry_label(
                            row.get("target_industry"),
                            label_by_code=naics_label_by_code,
                        ),
                        "acquirer_industry": _normalize_industry_label(
                            row.get("acquirer_industry"),
                            label_by_code=naics_label_by_code,
                        ),
                        "deal_count": deps._to_int(row.get("deal_count")),
                        "total_transaction_value": _to_float_or_none(
                            row.get("total_transaction_value")
                        ) or 0.0,
                    }
                    for row in industry_pairing_rows
                ],
            },
        }
        if cache_enabled:
            with agreement_trends_lock:
                agreement_trends_cache["payload"] = payload
                agreement_trends_cache["ts"] = now
        return _cacheable_json_response(
            cast(dict[str, object], payload),
            max_age=deps._AGREEMENTS_SUMMARY_TTL_SECONDS,
        )

    def get_filter_options() -> tuple[Response, int] | Response:
        def build_clause_types_payload() -> dict[str, object]:
            l1_rows = (
                deps.db.session.query(
                    deps.TaxonomyL1.standard_id,
                    deps.TaxonomyL1.label,
                ).all()
            )
            l2_rows = (
                deps.db.session.query(
                    deps.TaxonomyL2.standard_id,
                    deps.TaxonomyL2.label,
                    deps.TaxonomyL2.parent_id,
                ).all()
            )
            l3_rows = (
                deps.db.session.query(
                    deps.TaxonomyL3.standard_id,
                    deps.TaxonomyL3.label,
                    deps.TaxonomyL3.parent_id,
                ).all()
            )

            l2_by_parent: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for standard_id, label, parent_id in l2_rows:
                if (
                    not isinstance(standard_id, str)
                    or not isinstance(label, str)
                    or not isinstance(parent_id, str)
                ):
                    raise ValueError("taxonomy_l2 has invalid parent_id or label.")
                l2_by_parent[parent_id].append((standard_id, label))

            l3_by_parent: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for standard_id, label, parent_id in l3_rows:
                if (
                    not isinstance(standard_id, str)
                    or not isinstance(label, str)
                    or not isinstance(parent_id, str)
                ):
                    raise ValueError("taxonomy_l3 has invalid parent_id or label.")
                l3_by_parent[parent_id].append((standard_id, label))

            validated_l1_rows: list[tuple[str, str]] = []
            for standard_id, label in l1_rows:
                if not isinstance(standard_id, str) or not isinstance(label, str):
                    raise ValueError("taxonomy_l1 has invalid standard_id or label.")
                validated_l1_rows.append((standard_id, label))

            tree: dict[str, object] = {}
            for l1_standard_id, l1_label in sorted(validated_l1_rows, key=lambda row: row[1]):
                l2_children: dict[str, object] = {}
                for l2_standard_id, l2_label in sorted(
                    l2_by_parent.get(l1_standard_id, []),
                    key=lambda row: row[1],
                ):
                    l3_children: dict[str, object] = {}
                    for l3_standard_id, l3_label in sorted(
                        l3_by_parent.get(l2_standard_id, []),
                        key=lambda row: row[1],
                    ):
                        l3_children[l3_label] = {"id": l3_standard_id}
                    l2_children[l2_label] = {
                        "id": l2_standard_id,
                        "children": l3_children,
                    }
                tree[l1_label] = {"id": l1_standard_id, "children": l2_children}

            return tree

        allowed_fields = {
            "targets",
            "acquirers",
            "target_counsels",
            "acquirer_counsels",
            "target_industries",
            "acquirer_industries",
            "clause_types",
        }
        requested_fields_raw = [
            field.strip()
            for field in request.args.getlist("fields")
            if isinstance(field, str) and field.strip()
        ]
        if requested_fields_raw:
            unknown_fields = sorted(set(requested_fields_raw) - allowed_fields)
            if unknown_fields:
                abort(400, description=f"Unsupported filter option fields: {', '.join(unknown_fields)}.")
            requested_fields = tuple(dict.fromkeys(requested_fields_raw))
        else:
            requested_fields = (
                "targets",
                "acquirers",
                "target_counsels",
                "acquirer_counsels",
                "target_industries",
                "acquirer_industries",
                "clause_types",
            )

        now = deps.time.time()
        with deps._filter_options_lock:
            cached_payload = deps._filter_options_cache["payload"]
            cached_ts = deps._filter_options_cache["ts"]
            cache_is_valid = cached_payload is not None and (
                now - cached_ts < deps._FILTER_OPTIONS_TTL_SECONDS
            )
        if cache_is_valid and not requested_fields_raw:
            resp = jsonify(cached_payload)
            resp.headers["Cache-Control"] = (
                f"public, max-age={deps._FILTER_OPTIONS_TTL_SECONDS}"
            )
            return resp, 200

        db = deps.db
        summary_fields = tuple(
            field_name for field_name in requested_fields if field_name != "clause_types"
        )
        payload: dict[str, object] = {}

        if summary_fields:
            rows = (
                db.session.execute(
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
                        field_values.append(option_value)

        if "clause_types" in requested_fields:
            payload["clause_types"] = build_clause_types_payload()

        if not requested_fields_raw:
            with deps._filter_options_lock:
                deps._filter_options_cache["payload"] = payload
                deps._filter_options_cache["ts"] = now

        resp = jsonify(payload)
        resp.headers["Cache-Control"] = f"public, max-age={deps._FILTER_OPTIONS_TTL_SECONDS}"
        return resp, 200

    def get_filter_option_values(field_name: str) -> tuple[Response, int] | Response:
        normalized_field_name = field_name.strip().lower()
        if normalized_field_name not in {"target", "acquirer"}:
            abort(404)

        query = str(request.args.get("query") or "").strip()
        limit_raw = request.args.get("limit")
        try:
            limit = int(limit_raw) if limit_raw is not None else 100
        except (TypeError, ValueError):
            abort(400, description="Invalid limit.")
        limit = max(1, min(limit, 200))

        summary_field_name = "targets" if normalized_field_name == "target" else "acquirers"
        params: dict[str, object] = {
            "field_name": summary_field_name,
            "limit_value": limit,
        }
        where_sql = "field_name = :field_name"
        if query:
            where_sql += " AND option_value LIKE :query_like"
            params["query_like"] = f"{query}%"
        rows = (
            deps.db.session.execute(
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
            )
            .all()
        )
        payload = {
            "options": [
                cast(str, row[0])
                for row in rows
                if isinstance(row[0], str) and row[0].strip()
            ]
        }
        return jsonify(payload), 200

    target_app.add_url_rule(
        "/v1/agreements-index", view_func=get_agreements_index, methods=["GET"]
    )
    target_app.add_url_rule(
        "/v1/agreements-summary", view_func=get_agreements_summary, methods=["GET"]
    )
    target_app.add_url_rule(
        "/v1/counsel-leaderboards",
        view_func=get_counsel_leaderboards,
        methods=["GET"],
    )
    target_app.add_url_rule(
        "/v1/agreement-trends",
        view_func=get_agreement_trends,
        methods=["GET"],
    )
    target_app.add_url_rule(
        "/v1/agreements-status-summary",
        view_func=get_agreements_status_summary,
        methods=["GET"],
    )
    target_app.add_url_rule(
        "/v1/agreements-deal-types-summary",
        view_func=get_agreements_deal_types_summary,
        methods=["GET"],
    )
    target_app.add_url_rule(
        "/v1/filter-options", view_func=get_filter_options, methods=["GET"]
    )
    target_app.add_url_rule(
        "/v1/filter-options/<string:field_name>",
        view_func=get_filter_option_values,
        methods=["GET"],
    )

    return agreements_blp, sections_blp, agreement_search_blp
