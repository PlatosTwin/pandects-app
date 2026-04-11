from __future__ import annotations

from decimal import Decimal
from datetime import date, datetime
from threading import Lock
from typing import Any, Protocol, cast

from flask import Flask, Response, abort, jsonify, request
from flask.views import MethodView
from flask_smorest import Blueprint
from sqlalchemy import and_, asc, func, or_, text

from backend.counsel_leaderboards import build_counsel_leaderboards_from_assignments
from backend.filtering import (
    build_canonical_counsel_agreement_uuid_subquery,
    build_transaction_price_bucket_filter,
)
from backend.routes.deps import AgreementsDeps
from backend.schemas.public_api import (
    AgreementArgsPayload,
    AgreementArgsSchema,
    AgreementResponseSchema,
    AgreementsBulkArgsPayload,
    AgreementsBulkArgsSchema,
    AgreementsIndexArgsSchema,
    AgreementsListResponseSchema,
    SectionResponseSchema,
    TaxClauseListResponseSchema,
)


class _PublicEligibleAgreementModel(Protocol):
    __table__: Any
    verified: Any


def _agreement_is_public_eligible_expr(agreements: _PublicEligibleAgreementModel) -> object:
    agreement_table = agreements.__table__
    gated_col = agreement_table.c.get("gated")
    if gated_col is None:
        return text("1 = 1")
    return or_(
        func.coalesce(gated_col, 0) != 1,
        func.coalesce(agreements.verified, 0) == 1,
    )


def _to_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float, Decimal, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_industry_label(raw_value: object, *, label_by_code: dict[str, str]) -> str:
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return ""
    return label_by_code.get(raw_text, raw_text)


def _tax_clause_rows(
    deps: AgreementsDeps,
    *,
    agreement_uuid: str | None = None,
    section_uuid: str | None = None,
) -> list[dict[str, object]]:
    agreements = deps.Agreements
    clauses = deps.Clauses
    assignments = deps.TaxClauseAssignment
    sections = deps.Sections
    xml = deps.XML
    db = deps.db
    agreement_cols = agreements.__table__.c
    clause_cols = clauses.__table__.c
    section_cols = sections.__table__.c

    query = (
        db.session.query(
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
        .join(
            agreements,
            agreement_cols["agreement_uuid"] == clause_cols["agreement_uuid"],
        )
        .join(
            sections,
            and_(
                section_cols["section_uuid"] == clause_cols["section_uuid"],
                section_cols["agreement_uuid"] == clause_cols["agreement_uuid"],
                section_cols["xml_version"] == clause_cols["xml_version"],
            ),
        )
        .join(
            xml,
            deps._section_latest_xml_join_condition(),
        )
        .outerjoin(
            assignments,
            assignments.clause_uuid == clause_cols["clause_uuid"],
        )
        .filter(
            clause_cols["module"] == "tax",
            _agreement_is_public_eligible_expr(agreements),
        )
    )
    if agreement_uuid is not None:
        query = query.filter(clause_cols["agreement_uuid"] == agreement_uuid)
    if section_uuid is not None:
        query = query.filter(clause_cols["section_uuid"] == section_uuid)

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


_METADATA_FIELD_COVERAGE_CONFIG = (
    {
        "field": "transaction_consideration",
        "label": "Consideration",
        "eligible_sql": "1 = 1",
        "covered_sql": "a.transaction_consideration IS NOT NULL AND TRIM(a.transaction_consideration) <> ''",
        "note": "Expected for all eligible agreements.",
    },
    {
        "field": "transaction_price_total",
        "label": "Total price",
        "eligible_sql": "a.transaction_consideration IS NOT NULL AND TRIM(a.transaction_consideration) <> ''",
        "covered_sql": "a.transaction_price_total IS NOT NULL AND TRIM(a.transaction_price_total) <> ''",
        "note": "Derived from consideration and populated price components; some mixed deals legitimately have no total.",
    },
    {
        "field": "transaction_price_cash",
        "label": "Cash price",
        "eligible_sql": "COALESCE(a.transaction_consideration, '') IN ('cash', 'mixed')",
        "covered_sql": "a.transaction_price_cash IS NOT NULL AND TRIM(a.transaction_price_cash) <> ''",
        "note": "Only applies to cash or mixed deals.",
    },
    {
        "field": "transaction_price_stock",
        "label": "Stock price",
        "eligible_sql": "COALESCE(a.transaction_consideration, '') IN ('stock', 'mixed')",
        "covered_sql": "a.transaction_price_stock IS NOT NULL AND TRIM(a.transaction_price_stock) <> ''",
        "note": "Only applies to stock or mixed deals.",
    },
    {
        "field": "transaction_price_assets",
        "label": "Asset price",
        "eligible_sql": "COALESCE(a.transaction_consideration, '') = 'mixed'",
        "covered_sql": "a.transaction_price_assets IS NOT NULL AND TRIM(a.transaction_price_assets) <> ''",
        "note": "Shown only against mixed deals; null can still be valid when no asset component exists.",
    },
    {
        "field": "target_type",
        "label": "Target type",
        "eligible_sql": "1 = 1",
        "covered_sql": "a.target_type IS NOT NULL AND TRIM(a.target_type) <> ''",
        "note": "Expected for all eligible agreements.",
    },
    {
        "field": "acquirer_type",
        "label": "Acquirer type",
        "eligible_sql": "1 = 1",
        "covered_sql": "a.acquirer_type IS NOT NULL AND TRIM(a.acquirer_type) <> ''",
        "note": "Expected for all eligible agreements.",
    },
    {
        "field": "target_counsel",
        "label": "Target counsel",
        "eligible_sql": "1 = 1",
        "covered_sql": "a.target_counsel IS NOT NULL AND TRIM(a.target_counsel) <> ''",
        "note": "Optional in sourcing, but counted when present.",
    },
    {
        "field": "acquirer_counsel",
        "label": "Acquirer counsel",
        "eligible_sql": "1 = 1",
        "covered_sql": "a.acquirer_counsel IS NOT NULL AND TRIM(a.acquirer_counsel) <> ''",
        "note": "Optional in sourcing, but counted when present.",
    },
    {
        "field": "target_pe",
        "label": "Target PE",
        "eligible_sql": "1 = 1",
        "covered_sql": "a.target_pe IS NOT NULL",
        "note": "Optional in sourcing, but counted when present.",
    },
    {
        "field": "acquirer_pe",
        "label": "Acquirer PE",
        "eligible_sql": "1 = 1",
        "covered_sql": "a.acquirer_pe IS NOT NULL",
        "note": "Optional in sourcing, but counted when present.",
    },
    {
        "field": "target_industry",
        "label": "Target industry",
        "eligible_sql": "1 = 1",
        "covered_sql": "a.target_industry IS NOT NULL AND TRIM(a.target_industry) <> ''",
        "note": "Optional in sourcing, but counted when present.",
    },
    {
        "field": "acquirer_industry",
        "label": "Acquirer industry",
        "eligible_sql": "1 = 1",
        "covered_sql": "a.acquirer_industry IS NOT NULL AND TRIM(a.acquirer_industry) <> ''",
        "note": "Optional in sourcing, but counted when present.",
    },
    {
        "field": "announce_date",
        "label": "Announce date",
        "eligible_sql": "1 = 1",
        "covered_sql": "a.announce_date IS NOT NULL AND TRIM(a.announce_date) <> ''",
        "note": "Optional in sourcing, but counted when present.",
    },
    {
        "field": "close_date",
        "label": "Close date",
        "eligible_sql": "1 = 1",
        "covered_sql": "a.close_date IS NOT NULL AND TRIM(a.close_date) <> ''",
        "note": "Optional in sourcing, but counted when present.",
    },
    {
        "field": "deal_status",
        "label": "Deal status",
        "eligible_sql": "1 = 1",
        "covered_sql": "a.deal_status IS NOT NULL AND TRIM(a.deal_status) <> ''",
        "note": "Optional in sourcing, but counted when present.",
    },
    {
        "field": "attitude",
        "label": "Attitude",
        "eligible_sql": "1 = 1",
        "covered_sql": "a.attitude IS NOT NULL AND TRIM(a.attitude) <> ''",
        "note": "Optional in sourcing, but counted when present.",
    },
    {
        "field": "deal_type",
        "label": "Deal type",
        "eligible_sql": "1 = 1",
        "covered_sql": "a.deal_type IS NOT NULL AND TRIM(a.deal_type) <> ''",
        "note": "Expected for all eligible agreements.",
    },
    {
        "field": "purpose",
        "label": "Purpose",
        "eligible_sql": "1 = 1",
        "covered_sql": "a.purpose IS NOT NULL AND TRIM(a.purpose) <> ''",
        "note": "Optional in sourcing, but counted when present.",
    },
)


def register_agreements_routes(target_app: Flask, *, deps: AgreementsDeps) -> tuple[Blueprint, Blueprint]:
    agreement_trends_cache: dict[str, object] = {"ts": 0.0, "payload": None}
    agreement_trends_lock = Lock()
    counsel_leaderboards_cache: dict[str, object] = {"ts": 0.0, "payload": None}
    counsel_leaderboards_lock = Lock()

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

    def get_metadata_field_coverage() -> list[dict[str, object]]:
        coverage_rows: list[dict[str, object]] = []
        agreement_where_parts = ["1 = 1"]
        agreement_columns = deps.Agreements.__table__.c
        if agreement_columns.get("status") is not None:
            agreement_where_parts.append("COALESCE(LOWER(a.status), '') <> 'invalid'")
        if agreement_columns.get("gated") is not None and agreement_columns.get("verified") is not None:
            agreement_where_parts.append(
                "NOT (COALESCE(a.gated, 0) = 1 AND COALESCE(a.verified, 0) = 0)"
            )
        ingested_where = "\n                      AND ".join(agreement_where_parts)
        processed_where_parts = [*agreement_where_parts]
        processed_where_parts.append(
            "EXISTS ("
            f"SELECT 1 FROM {deps._schema_prefix()}xml x "
            "WHERE x.agreement_uuid = a.agreement_uuid "
            "AND x.latest = 1 "
            "AND (x.status IS NULL OR x.status = 'verified')"
            ")"
        )
        processed_where = "\n                      AND ".join(processed_where_parts)
        agreements_table = f"{deps._schema_prefix()}agreements"
        aggregate_select_lines: list[str] = []
        for config in _METADATA_FIELD_COVERAGE_CONFIG:
            field = cast(str, config["field"])
            eligible_sql = cast(str, config["eligible_sql"])
            covered_sql = cast(str, config["covered_sql"])
            aggregate_select_lines.extend(
                [
                    (
                        "SUM(CASE WHEN a._coverage_scope_ingested = 1 "
                        f"AND {eligible_sql} THEN 1 ELSE 0 END) "
                        f"AS {field}_ingested_eligible_agreements"
                    ),
                    (
                        "SUM(CASE WHEN a._coverage_scope_ingested = 1 "
                        f"AND {eligible_sql} AND {covered_sql} THEN 1 ELSE 0 END) "
                        f"AS {field}_ingested_covered_agreements"
                    ),
                    (
                        "SUM(CASE WHEN a._coverage_scope_processed = 1 "
                        f"AND {eligible_sql} THEN 1 ELSE 0 END) "
                        f"AS {field}_processed_eligible_agreements"
                    ),
                    (
                        "SUM(CASE WHEN a._coverage_scope_processed = 1 "
                        f"AND {eligible_sql} AND {covered_sql} THEN 1 ELSE 0 END) "
                        f"AS {field}_processed_covered_agreements"
                    ),
                ]
            )
        aggregate_select = ",\n                            ".join(aggregate_select_lines)
        row = (
            deps.db.session.execute(
                text(
                    f"""
                    SELECT
                        {aggregate_select}
                    FROM (
                        SELECT
                            a.*,
                            1 AS _coverage_scope_ingested,
                            0 AS _coverage_scope_processed
                        FROM {agreements_table} a
                        WHERE {ingested_where}
                        UNION ALL
                        SELECT
                            a.*,
                            0 AS _coverage_scope_ingested,
                            1 AS _coverage_scope_processed
                        FROM {agreements_table} a
                        WHERE {processed_where}
                    ) a
                    """
                )
            )
            .mappings()
            .first()
        )
        row_dict = deps._row_mapping_as_dict(cast(object, row)) if row is not None else {}

        for config in _METADATA_FIELD_COVERAGE_CONFIG:
            field = cast(str, config["field"])
            ingested_eligible_agreements = deps._to_int(
                cast(object, row_dict.get(f"{field}_ingested_eligible_agreements"))
            ) or 0
            ingested_covered_agreements = deps._to_int(
                cast(object, row_dict.get(f"{field}_ingested_covered_agreements"))
            ) or 0
            processed_eligible_agreements = deps._to_int(
                cast(object, row_dict.get(f"{field}_processed_eligible_agreements"))
            ) or 0
            processed_covered_agreements = deps._to_int(
                cast(object, row_dict.get(f"{field}_processed_covered_agreements"))
            ) or 0
            ingested_coverage_pct = (
                round((ingested_covered_agreements / ingested_eligible_agreements) * 100, 1)
                if ingested_eligible_agreements > 0
                else None
            )
            processed_coverage_pct = (
                round((processed_covered_agreements / processed_eligible_agreements) * 100, 1)
                if processed_eligible_agreements > 0
                else None
            )
            coverage_rows.append(
                {
                    "field": config["field"],
                    "label": config["label"],
                    "ingested_eligible_agreements": ingested_eligible_agreements,
                    "ingested_covered_agreements": ingested_covered_agreements,
                    "ingested_coverage_pct": ingested_coverage_pct,
                    "processed_eligible_agreements": processed_eligible_agreements,
                    "processed_covered_agreements": processed_covered_agreements,
                    "processed_coverage_pct": processed_coverage_pct,
                    "note": config["note"],
                }
            )

        return coverage_rows

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

        agreements = deps.Agreements
        db = deps.db
        year_expr = deps._agreement_year_expr()
        sort_map = {
            "year": year_expr,
            "target": agreements.target,
            "acquirer": agreements.acquirer,
        }
        sort_column = sort_map.get(sort_by, year_expr)
        sort_direction = sort_dir.lower()
        order_by = sort_column.desc() if sort_direction == "desc" else sort_column.asc()

        q = (
            db.session.query(
                agreements.agreement_uuid,
                year_expr.label("year"),
                agreements.target,
                agreements.acquirer,
                agreements.verified,
            )
            .join(deps.XML, deps._agreement_latest_xml_join_condition())
            .filter(deps.XML.status == "verified")
            .filter(_agreement_is_public_eligible_expr(agreements))
        )
        count_q = (
            db.session.query(func.count(agreements.agreement_uuid))
            .select_from(agreements)
            .join(deps.XML, deps._agreement_latest_xml_join_condition())
            .filter(deps.XML.status == "verified")
            .filter(_agreement_is_public_eligible_expr(agreements))
        )

        if query:
            if query.isdigit():
                year_value = int(query)
                q = q.filter(year_expr == year_value)
                count_q = count_q.filter(year_expr == year_value)
            else:
                like = f"{query}%"
                filters = or_(
                    agreements.target.ilike(like),
                    agreements.acquirer.ilike(like),
                )
                q = q.filter(filters)
                count_q = count_q.filter(filters)

        q = q.order_by(order_by, agreements.agreement_uuid)

        total_count = deps._to_int(cast(object, count_q.scalar()))
        offset = (page - 1) * page_size
        items = q.offset(offset).limit(page_size).all()
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

    def get_agreements_status_summary() -> dict[str, object]:
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
        return {
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

    def get_agreements_deal_types_summary() -> dict[str, object]:
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
        return {"years": years}

    def get_agreements_summary() -> dict[str, int]:
        now = deps.time.time()
        with deps._agreements_summary_lock:
            cached_payload = deps._agreements_summary_cache["payload"]
            cached_ts = deps._agreements_summary_cache["ts"]
            cache_is_valid = cached_payload is not None and (
                now - cached_ts < deps._AGREEMENTS_SUMMARY_TTL_SECONDS
            )
        if cache_is_valid and cached_payload is not None:
            return cached_payload

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

        row_dict = deps._row_mapping_as_dict(cast(object, row)) if row is not None else {}
        payload: dict[str, int] = {
            "agreements": deps._to_int(cast(object, row_dict.get("agreements"))),
            "sections": deps._to_int(cast(object, row_dict.get("sections"))),
            "pages": deps._to_int(cast(object, row_dict.get("pages"))),
        }
        with deps._agreements_summary_lock:
            deps._agreements_summary_cache["payload"] = payload
            deps._agreements_summary_cache["ts"] = now

        return payload

    def get_counsel_leaderboards() -> dict[str, object]:
        now = deps.time.time()
        with counsel_leaderboards_lock:
            cached_payload = counsel_leaderboards_cache["payload"]
            cached_ts = cast(float, counsel_leaderboards_cache["ts"])
            cache_is_valid = cached_payload is not None and (
                now - cached_ts < deps._AGREEMENTS_SUMMARY_TTL_SECONDS
            )
        if cache_is_valid and isinstance(cached_payload, dict):
            return cached_payload

        db = deps.db
        rows = (
            db.session.execute(
                text(
                    f"""
                    SELECT
                        CASE ac.side
                            WHEN 'acquirer' THEN 'buy_side'
                            WHEN 'target' THEN 'sell_side'
                            ELSE ac.side
                        END AS side,
                        c.canonical_name_normalized AS counsel_key,
                        c.canonical_name AS counsel,
                        a.filing_date AS filing_date,
                        a.transaction_price_total AS transaction_price_total
                    FROM {deps._schema_prefix()}agreement_counsel ac
                    JOIN {deps._schema_prefix()}counsel c
                      ON c.counsel_id = ac.counsel_id
                    JOIN {deps._schema_prefix()}agreements a
                      ON a.agreement_uuid = ac.agreement_uuid
                    JOIN {deps._schema_prefix()}xml x
                      ON x.agreement_uuid = a.agreement_uuid
                     AND x.latest = 1
                     AND (x.status IS NULL OR x.status = 'verified')
                    WHERE NOT (COALESCE(a.gated, 0) = 1 AND COALESCE(a.verified, 0) = 0)
                    ORDER BY a.filing_date ASC, ac.agreement_uuid ASC, ac.side ASC, ac.position ASC
                    """
                )
            )
            .mappings()
            .all()
        )

        payload = build_counsel_leaderboards_from_assignments(
            [
                deps._row_mapping_as_dict(cast(object, row))
                for row in rows
            ]
        )
        with counsel_leaderboards_lock:
            counsel_leaderboards_cache["payload"] = payload
            counsel_leaderboards_cache["ts"] = now
        return payload

    def get_agreement_trends() -> dict[str, object]:
        now = deps.time.time()
        with agreement_trends_lock:
            cached_payload = agreement_trends_cache["payload"]
            cached_ts = cast(float, agreement_trends_cache["ts"])
            cache_is_valid = cached_payload is not None and (
                now - cached_ts < deps._AGREEMENTS_SUMMARY_TTL_SECONDS
            )
        if cache_is_valid and isinstance(cached_payload, dict):
            return cached_payload

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
        with agreement_trends_lock:
            agreement_trends_cache["payload"] = payload
            agreement_trends_cache["ts"] = now
        return cast(dict[str, object], payload)

    def get_filter_options() -> tuple[Response, int] | Response:
        now = deps.time.time()
        with deps._filter_options_lock:
            cached_payload = deps._filter_options_cache["payload"]
            cached_ts = deps._filter_options_cache["ts"]
            cache_is_valid = cached_payload is not None and (
                now - cached_ts < deps._FILTER_OPTIONS_TTL_SECONDS
            )
        if cache_is_valid:
            resp = jsonify(cached_payload)
            resp.headers["Cache-Control"] = (
                f"public, max-age={deps._FILTER_OPTIONS_TTL_SECONDS}"
            )
            return resp, 200

        db = deps.db
        agreements = deps.Agreements
        schema_prefix = deps._schema_prefix
        _xml_eligible = (
            "EXISTS ("
            "  SELECT 1 FROM {t}xml x "
            "  WHERE x.agreement_uuid = a.agreement_uuid "
            "    AND (x.status IS NULL OR x.status = 'verified')"
            ")"
        ).format(t=schema_prefix())
        _has_sections = (
            "EXISTS ("
            "  SELECT 1 FROM {t}sections s "
            "  WHERE s.agreement_uuid = a.agreement_uuid"
            ")"
        ).format(t=schema_prefix())
        _is_public_eligible = (
            "NOT (COALESCE(a.gated, 0) = 1 AND COALESCE(a.verified, 0) = 0)"
            if "gated" in agreements.__table__.c
            else "1 = 1"
        )

        targets = [
            cast(str, row[0])
            for row in db.session.execute(
                text(
                    f"""
                    SELECT DISTINCT a.target
                    FROM {schema_prefix()}agreements a
                    WHERE a.target IS NOT NULL
                      AND a.target <> ''
                      AND {_is_public_eligible}
                      AND {_has_sections}
                      AND {_xml_eligible}
                    ORDER BY a.target
                    """
                )
            ).fetchall()
        ]
        acquirers = [
            cast(str, row[0])
            for row in db.session.execute(
                text(
                    f"""
                    SELECT DISTINCT a.acquirer
                    FROM {schema_prefix()}agreements a
                    WHERE a.acquirer IS NOT NULL
                      AND a.acquirer <> ''
                      AND {_is_public_eligible}
                      AND {_has_sections}
                      AND {_xml_eligible}
                    ORDER BY a.acquirer
                    """
                )
            ).fetchall()
        ]
        target_counsels = [
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
                      AND {_is_public_eligible}
                      AND {_has_sections}
                      AND {_xml_eligible}
                    ORDER BY c.canonical_name
                    """
                )
            ).fetchall()
        ]
        acquirer_counsels = [
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
                      AND {_is_public_eligible}
                      AND {_has_sections}
                      AND {_xml_eligible}
                    ORDER BY c.canonical_name
                    """
                )
            ).fetchall()
        ]
        target_industries = [
            cast(str, row[0])
            for row in db.session.execute(
                text(
                    f"""
                    SELECT DISTINCT a.target_industry
                    FROM {schema_prefix()}agreements a
                    WHERE a.target_industry IS NOT NULL
                      AND a.target_industry <> ''
                      AND {_is_public_eligible}
                      AND {_has_sections}
                      AND {_xml_eligible}
                    ORDER BY a.target_industry
                    """
                )
            ).fetchall()
        ]
        acquirer_industries = [
            cast(str, row[0])
            for row in db.session.execute(
                text(
                    f"""
                    SELECT DISTINCT a.acquirer_industry
                    FROM {schema_prefix()}agreements a
                    WHERE a.acquirer_industry IS NOT NULL
                      AND a.acquirer_industry <> ''
                      AND {_is_public_eligible}
                      AND {_has_sections}
                      AND {_xml_eligible}
                    ORDER BY a.acquirer_industry
                    """
                )
            ).fetchall()
        ]

        payload = {
            "targets": targets,
            "acquirers": acquirers,
            "target_counsels": target_counsels,
            "acquirer_counsels": acquirer_counsels,
            "target_industries": target_industries,
            "acquirer_industries": acquirer_industries,
        }
        with deps._filter_options_lock:
            deps._filter_options_cache["payload"] = payload
            deps._filter_options_cache["ts"] = now

        resp = jsonify(payload)
        resp.headers["Cache-Control"] = f"public, max-age={deps._FILTER_OPTIONS_TTL_SECONDS}"
        return resp, 200

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

    return agreements_blp, sections_blp
