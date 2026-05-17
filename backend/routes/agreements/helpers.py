"""Module-level helpers for the agreements blueprint.

Pulled out of ``__init__`` to keep the main file focused on
``register_agreements_routes``. Every helper takes its dependencies as
explicit arguments — none capture closure state.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any, Mapping, Protocol, cast

from flask import Response, jsonify
from sqlalchemy import and_, asc, func, or_, text

from backend.filtering import (
    build_canonical_counsel_agreement_uuid_subquery,
    build_transaction_price_bucket_filter,
)
from backend.routes.deps import AgreementsDeps
from backend.schemas.public_api import AgreementsBulkArgsPayload
from backend.schemas.sections import SectionsArgsPayload
from backend.summary_specs import METADATA_FIELD_COVERAGE_CONFIG


class _PublicEligibleAgreementModel(Protocol):
    __table__: Any
    verified: Any


_METADATA_FIELD_COVERAGE_DISPLAY_ORDER = {
    config["field"]: index
    for index, config in enumerate(METADATA_FIELD_COVERAGE_CONFIG)
}


def _metadata_field_coverage_sort_key(row_dict: dict[str, object]) -> tuple[int, str]:
    field_name = row_dict.get("field_name")
    normalized_field_name = field_name if isinstance(field_name, str) else ""
    return (
        _METADATA_FIELD_COVERAGE_DISPLAY_ORDER.get(
            normalized_field_name,
            len(_METADATA_FIELD_COVERAGE_DISPLAY_ORDER),
        ),
        normalized_field_name,
    )


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


_SNIPPET_TAG_RE = re.compile(r"<[^>]+>")
_SNIPPET_WS_RE = re.compile(r"\s+")


def _build_agreement_search_match_query(
    deps: AgreementsDeps,
    *,
    parsed_args: SectionsArgsPayload,
) -> tuple[Any, bool]:
    db = deps.db
    agreements = deps.Agreements
    agreement_counsel = deps.AgreementCounsel
    counsel = deps.Counsel
    latest = deps.LatestSectionsSearch

    years = parsed_args["year"]
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
    agreement_uuid = parsed_args["agreement_uuid"]
    section_uuid = parsed_args["section_uuid"]

    q = (
        db.session.query(
            latest.section_uuid.label("section_uuid"),
            latest.agreement_uuid.label("agreement_uuid"),
            latest.filing_date.label("filing_date"),
            latest.target.label("target"),
            latest.acquirer.label("acquirer"),
            latest.article_title.label("article_title"),
            latest.section_title.label("section_title"),
            latest.section_standard_ids.label("section_standard_ids"),
        )
        .join(agreements, agreements.agreement_uuid == latest.agreement_uuid)
        .filter(_agreement_is_public_eligible_expr(agreements))
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
        expanded_standard_ids = list(
            deps._expand_taxonomy_standard_ids_cached(standard_ids_key)
        )
        if expanded_standard_ids:
            standard_ids_expanded = set(expanded_standard_ids) != set(standard_ids_key)
            q = q.filter(deps._standard_id_filter_expr(expanded_standard_ids))

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

    return q, standard_ids_expanded


def _strip_xml_snippet(xml_value: object, *, max_chars: int = 220) -> str | None:
    if not isinstance(xml_value, str) or not xml_value.strip():
        return None
    text_only = _SNIPPET_TAG_RE.sub(" ", xml_value)
    normalized = _SNIPPET_WS_RE.sub(" ", text_only).strip()
    if not normalized:
        return None
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 1].rstrip()}…"


def _apply_agreement_metadata_filters(
    q: Any,
    *,
    db: Any,
    agreements: Any,
    agreement_counsel: Any,
    counsel: Any,
    sections: Any,
    parsed_args: SectionsArgsPayload | AgreementsBulkArgsPayload,
) -> Any:
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
            )
            .exists()
        )
        q = q.filter(section_exists)

    return q


def _cacheable_json_response(
    payload: Mapping[str, object],
    *,
    max_age: int,
) -> tuple[Response, int]:
    resp = jsonify(payload)
    resp.headers["Cache-Control"] = f"public, max-age={max_age}"
    return resp, 200


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
