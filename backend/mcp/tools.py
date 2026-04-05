from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from flask import abort
from marshmallow import Schema, ValidationError, fields
from sqlalchemy import and_, asc, or_

from backend.auth.mcp_runtime import McpPrincipal
from backend.routes.deps import AgreementsDeps, SectionsServiceDeps
from backend.schemas.public_api import (
    AgreementArgsPayload,
    AgreementArgsSchema,
    AgreementsBulkArgsPayload,
    AgreementsBulkArgsSchema,
)
from backend.schemas.sections import SectionsArgsPayload, SectionsArgsSchema
from backend.services.sections_service import run_sections


class McpAgreementArgsSchema(AgreementArgsSchema):
    agreement_uuid = fields.Str(required=True)


@dataclass(frozen=True)
class McpToolResult:
    text: str
    structured_content: dict[str, object]


def _require_scope(principal: McpPrincipal, scope: str) -> None:
    if scope in principal.scopes:
        return
    raise PermissionError(f"Missing required scope: {scope}")


def _validate_payload(schema: Schema, payload: dict[str, object]) -> dict[str, object]:
    try:
        loaded = schema.load(payload)
    except ValidationError:
        raise
    return cast(dict[str, object], loaded)


def _list_agreements(
    deps: AgreementsDeps,
    *,
    principal: McpPrincipal,
    payload: dict[str, object],
) -> McpToolResult:
    _require_scope(principal, "agreements:search")
    parsed_args = cast(
        AgreementsBulkArgsPayload,
        cast(object, _validate_payload(AgreementsBulkArgsSchema(), payload)),
    )
    include_xml = parsed_args["include_xml"]
    if include_xml:
        _require_scope(principal, "agreements:read_fulltext")

    page_size = parsed_args["page_size"]
    if page_size < 1 or page_size > 100:
        page_size = 25

    after_agreement_uuid = deps._decode_agreements_cursor(parsed_args["cursor"])
    agreements = deps.Agreements
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
    q = db.session.query(*item_columns).join(xml, deps._agreement_latest_xml_join_condition())

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

    list_filters = (
        ("target", agreements.target),
        ("acquirer", agreements.acquirer),
        ("transaction_price_total", agreements.transaction_price_total),
        ("transaction_price_stock", agreements.transaction_price_stock),
        ("transaction_price_cash", agreements.transaction_price_cash),
        ("transaction_price_assets", agreements.transaction_price_assets),
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

    for key, column in (
        ("target_pe", agreements.target_pe),
        ("acquirer_pe", agreements.acquirer_pe),
    ):
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
        item = {
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
            item["xml"] = row_map.get("xml")
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
    if allow_fulltext:
        response["xml"] = xml_content
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


def _search_sections(
    deps: SectionsServiceDeps,
    *,
    principal: McpPrincipal,
    payload: dict[str, object],
) -> McpToolResult:
    _require_scope(principal, "sections:search")
    parsed_args = cast(
        SectionsArgsPayload,
        cast(object, _validate_payload(SectionsArgsSchema(), payload)),
    )
    response = run_sections(deps, ctx=principal.access_context, parsed_args=parsed_args)
    returned = response.get("results", [])
    count = len(returned) if isinstance(returned, list) else 0
    return McpToolResult(
        text=f"Returned {count} section(s).",
        structured_content=response,
    )


def tool_definitions() -> list[dict[str, object]]:
    return [
        {
            "name": "search_sections",
            "description": "Search merger agreement sections with structured filters.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "year": {"type": "array", "items": {"type": "integer"}},
                    "target": {"type": "array", "items": {"type": "string"}},
                    "acquirer": {"type": "array", "items": {"type": "string"}},
                    "standard_id": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "array", "items": {"type": "string"}},
                    "agreement_uuid": {"type": ["string", "null"]},
                    "section_uuid": {"type": ["string", "null"]},
                    "sort_by": {"type": "string"},
                    "sort_direction": {"type": "string"},
                    "page": {"type": "integer"},
                    "page_size": {"type": "integer"},
                },
                "additionalProperties": True,
            },
        },
        {
            "name": "list_agreements",
            "description": "List agreements with keyset pagination and structured filters.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "cursor": {"type": ["string", "null"]},
                    "page_size": {"type": "integer"},
                    "include_xml": {"type": "boolean"},
                    "year": {"type": "array", "items": {"type": "integer"}},
                    "target": {"type": "array", "items": {"type": "string"}},
                    "acquirer": {"type": "array", "items": {"type": "string"}},
                    "agreement_uuid": {"type": ["string", "null"]},
                    "section_uuid": {"type": ["string", "null"]},
                },
                "additionalProperties": True,
            },
        },
        {
            "name": "get_agreement",
            "description": "Fetch one agreement, returning redacted XML unless full-text scope is present.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agreement_uuid": {"type": "string"},
                    "focus_section_uuid": {"type": ["string", "null"]},
                    "neighbor_sections": {"type": "integer"},
                },
                "required": ["agreement_uuid"],
                "additionalProperties": False,
            },
        },
    ]


def call_tool(
    name: str,
    *,
    arguments: dict[str, object],
    principal: McpPrincipal,
    sections_service_deps: SectionsServiceDeps,
    agreements_deps: AgreementsDeps,
) -> McpToolResult:
    if name == "search_sections":
        return _search_sections(sections_service_deps, principal=principal, payload=arguments)
    if name == "list_agreements":
        return _list_agreements(agreements_deps, principal=principal, payload=arguments)
    if name == "get_agreement":
        return _get_agreement(agreements_deps, principal=principal, payload=arguments)
    raise KeyError(name)


__all__ = ["McpToolResult", "call_tool", "tool_definitions"]
