from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, cast

from flask import abort
from marshmallow import Schema, ValidationError, fields as ma_fields, validate
from sqlalchemy import and_, asc, desc, func, or_, text

from backend.auth.mcp_runtime import McpPrincipal
from backend.filtering import build_canonical_counsel_agreement_uuid_subquery, build_transaction_price_bucket_filter
from backend.routes.agreements import (
    _agreement_is_public_eligible_expr,
    _normalize_industry_label,
    _tax_clause_rows,
    _to_float_or_none,
)
from backend.routes.deps import AgreementsDeps, ReferenceDataDeps, SectionsServiceDeps
from backend.schemas.public_api import (
    AgreementArgsPayload,
    AgreementArgsSchema,
    AgreementsBulkArgsPayload,
    AgreementsBulkArgsSchema,
)
from backend.schemas.sections import SECTIONS_RESULT_METADATA_FIELDS, SectionsArgsPayload, SectionsArgsSchema
from backend.services.sections_service import run_sections

_SECTION_LIST_SORT_FIELDS = ("article_title", "section_title", "section_uuid")
_TRANSACTION_PRICE_BUCKET_OPTIONS = (
    "0 - 100M",
    "100M - 250M",
    "250M - 500M",
    "500M - 750M",
    "750M - 1B",
    "1B - 5B",
    "5B - 10B",
    "10B - 20B",
    "20B+",
)
_FILTER_OPTIONS_FIELDS = (
    "targets",
    "acquirers",
    "transaction_price_totals",
    "transaction_price_stocks",
    "transaction_price_cashes",
    "transaction_price_assets",
    "transaction_considerations",
    "target_types",
    "acquirer_types",
    "target_counsels",
    "acquirer_counsels",
    "target_industries",
    "acquirer_industries",
    "deal_statuses",
    "attitudes",
    "deal_types",
    "purposes",
    "target_pes",
    "acquirer_pes",
)
_STRUCTURED_FILTER_ARRAY_FIELDS = (
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
)


def _array_schema_for_filter(field_name: str) -> dict[str, object]:
    item_type = "integer" if field_name == "year" else "string"
    return {"type": "array", "items": {"type": item_type}}


def _enum_array_schema(
    values: tuple[str, ...],
    *,
    description: str | None = None,
    examples: list[object] | None = None,
) -> dict[str, object]:
    schema: dict[str, object] = {"type": "array", "items": {"type": "string", "enum": list(values)}}
    if description is not None:
        schema["description"] = description
    if examples is not None:
        schema["examples"] = examples
    return schema


def _one_of_choices(validators: list[object]) -> list[object] | None:
    for validator_obj in validators:
        if isinstance(validator_obj, validate.OneOf):
            return list(validator_obj.choices)
    return None


def _field_json_schema(field: ma_fields.Field) -> dict[str, object]:
    schema: dict[str, object]
    if isinstance(field, ma_fields.List):
        item_schema = _field_json_schema(field.inner)
        item_schema.pop("description", None)
        item_schema.pop("example", None)
        item_schema.pop("examples", None)
        schema = {"type": "array", "items": item_schema}
    elif isinstance(field, ma_fields.Int):
        schema = {"type": "integer"}
    elif isinstance(field, ma_fields.Bool):
        schema = {"type": "boolean"}
    elif isinstance(field, ma_fields.Float):
        schema = {"type": "number"}
    else:
        schema = {"type": "string"}

    enum_choices = _one_of_choices(list(getattr(field, "validators", [])))
    if enum_choices is not None:
        schema["enum"] = enum_choices

    if field.allow_none and "type" in schema:
        schema["type"] = [cast(object, schema["type"]), "null"]

    description = field.metadata.get("description")
    if isinstance(description, str) and description.strip():
        schema["description"] = description
    example = field.metadata.get("example")
    if example is not None:
        schema["example"] = example
    examples = field.metadata.get("examples")
    if isinstance(examples, list) and examples:
        schema["examples"] = examples
    return schema


def _schema_input_schema(
    schema: Schema,
    *,
    additional_properties: bool = False,
    field_overrides: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    properties: dict[str, object] = {}
    required: list[str] = []
    for field_name, field in schema.fields.items():
        field_schema = _field_json_schema(field)
        if field_overrides and field_name in field_overrides:
            field_schema.update(field_overrides[field_name])
        properties[field_name] = field_schema
        if field.required:
            required.append(field_name)
    payload: dict[str, object] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": additional_properties,
    }
    if required:
        payload["required"] = required
    return payload


def _filter_option_metadata() -> dict[str, dict[str, object]]:
    return {
        "targets": {"retrieval_parameter": "target", "applies_to": ["agreements", "sections"], "value_kind": "exact_string"},
        "acquirers": {"retrieval_parameter": "acquirer", "applies_to": ["agreements", "sections"], "value_kind": "exact_string"},
        "transaction_price_totals": {
            "retrieval_parameter": "transaction_price_total",
            "applies_to": ["agreements", "sections"],
            "value_kind": "bucket",
            "allowed_values": list(_TRANSACTION_PRICE_BUCKET_OPTIONS),
        },
        "transaction_price_stocks": {
            "retrieval_parameter": "transaction_price_stock",
            "applies_to": ["agreements", "sections"],
            "value_kind": "bucket",
            "allowed_values": list(_TRANSACTION_PRICE_BUCKET_OPTIONS),
        },
        "transaction_price_cashes": {
            "retrieval_parameter": "transaction_price_cash",
            "applies_to": ["agreements", "sections"],
            "value_kind": "bucket",
            "allowed_values": list(_TRANSACTION_PRICE_BUCKET_OPTIONS),
        },
        "transaction_price_assets": {
            "retrieval_parameter": "transaction_price_assets",
            "applies_to": ["agreements", "sections"],
            "value_kind": "bucket",
            "allowed_values": list(_TRANSACTION_PRICE_BUCKET_OPTIONS),
        },
        "transaction_considerations": {"retrieval_parameter": "transaction_consideration", "applies_to": ["agreements", "sections"], "value_kind": "exact_string"},
        "target_types": {"retrieval_parameter": "target_type", "applies_to": ["agreements", "sections"], "value_kind": "exact_string"},
        "acquirer_types": {"retrieval_parameter": "acquirer_type", "applies_to": ["agreements", "sections"], "value_kind": "exact_string"},
        "target_counsels": {"retrieval_parameter": "target_counsel", "applies_to": ["agreements", "sections"], "value_kind": "exact_string"},
        "acquirer_counsels": {"retrieval_parameter": "acquirer_counsel", "applies_to": ["agreements", "sections"], "value_kind": "exact_string"},
        "target_industries": {"retrieval_parameter": "target_industry", "applies_to": ["agreements", "sections"], "value_kind": "exact_string"},
        "acquirer_industries": {"retrieval_parameter": "acquirer_industry", "applies_to": ["agreements", "sections"], "value_kind": "exact_string"},
        "deal_statuses": {"retrieval_parameter": "deal_status", "applies_to": ["agreements", "sections"], "value_kind": "exact_string"},
        "attitudes": {"retrieval_parameter": "attitude", "applies_to": ["agreements", "sections"], "value_kind": "exact_string"},
        "deal_types": {"retrieval_parameter": "deal_type", "applies_to": ["agreements", "sections"], "value_kind": "exact_string"},
        "purposes": {"retrieval_parameter": "purpose", "applies_to": ["agreements", "sections"], "value_kind": "exact_string"},
        "target_pes": {
            "retrieval_parameter": "target_pe",
            "applies_to": ["agreements", "sections"],
            "value_kind": "string_boolean",
            "allowed_values": ["true", "false"],
        },
        "acquirer_pes": {
            "retrieval_parameter": "acquirer_pe",
            "applies_to": ["agreements", "sections"],
            "value_kind": "string_boolean",
            "allowed_values": ["true", "false"],
        },
    }


def _structured_filter_properties(*, include_cursor: bool = False, include_xml: bool = False) -> dict[str, object]:
    properties: dict[str, object] = {}
    if include_cursor:
        properties["cursor"] = {"type": ["string", "null"], "description": "Opaque cursor from a previous list_agreements call."}
    properties["page_size"] = {"type": "integer", "description": "Maximum number of results to return."}
    if include_xml:
        properties["include_xml"] = {
            "type": "boolean",
            "description": "When true, include full agreement XML. Requires agreements:read_fulltext.",
        }
    for field_name in _STRUCTURED_FILTER_ARRAY_FIELDS:
        if field_name == "year":
            properties[field_name] = {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Agreement filing years to include.",
            }
        elif field_name in {
            "transaction_price_total",
            "transaction_price_stock",
            "transaction_price_cash",
            "transaction_price_assets",
        }:
            properties[field_name] = _enum_array_schema(
                _TRANSACTION_PRICE_BUCKET_OPTIONS,
                description="Transaction-price bucket filters.",
                examples=[["100M - 250M", "250M - 500M"]],
            )
        elif field_name in {"target_pe", "acquirer_pe"}:
            properties[field_name] = _enum_array_schema(
                ("true", "false"),
                description="Boolean-like PE filter values encoded as strings.",
                examples=[["true"]],
            )
        else:
            properties[field_name] = _array_schema_for_filter(field_name)
    properties["agreement_uuid"] = {"type": ["string", "null"], "description": "Filter to one agreement UUID."}
    properties["section_uuid"] = {"type": ["string", "null"], "description": "Filter to one section UUID."}
    return properties


class McpAgreementArgsSchema(AgreementArgsSchema):
    agreement_uuid = ma_fields.Str(required=True)


class McpAgreementIdentifierSchema(Schema):
    agreement_uuid = ma_fields.Str(required=True)


class McpSectionArgsSchema(Schema):
    section_uuid = ma_fields.Str(required=True)


class McpSearchAgreementsArgsSchema(Schema):
    query = ma_fields.Str(load_default="")
    page = ma_fields.Int(load_default=1)
    page_size = ma_fields.Int(load_default=25)
    sort_by = ma_fields.Str(load_default="year", validate=validate.OneOf(["year", "target", "acquirer"]))
    sort_dir = ma_fields.Str(load_default="desc", validate=validate.OneOf(["asc", "desc"]))
    year = ma_fields.List(ma_fields.Int(), load_default=[])
    target = ma_fields.List(ma_fields.Str(), load_default=[])
    acquirer = ma_fields.List(ma_fields.Str(), load_default=[])
    transaction_price_total = ma_fields.List(ma_fields.Str(), load_default=[])
    transaction_price_stock = ma_fields.List(ma_fields.Str(), load_default=[])
    transaction_price_cash = ma_fields.List(ma_fields.Str(), load_default=[])
    transaction_price_assets = ma_fields.List(ma_fields.Str(), load_default=[])
    transaction_consideration = ma_fields.List(ma_fields.Str(), load_default=[])
    target_type = ma_fields.List(ma_fields.Str(), load_default=[])
    acquirer_type = ma_fields.List(ma_fields.Str(), load_default=[])
    target_counsel = ma_fields.List(ma_fields.Str(), load_default=[])
    acquirer_counsel = ma_fields.List(ma_fields.Str(), load_default=[])
    target_industry = ma_fields.List(ma_fields.Str(), load_default=[])
    acquirer_industry = ma_fields.List(ma_fields.Str(), load_default=[])
    deal_status = ma_fields.List(ma_fields.Str(), load_default=[])
    attitude = ma_fields.List(ma_fields.Str(), load_default=[])
    deal_type = ma_fields.List(ma_fields.Str(), load_default=[])
    purpose = ma_fields.List(ma_fields.Str(), load_default=[])
    target_pe = ma_fields.List(ma_fields.Str(), load_default=[])
    acquirer_pe = ma_fields.List(ma_fields.Str(), load_default=[])
    agreement_uuid = ma_fields.Str(load_default=None, allow_none=True)
    section_uuid = ma_fields.Str(load_default=None, allow_none=True)


class McpListAgreementSectionsArgsSchema(Schema):
    agreement_uuid = ma_fields.Str(required=True)
    standard_id = ma_fields.List(ma_fields.Str(), load_default=[])
    page = ma_fields.Int(load_default=1)
    page_size = ma_fields.Int(load_default=25)
    sort_by = ma_fields.Str(
        load_default="section_uuid",
        validate=validate.OneOf(list(_SECTION_LIST_SORT_FIELDS)),
    )
    sort_direction = ma_fields.Str(load_default="asc", validate=validate.OneOf(["asc", "desc"]))


class McpFilterOptionsArgsSchema(Schema):
    fields = cast(
        Any,
        ma_fields.List(
            ma_fields.Str(validate=validate.OneOf(list(_FILTER_OPTIONS_FIELDS))),
            load_default=[],
        ),
    )


@dataclass(frozen=True)
class McpToolResult:
    text: str
    structured_content: object


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


def _normalized_page(page: int) -> int:
    return page if page >= 1 else 1


def _normalized_page_size(page_size: int) -> int:
    return page_size if 1 <= page_size <= 100 else 25


def _build_taxonomy_tree(*, l1_model: object, l2_model: object, l3_model: object, deps: ReferenceDataDeps) -> dict[str, object]:
    db = deps.db
    l1_rows = cast(
        list[tuple[object, object]],
        db.session.query(
            cast(Any, l1_model).standard_id,
            cast(Any, l1_model).label,
        ).all(),
    )
    l2_rows = cast(
        list[tuple[object, object, object]],
        db.session.query(
            cast(Any, l2_model).standard_id,
            cast(Any, l2_model).label,
            cast(Any, l2_model).parent_id,
        ).all(),
    )
    l3_rows = cast(
        list[tuple[object, object, object]],
        db.session.query(
            cast(Any, l3_model).standard_id,
            cast(Any, l3_model).label,
            cast(Any, l3_model).parent_id,
        ).all(),
    )

    l2_by_parent: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for standard_id, label, parent_id in l2_rows:
        if isinstance(standard_id, str) and isinstance(label, str) and isinstance(parent_id, str):
            l2_by_parent[parent_id].append((standard_id, label))

    l3_by_parent: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for standard_id, label, parent_id in l3_rows:
        if isinstance(standard_id, str) and isinstance(label, str) and isinstance(parent_id, str):
            l3_by_parent[parent_id].append((standard_id, label))

    validated_l1_rows: list[tuple[str, str]] = []
    for standard_id, label in l1_rows:
        if isinstance(standard_id, str) and isinstance(label, str):
            validated_l1_rows.append((standard_id, label))

    tree: dict[str, object] = {}
    for l1_standard_id, l1_label in sorted(validated_l1_rows, key=lambda row: row[1]):
        l2_children: dict[str, object] = {}
        for l2_standard_id, l2_label in sorted(l2_by_parent.get(l1_standard_id, []), key=lambda row: row[1]):
            l3_children: dict[str, object] = {}
            for l3_standard_id, l3_label in sorted(l3_by_parent.get(l2_standard_id, []), key=lambda row: row[1]):
                l3_children[l3_label] = {"id": l3_standard_id}
            l2_children[l2_label] = {"id": l2_standard_id, "children": l3_children}
        tree[l1_label] = {"id": l1_standard_id, "children": l2_children}
    return tree


def _naics_payload(deps: ReferenceDataDeps) -> dict[str, object]:
    db = deps.db
    sector_rows = cast(
        list[tuple[object, object, object, object]],
        db.session.query(
            deps.NaicsSector.sector_code,
            deps.NaicsSector.sector_desc,
            deps.NaicsSector.sector_group,
            deps.NaicsSector.super_sector,
        ).all(),
    )
    sub_sector_rows = cast(
        list[tuple[object, object, object]],
        db.session.query(
            deps.NaicsSubSector.sub_sector_code,
            deps.NaicsSubSector.sub_sector_desc,
            deps.NaicsSubSector.sector_code,
        ).all(),
    )

    sector_by_code: dict[int, dict[str, object]] = {}
    for sector_code, sector_desc, sector_group, super_sector in sector_rows:
        if not isinstance(sector_code, int):
            continue
        sector_by_code[sector_code] = {
            "sector_code": str(sector_code),
            "sector_desc": str(sector_desc or ""),
            "sector_group": str(sector_group or ""),
            "super_sector": str(super_sector or ""),
            "sub_sectors": [],
        }

    for sub_sector_code, sub_sector_desc, sector_code in sub_sector_rows:
        if not isinstance(sector_code, int) or not isinstance(sub_sector_code, int):
            continue
        parent = sector_by_code.get(sector_code)
        if parent is None:
            continue
        cast(list[dict[str, str]], parent["sub_sectors"]).append(
            {
                "sub_sector_code": str(sub_sector_code),
                "sub_sector_desc": str(sub_sector_desc or ""),
            }
        )

    sectors: list[dict[str, object]] = []
    for code in sorted(sector_by_code.keys()):
        sector = sector_by_code[code]
        sector["sub_sectors"] = sorted(
            cast(list[dict[str, str]], sector["sub_sectors"]),
            key=lambda row: int(row["sub_sector_code"]),
        )
        sectors.append(sector)
    return {"sectors": sectors}


def _counsel_payload(deps: ReferenceDataDeps) -> dict[str, object]:
    rows = cast(
        list[tuple[object, object]],
        deps.db.session.query(
            deps.Counsel.counsel_id,
            deps.Counsel.canonical_name,
        )
        .order_by(deps.Counsel.canonical_name.asc(), deps.Counsel.counsel_id.asc())
        .all(),
    )
    payload_rows: list[dict[str, object]] = []
    for counsel_id, canonical_name in rows:
        if isinstance(counsel_id, int) and isinstance(canonical_name, str):
            payload_rows.append(
                {
                    "counsel_id": counsel_id,
                    "canonical_name": canonical_name,
                }
            )
    return {"counsel": payload_rows}


def _agreements_summary_payload(deps: AgreementsDeps) -> dict[str, object]:
    row = deps.db.session.execute(
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
    return {
        "agreements": deps._to_int(cast(object, row_dict.get("agreements"))),
        "sections": deps._to_int(cast(object, row_dict.get("sections"))),
        "pages": deps._to_int(cast(object, row_dict.get("pages"))),
    }


def _agreement_trends_payload(deps: AgreementsDeps, *, reference_data_deps: ReferenceDataDeps) -> dict[str, object]:
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
        if sector_code is not None and isinstance(sector_desc, str):
            naics_label_by_code[str(sector_code)] = sector_desc
    for row in naics_sub_sector_rows:
        sub_sector_code = row.get("sub_sector_code")
        sub_sector_desc = row.get("sub_sector_desc")
        if sub_sector_code is not None and isinstance(sub_sector_desc, str):
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
            year_row["public_total_transaction_value"] = _to_float_or_none(row.get("total_transaction_value")) or 0.0
        elif target_bucket == "private":
            year_row["private_deal_count"] = deps._to_int(row.get("deal_count"))
            year_row["private_total_transaction_value"] = _to_float_or_none(row.get("total_transaction_value")) or 0.0

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
            year_row["public_p25_transaction_value"] = _to_float_or_none(row.get("p25_transaction_value"))
            year_row["public_median_transaction_value"] = _to_float_or_none(row.get("median_transaction_value"))
            year_row["public_p75_transaction_value"] = _to_float_or_none(row.get("p75_transaction_value"))
        elif target_bucket == "private":
            year_row["private_deal_count"] = deps._to_int(row.get("deal_count"))
            year_row["private_p25_transaction_value"] = _to_float_or_none(row.get("p25_transaction_value"))
            year_row["private_median_transaction_value"] = _to_float_or_none(row.get("median_transaction_value"))
            year_row["private_p75_transaction_value"] = _to_float_or_none(row.get("p75_transaction_value"))

    buyer_matrix_lookup = {
        (
            str(row.get("target_bucket") or ""),
            str(row.get("buyer_bucket") or ""),
        ): row
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
                        _to_float_or_none(row.get("median_transaction_value")) if row else None
                    ),
                }
            )

    return {
        "ownership": {
            "mix_by_year": [ownership_mix_by_year[year] for year in sorted(ownership_mix_by_year.keys())],
            "deal_size_by_year": [
                ownership_deal_size_by_year[year] for year in sorted(ownership_deal_size_by_year.keys())
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
                    "total_transaction_value": _to_float_or_none(row.get("total_transaction_value")) or 0.0,
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
                    "total_transaction_value": _to_float_or_none(row.get("total_transaction_value")) or 0.0,
                }
                for row in industry_pairing_rows
            ],
        },
        "catalogs": {
            "naics": _naics_payload(reference_data_deps),
        },
    }


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

    page_size = _normalized_page_size(parsed_args["page_size"])
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

    if after_agreement_uuid:
        q = q.filter(agreements.agreement_uuid > after_agreement_uuid)

    rows = cast(list[object], q.order_by(asc(agreements.agreement_uuid)).limit(page_size + 1).all())
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


def _search_agreements(
    deps: AgreementsDeps,
    *,
    principal: McpPrincipal,
    payload: dict[str, object],
) -> McpToolResult:
    _require_scope(principal, "agreements:search")
    parsed_args = _validate_payload(McpSearchAgreementsArgsSchema(), payload)
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
    db = deps.db
    year_expr = deps._agreement_year_expr()
    sort_map = {"year": year_expr, "target": agreements.target, "acquirer": agreements.acquirer}
    sort_column = sort_map.get(sort_by, year_expr)
    order_by = sort_column.desc() if sort_dir == "desc" else sort_column.asc()

    q = (
        db.session.query(
            agreements.agreement_uuid,
            year_expr.label("year"),
            agreements.target,
            agreements.acquirer,
            agreements.filing_date,
            agreements.url,
            agreements.verified,
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
                "agreement_uuid": row_map.get("agreement_uuid"),
                "year": row_map.get("year"),
                "target": row_map.get("target"),
                "acquirer": row_map.get("acquirer"),
                "filing_date": row_map.get("filing_date"),
                "url": row_map.get("url"),
                "verified": bool(verified_value) if verified_value is not None else False,
            }
        )

    response = {
        "results": results,
        "returned_count": len(results),
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

    total_count = deps._to_int(cast(object, q.order_by(None).count()))
    offset = (page - 1) * page_size
    rows = cast(list[object], q.offset(offset).limit(page_size).all())
    meta = deps._pagination_metadata(total_count=total_count, page=page, page_size=page_size)

    results: list[dict[str, object]] = []
    for row in rows:
        row_map = deps._row_mapping_as_dict(cast(object, row))
        results.append(
            {
                "id": row_map.get("section_uuid"),
                "agreement_uuid": row_map.get("agreement_uuid"),
                "section_uuid": row_map.get("section_uuid"),
                "standard_id": deps._parse_section_standard_ids(row_map.get("section_standard_ids")),
                "article_title": row_map.get("article_title"),
                "section_title": row_map.get("section_title"),
                "target": row_map.get("target"),
                "acquirer": row_map.get("acquirer"),
                "year": deps._year_from_filing_date_value(row_map.get("filing_date")),
                "verified": bool(row_map.get("verified")) if row_map.get("verified") is not None else False,
            }
        )

    response = {
        "agreement_uuid": agreement_uuid,
        "results": results,
        "returned_count": len(results),
        **meta,
    }
    return McpToolResult(
        text=f"Returned {len(results)} section(s) for agreement {agreement_uuid}.",
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
) -> McpToolResult:
    _require_scope(principal, "sections:search")
    response = _counsel_payload(deps)
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


def tool_definitions() -> list[dict[str, object]]:
    structured_filter_overrides = _structured_filter_properties()
    agreements_list_overrides = _structured_filter_properties(include_cursor=True, include_xml=True)
    search_agreements_overrides = {
        **structured_filter_overrides,
        "query": {
            "type": "string",
            "description": "Optional prefix search over target and acquirer names, or a 4-digit year string.",
            "examples": ["Slack", "2020"],
        },
    }
    sections_search_overrides = {
        **structured_filter_overrides,
        "metadata": _enum_array_schema(
            cast(tuple[str, ...], SECTIONS_RESULT_METADATA_FIELDS),
            description="Agreement metadata fields to include under results[].metadata.",
            examples=[["deal_type", "target_industry"]],
        ),
    }
    list_agreement_sections_overrides = {
        "sort_by": {
            "type": "string",
            "enum": list(_SECTION_LIST_SORT_FIELDS),
            "description": "Section list sort key.",
        },
    }
    return [
        {
            "name": "search_agreements",
            "description": "Discover agreements with text query and page-based results. Supports the same structured agreement filters as list_agreements, including counsel filters, but is best suited for interactive discovery rather than bulk exact retrieval.",
            "inputSchema": _schema_input_schema(
                McpSearchAgreementsArgsSchema(),
                field_overrides=search_agreements_overrides,
            ),
        },
        {
            "name": "search_sections",
            "description": "Search sections across the corpus when you are looking for clause language patterns or taxonomy matches.",
            "inputSchema": _schema_input_schema(
                SectionsArgsSchema(),
                field_overrides=sections_search_overrides,
            ),
        },
        {
            "name": "list_agreements",
            "description": "Retrieve agreements with exact structured filters and cursor pagination. Prefer this over search_agreements for bulk exact retrieval, especially when filters are already known.",
            "inputSchema": _schema_input_schema(
                AgreementsBulkArgsSchema(),
                field_overrides=agreements_list_overrides,
            ),
        },
        {
            "name": "list_agreement_sections",
            "description": "Navigate the sections inside one agreement before drilling into a specific section.",
            "inputSchema": _schema_input_schema(
                McpListAgreementSectionsArgsSchema(),
                field_overrides=list_agreement_sections_overrides,
            ),
        },
        {
            "name": "get_agreement",
            "description": "Fetch one agreement document, returning redacted XML unless full-text scope is present.",
            "inputSchema": _schema_input_schema(McpAgreementArgsSchema()),
        },
        {
            "name": "get_section",
            "description": "Fetch one section directly by section UUID when you already know the exact section to inspect.",
            "inputSchema": _schema_input_schema(McpSectionArgsSchema()),
        },
        {
            "name": "get_agreement_tax_clauses",
            "description": "Fetch extracted tax-module clauses for a specific agreement.",
            "inputSchema": _schema_input_schema(McpAgreementIdentifierSchema()),
        },
        {
            "name": "get_section_tax_clauses",
            "description": "Fetch extracted tax-module clauses for a specific section.",
            "inputSchema": _schema_input_schema(McpSectionArgsSchema()),
        },
        {
            "name": "list_filter_options",
            "description": "List valid filter values for agreement and section retrieval. Catalog groups are pluralized, while retrieval arguments are singular, for example target_counsels maps to target_counsel.",
            "inputSchema": _schema_input_schema(McpFilterOptionsArgsSchema()),
        },
        {
            "name": "get_clause_taxonomy",
            "description": "Fetch the clause taxonomy tree used for section search standard IDs.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "get_tax_clause_taxonomy",
            "description": "Fetch the tax clause taxonomy tree used for tax-clause research.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "get_counsel_catalog",
            "description": "Fetch canonical counsel names for filter selection and normalization.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "get_naics_catalog",
            "description": "Fetch NAICS sectors and subsectors for industry reasoning and filter selection.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "get_agreements_summary",
            "description": "Fetch high-level corpus counts for agreements, sections, and pages.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "get_agreement_trends",
            "description": "Fetch ownership and industry trend analytics for the agreement corpus.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    ]


def call_tool(
    name: str,
    *,
    arguments: dict[str, object],
    principal: McpPrincipal,
    sections_service_deps: SectionsServiceDeps,
    agreements_deps: AgreementsDeps,
    reference_data_deps: ReferenceDataDeps,
) -> McpToolResult:
    if name == "search_agreements":
        return _search_agreements(agreements_deps, principal=principal, payload=arguments)
    if name == "search_sections":
        return _search_sections(sections_service_deps, principal=principal, payload=arguments)
    if name == "list_agreements":
        return _list_agreements(agreements_deps, principal=principal, payload=arguments)
    if name == "list_agreement_sections":
        return _list_agreement_sections(sections_service_deps, principal=principal, payload=arguments)
    if name == "get_agreement":
        return _get_agreement(agreements_deps, principal=principal, payload=arguments)
    if name == "get_section":
        return _get_section(agreements_deps, principal=principal, payload=arguments)
    if name == "get_agreement_tax_clauses":
        return _get_agreement_tax_clauses(agreements_deps, principal=principal, payload=arguments)
    if name == "get_section_tax_clauses":
        return _get_section_tax_clauses(agreements_deps, principal=principal, payload=arguments)
    if name == "list_filter_options":
        return _list_filter_options(agreements_deps, principal=principal, payload=arguments)
    if name == "get_clause_taxonomy":
        return _get_clause_taxonomy(reference_data_deps, principal=principal)
    if name == "get_tax_clause_taxonomy":
        return _get_tax_clause_taxonomy(reference_data_deps, principal=principal)
    if name == "get_counsel_catalog":
        return _get_counsel_catalog(reference_data_deps, principal=principal)
    if name == "get_naics_catalog":
        return _get_naics_catalog(reference_data_deps, principal=principal)
    if name == "get_agreements_summary":
        return _get_agreements_summary(agreements_deps, principal=principal)
    if name == "get_agreement_trends":
        return _get_agreement_trends(
            agreements_deps,
            principal=principal,
            reference_data_deps=reference_data_deps,
        )
    raise KeyError(name)


__all__ = ["McpToolResult", "call_tool", "tool_definitions"]
