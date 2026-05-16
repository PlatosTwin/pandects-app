from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from marshmallow import Schema, fields as ma_fields, validate

from backend.mcp.tools.constants import (
    _STRUCTURED_FILTER_ARRAY_FIELDS,
    _TRANSACTION_PRICE_BUCKET_OPTIONS,
)

_FieldOverrides = Mapping[str, Mapping[str, object]]


def _merge_schema_instances(*schemas: Schema) -> Schema:
    merged_fields: dict[str, ma_fields.Field[Any]] = {}
    for schema in schemas:
        for field_name, field in schema.fields.items():
            merged_fields[field_name] = field
    merged_type = Schema.from_dict(merged_fields, name="MergedSchema")
    return cast(Schema, merged_type())


def _schema_from_fields(name: str, field_map: Mapping[str, ma_fields.Field[Any]]) -> Schema:
    schema_type = Schema.from_dict(dict(field_map), name=name)
    return cast(Schema, schema_type())


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


def _range_validator_bounds(field: ma_fields.Field[Any]) -> tuple[int | None, int | None]:
    for v in getattr(field, "validators", []):
        if isinstance(v, validate.Range):
            mn = v.min
            mx = v.max
            return (int(mn) if mn is not None else None), (int(mx) if mx is not None else None)
    return None, None


def _field_json_schema(field: ma_fields.Field[Any]) -> dict[str, object]:
    schema: dict[str, object]
    if isinstance(field, ma_fields.List):
        item_schema = _field_json_schema(field.inner)
        item_schema.pop("description", None)
        item_schema.pop("example", None)
        item_schema.pop("examples", None)
        schema = {"type": "array", "items": item_schema}
    elif isinstance(field, ma_fields.Int):
        schema = {"type": "integer"}
        mn, mx = _range_validator_bounds(field)
        if mn is not None:
            schema["minimum"] = mn
        if mx is not None:
            schema["maximum"] = mx
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
    field_overrides: _FieldOverrides | None = None,
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


def _object_schema(
    properties: dict[str, object],
    *,
    required: list[str] | None = None,
    additional_properties: bool = True,
) -> dict[str, object]:
    schema: dict[str, object] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": additional_properties,
    }
    if required:
        schema["required"] = required
    return schema


def _array_of(item_schema: dict[str, object]) -> dict[str, object]:
    return {"type": "array", "items": item_schema}


def _pagination_response_properties() -> dict[str, object]:
    return {
        "page": {"type": "integer"},
        "page_size": {"type": "integer"},
        "total_count": {"type": "integer"},
        "total_count_is_approximate": {"type": "boolean"},
        "total_pages": {"type": "integer"},
        "has_next": {"type": "boolean"},
        "has_prev": {"type": "boolean"},
        "next_num": {"type": ["integer", "null"]},
        "prev_num": {"type": ["integer", "null"]},
    }


def _filter_option_metadata() -> dict[str, dict[str, object]]:
    return {
        "targets": {"retrieval_parameter": "target", "applies_to": ["agreements", "sections"], "value_kind": "exact_string", "recommended_tools": ["search_agreements", "list_agreements", "search_sections"], "examples": [["Target A"]]},
        "acquirers": {"retrieval_parameter": "acquirer", "applies_to": ["agreements", "sections"], "value_kind": "exact_string", "recommended_tools": ["search_agreements", "list_agreements", "search_sections"], "examples": [["Acquirer A"]]},
        "transaction_price_totals": {
            "retrieval_parameter": "transaction_price_total",
            "applies_to": ["agreements", "sections"],
            "value_kind": "bucket",
            "allowed_values": list(_TRANSACTION_PRICE_BUCKET_OPTIONS),
            "recommended_tools": ["search_agreements", "list_agreements", "search_sections"],
            "examples": [["100M - 250M"]],
        },
        "transaction_price_stocks": {
            "retrieval_parameter": "transaction_price_stock",
            "applies_to": ["agreements", "sections"],
            "value_kind": "bucket",
            "allowed_values": list(_TRANSACTION_PRICE_BUCKET_OPTIONS),
            "recommended_tools": ["search_agreements", "list_agreements", "search_sections"],
        },
        "transaction_price_cashes": {
            "retrieval_parameter": "transaction_price_cash",
            "applies_to": ["agreements", "sections"],
            "value_kind": "bucket",
            "allowed_values": list(_TRANSACTION_PRICE_BUCKET_OPTIONS),
            "recommended_tools": ["search_agreements", "list_agreements", "search_sections"],
        },
        "transaction_price_assets": {
            "retrieval_parameter": "transaction_price_assets",
            "applies_to": ["agreements", "sections"],
            "value_kind": "bucket",
            "allowed_values": list(_TRANSACTION_PRICE_BUCKET_OPTIONS),
            "recommended_tools": ["search_agreements", "list_agreements", "search_sections"],
        },
        "transaction_considerations": {"retrieval_parameter": "transaction_consideration", "applies_to": ["agreements", "sections"], "value_kind": "exact_string", "recommended_tools": ["search_agreements", "list_agreements", "search_sections"]},
        "target_types": {"retrieval_parameter": "target_type", "applies_to": ["agreements", "sections"], "value_kind": "exact_string", "recommended_tools": ["search_agreements", "list_agreements", "search_sections"]},
        "acquirer_types": {"retrieval_parameter": "acquirer_type", "applies_to": ["agreements", "sections"], "value_kind": "exact_string", "recommended_tools": ["search_agreements", "list_agreements", "search_sections"]},
        "target_counsels": {"retrieval_parameter": "target_counsel", "applies_to": ["agreements", "sections"], "value_kind": "exact_string", "recommended_tools": ["search_agreements", "list_agreements", "search_sections"], "examples": [["Wachtell, Lipton, Rosen & Katz"]]},
        "acquirer_counsels": {"retrieval_parameter": "acquirer_counsel", "applies_to": ["agreements", "sections"], "value_kind": "exact_string", "recommended_tools": ["search_agreements", "list_agreements", "search_sections"]},
        "target_industries": {"retrieval_parameter": "target_industry", "applies_to": ["agreements", "sections"], "value_kind": "exact_string", "recommended_tools": ["search_agreements", "list_agreements", "search_sections"]},
        "acquirer_industries": {"retrieval_parameter": "acquirer_industry", "applies_to": ["agreements", "sections"], "value_kind": "exact_string", "recommended_tools": ["search_agreements", "list_agreements", "search_sections"]},
        "deal_statuses": {"retrieval_parameter": "deal_status", "applies_to": ["agreements", "sections"], "value_kind": "exact_string", "recommended_tools": ["search_agreements", "list_agreements", "search_sections"]},
        "attitudes": {"retrieval_parameter": "attitude", "applies_to": ["agreements", "sections"], "value_kind": "exact_string", "recommended_tools": ["search_agreements", "list_agreements", "search_sections"]},
        "deal_types": {"retrieval_parameter": "deal_type", "applies_to": ["agreements", "sections"], "value_kind": "exact_string", "recommended_tools": ["search_agreements", "list_agreements", "search_sections"]},
        "purposes": {"retrieval_parameter": "purpose", "applies_to": ["agreements", "sections"], "value_kind": "exact_string", "recommended_tools": ["search_agreements", "list_agreements", "search_sections"]},
        "target_pes": {
            "retrieval_parameter": "target_pe",
            "applies_to": ["agreements", "sections"],
            "value_kind": "string_boolean",
            "allowed_values": ["true", "false"],
            "recommended_tools": ["search_agreements", "list_agreements", "search_sections"],
        },
        "acquirer_pes": {
            "retrieval_parameter": "acquirer_pe",
            "applies_to": ["agreements", "sections"],
            "value_kind": "string_boolean",
            "allowed_values": ["true", "false"],
            "recommended_tools": ["search_agreements", "list_agreements", "search_sections"],
        },
    }


def _structured_filter_properties(*, include_cursor: bool = False, include_xml: bool = False) -> dict[str, dict[str, object]]:
    properties: dict[str, dict[str, object]] = {}
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
