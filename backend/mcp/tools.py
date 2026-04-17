from __future__ import annotations

import re
from collections.abc import Mapping
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from difflib import SequenceMatcher
from decimal import Decimal
from functools import lru_cache
from html import unescape
from typing import Any, Callable, cast

from flask import abort
from marshmallow import Schema, ValidationError, fields as ma_fields, validate
from sqlalchemy import and_, asc, desc, func, or_, text

from backend.auth.mcp_runtime import McpPrincipal
from backend.filtering import build_canonical_counsel_agreement_uuid_subquery, build_transaction_price_bucket_filter
from backend.mcp.metrics import get_mcp_metrics_registry
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
    AgreementsIndexArgsSchema,
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
_FIELD_REPRESENTATION_VALUES = (
    "first_class_agreement_field",
    "first_class_section_field",
    "taxonomy_assignment",
    "derived_from_text",
    "not_represented",
)
_COUNT_MODE_VALUES = ("auto", "exact")
_COUNT_METHOD_VALUES = ("query_count", "table_estimate", "filtered_lower_bound")
_COUNT_RELIABILITY_VALUES = ("high", "medium", "low")
_TAXONOMY_MATCH_MODE_VALUES = ("exact_node", "expanded_descendants")
_CLAUSE_FIT_VALUES = ("canonical", "proxy", "broad_match")
_CLAUSE_CONFIDENCE_VALUES = ("high", "medium", "low")

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


class McpAgreementArgsSchema(AgreementArgsSchema):
    agreement_uuid = ma_fields.Str(required=True)


class McpAgreementIdentifierSchema(Schema):
    agreement_uuid = ma_fields.Str(required=True)


class McpSectionArgsSchema(Schema):
    section_uuid = ma_fields.Str(required=True)


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


class McpOutputValidationError(ValueError):
    def __init__(self, messages: dict[str, object]):
        super().__init__("Tool result did not match the advertised output schema.")
        self.messages = messages


@dataclass(frozen=True)
class McpToolSpec:
    name: str
    description: str
    input_schema: dict[str, object]
    output_schema: dict[str, object]
    examples: tuple[dict[str, object], ...]
    response_examples: tuple[dict[str, object], ...]
    scopes: tuple[str, ...]
    selection_hint: str
    negative_guidance: tuple[str, ...]
    pagination: str
    access_behavior: str
    redaction_behavior: str
    fulltext_scope: str | None
    handler: Callable[..., McpToolResult]


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


def _matches_schema_type(expected_type: object, value: object) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return True


def _validate_output_against_schema(
    schema: dict[str, object],
    value: object,
    *,
    path: str = "structuredContent",
) -> dict[str, object]:
    errors: dict[str, object] = {}
    expected_type = schema.get("type")
    if isinstance(expected_type, list):
        if not any(_matches_schema_type(type_name, value) for type_name in expected_type):
            errors[path] = f"Expected one of {expected_type}, got {type(value).__name__}."
            return errors
    elif expected_type is not None and not _matches_schema_type(expected_type, value):
        errors[path] = f"Expected {expected_type}, got {type(value).__name__}."
        return errors

    if isinstance(value, dict):
        properties = cast(dict[str, dict[str, object]], schema.get("properties", {}))
        required = cast(list[str], schema.get("required", []))
        for field_name in required:
            if field_name not in value:
                errors[f"{path}.{field_name}"] = "Missing required field."
        for field_name, field_schema in properties.items():
            if field_name in value:
                errors.update(
                    _validate_output_against_schema(
                        field_schema,
                        value[field_name],
                        path=f"{path}.{field_name}",
                    )
                )
        additional_properties = schema.get("additionalProperties", True)
        extra_keys = [field_name for field_name in value.keys() if field_name not in properties]
        if additional_properties is False:
            for field_name in extra_keys:
                errors[f"{path}.{field_name}"] = "Unexpected field."
        elif isinstance(additional_properties, dict):
            for field_name in extra_keys:
                errors.update(
                    _validate_output_against_schema(
                        additional_properties,
                        value[field_name],
                        path=f"{path}.{field_name}",
                    )
                )
        return errors

    if isinstance(value, list):
        item_schema = cast(dict[str, object], schema.get("items", {}))
        for index, item in enumerate(value):
            errors.update(_validate_output_against_schema(item_schema, item, path=f"{path}[{index}]"))
        return errors

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        errors[path] = f"Unexpected enum value: {value!r}."
    return errors


def _normalized_page(page: int) -> int:
    return page if page >= 1 else 1


def _normalized_page_size(page_size: int) -> int:
    return page_size if 1 <= page_size <= 100 else 25


def _json_compatible_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _count_metadata_payload(
    *,
    mode: str,
    method: str,
    planning_reliability: str,
    exact_count_requested: bool,
) -> dict[str, object]:
    return {
        "mode": mode,
        "method": method,
        "planning_reliability": planning_reliability,
        "exact_count_requested": exact_count_requested,
    }


def _interpretation_payload(
    *,
    applied_filters: list[dict[str, str]],
    taxonomy_filters: list[dict[str, str]] | None = None,
    heuristics_used: list[str] | None = None,
    notes: list[str] | None = None,
) -> dict[str, object]:
    return {
        "applied_filters": applied_filters,
        "taxonomy_filters": taxonomy_filters or [],
        "heuristics_used": heuristics_used or [],
        "notes": notes or [],
    }


def _agreement_filter_interpretation(parsed_args: AgreementsBulkArgsPayload, *, query: str) -> dict[str, object]:
    applied_filters: list[dict[str, str]] = []
    for field_name in _STRUCTURED_FILTER_ARRAY_FIELDS:
        values = cast(list[object], parsed_args[field_name])
        if values:
            applied_filters.append(
                {
                    "field": field_name,
                    "representation": "first_class_agreement_field",
                    "match_kind": "exact_metadata_filter",
                }
            )
    for field_name in ("agreement_uuid", "section_uuid"):
        value = parsed_args.get(field_name)
        if isinstance(value, str) and value.strip():
            applied_filters.append(
                {
                    "field": field_name,
                    "representation": "first_class_agreement_field" if field_name == "agreement_uuid" else "first_class_section_field",
                    "match_kind": "exact_metadata_filter",
                }
            )
    heuristics_used: list[str] = []
    notes: list[str] = []
    if query:
        if query.isdigit():
            applied_filters.append(
                {
                    "field": "query",
                    "representation": "first_class_agreement_field",
                    "match_kind": "year_prefix_query",
                }
            )
            notes.append("The free-text query was interpreted as a 4-digit agreement year.")
        else:
            heuristics_used.append("prefix_name_match")
            notes.append("The free-text query uses prefix matching on target and acquirer names.")
    return _interpretation_payload(
        applied_filters=applied_filters,
        heuristics_used=heuristics_used,
        notes=notes,
    )


@dataclass(frozen=True)
class _TaxonomyEntry:
    standard_id: str
    label: str
    path: tuple[str, ...]


_WHITESPACE_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")
_CONCEPT_SYNONYM_HINTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("material adverse", "mae", "material adverse effect", "material adverse change"),
        (
            "mae",
            "material adverse effect",
            "material adverse change",
            "mae carveouts",
            "disproportionate effects",
            "carveout provisions",
        ),
    ),
    (
        ("fiduciary", "recommendation", "superior proposal", "intervening event"),
        (
            "fiduciary out",
            "change of recommendation",
            "superior proposal",
            "intervening event",
            "matching rights",
        ),
    ),
    (
        ("no shop", "no-shop", "solicit", "solicitation"),
        (
            "no shop",
            "no-shop",
            "non solicitation",
            "non-solicitation",
            "matching rights",
        ),
    ),
    (
        ("specific performance", "equitable relief"),
        ("specific performance", "equitable relief"),
    ),
    (
        ("antitrust", "regulatory", "hell or high water", "efforts"),
        ("antitrust efforts", "regulatory efforts", "hell or high water", "reasonable best efforts"),
    ),
)


def _normalized_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    lowered = value.lower().replace("&", " and ")
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def _normalized_tokens(value: object) -> set[str]:
    normalized = _normalized_text(value)
    if normalized == "":
        return set()
    return {token for token in normalized.split(" ") if token}


def _taxonomy_entries(
    *,
    l1_model: object,
    l2_model: object,
    l3_model: object,
    deps: ReferenceDataDeps,
) -> list[_TaxonomyEntry]:
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

    l1_by_id = {
        standard_id: label
        for standard_id, label in l1_rows
        if isinstance(standard_id, str) and isinstance(label, str)
    }
    l2_by_id = {
        standard_id: (label, parent_id)
        for standard_id, label, parent_id in l2_rows
        if isinstance(standard_id, str) and isinstance(label, str) and isinstance(parent_id, str)
    }

    entries: list[_TaxonomyEntry] = []
    for standard_id, label in sorted(l1_by_id.items(), key=lambda item: item[1]):
        entries.append(_TaxonomyEntry(standard_id=standard_id, label=label, path=(label,)))
    for standard_id, (label, parent_id) in sorted(l2_by_id.items(), key=lambda item: item[1][0]):
        parent_label = l1_by_id.get(parent_id)
        if parent_label is None:
            continue
        entries.append(_TaxonomyEntry(standard_id=standard_id, label=label, path=(parent_label, label)))
    for standard_id, label, parent_id in l3_rows:
        if not (isinstance(standard_id, str) and isinstance(label, str) and isinstance(parent_id, str)):
            continue
        parent_row = l2_by_id.get(parent_id)
        if parent_row is None:
            continue
        parent_label, grandparent_id = parent_row
        grandparent_label = l1_by_id.get(grandparent_id)
        if grandparent_label is None:
            continue
        entries.append(
            _TaxonomyEntry(
                standard_id=standard_id,
                label=label,
                path=(grandparent_label, parent_label, label),
            )
        )
    return sorted(entries, key=lambda entry: (len(entry.path), entry.path))


def _taxonomy_alias_terms(entry: _TaxonomyEntry) -> tuple[str, ...]:
    path_text = " ".join(entry.path)
    normalized_path = _normalized_text(path_text)
    aliases: set[str] = set()
    for trigger_terms, alias_terms in _CONCEPT_SYNONYM_HINTS:
        if any(trigger in normalized_path for trigger in trigger_terms):
            aliases.update(alias_terms)
    return tuple(sorted(aliases))


def _score_taxonomy_entry(entry: _TaxonomyEntry, concept: str) -> tuple[float, list[str]]:
    concept_normalized = _normalized_text(concept)
    concept_tokens = _normalized_tokens(concept)
    if concept_normalized == "":
        return 0.0, []

    search_terms = [entry.label, " > ".join(entry.path), *_taxonomy_alias_terms(entry)]
    matched_terms: list[str] = []
    best_ratio = 0.0
    token_overlap_bonus = 0.0

    for term in search_terms:
        term_normalized = _normalized_text(term)
        if term_normalized == "":
            continue
        ratio = SequenceMatcher(None, concept_normalized, term_normalized).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
        if concept_normalized in term_normalized or term_normalized in concept_normalized:
            matched_terms.append(term)
            best_ratio = max(best_ratio, 0.95)
        overlap = concept_tokens & _normalized_tokens(term)
        if overlap:
            token_overlap_bonus = max(token_overlap_bonus, len(overlap) / max(len(concept_tokens), 1))
            if term not in matched_terms:
                matched_terms.append(term)

    score = round((best_ratio * 0.7) + (token_overlap_bonus * 0.3), 4)
    return score, matched_terms[:4]


def _taxonomy_fit(score: float, *, matched_terms: list[str], entry: _TaxonomyEntry, concept: str) -> str:
    concept_normalized = _normalized_text(concept)
    entry_label = _normalized_text(entry.label)
    if concept_normalized and concept_normalized in entry_label:
        return "canonical"
    if matched_terms and score >= 0.75:
        return "proxy"
    return "broad_match"


def _taxonomy_confidence(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def _taxonomy_scope_note(*, fit: str) -> str:
    if fit == "canonical":
        return "This taxonomy node is a close canonical match for the concept."
    if fit == "proxy":
        return "This taxonomy node is a reasonable proxy and may be broader or narrower than the requested concept."
    return "This taxonomy node is a broad semantic match and should be verified before treating it as canonical."


def _ranked_taxonomy_matches(
    *,
    deps: ReferenceDataDeps,
    concept: str,
    taxonomy: str,
    top_k: int,
) -> list[dict[str, object]]:
    if taxonomy == "tax_clauses":
        entries = _taxonomy_entries(
            l1_model=deps.TaxClauseTaxonomyL1,
            l2_model=deps.TaxClauseTaxonomyL2,
            l3_model=deps.TaxClauseTaxonomyL3,
            deps=deps,
        )
    else:
        entries = _taxonomy_entries(
            l1_model=deps.TaxonomyL1,
            l2_model=deps.TaxonomyL2,
            l3_model=deps.TaxonomyL3,
            deps=deps,
        )

    scored: list[tuple[float, _TaxonomyEntry, list[str]]] = []
    for entry in entries:
        score, matched_terms = _score_taxonomy_entry(entry, concept)
        if score > 0:
            scored.append((score, entry, matched_terms))
    scored.sort(key=lambda item: (-item[0], len(item[1].path), item[1].path))

    results: list[dict[str, object]] = []
    for score, entry, matched_terms in scored[:top_k]:
        fit = _taxonomy_fit(score, matched_terms=matched_terms, entry=entry, concept=concept)
        results.append(
            {
                "standard_id": entry.standard_id,
                "label": entry.label,
                "path": list(entry.path),
                "score": score,
                "matched_terms": matched_terms,
                "fit": fit,
                "scope_note": _taxonomy_scope_note(fit=fit),
                "confidence": _taxonomy_confidence(score),
                "reason": "Matched concept tokens and clause-family synonyms."
                if matched_terms
                else "Closest taxonomy path by label similarity.",
            }
        )
    return results


def _extract_text_from_xml(xml: object) -> str:
    if not isinstance(xml, str):
        return ""
    text_content = unescape(_TAG_RE.sub(" ", xml))
    return _WHITESPACE_RE.sub(" ", text_content).strip()


def _focused_snippet(text_content: str, *, focus_terms: list[str], max_chars: int) -> tuple[str, list[str]]:
    cleaned_text = _WHITESPACE_RE.sub(" ", text_content).strip()
    if cleaned_text == "":
        return "", []

    lowered_text = cleaned_text.lower()
    matched_terms: list[str] = []
    best_index: int | None = None
    for term in focus_terms:
        normalized_term = term.strip().lower()
        if normalized_term == "":
            continue
        index = lowered_text.find(normalized_term)
        if index >= 0:
            matched_terms.append(term)
            if best_index is None or index < best_index:
                best_index = index

    if best_index is None:
        snippet = cleaned_text[:max_chars].rstrip()
        if len(cleaned_text) > len(snippet):
            snippet = snippet.rstrip(" .,;:") + "..."
        return snippet, []

    start = max(best_index - (max_chars // 3), 0)
    end = min(start + max_chars, len(cleaned_text))
    if end - start < max_chars and start > 0:
        start = max(end - max_chars, 0)
    snippet = cleaned_text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(cleaned_text):
        snippet = snippet.rstrip(" .,;:") + "..."
    return snippet, matched_terms


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
        _merge_schema_instances(AgreementsIndexArgsSchema(), AgreementsBulkArgsSchema()),
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
                "agreement_uuid": _json_compatible_value(row_map.get("agreement_uuid")),
                "year": _json_compatible_value(row_map.get("year")),
                "target": _json_compatible_value(row_map.get("target")),
                "acquirer": _json_compatible_value(row_map.get("acquirer")),
                "filing_date": _json_compatible_value(row_map.get("filing_date")),
                "url": _json_compatible_value(row_map.get("url")),
                "verified": bool(verified_value) if verified_value is not None else False,
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
    }
    return McpToolResult(
        text=f"Returned a focused snippet for section {section_uuid}.",
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


def _empty_schema() -> dict[str, object]:
    return {"type": "object", "properties": {}, "additionalProperties": False}


def _access_schema() -> dict[str, object]:
    return _object_schema(
        {
            "tier": {"type": "string"},
            "message": {"type": ["string", "null"]},
        },
        required=["tier", "message"],
        additional_properties=False,
    )


def _agreement_search_result_schema() -> dict[str, object]:
    return _object_schema(
        {
            "agreement_uuid": {"type": "string"},
            "year": {"type": ["integer", "null"]},
            "target": {"type": ["string", "null"]},
            "acquirer": {"type": ["string", "null"]},
            "filing_date": {"type": ["string", "null"]},
            "url": {"type": ["string", "null"]},
            "verified": {"type": "boolean"},
        },
        required=["agreement_uuid", "verified"],
    )


def _agreement_list_result_schema(*, include_xml: bool = False) -> dict[str, object]:
    properties: dict[str, object] = {
        "agreement_uuid": {"type": "string"},
        "year": {"type": ["integer", "null"]},
        "target": {"type": ["string", "null"]},
        "acquirer": {"type": ["string", "null"]},
        "filing_date": {"type": ["string", "null"]},
        "prob_filing": {"type": ["string", "null"]},
        "filing_company_name": {"type": ["string", "null"]},
        "filing_company_cik": {"type": ["string", "null"]},
        "form_type": {"type": ["string", "null"]},
        "exhibit_type": {"type": ["string", "null"]},
        "transaction_price_total": {"type": ["number", "null"]},
        "transaction_price_stock": {"type": ["number", "null"]},
        "transaction_price_cash": {"type": ["number", "null"]},
        "transaction_price_assets": {"type": ["number", "null"]},
        "transaction_consideration": {"type": ["string", "null"]},
        "target_type": {"type": ["string", "null"]},
        "acquirer_type": {"type": ["string", "null"]},
        "target_industry": {"type": ["string", "null"]},
        "acquirer_industry": {"type": ["string", "null"]},
        "announce_date": {"type": ["string", "null"]},
        "close_date": {"type": ["string", "null"]},
        "deal_status": {"type": ["string", "null"]},
        "attitude": {"type": ["string", "null"]},
        "deal_type": {"type": ["string", "null"]},
        "purpose": {"type": ["string", "null"]},
        "target_pe": {"type": ["integer", "null"]},
        "acquirer_pe": {"type": ["integer", "null"]},
        "url": {"type": ["string", "null"]},
    }
    if include_xml:
        properties["xml"] = {"type": ["string", "null"]}
    return _object_schema(properties, required=["agreement_uuid"])


def _section_result_schema() -> dict[str, object]:
    return _object_schema(
        {
            "id": {"type": "string"},
            "agreement_uuid": {"type": ["string", "null"]},
            "section_uuid": {"type": "string"},
            "standard_id": _array_of({"type": "string"}),
            "xml": {"type": ["string", "null"]},
            "article_title": {"type": ["string", "null"]},
            "section_title": {"type": ["string", "null"]},
            "acquirer": {"type": ["string", "null"]},
            "target": {"type": ["string", "null"]},
            "year": {"type": ["integer", "null"]},
            "verified": {"type": "boolean"},
            "metadata": {"type": "object", "additionalProperties": True},
        },
        required=["id", "section_uuid", "standard_id", "verified"],
    )


def _list_section_result_schema() -> dict[str, object]:
    return _object_schema(
        {
            "id": {"type": "string"},
            "agreement_uuid": {"type": ["string", "null"]},
            "section_uuid": {"type": "string"},
            "standard_id": _array_of({"type": "string"}),
            "article_title": {"type": ["string", "null"]},
            "section_title": {"type": ["string", "null"]},
            "target": {"type": ["string", "null"]},
            "acquirer": {"type": ["string", "null"]},
            "year": {"type": ["integer", "null"]},
            "verified": {"type": "boolean"},
        },
        required=["id", "section_uuid", "standard_id", "verified"],
    )


def _count_metadata_schema() -> dict[str, object]:
    return _object_schema(
        {
            "mode": {"type": "string", "enum": ["exact", "estimated"]},
            "method": {"type": "string", "enum": list(_COUNT_METHOD_VALUES)},
            "planning_reliability": {"type": "string", "enum": list(_COUNT_RELIABILITY_VALUES)},
            "exact_count_requested": {"type": "boolean"},
        },
        required=["mode", "method", "planning_reliability", "exact_count_requested"],
        additional_properties=False,
    )


def _interpretation_filter_schema() -> dict[str, object]:
    return _object_schema(
        {
            "field": {"type": "string"},
            "representation": {"type": "string", "enum": list(_FIELD_REPRESENTATION_VALUES)},
            "match_kind": {"type": "string"},
        },
        required=["field", "representation", "match_kind"],
        additional_properties=False,
    )


def _taxonomy_filter_schema() -> dict[str, object]:
    return _object_schema(
        {
            "standard_id": {"type": "string"},
            "match_mode": {"type": "string", "enum": list(_TAXONOMY_MATCH_MODE_VALUES)},
        },
        required=["standard_id", "match_mode"],
        additional_properties=False,
    )


def _interpretation_schema() -> dict[str, object]:
    return _object_schema(
        {
            "applied_filters": _array_of(_interpretation_filter_schema()),
            "taxonomy_filters": _array_of(_taxonomy_filter_schema()),
            "heuristics_used": _array_of({"type": "string"}),
            "notes": _array_of({"type": "string"}),
        },
        required=["applied_filters", "taxonomy_filters", "heuristics_used", "notes"],
        additional_properties=False,
    )


def _search_agreements_output_schema() -> dict[str, object]:
    properties = _pagination_response_properties()
    properties.update(
        {
            "returned_count": {"type": "integer"},
            "results": _array_of(_agreement_search_result_schema()),
            "count_metadata": _count_metadata_schema(),
            "interpretation": _interpretation_schema(),
        }
    )
    return _object_schema(
        properties,
        required=[
            "results",
            "returned_count",
            "count_metadata",
            "interpretation",
            "page",
            "page_size",
            "total_count",
            "total_pages",
            "has_next",
            "has_prev",
            "total_count_is_approximate",
        ],
    )


def _search_sections_output_schema() -> dict[str, object]:
    properties = _pagination_response_properties()
    properties.update(
        {
            "results": _array_of(_section_result_schema()),
            "access": _access_schema(),
            "count_metadata": _count_metadata_schema(),
            "interpretation": _interpretation_schema(),
        }
    )
    return _object_schema(
        properties,
        required=[
            "results",
            "access",
            "count_metadata",
            "interpretation",
            "page",
            "page_size",
            "total_count",
            "total_pages",
            "has_next",
            "has_prev",
            "total_count_is_approximate",
        ],
    )


def _list_agreements_output_schema() -> dict[str, object]:
    return _object_schema(
        {
            "results": _array_of(_agreement_list_result_schema()),
            "access": _access_schema(),
            "page_size": {"type": "integer"},
            "returned_count": {"type": "integer"},
            "has_next": {"type": "boolean"},
            "next_cursor": {"type": ["string", "null"]},
        },
        required=["results", "access", "page_size", "returned_count", "has_next", "next_cursor"],
    )


def _list_agreement_sections_output_schema() -> dict[str, object]:
    properties = _pagination_response_properties()
    properties.update(
        {
            "agreement_uuid": {"type": "string"},
            "results": _array_of(_list_section_result_schema()),
            "returned_count": {"type": "integer"},
        }
    )
    return _object_schema(
        properties,
        required=["agreement_uuid", "results", "returned_count", "page", "page_size", "total_count", "total_pages", "has_next", "has_prev", "total_count_is_approximate"],
    )


def _get_agreement_output_schema() -> dict[str, object]:
    return _object_schema(
        {
            "year": {"type": ["integer", "null"]},
            "target": {"type": ["string", "null"]},
            "acquirer": {"type": ["string", "null"]},
            "filing_date": {"type": ["string", "null"]},
            "prob_filing": {"type": ["string", "null"]},
            "filing_company_name": {"type": ["string", "null"]},
            "filing_company_cik": {"type": ["string", "null"]},
            "form_type": {"type": ["string", "null"]},
            "exhibit_type": {"type": ["string", "null"]},
            "transaction_price_total": {"type": ["number", "null"]},
            "transaction_price_stock": {"type": ["number", "null"]},
            "transaction_price_cash": {"type": ["number", "null"]},
            "transaction_price_assets": {"type": ["number", "null"]},
            "transaction_consideration": {"type": ["string", "null"]},
            "target_type": {"type": ["string", "null"]},
            "acquirer_type": {"type": ["string", "null"]},
            "target_industry": {"type": ["string", "null"]},
            "acquirer_industry": {"type": ["string", "null"]},
            "announce_date": {"type": ["string", "null"]},
            "close_date": {"type": ["string", "null"]},
            "deal_status": {"type": ["string", "null"]},
            "attitude": {"type": ["string", "null"]},
            "deal_type": {"type": ["string", "null"]},
            "purpose": {"type": ["string", "null"]},
            "target_pe": {"type": ["integer", "null"]},
            "acquirer_pe": {"type": ["integer", "null"]},
            "url": {"type": ["string", "null"]},
            "xml": {"type": "string"},
            "is_redacted": {"type": "boolean"},
        },
        required=["xml"],
    )


def _get_section_output_schema() -> dict[str, object]:
    return _object_schema(
        {
            "agreement_uuid": {"type": ["string", "null"]},
            "section_uuid": {"type": "string"},
            "standard_id": _array_of({"type": "string"}),
            "xml": {"type": ["string", "null"]},
            "article_title": {"type": ["string", "null"]},
            "section_title": {"type": ["string", "null"]},
        },
        required=["section_uuid", "standard_id"],
    )


def _taxonomy_match_schema() -> dict[str, object]:
    return _object_schema(
        {
            "standard_id": {"type": "string"},
            "label": {"type": "string"},
            "path": _array_of({"type": "string"}),
            "score": {"type": "number"},
            "matched_terms": _array_of({"type": "string"}),
            "fit": {"type": "string", "enum": list(_CLAUSE_FIT_VALUES)},
            "scope_note": {"type": "string"},
            "confidence": {"type": "string", "enum": list(_CLAUSE_CONFIDENCE_VALUES)},
            "reason": {"type": "string"},
        },
        required=[
            "standard_id",
            "label",
            "path",
            "score",
            "matched_terms",
            "fit",
            "scope_note",
            "confidence",
            "reason",
        ],
        additional_properties=False,
    )


def _suggest_clause_families_output_schema() -> dict[str, object]:
    return _object_schema(
        {
            "concept": {"type": "string"},
            "taxonomy": {"type": "string", "enum": ["clauses", "tax_clauses"]},
            "matches": _array_of(_taxonomy_match_schema()),
            "returned_count": {"type": "integer"},
        },
        required=["concept", "taxonomy", "matches", "returned_count"],
        additional_properties=False,
    )


def _section_snippet_output_schema() -> dict[str, object]:
    return _object_schema(
        {
            "agreement_uuid": {"type": ["string", "null"]},
            "section_uuid": {"type": "string"},
            "standard_id": _array_of({"type": "string"}),
            "article_title": {"type": ["string", "null"]},
            "section_title": {"type": ["string", "null"]},
            "snippet": {"type": "string"},
            "matched_terms": _array_of({"type": "string"}),
            "source_length": {"type": "integer"},
        },
        required=[
            "agreement_uuid",
            "section_uuid",
            "standard_id",
            "article_title",
            "section_title",
            "snippet",
            "matched_terms",
            "source_length",
        ],
        additional_properties=False,
    )


def _metrics_output_schema() -> dict[str, object]:
    return _object_schema(
        {
            "latency_bucket_bounds_ms": _array_of({"type": "integer"}),
            "tool_calls": {
                "type": "object",
                "additionalProperties": _object_schema(
                    {
                        "calls": {"type": "integer"},
                        "errors": {"type": "integer"},
                        "avg_latency_ms": {"type": "number"},
                        "max_latency_ms": {"type": "integer"},
                        "latency_buckets": _array_of({"type": "integer"}),
                        "error_categories": {"type": "object", "additionalProperties": {"type": "integer"}},
                    },
                    required=["calls", "errors", "avg_latency_ms", "max_latency_ms", "latency_buckets", "error_categories"],
                ),
            },
            "auth_failures": {"type": "object", "additionalProperties": {"type": "integer"}},
        },
        required=["latency_bucket_bounds_ms", "tool_calls", "auth_failures"],
    )


def _tax_clause_result_schema() -> dict[str, object]:
    return _object_schema(
        {
            "clause_uuid": {"type": "string"},
            "agreement_uuid": {"type": ["string", "null"]},
            "section_uuid": {"type": ["string", "null"]},
            "article_title": {"type": ["string", "null"]},
            "section_title": {"type": ["string", "null"]},
            "anchor_label": {"type": ["string", "null"]},
            "start_char": {"type": ["integer", "null"]},
            "end_char": {"type": ["integer", "null"]},
            "clause_text": {"type": ["string", "null"]},
            "context_type": {"type": ["string", "null"]},
            "standard_ids": _array_of({"type": "string"}),
        },
        required=["clause_uuid", "standard_ids"],
        additional_properties=False,
    )


def _get_agreement_tax_clauses_output_schema() -> dict[str, object]:
    return _object_schema(
        {
            "agreement_uuid": {"type": "string"},
            "clauses": _array_of(_tax_clause_result_schema()),
            "returned_count": {"type": "integer"},
        },
        required=["agreement_uuid", "clauses", "returned_count"],
        additional_properties=False,
    )


def _get_section_tax_clauses_output_schema() -> dict[str, object]:
    return _object_schema(
        {
            "section_uuid": {"type": "string"},
            "clauses": _array_of(_tax_clause_result_schema()),
            "returned_count": {"type": "integer"},
        },
        required=["section_uuid", "clauses", "returned_count"],
        additional_properties=False,
    )


def _filter_option_metadata_schema() -> dict[str, object]:
    return _object_schema(
        {
            "retrieval_parameter": {"type": "string"},
            "applies_to": _array_of({"type": "string"}),
            "value_kind": {
                "type": "string",
                "enum": ["exact_string", "bucket", "string_boolean"],
            },
            "allowed_values": _array_of({"type": "string"}),
            "recommended_tools": _array_of({"type": "string"}),
            "examples": _array_of(_array_of({"type": "string"})),
        },
        required=["retrieval_parameter", "applies_to", "value_kind", "recommended_tools"],
        additional_properties=False,
    )


def _list_filter_options_output_schema() -> dict[str, object]:
    properties: dict[str, object] = {
        "fields": _array_of({"type": "string", "enum": list(_FILTER_OPTIONS_FIELDS)}),
        "retrieval_parameter_map": {"type": "object", "additionalProperties": {"type": "string"}},
        "field_metadata": {"type": "object", "additionalProperties": _filter_option_metadata_schema()},
    }
    for field_name in _FILTER_OPTIONS_FIELDS:
        properties[field_name] = _array_of({"type": "string"})
    return _object_schema(
        properties,
        required=["fields", "retrieval_parameter_map", "field_metadata"],
        additional_properties=False,
    )


def _taxonomy_leaf_node_schema() -> dict[str, object]:
    return _object_schema({"id": {"type": "string"}}, required=["id"], additional_properties=False)


def _taxonomy_branch_node_schema(*, child_schema: dict[str, object]) -> dict[str, object]:
    return _object_schema(
        {
            "id": {"type": "string"},
            "children": {"type": "object", "additionalProperties": child_schema},
        },
        required=["id", "children"],
        additional_properties=False,
    )


def _taxonomy_output_schema() -> dict[str, object]:
    level_three = _taxonomy_leaf_node_schema()
    level_two = _taxonomy_branch_node_schema(child_schema=level_three)
    level_one = _taxonomy_branch_node_schema(child_schema=level_two)
    return {"type": "object", "additionalProperties": level_one}


def _counsel_catalog_output_schema() -> dict[str, object]:
    return _object_schema(
        {
            "counsel": _array_of(
                _object_schema(
                    {
                        "counsel_id": {"type": "integer"},
                        "canonical_name": {"type": "string"},
                    },
                    required=["counsel_id", "canonical_name"],
                    additional_properties=False,
                )
            )
        },
        required=["counsel"],
        additional_properties=False,
    )


def _naics_catalog_output_schema() -> dict[str, object]:
    return _object_schema(
        {
            "sectors": _array_of(
                _object_schema(
                    {
                        "sector_code": {"type": "string"},
                        "sector_desc": {"type": "string"},
                        "sector_group": {"type": "string"},
                        "super_sector": {"type": "string"},
                        "sub_sectors": _array_of(
                            _object_schema(
                                {
                                    "sub_sector_code": {"type": "string"},
                                    "sub_sector_desc": {"type": "string"},
                                },
                                required=["sub_sector_code", "sub_sector_desc"],
                                additional_properties=False,
                            )
                        ),
                    },
                    required=["sector_code", "sector_desc", "sector_group", "super_sector", "sub_sectors"],
                    additional_properties=False,
                )
            )
        },
        required=["sectors"],
        additional_properties=False,
    )


def _agreements_summary_output_schema() -> dict[str, object]:
    return _object_schema(
        {
            "agreements": {"type": "integer"},
            "sections": {"type": "integer"},
            "pages": {"type": "integer"},
        },
        required=["agreements", "sections", "pages"],
        additional_properties=False,
    )


def _agreement_trends_output_schema() -> dict[str, object]:
    ownership_mix_item = _object_schema(
        {
            "year": {"type": "integer"},
            "public_deal_count": {"type": "integer"},
            "private_deal_count": {"type": "integer"},
            "public_total_transaction_value": {"type": "number"},
            "private_total_transaction_value": {"type": "number"},
        },
        required=[
            "year",
            "public_deal_count",
            "private_deal_count",
            "public_total_transaction_value",
            "private_total_transaction_value",
        ],
        additional_properties=False,
    )
    ownership_deal_size_item = _object_schema(
        {
            "year": {"type": "integer"},
            "public_deal_count": {"type": "integer"},
            "private_deal_count": {"type": "integer"},
            "public_p25_transaction_value": {"type": ["number", "null"]},
            "public_median_transaction_value": {"type": ["number", "null"]},
            "public_p75_transaction_value": {"type": ["number", "null"]},
            "private_p25_transaction_value": {"type": ["number", "null"]},
            "private_median_transaction_value": {"type": ["number", "null"]},
            "private_p75_transaction_value": {"type": ["number", "null"]},
        },
        required=[
            "year",
            "public_deal_count",
            "private_deal_count",
            "public_p25_transaction_value",
            "public_median_transaction_value",
            "public_p75_transaction_value",
            "private_p25_transaction_value",
            "private_median_transaction_value",
            "private_p75_transaction_value",
        ],
        additional_properties=False,
    )
    buyer_matrix_item = _object_schema(
        {
            "target_bucket": {"type": "string"},
            "buyer_bucket": {"type": "string"},
            "deal_count": {"type": "integer"},
            "median_transaction_value": {"type": ["number", "null"]},
        },
        required=["target_bucket", "buyer_bucket", "deal_count", "median_transaction_value"],
        additional_properties=False,
    )
    target_industry_item = _object_schema(
        {
            "year": {"type": "integer"},
            "industry": {"type": "string"},
            "deal_count": {"type": "integer"},
            "total_transaction_value": {"type": "number"},
        },
        required=["year", "industry", "deal_count", "total_transaction_value"],
        additional_properties=False,
    )
    pairing_item = _object_schema(
        {
            "target_industry": {"type": "string"},
            "acquirer_industry": {"type": "string"},
            "deal_count": {"type": "integer"},
            "total_transaction_value": {"type": "number"},
        },
        required=["target_industry", "acquirer_industry", "deal_count", "total_transaction_value"],
        additional_properties=False,
    )
    return _object_schema(
        {
            "ownership": _object_schema(
                {
                    "mix_by_year": _array_of(ownership_mix_item),
                    "deal_size_by_year": _array_of(ownership_deal_size_item),
                    "buyer_type_matrix": _array_of(buyer_matrix_item),
                },
                required=["mix_by_year", "deal_size_by_year", "buyer_type_matrix"],
                additional_properties=False,
            ),
            "industries": _object_schema(
                {
                    "target_industries_by_year": _array_of(target_industry_item),
                    "pairings": _array_of(pairing_item),
                },
                required=["target_industries_by_year", "pairings"],
                additional_properties=False,
            ),
            "catalogs": _object_schema(
                {"naics": _naics_catalog_output_schema()},
                required=["naics"],
                additional_properties=False,
            ),
        },
        required=["ownership", "industries", "catalogs"],
        additional_properties=False,
    )


def _tool_example_schema() -> dict[str, object]:
    return _object_schema(
        {
            "description": {"type": "string"},
            "arguments": {"type": "object"},
        },
        required=["description", "arguments"],
        additional_properties=False,
    )


def _tool_response_example_schema() -> dict[str, object]:
    return _object_schema(
        {
            "description": {"type": "string"},
            "content": {"type": "object"},
        },
        required=["description", "content"],
        additional_properties=False,
    )


def _tool_limits_for_pagination(pagination: str) -> dict[str, object]:
    if pagination in {"page", "cursor"}:
        return {
            "mode": pagination,
            "default_page_size": 25,
            "max_page_size": 100,
        }
    return {"mode": "none"}


def _tool_access_metadata_schema() -> dict[str, object]:
    return _object_schema(
        {
            "scope_behavior": {
                "type": "string",
                "enum": ["strict_scope_required", "partial_access_with_redaction"],
            },
            "redaction": {
                "type": "string",
                "enum": ["none", "redacted_without_fulltext_scope"],
            },
            "failure_status_code": {"type": "integer"},
            "fulltext_scope": {"type": ["string", "null"]},
        },
        required=["scope_behavior", "redaction", "failure_status_code", "fulltext_scope"],
        additional_properties=False,
    )


def _tool_capabilities_output_schema() -> dict[str, object]:
    return _object_schema(
        {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "required_scopes": _array_of({"type": "string"}),
            "pagination": {"type": "string", "enum": ["page", "cursor", "none"]},
            "selection_hint": {"type": "string"},
            "negative_guidance": _array_of({"type": "string"}),
            "examples": _array_of(_tool_example_schema()),
            "response_examples": _array_of(_tool_response_example_schema()),
            "access": _tool_access_metadata_schema(),
            "limits": _object_schema(
                {
                    "mode": {"type": "string", "enum": ["page", "cursor", "none"]},
                    "default_page_size": {"type": "integer"},
                    "max_page_size": {"type": "integer"},
                },
                required=["mode"],
            ),
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
        },
        required=[
            "name",
            "description",
            "required_scopes",
            "pagination",
            "selection_hint",
            "negative_guidance",
            "examples",
            "response_examples",
            "access",
            "limits",
            "input_schema",
            "output_schema",
        ],
        additional_properties=False,
    )


def _workflow_output_schema() -> dict[str, object]:
    return _object_schema(
        {
            "name": {"type": "string"},
            "steps": _array_of({"type": "string"}),
        },
        required=["name", "steps"],
        additional_properties=False,
    )


def _auth_help_schema() -> dict[str, object]:
    return _object_schema(
        {
            "login_required": {"type": "boolean"},
            "relogin_hint": {"type": "string"},
            "fulltext_scope": {"type": "string"},
            "invalid_or_expired_token_message": {"type": "string"},
        },
        required=[
            "login_required",
            "relogin_hint",
            "fulltext_scope",
            "invalid_or_expired_token_message",
        ],
        additional_properties=False,
    )


def _field_inventory_item_schema() -> dict[str, object]:
    return _object_schema(
        {
            "name": {"type": "string"},
            "applies_to_tools": _array_of({"type": "string"}),
            "source_table_or_surface": {"type": "string"},
            "representation": {"type": "string", "enum": list(_FIELD_REPRESENTATION_VALUES)},
        },
        required=["name", "applies_to_tools", "source_table_or_surface", "representation"],
        additional_properties=False,
    )


def _field_inventory_schema() -> dict[str, object]:
    return _object_schema(
        {
            "agreement_fields": _array_of(_field_inventory_item_schema()),
            "section_fields": _array_of(_field_inventory_item_schema()),
            "taxonomy_assignment_fields": _array_of(_field_inventory_item_schema()),
        },
        required=["agreement_fields", "section_fields", "taxonomy_assignment_fields"],
        additional_properties=False,
    )


def _concept_note_schema() -> dict[str, object]:
    return _object_schema(
        {
            "concept": {"type": "string"},
            "recommended_tools": _array_of({"type": "string"}),
            "representation": {"type": "string", "enum": list(_FIELD_REPRESENTATION_VALUES)},
            "canonical_or_proxy": {"type": "string", "enum": ["canonical", "proxy"]},
            "scope_note": {"type": "string"},
        },
        required=["concept", "recommended_tools", "representation", "canonical_or_proxy", "scope_note"],
        additional_properties=False,
    )


def _tool_limitation_schema() -> dict[str, object]:
    return _object_schema(
        {
            "tool": {"type": "string"},
            "use_when": {"type": "string"},
            "do_not_use_for": {"type": "string"},
        },
        required=["tool", "use_when", "do_not_use_for"],
        additional_properties=False,
    )


def _server_capabilities_output_schema() -> dict[str, object]:
    return _object_schema(
        {
            "server": _object_schema(
                {
                    "name": {"type": "string"},
                    "primary_discovery_tool": {"type": "string"},
                    "introspection_tool": {"type": "string"},
                    "metrics_tool": {"type": "string"},
                    "transport": {"type": "string"},
                    "resources_supported": {"type": "boolean"},
                    "resource_templates_supported": {"type": "boolean"},
                },
                required=[
                    "name",
                    "primary_discovery_tool",
                    "introspection_tool",
                    "metrics_tool",
                    "transport",
                    "resources_supported",
                    "resource_templates_supported",
                ],
                additional_properties=False,
            ),
            "auth_help": _auth_help_schema(),
            "field_inventory": _field_inventory_schema(),
            "concept_notes": _array_of(_concept_note_schema()),
            "tool_limitations": _array_of(_tool_limitation_schema()),
            "tools": _array_of(_tool_capabilities_output_schema()),
            "workflows": _array_of(_workflow_output_schema()),
        },
        required=[
            "server",
            "auth_help",
            "field_inventory",
            "concept_notes",
            "tool_limitations",
            "tools",
            "workflows",
        ],
        additional_properties=False,
    )


@lru_cache(maxsize=1)
def _tool_specs() -> tuple[McpToolSpec, ...]:
    search_agreements_schema = _merge_schema_instances(AgreementsIndexArgsSchema(), AgreementsBulkArgsSchema())
    structured_filter_overrides = _structured_filter_properties()
    agreements_list_overrides = _structured_filter_properties(include_cursor=True, include_xml=True)
    search_agreements_overrides: dict[str, dict[str, object]] = {
        **structured_filter_overrides,
        "query": {
            "type": "string",
            "description": "Optional prefix search over target and acquirer names, or a 4-digit year string.",
            "examples": ["Slack", "2020"],
        },
    }
    sections_search_overrides: dict[str, dict[str, object]] = {
        **structured_filter_overrides,
        "count_mode": {
            "type": "string",
            "enum": list(_COUNT_MODE_VALUES),
            "description": "Count strategy for pagination planning. Use `exact` when planning depends on a guaranteed total count.",
            "examples": ["auto", "exact"],
        },
        "metadata": _enum_array_schema(
            cast(tuple[str, ...], SECTIONS_RESULT_METADATA_FIELDS),
            description="Agreement metadata fields to include under results[].metadata.",
            examples=[["deal_type", "target_industry"]],
        ),
    }
    list_agreement_sections_overrides: dict[str, dict[str, object]] = {
        "sort_by": {
            "type": "string",
            "enum": list(_SECTION_LIST_SORT_FIELDS),
            "description": "Section list sort key.",
        },
    }
    suggest_clause_families_schema = _schema_from_fields(
        "McpSuggestClauseFamiliesArgs",
        {
            "concept": ma_fields.Str(required=True, validate=validate.Length(min=1)),
            "taxonomy": ma_fields.Str(load_default="clauses", validate=validate.OneOf(["clauses", "tax_clauses"])),
            "top_k": ma_fields.Int(load_default=5, validate=validate.Range(min=1, max=10)),
        },
    )
    section_snippet_schema = _schema_from_fields(
        "McpSectionSnippetArgs",
        {
            "section_uuid": ma_fields.Str(required=True, validate=validate.Length(min=1)),
            "focus_terms": ma_fields.List(ma_fields.Str(), load_default=[]),
            "max_chars": ma_fields.Int(load_default=400, validate=validate.Range(min=120, max=1200)),
        },
    )
    return (
        McpToolSpec(
            name="search_agreements",
            description="Discover agreements with text query and page-based results. Supports the same structured agreement filters as list_agreements, including counsel filters, but is best suited for interactive discovery rather than bulk exact retrieval.",
            input_schema=_schema_input_schema(search_agreements_schema, field_overrides=search_agreements_overrides),
            output_schema=_search_agreements_output_schema(),
            examples=(
                {"description": "Find agreements involving a target counsel.", "arguments": {"target_counsel": ["Wachtell, Lipton, Rosen & Katz"]}},
                {"description": "Combine a text lookup with a year filter.", "arguments": {"query": "Target", "year": [2020]}},
            ),
            response_examples=(
                {"description": "Agreement discovery result page.", "content": {"returned_count": 1, "results": [{"agreement_uuid": "a1", "target": "Target A", "acquirer": "Acquirer A"}]}},
            ),
            scopes=("agreements:search",),
            selection_hint="Use for exploratory lookup when you may combine free-text discovery with filters and only need shallow pagination.",
            negative_guidance=(
                "Do not use this tool when you already know the exact structured filters and expect deep pagination; prefer list_agreements.",
                "Do not treat free-text query hits as normalized extracted facts beyond the documented agreement fields.",
            ),
            pagination="page",
            access_behavior="strict_scope_required",
            redaction_behavior="none",
            fulltext_scope=None,
            handler=_search_agreements,
        ),
        McpToolSpec(
            name="search_sections",
            description="Search sections across the corpus when you are looking for clause language samples, taxonomy-linked sections, or text to read. This is not a normalized document-facts surface.",
            input_schema=_schema_input_schema(SectionsArgsSchema(), field_overrides=sections_search_overrides),
            output_schema=_search_sections_output_schema(),
            examples=(
                {"description": "Find sections by taxonomy id.", "arguments": {"standard_id": ["s1"], "page_size": 10}},
                {"description": "Search no-shop style sections with counsel filtering.", "arguments": {"target_counsel": ["Wachtell, Lipton, Rosen & Katz"], "metadata": ["deal_type"]}},
                {"description": "Get an exact total count for pagination planning.", "arguments": {"standard_id": ["2.1"], "count_mode": "exact", "page_size": 10}},
            ),
            response_examples=(
                {"description": "Section search result page.", "content": {"results": [{"section_uuid": "00000000-0000-0000-0000-000000000001", "agreement_uuid": "a1", "standard_id": ["s1"]}], "access": {"tier": "mcp"}}},
            ),
            scopes=("sections:search",),
            selection_hint="Use for clause-language retrieval, taxonomy searches, and agreement-section sampling.",
            negative_guidance=(
                "Do not use this tool as a source of normalized document-level facts; it returns clause text and metadata attached to matching sections.",
                "Do not assume taxonomy hits are always canonical for the user concept; inspect interpretation notes and concept guidance first.",
            ),
            pagination="page",
            access_behavior="strict_scope_required",
            redaction_behavior="none",
            fulltext_scope=None,
            handler=_search_sections,
        ),
        McpToolSpec(
            name="list_agreements",
            description="Retrieve agreements with exact structured filters and cursor pagination. Prefer this over search_agreements for bulk exact retrieval, especially when filters are already known.",
            input_schema=_schema_input_schema(AgreementsBulkArgsSchema(), field_overrides=agreements_list_overrides),
            output_schema=_list_agreements_output_schema(),
            examples=(
                {"description": "Page through agreements by exact counsel filter.", "arguments": {"target_counsel": ["Wachtell, Lipton, Rosen & Katz"], "page_size": 50}},
                {"description": "Retrieve all cash deals with a cursor.", "arguments": {"transaction_consideration": ["cash"], "cursor": None}},
            ),
            response_examples=(
                {"description": "Cursor-based agreement page.", "content": {"returned_count": 1, "has_next": False, "next_cursor": None, "results": [{"agreement_uuid": "a1"}], "access": {"tier": "mcp"}}},
            ),
            scopes=("agreements:search",),
            selection_hint="Use when filters are already known and you expect to paginate deeply or export exact result sets.",
            negative_guidance=(
                "Do not use this tool for free-text exploration; prefer search_agreements when you are still discovering names or years.",
            ),
            pagination="cursor",
            access_behavior="strict_scope_required",
            redaction_behavior="none",
            fulltext_scope=None,
            handler=_list_agreements,
        ),
        McpToolSpec(
            name="list_agreement_sections",
            description="Navigate the sections inside one agreement before drilling into a specific section.",
            input_schema=_schema_input_schema(McpListAgreementSectionsArgsSchema(), field_overrides=list_agreement_sections_overrides),
            output_schema=_list_agreement_sections_output_schema(),
            examples=(
                {"description": "List sections inside one agreement.", "arguments": {"agreement_uuid": "a1", "page_size": 25}},
            ),
            response_examples=(
                {"description": "Section listing for an agreement.", "content": {"agreement_uuid": "a1", "returned_count": 2, "results": [{"section_uuid": "00000000-0000-0000-0000-000000000001"}]}},
            ),
            scopes=("sections:search",),
            selection_hint="Use after identifying an agreement and before calling get_section on one section UUID.",
            negative_guidance=(),
            pagination="page",
            access_behavior="strict_scope_required",
            redaction_behavior="none",
            fulltext_scope=None,
            handler=_list_agreement_sections,
        ),
        McpToolSpec(
            name="get_agreement",
            description="Fetch one agreement document, returning redacted XML unless full-text scope is present.",
            input_schema=_schema_input_schema(McpAgreementArgsSchema()),
            output_schema=_get_agreement_output_schema(),
            examples=(
                {"description": "Fetch one agreement body.", "arguments": {"agreement_uuid": "a1"}},
            ),
            response_examples=(
                {"description": "Redacted agreement response without fulltext scope.", "content": {"target": "Target A", "acquirer": "Acquirer A", "xml": "<document />", "is_redacted": True}},
                {"description": "Full agreement response with fulltext scope.", "content": {"target": "Target A", "acquirer": "Acquirer A", "xml": "<document><article>...</article></document>", "is_redacted": False}},
            ),
            scopes=("agreements:read",),
            selection_hint="Use when you already know the exact agreement UUID and need the agreement payload or XML.",
            negative_guidance=(
                "Do not use this tool for bulk discovery or corpus filtering.",
            ),
            pagination="none",
            access_behavior="partial_access_with_redaction",
            redaction_behavior="redacted_without_fulltext_scope",
            fulltext_scope="agreements:read_fulltext",
            handler=_get_agreement,
        ),
        McpToolSpec(
            name="get_section",
            description="Fetch one section directly by section UUID when you already know the exact section to inspect.",
            input_schema=_schema_input_schema(McpSectionArgsSchema()),
            output_schema=_get_section_output_schema(),
            examples=(
                {"description": "Fetch one section after search_sections.", "arguments": {"section_uuid": "00000000-0000-0000-0000-000000000001"}},
            ),
            response_examples=(
                {"description": "Exact section payload.", "content": {"section_uuid": "00000000-0000-0000-0000-000000000001", "agreement_uuid": "a1", "standard_id": ["s1"]}},
            ),
            scopes=("agreements:read",),
            selection_hint="Use when you already have a section UUID and want the exact section payload.",
            negative_guidance=(),
            pagination="none",
            access_behavior="strict_scope_required",
            redaction_behavior="none",
            fulltext_scope=None,
            handler=_get_section,
        ),
        McpToolSpec(
            name="get_agreement_tax_clauses",
            description="Fetch extracted tax-module clauses for a specific agreement.",
            input_schema=_schema_input_schema(McpAgreementIdentifierSchema()),
            output_schema=_get_agreement_tax_clauses_output_schema(),
            examples=(
                {"description": "Retrieve tax clauses for one agreement.", "arguments": {"agreement_uuid": "a1"}},
            ),
            response_examples=(
                {"description": "Agreement tax clause list.", "content": {"agreement_uuid": "a1", "returned_count": 2, "clauses": [{"clause_uuid": "clause-a1-1", "standard_ids": ["tax_transfer"]}]}},
            ),
            scopes=("agreements:read",),
            selection_hint="Use for agreement-level tax clause extraction once you know the agreement UUID.",
            negative_guidance=(),
            pagination="none",
            access_behavior="strict_scope_required",
            redaction_behavior="none",
            fulltext_scope=None,
            handler=_get_agreement_tax_clauses,
        ),
        McpToolSpec(
            name="get_section_tax_clauses",
            description="Fetch extracted tax-module clauses for a specific section.",
            input_schema=_schema_input_schema(McpSectionArgsSchema()),
            output_schema=_get_section_tax_clauses_output_schema(),
            examples=(
                {"description": "Retrieve tax clauses for one section.", "arguments": {"section_uuid": "00000000-0000-0000-0000-000000000001"}},
            ),
            response_examples=(
                {"description": "Section tax clause list.", "content": {"section_uuid": "00000000-0000-0000-0000-000000000001", "returned_count": 2, "clauses": [{"clause_uuid": "clause-a1-1", "standard_ids": ["tax_transfer"]}]}},
            ),
            scopes=("agreements:read",),
            selection_hint="Use for section-level tax clause extraction when you already have a section UUID.",
            negative_guidance=(),
            pagination="none",
            access_behavior="strict_scope_required",
            redaction_behavior="none",
            fulltext_scope=None,
            handler=_get_section_tax_clauses,
        ),
        McpToolSpec(
            name="list_filter_options",
            description="List valid filter values for agreement and section retrieval. Catalog groups are pluralized, while retrieval arguments are singular, for example target_counsels maps to target_counsel.",
            input_schema=_schema_input_schema(McpFilterOptionsArgsSchema()),
            output_schema=_list_filter_options_output_schema(),
            examples=(
                {"description": "List valid counsel filter values.", "arguments": {"fields": ["target_counsels", "acquirer_counsels"]}},
                {"description": "Inspect deal-status and transaction-price filter catalogs.", "arguments": {"fields": ["deal_statuses", "transaction_price_totals"]}},
            ),
            response_examples=(
                {"description": "Filter catalog response with mapping metadata.", "content": {"fields": ["target_counsels"], "target_counsels": ["Wachtell, Lipton, Rosen & Katz"], "retrieval_parameter_map": {"target_counsels": "target_counsel"}}},
            ),
            scopes=("agreements:search",),
            selection_hint="Use first when you need canonical filter values or need to translate plural catalog groups into retrieval parameter names.",
            negative_guidance=(
                "Do not send the pluralized catalog keys directly to retrieval tools; use retrieval_parameter_map to translate them.",
            ),
            pagination="none",
            access_behavior="strict_scope_required",
            redaction_behavior="none",
            fulltext_scope=None,
            handler=_list_filter_options,
        ),
        McpToolSpec(
            name="suggest_clause_families",
            description="Map a plain-English concept to likely clause-family taxonomy nodes so you can choose the right standard_id before section search. Suggestions may be canonical matches or broader/narrower proxies.",
            input_schema=_schema_input_schema(suggest_clause_families_schema),
            output_schema=_suggest_clause_families_output_schema(),
            examples=(
                {"description": "Find likely taxonomy nodes for MAE carveouts.", "arguments": {"concept": "MAE carveouts", "top_k": 3}},
                {"description": "Map a tax concept to the tax-clause taxonomy.", "arguments": {"concept": "tax-free reorganization", "taxonomy": "tax_clauses"}},
            ),
            response_examples=(
                {"description": "Ranked taxonomy suggestions for a concept.", "content": {"concept": "MAE carveouts", "taxonomy": "clauses", "returned_count": 1, "matches": [{"standard_id": "2.1", "label": "Material Adverse Effect", "path": ["Definitions", "Material Adverse Effect"], "score": 0.93, "matched_terms": ["MAE", "disproportionate effects"], "fit": "proxy", "scope_note": "This taxonomy node is a reasonable proxy and may be broader or narrower than the requested concept.", "confidence": "high", "reason": "Matched concept tokens and clause-family synonyms."}]}},
            ),
            scopes=("sections:search",),
            selection_hint="Use when you know the legal concept but not the right taxonomy id.",
            negative_guidance=(
                "Do not treat the top suggestion as automatically canonical; check fit, confidence, and scope_note before relying on it.",
            ),
            pagination="none",
            access_behavior="strict_scope_required",
            redaction_behavior="none",
            fulltext_scope=None,
            handler=_suggest_clause_families,
        ),
        McpToolSpec(
            name="get_section_snippet",
            description="Extract a focused plain-text snippet from one section, optionally centered on search terms, instead of returning the whole XML block. This is a reading aid, not a canonical extracted-facts surface.",
            input_schema=_schema_input_schema(section_snippet_schema),
            output_schema=_section_snippet_output_schema(),
            examples=(
                {"description": "Extract a short excerpt around a carveout phrase.", "arguments": {"section_uuid": "00000000-0000-0000-0000-000000000001", "focus_terms": ["disproportionate effects"], "max_chars": 350}},
            ),
            response_examples=(
                {"description": "Focused section snippet.", "content": {"agreement_uuid": "a1", "section_uuid": "00000000-0000-0000-0000-000000000001", "standard_id": ["1.2"], "article_title": "ARTICLE I", "section_title": "Material Adverse Effect", "snippet": "...disproportionate effects on the Company relative to others in the industry...", "matched_terms": ["disproportionate effects"], "source_length": 512}},
            ),
            scopes=("sections:search",),
            selection_hint="Use after search_sections when you need a quick excerpt for comparison or quoting.",
            negative_guidance=(
                "Do not treat the snippet as the complete section; fetch get_section if exact full context matters.",
            ),
            pagination="none",
            access_behavior="strict_scope_required",
            redaction_behavior="none",
            fulltext_scope=None,
            handler=_get_section_snippet,
        ),
        McpToolSpec(
            name="get_server_metrics",
            description="Return in-process MCP metrics, including per-tool call counts, latency buckets, error categories, and auth-failure counts since process start.",
            input_schema=_empty_schema(),
            output_schema=_metrics_output_schema(),
            examples=(
                {"description": "Inspect MCP usage and latency metrics.", "arguments": {}},
            ),
            response_examples=(
                {"description": "Metrics snapshot with one recorded tool.", "content": {"latency_bucket_bounds_ms": [50, 100, 250], "tool_calls": {"search_agreements": {"calls": 1, "errors": 0}}, "auth_failures": {}}},
            ),
            scopes=("agreements:search",),
            selection_hint="Use for operational monitoring and to see which MCP tools are slow or error-prone.",
            negative_guidance=(),
            pagination="none",
            access_behavior="strict_scope_required",
            redaction_behavior="none",
            fulltext_scope=None,
            handler=_get_server_metrics,
        ),
        McpToolSpec(
            name="get_server_capabilities",
            description="Explain available MCP tools, selection guidance, pagination styles, scope requirements, and corpus semantics for this server. Use it to learn which concepts are first-class, taxonomy-based proxies, text-derived, or not represented.",
            input_schema=_empty_schema(),
            output_schema=_server_capabilities_output_schema(),
            examples=(
                {"description": "Inspect tool guidance before starting a workflow.", "arguments": {}},
            ),
            response_examples=(
                {"description": "Capabilities payload with tool guidance.", "content": {"server": {"name": "pandects-mcp", "transport": "http_jsonrpc"}, "tools": [{"name": "search_agreements"}], "workflows": [{"name": "discover agreements by counsel"}]}},
            ),
            scopes=("agreements:search",),
            selection_hint="Use when you need a machine-readable guide to tool choice, filters, scopes, and supported workflows.",
            negative_guidance=(
                "Do not guess whether a concept is canonical or proxy-based when this tool can tell you directly.",
            ),
            pagination="none",
            access_behavior="strict_scope_required",
            redaction_behavior="none",
            fulltext_scope=None,
            handler=_get_server_capabilities,
        ),
        McpToolSpec(
            name="get_clause_taxonomy",
            description="Fetch the clause taxonomy tree used for section search standard IDs.",
            input_schema=_empty_schema(),
            output_schema=_taxonomy_output_schema(),
            examples=(
                {"description": "Inspect the clause taxonomy.", "arguments": {}},
                {"description": "Discover valid standard_id values before taxonomy-filtered section search.", "arguments": {}},
            ),
            response_examples=(
                {"description": "Clause taxonomy tree.", "content": {"Deal Protection": {"id": "1", "children": {}}}},
            ),
            scopes=("sections:search",),
            selection_hint="Use when you need valid standard_id values for section taxonomy filtering.",
            negative_guidance=(),
            pagination="none",
            access_behavior="strict_scope_required",
            redaction_behavior="none",
            fulltext_scope=None,
            handler=_get_clause_taxonomy,
        ),
        McpToolSpec(
            name="get_tax_clause_taxonomy",
            description="Fetch the tax clause taxonomy tree used for tax-clause research.",
            input_schema=_empty_schema(),
            output_schema=_taxonomy_output_schema(),
            examples=(
                {"description": "Inspect the tax clause taxonomy.", "arguments": {}},
                {"description": "Look up valid tax clause ids before agreement tax-clause retrieval.", "arguments": {}},
            ),
            response_examples=(
                {"description": "Tax clause taxonomy tree.", "content": {"Tax": {"id": "tax", "children": {}}}},
            ),
            scopes=("sections:search",),
            selection_hint="Use when you need tax-clause taxonomy ids before calling a tax-clause retrieval tool.",
            negative_guidance=(),
            pagination="none",
            access_behavior="strict_scope_required",
            redaction_behavior="none",
            fulltext_scope=None,
            handler=_get_tax_clause_taxonomy,
        ),
        McpToolSpec(
            name="get_counsel_catalog",
            description="Fetch canonical counsel names for filter selection and normalization.",
            input_schema=_empty_schema(),
            output_schema=_counsel_catalog_output_schema(),
            examples=(
                {"description": "List canonical counsel names.", "arguments": {}},
                {"description": "Get a normalized counsel name before applying target_counsel filters.", "arguments": {}},
            ),
            response_examples=(
                {"description": "Canonical counsel catalog.", "content": {"counsel": [{"counsel_id": 1, "canonical_name": "Wachtell, Lipton, Rosen & Katz"}]}},
            ),
            scopes=("sections:search",),
            selection_hint="Use when you need canonical firm names before counsel-filtered agreement or section retrieval.",
            negative_guidance=(),
            pagination="none",
            access_behavior="strict_scope_required",
            redaction_behavior="none",
            fulltext_scope=None,
            handler=_get_counsel_catalog,
        ),
        McpToolSpec(
            name="get_naics_catalog",
            description="Fetch NAICS sectors and subsectors for industry reasoning and filter selection.",
            input_schema=_empty_schema(),
            output_schema=_naics_catalog_output_schema(),
            examples=(
                {"description": "List NAICS sectors and subsectors.", "arguments": {}},
                {"description": "Find canonical industry labels before target_industry filtering.", "arguments": {}},
            ),
            response_examples=(
                {"description": "NAICS sector catalog.", "content": {"sectors": [{"sector_code": "11", "sub_sectors": [{"sub_sector_code": "111"}]}]}},
            ),
            scopes=("sections:search",),
            selection_hint="Use when you need canonical industry labels before industry-filtered retrieval.",
            negative_guidance=(),
            pagination="none",
            access_behavior="strict_scope_required",
            redaction_behavior="none",
            fulltext_scope=None,
            handler=_get_naics_catalog,
        ),
        McpToolSpec(
            name="get_agreements_summary",
            description="Fetch high-level corpus counts for agreements, sections, and pages.",
            input_schema=_empty_schema(),
            output_schema=_agreements_summary_output_schema(),
            examples=({"description": "Get top-level corpus counts.", "arguments": {}},),
            response_examples=(
                {"description": "Corpus count summary.", "content": {"agreements": 1, "sections": 2, "pages": 5}},
            ),
            scopes=("agreements:search",),
            selection_hint="Use for top-level corpus sizing before deeper analysis.",
            negative_guidance=(),
            pagination="none",
            access_behavior="strict_scope_required",
            redaction_behavior="none",
            fulltext_scope=None,
            handler=_get_agreements_summary,
        ),
        McpToolSpec(
            name="get_agreement_trends",
            description="Fetch ownership and industry trend analytics for the agreement corpus.",
            input_schema=_empty_schema(),
            output_schema=_agreement_trends_output_schema(),
            examples=(
                {"description": "Inspect ownership and industry trends.", "arguments": {}},
                {"description": "Compare public/private deal mix and buyer-type patterns by year.", "arguments": {}},
            ),
            response_examples=(
                {"description": "Aggregated trend payload.", "content": {"ownership": {"mix_by_year": [{"year": 2020, "public_deal_count": 1}]}, "industries": {"target_industries_by_year": [{"year": 2020, "industry": "Crop Production"}]}}},
            ),
            scopes=("agreements:search",),
            selection_hint="Use for aggregated corpus analytics rather than document retrieval.",
            negative_guidance=(),
            pagination="none",
            access_behavior="strict_scope_required",
            redaction_behavior="none",
            fulltext_scope=None,
            handler=_get_agreement_trends,
        ),
    )


@lru_cache(maxsize=1)
def _tool_spec_map() -> dict[str, McpToolSpec]:
    return {spec.name: spec for spec in _tool_specs()}


def _field_inventory_payload() -> dict[str, object]:
    return {
        "agreement_fields": [
            {
                "name": field_name,
                "applies_to_tools": ["search_agreements", "list_agreements", "get_agreement", "search_sections"],
                "source_table_or_surface": "agreements",
                "representation": "first_class_agreement_field",
            }
            for field_name in (
                "year",
                "target",
                "acquirer",
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
                "filing_date",
                "announce_date",
                "close_date",
                "url",
            )
        ],
        "section_fields": [
            {
                "name": field_name,
                "applies_to_tools": ["search_sections", "get_section", "get_section_snippet", "list_agreement_sections"],
                "source_table_or_surface": "sections/latest_sections_search",
                "representation": "first_class_section_field",
            }
            for field_name in (
                "section_uuid",
                "article_title",
                "section_title",
                "xml",
            )
        ],
        "taxonomy_assignment_fields": [
            {
                "name": "standard_id",
                "applies_to_tools": ["search_sections", "get_section", "suggest_clause_families", "get_clause_taxonomy"],
                "source_table_or_surface": "latest_sections_search_standard_ids",
                "representation": "taxonomy_assignment",
            }
        ],
    }


def _concept_notes_payload() -> list[dict[str, object]]:
    return [
        {
            "concept": "no-shop",
            "recommended_tools": ["suggest_clause_families", "get_clause_taxonomy", "search_sections"],
            "representation": "taxonomy_assignment",
            "canonical_or_proxy": "proxy",
            "scope_note": "No-shop style concepts are usually approached through clause-family taxonomy and section text rather than a dedicated normalized agreement field.",
        },
        {
            "concept": "fiduciary out",
            "recommended_tools": ["get_clause_taxonomy", "search_sections", "get_section_snippet"],
            "representation": "taxonomy_assignment",
            "canonical_or_proxy": "canonical",
            "scope_note": "This concept is represented primarily through taxonomy-assigned sections, then inspected in section text.",
        },
        {
            "concept": "MAE carveouts",
            "recommended_tools": ["suggest_clause_families", "search_sections", "get_section_snippet"],
            "representation": "taxonomy_assignment",
            "canonical_or_proxy": "proxy",
            "scope_note": "MAE carveouts are often proxied by Material Adverse Effect taxonomy nodes plus section text, especially disproportionate-effects language.",
        },
        {
            "concept": "disproportionate effects",
            "recommended_tools": ["suggest_clause_families", "search_sections", "get_section_snippet"],
            "representation": "taxonomy_assignment",
            "canonical_or_proxy": "canonical",
            "scope_note": "This concept is commonly represented by a specific descendant taxonomy node inside MAE definitions.",
        },
        {
            "concept": "specific performance",
            "recommended_tools": ["suggest_clause_families", "search_sections", "get_section_snippet"],
            "representation": "taxonomy_assignment",
            "canonical_or_proxy": "canonical",
            "scope_note": "Specific performance is typically represented through clause-family assignments and supporting section text.",
        },
        {
            "concept": "antitrust efforts",
            "recommended_tools": ["suggest_clause_families", "search_sections", "get_section_snippet"],
            "representation": "taxonomy_assignment",
            "canonical_or_proxy": "proxy",
            "scope_note": "Antitrust-efforts concepts may span several nearby clause families and should be validated in the returned section text.",
        },
    ]


def _tool_limitations_payload(specs: tuple[McpToolSpec, ...]) -> list[dict[str, object]]:
    limitations: list[dict[str, object]] = []
    for spec in specs:
        for note in spec.negative_guidance:
            limitations.append(
                {
                    "tool": spec.name,
                    "use_when": spec.selection_hint,
                    "do_not_use_for": note,
                }
            )
    return limitations


def _tool_list_entry(spec: McpToolSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "description": spec.description,
        "inputSchema": spec.input_schema,
        "outputSchema": spec.output_schema,
        "examples": list(spec.examples),
        "annotations": {
            "selectionHint": spec.selection_hint,
            "negativeGuidance": list(spec.negative_guidance),
            "pagination": spec.pagination,
            "requiredScopes": list(spec.scopes),
        },
    }


def _server_capabilities_payload() -> dict[str, object]:
    specs = _tool_specs()
    return {
        "server": {
            "name": "pandects-mcp",
            "primary_discovery_tool": "list_filter_options",
            "introspection_tool": "get_server_capabilities",
            "metrics_tool": "get_server_metrics",
            "transport": "http_jsonrpc",
            "resources_supported": False,
            "resource_templates_supported": False,
        },
        "auth_help": {
            "login_required": True,
            "relogin_hint": "If a bearer token is missing, invalid, or expired, re-authenticate and retry the MCP call.",
            "fulltext_scope": "agreements:read_fulltext",
            "invalid_or_expired_token_message": "If credentials look stale, sign in again and resend the bearer token.",
        },
        "field_inventory": _field_inventory_payload(),
        "concept_notes": _concept_notes_payload(),
        "tool_limitations": _tool_limitations_payload(specs),
        "tools": [
            {
                "name": spec.name,
                "description": spec.description,
                "required_scopes": list(spec.scopes),
                "pagination": spec.pagination,
                "selection_hint": spec.selection_hint,
                "negative_guidance": list(spec.negative_guidance),
                "examples": list(spec.examples),
                "response_examples": list(spec.response_examples),
                "access": {
                    "scope_behavior": spec.access_behavior,
                    "redaction": spec.redaction_behavior,
                    "failure_status_code": 403,
                    "fulltext_scope": spec.fulltext_scope,
                },
                "limits": _tool_limits_for_pagination(spec.pagination),
                "input_schema": spec.input_schema,
                "output_schema": spec.output_schema,
            }
            for spec in specs
        ],
        "workflows": [
            {
                "name": "discover agreements by counsel",
                "steps": ["list_filter_options", "search_agreements", "get_agreement"],
            },
            {
                "name": "sample clause language by taxonomy",
                "steps": ["get_clause_taxonomy", "search_sections", "get_section"],
            },
            {
                "name": "map a plain-English concept to clause samples",
                "steps": ["suggest_clause_families", "search_sections", "get_section_snippet"],
            },
            {
                "name": "filter agreements exactly and paginate deeply",
                "steps": ["list_filter_options", "list_agreements", "get_agreement"],
            },
            {
                "name": "inspect MCP health and hot paths",
                "steps": ["get_server_metrics", "get_server_capabilities"],
            },
        ],
    }


def _get_server_capabilities(
    *,
    principal: McpPrincipal,
) -> McpToolResult:
    if not principal.scopes:
        raise PermissionError("Missing required scope: agreements:search")
    response = _server_capabilities_payload()
    return McpToolResult(
        text=f"Returned MCP server capabilities for {len(cast(list[object], response['tools']))} tool(s).",
        structured_content=response,
    )


def _get_server_metrics(
    *,
    principal: McpPrincipal,
) -> McpToolResult:
    if not principal.scopes:
        raise PermissionError("Missing required scope: agreements:search")
    response = get_mcp_metrics_registry().snapshot()
    tool_calls = cast(dict[str, object], response["tool_calls"])
    return McpToolResult(
        text=f"Returned MCP server metrics for {len(tool_calls)} recorded tool(s).",
        structured_content=response,
    )


def tool_definitions() -> list[dict[str, object]]:
    return [_tool_list_entry(spec) for spec in _tool_specs()]


def call_tool(
    name: str,
    *,
    arguments: dict[str, object],
    principal: McpPrincipal,
    sections_service_deps: SectionsServiceDeps,
    agreements_deps: AgreementsDeps,
    reference_data_deps: ReferenceDataDeps,
) -> McpToolResult:
    spec = _tool_spec_map().get(name)
    if spec is None:
        raise KeyError(name)
    handler_kwargs: dict[str, object] = {"principal": principal}
    if name in {"search_agreements", "list_agreements", "get_agreement", "get_section", "get_section_snippet", "get_agreement_tax_clauses", "get_section_tax_clauses", "list_filter_options", "get_agreements_summary"}:
        handler_kwargs["deps"] = agreements_deps
    if name in {"search_sections", "list_agreement_sections"}:
        handler_kwargs["deps"] = sections_service_deps
    if name in {"get_clause_taxonomy", "get_tax_clause_taxonomy", "get_counsel_catalog", "get_naics_catalog", "suggest_clause_families"}:
        handler_kwargs["deps"] = reference_data_deps
    if name in {"search_agreements", "search_sections", "list_agreements", "list_agreement_sections", "get_agreement", "get_section", "get_section_snippet", "get_agreement_tax_clauses", "get_section_tax_clauses", "list_filter_options", "suggest_clause_families"}:
        handler_kwargs["payload"] = arguments
    if name == "get_agreement_trends":
        handler_kwargs["deps"] = agreements_deps
        handler_kwargs["reference_data_deps"] = reference_data_deps
    result = spec.handler(**handler_kwargs)
    output_errors = _validate_output_against_schema(spec.output_schema, result.structured_content)
    if output_errors:
        raise McpOutputValidationError(output_errors)
    return result


__all__ = ["McpOutputValidationError", "McpToolResult", "call_tool", "tool_definitions"]
