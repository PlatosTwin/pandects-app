from __future__ import annotations

from backend.mcp.tools.constants import (
    _CLAUSE_CONFIDENCE_VALUES,
    _CLAUSE_FIT_VALUES,
    _COUNT_METHOD_VALUES,
    _COUNT_RELIABILITY_VALUES,
    _FIELD_REPRESENTATION_VALUES,
    _FILTER_OPTIONS_FIELDS,
    _TAXONOMY_MATCH_MODE_VALUES,
)
from backend.mcp.tools.schema_utils import (
    _array_of,
    _filter_option_metadata,
    _object_schema,
    _pagination_response_properties,
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
            "section_count": {"type": ["integer", "null"]},
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
        "prob_filing": {"type": ["number", "null"]},
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
        "section_count": {"type": ["integer", "null"]},
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
            "filing_date": {"type": ["string", "null"]},
            "transaction_price_total": {"type": ["number", "null"]},
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
        required=["id", "section_uuid", "verified"],
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
            "unique_agreement_count": {"type": "integer"},
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
            "total_agreement_sections": {"type": "integer"},
            "results": _array_of(_list_section_result_schema()),
            "returned_count": {"type": "integer"},
        }
    )
    return _object_schema(
        properties,
        required=["agreement_uuid", "total_agreement_sections", "results", "returned_count", "page", "page_size", "total_count", "total_pages", "has_next", "has_prev", "total_count_is_approximate"],
    )


def _batch_agreement_sections_output_schema() -> dict[str, object]:
    per_agreement_schema = _object_schema(
        {
            "agreement_uuid": {"type": "string"},
            "total_agreement_sections": {"type": "integer"},
            "sections": _array_of(_list_section_result_schema()),
            "section_count": {"type": "integer"},
        },
        required=["agreement_uuid", "total_agreement_sections", "sections", "section_count"],
    )
    return _object_schema(
        {
            "results": _array_of(per_agreement_schema),
            "returned_agreement_count": {"type": "integer"},
            "total_section_count": {"type": "integer"},
        },
        required=["results", "returned_agreement_count", "total_section_count"],
    )


def _batch_section_snippet_output_schema() -> dict[str, object]:
    item_schema = _object_schema(
        {
            "agreement_uuid": {"type": ["string", "null"]},
            "section_uuid": {"type": "string"},
            "standard_id": _array_of({"type": "string"}),
            "article_title": {"type": ["string", "null"]},
            "section_title": {"type": ["string", "null"]},
            "snippet": {"type": "string"},
            "matched_terms": _array_of({"type": "string"}),
            "source_length": {"type": "integer"},
            "monetary_values": _array_of({"type": "string"}),
        },
        required=["agreement_uuid", "section_uuid", "standard_id", "article_title", "section_title", "snippet", "matched_terms", "source_length", "monetary_values"],
        additional_properties=False,
    )
    return _object_schema(
        {
            "results": _array_of(item_schema),
            "returned_count": {"type": "integer"},
        },
        required=["results", "returned_count"],
        additional_properties=False,
    )


def _batch_sections_output_schema() -> dict[str, object]:
    item_schema = _object_schema(
        {
            "agreement_uuid": {"type": ["string", "null"]},
            "section_uuid": {"type": "string"},
            "standard_id": _array_of({"type": "string"}),
            "article_title": {"type": ["string", "null"]},
            "section_title": {"type": ["string", "null"]},
            "xml": {"type": ["string", "null"]},
            "xml_truncated": {"type": "boolean"},
            "target": {"type": ["string", "null"]},
            "acquirer": {"type": ["string", "null"]},
            "year": {"type": ["integer", "null"]},
            "filing_date": {"type": ["string", "null"]},
            "transaction_price_total": {"type": ["number", "null"]},
            "monetary_values": _array_of({"type": "string"}),
        },
        required=["section_uuid", "standard_id", "xml_truncated", "monetary_values"],
        additional_properties=False,
    )
    return _object_schema(
        {
            "results": _array_of(item_schema),
            "returned_count": {"type": "integer"},
        },
        required=["results", "returned_count"],
        additional_properties=False,
    )


def _get_agreement_output_schema() -> dict[str, object]:
    return _object_schema(
        {
            "year": {"type": ["integer", "null"]},
            "target": {"type": ["string", "null"]},
            "acquirer": {"type": ["string", "null"]},
            "filing_date": {"type": ["string", "null"]},
            "prob_filing": {"type": ["number", "null"]},
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
            "monetary_values": _array_of({"type": "string"}),
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
            "monetary_values",
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
        "industry_labels": {"type": "object", "additionalProperties": {"type": "string"}},
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
            "sections_returned": _array_of({"type": "string"}),
        },
        required=["sections_returned"],
        additional_properties=False,
    )


_ANY_COUNSEL_OVERRIDE: dict[str, object] = {
    "type": "array",
    "items": {"type": "string"},
    "description": "Filter to agreements where the firm appears on either side (target or acquirer). Use instead of target_counsel + acquirer_counsel when side is unknown.",
}

_YEAR_RANGE_OVERRIDES: dict[str, dict[str, object]] = {
    "year_min": {
        "type": ["integer", "null"],
        "description": "Earliest filing year to include (inclusive). Use with year_max for a range; use year for specific individual years.",
    },
    "year_max": {
        "type": ["integer", "null"],
        "description": "Latest filing year to include (inclusive). Use with year_min for a range.",
    },
    "filed_after": {
        "type": ["string", "null"],
        "description": "Include only agreements with filing_date >= this date (ISO 8601: YYYY-MM-DD). Provides sub-year precision over year_min.",
        "examples": ["2022-06-01"],
    },
    "filed_before": {
        "type": ["string", "null"],
        "description": "Include only agreements with filing_date < this date (ISO 8601: YYYY-MM-DD). Provides sub-year precision over year_max.",
        "examples": ["2024-01-01"],
    },
}


