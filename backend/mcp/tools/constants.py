from __future__ import annotations

_SECTION_LIST_SORT_FIELDS = ("article_title", "section_title", "section_uuid", "document_order")
_TRANSACTION_PRICE_BUCKET_OPTIONS = (
    "0 - 100M",
    "100M - 250M",
    "250M - 500M",
    "500M - 750M",
    "750M - 1B",
    "1B - 2B",
    "2B - 5B",
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
_CAPABILITIES_SECTIONS_ALL = ("server", "auth_help", "field_inventory", "concept_notes", "tool_limitations", "workflows", "tools")
_CAPABILITIES_SECTIONS_DEFAULT = ("server", "auth_help", "field_inventory", "concept_notes", "tool_limitations", "workflows")
_COUNT_MODE_VALUES = ("auto", "exact")
_COUNT_METHOD_VALUES = ("query_count", "table_estimate", "filtered_lower_bound")
_COUNT_RELIABILITY_VALUES = ("high", "medium", "low")
_TAXONOMY_MATCH_MODE_VALUES = ("exact_node", "expanded_descendants")
_CLAUSE_FIT_VALUES = ("canonical", "proxy", "broad_match")
_CLAUSE_CONFIDENCE_VALUES = ("high", "medium", "low")
