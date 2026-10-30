from __future__ import annotations

from typing import cast

# Argument descriptions for the MCP surface of the four structured-filter search tools.
#
# They replace the longer descriptions the shared REST schemas carry, for two reasons:
# - Codex compacts any MCP tool input schema over 5,000 bytes (after dropping keys its
#   schema type does not model, such as examples and minimum) by stripping every
#   argument description from it. The REST descriptions pushed search_sections,
#   search_agreements and list_agreements over that budget, so Codex users saw these
#   tools with no argument guidance at all.
# - Tool-search hosts (Claude, Codex) index argument names, and Claude also indexes
#   their descriptions, so a bare filter such as target_counsel was invisible to a
#   query like "law firm".
# Keep entries short; tests hold every tool under the Codex budget. A name missing here
# keeps whatever description its schema already has; an empty string removes it.
_SHARED: dict[str, str] = {
    "year": "Filing years.",
    "year_min": "Earliest filing year (inclusive).",
    "year_max": "Latest filing year (inclusive).",
    "filed_after": "Only filings on or after this date (YYYY-MM-DD).",
    "filed_before": "Only filings before this date (YYYY-MM-DD).",
    "target": "Exact target company names.",
    "acquirer": "Exact acquirer company names.",
    "transaction_price_total": "Total deal-value buckets.",
    "transaction_price_stock": "Stock deal-value buckets; same labels as transaction_price_total.",
    "transaction_price_cash": "Cash deal-value buckets; same labels as transaction_price_total.",
    "transaction_price_assets": "Asset deal-value buckets; same labels as transaction_price_total.",
    "transaction_consideration": "cash, stock, or mixed.",
    "target_type": "public or private.",
    "acquirer_type": "public or private (PE-backed buyers: acquirer_pe).",
    "target_counsel": "Target-side law firms (get_counsel_catalog names).",
    "acquirer_counsel": "Acquirer-side law firms (get_counsel_catalog names).",
    "any_counsel": "Law firms on either side; use when the side is unknown.",
    "target_industry": 'NAICS codes (e.g. "511"), not labels.',
    "acquirer_industry": 'NAICS codes (e.g. "511"), not labels.',
    "deal_status": "complete, pending, or cancelled.",
    "attitude": "friendly, unsolicited, or hostile.",
    "deal_type": "merger, stock_acquisition, asset_acquisition, membership_interest_purchase, or tender_offer.",
    "purpose": "strategic or financial.",
    "target_pe": "Target is PE-owned or -backed.",
    "acquirer_pe": "Acquirer is PE-owned or -backed.",
    "include_dump": "Include dump_version in the body.",
    "count_mode": "`exact` guarantees total_count; `auto` may estimate.",
    "page": "1-based page number.",
    # An empty string drops the description: the name and enum already say it all.
    "agreement_uuid": "",
    "section_uuid": "",
    "page_size": "",
    "sort_by": "",
    "sort_dir": "",
    "sort_direction": "",
}

_AGREEMENT_STANDARD_ID = (
    "Only agreements with a section tagged with any of these clause-family ids "
    "(16-hex, from suggest_clause_families). Unknown ids are listed in "
    "interpretation.unrecognized_standard_ids."
)
_AGREEMENT_FIELDS = (
    'Return only these row keys (agreement_uuid is always kept), e.g. ["agreement_uuid", "url"] '
    "for a bulk export."
)

_PER_TOOL: dict[str, dict[str, str]] = {
    "search_agreements": {
        "standard_id": _AGREEMENT_STANDARD_ID,
        "fields": _AGREEMENT_FIELDS,
    },
    "list_agreements": {
        "standard_id": _AGREEMENT_STANDARD_ID,
        "fields": _AGREEMENT_FIELDS,
        "cursor": "Cursor from the previous page.",
        "include_xml": "Include full agreement XML (large; needs agreements:read_fulltext).",
        "sort_by": "Sort key (default agreement_uuid). A cursor works only with the sort_by/sort_dir it came from.",
    },
    "search_sections": {
        "standard_id": (
            "16-hex clause-family ids (suggest_clause_families, get_clause_taxonomy), not dotted "
            "numbers. Parents include descendants; unknown ids are listed in interpretation."
        ),
        "text_query": (
            "Literal, case-insensitive text (max 256 chars, 24 terms). `*` only at a word end "
            "(max 4, stems of 3+ chars). Not semantic."
        ),
        "text_match_mode": "phrase keeps term order.",
        "metadata": "Extra fields under results[].metadata.",
        "include_xml": "Include full section XML (large).",
        "include_snippet": "Add a snippet, matched_terms and monetary_values to each result.",
        "snippet_focus_terms": "Snippet focus terms (default: text_query terms).",
        "snippet_max_chars": "120-1200 (default 400).",
    },
    "search_tax_clauses": {
        "tax_standard_id": (
            "Dotted tax.* taxonomy ids (from suggest_clause_families with taxonomy='tax_clauses'). "
            "Parents include descendants."
        ),
        "include_rep_warranty": "Also include clauses inside reps & warranties (default false).",
        "clause_uuid": "Only this tax clause.",
    },
}


def with_compact_arg_descriptions(tool_name: str, input_schema: dict[str, object]) -> dict[str, object]:
    overrides = _PER_TOOL.get(tool_name)
    if overrides is None:
        return input_schema
    properties = cast(dict[str, dict[str, object]], input_schema.get("properties", {}))
    compacted: dict[str, object] = {}
    for name, schema in properties.items():
        description = overrides.get(name, _SHARED.get(name))
        if description is None:
            compacted[name] = schema
        elif description:
            compacted[name] = {**schema, "description": description}
        else:
            compacted[name] = {key: value for key, value in schema.items() if key != "description"}
    return {**input_schema, "properties": compacted}
