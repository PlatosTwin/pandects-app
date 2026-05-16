from __future__ import annotations

from functools import lru_cache
from typing import cast

from marshmallow import fields as ma_fields, validate

from backend.auth.mcp_runtime import McpPrincipal
from backend.mcp.metrics import get_mcp_metrics_registry
from backend.mcp.tools.args_schemas import (
    McpAgreementArgsSchema,
    McpAgreementIdentifierSchema,
    McpBatchAgreementSectionsArgsSchema,
    McpFilterOptionsArgsSchema,
    McpListAgreementSectionsArgsSchema,
    McpSearchAgreementsExtraArgsSchema,
    McpSectionArgsSchema,
)
from backend.mcp.tools.constants import (
    _CAPABILITIES_SECTIONS_ALL,
    _CAPABILITIES_SECTIONS_DEFAULT,
    _COUNT_MODE_VALUES,
    _SECTION_LIST_SORT_FIELDS,
)
from backend.mcp.tools.dispatch import (
    McpOutputValidationError,
    McpToolResult,
    McpToolSpec,
    _validate_output_against_schema,
)
from backend.mcp.tools.handlers import (
    _get_agreement,
    _get_agreement_tax_clauses,
    _get_agreement_trends,
    _get_agreements_summary,
    _get_clause_taxonomy,
    _get_counsel_catalog,
    _get_naics_catalog,
    _get_section,
    _get_section_snippet,
    _get_section_snippets_batch,
    _get_section_tax_clauses,
    _get_sections_batch,
    _get_tax_clause_taxonomy,
    _list_agreement_sections,
    _list_agreement_sections_batch,
    _list_agreements,
    _list_filter_options,
    _search_agreements,
    _search_sections,
    _suggest_clause_families,
)
from backend.mcp.tools.output_schemas import (
    _agreement_trends_output_schema,
    _agreements_summary_output_schema,
    _batch_agreement_sections_output_schema,
    _batch_section_snippet_output_schema,
    _batch_sections_output_schema,
    _counsel_catalog_output_schema,
    _empty_schema,
    _get_agreement_output_schema,
    _get_agreement_tax_clauses_output_schema,
    _get_section_output_schema,
    _get_section_tax_clauses_output_schema,
    _list_agreement_sections_output_schema,
    _list_agreements_output_schema,
    _list_filter_options_output_schema,
    _metrics_output_schema,
    _naics_catalog_output_schema,
    _search_agreements_output_schema,
    _search_sections_output_schema,
    _section_snippet_output_schema,
    _server_capabilities_output_schema,
    _suggest_clause_families_output_schema,
    _taxonomy_output_schema,
    _tool_limits_for_pagination,
)
from backend.mcp.tools.schema_utils import (
    _enum_array_schema,
    _merge_schema_instances,
    _object_schema,
    _schema_from_fields,
    _schema_input_schema,
    _structured_filter_properties,
)
from backend.routes.deps import AgreementsDeps, ReferenceDataDeps, SectionsServiceDeps
from backend.schemas.public_api import (
    AgreementsBulkArgsSchema,
    AgreementsIndexArgsSchema,
)
from backend.schemas.sections import SECTIONS_RESULT_METADATA_FIELDS, SectionsArgsSchema



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


@lru_cache(maxsize=1)
def _tool_specs() -> tuple[McpToolSpec, ...]:
    search_agreements_schema = _merge_schema_instances(AgreementsIndexArgsSchema(), AgreementsBulkArgsSchema(), McpSearchAgreementsExtraArgsSchema())
    structured_filter_overrides = _structured_filter_properties()
    agreements_list_overrides: dict[str, dict[str, object]] = {
        **_structured_filter_properties(include_cursor=True, include_xml=True),
        "any_counsel": _ANY_COUNSEL_OVERRIDE,
        **_YEAR_RANGE_OVERRIDES,
    }
    search_agreements_overrides: dict[str, dict[str, object]] = {
        **structured_filter_overrides,
        "any_counsel": _ANY_COUNSEL_OVERRIDE,
        **_YEAR_RANGE_OVERRIDES,
        "query": {
            "type": "string",
            "description": "Optional prefix search over target and acquirer names, or a 4-digit year string.",
            "examples": ["Slack", "2020"],
        },
    }
    sections_search_overrides: dict[str, dict[str, object]] = {
        **structured_filter_overrides,
        **_YEAR_RANGE_OVERRIDES,
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
        "include_xml": {
            "type": "boolean",
            "description": "When true, include full section XML in each result. Omitted by default to keep responses compact.",
        },
    }
    list_agreement_sections_overrides: dict[str, dict[str, object]] = {
        "sort_by": {
            "type": "string",
            "enum": list(_SECTION_LIST_SORT_FIELDS),
            "description": "Section list sort key. One of: `article_title`, `section_title`, `section_uuid`, `document_order`.",
        },
        "include_standard_ids": {
            "type": "boolean",
            "description": "When false, omit standard_id from each section entry to reduce response size. Default true.",
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
            description="Find merger agreements by target/acquirer name, year, counsel, industry, deal type, any of the standard M&A filters, or by clause taxonomy (standard_id). Passing standard_id filters to agreements that contain at least one section tagged with that taxonomy node — this is the efficient way to find all deals with a specific clause type without paginating search_sections. Best for exploratory discovery where you may combine a free-text hint with structured filters. For deep pagination or bulk exports of a known filter set, use list_agreements instead.",
            input_schema=_schema_input_schema(search_agreements_schema, field_overrides=search_agreements_overrides),
            output_schema=_search_agreements_output_schema(),
            examples=(
                {"description": "Find agreements involving a target counsel.", "arguments": {"target_counsel": ["Wachtell, Lipton, Rosen & Katz"]}},
                {"description": "Combine a text lookup with a year filter.", "arguments": {"query": "Target", "year": [2020]}},
                {"description": "Find all agreements containing a go-shop or no-shop clause by taxonomy id.", "arguments": {"standard_id": ["1a7aeab47932d0d4"]}},
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
            description="Search individual sections (clause-level text) across the corpus by taxonomy node, keywords, and the same M&A filters as search_agreements. Returns clause language and the agreement context, not extracted document-level facts. Pair with suggest_clause_families when you only know the concept in plain English.",
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
            description="Paginate through agreements that match an exact structured filter set, with cursor pagination suitable for exporting or iterating large result sets. Supports standard_id to filter by clause taxonomy — pass a taxonomy node id to get only agreements containing that clause type. Use when filters are already known and you expect to scan many pages; use search_agreements for exploratory discovery.",
            input_schema=_schema_input_schema(AgreementsBulkArgsSchema(), field_overrides=agreements_list_overrides),
            output_schema=_list_agreements_output_schema(),
            examples=(
                {"description": "Page through agreements by exact counsel filter.", "arguments": {"target_counsel": ["Wachtell, Lipton, Rosen & Katz"], "page_size": 50}},
                {"description": "Retrieve all cash deals with a cursor.", "arguments": {"transaction_consideration": ["cash"], "cursor": None}},
                {"description": "Export all agreements containing a specific clause type by taxonomy id.", "arguments": {"standard_id": ["1a7aeab47932d0d4"], "page_size": 100}},
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
            description="List the article and section headings inside a single agreement so you can pick which section to fetch. Defaults to document_order sort, which preserves the original article/section sequence. Call after you have an agreement UUID, before get_section or get_section_snippet. For multiple agreements at once, use list_agreement_sections_batch.",
            input_schema=_schema_input_schema(McpListAgreementSectionsArgsSchema(), field_overrides=list_agreement_sections_overrides),
            output_schema=_list_agreement_sections_output_schema(),
            examples=(
                {"description": "List sections inside one agreement in document order.", "arguments": {"agreement_uuid": "a1", "page_size": 25}},
                {"description": "List sections alphabetically by title.", "arguments": {"agreement_uuid": "a1", "sort_by": "section_title", "page_size": 25}},
            ),
            response_examples=(
                {"description": "Section listing for an agreement.", "content": {"agreement_uuid": "a1", "returned_count": 2, "results": [{"section_uuid": "00000000-0000-0000-0000-000000000001"}]}},
            ),
            scopes=("sections:search",),
            selection_hint="Use after identifying one agreement and before calling get_section. Use list_agreement_sections_batch when you have multiple agreement UUIDs.",
            negative_guidance=(
                "Do not call this tool N times in a loop for N agreements; use list_agreement_sections_batch instead.",
            ),
            pagination="page",
            access_behavior="strict_scope_required",
            redaction_behavior="none",
            fulltext_scope=None,
            handler=_list_agreement_sections,
        ),
        McpToolSpec(
            name="list_agreement_sections_batch",
            description="List article and section headings for up to 20 agreements in a single call, returning one section list per agreement UUID. Eliminates the need for N sequential list_agreement_sections calls when comparing or exploring multiple agreements. Defaults to document_order sort.",
            input_schema=_schema_input_schema(McpBatchAgreementSectionsArgsSchema(), field_overrides=list_agreement_sections_overrides),
            output_schema=_batch_agreement_sections_output_schema(),
            examples=(
                {"description": "Batch-fetch sections for three agreements.", "arguments": {"agreement_uuids": ["a1", "a2", "a3"]}},
                {"description": "Batch-fetch with taxonomy filter.", "arguments": {"agreement_uuids": ["a1", "a2"], "standard_id": ["1a7aeab47932d0d4"]}},
            ),
            response_examples=(
                {"description": "Batch section listing.", "content": {"returned_agreement_count": 2, "total_section_count": 80, "results": [{"agreement_uuid": "a1", "section_count": 48, "sections": []}]}},
            ),
            scopes=("sections:search",),
            selection_hint="Use when you already have multiple agreement UUIDs and need all their section structures at once.",
            negative_guidance=(
                "Do not use for more than 20 agreements in one call; paginate with multiple batch calls instead.",
            ),
            pagination="none",
            access_behavior="strict_scope_required",
            redaction_behavior="none",
            fulltext_scope=None,
            handler=_list_agreement_sections_batch,
        ),
        McpToolSpec(
            name="get_agreement",
            description="Fetch one agreement by UUID, including its metadata (parties, counsel, consideration, deal status) and the agreement XML. XML is redacted to headings and structure unless the caller holds the full-text scope; metadata fields are always returned.",
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
            description="Fetch one section by UUID, returning its article/section titles, taxonomy assignments, and full XML content. Use after search_sections or list_agreement_sections; use get_section_snippet if you only need a short focused excerpt.",
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
            description="Fetch tax clauses extracted from one agreement, with their tax-taxonomy assignments. Use for focused tax-structure research; use get_agreement when you need the full agreement body.",
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
            description="Fetch tax clauses extracted from one section, with their tax-taxonomy assignments. Use when you have already isolated the right section and want just the tax-relevant clauses inside it.",
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
            description="List the valid values for every filter you can apply on search/list tools (counsels, industries, deal types, consideration, price buckets, PE flags, and more). The returned payload also includes retrieval_parameter_map, which maps each catalog key to the exact argument name used by the retrieval tools.",
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
            description="Translate a plain-English M&A concept (e.g. 'MAE carveouts', 'no-shop', 'reverse termination fee') into ranked clause-family taxonomy nodes. Each suggestion reports whether it is a canonical match, a proxy, or a broader semantic match, plus a confidence score. Call before search_sections when you know the concept but not the taxonomy.",
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
            description="Return a short, plain-text excerpt from one section, optionally centered on focus terms. Useful for quoting and side-by-side comparison when the full section XML would be too noisy. Fetch get_section instead when you need the complete, authoritative text.",
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
            name="get_section_snippets_batch",
            description="Return focused plain-text excerpts for up to 20 sections in a single call. Eliminates N sequential get_section_snippet calls when comparing clause language across multiple agreements. Accepts the same focus_terms and max_chars as get_section_snippet.",
            input_schema=_schema_input_schema(
                _schema_from_fields(
                    "McpBatchSectionSnippetsArgs",
                    {
                        "section_uuids": ma_fields.List(
                            ma_fields.Str(),
                            required=True,
                            validate=validate.Length(min=1, max=20),
                        ),
                        "focus_terms": ma_fields.List(ma_fields.Str(), load_default=[]),
                        "max_chars": ma_fields.Int(load_default=400, validate=validate.Range(min=120, max=1200)),
                    },
                )
            ),
            output_schema=_batch_section_snippet_output_schema(),
            examples=(
                {"description": "Fetch snippets for sections from multiple agreements at once.", "arguments": {"section_uuids": ["00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002"], "focus_terms": ["termination fee"], "max_chars": 400}},
            ),
            response_examples=(
                {"description": "Batch snippet result.", "content": {"returned_count": 2, "results": [{"section_uuid": "00000000-0000-0000-0000-000000000001", "snippet": "...termination fee of $50M...", "matched_terms": ["termination fee"]}]}},
            ),
            scopes=("sections:search",),
            selection_hint="Use when comparing clause language across many sections; replaces N calls to get_section_snippet.",
            negative_guidance=(
                "Do not call this tool N times in a loop for N sections — that defeats the purpose; pass all section_uuids at once.",
                "Do not treat snippets as complete section text; use get_section for the full XML when exact context matters.",
            ),
            pagination="none",
            access_behavior="strict_scope_required",
            redaction_behavior="none",
            fulltext_scope=None,
            handler=_get_section_snippets_batch,
        ),
        McpToolSpec(
            name="get_sections_batch",
            description="Fetch full XML content for up to 10 sections in a single call, with inline agreement metadata (target, acquirer, year, filing_date, transaction_price_total) and extracted monetary values. Eliminates N sequential get_section calls when you need the complete authoritative text for multiple sections. XML is capped at max_xml_chars per section (default 10000) to prevent context overload; xml_truncated signals when a result was cut. Pass max_xml_chars=null only if you specifically need uncapped XML. Use get_section_snippets_batch instead when only excerpts are needed.",
            input_schema=_schema_input_schema(
                _schema_from_fields(
                    "McpBatchSectionsArgs",
                    {
                        "section_uuids": ma_fields.List(
                            ma_fields.Str(),
                            required=True,
                            validate=validate.Length(min=1, max=10),
                        ),
                        "max_xml_chars": ma_fields.Int(load_default=10000, allow_none=True, validate=validate.Range(min=500, max=20000)),
                    },
                )
            ),
            output_schema=_batch_sections_output_schema(),
            examples=(
                {"description": "Fetch full XML for sections from multiple agreements at once.", "arguments": {"section_uuids": ["00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002"]}},
            ),
            response_examples=(
                {"description": "Batch full-section result.", "content": {"returned_count": 2, "results": [{"section_uuid": "00000000-0000-0000-0000-000000000001", "target": "Target A", "acquirer": "Acquirer B", "year": 2022, "filing_date": "2022-03-15", "transaction_price_total": 1500000000.0, "monetary_values": ["$1.5 billion"], "xml": "<section>...</section>"}]}},
            ),
            scopes=("sections:search",),
            selection_hint="Use when you need full section XML for multiple sections at once; replaces N calls to get_section.",
            negative_guidance=(
                "Do not use for snippets or excerpts — use get_section_snippets_batch instead.",
                "Limited to 10 sections per call; split into multiple calls for larger sets.",
            ),
            pagination="none",
            access_behavior="strict_scope_required",
            redaction_behavior="none",
            fulltext_scope=None,
            handler=_get_sections_batch,
        ),
        McpToolSpec(
            name="get_server_metrics",
            description="Operator-oriented MCP telemetry: per-tool call counts, latency buckets, error categories, and auth-failure counts since the server started. Not needed for research workflows.",
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
            description="Self-describing guide to this server: the tool inventory with selection hints and negative guidance, the field inventory (which concepts are first-class, taxonomy-backed, text-derived, or unrepresented), ready-made research workflows, and auth help. Call first when onboarding a new agent or planning a multi-step research task. Use `sections` to request only the parts you need — omit `tools` unless you specifically want full per-tool schemas (those are large; use the MCP tools/list endpoint instead).",
            input_schema=_object_schema(
                {
                    "sections": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(_CAPABILITIES_SECTIONS_ALL)},
                        "description": (
                            "Subset of capability sections to return. "
                            f"Valid values: {', '.join(_CAPABILITIES_SECTIONS_ALL)}. "
                            "Defaults to all sections except `tools` (which is large). "
                            "Pass `[\"tools\"]` or include `tools` explicitly if you need full per-tool schemas."
                        ),
                        "default": list(_CAPABILITIES_SECTIONS_DEFAULT),
                    }
                },
                required=[],
                additional_properties=False,
            ),
            output_schema=_server_capabilities_output_schema(),
            examples=(
                {"description": "Inspect tool guidance and workflows (default — omits full tool schemas).", "arguments": {}},
                {"description": "Fetch only the concept notes and field inventory.", "arguments": {"sections": ["concept_notes", "field_inventory"]}},
                {"description": "Fetch full per-tool schemas (large response).", "arguments": {"sections": ["tools"]}},
            ),
            response_examples=(
                {"description": "Capabilities payload with default sections.", "content": {"server": {"name": "pandects-mcp", "transport": "http_jsonrpc"}, "workflows": [{"name": "discover agreements by counsel"}], "sections_returned": ["server", "auth_help", "field_inventory", "concept_notes", "tool_limitations", "workflows"]}},
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
            description="Return the full clause-family taxonomy tree (L1 → L2 → L3) used for section-level search. Prefer suggest_clause_families when you want to look up a single concept; use this tool when you need the whole tree for browsing or planning.",
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
            description="Return the full tax-clause taxonomy tree used for tax-structure research. Pair with get_agreement_tax_clauses or get_section_tax_clauses once you identify the right tax taxonomy nodes.",
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
            description=(
                "Return canonical law-firm names used as counsel filters. "
                "Pass `query` to filter by substring (case-insensitive) — e.g. query='Kirkland' returns only Kirkland & Ellis entries. "
                "Without `query` the full catalog (~1 300 entries) is returned; prefer a scoped query whenever you know the firm name."
            ),
            input_schema=_object_schema(
                {
                    "query": {
                        "type": "string",
                        "description": "Case-insensitive substring filter on canonical_name. Omit to return all entries.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of entries to return. Omit for no limit.",
                    },
                },
                required=[],
                additional_properties=False,
            ),
            output_schema=_counsel_catalog_output_schema(),
            examples=(
                {"description": "Look up Kirkland & Ellis by partial name.", "arguments": {"query": "Kirkland"}},
                {"description": "Find all Sullivan & Cromwell entries.", "arguments": {"query": "Sullivan"}},
                {"description": "Return the full catalog (use sparingly — large response).", "arguments": {}},
            ),
            response_examples=(
                {"description": "Filtered result for query='Kirkland'.", "content": {"counsel": [{"counsel_id": 42, "canonical_name": "Kirkland & Ellis"}]}},
            ),
            scopes=("sections:search",),
            selection_hint="Use when you need canonical firm names before counsel-filtered agreement or section retrieval. Pass query= to avoid fetching the full catalog.",
            negative_guidance=(),
            pagination="none",
            access_behavior="strict_scope_required",
            redaction_behavior="none",
            fulltext_scope=None,
            handler=_get_counsel_catalog,
        ),
        McpToolSpec(
            name="get_naics_catalog",
            description="Return the NAICS industry hierarchy (sectors and subsectors) used to normalize target_industry and acquirer_industry filters. Use when translating an industry description into the canonical label the filters accept.",
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
            description="Corpus sizing: total agreements, sections, and page counts. Use to size the dataset before planning a survey or estimating how much of the corpus a filter covers.",
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
            description="Pre-aggregated year-over-year trends for the corpus: ownership mix (public/private/PE), buyer-type mix, and target/acquirer industry distributions. Use for quick macro context; use list_agreements for row-level breakdowns.",
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


def _server_capabilities_payload(sections: frozenset[str] | None = None) -> dict[str, object]:
    if sections is None:
        sections = frozenset(_CAPABILITIES_SECTIONS_DEFAULT)
    specs = _tool_specs() if (sections & {"tools", "tool_limitations"}) else ()
    sections_returned = [section for section in _CAPABILITIES_SECTIONS_ALL if section in sections]
    result: dict[str, object] = {"sections_returned": sections_returned}
    if "server" in sections:
        result["server"] = {
            "name": "pandects-mcp",
            "primary_discovery_tool": "list_filter_options",
            "introspection_tool": "get_server_capabilities",
            "metrics_tool": "get_server_metrics",
            "transport": "http_jsonrpc",
            "resources_supported": False,
            "resource_templates_supported": False,
        }
    if "auth_help" in sections:
        result["auth_help"] = {
            "login_required": True,
            "relogin_hint": "If a bearer token is missing, invalid, or expired, re-authenticate and retry the MCP call.",
            "fulltext_scope": "agreements:read_fulltext",
            "invalid_or_expired_token_message": "If credentials look stale, sign in again and resend the bearer token.",
        }
    if "field_inventory" in sections:
        result["field_inventory"] = _field_inventory_payload()
    if "concept_notes" in sections:
        result["concept_notes"] = _concept_notes_payload()
    if "tool_limitations" in sections:
        result["tool_limitations"] = _tool_limitations_payload(specs)
    if "tools" in sections:
        result["tools"] = [
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
        ]
    if "workflows" in sections:
        result["workflows"] = [
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
        ]
    return result


def _get_server_capabilities(
    *,
    principal: McpPrincipal,
    payload: dict[str, object],
) -> McpToolResult:
    if not principal.scopes:
        raise PermissionError("Missing required scope: agreements:search")
    raw_sections = payload.get("sections")
    if isinstance(raw_sections, list) and all(isinstance(s, str) for s in raw_sections):
        sections = frozenset(cast(list[str], raw_sections)) & frozenset(_CAPABILITIES_SECTIONS_ALL)
    else:
        sections = frozenset(_CAPABILITIES_SECTIONS_DEFAULT)
    response = _server_capabilities_payload(sections)
    returned = cast(list[object], response["sections_returned"])
    return McpToolResult(
        text=f"Returned MCP server capabilities: {len(returned)} section(s) ({', '.join(str(s) for s in returned)}).",
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
    if name in {"search_agreements", "list_agreements", "get_agreement", "get_section", "get_section_snippet", "get_section_snippets_batch", "get_sections_batch", "get_agreement_tax_clauses", "get_section_tax_clauses", "list_filter_options", "get_agreements_summary"}:
        handler_kwargs["deps"] = agreements_deps
    if name in {"search_sections", "list_agreement_sections", "list_agreement_sections_batch"}:
        handler_kwargs["deps"] = sections_service_deps
    if name == "search_sections":
        handler_kwargs["agreements_deps"] = agreements_deps
    if name in {"get_clause_taxonomy", "get_tax_clause_taxonomy", "get_counsel_catalog", "get_naics_catalog", "suggest_clause_families"}:
        handler_kwargs["deps"] = reference_data_deps
    if name in {"search_agreements", "search_sections", "list_agreements", "list_agreement_sections", "list_agreement_sections_batch", "get_agreement", "get_section", "get_section_snippet", "get_section_snippets_batch", "get_sections_batch", "get_agreement_tax_clauses", "get_section_tax_clauses", "list_filter_options", "suggest_clause_families", "get_counsel_catalog", "get_server_capabilities"}:
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
