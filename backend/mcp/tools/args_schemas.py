from __future__ import annotations

from typing import Any, cast

from marshmallow import Schema, fields as ma_fields, validate

from backend.mcp.tools.constants import (
    _FEEDBACK_CATEGORY_VALUES,
    _FEEDBACK_SEVERITY_VALUES,
    _FILTER_OPTIONS_FIELDS,
    _LIST_AGREEMENTS_ROW_FIELDS,
    _LIST_AGREEMENTS_SORT_FIELDS,
    _SEARCH_AGREEMENTS_ROW_FIELDS,
    _SECTION_LIST_SORT_FIELDS,
    _TRENDS_SECTIONS_ALL,
    _TRENDS_SECTIONS_DEFAULT,
)
from backend.schemas.public_api import AgreementArgsSchema
from backend.schemas.sections import SectionsArgsSchema


class McpSectionsArgsSchema(SectionsArgsSchema):
    """search_sections arguments plus MCP-only inline-snippet controls.

    The snippet options are deliberately not on the shared web schema: they exist to
    collapse the search -> get_section_snippets_batch round-trip that every clause
    comparison would otherwise pay, and the section text is already loaded by the
    service, so returning an excerpt costs no additional query.
    """

    include_snippet = ma_fields.Bool(
        load_default=False,
        metadata={
            "description": (
                "When true, include a short plain-text excerpt of each section as "
                "`snippet`, plus `matched_terms`, `monetary_values`, and `source_length`. "
                "Lets one search answer 'what does this clause say' without a follow-up "
                "get_section_snippets_batch call. Far smaller than include_xml."
            ),
        },
    )
    snippet_focus_terms = ma_fields.List(
        ma_fields.Str(),
        load_default=list,
        metadata={
            "description": (
                "Terms used to centre each snippet, as in get_section_snippet. Only "
                "applies when include_snippet is true."
            ),
        },
    )
    snippet_max_chars = ma_fields.Int(
        load_default=400,
        validate=validate.Range(min=120, max=1200),
        metadata={
            "description": (
                "Maximum characters per snippet (120-1200). Only applies when "
                "include_snippet is true. Defaults to 400 to keep result pages compact."
            ),
        },
    )


class McpAgreementArgsSchema(AgreementArgsSchema):
    agreement_uuid = ma_fields.Str(required=True)
    include_xml = ma_fields.Bool(
        load_default=False,
        metadata={
            "description": (
                "When true, include the agreement XML (redacted to headings/structure "
                "unless the caller holds the agreements:read_fulltext scope). Defaults to "
                "false: metadata only, which keeps the response small. focus_section_uuid "
                "and neighbor_sections only take effect when include_xml is true and the "
                "response is redacted."
            ),
        },
    )


class McpAgreementTrendsArgsSchema(Schema):
    sections = ma_fields.List(
        ma_fields.Str(validate=validate.OneOf(list(_TRENDS_SECTIONS_ALL))),
        load_default=list(_TRENDS_SECTIONS_DEFAULT),
        metadata={
            "description": (
                "Subset of trend sections to return. Valid values: ownership, "
                "target_industries, pairings, naics_catalog. Defaults to "
                "[ownership, target_industries]. pairings (every industry-by-industry "
                "cell) and naics_catalog (the full NAICS hierarchy) are large; request "
                "them explicitly, or call get_naics_catalog for the catalog."
            ),
        },
    )
    year_min = ma_fields.Int(
        load_default=None,
        allow_none=True,
        validate=validate.Range(min=1900, max=2100),
        metadata={
            "description": (
                "Earliest year to include in the per-year trend arrays (inclusive). "
                "Filters the year-keyed sections (ownership mix/deal size, "
                "target_industries); the buyer-type matrix aggregates across all years "
                "and is unaffected."
            ),
        },
    )
    year_max = ma_fields.Int(
        load_default=None,
        allow_none=True,
        validate=validate.Range(min=1900, max=2100),
        metadata={
            "description": "Latest year to include in the per-year trend arrays (inclusive).",
        },
    )


class McpAgreementIdentifierSchema(Schema):
    agreement_uuid = ma_fields.Str(required=True)


class McpSectionArgsSchema(Schema):
    section_uuid = ma_fields.Str(required=True)


_TAX_CLAUSE_STANDARD_ID_FIELD_KWARGS: dict[str, Any] = {
    "load_default": [],
    "metadata": {
        "description": (
            "Tax clause taxonomy standard IDs (dotted tax.* ids from get_tax_clause_taxonomy "
            "or suggest_clause_families with taxonomy='tax_clauses'). Parent IDs expand to "
            "descendants. Omit to return every extracted tax-module clause, most of which "
            "carry no taxonomy assignment."
        ),
    },
}


class McpAgreementTaxClausesArgsSchema(Schema):
    agreement_uuid = ma_fields.Str(required=True)
    tax_standard_id = ma_fields.List(ma_fields.Str(), **_TAX_CLAUSE_STANDARD_ID_FIELD_KWARGS)
    page = ma_fields.Int(load_default=1, validate=validate.Range(min=1))
    page_size = ma_fields.Int(load_default=25, validate=validate.Range(min=1, max=200))


class McpSectionTaxClausesArgsSchema(Schema):
    section_uuid = ma_fields.Str(required=True)
    tax_standard_id = ma_fields.List(ma_fields.Str(), **_TAX_CLAUSE_STANDARD_ID_FIELD_KWARGS)
    page = ma_fields.Int(load_default=1, validate=validate.Range(min=1))
    page_size = ma_fields.Int(load_default=25, validate=validate.Range(min=1, max=200))


class McpListAgreementSectionsArgsSchema(Schema):
    agreement_uuid = ma_fields.Str(required=True)
    standard_id = ma_fields.List(ma_fields.Str(), load_default=[])
    include_standard_ids = ma_fields.Bool(load_default=True)
    page = ma_fields.Int(load_default=1)
    page_size = ma_fields.Int(load_default=25)
    sort_by = ma_fields.Str(
        load_default="document_order",
        validate=validate.OneOf(list(_SECTION_LIST_SORT_FIELDS)),
    )
    sort_direction = ma_fields.Str(load_default="asc", validate=validate.OneOf(["asc", "desc"]))


class McpBatchAgreementSectionsArgsSchema(Schema):
    agreement_uuids = ma_fields.List(
        ma_fields.Str(),
        required=True,
        validate=validate.Length(min=1, max=20),
    )
    standard_id = ma_fields.List(ma_fields.Str(), load_default=[])
    include_standard_ids = ma_fields.Bool(load_default=True)
    sort_by = ma_fields.Str(
        load_default="document_order",
        validate=validate.OneOf(list(_SECTION_LIST_SORT_FIELDS)),
    )
    sort_direction = ma_fields.Str(load_default="asc", validate=validate.OneOf(["asc", "desc"]))
    max_sections_per_agreement = ma_fields.Int(
        load_default=200,
        validate=validate.Range(min=1, max=1000),
    )


class McpSearchAgreementsExtraArgsSchema(Schema):
    any_counsel = ma_fields.List(ma_fields.Str(), load_default=[])
    year_min = ma_fields.Int(load_default=None, allow_none=True, validate=validate.Range(min=1900, max=2100))
    year_max = ma_fields.Int(load_default=None, allow_none=True, validate=validate.Range(min=1900, max=2100))
    filed_after = ma_fields.Str(load_default=None, allow_none=True, validate=validate.Regexp(r'^\d{4}-\d{2}-\d{2}$'))
    filed_before = ma_fields.Str(load_default=None, allow_none=True, validate=validate.Regexp(r'^\d{4}-\d{2}-\d{2}$'))


_SEARCH_AGREEMENTS_FIELDS_KWARGS: dict[str, Any] = {
    "load_default": [],
    "metadata": {
        "description": (
            "Restrict each result row to only these keys (agreement_uuid is always "
            "included regardless). Omit for the full row. Use for bulk export or "
            "corpus-scale scans to keep responses small, e.g. "
            "fields=[\"agreement_uuid\", \"url\"] to fetch EDGAR URLs for out-of-band "
            "full-text analysis."
        ),
        "examples": [["agreement_uuid", "url"]],
    },
}


class McpSearchAgreementsFieldsArgsSchema(Schema):
    fields = cast(
        Any,
        ma_fields.List(
            ma_fields.Str(validate=validate.OneOf(list(_SEARCH_AGREEMENTS_ROW_FIELDS))),
            **_SEARCH_AGREEMENTS_FIELDS_KWARGS,
        ),
    )


class McpListAgreementsExtraArgsSchema(Schema):
    fields = cast(
        Any,
        ma_fields.List(
            ma_fields.Str(validate=validate.OneOf(list(_LIST_AGREEMENTS_ROW_FIELDS))),
            **_SEARCH_AGREEMENTS_FIELDS_KWARGS,
        ),
    )
    sort_by = ma_fields.Str(
        load_default="agreement_uuid",
        validate=validate.OneOf(list(_LIST_AGREEMENTS_SORT_FIELDS)),
        metadata={
            "description": (
                "Sort key for cursor pagination. One of: agreement_uuid (default, the "
                "original behavior), year, target, acquirer, filing_date. Changing "
                "sort_by or sort_dir invalidates any cursor from a previous call with "
                "different sort settings -- restart pagination without a cursor."
            ),
        },
    )
    sort_dir = ma_fields.Str(
        load_default="asc",
        validate=validate.OneOf(["asc", "desc"]),
        metadata={"description": "Sort direction for sort_by. One of: asc, desc."},
    )


class McpBatchAgreementsArgsSchema(Schema):
    agreement_uuids = ma_fields.List(
        ma_fields.Str(),
        required=True,
        validate=validate.Length(min=1, max=25),
        metadata={
            "description": "Known agreement UUIDs to fetch metadata for (1-25 per call).",
        },
    )


class McpSubmitFeedbackArgsSchema(Schema):
    summary = ma_fields.Str(
        required=True,
        validate=validate.Length(min=1, max=200),
        metadata={
            "description": "One-line gist of the feedback (max 200 chars). Must contain non-whitespace text.",
        },
    )
    detail = ma_fields.Str(
        required=True,
        validate=validate.Length(min=1, max=5000),
        metadata={
            "description": (
                "Full description (max 5000 chars). Most useful shape: what you were "
                "trying to do, the tool and arguments involved, what you expected, and "
                "what actually happened."
            ),
        },
    )
    category = ma_fields.Str(
        load_default="other",
        validate=validate.OneOf(list(_FEEDBACK_CATEGORY_VALUES)),
        metadata={
            "description": f"Feedback category. One of: {', '.join(_FEEDBACK_CATEGORY_VALUES)}. Defaults to `other`.",
        },
    )
    severity = ma_fields.Str(
        load_default=None,
        allow_none=True,
        validate=validate.OneOf(list(_FEEDBACK_SEVERITY_VALUES)),
        metadata={
            "description": "Optional impact estimate: low, medium, or high. Omit when unsure.",
        },
    )
    tool_name = ma_fields.Str(
        load_default=None,
        allow_none=True,
        validate=validate.Length(min=1, max=128),
        metadata={
            "description": "MCP tool the feedback concerns, when it concerns one specific tool. Must be a real tool name from this server.",
        },
    )
    suggestions = ma_fields.Str(
        load_default=None,
        allow_none=True,
        validate=validate.Length(min=1, max=2000),
        metadata={
            "description": "Optional concrete improvement suggestion (max 2000 chars).",
        },
    )
    context = ma_fields.Dict(
        keys=ma_fields.Str(),
        load_default=None,
        allow_none=True,
        metadata={
            "description": (
                "Optional small JSON object with reproduction context — typically the "
                "arguments that triggered the issue. Capped at 2000 characters when "
                "serialized; do not include document text or user data."
            ),
        },
    )


class McpFilterOptionsArgsSchema(Schema):
    fields = cast(
        Any,
        ma_fields.List(
            ma_fields.Str(validate=validate.OneOf(list(_FILTER_OPTIONS_FIELDS))),
            load_default=[],
        ),
    )
