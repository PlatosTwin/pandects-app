from __future__ import annotations

from typing import Any, cast

from marshmallow import Schema, fields as ma_fields, validate

from backend.mcp.tools.constants import (
    _FILTER_OPTIONS_FIELDS,
    _SECTION_LIST_SORT_FIELDS,
    _TRENDS_SECTIONS_ALL,
    _TRENDS_SECTIONS_DEFAULT,
)
from backend.schemas.public_api import AgreementArgsSchema


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


class McpAgreementIdentifierSchema(Schema):
    agreement_uuid = ma_fields.Str(required=True)


class McpSectionArgsSchema(Schema):
    section_uuid = ma_fields.Str(required=True)


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


class McpSearchAgreementsExtraArgsSchema(Schema):
    any_counsel = ma_fields.List(ma_fields.Str(), load_default=[])
    year_min = ma_fields.Int(load_default=None, allow_none=True, validate=validate.Range(min=1900, max=2100))
    year_max = ma_fields.Int(load_default=None, allow_none=True, validate=validate.Range(min=1900, max=2100))
    filed_after = ma_fields.Str(load_default=None, allow_none=True, validate=validate.Regexp(r'^\d{4}-\d{2}-\d{2}$'))
    filed_before = ma_fields.Str(load_default=None, allow_none=True, validate=validate.Regexp(r'^\d{4}-\d{2}-\d{2}$'))


class McpFilterOptionsArgsSchema(Schema):
    fields = cast(
        Any,
        ma_fields.List(
            ma_fields.Str(validate=validate.OneOf(list(_FILTER_OPTIONS_FIELDS))),
            load_default=[],
        ),
    )
