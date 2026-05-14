from __future__ import annotations

from typing import TypedDict, cast

from marshmallow import Schema, fields, validate

from backend.schemas.sections import AccessInfoSchema, DumpVersionInfoSchema


class AgreementArgsSchema(Schema):
    focus_section_uuid = fields.Str(
        required=False,
        allow_none=True,
        metadata={
            "description": (
                "Optional section UUID used when redacting anonymous responses to keep a "
                "focused neighborhood visible."
            ),
            "example": "5e59453aaa9255c4",
        },
    )
    neighbor_sections = fields.Int(
        load_default=1,
        metadata={
            "description": (
                "Number of neighboring sections to include around `focus_section_uuid` when "
                "response XML is redacted."
            ),
            "example": 1,
        },
    )


class AgreementsBulkArgsSchema(Schema):
    cursor = fields.Str(
        load_default=None,
        allow_none=True,
        metadata={"description": "Opaque cursor from a previous agreements listing response."},
    )
    page_size = fields.Int(
        load_default=25,
        metadata={"description": "Maximum number of agreements to return.", "example": 25},
    )
    include_xml = fields.Bool(
        load_default=False,
        metadata={"description": "When true, include full agreement XML for each result."},
    )
    year = fields.List(fields.Int(), load_default=[])
    target = fields.List(fields.Str(), load_default=[])
    acquirer = fields.List(fields.Str(), load_default=[])
    transaction_price_total = fields.List(fields.Str(), load_default=[])
    transaction_price_stock = fields.List(fields.Str(), load_default=[])
    transaction_price_cash = fields.List(fields.Str(), load_default=[])
    transaction_price_assets = fields.List(fields.Str(), load_default=[])
    transaction_consideration = fields.List(fields.Str(), load_default=[])
    target_type = fields.List(fields.Str(), load_default=[])
    acquirer_type = fields.List(fields.Str(), load_default=[])
    target_counsel = fields.List(fields.Str(), load_default=[])
    acquirer_counsel = fields.List(fields.Str(), load_default=[])
    target_industry = fields.List(fields.Str(), load_default=[])
    acquirer_industry = fields.List(fields.Str(), load_default=[])
    deal_status = fields.List(fields.Str(), load_default=[])
    attitude = fields.List(fields.Str(), load_default=[])
    deal_type = fields.List(fields.Str(), load_default=[])
    purpose = fields.List(fields.Str(), load_default=[])
    target_pe = fields.List(fields.Str(), load_default=[])
    acquirer_pe = fields.List(fields.Str(), load_default=[])
    agreement_uuid = fields.Str(load_default=None, allow_none=True)
    section_uuid = fields.Str(load_default=None, allow_none=True)
    include_dump = fields.Bool(
        load_default=True,
        metadata={
            "description": (
                "When false, omit `dump_version` from the response body. "
                "The `X-Pandects-Dump-Hash` response header is always included."
            ),
        },
    )
    standard_id = fields.List(
        fields.Str(),
        load_default=[],
        metadata={"description": "Filter to agreements that contain at least one section tagged with any of these taxonomy standard_ids. Accepts the same ids as search_sections."},
    )


class AgreementsIndexArgsSchema(Schema):
    page = fields.Int(load_default=1, metadata={"description": "1-based page number.", "example": 1})
    page_size = fields.Int(
        load_default=25,
        metadata={"description": "Maximum number of agreements to return.", "example": 25},
    )
    sort_by = fields.Str(
        load_default="year",
        validate=validate.OneOf(["year", "target", "acquirer"]),
        metadata={"description": "Sort key. One of: `year`, `target`, `acquirer`."},
    )
    sort_dir = fields.Str(
        load_default="desc",
        validate=validate.OneOf(["asc", "desc"]),
        metadata={"description": "Sort direction. One of: `asc`, `desc`."},
    )
    query = fields.Str(
        load_default="",
        metadata={
            "description": "Optional prefix search over target and acquirer names, or a 4-digit year string.",
            "example": "Slack",
        },
    )


class AgreementArgsPayload(TypedDict):
    focus_section_uuid: str | None
    neighbor_sections: int


class AgreementsBulkArgsPayload(TypedDict):
    cursor: str | None
    page_size: int
    include_xml: bool
    year: list[int]
    target: list[str]
    acquirer: list[str]
    transaction_price_total: list[str]
    transaction_price_stock: list[str]
    transaction_price_cash: list[str]
    transaction_price_assets: list[str]
    transaction_consideration: list[str]
    target_type: list[str]
    acquirer_type: list[str]
    target_counsel: list[str]
    acquirer_counsel: list[str]
    target_industry: list[str]
    acquirer_industry: list[str]
    deal_status: list[str]
    attitude: list[str]
    deal_type: list[str]
    purpose: list[str]
    target_pe: list[str]
    acquirer_pe: list[str]
    agreement_uuid: str | None
    section_uuid: str | None
    include_dump: bool
    standard_id: list[str]


class AgreementResponseSchema(Schema):
    year = fields.Int()
    target = fields.Str()
    acquirer = fields.Str()
    filing_date = fields.Str(allow_none=True)
    prob_filing = fields.Float(allow_none=True)
    filing_company_name = fields.Str(allow_none=True)
    filing_company_cik = fields.Str(allow_none=True)
    form_type = fields.Str(allow_none=True)
    exhibit_type = fields.Str(allow_none=True)
    transaction_price_total = fields.Float(allow_none=True)
    transaction_price_stock = fields.Float(allow_none=True)
    transaction_price_cash = fields.Float(allow_none=True)
    transaction_price_assets = fields.Float(allow_none=True)
    transaction_consideration = fields.Str(allow_none=True)
    target_type = fields.Str(allow_none=True)
    acquirer_type = fields.Str(allow_none=True)
    target_industry = fields.Str(allow_none=True)
    acquirer_industry = fields.Str(allow_none=True)
    announce_date = fields.Str(allow_none=True)
    close_date = fields.Str(allow_none=True)
    deal_status = fields.Str(allow_none=True)
    attitude = fields.Str(allow_none=True)
    deal_type = fields.Str(allow_none=True)
    purpose = fields.Str(allow_none=True)
    target_pe = fields.Bool(allow_none=True)
    acquirer_pe = fields.Bool(allow_none=True)
    url = fields.Str()
    xml = fields.Str()
    is_redacted = fields.Bool(required=False)


class AgreementListItemSchema(Schema):
    agreement_uuid = fields.Str()
    year = fields.Int(allow_none=True)
    target = fields.Str(allow_none=True)
    acquirer = fields.Str(allow_none=True)
    filing_date = fields.Str(allow_none=True)
    prob_filing = fields.Float(allow_none=True)
    filing_company_name = fields.Str(allow_none=True)
    filing_company_cik = fields.Str(allow_none=True)
    form_type = fields.Str(allow_none=True)
    exhibit_type = fields.Str(allow_none=True)
    transaction_price_total = fields.Float(allow_none=True)
    transaction_price_stock = fields.Float(allow_none=True)
    transaction_price_cash = fields.Float(allow_none=True)
    transaction_price_assets = fields.Float(allow_none=True)
    transaction_consideration = fields.Str(allow_none=True)
    target_type = fields.Str(allow_none=True)
    acquirer_type = fields.Str(allow_none=True)
    target_industry = fields.Str(allow_none=True)
    acquirer_industry = fields.Str(allow_none=True)
    announce_date = fields.Str(allow_none=True)
    close_date = fields.Str(allow_none=True)
    deal_status = fields.Str(allow_none=True)
    attitude = fields.Str(allow_none=True)
    deal_type = fields.Str(allow_none=True)
    purpose = fields.Str(allow_none=True)
    target_pe = fields.Bool(allow_none=True)
    acquirer_pe = fields.Bool(allow_none=True)
    url = fields.Str(allow_none=True)
    xml = fields.Str(allow_none=True, required=False)


class AgreementsListResponseSchema(Schema):
    results: object = cast(
        object,
        fields.List(
            fields.Nested(AgreementListItemSchema),
        ),
    )
    access = fields.Nested(AccessInfoSchema)
    page_size = fields.Int()
    returned_count = fields.Int()
    has_next = fields.Bool()
    next_cursor = fields.Str(allow_none=True)
    dump_version = fields.Nested(DumpVersionInfoSchema, allow_none=True)


class AgreementSearchMatchedSectionSchema(Schema):
    section_uuid = fields.Str(required=True)
    article_title = fields.Str(allow_none=True)
    section_title = fields.Str(allow_none=True)
    standard_id = fields.List(fields.Str(), required=True)
    snippet = fields.Str(allow_none=True)


class AgreementSearchResultSchema(Schema):
    agreement_uuid = fields.Str(required=True)
    year = fields.Int(allow_none=True)
    target = fields.Str(allow_none=True)
    acquirer = fields.Str(allow_none=True)
    filing_date = fields.Str(allow_none=True)
    prob_filing = fields.Float(allow_none=True)
    filing_company_name = fields.Str(allow_none=True)
    filing_company_cik = fields.Str(allow_none=True)
    form_type = fields.Str(allow_none=True)
    exhibit_type = fields.Str(allow_none=True)
    transaction_price_total = fields.Float(allow_none=True)
    transaction_price_stock = fields.Float(allow_none=True)
    transaction_price_cash = fields.Float(allow_none=True)
    transaction_price_assets = fields.Float(allow_none=True)
    transaction_consideration = fields.Str(allow_none=True)
    target_type = fields.Str(allow_none=True)
    acquirer_type = fields.Str(allow_none=True)
    target_industry = fields.Str(allow_none=True)
    acquirer_industry = fields.Str(allow_none=True)
    announce_date = fields.Str(allow_none=True)
    close_date = fields.Str(allow_none=True)
    deal_status = fields.Str(allow_none=True)
    attitude = fields.Str(allow_none=True)
    deal_type = fields.Str(allow_none=True)
    purpose = fields.Str(allow_none=True)
    target_pe = fields.Bool(allow_none=True)
    acquirer_pe = fields.Bool(allow_none=True)
    url = fields.Str(allow_none=True)
    match_count = fields.Int(required=True)
    matched_sections = fields.List(
        fields.Nested(AgreementSearchMatchedSectionSchema),
        required=True,
    )


class AgreementSearchResponseSchema(Schema):
    results: object = cast(
        object,
        fields.List(
            fields.Nested(AgreementSearchResultSchema),
        ),
    )
    access = fields.Nested(AccessInfoSchema)
    page = fields.Int(required=True)
    page_size = fields.Int(required=True)
    total_count = fields.Int(required=True)
    total_count_is_approximate = fields.Bool(required=True)
    count_metadata = fields.Dict()
    total_pages = fields.Int(required=True)
    has_next = fields.Bool(required=True)
    has_prev = fields.Bool(required=True)
    next_num = fields.Int(allow_none=True)
    prev_num = fields.Int(allow_none=True)
    dump_version = fields.Nested(DumpVersionInfoSchema, allow_none=True)


class SectionResponseSchema(Schema):
    agreement_uuid = fields.Str()
    section_uuid = fields.Str()
    section_standard_id = fields.List(fields.Str())
    xml = fields.Str()
    article_title = fields.Str()
    section_title = fields.Str()


class TaxClauseSchema(Schema):
    clause_uuid = fields.Str()
    agreement_uuid = fields.Str()
    section_uuid = fields.Str()
    article_title = fields.Str(allow_none=True)
    section_title = fields.Str(allow_none=True)
    anchor_label = fields.Str(allow_none=True)
    start_char = fields.Int()
    end_char = fields.Int()
    clause_text = fields.Str()
    context_type = fields.Str()
    standard_ids = fields.List(fields.Str())


class TaxClauseListResponseSchema(Schema):
    clauses: object = cast(
        object,
        fields.List(
            fields.Nested(TaxClauseSchema),
            required=True,
        ),
    )


class AgreementSectionIndexItemSchema(Schema):
    section_uuid = fields.Str(required=True)
    article_title = fields.Str(allow_none=True)
    section_title = fields.Str(allow_none=True)
    article_order = fields.Int(allow_none=True)
    section_order = fields.Int(allow_none=True)
    standard_id = fields.List(fields.Str(), required=True)


class AgreementSectionIndexResponseSchema(Schema):
    agreement_uuid = fields.Str(required=True)
    results = fields.List(
        fields.Nested(AgreementSectionIndexItemSchema),
        required=True,
    )


class DumpEntrySchema(Schema):
    timestamp = fields.Str(required=True)
    sql = fields.Url(required=False, allow_none=True)
    sha256 = fields.Str(required=False, allow_none=True)
    sha256_url = fields.Url(required=False, allow_none=True)
    manifest = fields.Url(required=False, allow_none=True)
    size_bytes = fields.Int(required=False, allow_none=True)
    warning = fields.Str(required=False, allow_none=True)


class NaicsSubSectorSchema(Schema):
    sub_sector_code = fields.Str(required=True)
    sub_sector_desc = fields.Str(required=True)


class NaicsSectorSchema(Schema):
    sector_code = fields.Str(required=True)
    sector_desc = fields.Str(required=True)
    sector_group = fields.Str(required=True)
    super_sector = fields.Str(required=True)
    sub_sectors: object = cast(
        object,
        fields.List(
            fields.Nested(NaicsSubSectorSchema),
            required=True,
        ),
    )


class NaicsResponseSchema(Schema):
    sectors: object = cast(
        object,
        fields.List(
            fields.Nested(NaicsSectorSchema),
            required=True,
        ),
    )


class CounselEntrySchema(Schema):
    counsel_id = fields.Int(required=True)
    canonical_name = fields.Str(required=True)


class CounselResponseSchema(Schema):
    counsel: object = cast(
        object,
        fields.List(
            fields.Nested(CounselEntrySchema),
            required=True,
        ),
    )


__all__ = [
    "AgreementArgsPayload",
    "AgreementArgsSchema",
    "AgreementListItemSchema",
    "AgreementResponseSchema",
    "AgreementsBulkArgsPayload",
    "AgreementsBulkArgsSchema",
    "AgreementsIndexArgsSchema",
    "AgreementsListResponseSchema",
    "CounselEntrySchema",
    "CounselResponseSchema",
    "DumpEntrySchema",
    "DumpVersionInfoSchema",
    "NaicsResponseSchema",
    "NaicsSectorSchema",
    "NaicsSubSectorSchema",
    "SectionResponseSchema",
    "TaxClauseListResponseSchema",
    "TaxClauseSchema",
]
