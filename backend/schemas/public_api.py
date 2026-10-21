from __future__ import annotations

from typing import TypedDict, cast

from marshmallow import Schema, fields, validate

from backend.schemas.labels import (
    IndustryLabelsSchema,
    StandardIdLabelsSchema,
    TaxClauseStandardIdLabelsSchema,
    pe_flag_field,
)
from backend.schemas.sections import (
    AccessInfoSchema,
    DumpVersionInfoSchema,
    price_bucket_list_field,
)


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
    year_min = fields.Int(
        load_default=None,
        allow_none=True,
        validate=validate.Range(min=1900, max=2100),
        metadata={"description": "Earliest filing year to include (inclusive)."},
    )
    year_max = fields.Int(
        load_default=None,
        allow_none=True,
        validate=validate.Range(min=1900, max=2100),
        metadata={"description": "Latest filing year to include (inclusive)."},
    )
    filed_after = fields.Str(
        load_default=None,
        allow_none=True,
        validate=validate.Regexp(r"^\d{4}-\d{2}-\d{2}$"),
        metadata={"description": "Only agreements filed on or after this date (YYYY-MM-DD)."},
    )
    filed_before = fields.Str(
        load_default=None,
        allow_none=True,
        validate=validate.Regexp(r"^\d{4}-\d{2}-\d{2}$"),
        metadata={"description": "Only agreements filed before this date (YYYY-MM-DD, exclusive)."},
    )
    target = fields.List(fields.Str(), load_default=[])
    acquirer = fields.List(fields.Str(), load_default=[])
    transaction_price_total = price_bucket_list_field()
    transaction_price_stock = price_bucket_list_field()
    transaction_price_cash = price_bucket_list_field()
    transaction_price_assets = price_bucket_list_field()
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
        metadata={
            "description": "Filter to agreements that contain at least one section tagged with any of these taxonomy standard_ids. Accepts the same ids as search_sections: opaque 16-character hex node IDs from get_clause_taxonomy or suggest_clause_families (not dotted numbers). Unrecognized IDs are ignored and reported under interpretation.unrecognized_standard_ids.",
            "example": ["4207f2e8f6698935"],
        },
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


class AgreementsIndexQueryArgsSchema(AgreementsIndexArgsSchema):
    """`/v1/agreements-index` arguments.

    Accepts `sort_direction` like every other paginated endpoint; `sort_dir`, the name
    this endpoint originally required, remains a deprecated alias. The base schema keeps
    `sort_dir` for the MCP tools that merge it.
    """

    sort_direction = fields.Str(
        load_default=None,
        allow_none=True,
        validate=validate.OneOf(["asc", "desc"]),
        metadata={"description": "Sort direction. One of: `asc`, `desc`. Defaults to `desc`."},
    )
    sort_dir = fields.Str(
        load_default=None,
        allow_none=True,
        validate=validate.OneOf(["asc", "desc"]),
        metadata={
            "description": "Deprecated alias of `sort_direction`, used only when `sort_direction` is absent."
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
    year_min: int | None
    year_max: int | None
    filed_after: str | None
    filed_before: str | None
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


class AgreementResponseSchema(IndustryLabelsSchema):
    agreement_uuid = fields.Str(required=True)
    year = fields.Int(required=True)
    target = fields.Str(required=True)
    acquirer = fields.Str(required=True)
    filing_date = fields.Str(allow_none=True, required=True)
    prob_filing = fields.Float(allow_none=True, required=True)
    filing_company_name = fields.Str(allow_none=True, required=True)
    filing_company_cik = fields.Str(allow_none=True, required=True)
    form_type = fields.Str(allow_none=True, required=True)
    exhibit_type = fields.Str(allow_none=True, required=True)
    transaction_price_total = fields.Float(allow_none=True, required=True)
    transaction_price_stock = fields.Float(allow_none=True, required=True)
    transaction_price_cash = fields.Float(allow_none=True, required=True)
    transaction_price_assets = fields.Float(allow_none=True, required=True)
    transaction_consideration = fields.Str(allow_none=True, required=True)
    target_type = fields.Str(allow_none=True, required=True)
    acquirer_type = fields.Str(allow_none=True, required=True)
    target_industry = fields.Str(allow_none=True, required=True)
    acquirer_industry = fields.Str(allow_none=True, required=True)
    announce_date = fields.Str(allow_none=True, required=True)
    close_date = fields.Str(allow_none=True, required=True)
    deal_status = fields.Str(allow_none=True, required=True)
    attitude = fields.Str(allow_none=True, required=True)
    deal_type = fields.Str(allow_none=True, required=True)
    purpose = fields.Str(allow_none=True, required=True)
    target_pe = pe_flag_field(required=True)
    acquirer_pe = pe_flag_field(required=True)
    url = fields.Str(required=True)
    xml = fields.Str(required=True)
    is_redacted = fields.Bool(required=False)


class AgreementListItemSchema(IndustryLabelsSchema):
    agreement_uuid = fields.Str(required=True)
    year = fields.Int(allow_none=True, required=True)
    target = fields.Str(allow_none=True, required=True)
    acquirer = fields.Str(allow_none=True, required=True)
    filing_date = fields.Str(allow_none=True, required=True)
    prob_filing = fields.Float(allow_none=True, required=True)
    filing_company_name = fields.Str(allow_none=True, required=True)
    filing_company_cik = fields.Str(allow_none=True, required=True)
    form_type = fields.Str(allow_none=True, required=True)
    exhibit_type = fields.Str(allow_none=True, required=True)
    transaction_price_total = fields.Float(allow_none=True, required=True)
    transaction_price_stock = fields.Float(allow_none=True, required=True)
    transaction_price_cash = fields.Float(allow_none=True, required=True)
    transaction_price_assets = fields.Float(allow_none=True, required=True)
    transaction_consideration = fields.Str(allow_none=True, required=True)
    target_type = fields.Str(allow_none=True, required=True)
    acquirer_type = fields.Str(allow_none=True, required=True)
    target_industry = fields.Str(allow_none=True, required=True)
    acquirer_industry = fields.Str(allow_none=True, required=True)
    announce_date = fields.Str(allow_none=True, required=True)
    close_date = fields.Str(allow_none=True, required=True)
    deal_status = fields.Str(allow_none=True, required=True)
    attitude = fields.Str(allow_none=True, required=True)
    deal_type = fields.Str(allow_none=True, required=True)
    purpose = fields.Str(allow_none=True, required=True)
    target_pe = pe_flag_field(required=True)
    acquirer_pe = pe_flag_field(required=True)
    url = fields.Str(allow_none=True, required=True)
    xml = fields.Str(allow_none=True, required=False)


class AgreementsListResponseSchema(Schema):
    results: object = cast(
        object,
        fields.List(
            fields.Nested(AgreementListItemSchema),
            required=True,
        ),
    )
    access = fields.Nested(AccessInfoSchema, required=True)
    page_size = fields.Int(required=True)
    returned_count = fields.Int(required=True)
    has_next = fields.Bool(required=True)
    next_cursor = fields.Str(allow_none=True, required=True)
    dump_version = fields.Nested(DumpVersionInfoSchema, allow_none=True)


class AgreementSearchMatchedSectionSchema(StandardIdLabelsSchema):
    section_uuid = fields.Str(required=True)
    article_title = fields.Str(allow_none=True, required=True)
    section_title = fields.Str(allow_none=True, required=True)
    standard_id = fields.List(fields.Str(), required=True)
    snippet = fields.Str(allow_none=True, required=True)
    matched_terms = fields.List(
        fields.Str(),
        required=False,
        metadata={
            "description": (
                "Present when `text_query` is set: the matched text spans as they appear in "
                "the section, for highlighting. `snippet` is centered on the first match."
            ),
        },
    )


class AgreementSearchResultSchema(IndustryLabelsSchema):
    agreement_uuid = fields.Str(required=True)
    year = fields.Int(allow_none=True, required=True)
    target = fields.Str(allow_none=True, required=True)
    acquirer = fields.Str(allow_none=True, required=True)
    filing_date = fields.Str(allow_none=True, required=True)
    prob_filing = fields.Float(allow_none=True, required=True)
    filing_company_name = fields.Str(allow_none=True, required=True)
    filing_company_cik = fields.Str(allow_none=True, required=True)
    form_type = fields.Str(allow_none=True, required=True)
    exhibit_type = fields.Str(allow_none=True, required=True)
    transaction_price_total = fields.Float(allow_none=True, required=True)
    transaction_price_stock = fields.Float(allow_none=True, required=True)
    transaction_price_cash = fields.Float(allow_none=True, required=True)
    transaction_price_assets = fields.Float(allow_none=True, required=True)
    transaction_consideration = fields.Str(allow_none=True, required=True)
    target_type = fields.Str(allow_none=True, required=True)
    acquirer_type = fields.Str(allow_none=True, required=True)
    target_industry = fields.Str(allow_none=True, required=True)
    acquirer_industry = fields.Str(allow_none=True, required=True)
    announce_date = fields.Str(allow_none=True, required=True)
    close_date = fields.Str(allow_none=True, required=True)
    deal_status = fields.Str(allow_none=True, required=True)
    attitude = fields.Str(allow_none=True, required=True)
    deal_type = fields.Str(allow_none=True, required=True)
    purpose = fields.Str(allow_none=True, required=True)
    target_pe = pe_flag_field(required=True)
    acquirer_pe = pe_flag_field(required=True)
    url = fields.Str(allow_none=True, required=True)
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
            required=True,
        ),
    )
    access = fields.Nested(AccessInfoSchema, required=True)
    page = fields.Int(required=True)
    page_size = fields.Int(required=True)
    total_count = fields.Int(required=True)
    total_count_is_approximate = fields.Bool(required=True)
    total_pages = fields.Int(required=True)
    has_next = fields.Bool(required=True)
    has_prev = fields.Bool(required=True)
    next_num = fields.Int(allow_none=True, required=True)
    prev_num = fields.Int(allow_none=True, required=True)
    count_metadata = fields.Dict(
        required=True,
        metadata={
            "description": (
                "How total_count was computed: `mode` (exact/estimated), `method`, "
                "`planning_reliability`, `exact_count_requested`. With `text_query`, a total "
                "too expensive to count in time is reported as a lower bound "
                "(`method: filtered_lower_bound`, `total_count_is_approximate: true`)."
            ),
        },
    )
    dump_version = fields.Nested(DumpVersionInfoSchema, allow_none=True)


class SectionResponseSchema(StandardIdLabelsSchema):
    agreement_uuid = fields.Str(required=True)
    section_uuid = fields.Str(required=True)
    standard_id = fields.List(fields.Str(), required=True)
    section_standard_id = fields.List(
        fields.Str(),
        required=True,
        metadata={"description": "Deprecated alias of `standard_id`."},
    )
    xml = fields.Str(required=True)
    article_title = fields.Str(required=True)
    section_title = fields.Str(required=True)


class TaxClauseSchema(TaxClauseStandardIdLabelsSchema):
    clause_uuid = fields.Str(required=True)
    agreement_uuid = fields.Str(required=True)
    section_uuid = fields.Str(required=True)
    article_title = fields.Str(allow_none=True, required=True)
    section_title = fields.Str(allow_none=True, required=True)
    anchor_label = fields.Str(allow_none=True, required=True)
    start_char = fields.Int(required=True)
    end_char = fields.Int(required=True)
    clause_text = fields.Str(required=True)
    context_type = fields.Str(
        required=True, validate=validate.OneOf(("operative", "rep_warranty"))
    )
    standard_ids = fields.List(fields.Str(), required=True)


class TaxClauseListResponseSchema(Schema):
    clauses: object = cast(
        object,
        fields.List(
            fields.Nested(TaxClauseSchema),
            required=True,
        ),
    )


class AgreementSectionIndexItemSchema(StandardIdLabelsSchema):
    section_uuid = fields.Str(required=True)
    article_title = fields.Str(allow_none=True, required=True)
    section_title = fields.Str(allow_none=True, required=True)
    article_order = fields.Int(allow_none=True, required=True)
    section_order = fields.Int(allow_none=True, required=True)
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
    changelog_url = fields.Url(required=False, allow_none=True)
    warning = fields.Str(required=False, allow_none=True)


class ChangelogArgsSchema(Schema):
    since = fields.Str(
        required=False,
        load_default=None,
        metadata={
            "description": (
                "Only return releases strictly newer than this version (release versions "
                "are dump dates, e.g. 2026-07-19; same-day repushes get a numeric suffix "
                "such as 2026-07-19.2, which sorts after its base version)."
            ),
            "example": "2026-07-19",
        },
    )
    dump_sha256 = fields.Str(
        required=False,
        load_default=None,
        metadata={
            "description": (
                "Only return the release for this dump sha256. Match it against the "
                "X-Pandects-Dump-Hash response header (or the dump_version.hash body "
                "field) to see exactly which changes the data you are querying includes."
            ),
            "example": "fd703d72dfadd59a0fdc720399f5ecc27c23a8c6f99b76279c80983bf7fd99a8",
        },
    )


class ChangelogChangeSchema(Schema):
    type = fields.Str(
        required=True,
        validate=validate.OneOf(["schema", "data", "api", "mcp", "docs", "pipeline"]),
    )
    severity = fields.Str(
        required=True,
        validate=validate.OneOf(["breaking", "notable", "minor"]),
    )
    summary = fields.Str(required=True)
    details = fields.Str(required=False, allow_none=True)
    tables = fields.List(fields.Str(), required=False, allow_none=True)
    migration = fields.Str(required=False, allow_none=True)
    refs = fields.List(fields.Str(), required=False, allow_none=True)


class ChangelogStatsSchema(Schema):
    row_counts = fields.Dict(
        keys=fields.Str(), values=fields.Int(), required=False, allow_none=True
    )


class ChangelogReleaseSchema(Schema):
    version = fields.Str(required=True)
    released = fields.Str(required=True)
    dump_sha256 = fields.Str(required=True)
    dump_key = fields.Str(required=True)
    schema_fingerprint = fields.Str(required=False, allow_none=True)
    stats = fields.Nested(ChangelogStatsSchema, required=False, allow_none=True)
    changes = fields.List(fields.Nested(ChangelogChangeSchema), required=True)


class ChangelogResponseSchema(Schema):
    latest_version = fields.Str(required=False, allow_none=True)
    latest_released = fields.Str(required=False, allow_none=True)
    releases = fields.List(fields.Nested(ChangelogReleaseSchema), required=True)


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


class TaxonomyCountsResponseSchema(Schema):
    counts = fields.Dict(
        keys=fields.Str(),
        values=fields.Int(),
        required=True,
        metadata={"description": "Count per taxonomy node, keyed by standard ID."},
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
    "AgreementsIndexQueryArgsSchema",
    "AgreementsListResponseSchema",
    "CounselEntrySchema",
    "CounselResponseSchema",
    "DumpEntrySchema",
    "DumpVersionInfoSchema",
    "NaicsResponseSchema",
    "NaicsSectorSchema",
    "NaicsSubSectorSchema",
    "TaxonomyCountsResponseSchema",
    "SectionResponseSchema",
    "TaxClauseListResponseSchema",
    "TaxClauseSchema",
]
