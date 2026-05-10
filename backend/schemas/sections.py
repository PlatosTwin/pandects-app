from __future__ import annotations

from typing import TypedDict, cast

from marshmallow import Schema, fields, post_dump, validate


SECTIONS_RESULT_METADATA_FIELDS = (
    "filing_date",
    "prob_filing",
    "filing_company_name",
    "filing_company_cik",
    "form_type",
    "exhibit_type",
    "transaction_price_total",
    "transaction_price_stock",
    "transaction_price_cash",
    "transaction_price_assets",
    "transaction_consideration",
    "target_type",
    "acquirer_type",
    "target_industry",
    "acquirer_industry",
    "announce_date",
    "close_date",
    "deal_status",
    "attitude",
    "deal_type",
    "purpose",
    "target_pe",
    "acquirer_pe",
    "url",
)


class SectionsArgsPayload(TypedDict):
    year: list[int]
    year_min: int | None
    year_max: int | None
    target: list[str]
    acquirer: list[str]
    standard_id: list[str]
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
    metadata: list[str]
    filed_after: str | None
    filed_before: str | None
    agreement_uuid: str | None
    section_uuid: str | None
    include_dump: bool
    include_xml: bool
    count_mode: str
    sort_by: str
    sort_direction: str
    page: int
    page_size: int


class SectionsArgsSchema(Schema):
    year = fields.List(
        fields.Int(),
        load_default=[],
        metadata={
            "description": "Agreement year filter. Repeat query key for multiple values.",
            "example": [2022, 2023],
        },
    )
    year_min = fields.Int(load_default=None, allow_none=True, validate=validate.Range(min=1900, max=2100))
    year_max = fields.Int(load_default=None, allow_none=True, validate=validate.Range(min=1900, max=2100))
    filed_after = fields.Str(load_default=None, allow_none=True, validate=validate.Regexp(r'^\d{4}-\d{2}-\d{2}$'))
    filed_before = fields.Str(load_default=None, allow_none=True, validate=validate.Regexp(r'^\d{4}-\d{2}-\d{2}$'))
    target = fields.List(
        fields.Str(),
        load_default=[],
        metadata={
            "description": "Exact target company names to include.",
            "example": ["Slack Technologies, Inc."],
        },
    )
    acquirer = fields.List(
        fields.Str(),
        load_default=[],
        metadata={
            "description": "Exact acquirer company names to include.",
            "example": ["salesforce.com, inc."],
        },
    )
    standard_id = fields.List(
        fields.Str(),
        load_default=[],
        metadata={
            "description": (
                "Clause type taxonomy standard IDs (Clause Type in the Search UI). "
                "Parent IDs expand to include descendant taxonomy nodes."
            ),
            "example": ["1.1", "1.2.3"],
        },
    )
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
    metadata = fields.List(
        fields.Str(validate=validate.OneOf(SECTIONS_RESULT_METADATA_FIELDS)),
        load_default=[],
        metadata={
            "description": (
                "Additional agreement metadata fields to include in each result under "
                "`results[].metadata`. Repeat query key for multiple values."
            ),
            "example": ["deal_type", "target_industry"],
        },
    )
    agreement_uuid = fields.Str(
        load_default=None,
        allow_none=True,
        metadata={"description": "Filter to one agreement UUID."},
    )
    section_uuid = fields.Str(
        load_default=None,
        allow_none=True,
        metadata={
            "description": "Filter to one section UUID.",
            "example": "5e59453aaa9255c4",
        },
    )
    count_mode = fields.Str(
        load_default="auto",
        validate=validate.OneOf(["auto", "exact"]),
        metadata={
            "description": (
                "Count strategy for pagination planning. `auto` may return estimates "
                "for broad or paginated searches; `exact` forces an exact total count."
            ),
            "example": "auto",
        },
    )
    sort_by = fields.Str(
        load_default="year",
        validate=validate.OneOf(["year", "target", "acquirer"]),
        metadata={"description": "Sort key. One of: `year`, `target`, `acquirer`."},
    )
    sort_direction = fields.Str(
        load_default="desc",
        validate=validate.OneOf(["asc", "desc"]),
        metadata={"description": "Sort direction. One of: `asc`, `desc`."},
    )
    page = fields.Int(
        load_default=1,
        metadata={"description": "1-based page number.", "example": 1},
    )
    page_size = fields.Int(
        load_default=25,
        metadata={
            "description": (
                "Page size. Effective max is 10 for unauthenticated callers and 100 for "
                "authenticated callers."
            ),
            "example": 25,
        },
    )
    include_dump = fields.Bool(
        load_default=True,
        metadata={
            "description": (
                "When false, omit `dump_version` from the response body. "
                "The `X-Pandects-Dump-Hash` response header is always included."
            ),
        },
    )
    include_xml = fields.Bool(
        load_default=False,
        metadata={
            "description": "Deprecated compatibility parameter. Section search results include section XML for all callers.",
        },
    )


class SectionsResultMetadataSchema(Schema):
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


class SectionItemSchema(Schema):
    id = fields.Str()
    agreement_uuid = fields.Str()
    section_uuid = fields.Str()
    standard_id = fields.List(fields.Str())
    xml = fields.Str()
    article_title = fields.Str()
    section_title = fields.Str()
    acquirer = fields.Str()
    target = fields.Str()
    year = fields.Int()
    verified = fields.Bool()
    metadata = fields.Nested(
        lambda: SectionsResultMetadataSchema(),
        required=False,
        allow_none=True,
    )

    @post_dump
    def _drop_missing_xml(self, data: dict[str, object], **_kwargs: object) -> dict[str, object]:
        if data.get("xml") is None:
            _ = data.pop("xml", None)
        return data


class AccessInfoSchema(Schema):
    tier = fields.Str(required=True)
    message = fields.Str(required=False, allow_none=True)


class DumpVersionInfoSchema(Schema):
    hash = fields.Str(required=True)
    dump_ts = fields.Str(required=True)


class SectionsResponseSchema(Schema):
    results: object = cast(
        object,
        fields.List(
            fields.Nested(SectionItemSchema),
        ),
    )
    access = fields.Nested(AccessInfoSchema)
    page = fields.Int()
    page_size = fields.Int()
    total_count = fields.Int()
    total_count_is_approximate = fields.Bool()
    count_metadata = fields.Dict()
    interpretation = fields.Dict()
    total_pages = fields.Int()
    has_next = fields.Bool()
    has_prev = fields.Bool()
    next_num = fields.Int(allow_none=True)
    prev_num = fields.Int(allow_none=True)
    dump_version = fields.Nested(DumpVersionInfoSchema, allow_none=True)


__all__ = [
    "AccessInfoSchema",
    "DumpVersionInfoSchema",
    "SECTIONS_RESULT_METADATA_FIELDS",
    "SectionsArgsPayload",
    "SectionsArgsSchema",
    "SectionsResponseSchema",
    "SectionsResultMetadataSchema",
    "SectionItemSchema",
]
