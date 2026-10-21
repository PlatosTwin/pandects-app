from __future__ import annotations

from typing import TypedDict, cast

from marshmallow import Schema, fields, validate

from backend.schemas.labels import TaxStandardIdLabelsSchema
from backend.schemas.sections import AccessInfoSchema, DumpVersionInfoSchema


class TaxClausesArgsPayload(TypedDict):
    year: list[int]
    target: list[str]
    acquirer: list[str]
    tax_standard_id: list[str]
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
    clause_uuid: str | None
    include_rep_warranty: bool
    count_mode: str
    sort_by: str
    sort_direction: str
    page: int
    page_size: int


class TaxClausesArgsSchema(Schema):
    year = fields.List(fields.Int(), load_default=[])
    target = fields.List(fields.Str(), load_default=[])
    acquirer = fields.List(fields.Str(), load_default=[])
    tax_standard_id = fields.List(
        fields.Str(),
        load_default=[],
        metadata={
            "description": (
                "Tax clause taxonomy standard IDs. Parent IDs expand to descendants."
            ),
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
    agreement_uuid = fields.Str(load_default=None, allow_none=True)
    section_uuid = fields.Str(load_default=None, allow_none=True)
    clause_uuid = fields.Str(load_default=None, allow_none=True)
    include_rep_warranty = fields.Bool(
        load_default=False,
        metadata={
            "description": (
                "Include clauses found inside representations & warranties articles. "
                "Defaults to false so that operative drafting precedents are surfaced first."
            ),
        },
    )
    count_mode = fields.Str(
        load_default="auto",
        validate=validate.OneOf(["auto", "exact"]),
    )
    sort_by = fields.Str(
        load_default="year",
        validate=validate.OneOf(["year", "target", "acquirer"]),
    )
    sort_direction = fields.Str(
        load_default="desc",
        validate=validate.OneOf(["asc", "desc"]),
    )
    page = fields.Int(load_default=1)
    page_size = fields.Int(load_default=25)


class TaxClausesQueryArgsSchema(TaxClausesArgsSchema):
    """`/v1/tax-clauses` arguments: the shared filter schema plus REST-only `include_dump`.

    The MCP `search_tax_clauses` tool validates against the base schema and returns no
    dump version, so the flag stays off its input schema.
    """

    include_dump = fields.Bool(
        load_default=True,
        metadata={
            "description": (
                "When false, omit `dump_version` from the response body. "
                "The `X-Pandects-Dump-Hash` response header is always included."
            ),
        },
    )


class TaxClauseItemSchema(TaxStandardIdLabelsSchema):
    id = fields.Str(required=True)
    clause_uuid = fields.Str(required=True)
    agreement_uuid = fields.Str(required=True)
    section_uuid = fields.Str(required=True)
    clause_text = fields.Str(required=True)
    anchor_label = fields.Str(allow_none=True, required=True)
    context_type = fields.Str(
        required=True, validate=validate.OneOf(("operative", "rep_warranty"))
    )
    source_method = fields.Str(allow_none=True, required=True)
    tax_standard_ids = fields.List(fields.Str(), required=True)
    year = fields.Int(allow_none=True, required=True)
    target = fields.Str(allow_none=True, required=True)
    acquirer = fields.Str(allow_none=True, required=True)
    verified = fields.Bool(required=True)
    transaction_price_total = fields.Str(allow_none=True, required=True)
    transaction_consideration = fields.Str(allow_none=True, required=True)
    deal_status = fields.Str(allow_none=True, required=True)
    deal_type = fields.Str(allow_none=True, required=True)
    target_counsel = fields.Str(allow_none=True, required=True)
    acquirer_counsel = fields.Str(allow_none=True, required=True)


class TaxClausesResponseSchema(Schema):
    results: object = cast(
        object,
        fields.List(fields.Nested(TaxClauseItemSchema), required=True),
    )
    access = fields.Nested(AccessInfoSchema, required=True)
    page = fields.Int(required=True)
    page_size = fields.Int(required=True)
    total_count = fields.Int(required=True)
    total_count_is_approximate = fields.Bool(required=True)
    count_metadata = fields.Dict(required=True)
    total_pages = fields.Int(required=True)
    has_next = fields.Bool(required=True)
    has_prev = fields.Bool(required=True)
    next_num = fields.Int(allow_none=True, required=True)
    prev_num = fields.Int(allow_none=True, required=True)
    dump_version = fields.Nested(DumpVersionInfoSchema, allow_none=True)


__all__ = [
    "TaxClausesArgsPayload",
    "TaxClausesArgsSchema",
    "TaxClausesQueryArgsSchema",
    "TaxClauseItemSchema",
    "TaxClausesResponseSchema",
]
