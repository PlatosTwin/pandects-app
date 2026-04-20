from __future__ import annotations

from typing import TypedDict, cast

from marshmallow import Schema, fields, validate

from backend.schemas.sections import AccessInfoSchema


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


class TaxClauseItemSchema(Schema):
    id = fields.Str()
    clause_uuid = fields.Str()
    agreement_uuid = fields.Str()
    section_uuid = fields.Str()
    clause_text = fields.Str()
    anchor_label = fields.Str(allow_none=True)
    context_type = fields.Str()
    source_method = fields.Str(allow_none=True)
    tax_standard_ids = fields.List(fields.Str())
    year = fields.Int(allow_none=True)
    target = fields.Str(allow_none=True)
    acquirer = fields.Str(allow_none=True)
    verified = fields.Bool()
    transaction_price_total = fields.Str(allow_none=True)
    transaction_consideration = fields.Str(allow_none=True)
    deal_status = fields.Str(allow_none=True)
    deal_type = fields.Str(allow_none=True)
    target_counsel = fields.Str(allow_none=True)
    acquirer_counsel = fields.Str(allow_none=True)


class TaxClausesResponseSchema(Schema):
    results: object = cast(
        object,
        fields.List(fields.Nested(TaxClauseItemSchema)),
    )
    access = fields.Nested(AccessInfoSchema)
    page = fields.Int()
    page_size = fields.Int()
    total_count = fields.Int()
    total_count_is_approximate = fields.Bool()
    count_metadata = fields.Dict()
    total_pages = fields.Int()
    has_next = fields.Bool()
    has_prev = fields.Bool()
    next_num = fields.Int(allow_none=True)
    prev_num = fields.Int(allow_none=True)


__all__ = [
    "TaxClausesArgsPayload",
    "TaxClausesArgsSchema",
    "TaxClauseItemSchema",
    "TaxClausesResponseSchema",
]
