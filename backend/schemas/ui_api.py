"""Response schemas for the web client's internal data routes.

These routes (`/v1/agreements-summary`, `/v1/filter-options`, the dataset
summaries, ...) are plain Flask views built in ``backend/routes/agreements``
and ``backend/routes/telemetry.py``; they are not part of the public spec. The
schemas below describe what those views return and feed the UI spec only (see
``ui_operations.py``). They are never used to serialize a response, so a field
must match the payload the service builds: ``required=True`` means the key is
always present, ``allow_none=True`` that it can be null.

``UI_OPERATIONS`` is the full registry of UI-only operations: these data routes
plus the auth (``ui_auth.py``) and favorites (``ui_favorites.py``) routes.
"""

from __future__ import annotations

from marshmallow import Schema, fields, validate

from backend.schemas.public_api import AgreementsIndexQueryArgsSchema
from backend.schemas.ui_auth import AUTH_OPERATIONS
from backend.schemas.ui_favorites import FAVORITES_OPERATIONS
from backend.schemas.ui_operations import UiOperation

_PAGINATION_DESCRIPTION = "Offset pagination, as on `/v1/sections`."


# ── /v1/agreements-summary ────────────────────────────────────────────────
class AgreementsSummaryResponseSchema(Schema):
    agreements = fields.Int(
        required=True, metadata={"description": "Eligible agreements ingested."}
    )
    processed_agreements = fields.Int(
        required=True,
        allow_none=True,
        metadata={
            "description": (
                "Agreements a user can browse and search (eligible, latest XML verified); "
                "null when the index summary is unavailable."
            )
        },
    )
    sections = fields.Int(required=True)
    pages = fields.Int(required=True)
    latest_filing_date = fields.Str(
        required=True, allow_none=True, metadata={"description": "ISO date (YYYY-MM-DD)."}
    )


# ── /v1/agreements-index ──────────────────────────────────────────────────
class AgreementIndexRowSchema(Schema):
    agreement_uuid = fields.Str(required=True)
    year = fields.Int(required=True, allow_none=True)
    target = fields.Str(required=True, allow_none=True)
    acquirer = fields.Str(required=True, allow_none=True)
    consideration_type = fields.Str(required=True, allow_none=True)
    total_consideration = fields.Float(required=True, allow_none=True)
    verified = fields.Bool(required=True)


class AgreementIndexResponseSchema(Schema):
    results = fields.List(fields.Nested(AgreementIndexRowSchema), required=True)
    page = fields.Int(required=True)
    page_size = fields.Int(required=True)
    total_count = fields.Int(required=True)
    total_count_is_approximate = fields.Bool(required=True)
    total_pages = fields.Int(required=True)
    has_next = fields.Bool(required=True)
    has_prev = fields.Bool(required=True)
    next_num = fields.Int(required=True, allow_none=True)
    prev_num = fields.Int(required=True, allow_none=True)


# ── /v1/agreements-status-summary ─────────────────────────────────────────
class AgreementStatusYearRowSchema(Schema):
    year = fields.Int(required=True)
    color = fields.Str(
        required=True,
        validate=validate.OneOf(("green", "yellow", "red", "gray")),
    )
    current_stage = fields.Str(required=True)
    count = fields.Int(required=True)


class MetadataFieldCoverageRowSchema(Schema):
    field = fields.Str(required=True)
    label = fields.Str(required=True)
    ingested_eligible_agreements = fields.Int(required=True)
    ingested_covered_agreements = fields.Int(required=True)
    ingested_coverage_pct = fields.Float(required=True, allow_none=True)
    processed_eligible_agreements = fields.Int(required=True)
    processed_covered_agreements = fields.Int(required=True)
    processed_coverage_pct = fields.Float(required=True, allow_none=True)
    note = fields.Str(required=True, allow_none=True)


class AgreementStatusSummaryResponseSchema(Schema):
    years = fields.List(fields.Nested(AgreementStatusYearRowSchema), required=True)
    latest_filing_date = fields.Str(required=True, allow_none=True)
    metadata_covered_agreements = fields.Int(required=True)
    metadata_coverage_pct = fields.Float(required=True, allow_none=True)
    metadata_field_coverage = fields.List(
        fields.Nested(MetadataFieldCoverageRowSchema), required=True
    )
    taxonomy_covered_sections = fields.Int(required=True)
    taxonomy_coverage_pct = fields.Float(required=True, allow_none=True)


# ── /v1/agreements-deal-types-summary ─────────────────────────────────────
class AgreementDealTypeYearRowSchema(Schema):
    year = fields.Int(required=True)
    deal_type = fields.Str(required=True)
    count = fields.Int(required=True)


class AgreementDealTypeSummaryResponseSchema(Schema):
    years = fields.List(fields.Nested(AgreementDealTypeYearRowSchema), required=True)


# ── /v1/counsel-leaderboards ──────────────────────────────────────────────
class CounselLeaderboardYearSchema(Schema):
    year = fields.Int(required=True)
    deal_count = fields.Int(required=True)
    total_transaction_value = fields.Float(required=True)


class CounselLeaderboardEntrySchema(Schema):
    counsel = fields.Str(required=True)
    deal_count = fields.Int(required=True)
    total_transaction_value = fields.Float(required=True)
    years = fields.List(fields.Nested(CounselLeaderboardYearSchema), required=True)


class CounselLeaderboardAnnualSchema(Schema):
    year = fields.Int(required=True)
    top_by_count = fields.List(fields.Nested(CounselLeaderboardEntrySchema), required=True)
    top_by_value = fields.List(fields.Nested(CounselLeaderboardEntrySchema), required=True)


class CounselLeaderboardSideSchema(Schema):
    annual = fields.List(fields.Nested(CounselLeaderboardAnnualSchema), required=True)
    top_by_count = fields.List(fields.Nested(CounselLeaderboardEntrySchema), required=True)
    top_by_value = fields.List(fields.Nested(CounselLeaderboardEntrySchema), required=True)


class CounselLeaderboardResponseSchema(Schema):
    buy_side = fields.Nested(CounselLeaderboardSideSchema, required=True)
    sell_side = fields.Nested(CounselLeaderboardSideSchema, required=True)


# ── /v1/agreement-trends ──────────────────────────────────────────────────
class OwnershipMixRowSchema(Schema):
    year = fields.Int(required=True)
    public_deal_count = fields.Int(required=True)
    private_deal_count = fields.Int(required=True)
    public_total_transaction_value = fields.Float(required=True)
    private_total_transaction_value = fields.Float(required=True)


class OwnershipDealSizeRowSchema(Schema):
    year = fields.Int(required=True)
    public_deal_count = fields.Int(required=True)
    private_deal_count = fields.Int(required=True)
    public_p25_transaction_value = fields.Float(required=True, allow_none=True)
    public_median_transaction_value = fields.Float(required=True, allow_none=True)
    public_p75_transaction_value = fields.Float(required=True, allow_none=True)
    private_p25_transaction_value = fields.Float(required=True, allow_none=True)
    private_median_transaction_value = fields.Float(required=True, allow_none=True)
    private_p75_transaction_value = fields.Float(required=True, allow_none=True)


class BuyerTypeMatrixRowSchema(Schema):
    target_bucket = fields.Str(required=True)
    buyer_bucket = fields.Str(required=True)
    deal_count = fields.Int(required=True)
    median_transaction_value = fields.Float(required=True, allow_none=True)


class TargetIndustryByYearRowSchema(Schema):
    year = fields.Int(required=True)
    industry = fields.Str(required=True)
    deal_count = fields.Int(required=True)
    total_transaction_value = fields.Float(required=True)


class IndustryPairingRowSchema(Schema):
    target_industry = fields.Str(required=True)
    acquirer_industry = fields.Str(required=True)
    deal_count = fields.Int(required=True)
    total_transaction_value = fields.Float(required=True)


class AgreementTrendsOwnershipSchema(Schema):
    mix_by_year = fields.List(fields.Nested(OwnershipMixRowSchema), required=True)
    deal_size_by_year = fields.List(fields.Nested(OwnershipDealSizeRowSchema), required=True)
    buyer_type_matrix = fields.List(fields.Nested(BuyerTypeMatrixRowSchema), required=True)


class AgreementTrendsIndustriesSchema(Schema):
    target_industries_by_year = fields.List(
        fields.Nested(TargetIndustryByYearRowSchema), required=True
    )
    pairings = fields.List(fields.Nested(IndustryPairingRowSchema), required=True)


class AgreementTrendsResponseSchema(Schema):
    ownership = fields.Nested(AgreementTrendsOwnershipSchema, required=True)
    industries = fields.Nested(AgreementTrendsIndustriesSchema, required=True)


# ── /v1/filter-options ────────────────────────────────────────────────────
def _option_list(description: str) -> fields.List[str]:
    return fields.List(fields.Str(), metadata={"description": description})


class FilterOptionsQueryArgsSchema(Schema):
    fields_ = fields.List(
        fields.Str(),
        data_key="fields",
        metadata={
            "description": (
                "Option lists to return (repeat the key). Default: all of `targets`, "
                "`acquirers`, `target_counsels`, `acquirer_counsels`, `target_industries`, "
                "`acquirer_industries`, `clause_types`."
            )
        },
    )


class FilterOptionsResponseSchema(Schema):
    """Each list is present only when requested through `fields` (default: all)."""

    targets = _option_list("Distinct target names.")
    acquirers = _option_list("Distinct acquirer names.")
    target_counsels = _option_list("Distinct target counsel names.")
    acquirer_counsels = _option_list("Distinct acquirer counsel names.")
    target_industries = _option_list("Distinct target NAICS codes.")
    acquirer_industries = _option_list("Distinct acquirer NAICS codes.")
    industry_labels = fields.Dict(
        keys=fields.Str(),
        values=fields.Str(),
        metadata={
            "description": (
                "NAICS label per code in `target_industries`/`acquirer_industries`; "
                "present when either list is returned."
            )
        },
    )
    clause_types = fields.Dict(
        keys=fields.Str(),
        values=fields.Raw(),
        metadata={
            "description": "Deprecated: the `GET /v1/taxonomy` tree. Read `/v1/taxonomy` instead."
        },
    )


class FilterOptionValuesQueryArgsSchema(Schema):
    query = fields.Str(metadata={"description": "Case-insensitive prefix."})
    limit = fields.Int(
        metadata={"description": "Maximum values to return (clamped to 1-200; default 100)."}
    )


class FilterOptionValuesResponseSchema(Schema):
    options = fields.List(fields.Str(), required=True)


# ── /v1/page-views ────────────────────────────────────────────────────────
class PageViewRequestSchema(Schema):
    path = fields.Str(required=True, metadata={"description": "Site-relative path."})
    referrer = fields.Str(
        allow_none=True, metadata={"description": "Absolute http(s) referrer URL."}
    )


_DATA_TAG = "ui-data"

DATA_OPERATIONS: tuple[UiOperation, ...] = (
    UiOperation(
        path="/v1/agreements-summary",
        method="get",
        operation_id="getAgreementsSummary",
        tag=_DATA_TAG,
        summary="Corpus totals for the landing page",
        response=AgreementsSummaryResponseSchema,
    ),
    UiOperation(
        path="/v1/agreements-index",
        method="get",
        operation_id="getAgreementsIndex",
        tag=_DATA_TAG,
        summary="Paginated agreement index",
        description=_PAGINATION_DESCRIPTION,
        response=AgreementIndexResponseSchema,
        query=AgreementsIndexQueryArgsSchema,
    ),
    UiOperation(
        path="/v1/agreements-status-summary",
        method="get",
        operation_id="getAgreementsStatusSummary",
        tag=_DATA_TAG,
        summary="Processing status and metadata coverage by year",
        response=AgreementStatusSummaryResponseSchema,
    ),
    UiOperation(
        path="/v1/agreements-deal-types-summary",
        method="get",
        operation_id="getAgreementsDealTypesSummary",
        tag=_DATA_TAG,
        summary="Agreement counts by year and deal type",
        response=AgreementDealTypeSummaryResponseSchema,
    ),
    UiOperation(
        path="/v1/counsel-leaderboards",
        method="get",
        operation_id="getCounselLeaderboards",
        tag=_DATA_TAG,
        summary="Buy-side and sell-side counsel leaderboards",
        response=CounselLeaderboardResponseSchema,
    ),
    UiOperation(
        path="/v1/agreement-trends",
        method="get",
        operation_id="getAgreementTrends",
        tag=_DATA_TAG,
        summary="Ownership and industry trends",
        response=AgreementTrendsResponseSchema,
    ),
    UiOperation(
        path="/v1/filter-options",
        method="get",
        operation_id="getFilterOptions",
        tag=_DATA_TAG,
        summary="Search filter option lists",
        response=FilterOptionsResponseSchema,
        query=FilterOptionsQueryArgsSchema,
    ),
    UiOperation(
        path="/v1/filter-options/{field_name}",
        method="get",
        operation_id="getFilterOptionValues",
        tag=_DATA_TAG,
        summary="Typeahead values for `target` or `acquirer`",
        response=FilterOptionValuesResponseSchema,
        query=FilterOptionValuesQueryArgsSchema,
    ),
    UiOperation(
        path="/v1/page-views",
        method="post",
        operation_id="recordPageView",
        tag=_DATA_TAG,
        summary="Record a page view",
        description="Always answers 204; invalid payloads are ignored.",
        response=None,
        status=204,
        response_description="Recorded or ignored",
        request_body=PageViewRequestSchema,
    ),
)

UI_OPERATIONS: tuple[UiOperation, ...] = DATA_OPERATIONS + AUTH_OPERATIONS + FAVORITES_OPERATIONS

__all__ = ["DATA_OPERATIONS", "UI_OPERATIONS"]
