from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from flask_smorest import Blueprint
from flask.views import MethodView

from backend.routes.deps import AccessContextProtocol
from backend.schemas.tax_clauses import (
    TaxClausesArgsPayload,
    TaxClausesArgsSchema,
    TaxClausesResponseSchema,
)
from backend.services.tax_clauses_service import TaxClausesServiceDeps, run_tax_clauses


@dataclass(frozen=True)
class TaxClausesDeps:
    _current_access_context: Callable[[], AccessContextProtocol]
    tax_clauses_service_deps: TaxClausesServiceDeps


def register_tax_clauses_routes(*, deps: TaxClausesDeps) -> Blueprint:
    tax_clauses_blp = Blueprint(
        "tax-clauses",
        "tax-clauses",
        url_prefix="/v1/tax-clauses",
        description="List tax clause precedents extracted from merger agreements",
    )

    @tax_clauses_blp.route("")
    class TaxClausesResource(MethodView):
        @tax_clauses_blp.doc(
            operationId="listTaxClauses",
            summary="List tax clause precedents",
            tags=["tax-clauses"],
            description=(
                "Searches per-clause tax precedents using structured filters and taxonomy IDs. "
                "Defaults to excluding representations & warranties articles; pass "
                "`include_rep_warranty=true` to include them."
            ),
        )
        @tax_clauses_blp.arguments(TaxClausesArgsSchema, location="query")
        @tax_clauses_blp.response(200, TaxClausesResponseSchema)
        def get(self, args: dict[str, object]) -> dict[str, object]:
            ctx = deps._current_access_context()
            parsed_args = cast(TaxClausesArgsPayload, cast(object, args))
            return run_tax_clauses(
                deps.tax_clauses_service_deps,
                ctx=ctx,
                parsed_args=parsed_args,
            )

    return tax_clauses_blp


__all__ = ["TaxClausesDeps", "register_tax_clauses_routes"]
