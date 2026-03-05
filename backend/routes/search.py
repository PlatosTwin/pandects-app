from __future__ import annotations

from typing import cast

from flask_smorest import Blueprint
from flask.views import MethodView

from backend.routes.deps import SearchDeps
from backend.schemas.search import SearchArgsPayload, SearchArgsSchema, SearchResponseSchema
from backend.services.search_service import run_search


def register_search_routes(*, deps: SearchDeps) -> Blueprint:
    search_blp = Blueprint(
        "search",
        "search",
        url_prefix="/v1/search",
        description="Search merger agreement sections",
    )

    @search_blp.route("")
    class SearchResource(MethodView):
        @search_blp.doc(
            operationId="searchSections",
            summary="Search agreement sections",
            description=(
                "Searches sections using structured filters and taxonomy IDs. For list filters, "
                "repeat query keys (for example `year=2023&year=2024`)."
            ),
        )
        @search_blp.arguments(SearchArgsSchema, location="query")
        @search_blp.response(200, SearchResponseSchema)
        def get(self, args: dict[str, object]) -> dict[str, object]:
            ctx = cast(object, deps._current_access_context())
            parsed_args = cast(SearchArgsPayload, cast(object, args))
            return run_search(deps.search_service_deps, ctx=ctx, parsed_args=parsed_args)

    return search_blp
