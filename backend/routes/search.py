from __future__ import annotations

from typing import cast

from flask_smorest import Blueprint
from flask.views import MethodView

from backend.schemas.search import SearchArgsPayload, SearchArgsSchema, SearchResponseSchema
from backend.services.search_service import run_search


def register_search_routes(*, app_module: object) -> Blueprint:
    search_blp = Blueprint(
        "search",
        "search",
        url_prefix="/v1/search",
        description="Search merger agreement sections",
    )

    @search_blp.route("")  # pyright: ignore[reportUnknownMemberType]
    class SearchResource(MethodView):  # pyright: ignore[reportUnusedClass]
        @search_blp.doc(  # pyright: ignore[reportUnknownMemberType]
            operationId="searchSections",
            summary="Search agreement sections",
            description=(
                "Searches sections using structured filters and taxonomy IDs. For list filters, "
                "repeat query keys (for example `year=2023&year=2024`)."
            ),
        )
        @search_blp.arguments(SearchArgsSchema, location="query")  # pyright: ignore[reportUnknownMemberType]
        @search_blp.response(200, SearchResponseSchema)  # pyright: ignore[reportUnknownMemberType]
        def get(self, args: dict[str, object]) -> dict[str, object]:
            ctx = cast(object, getattr(app_module, "_current_access_context")())
            parsed_args = cast(SearchArgsPayload, cast(object, args))
            return run_search(app_module, ctx=ctx, parsed_args=parsed_args)

    return search_blp
