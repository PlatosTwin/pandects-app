from __future__ import annotations

from typing import cast

from flask_smorest import Blueprint
from flask.views import MethodView

from backend.routes.deps import SectionsDeps
from backend.schemas.sections import SectionsArgsPayload, SectionsArgsSchema, SectionsResponseSchema
from backend.services.sections_service import run_sections


def register_sections_routes(*, deps: SectionsDeps) -> Blueprint:
    sections_blp = Blueprint(
        "sections_list",
        "sections_list",
        url_prefix="/v1/sections",
        description="List merger agreement sections",
    )

    @sections_blp.route("")
    class SectionsResource(MethodView):
        @sections_blp.doc(
            operationId="listSections",
            summary="List agreement sections",
            tags=["sections"],
            description=(
                "Searches sections using structured filters and taxonomy IDs. For list filters, "
                "repeat query keys (for example `year=2023&year=2024`)."
            ),
        )
        @sections_blp.arguments(SectionsArgsSchema, location="query")
        @sections_blp.response(200, SectionsResponseSchema)
        def get(self, args: dict[str, object]) -> dict[str, object]:
            ctx = deps._current_access_context()
            parsed_args = cast(SectionsArgsPayload, cast(object, args))
            return run_sections(deps.sections_service_deps, ctx=ctx, parsed_args=parsed_args)

    return sections_blp
