"""HTTP wiring for the agreements domain.

Each view validates its request, delegates to ``backend.services.agreements_service``,
and shapes the response (access checks, dump version, cache headers). The summary
endpoints and the default filter-options payload keep an in-process
``ttl_cached`` entry each (one slot per payload, see ``backend/core/caching.py``).
"""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Hashable
from typing import cast

from flask import Flask, Response, abort, g, jsonify, request
from flask.views import MethodView
from flask_smorest import Blueprint

from backend.core.caching import ttl_cached
from backend.redaction import READER_INDEX_REDACTION_VERSION, redact_reader_index
from backend.routes.agreements.helpers import (
    _agreement_is_public_eligible_expr,
    _cacheable_json_response,
    _to_float_or_none,
)
from backend.routes.deps import AgreementsDeps
from backend.schemas.public_api import (
    AgreementArgsPayload,
    AgreementArgsSchema,
    AgreementResponseSchema,
    AgreementSearchResponseSchema,
    AgreementSectionIndexResponseSchema,
    AgreementsBulkArgsPayload,
    AgreementsBulkArgsSchema,
    AgreementsIndexQueryArgsSchema,
    AgreementsListResponseSchema,
    ReaderIndexResponseSchema,
    SectionResponseSchema,
    TaxClauseListResponseSchema,
)
from backend.schemas.sections import SectionsArgsPayload, SectionsArgsSchema
from backend.services import agreements_service

__all__ = [
    "AGREEMENTS_SUMMARY_TTL_SECONDS",
    "FILTER_OPTIONS_TTL_SECONDS",
    "_agreement_is_public_eligible_expr",
    "_to_float_or_none",
    "register_agreements_routes",
]

AGREEMENTS_SUMMARY_TTL_SECONDS = int(os.environ.get("AGREEMENTS_SUMMARY_TTL_SECONDS", "60"))
FILTER_OPTIONS_TTL_SECONDS = int(os.environ.get("FILTER_OPTIONS_TTL_SECONDS", "21600"))


def _single_slot(_deps: AgreementsDeps, /) -> Hashable:
    """Cache key for a payload that does not vary by argument (the deps handle)."""
    return ()


@ttl_cached(AGREEMENTS_SUMMARY_TTL_SECONDS, key=_single_slot)
def agreements_summary_cached(deps: AgreementsDeps) -> dict[str, object]:
    return agreements_service.agreements_corpus_summary_payload(deps)


@ttl_cached(AGREEMENTS_SUMMARY_TTL_SECONDS, key=_single_slot)
def agreements_status_summary_cached(deps: AgreementsDeps) -> dict[str, object]:
    return agreements_service.agreements_status_summary_payload(deps)


@ttl_cached(AGREEMENTS_SUMMARY_TTL_SECONDS, key=_single_slot)
def agreements_deal_types_summary_cached(deps: AgreementsDeps) -> dict[str, object]:
    return agreements_service.agreements_deal_types_summary_payload(deps)


@ttl_cached(AGREEMENTS_SUMMARY_TTL_SECONDS, key=_single_slot)
def counsel_leaderboards_cached(deps: AgreementsDeps) -> dict[str, object]:
    return agreements_service.counsel_leaderboards_payload(deps)


@ttl_cached(AGREEMENTS_SUMMARY_TTL_SECONDS, key=_single_slot)
def agreement_trends_cached(deps: AgreementsDeps) -> dict[str, object]:
    return agreements_service.agreement_trends_payload(deps)


@ttl_cached(FILTER_OPTIONS_TTL_SECONDS, key=_single_slot)
def default_filter_options_cached(deps: AgreementsDeps) -> dict[str, object]:
    """The full filter-options payload; a `fields=` subset is never cached."""
    return agreements_service.filter_options_payload(deps, agreements_service.FILTER_OPTION_FIELDS)


def _with_dump_version(payload: dict[str, object], args: dict[str, object]) -> dict[str, object]:
    dump_version = getattr(g, "dump_version", None)
    if dump_version is not None and bool(args.get("include_dump", True)):
        payload["dump_version"] = dump_version
    return payload


def register_agreements_routes(
    target_app: Flask,
    *,
    deps: AgreementsDeps,
) -> tuple[Blueprint, Blueprint, Blueprint]:
    agreements_blp = Blueprint(
        "agreements",
        "agreements",
        url_prefix="/v1/agreements",
        description="Retrieve full text for a given agreement",
    )
    sections_blp = Blueprint(
        "sections",
        "sections",
        url_prefix="/v1/sections",
        description="Retrieve full text for a given section",
    )
    agreement_search_blp = Blueprint(
        "agreements_search",
        "agreements_search",
        url_prefix="/v1/search/agreements",
        description="Search agreements and return one result per deal with matched section previews.",
    )

    @agreements_blp.route("")
    class AgreementsListResource(MethodView):
        @agreements_blp.doc(
            operationId="listAgreements",
            summary="List agreements with keyset pagination",
            description=(
                "Lists eligible agreements using a base64 cursor. Supports the same agreement-level "
                "filters as `/v1/sections` (including `year_min`/`year_max` and "
                "`filed_after`/`filed_before`) except clause-type taxonomy filtering."
            ),
        )
        @agreements_blp.arguments(AgreementsBulkArgsSchema, location="query")
        @agreements_blp.response(200, AgreementsListResponseSchema)
        def get(self, args: dict[str, object]) -> dict[str, object]:
            ctx = deps._current_access_context()
            parsed_args = cast(AgreementsBulkArgsPayload, cast(object, args))
            if "standard_id" in request.args:
                abort(400, description="The standard_id filter is not supported on /v1/agreements.")
            if parsed_args["include_xml"] and not ctx.is_authenticated:
                abort(403, description="Authentication required when include_xml=true.")
            payload = agreements_service.list_agreements(deps, ctx=ctx, parsed_args=parsed_args)
            return _with_dump_version(payload, args)

    @agreement_search_blp.route("")
    class AgreementSearchResource(MethodView):
        @agreement_search_blp.doc(
            operationId="searchAgreements",
            summary="Search agreements with matched section previews",
            description=(
                "Searches agreements with the same structured filters as `/v1/sections`, "
                "including taxonomy IDs and `text_query`, and returns one row per agreement. "
                "With `standard_id`, `section_uuid`, or `text_query`, an agreement matches when "
                "at least one of its latest sections matches them all; `match_count` counts those "
                "sections and `matched_sections` previews the first three in document order "
                "(with `text_query`, each snippet is centered on the match and `matched_terms` "
                "lists the matched text). `metadata` is rejected: every row already carries the "
                "full agreement metadata."
            ),
        )
        @agreement_search_blp.arguments(SectionsArgsSchema, location="query")
        @agreement_search_blp.response(200, AgreementSearchResponseSchema)
        def get(self, args: dict[str, object]) -> dict[str, object]:
            ctx = deps._current_access_context()
            parsed_args = cast(SectionsArgsPayload, cast(object, args))
            payload = agreements_service.search_agreements(deps, ctx=ctx, parsed_args=parsed_args)
            return _with_dump_version(payload, args)

    @agreements_blp.route("/<string:agreement_uuid>")
    class AgreementResource(MethodView):
        @agreements_blp.doc(
            operationId="getAgreement",
            summary="Retrieve agreement text by UUID",
            description="Returns agreement metadata and XML content. For anonymous callers, XML can be redacted based on `focus_section_uuid` and `neighbor_sections`.",
        )
        @agreements_blp.arguments(AgreementArgsSchema, location="query")
        @agreements_blp.response(200, AgreementResponseSchema)
        def get(self, args: dict[str, object], agreement_uuid: str) -> dict[str, object]:
            ctx = deps._current_access_context()
            parsed_args = cast(AgreementArgsPayload, cast(object, args))
            focus_section_uuid = parsed_args.get("focus_section_uuid")
            if focus_section_uuid is not None:
                focus_section_uuid = focus_section_uuid.strip()
                if not deps._SECTION_ID_RE.match(focus_section_uuid):
                    abort(400, description="Invalid focus_section_uuid.")
            payload = agreements_service.get_agreement(
                deps,
                ctx=ctx,
                agreement_uuid=agreement_uuid,
                focus_section_uuid=focus_section_uuid,
                neighbor_sections=parsed_args["neighbor_sections"],
            )
            if payload is None:
                abort(404)
            return payload

    @agreements_blp.route("/<string:agreement_uuid>/sections")
    class AgreementSectionsIndexResource(MethodView):
        @agreements_blp.doc(
            operationId="listAgreementSections",
            summary="List ordered agreement sections",
            description=(
                "Returns ordered section metadata for one agreement, including section UUIDs "
                "and taxonomy IDs, for in-document navigation."
            ),
        )
        @agreements_blp.response(200, AgreementSectionIndexResponseSchema)
        def get(self, agreement_uuid: str) -> dict[str, object]:
            payload = agreements_service.list_agreement_sections(deps, agreement_uuid)
            if payload is None:
                abort(404)
            return payload

    @agreements_blp.route("/<string:agreement_uuid>/reader-index")
    class AgreementReaderIndexResource(MethodView):
        @agreements_blp.doc(
            operationId="getAgreementReaderIndex",
            summary="Retrieve the reader index for an agreement",
            description=(
                "Returns the outline, defined terms (declarations and uses) and cross-references "
                "of the XML version `/v1/agreements/{agreement_uuid}` serves, as stand-off "
                "anchors into that XML; no agreement text is included. For anonymous callers, "
                "items inside redacted text are dropped and declarations there are masked, "
                "following the same `focus_section_uuid` and `neighbor_sections` as the "
                "agreement endpoint. Returns 404 when no index has been built for the current "
                "XML version. Supports `If-None-Match`."
            ),
        )
        @agreements_blp.arguments(AgreementArgsSchema, location="query")
        @agreements_blp.response(200, ReaderIndexResponseSchema)
        def get(self, args: dict[str, object], agreement_uuid: str) -> Response:
            ctx = deps._current_access_context()
            parsed_args = cast(AgreementArgsPayload, cast(object, args))
            focus_section_uuid = parsed_args.get("focus_section_uuid")
            if focus_section_uuid is not None:
                focus_section_uuid = focus_section_uuid.strip()
                if not deps._SECTION_ID_RE.match(focus_section_uuid):
                    abort(400, description="Invalid focus_section_uuid.")
            neighbor_sections = parsed_args["neighbor_sections"]
            if neighbor_sections < 0 or neighbor_sections > 5:
                abort(400, description="neighbor_sections must be between 0 and 5.")
            redact = not ctx.is_authenticated
            agreement_uuid = agreement_uuid.strip()
            # versions first: a browser revalidates on every view, and a 304 should not
            # pull the payload (or, for anonymous callers, the whole XML) from the database
            row = agreements_service.get_reader_index_row(
                deps, agreement_uuid, include_payload=False, include_xml=False
            )
            if row is None:
                abort(404)
            window = f"-r{READER_INDEX_REDACTION_VERSION}-{focus_section_uuid or ''}-{neighbor_sections}"

            def _etag(r: dict[str, object]) -> str:
                return f"{r['xml_version']}-{r['enricher_version']}" + (window if redact else "")

            etag = _etag(row)
            if request.if_none_match.contains_weak(etag):
                resp = Response(status=304)
            else:
                row = agreements_service.get_reader_index_row(
                    deps, agreement_uuid, include_payload=True, include_xml=redact
                )
                if row is None:
                    abort(404)
                etag = _etag(row)  # the row may have been rebuilt between the two reads
                body = cast(bytes, row["payload_gz"])
                if redact:
                    payload = json.loads(gzip.decompress(body))
                    payload = redact_reader_index(
                        payload,
                        cast(str, row.get("xml") or ""),
                        focus_section_uuid=focus_section_uuid,
                        neighbor_sections=neighbor_sections,
                    )
                    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                    body = gzip.compress(raw, compresslevel=6)
                if request.accept_encodings.quality("gzip") > 0:
                    resp = Response(body, status=200, mimetype="application/json")
                    resp.headers["Content-Encoding"] = "gzip"
                else:
                    resp = Response(gzip.decompress(body), status=200, mimetype="application/json")
            # weak: the same payload is sent gzip-encoded or not
            resp.set_etag(etag, weak=True)
            # The payload differs once a caller signs in, so no shared cache keeps it and the
            # browser revalidates every time; an unchanged index costs a 304.
            resp.headers["Cache-Control"] = "private, no-cache"
            resp.vary.add("Accept-Encoding")
            return resp

    @sections_blp.route("/<string:section_uuid>")
    class SectionResource(MethodView):
        @sections_blp.doc(
            operationId="getSection",
            summary="Retrieve section text by UUID",
            description=(
                "Returns one section payload including taxonomy IDs (`standard_id`; "
                "`section_standard_id` is a deprecated alias) and XML content."
            ),
        )
        @sections_blp.response(200, SectionResponseSchema)
        def get(self, section_uuid: str) -> dict[str, object]:
            section_uuid = section_uuid.strip()
            if not deps._SECTION_ID_RE.match(section_uuid):
                abort(400, description="Invalid section_uuid.")
            payload = agreements_service.get_section(deps, section_uuid)
            if payload is None:
                abort(404)
            return payload

    @agreements_blp.route("/<string:agreement_uuid>/tax-clauses")
    class AgreementTaxClausesResource(MethodView):
        @agreements_blp.doc(
            operationId="getAgreementTaxClauses",
            summary="Retrieve tax clauses for an agreement",
            description="Returns extracted tax-module clauses for the latest verified XML of an agreement.",
        )
        @agreements_blp.response(200, TaxClauseListResponseSchema)
        def get(self, agreement_uuid: str) -> dict[str, object]:
            agreement_uuid = agreement_uuid.strip()
            if agreement_uuid == "":
                abort(400, description="Invalid agreement_uuid.")
            return {"clauses": agreements_service.tax_clause_rows(deps, agreement_uuid=agreement_uuid)}

    @sections_blp.route("/<string:section_uuid>/tax-clauses")
    class SectionTaxClausesResource(MethodView):
        @sections_blp.doc(
            operationId="getSectionTaxClauses",
            summary="Retrieve tax clauses for a section",
            description="Returns extracted tax-module clauses for a specific latest section.",
        )
        @sections_blp.response(200, TaxClauseListResponseSchema)
        def get(self, section_uuid: str) -> dict[str, object]:
            section_uuid = section_uuid.strip()
            if not deps._SECTION_ID_RE.match(section_uuid):
                abort(400, description="Invalid section_uuid.")
            return {"clauses": agreements_service.tax_clause_rows(deps, section_uuid=section_uuid)}

    def get_agreements_index() -> dict[str, object]:
        ctx = deps._current_access_context()
        args = deps._load_query(AgreementsIndexQueryArgsSchema())
        # `sort_direction` matches every other endpoint; `sort_dir` is the deprecated
        # spelling this endpoint used to require, and still works when it is the only one.
        sort_direction = args.get("sort_direction") or args.get("sort_dir") or "desc"
        return agreements_service.agreements_index(
            deps,
            ctx=ctx,
            page=cast(int, args["page"]),
            page_size=cast(int, args["page_size"]),
            sort_by=str(args["sort_by"] or "year"),
            sort_direction=str(sort_direction),
            query=str(args.get("query") or "").strip(),
        )

    def _summary_response(payload: dict[str, object]) -> tuple[Response, int]:
        return _cacheable_json_response(payload, max_age=AGREEMENTS_SUMMARY_TTL_SECONDS)

    def get_agreements_status_summary() -> tuple[Response, int]:
        return _summary_response(agreements_status_summary_cached(deps))

    def get_agreements_deal_types_summary() -> tuple[Response, int]:
        return _summary_response(agreements_deal_types_summary_cached(deps))

    def get_counsel_leaderboards() -> tuple[Response, int]:
        return _summary_response(counsel_leaderboards_cached(deps))

    def get_agreement_trends() -> tuple[Response, int]:
        return _summary_response(agreement_trends_cached(deps))

    def get_agreements_summary() -> tuple[Response, int]:
        return _summary_response(agreements_summary_cached(deps))

    def get_filter_options() -> tuple[Response, int]:
        allowed_fields = set(agreements_service.FILTER_OPTION_FIELDS)
        requested_fields_raw = [
            field.strip()
            for field in request.args.getlist("fields")
            if isinstance(field, str) and field.strip()
        ]
        if requested_fields_raw:
            unknown_fields = sorted(set(requested_fields_raw) - allowed_fields)
            if unknown_fields:
                abort(400, description=f"Unsupported filter option fields: {', '.join(unknown_fields)}.")
            requested_fields = tuple(dict.fromkeys(requested_fields_raw))
            payload = agreements_service.filter_options_payload(deps, requested_fields)
        else:
            payload = default_filter_options_cached(deps)
        resp = jsonify(payload)
        resp.headers["Cache-Control"] = f"public, max-age={FILTER_OPTIONS_TTL_SECONDS}"
        return resp, 200

    def get_filter_option_values(field_name: str) -> tuple[Response, int]:
        normalized_field_name = field_name.strip().lower()
        if normalized_field_name not in {"target", "acquirer"}:
            abort(404)
        limit_raw = request.args.get("limit")
        try:
            limit = int(limit_raw) if limit_raw is not None else 100
        except (TypeError, ValueError):
            abort(400, description="Invalid limit.")
        options = agreements_service.filter_option_values(
            deps,
            field_name=normalized_field_name,
            query=str(request.args.get("query") or "").strip(),
            limit=max(1, min(limit, 200)),
        )
        return jsonify({"options": options}), 200

    target_app.add_url_rule("/v1/agreements-index", view_func=get_agreements_index, methods=["GET"])
    target_app.add_url_rule("/v1/agreements-summary", view_func=get_agreements_summary, methods=["GET"])
    target_app.add_url_rule(
        "/v1/counsel-leaderboards", view_func=get_counsel_leaderboards, methods=["GET"]
    )
    target_app.add_url_rule("/v1/agreement-trends", view_func=get_agreement_trends, methods=["GET"])
    target_app.add_url_rule(
        "/v1/agreements-status-summary", view_func=get_agreements_status_summary, methods=["GET"]
    )
    target_app.add_url_rule(
        "/v1/agreements-deal-types-summary",
        view_func=get_agreements_deal_types_summary,
        methods=["GET"],
    )
    target_app.add_url_rule("/v1/filter-options", view_func=get_filter_options, methods=["GET"])
    target_app.add_url_rule(
        "/v1/filter-options/<string:field_name>",
        view_func=get_filter_option_values,
        methods=["GET"],
    )

    return agreements_blp, sections_blp, agreement_search_blp
