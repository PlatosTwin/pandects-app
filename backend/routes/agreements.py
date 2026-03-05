from __future__ import annotations

from datetime import date, datetime
from typing import cast

from flask import Flask, Response, abort, jsonify, request
from flask.views import MethodView
from flask_smorest import Blueprint
from sqlalchemy import and_, asc, func, or_, text

from backend.routes.deps import AgreementsDeps
from backend.schemas.public_api import (
    AgreementArgsPayload,
    AgreementArgsSchema,
    AgreementResponseSchema,
    AgreementsBulkArgsPayload,
    AgreementsBulkArgsSchema,
    AgreementsIndexArgsSchema,
    AgreementsListResponseSchema,
    SectionResponseSchema,
)


def register_agreements_routes(target_app: Flask, *, deps: AgreementsDeps) -> tuple[Blueprint, Blueprint]:
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

    @agreements_blp.route("")
    class AgreementsListResource(MethodView):
        @agreements_blp.doc(
            operationId="listAgreements",
            summary="List agreements with keyset pagination",
            description=(
                "Lists eligible agreements using a base64 cursor. Supports the same agreement-level "
                "filters as `/v1/search` except clause-type taxonomy filtering."
            ),
        )
        @agreements_blp.arguments(AgreementsBulkArgsSchema, location="query")
        @agreements_blp.response(200, AgreementsListResponseSchema)
        def get(self, args: dict[str, object]) -> dict[str, object]:
            ctx = deps._current_access_context()
            parsed_args = cast(AgreementsBulkArgsPayload, cast(object, args))

            if "standard_id" in request.args:
                abort(400, description="The standard_id filter is not supported on /v1/agreements.")

            include_xml = parsed_args["include_xml"]
            if include_xml and not getattr(ctx, "is_authenticated"):
                abort(403, description="Authentication required when include_xml=true.")

            page_size = parsed_args["page_size"]
            if page_size < 1 or page_size > 100:
                page_size = 25

            after_agreement_uuid = deps._decode_agreements_cursor(parsed_args["cursor"])

            years = parsed_args["year"]
            targets = parsed_args["target"]
            acquirers = parsed_args["acquirer"]
            transaction_price_totals = parsed_args["transaction_price_total"]
            transaction_price_stocks = parsed_args["transaction_price_stock"]
            transaction_price_cashes = parsed_args["transaction_price_cash"]
            transaction_price_assets = parsed_args["transaction_price_assets"]
            transaction_considerations = parsed_args["transaction_consideration"]
            target_types = parsed_args["target_type"]
            acquirer_types = parsed_args["acquirer_type"]
            target_industries = parsed_args["target_industry"]
            acquirer_industries = parsed_args["acquirer_industry"]
            deal_statuses = parsed_args["deal_status"]
            attitudes = parsed_args["attitude"]
            deal_types = parsed_args["deal_type"]
            purposes = parsed_args["purpose"]
            target_pes = parsed_args["target_pe"]
            acquirer_pes = parsed_args["acquirer_pe"]
            agreement_uuid = parsed_args["agreement_uuid"]
            section_uuid = parsed_args["section_uuid"]

            agreements = deps.Agreements
            xml = deps.XML
            sections = deps.Sections
            db = deps.db
            year_expr = deps._agreement_year_expr().label("year")
            item_columns = [
                agreements.agreement_uuid.label("agreement_uuid"),
                year_expr,
                agreements.target.label("target"),
                agreements.acquirer.label("acquirer"),
                agreements.filing_date.label("filing_date"),
                agreements.prob_filing.label("prob_filing"),
                agreements.filing_company_name.label("filing_company_name"),
                agreements.filing_company_cik.label("filing_company_cik"),
                agreements.form_type.label("form_type"),
                agreements.exhibit_type.label("exhibit_type"),
                agreements.transaction_price_total.label("transaction_price_total"),
                agreements.transaction_price_stock.label("transaction_price_stock"),
                agreements.transaction_price_cash.label("transaction_price_cash"),
                agreements.transaction_price_assets.label("transaction_price_assets"),
                agreements.transaction_consideration.label("transaction_consideration"),
                agreements.target_type.label("target_type"),
                agreements.acquirer_type.label("acquirer_type"),
                agreements.target_industry.label("target_industry"),
                agreements.acquirer_industry.label("acquirer_industry"),
                agreements.announce_date.label("announce_date"),
                agreements.close_date.label("close_date"),
                agreements.deal_status.label("deal_status"),
                agreements.attitude.label("attitude"),
                agreements.deal_type.label("deal_type"),
                agreements.purpose.label("purpose"),
                agreements.target_pe.label("target_pe"),
                agreements.acquirer_pe.label("acquirer_pe"),
                agreements.url.label("url"),
            ]
            q = (
                db.session.query(*item_columns)
                .join(xml, deps._agreement_latest_xml_join_condition())
            )

            if include_xml:
                q = q.add_columns(xml.xml.label("xml"))

            if years:
                year_filters = tuple(
                    and_(
                        agreements.filing_date >= f"{year:04d}-01-01",
                        agreements.filing_date < f"{year + 1:04d}-01-01",
                    )
                    for year in years
                )
                q = q.filter(or_(*year_filters))

            if targets:
                q = q.filter(agreements.target.in_(targets))
            if acquirers:
                q = q.filter(agreements.acquirer.in_(acquirers))
            if transaction_price_totals:
                q = q.filter(agreements.transaction_price_total.in_(transaction_price_totals))
            if transaction_price_stocks:
                q = q.filter(agreements.transaction_price_stock.in_(transaction_price_stocks))
            if transaction_price_cashes:
                q = q.filter(agreements.transaction_price_cash.in_(transaction_price_cashes))
            if transaction_price_assets:
                q = q.filter(agreements.transaction_price_assets.in_(transaction_price_assets))
            if transaction_considerations:
                q = q.filter(agreements.transaction_consideration.in_(transaction_considerations))
            if target_types:
                q = q.filter(agreements.target_type.in_(target_types))
            if acquirer_types:
                q = q.filter(agreements.acquirer_type.in_(acquirer_types))
            if target_industries:
                q = q.filter(agreements.target_industry.in_(target_industries))
            if acquirer_industries:
                q = q.filter(agreements.acquirer_industry.in_(acquirer_industries))
            if deal_statuses:
                q = q.filter(agreements.deal_status.in_(deal_statuses))
            if attitudes:
                q = q.filter(agreements.attitude.in_(attitudes))
            if deal_types:
                q = q.filter(agreements.deal_type.in_(deal_types))
            if purposes:
                q = q.filter(agreements.purpose.in_(purposes))

            if target_pes:
                db_target_pes: list[int] = []
                for pe in target_pes:
                    if pe == "true":
                        db_target_pes.append(1)
                    elif pe == "false":
                        db_target_pes.append(0)
                if db_target_pes:
                    q = q.filter(agreements.target_pe.in_(db_target_pes))

            if acquirer_pes:
                db_acquirer_pes: list[int] = []
                for pe in acquirer_pes:
                    if pe == "true":
                        db_acquirer_pes.append(1)
                    elif pe == "false":
                        db_acquirer_pes.append(0)
                if db_acquirer_pes:
                    q = q.filter(agreements.acquirer_pe.in_(db_acquirer_pes))

            if agreement_uuid and agreement_uuid.strip():
                q = q.filter(agreements.agreement_uuid == agreement_uuid.strip())

            if section_uuid and section_uuid.strip():
                section_exists = (
                    db.session.query(sections.section_uuid)
                    .filter(
                        sections.agreement_uuid == agreements.agreement_uuid,
                        sections.section_uuid == section_uuid.strip(),
                        sections.xml_version == xml.version,
                    )
                    .exists()
                )
                q = q.filter(section_exists)

            if after_agreement_uuid:
                q = q.filter(agreements.agreement_uuid > after_agreement_uuid)

            rows = cast(
                list[object],
                q.order_by(asc(agreements.agreement_uuid))
                .limit(page_size + 1)
                .all(),
            )
            has_next = len(rows) > page_size
            page_rows = rows[:page_size]

            results: list[dict[str, object]] = []
            for row in page_rows:
                row_map = deps._row_mapping_as_dict(row)
                payload = {
                    "agreement_uuid": row_map.get("agreement_uuid"),
                    "year": row_map.get("year"),
                    "target": row_map.get("target"),
                    "acquirer": row_map.get("acquirer"),
                    "filing_date": row_map.get("filing_date"),
                    "prob_filing": row_map.get("prob_filing"),
                    "filing_company_name": row_map.get("filing_company_name"),
                    "filing_company_cik": row_map.get("filing_company_cik"),
                    "form_type": row_map.get("form_type"),
                    "exhibit_type": row_map.get("exhibit_type"),
                    "transaction_price_total": row_map.get("transaction_price_total"),
                    "transaction_price_stock": row_map.get("transaction_price_stock"),
                    "transaction_price_cash": row_map.get("transaction_price_cash"),
                    "transaction_price_assets": row_map.get("transaction_price_assets"),
                    "transaction_consideration": row_map.get("transaction_consideration"),
                    "target_type": row_map.get("target_type"),
                    "acquirer_type": row_map.get("acquirer_type"),
                    "target_industry": row_map.get("target_industry"),
                    "acquirer_industry": row_map.get("acquirer_industry"),
                    "announce_date": row_map.get("announce_date"),
                    "close_date": row_map.get("close_date"),
                    "deal_status": row_map.get("deal_status"),
                    "attitude": row_map.get("attitude"),
                    "deal_type": row_map.get("deal_type"),
                    "purpose": row_map.get("purpose"),
                    "target_pe": row_map.get("target_pe"),
                    "acquirer_pe": row_map.get("acquirer_pe"),
                    "url": row_map.get("url"),
                }
                if include_xml:
                    payload["xml"] = row_map.get("xml")
                results.append(payload)

            next_cursor: str | None = None
            if has_next:
                last_row = deps._row_mapping_as_dict(page_rows[-1])
                last_agreement_uuid = last_row.get("agreement_uuid")
                if not isinstance(last_agreement_uuid, str) or not last_agreement_uuid:
                    raise RuntimeError("Agreements list query returned a row without agreement_uuid.")
                next_cursor = deps._encode_agreements_cursor(last_agreement_uuid)

            return {
                "results": results,
                "access": {
                    "tier": getattr(ctx, "tier"),
                    "message": None
                    if getattr(ctx, "is_authenticated")
                    else "XML access requires authentication. Use include_xml=true with a signed-in user or API key.",
                },
                "page_size": page_size,
                "returned_count": len(results),
                "has_next": has_next,
                "next_cursor": next_cursor,
            }

    @agreements_blp.route("/<string:agreement_uuid>")
    class AgreementResource(MethodView):
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
            neighbor_sections_int = parsed_args["neighbor_sections"]

            agreements = deps.Agreements
            xml = deps.XML
            db = deps.db
            year_expr = deps._agreement_year_expr().label("year")
            row = (
                db.session.query(
                    year_expr,
                    agreements.target,
                    agreements.acquirer,
                    agreements.filing_date,
                    agreements.prob_filing,
                    agreements.filing_company_name,
                    agreements.filing_company_cik,
                    agreements.form_type,
                    agreements.exhibit_type,
                    agreements.transaction_price_total,
                    agreements.transaction_price_stock,
                    agreements.transaction_price_cash,
                    agreements.transaction_price_assets,
                    agreements.transaction_consideration,
                    agreements.target_type,
                    agreements.acquirer_type,
                    agreements.target_industry,
                    agreements.acquirer_industry,
                    agreements.announce_date,
                    agreements.close_date,
                    agreements.deal_status,
                    agreements.attitude,
                    agreements.deal_type,
                    agreements.purpose,
                    agreements.target_pe,
                    agreements.acquirer_pe,
                    agreements.url,
                    xml.xml,
                )
                .join(xml, deps._agreement_latest_xml_join_condition())
                .filter(agreements.agreement_uuid == agreement_uuid)
                .first()
            )

            if row is None:
                abort(404)

            row_map = deps._row_mapping_as_dict(cast(object, row))
            xml_content_obj = row_map.get("xml")
            xml_content = xml_content_obj if isinstance(xml_content_obj, str) else ""
            payload = {
                "year": row_map.get("year"),
                "target": row_map.get("target"),
                "acquirer": row_map.get("acquirer"),
                "filing_date": row_map.get("filing_date"),
                "prob_filing": row_map.get("prob_filing"),
                "filing_company_name": row_map.get("filing_company_name"),
                "filing_company_cik": row_map.get("filing_company_cik"),
                "form_type": row_map.get("form_type"),
                "exhibit_type": row_map.get("exhibit_type"),
                "transaction_price_total": row_map.get("transaction_price_total"),
                "transaction_price_stock": row_map.get("transaction_price_stock"),
                "transaction_price_cash": row_map.get("transaction_price_cash"),
                "transaction_price_assets": row_map.get("transaction_price_assets"),
                "transaction_consideration": row_map.get("transaction_consideration"),
                "target_type": row_map.get("target_type"),
                "acquirer_type": row_map.get("acquirer_type"),
                "target_industry": row_map.get("target_industry"),
                "acquirer_industry": row_map.get("acquirer_industry"),
                "announce_date": row_map.get("announce_date"),
                "close_date": row_map.get("close_date"),
                "deal_status": row_map.get("deal_status"),
                "attitude": row_map.get("attitude"),
                "deal_type": row_map.get("deal_type"),
                "purpose": row_map.get("purpose"),
                "target_pe": row_map.get("target_pe"),
                "acquirer_pe": row_map.get("acquirer_pe"),
                "url": row_map.get("url"),
            }
            if not getattr(ctx, "is_authenticated"):
                redacted_xml = deps._redact_agreement_xml(
                    xml_content,
                    focus_section_uuid=focus_section_uuid,
                    neighbor_sections=neighbor_sections_int,
                )
                payload["xml"] = redacted_xml
                payload["is_redacted"] = True
                return payload
            payload["xml"] = xml_content
            return payload

    @sections_blp.route("/<string:section_uuid>")
    class SectionResource(MethodView):
        @sections_blp.response(200, SectionResponseSchema)
        def get(self, section_uuid: str) -> dict[str, object]:
            section_uuid = section_uuid.strip()
            if not deps._SECTION_ID_RE.match(section_uuid):
                abort(400, description="Invalid section_uuid.")

            sections = deps.Sections
            xml = deps.XML
            db = deps.db
            section_cols = sections.__table__.c
            section_standard_ids_expr = deps._coalesced_section_standard_ids().label(
                "section_standard_ids"
            )
            row = (
                db.session.query(
                    section_cols["agreement_uuid"].label("agreement_uuid"),
                    section_cols["section_uuid"].label("section_uuid"),
                    section_standard_ids_expr,
                    section_cols["xml_content"].label("xml_content"),
                    section_cols["article_title"].label("article_title"),
                    section_cols["section_title"].label("section_title"),
                )
                .join(
                    xml,
                    deps._section_latest_xml_join_condition(),
                )
                .filter(section_cols["section_uuid"] == section_uuid)
                .first()
            )

            if row is None:
                abort(404)

            (
                agreement_uuid,
                section_uuid_value,
                section_standard_ids_raw,
                xml_content,
                article_title,
                section_title,
            ) = cast(
                tuple[object, object, object, object, object, object],
                cast(object, row),
            )

            section_standard_ids = deps._parse_section_standard_ids(section_standard_ids_raw)

            return {
                "agreement_uuid": agreement_uuid,
                "section_uuid": section_uuid_value,
                "section_standard_id": section_standard_ids,
                "xml": xml_content,
                "article_title": article_title,
                "section_title": section_title,
            }

    def get_agreements_index() -> dict[str, object]:
        ctx = deps._current_access_context()
        args = deps._load_query(AgreementsIndexArgsSchema())
        page = cast(int, args["page"])
        page_size = cast(int, args["page_size"])
        sort_by = str(args["sort_by"] or "year")
        sort_dir = str(args["sort_dir"] or "desc")
        query = str(args.get("query") or "").strip()

        if page < 1:
            page = 1

        max_page_size = 100 if getattr(ctx, "is_authenticated") else 10
        if page_size < 1 or page_size > max_page_size:
            page_size = min(25, max_page_size)

        agreements = deps.Agreements
        db = deps.db
        year_expr = deps._agreement_year_expr()
        sort_map = {
            "year": year_expr,
            "target": agreements.target,
            "acquirer": agreements.acquirer,
        }
        sort_column = sort_map.get(sort_by, year_expr)
        sort_direction = sort_dir.lower()
        order_by = sort_column.desc() if sort_direction == "desc" else sort_column.asc()

        q = (
            db.session.query(
                agreements.agreement_uuid,
                year_expr.label("year"),
                agreements.target,
                agreements.acquirer,
                agreements.verified,
            )
            .join(deps.XML, deps._agreement_latest_xml_join_condition())
        )
        count_q = (
            db.session.query(func.count(agreements.agreement_uuid))
            .select_from(agreements)
            .join(deps.XML, deps._agreement_latest_xml_join_condition())
        )

        if query:
            if query.isdigit():
                year_value = int(query)
                q = q.filter(year_expr == year_value)
                count_q = count_q.filter(year_expr == year_value)
            else:
                like = f"{query}%"
                filters = or_(
                    agreements.target.ilike(like),
                    agreements.acquirer.ilike(like),
                )
                q = q.filter(filters)
                count_q = count_q.filter(filters)

        q = q.order_by(order_by, agreements.agreement_uuid)

        total_count = deps._to_int(cast(object, count_q.scalar()))
        offset = (page - 1) * page_size
        items = q.offset(offset).limit(page_size).all()
        meta = deps._pagination_metadata(total_count=total_count, page=page, page_size=page_size)

        results: list[dict[str, object]] = []
        for row in items:
            row_map = deps._row_mapping_as_dict(cast(object, row))
            verified_value = row_map.get("verified")
            results.append(
                {
                    "agreement_uuid": row_map.get("agreement_uuid"),
                    "year": row_map.get("year"),
                    "target": row_map.get("target"),
                    "acquirer": row_map.get("acquirer"),
                    "consideration_type": None,
                    "total_consideration": None,
                    "target_industry": None,
                    "acquirer_industry": None,
                    "verified": bool(verified_value) if verified_value is not None else False,
                }
            )

        return {"results": results, **meta}

    def get_agreements_status_summary() -> dict[str, object]:
        agreements = deps.Agreements
        db = deps.db
        latest_filing_date = cast(
            object | None,
            db.session.query(func.max(agreements.filing_date))
            .filter(agreements.filing_date.isnot(None), agreements.filing_date != "")
            .scalar(),
        )
        if isinstance(latest_filing_date, (date, datetime)):
            latest_filing_date = latest_filing_date.isoformat()
        elif latest_filing_date is not None:
            latest_filing_date = str(latest_filing_date)
        rows = (
            db.session.execute(
                text(
                    f"""
                    SELECT
                        year,
                        color,
                        current_stage,
                        count
                    FROM {deps._schema_prefix()}agreement_status_summary
                    WHERE year IS NOT NULL
                    ORDER BY year ASC, current_stage ASC, color ASC
                    """
                )
            )
            .mappings()
            .all()
        )

        years: list[dict[str, object]] = []
        for row in rows:
            row_dict = deps._row_mapping_as_dict(cast(object, row))
            years.append(
                {
                    "year": deps._to_int(cast(object, row_dict.get("year"))),
                    "color": row_dict.get("color"),
                    "current_stage": row_dict.get("current_stage"),
                    "count": deps._to_int(cast(object, row_dict.get("count"))),
                }
            )
        return {"years": years, "latest_filing_date": latest_filing_date}

    def get_agreements_deal_types_summary() -> dict[str, object]:
        db = deps.db
        rows = (
            db.session.execute(
                text(
                    f"""
                    SELECT
                        year,
                        deal_type,
                        `count`
                    FROM {deps._schema_prefix()}agreement_deal_type_summary
                    WHERE year IS NOT NULL
                    ORDER BY year ASC, deal_type ASC
                    """
                )
            )
            .mappings()
            .all()
        )

        years: list[dict[str, object]] = []
        for row in rows:
            row_dict = deps._row_mapping_as_dict(cast(object, row))
            years.append(
                {
                    "year": deps._to_int(cast(object, row_dict.get("year"))),
                    "deal_type": str(row_dict.get("deal_type") or "unknown"),
                    "count": deps._to_int(cast(object, row_dict.get("count"))),
                }
            )
        return {"years": years}

    def get_agreements_summary() -> dict[str, int]:
        now = deps.time.time()
        with deps._agreements_summary_lock:
            cached_payload = deps._agreements_summary_cache["payload"]
            cached_ts = deps._agreements_summary_cache["ts"]
            cache_is_valid = cached_payload is not None and (
                now - cached_ts < deps._AGREEMENTS_SUMMARY_TTL_SECONDS
            )
        if cache_is_valid and cached_payload is not None:
            return cached_payload

        db = deps.db
        row = db.session.execute(
            text(
                f"""
                SELECT
                  COALESCE(SUM(count_agreements), 0) AS agreements,
                  COALESCE(SUM(count_sections), 0) AS sections,
                  COALESCE(SUM(count_pages), 0) AS pages
                FROM {deps._schema_prefix()}summary_data
                """
            )
        ).mappings().first()

        row_dict = deps._row_mapping_as_dict(cast(object, row)) if row is not None else {}
        payload = {
            "agreements": deps._to_int(cast(object, row_dict.get("agreements"))),
            "sections": deps._to_int(cast(object, row_dict.get("sections"))),
            "pages": deps._to_int(cast(object, row_dict.get("pages"))),
        }
        with deps._agreements_summary_lock:
            deps._agreements_summary_cache["payload"] = payload
            deps._agreements_summary_cache["ts"] = now

        return payload

    def get_filter_options() -> tuple[Response, int] | Response:
        now = deps.time.time()
        with deps._filter_options_lock:
            cached_payload = deps._filter_options_cache["payload"]
            cached_ts = deps._filter_options_cache["ts"]
            cache_is_valid = cached_payload is not None and (
                now - cached_ts < deps._FILTER_OPTIONS_TTL_SECONDS
            )
        if cache_is_valid:
            resp = jsonify(cached_payload)
            resp.headers["Cache-Control"] = (
                f"public, max-age={deps._FILTER_OPTIONS_TTL_SECONDS}"
            )
            return resp, 200

        db = deps.db
        schema_prefix = deps._schema_prefix
        _xml_eligible = (
            "EXISTS ("
            "  SELECT 1 FROM {t}xml x "
            "  WHERE x.agreement_uuid = a.agreement_uuid "
            "    AND (x.status IS NULL OR x.status = 'verified')"
            ")"
        ).format(t=schema_prefix())
        _has_sections = (
            "EXISTS ("
            "  SELECT 1 FROM {t}sections s "
            "  WHERE s.agreement_uuid = a.agreement_uuid"
            ")"
        ).format(t=schema_prefix())

        targets = [
            cast(str, row[0])
            for row in db.session.execute(
                text(
                    f"""
                    SELECT DISTINCT a.target
                    FROM {schema_prefix()}agreements a
                    WHERE a.target IS NOT NULL
                      AND a.target <> ''
                      AND {_has_sections}
                      AND {_xml_eligible}
                    ORDER BY a.target
                    """
                )
            ).fetchall()
        ]
        acquirers = [
            cast(str, row[0])
            for row in db.session.execute(
                text(
                    f"""
                    SELECT DISTINCT a.acquirer
                    FROM {schema_prefix()}agreements a
                    WHERE a.acquirer IS NOT NULL
                      AND a.acquirer <> ''
                      AND {_has_sections}
                      AND {_xml_eligible}
                    ORDER BY a.acquirer
                    """
                )
            ).fetchall()
        ]
        target_industries = [
            cast(str, row[0])
            for row in db.session.execute(
                text(
                    f"""
                    SELECT DISTINCT a.target_industry
                    FROM {schema_prefix()}agreements a
                    WHERE a.target_industry IS NOT NULL
                      AND a.target_industry <> ''
                      AND {_has_sections}
                      AND {_xml_eligible}
                    ORDER BY a.target_industry
                    """
                )
            ).fetchall()
        ]
        acquirer_industries = [
            cast(str, row[0])
            for row in db.session.execute(
                text(
                    f"""
                    SELECT DISTINCT a.acquirer_industry
                    FROM {schema_prefix()}agreements a
                    WHERE a.acquirer_industry IS NOT NULL
                      AND a.acquirer_industry <> ''
                      AND {_has_sections}
                      AND {_xml_eligible}
                    ORDER BY a.acquirer_industry
                    """
                )
            ).fetchall()
        ]

        payload = {
            "targets": targets,
            "acquirers": acquirers,
            "target_industries": target_industries,
            "acquirer_industries": acquirer_industries,
        }
        with deps._filter_options_lock:
            deps._filter_options_cache["payload"] = payload
            deps._filter_options_cache["ts"] = now

        resp = jsonify(payload)
        resp.headers["Cache-Control"] = f"public, max-age={deps._FILTER_OPTIONS_TTL_SECONDS}"
        return resp, 200

    target_app.add_url_rule(
        "/v1/agreements-index", view_func=get_agreements_index, methods=["GET"]
    )
    target_app.add_url_rule(
        "/v1/agreements-summary", view_func=get_agreements_summary, methods=["GET"]
    )
    target_app.add_url_rule(
        "/v1/agreements-status-summary",
        view_func=get_agreements_status_summary,
        methods=["GET"],
    )
    target_app.add_url_rule(
        "/v1/agreements-deal-types-summary",
        view_func=get_agreements_deal_types_summary,
        methods=["GET"],
    )
    target_app.add_url_rule(
        "/v1/filter-options", view_func=get_filter_options, methods=["GET"]
    )

    return agreements_blp, sections_blp
