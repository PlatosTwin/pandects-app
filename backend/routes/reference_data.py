from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, cast

from flask import Response, jsonify
from flask.views import MethodView
from flask_smorest import Blueprint

from backend.routes.deps import ReferenceDataDeps
from backend.schemas.public_api import CounselResponseSchema, DumpEntrySchema, NaicsResponseSchema


def register_reference_data_routes(
    *,
    deps: ReferenceDataDeps,
) -> tuple[Blueprint, Blueprint, Blueprint, Blueprint, Blueprint]:
    taxonomy_blp = Blueprint(
        "taxonomy",
        "taxonomy",
        url_prefix="/v1/taxonomy",
        description="Access the Pandects agreement taxonomy",
    )

    naics_blp = Blueprint(
        "naics",
        "naics",
        url_prefix="/v1/naics",
        description="Access the NAICS sector and subsector taxonomy",
    )

    counsel_blp = Blueprint(
        "counsel",
        "counsel",
        url_prefix="/v1/counsel",
        description="Access canonical counsel names",
    )

    dumps_blp = Blueprint(
        "dumps",
        "dumps",
        url_prefix="/v1/dumps",
        description="Access metadata about bulk data on Cloudflare",
    )

    tax_clause_taxonomy_blp = Blueprint(
        "tax-clause-taxonomy",
        "tax-clause-taxonomy",
        url_prefix="/v1/taxonomy/tax-clauses",
        description="Access the Pandects tax-clause taxonomy",
    )

    def _build_taxonomy_tree(*, l1_model: object, l2_model: object, l3_model: object) -> dict[str, object]:
        db = deps.db
        l1_rows = cast(
            list[tuple[object, object]],
            db.session.query(
                cast(Any, l1_model).standard_id,
                cast(Any, l1_model).label,
            ).all(),
        )
        l2_rows = cast(
            list[tuple[object, object, object]],
            db.session.query(
                cast(Any, l2_model).standard_id,
                cast(Any, l2_model).label,
                cast(Any, l2_model).parent_id,
            ).all(),
        )
        l3_rows = cast(
            list[tuple[object, object, object]],
            db.session.query(
                cast(Any, l3_model).standard_id,
                cast(Any, l3_model).label,
                cast(Any, l3_model).parent_id,
            ).all(),
        )

        l2_by_parent: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for standard_id, label, parent_id in l2_rows:
            if (
                not isinstance(standard_id, str)
                or not isinstance(parent_id, str)
                or not isinstance(label, str)
            ):
                raise ValueError("taxonomy_l2 has invalid parent_id or label.")
            l2_by_parent[parent_id].append((standard_id, label))

        l3_by_parent: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for standard_id, label, parent_id in l3_rows:
            if (
                not isinstance(standard_id, str)
                or not isinstance(parent_id, str)
                or not isinstance(label, str)
            ):
                raise ValueError("taxonomy_l3 has invalid parent_id or label.")
            l3_by_parent[parent_id].append((standard_id, label))

        validated_l1_rows: list[tuple[str, str]] = []
        for standard_id, label in l1_rows:
            if not isinstance(standard_id, str) or not isinstance(label, str):
                raise ValueError("taxonomy_l1 has invalid standard_id or label.")
            validated_l1_rows.append((standard_id, label))

        tree: dict[str, object] = {}
        for l1_standard_id, l1_label in sorted(validated_l1_rows, key=lambda r: r[1]):
            l2_children: dict[str, object] = {}
            for l2_standard_id, l2_label in sorted(l2_by_parent.get(l1_standard_id, []), key=lambda r: r[1]):
                l3_children: dict[str, object] = {}
                for l3_standard_id, l3_label in sorted(l3_by_parent.get(l2_standard_id, []), key=lambda r: r[1]):
                    l3_children[l3_label] = {"id": l3_standard_id}
                l2_children[l2_label] = {"id": l2_standard_id, "children": l3_children}
            tree[l1_label] = {"id": l1_standard_id, "children": l2_children}

        return tree

    def _taxonomy_tree() -> dict[str, object]:
        return _build_taxonomy_tree(
            l1_model=deps.TaxonomyL1,
            l2_model=deps.TaxonomyL2,
            l3_model=deps.TaxonomyL3,
        )

    def _tax_clause_taxonomy_tree() -> dict[str, object]:
        return _build_taxonomy_tree(
            l1_model=deps.TaxClauseTaxonomyL1,
            l2_model=deps.TaxClauseTaxonomyL2,
            l3_model=deps.TaxClauseTaxonomyL3,
        )

    def _naics_tree() -> dict[str, object]:
        db = deps.db
        naics_sector = deps.NaicsSector
        naics_sub_sector = deps.NaicsSubSector
        sector_rows = cast(
            list[tuple[object, object, object, object]],
            db.session.query(
                naics_sector.sector_code,
                naics_sector.sector_desc,
                naics_sector.sector_group,
                naics_sector.super_sector,
            ).all(),
        )
        sub_sector_rows = cast(
            list[tuple[object, object, object]],
            db.session.query(
                naics_sub_sector.sub_sector_code,
                naics_sub_sector.sub_sector_desc,
                naics_sub_sector.sector_code,
            ).all(),
        )

        sector_by_code: dict[int, dict[str, object]] = {}
        for sector_code, sector_desc, sector_group, super_sector in sector_rows:
            if not isinstance(sector_code, int):
                raise ValueError("naics_sectors.sector_code must be an integer.")
            if not isinstance(sector_desc, str):
                raise ValueError("naics_sectors.sector_desc must be a string.")
            if not isinstance(sector_group, str):
                raise ValueError("naics_sectors.sector_group must be a string.")
            if not isinstance(super_sector, str):
                raise ValueError("naics_sectors.super_sector must be a string.")
            sector_by_code[sector_code] = {
                "sector_code": str(sector_code),
                "sector_desc": sector_desc,
                "sector_group": sector_group,
                "super_sector": super_sector,
                "sub_sectors": [],
            }

        for sub_sector_code, sub_sector_desc, sector_code in sub_sector_rows:
            if not isinstance(sector_code, int):
                raise ValueError("naics_sub_sectors.sector_code must be an integer.")
            if not isinstance(sub_sector_code, int):
                raise ValueError("naics_sub_sectors.sub_sector_code must be an integer.")
            if not isinstance(sub_sector_desc, str):
                raise ValueError("naics_sub_sectors.sub_sector_desc must be a string.")
            parent = sector_by_code.get(sector_code)
            if parent is None:
                raise ValueError("naics_sub_sectors has sector_code with no matching sector.")
            parent_sub_sectors = cast(list[dict[str, str]], parent["sub_sectors"])
            parent_sub_sectors.append(
                {
                    "sub_sector_code": str(sub_sector_code),
                    "sub_sector_desc": sub_sector_desc,
                }
            )

        sorted_sector_codes = sorted(sector_by_code.keys())
        sectors: list[dict[str, object]] = []
        for code in sorted_sector_codes:
            sector = sector_by_code[code]
            sub_sectors = cast(list[dict[str, str]], sector["sub_sectors"])
            sector["sub_sectors"] = sorted(
                sub_sectors,
                key=lambda sub_sector: int(sub_sector["sub_sector_code"]),
            )
            sectors.append(sector)

        return {"sectors": sectors}

    def _get_naics_payload_cached() -> tuple[dict[str, object], bool]:
        now = deps.time.time()
        with deps._naics_lock:
            cached_payload = deps._naics_cache["payload"]
            cached_ts = deps._naics_cache["ts"]
            cache_is_valid = cached_payload is not None and (
                now - cached_ts < deps._NAICS_TTL_SECONDS
            )
        if cache_is_valid and cached_payload is not None:
            return cached_payload, True

        payload = _naics_tree()
        with deps._naics_lock:
            deps._naics_cache["payload"] = payload
            deps._naics_cache["ts"] = now

        return payload, False

    def _counsel_payload() -> dict[str, object]:
        db = deps.db
        counsel = deps.Counsel
        rows = cast(
            list[tuple[object, object]],
            db.session.query(
                counsel.counsel_id,
                counsel.canonical_name,
            )
            .order_by(counsel.canonical_name.asc(), counsel.counsel_id.asc())
            .all(),
        )

        payload_rows: list[dict[str, object]] = []
        for counsel_id, canonical_name in rows:
            if not isinstance(counsel_id, int):
                raise ValueError("counsel.counsel_id must be an integer.")
            if not isinstance(canonical_name, str):
                raise ValueError("counsel.canonical_name must be a string.")
            payload_rows.append(
                {
                    "counsel_id": counsel_id,
                    "canonical_name": canonical_name,
                }
            )
        return {"counsel": payload_rows}

    def _get_counsel_payload_cached() -> tuple[dict[str, object], bool]:
        now = deps.time.time()
        with deps._counsel_lock:
            cached_payload = deps._counsel_cache["payload"]
            cached_ts = deps._counsel_cache["ts"]
            cache_is_valid = cached_payload is not None and (
                now - cached_ts < deps._COUNSEL_TTL_SECONDS
            )
        if cache_is_valid and cached_payload is not None:
            return cached_payload, True

        payload = _counsel_payload()
        with deps._counsel_lock:
            deps._counsel_cache["payload"] = payload
            deps._counsel_cache["ts"] = now

        return payload, False

    def _get_taxonomy_payload_cached() -> tuple[dict[str, object], bool]:
        now = deps.time.time()
        with deps._taxonomy_lock:
            cached_payload = deps._taxonomy_cache["payload"]
            cached_ts = deps._taxonomy_cache["ts"]
            cache_is_valid = cached_payload is not None and (
                now - cached_ts < deps._TAXONOMY_TTL_SECONDS
            )
        if cache_is_valid and cached_payload is not None:
            return cached_payload, True

        payload = _taxonomy_tree()
        with deps._taxonomy_lock:
            deps._taxonomy_cache["payload"] = payload
            deps._taxonomy_cache["ts"] = now

        return payload, False

    def _get_tax_clause_taxonomy_payload_cached() -> tuple[dict[str, object], bool]:
        now = deps.time.time()
        with deps._tax_clause_taxonomy_lock:
            cached_payload = deps._tax_clause_taxonomy_cache["payload"]
            cached_ts = deps._tax_clause_taxonomy_cache["ts"]
            cache_is_valid = cached_payload is not None and (
                now - cached_ts < deps._TAXONOMY_TTL_SECONDS
            )
        if cache_is_valid and cached_payload is not None:
            return cached_payload, True

        payload = _tax_clause_taxonomy_tree()
        with deps._tax_clause_taxonomy_lock:
            deps._tax_clause_taxonomy_cache["payload"] = payload
            deps._tax_clause_taxonomy_cache["ts"] = now

        return payload, False

    @taxonomy_blp.route("")
    class TaxonomyResource(MethodView):
        @taxonomy_blp.doc(
            operationId="getTaxonomy",
            summary="Retrieve clause taxonomy",
            description=(
                "Returns the hierarchical Pandects taxonomy tree keyed by standard ID."
            ),
        )
        def get(self) -> Response:
            payload, _ = _get_taxonomy_payload_cached()
            resp = jsonify(payload)
            resp.headers["Cache-Control"] = f"public, max-age={deps._TAXONOMY_TTL_SECONDS}"
            return resp

    @tax_clause_taxonomy_blp.route("")
    class TaxClauseTaxonomyResource(MethodView):
        @tax_clause_taxonomy_blp.doc(
            operationId="getTaxClauseTaxonomy",
            summary="Retrieve tax-clause taxonomy",
            description="Returns the hierarchical Pandects tax-clause taxonomy tree.",
        )
        def get(self) -> Response:
            payload, _ = _get_tax_clause_taxonomy_payload_cached()
            resp = jsonify(payload)
            resp.headers["Cache-Control"] = f"public, max-age={deps._TAXONOMY_TTL_SECONDS}"
            return resp

    @naics_blp.route("")
    class NaicsResource(MethodView):
        @naics_blp.doc(
            operationId="getNaics",
            summary="Retrieve NAICS sectors and subsectors",
            description=(
                "Returns NAICS sectors with nested subsectors."
            ),
        )
        @naics_blp.response(200, NaicsResponseSchema)
        def get(self) -> Response:
            payload, _ = _get_naics_payload_cached()
            resp = jsonify(payload)
            resp.headers["Cache-Control"] = f"public, max-age={deps._NAICS_TTL_SECONDS}"
            return resp

    @counsel_blp.route("")
    class CounselResource(MethodView):
        @counsel_blp.doc(
            operationId="getCounsel",
            summary="Retrieve canonical counsel names",
            description=(
                "Returns canonical counsel IDs and names for counsel entities referenced across agreements."
            ),
        )
        @counsel_blp.response(200, CounselResponseSchema)
        def get(self) -> Response:
            payload, _ = _get_counsel_payload_cached()
            resp = jsonify(payload)
            resp.headers["Cache-Control"] = f"public, max-age={deps._COUNSEL_TTL_SECONDS}"
            return resp

    @dumps_blp.route("")
    class DumpListResource(MethodView):
        @dumps_blp.doc(
            operationId="listDumps",
            summary="List available bulk dumps",
            description=(
                "Returns newest-first metadata for publicly available database dump artifacts."
            ),
        )
        @dumps_blp.response(200, DumpEntrySchema(many=True))
        def get(self) -> list[dict[str, object]]:
            now = deps.time.time()
            with deps._dumps_cache_lock:
                cached_payload = deps._dumps_cache["payload"]
                cached_ts = deps._dumps_cache["ts"]
                cache_is_valid = cached_payload is not None and (
                    now - cached_ts < deps._DUMPS_CACHE_TTL_SECONDS
                )
            if cache_is_valid and cached_payload is not None:
                return cached_payload
            client = deps.client
            if client is None:
                return []
            s3_client = client
            paginator = s3_client.get_paginator("list_objects_v2")
            pages: Iterable[dict[str, object]] = paginator.paginate(
                Bucket=deps.R2_BUCKET_NAME, Prefix="dumps/"
            )

            dumps_map: dict[str, dict[str, str]] = {}
            for page in pages:
                for obj in cast(list[dict[str, object]], page.get("Contents", [])):
                    key = obj.get("Key")
                    if not isinstance(key, str):
                        continue
                    etag = obj.get("ETag")
                    filename = key.rsplit("/", 1)[-1]
                    files: dict[str, str] | None = None

                    if filename.endswith(".sql.gz.manifest.json"):
                        prefix = filename[: -len(".sql.gz.manifest.json")]
                        files = dumps_map.get(prefix)
                        if files is None:
                            files = {}
                            dumps_map[prefix] = files
                        files["manifest"] = key
                        if isinstance(etag, str):
                            files["manifest_etag"] = etag.strip('"')

                    elif filename.endswith(".sql.gz.sha256"):
                        prefix = filename[: -len(".sql.gz.sha256")]
                        files = dumps_map.get(prefix)
                        if files is None:
                            files = {}
                            dumps_map[prefix] = files
                        files["sha256"] = key

                    elif filename.endswith(".sql.gz"):
                        prefix = filename[: -len(".sql.gz")]
                        files = dumps_map.get(prefix)
                        if files is None:
                            files = {}
                            dumps_map[prefix] = files
                        files["sql"] = key

                    elif filename.endswith(".json"):
                        prefix = filename[: -len(".json")]
                        files = dumps_map.get(prefix)
                        if files is None:
                            files = {}
                            dumps_map[prefix] = files
                        files["manifest"] = key

            dump_list: list[dict[str, object]] = []
            for prefix, files in sorted(dumps_map.items(), reverse=True):
                label = prefix.replace("db_dump_", "")
                entry: dict[str, object] = {"timestamp": label}

                if "sql" in files:
                    entry["sql"] = f"{deps.PUBLIC_DEV_BASE}/{files['sql']}"

                if "sha256" in files:
                    entry["sha256_url"] = f"{deps.PUBLIC_DEV_BASE}/{files['sha256']}"

                if "manifest" in files:
                    entry["manifest"] = f"{deps.PUBLIC_DEV_BASE}/{files['manifest']}"
                    manifest_key = files["manifest"]
                    manifest_etag = files.get("manifest_etag")
                    cached_manifest = None
                    now = deps.time.time()
                    if manifest_key and isinstance(manifest_etag, str):
                        with deps._dumps_manifest_cache_lock:
                            cached_manifest = deps._dumps_manifest_cache.get(manifest_key)
                            if cached_manifest is not None:
                                cache_age = now - float(cached_manifest.get("ts", 0.0))
                                if (
                                    cached_manifest.get("etag") != manifest_etag
                                    or cache_age >= deps._DUMPS_MANIFEST_CACHE_TTL_SECONDS
                                ):
                                    cached_manifest = None
                    if cached_manifest is not None:
                        data = cached_manifest.get("payload") or {}
                    else:
                        try:
                            body = s3_client.get_object(
                                Bucket=deps.R2_BUCKET_NAME, Key=files["manifest"]
                            )["Body"].read()
                            parsed_data = cast(object, json.loads(body))
                            data = (
                                cast(dict[str, object], parsed_data)
                                if isinstance(parsed_data, dict)
                                else {}
                            )
                            if manifest_key and isinstance(manifest_etag, str):
                                with deps._dumps_manifest_cache_lock:
                                    deps._dumps_manifest_cache[manifest_key] = {
                                        "etag": manifest_etag,
                                        "payload": data,
                                        "ts": now,
                                    }
                        except Exception as e:
                            entry["warning"] = f"couldn't read manifest: {e}"
                            data = {}
                    if "size_bytes" in data:
                        entry["size_bytes"] = data["size_bytes"]
                    if "sha256" in data:
                        entry["sha256"] = data["sha256"]

                dump_list.append(entry)

            with deps._dumps_cache_lock:
                deps._dumps_cache["payload"] = dump_list
                deps._dumps_cache["ts"] = now

            return dump_list

    return taxonomy_blp, naics_blp, counsel_blp, dumps_blp, tax_clause_taxonomy_blp
