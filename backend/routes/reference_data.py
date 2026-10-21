"""Reference-data routes: taxonomies, NAICS, counsel, bulk dumps, changelog.

Each read-mostly payload is memoized with ``ttl_cached`` (one slot per payload;
the dump-manifest cache is keyed by object key + ETag). The taxonomy trees and
counts are cached per database by ``taxonomy_service`` instead.
"""

from __future__ import annotations

import json
import os
from collections.abc import Hashable, Iterable
from typing import Any, cast

from flask import Response, jsonify
from flask.views import MethodView
from flask_smorest import Blueprint

from backend.core.caching import ttl_cached
from backend.routes.deps import ReferenceDataDeps
from backend.services import taxonomy_service
from backend.schemas.public_api import (
    ChangelogArgsSchema,
    ChangelogResponseSchema,
    CounselResponseSchema,
    DumpEntrySchema,
    NaicsResponseSchema,
    TaxonomyCountsResponseSchema,
)

TAXONOMY_TTL_SECONDS = taxonomy_service.TAXONOMY_TTL_SECONDS
COUNSEL_TTL_SECONDS = int(os.environ.get("COUNSEL_TTL_SECONDS", "21600"))
NAICS_TTL_SECONDS = int(os.environ.get("NAICS_TTL_SECONDS", "21600"))
DUMPS_CACHE_TTL_SECONDS = int(os.environ.get("DUMPS_CACHE_TTL_SECONDS", "300"))
CHANGELOG_CACHE_TTL_SECONDS = int(os.environ.get("CHANGELOG_CACHE_TTL_SECONDS", "300"))
DUMPS_MANIFEST_CACHE_TTL_SECONDS = int(os.environ.get("DUMPS_MANIFEST_CACHE_TTL_SECONDS", "1800"))
# Negative-cache window after a failed R2 read, so an absent/unreachable
# changelog.json doesn't turn every request into a live R2 GetObject.
CHANGELOG_FAIL_TTL_SECONDS = 60
_DUMPS_MANIFEST_CACHE_MAX_KEYS = 256


def _single_slot(_deps: ReferenceDataDeps, /) -> Hashable:
    """Cache key for a payload that does not vary by argument (the deps handle)."""
    return ()


@ttl_cached(NAICS_TTL_SECONDS, key=_single_slot)
def naics_payload_cached(deps: ReferenceDataDeps) -> dict[str, object]:
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


@ttl_cached(COUNSEL_TTL_SECONDS, key=_single_slot)
def counsel_payload_cached(deps: ReferenceDataDeps) -> dict[str, object]:
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


def _dump_manifest_key(_client: object, _bucket: str, key: str, etag: str, /) -> Hashable:
    return (key, etag)


@ttl_cached(
    DUMPS_MANIFEST_CACHE_TTL_SECONDS,
    key=_dump_manifest_key,
    max_keys=_DUMPS_MANIFEST_CACHE_MAX_KEYS,
)
def _dump_manifest_cached(client: object, bucket: str, key: str, etag: str) -> dict[str, object]:
    """A dump manifest's JSON, reused while its ETag is unchanged. Raises on read failure."""
    del etag  # part of the cache key only
    return _read_dump_manifest(client, bucket, key)


def _read_dump_manifest(client: object, bucket: str, key: str) -> dict[str, object]:
    body = cast(Any, client).get_object(Bucket=bucket, Key=key)["Body"].read()
    parsed_data = cast(object, json.loads(body))
    return cast(dict[str, object], parsed_data) if isinstance(parsed_data, dict) else {}


@ttl_cached(DUMPS_CACHE_TTL_SECONDS, key=_single_slot)
def dump_list_cached(deps: ReferenceDataDeps) -> list[dict[str, object]]:
    """Newest-first dump entries built from the R2 `dumps/` listing."""
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
            # changelog.json / changelog_<ts>.json live under dumps/
            # but are not dump artifacts; without this skip they'd be
            # misread as manifests of nonexistent dumps.
            if filename == "changelog.json" or filename.startswith("changelog_"):
                continue
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
            try:
                if manifest_key and isinstance(manifest_etag, str):
                    data = _dump_manifest_cached(
                        s3_client, deps.R2_BUCKET_NAME, manifest_key, manifest_etag
                    )
                else:
                    data = _read_dump_manifest(s3_client, deps.R2_BUCKET_NAME, manifest_key)
            except Exception as e:
                entry["warning"] = f"couldn't read manifest: {e}"
                data = {}
            if "size_bytes" in data:
                entry["size_bytes"] = data["size_bytes"]
            if "sha256" in data:
                entry["sha256"] = data["sha256"]
            if "changelog_url" in data:
                entry["changelog_url"] = data["changelog_url"]

        dump_list.append(entry)

    return dump_list


@ttl_cached(
    CHANGELOG_CACHE_TTL_SECONDS,
    key=_single_slot,
    retry_none_after=CHANGELOG_FAIL_TTL_SECONDS,
)
def changelog_payload_cached(deps: ReferenceDataDeps) -> dict[str, object] | None:
    """The published changelog JSON, or None when R2 is unavailable.

    The changelog is served from the published R2 artifact (not the repo
    file): code deploys continuously while data refreshes ~monthly, so only
    the R2 copy is guaranteed to describe published dumps. A failed read
    keeps serving the last good copy for ``CHANGELOG_FAIL_TTL_SECONDS``
    (``retry_none_after``): a transient R2 blip must never blank the
    changelog, because {"releases": []} reads as "no new releases".
    """
    client = deps.client
    if client is None:
        return None
    try:
        body = client.get_object(Bucket=deps.R2_BUCKET_NAME, Key="dumps/changelog.json")[
            "Body"
        ].read()
        parsed = cast(object, json.loads(body))
    except Exception:
        return None
    return cast(dict[str, object], parsed) if isinstance(parsed, dict) else None


def register_reference_data_routes(
    *,
    deps: ReferenceDataDeps,
) -> tuple[Blueprint, Blueprint, Blueprint, Blueprint, Blueprint, Blueprint]:
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

    changelog_blp = Blueprint(
        "changelog",
        "changelog",
        url_prefix="/v1/changelog",
        description="Dataset, schema, and API change history per bulk release",
    )

    @taxonomy_blp.route("")
    class TaxonomyResource(MethodView):
        @taxonomy_blp.doc(
            operationId="getTaxonomy",
            summary="Retrieve clause taxonomy",
            description=(
                "Returns the hierarchical Pandects clause taxonomy: nodes keyed by label, "
                "each carrying its `id` (standard ID) and, below L3, its `children`."
            ),
        )
        @taxonomy_blp.response(
            200,
            {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "required": ["id"],
                    "properties": {
                        "id": {"type": "string"},
                        "children": {
                            "type": "object",
                            "additionalProperties": {"type": "object"},
                        },
                    },
                },
            },
        )
        def get(self) -> Response:
            payload = taxonomy_service.taxonomy_tree("clauses")
            resp = jsonify(payload)
            resp.headers["Cache-Control"] = f"public, max-age={TAXONOMY_TTL_SECONDS}"
            return resp

    @tax_clause_taxonomy_blp.route("")
    class TaxClauseTaxonomyResource(MethodView):
        @tax_clause_taxonomy_blp.doc(
            operationId="getTaxClauseTaxonomy",
            summary="Retrieve tax-clause taxonomy",
            description="Returns the hierarchical Pandects tax-clause taxonomy tree.",
        )
        @tax_clause_taxonomy_blp.response(
            200,
            {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "required": ["id"],
                    "properties": {
                        "id": {"type": "string"},
                        "children": {
                            "type": "object",
                            "additionalProperties": {"type": "object"},
                        },
                    },
                },
            },
        )
        def get(self) -> Response:
            payload = taxonomy_service.taxonomy_tree("tax_clauses")
            resp = jsonify(payload)
            resp.headers["Cache-Control"] = f"public, max-age={TAXONOMY_TTL_SECONDS}"
            return resp

    @taxonomy_blp.route("/counts")
    class TaxonomyCountsResource(MethodView):
        @taxonomy_blp.doc(
            operationId="getTaxonomyCounts",
            summary="Section counts per clause-taxonomy node",
            description=(
                "Returns, for every clause-taxonomy node, how many sections "
                "`GET /v1/sections?standard_id=<id>` matches: a parent node counts the "
                "distinct sections tagged with it or any descendant. Nodes with no "
                "sections are omitted."
            ),
        )
        @taxonomy_blp.response(200, TaxonomyCountsResponseSchema)
        def get(self) -> Response:
            resp = jsonify({"counts": taxonomy_service.taxonomy_section_counts()})
            resp.headers["Cache-Control"] = f"public, max-age={TAXONOMY_TTL_SECONDS}"
            return resp

    @tax_clause_taxonomy_blp.route("/counts")
    class TaxClauseTaxonomyCountsResource(MethodView):
        @tax_clause_taxonomy_blp.doc(
            operationId="getTaxClauseTaxonomyCounts",
            summary="Tax-clause counts per tax-clause-taxonomy node",
            description=(
                "Returns, for every tax-clause-taxonomy node, how many clauses "
                "`GET /v1/tax-clauses?tax_standard_id=<id>` matches with default "
                "options (rep-and-warranty clauses excluded): a parent node counts the "
                "distinct clauses tagged with it or any descendant. Nodes with no "
                "clauses are omitted."
            ),
        )
        @tax_clause_taxonomy_blp.response(200, TaxonomyCountsResponseSchema)
        def get(self) -> Response:
            resp = jsonify({"counts": taxonomy_service.tax_clause_taxonomy_clause_counts()})
            resp.headers["Cache-Control"] = f"public, max-age={TAXONOMY_TTL_SECONDS}"
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
            payload = naics_payload_cached(deps)
            resp = jsonify(payload)
            resp.headers["Cache-Control"] = f"public, max-age={NAICS_TTL_SECONDS}"
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
            payload = counsel_payload_cached(deps)
            resp = jsonify(payload)
            resp.headers["Cache-Control"] = f"public, max-age={COUNSEL_TTL_SECONDS}"
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
            return dump_list_cached(deps)

    def _sanitized_changelog_release(release: dict[str, object]) -> dict[str, object]:
        # The payload is external input (an R2 object); anything the response
        # schema would choke on at dump time must be dropped here, never 500.
        # The 500 class is list-typed fields fed a non-iterable (marshmallow's
        # List._serialize iterates) and Int fields fed non-numerics.
        release = dict(release)

        stats = release.get("stats")
        if not isinstance(stats, dict):
            release["stats"] = None
        else:
            row_counts = cast(dict[str, object], stats).get("row_counts")
            clean_counts: dict[str, int] = {}
            if isinstance(row_counts, dict):
                for table, count in cast(dict[object, object], row_counts).items():
                    if (
                        isinstance(table, str)
                        and isinstance(count, int)
                        and not isinstance(count, bool)
                    ):
                        clean_counts[table] = count
            release["stats"] = {"row_counts": clean_counts}

        raw_changes = release.get("changes")
        clean_changes: list[dict[str, object]] = []
        if isinstance(raw_changes, list):
            for change in cast(list[object], raw_changes):
                if not isinstance(change, dict):
                    continue
                change = dict(cast(dict[str, object], change))
                for list_key in ("tables", "refs"):
                    if list_key in change and not isinstance(change[list_key], list):
                        del change[list_key]
                clean_changes.append(change)
        release["changes"] = clean_changes
        return release

    @changelog_blp.route("")
    class ChangelogResource(MethodView):
        @changelog_blp.doc(
            operationId="getChangelog",
            summary="Dataset and schema change history",
            description=(
                "Returns the changelog for the public database dumps, the REST API, and "
                "the MCP surface, newest release first. Each release corresponds to one "
                "published bulk dump; match `dump_sha256` against the "
                "`X-Pandects-Dump-Hash` response header (or the `dump_version.hash` body "
                "field) to see exactly which changes the data you are querying includes. "
                "`latest_version`/`latest_released` always describe the newest published "
                "release, even when `since`/`dump_sha256` filter the `releases` list."
            ),
        )
        @changelog_blp.arguments(ChangelogArgsSchema, location="query")
        @changelog_blp.response(200, ChangelogResponseSchema)
        def get(self, args: dict[str, object]) -> dict[str, object]:
            payload = changelog_payload_cached(deps)
            if payload is None:
                return {"latest_version": None, "latest_released": None, "releases": []}
            raw_releases = payload.get("releases")
            releases = [
                _sanitized_changelog_release(cast(dict[str, object], release))
                for release in (raw_releases if isinstance(raw_releases, list) else [])
                if isinstance(release, dict)
            ]
            dump_sha256 = args.get("dump_sha256")
            if isinstance(dump_sha256, str) and dump_sha256:
                releases = [r for r in releases if r.get("dump_sha256") == dump_sha256]
            since = args.get("since")
            if isinstance(since, str) and since:
                releases = [
                    r
                    for r in releases
                    if isinstance(r.get("version"), str) and cast(str, r["version"]) > since
                ]
            return {
                "latest_version": payload.get("latest_version"),
                "latest_released": payload.get("latest_released"),
                "releases": releases,
            }

    return taxonomy_blp, naics_blp, counsel_blp, dumps_blp, tax_clause_taxonomy_blp, changelog_blp
