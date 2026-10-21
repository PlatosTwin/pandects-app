"""Bulk-dump storage: the R2 client and the active dump version.

The R2 client backs ``/v1/dumps`` and ``/v1/changelog`` (wired through
``ReferenceDataDeps``). The dump version is read from the public bulk host's
``latest.json`` before every request and surfaced as ``X-Pandects-Dump-Hash``
and in payloads that carry ``dump_version``.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Iterable
from typing import Protocol, TypedDict, cast

from boto3.session import Session
from botocore.config import Config as BotoConfig
from flask import Response, current_app, g

from backend.core.caching import ttl_cached

R2_BUCKET_NAME = "pandects-bulk"
R2_ENDPOINT = "https://7b5e7846d94ee35b35e21999fc4fad5b.r2.cloudflarestorage.com"
PUBLIC_DEV_BASE = "https://bulk.pandects.org"

DUMP_VERSION_CACHE_TTL_SECONDS = int(os.environ.get("DUMP_VERSION_CACHE_TTL_SECONDS", "3600"))
# How long to back off after a failed fetch. Without this, an outage of the
# bulk host would make every request re-attempt the fetch and block on its
# timeout — this hook runs before every request.
DUMP_VERSION_FAILURE_TTL_SECONDS = int(
    os.environ.get("DUMP_VERSION_FAILURE_TTL_SECONDS", "120")
)


class DumpVersionInfo(TypedDict):
    hash: str
    dump_ts: str


class S3ListObject(TypedDict, total=False):
    Key: str
    ETag: str


class S3ListPage(TypedDict, total=False):
    Contents: list[S3ListObject]


class S3Paginator(Protocol):
    def paginate(self, *, Bucket: str, Prefix: str) -> Iterable[S3ListPage]:
        ...


class S3BodyReader(Protocol):
    def read(self) -> bytes:
        ...


class S3GetObjectResult(TypedDict):
    Body: S3BodyReader


class S3Client(Protocol):
    def get_paginator(self, operation_name: str) -> S3Paginator:
        ...

    def get_object(self, *, Bucket: str, Key: str) -> S3GetObjectResult:
        ...


class _Boto3SessionLike(Protocol):
    def client(
        self,
        service_name: str,
        *,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        endpoint_url: str,
        config: object,
    ) -> object:
        ...


def build_r2_client() -> S3Client | None:
    """R2 client from ``R2_ACCESS_KEY_ID`` / ``R2_SECRET_ACCESS_KEY``, else None."""
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        return None
    session = cast(_Boto3SessionLike, Session())
    raw_client = session.client(
        service_name="s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=R2_ENDPOINT,
        # Botocore defaults are 60s connect/read; a slow R2 would pin request
        # workers on the routes that read it (/v1/dumps, /v1/changelog).
        config=BotoConfig(
            connect_timeout=10,
            read_timeout=30,
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )
    return cast(S3Client, raw_client)


@ttl_cached(DUMP_VERSION_CACHE_TTL_SECONDS, retry_none_after=DUMP_VERSION_FAILURE_TTL_SECONDS)
def fetch_latest_dump_version() -> DumpVersionInfo | None:
    """The active dump's hash and timestamp, or None when the host is unreachable.

    A failed fetch serves the last good value (or None) for
    ``DUMP_VERSION_FAILURE_TTL_SECONDS`` instead of retrying on every request.
    """
    try:
        url = f"{PUBLIC_DEV_BASE}/dumps/latest.json"
        req = urllib.request.Request(url, headers={"User-Agent": "pandects-api/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            manifest = cast(object, json.loads(resp.read()))
    except Exception:
        return None
    if not isinstance(manifest, dict):
        return None
    manifest_map = cast(dict[str, object], manifest)
    sha256_val = manifest_map.get("sha256")
    dump_ts_val = manifest_map.get("timestamp")
    if not isinstance(sha256_val, str) or not isinstance(dump_ts_val, str):
        return None
    return {"hash": sha256_val, "dump_ts": dump_ts_val}


def populate_dump_version() -> None:
    """before_request hook: stash the dump version on ``g``."""
    # Tests must not depend on (or wait for) the live bulk host.
    if current_app.testing:
        g.dump_version = None
        return
    g.dump_version = fetch_latest_dump_version()


def attach_dump_version_header(response: Response) -> Response:
    info = cast(object, getattr(g, "dump_version", None))
    if isinstance(info, dict):
        hash_value = cast(dict[str, object], info).get("hash")
        if isinstance(hash_value, str):
            response.headers["X-Pandects-Dump-Hash"] = hash_value
    return response


def current_dump_version_token() -> str:
    """Stable token for the active data dump, scoping cached counts to it.

    Returns ``"none"`` outside a request context (e.g. direct unit calls) so the
    cached-count helper stays usable without a live request.
    """
    try:
        dump = cast(object, getattr(g, "dump_version", None))
    except RuntimeError:
        return "none"
    if isinstance(dump, dict):
        hash_value = cast(dict[str, object], dump).get("hash")
        if isinstance(hash_value, str) and hash_value:
            return hash_value
    return "none"


__all__ = [
    "DUMP_VERSION_CACHE_TTL_SECONDS",
    "DUMP_VERSION_FAILURE_TTL_SECONDS",
    "DumpVersionInfo",
    "PUBLIC_DEV_BASE",
    "R2_BUCKET_NAME",
    "R2_ENDPOINT",
    "S3Client",
    "attach_dump_version_header",
    "build_r2_client",
    "current_dump_version_token",
    "fetch_latest_dump_version",
    "populate_dump_version",
]
