from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, cast

from flask import Flask
from flask_cors import CORS

from backend.extensions import api, db
from backend.models.main_db import (
    main_db_schema_from_env,
    main_db_uri_from_env,
    schema_translate_map,
)


class _FlaskExtension(Protocol):
    def init_app(self, app: Flask) -> object:
        ...


# Always allowed regardless of the CORS_ORIGINS env var — these are the
# production hostnames that must never be accidentally dropped when the env
# var is set to a restricted list.
_ALWAYS_CORS_ORIGINS = (
    "https://pandects.org",
    "https://www.pandects.org",
    "https://docs.pandects.org",
    "https://www.docs.pandects.org",
)

_DEV_CORS_ORIGINS = (
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)

_DEFAULT_CORS_ORIGINS = _ALWAYS_CORS_ORIGINS + _DEV_CORS_ORIGINS
_DB_POOL_RECYCLE_SECONDS = 240


def app_config_map(app: Flask) -> dict[str, object]:
    return cast(dict[str, object], app.config)


def cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "").strip()
    if not raw:
        return list(_DEFAULT_CORS_ORIGINS)
    extra = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    if "*" in extra:
        raise RuntimeError(
            "CORS_ORIGINS cannot include '*' when supports_credentials=True. "
            + "Specify explicit origins instead."
        )
    # Merge env var origins with always-allowed production origins so that
    # setting CORS_ORIGINS in production never silently drops the docs site.
    combined = list(dict.fromkeys([*_ALWAYS_CORS_ORIGINS, *(extra or _DEV_CORS_ORIGINS)]))
    return combined


def normalize_database_uri(uri: str) -> str:
    normalized = uri.strip()
    if normalized.startswith("postgres://"):
        normalized = f"postgresql://{normalized[len('postgres://'):]}"
    if normalized.startswith("postgresql://") and "connect_timeout=" not in normalized:
        joiner = "&" if "?" in normalized else "?"
        normalized = f"{normalized}{joiner}connect_timeout=5"
    return normalized


def effective_auth_database_uri() -> str | None:
    auth_uri = os.environ.get("AUTH_DATABASE_URI")
    auth_uri = auth_uri.strip() if isinstance(auth_uri, str) else ""
    db_url = os.environ.get("DATABASE_URL")
    db_url = db_url.strip() if isinstance(db_url, str) else ""

    raw = auth_uri or db_url
    if not raw:
        return None
    return normalize_database_uri(raw)


def _default_auth_sqlite_uri() -> str:
    return f"sqlite:///{Path(__file__).resolve().parents[1] / 'auth_dev.sqlite'}"


def _resilient_engine_options(existing: dict[str, object]) -> dict[str, object]:
    options = dict(existing)
    _ = options.setdefault("pool_pre_ping", True)
    _ = options.setdefault("pool_recycle", _DB_POOL_RECYCLE_SECONDS)
    return options


def _auth_bind_config(database_uri: str) -> dict[str, object]:
    bind_config = _resilient_engine_options({})
    bind_config["url"] = database_uri
    return bind_config


def configure_auth_bind(target_app: Flask, *, auth_database_uri: str | None) -> None:
    config = app_config_map(target_app)
    if auth_database_uri is not None:
        config["SQLALCHEMY_BINDS"] = {"auth": _auth_bind_config(auth_database_uri)}
    else:
        config["SQLALCHEMY_BINDS"] = {"auth": _auth_bind_config(_default_auth_sqlite_uri())}


def configure_auth_bind_engine_options(target_app: Flask) -> None:
    config = app_config_map(target_app)
    raw_binds = config.get("SQLALCHEMY_BINDS")
    if not isinstance(raw_binds, dict):
        return
    binds = dict(cast(dict[str, object], raw_binds))
    raw_auth_bind = binds.get("auth")
    if isinstance(raw_auth_bind, str):
        binds["auth"] = _auth_bind_config(raw_auth_bind)
    elif isinstance(raw_auth_bind, dict):
        auth_bind = _resilient_engine_options(cast(dict[str, object], raw_auth_bind))
        binds["auth"] = auth_bind
    config["SQLALCHEMY_BINDS"] = binds


def max_content_length() -> int:
    raw = os.environ.get("MAX_CONTENT_LENGTH_BYTES", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError as exc:
            raise RuntimeError("MAX_CONTENT_LENGTH_BYTES must be an integer.") from exc
        if value <= 0:
            raise RuntimeError("MAX_CONTENT_LENGTH_BYTES must be positive.")
        return value
    return 1 * 1024 * 1024


def configure_openapi(target_app: Flask) -> None:
    config = app_config_map(target_app)
    config.update(
        {
            "API_TITLE": "Pandects API",
            "API_VERSION": "v1",
            "OPENAPI_VERSION": "3.0.2",
            "API_SPEC_OPTIONS": {
                "servers": [
                    {
                        "url": "https://api.pandects.org",
                        "description": "Production API",
                    },
                    {
                        "url": "http://localhost:5113",
                        "description": "Local development API",
                    },
                ]
            },
            "OPENAPI_URL_PREFIX": "/",
            "OPENAPI_SWAGGER_UI_PATH": "/swagger-ui",
            "OPENAPI_SWAGGER_UI_URL": "https://cdn.jsdelivr.net/npm/swagger-ui-dist/",
            "MAX_CONTENT_LENGTH": None,
        }
    )
    config["MAX_CONTENT_LENGTH"] = max_content_length()


def configure_main_db(target_app: Flask) -> None:
    config = app_config_map(target_app)
    if "SQLALCHEMY_DATABASE_URI" not in config:
        config["SQLALCHEMY_DATABASE_URI"] = main_db_uri_from_env()
    _ = config.setdefault("MAIN_DB_SCHEMA", main_db_schema_from_env())
    config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    configured_engine_options = config.get("SQLALCHEMY_ENGINE_OPTIONS", {})
    engine_options = (
        dict(cast(dict[str, object], configured_engine_options))
        if isinstance(configured_engine_options, dict)
        else {}
    )
    engine_options = _resilient_engine_options(engine_options)
    raw_execution_options = engine_options.get("execution_options", {})
    execution_options = (
        dict(cast(dict[str, object], raw_execution_options))
        if isinstance(raw_execution_options, dict)
        else {}
    )
    _ = execution_options.setdefault(
        "schema_translate_map",
        schema_translate_map(cast(str | None, config.get("MAIN_DB_SCHEMA"))),
    )
    engine_options["execution_options"] = execution_options
    config["SQLALCHEMY_ENGINE_OPTIONS"] = engine_options
    configure_auth_bind_engine_options(target_app)


def configure_extensions(target_app: Flask) -> None:
    _ = cast(_FlaskExtension, cast(object, api)).init_app(target_app)
    _ = cast(_FlaskExtension, cast(object, db)).init_app(target_app)


def configure_cors(target_app: Flask) -> None:
    _ = CORS(
        target_app,
        resources={r"/v1/*": {"origins": cors_origins()}},
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-CSRF-Token"],
        supports_credentials=True,
    )


def configure_app(
    target_app: Flask,
    *,
    auth_database_uri: str | None,
    config_overrides: dict[str, object] | None = None,
) -> None:
    configure_auth_bind(target_app, auth_database_uri=auth_database_uri)
    configure_openapi(target_app)
    if config_overrides:
        app_config_map(target_app).update(config_overrides)
    configure_main_db(target_app)
    configure_extensions(target_app)
    configure_cors(target_app)
