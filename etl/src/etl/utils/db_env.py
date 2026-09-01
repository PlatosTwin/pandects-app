"""Local MariaDB engine construction from etl/.env, plus SQL-name validators.

Committed home for the small pieces of DB plumbing that maintenance scripts
share. Historically these lived in etl.utils.reset_stuck_agreements, which is
gitignored, so committed modules importing from it could not run from a clean
checkout.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

SCHEMA_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class RuntimeDb:
    engine: Engine
    database: str

    def get_engine(self) -> Engine:
        return self.engine


def _load_local_env() -> None:
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        raise RuntimeError(f"Required environment variable {name} is missing.")
    return value


def build_engine_from_env() -> RuntimeDb:
    _load_local_env()
    user = _require_env("MARIADB_USER")
    password = _require_env("MARIADB_PASSWORD")
    host = _require_env("MARIADB_HOST")
    port = _require_env("MARIADB_PORT")
    database = _require_env("MARIADB_DATABASE")
    engine = create_engine(
        f"mariadb+mysqldb://{user}:{password}@{host}:{port}/{database}"
    )
    return RuntimeDb(engine=engine, database=database)


def validate_schema_name(schema: str) -> str:
    if not SCHEMA_RE.fullmatch(schema):
        raise ValueError(f"Invalid schema name: {schema!r}")
    return schema
