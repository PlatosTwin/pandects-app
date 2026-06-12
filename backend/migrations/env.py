"""Alembic environment for the auth database.

Resolves the URL the same way the app does (AUTH_DATABASE_URI / DATABASE_URL,
falling back to the local dev SQLite file) and targets the metadata of the
models bound to the "auth" bind key.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import create_engine, pool

# Allow `alembic -c backend/alembic.ini ...` to import the backend package no
# matter the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_ = os.environ.setdefault("SKIP_MAIN_DB_REFLECTION", "1")
# Match backend.app: pick up AUTH_DATABASE_URI / DATABASE_URL from
# backend/.env when the alembic CLI is run locally.
_ = load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
_ = load_dotenv()

from backend.core.config import auth_database_uri_or_default  # noqa: E402
from backend.extensions import db  # noqa: E402
import backend.models  # noqa: E402,F401  (registers models on the auth metadata)

target_metadata = db.metadatas["auth"]


def run_migrations_offline() -> None:
    url = auth_database_uri_or_default()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = auth_database_uri_or_default()
    engine = create_engine(url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # SQLite cannot ALTER most things in place; batch mode rewrites
            # the table instead.
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
