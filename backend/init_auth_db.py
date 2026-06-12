"""Migrate the auth database to the current schema and validate the result.

Runs as the Fly release command (see backend/fly.toml) and behind the
``init-auth-db`` Flask CLI command. All schema changes — including first-time
bootstrap — are Alembic migrations under backend/migrations/.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_ = os.environ.setdefault("SKIP_MAIN_DB_REFLECTION", "1")
# Match backend.app: pick up AUTH_DATABASE_URI / DATABASE_URL from
# backend/.env when run locally (production supplies real env vars).
_ = load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
_ = load_dotenv()

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from sqlalchemy import create_engine, inspect  # noqa: E402
from sqlalchemy.engine.url import make_url  # noqa: E402

from backend.core.config import auth_database_uri_or_default  # noqa: E402

EXPECTED_TABLES = {
    "auth_users",
    "auth_sessions",
    "auth_password_reset_tokens",
    "auth_external_subjects",
    "auth_signon_events",
    "legal_acceptances",
    "api_keys",
    "api_usage_daily",
    "api_usage_hourly",
    "api_request_events",
    "api_usage_daily_ips",
    "auth_oauth_clients",
    "auth_oauth_authorization_codes",
    "auth_oauth_refresh_tokens",
    "auth_oauth_signing_keys",
    "auth_oauth_user_grants",
    "favorite_projects",
    "favorite_project_assignments",
    "favorites",
    "favorite_tags",
    "favorite_tag_assignments",
}


def upgrade_auth_db() -> str:
    """Run ``alembic upgrade head`` and verify the expected tables exist.

    Returns the redacted database URL for logging.
    """
    config = Config(str(Path(__file__).with_name("alembic.ini")))
    command.upgrade(config, "head")

    uri = auth_database_uri_or_default()
    engine = create_engine(uri)
    try:
        inspector = inspect(engine)
        existing = set(inspector.get_table_names())
    finally:
        engine.dispose()
    missing = sorted(EXPECTED_TABLES - existing)
    if missing:
        raise RuntimeError(
            f"Auth DB migration failed (missing tables: {', '.join(missing)})."
        )
    return make_url(uri).render_as_string(hide_password=True)


def main() -> None:
    url = upgrade_auth_db()
    print(f"Auth DB migrated ({url}).")


if __name__ == "__main__":
    main()
