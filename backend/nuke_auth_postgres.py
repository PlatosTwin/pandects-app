import argparse
import os
import sys

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine.url import make_url


def _normalize_database_uri(uri: str) -> str:
    normalized = uri.strip()
    if normalized.startswith("postgres://"):
        normalized = f"postgresql://{normalized[len('postgres://'):]}"
    if normalized.startswith("postgresql://") and "connect_timeout=" not in normalized:
        joiner = "&" if "?" in normalized else "?"
        normalized = f"{normalized}{joiner}connect_timeout=5"
    return normalized


def _effective_database_uri(*, explicit_uri: str | None) -> str:
    if isinstance(explicit_uri, str) and explicit_uri.strip():
        return _normalize_database_uri(explicit_uri)

    auth_uri = os.environ.get("AUTH_DATABASE_URI")
    db_url = os.environ.get("DATABASE_URL")
    raw = (auth_uri or db_url or "").strip()
    if not raw:
        raise RuntimeError("Missing AUTH_DATABASE_URI (or DATABASE_URL) for Postgres connection.")
    return _normalize_database_uri(raw)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="nuke_auth_postgres.py",
        description=(
            "Truncate (wipe) all tables in the auth Postgres database. "
            "Uses AUTH_DATABASE_URI (or DATABASE_URL) unless --uri is provided."
        ),
    )
    parser.add_argument("--uri", help="Postgres connection URI (overrides env vars).")
    parser.add_argument(
        "--schema",
        default="public",
        help="Schema to truncate tables from (default: public).",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        help="Optional explicit table list to truncate (default: all tables in schema).",
    )
    parser.add_argument(
        "--include-alembic-version",
        action="store_true",
        help="Also truncate alembic_version if present (default: keep it).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be truncated, without executing.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the safety check prompt.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    uri = _effective_database_uri(explicit_uri=args.uri)

    url = make_url(uri)
    if url.get_backend_name() != "postgresql":
        raise RuntimeError(f"Refusing to truncate non-Postgres database ({url.get_backend_name()}).")

    safe_url = url.render_as_string(hide_password=True)
    engine = create_engine(uri, future=True)

    inspector = inspect(engine)
    schema = args.schema.strip() if isinstance(args.schema, str) else "public"
    schema = schema or "public"

    if args.tables:
        tables = list(dict.fromkeys(args.tables))
    else:
        tables = inspector.get_table_names(schema=schema)

    if not args.include_alembic_version:
        tables = [name for name in tables if name != "alembic_version"]

    if not tables:
        raise RuntimeError(f"No tables found to truncate (schema={schema!r}).")

    tables_sorted = sorted(tables)

    if not args.yes:
        print("About to TRUNCATE the following tables:")
        for name in tables_sorted:
            print(f"  - {schema}.{name}")
        print(f"Target database: {safe_url}")
        print("Re-run with --yes to confirm.")
        return 2

    if args.dry_run:
        print("Dry run: would TRUNCATE the following tables:")
        for name in tables_sorted:
            print(f"  - {schema}.{name}")
        print(f"Target database: {safe_url}")
        return 0

    preparer = engine.dialect.identifier_preparer
    schema_sql = preparer.quote_schema(schema)
    qualified = [f"{schema_sql}.{preparer.quote(name)}" for name in tables_sorted]
    stmt = f"TRUNCATE TABLE {', '.join(qualified)} RESTART IDENTITY CASCADE"

    with engine.begin() as conn:
        conn.execute(text(stmt))

    print(f"Truncated {len(tables_sorted)} tables ({safe_url}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
