import os

from sqlalchemy import text
from sqlalchemy import inspect
from sqlalchemy.engine.url import make_url

os.environ.setdefault("SKIP_MAIN_DB_REFLECTION", "1")

from app import app, db, _LEGAL_DOCS  # noqa: E402


def main() -> None:
    with app.app_context():
        db.create_all(bind_key="auth")
        engine = db.engines.get("auth")
        if engine is None:
            raise RuntimeError("Auth DB bind is missing.")
        url = make_url(str(engine.url)).render_as_string(hide_password=True)
        inspector = inspect(engine)
        if engine.dialect.name == "postgresql":
            user_columns = {c["name"]: c for c in inspector.get_columns("auth_users")}
            if "email_verified_at" not in user_columns:
                with engine.begin() as conn:
                    conn.execute(text("ALTER TABLE auth_users ADD COLUMN email_verified_at TIMESTAMP"))
                    conn.execute(
                        text(
                            "UPDATE auth_users "
                            "SET email_verified_at = created_at "
                            "WHERE email_verified_at IS NULL"
                        )
                    )

            columns = {c["name"]: c for c in inspector.get_columns("api_keys")}
            prefix_type = columns.get("prefix", {}).get("type")
            prefix_len = getattr(prefix_type, "length", None)
            if isinstance(prefix_len, int) and prefix_len < 18:
                with engine.begin() as conn:
                    conn.execute(
                        text("ALTER TABLE api_keys ALTER COLUMN prefix TYPE VARCHAR(18)")
                    )

            legal_columns = {c["name"]: c for c in inspector.get_columns("legal_acceptances")}
            if "document_hash" not in legal_columns:
                with engine.begin() as conn:
                    conn.execute(text("ALTER TABLE legal_acceptances ADD COLUMN document_hash VARCHAR(64)"))
            if "ip_address" not in legal_columns:
                with engine.begin() as conn:
                    conn.execute(text("ALTER TABLE legal_acceptances ADD COLUMN ip_address VARCHAR(64)"))
            if "user_agent" not in legal_columns:
                with engine.begin() as conn:
                    conn.execute(text("ALTER TABLE legal_acceptances ADD COLUMN user_agent VARCHAR(512)"))

            # Backfill document hashes for existing Terms/Privacy acceptance rows.
            with engine.begin() as conn:
                for doc, meta in _LEGAL_DOCS.items():
                    conn.execute(
                        text(
                            "UPDATE legal_acceptances "
                            "SET document_hash = :hash "
                            "WHERE document = :doc AND version = :version AND document_hash IS NULL"
                        ),
                        {"hash": meta["sha256"], "doc": doc, "version": meta["version"]},
                    )
        expected = {
            "auth_users",
            "api_keys",
            "api_usage_daily",
            "legal_acceptances",
            "auth_signon_events",
        }
        existing = set(inspector.get_table_names())
        missing = sorted(expected - existing)
        if missing:
            raise RuntimeError(
                f"Auth DB initialization failed (missing tables: {', '.join(missing)})."
            )
    print(f"Auth DB initialized ({url}).")


if __name__ == "__main__":
    main()
