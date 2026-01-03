import os

from sqlalchemy import inspect
from sqlalchemy.engine.url import make_url

os.environ.setdefault("SKIP_MAIN_DB_REFLECTION", "1")

from backend.app import app, db  # noqa: E402


def main() -> None:
    with app.app_context():
        db.create_all(bind_key="auth")
        engine = db.engines.get("auth")
        if engine is None:
            raise RuntimeError("Auth DB bind is missing.")
        url = make_url(str(engine.url)).render_as_string(hide_password=True)
        inspector = inspect(engine)
        expected = {
            "auth_users",
            "api_keys",
            "api_usage_daily",
            "api_usage_hourly",
            "api_request_events",
            "api_usage_daily_ips",
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
