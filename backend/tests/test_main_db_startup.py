import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.exc import SAWarning


_MAIN_DB_PATH = Path(__file__).resolve().parents[1] / "models" / "main_db.py"


def _set_default_env() -> None:
    os.environ.setdefault("SKIP_MAIN_DB_REFLECTION", "1")
    os.environ.setdefault("MARIADB_USER", "root")
    os.environ.setdefault("MARIADB_PASSWORD", "password")
    os.environ.setdefault("MARIADB_HOST", "127.0.0.1")
    os.environ.setdefault("MARIADB_DATABASE", "pdx")


def _load_main_db_module_isolated(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _MAIN_DB_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load backend/models/main_db.py for startup probe.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"This declarative base already contains a class "
                r"with the same class name and module name as .*"
            ),
            category=SAWarning,
        )
        spec.loader.exec_module(module)
    _ = sys.modules.pop(module_name, None)
    return module


_set_default_env()

_NULL_FALLBACK_COLUMNS = ("transaction_price_rollover", "transaction_price_buyer_owned")

# Run in a fresh interpreter: reflect main_db from MAIN_DATABASE_URI (a SQLite file, so
# the schema token maps to None), then call one REST route and print its JSON.
_ROLLOVER_ROUTE_PROBE = """
import json, os
from unittest.mock import patch
import sqlalchemy

_create_engine = sqlalchemy.create_engine


def _sqlite_engine(url, **kwargs):
    options = dict(kwargs.pop("execution_options", {}) or {})
    options["schema_translate_map"] = {key: None for key in options.get("schema_translate_map", {})}
    return _create_engine(url, execution_options=options)


with patch("sqlalchemy.create_engine", _sqlite_engine):
    import backend.models.main_db  # noqa: F401  (reflects the tables once, here)

import backend.app as app_module

app = app_module.create_test_app(
    config_overrides={
        "MAIN_DB_SCHEMA": "",
        "SQLALCHEMY_DATABASE_URI": os.environ["MAIN_DATABASE_URI"],
        "SQLALCHEMY_BINDS": {"auth": os.environ["AUTH_DATABASE_URI"]},
    }
)
with app.app_context():
    app_module.db.create_all(bind_key="auth")
res = app.test_client().get("/v1/agreements?page_size=10")
print(json.dumps({"status": res.status_code, "body": res.get_json()}))
"""


def _sqlite_main_db_engine(*, include_section_text_search: bool):
    fallback = _load_main_db_module_isolated("_main_db_probe_fallback_schema")
    db_file = tempfile.NamedTemporaryFile(prefix="pandects_reflect_", suffix=".sqlite", delete=False)
    db_file.close()
    engine = create_engine(
        f"sqlite:///{db_file.name}",
        execution_options={"schema_translate_map": {fallback.MAIN_SCHEMA_TOKEN: None}},
    )
    tables = [
        table
        for table in fallback.metadata.sorted_tables
        if include_section_text_search or table.name != "section_text_search"
    ]
    fallback.metadata.create_all(engine, tables=tables)
    return engine


def _load_reflected_main_db(module_name: str, engine):
    with patch.dict(
        os.environ,
        {
            "ENABLE_MAIN_DB_REFLECTION": "1",
            "SKIP_MAIN_DB_REFLECTION": "0",
            "MAIN_DATABASE_URI": str(engine.url),
        },
        clear=False,
    ), patch("sqlalchemy.create_engine", return_value=engine):
        return _load_main_db_module_isolated(module_name)


class MainDbStartupTests(unittest.TestCase):
    def test_missing_section_text_search_table_boots_with_text_search_unavailable(self):
        engine = _sqlite_main_db_engine(include_section_text_search=False)
        module_name = "_main_db_probe_missing_text_search"

        with self.assertLogs(module_name, level="WARNING") as logs:
            mod = _load_reflected_main_db(module_name, engine)

        self.assertIs(mod.engine, engine)
        self.assertFalse(mod.SECTION_TEXT_SEARCH_AVAILABLE)
        self.assertIsNone(mod.SectionTextSearch)
        self.assertEqual(mod.LatestSectionsSearch.__table__.name, "latest_sections_search")
        self.assertEqual(mod.Sections.__table__.name, "sections")
        self.assertEqual(len(logs.output), 1)
        self.assertIn("section_text_search table is missing", logs.output[0])

    def test_empty_section_text_search_table_warns_but_stays_available(self):
        engine = _sqlite_main_db_engine(include_section_text_search=True)
        module_name = "_main_db_probe_empty_text_search"

        with self.assertLogs(module_name, level="WARNING") as logs:
            mod = _load_reflected_main_db(module_name, engine)

        self.assertTrue(mod.SECTION_TEXT_SEARCH_AVAILABLE)
        self.assertIsNotNone(mod.SectionTextSearch)
        self.assertEqual(mod.SectionTextSearch.__table__.name, "section_text_search")
        self.assertEqual(len(logs.output), 1)
        self.assertIn("section_text_search table is empty", logs.output[0])

    def test_missing_rollover_column_is_served_as_null(self):
        # The prod schema gets transaction_price_rollover and transaction_price_buyer_owned
        # only with a data push; until then the reflected models must map each to
        # CAST(NULL AS DECIMAL(18,2)).
        engine = _sqlite_main_db_engine(include_section_text_search=True)
        with engine.begin() as conn:
            for table in ("agreements", "latest_sections_search"):
                for column_name in _NULL_FALLBACK_COLUMNS:
                    _ = conn.exec_driver_sql(f"ALTER TABLE {table} DROP COLUMN {column_name}")
        mod = _load_reflected_main_db("_main_db_probe_missing_rollover", engine)

        from sqlalchemy import select
        from sqlalchemy.dialects import mysql

        from backend.filtering import build_transaction_price_bucket_filter

        for model in (mod.Agreements, mod.LatestSectionsSearch):
            for column_name in _NULL_FALLBACK_COLUMNS:
                self.assertNotIn(column_name, model.__table__.c)
                column = getattr(model, column_name)
                bucket = build_transaction_price_bucket_filter(column, ["0 - 100M"])
                query = select(column.label("price")).where(bucket).order_by(column)
                sql = str(query.compile(dialect=mysql.dialect(), compile_kwargs={"literal_binds": True}))
                self.assertIn("CAST(NULL AS DECIMAL(18, 2)) AS price", sql)
                self.assertIn("WHERE CAST(NULL AS DECIMAL(18, 2)) >= 0", sql)
                self.assertIn("ORDER BY CAST(NULL AS DECIMAL(18, 2))", sql)

        with engine.begin() as conn:
            _ = conn.exec_driver_sql(
                "INSERT INTO agreements (agreement_uuid, filing_date, target, acquirer, verified, url, "
                "transaction_price_total, transaction_consideration) VALUES "
                "('a1', '2021-02-01', 'Target A', 'Acquirer A', 1, 'http://example.com/a1', '100', 'cash')"
            )
            _ = conn.exec_driver_sql(
                "INSERT INTO xml (agreement_uuid, xml, version, status, latest) VALUES "
                "('a1', '<document><article><section uuid=\"00000000-0000-0000-0000-000000000001\">"
                "<text>A1</text></section></article></document>', 1, 'verified', 1)"
            )
        # backend.app binds the models at import, so the route check needs a fresh interpreter.
        auth_db = tempfile.NamedTemporaryFile(prefix="pandects_auth_", suffix=".sqlite", delete=False)
        auth_db.close()
        env = {
            **os.environ,
            "ENABLE_MAIN_DB_REFLECTION": "1",
            "SKIP_MAIN_DB_REFLECTION": "0",
            "MAIN_DATABASE_URI": str(engine.url),
            "MAIN_DB_SCHEMA": "",
            "AUTH_DATABASE_URI": f"sqlite:///{auth_db.name}",
            "TEMPORARY_ACCESS_LOCKDOWN": "0",
            "AUTH_SECRET_KEY": "test-auth-secret",
            "PUBLIC_API_BASE_URL": "http://localhost:5000",
            "PUBLIC_FRONTEND_BASE_URL": "http://localhost:8080",
            "AUTH_SESSION_TRANSPORT": "bearer",
        }
        result = subprocess.run(
            [sys.executable, "-c", _ROLLOVER_ROUTE_PROBE],
            cwd=str(_MAIN_DB_PATH.parents[2]),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(payload["status"], 200, payload)
        rows = payload["body"]["results"]
        self.assertEqual([row["agreement_uuid"] for row in rows], ["a1"])
        for column_name in _NULL_FALLBACK_COLUMNS:
            self.assertIn(column_name, rows[0])
            self.assertIsNone(rows[0][column_name])

    def test_missing_buyer_owned_column_alone_is_served_as_null(self):
        # The fallback is per column: a table that has the rollover column but not the
        # buyer-owned one keeps the real rollover column.
        engine = _sqlite_main_db_engine(include_section_text_search=True)
        with engine.begin() as conn:
            for table in ("agreements", "latest_sections_search"):
                _ = conn.exec_driver_sql(f"ALTER TABLE {table} DROP COLUMN transaction_price_buyer_owned")
        mod = _load_reflected_main_db("_main_db_probe_missing_buyer_owned", engine)

        from sqlalchemy import select
        from sqlalchemy.dialects import mysql

        for model in (mod.Agreements, mod.LatestSectionsSearch):
            self.assertIn("transaction_price_rollover", model.__table__.c)
            self.assertNotIn("transaction_price_buyer_owned", model.__table__.c)
            query = select(model.transaction_price_buyer_owned.label("buyer_owned"))
            sql = str(query.compile(dialect=mysql.dialect(), compile_kwargs={"literal_binds": True}))
            self.assertIn("CAST(NULL AS DECIMAL(18, 2)) AS buyer_owned", sql)

    def test_static_fallback_always_exposes_section_text_search(self):
        mod = _load_main_db_module_isolated("_main_db_probe_fallback_text_search")
        self.assertTrue(mod.SECTION_TEXT_SEARCH_AVAILABLE)
        self.assertIsNotNone(mod.SectionTextSearch)

    def test_reflection_enabled_missing_env_vars_fails_fast(self):
        with patch.dict(
            os.environ,
            {
                "ENABLE_MAIN_DB_REFLECTION": "1",
                "SKIP_MAIN_DB_REFLECTION": "0",
                "MAIN_DATABASE_URI": "",
                "MARIADB_USER": "",
                "MARIADB_PASSWORD": "",
                "MARIADB_HOST": "",
                "MARIADB_DATABASE": "",
            },
            clear=False,
        ):
            with self.assertRaises(RuntimeError) as err:
                _ = _load_main_db_module_isolated("_main_db_probe_missing_env")
        message = str(err.exception)
        self.assertIn("MAIN_DATABASE_URI", message)
        self.assertIn("MARIADB_USER", message)
        self.assertIn("MARIADB_PASSWORD", message)
        self.assertIn("MARIADB_HOST", message)
        self.assertIn("MARIADB_DATABASE", message)

    def test_reflection_disabled_does_not_require_mariadb_env_on_import(self):
        with patch.dict(
            os.environ,
            {
                "ENABLE_MAIN_DB_REFLECTION": "0",
                "SKIP_MAIN_DB_REFLECTION": "1",
                "MAIN_DATABASE_URI": "",
                "MARIADB_USER": "",
                "MARIADB_PASSWORD": "",
                "MARIADB_HOST": "",
                "MARIADB_DATABASE": "",
            },
            clear=False,
        ):
            mod = _load_main_db_module_isolated("_main_db_probe_no_reflection")
            self.assertFalse(mod.ENABLE_MAIN_DB_REFLECTION)
            self.assertTrue(mod.SKIP_MAIN_DB_REFLECTION)
            self.assertIsNone(mod.engine)


if __name__ == "__main__":
    unittest.main()
