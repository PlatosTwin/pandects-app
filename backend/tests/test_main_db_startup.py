import importlib.util
import os
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

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


class MainDbStartupTests(unittest.TestCase):
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
