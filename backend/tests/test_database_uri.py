import unittest

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

from backend.core.config import normalize_database_uri
from backend.nuke_auth_postgres import _normalize_database_uri as nuke_normalize


class NormalizeDatabaseUriTests(unittest.TestCase):
    """Bare Postgres URIs must load the installed psycopg2 driver.

    SQLAlchemy 2.1 maps a bare ``postgresql://`` URI to psycopg 3, which the
    backend does not install, so every worker would fail at boot.
    """

    def _check(self, normalize: object) -> None:
        assert callable(normalize)
        for raw in (
            "postgres://u:p@h:5432/db",
            "postgresql://u:p@h:5432/db",
            "  postgres://u:p@h/db?sslmode=disable  ",
        ):
            with self.subTest(normalize=normalize, raw=raw):
                uri = normalize(raw)
                url = make_url(uri)
                self.assertEqual(url.drivername, "postgresql+psycopg2")
                self.assertEqual(url.query.get("connect_timeout"), "5")
                self.assertEqual(url.host, "h")
                self.assertEqual(url.database, "db")
                # Builds an engine without connecting; fails if the driver is missing.
                self.assertEqual(create_engine(uri).dialect.driver, "psycopg2")

    def test_app_normalizer_pins_psycopg2(self) -> None:
        self._check(normalize_database_uri)

    def test_nuke_script_normalizer_pins_psycopg2(self) -> None:
        self._check(nuke_normalize)

    def test_existing_query_and_timeout_are_kept(self) -> None:
        uri = normalize_database_uri("postgres://u:p@h/db?sslmode=require&connect_timeout=9")
        url = make_url(uri)
        self.assertEqual(url.drivername, "postgresql+psycopg2")
        self.assertEqual(url.query.get("sslmode"), "require")
        self.assertEqual(url.query.get("connect_timeout"), "9")

    def test_explicit_driver_and_sqlite_are_unchanged(self) -> None:
        self.assertEqual(
            normalize_database_uri("postgresql+psycopg2://u:p@h/db?connect_timeout=3"),
            "postgresql+psycopg2://u:p@h/db?connect_timeout=3",
        )
        self.assertEqual(normalize_database_uri("sqlite:///x.db"), "sqlite:///x.db")


if __name__ == "__main__":
    _ = unittest.main()
