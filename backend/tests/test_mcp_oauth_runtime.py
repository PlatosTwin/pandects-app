from __future__ import annotations

import os
import tempfile
import unittest
from types import ModuleType
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from flask import Flask


def _set_default_env(*, auth_db_uri: str) -> None:
    os.environ["SKIP_MAIN_DB_REFLECTION"] = "1"
    os.environ["MARIADB_USER"] = "root"
    os.environ["MARIADB_PASSWORD"] = "password"
    os.environ["MARIADB_HOST"] = "127.0.0.1"
    os.environ["MARIADB_DATABASE"] = "pdx"
    os.environ["AUTH_SECRET_KEY"] = "test-auth-secret"
    os.environ["PUBLIC_API_BASE_URL"] = "http://localhost:5000"
    os.environ["PUBLIC_FRONTEND_BASE_URL"] = "http://localhost:8080"
    os.environ["AUTH_DATABASE_URI"] = auth_db_uri


class McpOAuthRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        auth_db = tempfile.NamedTemporaryFile(prefix="pandects_mcp_oauth_", suffix=".sqlite", delete=False)
        auth_db.close()
        _set_default_env(auth_db_uri=f"sqlite:///{auth_db.name}")
        import backend.auth.mcp_oauth_runtime as mcp_oauth_runtime
        from backend.app import create_test_app

        cls.runtime: ModuleType = mcp_oauth_runtime
        cls.app: Flask = create_test_app(
            config_overrides={
                "SQLALCHEMY_DATABASE_URI": "sqlite://",
                "SQLALCHEMY_BINDS": {"auth": f"sqlite:///{auth_db.name}"},
            }
        )

    def test_oauth_metadata_includes_required_oidc_arrays(self) -> None:
        payload = self.runtime.mcp_oauth_metadata()

        self.assertEqual(payload["issuer"], "http://localhost:5000/v1/auth/oauth")
        self.assertEqual(payload["subject_types_supported"], ["public"])
        self.assertEqual(payload["id_token_signing_alg_values_supported"], ["RS256"])

    def test_openid_configuration_endpoint_includes_required_oidc_arrays(self) -> None:
        client = self.app.test_client()

        response = client.get("/v1/auth/oauth/.well-known/openid-configuration")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        payload_obj = cast(object, response.get_json())
        self.assertIsInstance(payload_obj, dict)
        payload = cast(dict[str, object], payload_obj)
        self.assertEqual(payload["issuer"], "http://localhost:5000/v1/auth/oauth")
        self.assertEqual(payload["subject_types_supported"], ["public"])
        self.assertEqual(payload["id_token_signing_alg_values_supported"], ["RS256"])


if __name__ == "__main__":
    _ = unittest.main()
