"""The OpenAPI specs cover every `/v1` route.

`backend/scripts/export_openapi.py` writes the public spec (flask-smorest
resources) and the UI spec (public + `backend/schemas/ui_api.UI_OPERATIONS`).
These tests fail when a `/v1` Flask rule is documented in neither, or when a
UI operation names a route that no longer exists. CI separately runs the
exporter with `--check` so the committed files match.
"""

import json
import os
import re
import tempfile
import unittest


def _set_default_env() -> None:
    os.environ.setdefault("SKIP_MAIN_DB_REFLECTION", "1")
    os.environ.setdefault("TEMPORARY_ACCESS_LOCKDOWN", "0")
    os.environ.setdefault("MARIADB_USER", "root")
    os.environ.setdefault("MARIADB_PASSWORD", "password")
    os.environ.setdefault("MARIADB_HOST", "127.0.0.1")
    os.environ.setdefault("MARIADB_DATABASE", "pdx")
    os.environ.setdefault("AUTH_SECRET_KEY", "test-auth-secret")
    os.environ.setdefault("PUBLIC_API_BASE_URL", "http://localhost:5000")
    os.environ.setdefault("PUBLIC_FRONTEND_BASE_URL", "http://localhost:8080")


_set_default_env()
_AUTH_DB_TEMP = tempfile.NamedTemporaryFile(
    prefix="pandects_openapi_spec_", suffix=".sqlite", delete=False
)
_AUTH_DB_TEMP.close()
os.environ.setdefault("AUTH_DATABASE_URI", f"sqlite:///{_AUTH_DB_TEMP.name}")

from backend.app import create_test_app  # noqa: E402
from backend.scripts.export_openapi import render_specs  # noqa: E402
from backend.schemas.ui_api import UI_OPERATIONS  # noqa: E402

# `/v1` rules the web client never calls as JSON APIs: the MCP OAuth
# authorization server (spec'd by RFC 8414 / OpenID discovery, not this API)
# and the Zitadel server-to-server notification webhook.
UNDOCUMENTED_V1_RULES = frozenset(
    {
        ("get", "/v1/auth/oauth/.well-known/oauth-authorization-server"),
        ("get", "/v1/auth/oauth/.well-known/openid-configuration"),
        ("get", "/v1/auth/oauth/jwks.json"),
        ("post", "/v1/auth/oauth/register"),
        ("post", "/v1/auth/oauth/token"),
        ("post", "/v1/auth/zitadel/notifications/email"),
    }
)

_CONVERTER_RE = re.compile(r"<(?:[a-z]+:)?([a-zA-Z_][a-zA-Z0-9_]*)>")


def _openapi_path(rule: str) -> str:
    return _CONVERTER_RE.sub(r"{\1}", rule)


class OpenApiSpecCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_test_app(
            config_overrides={"SQLALCHEMY_BINDS": {"auth": f"sqlite:///{_AUTH_DB_TEMP.name}"}}
        )
        _public_yaml, ui_json = render_specs(cls.app)
        cls.ui_spec = json.loads(ui_json)
        cls.v1_rules = {
            (method.lower(), _openapi_path(rule.rule))
            for rule in cls.app.url_map.iter_rules()
            if rule.rule.startswith("/v1/")
            for method in (rule.methods or set()) - {"HEAD", "OPTIONS"}
        }

    def _documented(self) -> set[tuple[str, str]]:
        paths: dict[str, dict[str, object]] = self.ui_spec["paths"]
        return {
            (method, path)
            for path, methods in paths.items()
            for method in methods
            if method in {"get", "post", "put", "patch", "delete"}
        }

    def test_every_v1_rule_is_documented(self) -> None:
        missing = sorted(self.v1_rules - self._documented() - UNDOCUMENTED_V1_RULES)
        self.assertEqual(
            missing,
            [],
            "Document these routes: add a flask-smorest @blp.response, or a UiOperation in "
            "backend/schemas/ui_api.py (ui_auth.py / ui_favorites.py), then run "
            "`python -m backend.scripts.export_openapi`.",
        )

    def test_every_ui_operation_matches_a_route(self) -> None:
        stale = sorted(
            (op.method, op.path)
            for op in UI_OPERATIONS
            if (op.method, op.path) not in self.v1_rules
        )
        self.assertEqual(stale, [], "UiOperation entries without a matching Flask rule")

    def test_undocumented_allowlist_is_current(self) -> None:
        self.assertEqual(sorted(UNDOCUMENTED_V1_RULES - self.v1_rules), [])

    def test_operation_ids_are_unique(self) -> None:
        ids = [
            operation["operationId"]
            for methods in self.ui_spec["paths"].values()
            for operation in methods.values()
            if isinstance(operation, dict) and "operationId" in operation
        ]
        duplicates = sorted({op_id for op_id in ids if ids.count(op_id) > 1})
        self.assertEqual(duplicates, [])

    def test_error_envelope_is_documented(self) -> None:
        error = self.ui_spec["components"]["schemas"]["Error"]
        self.assertEqual(sorted(error["properties"]), ["error", "message"])
        self.assertEqual(sorted(error["required"]), ["error", "message"])


def tearDownModule() -> None:
    try:
        os.unlink(_AUTH_DB_TEMP.name)
    except OSError:
        pass


if __name__ == "__main__":
    unittest.main()
