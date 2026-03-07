import os
import tempfile
import unittest

from backend.app import api, create_test_app, db


def _set_default_env() -> None:
    os.environ.setdefault("SKIP_MAIN_DB_REFLECTION", "1")
    os.environ.setdefault("MARIADB_USER", "root")
    os.environ.setdefault("MARIADB_PASSWORD", "password")
    os.environ.setdefault("MARIADB_HOST", "127.0.0.1")
    os.environ.setdefault("MARIADB_DATABASE", "pdx")
    os.environ.setdefault("AUTH_SECRET_KEY", "test-auth-secret")
    os.environ.setdefault("PUBLIC_API_BASE_URL", "http://localhost:5000")
    os.environ.setdefault("PUBLIC_FRONTEND_BASE_URL", "http://localhost:8080")
    os.environ.setdefault("GOOGLE_OAUTH_CLIENT_ID", "test-google-client-id")
    os.environ.setdefault("GOOGLE_OAUTH_CLIENT_SECRET", "test-google-client-secret")
    os.environ.setdefault("AUTH_SESSION_TRANSPORT", "bearer")


_set_default_env()
_AUTH_DB_TEMP = tempfile.NamedTemporaryFile(
    prefix="pandects_auth_routes_",
    suffix=".sqlite",
    delete=False,
)
_AUTH_DB_TEMP.close()
os.environ.setdefault("AUTH_DATABASE_URI", f"sqlite:///{_AUTH_DB_TEMP.name}")


class RouteContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_test_app(
            config_overrides={
                "SQLALCHEMY_BINDS": {"auth": f"sqlite:///{_AUTH_DB_TEMP.name}"},
            }
        )
        with cls.app.app_context():
            db.create_all(bind_key="auth")

    def test_key_routes_are_registered(self) -> None:
        rules = {rule.rule for rule in self.app.url_map.iter_rules()}
        expected = {
            "/v1/sections",
            "/v1/agreements",
            "/v1/agreements/<string:agreement_uuid>",
            "/v1/sections/<string:section_uuid>",
            "/v1/taxonomy",
            "/v1/naics",
            "/v1/dumps",
            "/v1/auth/register",
            "/v1/auth/login",
        }
        for route in expected:
            self.assertIn(route, rules)

    def test_operation_ids_remain_stable(self) -> None:
        with self.app.test_request_context():
            spec = api.spec
            if spec is None:
                self.fail("Expected API spec to be initialized")
            spec_dict = spec.to_dict()
        paths = spec_dict.get("paths", {})
        self.assertEqual(paths["/v1/sections"]["get"]["operationId"], "listSections")
        self.assertEqual(paths["/v1/agreements"]["get"]["operationId"], "listAgreements")
        self.assertEqual(paths["/v1/taxonomy"]["get"]["operationId"], "getTaxonomy")
        self.assertEqual(paths["/v1/naics"]["get"]["operationId"], "getNaics")
        self.assertEqual(paths["/v1/dumps"]["get"]["operationId"], "listDumps")


if __name__ == "__main__":
    unittest.main()
