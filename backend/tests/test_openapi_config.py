import os
import unittest


class ExposeSwaggerUiTests(unittest.TestCase):
    """Swagger UI pulls scripts from cdn.jsdelivr.net (not in our /v1 CSP) and
    enumerates every authenticated endpoint with parameter schemas. Keep it on
    by default for local dev but off on Fly unless explicitly opted in."""

    _saved_fly: str | None = None
    _saved_region: str | None = None
    _saved_expose: str | None = None

    def setUp(self) -> None:
        self._saved_fly = os.environ.pop("FLY_APP_NAME", None)
        self._saved_region = os.environ.pop("FLY_REGION", None)
        self._saved_expose = os.environ.pop("EXPOSE_SWAGGER_UI", None)

    def tearDown(self) -> None:
        for k in ("FLY_APP_NAME", "FLY_REGION", "EXPOSE_SWAGGER_UI"):
            os.environ.pop(k, None)
        if self._saved_fly is not None:
            os.environ["FLY_APP_NAME"] = self._saved_fly
        if self._saved_region is not None:
            os.environ["FLY_REGION"] = self._saved_region
        if self._saved_expose is not None:
            os.environ["EXPOSE_SWAGGER_UI"] = self._saved_expose

    def test_local_dev_exposes_swagger_ui(self) -> None:
        from backend.core.config import _expose_swagger_ui

        self.assertTrue(_expose_swagger_ui())

    def test_fly_hides_swagger_ui_by_default(self) -> None:
        os.environ["FLY_APP_NAME"] = "pandects-test"
        from backend.core.config import _expose_swagger_ui

        self.assertFalse(_expose_swagger_ui())

    def test_fly_with_explicit_opt_in_exposes_swagger_ui(self) -> None:
        os.environ["FLY_APP_NAME"] = "pandects-test"
        os.environ["EXPOSE_SWAGGER_UI"] = "1"
        from backend.core.config import _expose_swagger_ui

        self.assertTrue(_expose_swagger_ui())

    def test_explicit_opt_out_hides_swagger_ui_even_locally(self) -> None:
        os.environ["EXPOSE_SWAGGER_UI"] = "0"
        from backend.core.config import _expose_swagger_ui

        self.assertFalse(_expose_swagger_ui())


if __name__ == "__main__":
    unittest.main()
