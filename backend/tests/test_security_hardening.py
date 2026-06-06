import os
import unittest

from flask import Flask


class AssertZitadelNotificationSigningKeyTests(unittest.TestCase):
    """Fail-fast startup check: on Fly the notification signing key must be
    configured or every webhook is rejected and email delivery silently
    breaks. Outside Fly the check is a no-op so local dev / tests don't need
    to set the env var."""

    _saved_fly: str | None = None
    _saved_region: str | None = None
    _saved_key: str | None = None

    def setUp(self) -> None:
        self._saved_fly = os.environ.pop("FLY_APP_NAME", None)
        self._saved_region = os.environ.pop("FLY_REGION", None)
        self._saved_key = os.environ.pop("AUTH_ZITADEL_NOTIFICATION_SIGNING_KEY", None)

    def tearDown(self) -> None:
        for k in ("FLY_APP_NAME", "FLY_REGION", "AUTH_ZITADEL_NOTIFICATION_SIGNING_KEY"):
            os.environ.pop(k, None)
        if self._saved_fly is not None:
            os.environ["FLY_APP_NAME"] = self._saved_fly
        if self._saved_region is not None:
            os.environ["FLY_REGION"] = self._saved_region
        if self._saved_key is not None:
            os.environ["AUTH_ZITADEL_NOTIFICATION_SIGNING_KEY"] = self._saved_key

    def test_off_fly_no_key_is_fine(self) -> None:
        from backend.auth.email_runtime import (
            assert_zitadel_notification_signing_key_configured,
        )

        assert_zitadel_notification_signing_key_configured()

    def test_on_fly_missing_key_raises(self) -> None:
        os.environ["FLY_APP_NAME"] = "pandects-test"
        from backend.auth.email_runtime import (
            assert_zitadel_notification_signing_key_configured,
        )

        with self.assertRaises(RuntimeError) as ctx:
            assert_zitadel_notification_signing_key_configured()
        self.assertIn("AUTH_ZITADEL_NOTIFICATION_SIGNING_KEY", str(ctx.exception))

    def test_on_fly_with_key_is_fine(self) -> None:
        os.environ["FLY_APP_NAME"] = "pandects-test"
        os.environ["AUTH_ZITADEL_NOTIFICATION_SIGNING_KEY"] = "test-key"
        from backend.auth.email_runtime import (
            assert_zitadel_notification_signing_key_configured,
        )

        assert_zitadel_notification_signing_key_configured()


class RequestIpAddressOnFlyTests(unittest.TestCase):
    """On Fly the edge always sets Fly-Client-IP. We deliberately do NOT
    fall back to X-Forwarded-For, since XFF's first hop is client-injectable
    if the edge ever fails to strip it."""

    app: Flask = Flask(__name__)

    def setUp(self) -> None:
        self.app = Flask(__name__)

    def test_fly_returns_fly_client_ip(self) -> None:
        from backend.core.runtime_utils import request_ip_address

        with self.app.test_request_context(
            "/",
            headers={
                "Fly-Client-IP": "203.0.113.5",
                "X-Forwarded-For": "10.0.0.1",
            },
        ):
            self.assertEqual(request_ip_address(is_running_on_fly=True), "203.0.113.5")

    def test_fly_without_fly_client_ip_returns_none_even_if_xff_present(self) -> None:
        # Spoof-defense: a malicious client setting XFF must not be trusted
        # as a source IP for rate-limiting / audit fields when Fly's edge
        # didn't tag the request.
        from backend.core.runtime_utils import request_ip_address

        with self.app.test_request_context(
            "/",
            headers={"X-Forwarded-For": "198.51.100.9"},
            environ_base={"REMOTE_ADDR": "10.0.0.2"},
        ):
            self.assertIsNone(request_ip_address(is_running_on_fly=True))

    def test_off_fly_uses_remote_addr(self) -> None:
        from backend.core.runtime_utils import request_ip_address

        with self.app.test_request_context(
            "/",
            environ_base={"REMOTE_ADDR": "127.0.0.1"},
        ):
            self.assertEqual(request_ip_address(is_running_on_fly=False), "127.0.0.1")


if __name__ == "__main__":
    unittest.main()
