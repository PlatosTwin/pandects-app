import unittest

import backend.app as backend_app


class AuthDependencyTests(unittest.TestCase):
    def test_build_route_deps_contains_auth_contract(self) -> None:
        _search_deps, _agreements_deps, _reference_data_deps, auth_deps = (
            backend_app._build_route_deps()
        )
        self.assertTrue(callable(auth_deps._require_auth_db))
        self.assertTrue(callable(auth_deps._load_json))
        self.assertTrue(callable(auth_deps._issue_session_token))
        self.assertTrue(callable(auth_deps._verify_turnstile_token))
        self.assertIs(auth_deps.AuthUser, backend_app.AuthUser)
        self.assertIs(auth_deps.AuthSession, backend_app.AuthSession)

    def test_auth_dep_wrappers_follow_runtime_monkeypatches(self) -> None:
        _search_deps, _agreements_deps, _reference_data_deps, auth_deps = (
            backend_app._build_route_deps()
        )
        original_verify = backend_app._verify_turnstile_token
        calls: list[str] = []
        try:
            backend_app._verify_turnstile_token = lambda *, token: calls.append(token)
            auth_deps._verify_turnstile_token(token="abc")
            self.assertEqual(calls, ["abc"])
        finally:
            backend_app._verify_turnstile_token = original_verify


if __name__ == "__main__":
    unittest.main()
