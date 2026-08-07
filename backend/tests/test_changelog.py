from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RENDERER_PATH = _REPO_ROOT / "bulk" / "changelog" / "render_changelog.py"
_CHANGELOG_PATH = _REPO_ROOT / "bulk" / "changelog" / "changelog.yml"
_CHANGELOG_INFO_PATH = _REPO_ROOT / "backend" / "mcp" / "tools" / "changelog_info.py"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_renderer() -> Any:
    return _load_module("render_changelog", _RENDERER_PATH)


class ChangelogTests(unittest.TestCase):
    """The committed changelog must stay valid, and the roll-up must round-trip.

    bulk/changelog/changelog.yml is the source of truth for the dataset
    changelog shipped next to the bulk dumps and served by /v1/changelog and
    the MCP capabilities changelog section (see bulk/changelog/DESIGN.md).
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.renderer = _load_renderer()

    def test_committed_changelog_is_valid(self) -> None:
        data = self.renderer.load_changelog()
        self.renderer.validate(data)

    def test_validate_rejects_bad_entries(self) -> None:
        base_release = {
            "version": "2026-07-19",
            "released": "2026-07-20T06:32:18Z",
            "dump_sha256": "a" * 64,
            "dump_key": "dumps/public_x.sql.gz",
            "schema_fingerprint": "b" * 64,
            "changes": [{"type": "docs", "severity": "minor", "summary": "baseline"}],
        }
        cases: list[tuple[str, dict[str, object]]] = [
            (
                "unknown change type",
                {
                    "unreleased": [{"type": "bogus", "severity": "minor", "summary": "x"}],
                    "releases": [],
                },
            ),
            (
                "unknown severity",
                {
                    "unreleased": [{"type": "data", "severity": "huge", "summary": "x"}],
                    "releases": [],
                },
            ),
            (
                "empty summary",
                {
                    "unreleased": [{"type": "data", "severity": "minor", "summary": "  "}],
                    "releases": [],
                },
            ),
            (
                "table not in allowlist",
                {
                    "unreleased": [
                        {
                            "type": "data",
                            "severity": "minor",
                            "summary": "x",
                            "tables": ["not_a_public_table"],
                        }
                    ],
                    "releases": [],
                },
            ),
            (
                "unknown change key",
                {
                    "unreleased": [
                        {"type": "data", "severity": "minor", "summary": "x", "extra": 1}
                    ],
                    "releases": [],
                },
            ),
            (
                "duplicate dump_sha256",
                {
                    "unreleased": [],
                    "releases": [
                        dict(base_release, version="2026-08-01"),
                        base_release,
                    ],
                },
            ),
            (
                "releases not newest-first",
                {
                    "unreleased": [],
                    "releases": [
                        base_release,
                        dict(base_release, version="2026-08-01", dump_sha256="c" * 64),
                    ],
                },
            ),
            (
                "duplicate version",
                {
                    "unreleased": [],
                    "releases": [
                        dict(base_release, dump_sha256="c" * 64),
                        base_release,
                    ],
                },
            ),
            (
                "release with empty changes",
                {"unreleased": [], "releases": [dict(base_release, changes=[])]},
            ),
        ]
        for label, data in cases:
            with self.subTest(label):
                with self.assertRaises(SystemExit):
                    self.renderer.validate(data)

    def test_release_roll_up_round_trips(self) -> None:
        renderer = _load_renderer()  # fresh module: path globals get monkeypatched
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            changelog_copy = tmp_path / "changelog.yml"
            changelog_copy.write_text(_CHANGELOG_PATH.read_text())
            dbml = tmp_path / "pandects.dbml"
            dbml.write_text("Table agreements {\n}\n")
            counts_in = tmp_path / "row_counts.json"
            counts_in.write_text(json.dumps({"agreements": 14100}))
            renderer.CHANGELOG_PATH = changelog_copy
            renderer.MARKDOWN_OUT_PATH = tmp_path / "CHANGELOG.md"
            renderer.DOCS_GUIDE_OUT_PATH = tmp_path / "changelog-guide.md"

            renderer.release(
                version="2026-09-01",
                released="2026-09-01T00:00:00Z",
                dump_sha256="e" * 64,
                dump_key="dumps/public_2026-09-01.sql.gz",
                dbml_path=dbml,
                counts_in=counts_in,
                json_out=tmp_path / "changelog.json",
            )

            rolled = renderer.load_changelog(changelog_copy)
            renderer.validate(rolled)
            self.assertEqual(rolled["unreleased"], [])
            releases = cast(list[dict[str, object]], rolled["releases"])
            self.assertEqual(releases[0]["version"], "2026-09-01")
            self.assertEqual(releases[0]["dump_sha256"], "e" * 64)
            stats = cast(dict[str, Any], releases[0]["stats"])
            self.assertEqual(stats["row_counts"], {"agreements": 14100})
            # Unreleased entries (the committed file always has one or the
            # roll-up injects the routine-refresh entry) became the release's changes.
            self.assertTrue(cast(list[object], releases[0]["changes"]))
            # The hand-written comment header survives the rewrite.
            self.assertTrue(changelog_copy.read_text().startswith("# Pandects dataset changelog"))

            payload = json.loads((tmp_path / "changelog.json").read_text())
            self.assertEqual(payload["latest_version"], "2026-09-01")
            self.assertEqual(
                [r["version"] for r in payload["releases"]],
                [r["version"] for r in releases],
            )
            # A second release with the same dump is rejected.
            with self.assertRaises(SystemExit):
                renderer.release(
                    version="2026-09-02",
                    released="2026-09-02T00:00:00Z",
                    dump_sha256="e" * 64,
                    dump_key="dumps/public_2026-09-02.sql.gz",
                    dbml_path=dbml,
                    counts_in=counts_in,
                    json_out=tmp_path / "changelog2.json",
                )
            # A same-day repush (new dump, same date version) gets a .2 suffix
            # so `?since=2026-09-01` still surfaces it.
            renderer.release(
                version="2026-09-01",
                released="2026-09-01T06:00:00Z",
                dump_sha256="f" * 64,
                dump_key="dumps/public_2026-09-01_2.sql.gz",
                dbml_path=dbml,
                counts_in=counts_in,
                json_out=tmp_path / "changelog3.json",
            )
            rolled = renderer.load_changelog(changelog_copy)
            renderer.validate(rolled)
            releases = cast(list[dict[str, object]], rolled["releases"])
            self.assertEqual(releases[0]["version"], "2026-09-01.2")
            self.assertGreater(str(releases[0]["version"]), "2026-09-01")

    def test_soft_gate_flags_anomalous_row_count_deltas(self) -> None:
        releases = [
            {
                "version": "2026-07-19",
                "stats": {"row_counts": {"agreements": 10000, "sections": 900000}},
            }
        ]
        # Small drift (under 5% and under the absolute floor logic) passes.
        no_anomaly = self.renderer.find_anomalous_tables(
            {"agreements": 10020, "sections": 901000}, releases
        )
        self.assertEqual(no_anomaly, [])
        # A large unexplained move is flagged with its signed delta.
        anomalies = self.renderer.find_anomalous_tables(
            {"agreements": 4000, "sections": 901000}, releases
        )
        self.assertEqual(anomalies, [("agreements", 10000, -6000)])
        # No release history or no stats => nothing to compare against.
        self.assertEqual(self.renderer.find_anomalous_tables({"agreements": 1}, []), [])
        self.assertEqual(
            self.renderer.find_anomalous_tables(
                {"agreements": 1}, [{"version": "2026-07-19"}]
            ),
            [],
        )

    def test_soft_gate_uses_median_historical_delta_when_history_exists(self) -> None:
        # Four releases => three historical deltas of 1000 each; threshold
        # becomes max(2 * 1000, floor), so a 1500 move is fine and 2500 is not.
        releases = [
            {"version": f"2026-0{i}-01", "stats": {"row_counts": {"sections": 100000 + i * 1000}}}
            for i in range(4, 0, -1)
        ]
        newest = 104000
        self.assertEqual(
            self.renderer.find_anomalous_tables({"sections": newest + 1500}, releases), []
        )
        self.assertEqual(
            self.renderer.find_anomalous_tables({"sections": newest + 2500}, releases),
            [("sections", newest, 2500)],
        )

    def test_mdx_escape_spares_code_spans(self) -> None:
        # Prose outside backticks is escaped (bare {, }, < break the MDX
        # build); code spans stay literal (entities inside them would render
        # verbatim on the docs site).
        self.assertEqual(
            self.renderer._mdx_escape("Renamed the `{old}` field to <new>"),
            "Renamed the `{old}` field to &lt;new>",
        )
        self.assertEqual(
            self.renderer._mdx_escape("bare {expr} and <tag>"),
            "bare &#123;expr&#125; and &lt;tag>",
        )
        # Unbalanced backticks: no well-formed span, everything is escaped.
        self.assertEqual(
            self.renderer._mdx_escape("oops `unclosed {x}"),
            "oops `unclosed &#123;x&#125;",
        )
        lines = self.renderer._render_change_lines(
            {"type": "api", "severity": "minor", "summary": "Renamed `{old}` to <new>"},
            mdx=True,
        )
        self.assertEqual(lines, ["- **[api/minor]** Renamed `{old}` to &lt;new>"])

    def test_rendered_artifacts_are_current(self) -> None:
        """CHANGELOG.md and the docs guide must match changelog.yml (regenerate
        with: bulk/changelog/render_changelog.py render)."""
        data = self.renderer.load_changelog()
        markdown_path = _REPO_ROOT / "bulk" / "changelog" / "CHANGELOG.md"
        guide_path = _REPO_ROOT / "docs" / "docs" / "guides" / "changelog.md"
        self.assertEqual(markdown_path.read_text(), self.renderer.render_markdown(data))
        self.assertEqual(guide_path.read_text(), self.renderer.render_docs_guide(data))


class ChangelogInfoFetchTests(unittest.TestCase):
    """The MCP capabilities changelog section fetches the published artifact.

    Loaded by path so the test doesn't pull in the heavy backend.mcp.tools
    package (changelog_info.py imports only the stdlib).
    """

    def setUp(self) -> None:
        self.module = _load_module("changelog_info", _CHANGELOG_INFO_PATH)
        os.environ.pop("MCP_CHANGELOG_FETCH", None)

    def test_fetch_sends_explicit_user_agent(self) -> None:
        # Cloudflare 403s urllib's default Python-urllib/x.y agent, which
        # nulled this section in production until the header was set.
        seen: dict[str, object] = {}

        class _FakeResponse:
            def __enter__(self) -> "_FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {"latest_version": "2026-07-19", "latest_released": "x", "releases": []}
                ).encode()

        def _fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResponse:
            seen["headers"] = dict(request.headers)
            seen["url"] = request.full_url
            seen["timeout"] = timeout
            return _FakeResponse()

        self.module.urllib.request.urlopen = _fake_urlopen
        section = self.module.changelog_capabilities_section()

        headers = cast(dict[str, str], seen["headers"])
        agent = next(v for k, v in headers.items() if k.lower() == "user-agent")
        self.assertIn("pandects", agent.lower())
        self.assertNotIn("python-urllib", agent.lower())
        self.assertEqual(seen["timeout"], self.module._FETCH_TIMEOUT_SECONDS)
        self.assertEqual(section["latest_version"], "2026-07-19")

    def test_fetch_failure_degrades_to_nulls(self) -> None:
        def _boom(request: Any, timeout: float | None = None) -> object:
            raise OSError("network down")

        self.module.urllib.request.urlopen = _boom
        section = self.module.changelog_capabilities_section()
        self.assertIsNone(section["latest_version"])
        self.assertIsNone(section["breaking_changes_in_latest"])
        self.assertEqual(section["api_route"], "/v1/changelog")

    def test_fetch_disabled_by_env_makes_no_request(self) -> None:
        called: list[object] = []

        def _tracker(request: Any, timeout: float | None = None) -> object:
            called.append(request)
            raise AssertionError("should not fetch")

        self.module.urllib.request.urlopen = _tracker
        os.environ["MCP_CHANGELOG_FETCH"] = "0"
        try:
            section = self.module.changelog_capabilities_section()
        finally:
            os.environ.pop("MCP_CHANGELOG_FETCH", None)
        self.assertEqual(called, [])
        self.assertIsNone(section["latest_version"])


if __name__ == "__main__":
    unittest.main()
