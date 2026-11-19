from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from datetime import date, timedelta
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

    def test_row_counts_use_the_dump_row_filters(self) -> None:
        # Release stats count what the dump holds: only the latest xml version.
        filters = self.renderer.load_row_filters()
        self.assertEqual(filters, {"xml": "latest = 1"})
        self.assertTrue(set(filters) <= set(self.renderer.load_allowlist()))

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

            # Derived, not hardcoded: releases must stay sorted newest-first, so
            # a fixed version starts failing as soon as a later release is
            # stamped into the committed changelog.
            committed = renderer.load_changelog(changelog_copy)
            newest = cast(list[dict[str, object]], committed["releases"])[0]["version"]
            version = (date.fromisoformat(str(newest)) + timedelta(days=1)).isoformat()
            next_version = (date.fromisoformat(version) + timedelta(days=1)).isoformat()

            renderer.release(
                version=version,
                released=f"{version}T00:00:00Z",
                dump_sha256="e" * 64,
                dump_key=f"dumps/public_{version}.sql.gz",
                dbml_path=dbml,
                counts_in=counts_in,
                json_out=tmp_path / "changelog.json",
            )

            rolled = renderer.load_changelog(changelog_copy)
            renderer.validate(rolled)
            self.assertEqual(rolled["unreleased"], [])
            releases = cast(list[dict[str, object]], rolled["releases"])
            self.assertEqual(releases[0]["version"], version)
            self.assertEqual(releases[0]["dump_sha256"], "e" * 64)
            stats = cast(dict[str, Any], releases[0]["stats"])
            self.assertEqual(stats["row_counts"], {"agreements": 14100})
            # Unreleased entries (the committed file always has one or the
            # roll-up injects the routine-refresh entry) became the release's changes.
            self.assertTrue(cast(list[object], releases[0]["changes"]))
            # The hand-written comment header survives the rewrite.
            self.assertTrue(changelog_copy.read_text().startswith("# Pandects dataset changelog"))

            payload = json.loads((tmp_path / "changelog.json").read_text())
            self.assertEqual(payload["latest_version"], version)
            self.assertEqual(
                [r["version"] for r in payload["releases"]],
                [r["version"] for r in releases],
            )
            # A second release with the same dump is rejected.
            with self.assertRaises(SystemExit):
                renderer.release(
                    version=next_version,
                    released=f"{next_version}T00:00:00Z",
                    dump_sha256="e" * 64,
                    dump_key=f"dumps/public_{next_version}.sql.gz",
                    dbml_path=dbml,
                    counts_in=counts_in,
                    json_out=tmp_path / "changelog2.json",
                )
            # A same-day repush (new dump, same date version) gets a .2 suffix
            # so `?since=<that date>` still surfaces it.
            renderer.release(
                version=version,
                released=f"{version}T06:00:00Z",
                dump_sha256="f" * 64,
                dump_key=f"dumps/public_{version}_2.sql.gz",
                dbml_path=dbml,
                counts_in=counts_in,
                json_out=tmp_path / "changelog3.json",
            )
            rolled = renderer.load_changelog(changelog_copy)
            renderer.validate(rolled)
            releases = cast(list[dict[str, object]], rolled["releases"])
            self.assertEqual(releases[0]["version"], f"{version}.2")
            self.assertGreater(str(releases[0]["version"]), version)

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
        lines = self.renderer._render_change(
            {"type": "api", "severity": "minor", "summary": "Renamed `{old}` to <new>"},
            mdx=True,
        )
        self.assertEqual(lines, ["**API:** Renamed `{old}` to &lt;new>", ""])

    def test_mdx_escape_keeps_colons_from_starting_directives(self) -> None:
        # The docs site parses ":name" as a text directive, so a colon before
        # a letter becomes an entity in MDX prose.
        mdx = self.renderer._mdx_escape
        self.assertEqual(mdx("declare scope:name too"), "declare scope&#58;name too")
        self.assertEqual(mdx("x:y[label]"), "x&#58;y[label]")
        for kept in (
            "in `scope:name` code",
            "see https://x.y/z",
            "at 10:30 today",
            "Label: value",
            "a::b",
        ):
            self.assertEqual(mdx(kept), kept)
        # CHANGELOG.md is not MDX and keeps the colon.
        self.assertEqual(self.renderer._markdown_escape("scope:name"), "scope:name")
        self.assertEqual(
            self.renderer._render_change(
                {"type": "mcp", "severity": "minor", "summary": "Adds agreements:read_fulltext"},
                mdx=True,
            ),
            ["**MCP:** Adds agreements&#58;read_fulltext", ""],
        )

    def test_entry_prose_keeps_paragraph_breaks(self) -> None:
        # A line break in details or migration starts a paragraph (a blank
        # line in a folded YAML scalar loads as one); other whitespace folds.
        self.assertEqual(
            self.renderer._paragraphs("First  point,\twrapped.\n\nSecond point.\nThird.\n"),
            ["First point, wrapped.", "Second point.", "Third."],
        )
        self.assertEqual(self.renderer._paragraphs(None), [])
        change = {
            "type": "data",
            "severity": "breaking",
            "summary": "Prices corrected",
            "details": "What changed.\n\nWhy <it> changed.\n",
            "tables": ["agreements", "sections"],
            "migration": "Recompute sums.\n\nRe-fetch ids.",
            "refs": ["#1", "abc123"],
        }
        self.assertEqual(
            self.renderer._render_change(change, mdx=True),
            [
                "<details>",
                "<summary>**Data:** Prices corrected</summary>",
                "",
                "What changed.",
                "",
                "Why &lt;it> changed.",
                "",
                "**Tables:** `agreements`, `sections`",
                "",
                "**Migration:** Recompute sums.",
                "",
                "Re-fetch ids.",
                "",
                "**Refs:** #1, abc123",
                "",
                "</details>",
                "",
            ],
        )
        self.assertEqual(
            self.renderer._render_change(change),
            [
                "- **Data:** Prices corrected",
                "",
                "  What changed.",
                "",
                "  Why &lt;it> changed.",
                "",
                "  **Tables:** `agreements`, `sections`",
                "",
                "  **Migration:** Recompute sums.",
                "",
                "  Re-fetch ids.",
                "",
                "  **Refs:** #1, abc123",
                "",
            ],
        )

    def test_paragraph_starts_and_tags_cannot_change_the_page_structure(self) -> None:
        # Every paragraph of an entry starts its own line, so a block marker
        # at its start would turn the paragraph into a heading or a list.
        escape = self.renderer._escape_block_start
        self.assertEqual(escape("# of rows changed"), "\\# of rows changed")
        self.assertEqual(escape("- 5 rows"), "\\- 5 rows")
        self.assertEqual(escape("3) Spacing"), "3\\) Spacing")
        self.assertEqual(escape("12. Twelve, and 3. later"), "12\\. Twelve, and 3. later")
        self.assertEqual(escape("> quoted"), "\\> quoted")
        self.assertEqual(escape("---"), "\\---")
        for plain in ("#175 fixed", "-5 degrees", "1,432 rows", "*kept* text", "(1) first"):
            self.assertEqual(escape(plain), plain)
        # Three or more backticks or tildes would open a code block.
        self.assertEqual(escape("```sql rows"), "\\```sql rows")
        self.assertEqual(escape("````"), "\\````")
        self.assertEqual(escape("~~~ text"), "\\~~~ text")
        self.assertEqual(escape("~~~~"), "\\~~~~")
        # Inline code spans are not fences: one or two backticks, or a
        # triple-backtick span closed on the same line, stay as written.
        for code in ("`id` is now text", "``a`b`` text", "```x``` text", "~~ two tildes"):
            self.assertEqual(escape(code), code)
        lines = self.renderer._render_change(
            {"type": "api", "severity": "minor", "summary": "s", "details": "```json is gone"},
            mdx=True,
        )
        self.assertEqual(lines[3], "\\```json is gone")
        change = {
            "type": "api",
            "severity": "minor",
            "summary": "Snippets drop <page> markers",
            "details": "1. The `<page>` tag and <pageUUID> are dropped.\n\n# of rows: 5",
        }
        # CHANGELOG.md is read on GitHub, which would drop a bare <tag>.
        self.assertEqual(
            self.renderer._render_change(change),
            [
                "- **API:** Snippets drop &lt;page> markers",
                "",
                "  1\\. The `<page>` tag and &lt;pageUUID> are dropped.",
                "",
                "  \\# of rows: 5",
                "",
            ],
        )

    def test_paragraph_breaks_survive_the_release_roll_up(self) -> None:
        # write_changelog must keep the line breaks that mark paragraphs.
        text = "First paragraph, long enough to be folded at the width the writer uses for a scalar.\n\nSecond paragraph.\n"
        data = {
            "unreleased": [
                {"type": "data", "severity": "minor", "summary": "s", "details": text}
            ],
            "releases": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "changelog.yml"
            path.write_text("# header\n\nunreleased: []\nreleases: []\n")
            self.renderer.write_changelog(data, path)
            loaded = self.renderer.load_changelog(path)
        self.assertEqual(loaded["unreleased"][0]["details"], text)
        self.assertEqual(len(self.renderer._paragraphs(loaded["unreleased"][0]["details"])), 2)

    def test_release_changes_are_grouped_by_severity(self) -> None:
        def change(change_type: str, severity: str, summary: str) -> dict[str, object]:
            return {"type": change_type, "severity": severity, "summary": summary}

        few = [change("mcp", "minor", "m1"), change("data", "breaking", "b1")]
        lines = self.renderer._render_changes(few, anchor="2026-01-02", mdx=True)
        # A short release gets the count line and no table.
        self.assertEqual(lines[0], "2 changes: 1 breaking, 1 minor.")
        self.assertFalse(any(line.startswith("|") for line in lines))
        headings = [line for line in lines if line.startswith("### ")]
        self.assertEqual(
            headings,
            [
                "### Breaking (1) {/* #2026-01-02-breaking */}",
                "### Minor (1) {/* #2026-01-02-minor */}",
            ],
        )
        self.assertEqual(
            self.renderer._render_changes([few[0]], anchor="x")[0], "1 change: 1 minor."
        )

        # One change short of the threshold: still no table.
        nine = [change("api", "minor", f"a{i}") for i in range(9)]
        self.assertFalse(
            any(line.startswith("|") for line in self.renderer._render_changes(nine, anchor="x"))
        )

        many = (
            [change("api", "minor", f"a{i}") for i in range(6)]
            + [change("data", "notable", "d1"), change("schema", "notable", "s1")]
            + [change("data", "breaking", "d2"), change("mcp", "minor", "m1")]
        )
        lines = self.renderer._render_changes(many, anchor="2026-01-03")
        self.assertEqual(lines[0], "10 changes: 1 breaking, 2 notable, 7 minor.")
        table = [line for line in lines if line.startswith("|")]
        self.assertEqual(
            table,
            [
                "| Area | Breaking | Notable | Minor | Total |",
                "| --- | ---: | ---: | ---: | ---: |",
                "| Schema | – | 1 | – | 1 |",
                "| Data | 1 | 1 | – | 2 |",
                "| API | – | – | 6 | 6 |",
                "| MCP | – | – | 1 | 1 |",
            ],
        )
        # Plain markdown headings carry no docs-site id; within a severity,
        # entries follow the area order and then the order they were written.
        self.assertIn("### Notable (2)", lines)
        entries = [line for line in lines if line.startswith("- **")]
        self.assertEqual(
            entries[:3], ["- **Data:** d2", "- **Schema:** s1", "- **Data:** d1"]
        )
        self.assertEqual(entries[3:], [f"- **API:** a{i}" for i in range(6)] + ["- **MCP:** m1"])

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
