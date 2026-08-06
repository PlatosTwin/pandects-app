#!/usr/bin/env python3
"""Validate, gate, roll up, and render the Pandects dataset changelog.

Source of truth: bulk/changelog/changelog.yml (see bulk/changelog/DESIGN.md).
Subcommands, in the order bulk/push_to_r2.sh uses them:

  validate   Schema-check changelog.yml and exit.
  precheck   Pre-dump gates (push step 0b): hard-abort if the schema
             fingerprint changed since the last release without a `schema`
             entry in `unreleased`; compute per-table row counts from the live
             DB, warn (and require confirmation) on anomalous deltas with no
             `data` entry. Writes counts JSON for the release step.
  release    Post-dump roll-up: move `unreleased` into a new stamped release,
             rewrite changelog.yml, and render all derived artifacts.
  render     Re-render derived artifacts (CHANGELOG.md, docs guide, JSON)
             from the current changelog.yml without changing it.

Derived artifacts:
  bulk/changelog/CHANGELOG.md       (committed, human-readable)
  docs/docs/guides/changelog.md     (committed, docs site + llms.txt)
  changelog.json (path via --json-out; uploaded to R2, not committed)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHANGELOG_PATH = REPO_ROOT / "bulk" / "changelog" / "changelog.yml"
ALLOWLIST_PATH = REPO_ROOT / "bulk" / "public_tables.txt"
MARKDOWN_OUT_PATH = REPO_ROOT / "bulk" / "changelog" / "CHANGELOG.md"
DOCS_GUIDE_OUT_PATH = REPO_ROOT / "docs" / "docs" / "guides" / "changelog.md"

CHANGE_TYPES = ("schema", "data", "api", "mcp", "docs", "pipeline")
SEVERITIES = ("breaking", "notable", "minor")
_CHANGE_KEYS = {"type", "severity", "summary", "details", "tables", "migration", "refs"}
_RELEASE_KEYS = {
    "version",
    "released",
    "dump_sha256",
    "dump_key",
    "schema_fingerprint",
    "stats",
    "changes",
}

# Soft data-change gate: a table's count delta is anomalous when it exceeds
# max(2x the table's median historical per-release delta, ABS_FLOOR) — or,
# with fewer than MIN_HISTORY historical deltas, max(FRACTION x previous
# count, ABS_FLOOR).
_SOFT_GATE_MIN_HISTORY = 3
_SOFT_GATE_FRACTION = 0.05
_SOFT_GATE_ABS_FLOOR = 25


def _fail(message: str) -> None:
    raise SystemExit(f"❌ changelog: {message}")


def load_allowlist() -> list[str]:
    tables: list[str] = []
    for raw_line in ALLOWLIST_PATH.read_text().splitlines():
        entry = raw_line.split("#", 1)[0].strip()
        if entry:
            tables.append(entry)
    if not tables:
        _fail(f"no tables found in {ALLOWLIST_PATH}")
    return tables


def load_changelog(path: Path | None = None) -> dict[str, object]:
    # CHANGELOG_PATH is resolved at call time so tests can repoint the module
    # globals at a sandbox copy.
    path = path if path is not None else CHANGELOG_PATH
    if not path.is_file():
        _fail(f"missing {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        _fail(f"{path} must be a mapping with 'unreleased' and 'releases' keys")
    return data


def _validate_change(change: object, where: str, allowlist: set[str]) -> None:
    if not isinstance(change, dict):
        _fail(f"{where}: each entry must be a mapping")
    unknown = set(change) - _CHANGE_KEYS
    if unknown:
        _fail(f"{where}: unknown keys {sorted(unknown)}")
    if change.get("type") not in CHANGE_TYPES:
        _fail(f"{where}: type must be one of {CHANGE_TYPES}, got {change.get('type')!r}")
    if change.get("severity") not in SEVERITIES:
        _fail(f"{where}: severity must be one of {SEVERITIES}, got {change.get('severity')!r}")
    summary = change.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        _fail(f"{where}: summary must be a non-empty string")
    tables = change.get("tables")
    if tables is not None:
        if not isinstance(tables, list) or not all(isinstance(t, str) for t in tables):
            _fail(f"{where}: tables must be a list of strings")
        rogue = set(tables) - allowlist
        if rogue:
            _fail(f"{where}: tables {sorted(rogue)} are not in {ALLOWLIST_PATH.name}")
    refs = change.get("refs")
    if refs is not None and (
        not isinstance(refs, list) or not all(isinstance(r, str) for r in refs)
    ):
        _fail(f"{where}: refs must be a list of strings")
    for key in ("details", "migration"):
        value = change.get(key)
        if value is not None and not isinstance(value, str):
            _fail(f"{where}: {key} must be a string or null")


def validate(data: dict[str, object]) -> None:
    unknown_top = set(data) - {"unreleased", "releases"}
    if unknown_top:
        _fail(f"unknown top-level keys {sorted(unknown_top)}")
    allowlist = set(load_allowlist())

    unreleased = data.get("unreleased") or []
    if not isinstance(unreleased, list):
        _fail("'unreleased' must be a list (or empty)")
    for i, change in enumerate(unreleased):
        _validate_change(change, f"unreleased[{i}]", allowlist)

    releases = data.get("releases") or []
    if not isinstance(releases, list):
        _fail("'releases' must be a list (or empty)")
    seen_sha: set[str] = set()
    versions: list[str] = []
    for i, release in enumerate(releases):
        where = f"releases[{i}]"
        if not isinstance(release, dict):
            _fail(f"{where}: must be a mapping")
        unknown = set(release) - _RELEASE_KEYS
        if unknown:
            _fail(f"{where}: unknown keys {sorted(unknown)}")
        for key in ("version", "released", "dump_sha256", "dump_key", "schema_fingerprint"):
            value = release.get(key)
            if not isinstance(value, str) or not value.strip():
                _fail(f"{where}: {key} must be a non-empty string")
        sha = str(release["dump_sha256"])
        if sha in seen_sha:
            _fail(f"{where}: duplicate dump_sha256 {sha}")
        seen_sha.add(sha)
        versions.append(str(release["version"]))
        stats = release.get("stats")
        if stats is not None:
            if not isinstance(stats, dict) or set(stats) - {"row_counts"}:
                _fail(f"{where}: stats must be a mapping with only 'row_counts'")
            row_counts = stats.get("row_counts")
            if not isinstance(row_counts, dict) or not all(
                isinstance(k, str) and isinstance(v, int) for k, v in row_counts.items()
            ):
                _fail(f"{where}: stats.row_counts must map table name -> int")
        changes = release.get("changes")
        if not isinstance(changes, list) or not changes:
            _fail(f"{where}: changes must be a non-empty list")
        for j, change in enumerate(changes):
            _validate_change(change, f"{where}.changes[{j}]", allowlist)
    if versions != sorted(versions, reverse=True):
        _fail("releases must be sorted newest-first by version")


# ── Rendering ────────────────────────────────────────────────────


def _render_change_lines(change: dict[str, object]) -> list[str]:
    lines = [f"- **[{change['type']}/{change['severity']}]** {str(change['summary']).strip()}"]
    details = change.get("details")
    if isinstance(details, str) and details.strip():
        lines.append(f"  - {' '.join(details.split())}")
    tables = change.get("tables")
    if isinstance(tables, list) and tables:
        lines.append(f"  - Tables: {', '.join(str(t) for t in tables)}")
    migration = change.get("migration")
    if isinstance(migration, str) and migration.strip():
        lines.append(f"  - Migration: {' '.join(migration.split())}")
    refs = change.get("refs")
    if isinstance(refs, list) and refs:
        lines.append(f"  - Refs: {', '.join(str(r) for r in refs)}")
    return lines


_SEVERITY_ORDER = {name: i for i, name in enumerate(SEVERITIES)}


def _render_body(data: dict[str, object], include_unreleased: bool) -> list[str]:
    lines: list[str] = []
    unreleased = data.get("unreleased") or []
    if include_unreleased and isinstance(unreleased, list) and unreleased:
        lines.append("## Unreleased")
        lines.append("")
        for change in sorted(
            unreleased, key=lambda c: _SEVERITY_ORDER.get(str(c.get("severity")), 99)
        ):
            lines.extend(_render_change_lines(change))
        lines.append("")
    releases = data.get("releases") or []
    for release in releases if isinstance(releases, list) else []:
        released = str(release.get("released", ""))[:10]
        lines.append(f"## {release['version']} — released {released}")
        lines.append("")
        lines.append(f"- Dump: `{release['dump_key']}`")
        lines.append(f"- SHA-256: `{release['dump_sha256']}`")
        lines.append("")
        for change in sorted(
            release.get("changes", []),
            key=lambda c: _SEVERITY_ORDER.get(str(c.get("severity")), 99),
        ):
            lines.extend(_render_change_lines(change))
        stats = release.get("stats")
        if isinstance(stats, dict) and isinstance(stats.get("row_counts"), dict):
            total = sum(stats["row_counts"].values())
            lines.append(
                f"- Stats: {len(stats['row_counts'])} tables, {total:,} total rows"
                " (per-table counts in the machine-readable changelog)"
            )
        lines.append("")
    return lines


def render_markdown(data: dict[str, object]) -> str:
    lines = [
        "# Pandects Dataset Changelog",
        "",
        "<!-- GENERATED FILE — do not edit by hand.",
        "     Edit bulk/changelog/changelog.yml and run",
        "     bulk/changelog/render_changelog.py render (push_to_r2.sh does this",
        "     automatically at release time). -->",
        "",
        "Change history for the public Pandects database dumps, the REST API, and",
        "the MCP surface. Machine-readable version:",
        "<https://bulk.pandects.org/dumps/changelog.json>.",
        "",
    ]
    lines.extend(_render_body(data, include_unreleased=True))
    return "\n".join(lines).rstrip() + "\n"


def render_docs_guide(data: dict[str, object]) -> str:
    lines = [
        "---",
        "id: changelog",
        "title: Changelog",
        "description: Dataset, schema, and API change history per bulk release.",
        "sidebar_position: 5",
        "---",
        "",
        "{/* GENERATED FILE — do not edit by hand.",
        "",
        "    Source of truth: bulk/changelog/changelog.yml. Regenerated by",
        "    bulk/changelog/render_changelog.py (run by bulk/push_to_r2.sh at",
        "    release time). */}",
        "",
        "# Changelog",
        "",
        "Change history for the public Pandects database dumps, the REST API, and the",
        "MCP surface. Each release corresponds to one published bulk dump; the",
        "`SHA-256` line matches the `X-Pandects-Dump-Hash` header the API returns on",
        "every response, so you can tell exactly which changes the data you are",
        "querying includes.",
        "",
        "Machine-readable formats:",
        "",
        "- [`changelog.json`](https://bulk.pandects.org/dumps/changelog.json) — full",
        "  history, published next to the dumps",
        "- `GET /v1/changelog` — same data via the API, filterable with",
        "  `?since=<version>` or `?dump_sha256=<hash>`",
        "",
    ]
    # The docs site documents published releases only; unreleased entries are
    # repo-internal until the next push.
    lines.extend(_render_body(data, include_unreleased=False))
    return "\n".join(lines).rstrip() + "\n"


def render_json(data: dict[str, object]) -> str:
    releases = data.get("releases") or []
    latest = releases[0] if isinstance(releases, list) and releases else None
    payload = {
        "latest_version": latest.get("version") if isinstance(latest, dict) else None,
        "latest_released": latest.get("released") if isinstance(latest, dict) else None,
        "releases": releases,
    }
    return json.dumps(payload, indent=2, default=str) + "\n"


def _write_derived(data: dict[str, object], json_out: Path | None) -> None:
    MARKDOWN_OUT_PATH.write_text(render_markdown(data))
    print(f"📝 Wrote {MARKDOWN_OUT_PATH}")
    DOCS_GUIDE_OUT_PATH.write_text(render_docs_guide(data))
    print(f"📝 Wrote {DOCS_GUIDE_OUT_PATH}")
    if json_out is not None:
        json_out.write_text(render_json(data))
        print(f"📝 Wrote {json_out}")


# ── changelog.yml rewrite (release roll-up) ──────────────────────


def _yaml_header(path: Path | None = None) -> str:
    path = path if path is not None else CHANGELOG_PATH
    lines: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            lines.append(line)
        else:
            break
    return "\n".join(lines).rstrip() + "\n\n"


def write_changelog(data: dict[str, object], path: Path | None = None) -> None:
    path = path if path is not None else CHANGELOG_PATH
    body = yaml.safe_dump(
        {"unreleased": data.get("unreleased") or [], "releases": data.get("releases") or []},
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=88,
    )
    path.write_text(_yaml_header(path) + body)


# ── Gates (precheck) ─────────────────────────────────────────────


def _schema_fingerprint(dbml_path: Path) -> str:
    return hashlib.sha256(dbml_path.read_bytes()).hexdigest()


def _query_row_counts() -> dict[str, int]:
    # Lazy import: only precheck needs DB access (mirrors generate_schema_docs.py).
    sys.path.insert(0, str(REPO_ROOT / "bulk" / "schema_docs"))
    from generate_schema_docs import load_db_credentials  # pyright: ignore[reportMissingImports]

    import pymysql

    creds = load_db_credentials()
    connection = pymysql.connect(
        host=creds["MARIADB_HOST"],
        port=int(creds.get("MARIADB_PORT") or 3306),
        user=creds["MARIADB_USER"],
        password=creds["MARIADB_PASSWORD"],
        database=creds["MARIADB_DATABASE"],
    )
    counts: dict[str, int] = {}
    try:
        with connection.cursor() as cursor:
            for table in load_allowlist():
                cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
                counts[table] = int(cursor.fetchone()[0])
    finally:
        connection.close()
    return counts


def _historical_deltas(releases: list[dict[str, object]]) -> dict[str, list[int]]:
    deltas: dict[str, list[int]] = {}
    counted = [
        r["stats"]["row_counts"]
        for r in releases
        if isinstance(r.get("stats"), dict) and isinstance(r["stats"].get("row_counts"), dict)
    ]
    for newer, older in zip(counted, counted[1:]):
        for table, count in newer.items():
            if table in older:
                deltas.setdefault(table, []).append(abs(int(count) - int(older[table])))
    return deltas


def find_anomalous_tables(
    current: dict[str, int],
    releases: list[dict[str, object]],
) -> list[tuple[str, int, int]]:
    """Return (table, previous_count, delta) rows exceeding the soft-gate threshold."""
    if not releases:
        return []
    newest = releases[0]
    stats = newest.get("stats")
    if not isinstance(stats, dict) or not isinstance(stats.get("row_counts"), dict):
        return []
    previous: dict[str, int] = stats["row_counts"]
    history = _historical_deltas(releases)
    anomalous: list[tuple[str, int, int]] = []
    for table, prev_count in previous.items():
        if table not in current:
            continue  # dropped table => schema change => hard gate's job
        delta = abs(current[table] - int(prev_count))
        table_history = history.get(table, [])
        if len(table_history) >= _SOFT_GATE_MIN_HISTORY:
            threshold = max(2 * statistics.median(table_history), _SOFT_GATE_ABS_FLOOR)
        else:
            threshold = max(_SOFT_GATE_FRACTION * int(prev_count), _SOFT_GATE_ABS_FLOOR)
        if delta > threshold:
            anomalous.append((table, int(prev_count), current[table] - int(prev_count)))
    return anomalous


def precheck(dbml_path: Path, counts_out: Path) -> None:
    data = load_changelog()
    validate(data)
    releases = [r for r in (data.get("releases") or []) if isinstance(r, dict)]
    unreleased = [c for c in (data.get("unreleased") or []) if isinstance(c, dict)]
    unreleased_types = {str(c.get("type")) for c in unreleased}

    # Hard gate: schema changed since last release => require a `schema` entry.
    fingerprint = _schema_fingerprint(dbml_path)
    if releases:
        last_fingerprint = str(releases[0].get("schema_fingerprint", ""))
        if fingerprint != last_fingerprint and "schema" not in unreleased_types:
            _fail(
                "the public schema changed since the last release "
                f"({releases[0]['version']}) but bulk/changelog/changelog.yml has no "
                "'schema' entry under 'unreleased'. Describe the change (type: schema) "
                "and re-run."
            )

    # Soft gate: anomalous row-count movement without a `data` entry.
    counts = _query_row_counts()
    counts_out.write_text(json.dumps(counts, indent=2) + "\n")
    print(f"📊 Row counts for {len(counts)} tables written to {counts_out}")
    anomalous = find_anomalous_tables(counts, releases)
    if anomalous and "data" not in unreleased_types:
        print("⚠️  Anomalous row-count changes since the last release, and no 'data'")
        print("   entry under 'unreleased' in bulk/changelog/changelog.yml:")
        for table, prev_count, delta in anomalous:
            print(f"     {table}: {prev_count:,} -> {prev_count + delta:,} ({delta:+,})")
        print("   If this data change is worth announcing, abort and add an entry.")
        if not sys.stdin.isatty():
            _fail("anomalous data change unconfirmed (stdin is not a terminal).")
        answer = input("   Proceed without a data changelog entry? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            _fail("aborted at the data-change soft gate.")
    print("✅ Changelog prechecks passed")


# ── Release roll-up ──────────────────────────────────────────────


def release(
    *,
    version: str,
    released: str,
    dump_sha256: str,
    dump_key: str,
    dbml_path: Path,
    counts_in: Path,
    json_out: Path,
) -> None:
    data = load_changelog()
    validate(data)
    releases = [r for r in (data.get("releases") or []) if isinstance(r, dict)]
    if any(str(r.get("dump_sha256")) == dump_sha256 for r in releases):
        _fail(f"a release with dump_sha256 {dump_sha256} already exists.")
    changes = list(data.get("unreleased") or [])
    if not changes:
        changes = [
            {
                "type": "data",
                "severity": "minor",
                "summary": "Routine data refresh.",
            }
        ]
    row_counts = json.loads(counts_in.read_text())
    new_release: dict[str, object] = {
        "version": version,
        "released": released,
        "dump_sha256": dump_sha256,
        "dump_key": dump_key,
        "schema_fingerprint": _schema_fingerprint(dbml_path),
        "stats": {"row_counts": row_counts},
        "changes": changes,
    }
    data = {"unreleased": [], "releases": [new_release] + releases}
    validate(data)
    write_changelog(data)
    print(f"📦 Rolled {len(changes)} change(s) into release {version}")
    _write_derived(data, json_out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate")

    render_parser = sub.add_parser("render")
    render_parser.add_argument("--json-out", type=Path, default=None)

    precheck_parser = sub.add_parser("precheck")
    precheck_parser.add_argument("--dbml", type=Path, required=True)
    precheck_parser.add_argument("--counts-out", type=Path, required=True)

    release_parser = sub.add_parser("release")
    release_parser.add_argument("--version", required=True)
    release_parser.add_argument("--released", required=True)
    release_parser.add_argument("--dump-sha256", required=True)
    release_parser.add_argument("--dump-key", required=True)
    release_parser.add_argument("--dbml", type=Path, required=True)
    release_parser.add_argument("--counts-in", type=Path, required=True)
    release_parser.add_argument("--json-out", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "validate":
        validate(load_changelog())
        print("✅ bulk/changelog/changelog.yml is valid")
    elif args.command == "render":
        data = load_changelog()
        validate(data)
        _write_derived(data, args.json_out)
    elif args.command == "precheck":
        precheck(args.dbml, args.counts_out)
    elif args.command == "release":
        release(
            version=args.version,
            released=args.released,
            dump_sha256=args.dump_sha256,
            dump_key=args.dump_key,
            dbml_path=args.dbml,
            counts_in=args.counts_in,
            json_out=args.json_out,
        )


if __name__ == "__main__":
    main()
