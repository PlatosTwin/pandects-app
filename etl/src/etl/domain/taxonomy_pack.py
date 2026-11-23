"""
Load and validate the taxonomy packs under etl/taxonomy/.

The YAML files there are the source of truth for the span taxonomy
(docs/plans/span-taxonomy.md, docs/plans/full-taxonomy.md). This module
reads them and checks them. It needs no network and no database.

Run from the repo root:
  etl/.venv/bin/python -m etl.domain.taxonomy_pack
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "taxonomy"

PACKS = ("ma", "tax")
NAV_KINDS = ("branch", "group")
CONCEPT_KINDS = ("provision", "component", "definition", "modifier")
RELATIONS = ("exact", "narrower", "broader", "related")
TIERS = ("value", "label")
VALUE_SCHEMA_STATUSES = ("piloted", "stub")
PARTY_ROLES = ("maker", "obligor", "tested_party", "holder", "indemnitor", "subject")
CROSSWALK_PARTIES = ("target", "buyer")
# Legacy id schemes and the snapshot file that lists their ids.
LEGACY_SCHEMES = {
    "section_v1": "section_v1.yaml",
    "section_v2": "section_v2.yaml",
    "tax_v1": "tax_v1.yaml",
}

_ID_RE = re.compile(r"^(?:ma|tax)(?:\.[a-z0-9_]+)+$")
_ID_IN_TEXT_RE = re.compile(r"\b(?:ma|tax)\.[a-z0-9_]+(?:\.[a-z0-9_]+)*")
_RELEASE_RE = re.compile(r"^\d{4}\.\d+$")


@dataclass(frozen=True)
class Node:
    """One taxonomy node: a navigation node or a concept."""

    id: str
    pack: str
    kind: str
    label: str
    nav_parent: str | None
    source: str
    data: dict[str, Any]

    @property
    def is_nav(self) -> bool:
        return self.kind in NAV_KINDS

    @property
    def is_concept(self) -> bool:
        return self.kind in CONCEPT_KINDS


@dataclass(frozen=True)
class Taxonomy:
    """All packs, plus the legacy id snapshots and the DealOS links."""

    nodes: dict[str, Node]
    legacy: dict[str, dict[str, dict[str, Any]]]
    dealos_links: list[dict[str, Any]]
    load_errors: list[str]


def _read_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def load(root: Path = DEFAULT_ROOT) -> Taxonomy:
    """Read every pack file under `root`. Problems that stop a file from loading go to `load_errors`."""
    nodes: dict[str, Node] = {}
    errors: list[str] = []

    def add(raw: Any, pack: str, source: str, default_kind: str | None = None) -> None:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
            errors.append(f"{source}: a node has no string id")
            return
        node_id = raw["id"]
        if node_id in nodes:
            errors.append(f"{source}: duplicate id {node_id} (also in {nodes[node_id].source})")
            return
        nav_parent = raw.get("nav_parent")
        nodes[node_id] = Node(
            id=node_id,
            pack=pack,
            kind=str(raw.get("kind") or default_kind or ""),
            label=str(raw.get("label") or ""),
            nav_parent=nav_parent if isinstance(nav_parent, str) else None,
            source=source,
            data=raw,
        )

    for pack in PACKS:
        pack_dir = root / pack
        files = sorted(pack_dir.glob("*.yaml"))
        if not files:
            errors.append(f"{pack_dir}: no YAML files")
        for path in files:
            source = f"{pack}/{path.name}"
            doc = _read_yaml(path)
            if not isinstance(doc, dict):
                errors.append(f"{source}: not a mapping")
                continue
            if doc.get("pack") != pack:
                errors.append(f"{source}: pack is {doc.get('pack')!r}, expected {pack!r}")
            if not _RELEASE_RE.match(str(doc.get("release") or "")):
                errors.append(f"{source}: release must look like 2026.1")
            add(doc.get("branch"), pack, source, "branch")
            for raw in _as_list(doc.get("groups")):
                add(raw, pack, source, "group")
            for raw in _as_list(doc.get("concepts")):
                add(raw, pack, source)

    legacy: dict[str, dict[str, dict[str, Any]]] = {}
    for scheme, filename in LEGACY_SCHEMES.items():
        path = root / "legacy" / filename
        if not path.exists():
            errors.append(f"legacy/{filename}: missing")
            legacy[scheme] = {}
            continue
        legacy[scheme] = {row["id"]: row for row in _as_list(_read_yaml(path).get("nodes"))}

    dealos_path = root / "crosswalk" / "dealos.yaml"
    dealos_links = _as_list(_read_yaml(dealos_path).get("links")) if dealos_path.exists() else []
    return Taxonomy(nodes=nodes, legacy=legacy, dealos_links=dealos_links, load_errors=errors)


def _string_list_errors(node: Node, field: str) -> Iterable[str]:
    value = node.data.get(field)
    if value is None:
        return
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        yield f"{node.id}: {field} must be a list of non-empty strings"


def _cycle_errors(nodes: dict[str, Node], field: str) -> Iterable[str]:
    """Follow `field` (nav_parent or parent) from every node and report a loop once."""
    reported: set[str] = set()
    for start in nodes:
        seen: list[str] = []
        current: str | None = start
        while current is not None and current in nodes:
            if current in seen:
                loop = seen[seen.index(current):]
                key = min(loop)
                if key not in reported:
                    reported.add(key)
                    yield f"cycle in {field}: {' -> '.join(loop + [current])}"
                break
            seen.append(current)
            nxt = nodes[current].data.get(field)
            current = nxt if isinstance(nxt, str) else None


def validate(taxonomy: Taxonomy) -> list[str]:
    """Return every problem found. An empty list means the packs are valid."""
    nodes = taxonomy.nodes
    errors = list(taxonomy.load_errors)

    for node in nodes.values():
        data = node.data
        if not _ID_RE.match(node.id):
            errors.append(f"{node.id}: id must be lowercase dotted, such as ma.governing_law")
        if not node.id.startswith(node.pack + "."):
            errors.append(f"{node.id}: id must start with its pack prefix {node.pack}.")
        if not node.label.strip():
            errors.append(f"{node.id}: label is empty")
        if not str(data.get("description") or "").strip():
            errors.append(f"{node.id}: description is empty")
        if not _RELEASE_RE.match(str(data.get("introduced_in") or "")):
            errors.append(f"{node.id}: introduced_in must look like 2026.1")
        if "deprecated_in" in data and not _RELEASE_RE.match(str(data["deprecated_in"])):
            errors.append(f"{node.id}: deprecated_in must look like 2026.1")
        for target in _as_list(data.get("replaced_by")):
            if target not in nodes:
                errors.append(f"{node.id}: replaced_by names unknown id {target}")

        if node.kind not in NAV_KINDS + CONCEPT_KINDS:
            errors.append(f"{node.id}: unknown kind {node.kind!r}")
            continue

        # Navigation parent.
        if node.kind == "branch":
            if node.nav_parent is not None:
                errors.append(f"{node.id}: a branch has no nav_parent")
        else:
            parent = nodes.get(node.nav_parent or "")
            if parent is None:
                errors.append(f"{node.id}: nav_parent {node.nav_parent!r} does not exist")
            elif parent.pack != node.pack:
                errors.append(f"{node.id}: nav_parent {parent.id} is in another pack")
            elif node.kind == "group" and parent.kind != "branch":
                errors.append(f"{node.id}: a group's nav_parent must be a branch")
            elif node.is_concept and not parent.is_nav:
                errors.append(f"{node.id}: nav_parent {parent.id} is not a navigation node")

        for target in _as_list(data.get("also_listed_under")):
            listed = nodes.get(target)
            if listed is None:
                errors.append(f"{node.id}: also_listed_under names unknown id {target}")
            elif not listed.is_nav:
                errors.append(f"{node.id}: also_listed_under {target} is not a navigation node")

        if node.is_nav:
            for field in ("parent", "tier", "value_schema", "attributes"):
                if field in data:
                    errors.append(f"{node.id}: a navigation node cannot have {field}")
        else:
            # Concept parent (part-of), separate from navigation.
            concept_parent = data.get("parent")
            if concept_parent is not None:
                target_node = nodes.get(concept_parent)
                if target_node is None:
                    errors.append(f"{node.id}: parent {concept_parent!r} does not exist")
                elif not target_node.is_concept:
                    errors.append(f"{node.id}: parent {concept_parent} is not a concept")
            elif node.kind in ("component", "modifier"):
                errors.append(f"{node.id}: a {node.kind} needs a parent concept")
            if "assignable" in data:
                errors.append(f"{node.id}: assignable applies to navigation nodes only")

            tier = data.get("tier")
            if tier not in TIERS:
                errors.append(f"{node.id}: tier must be one of {TIERS}")
            schema = data.get("value_schema")
            status = schema.get("status") if isinstance(schema, dict) else None
            if schema is not None:
                if status not in VALUE_SCHEMA_STATUSES:
                    errors.append(f"{node.id}: value_schema.status must be one of {VALUE_SCHEMA_STATUSES}")
                elif not _as_list(schema.get("fields")):
                    errors.append(f"{node.id}: value_schema.fields is empty")
            # A value concept is one whose schema has been piloted. A stub marks a candidate only.
            if (tier == "value") != (status == "piloted"):
                errors.append(f"{node.id}: tier value needs a piloted value_schema, and only then")

            attributes = data.get("attributes")
            if attributes is not None:
                if not isinstance(attributes, dict) or set(attributes) != {"party"}:
                    errors.append(f"{node.id}: attributes supports only the key party")
                elif attributes["party"] not in PARTY_ROLES:
                    errors.append(f"{node.id}: attributes.party must be one of {PARTY_ROLES}")

            for field in ("canonical_terms", "example_phrases", "exclusion_cues"):
                errors.extend(_string_list_errors(node, field))
            for cue in _as_list(data.get("exclusion_cues")):
                for ref in _ID_IN_TEXT_RE.findall(str(cue)):
                    if ref not in nodes:
                        errors.append(f"{node.id}: exclusion cue names unknown id {ref}")

        # Crosswalks to legacy ids.
        crosswalk = data.get("crosswalk")
        if crosswalk is not None and not isinstance(crosswalk, dict):
            errors.append(f"{node.id}: crosswalk must be a mapping")
            crosswalk = None
        for scheme, entries in (crosswalk or {}).items():
            if scheme not in LEGACY_SCHEMES:
                errors.append(f"{node.id}: unknown crosswalk scheme {scheme}")
                continue
            for entry in _as_list(entries):
                legacy_id = entry.get("id") if isinstance(entry, dict) else None
                if legacy_id not in taxonomy.legacy[scheme]:
                    errors.append(f"{node.id}: crosswalk id {legacy_id!r} is not in {scheme}")
                    continue
                if entry.get("relation") not in RELATIONS:
                    errors.append(f"{node.id}: crosswalk relation for {legacy_id} must be one of {RELATIONS}")
                if "party" in entry and entry["party"] not in CROSSWALK_PARTIES:
                    errors.append(f"{node.id}: crosswalk party for {legacy_id} must be one of {CROSSWALK_PARTIES}")

    errors.extend(_cycle_errors(nodes, "nav_parent"))
    errors.extend(_cycle_errors(nodes, "parent"))

    # Every legacy leaf must map somewhere, so that old filters keep working.
    mapped: dict[str, Counter[str]] = {scheme: Counter() for scheme in LEGACY_SCHEMES}
    exact: dict[str, Counter[str]] = {scheme: Counter() for scheme in LEGACY_SCHEMES}
    for node in nodes.values():
        crosswalk = node.data.get("crosswalk")
        if not isinstance(crosswalk, dict):
            continue
        for scheme, entries in crosswalk.items():
            if scheme not in LEGACY_SCHEMES:
                continue
            for entry in _as_list(entries):
                if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                    mapped[scheme][entry["id"]] += 1
                    if entry.get("relation") == "exact":
                        exact[scheme][entry["id"]] += 1
    for scheme, rows in taxonomy.legacy.items():
        for legacy_id, row in rows.items():
            if row.get("leaf") and mapped[scheme][legacy_id] == 0:
                errors.append(f"{scheme}: legacy leaf {legacy_id} ({row.get('label')}) has no crosswalk")
    # The tax tree is kept one to one: each legacy tax node has exactly one exact match.
    for legacy_id in taxonomy.legacy.get("tax_v1", {}):
        if exact["tax_v1"][legacy_id] != 1:
            errors.append(f"tax_v1: {legacy_id} must have exactly one exact crosswalk, found {exact['tax_v1'][legacy_id]}")

    for link in taxonomy.dealos_links:
        concept = link.get("concept") if isinstance(link, dict) else None
        if concept not in nodes:
            errors.append(f"crosswalk/dealos.yaml: unknown concept {concept!r}")
        elif link.get("relation") not in RELATIONS:
            errors.append(f"crosswalk/dealos.yaml: relation for {concept} must be one of {RELATIONS}")
        elif not isinstance(link.get("dealos_id"), str):
            errors.append(f"crosswalk/dealos.yaml: {concept} has no dealos_id")

    return errors


def summary(taxonomy: Taxonomy) -> dict[str, Counter[str]]:
    """Node counts by pack and kind, plus counts of value concepts and proposed nodes."""
    counts: dict[str, Counter[str]] = {pack: Counter() for pack in PACKS}
    for node in taxonomy.nodes.values():
        bucket = counts.setdefault(node.pack, Counter())
        bucket[node.kind] += 1
        if node.is_concept:
            bucket["concepts"] += 1
            if node.data.get("tier") == "value":
                bucket["value_concepts"] += 1
        if node.data.get("proposed"):
            bucket["proposed"] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the taxonomy packs under etl/taxonomy/.")
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT, help="taxonomy folder")
    args = parser.parse_args(argv)

    taxonomy = load(args.root)
    errors = validate(taxonomy)
    for pack, bucket in summary(taxonomy).items():
        parts = ", ".join(f"{name}={bucket[name]}" for name in sorted(bucket))
        print(f"{pack}: {parts}")
    if errors:
        print(f"\n{len(errors)} problem(s):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("OK: taxonomy packs are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
