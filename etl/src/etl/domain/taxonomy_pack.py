"""
Load and validate the taxonomy packs under etl/taxonomy/.

The YAML files there are the source of truth for the span taxonomy
(the span taxonomy plan of draft PR 317, https://github.com/PlatosTwin/pandects-app/pull/317,
and docs/plans/full-taxonomy.md). This module
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
# Labeling wave of an M&A concept: 1 = section-level topics and catch-alls, 2 = clause-level concepts.
WAVES = (1, 2)
VALUE_SCHEMA_STATUSES = ("piloted", "stub")
PARTY_ROLES = ("maker", "obligor", "tested_party", "holder", "indemnitor", "subject")
CROSSWALK_PARTIES = ("target", "buyer")
# Legacy id schemes and the snapshot file that lists their ids.
LEGACY_SCHEMES = {
    "section_v1": "section_v1.yaml",
    "section_v2": "section_v2.yaml",
    "tax_v1": "tax_v1.yaml",
}

# The fields each kind of node may carry (etl/taxonomy/README.md, "Fields").
COMMON_FIELDS = (
    "id",
    "label",
    "kind",
    "description",
    "also_listed_under",
    "crosswalk",
    "new_in_redesign",
    "introduced_in",
    "deprecated_in",
    "replaced_by",
)
CONCEPT_ONLY_FIELDS = (
    "parent",
    "tier",
    "wave",
    "derived",
    "attributes",
    "canonical_terms",
    "example_phrases",
    "exclusion_cues",
    "value_schema",
)
FIELDS_BY_KIND: dict[str, tuple[str, ...]] = {
    "branch": COMMON_FIELDS + ("order", "assignable"),
    "group": COMMON_FIELDS + ("nav_parent", "assignable"),
    **{kind: COMMON_FIELDS + ("nav_parent",) + CONCEPT_ONLY_FIELDS for kind in CONCEPT_KINDS},
}
KNOWN_FIELDS = frozenset(field for fields in FIELDS_BY_KIND.values() for field in fields)
FILE_FIELDS = ("pack", "release", "branch", "groups", "concepts")
VALUE_SCHEMA_FIELDS = ("status", "version", "source", "fields")
VALUE_FIELD_KEYS = ("name", "type")
CROSSWALK_ENTRY_FIELDS = ("id", "relation", "party")
# The kinds that each list of a pack file holds.
SECTION_KINDS = {"branch": ("branch",), "groups": ("group",), "concepts": CONCEPT_KINDS}

_ID_RE = re.compile(r"(?:ma|tax)(?:\.[a-z0-9_]+)+")
_ID_IN_TEXT_RE = re.compile(r"\b(?:ma|tax)\.[a-z0-9_]+(?:\.[a-z0-9_]+)*")
_RELEASE_RE = re.compile(r"\d{4}\.\d+")
_YAML_MERGE_TAG = "tag:yaml.org,2002:merge"


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
    """All packs, plus the legacy id snapshots."""

    nodes: dict[str, Node]
    legacy: dict[str, dict[str, dict[str, Any]]]
    load_errors: list[str]


class _UniqueKeyLoader(yaml.SafeLoader):
    """A safe loader that rejects a mapping that states the same key twice. Plain PyYAML keeps the last one."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if key_node.tag == _YAML_MERGE_TAG:
                continue
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError:
                continue  # An unhashable key. The base loader reports it.
            if duplicate:
                raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _yaml_problem(exc: yaml.YAMLError) -> str:
    """One line that says what is wrong and where."""
    if isinstance(exc, yaml.MarkedYAMLError):
        mark = exc.problem_mark or exc.context_mark
        where = f"line {mark.line + 1}: " if mark is not None else ""
        return f"{where}{exc.problem or exc.context or 'invalid YAML'}"
    return (str(exc).splitlines() or [type(exc).__name__])[0]


def _read_mapping(path: Path, source: str, errors: list[str]) -> dict[str, Any] | None:
    """Parse one YAML file that must hold a mapping. A problem goes to `errors`, and the result is None."""
    try:
        with path.open(encoding="utf-8") as handle:
            doc = yaml.load(handle, Loader=_UniqueKeyLoader)  # noqa: S506 - a safe loader
    except yaml.YAMLError as exc:
        errors.append(f"{source}: cannot parse YAML: {_yaml_problem(exc)}")
        return None
    except UnicodeDecodeError:
        errors.append(f"{source}: is not UTF-8 text")
        return None
    if not isinstance(doc, dict):
        errors.append(f"{source}: not a mapping")
        return None
    return doc


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _unknown_keys(mapping: dict[Any, Any], allowed: Iterable[str]) -> list[Any]:
    allowed_set = set(allowed)
    return [key for key in mapping if key not in allowed_set]


def _release_errors(owner: str, field: str, value: Any) -> Iterable[str]:
    """A release must be a quoted string. Unquoted, 2026.10 loads as the number 2026.1."""
    if value is None:
        yield f"{owner}: {field} is missing"
    elif not isinstance(value, str):
        yield f"{owner}: {field} must be a quoted string such as '2026.1', found {value!r}"
    elif not _RELEASE_RE.fullmatch(value):
        yield f"{owner}: {field} must look like 2026.1"


def _load_pack_file(path: Path, pack: str, source: str, nodes: dict[str, Node], errors: list[str]) -> None:
    doc = _read_mapping(path, source, errors)
    if doc is None:
        return
    for key in _unknown_keys(doc, FILE_FIELDS):
        errors.append(f"{source}: unknown field {key}")
    if doc.get("pack") != pack:
        errors.append(f"{source}: pack is {doc.get('pack')!r}, expected {pack!r}")
    errors.extend(_release_errors(source, "release", doc.get("release")))

    def add(raw: Any, section: str) -> None:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
            errors.append(f"{source}: a node has no string id")
            return
        node_id = raw["id"]
        if node_id in nodes:
            errors.append(f"{source}: duplicate id {node_id} (also in {nodes[node_id].source})")
            return
        expected = SECTION_KINDS[section]
        default_kind = expected[0] if len(expected) == 1 else ""
        kind = str(raw.get("kind") or default_kind)
        if kind in NAV_KINDS + CONCEPT_KINDS and kind not in expected:
            wanted = f"kind {expected[0]}" if len(expected) == 1 else "a concept kind"
            errors.append(f"{node_id}: a row under {section} must have {wanted}")
        nav_parent = raw.get("nav_parent")
        nodes[node_id] = Node(
            id=node_id,
            pack=pack,
            kind=kind,
            label=str(raw.get("label") or ""),
            nav_parent=nav_parent if isinstance(nav_parent, str) else None,
            source=source,
            data=raw,
        )

    add(doc.get("branch"), "branch")
    for section in ("groups", "concepts"):
        rows = doc.get(section)
        if rows is not None and not isinstance(rows, list):
            errors.append(f"{source}: {section} must be a list")
            continue
        for raw in rows or []:
            add(raw, section)


def _load_legacy(root: Path, errors: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    legacy: dict[str, dict[str, dict[str, Any]]] = {}
    for scheme, filename in LEGACY_SCHEMES.items():
        source = f"legacy/{filename}"
        path = root / "legacy" / filename
        rows: dict[str, dict[str, Any]] = {}
        legacy[scheme] = rows
        if not path.exists():
            errors.append(f"{source}: missing")
            continue
        doc = _read_mapping(path, source, errors)
        if doc is None:
            continue
        if not isinstance(doc.get("nodes"), list):
            errors.append(f"{source}: nodes must be a list")
            continue
        for row in doc["nodes"]:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                errors.append(f"{source}: a row has no string id")
            elif row["id"] in rows:
                errors.append(f"{source}: duplicate id {row['id']}")
            else:
                rows[row["id"]] = row
    return legacy


def load(root: Path = DEFAULT_ROOT) -> Taxonomy:
    """Read every pack file under `root`. Problems that stop a file or a row from loading go to `load_errors`."""
    nodes: dict[str, Node] = {}
    errors: list[str] = []
    for pack in PACKS:
        pack_dir = root / pack
        files = sorted(pack_dir.glob("*.yaml"))
        if not files:
            errors.append(f"{pack_dir}: no YAML files")
        for path in files:
            _load_pack_file(path, pack, f"{pack}/{path.name}", nodes, errors)
    legacy = _load_legacy(root, errors)
    return Taxonomy(nodes=nodes, legacy=legacy, load_errors=errors)


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


def _ids_in(node: Node, field: str, errors: list[str]) -> list[str]:
    """The ids that `field` lists. A value that is not a list of strings is reported."""
    value = node.data.get(field)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        errors.append(f"{node.id}: {field} must be a list of ids")
    return [item for item in _as_list(value) if isinstance(item, str)]


def _field_errors(node: Node) -> Iterable[str]:
    """Reject a field outside the documented list for the node's kind, such as a misspelled exclusion_cue."""
    allowed = FIELDS_BY_KIND.get(node.kind, KNOWN_FIELDS)
    for field in node.data:
        if field in allowed:
            continue
        if field not in KNOWN_FIELDS:
            yield f"{node.id}: unknown field {field}"
        elif node.is_nav and field in CONCEPT_ONLY_FIELDS:
            yield f"{node.id}: a navigation node cannot have {field}"
        else:
            yield f"{node.id}: a {node.kind} cannot have {field}"


def _id_form_errors(node: Node) -> Iterable[str]:
    if not _ID_RE.fullmatch(node.id):
        yield f"{node.id}: id must be lowercase dotted, such as ma.governing_law"
    if not node.id.startswith(node.pack + "."):
        yield f"{node.id}: id must start with its pack prefix {node.pack}."
    nav_prefix = f"{node.pack}.nav."
    if node.is_nav and not node.id.startswith(nav_prefix):
        yield f"{node.id}: a navigation node id must start with {nav_prefix}"
    elif node.is_concept and node.id.startswith(nav_prefix):
        yield f"{node.id}: a concept id must not start with {nav_prefix}"


def _value_field_ok(entry: Any) -> bool:
    """A value_schema field is a name, or a mapping with a name and an optional type."""
    if isinstance(entry, str):
        return bool(entry.strip())
    return (
        isinstance(entry, dict)
        and not _unknown_keys(entry, VALUE_FIELD_KEYS)
        and isinstance(entry.get("name"), str)
        and bool(entry["name"].strip())
        and isinstance(entry.get("type", ""), str)
    )


def _value_schema_errors(node: Node, schema: Any) -> Iterable[str]:
    if not isinstance(schema, dict):
        yield f"{node.id}: value_schema must be a mapping"
        return
    for key in _unknown_keys(schema, VALUE_SCHEMA_FIELDS):
        yield f"{node.id}: value_schema has unknown field {key}"
    if schema.get("status") not in VALUE_SCHEMA_STATUSES:
        yield f"{node.id}: value_schema.status must be one of {VALUE_SCHEMA_STATUSES}"
    for key in ("version", "source"):
        if key in schema and not isinstance(schema[key], str):
            yield f"{node.id}: value_schema.{key} must be a quoted string"
    fields = schema.get("fields")
    if not _as_list(fields):
        yield f"{node.id}: value_schema.fields is empty"
    for entry in _as_list(fields):
        if not _value_field_ok(entry):
            yield f"{node.id}: value_schema.fields entry {entry!r} must be a name or a mapping of name and type"


def _crosswalk_errors(node: Node, legacy: dict[str, dict[str, dict[str, Any]]]) -> Iterable[str]:
    """Check the links from a node to legacy ids."""
    crosswalk = node.data.get("crosswalk")
    if crosswalk is None:
        return
    if not isinstance(crosswalk, dict):
        yield f"{node.id}: crosswalk must be a mapping"
        return
    for scheme, entries in crosswalk.items():
        if scheme not in LEGACY_SCHEMES:
            yield f"{node.id}: unknown crosswalk scheme {scheme}"
            continue
        if not isinstance(entries, list):
            yield f"{node.id}: crosswalk {scheme} must be a list"
            continue
        for entry in entries:
            legacy_id = entry.get("id") if isinstance(entry, dict) else None
            if not isinstance(legacy_id, str) or legacy_id not in legacy[scheme]:
                yield f"{node.id}: crosswalk id {legacy_id!r} is not in {scheme}"
                continue
            for key in _unknown_keys(entry, CROSSWALK_ENTRY_FIELDS):
                yield f"{node.id}: crosswalk entry {legacy_id} in {scheme} has unknown field {key}"
            if entry.get("relation") not in RELATIONS:
                yield f"{node.id}: crosswalk relation for {legacy_id} must be one of {RELATIONS}"
            if "party" in entry and entry["party"] not in CROSSWALK_PARTIES:
                yield f"{node.id}: crosswalk party for {legacy_id} must be one of {CROSSWALK_PARTIES}"


def groups_with_wave_two_content(taxonomy: Taxonomy) -> set[str]:
    """
    Ids of the M&A groups that hold a wave 2 provision or definition.

    A part (component or modifier) does not count: its text sits inside the text of its parent concept,
    and the parent carries the label. A derived concept does not count either: it labels no spans of its own.
    A concept placed directly under a branch is an error that the validator reports, and it adds nothing here.
    """
    nodes = taxonomy.nodes
    return {
        node.nav_parent
        for node in nodes.values()
        if node.pack == "ma"
        and node.is_concept
        and node.nav_parent is not None
        and node.nav_parent in nodes
        and nodes[node.nav_parent].kind == "group"
        and node.data.get("wave") == 2
        and node.data.get("parent") is None
        and node.data.get("derived") is not True
    }


def validate(taxonomy: Taxonomy) -> list[str]:
    """Return every problem found. An empty list means the packs are valid."""
    nodes = taxonomy.nodes
    errors = list(taxonomy.load_errors)
    wave_two_groups = groups_with_wave_two_content(taxonomy)

    for node in nodes.values():
        data = node.data
        errors.extend(_id_form_errors(node))
        errors.extend(_field_errors(node))
        label = data.get("label")
        if label is not None and not isinstance(label, str):
            errors.append(f"{node.id}: label must be a string")
        elif not node.label.strip():
            errors.append(f"{node.id}: label is empty")
        description = data.get("description")
        if not isinstance(description, str) or not description.strip():
            errors.append(f"{node.id}: description must be a non-empty string")
        errors.extend(_release_errors(node.id, "introduced_in", data.get("introduced_in")))
        if "deprecated_in" in data:
            errors.extend(_release_errors(node.id, "deprecated_in", data["deprecated_in"]))
        for target in _ids_in(node, "replaced_by", errors):
            if target not in nodes:
                errors.append(f"{node.id}: replaced_by names unknown id {target}")

        if node.kind not in NAV_KINDS + CONCEPT_KINDS:
            errors.append(f"{node.id}: unknown kind {node.kind!r}")
            continue

        # Navigation parent. A branch has none: the field check reports one.
        if node.kind != "branch":
            parent = nodes.get(node.nav_parent or "")
            if parent is None:
                errors.append(f"{node.id}: nav_parent {node.nav_parent!r} does not exist")
            elif parent.pack != node.pack:
                errors.append(f"{node.id}: nav_parent {parent.id} is in another pack")
            elif node.kind == "group" and parent.kind != "branch":
                errors.append(f"{node.id}: a group's nav_parent must be a branch")
            elif node.is_concept and not parent.is_nav:
                errors.append(f"{node.id}: nav_parent {parent.id} is not a navigation node")
            elif node.is_concept and node.pack == "ma" and parent.kind == "branch":
                # Some tax concepts sit directly under a branch on purpose. Every M&A concept has a group.
                errors.append(f"{node.id}: an M&A concept must sit under a group, not under branch {parent.id}")

        for target in _ids_in(node, "also_listed_under", errors):
            listed = nodes.get(target)
            if listed is None:
                errors.append(f"{node.id}: also_listed_under names unknown id {target}")
            elif not listed.is_nav:
                errors.append(f"{node.id}: also_listed_under {target} is not a navigation node")
            elif target == node.nav_parent:
                errors.append(f"{node.id}: also_listed_under repeats its nav_parent {target}")

        if node.is_nav:
            assignable = data.get("assignable")
            if assignable is not None and not isinstance(assignable, bool):
                errors.append(f"{node.id}: assignable must be true or false")
            elif node.pack == "tax":
                # The tax labeler assigns a level 1 or level 2 node today.
                if assignable is not True:
                    errors.append(f"{node.id}: every tax navigation node needs assignable: true")
            elif node.pack == "ma":
                # In wave 1, a section that holds only wave 2 concepts gets its group as the label.
                if node.kind == "branch":
                    if assignable:
                        errors.append(f"{node.id}: an M&A branch is not assignable")
                elif node.id in wave_two_groups and not assignable:
                    errors.append(f"{node.id}: a group that holds a wave 2 provision or definition needs assignable: true")
                elif node.id not in wave_two_groups and assignable:
                    errors.append(f"{node.id}: assignable is for a group that holds a wave 2 provision or definition")
        else:
            # Concept parent (part-of), separate from navigation.
            concept_parent = data.get("parent")
            if concept_parent is not None:
                target_node = nodes.get(concept_parent) if isinstance(concept_parent, str) else None
                if not isinstance(concept_parent, str):
                    errors.append(f"{node.id}: parent must be a concept id")
                elif target_node is None:
                    errors.append(f"{node.id}: parent {concept_parent!r} does not exist")
                elif not target_node.is_concept:
                    errors.append(f"{node.id}: parent {concept_parent} is not a concept")
                elif target_node.pack != node.pack:
                    errors.append(f"{node.id}: parent {concept_parent} is in another pack")
            elif node.kind in ("component", "modifier"):
                errors.append(f"{node.id}: a {node.kind} needs a parent concept")

            # A derived concept labels no spans of its own. Its value is built from the spans of other concepts.
            derived = data.get("derived")
            if derived is not None and not isinstance(derived, bool):
                errors.append(f"{node.id}: derived must be true or false")

            tier = data.get("tier")
            if tier not in TIERS:
                errors.append(f"{node.id}: tier must be one of {TIERS}")
            schema = data.get("value_schema")
            status = schema.get("status") if isinstance(schema, dict) else None
            if schema is not None:
                errors.extend(_value_schema_errors(node, schema))
            # A value concept is one whose schema has been piloted. A stub marks a candidate only.
            if (tier == "value") != (status == "piloted"):
                errors.append(f"{node.id}: tier value needs a piloted value_schema, and only then")

            # Each M&A concept states its labeling wave. The tax pack has no waves: one tax pass labels it.
            wave = data.get("wave")
            wave_ok = isinstance(wave, int) and not isinstance(wave, bool) and wave in WAVES
            if node.pack != "ma":
                if "wave" in data:
                    errors.append(f"{node.id}: the {node.pack} pack has no waves")
            elif not wave_ok:
                errors.append(f"{node.id}: wave must be one of {WAVES}")
            elif wave == 1 and node.kind != "provision":
                errors.append(f"{node.id}: a {node.kind} belongs to wave 2")
            elif wave == 1 and concept_parent is not None:
                errors.append(f"{node.id}: a concept with a parent belongs to wave 2")

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

        errors.extend(_crosswalk_errors(node, taxonomy.legacy))

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

    return errors


def summary(taxonomy: Taxonomy) -> dict[str, Counter[str]]:
    """Node counts by pack and kind, plus counts of value, derived and wave concepts and nodes new in the redesign."""
    counts: dict[str, Counter[str]] = {pack: Counter() for pack in PACKS}
    for node in taxonomy.nodes.values():
        bucket = counts.setdefault(node.pack, Counter())
        bucket[node.kind] += 1
        if node.is_concept:
            bucket["concepts"] += 1
            if node.data.get("tier") == "value":
                bucket["value_concepts"] += 1
            if node.data.get("derived") is True:
                bucket["derived"] += 1
            if node.data.get("wave") in WAVES:
                bucket[f"wave_{node.data['wave']}"] += 1
        elif node.data.get("assignable") is True:
            bucket[f"assignable_{node.kind}"] += 1
        if node.data.get("new_in_redesign"):
            bucket["new_in_redesign"] += 1
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
