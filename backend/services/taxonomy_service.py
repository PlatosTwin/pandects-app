"""The clause and tax-clause taxonomies: one tree builder, cached trees, labels and counts.

``GET /v1/taxonomy``, ``GET /v1/taxonomy/tax-clauses``, the deprecated
``clause_types`` field of ``GET /v1/filter-options``, and the MCP
``get_clause_taxonomy``/``get_tax_clause_taxonomy`` tools all return the tree
built by ``build_taxonomy_tree``.

The taxonomies and the NAICS catalog change only when a dump is restored, so the
derived payloads are cached per process for ``TAXONOMY_TTL_SECONDS`` (the same
setting the taxonomy routes use for ``Cache-Control``).
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Literal, TypeVar, cast

from flask import current_app
from sqlalchemy import and_, distinct, func, not_, select, union_all

from backend.extensions import db
from backend.filtering import agreement_is_public_eligible_expr
from backend.models.main_db import (
    XML,
    Agreements,
    Clauses,
    LatestSectionsSearchStandardId,
    TaxClauseAssignment,
    TaxClauseTaxonomyL1,
    TaxClauseTaxonomyL2,
    TaxClauseTaxonomyL3,
    TaxonomyL1,
    TaxonomyL2,
    TaxonomyL3,
    xml_latest_ok_filter,
)

TaxonomyKind = Literal["clauses", "tax_clauses"]

TAXONOMY_TTL_SECONDS = int(os.environ.get("TAXONOMY_TTL_SECONDS", "21600"))

_T = TypeVar("_T")


# ---------------------------------------------------------------------------
# Tree builder
# ---------------------------------------------------------------------------


def build_taxonomy_tree(
    session: Any, *, l1_model: object, l2_model: object, l3_model: object
) -> dict[str, object]:
    """``{l1_label: {"id", "children": {l2_label: {"id", "children": {l3_label: {"id"}}}}}}``.

    Levels are sorted by label. A row with a non-string id, label, or parent id
    raises: the taxonomy tables are seeded out-of-band, and a silently dropped node
    would hide every section filed under it.
    """
    l1 = cast(Any, l1_model)
    l2 = cast(Any, l2_model)
    l3 = cast(Any, l3_model)
    l1_rows = cast(list[tuple[object, object]], session.query(l1.standard_id, l1.label).all())
    l2_rows = cast(
        list[tuple[object, object, object]],
        session.query(l2.standard_id, l2.label, l2.parent_id).all(),
    )
    l3_rows = cast(
        list[tuple[object, object, object]],
        session.query(l3.standard_id, l3.label, l3.parent_id).all(),
    )

    def _children_by_parent(
        rows: list[tuple[object, object, object]], table: str
    ) -> dict[str, list[tuple[str, str]]]:
        by_parent: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for standard_id, label, parent_id in rows:
            if (
                not isinstance(standard_id, str)
                or not isinstance(label, str)
                or not isinstance(parent_id, str)
            ):
                raise ValueError(f"{table} has invalid parent_id or label.")
            by_parent[parent_id].append((standard_id, label))
        return by_parent

    l2_by_parent = _children_by_parent(l2_rows, "taxonomy_l2")
    l3_by_parent = _children_by_parent(l3_rows, "taxonomy_l3")

    validated_l1_rows: list[tuple[str, str]] = []
    for standard_id, label in l1_rows:
        if not isinstance(standard_id, str) or not isinstance(label, str):
            raise ValueError("taxonomy_l1 has invalid standard_id or label.")
        validated_l1_rows.append((standard_id, label))

    def _by_label(row: tuple[str, str]) -> str:
        return row[1]

    tree: dict[str, object] = {}
    for l1_id, l1_label in sorted(validated_l1_rows, key=_by_label):
        l2_children: dict[str, object] = {}
        for l2_id, l2_label in sorted(l2_by_parent.get(l1_id, []), key=_by_label):
            l3_children: dict[str, object] = {}
            for l3_id, l3_label in sorted(l3_by_parent.get(l2_id, []), key=_by_label):
                l3_children[l3_label] = {"id": l3_id}
            l2_children[l2_label] = {"id": l2_id, "children": l3_children}
        tree[l1_label] = {"id": l1_id, "children": l2_children}
    return tree


def iter_taxonomy_nodes(
    tree: dict[str, object],
) -> list[tuple[str, str, tuple[str, ...]]]:
    """Flatten a tree into ``(standard_id, label, ancestor_ids)`` rows, root first."""
    nodes: list[tuple[str, str, tuple[str, ...]]] = []

    def _visit(children: dict[str, object], ancestors: tuple[str, ...]) -> None:
        for label, raw_node in children.items():
            node = cast(dict[str, object], raw_node)
            standard_id = cast(str, node["id"])
            nodes.append((standard_id, label, ancestors))
            grandchildren = node.get("children")
            if isinstance(grandchildren, dict):
                _visit(cast(dict[str, object], grandchildren), (*ancestors, standard_id))

    _visit(tree, ())
    return nodes


# ---------------------------------------------------------------------------
# Process cache
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_cache: dict[tuple[str, str, str], tuple[float, object]] = {}


def _cache_scope() -> tuple[str, str]:
    # Keyed by database so two apps in one process (the test suite) never share
    # entries.
    schema = current_app.config.get("MAIN_DB_SCHEMA", "")
    return str(db.engine.url), schema if isinstance(schema, str) else ""


def cached_reference_value(name: str, build: Callable[[], _T]) -> _T:
    """Return ``build()``, cached for ``TAXONOMY_TTL_SECONDS`` per database."""
    key = (*_cache_scope(), name)
    now = time.monotonic()
    with _cache_lock:
        entry = _cache.get(key)
    if entry is not None and now - entry[0] < TAXONOMY_TTL_SECONDS:
        return cast(_T, entry[1])
    value = build()
    with _cache_lock:
        _cache[key] = (now, value)
    return value


def clear_reference_caches() -> None:
    with _cache_lock:
        _cache.clear()


# ---------------------------------------------------------------------------
# Cached trees and labels
# ---------------------------------------------------------------------------


def _models(kind: TaxonomyKind) -> tuple[object, object, object]:
    if kind == "tax_clauses":
        return TaxClauseTaxonomyL1, TaxClauseTaxonomyL2, TaxClauseTaxonomyL3
    return TaxonomyL1, TaxonomyL2, TaxonomyL3


def taxonomy_tree(kind: TaxonomyKind = "clauses") -> dict[str, object]:
    def _build() -> dict[str, object]:
        l1_model, l2_model, l3_model = _models(kind)
        return build_taxonomy_tree(
            db.session, l1_model=l1_model, l2_model=l2_model, l3_model=l3_model
        )

    return cached_reference_value(f"taxonomy_tree:{kind}", _build)


def taxonomy_label_by_id(kind: TaxonomyKind = "clauses") -> dict[str, str]:
    """Every node's label keyed by ``standard_id`` (L1, L2 and L3)."""

    def _build() -> dict[str, str]:
        return {
            standard_id: label
            for standard_id, label, _ in iter_taxonomy_nodes(taxonomy_tree(kind))
        }

    return cached_reference_value(f"taxonomy_labels:{kind}", _build)


def standard_id_labels(
    standard_ids: object, *, kind: TaxonomyKind = "clauses"
) -> dict[str, str]:
    """``{standard_id: label}`` for the ids in ``standard_ids`` that are taxonomy nodes."""
    if not isinstance(standard_ids, (list, tuple)):
        return {}
    label_by_id = taxonomy_label_by_id(kind)
    labels: dict[str, str] = {}
    for standard_id in cast(list[object], standard_ids):
        if isinstance(standard_id, str):
            label = label_by_id.get(standard_id)
            if label is not None:
                labels[standard_id] = label
    return labels


# ---------------------------------------------------------------------------
# Corpus counts
# ---------------------------------------------------------------------------


def _section_counts_query(*, agreement_uuids: list[str] | None = None) -> Any:
    """Distinct latest sections per node; L1/L2 nodes count their descendants too."""
    assignments = LatestSectionsSearchStandardId
    # (assigned standard_id, node it counts toward): itself plus every ancestor.
    node_map = union_all(
        select(TaxonomyL3.standard_id.label("standard_id"), TaxonomyL3.standard_id.label("node_id")),
        select(TaxonomyL3.standard_id, TaxonomyL3.parent_id),
        select(TaxonomyL3.standard_id, TaxonomyL2.parent_id).join(
            TaxonomyL2, TaxonomyL2.standard_id == TaxonomyL3.parent_id
        ),
        select(TaxonomyL2.standard_id, TaxonomyL2.standard_id),
        select(TaxonomyL2.standard_id, TaxonomyL2.parent_id),
        select(TaxonomyL1.standard_id, TaxonomyL1.standard_id),
    ).subquery("node_map")
    query = select(
        node_map.c.node_id,
        func.count(distinct(assignments.section_uuid)).label("section_count"),
    ).join(node_map, node_map.c.standard_id == assignments.standard_id)
    if agreement_uuids is not None:
        query = query.where(assignments.agreement_uuid.in_(agreement_uuids))
    return query.group_by(node_map.c.node_id)


def _ineligible_agreements_with_sections() -> list[str]:
    """Gated, unverified agreements that still have rows in the standard-id map."""
    assignments = LatestSectionsSearchStandardId
    if Agreements.__table__.c.get("gated") is None:
        return []
    ineligible = select(Agreements.agreement_uuid).where(
        not_(agreement_is_public_eligible_expr(cast(Any, Agreements)))
    )
    rows = db.session.execute(
        select(distinct(assignments.agreement_uuid)).where(
            assignments.agreement_uuid.in_(ineligible)
        )
    ).all()
    return [value for (value,) in rows if isinstance(value, str)]


def taxonomy_section_counts() -> dict[str, int]:
    """Latest sections per clause-taxonomy node, as ``/v1/sections?standard_id=`` counts them.

    Search expands a parent id to its descendants, so a parent's count is the number
    of distinct sections tagged with it or any descendant. Search also excludes
    sections of gated, unverified agreements. Filtering every assignment row by
    eligibility costs seconds, so the ineligible agreements that have assignments
    are found first (a semi-join over the small gated set; normally none, because
    the map is built from eligible agreements) and their sections are counted
    separately and subtracted. A section belongs to one agreement, so the two
    distinct counts are disjoint and the difference is exact.
    """

    def _build() -> dict[str, int]:
        counts: dict[str, int] = {}
        for node_id, section_count in db.session.execute(_section_counts_query()).all():
            if isinstance(node_id, str):
                counts[node_id] = int(section_count)
        ineligible_uuids = _ineligible_agreements_with_sections()
        if ineligible_uuids:
            for node_id, section_count in db.session.execute(
                _section_counts_query(agreement_uuids=ineligible_uuids)
            ).all():
                if isinstance(node_id, str) and node_id in counts:
                    counts[node_id] -= int(section_count)
        return {node_id: count for node_id, count in counts.items() if count > 0}

    return cached_reference_value("taxonomy_section_counts", _build)


def tax_clause_taxonomy_clause_counts() -> dict[str, int]:
    """Tax clauses per tax-taxonomy node, as ``/v1/tax-clauses`` counts them by default.

    Same gates as the default tax-clause search: tax module, eligible agreement,
    latest valid XML, rep-and-warranty context excluded. A parent counts the
    distinct clauses tagged with it or any descendant.
    """

    def _build() -> dict[str, int]:
        ancestors_by_id = {
            standard_id: ancestors
            for standard_id, _, ancestors in iter_taxonomy_nodes(taxonomy_tree("tax_clauses"))
        }
        rows = db.session.execute(
            select(TaxClauseAssignment.clause_uuid, TaxClauseAssignment.standard_id)
            .join(Clauses, Clauses.clause_uuid == TaxClauseAssignment.clause_uuid)
            .join(Agreements, Agreements.agreement_uuid == Clauses.agreement_uuid)
            .join(
                XML,
                and_(
                    XML.agreement_uuid == Clauses.agreement_uuid,
                    XML.version == Clauses.xml_version,
                    xml_latest_ok_filter(),
                ),
            )
            .where(
                Clauses.module == "tax",
                Clauses.context_type != "rep_warranty",
                agreement_is_public_eligible_expr(cast(Any, Agreements)),
            )
        ).all()
        clauses_by_node: dict[str, set[str]] = defaultdict(set)
        for clause_uuid, standard_id in rows:
            if not isinstance(clause_uuid, str) or not isinstance(standard_id, str):
                continue
            ancestors = ancestors_by_id.get(standard_id)
            if ancestors is None:
                continue
            for node_id in (standard_id, *ancestors):
                clauses_by_node[node_id].add(clause_uuid)
        return {node_id: len(clause_uuids) for node_id, clause_uuids in clauses_by_node.items()}

    return cached_reference_value("tax_clause_taxonomy_clause_counts", _build)


__all__ = [
    "TAXONOMY_TTL_SECONDS",
    "TaxonomyKind",
    "build_taxonomy_tree",
    "cached_reference_value",
    "clear_reference_caches",
    "iter_taxonomy_nodes",
    "standard_id_labels",
    "tax_clause_taxonomy_clause_counts",
    "taxonomy_label_by_id",
    "taxonomy_section_counts",
    "taxonomy_tree",
]
