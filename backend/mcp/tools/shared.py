from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from difflib import SequenceMatcher
from decimal import Decimal
from html import unescape
from typing import Any, NoReturn, cast

from sqlalchemy import asc, desc, func, text
from werkzeug.exceptions import BadRequest

from backend.mcp.tools.constants import _STRUCTURED_FILTER_ARRAY_FIELDS
from backend.routes.agreements import (
    _agreement_is_public_eligible_expr,
    _normalize_industry_label,
    _to_float_or_none,
)
from backend.routes.deps import AgreementsDeps, ReferenceDataDeps
from backend.schemas.public_api import AgreementsBulkArgsPayload


class AgentVisibleBadRequest(BadRequest):
    """A 400 whose description is written for the calling agent and is safe to return.

    A BadRequest raised deeper in the stack can carry internal detail, so the MCP error
    contract deliberately keeps those opaque. Handlers raise this type instead to opt a
    specific, self-correcting message ("Invalid section_uuid: <x>") into the JSON-RPC
    error, which is the difference between an agent fixing its call and blindly retrying.
    """


def _abort_invalid_argument(description: str) -> NoReturn:
    """Reject a tool call with a reason the calling agent is allowed to see."""
    raise AgentVisibleBadRequest(description=description)


def _normalized_page(page: int) -> int:
    return page if page >= 1 else 1


def _normalized_page_size(page_size: int) -> int:
    return page_size if 1 <= page_size <= 100 else 25


def _json_compatible_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _json_compatible_structure(value: object) -> object:
    """Recursively coerce a tool result into JSON-native types.

    Handlers occasionally surface raw SQLAlchemy values (``date``/``Decimal``)
    that are JSON-serializable only at the HTTP encoding layer. Output-schema
    validation runs on the in-memory object, so normalize the whole structure
    here to keep validation and serialization in agreement regardless of which
    handler produced the payload.
    """
    if isinstance(value, dict):
        return {str(key): _json_compatible_structure(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible_structure(item) for item in value]
    return _json_compatible_value(value)


def _count_metadata_payload(
    *,
    mode: str,
    method: str,
    planning_reliability: str,
    exact_count_requested: bool,
) -> dict[str, object]:
    return {
        "mode": mode,
        "method": method,
        "planning_reliability": planning_reliability,
        "exact_count_requested": exact_count_requested,
    }


def _interpretation_payload(
    *,
    applied_filters: list[dict[str, str]],
    taxonomy_filters: list[dict[str, str]] | None = None,
    unrecognized_standard_ids: list[str] | None = None,
    heuristics_used: list[str] | None = None,
    notes: list[str] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "applied_filters": applied_filters,
        "taxonomy_filters": taxonomy_filters or [],
        "heuristics_used": heuristics_used or [],
        "notes": notes or [],
    }
    if unrecognized_standard_ids:
        payload["unrecognized_standard_ids"] = unrecognized_standard_ids
    return payload


def _agreement_filter_interpretation(
    parsed_args: AgreementsBulkArgsPayload,
    *,
    query: str,
    taxonomy_filters: list[dict[str, str]] | None = None,
    unrecognized_standard_ids: list[str] | None = None,
) -> dict[str, object]:
    applied_filters: list[dict[str, str]] = []
    for field_name in _STRUCTURED_FILTER_ARRAY_FIELDS:
        values = cast(list[object], parsed_args[field_name])
        if values:
            applied_filters.append(
                {
                    "field": field_name,
                    "representation": "first_class_agreement_field",
                    "match_kind": "exact_metadata_filter",
                }
            )
    for field_name in ("agreement_uuid", "section_uuid"):
        value = parsed_args.get(field_name)
        if isinstance(value, str) and value.strip():
            applied_filters.append(
                {
                    "field": field_name,
                    "representation": "first_class_agreement_field" if field_name == "agreement_uuid" else "first_class_section_field",
                    "match_kind": "exact_metadata_filter",
                }
            )
    for range_field in ("year_min", "year_max", "filed_after", "filed_before"):
        if parsed_args.get(range_field) is not None:
            applied_filters.append(
                {
                    "field": range_field,
                    "representation": "first_class_agreement_field",
                    "match_kind": "range_metadata_filter",
                }
            )
    heuristics_used: list[str] = []
    notes: list[str] = []
    if query:
        if query.isdigit():
            applied_filters.append(
                {
                    "field": "query",
                    "representation": "first_class_agreement_field",
                    "match_kind": "year_prefix_query",
                }
            )
            notes.append("The free-text query was interpreted as a 4-digit agreement year.")
        else:
            heuristics_used.append("prefix_name_match")
            notes.append("The free-text query uses prefix matching on target and acquirer names.")
    if taxonomy_filters:
        notes.append("Taxonomy filters reflect clause-family assignments and may act as proxies for broader legal concepts.")
    if unrecognized_standard_ids:
        notes.append(
            "Ignored unrecognized standard_id(s): "
            + ", ".join(unrecognized_standard_ids)
            + ". Use get_clause_taxonomy or suggest_clause_families to obtain valid taxonomy IDs."
        )
    return _interpretation_payload(
        applied_filters=applied_filters,
        taxonomy_filters=taxonomy_filters,
        unrecognized_standard_ids=unrecognized_standard_ids,
        heuristics_used=heuristics_used,
        notes=notes,
    )


@dataclass(frozen=True)
class _TaxonomyEntry:
    standard_id: str
    label: str
    path: tuple[str, ...]


_WHITESPACE_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")
_CONCEPT_SYNONYM_HINTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("material adverse", "mae", "material adverse effect", "material adverse change"),
        (
            "mae",
            "material adverse effect",
            "material adverse change",
            "mae carveouts",
            "disproportionate effects",
            "carveout provisions",
        ),
    ),
    (
        ("fiduciary", "recommendation", "superior proposal", "intervening event"),
        (
            "fiduciary out",
            "change of recommendation",
            "superior proposal",
            "intervening event",
            "matching rights",
        ),
    ),
    (
        ("no shop", "no-shop", "solicit", "solicitation"),
        (
            "no shop",
            "no-shop",
            "non solicitation",
            "non-solicitation",
            "matching rights",
        ),
    ),
    (
        ("specific performance", "equitable relief"),
        ("specific performance", "equitable relief"),
    ),
    (
        ("antitrust", "regulatory", "hell or high water", "efforts"),
        ("antitrust efforts", "regulatory efforts", "hell or high water", "reasonable best efforts"),
    ),
)


def _normalized_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    lowered = value.lower().replace("&", " and ")
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def _normalized_tokens(value: object) -> set[str]:
    normalized = _normalized_text(value)
    if normalized == "":
        return set()
    return {token for token in normalized.split(" ") if token}


def _taxonomy_entries(
    *,
    l1_model: object,
    l2_model: object,
    l3_model: object,
    deps: ReferenceDataDeps,
) -> list[_TaxonomyEntry]:
    db = deps.db
    l1_rows = cast(
        list[tuple[object, object]],
        db.session.query(
            cast(Any, l1_model).standard_id,
            cast(Any, l1_model).label,
        ).all(),
    )
    l2_rows = cast(
        list[tuple[object, object, object]],
        db.session.query(
            cast(Any, l2_model).standard_id,
            cast(Any, l2_model).label,
            cast(Any, l2_model).parent_id,
        ).all(),
    )
    l3_rows = cast(
        list[tuple[object, object, object]],
        db.session.query(
            cast(Any, l3_model).standard_id,
            cast(Any, l3_model).label,
            cast(Any, l3_model).parent_id,
        ).all(),
    )

    l1_by_id = {
        standard_id: label
        for standard_id, label in l1_rows
        if isinstance(standard_id, str) and isinstance(label, str)
    }
    l2_by_id = {
        standard_id: (label, parent_id)
        for standard_id, label, parent_id in l2_rows
        if isinstance(standard_id, str) and isinstance(label, str) and isinstance(parent_id, str)
    }

    entries: list[_TaxonomyEntry] = []
    for standard_id, label in sorted(l1_by_id.items(), key=lambda item: item[1]):
        entries.append(_TaxonomyEntry(standard_id=standard_id, label=label, path=(label,)))
    for standard_id, (label, parent_id) in sorted(l2_by_id.items(), key=lambda item: item[1][0]):
        parent_label = l1_by_id.get(parent_id)
        if parent_label is None:
            continue
        entries.append(_TaxonomyEntry(standard_id=standard_id, label=label, path=(parent_label, label)))
    for standard_id, label, parent_id in l3_rows:
        if not (isinstance(standard_id, str) and isinstance(label, str) and isinstance(parent_id, str)):
            continue
        parent_row = l2_by_id.get(parent_id)
        if parent_row is None:
            continue
        parent_label, grandparent_id = parent_row
        grandparent_label = l1_by_id.get(grandparent_id)
        if grandparent_label is None:
            continue
        entries.append(
            _TaxonomyEntry(
                standard_id=standard_id,
                label=label,
                path=(grandparent_label, parent_label, label),
            )
        )
    return sorted(entries, key=lambda entry: (len(entry.path), entry.path))


def _taxonomy_alias_terms(entry: _TaxonomyEntry) -> tuple[str, ...]:
    path_text = " ".join(entry.path)
    normalized_path = _normalized_text(path_text)
    aliases: set[str] = set()
    for trigger_terms, alias_terms in _CONCEPT_SYNONYM_HINTS:
        if any(trigger in normalized_path for trigger in trigger_terms):
            aliases.update(alias_terms)
    return tuple(sorted(aliases))


def _score_taxonomy_entry(entry: _TaxonomyEntry, concept: str) -> tuple[float, list[str]]:
    concept_normalized = _normalized_text(concept)
    concept_tokens = _normalized_tokens(concept)
    if concept_normalized == "":
        return 0.0, []

    search_terms = [entry.label, " > ".join(entry.path), *_taxonomy_alias_terms(entry)]
    matched_terms: list[str] = []
    best_ratio = 0.0
    token_overlap_bonus = 0.0

    for term in search_terms:
        term_normalized = _normalized_text(term)
        if term_normalized == "":
            continue
        ratio = SequenceMatcher(None, concept_normalized, term_normalized).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
        if concept_normalized in term_normalized or term_normalized in concept_normalized:
            matched_terms.append(term)
            best_ratio = max(best_ratio, 0.95)
        overlap = concept_tokens & _normalized_tokens(term)
        if overlap:
            token_overlap_bonus = max(token_overlap_bonus, len(overlap) / max(len(concept_tokens), 1))
            if term not in matched_terms:
                matched_terms.append(term)

    score = round((best_ratio * 0.7) + (token_overlap_bonus * 0.3), 4)
    return score, matched_terms[:4]


def _taxonomy_fit(score: float, *, matched_terms: list[str], entry: _TaxonomyEntry, concept: str) -> str:
    concept_normalized = _normalized_text(concept)
    entry_label = _normalized_text(entry.label)
    if concept_normalized and concept_normalized in entry_label:
        return "canonical"
    if matched_terms and score >= 0.75:
        return "proxy"
    return "broad_match"


def _taxonomy_confidence(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def _taxonomy_scope_note(*, fit: str) -> str:
    if fit == "canonical":
        return "This taxonomy node is a close canonical match for the concept."
    if fit == "proxy":
        return "This taxonomy node is a reasonable proxy and may be broader or narrower than the requested concept."
    return "This taxonomy node is a broad semantic match and should be verified before treating it as canonical."


def _taxonomy_coverage(matches: list[dict[str, object]]) -> tuple[str, str]:
    """Summarize how well a concept is actually represented in the taxonomy.

    The per-match `fit`/`confidence` describe each node in isolation; an agent
    still has to notice that the *best* available node is only a broad match — or
    even a near-opposite — of what was asked. This rolls that judgment up into a
    single honest signal so a weak or unrepresented concept is not silently
    treated as covered.
    """
    if not matches:
        return (
            "none",
            "No taxonomy nodes matched this concept; it is almost certainly not "
            "represented as a distinct clause family. Rephrase the concept or treat "
            "it as unsupported rather than filtering on an unrelated node.",
        )
    fits = {str(match.get("fit")) for match in matches}
    if "canonical" in fits:
        return (
            "canonical",
            "A close canonical taxonomy node exists for this concept; the top match is a direct fit.",
        )
    if "proxy" in fits:
        return (
            "proxy",
            "No exact node exists, but a reasonable proxy is available. Confirm the "
            "returned clause text matches your concept before relying on the filter.",
        )
    return (
        "weak",
        "No close taxonomy match. The returned nodes are broad, nearest-neighbor "
        "matches that may be adjacent to — or even the opposite of — the requested "
        "concept (for example, 'go-shop' only surfaces the distinct 'no-shop' node). "
        "Treat them as leads to verify in section text, not equivalents, and consider "
        "that this concept may not be a separate clause family in the taxonomy.",
    )


def _ranked_taxonomy_matches(
    *,
    deps: ReferenceDataDeps,
    concept: str,
    taxonomy: str,
    top_k: int,
) -> list[dict[str, object]]:
    if taxonomy == "tax_clauses":
        entries = _taxonomy_entries(
            l1_model=deps.TaxClauseTaxonomyL1,
            l2_model=deps.TaxClauseTaxonomyL2,
            l3_model=deps.TaxClauseTaxonomyL3,
            deps=deps,
        )
    else:
        entries = _taxonomy_entries(
            l1_model=deps.TaxonomyL1,
            l2_model=deps.TaxonomyL2,
            l3_model=deps.TaxonomyL3,
            deps=deps,
        )

    scored: list[tuple[float, _TaxonomyEntry, list[str]]] = []
    for entry in entries:
        score, matched_terms = _score_taxonomy_entry(entry, concept)
        if score > 0:
            scored.append((score, entry, matched_terms))
    scored.sort(key=lambda item: (-item[0], len(item[1].path), item[1].path))

    qualified_label_by_id = _qualified_labels_by_standard_id(entries)

    results: list[dict[str, object]] = []
    for score, entry, matched_terms in scored[:top_k]:
        fit = _taxonomy_fit(score, matched_terms=matched_terms, entry=entry, concept=concept)
        qualified_label = qualified_label_by_id.get(entry.standard_id, entry.label)
        results.append(
            {
                "standard_id": entry.standard_id,
                "label": entry.label,
                "qualified_label": qualified_label,
                "label_is_ambiguous": qualified_label != entry.label,
                "path": list(entry.path),
                "score": score,
                "matched_terms": matched_terms,
                "fit": fit,
                "scope_note": _taxonomy_scope_note(fit=fit),
                "confidence": _taxonomy_confidence(score),
                "reason": "Matched concept tokens and clause-family synonyms."
                if matched_terms
                else "Closest taxonomy path by label similarity.",
            }
        )
    return results


def _qualified_labels_by_standard_id(
    entries: list[_TaxonomyEntry],
) -> dict[str, str]:
    """Map standard_id -> a label that is unique across the taxonomy.

    Many leaf labels repeat verbatim under different parents (the reps taxonomy carries
    one copy under Buyer/Parent and another under Company/Seller), and the duplicates
    score identically, so a caller selecting by label alone has no way to break the tie.
    Colliding labels are qualified with the shallowest ancestor segment that actually
    differs between them; unique labels are returned unchanged.
    """
    entries_by_label: dict[str, list[_TaxonomyEntry]] = {}
    for entry in entries:
        entries_by_label.setdefault(entry.label, []).append(entry)

    qualified: dict[str, str] = {}
    for label, group in entries_by_label.items():
        if len(group) < 2:
            for entry in group:
                qualified[entry.standard_id] = label
            continue
        # The ancestor path excludes the node's own label, which is identical by definition.
        ancestries = [entry.path[:-1] if entry.path[-1:] == (label,) else entry.path for entry in group]
        depth = max((len(a) for a in ancestries), default=0)
        distinguishing_index: int | None = None
        for index in range(depth):
            segments = {a[index] if index < len(a) else None for a in ancestries}
            # A segment equal to the label itself qualifies nothing: one taxonomy branch
            # repeats the same label at L1, L2 and L3, so "X [X]" would be the result.
            if len(segments) > 1 and not segments <= {label, None}:
                distinguishing_index = index
                break
        candidates: dict[str, str] = {}
        for entry, ancestry in zip(group, ancestries):
            if distinguishing_index is not None and distinguishing_index < len(ancestry):
                candidates[entry.standard_id] = f"{label} [{ancestry[distinguishing_index]}]"
            else:
                # No ancestor separates them; the id is the only discriminator left.
                candidates[entry.standard_id] = f"{label} [{entry.standard_id}]"
        # The chosen segment can still repeat across the group (equal-length ancestries
        # that differ only deeper). Uniqueness is the whole point of this label, so any
        # value still shared falls back to the id, which is unique by construction.
        counts: dict[str, int] = {}
        for value in candidates.values():
            counts[value] = counts.get(value, 0) + 1
        for standard_id, value in candidates.items():
            qualified[standard_id] = (
                value if counts[value] == 1 else f"{label} [{standard_id}]"
            )
    return qualified


def _extract_text_from_xml(xml: object) -> str:
    if not isinstance(xml, str):
        return ""
    text_content = unescape(_TAG_RE.sub(" ", xml))
    return _WHITESPACE_RE.sub(" ", text_content).strip()


def _focused_snippet(text_content: str, *, focus_terms: list[str], max_chars: int) -> tuple[str, list[str]]:
    cleaned_text = _WHITESPACE_RE.sub(" ", text_content).strip()
    if cleaned_text == "":
        return "", []

    lowered_text = cleaned_text.lower()
    matched_terms: list[str] = []
    best_index: int | None = None
    for term in focus_terms:
        normalized_term = term.strip().lower()
        if normalized_term == "":
            continue
        index = lowered_text.find(normalized_term)
        if index >= 0:
            matched_terms.append(term)
            if best_index is None or index < best_index:
                best_index = index

    if best_index is None:
        snippet = cleaned_text[:max_chars].rstrip()
        if len(cleaned_text) > len(snippet):
            snippet = snippet.rstrip(" .,;:") + "..."
        return snippet, []

    start = max(best_index - (max_chars // 3), 0)
    end = min(start + max_chars, len(cleaned_text))
    if end - start < max_chars and start > 0:
        start = max(end - max_chars, 0)
    snippet = cleaned_text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(cleaned_text):
        snippet = snippet.rstrip(" .,;:") + "..."
    return snippet, matched_terms


# Two guards keep this from capturing sentence text as part of an amount:
#   - the digit run must end in a digit, so "$1,000,000, but" does not swallow the comma;
#   - the unit suffix needs a trailing word boundary, so the single letters do not match the
#     first character of the next word ("$3,000,000 to" -> "$3,000,000 t", "$X by wire" -> "$X b").
# Both were live defects: they corrupted ~11% of extracted values across the corpus.
_MONEY_RE = re.compile(
    r'\$\s*\d(?:[\d,]*\d)?(?:\.\d+)?'
    r'(?:\s*(?:billion|million|trillion|thousand|bn|mm|[BMTK])\b)?'
    r'|\d(?:[\d,]*\d)?(?:\.\d+)?\s+(?:billion|million|trillion|thousand)\s+dollars?',
    re.IGNORECASE,
)


_MONETARY_VALUE_LIMIT = 20


def _extract_monetary_values(text: str) -> list[str]:
    values, _ = _extract_monetary_values_with_truncation(text)
    return values


def _extract_monetary_values_with_truncation(text: str) -> tuple[list[str], bool]:
    """Extract distinct monetary values, reporting whether the cap dropped any.

    The cap bounds the payload, but a caller reading the list as "the amounts in this
    clause" cannot see that it stopped early -- the same partial-result-reads-as-complete
    failure the unresolved-id and section-truncation signals exist to prevent. The flag
    is what makes the list safe to read as a complete set when it is one.
    """
    seen: set[str] = set()
    out: list[str] = []
    truncated = False
    for m in _MONEY_RE.finditer(text):
        val = m.group(0).strip()
        norm = val.lower()
        if norm in seen:
            continue
        if len(out) >= _MONETARY_VALUE_LIMIT:
            truncated = True
            break
        seen.add(norm)
        out.append(val)
    return out, truncated


def _build_taxonomy_tree(*, l1_model: object, l2_model: object, l3_model: object, deps: ReferenceDataDeps) -> dict[str, object]:
    db = deps.db
    l1_rows = cast(
        list[tuple[object, object]],
        db.session.query(
            cast(Any, l1_model).standard_id,
            cast(Any, l1_model).label,
        ).all(),
    )
    l2_rows = cast(
        list[tuple[object, object, object]],
        db.session.query(
            cast(Any, l2_model).standard_id,
            cast(Any, l2_model).label,
            cast(Any, l2_model).parent_id,
        ).all(),
    )
    l3_rows = cast(
        list[tuple[object, object, object]],
        db.session.query(
            cast(Any, l3_model).standard_id,
            cast(Any, l3_model).label,
            cast(Any, l3_model).parent_id,
        ).all(),
    )

    l2_by_parent: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for standard_id, label, parent_id in l2_rows:
        if isinstance(standard_id, str) and isinstance(label, str) and isinstance(parent_id, str):
            l2_by_parent[parent_id].append((standard_id, label))

    l3_by_parent: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for standard_id, label, parent_id in l3_rows:
        if isinstance(standard_id, str) and isinstance(label, str) and isinstance(parent_id, str):
            l3_by_parent[parent_id].append((standard_id, label))

    validated_l1_rows: list[tuple[str, str]] = []
    for standard_id, label in l1_rows:
        if isinstance(standard_id, str) and isinstance(label, str):
            validated_l1_rows.append((standard_id, label))

    tree: dict[str, object] = {}
    for l1_standard_id, l1_label in sorted(validated_l1_rows, key=lambda row: row[1]):
        l2_children: dict[str, object] = {}
        for l2_standard_id, l2_label in sorted(l2_by_parent.get(l1_standard_id, []), key=lambda row: row[1]):
            l3_children: dict[str, object] = {}
            for l3_standard_id, l3_label in sorted(l3_by_parent.get(l2_standard_id, []), key=lambda row: row[1]):
                l3_children[l3_label] = {"id": l3_standard_id}
            l2_children[l2_label] = {"id": l2_standard_id, "children": l3_children}
        tree[l1_label] = {"id": l1_standard_id, "children": l2_children}
    return tree


def _naics_payload(deps: ReferenceDataDeps) -> dict[str, object]:
    db = deps.db
    sector_rows = cast(
        list[tuple[object, object, object, object]],
        db.session.query(
            deps.NaicsSector.sector_code,
            deps.NaicsSector.sector_desc,
            deps.NaicsSector.sector_group,
            deps.NaicsSector.super_sector,
        ).all(),
    )
    sub_sector_rows = cast(
        list[tuple[object, object, object]],
        db.session.query(
            deps.NaicsSubSector.sub_sector_code,
            deps.NaicsSubSector.sub_sector_desc,
            deps.NaicsSubSector.sector_code,
        ).all(),
    )

    sector_by_code: dict[int, dict[str, object]] = {}
    for sector_code, sector_desc, sector_group, super_sector in sector_rows:
        if not isinstance(sector_code, int):
            continue
        sector_by_code[sector_code] = {
            "sector_code": str(sector_code),
            "sector_desc": str(sector_desc or ""),
            "sector_group": str(sector_group or ""),
            "super_sector": str(super_sector or ""),
            "sub_sectors": [],
        }

    for sub_sector_code, sub_sector_desc, sector_code in sub_sector_rows:
        if not isinstance(sector_code, int) or not isinstance(sub_sector_code, int):
            continue
        parent = sector_by_code.get(sector_code)
        if parent is None:
            continue
        cast(list[dict[str, str]], parent["sub_sectors"]).append(
            {
                "sub_sector_code": str(sub_sector_code),
                "sub_sector_desc": str(sub_sector_desc or ""),
            }
        )

    sectors: list[dict[str, object]] = []
    for code in sorted(sector_by_code.keys()):
        sector = sector_by_code[code]
        sector["sub_sectors"] = sorted(
            cast(list[dict[str, str]], sector["sub_sectors"]),
            key=lambda row: int(row["sub_sector_code"]),
        )
        sectors.append(sector)
    return {"sectors": sectors}


def _naics_label_map(*, db: Any, schema_prefix: str) -> dict[str, str]:
    """Map raw NAICS sector/subsector codes to their human-readable descriptions.

    Agreement industry columns store bare codes (e.g. "334"); an LLM otherwise has
    to make a separate get_naics_catalog call to decode them. Both sector (2-digit)
    and subsector (3-digit) codes share one flat map since either can appear.
    """
    label_by_code: dict[str, str] = {}
    for table, code_col, desc_col in (
        ("naics_sectors", "sector_code", "sector_desc"),
        ("naics_sub_sectors", "sub_sector_code", "sub_sector_desc"),
    ):
        rows = db.session.execute(
            text(f"SELECT CAST({code_col} AS CHAR), {desc_col} FROM {schema_prefix}{table}")
        ).fetchall()
        for code, desc in rows:
            if code is not None and isinstance(desc, str) and desc:
                label_by_code[str(code)] = desc
    return label_by_code


def _decorate_industry_labels(
    payload: dict[str, object],
    *,
    label_by_code: dict[str, str],
    fields: tuple[str, ...] = ("target_industry", "acquirer_industry"),
) -> None:
    """Add a ``<field>_label`` sibling for each present, decodable industry code."""
    for field_name in fields:
        raw = payload.get(field_name)
        if isinstance(raw, str) and raw.strip():
            label = label_by_code.get(raw.strip())
            if label:
                payload[f"{field_name}_label"] = label


def _counsel_payload(
    deps: ReferenceDataDeps,
    *,
    query: str | None = None,
    limit: int | None = None,
) -> dict[str, object]:
    q = deps.db.session.query(
        deps.Counsel.counsel_id,
        deps.Counsel.canonical_name,
    ).order_by(deps.Counsel.canonical_name.asc(), deps.Counsel.counsel_id.asc())
    if query:
        q = q.filter(cast(Any, deps.Counsel.canonical_name).ilike(f"%{query}%"))
    if limit is not None:
        q = q.limit(limit)
    rows = cast(list[tuple[object, object]], q.all())
    payload_rows: list[dict[str, object]] = []
    for counsel_id, canonical_name in rows:
        if isinstance(counsel_id, int) and isinstance(canonical_name, str):
            payload_rows.append(
                {
                    "counsel_id": counsel_id,
                    "canonical_name": canonical_name,
                }
            )
    return {"counsel": payload_rows}


def _agreements_summary_payload(deps: AgreementsDeps) -> dict[str, object]:
    # All three figures must describe one universe: the agreements this server can
    # actually return. A payload whose fields are scoped differently invites a ratio
    # between them, and pages/agreements across mismatched scopes is silently wrong.
    #
    # summary_data already applies the public-eligibility gate; what it does not apply is
    # the join to the latest verified XML row that every retrieval tool inner-joins. Its
    # count_agreements therefore includes agreements no tool here can return, and its
    # count_pages counts those agreements' pages too. count_sections is the exception:
    # the ETL builds it through that same XML join, so it is already retrievable-scoped
    # by construction and is read from the rollup unchanged.
    prefix = deps._schema_prefix()
    row = deps.db.session.execute(
        text(
            f"""
            SELECT COALESCE(SUM(count_sections), 0) AS sections
            FROM {prefix}summary_data
            """
        )
    ).mappings().first()
    row_dict = deps._row_mapping_as_dict(cast(object, row)) if row is not None else {}

    # Counted through the same expressions the listings build, so this cannot drift from
    # what search_agreements/list_agreements return.
    agreements = deps.Agreements
    retrievable = (
        deps.db.session.query(func.count(func.distinct(agreements.agreement_uuid)))
        .join(deps.XML, deps._agreement_latest_xml_join_condition())
        .filter(_agreement_is_public_eligible_expr(agreements))
        .scalar()
    )
    # There is no Pages model to hang the same expressions off, so the predicate is
    # restated here; the summary tests pin it against the ORM count above using a fixture
    # agreement that is deliberately unreachable, which is what catches any drift.
    pages = deps.db.session.execute(
        text(
            f"""
            SELECT COUNT(*) AS pages
            FROM {prefix}pages pg
            JOIN {prefix}agreements a ON a.agreement_uuid = pg.agreement_uuid
            JOIN {prefix}xml x
              ON x.agreement_uuid = a.agreement_uuid
             AND x.latest = 1
             AND (x.status IS NULL OR x.status = 'verified')
            WHERE (COALESCE(a.gated, 0) <> 1 OR COALESCE(a.verified, 0) = 1)
            """
        )
    ).scalar()
    return {
        "agreements": deps._to_int(cast(object, retrievable)),
        "sections": deps._to_int(cast(object, row_dict.get("sections"))),
        "pages": deps._to_int(cast(object, pages)),
    }


def _year_in_range(value: object, *, year_min: int | None, year_max: int | None) -> bool:
    if not isinstance(value, int):
        return True
    if year_min is not None and value < year_min:
        return False
    if year_max is not None and value > year_max:
        return False
    return True


_YEAR_SCOPED_TREND_PATHS: dict[str, tuple[str, ...]] = {
    "ownership": ("ownership.mix_by_year", "ownership.deal_size_by_year"),
    "target_industries": ("industries.target_industries_by_year",),
}

_YEAR_UNSCOPED_TREND_PATHS: dict[str, tuple[str, ...]] = {
    "ownership": ("ownership.buyer_type_matrix",),
    "pairings": ("industries.pairings",),
}


def _agreement_trends_payload(
    deps: AgreementsDeps,
    *,
    reference_data_deps: ReferenceDataDeps,
    sections: list[str],
    year_min: int | None = None,
    year_max: int | None = None,
) -> dict[str, object]:
    requested = set(sections)
    db = deps.db
    schema_prefix = deps._schema_prefix()
    result: dict[str, object] = {"sections_returned": sorted(requested)}

    def _keep_year(row: dict[str, Any]) -> bool:
        return _year_in_range(row.get("year"), year_min=year_min, year_max=year_max)

    if "ownership" in requested:
        ownership_mix_rows = db.session.execute(
            text(
                f"""
                SELECT year, target_bucket, deal_count, total_transaction_value
                FROM {schema_prefix}agreement_ownership_mix_summary
                ORDER BY year ASC, target_bucket ASC
                """
            )
        ).mappings().all()
        ownership_deal_size_rows = db.session.execute(
            text(
                f"""
                SELECT year, target_bucket, deal_count, p25_transaction_value, median_transaction_value, p75_transaction_value
                FROM {schema_prefix}agreement_ownership_deal_size_summary
                ORDER BY year ASC, target_bucket ASC
                """
            )
        ).mappings().all()
        buyer_matrix_rows = db.session.execute(
            text(
                f"""
                SELECT target_bucket, buyer_bucket, deal_count, median_transaction_value
                FROM {schema_prefix}agreement_buyer_type_matrix_summary
                ORDER BY target_bucket ASC, buyer_bucket ASC
                """
            )
        ).mappings().all()

        ownership_mix_by_year: dict[int, dict[str, object]] = {}
        for row in ownership_mix_rows:
            year = deps._to_int(row.get("year"))
            year_row = ownership_mix_by_year.setdefault(
                year,
                {
                    "year": year,
                    "public_deal_count": 0,
                    "private_deal_count": 0,
                    "public_total_transaction_value": 0.0,
                    "private_total_transaction_value": 0.0,
                },
            )
            target_bucket = str(row.get("target_bucket") or "")
            if target_bucket == "public":
                year_row["public_deal_count"] = deps._to_int(row.get("deal_count"))
                year_row["public_total_transaction_value"] = _to_float_or_none(row.get("total_transaction_value")) or 0.0
            elif target_bucket == "private":
                year_row["private_deal_count"] = deps._to_int(row.get("deal_count"))
                year_row["private_total_transaction_value"] = _to_float_or_none(row.get("total_transaction_value")) or 0.0

        ownership_deal_size_by_year: dict[int, dict[str, object]] = {}
        for row in ownership_deal_size_rows:
            year = deps._to_int(row.get("year"))
            year_row = ownership_deal_size_by_year.setdefault(
                year,
                {
                    "year": year,
                    "public_deal_count": 0,
                    "private_deal_count": 0,
                    "public_p25_transaction_value": None,
                    "public_median_transaction_value": None,
                    "public_p75_transaction_value": None,
                    "private_p25_transaction_value": None,
                    "private_median_transaction_value": None,
                    "private_p75_transaction_value": None,
                },
            )
            target_bucket = str(row.get("target_bucket") or "")
            if target_bucket == "public":
                year_row["public_deal_count"] = deps._to_int(row.get("deal_count"))
                year_row["public_p25_transaction_value"] = _to_float_or_none(row.get("p25_transaction_value"))
                year_row["public_median_transaction_value"] = _to_float_or_none(row.get("median_transaction_value"))
                year_row["public_p75_transaction_value"] = _to_float_or_none(row.get("p75_transaction_value"))
            elif target_bucket == "private":
                year_row["private_deal_count"] = deps._to_int(row.get("deal_count"))
                year_row["private_p25_transaction_value"] = _to_float_or_none(row.get("p25_transaction_value"))
                year_row["private_median_transaction_value"] = _to_float_or_none(row.get("median_transaction_value"))
                year_row["private_p75_transaction_value"] = _to_float_or_none(row.get("p75_transaction_value"))

        buyer_matrix_lookup = {
            (
                str(row.get("target_bucket") or ""),
                str(row.get("buyer_bucket") or ""),
            ): row
            for row in buyer_matrix_rows
        }
        buyer_matrix: list[dict[str, object]] = []
        for target_bucket in ("public", "private"):
            for buyer_bucket in ("public_buyer", "private_strategic", "private_equity", "other"):
                row = buyer_matrix_lookup.get((target_bucket, buyer_bucket))
                buyer_matrix.append(
                    {
                        "target_bucket": target_bucket,
                        "buyer_bucket": buyer_bucket,
                        "deal_count": deps._to_int(row.get("deal_count")) if row else 0,
                        "median_transaction_value": (
                            _to_float_or_none(row.get("median_transaction_value")) if row else None
                        ),
                    }
                )

        result["ownership"] = {
            "mix_by_year": [
                ownership_mix_by_year[year]
                for year in sorted(ownership_mix_by_year.keys())
                if _keep_year(ownership_mix_by_year[year])
            ],
            "deal_size_by_year": [
                ownership_deal_size_by_year[year]
                for year in sorted(ownership_deal_size_by_year.keys())
                if _keep_year(ownership_deal_size_by_year[year])
            ],
            "buyer_type_matrix": buyer_matrix,
        }

    need_industry_labels = bool(requested & {"target_industries", "pairings"})
    naics_label_by_code: dict[str, str] = {}
    if need_industry_labels:
        naics_sector_rows = db.session.execute(
            text(f"SELECT sector_code, sector_desc FROM {schema_prefix}naics_sectors")
        ).mappings().all()
        naics_sub_sector_rows = db.session.execute(
            text(f"SELECT sub_sector_code, sub_sector_desc FROM {schema_prefix}naics_sub_sectors")
        ).mappings().all()
        for row in naics_sector_rows:
            sector_code = row.get("sector_code")
            sector_desc = row.get("sector_desc")
            if sector_code is not None and isinstance(sector_desc, str):
                naics_label_by_code[str(sector_code)] = sector_desc
        for row in naics_sub_sector_rows:
            sub_sector_code = row.get("sub_sector_code")
            sub_sector_desc = row.get("sub_sector_desc")
            if sub_sector_code is not None and isinstance(sub_sector_desc, str):
                naics_label_by_code[str(sub_sector_code)] = sub_sector_desc

    industries: dict[str, object] = {}
    if "target_industries" in requested:
        target_industry_rows = db.session.execute(
            text(
                f"""
                SELECT year, industry, deal_count, total_transaction_value
                FROM {schema_prefix}agreement_target_industry_summary
                ORDER BY year ASC, industry ASC
                """
            )
        ).mappings().all()
        industries["target_industries_by_year"] = [
            entry
            for row in target_industry_rows
            for entry in [
                {
                    "year": deps._to_int(row.get("year")),
                    "industry": _normalize_industry_label(
                        row.get("industry"),
                        label_by_code=naics_label_by_code,
                    ),
                    "deal_count": deps._to_int(row.get("deal_count")),
                    "total_transaction_value": _to_float_or_none(row.get("total_transaction_value")) or 0.0,
                }
            ]
            if _keep_year(entry)
        ]
    if "pairings" in requested:
        industry_pairing_rows = db.session.execute(
            text(
                f"""
                SELECT target_industry, acquirer_industry, deal_count, total_transaction_value
                FROM {schema_prefix}agreement_industry_pairing_summary
                ORDER BY deal_count DESC, total_transaction_value DESC, target_industry ASC, acquirer_industry ASC
                """
            )
        ).mappings().all()
        industries["pairings"] = [
            {
                "target_industry": _normalize_industry_label(
                    row.get("target_industry"),
                    label_by_code=naics_label_by_code,
                ),
                "acquirer_industry": _normalize_industry_label(
                    row.get("acquirer_industry"),
                    label_by_code=naics_label_by_code,
                ),
                "deal_count": deps._to_int(row.get("deal_count")),
                "total_transaction_value": _to_float_or_none(row.get("total_transaction_value")) or 0.0,
            }
            for row in industry_pairing_rows
        ]
    if industries:
        result["industries"] = industries

    if "naics_catalog" in requested:
        result["catalogs"] = {"naics": _naics_payload(reference_data_deps)}

    if year_min is not None or year_max is not None:
        applied_to: list[str] = []
        not_applied_to: list[str] = []
        for section in sorted(requested):
            applied_to.extend(_YEAR_SCOPED_TREND_PATHS.get(section, ()))
            not_applied_to.extend(_YEAR_UNSCOPED_TREND_PATHS.get(section, ()))
        result["year_filter"] = {
            "year_min": year_min,
            "year_max": year_max,
            "applied_to": applied_to,
            "not_applied_to": not_applied_to,
            "note": (
                "Paths under not_applied_to come from summary tables that are pre-aggregated "
                "across the entire corpus and carry no year dimension, so they ignore the year "
                "window and cover all years regardless of year_min/year_max."
            ),
        }

    return result

