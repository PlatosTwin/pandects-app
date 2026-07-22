"""
Shared helpers for the hybrid-retrieval taxonomy enrichment stage.

Canonical content lives in taxonomy_enrichment.yaml (versioned, hand-reviewed).
This module centralizes:
  - DB engine construction from backend/.env
  - the live L2/L3 taxonomy snapshot (the authoritative node set)
  - the enriched-text recipe (the exact document handed to Voyage)
  - the content hash used to gate re-embedding
  - file load + structural/consistency validation against the live taxonomy

Nothing here talks to Voyage or writes to the DB; that is enrichment_load.py.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine

REPO_ROOT = Path(__file__).resolve().parents[6]
DEEPSEEK_DIR = Path(__file__).resolve().parent
BACKEND_ENV = REPO_ROOT / "backend" / ".env"
ETL_ENV = REPO_ROOT / "etl" / ".env"
ENRICHMENT_PATH = DEEPSEEK_DIR / "taxonomy_enrichment.yaml"


def load_envs() -> None:
    """Load backend/.env (authoritative for DB creds per repo convention),
    then etl/.env without override (source of VOYAGE_API_KEY locally)."""
    load_dotenv(BACKEND_ENV)
    load_dotenv(ETL_ENV, override=False)

# Bumping RECIPE_VERSION (or the embed model) changes every content hash and
# therefore forces a full re-embed on the next load. Keep in lockstep with
# build_enriched_text below.
RECIPE_VERSION = 1
EMBED_MODEL = "voyage-4-large"

# Content sanity bounds. These are guardrails against empty/garbage fields and
# accidental essays, not quality judgements.
DESCRIPTION_MIN_CHARS = 20
DESCRIPTION_MAX_CHARS = 800
CANONICAL_TERMS_MIN = 1
CANONICAL_TERMS_MAX = 30
EXAMPLE_PHRASES_MIN = 2
EXAMPLE_PHRASES_MAX = 12
EXCLUSION_CUES_MIN = 0
EXCLUSION_CUES_MAX = 20
# Per-field item length caps. example_phrases hold realistic operative clause
# snippets and are legitimately long (they feed the embedded doc + BM25);
# terms and cues must stay short and keyword-like.
_ITEM_MAX_CHARS = {
    "canonical_terms": 120,
    "example_phrases": 600,
    "exclusion_cues": 240,
}

_LIST_FIELDS = ("canonical_terms", "example_phrases", "exclusion_cues")
_REQUIRED_NODE_KEYS = (
    "standard_id",
    "level",
    "label",
    "l1_label",
    "l2_label",
    "description",
    *_LIST_FIELDS,
)


def build_engine() -> tuple[Engine, str]:
    """Write-capable engine + schema name, using backend/.env credentials."""
    load_envs()
    user = os.environ["MARIADB_USER"]
    password = os.environ["MARIADB_PASSWORD"]
    host = os.environ["MARIADB_HOST"]
    database = os.environ["MARIADB_DATABASE"]
    url = f"mysql+mysqldb://{user}:{password}@{host}/{database}?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True), database


def fetch_live_nodes(conn: Connection, schema: str) -> list[dict[str, Any]]:
    """Every assignable taxonomy node (62 L2 + 140 L3).

    L1 is never an assignable target so it is excluded. For an L2 node,
    `label` and `l2_label` are its own label; for an L3 node, `l2_label` is
    its parent L2 label. `l1_label`/`l2_label` form the human path used both
    in the embedded document and in reviewer-facing scaffolding.
    """
    rows = conn.execute(
        text(
            f"""
            SELECT l3.standard_id AS standard_id, 'l3' AS level,
                   l3.label AS label,
                   l1.label AS l1_label, l2.label AS l2_label
            FROM {schema}.taxonomy_l3 l3
            JOIN {schema}.taxonomy_l2 l2 ON l3.parent_id = l2.standard_id
            JOIN {schema}.taxonomy_l1 l1 ON l2.parent_id = l1.standard_id
            UNION ALL
            SELECT l2.standard_id AS standard_id, 'l2' AS level,
                   l2.label AS label,
                   l1.label AS l1_label, l2.label AS l2_label
            FROM {schema}.taxonomy_l2 l2
            JOIN {schema}.taxonomy_l1 l1 ON l2.parent_id = l1.standard_id
            ORDER BY l1_label, l2_label, level, label
            """
        )
    ).mappings()
    return [dict(r) for r in rows]


def build_enriched_text(node: dict[str, Any]) -> str:
    """The exact string embedded by Voyage for this node.

    Recipe (RECIPE_VERSION-gated): taxonomy path + label, then the scope
    description, canonical terms, and example phrases. Exclusion cues are
    deliberately omitted — they are negative ("this is NOT ...") signals that
    would pollute a dense vector; they are consumed only by the keyword stage
    of the retriever.
    """
    if node["level"] == "l2":
        path = f'{node["l1_label"]} > {node["label"]}'
    else:
        path = f'{node["l1_label"]} > {node["l2_label"]} > {node["label"]}'
    terms = "; ".join(node.get("canonical_terms") or [])
    examples = " ".join(node.get("example_phrases") or [])
    return (
        f"{path}\n"
        f"{node['description']}\n"
        f"Key terms: {terms}\n"
        f"Examples: {examples}"
    ).strip()


def content_hash(node: dict[str, Any], enriched_text: str) -> str:
    """sha256 over everything that determines the stored row + embedding.

    Any change to authored content, the path/label, the recipe, or the embed
    model flips the hash, which is what gates re-embedding in the loader.
    """
    payload = {
        "recipe_version": RECIPE_VERSION,
        "embed_model": EMBED_MODEL,
        "standard_id": node["standard_id"],
        "level": node["level"],
        "label": node["label"],
        "l1_label": node["l1_label"],
        "l2_label": node["l2_label"],
        "description": node["description"],
        "canonical_terms": node.get("canonical_terms") or [],
        "example_phrases": node.get("example_phrases") or [],
        "exclusion_cues": node.get("exclusion_cues") or [],
        "enriched_text": enriched_text,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_enrichment_file(path: Path = ENRICHMENT_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run enrichment_scaffold.py to create it."
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "nodes" not in data:
        raise ValueError(f"{path}: expected a mapping with a top-level 'nodes' list")
    return data


def dump_enrichment_file(data: dict[str, Any], path: Path = ENRICHMENT_PATH) -> None:
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            data,
            fh,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=100,
        )


def _validate_one(node: Any, live: dict[str, dict[str, Any]]) -> list[str]:
    errs: list[str] = []
    if not isinstance(node, dict):
        return [f"node is not a mapping: {node!r}"]
    sid = node.get("standard_id")
    tag = f"node {sid!r}"
    missing = [k for k in _REQUIRED_NODE_KEYS if k not in node]
    if missing:
        errs.append(f"{tag}: missing keys {missing}")
        return errs
    if sid not in live:
        errs.append(f"{tag}: standard_id not in live taxonomy")
        return errs
    lv = live[sid]
    for field in ("level", "label", "l1_label", "l2_label"):
        if node[field] != lv[field]:
            errs.append(
                f"{tag}: {field}={node[field]!r} does not match live {lv[field]!r}"
            )
    desc = node["description"]
    if not isinstance(desc, str) or not desc.strip():
        errs.append(f"{tag}: description empty")
    elif not (DESCRIPTION_MIN_CHARS <= len(desc.strip()) <= DESCRIPTION_MAX_CHARS):
        errs.append(
            f"{tag}: description length {len(desc.strip())} outside "
            f"[{DESCRIPTION_MIN_CHARS},{DESCRIPTION_MAX_CHARS}]"
        )
    bounds = {
        "canonical_terms": (CANONICAL_TERMS_MIN, CANONICAL_TERMS_MAX),
        "example_phrases": (EXAMPLE_PHRASES_MIN, EXAMPLE_PHRASES_MAX),
        "exclusion_cues": (EXCLUSION_CUES_MIN, EXCLUSION_CUES_MAX),
    }
    for field in _LIST_FIELDS:
        val = node[field]
        lo, hi = bounds[field]
        if not isinstance(val, list):
            errs.append(f"{tag}: {field} must be a list")
            continue
        if not (lo <= len(val) <= hi):
            errs.append(f"{tag}: {field} count {len(val)} outside [{lo},{hi}]")
        item_max = _ITEM_MAX_CHARS[field]
        for item in val:
            if not isinstance(item, str) or not item.strip():
                errs.append(f"{tag}: {field} has empty/non-string item {item!r}")
            elif len(item) > item_max:
                errs.append(f"{tag}: {field} item exceeds {item_max} chars")
    return errs


def validate(
    file_data: dict[str, Any], live_nodes: list[dict[str, Any]]
) -> list[str]:
    """Return a list of human-readable errors; empty list means valid.

    Enforces exact 1:1 coverage of the live taxonomy (no missing, no extra,
    no duplicate ids), path/label/level agreement, and per-field bounds.
    """
    errs: list[str] = []
    if file_data.get("embed_model") != EMBED_MODEL:
        errs.append(
            f"file embed_model={file_data.get('embed_model')!r} != {EMBED_MODEL!r}"
        )
    nodes = file_data.get("nodes")
    if not isinstance(nodes, list):
        return ["'nodes' is not a list"]

    live = {n["standard_id"]: n for n in live_nodes}
    seen: set[str] = set()
    for node in nodes:
        errs.extend(_validate_one(node, live))
        if isinstance(node, dict):
            sid = node.get("standard_id")
            if isinstance(sid, str):
                if sid in seen:
                    errs.append(f"duplicate standard_id {sid!r} in file")
                seen.add(sid)

    missing = sorted(set(live) - seen)
    if missing:
        errs.append(
            f"{len(missing)} live nodes missing from file (e.g. {missing[:5]})"
        )
    return errs
