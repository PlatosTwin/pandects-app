"""
Load taxonomy_enrichment.yaml into the DB sidecar tables.

Pipeline:
  1. validate the file against the live taxonomy (hard stop on any error)
  2. build the enriched document + content hash for every node
  3. diff hashes against taxonomy_node_meta / taxonomy_node_embedding
  4. upsert changed meta rows
  5. re-embed only changed/new nodes via Voyage (voyage-4-large, document)
  6. upsert embeddings

Idempotent and incremental: a no-op run re-embeds nothing. The hash covers
the recipe version and embed model, so bumping either forces a full refresh.

Run from repo root:
  caffeinate -i etl/.venv/bin/python \
      etl/src/etl/models/taxonomy/deepseek/enrichment_load.py [--dry-run]
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from enrichment_lib import (
    EMBED_MODEL,
    build_engine,
    build_enriched_text,
    content_hash,
    fetch_live_nodes,
    load_enrichment_file,
    load_envs,
    validate,
)

_VOYAGE_EMBED_BATCH_SIZE = 128


def _voyage_client() -> Any:
    load_envs()
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError("VOYAGE_API_KEY is required to embed taxonomy nodes.")
    voyageai = importlib.import_module("voyageai")
    return voyageai.Client(api_key=api_key)


def _embed_documents(client: Any, documents: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(documents), _VOYAGE_EMBED_BATCH_SIZE):
        batch = documents[i : i + _VOYAGE_EMBED_BATCH_SIZE]
        resp = client.embed(batch, model=EMBED_MODEL, input_type="document")
        out.extend(resp.embeddings)
    return out


def _existing_hashes(conn: Connection, schema: str) -> tuple[dict[str, str], dict[str, str]]:
    meta = {
        str(r[0]): str(r[1])
        for r in conn.execute(
            text(f"SELECT standard_id, content_hash FROM {schema}.taxonomy_node_meta")
        ).all()
    }
    emb = {
        str(r[0]): str(r[1])
        for r in conn.execute(
            text(f"SELECT standard_id, content_hash FROM {schema}.taxonomy_node_embedding")
        ).all()
    }
    return meta, emb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the diff and Voyage call size; write nothing.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s :: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("enrichment_load")

    file_data = load_enrichment_file()
    engine, schema = build_engine()
    with engine.begin() as conn:
        live_nodes = fetch_live_nodes(conn, schema)

    errors = validate(file_data, live_nodes)
    if errors:
        log.error("validation failed (%d errors); refusing to load. "
                  "Run enrichment_validate.py for the full report.", len(errors))
        for e in errors[:10]:
            log.error("  - %s", e)
        return 1
    log.info("validation OK: %d nodes", len(file_data["nodes"]))

    prepared: list[dict[str, Any]] = []
    for node in file_data["nodes"]:
        enriched = build_enriched_text(node)
        prepared.append(
            {
                "standard_id": node["standard_id"],
                "level": node["level"],
                "label": node["label"],
                "l1_label": node["l1_label"],
                "l2_label": node["l2_label"],
                "description": node["description"],
                "canonical_terms": json.dumps(node["canonical_terms"], ensure_ascii=False),
                "example_phrases": json.dumps(node["example_phrases"], ensure_ascii=False),
                "exclusion_cues": json.dumps(node["exclusion_cues"], ensure_ascii=False),
                "enriched_text": enriched,
                "content_hash": content_hash(node, enriched),
            }
        )

    with engine.begin() as conn:
        meta_hashes, emb_hashes = _existing_hashes(conn, schema)

    meta_changed = [p for p in prepared if meta_hashes.get(p["standard_id"]) != p["content_hash"]]
    emb_changed = [p for p in prepared if emb_hashes.get(p["standard_id"]) != p["content_hash"]]
    log.info("diff: %d/%d meta rows changed, %d/%d embeddings need refresh",
             len(meta_changed), len(prepared), len(emb_changed), len(prepared))

    if args.dry_run:
        for p in meta_changed[:5]:
            log.info("  meta change: %s (%s)", p["standard_id"], p["label"])
        log.info("dry-run: no writes, no Voyage calls")
        return 0

    if not meta_changed and not emb_changed:
        log.info("nothing to do; sidecar already in sync")
        return 0

    embeddings: dict[str, str] = {}
    if emb_changed:
        client = _voyage_client()
        log.info("embedding %d nodes via Voyage %s", len(emb_changed), EMBED_MODEL)
        vecs = _embed_documents(client, [p["enriched_text"] for p in emb_changed])
        if len(vecs) != len(emb_changed):
            raise RuntimeError("Voyage returned a mismatched embedding count")
        embeddings = {
            p["standard_id"]: json.dumps(v, separators=(",", ":"))
            for p, v in zip(emb_changed, vecs)
        }

    meta_sql = text(
        f"""
        INSERT INTO {schema}.taxonomy_node_meta
            (standard_id, level, label, l1_label, l2_label, description,
             canonical_terms, example_phrases, exclusion_cues,
             enriched_text, content_hash)
        VALUES
            (:standard_id, :level, :label, :l1_label, :l2_label, :description,
             :canonical_terms, :example_phrases, :exclusion_cues,
             :enriched_text, :content_hash)
        ON DUPLICATE KEY UPDATE
            level=VALUES(level), label=VALUES(label), l1_label=VALUES(l1_label),
            l2_label=VALUES(l2_label), description=VALUES(description),
            canonical_terms=VALUES(canonical_terms),
            example_phrases=VALUES(example_phrases),
            exclusion_cues=VALUES(exclusion_cues),
            enriched_text=VALUES(enriched_text),
            content_hash=VALUES(content_hash)
        """
    )
    emb_sql = text(
        f"""
        INSERT INTO {schema}.taxonomy_node_embedding
            (standard_id, model, embedding, content_hash)
        VALUES (:standard_id, :model, VEC_FromText(:embedding), :content_hash)
        ON DUPLICATE KEY UPDATE
            model=VALUES(model), embedding=VALUES(embedding),
            content_hash=VALUES(content_hash)
        """
    )

    with engine.begin() as conn:
        if meta_changed:
            conn.execute(meta_sql, meta_changed)
        if emb_changed:
            conn.execute(
                emb_sql,
                [
                    {
                        "standard_id": p["standard_id"],
                        "model": EMBED_MODEL,
                        "embedding": embeddings[p["standard_id"]],
                        "content_hash": p["content_hash"],
                    }
                    for p in emb_changed
                ],
            )

    log.info("done: meta upserts=%d embedding upserts=%d",
             len(meta_changed), len(emb_changed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
