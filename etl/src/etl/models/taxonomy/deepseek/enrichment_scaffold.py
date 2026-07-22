"""
Generate (or refresh) the taxonomy_enrichment.yaml scaffold from the live
taxonomy.

Default behaviour is non-destructive:
  - file absent           -> write a full scaffold with empty content fields
  - file present, --merge -> add any live nodes missing from the file and
                             drop file nodes no longer in the taxonomy, while
                             preserving every authored content field
  - file present, no flag -> refuse (use --merge or --force)

--force overwrites with a blank scaffold (discards authored content).

Run from repo root:
  caffeinate -i etl/.venv/bin/python \
      etl/src/etl/models/taxonomy/deepseek/enrichment_scaffold.py [--merge|--force]
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from enrichment_lib import (
    EMBED_MODEL,
    ENRICHMENT_PATH,
    RECIPE_VERSION,
    build_engine,
    dump_enrichment_file,
    fetch_live_nodes,
    load_enrichment_file,
)


def _blank_node(live: dict[str, Any]) -> dict[str, Any]:
    return {
        "standard_id": live["standard_id"],
        "level": live["level"],
        "label": live["label"],
        "l1_label": live["l1_label"],
        "l2_label": live["l2_label"],
        "description": "",
        "canonical_terms": [],
        "example_phrases": [],
        "exclusion_cues": [],
    }


def _build_doc(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": RECIPE_VERSION,
        "embed_model": EMBED_MODEL,
        "nodes": nodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge", action="store_true",
                        help="Preserve authored content; only add missing / "
                             "drop removed nodes and refresh path labels.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite with a blank scaffold (DESTRUCTIVE).")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s :: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("enrichment_scaffold")

    engine, schema = build_engine()
    with engine.begin() as conn:
        live_nodes = fetch_live_nodes(conn, schema)
    live_by_id = {n["standard_id"]: n for n in live_nodes}
    log.info("live taxonomy: %d assignable nodes", len(live_nodes))

    exists = ENRICHMENT_PATH.exists()
    if exists and not (args.merge or args.force):
        log.error("%s exists; pass --merge to update or --force to overwrite",
                  ENRICHMENT_PATH)
        return 2

    if args.merge and exists:
        existing = {n["standard_id"]: n for n in load_enrichment_file()["nodes"]}
        merged: list[dict[str, Any]] = []
        kept = added = 0
        for live in live_nodes:
            sid = live["standard_id"]
            if sid in existing:
                node = dict(existing[sid])
                # Refresh structural fields from the live taxonomy; never
                # touch authored content.
                node.update(
                    level=live["level"],
                    label=live["label"],
                    l1_label=live["l1_label"],
                    l2_label=live["l2_label"],
                )
                kept += 1
            else:
                node = _blank_node(live)
                added += 1
            merged.append(node)
        dropped = sorted(set(existing) - set(live_by_id))
        dump_enrichment_file(_build_doc(merged))
        log.info("merged: kept=%d added=%d dropped=%d -> %s",
                 kept, added, len(dropped), ENRICHMENT_PATH)
        if dropped:
            log.info("dropped ids no longer in taxonomy: %s", dropped)
        return 0

    dump_enrichment_file(_build_doc([_blank_node(n) for n in live_nodes]))
    log.info("wrote blank scaffold (%d nodes) -> %s",
             len(live_nodes), ENRICHMENT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
