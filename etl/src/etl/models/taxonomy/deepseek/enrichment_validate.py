"""
Validate taxonomy_enrichment.yaml against the live taxonomy.

Checks exact 1:1 node coverage (no missing / extra / duplicate ids),
path/label/level agreement with pdx.taxonomy_l*, and per-field content
bounds. Exits non-zero with a readable report on any failure, so it can
gate enrichment_load.py and CI.

Run from repo root:
  caffeinate -i etl/.venv/bin/python \
      etl/src/etl/models/taxonomy/deepseek/enrichment_validate.py
"""

from __future__ import annotations

import argparse
import logging
import sys

from enrichment_lib import build_engine, fetch_live_nodes, load_enrichment_file, validate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-show", type=int, default=50,
                        help="Max number of errors to print.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s :: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("enrichment_validate")

    file_data = load_enrichment_file()
    engine, schema = build_engine()
    with engine.begin() as conn:
        live_nodes = fetch_live_nodes(conn, schema)

    errors = validate(file_data, live_nodes)
    n_nodes = len(file_data.get("nodes") or [])
    if errors:
        log.error("INVALID: %d node(s) in file, %d live, %d error(s):",
                  n_nodes, len(live_nodes), len(errors))
        for e in errors[: args.max_show]:
            log.error("  - %s", e)
        if len(errors) > args.max_show:
            log.error("  ... and %d more", len(errors) - args.max_show)
        return 1

    log.info("VALID: %d nodes, exact 1:1 with live taxonomy, all bounds OK",
             n_nodes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
