"""
Pre-embed collision check.

Computes the exact text that build_enriched_text() will hand to Voyage for
every node, then reports the most similar node pairs (TF-IDF cosine). This
catches residual retrieval collisions (verbatim Company<->Buyer rep twins,
L2<->L3 duplication) before spending Voyage credits and HPC time.

Exits non-zero if any pair is at or above --threshold.

Run from repo root:
  caffeinate -i etl/.venv/bin/python \
      etl/src/etl/models/taxonomy/deepseek/enrichment_collisions.py [--threshold 0.93]
"""

from __future__ import annotations

import argparse
import logging
import sys

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from enrichment_lib import build_enriched_text, load_enrichment_file

_REP_L1S = {
    "Representations and Warranties (Company/Seller)",
    "Representations and Warranties (Buyer/Parent)",
}


def _is_sanctioned_mirror(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """The Company<->Buyer rep mirror is an intentional design decision: the
    same concept on both sides, differentiated only by branch/party (the L1
    path token). Such a pair being near-duplicate is expected, not a defect —
    it must not fail the gate. Any other near-duplicate is a real collision."""
    return (
        a["level"] == b["level"]
        and a["label"] == b["label"]
        and a["l2_label"] == b["l2_label"]
        and a["l1_label"] != b["l1_label"]
        and {a["l1_label"], b["l1_label"]} == _REP_L1S
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.93,
                        help="Fail if any node pair has cosine >= this.")
    parser.add_argument("--top", type=int, default=25,
                        help="How many of the most-similar pairs to print.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s :: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("enrichment_collisions")

    nodes = load_enrichment_file()["nodes"]
    texts = [build_enriched_text(n) for n in nodes]
    labels = [
        f'{n["level"]} | {n["l1_label"]} > {n["l2_label"]} > {n["label"]}'
        for n in nodes
    ]

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    matrix = tfidf.fit_transform(texts)
    sim = cosine_similarity(matrix)

    pairs: list[tuple[float, int, int]] = []
    n = len(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((float(sim[i, j]), i, j))
    pairs.sort(reverse=True)

    real = [(s, i, j) for s, i, j in pairs
            if not _is_sanctioned_mirror(nodes[i], nodes[j])]
    mirror_over = [(s, i, j) for s, i, j in pairs
                   if _is_sanctioned_mirror(nodes[i], nodes[j])
                   and s >= args.threshold]

    log.info("most similar NON-mirror node pairs (TF-IDF cosine):")
    for score, i, j in real[: args.top]:
        flag = "  <<< OVER THRESHOLD" if score >= args.threshold else ""
        log.info("  %.3f  %s  ||  %s%s", score, labels[i], labels[j], flag)

    if mirror_over:
        mx = max(s for s, _, _ in mirror_over)
        log.info("expected Company<->Buyer rep mirror pairs >= %.2f: %d "
                 "(max %.3f) — disambiguated by the L1 branch path, not a defect",
                 args.threshold, len(mirror_over), mx)

    over = [(s, i, j) for s, i, j in real if s >= args.threshold]
    if over:
        log.error("FAIL: %d unintended pair(s) >= threshold %.2f",
                  len(over), args.threshold)
        return 1
    log.info("OK: no unintended pair >= threshold %.2f (max non-mirror = %.3f)",
             args.threshold, real[0][0] if real else 0.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
