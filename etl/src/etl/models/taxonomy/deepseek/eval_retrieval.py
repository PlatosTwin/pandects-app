# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false
"""
Recall gate: does the hybrid retriever surface the gold leaf in its top-K?

This is the HARD GATE before any DeepSeek/HPC spend. The retriever caps the
final accuracy of everything downstream (the LLM and the future reranker can
only pick from what was retrieved), so if a gold label is not in the top-K
candidate set it is unrecoverably lost.

Reference truth = data/gold_new_taxonomy.out.jsonl (gold_generate.py: a strong
model with the FULL new taxonomy). Metrics are micro (per gold label):

  recall@K strict   : gold L3 within the top-K ranked L3 (the metric gated)
  recall@K lenient  : strict OR the gold L3's parent L2 surfaced as a backoff
                      (for L2 gold: the L2 is the parent of some top-K L3)
  MRR               : 1/(rank+1) of the gold leaf (L2 -> its best child rank)

Exit non-zero if strict recall@50 < --threshold.

Run from repo root (needs DB + VOYAGE_API_KEY):
  caffeinate -i etl/.venv/bin/python \
      etl/src/etl/models/taxonomy/deepseek/eval_retrieval.py --threshold 0.97
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import bindparam, text

from enrichment_lib import build_engine
from retrieval import HybridRetriever, RetrievalConfig, Section
from section_select import clean_body

DEEPSEEK_DIR = Path(__file__).resolve().parent
DATA_DIR = DEEPSEEK_DIR / "data"
_KS = (10, 25, 50)


def _load_gold(path: Path) -> tuple[list[dict[str, Any]], int, int]:
    scored: list[dict[str, Any]] = []
    empties = errors = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("error"):
                errors += 1
                continue
            cats = [str(c).strip() for c in (r.get("categories") or []) if str(c).strip()]
            if not cats:
                empties += 1
                continue
            scored.append({"section_uuid": str(r["section_uuid"]), "categories": cats})
    return scored, empties, errors


def _fetch_sections(section_uuids: list[str]) -> dict[str, dict[str, str]]:
    engine, schema = build_engine()
    out: dict[str, dict[str, str]] = {}
    with engine.begin() as conn:
        for i in range(0, len(section_uuids), 1000):
            chunk = section_uuids[i : i + 1000]
            rows = conn.execute(
                text(
                    f"""
                    SELECT section_uuid, article_title, section_title, xml_content
                    FROM {schema}.sections
                    WHERE section_uuid IN :uuids
                    """
                ).bindparams(bindparam("uuids", expanding=True)),
                {"uuids": tuple(chunk)},
            ).mappings().all()
            for r in rows:
                out[str(r["section_uuid"])] = {
                    "article_title": str(r["article_title"] or ""),
                    "section_title": str(r["section_title"] or ""),
                    "body": clean_body(str(r["xml_content"] or "")),
                }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default=str(DATA_DIR / "gold_new_taxonomy.out.jsonl"))
    parser.add_argument("--threshold", type=float, default=0.97,
                        help="Hard gate: strict micro recall@50 must be >= this.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s :: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("eval_retrieval")

    gold_path = Path(args.gold)
    if not gold_path.exists():
        log.error("gold file not found: %s", gold_path)
        return 2
    scored, empties, errors = _load_gold(gold_path)
    log.info("gold: %d scored sections (%d empty, %d errored skipped)",
             len(scored), empties, errors)
    if not scored:
        log.error("no scored gold sections")
        return 2

    sec_meta = _fetch_sections([s["section_uuid"] for s in scored])
    scored = [s for s in scored if s["section_uuid"] in sec_meta]
    log.info("resolved %d sections in DB", len(scored))

    # Full ranking: k=140 returns every L3 in ranked order + parent L2s.
    retriever = HybridRetriever.from_db(RetrievalConfig(k=140, include_parent_l2=True))
    node_by_id = {n.standard_id: n for n in retriever.nodes}
    l2_id_by_path: dict[tuple[str, str], str] = {
        (n.l1_label, n.l2_label): n.standard_id
        for n in retriever.nodes if n.level == "l2"
    }

    sections = [
        Section(sec_meta[s["section_uuid"]]["article_title"],
                sec_meta[s["section_uuid"]]["section_title"],
                sec_meta[s["section_uuid"]]["body"],
                s["section_uuid"])
        for s in scored
    ]
    log.info("retrieving for %d sections ...", len(sections))
    all_cands = retriever.retrieve_batch(sections)

    # Aggregates
    gate_k = 50
    n_labels = 0
    n_l2 = n_l3 = invalid = 0
    strict = {k: 0 for k in _KS}
    lenient = {k: 0 for k in _KS}
    mrr_sum = 0.0
    per_l1: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # l1 -> [hit, tot]
    leaf_stat: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # l3 -> [hit, tot]
    sec_all_hit = 0

    for s, cands in zip(scored, all_cands):
        ordered_l3 = [c.standard_id for c in cands if not c.is_backoff]  # ranked
        rank_of = {sid: i for i, sid in enumerate(ordered_l3)}
        topK_parent_l2: dict[int, set[str]] = {}
        for k in _KS:
            pl2: set[str] = set()
            for sid in ordered_l3[:k]:
                n = node_by_id[sid]
                pid = l2_id_by_path.get((n.l1_label, n.l2_label))
                if pid:
                    pl2.add(pid)
            topK_parent_l2[k] = pl2

        sec_labels_hit = True
        for g in s["categories"]:
            n_labels += 1
            node = node_by_id.get(g)
            if node is None:
                invalid += 1
                sec_labels_hit = False
                continue

            if node.level == "l3":
                n_l3 += 1
                rank = rank_of.get(g)
                parent_l2 = l2_id_by_path.get((node.l1_label, node.l2_label))
                for k in _KS:
                    sh = rank is not None and rank < k
                    strict[k] += int(sh)
                    lenient[k] += int(sh or (parent_l2 in topK_parent_l2[k]))
                best_rank = rank
                hit_gate = rank is not None and rank < gate_k
                leaf_stat[g][1] += 1
                leaf_stat[g][0] += int(hit_gate)
            else:
                n_l2 += 1
                child_ranks = [
                    i for i, sid in enumerate(ordered_l3)
                    if node_by_id[sid].l1_label == node.l1_label
                    and node_by_id[sid].l2_label == node.label
                ]
                best_rank = min(child_ranks) if child_ranks else None
                for k in _KS:
                    hit = g in topK_parent_l2[k]
                    strict[k] += int(hit)
                    lenient[k] += int(hit)
                hit_gate = g in topK_parent_l2[gate_k]

            mrr_sum += 0.0 if best_rank is None else 1.0 / (best_rank + 1)
            per_l1[node.l1_label][1] += 1
            per_l1[node.l1_label][0] += int(hit_gate)
            if not hit_gate:
                sec_labels_hit = False
        if sec_labels_hit:
            sec_all_hit += 1

    if n_labels == 0:
        log.error("no gold labels to score")
        return 2

    log.info("=" * 64)
    log.info("gold labels=%d (L3=%d, L2=%d, invalid=%d) over %d sections",
             n_labels, n_l3, n_l2, invalid, len(scored))
    for k in _KS:
        log.info("recall@%-2d  strict=%.4f  lenient=%.4f",
                 k, strict[k] / n_labels, lenient[k] / n_labels)
    log.info("MRR=%.4f  section-all-labels-hit@%d=%.4f",
             mrr_sum / n_labels, gate_k, sec_all_hit / len(scored))

    log.info("-- per-L1 recall@%d (strict) --", gate_k)
    for l1, (h, tot) in sorted(per_l1.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        log.info("  %.3f  (%d/%d)  %s", h / tot, h, tot, l1)

    worst = sorted(
        ((h / tot, h, tot, lid) for lid, (h, tot) in leaf_stat.items() if tot >= 3),
        key=lambda x: (x[0], -x[2]),
    )[:15]
    log.info("-- worst L3 leaves recall@%d (support>=3) --", gate_k)
    for rate, h, tot, lid in worst:
        n = node_by_id[lid]
        log.info("  %.3f  (%d/%d)  %s > %s", rate, h, tot, n.l2_label, n.label)

    gated = strict[50] / n_labels
    log.info("=" * 64)
    if gated >= args.threshold:
        log.info("GATE PASS: strict recall@50 = %.4f >= %.2f -> GO",
                 gated, args.threshold)
        return 0
    log.error("GATE FAIL: strict recall@50 = %.4f < %.2f -> NO-GO",
              gated, args.threshold)
    return 1


if __name__ == "__main__":
    sys.exit(main())
