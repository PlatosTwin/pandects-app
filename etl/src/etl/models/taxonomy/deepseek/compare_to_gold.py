"""
Score a DeepSeek run against the GPT gold set (same sections).

Given the per-section output of batch_inference.py (the retrieval-limited
hybrid batch) and the gold file from gold_generate.py, report how well the
DeepSeek prompt performs: exact-set agreement, micro P/R/F1 over labels,
L2-collapsed ("right family") agreement, abstention agreement, and the most
frequent disagreements — so the prompt can be judged before the full run.

Both inputs are per-section JSONL: {"section_uuid","categories","error"}.

Run from repo root:
  caffeinate -i etl/.venv/bin/python \
      etl/src/etl/models/taxonomy/deepseek/compare_to_gold.py \
      --pred data/gold_hybrid_2026-05-17.out.jsonl \
      --gold data/gold_new_taxonomy.out.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import text

from enrichment_lib import build_engine

DEEPSEEK_DIR = Path(__file__).resolve().parent
DATA_DIR = DEEPSEEK_DIR / "data"


def _load(path: Path) -> dict[str, set[str]]:
    """section_uuid -> set(categories); error rows skipped, []-rows kept."""
    out: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("error"):
                continue
            sid = r.get("section_uuid")
            if not isinstance(sid, str):
                continue
            out[sid] = {
                str(c).strip() for c in (r.get("categories") or [])
                if str(c).strip()
            }
    return out


def _taxonomy() -> dict[str, dict[str, str]]:
    engine, schema = build_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                f"""SELECT standard_id, level, label, l1_label, l2_label
                    FROM {schema}.taxonomy_node_meta"""
            )
        ).mappings().all()
    return {
        str(r["standard_id"]): {
            "level": str(r["level"]), "label": str(r["label"]),
            "l1": str(r["l1_label"]), "l2": str(r["l2_label"]),
        }
        for r in rows
    }


def _family(cat: str, tax: dict[str, dict[str, str]]) -> str:
    n = tax.get(cat)
    if not n:
        return f"?{cat}"
    return f"{n['l1']} > {n['l2']}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True,
                        help="DeepSeek batch_inference output JSONL (relative -> data/).")
    parser.add_argument("--gold", default=str(DATA_DIR / "gold_new_taxonomy.out.jsonl"))
    parser.add_argument("--top-disagreements", type=int, default=20)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s :: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("compare_to_gold")

    pred_path = Path(args.pred)
    if not pred_path.is_absolute() and not pred_path.exists():
        pred_path = DATA_DIR / args.pred
    gold = _load(Path(args.gold))
    pred = _load(pred_path)
    tax = _taxonomy()

    common = sorted(set(gold) & set(pred))
    only_gold = len(set(gold) - set(pred))
    only_pred = len(set(pred) - set(gold))
    log.info("sections: gold=%d pred=%d common=%d (gold-only=%d pred-only=%d)",
             len(gold), len(pred), len(common), only_gold, only_pred)
    if not common:
        log.error("no overlapping sections")
        return 2

    exact = 0
    fam_exact = 0
    tp = fp = fn = 0
    g_empty = p_empty = both_empty = g_ne_p_ne = 0
    g_nonempty = g_empty_total = 0
    invalid_pred = 0
    disagree: Counter[tuple[str, str]] = Counter()

    for sid in common:
        g = gold[sid]
        p = pred[sid]
        for c in p:
            if c not in tax:
                invalid_pred += 1
        if g == p:
            exact += 1
        gfam = {_family(c, tax) for c in g}
        pfam = {_family(c, tax) for c in p}
        if gfam == pfam:
            fam_exact += 1
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        if not g:
            g_empty_total += 1
            if not p:
                both_empty += 1
        else:
            g_nonempty += 1
            if p:
                g_ne_p_ne += 1
        if not g:
            g_empty += 1
        if not p:
            p_empty += 1
        # most common confusions: single-label gold vs single-label pred
        if len(g) == 1 and len(p) == 1 and g != p:
            (gc,), (pc,) = list(g), list(p)
            disagree[(gc, pc)] += 1

    n = len(common)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    log.info("=" * 64)
    log.info("exact set match      : %.4f (%d/%d)", exact / n, exact, n)
    log.info("L2-family set match  : %.4f (%d/%d)  [tolerates L3-sibling slips]",
             fam_exact / n, fam_exact, n)
    log.info("micro precision      : %.4f", prec)
    log.info("micro recall         : %.4f", rec)
    log.info("micro F1             : %.4f", f1)
    log.info("labels: tp=%d fp=%d fn=%d invalid_pred_ids=%d",
             tp, fp, fn, invalid_pred)
    log.info("-- abstention --")
    log.info("gold-empty sections       : %d", g_empty_total)
    log.info("  of which pred also []   : %d (%.3f)", both_empty,
             both_empty / g_empty_total if g_empty_total else 0.0)
    log.info("gold-nonempty sections    : %d", g_nonempty)
    log.info("  of which pred nonempty  : %d (%.3f)", g_ne_p_ne,
             g_ne_p_ne / g_nonempty if g_nonempty else 0.0)
    log.info("pred-empty total          : %d", p_empty)

    log.info("-- top single-label disagreements (gold -> pred) --")
    for (gc, pc), cnt in disagree.most_common(args.top_disagreements):
        gn = tax.get(gc, {})
        pn = tax.get(pc, {})
        log.info("  %3d  %s > %s  ->  %s > %s", cnt,
                 gn.get("l2", "?"), gn.get("label", gc),
                 pn.get("l2", "?"), pn.get("label", pc))
    log.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
