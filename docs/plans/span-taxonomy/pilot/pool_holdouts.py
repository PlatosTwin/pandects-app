"""Pooled scores across holdouts (plan section 7.20): the bars of section 9.2 checked on the sum
of several holdouts' counts, so that a field reaches enough cases to judge.

Each ARG is D:ARM:GOLD, or D:ARM:GOLD:SPLIT to score one manifest split. Detection sums true and false positives and negatives; scalar fields
sum agreements right and compared; set fields sum the matches behind set F1. Scalar fields carry
an exact one-sided 95% lower bound on accuracy.

Usage: python3 pool_holdouts.py D1:ARM:GOLD[:SPLIT] D2:ARM:GOLD[:SPLIT] ...
"""
import json, math, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
import score_v2


def lower_bound(k, n, alpha=0.05):
    """Exact (Clopper-Pearson) one-sided lower bound on a proportion, k of n."""
    if n == 0 or k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        p = (lo + hi) / 2
        tail = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))
        lo, hi = (p, hi) if tail < alpha else (lo, p)
    return round(lo, 3)


def pool(specs):
    det = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    fld, setc = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0, 0])
    agreements = 0
    for spec in specs:
        D, arm, gold, split = (spec.split(":") + [None])[:4]
        r = score_v2.main(D, arm, gold, split)
        agreements += r["agreements"]
        for c, d in r["detection"].items():
            for k in ("tp", "fp", "fn", "tn"):
                det[c][k] += d[k]
        for k, (ok, n) in r["fields"].items():
            fld[k][0] += ok; fld[k][1] += n
        for k, v in r["set_field_counts"].items():
            for i in range(3):
                setc[k][i] += v[i]
    out = {"holdouts": specs, "agreements": agreements, "detection": {}, "fields": {}}
    for c, d in det.items():
        p, r = d["tp"] / max(d["tp"] + d["fp"], 1), d["tp"] / max(d["tp"] + d["fn"], 1)
        out["detection"][c] = dict(d, precision=round(p, 3), recall=round(r, 3),
                                   fp_per_agreement=round(d["fp"] / max(agreements, 1), 3),
                                   meets_bar=p >= 0.95 and r >= 0.95)
    for k, (ok, n) in sorted(fld.items()):
        e = {"right": ok, "n": n, "accuracy": round(ok / n, 3) if n else None, "lower95": lower_bound(ok, n)}
        if k in setc:
            tp, pr, g = setc[k]
            e["set_f1"] = round(2 * tp / max(pr + g, 1), 3)
        score = e.get("set_f1", e["accuracy"])
        e["meets_bar"] = score is not None and score >= 0.97
        out["fields"][k] = e
    return out


if __name__ == "__main__":
    print(json.dumps(pool(sys.argv[1:]), indent=1))
