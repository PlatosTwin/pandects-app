"""Run-to-run variance (plan section 7.18): two extraction runs of the same agreements, compared
with each other and with gold, field by field.

For every concept, counts the agreements where the two runs disagree on whether it is present;
for every key field (score_v2.fields) of a concept both runs found, counts the agreements where
the runs give different values, and each run's accuracy against gold.

Usage: python3 compare_runs.py D ARM_A ARM_B GOLD [--only UUIDS.json]
"""
import json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from score import load
from score_v2 import CONCEPTS, fields, show


def main(D, a, b, gold, only=None):
    uuids = [m["uuid"] for m in json.load(open(f"{D}/manifest.json"))]
    if only:
        keep = set(json.load(open(only)))
        uuids = [u for u in uuids if u in keep]
    det = defaultdict(lambda: [0, 0])  # concept -> [runs differ, agreements]
    fld = defaultdict(lambda: {"n": 0, "differ": 0, "a_ok": 0, "a_n": 0, "b_ok": 0, "b_n": 0})
    diffs = []
    for u in uuids:
        A, B, G = load(f"{D}/out/{a}", u), load(f"{D}/out/{b}", u), load(f"{D}/out/{gold}", u)
        for c in CONCEPTS:
            ia, ib, ig = (X.get(c) or {} for X in (A, B, G))
            pa, pb, pg = (i.get("finding") == "present" for i in (ia, ib, ig))
            det[c][1] += 1
            if pa != pb:
                det[c][0] += 1
                diffs.append({"agreement": u, "concept": c, "finding": [ia.get("finding"), ib.get("finding")]})
            fa = fields(c, ia) if pa else {}
            fb = fields(c, ib) if pb else {}
            fg = fields(c, ig) if pg else {}
            for k in set(fa) | set(fb):
                f = fld[f"{c.split('.', 1)[1]}.{k}"]
                if pa and pb:
                    f["n"] += 1
                    if fa[k] != fb[k]:
                        f["differ"] += 1
                        diffs.append({"agreement": u, "concept": c, "field": k,
                                      "a": show(fa[k]), "b": show(fb[k]), "gold": show(fg.get(k)) if pg else None})
                if pg and pa:
                    f["a_n"] += 1; f["a_ok"] += fa[k] == fg.get(k)
                if pg and pb:
                    f["b_n"] += 1; f["b_ok"] += fb[k] == fg.get(k)
    acc = lambda ok, n: round(ok / n, 3) if n else None
    return {"arms": [a, b], "gold": gold, "agreements": len(uuids),
            "detection_differ": {c.split(".", 1)[1]: d for c, d in det.items()},
            "fields": {k: dict(v, a_acc=acc(v["a_ok"], v["a_n"]), b_acc=acc(v["b_ok"], v["b_n"]))
                       for k, v in sorted(fld.items())},
            "field_values_compared": sum(v["n"] for v in fld.values()),
            "field_values_differ": sum(v["differ"] for v in fld.values()),
            "diffs": diffs}


if __name__ == "__main__":
    x = sys.argv[1:]
    opts = dict(zip(x[4::2], x[5::2]))
    print(json.dumps(main(*x[:4], opts.get("--only")), indent=1))
