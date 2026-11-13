"""Final M3 holdout numbers (plan section 7.8): Sonnet as extracted and after the post-checks,
both against adjudicated gold, with stage 1 recall, verdicts and the post-check changes.

Usage: python3 report_m3.py D [C1,C2,...] > results.json   (concepts default to all ten;
the financing-retrieval round scores four: see pilot/README.md)
"""
import glob, json, os, re, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
import postchecks, score, score_v2

D = sys.argv[1]
concepts = sys.argv[2].split(",") if len(sys.argv) > 2 else None
out = {"agreements": len(json.load(open(f"{D}/manifest.json"))), "arms": {}}
verdicts = [json.loads(l) for f in sorted(glob.glob(f"{D}/adj/verdicts_*.jsonl")) for l in open(f) if l.strip()]
for arm in ("sonnet", "sonnet_pc"):
    rep = score_v2.main(D, arm, "gold", None, f"{D}/dis_{arm}_final.json", concepts)
    rep["detection_before_adjudication"] = score_v2.main(D, arm, "gold_raw", None, None, concepts)["detection"]
    v = [x for x in verdicts if x.get("arm") == arm]
    rep["verdicts"] = Counter(x["verdict"] for x in v)
    rep["extractor_errors"] = Counter(x.get("extractor_error") for x in v if x["verdict"] in ("gold_correct", "both_wrong"))
    out["arms"][arm] = rep

# sensitivity: percentage-of-price amounts compared on the percentage alone (score_v2.PCT_FIRST)
score_v2.PCT_FIRST = True
pf = score_v2.main(D, "sonnet", "gold", None, None, concepts)
score_v2.PCT_FIRST = False
out["pct_first_sonnet"] = {"fields": pf["fields"], "set_field_f1": pf["set_field_f1"]}

# what the post-checks changed, and whether each change agrees with adjudicated gold
changes = {}
for p in sorted(glob.glob(f"{D}/out/sonnet/*.json")):
    u = os.path.basename(p)[:-5]
    _, log = postchecks.apply(json.load(open(p)), open(f"{D}/candidates/{u}.md").read())
    if log:
        changes[u] = log
out["postcheck_changes"] = {"agreements": len(changes), "by_rule": Counter(c["rule"] for v in changes.values() for c in v),
                            "detail": changes}

# stage 1: share of gold spans inside a section the extractor was shown
hit = tot = 0
missed = Counter()
for m in json.load(open(f"{D}/manifest.json")):
    u = m["uuid"]
    if not os.path.exists(f"{D}/out/gold/{u}.json"):
        continue
    layer = json.load(open(f"{D}/layer/{u}.json"))
    secs = sorted(layer["sections"], key=lambda s: s["start"])
    keep = {int(n) for n in re.findall(r"^@@ S(\d+) ", open(f"{D}/candidates/{u}.md").read(), re.M)}
    ranges = [(s["start"], s["end"]) for n, s in enumerate(secs, 1) if n in keep]
    L = score.Layer(f"{D}/layer/{u}.json")
    for inst in json.load(open(f"{D}/out/gold/{u}.json"))["instances"]:
        if inst.get("finding") != "present" or concepts and inst["concept"] not in concepts:
            continue
        for role, a, b in score.span_ranges(L, inst):
            tot += 1
            if any(x <= a < y for x, y in ranges):
                hit += 1
            else:
                missed[inst["concept"].split(".", 1)[1]] += 1
out["stage1_gold_span_recall"] = {"hit": hit, "total": tot, "recall": round(hit / max(tot, 1), 3), "missed_by_concept": missed}
print(json.dumps(out, indent=1, default=dict))
