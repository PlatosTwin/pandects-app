"""Final M1 pilot numbers: both extractor arms against adjudicated gold, stage 1 recall,
adjudication verdicts. Usage: python3 report.py D > m1_results.json
"""
import contextlib, glob, io, json, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
import score
from prepare import candidates

D = sys.argv[1]


def run(arm):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        score.main(D, arm, "gold", f"{D}/dis_{arm}_final.json")
    return json.loads(buf.getvalue())


verdicts = [json.loads(l) for f in sorted(glob.glob(f"{D}/adj/verdicts_*.jsonl")) for l in open(f) if l.strip()]
ambiguous = {(v["agreement"], v["concept"]) for v in verdicts if v["verdict"] == "schema_ambiguous"}

out = {"verdicts": Counter(v["verdict"] for v in verdicts), "arms": {}}
for arm in ("sonnet", "haiku"):
    rep = run(arm)
    dis = json.load(open(f"{D}/dis_{arm}_final.json"))
    # field accuracy with schema-ambiguous instances left out (their rule is not written yet)
    amb = Counter()
    for d in dis:
        if (d["agreement"], d["concept"]) in ambiguous:
            for k in d["diff"]:
                amb[f"{d['concept'].split('.')[-1]}.{k}"] += 1
    rep["fields_excluding_schema_ambiguous"] = {}
    for k, v in rep["fields"].items():
        ok, n = map(int, v.split(" = ")[0].split("/"))
        a = amb.get(k, 0)
        rep["fields_excluding_schema_ambiguous"][k] = f"{ok}/{n - a} = {ok / max(n - a, 1):.3f}"
    out["arms"][arm] = rep

# stage 1: share of gold spans that fall inside a candidate section
hit = tot = 0
missed = Counter()
for m in json.load(open(f"{D}/manifest.json")):
    u = m["uuid"]
    layer = json.load(open(f"{D}/layer/{u}.json"))
    keep = candidates(layer)
    ranges = [(s["start"], s["end"]) for s in layer["sections"] if s["uuid"] in keep]
    L = score.Layer(f"{D}/layer/{u}.json")
    for inst in json.load(open(f"{D}/out/gold/{u}.json"))["instances"]:
        if inst.get("finding") != "present":
            continue
        for role, a, b in score.span_ranges(L, inst):
            tot += 1
            if any(x <= a < y for x, y in ranges):
                hit += 1
            else:
                missed[f"{inst['concept'].split('.')[-1]}.{role}"] += 1
out["stage1_gold_span_recall"] = {"hit": hit, "total": tot, "recall": round(hit / max(tot, 1), 3), "missed_by_role": missed}
print(json.dumps(out, indent=1, default=dict))
