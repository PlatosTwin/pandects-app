"""Final M2 pilot numbers: Sonnet against adjudicated gold on dev and test, stage 1
recall, adjudication verdicts, and target fees as a share of equity value.
Usage: python3 report_v2.py D [SPLIT ...] > m2_results.json   (default splits: dev test;
the M2 holdout is reported with: python3 report_v2.py D holdout > m2_holdout_results.json)
"""
import glob, json, os, re, statistics, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
import score, score_v2

D = sys.argv[1]
manifest = json.load(open(f"{D}/manifest.json"))
out = {"agreements": Counter(m["split"] for m in manifest), "splits": {}}

for split in sys.argv[2:] or ("dev", "test"):
    rep = score_v2.main(D, "sonnet", "gold", split, f"{D}/dis_{split}_final.json")
    raw = score_v2.main(D, "sonnet", "gold_raw", split)
    rep["detection_before_adjudication"] = raw["detection"]
    verdicts = [json.loads(l) for f in sorted(glob.glob(f"{D}/adj/verdicts_{split}_*.jsonl")) for l in open(f) if l.strip()]
    rep["verdicts"] = Counter(v["verdict"] for v in verdicts)
    rep["extractor_errors"] = Counter(v.get("extractor_error") for v in verdicts
                                      if v["verdict"] in ("gold_correct", "both_wrong"))
    # stage 1: share of gold spans that fall inside a section the extractor was shown
    hit = tot = 0
    missed = Counter()
    for m in manifest:
        if m["split"] != split or not os.path.exists(f"{D}/out/gold/{m['uuid']}.json"):
            continue
        u = m["uuid"]
        layer = json.load(open(f"{D}/layer/{u}.json"))
        secs = sorted(layer["sections"], key=lambda s: s["start"])
        keep = {int(n) for n in re.findall(r"^@@ S(\d+) ", open(f"{D}/candidates/{u}.md").read(), re.M)}
        ranges = [(s["start"], s["end"]) for n, s in enumerate(secs, 1) if n in keep]
        L = score.Layer(f"{D}/layer/{u}.json")
        for inst in json.load(open(f"{D}/out/gold/{u}.json"))["instances"]:
            if inst.get("finding") != "present":
                continue
            for role, a, b in score.span_ranges(L, inst):
                tot += 1
                if any(x <= a < y for x, y in ranges):
                    hit += 1
                else:
                    missed[inst["concept"].split(".", 1)[1]] += 1
    rep["stage1_gold_span_recall"] = {"hit": hit, "total": tot, "recall": round(hit / max(tot, 1), 3),
                                      "missed_by_concept": missed}
    out["splits"][split] = rep

# target fee as a share of equity value (test split; equity from agreement or web, see equity/)
eq = {r["uuid"]: r for f in sorted(glob.glob(f"{D}/equity/result_*.json")) for r in json.load(open(f))}
pcts = []
for u, r in eq.items():
    p = f"{D}/out/gold/{u}.json"
    if not r.get("equity_value_usd") or not os.path.exists(p):
        continue
    fee = next((i for i in json.load(open(p))["instances"] if i["concept"] == "ma.termination.fee.target"), {})
    usd = [a for a, c in ((a.get("amount"), a.get("currency")) for a in score.as_list(fee.get("amounts")) if isinstance(a, dict))
           if isinstance(a, (int, float)) and c in (None, "USD")]
    if fee.get("finding") == "present" and usd and not r.get("currency_note"):
        pcts.append(round(100 * max(usd) / r["equity_value_usd"], 2))
pcts.sort()
out["target_fee_pct_of_equity"] = {
    "deals_with_equity_value": sum(1 for r in eq.values() if r.get("equity_value_usd")),
    "equity_source": Counter(r["source"] for r in eq.values()),
    "n": len(pcts), "median": statistics.median(pcts) if pcts else None,
    "p10_p90": [pcts[len(pcts) // 10], pcts[(9 * len(pcts)) // 10]] if pcts else None,
    "above_8pct": [p for p in pcts if p > 8], "values": pcts}
print(json.dumps(out, indent=1, default=dict))
