"""Model-chosen stage 1 (plan section 7.13): a small model reads an outline of the agreement and
picks the sections to extract from, instead of pattern rules.

The outline lists every section with its article and title, its opening words, and the opening
words of each lettered or numbered paragraph. Definitions sections are listed but not offered:
their relevant terms are kept, and the terms that chosen sections use are resolved, exactly as
in prepare_v7, so the comparison isolates the choice of operative sections.

Usage: python3 router.py outline D OUT          (writes OUT/<uuid>.md for every agreement in D)
       python3 router.py candidates D PICKS OUT  (picked sections + definitions, as candidate views)
       python3 router.py score D PICKS           (gold-span recall and candidate share vs prepare_v7)
"""
import glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
import prepare_v4 as v4
import prepare_v7  # v6 patterns on v5's GROUPS and v7's cap terms, used for definitions
import prepare_v5 as v5
from prepare_v2 import candidate_view
from text_layer import DEFS_TITLE

PARA = re.compile(r"^\s*(\(?[a-zA-Z0-9]{1,4}[.)]|Section\s+\d|SECTION\s+\d|\d+(\.\d+)+)")


def is_defs(layer, s, leaves):
    return (DEFS_TITLE.search(f"{s['title'] or ''} {s['article'] or ''}") or s["uuid"].startswith("exhibit:")) \
        and sum(l["kind"] == "definition" for l in leaves) >= 5


def outline(layer):
    text, by_sec = layer["text"], {}
    for lf in layer["leaves"]:
        by_sec.setdefault(lf["section"], []).append(lf)
    out = []
    for n, s in enumerate(sorted(layer["sections"], key=lambda s: s["start"]), 1):
        leaves = by_sec.get(s["uuid"], [])
        head = f"S{n} | {(s['article'] or '').strip()[:80]} | {(s['title'] or '').strip()[:100]}"
        if is_defs(layer, s, leaves):
            out.append(head + f" | [definitions section, {len(leaves)} paragraphs: supplied automatically]")
            continue
        out.append(head)
        for i, lf in enumerate(leaves[:40]):
            t = " ".join(text[lf["start"]:lf["end"]].split())
            if i == 0:
                out.append("  > " + t[:200])
            elif PARA.match(t):
                out.append("  - " + t[:90])
    return "\n".join(out) + "\n"


def select(layer, picked):
    """Sections the model picked, plus v5's definitions handling and term resolution."""
    secs = sorted(layer["sections"], key=lambda s: s["start"])
    by_sec = {}
    for lf in layer["leaves"]:
        by_sec.setdefault(lf["section"], []).append(lf)
    groups, leaf_keep = {}, {}
    rule_groups, rule_keep = v5.select(layer)  # only its definitions-section trimming is reused
    for n, s in enumerate(secs, 1):
        if "definitions" in rule_groups.get(s["uuid"], set()):
            groups[s["uuid"]] = {"definitions"}
            leaf_keep[s["uuid"]] = rule_keep[s["uuid"]]
        elif n in picked and not is_defs(layer, s, by_sec.get(s["uuid"], [])):
            groups[s["uuid"]] = {"picked"}
    v4.resolve(layer, by_sec, groups, leaf_keep)
    return groups, leaf_keep


def score_stage1(D, picks):
    """Gold spans inside a candidate section, for the model's picks against prepare_v7."""
    import statistics
    from collections import Counter
    import score
    arms = {"v7": prepare_v7.select}
    res = {a: {"hit": 0, "total": 0, "missed": Counter(), "shares": []} for a in ("router", "v7")}
    for m in json.load(open(f"{D}/manifest.json")):
        u = m["uuid"]
        if not os.path.exists(f"{D}/out/gold/{u}.json") or not os.path.exists(f"{picks}/{u}.json"):
            continue
        layer = json.load(open(f"{D}/layer/{u}.json"))
        picked = {int(x) for v in json.load(open(f"{picks}/{u}.json")).values() for x in v}
        L = score.Layer(f"{D}/layer/{u}.json")
        gold = [i for i in json.load(open(f"{D}/out/gold/{u}.json"))["instances"] if i.get("finding") == "present"]
        for arm in res:
            groups, keep = select(layer, picked) if arm == "router" else arms[arm](layer)
            r = res[arm]
            r["shares"].append(len(candidate_view(layer, groups, keep)) / m["chars"])
            secs = [s for s in layer["sections"] if s["uuid"] in groups]
            for inst in gold:
                for role, a, b in score.span_ranges(L, inst):
                    r["total"] += 1
                    if any(s["start"] <= a < s["end"] for s in secs):
                        r["hit"] += 1
                    else:
                        r["missed"][inst["concept"].split(".", 1)[1]] += 1
    return {a: {"hit": r["hit"], "total": r["total"], "recall": round(r["hit"] / max(r["total"], 1), 4),
                "missed": dict(r["missed"]), "median_share": round(statistics.median(r["shares"]), 3)}
            for a, r in res.items()}


def main():
    cmd, D = sys.argv[1], sys.argv[2]
    if cmd == "score":
        print(json.dumps(score_stage1(D, sys.argv[3]), indent=1))
    elif cmd == "outline":
        out = sys.argv[3]
        os.makedirs(out, exist_ok=True)
        for m in json.load(open(f"{D}/manifest.json")):
            layer = json.load(open(f"{D}/layer/{m['uuid']}.json"))
            open(f"{out}/{m['uuid']}.md", "w").write(outline(layer))
    else:
        picks, out = sys.argv[3], sys.argv[4]
        os.makedirs(out, exist_ok=True)
        for p in glob.glob(f"{picks}/*.json"):
            u = os.path.basename(p)[:-5]
            if not os.path.exists(f"{D}/layer/{u}.json"):
                continue
            layer = json.load(open(f"{D}/layer/{u}.json"))
            picked = {int(x) for v in json.load(open(p)).values() for x in v}
            open(f"{out}/{u}.md", "w").write(candidate_view(layer, *select(layer, picked)))


if __name__ == "__main__":
    main()
