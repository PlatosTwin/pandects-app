"""Score M1 pilot extractions against gold (docs/plans/span-taxonomy.md, section 9).

Usage: python3 score.py D ARM GOLD [--disagreements OUT.json]
  D     pilot data dir (layer/, out/)
  ARM   extractor output dir name under D/out (e.g. sonnet)
  GOLD  gold dir name under D/out (gold_raw or gold)

Prints a JSON report: quote validity, detection P/R, span P/R, per-field accuracy.
"""
import json, os, re, sys
from collections import defaultdict

CONCEPTS = ["ma.termination.fee.target", "ma.termination.fee.reverse", "ma.termination.outside_date"]
QUOTE_MAP = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", " ": " "})


def nrm(s):
    return re.sub(r"\s+", " ", (s or "").translate(QUOTE_MAP)).strip().lower()


class Layer:
    def __init__(self, path):
        L = json.load(open(path))
        self.text = L["text"]
        secs = sorted(L["sections"], key=lambda s: s["start"])
        self.secs = {f"S{i}": (s["start"], s["end"]) for i, s in enumerate(secs, 1)}
        # normalized text with a map back to raw offsets
        out, idx, prev_space = [], [], False
        for i, ch in enumerate(self.text.translate(QUOTE_MAP).lower()):
            if ch.isspace():
                if prev_space:
                    continue
                ch, prev_space = " ", True
            else:
                prev_space = False
            out.append(ch); idx.append(i)
        self.ntext, self.nidx = "".join(out), idx

    def find(self, quote, section=None, after=0):
        """Raw [start, end) of quote, preferring the hinted section; None if absent."""
        q = nrm(quote)
        if len(q) < 3:
            return None
        hits, p = [], self.ntext.find(q)
        while p != -1:
            hits.append((self.nidx[p], self.nidx[p + len(q) - 1] + 1))
            p = self.ntext.find(q, p + 1)
        hits = [h for h in hits if h[0] >= after] or hits
        if not hits:
            return None
        if section in self.secs:
            a, b = self.secs[section]
            inside = [h for h in hits if a <= h[0] < b]
            if inside:
                return inside[0]
        return hits[0]


def quotes(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("quote", "quote_end") and isinstance(v, str):
                yield v
            else:
                yield from quotes(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from quotes(v)


def span_ranges(layer, inst):
    out = []
    for sp in as_list(inst.get("spans")):
        r = layer.find(sp.get("quote", ""), sp.get("section"))
        if not r:
            continue
        end = r[1]
        if sp.get("quote_end"):
            r2 = layer.find(sp["quote_end"], sp.get("section"), after=r[0])
            if r2 and r2[1] > r[0]:
                end = r2[1]
        nl = layer.text.find("\n", end)
        out.append((sp.get("role"), r[0], max(end, nl if nl != -1 else end)))
    return out


def as_list(v):
    return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []


def code(v):
    if isinstance(v, dict):
        return v.get("code")
    return v


def fields(concept, inst):
    """Comparable key fields of an instance."""
    f = {}
    if concept != "ma.termination.outside_date":
        amts = set()
        for a in as_list(inst.get("amounts")):
            try:
                amts.add(round(float(a.get("amount")), -3) if a.get("amount") not in (None, "not_stated") else None)
            except (TypeError, ValueError):
                amts.add(str(a.get("amount")))
        f["amounts"] = frozenset(amts)
        f["triggers"] = frozenset(code(t) for t in as_list(inst.get("triggers")))
        payor = inst.get("payor") or {}
        f["payor"] = frozenset(payor.get("roles") or []) if isinstance(payor, dict) else payor
        f["remedy"] = code(inst.get("remedy"))
        f["pay_once_only"] = inst.get("pay_once_only")
        if concept == "ma.termination.fee.reverse":
            f["kind"] = code(inst.get("kind"))
    else:
        ini = inst.get("initial") or {}
        dur = ini.get("duration") if isinstance(ini, dict) else None
        f["initial"] = (ini.get("date") if isinstance(ini, dict) else None) or (
            f"{dur.get('value')} {dur.get('unit')}" if isinstance(dur, dict) else None)
        f["effect"] = code(inst.get("effect"))
        exts = as_list(inst.get("extensions"))
        f["n_extensions"] = len(exts)
        f["extension_conditions"] = frozenset(code(e.get("condition")) for e in exts)
        f["breach_bar"] = code(inst.get("breach_bar"))
    return f


def load(d, uuid):
    p = os.path.join(d, uuid + ".json")
    if not os.path.exists(p):
        return None
    data = json.load(open(p))
    return {i.get("concept"): i for i in data.get("instances", [])}


def main(D, arm, gold, dis_path=None):
    uuids = [m["uuid"] for m in json.load(open(f"{D}/manifest.json"))]
    q_ok = q_all = 0
    det = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    span = defaultdict(lambda: {"gold": 0, "gold_hit": 0, "pred": 0, "pred_hit": 0})
    span_role = defaultdict(lambda: [0, 0])
    setf1 = defaultdict(lambda: [0, 0, 0])  # tp, pred, gold for set-valued fields
    fld = defaultdict(lambda: [0, 0])
    disagreements, missing = [], []
    for u in uuids:
        G, P = load(f"{D}/out/{gold}", u), load(f"{D}/out/{arm}", u)
        if G is None or P is None:
            missing.append(u)
            continue
        layer = Layer(f"{D}/layer/{u}.json")
        for q in quotes(P):
            q_all += 1
            q_ok += layer.find(q) is not None
        for c in CONCEPTS:
            g, p = G.get(c) or {}, P.get(c) or {}
            gp, pp = g.get("finding") == "present", p.get("finding") == "present"
            det[c]["tp" if gp and pp else "fn" if gp else "fp" if pp else "tn"] += 1
            diff = {}
            if gp != pp:
                diff["finding"] = [g.get("finding"), p.get("finding")]
            if gp and pp:
                gs, ps = span_ranges(layer, g), span_ranges(layer, p)
                hit = lambda r, rs: any(r[1] < s[2] and s[1] < r[2] for s in rs)
                span[c]["gold"] += len(gs); span[c]["gold_hit"] += sum(hit(r, ps) for r in gs)
                span[c]["pred"] += len(ps); span[c]["pred_hit"] += sum(hit(r, gs) for r in ps)
                for r in gs:
                    span_role[(c, r[0])][1] += 1
                    span_role[(c, r[0])][0] += hit(r, ps)
                gf, pf = fields(c, g), fields(c, p)
                for k in gf:
                    if isinstance(gf[k], frozenset) and isinstance(pf.get(k), frozenset):
                        t = setf1[(c, k)]
                        t[0] += len(gf[k] & pf[k]); t[1] += len(pf[k]); t[2] += len(gf[k])
                    fld[(c, k)][1] += 1
                    if gf[k] == pf.get(k):
                        fld[(c, k)][0] += 1
                    else:
                        diff[k] = [sorted(map(str, gf[k])) if isinstance(gf[k], frozenset) else gf[k],
                                   sorted(map(str, pf.get(k))) if isinstance(pf.get(k), frozenset) else pf.get(k)]
            if diff:
                disagreements.append({"agreement": u, "concept": c, "diff": diff})
    rep = {"arm": arm, "gold": gold, "missing": missing,
           "quote_validity": round(q_ok / max(q_all, 1), 4), "quotes": q_all, "detection": {}, "spans": {}, "fields": {}}
    for c, d in det.items():
        rep["detection"][c] = dict(d, precision=round(d["tp"] / max(d["tp"] + d["fp"], 1), 3),
                                   recall=round(d["tp"] / max(d["tp"] + d["fn"], 1), 3))
    for c, s in span.items():
        rep["spans"][c] = dict(s, recall=round(s["gold_hit"] / max(s["gold"], 1), 3),
                               precision=round(s["pred_hit"] / max(s["pred"], 1), 3))
    for (c, k), (ok, n) in sorted(fld.items()):
        rep["fields"][f"{c.split('.')[-1]}.{k}"] = f"{ok}/{n} = {ok / max(n, 1):.3f}"
    rep["span_recall_by_role"] = {f"{c.split('.')[-1]}.{r}": f"{h}/{n}" for (c, r), (h, n) in sorted(span_role.items(), key=str)}
    rep["set_field_f1"] = {f"{c.split('.')[-1]}.{k}": round(2 * tp / max(pr + g, 1), 3) for (c, k), (tp, pr, g) in sorted(setf1.items())}
    print(json.dumps(rep, indent=1))
    if dis_path:
        json.dump(disagreements, open(dis_path, "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], a[4] if len(a) > 4 and a[3] == "--disagreements" else None)
