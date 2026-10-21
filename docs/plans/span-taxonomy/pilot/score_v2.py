"""Score M2 extractions (schema_v2.md, ten concepts) against gold.

Usage: python3 score_v2.py D ARM GOLD [--split dev|test] [--disagreements OUT.json]
Reuses score.py's text matching and span anchoring; only the concept list and the
comparable key fields differ.
"""
import json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from score import Layer, as_list, code, load, quotes, span_ranges

CONCEPTS = ["ma.termination.fee.target", "ma.termination.fee.reverse", "ma.termination.expense_reimbursement",
            "ma.termination.outside_date", "ma.mae.definition", "ma.financing.structure", "ma.indemn.cap",
            "ma.indemn.basket", "ma.indemn.survival", "tax.election.338h10"]


def money(m):
    if not isinstance(m, dict):
        return m if isinstance(m, str) else None
    a = m.get("amount")
    if isinstance(a, (int, float)):
        return round(float(a), -3)
    if m.get("pct") is not None:
        return f"{m.get('basis')}:{m.get('pct')}"
    return a


def sides(parties):
    return frozenset(obj(p).get("side") for p in as_list(parties))


def obj(x):
    """Labelers write "not_stated" where a schema object is silent; treat that as empty."""
    return x if isinstance(x, dict) else {}


def val(v):
    return v.get("value") if isinstance(v, dict) else v


def period(p):
    p = obj(obj(p).get("period"))
    d = obj(p.get("duration"))
    return (p.get("kind"), d.get("value"), d.get("unit"), p.get("offset_days"), p.get("date"))


def fields(c, inst):
    f = {}
    if c in ("ma.termination.fee.target", "ma.termination.fee.reverse"):
        f["amounts"] = frozenset(money(a) for a in as_list(inst.get("amounts")))
        f["triggers"] = frozenset(code(t) for t in as_list(inst.get("triggers")))
        f["payor_side"] = sides(inst.get("payor"))
        f["remedy"] = code(inst.get("remedy"))
        if c.endswith("reverse"):
            f["kind"] = code(inst.get("kind"))
    elif c == "ma.termination.expense_reimbursement":
        f["cap"] = money(inst.get("cap"))
        f["payor_side"] = sides(inst.get("payor"))
    elif c == "ma.termination.outside_date":
        ini = obj(inst.get("initial"))
        dur = ini.get("duration")
        f["initial"] = ini.get("date") or (f"{dur.get('value')} {dur.get('unit')}" if isinstance(dur, dict) else None)
        f["effect"] = code(inst.get("effect"))
        exts = as_list(inst.get("extensions"))
        f["n_extensions"] = len(exts)
        f["extension_conditions"] = frozenset(code(obj(e).get("condition")) for e in exts)
        bb = inst.get("breach_bar")  # one object in v2/v2.1, a list from v2.2
        f["breach_causation"] = (frozenset(obj(b).get("causation") for b in bb) if isinstance(bb, list)
                                 else frozenset([bb.get("causation")]) if isinstance(bb, dict) else bb)
    elif c == "ma.mae.definition":
        defs = [obj(d) for d in as_list(inst.get("definitions"))]
        tgt = next((d for d in defs if d.get("party_side") in ("target_side", "both")), defs[0] if defs else {})
        f["carveout_categories"] = frozenset(k for co in as_list(tgt.get("carveouts")) for k in (obj(co).get("categories") if isinstance(obj(co).get("categories"), list) else []))
        pr = obj(tgt.get("prongs"))
        f["prospects"] = pr.get("prospects")
        f["ability_to_consummate"] = pr.get("ability_to_consummate")
        f["n_definitions"] = len(defs)
        de = tgt.get("disproportionate_effect_exception")
        f["disproportionate_exception"] = isinstance(de, dict)
    elif c == "ma.financing.structure":
        f["financing_condition"] = val(inst.get("financing_condition"))
        f["committed"] = frozenset(code(x) for x in as_list(inst.get("committed_financing")))
        f["efforts"] = code(inst.get("efforts_standard"))
        f["marketing_period"] = isinstance(inst.get("marketing_period"), dict)
        f["cooperation"] = val(inst.get("financing_cooperation"))
    elif c == "ma.indemn.cap":
        f["cap_amounts"] = frozenset(money(obj(x).get("amount")) for x in as_list(inst.get("caps")))
        f["escrow_sole_recourse"] = val(inst.get("escrow_sole_recourse"))
    elif c == "ma.indemn.basket":
        f["baskets"] = frozenset((obj(b).get("type"), money(obj(b).get("amount"))) for b in as_list(inst.get("baskets")))
        dm = inst.get("de_minimis")
        f["de_minimis"] = money(dm.get("amount")) if isinstance(dm, dict) else dm
    elif c == "ma.indemn.survival":
        f["no_survival"] = val(inst.get("no_survival"))
        ps = {obj(p).get("item"): period(p) for p in as_list(inst.get("periods"))}
        f["general_reps"] = ps.get("general_reps") or ps.get("all_reps")
        f["fundamental_reps"] = ps.get("fundamental_reps")
    elif c == "tax.election.338h10":
        f["election"] = code(inst.get("election"))
        f["decides"] = code(inst.get("decides"))
        tcs = inst.get("tax_cost_shift")
        f["tax_cost_payer"] = sides(tcs.get("payer")) if isinstance(tcs, dict) else tcs
    return f


def show(v):
    return sorted(map(str, v)) if isinstance(v, frozenset) else v


def main(D, arm, gold, split=None, dis_path=None):
    """split=None scores every agreement in the manifest."""
    uuids = [m["uuid"] for m in json.load(open(f"{D}/manifest.json")) if split in (None, m["split"])]
    q_ok = q_all = 0
    det = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    span = defaultdict(lambda: [0, 0, 0, 0])  # gold, gold_hit, pred, pred_hit
    fld = defaultdict(lambda: [0, 0])
    setf1 = defaultdict(lambda: [0, 0, 0])
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
                s = span[c]
                s[0] += len(gs); s[1] += sum(hit(r, ps) for r in gs)
                s[2] += len(ps); s[3] += sum(hit(r, gs) for r in ps)
                gf, pf = fields(c, g), fields(c, p)
                for k in gf:
                    if isinstance(gf[k], frozenset) and isinstance(pf.get(k), frozenset):
                        t = setf1[(c, k)]
                        t[0] += len(gf[k] & pf[k]); t[1] += len(pf[k]); t[2] += len(gf[k])
                    fld[(c, k)][1] += 1
                    if gf[k] == pf.get(k):
                        fld[(c, k)][0] += 1
                    else:
                        diff[k] = [show(gf[k]), show(pf.get(k))]
            if diff:
                disagreements.append({"agreement": u, "concept": c, "diff": diff})
    short = lambda c: c.split(".", 1)[1]
    rep = {"arm": arm, "gold": gold, "split": split, "agreements": len(uuids) - len(missing), "missing": missing,
           "quote_validity": round(q_ok / max(q_all, 1), 4), "quotes": q_all,
           "detection": {short(c): dict(d, precision=round(d["tp"] / max(d["tp"] + d["fp"], 1), 3),
                                        recall=round(d["tp"] / max(d["tp"] + d["fn"], 1), 3)) for c, d in det.items()},
           "spans": {short(c): {"recall": round(s[1] / max(s[0], 1), 3), "precision": round(s[3] / max(s[2], 1), 3),
                                "gold": s[0], "pred": s[2]} for c, s in span.items()},
           "fields": {f"{short(c)}.{k}": [ok, n] for (c, k), (ok, n) in sorted(fld.items())},
           "set_field_f1": {f"{short(c)}.{k}": round(2 * tp / max(pr + g, 1), 3) for (c, k), (tp, pr, g) in sorted(setf1.items())}}
    if dis_path:
        json.dump(disagreements, open(dis_path, "w"), indent=1)
    return rep


if __name__ == "__main__":
    a = sys.argv[1:]
    opts = dict(zip(a[3::2], a[4::2]))
    print(json.dumps(main(a[0], a[1], a[2], opts.get("--split"), opts.get("--disagreements")), indent=1))
