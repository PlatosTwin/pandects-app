"""Build M2 holdout inputs: prepare_v2 with the stage 1 fixes from plan section 7.5.

Changes from prepare_v2:
- the text layer admits defined-terms exhibits and splits section-less body text
  (text_layer.project(exhibits=True));
- deal_risk also finds funds and financial-capability reps, and "Financing" glued to
  a section number ("4.05Financing");
- indemnification also finds limitation, cap, threshold and maximum-liability sections;
- a section whose title is only a number is matched on its first line of text too;
- only sections titled as definitions (or defined-terms exhibits) are trimmed to
  relevant terms; other sections with many inline definitions are kept whole.

Usage: python3 prepare_v3.py holdout.tsv OUT
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from text_layer import DEFS_TITLE, decode, project, read_dump
from prepare import view
from prepare_v2 import TERMS, candidate_view

GROUPS = {
    "termination": (re.compile(r"terminat|fee|expense|outside|end date|remed|specific performance|enforcement", re.I),
                    re.compile(r"termination fee|break-?up fee|termination payment|reverse termination|outside date|"
                               r"end date|termination date|drop[- ]dead|expense reimbursement|reimburse\w* .{0,40}expenses", re.I)),
    "deal_risk": (re.compile(r"material adverse|financing\b|debt commitment|marketing period|conditions to|"
                             r"sufficien\w* (of )?funds|availab\w* (of )?funds|financial (ability|capability|resources)|"
                             r"\bfunds\b", re.I),
                  re.compile(r"material adverse (effect|change)[\"”]? (means|shall mean|shall have)|[\"“](\w+ )?material adverse (effect|change)[\"”]\)|debt financing|commitment letter|"
                             r"marketing period|limited guarant|equity commitment|financing (is not|shall not be) a condition|"
                             r"(sufficient|immediately available) (cash|funds)|funds sufficient|financial (ability|capability) to", re.I)),
    "indemnification": (re.compile(r"(?<!officers' )(?<!director )indemnif(?!.{0,40}(director|officer|D&O))|surviv|basket|deductible|"
                                   r"limitations? (on|of)|limits on|maximum (liability|amount)|minimum (claims?|amount)|"
                                   r"\bcap\b|threshold|liabilit|escrow", re.I),
                        re.compile(r"(shall|will|do|does)( not)? survive|deductible|basket|de minimis|indemnification (cap|threshold)|"
                                   r"representations and warranties insurance|R&W (insurance|policy)|"
                                   r"(aggregate|maximum) (liability|amount) of|shall not be liable .{0,80}(exceed|unless)", re.I)),
    "tax": (re.compile(r"338|tax election|allocation", re.I),
            re.compile(r"338\s*\(h\)|336\s*\(e\)|338\s*\(g\)|section 338|purchase price allocation|allocation schedule", re.I)),
}


def select(layer):
    by_sec = {}
    for lf in layer["leaves"]:
        by_sec.setdefault(lf["section"], []).append(lf)
    groups, leaf_keep = {}, {}
    for s in layer["sections"]:
        leaves = by_sec.get(s["uuid"], [])
        body = "".join(layer["text"][l["start"]:l["end"]] for l in leaves)
        title = s["title"] or ""
        if not re.search(r"[A-Za-z]{3}", title) and leaves:
            title += " " + layer["text"][leaves[0]["start"]:leaves[0]["end"]][:120]
        g = {name for name, (t, x) in GROUPS.items() if t.search(title) or x.search(body)}
        defs = [l for l in leaves if l["kind"] == "definition"]
        is_defs = DEFS_TITLE.search(f"{s['title'] or ''} {s['article'] or ''}") or s["uuid"].startswith("exhibit:")
        if is_defs and len(defs) >= 5:
            # a definitions section: keep it, trimmed to relevant terms
            keep = {id(leaves[0])} if leaves and leaves[0]["kind"] != "definition" else set()
            keep |= {id(l) for l in defs if TERMS.search(l["term"] or "")}
            if keep:
                groups[s["uuid"]] = g | {"definitions"}
                leaf_keep[s["uuid"]] = keep
        elif g:
            groups[s["uuid"]] = g
    return groups, leaf_keep


def main(dump, out):
    for d in ("layer", "read", "candidates"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    manifest = []
    for row in read_dump(dump):
        uuid, ttype, era, deal_type, target, acquirer, dt, price = row[:8]
        layer = project(decode(row[-1]), exhibits=True)
        groups, leaf_keep = select(layer)
        json.dump(layer, open(f"{out}/layer/{uuid}.json", "w"))
        full, cand = view(layer), candidate_view(layer, groups, leaf_keep)
        open(f"{out}/read/{uuid}.md", "w").write(full)
        open(f"{out}/candidates/{uuid}.md", "w").write(cand)
        manifest.append({"uuid": uuid, "target_type": ttype, "era": era, "deal_type": deal_type,
                         "target": target, "acquirer": acquirer, "date": dt, "price_total": price,
                         "layer_hash": layer["layer_hash"], "sections": len(layer["sections"]),
                         "exhibit_sections": sum(s["uuid"].startswith("exhibit:") for s in layer["sections"]),
                         "candidate_sections": len(groups), "chars": len(full), "candidate_chars": len(cand),
                         "split": "holdout"})
    manifest.sort(key=lambda m: (m["target_type"], m["era"], m["uuid"]))
    json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
