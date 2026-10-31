"""Build M2 inputs (ten concepts) from a dump of sampled agreements.

Like prepare.py, but stage 1 retrieval covers four concept groups, and long definitions
sections are trimmed to the definitions whose term is relevant to some concept. Each
candidate marker line names the groups that selected it:

  @@ S<n> | article | title | groups=termination,financing

Writes OUT/layer, OUT/read, OUT/candidates, OUT/manifest.json (with split dev/test).
Usage: python3 prepare_v2.py m2.tsv OUT
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from text_layer import decode, project, read_dump
from prepare import view

GROUPS = {
    "termination": (re.compile(r"terminat|fee|expense|outside|end date|remed|specific performance|enforcement", re.I),
                    re.compile(r"termination fee|break-?up fee|termination payment|reverse termination|outside date|"
                               r"end date|termination date|drop[- ]dead|expense reimbursement|reimburse\w* .{0,40}expenses", re.I)),
    "deal_risk": (re.compile(r"material adverse|\bfinancing\b|debt commitment|marketing period|conditions to", re.I),
                  re.compile(r"material adverse (effect|change)[\"\u201d]? (means|shall mean|shall have)|[\"\u201c](\w+ )?material adverse (effect|change)[\"\u201d]\)|debt financing|commitment letter|"
                             r"marketing period|limited guarant|equity commitment|financing (is not|shall not be) a condition", re.I)),
    "indemnification": (re.compile(r"(?<!officers' )(?<!director )indemnif(?!.{0,40}(director|officer|D&O))|surviv|basket|deductible|limitations? on|escrow", re.I),
                        re.compile(r"(shall|will|do|does)( not)? survive|deductible|basket|de minimis|indemnification (cap|threshold)|"
                                   r"representations and warranties insurance|R&W (insurance|policy)", re.I)),
    "tax": (re.compile(r"338|tax election|allocation", re.I),
            re.compile(r"338\s*\(h\)|336\s*\(e\)|338\s*\(g\)|section 338|purchase price allocation|allocation schedule", re.I)),
}
TERMS = re.compile(r"fee|outside|end date|termination date|expense|material adverse|acquisition proposal|"
                   r"superior proposal|takeover proposal|financ|debt|commitment|marketing period|guarant|loss|"
                   r"deductible|basket|cap\b|threshold|escrow|fundamental|surviv|tax|338|purchase price|"
                   r"merger consideration|willful|intentional breach|consideration", re.I)


def select(layer):
    by_sec = {}
    for lf in layer["leaves"]:
        by_sec.setdefault(lf["section"], []).append(lf)
    groups, leaf_keep = {}, {}
    for s in layer["sections"]:
        leaves = by_sec.get(s["uuid"], [])
        body = "".join(layer["text"][l["start"]:l["end"]] for l in leaves)
        g = {name for name, (title, text) in GROUPS.items()
             if title.search(s["title"] or "") or text.search(body)}
        defs = [l for l in leaves if l["kind"] == "definition"]
        if len(defs) >= 5:
            # a definitions section: keep it, trimmed to relevant terms
            keep = {id(l) for l in leaves if l["kind"] != "definition" and leaves.index(l) == 0}
            keep |= {id(l) for l in defs if TERMS.search(l["term"] or "")}
            if len(keep) > 1:
                groups[s["uuid"]] = g | {"definitions"}
                leaf_keep[s["uuid"]] = keep
        elif g:
            groups[s["uuid"]] = g
    return groups, leaf_keep


def candidate_view(layer, groups, leaf_keep):
    out, secs = [], sorted(layer["sections"], key=lambda s: s["start"])
    by_sec = {}
    for lf in layer["leaves"]:
        by_sec.setdefault(lf["section"], []).append(lf)
    for n, s in enumerate(secs, 1):
        if s["uuid"] not in groups:
            continue
        out.append(f"@@ S{n} | {s['article'] or ''} | {s['title'] or ''} | groups={','.join(sorted(groups[s['uuid']]))}")
        keep = leaf_keep.get(s["uuid"])
        for lf in by_sec.get(s["uuid"], []):
            if keep is None or id(lf) in keep:
                out.append(layer["text"][lf["start"]:lf["end"]])
    return "\n".join(out) + "\n"


def main(dump, out):
    for d in ("layer", "read", "candidates"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    manifest = []
    for row in read_dump(dump):
        uuid, ttype, era, deal_type, target, acquirer, dt, price = row[:8]
        layer = project(decode(row[-1]))
        groups, leaf_keep = select(layer)
        json.dump(layer, open(f"{out}/layer/{uuid}.json", "w"))
        full, cand = view(layer), candidate_view(layer, groups, leaf_keep)
        open(f"{out}/read/{uuid}.md", "w").write(full)
        open(f"{out}/candidates/{uuid}.md", "w").write(cand)
        manifest.append({"uuid": uuid, "target_type": ttype, "era": era, "deal_type": deal_type,
                         "target": target, "acquirer": acquirer, "date": dt, "price_total": price,
                         "layer_hash": layer["layer_hash"], "sections": len(layer["sections"]),
                         "candidate_sections": len(groups), "chars": len(full), "candidate_chars": len(cand)})
    manifest.sort(key=lambda m: (m["target_type"], m["era"], m["uuid"]))
    for i, m in enumerate(manifest):
        m["split"] = "dev" if i % 3 == 0 else "test"
    json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)


def rebuild_candidates(out, split=None):
    """Re-run stage 1 from saved layers (after a change to the patterns)."""
    manifest = json.load(open(f"{out}/manifest.json"))
    for m in manifest:
        if split and m["split"] != split:
            continue
        layer = json.load(open(f"{out}/layer/{m['uuid']}.json"))
        groups, leaf_keep = select(layer)
        cand = candidate_view(layer, groups, leaf_keep)
        open(f"{out}/candidates/{m['uuid']}.md", "w").write(cand)
        m["candidate_sections"], m["candidate_chars"] = len(groups), len(cand)
    json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "--rebuild":
        rebuild_candidates(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        main(*sys.argv[1:3])
