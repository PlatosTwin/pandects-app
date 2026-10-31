"""Build the M1 pilot inputs from a dump of the 50 sampled agreements.

For each agreement writes, under OUT/:
  layer/<uuid>.json       the canonical text layer (text_layer.project)
  read/<uuid>.md          the whole body as a reading view, for gold labelers
  candidates/<uuid>.md    stage 1 output: candidate sections only, for extractors
  manifest.json           sample metadata and candidate sizes

In the reading views each section starts with a marker line "@@ S<n> | article | title".
Marker lines are not part of the layer; every other line is one leaf, verbatim.

Usage: python3 prepare.py m1.tsv OUT
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from text_layer import decode, project, read_dump

TITLE = re.compile(r"terminat|fee|expense|outside|end date|remed|specific performance|enforcement", re.I)
TEXT = re.compile(r"termination fee|break-?up fee|termination payment|termination amount|"
                  r"reverse termination|outside date|end date|termination date|drop[- ]dead|"
                  r"expense reimbursement|reimburse\w* .{0,40}expenses", re.I)
TERM = re.compile(r"fee|outside|end date|termination date|terminat|expense", re.I)


def view(layer, keep=None):
    out, secs = [], sorted(layer["sections"], key=lambda s: s["start"])
    by_sec = {}
    for lf in layer["leaves"]:
        by_sec.setdefault(lf["section"], []).append(lf)
    for n, s in enumerate(secs, 1):
        if keep is not None and s["uuid"] not in keep:
            continue
        out.append(f"@@ S{n} | {s['article'] or ''} | {s['title'] or ''}")
        for lf in by_sec.get(s["uuid"], []):
            out.append(layer["text"][lf["start"]:lf["end"]])
    return "\n".join(out) + "\n"


def candidates(layer):
    keep = set()
    by_sec = {}
    for lf in layer["leaves"]:
        by_sec.setdefault(lf["section"], []).append(lf)
    for s in layer["sections"]:
        body = "".join(layer["text"][l["start"]:l["end"]] for l in by_sec.get(s["uuid"], []))
        terms = " ".join(l["term"] or "" for l in by_sec.get(s["uuid"], []) if l["kind"] == "definition")
        if (TITLE.search(s["title"] or "") or TITLE.search(s["article"] or "") and re.search(r"terminat", s["article"] or "", re.I)
                or TEXT.search(body)):
            keep.add(s["uuid"])
        elif terms and TERM.search(terms):
            keep.add(s["uuid"])
    return keep


def main(dump, out):
    for d in ("layer", "read", "candidates"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    manifest = []
    for row in read_dump(dump):
        uuid, ttype, era, deal_type, target, acquirer, dt, price = row[:8]
        layer = project(decode(row[-1]))
        keep = candidates(layer)
        json.dump(layer, open(f"{out}/layer/{uuid}.json", "w"))
        full, cand = view(layer), view(layer, keep)
        open(f"{out}/read/{uuid}.md", "w").write(full)
        open(f"{out}/candidates/{uuid}.md", "w").write(cand)
        manifest.append({"uuid": uuid, "target_type": ttype, "era": era, "deal_type": deal_type,
                         "target": target, "acquirer": acquirer, "date": dt, "price_total": price,
                         "layer_hash": layer["layer_hash"], "sections": len(layer["sections"]),
                         "candidate_sections": len(keep), "chars": len(full), "candidate_chars": len(cand)})
    json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
