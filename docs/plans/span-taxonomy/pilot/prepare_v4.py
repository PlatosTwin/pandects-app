"""Build definition-resolution inputs: prepare_v3 plus the stage 1 fixes from plan section 7.6.

Changes from prepare_v3:
- the text layer also admits defined-terms exhibits stored one page per <text>
  (text_layer.project(page_blobs=True)), and a definitions-section line that defines
  several terms is trimmed on any of them;
- definition resolution: when a candidate section uses a relevant defined term
  (Adverse Recommendation Change, Triggering Event, Indemnity Escrow Amount, Enterprise
  Value, ...) whose definition is not already in the candidates, the defining paragraph
  is added. Definitions come from definition leaves and from inline definitions
  ('... (an "Adverse Recommendation Change")', '"X" means ...'). One pass only: terms used
  inside an added definition are not resolved in turn.

Usage: python3 prepare_v4.py holdout.tsv OUT
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from text_layer import DEFS_TITLE, decode, project, read_dump
from prepare import view
from prepare_v2 import TERMS, candidate_view
from prepare_v3 import GROUPS

# terms worth resolving: the trimming list plus the terms fee triggers and caps key off
RESOLVE = re.compile(TERMS.pattern + r"|recommendation|triggering|acquisition|alternative|competing|"
                     r"holdback|retention|enterprise value|regulatory|antitrust|indemni", re.I)
INLINE = re.compile(r"[\"“]([A-Z][^\"”]{1,60})[\"”]\s*\)")
# '"X" means ...', and event definitions: '"Triggering Event" shall be deemed to have occurred if'
DEFINED = re.compile(r"[\"“]([^\"”]{2,80})[\"”](\s+of any \w+)?\s*(shall\s+)?"
                     r"(means?|has the meaning|shall have the meaning|is defined|(shall|will) be deemed)")


def leaf_terms(lf):
    return lf.get("terms") or ([lf["term"]] if lf["term"] else [])


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
            keep |= {id(l) for l in defs if any(TERMS.search(t) for t in leaf_terms(l))}
            if keep:
                groups[s["uuid"]] = g | {"definitions"}
                leaf_keep[s["uuid"]] = keep
        elif g:
            groups[s["uuid"]] = g
    resolve(layer, by_sec, groups, leaf_keep)
    return groups, leaf_keep


def resolve(layer, by_sec, groups, leaf_keep):
    """Add the defining leaves of relevant terms that candidate sections use."""
    text = layer["text"]
    index = {}  # term -> leaves that define it
    for lf in layer["leaves"]:
        found = set(leaf_terms(lf)) if lf["kind"] == "definition" else set()
        t = text[lf["start"]:lf["end"]]
        found |= {m.group(1) for m in INLINE.finditer(t)} | {m.group(1) for m in DEFINED.finditer(t)}
        for term in found:
            if RESOLVE.search(term):
                index.setdefault(term.strip(), []).append(lf)
    kept = lambda lf: lf["section"] in groups and (leaf_keep.get(lf["section"]) is None or id(lf) in leaf_keep[lf["section"]])
    used = set()
    for s in layer["sections"]:
        if groups.get(s["uuid"], {"definitions"}) <= {"definitions"}:
            continue  # resolve from operative candidate sections only
        body = "".join(text[l["start"]:l["end"]] for l in by_sec.get(s["uuid"], []) if kept(l))
        used |= {term for term in index if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", body)}
    for term in used:
        for lf in index[term]:
            if kept(lf):
                continue
            sec = lf["section"]
            leaf_keep.setdefault(sec, set()).add(id(lf))
            groups[sec] = groups.get(sec, set()) | {"resolved"}


def main(dump, out):
    for d in ("layer", "read", "candidates"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    manifest = []
    for row in read_dump(dump):
        uuid, ttype, era, deal_type, target, acquirer, dt, price = row[:8]
        layer = project(decode(row[-1]), exhibits=True, page_blobs=True)
        groups, leaf_keep = select(layer)
        json.dump(layer, open(f"{out}/layer/{uuid}.json", "w"))
        full, cand = view(layer), candidate_view(layer, groups, leaf_keep)
        open(f"{out}/read/{uuid}.md", "w").write(full)
        open(f"{out}/candidates/{uuid}.md", "w").write(cand)
        manifest.append({"uuid": uuid, "target_type": ttype, "era": era, "deal_type": deal_type,
                         "target": target, "acquirer": acquirer, "date": dt, "price_total": price,
                         "layer_hash": layer["layer_hash"], "sections": len(layer["sections"]),
                         "exhibit_sections": sum(s["uuid"].startswith("exhibit:") for s in layer["sections"]),
                         "candidate_sections": len(groups),
                         "resolved_sections": sum("resolved" in g for g in groups.values()),
                         "chars": len(full), "candidate_chars": len(cand), "split": "holdout"})
    manifest.sort(key=lambda m: (m["target_type"], m["era"], m["uuid"]))
    json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
