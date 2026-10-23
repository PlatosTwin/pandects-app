"""Build inputs with financing selected by structure, not by phrase: prepare_v6 plus the
stage 1 change from plan section 7.12.

Each holdout since M3 has missed funds representations worded in a way no earlier pattern
covered, so v7 adds whole sections by where they sit:
- the preamble and recitals (sections before the first article);
- a section of a representations article whose title or body speaks of the party having
  cash, funds, funding or capital resources ("Funding", "Capital Resources", "Obligation to
  Fund", "all cash necessary");
- any section outside a definitions section that uses the word "financing".

It also resolves and keeps definitions of cap and limit terms ("Maximum Individual Amount",
"Cap Amount", "Liability Limit"), which a cap often names without stating.

Usage: python3 prepare_v7.py holdout.tsv OUT
       python3 prepare_v7.py --rebuild D OUT   (stage 1 again from D/layer, candidates to OUT)
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
import prepare_v4 as v4
import prepare_v6 as v6
from prepare_v2 import candidate_view
from text_layer import DEFS_TITLE

REPS = re.compile(r"representations?\s+and\s+warrant", re.I)
FUNDS_TITLE = re.compile(r"\bfund(s|ing)?\b|capital resources|financ|financial (capability|ability)|sufficien", re.I)
FUNDS_BODY = re.compile(r"\b(cash|funds|funding|capital resources|financial resources|credit facilit\w*)\b[^.;]{0,120}?"
                        r"\b(sufficient|necessary|adequate|enough)\b|\b(sufficient|necessary|adequate)\b[^.;]{0,60}?"
                        r"\b(cash|funds|funding|financial resources)\b", re.I)
FINANCING = re.compile(r"\bfinancing\b", re.I)
# v4.select and v4.resolve read these module patterns at call time
CAP_TERMS = r"|maximum|\blimit|\bcap\b|cap amount"
v4.TERMS = re.compile(v4.TERMS.pattern + CAP_TERMS, re.I)
v4.RESOLVE = re.compile(v4.RESOLVE.pattern + CAP_TERMS, re.I)


def select(layer):
    groups, leaf_keep = v6.select(layer)
    by_sec = {}
    for lf in layer["leaves"]:
        by_sec.setdefault(lf["section"], []).append(lf)
    secs = sorted(layer["sections"], key=lambda s: s["start"])
    first_article = next((n for n, s in enumerate(secs) if (s["article"] or "").strip()), 0)
    for n, s in enumerate(secs):
        if s["uuid"] in groups:
            continue
        body = "".join(layer["text"][l["start"]:l["end"]] for l in by_sec.get(s["uuid"], []))
        article, title = s["article"] or "", s["title"] or ""
        if DEFS_TITLE.search(f"{title} {article}") or s["uuid"].startswith("exhibit:"):
            continue
        if (n < first_article
                or REPS.search(article) and (FUNDS_TITLE.search(title) or FUNDS_BODY.search(body))
                or FINANCING.search(body)):
            groups[s["uuid"]] = {"deal_risk", "structure"}
    return groups, leaf_keep


def rebuild(D, out):
    os.makedirs(f"{out}/candidates", exist_ok=True)
    for m in json.load(open(f"{D}/manifest.json")):
        layer = json.load(open(f"{D}/layer/{m['uuid']}.json"))
        groups, leaf_keep = select(layer)
        open(f"{out}/candidates/{m['uuid']}.md", "w").write(candidate_view(layer, groups, leaf_keep))


def main(dump, out):
    v4.select = select  # v4.main builds the views with whichever select it is given
    v4.main(dump, out)


if __name__ == "__main__":
    if sys.argv[1] == "--rebuild":
        rebuild(sys.argv[2], sys.argv[3])
    else:
        main(*sys.argv[1:3])
