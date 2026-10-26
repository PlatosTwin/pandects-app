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

Definition resolution also reads the colon forms '"General Indemnity Holdback Amount:" $37,700,000'
and '"Special Indemnity Holdback Amount": means $18,850,000' (plan section 7.14), which hid the
cap figures in one M6 agreement.

From the M7 holdout (plan section 7.17):
- date and representation terms the outside date and survival clauses name ("Drop Dead Date",
  "Longstop Date", "Excepted Representations") are kept and resolved, and "MaterialAdverse
  Effect" printed without its space still counts as an MAE term;
- a kept definition that runs on into the next paragraphs, across a page number or because the
  text layer split it, brings those paragraphs, up to the next quoted term.

From the M7 to M9 extension misses (plan section 7.24): a sentence that extends the outside date
often names its condition only by section number ("other than the condition set forth in
Section 7.1(b)"), so the sections such a sentence cites are kept whole.

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
CAP_TERMS = r"|maximum|\blimit|\bcap\b|cap amount" + r"|drop dead|long ?stop|cut-?off date|excepted rep|specified rep|material\s*adverse"
v4.TERMS = re.compile(v4.TERMS.pattern + CAP_TERMS, re.I)
v4.RESOLVE = re.compile(v4.RESOLVE.pattern + CAP_TERMS, re.I)
v4.DEFINED = re.compile(r"[\"“]([^\"”:]{2,80})(?::[\"”]|[\"”]\s*:\s|[\"”](\s+of any \w+)?\s*(shall\s+)?"
                        r"(means?|has the meaning|shall have the meaning|is defined|(shall|will) be deemed))")


NEW_DEF = re.compile(r"^\s*(?:(?:A|An|The)\s+)?[\"“]")
OD_TERM = r"(?:Outside|End|Termination|Expiration|Drop Dead|Long ?stop|Cut-?off)\s+Date"
OD_EXTEND = re.compile(OD_TERM + r"[^.]{0,250}?\bextend|\bextend\w*[^.]{0,120}?" + OD_TERM, re.I)
CITED = re.compile(r"\bSections?\s+((?:\d+\.\d+(?:\([A-Za-z0-9]{1,4}\))*(?:\s*(?:,|and/or|or|and)\s*(?:Section\s+)?)?)+)")
SEC_NUMBER = re.compile(r"^\s*(?:Section\s+|SECTION\s+)?(\d+\.\d+)\b")


def outside_date_citations(layer):
    """Sections cited by any sentence that extends the outside date."""
    numbers = {}
    for s in layer["sections"]:
        m = SEC_NUMBER.match(s["title"] or "")
        if m:
            numbers.setdefault(m.group(1), s["uuid"])
    cited = set()
    for lf in layer["leaves"]:
        t = layer["text"][lf["start"]:lf["end"]]
        for sent in re.split(r"(?<=[.;])\s+", t):
            if OD_EXTEND.search(sent):
                cited |= {numbers[n] for m in CITED.finditer(sent) for n in re.findall(r"\d+\.\d+", m.group(1)) if n in numbers}
    return cited


def extend_definitions(layer, leaf_keep, limit=6):
    """Keep the paragraphs a kept definition runs on into, up to the next quoted term."""
    by_sec = {}
    for lf in layer["leaves"]:
        by_sec.setdefault(lf["section"], []).append(lf)
    for sec, keep in leaf_keep.items():
        leaves, add = by_sec.get(sec, []), set()
        for i, lf in enumerate(leaves):
            if id(lf) not in keep or lf["kind"] != "definition":
                continue
            for x in leaves[i + 1:i + 1 + limit]:
                if id(x) in keep or NEW_DEF.match(layer["text"][x["start"]:x["end"]]):
                    break
                add.add(id(x))
        keep |= add


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
    for uuid in outside_date_citations(layer):
        sec = next(s for s in secs if s["uuid"] == uuid)
        if uuid not in groups and not DEFS_TITLE.search(f"{sec['title'] or ''} {sec['article'] or ''}"):
            groups[uuid] = {"outside_date"}
    extend_definitions(layer, leaf_keep)
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
