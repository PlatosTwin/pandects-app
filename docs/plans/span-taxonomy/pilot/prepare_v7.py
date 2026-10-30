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

From the M11 MAE misses (plan section 7.28): an undefined lowercase "material adverse effect" on the
buyer sits in a buyer representation or a closing condition, so those sections are kept when they
use it, as is a section that defines the MAE inline with a qualifier ('a "Material Adverse Effect"
with respect to any Person means').

From the M12 MAE misses (plan section 7.30): an MAE defined inline in an interpretation section is kept,
"Material Impact" counts as an MAE term, and the definitions of terms an MAE definition names (its
carve-outs as an "Excluded Matter") are kept with it.

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
CAP_TERMS = r"|maximum|\blimit|\bcap\b|cap amount" + r"|drop dead|long ?stop|cut-?off date|excepted rep|specified rep|material\s*adverse|material impact"
# 7.32: fee amounts defined under other names, and an MAE abbreviated ("Snacks Business MAE", M13 967bef44)
CAP_TERMS += r"|termination (?:compensation|payment)|\bdeposit\b|\bMAE\b"
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


BUYER = re.compile(r"\b(?:Parent|Purchasers?|Buyers?|Acquir(?:er|or|ing)|Merger Sub\w*|Acquisition (?:Sub|Corp)\w*)\b", re.I)
CONDITIONS = re.compile(r"\bconditions?\b", re.I)
LOWER_MAE = re.compile(r"\bmaterial\s+adverse\s+(?:effect|change|impact)\b")  # lowercase: undefined, not the defined term
# an MAE defined inline with a qualifier before "means": 'a "Material Adverse Effect" with respect to any Person means'
MAE_TERM = r"Material\s*(?:Adverse\s+(?:Effect|Change)|Impact)|\bMAE\b"  # "Material Impact": a target MAE without the usual words (M12 daef34e9)
# '"X Material Adverse Effect" ... means', or 'will be deemed to have a "Material Adverse Effect" on' (M13 a621ea6f)
INLINE_MAE = re.compile(r"[\"“](?:\w+\s+){0,3}(?:" + MAE_TERM + r")[\"”][^.;]{1,60}?\b(?:means|shall mean)\b"
                        r"|deemed to (?:have|be) (?:had )?(?:a |an )?[\"“](?:" + MAE_TERM + r")[\"”]")


def mae_sections(layer):
    """Sections stage 1 missed MAE text in on M11 (plan section 7.28): a section of the buyer's representations, or a
    closing-conditions section, that uses "material adverse effect" in lowercase (an undefined buyer-side MAE,
    schema v2.12 rule 3), and a section outside the definitions that defines the MAE inline with a qualifier."""
    by_sec = {}
    for lf in layer["leaves"]:
        by_sec.setdefault(lf["section"], []).append(lf)
    out = set()
    for s in layer["sections"]:
        article, title = s["article"] or "", s["title"] or ""
        if DEFS_TITLE.search(f"{title} {article}") or s["uuid"].startswith("exhibit:"):
            continue
        body = [layer["text"][l["start"]:l["end"]] for l in by_sec.get(s["uuid"], [])]
        if any(INLINE_MAE.search(x) for x in body):
            out.add(s["uuid"])
            continue
        buyer_reps = REPS.search(article) and BUYER.search(article) or REPS.search(title) and BUYER.search(title)
        if (buyer_reps or CONDITIONS.search(title) or CONDITIONS.search(article)) and any(LOWER_MAE.search(x) for x in body):
            out.add(s["uuid"])
    return out


def mae_definition_leaves(layer, groups, leaf_keep, limit=6):
    """MAE text stage 1 missed on M12 (plan section 7.30): an MAE defined inline in an interpretation
    or definitions section that the definition patterns do not read ('any reference to a "Material
    Adverse Effect" with respect to any entity ... means', 775de59e), and the definitions of terms an
    MAE definition names, such as its carve-outs ("Excluded Matter", 41b903c1)."""
    text, by_sec = layer["text"], {}
    for lf in layer["leaves"]:
        by_sec.setdefault(lf["section"], []).append(lf)
    kept = lambda lf: lf["section"] in groups and (leaf_keep.get(lf["section"]) is None or id(lf) in leaf_keep[lf["section"]])
    keep = lambda lf: (leaf_keep.setdefault(lf["section"], set()).add(id(lf)),
                       groups.__setitem__(lf["section"], groups.get(lf["section"], set()) | {"resolved"}))
    body = []  # the text of each MAE definition, with the paragraphs it runs on into
    for sec, leaves in by_sec.items():
        for i, lf in enumerate(leaves):
            t = text[lf["start"]:lf["end"]]
            if not (INLINE_MAE.search(t) or any(re.search(MAE_TERM, x, re.I) for x in v4.leaf_terms(lf))):
                continue
            if not kept(lf):
                keep(lf)
            body.append(t)
            for x in leaves[i + 1:i + 1 + limit]:
                xt = text[x["start"]:x["end"]]
                if NEW_DEF.match(xt):
                    break
                if not kept(x):
                    keep(x)
                body.append(xt)
    body = "\n".join(body)
    for lf in layer["leaves"]:
        if lf["kind"] != "definition" or kept(lf):
            continue
        terms = [x for x in v4.leaf_terms(lf) if x and not re.search(MAE_TERM, x, re.I)]  # stored lowercase
        if any(re.search(r"(?<!\w)" + re.escape(x) + r"(?!\w)", body, re.I) for x in terms):
            keep(lf)


def undefined_mae_leaves(layer, groups, leaf_keep):
    """With no MAE defined anywhere, the paragraphs of any party's representations that use the
    undefined lowercase one (plan section 7.32, M13 2c3e8ffb), not their whole sections."""
    # a capitalized MAE term is a defined one, however it is defined (7989f883), but not a heading
    # such as "(d) No Material Adverse Change." (2c3e8ffb)
    if re.search(MAE_TERM, re.sub(r"\bNo (?:" + MAE_TERM + r")\s*\.", "", layer["text"])):
        return
    secs = {s["uuid"]: s for s in layer["sections"]}
    for lf in layer["leaves"]:
        s = secs[lf["section"]]
        if not (REPS.search(s["article"] or "") or REPS.search(s["title"] or "")) or not LOWER_MAE.search(layer["text"][lf["start"]:lf["end"]]):
            continue
        if s["uuid"] in groups and leaf_keep.get(s["uuid"]) is None:
            continue  # kept whole already
        leaf_keep.setdefault(s["uuid"], set()).add(id(lf))
        groups[s["uuid"]] = groups.get(s["uuid"], set()) | {"deal_risk"}


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
    for uuid in mae_sections(layer):
        groups.setdefault(uuid, {"deal_risk"})
    extend_definitions(layer, leaf_keep)
    mae_definition_leaves(layer, groups, leaf_keep)
    undefined_mae_leaves(layer, groups, leaf_keep)
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
