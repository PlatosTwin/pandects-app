"""Build financing-retrieval inputs: prepare_v4 plus the stage 1 fixes from plan section 7.8.

Changes from prepare_v4 (all from the M3 holdout's stage 1 misses):
- funds representations worded "the funds necessary", "adequate (cash) resources", "access to
  sufficient resources" or "financing sufficient";
- SPAC trust accounts, PIPE and subscription investors, and buyer deposits (including those
  named in the recitals);
- financing closing conditions and covenants: lenders' consent, financing letters, cooperation
  with the buyer's banks or lenders;
- a buyer's guarantee of its own or an affiliate's obligations;
- every section of an indemnification article, whatever its title ("Limitations", "Payment of
  Claims", "Fraud").
- definitions sections keep a definition whose opening words name a relevant term even when the
  layer did not parse the term.

Usage: python3 prepare_v5.py holdout.tsv OUT
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
import prepare_v4 as v4
from prepare_v3 import GROUPS as V3_GROUPS
from text_layer import DEFS_TITLE

FINANCING = (r"|the funds (necessary|required|needed)|has available[^.;]{0,80}funds|adequate (cash )?resources|"
             r"access to sufficient|financing sufficient|sufficient (financing|resources)|trust account|"
             r"\bPIPE\b|subscription agreements?|(common stock|equity) investors|"
             r"[\"“](performance |earnest money |good faith |escrow )?deposit[\"”]|lenders?['’]? consent|"
             r"financing letter|cooperat\w*[^.;]{0,150}\b(financing|banks|lenders|syndicat\w*)\b|"
             r"hereby (unconditionally )?(and irrevocably )?guarantee")
t, x = V3_GROUPS["deal_risk"]
GROUPS = dict(V3_GROUPS, deal_risk=(re.compile(t.pattern + r"|deposit|trust account|guarant", re.I),
                                   re.compile(x.pattern + FINANCING, re.I)))
INDEMN_ARTICLE = re.compile(r"(?<!officers' )(?<!director )indemni(f|ty)", re.I)


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
        if INDEMN_ARTICLE.search(s["article"] or ""):
            g.add("indemnification")
        defs = [l for l in leaves if l["kind"] == "definition"]
        is_defs = DEFS_TITLE.search(f"{s['title'] or ''} {s['article'] or ''}") or s["uuid"].startswith("exhibit:")
        if is_defs and len(defs) >= 5:
            keep = {id(leaves[0])} if leaves and leaves[0]["kind"] != "definition" else set()
            # also a definition whose term the layer did not parse ('"Material Adverse Effect," as used ...')
            keep |= {id(l) for l in defs if any(v4.TERMS.search(t) for t in v4.leaf_terms(l))
                     or v4.TERMS.search(layer["text"][l["start"]:l["start"] + 60])}
            if keep:
                groups[s["uuid"]] = g | {"definitions"}
                leaf_keep[s["uuid"]] = keep
        elif g:
            groups[s["uuid"]] = g
    v4.resolve(layer, by_sec, groups, leaf_keep)
    return groups, leaf_keep


def main(dump, out):
    v4.select = select  # v4.main builds the views with whichever select it is given
    v4.main(dump, out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
