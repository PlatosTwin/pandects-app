"""Build inputs with the financing wordings the financing-retrieval round missed: prepare_v5
plus the stage 1 fixes from plan section 7.10.

Changes from prepare_v5 (all from the M4 holdout's stage 1 misses):
- funds representations worded "access to ... the financial resources", "available funds, or
  immediately available capacity under committed credit facilities" or "borrowing capacity";
- covenants to maintain sufficient cash or liquid accounts;
- a sponsor capitalizing the buyer with cash, including a defined "Cash Equity Capitalization";
- a closing that is "not conditioned upon the receipt of" financing;
- covenants to keep or amend a credit agreement, or to arrange alternative financing, so the
  buyer can finance the price.

Usage: python3 prepare_v6.py holdout.tsv OUT
       python3 prepare_v6.py --rebuild D OUT   (stage 1 again from D/layer, candidates to OUT)
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
import prepare_v4 as v4
import prepare_v5 as v5
from prepare_v2 import candidate_view

FINANCING = (r"|access to[^.;]{0,80}financial resources|capacity under[^.;]{0,40}credit facilit|borrowing capacity|"
             r"maintain[^.;]{0,30}(cash|liquid accounts)[^.;]{0,30}sufficient|sufficient liquid|"
             r"cash equity capitalization|capitalize (parent|buyer|purchaser|acquir\w*)[^.;]{0,80}\bcash\b|"
             r"conditioned (up)?on[^.;]{0,40}financing|credit agreement[^.;]{0,120}commitment|"
             r"alternative (sources of )?financing|financing from alternative|"
             r"finance the (purchase price|merger consideration|transactions?)")
t, x = v5.GROUPS["deal_risk"]
v5.GROUPS["deal_risk"] = (t, re.compile(x.pattern + FINANCING, re.I))  # v5.select reads its module GROUPS
select = v5.select


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
