"""Deterministic post-checks on extractor output (plan section 7.8): rules the extractor
saw the text for but did not apply in the definition-resolution re-check (section 7.7).

- v2.2 rule 12: a fee trigger on an adverse recommendation change also gets
  `recommendation_failure` when the agreement's definition of that change includes failing
  to make, include or reaffirm the recommendation.
- v2.3 rule 2: a cap of "the Purchase Price" (or merger consideration, consideration shares,
  or 100% of the price) is `pct_of_purchase_price` 100, unless the model already coded a
  percentage or a formula, or the price is followed by "less" or "minus" (v2.8 rule 7).
- An amount labeled a termination fee that is defined as the payee's out-of-pocket expenses
  (up to a stated maximum) is expense reimbursement, not a fee (v2.1 rule 10).
- v2.6 rule 1: an amount stated as a percentage of a price keeps a dollar figure only when the
  text states that figure; one the extractor computed becomes `"not_stated"`.
- v2.5 rule 3: a buyer funds representation that counts the buyer's own cash ("together with
  any available cash", "cash on hand", "marketable securities") adds `cash_on_hand` to committed
  financing (plan section 7.14). Contingency covenants ("In the event ... becomes unavailable",
  substitute financing) are not representations and are left alone.

Each check reads only the candidates file the extractor saw, so every quote it adds is
verbatim from one of its lines. Changed entries carry "postcheck": "<rule>".

Usage: python3 postchecks.py D IN_ARM OUT_ARM   (reads D/out/IN_ARM, writes D/out/OUT_ARM)
"""
import copy, glob, json, os, re, sys

REC_TERM = re.compile(r"[\"“]((?!Notice)[^\"”]{0,60}?(Recommendation Change|Change (?:of|in) (?:the )?[\w ]{0,25}?Recommendation"
                      r"|Recommendation Withdrawal|Adverse Recommendation)[^\"”]{0,10})[\"”]")
# "fail to include the Company Recommendation", "fail to publicly reaffirm", "fail to recommend against"
FAIL = re.compile(r"\bfail(?:s|ed|ure)?\s+(?:to\s+)?(?:\w+\s+){0,2}?(?:include|make|reaffirm|affirm|recommend)\b"
                  r"(?![^.;]{0,40}?(?:Change|would be inconsistent|fiduciary))[^.;,]{0,60}")
# "may refrain ... until ... following a written request by Parent ... to affirm the Board Recommendation"
AFFIRM_REQUEST = re.compile(r"request[^.;]{0,80}?\bto (?:re)?affirm the [\w ]{0,30}Recommendation")
ITEM = re.compile(r"^\s*\((?:[ivx]{1,5}|[A-Z]|\d{1,2})\)\s")

PRICE = (r"(?:Purchase Price|Merger Consideration|Aggregate Consideration|Total Consideration|Closing Consideration"
         r"|Transaction Consideration|Consideration Shares|Consideration)(?!\s+(?:Adjustment|Escrow|Holdback|Allocation))")
PRICE_CAP = re.compile(r"\b(?:exceed|in excess of|greater than|more than|up to|limited to|equal to)\s+(?:an amount equal to\s+)?"
                       r"(?:100%\s+of\s+|one hundred percent \(100%\) of\s+)?(?:the\s+)?(?:aggregate\s+|total\s+|final\s+)?"
                       r"(?:(?:value|amount) of the\s+)?" + PRICE, re.I)
LESS = re.compile(r",?\s*(?:less|minus|reduced by)\b", re.I)  # "the Purchase Price, less ...": a formula
PRICE_RECEIVED = re.compile(r"portion of the\s+" + PRICE + r"[^.;]{0,40}?\b(?:actually\s+)?(?:received|paid)", re.I)

EXPENSE_FEE = [
    re.compile(r"(?:not to exceed|up to|maximum of|capped at)\s+\$[\d,.]+[^.;]{0,120}?\bequal to\b[^.;]{0,60}?"
               r"(?:out-of-pocket|costs and expenses|fees and expenses)", re.I),
    re.compile(r"\bequal to\b[^.;]{0,60}?out-of-pocket[^.;]{0,200}?(?:not to exceed|up to|in excess of)\s+\$", re.I)]


def lines_of(cand):
    return cand.split("\n")


def rec_change_definitions(lines):
    """Text of each recommendation-change definition: the defining line up to the term, plus the
    earlier list items and lead-in line when the term closes a list."""
    out = []
    for i, l in enumerate(lines):
        if l.startswith("@@"):
            continue
        for m in REC_TERM.finditer(l):
            if re.match(r"\s*(means|has the meaning|shall have the meaning)", l[m.end():]) and "Section" in l[m.end():m.end() + 60]:
                continue  # a cross-reference to the definition elsewhere
            body = [l[:m.start()] if not re.match(r"\s*means", l[m.end():]) else l]
            if ITEM.match(l):
                for j in range(i - 1, max(i - 9, -1), -1):
                    if lines[j].startswith("@@"):
                        break
                    body.insert(0, lines[j])
                    if not ITEM.match(lines[j]):
                        break
            out.append(body)
    return out


def failure_phrase(lines):
    """The first verbatim failure-to-recommend phrase in a recommendation-change definition."""
    for body in rec_change_definitions(lines):
        for t in body:
            m = FAIL.search(t) or AFFIRM_REQUEST.search(t)
            if m:
                return m.group(0).strip().rstrip(",;(")
    return None


def check_recommendation_failure(inst, lines, log):
    codes = [t.get("code") for t in inst.get("triggers") or [] if isinstance(t, dict)]
    if "adverse_recommendation_change" not in codes or "recommendation_failure" in codes:
        return
    phrase = failure_phrase(lines)
    if not phrase:
        return
    arc = next(t for t in inst["triggers"] if isinstance(t, dict) and t.get("code") == "adverse_recommendation_change")
    inst["triggers"].append({"code": "recommendation_failure", "text": phrase, "quote": phrase,
                             "tail": copy.deepcopy(arc.get("tail")), "postcheck": "v2.2 rule 12"})
    log.append({"concept": inst["concept"], "rule": "v2.2 rule 12", "added": phrase})


def check_cap_equals_price(inst, lines, log):
    for cap in inst.get("caps") or []:
        if not isinstance(cap, dict):
            continue
        amt = cap.get("amount") if isinstance(cap.get("amount"), dict) else {}
        if amt.get("basis") == "formula" or amt.get("pct") is not None:
            continue  # a stated percentage, or a price less a part of it (v2.8 rule 7), is already coded
        for q in (cap.get("quote"), amt.get("quote")):
            m = isinstance(q, str) and (PRICE_CAP.search(q) or PRICE_RECEIVED.search(q))
            if m and not LESS.match(q, m.end()):
                cap["amount"] = {"amount": "not_stated", "currency": amt.get("currency", "USD"),
                                 "basis": "pct_of_purchase_price", "pct": 100, "quote": q}
                cap["postcheck"] = "v2.3 rule 2"
                log.append({"concept": inst["concept"], "rule": "v2.3 rule 2", "was": amt, "match": m.group(0)})
                break


SUFFICIENT = re.compile(r"\b(?:sufficient|necessary|adequate|enough)\b", re.I)
OWN_CASH = re.compile(r"cash on hand|available cash|cash and cash equivalents|cash resources|marketable securities|existing cash"
                      r"|own (?:cash|funds)|available funds|together with[^.;]{0,80}\bcash\b|\bcash\b[^.;]{0,40}\bon hand\b", re.I)
BUYER = re.compile(r"\b(?:Parent|Buyer|Purchaser|Acquir\w*|Merger Sub|Pubco|SPAC)\b")
PAYABLE = re.compile(r"\b(?:consummate|pay|payment|amounts? payable|Merger Consideration|Purchase Price|obligations)\b", re.I)
CONTINGENCY = re.compile(r"^\s*(?:\(\w{1,4}\)\s*)?(?:in the event|if|provided)\b|becomes? unavailable|\bdeposit|substitut", re.I)


def funds_rep_with_cash(cand):
    """The first candidate sentence in which the buyer represents its own cash is sufficient."""
    for line in lines_of(cand):
        if line.startswith("@@"):
            continue
        for sent in re.split(r"(?<=[.;])\s+", line):
            if (SUFFICIENT.search(sent) and OWN_CASH.search(sent) and BUYER.search(sent) and PAYABLE.search(sent)
                    and not CONTINGENCY.search(sent)):
                return sent[:300]
    return None


def check_cash_on_hand(inst, cand, log):
    committed = inst.get("committed_financing")
    if not isinstance(committed, list) or any(isinstance(c, dict) and c.get("code") == "cash_on_hand" for c in committed):
        return
    quote = funds_rep_with_cash(cand)
    if quote:
        committed.append({"code": "cash_on_hand", "sources": [], "quote": quote, "postcheck": "v2.5 rule 3"})
        log.append({"concept": inst["concept"], "rule": "v2.5 rule 3", "added": quote[:120]})


# "$3,400,000", "3.4 million", "$1.29 billion", "105,000"
FIGURE = re.compile(r"(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?:\s*(million|billion))?", re.I)
SCALE = {None: 1, "million": 1e6, "billion": 1e9}


def stated_figures(cand):
    return {float(m.group(1).replace(",", "") + (m.group(2) or "")) * SCALE[(m.group(3) or "").lower() or None]
            for m in FIGURE.finditer(cand)}


def money_values(node):
    """Every Money object in an instance, at any depth."""
    if isinstance(node, dict):
        if "basis" in node and "amount" in node:
            yield node
        for v in node.values():
            yield from money_values(v)
    elif isinstance(node, list):
        for v in node:
            yield from money_values(v)


def check_stated_amounts(inst, figures, log):
    for m in money_values(inst):
        a = m.get("amount")
        if not str(m.get("basis", "")).startswith("pct_of") or m.get("pct") is None or not isinstance(a, (int, float)):
            continue
        if any(abs(a - f) < 0.5 for f in figures):
            continue
        m["amount"] = "not_stated"
        m["postcheck"] = "v2.6 rule 1"
        log.append({"concept": inst["concept"], "rule": "v2.6 rule 1", "was": a, "basis": m["basis"], "pct": m["pct"]})


def line_with(lines, quote):
    return next((l for l in lines if quote and quote in l and not l.startswith("@@")), "")


def is_expense_amount(a, lines):
    if not isinstance(a, dict):
        return False
    line = line_with(lines, a.get("quote"))
    return bool(line) and any(p.search(line) for p in EXPENSE_FEE) and not re.search(r"\bplus\b[^.;]{0,80}expenses", line, re.I)


def check_expense_labeled_fee(insts, lines, log):
    by = {i.get("concept"): i for i in insts}
    for c in ("ma.termination.fee.target", "ma.termination.fee.reverse"):
        fee = by.get(c)
        if not fee or fee.get("finding") != "present":
            continue
        amounts = [a for a in fee.get("amounts") or []]
        if not amounts or not all(is_expense_amount(a, lines) for a in amounts):
            continue
        er = by.get("ma.termination.expense_reimbursement")
        if er is None:
            er = {"concept": "ma.termination.expense_reimbursement"}
            insts.append(er)
            by[er["concept"]] = er
        if er.get("finding") != "present":
            a = amounts[0]
            er.clear()
            er.update({"concept": "ma.termination.expense_reimbursement", "finding": "present",
                       "scope_reviewed": fee.get("scope_reviewed", []), "spans": fee.get("spans", []),
                       "payor": fee.get("payor"), "payee": fee.get("payee"),
                       "cap": {"amount": a.get("amount"), "currency": a.get("currency", "USD"), "basis": "fixed",
                               "pct": None, "quote": a.get("quote")},
                       "triggers": fee.get("triggers", []), "credited_against_fee": "not_stated",
                       "notes": f"moved from {c} by post-check", "postcheck": "expense-labeled fee"})
        else:
            er["notes"] = (er.get("notes") or "") + f" Also owed in the other direction ({c}, moved by post-check)."
        note = fee.get("notes") or ""
        fee.clear()
        fee.update({"concept": c, "finding": "absent", "scope_reviewed": er.get("scope_reviewed", []),
                    "notes": (note + " The amount labeled a termination fee is out-of-pocket expense reimbursement"
                              " (moved to ma.termination.expense_reimbursement by post-check).").strip(),
                    "postcheck": "expense-labeled fee"})
        log.append({"concept": c, "rule": "expense-labeled fee", "moved": [a.get("amount") for a in amounts if isinstance(a, dict)]})


def apply(doc, cand):
    """Return (checked copy of doc, change log)."""
    doc, log, lines = copy.deepcopy(doc), [], lines_of(cand)
    insts, figures = doc["instances"], stated_figures(cand)
    check_expense_labeled_fee(insts, lines, log)
    for inst in insts:
        if inst.get("finding") != "present":
            continue
        if inst["concept"] in ("ma.termination.fee.target", "ma.termination.expense_reimbursement"):
            check_recommendation_failure(inst, lines, log)
        if inst["concept"] == "ma.indemn.cap":
            check_cap_equals_price(inst, lines, log)
        if inst["concept"] == "ma.financing.structure":
            check_cash_on_hand(inst, cand, log)
        check_stated_amounts(inst, figures, log)
    return doc, log


def main(D, arm_in, arm_out):
    os.makedirs(f"{D}/out/{arm_out}", exist_ok=True)
    changes = {}
    for p in sorted(glob.glob(f"{D}/out/{arm_in}/*.json")):
        u = os.path.basename(p)[:-5]
        doc, log = apply(json.load(open(p)), open(f"{D}/candidates/{u}.md").read())
        json.dump(doc, open(f"{D}/out/{arm_out}/{u}.json", "w"), indent=1)
        if log:
            changes[u] = log
    return changes


if __name__ == "__main__":
    print(json.dumps(main(*sys.argv[1:4]), indent=1, default=str))
