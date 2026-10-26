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
- v2.2 rule 11: a target-fee or reimbursement trigger coded `other` that is a termination for
  breaching the covenant by failing to call or hold the stockholder meeting, with no tail, is
  `breach_no_tail` (plan section 7.17); on a reverse fee it is `buyer_breach` (section 7.19).
- v2.5 rule 3: a buyer funds representation that counts the buyer's own cash ("together with
  any available cash", "cash on hand", "marketable securities") adds `cash_on_hand` to committed
  financing (plan section 7.14). Contingency covenants ("In the event ... becomes unavailable",
  substitute financing) are not representations and are left alone.
- v2.5 rule 6 and v2.8 rule 8: an indemnity paid "first" from an escrow, holdback, policy or note
  and then by the sellers (or "the first source of payment") makes `escrow_sole_recourse` false
  (plan section 7.19). Purchase-price adjustments are left alone.
- v2.10 rule 3: with no such wording, a set-off right that "shall not constitute an election of
  remedies" or limit other remedies makes a silent `escrow_sole_recourse` false.
- v2.10 rule 1: a fee trigger coded `other` that is a termination at a time the payee "had the
  right to terminate" on another ground is dropped; the underlying trigger is already listed.
- v2.10 rule 2: on a reverse fee, a `buyer_approval_failure` quoting the buyer board's failure
  to make or include its recommendation becomes `recommendation_failure`, and a
  `buyer_recommendation_change` whose cited termination clause also covers that failure gets
  `recommendation_failure` beside it.

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


MEETING_BREACH = re.compile(r"\bbreach\w*[^;]{0,80}?\bby\s+(?:its\s+)?fail(?:ing|ure)\s+(?:to\s+)?(?:duly\s+)?"
                            r"(?:call|convene|hold|give notice of)\b[^.;]{0,80}?\bmeeting\b", re.I)


def check_meeting_breach(inst, log, code="breach_no_tail"):
    for t in inst.get("triggers") or []:
        if not isinstance(t, dict) or t.get("code") != "other" or t.get("tail"):
            continue
        q = " ".join(x for x in (t.get("quote"), t.get("text")) if isinstance(x, str))
        if MEETING_BREACH.search(q):
            t["code"], t["postcheck"] = code, "v2.2 rule 11"
            log.append({"concept": inst["concept"], "rule": "v2.2 rule 11", "match": MEETING_BREACH.search(q).group(0)})


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
                      r"|own (?:cash|funds)|available funds|together with[^.;]{0,80}\bcash\b|\bcash\b[^.;]{0,40}\bon hand\b"
                      # v2.12 rule 1: other funds counted beside the commitments, or the buyer itself having the funds
                      r"|cash-on-hand|other funds (?:available|of)|funds available to"
                      r"|\b(?:Parent|Buyer|Purchaser|Acquir\w*)\s+(?:(?:will|shall)\s+)?(?:at the Closing\s+)?(?:has|have)\s+"
                      r"(?:access to\s+)?(?:sufficient|adequate)\s+(?:funds|cash|financial resources)",
                      re.I)
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


def sentences(lines):
    for line in lines:
        if line.startswith("@@"):
            continue
        yield from re.split(r"(?<=[.;])\s+(?=[A-Z(])", line)


# v2.5 rule 6 and v2.8 rule 8: indemnity paid first from an escrow, holdback, policy or note and
# then by the sellers. Purchase-price adjustments and escrow-then-escrow chains are not this.
FIRST = re.compile(r"\bfirst source of (?:payment|recovery)|\b(?:shall|will|be|is|are)\s+(?:be\s+)?(?:first|initially)\s+(?:be\s+)?"
                   r"(?:made|paid|satisfied|recovered|sought|applied|recoverable|deducted|set ?off)|\b(?:first|initially),?\s+"
                   r"(?:from|out of|against|by)\b|\bfirst\s+(?:from|out of|against)\b", re.I)
FIRST_SOURCE = re.compile(r"\bfirst source of (?:payment|recovery)", re.I)
HELD = re.compile(r"escrow|holdback|hold-?back|r&w|policy|promissory|note\b|shares|retention|insurance", re.I)
INDEMNITY = re.compile(r"indemn|\blosses\b|\bdamages\b|adverse consequences", re.I)
ADJUSTMENT = re.compile(r"adjustment|working capital|true-?up|expense fund|purchase price", re.I)
THEN_SELLERS = re.compile(r"(?:\band then\b|,\s*then\b|\bthereafter\b|\bsecond(?:ly)?\b|\bfinally\b|\bbefore any\b)[^.]{0,250}?"
                          r"\b(?:sellers?|stockholders?|shareholders?|holders?|indemnitors?|members?|equityholders?|"
                          r"securityholders?|owners?|out-of-pocket|directly|indemnifying part)", re.I)
# v2.10 rule 3: a set-off right that expressly does not limit the buyer's other remedies
SETOFF = re.compile(r"set-?\s?off", re.I)
NOT_EXCLUSIVE = re.compile(r"election of remedies|shall not (?:be deemed to )?(?:limit|be (?:the )?exclusive)|"
                           r"not (?:be )?(?:the )?(?:sole|exclusive)|in addition to (?:any|all) other", re.I)


def recourse_beyond_escrow(lines):
    """(rule, sentence) for the first sentence showing recovery may go beyond the escrow."""
    for s in sentences(lines):
        m = FIRST.search(s)
        if (m and HELD.search(s) and INDEMNITY.search(s) and not ADJUSTMENT.search(s)
                and (FIRST_SOURCE.search(s) or THEN_SELLERS.search(s[m.start():]))):
            return "v2.8 rule 8", s[:300]
    for s in sentences(lines):
        if SETOFF.search(s) and NOT_EXCLUSIVE.search(s):
            return "v2.10 rule 3", s[:300]
    return None


def check_escrow_sole_recourse(inst, lines, log):
    v = inst.get("escrow_sole_recourse")
    value = v.get("value") if isinstance(v, dict) else v
    if value is False:
        return
    found = recourse_beyond_escrow(lines)
    if not found or found[0] == "v2.10 rule 3" and value is not None and value != "not_stated":
        return  # a non-exclusive set-off overrides only a silent extraction
    rule, quote = found
    inst["escrow_sole_recourse"] = {"value": False, "quote": quote, "postcheck": rule}
    log.append({"concept": inst["concept"], "rule": rule, "changed": f"escrow_sole_recourse {value} -> false",
                "quote": quote[:120]})


# v2.10 rule 1: a termination on another ground at a time the payee could have terminated on a
# fee trigger is that trigger, not `other`
HAD_RIGHT = re.compile(r"had (?:the |a )?right to terminate|at such time[^.]{0,80}right to terminate|could have terminated|"
                       r"(?:was|were) entitled to terminate", re.I)


def check_had_right(inst, log):
    trig = inst.get("triggers")
    if not isinstance(trig, list):
        return
    drop = [t for t in trig if isinstance(t, dict) and t.get("code") == "other"
            and HAD_RIGHT.search(f"{t.get('text') or ''} {t.get('quote') or ''}")]
    if drop and len(drop) < len(trig):
        inst["triggers"] = [t for t in trig if t not in drop]
        for t in drop:
            log.append({"concept": inst["concept"], "rule": "v2.10 rule 1", "dropped": (t.get("text") or "")[:120]})


# v2.10 rule 2: v2.2 rule 12 on the reverse fee. The buyer board's failure to make or include its
# recommendation is `recommendation_failure`, beside `buyer_recommendation_change`.
REC_FAIL = re.compile(r"\bfail(?:s|ed|ure)?\s+(?:to\s+)?(?:\w+\s+){0,2}?(?:include|make|recommend)\b[^.;]{0,80}?recommend|"
                      r"\bfail(?:s|ed|ure)?\s+to\s+recommend\b", re.I)
CLAUSE_REF = re.compile(r"\d+\.\d+\s*\(([a-z])\)")


def check_reverse_recommendation_failure(inst, lines, log):
    trig = [t for t in inst.get("triggers") or [] if isinstance(t, dict)]
    for t in trig:
        m = REC_FAIL.search(t.get("quote") or "")
        if t.get("code") == "buyer_approval_failure" and m:
            t["code"], t["postcheck"] = "recommendation_failure", "v2.10 rule 2"
            log.append({"concept": inst["concept"], "rule": "v2.10 rule 2", "recoded": (t.get("quote") or "")[:120]})
    codes = {t.get("code") for t in trig}
    if "buyer_recommendation_change" not in codes or "recommendation_failure" in codes:
        return
    brc = next(t for t in trig if t.get("code") == "buyer_recommendation_change")
    refs = set(CLAUSE_REF.findall(f"{brc.get('quote') or ''} {brc.get('text') or ''}"))
    for line in lines:
        m = re.match(r"\s*\(([a-z])\)\s+by\b", line)
        f = REC_FAIL.search(line) if m and m.group(1) in refs else None
        if f:
            phrase = f.group(0).strip()
            inst["triggers"].append({"code": "recommendation_failure", "text": phrase, "quote": phrase,
                                     "tail": copy.deepcopy(brc.get("tail")), "postcheck": "v2.10 rule 2"})
            log.append({"concept": inst["concept"], "rule": "v2.10 rule 2", "added": phrase})
            return


# v2.11 rule 8: a breach trigger that names its covenants only by number ("breached its obligations
# under Section 6.3 or 6.11"); the section titles say which covenant each number is
HEADER = re.compile(r"^@@ S\d+ \|[^|]*\| (?:Section\s+|SECTION\s+)?(\d+\.\d+)\.?\s+([^|]*)")
CITED = re.compile(r"\bSections?\s+((?:\d+\.\d+(?:\([A-Za-z0-9]{1,4}\))*(?:\s*(?:,|and/or|or|and)\s*(?:Section\s+)?)?)+)")
MEETING_TITLE = re.compile(r"(?:stock|share)holders?['’]?\s+(?:meetings?|approvals?|votes?)|special meeting|"
                           r"company meeting|proxy statement", re.I)
NO_SHOP_TITLE = re.compile(r"solicitation|no[- ]shop|acquisition proposals?|takeover proposals?|other offers", re.I)
BREACH = re.compile(r"\bbreach|fail(?:s|ed|ure)?\s+to\s+(?:comply|perform)", re.I)


def section_titles(lines):
    return {m.group(1): m.group(2).strip() for m in map(HEADER.match, lines) if m}


def cited_sections(text):
    return [n for m in CITED.finditer(text) for n in re.findall(r"\d+\.\d+", m.group(1))]


def check_cited_meeting_breach(inst, titles, log, code="breach_no_tail"):
    trig = [t for t in inst.get("triggers") or [] if isinstance(t, dict)]
    for t in list(trig):
        if t.get("tail") or t.get("code") not in ("other", "breach_of_no_shop" if code == "breach_no_tail" else "other"):
            continue
        q = " ".join(x for x in (t.get("quote"), t.get("text")) if isinstance(x, str))
        cited = [titles.get(n, "") for n in cited_sections(q)]
        if not BREACH.search(q) or not any(MEETING_TITLE.search(x) for x in cited):
            continue
        codes = {x.get("code") for x in trig}
        no_shop = code == "breach_no_tail" and any(NO_SHOP_TITLE.search(x) for x in cited)
        want = [code] + (["breach_of_no_shop"] if no_shop else [])
        if t["code"] == "other":
            t["code"], t["postcheck"] = want.pop(0), "v2.11 rule 8"
            log.append({"concept": inst["concept"], "rule": "v2.11 rule 8", "recoded": q[:120]})
        for c in want:
            if c not in codes:
                new = {"code": c, "text": t.get("text"), "quote": t.get("quote"), "tail": None, "postcheck": "v2.11 rule 8"}
                inst["triggers"].append(new)
                trig.append(new)
                log.append({"concept": inst["concept"], "rule": "v2.11 rule 8", "added": c})


# v2.11 rule 9: a tail trigger citing a termination clause that is the tender offer lapsing unaccepted
OFFER_LAPSE = re.compile(r"\bOffer\b[^;]{0,60}?\b(?:expired?|terminated|withdrawn)\b[^;]{0,160}?\bwithout\b[^;]{0,80}?"
                         r"\b(?:purchas|accept)", re.I)
CLAUSE = re.compile(r"(\d+\.\d+)\s*\(([a-z])\)")


def clause_text(lines, number, letter):
    """The text of clause (letter) of the section numbered number, up to the next lettered clause."""
    start = next((i for i, l in enumerate(lines) if (m := HEADER.match(l)) and m.group(1) == number), None)
    if start is None:
        return ""
    out = []
    for l in lines[start + 1:]:
        if l.startswith("@@"):
            break
        m = re.match(r"\s*\(([a-z])\)\s", l)
        if m and out and m.group(1) == chr(ord(letter) + 1):
            break
        if out or m and m.group(1) == letter:
            out.append(l)
    return " ".join(out)


def check_offer_lapse(inst, lines, log):
    trig = [t for t in inst.get("triggers") or [] if isinstance(t, dict)]
    if any(t.get("code") == "approval_failure_with_tail" for t in trig):
        return
    for t in trig:
        if not (str(t.get("code", "")).endswith("_with_tail") or t.get("code") == "other" and t.get("tail")):
            continue
        q = " ".join(x for x in (t.get("quote"), t.get("text")) if isinstance(x, str))
        for number, letter in CLAUSE.findall(q):
            m = OFFER_LAPSE.search(clause_text(lines, number, letter))
            if m:
                inst["triggers"].append({"code": "approval_failure_with_tail", "text": f"termination under Section {number}({letter})"
                                         " after the Offer lapses unaccepted", "quote": m.group(0),
                                         "tail": copy.deepcopy(t.get("tail")), "postcheck": "v2.11 rule 9"})
                log.append({"concept": inst["concept"], "rule": "v2.11 rule 9", "added": m.group(0)[:120]})
                return


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


def apply(doc, cand, full=None):
    """Return (checked copy of doc, change log). full, the whole agreement text, lets section
    numbers resolve to sections stage 1 did not keep."""
    doc, log, lines = copy.deepcopy(doc), [], lines_of(cand)
    titles = section_titles(lines_of(full) if full else lines)
    insts, figures = doc["instances"], stated_figures(cand)
    check_expense_labeled_fee(insts, lines, log)
    for inst in insts:
        if inst.get("finding") != "present":
            continue
        if inst["concept"] in ("ma.termination.fee.target", "ma.termination.expense_reimbursement"):
            check_recommendation_failure(inst, lines, log)
        if inst["concept"] in ("ma.termination.fee.target", "ma.termination.expense_reimbursement"):
            check_meeting_breach(inst, log)
            check_cited_meeting_breach(inst, titles, log)
            check_offer_lapse(inst, lines, log)
        if inst["concept"] in ("ma.termination.fee.target", "ma.termination.fee.reverse",
                               "ma.termination.expense_reimbursement"):
            check_had_right(inst, log)
        if inst["concept"] == "ma.termination.fee.reverse":
            check_meeting_breach(inst, log, "buyer_breach")  # the buyer-side code (v2.10 rule 2)
            check_cited_meeting_breach(inst, titles, log, "buyer_breach")
            check_reverse_recommendation_failure(inst, lines, log)
        if inst["concept"] == "ma.indemn.cap":
            check_cap_equals_price(inst, lines, log)
            check_escrow_sole_recourse(inst, lines, log)
        if inst["concept"] == "ma.financing.structure":
            check_cash_on_hand(inst, cand, log)
        check_stated_amounts(inst, figures, log)
    return doc, log


def main(D, arm_in, arm_out):
    os.makedirs(f"{D}/out/{arm_out}", exist_ok=True)
    changes = {}
    for p in sorted(glob.glob(f"{D}/out/{arm_in}/*.json")):
        u = os.path.basename(p)[:-5]
        full = f"{D}/read/{u}.md"
        doc, log = apply(json.load(open(p)), open(f"{D}/candidates/{u}.md").read(),
                         open(full).read() if os.path.exists(full) else None)
        json.dump(doc, open(f"{D}/out/{arm_out}/{u}.json", "w"), indent=1)
        if log:
            changes[u] = log
    return changes


if __name__ == "__main__":
    print(json.dumps(main(*sys.argv[1:4]), indent=1, default=str))
