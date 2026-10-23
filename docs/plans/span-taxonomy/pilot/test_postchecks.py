"""Tests for postchecks.py. The positive cases are clauses the extractor got wrong in the
definition-resolution re-check (plan section 7.7), trimmed; the negative cases are the
nearby wording each check must leave alone.

Run: python3 -m unittest test_postchecks   (from this directory)
"""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(__file__))
from postchecks import apply

# e3f57a62: the term closes a list whose item (ii) is the failure
E3F = """@@ S64 | ARTICLE V COVENANTS | Section 5.5 Exclusive Dealing. | groups=resolved
(e) Except as permitted by Section 5.5(f), the Board and each committee of the Board shall not:
(i) withhold, withdraw, qualify or modify (or publicly propose or resolve to withhold, withdraw, qualify or modify), in a manner adverse to Parent, the Company Recommendation;
(ii) fail to include the Company Recommendation in the Shareholder Materials;
(iii) approve or recommend, or publicly declare advisable or publicly propose to enter into, any letter of intent providing for any Acquisition Proposal (an “Alternative Acquisition Agreement”, and any of the actions set forth in these clauses (i)–(iii), a “Change of Recommendation”); or"""

# c28535d9: one long line, failure clauses before an inline definition of two terms
C28 = """@@ S40 | ARTICLE VI COVENANTS | Section 6.5 No Solicitation | groups=termination
(d) Except as permitted by this Section 6.5(d), neither the Starwood Waypoint Board, the Invitation Homes Board, nor any committee of either such board, shall (i) withhold, withdraw, modify or qualify, in a manner adverse to the other Party, the Starwood Waypoint Recommendation or the Invitation Homes Recommendation, as applicable, (ii) approve, adopt or recommend any Acquisition Proposal, (iii) fail to include the Starwood Waypoint Recommendation or the Invitation Homes Recommendation, as applicable, in the Disclosure Document (any of the actions described in clauses (i), (ii) or (iii) of this Section 6.5(d), a “Starwood Waypoint Adverse Recommendation Change” or a “Invitation Homes Adverse Recommendation Change”, respectively)."""

# 305c1d7a: the failure is implied by the request-to-affirm proviso
F305 = """@@ S90 | ARTICLE VIII COVENANTS | Section 8.3 No Solicitation | groups=termination
(c) Except as expressly permitted by this Section 8.3 the Company Board shall not (i) withhold, withdraw, modify or qualify, in any manner adverse to Parent or Purchaser, the Board Recommendation (it being understood that the Company Board may refrain from taking a position with respect to an Acquisition Proposal until the close of business of the tenth (10th) Business Day following a written request by Parent to the Company Board to affirm the Board Recommendation after the first public announcement of such Acquisition Proposal without such action being considered an adverse modification), (ii) approve, authorize or recommend any Acquisition Proposal (any action described in clause (i) or clause (ii), a “Change in Recommendation”) or (iii) enter into any letter of intent."""

NO_FAILURE = """@@ S90 | ARTICLE VIII COVENANTS | Section 8.3 No Solicitation | groups=termination
“Change in Recommendation” has the meaning set forth in Section 8.3(c), and the Company shall not fail to include the Board Recommendation in the Proxy Statement.
(c) The Company Board shall not (i) withhold, withdraw, modify or qualify, in any manner adverse to Parent, the Board Recommendation, (ii) approve or recommend any Acquisition Proposal (any action described in clause (i) or clause (ii), a “Change in Recommendation”).
(d) The Company Board may make a Change in Recommendation if it determines that the failure to make such Change in Recommendation would be inconsistent with its fiduciary duties."""


def fee(*codes):
    return {"agreement": "x", "instances": [{"concept": "ma.termination.fee.target", "finding": "present",
            "amounts": [{"amount": 1e7, "currency": "USD", "basis": "fixed", "pct": None, "quote": "$10,000,000"}],
            "triggers": [{"code": c, "text": "", "quote": "", "tail": None} for c in codes]}]}


def codes(doc, concept="ma.termination.fee.target"):
    inst = next(i for i in doc["instances"] if i["concept"] == concept)
    return [t["code"] for t in inst.get("triggers", [])]


class RecommendationFailure(unittest.TestCase):
    def check(self, text, expect):
        out, log = apply(fee("adverse_recommendation_change", "superior_proposal_termination"), text)
        self.assertEqual("recommendation_failure" in codes(out), expect)
        if expect:
            added = out["instances"][0]["triggers"][-1]["quote"]
            self.assertTrue(any(added in l for l in text.split("\n")), "quote must be verbatim from one line")
        return out

    def test_list_item_failure(self):
        out = self.check(E3F, True)
        self.assertEqual(out["instances"][0]["triggers"][-1]["quote"],
                         "fail to include the Company Recommendation in the Shareholder Materials")

    def test_inline_definition_failure(self):
        self.check(C28, True)

    def test_request_to_affirm(self):
        self.check(F305, True)

    def test_no_failure_in_definition(self):
        self.check(NO_FAILURE, False)

    def test_needs_an_adverse_recommendation_trigger(self):
        out, log = apply(fee("superior_proposal_termination"), E3F)
        self.assertEqual(codes(out), ["superior_proposal_termination"])
        self.assertEqual(log, [])

    def test_already_coded(self):
        out, log = apply(fee("adverse_recommendation_change", "recommendation_failure"), E3F)
        self.assertEqual(log, [])


def cap(quote, basis="fixed", amount=35e6, pct=None):
    return {"agreement": "x", "instances": [{"concept": "ma.indemn.cap", "finding": "present", "caps": [
        {"indemnitor": [], "amount": {"amount": amount, "currency": "USD", "basis": basis, "pct": pct, "quote": quote},
         "applies_to": ["fundamental_reps"], "excludes": [], "quote": quote}]}]}


def cap_amount(doc):
    return doc["instances"][0]["caps"][0]["amount"]


class CapEqualsPrice(unittest.TestCase):
    def test_consideration_shares(self):  # ab10f286
        a = cap_amount(apply(cap("shall not exceed the aggregate value of the Consideration Shares as of the applicable payment date"), "")[0])
        self.assertEqual((a["basis"], a["pct"], a["amount"]), ("pct_of_purchase_price", 100, "not_stated"))

    def test_hundred_percent(self):  # 2c7bb87e
        a = cap_amount(apply(cap("shall not exceed 100% of the Purchase Price"), "")[0])
        self.assertEqual((a["basis"], a["pct"]), ("pct_of_purchase_price", 100))

    def test_per_seller_received(self):
        a = cap_amount(apply(cap("in no event shall exceed the portion of the Purchase Price actually received by such Seller"), "")[0])
        self.assertEqual(a["pct"], 100)

    def test_percentage_is_left_alone(self):
        q = "shall not exceed ten percent (10%) of the Purchase Price"
        out, log = apply(cap(q, basis="pct_of_purchase_price", amount="not_stated", pct=10), "")
        self.assertEqual(cap_amount(out)["pct"], 10)
        self.assertEqual(log, [])

    def test_escrow_amount_is_left_alone(self):
        out, log = apply(cap("shall not exceed the Indemnity Escrow Amount"), "")
        self.assertEqual(log, [])

    def test_price_adjustment_is_left_alone(self):
        out, log = apply(cap("shall not exceed the Purchase Price Adjustment Escrow Amount"), "")
        self.assertEqual(log, [])


# b5eb61e9: both "termination fees" are capped expense reimbursements
B5E = """@@ S67 | ARTICLE IX TERMINATION | Section 9.03 Termination Fees and Expenses. | groups=deal_risk,termination
(a) In the event that this Agreement is terminated by Buyer pursuant to Section 9.01(a), then within five (5) Business Days after the date of such termination, Seller and the Shareholder (jointly and severally) shall pay to Buyer an amount not to exceed $200,000 (such amount, the “Buyer Termination Fee”) equal to Buyer’s out-of-pocket costs and expenses incurred in connection with the documentation and negotiation of this Agreement.
(b) In the event that this Agreement is terminated by Seller pursuant to Section 9.01(b), then within five (5) Business Days, Buyer shall pay to Seller an amount not to exceed $200,000 (such amount, the “Seller Termination Fee”) equal to Seller’s and the Shareholder’s out-of-pocket costs and expenses incurred in connection with the attempted consummation of the transactions.
(c) If this Agreement is terminated pursuant to Section 9.01(c), the Company shall pay Parent the Company Termination Fee of $5,000,000, plus Parent’s out-of-pocket expenses up to $1,000,000."""


def fees(target_quote, reverse_quote=None):
    side = lambda s: [{"side": s, "entity": "other", "defined_term": ""}]
    mk = lambda c, q, payor: {"concept": c, "finding": "present", "payor": side(payor), "payee": [],
                              "amounts": [{"amount": 200000, "currency": "USD", "basis": "fixed", "pct": None, "quote": q}],
                              "triggers": [{"code": "breach_no_tail", "text": "", "quote": "", "tail": None}]}
    insts = [mk("ma.termination.fee.target", target_quote, "target_side"),
             {"concept": "ma.termination.expense_reimbursement", "finding": "absent", "scope_reviewed": ["S67"]}]
    if reverse_quote:
        insts.append(mk("ma.termination.fee.reverse", reverse_quote, "buyer_side"))
    return {"agreement": "x", "instances": insts}


class ExpenseLabeledFee(unittest.TestCase):
    def test_both_directions_move(self):
        out, log = apply(fees("an amount not to exceed $200,000", "Buyer shall pay to Seller an amount not to exceed $200,000"), B5E)
        by = {i["concept"]: i for i in out["instances"]}
        self.assertEqual(by["ma.termination.fee.target"]["finding"], "absent")
        self.assertEqual(by["ma.termination.fee.reverse"]["finding"], "absent")
        er = by["ma.termination.expense_reimbursement"]
        self.assertEqual(er["finding"], "present")
        self.assertEqual(er["cap"]["amount"], 200000)
        self.assertEqual(er["payor"][0]["side"], "target_side")  # equal amounts: the target side's (schema)

    def test_fixed_fee_plus_expenses_stays(self):
        out, log = apply(fees("the Company Termination Fee of $5,000,000"), B5E)
        self.assertEqual(out["instances"][0]["finding"], "present")
        self.assertEqual(log, [])


# 6498472c: the price is stated as one number, the cap and basket only as percentages of it
P64 = """@@ S12 | ARTICLE II PURCHASE AND SALE | Section 2.2 Purchase Price. | groups=indemnification
The aggregate purchase price shall be $340,000,000 (the “Purchase Price”), as adjusted pursuant to Section 2.4.
@@ S80 | ARTICLE IX INDEMNIFICATION | Section 9.4 Limitations. | groups=indemnification
(a) Sellers shall not be liable unless Losses exceed three percent (3%) of the unadjusted Purchase Price (the “Deductible”), and then only in excess of the Deductible.
(b) In no event shall Sellers’ aggregate liability exceed twenty percent (20%) of the unadjusted Purchase Price."""


class StatedAmounts(unittest.TestCase):
    def test_computed_amount_dropped(self):
        out, log = apply(cap("twenty percent (20%) of the unadjusted Purchase Price", "pct_of_purchase_price", 68e6, 20), P64)
        a = cap_amount(out)
        self.assertEqual((a["amount"], a["pct"]), ("not_stated", 20))
        self.assertEqual(log[0]["rule"], "v2.6 rule 1")

    def test_basket_computed_amount_dropped(self):
        doc = {"agreement": "x", "instances": [{"concept": "ma.indemn.basket", "finding": "present", "baskets": [
            {"type": "deductible", "amount": {"amount": 10.2e6, "currency": "USD", "basis": "pct_of_purchase_price",
                                              "pct": 3, "quote": "three percent (3%)"}}], "de_minimis": "none"}]}
        out, log = apply(doc, P64)
        self.assertEqual(out["instances"][0]["baskets"][0]["amount"]["amount"], "not_stated")

    def test_stated_price_kept(self):  # v2.3 rule 2: a cap equal to a price stated as one number
        out, log = apply(cap("shall not exceed the Purchase Price", "pct_of_purchase_price", 340e6, 100), P64)
        self.assertEqual(cap_amount(out)["amount"], 340e6)
        self.assertEqual(log, [])

    def test_stated_in_words_kept(self):
        out, log = apply(cap("ten percent (10%) of the Purchase Price", "pct_of_purchase_price", 34e6, 10),
                         "Sellers' liability shall not exceed $34.0 million (ten percent (10%) of the Purchase Price).")
        self.assertEqual(cap_amount(out)["amount"], 34e6)

    def test_fixed_amount_left_alone(self):
        out, log = apply(cap("shall not exceed $35,000,000"), "")
        self.assertEqual(log, [])


if __name__ == "__main__":
    unittest.main()
