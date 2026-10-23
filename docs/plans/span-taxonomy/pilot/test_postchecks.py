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

    def test_price_less_claims_is_left_alone(self):  # 2fafb339
        q = "exceed an amount equal to the Purchase Price, less the aggregate amount of all Buyer Claims"
        out, log = apply(cap(q, basis="formula", amount="not_stated"), "")
        self.assertEqual(log, [])
        out, log = apply(cap(q), "")
        self.assertEqual(log, [])

    def test_stated_pct_of_received_is_left_alone(self):  # 67e84aa3
        q = "limited to thirty five percent (35%) of the portion of the Merger Consideration that has been paid"
        out, log = apply(cap(q, basis="pct_of_purchase_price", amount="not_stated", pct=35), "")
        self.assertEqual((cap_amount(out)["pct"], log), (35, []))

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



def escrow_cap(value):
    doc = cap("shall not exceed the Indemnity Escrow Amount")
    doc["instances"][0]["escrow_sole_recourse"] = value
    return doc


def recourse(doc):
    v = doc["instances"][0]["escrow_sole_recourse"]
    return v.get("value") if isinstance(v, dict) else v


S = "@@ S80 | ARTICLE X INDEMNIFICATION | Section 10.4 Manner of Payment | groups=indemnification\n"


class EscrowSoleRecourse(unittest.TestCase):
    def test_first_source_of_payment(self):  # 14e30716
        out, log = apply(escrow_cap({"value": True, "quote": "x"}), S + "(i) The first source of payment for Losses in respect of the Securityholder Indemnifiable Matters will be payment from the Indemnity Escrow Amount.")
        self.assertEqual((recourse(out), log[-1]["rule"]), (False, "v2.8 rule 8"))

    def test_escrow_then_sellers(self):  # 1509f5f1
        out, log = apply(escrow_cap("not_stated"), S + "Any Adverse Consequences recoverable by a Buyer Indemnified Party under this Article 7 will first be recovered from the Escrow Funds and then from the applicable Seller or Sellers.")
        self.assertIs(recourse(out), False)

    def test_escrow_then_policy_is_left_alone(self):  # 749cfebe: the second source is not the sellers
        out, log = apply(escrow_cap({"value": True, "quote": "x"}), S + "(a) In the case of any amount payable to a Purchaser Indemnitee under Section 7.2(a), such amount shall be satisfied (i) first, from the Escrow Amount until the Escrow Amount has been depleted, and (ii) second, from the R&W Insurance Policy.")
        self.assertIs(recourse(out), True)

    def test_price_adjustment_is_left_alone(self):  # 1818c1ec
        out, log = apply(escrow_cap({"value": True, "quote": "x"}), S + "Any Net Negative Adjustment Amount shall be recovered first from the Net Working Capital Expense Fund, second from the Escrow Fund and finally, from the Founder and the Outside Sellers according to their respective Indemnification Percentages.")
        self.assertIs(recourse(out), True)

    def test_adverbial_then_is_left_alone(self):  # 320c861f: "then remaining outstanding"
        out, log = apply(escrow_cap({"value": True, "quote": "x"}), S + "The indemnity obligation of the Escrow Participants shall be satisfied first by reducing the principal amount of the Stockholder Term B Loan by the amount of such Distribution Entitlement, up to the principal amount of the Stockholder Term B Loan then remaining outstanding.")
        self.assertIs(recourse(out), True)

    def test_non_exclusive_setoff(self):  # d166d547
        out, log = apply(escrow_cap("not_stated"), S + "(b) Buyer may set off any amount to which it is entitled under this Article IX. Neither the exercise nor the failure to exercise such right of setoff shall constitute an election of remedies or limit Buyer in any manner in the enforcement of any other remedies that may be available to it.")
        self.assertEqual((recourse(out), log[-1]["rule"]), (False, "v2.10 rule 3"))

    def test_non_exclusive_setoff_leaves_a_coded_value(self):
        out, log = apply(escrow_cap({"value": True, "quote": "x"}), S + "Neither the exercise nor the failure to exercise such right of setoff shall constitute an election of remedies.")
        self.assertIs(recourse(out), True)


def reverse(*triggers):
    return {"agreement": "x", "instances": [{"concept": "ma.termination.fee.reverse", "finding": "present",
            "amounts": [], "triggers": [dict({"tail": None, "text": ""}, **t) for t in triggers]}]}


R = "ma.termination.fee.reverse"


class HadRight(unittest.TestCase):
    def test_outside_date_while_payee_could_terminate(self):  # 9ac16a58
        out, log = apply(reverse({"code": "buyer_breach", "quote": "a"}, {"code": "failure_to_close", "quote": "b"},
                                 {"code": "other", "quote": "(ii) Parent pursuant to Section 8.01(b)(i) and at such time the Company had the right to terminate this Agreement pursuant to Section 8.01(d)(ii) or Section 8.01(d)(iii)"}), "")
        self.assertEqual((codes(out, R), log[-1]["rule"]), (["buyer_breach", "failure_to_close"], "v2.10 rule 1"))

    def test_could_have_terminated_in_text(self):  # 28803f36
        out, log = apply(reverse({"code": "buyer_breach", "quote": "a"},
                                 {"code": "other", "quote": "9.01(b)", "text": "termination at the End Date when the Company could have terminated under 9.01(f)"}), "")
        self.assertEqual(codes(out, R), ["buyer_breach"])

    def test_sole_trigger_is_kept(self):
        out, log = apply(reverse({"code": "other", "quote": "at such time the Company had the right to terminate"}), "")
        self.assertEqual((codes(out, R), log), (["other"], []))

    def test_other_other_is_kept(self):
        out, log = apply(reverse({"code": "buyer_breach", "quote": "a"}, {"code": "other", "quote": "a Parent Triggering Event shall have occurred"}), "")
        self.assertEqual(codes(out, R), ["buyer_breach", "other"])


BA52 = """@@ S70 | ARTICLE X TERMINATION | Section 10.01 Termination | groups=termination
(c) by the Company, if Pyramid shall have breached any of its representations, warranties or covenants;
(f) by the Company, if the Board of Directors of Pyramid shall have failed to recommend, or shall have withdrawn, modified or amended in a manner adverse to the Company in any material respect the Pyramid Board Recommendation;
(h) by Pyramid, if the Board of Directors of the Company shall have failed to recommend, or shall have withdrawn the Company Board Recommendation;"""


class ReverseRecommendationFailure(unittest.TestCase):
    def test_cited_clause_covers_failure(self):  # ba52fd38
        out, log = apply(reverse({"code": "buyer_breach", "quote": "a"},
                                 {"code": "buyer_recommendation_change", "quote": "the Company terminates this Agreement pursuant to Sections 10.01(c) (as a result of a breach by Pyramid) or 10.01(f)"}), BA52)
        self.assertEqual(codes(out, R), ["buyer_breach", "buyer_recommendation_change", "recommendation_failure"])
        self.assertEqual(out["instances"][0]["triggers"][-1]["quote"], "failed to recommend")

    def test_uncited_clause_is_left_alone(self):  # (h) is the target board's failure
        out, log = apply(reverse({"code": "buyer_recommendation_change", "quote": "pursuant to Section 10.01(c)"}), BA52)
        self.assertEqual(codes(out, R), ["buyer_recommendation_change"])

    def test_failure_to_include_recoded(self):  # 4d5b8c58
        out, log = apply(reverse({"code": "buyer_recommendation_change", "quote": "a Parent Adverse Recommendation Change shall have occurred"},
                                 {"code": "buyer_approval_failure", "quote": "the Board of Directors of Parent shall have failed to include its recommendation to the stockholders of Parent for the approval of the Parent Share Issuance in the Joint Proxy Statement"}), "")
        self.assertEqual(codes(out, R), ["buyer_recommendation_change", "recommendation_failure"])

    def test_vote_failure_is_left_alone(self):
        out, log = apply(reverse({"code": "buyer_approval_failure", "quote": "the Parent Stockholder Approval shall not have been obtained"}), "")
        self.assertEqual((codes(out, R), log), (["buyer_approval_failure"], []))

    def test_meeting_breach_is_buyer_breach(self):  # 4d5b8c58
        out, log = apply(reverse({"code": "other", "quote": "(ii) Parent shall have materially breached its obligations under Section 5.01 by failing to hold the Parent Stockholders' Meeting"}), "")
        self.assertEqual(codes(out, R), ["buyer_breach"])

if __name__ == "__main__":
    unittest.main()


def financing(*codes):
    return {"agreement": "x", "instances": [{"concept": "ma.financing.structure", "finding": "present",
            "committed_financing": [{"code": c, "sources": [], "quote": ""} for c in codes]}]}


def committed(doc):
    return [c["code"] for c in doc["instances"][0]["committed_financing"]]


# 019d7ad0: the Financing proceeds "together with any available cash of Parent" are sufficient
F019 = """@@ S53 | ARTICLE IV REPRESENTATIONS AND WARRANTIES OF PARENT | Section 4.6 Financing | groups=deal_risk
(b) Assuming the accuracy of the representations and warranties set forth in Article III, the proceeds of the Financing, if funded, together with any available cash of Parent, the Company and their respective Subsidiaries, shall constitute sufficient funds to pay the Merger Consideration and all other amounts payable by Parent hereunder."""

# the same wording inside a covenant about replacing unavailable financing
COVENANT = """@@ S74 | ARTICLE VI COVENANTS | Section 6.10 Financing | groups=deal_risk
(c) In the event that all or any portion of the Financing becomes unavailable, Parent shall use reasonable best efforts to obtain alternative financing in an amount that, together with any available cash of Parent, is sufficient to pay the Merger Consideration."""


class CashOnHand(unittest.TestCase):
    def test_funds_rep_counting_buyer_cash(self):
        out, log = apply(financing("debt"), F019)
        self.assertEqual(committed(out), ["debt", "cash_on_hand"])
        self.assertIn("together with any available cash", out["instances"][0]["committed_financing"][1]["quote"])

    def test_covenant_is_left_alone(self):
        out, log = apply(financing("debt"), COVENANT)
        self.assertEqual((committed(out), log), (["debt"], []))

    def test_debt_only_rep_is_left_alone(self):
        rep = "@@ S53 | x | y | groups=deal_risk\n(b) The net proceeds of the Debt Financing will be sufficient to pay the Merger Consideration."
        out, log = apply(financing("debt"), rep)
        self.assertEqual(log, [])

    def test_already_coded(self):
        out, log = apply(financing("debt", "cash_on_hand"), F019)
        self.assertEqual((committed(out), log), (["debt", "cash_on_hand"], []))

    def test_substitute_financing_covenant_is_left_alone(self):  # 0d305abd
        cov = ("@@ S74 | x | y | groups=deal_risk\nParent shall have the right to substitute commitments for all or any portion of "
               "the Debt Financing so long as the Debt Financing, together with other cash and cash equivalents available to Parent, "
               "is sufficient to pay the Merger Consideration.")
        out, log = apply(financing("debt"), cov)
        self.assertEqual(log, [])


def other_trigger(quote, tail=None):
    doc = fee("adverse_recommendation_change")
    doc["instances"][0]["triggers"].append({"code": "other", "text": "", "quote": quote, "tail": tail})
    return doc


class MeetingBreach(unittest.TestCase):
    def test_failing_to_call_and_hold(self):  # 20205629
        out, log = apply(other_trigger("shall have breached Section 6.3 by failing to call and hold the Company Meeting, as provided therein"), "")
        self.assertEqual(codes(out), ["adverse_recommendation_change", "breach_no_tail"])

    def test_failing_to_hold(self):  # 8eeea4af
        out, log = apply(other_trigger("(1) materially breached its obligations under Section 5.1 by failing to hold the Company Stockholders Meeting"), "")
        self.assertEqual(codes(out)[-1], "breach_no_tail")

    def test_tail_is_left_alone(self):
        tail = {"months": 12, "threshold_pct": 50, "requires": "either"}
        out, log = apply(other_trigger("failing to hold the Company Stockholders Meeting", tail), "")
        self.assertEqual((codes(out)[-1], log), ("other", []))

    def test_other_trigger_is_left_alone(self):
        out, log = apply(other_trigger("a Triggering Event shall have occurred"), "")
        self.assertEqual(log, [])

    def test_failing_to_hold_by_a_date_is_left_alone(self):  # 679089f9: no breach
        out, log = apply(other_trigger("(vii) the Company shall have failed to hold the Stockholders' Meeting as promptly as practicable and in any event by no later than July 1, 2006"), "")
        self.assertEqual(log, [])

    def test_meeting_date_reference_is_left_alone(self):  # 7051d0a3
        out, log = apply(other_trigger("on or prior to the date of the Stockholders Meeting, any event giving rise to Parent's right to terminate (other than a breach) occurred"), "")
        self.assertEqual(log, [])
