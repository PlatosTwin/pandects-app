# Extraction schema v2.12: ten concepts

Ten concepts, one instance per agreement per concept. Output one JSON file per agreement:

```json
{"agreement": "<uuid>", "instances": [ <one instance per concept, all ten> ]}
```

Always emit all ten. If the agreement has no such provision, emit `{"concept": ..., "finding": "absent", "scope_reviewed": ["S12", "S40"], "notes": "..."}`.

## Common rules

- **Quotes are evidence.** Every `quote` is copied character for character from ONE line of the text (never a `@@` marker line, never joined across lines). Keep it short: the shortest phrase that proves the value, usually 5–40 words.
- **Spans** (`spans[]`): `{"role", "section": "S<n>", "quote": "<first 10-20 words of the passage>", "quote_end": "<last 5-15 words, if the passage runs past one line>"}`. Roles: `operative` (the clause that creates the obligation, right or definition), `trigger` (a termination right or event a fee keys off; **one trigger span per trigger**), `amount`, `definition` (a defined term that supplies a value), `extension`, `remedy`, `carveout`, `exception`, `other`.
- **Absence.** `finding: "absent"` means the agreement has no such provision. Inside a present instance, a field the agreement is silent on is `"not_stated"`. Never guess. A value defined outside the text you have (an exhibit, an annex, a missing article) is `"not_stated"` with a note naming where the text says it is.
- **Coded fields** are `{"code": ..., "text": "<verbatim phrase>", "quote": "..."}`. Use a listed code when it fits. Otherwise use `"other"` with `text`, or `"proposed:<snake_case>"` if a new code is warranted.
- **Parties** (v2): every party field is a list of `{"side": "target_side" | "buyer_side" | "both" | "other", "entity": "company" | "parent" | "merger_sub" | "buyer" | "seller" | "stockholders" | "sponsor" | "guarantor" | "other", "defined_term": "Parent"}`. `side` is the side of the deal. The target and its sellers or stockholders are `target_side`; the acquirer and its affiliates are `buyer_side`. `entity` is the party's structural role: `parent` for an entity defined as Parent above a merger sub; `buyer` for a direct purchaser defined as Buyer, Purchaser or Acquiror.
- **Money:** `{"amount": number | "not_stated", "currency": "USD", "basis": "fixed" | "pct_of_equity_value" | "pct_of_enterprise_value" | "pct_of_purchase_price" | "pct_of_market_cap" | "formula" | "other", "pct": number | null, "quote": "..."}`.
- **Duration:** `{"value": number, "unit": "days" | "business_days" | "months" | "years", "anchor": "signing" | "closing" | "termination" | "other"}`.
- **Lists stay lists.** A list field with no values is `[]` only when the provision truly has none; an amount defined outside the text is a list entry with `"amount": "not_stated"`.

## Termination concepts

### `ma.termination.fee.target`: termination fee payable by the target side

```json
{"concept": "ma.termination.fee.target", "finding": "present", "scope_reviewed": [...], "spans": [...],
 "payor": [<party>], "payee": [<party>], "liability": "single" | "joint_and_several" | "several",
 "amounts": [{<Money>, "label": "standard" | "go_shop" | "superior_proposal" | "alternative" | "other"}],
 "triggers": [{"code": "...", "text": "...", "quote": "...",
               "tail": null | {"months": 12, "threshold_pct": 50, "requires": "definitive_agreement" | "consummation" | "either" | "not_stated"}}],
 "remedy": {"code": "exclusive" | "non_exclusive" | "not_stated", "carveouts": ["fraud", "willful_breach"], "quote": "..."},
 "payment_timing": {"text": "...", "quote": "..."} | "not_stated",
 "pay_once_only": true | false | "not_stated",
 "expenses_credited_against_fee": true | false | "not_stated",
 "notes": ""}
```

Trigger codes:
- `superior_proposal_termination`: the target terminates to accept a superior proposal.
- `adverse_recommendation_change`: termination following a change of board recommendation (including approving or recommending a competing proposal).
- `recommendation_failure`: failure to reaffirm, to include the recommendation in the proxy, or to oppose a tender offer.
- `recommendation_change_no_termination`: the fee is payable on a recommendation change without any termination.
- `breach_of_no_shop`: termination for a breach of the non-solicitation covenant.
- `approval_failure_with_tail`: the stockholder vote fails, or a tender offer expires with the minimum condition unmet, plus an acquisition proposal and a tail.
- `outside_date_with_tail`: termination at the outside date, plus an acquisition proposal and a tail.
- `breach_with_tail`: termination for breach, plus an acquisition proposal and a tail.
- `approval_failure_naked`: a fee on a failed vote or minimum condition alone, with no tail.
- `other`.

When one termination right covers several events (for example a recommendation change "or" a failure to reaffirm), emit one trigger per matching code. Each trigger gets its own `trigger` span on the termination right it refers to.

**Remedy rule:**
- `exclusive` when the text says the fee is the sole and exclusive remedy, or that after termination and payment no party has further liability.
- `non_exclusive` only when the text says the fee is in addition to other remedies, or is cumulative.
- Otherwise `not_stated`.

List in `carveouts` the claims expressly preserved (`fraud`, `willful_breach`, `other`).

### `ma.termination.fee.reverse`: termination fee payable by the buyer side

Same fields as the target fee, plus:

```json
 "kind": {"code": "financing" | "regulatory" | "general" | "reciprocal" | "two_tier" | "other", "text": "...", "quote": "..."},
 "guarantee": {"code": "limited_guarantee" | "equity_commitment" | "parent_guarantee" | "none" | "not_stated", "quote": "..."},
 "liability_cap": {"text": "...", "quote": "..."} | "none" | "not_stated",
 "specific_performance_interaction": {"code": "exclusive_alternative" | "both_available" | "conditional" | "not_stated", "quote": "..."}
```

`kind` codes:
- `financing`: the fee keys off a failure of the buyer's debt financing, or the buyer's failure to close when financing is unavailable.
- `regulatory`: the fee keys off an antitrust or regulatory failure.
- `general`: the fee keys off buyer breach or failure to close, with no tie to financing or regulation.
- `reciprocal`: the same fee, owed by the buyer on triggers that mirror the target's.
- `two_tier`: two or more distinct reverse fees for different triggers. Record each amount with a label.

Extra trigger codes:
- `failure_to_close`: the buyer fails to close when the conditions are satisfied.
- `regulatory_failure`: an approval is not obtained, an antitrust injunction, or the outside date with only regulatory conditions open.
- `financing_failure`.
- `buyer_breach`.
- `buyer_approval_failure`: the buyer's stockholder vote fails.
- `buyer_recommendation_change`.

Mirrored tail triggers use the target codes.

`specific_performance_interaction`:
- `exclusive_alternative`: the target may seek specific performance or the fee, not both.
- `both_available`: the target may pursue both, though it can receive only one.
- `conditional`: specific performance is available only under conditions, for example when the debt is funded.

### `ma.termination.expense_reimbursement`: reimbursement of a party's transaction expenses on termination

```json
 "payor": [<party>], "payee": [<party>],
 "cap": <Money> | "none" | "not_stated",
 "triggers": [<trigger, same codes as the fees>],
 "credited_against_fee": true | false | "not_stated"
```

Only a payment of a party's own deal expenses because the agreement terminated. These are NOT this concept:
- reimbursement of financing-cooperation costs;
- costs of collecting an unpaid fee;
- a split of filing fees;
- a general "each party pays its own expenses" clause.

If both sides reimburse, emit one instance with both directions in `notes`. Code the larger obligation, and when the two are equal, the target side's.

### `ma.termination.outside_date`: outside date / end date

```json
 "defined_term": "End Date" | null,
 "initial": {"date": "2016-12-31" | null, "duration": <Duration> | null, "quote": "..."},
 "effect": {"code": "termination_right" | "automatic_termination", "quote": "..."},
 "extensions": [{"length": <Duration> | null, "new_date": "..." | null,
                 "condition": {"code": "regulatory_pending" | "marketing_period" | "sec_review" | "notice_period" | "other", "text": "...", "quote": "..."},
                 "exercise": "automatic" | "either_party" | "buyer" | "target" | "mutual" | "not_stated", "count": 1 | null, "quote": "..."}],
 "breach_bar": {"conduct": "breach" | "failure_to_perform" | "any_action_or_inaction" | "willful_breach" | "other",
                "causation": "principal_or_primary_cause" | "cause_or_result" | "no_causation_required" | "other",
                "text": "...", "quote": "..."} | "none" | "not_stated"
```

- "Or such later date as the parties may agree" is **not** an extension.
- Fill `length` only when the text states one. Do not compute it from dates.
- `breach_bar.causation`:
  - `principal_or_primary_cause` for "the principal cause of", "primarily resulted in", or "the primary cause";
  - `cause_or_result` for "the cause of, or resulted in", "resulted from", "is the result of", "attributable to", or "due to";
  - `no_causation_required` when any breach, or failure to perform, bars the right whether or not it caused the delay.

## Deal-risk concepts

### `ma.mae.definition`: material adverse effect definition

```json
 "definitions": [{"defined_term": "Company Material Adverse Effect", "party_side": "target_side" | "buyer_side" | "both",
                  "prongs": {"business_condition": true, "ability_to_consummate": true | false, "prospects": true | false},
                  "standard": {"text": "would reasonably be expected to", "quote": "..."},
                  "carveouts": [{"categories": ["general_economic", "industry"], "text": "...", "quote": "..."}],
                  "disproportionate_effect_exception": {"applies_to": ["general_economic", "industry"], "quote": "..."} | "none",
                  "quote": "..."}]
```

- One entry per MAE definition (target and, if separately defined, buyer).
- Each lettered or numbered carve-out clause is one `carveouts` entry, with every category it covers.
- `prospects`: true only if the word "prospects" (or similar) appears in the business prong.
- `disproportionate_effect_exception.applies_to` lists the categories the "except to the extent disproportionately affects" proviso reaches.

Carve-out category codes:
- `general_economic`, `financial_markets`, `industry`;
- `change_in_law`, `change_in_gaap`;
- `political_war_terrorism`, `pandemic`, `natural_disaster`;
- `announcement_or_pendency`, `compliance_with_agreement`, `buyer_action_or_request`;
- `failure_to_meet_projections`, `stock_price_or_volume`, `credit_rating`;
- `stockholder_litigation`, `disclosure_schedule_items`;
- `seasonality`, `labor`, `other`.

### `ma.financing.structure`: the buyer's acquisition financing

```json
 "financing_condition": {"value": true | false, "quote": "..."},
 "committed_financing": [{"code": "debt" | "equity" | "cash_on_hand" | "other", "sources": ["JPMorgan Chase Bank"], "quote": "..."}],
 "efforts_standard": {"code": "hell_or_high_water" | "reasonable_best" | "commercially_reasonable" | "other", "text": "...", "quote": "..."} | "not_stated",
 "marketing_period": {"duration": <Duration>, "quote": "..."} | "none",
 "financing_cooperation": {"value": true | false, "quote": "..."},
 "no_financing_condition_statement": {"value": true | false, "quote": "..."}
```

- Present when the agreement has any provision about the buyer's financing: a commitment letter, a financing covenant, a cooperation covenant, a financing condition, or a statement that obtaining financing is not a condition.
- Absent when it says nothing about financing.
- `financing_condition.value` is true only if obtaining financing is a closing condition.
- `no_financing_condition_statement` records an express "it is not a condition that Parent obtain financing".
- `efforts_standard` is the buyer's obligation to obtain its financing: the debt financing when there is a covenant on it, otherwise the financing the buyer must obtain (v2.10 rule 4).

## Indemnification concepts (usually private deals; usually absent in public-target mergers)

### `ma.indemn.cap`: cap on indemnification

```json
 "caps": [{"indemnitor": [<party>], "amount": <Money>,
           "applies_to": ["general_reps"], "excludes": ["fundamental_reps", "fraud"], "quote": "..."}],
 "escrow_sole_recourse": {"value": true | false, "quote": "..."} | "not_stated",
 "rwi": {"text": "...", "quote": "..."} | "none"
```

- `applies_to` / `excludes` codes: `all_reps`, `general_reps`, `fundamental_reps`, `tax`, `covenants`, `specific_indemnity`, `fraud`, `willful_breach`, `other`.
- Each indemnitor, and each distinct cap amount, is its own entry. Example: general reps capped at 10% and fundamental reps capped at the purchase price are two entries.
- `escrow_sole_recourse` is true when the escrow (or RWI policy) is the indemnified party's only source of recovery.

### `ma.indemn.basket`: basket / deductible

```json
 "baskets": [{"indemnitor": [<party>], "type": "deductible" | "tipping" | "other",
              "amount": <Money>, "applies_to": [...], "quote": "..."}],
 "de_minimis": {"amount": <Money>, "counts_toward_basket": true | false | "not_stated", "quote": "..."} | "none"
```

- `deductible`: the indemnitor pays only losses above the threshold.
- `tipping` (first-dollar): once the threshold is reached, the indemnitor pays from the first dollar.

### `ma.indemn.survival`: survival of representations and covenants

```json
 "no_survival": {"value": true | false, "quote": "..."},
 "periods": [{"item": "general_reps" | "fundamental_reps" | "tax_reps" | "all_reps" | "covenants_pre_closing" | "covenants_post_closing" | "specific_reps" | "other",
              "period": {"kind": "duration" | "indefinite" | "statute_of_limitations_plus" | "fixed_date",
                         "duration": <Duration> | null, "offset_days": 60 | null, "date": null},
              "quote": "..."}]
```

- Present when the agreement says anything about survival.
- A public-deal clause that representations do not survive the closing (or the effective time) is present, with `no_survival.value` true and `periods` holding whatever does survive, usually post-closing covenants.

## Tax concept

### `tax.election.338h10`: Section 338(h)(10), 336(e) or 338(g) election

```json
 "election": {"code": "338h10" | "336e" | "338g" | "prohibited" | "other", "quote": "..."},
 "covered_entities": "all" | "some" | "not_stated",
 "decides": {"code": "buyer" | "seller" | "joint" | "mandatory_joint", "conditions": "...", "quote": "..."},
 "seller_must_join": true | false | "not_stated",
 "allocation": {"preparer": [<party>], "deadline": "...", "dispute": "accountant" | "other" | "not_stated", "quote": "..."} | "not_stated",
 "tax_cost_shift": {"payer": [<party>], "payee": [<party>], "scope": "incremental_federal" | "incremental_state_local" | "all" | "other", "quote": "..."} | "none" | "not_stated"
```

- Present only if the agreement mentions one of these elections, including a covenant that prohibits making one (`prohibited`).
- A purchase price allocation without any election is NOT this concept.

## v2.1 clarifications (from the dev split; these override anything above)

1. **Provision used but defined outside the text.** If the agreement uses the provision (for example "Company Material Adverse Effect" appears in the conditions) but its definition is in an exhibit or otherwise not in the text, the concept is `present` with the missing fields `"not_stated"` and a note. `absent` is only for an agreement that has no such provision at all.
2. **Silent fields.** A field the text does not address is the string `"not_stated"`, with no quote. Use `{"value": false, "quote": ...}` only when the text says so expressly.
3. **Financing.** A representation that the buyer has, or will have, sufficient funds is enough to make `ma.financing.structure` present. In that case `committed_financing` is `cash_on_hand`, and other fields are `not_stated` unless stated.
4. **Survival periods.**
   - Add `period.kind` values:
     - `per_terms`: survives until fully performed or "in accordance with its terms";
     - `until_closing`;
     - `none`.
   - Add `item` value `covenants` for a covenant clause that does not separate pre-closing from post-closing.
5. **Breach bar wording that combines standards.** For "the principal cause of or resulted in", the weaker disjunct controls, so code `cause_or_result`. "For any reason other than a breach" is `cause_or_result`.
6. **Tail `requires`:**
   - `definitive_agreement` when the fee is payable on signing a definitive agreement;
   - `consummation` when it is payable only on consummation;
   - `either` when signing or consummation triggers payment.
7. **Breach-based fee triggers.** Code by what the fee requires:
   - a fee payable on termination for a no-shop breach, with no tail: `breach_of_no_shop`;
   - a fee that also requires a tail: `breach_with_tail`, with the tail filled in.
8. **Redacted amounts** (`**`, `[ ]`, `[●]`) are `"not_stated"`, with the note "redacted". A bracketed placeholder period (for example "[six] months") is recorded as written, with a note.
9. **Currency.** Use the currency the agreement names. A bare "$" is USD unless the agreement defines dollars otherwise.
10. **What a reverse fee is not.**
    - A loan, a note, an "additional payment" for a specific covenant breach, or reimbursement of named one-off costs is not a reverse fee, and not expense reimbursement. Put it in `notes` on the nearest concept.
    - Expense reimbursement is only reimbursement of a party's transaction expenses because the agreement terminated.
11. **Merger of equals or two-way exchange.**
    - `target_side` is the party whose shares or assets are converted or acquired.
    - `buyer_side` is the issuer or surviving parent.
    - If that still does not decide it, use `both` and explain in notes.
12. **Reverse merger.** When a public company issuing shares pays a fee on triggers that mirror the private target's, it is a reverse fee of kind `reciprocal`. If no target fee mirrors it, the kind is `general`.
13. **Outside-date extension exercise.**
    - "Shall be extended" with no one choosing is `automatic`.
    - "May be extended by Parent", or "shall be extended at Parent's election", is `buyer`.
14. **MAE standard missing.** If the definition has no "would reasonably be expected to" language, `standard` is `"not_stated"`.

## v2.2 clarifications (from the M2 test adjudication; these override anything above)

1. **Remedy and general clauses.** A general clause that does not mention the fee ("all remedies are cumulative", "no further liability except as provided", a general specific-performance clause) does not set the fee's `remedy`. Only language in or expressly about the fee clause sets `exclusive` or `non_exclusive`; otherwise `not_stated`, with any fee-specific carve-outs.
2. **Prohibited elections.** When the agreement prohibits a 338 or 336(e) election, `election` is `prohibited` and `decides` is `{"code": "not_applicable"}`.
3. **Reverse fee `kind`.** Add `mixed`: one fee payable on triggers of different kinds (for example financing failure, regulatory failure and buyer breach). Add `deposit`: a buyer deposit released to the seller as liquidated damages on termination. Use `financing` or `regulatory` only when every trigger is of that kind.
4. **Breach bar is a list.** `breach_bar` is a list of `{conduct, causation, quote}`, one entry per bar. Code "proximate cause", "proximately caused" or "proximately contributed to" as `cause_or_result`. "Shall have been the reason" is `cause_or_result`.
5. **`none` versus `not_stated`.** `none` means the provision is in the text and has no such element (an outside-date right with no breach bar; a basket with no de minimis). `not_stated` means the element exists or is referred to but its terms are not in the text, or the text does not address it.
6. **Survival.** `none` and `until_closing` are merged into `none`. When `no_survival` is true, do not also list the same reps in `periods`; list only items that do survive (for example fundamental reps or covenants).
7. **Not a fee.** A "ticking" payment owed while the deal is pending, not on termination, is not a termination fee; note it on the nearest concept.
8. **Fixed fee plus expenses.** A fee of a fixed amount "plus" the payee's expenses (capped or not): the fixed amount goes in the fee's `amounts`, the expenses are an `ma.termination.expense_reimbursement` instance with its cap, and the fee's `expenses_credited_against_fee` is `false` unless the text credits them.
9. **De-SPAC and stock deals.** In a de-SPAC, the trust account (with any minimum-cash condition) is `committed_financing` `cash_on_hand`, and a PIPE is `equity`. A minimum-cash or minimum-offering condition in a stock-for-stock deal with no buyer cash obligation does not make financing present.
10. **Defined terms outside the text.** A fee trigger keyed to a defined term whose definition is not in the text (for example "Triggering Event") is `other`, with the term in `text`.
11. **Plain breach triggers.** Add trigger code `breach_no_tail`: a fee or reimbursement payable on termination for the payor's breach (other than of the no-shop), with no tail.
12. **Recommendation failure.** When the agreement's definition of an adverse recommendation change includes failing to make or include the recommendation, emit both `adverse_recommendation_change` and `recommendation_failure`.
13. **Exhibit sections.** Sections whose marker reads `| EXHIBIT |` are defined-terms exhibits. They are part of the text: quote from them like any other section.

## v2.3 clarifications (from the M2 holdout adjudication; these override anything above)

1. **Resolved definitions.** A candidate section whose marker lists `groups=resolved` holds only the defining paragraphs of terms that other candidate sections use. Code a trigger, cap or carve-out from that definition when it is anywhere in the text. v2.2 rule 10 (`other`) applies only when the definition is not in the text at all.
2. **Cap equal to the price.** A cap of "the Purchase Price", "the Merger Consideration" or the consideration a seller actually received (per seller) is `basis: "pct_of_purchase_price"` with `pct: 100`. `amount` is the dollar figure only if the text states the price as a number; otherwise `"not_stated"`.
3. **Cap equal to a defined amount.** A cap of a defined escrow, holdback or retention amount, or of a defined enterprise value, takes the amount and basis from that definition (for example `fixed` $35,000,000, or `pct_of_purchase_price` 10). Quote the definition with role `definition`.
4. **Several sources for one cap.** When a claim type may be recovered from more than one escrow (for example a special escrow plus the general escrow), the cap is their sum, `basis: "formula"`, with the components in `notes`.
5. **Interest on termination.** A payment owed by the buyer because the agreement terminated, even one computed as interest on the price, is a reverse fee coded by its trigger. v2.2 rule 7 (ticking payment, not a fee) applies only to a payment owed while the deal is pending.
6. **Share purchases are not fees.** A buyer's obligation, on termination, to buy the target's shares or make an equity investment is not a reverse fee. Record it in `notes` on `ma.termination.fee.reverse`.
7. **Recommendation-change notices.** Add trigger code `recommendation_change_notice`: termination after the target delivers notice that it intends to make a recommendation change or to terminate for a superior proposal. It is separate from `adverse_recommendation_change`.
8. **Proposal pending, no tail.** A fee or reimbursement payable on an outside-date, failed-vote or breach termination while an acquisition proposal is pending, with no later deal required, uses the matching `*_with_tail` code with `"tail": {"months": null, "threshold_pct": <if stated>, "requires": "none"}`.
9. **Breach-based reverse-fee triggers.** Use `buyer_breach` for any buyer breach. `breach_no_tail` is for target-side fees.
10. **MAE definitions by defined term.** One `definitions` entry per defined term. When one generic definition serves two defined terms (for example "Company Material Adverse Effect" and "Parent Material Adverse Effect" defined together), emit an entry for each, with its `party_side`. `both` is only for a single defined term that applies to either party.
11. **MAE carve-out placement.**
    - "General economic, political or market conditions", including those "in the industries in which the Company operates", is `general_economic`. Add `industry` only when the clause separately carves out changes affecting the industry (for example "conditions generally affecting the industry").
    - "The announcement, pendency, execution or performance of this Agreement" or "consummation of the transactions" is `announcement_or_pendency`.
    - Actions or omissions required by the agreement are `compliance_with_agreement`. Actions taken at the buyer's request or with its consent are `buyer_action_or_request`.
    - "Act of God", "calamity", "catastrophe", "natural disaster", "weather" and "earthquake" are `natural_disaster`. "Crisis" in a list with war, terrorism or hostilities is `political_war_terrorism`.
12. **Breach bar scope.** Plain "contributed to" is `cause_or_result`. A bar on terminating while a specific-performance action is pending is not a breach bar; record it in `notes`.
13. **Partial non-survival.** When the general representations do not survive the closing but fundamental representations (or some others) do, `no_survival.value` is true and `periods` lists what survives.
14. **Financing conditions.** `financing_condition` concerns the buyer's obligation to close. A condition to the target's obligation (for example that the buyer has the funds or that the trust holds a minimum amount) does not make it true. With no express statement, it stays `"not_stated"` (v2.1 rule 2).

## v2.4 clarifications (rulings before the M3 holdout; these override anything above)

1. **Refund of another bidder's fee.** A buyer's obligation, on termination, to pay or refund a termination fee the target owed a prior bidder (as liquidated damages or otherwise) is not a reverse fee. Record it in `notes` on `ma.termination.fee.reverse` (v2.1 rule 10).
2. **Indemnification outside the text.** When the agreement says indemnification, or its limits, are governed by an exhibit, annex or separate agreement that is not in the text, `ma.indemn.cap` and `ma.indemn.basket` are `absent`, with a note naming where the text says the terms are. This overrides v2.1 rule 1 for these two concepts.
3. **R&W insurance only.** When the buyer's only recourse for breaches of representations is an R&W insurance policy and the sellers give no indemnity, `ma.indemn.cap` is `absent`, with a note. A seller indemnity capped at the retention, with the policy above it, is a cap as usual.
4. **Price stated as a sum of parts.** For a cap equal to the price (v2.3 rule 2), `amount` is the dollar figure only when the agreement states the total as one number. A price stated only as components (a base amount plus or minus adjustments, or cash plus shares) gives `amount: "not_stated"`, with the components in `notes`.

## v2.5 clarifications (from the M3 holdout adjudication; these override anything above)

1. **Committed financing.** List a source in `committed_financing` only when the text says it is committed or in hand: a commitment or financing letter, a subscription or PIPE agreement, cash on hand, a trust account, or a funds representation. A planned or possible capital raise, offering or syndication with no commitment is not committed financing; record it in `notes`. A commitment from a person defined as a lender, even an individual, is `debt`.
2. **Funds representations.** "Has available the funds necessary", "adequate resources", "access to sufficient resources" and "financing sufficient to" are funds representations (v2.1 rule 3).
3. **Lenders' consent.** A condition to the buyer's obligation that its lenders consent to the transaction is a financing condition: `financing_condition.value` is true.
4. **Reverse-fee triggers.**
   - A bankruptcy-sale or auction termination because the target accepts another bid is `superior_proposal_termination`.
   - A fee payable on termination at the outside date when the buyer has not closed is `failure_to_close`, unless only regulatory conditions are open (`regulatory_failure`).
   - The same liquidated-damages amount payable by either party for its breach goes in both fee instances; the reverse fee's `kind` is `reciprocal`.
5. **Cap per seller.** A cap of "the portion of the Purchase Price actually received by such Seller" (or a seller's pro rata share of the price) is `pct_of_purchase_price` 100 with `amount: "not_stated"`. v2.3 rule 2's dollar figure applies only to a cap on the whole price.
6. **Escrow as sole recourse.** `escrow_sole_recourse.value` is true when the escrow (or policy) is the only source of recovery for claims on the general representations, even if fundamental-representation or fraud claims may go further; say so in `notes`. When claims are paid "first" or "initially" from the escrow and may then go to the sellers, it is false.
7. **Basket type.** `deductible` only when the text says the indemnitor is liable only for losses in excess of the threshold. "Not liable until losses exceed" with no such words is `tipping`.
8. **Outside-date extensions.**
   - A later outside date that applies only when regulatory conditions are unmet is an extension with condition `regulatory_pending` and exercise `automatic` (or whoever may elect it).
   - An automatic extension while a specific-performance action is pending is an extension with condition `other`, text "specific performance pending". (v2.3 rule 12 still applies to a bar on terminating, which is not an extension.)
   - Mirrored rights (each party may extend to cure the other's breach) are one extension entry with exercise `either_party`.
9. **MAE carve-outs outside the definition.** Code `carveouts` from the definition only. Carve-outs that sit in a closing condition or a representation's proviso go in `notes`.
10. **MAE without a definition.** A capitalized MAE term used in a condition or representation is present (v2.1 rule 1), coded from whatever text describes it. A lowercase, undefined "material adverse effect" used only in passing is `absent`, with a note.
11. **More carve-out placement.**
    - "The identity of Parent" (or of the buyer or its affiliates) is `announcement_or_pendency`.
    - "Cyber-attack" listed with war, sabotage or terrorism is `political_war_terrorism`.
    - "Changes in regulatory conditions" is `change_in_law`.
12. **Elections needing consent.** A 338 or 336(e) election that one party may make only with the other's consent is `decides: joint`, not `prohibited`.

## v2.6 clarifications (from the financing-retrieval round's adjudication; these override anything above)

1. **Stated amounts only.** `amount` holds a dollar figure only when the text states that figure. For a fee, cap, basket or de minimis stated as a percentage of a price, record `basis` and `pct` and leave `amount: "not_stated"`, even when the price is stated as one number: do not compute the dollar figure. A cap equal to a price the text states as one number keeps that number (v2.3 rule 2), and a figure stated next to its percentage (for example "$34,000,000 (ten percent (10%) of the Purchase Price)") is kept.
2. **More funds representations.** These are also funds representations (v2.5 rule 2, `cash_on_hand`): "access to ... the financial resources to consummate", "available funds, or immediately available capacity under committed credit facilities", and a buyer covenant "to maintain sufficient liquid accounts or borrowing capacity".

## v2.7 clarifications (the remaining financing-retrieval round questions; these override anything above)

1. **Words against numerals.** When a number in words and its numerals disagree (for example "three percent (6.0%)"), the words control, unless the agreement says otherwise. Record the words' value and quote the conflict in `notes`.
2. **Commitments named in a funds representation.** A funds representation is `cash_on_hand` (v2.1 rule 3) even when it mentions commitments (for example "sufficient cash, or commitments from the Guarantors"). Add `equity` or `debt` only when the text also contains or names the commitment itself: a commitment or financing letter, a subscription agreement or a guarantee of the funding.
3. **Cumulative-remedies clause scoped to a section.** A clause making the remedies of a section or article cumulative is about the fee when the fee is in that section or article and the clause does not except it. The fee's `remedy` is then `non_exclusive`. A general boilerplate clause for the whole agreement still is not (v2.2 rule 1).
4. **De minimis that differs by indemnitor.** `de_minimis` holds the threshold for claims against the target side (the sellers or target equityholders). Record the other side's threshold, and any higher threshold for particular representations, in `notes`.
5. **Reverse-fee kind with a regulatory prerequisite.** Code `kind` from the triggers, not from conditions that every route shares. When one route is a buyer breach and another a regulatory failure, the kind is `mixed` (v2.2 rule 3), even if every route also requires a regulatory condition to be unmet.

## v2.8 clarifications (from the M5 holdout adjudication; these override anything above)

1. **Parent guarantees.** A parent's or affiliate's guarantee of the buyer's payment obligations is committed financing with code `other`, the guarantor in `sources`. It does not make a funds representation `equity`.
2. **Equity rollover.** Shares that target holders contribute to the buyer instead of receiving cash are `other`, with "rollover" in `notes`. Cash equity from a sponsor or investor stays `equity`.
3. **Recovery by set-off only.** `escrow_sole_recourse.value` is true when the indemnified party's only recovery for general-representation claims is set-off against held-back, deferred or seller-note payments, as for an escrow or holdback (v2.3 rule 3).
4. **Caps from escrows and holdbacks.** A cap equal to an escrow or holdback amount that is defined as a percentage of the price (for example "10% of the cash consideration plus 10% of the Parent shares") is `pct_of_purchase_price` with that percentage and `amount: "not_stated"`. Read the definition, which stage 1 now supplies (role `definition`).
5. **One cap per route.** Separate limits for different claims are separate entries, even in one sentence: for example 50% of the R&W retention for claims the policy covers and the escrow for claims it does not, or a per-seller limit beside an aggregate one.
6. **Price actually received.** A cap of the payments a seller actually receives (for example the closing payment plus any contingent payment received) is `pct_of_purchase_price` 100 with `amount: "not_stated"` (v2.3 rule 2, v2.5 rule 5).
7. **Formula caps.** A cap of the price less a part of it (for example "Total Consideration less" one class's share), or the sum of an escrow and an earn-out set-off right, is `basis: "formula"` with `pct: null`, the components in `notes`.
8. **Holdback first.** When claims are paid first from a holdback or escrow and then from the sellers or by set-off against an earn-out, `escrow_sole_recourse.value` is false (v2.5 rule 6). An escrow that only "partially supports" the indemnity is false too.

## v2.9 clarifications (from the M7 holdout adjudication; these override anything above)

1. **Breach bars from a general clause.** A general clause that bars a party from relying on a condition failure its own breach caused counts as the outside-date right's breach bar (`cause_or_result`) only when it says it applies to terminating ("as a basis for terminating"). A clause that speaks only of conditions is not a breach bar.
2. **Two outside dates.** When each party has its own outside date and termination right, `initial` is the earlier date; record the other in `notes`.
3. **Prospects.** `prospects` is true only for "prospects" or an equally forward-looking prong (future results, expected or projected performance). A time qualifier on financial condition ("near-term or long-term financial condition") is not prospects.
4. **Fee bases.** A fee stated as a percentage of the merger consideration plus the value of options or other equity awards is `pct_of_equity_value`. A percentage of the merger consideration alone is `pct_of_purchase_price`.
5. **Escrow as collateral.** `escrow_sole_recourse.value` is false when the seller is directly liable beyond the escrow, including when the escrow is only "collateral" or security for claims against the seller, even without an express statement (this narrows v2.1 rule 2).
6. **"Legal conditions."** A general-conditions carve-out that lists "legal" conditions among economic, market and political ones does not add `change_in_law`. Only a change in law, regulation or its interpretation (or "regulatory conditions", v2.5 rule 11) does.
7. **Interim-covenant carve-outs.** A carve-out for failing to act because the interim operating covenants forbid it, including when the other party refused a requested consent or waiver, is `compliance_with_agreement`, in every definition it appears in.
8. **Mixed commitments.** One commitment to buy both debt and equity securities (for example senior subordinated notes and common stock) is two entries, `debt` and `equity`, each naming that source. It is not `other`.
9. **Affiliate covenants to fund.** A parent's or affiliate's covenant to provide the cash the buyer pays at closing is `other`, like a guarantee (v2.8 rule 1), not `cash_on_hand`. A funds representation stays `cash_on_hand`.
10. **Remedies in an outside document.** A buyer-side remedy fixed only by a document whose terms are not in the text ("the remedy as agreed to in the Contingency Letter Agreement") does not make `ma.termination.fee.reverse` present unless the text says a payment is due. Record it in `notes`.
11. **Cost of a prohibited election.** When a 338 or 336(e) election is prohibited, `tax_cost_shift` is `"none"`, unless the agreement assigns the taxes from elections the buyer may still make (for example a 338(g) election after closing), in which case code that payer. An indemnity for breaching the prohibition is a breach remedy, not a cost shift; put it in `notes`.

## v2.10 clarifications (from the M7 run-to-run comparison; these override anything above)

1. **Terminations while a fee trigger was available.** A fee owed when a party terminates on another ground (the outside date, a failed vote, an expired offer) at a time the payee "had the right to terminate", "could have terminated" or "was entitled to terminate" on a fee trigger is that trigger. It adds no code when that trigger is already listed and is never `other`. Record it in `notes`.
2. **Buyer-side recommendation and meeting triggers.** v2.2 rules 11 and 12 apply to the reverse fee:
   - when the buyer board's failure to make, include or reaffirm its recommendation is a ground for the payee to terminate, emit `recommendation_failure` beside `buyer_recommendation_change`;
   - a termination for the buyer's breach of its meeting covenant ("by failing to hold the Parent Stockholders' Meeting") is `buyer_breach`.
   `buyer_approval_failure` is only the buyer's stockholders failing to approve.
3. **Set-off without an escrow.** With no escrow or holdback, `escrow_sole_recourse` is `"not_stated"` unless the text says set-off does not limit the buyer's other remedies ("shall not constitute an election of remedies"), which makes it false. A set-off that is the only recovery is still true (v2.8 rule 3).
4. **Which efforts standard.** `efforts_standard` is the buyer's covenant on the debt financing when there is one. Otherwise it is the covenant on the financing the buyer must obtain: an equity commitment, or in a de-SPAC the PIPE or other transaction financing (efforts to keep cash in the trust account count). It is `"not_stated"` when the buyer has no financing covenant. When several covenants with different standards exist, use the debt one and record the others in `notes`.

## v2.11 clarifications (from the M8 holdout adjudication; these override anything above)

1. **When a 338 or 336(e) provision is present.** `tax.election.338h10` is present when the agreement sets a term for an election in this deal: it permits, requires, prohibits or conditions one, or says who bears the tax cost of one ("the tax consequences of any deemed asset sale under IRC Section 338 shall be solely for Buyer's account"). Code an election whose type the text does not name as `other`. A representation about elections made before closing, or about their effect, with no term on an election in this deal, is `absent`; note it.
2. **Undefined MAE used operatively.** This amends v2.5 rule 10. A lowercase "material adverse effect" or "material adverse change" that the agreement never defines is present when it is used operatively: in a closing condition, a representation or a termination right. Code it from the words that describe it where it is used. Only a passing use (a recital, a reference in a covenant) stays `absent`.
3. **Party sides in a de-SPAC.** The SPAC (with its merger subsidiaries and any holding company formed to own it) is `buyer_side`. The operating business is `target_side`. This holds whichever entity issues the shares or survives, and it overrides v2.1 rule 11 for de-SPACs.
4. **"Political conditions."** This amends v2.3 rule 11. A general-conditions carve-out that names political conditions ("general economic or political conditions") adds `political_war_terrorism` beside `general_economic`. This matches how gold has coded 22 definitions in M7 and M8.
5. **Force-majeure catch-alls.** "Any other force majeure event" closing a list of wars, disasters or epidemics adds no code beyond the listed ones. "Force majeure" standing alone is `natural_disaster`.
6. **Transaction litigation.** A proceeding arising from the agreement or the transactions is `stockholder_litigation` whoever brings it, in a private deal as in a public one.
7. **`joint` and `mandatory_joint`.**
   - `mandatory_joint`: the agreement obliges the parties to make or join the election, or to take steps to secure the treatment it gives ("shall make a joint election", "shall take reasonable steps to ensure such treatment").
   - `joint`: the election may be made only by the parties together or with the other's consent, and nothing obliges them to make it (v2.5 rule 12).
   - `buyer` or `seller`: one party may make it alone.
8. **Breach triggers that cite covenants by number.** When a trigger names the breached covenants only by section number ("breached its obligations under Section 6.3 or 6.11"), code it by what each cited section is. The stockholder-meeting or approval covenant is `breach_no_tail` (`buyer_breach` on a reverse fee), and the no-shop is `breach_of_no_shop`. Emit both when both are cited. Never use `other`. A post-check resolves the numbers from the section titles.
9. **A tender offer lapsing.** A tail trigger on a termination clause that covers the offer expiring or being terminated without the buyer accepting any shares is `approval_failure_with_tail` (the definition above). When the same clause also covers the outside date, emit `outside_date_with_tail` as well. A post-check reads the cited clause.
10. **A fixed "reimbursement" on termination.** A fixed sum payable on termination that the text does not tie to named costs or to the payee's expenses is a fee amount, even when the clause says "reimburse" ("Target shall reimburse Parent $25.3 million"). A refund of a named one-off payment, such as a fee the payee paid a prior bidder, is neither a fee nor expense reimbursement (v2.1 rule 10).

## v2.12 clarifications (from the M9 holdout adjudication; these override anything above)

1. **Funds representations that combine sources.** A funds representation that counts the buyer's own funds beside its commitments adds `cash_on_hand` next to `debt` or `equity` (v2.7 rule 2). Examples: "when combined with other funds available to Parent", "together with any available cash-on-hand", "at the Closing the Buyer will have sufficient funds", "after giving effect to the Equity Financing, Parent will have sufficient financial resources". A covenant not to reduce the financing below what is sufficient is not a funds representation, nor is one about another entity's funds ("Parent shall ensure that the Surviving Corporation has sufficient funds"). A post-check adds the code.
2. **Survival periods that are the earlier or later of two measures.** Set `kind` to the first measure the text names. Record the other measure in the same period: its `duration`, `offset_days` or `date`. Say "earlier of" or "later of" in `notes`.
   - "Until the earlier of (A) the statute of limitations and (B) the six-year anniversary of the Closing Date" is `statute_of_limitations_plus` with `duration` 6 years.
   - "The later of four months after the Closing Date and 31 March 2012" is `duration` 4 months with `date` "2012-03-31".
3. **An undefined MAE beside a defined one.** An undefined lowercase "material adverse effect" used operatively for a party that has no defined MAE term of its own adds a `definitions` entry for that party (v2.11 rule 2). Use its `party_side` and the words as written for `defined_term`. A typical case is a buyer bring-down condition on the buyer's ability to close, when only the target's MAE is defined.
4. **One extension measured by several delays.** An extension the text grants once is one `extensions` entry, even when its length is measured by more than one delay ("the Extension", the greater of the SEC-review and HSR delays). When the delays are of different kinds, `condition` is `other` and its `text` names each.
