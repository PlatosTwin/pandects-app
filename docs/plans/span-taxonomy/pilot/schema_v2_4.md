# Extraction schema v2.4 (M3 holdout): ten concepts

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
- `efforts_standard` is the buyer's obligation to obtain the debt financing.

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
