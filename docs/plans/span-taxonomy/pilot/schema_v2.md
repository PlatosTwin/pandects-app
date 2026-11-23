# Extraction schema v2 (M2): ten concepts

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
