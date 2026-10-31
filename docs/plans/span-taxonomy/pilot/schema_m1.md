# M1 pilot extraction schema

Three concepts, one instance per agreement per concept. Output one JSON file per agreement:

```json
{
  "agreement": "<uuid>",
  "instances": [ <instance>, <instance>, <instance> ]
}
```

Always emit all three concepts. If the agreement has no such provision, emit the instance with `"finding": "absent"` and list the sections you checked in `scope_reviewed`.

## Common rules

- **Quotes are evidence.** Every `quote` must be copied character for character from ONE line of the text you were given (never a `@@` marker line, never joined across lines). Keep each quote short: the shortest phrase that proves the value, usually 5–40 words. Use several quotes if the evidence is on several lines.
- **Spans** (`spans[]`) are the passages that make up the provision. Each span is `{"role": ..., "section": "S<n>", "quote": "..."}`. Here the quote is the first 10–20 words of the passage (enough to locate it uniquely), plus `"quote_end"` with its last 5–15 words when the passage is longer than one line. Roles: `operative` (the clause that imposes the fee or sets the date), `trigger` (a termination right or condition the fee keys off), `amount` (where the amount is stated, if separate), `definition` (a defined term that supplies the value, e.g. "Termination Fee" means…), `extension`, `remedy`, `other`.
- **Absent versus not stated.** `finding: "absent"` means the agreement has no such provision. Inside a present instance, a field the agreement is silent on is `"not_stated"`. Never guess.
- **Coded fields** are `{"code": ..., "text": "<verbatim phrase>", "quote": "..."}`. Use a listed code when it fits. Otherwise use `"other"` with `text`, or `"proposed:<snake_case>"` if you think a new code is warranted.
- Body only. If the text says a value is in an exhibit or schedule that is not in the text, record `"not_stated"` and say so in `notes`.

## `ma.termination.fee.target`: termination fee payable by the target (or the seller side)

```json
{
  "concept": "ma.termination.fee.target",
  "finding": "present" | "absent",
  "scope_reviewed": ["S41", "S42"],
  "spans": [...],
  "payor": {"roles": ["target"], "defined_terms": ["the Company"], "liability": "single" | "joint_and_several" | "several", "quote": "..."},
  "payee": {"roles": ["buyer"], "defined_terms": ["Parent"], "quote": "..."},
  "amounts": [{"amount": 25000000, "currency": "USD", "basis": "fixed", "pct": null,
               "label": "standard", "quote": "..."}],
  "triggers": [{"code": "superior_proposal_termination", "text": "...", "quote": "...",
                "tail": null | {"months": 12, "threshold_pct": 50, "requires": "definitive_agreement" | "consummation" | "either" | "not_stated"}}],
  "remedy": {"code": "exclusive" | "election_fee_or_damages" | "non_exclusive" | "not_stated", "carveouts": ["fraud", "willful_breach"], "quote": "..."},
  "payment_timing": {"text": "within two business days after termination", "quote": "..."} | "not_stated",
  "pay_once_only": true | false | "not_stated",
  "expenses_credited_against_fee": true | false | "not_stated",
  "notes": ""
}
```

- PartyRole codes: `target`, `buyer`, `parent`, `merger_sub`, `seller`, `sponsor`, `guarantor`, `other`. In an asset or stock purchase, the seller side pays a "target" fee.
- `basis` codes: `fixed`, `pct_of_equity_value`, `pct_of_enterprise_value`, `pct_of_purchase_price`, `pct_of_market_cap`, `formula`, `other`.
- Amount `label` codes: `standard`, `go_shop` (reduced fee during a go-shop or window-shop period), `superior_proposal`, `alternative` (another amount that applies in another situation, not a reduced tier), `other`.
- Trigger codes:
  - `superior_proposal_termination`: the target terminates to accept a superior proposal;
  - `adverse_recommendation_change`: the buyer terminates after a change of board recommendation;
  - `recommendation_failure`: failure to reaffirm, to include the recommendation in the proxy, or to oppose a tender offer;
  - `breach_of_no_shop`: a willful or material breach of the non-solicitation covenant;
  - `vote_failure_with_tail`: the stockholder vote fails, plus an acquisition proposal and a tail;
  - `outside_date_with_tail`: termination at the outside date, plus an acquisition proposal and a tail;
  - `breach_with_tail`: termination for breach, plus an acquisition proposal and a tail;
  - `vote_failure_naked`: a fee on a failed vote alone, without a tail;
  - `other`.
- `tail.threshold_pct`: the percentage that "Acquisition Proposal" is deemed to use for tail purposes (often 50), as a number.

## `ma.termination.fee.reverse`: termination fee payable by the buyer side

Same fields as the target fee, plus:

```json
  "kind": {"code": "financing" | "regulatory" | "general" | "two_tier" | "other", "text": "...", "quote": "..."},
  "guarantee": {"code": "limited_guarantee" | "equity_commitment" | "parent_guarantee" | "none" | "not_stated", "quote": "..."},
  "liability_cap": {"text": "the Parent Termination Fee plus reimbursement obligations", "quote": "..."} | "none" | "not_stated",
  "specific_performance_interaction": {"code": "exclusive_alternative" | "both_available" | "conditional" | "not_stated", "quote": "..."}
```

Extra trigger codes for the reverse fee: `failure_to_close` (the buyer fails to close when the conditions are satisfied), `regulatory_failure` (antitrust or regulatory approval not obtained, an injunction under antitrust law, or the outside date with only regulatory conditions open), `financing_failure`, `buyer_breach`, `buyer_vote_failure`.

A plain expense reimbursement is NOT a reverse fee; it belongs to another concept and is out of scope here. Mention it in `notes` if you see one.

## `ma.termination.outside_date`: outside date / end date

```json
{
  "concept": "ma.termination.outside_date",
  "finding": "present" | "absent",
  "scope_reviewed": [...],
  "spans": [...],
  "defined_term": "End Date" | null,
  "initial": {"date": "2016-12-31" | null, "duration": null | {"value": 9, "unit": "months", "anchor": "signing"}, "quote": "..."},
  "effect": {"code": "termination_right" | "automatic_termination", "quote": "..."},
  "extensions": [{"length": {"value": 3, "unit": "months"} | null, "new_date": "2017-03-31" | null,
                  "condition": {"code": "regulatory_pending" | "marketing_period" | "sec_review" | "notice_period" | "other", "text": "...", "quote": "..."},
                  "exercise": "automatic" | "either_party" | "buyer" | "target" | "mutual" | "not_stated",
                  "count": 1, "quote": "..."}],
  "breach_bar": {"code": "none" | "principal_cause" | "proximate_cause" | "contributed" | "breach_only" | "other", "text": "...", "quote": "..."} | "not_stated",
  "notes": ""
}
```

- Duration `unit` codes: `days`, `business_days`, `months`, `years`. Anchor codes: `signing`, `other`.
- `breach_bar`: the standard by which a party in breach loses the right to terminate at the outside date. Use `principal_cause` for "the principal cause of or resulted in", `proximate_cause` for "proximately caused", `contributed` for "contributed to", `breach_only` when any breach bars the right regardless of causation, and `none` when there is no bar.
