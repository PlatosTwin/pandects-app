# Full taxonomy for M&A and tax provisions: design

Status: draft for the maintainer's review, 2026-11-22. The M&A pack was trimmed the same day, after the maintainer's first review. An independent review of the trim on 2026-11-23 split five concepts back out (section 4.4). An adversarial review of the whole PR on the same day led to the fixes in section 11.3. A labeling trial on 50 agreements, also on 2026-11-23, led to the changes in section 11.4. Nothing in this document is live. No paid model call was made for this work, and no row in any database was changed.

This document belongs to the span taxonomy plan (draft PR 317, https://github.com/PlatosTwin/pandects-app/pull/317, not yet on main). That plan tested ten value concepts in depth. This document gives the breadth step: the full list of provisions, for both subject areas, before any paid labeling run.

## 1. Summary

- **What exists now.** Two sets of taxonomy files, called packs, are in `etl/taxonomy/`. The M&A pack (`ma`) has 15 branches, 54 groups and 272 concepts. The tax pack (`tax`) has 10 branches, 35 groups and 115 concepts. A validator and 80 tests check them without network or database access.
- **A concept is one thing a researcher filters on.** `ma.governing_law` answers "which law governs this agreement". It does not also hold the forum clause or the jury waiver. Each of those has its own concept.
- **Groups are the default filter level.** A researcher browses and filters by the 54 M&A groups first. A label on a concept also counts for its group and its branch. A filter on one concept is the finer choice.
- **Two labeling waves.** Each M&A concept states its wave. Wave 1 has the 199 section-level topics and catch-alls. Wave 2 has the 73 clause-level provisions, parts and definitions. Section 4.4 gives the rule. In wave 1, a section that holds only wave 2 concepts gets its group as the label.
- **The tax tree keeps its structure.** All 150 nodes of the tax clause tree are present with the same labels and the same parents. This design has 10 additions and no removals, renames or merges. Section 8 lists each addition with its reason. The maintainer accepted 7 on 2026-11-22 and 3 on 2026-11-23.
- **Party is an attribute.** "Organization" as a representation is one concept. Each labeled instance records who made it. The older trees held a target copy and a buyer copy of each representation.
- **Coverage by section title.** Of 1,138,095 titled sections in the corpus, 85.7% map to one specific concept and a further 8.0% map to a group that holds several concepts by design. 4.2% map to nothing. A sample of those shows that most are other wordings of provisions the tree already has, or headings that say too little to place.
- **Old filters keep working.** Each of the 99 leaves of the live tree, each of the 140 leaves of the unreleased redesign and each of the 150 tax nodes maps to at least one new node.
- **Labels must come from body text.** 153 of the 387 concepts never appear as a section title. Section 9 describes the labeling run and gives a cost estimate: roughly $350 to $800 for a first full pass, with a small model for the first read and a stronger model for the hard cases. That figure is an estimate from list prices. A billed trial on 50 agreements must replace it before any full run.

## 2. Files

| Path | Contents |
|---|---|
| `etl/taxonomy/ma/*.yaml` | The M&A pack, one file per branch (15 files). |
| `etl/taxonomy/tax/*.yaml` | The tax pack, one file per branch (10 files). |
| `etl/taxonomy/legacy/*.yaml` | Read-only snapshots of the ids of the three older trees. The validator checks crosswalks against them. |
| `etl/taxonomy/README.md` | Field reference and the rules for changes. |
| `etl/src/etl/domain/taxonomy_pack.py` | Loader and validator. Run `etl/.venv/bin/python -m etl.domain.taxonomy_pack`. |
| `etl/tests/test_taxonomy_pack.py` | 80 tests. Run from `etl/` with `.venv/bin/python -m unittest tests.test_taxonomy_pack`. |
| `docs/plans/full-taxonomy/tree.md` | Every node, generated from the packs. |
| `docs/plans/full-taxonomy/crosswalk.md` | Every older leaf and the nodes it maps to, generated from the packs. |
| `docs/plans/full-taxonomy/title_coverage.py` | The coverage measurement in section 6. |
| `docs/plans/full-taxonomy/prevalence.py` | The prevalence measurement in section 4.4. It reads two read-only exports from the database. |
| `docs/plans/full-taxonomy/prevalence_patterns.tsv` | The 159 text patterns that `prevalence.py` uses. |
| `docs/plans/full-taxonomy/render_appendix.py` | Writes `tree.md` and `crosswalk.md`. |

## 3. Design

### 3.1 Two layers

The packs separate two things that the older trees mixed.

- A **navigation node** is a place in the browse tree. There are two levels: a branch ("Termination") and a group ("Termination Fees and Expenses"). Navigation ids start with `ma.nav.` or `tax.nav.`.
- A **concept** is a thing that a label points to. Each concept names the one navigation node that holds it (`nav_parent`).

The id of a concept never states its place in the tree. `ma.governing_law` sits under "Dispute Resolution and Remedies" today. If a later release moves it, the id and every stored label stay the same.

A concept can also appear under other navigation nodes (`also_listed_under`). The Material Adverse Effect definition sits under "Definitions" and is also listed under "Conditions to Closing", where a researcher will look for it.

### 3.2 Four kinds of concept

| Kind | Meaning | M&A | Tax |
|---|---|---|---|
| `provision` | A provision that answers one research question. | 232 | 112 |
| `component` | A part of a provision that researchers study by itself. It names its parent concept. | 21 | 0 |
| `definition` | A defined term whose content is a research object. | 16 | 3 |
| `modifier` | A clause that changes how another concept applies. It names its parent concept. | 3 | 0 |

Examples: `ma.conduct.negative_covenants` is a provision, and `ma.conduct.restriction.dividends` is one of its 10 components. `ma.mae.definition` is a definition, and `ma.mae.disproportionate_effect` is a modifier of it.

### 3.3 Ids

- Lowercase words joined by dots, with the pack as the first word.
- A family of related concepts shares a prefix: `ma.termination.fee.target`, `ma.termination.fee.reverse`.
- A provision that stands alone has a two-word id: `ma.governing_law`, `ma.notices`, `ma.expenses`.
- An id is never renamed or reused after release. A retired node stays in its file with `deprecated_in` and `replaced_by`.
- The 11 concept ids of the value pilot are unchanged, and so are its 3 sibling ids. A test checks this.

### 3.4 Party is an attribute of the instance

The live tree has "Organization (Company rep)" and "Organization (Buyer rep)" as two leaves. This design has one concept, `ma.rep.organization`, with `attributes: {party: maker}`. The labeled instance stores who the maker is.

The role names are `maker` (of a representation), `obligor` (of a covenant), `tested_party` (of a condition), `holder` (of a right), `indemnitor` and `subject` (of a definition). 104 M&A concepts and 35 tax concepts carry a role.

The reasons:

1. A researcher who wants "all Organization representations" gets them with one filter. A researcher who wants only the target's adds one attribute filter.
2. Private deals have more than two sides: sellers, the company, the buyer, a merger subsidiary, a sponsor. A copy of the subtree per party does not scale.
3. Separate copies of the same provision for each party are easy to confuse. With one concept, the model picks the topic, and a separate check with simple rules sets the party from the article heading.

### 3.5 Catch-all concepts

Each large family has an explicit catch-all: `ma.rep.other`, `ma.covenant.other`, `ma.condition.other`, `ma.termination.right.other`, `ma.misc.other`, `ma.other.reserved` (for "[Reserved]" and "Intentionally Omitted") and `ma.other.unclassified`. A catch-all makes the unassigned share measurable. When a topic grows inside a catch-all, a later release can give it its own concept. `ma.other.lead_in` (added after the trial, section 11.4) is counted with them. It marks the opening words of an article, such as "The Company represents and warrants as follows", when they stand as a section of their own and state no term.

The tax pack follows the present tax labeler instead. That labeler may assign a clause to a level 1 or level 2 node when no leaf fits. The tax navigation nodes are therefore marked `assignable: true`, and the tax pack keeps its three existing "Other" leaves.

14 of the 54 M&A groups are also marked `assignable: true` (decided 2026-11-23). The purpose differs from the tax pack. In wave 1, a section whose content is only wave 2 concepts gets its group as the label, so that group filters work from the first wave. A section titled "Termination" gets `ma.nav.termination.rights`.

- **The rule.** A group is assignable when it holds a wave 2 provision or definition. The validator checks the rule in both directions.
- **Parts do not count.** Five groups hold a wave 2 part of a wave 1 concept and no other wave 2 concept: `ma.nav.conduct.interim`, `ma.nav.deal_protection.stockholder_approval`, `ma.nav.employees.continuing`, `ma.nav.financing.covenants` and `ma.nav.indemnification.scope`. The text of a part sits inside the text of its parent, and the parent carries the label in wave 1.
- **A derived concept does not count.** `ma.financing.structure` is marked `derived: true`. It labels no text: its values are built from the spans of the other financing concepts. It is the only wave 2 provision of `ma.nav.financing.arrangements`, so that group is not assignable.
- **The other 40 groups** hold wave 1 concepts only, or the parts and the derived concept above. A section there gets its concepts in wave 1.
- **No M&A branch is assignable.** A section that holds content of two groups gets two group labels.
- **Mixed sections.** A "Termination" section normally also holds the mutual consent clause, which is a wave 1 concept. That clause gets its concept, and the group label covers the rest of the section. This reading goes one step past the words "only wave 2 concepts".
- **A group label does not replace a catch-all.** Content that fits no concept of any wave goes to the catch-all concept.
- **Wave notes in cues.** A wave 1 reader has the wave 1 concepts and the 14 groups only. A cue of a wave 1 concept that names a wave 2 concept therefore also names the wave 1 label of that concept: its parent for a part, its group otherwise. A wave 1 concept also names its own parts and says that they keep its label in wave 1. The validator checks both rules (section 11.4).

### 3.6 The seam between the packs

Some provisions belong to both subject areas. Each has one owner pack, and the other pack lists it. `tree.md` ends with the full seam table (21 rows).

- **Tax owns** the tax representations, elections, purchase price allocation, withholding, transfer taxes, the tax opinion condition, pre-closing restructuring, tax covenants before closing, the tax indemnity and the tax treatment of indemnity payments. A section titled "Taxes" in the representations article gets the tax node `tax.nav.reps`, which the M&A tree lists under "Financial" representations.
- **M&A owns** the Section 280G stockholder vote (`ma.employee.section_280g`, listed under `tax.nav.other`) and the Section 409A and plan qualification statements, which are part of `ma.rep.employee_benefits`. These are compensation provisions that employment and benefits lawyers draft. Open question 2 asks the maintainer to confirm this. M&A also owns the SPAC domestication (`ma.spac.domestication`), which the tax tree lists under "Pre-Closing Restructuring" (`tax.nav.deal.restructuring`).

### 3.7 Fields

Each concept has an id, a label, a kind, a navigation parent, a description, canonical terms, example phrases, exclusion cues, a crosswalk to older ids and the release that introduced it. `etl/taxonomy/README.md` defines each field.

Three points need a note here:

- **Derived concepts.** `derived: true` marks a concept that labels no spans. Only `ma.financing.structure` has it. A statement that closing does not depend on financing goes to `ma.rep.financing` or `ma.financing.buyer_covenant`, and the derived concept reads it from there.
- **Exclusion cues name the right concept.** "The court that hears disputes is `ma.forum_selection`." The validator checks that each id named in a cue exists.
- **Value schemas.** The 11 pilot concepts keep their tested fields (`status: piloted`) and are the only concepts with `tier: value`. A further 76 M&A concepts carry a `status: stub` schema: a first sketch of fields for a later wave, which nobody has tested.
- **Example phrases are illustrations.** They show typical wording. They are not verified quotations from the corpus.

### 3.8 Releases

Each node carries `introduced_in: '2026.1'`. The packs are a draft until the maintainer approves them. After approval, a change adds a release number, and released ids are frozen.

## 4. Granularity

### 4.1 The rules

1. **One concept, one research question.** A reader must be able to state the question in one sentence, and the answers must be comparable across agreements.
2. **Split a joined title.** The older trees have leaves such as "Entire Agreement; Severability; Counterparts; Electronic Signatures; No Third-Party Beneficiaries". Each part answers a different question and becomes its own concept.
3. **Do not split by party.** See 3.4.
4. **Do not split by deal type.** A provision gets a deal-type concept only when it exists in that deal type alone: asset deal scope, tender offer mechanics, SPAC terms.
5. **A fact inside a provision is a value field.** The size of a cap and the type of a basket are fields of `ma.indemn.cap` and `ma.indemn.basket`. They are not new concepts.
6. **A part becomes a component only when researchers treat it as a variable of its own** and the part has a text span that a reader can mark.
7. **A concept must be common enough to study.** The working floor is about 1% to 2% of agreements (120 to 250 agreements). A topic below the floor goes into a value field of its parent or into a catch-all, unless it is a known research variable.

### 4.2 How the rules were applied

| Decision | Evidence from the corpus | Result |
|---|---|---|
| Governing law, forum, jury waiver, specific performance | Each is a separate research variable. Titles alone find them in 96.0%, 54.3%, 68.6% and 72.7% of the 12,198 agreements. | Four concepts. |
| Termination rights | The live tree has one leaf for all termination events. | 9 concepts: mutual consent, outside date, legal restraint, vote failure, breach, change of recommendation, superior proposal, failure to close, and other. The notice of termination is part of `ma.termination.effect`. |
| Termination fees | One older leaf joined the target fee, the reverse fee and expenses. | 5 concepts. Three of them are pilot value concepts. |
| No-shop and fiduciary out | "Superior Proposal" appears in 37.1% of agreements and "Intervening Event" in 19.0%. | Separate concepts for the non-solicitation covenant, the go-shop, the exception for unsolicited proposals, notice of proposals, and each ground for a change of recommendation. |
| Negative covenants before closing | The restricted actions (dividends, new debt, capital spending and others) are studied one by one. | One provision with 10 components. |
| Defined terms | The content of a few definitions drives the deal terms. | 16 definition concepts, such as Knowledge, Willful Breach, Fraud, Indebtedness and Working Capital. All other defined terms fall under `ma.definitions.general`. |
| Representations | 49 topics, each with a party role. | 49 concepts in place of two copies of each. |
| Collar and walk-away right | The title "collar" occurs once. "Walk away" appears in 0.6% of agreements. | Value fields of `ma.consideration.exchange_ratio`. |
| Debt and equity commitments | The agreement describes them inside the buyer's financing representation. | Two components of `ma.rep.financing`. |
| Release of seller guarantees | 601 sections in 536 agreements carry it as a title. | Its own concept, `ma.credit_support_release`. |
| Locked box | About 1% of agreements, in UK-style deals, and the term collides with "lockbox" bank accounts. | No concept. US agreements are the focus. |
| SPAC deals | "Trust Account" appears in 9.8% of agreements, a minimum cash term in 3.5%, a PIPE in 3.0%. | 7 concepts in their own group: trust account, redemption, PIPE, sponsor support, minimum cash, extension of the deadline and domestication. |
| Asset deals | Asset purchases hold about 11% of titled sections. | 8 concepts for purchased and excluded assets, assumed and excluded liabilities and related mechanics. |
| Industry representations | Banking, healthcare, insurance, energy and communications deals carry long representations with no general equivalent. | 5 industry concepts and `ma.rep.other`. The communications concept was added after the trial (11.4). See open question 4. |

Four concepts were added because the title data showed a gap in the first draft: `ma.rule_145_affiliates` (titles in 981 agreements), `ma.asset.title_and_survey` (594), `ma.conduct.integration_planning` (483) and `ma.spac.domestication` (172). A fifth, for SPAC business activities (520), is now part of `ma.rep.merger_sub` (see 4.4).

Three topics were left to catch-alls because they are rare: comfort letters (114), employee stock ownership plans (100) and like-kind exchanges (83). The SPAC "no trading" covenant (315 agreements) is above the 1% bar; it stays in `ma.covenant.other` for now, and open question 18 asks whether to give it a concept.

### 4.3 Size compared with the references

| Tree | Leaves or concepts | Note |
|---|---|---|
| Live section tree | 99 | Many leaves join several provisions, and representations are doubled by party. |
| Unreleased redesign | 140 | Finer, still doubled by party. |
| This design, M&A pack | 272 | 232 provisions, 24 parts and 16 definitions. The first draft had 281. |
| This design, tax pack | 115 | 106 existing nodes and 9 new concepts (6 accepted on 2026-11-22, 3 on 2026-11-23). |

Accuracy falls as the number of classes in one decision grows. This design is finer than the older trees, so the labeling run in section 9 never shows a model the full list. It shows a short list for the section at hand.

### 4.4 Granularity review and trim (2026-11-22 and 2026-11-23)

The maintainer asked whether 281 M&A concepts and 112 tax concepts are finer than needed. This section gives the measurement, the decision and the changes.

**Measurement.** `prevalence.py` counts the agreements in which each M&A concept occurs. It gives two counts:

- The **heading** count: the concept is the heading of a section of its own, by the title rules of section 6. This is a lower bound, because it misses a clause inside a larger section.
- The **text** count: a hand-written pattern for the concept matches the body of a section. Nobody checked samples of the matches, and several patterns match too much.

The best estimate is the larger of the two counts. The corpus has 12,198 agreements, so 1% is 122 agreements. The catch-alls are left out: 7 in the first draft, and 8 now with the lead-in concept. The table counts concepts. The "current packs" columns were measured again on 2026-11-23, after the trial fixes.

| Share of agreements | First draft: heading | First draft: best estimate | Current packs: heading | Current packs: best estimate |
|---|---|---|---|---|
| 50% or more | 56 | 92 | 54 | 89 |
| 10% to 50% | 69 | 150 | 66 | 142 |
| 1% to 10% | 78 | 31 | 80 | 32 |
| Under 1% | 18 | 1 | 20 | 1 |
| Never a heading | 53 | 0 | 44 | 0 |
| Concepts measured | 274 | 274 | 264 | 264 |

What the numbers show:

- **Rare concepts are not the problem.** By the best estimate, one concept is under 1% of agreements: the appraisal-rights closing condition (`ma.condition.dissenting_shares`, 82 agreements). It stays as a known research variable (rule 7 in 4.1).
- **The first draft mixed three grains.** Of its 234 provisions (catch-alls left out), 196 were the heading of a section of their own in at least 1% of agreements, and 124 in at least 10%. The other 38 were clauses inside larger sections: termination fees and triggers, the outside date, the cap and the basket, the fiduciary out. The remaining 40 concepts were parts and definitions.
- **Size against the older trees.** With the party copies merged, the live tree has 80 topics and the 140-leaf redesign has 115. The growth comes from splitting joined titles and from the clause-level concepts.
- **The tax pack is inherited.** 106 of its 115 concepts are nodes of the tree built with the tax professor. It was not trimmed.

**Splits that stay.** A filter on each concept returns different agreements.

| Concepts | Evidence |
|---|---|
| Governing law, forum, jury waiver | By text, the law clause is in 11,097 agreements, the forum clause in 10,288 and the jury waiver in 9,736. 1,645 agreements have a law clause and no forum clause, and 2,000 have no jury waiver. 43% of governing law sections also hold the forum clause and 26% hold the jury waiver, so only a label on the clause separates them. |
| Target fee and reverse fee | About 7,060 and 1,610 agreements, with a different payer. |
| Change of recommendation for a superior proposal and for an intervening event | About 4,190 and 2,330 agreements. |
| Go-shop and no-shop | The go-shop is in 346 agreements (2.8%) and has its own literature. |
| Cap, basket, sandbagging, materiality scrape | Each is a separate variable in deal-points studies of private deals. |

**Merges (2026-11-22).** 20 concepts were merged: 16 into a neighbor and 4 into a catch-all. In each case the parts seemed to occur together or to answer no separate research question. The merged ids had never been released, so they were removed and not deprecated.

**Decision.** On 2026-11-22 the maintainer chose to trim the M&A pack to 261 concepts and to label it in two waves. The tax pack stays as it is. The 10 pilot value ids and their 3 sibling ids are unchanged.

**Independent review of the merges (2026-11-23).** One reviewer agent (Opus) checked each of the 20 merges, with heading counts from its own read-only queries on `latest_sections_search`. It confirmed 13 merges. It asked for 5 concepts to be split back out and for 2 to move to another neighbor. All 7 changes are applied. No merge into a catch-all remains. The M&A pack then had 266 concepts. The trial fixes of the same day added 6 more (section 11.4).

Merges into a neighbor that stay (15 concepts):

| Surviving concept | Merged into it | Evidence |
|---|---|---|
| `ma.structure.effective_time` | Effects of the merger; charter and bylaws of the surviving entity; its directors and officers | Headings in 4,732, 6,028 and 6,377 agreements. Standard mechanics that do not vary. |
| `ma.severability` | Counterparts | 11,134 agreements have both severability and counterparts (98% of the agreements with either). |
| `ma.specific_performance` | Remedies cumulative | Moved on 2026-11-23. The trim had put it in `ma.severability`. The reviewer found that it shares a heading with specific performance or enforcement in 187 of 745 agreements, and with severability in 1. |
| `ma.amendment` | Waiver | 8,651 of the 8,912 agreements with a waiver section also have an amendment section. 1,867 share one heading. |
| `ma.exchange.procedures` | Fractional shares | A step of the exchange. |
| `ma.tender.company_actions` | Rule 14d-10 approvals; board designees after the offer | 567 and 429 agreements. |
| `ma.spac.trust_account` | Waiver of claims against the trust | 807 of the 825 agreements with a trust waiver also have the trust account section. |
| `ma.rep.merger_sub` | SPAC business activities | Moved on 2026-11-23. The trim had put it in the catch-all `ma.rep.other`. It is the same "no prior activities" statement, made by a SPAC. |
| `ma.employee.comp_benefits_continuation` | Service credit; termination of plans | Clauses of one "Employee Matters" section. |
| `ma.termination.effect` | Notice of termination | 1,601 of 1,654 agreements. 651 share one heading. |
| `ma.indemn.claims_procedure` | Third-party claims | One procedure section in most deals. |

Concepts restored on 2026-11-23, each with its old id (5 concepts). The heading counts are from the title rules of section 6. The wave follows the 1% heading rule.

| Concept | The trim had merged it into | Why it is back | Heading | Wave |
|---|---|---|---|---|
| `ma.spac.extension` | `ma.covenant.other` (catch-all) | SPAC studies track deadline extensions, redemptions at the extension vote and who pays for the extension. | 443 agreements | 1 |
| `ma.spac.domestication` | `ma.covenant.other` (catch-all) | The move of the SPAC from the Cayman Islands to Delaware is a research variable. The tax tree also lists it, under "Pre-Closing Restructuring". | 172 | 1 |
| `ma.condition.frustration` | `ma.condition.other` (catch-all) | The heading occurs in 21% of agreements. The reviewer pointed to the prevention doctrine cases (Snow Phipps v. KCAKE, AB Stable). It now sits under "Mutual Conditions", because the rule applies to every party. | 2,569 | 1 |
| `ma.condition.covenant_performance` | `ma.condition.reps_bringdown` | AB Stable v. MAPS Hotels turned on the covenant condition, and practitioners track the two standards separately. It has stub fields for the covenant standard and the cure right. | 870 | 1 |
| `ma.employee.no_third_party_rights` | `ma.employee.comp_benefits_continuation` | It is the variable for studies of whether employees can enforce the covenants. It is a component of the surviving concept. | 256 | 2 |

The first count for the frustration heading was wrong. The title rule for the conditions group matched "Frustration of Closing Conditions" before the frustration rule did, so the heading seemed to occur in 42 agreements. The rule order is fixed. The same fix moved about 400 "Performance of Obligations" headings to the covenant condition.

Small additions from the same review:

- `ma.structure.effective_time` has an exclusion cue that sends board and officer terms at the parent, as in a merger of equals, to `ma.governance.post_closing`.
- `ma.indemn.claims_procedure` has a stub field `defense_control` (indemnitor, indemnitee or conditional).
- The ticking fee had no home. It is now a term and a stub field of `ma.consideration.merger_consideration`.
- Special committee approval clauses (headings in 111 agreements) are now a term of `ma.amendment`, with a title rule.

The group "Condition Mechanics" held only the frustration concept. It stays removed, so the M&A pack has 54 groups and not 55.

**Title-rule audit (2026-11-23).** The adversarial review found that some headings were credited to a neighbor concept or to a group, so the concept they name seemed to have no heading. Each wave 2 provision was checked: which headings name it, and where the title rules sent them. Twelve rules were added or changed, and the waves were set again by the 1% heading rule (122 agreements).

| Concept | Headings that had gone elsewhere | Heading before | Heading now | Wave |
|---|---|---|---|---|
| `ma.conduct.negative_covenants` | "Negative Covenants", "Forbearances" (to the ordinary course covenant) | 0 | 908 | 2 to 1 |
| `ma.price_adjustment.closing_statement` | "Closing Statement", "Closing Balance Sheet" (to closing time and place) | 0 | 453 | 2 to 1 |
| `ma.termination.fee.reverse` | "Parent Termination Fee", "Reverse Termination Fee" (to the fees group) | 5 | 212 | 2 to 1 |
| `ma.indemn.basket` | "Basket", "Threshold", "Deductible" (to the limits group) | 0 | 176 | 2 to 1 |
| `ma.termination.fee.target` | "Company Termination Fee" (to the fees group) | 0 | 106 | stays 2 |
| `ma.termination.outside_date` | "Termination Date" (to the rights group) | 63 | 88 | stays 2 |
| `ma.indemn.cap` | "Cap", "Maximum Liability" (to the limits group) | 0 | 68 | stays 2 |
| `ma.financing.cooperation` | "Financing Cooperation" (to the general efforts covenant) | 194 | 1,461 | already 1 |

Three results of the audit need a note:

- The target fee is in wave 2 and the reverse fee is in wave 1. A bare "Termination Fee" heading stays with the fees group, because such a section often holds both fees. The value pass covers both fees in any case.
- "Company Board Recommendation" headings (about 620 agreements) stay with the recommendation group. That section normally holds the recommendation covenant and the exceptions for a change of recommendation together.
- A "Confidentiality" heading (about 8,200 agreements) cannot tell the covenant before closing from the sellers' duty after closing. All of them count for `ma.access.confidentiality`, so `ma.restrictive.confidentiality` has no heading count and stays in wave 2. Body text must settle it.

**Waves.** Each M&A concept carries `wave: 1` or `wave: 2`, and the validator checks the field.

| Wave | Contents | M&A concepts |
|---|---|---|
| 1 | Section-level topics: a provision that is the heading of a section of its own in at least 1% of agreements. Also the catch-alls and the lead-in concept. | 199 (192 topics, 6 catch-alls and `ma.other.lead_in`) |
| 2 | Clause-level provisions, parts (components and modifiers) and definitions. | 73 (33 provisions, 24 parts, 16 definitions) |

- The catch-all for termination rights, `ma.termination.right.other`, is in wave 2 with its siblings. All other termination triggers except mutual consent are clause-level.
- `ma.employee.comp_benefits_continuation` is in wave 1 after the merge, because the "Employee Matters" section is its heading.
- Wave 2 holds 6 of the 10 pilot value concepts: 4 provisions (one is the derived financing concept), the Material Adverse Effect definition and its carve-outs. The value pass of the span taxonomy plan covers those.
- The tax pack has no waves. One tax pass labels it (step 5 in 9.2).
- In wave 1, a section whose content is only wave 2 concepts gets its group as the label (decided 2026-11-23). 14 groups are marked `assignable: true` for this. Section 3.5 gives the rule, and `tree.md` marks the groups.

**Default filter level.** The 54 groups are the default level for browsing and filtering. A label on a concept counts for its group and its branch, so a filter on a group returns every section that carries one of the group's concepts. An error between two concepts of one group does not change a filter on the group. The groups cannot be the only level: "Governing Law and Forum" as one topic cannot answer "which law governs" or "which deals have no jury waiver".

**How to reproduce.** Export the section titles and the section text with the two read-only queries in the docstring of `prevalence.py` (the text export is about 3 GB). Then run `etl/.venv/bin/python docs/plans/full-taxonomy/prevalence.py sections.tsv text.tsv --per-concept`. It needs ripgrep and about 10 minutes. The "current packs" columns are its output. The "first draft" columns come from the same method on the first draft.

**Limits.** The text patterns are unreviewed. The patterns for `ma.rep.insurance_business`, `ma.consideration.seller_note` and `ma.asset.risk_of_loss` clearly match too much. The merge list rests on one reader's judgment of what researchers filter on, plus the counts above, and on one model review. A person who knows the literature should confirm it. No model accuracy was measured.

## 5. The trees

`tree.md` lists every node. This section gives the counts.

### 5.1 M&A pack

| Level | Count |
|---|---|
| Branches | 15 |
| Groups | 54 |
| Concepts directly under a group | 248 (232 provisions, 16 definitions) |
| Parts of a concept | 24 (21 components, 3 modifiers) |
| All concepts | 272 |

| Branch | Groups | Concepts | Groups (concepts in each) |
|---|---|---|---|
| Deal Structure and Closing | 5 | 27 | The Transaction (4); Asset Deal Scope (8); Tender Offer (5); Closing (3); SPAC Transactions (7) |
| Consideration and Payment Mechanics | 4 | 21 | Price and Form of Consideration (11); Equity Awards and Other Securities (3); Exchange and Payment Procedures (3); Purchase Price Adjustment (4) |
| Representations and Warranties | 10 | 49 | Fundamental (7); Financial (7); Legal and Compliance (4); Commercial (6); Assets, IP and Environmental (5); Employees (2); Transaction-Specific (4); Buyer-Specific (6); Disclaimers (2); Industry-Specific (6) |
| Conduct of Business and Access | 2 | 20 | Interim Operations (15); Information, Confidentiality and Publicity (5) |
| Deal Protection and Stockholder Approval | 4 | 17 | No-Shop and Go-Shop (5); Board Recommendation and Fiduciary Out (5); Stockholder Approval Process (5); Takeover Statutes and Rights Plans (2) |
| Efforts and Regulatory Approvals | 3 | 11 | Efforts to Close (2); Antitrust (6); Other Regulatory Approvals (3) |
| Financing | 3 | 9 | Financing Arrangements (4); Financing Covenants (4); Financing Source Protections (1) |
| Employees and Benefits | 2 | 6 | Continuing Employees (2); Transaction-Related Employee Matters (4) |
| Other Covenants | 4 | 24 | Directors, Officers and Governance (5); Securities, Listing and Deal Litigation (4); Post-Closing and Separation (15); Tax Matters (lists tax pack nodes only) |
| Conditions to Closing | 2 | 13 | Mutual Conditions (5); Conditions for One Party (8) |
| Termination | 3 | 16 | Termination Rights (9); Effect of Termination (2); Termination Fees and Expenses (5) |
| Indemnification and Risk Allocation | 4 | 16 | Survival and Scope (3); Limits on Recovery (7); Claims and Sources of Recovery (5); R&W Insurance (1) |
| Dispute Resolution and Remedies | 2 | 8 | Governing Law and Forum (4); Remedies and Enforcement (4) |
| Definitions and Interpretation | 3 | 22 | Defined Terms (1); Key Defined Terms (18); Interpretation (3) |
| General Provisions | 3 | 13 | The Agreement (6); Costs and Parties (3); Other (4) |

### 5.2 Tax pack

| Level | Existing tree | This design |
|---|---|---|
| Level 1 | 9 | 10 branches (9 existing, 1 new) |
| Level 2 | 45 | 35 groups and 10 concepts. The 10 are existing level 2 nodes that have no children. |
| Level 3 | 96 | 96 existing concepts and 6 new |
| Under the new branch | 0 | 3 new definition concepts |
| All concepts | 106 leaves | 115 |

| Branch | Groups | Concepts |
|---|---|---|
| Intended Tax Treatment, Elections, and Purchase Price Allocation | 6 | 18 |
| Tax Interim Operating Covenants | 0 | 9 |
| Transfer and Withholding Taxes | 3 | 7 |
| Tax Cooperation, Returns, Contests, and Refunds | 5 | 18 (2 new) |
| Tax Allocation and Indemnity | 3 | 15 |
| Special Regimes | 3 | 12 (1 new) |
| Tax Representations | 9 | 25 (3 new) |
| Tax Attributes | 5 | 7 |
| Other / Unclassified | 1 | 1 |
| Tax Definitions (new) | 0 | 3 (all new) |

## 6. Coverage measurement

### 6.1 Method

The question: what share of the sections in the corpus map cleanly to a leaf?

The corpus is the table `latest_sections_search`: 12,198 agreements and 1,196,581 sections, read on 2026-11-22 with one read-only query. 58,486 sections (4.9%) have no title. The rest carry 133,055 distinct titles after normalization.

`docs/plans/full-taxonomy/title_coverage.py` holds 403 rules. A rule is a pattern for a title, sometimes limited to an article type. "Insurance" under a representations article maps to `ma.rep.insurance`, and "Insurance" under a covenants article maps to the directors' and officers' insurance covenant. Each rule names one target: a concept, or a navigation node when a section with that title normally holds several concepts.

The script splits a title on ";" and maps each part. It then puts the section in one class:

| Class | Meaning |
|---|---|
| leaf | Every part maps to a specific concept. |
| group | Every part maps, and at least one maps to a navigation node. |
| partial | Some parts map and some do not. |
| catch-all | The title maps only to a catch-all concept, such as "[Reserved]". |
| untitled | The section has no title. |
| unmapped | No part maps. |

This is a measurement of the tree. It is not a labeler. Section 9 explains why labels must come from body text.

### 6.2 Result

| Class | Sections | Share of all | Share of titled |
|---|---|---|---|
| leaf | 975,410 | 81.5% | 85.7% |
| group | 90,583 | 7.6% | 8.0% |
| partial | 8,509 | 0.7% | 0.7% |
| catch-all | 15,982 | 1.3% | 1.4% |
| untitled | 58,486 | 4.9% | |
| unmapped | 47,611 | 4.0% | 4.2% |

**85.7% of titled sections map cleanly to a leaf.** With the group class, 93.7% of titled sections have a home in the tree.

Five changes after the first draft moved these figures:

- **The trim (2026-11-22).** About 12,900 "Employee Matters" sections moved from the group class to the leaf class, because that heading now belongs to the one merged employee concept.
- **The review (2026-11-23).** The SPAC extension, SPAC domestication and frustration concepts are back. About 1,100 sections that the trim had sent to the catch-all class are in the leaf class again.
- **A rule-order fix (2026-11-23).** About 2,750 sections on frustration of conditions and on performance of obligations moved from the group class to the leaf class. See 4.4.
- **The title-rule audit (2026-11-23).** About 500 sections moved from the group class to the leaf class: headings that name one fee, the cap or the basket. Other headings moved from one concept to another, which does not change the classes. See 4.4.
- **The trial fixes (2026-11-23).** Ten rules were added for the new concepts and for headings that had gone to the wrong concept (11.4). About 300 sections left the unmapped class. About 430 sections moved to the catch-all class: headings on relief-program loans and government grants that the banking rule had taken, and "SPAC Public Filings" covenants that the public announcements rule had taken.

In the first draft, the leaf share was 84.3% and the leaf or group share was 93.7%.

The group class is expected. A section on the conditions to one party's obligations (20,224 sections) holds several conditions. "Taxes" in a representations article (15,090 sections) is the tax umbrella that the tax pass refines. "Termination" (13,658 sections) lists every termination right. These sections need labels at the clause level, below the section.

The group class splits three ways by the node that the title rule names:

| Node that the rule names | Sections (about) | What wave 1 gives the section |
|---|---|---|
| One of 5 assignable M&A groups: conditions for one party, termination rights, termination fees, limits on recovery, board recommendation | 43,900 | The group as its label (section 3.5). |
| A tax navigation node | 35,700 | The tax node. The tax pass refines it. |
| An M&A node that holds wave 1 concepts only: mutual conditions, the representations branch, claims procedure | 11,600 | Several wave 1 concepts. No group label is needed. |

Coverage holds across deal types. The share of titled sections in the leaf or group class:

| Deal type | Target | Sections | Leaf or group |
|---|---|---|---|
| Merger | public | 321,972 | 96.3% |
| Merger | private | 299,289 | 94.0% |
| Tender offer | public | 64,357 | 97.2% |
| Stock acquisition | private | 198,668 | 91.0% |
| Stock acquisition | public | 14,678 | 86.6% |
| Asset acquisition | private | 117,547 | 90.2% |
| Asset acquisition | public | 9,452 | 91.1% |
| Membership interest purchase | private | 111,267 | 92.8% |

Private stock and asset deals score 5 to 6 points lower than public mergers. Their titles vary more, and they carry more deal-specific sections.

### 6.3 What is in the unmapped 4.2%

The unmapped and partial sections carry 27,822 distinct titles. The most frequent one occurs 160 times, so the group is a long tail and has no large missing topic.

The script draws a random sample with a fixed seed (`--unmapped 150`). One reader sorted the 150 titles by eye, with no second review. The sample was drawn before the later rule changes, which moved about 300 sections out of the unmapped class.

| Kind of title | Count | Share | Examples |
|---|---|---|---|
| Another wording of a provision the tree has | 66 | 44% | "consent to jurisdiction; receipt of process", "no contrary judgment", "time limits" |
| Too vague to place without the body | 37 | 25% | "restrictions", "securities", "action", "post-closing" |
| Specific to one deal | 27 | 18% | "interim supply agreements", "detained aircraft", a party's name |
| Specific to one industry | 11 | 7% | "water rights", "nuclear decommissioning trusts", "gift cards" |
| Foreign-law provision | 5 | 3% | "squeeze out", "position statement", locked box leakage |
| General topic with no concept | 4 | 3% | "section 1031 exchange", "deferred underwriting discount" |

What the sample shows:

- 44% of the sampled sections fit an existing concept under another wording. A labeler that reads the body will place them.
- 25% can be placed only from the body.
- 31% belong in a catch-all: deal-specific, industry, foreign-law and uncovered general topics.
- Industry topics and uncovered general topics are the real gaps. They are 10% of the sample. The sample comes from the unmapped and partial classes, which were 5.0% of titled sections when it was drawn, so the real gaps are about 0.5% of titled sections.

### 6.4 Limits of this measurement

- A title is a weak signal. A section titled "Governing Law" can also hold the forum clause. The measurement shows that the tree has a place for the section, and it does not show that a label would be right.
- 234 of the 387 concepts are the target of at least one title rule. The other 153 never appear as a title: components, definitions, clauses inside a larger section, and most tax leaves. Title coverage says nothing about them.
- The rules were tuned on this corpus. The figures describe fit to this corpus and are not a forecast for new filings.
- Agreements under UK or Australian law are in the corpus and were not removed. They lower the figures slightly.

## 7. Crosswalk

`crosswalk.md` lists every older leaf with its new nodes. The relation describes the new node compared with the older one: `exact` (same scope), `narrower` (the older leaf was split), `broader` (older leaves were merged), `related` (overlap).

| Older tree | Leaves | Mapped | One exact match | Split into narrower concepts | Merged into a broader concept |
|---|---|---|---|---|---|
| Live section tree (12 / 61 / 99) | 99 | 99 | 54 | 41 | 4 |
| Unreleased redesign (140 leaves) | 140 | 140 | 89 | 42 | 9 |
| Tax clause tree (9 / 45 / 96) | 106 leaves, 150 nodes | 150 | 150 | 0 | 0 |

- **Party copies.** A target leaf and a buyer leaf of the same representation both map to one concept as `exact`, each with `party: target` or `party: buyer`. The merge of the two copies is not counted as "broader", because the party survives on the instance.
- **Splits.** A split leaf maps to 2 to 8 concepts in the live tree and to 2 to 12 in the redesign. The widest split is the conduct of business leaf, which maps to the ordinary course covenant, the negative covenants and their 10 components.
- **Merges.** 4 live leaves and 9 redesign leaves map to one broader concept. Two live leaves on protections for lenders map to `ma.financing.source_protections`. The leaves for specific performance map to `ma.specific_performance`, which also holds the remedies cumulative clause. The "(non-indemnity)" leaves for the damages waiver map to `ma.damages_waiver`, which also covers the clause in the indemnification article. The redesign's leaves for amendment and waiver, for severability and counterparts, and for exchange procedures map to the merged concepts of 4.4. Its two fee leaves map to the fee concepts, which also hold the fee triggers.
- **New concepts.** 97 M&A concepts have no older leaf. They include the 7 SPAC concepts, most of the definitions, covenants after closing such as the non-compete and the release, 20 representations and the 6 concepts added after the trial. `crosswalk.md` lists them.
- **How to use it.** An old filter on a live leaf becomes a filter on the set of new concepts that the leaf maps to. A new label on a narrower concept can also be shown under its old leaf while the old tree is still served.

## 8. Tax changes

The tax clause tree was built with a tax professor. This design keeps it.

### 8.1 What does not change

- All 9 level 1 nodes, all 45 level 2 nodes and all 96 level 3 nodes are present.
- Each label is the same, character for character. A test compares them.
- Each node has the same parent.
- No node was removed, renamed, merged or moved.

### 8.2 Changes of form

These do not change what any node means.

| Change | Reason |
|---|---|
| Each node gets a readable id next to its number: `tax.4.1.3` is `tax.returns.review_rights`. The number stays in the crosswalk as an exact match. | The plan asks for stable, readable ids that do not encode position. A number such as 4.1.3 changes meaning if the tree is ever reordered. |
| A level 2 node with children becomes a group. A level 2 node without children (10 nodes) becomes a concept. | The pack format separates places in the tree from things that a label points to. |
| Each tax navigation node is marked `assignable: true`. | The present tax labeler may assign a level 1 or level 2 node when no leaf fits. This keeps that behavior. |
| 19 tax nodes are also listed under M&A navigation nodes. | A reader who browses the M&A tree finds the tax provisions where expected. The tax pack still owns them. |
| Each concept gets a description, canonical terms, example phrases and, where useful, exclusion cues. | The labeling run needs them. The tax professor has not reviewed this text. |

### 8.3 Additions (accepted 2026-11-22)

Each node below is marked `new_in_redesign: true` in the YAML. The maintainer accepted all 7 on 2026-11-22.

The evidence comes from the 243 agreements that have tax clause labels today, and from section titles across the corpus.

| # | Node | Where | Reason and evidence |
|---|---|---|---|
| 1 | `tax.nav.definitions`, "Tax Definitions" (a new level 1 branch) | Top level | The tree has no place for tax definitions. They appear in 65 of the 243 labeled agreements. 59 of those clauses have no label and 49 sit in "Other / Unclassified". |
| 2 | `tax.def.tax`, "Tax" / "Taxes" definition | Under 1 | The scope of "Taxes" sets the scope of the tax indemnity and the tax representations. Researchers compare it. |
| 3 | `tax.def.tax_return`, "Tax Return" definition | Under 1 | It is the usual companion of 2. It can be dropped if the maintainer wants a smaller branch. |
| 4 | `tax.def.tax_period`, tax period definitions (Pre-Closing Tax Period, Straddle Period) | Under 1 | These terms set the line between the seller's and the buyer's tax years. The straddle rules in `tax.nav.indemnity` depend on them. |
| 5 | `tax.international.transfer_pricing`, "Transfer Pricing" | Special Regimes, International Tax | "Transfer pricing" appears in 12.4% of all agreements. In the labeled set it appears in 27 agreements, with 11 clauses in "Other" and 10 without a label. The International Tax group has no node for it. |
| 6 | `tax.cooperation.post_closing_actions`, "Restrictions on Post-Closing Tax Actions" | Cooperation, Returns, Contests, and Refunds; Post-Closing Cooperation and Records | The covenant that the buyer will not amend returns or make elections for pre-closing periods. It appears in 44 labeled agreements and is now forced into the interim covenant nodes, which cover the period before closing, or into return preparation. 173 sections carry it as a title. |
| 7 | `tax.cooperation.tax_sharing_termination`, "Termination of Tax Sharing Agreements" | Same group as 6 | The covenant that tax sharing agreements end at closing. 624 sections in 621 agreements carry it as a title. The labeled set shows it in 30 agreements, where it is mostly labeled as the tax sharing representation (`tax.7.3.2`), which is a different statement. The evidence is weaker than for 5 and 6. |

### 8.4 Additions after the labeling trial (accepted 2026-11-23)

The tax pass of the labeling trial (section 11.4) read 300 tax sections in 50 agreements. A blind second read covered 75 of them. Both reads marked the topics below as having no leaf. The maintainer accepted three new leaves on 2026-11-23. Each is marked `new_in_redesign: true`.

| # | Node | Where | Reason and evidence |
|---|---|---|---|
| 8 | `tax.rep.returns_made_available`, "Tax Returns and Audit Reports Made Available" | Tax Representations, Returns and Payments | The representation that the company gave the buyer copies of its returns, examination reports and deficiency notices. Seen in 5 of the 50 agreements. It is a different statement from "returns filed and taxes paid". |
| 9 | `tax.rep.no_power_of_attorney`, "No Tax Power of Attorney" | Tax Representations, Audits and Proceedings | The representation that no power of attorney for taxes is in force or stays in force after closing. Seen in 3 agreements. |
| 10 | `tax.rep.section_6662_disclosure`, "Disclosure of Positions Under Section 6662" | Tax Representations, Returns and Payments | The representation that every position that could draw a substantial understatement penalty was disclosed on the returns. Seen in 3 agreements. |

Three other topics got no leaf. Each was seen in 2 agreements. A cue on an existing leaf says where the text goes:

| Topic | Cue on | The text takes |
|---|---|---|
| Section 965 transition tax | `tax.international.cfc` | That concept. |
| Who owns an escrow fund or an exchange fund for tax purposes, and who reports its income | `tax.adjustment.price_adjustments` and `tax.adjustment.earnouts` | The group `tax.nav.deal.adjustment_treatment`. |
| Tax clearance certificates | `tax.cooperation.interim` | That concept. |

No other tax node changed.

### 8.5 A change of scope outside the tree

466 clauses in 178 of the labeled agreements concern Section 409A or Section 280G. The tax module picks them up today. This design assigns them to the M&A pack (see 3.6) and lists the 280G concept under `tax.nav.other`. This changes which pack labels them and changes no tax node. The maintainer confirmed this on 2026-11-22.

## 9. How labeling would run

This section is a plan. No full labeling ran. A trial of steps 1 to 6 on 50 agreements ran on 2026-11-23, with model subagents and no billed API call. Section 11.4 gives its results.

### 9.1 Why body text

- 153 of the 387 concepts never appear as a title.
- 4.9% of sections have no title, and about 1 in 10 titled sections sits in the group class and holds several concepts.
- A title names the main topic and misses the rest. The governing law section often holds the jury waiver.
- The title rules of section 6 serve only as a starting hint and as a check on the result.

### 9.2 Steps

1. **Unit.** The unit is the section, cut into paragraphs with character offsets, as in the text layer of the span taxonomy plan. A label instance is one concept on one span of text, with the party, a confidence and the model's evidence quote. A section can carry several instances.
2. **Short list (no model cost).** For each section, build a short list of 15 to 30 candidate concepts from the article heading, the title hint, the canonical terms found in the body and the defined terms the section uses. Embeddings can be added. The goal is near-complete recall with a short list.
3. **Classification on body text.** A small model reads the section and the cards of its candidates (label, description, terms, exclusion cues). It returns the concepts present, the span for each, and the party. "None of these" is allowed and becomes a catch-all. The model never sees the full list of 387, because accuracy falls as the class count grows.
4. **Party.** Simple rules set the party from the article heading ("Representations and Warranties of the Company") and check the model's answer. A conflict sends the section to step 6.
5. **Tax pass.** A section or clause that receives a tax node goes to a second pass with the tax pack only. That pass assigns the tax leaf, as the tax module does today. The M&A pass does not need the tax leaves, which keeps its short lists small.
6. **Second read.** A stronger model reads the sections that fail a check: low confidence, a conflict between the title hint and the label, a party conflict, or a representation concept outside a representations article.
7. **Wave 2.** The first run of steps 2 to 6 uses the wave 1 concepts and the 14 assignable groups (section 3.5). The card of a group lists the labels of its wave 2 concepts. When a section holds content of such a group that no wave 1 concept covers, the model returns the group. Wave 2 then reads the sections that carry a group label. It assigns the clause-level provisions, the parts and the definitions (the `wave` field, section 4.4), and those concept labels replace the group label. A section where wave 2 finds no concept of the group goes to the second read. A component is labeled only inside sections that already carry the parent concept. Definition concepts come mostly from the defined-term graph: the term string identifies them, so most need no model.
8. **Gold and bars.** A stratified gold set of 150 to 300 agreements (by deal type, target type and year) with per-concept precision and recall bars, as the plan sets for value concepts. The older labels serve as a second comparison through the crosswalk.
9. **Monitors.** Expected rates per concept catch drift. A governing law concept found in under 95% of agreements is an alarm.
10. **Serving.** New labels are served next to the old ones until they meet the bar. The crosswalk keeps old filters working.

### 9.3 Cost estimate

**This is an estimate. Nothing was billed.** It uses the list prices quoted in the span taxonomy plan (2026-10, per million tokens: Haiku 5.5 $0.10 in and $0.50 out; Sonnet 5.5 $2 and $10; Opus 5.5 $4 and $20; batch runs at half price) and sizes measured on 2026-11-22.

Assumptions:

- **Text.** The 1,196,581 sections hold 3.17 billion characters of XML (mean 2,652 per section), measured with one read-only query. At 3 to 4 characters per token, that is 0.8 to 1.0 billion tokens. Each section is read once.
- **Prompt overhead.** A short list of 15 to 30 concept cards and the rules add 2,500 to 4,000 tokens per request. With about 15 requests per agreement, that is 0.5 to 0.7 billion tokens. Most of it is a shared prefix that the cache serves at one tenth of the input price. It counts here as 0.1 to 0.3 billion full-price tokens.
- **Output.** 50 to 120 tokens per section, or 60 to 145 million tokens.
- **Batch prices** throughout.

| Item | Input | Output | Estimated cost |
|---|---|---|---|
| Step 3 on Haiku 5.5, all sections | 0.9 to 1.3 billion | 60 to 145 million | $60 to $100 |
| Step 3 on Sonnet 5.5, all sections (alternative) | same | same | $1,200 to $2,000 |
| Step 6 on Sonnet 5.5, 10% to 20% of sections (assumed share) | 0.09 to 0.26 billion | 12 to 29 million | $150 to $400 |
| Step 5, tax pass on Sonnet 5.5, 3% to 5% of the text (assumed share) | 30 to 50 million | 4 to 8 million | $50 to $90 |
| Step 7, components on Haiku 5.5 | about 100 million | about 10 million | under $10 |
| Step 8, gold for 200 agreements on Opus 5.5, if billed through the API | about 30 million | about 4 million | $100 to $200 |

- **Recommended setup** (Haiku first, Sonnet for the second read and the tax pass): about **$350 to $800** for the first full pass, gold included.
- **Sonnet reads everything:** about $1,400 to $2,300, with the tax pass and gold.
- **Check against the pilot.** The plan measured $0.092 per agreement for the ten-concept value pass on Sonnet, about $1,200 for the corpus. The all-Sonnet row above works out to $0.10 to $0.16 per agreement, which fits a pass that reads all the text and not a selected part.
- **Monthly intake** of about 100 agreements: under $10 with the recommended setup.

The table was made before the wave split. Wave 2 reads some sections a second time: those that hold a clause-level provision. That share is not measured, and the trial must give it.

What can move the estimate: the share of sections that need a second read, how well the cache works inside a batch, the reasoning effort setting (reasoning tokens bill as output), and retries.

**Before any full run:** a billed trial on 50 agreements, estimated under $10, gives the real cost per agreement and the real second-read share. Each paid run needs the maintainer's go-ahead with the amount stated.

## 10. Open questions for the maintainer

1. **Tax additions.** Decided 2026-11-22: all 7 nodes in 8.3 are accepted. Decided 2026-11-23: the 3 leaves in 8.4 are accepted, and 3 other topics get a cue only.
2. **Sections 280G and 409A.** Decided 2026-11-22: the M&A pack owns them, and the tax tree lists the 280G concept for navigation.
3. **Umbrella labels.** Partly decided 2026-11-23 (see 16). The tax pack lets a label point to any branch or group. The M&A pack lets a label point to 14 groups, in wave 1, and uses catch-all concepts for content that fits no concept. Still open: should the tax pack keep the wider rule?
4. **Industry topics.** The M&A pack has 5 industry representation concepts. Industry titles are most of the real gaps in 6.3. Is a later industry pack (banking, healthcare, energy, insurance) wanted, or is `ma.rep.other` enough?
5. **SPAC concepts.** 7 SPAC concepts sit in the M&A pack. Keep them there, or plan a separate pack?
6. **Foreign-law provisions.** Locked box, squeeze-out and similar UK or Australian provisions have no concept, in line with the US focus. Confirm.
7. **Waves.** Decided 2026-11-22: provisions first, then parts and definitions. The trim in 4.4 refined this the same day. After the reviews and the trial fixes of 2026-11-23, wave 1 has 199 M&A concepts (192 section-level topics, 6 catch-alls and the lead-in concept). Wave 2 has the other 73: 33 clause-level provisions, 24 parts and 16 definitions. The tax pack (115 concepts) has no waves.
8. **Defined terms.** Are the 16 M&A definition concepts the right set? Which should get value schemas first?
9. **Value schema stubs.** 76 M&A concepts carry an untested sketch of fields. Keep the stubs in the pack as candidates, or remove them until each is piloted?
10. Withdrawn on 2026-11-23. The other questions keep their numbers.
11. **Party values.** The role is on the concept and the party is on the instance. Which party values should an instance allow: target, buyer, seller, merger subsidiary, sponsor, mutual?
12. **The unreleased 140-leaf redesign.** This design replaces it. Can the redesign file be retired after approval, with its crosswalk kept?
13. **Id freeze.** After approval the ids are frozen. Any id the maintainer wants changed must change before the first labeling run.
14. **Budget and models.** Decided 2026-11-22: the billed 50-agreement trial waits until the maintainer has reviewed the tree. The setup and ceiling are decided then.
15. **Granularity.** Decided 2026-11-22: trim the M&A pack to 261 concepts with 20 merges, label in two waves, and use the groups as the default filter level. The review of 2026-11-23 restored 5 concepts, so the pack had 266 concepts and 15 merges. See 4.4. The trial fixes added 6 concepts, so the pack has 272. See 11.4.
16. **Sections that hold only wave 2 concepts.** Decided 2026-11-23: yes. In wave 1 such a section gets its group as the label. 14 groups are marked `assignable: true` (section 3.5). No M&A branch is assignable.
17. **The merge list.** Reviewed on 2026-11-23 by an independent model reviewer: 13 merges confirmed, 5 concepts restored, 2 moved (section 4.4). Still open: should a person who knows the literature confirm the 15 merges that stay before the ids are frozen?
18. **Topics above the 1% heading bar that have no concept.** The trial fixes measured five: representations on government grants and subsidies (headings in 155 agreements), representations on relief-program loans (166), the SPAC "no trading" covenant (315), the SPAC covenant to keep SEC reports current (402) and the officer's certificate condition (310). Each stays in a catch-all, and the card of the catch-all names it (11.4). Should any of them get a concept?

## 11. Independent review

One reviewer agent (Opus) read `tree.md`, this document and all 25 YAML files, with a brief to find overlaps, mixed concepts, gaps and legal errors. It returned 29 findings. The two tables show what happened to them. A second reviewer agent checked the 20 merges on 2026-11-23. Section 4.4 records its verdicts and the changes.

### 11.1 Applied

| Finding | Change |
|---|---|
| An example phrase of `ma.rep.financing` was an editor's note, and it was wrong on the substance. | Replaced with contract wording. |
| The debt and equity commitment concepts overlapped the financing representation. | Both are now components of `ma.rep.financing`. They mark the spans that describe each commitment. |
| Ten example phrases belonged to a sibling concept. | Removed. An exclusion cue now points to the sibling where one was missing. |
| The liability cap before closing had three possible homes. | It belongs to `ma.termination.fee.exclusive_remedy` only. |
| `ma.recommendation.change_superior_proposal` also described the termination right. | The description now covers the change of recommendation only. |
| `ma.access.confidentiality` joined confidentiality before closing with the sellers' duty after closing. | New concept `ma.restrictive.confidentiality`. |
| `ma.consideration.buyer_securities` joined buyer stock with seller notes. | New concept `ma.consideration.seller_note`. |
| `ma.regulatory.strategy_control` also held the ban on acquisitions that delay clearance. | New concept `ma.regulatory.no_interfering_acquisitions`. |
| The Rule 14d-9 and Rule 14e-2 disclosure exception had no concept. | New component `ma.recommendation.permitted_disclosure`. |
| The release of seller guarantees and credit support had no concept. | New concept `ma.credit_support_release`. |
| `ma.consideration.exchange_ratio` had the kind `definition`. | It is a `provision`, so the first wave labels it. |
| The match right had the superior proposal concept as its parent. | Its parent became `ma.recommendation.board_recommendation`, with a field for the grounds it applies to. The review of 2026-11-23 moved it back to `ma.recommendation.change_superior_proposal` (11.3). |
| Several tax descriptions and terms overlapped (tax benefit offsets, closing agreements, status and attribute representations). | Descriptions, terms and exclusion cues were narrowed. No tax node changed its label or parent. The review of 2026-11-23 found that seven of these edits changed what a node means. They are reverted (11.3). |
| Two seam listings pointed to the wrong M&A group. | The tax opinion condition is listed under "Conditions for One Party". The tax treatment of indemnity payments is listed under "Claims and Sources of Recovery". |

### 11.2 Left for the maintainer

These suggestions change scope or ownership, so they are not applied.

| Suggestion | Note |
|---|---|
| Give M&A its own `ma.structure.pre_closing_reorganization`, and limit `tax.restructuring.pre_closing` to steps taken for tax reasons. | A labeler would have to judge the reason for a restructuring, which is often not stated. The tax pack keeps the one concept for now. |
| Add an 8th tax proposal for transaction tax deductions (allocation to the pre-closing period, the 70% safe-harbor election). | Common in private equity deals. It sits in `tax.attribute.other` today. A decision for the maintainer and the tax professor. |
| Fold `tax.def.tax_return` into `tax.def.tax`. | The definition is close to constant across agreements. See open question 1. |
| Split joint or several liability of sellers out of `ma.indemn.order_of_recovery`. | A real research variable. It could also be a value field. |
| Split non-disparagement out of `ma.restrictive.non_solicit`. | No corpus count yet. |
| Split `ma.rep.no_conflict` into contract conflicts and governmental consents. | Many agreements give them separate sections. Both older trees keep them as one leaf. |
| Mark `ma.financing.structure` as derived from other concepts' spans. | It is a pilot value concept. The pilot plan owns that choice. |
| Add concepts for the buyer's access to the seller's insurance after closing, and for prorations of rents and utilities in asset deals. | The insurance concept was added after the trial: `ma.separation.insurance_access` (11.4). Prorations are not measured in the corpus yet. |
| Add exclusion cues among the three tax nodes that mention the partnership representative. | Small. It needs the tax professor's view on where the push-out election belongs. |

### 11.3 Adversarial review of the whole PR (2026-11-23)

A third reviewer agent (Opus) read the PR with a brief to break it. Its findings are applied as follows.

| Area | What changed |
|---|---|
| Title rules and waves | Headings that name one concept had been credited to a neighbor or a group. Twelve rules were added or changed, and four provisions moved to wave 1 (the audit table in 4.4). |
| Derived concept | `ma.financing.structure` is `derived: true` and labels no text. "No financing condition" text goes to `ma.rep.financing` or `ma.financing.buyer_covenant`. Its group is no longer assignable. |
| Validator and tests | The loader reports bad YAML, duplicate keys, unknown fields and unquoted release numbers. New checks cover the pack of a concept's parent, an M&A concept under a branch, a wave on a tax concept, assignable tax navigation nodes and id forms. 66 tests (was 35). |
| M&A overlaps | Cues or rules for five pairs: merger consideration and purchase price, the two escrows, no other representations and non-reliance (the speaker decides), limited guarantee and merger sub obligations, release and indemnification. |
| M&A single concepts | Closing deliverables, damages waiver (the older "(non-indemnity)" leaves are now `broader`), termination for a legal restraint, the parent of the match right, 13 missing party roles, and "Bank Merger" as a term of `ma.structure.merger`. |
| Tax pack | See below. No id, label, kind or parent changed, and no node was added. |
| Seams | Cues between the packs for the Section 382 rights plan, tax contests and the claims procedure, tax items listed in the general indemnity, the SPAC domestication, and records access. A covenant that a payment complies with Section 409A belongs to the concept of that payment (`ma.consideration.earnout`, `ma.equity.awards` or `ma.employee.retention_severance`). |
| Older trees | Nine crosswalk fixes: for example, the live takeover statutes leaf also holds rights plans, the two escrow leaves also map to `ma.price_adjustment.escrow`, and "Further Assurances; Post-Closing Cooperation" also maps to `ma.records_access`. |

Tax pack changes:

- **Indemnified Taxes.** The promise to indemnify and its lead-in take `tax.indemnity.obligation`. Each listed category of taxes takes its own concept. The examples of the category concepts are now list items, and cues point both ways.
- **Cooperation.** One standard cooperation sentence takes `tax.cooperation.post_closing`. The concepts for filing information and for record retention apply only to a clause that is specifically about that topic. `tax.cooperation.records_access` now says "before closing".
- **Reverts.** Seven nodes are back to the meaning of their legacy label: `tax.indemnity.limitations`, `tax.indemnity.benefit_offset`, `tax.indemnity.rep_breach`, `tax.rep.s_corp_no_termination`, `tax.rep.partnership_classification`, `tax.rep.no_post_closing_income` and `tax.interim.maintain_elections`. The legacy tree holds labels only, so the label is the measure.
- **Reverts that needed a cue.** Five overlaps came back with the reverts and are handled by cues:
  1. The working capital exclusion ("not liable for a Tax taken into account in Closing Working Capital") belongs to `tax.indemnity.limitations`, with cues between it and `tax.indemnity.benefit_offset`.
  2. A listed indemnity item for the breach of a tax covenant has no concept. A cue sends it to the group `tax.nav.indemnity.indemnified_taxes`.
  3. A representation on built-in gains (Section 1374) has no concept. A cue sends it to the group `tax.nav.reps.s_corporation`.
  4. A representation that an entity is disregarded goes to `tax.rep.entity_status`.
  5. In the income-timing representation, the lead-in takes `tax.rep.no_post_closing_income` and each listed cause takes `tax.rep.no_deferred_income`.
- **Roles.** The 9 interim covenants have the role `obligor`, and `tax.indemnity.obligation` has `indemnitor`.
- **Definitions.** "Indemnified Taxes", "Pre-Closing Taxes" and "Transfer Taxes" now have rules in cues, on existing concepts and groups.

Not done, with the reason:

| Finding | Reason |
|---|---|
| A role on `tax.opinion.closing_condition` | The condition tests no party's conduct. The party that matters receives the opinion. `tested_party` would then mean something else than in the M&A pack. |
| A split of "Company Board Recommendation" and "Confidentiality" headings | A title cannot settle either one (4.4). |

### 11.4 Labeling trial and the fixes it led to (2026-11-23)

**The trial.** Steps 1 to 6 of section 9.2 ran on 50 agreements under US law, drawn by deal type, target type and year (4,879 sections). All model work ran in subagents. No billed API call was made, and the database was read only. The report and the outputs of the trial are kept outside this repo.

| Measure | Result |
|---|---|
| First read (small model, short list of about 15 labels) against a blind second read by a stronger model (731 sections) | precision 0.938, recall 0.837, F1 0.885 |
| The same at group level | F1 0.895 |
| Blind labels that were on the short list | 0.887 |
| Party, on matched labels with a party role | 95.4% exact |
| Sections flagged for a second read | 23.1% |
| Tax pass against a blind tax re-read (75 sections) | F1 0.928 at node level |

The first reader was precise. Most of its errors were labels that it did not give, and half of those were labels that the short list did not offer. Swaps between two concepts of one family were rare. The fixes to the short list and to the party rule belong to the labeling code and are not part of this document.

**What the trial showed about the tree.**

- A specific concept that the reader did not have ended in a catch-all: `ma.rep.other` (precision 0.47), `ma.covenant.other` or `ma.condition.other`.
- Cues pointed to wave 2 concepts, which a wave 1 reader cannot give. Text of three parts was lost this way: the force-the-vote clause (5 agreements), the costs of financing cooperation (7) and the statement that employees get no rights (at least 9).
- About 1% of sections are the opening words of an article and state no term. No label fitted, and readers split three ways.
- Some recurring topics had no concept.

**Changes to the M&A pack.** The numbers are those of the trial report.

| # | Finding | Change |
|---|---|---|
| 1 | Article lead-ins: about 50 sections in 35 of the 50 agreements. | New concept `ma.other.lead_in`. A lead-in that states how the disclosure schedules or SEC filings qualify the statements also gets `ma.interpretation.disclosure_schedules`. |
| 2 | A cue named a wave 2 part, and readers did not know that the part's text takes the label of its parent. | A new rule, which the validator checks. A cue of a wave 1 concept that names a wave 2 concept also names its wave 1 label: the parent for a part, the group otherwise. A wave 1 concept names each of its own parts and says that the part keeps its label in wave 1. 41 cues changed, and 8 wave 1 concepts now name their parts. |
| 3 | A seller's confidentiality duty after closing was labeled as the confidentiality covenant before closing. | Covered by the rule of row 2: the cue now gives the group `ma.nav.covenants.post_closing`. |
| 4 | `ma.rep.other` took statements that have a concept. | Its card now says when it applies, lists the industries that belong to it, and has cues for the statements it took: banking, litigation, full disclosure, real property and governmental consents. |
| 5 | Industry representations with no concept. | Measured (table below). New concept `ma.rep.communications`. Gaming, education, franchise systems and transportation are below the 1% heading bar and are listed on the card of `ma.rep.other`. |
| 6 | Closing conditions with no concept. | New concept `ma.bankruptcy.court_approval` for sales by a debtor in bankruptcy. The other items stay in `ma.condition.other`, and its card now names them: the officer's certificate, a limit on net debt, the SPAC net tangible assets floor, and court approval of a plan or scheme of arrangement under Canadian or UK-style law. A SPAC minimum cash condition is `ma.spac.minimum_cash`. |
| 7 | Environmental clean-up covenants and special environmental indemnities (5 agreements). | New concept `ma.environmental.remediation` for the covenant. The indemnity stays with `ma.indemn.specific`, which now names it. |
| 8 | Subrogation of the indemnifying party (2 agreements). | No new part. It is part of `ma.indemn.recovery_offsets`, which now says so and has the terms. The title rules already sent "Subrogation" headings (257 agreements) there. No study treats it as a variable of its own (rule 6 in 4.1). The insurer's waiver of subrogation stays with `ma.indemn.rwi`. |
| 9 | Carve-out separation terms: shared contracts and assets, and access to the seller's insurance (3 to 4 agreements). | New concepts `ma.separation.shared_contracts` (wave 1) and `ma.separation.insurance_access` (wave 2). |
| 10 | SPAC covenants: no trading in SPAC securities, and keeping SEC reports current. | No new concept. The card of `ma.covenant.other` names both. A SPAC's statement that it is an emerging growth company is part of `ma.rep.investment_company`, which now says so. |
| 11 | Protections for the lenders sit inside other sections. Recall was 0 of 5. | A cue on `ma.governing_law`, `ma.forum_selection`, `ma.jury_waiver`, `ma.amendment`, `ma.third_party_beneficiaries` and `ma.non_recourse`: a lender clause inside the section also gets `ma.financing.source_protections`. |
| 12 | `ma.rep.no_other_reps` and `ma.rep.non_reliance` were swapped. | Both cards state the speaker test, and that one section often holds both statements. |
| 13 | Readers gave a party on 740 labels whose concept has no party role. | `obligor` on `ma.covenant.other`, `ma.efforts.general`, `ma.employee.comp_benefits_continuation` and `ma.financing.buyer_covenant`. |

One proposal of the report was not applied to the packs: canonical terms of their own for the assignable groups. A navigation node has no terms in the packs. The labeling code builds the terms of a group from its wave 2 concepts, and that code is the place for the change.

**Prevalence of the new concepts.** Agreements of 12,198, from `prevalence.py`. 1% is 122 agreements. The text patterns are unreviewed, as in 4.4.

| Concept | Heading | Text | Wave |
|---|---|---|---|
| `ma.separation.shared_contracts` | 335 | 498 | 1 |
| `ma.environmental.remediation` | 238 | 701 | 1 |
| `ma.bankruptcy.court_approval` | 232 | 643 | 1 |
| `ma.rep.communications` | 154 | 596 | 1 |
| `ma.separation.insurance_access` | 57 | 604 | 2 |

`ma.other.lead_in` has no heading and no text pattern. It is in wave 1 with the catch-alls.

**Topics measured and not added.** Heading counts in agreements.

| Topic | Heading | Where it goes | Reason |
|---|---|---|---|
| Gaming representations | 48 | `ma.rep.other` | Below the 1% bar. |
| Education representations (Title IV, accreditation) | 17 | `ma.rep.other` | Below the bar. |
| Franchise system representations | 86 | `ma.rep.other` | Below the bar. |
| Transportation representations (aviation, vessels, motor carriers) | 119 | `ma.rep.other` | Below the bar. |
| Government grants and subsidies representations | 155 | `ma.rep.other` | Above the bar, but the trial saw it in one agreement. See open question 18. |
| Relief-program loans (PPP, CARES Act) representations | 166 | `ma.rep.other` | The same. |
| SPAC "no trading" covenant | 315 | `ma.covenant.other` | Above the bar, but a standard clause that answers no research question. See open question 18. |
| SPAC covenant to keep SEC reports current | 402 | `ma.covenant.other` | The same. |
| Officer's certificate as a closing condition | 310 | `ma.condition.other` | The same. It is the proof of the bring-down and covenant conditions, which have their own concepts. |
| Plan or scheme of arrangement: interim order, final order | 189 | `ma.covenant.other`, `ma.condition.other` | Canadian and UK-style law. The focus is US agreements (open question 6). |

**Title rules.** Ten rules were added. Five of them fix headings that had gone to the wrong concept: "PPP Loan" to the banking representation, "Communications Regulatory Matters" to the healthcare representation, "Shared Contracts" to non-assignable assets, "SPAC Public Filings" to public announcements, and "Remediation" to public announcements (the pattern "media" matched it). Section 6.2 gives the new coverage figures.

**Validator and tests.** `taxonomy_pack.wave_one_label` gives the label that a wave 1 reader uses for any M&A concept, and `taxonomy_pack.wave_two_parts` lists the parts of each wave 1 concept. The validator uses both for the rule of row 2. 80 tests (was 66).

**Tax pack.** Three leaves were added and three topics got a cue. See 8.4.

## 12. What this work did not do

- **No full labeling.** The coverage figure rests on titles. The 50-agreement trial (11.4) is the only check on body text. Its reference is one blind read by a stronger model. It is not a gold set.
- **No expert review by a person.** The tax professor has not seen the 10 additions (accepted by the maintainer) or the new descriptions. No M&A lawyer has reviewed the M&A pack. Three model reviews are recorded in sections 4.4 and 11.
- **No second reader** for the sample in 6.3.
- **No verified prevalence.** Section 4.4 gives a heading count and a text count for each M&A concept. The text patterns are hand-written and nobody checked samples of their matches. The tax pack was not measured.
- **No database tables.** The plan's `taxonomy_concept` tables are not created. The YAML is the source of truth, and a loader into the database is a later step.
- **Nothing under `backend/` or `frontend/` changed.**

## 13. Sources

- The span taxonomy plan on the branch of draft PR 317 (https://github.com/PlatosTwin/pandects-app/pull/317): sections 3, 5, 7, 12, 14 and 15.
- The live trees in the local database (`taxonomy_l1`, `taxonomy_l2`, `taxonomy_l3`, `tax_clause_taxonomy_l1` to `l3`), read only.
- `etl/src/etl/models/taxonomy/deepseek/taxonomy_enrichment.yaml`: the 140-leaf redesign.
- The corpus: `latest_sections_search` and `sections`, read only.
