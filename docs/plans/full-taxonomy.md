# Full taxonomy for M&A and tax provisions: design

Status: draft for the maintainer's review, 2026-11-22. Nothing in this document is live. No paid model call was made for this work, and no row in any database was changed.

This document belongs to the span taxonomy plan (`docs/plans/span-taxonomy.md`, draft PR 189, not yet on main). That plan tested ten value concepts in depth. This document gives the breadth step: the full list of provisions, for both subject areas, before any paid labeling run.

## 1. Summary

- **What exists now.** Two sets of taxonomy files, called packs, are in `etl/taxonomy/`. The M&A pack (`ma`) has 15 branches, 55 groups and 276 concepts. The tax pack (`tax`) has 10 branches, 35 groups and 112 concepts. A validator and 24 tests check them without network or database access.
- **A concept is one thing a researcher filters on.** `ma.governing_law` answers "which law governs this agreement". It does not also hold the forum clause or the jury waiver. Each of those has its own concept.
- **The tax tree keeps its structure.** All 150 nodes of the tax clause tree are present with the same labels and the same parents. This design proposes 7 additions and no removals, renames or merges. Section 8 lists each addition with its reason, so that the maintainer can veto it.
- **Party is an attribute.** "Organization" as a representation is one concept. Each labeled instance records who made it. The older trees held a target copy and a buyer copy of each representation.
- **Coverage by section title.** Of 1,138,095 titled sections in the corpus, 84.3% map to one specific concept and a further 9.4% map to a group that holds several concepts by design. 4.2% map to nothing. A sample of those shows that most are other wordings of provisions the tree already has, or headings that say too little to place.
- **Old filters keep working.** Each of the 99 leaves of the live tree, each of the 140 leaves of the unreleased redesign and each of the 150 tax nodes maps to at least one new node.
- **Labels must come from body text.** 155 of the 388 concepts never appear as a section title. Section 9 describes the labeling run and gives a cost estimate: roughly $350 to $800 for a first full pass, with a small model for the first read and a stronger model for the hard cases. That figure is an estimate from list prices. A billed trial on 50 agreements must replace it before any full run.

## 2. Files

| Path | Contents |
|---|---|
| `etl/taxonomy/ma/*.yaml` | The M&A pack, one file per branch (15 files). |
| `etl/taxonomy/tax/*.yaml` | The tax pack, one file per branch (10 files). |
| `etl/taxonomy/legacy/*.yaml` | Read-only snapshots of the ids of the three older trees. The validator checks crosswalks against them. |
| `etl/taxonomy/crosswalk/dealos.yaml` | 132 links from concepts to DealOS knowledge graph ids. |
| `etl/taxonomy/README.md` | Field reference and the rules for changes. |
| `etl/src/etl/domain/taxonomy_pack.py` | Loader and validator. Run `etl/.venv/bin/python -m etl.domain.taxonomy_pack`. |
| `etl/tests/test_taxonomy_pack.py` | 24 tests. Run from `etl/` with `.venv/bin/python -m unittest tests.test_taxonomy_pack`. |
| `docs/plans/full-taxonomy/tree.md` | Every node, generated from the packs. |
| `docs/plans/full-taxonomy/crosswalk.md` | Every older leaf and the nodes it maps to, generated from the packs. |
| `docs/plans/full-taxonomy/title_coverage.py` | The coverage measurement in section 6. |
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
| `provision` | A provision that answers one research question. | 238 | 109 |
| `component` | A part of a provision that researchers study by itself. It names its parent concept. | 18 | 0 |
| `definition` | A defined term whose content is a research object. | 17 | 3 |
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

The role names are `maker` (of a representation), `obligor` (of a covenant), `tested_party` (of a condition), `holder` (of a right), `indemnitor` and `subject` (of a definition). 82 M&A concepts and 22 tax concepts carry a role.

The reasons:

1. A researcher who wants "all Organization representations" gets them with one filter. A researcher who wants only the target's adds one attribute filter.
2. Private deals have more than two sides: sellers, the company, the buyer, a merger subsidiary, a sponsor. A copy of the subtree per party does not scale.
3. In the Praxis classification work, confusion between the two parties' copies of the same provision was the most common error. With one concept, the model picks the topic, and a separate check with simple rules sets the party from the article heading.

### 3.5 Catch-all concepts

Each large family has an explicit catch-all: `ma.rep.other`, `ma.covenant.other`, `ma.condition.other`, `ma.termination.right.other`, `ma.misc.other`, `ma.other.reserved` (for "[Reserved]" and "Intentionally Omitted") and `ma.other.unclassified`. A catch-all makes the unassigned share measurable. When a topic grows inside a catch-all, a later release can give it its own concept.

The tax pack follows the present tax labeler instead. That labeler may assign a clause to a level 1 or level 2 node when no leaf fits. The tax navigation nodes are therefore marked `assignable: true`, and the tax pack keeps its three existing "Other" leaves.

### 3.6 The seam between the packs

Some provisions belong to both subject areas. Each has one owner pack, and the other pack lists it. `tree.md` ends with the full seam table (20 rows).

- **Tax owns** the tax representations, elections, purchase price allocation, withholding, transfer taxes, the tax opinion condition, pre-closing restructuring, tax covenants before closing, the tax indemnity and the tax treatment of indemnity payments. A section titled "Taxes" in the representations article gets the tax node `tax.nav.reps`, which the M&A tree lists under "Financial" representations.
- **M&A owns** the Section 280G stockholder vote (`ma.employee.section_280g`, listed under `tax.nav.other`) and the Section 409A and plan qualification statements, which are part of `ma.rep.employee_benefits`. These are compensation provisions that employment and benefits lawyers draft. Open question 2 asks the maintainer to confirm this.

### 3.7 Fields

Each concept has an id, a label, a kind, a navigation parent, a description, canonical terms, example phrases, exclusion cues, a crosswalk to older ids and the release that introduced it. `etl/taxonomy/README.md` defines each field.

Three points need a note here:

- **Exclusion cues name the right concept.** "The court that hears disputes is `ma.forum_selection`." The validator checks that each id named in a cue exists.
- **Value schemas.** The 11 pilot concepts keep their tested fields (`status: piloted`) and are the only concepts with `tier: value`. A further 71 M&A concepts carry a `status: stub` schema: a first sketch of fields for a later wave, which nobody has tested.
- **Example phrases are illustrations.** They show typical wording. They are not verified quotations from the corpus. No text was copied from the DealOS knowledge graph or from the Praxis repository. Those two sources were used for vocabulary and lessons only.

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
| Termination rights | The live tree has one leaf for all termination events. | 10 concepts: mutual consent, outside date, legal restraint, vote failure, breach, change of recommendation, superior proposal, failure to close, other, and the notice of termination. |
| Termination fees | One older leaf joined the target fee, the reverse fee and expenses. | 5 concepts. Three of them are pilot value concepts. |
| No-shop and fiduciary out | "Superior Proposal" appears in 37.1% of agreements and "Intervening Event" in 19.0%. | Separate concepts for the non-solicitation covenant, the go-shop, the exception for unsolicited proposals, notice of proposals, and each ground for a change of recommendation. |
| Negative covenants before closing | The restricted actions (dividends, new debt, capital spending and others) are studied one by one. | One provision with 10 components. |
| Defined terms | The content of a few definitions drives the deal terms. | 17 definition concepts, such as Knowledge, Willful Breach, Fraud, Indebtedness and Working Capital. All other defined terms fall under `ma.definitions.general`. |
| Representations | 48 topics, each with a party role. | 48 concepts in place of two copies of each. |
| Collar and walk-away right | The title "collar" occurs once. "Walk away" appears in 0.6% of agreements. | Value fields of `ma.consideration.exchange_ratio`. |
| Locked box | About 1% of agreements, in UK-style deals, and the term collides with "lockbox" bank accounts. | No concept. US agreements are the focus. |
| SPAC deals | "Trust Account" appears in 9.8% of agreements, a minimum cash term in 3.5%, a PIPE in 3.0%. | 9 concepts in their own group. |
| Asset deals | Asset purchases hold about 11% of titled sections. | 8 concepts for purchased and excluded assets, assumed and excluded liabilities and related mechanics. |
| Industry representations | Banking, healthcare, insurance and energy deals carry long representations with no general equivalent. | 4 industry concepts and `ma.rep.other`. See open question 4. |

Five concepts were added because the title data showed a gap in the first draft: `ma.rule_145_affiliates` (titles in 981 agreements), `ma.asset.title_and_survey` (594), `ma.spac.business_activities` (520), `ma.conduct.integration_planning` (483) and `ma.spac.domestication` (172).

Four topics were left to catch-alls because they are rare: the SPAC "no trading" covenant (317 sections), comfort letters (114), employee stock ownership plans (100) and like-kind exchanges (83).

### 4.3 Size compared with the references

| Tree | Leaves or concepts | Note |
|---|---|---|
| Live section tree | 99 | Many leaves join several provisions, and representations are doubled by party. |
| Unreleased redesign | 140 | Finer, still doubled by party. |
| This design, M&A pack | 276 | 238 provisions, 21 parts and 17 definitions. |
| This design, tax pack | 112 | 106 existing nodes and 6 proposed concepts. |
| DealOS knowledge graph v6 | 993 | Built to explain mechanics to lawyers. It includes explanatory nodes that a filter does not need. |

The Praxis classification work found that accuracy falls as the number of classes in one decision grows. This design is finer than the older trees, so the labeling run in section 9 never shows a model the full list. It shows a short list for the section at hand.

## 5. The trees

`tree.md` lists every node. This section gives the counts.

### 5.1 M&A pack

| Level | Count |
|---|---|
| Branches | 15 |
| Groups | 55 |
| Concepts directly under a group | 255 (238 provisions, 17 definitions) |
| Parts of a concept | 21 (18 components, 3 modifiers) |
| All concepts | 276 |

| Branch | Groups | Concepts | Groups (concepts in each) |
|---|---|---|---|
| Deal Structure and Closing | 5 | 34 | The Transaction (7); Asset Deal Scope (8); Tender Offer (7); Closing (3); SPAC Transactions (9) |
| Consideration and Payment Mechanics | 4 | 21 | Price and Form of Consideration (10); Equity Awards and Other Securities (3); Exchange and Payment Procedures (4); Purchase Price Adjustment (4) |
| Representations and Warranties | 10 | 48 | Fundamental (7); Financial (7); Legal and Compliance (4); Commercial (6); Assets, IP and Environmental (5); Employees (2); Transaction-Specific (4); Buyer-Specific (6); Disclaimers (2); Industry-Specific (5) |
| Conduct of Business and Access | 2 | 20 | Interim Operations (15); Information, Confidentiality and Publicity (5) |
| Deal Protection and Stockholder Approval | 4 | 16 | No-Shop and Go-Shop (5); Board Recommendation and Fiduciary Out (4); Stockholder Approval Process (5); Takeover Statutes and Rights Plans (2) |
| Efforts and Regulatory Approvals | 3 | 9 | Efforts to Close (2); Antitrust (5); Other Regulatory Approvals (2) |
| Financing | 3 | 9 | Financing Arrangements (4); Financing Covenants (4); Financing Source Protections (1) |
| Employees and Benefits | 2 | 8 | Continuing Employees (4); Transaction-Related Employee Matters (4) |
| Other Covenants | 4 | 19 | Directors, Officers and Governance (5); Securities, Listing and Deal Litigation (4); Post-Closing and Separation (10); Tax Matters (lists tax pack nodes only) |
| Conditions to Closing | 3 | 13 | Mutual Conditions (4); Conditions for One Party (8); Condition Mechanics (1) |
| Termination | 3 | 17 | Termination Rights (10); Effect of Termination (2); Termination Fees and Expenses (5) |
| Indemnification and Risk Allocation | 4 | 17 | Survival and Scope (3); Limits on Recovery (7); Claims and Sources of Recovery (6); R&W Insurance (1) |
| Dispute Resolution and Remedies | 2 | 9 | Governing Law and Forum (4); Remedies and Enforcement (5) |
| Definitions and Interpretation | 3 | 22 | Defined Terms (1); Key Defined Terms (18); Interpretation (3) |
| General Provisions | 3 | 14 | The Agreement (8); Costs and Parties (3); Other (3) |

### 5.2 Tax pack

| Level | Existing tree | This design |
|---|---|---|
| Level 1 | 9 | 10 branches (9 existing, 1 proposed) |
| Level 2 | 45 | 35 groups and 10 concepts. The 10 are existing level 2 nodes that have no children. |
| Level 3 | 96 | 96 existing concepts and 3 proposed |
| Under the proposed branch | 0 | 3 proposed definition concepts |
| All concepts | 106 leaves | 112 |

| Branch | Groups | Concepts |
|---|---|---|
| Intended Tax Treatment, Elections, and Purchase Price Allocation | 6 | 18 |
| Tax Interim Operating Covenants | 0 | 9 |
| Transfer and Withholding Taxes | 3 | 7 |
| Tax Cooperation, Returns, Contests, and Refunds | 5 | 18 (2 proposed) |
| Tax Allocation and Indemnity | 3 | 15 |
| Special Regimes | 3 | 12 (1 proposed) |
| Tax Representations | 9 | 22 |
| Tax Attributes | 5 | 7 |
| Other / Unclassified | 1 | 1 |
| Tax Definitions (proposed) | 0 | 3 (all proposed) |

## 6. Coverage measurement

### 6.1 Method

The question: what share of the sections in the corpus map cleanly to a leaf?

The corpus is the table `latest_sections_search`: 12,198 agreements and 1,196,581 sections, read on 2026-11-22 with one read-only query. 58,486 sections (4.9%) have no title. The rest carry 133,055 distinct titles after normalization.

`docs/plans/full-taxonomy/title_coverage.py` holds 377 rules. A rule is a pattern for a title, sometimes limited to an article type. "Insurance" under a representations article maps to `ma.rep.insurance`, and "Insurance" under a covenants article maps to the directors' and officers' insurance covenant. Each rule names one target: a concept, or a navigation node when a section with that title normally holds several concepts.

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
| leaf | 959,051 | 80.1% | 84.3% |
| group | 106,746 | 8.9% | 9.4% |
| partial | 8,585 | 0.7% | 0.8% |
| catch-all | 15,629 | 1.3% | 1.4% |
| untitled | 58,486 | 4.9% | |
| unmapped | 48,084 | 4.0% | 4.2% |

**84.3% of titled sections map cleanly to a leaf.** With the group class, 93.6% of titled sections have a home in the tree.

The group class is expected. A section on the conditions to one party's obligations (22,984 sections) holds several conditions. "Taxes" in a representations article (15,090 sections) is the tax umbrella that the tax pass refines. "Termination" (13,683 sections) lists every termination right. These sections need labels at the clause level, below the section.

Coverage holds across deal types. The share of titled sections in the leaf or group class:

| Deal type | Target | Sections | Leaf or group |
|---|---|---|---|
| Merger | public | 321,972 | 96.2% |
| Merger | private | 299,289 | 94.1% |
| Tender offer | public | 64,357 | 97.2% |
| Stock acquisition | private | 198,668 | 91.0% |
| Stock acquisition | public | 14,678 | 86.6% |
| Asset acquisition | private | 117,547 | 90.1% |
| Asset acquisition | public | 9,452 | 89.9% |
| Membership interest purchase | private | 111,267 | 92.7% |

Private stock and asset deals score 5 to 6 points lower than public mergers. Their titles vary more, and they carry more deal-specific sections.

### 6.3 What is in the unmapped 4.2%

The unmapped and partial sections carry 27,981 distinct titles. The most frequent one occurs 160 times, so the group is a long tail and has no large missing topic.

The script draws a random sample with a fixed seed (`--unmapped 150`). One reader sorted the 150 titles by eye, with no second review:

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
- Industry topics and uncovered general topics are the real gaps. They are 10% of the sample. The sample comes from the unmapped and partial classes, which are 5.0% of titled sections, so the real gaps are about 0.5% of titled sections.

### 6.4 Limits of this measurement

- A title is a weak signal. A section titled "Governing Law" can also hold the forum clause. The measurement shows that the tree has a place for the section, and it does not show that a label would be right.
- 233 of the 388 concepts are the target of at least one title rule. The other 155 never appear as a title: components, definitions, clauses inside a larger section, and most tax leaves. Title coverage says nothing about them.
- The rules were tuned on this corpus. The figures describe fit to this corpus and are not a forecast for new filings.
- Agreements under UK or Australian law are in the corpus and were not removed. They lower the figures slightly.

## 7. Crosswalk

`crosswalk.md` lists every older leaf with its new nodes. The relation describes the new node compared with the older one: `exact` (same scope), `narrower` (the older leaf was split), `broader` (older leaves were merged), `related` (overlap).

| Older tree | Leaves | Mapped | One exact match | Split into narrower concepts | Merged into a broader concept |
|---|---|---|---|---|---|
| Live section tree (12 / 61 / 99) | 99 | 99 | 56 | 41 | 2 |
| Unreleased redesign (140 leaves) | 140 | 140 | 100 | 40 | 0 |
| Tax clause tree (9 / 45 / 96) | 106 leaves, 150 nodes | 150 | 150 | 0 | 0 |

- **Party copies.** A target leaf and a buyer leaf of the same representation both map to one concept as `exact`, each with `party: target` or `party: buyer`. The merge of the two copies is not counted as "broader", because the party survives on the instance.
- **Splits.** A split leaf maps to 2 to 8 concepts in the live tree and to 2 to 12 in the redesign. The widest split is the conduct of business leaf, which maps to the ordinary course covenant, the negative covenants and their 10 components.
- **Merges.** Two live leaves on protections for lenders map to the one concept `ma.financing.source_protections`.
- **New concepts.** 95 M&A concepts have no older leaf. They include the 9 SPAC concepts, most of the definitions, covenants after closing such as the non-compete and the release, and 19 representations. `crosswalk.md` lists them.
- **How to use it.** An old filter on a live leaf becomes a filter on the set of new concepts that the leaf maps to. A new label on a narrower concept can also be shown under its old leaf while the old tree is still served.
- **DealOS.** `crosswalk/dealos.yaml` holds 132 links (78 exact, 37 narrower, 10 broader, 7 related). It is a separate file so that a public dump can leave it out. The span taxonomy plan leaves that choice open.

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

### 8.3 Proposed additions

Each node below is marked `proposed: true` in the YAML. To veto one, delete it from its file.

The evidence comes from the 243 agreements that have tax clause labels today, and from section titles across the corpus.

| # | Proposed node | Where | Reason and evidence |
|---|---|---|---|
| 1 | `tax.nav.definitions`, "Tax Definitions" (a new level 1 branch) | Top level | The tree has no place for tax definitions. They appear in 65 of the 243 labeled agreements. 59 of those clauses have no label and 49 sit in "Other / Unclassified". |
| 2 | `tax.def.tax`, "Tax" / "Taxes" definition | Under 1 | The scope of "Taxes" sets the scope of the tax indemnity and the tax representations. Researchers compare it. |
| 3 | `tax.def.tax_return`, "Tax Return" definition | Under 1 | It is the usual companion of 2. It can be dropped if the maintainer wants a smaller branch. |
| 4 | `tax.def.tax_period`, tax period definitions (Pre-Closing Tax Period, Straddle Period) | Under 1 | These terms set the line between the seller's and the buyer's tax years. The straddle rules in `tax.nav.indemnity` depend on them. |
| 5 | `tax.international.transfer_pricing`, "Transfer Pricing" | Special Regimes, International Tax | "Transfer pricing" appears in 12.4% of all agreements. In the labeled set it appears in 27 agreements, with 11 clauses in "Other" and 10 without a label. The International Tax group has no node for it. |
| 6 | `tax.cooperation.post_closing_actions`, "Restrictions on Post-Closing Tax Actions" | Cooperation, Returns, Contests, and Refunds; Post-Closing Cooperation and Records | The covenant that the buyer will not amend returns or make elections for pre-closing periods. It appears in 44 labeled agreements and is now forced into the interim covenant nodes, which cover the period before closing, or into return preparation. 173 sections carry it as a title. |
| 7 | `tax.cooperation.tax_sharing_termination`, "Termination of Tax Sharing Agreements" | Same group as 6 | The covenant that tax sharing agreements end at closing. 624 sections in 621 agreements carry it as a title. The labeled set shows it in 30 agreements, where it is mostly labeled as the tax sharing representation (`tax.7.3.2`), which is a different statement. The evidence is weaker than for 5 and 6. |

### 8.4 A change of scope outside the tree

466 clauses in 178 of the labeled agreements concern Section 409A or Section 280G. The tax module picks them up today. This design assigns them to the M&A pack (see 3.6) and lists the 280G concept under `tax.nav.other`. This changes which pack labels them and changes no tax node. It is open question 2.

## 9. How labeling would run

No labeling ran for this work. This section is a plan.

### 9.1 Why body text

- 155 of the 388 concepts never appear as a title.
- 4.9% of sections have no title, and about 1 in 10 titled sections sits in the group class and holds several concepts.
- A title names the main topic and misses the rest. The governing law section often holds the jury waiver.
- The title rules of section 6 serve only as a starting hint and as a check on the result.

### 9.2 Steps

1. **Unit.** The unit is the section, cut into paragraphs with character offsets, as in the text layer of the span taxonomy plan. A label instance is one concept on one span of text, with the party, a confidence and the model's evidence quote. A section can carry several instances.
2. **Short list (no model cost).** For each section, build a short list of 15 to 30 candidate concepts from the article heading, the title hint, the canonical terms found in the body and the defined terms the section uses. Embeddings can be added. The Praxis work measured 99.3% recall with 50 candidates. The goal here is the same recall with fewer.
3. **Classification on body text.** A small model reads the section and the cards of its candidates (label, description, terms, exclusion cues). It returns the concepts present, the span for each, and the party. "None of these" is allowed and becomes a catch-all. The model never sees the full list of 388, because accuracy falls as the class count grows.
4. **Party.** Simple rules set the party from the article heading ("Representations and Warranties of the Company") and check the model's answer. A conflict sends the section to step 6.
5. **Tax pass.** A section or clause that receives a tax node goes to a second pass with the tax pack only. That pass assigns the tax leaf, as the tax module does today. The M&A pass does not need the tax leaves, which keeps its short lists small.
6. **Second read.** A stronger model reads the sections that fail a check: low confidence, a conflict between the title hint and the label, a party conflict, or a representation concept outside a representations article.
7. **Parts and definitions (second wave).** Components are labeled only inside sections that already carry the parent concept. Definition concepts come mostly from the defined-term graph: the term string identifies them, so most need no model.
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

What can move the estimate: the share of sections that need a second read, how well the cache works inside a batch, the reasoning effort setting (reasoning tokens bill as output), and retries.

**Before any full run:** a billed trial on 50 agreements, estimated under $10, gives the real cost per agreement and the real second-read share. Each paid run needs the maintainer's go-ahead with the amount stated.

## 10. Open questions for the maintainer

1. **Tax additions.** Approve or veto each of the 7 proposed tax nodes in 8.3. Should the tax professor review them first? My recommendation: keep 1, 2, 4, 5 and 6, and let the professor decide 3 and 7.
2. **Sections 280G and 409A.** This design gives them to the M&A pack. The tax module labels 466 such clauses today. Confirm the owner, or move them to the tax pack.
3. **Umbrella labels.** The tax pack lets a label point to a branch or group. The M&A pack uses catch-all concepts. Is the difference acceptable, or should both packs use one way?
4. **Industry topics.** The M&A pack has 4 industry representation concepts. Industry titles are most of the real gaps in 6.3. Is a later industry pack (banking, healthcare, energy, insurance) wanted, or is `ma.rep.other` enough?
5. **SPAC concepts.** 9 SPAC concepts sit in the M&A pack. Keep them there, or plan a separate pack?
6. **Foreign-law provisions.** Locked box, squeeze-out and similar UK or Australian provisions have no concept, in line with the US focus. Confirm.
7. **Waves.** Should the first labeling run cover provisions only (347 concepts) and leave the 21 parts and 20 definitions to a second wave, as 9.2 assumes?
8. **Defined terms.** Are the 17 M&A definition concepts the right set? Which should get value schemas first?
9. **Value schema stubs.** 71 M&A concepts carry an untested sketch of fields. Keep the stubs in the pack as candidates, or remove them until each is piloted?
10. **DealOS links.** Publish `crosswalk/dealos.yaml` in the public dump, or keep it internal?
11. **Party values.** The role is on the concept and the party is on the instance. Which party values should an instance allow: target, buyer, seller, merger subsidiary, sponsor, mutual?
12. **The unreleased 140-leaf redesign.** This design replaces it. Can the redesign file be retired after approval, with its crosswalk kept?
13. **Id freeze.** After approval the ids are frozen. Any id the maintainer wants changed must change before the first labeling run.
14. **Budget and models.** Which setup in 9.3 should the 50-agreement trial test, and what is the spending ceiling for it?

## 11. What this work did not do

- **No labeling and no model check of the tree.** The coverage figure rests on titles. How well a model can tell neighboring concepts apart on body text is unknown until the trial.
- **No expert review.** The tax professor has not seen the proposed additions or the new descriptions. No M&A lawyer has reviewed the M&A pack.
- **No second reader** for the sample in 6.3.
- **No measured prevalence for concepts that titles do not reach.** The floor in 4.1 was checked with phrase counts where a phrase is distinctive. Short or hyphenated terms could not be counted reliably with the full-text index.
- **No database tables.** The plan's `taxonomy_concept` tables are not created. The YAML is the source of truth, and a loader into the database is a later step.
- **Nothing under `backend/` or `frontend/` changed.**

## 12. Sources

- `docs/plans/span-taxonomy.md` on the branch of draft PR 189: sections 3, 5, 7, 12, 14 and 15.
- The live trees in the local database (`taxonomy_l1`, `taxonomy_l2`, `taxonomy_l3`, `tax_clause_taxonomy_l1` to `l3`), read only.
- `etl/src/etl/models/taxonomy/deepseek/taxonomy_enrichment.yaml`: the 140-leaf redesign.
- The DealOS knowledge graph (v6) and the Praxis background provision classification repository: vocabulary and lessons only. No prose was copied.
- The corpus: `latest_sections_search` and `sections`, read only.
