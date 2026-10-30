# Reader-index gold labeling rubric (v1.5)

Used to label the gold set in docs/plans/smart-reader.md Section 9. A drafter labels one packet; a checker reviews the draft. Label what a careful lawyer reading the filing would say, **not** what a regex would find. You never see the enricher's output.

## The packet

- **Outline:** `id | title` for each article and section in the filing, and any attachment headings found in back matter.
- **Paragraphs:** `[P12] text`, grouped under the section or region they belong to. Quote paragraph ids exactly.
- **Excerpts:** a definitions excerpt (up to 40 paragraphs) and 4 sampled body sections.

## Target ids

- Article: `a` + the arabic number (`Article IV` → `a4`; `ARTICLE 8` → `a8`).
- Section: `s` + the normalized number. Drop leading zeros (`7.02` → `s7.2`) and keep letters (`s5A`).
- Subsection: append the enumerators exactly as printed (`s7.2(b)(ii)`). Use this even if you cannot see the target paragraph.
- Attachment: `x-exhibit-b`, `x-schedule-2`, `x-annex-i`, `x-appendix-a`, all lowercase. Use this **only** when the outline lists that attachment's heading; an exhibit or annex that is mentioned but not filed is `not_in_filing`. *(ruling 2026-10-23)*
- When there is no target in this filing, use one of these instead of an id:
  - `external`: a statute, regulation, rule, SEC form or schedule, or a section of another document (credit agreement, voting agreement, charter, bylaws, "the Original Agreement"). "Section 409A of the Code" is external. A defined term named like a head word and number ("the Schedule 13E-3") is labeled as an `external` xref at each mention. *(v1.5)*
  - `disclosure_schedule`: a section of a disclosure letter or schedule ("Section 3.4 of the Company Disclosure Letter", "Section 3.2(a) of the Company Schedules"), or a "Schedule 12.2" whose number mirrors the agreement's sections, as disclosure schedules do., or a "Schedule 1" that is itself the disclosure schedules. *(rulings 2026-10-23)*
  - `attachment_local`: a paragraph or section inside an exhibit or schedule, referred to by its own numbering ("paragraph 3 of Schedule 2").
  - `inline`: a clause or item enumerated inline within the same sentence or paragraph ("clauses (i) and (ii) above", meaning items in the same paragraph), and a clause or paragraph of a definition ("clauses (a) through (c)" of the Marketing Period definition, "clause (f) of the definition of Indebtedness"), even when the definition runs over several paragraphs. *(v1.4)*
  - `not_in_filing`: an internal reference whose target does not exist in the outline (for example, a section number missing from the filing), and an attachment whose heading is marked "[omitted]". *(v1.4)*
- **Drafting typos:** label the number as printed, even when another target was obviously meant ("this Section 4.04" inside Section 4.03 is `s4.4`; "Sections 12(a) and 12(b)" with no Section 12(a) is `not_in_filing`). *(v1.4)*

## What to label

### 1. Declarations (`declarations`)

Every term declared in the definitions excerpt, and every parenthetical or inline definition in the 4 sampled sections:

- `"Acquisition Proposal" means...` and `Acquisition Proposal. "Acquisition Proposal" shall mean...` → kind `means`.
- `(the "Company")`, `(each, a "Party")` → kind `paren`.
- `"Closing Date" has the meaning set forth in Section 2.2` → kind `pointer`, with `pointer` set to the target id (`s2.2`). A pointer to the preamble or recitals ("has the meaning set forth in the introductory paragraph") has `pointer` set to `preamble`. *(v1.3)* A pointer target is always a target id or a no-target value from the list above, never a paragraph id like `P18`. When a pointer names another term ("has the meaning given in the definition of Environmental Laws"), point to the section holding that definition. *(ruling 2026-10-23)* A pointer to another document ("has the meaning given in the Confirmations") is `external`. *(v1.5)* A TERM/SECTION index table (terms listed with the section that defines them, without quotes or "means") is a list of pointer declarations, one per row, with `pointer` set to the row's section. *(v1.5)*
- The term is the text inside the quotes, exactly as printed (case, punctuation), except that a period or comma just inside the closing quote is dropped unless it ends an abbreviation ("U.S.", "Vandalia Corp."). *(v1.4)* If one parenthetical or one definition declares two terms (`("Parent" or "Buyer")`, `"Tax" or "Taxation" means`), list two entries. *(v1.3)*
- Do not label quoted words that are not declarations (`the word "including"`, titles of other documents, a statute's official title after "commonly referred to as").
- A term starts with a capital letter, a digit or a symbol such as `$`. Lowercase correlatives ("control", "controlled", "controlling", "to Control") are not declarations. *(ruling 2026-10-23, v1.5)* Capitalized correlatives ("Controlled", "Controlled by", "Controlling" in "shall have correlative meanings") are, with kind `means`. *(v1.5)* A lowercase-first acronym with an internal capital (`"cGMP" means`) is a declaration. *(v1.3)*
- Single-letter formula variables (`Where "N" is the aggregate amount...`) are not declarations. *(ruling 2026-10-23)*

### 2. Cross-references (`xrefs`)

Every reference mention in the 4 sampled sections and the definitions excerpt that uses one of these head words: Section(s), Article(s), clause(s), paragraph(s), Exhibit(s), Schedule(s), Annex(es), Appendix/Appendices. Other words ("subsection", "subparagraph", "subclause", "para", "Item", "Rule") are not head words. *(v1.4)*

- **Lists and ranges:** label one entry per printed number. "Sections 3.1, 3.2 and 3.4" gives three entries; "Sections 3.1 through 3.4" gives two (3.1 and 3.4); "Section 7.6(b), (c)" gives two (`s7.6(b)`, `s7.6(c)`).
- **Fields:**
  - `para`: the paragraph id.
  - `mention`: the number as printed, without the head word (`7.2(b)`, `IV`, `B`, `(iii)`).
  - `context`: 4 to 10 words copied **exactly** from the paragraph, containing the mention once.
  - `head`: lowercase singular head word (`section`, `article`, `clause`, `paragraph`, `exhibit`, `schedule`, `annex`, `appendix`).
  - `target`: an id or one of the no-target values above.
- **Rules for the target:**
  - Qualified clauses: "clause (ii) of Section 4.2" → `s4.2(ii)`, and the "Section 4.2" it names gets its own entry (`s4.2`). *(v1.3)*
  - Local clauses: "clause (a) above" referring to subsection (a) of the current section → `s{current}(a)`. If it refers to an item enumerated inline in the same paragraph, use `inline`.
  - "of this Agreement" and "hereof" mean this filing. In an exhibit that is the agreement's own glossary, a bare "Section 5.4" means the agreement's Section 5.4.
  - UK drafting: "clause 9.4" is a section (`s9.4`), and "clause 5" is the top-level clause (`a5`).
  - "Section 8" in an agreement whose top level is "SECTION 8 ..." → `a8`.
  - Do not label "this Agreement", "herein" or "hereof" on their own; there is no head word and number.

### 3. Term uses (`term_uses`)

Pick 5 terms declared in the definitions excerpt that are **used at least once in the sampled sections**: go through the excerpt in order and take the 1st, 3rd, 5th... qualifying term until you have 5. If that gives fewer than 5, fill up with the skipped qualifying terms in order; if fewer than 5 qualify at all, take every one. *(v1.3)* For each, count its uses in each of the 4 sampled sections. (v1 took the 1st, 5th, 10th... terms, which left 3 of 5 pilot agreements with no term-use labels.)

- A use is the defined term used in its defined sense. Plurals and possessives count ("Acquisition Proposals", "Parent's"), and so does the singular of a term declared in the plural ("any Company" for "Companies"). *(v1.5)* Capitalized as defined: a lowercase "person" is not a use of an all-caps PERSON. *(v1.5)*
- Not a use: an all-caps run of text ("THIS AGREEMENT" in a legend) *(v1.4)*, the declaration itself, a lowercase ordinary-English use ("the law"), part of a longer defined term ("Parent" inside "Parent Material Adverse Effect" when that is its own defined term), or text inside quotes.
- Fields: `term`, `para`, `count`. Omit paragraphs with a count of 0.

## Output

Write a single JSON file:

```json
{
  "agreement_uuid": "...",
  "rubric": "v1.5",
  "declarations": [{"term": "...", "para": "P3", "kind": "means|paren|pointer", "pointer": "s2.2"}],
  "xrefs": [{"para": "P7", "head": "section", "mention": "7.2(b)", "context": "subject to Section 7.2(b) and the", "target": "s7.2(b)"}],
  "term_uses": [{"term": "...", "para": "P20", "count": 2}],
  "notes": "anything ambiguous, briefly"
}
```

## Checker

The checker reads the same packet and the draft, then writes:

```json
{
  "agreement_uuid": "...",
  "disagree": [{"section": "xrefs", "index": 4, "fix": {"target": "external"}, "reason": "..."}],
  "remove": [{"section": "declarations", "index": 9, "reason": "..."}],
  "missing": [{"section": "xrefs", "item": {"para": "...", "...": "..."}, "reason": "..."}],
  "notes": "..."
}
```

- `index` is the 0-based position in the draft's list.
- Check every draft item. List only the ones you disagree with: a wrong field goes under `disagree` with the corrected fields in `fix`, and an item that should not exist goes under `remove`. Every draft item you do not list counts as agreed.
- List anything the drafter missed under `missing`.
- Keep reasons to one line.
