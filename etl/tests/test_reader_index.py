import unittest
from typing import Any

from etl.domain.reader_index import ENRICHER_VERSION, build_reader_index, text_hash


def _doc(body: str, back: str = "", front: str = "") -> str:
    return (
        '<document uuid="doc-1"><metadata><url>x</url></metadata>'
        f"<frontMatter>{front}</frontMatter><body>{body}</body><backMatter>{back}</backMatter></document>"
    )


def _article(uuid: str, title: str, inner: str) -> str:
    return f'<article uuid="{uuid}" title="{title}">{inner}</article>'


def _section(uuid: str, title: str, *texts: str) -> str:
    return f'<section uuid="{uuid}" title="{title}">' + "".join(f"<text>{t}</text>" for t in texts) + "</section>"


def _xrefs(p: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {x["text"]: x for x in p["xrefs"]}


def _term(p: dict[str, Any], term: str) -> dict[str, Any]:
    return next(t for t in p["terms"] if t["term"] == term)


BASIC = _doc(
    _article(
        "a1",
        "ARTICLE I DEFINITIONS",
        _section(
            "s11",
            "Section 1.1 Definitions",
            "“Acquisition Proposal” means any offer.",
            "“Company Material Adverse Effect” has the meaning set forth in Section 3.1.",
        ),
    )
    + _article(
        "a3",
        "ARTICLE III REPRESENTATIONS",
        _section(
            "s31",
            "Section 3.01 Organization",
            "(a) Parent (“Parent”) is duly organized.",
            "(b) The Company is in good standing.",
            "(i) first item.",
            "(ii) second item.",
        )
        + _section(
            "s32",
            "Section 3.2 Authority",
            "Subject to Section 3.1(a), Section 3.1(b)(ii) and Section 3.1(g), Parent may act. "
            "See Article III and Sections 3.1 and 3.2. Under Section 409A of the Code, "
            "Section 4.4 of the Company Disclosure Letter and Exhibit B apply. "
            "Any Acquisition Proposals and each Acquisition Proposal’s terms.",
        ),
    )
)


class OutlineTests(unittest.TestCase):
    def test_articles_sections_and_subsections(self) -> None:
        p = build_reader_index(BASIC, xml_version=3)
        self.assertEqual(p["enricher"], ENRICHER_VERSION)
        self.assertEqual(p["xml_version"], 3)
        self.assertEqual(p["agreement_uuid"], "doc-1")
        art3 = p["outline"][1]
        self.assertEqual((art3["id"], art3["label"], art3["anchor"]), ("a3", "III", {"s": "a3"}))
        sec = art3["children"][0]
        self.assertEqual(sec["id"], "s3.1")  # 3.01 normalizes to 3.1
        b = next(c for c in sec["children"] if c["id"] == "s3.1(b)")
        self.assertEqual(b["anchor"], {"s": "s31", "t": 1})
        self.assertEqual([c["id"] for c in b["children"]], ["s3.1(b)(i)", "s3.1(b)(ii)"])

    def test_glued_article_title_accepted_only_in_sequence(self) -> None:
        xml = _doc(
            _article("a6", "ARTICLE VI CONDITIONS", "")
            + _article("a7", "ARTICLE VIISURVIVAL", "")
            + _article("a9", "ARTICLE IXTAX", "")
        )
        ids = [n["id"] for n in build_reader_index(xml)["outline"]]
        self.assertEqual(ids, ["a6", "a7"])

    def test_duplicate_section_numbers_are_ambiguous(self) -> None:
        xml = _doc(
            _section("x1", "Section 5.1 One", "a")
            + _section("x2", "Section 5.1 Two", "b")
            + _section("x3", "Section 6.1 Refs", "See Section 5.1.")
        )
        p = build_reader_index(xml)
        x = _xrefs(p)["Section 5.1"]
        self.assertEqual(x["status"], "ambiguous")
        self.assertEqual(x["candidates"], ["s5.1", "s5.1~2"])
        self.assertNotIn("target", x)


class TermTests(unittest.TestCase):
    def test_declarations_and_pointer_demotion(self) -> None:
        p = build_reader_index(BASIC)
        ap = _term(p, "Acquisition Proposal")
        self.assertEqual(ap["decls"], [{"anchor": {"s": "s11", "t": 0}, "span": [1, 21], "kind": "means", "in": "Section 1.1"}])
        parent = _term(p, "Parent")
        self.assertEqual(parent["decls"][0]["kind"], "paren")
        pointer = _term(p, "Company Material Adverse Effect")
        self.assertEqual(pointer["decls"][0]["kind"], "pointer")
        self.assertEqual(pointer["decls"][0]["pointer"], "Section 3.1")

    def test_pointer_dropped_when_real_declaration_exists(self) -> None:
        xml = _doc(
            _section("s1", "Section 1.1 Defs", "“Closing” has the meaning set forth in Section 2.1.")
            + _section("s2", "Section 2.1 Closing", "The closing (the “Closing”) shall occur.")
        )
        decls = _term(build_reader_index(xml), "Closing")["decls"]
        self.assertEqual([d["kind"] for d in decls], ["paren"])

    def test_uses_cover_plural_and_possessive_but_not_the_declaration(self) -> None:
        p = build_reader_index(BASIC)
        tid = _term(p, "Acquisition Proposal")["id"]
        decl_uses = [u for u in p["uses"].get("s11/0", []) if u[2] == tid]
        self.assertEqual(decl_uses, [])
        hits = [u for u in p["uses"]["s32/0"] if u[2] == tid]
        self.assertEqual(len(hits), 2)

    def test_glossary_layout_is_a_declaration_and_its_heading_is_not_a_use(self) -> None:
        back = (
            "<text>EXHIBIT A</text>"
            "<text>Awards. “Awards” is defined in Section 1.1.</text>"
            "<text>Acquisition Proposal. “Acquisition Proposal” shall mean any offer.</text>"
            "<text>Business Day. “Business Day” shall mean a weekday.</text>"
        )
        xml = _doc(_section("s1", "Section 1.1 Awards", "Each award (an “Option”)."), back=back)
        p = build_reader_index(xml)
        ap = _term(p, "Acquisition Proposal")
        self.assertEqual(ap["decls"][0]["kind"], "means")
        self.assertEqual(ap["decls"][0]["anchor"], {"s": "backMatter", "t": 2})
        self.assertNotIn("backMatter/2", p["uses"])
        # the glossary is the agreement's own: its bare "Section 1.1" resolves to the body
        self.assertEqual(_xrefs(p)["Section 1.1"]["target"], "s1.1")

    def test_common_word_terms_are_low_confidence_and_skip_sentence_starts(self) -> None:
        xml = _doc(
            _section(
                "s1",
                "Section 1.1 Defs",
                "“Law” means any statute.",
                "Law shall govern. Each applicable Law applies.",
            )
        )
        p = build_reader_index(xml)
        law = _term(p, "Law")
        self.assertEqual(law["confidence"], "low")
        self.assertEqual(len(p["uses"]["s1/1"]), 1)


class XrefTests(unittest.TestCase):
    def test_statuses(self) -> None:
        x = _xrefs(build_reader_index(BASIC))
        self.assertEqual((x["Section 3.1(a)"]["status"], x["Section 3.1(a)"]["target"]), ("resolved", "s3.1(a)"))
        self.assertEqual(x["Section 3.1(b)(ii)"]["target"], "s3.1(b)(ii)")
        # (g) does not exist: link only the "Section 3.1" part
        self.assertEqual((x["Section 3.1"]["status"], x["Section 3.1"]["target"]), ("section_fallback", "s3.1"))
        self.assertEqual(x["Article III"]["target"], "a3")
        self.assertEqual(x["Sections 3.1"]["target"], "s3.1")
        self.assertEqual(x["3.2"]["target"], "s3.2")
        self.assertEqual(x["Section 409A"]["status"], "external")
        self.assertEqual(x["Section 4.4"]["status"], "disclosure_schedule")
        self.assertEqual(x["Exhibit B"]["status"], "unlocated_attachment")

    def test_offsets_are_utf16_over_decoded_text(self) -> None:
        # "&amp;" decodes to one character; the emoji is two UTF-16 code units
        xml = _doc(
            _section("s1", "Section 1.1 A", "x")
            + _section("s2", "Section 2.1 B", "Buyer &amp; Seller \U0001F600 see Section 1.1.")
        )
        x = _xrefs(build_reader_index(xml))["Section 1.1"]
        rendered = "Buyer & Seller \U0001F600 see Section 1.1."
        start = len(rendered[: rendered.index("Section")].encode("utf-16-le")) // 2
        self.assertEqual(x["span"], [start, start + len("Section 1.1")])
        self.assertEqual(x["at"], {"s": "s2", "t": 0})

    def test_exhibit_local_references_are_not_linked(self) -> None:
        back = "<text>EXHIBIT C</text><text>VOTING AGREEMENT</text><text>As provided in Section 1.1, the Holder votes.</text>"
        xml = _doc(_section("s1", "Section 1.1 A", "x"), back=back)
        x = _xrefs(build_reader_index(xml))["Section 1.1"]
        self.assertEqual(x["status"], "unresolved")
        self.assertNotIn("target", x)

    def test_uk_numbered_clauses_resolve_but_paragraphs_do_not(self) -> None:
        xml = _doc(
            _article("c9", "9 WARRANTIES", _section("s94", "9.4 Limits", "x"))
            + _section("s95", "9.5 Refs", "Subject to clause 9.4 and clause 9, see paragraph 9.4.")
        )
        x = _xrefs(build_reader_index(xml))
        self.assertEqual(x["clause 9.4"]["target"], "s9.4")
        self.assertEqual(x["clause 9"]["target"], "a9")
        self.assertEqual(x["paragraph 9.4"]["status"], "unresolved")

    def test_references_into_a_named_attachment_are_not_linked(self) -> None:
        xml = _doc(_section("s1", "Section 2.1 A", "Under Section 2.1 of Exhibit B and clause 2.1 of Schedule 3."))
        x = _xrefs(build_reader_index(xml))
        self.assertEqual(x["Section 2.1"]["status"], "unresolved")
        self.assertEqual(x["clause 2.1"]["status"], "unresolved")

    def test_gold_pilot_precision_fixes(self) -> None:
        xml = _doc(
            _section("s3", "Section 3.2 Capital", "(a) shares.", "(b) options.")
            + _section(
                "s9",
                "Section 9.1 Defs",
                "“Excluded” means (i) any tax, (ii) any fee and (iii) any cost, other than in clauses (i) and (iii) hereof.",
                "As set forth in Section 3.2(a) of the Company Schedules and Section 3 of the Spanish Civil Code.",
            ),
            back="<text>Annex 3.7</text><text>Earn-out terms.</text>",
        )
        x = _xrefs(build_reader_index(xml))
        self.assertEqual(x["clauses (i)"]["status"], "unresolved")
        self.assertEqual(x["Section 3.2(a)"]["status"], "disclosure_schedule")
        self.assertEqual(x["Section 3"]["status"], "external")
        self.assertNotIn("Annex 3.7", x)

    def test_joiner_before_a_tail_is_not_consumed(self) -> None:
        xml = _doc(
            _section("s3", "Section 3.14 Benefits", "(a) plans.")
            + _section("s4", "Section 4.1 Refs", "Section 3.14(a) to the Anadarko Disclosure Schedule lists the plans.")
        )
        self.assertEqual(_xrefs(build_reader_index(xml))["Section 3.14(a)"]["status"], "disclosure_schedule")

    def test_lowercase_disclosure_schedule_is_not_a_link(self) -> None:
        xml = _doc(
            _section("s3", "Section 3.2 Capital", "(a) shares.")
            + _section("s4", "Section 4.1 Refs", "Except as set forth in Section 3.2 to the disclosure schedule, none.")
        )
        self.assertEqual(_xrefs(build_reader_index(xml))["Section 3.2"]["status"], "disclosure_schedule")

    def test_local_clause_resolves_within_current_section(self) -> None:
        xml = _doc(_section("s1", "Section 7.2 Conditions", "(a) first.", "(b) second, subject to clause (a) above."))
        x = _xrefs(build_reader_index(xml))["clause (a)"]
        self.assertEqual((x["status"], x["target"]), ("resolved", "s7.2(a)"))

    def test_qualified_clause_lists_fall_back_to_the_named_section(self) -> None:
        xml = _doc(
            _section("s3", "Section 3.4 Consents", "(a) none.", "(c) filings.")
            + _section("s5", "Section 5.1 Refs", "Except the filings in clauses (i) through (v) of Section 3.4(c), none.")
        )
        x = _xrefs(build_reader_index(xml))
        self.assertEqual((x["clauses (i)"]["status"], x["clauses (i)"]["target"]), ("section_fallback", "s3.4(c)"))
        self.assertEqual(x["Section 3.4(c)"]["target"], "s3.4(c)")

    def test_spaced_enumerators_and_capitalized_heads(self) -> None:
        xml = _doc(
            _article("a7", "ARTICLE VII CONDITIONS", _section("s71", "Section 7.1 Mutual", "(a) approval."))
            + _article(
                "a8",
                "ARTICLE VIII TERMINATION",
                _section("s81", "Section 8.1 Termination", "ARTICLE VII applies; see Sections 7.1 (a) and 8.1, and the conditions set forth in ARTICLE VII."),
            )
        )
        x = _xrefs(build_reader_index(xml))
        self.assertEqual(x["Sections 7.1 (a)"]["target"], "s7.1(a)")
        self.assertEqual(x["8.1"]["target"], "s8.1")
        # the paragraph-opening "ARTICLE VII" is treated as a heading; the later one links
        arts = [r for r in build_reader_index(xml)["xrefs"] if r["text"] == "ARTICLE VII"]
        self.assertEqual([r["target"] for r in arts], ["a7"])

    def test_tables_of_contents_and_backward_items_are_not_linked(self) -> None:
        xml = _doc(
            _section("s32", "Section 3.2 Deposit", "(b) escrow.", "(d) release.")
            + _section(
                "s91",
                "Section 9.1 Refs",
                "SECTION 3.1 Shelf 8 SECTION 3.2 Incidental Registrations 10 SECTION 3.3 Piggyback 12",
                "Payable upon the earlier of (a) its release under Section 3.2 (d), or (b) the termination of this Agreement.",
            )
        )
        p = build_reader_index(xml)
        self.assertFalse([r for r in p["xrefs"] if r["text"].startswith("SECTION")])
        self.assertNotIn("(b)", _xrefs(p))
        self.assertEqual(_xrefs(p)["Section 3.2 (d)"]["target"], "s3.2(d)")

    def test_heldout_v2_reference_tails_and_lists(self) -> None:
        xml = _doc(
            _section("s3", "Section 3.2 Taxes", "(a) returns.")
            + _section("s8", "Section 8.2 Survival", "Survival.")
            + _section("s10", "Section 10.2 Remedies", "(a) cure.")
            + _section(
                "s11",
                "Section 11.1 Refs",
                "Except as set forth in Section 3.2 of the Seller’s Disclosure Letter, none.",
                "Listed in Section 3.2. of the Company Disclosure Letter.",
                "Under Section 4 para 1 and 2 AWG.",
                "The Partnership Agreement and Section 8.2 thereof continue.",
                "Remedies pursuant to Sections 10.2(a), (b) and (c) shall not apply.",
                "(other than Sections 3.2(a)), 8.2 and 10.2, any remedy.",
            )
        )
        refs = build_reader_index(xml)["xrefs"]
        by_text = {(r["at"]["t"], r["text"]): r for r in refs}
        self.assertEqual(by_text[(0, "Section 3.2")]["status"], "disclosure_schedule")
        self.assertEqual(by_text[(1, "Section 3.2")]["status"], "disclosure_schedule")
        self.assertEqual(by_text[(2, "Section 4")]["status"], "external")
        self.assertEqual(by_text[(3, "Section 8.2")]["status"], "unresolved")
        self.assertEqual([by_text[(4, t)]["target"] for t in ("(b)", "(c)")], ["s10.2", "s10.2"])
        self.assertEqual(by_text[(5, "8.2")]["target"], "s8.2")


class ReaderHealthShapeTests(unittest.TestCase):
    """Shapes the 2026-10-30 reader health report found the enricher missing (0.5.0)."""

    def _ids(self, nodes: list[dict[str, Any]]) -> list[str]:
        return [i for n in nodes for i in (n["id"], *self._ids(n.get("children", [])))]

    def test_dotted_subsections_own_their_enumerated_paragraphs(self) -> None:
        xml = _doc(
            _article(
                "a5",
                "ARTICLE V COVENANTS",
                _section(
                    "s516",
                    "Section 5.16 Earnout",
                    "Section 5.16.1 Payments. Buyer pays.",
                    "(a) first.",
                    "(b) second.",
                    "5.16.2 Statements.",
                    "(a) delivered.",
                ),
            )
            + _section(
                "s61",
                "Section 6.1 Refs",
                "As in Section 5.16.1(b) and Section 5.16.2(a).",
                "See Section 5.16.2(d) and Article 5.16 and Section 5.16.9.",
            )
        )
        p = build_reader_index(xml)
        ids = self._ids(p["outline"])
        for sub in ("s5.16.1", "s5.16.1(a)", "s5.16.1(b)", "s5.16.2", "s5.16.2(a)"):
            self.assertIn(sub, ids)
        self.assertNotIn("s5.16(a)", ids)
        refs = {(r["at"]["t"], r["text"]): r for r in p["xrefs"]}
        # the paragraph's own number is not a reference to itself
        self.assertNotIn((0, "Section 5.16.1"), refs)
        self.assertEqual(refs[(0, "Section 5.16.1(b)")]["target"], "s5.16.1(b)")
        self.assertEqual(refs[(0, "Section 5.16.2(a)")]["target"], "s5.16.2(a)")
        self.assertEqual(
            (refs[(1, "Section 5.16.2")]["status"], refs[(1, "Section 5.16.2")]["target"]),
            ("section_fallback", "s5.16.2"),
        )
        self.assertEqual(refs[(1, "Article 5.16")]["target"], "s5.16")
        # no dotted paragraph 5.16.9: left unlinked rather than sent to 5.16
        self.assertEqual(refs[(1, "Section 5.16.9")]["status"], "unresolved")

    def test_flat_agreement_items_tagged_as_articles(self) -> None:
        xml = _doc(
            _article("a1", "Section 1. Sale", "<text>Seller sells.</text>")
            + _article("a2", "Section 2. Price", "<text>(a) Cash.</text><text>(b) Stock.</text>")
            + _article("a3", "Section 3. Refs", "<text>Under Section 2(b) and Section 2(c) and Section 1(a).</text>")
        )
        p = build_reader_index(xml)
        self.assertIn("s2(b)", self._ids(p["outline"]))
        refs = _xrefs(p)
        self.assertEqual(refs["Section 2(b)"]["target"], "s2(b)")
        self.assertEqual((refs["Section 2"]["status"], refs["Section 2"]["target"]), ("section_fallback", "a2"))
        self.assertEqual(refs["Section 1"]["target"], "a1")


    def _status(self, xml: str) -> dict[str, tuple[str, str | None]]:
        return {r["text"]: (r["status"], r.get("target")) for r in build_reader_index(xml)["xrefs"]}

    def test_duplicated_numbers_give_no_subsection_targets(self) -> None:
        flat = self._status(
            _doc(
                _article("a2", "Section 2. Price", "<text>(a) Cash.</text><text>(b) Stock.</text>")
                + _article("a2b", "Section 2. Other", "<text>(a) x.</text>")
                + _article("a3", "Section 3. Refs", "<text>Under Section 2(b) and Section 2.</text>")
            )
        )
        self.assertEqual(flat["Section 2(b)"], ("unresolved", None))
        self.assertEqual(flat["Section 2"][0], "ambiguous")
        dotted = self._status(
            _doc(
                _section("x", "Section 5.16 A", "5.16.1 p.", "(a) q.")
                + _section("y", "Section 5.16 B", "5.16.1 p.", "(a) q.")
                + _section("z", "Section 6.1 Refs", "See Section 5.16.1(a) and Section 5.16.1.")
            )
        )
        self.assertEqual(dotted["Section 5.16.1(a)"], ("unresolved", None))
        self.assertEqual(dotted["Section 5.16.1"], ("unresolved", None))

    def test_section_list_resumes_after_dotted_paragraphs(self) -> None:
        refs = self._status(
            _doc(
                _section("x", "Section 5.16 A", "(a) Payments:", "5.16.1 first.", "5.16.2 second.", "(b) Statements.")
                + _section("y", "Section 5.17 B", "(a) Payments:", "5.17.1 first.", "(a) one.", "(b) two.")
                + _section("z", "Section 6.1 Refs", "See Section 5.16(b) and Section 5.17.1(b).")
            )
        )
        self.assertEqual(refs["Section 5.16(b)"], ("resolved", "s5.16(b)"))
        self.assertEqual(refs["Section 5.17.1(b)"], ("resolved", "s5.17.1(b)"))

    def test_paragraphs_numbered_for_another_section_are_skipped(self) -> None:
        refs = self._status(
            _doc(
                _section("x", "Section 5.16 A", "5.16.1 Pay.", "(a) one.", "5.17.1 Other.", "(b) belongs.")
                + _section("z", "Section 6.1 Refs", "See Section 5.16.1(b).")
            )
        )
        self.assertEqual(refs["Section 5.16.1"], ("section_fallback", "s5.16.1"))

    def test_article_with_untagged_sections_is_not_flat(self) -> None:
        refs = self._status(
            _doc(
                _article("a1", "1. DEFINITIONS", "<text>1.1 In this agreement:</text><text>(b) Business Day.</text>")
                + _article("a2", "2. Refs", "<text>see clause 1(b).</text>")
            )
        )
        self.assertEqual(refs["clause 1(b)"], ("unresolved", None))

    def test_only_a_subsection_paragraph_loses_its_leading_number(self) -> None:
        p = build_reader_index(
            _doc(
                _section("s", "Section 9.1 S", "Section 4.1.1 shall survive.", "Section 2.1.1 of the Company Disclosure Letter lists.")
                + _section("t", "Section 4.1 X", "4.1.1 thing.")
            )
        )
        refs = {(r["at"]["t"], r["text"]): r for r in p["xrefs"]}
        self.assertEqual(refs[(0, "Section 4.1.1")]["target"], "s4.1.1")
        self.assertEqual(refs[(1, "Section 2.1.1")]["status"], "disclosure_schedule")


class LetterSuffixTests(unittest.TestCase):
    """"Section 8.1A(d)": lettered paragraph groups and glued text (0.6.0)."""

    def _status(self, xml: str) -> dict[str, tuple[str, str | None]]:
        return {r["text"]: (r["status"], r.get("target")) for r in build_reader_index(xml)["xrefs"]}

    def test_lettered_groups_own_their_paragraphs(self) -> None:
        refs = self._status(
            _doc(
                _section(
                    "s81", "Section 8.1 Termination", "This Agreement may be terminated:",
                    "A. before the Acceptance Time:", "(a) by consent;", "(b) by either party:", "(i) at the Outside Date;",
                    "B. after the Acceptance Time:", "(a) by Parent;",
                )
                + _section(
                    "s82", "Section 8.2 Effect",
                    "Under Section 8.1A(b)(i), Section 8.1B, Section 8.1(A)(a) and Section 8.1C(a); not Section 8.1(a).",
                )
            )
        )
        self.assertEqual(refs["Section 8.1A(b)(i)"], ("resolved", "s8.1(A)(b)(i)"))
        self.assertEqual(refs["Section 8.1B"], ("resolved", "s8.1(B)"))
        self.assertEqual(refs["Section 8.1(A)(a)"], ("resolved", "s8.1(A)(a)"))
        # no group C: left unlinked rather than sent to Section 8.1
        self.assertEqual(refs["Section 8.1C(a)"], ("unresolved", None))
        # "(a)" in a grouped section does not say which group when both have one
        self.assertEqual(refs["Section 8.1"], ("section_fallback", "s8.1"))

    def test_a_paragraph_in_only_one_group_resolves(self) -> None:
        refs = self._status(
            _doc(
                _section("s214", "Section 2.14 Reps", "A. Solely for purposes of this Section:", "B. Except as disclosed:", "(a) one.", "(t) taxes.")
                + _section("s61", "Section 6.1 Refs", "See Section 2.14(t).")
            )
        )
        self.assertEqual(refs["Section 2.14(t)"], ("resolved", "s2.14(B)(t)"))

    def test_one_lettered_paragraph_is_not_a_group(self) -> None:
        refs = self._status(
            _doc(
                _section("s53", "Section 5.3 X", "A. Smith shall serve.", "(a) one.", "(b) two.")
                + _section("s61", "Section 6.1 Refs", "See Section 5.3(b) and Section 5.3A.")
            )
        )
        self.assertEqual(refs["Section 5.3(b)"], ("resolved", "s5.3(b)"))
        self.assertEqual(refs["Section 5.3A"], ("unresolved", None))

    def test_lettered_list_inside_a_paragraph_is_not_a_group(self) -> None:
        refs = self._status(
            _doc(
                _section("s31", "Section 3.1 Reps", "(a) Organization.", "(b) Compliance:", "A. first.", "B. second.", "(c) Taxes.")
                + _section("s26", "Section 2.6 Price", "A. an increase;", "B. a decrease;", "(b) Adjustment.")
                + _section("s61", "Section 6.1 Refs", "See Section 3.1(c) and Section 2.6(b).")
            )
        )
        self.assertEqual(refs["Section 3.1(c)"], ("resolved", "s3.1(c)"))
        # the groups' lists start at (a); a "(b)" right after "B." is the section's own
        self.assertEqual(refs["Section 2.6(b)"], ("resolved", "s2.6(b)"))

    def test_real_letter_suffix_sections_still_resolve(self) -> None:
        refs = self._status(
            _doc(
                _section("s24", "Section 2.4 Price", "(a) x.")
                + _section("s24a", "Section 2.4A Earnout", "(a) y.")
                + _section("s61", "Section 6.1 Refs", "See Section 2.4A(a).", "Under Section 409A of the Code.")
            )
        )
        self.assertEqual(refs["Section 2.4A(a)"], ("resolved", "s2.4A(a)"))
        self.assertEqual(refs["Section 409A"][0], "external")

    def test_glued_words_are_not_letter_suffixes(self) -> None:
        refs = self._status(
            _doc(
                _section("s63", "Section 6.3 A", "(a) x.")
                + _section("s64", "Section 6.4 B", "y.")
                + _section(
                    "s11", "Section 1.1 Refs",
                    "subject to Section 6.4Section 6.3(a), resolved.",
                )
            )
        )
        self.assertEqual(refs["Section 6.3(a)"], ("resolved", "s6.3(a)"))
        self.assertFalse(any(t.startswith("Section 6.4") for t in refs))

    def test_lowercase_word_glued_to_a_suffix_keeps_the_reference(self) -> None:
        refs = self._status(
            _doc(
                _section("s24a", "Section 2.4A Earnout", "x.")
                + _section("s61", "Section 6.1 Refs", "subject to Section 2.4Aand the rest.", "Under Section 409Aof the Code.")
            )
        )
        self.assertEqual(refs["Section 2.4A"], ("resolved", "s2.4A"))
        self.assertEqual(refs["Section 409A"][0], "external")

    def test_groups_and_dotted_paragraphs(self) -> None:
        refs = self._status(
            _doc(
                _section(
                    "s516", "Section 5.16 Employees", "A. General.", "(a) one, as follows:", "5.16.1 Dotted one.",
                    "5.16.2 Dotted two.", "(b) two.", "(c) three.", "B. Benefits.", "(a) four.", "(b) five.",
                )
                + _section("s61", "Section 6.1 Refs", "See Section 5.16A(c) and Section 5.16B(b) and Section 5.16(c).")
            )
        )
        self.assertEqual(refs["Section 5.16A(c)"], ("resolved", "s5.16(A)(c)"))
        self.assertEqual(refs["Section 5.16B(b)"], ("resolved", "s5.16(B)(b)"))
        self.assertEqual(refs["Section 5.16(c)"], ("resolved", "s5.16(A)(c)"))

    def test_list_running_on_into_the_next_letter_is_not_grouped(self) -> None:
        refs = self._status(
            _doc(
                _section(
                    "s31", "Section 3.1 Reps", "A. Corporate Matters.", *[f"({c}) item." for c in "abcdefgh"],
                    "B. Business Matters.", "(i) nine.", "(j) ten.",
                )
                + _section("s61", "Section 6.1 Refs", "See Section 3.1(j).")
            )
        )
        self.assertEqual(refs["Section 3.1(j)"], ("resolved", "s3.1(j)"))

    def test_flat_article_groups(self) -> None:
        refs = self._status(
            _doc(
                _article("a2", "2. Covenants", "".join(f"<text>{t}</text>" for t in ("A. Pre-Closing.", "(a) x.", "(b) y.", "B. Post-Closing.", "(a) z.", "(b) w.", "(c) v.")))
                + _article("a3", "3. Refs", "<text>Under Section 2(c).</text>")
            )
        )
        self.assertEqual(refs["Section 2(c)"], ("resolved", "s2(B)(c)"))

    def test_groups_of_a_pruned_section_do_not_crash(self) -> None:
        refs = self._status(
            _doc(
                _section("x", "Section 8 One", "a.")
                + _section("y", "Section 8 Two", "b.")
                + _section("z", "Section 8.1 T", "A. before:", "(a) one;", "B. after:", "(a) two;")
                + _section("r", "Section 9.1 Refs", "See Section 8.1A.")
            )
        )
        self.assertEqual(refs["Section 8.1A"], ("unresolved", None))

    def test_group_lookup_inside_qualified_and_clause_references(self) -> None:
        refs = self._status(
            _doc(
                _section(
                    "s81", "Section 8.1 T", "A. before:", "(a) one;", "(b) two:", "(A) x;", "(B) y.",
                    "(c) three, as follows (i) x and (ii) y.", "B. after:", "(a) four;", "(d) five.",
                    "Subject to clause (B) above.",
                )
                + _section("s82", "Section 8.2 Refs", "Under clause (ii) of Section 8.1(c).")
            )
        )
        self.assertEqual(refs["clause (B)"], ("resolved", "s8.1(A)(b)(B)"))
        self.assertEqual(refs["clause (ii)"], ("section_fallback", "s8.1(A)(c)"))

    def test_exhibit_glued_to_a_word_keeps_the_reference(self) -> None:
        p = build_reader_index(_doc(_section("s1", "Section 1.1 X", "See Exhibit A-1Form of Note."), back="<text>Exhibit A-1</text>"))
        self.assertEqual([r["text"] for r in p["xrefs"]], ["Exhibit A-1"])

    def test_nested_romans_under_h_do_not_make_groups(self) -> None:
        refs = self._status(
            _doc(
                _section(
                    "s31", "Section 3.1 Reps", "A. Corporate Matters.", *[f"({c}) item." for c in "abcdefg"],
                    "(h) item, including:", "(i) sub one;", "(ii) sub two.", "B. Business Matters.", "(i) nine.", "(j) ten.",
                )
                + _section("s61", "Section 6.1 Refs", "See Section 3.1(j).")
            )
        )
        self.assertEqual(refs["Section 3.1(j)"], ("resolved", "s3.1(j)"))

    def test_uppercase_enumerators_are_not_groups(self) -> None:
        refs = self._status(
            _doc(
                _section("s81", "Section 8.1 T", "(A) by consent;", "(1) first;", "(2) second;", "(B) by Parent.")
                + _section("s82", "Section 8.2 Refs", "Under Section 8.1A(2).", "Under Section 8.1(2).")
            )
        )
        self.assertEqual(refs["Section 8.1A(2)"], ("unresolved", None))
        self.assertEqual(refs["Section 8.1"], ("section_fallback", "s8.1"))

    def test_disclosure_memorandum_is_a_disclosure_schedule(self) -> None:
        refs = self._status(
            _doc(
                _section("s612", "Section 6.12 Permits", "Section 6.12 of the Disclosure Memorandum lists permits.")
                + _section("s61", "Section 6.1 Refs", "Except as set forth in Section 6.12 of the Disclosure Memorandum.")
            )
        )
        self.assertEqual(refs["Section 6.12"][0], "disclosure_schedule")


class DeclarationPatternTests(unittest.TestCase):
    def _decls(self, *texts: str) -> dict[str, str]:
        p = build_reader_index(_doc(_section("s1", "Section 1.1 Definitions", *texts)))
        return {t["term"]: t["decls"][0]["kind"] for t in p["terms"]}

    def test_runs_of_terms_and_qualifiers_before_means(self) -> None:
        d = self._decls(
            "(bb) “Tax” or “Taxes” means any tax.",
            "An \"Affiliate\" of any Person shall mean any controlled Person.",
            "\"Material Adverse Effect\" on or with respect to the Company means a change.",
            "\"Liberty Media Affiliate,\" \"Affiliate of Liberty Media\" and similar phrases shall mean Liberty.",
            "“dollars” or “$” means lawful money.",
        )
        for term in ("Tax", "Taxes", "Affiliate", "Material Adverse Effect", "Liberty Media Affiliate", "Affiliate of Liberty Media", "$"):
            self.assertEqual(d.get(term), "means", term)
        self.assertNotIn("dollars", d)

    def test_correlatives_references_and_the_term(self) -> None:
        d = self._decls(
            "“Control” means power. “Controlled”, “Controlling” and “under common Control with” have correlative meanings.",
            "All references to “$” or “dollar” are to U.S. dollars.",
            "(o) the term “Person” or “person” shall include individuals.",
            "except for the use of the term \"Closing Date\" in Sections 6.2 and 6.3.",
            "“cGMP” means current good manufacturing practice.",
        )
        for term in ("Controlled", "Controlling", "$", "Person", "cGMP"):
            self.assertEqual(d.get(term), "means", term)
        self.assertNotIn("under common Control with", d)
        self.assertNotIn("Closing Date", d)

    def test_parentheticals_ending_in_a_term(self) -> None:
        d = self._decls(
            "This Agreement (this “Agreement”) is made (the time the Merger becomes effective being the “Effective Time”).",
            "Cash (the “Cash Consideration” and together with the Stock Consideration, the “Merger Consideration”).",
            "Liability (clause (a) and clause (b), collectively, the “Surviving Obligations”).",
            "Groups (a \"Past Group\", together with Current Groups, an \"Affiliated Group\").",
            "Ratios (the \"Preferred Ratio\" and, collectively with the \"Common Ratio,\" the \"Exchange Ratios\").",
            "Debt (as defined in the “Credit Agreement”).",
        )
        for term in ("Agreement", "Effective Time", "Cash Consideration", "Merger Consideration", "Surviving Obligations", "Past Group", "Affiliated Group", "Preferred Ratio", "Exchange Ratios"):
            self.assertEqual(d.get(term), "paren", term)
        self.assertNotIn("Common Ratio", d)
        self.assertNotIn("Credit Agreement", d)

    def test_chained_as_clauses_and_sentence_breaks(self) -> None:
        d = self._decls(
            "The balance sheet is referred to herein as the “Balance Sheet” and the date thereof as the “Balance Sheet Date”.",
            "Each is referred to herein as a “Material Contract” and collectively, the “Material Contracts”.",
            "Payments are also deemed to be \"Benefit Rights.\" \"FUNDED PLAN\" means a plan.",
        )
        for term in ("Balance Sheet", "Balance Sheet Date", "Material Contract", "Material Contracts"):
            self.assertEqual(d.get(term), "paren", term)
        self.assertEqual(d.get("FUNDED PLAN"), "means")
        self.assertNotIn("Benefit Rights", d)

    def test_corpus_audit_false_positives(self) -> None:
        d = self._decls(
            'The representations in Section 2.1 ("Due Incorporation, Etc.") and Section 2.3 ("Capitalization") survive.',
            "(f) references to “Schedules” shall be to the Company Disclosure Schedule.",
            "1. Each reference to “Chairman of the Board” shall be amended to state “Executive Chairman.”",
            "except that references to “15%” and “85%” shall be replaced by “50%.”",
            "Buyer (“Vandalia Corp.”) and the United States (the “U.S.”) agree.",
        )
        for term in ("Due Incorporation, Etc", "Capitalization", "Schedules", "Chairman of the Board", "85%", "50%"):
            self.assertNotIn(term, d)
        self.assertEqual(d.get("Vandalia Corp."), "paren")
        self.assertEqual(d.get("U.S."), "paren")

    def test_heldout_v2_declaration_forms(self) -> None:
        d = self._decls(
            "The date of such filing is herein called the \"Effective Date.\"",
            "the provisions of ERISA, which provisions are collectively referred as “COBRA,” and any state law.",
            "The term \"INTELLECTUAL PROPERTY\" shall include all trade names.",
            "\"Corresponding Saving\", in relation to a Tax Liability, means any repayment.",
            "“Subsidiary” when used with respect to any party, shall mean any corporation.",
            "“Company’s Knowledge”, “Knowledge of the Company” or similar terms used in this Agreement mean the knowledge of X.",
            "“Agreement”, “this Agreement”, “herein” and similar expressions refer to this agreement.",
            "As used herein, a \"Tax Protection Agreement\" is an agreement to indemnify. The \"Parent Price\" shall be the average.",
        )
        for term in ("Effective Date", "COBRA", "INTELLECTUAL PROPERTY", "Corresponding Saving", "Subsidiary", "Company’s Knowledge", "Knowledge of the Company", "Agreement", "Tax Protection Agreement", "Parent Price"):
            self.assertIn(term, d, term)

    def test_unquoted_colon_glossary_but_not_notice_blocks(self) -> None:
        entries = [f"{t}: {t.lower()} has the meaning a careful drafter gives it here." for t in ("Business Day", "Affiliate", "Encumbrance", "IRS", "Losses", "Tax Return")]
        notices = ["Attention: John Smith, General Counsel of the Company", "Facsimile No.: (212) 555-0100 and (212) 555-0101", "E-mail: legal@example.com and notices@example.com"] * 2
        p = build_reader_index(_doc(_section("s1", "Section 10.1 Definitions", *entries) + _section("s2", "Section 11.2 Notices", *notices)))
        terms = {t["term"] for t in p["terms"]}
        self.assertTrue({"Business Day", "Affiliate", "Encumbrance", "IRS", "Losses", "Tax Return"} <= terms)
        self.assertFalse({"Attention", "Facsimile No.", "E-mail"} & terms)

    def test_all_caps_terms_merge_and_match_title_case(self) -> None:
        p = build_reader_index(
            _doc(
                _section("s0", "Section 1.1 Parties", "Acme Inc. (the “Company”) and the HSR rules apply.")
                + _section("s1", "Section 1.2 Definitions", '"COMPANY" means Acme.', '"HSR ACT" means the Hart-Scott-Rodino Act.', '"ANNUAL BONUSES" means the bonuses.')
                + _section("s2", "Section 5.1 Covenants", "The Company shall file under the HSR Act and pay each Annual Bonus.")
            )
        )
        company = _term(p, "Company")
        self.assertEqual({d.get("as", "Company") for d in company["decls"]}, {"Company", "COMPANY"})
        names = {t["id"]: t["term"] for t in p["terms"]}
        used = [names[u[2]] for u in p["uses"]["s2/0"]]
        self.assertEqual(used, ["Company", "HSR ACT", "ANNUAL BONUSES"])

    def test_pointers_to_the_preamble_are_pointers(self) -> None:
        p = build_reader_index(
            _doc(
                '<section uuid="s1" title="Section 1.1 Definitions">'
                '<definition term="Buyer"><text>“Buyer” shall have the meaning set forth in the introductory paragraph.</text></definition>'
                "<text>“Merger” shall have the meaning specified in the preamble hereto.</text></section>"
            )
        )
        self.assertEqual(_term(p, "Buyer")["decls"][0]["kind"], "pointer")
        self.assertEqual(_term(p, "Merger")["decls"][0]["pointer"], "preamble")


class HeldoutV3FormTests(unittest.TestCase):
    """Forms the held-out v3 gold found (docs/plans/smart-reader.md 10.6)."""

    def _decls(self, *texts: str) -> dict[str, str]:
        return DeclarationPatternTests._decls(self, *texts)  # pyright: ignore[reportArgumentType]

    def test_declaration_forms(self) -> None:
        d = self._decls(
            "“Beneficial Owner” and derivatives thereof, with respect to any Shares, mean a holder.",
            "“Knowledge of the Company” or “Company’s Knowledge,” or a variation thereof, means actual knowledge.",
            "(b) \"KNOWLEDGE\" with respect to (i) the Company means the actual knowledge of its officers.",
            "As used herein, “Fully-Diluted Shares” as of any time shall mean all shares.",
            "“Expenses” includes all reasonable out-of-pocket expenses.",
            "The parties pay their costs. “Fees”, as used in this Agreement, shall include all fees.",
            "The date on which the Closing occurs is referred to in this Agreement as the “Closing Date.”",
            "“Exhibits” means the exhibits listed in the table of contents.",
            "each certificate (a \"Certificate\", it being understood that any reference applies) shall be cancelled.",
            "(each Share referred to in clause (i) or (ii) above being an \"Excluded Share\" and collectively, \"Excluded Shares\")",
        )
        for term in (
            "Beneficial Owner", "Knowledge of the Company", "Company’s Knowledge", "KNOWLEDGE", "Fully-Diluted Shares", "Expenses",
            "Fees", "Closing Date", "Exhibits", "Certificate", "Excluded Share", "Excluded Shares",
        ):
            self.assertIn(term, d)
        # a closing curly quote used to open ("(c) ”Knowledge” of any person ... means")
        self.assertIn("Knowledge", self._decls("(c) ”Knowledge” of any person that is not an individual means the knowledge of its officers."))
        # a reading rule still declares no document part, and "includes" mid-sentence declares nothing
        d = self._decls("The terms “Section” and “Article” refer to parts of this Agreement.", "the “Company” includes its Subsidiaries.")
        self.assertFalse({"Section", "Article", "Company"} & set(d))

    def test_reference_forms(self) -> None:
        p = build_reader_index(
            _doc(
                _article("a3", "ARTICLE III REPRESENTATIONS", _section("s31", "Section 3.1 Organization", "(a) It is organized.") + _section("s35", "Section 3.5 Title", "Title is good."))
                + _article(
                    "a4",
                    "ARTICLE IV PARENT",
                    _section("s41", "Section 4.1 Parent", "Parent is organized.")
                    + _section("s42", "Section 4.2 Authority", "Parent has authority.")
                    + _section("s47", "Section 4.7 Brokers", "No brokers."),
                )
                + _section(
                    "s9",
                    "Section 9.1 Other",
                    "The warranty in section 3.5 and the conditions in article III apply. See Article Three. "
                    + "Parent makes Sections 4.1 (except that “Delaware” reads “Australia”), 4.2 (Authority) and 4.7 (Brokers) again. "
                    + "A single employer under Section 4.14(b), (c), (m) or (o) of the Code. Any Investment Company Act filing.",
                ),
            )
        )
        refs = [x for x in p["xrefs"] if x["at"]["s"] == "s9"]
        by_text = {x["text"]: x for x in refs}
        self.assertEqual(by_text["section 3.5"]["target"], "s3.5")
        self.assertEqual(by_text["article III"]["target"], "a3")
        self.assertEqual(by_text["Article Three"]["target"], "a3")
        self.assertEqual([by_text[k]["target"] for k in ("Sections 4.1", "4.2", "4.7")], ["s4.1", "s4.2", "s4.7"])
        self.assertEqual({by_text[k]["status"] for k in ("Section 4.14(b)", "(c)")}, {"external"})


    def test_other_documents_and_typed_numbers(self) -> None:
        # found by the 0.4.0 corpus audit (docs/plans/smart-reader.md 10.7)
        from etl.domain.reader_index import _find_refs

        def refs(t: str) -> list[tuple[str, str]]:
            return [(r.target, r.status) for r in _find_refs(t, False)]

        self.assertEqual(refs("under section 3 of the European Union (Withdrawal) Act 2018"), [("3", "external")])
        self.assertEqual(refs("pursuant to article 15 of the Decree"), [("15", "external")])
        self.assertEqual(refs("in Seller Disclosure Schedule section 3.4(c) and"), [("3.4(c)", "disclosure_schedule")])
        self.assertEqual(refs("Articles Four and Five of the Company Articles"), [("Four", "external"), ("Five", "external")])
        self.assertEqual(refs("the Dutch Civil Code articles 7:17 and"), [("7", "external")])
        self.assertEqual(refs("set forth in Section 4,02 of this Agreement"), [("4.02", "")])
        # this agreement's own lowercase references and clause lists are untouched
        self.assertEqual(refs("in section 4.1 of this Agreement"), [("4.1", "")])
        self.assertEqual(refs("clauses (i) through (v) of Section 3.4(c), conflict"), [("3.4(c)(i)", ""), ("3.4(c)(v)", ""), ("3.4(c)", "")])


class AnchorTests(unittest.TestCase):
    def test_text_inside_definition_counts_toward_the_ordinal(self) -> None:
        xml = _doc(
            '<section uuid="s1" title="Section 1.1 Defs">'
            "<text>The following terms apply.</text>"
            '<definition term="Party"><text>“Party” means a signatory.</text></definition>'
            "<text>See Section 1.1.</text></section>"
        )
        p = build_reader_index(xml)
        self.assertEqual(_term(p, "Party")["decls"][0]["kind"], "element")
        self.assertEqual(_term(p, "Party")["decls"][0]["anchor"], {"s": "s1", "t": 1})
        self.assertEqual(_xrefs(p)["Section 1.1"]["at"], {"s": "s1", "t": 2})

    def test_text_hash_ignores_attributes(self) -> None:
        a = _doc(_section("s1", "Section 1.1 A", "Hello"))
        b = a.replace('title="Section 1.1 A"', 'title="Section 1.1 A" standardId="x"')
        self.assertEqual(text_hash(a), text_hash(b))
        self.assertNotEqual(text_hash(a), text_hash(a.replace("Hello", "Hullo")))

    def test_deterministic(self) -> None:
        self.assertEqual(build_reader_index(BASIC), build_reader_index(BASIC))


if __name__ == "__main__":
    unittest.main()
