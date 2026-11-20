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


class BareSectionToArticleTests(unittest.TestCase):
    """A bare "Section 2" links to article 2 only when that article is printed as a section."""

    def _status(self, xml: str) -> dict[str, tuple[str, str | None]]:
        return {r["text"]: (r["status"], r.get("target")) for r in build_reader_index(xml)["xrefs"]}

    def test_section_headed_top_level_resolves(self) -> None:
        refs = self._status(
            _doc(
                _article("a2", "SECTION 2 REPRESENTATIONS", _section("s21", "2.1 Organization", "x"))
                + _article("a3", "3. Covenants", _section("s31", "3.1 Conduct", "x"))
                + _article("a4", "SECTION 4 TERMINATION", _section("s41", "4.1 Refs", "See Section 2 and Section 3."))
            )
        )
        self.assertEqual(refs["Section 2"], ("resolved", "a2"))
        self.assertEqual(refs["Section 3"], ("resolved", "a3"))

    def test_article_headed_top_level_does_not_resolve(self) -> None:
        # TechTarget (f02240cd): "this Section 2" in Annex B and the index's "Annex B, Section 1"
        # linked to ARTICLE II and ARTICLE I of the agreement
        refs = self._status(
            _doc(
                _article("a1", "ARTICLE I The Separation", _section("s11", "Section 1.01. Closing.", "x"))
                + _article("a2", "Article 2 Merger", _section("s21", "Section 2.01 Effect", "x"))
                + _article(
                    "a3",
                    "ARTICLE III Definitions",
                    _section("s31", "Section 3.01 Index", "Bluefin Transferred Contracts Annex B, Section 1"),
                    )
                + _article("a4", "ARTICLE IV Misc", _section("s41", "Section 4.01 Refs", "See Section 2 and Section 2.01 and Article II."))
            )
        )
        self.assertEqual(refs["Section 1"], ("unresolved", None))
        self.assertEqual(refs["Section 2"], ("unresolved", None))
        self.assertEqual(refs["Section 2.01"], ("resolved", "s2.1"))
        self.assertEqual(refs["Article II"], ("resolved", "a2"))

    def test_flat_article_still_resolves(self) -> None:
        refs = self._status(
            _doc(
                _article("a5", "ARTICLE 5 Representations", "<text>(a) Organization.</text><text>(b) Disclosure.</text>")
                + _article("a6", "ARTICLE 6 Refs", "<text>The representations in this Section 5 are true.</text>")
            )
        )
        self.assertEqual(refs["Section 5"], ("resolved", "a5"))

    def test_numbered_section_wins_and_statutes_stay_external(self) -> None:
        xml = _doc(
            _article("a2", "ARTICLE 2 Price", _section("s2", "Section 2. Price", "x"))
            + _article("a3", "ARTICLE 3 Refs", _section("s3", "Section 3. Refs", "Under Section 2 the price is paid. Registration follows Section 2 of the Securities Act."))
        )
        got = [(r["status"], r.get("target")) for r in build_reader_index(xml)["xrefs"]]
        self.assertEqual(got, [("resolved", "s2"), ("external", None)])


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


class RomanSectionTests(unittest.TestCase):
    """"Section VIII.2(e)": sections numbered under their Roman article number (0.7.0)."""

    ROMAN = _doc(
        _article(
            "a1",
            "ARTICLE I. DEFINITIONS",
            _section("s11", "Section I.1 Definitions.", "“Closing” means the closing."),
        )
        + _article(
            "a8",
            "ARTICLE VIII. CONDITIONS",
            _section("s81", "Section VIII.1 Mutual Conditions.", "(a) no injunction;", "(b) approvals.")
            + _section("s82", "Section VIII.2 Parent Conditions.", "(a) accuracy;", "(b) covenants;", "(i) performed.")
            + _section(
                "s83",
                "Section VIII.3 Frustration.",
                "No party may rely on Section VIII.2(b)(i), Sections VIII.1 and VIII.2, clause (b) of Section VIII.2, "
                "Article VIII or Article VIII.1. Section VIII.1(a) applies, and Exhibit C.2 is Exhibit C.",
            ),
        )
    )

    def _status(self, xml: str) -> dict[str, tuple[str, str | None]]:
        return {r["text"]: (r["status"], r.get("target")) for r in build_reader_index(xml)["xrefs"]}

    def test_references_resolve_to_roman_sections(self) -> None:
        refs = self._status(self.ROMAN)
        self.assertEqual(refs["Section VIII.2(b)(i)"], ("resolved", "s8.2(b)(i)"))
        self.assertEqual(refs["Sections VIII.1"], ("resolved", "s8.1"))
        self.assertEqual(refs["VIII.2"], ("resolved", "s8.2"))
        self.assertEqual(refs["clause (b)"], ("resolved", "s8.2(b)"))
        self.assertEqual(refs["Article VIII"], ("resolved", "a8"))
        self.assertEqual(refs["Article VIII.1"], ("resolved", "s8.1"))
        self.assertEqual(refs["Section VIII.1(a)"], ("resolved", "s8.1(a)"))
        # attachments keep their own lettering
        self.assertEqual(refs["Exhibit C"][0], "unlocated_attachment")
        self.assertNotIn("Exhibit C.2", refs)

    def test_outline_and_terms_keep_the_printed_numbers(self) -> None:
        p = build_reader_index(self.ROMAN)
        art = p["outline"][1]
        sec = art["children"][1]
        self.assertEqual((art["label"], sec["id"], sec["label"]), ("VIII", "s8.2", "VIII.2"))
        self.assertEqual([c["label"] for c in sec["children"]], ["VIII.2(a)", "VIII.2(b)"])
        self.assertEqual(sec["children"][1]["children"][0]["label"], "VIII.2(b)(i)")
        self.assertEqual(_term(p, "Closing")["decls"][0]["in"], "Section I.1")

    def test_roman_references_need_a_roman_scheme(self) -> None:
        refs = self._status(
            _doc(
                _article("a4", "ARTICLE IV TERMS", _section("s42", "Section 4.2 Fees", "(a) fees."))
                + _section("s43", "Section 4.3 Refs", "Under Section IV.2(a) of the Plan, Section IV.2 and Article IV.2.")
            )
        )
        # an arabic agreement's "Section IV.2" is some other document's numbering, not Section 4.2
        self.assertEqual(refs["Section IV.2"], ("unresolved", None))
        self.assertEqual(refs["Section IV.2(a)"][0], "external")
        # "Article IV.2" still names Article IV, as before
        self.assertEqual(refs["Article IV.2"], ("resolved", "a4"))

    def test_a_stray_roman_section_is_not_a_scheme(self) -> None:
        p = build_reader_index(
            _doc(
                _article("a1", "ARTICLE I DEFS", _section("s11", "Section 1.1 Defs", "(a) x."))
                + _article("a4", "ARTICLE IV TERMS", _section("s42", "Section 4.2 Fees", "(a) fees.", "(b) costs."))
                + _section("s43", "Section 4.3 Refs", "See Section 1.1(a), Section IV.2(b) and Section 4.2.")
                + _article("a9", "ARTICLE IX PLAN", _section("sx", "I.1 Plan definitions", "(a) z.") + _section("sy", "Section IX.1 Misc", "(a) y."))
            )
        )
        refs = {r["text"]: (r["status"], r.get("target")) for r in p["xrefs"]}
        self.assertEqual(refs["Section 1.1(a)"], ("resolved", "s1.1(a)"))
        self.assertEqual(refs["Section IV.2(b)"], ("unresolved", None))
        self.assertNotIn("s1.1~2", {c["id"] for a in p["outline"] for c in a.get("children", [])})

    def test_lettered_articles_are_not_roman(self) -> None:
        refs = self._status(
            _doc(
                _article("aa", "ARTICLE A ONE", _section("sa", "A.1 Foo", "x."))
                + _article("ab", "ARTICLE B TWO", _section("sb", "B.1 Bar", "x."))
                + _article("ac", "ARTICLE C THREE", _section("sc", "C.1 Baz", "See Section C.1 and Section I.1."))
                + _article("ad", "ARTICLE D FOUR", _section("sd", "D.1 Qux", "x."))
            )
        )
        # "C" is not read as Roman 100: the reference stays the lettered "Section C", as before 0.7.0
        self.assertNotIn("Section C.1", refs)
        self.assertEqual(refs["Section C"], ("unresolved", None))

    def test_roman_dotted_paragraphs_are_outlined_as_printed(self) -> None:
        secs = "".join(_section(f"s{i}", f"Section VIII.{i} T{i}", "x.") for i in (1, 3))
        body = "VIII.2.1 First.</text><text>(a) a.</text><text>VIII.2.2 Second. See Section VIII.2.1(a) and Section VIII.2(a)."
        p = build_reader_index(_doc(_article("a8", "ARTICLE VIII MISC", _section("s2", "Section VIII.2 P", body) + secs)))
        s82 = next(c for c in p["outline"][0]["children"] if c["id"] == "s8.2")
        self.assertEqual([(c["id"], c["label"]) for c in s82["children"]], [("s8.2.1", "VIII.2.1"), ("s8.2.2", "VIII.2.2")])
        self.assertEqual([(c["id"], c["label"]) for c in s82["children"][0]["children"]], [("s8.2.1(a)", "VIII.2.1(a)")])
        refs = {r["text"]: (r["status"], r.get("target")) for r in p["xrefs"]}
        self.assertEqual(refs["Section VIII.2.1(a)"], ("resolved", "s8.2.1(a)"))
        # the (a) belongs to VIII.2.1, not to VIII.2
        self.assertEqual(refs["Section VIII.2"], ("section_fallback", "s8.2"))

    def test_roman_qualifier_outside_a_scheme_does_not_link_locally(self) -> None:
        # the clauses belong to another document's numbering: before 0.7.0 they linked to this section's (a), (b)
        refs = self._status(
            _doc(_article("a4", "ARTICLE IV ONE", _section("s41", "Section 4.1 Foo", "(a) x.</text><text>(b) Nothing in clause (b) of Section IV.2 of the Plan applies.")))
        )
        self.assertNotEqual(refs.get("clause (b)", ("unresolved", None))[0], "resolved")

    def test_french_code_articles_are_not_roman(self) -> None:
        refs = self._status(
            _doc(_article("a1", "ARTICLE I ONE", _section("s1", "Section 1.1 Foo", "Under Article L.233-3 and Section L.3323-8.")))
        )
        self.assertNotIn("Article L.233", refs)
        self.assertNotIn("Section L.3323", refs)

    def test_dotted_paragraphs_keep_their_printed_numbers(self) -> None:
        p = build_reader_index(
            _doc(
                _article(
                    "a8",
                    "ARTICLE VIII C",
                    _section("s81", "Section VIII.1 A", "x.")
                    + _section("s82", "Section VIII.2 P", "8.2.1 First thing.", "(a) sub a.", "8.2.2 Second.")
                    + _section("s83", "Section VIII.3 B", "x."),
                )
            )
        )
        sec = p["outline"][0]["children"][1]
        self.assertEqual(sec["label"], "VIII.2")
        self.assertEqual([c["label"] for c in sec["children"]], ["8.2.1", "8.2.2"])


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


    def test_space_inside_a_section_number(self) -> None:
        # found by the 0.8.0 rebuild: Vurv / Taleo (c1cf60c6) prints every reference as "Section 1. 6(c)"
        from etl.domain.reader_index import _find_refs

        def refs(t: str) -> list[tuple[str, str]]:
            return [(r.target, t[r.start : r.end]) for r in _find_refs(t, False)]

        self.assertEqual(refs("pursuant to the terms of Section 1. 6(c) hereof"), [("1.6(c)", "Section 1. 6(c)")])
        self.assertEqual(refs("in accordance with Section 7. 4(e)(ii)) shall"), [("7.4(e)(ii)", "Section 7. 4(e)(ii)")])
        self.assertEqual(refs("as provided in Section 2. 15 and Section 2.16"), [("2.15", "Section 2. 15"), ("2.16", "Section 2.16")])
        self.assertEqual(
            refs("under Sections 3. 1, 3. 2 and 3.3 hereof"), [("3.1", "Sections 3. 1"), ("3.2", "3. 2"), ("3.3", "3.3")]
        )
        self.assertEqual(refs("see Section 6. 2A hereof"), [("6.2A", "Section 6. 2A")])
        # a sentence that ends at "Section 8." and goes on with a count, a heading or a word stays "Section 8"
        self.assertEqual(refs("terminated under Section 8. 15 days after notice"), [("8", "Section 8")])
        self.assertEqual(refs("delivered under Section 4. 5 Business Days later"), [("4", "Section 4")])
        self.assertEqual(refs("as described in Section 4. 4.2 Covenants. The parties"), [("4", "Section 4")])
        self.assertEqual(refs("as set forth in Section 5. The Company shall"), [("5", "Section 5")])
        self.assertEqual(refs("a rate under Section 1. 6% of the amount"), [("1", "Section 1")])
        self.assertEqual(refs("until Section 8. 2 shares are issued"), [("8", "Section 8")])
        self.assertEqual(refs("under Section 8. 3 or more Persons may"), [("8", "Section 8")])
        # a number never continues on the next line
        self.assertEqual(refs("as stated in Section 7.\n2. Notices."), [("7", "Section 7")])
        # only section references are joined
        self.assertEqual(refs("see Article 2. 3 shares were issued"), [("2", "Article 2")])
        # the space can sit before the period, before the enumerator, or after the zero of "9.03"
        self.assertEqual(refs("in Section 1 .6 hereof"), [("1.6", "Section 1 .6")])
        self.assertEqual(refs("in Section 1. 6 (c) hereof"), [("1.6(c)", "Section 1. 6 (c)")])
        self.assertEqual(refs("under Section 9. 0 3(a) hereof"), [("9.03(a)", "Section 9. 0 3(a)")])
        self.assertEqual(refs("as stated in Section 2.0 4 hereof"), [("2.04", "Section 2.0 4")])
        self.assertEqual(refs("in section 4. 2 of this Agreement"), [("4.2", "section 4. 2")])
        # a heading that follows a full section number is not part of it
        self.assertEqual(refs("as in Section 6.06. 280G Calculation"), [("6.06", "Section 6.06")])
        # a clause takes the joined section, and the tail rules still see what follows the number
        self.assertEqual(
            refs("see clause (ii) of Section 4. 2 hereof"), [("4.2(ii)", "clause (ii)"), ("4.2", "Section 4. 2")]
        )

        def status(t: str) -> list[tuple[str, str]]:
            return [(r.target, r.status) for r in _find_refs(t, False)]

        self.assertEqual(status("under Section 4. 15 of the Securities Act"), [("4.15", "external")])
        self.assertEqual(
            status("Liens identified on Section 1. 0 1 of the Company Disclosure Schedule"), [("1.01", "disclosure_schedule")]
        )

    def test_split_section_number_links_the_section(self) -> None:
        p = build_reader_index(
            _doc(
                _article("a1", "ARTICLE I THE MERGER", _section("s16", "1.6 Effect on Capital Stock", "(c) Options. x"))
                + _article(
                    "a2",
                    "ARTICLE II REPRESENTATIONS",
                    _section("s21", "2.1 Organization", "As provided in Section 1. 6(c) and in Section 1. 9 hereof."),
                )
            )
        )
        got = {x["text"]: (x["status"], x.get("target")) for x in p["xrefs"]}
        self.assertEqual(got["Section 1. 6(c)"], ("resolved", "s1.6(c)"))
        # a joined number with no such section stays unresolved; it does not fall back to the article
        self.assertEqual(got["Section 1. 9"], ("unresolved", None))

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


def _back(*texts: str) -> str:
    return "".join(f"<text>{t}</text>" for t in texts)


# The agreement the back-matter tests cite: Sections 1.1 and 2.1 under ARTICLE headings
MAIN = _article("a1", "ARTICLE I DEFINITIONS", _section("s11", "Section 1.1 Definitions", "x")) + _article(
    "a2", "ARTICLE II THE MERGER", _section("s21", "Section 2.1 The Merger", "y")
)


class BackMatterSegmentTests(unittest.TestCase):
    # Rule 1: own numbering ends a glossary segment
    def test_attached_agreement_with_own_numbering_is_not_a_glossary(self) -> None:
        back = _back(
            "EXHIBIT C",
            "REGISTRATION RIGHTS AGREEMENT",
            "Section 1.1 Definitions. As used in this Agreement:",
            "“Holder” means a holder of Registrable Securities.",
            "“Registrable Securities” means the Shares.",
            "“Shelf” means a shelf registration statement.",
            "Section 2.1 Demand Registration. Subject to Section 1.1, a Holder may demand a registration.",
            "Section 2.2 Piggyback. As provided in Section 2.1 of the Merger Agreement, the Holder may join.",
        )
        x = _xrefs(build_reader_index(_doc(MAIN, back=back)))
        self.assertEqual(x["Section 1.1"]["status"], "unresolved")
        self.assertNotIn("target", x["Section 1.1"])
        # a tail that names the main agreement still resolves
        self.assertEqual(x["Section 2.1"]["target"], "s2.1")

    def test_attachment_names_in_an_attached_agreement_that_defines_terms_still_resolve(self) -> None:
        back = _back(
            "ANNEX 3.7",
            "“Earn-Out Payment” means the payment under Section 3.7.1 of this Annex 3.7.",
            "“EBITDA” means net income.",
            "“Target” means USD 1.",
            "3.7.1 FY 2017 Earn-Out. Paid as set out in Schedule A.",
            "3.7.2 FY 2018 Earn-Out. Paid as set out in Section 2.1.",
            "3.7.3 Maximum. Capped.",
            "SCHEDULE A",
            "PAYMENTS",
        )
        x = _xrefs(build_reader_index(_doc(MAIN, back=back)))
        self.assertEqual(x["Annex 3.7"]["target"], "x-annex-3.7")
        self.assertEqual(x["Schedule A"]["target"], "x-schedule-a")
        # "3.7.1", "3.7.2" under "ANNEX 3.7" continue the agreement's Section 3.7 (gold fixture 03f4e108), so the
        # annex is not an attached document and its bare "Section 2.1" is the agreement's
        self.assertEqual(x["Section 2.1"]["target"], "s2.1")

    def test_numbered_definitions_annex_is_still_a_glossary(self) -> None:
        back = _back(
            "ANNEX A",
            "DEFINITIONS",
            "1.1 “Accounting Principles” means GAAP.",
            "1.2 “Business Day” means a weekday.",
            "1.3 “Closing” has the meaning set forth in Section 2.1.",
            "1.4 “Holder” means a holder.",
        )
        x = _xrefs(build_reader_index(_doc(MAIN, back=back)))
        self.assertEqual(x["Section 2.1"]["target"], "s2.1")

    def test_a_long_glossary_with_a_short_numbered_tail_stays_a_glossary(self) -> None:
        defs = [f"“Term {i}” means item {i}." for i in range(16)]
        back = _back(
            "EXHIBIT A",
            *defs,
            "“Board” means the board, as provided in Section 2.1.",
            "1. Amendment. Section 2.1 of the Merger Agreement is amended.",
            "2. Effect. The rest stays.",
            "3. Miscellaneous.",
        )
        first = next(x for x in build_reader_index(_doc(MAIN, back=back))["xrefs"] if x["text"] == "Section 2.1")
        self.assertEqual(first["target"], "s2.1")

    def test_an_sec_form_stays_external_in_an_attached_agreement(self) -> None:
        back = _back(
            "EXHIBIT C",
            "“Holder” means a holder.",
            "“Shares” means shares.",
            "“Shelf” means a shelf.",
            "Section 1.1 Transfers. No Holder transfers to a Person that filed a Schedule 13D.",
            "Section 1.2 Notices. In writing.",
            "Section 1.3 Law. Delaware.",
        )
        x = _xrefs(build_reader_index(_doc(MAIN, back=back)))
        self.assertEqual(x["Schedule 13D"]["status"], "external")

    def test_two_numbered_paragraphs_keep_the_glossary(self) -> None:
        back = _back(
            "EXHIBIT A",
            "Section 1.1 Definitions.",
            "“Holder” means a holder.",
            "“Shelf” means a shelf registration, as described in Section 2.1.",
            "“Shares” means common stock.",
            "Section 1.2 Interpretation. Headings are for convenience.",
        )
        x = _xrefs(build_reader_index(_doc(MAIN, back=back)))
        self.assertEqual(x["Section 2.1"]["target"], "s2.1")

    def test_own_number_shapes(self) -> None:
        from etl.domain.reader_index import OWN_NUMBER

        for t in ("Section 2.1 Demand Registration.", "2.1 Demand", "2. Transition Services.", "ARTICLE II GENERAL", "1 DEFINITIONS", "Section 7. Notices."):
            self.assertIsNotNone(OWN_NUMBER.match(t), t)
        for t in ("1.1 “Accounting Principles” means GAAP.", "100 Main Street", "2019 Annual Report", "(a) Parent shall", "1 Within 50 Business Days"):
            self.assertIsNone(OWN_NUMBER.match(t), t)


    # Rule 4: an index of defined terms in back matter belongs to the agreement
    INDEX = "Closing Section 2.1 Closing Date Section 2.1(a) Certificate of Merger Section 2.1(b) Definitions Section 1.1 Agreement Preamble"

    def test_index_of_defined_terms_resolves_against_the_agreement(self) -> None:
        main = MAIN.replace("<text>y</text>", "<text>(a) first.</text><text>(b) second.</text>")
        x = _xrefs(build_reader_index(_doc(main, back=_back("EXHIBIT B", "INDEX OF DEFINED TERMS", self.INDEX))))
        self.assertEqual(x["Section 2.1"]["target"], "s2.1")
        self.assertEqual(x["Section 2.1(a)"]["target"], "s2.1(a)")
        self.assertEqual(x["Section 1.1"]["target"], "s1.1")

    def test_index_needs_four_pairs_and_no_own_numbering(self) -> None:
        short = "Closing Section 2.1 Closing Date Section 2.1(a) Certificate of Merger Section 1.1"
        x = _xrefs(build_reader_index(_doc(MAIN, back=_back("EXHIBIT B", short))))
        self.assertEqual(x["Section 2.1"]["status"], "unresolved")
        own = _back("EXHIBIT B", "1. Defined Terms.", self.INDEX, "2. Notices. Each notice is in writing.", "3. Counterparts.")
        x = _xrefs(build_reader_index(_doc(MAIN, back=own)))
        self.assertEqual(x["Section 2.1"]["status"], "unresolved")
        self.assertNotIn("target", x["Section 2.1"])

    def test_index_pairs_skip_prose_and_tables_of_contents(self) -> None:
        from etl.domain.reader_index import _index_pairs

        self.assertEqual(_index_pairs(self.INDEX), 4)
        prose = "as set forth in Section 4.01 (Organization), Section 4.02 (Authority), the first sentence of Section 4.03(a), Section 4.04 and Section 4.05"
        self.assertEqual(_index_pairs(prose), 0)
        toc = "ARTICLE I DEFINITIONS 1 Section 1.1 Definitions 1 Section 1.2 Interpretation 4 Section 1.3 Other 5 Section 1.4 More 6"
        self.assertEqual(_index_pairs(toc), 0)


    # Rule 2: back-matter headings and tables of contents are not references
    def test_back_matter_headings_are_not_references(self) -> None:
        back = _back(
            "EXHIBIT D",
            "Section 9.1 Amendments; Waivers. This agreement may be amended in writing.",
            "Section 8. Fees and Expenses.",
            "Section 1.1 Definitions 1",
            "Section 2.1 The Company shall act as provided in Section 1.1 of the Merger Agreement.",
            "Section 2.1 of the Merger Agreement applies.",
        )
        p = build_reader_index(_doc(MAIN, back=back))
        texts = [(x["at"]["t"], x["text"]) for x in p["xrefs"]]
        self.assertNotIn((1, "Section 9.1"), texts)
        self.assertNotIn((2, "Section 8"), texts)
        self.assertNotIn((3, "Section 1.1"), texts)
        # a paragraph that opens with a real reference keeps it
        self.assertIn((4, "Section 2.1"), texts)
        self.assertIn((4, "Section 1.1"), texts)
        self.assertIn((5, "Section 2.1"), texts)

    def test_body_headings_are_unchanged(self) -> None:
        main = MAIN + _section("s31", "Section 3.1 Other", "Section 2.1 Merger Sub. Merger Sub exists.")
        x = _xrefs(build_reader_index(_doc(main)))
        self.assertEqual(x["Section 2.1"]["target"], "s2.1")

    def test_mixed_case_table_of_contents_in_back_matter(self) -> None:
        toc = "Section 1.1 Definitions 1 Section 1.2 Interpretation 4 Section 2.1 Demand Registration 5 Section 2.2 Piggyback Rights 7"
        p = build_reader_index(_doc(MAIN, back=_back("EXHIBIT D", toc, "Subject to Section 1.1, the Holder may act.")))
        self.assertEqual([x["at"]["t"] for x in p["xrefs"]], [2])
        # three entries are not enough
        three = "Section 1.1 Definitions 1 Section 1.2 Interpretation 4 Section 2.1 Demand Registration 5"
        p = build_reader_index(_doc(MAIN, back=_back("EXHIBIT D", three)))
        self.assertEqual(len(p["xrefs"]), 3)


    def test_back_matter_lines_that_print_the_agreements_own_titles_stay_references(self) -> None:
        # a list of omitted disclosure schedules and the agreement's own table of contents, printed in back matter
        back = _back(
            "EXHIBIT A",
            "“Holder” means a holder.",
            "“Notice” means a notice.",
            "“Share” means a share.",
            "Section 2.1 — The Merger",
            "Section 1.1 Definitions 1 Section 2.1 The Merger 2 Section 1.1 Definitions 3 Section 2.1 The Merger 4",
            "Section 1.1 Notices. Notices go to the Holder.",
        )
        x = build_reader_index(_doc(MAIN, back=back))["xrefs"]
        self.assertEqual([(r["at"]["t"], r["target"]) for r in x if r["at"]["t"] == 4], [(4, "s2.1")])
        self.assertEqual(len([r for r in x if r["at"]["t"] == 5]), 4)
        # a heading whose title is not the agreement's stays out
        self.assertEqual([r for r in x if r["at"]["t"] == 6], [])

    # Rule 6: attachment names in back matter resolve to the agreement's attachments
    def test_attachment_names_in_back_matter_resolve(self) -> None:
        back = _back(
            "EXHIBIT A",
            "FORM OF CERTIFICATE OF MERGER",
            "The form of letter of transmittal is attached as Exhibit B.",
            "EXHIBIT B",
            "LETTER OF TRANSMITTAL",
        )
        x = _xrefs(build_reader_index(_doc(MAIN, back=back)))
        self.assertEqual((x["Exhibit B"]["status"], x["Exhibit B"]["target"]), ("resolved", "x-exhibit-b"))

    def test_attachment_names_in_attached_agreements_stay_local(self) -> None:
        back = _back(
            "EXHIBIT A",
            "Section 1.1 Definitions. Terms have their meanings.",
            "Section 1.2 Exhibits. The form of joinder is attached as Exhibit B.",
            "Section 1.3 Notices. Notices are in writing.",
            "EXHIBIT B",
            "LETTER OF TRANSMITTAL",
        )
        x = _xrefs(build_reader_index(_doc(MAIN, back=back)))
        self.assertEqual(x["Exhibit B"]["status"], "unresolved")
        self.assertNotIn("target", x["Exhibit B"])


    def test_attachment_names_hereto_or_of_another_document_stay_local(self) -> None:
        back = _back(
            "EXHIBIT A",
            "FORM OF CERTIFICATE OF MERGER",
            "The charter shall read as set forth on Exhibit A attached hereto.",
            "The plan is attached hereto as Exhibit A.",
            "The prices are set out in Annex A of this release.",
            "The form of letter of transmittal is attached to the Agreement as Exhibit B.",
            "EXHIBIT B",
            "LETTER OF TRANSMITTAL",
            "ANNEX A",
            "PRICES",
        )
        # "hereto" naming an attachment printed before the reference (here the exhibit itself) stays local
        x = build_reader_index(_doc(MAIN, back=back))["xrefs"]
        self.assertEqual(
            [(r["text"], r["status"], r.get("target")) for r in x if r["text"] in ("Exhibit A", "Exhibit B", "Annex A") and r["at"]["t"] > 1],
            [("Exhibit A", "unresolved", None), ("Exhibit A", "unresolved", None), ("Annex A", "unresolved", None), ("Exhibit B", "resolved", "x-exhibit-b")],
        )

    def test_attachment_names_hereto_resolve_when_printed_after(self) -> None:
        # gold fixture b25fc0d0: an amendment in Exhibit E cites "Schedule A, attached hereto", printed right after it
        back = _back(
            "EXHIBIT A",
            "FORM OF AMENDMENT",
            "The terms are described on Schedule A, attached hereto. The charter is attached hereto as Exhibit A.",
            "SCHEDULE A",
            "TERMS",
        )
        x = _xrefs(build_reader_index(_doc(MAIN, back=back)))
        self.assertEqual(x["Schedule A"]["target"], "x-schedule-a")
        self.assertEqual(x["Exhibit A"]["status"], "unresolved")

    # Rule 7: a UK "clause 9.2" inside a Schedule is the agreement's clause
    UK = _article("c9", "9 WARRANTIES", _section("s92", "9.2 Limits", "x"))

    def test_uk_clause_in_a_schedule_resolves(self) -> None:
        back = _back(
            "SCHEDULE 4",
            "LIMITATIONS",
            "1 DEFINITIONS",
            "1.1 In this Schedule the Warranties are those given under clause 9.2.",
            "2 LIMITS",
            "2.1 Nothing in paragraph 1.1 or clause 9 limits liability.",
        )
        x = _xrefs(build_reader_index(_doc(self.UK, back=back)))
        self.assertEqual(x["clause 9.2"]["target"], "s9.2")
        # paragraphs stay unlinked and a bare top-level "clause 9" stays exhibit-local
        self.assertEqual(x["paragraph 1.1"]["status"], "unresolved")
        self.assertEqual(x["clause 9"]["status"], "unresolved")
        self.assertNotIn("target", x["clause 9"])

    def test_uk_clause_outside_a_schedule_stays_local(self) -> None:
        back = _back("EXHIBIT 4", "DEED OF ADHERENCE", "1. The Covenantor is bound by clause 9.2 of the Deed.", "2. Signed.", "3. Law.")
        x = _xrefs(build_reader_index(_doc(self.UK, back=back)))
        self.assertEqual(x["clause 9.2"]["status"], "unresolved")

    def test_uk_clause_in_a_schedule_stays_local_when_it_is_the_schedules_own(self) -> None:
        own = _back("SCHEDULE 6", "WARRANTIES", "9.3 Contracts", "(a) agreements (other than those under this clause 9.3) are listed.")
        x = _xrefs(build_reader_index(_doc(self.UK, back=own)))
        self.assertEqual(x["clause 9.3"]["status"], "unresolved")
        tail = _back("SCHEDULE 3", "TAX", "1.1 No liability arises if excluded under clause 9.2 of the Tax Covenant.", "1.2 Or under clause 9.2 of this agreement.")
        x = build_reader_index(_doc(self.UK, back=tail))["xrefs"]
        self.assertEqual([(r["status"], r.get("target")) for r in x if r["text"] == "clause 9.2"], [("unresolved", None), ("resolved", "s9.2")])
        deed = _back("SCHEDULE 7", "TAX DEED", "1.1 This Deed is made by the Covenantor.", "7.1 If the Covenantor pays under clause 9.2, this clause 7 applies.")
        x = _xrefs(build_reader_index(_doc(self.UK, back=deed)))
        self.assertEqual(x["clause 9.2"]["status"], "unresolved")

    def test_uk_clause_in_a_schedule_stays_local_above_another_document_or_in_us_drafting(self) -> None:
        back = _back(
            "SCHEDULE 10",
            "WARRANTIES",
            "2.3 Except as set forth in clause 9.2 above, nothing is issued.",
            "2.4 The declaration (as defined in, and pursuant to clause 9.2 of, the Primary Policy) is delivered.",
            "2.5 Liability is capped under clause 9.2.",
        )
        x = build_reader_index(_doc(self.UK, back=back))["xrefs"]
        self.assertEqual([(r["status"], r.get("target")) for r in x if r["text"] == "clause 9.2"], [("unresolved", None), ("unresolved", None), ("resolved", "s9.2")])
        # the same schedule under a US agreement that cites "Section 9.2" in its body
        us = _article("c9", "ARTICLE IX WARRANTIES", _section("s92", "Section 9.2 Limits", "As set out in Section 9.2 and Section 9.3."))
        x = build_reader_index(_doc(us, back=back))["xrefs"]
        self.assertEqual({r["status"] for r in x if r["text"] == "clause 9.2"}, {"unresolved"})


if __name__ == "__main__":
    unittest.main()
