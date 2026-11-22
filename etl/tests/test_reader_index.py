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


def _sections_by_id(p: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    stack = list(p["outline"])
    while stack:
        n = stack.pop()
        if n["kind"] == "section":
            out[n["id"]] = n
        stack.extend(n.get("children", []))
    return out


class PositionalSectionTests(unittest.TestCase):
    """Headings that lost their numbers: the k-th section of Article N is Section N.k, when the text backs it."""

    def test_unnumbered_headings_take_their_position(self) -> None:
        xml = _doc(
            _article(
                "a1",
                "ARTICLE I THE MERGER",
                _section("s1", "Section i.. Merger.", "As set out in this Section 1.01, the parties merge.")
                + _section("s2", "Section ii.. Closing.", "(a) The Closing occurs under this Section 1.02.", "(b) See Section 2.1."),
            )
            + _article(
                "a2",
                "ARTICLE II CONSIDERATION",
                _section("s3", "Conversion.", "Subject to Section 1.2(a) and Section 1.01, shares convert.")
                + _section("s4", "Exchange.", "See Section 2.3."),
            )
        )
        p = build_reader_index(xml)
        secs = _sections_by_id(p)
        self.assertEqual(sorted(secs), ["s1.1", "s1.2", "s2.1", "s2.2"])
        self.assertEqual((secs["s1.2"]["label"], secs["s1.2"]["title"], secs["s1.2"]["anchor"]), ("1.2", "Section ii.. Closing.", {"s": "s2"}))
        x = _xrefs(p)
        self.assertEqual((x["Section 1.2(a)"]["status"], x["Section 1.2(a)"]["target"]), ("resolved", "s1.2(a)"))
        self.assertEqual(x["Section 1.01"]["target"], "s1.1")
        self.assertEqual(x["Section 2.1"]["target"], "s2.1")
        self.assertEqual(x["Section 2.3"]["status"], "unresolved")
        self.assertEqual((p["stats"]["sections_positional"], p["stats"]["sections_unnumbered"]), (4, 0))

    def test_no_backing_in_the_text_means_no_numbers(self) -> None:
        xml = _doc(
            _article("a1", "ARTICLE I THE MERGER", _section("s1", "Merger.", "The parties merge.") + _section("s2", "Closing.", "x"))
            + _article("a2", "ARTICLE II TERMS", _section("s3", "Conversion.", "See Section 1.2."))
        )
        p = build_reader_index(xml)
        self.assertEqual(_sections_by_id(p), {})
        self.assertEqual(_xrefs(p)["Section 1.2"]["status"], "unresolved")
        self.assertEqual((p["stats"]["sections_positional"], p["stats"]["sections_unnumbered"]), (0, 3))

    def test_one_stated_number_fixes_what_is_before_it_only(self) -> None:
        xml = _doc(
            _article(
                "a1",
                "ARTICLE I THE MERGER",
                _section("s1", "Merger.", "The parties merge.") + _section("s2", "Closing.", "See this Section 1.2.") + _section("s3", "Effects.", "x"),
            )
            + _article("a2", "ARTICLE II TERMS", _section("s4", "Conversion.", "See Section 1.1 and Section 1.3."))
        )
        p = build_reader_index(xml)
        # "Merger." is the one section before Section 1.2; nothing shows that counting works past it
        self.assertEqual(sorted(_sections_by_id(p)), ["s1.1", "s1.2"])
        x = _xrefs(p)
        self.assertEqual((x["Section 1.1"]["target"], x["Section 1.3"]["status"]), ("s1.1", "unresolved"))

    def test_gap_between_printed_numbers_is_filled_only_when_it_fits(self) -> None:
        xml = _doc(
            _article(
                "a1",
                "ARTICLE I TERMS",
                _section("s1", "Section 1.1 One.", "a")
                + _section("s2", "Two.", "b")
                + _section("s3", "Section 1.3 Three.", "See Section 1.2 and Section 1.5.")
                + _section("s4", "Four.", "c")
                + _section("s5", "Section 1.6 Six.", "d")
                + _section("s6", "Seven.", "e"),
            )
        )
        p = build_reader_index(xml)
        # "Two." sits between 1.1 and 1.3; "Four." cannot be both 1.4 and 1.5; "Seven." has nothing after it
        self.assertEqual(sorted(_sections_by_id(p)), ["s1.1", "s1.2", "s1.3", "s1.6"])
        x = _xrefs(p)
        self.assertEqual(x["Section 1.2"]["target"], "s1.2")
        self.assertEqual(x["Section 1.5"]["status"], "unresolved")

    def test_a_missing_section_stops_the_count(self) -> None:
        def stated(uuid: str, num: str) -> str:
            return _section(uuid, "Heading.", f"this Section {num} applies")

        xml = _doc(
            _article("a1", "ARTICLE I A", stated("s1", "1.1") + stated("s2", "1.2") + stated("s3", "1.3") + _section("s4", "Four.", "x"))
            + _article("a2", "ARTICLE II B", _section("s5", "One.", "x") + stated("s6", "2.3") + _section("s7", "Four.", "x"))
            + _article(
                "a3",
                "ARTICLE III C",
                _section("s8", "One.", "this Section 3.1 applies, and so does this Section 3.2, and again this Section 3.1") + stated("s9", "3.2"),
            )
        )
        secs = _sections_by_id(build_reader_index(xml))
        # Article II: its second section says it is 2.3. Either a section is missing before it or the phrase is left
        # from an earlier draft; nothing tells which, so no section of the article gets a number
        self.assertEqual(
            {k: v["anchor"]["s"] for k, v in secs.items()},
            {"s1.1": "s1", "s1.2": "s2", "s1.3": "s3", "s1.4": "s4", "s3.1": "s8", "s3.2": "s9"},
        )

    def test_a_stated_number_that_the_count_disagrees_with_numbers_nothing(self) -> None:
        def stated(uuid: str, num: str) -> str:
            return _section(uuid, "Heading.", f"this Section {num} applies")

        xml = _doc(
            _article(
                "a3",
                "ARTICLE III TERMS",
                stated("s1", "3.1")
                + stated("s2", "3.2")
                + _section("s3", "Three.", "x")
                + _section("s4", "Adjustments.", "under this Section 3.3, and this Section 3.3 again")
                + _section("s5", "Five.", "x")
                + stated("s6", "3.6"),
            )
        )
        secs = _sections_by_id(build_reader_index(xml))
        # "Adjustments." is the fourth section and the count from 3.2 and back from 3.6 says 3.4: a stale phrase
        self.assertEqual({k: v["anchor"]["s"] for k, v in secs.items()}, {"s3.1": "s1", "s3.2": "s2", "s3.6": "s6"})

    def test_two_sections_stating_one_number_fix_nothing(self) -> None:
        xml = _doc(
            _article("a1", "ARTICLE I A", _section("s1", "One.", "this Section 1.1 applies") + _section("s2", "Two.", "this Section 1.1 applies"))
        )
        self.assertEqual(_sections_by_id(build_reader_index(xml)), {})

    def test_one_number_printed_in_every_article(self) -> None:
        xml = _doc(
            _article("a1", "ARTICLE I A", _section("s1", "Section 1.1 Terms.", "this Section 1.1") + _section("s2", "Section 1.2 More.", "x"))
            + _article("a2", "ARTICLE II B", _section("s3", "Section 1.1 Merger.", "this Section 2.1") + _section("s4", "Section 1.2 Closing.", "See Section 1.2 and Section 2.2."))
        )
        p = build_reader_index(xml)
        secs = _sections_by_id(p)
        self.assertEqual({k: v["anchor"]["s"] for k, v in secs.items()}, {"s1.1": "s1", "s1.2": "s2", "s2.1": "s3", "s2.2": "s4"})
        self.assertEqual(p["stats"]["duplicate_section_numbers"], [])
        x = _xrefs(p)
        self.assertEqual((x["Section 1.2"]["target"], x["Section 2.2"]["target"]), ("s1.2", "s2.2"))

    def test_text_inside_a_renumbered_section_goes_by_the_new_number(self) -> None:
        xml = _doc(
            _article("a1", "ARTICLE I A", _section("s1", "Section 1.1 Terms.", "this Section 1.1") + _section("s2", "Section 1.2 More.", "x"))
            + _article(
                "a2",
                "ARTICLE II B",
                _section("s3", "Section 1.1 Merger.", "this Section 2.1")
                + _section("s4", "Section 1.2 Closing.", "(a) “Closing Date” means the day of the Closing.", "(b) Subject to clause (a), the parties close."),
            )
        )
        p = build_reader_index(xml)
        self.assertEqual(_xrefs(p)["clause (a)"]["target"], "s2.2(a)")
        self.assertEqual(_term(p, "Closing Date")["decls"][0]["in"], "Section 2.2")

    def test_articles_that_lost_their_numbers_count_by_position(self) -> None:
        xml = _doc(
            _article("a1", "DEFINITIONS", _section("s1", "Terms.", "this Section 1.1"))
            + _article("a2", "THE MERGER", _section("s2", "Merger.", "this Section 2.1") + _section("s3", "Closing.", "See Article III and Section 3.1."))
            + _article("a3", "REPRESENTATIONS", _section("s4", "Organization.", "x"))
        )
        p = build_reader_index(xml)
        self.assertEqual([(n["id"], n["label"], n["title"]) for n in p["outline"]], [("a1", "1", "DEFINITIONS"), ("a2", "2", "THE MERGER"), ("a3", "3", "REPRESENTATIONS")])
        x = _xrefs(p)
        self.assertEqual((x["Article III"]["target"], x["Section 3.1"]["target"]), ("a3", "s3.1"))

    def test_numbered_agreement_is_left_alone(self) -> None:
        p = build_reader_index(BASIC)
        self.assertEqual(p["stats"]["sections_positional"], 0)
        self.assertEqual(sorted(_sections_by_id(p)), ["s1.1", "s3.1", "s3.2"])

    def test_complete_printed_numbers_outrank_stale_self_references(self) -> None:
        xml = _doc(
            _article("a1", "1. A", _section("s1", "1.1 One.", "x"))
            + _article("a2", "2. B", _section("s2", "2.1 One.", "x"))
            + _article("a3", "3. C", _section("s3", "3.1 One.", "x") + _section("s4", "3.2 Two.", "x"))
            + _article(
                "a4",
                "4. D",
                _section("s5", "4.1 One.", "x")
                + _section("s6", "4.2 Two.", "under this Section 3.2")
                + _section("s7", "4.3 Three.", "See Section 4.2.")
                + _section("s8", "4.4 Four.", "under this Section 3.4"),
            )
        )
        p = build_reader_index(xml)
        # A draft that once lacked Article C still says "this Section 3.2"; the headings print their numbers
        self.assertEqual(p["stats"]["sections_positional"], 0)
        self.assertEqual([n["id"] for n in p["outline"]], ["a1", "a2", "a3", "a4"])
        self.assertEqual(_xrefs(p)["Section 4.2"]["target"], "s4.2")

    def test_garbled_printed_article_numbers_stand_only_where_the_count_agrees(self) -> None:
        xml = _doc(
            _article("a1", "1 ARTICLE THE MERGER", _section("s1", "Merger.", "x"))
            + _article("a2", "3 ARTICLE CONVERSION", _section("s2", "Conversion.", "under this Section 2.1"))
            + _article("a3", "5 ARTICLE DISCLOSURE", _section("s3", "Schedules.", "x"))
            + _article(
                "a4",
                "7 ARTICLE REPRESENTATIONS",
                _section("s4", "Organization.", "this Section 4.1") + _section("s5", "Authority.", "Subject to Article III, this Section 4.2"),
            )
        )
        p = build_reader_index(xml)
        self.assertEqual([(n["id"], n["title"]) for n in p["outline"]][:3], [("a1", "1 ARTICLE THE MERGER"), ("a2", "3 ARTICLE CONVERSION"), ("a3", "5 ARTICLE DISCLOSURE")])
        self.assertEqual(_xrefs(p)["Article III"]["target"], "a3")

    def test_disclosure_schedule_item_is_not_a_numbered_paragraph(self) -> None:
        xml = _doc(
            _article(
                "a5",
                "ARTICLE V REPRESENTATIONS",
                _section(
                    "s1",
                    "Section 5.12 Employee Benefits.",
                    "(a) Each Plan is listed.",
                    "Section 5.12.5 of the Company Disclosure Schedule sets forth each severance plan.",
                    "(b) No Plan is subject to Title IV.",
                    "5.12.6 Other Plans. See Section 5.12(b).",
                ),
            )
        )
        p = build_reader_index(xml)
        x = _xrefs(p)
        self.assertEqual(x["Section 5.12.5"]["status"], "disclosure_schedule")
        self.assertEqual(x["Section 5.12(b)"]["target"], "s5.12(b)")
        subs = [n["id"] for n in _sections_by_id(p)["s5.12"].get("children", [])]
        self.assertEqual(subs, ["s5.12(a)", "s5.12(b)", "s5.12.6"])


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


ROUND_TWO = _doc(
    _article(
        "a3",
        "ARTICLE III REPRESENTATIONS",
        _section("s322", "Section 3.22 Taxes", "(a) Filed.", "(g) Paid."),
    )
    + _article(
        "a5",
        "ARTICLE V COVENANTS",
        _section("s52", "Section 5.2 Conduct", "(a) Ordinary course.", "(b) Restrictions.", "(i) no dividends.", "(ii) no splits."),
    )
)


def _round_two(text: str) -> dict[str, dict[str, Any]]:
    xml = ROUND_TWO.replace("</body>", _article("a9", "ARTICLE IX MISC", _section("s91", "Section 9.1 Refs", text)) + "</body>")
    return _xrefs(build_reader_index(xml))


class RoundTwoRuleTests(unittest.TestCase):
    """Round-2 census rules (2026-11-20): U05 "thereof" and U13 clause lists qualified by a section."""

    def test_thereof_inside_a_parenthetical_keeps_the_section(self) -> None:
        x = _round_two("The representations in Section 3.22 (except for paragraph (g) thereof) are true.")
        self.assertEqual((x["Section 3.22"]["status"], x["Section 3.22"].get("target")), ("resolved", "s3.22"))
        x = _round_two("A breach of Section 5.2(b) (other than clause (ii)(B) thereof) applies.")
        self.assertEqual(x["Section 5.2(b)"]["status"], "resolved")

    def test_thereof_after_the_reference_still_names_another_document(self) -> None:
        x = _round_two("The Company complies with the DGCL, including Section 3.22 (as amended) thereof.")
        self.assertEqual(x["Section 3.22"]["status"], "unresolved")
        x = _round_two("Under the Lease, Sections 5.2 and 3.22 thereof apply.")
        self.assertEqual(x["Sections 5.2"]["status"], "unresolved")

    def test_section_printed_after_a_clause_list(self) -> None:
        x = _round_two("The covenants in clauses (i) through (xxi) of Section 5.2(b) apply.")
        self.assertEqual((x["clauses (i)"]["status"], x["clauses (i)"].get("target")), ("resolved", "s5.2(b)(i)"))
        x = _round_two("Those proposals in clauses (D), (E) and (J), of Section 5.2(b), in each case, need a vote.")
        self.assertEqual((x["clauses (D)"]["status"], x["clauses (D)"].get("target")), ("section_fallback", "s5.2(b)"))
        x = _round_two("A breach of clause (ii) of the first sentence of Section 5.2(b) counts.")
        self.assertEqual((x["clause (ii)"]["status"], x["clause (ii)"].get("target")), ("resolved", "s5.2(b)(ii)"))
        x = _round_two("The amounts in clauses (i) through (xxi) above of this Section 3.22 are paid.")
        self.assertEqual(x["clauses (i)"].get("target"), "s3.22")

    def test_clause_thereof_takes_the_section_printed_before(self) -> None:
        x = _round_two("A breach of Section 5.2(b) (other than clauses (i) and (ii) thereof) applies.")
        self.assertEqual((x["clauses (i)"]["status"], x["clauses (i)"].get("target")), ("resolved", "s5.2(b)(i)"))
        self.assertEqual(x["(ii)"].get("target"), "s5.2(b)(ii)")
        x = _round_two("Section 5.2 (Conduct) shall be enforced as provided in clause (b) thereof.")
        self.assertEqual(x["clause (b)"].get("target"), "s5.2(b)")
        x = _round_two("Section 3.22 (other than paragraph (c) thereof) applies.")
        self.assertEqual((x["paragraph (c)"]["status"], x["paragraph (c)"].get("target")), ("section_fallback", "s3.22"))

    def test_clause_thereof_after_a_named_thing_stays_unlinked(self) -> None:
        x = _round_two("For this Section 5.2(b) only, the exceptions to the Material Adverse Effect definition in clause (i) thereof do not apply.")
        self.assertNotIn("target", x["clause (i)"])
        x = _round_two("Under Section 5.2. The Lease and clause (i) thereof apply.")
        self.assertNotIn("target", x["clause (i)"])
        x = _round_two("The DGCL, including Section 262 and clause (i) thereof, applies.")
        self.assertNotIn("target", x["clause (i)"])

    def test_roman_article_title_without_the_word_article(self) -> None:
        xml = _doc(
            _article("r1", "I. THE MERGER", _section("r11", "1.1 The Merger.", "The Merger occurs."))
            + _article("r3", "III. REPRESENTATIONS AND WARRANTIES", _section("r31", "3.1 Organization.", "Subject to this Article III and Article I. A Tax under Section 1.1502-6 or Section 3 applies."))
            + _article("r4", "IV. other", "")
        )
        p = build_reader_index(xml)
        self.assertEqual([n["id"] for n in p["outline"]], ["a1", "a3"])
        x = _xrefs(p)
        self.assertEqual((x["Article III"]["status"], x["Article III"].get("target")), ("resolved", "a3"))
        self.assertEqual(x["Article I"].get("target"), "a1")
        self.assertNotIn("target", x["Section 3"])  # "III." is an article title, not a "SECTION 3" heading

    def test_late_section_of_another_document_does_not_qualify(self) -> None:
        x = _round_two("A default under clause (ii) of the first sentence of Section 5.2(b) of the Credit Agreement.")
        self.assertNotIn("target", x["clause (ii)"])
        x = _round_two("Payments under clause (ii) and the fees of Counsel under Section 5.2 apply.")
        self.assertNotIn("target", x["clause (ii)"])


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


class BodyClauseGuardTests(unittest.TestCase):
    """Body clause references that the inline and definitions guards left unlinked (2026-11-21 blind sample)."""

    def _all(self, xml: str) -> list[tuple[str, str, str | None]]:
        return [(r["text"], r["status"], r.get("target")) for r in build_reader_index(xml)["xrefs"]]

    # B1: "this clause (b)" inside paragraph (b) is that paragraph, not an inline item
    def test_this_clause_links_own_paragraph(self) -> None:
        xml = _doc(
            _section(
                "s31",
                "Section 3.1 Purchased Assets",
                "(a) all inventory;",
                "(b) all equipment, and all personal property described in this clause (b) is transferred;",
                "(c) all contracts.",
            )
        )
        self.assertEqual(self._all(xml), [("clause (b)", "resolved", "s3.1(b)")])

    def test_this_clause_in_nested_paragraph(self) -> None:
        xml = _doc(
            _section(
                "s51",
                "Section 5.1 Covenants",
                "(a) General.",
                "(b)(i) the Company shall not, subject to the last sentence of this clause (i), sell assets;",
                "(ii) other items.",
            )
        )
        self.assertEqual(self._all(xml), [("clause (i)", "resolved", "s5.1(b)(i)")])

    def test_this_clause_needs_own_leading_item(self) -> None:
        # "this clause (c)" inside paragraph (b): not the paragraph's own number, so the inline reading stays
        xml = _doc(
            _section(
                "s31",
                "Section 3.1 Assets",
                "(a) inventory;",
                "(b) equipment (c) and the items in this clause (c) and (c) again;",
                "(c) contracts.",
            )
        )
        self.assertEqual(self._all(xml)[0][:2], ("clause (c)", "unresolved"))

    def test_bare_clause_in_own_paragraph(self) -> None:
        # no "this": B1 does not apply, and the B3 guard keeps a link from a paragraph to itself unlinked
        xml = _doc(_section("s31", "Section 3.1 Assets", "(a) inventory;", "(b) equipment and the items in clause (b);"))
        self.assertEqual(self._all(xml), [("clause (b)", "unresolved", None)])
        # "this clause (b)" still links its own paragraph (B1)
        xml = _doc(_section("s31", "Section 3.1 Assets", "(a) inventory;", "(b) equipment and the items in this clause (b);"))
        self.assertEqual(self._all(xml), [("clause (b)", "resolved", "s3.1(b)")])

    # T1 (third review): B3 keeps the inline reading for a sentence or "such" tail, an inline lead-in list, and a
    # spaced item that opens an inline list
    def test_t1_sentence_tail_and_lead_in_list_stay_inline(self) -> None:
        xml = _doc(
            _section(
                "s51",
                "Section 5.1 Covenants",
                "The Company shall not (a) sell assets or (b) incur debt, except as set forth below.",
                "(a) Exceptions for assets.",
                "(b) The restriction in clause (a) of the first sentence of this Section and clause (a) above do not apply to Parent.",
            )
        )
        self.assertEqual([x[1] for x in self._all(xml)], ["unresolved", "unresolved"])

    def test_t1_preceding_sentence_and_such_letter_stay_inline(self) -> None:
        xml = _doc(
            _section("s51", "Section 5.1 Covenants", "(a) Exceptions for assets.", "(b) Other.", "(c) Nothing in clause (a) of the preceding sentence or clause (a) of such letter applies.")
        )
        self.assertEqual([x[1] for x in self._all(xml)], ["unresolved", "unresolved"])

    def test_t1_spaced_item_after_section_opens_inline_list(self) -> None:
        xml = _doc(
            _section("s42", "Section 4.2 X", "y")
            + _section(
                "s51",
                "Section 5.1 Covenants",
                "(a) Exceptions for assets.",
                "(b) Subject to Section 4.2 (a) the Company shall do X and (b) Parent shall do Y, and clause (a) shall survive.",
            )
        )
        self.assertEqual(self._all(xml)[1][:2], ("clause (a)", "unresolved"))

    def test_t1_qualified_clause_is_not_a_spaced_item(self) -> None:
        # "clause (a) of this Section 6.3" is a reference, not an inline "(a)": the later "clause (a)" still links
        xml = _doc(
            _section(
                "s63",
                "Section 6.3 Covenants",
                "(a) enter into any new terminaling Contract;",
                "(b) other than a Contract of the type contemplated by clause (a) of this Section 6.3, as to which the provisions of clause (a) shall apply.",
            )
        )
        self.assertEqual(self._all(xml)[-1], ("clause (a)", "resolved", "s6.3(a)"))

    def test_relaxed_link_to_another_paragraph_stays(self) -> None:
        # "(a)" printed twice as a reference inside paragraph (b): B3 lifts the old count and links paragraph (a)
        xml = _doc(_section("s31", "Section 3.1 Assets", "(a) inventory;", "(b) equipment, other than items in clause (a), and subject to clause (a) above;"))
        self.assertEqual(self._all(xml), [("clause (a)", "resolved", "s3.1(a)"), ("clause (a)", "resolved", "s3.1(a)")])

    # B2: a definitions section keeps a clause link inside the reference's own scope chain
    _DEFS = (
        "(a) Defined terms: “Alpha” means a. “Beta” means b. “Gamma” means c. “Delta” means d. “Epsilon” means e.",
        "(b) Limits.",
        "(i) Losses are capped.",
        "(ii) Deductible applies.",
        "(iii) Losses may be aggregated for purposes of clause (i).",
    )

    def test_definitions_section_keeps_scoped_clause(self) -> None:
        xml = _doc(_section("s102", "Section 10.2 Indemnification", *self._DEFS))
        self.assertEqual(self._all(xml), [("clause (i)", "resolved", "s10.2(b)(i)")])

    def test_definitions_section_drops_definition_tail(self) -> None:
        texts = (*self._DEFS[:4], "(iii) Liens in clause (i) of the definition of Permitted Liens are excluded.")
        xml = _doc(_section("s102", "Section 10.2 Indemnification", *texts))
        self.assertEqual(self._all(xml)[0][:2], ("clause (i)", "unresolved"))

    def test_definitions_section_drops_top_level_item(self) -> None:
        # the section's own "(b)" is in every definition's scope chain, so it proves nothing
        texts = (*self._DEFS[:2], "(c) “Zeta” means the items in clause (b).")
        xml = _doc(_section("s102", "Section 10.2 Indemnification", *texts))
        self.assertEqual(self._all(xml)[0][:2], ("clause (b)", "unresolved"))

    def test_definitions_section_drops_link_across_a_declaration(self) -> None:
        # a definition paragraph between the target and the reference: the node may be another definition's item
        texts = (
            "(a) Terms:",
            "(i) first.",
            "“Beta” means b. “Gamma” means c. “Delta” means d. “Epsilon” means e. “Zeta” means f.",
            "(ii) the items in clause (i) above.",
        )
        xml = _doc(_section("s11", "Section 1.1 Definitions", *texts))
        self.assertEqual(self._all(xml)[0][:2], ("clause (i)", "unresolved"))
        # an item that declares a term in passing does not break the scope
        texts = ("(a) Terms:", "(i) first.", "(ii) “Beta” means b. “Gamma” means c. “Delta” means d. “Epsilon” means e. “Zeta” means f.", "(iii) the items in clause (i) above.")
        xml = _doc(_section("s11", "Section 1.1 Definitions", *texts))
        self.assertEqual(self._all(xml), [("clause (i)", "resolved", "s1.1(a)(i)")])

    # B3: an item cited twice is not an inline item; one that also occurs outside the references still is
    def test_clause_cited_twice_links(self) -> None:
        xml = _doc(
            _section(
                "s412",
                "Section 4.12 Taxes",
                "(a) Except as set forth in clause (b) below, all Taxes are paid and, except as set forth in clause (b) below, filed.",
                "(b) No audit is pending.",
            )
        )
        self.assertEqual(self._all(xml), [("clause (b)", "resolved", "s4.12(b)"), ("clause (b)", "resolved", "s4.12(b)")])

    def test_inline_item_stays_unlinked(self) -> None:
        xml = _doc(
            _section(
                "s412",
                "Section 4.12 Taxes",
                "(a) The Company shall (i) file and (ii) pay, provided that clause (i) applies only to returns.",
                "(b) No audit is pending.",
                "(i) one.",
            )
        )
        self.assertEqual(self._all(xml)[0][:2], ("clause (i)", "unresolved"))

    def test_definition_tail_keeps_old_count(self) -> None:
        xml = _doc(
            _section(
                "s412",
                "Section 4.12 Liens",
                "(a) Liens in clause (b) of the definition of Permitted Liens and clause (b) of the definition of Permitted Liens.",
                "(b) No audit is pending.",
            )
        )
        self.assertEqual([x[1] for x in self._all(xml)], ["unresolved", "unresolved"])

    def test_section_named_later_keeps_old_count(self) -> None:
        xml = _doc(
            _section(
                "s62",
                "Section 6.2 Conditions",
                "(a) Representations. (i) Section 3.01(a) is true; (ii) the representations in clauses (a) and (b) and, "
                "solely as to the Company, clauses (d) and (e) of Section 3.02 are true; (iii) clauses (a) and (b) above.",
                "(b) Covenants.",
            )
        )
        self.assertEqual(self._all(xml)[1][:2], ("clauses (a)", "unresolved"))

    def test_leading_item_cited_in_its_paragraph_stays_inline(self) -> None:
        # "(i)" opens the paragraph (after "(b)") and is cited later in it: the nearest "(i)" is that paragraph itself,
        # so the B3 self-link guard keeps the old inline reading
        xml = _doc(
            _section(
                "s51",
                "Section 5.1 Covenants",
                "(a) Hiring.",
                "(i) no hires.",
                "(b) (i) no raises, (ii) no loans, except, in the case of clauses (i) and (ii), as required by Law.",
            )
        )
        self.assertEqual(self._all(xml)[0], ("clauses (i)", "unresolved", None))


class SecondReviewFixTests(unittest.TestCase):
    """Fixes from the second code review of enricher 0.10.0 (2026-11-21)."""

    def _all(self, xml: str) -> list[tuple[str, str, str | None]]:
        return [(r["text"], r["status"], r.get("target")) for r in build_reader_index(xml)["xrefs"]]

    # R1: a qualified clause whose section lists its own items in the lead-in stays at section level
    def test_qualified_clause_of_lead_in_item(self) -> None:
        xml = _doc(
            _section(
                "s64",
                "Section 6.4 Filings",
                "Parent shall (i) file the forms and (ii) pay the fees, subject to the following:",
                "(a) Filings.",
                "(b) Fees.",
                "(c) Expenses:",
                "(i) costs.",
            )
            + _section("s71", "Section 7.1 Conditions", "the filing under clause (i) of Section 6.4 is made.")
        )
        self.assertEqual(self._all(xml), [("clause (i)", "section_fallback", "s6.4"), ("Section 6.4", "resolved", "s6.4")])

    def test_qualified_clause_deep_lookup_still_links(self) -> None:
        xml = _doc(
            _section("s64", "Section 6.4 Filings", "Parent shall file the forms:", "(a) Filings.", "(i) costs.", "(b) Fees.")
            + _section("s71", "Section 7.1 Conditions", "the filing under clause (i) of Section 6.4 is made.")
        )
        self.assertEqual(self._all(xml), [("clause (i)", "resolved", "s6.4(a)(i)"), ("Section 6.4", "resolved", "s6.4")])

    # T2: a reference inside the lead-in ("Subject to Section 6.4(i)") is not a lead-in item
    def test_qualified_clause_lead_in_reference_is_not_an_item(self) -> None:
        xml = _doc(
            _section("s64", "Section 6.4 A", "(i) q")
            + _section("s65", "Section 6.5 B", "Subject to Section 6.4(i), Parent shall:", "(a) x.", "(b) y:", "(i) z.")
            + _section("s71", "Section 7.1 C", "the filing under clause (i) of Section 6.5 is made.")
        )
        self.assertEqual(self._all(xml)[1], ("clause (i)", "resolved", "s6.5(b)(i)"))

    # R2: paragraph nodes under an inferred section are inferred too; printed outlines do not change
    def test_inferred_section_marks_its_paragraphs(self) -> None:
        xml = _doc(
            _article("a1", "ARTICLE I A", _section("s1", "Section 1.1 A", "(a) x") + _section("s2", "B.", "(a) y", "(i) z") + _section("s3", "Section 1.3 C", "(a) z"))
        )
        p = build_reader_index(xml)
        secs = _sections_by_id(p)
        self.assertEqual(secs["s1.2"].get("inferred"), True)
        flags = {}
        stack = list(p["outline"])
        while stack:
            n = stack.pop()
            flags[n["id"]] = n.get("inferred")
            stack.extend(n.get("children", []))
        self.assertEqual((flags["s1.2(a)"], flags["s1.2(a)(i)"]), (True, True))
        self.assertEqual((flags["s1.1(a)"], flags["s1.3(a)"]), (None, None))

    # R3: L1 judges a qualified clause by the text after its section, and a tail naming only a party is not a document
    def _one(self, sentence: str) -> list[tuple[str, str, str | None]]:
        return self._all(_doc(_section("s11", "Section 1.1 A", "x") + _section("s12", "Section 1.2 B", sentence)))

    def test_qualified_clause_into_missing_section_matches_its_section(self) -> None:
        got = self._one("Subject to clause (a) of Section 4.2 and Section 4.2, the price is paid.")
        self.assertEqual([x[1] for x in got], ["unresolved", "unresolved", "unresolved"])
        got = self._one("See clause (a) of Section 4.2 of the Escrow Agreement.")
        self.assertEqual([x[1] for x in got], ["external", "external"])

    def test_party_tail_is_not_another_document(self) -> None:
        self.assertEqual(self._one("The representations in Section 4.7 of the Company are true.")[0][1], "unresolved")
        self.assertEqual(self._one("See Section 4.7 of Parent.")[0][1], "unresolved")
        self.assertEqual(self._one("See Section 4.7 of the Company Charter.")[0][1], "external")
        self.assertEqual(self._one("See Section 4.7 of the Company’s Bylaws.")[0][1], "external")
        # R3 sample: a party name that starts a document name
        self.assertEqual(self._one("Purchase rights end under Section 19(c) of the Company 423 Plan.")[0][1], "external")
        self.assertEqual(self._one("Perform the fifth sentence of Section 2(a) of the Parent/HIM Agreement.")[0][1], "external")

    def test_qualified_clause_of_section_thereof_stays_external(self) -> None:
        got = self._one("The Confidentiality Agreement continues, notwithstanding clause (i) of Section 16 thereof.")
        self.assertEqual(got[0][:2], ("clause (i)", "external"))

    # R4: "Article I&II" keeps its link; "Schedule C&D" and "D&O" stay names
    def test_roman_ampersand_roman_is_a_reference(self) -> None:
        xml = _doc(
            _article("a1", "ARTICLE I A", _section("s11", "Section 1.1 A", "x"))
            + _article("a2", "ARTICLE II B", _section("s21", "Section 2.1 B", "See Article I&amp;II. Schedule C&amp;D applies. Under the D&amp;O policy."))
        )
        self.assertEqual(self._all(xml), [("Article I", "resolved", "a1")])

    # R5: a court's Disclosure Statement is not a disclosure schedule; a party-named Disclosure Letter is one
    def test_disclosure_statement_order_and_party_letter(self) -> None:
        self.assertEqual(self._one("See Section 3.1(b) of the Disclosure Statement Order.")[0][1], "external")
        self.assertEqual(self._one("See Section 4.2 of the Debtors’ Disclosure Statement.")[0][1], "external")
        self.assertEqual(self._one("See Section 4.2 of the Disclosure Statement.")[0][1], "disclosure_schedule")
        self.assertEqual(self._one("Except as set forth in Section A of Power Disclosure Letter, x.")[0][1], "disclosure_schedule")
        self.assertEqual(self._one("See Section 4.2 of Seller Disclosure Schedule.")[0][1], "disclosure_schedule")

    def test_printed_outline_is_byte_identical(self) -> None:
        import json

        xml = _doc(_article("a1", "ARTICLE I A", _section("s1", "Section 1.1 A", "(a) x", "(i) y") + _section("s2", "Section 1.2 B", "(a) z")))
        got = json.dumps(build_reader_index(xml)["outline"], sort_keys=True)
        self.assertNotIn('"inferred"', got)
        self.assertEqual(
            got,
            json.dumps(
                [
                    {"anchor": {"s": "a1"}, "children": [
                        {"anchor": {"s": "s1"}, "children": [
                            {"anchor": {"s": "s1", "t": 0}, "children": [
                                {"anchor": {"s": "s1", "t": 1}, "id": "s1.1(a)(i)", "kind": "sub", "label": "1.1(a)(i)", "title": "(i) y"}],
                             "id": "s1.1(a)", "kind": "sub", "label": "1.1(a)", "title": "(a) x"}],
                         "id": "s1.1", "kind": "section", "label": "1.1", "title": "Section 1.1 A"},
                        {"anchor": {"s": "s2"}, "children": [
                            {"anchor": {"s": "s2", "t": 0}, "id": "s1.2(a)", "kind": "sub", "label": "1.2(a)", "title": "(a) z"}],
                         "id": "s1.2", "kind": "section", "label": "1.2", "title": "Section 1.2 B"}],
                     "id": "a1", "kind": "article", "label": "I", "title": "ARTICLE I A"}
                ],
                sort_keys=True,
            ),
        )


class ReviewFixTests(unittest.TestCase):
    """Fixes from the code review of enricher 0.10.0 (2026-11-20)."""

    def _all(self, xml: str) -> list[tuple[str, str, str | None]]:
        return [(r["text"], r["status"], r.get("target")) for r in build_reader_index(xml)["xrefs"]]

    # F1: a long run of spaced enumerators made LATE_QUALIFIED backtrack exponentially
    def test_spaced_enumerator_run_is_fast(self) -> None:
        import time

        items = " ".join(f"({i})" for i in range(1, 31))
        xml = _doc(_section("s1", "Section 1.1 Payments", f"(a) the amounts in clause {items} are paid. See clause {items} of Section 1.1."))
        t0 = time.perf_counter()
        got = self._all(xml)
        self.assertLess(time.perf_counter() - t0, 0.5)
        self.assertEqual(got[0][:2], ("clause (1)", "unresolved"))
        self.assertEqual(got[1], ("clause (1)", "section_fallback", "s1.1"))
        # a long run of spaces after a clause was cubic (three \s* in a row)
        t0 = time.perf_counter()
        self._all(_doc(_section("s1", "Section 1.1 Payments", "(a) see clause (a)" + " " * 3000 + "x.")))
        self.assertLess(time.perf_counter() - t0, 0.5)

    # F2: a clause whose tail names its parent paragraph takes that paragraph's clause, not the nearest one
    def test_clause_of_named_subparagraph_takes_that_parent(self) -> None:
        def run(text: str) -> list[tuple[str, str, str | None]]:
            return self._all(
                _doc(_section("s51", "Section 5.1 Caps", "(a) caps:", "(i) one;", "(ii) two;", "(b) more:", "(i) three;", "(ii) four;", text))
            )

        for tail in ("subparagraph (a) above", "subsection (a)", "sub-clause (a)", "clause (a)", "paragraph (a)"):
            self.assertEqual(run(f"(iii) the cap in clause (ii) of {tail} applies.")[0], ("clause (ii)", "resolved", "s5.1(a)(ii)"), tail)
        # no unique (iii) under (c): as before the tie-break, ambiguous
        self.assertEqual(run("(iii) the cap in clause (ii) of paragraph (c) applies.")[0][1], "ambiguous")
        # the enumerator belongs to something named after it: no parent, no tie-break
        self.assertEqual(run("(iii) the cap in clause (ii) of paragraph (a) of Exhibit B applies.")[0][1], "ambiguous")
        self.assertEqual(run("(iii) the cap in clause (ii) of clause (a)(i) thereof applies.")[0][1], "ambiguous")
        # no named parent: the nearest scope still applies
        self.assertEqual(run("(iii) the cap in clause (ii) applies.")[0], ("clause (ii)", "resolved", "s5.1(b)(ii)"))

    # F4: an alias key ("annex 2.02" for "Annex 2.02(a)") is not another located attachment
    def test_attachments_located_counts_nodes_not_aliases(self) -> None:
        back = "<text>ANNEX 2.02(a)</text><text>Purchased assets.</text><text>ANNEX 2.02(b)</text><text>Excluded assets.</text>"
        p = build_reader_index(_doc(_section("s1", "Section 1.1 Refs", "See Annex 2.02."), back=back))
        self.assertEqual(len([n for n in p["outline"] if n["kind"] == "attachment"]), 2)
        self.assertEqual(p["stats"]["attachments_located"], 2)

    # F3: a section numbered by position says so; a printed one does not carry the field at all
    def test_inferred_numbers_are_marked(self) -> None:
        xml = _doc(
            _article("a1", "ARTICLE I A", _section("s1", "Section 1.1 A", "x") + _section("s2", "B.", "y") + _section("s3", "Section 1.3 C", "z"))
            + _article("a2", "ARTICLE II B", _section("s4", "Section 2.1 A", "x") + _section("s5", "Closing.", "y") + _section("s6", "Section 2.3 C", "z"))
        )
        p = build_reader_index(xml)
        secs = _sections_by_id(p)
        self.assertEqual({k: v.get("inferred") for k, v in secs.items()}, {"s1.1": None, "s1.2": True, "s1.3": None, "s2.1": None, "s2.2": True, "s2.3": None})
        self.assertNotIn("inferred", p["outline"][0])
        self.assertNotIn("inferred", build_reader_index(BASIC)["outline"][1]["children"][0])

    # F6: a body with no "clause N.M" (and no "Section N.M") is not UK drafting
    def test_body_without_dotted_clauses_is_not_uk_drafting(self) -> None:
        back = "<text>SCHEDULE 4</text><text>LIMITATIONS</text><text>1.1 The Warranties are given under clause 9.2.</text><text>2.1 Limits.</text>"
        body = _article("c9", "9 WARRANTIES", _section("s92", "9.2 Limits", "x"))
        x = _xrefs(build_reader_index(_doc(body, back=back)))
        self.assertEqual((x["clause 9.2"]["status"], x["clause 9.2"].get("target")), ("unresolved", None))
        body = _article("c9", "9 WARRANTIES", _section("s92", "9.2 Limits", "x, subject to clause 9.1."))
        x = _xrefs(build_reader_index(_doc(body, back=back)))
        self.assertEqual(x["clause 9.2"]["target"], "s9.2")


class StatusAndTargetRuleTests(unittest.TestCase):
    """Enricher 0.10.0 rules on statuses and targets (reader_090 census, PROPOSALS.md rules 3, 5, 8-12)."""

    def _status(self, xml: str) -> dict[str, tuple[str, str | None]]:
        return {r["text"]: (r["status"], r.get("target")) for r in build_reader_index(xml)["xrefs"]}

    def _all(self, xml: str) -> list[tuple[str, str, str | None]]:
        return [(r["text"], r["status"], r.get("target")) for r in build_reader_index(xml)["xrefs"]]

    # rule 3: a large bare section number is a statute or another document
    def test_large_bare_section_numbers_are_external(self) -> None:
        refs = self._status(
            _doc(
                _article("a1", "ARTICLE I Merger", _section("s11", "Section 1.1 Merger", "x"))
                + _article(
                    "a2",
                    "ARTICLE II Refs",
                    _section(
                        "s21",
                        "Section 2.1 Refs",
                        "Appraisal rights under Section 262 apply. No payment is subject to Section 280G, "
                        + "and the election under Section 338(h)(10) is made. Reports under Section 16 are filed.",
                    ),
                )
            )
        )
        self.assertEqual(refs["Section 262"], ("external", None))
        self.assertEqual(refs["Section 280G"], ("external", None))
        self.assertEqual(refs["Section 338(h)(10)"], ("external", None))
        # below 40: may be the agreement's own article, so it stays unresolved
        self.assertEqual(refs["Section 16"], ("unresolved", None))

    def test_large_bare_number_with_its_own_node_still_resolves(self) -> None:
        flat = "".join(_article(f"f{n}", f"Section {n}. Item", "<text>(a) text.</text>") for n in (44, 45))
        refs = self._status(
            _doc(
                _section("s40", "Section 40 Notices", "x")
                + flat
                + _article("f46", "Section 46. Refs", "<text>See Section 40, Section 45 and Section 44(a). Section 47 is a statute.</text>")
            )
        )
        self.assertEqual(refs["Section 40"], ("resolved", "s40"))
        self.assertEqual(refs["Section 45"], ("resolved", "a45"))
        self.assertEqual(refs["Section 44(a)"], ("resolved", "s44(a)"))
        self.assertEqual(refs["Section 47"], ("external", None))

    def test_large_number_rule_leaves_other_heads_and_dotted_numbers(self) -> None:
        refs = self._status(_doc(_section("s1", "Section 1.1 Refs", "See Article 52, clause 41 and Section 41.2.")))
        self.assertEqual(refs["Article 52"], ("unresolved", None))
        self.assertEqual(refs["clause 41"], ("unresolved", None))
        self.assertEqual(refs["Section 41.2"], ("unresolved", None))

    # rule 10: self markers beat statute and other-document tails
    def _rule10(self, *texts: str) -> list[tuple[str, str, str | None]]:
        return self._all(_doc(_section("s63", "Section 6.3 Filings", "(a) x.") + _section("s64", "Section 6.4 Refs", *texts)))

    def test_this_and_hereof_keep_a_reference_internal(self) -> None:
        got = self._rule10(
            "The limits in this Section 6.3 apply to any registration of the Securities Act.",
            "Section 6.3 hereof and any registration of the Securities Act apply.",
            "This Section 6.3 under the Company Option Plan governs.",
        )
        self.assertEqual(got, [("Section 6.3", "resolved", "s6.3")] * 3)

    def test_a_verb_before_the_act_keeps_a_reference_internal(self) -> None:
        got = self._rule10("Section 6.3 shall not apply to the filings of the Exchange Act.")
        self.assertEqual(got, [("Section 6.3", "resolved", "s6.3")])

    def test_this_section_run_is_not_pushed_external_by_a_joined_run(self) -> None:
        got = self._rule10("Under this Section 6.3 or Section 13 of the Exchange Act, the filing is made.")
        self.assertEqual(got, [("Section 6.3", "resolved", "s6.3"), ("Section 13", "external", None)])

    def test_statute_lists_and_other_documents_stay_external(self) -> None:
        got = self._rule10(
            "Reports under Section 15, Section 15B or Section 15C of the Exchange Act are filed.",
            "Proxy rules in Section 14a-13 of the Exchange Act apply.",
            "As provided in Section 6.3 of the Company Option Plan.",
            "Section 6.3 or Section 13 of the Exchange Act applies.",
        )
        self.assertEqual([s for _, s, _ in got], ["external"] * 7)

    def test_self_markers_do_not_apply_in_back_matter(self) -> None:
        # a voting agreement's "this Section 4.3" is its own section, not the merger agreement's Section 4.3
        back = (
            "<text>EXHIBIT A</text><text>“Alpha” means a.</text><text>“Beta” means b.</text><text>“Gamma” means c.</text>"
            + "<text>No proposal arising out of a breach of this Section 6.3 or Section 5.2(a) of the Merger Agreement.</text>"
            + "<text>Section 6.3 shall not apply to the filings of the Exchange Act.</text>"
        )
        got = self._all(_doc(_section("s63", "Section 6.3 Filings", "(a) x."), back=back))
        self.assertEqual([s for _, s, _ in got], ["external"] * 3)

    # rule 9: a digit after a hyphen in a regulation number is not a list member
    def test_regulation_numbers_are_one_reference(self) -> None:
        xml = _doc(
            _article("a3", "SECTION 3 Covenants", _section("s31", "3.1 Tax", "x"))
            + _section(
                "s2",
                "Section 2.1 Refs",
                "A reorganization within the meaning of Section 1.368-2(g) of the Treasury Regulations.",
                "Liability under Section 1.1502-6 (or any similar provision) is excluded.",
                "See Treasury Regulations Section 301.7701-3.",
                "See Sections 2.1-3.1 and Section 2.1 - 3.",
            )
        )
        got = self._all(xml)
        self.assertEqual(
            got,
            [
                ("Section 1.368-2(g)", "external", None),
                ("Section 1.1502-6", "unresolved", None),
                ("Section 301.7701-3", "external", None),
                ("Sections 2.1", "resolved", "s2.1"),
                ("3.1", "resolved", "s3.1"),
                ("Section 2.1", "resolved", "s2.1"),
                ("3", "resolved", "a3"),
            ],
        )

    # rule 8: full attachment labels
    def test_letter_hyphen_number_exhibits_are_one_label(self) -> None:
        back = "<text>EXHIBIT C-1</text><text>Form of note.</text><text>Exhibit I-1</text><text>Form of letter.</text>"
        got = self._all(_doc(_section("s1", "Section 1.1 Refs", "See Exhibit C-1, Exhibits I-1 and C-2, and Exhibits I-III."), back=back))
        self.assertEqual(
            got,
            [
                ("Exhibit C-1", "resolved", "x-exhibit-c-1"),
                ("Exhibits I-1", "resolved", "x-exhibit-i-1"),
                ("C-2", "unlocated_attachment", None),
                ("Exhibits I", "unlocated_attachment", None),
                ("III", "unlocated_attachment", None),
            ],
        )

    def test_attachment_headings_keep_their_enumerators(self) -> None:
        back = (
            "<text>Annex 2.02(a)</text><text>Purchased assets.</text>"
            "<text>Annex 2.02(b)</text><text>Excluded assets.</text>"
            "<text>Exhibit B</text><text>Form of deed.</text>"
        )
        p = build_reader_index(_doc(_section("s1", "Section 1.1 Refs", "See Annex 2.02(a), Annex 2.02(b), Annex 2.02 and Exhibit B."), back=back))
        got = [(r["text"], r["status"], r.get("target")) for r in p["xrefs"]]
        self.assertEqual(
            got,
            [
                ("Annex 2.02(a)", "resolved", "x-annex-2.02(a)"),
                ("Annex 2.02(b)", "resolved", "x-annex-2.02(b)"),
                ("Annex 2.02", "resolved", "x-annex-2.02(a)"),
                ("Exhibit B", "resolved", "x-exhibit-b"),
            ],
        )
        labels = [n["label"] for n in p["outline"] if n["kind"] == "attachment"]
        self.assertEqual(labels, ["2.02(a)", "2.02(b)", "B"])

    # rule 12: small tail fixes
    def test_small_tail_fixes(self) -> None:
        got = self._all(
            _doc(
                _article("a13", "SECTION 13 Reports", _section("s131", "13.1 Filings", "x"))
                + _section("s22", "Section 2.2 Assets", "(a) x.")
                + _section("s25", "Section 2.5 Liabilities", "x")
                + _section("s71", "Section 7.1 Charter", "(c) x.")
                + _section(
                    "s9",
                    "Section 9.1 Refs",
                    "Reports under Sections 13 or 14 of the 1934 Act are filed.",
                    "As set out in Section 7.1(c) of the Company’s Amended and Restated Certificate of Incorporation.",
                    "Under Section 2.2 of Parent's Credit Agreement, and Sections 2.2 to 2.5, inclusive, of the Distribution Agreement.",
                    "See 12 CFR Section 225.2 and Treas. Regs. Section 1.1502-6.",
                    "Sections 2.2 to 2.5, inclusive, hereof apply. Sections 2.2 to 2.5, inclusive, apply.",
                )
            )
        )
        self.assertEqual([s for _, s, _ in got[:8]], ["external"] * 8)
        self.assertEqual(got[8:], [("Sections 2.2", "resolved", "s2.2"), ("2.5", "resolved", "s2.5")] * 2)

    def test_possessive_disclosure_schedule_is_a_disclosure_reference(self) -> None:
        got = self._all(
            _doc(
                _section("s71", "Section 7.1 Contracts", "(c) x.")
                + _section(
                    "s9",
                    "Section 9.1 Refs",
                    "Except as set forth in Section 7.1(c) of Seller’s Disclosure Schedule, and Section 7.1 of Parent's "
                    + "Disclosure Letter. Under Section 7.1 of Parent's Credit Agreement.",
                )
            )
        )
        self.assertEqual([s for _, s, _ in got], ["disclosure_schedule", "disclosure_schedule", "external"])

    # rule 11: "Section N" links to ARTICLE N in an agreement that writes "this Section N" inside article N
    def _self_style(self, signal: str, refs: str, back: str = "") -> list[tuple[str, str, str | None]]:
        return self._all(
            _doc(
                _article("a7", "ARTICLE VII CONDITIONS", _section("s71", "Section 7.1 Conditions", "x"))
                + _article("a8", "ARTICLE VIII TERMINATION", _section("s81", "Section 8.1 Termination", signal))
                + _article("a9", "ARTICLE IX MISCELLANEOUS", _section("s91", "Section 9.1 Notices", refs)),
                back=back,
            )
        )

    def test_self_style_agreement_links_bare_sections_to_articles(self) -> None:
        got = self._self_style(
            "The rights under this Section 8 survive.",
            "Termination under Section 8 hereof ends it. See Sections 7 and 8. Also Section 8 of the Voting Agreement, "
            + "Section 8 of Revenue Procedure 2004-53 and Section 8.l above.",
        )
        self.assertEqual(
            got,
            [
                ("Section 8", "resolved", "a8"),
                ("Section 8", "resolved", "a8"),
                ("Sections 7", "resolved", "a7"),
                ("8", "unresolved", None),
                ("Section 8", "external", None),
                ("Section 8", "external", None),  # L1: "of Revenue Procedure 2004-53" names another document
                ("Section 8", "unresolved", None),
            ],
        )

    def test_self_style_skips_statute_citations(self) -> None:
        got = self._self_style(
            "The rights under this Section 8 survive.",
            "Arbitration under 9 U.S.C., Section 8 et seq. applies. See Health Code, Chapter II, Section 8 et seq. "
            + "Registered form under Section 8f.103-1(c) of the Treasury Regulations. See Section 8.",
        )
        self.assertEqual([s for _, s, _ in got], ["resolved", "unresolved", "unresolved", "unresolved", "resolved"])

    def test_self_style_needs_the_signal_inside_its_own_article(self) -> None:
        got = self._self_style("The rights under Section 8 survive.", "As provided in this Section 8, and Section 8.")
        self.assertEqual([s for _, s, _ in got], ["unresolved"] * 3)

    def test_self_style_never_applies_in_back_matter(self) -> None:
        back = (
            "<text>EXHIBIT A</text><text>“Alpha” means the notice under Section 8.</text>"
            + "<text>“Beta” means b.</text><text>“Gamma” means c.</text>"
        )
        got = self._self_style("The rights under this Section 8 survive.", "x", back=back)
        self.assertEqual(got, [("Section 8", "resolved", "a8"), ("Section 8", "unresolved", None)])

    # rule 5: nearest-scope tie-break for ambiguous clauses
    def test_ambiguous_clause_takes_the_nearest_scope(self) -> None:
        got = self._all(
            _doc(
                _section(
                    "s42",
                    "Section 4.2 Conduct",
                    "(a) Parent shall not:",
                    "(i) issue shares;",
                    "(ii) sell assets, except as permitted by clause (i);",
                    "(b) The Company shall not:",
                    "(i) amend its charter;",
                    "(ii) merge, other than as provided in clause (i) above;",
                    "(c) Nothing in clause (i) of the Credit Agreement or clause (ii) of paragraph 2 of Annex C applies.",
                )
            )
        )
        self.assertEqual(
            got,
            [
                ("clause (i)", "resolved", "s4.2(a)(i)"),
                ("clause (i)", "resolved", "s4.2(b)(i)"),
                ("clause (i)", "external", None),
                # a clause of another document's paragraph: no tie-break
                ("clause (ii)", "ambiguous", None),
                ("paragraph 2", "unresolved", None),
                ("Annex C", "unlocated_attachment", None),
            ],
        )
        # gold fixture 869a5c84: "(other than clause (i) thereto)" is a clause of another document
        got = self._all(
            _doc(_section("s42", "Section 4.2 Conduct", "(a) x:", "(i) y.", "(ii) any transfer (other than clause (i) thereto).", "(b) z:", "(i) w."))
        )
        self.assertEqual(got, [("clause (i)", "ambiguous", None)])

    def test_ambiguous_clause_outside_any_paragraph_stays_ambiguous(self) -> None:
        got = self._all(
            _doc(_section("s42", "Section 4.2 Conduct", "Subject to clause (i), the parties agree:", "(a) x:", "(i) y.", "(b) z:", "(i) w."))
        )
        self.assertEqual(got, [("clause (i)", "ambiguous", None)])

    def test_qualifier_after_a_comma_or_a_list(self) -> None:
        got = self._all(
            _doc(
                _section("s52", "Section 5.2 Covenants", "(a) items:", "(A) one.", "(D) four.")
                + _section("s412", "Section 4.12 Plans", "(a) plans:", "(i) one.", "(xvi) sixteen.")
                + _section(
                    "s6",
                    "Section 6.1 Refs",
                    "Except clauses (A)-(D) of, the foregoing Section 5.2(a), none.",
                    "Under clauses (i) through (xiii) or clause (xvi) of Section 4.12(a), none.",
                )
            )
        )
        self.assertEqual(
            got,
            [
                ("clauses (A)", "resolved", "s5.2(a)(A)"),
                ("(D)", "resolved", "s5.2(a)(D)"),
                ("Section 5.2(a)", "resolved", "s5.2(a)"),
                ("clauses (i)", "resolved", "s4.12(a)(i)"),
                ("(xiii)", "section_fallback", "s4.12(a)"),
                ("clause (xvi)", "resolved", "s4.12(a)(xvi)"),
                ("Section 4.12(a)", "resolved", "s4.12(a)"),
            ],
        )


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


class RoundTwoLeadRuleTests(unittest.TestCase):
    """Round-2 lead rules L1 to L7 (2026-11-20, round2_PROPOSALS.md "For the lead after the merge")."""

    def _all(self, body_text: str, back: str = "", main: str = MAIN) -> list[tuple[str, str, str | None]]:
        xml = _doc(main + _article("a9", "ARTICLE IX MISC", _section("s91", "Section 9.1 Refs", body_text)), back=back)
        return [(x["text"], x["status"], x.get("target")) for x in build_reader_index(xml)["xrefs"]]

    # L1: no node and a tail that names something else
    def test_l1_tail_naming_another_document_is_external(self) -> None:
        refs = self._all("As set out in Section 7.3 of Revenue Procedure 2004-53 and Article 12 under the Stockholders Agreement.")
        self.assertIn(("Section 7.3", "external", None), refs)
        self.assertIn(("Article 12", "external", None), refs)

    def test_l1_keeps_self_markers_articles_and_existing_nodes(self) -> None:
        refs = self._all(
            "Under this Section 7.3 pursuant to Regulation D, Section 8.5 of Article IX, Section 2.1 under the Escrow Terms "
            "and Section 7.7 of the Company Disclosure Letter."
        )
        self.assertIn(("Section 7.3", "unresolved", None), refs)
        self.assertIn(("Section 8.5", "unresolved", None), refs)
        self.assertIn(("Section 2.1", "resolved", "s2.1"), refs)  # a node exists: L1 never applies
        self.assertIn(("Section 7.7", "disclosure_schedule", None), refs)

    def test_l1_in_back_matter_only_without_a_main_node(self) -> None:
        back = _back("EXHIBIT A", "ESCROW TERMS", "Section 7.3 of Revenue Procedure 2004-53 and Section 2.1 of Revenue Procedure 2004-53 apply.")
        refs = self._all("x", back=back)
        self.assertIn(("Section 7.3", "external", None), refs)
        self.assertIn(("Section 2.1", "unresolved", None), refs)

    # L5: an attachment of another document
    def test_l5_attachment_of_another_document(self) -> None:
        refs = self._all(
            "Under Section 2 of Exhibit B to the Escrow Agreement, Section 3.1 of Schedule 1 to the Company Disclosure Schedule, "
            "Section 4 of Exhibit C to the Agreement and Section 5 of Exhibit D hereto."
        )
        self.assertIn(("Section 2", "external", None), refs)
        self.assertIn(("Section 3.1", "disclosure_schedule", None), refs)
        self.assertIn(("Section 4", "unresolved", None), refs)  # this agreement's own exhibit
        self.assertIn(("Section 5", "unresolved", None), refs)

    # L3: an article title that spells its number
    def test_l3_spelled_article_title_is_numbered(self) -> None:
        xml = _doc(
            _article("a1", "ARTICLE ONE THE MERGER", _section("s11", "Section 1.1 Merger", "x"))
            + _article("a2", "Article Seventeen Terms", _section("s21", "Section 17.1 Terms", "Under Article One, Article 1, Article XVII, Article Two and Section 1."))
        )
        p = build_reader_index(xml)
        self.assertEqual([(n["id"], n["label"]) for n in p["outline"]], [("a1", "ONE"), ("a17", "Seventeen")])
        refs = [(x["text"], x["status"], x.get("target")) for x in p["xrefs"]]
        self.assertIn(("Article One", "resolved", "a1"), refs)
        self.assertIn(("Article 1", "resolved", "a1"), refs)
        self.assertIn(("Article XVII", "resolved", "a17"), refs)
        self.assertIn(("Article Two", "unresolved", None), refs)
        self.assertIn(("Section 1", "unresolved", None), refs)  # "ARTICLE ONE" is not cited as "Section 1"

    # L2: a bare Roman "Section VIII" is article VIII
    def test_l2_roman_section_links_to_its_article(self) -> None:
        xml = _doc(
            _article("a8", "SECTION VIII TERMINATION", _section("s81", "8.1 Termination", "x"))
            + _article(
                "a9",
                "ARTICLE IX MISC",
                _section(
                    "s91",
                    "9.1 Terms",
                    "Under Section VIII, the parties act. See Sections VIII and IX of this Agreement. "
                    "Section VIII of that certain report applies. Section VIII B and IX.D of the Order apply. Section VIII(1) applies. "
                    "Under this Section IX. Also Section IX applies. Section VIII of the Seller applies.",
                ),
            )
        )
        refs = [(x["text"], x["status"], x.get("target")) for x in build_reader_index(xml)["xrefs"]]
        self.assertEqual(
            refs,
            [
                ("Section VIII", "resolved", "a8"),
                ("Sections VIII", "resolved", "a8"),
                ("IX", "unresolved", None),  # "ARTICLE IX" is cited "Section IX" only as "this Section IX"
                ("Section VIII", "unresolved", None),
                ("Section VIII", "unresolved", None),
                ("Section VIII", "unresolved", None),
                ("Section IX", "resolved", "a9"),
                ("Section IX", "unresolved", None),
                ("Section VIII", "unresolved", None),  # R3: "of the Seller" names a party, not a document
            ],
        )

    def test_l2_needs_a_roman_heading(self) -> None:
        xml = _doc(
            _article("a4", "4. Escrow Accounts", _section("s41", "4.1 Escrow", "x"))
            + _article("a23", "23. Other", _section("s231", "23.1 Refs", "Claims under this Section IV (Escrow) are paid."))
        )
        x = _xrefs(build_reader_index(xml))
        self.assertNotIn("target", x["Section IV"])

    # L6: "K&S", "D&O", "U.S." are names, not single-letter items
    def test_l6_ampersand_and_dotted_initials_are_not_items(self) -> None:
        refs = self._all("Under the Schedule D&amp;O policy, Section U.S. Holder rules and Schedules K&amp;S Draft, see Exhibit A, B and C and Article IV.")
        self.assertEqual([t for t, _, _ in refs], ["Exhibit A", "B", "C", "Article IV"])
        refs = self._all("The Closing is set forth in Article I.A. above and Section I.E.1 below.")
        self.assertEqual([t for t, _, _ in refs], ["Article I", "Section I"])

    # L4: a dotted article reference falls back to its longest existing section prefix
    def test_l4_dotted_article_falls_back_to_section(self) -> None:
        refs = self._all("Under Article 2.1.11 the parties act. Article 2.1.4 of the Plan applies. See Article 5.12. See Article 7.1.1.")
        self.assertEqual(
            refs,
            [
                ("Article 2.1.11", "section_fallback", "s2.1"),
                ("Article 2.1.4", "external", None),
                ("Article 5.12", "unresolved", None),
                ("Article 7.1.1", "unresolved", None),
            ],
        )

    # L7: "of Seller's Disclosure Statement" is the disclosure schedule
    def test_l7_disclosure_statement_is_the_disclosure_schedule(self) -> None:
        refs = self._all("Except as set forth in Section 3.13(b) of the Matrix Disclosure Statement or Section 2.1 of Seller’s Disclosure Statement.")
        self.assertEqual([s for _, s, _ in refs], ["disclosure_schedule", "disclosure_schedule"])
        refs = self._all("Except as set forth in Section 3.13(b) of the Schedule of Exceptions. A reference in this Article II to such Schedule of Exceptions counts.")
        self.assertEqual([(s, g) for _, s, g in refs], [("disclosure_schedule", None), ("resolved", "a2")])

    def test_l1_reaches_numbers_no_section_can_have(self) -> None:
        refs = self._all("Under Section 1.368-2(g) of the Treasury regulations and Section 2510.3-101 of the Department of Labor regulations.")
        self.assertEqual([s for _, s, _ in refs], ["external", "external"])

    # L9: a clause qualified by a section finds its one matching descendant
    def test_l9_qualified_clause_finds_the_one_descendant(self) -> None:
        main = _article(
            "a6",
            "ARTICLE VI COVENANTS",
            _section("s64", "Section 6.4 Conduct", "(a) The Company shall not:", "(i) amend;", "(ii) sell;", "(iii) borrow.", "(b) Parent shall not:", "(A) merge.")
            + _section("s65", "Section 6.5 Other", "(a) Each party:", "(i) acts;", "(b) The other:", "(i) waits."),
        )
        refs = self._all(
            "Breach of clauses (i) through (iii) of Section 6.4. Also clause (A) of Section 6.4. Also clause (i) of Section 6.5. "
            "Also clause (x) of Section 6.4.",
            main=main,
        )
        got = [(t, s, g) for t, s, g in refs if not t.startswith("Section")]
        self.assertEqual(
            got,
            [
                ("clauses (i)", "resolved", "s6.4(a)(i)"),
                ("(iii)", "resolved", "s6.4(a)(iii)"),
                ("clause (A)", "resolved", "s6.4(b)(A)"),
                ("clause (i)", "section_fallback", "s6.5"),  # two (i) under 6.5: no guess
                ("clause (x)", "section_fallback", "s6.4"),
            ],
        )
        refs = self._all(
            "Except as set forth in clauses (ii) through (iii) of Section 6.4.(a), nothing applies.",
            main=main,
        )
        self.assertEqual([g for t, _, g in refs if t.startswith("clauses")], ["s6.4"])
        inline = main.replace("<text>(A) merge.</text>", "<text>(c) The remedy for (i) claims, (ii) consents or (iii) targets is the escrow, waiving the foregoing clauses (i)-(iii) in this Section 6.4.</text>")
        p = build_reader_index(_doc(inline))
        self.assertEqual([x.get("target") for x in p["xrefs"] if x["text"].startswith("clauses")], ["s6.4"])

    # L8: "Sections 7.16-17" is a range of two sections, not a regulation number
    def test_l8_short_hyphen_range_links_both_ends(self) -> None:
        main = _article(
            "a7", "ARTICLE VII COVENANTS", _section("s716", "Section 7.16 Taxes", "x") + _section("s717", "Section 7.17 Other", "y")
        )
        refs = self._all(
            "Sections 7.16-17 (Taxes) apply. Treasury Regulations Section 7.16-17 applies. Section 1.368-2(g) of the Treasury "
            "regulations applies. See Schedule 1.1-2 and Section 54.4980B-9 and Section 7.16-07.",
            main=main,
        )
        self.assertEqual(refs[:2], [("Sections 7.16", "resolved", "s7.16"), ("17", "resolved", "s7.17")])
        items = self._all("The Persons set forth in Section 1.1-5 of the Disclosure Schedule and Section 4.1-2 to the Disclosure Letter.", main=main)
        self.assertEqual([t for t, _, _ in items], ["Section 1.1-5", "Section 4.1-2"])  # one schedule item each
        texts = [t for t, _, _ in refs]
        for t in ("Section 7.16-17", "Section 1.368-2(g)", "Schedule 1.1-2", "Section 54.4980B-9", "Section 7.16-07"):
            self.assertIn(t, texts)


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
    UK = _article("c9", "9 WARRANTIES", _section("s92", "9.2 Limits", "x, subject to clause 9.1."))

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


class NearestParagraphFallbackTests(unittest.TestCase):
    """D1: "Section 6.4(e)(ii)" with no node 6.4(e)(ii) links paragraph 6.4(e), not Section 6.4."""

    def _all(self, sentence: str, *paras: str) -> list[tuple[str, str, str | None]]:
        target = _section("s64", "Section 6.4 Filings", *paras)
        cite = _section("s71", "Section 7.1 Conditions", sentence)
        p = build_reader_index(_doc(_article("a6", "ARTICLE VI COVENANTS", target) + _article("a7", "ARTICLE VII CONDITIONS", cite)))
        return [(x["text"], x["status"], x.get("target")) for x in p["xrefs"] if x["at"]["s"] == "s71"]

    PARAS = ("(a) Parent shall file.", "(b) The Company shall file.", "(e) Each party shall (i) consult and (ii) cooperate.")

    def test_two_levels_land_on_the_paragraph(self) -> None:
        got = self._all("The obligations under Section 6.4(e)(ii) survive.", *self.PARAS)
        self.assertEqual(got, [("Section 6.4(e)(ii)", "section_fallback", "s6.4(e)")])

    def test_three_levels_land_on_the_second(self) -> None:
        paras = ("(a) Parent shall file.", "(e) Each party shall:", "(i) consult on (A) timing and (B) content;", "(ii) cooperate.")
        got = self._all("The obligations under Section 6.4(e)(i)(B) survive.", *paras)
        self.assertEqual(got, [("Section 6.4(e)(i)(B)", "section_fallback", "s6.4(e)(i)")])

    def test_a_paragraph_printed_twice_is_not_used(self) -> None:
        paras = ("(a) Parent shall file.", "(e) Each party shall consult.", "(e) Each party shall (i) cooperate.")
        got = self._all("The obligations under Section 6.4(e)(i) survive.", *paras)
        self.assertEqual(got, [("Section 6.4", "section_fallback", "s6.4")])

    def test_the_sentences_own_list_marker_lands_on_the_paragraph(self) -> None:
        paras = ("(a) Parent shall file.", "(b) The Company shall file.", "(c) Seller grants Buyer a license to use the Marks.")
        sentence = "The license granted in Section 6.4(c) (i) shall not be terminated and (ii) is freely assignable."
        got = self._all(sentence, *paras)
        self.assertEqual([(t, s, g) for t, s, g in got if t.startswith("Section")], [("Section 6.4(c) (i)", "section_fallback", "s6.4(c)")])

    def test_a_lettered_group_is_not_used(self) -> None:
        paras = ("A. Filings.", "(a) Parent shall file.", "B. Cooperation.", "(a) Each party shall cooperate.")
        got = self._all("The obligations under Section 6.4(A)(c) survive.", *paras)
        self.assertNotIn("s6.4(A)", [g for _, _, g in got])

    def test_references_that_resolve_today_are_unchanged(self) -> None:
        got = self._all("Under Section 6.4(e) and clause (ii) of Section 6.4(e), and Section 6.4.", *self.PARAS)
        self.assertEqual(
            got,
            [
                ("Section 6.4(e)", "resolved", "s6.4(e)"),
                ("clause (ii)", "section_fallback", "s6.4(e)"),  # the qualified path (L9 ancestor), as before D1
                ("Section 6.4(e)", "resolved", "s6.4(e)"),
                ("Section 6.4", "resolved", "s6.4"),
            ],
        )


class AttachmentParentTests(unittest.TestCase):
    """X1: "Exhibit L-1" with no node L-1 links the one heading "Exhibit L" that holds the forms, when its title fits."""

    def _all(self, sentence: str, back: str) -> list[tuple[str, str, str | None]]:
        p = build_reader_index(_doc(MAIN + _article("a3", "ARTICLE III COVENANTS", _section("s31", "Section 3.1 Legends", sentence)), back=back))
        return [(x["text"], x["status"], x.get("target")) for x in p["xrefs"] if x["at"]["s"] == "s31"]

    def test_forms_under_one_parent_heading_link_the_parent(self) -> None:
        sentence = "The recipient delivers a stockholder representation letter substantially in the form attached hereto as Exhibit L-1 or L-2."
        got = self._all(sentence, _back("Exhibit L", "Form of Stockholder Representation Letter", "Exhibit M", "Form of Termination Agreement"))
        self.assertEqual(got, [("Exhibit L-1", "resolved", "x-exhibit-l"), ("L-2", "resolved", "x-exhibit-l")])

    def test_title_on_the_heading_line_counts(self) -> None:
        sentence = "A counterpart to the offer letters in the form attached hereto as Exhibit C-1 through C-2."
        got = self._all(sentence, _back("Exhibit C – Executive Offer Letters", "Exhibit D – Amended and Restated Sublease"))
        self.assertEqual(got, [("Exhibit C-1", "resolved", "x-exhibit-c"), ("C-2", "resolved", "x-exhibit-c")])

    def test_forms_filed_under_other_numbers_stay_unlocated(self) -> None:
        # the printed L-1 and L-2 were filed as K-1 and K-2; Exhibit L is another document
        sentence = "Transferor and GPI enter into a sublease or lease assignment in the form of one of Exhibits L-1 and L-2, as applicable."
        back = _back("Exhibit K-1", "Form of Sublease", "Exhibit K-2", "Form of Lease Assignment", "Exhibit L", "TAX-EXEMPT BOND AGREEMENT")
        self.assertEqual(self._all(sentence, back), [("Exhibits L-1", "unlocated_attachment", None), ("L-2", "unlocated_attachment", None)])

    def test_parent_of_another_subject_stays_unlocated(self) -> None:
        sentence = "“Net Working Capital” has the meaning specified in Exhibit C-1."
        self.assertEqual(self._all(sentence, _back("EXHIBIT C – AMENDMENT TO SITE LICENSE AGREEMENT")), [("Exhibit C-1", "unlocated_attachment", None)])

    def test_a_shared_drafting_word_is_not_enough(self) -> None:
        # "Assets" and "Subject" are in both, but they name no document: A-1 lists mineral interests
        sentence = "Nothing will reduce the use of the Assets subject thereto below the Net Revenue Interest shown in Exhibit A-1."
        got = self._all(sentence, _back("EXHIBIT A", "EXCLUDED ASSETS SUBJECT TO ASSIGNMENT"))
        self.assertEqual(got, [("Exhibit A-1", "unlocated_attachment", None)])

    def test_review_inputs_stay_unlocated(self) -> None:
        cases = [
            # a generic document noun alone ("certificate")
            ("The parties shall file the Certificate of Merger in the form of Exhibit A-1.",
             _back("EXHIBIT A", "FORM OF AMENDED AND RESTATED CERTIFICATE OF INCORPORATION OF THE SURVIVING CORPORATION"), "Exhibit A-1"),
            # a party name ("Acme" of "Acme Holdings")
            ("Acme Holdings shall deliver the Acme Voting Agreement in the form of Exhibit B-1.",
             _back("EXHIBIT B", "FORM OF ACME HOLDINGS ESCROW AGREEMENT"), "Exhibit B-1"),
            # drafting words ("Effective Time")
            ("At the Effective Time, the director resignations in the form of Exhibit C-1 are effective.",
             _back("EXHIBIT C", "FORM OF EFFECTIVE TIME OFFICER CERTIFICATE"), "Exhibit C-1"),
            # the line after the heading is body text, not a title
            ("Each Key Employee shall sign an employment offer in the form of Exhibit D-1.",
             _back("EXHIBIT D", "This Escrow Agreement for each Key Employee is made as follows."), "Exhibit D-1"),
        ]
        for sentence, back, label in cases:
            with self.subTest(label=label):
                self.assertEqual(self._all(sentence, back), [(label, "unlocated_attachment", None)])

    def test_own_node_wins_and_a_filed_sibling_blocks_the_parent(self) -> None:
        sentence = "The officer certificates attached as Exhibits C-1 and C-2 hereto."
        back = _back("EXHIBIT C", "FORM OF TAX CERTIFICATES", "EXHIBIT C-1", "Officer certificate of the Company.")
        self.assertEqual(self._all(sentence, back), [("Exhibits C-1", "resolved", "x-exhibit-c-1"), ("C-2", "unlocated_attachment", None)])



def _nodes(p: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    stack = list(p["outline"])
    while stack:
        n = stack.pop()
        out[n["id"]] = n
        stack.extend(n.get("children") or [])
    return out


class RunOnLetterTests(unittest.TestCase):
    """R3: paragraph letters that run on from section to section ("(d)" opens Section 3.2) are numbered by position."""

    REPS = (
        _section("s31", "Section 3.1 Organization", "(a) The Company is duly organized.", "(b) Each Subsidiary is in good standing.", "(c) No violation."),
        _section("s32", "Section 3.2 Capitalization", "(d) Capital Stock. The authorized stock is 100 shares.", "(e) Awards. Each award:", "(i) vests;", "(ii) settles.", "(f) Voting Debt. None."),
        _section("s33", "Section 3.3 Authority", "(g) Authority. The Company has power.", "(h) Board Approval. The board approved."),
        _section("s34", "Section 3.4 Consents", "(i) Governmental Consents. None are needed.", "(j) Third Party Consents. None are needed."),
        _section("s35", "Section 3.5 Litigation", "(k) There is no action.", "(l) There is no order."),
    )
    CITES = "The representations in Section 3.2(a), clause (ii) of Section 3.2(b), Section 3.4(b) and Section 3.2(f) survive."

    def _build(self, *sections: str, cites: str = CITES) -> dict[str, Any]:
        reps = _article("a3", "ARTICLE III REPRESENTATIONS", "".join(sections))
        cond = _article("a7", "ARTICLE VII CONDITIONS", _section("s71", "Section 7.1 Conditions", cites))
        return build_reader_index(_doc(reps + cond))

    def _links(self, p: dict[str, Any]) -> list[tuple[str, str, Any]]:
        nodes = _nodes(p)
        out = []
        for x in p["xrefs"]:
            if x["at"]["s"] == "s71":
                n = nodes.get(x.get("target"))
                out.append((x["text"], x["status"], (n["anchor"]["s"], n["anchor"].get("t")) if n else None))
        return out

    def test_run_on_letters_follow_position(self) -> None:
        p = self._build(*self.REPS)
        self.assertEqual(
            self._links(p),
            [
                ("Section 3.2(a)", "resolved", ("s32", 0)),  # printed "(d)"
                ("clause (ii)", "resolved", ("s32", 3)),  # the item "(ii)" under printed "(e)"
                ("Section 3.2(b)", "resolved", ("s32", 1)),
                ("Section 3.4(b)", "resolved", ("s34", 1)),  # printed "(j)"; "(i)" opens the section as a letter
                ("Section 3.2", "section_fallback", ("s32", None)),  # printed "(f)" is position (c); no (f) by position
            ],
        )
        nodes = _nodes(p)
        self.assertEqual([c["label"] for c in nodes["s3.4"]["children"]], ["3.4(a)", "3.4(b)"])
        self.assertTrue(nodes["s3.2(a)"]["inferred"] and nodes["s3.2(b)(ii)"]["inferred"])
        self.assertTrue(nodes["s3.2(a)"]["title"].startswith("(d) Capital Stock"))  # the printed text stays as filed
        self.assertNotIn("inferred", nodes["s3.1(a)"])  # the first section prints its own letters

    def test_two_run_on_sections_are_not_enough(self) -> None:
        p = self._build(*self.REPS[:3])
        nodes = _nodes(p)
        self.assertIn("s3.2(d)", nodes)
        self.assertNotIn("inferred", nodes["s3.2(d)"])
        self.assertIn(("Section 3.2(f)", "resolved", ("s32", 4)), self._links(p))

    def test_letters_that_restart_are_unchanged(self) -> None:
        secs = [
            _section(f"s3{i}", f"Section 3.{i} Topic", "(a) First.", "(b) Second.") for i in range(1, 6)
        ]
        p = self._build(*secs, cites="Section 3.2(b) and Section 3.4(a) apply.")
        self.assertEqual(self._links(p), [("Section 3.2(b)", "resolved", ("s32", 1)), ("Section 3.4(a)", "resolved", ("s34", 0))])
        self.assertFalse(any(n.get("inferred") for n in _nodes(p).values()))

    def test_an_inline_list_in_the_lead_in_keeps_the_printed_letters(self) -> None:
        lead = _section("s32", "Section 3.2 Capitalization", "Except as disclosed: (a) no shares are issued; (b) no options exist;", "(c) no debt exists.", "(d) no rights exist.")
        more = (
            _section("s36", "Section 3.6 Taxes", "(m) Returns are filed.", "(n) Taxes are paid."),
            _section("s37", "Section 3.7 Contracts", "(o) Each Contract is valid.", "(p) No breach exists."),
        )
        secs = (self.REPS[0], lead) + self.REPS[2:] + more
        nodes = _nodes(self._build(*secs))
        self.assertIn("s3.4(a)", nodes)  # the agreement qualifies
        self.assertIn("s3.2(c)", nodes)
        self.assertNotIn("inferred", nodes["s3.2(c)"])

    def test_a_stray_letter_keeps_the_printed_letters(self) -> None:
        stray = _section("s33", "Section 3.3 Authority", "(g) Authority. The Company has power.", "(a) The board approved.")
        secs = self.REPS[:2] + (stray,) + self.REPS[3:] + (
            _section("s36", "Section 3.6 Taxes", "(m) Returns are filed.", "(n) Taxes are paid."),
        )
        nodes = _nodes(self._build(*secs))
        self.assertIn("s3.3(g)", nodes)
        self.assertIn("s3.2(a)", nodes)


class DoubledLetterTests(unittest.TestCase):
    """D2: in a run-on section, "(ii)" after a "(hh)" paragraph is the next doubled letter; a roman list stays a list."""

    LETTERS = [f"({chr(c)}) Item." for c in range(97, 123)] + ["(aa) Item.", "(bb) Item."]
    RUN = (
        _section("s31", "Section 3.1 Organization", *LETTERS),
        _section("s32", "Section 3.2 Authority", "(cc) Power.", "(dd) Approval."),
        _section("s33", "Section 3.3 Consents", "(ee) None.", "(ff) None."),
    )
    RESTART = tuple(_section(f"s3{i}", f"Section 3.{i} Topic", "(a) First.", "(b) Second.") for i in range(1, 4))

    def _labels(self, *paras: str, run_on: bool = True) -> list[str]:
        secs = "".join(self.RUN if run_on else self.RESTART) + _section("s34", "Section 3.4 Taxes", "(gg) Filings.", *paras)
        p = build_reader_index(_doc(_article("a3", "ARTICLE III REPRESENTATIONS", secs)))
        return [n["label"] for n in sorted(_nodes(p).values(), key=lambda n: n["anchor"].get("t", -1)) if n["kind"] == "sub" and n["anchor"]["s"] == "s34"]

    def test_doubled_letters_in_a_run_on_section_are_siblings(self) -> None:
        self.assertEqual(self._labels("(hh) Payments.", "(ii) Liens.", "(jj) Audits."), ["3.4(a)", "3.4(b)", "3.4(c)", "3.4(d)"])

    def test_a_roman_list_under_hh_stays_nested(self) -> None:
        got = self._labels("(hh) Payments:", "(i) federal;", "(ii) state.", "(ii) Liens.", "(jj) Audits.")
        self.assertEqual(got, ["3.4(a)", "3.4(b)", "3.4(b)(i)", "3.4(b)(ii)", "3.4(c)", "3.4(d)"])

    def test_doubled_letters_elsewhere_are_unchanged(self) -> None:
        # a definitions list (gg), (hh), (ii) keeps the old reading, so "clause (ii)" items do not become ambiguous
        self.assertEqual(self._labels("(hh) Payments.", "(ii) Liens.", "(jj) Audits.", run_on=False), ["3.4(gg)", "3.4(hh)", "3.4(hh)(ii)", "3.4(jj)"])


if __name__ == "__main__":
    unittest.main()
