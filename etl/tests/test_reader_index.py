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

    def test_local_clause_resolves_within_current_section(self) -> None:
        xml = _doc(_section("s1", "Section 7.2 Conditions", "(a) first.", "(b) second, subject to clause (a) above."))
        x = _xrefs(build_reader_index(xml))["clause (a)"]
        self.assertEqual((x["status"], x["target"]), ("resolved", "s7.2(a)"))


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
