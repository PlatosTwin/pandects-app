# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
from datetime import date
import unittest
import xml.etree.ElementTree as ET

import pandas as pd

from etl.domain import f_xml
from etl.domain.g_sections import extract_sections_from_xml, section_plain_text_from_xml
from etl.domain.sig_tail_split import find_signature_tail, split_signature_tail

CLAUSE = (
    "This Agreement may be executed in counterparts, each of which shall be deemed an original, "
    "and all of which together shall constitute one instrument."
)
TESTIMONIUM = (
    "IN WITNESS WHEREOF, the parties hereto have caused this Agreement to be executed "
    "as of the date first written above."
)
BLOCKS = "ACME CORP.\n\nBy: /s/ Jane Doe\n\nName: Jane Doe\n\nTitle: Chief Executive Officer"


def _mixed_page(*middle: str) -> str:
    return "\n\n".join(["<section>Section 9.5 Counterparts.</section>", CLAUSE, *middle])


def _split(text: str, sig_orders: list[int] | None = None):
    return split_signature_tail(["<article>ARTICLE IX</article>\n\nEarlier text.", text], [1, 2], sig_orders or [])


class FindSignatureTailTests(unittest.TestCase):
    def test_splits_at_the_testimonium(self) -> None:
        result = _split(_mixed_page(TESTIMONIUM, BLOCKS, "<page>45</page>"))

        self.assertEqual(result.trigger, "testimonium")
        self.assertEqual(result.moved_index, 1)
        assert result.moved_text is not None
        self.assertTrue(result.moved_text.startswith("IN WITNESS WHEREOF"))
        self.assertTrue(result.moved_text.endswith("Title: Chief Executive Officer"))
        # The clause stays; the page number marker stays with the page.
        self.assertTrue(result.body_page_texts[1].startswith("<section>Section 9.5 Counterparts.</section>"))
        self.assertIn(CLAUSE, result.body_page_texts[1])
        self.assertTrue(result.body_page_texts[1].endswith("<page>45</page>"))
        self.assertNotIn("WITNESS", result.body_page_texts[1])
        self.assertNotIn("<page>", result.moved_text)
        # Earlier pages are untouched.
        self.assertEqual(result.body_page_texts[0], "<article>ARTICLE IX</article>\n\nEarlier text.")

    def test_page_marker_glued_into_a_moved_line_stays_in_the_body(self) -> None:
        result = _split(_mixed_page(TESTIMONIUM, "ACME CORP. <page>45</page>", "By: /s/ Jane Doe", "Name: Jane Doe"))

        assert result.moved_text is not None
        self.assertNotIn("<page>", result.moved_text)
        self.assertIn("ACME CORP.", result.moved_text)
        self.assertTrue(result.body_page_texts[1].endswith("<page>45</page>"))

    def test_letter_spaced_and_other_testimonium_forms(self) -> None:
        for opening in (
            "I N W I T N E S S W H E R E O F, the parties have signed this Agreement.",
            "In Witness Whereof, the parties have signed this Agreement.",
            "The parties hereto, intending to be legally bound hereby, have duly executed this Agreement.",
            "EXECUTED as of the date first above written.",
            "Executed as a Deed by the parties on the date of this Agreement.",
            "IN TESTIMONY WHEREOF, the parties have signed this Agreement.",
            "AS WITNESS this Agreement has been signed on the date first written above.",
            "Sterling and Klamath have caused this Agreement to be executed by their officers as of the date above.",
        ):
            with self.subTest(opening=opening):
                result = _split(_mixed_page(opening, BLOCKS))
                self.assertEqual(result.trigger, "testimonium")
                assert result.moved_text is not None
                self.assertTrue(result.moved_text.startswith(opening))

    def test_signature_page_follows_line_stays_in_the_body(self) -> None:
        result = _split(_mixed_page("[Signature page follows]", TESTIMONIUM, BLOCKS))

        self.assertTrue(result.body_page_texts[1].endswith("[Signature page follows]"))
        assert result.moved_text is not None
        self.assertTrue(result.moved_text.startswith("IN WITNESS WHEREOF"))

    def test_testimonium_alone_moves_when_the_blocks_are_on_the_next_page(self) -> None:
        result = _split(_mixed_page("[Remainder of page intentionally left blank]", TESTIMONIUM), sig_orders=[3])

        self.assertEqual(result.moved_text, TESTIMONIUM)
        self.assertTrue(result.body_page_texts[1].endswith("[Remainder of page intentionally left blank]"))

    def test_splits_at_the_first_signature_block_without_a_testimonium(self) -> None:
        result = _split(_mixed_page("[Signature page follows]", "PARENT:", BLOCKS, "BUYER LLC", "By: /s/ John Roe", "Name: John Roe"))

        self.assertEqual(result.trigger, "signature_block")
        assert result.moved_text is not None
        self.assertTrue(result.moved_text.startswith("PARENT:\n\nACME CORP."))
        self.assertTrue(result.body_page_texts[1].endswith("[Signature page follows]"))

    def test_a_bare_by_line_right_after_clause_text_is_not_a_block(self) -> None:
        text = _mixed_page("By: the Company, acting through its board, as provided above; Name: none.")
        self.assertIsNone(find_signature_tail(text))

    def test_testimonium_at_the_top_of_the_page_moves_the_whole_page(self) -> None:
        result = _split("\n\n".join([TESTIMONIUM, BLOCKS, "<page>46</page>"]))

        self.assertEqual(result.body_page_texts[1], "<page>46</page>")
        assert result.moved_text is not None
        self.assertTrue(result.moved_text.startswith("IN WITNESS WHEREOF"))

    def test_final_section_heading_stays_in_the_body(self) -> None:
        text = "\n\n".join([CLAUSE, "<section>Section 9.6 Entire Agreement.</section>", "This Agreement is the entire agreement.", TESTIMONIUM, BLOCKS])
        result = _split(text)

        self.assertTrue(result.body_page_texts[1].endswith("This Agreement is the entire agreement."))
        self.assertIn("<section>Section 9.6 Entire Agreement.</section>", result.body_page_texts[1])

    def test_quoted_testimonium_in_a_clause_does_not_split(self) -> None:
        text = _mixed_page(
            "The joinder shall read: “IN WITNESS WHEREOF, the undersigned has executed this joinder.”",
            "“IN WITNESS WHEREOF, the undersigned has executed this instrument.”",
        )
        self.assertIsNone(find_signature_tail(text))
        self.assertIsNone(_split(text).moved_text)

    def test_testimonium_of_a_form_printed_in_the_body_does_not_split(self) -> None:
        text = _mixed_page("IN WITNESS WHEREOF, the undersigned has executed this Joinder as of the date below.", BLOCKS)
        self.assertIsNone(find_signature_tail(text))

    def test_signature_block_followed_by_a_heading_does_not_split(self) -> None:
        # A joinder signed in the middle of the body, with more agreement after it.
        text = "\n\n".join([CLAUSE, TESTIMONIUM, BLOCKS, "<section>Section 9.6 Joinder.</section>", "More agreement text."])
        self.assertIsNone(find_signature_tail(text))

    def test_prose_between_testimonium_and_blocks_does_not_split(self) -> None:
        prose = (
            "The Company shall indemnify each party for all losses arising out of any breach of this Agreement, "
            "and the Company hereby agrees that such indemnity will survive the Closing for a period of three years "
            "after the Closing Date and will not be subject to any cap or basket of any kind whatsoever under this Agreement."
        )
        self.assertIsNone(find_signature_tail(_mixed_page(TESTIMONIUM, prose, BLOCKS)))
        self.assertIsNone(find_signature_tail(_mixed_page(TESTIMONIUM, "(a) the Company shall deliver the closing certificate to Parent.", BLOCKS)))

    def test_consent_signed_under_the_blocks_moves_with_them(self) -> None:
        consent = (
            "The undersigned hereby consents to the assignment of all of its right, title and interest in the agreement "
            "described above to the purchaser named above and hereby agrees that it shall be bound by the terms thereof "
            "as if it were an original party to that agreement."
        )
        result = _split(_mixed_page(TESTIMONIUM, BLOCKS, consent, "By: /s/ Ann Poe"))
        assert result.moved_text is not None
        self.assertIn(consent, result.moved_text)

    def test_signature_page_before_the_last_body_page_does_not_split(self) -> None:
        self.assertIsNone(_split(_mixed_page(TESTIMONIUM, BLOCKS), sig_orders=[1]).moved_text)

    def test_testimonium_glued_after_the_last_sentence_splits_inside_the_line(self) -> None:
        result = _split(_mixed_page(CLAUSE + " IN WITNESS WHEREOF, the parties have signed this Agreement.", BLOCKS))

        self.assertEqual(result.trigger, "testimonium_glued")
        self.assertTrue(result.body_page_texts[1].endswith(CLAUSE))
        assert result.moved_text is not None
        self.assertTrue(result.moved_text.startswith("IN WITNESS WHEREOF, the parties"))

    def test_execution_words_inside_clause_text_do_not_block_the_split(self) -> None:
        clause = "This Agreement may be executed by facsimile, and the parties have executed it in counterparts."
        result = _split("\n\n".join(["<section>Section 9.5 Counterparts.</section>", clause, TESTIMONIUM, BLOCKS]))

        self.assertEqual(result.trigger, "testimonium")
        self.assertTrue(result.body_page_texts[1].endswith(clause))

    def test_testimonium_inside_a_sentence_does_not_split(self) -> None:
        # Neither the phrase nor the blocks after it move.
        text = _mixed_page("The parties shall sign the form beginning in witness whereof the parties agree.", BLOCKS)
        self.assertIsNone(_split(text).moved_text)

    def test_pages_without_orders_do_not_split(self) -> None:
        self.assertIsNone(split_signature_tail(["x", _mixed_page(TESTIMONIUM, BLOCKS)], [], []).moved_text)

    def test_page_without_signatures_is_unchanged(self) -> None:
        text = _mixed_page("<page>45</page>")
        result = _split(text)
        self.assertIsNone(result.moved_text)
        self.assertEqual(result.body_page_texts[1], text)


def _row(page_uuid: str, order: int, label: str, text: str, raw: str = "") -> dict[str, object]:
    return {
        "agreement_uuid": "11111111-1111-1111-1111-111111111111",
        "page_uuid": page_uuid,
        "page_order": order,
        "raw_page_content": raw,
        "source_page_type": label,
        "gold_label": None,
        "tagged_output": text,
        "url": "https://example.com/agreement",
        "filing_date": date(2024, 1, 1),
        "source_is_txt": False,
        "source_is_html": True,
    }


def _words(root: ET.Element) -> list[str]:
    return sorted(
        word
        for container in root
        if container.tag != "metadata"
        for el in container.iter()
        if el.tag not in ("page", "pageUUID")
        for word in (el.text or "").split()
    )


class GenerateXMLSignatureTailTests(unittest.TestCase):
    def _generate(self, rows: list[dict[str, object]]) -> ET.Element:
        generated, failures = f_xml.generate_xml(pd.DataFrame(rows))
        self.assertEqual(failures, [])
        return ET.fromstring(generated[0].xml)

    def _rows(self, last_page: str, *extra: dict[str, object]) -> list[dict[str, object]]:
        return [
            _row("p1", 1, "body", "<article>ARTICLE IX MISCELLANEOUS</article>\n\n<section>Section 9.4 Notices.</section>\n\nAll notices shall be in writing."),
            _row("p2", 2, "body", last_page),
            *extra,
        ]

    def test_mixed_last_body_page_serves_the_signature_block_in_sig_pages(self) -> None:
        last = _mixed_page("[Signature page follows]", TESTIMONIUM, BLOCKS, "<page>2</page>")
        root = self._generate(self._rows(last, _row("p3", 3, "sig", "BUYER LLC\n\nBy: /s/ John Roe\n\nName: John Roe")))
        unsplit = self._generate(self._rows(last.replace("IN WITNESS", "AT WITNESS"), _row("p3", 3, "sig", "BUYER LLC\n\nBy: /s/ John Roe\n\nName: John Roe")))

        sections = extract_sections_from_xml(ET.tostring(root, encoding="unicode"))
        last_section = section_plain_text_from_xml(sections[-1]["xml_content"])
        self.assertEqual(sections[-1]["section_title"], "Section 9.5 Counterparts.")
        self.assertTrue(last_section.endswith("one instrument. [Signature page follows]"))
        self.assertNotIn("WITNESS", last_section)
        self.assertEqual(len(sections), len(extract_sections_from_xml(ET.tostring(unsplit, encoding="unicode"))))

        sig = root.find("sigPages")
        assert sig is not None
        children = [(el.tag, (el.text or "").strip()) for el in sig]
        self.assertEqual(children[0], ("text", TESTIMONIUM))
        # The moved text is served under its own page, ahead of the signature pages.
        self.assertEqual([t for tag, t in children if tag == "pageUUID"], ["p2", "p3"])
        self.assertIn(("text", "By: /s/ Jane Doe"), children)
        # The page number stays in the body.
        body_pages = [(el.text or "").strip() for el in root.iter("page")]
        self.assertEqual(body_pages, ["2"])
        # Nothing is lost: the same words are served, just in another container.
        unsplit_words = [w.replace("AT", "IN") if w == "AT" else w for w in _words(unsplit)]
        self.assertEqual(sorted(unsplit_words), _words(root))

    def test_creates_sig_pages_when_the_agreement_has_no_signature_page(self) -> None:
        root = self._generate(self._rows(_mixed_page(TESTIMONIUM, BLOCKS)))
        sig = root.find("sigPages")
        assert sig is not None
        self.assertEqual([(el.text or "").strip() for el in sig.findall("pageUUID")], ["p2"])
        self.assertEqual(list(root).index(sig), list(root).index(root.find("body")) + 1)

    def test_no_split_leaves_the_xml_unchanged(self) -> None:
        rows = self._rows(_mixed_page("<page>2</page>"), _row("p3", 3, "sig", TESTIMONIUM + "\n\n" + BLOCKS))
        root = self._generate(rows)
        sig = root.find("sigPages")
        assert sig is not None
        self.assertEqual([(el.text or "").strip() for el in sig.findall("pageUUID")], ["p3"])


if __name__ == "__main__":
    unittest.main()
