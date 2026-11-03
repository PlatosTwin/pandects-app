# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
"""Word spaces the page text lost are restored, and hyphen spaces it added are removed,
only where the filed page proves it.

The positive cases are trimmed from served agreements:
* 38a10980 page 34: a notice block whose lines end in <br/> ("Corporation2001 Edmund Drive");
* 7d084685: a whitespace-only <font> between two words ("TaxReturn");
* 0bf2fd35: a defined term and "means" in two paragraphs' worth of block tags;
* f00c52dd page 44: "<b>nti</b><b>-Corruption Matters</b>" read "Anti -Corruption";
* 9e6abc16: "Form S-</font><font>4" read "Form S- 4";
* 00b0aa8c and fa14a726: Symbol bullets and a logo glyph in the private-use area.
"""
import unittest
import xml.etree.ElementTree as ET
from datetime import date

import pandas as pd

from etl.domain import f_xml
from etl.domain.raw_spaces import (
    map_private_use_glyphs,
    restore_page_text,
    restore_raw_spaces,
)

NOTICE_RAW = (
    "<div><p>if to Parent, to:</p><p>Sprint Nextel Corporation<br/>2001 Edmund Drive<br/>Reston, VA 20191<br/>"
    "Telecopier: (703) 433-4846<br/>Attention: General Counsel</p></div>"
)
NOTICE_TEXT = (
    "if to Parent, to:\n\nSprint Nextel Corporation2001 Edmund DriveReston, VA 20191Telecopier: (703) 433-4846"
    "Attention: General Counsel"
)


def restore(text: str, raw: str) -> str:
    return restore_raw_spaces(text, raw).text


class InsertTests(unittest.TestCase):
    def test_br_separators_in_a_notice_block(self) -> None:
        self.assertEqual(
            restore(NOTICE_TEXT, NOTICE_RAW),
            "if to Parent, to:\n\nSprint Nextel Corporation 2001 Edmund Drive Reston, VA 20191 Telecopier: "
            "(703) 433-4846 Attention: General Counsel",
        )

    def test_whitespace_only_span(self) -> None:
        raw = "<p>each Seller shall file each <font>Tax</font><font> </font><font>Return</font> required to be filed by it</p>"
        text = "each Seller shall file each TaxReturn required to be filed by it"
        self.assertEqual(restore(text, raw), "each Seller shall file each Tax Return required to be filed by it")

    def test_block_boundary_after_a_defined_term(self) -> None:
        raw = (
            "<p>Definitions. The following terms have the meanings below.</p>"
            "<p><b>“Affiliate”</b></p><p>of any Person means any other Person that controls it.</p>"
        )
        text = "Definitions. The following terms have the meanings below. “Affiliate”of any Person means any other Person that controls it."
        self.assertIn("“Affiliate” of any Person", restore(text, raw))

    def test_table_cells(self) -> None:
        raw = "<table><tr><td>Name of Shareholder Percentage</td><td>Interest in the Company</td></tr></table>"
        text = "Name of Shareholder PercentageInterest in the Company"
        self.assertEqual(restore(text, raw), "Name of Shareholder Percentage Interest in the Company")

    def test_glued_number_and_name(self) -> None:
        raw = "<p>the “Schedule</p><p>14D-9”), including a description of the Offer to Purchase</p>"
        text = "the “Schedule14D-9”), including a description of the Offer to Purchase"
        self.assertEqual(restore(text, raw), "the “Schedule 14D-9”), including a description of the Offer to Purchase")


    def test_relatively_positioned_block_is_a_block(self) -> None:
        raw = (
            '<div style="position:relative">Attention: General Counsel</div>'
            '<div style="position:relative">Email: legal@example.com and notices@example.com</div>'
        )
        text = "Attention: General CounselEmail: legal@example.com and notices@example.com"
        self.assertEqual(restore(text, raw), "Attention: General Counsel Email: legal@example.com and notices@example.com")


class NoInsertTests(unittest.TestCase):
    def test_section_number_printed_without_a_separator(self) -> None:
        text = "as provided in Section5 of the Disclosure Letter delivered by the Company today"
        for raw in (
            "<p>as provided in Section5 of the Disclosure Letter delivered by the Company today</p>",
            "<p>as provided in <b>Section</b>5 of the Disclosure Letter delivered by the Company today</p>",
            "<p>as provided in <font>Section</font><font>5</font> of the Disclosure Letter delivered by the Company today</p>",
        ):
            self.assertEqual(restore(text, raw), text)

    def test_positioned_boxes_are_not_evidence(self) -> None:
        raw = (
            '<div style="position:absolute;left:10px">as provided in the Purchase Agreement by the Corporat</div>'
            '<div style="position:absolute;left:300px">ion and the Buyer named in it</div>'
        )
        text = "as provided in the Purchase Agreement by the Corporation and the Buyer named in it"
        self.assertEqual(restore(text, raw), text)

    def test_inline_block_boxes_are_not_evidence(self) -> None:
        raw = (
            '<p>as provided in the Purchase Agreement by the <span style="display:inline-block">Corporat</span>'
            '<span style="display:inline-block">ion</span> and the Buyer named in it</p>'
        )
        text = "as provided in the Purchase Agreement by the Corporation and the Buyer named in it"
        self.assertEqual(restore(text, raw), text)

    def test_hidden_element_is_not_evidence(self) -> None:
        raw = '<p>each Seller shall file each Tax<span style="display:none">x</span>Return required to be filed by it</p>'
        text = "each Seller shall file each TaxReturn required to be filed by it"
        self.assertEqual(restore(text, raw), text)

    def test_no_space_after_a_hyphen_at_a_line_break(self) -> None:
        raw = "<p>the Delaware General Corpo-<br/>ration Law, as amended from time to time</p>"
        text = "the Delaware General Corpo-ration Law, as amended from time to time"
        self.assertEqual(restore(text, raw), text)

    def test_no_space_before_closing_punctuation(self) -> None:
        raw = "<p>Sprint Nextel Corporation<br/>, a Kansas corporation, and the other parties hereto</p>"
        text = "Sprint Nextel Corporation, a Kansas corporation, and the other parties hereto"
        self.assertEqual(restore(text, raw), text)

    def test_ambiguous_key_is_left_alone(self) -> None:
        # The same words print twice, once with a break and once without: no unique place.
        raw = "<p>Attention: General Counsel<br/>Email</p><p>Attention: General CounselEmail</p>"
        text = "Attention: General CounselEmail Attention: General CounselEmail"
        self.assertEqual(restore(text, raw), text)

    def test_page_text_that_does_not_match_the_raw(self) -> None:
        raw = "<p>Sprint Nextel Corporation<br/>2001 Edmund Drive, Reston, Virginia</p>"
        text = "Sprint Nextel Corporation;2001 Edmund Drive, Reston, Virginia"
        self.assertEqual(restore(text, raw), text)

    def test_missing_raw(self) -> None:
        self.assertEqual(restore_raw_spaces(NOTICE_TEXT, None).text, NOTICE_TEXT)
        self.assertEqual(restore_raw_spaces(NOTICE_TEXT, "").text, NOTICE_TEXT)

    def test_txt_pages_are_left_alone(self) -> None:
        row = {"raw_page_content": NOTICE_RAW, "source_is_html": False}
        self.assertEqual(restore_page_text(NOTICE_TEXT, row), NOTICE_TEXT)
        row_nan = {"raw_page_content": NOTICE_RAW, "source_is_html": float("nan")}
        self.assertEqual(restore_page_text(NOTICE_TEXT, row_nan), NOTICE_TEXT)

    def test_no_space_next_to_markup(self) -> None:
        raw = "<p><b>Section 5.1 Closing.</b></p><p>The Closing shall take place at the offices of counsel</p>"
        text = "<section>Section 5.1 Closing.</section>The Closing shall take place at the offices of counsel"
        self.assertEqual(restore(text, raw), text)


    def test_hand_wrapped_word_in_a_narrow_cell(self) -> None:
        raw = (
            "<p>Siemens Aktiengesellschaft, a stock corporation organized under the Laws of Germany</p>"
            "<table><tr><td>EP 1234567</td><td>Siemens<br/>Aktiengese<br/>llschaft</td><td>DE</td></tr></table>"
        )
        text = "Siemens Aktiengesellschaft, a stock corporation organized under the Laws of Germany EP 1234567 SiemensAktiengesellschaft DE"
        self.assertEqual(
            restore(text, raw),
            "Siemens Aktiengesellschaft, a stock corporation organized under the Laws of Germany EP 1234567 Siemens Aktiengesellschaft DE",
        )


    def test_glued_words_are_split_even_when_a_shorter_word_spans_them(self) -> None:
        raw = "<p>these terms apply to the<br/>securities issued by the Company under this Agreement</p>"
        text = "these terms apply to thesecurities issued by the Company under this Agreement"
        self.assertEqual(restore(text, raw), "these terms apply to the securities issued by the Company under this Agreement")


class HyphenTests(unittest.TestCase):
    def test_split_hyphenated_title(self) -> None:
        raw = (
            '<p style="text-indent:4%"><b>6.24</b>\xa0\xa0\xa0\xa0<b>A</b><b>nti</b><b>-Corruption Matters</b>. '
            "Since January 1, 2020, none of the Companies has violated any law</p>"
        )
        text = "<section>6.24 Anti -Corruption Matters.</section> Since January 1, 2020, none of the Companies has violated any law"
        self.assertEqual(
            restore(text, raw),
            "<section>6.24 Anti-Corruption Matters.</section> Since January 1, 2020, none of the Companies has violated any law",
        )

    def test_space_after_hyphen(self) -> None:
        raw = "<p>Section 5.4 Preparation of the Form S-</font><font>4 and the Proxy Statement; Company Meeting.</p>"
        text = "Section 5.4 Preparation of the Form S- 4 and the Proxy Statement; Company Meeting."
        self.assertEqual(restore(text, raw), "Section 5.4 Preparation of the Form S-4 and the Proxy Statement; Company Meeting.")

    def test_printed_hyphen_space_is_kept(self) -> None:
        raw = "<p>all reasonable out-of- pocket costs and expenses incurred by the Company in connection</p>"
        text = "all reasonable out-of- pocket costs and expenses incurred by the Company in connection"
        self.assertEqual(restore(text, raw), text)

    def test_hyphen_at_a_line_break_is_kept(self) -> None:
        raw = "<p>Section 9.12 No Third-<br/>Party Beneficiaries; Non-Recourse Against Others.</p>"
        text = "Section 9.12 No Third- Party Beneficiaries; Non-Recourse Against Others."
        self.assertEqual(restore(text, raw), text)

    def test_spaced_dash_is_kept(self) -> None:
        raw = "<p>Section 2.1 - Definitions and Rules of Construction for this Agreement</p>"
        text = "Section 2.1 - Definitions and Rules of Construction for this Agreement"
        self.assertEqual(restore(text, raw), text)


    def test_dash_leaders_are_kept(self) -> None:
        raw = "<p>Target Working Capital ----------- Closing Merger Consideration $56,023</p>"
        text = "Target Working Capital - - - - - - Closing Merger Consideration $56,023"
        self.assertEqual(restore(text, raw), text)


class MarkupTests(unittest.TestCase):
    def test_heading_number_split_by_a_line_break_stays_one_number(self) -> None:
        raw = "<p><b>ARTICLE X<br/>I</b></p><p><b>MISCELLANEOUS</b></p><p>Section 11.1 Notices. All notices hereunder</p>"
        text = "<article>ARTICLE XI MISCELLANEOUS</article>Section 11.1 Notices. All notices hereunder"
        self.assertEqual(restore(text, raw), text)
        raw2 = "<p><b>SECTION 1<br/>1.14. Counterparts.</b> This Agreement may be executed in counterparts</p>"
        text2 = "<section>SECTION 11.14. Counterparts.</section> This Agreement may be executed in counterparts"
        self.assertEqual(restore(text2, raw2), text2)

    def test_glued_words_in_a_heading_are_split(self) -> None:
        raw = "<p><b>ARTICLE III</b></p><p><b>EFFECT ON THE EQUITY INTERESTS</b><br/><b>OF THE CONSTITUENT COMPANIES</b></p>"
        text = "<article>ARTICLE III EFFECT ON THE EQUITY INTERESTSOF THE CONSTITUENT COMPANIES</article>"
        self.assertEqual(
            restore(text, raw), "<article>ARTICLE III EFFECT ON THE EQUITY INTERESTS OF THE CONSTITUENT COMPANIES</article>"
        )

    def test_page_marker_values_are_not_edited(self) -> None:
        raw = "<p>and the Buyer shall pay the Purchase Price at the Closing to the Seller</p><p>- 2<br/>-</p>"
        text = "and the Buyer shall pay the Purchase Price at the Closing to the Seller <page>- 2-</page>"
        self.assertEqual(restore(text, raw), text)


class GlyphTests(unittest.TestCase):
    def test_bullets_map_to_a_bullet(self) -> None:
        self.assertEqual(map_private_use_glyphs(" Items that violate code"), "• Items that violate code")
        self.assertEqual(map_private_use_glyphs("dated as of [], 2016"), "dated as of [•], 2016")
        self.assertEqual(map_private_use_glyphs("Scope of Work  Furnish"), "Scope of Work • Furnish")

    def test_decoration_is_dropped(self) -> None:
        self.assertEqual(
            map_private_use_glyphs("Agreement Contents  King & Wood Mallesons"),
            "Agreement Contents King & Wood Mallesons",
        )
        self.assertEqual(map_private_use_glyphs(" King & Wood Mallesons"), "King & Wood Mallesons")

    def test_other_text_is_unchanged(self) -> None:
        self.assertEqual(map_private_use_glyphs("• Items "), "• Items ")

    def test_glyphs_are_mapped_on_txt_pages_too(self) -> None:
        self.assertEqual(restore_page_text(" Items", {"source_is_html": False}), "• Items")


class GenerateXmlTests(unittest.TestCase):
    def _df(self, tagged: str, raw: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "agreement_uuid": "11111111-1111-1111-1111-111111111111",
                    "page_uuid": "body-page",
                    "page_order": 1,
                    "raw_page_content": raw,
                    "source_page_type": "body",
                    "gold_label": None,
                    "tagged_output": tagged,
                    "url": "https://example.com/agreement",
                    "filing_date": date(2024, 1, 1),
                    "source_is_txt": False,
                    "source_is_html": True,
                }
            ]
        )

    def test_title_is_fixed_and_ids_do_not_move(self) -> None:
        raw = (
            "<p><b>ARTICLE VI</b></p><p><b>REPRESENTATIONS</b></p>"
            "<p><b>6.24</b> <b>A</b><b>nti</b><b>-Corruption Matters</b>. Since January 1, 2020, none of the "
            "Companies has filed any <font>Tax</font><font> </font><font>Return</font> late.</p>"
        )
        tagged = (
            "<article>ARTICLE VI REPRESENTATIONS</article><section>6.24 Anti -Corruption Matters.</section> "
            "Since January 1, 2020, none of the Companies has filed any TaxReturn late."
        )
        generated, failures = f_xml.generate_xml(self._df(tagged, raw))
        self.assertEqual(failures, [])
        section = ET.fromstring(generated[0].xml).find(".//section")
        assert section is not None
        self.assertEqual(section.get("title"), "6.24 Anti-Corruption Matters.")
        self.assertIn("any Tax Return late", "".join(section.itertext()))

        original = f_xml._restore_raw_spaces
        try:
            f_xml._restore_raw_spaces = lambda groups: [list(texts) for _, texts in groups]
            baseline, _ = f_xml.generate_xml(self._df(tagged, raw))
        finally:
            f_xml._restore_raw_spaces = original
        base_root = ET.fromstring(baseline[0].xml)
        new_root = ET.fromstring(generated[0].xml)
        for tag in ("article", "section"):
            self.assertEqual(
                [el.get("uuid") for el in base_root.iter(tag)],
                [el.get("uuid") for el in new_root.iter(tag)],
            )
        base_section = base_root.find(".//section")
        assert base_section is not None
        self.assertEqual(base_section.get("title"), "6.24 Anti -Corruption Matters.")


if __name__ == "__main__":
    unittest.main()
