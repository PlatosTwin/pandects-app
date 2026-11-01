# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
"""EDGAR SGML strip for pages from .txt filings.

The page text in these tests is copied from served agreements whose section
text carried SGML markup: 1305cbea, 46ec21d5, 4818e5db, 48e3b7de, a0086b17.
"""
from datetime import date
import unittest

import pandas as pd

from etl.domain import f_xml
from etl.domain.edgar_sgml import strip_edgar_sgml, strip_edgar_sgml_with_report


class StripEdgarSgmlTests(unittest.TestCase):
    def test_document_header_unit_is_removed_with_its_values(self) -> None:
        # 1305cbea, page 0
        text = (
            "<DOCUMENT> <TYPE>EX-10.2 <SEQUENCE>3 <FILENAME>k13277exv10w2.txt "
            "<DESCRIPTION>AGREEMENT OF PURCHASE AND SALE DATED MARCH 12, 2007 <TEXT>\n"
            "\n"
            "EXHIBIT 10.2"
        )
        result = strip_edgar_sgml_with_report(text)
        self.assertEqual(result.text, "\nEXHIBIT 10.2")
        self.assertEqual(len(result.tags_removed), 1)
        self.assertTrue(result.tags_removed[0].startswith("<DOCUMENT>"))

    def test_document_header_keeps_text_that_follows_on_the_line(self) -> None:
        text = (
            "<DOCUMENT> <TYPE>EX-2 <SEQUENCE>2 <FILENAME>merg.txt "
            "<DESCRIPTION>EXHIBIT 2.2 <TEXT> Exhibit 2.2"
        )
        self.assertEqual(strip_edgar_sgml(text), "Exhibit 2.2")

    def test_document_trailer_is_removed_and_page_number_kept(self) -> None:
        self.assertEqual(strip_edgar_sgml("A-9 </TEXT> </DOCUMENT>"), "A-9")
        self.assertEqual(strip_edgar_sgml("x\n\n</TEXT> </DOCUMENT>"), "x\n")

    def test_toc_table_markup_is_removed(self) -> None:
        # a0086b17, page 2 (toc)
        text = (
            "TABLE OF CONTENTS\n"
            "\n"
            "<TABLE> <CAPTION> Page ---- <S> <C> 1. DEFINITIONS......................"
            "........ 1 1.1. Cross Reference Table"
        )
        self.assertEqual(
            strip_edgar_sgml(text),
            "TABLE OF CONTENTS\n\nPage ---- 1. DEFINITIONS.............................. 1 "
            "1.1. Cross Reference Table",
        )

    def test_mixed_case_table_markup_is_removed(self) -> None:
        # 46ec21d5, page 3 (toc)
        text = (
            "<Table> <S> <C> <C> 5.21 Opinion of Financial Advisors.............. 20 "
            "5.22 Brokers.............. 20"
        )
        self.assertEqual(
            strip_edgar_sgml(text),
            "5.21 Opinion of Financial Advisors.............. 20 5.22 Brokers.............. 20",
        )

    def test_definitions_table_in_body_keeps_its_text(self) -> None:
        # 4818e5db, page 3 (body)
        text = (
            "<TABLE> <S> <C> A Ordinary Shares A ordinary shares of 0.01p each in the "
            "capital of the Company\n"
            "\n"
            "B Ordinary Shares B ordinary shares of 0.01p each in the capital of the "
            "Company </TABLE>"
        )
        self.assertEqual(
            strip_edgar_sgml(text),
            "A Ordinary Shares A ordinary shares of 0.01p each in the capital of the Company\n"
            "\n"
            "B Ordinary Shares B ordinary shares of 0.01p each in the capital of the Company",
        )

    def test_section_tags_and_page_markers_survive(self) -> None:
        # 48e3b7de, pages 50-51 (body)
        text = (
            "<section>Section 15.11. Schedules and Exhibits.</section> Seller is "
            "delivering to Buyer the following Schedules:\n"
            "\n"
            "<TABLE> <CAPTION> Disclosure Schedule <S> <C> Schedule 3.3 Subsidiaries "
            "Schedule 3.5 Litigation\n"
            "\n"
            "<page>51</page>\n"
            "51 <TABLE> <S> <C> Schedule 13.1 Cabot Benefit Plans </TABLE>"
        )
        self.assertEqual(
            strip_edgar_sgml(text),
            "<section>Section 15.11. Schedules and Exhibits.</section> Seller is "
            "delivering to Buyer the following Schedules:\n"
            "\n"
            "Disclosure Schedule Schedule 3.3 Subsidiaries Schedule 3.5 Litigation\n"
            "\n"
            "<page>51</page>\n"
            "51 Schedule 13.1 Cabot Benefit Plans",
        )

    def test_orphan_lowercase_page_break_is_removed_but_paired_marker_kept(self) -> None:
        text = "57 <page>\n\n<page>14</page>\n\n<page> Definition Location"
        self.assertEqual(strip_edgar_sgml(text), "57\n\n<page>14</page>\n\nDefinition Location")
        self.assertEqual(strip_edgar_sgml("<PAGE> 2"), "2")
        self.assertEqual(strip_edgar_sgml("<Page> 2"), "2")

    def test_footnote_markup_is_removed(self) -> None:
        text = (
            "GROSS PROFITS 40.50% 38.45% 37.65% <FN> Refer to Schedule 5.1.6(b) of "
            "the purchase agreement </FN> </TABLE>"
        )
        self.assertEqual(
            strip_edgar_sgml(text),
            "GROSS PROFITS 40.50% 38.45% 37.65% Refer to Schedule 5.1.6(b) of the "
            "purchase agreement",
        )

    def test_sec_header_tags_are_removed(self) -> None:
        text = (
            "FORMER COMPANY: FORMER CONFORMED NAME: USAT HOLDINGS INC "
            "</FORMER-COMPANY> </FILER> </SEC-HEADER>"
        )
        self.assertEqual(
            strip_edgar_sgml(text),
            "FORMER COMPANY: FORMER CONFORMED NAME: USAT HOLDINGS INC",
        )

    def test_rule_lines_are_removed(self) -> None:
        # 1305cbea, page 1 (front matter) and 4818e5db, page 122
        text = (
            "EXHIBIT 10.2\n"
            "\n"
            "================================================================================\n"
            "\n"
            "AGREEMENT OF PURCHASE AND SALE\n"
            "\n"
            "-------------------------------------\n"
            "\n"
            "-------- ----------- ----------"
        )
        result = strip_edgar_sgml_with_report(text)
        self.assertEqual(result.text, "EXHIBIT 10.2\n\n\nAGREEMENT OF PURCHASE AND SALE\n\n")
        self.assertEqual(len(result.lines_removed), 3)

    def test_short_rules_and_rules_inside_text_are_kept(self) -> None:
        # a0086b17, page 67 (sig): the dash run is a signature line inside text.
        sig = (
            "/s/ S. Anthony Margolis ----------------------------------- "
            "S. Anthony Margolis c/o Viewpoint International, Inc."
        )
        self.assertEqual(strip_edgar_sgml(sig), sig)
        self.assertEqual(strip_edgar_sgml("---------"), "---------")
        self.assertEqual(strip_edgar_sgml("- ii -"), "- ii -")
        self.assertEqual(strip_edgar_sgml("-i-"), "-i-")
        totals = "Net Contribution -16.48% -16.25% -7.52% ======= ======= ======="
        self.assertEqual(strip_edgar_sgml(totals), totals)

    def test_underscore_lines_are_never_removed(self) -> None:
        # a0086b17, page 67 (sig)
        text = (
            "By: /s/ CT Yeung ______________________________ Name: CT Yeung\n"
            "\n"
            "______________________________\n"
            "\n"
            "By:_____________________________________ Name:___________________"
        )
        self.assertEqual(strip_edgar_sgml(text), text)

    def test_angle_bracket_content_is_kept(self) -> None:
        for text in (
            "the Company's market share is <1% in each such market",
            "with a copy to John Smith <jsmith@example.com>",
            "<21,178,538> shares",
            "<-1- 95>",
            "if x <5 and y >3",
            "Pubco <Sub> and <Company>",
            "<Tables> <Caption 1> <SECTION> <Subsidiary>",
            "Section 2.1 <Text> of the Agreement",
            "<text>unchanged</text>",
            "<pageUUID>abc</pageUUID>",
            "<article>ARTICLE I DEFINITIONS</article>",
        ):
            self.assertEqual(strip_edgar_sgml(text), text, text)

    def test_text_without_markup_is_unchanged(self) -> None:
        text = "  Section 1.1  Definitions.\n\n  (a) indented\t\ttext  "
        self.assertEqual(strip_edgar_sgml(text), text)
        self.assertEqual(strip_edgar_sgml(""), "")


class XmlPageTextHookTests(unittest.TestCase):
    def _row(self, **overrides: object) -> pd.Series:
        row: dict[str, object] = {
            "agreement_uuid": "agreement-1",
            "page_uuid": "page-1",
            "page_order": 1,
            "source_page_type": "body",
            "tagged_output": "<TABLE> <S> <C> Schedule 3.3 Subsidiaries </TABLE>",
            "raw_page_content": None,
            "url": "https://example.com",
            "filing_date": date(2003, 4, 26),
            "source_is_txt": 1,
            "source_is_html": 0,
        }
        row.update(overrides)
        return pd.Series(row)

    def test_txt_page_is_stripped(self) -> None:
        self.assertEqual(f_xml._xml_page_text(self._row()), "Schedule 3.3 Subsidiaries")

    def test_html_page_is_not_stripped(self) -> None:
        row = self._row(source_is_txt=0, source_is_html=1)
        self.assertEqual(
            f_xml._xml_page_text(row), "<TABLE> <S> <C> Schedule 3.3 Subsidiaries </TABLE>"
        )

    def test_null_txt_flag_is_not_stripped(self) -> None:
        for flag in (None, float("nan")):
            row = self._row(source_is_txt=flag, source_is_html=None)
            self.assertIn("<TABLE>", f_xml._xml_page_text(row))

    def test_generated_section_text_has_no_sgml(self) -> None:
        body = (
            "<article>ARTICLE I DEFINITIONS</article>\n\n"
            "<section>Section 1.1 Definitions.</section> The following terms:\n\n"
            "<TABLE> <S> <C> Affiliates Section 11.3 </TABLE>\n\n"
            "-------------------------------------\n\n"
            "<article>ARTICLE II THE SALE</article>\n\n"
            "<section>Section 2.1 Sale.</section> Seller sells. <page>2</page>"
        )
        df = pd.DataFrame([self._row(tagged_output=body)])
        xml_rows, failures = f_xml.generate_xml(df)
        self.assertEqual(failures, [])
        xml = xml_rows[0].xml
        self.assertIn("Affiliates Section 11.3", xml)
        self.assertNotIn("&lt;TABLE&gt;", xml)
        self.assertNotIn("&lt;S&gt;", xml)
        self.assertNotIn("-----------", xml)
        self.assertIn("<page>2</page>", xml)


if __name__ == "__main__":
    unittest.main()
