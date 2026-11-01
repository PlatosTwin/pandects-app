# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
"""EDGAR SGML strip for pages from .txt filings.

The page text in these tests is copied from served agreements whose section
text carried SGML markup: 1305cbea, 46ec21d5, 4818e5db, 48e3b7de, a0086b17.
"""
from datetime import date
import time
import unittest

import pandas as pd

from etl.domain import f_xml
from etl.domain.edgar_sgml import strip_edgar_sgml, strip_edgar_sgml_with_report


EQ = "=" * 70


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

    def test_title_decoration_is_removed(self) -> None:
        # 1305cbea, page 1 (front matter)
        text = (
            "EXHIBIT 10.2\n\n" + EQ + "\n\nAGREEMENT OF PURCHASE AND SALE\n\n"
            "MARCH 12, 2007\n\n" + EQ
        )
        result = strip_edgar_sgml_with_report(text, page_type="front_matter", page_order=1)
        self.assertEqual(result.lines_removed, [EQ, EQ])
        self.assertEqual(result.rule_lines_kept, [])

    def test_dated_as_of_decoration_is_removed(self) -> None:
        # f8698e9e, page 1 (front matter): dash borders next to "=" borders.
        dash = "-" * 32
        text = (
            "EXHIBIT 2.1\n\n" + EQ + "\n\nAGREEMENT AND PLAN OF MERGER AND REORGANIZATION\n\n"
            "EOS BIOTECHNOLOGY, INC., a Delaware corporation\n\n" + dash + "\n\n"
            "Dated as of February 3, 2003\n\n" + dash + "\n\n" + EQ
        )
        result = strip_edgar_sgml_with_report(text, page_type="front_matter", page_order=1)
        self.assertEqual(result.lines_removed, [EQ, dash, dash, EQ])

    def test_stacked_cover_borders_are_removed(self) -> None:
        # 52358497, page 1 (front matter)
        rule = "-" * 80
        text = "EXHIBIT 2.1\n\n" + rule + "\n\n" + rule + "\n\nASSET PURCHASE AGREEMENT"
        result = strip_edgar_sgml_with_report(text, page_type="front_matter", page_order=1)
        self.assertEqual(result.lines_removed, [rule, rule])

    def test_equals_border_around_an_exhibit_title_is_removed_after_page_two(self) -> None:
        # 424dbca3, page 106 (back matter): exhibit cover inside the filing.
        text = "EXHIBIT B-1\n\n" + EQ + "\n\nDOMESTIC DISTRIBUTION AGREEMENT\n\nAS OF [ ], 2005\n\n" + EQ
        result = strip_edgar_sgml_with_report(text, page_type="back_matter", page_order=106)
        self.assertEqual(result.lines_removed, [EQ, EQ])

    def test_footnote_separator_is_removed(self) -> None:
        # f8698e9e, page 11 (body)
        text = (
            "<section>1.13 Exemption from Registration, California Permit.</section> The "
            "shares shall be issued.\n\n---------------------\n\n* Certain information on "
            "this page has been omitted and filed separately with the Commission."
        )
        result = strip_edgar_sgml_with_report(text, page_type="body", page_order=11)
        self.assertEqual(result.lines_removed, ["---------------------"])

    def test_table_row_separators_are_removed(self) -> None:
        # c0a51bb3, page 13 (body)
        rule = "-" * 70
        text = (
            "Equal or exceed $20,000,000 $1,500,000 but are less than $22,500,000\n\n" + rule + "\n\n"
            "Equal or exceed $22,500,000 $4,500,000 but are less than $25,000,000\n\n" + rule + "\n\n"
            "Equal or exceed $25,000,000 $5,000,000\n\n" + rule + "\n\n"
            '(b) As used herein, "Media Revenues" shall mean net cash revenues.'
        )
        result = strip_edgar_sgml_with_report(text, page_type="body", page_order=13)
        self.assertEqual(result.lines_removed, [rule, rule, rule])

    def test_rules_inside_a_text_table_are_removed_even_when_stacked(self) -> None:
        # 91a6e6fb, page 119 (back matter)
        rule = "-" * 70
        text = (
            "<TABLE> <CAPTION> PRO RATA PERCENTAGE SHARES TO BE HELD IN INTEREST\n\n"
            + rule + "\n\n" + rule + "\n\n" + rule + "\n\n</TABLE>"
        )
        result = strip_edgar_sgml_with_report(text, page_type="back_matter", page_order=119)
        self.assertEqual(result.lines_removed, [rule, rule, rule])
        self.assertEqual(result.text, "PRO RATA PERCENTAGE SHARES TO BE HELD IN INTEREST\n\n\n\n")

    def test_rule_lines_on_signature_pages_are_kept(self) -> None:
        # 1be86269, page 66 (sig): signature lines under "Very truly yours,".
        rule = "-" * 30
        text = "Very truly yours,\n\nGANNETT CO., INC.\n\nBy:\n\n" + rule + "\n\n" + rule
        result = strip_edgar_sgml_with_report(text, page_type="sig", page_order=66)
        self.assertEqual(result.text, text)
        self.assertEqual([r for _, r in result.rule_lines_kept], ["sig_page", "sig_page"])

    def test_notice_address_blanks_are_kept(self) -> None:
        # 3def7680, page 38 (body)
        rule = "-" * 32
        text = (
            "Nuevo Energy Company\n\n" + rule + "\n\n" + rule + "\n\n"
            "Attention: ----------------------"
        )
        result = strip_edgar_sgml_with_report(text, page_type="body", page_order=38)
        self.assertEqual(result.text, text)
        self.assertEqual(result.lines_removed, [])

    def test_witness_address_stack_is_kept(self) -> None:
        # 4818e5db, page 122 (back matter)
        rule = "-" * 37
        text = "Address\n\n" + "\n\n".join([rule] * 3) + "\n\nOccupation Solicitor"
        result = strip_edgar_sgml_with_report(text, page_type="back_matter", page_order=122)
        self.assertEqual(result.text, text)
        self.assertEqual(
            [r for _, r in result.rule_lines_kept], ["form_label", "next_to_blank", "next_to_blank"]
        )

    def test_initials_blanks_are_kept(self) -> None:
        # 5484d1a7, page 12 (body)
        text = (
            "Seller's Initials Purchaser's Initials\n\n---------------------- ------------------------\n\n"
            "<section>3.16 Financial Information.</section> Seller has previously delivered"
        )
        self.assertEqual(strip_edgar_sgml(text, page_type="body", page_order=12), text)

    def test_blanks_after_a_colon_are_kept(self) -> None:
        # ae704b04, page 62 (back matter) and 91a6e6fb, page 82 (proxy answer blanks)
        rule = "-" * 24
        notice = "If to Shareholder:\n\n" + "\n\n".join([rule] * 3) + "\n\nWith a copy to:"
        self.assertEqual(strip_edgar_sgml(notice, page_type="back_matter", page_order=62), notice)
        proxy = (
            "Number of shares of common stock of the Company owned of record as of the date "
            "of this proxy:\n\n" + "-" * 35 + "\n\nNumber of shares of preferred stock of the "
            "Company owned of record as of the date of this proxy:\n\n" + "-" * 35
        )
        self.assertEqual(strip_edgar_sgml(proxy, page_type="back_matter", page_order=82), proxy)

    def test_rule_next_to_a_signatory_name_is_kept_after_page_two(self) -> None:
        rule = "-" * 35
        text = "VIEWPOINT INTERNATIONAL, INC.\n\n" + rule
        self.assertEqual(strip_edgar_sgml(text, page_type="body", page_order=40), text)
        self.assertEqual(
            strip_edgar_sgml(text, page_type="front_matter", page_order=1),
            "VIEWPOINT INTERNATIONAL, INC.\n",
        )

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

    def test_long_dash_run_that_is_not_a_rule_is_fast(self) -> None:
        # A nested-quantifier rule pattern backtracked exponentially on this.
        text = "-" * 60 + " " + "-" * 60 + " x"
        start = time.perf_counter()
        self.assertEqual(strip_edgar_sgml(text), text)
        self.assertLess(time.perf_counter() - start, 0.5)

    def test_long_whitespace_runs_are_linear(self) -> None:
        for text in (" " * 50_000, "<page>" + " " * 50_000, "<S>" + " " * 50_000 + "x"):
            start = time.perf_counter()
            strip_edgar_sgml(text)
            self.assertLess(time.perf_counter() - start, 0.1, repr(text[:10]))

    def test_paired_page_marker_split_across_lines_is_kept(self) -> None:
        self.assertEqual(strip_edgar_sgml("text <page>\n12</page>"), "text <page>\n12</page>")

    def test_sec_header_block_is_removed_whole(self) -> None:
        # Shape of the SEC header in babaaafd / 33784f2c (front matter).
        text = (
            "<SEC-DOCUMENT>0001035704-00-000289.txt : 20000509 <SEC-HEADER>0001035704-00-000289.hdr.sgml "
            ": 20000509 ACCESSION NUMBER: 0001035704-00-000289\n\nFILER:\n\n"
            "MAIL ADDRESS: STREET 1: 50 CHARLES LINDBERG BLVD </MAIL-ADDRESS>\n\n"
            "FORMER COMPANY: USAT HOLDINGS INC </FORMER-COMPANY> </FILER> </SEC-HEADER> "
            "<DOCUMENT> <TYPE>8-K <SEQUENCE>1 <FILENAME>0001.txt <DESCRIPTION>FORM 8-K <TEXT>\n\n"
            "SECURITIES AND EXCHANGE COMMISSION"
        )
        result = strip_edgar_sgml_with_report(text, page_type="front_matter", page_order=0)
        self.assertEqual(
            result.text, "0001035704-00-000289.txt : 20000509\n\nSECURITIES AND EXCHANGE COMMISSION"
        )
        self.assertIn("<SEC-HEADER>...</SEC-HEADER>", result.tags_removed)

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

    def test_hook_passes_page_type_so_sig_rule_lines_stay(self) -> None:
        text = "By:\n\n" + "-" * 30 + "\n\n<S> John Smith"
        sig = self._row(source_page_type="sig", tagged_output=text, page_order=60)
        self.assertEqual(f_xml._xml_page_text(sig), "By:\n\n" + "-" * 30 + "\n\nJohn Smith")
        body = self._row(tagged_output="Text.\n\n" + "-" * 30 + "\n\nMore.", page_order=1)
        self.assertEqual(f_xml._xml_page_text(body), "Text.\n\n\nMore.")

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
