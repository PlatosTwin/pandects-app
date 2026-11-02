# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
"""Underline dash runs fused into .txt page text.

The raw and page text in these tests is copied from served agreement
b7141f3f-ddba-591b-a10a-8140ea5bbdfe (page_order 10 and 11), where 89 of 92
sections carried runs such as "Company Common Stock, ---------- $0.001 par value".
"""
import time
import unittest

from etl.domain import f_xml
from etl.domain.txt_underlines import raw_underlines, strip_inline_underlines

RAW_1_9 = (
    "     1.9  Lost, Stolen or Destroyed Certificates. In the event that any\n"
    "          --------------------------------------\n"
    "Certificates shall have been lost, stolen or destroyed, the Exchange Agent shall\n"
    "if any, as may be required pursuant to Section 1.6(f) and any dividends or\n"
    "distributions payable pursuant to Section 1.7(d); provided, however, that Parent\n"
    "                                                  --------  -------\n"
    "may, in its discretion and as a condition precedent to the issuance of such ADRs\n"
)
TAGGED_1_9 = (
    "<section>1.9 Lost, Stolen or Destroyed Certificates.</section> In the event that any "
    "-------------------------------------- Certificates shall have been lost, stolen or destroyed, "
    "the Exchange Agent shall if any, as may be required pursuant to Section 1.6(f) and any dividends or "
    "distributions payable pursuant to Section 1.7(d); provided, however, that Parent -------- ------- "
    "may, in its discretion and as a condition precedent to the issuance of such ADRs"
)

RAW_2_1 = (
    "     2.1  Organization and Qualification; Subsidiaries.\n"
    "          --------------------------------------------\n"
    "\n"
    "          (a)  Each of Company and its subsidiaries is a corporation duly organized,\n"
)
TAGGED_2_1 = (
    "<section>2.1 Organization and Qualification; Subsidiaries.</section> --------------------------------------------\n"
    "\n"
    "(a) Each of Company and its subsidiaries is a corporation duly organized,"
)


class StripInlineUnderlinesTests(unittest.TestCase):
    def test_reflowed_caption_underline_is_removed(self) -> None:
        result = strip_inline_underlines(TAGGED_1_9, RAW_1_9)
        self.assertIn("In the event that any Certificates shall have been lost", result.text)
        self.assertIn("provided, however, that Parent may, in its discretion", result.text)
        self.assertNotIn("--", result.text)
        self.assertEqual(len(result.removed), 2)

    def test_underline_after_a_heading_tag_is_removed(self) -> None:
        result = strip_inline_underlines(TAGGED_2_1, RAW_2_1)
        self.assertEqual(
            result.text,
            "<section>2.1 Organization and Qualification; Subsidiaries.</section>\n\n"
            "(a) Each of Company and its subsidiaries is a corporation duly organized,",
        )

    def test_fill_in_blanks_are_kept(self) -> None:
        raw = (
            "     Address: ----------------------\n"
            "     Purchase price: $----------\n"
            "     Attention:\n"
            "                 --------------------\n"
        )
        tagged = "Address: ---------------------- Purchase price: $---------- Attention: --------------------"
        self.assertEqual(raw_underlines(raw), {})
        self.assertEqual(strip_inline_underlines(tagged, raw).text, tagged)

    def test_run_must_sit_under_printed_words(self) -> None:
        # The run starts in the middle of a word and overhangs white space: not an underline.
        raw = "     Notice to Buyer\n        ------------------------\n"
        self.assertEqual(raw_underlines(raw), {})

    def test_underline_one_column_off_still_counts(self) -> None:
        # 0415d686, page_order 24: the banner's underline is typed one column right.
        raw = (
            "                                                                 EXECUTION COPY\n"
            "                                                                  --------------\n"
            "\n"
            "                                   ARTICLE II\n"
        )
        tagged = "EXECUTION COPY --------------\n\n<article>ARTICLE II"
        self.assertEqual(strip_inline_underlines(tagged, raw).text, "EXECUTION COPY\n\n<article>ARTICLE II")

    def test_group_needs_the_raw_anchor_word_and_lengths(self) -> None:
        # Same run length, but the word before it in the page text is not the end
        # of the underlined raw line: the dashes are something else and stay.
        tagged = "Parent shall pay -------------------------------------- the amount"
        self.assertEqual(strip_inline_underlines(tagged, RAW_1_9).text, tagged)

    def test_each_raw_underline_removes_one_group(self) -> None:
        tagged = (
            "In the event that any -------------------------------------- Certificates; "
            "In the event that any -------------------------------------- again"
        )
        result = strip_inline_underlines(tagged, RAW_1_9)
        self.assertEqual(len(result.removed), 1)
        self.assertEqual(result.text.count("-" * 38), 1)

    def test_whole_dash_lines_are_left_to_the_rule_line_strip(self) -> None:
        tagged = "Subsidiaries.\n--------------------------------------------\n(a) Each"
        raw = RAW_2_1
        self.assertEqual(strip_inline_underlines(tagged, raw).text, tagged)

    def test_long_runs_are_fast(self) -> None:
        raw = "x" * 5000 + "\n" + "-" * 5000 + "\n"
        tagged = ("x" * 5000 + " " + "-" * 5000 + " y ") * 20 + "-" * 20000 + "z"
        start = time.perf_counter()
        strip_inline_underlines(tagged, raw)
        self.assertLess(time.perf_counter() - start, 0.5)


class UnderlineShapesTests(unittest.TestCase):
    """Shapes the first rule missed (round-7 audit C3), from served .txt pages."""

    def test_underline_typed_columns_right_of_its_words(self) -> None:
        # a6a4e945, page ee86e570: the caption's underline is typed five columns
        # right of "Capitalization of the Company and its Subsidiaries.".
        raw = (
            "     Section 2.2.  Capitalization of the Company and its Subsidiaries.  The\n"
            "                        --------------------------------------------------\n"
            "authorized capital stock of the Company consists of Twenty-Five Million\n"
        )
        tagged = (
            "<section>Section 2.2. Capitalization of the Company and its Subsidiaries.</section> The "
            + "-" * 50
            + " authorized capital stock of the Company consists of Twenty-Five Million"
        )
        self.assertEqual(
            strip_inline_underlines(tagged, raw).text,
            "<section>Section 2.2. Capitalization of the Company and its Subsidiaries.</section> The "
            "authorized capital stock of the Company consists of Twenty-Five Million",
        )

    def test_underline_off_by_more_columns_mid_line(self) -> None:
        # 38f930c8, page 9c243bb6: "or fails to ----------- notify".
        raw = (
            "     respect to the Third Party Claim under Section 11.01 or fails to\n"
            "                                                 -----------\n"
            "     notify the Indemnified Party within the Dispute Period whether the\n"
        )
        tagged = "under Section 11.01 or fails to ----------- notify the Indemnified Party"
        self.assertEqual(
            strip_inline_underlines(tagged, raw).text,
            "under Section 11.01 or fails to notify the Indemnified Party",
        )

    def test_wrapped_underline_leaves_a_lone_dash_at_column_zero(self) -> None:
        # 2b36bf00, page ef04d220: "Schedule" / "A" wraps, so the next underline
        # line starts with "-" under the "A".
        raw = (
            "interest that would be an Asset hereunder, but that is not described in Schedule\n"
            "                                                                        --------\n"
            "A or Schedule A-1 hereto, including any interest that entitles Cody Energy or\n"
            "-    ------------\n"
            "any of its Subsidiaries to receive more than the NRI set forth in Schedule A or\n"
        )
        tagged = (
            "described in Schedule -------- A or Schedule A-1 hereto, including any interest that "
            "entitles Cody Energy or - ------------ any of its Subsidiaries"
        )
        self.assertEqual(
            strip_inline_underlines(tagged, raw).text,
            "described in Schedule A or Schedule A-1 hereto, including any interest that "
            "entitles Cody Energy or any of its Subsidiaries",
        )

    def test_wrapped_underline_ending_in_two_dashes(self) -> None:
        # 4c070c88, page 4f8bfd6a: "Article" / "IX)" wraps; "--" sits under "IX".
        raw = (
            "Liabilities (the latter being subject to the applicable provisions of Article\n"
            "                                                                      -------\n"
            "IX), including certain actions, proceedings, Tax audits or investigations to\n"
            "--\n"
            "which Sellers or their Affiliates are subject relating to such Retained\n"
        )
        tagged = "Tax audits or investigations to -- which Sellers or their Affiliates"
        self.assertEqual(
            strip_inline_underlines(tagged, raw).text,
            "Tax audits or investigations to which Sellers or their Affiliates",
        )

    def test_lone_dash_without_a_wrapped_underline_is_kept(self) -> None:
        raw = "A or Schedule A-1 hereto, including any interest that entitles Cody Energy or\n--\nany of its\n"
        tagged = "entitles Cody Energy or -- any of its"
        self.assertEqual(raw_underlines(raw), {})
        self.assertEqual(strip_inline_underlines(tagged, raw).text, tagged)

    def test_paired_two_dash_underlines_under_abbreviations(self) -> None:
        # 207ad8d9, page b7238776: "--" under "TE" and under "PP".
        raw = (
            'Company ("OE"), The Cleveland Electric Illuminating Company ("CEI"), The\n'
            "          --                                                  ---\n"
            'Toledo Edison Company ("TE") or Pennsylvania Power Company ("PP"), except,\n'
            "                        --                                   --\n"
            "in the case of any such Subsidiary, to FirstEnergy or such Subsidiary's\n"
        )
        tagged = 'Power Company ("PP"), except, -- -- in the case of any such Subsidiary'
        self.assertEqual(
            strip_inline_underlines(tagged, raw).text,
            'Power Company ("PP"), except, in the case of any such Subsidiary',
        )

    def test_single_two_dash_fraction_bar_is_kept(self) -> None:
        # be78ab02, page 24b2ed8a: SE = AE x AA / PA, typed over three lines.
        raw = (
            "                                  AE x AA = SE\n"
            "                                       --\n"
            "                                       PA\n"
        )
        tagged = "the following formula: AE x AA = SE -- PA"
        self.assertEqual(raw_underlines(raw), {})
        self.assertEqual(strip_inline_underlines(tagged, raw).text, tagged)

    def test_stray_word_between_a_line_and_its_underline(self) -> None:
        # 207ad8d9, page b722ac3e: "GPU" wrapped onto its own line above the
        # underline of "Environmental Protection." and "Compliance.".
        raw = (
            "     Section 3.19  Environmental Protection.  (a)  Compliance.  Each of\n"
            "GPU\n"
            "                   ------------------------        ----------\n"
            "and its Subsidiaries is in compliance with all applicable Environmental\n"
        )
        tagged = (
            "<section>Section 3.19 Environmental Protection.</section> (a) Compliance. Each of GPU "
            "------------------------ ---------- and its Subsidiaries is in compliance"
        )
        self.assertEqual(
            strip_inline_underlines(tagged, raw).text,
            "<section>Section 3.19 Environmental Protection.</section> (a) Compliance. Each of GPU "
            "and its Subsidiaries is in compliance",
        )

    def test_stray_label_above_a_signature_line_is_kept(self) -> None:
        raw = "                   ACME HOLDINGS CORP.\nBy:\n                   ------------------\n"
        tagged = "ACME HOLDINGS CORP. By: ------------------ Name:"
        self.assertEqual(raw_underlines(raw), {})
        self.assertEqual(strip_inline_underlines(tagged, raw).text, tagged)

    def test_bare_label_above_a_signature_line_is_kept(self) -> None:
        raw = "                   ACME HOLDINGS CORP.\nBy\n                   ------------------\n"
        tagged = "ACME HOLDINGS CORP. By ------------------ Name:"
        self.assertEqual(strip_inline_underlines(tagged, raw).text, tagged)

    def test_table_rule_under_a_row_of_figures_is_kept(self) -> None:
        # 329556e7, page edce28e6: a consolidating balance sheet's total rule.
        raw = (
            "   Total current assets   4,502,742      (151)   743,214    11,981,424       (86,930)   (150)  "
            "21,071,446   (44,938)    38,166,657\n"
            "                        " + "-" * 106 + "\n"
            "Restricted cash                   0         0          0       690,608\n"
        )
        tagged = "(150) 21,071,446 (44,938) 38,166,657 " + "-" * 106 + " Restricted cash 0"
        self.assertEqual(raw_underlines(raw), {})
        self.assertEqual(strip_inline_underlines(tagged, raw).text, tagged)

    def test_prose_dash_after_the_underline_stays(self) -> None:
        # c7133453: "PREMIUM YEAR 1" is underlined; the next line opens with "- ".
        raw = "     PREMIUM YEAR 1\n     --------------\n     - Net Sales of Pediatric Products = $30,000,000\n"
        tagged = "PREMIUM YEAR 1 -------------- - Net Sales of Pediatric Products = $30,000,000"
        self.assertEqual(
            strip_inline_underlines(tagged, raw).text,
            "PREMIUM YEAR 1 - Net Sales of Pediatric Products = $30,000,000",
        )

    def test_signature_line_under_a_typed_name_is_kept(self) -> None:
        # 767fb39a, page 07868570: a run-in signature block in a body section.
        raw = (
            "                                   By: /s/ James A. Sharman\n"
            "                                       ------------------------\n"
            "                                   Name:  James A. Sharman\n"
        )
        tagged = "By: /s/ James A. Sharman ------------------------ Name: James A. Sharman"
        self.assertEqual(raw_underlines(raw), {})
        self.assertEqual(strip_inline_underlines(tagged, raw).text, tagged)
        # 1dbacab6, page 5adcb316, and 74731e9b, page b70e56da: overhang of two
        # or fewer columns, still a signature line.
        for raw in (
            "                                          by  /s/ J. Wayne Leonard\n"
            "                                              ----------------------\n",
            "                                       Title: Executive Vice President and CFO\n"
            "                                             -----------------------------------\n",
            "                       PHOTONIC INTEGRATION RESEARCH, INC.\n"
            "                            -------------------------\n",
        ):
            self.assertEqual(raw_underlines(raw), {})

    def test_shifted_run_under_a_figure_is_kept(self) -> None:
        # 183af65b, page 8b0f1ba2: a total rule under a share count.
        raw = (
            "          Swag Mukerji                   Hydron Investments Limited          500,000\n"
            "                                                                             ---------\n"
        )
        self.assertEqual(raw_underlines(raw), {})

    def test_column_header_rules_and_blanks_are_not_shifted(self) -> None:
        # 2b0b2c20, page e6f58566 (header rules) and afcb2464, page 48b8f258 (a
        # blank after "as of this").
        headers = "                                                         For      Against\n" + " " * 57 + "-----    -------\n"
        blank = (
            "          THIS ASSUMPTION AGREEMENT, made, executed and delivered as of this\n"
            + " " * 77 + "---\n"
            "day of      , 2003 by United States Steel Corporation, a Delaware corporation\n"
        )
        self.assertEqual(raw_underlines(headers), {})
        self.assertEqual(raw_underlines(blank), {})

    def test_dashes_printed_inline_in_the_raw_line_are_kept(self) -> None:
        # 97078e97, page ec69e7c8: the filing itself prints the dashes in the
        # sentence, so they cannot be told from a fill-in blank.
        raw = "          Liens that secure liabilities that are reflected in the ----- Company\n"
        tagged = "Liens that secure liabilities that are reflected in the ----- Company"
        self.assertEqual(strip_inline_underlines(tagged, raw).text, tagged)


class XmlPageTextUnderlineTests(unittest.TestCase):
    def test_body_txt_page_loses_underlines_but_sig_page_keeps_them(self) -> None:
        row = {
            "tagged_output": TAGGED_1_9,
            "raw_page_content": RAW_1_9,
            "source_page_type": "body",
            "source_is_txt": True,
            "source_is_html": False,
            "page_order": 10,
        }
        self.assertNotIn("--", f_xml._xml_page_text(row))
        sig_row = dict(row, source_page_type="sig")
        self.assertIn("-" * 38, f_xml._xml_page_text(sig_row))

    def test_html_page_is_untouched(self) -> None:
        row = {
            "tagged_output": TAGGED_1_9,
            "raw_page_content": RAW_1_9,
            "source_page_type": "body",
            "source_is_txt": False,
            "source_is_html": True,
            "page_order": 10,
        }
        self.assertEqual(f_xml._xml_page_text(row), TAGGED_1_9)


if __name__ == "__main__":
    unittest.main()
