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
