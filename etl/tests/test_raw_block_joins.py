# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
"""Page-text lines split inside one printed HTML block are rejoined.

The positive cases are trimmed from served agreements:
* 2058073b-8cd0-5d99-b597-c56e1c4c023a page 12: the page starts mid-block, so its
  text has no enclosing element and each <u> run became its own line;
* bf2abb1c-c1a2-5214-b631-b11166225648 page 4: blank lines inside one <div>;
* cfc1e262-8ead-5ab2-83bb-2e47f04aaffd page 77: ``provided`` / ``, that``;
* cde2064f-e73e-503b-94de-a5e4f7ed7834: ``P`` / ``ropriety`` printed in two
  adjacent <font> runs with no space between them.
"""
import unittest
from unittest import mock

from etl.domain import f_xml
from etl.domain import raw_block_joins
from etl.domain.raw_block_joins import join_raw_block_splits, raw_stream

RAW_ROOT_INLINE = (
    "reasonably request), be entitled to receive, and Parent shall cause the Paying Agent to pay and deliver as "
    "promptly as reasonably practicable after the Effective Time, a cash amount in immediately available funds "
    "(after giving effect to any required\n    Tax withholdings as provided in <u>Section 2.3(e))</u> equal to the "
    "product obtained by <u>multiplying</u> (A) the number of Shares represented by such Book-Entry Shares by (B) "
    "the Per Share Merger Consideration. No interest will be paid or accrued\n    on any amount payable upon due "
    "surrender of the Book-Entry Shares.\n    <div style=\"text-indent: 108pt;\"><br/>\n</div>\n"
    "<div>(c) Next paragraph of the agreement starts here.</div>"
)
TEXT_ROOT_INLINE = (
    "reasonably request), be entitled to receive, and Parent shall cause the Paying Agent to pay and deliver as "
    "promptly as reasonably practicable after the Effective Time, a cash amount in immediately available funds "
    "(after giving effect to any required Tax withholdings as provided in\n\n"
    "Section 2.3(e))\n\n"
    "equal to the product obtained by\n\n"
    "multiplying\n\n"
    "(A) the number of Shares represented by such Book-Entry Shares by (B) the Per Share Merger Consideration. "
    "No interest will be paid or accrued on any amount payable upon due surrender of the Book-Entry Shares.\n\n"
    "(c) Next paragraph of the agreement starts here."
)

RAW_BLANK_LINES = (
    "<div style=\"text-align: justify;\">This PURCHASE AND SALE AGREEMENT is entered into as of November 19, 2024, "
    "by and among ONEOK Partners Intermediate Limited\n          Partnership, a Delaware limited partnership "
    "(“<font style=\"font-weight: bold;\">Seller One</font>”), DTM Interstate Transportation, LLC, a Delaware "
    "limited liability company (“<font style=\"font-weight: bold;\">Buyer</font>”),\n\n\n\n          and, solely "
    "for the purposes of <u>Section\xa09.13</u>, DT Midstream, Inc., a Delaware corporation (the "
    "“<font style=\"font-weight: bold;\">Guarantor</font>”).</div>\n<div>\xa0</div>\n"
    "<div style=\"text-align: center; font-weight: bold;\">RECITALS</div>\n<div>\xa0</div>\n"
    "<div style=\"text-align: justify; text-indent: 36pt;\">WHEREAS, Seller One owns one hundred percent (100%) "
    "of the issued and outstanding Equity Interests.</div>"
)
TEXT_BLANK_LINES = (
    "This PURCHASE AND SALE AGREEMENT is entered into as of November 19, 2024, by and among ONEOK Partners "
    "Intermediate Limited Partnership, a Delaware limited partnership (“Seller One”), DTM Interstate "
    "Transportation, LLC, a Delaware limited liability company (“Buyer”),\n\n"
    "and, solely for the purposes of Section 9.13, DT Midstream, Inc., a Delaware corporation (the “Guarantor”).\n\n"
    "RECITALS\n\n"
    "WHEREAS, Seller One owns one hundred percent (100%) of the issued and outstanding Equity Interests."
)

RAW_NO_SPACE_PUNCT = (
    "<p><font face=\"Times New Roman\">For the avoidance of doubt, Sellers and their Affiliates shall retain all "
    "Liabilities arising out of the employment or termination of employment of any Noticed Business Employee; "
    "</font><u><font face=\"Times New Roman\">provided</font></u><font face=\"Times New Roman\">, that Purchaser "
    "agrees to reimburse Parent promptly for the base salary of any Noticed Business Employee.</font></p>"
)
TEXT_NO_SPACE_PUNCT = (
    "For the avoidance of doubt, Sellers and their Affiliates shall retain all Liabilities arising out of the "
    "employment or termination of employment of any Noticed Business Employee;\n\n"
    "provided\n\n"
    ", that Purchaser agrees to reimburse Parent promptly for the base salary of any Noticed Business Employee."
)

RAW_FONT_SPLIT_WORD = (
    "services or facilities to any Seller or any officer, manager, member or Affiliate of any Seller (other than "
    "such Company).</font>\n</p><a name=\"_Toc11273528\"></a>"
    "<font style=\"display: inline;font-family:Times New Roman;\">P</font>"
    "<font style=\"display: inline;font-family:Times New Roman;\">ropriety of Past Payments</font> and the "
    "related matters described in the Disclosure Schedule\n<p>Next paragraph.</p>"
)
TEXT_FONT_SPLIT_WORD = (
    "services or facilities to any Seller or any officer, manager, member or Affiliate of any Seller (other than "
    "such Company).\n\n"
    "P\n\n"
    "ropriety of Past Payments and the related matters described in the Disclosure Schedule\n\n"
    "Next paragraph."
)

LONG = (
    "the Company shall use its commercially reasonable efforts to preserve substantially intact in all material "
    "respects its business organization and material business relationships"
)


class JoinRawBlockSplitsTests(unittest.TestCase):
    def test_top_level_inline_runs_are_one_block(self) -> None:
        result = join_raw_block_splits(TEXT_ROOT_INLINE, RAW_ROOT_INLINE, collect=True)
        self.assertIn(
            "as provided in Section 2.3(e)) equal to the product obtained by multiplying (A) the number of Shares",
            result.text,
        )
        # The (c) paragraph is its own <div>: kept apart.
        self.assertTrue(result.text.endswith("Book-Entry Shares.\n\n(c) Next paragraph of the agreement starts here."))
        # "multiplying" / "(A) the number": an inline enumerator, same source line.
        self.assertEqual(len(result.joins), 4)

    def test_blank_lines_inside_a_div_are_one_paragraph(self) -> None:
        result = join_raw_block_splits(TEXT_BLANK_LINES, RAW_BLANK_LINES, collect=True)
        self.assertIn("limited liability company (“Buyer”), and, solely for the purposes of Section 9.13", result.text)
        # The heading and the next paragraph are separate blocks.
        self.assertIn("(the “Guarantor”).\n\nRECITALS\n\nWHEREAS,", result.text)
        self.assertEqual(len(result.joins), 1)

    def test_join_keeps_the_raw_separator_before_punctuation(self) -> None:
        result = join_raw_block_splits(TEXT_NO_SPACE_PUNCT, RAW_NO_SPACE_PUNCT)
        self.assertIn("Noticed Business Employee; provided, that Purchaser agrees", result.text)
        self.assertNotIn("provided ,", result.text)

    def test_word_split_across_font_tags_is_joined_without_a_space(self) -> None:
        result = join_raw_block_splits(TEXT_FONT_SPLIT_WORD, RAW_FONT_SPLIT_WORD)
        self.assertIn("\n\nPropriety of Past Payments and the related matters", result.text)
        self.assertNotIn("P ropriety", result.text)
        # The sentence before it ends a separate <p>.
        self.assertIn("(other than such Company).\n\nPropriety", result.text)

    def test_only_whitespace_changes(self) -> None:
        for text, raw in (
            (TEXT_ROOT_INLINE, RAW_ROOT_INLINE),
            (TEXT_BLANK_LINES, RAW_BLANK_LINES),
            (TEXT_NO_SPACE_PUNCT, RAW_NO_SPACE_PUNCT),
            (TEXT_FONT_SPLIT_WORD, RAW_FONT_SPLIT_WORD),
        ):
            result = join_raw_block_splits(text, raw)
            self.assertEqual("".join(result.text.split()), "".join(text.split()))

    def test_separate_paragraphs_are_kept(self) -> None:
        raw = f"<p>{LONG}, provided that</p>\n<p>the Company Board may act</p>"
        text = f"{LONG}, provided that\n\nthe Company Board may act"
        self.assertEqual(join_raw_block_splits(text, raw).text, text)

    def test_br_breaks_are_kept(self) -> None:
        raw = f"<div>{LONG} of the Company<br/>and its Subsidiaries taken as a whole</div>"
        text = f"{LONG} of the Company\n\nand its Subsidiaries taken as a whole"
        self.assertEqual(join_raw_block_splits(text, raw).text, text)

    def test_list_items_are_kept(self) -> None:
        raw = f"<ul><li>{LONG} and</li><li>the books and records of the Company</li></ul>"
        text = f"{LONG} and\n\nthe books and records of the Company"
        self.assertEqual(join_raw_block_splits(text, raw).text, text)

    def test_line_starting_with_an_enumerator_is_kept(self) -> None:
        raw = f"<p>{LONG} as follows\n\n(b) the books and records of the Company</p>"
        text = f"{LONG} as follows\n\n(b) the books and records of the Company"
        self.assertEqual(join_raw_block_splits(text, raw).text, text)

    def test_table_cells_and_rows_are_kept(self) -> None:
        raw = (
            f"<table><tr><td>{LONG} of the</td><td>Company and Parent</td></tr>"
            "<tr><td>and its Subsidiaries taken as a whole</td></tr></table>"
        )
        text = f"{LONG} of the\n\nCompany and Parent\n\nand its Subsidiaries taken as a whole"
        self.assertEqual(join_raw_block_splits(text, raw).text, text)

    def test_headings_are_never_joined(self) -> None:
        raw = (
            f"<p><b>Section 5.1 Conduct of Business</b>\n\nof the Company. {LONG}</p>"
            f"<p>ARTICLE VI\n\nCOVENANTS AND AGREEMENTS\n\nthe parties agree {LONG}</p>"
        )
        text = (
            f"<section>Section 5.1 Conduct of Business</section>\n\nof the Company. {LONG}\n\n"
            f"<article>ARTICLE VI\n\nCOVENANTS AND AGREEMENTS</article>\n\nthe parties agree {LONG}"
        )
        self.assertEqual(join_raw_block_splits(text, raw).text, text)

    def test_untagged_caps_heading_is_kept(self) -> None:
        raw = f"<p>REPRESENTATIONS AND WARRANTIES OF THE COMPANY\n\nThe Company represents {LONG}</p>"
        text = f"REPRESENTATIONS AND WARRANTIES OF THE COMPANY\n\nThe Company represents {LONG}"
        self.assertEqual(join_raw_block_splits(text, raw).text, text)

    def test_sentence_end_before_a_capital_is_kept(self) -> None:
        raw = f"<p>{LONG}.\n\nThe Company represents and warrants to Parent as follows</p>"
        text = f"{LONG}.\n\nThe Company represents and warrants to Parent as follows"
        self.assertEqual(join_raw_block_splits(text, raw).text, text)

    def test_pre_line_breaks_are_kept(self) -> None:
        raw = f"<pre>{LONG} of the\nCompany and its Subsidiaries taken as a whole</pre>"
        text = f"{LONG} of the\n\nCompany and its Subsidiaries taken as a whole"
        self.assertEqual(join_raw_block_splits(text, raw).text, text)

    def test_blank_line_in_top_level_text_is_kept(self) -> None:
        raw = f"{LONG} of the\n\nCompany and its Subsidiaries taken as a whole"
        text = f"{LONG} of the\n\nCompany and its Subsidiaries taken as a whole"
        self.assertEqual(join_raw_block_splits(text, raw).text, text)

    def test_hidden_element_between_is_kept(self) -> None:
        raw = f"<p>{LONG} of the <span style=\"display:none\">x</span>Company and its Subsidiaries</p>"
        stream = raw_stream(raw)
        self.assertNotIn("x", stream.text.replace("taxes", ""))
        text = f"{LONG} of the\n\nCompany and its Subsidiaries"
        self.assertEqual(join_raw_block_splits(text, raw).text, text)

    def test_ambiguous_context_is_kept(self) -> None:
        # The same words occur twice, once split by a <p> and once not: no unique match.
        raw = f"<p>{LONG}</p><p>and the Company</p><p>{LONG} and the Company</p>"
        text = f"{LONG}\n\nand the Company"
        self.assertEqual(join_raw_block_splits(text, raw).text, text)

    def test_missing_raw_is_a_no_op(self) -> None:
        self.assertEqual(join_raw_block_splits(TEXT_ROOT_INLINE, None).text, TEXT_ROOT_INLINE)
        self.assertEqual(join_raw_block_splits(TEXT_ROOT_INLINE, "  ").text, TEXT_ROOT_INLINE)

    def test_stray_end_tag_at_a_page_cut_is_a_break(self) -> None:
        # A page cut mid-block starts inside an element whose start tag is on
        # the previous page; html.parser drops the unmatched end tag.
        raw = f"{LONG} about the Company</div>\nNext paragraph of the agreement begins here"
        text = f"{LONG} about the Company\n\nNext paragraph of the agreement begins here"
        self.assertEqual(join_raw_block_splits(text, raw).text, text)
        raw = f"{LONG} shall be paid</font></p>\n<font>in addition to the amounts set out above</font>"
        text = f"{LONG} shall be paid\n\nin addition to the amounts set out above"
        self.assertEqual(join_raw_block_splits(text, raw).text, text)

    def test_stray_inline_end_tag_still_joins(self) -> None:
        raw = f"{LONG} shall be paid</font>\n<font>in addition to the amounts set out above</font>"
        text = f"{LONG} shall be paid\n\nin addition to the amounts set out above"
        self.assertEqual(
            join_raw_block_splits(text, raw).text,
            f"{LONG} shall be paid in addition to the amounts set out above",
        )

    def test_inline_block_spans_are_separate_boxes(self) -> None:
        raw = (
            f"<p>{LONG}<span style=\"display: inline-block; width: 200pt\">Name of Company</span>"
            "<span style=\"display:inline-block\">Address of Company</span></p>"
        )
        text = f"{LONG}\n\nName of Company\n\nAddress of Company"
        self.assertEqual(join_raw_block_splits(text, raw).text, text)

    def test_page_without_an_unbroken_junction_is_not_parsed(self) -> None:
        raw = f"<p>{LONG}, provided that</p>\n<p>the Company Board may act</p>"
        text = f"{LONG}, provided that\n\nthe Company Board may act"
        with mock.patch.object(raw_block_joins, "raw_stream", side_effect=AssertionError("parsed")):
            self.assertEqual(join_raw_block_splits(text, raw).text, text)

    def test_joins_are_listed_only_on_request(self) -> None:
        self.assertEqual(join_raw_block_splits(TEXT_NO_SPACE_PUNCT, RAW_NO_SPACE_PUNCT).joins, [])
        self.assertEqual(len(join_raw_block_splits(TEXT_NO_SPACE_PUNCT, RAW_NO_SPACE_PUNCT, collect=True).joins), 2)

    def test_break_marker_keeps_text_after_a_self_closing_br(self) -> None:
        # 1aa905b9 page 170: with a <br> start tag as the marker, html.parser
        # nested the text after the next <br/> inside it, the first heading
        # line lost its own next line, and the key matched the body sentence.
        raw = (
            '<p style="text-align: center"><br/>\nNUVATION BIO INC.</p>\n<p>\xa0</p>\n'
            '<p style="text-align: center">Pursuant to Section 251 of the<br/>\n'
            "General Corporation Law of the State of Delaware</p>\n<p>\xa0</p>\n"
            "<p>Pursuant to Section 251\nof the General Corporation Law of the State of Delaware (“DGCL”), "
            "the undersigned corporation, a Delaware corporation, does hereby certify</p>"
        )
        text = (
            "NUVATION BIO INC.\n\nPursuant to Section 251 of the\n\nGeneral Corporation Law of the State of "
            "Delaware\n\nPursuant to Section 251 of the General Corporation Law of the State of Delaware (“DGCL”), "
            "the undersigned corporation, a Delaware corporation, does hereby certify"
        )
        self.assertEqual(raw_stream(raw).text.count("General Corporation Law of the State of Delaware"), 2)
        self.assertEqual(join_raw_block_splits(text, raw).text, text)


class XmlPageTextTests(unittest.TestCase):
    def _row(self, page_type: str, *, html: bool = True) -> dict[str, object]:
        return {
            "tagged_output": TEXT_NO_SPACE_PUNCT,
            "raw_page_content": RAW_NO_SPACE_PUNCT,
            "source_page_type": page_type,
            "gold_label": None,
            "source_is_html": html,
            "source_is_txt": not html,
            "page_order": 3,
        }

    def test_body_front_and_back_matter_pages_are_joined(self) -> None:
        for page_type in ("body", "front_matter", "back_matter"):
            self.assertIn("Employee; provided, that", f_xml._xml_page_text(self._row(page_type)))

    def test_txt_pages_are_not_joined(self) -> None:
        self.assertIn("Employee;\n\nprovided", f_xml._xml_page_text(self._row("body", html=False)))


if __name__ == "__main__":
    unittest.main()
