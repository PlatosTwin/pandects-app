import unittest
from collections import Counter
from datetime import date

import pandas as pd

from etl.domain.f_xml import generate_xml
from etl.domain.page_joins import JoinPage, agreement_words, join_across_pages, read_raw_page


def _pages(*texts: str, raws: tuple[str, ...] | None = None, orders: tuple[int, ...] | None = None) -> list[JoinPage]:
    return [
        JoinPage(
            text=text,
            raw=raws[i] if raws else None,
            is_html=raws is not None,
            page_order=orders[i] if orders else i,
        )
        for i, text in enumerate(texts)
    ]


def _html(*paragraphs: str, style: str = "text-align:justify") -> str:
    return "".join(f'<p style="{style}">{p}</p>' for p in paragraphs)


class TextRuleTests(unittest.TestCase):
    def joined(self, left: str, right: str) -> list[str]:
        return join_across_pages(_pages(left, right)).page_texts

    def assert_kept(self, left: str, right: str, reason: str) -> None:
        result = join_across_pages(_pages(left, right))
        self.assertEqual(result.page_texts, [left, right])
        self.assertEqual(result.skipped[reason], 1, result.skipped)

    def test_joins_lowercase_continuation_and_keeps_marker_after_it(self) -> None:
        left = "<section>3.11</section> any amounts payable by the Purchaser\n\n<page>5</page>"
        right = "to the Sellers as the case may be.\n\n<section>3.12</section> Each Seller agrees."
        self.assertEqual(
            self.joined(left, right),
            [
                "<section>3.11</section> any amounts payable by the Purchaser to the Sellers as the case may be.\n\n<page>5</page>",
                "<section>3.12</section> Each Seller agrees.",
            ],
        )

    def test_right_text_stops_at_a_heading_on_its_line(self) -> None:
        self.assertEqual(
            self.joined("the amount payable by the", "buyer at closing. <section>3.12</section> Each Seller agrees."),
            ["the amount payable by the buyer at closing.", "<section>3.12</section> Each Seller agrees."],
        )

    def test_right_text_behind_a_top_page_number(self) -> None:
        self.assertEqual(
            self.joined("other than those services identified", "<page>37</page> as Omitted Services.\n\nNext paragraph."),
            ["other than those services identified as Omitted Services.", "<page>37</page> Next paragraph."],
        )

    def test_left_line_under_a_top_page_number_joins(self) -> None:
        self.assertEqual(
            self.joined("<page>4</page>\n\nthe amount payable by the buyer", "at closing."),
            ["<page>4</page>\n\nthe amount payable by the buyer at closing.", ""],
        )

    def test_open_end_word_joins_capitalized_right(self) -> None:
        self.assertEqual(
            self.joined("shall deliver to the Purchaser all of the", "Company Records at Closing.")[0],
            "shall deliver to the Purchaser all of the Company Records at Closing.",
        )

    def test_all_caps_runs_join(self) -> None:
        left = "EACH PARTY WAIVES " * 7 + "ANY OTHER MATERIALS OR"
        self.assertEqual(self.joined(left, "INFORMATION MADE AVAILABLE.")[0], left + " INFORMATION MADE AVAILABLE.")

    def test_closing_punctuation_joins_without_space(self) -> None:
        self.assertEqual(
            self.joined("the amount, if any, of the deductible provided", ", however, that it applies.")[0],
            "the amount, if any, of the deductible provided, however, that it applies.",
        )

    def test_chain_through_a_page_of_one_paragraph(self) -> None:
        texts = join_across_pages(_pages("the amount payable by", "the buyer at the closing of", "the transactions.")).page_texts
        self.assertEqual(texts, ["the amount payable by the buyer at the closing of the transactions.", "", ""])

    def test_left_ends_sentence(self) -> None:
        self.assert_kept("The Buyer shall pay.", "the amount due.", "left_ends_sentence")
        self.assert_kept("The Buyer shall pay:", "the amount due.", "left_ends_sentence")
        self.assert_kept("the “Company.”", "the amount due.", "left_ends_sentence")

    def test_left_ends_list_item(self) -> None:
        self.assert_kept("to pay all Taxes due; and", "to file all returns.", "left_list_item_end")

    def test_left_heading(self) -> None:
        self.assert_kept("<section>Section 5.1 Taxes.</section>", "the amount due.", "left_heading")
        self.assert_kept("<article>ARTICLE V", "the amount due.", "left_heading")

    def test_left_caption(self) -> None:
        self.assert_kept("Employee Matters", "the amount due.", "left_caption")
        self.assert_kept("SCHEDULE OF TAXES", "the amount due.", "left_caption")

    def test_left_after_a_foot_page_number(self) -> None:
        self.assert_kept("the amount payable\n\n<page>5</page>\n\nMerger agreement draft", "to the Sellers.", "left_after_page_number")

    def test_left_dash(self) -> None:
        self.assert_kept("the accounts shall -", "be prepared on time.", "left_dash")

    def test_notice_blocks(self) -> None:
        self.assert_kept("Email: jane@example.com", "with a copy to:", "left_contact")
        self.assert_kept("Facsimile: (212) 558-3588", "or to such other address.", "left_contact")
        self.assert_kept("Houston and the other offices of the firm", "with a copy to:", "right_contact")

    def test_right_heading(self) -> None:
        self.assert_kept("the amount payable by the", "<section>3.12</section> Each Seller agrees.", "right_heading")

    def test_right_enumerator(self) -> None:
        for right in ("(a) the amount due.", "(iv) the amount due.", "a. the amount due.", "ii. the amount due.", "• the amount due."):
            self.assert_kept("the amount payable by the", right, "right_enumerator")

    def test_bare_page_number_after_the_cut_stays(self) -> None:
        left = "to Articles, Sections, Schedules and Exhibits mean the"
        self.assert_kept(left, "28", "right_no_letters")
        self.assert_kept(left, "(28)", "right_no_letters")
        self.assert_kept(left, "A-8", "right_label")
        self.assert_kept(left, "iv", "right_label")

    def test_decimal_amount_with_words_after_an_open_word_joins(self) -> None:
        self.assertEqual(
            self.joined("the Purchaser shall pay an amount equal to", "28 days of interest on the balance.")[0],
            "the Purchaser shall pay an amount equal to 28 days of interest on the balance.",
        )

    def test_page_of_footnotes_after_the_cut_stays(self) -> None:
        left = "in the ordinary course of business consistent with past practice that"
        self.assert_kept(left, "1 Please provide schedule.\n\n<page>50</page>", "right_footnote")
        self.assert_kept(left, "1 Please provide schedule.\n\n2 Same for the other.", "right_footnote")

    def test_number_then_capital_with_text_after_it_joins(self) -> None:
        left = "in the ordinary course of business within the"
        self.assertEqual(len(join_across_pages(_pages(left, "30 Business Days after Closing.\n\nThe next paragraph.")).joins), 1)

    def test_dagger_note_stays(self) -> None:
        self.assert_kept("in the ordinary course of business that", "† Please provide schedule.", "right_enumerator")

    def test_multi_level_list_number_stays(self) -> None:
        left = "7.7.5 under the Earn-Out Agreement and"
        self.assert_kept(left, "7.7.6 then, with respect to Claims, the Buyer shall", "right_enumerator")
        self.assert_kept("the Buyer shall deliver the amount under this Agreement or", "7.7 then, with respect to Claims,", "right_enumerator")
        self.assert_kept("the Buyer shall deliver the amount under all of the", "7.8 Indemnification Claims.", "right_enumerator")

    def test_section_number_list_and_decimals_join(self) -> None:
        for left, right in (
            ("the conditions set forth in Sections 6.02, 6.03 and", "6.06 of this Agreement."),
            ("the conditions set forth in Section 7.2.1 or", "7.2.2 of this Agreement."),
            ("as provided in Section 5(a) and", "5.2 of this Agreement."),
            ("the Leverage Ratio is not greater than", "2.5 times the amount."),
        ):
            self.assertEqual(len(join_across_pages(_pages(left, right)).joins), 1, left)

    def test_dot_leaders_stay(self) -> None:
        self.assert_kept("Purchase Price and the rest of the items ........ of", "the Sellers", "left_dot_leader")
        self.assert_kept("the amount payable by the buyer", "at closing . . . . 12", "right_dot_leader")

    def test_capital_a_ending_an_index_line_is_not_an_open_word(self) -> None:
        self.assert_kept("the form of the Agreement in Annex A", "Definitions of the Terms", "right_not_continuation")

    def test_open_word_floor_counts_four_words(self) -> None:
        self.assertEqual(len(join_across_pages(_pages("sold to the", "Company Records.")).joins), 0)
        self.assertEqual(len(join_across_pages(_pages("assets sold to the", "Company Records.")).joins), 1)

    def test_caption_limit_is_ten_words(self) -> None:
        self.assert_kept("Purchase Price Adjustment And Working Capital Escrow Release Date Mechanics", "for the Sellers", "left_caption")
        left = "Purchase Price Adjustment And Working Capital Escrow Release Date Closing Mechanics"
        self.assertEqual(len(join_across_pages(_pages(left, "for the Sellers")).joins), 1)

    def test_right_label(self) -> None:
        self.assert_kept("the amount payable by the", "(continued)", "right_label")
        self.assert_kept("the amount payable by the", "page 12", "right_label")

    def test_capital_after_an_open_sentence_stays(self) -> None:
        self.assert_kept("the amount payable by Buyer", "Company shall pay it.", "right_not_continuation")
        self.assert_kept("of the", "Company shall pay it.", "right_not_continuation")

    def test_all_caps_opening_stays(self) -> None:
        self.assert_kept("in respect of the Limited Guarantee and the", "NOW, THEREFORE, the parties agree.", "right_caps")

    def test_pages_must_be_adjacent(self) -> None:
        pages = _pages("the amount payable by the", "buyer at closing.", orders=(3, 5))
        result = join_across_pages(pages)
        self.assertEqual(result.page_texts, ["the amount payable by the", "buyer at closing."])
        self.assertEqual(result.skipped["pages_not_adjacent"], 1)

    def test_unknown_order_stays(self) -> None:
        pages = [JoinPage(text="the amount payable by the"), JoinPage(text="buyer at closing.")]
        self.assertEqual(join_across_pages(pages).skipped["pages_unordered"], 1)

    def test_repeated_edge_line_stays(self) -> None:
        result = join_across_pages(
            _pages(
                "the amount payable by Buyer\n\nmerger agreement draft",
                "to the Sellers.\n\nmerger agreement draft",
                "to the Sellers again.",
            )
        )
        self.assertEqual(result.skipped["repeated_line"], 2)
        self.assertEqual(result.joins, [])

    def test_collect_lists_rejected_pairs(self) -> None:
        result = join_across_pages(_pages("The Buyer shall pay.", "the amount due."), collect=True)
        self.assertEqual(result.rejected[0]["reason"], "left_ends_sentence")
        self.assertEqual(join_across_pages(_pages("The Buyer shall pay.", "the amount due.")).rejected, [])


class HyphenTests(unittest.TestCase):
    def join(self, left: str, right: str, others: str = "") -> str:
        return join_across_pages(_pages(others + "\n\n" + left, right)).page_texts[0].split("\n\n")[-1]

    def test_closed_word_printed(self) -> None:
        self.assertEqual(self.join("subject to the con-", "sent of Buyer.", "consent consent"), "subject to the consent of Buyer.")

    def test_hyphenated_word_printed(self) -> None:
        self.assertEqual(self.join("a third-", "party claim.", "third-party"), "a third-party claim.")

    def test_more_often_printed_form_wins(self) -> None:
        self.assertEqual(self.join("fully paid and non-", "assessable shares.", "nonassessable non-assessable"), "fully paid and non-assessable shares.")

    def test_no_evidence_keeps_the_printed_hyphen(self) -> None:
        self.assertEqual(self.join("the post-", "employment benefits.", ""), "the post-employment benefits.")

    def test_number_before_the_hyphen(self) -> None:
        self.assertEqual(self.join("such twenty (20)-", "day period.", ""), "such twenty (20)-day period.")

    def test_suspended_pair_keeps_the_space(self) -> None:
        self.assertEqual(self.join("the pre-", "and post-closing periods.", ""), "the pre- and post-closing periods.")

    def test_compound_after_a_small_word_is_not_suspended(self) -> None:
        self.assertEqual(self.join("complete, accurate, up-", "to-date records.", ""), "complete, accurate, up-to-date records.")
        self.assertEqual(self.join("a customary stop-look-", "and-listen notice.", ""), "a customary stop-look-and-listen notice.")

    def test_page_label_at_the_end_of_the_line_stays(self) -> None:
        result = join_across_pages(_pages("four of whom -8-", "shall be designated by the Company."))
        self.assertEqual(result.joins, [])
        self.assertEqual(result.skipped["hyphen_page_label"], 1)

    def test_number_compound_joins(self) -> None:
        self.assertEqual(self.join("during the 12-", "month period.", ""), "during the 12-month period.")

    def test_uppercase_right_stays(self) -> None:
        result = join_across_pages(_pages("under the Non-", "Assignable Assets clause."))
        self.assertEqual(result.joins, [])

    def test_counts_cover_every_page_of_the_agreement(self) -> None:
        result = join_across_pages(_pages("the agreement requires con-", "sent of the buyer", "Their consent and consent."))
        self.assertEqual(result.page_texts[0], "the agreement requires consent of the buyer")

    def test_printed_closed_word_beats_the_suspended_words(self) -> None:
        result = join_across_pages(_pages("from here-", "to the parties. Hereto and hereto."))
        self.assertEqual(result.page_texts[0], "from hereto the parties. Hereto and hereto.")

    def test_page_label_glued_to_a_word_stays(self) -> None:
        for left in ("four of whom A-8-", "four of whom-8-"):
            self.assertEqual(join_across_pages(_pages(left, "shall be")).skipped["hyphen_page_label"], 1)

    def test_closing_punctuation_after_a_hyphen_stays(self) -> None:
        result = join_across_pages(_pages("the pre-", ", and the post-closing period."))
        self.assertEqual(result.skipped["hyphen_upper"], 1)

    def test_agreement_words_counts_hyphenated_words_whole(self) -> None:
        self.assertEqual(agreement_words(["third-party third Party"]), Counter({"third-party": 1, "third": 1, "party": 1}))


class RawEvidenceTests(unittest.TestCase):
    left = "the amount payable by the Purchaser under this clause"
    right = "to the Sellers as the case may be."

    def result(self, left_raw: str, right_raw: str):
        return join_across_pages(_pages(self.left, self.right, raws=(left_raw, right_raw)))

    def test_plain_paragraphs_join(self) -> None:
        result = self.result(_html(self.left), _html(self.right))
        self.assertEqual(len(result.joins), 1)

    def test_table_cell_stays(self) -> None:
        result = self.result(f"<table><tr><td>{self.left}</td></tr></table>", _html(self.right))
        self.assertEqual(result.skipped["raw_table"], 1)
        result = self.result(_html(self.left), f"<table><tr><td>{self.right}</td></tr></table>")
        self.assertEqual(result.skipped["raw_table"], 1)

    def test_table_cut_by_the_page_stays(self) -> None:
        result = self.result(_html(self.left), f"{self.right}</td></tr></table><p>Next.</p>")
        self.assertEqual(result.skipped["raw_table"], 1)

    def test_list_item_stays(self) -> None:
        result = self.result(_html(self.left), f"<ul><li>{self.right}</li></ul>")
        self.assertEqual(result.skipped["raw_list"], 1)

    def test_short_heading_element_stays(self) -> None:
        result = self.result(f"<h3>{self.left}</h3>", _html(self.right))
        self.assertEqual(result.skipped["raw_heading"], 1)

    def test_prose_in_a_heading_element_joins(self) -> None:
        long_left = "the amount payable by the Purchaser under this clause in the ordinary course and thereafter"
        result = join_across_pages(_pages(long_left, self.right + " and on each date after that one until paid", raws=(f"<h5>{long_left}</h5>", _html(self.right + " and on each date after that one until paid"))))
        self.assertEqual(len(result.joins), 1)

    def test_centered_block_stays(self) -> None:
        result = self.result(_html(self.left, style="text-align:center"), _html(self.right))
        self.assertEqual(result.skipped["raw_centered"], 1)
        result = self.result(f'<div align="center">{self.left}</div>', _html(self.right))
        self.assertEqual(result.skipped["raw_centered"], 1)

    def test_justified_paragraph_in_a_centered_wrapper_joins(self) -> None:
        result = self.result(f'<div style="text-align:center">{_html(self.left)}</div>', _html(self.right))
        self.assertEqual(len(result.joins), 1)

    def test_hidden_text_stays(self) -> None:
        result = self.result(f'<div style="display:none">{self.left}</div>', _html(self.right))
        self.assertEqual(result.skipped["raw_hidden"], 1)

    def test_overflow_hidden_is_not_hidden(self) -> None:
        result = self.result(f'<div style="overflow:hidden">{self.left}</div>', _html(self.right))
        self.assertEqual(len(result.joins), 1)

    def test_footnote_font_size_stays(self) -> None:
        left_raw = f'<p><span style="font-size:8pt">{self.left}</span></p>'
        right_raw = f'<p><span style="font-size:12pt">{self.right}</span></p>'
        self.assertEqual(self.result(left_raw, right_raw).skipped["raw_font_size"], 1)

    def test_equal_font_size_written_differently_joins(self) -> None:
        left_raw = f'<p><span style="font-size:10.0pt">{self.left}</span></p>'
        right_raw = f'<p><span style="font-size:10pt">{self.right}</span></p>'
        self.assertEqual(len(self.result(left_raw, right_raw).joins), 1)

    def test_font_attribute_sizes_differ(self) -> None:
        result = self.result(f'<p><font size="2">{self.left}</font></p>', f'<p><font size="1">{self.right}</font></p>')
        self.assertEqual(result.skipped["raw_font_size"], 1)

    def test_font_shorthand_sizes_differ(self) -> None:
        result = self.result(f'<p style="font: italic 8pt Times">{self.left}</p>', f'<p style="font: 12pt/14pt Times">{self.right}</p>')
        self.assertEqual(result.skipped["raw_font_size"], 1)

    def test_same_size_footnote_mark_opens_the_right_side(self) -> None:
        left = self.left + " that"
        for mark in ("<sup>1</sup>", '<span style="vertical-align: super">1</span>'):
            raws = (_html(left), _html(f"{mark} Please provide schedule of the items."))
            pages = _pages(left, "1 Please provide schedule of the items.\n\nAnd the rest of the note.", raws=raws)
            self.assertEqual(join_across_pages(pages).skipped["raw_superscript"], 1, mark)

    def test_footnote_ending_the_left_page_stays(self) -> None:
        left = "1 Please provide the schedule of all of the"
        pages = _pages(left, self.right, raws=(_html("<sup>1</sup> Please provide the schedule of all of the"), _html(self.right)))
        self.assertEqual(join_across_pages(pages).skipped["raw_superscript"], 1)

    def test_footnote_call_at_the_end_of_the_left_side_joins(self) -> None:
        left = "the amount payable by the Purchaser under this clause 1"
        pages = _pages(left, self.right, raws=(_html("the amount payable by the Purchaser under this clause<sup>1</sup>"), _html(self.right)))
        self.assertEqual(len(join_across_pages(pages).joins), 1)

    def test_later_printing_of_the_left_side_decides(self) -> None:
        raw = f"<table><tr><td>{self.left}</td></tr></table>" + _html(self.left)
        self.assertEqual(len(self.result(raw, _html(self.right)).joins), 1)

    def test_earlier_printing_of_the_right_side_decides(self) -> None:
        raw = _html(self.right) + f"<table><tr><td>{self.right}</td></tr></table>"
        self.assertEqual(len(self.result(_html(self.left), raw).joins), 1)

    def test_left_side_is_read_at_its_last_letter(self) -> None:
        left_raw = (
            '<p><span style="font-size:8pt">the amount payable by the Purchaser</span>'
            ' <span style="font-size:10pt">under this clause</span></p>'
        )
        right_raw = f'<p><span style="font-size:10pt">{self.right}</span></p>'
        self.assertEqual(len(self.result(left_raw, right_raw).joins), 1)

    def test_text_not_on_the_filed_page_stays(self) -> None:
        self.assertEqual(self.result(_html("something else entirely"), _html(self.right)).skipped["raw_unlocated"], 1)

    def test_html_page_without_raw_stays(self) -> None:
        pages = [JoinPage(self.left, None, True, 0), JoinPage(self.right, None, True, 1)]
        self.assertEqual(join_across_pages(pages).skipped["raw_missing"], 1)

    def test_comment_and_script_text_are_not_read(self) -> None:
        page = read_raw_page("<!-- hidden words --><script>var a = 1;</script><p>Printed</p>")
        self.assertEqual(page.alnum, "Printed")


class GenerateXmlTests(unittest.TestCase):
    def build(self, pages: list[str], raws: list[str] | None = None) -> str:
        df = pd.DataFrame(
            {
                "agreement_uuid": ["a"] * len(pages),
                "page_uuid": [f"p{i}" for i in range(len(pages))],
                "page_order": list(range(len(pages))),
                "raw_page_content": raws or [None] * len(pages),
                "source_page_type": ["body"] * len(pages),
                "tagged_output": pages,
                "url": ["http://x"] * len(pages),
                "filing_date": [date(2020, 1, 1)] * len(pages),
                "source_is_txt": [raws is None] * len(pages),
                "source_is_html": [raws is not None] * len(pages),
            }
        )
        xml, failures = generate_xml(df, {})
        self.assertEqual(failures, [])
        return xml[0].xml

    def test_marker_follows_the_joined_paragraph(self) -> None:
        xml = self.build(
            [
                "<article>ARTICLE I</article>\n\n<section>1.1 Sale.</section> the amount payable by the Purchaser\n\n<page>5</page>",
                "to the Sellers at closing.\n\n<section>1.2 Closing.</section> The Closing occurs.\n\n<page>6</page>",
            ]
        )
        self.assertIn("the amount payable by the Purchaser to the Sellers at closing.</text>", xml)
        self.assertLess(xml.index("to the Sellers at closing."), xml.index("<page>5</page>"))
        self.assertLess(xml.index("<pageUUID>p0</pageUUID>"), xml.index('title="1.2 Closing."'))
        self.assertIn('title="1.2 Closing."', xml)
        self.assertRegex(xml, r'title="1\.2 Closing\."[^>]*pageUUID="p1"')

    def test_html_rows_join_on_raw_evidence(self) -> None:
        left, right = "the amount payable by the Purchaser under this", "clause to the Sellers at closing."
        heading = "<article>ARTICLE I</article>\n\n<section>1.1 Sale.</section> "
        xml = self.build([heading + left, right], [_html("ARTICLE I", "1.1 Sale.", left), _html(right)])
        self.assertIn(f"{left} {right}", xml)

    def test_html_rows_reach_the_raw_guards(self) -> None:
        left, right = "the amount payable by the Purchaser under this", "clause to the Sellers at closing."
        heading = "<article>ARTICLE I</article>\n\n<section>1.1 Sale.</section> "
        xml = self.build([heading + left, right], [_html("ARTICLE I", "1.1 Sale.") + f"<table><tr><td>{left}</td></tr></table>", _html(right)])
        self.assertNotIn(f"{left} {right}", xml)

    def test_section_ids_do_not_move(self) -> None:
        tail = "\n\n<section>1.2 Closing.</section> The Closing occurs."
        joined = self.build(["<article>ARTICLE I</article>\n\n<section>1.1 Sale.</section> the amount payable by the", "buyer at closing." + tail])
        apart = self.build(["<article>ARTICLE I</article>\n\n<section>1.1 Sale.</section> the amount payable by the.", "Buyer at closing." + tail])
        self.assertEqual(
            [line for line in joined.splitlines() if "<section" in line or "<article" in line],
            [line for line in apart.splitlines() if "<section" in line or "<article" in line],
        )


if __name__ == "__main__":
    unittest.main()
