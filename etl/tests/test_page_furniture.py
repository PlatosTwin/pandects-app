# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
from datetime import date
import unittest
import xml.etree.ElementTree as ET

import pandas as pd

from etl.domain import f_xml
from etl.domain.page_furniture import furniture_kind, running_title_kind, strip_page_furniture


def _page(body: str, *, header: str | None = None, footer: str | None = None) -> str:
    lines: list[str] = []
    if header is not None:
        lines.append(header)
    lines.append(body)
    if footer is not None:
        lines.append(footer)
    return "\n".join(lines)


class FurnitureKindTests(unittest.TestCase):
    def test_matches_whole_line_furniture(self) -> None:
        cases = {
            "Table of Contents": "toc_link",
            "TABLE OF CONTENTS": "toc_link",
            "Back to Contents": "toc_link",
            "US-DOCS\\159043691.20": "doc_id",
            "|US-DOCS\\159043691.20||": "doc_id",
            "ACTIVE/123456789.12": "doc_id",
            "NY 12345678v3": "doc_id",
            "H:836720 2": "doc_id",
            "33067829.14": "doc_id",
            "62323062_2": "doc_id",
            "249717839 v15": "doc_id",
            "041945-0274-Active.21403067.2": "doc_id",
            "EXECUTION VERSION": "banner",
            "Execution Copy": "banner",
            "CONFIDENTIAL": "banner",
            "***Confidential Treatment Requested": "banner",
            "12": "page_number",
            "- 12 -": "page_number",
            "-12-": "page_number",
            "Page 3 of 80": "page_number",
            "6/125": "page_number",
            "A-12": "page_number",
            "Exhibit 2.1": "exhibit_label",
        }
        for line, kind in cases.items():
            with self.subTest(line=line):
                self.assertEqual(furniture_kind(line), kind)

    def test_rejects_lines_with_other_words_or_tags(self) -> None:
        for line in [
            "See the Table of Contents for details.",
            "Confidential Information",
            "Section 12",
            "(10)",
            "New York, NY 10036",
            "PA 19103",
            "USD 1000000",
            "EUR 2500000.00",
            "Exhibit A",
            "<section>Section 5.4 Confidentiality.</section>",
            "<page>12</page>",
            "12 Business Days",
        ]:
            with self.subTest(line=line):
                self.assertIsNone(furniture_kind(line))


class StripPageFurnitureTests(unittest.TestCase):
    def test_strips_repeated_edge_furniture(self) -> None:
        pages = [
            _page(f"Body text of page {i}.", header="Table of Contents", footer="US-DOCS\\12345678.9")
            for i in range(4)
        ]

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts, [f"Body text of page {i}." for i in range(4)])
        self.assertEqual(len(result.removed), 8)
        self.assertEqual({r.kind for r in result.removed}, {"toc_link", "doc_id"})

    def test_varying_digits_count_as_one_repeated_shape(self) -> None:
        pages = [_page(f"Text {i}.", footer=f"- {i + 1} -") for i in range(3)]

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts, [f"Text {i}." for i in range(3)])

    def test_keeps_furniture_seen_on_fewer_than_three_pages(self) -> None:
        pages = [
            _page("Text one.", header="EXECUTION VERSION"),
            _page("Text two.", header="EXECUTION VERSION"),
            _page("Text three."),
        ]

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts, pages)
        self.assertEqual(result.removed, [])

    def test_never_strips_mid_page_lines(self) -> None:
        middle = "\n".join(["Line one.", "Line two.", "Line three.", "CONFIDENTIAL", "Line five.", "Line six.", "Line seven."])
        pages = [_page(middle, header="CONFIDENTIAL") for _ in range(3)]

        result = strip_page_furniture(pages)

        for text in result.page_texts:
            self.assertEqual(text, middle)

    def test_page_marker_lines_do_not_use_up_the_edge_window(self) -> None:
        body = "\n".join(["Line one.", "Line two.", "Line three.", "Line four."])
        pages = [
            "\n".join([body, "<page>1</page>", "<page>2</page>", "<page>3</page>", "Table of Contents"])
            for _ in range(3)
        ]

        result = strip_page_furniture(pages)

        for text in result.page_texts:
            self.assertNotIn("Table of Contents", text)
            self.assertIn("<page>3</page>", text)

    def test_never_strips_inside_a_multiline_heading(self) -> None:
        pages = [
            "\n".join(["<article>ARTICLE 12", "12", "</article>", "Body text."]) for _ in range(3)
        ]

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts, pages)

    def test_never_changes_words_on_other_lines(self) -> None:
        pages = [
            "\n".join(["EXECUTION COPY", "", "  The Company shall    pay  the fee.  ", str(i + 1)]) for i in range(3)
        ]

        result = strip_page_furniture(pages)

        for text in result.page_texts:
            self.assertEqual(text, "\n  The Company shall    pay  the fee.  ")

    def test_numbered_clause_above_unnumbered_heading_is_kept(self) -> None:
        pages = [f"Body text {i}.\n{i + 1}" for i in range(5)]
        pages.append("12\n<section>Miscellaneous</section>\nThe parties agree as follows.")

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts[:5], [f"Body text {i}." for i in range(5)])
        self.assertEqual(result.page_texts[5], pages[5])

    def test_page_number_above_letter_numbered_heading_is_removed(self) -> None:
        pages = [f"- {i + 1} -\nBody text {i}." for i in range(5)]
        pages.append("- 6 -\n<section>B. Representations.</section> The parties agree.")

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts[5], "<section>B. Representations.</section> The parties agree.")

    def test_two_fitting_numbers_on_one_page_keep_the_clause_number(self) -> None:
        # Page index 11 is printed page 12; the agreement numbers its pages at the bottom.
        openers = {
            "S1": "12\nMISCELLANEOUS",
            "S2": "12\n<article>ARTICLE XII MISCELLANEOUS</article>",
            "S3": "12\n<section>12.1 Notices.</section>",
            "S4": "12\nThe parties further agree as follows.",
        }
        for name, opener in openers.items():
            with self.subTest(name):
                pages = [f"Body text {i}.\n{i + 1}" for i in range(11)]
                pages.append(f"{opener}\nMore body text.\n12")
                pages += [f"Body text {i}.\n{i + 1}" for i in range(12, 15)]

                result = strip_page_furniture(pages)

                self.assertEqual(result.page_texts[11], f"{opener}\nMore body text.")

    def test_two_fitting_numbers_on_one_page_top_numbered_agreement(self) -> None:
        pages = [f"{i + 1}\nBody text {i}." for i in range(11)]
        pages.append("12\nThe Company shall deliver:\n12")
        pages += [f"{i + 1}\nBody text {i}." for i in range(12, 15)]

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts[11], "The Company shall deliver:\n12")

    def test_two_fitting_numbers_with_no_dominant_edge_strips_neither(self) -> None:
        pages = ["Body 0.\n1", "2\nBody 1.", "Body 2.\n3", "4\nBody 3.", "5\nBody 4.\n5"]

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts[4], pages[4])

    def test_clause_number_at_top_of_page_without_page_number_is_kept(self) -> None:
        # S5: printed page 12 lost its bottom number; its top line is clause 12.
        pages = [f"Body text {i}.\n{i + 1}" for i in range(11)]
        pages.append("12\nMISCELLANEOUS\nThe parties further agree as follows.")
        pages += [f"Body text {i}.\n{i + 1}" for i in range(12, 15)]

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts[11], pages[11])

    def test_sparse_chain_is_not_a_page_sequence(self) -> None:
        # S9: three fitting numbers spread over 47 pages are not running page numbers.
        body = "Body text.\nMore text.\nEven more.\nLast line."
        pages = [body for _ in range(50)]
        for index in (2, 10, 48):
            pages[index] = f"{index + 1}\n{body}"

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts, pages)
        self.assertEqual(result.removed, [])

    def test_bare_numbers_must_keep_a_constant_gap_to_tagged_numbers(self) -> None:
        # Pages tagged 11-13 open with list items 3, 4, 6: the gap to the tag is not constant.
        pages = [f"{n}\nList item text.\nMore text.\nEven more.\n<page>{t}</page>" for n, t in ((3, 11), (4, 12), (6, 13))]

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts, pages)

    def test_glued_page_tags_are_not_tagged_page_numbers(self) -> None:
        # The tagger split a document ID into "<page>4893-</page>2596-7688v2" on two pages.
        pages = [f"Body text {i}.\nMore text.\nEven more.\n{i + 2}\n4893-2596-7688v2" for i in range(5)]
        for index in (1, 3):
            pages[index] = f"Body text {index}.\nMore text.\nEven more.\n{index + 2}\n<page>4893-</page>2596-7688v2"

        result = strip_page_furniture(pages)

        self.assertEqual([r.line for r in result.removed if r.kind == "page_number"], ["2", "3", "4", "5", "6"])

    def test_bare_numbers_with_a_constant_gap_to_tagged_numbers_are_removed(self) -> None:
        # Agreement page numbers 2-5 printed above filing page numbers 6-9.
        pages = [f"Body text {i}.\nMore text.\nEven more.\n{i + 2}\n<page>{i + 6}</page>" for i in range(4)]

        result = strip_page_furniture(pages)

        self.assertEqual([r.line for r in result.removed], ["2", "3", "4", "5"])

    def test_sparse_chain_counts_pages_with_tagged_numbers(self) -> None:
        pages = [f"Body text {i}.\nMore text.\nEven more.\nLast line.\n<page>{i + 1}</page>" for i in range(50)]
        for index in (2, 10, 48):
            pages[index] = f"{index + 1}\nBody text {index}.\nMore text.\nEven more.\nLast line."

        result = strip_page_furniture(pages)

        self.assertEqual([r.line for r in result.removed], ["3", "11", "49"])

    def test_number_off_the_page_sequence_is_kept(self) -> None:
        pages = [f"Body text {i}.\n{i + 1}" for i in range(5)]
        pages.append("Body text.\nNumber of Shares held by Holder:\n100")

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts[5], pages[5])

    def test_only_one_page_number_per_page_edge(self) -> None:
        pages = [f"Body text {i}.\n{i + 1}" for i in range(5)]
        pages.append("Body text.\nItems:\n4\n5\n6")

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts[5], "Body text.\nItems:\n4\n5")

    def test_list_numbers_off_sequence_in_numbered_agreement_are_kept(self) -> None:
        pages = [f"Body text {i}.\nfooter\n{i + 1}" for i in range(5)]
        pages.append("Body text.\nItems:\n1\n2\n3")

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts[5], pages[5])

    def test_list_numbers_without_page_sequence_are_kept(self) -> None:
        pages = ["Intro text.", "Items:\n1\n2\n3", "More text."]

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts, pages)
        self.assertEqual(result.removed, [])

    def test_constant_number_is_not_a_page_sequence(self) -> None:
        pages = [f"Body text {i}.\n100" for i in range(4)]

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts, pages)

    def test_banner_heading_after_clause_number_is_kept(self) -> None:
        pages = ["12.\nCONFIDENTIAL\nEach party shall keep the terms confidential." for _ in range(3)]

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts, pages)

    def test_banner_below_stripped_furniture_is_removed(self) -> None:
        pages = [f"Table of Contents\nCONFIDENTIAL\nBody text {i}." for i in range(3)]

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts, [f"Body text {i}." for i in range(3)])

    def test_numeric_ids_need_identical_text(self) -> None:
        amounts = ["250000.00", "300000.00", "410000.00"]
        pages = [f"The fee is set out below.\n{amount}" for amount in amounts]

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts, pages)

        same = [f"Body text {i}.\n33067829.14" for i in range(3)]
        self.assertEqual(strip_page_furniture(same).page_texts, [f"Body text {i}." for i in range(3)])

    def test_unmarked_ids_with_trailing_page_numbers_are_removed(self) -> None:
        for doc_id in ["51175158.1", "ASIA 34949044"]:
            with self.subTest(doc_id=doc_id):
                pages = [f"Body text {i}.\n{doc_id} {i + 2}" for i in range(3)]
                self.assertEqual(strip_page_furniture(pages).page_texts, [f"Body text {i}." for i in range(3)])

    def test_banner_above_a_tagged_page_number_line_is_outermost(self) -> None:
        pages = [f"Body text {i}.\nExecution Version\n\nPage <page>{i + 1}</page>" for i in range(3)]

        result = strip_page_furniture(pages)

        # The "Page" label also goes; the <page> tag stays.
        self.assertEqual(result.page_texts, [f"Body text {i}.\n\n<page>{i + 1}</page>" for i in range(3)])


class RunningTitleTests(unittest.TestCase):
    def test_titled_page_numbers_are_removed(self) -> None:
        # 1f67a120 shape: "ASSET PURCHASE AGREEMENT PAGE N", some numbers tagged as <page>.
        pages = ["ASSET PURCHASE AGREEMENT\nThis Asset Purchase Agreement is made today.\nBody."]
        for i in range(1, 10):
            number = f"<page>{i + 1}</page>" if i % 3 == 0 else str(i + 1)
            pages.append(f"ASSET PURCHASE AGREEMENT PAGE {number}\nBody text {i}.\nMore text.\nEven more.")

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts[0], pages[0])
        self.assertEqual(result.page_texts[1], "Body text 1.\nMore text.\nEven more.")
        self.assertEqual(result.page_texts[3], "<page>4</page>\nBody text 3.\nMore text.\nEven more.")
        self.assertEqual(len(result.removed), 9)
        self.assertEqual({r.kind for r in result.removed}, {"running_title"})

    def test_bare_running_title_is_removed_except_the_cover_copy(self) -> None:
        # 5a95c633 shape: "Share Sale Agreement" atop every page, "page <page>N</page>" below.
        pages = [f"Share Sale Agreement\nBody text {i}.\nMore text.\nEven more.\npage <page>{i + 1}</page>" for i in range(8)]

        result = strip_page_furniture(pages)

        # The "page" label goes from every page; the <page> tag stays.
        self.assertEqual(result.page_texts[0], "Share Sale Agreement\nBody text 0.\nMore text.\nEven more.\n<page>1</page>")
        for i in range(1, 8):
            self.assertEqual(result.page_texts[i], f"Body text {i}.\nMore text.\nEven more.\n<page>{i + 1}</page>")

    def test_title_on_too_few_pages_is_kept(self) -> None:
        pages = [f"Body text {i}.\nMore text.\nEven more.\nLast line." for i in range(12)]
        pages[0] = "Share Sale Agreement\n" + pages[0]
        for i in (3, 5, 7, 9):
            pages[i] = "Share Sale Agreement\n" + pages[i]

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts, pages)

    def test_title_equal_to_a_tagged_heading_is_kept(self) -> None:
        pages = [f"AGREEMENT AND PLAN OF MERGER\nBody text {i}.\nMore text.\nEven more." for i in range(8)]
        pages[4] = "AGREEMENT AND PLAN OF MERGER\n<article>AGREEMENT AND PLAN OF MERGER</article>\nBody text.\nMore."

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts, pages)

    def test_titled_numbers_off_the_page_sequence_are_kept(self) -> None:
        pages = [f"SHARE SALE AGREEMENT PAGE 1\nBody text {i}.\nMore text.\nEven more." for i in range(8)]

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts, pages)

    def test_title_not_printed_as_the_agreement_title_is_kept(self) -> None:
        pages = ["AGREEMENT AND PLAN OF MERGER\nThis Agreement and Plan of Merger is made today.\nBody."]
        pages += [f"Voting Agreement\nBody text {i}.\nMore text.\nEven more." for i in range(1, 8)]

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts, pages)

    def test_one_word_agreement_line_is_kept(self) -> None:
        pages = [f"Agreement\nmeans this agreement, as amended {i}.\nMore text.\nEven more." for i in range(8)]

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts, pages)
        self.assertIsNone(running_title_kind("Agreement"))

    def test_title_opening_the_preamble_on_page_two_is_kept(self) -> None:
        # Page 0 is a body-labeled cover/TOC; the real title opens page 1.
        pages = ["Share Sale Agreement\nContents\n1 Definitions\n2 Sale and purchase"]
        pages.append("Share Sale Agreement\nThis Agreement is made on 1 July 2014\nBetween the parties.\nBody.")
        pages += [f"Share Sale Agreement\nBody text {i}.\nMore text.\nEven more." for i in range(2, 9)]

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts[:2], pages[:2])
        self.assertEqual(result.page_texts[2], "Body text 2.\nMore text.\nEven more.")

    def test_exhibit_titled_like_the_agreement_is_kept(self) -> None:
        # A body exhibit whose title page names the agreement, followed by its own preamble.
        pages = [f"Body text {i}.\nMore text.\nEven more.\nAsset Purchase Agreement" for i in range(8)]
        pages[0] = "Asset Purchase Agreement\nThis Asset Purchase Agreement is dated today.\nBody.\nMore."
        pages[5] = "Asset Purchase Agreement\nby and between Buyer and Seller\nExhibit text.\nAsset Purchase Agreement"

        result = strip_page_furniture(pages)

        self.assertEqual(
            result.page_texts[5], "Asset Purchase Agreement\nby and between Buyer and Seller\nExhibit text."
        )
        self.assertEqual(result.page_texts[3], "Body text 3.\nMore text.\nEven more.")

    def test_header_above_a_lowercase_continuation_line_is_removed(self) -> None:
        # eb1d32ce: "this Agreement and any agreement ..." continues a sentence; it is not a preamble.
        pages = [f"Share Sale Agreement\nBody text {i}.\nMore text.\nEven more." for i in range(8)]
        pages[4] = "Share Sale Agreement\nthis Agreement and any related document.\nMore.\nEven more."

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts[4], "this Agreement and any related document.\nMore.\nEven more.")

    def test_title_off_the_dominant_edge_is_kept(self) -> None:
        pages = [f"Body text {i}.\nMore text.\nEven more.\nAsset Purchase Agreement" for i in range(8)]
        pages[0] = "Asset Purchase Agreement\nThe parties agree.\nBody.\nMore."
        pages[5] = "Asset Purchase Agreement\nExhibit text.\nMore text.\nEven more.\nAsset Purchase Agreement"

        result = strip_page_furniture(pages)

        self.assertEqual(
            result.page_texts[5], "Asset Purchase Agreement\nExhibit text.\nMore text.\nEven more."
        )

    def test_header_above_an_identical_untagged_heading_goes_alone(self) -> None:
        pages = [f"Share Sale Agreement\nBody text {i}.\nMore text.\nEven more." for i in range(8)]
        pages[4] = "Share Sale Agreement\nShare Sale Agreement\nThe heading above is a real heading.\nMore."

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts[4], "Share Sale Agreement\nThe heading above is a real heading.\nMore.")

    def test_firm_footer_with_page_numbers_is_removed(self) -> None:
        pages = [f"Body text {i}.\nMore text.\nEven more.\nDLA Piper | {i + 1}" for i in range(8)]

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts, [f"Body text {i}.\nMore text.\nEven more." for i in range(8)])

    def test_running_title_kind(self) -> None:
        cases = {
            "ASSET PURCHASE AGREEMENT PAGE 12": "titled_page_number",
            "ASSET PURCHASE AGREEMENT PAGE <page>12</page>": "titled_page_number",
            "Share Sale Agreement - 12": "titled_page_number",
            "Share Sale Agreement | 12": "titled_page_number",
            "12 Share Sale Agreement": "titled_page_number",
            "Share Sale Agreement": "title",
            "AGREEMENT AND PLAN OF MERGER": "title",
            "Agreement": None,
            "The Company shall pay the fee.": None,
            "Table of Contents": None,
            "<article>ARTICLE I</article>": None,
        }
        for line, kind in cases.items():
            with self.subTest(line=line):
                self.assertEqual(running_title_kind(line), kind)


class RepeatedLineTests(unittest.TestCase):
    @staticmethod
    def _pages(top: str | None, bottom: str | None, count: int = 8) -> list[str]:
        pages = []
        for i in range(count):
            lines = [f"Body text {i} begins here", f"middle line {i}", f"Body text {i} ends here"]
            if top is not None:
                lines.insert(0, top.format(n=i + 1))
            if bottom is not None:
                lines.append(bottom.format(n=i + 1))
            pages.append("\n".join(lines))
        return pages

    def test_page_label_word_goes_and_tag_stays(self) -> None:
        pages = self._pages(None, "Page <page>{n}</page>")

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts[2], "Body text 2 begins here\nmiddle line 2\nBody text 2 ends here\n<page>3</page>")
        self.assertEqual({r.kind for r in result.removed}, {"page_label"})

    def test_split_page_counter_keeps_only_the_tag(self) -> None:
        # fbdc39d9: "Page <page>2</page> of 23" left " of 23" in the served text.
        pages = self._pages(None, "Page <page>{n}</page> of 23")

        result = strip_page_furniture(pages)

        self.assertTrue(result.page_texts[4].endswith("\n<page>5</page>"))

    def test_partly_tagged_document_id_is_removed_whole(self) -> None:
        # 039890a0: the tagger read "4848-" of the ID 4848-1716-0445 as a page number.
        pages = self._pages(None, "<page>4848-</page>1716-0445")

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts[1], "Body text 1 begins here\nmiddle line 1\nBody text 1 ends here")

    def test_deal_name_header_and_doc_id_footer_are_removed(self) -> None:
        pages = self._pages("Project Beaker - SPA", "AMS6540051 168372-0004")

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts[3], "Body text 3 begins here\nmiddle line 3\nBody text 3 ends here")

    def test_line_on_fewer_than_half_the_pages_is_kept(self) -> None:
        pages = self._pages(None, None, count=12)
        for i in range(5):
            pages[i] += "\nProject Beaker - SPA"

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts, pages)

    def test_repeated_body_sentence_is_kept(self) -> None:
        pages = self._pages(None, "The parties agree to the foregoing.")

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts, pages)

    def test_repeated_legend_sentence_is_removed(self) -> None:
        pages = self._pages(None, "*** Confidential Treatment Requested.")

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts[1], "Body text 1 begins here\nmiddle line 1\nBody text 1 ends here")

    def test_single_words_rules_and_blanks_are_kept(self) -> None:
        for line in ["each", "Agreement", "----------", "[______]"]:
            with self.subTest(line=line):
                pages = self._pages(line, None)
                self.assertEqual(strip_page_furniture(pages).page_texts, pages)

    def test_back_link_word_is_removed(self) -> None:
        pages = self._pages("BACK", None)

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts[1], "Body text 1 begins here\nmiddle line 1\nBody text 1 ends here")

    def test_repeated_line_equal_to_a_tagged_heading_is_kept(self) -> None:
        pages = self._pages("Project Horizon", None)
        pages[3] = "Project Horizon\n<article>Project Horizon</article>\nBody text."

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts, pages)

    def test_repeated_line_inside_the_page_is_kept(self) -> None:
        pages = [f"Top {i}\nNext {i}\nThird {i}\nProject Beaker - SPA\nFifth {i}\nSixth {i}\nSeventh {i}" for i in range(8)]

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts, pages)


    def test_schedule_header_and_identical_row_keep_their_first_copy(self) -> None:
        # An embedded schedule after the body: the header and a row repeat on
        # every continuation page, and the schedule's first page is real text.
        pages = [f"Body {i} begins\nBody {i} middle\nBody {i} ends" for i in range(2)] + [
            f"Holder Number of Shares\nAcme Holdings LLC 1,000\nBeta Partners LP {i},500\nGamma Fund {i}\nRow end {i}"
            for i in range(8)
        ]

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts[2], pages[2])

    def test_line_below_two_running_lines_does_not_cascade(self) -> None:
        pages = [
            f"Project Beaker - SPA\nSeller Disclosure Letter v2\nAcme Holdings LLC 1,000\nBody {i}\nMore {i}\nEnd {i}"
            for i in range(8)
        ]

        result = strip_page_furniture(pages)

        self.assertTrue(all("Acme Holdings LLC 1,000" in text for text in result.page_texts))

    def test_three_line_legend_is_removed_whole(self) -> None:
        # 0d8e4ef9: the legend's third line sits under two stripped legend lines.
        legend = "PORTIONS OF THIS DOCUMENT MARKED BY [***] HAVE BEEN\nOMITTED PURSUANT TO A REQUEST FOR CONFIDENTIAL TREATMENT\nCONFIDENTIAL PORTIONS FILED SEPARATELY WITH THE COMMISSION"
        pages = self._pages(legend, None)

        result = strip_page_furniture(pages)

        self.assertEqual(result.page_texts[1], "Body text 1 begins here\nmiddle line 1\nBody text 1 ends here")

    def test_structural_repeated_lines_are_kept(self) -> None:
        cases = [
            ("(continued)", None),
            ("Schedule 3.1 (continued)", None),
            ("(a)", None),
            ("Exhibit A \u2013 Form of Joinder Agreement", None),
            (None, "By: ____________________"),
            (None, "Name: John Smith"),
            (None, "/s/ John Smith"),
            (None, "[Signature Page Follows]"),
            (None, "[Remainder of page intentionally left blank]"),
            (None, "This page intentionally left blank"),
        ]
        for top, bottom in cases:
            with self.subTest(line=top or bottom):
                pages = self._pages(top, bottom)
                self.assertEqual(strip_page_furniture(pages).page_texts, pages)

class GenerateXmlStripsFurnitureTests(unittest.TestCase):
    def test_section_text_has_no_running_headers(self) -> None:
        agreement_uuid = "22222222-2222-2222-2222-222222222222"
        texts = [
            "Table of Contents\n<article>ARTICLE I GENERAL</article>\n<section>1.1 Notices.</section>\nAll notices shall be in writing\nNY 12345678v3",
            "Table of Contents\nand delivered by hand.\nNY 12345678v3",
            "Table of Contents\nNotices are effective on receipt.\nNY 12345678v3",
        ]
        df = pd.DataFrame(
            [
                {
                    "agreement_uuid": agreement_uuid,
                    "page_uuid": f"body-{i}",
                    "page_order": i,
                    "raw_page_content": "",
                    "source_page_type": "body",
                    "gold_label": None,
                    "tagged_output": t,
                    "url": "https://example.com/agreement",
                    "filing_date": date(2024, 1, 1),
                    "source_is_txt": True,
                    "source_is_html": False,
                }
                for i, t in enumerate(texts)
            ]
        )

        xml_rows, failures = f_xml.generate_xml(df)

        self.assertEqual(failures, [])
        root = ET.fromstring(xml_rows[0].xml)
        section = root.find(".//section")
        assert section is not None
        section_text = [t.text for t in section.iter("text")]
        self.assertEqual(
            section_text,
            ["All notices shall be in writing", "and delivered by hand.", "Notices are effective on receipt."],
        )
        self.assertEqual([p.text for p in section.iter("pageUUID")], ["body-0", "body-1", "body-2"])
        self.assertEqual([p.text for p in root.iter("pageUUID")], ["body-0", "body-1", "body-2"])
        self.assertEqual(root.find(".//section").attrib.get("pageUUID"), "body-0")

    def test_page_uuid_order_and_count_unchanged_by_strip(self) -> None:
        agreement_uuid = "33333333-3333-3333-3333-333333333333"
        texts = ["<article>ARTICLE I GENERAL</article>\n<section>1.1 Terms.</section>\nText one.\n1"]
        texts += [f"Text {i}.\n{i}" for i in range(2, 7)]
        texts.append("Table of Contents")  # one-off, so it is kept
        rows = [
            {
                "agreement_uuid": agreement_uuid,
                "page_uuid": f"p-{i:02d}",
                "page_order": i,
                "raw_page_content": "",
                "source_page_type": "body",
                "gold_label": None,
                "tagged_output": t,
                "url": "https://example.com/agreement",
                "filing_date": date(2024, 1, 1),
                "source_is_txt": True,
                "source_is_html": False,
            }
            for i, t in enumerate(texts)
        ]
        df = pd.DataFrame(rows)

        xml_rows, failures = f_xml.generate_xml(df)

        self.assertEqual(failures, [])
        root = ET.fromstring(xml_rows[0].xml)
        self.assertEqual([p.text for p in root.iter("pageUUID")], [f"p-{i:02d}" for i in range(len(texts))])
        body_text = [t.text for t in root.iter("text")]
        self.assertNotIn("1", body_text)
        self.assertNotIn("6", body_text)
        self.assertIn("Table of Contents", body_text)


if __name__ == "__main__":
    unittest.main()
