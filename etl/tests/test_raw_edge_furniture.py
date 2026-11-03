# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
"""Furniture that the filed page prints at its edges (strip_raw_edge_furniture).

The page edges in these tests are copied from served agreements: f30d4090 ("-77-"),
5101d349 ("102 <page>102</page>"), 732c4cf3 ("41."), a427b951 ("|"), bfd9b770
(". 50"), 74e54b39 ("- 12"), 58b61cab (roman body pages), c3e63836 ("Sale and
Purchase Agreement"), d6645b1b (SEC 601 legend), d6528b01 ("Appendix A - Page 14"),
4bc81f2f ("DB1/ 124297106.28"), b48c3673 (.txt top counters), 38a21edb
("|US-DOCS\\140904150.6||"), 3c87786e ("- 66 -").
"""
from datetime import date
import unittest
import xml.etree.ElementTree as ET

import pandas as pd

from etl.domain import f_xml
from etl.domain.page_furniture import RawEdgePage, raw_page_lines, strip_raw_edge_furniture


def _html(*lines: str) -> str:
    return "\n".join(f'<p style="margin: 0">{line}</p>' for line in lines)


def _page(text: str, raw: str, order: int, zone: str = "body", is_html: bool = True) -> RawEdgePage:
    return RawEdgePage(text=text, raw=raw, is_html=is_html, page_order=order, zone=zone)


def _strip(pages: list[RawEdgePage]) -> list[str]:
    return strip_raw_edge_furniture(pages).page_texts


# Clause lines between a page's edges, so the top and bottom edge windows do not overlap.
FILLER = [f"({letter}) the Seller shall deliver the item described in clause {letter}." for letter in "abcde"]
BODY = "\n".join(FILLER)
RAW_BODY = _html(*FILLER)

SEC_LEGEND = (
    "CERTAIN CONFIDENTIAL INFORMATION CONTAINED IN THIS DOCUMENT, MARKED BY [***], HAS BEEN OMITTED BECAUSE "
    "IT IS NOT MATERIAL AND WOULD LIKELY CAUSE COMPETITIVE HARM TO THE COMPANY IF PUBLICLY DISCLOSED."
)


class RawPageLinesTests(unittest.TestCase):
    def test_html_blocks_and_breaks_end_lines_but_source_newlines_do_not(self) -> None:
        raw = '<div><p>any of the Company’s\nrepresentations</p><p style="x">41.</p>A<br/>B</div>'
        self.assertEqual([line.text for line in raw_page_lines(raw, True)], ["any of the Company’s representations", "41.", "A", "B"])

    def test_text_lines_are_whole_only_when_set_off_by_blank_lines(self) -> None:
        lines = raw_page_lines("   7\nor to be so licensed\n\n   - 7 -\n", False)
        self.assertEqual([(line.text, line.whole) for line in lines], [("7", False), ("or to be so licensed", False), ("- 7 -", True)])


class PageNumberTests(unittest.TestCase):
    def test_untagged_number_that_continues_the_tagged_numbers(self) -> None:
        # f30d4090 p79-80: the tagger wrapped "-76-" but not "-77-".
        pages = [
            _page("Each of the Parties ... such courts.\n-<page>76</page>-", _html("Each of the Parties ... such courts.", "-76-"), 79),
            _page("<section>Section 10.13 No Recourse.</section> All claims ...\n[Signature Page Follows]\n-77-",
                  _html("Section 10.13 No Recourse. All claims ...", "[Signature Page Follows]", "-77-"), 80),
        ]
        out = _strip(pages)
        self.assertEqual(out[1], "<section>Section 10.13 No Recourse.</section> All claims ...\n[Signature Page Follows]")
        self.assertEqual(out[0], pages[0].text)

    def test_number_printed_twice_beside_its_tag(self) -> None:
        # 5101d349 (.txt): the page number is printed at the left and the right.
        raw = "   102\n                                    102\n\n\n   (x) any reorganization of the Companies\n"
        [out] = _strip([_page("102 <page>102</page>\n(x) any reorganization of the Companies", raw, 102, is_html=False)])
        self.assertEqual(out, "<page>102</page>\n(x) any reorganization of the Companies")

    def test_stray_period_and_bar_beside_the_page_number(self) -> None:
        pages = [
            # 732c4cf3 p41: raw "41."
            _page("(a) During the Pre-Closing Period ...\n<page>41</page>.", _html("(a) During the Pre-Closing Period ...", "41."), 41),
            # bfd9b770 p59: raw ". 50"
            _page("(b) From and after the Closing ...\n. <page>50</page>", _html("(b) From and after the Closing ...", ". 50"), 59),
            # a427b951 p5: raw "2" then "|"
            _page("<section>Section 1.05. Articles.</section> At the Effective Time ...\n<page>2</page>\n|",
                  _html("Section 1.05. Articles. At the Effective Time ...", "2", "|"), 5),
        ]
        out = _strip(pages)
        self.assertEqual(out[0], "(a) During the Pre-Closing Period ...\n<page>41</page>")
        self.assertEqual(out[1], "(b) From and after the Closing ...\n<page>50</page>")
        self.assertEqual(out[2], "<section>Section 1.05. Articles.</section> At the Effective Time ...\n<page>2</page>")

    def test_one_sided_dash_numbers_chained_across_pages(self) -> None:
        # 74e54b39: "- 2", "- 3", "- 4" at the top of consecutive pages.
        pages = [
            _page(f"- {n}\n\"Term {n}\" has the meaning given to it in Section 3.1(j);",
                  _html(f"- {n}", f"\"Term {n}\" has the meaning given to it in Section 3.1(j);"), n + 2)
            for n in (2, 3, 4)
        ]
        self.assertTrue(all(not text.startswith("-") for text in _strip(pages)))

    def test_roman_body_page_numbers(self) -> None:
        # 58b61cab numbers its body pages in romans.
        pages = [_page(f"(xviii) Neither CVCY nor any of its Subsidiaries ...\n{r}", _html("(xviii) Neither CVCY ...", r), o)
                 for r, o in (("xlv", 45), ("xlvi", 46), ("xlvii", 47))]
        self.assertEqual(_strip(pages), ["(xviii) Neither CVCY nor any of its Subsidiaries ..."] * 3)

    def test_exhibit_page_labels_in_back_matter(self) -> None:
        # d6528b01: "Appendix A - Page 13" .. "15".
        pages = [_page(f"{BODY}\n“Term {n}” means ...\nAppendix A - Page {n}",
                       RAW_BODY + _html(f"“Term {n}” means ...", f"Appendix A - Page {n}"), 90 + n, "back_matter")
                 for n in (13, 14, 15)]
        self.assertEqual(_strip(pages), [f"{BODY}\n“Term {n}” means ..." for n in (13, 14, 15)])

    def test_toc_romans_and_page_column_headers(self) -> None:
        pages = [
            _page("TABLE OF CONTENTS\nPage\nARTICLE I DEFINITIONS 1\ni", _html("TABLE OF CONTENTS", "Page", "ARTICLE I DEFINITIONS 1", "i"), 2, "toc"),
            _page("TABLE OF CONTENTS (continued)\nPage\nARTICLE V COVENANTS 30\nii",
                  _html("TABLE OF CONTENTS (continued)", "Page", "ARTICLE V COVENANTS 30", "ii"), 3, "toc"),
        ]
        out = _strip(pages)
        self.assertEqual(out[0], "TABLE OF CONTENTS\nPage\nARTICLE I DEFINITIONS 1")
        self.assertEqual(out[1], "ARTICLE V COVENANTS 30")

    # --- what stays ---

    def test_lone_number_without_a_numbering_stays(self) -> None:
        # A table's last cell, a footnote marker at the top, an enumerator at the bottom.
        pages = [
            _page(f"{BODY}\nNumber of shares\n45", RAW_BODY + _html("Number of shares", "45"), 10),
            _page(f"1\ncarried over footnote text\n{BODY}", _html("1", "carried over footnote text") + RAW_BODY, 11),
            _page(f"{BODY}\nThe Company shall deliver the following:\n2.", RAW_BODY + _html("The Company shall deliver the following:", "2."), 12),
        ]
        self.assertEqual(_strip(pages), [page.text for page in pages])

    def test_number_above_an_unnumbered_heading_stays(self) -> None:
        pages = [_page(f"{n}\n<section>Miscellaneous</section>\ntext", _html(str(n), "Miscellaneous", "text"), n) for n in (12, 13, 14)]
        self.assertEqual(_strip(pages), [page.text for page in pages])

    def test_caption_and_lone_label_stay(self) -> None:
        pages = [
            _page("EXHIBIT C-1\nASSIGNMENT OF AGREEMENT", _html("EXHIBIT C-1", "ASSIGNMENT OF AGREEMENT"), 98, "back_matter"),
            _page("EXHIBIT C-2\nFORM OF RELEASE", _html("EXHIBIT C-2", "FORM OF RELEASE"), 99, "back_matter"),
            _page("- B-2 -", _html("- B-2 -"), 105, "back_matter"),  # a page is never emptied
        ]
        self.assertEqual(_strip(pages), [page.text for page in pages])


class RepeatedLineTests(unittest.TestCase):
    def _running(self, footer: str, count: int = 6, zone: str = "body") -> list[RawEdgePage]:
        return [
            _page(f"{BODY}\n“Term {n}” has the meaning set forth in Section 2.0{n}.\n{footer}\nPage <page>{n}</page> of 92",
                  RAW_BODY + _html(f"“Term {n}” has the meaning set forth in Section 2.0{n}.", footer, f"Page {n} of 92"), n + 3, zone)
            for n in range(1, count + 1)
        ]

    def test_running_header_above_the_page_counter(self) -> None:
        # c3e63836: "Sale and Purchase Agreement" over "Page 7 of 92" on every page.
        out = _strip(self._running("Sale and Purchase Agreement"))
        self.assertIn("Sale and Purchase Agreement", out[0])  # the first copy stays
        for n, text in enumerate(out[1:], start=2):
            self.assertNotIn("Sale and Purchase Agreement", text)
            self.assertIn(f"<page>{n}</page>", text)

    def test_legend_on_every_page(self) -> None:
        ends = ["(f) the Escrow Agent", "(g) the Buyer", "(h) any Seller", "(i) the Company", "(j) each Party", "(k) the Guarantor"]
        pages = [_page(f"{SEC_LEGEND}\n{BODY}\n{end}", _html(SEC_LEGEND) + RAW_BODY + _html(end), n, "back_matter")
                 for n, end in enumerate(ends, 1)]
        out = _strip(pages)
        self.assertEqual(out[0], pages[0].text)  # the first copy stays
        self.assertEqual(out[1:], [f"{BODY}\n{end}" for end in ends[1:]])

    def test_running_header_glued_to_the_last_clause(self) -> None:
        pages = self._running("Sale and Purchase Agreement")
        glued = pages[3].text.replace(".\nSale and Purchase Agreement", ". Sale and Purchase Agreement")
        pages[3] = _page(glued, pages[3].raw or "", pages[3].page_order)
        out = _strip(pages)
        self.assertEqual(out[3].split("\n")[-2], "“Term 4” has the meaning set forth in Section 2.04.")

    def test_document_id_glued_to_pages_filed_as_one_run(self) -> None:
        # 4bc81f2f: the filing has no line breaks; every page opens "DB1/ 124297106.28 <n>".
        texts = ["PURCHASE AGREEMENT This Purchase Agreement is made", "sufficiency of which are hereby acknowledged",
                 "and the Member Representative may agree", "Buyer shall pay the Closing Payment",
                 "(a) Each Seller is duly organized", "Each Party shall bear its own costs"]
        pages = [_page(f"DB1/ 124297106.28 {n} {text}", f"<div>DB1/ 124297106.28 {n} {text}</div>", n) for n, text in enumerate(texts, 1)]
        self.assertEqual(_strip(pages), texts)

    def test_txt_top_counter_fused_into_a_sentence(self) -> None:
        # b48c3673: "authority" ends one page, "7" heads the next, "or to be so licensed" follows.
        words = ["or", "and", "the", "such", "any", "each"]
        pages = [_page(f"{n} {word} to be so licensed or qualified\n<page>- {n} -</page>",
                       f"   {n}\n{word} to be so licensed or qualified\n\n   - {n} -\n", n, is_html=False)
                 for n, word in zip(range(5, 11), words)]
        self.assertEqual(_strip(pages), [f"{word} to be so licensed or qualified\n<page>- {n} -</page>"
                                         for n, word in zip(range(5, 11), words)])

    def test_footer_longer_than_the_glued_run_limit_stays_whole(self) -> None:
        # fa14a726: a 13-word footer glued to the text of pages filed as one run. The
        # 12-word prefix repeats, but "February 2025 N" goes on past it: no partial cut.
        clauses = ["(a) The Buyer", "(b) Each Seller", "(c) From the time", "(d) The parties", "(e) No party", "(f) Each party"]
        footer = "King & Wood Mallesons 72330758_14 Asset Sale and Purchase Agreement 17 February 2025"
        pages = [_page(f"{footer} {n} {clause} shall act.", f"<p>{footer} {n} {clause} shall act.</p>", n)
                 for n, clause in zip(range(29, 35), clauses)]
        self.assertEqual(_strip(pages), [page.text for page in pages])

    def test_page_tag_inside_a_glued_run_stays_and_the_run_goes(self) -> None:
        # fa0fd3d5: "<page>4</page> LEGAL\67828860\11" opens each page filed as one run.
        clauses = ["“Cash” means", "“Contract” means", "“Debt” means", "“Fund” means", "“Lien” means", "“Plan” means"]
        pages = [_page(f"<page>{n}</page> LEGAL\\67828860\\11 {clause} money.", f"<p>{n} LEGAL\\67828860\\11 {clause} money.</p>", n + 4)
                 for n, clause in zip(range(4, 10), clauses)]
        self.assertEqual(_strip(pages), [f"<page>{n}</page> {clause} money." for n, clause in zip(range(4, 10), clauses)])

    def test_whole_document_id_comes_off_when_exhibits_print_another_id(self) -> None:
        # 86d00953: body pages open "9 SK 39196 0005 12134921 v17", exhibit pages
        # "SK 39196 0005 12138149 v5"; the body ID goes whole, not just "9 SK".
        words = ["firmware", "software", "hardware", "data", "records", "files", "code", "systems"]
        pages = [_page(f"<page>{n}</page> SK 39196 0005 12134921 v17 {word} of the Company.",
                       f"<p>{n} SK 39196 0005 12134921 v17 {word} of the Company.</p>", n + 1)
                 for n, word in zip(range(2, 8), words)]
        pages += [_page(f"<page>{n}</page> SK 39196 0005 12138149 v5 {word} of the Buyer.",
                        f"<p>{n} SK 39196 0005 12138149 v5 {word} of the Buyer.</p>", n + 1, "back_matter")
                  for n, word in zip(range(8, 10), words[6:])]
        stripped = _strip(pages)
        self.assertEqual(stripped[:6], [f"<page>{n}</page> {word} of the Company." for n, word in zip(range(2, 8), words)])

    def test_index_page_reference_equal_to_the_tagged_page_number_stays(self) -> None:
        # 3487d2cc: an index of defined terms on page 52 ends "affiliates 52", and the
        # page number 52 is already its own tag; the filing prints "52" twice.
        terms = ["Agreement", "Board", "Closing", "Code", "affiliates", "Person"]
        pages = [_page(f"{term} means the {term.lower()} {n}\n<page>{n}</page>",
                       _html(f"{term} means the {term.lower()}", str(n), str(n)), n + 4)
                 for n, term in zip(range(48, 54), terms)]
        self.assertEqual(_strip(pages), [page.text for page in pages])

    def test_number_the_filing_prints_inside_the_sentence_stays(self) -> None:
        pages = [_page(f"{n} or to be so licensed\n<page>{n}</page>", f"{n} or to be so licensed\nmore text\n\n{n}\n", n, is_html=False)
                 for n in range(5, 11)]
        self.assertEqual(_strip(pages), [page.text for page in pages])

    def test_law_firm_document_id_on_two_pages(self) -> None:
        # 38a21edb: the ID is printed on the first body page and on a sig page only.
        pages = [
            _page("NOW, THEREFORE ...\n<page>1</page>\n|US-DOCS\\140904150.6||", _html("NOW, THEREFORE ...", "1", "|US-DOCS\\140904150.6||"), 5),
            _page("SELLERS\nBy: /s/ R. Clyde Parker, Jr.\n|US-DOCS\\140904150.6||", _html("SELLERS", "By: /s/ R. Clyde Parker, Jr.", "|US-DOCS\\140904150.6||"), 80, "sig"),
        ]
        out = _strip(pages)
        self.assertEqual(out, ["NOW, THEREFORE ...\n<page>1</page>", "SELLERS\nBy: /s/ R. Clyde Parker, Jr."])

    # --- what stays ---

    def test_signature_block_names_repeated_on_counterpart_pages_stay(self) -> None:
        pages = [_page(f"AMBION, INC.\nBy: /s/ Signer {n}\nName: Signer {n}", _html("AMBION, INC.", f"By: /s/ Signer {n}", f"Name: Signer {n}"), 60 + n, "sig")
                 for n in range(6)]
        self.assertEqual(_strip(pages), [page.text for page in pages])

    def test_numbered_signer_names_on_consecutive_pages_stay(self) -> None:
        # ef3dd3d1: "POLARIS SUB 1 INC." to "POLARIS SUB 5 INC." head pages 107 to 111.
        pages = [_page(f"POLARIS SUB {n} INC.\nBy: /s/ David Babin\nName: David Babin\nSignature Page to Agreement and Plan of Merger",
                       _html(f"POLARIS SUB {n} INC.", "By:", "/s/ David Babin", "Name:", "David Babin",
                             "Signature Page to Agreement and Plan of Merger"), 106 + n, "sig")
                 for n in range(1, 7)]
        self.assertEqual(_strip(pages), [page.text for page in pages])

    def test_signature_page_legend_and_enumerators_stay(self) -> None:
        pages = [_page(f"(b)\ntext {n}\n[Signature Page to Stock Purchase Agreement]",
                       _html("(b)", f"text {n}", "[Signature Page to Stock Purchase Agreement]"), n, "sig") for n in range(6)]
        self.assertEqual(_strip(pages), [page.text for page in pages])

    def test_article_headings_at_the_top_of_exhibit_pages_stay(self) -> None:
        pages = [_page(f"ARTICLE {n}\nSHAREHOLDERS\ntext", _html(f"ARTICLE {n}", "SHAREHOLDERS", "text"), 100 + n, "back_matter") for n in range(1, 8)]
        self.assertEqual(_strip(pages), [page.text for page in pages])

    def test_dated_table_rows_stay(self) -> None:
        # d6645b1b: a patent table whose rows end many pages.
        dates = ["07-15-2010", "01-30-2009", "11-11-2015", "04-26-2014", "03-02-2012", "09-09-2011"]
        pages = [_page(f"Patent table\n{d} ENDOSCOPIC ATTACHMENT DEVICE n/a Closed",
                       _html("Patent table", d, "ENDOSCOPIC ATTACHMENT DEVICE", "n/a", "Closed"), 150 + k, "back_matter")
                 for k, d in enumerate(dates)]
        self.assertEqual(_strip(pages), [page.text for page in pages])

    def test_sentence_wrapped_onto_a_repeated_line_stays(self) -> None:
        # 93c1be20: each annex placeholder page prints one sentence over two <br> lines.
        tail = "statement/prospectus on Form S-4 (File No. 288222) filed on July 9, 2025)"
        pages = [
            _page(f"{BODY}\n(See Annex {letter} to the proxy {tail}",
                  RAW_BODY + f"<p>(See Annex {letter} to the proxy<br/>{tail}</p>", 70 + n, "back_matter")
            for n, letter in enumerate("ABCDEFG")
        ]
        self.assertEqual(_strip(pages), [page.text for page in pages])

    def test_redaction_and_omission_placeholders_stay(self) -> None:
        pages = [_page(f"Schedule {n}\n{BODY}\n[REDACTED]", _html(f"Schedule {n}") + RAW_BODY + _html("[REDACTED]"), 70 + n, "back_matter")
                 for n in range(7)]
        self.assertEqual(_strip(pages), [page.text for page in pages])

    def test_amounts_and_headings_repeated_in_the_body_stay(self) -> None:
        pages = [_page(f"<section>Section {n}.1 Purchase Price.</section>\n(a) $1,000,000;", _html(f"Section {n}.1 Purchase Price.", "(a) $1,000,000;"), n)
                 for n in range(1, 8)]
        self.assertEqual(_strip(pages), [page.text for page in pages])


def _clause(n: int) -> str:
    """A body line that differs from page to page."""
    return f"({n}) The Seller shall deliver to the Buyer the {['deed', 'title', 'bill', 'note', 'lease', 'permit', 'policy', 'record', 'consent', 'release', 'notice', 'waiver'][n % 12]} numbered {n}."


class FollowUpTests(unittest.TestCase):
    """Round 9: legends past the window, remnants after a cut, and shapes the pass missed."""

    LEGEND = [
        "CONFIDENTIAL INFORMATION, MARKED BY BRACKETS AND ASTERISKS",
        "([***]), IN THIS EXHIBIT HAS BEEN OMITTED AND FILED SEPARATELY WITH",
        "THE SECURITIES AND EXCHANGE COMMISSION. CONFIDENTIAL",
        "TREATMENT HAS BEEN REQUESTED WITH RESPECT TO THIS OMITTED",
        "INFORMATION.",
    ]

    def test_a_legend_longer_than_the_window_goes_with_its_last_line(self) -> None:
        # 0cf0c20d: five filed lines; the window sees four, and "INFORMATION." stayed.
        pages = [_page("\n".join(self.LEGEND[3:] + [_clause(n), BODY, _clause(n + 20)]),
                       _html(*self.LEGEND, _clause(n)) + RAW_BODY + _html(_clause(n + 20)), n)
                 for n in range(10, 16)]
        out = _strip(pages)
        self.assertEqual(out[0], pages[0].text)  # the first copy of a line without digits stays
        self.assertEqual(out[1:], [f"{_clause(n)}\n{BODY}\n{_clause(n + 20)}" for n in range(11, 16)])

    def test_the_line_after_a_finished_legend_stays(self) -> None:
        legend = ["CONFIDENTIAL TREATMENT HAS BEEN REQUESTED FOR PORTIONS OF THIS", "EXHIBIT. OMITTED PORTIONS ARE MARKED [***]."]
        pages = [_page("\n".join(legend + ["EXHIBIT A", _clause(n)]), _html(*legend, "EXHIBIT A", _clause(n)), n, "back_matter")
                 for n in range(10, 16)]
        self.assertEqual(_strip(pages)[1:], [f"EXHIBIT A\n{_clause(n)}" for n in range(11, 16)])

    def test_a_legend_run_together_on_a_contents_page_is_not_split(self) -> None:
        # 0cf0c20d p5: the contents layout runs the legend's lines together.
        toc = _page("CONFIDENTIAL INFORMATION, MARKED BY BRACKETS AND ASTERISKS ([***]), IN THIS\n1.1 Definitions 1",
                    _html(*self.LEGEND, "1.1 Definitions", "1"), 3, "toc")
        body = [_page("\n".join(self.LEGEND[3:] + [_clause(n)]), _html(*self.LEGEND, _clause(n)), n) for n in range(10, 16)]
        self.assertEqual(_strip([toc] + body)[0], toc.text)

    def test_page_number_printed_before_a_document_id(self) -> None:
        # aa74da09: "© King & Wood Mallesons" / "Share Sale Agreement" / "57" / "30379804_3".
        pages = [_page(f"{_clause(n)}\n© King & Wood Mallesons Share Sale Agreement {n} 30379804_3",
                       _html(_clause(n), "© King & Wood Mallesons", "Share Sale Agreement", str(n), "30379804_3"), n + 4, "back_matter")
                 for n in range(57, 63)]
        self.assertEqual(_strip(pages), [_clause(n) for n in range(57, 63)])

    def test_label_and_page_number_on_two_filed_lines(self) -> None:
        # 084f7187: "Confidential" / "Page 3", run together by the tagger.
        pages = [_page(f"{_clause(n)}\nConfidential Page {n}", _html(_clause(n), "Confidential", f"Page {n}"), n + 80, "back_matter")
                 for n in range(2, 8)]
        self.assertEqual(_strip(pages), [_clause(n) for n in range(2, 8)])

    def test_page_no_column_header_on_two_lines(self) -> None:
        # 8594846e: "Page" / "No." heads each contents page.
        pages = [_page(f"Page\nNo.\n2.{n} Real Property {n}", _html("EXECUTION COPY", "Page", "No.", f"2.{n} Real Property", str(n)), n, "toc")
                 for n in range(2, 5)]
        self.assertEqual(_strip(pages), [f"2.{n} Real Property {n}" for n in range(2, 5)])

    def test_cut_that_would_leave_only_a_document_id_is_not_made(self) -> None:
        # 658630f7 p22: the page prints only "A05230281/1.15/02 Jul 2005 - 22 -".
        pages = [_page(f"{_clause(n)}\nA05230281/1.15/02 Jul 2005 - {n} -", _html(_clause(n), f"A05230281/1.15/02 Jul 2005 - {n} -"), n, "back_matter")
                 for n in range(17, 22)]
        lone = _page("A05230281/1.15/02 Jul 2005 - 22 -", _html("A05230281/1.15/02 Jul 2005 - 22 -"), 22, "back_matter")
        self.assertEqual(_strip(pages + [lone])[-1], lone.text)

    def test_misprinted_running_title_left_by_a_cut(self) -> None:
        # 4dda3f3e p26: "Refinery Assel Purchase Agreement" / "Page 20 of 63"; other pages print "Asset".
        pages = [_page(f"{_clause(n)}\n<page>{n}</page>", _html(_clause(n), "Refinery Asset Purchase Agreement", f"Page {n} of 63"), n + 6)
                 for n in range(14, 20)]
        typo = _page(f"{_clause(20)}\nRefinery Assel Purchase Agreement Page <page>20</page> of 63",
                     _html(_clause(20), "Refinery Assel Purchase Agreement", "Page 20 of 63"), 26)
        self.assertEqual(_strip(pages + [typo])[-1], f"{_clause(20)}\n<page>20</page>")

    def test_exhibit_running_footers_that_open_with_a_heading_word(self) -> None:
        # c3e4cc2c: "EXHIBIT 5.19" / "MUTUAL TERMINATION OF EMPLOYMENT AGREEMENT" / "PAGE 2".
        footer = ["EXHIBIT 5.19", "MUTUAL TERMINATION OF EMPLOYMENT AGREEMENT"]
        pages = [_page(f"{BODY}\n{_clause(n)}\nEXHIBIT 5.19 MUTUAL TERMINATION OF EMPLOYMENT AGREEMENT PAGE {n}",
                       RAW_BODY + _html(_clause(n), *footer, f"PAGE {n}"), n + 102, "back_matter") for n in range(1, 7)]
        self.assertEqual(_strip(pages), [f"{BODY}\n{_clause(n)}" for n in range(1, 7)])

    def test_unnumbered_schedule_footer_on_most_of_its_pages(self) -> None:
        # 98763580: "Schedule 1 Definitions" heads the schedule and foots each of its pages.
        first = _page(f"Schedule 1 Definitions\n1 Definitions\n{BODY}\n{_clause(0)}\nSchedule 1 Definitions",
                      _html("Schedule 1 Definitions", "1", "Definitions") + RAW_BODY + _html(_clause(0), "Schedule 1 Definitions"),
                      74, "back_matter")
        rest = [_page(f"{BODY}\n{_clause(n)}\nSchedule 1 Definitions", RAW_BODY + _html(_clause(n), "Schedule 1 Definitions"),
                      74 + n, "back_matter") for n in range(1, 12)]
        out = _strip([first] + rest)
        self.assertEqual(out[0].split("\n")[0], "Schedule 1 Definitions")  # the heading stays
        self.assertEqual(out[1:], [f"{BODY}\n{_clause(n)}" for n in range(1, 12)])

    def test_unnumbered_back_matter_line_on_few_pages_stays(self) -> None:
        pages = [_page(f"{_clause(n)}\nSchedule 1 Definitions", _html(_clause(n), "Schedule 1 Definitions"), 74 + n, "back_matter")
                 for n in range(1, 7)]
        self.assertEqual(_strip(pages), [page.text for page in pages])

    def test_contents_page_number_left_by_a_document_id_cut(self) -> None:
        # 7b305b77: "iv 1103593863\\14\\AMERICAS" on each contents page.
        pages = [_page(f"Schedule {chr(64 + n)} - Form {n}\n{roman} 1103593863\\14\\AMERICAS",
                       _html(f"Schedule {chr(64 + n)} - Form {n}", roman, "1103593863\\14\\AMERICAS"), n, "toc")
                 for n, roman in zip(range(1, 6), ["i", "ii", "iii", "iv", "v"])]
        self.assertEqual(_strip(pages), [f"Schedule {chr(64 + n)} - Form {n}" for n in range(1, 6)])

    def test_tag_on_the_page_total_leaves_the_line_as_filed(self) -> None:
        # b10f7961: "56 | 76" with the total tagged; removing the line would drop the real number.
        pages = [_page(f"{_clause(n)}\n{n} | <page>76</page>", _html(_clause(n), f"{n} | 76"), n - 1) for n in range(54, 60)]
        self.assertEqual(_strip(pages), [page.text for page in pages])

    def test_annex_page_labels_with_a_number_or_dashes(self) -> None:
        # 31ce40ab "Annex 1-6", ea75ff9e "Annex A -5-".
        for label in ("Annex 1-{n}", "Annex A -{n}-"):
            pages = [_page(f"{_clause(n)}\n{label.format(n=n)}", _html(_clause(n), label.format(n=n)), n + 82, "back_matter")
                     for n in range(2, 8)]
            self.assertEqual(_strip(pages), [_clause(n) for n in range(2, 8)], label)

    def test_legend_printed_only_on_the_pages_it_concerns(self) -> None:
        # 54234f59: "**" / "The appearance of a double asterisk denotes ..." on 36 of 240 pages.
        legend = "The appearance of a double asterisk denotes confidential information that has been omitted"
        pages = [_page(f"{_clause(n + 40)}\n{BODY}\n{_clause(n)}\n** {legend}" if n % 4 == 0 else f"{_clause(n + 40)}\n{BODY}\n{_clause(n)}",
                       _html(_clause(n + 40)) + RAW_BODY + (_html(_clause(n), "**", legend) if n % 4 == 0 else _html(_clause(n))), n)
                 for n in range(40)]
        out = _strip(pages)
        self.assertEqual(out[0], pages[0].text)  # the first copy of a line without digits stays
        self.assertEqual(out[1:], [f"{_clause(n + 40)}\n{BODY}\n{_clause(n)}" for n in range(1, 40)])

    def test_a_page_of_furniture_is_never_emptied(self) -> None:
        # adf3a88b p168: a page holding only "Confidential and Proprietary" and the SEC legend.
        legend = "CONFIDENTIAL TREATMENT HAS BEEN REQUESTED FOR PORTIONS OF THIS EXHIBIT"
        pages = [_page(f"{_clause(n)}\n{BODY}\n<page>{n}</page>\nConfidential and Proprietary\n{legend}",
                       _html(_clause(n)) + RAW_BODY + _html(str(n), "Confidential and Proprietary", legend), n + 5)
                 for n in range(15, 21)]
        empty = _page(f"Confidential and Proprietary\n{legend}", _html("163", "Confidential and Proprietary", legend), 168, "back_matter")
        out = _strip(pages + [empty])
        self.assertEqual(out[1:6], [f"{_clause(n)}\n{BODY}\n<page>{n}</page>" for n in range(16, 21)])
        self.assertEqual(out[6], empty.text)


class GenerateXmlRawEdgeFurnitureTests(unittest.TestCase):
    def test_served_xml_loses_the_page_number_and_keeps_its_structure(self) -> None:
        # 3c87786e: "- 66 -" closed 10.11 because the tagger did not wrap it.
        pages = [
            ("toc", "TABLE OF CONTENTS\nARTICLE I GENERAL 1\ni", _html("TABLE OF CONTENTS", "ARTICLE I GENERAL 1", "i")),
            ("toc", "TABLE OF CONTENTS (continued)\nARTICLE X MISCELLANEOUS 60\nii",
             _html("TABLE OF CONTENTS (continued)", "ARTICLE X MISCELLANEOUS 60", "ii")),
            ("body", f"<article>ARTICLE X MISCELLANEOUS</article>\n<section>10.10 Notices.</section>\n{BODY}\n<page>- 65 -</page>",
             _html("ARTICLE X MISCELLANEOUS", "10.10 Notices.") + RAW_BODY + _html("- 65 -")),
            ("body", f"<section>10.11 Counterparts</section>\n{BODY}\n- 66 -", _html("10.11", "Counterparts") + RAW_BODY + _html("- 66 -")),
        ]
        df = pd.DataFrame(
            [
                {
                    "agreement_uuid": "44444444-4444-4444-4444-444444444444",
                    "page_uuid": f"p-{i}",
                    "page_order": 2 + i,
                    "raw_page_content": raw,
                    "source_page_type": page_type,
                    "gold_label": None,
                    "tagged_output": text,
                    "url": "https://example.com/agreement",
                    "filing_date": date(2024, 1, 1),
                    "source_is_txt": False,
                    "source_is_html": False,
                }
                for i, (page_type, text, raw) in enumerate(pages)
            ]
        )
        df["source_is_html"] = True
        xml_rows, failures = f_xml.generate_xml(df)
        self.assertEqual(failures, [])
        root = ET.fromstring(xml_rows[0].xml)
        body_texts = [t.text for t in root.find("body").iter("text")]  # type: ignore[union-attr]
        self.assertNotIn("- 66 -", body_texts)
        self.assertEqual(body_texts[-1], FILLER[-1])
        self.assertEqual([s.get("title") for s in root.iter("section")], ["10.10 Notices.", "10.11 Counterparts"])
        self.assertEqual([p.text for p in root.iter("pageUUID")], [f"p-{i}" for i in range(4)])


if __name__ == "__main__":
    unittest.main()
