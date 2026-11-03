# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
"""The served TOC keeps the printed text around its entry tables.

The TOC page formatter kept only the tables that hold section entries, so the exhibit and
schedule lists and index headings printed after the last entry were dropped (6ce7df4b,
67dd911e, 96d7f4ea, 01439258), and so were article headings printed between entry tables
(95318392). Page numbers, TOC labels, doc IDs, running titles and filing legends stay out.
"""
import unittest

from etl.domain.f_xml import format_toc_html_like_screen, repeated_toc_extra_keys


def _entry_table(*rows: tuple[str, str, str]) -> str:
    cells = "".join(
        f"<tr><td><p>{number}</p></td><td><p>{title}</p></td><td><p align='right'>{page}</p></td></tr>"
        for number, title, page in rows
    )
    return f"<table>{cells}</table>"


ENTRIES_11 = _entry_table(
    ("Section 11.12", "Jurisdiction; Consent to Jurisdiction", "94"),
    ("Section 11.13", "Specific Performance", "95"),
    ("Section 11.14", "Certain Lender Agreements", "95"),
)


def _lines(html: str) -> list[str]:
    return [line.strip() for line in format_toc_html_like_screen(html).splitlines() if line.strip()]


def _lines_with(html: str, drop_keys: frozenset[str]) -> list[str]:
    text = format_toc_html_like_screen(html, drop_keys=drop_keys)
    return [line.strip() for line in text.splitlines() if line.strip()]


class TocPrintedTailTests(unittest.TestCase):
    def test_exhibit_list_after_the_last_entry_is_kept(self) -> None:
        # 6ce7df4b page 5: the exhibit and annex list after the entry table, then the page number.
        html = (
            "<p align='center'><b>TABLE OF CONTENTS</b></p><p align='center'>(continued)</p>"
            "<div>" + ENTRIES_11 + "<p><font> </font></p><p><b><font>EXHIBITS</font></b></p>"
            "<p><font>Exhibit A — Separation Agreement</font></p>"
            "<p><font>Annex A — Certificate of Incorporation of the Surviving Corporation</font></p>"
            "<p><font>Annex B — Bylaws of the Surviving Corporation</font></p>"
            "<p align='center'><font>v</font></p></div>"
        )
        lines = _lines(html)
        self.assertEqual(
            lines[-4:],
            [
                "EXHIBITS",
                "Exhibit A — Separation Agreement",
                "Annex A — Certificate of Incorporation of the Surviving Corporation",
                "Annex B — Bylaws of the Surviving Corporation",
            ],
        )
        self.assertTrue(lines[-5].startswith("Section 11.14"))
        self.assertNotIn("v", lines)

    def test_exhibit_table_after_the_entry_table_is_kept(self) -> None:
        # 96d7f4ea page 3: "LIST OF EXHIBITS" and a two-column exhibit table.
        html = (
            ENTRIES_11
            + "<p align='center'><b>LIST OF EXHIBITS</b></p>"
            + "<table><tr><td>Exhibit A</td><td>Definitions</td></tr>"
            + "<tr><td>Exhibit B</td><td>Required Governmental Authorizations</td></tr></table>"
            + "<p align='center'>iii</p>"
        )
        lines = _lines(html)
        self.assertEqual(lines[-3:], ["LIST OF EXHIBITS", "Exhibit A Definitions", "Exhibit B Required Governmental Authorizations"])

    def test_article_headings_between_entry_tables_are_kept(self) -> None:
        # 95318392 page 1: the article headings sit in divs between per-article tables.
        html = (
            "<div>Table of Contents</div><div>Page</div>"
            "<div><div>ARTICLE I<br/></div><div> Definitions</div></div>"
            + _entry_table(("Section 1.1", "Certain Defined Terms.", "1"), ("Section 1.2", "Other Interpretive Provisions.", "17"),
                           ("Section 1.3", "Rules of Construction.", "17"))
            + "<div> </div><div><div>ARTICLE II<br/></div><div> Purchase, Sale, Assignment and Assumption</div></div>"
            + _entry_table(("Section 2.1", "Purchase and Sale of Sold Interests.", "17"),
                           ("Section 2.2", "Purchase and Sale of the Purchased Assets.", "17"),
                           ("Section 2.3", "Excluded Assets of the Business.", "19"))
            + "<p align='center'>i</p>"
        )
        lines = _lines(html)
        start = lines.index("ARTICLE I")
        self.assertEqual(lines[start + 1], "Definitions")
        self.assertTrue(lines[start + 2].startswith("Section 1.1"))
        middle = lines.index("ARTICLE II")
        self.assertTrue(lines[middle - 1].startswith("Section 1.3"))
        self.assertEqual(lines[middle + 1], "Purchase, Sale, Assignment and Assumption")
        self.assertTrue(lines[middle + 2].startswith("Section 2.1"))
        self.assertNotIn("i", lines)

    def test_furniture_around_the_entries_stays_out(self) -> None:
        html = (
            "<p>AGREEMENT AND PLAN OF MERGER</p><p>by and among</p>"  # cover text before the entries
            + ENTRIES_11
            + "<p>- iv -</p><p>#25932618 v26</p><p>328972.00001/116443569.20</p>"
            + "<p>MinterEllison | Ref: SLRB:SDYC 1136588 Page 3</p><p>Share sale agreement</p>"
            + "<p>TABLE OF CONTENTS (continued)</p><p>Page</p><p>Section Description</p><p>ii.</p>"
            + "<p>#4854-0942-8064v43</p><p>A-ii</p><p>()</p><p>________</p><p>2 84</p>"
            + '<p>IF " DOCVARIABLE "SWDocIDLocation" 1" = "1" " DOCPROPERTY "SWDocID" SL1 3716757v12 101000.00058" ""</p>'
            + "<p>1 The Table of Contents is not a part of this Agreement.</p>"
            + "<p>Page | ii</p><p>PAGE || iii</p><p>Page ii of iv</p><p>(CONTINUED ON NEXT PAGE)</p>"
            + "<p>[Remainder of page intentionally left blank]</p>"
            + "<p>EXHIBITS</p><p>Exhibit A Form of Voting Agreement</p><p>B Mid-Tier Merger Agreement</p>"
            + "<p>Confidential and Proprietary</p><p>Annex A Closing Steps</p>"
        )
        lines = _lines(html)
        # A legend line stays out; the list resumes at the next listed item.
        self.assertEqual(
            lines[-4:],
            ["EXHIBITS", "Exhibit A Form of Voting Agreement", "B Mid-Tier Merger Agreement", "Annex A Closing Steps"],
        )
        for furniture in (
            "AGREEMENT AND PLAN OF MERGER", "by and among", "- iv -", "#25932618 v26", "328972.00001/116443569.20",
            "Share sale agreement", "Section Description", "ii.", "Confidential and Proprietary",
            "#4854-0942-8064v43", "A-ii", "()", "________", "2 84", "1 The Table of Contents is not a part of this Agreement.",
            "Page | ii", "PAGE || iii", "Page ii of iv", "(CONTINUED ON NEXT PAGE)",
            "[Remainder of page intentionally left blank]",
        ):
            with self.subTest(furniture=furniture):
                self.assertNotIn(furniture, lines)
        self.assertFalse(any("MinterEllison" in line for line in lines))

    def test_a_filing_legend_ends_the_tail(self) -> None:
        # b0c7daba, dabe9dc3: a confidential-treatment note and its continuation lines.
        html = (
            ENTRIES_11
            + "<p>Exhibit A Form of Escrow Agreement</p>"
            + "<p>*The Schedules listed above have been omitted from this filing. The Company hereby agrees to provide the Commission,</p>"
            + "<p>upon request, copies of any omitted exhibits or schedules to this exhibit required by Item 601 (b)(2) of Regulation</p>"
            + "<p>S-K.</p>"
        )
        lines = _lines(html)
        self.assertEqual(lines[-1], "Exhibit A Form of Escrow Agreement")
        self.assertNotIn("S-K.", lines)

    def test_prose_after_the_entries_ends_the_tail(self) -> None:
        prose = "This AGREEMENT AND PLAN OF MERGER, dated as of May 1, 2024, is entered into by and among " * 4
        html = ENTRIES_11 + "<p>EXHIBITS</p><p>Exhibit A Form of Escrow Agreement</p>" + f"<p>{prose}</p><p>Exhibit B Not here</p>"
        lines = _lines(html)
        self.assertEqual(lines[-2:], ["EXHIBITS", "Exhibit A Form of Escrow Agreement"])

    def test_running_footers_repeated_on_toc_pages_stay_out(self) -> None:
        # 8fffafb6, 522db544, 209d338f: a footer printed on every TOC page, some with the page number.
        footer = "<p>Purchase and Sale Agreement \u2013 WCG</p><p>Hamilton Locke \u258c Share Purchase Agreement {}</p>"
        pages = [
            ENTRIES_11 + "<p>EXHIBITS</p><p>Exhibit A Form of Escrow Agreement</p>" + footer.format("i"),
            ENTRIES_11 + "<p>Exhibit B Form of Guaranty</p><p>Annex A Closing Steps</p>" + footer.format("ii"),
        ]
        keys = repeated_toc_extra_keys(pages)
        self.assertEqual(_lines_with(pages[0], keys)[-2:], ["EXHIBITS", "Exhibit A Form of Escrow Agreement"])
        self.assertEqual(_lines_with(pages[1], keys)[-2:], ["Exhibit B Form of Guaranty", "Annex A Closing Steps"])
        # On a single page nothing repeats, so nothing is dropped for repetition.
        self.assertEqual(repeated_toc_extra_keys(pages[:1]), frozenset())

    def test_execution_banner_and_copyright_lines_stay_out(self) -> None:
        html = (
            ENTRIES_11 + "<p>EXHIBITS</p><p>Exhibit A Form of Escrow Agreement</p>"
            + "<p>AGREEMENT AND PLAN OF MERGER BETWEEN A, INC. AND B, INC. EXECUTION VERSION</p><p>i .</p>"
            + "<p>Herbert Smith Freehills owns the copyright in this document</p><p>Restricted circulation</p>"
            + "<p>Exhibit B Form of Guaranty</p>"
        )
        lines = _lines(html)
        self.assertEqual(lines[-3:], ["EXHIBITS", "Exhibit A Form of Escrow Agreement", "Exhibit B Form of Guaranty"])
        self.assertNotIn("Restricted circulation", lines)

    def test_doc_id_footers_with_a_page_number_stay_out(self) -> None:
        # 74b1ab4f, c0b21bbb, e3f224a9: a doc ID with a roman page number; a5ec15f8: a slashed doc ID.
        for footer in (
            "<table><tr><td>ASIA 34949044</td><td>iii</td></tr></table>",
            "<p>EAST\\134549672.25 vii</p>",
            "<p>Active 37496756.2 iii</p>",
            "<p>HBM/HBM/411669/3/UKM/108442647.1</p>",
        ):
            with self.subTest(footer=footer):
                html = ENTRIES_11 + "<p>EXHIBITS</p><p>Exhibit A Form of Escrow Agreement</p>" + footer
                self.assertEqual(_lines(html)[-2:], ["EXHIBITS", "Exhibit A Form of Escrow Agreement"])

    def test_page_labels_and_running_titles_with_labels_stay_out(self) -> None:
        # 4b870189 "ToC-i"; 522db544 "... Agreement -ii-"; 8f517491 "Index - iv"; f7894812 "A- ii";
        # 1e789eee "P AGE 3"; d206a323 "i v"; 4c5903f5 a piped running footer.
        for footer in (
            "ToC-iii", "2 of 42", "Membership Interest Purchase Agreement -ii-", "SHARE PURCHASE AGREEMENT iii",
            "Index - iv", "A- ii", "P AGE 3", "i v", "Share sale agreement | Slinger Bag, Inc.",
            "Project Grease - share purchase agreement 1",
        ):
            with self.subTest(footer=footer):
                html = ENTRIES_11 + f"<p>EXHIBITS</p><p>Exhibit A Form of Escrow Agreement</p><p>{footer}</p>"
                self.assertEqual(_lines(html)[-2:], ["EXHIBITS", "Exhibit A Form of Escrow Agreement"])
        # A listed exhibit with its page label stays.
        html = ENTRIES_11 + "<p>EXHIBITS</p><p>Exhibit A Bank Merger Agreement A-1</p>"
        self.assertEqual(_lines(html)[-1], "Exhibit A Bank Merger Agreement A-1")

    def test_index_rows_and_headings_with_page_numbers_stay(self) -> None:
        # 28a080ad, 2cc4adcc: defined-terms index rows; a4a44e4d: an article heading.
        rows = [
            "Credit Agreement 75", "Merger Agreement A-3", "ARTICLE VIII REMEDIES FOR BREACH OF THIS AGREEMENT 39",
            "ARTICLE IX TERMINATION OF AGREEMENT",
        ]
        html = ENTRIES_11 + "<p>INDEX OF DEFINED TERMS</p>" + "".join(f"<p>{row}</p>" for row in rows)
        self.assertEqual(_lines(html)[-4:], rows)

    def test_a_leading_page_label_repeats_as_one_header(self) -> None:
        # c6d40edf: "(i) SHARE PURCHASE AGREEMENT" ... "(iv) SHARE PURCHASE AGREEMENT".
        header = "<p>({}) SHARE PURCHASE AGREEMENT</p>"
        pages = [
            header.format("i") + ENTRIES_11 + "<p>EXHIBITS</p><p>Exhibit A Form of Escrow Agreement</p>",
            header.format("ii") + ENTRIES_11 + "<p>Exhibit B Form of Guaranty</p>",
        ]
        keys = repeated_toc_extra_keys(pages)
        self.assertNotIn("(ii) SHARE PURCHASE AGREEMENT", _lines_with(pages[1], keys))
        self.assertEqual(_lines_with(pages[1], keys)[-1], "Exhibit B Form of Guaranty")

    def test_by_and_among_in_an_exhibit_item_does_not_end_the_tail(self) -> None:
        items = [
            "Exhibit A Form of Support Agreement, by and among Parent and the Holders",
            "Exhibit B Form of Escrow Agreement",
        ]
        html = ENTRIES_11 + "<p>EXHIBITS</p>" + "".join(f"<p>{item}</p>" for item in items)
        self.assertEqual(_lines(html)[-2:], items)

    def test_legend_words_inside_list_items_do_not_end_the_list(self) -> None:
        # 362a4af3 "Copyright Assignment"; 3e061898 "A-7 [Intentionally Omitted]"; 7519a2de.
        items = [
            "Exhibit H \u2013 Form of Trademark Assignment",
            "Exhibit I \u2013 Form of Copyright Assignment",
            "Exhibit K - [intentionally Omitted]",
            "A-7 [Intentionally Omitted]",
            "Exhibit L \u2013 Form of Escrow Agreement",
        ]
        html = ENTRIES_11 + "<p>EXHIBITS</p>" + "".join(f"<p>{item}</p>" for item in items)
        self.assertEqual(_lines(html)[-5:], items)

    def test_filing_legends_still_end_the_list(self) -> None:
        # 52450ccb, ce92e4e8: "furnish supplementally" notes; a schedules-omitted legend.
        for legend in (
            "The Company will furnish supplementally to the Commission a copy of any exhibit to this Agreement",
            "Schedules omitted pursuant to Item 601(a)(5) of Regulation S-K.",
            "* Certain confidential information has been omitted.",
        ):
            with self.subTest(legend=legend):
                html = (
                    ENTRIES_11 + "<p>EXHIBITS</p><p>Exhibit A Form of Escrow Agreement</p>"
                    + f"<p>{legend}</p><p>the SEC upon request.</p>"
                )
                self.assertEqual(_lines(html)[-2:], ["EXHIBITS", "Exhibit A Form of Escrow Agreement"])

    def test_an_exhibit_list_printed_as_one_paragraph_is_kept(self) -> None:
        # 36fbf519: the whole list in one paragraph, over 300 characters.
        paragraph = " ".join(f"Exhibit {c} Form of Ancillary Agreement Number {i} Among the Parties" for i, c in enumerate("ABCDEF"))
        html = ENTRIES_11 + "<p>EXHIBITS</p><p>" + paragraph + "</p>"
        text = " ".join(_lines(html))
        self.assertIn("Exhibit A Form of Ancillary", text)
        self.assertIn("Exhibit F Form of Ancillary Agreement Number 5 Among the Parties", text)

    def test_list_items_that_read_as_titles_are_kept(self) -> None:
        # ce014d52 "A. Voting Agreement"; 301bb52d "Escrow Agreement Exhibit A"; e745caee
        # "Management Agreement B"; 57a26f02 "Asset Purchase Agreement Exhibits"; a mid-list
        # "Tab 3 - Stock Purchase Agreement".
        items = [
            "Asset Purchase Agreement Exhibits",
            "A. Voting Agreement",
            "B. Non-competition Agreement",
            "Escrow Agreement Exhibit C",
            "Management Agreement D",
            "Tab 3 - Stock Purchase Agreement",
            "EXHIBITG Employee Agreement or Restrictive Covenant Agreement",
            "Exhibit H Form of Guaranty",
        ]
        html = ENTRIES_11 + "".join(f"<p>{item}</p>" for item in items)
        self.assertEqual(_lines(html)[-len(items):], items)

    def test_a_bare_title_is_dropped_only_at_the_end_of_the_page(self) -> None:
        # A running footer is the page's last extra; inside an index the same shape is a row.
        html = (
            ENTRIES_11 + "<p>INDEX OF DEFINED TERMS</p><p>Agreement Preamble</p><p>Business Day 1.1</p>"
            "<p>Stock Purchase Agreement</p>"
        )
        self.assertEqual(_lines(html)[-3:], ["INDEX OF DEFINED TERMS", "Agreement Preamble", "Business Day 1.1"])

    def test_title_page_labels_are_real_roman_numerals(self) -> None:
        from etl.domain.f_xml import _TOC_TITLE_PAGE_RE

        self.assertEqual(_TOC_TITLE_PAGE_RE.sub("", "Stock Purchase Agreement iii"), "Stock Purchase Agreement")
        for line in ("Settlement Agreement civil", "Settlement Agreement ill"):
            self.assertEqual(_TOC_TITLE_PAGE_RE.sub("", line), line)

    def test_bracketed_items_headings_and_entries_do_not_stop_the_list(self) -> None:
        # cdf43fb9 "[Exhibit D \u2013 Intentionally Omitted]"; 7943689e "CONFIDENTIAL SCHEDULES" in the
        # items' block; 77a4901b an item with a filing note; 9d6468e2 "Schedule I [**]".
        items = [
            "[Exhibit D \u2013 Intentionally Omitted]",
            "Exhibit E \u2013 Term Sheet for Long-Term Supply Agreement",
            "CONFIDENTIAL SCHEDULES",
            "FORM OF BANK MERGER AGREEMENT EXHIBIT A",
            "Exhibit A - Accounting Procedures Omitted pursuant to Item 601(b)(2) of Regulation SK.",
            "Exhibit B - Form of Non-competition, Non-solicitation Agreement *",
            "Schedule I [**]",
            "Schedule II Net Cash",
        ]
        html = ENTRIES_11 + "<p>EXHIBITS</p>" + "".join(f"<p>{item}</p>" for item in items)
        self.assertEqual(_lines(html)[-len(items):], items)

    def test_an_omitted_toc_entry_does_not_stop_the_toc(self) -> None:
        # 039890a0: "Section 8.3 Omitted" then "ARTICLE IX. MISCELLANEOUS 45".
        html = ENTRIES_11 + "<p>Section 8.3 Omitted</p><p>ARTICLE IX. MISCELLANEOUS 45</p>"
        self.assertEqual(_lines(html)[-2:], ["Section 8.3 Omitted", "ARTICLE IX. MISCELLANEOUS 45"])

    def test_the_list_under_an_omitted_exhibits_note_is_kept(self) -> None:
        # 275cf259, 1951fa70, f1450781, 1fadca28: the note stays out, the list under it is served.
        html = (
            ENTRIES_11 + "<p>SCHEDULES *</p><p>* Omitted pursuant to Instruction 4 of Item 1.01 of Form 8-K.</p>"
            "<p>[ The following Exhibits have been omitted :</p><p>EXHIBITS *</p>"
            "<p>\u2022 Exhibit A: Form of Certificate of Merger</p><p>Exhibit K Form of Paying Agent Agreement]</p>"
        )
        self.assertEqual(
            _lines(html)[-4:],
            ["SCHEDULES *", "EXHIBITS *", "\u2022 Exhibit A: Form of Certificate of Merger", "Exhibit K Form of Paying Agent Agreement]"],
        )

    def test_all_caps_legends_still_stay_out(self) -> None:
        for legend in ("EXHIBITS AND SCHEDULES OMITTED", "A CONFIDENTIAL TREATMENT REQUEST HAS BEEN FILED"):
            with self.subTest(legend=legend):
                html = (
                    ENTRIES_11 + "<p>EXHIBITS</p><p>Exhibit A Form of Escrow Agreement</p>"
                    + f"<p>{legend}</p><p>PURSUANT TO THE RULES OF THE COMMISSION</p>"
                )
                self.assertEqual(_lines(html)[-2:], ["EXHIBITS", "Exhibit A Form of Escrow Agreement"])

    def test_headings_and_notes_that_mention_the_agreement_are_kept(self) -> None:
        # 2eb9f0a3, 3625b546, de8d6c39, c6d806eb.
        for line in (
            "Purchaser Schedules to Asset Purchase Agreement",
            "AGREED FORM DOCUMENTS REFERRED TO IN THIS AGREEMENT",
            "LIST OF SCHEDULES TO THE ASSET PURCHASE AGREEMENT",
            "Delivered contemporaneously with this Agreement",
        ):
            with self.subTest(line=line):
                html = ENTRIES_11 + f"<p>DISCLOSURE SCHEDULES:</p><p>{line}</p>"
                self.assertEqual(_lines(html)[-1], line)

    def test_only_one_bare_title_is_peeled_at_a_page_end(self) -> None:
        html = ENTRIES_11 + "<p>ANCILLARY DOCUMENTS</p><p>Voting Agreement</p><p>Escrow Agreement</p><p>Stock Purchase Agreement</p>"
        self.assertEqual(_lines(html)[-3:], ["ANCILLARY DOCUMENTS", "Voting Agreement", "Escrow Agreement"])

    def test_a_title_with_an_uppercase_roman_page_label_is_a_footer(self) -> None:
        for footer in ("Merger Agreement I", "Agreement and Plan of Merger - V", "Merger Agreement - X"):
            with self.subTest(footer=footer):
                html = ENTRIES_11 + f"<p>EXHIBITS</p><p>Exhibit A Form of Escrow Agreement</p><p>{footer}</p>"
                self.assertEqual(_lines(html)[-2:], ["EXHIBITS", "Exhibit A Form of Escrow Agreement"])
        # Inside a list labeled at the end, a trailing roman letter is the item's label.
        html = ENTRIES_11 + "<p>Lock-up and Voting Agreement J</p><p>Purchase and Sale Agreement L</p>"
        self.assertEqual(_lines(html)[-2:], ["Lock-up and Voting Agreement J", "Purchase and Sale Agreement L"])

    def test_article_headings_repeated_on_overlapping_pages_are_kept(self) -> None:
        # e865462b, 8f32f1f5: the same ARTICLE heading and its wrapped title printed on two pages.
        block = "<p>ARTICLE III</p><p>REPRESENTATIONS AND WARRANTIES</p><p>OF THE SELLER</p>"
        pages = [ENTRIES_11 + block + "<p>Stock Purchase Agreement</p>", ENTRIES_11 + block]
        keys = repeated_toc_extra_keys(pages)
        for page in pages:
            self.assertEqual(
                _lines_with(page, keys)[-3:], ["ARTICLE III", "REPRESENTATIONS AND WARRANTIES", "OF THE SELLER"]
            )

    def test_dashed_page_labels_repeat_as_one_footer(self) -> None:
        footer = "<p>Purchase Agreement for Acme Holdings {}</p>"
        pages = [
            ENTRIES_11 + "<p>EXHIBITS</p><p>Exhibit A Form of Escrow Agreement</p>" + footer.format("-i-"),
            ENTRIES_11 + "<p>Exhibit B Form of Guaranty</p>" + footer.format("-ii-"),
        ]
        keys = repeated_toc_extra_keys(pages)
        self.assertEqual(_lines_with(pages[1], keys)[-1], "Exhibit B Form of Guaranty")

    def test_a_preamble_on_a_toc_page_ends_the_tail(self) -> None:
        # f29a2c63: the agreement's preamble starts on the TOC-labeled page.
        html = (
            ENTRIES_11 + "<p>EXHIBITS</p><p>Exhibit 1 - Shareholders</p>"
            + "<p>THIS UNIT PURCHASE AGREEMENT (this \u201c Agreement \u201d) is effective as of August 30, 2023, "
            + "by and among the parties.</p><p>RECITALS</p>"
        )
        self.assertEqual(_lines(html)[-2:], ["EXHIBITS", "Exhibit 1 - Shareholders"])

    def test_a_page_without_extras_is_unchanged(self) -> None:
        html = "<p>TABLE OF CONTENTS</p><p>Page</p>" + ENTRIES_11 + "<p align='center'>ii</p>"
        self.assertEqual(
            _lines(html)[:2] + [line.split()[0] for line in _lines(html)[2:]],
            ["TABLE OF CONTENTS", "Page", "Section", "Section", "Section"],
        )

    def test_positioned_rows_keep_the_tail(self) -> None:
        row = "<div style='position:relative; overflow:hidden'>{}</div>"
        html = (
            row.format("Section 1.1 Definitions 1")
            + row.format("Section 1.2 Interpretation 9")
            + "<div><p>SCHEDULES</p><p>Schedule 1.1 Knowledge Persons</p><p>12</p></div>"
        )
        lines = _lines(html)
        self.assertEqual(lines[-2:], ["SCHEDULES", "Schedule 1.1 Knowledge Persons"])


if __name__ == "__main__":
    unittest.main()
