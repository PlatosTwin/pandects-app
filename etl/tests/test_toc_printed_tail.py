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
            + "<p>#4854-0942-8064v43</p><p>A-ii</p><p>()</p><p>________</p>"
            + '<p>IF " DOCVARIABLE "SWDocIDLocation" 1" = "1" " DOCPROPERTY "SWDocID" SL1 3716757v12 101000.00058" ""</p>'
            + "<p>1 The Table of Contents is not a part of this Agreement.</p>"
            + "<p>Page | ii</p><p>PAGE || iii</p><p>Page ii of iv</p><p>(CONTINUED ON NEXT PAGE)</p>"
            + "<p>[Remainder of page intentionally left blank]</p>"
            + "<p>EXHIBITS</p><p>Exhibit A Form of Voting Agreement</p><p>B Mid-Tier Merger Agreement</p>"
            + "<p>Confidential and Proprietary</p><p>Annex A Not a continuation of the list</p>"
        )
        lines = _lines(html)
        self.assertEqual(lines[-3:], ["EXHIBITS", "Exhibit A Form of Voting Agreement", "B Mid-Tier Merger Agreement"])
        for furniture in (
            "AGREEMENT AND PLAN OF MERGER", "by and among", "- iv -", "#25932618 v26", "328972.00001/116443569.20",
            "Share sale agreement", "Section Description", "ii.", "Confidential and Proprietary",
            "#4854-0942-8064v43", "A-ii", "()", "________", "1 The Table of Contents is not a part of this Agreement.",
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
            + "<p>Herbert Smith Freehills owns the copyright in this document</p><p>Exhibit B Not kept</p>"
        )
        self.assertEqual(_lines(html)[-2:], ["EXHIBITS", "Exhibit A Form of Escrow Agreement"])

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
