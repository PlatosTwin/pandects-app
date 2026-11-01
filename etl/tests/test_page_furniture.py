# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
from datetime import date
import unittest
import xml.etree.ElementTree as ET

import pandas as pd

from etl.domain import f_xml
from etl.domain.page_furniture import furniture_kind, strip_page_furniture


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
            "\n".join(["EXECUTION COPY", "", "  The Company shall    pay  the fee.  ", "12"]) for _ in range(3)
        ]

        result = strip_page_furniture(pages)

        for text in result.page_texts:
            self.assertEqual(text, "\n  The Company shall    pay  the fee.  ")


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


if __name__ == "__main__":
    unittest.main()
