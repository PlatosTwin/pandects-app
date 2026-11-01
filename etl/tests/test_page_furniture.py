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

        self.assertEqual(result.page_texts, [f"Body text {i}.\n\nPage <page>{i + 1}</page>" for i in range(3)])


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
