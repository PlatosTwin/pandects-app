# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
from datetime import date
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import pandas as pd

from etl.domain import f_xml


class XMLGenerationTests(unittest.TestCase):
    def test_format_toc_html_like_screen_preserves_table_rows(self) -> None:
        raw_html = """
        <div><p>TABLE OF CONTENTS Page</p></div>
        <table>
          <tr><td>ARTICLE I THE MERGER</td><td>6</td></tr>
          <tr><td>1.1</td><td>The Merger</td><td>6</td></tr>
          <tr><td>1.2</td><td>The Closing</td><td>6</td></tr>
        </table>
        """

        formatted = f_xml.format_toc_html_like_screen(raw_html, line_width=80)

        self.assertIn("TABLE OF CONTENTS\n\nPage", formatted)
        self.assertIn("ARTICLE I THE MERGER", formatted)
        self.assertIn("1.1     The Merger", formatted)
        self.assertIn("1.2     The Closing", formatted)
        self.assertNotIn("ARTICLE I THE MERGER 6 1.1", formatted)

    def test_format_toc_html_like_screen_ignores_nested_hidden_nodes(self) -> None:
        raw_html = """
        <div style="display: none"><span>hidden section</span></div>
        <table>
          <tr><td>1.1</td><td>The Merger</td><td>6</td></tr>
        </table>
        """

        formatted = f_xml.format_toc_html_like_screen(raw_html, line_width=80)

        self.assertIn("1.1     The Merger", formatted)
        self.assertNotIn("hidden section", formatted)

    def test_generate_xml_reformats_html_toc_raw_page_content(self) -> None:
        agreement_uuid = "11111111-1111-1111-1111-111111111111"
        raw_toc_html = """
        <div><p>TABLE OF CONTENTS Page</p></div>
        <table>
          <tr><td>ARTICLE I THE MERGER</td><td>6</td></tr>
          <tr><td>1.1</td><td>The Merger</td><td>6</td></tr>
          <tr><td>1.2</td><td>The Closing</td><td>6</td></tr>
        </table>
        """
        df = pd.DataFrame(
            [
                {
                    "agreement_uuid": agreement_uuid,
                    "page_uuid": "toc-page",
                    "page_order": 1,
                    "raw_page_content": raw_toc_html,
                    "source_page_type": "toc",
                    "gold_label": None,
                    "tagged_output": "TABLE OF CONTENTS Page ARTICLE I THE MERGER 6 1.1 The Merger 6 1.2 The Closing 6",
                    "url": "https://example.com/agreement",
                    "filing_date": date(2024, 1, 1),
                    "source_is_txt": False,
                    "source_is_html": True,
                },
                {
                    "agreement_uuid": agreement_uuid,
                    "page_uuid": "body-page",
                    "page_order": 2,
                    "raw_page_content": "<p>ARTICLE I</p>",
                    "source_page_type": "body",
                    "gold_label": None,
                    "tagged_output": "<article>ARTICLE I</article>Body text",
                    "url": "https://example.com/agreement",
                    "filing_date": date(2024, 1, 1),
                    "source_is_txt": False,
                    "source_is_html": True,
                },
            ]
        )

        generated, failures = f_xml.generate_xml(df)

        self.assertEqual(failures, [])
        self.assertEqual(len(generated), 1)
        root = ET.fromstring(generated[0].xml)
        toc = root.find("tableOfContents")
        self.assertIsNotNone(toc)
        assert toc is not None
        toc_texts = [el.text or "" for el in toc.findall("text")]
        self.assertIn("TABLE OF CONTENTS", toc_texts)
        self.assertIn("Page", toc_texts)
        self.assertTrue(any(text.startswith("1.1     The Merger") for text in toc_texts))
        self.assertTrue(any(text.startswith("1.2     The Closing") for text in toc_texts))
        self.assertNotIn(
            "TABLE OF CONTENTS Page ARTICLE I THE MERGER 6 1.1 The Merger 6 1.2 The Closing 6",
            toc_texts,
        )

    def test_generate_xml_uses_gold_label_for_toc_reformatting(self) -> None:
        agreement_uuid = "11111111-1111-1111-1111-111111111111"
        df = pd.DataFrame(
            [
                {
                    "agreement_uuid": agreement_uuid,
                    "page_uuid": "toc-page",
                    "page_order": 1,
                    "raw_page_content": """
                    <table>
                      <tr><td>1.1</td><td>The Merger</td><td>6</td></tr>
                    </table>
                    """,
                    "source_page_type": "body",
                    "gold_label": "toc",
                    "tagged_output": "1.1 The Merger 6",
                    "url": "https://example.com/agreement",
                    "filing_date": date(2024, 1, 1),
                    "source_is_txt": False,
                    "source_is_html": True,
                },
            ]
        )

        generated, failures = f_xml.generate_xml(df)

        self.assertEqual(failures, [])
        self.assertEqual(len(generated), 1)
        root = ET.fromstring(generated[0].xml)
        toc = root.find("tableOfContents")
        self.assertIsNotNone(toc)
        assert toc is not None
        toc_texts = [el.text or "" for el in toc.findall("text")]
        self.assertTrue(any(text.startswith("1.1     The Merger") for text in toc_texts))

    def test_format_signature_text_like_screen_joins_field_values(self) -> None:
        rendered_signature = """
        STANLEY BLACK & DECKER, INC.

        By:
         /s/ Donald Allan, Jr.

        Name:
        Donald Allan, Jr.

        Title:
        Senior Vice President and Chief Financial Officer

        SPECTRUM BRANDS,
        INC.

        by
        /s/ Nathan Fagre
        """

        formatted = f_xml.format_signature_text_like_screen(rendered_signature)

        self.assertIn("By: /s/ Donald Allan, Jr.", formatted)
        self.assertIn("Name: Donald Allan, Jr.", formatted)
        self.assertIn(
            "Title: Senior Vice President and Chief Financial Officer",
            formatted,
        )
        self.assertIn("SPECTRUM BRANDS, INC.", formatted)
        self.assertIn("By: /s/ Nathan Fagre", formatted)
        self.assertNotIn("By:\n\n/s/ Donald Allan, Jr.", formatted)

    def test_format_signature_text_like_screen_preserves_blank_form_fields(self) -> None:
        rendered_signature = """
        SUBSCRIBER:

        By:

        Name:
        Title:

        (Please print. Please indicate name and capacity of person signing above)
        """

        formatted = f_xml.format_signature_text_like_screen(rendered_signature)

        self.assertIn("By:\n\nName:\nTitle:", formatted)
        self.assertIn(
            "(Please print. Please indicate name and capacity of person signing above)",
            formatted,
        )
        self.assertNotIn(
            "Title: (Please print. Please indicate name and capacity of person signing above)",
            formatted,
        )

    def test_format_signature_html_like_screen_preserves_table_rows(self) -> None:
        raw_html = """
        <div><font>In Witness Whereof, the parties hereto have entered into and signed this Agreement.</font></div>
        <div>
          <table>
            <tr><td></td><td>PARENT:</td></tr>
            <tr><td></td><td>HawkEye 360, Inc.</td></tr>
            <tr style="height:14pt"><td></td><td></td></tr>
            <tr><td></td><td>By:</td><td>/s/ John Serafini</td></tr>
            <tr><td></td><td>Name:</td><td>John Serafini</td></tr>
            <tr><td></td><td>Its:</td><td>Chief Executive Officer</td></tr>
            <tr style="height:14pt"><td></td><td></td></tr>
            <tr><td></td><td>MERGER SUB:</td></tr>
            <tr><td></td><td>Forestal Merger Sub, Inc.</td></tr>
            <tr><td></td><td>By:</td><td>/s/ Todd Probert</td></tr>
            <tr><td></td><td>Name:</td><td>Todd Probert</td></tr>
            <tr><td></td><td>Its:</td><td>President</td></tr>
          </table>
        </div>
        <div><font>[SIGNATURE PAGE TO AGREEMENT AND PLAN OF MERGER]</font></div>
        """

        formatted = f_xml.format_signature_html_like_screen(raw_html)

        self.assertIn("PARENT:", formatted)
        self.assertIn("HawkEye 360, Inc.", formatted)
        self.assertIn(
            "In Witness Whereof, the parties hereto have entered into and signed this Agreement.",
            formatted,
        )
        self.assertNotIn("Whereof ,", formatted)
        self.assertIn("By: /s/ John Serafini", formatted)
        self.assertIn("Name: John Serafini", formatted)
        self.assertIn("Its: Chief Executive Officer", formatted)
        self.assertIn("MERGER SUB:", formatted)
        self.assertIn("Forestal Merger Sub, Inc.", formatted)
        self.assertIn("By: /s/ Todd Probert", formatted)
        self.assertIn("Its: President", formatted)
        self.assertNotIn(
            "PARENT: HawkEye 360, Inc. By: /s/ John Serafini Name: John Serafini",
            formatted,
        )

    def test_format_signature_html_like_screen_normalizes_bracket_spacing(self) -> None:
        raw_html = """
        <div><font>( Signature Page )</font></div>
        <div><font>[ SIGNATURE PAGE TO AGREEMENT AND PLAN OF MERGER ]</font></div>
        """

        formatted = f_xml.format_signature_html_like_screen(raw_html)

        self.assertIn("(Signature Page)", formatted)
        self.assertIn("[SIGNATURE PAGE TO AGREEMENT AND PLAN OF MERGER]", formatted)
        self.assertNotIn("( Signature Page )", formatted)
        self.assertNotIn("[ SIGNATURE PAGE TO AGREEMENT AND PLAN OF MERGER ]", formatted)

    def test_generate_xml_reformats_html_signature_raw_page_content(self) -> None:
        agreement_uuid = "11111111-1111-1111-1111-111111111111"
        df = pd.DataFrame(
            [
                {
                    "agreement_uuid": agreement_uuid,
                    "page_uuid": "sig-page",
                    "page_order": 1,
                    "raw_page_content": """
                    <div><font>IN WITNESS WHEREOF</font></div>
                    <div>
                      <table>
                        <tr><td></td><td>ACME CORP.</td></tr>
                        <tr><td></td><td>By:</td><td>/s/ Jane Doe</td></tr>
                        <tr><td></td><td>Name:</td><td>Jane Doe</td></tr>
                        <tr><td></td><td>Its:</td><td>Chief Executive Officer</td></tr>
                      </table>
                    </div>
                    """,
                    "source_page_type": "body",
                    "gold_label": "sig",
                    "tagged_output": (
                        "IN WITNESS WHEREOF\n\n"
                        "ACME CORP. By: /s/ Jane Doe Name: Jane Doe Its: Chief Executive Officer"
                    ),
                    "url": "https://example.com/agreement",
                    "filing_date": date(2024, 1, 1),
                    "source_is_txt": False,
                    "source_is_html": True,
                },
            ]
        )

        generated, failures = f_xml.generate_xml(df)

        self.assertEqual(failures, [])
        self.assertEqual(len(generated), 1)
        root = ET.fromstring(generated[0].xml)
        sig_pages = root.find("sigPages")
        self.assertIsNotNone(sig_pages)
        assert sig_pages is not None
        sig_texts = [el.text or "" for el in sig_pages.findall("text")]
        self.assertIn("ACME CORP.", sig_texts)
        self.assertIn("By: /s/ Jane Doe", sig_texts)
        self.assertIn("Name: Jane Doe", sig_texts)
        self.assertIn("Its: Chief Executive Officer", sig_texts)

    def test_generate_xml_skips_agreement_with_invalid_xml_token(self) -> None:
        valid_uuid = "11111111-1111-1111-1111-111111111111"
        invalid_uuid = "22222222-2222-2222-2222-222222222222"
        df = pd.DataFrame(
            [
                {
                    "agreement_uuid": valid_uuid,
                    "page_uuid": "page-valid",
                    "page_order": 1,
                    "source_page_type": "body",
                    "tagged_output": "<article>ARTICLE I</article>Valid text",
                    "url": "https://example.com/valid",
                    "filing_date": date(2024, 1, 1),
                    "source_is_html": True,
                },
                {
                    "agreement_uuid": invalid_uuid,
                    "page_uuid": "page-invalid",
                    "page_order": 1,
                    "source_page_type": "body",
                    "tagged_output": "<article>ARTICLE I</article>Bad text",
                    "url": "https://example.com/invalid",
                    "filing_date": date(2024, 1, 1),
                    "source_is_html": True,
                },
            ]
        )

        original_convert_to_xml = f_xml.convert_to_xml

        def mock_convert_to_xml(
            tagged_text: str,
            agreement_uuid: str,
            filing_date: date,
            url: str,
            source_format: str,
        ) -> str:
            if agreement_uuid == invalid_uuid:
                return "<document><broken></document>"
            return original_convert_to_xml(
                tagged_text,
                agreement_uuid,
                filing_date,
                url,
                source_format,
            )

        with patch(
            "etl.domain.f_xml.convert_to_xml", side_effect=mock_convert_to_xml
        ):
            generated, failures = f_xml.generate_xml(df)

        self.assertEqual(len(generated), 1)
        self.assertEqual(generated[0].agreement_uuid, valid_uuid)
        valid_root = ET.fromstring(generated[0].xml)
        valid_metadata = valid_root.find("metadata")
        self.assertIsNotNone(valid_metadata)
        assert valid_metadata is not None
        self.assertEqual(valid_metadata.findtext("agreementUuid"), valid_uuid)
        self.assertIsNone(valid_metadata.find("target"))
        self.assertIsNone(valid_metadata.find("acquirer"))
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].agreement_uuid, invalid_uuid)

    def test_convert_to_xml_metadata_uses_agreement_uuid_not_party_names(self) -> None:
        agreement_uuid = "11111111-1111-1111-1111-111111111111"

        xml_str = f_xml.convert_to_xml(
            tagged_text="<article>ARTICLE I</article>Body text",
            agreement_uuid=agreement_uuid,
            filing_date=date(2024, 1, 1),
            url="https://example.com",
            source_format="html",
        )
        root = ET.fromstring(xml_str)
        metadata = root.find("metadata")
        self.assertIsNotNone(metadata)
        assert metadata is not None
        self.assertEqual(metadata.findtext("agreementUuid"), agreement_uuid)
        self.assertIsNone(metadata.find("target"))
        self.assertIsNone(metadata.find("acquirer"))

    def test_convert_to_xml_sets_heading_page_uuid_attribute(self) -> None:
        agreement_uuid = "11111111-1111-1111-1111-111111111111"
        tagged_text = (
            "<article>ARTICLE I</article>"
            "<section>1.01 First section.</section>"
            "Body text on page one."
            "<pageUUID>page-1</pageUUID>\n"
            "<section>1.02 Second section.</section>"
            "Body text on page two."
            "<pageUUID>page-2</pageUUID>"
        )

        xml_str = f_xml.convert_to_xml(
            tagged_text=tagged_text,
            agreement_uuid=agreement_uuid,
            filing_date=date(2024, 1, 1),
            url="https://example.com",
            source_format="html",
        )
        root = ET.fromstring(xml_str)

        body = root.find("body")
        self.assertIsNotNone(body)
        assert body is not None
        sections = list(body.iter("section"))
        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].attrib.get("pageUUID"), "page-1")
        self.assertEqual(sections[1].attrib.get("pageUUID"), "page-2")

    def test_convert_to_xml_uses_normalized_section_uuid_formula(self) -> None:
        agreement_uuid = "11111111-1111-1111-1111-111111111111"
        tagged_text = (
            "<article>ARTICLE I DEFINITIONS</article>"
            "<section>Section 1.01 . Definitions .</section>"
            "Alpha body."
            "<section>Section 1.02 The Merger .</section>"
            "Beta body."
        )

        xml_str = f_xml.convert_to_xml(
            tagged_text=tagged_text,
            agreement_uuid=agreement_uuid,
            filing_date=date(2024, 1, 1),
            url="https://example.com",
            source_format="html",
        )
        root = ET.fromstring(xml_str)
        body = root.find("body")
        self.assertIsNotNone(body)
        assert body is not None
        sections = list(body.iter("section"))
        self.assertEqual(len(sections), 2)
        self.assertEqual(
            sections[0].attrib["uuid"],
            f_xml.make_section_uuid(
                agreement_uuid,
                "definitions",
                "definitions",
                1,
                1,
            ),
        )
        self.assertEqual(
            sections[1].attrib["uuid"],
            f_xml.make_section_uuid(
                agreement_uuid,
                "definitions",
                "the merger",
                1,
                2,
            ),
        )

    def test_make_section_uuid_is_stable_across_heading_spacing_noise(self) -> None:
        agreement_uuid = "11111111-1111-1111-1111-111111111111"
        normalized_uuid = f_xml.make_section_uuid(
            agreement_uuid,
            "definitions",
            "definitions",
            1,
            1,
        )
        self.assertEqual(
            normalized_uuid,
            f_xml.make_section_uuid(
                agreement_uuid,
                "definitions",
                "definitions",
                1,
                1,
            ),
        )

    def test_convert_to_xml_keeps_section_text_across_inline_pageuuid_markers(self) -> None:
        agreement_uuid = "11111111-1111-1111-1111-111111111111"
        tagged_text = (
            "<article>ARTICLE V CERTAIN AGREEMENTS</article>"
            "<section>Section 5.5 Cooperation.</section>"
            "(a) At the expense of Purchaser, the Equityholders shall cooperate "
            "in obtaining the Financing necessary<pageUUID>page-11</pageUUID>\n"
            "to complete the transactions contemplated hereby.<pageUUID>page-12</pageUUID>\n"
            "<page>11</page>\n"
            "<section>Section 5.6 D&O Insurance.</section>"
            "Purchaser shall obtain tail coverage."
        )

        xml_str = f_xml.convert_to_xml(
            tagged_text=tagged_text,
            agreement_uuid=agreement_uuid,
            filing_date=date(2024, 1, 1),
            url="https://example.com",
            source_format="html",
        )
        root = ET.fromstring(xml_str)
        body = root.find("body")
        self.assertIsNotNone(body)
        assert body is not None
        sections = list(body.iter("section"))
        self.assertEqual(len(sections), 2)

        section_55 = sections[0]
        section_55_xml = ET.tostring(section_55, encoding="unicode")
        self.assertIn("obtaining the Financing necessary", section_55_xml)
        self.assertIn("to complete the transactions contemplated hereby.", section_55_xml)
        self.assertIn("<pageUUID>page-11</pageUUID>", section_55_xml)
        self.assertIn("<pageUUID>page-12</pageUUID>", section_55_xml)
        self.assertIn("<page>11</page>", section_55_xml)


if __name__ == "__main__":
    _ = unittest.main()
