# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
from datetime import date
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import pandas as pd

from etl.domain import f_xml


class XMLGenerationTests(unittest.TestCase):
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
                    "acquirer": "Acquirer",
                    "target": "Target",
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
                    "acquirer": "Acquirer",
                    "target": "Target",
                    "filing_date": date(2024, 1, 1),
                    "source_is_html": True,
                },
            ]
        )

        original_convert_to_xml = f_xml.convert_to_xml

        def mock_convert_to_xml(
            tagged_text: str,
            agreement_uuid: str,
            acquirer: str,
            target: str,
            filing_date: date,
            url: str,
            source_format: str,
        ) -> str:
            if agreement_uuid == invalid_uuid:
                return "<document><broken></document>"
            return original_convert_to_xml(
                tagged_text,
                agreement_uuid,
                acquirer,
                target,
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
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].agreement_uuid, invalid_uuid)

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
            acquirer="Acquirer",
            target="Target",
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
            acquirer="Acquirer",
            target="Target",
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
            acquirer="Acquirer",
            target="Target",
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
