# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
from datetime import date
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    _ = unittest.main()
