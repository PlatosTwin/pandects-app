import unittest
import xml.etree.ElementTree as ET

from etl.defs.f_xml_asset import (
    XML_REASON_SECTION_TITLE_INVALID_NUMBERING,
    _extract_section_numbers,
    extract_toc_section_sequences,
    find_hard_rule_violations,
)


class SectionSignNumberTests(unittest.TestCase):
    def test_section_sign_numbers_parse(self) -> None:
        self.assertEqual(_extract_section_numbers("§4.9 Taxes"), (4, 9))
        self.assertEqual(_extract_section_numbers("§ 4.9 Taxes"), (4, 9))
        self.assertEqual(_extract_section_numbers("  §12.01. Notices"), (12, 1))

    def test_existing_forms_unchanged(self) -> None:
        self.assertEqual(_extract_section_numbers("Section 4.9 Taxes"), (4, 9))
        self.assertEqual(_extract_section_numbers("SECTIONS 4.9 and 4.10"), (4, 9))
        self.assertEqual(_extract_section_numbers("4.9 Taxes"), (4, 9))
        # A prefix glued to the number is still not a section number.
        self.assertIsNone(_extract_section_numbers("SECTION4.9 Taxes"))

    def test_section_sign_negatives(self) -> None:
        self.assertIsNone(_extract_section_numbers("§4.9.1 Sub-clause"))
        self.assertIsNone(_extract_section_numbers("§§ 4.9 Taxes"))
        self.assertIsNone(_extract_section_numbers("Taxes §4.9"))
        self.assertIsNone(_extract_section_numbers("§ Taxes"))

    def test_section_sign_titles_pass_hard_rules(self) -> None:
        root = ET.fromstring(
            """
            <document>
              <body>
                <article title="ARTICLE IV REPRESENTATIONS">
                  <section title="§4.1 Organization" pageUUID="p1" />
                  <section title="§4.2 Authority" pageUUID="p1" />
                  <section title="§ 4.3 Taxes" pageUUID="p2" />
                </article>
              </body>
            </document>
            """
        )

        violations = find_hard_rule_violations(root)

        self.assertFalse(
            any(v.reason_code == XML_REASON_SECTION_TITLE_INVALID_NUMBERING for v in violations),
            violations,
        )

    def test_toc_sequences_read_section_sign_numbers(self) -> None:
        root = ET.fromstring(
            "<document><tableOfContents>§4.1 Organization 12\n§4.2 Authority 13\n"
            "§ 5.1 Covenants 20</tableOfContents></document>"
        )

        self.assertEqual(extract_toc_section_sequences(root), {4: [1, 2], 5: [1]})


if __name__ == "__main__":
    unittest.main()
