# pyright: reportAny=false
import unittest

from etl.domain.tax_module import extract_tax_clauses


class TaxModuleDomainTests(unittest.TestCase):
    def test_extract_tax_clauses_splits_top_level_lettered_blocks(self) -> None:
        clauses = extract_tax_clauses(
            {
                "agreement_uuid": "agreement-1",
                "section_uuid": "section-1",
                "article_title": "ARTICLE VI ADDITIONAL COVENANTS",
                "article_title_normed": "additional covenants",
                "section_title": "6.18 Tax Matters",
                "section_title_normed": "tax matters",
                "xml_content": (
                    "<text>Preamble sentence.</text>"
                    "<text>(a) Parent shall bear all transfer taxes.</text>"
                    "<text>(i) This includes stamp taxes.</text>"
                    "<text>(b) The parties intend the merger to qualify as tax-free.</text>"
                ),
                "xml_version": 4,
                "section_standard_id": None,
                "section_standard_id_gold_label": None,
            }
        )

        self.assertEqual(len(clauses), 2)
        self.assertEqual(clauses[0]["anchor_label"], "(a)")
        self.assertEqual(clauses[0]["source_method"], "enumerated_split")
        self.assertIn("Preamble sentence.", clauses[0]["clause_text"])
        self.assertIn("(i) This includes stamp taxes.", clauses[0]["clause_text"])
        self.assertEqual(clauses[1]["anchor_label"], "(b)")
        self.assertEqual(clauses[1]["context_type"], "operative")
        self.assertLess(clauses[0]["start_char"], clauses[0]["end_char"])

    def test_extract_tax_clauses_falls_back_to_whole_section(self) -> None:
        clauses = extract_tax_clauses(
            {
                "agreement_uuid": "agreement-2",
                "section_uuid": "section-2",
                "article_title": "ARTICLE VI COVENANTS",
                "article_title_normed": "covenants",
                "section_title": "Transfer Taxes",
                "section_title_normed": "transfer taxes",
                "xml_content": "<text>The parties shall split transfer taxes equally.</text>",
                "xml_version": 1,
                "section_standard_id": None,
                "section_standard_id_gold_label": None,
            }
        )

        self.assertEqual(len(clauses), 1)
        self.assertIsNone(clauses[0]["anchor_label"])
        self.assertEqual(clauses[0]["source_method"], "whole_section_fallback")
        self.assertEqual(clauses[0]["clause_text"], "The parties shall split transfer taxes equally.")

    def test_extract_tax_clauses_marks_rep_warranty_context(self) -> None:
        clauses = extract_tax_clauses(
            {
                "agreement_uuid": "agreement-3",
                "section_uuid": "section-3",
                "article_title": "ARTICLE III REPRESENTATIONS AND WARRANTIES OF THE COMPANY",
                "article_title_normed": "representations and warranties of the company",
                "section_title": "3.18 Tax Matters",
                "section_title_normed": "tax matters",
                "xml_content": "<text>(a) The Company has filed all Tax Returns.</text>",
                "xml_version": 2,
                "section_standard_id": None,
                "section_standard_id_gold_label": None,
            }
        )

        self.assertEqual(len(clauses), 1)
        self.assertEqual(clauses[0]["context_type"], "rep_warranty")


if __name__ == "__main__":
    unittest.main()
