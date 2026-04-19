# pyright: reportAny=false
import unittest
from typing import cast

from etl.domain.tax_module import (
    TaxSectionRow,
    extract_tax_clauses,
    has_tax_signal,
    is_tax_related_section,
)


class TaxModuleDomainTests(unittest.TestCase):
    def test_tax_signal_matches_observed_tax_heading_families(self) -> None:
        tax_headings = [
            "No 338(g) Election",
            "Section 338(h)(10) Election",
            "Sections 336 and 338 of the Code",
            "Section 754 Election",
            "Section 83(b) Elections",
            "FIRPTA Certificate",
            "Required Withholding",
            "Transfer and Gains Taxes",
            "GST/HST Gross Up",
            "Straddle Period Allocation",
            "Refunds and Credits",
            "Tax Contest",
            "Tax-Free Reorganization Treatment",
            "Section 368(a) Reorganization",
            "Purchase Price Allocation for Tax Purposes",
            "Entity Classification Elections",
            "Push-Out Election",
            "Golden Parachute Payments",
            "Tax-Sharing Agreements",
            "IRS Ruling",
            "REIT Status",
            "S Corporation Status",
        ]

        for heading in tax_headings:
            with self.subTest(heading=heading):
                self.assertTrue(has_tax_signal(heading))

    def test_tax_signal_does_not_match_common_false_positive_headings(self) -> None:
        non_tax_headings = [
            "Information Technology",
            "Intellectual Property and Information Technology",
            "Purchase Price",
            "Purchase Price Adjustment",
            "Restrictions on Business Activities",
            "Restrictive Covenants",
            "Credit Support",
            "Letters of Credit",
            "Private Placement",
            "Accredited Investor",
            "Election Procedures",
            "Stock Election",
            "Election of Directors",
        ]

        for heading in non_tax_headings:
            with self.subTest(heading=heading):
                self.assertFalse(has_tax_signal(heading))

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
        self.assertEqual(clauses[0]["source_method"], "section_title_match")
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

    def test_specific_tax_section_heading_is_one_section_level_clause(self) -> None:
        row = {
            "agreement_uuid": "agreement-4",
            "section_uuid": "section-4",
            "article_title": "ARTICLE VI COVENANTS",
            "article_title_normed": "covenants",
            "section_title": "Section 6.18 338(g) Election",
            "section_title_normed": "338(g) election",
            "xml_content": (
                "<text>(a) Parent may make an election under Section 338(g).</text>"
                "<text>(b) Company shall cooperate with forms and filings.</text>"
            ),
            "xml_version": 3,
            "section_standard_id": None,
            "section_standard_id_gold_label": None,
        }

        tax_section_row = cast(TaxSectionRow, cast(object, row))
        self.assertTrue(is_tax_related_section(tax_section_row, tax_standard_ids=set()))
        clauses = extract_tax_clauses(tax_section_row)

        self.assertEqual(len(clauses), 1)
        self.assertIsNone(clauses[0]["anchor_label"])
        self.assertEqual(clauses[0]["source_method"], "section_title_match")
        self.assertIn("Section 338(g)", clauses[0]["clause_text"])
        self.assertIn("Company shall cooperate", clauses[0]["clause_text"])

    def test_generic_tax_section_without_enumerators_splits_paragraphs(self) -> None:
        clauses = extract_tax_clauses(
            {
                "agreement_uuid": "agreement-5",
                "section_uuid": "section-5",
                "article_title": "ARTICLE VI COVENANTS",
                "article_title_normed": "covenants",
                "section_title": "Section 6.18 Tax Matters",
                "section_title_normed": "tax matters",
                "xml_content": (
                    "<text>The parties intend the merger to qualify as a reorganization under Section 368.</text>"
                    "<text>Parent may not make a Section 338(g) election without Seller consent.</text>"
                ),
                "xml_version": 3,
                "section_standard_id": None,
                "section_standard_id_gold_label": None,
            }
        )

        self.assertEqual(len(clauses), 2)
        self.assertEqual([clause["source_method"] for clause in clauses], ["paragraph_split", "paragraph_split"])
        self.assertIn("Section 368", clauses[0]["clause_text"])
        self.assertIn("Section 338(g)", clauses[1]["clause_text"])

    def test_generic_tax_section_merges_paragraph_text_split_across_pages(self) -> None:
        clauses = extract_tax_clauses(
            {
                "agreement_uuid": "agreement-6",
                "section_uuid": "section-6",
                "article_title": "ARTICLE VI COVENANTS",
                "article_title_normed": "covenants",
                "section_title": "Section 6.18 Tax Matters",
                "section_title_normed": "tax matters",
                "xml_content": (
                    "<text>The parties intend the merger to qualify</text>"
                    "<pageUUID>page-1</pageUUID>"
                    "<text>as a reorganization within the meaning of Section 368.</text>"
                    "<text>Parent shall pay all transfer taxes.</text>"
                ),
                "xml_version": 3,
                "section_standard_id": None,
                "section_standard_id_gold_label": None,
            }
        )

        self.assertEqual(len(clauses), 2)
        self.assertEqual(clauses[0]["source_method"], "paragraph_split")
        self.assertIn("qualify\n\nas a reorganization", clauses[0]["clause_text"])
        self.assertIn("transfer taxes", clauses[1]["clause_text"])


if __name__ == "__main__":
    _ = unittest.main()
