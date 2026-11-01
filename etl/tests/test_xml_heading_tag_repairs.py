# pyright: reportAny=false
import unittest

from etl.domain.heading_titles import (
    normalize_glued_article_title,
    normalize_glued_section_title,
)
from etl.domain.xml_tag_repairs import (
    SectionGap,
    dedupe_adjacent_same_number_headings,
    insert_missing_section_heading_tags,
    split_combined_missing_section_tags,
    merge_number_only_section_titles,
    unwrap_bare_article_tags_before_other_article_sections,
    unwrap_same_line_cross_article_section_tags,
    unwrap_section_of_reference_tags,
    unwrap_stray_decimal_section_tags,
)


class SectionOfReferenceTagTests(unittest.TestCase):
    def test_unwraps_disclosure_schedule_reference(self) -> None:
        text = "as set forth in\n\n<section>Section 4.21 of the Company Disclosure Schedule.</section>\n\nThe"
        repaired, stats = unwrap_section_of_reference_tags(text)
        self.assertEqual(
            repaired,
            "as set forth in\n\nSection 4.21 of the Company Disclosure Schedule.\n\nThe",
        )
        self.assertEqual(stats.applied["unwrap_section_of_reference_tags"], 1)

    def test_keeps_heading_that_contains_of_later(self) -> None:
        text = "<section>Section 4.21 Absence of Changes.</section>"
        repaired, stats = unwrap_section_of_reference_tags(text)
        self.assertEqual(repaired, text)
        self.assertEqual(stats.applied["unwrap_section_of_reference_tags"], 0)


class MergeNumberOnlySectionTitleTests(unittest.TestCase):
    def test_merges_number_into_title_tag(self) -> None:
        text = "<section>Section 10.6</section> <section>Section 56.4 Election.</section> Text"
        repaired, _ = merge_number_only_section_titles(text)
        self.assertEqual(repaired, "<section>Section 10.6 Section 56.4 Election.</section> Text")

    def test_merges_number_into_untitled_heading(self) -> None:
        text = "<section>6.8</section>\n<section>Section 16(b) Exemption.</section>"
        repaired, _ = merge_number_only_section_titles(text)
        self.assertEqual(repaired, "<section>6.8\nSection 16(b) Exemption.</section>")

    def test_keeps_next_section_of_same_article(self) -> None:
        text = "<section>6.8</section>\n<section>6.9 Notices.</section>"
        repaired, stats = merge_number_only_section_titles(text)
        self.assertEqual(repaired, text)
        self.assertEqual(stats.applied["merge_number_only_section_titles"], 0)


class StrayDecimalSectionTagTests(unittest.TestCase):
    def test_unwraps_fragment_after_title(self) -> None:
        text = "<section>4.5 Litigation</section>\n\n<section>4.6 .</section> As of the date"
        repaired, _ = unwrap_stray_decimal_section_tags(text)
        self.assertEqual(repaired, "<section>4.5 Litigation</section>\n\n4.6 . As of the date")

    def test_unwraps_reference_number(self) -> None:
        text = "<section>2.23 Insurance.</section>\n\n<section>Section 2.24</section>\n\nof the Schedule"
        repaired, _ = unwrap_stray_decimal_section_tags(text)
        self.assertEqual(repaired, "<section>2.23 Insurance.</section>\n\nSection 2.24\n\nof the Schedule")

    def test_unwraps_number_out_of_sequence(self) -> None:
        text = "<section>5.13 Financial Statements</section>\n\n<section>5.15</section> Except"
        repaired, _ = unwrap_stray_decimal_section_tags(text)
        self.assertEqual(repaired, "<section>5.13 Financial Statements</section>\n\n5.15 Except")

    def test_keeps_next_clause_number(self) -> None:
        text = "<section>2.1 the Scheme becoming effective; and</section>\n\n<section>2.2</section> the"
        repaired, stats = unwrap_stray_decimal_section_tags(text)
        self.assertEqual(repaired, text)
        self.assertEqual(stats.applied["unwrap_stray_decimal_section_tags"], 0)

    def test_keeps_first_clause_after_numbered_heading(self) -> None:
        text = "<section>4 Completion</section> <section>4.1</section> Completion shall take place"
        repaired, stats = unwrap_stray_decimal_section_tags(text)
        self.assertEqual(repaired, text)
        self.assertEqual(stats.applied["unwrap_stray_decimal_section_tags"], 0)


class DedupeAdjacentHeadingTests(unittest.TestCase):
    def test_unwraps_first_of_repeated_article_heading(self) -> None:
        text = "<article>ARTICLE VIII</article>\n\n<article>ARTICLE VIII TERMINATION</article>"
        repaired, _ = dedupe_adjacent_same_number_headings(text)
        self.assertEqual(repaired, "ARTICLE VIII\n\n<article>ARTICLE VIII TERMINATION</article>")

    def test_unwraps_first_of_repeated_section_heading(self) -> None:
        text = "<section>Section 6.11. Further Assurances.</section> <section>Section 6.11. Further Assurances.</section>"
        repaired, _ = dedupe_adjacent_same_number_headings(text)
        self.assertEqual(
            repaired,
            "Section 6.11. Further Assurances. <section>Section 6.11. Further Assurances.</section>",
        )

    def test_keeps_consecutive_headings(self) -> None:
        text = "<section>6.11 Further Assurances.</section> <section>6.12 Notices.</section>"
        repaired, stats = dedupe_adjacent_same_number_headings(text)
        self.assertEqual(repaired, text)
        self.assertEqual(stats.applied["dedupe_adjacent_same_number_headings"], 0)


class SameLineCrossArticleSectionTests(unittest.TestCase):
    def test_unwraps_other_article_section_on_same_line(self) -> None:
        text = "<section>Section 9.12 Beneficiaries</section> <section>8.3 .</section> Except"
        repaired, _ = unwrap_same_line_cross_article_section_tags(text)
        self.assertEqual(repaired, "<section>Section 9.12 Beneficiaries</section> 8.3 . Except")

    def test_keeps_section_on_next_line(self) -> None:
        text = "<section>8.12 Beneficiaries</section>\n<section>9.1 Notices</section>"
        repaired, stats = unwrap_same_line_cross_article_section_tags(text)
        self.assertEqual(repaired, text)
        self.assertEqual(stats.applied["unwrap_same_line_cross_article_section_tags"], 0)

    def test_keeps_same_article_section_on_same_line(self) -> None:
        text = "<section>9.12 Beneficiaries</section> <section>9.13 Notices</section>"
        repaired, stats = unwrap_same_line_cross_article_section_tags(text)
        self.assertEqual(repaired, text)
        self.assertEqual(stats.applied["unwrap_same_line_cross_article_section_tags"], 0)


class BareArticleBeforeOtherArticleSectionTests(unittest.TestCase):
    def test_unwraps_bare_article_reference(self) -> None:
        text = "pursuant to\n\n<article>Article VIII</article>\n\n<section>Section 6.5 Notices.</section>"
        repaired, _ = unwrap_bare_article_tags_before_other_article_sections(text)
        self.assertEqual(repaired, "pursuant to\n\nArticle VIII\n\n<section>Section 6.5 Notices.</section>")

    def test_keeps_bare_article_before_its_own_section(self) -> None:
        text = "<article>ARTICLE VI</article>\n\n<section>Section 6.1 Notices.</section>"
        repaired, stats = unwrap_bare_article_tags_before_other_article_sections(text)
        self.assertEqual(repaired, text)
        self.assertEqual(
            stats.applied["unwrap_bare_article_tags_before_other_article_sections"],
            0,
        )

    def test_keeps_titled_article(self) -> None:
        text = "<article>ARTICLE VIII TERMINATION</article>\n\n<section>Section 6.5 Notices.</section>"
        repaired, stats = unwrap_bare_article_tags_before_other_article_sections(text)
        self.assertEqual(repaired, text)
        self.assertEqual(
            stats.applied["unwrap_bare_article_tags_before_other_article_sections"],
            0,
        )


class InsertMissingSectionHeadingTagTests(unittest.TestCase):
    def _insert(self, text: str, article: int, expected: int) -> str:
        repaired, _ = insert_missing_section_heading_tags(
            text,
            [SectionGap(article_num=article, expected=expected, found=expected + 1)],
        )
        return repaired

    def test_skips_mid_sentence_cross_reference(self) -> None:
        text = (
            "<section>2.1 Effect on Stock.</section>\n"
            "Subject to Section 2.2, each share of Merger Partner Common Stock shall be converted.\n"
            "<section>2.3 Exchange.</section>"
        )
        self.assertEqual(self._insert(text, 2, 2), text)

    def test_skips_line_start_cross_reference_followed_by_comma(self) -> None:
        text = (
            "<section>5.10 Notices.</section>\nThe provisions of\n"
            "Section 5.11, however, shall be deemed to amend the Agreement.\n"
            "<section>5.12 Expenses.</section>"
        )
        self.assertEqual(self._insert(text, 5, 11), text)

    def test_skips_number_followed_by_lowercase_word(self) -> None:
        text = (
            "<section>1.39 Material Adverse Effect</section> means, as of the\n"
            "1.40 date hereof; (vii) any changes in applicable Laws\n"
            "<section>1.41 Merger</section>"
        )
        self.assertEqual(self._insert(text, 1, 40), text)

    def test_skips_number_followed_by_parenthetical_or_of(self) -> None:
        for line in (
            "Section 3.20(a) of the Disclosure Schedule is a complete list.",
            "Section 3.20 of the Disclosure Schedule is a complete list.",
            "Section 3.20 and Section 3.21 apply.",
            "3.20) and the Company shall comply.",
        ):
            with self.subTest(line=line):
                text = f"<section>3.19 Taxes.</section>\n{line}\n<section>3.21 Labor.</section>"
                self.assertEqual(self._insert(text, 3, 20), text)

    def test_tags_line_start_heading_and_closes_at_title_sentence(self) -> None:
        text = (
            "<section>2.1 Effect on Stock.</section>\nBody.\n"
            "Section 2.2 Merger Consideration. At the Effective Time, each share shall convert.\n"
            "<section>2.3 Exchange.</section>"
        )
        self.assertEqual(
            self._insert(text, 2, 2),
            (
                "<section>2.1 Effect on Stock.</section>\nBody.\n"
                "<section>Section 2.2 Merger Consideration.</section> At the Effective Time, "
                "each share shall convert.\n<section>2.3 Exchange.</section>"
            ),
        )

    def test_tags_heading_line_without_sentence_break(self) -> None:
        text = (
            "<section>4.1 Organization.</section>\nBody.\n"
            "12 4.2 U.S. Tax Matters\nThe Company has filed all Tax Returns.\n"
            "<section>4.3 Labor.</section>"
        )
        self.assertEqual(
            self._insert(text, 4, 2),
            (
                "<section>4.1 Organization.</section>\nBody.\n"
                "12 <section>4.2 U.S. Tax Matters</section>\nThe Company has filed all Tax Returns.\n"
                "<section>4.3 Labor.</section>"
            ),
        )

    def test_number_alone_on_line_takes_next_line_as_title(self) -> None:
        text = (
            "<section>4.1 Organization.</section>\nBody.\n4.2\nCapitalization.\n"
            "The authorized capital stock consists of shares.\n<section>4.3 Labor.</section>"
        )
        self.assertEqual(
            self._insert(text, 4, 2),
            (
                "<section>4.1 Organization.</section>\nBody.\n<section>4.2\nCapitalization.</section>\n"
                "The authorized capital stock consists of shares.\n<section>4.3 Labor.</section>"
            ),
        )

    def test_tags_heading_after_sentence_end_in_flattened_text(self) -> None:
        text = (
            "<section>1.1 2011 Option Rights</section> has the meaning set out in the background. "
            "1.2 2014 Option Rights. Has the meaning set out in the background. "
            "<section>1.3 2016 Option Rights</section> has the meaning set out in the background."
        )
        self.assertIn(
            "background. <section>1.2 2014 Option Rights.</section> Has the meaning",
            self._insert(text, 1, 2),
        )

    def test_skips_mid_sentence_number_in_flattened_text(self) -> None:
        text = (
            "<section>1.39 Healthcare Laws</section> means the rules promulgated thereunder "
            "1.40 IFRS International Financial Reporting Standards, as in effect from time to time. "
            "<section>1.41 Intellectual Property</section> means all intellectual property."
        )
        self.assertEqual(self._insert(text, 1, 40), text)

    def test_tags_title_case_heading_mid_line_after_table_text(self) -> None:
        text = (
            "<section>1.1 Certain Definitions.</section> Unresolved Claim Section 10.4(d) "
            "1.2 Other Capitalized Terms. For all purposes of this Agreement, capitalized terms apply. "
            "<section>1.3 Certain Interpretations.</section>"
        )
        self.assertIn(
            "10.4(d) <section>1.2 Other Capitalized Terms.</section> For all purposes",
            self._insert(text, 1, 2),
        )

    def test_skips_mid_line_heading_shape_after_lowercase_word(self) -> None:
        text = (
            "<section>2.1 Effect.</section> as provided in 2.2 Merger Consideration. The shares convert. "
            "<section>2.3 Exchange.</section>"
        )
        self.assertEqual(self._insert(text, 2, 2), text)

    def test_tags_untitled_clause_right_after_heading_tag(self) -> None:
        text = (
            "<section>9.13 Waiver of Jury Trial</section> 9.14. EACH OF THE PARTIES HERETO HEREBY "
            "IRREVOCABLY WAIVES ALL RIGHT TO TRIAL BY JURY IN ANY ACTION, PROCEEDING OR COUNTERCLAIM "
            "ARISING OUT OF THIS AGREEMENT.\n<section>9.15 Counterparts.</section>"
        )
        self.assertIn(
            "</section> <section>9.14.</section> EACH OF THE PARTIES",
            self._insert(text, 9, 14),
        )

    def test_tags_clause_label_followed_by_uppercase_word(self) -> None:
        text = (
            "<section>1.1 Contribution.</section>\nBody; and\n\n"
            "1.2 (b) Northwest shall redeem such number of shares of Northwest Common Stock as is equal to "
            "the total number of outstanding shares.\n<section>1.3 Closing.</section>"
        )
        self.assertIn("\n\n<section>1.2</section> (b) Northwest shall redeem", self._insert(text, 1, 2))

    def test_tags_unquoted_defined_term_in_flattened_definitions(self) -> None:
        text = (
            "<section>1.26 Interest Subsidy Loan</section> has the meaning set out in the Recitals, "
            "Section C 1.27 IT Systems has the meaning set out in Section 6.11(a). "
            "<section>1.28 Law</section> means any applicable law."
        )
        self.assertIn(
            "Section C <section>1.27 IT Systems</section> has the meaning set out in Section 6.11(a).",
            self._insert(text, 1, 27),
        )
        text = (
            "<section>1.41 Euro Rate</section> means the rate on such date "
            "1.42 Existing Debt Facilities means (i) the facilities under the agreement. "
            "<section>1.43 Fraud</section> means fraud."
        )
        self.assertIn(
            "such date <section>1.42 Existing Debt Facilities</section> means (i) the facilities",
            self._insert(text, 1, 42),
        )

    def test_long_title_case_heading_sentence_is_kept(self) -> None:
        heading = (
            "Section 13.04. Additional Conditions to the Obligations of the CSC Blockers, Court Square GPs, "
            "Blocker Parents, OpenMail, the Redeemed OM Members and the Protected Rollover Parties."
        )
        text = (
            "<section>Section 13.03 Conditions.</section>\nBody.\n\n"
            f"{heading} The obligation of each party is subject to the following conditions.\n"
            "<section>Section 13.05 Frustration.</section>"
        )
        self.assertIn(f"<section>{heading}</section> The obligation", self._insert(text, 13, 4))

    def test_tags_heading_with_dash_after_number(self) -> None:
        text = (
            "<section>4.1 Organization.</section>\nBody.\n"
            "4.2 - Capitalization. The authorized capital stock consists of shares.\n"
            "<section>4.3 Labor.</section>"
        )
        self.assertEqual(
            self._insert(text, 4, 2),
            (
                "<section>4.1 Organization.</section>\nBody.\n"
                "<section>4.2 - Capitalization.</section> The authorized capital stock consists of shares.\n"
                "<section>4.3 Labor.</section>"
            ),
        )

    def test_line_longer_than_cap_gets_number_only_title(self) -> None:
        long_line = "Capitalization And Other Matters " * 10
        text = f"<section>4.1 Organization.</section>\nSection 4.2 {long_line}\n<section>4.3 Labor.</section>"
        self.assertEqual(
            self._insert(text, 4, 2),
            f"<section>4.1 Organization.</section>\n<section>Section 4.2</section> {long_line}\n"
            "<section>4.3 Labor.</section>",
        )

    def test_untitled_clause_gets_number_only_title(self) -> None:
        text = (
            "<section>8.1 SUI</section>\nBody.\n"
            "8.2 All references in this Agreement to the knowledge of the seller shall refer to the actual\n"
            "knowledge of its officers.\n<section>8.3 Taxes.</section>"
        )
        self.assertEqual(
            self._insert(text, 8, 2),
            (
                "<section>8.1 SUI</section>\nBody.\n"
                "<section>8.2</section> All references in this Agreement to the knowledge of the seller "
                "shall refer to the actual\nknowledge of its officers.\n<section>8.3 Taxes.</section>"
            ),
        )

    def test_defined_term_heading_closes_after_term(self) -> None:
        text = (
            "<section>1.69 “Tax Return”</section> means any return.\n"
            "1.70 “UCC” means the Uniform Commercial Code of the State of New York, or any "
            "corresponding or succeeding provisions of Laws of the State of New York.\n"
            "<section>1.71 “U.S. GAAP”</section> means generally accepted accounting principles."
        )
        repaired = self._insert(text, 1, 70)
        self.assertIn(
            "<section>1.70 “UCC”</section> means the Uniform Commercial Code",
            repaired,
        )

    def test_split_combined_ignores_cross_reference_inside_tag(self) -> None:
        text = "<section>2.1 Effect on Stock. Subject to Section 2.2, each share converts.</section>"
        repaired, stats = split_combined_missing_section_tags(
            text,
            [SectionGap(article_num=2, expected=2, found=3)],
        )
        self.assertEqual(repaired, text)
        self.assertEqual(stats.applied["split_combined_missing_section_tags"], 0)


class GluedHeadingTitleTests(unittest.TestCase):
    def test_splits_glued_roman_article_titles(self) -> None:
        cases = {
            "ARTICLE IVREPRESENTATIONS AND WARRANTIES OF PARENT": "ARTICLE IV REPRESENTATIONS AND WARRANTIES OF PARENT",
            "ARTICLE VIITERMINATION, AMENDMENT AND WAIVER": "ARTICLE VII TERMINATION, AMENDMENT AND WAIVER",
            "ARTICLE VICOVENANTS OF PARENT": "ARTICLE VI COVENANTS OF PARENT",
            "ARTICLE IXMISCELLANEOUS": "ARTICLE IX MISCELLANEOUS",
            "ARTICLE IDEFINITIONS": "ARTICLE I DEFINITIONS",
            "ARTICLE XCLOSING": "ARTICLE X CLOSING",
            "ARTICLE XIGENERAL PROVISIONS": "ARTICLE XI GENERAL PROVISIONS",
            "ARTICLE IIIMERGER CONSIDERATION": "ARTICLE III MERGER CONSIDERATION",
            "ARTICLE VIITermination, Amendment and Waiver": "ARTICLE VII Termination, Amendment and Waiver",
            "Article IVRepresentations": "Article IV Representations",
        }
        for glued, expected in cases.items():
            with self.subTest(glued=glued):
                self.assertEqual(normalize_glued_article_title(glued), expected)

    def test_resolves_numeral_letter_ambiguity_with_title_words(self) -> None:
        self.assertEqual(
            normalize_glued_article_title("ARTICLE VIIINDEMNIFICATION"),
            "ARTICLE VII INDEMNIFICATION",
        )
        self.assertEqual(
            normalize_glued_article_title("ARTICLE VIIIIndemnification"),
            "ARTICLE VIII Indemnification",
        )
        self.assertEqual(
            normalize_glued_article_title("ARTICLE XIIIINTERPRETATION"),
            "ARTICLE XIII INTERPRETATION",
        )

    def test_uses_expected_number_for_unknown_numeral_letter_words(self) -> None:
        self.assertEqual(normalize_glued_article_title("ARTICLE IIIXYLO", 3), "ARTICLE III XYLO")
        self.assertEqual(normalize_glued_article_title("ARTICLE IIIXYLO", 4), "ARTICLE IIIXYLO")

    def test_splits_glued_arabic_and_keyword_numbers(self) -> None:
        self.assertEqual(normalize_glued_article_title("ARTICLE 4DEFINITIONS"), "ARTICLE 4 DEFINITIONS")
        self.assertEqual(
            normalize_glued_article_title("ARTICLE 10General Provisions"),
            "ARTICLE 10 General Provisions",
        )
        self.assertEqual(normalize_glued_article_title("ARTICLEVII TERMINATION"), "ARTICLE VII TERMINATION")
        self.assertEqual(normalize_glued_article_title("ARTICLE1 DEFINITIONS"), "ARTICLE 1 DEFINITIONS")

    def test_leaves_valid_and_non_glued_titles_alone(self) -> None:
        for title in (
            "ARTICLE XI",
            "ARTICLE VIII TERMINATION",
            "ARTICLE IIIA REPRESENTATIONS AND WARRANTIES OF GUARANTOR",
            "ARTICLE IVA",
            "ARTICLE 4A",
            "ARTICLE IV. REPRESENTATIONS",
            "ARTICLE INDEMNIFICATION",
            "ARTICLE MISCELLANEOUS",
            "ARTICLE VOTING AGREEMENT",
            "ARTICLES",
            "Article IXdefinitions, Construction",
        ):
            with self.subTest(title=title):
                self.assertEqual(normalize_glued_article_title(title), title)

    def test_splits_glued_section_titles(self) -> None:
        self.assertEqual(normalize_glued_section_title("Section 5.1Closing"), "Section 5.1 Closing")
        self.assertEqual(
            normalize_glued_section_title("5.06Acquisition Proposals."),
            "5.06 Acquisition Proposals.",
        )
        for title in ("Section 4.1A Foo", "Section 5.1(a) Closing", "Section 5.1 Closing"):
            with self.subTest(title=title):
                self.assertEqual(normalize_glued_section_title(title), title)


if __name__ == "__main__":
    _ = unittest.main()
