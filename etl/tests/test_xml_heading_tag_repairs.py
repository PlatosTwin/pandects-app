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
    repair_sentence_split_section_titles,
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
            "<section>1.39 Healthcare Laws.</section> The Company complies with the rules promulgated "
            "thereunder 1.40 Financial Reporting Standards, as in effect from time to time. "
            "<section>1.41 Intellectual Property.</section> The Company owns its property."
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

    def test_caps_definition_after_lowercase_word_in_definitions_run(self) -> None:
        text = (
            "<section>1.39 Healthcare Laws</section> means the rules promulgated thereunder "
            "1.40 IFRS International Financial Reporting Standards, as in effect from time to time. "
            "<section>1.41 Intellectual Property</section> means all intellectual property."
        )
        self.assertIn(
            "thereunder <section>1.40 IFRS</section> International Financial Reporting",
            self._insert(text, 1, 40),
        )

    def test_caps_term_outside_definitions_run_is_skipped(self) -> None:
        text = (
            "<section>4.1 Organization.</section> The Company files with the agency under "
            "4.2 SEC Reports filed from time to time. <section>4.3 Labor.</section>"
        )
        self.assertEqual(self._insert(text, 4, 2), text)

    def test_cross_reference_word_before_number_is_skipped(self) -> None:
        text = (
            "<section>1.1 Definitions.</section> \u201cTax\u201d has the meaning given in "
            "1.2 \u201cTax Return\u201d means any return. <section>1.3 Other.</section>"
        )
        self.assertEqual(self._insert(text, 1, 2), text)

    def test_digit_led_unquoted_defined_term(self) -> None:
        text = (
            "<section>1.1 2011 Option Rights</section> has the meaning set out in the background. "
            "1.2 2014 Option Rights has the meaning set out in the background. "
            "<section>1.3 2016 Option Rights</section> has the meaning set out in the background."
        )
        self.assertIn(
            "background. <section>1.2 2014 Option Rights</section> has the meaning",
            self._insert(text, 1, 2),
        )

    def test_defined_term_absorbs_trailing_period_or_colon(self) -> None:
        text = (
            "<section>Section 11.10 Notices.</section>\nBody.\n"
            "Section 11.11 \u201cKnowledge\u201d. As used in Article IV hereof, all references apply.\n"
            "<section>Section 11.12 Other.</section>"
        )
        self.assertIn("<section>Section 11.11 \u201cKnowledge\u201d.</section> As used", self._insert(text, 11, 11))
        text = (
            "<section>1.164 \u201cTax\u201d</section> means tax.\n"
            "1.165 \u201cTransaction Documents\u201d: shall mean this Agreement and each other document.\n"
            "<section>1.166 \u201cTransfer Taxes\u201d</section> means taxes."
        )
        self.assertIn(
            "<section>1.165 \u201cTransaction Documents\u201d:</section> shall mean",
            self._insert(text, 1, 165),
        )

    def test_two_term_definitions_keep_both_terms(self) -> None:
        text = (
            "<section>11.43 \u201cPerson\u201d</section> means any person.\n"
            "11.44. \u201cParty\u201d and \u201cParties\u201d shall have the meaning set forth in the preamble.\n"
            "<section>11.45 \u201cPermits\u201d</section> means permits."
        )
        self.assertIn(
            "<section>11.44. \u201cParty\u201d and \u201cParties\u201d</section> shall have the meaning",
            self._insert(text, 11, 44),
        )
        text = (
            "<section>1.167 \u201cTax\u201d</section> means tax.\n"
            "1.168 \u201cU.S. GAAP\u201d or \u201cGAAP\u201d means U.S. generally accepted accounting "
            "principles.\n<section>1.169 \u201cWARN\u201d</section> means the act."
        )
        self.assertIn(
            "<section>1.168 \u201cU.S. GAAP\u201d or \u201cGAAP\u201d</section> means U.S.",
            self._insert(text, 1, 168),
        )

    def test_title_case_run_printed_without_period(self) -> None:
        cases = [
            (
                5,
                2,
                "Section 5.2 Merger Without a Stockholders\u2019 Meeting As promptly as practicable after "
                "the Acceptance Time, the parties shall take all necessary action to cause the Merger.",
                "<section>Section 5.2 Merger Without a Stockholders\u2019 Meeting</section> As promptly",
            ),
            (
                7,
                2,
                "7.2 Coeur\u2019s indemnity Coeur agrees with Palmarejo to indemnify Palmarejo against all "
                "losses arising from any breach of this agreement by Coeur.",
                "<section>7.2 Coeur\u2019s indemnity</section> Coeur agrees",
            ),
            (
                1,
                3,
                "1.3 Number and Gender In this Agreement, unless the contrary intention appears, words "
                "importing the singular include the plural.",
                "<section>1.3 Number and Gender</section> In this Agreement",
            ),
            (
                1,
                1,
                "1.1 Defined terms In this document:",
                "<section>1.1 Defined terms</section> In this document:",
            ),
        ]
        for article, number, line, expected in cases:
            with self.subTest(line=line[:30]):
                text = (
                    f"<section>{article}.{number - 1 if number > 1 else 0} Previous.</section>\nBody.\n"
                    f"{line}\n<section>{article}.{number + 1} Next.</section>"
                )
                self.assertIn(expected, self._insert(text, article, number))

    def test_proper_noun_subject_stays_number_only(self) -> None:
        text = (
            "<section>1.5 Kabel BW</section>\nBody.\n"
            "1.6 Kabel BW Holdings is the sole limited partner of the partnership and holds all interests "
            "in the partnership as of the date of this agreement.\n<section>1.7 Next.</section>"
        )
        self.assertIn("<section>1.6</section> Kabel BW Holdings is", self._insert(text, 1, 6))

    def test_short_prose_lead_ins_get_number_only_titles(self) -> None:
        for line, expected in (
            (
                "1.2 In this Agreement, unless the context otherwise requires \u2013",
                "<section>1.2</section> In this Agreement",
            ),
            ("1.2 If to Buyer:", "<section>1.2</section> If to Buyer:"),
        ):
            with self.subTest(line=line):
                text = f"<section>1.1 Definitions.</section>\nBody.\n{line}\n<section>1.3 Next.</section>"
                self.assertIn(expected, self._insert(text, 1, 2))

    def test_unicode_spaces_after_number(self) -> None:
        for space in ("\u2003", "\u2002", "\u2009", "\u00a0"):
            with self.subTest(space=repr(space)):
                text = (
                    "<section>4.1 Organization.</section>\nBody.\n"
                    f"4.2{space}Capitalization.{space}The authorized capital stock consists of shares.\n"
                    "<section>4.3 Labor.</section>"
                )
                self.assertIn(f"<section>4.2{space}Capitalization.</section>", self._insert(text, 4, 2))

    def test_newline_after_number_is_not_a_follower_space(self) -> None:
        text = (
            "<section>2.1 Effect.</section>\nas described in Section 2.2\n(a) of the Disclosure "
            "Schedule.\n<section>2.3 Exchange.</section>"
        )
        self.assertEqual(self._insert(text, 2, 2), text)

    def test_sentence_start_requires_title_case_heading(self) -> None:
        text = (
            "<section>5.10 Notices.</section> Each party shall comply with this Agreement. "
            "Section 5.11 Disclosure Schedule sets forth all items. <section>5.12 Expenses.</section>"
        )
        self.assertEqual(self._insert(text, 5, 11), text)

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

    def test_leaves_unknown_numeral_letter_words_unsplit(self) -> None:
        for title, expected_number in (
            ("ARTICLE IIIXYLO", 3),
            ("ARTICLE IIIXYLO", 4),
            ("ARTICLE IIXYLOPHONE", 2),
        ):
            with self.subTest(title=title):
                self.assertEqual(normalize_glued_article_title(title, expected_number), title)

    def test_splits_newly_listed_numeral_letter_words(self) -> None:
        cases = {
            "ARTICLE IIIDENTIFICATION OF ASSETS": "ARTICLE II IDENTIFICATION OF ASSETS",
            "ARTICLE VIIILLEGALITY": "ARTICLE VII ILLEGALITY",
            "ARTICLE IVESTING": "ARTICLE I VESTING",
            "ARTICLE XINTEGRATION": "ARTICLE X INTEGRATION",
            "ARTICLE IIMPORTANT NOTICES": "ARTICLE I IMPORTANT NOTICES",
        }
        for glued, expected in cases.items():
            with self.subTest(glued=glued):
                self.assertEqual(normalize_glued_article_title(glued), expected)

    def test_letter_suffix_ambiguity_is_left_alone(self) -> None:
        self.assertEqual(normalize_glued_article_title("ARTICLE 4ATAXES"), "ARTICLE 4ATAXES")
        self.assertEqual(normalize_glued_article_title("ARTICLE IIIATAXES"), "ARTICLE IIIATAXES")
        self.assertEqual(
            normalize_glued_article_title("ARTICLE 4ADDITIONAL AGREEMENTS"),
            "ARTICLE 4 ADDITIONAL AGREEMENTS",
        )

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


class SentenceSplitSectionTitleTests(unittest.TestCase):
    def _repair(self, text: str) -> str:
        repaired, _ = repair_sentence_split_section_titles(text)
        self.assertEqual(
            repaired.replace("<section>", "").replace("</section>", ""),
            text.replace("<section>", "").replace("</section>", ""),
        )
        return repaired

    def test_split_at_curly_apostrophe_gets_number_only_title(self) -> None:
        # f28b4547 (ESCO), page 45, as tagged.
        text = (
            "<article>23 Cumulative Rights</article>\n\n<section>23.1</section> The rights and remedies of any "
            "party expressly conferred by this Agreement are cumulative.\n\n<section>23.2 A party\u2019</section>s "
            "exercise of any right or discretion conferred on it under or in connection with this Agreement shall "
            "be a matter of its absolute discretion"
        )
        repaired, stats = repair_sentence_split_section_titles(text)
        self.assertIn("<section>23.2</section> A party\u2019s exercise of any right", repaired)
        self.assertIn("<section>23.1</section> The rights", repaired)
        self.assertEqual(stats.applied["repair_sentence_split_section_titles"], 1)

    def test_split_at_cross_reference_number_gets_number_only_title(self) -> None:
        # bb29051b (Oval/Friary), page 25, as tagged.
        text = (
            "<article>15. ASSIGNMENT</article>\n\n<section>15.1 Subject to clauses</section> 15.2 and 15.3, no party "
            "may assign or grant any encumbrance over the benefit of this agreement"
        )
        self.assertIn(
            "<section>15.1</section> Subject to clauses 15.2 and 15.3, no party may assign",
            self._repair(text),
        )

    def test_untitled_clause_sentence_gets_number_only_title(self) -> None:
        text = "<section>14.2 No Party may assign</section> its rights and obligations under this Agreement."
        self.assertIn("<section>14.2</section> No Party may assign its rights", self._repair(text))

    def test_correct_titles_are_untouched(self) -> None:
        for text in (
            "<section>Section 4.2 No Conflict.</section> The execution of this Agreement does not conflict.",
            "<section>1.26 Interest Subsidy Loan</section> has the meaning given in Clause 4.",
            "<section>4.1 Organization</section> and Good Standing. The Company is duly organized.",
            "<section>1.12 \u201cPlans\u201d</section> shall have the meaning set forth in Section 3.",
            "<section>2.4 Non-U.S.</section> Plans. Each Non-U.S. Plan is in compliance.",
            "<section>3.1 Purchase Price</section>\n\nThe purchase price is $10.",
            "<section>5.2</section> The Buyer shall pay the price.",
            "<section>7.1 Conditions</section> (a) the approval of the shareholders.",
        ):
            with self.subTest(text=text):
                repaired, stats = repair_sentence_split_section_titles(text)
                self.assertEqual(repaired, text)
                self.assertEqual(stats.applied["repair_sentence_split_section_titles"], 0)

    def test_heading_cut_short_at_apostrophe_is_extended(self) -> None:
        # 616b8f4f and c27a3e9e.
        self.assertIn(
            "<section>7.5 Purchaser\u2019s Receipt of Approvals of Governmental Authorities.</section> The Purchaser",
            self._repair(
                "<section>7.5 Purchaser\u2019</section>s Receipt of Approvals of Governmental Authorities. "
                "The Purchaser shall have received all approvals."
            ),
        )
        self.assertIn(
            "<section>1.19 Buyer\u2019s Representative</section> means Eric",
            self._repair("<section>1.19 Buyer\u2019</section>s Representative means Eric Ramsay."),
        )

    def test_line_end_heading_cut_short_is_extended(self) -> None:
        self.assertIn(
            "<section>3.1 Purchase Price and payment thereof</section>\n\nSubject to",
            self._repair("<section>3.1 Purchase Price and payment</section> thereof\n\nSubject to the terms."),
        )

    def test_extension_never_starts_with_a_cross_reference(self) -> None:
        self.assertIn(
            "<section>11.14</section> Clause 11.12 and 11.13 apply notwithstanding",
            self._repair("<section>11.14 Clause 11.12 and 11.13 apply</section> notwithstanding Clause 11.11.\n\n"),
        )
        self.assertIn(
            "<section>2.2</section> Annex 2.2 sets forth",
            self._repair("<section>2.2 Annex 2.</section>2 sets forth a list of the Exhibits to this Agreement.\n\n"),
        )

    def test_title_printed_before_sentence_is_kept(self) -> None:
        self.assertIn(
            "<section>1.2 Interpretation</section> In this agreement, headings are",
            self._repair("<section>1.2 Interpretation In this agreement, headings</section> are for convenience only."),
        )
        self.assertIn(
            "<section>3.2 Sellers\u2019 obligations</section> On Closing the Sellers",
            self._repair("<section>3.2 Sellers\u2019 obligations On Closing</section> the Sellers will:\n\n(a) deliver"),
        )
        self.assertIn(
            "<section>20.4 Confidentiality</section> - The parties understand",
            self._repair("<section>20.4 Confidentiality - The parties</section> understand and agree that"),
        )

    def test_unquoted_definition_keeps_term_as_title(self) -> None:
        self.assertIn(
            "<section>1.48 Proprietary IPR</section> means the IPR",
            self._repair("<section>1.48 Proprietary IPR means</section> the IPR set out in Appendix 11."),
        )

    def test_run_of_names_is_not_a_title(self) -> None:
        self.assertIn(
            "<section>4.1</section> Orezone and New Orezone hereby covenant",
            self._repair("<section>4.1 Orezone and New Orezone hereby covenant and agree with IAMGOLD</section> that, prior"),
        )

    def test_trailing_reference_prefix_must_be_a_title(self) -> None:
        self.assertIn(
            "<section>10.5 Taxes</section> Section 10.3 shall",
            self._repair("<section>10.5 Taxes Section 10.</section>3 shall have no application."),
        )
        self.assertIn(
            "<section>16.1</section> Subject to Clause 16.2 the",
            self._repair("<section>16.1 Subject to Clause 16.</section>2 the Seller shall indemnify."),
        )
