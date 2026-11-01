# pyright: reportAny=false
import unittest

from etl.domain.heading_titles import (
    normalize_glued_article_title,
    normalize_glued_section_title,
)
from etl.domain.sentence_split_titles import repair_sentence_split_section_titles
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

    def test_title_glued_to_first_word_of_text_is_split(self) -> None:
        # 448ba341, page 71: the source prints "Calculation of LossesNotwithstanding".
        self.assertIn(
            "<section>Section 11.5. Calculation of Losses</section>Notwithstanding anything contained",
            self._repair(
                "<section>Section 11.5. Calculation of LossesNotwithstanding</section> anything contained in this "
                "Agreement, all materiality qualifications shall be disregarded."
            ),
        )
        self.assertEqual(
            repair_sentence_split_section_titles("<section>3.4 Notice to McDonald</section>\n\nThe Seller.")[0],
            "<section>3.4 Notice to McDonald</section>\n\nThe Seller.",
        )

    def test_title_case_subject_followed_by_its_verb_is_not_a_title(self) -> None:
        # f28b4547 and bb29051b: "The Buyer" + " may ...", "Each of the Warranties" + " is ...".
        self.assertIn("<section>7.9</section> The Buyer may", self._repair("<section>7.9 The Buyer</section> may (in addition) rescind."))
        self.assertIn(
            "<section>8.1</section> Each of the Institutional Sellers severally warrants",
            self._repair("<section>8.1 Each of the Institutional Sellers</section> severally warrants to the Buyer that"),
        )
        self.assertIn(
            "<section>6.1</section> Completion of this agreement shall",
            self._repair("<section>6.1 Completion of</section> this agreement shall take place on the Completion Date"),
        )

    def test_cross_references_tagged_as_headings_are_left_for_unwrap_repairs(self) -> None:
        for text in (
            "except as set forth in <section>Section 3.11 of the Disclosure Schedules describes</section> all services",
            "<section>Section 3.11 of the Disclosure Schedules describes</section> all material affiliated services",
            "<section>Section 13.2, Section 13.</section>3, Section 15.17 and Section 15.19(c), and any claim",
            "<section>4.6, no representation</section> or warranty is made by Parent",
        ):
            with self.subTest(text=text):
                self.assertEqual(repair_sentence_split_section_titles(text)[0], text)

    def test_complete_title_case_heading_is_kept_before_its_verb(self) -> None:
        text = "<section>Section 4.10. Related Person Transactions.</section> there are no Contracts with"
        self.assertEqual(repair_sentence_split_section_titles(text)[0], text)

    def test_run_in_heading_before_party_subject_keeps_heading(self) -> None:
        self.assertIn(
            "<section>Section 4.7 Financing</section> Parent has delivered",
            self._repair("<section>Section 4.7 Financing Parent</section> has delivered to the Company the Commitment Letters."),
        )
        self.assertIn(
            "<section>5.5</section> Sellers and Asset Sellers shall",
            self._repair("<section>5.5 Sellers and Asset Sellers</section> shall to the extent commercially reasonable"),
        )
        self.assertIn(
            "<section>6.5</section> Seller shall",
            self._repair("<section>6.5 Seller</section> shall deliver the Shares."),
        )

    def test_glued_title_before_party_or_short_phrase_is_split(self) -> None:
        self.assertIn(
            "<section>5.6. Spin-Out</section>Parent shall take",
            self._repair("<section>5.6. Spin-OutParent</section> shall take the following actions"),
        )
        self.assertIn(
            "<section>2.4. SRx Meeting</section>Subject to the terms",
            self._repair("<section>2.4. SRx MeetingSubject to</section> the terms of this Agreement:"),
        )
        self.assertIn(
            "<section>6.4. Notice and Cure Provisions</section>Each Party will give",
            self._repair("<section>6.4. Notice and Cure ProvisionsEach Party</section> will give prompt notice"),
        )


class SentenceSplitHeadingGuardTests(unittest.TestCase):
    """Real headings the repair must keep (adversarial review of PR #229)."""

    def _title(self, inner: str, follow: str, raw: str | None = None) -> str:
        text = f"<section>{inner}</section>{follow}"
        repaired, _ = repair_sentence_split_section_titles(text, raw)
        self.assertEqual(
            repaired.replace("<section>", "").replace("</section>", ""),
            text.replace("<section>", "").replace("</section>", ""),
        )
        return repaired[len("<section>") : repaired.index("</section>")]

    def test_heading_ending_in_a_period_inside_the_tag_is_kept(self) -> None:
        cases = {
            # 0ba152c0, and synthetic repros.
            ("Section 6.11. Guarantee. RTL hereby absolutely", " and unconditionally guarantees"): "Section 6.11. Guarantee.",
            ("Section 5.4 Access to Information. The Company", " shall afford Parent access"): (
                "Section 5.4 Access to Information."
            ),
            ("9.3 Assignment. Neither party", " may assign this Agreement"): "9.3 Assignment.",
        }
        for (inner, follow), title in cases.items():
            with self.subTest(inner=inner):
                self.assertEqual(self._title(inner, follow), title)

    def test_complete_heading_ending_in_a_period_is_untouched(self) -> None:
        # 154f6f77, 2a1dda61, 2f02f3b6, a918ecb6, c3e4cc2c, 642f01cf, 0d8a04b8.
        for inner, follow in (
            ("5.7 eSpeed SEC Documents.", " eSpeed has filed all reports"),
            ("Section 5.3 cDistro Stockholder Approval.", " cDistro shall call a meeting"),
            ("5.16 Filing of ev3 Charter Amendment.", " ev3 shall file the amendment"),
            ("Section 2.14 iDcentrix Schedules.", " iDcentrix has delivered the schedules"),
            ("3.22 Broker’ s, Finder’ s, or Other Fees.", " no broker is entitled to a fee"),
            ("7.03 Conditions to aaiPharma's Obligation.", " the obligation of aaiPharma is subject"),
            ("3.8. 2021 Dividend (up to the Closing Date) declaration and distribution.", " the Company shall pay"),
        ):
            with self.subTest(inner=inner):
                self.assertEqual(self._title(inner, follow), inner)

    def test_number_only_needs_a_sentence_signal(self) -> None:
        for inner, follow in (
            ("7.1 Completion Matters", " relating to the Business"),
            ("5.6 iPhone Matters relating to", " the iPhone business"),
            ("12.4 Termination", " under this Article 12 shall not become effective"),
            ("3.4 Interest", " accrued on the Retention Amount"),
        ):
            with self.subTest(inner=inner):
                self.assertEqual(self._title(inner, follow), inner)
        self.assertEqual(self._title("10.2 No Seller", " shall make any claim"), "10.2")
        self.assertEqual(self._title("2.3 NewCompany", " shall be formed by the Sellers"), "2.3")
        self.assertEqual(self._title("9.7 No Majority Seller", " shall be liable"), "9.7")

    def test_defined_terms_keep_their_title(self) -> None:
        # 936e9204 and 50a57067.
        self.assertEqual(self._title("1.7 Effective Date", " is the date and time at which"), "1.7 Effective Date")
        self.assertEqual(self._title("1.13 Permitted Liens", " are any retention of title"), "1.13 Permitted Liens")
        self.assertEqual(self._title("1.12 Know-How is", " all technical information"), "1.12 Know-How")
        self.assertEqual(self._title("1.25 Escrow Agent is defined in", " clause 3.4."), "1.25 Escrow Agent")

    def test_run_in_heading_ends_before_an_opener_or_the_subject(self) -> None:
        cases = {
            ("11.6. Severability In the event that", " any provision is held invalid"): "11.6. Severability",
            ("Section 9.01. Survival None of the representations", ", warranties and covenants"): "Section 9.01. Survival",
            ("Section 4.25 Cybersecurity To the Knowledge of the Parent, no third party", " has gained access"): (
                "Section 4.25 Cybersecurity"
            ),
            ("3.2 Taxes Except", " as set forth in the Disclosure Letter"): "3.2 Taxes",
            ("3.1 Corporate Organization The Company", " is a corporation duly organized"): "3.1 Corporate Organization",
            ("Section 4.1 Organization; Authority The undersigned Shareholder", " has full power"): (
                "Section 4.1 Organization; Authority"
            ),
            ("Section 8.2 Procedures and Effect of Termination Any Party desiring to terminate", " this Agreement"): (
                "Section 8.2 Procedures and Effect of Termination"
            ),
            ("Section 3.11 No Other Representations Except for the representations", " and warranties"): (
                "Section 3.11 No Other Representations"
            ),
            ("7.6 No Restraints No temporary restraining", " order shall be in effect"): "7.6 No Restraints",
            ("Section 9.5 No Material Adverse Effect No Effect shall exist", " that has had"): (
                "Section 9.5 No Material Adverse Effect"
            ),
            ("3.12 No Undisclosed Liabilities, (Except for liabilities", " incurred in the ordinary course)"): (
                "3.12 No Undisclosed Liabilities"
            ),
            # 11f7e478, 679f2048, 895406d2, 8b8d6bc8, 40f4ab4d, 802dd65d.
            ("5.9. No Other Representations Buyer", " acknowledges that the Seller makes"): "5.9. No Other Representations",
            ("6.2. Authority Relative to this Agreement Target", " has full corporate power"): (
                "6.2. Authority Relative to this Agreement"
            ),
            ("7.18 Target Company Designees Buyer", " shall cause the designees"): "7.18 Target Company Designees",
            ("7.21 Additional Voting Agreements Parent will use", " its reasonable best efforts"): (
                "7.21 Additional Voting Agreements"
            ),
            # 5fc93c94, bae335ef, 123b28b7: a repeat of the run's last words.
            ("Section 4.18 Organization of Merger Sub Merger Sub", " is a corporation duly organized"): (
                "Section 4.18 Organization of Merger Sub"
            ),
            ("5.9. Operations and Ownership of Acquisition Sub Acquisition Sub", " was formed solely"): (
                "5.9. Operations and Ownership of Acquisition Sub"
            ),
            # 28ac6afd, 297cb7a8.
            ("7.20 Delivery of Century Bank Disclosure Memorandum Century", " shall deliver to Buyer"): (
                "7.20 Delivery of Century Bank Disclosure Memorandum"
            ),
            ("12.1 Vendor’s Conditions The Vendor’s obligations to complete", " are conditional on"): (
                "12.1 Vendor’s Conditions"
            ),
        }
        for (inner, follow), title in cases.items():
            with self.subTest(inner=inner):
                self.assertEqual(self._title(inner, follow), title)

    def test_layout_keeps_bold_underlined_or_own_line_heading(self) -> None:
        # bb80b652: the heading is underlined in its own cell.
        inner, follow = "12.4 Amendments and waivers Changes and additions", " to this Agreement must be in writing."
        raw = (
            "<tr><td>12.4</td><td><u>Amendments and waivers</u></td></tr>"
            "<tr><td>Changes and additions to this Agreement must be in writing.</td></tr>"
        )
        self.assertEqual(self._title(inner, follow), inner)
        self.assertEqual(self._title(inner, follow, raw), "12.4 Amendments and waivers")
        # a80a7eae: "17.3 Separate undertakings" in bold, then the clause.
        raw = (
            "<tr><td><b>17.3</b></td><td><b>Separate undertakings</b></td></tr>"
            "<tr><td>Clauses 17.1.1, 17.1.2 have effect as separate undertakings.</td></tr>"
        )
        self.assertEqual(
            self._title("17.3 Separate undertakings Clauses", " 17.1.1, 17.1.2 have effect as separate undertakings.", raw),
            "17.3 Separate undertakings",
        )
        # 348443d0: an underlined heading before an all-caps bold clause.
        raw = (
            "<p>13.20 <u><b>Disclosure</b></u><b> IF THE DISCLOSURE SUMMARY REQUIRED BY SECTION 720.401, "
            "FLORIDA STATUTES, HAS NOT BEEN PROVIDED</b></p>"
        )
        self.assertEqual(
            self._title(
                "13.20 Disclosure IF THE DISCLOSURE SUMMARY REQUIRED BY SECTION 720.",
                "401, FLORIDA STATUTES, HAS NOT BEEN PROVIDED",
                raw,
            ),
            "13.20 Disclosure",
        )
        # A bold defined subject is not a heading: "Seller" + " shall ...".
        raw = "<p>6.5 <b>Seller</b> shall deliver the Shares.</p>"
        self.assertEqual(self._title("6.5 Seller", " shall deliver the Shares.", raw), "6.5")

    def test_glued_opener_with_a_long_tail_is_split(self) -> None:
        # bf60dea1.
        self.assertEqual(
            self._title(
                "6.3. Additional Conditions Precedent in Favor of SRxThe obligation of SRx to complete",
                " the Arrangement is subject to",
            ),
            "6.3. Additional Conditions Precedent in Favor of SRx",
        )

    def test_title_never_takes_in_a_quoted_term(self) -> None:
        self.assertNotIn(
            "“",
            self._title("1.7. JBT’s board of directors", " (the “JBT Board”) has, based on the information"),
        )

    def test_defined_term_variants_keep_their_title(self) -> None:
        # 50a57067 and 936e9204.
        self.assertEqual(self._title("1.15 Claim Notice is defined in clause 10.", "4."), "1.15 Claim Notice")
        self.assertEqual(
            self._title("1.18 Vanessa Licensed Business", " shall include the development, design and testing"),
            "1.18 Vanessa Licensed Business",
        )

    def test_untitled_clauses_still_get_a_number_only_title(self) -> None:
        for inner, follow, title in (
            ("9.3 Breach of the Sellers' Warranties", " shall constitute the only valid basis for any claims", "9.3"),
            ("4.7 Amounts", " may only be released to the Sellers from the Retention Account", "4.7"),
            ("10.10 Parties", " agree that the sale of the Transferring Activities", "10.10"),
            ("11.3 No amendment", " of this Agreement shall be effective unless it is made in writing", "11.3"),
        ):
            with self.subTest(inner=inner):
                self.assertEqual(self._title(inner, follow), title)

    def test_extension_never_takes_in_the_next_section_number(self) -> None:
        for inner, follow in (
            ("1.11 Allocation of Payment", " 1.12\n\nPurchase Price 1.13"),
            ("Section 5.05 Governmental Authorities; Consents", " 4.1\n\nSection 5.06 Brokers"),
        ):
            with self.subTest(inner=inner):
                self.assertEqual(self._title(inner, follow), inner)
        self.assertEqual(
            self._title("Section 7.04 Conditions to Obligations of Plum Creek to Effect", " the Merger ------\n\nThe"),
            "Section 7.04 Conditions to Obligations of Plum Creek to Effect the Merger",
        )


class SentenceSplitReReviewTests(unittest.TestCase):
    """Second review of PR #229."""

    def _title(self, inner: str, follow: str, raw: str | None = None, page: str = "") -> str:
        text = f"{page}<section>{inner}</section>{follow}"
        repaired, _ = repair_sentence_split_section_titles(text, raw)
        self.assertEqual(
            repaired.replace("<section>", "").replace("</section>", ""),
            text.replace("<section>", "").replace("</section>", ""),
        )
        start = repaired.index("<section>", len(page)) + len("<section>")
        return repaired[start : repaired.index("</section>", start)]

    def test_bold_sentence_is_not_a_layout_heading(self) -> None:
        for inner, follow, sentence in (
            ("10.1 No party", " may assign this Agreement.", "10.1 No party may assign this Agreement."),
            ("4.1 Each Seller", " warrants to the Buyer as follows.", "4.1 Each Seller warrants to the Buyer as follows."),
            ("6.1 Completion", " shall take place on the Completion Date.", "6.1 Completion shall take place on the Completion Date."),
            ("12.4 The Bidder", " values the Company's SAP applications.", "12.4 The Bidder values the Company's SAP applications."),
        ):
            with self.subTest(inner=inner):
                raw = f"<p><b>{sentence}</b></p><p>Next paragraph.</p>"
                self.assertEqual(self._title(inner, follow + "\n\nNext paragraph.", raw), inner.split()[0])

    def test_bold_statutory_reference_heading_is_kept(self) -> None:
        # baf68ca1.
        raw = "<p>3.22 <b>Section 13.1-728.2 et seq. of the VSCA.</b> The Company Board has taken all action.</p>"
        self.assertEqual(
            self._title("3.22 Section 13.", "1-728.2 et seq. of the VSCA. The Company Board has taken all action.", raw),
            "3.22 Section 13.1-728.2 et seq. of the VSCA.",
        )

    def test_determiner_is_never_a_defined_term(self) -> None:
        # 564dbf71, 70eaaf56, bb680dd1.
        for inner, follow in (
            ("6.9 This Agreement", " is intended to be binding on the parties"),
            ("7.1 It is", " agreed that the Purchaser shall"),
            ("2.3 The Transfer", " is subject to the satisfaction of the Conditions"),
        ):
            with self.subTest(inner=inner):
                self.assertEqual(self._title(inner, follow), inner.split()[0])

    def test_recurring_party_name_is_not_split(self) -> None:
        # 687bf50e, b6f62e1d.
        page = "The Target Shareholders shall deliver the shares. The Secondary Guarantor shall pay.\n\n"
        self.assertEqual(
            self._title("5.2 Target Shareholders", " shall deliver the Shares at Completion.", page=page), "5.2"
        )
        self.assertEqual(self._title("9.4 Secondary Guarantor", " shall pay the amount.", page=page), "9.4")
        self.assertEqual(
            self._title("Section 4.7 Financing Parent", " has delivered the Commitment Letters.", page=page),
            "Section 4.7 Financing",
        )
        self.assertNotEqual(self._title("9.4 Seller; Buyer", " shall each bear their own costs."), "9.4 Seller")

    def test_list_subject_is_not_split_at_a_repeat(self) -> None:
        # 40773878, and a synthetic repro.
        for inner, follow in (
            ("19.3 Bagger-S\u00f8rensen and", " each of Claus Bagger-S\u00f8rensen and Steen Bagger-S\u00f8rensen agrees"),
            ("8.2 Holdings and", " each of Alpha Holdings and Beta Holdings agrees to pay"),
        ):
            with self.subTest(inner=inner):
                self.assertIn(self._title(inner, follow), {inner, inner.split()[0]})

    def test_is_followed_by_an_adverb_is_not_a_definition(self) -> None:
        # b2a8c999.
        self.assertEqual(self._title("12.6 Time is", " expressly of the essence of this Agreement."), "12.6")


class SentenceSplitThirdReviewTests(SentenceSplitReReviewTests):
    """Third review of PR #229."""

    def test_title_case_layout_heading_starting_with_an_opener_is_kept(self) -> None:
        # bb80b652: "6.3 The Parties' Covenants" is followed by 6.3.1 and printed bold.
        raw = "<p><b>6.3</b></p><p><b>The Parties\u2019 Covenants</b></p><p>6.3.1 Each Party shall keep the terms.</p>"
        self.assertEqual(
            self._title("6.3 The Parties\u2019 Covenants", " 6.3.1 Each Party shall keep the terms.", raw),
            "6.3 The Parties\u2019 Covenants",
        )

    def test_rejected_short_layout_heading_is_never_stripped(self) -> None:
        raw = "<p>4.4 <b>Escrow Release Procedure,</b></p><p>The Escrow Agent shall release the funds.</p>"
        self.assertEqual(
            self._title("4.4 Escrow Release Procedure, The Escrow Agent", " shall release the funds.", raw),
            "4.4 Escrow Release Procedure, The Escrow Agent",
        )

    def test_title_ends_before_the_sections_own_first_subsection(self) -> None:
        # ea63946b, bb80b652 without HTML, and an "8.2 The Escrow Agent" repro.
        cases = {
            ("5.2 The Preliminary Purchase Price and payment", " thereof 5.2.1 The Purchaser shall pay"): (
                "5.2 The Preliminary Purchase Price and payment thereof"
            ),
            ("6.3 The Parties\u2019 Covenants", " 6.3.1 Each Party shall keep the terms."): "6.3 The Parties\u2019 Covenants",
            ("8.2 The Escrow Agent", " 8.2.1 The Escrow Agent shall hold the Escrow Amount."): "8.2 The Escrow Agent",
        }
        for (inner, follow), title in cases.items():
            with self.subTest(inner=inner):
                self.assertEqual(self._title(inner, follow), title)
        # A lead-in to a list is not a heading.
        self.assertEqual(
            self._title("17.1 A notice given", " under this agreement:\n\n17.1.1 shall be in writing"), "17.1"
        )

    def test_is_followed_by_of_not_to_keeps_the_term(self) -> None:
        self.assertEqual(self._title("1.3 Business Day is", " not a Saturday or a Sunday."), "1.3 Business Day is")
        self.assertEqual(self._title("1.4 Knowledge is", " to be construed as actual knowledge."), "1.4 Knowledge is")
        self.assertEqual(self._title("12.6 Time is", " expressly of the essence of this Agreement."), "12.6")

    def test_term_inside_the_tag_ends_the_title(self) -> None:
        # 936e9204.
        self.assertEqual(
            self._title("1.14 Parties are Sellers, TriQuint and Purchaser collectively.", " Party is any of them."),
            "1.14 Parties",
        )
        self.assertEqual(self._title("4.1 Completion is conditional", " on the Conditions."), "4.1")
        self.assertEqual(self._title("9.2 BidCo is entitled", " to waive the Conditions."), "9.2")
