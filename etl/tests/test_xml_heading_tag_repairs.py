# pyright: reportAny=false
import unittest

from etl.domain.xml_tag_repairs import (
    dedupe_adjacent_same_number_headings,
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


if __name__ == "__main__":
    _ = unittest.main()
