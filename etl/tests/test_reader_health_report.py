import unittest
from collections import Counter

from etl.utils.reader_health_report import gap_shapes, jurisdiction, sections_tagged_as_articles, structure_problems


class StructureProblemTests(unittest.TestCase):
    def test_sound_structure_has_no_problems(self) -> None:
        stats = {"sections_numbered": 90, "sections_unnumbered": 2, "duplicate_section_numbers": ["4.1"]}
        self.assertEqual(structure_problems(stats), [])

    def test_no_numbered_sections(self) -> None:
        self.assertEqual(structure_problems({"sections_numbered": 0, "sections_unnumbered": 40}), ["no_numbered_sections"])

    def test_mostly_unnumbered_and_duplicates(self) -> None:
        stats = {"sections_numbered": 10, "sections_unnumbered": 30, "duplicate_section_numbers": ["1.1", "1.2", "2.1"]}
        self.assertEqual(structure_problems(stats, 0.3), ["mostly_unnumbered_sections", "duplicate_section_numbers"])

    def test_duplicates_in_a_well_resolving_agreement_are_not_flagged(self) -> None:
        stats = {"sections_numbered": 100, "duplicate_section_numbers": ["1.1", "1.2", "2.1"]}
        self.assertEqual(structure_problems(stats, 0.9), [])


class GapShapeTests(unittest.TestCase):
    def test_counts_only_unresolved_references_by_shape(self) -> None:
        xrefs = [
            {"text": "Section 5.16.1", "status": "unresolved"},
            {"text": "Sections 2.1.2", "status": "unresolved"},
            {"text": "Article 2.1", "status": "unresolved"},
            {"text": "Section 7.10A", "status": "unresolved"},
            {"text": "Section 5.16.1", "status": "resolved"},
            {"text": "clause (a)", "status": "unresolved"},
        ]
        self.assertEqual(
            gap_shapes(xrefs),
            Counter({"dotted_subsection": 2, "article_as_section": 1, "letter_suffix_section": 1}),
        )


class FlatAgreementTests(unittest.TestCase):
    def test_numbered_sections_tagged_as_articles(self) -> None:
        flat = "".join(f'<article uuid="a{i}" title="Section {i}. Item {i}.">' for i in range(1, 6))
        self.assertTrue(sections_tagged_as_articles(flat))
        bare = "".join(f'<article title="{i}. Item.">' for i in range(1, 6))
        self.assertTrue(sections_tagged_as_articles(bare))

    def test_ordinary_articles(self) -> None:
        normal = "".join(f'<article title="ARTICLE {r} THINGS">' for r in ("I", "II", "III", "IV"))
        self.assertFalse(sections_tagged_as_articles(normal))


class JurisdictionTests(unittest.TestCase):
    def test_us_state_governing_law(self) -> None:
        self.assertEqual(jurisdiction("<text>governed by the laws of the State of Delaware</text>"), "us")
        self.assertEqual(jurisdiction("in accordance with the law of the State of New York"), "us")
        # A US-law agreement that mentions a UK statute is still US.
        self.assertEqual(jurisdiction("the Companies Act 2006; the laws of the State of Texas"), "us")

    def test_markers_name_the_rest(self) -> None:
        self.assertEqual(jurisdiction("the Companies Act 2006 and the courts of England and Wales"), "uk")
        self.assertEqual(jurisdiction("governed by the laws of New South Wales; Acme Pty Ltd"), "au")
        self.assertEqual(jurisdiction("governed by the laws of the Netherlands"), "other")


if __name__ == "__main__":
    unittest.main()
