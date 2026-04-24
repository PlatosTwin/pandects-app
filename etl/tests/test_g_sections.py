import unittest
from typing import cast

from etl.domain.g_sections import extract_sections_from_xml


class ExtractSectionsFromXmlTests(unittest.TestCase):
    def test_extract_sections_from_xml_returns_nested_sections(self) -> None:
        xml = """
        <document uuid="agreement-1">
          <body>
            <article title="ARTICLE I" uuid="article-1" order="1">
              <section title="Section 1.1 First" uuid="section-1" order="1">Alpha</section>
              <section title="Section 1.2 Second" uuid="section-2" order="2">Beta</section>
            </article>
          </body>
        </document>
        """

        sections = extract_sections_from_xml(xml)

        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0]["section_uuid"], "section-1")
        self.assertEqual(sections[0]["article_title"], "ARTICLE I")
        self.assertEqual(sections[0]["section_title"], "Section 1.1 First")
        self.assertEqual(sections[0]["section_order"], 1)
        self.assertEqual(sections[1]["section_uuid"], "section-2")
        self.assertEqual(sections[1]["section_title"], "Section 1.2 Second")
        self.assertEqual(sections[1]["section_order"], 2)

    def test_extract_sections_from_xml_returns_blank_section_for_sectionless_article(self) -> None:
        xml = """
        <document uuid="agreement-1">
          <body>
            <article title="ARTICLE V CERTAIN AGREEMENTS" uuid="article-5" order="5">
              <text>Standalone article text.</text>
            </article>
          </body>
        </document>
        """

        sections = extract_sections_from_xml(xml)

        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]["section_uuid"], "article-5")
        self.assertEqual(sections[0]["article_title"], "ARTICLE V CERTAIN AGREEMENTS")
        self.assertEqual(sections[0]["article_title_normed"], "certain agreements")
        self.assertEqual(sections[0]["article_order"], 5)
        self.assertEqual(sections[0]["section_title"], "")
        self.assertEqual(sections[0]["section_title_normed"], "")
        self.assertIsNone(cast(object, sections[0]["section_order"]))
        self.assertIn("Standalone article text.", cast(str, sections[0]["xml_content"]))

    def test_extract_sections_from_xml_preserves_article_level_text_alongside_sections(self) -> None:
        xml = """
        <document uuid="agreement-1">
          <body>
            <article title="ARTICLE II COVENANTS" uuid="article-2" order="2">
              Intro article text.
              <section title="Section 2.1 First" uuid="section-21" order="1">Alpha</section>
              Bridge article text.
              <section title="Section 2.2 Second" uuid="section-22" order="2">Beta</section>
              Closing article text.
            </article>
          </body>
        </document>
        """

        sections = extract_sections_from_xml(xml)

        self.assertEqual(len(sections), 3)
        self.assertEqual(sections[0]["section_uuid"], "article-2")
        self.assertEqual(sections[0]["section_title"], "")
        self.assertIn("Intro article text.", cast(str, sections[0]["xml_content"]))
        self.assertIn("Bridge article text.", cast(str, sections[0]["xml_content"]))
        self.assertIn("Closing article text.", cast(str, sections[0]["xml_content"]))
        self.assertNotIn("Section 2.1 First", cast(str, sections[0]["xml_content"]))
        self.assertNotIn("Section 2.2 Second", cast(str, sections[0]["xml_content"]))
        self.assertEqual(sections[1]["section_uuid"], "section-21")
        self.assertEqual(sections[2]["section_uuid"], "section-22")


if __name__ == "__main__":
    _ = unittest.main()
