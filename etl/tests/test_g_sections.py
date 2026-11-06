import unittest
from typing import cast

from etl.domain.g_sections import (
    build_section_search_text,
    extract_sections_from_xml,
    normalize_section_search_text,
    section_plain_text_from_xml,
)


def _definitions_xml(lead_in: str, definitions: int = 10) -> str:
    rows = "".join(f'<definition term="t{i}">“Term {i}” means thing {i}.</definition>' for i in range(definitions))
    return f"""
    <document uuid="agreement-1">
      <body>
        <text>THIS AGREEMENT is made by and among the parties.</text>
        <text>WHEREAS, the parties have defined terms below.</text>
        {lead_in}{rows}<pageUUID>page-2</pageUUID>
        <article title="ARTICLE I" uuid="article-1" order="1">
          <section title="Section 1.1 First" uuid="section-1" order="1"><text>Alpha</text></section>
        </article>
      </body>
    </document>
    """


class PreArticleDefinitionsRowTests(unittest.TestCase):
    def test_block_gets_an_article_level_row_with_its_printed_heading(self) -> None:
        sections = extract_sections_from_xml(
            _definitions_xml("<text>DEFINITIONS</text><text>The following terms have these meanings:</text>")
        )

        self.assertEqual(len(sections), 2)
        row = sections[0]
        self.assertEqual(row["article_title"], "DEFINITIONS")
        self.assertEqual(row["article_title_normed"], "definitions")
        self.assertEqual(row["section_title"], "")
        self.assertIsNone(row["article_order"])
        self.assertIsNone(row["section_order"])
        plain_text = section_plain_text_from_xml(row["xml_content"])
        self.assertTrue(plain_text.startswith("DEFINITIONS The following terms have these meanings:"))
        self.assertTrue(plain_text.endswith("“Term 9” means thing 9."))
        self.assertNotIn("WHEREAS", plain_text)
        self.assertEqual(sections[1]["section_uuid"], "section-1")

    def test_run_in_heading_is_the_printed_label(self) -> None:
        sections = extract_sections_from_xml(
            _definitions_xml("<text>Definitions: For all purposes of this Agreement, the following terms apply.</text>")
        )

        self.assertEqual(sections[0]["article_title"], "Definitions")

    def test_no_heading_is_made_up(self) -> None:
        sections = extract_sections_from_xml(_definitions_xml(""))

        self.assertEqual(sections[0]["article_title"], "")
        self.assertEqual(sections[0]["section_title"], "")

    def test_row_id_is_stable(self) -> None:
        first = extract_sections_from_xml(_definitions_xml(""))[0]["section_uuid"]
        self.assertEqual(first, extract_sections_from_xml(_definitions_xml("<text>DEFINITIONS</text>"))[0]["section_uuid"])

    def test_definition_paragraphs_served_as_text_count(self) -> None:
        xml = _definitions_xml("", definitions=4).replace(
            "<pageUUID>page-2</pageUUID>",
            "".join(f"<text>“Other {i}” has the meaning set forth in Section 1.{i}.</text>" for i in range(6)),
        )

        self.assertEqual(len(extract_sections_from_xml(xml)), 2)

    def test_a_few_inline_terms_get_no_row(self) -> None:
        sections = extract_sections_from_xml(_definitions_xml("<text>DEFINITIONS</text>", definitions=9))

        self.assertEqual([section["section_uuid"] for section in sections], ["section-1"])


class ExtractSectionsFromXmlTests(unittest.TestCase):
    def test_section_plain_text_decodes_entities_and_omits_page_metadata(self) -> None:
        fragment = (
            "<text>Buyer &amp; Seller shall use</text>"
            "<page>42</page><pageUUID>internal-page-id</pageUUID>"
            "<text>reasonable best efforts.</text>"
        )

        self.assertEqual(
            section_plain_text_from_xml(fragment),
            "Buyer & Seller shall use reasonable best efforts.",
        )

    def test_section_plain_text_collapses_nested_markup_and_unicode_whitespace(self) -> None:
        fragment = "<text>  Cafe\u0301\u00a0<em>Agreement</em>\n shall apply. </text>"

        plain_text, normalized_text = build_section_search_text(fragment)

        self.assertEqual(plain_text, "Café Agreement shall apply.")
        self.assertEqual(normalized_text, "café agreement shall apply.")

    def test_normalized_section_text_uses_nfkc_and_casefold(self) -> None:
        self.assertEqual(
            normalize_section_search_text("Kelvin  Straße\tTerms"),
            "kelvin strasse terms",
        )

    def test_section_plain_text_rejects_malformed_xml(self) -> None:
        with self.assertRaisesRegex(ValueError, "not well-formed"):
            _ = section_plain_text_from_xml("<text>unclosed")

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
