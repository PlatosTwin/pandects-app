import unittest
import xml.etree.ElementTree as ET

from backend.redaction import redact_agreement_xml


class TestXmlRedaction(unittest.TestCase):
    def test_redacts_section_body_outside_window_keeps_titles(self) -> None:
        xml_in = """<document uuid="doc">
  <metadata><acquirer>A</acquirer></metadata>
  <body>
    <article title="ARTICLE I" uuid="a1">
      <section title="Section 1" uuid="s1"><text>KEEP</text></section>
      <section title="Section 2" uuid="s2"><text>HIDE</text></section>
    </article>
  </body>
</document>"""

        xml_out = redact_agreement_xml(
            xml_in, focus_section_uuid="s1", neighbor_sections=0
        )
        root = ET.fromstring(xml_out)
        sections = [el for el in root.iter() if el.tag == "section"]
        self.assertEqual([s.get("uuid") for s in sections], ["s1", "s2"])

        s1 = sections[0]
        s2 = sections[1]
        self.assertIn("KEEP", ET.tostring(s1, encoding="unicode"))
        self.assertNotIn("HIDE", ET.tostring(s2, encoding="unicode"))

        placeholder = next(iter(s2), None)
        self.assertIsNotNone(placeholder)
        self.assertEqual(placeholder.tag, "text")
        self.assertEqual(placeholder.attrib.get("redacted"), "true")
        self.assertEqual((placeholder.text or "").strip(), "[REDACTED]")

    def test_redacts_article_preamble_text_nodes(self) -> None:
        xml_in = """<document uuid="doc">
  <body>
    <article title="ARTICLE III" uuid="a3">
      <text>TOP LEVEL PREAMBLE</text>
      <section title="Section 3.1" uuid="s31"><text>KEEP</text></section>
    </article>
  </body>
</document>"""

        xml_out = redact_agreement_xml(
            xml_in, focus_section_uuid="s31", neighbor_sections=0
        )
        root = ET.fromstring(xml_out)
        article = next(el for el in root.iter() if el.tag == "article")

        self.assertNotIn("TOP LEVEL PREAMBLE", ET.tostring(article, encoding="unicode"))
        first_child = next(iter(article))
        self.assertEqual(first_child.tag, "text")
        self.assertEqual(first_child.attrib.get("redacted"), "true")
        self.assertEqual((first_child.text or "").strip(), "[REDACTED]")

        section = next(el for el in article.iter() if el.tag == "section")
        self.assertIn("KEEP", ET.tostring(section, encoding="unicode"))

    def test_neighbor_sections_validation(self) -> None:
        with self.assertRaises(ValueError):
            redact_agreement_xml("<document/>", focus_section_uuid=None, neighbor_sections=-1)
        with self.assertRaises(ValueError):
            redact_agreement_xml("<document/>", focus_section_uuid=None, neighbor_sections=6)


if __name__ == "__main__":
    unittest.main()

