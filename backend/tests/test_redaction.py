import importlib.util
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from backend.redaction import redact_agreement_xml, redact_reader_index


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
        if placeholder is None:
            self.fail("Expected a redaction placeholder node.")
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

    def test_billion_laughs_rejected_by_defusedxml(self) -> None:
        # defusedxml refuses entity expansion attacks even when the input is
        # technically well-formed XML. Confirms the parser swap took effect.
        from defusedxml.common import EntitiesForbidden

        bomb = (
            '<?xml version="1.0"?>'
            '<!DOCTYPE doc ['
            '<!ENTITY a "AAAAAAAAAA">'
            '<!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">'
            ']><doc>&b;</doc>'
        )
        with self.assertRaises(EntitiesForbidden):
            redact_agreement_xml(bomb, focus_section_uuid=None, neighbor_sections=0)


if __name__ == "__main__":
    unittest.main()


class TestReaderIndexRedaction(unittest.TestCase):
    XML = """<document uuid="doc">
  <body>
    <article uuid="a1"><text>Preamble</text>
      <section uuid="s1"><text>KEEP</text></section>
      <section uuid="s2"><text>HIDE</text><section uuid="s2n"><text>NESTED</text></section></section>
    </article>
  </body>
  <backMatter><text>Exhibit A</text></backMatter>
</document>"""

    def _payload(self) -> dict[str, Any]:
        return {
            "outline": [],
            "terms": [
                {"id": 0, "term": "A", "decls": [{"anchor": {"s": "s1", "t": 0}, "in": "Section 1.1"}]},
                {"id": 1, "term": "B", "decls": [{"anchor": {"s": "s2n", "t": 0}, "in": "Section 1.2"}]},
            ],
            "uses": {"s1/0": [[0, 1, 1]], "s2n/0": [[0, 1, 0]], "a1/0": [[0, 1, 0]], "backMatter/0": [[0, 1, 0]]},
            "xrefs": [],
        }

    def test_items_outside_the_visible_text_are_dropped(self) -> None:
        out = redact_reader_index(self._payload(), self.XML, focus_section_uuid="s1", neighbor_sections=0)
        self.assertEqual(sorted(out["uses"]), ["backMatter/0", "s1/0"])
        self.assertEqual(out["terms"][0]["decls"][0]["in"], "Section 1.1")
        self.assertEqual(out["terms"][1]["decls"], [{"redacted": True, "label": "Section 1.2"}])

    def test_a_nested_section_stays_hidden_with_its_parent(self) -> None:
        # s2n is within one of s1 in document order, but redact_agreement_xml drops it with s2
        out = redact_reader_index(self._payload(), self.XML, focus_section_uuid="s2n", neighbor_sections=0)
        self.assertNotIn("s2n/0", out["uses"])
        self.assertNotIn("NESTED", redact_agreement_xml(self.XML, focus_section_uuid="s2n", neighbor_sections=0))

    def test_the_input_payload_is_not_changed(self) -> None:
        payload = self._payload()
        _ = redact_reader_index(payload, self.XML, focus_section_uuid=None, neighbor_sections=1)
        self.assertIn("s1/0", payload["uses"])


def _load_enricher() -> Any:
    # The ETL enricher is stdlib-only, so load it by path rather than installing the ETL package.
    path = Path(__file__).resolve().parents[2] / "etl" / "src" / "etl" / "domain" / "reader_index.py"
    spec = importlib.util.spec_from_file_location("reader_index_enricher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses look their module up in sys.modules while the class is built
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestReaderIndexRedactionAgainstEnricher(unittest.TestCase):
    """Every item an anonymous caller keeps must point at text the redacted XML still
    shows, for anchors built by the real enricher, including shapes the XML builder
    does not emit today (uuid-less sections and articles)."""

    XML = """<document uuid="doc">
  <frontMatter><text>MERGER AGREEMENT among Parent and the Company (the "Company").</text></frontMatter>
  <body>
    <article title="ARTICLE I DEFINITIONS" uuid="a1">
      <text>Article preamble: "Preamble Term" means a preamble thing. See Section 1.2.</text>
      <section title="Section 1.1 Defined Terms" uuid="s11">
        <text>"Visible Fee" means the fee. The Company pays the Visible Fee under Section 1.2.</text>
      </section>
      <section title="Section 1.2 Secret" uuid="s12">
        <text>"Secret Fee" means the hidden fee. The Company pays the Secret Fee under Section 1.1.</text>
      </section>
      <section title="Section 1.3 No Uuid">
        <text>"Orphan Fee" means another hidden fee. See Section 1.1 and the Secret Fee.</text>
      </section>
    </article>
    <article title="ARTICLE II OTHER">
      <text>"Loose Term" means text in an article without a uuid. See Section 1.1.</text>
      <section title="Section 2.1 Payment" uuid="s21">
        <text>(a) The Secret Fee and the Visible Fee are paid under Section 1.2.</text>
        <text>(b) The Orphan Fee is paid under Section 1.1.</text>
      </section>
    </article>
  </body>
  <backMatter>
    <text>EXHIBIT A FORM OF ESCROW AGREEMENT</text>
    <text>The Company and the Escrow Agent (the "Escrow Agent") agree as set forth in Section 1.1.</text>
  </backMatter>
</document>"""

    def _visible_texts(self, focus: str | None, neighbors: int) -> set[str]:
        root = ET.fromstring(redact_agreement_xml(self.XML, focus_section_uuid=focus, neighbor_sections=neighbors))
        return {(el.text or "").strip() for el in root.iter("text") if el.get("redacted") != "true"}

    def test_kept_items_point_only_at_visible_text(self) -> None:
        enricher = _load_enricher()
        payload = enricher.build_reader_index(self.XML)
        texts = {f"{n.s}/{n.t}": n.text.strip() for n in enricher._collect_texts(ET.fromstring(self.XML))[0]}
        cases = [(None, 1), ("s11", 0), ("s12", 0), ("s21", 0), ("s11", 5)]
        for focus, neighbors in cases:
            with self.subTest(focus=focus, neighbors=neighbors):
                visible = self._visible_texts(focus, neighbors)
                out = redact_reader_index(payload, self.XML, focus_section_uuid=focus, neighbor_sections=neighbors)
                kept = list(out["uses"])
                kept += [f"{x['at']['s']}/{x['at']['t']}" for x in out["xrefs"]]
                kept += [f"{d['anchor']['s']}/{d['anchor']['t']}" for t in out["terms"] for d in t["decls"] if not d.get("redacted")]
                for key in kept:
                    self.assertIn(texts[key], visible, key)
                for node in out["outline"]:
                    for sub in [node, *node.get("children", [])]:
                        if sub["title"] and "t" in sub["anchor"]:
                            self.assertIn(texts[f"{sub['anchor']['s']}/{sub['anchor']['t']}"], visible)
                dumped = str(out)
                for secret in ("Secret Fee", "Orphan Fee", "Preamble Term", "Loose Term"):
                    if not any(secret in v for v in visible):
                        self.assertNotIn(f"'{secret}'", dumped.replace(f"'term': '{secret}'", ""))

    def test_the_payload_has_items_in_hidden_text_to_drop(self) -> None:
        # guards the test above against passing because the enricher found nothing
        payload = _load_enricher().build_reader_index(self.XML)
        terms = {t["term"] for t in payload["terms"]}
        self.assertTrue({"Secret Fee", "Orphan Fee", "Visible Fee"} <= terms, terms)
        self.assertTrue(any(x["at"]["s"] == "s12" for x in payload["xrefs"]))
