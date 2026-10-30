"""The anchor and offset contract between the enricher and the reader
(docs/plans/smart-reader.md, Section 6.3).

fixtures/reader_offset_contract/agreement.xml has curly quotes, entities, a
non-breaking space, a non-BMP character, a <definition>, a page marker and a
section without a uuid. expected.json holds the text the enricher reads for
each anchor key plus the payload it builds; the frontend's
client/lib/reader-index.spec.ts reads the same two files and checks that its
own walk produces the same keys and texts and that every span lines up.

After a deliberate enricher change, regenerate with
REGEN_READER_CONTRACT=1 uv run python -m unittest tests.test_reader_offset_contract
"""

import json
import os
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from etl.domain.reader_index import _collect_texts, build_reader_index

FIXTURE = Path(__file__).parent / "fixtures" / "reader_offset_contract"


def _contract() -> dict[str, Any]:
    xml = (FIXTURE / "agreement.xml").read_text(encoding="utf-8")
    texts, _ = _collect_texts(ET.fromstring(xml))
    payload = build_reader_index(xml, agreement_uuid="contract-doc", xml_version=1)
    return {
        "texts": {f"{t.s}/{t.t}": t.text for t in texts},
        "payload": {k: payload[k] for k in ("outline", "terms", "uses", "xrefs")},
    }


class ReaderOffsetContractTests(unittest.TestCase):
    def test_expected_json_matches_the_enricher(self) -> None:
        actual = _contract()
        path = FIXTURE / "expected.json"
        if os.environ.get("REGEN_READER_CONTRACT") == "1":
            _ = path.write_text(json.dumps(actual, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        expected = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(actual, expected)

    def test_spans_are_utf16_offsets_into_the_anchor_text(self) -> None:
        contract = _contract()
        texts: dict[str, str] = contract["texts"]

        def utf16_slice(text: str, a: int, b: int) -> str:
            return text.encode("utf-16-le")[2 * a : 2 * b].decode("utf-16-le")

        for xref in contract["payload"]["xrefs"]:
            key = f"{xref['at']['s']}/{xref['at']['t']}"
            self.assertEqual(utf16_slice(texts[key], *xref["span"]), xref["text"])
        # The fixture exercises a use after a non-BMP character, where UTF-16
        # and code-point offsets differ.
        self.assertTrue(any(ord(c) > 0xFFFF for c in texts["sec-1-1/1"]))
        self.assertIn("sec-1-1/1", contract["payload"]["uses"])


if __name__ == "__main__":
    unittest.main()
