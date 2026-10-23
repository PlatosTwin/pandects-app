# pyright: reportAny=false
import json
import re
import unittest
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures" / "reader_gold"
NO_TARGET = {"external", "disclosure_schedule", "attachment_local", "inline", "not_in_filing"}
TARGET_ID = re.compile(r"^(?:a[0-9A-Z]+|s[0-9A-Z.]+(?:\([A-Za-z0-9]+\))*|x-[a-z]+-[a-z0-9.\-]+)$")


class ReaderGoldFixtureTests(unittest.TestCase):
    """The gold set is scored against the database's XML (etl.utils.reader_gold), so CI only
    checks that every fixture is adjudicated and internally consistent."""

    def test_fixtures_are_adjudicated_and_consistent(self) -> None:
        files = sorted(FIXTURES.glob("*.json"))
        self.assertGreaterEqual(len(files), 5)
        for f in files:
            with self.subTest(fixture=f.name):
                g = json.loads(f.read_text(encoding="utf-8"))
                self.assertEqual(f.stem, g["agreement_uuid"])
                self.assertIsInstance(g["xml_version"], int)
                # held-out files carry their open disputes until the user rules; pilot files are settled
                if g.get("set") != "heldout-v1":
                    self.assertNotIn("pending_disputes", g)
                for d in g.get("pending_disputes", []):
                    self.assertIn(d["type"], {"disagree", "remove", "missing"})
                paras = set(g["para_map"])
                for d in g["declarations"]:
                    self.assertIn(d["para"], paras)
                    self.assertIn(d["kind"], {"means", "paren", "pointer"})
                    self.assertFalse(d["term"][:1].islower(), d["term"])
                for x in g["xrefs"]:
                    self.assertIn(x["para"], paras)
                    self.assertIn(x["mention"], x["context"])
                    self.assertTrue(x["target"] in NO_TARGET or TARGET_ID.match(x["target"]), x)
                for u in g["term_uses"]:
                    self.assertIn(u["para"], paras)
                    self.assertGreater(u["count"], 0)


if __name__ == "__main__":
    _ = unittest.main()
