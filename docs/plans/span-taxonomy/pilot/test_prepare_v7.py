"""Tests for prepare_v7.py's definition forms.

Run: python3 -m unittest test_prepare_v7   (from this directory)
"""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(__file__))
import prepare_v7  # noqa: F401  (patches prepare_v4.DEFINED)
import prepare_v4 as v4


def terms(text):
    return [m.group(1) for m in v4.DEFINED.finditer(text)]


class DefinedForms(unittest.TestCase):
    def test_colon_inside_quotes(self):  # 67e84aa3
        self.assertEqual(terms("“General Indemnity Holdback Amount:” $37,700,000."), ["General Indemnity Holdback Amount"])

    def test_colon_after_quotes(self):  # 67e84aa3
        self.assertEqual(terms("“Special Indemnity Holdback Amount”: means $18,850,000."), ["Special Indemnity Holdback Amount"])

    def test_means(self):
        self.assertEqual(terms("\"Cap Amount\" means $5,000,000."), ["Cap Amount"])

    def test_deemed(self):
        self.assertEqual(terms("“Triggering Event” shall be deemed to have occurred if"), ["Triggering Event"])

    def test_plain_quote_is_not_a_definition(self):
        self.assertEqual(terms("the parties agree that the “Closing” will occur in New York, and"), [])


def layer(*paras, section="defs"):
    text, leaves, pos = "", [], 0
    for kind, para in paras:
        leaves.append({"kind": kind, "term": None, "section": section, "start": pos, "end": pos + len(para)})
        text += para + "\n"
        pos = len(text)
    return {"text": text, "leaves": leaves}


class RunOnDefinitions(unittest.TestCase):
    def kept(self, lay, first=0):
        keep = {"defs": {id(lay["leaves"][first])}}
        prepare_v7.extend_definitions(lay, keep)
        return [i for i, lf in enumerate(lay["leaves"]) if id(lf) in keep["defs"]]

    def test_page_break(self):  # e446bd34
        lay = layer(("definition", "A “Triggering Event” shall be deemed to have occurred if: (i) the Company Board"),
                    ("definition", "A-14"), ("definition", "Table of Contents"),
                    ("definition", "disclosed; (vii) a willful and material breach of Section 5.4."),
                    ("definition", "“Withdrawal Liability” shall mean liability to a Multiemployer Plan."))
        self.assertEqual(self.kept(lay), [0, 1, 2, 3])

    def test_split_definition(self):  # cff15b20
        lay = layer(("definition", "“Material Adverse Effect” – an event, change or development, other than (i) changes in"),
                    ("definition", "Companies operate; (iii) any change that is generally applicable to the industries"),
                    ("definition", "“Material Consents” – as defined in Section 9.4."))
        self.assertEqual(self.kept(lay), [0, 1])

    def test_next_definition_stops(self):
        lay = layer(("definition", "“Cap Amount” means $5,000,000."), ("definition", "“Closing” has the meaning in Section 2.1."))
        self.assertEqual(self.kept(lay), [0])

    def test_text_leaves_are_not_extended(self):
        lay = layer(("text", "Section 8.1 Termination. This Agreement may be terminated"), ("text", "(a) by mutual consent;"))
        self.assertEqual(self.kept(lay), [0])


class Terms(unittest.TestCase):
    def test_new_terms(self):
        for term in ("Drop Dead Date", "Longstop Date", "Excepted Representations", "MaterialAdverse Effect"):
            self.assertTrue(v4.TERMS.search(term) and v4.RESOLVE.search(term), term)
