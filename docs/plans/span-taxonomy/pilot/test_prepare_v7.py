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
