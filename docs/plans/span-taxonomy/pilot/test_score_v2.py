"""Tests for score_v2's field normalization.

Run: python3 -m unittest test_score_v2   (from this directory)
"""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(__file__))
from score_v2 import period, fields


def survival(value, unit):
    return {"period": {"kind": "fixed", "duration": {"value": value, "unit": unit, "anchor": "closing"}}}


class Durations(unittest.TestCase):
    def test_years_equal_months(self):  # 1c996793: "1 year" against "12 months"
        self.assertEqual(period(survival(1, "years")), period(survival(12, "months")))

    def test_fractional_years(self):
        self.assertEqual(period(survival(1.5, "years")), period(survival(18, "months")))

    def test_different_lengths_differ(self):
        self.assertNotEqual(period(survival(1, "years")), period(survival(18, "months")))

    def test_days_are_not_months(self):
        self.assertNotEqual(period(survival(30, "days")), period(survival(1, "months")))

    def test_null_offset_is_zero(self):  # a5fad085: statute of limitations with offset_days null against 0
        a = {"period": {"kind": "statute_of_limitations_plus", "offset_days": None}}
        b = {"period": {"kind": "statute_of_limitations_plus", "offset_days": 0}}
        self.assertEqual(period(a), period(b))
        self.assertNotEqual(period(a), period({"period": {"kind": "statute_of_limitations_plus", "offset_days": 60}}))

    def test_outside_date_duration(self):
        a = fields("ma.termination.outside_date", {"initial": {"duration": {"value": 1, "unit": "years"}}})
        b = fields("ma.termination.outside_date", {"initial": {"duration": {"value": 12, "unit": "months"}}})
        self.assertEqual(a["initial"], b["initial"])


if __name__ == "__main__":
    unittest.main()
