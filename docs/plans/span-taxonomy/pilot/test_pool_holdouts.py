"""Tests for pooling a saved result (plan section 7.25).

Run: python3 -m unittest test_pool_holdouts   (from this directory)
"""
import json, os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(__file__))
import pool_holdouts

HERE = os.path.dirname(os.path.abspath(__file__))


class Saved(unittest.TestCase):
    def test_saved_result_round_trips(self):  # the M7 to M9 pool, re-read from m9_v213_results.json
        spec = os.path.join(HERE, "m9_v213_results.json") + "#pooled_us_142_after"
        before = json.load(open(spec.split("#")[0]))["pooled_us_142_after"]
        after = pool_holdouts.pool([spec])
        self.assertEqual(after["agreements"], before["agreements"])
        self.assertEqual(after["detection"], before["detection"])
        for k, v in before["fields"].items():
            self.assertEqual({x: y for x, y in after["fields"][k].items() if x != "set_f1_approx"}, v)

    def test_two_saved_results_weight_set_f1_by_cases(self):
        def result(n, right, f1):
            return {"agreements": n, "detection": {"x": {"tp": n, "fp": 0, "fn": 0, "tn": 0}},
                    "fields": {"x.s": {"right": right, "n": n, "set_f1": f1}}}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "r.json")
            json.dump({"a": result(30, 30, 1.0), "b": result(10, 8, 0.9)}, open(p, "w"))
            out = pool_holdouts.pool([p + "#a", p + "#b"])
        e = out["fields"]["x.s"]
        self.assertEqual((e["right"], e["n"]), (38, 40))
        self.assertAlmostEqual(e["set_f1"], 0.975)
        self.assertTrue(e["set_f1_approx"])
        self.assertTrue(e["meets_bar"])


if __name__ == "__main__":
    unittest.main()
