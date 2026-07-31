# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportAny=false
# pyright: reportMissingImports=false
# retrieval is imported via runtime sys.path injection (standalone script dir).
"""Unit tests for the hybrid taxonomy retriever (pure logic; no DB/Voyage).

retrieval.py lives in the standalone deepseek script dir (not an importable
package) and uses same-dir imports, so the dir is put on sys.path here.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

_DEEPSEEK = (
    Path(__file__).resolve().parents[1]
    / "src" / "etl" / "models" / "taxonomy" / "deepseek"
)
sys.path.insert(0, str(_DEEPSEEK))

from retrieval import (  # noqa: E402
    BM25,
    HybridRetriever,
    Node,
    RetrievalConfig,
    Section,
    build_queries,
    fuse,
    tokenize,
)


def _fixture() -> tuple[list[Node], np.ndarray]:
    nodes = [
        Node("l2EFF", "l2", "Efforts and Regulatory", "Covenants",
             "Efforts and Regulatory",
             "umbrella category route to a specific sub-leaf efforts regulatory",
             ["umbrella category (route to a specific sub-leaf)"], ["cueA"]),
        Node("l3HSR", "l3", "Antitrust / HSR Filings and Clearances",
             "Covenants", "Efforts and Regulatory",
             "Covenants > Efforts and Regulatory > Antitrust HSR filings and "
             "clearances. HSR Act notification and report form second request "
             "antitrust waiting period early termination.",
             ["hsr act", "antitrust", "second request"], ["antitrust cue"]),
        Node("l3EFF", "l3", "Reasonable Best Efforts to Consummate",
             "Covenants", "Efforts and Regulatory",
             "Covenants > Efforts and Regulatory > reasonable best efforts to "
             "consummate cooperation obtain consents.",
             ["reasonable best efforts to consummate"], ["efforts cue"]),
        Node("l2MUT", "l2", "Mutual Conditions", "Conditions to Closing",
             "Mutual Conditions",
             "umbrella category route to a specific sub-leaf mutual conditions",
             ["umbrella category (route to a specific sub-leaf)"], ["cueB"]),
        Node("l3STK", "l3", "Stockholder / Shareholder Approval (condition)",
             "Conditions to Closing", "Mutual Conditions",
             "Conditions to Closing > Mutual Conditions > stockholder approval "
             "condition requisite vote adopted this agreement.",
             ["company stockholder approval", "requisite vote"], ["stk cue"]),
        Node("l3NLR", "l3",
             "No Legal Restraint; No Injunction; No Governmental Order (condition)",
             "Conditions to Closing", "Mutual Conditions",
             "Conditions to Closing > Mutual Conditions > no legal restraint no "
             "injunction no governmental order prohibiting consummation.",
             ["no injunction", "no legal restraint"], ["nlr cue"]),
    ]
    # 4-dim toy embeddings: HSR / EFF / STK / NLR on basis axes; L2s blended.
    emb = np.array([
        [0.7, 0.7, 0.0, 0.0],   # l2EFF
        [1.0, 0.0, 0.0, 0.0],   # l3HSR
        [0.0, 1.0, 0.0, 0.0],   # l3EFF
        [0.0, 0.0, 0.7, 0.7],   # l2MUT
        [0.0, 0.0, 1.0, 0.0],   # l3STK
        [0.0, 0.0, 0.0, 1.0],   # l3NLR
    ], dtype=np.float64)
    return nodes, emb


def _fake_embedder(texts: list[str]) -> np.ndarray:
    out = []
    for t in texts:
        tl = t.lower()
        if "hsr" in tl or "antitrust" in tl:
            out.append([1.0, 0.0, 0.0, 0.0])
        elif "stockholder" in tl:
            out.append([0.0, 0.0, 1.0, 0.0])
        else:
            out.append([0.25, 0.25, 0.25, 0.25])
    return np.asarray(out, dtype=np.float64)


class TokenizeTests(unittest.TestCase):
    def test_strips_placeholders_and_lowercases(self) -> None:
        toks = tokenize("The Company shall pay $[•] within [NUMBER] Days HSR")
        self.assertNotIn("[", " ".join(toks))
        self.assertIn("company", toks)
        self.assertIn("hsr", toks)
        # single-char tokens dropped
        self.assertNotIn("a", toks)

    def test_empty(self) -> None:
        self.assertEqual(tokenize(""), [])

    def test_build_queries_strips_section_lead_and_caps_body(self) -> None:
        dense, sparse, hay = build_queries(
            "Article IV Conditions", "Section 7.02 Mutual Conditions",
            "word " * 50, RetrievalConfig(max_body_words=10),
        )
        self.assertIn("Conditions", dense)
        self.assertNotIn("Section 7.02", dense)
        self.assertEqual(dense.split("\n", 1)[1].split(), ["word"] * 10)
        # title tokens upweighted in the sparse query
        self.assertGreaterEqual(sparse.count("mutual"), 3)
        self.assertEqual(hay, hay.lower())


class BM25Tests(unittest.TestCase):
    def test_ranks_lexical_match_first(self) -> None:
        corpus = [
            tokenize("antitrust hsr act waiting period second request"),
            tokenize("governing law delaware jury trial waiver"),
            tokenize("stockholder approval requisite vote meeting"),
        ]
        bm = BM25(corpus, k1=1.5, b=0.75)
        s = bm.scores(tokenize("hsr antitrust filing"))
        self.assertEqual(int(np.argmax(s)), 0)
        self.assertGreater(s[0], s[1])

    def test_unknown_terms_score_zero(self) -> None:
        bm = BM25([tokenize("alpha beta")], k1=1.5, b=0.75)
        self.assertAlmostEqual(float(bm.scores(tokenize("zzz qqq"))[0]), 0.0)


class FuseTests(unittest.TestCase):
    def test_constant_signal_no_nan(self) -> None:
        d = np.array([0.5, 0.5, 0.5])
        sp = np.array([1.0, 2.0, 3.0])
        kw = np.array([0.0, 0.0, 0.0])
        f = fuse(d, sp, kw, RetrievalConfig())
        self.assertFalse(np.isnan(f).any())
        self.assertEqual(int(np.argmax(f)), 2)  # driven by sparse

    def test_weighting_prefers_dense(self) -> None:
        d = np.array([1.0, 0.0])
        sp = np.array([0.0, 1.0])
        kw = np.array([0.0, 0.0])
        f = fuse(d, sp, kw, RetrievalConfig(dense_w=0.9, sparse_w=0.1, keyword_w=0.0))
        self.assertGreater(f[0], f[1])


class HybridRetrieverTests(unittest.TestCase):
    def test_length_mismatch_raises(self) -> None:
        nodes, emb = _fixture()
        with self.assertRaises(ValueError):
            HybridRetriever(nodes, emb[:-1], _fake_embedder)

    def test_hsr_section_ranks_antitrust_first_with_l2_backoff(self) -> None:
        nodes, emb = _fixture()
        r = HybridRetriever(nodes, emb, _fake_embedder, RetrievalConfig(k=2))
        sec = Section(
            "Article VI Covenants", "Regulatory Approvals; HSR",
            "Each party shall make its HSR Act filing and use reasonable best "
            "efforts to obtain expiration of the antitrust waiting period.",
        )
        cands = r.retrieve(sec)
        top = cands[0]
        self.assertEqual(top.standard_id, "l3HSR")
        self.assertEqual(top.level, "l3")
        self.assertFalse(top.is_backoff)
        # exactly k L3 candidates, then parent-L2 backoff appended
        l3 = [c for c in cands if not c.is_backoff]
        backoff = [c for c in cands if c.is_backoff]
        self.assertEqual(len(l3), 2)
        self.assertTrue(all(c.level == "l3" for c in l3))
        self.assertIn("l2EFF", {c.standard_id for c in backoff})
        self.assertTrue(all(c.level == "l2" for c in backoff))

    def test_deterministic(self) -> None:
        nodes, emb = _fixture()
        r = HybridRetriever(nodes, emb, _fake_embedder, RetrievalConfig(k=3))
        sec = Section("Article I", "Stockholder Approval",
                      "The requisite company stockholder approval shall be obtained.")
        a = [(c.standard_id, round(c.score, 9)) for c in r.retrieve(sec)]
        b = [(c.standard_id, round(c.score, 9)) for c in r.retrieve(sec)]
        self.assertEqual(a, b)
        self.assertEqual(r.retrieve(sec)[0].standard_id, "l3STK")

    def test_include_parent_l2_false_omits_backoff(self) -> None:
        nodes, emb = _fixture()
        r = HybridRetriever(
            nodes, emb, _fake_embedder,
            RetrievalConfig(k=2, include_parent_l2=False),
        )
        sec = Section("Article VI", "HSR", "HSR Act antitrust filing.")
        self.assertTrue(all(not c.is_backoff for c in r.retrieve(sec)))

    def test_keyword_signal_rewards_canonical_term_hit(self) -> None:
        nodes, emb = _fixture()
        r = HybridRetriever(nodes, emb, _fake_embedder, RetrievalConfig(k=4))
        sec = Section("Article VI", "Filings",
                      "The parties will coordinate the second request response.")
        by_id = {c.standard_id: c for c in r.retrieve(sec)}
        # "second request" is an l3HSR canonical term -> nonzero keyword score
        self.assertGreater(by_id["l3HSR"].keyword, 0.0)


if __name__ == "__main__":
    unittest.main()
