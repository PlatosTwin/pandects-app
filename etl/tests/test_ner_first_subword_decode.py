# pyright: reportPrivateUsage=false
import unittest
from types import SimpleNamespace

import torch

from etl.models.ner.ner import NERInference
from etl.models.ner.ner_classes import NERTagger, build_bioes_constraints

ID2LABEL = {
    0: "O",
    1: "B-SECTION",
    2: "I-SECTION",
    3: "E-SECTION",
    4: "B-ARTICLE",
    5: "I-ARTICLE",
    6: "E-ARTICLE",
    7: "S-PAGE",
}
O, B, I, E = 0, 1, 2, 3


class _ConstrainedDecoder:
    """Stand-in for NERTagger that runs the real constrained Viterbi."""

    def __init__(self) -> None:
        trans, start, end = build_bioes_constraints(ID2LABEL)
        self.use_crf = False
        self.crf = None
        self._crf_trans = trans
        self._crf_start = start
        self._crf_end = end
        self.seen_lengths: list[int] = []

    def decode_constrained_doc(self, emissions: torch.Tensor) -> torch.Tensor:
        self.seen_lengths.append(int(emissions.size(0)))
        return NERTagger._viterbi_constrained_doc(self, emissions)  # pyright: ignore[reportArgumentType]


def _emissions(rows: list[int]) -> torch.Tensor:
    out = torch.full((len(rows), len(ID2LABEL)), -5.0)
    for t, lab in enumerate(rows):
        out[t, lab] = 5.0
    return out


def _make_inference() -> tuple[NERInference, _ConstrainedDecoder]:
    inference = NERInference.__new__(NERInference)
    inference.id2label = dict(ID2LABEL)
    inference.label2id = {v: k for k, v in ID2LABEL.items()}
    inference.device = "cpu"
    decoder = _ConstrainedDecoder()
    inference.model = decoder  # pyright: ignore[reportAttributeAccessIssue]
    return inference, decoder


class FirstSubwordDecodeTests(unittest.TestCase):
    # "Section 10.6 Notices": "10.6" is "10" "." "6", "Notices" is "Not" "ices".
    # Continuation subwords are never trained, so their emissions are noise.
    # Here "6" points at E-SECTION, which forces the next word to restart.
    WORD_IDS: list[int | None] = [0, 1, 1, 1, 2, 2]
    EMISSIONS = [B, I, I, E, E, E]

    def test_continuation_subwords_do_not_split_heading(self) -> None:
        inference, decoder = _make_inference()

        preds = inference._decode_first_subwords(_emissions(self.EMISSIONS), self.WORD_IDS)

        self.assertEqual(decoder.seen_lengths, [3])
        self.assertEqual(preds, [B, I, I, I, E, E])

    def test_full_subword_decode_splits_heading(self) -> None:
        # Documents the old behavior this change removes: decoding every subword
        # gives "Notices" a B-SECTION, so the printer opens a second tag.
        _, decoder = _make_inference()
        old = decoder.decode_constrained_doc(_emissions(self.EMISSIONS))
        self.assertTrue(torch.equal(old, torch.tensor([B, I, I, E, B, E])))

    def test_special_tokens_get_outside_label(self) -> None:
        inference, decoder = _make_inference()
        logits = _emissions([B, B, E, I, E])
        word_ids: list[int | None] = [None, 0, 0, 1, None]

        preds = inference._decode_first_subwords(logits, word_ids)

        self.assertEqual(decoder.seen_lengths, [2])
        self.assertEqual(preds, [O, B, B, E, O])

    def test_no_words_returns_outside(self) -> None:
        inference, decoder = _make_inference()
        preds = inference._decode_first_subwords(_emissions([B, E]), [None, None])
        self.assertEqual(preds, [O, O])
        self.assertEqual(decoder.seen_lengths, [])

    def test_collapse_reads_projected_label_when_first_subword_is_empty(self) -> None:
        inference, _ = _make_inference()
        inference.C = len(ID2LABEL)
        text = "Section 10.6"
        word_ids: list[int | None] = [0, 1, 1, 1]
        # First subword of "10.6" has an empty offset, so collapse keeps the next one.
        offsets = [(0, 7), (8, 8), (8, 10), (10, 12)]
        logits = _emissions([B, O, O, O])
        logits[1] = _emissions([E])[0]

        preds = inference._decode_first_subwords(logits, word_ids)
        collapsed, _, word_offsets, _, _ = inference._collapse_to_word_level(
            text, preds, [1.0] * 4, offsets, logits, word_ids
        )

        self.assertEqual(word_offsets, [(0, 7), (8, 12)])
        self.assertEqual(collapsed, [B, E])


class CollectEvalDocsTests(unittest.TestCase):
    def test_eval_decodes_only_labeled_positions(self) -> None:
        decoder = _ConstrainedDecoder()
        ns = SimpleNamespace(
            _tok_sum={0: _emissions([B, I, E, B, E])},
            _tok_cnt={0: torch.ones(5)},
            _tok_gold={0: torch.tensor([B, I, -100, -100, E])},
            _tok_offsets={0: [(0, 7), (8, 10), (10, 11), (11, 12), (13, 20)]},
            _doc_raw={0: "Section 10.6 Notices"},
            ignore_index=-100,
            id2label=ID2LABEL,
            _viterbi_constrained_doc=decoder.decode_constrained_doc,
        )

        records = NERTagger.collect_eval_docs(ns)  # pyright: ignore[reportArgumentType]

        self.assertEqual(decoder.seen_lengths, [3])
        self.assertEqual(records[0]["pred_tags_raw"], ["B-SECTION", "I-SECTION", "E-SECTION"])


if __name__ == "__main__":
    _ = unittest.main()
