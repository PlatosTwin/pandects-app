import re
import tempfile
import unittest
from pathlib import Path
from typing import cast

from etl.models.exhibit_classifier.exhibit_classifier import ExhibitClassifier


def _normalize_text_for_overlap_check(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _contains_normalized_phrase(*, text: str, phrase: str) -> bool:
    pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
    return re.search(pattern, text) is not None


def _build_training_data(*, repeats: int = 1) -> tuple[list[str], list[int]]:
    positives = [
        "agreement and plan of merger effective time surviving corporation stockholder approval",
        "merger agreement acquisition closing conditions merger consideration effective time",
        "asset purchase agreement acquisition consideration board of directors",
        "stock purchase agreement acquisition closing representations and warranties",
        "agreement and plan of merger antitrust clearance hsr act",
        "merger agreement merger consideration effective time board recommendation",
    ]
    negatives = [
        "license agreement consulting services employment terms and compensation",
        "credit agreement loan agreement promissory note and collateral",
        "termination agreement cooperation agreement services agreement",
        "supply agreement lease agreement consulting agreement fees",
        "employment agreement compensation benefits and consulting obligations",
        "loan agreement credit agreement and promissory note",
    ]
    base_texts = positives + negatives
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if repeats == 1:
        texts = base_texts
    else:
        texts = [" ".join([text] * repeats) for text in base_texts]
    labels = [1] * len(positives) + [0] * len(negatives)
    return texts, labels


class ExhibitClassifierTests(unittest.TestCase):
    def test_start_features_detect_title_and_hard_negative(self) -> None:
        classifier = ExhibitClassifier()
        text = (
            "EX-2.1 AGREEMENT AND PLAN OF MERGER by and among A, B and C. "
            "This filing includes a VOTING AGREEMENT as an ancillary addendum. "
            "A second VOTING AGREEMENT appears in the same opening section. "
            "Additional merger terms follow."
        )
        features = classifier.extract_document_features(text)
        num_words = cast(float, features[0])
        start_has_title = cast(float, features[3])
        start_has_title_pos = cast(float, features[4])
        start_hard_negative = cast(float, features[5])
        start_hard_negative_count = cast(float, features[6])
        start_hard_negative_pos = cast(float, features[7])
        start_hard_negative_all_caps = cast(float, features[8])
        start_hard_negative_all_caps_count = cast(float, features[9])
        # num_words
        self.assertGreater(num_words, 0.0)
        # start_has_agreement_title
        self.assertEqual(start_has_title, 1.0)
        # start_has_agreement_title_pos
        self.assertGreaterEqual(start_has_title_pos, 0.0)
        self.assertLessEqual(start_has_title_pos, 1.0)
        # start_hard_negative
        self.assertEqual(start_hard_negative, 1.0)
        # start_hard_negative_count
        self.assertEqual(start_hard_negative_count, 2.0)
        # start_hard_negative_pos
        self.assertGreaterEqual(start_hard_negative_pos, 0.0)
        self.assertLessEqual(start_hard_negative_pos, 1.0)
        # start_hard_negative_all_caps
        self.assertEqual(start_hard_negative_all_caps, 1.0)
        # start_hard_negative_all_caps_count
        self.assertEqual(start_hard_negative_all_caps_count, 2.0)

    def test_tender_offer_does_not_trigger_hard_negative_phrase(self) -> None:
        classifier = ExhibitClassifier()
        text = (
            "TENDER OFFER AGREEMENT. The tender offer consideration is payable at closing."
        )
        features = classifier.extract_document_features(text)
        start_hard_negative = cast(float, features[5])
        start_hard_negative_count = cast(float, features[6])
        self.assertEqual(start_hard_negative, 0.0)
        self.assertEqual(start_hard_negative_count, 0.0)

    def test_keyword_count_uses_word_boundaries(self) -> None:
        classifier = ExhibitClassifier()
        text = "This machine maintenance memo covers same-day service obligations."
        features = classifier.extract_document_features(text)
        ma_keyword_count = cast(float, features[1])
        self.assertEqual(ma_keyword_count, 0.0)

    def test_keywords_and_hard_negatives_have_no_exact_overlap(self) -> None:
        classifier = ExhibitClassifier()
        normalized_keywords = {
            _normalize_text_for_overlap_check(keyword)
            for keyword in classifier.ma_keywords
        }
        normalized_hard_negatives = {
            _normalize_text_for_overlap_check(phrase)
            for phrase in classifier.ma_hard_negative_phrases
        }
        self.assertEqual(normalized_keywords & normalized_hard_negatives, set())

    def test_keywords_are_not_subphrases_of_hard_negatives(self) -> None:
        classifier = ExhibitClassifier()
        normalized_keywords = [
            _normalize_text_for_overlap_check(keyword)
            for keyword in classifier.ma_keywords
        ]
        normalized_hard_negatives = [
            _normalize_text_for_overlap_check(phrase)
            for phrase in classifier.ma_hard_negative_phrases
        ]
        conflicts = {
            (keyword, hard_negative)
            for keyword in normalized_keywords
            for hard_negative in normalized_hard_negatives
            if keyword != hard_negative
            and _contains_normalized_phrase(text=hard_negative, phrase=keyword)
        }
        self.assertEqual(conflicts, set())

    def test_predict_defaults_to_decision_threshold(self) -> None:
        classifier = ExhibitClassifier()
        classifier.is_fitted = True
        classifier.decision_threshold = 0.8
        classifier.predict_proba = lambda text: 0.75  # type: ignore[method-assign]
        long_text = "agreement and plan of merger " * 900
        self.assertFalse(classifier.predict(long_text))
        self.assertTrue(classifier.predict(long_text, threshold=0.7))

    def test_save_load_preserves_feature_configuration(self) -> None:
        texts, labels = _build_training_data(repeats=450)
        classifier = ExhibitClassifier()
        _ = classifier.fit(texts, labels=labels)
        classifier.start_scan_chars = 1234
        classifier.min_chars_hard_negative = 18000
        classifier.start_agreement_title_phrases = ["custom title phrase"]
        classifier.ma_hard_negative_phrases = ["custom negative phrase"]
        classifier.decision_threshold = 0.77

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            classifier.save(model_path)
            loaded = ExhibitClassifier.load(model_path)

        self.assertEqual(loaded.start_scan_chars, 1234)
        self.assertEqual(loaded.min_chars_hard_negative, 18000)
        self.assertEqual(loaded.start_agreement_title_phrases, ["custom title phrase"])
        self.assertEqual(loaded.ma_hard_negative_phrases, ["custom negative phrase"])
        self.assertAlmostEqual(loaded.decision_threshold, 0.77)

    def test_short_document_is_hard_blocked_negative(self) -> None:
        texts, labels = _build_training_data(repeats=450)
        classifier = ExhibitClassifier(min_chars_hard_negative=20000)
        _ = classifier.fit(texts, labels=labels)

        short_text = "agreement and plan of merger"
        self.assertEqual(classifier.predict_proba(short_text), 0.0)
        self.assertFalse(classifier.predict(short_text, threshold=0.0))
        self.assertEqual(
            classifier.predict_proba_batch([short_text, short_text]), [0.0, 0.0]
        )
        self.assertEqual(
            classifier.predict_batch([short_text, short_text], threshold=0.0),
            [False, False],
        )

    def test_fit_rejects_short_positive_examples(self) -> None:
        texts, labels = _build_training_data()
        classifier = ExhibitClassifier(min_chars_hard_negative=20000)
        with self.assertRaisesRegex(RuntimeError, "positive labels with text length"):
            _ = classifier.fit(texts, labels=labels)


if __name__ == "__main__":
    _ = unittest.main()
