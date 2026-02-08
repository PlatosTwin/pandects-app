import tempfile
import unittest
from pathlib import Path
from typing import cast

from etl.models.exhibit_classifier.exhibit_classifier import ExhibitClassifier


def _build_training_data() -> tuple[list[str], list[int]]:
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
    texts = positives + negatives
    labels = [1] * len(positives) + [0] * len(negatives)
    return texts, labels


class ExhibitClassifierTests(unittest.TestCase):
    def test_start_features_detect_title_and_hard_negative(self) -> None:
        classifier = ExhibitClassifier()
        text = (
            "EX-2.1 AGREEMENT AND PLAN OF MERGER by and among A, B and C. "
            "This filing includes a Voting Agreement as an ancillary exhibit. "
            "Additional merger terms follow."
        )
        features = classifier.extract_document_features(text)
        num_words = cast(float, features[0])
        start_has_title = cast(float, features[3])
        start_has_title_pos = cast(float, features[4])
        start_hard_negative = cast(float, features[5])
        start_hard_negative_pos = cast(float, features[6])
        # num_words
        self.assertGreater(num_words, 0.0)
        # start_has_agreement_title
        self.assertEqual(start_has_title, 1.0)
        # start_has_agreement_title_pos
        self.assertGreaterEqual(start_has_title_pos, 0.0)
        self.assertLessEqual(start_has_title_pos, 1.0)
        # start_hard_negative
        self.assertEqual(start_hard_negative, 1.0)
        # start_hard_negative_pos
        self.assertGreaterEqual(start_hard_negative_pos, 0.0)
        self.assertLessEqual(start_hard_negative_pos, 1.0)

    def test_predict_defaults_to_decision_threshold(self) -> None:
        classifier = ExhibitClassifier()
        classifier.decision_threshold = 0.8
        classifier.predict_proba = lambda text: 0.75  # type: ignore[method-assign]
        self.assertFalse(classifier.predict("agreement and plan of merger"))
        self.assertTrue(classifier.predict("agreement and plan of merger", threshold=0.7))

    def test_save_load_preserves_feature_configuration(self) -> None:
        texts, labels = _build_training_data()
        classifier = ExhibitClassifier()
        _ = classifier.fit(texts, labels=labels)
        classifier.start_scan_chars = 1234
        classifier.start_agreement_title_phrases = ["custom title phrase"]
        classifier.ma_hard_negative_phrases = ["custom negative phrase"]
        classifier.decision_threshold = 0.77

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            classifier.save(model_path)
            loaded = ExhibitClassifier.load(model_path)

        self.assertEqual(loaded.start_scan_chars, 1234)
        self.assertEqual(loaded.start_agreement_title_phrases, ["custom title phrase"])
        self.assertEqual(loaded.ma_hard_negative_phrases, ["custom negative phrase"])
        self.assertAlmostEqual(loaded.decision_threshold, 0.77)


if __name__ == "__main__":
    _ = unittest.main()
