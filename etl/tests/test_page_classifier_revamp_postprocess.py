import unittest

from etl.models.page_classifier_revamp.crf_pipeline import (
    enforce_monotonic_prediction_sequence,
    postprocess_prediction_sequence,
)


class PageClassifierRevampPostprocessTests(unittest.TestCase):
    def test_enforce_monotonic_prediction_sequence_clamps_backward_labels(self) -> None:
        labels = [
            "front_matter",
            "toc",
            "body",
            "sig",
            "back_matter",
            "body",
            "sig",
            "back_matter",
        ]

        clamped, modified_mask = enforce_monotonic_prediction_sequence(labels)

        self.assertEqual(
            clamped,
            [
                "front_matter",
                "toc",
                "body",
                "sig",
                "back_matter",
                "back_matter",
                "back_matter",
                "back_matter",
            ],
        )
        self.assertEqual(
            modified_mask,
            [False, False, False, False, False, True, True, False],
        )

    def test_postprocess_prediction_sequence_returns_hard_monotonic_sequence(self) -> None:
        labels = [
            "front_matter",
            "toc",
            "body",
            "sig",
            "back_matter",
            "body",
            "sig",
            "back_matter",
        ]
        feature_sequence = [
            {"relative_page": float(index + 1) / float(len(labels))}
            for index in range(len(labels))
        ]

        postprocessed, modified_mask = postprocess_prediction_sequence(
            labels,
            feature_sequence,
        )

        self.assertEqual(
            postprocessed,
            [
                "front_matter",
                "toc",
                "body",
                "sig",
                "back_matter",
                "back_matter",
                "back_matter",
                "back_matter",
            ],
        )
        self.assertEqual(
            modified_mask,
            [False, False, False, False, False, True, True, False],
        )

    def test_postprocess_prediction_sequence_clamps_without_back_matter(self) -> None:
        labels = ["front_matter", "body", "toc", "body", "sig"]
        feature_sequence = [
            {"relative_page": float(index + 1) / float(len(labels))}
            for index in range(len(labels))
        ]

        postprocessed, modified_mask = postprocess_prediction_sequence(
            labels,
            feature_sequence,
        )

        self.assertEqual(postprocessed, ["front_matter", "body", "body", "body", "sig"])
        self.assertEqual(modified_mask, [False, False, True, False, False])


if __name__ == "__main__":
    _ = unittest.main()
