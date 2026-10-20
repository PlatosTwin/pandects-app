import itertools
import math
import unittest
from typing import cast

import sklearn_crfsuite

from etl.models.page_classifier_revamp.crf_pipeline import (
    LABEL_TO_INDEX,
    DecodableCRFProtocol,
    build_feature_sequences,
    fit_tfidf_vectorizer,
    guard_back_matter_start,
    monotonic_transition_overrides,
    predict_with_transition_overrides,
)
from etl.models.page_classifier_revamp.page_types import PAGE_LABELS, AgreementDocument, FeatureDict
from etl.models.page_classifier_revamp.typesafe_judgments import PageJudgment


class _FakeCRF:
    def __init__(
        self,
        state_features: dict[tuple[str, str], float],
        transition_features: dict[tuple[str, str], float],
    ) -> None:
        self.classes_: list[str] = list(PAGE_LABELS)
        self.state_features_: dict[tuple[str, str], float] = state_features
        self.transition_features_: dict[tuple[str, str], float] = transition_features


def _sequence_score(
    model: _FakeCRF,
    sequence: list[FeatureDict],
    labels: tuple[str, ...],
    overrides: dict[tuple[str, str], float],
) -> float:
    score = 0.0
    for feature_dict, label in zip(sequence, labels, strict=True):
        for name, value in feature_dict.items():
            score += float(value) * model.state_features_.get((name, label), 0.0)
    for previous_label, label in zip(labels, labels[1:]):
        transition = (previous_label, label)
        score += overrides.get(transition, model.transition_features_.get(transition, 0.0))
    return score


def _brute_force(
    model: _FakeCRF,
    sequence: list[FeatureDict],
    overrides: dict[tuple[str, str], float],
) -> tuple[list[str], list[dict[str, float]]]:
    paths = list(itertools.product(PAGE_LABELS, repeat=len(sequence)))
    scores = [_sequence_score(model, sequence, path, overrides) for path in paths]
    best_path = paths[max(range(len(paths)), key=lambda index: scores[index])]
    peak = max(scores)
    weights = [math.exp(score - peak) for score in scores]
    total = sum(weights)
    marginals = [{label: 0.0 for label in PAGE_LABELS} for _ in sequence]
    for path, weight in zip(paths, weights, strict=True):
        for page_index, label in enumerate(path):
            marginals[page_index][label] += weight / total
    return list(best_path), marginals


# Page 2 looks like a toc page in the middle of the body, which the learned
# transitions alone do not rule out.
MODEL = _FakeCRF(
    state_features={
        ("cover", "front_matter"): 3.0,
        ("text", "body"): 2.0,
        ("dots", "toc"): 4.0,
        ("dots", "body"): 0.5,
        ("witness", "sig"): 3.0,
    },
    transition_features={
        ("front_matter", "body"): 0.5,
        ("body", "body"): 1.0,
        ("body", "toc"): -0.5,
        ("toc", "body"): 0.5,
        ("body", "sig"): 0.5,
    },
)
SEQUENCE: list[FeatureDict] = [
    {"cover": 1.0},
    {"text": 1.0},
    {"dots": 1.0, "text": 0.3},
    {"text": 1.0},
    {"witness": True, "text": False},
]


class PredictWithTransitionOverridesTests(unittest.TestCase):
    def test_matches_brute_force_without_overrides(self) -> None:
        predicted, marginals = predict_with_transition_overrides(MODEL, [SEQUENCE], {})
        expected_path, expected_marginals = _brute_force(MODEL, SEQUENCE, {})

        self.assertEqual(predicted[0], expected_path)
        self.assertEqual(predicted[0][2], "toc")
        for row, expected_row in zip(marginals[0], expected_marginals, strict=True):
            for label in PAGE_LABELS:
                self.assertAlmostEqual(row[label], expected_row[label], places=9)

    def test_monotonic_overrides_forbid_backward_transitions(self) -> None:
        overrides = monotonic_transition_overrides()
        predicted, marginals = predict_with_transition_overrides(
            MODEL, [SEQUENCE], overrides
        )
        expected_path, expected_marginals = _brute_force(MODEL, SEQUENCE, overrides)

        self.assertEqual(predicted[0], expected_path)
        self.assertEqual(predicted[0], ["front_matter", "body", "body", "body", "sig"])
        for row, expected_row in zip(marginals[0], expected_marginals, strict=True):
            for label in PAGE_LABELS:
                self.assertAlmostEqual(row[label], expected_row[label], places=9)

    def test_monotonic_overrides_cover_every_backward_pair(self) -> None:
        overrides = monotonic_transition_overrides(illegal_transition_weight=-5.0)

        self.assertEqual(len(overrides), 10)
        for (source_label, target_label), weight in overrides.items():
            self.assertLess(LABEL_TO_INDEX[target_label], LABEL_TO_INDEX[source_label])
            self.assertEqual(weight, -5.0)

    def test_empty_sequence(self) -> None:
        predicted, marginals = predict_with_transition_overrides(MODEL, [[]], {})

        self.assertEqual(predicted, [[]])
        self.assertEqual(marginals, [[]])

    def test_reproduces_crfsuite_tagger_without_overrides(self) -> None:
        x_train: list[list[FeatureDict]] = [
            [{"cover": 1.0}, {"dots": 1.0}, {"text": 1.0}, {"text": 0.8}, {"witness": True}, {"exhibit": 1.0}],
            [{"cover": 0.9}, {"text": 1.0}, {"text": 1.0, "dots": 0.2}, {"witness": True}],
            [{"dots": 1.0}, {"dots": 0.7}, {"text": 1.0}, {"witness": True}, {"exhibit": 0.6, "text": 0.5}],
        ]
        y_train = [
            ["front_matter", "toc", "body", "body", "sig", "back_matter"],
            ["front_matter", "body", "body", "sig"],
            ["toc", "toc", "body", "sig", "back_matter"],
        ]
        crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs", c1=0.01, c2=0.01, max_iterations=50, all_possible_transitions=True
        )
        _ = crf.fit(x_train, y_train)
        x_test: list[list[FeatureDict]] = [
            [{"cover": 1.0}, {"text": 0.6, "dots": 0.6}, {"text": 1.0}, {"witness": True}, {"exhibit": 1.0}],
        ]

        predicted, marginals = predict_with_transition_overrides(
            cast(DecodableCRFProtocol, cast(object, crf)), x_test, {}
        )

        self.assertEqual(predicted, [[str(label) for label in sequence] for sequence in crf.predict(x_test)])
        for row, expected_row in zip(marginals[0], crf.predict_marginals(x_test)[0], strict=True):
            for label, probability in expected_row.items():
                self.assertAlmostEqual(row[str(label)], float(probability), places=3)

    def test_blocked_labels_are_avoided(self) -> None:
        blocked = [[frozenset[str](), frozenset[str](), frozenset({"toc"}), frozenset[str](), frozenset[str]()]]

        predicted, _ = predict_with_transition_overrides(MODEL, [SEQUENCE], {}, blocked)

        self.assertEqual(predicted[0], ["front_matter", "body", "body", "body", "sig"])

    def test_blocked_labels_must_match_page_count(self) -> None:
        with self.assertRaises(ValueError):
            _ = predict_with_transition_overrides(MODEL, [SEQUENCE], {}, [[frozenset()]])


# Back matter's self-transition outweighs body's, so six body-ish pages followed by an exhibit
# decode as back matter throughout.
COLLAPSING_MODEL = _FakeCRF(
    state_features={
        ("text", "body"): 1.0,
        ("text", "back_matter"): 0.8,
        ("exhibit", "back_matter"): 3.0,
    },
    transition_features={
        ("body", "body"): 0.5,
        ("back_matter", "back_matter"): 2.0,
    },
)
COLLAPSING_TEXTS = ["The parties agree as follows."] * 6 + ["Exhibit A"]
# Real page features so postprocessing can run; the fake model only weighs "text" and "exhibit".
COLLAPSING_SEQUENCE: list[FeatureDict] = [
    {**feature_dict, ("exhibit" if page_index == 6 else "text"): 1.0}
    for page_index, feature_dict in enumerate(
        build_feature_sequences(
            [AgreementDocument("a", COLLAPSING_TEXTS, [])],
            fit_tfidf_vectorizer(COLLAPSING_TEXTS, max_features=10),
        )[0]
    )
]


def _judgments(attachment_starts: list[float]) -> list[PageJudgment]:
    return [
        {
            "part_probs": {label: 1.0 if label == "body" else 0.0 for label in PAGE_LABELS},
            "main_sig": 0.0,
            "attachment_start": attachment_start,
        }
        for attachment_start in attachment_starts
    ]


class GuardBackMatterStartTests(unittest.TestCase):
    def _decode(self) -> tuple[list[str], list[dict[str, float]]]:
        labels, marginals = predict_with_transition_overrides(
            COLLAPSING_MODEL, [COLLAPSING_SEQUENCE], monotonic_transition_overrides()
        )
        return labels[0], marginals[0]

    def test_back_matter_without_evidence_is_pushed_to_the_next_evidence_page(self) -> None:
        labels, marginals = self._decode()
        self.assertEqual(labels, ["back_matter"] * 7)

        guarded, guarded_marginals, _ = guard_back_matter_start(
            COLLAPSING_MODEL,
            COLLAPSING_SEQUENCE,
            _judgments([0.0] * 6 + [0.9]),
            monotonic_transition_overrides(),
            labels,
            marginals,
            [False] * 7,
        )

        self.assertEqual(guarded, ["body"] * 6 + ["back_matter"])
        self.assertLess(guarded_marginals[0]["back_matter"], 1e-6)

    def test_back_matter_with_evidence_is_kept(self) -> None:
        labels, marginals = self._decode()

        guarded, _, _ = guard_back_matter_start(
            COLLAPSING_MODEL,
            COLLAPSING_SEQUENCE,
            _judgments([0.5] + [0.0] * 6),
            monotonic_transition_overrides(),
            labels,
            marginals,
            [False] * 7,
        )

        self.assertEqual(guarded, labels)


if __name__ == "__main__":
    _ = unittest.main()
