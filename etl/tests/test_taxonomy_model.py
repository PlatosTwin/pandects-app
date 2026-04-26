# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import ANY, patch

import lightning.pytorch as pl
import pandas as pd
import torch
import numpy as np
from optuna import Trial
from optuna.exceptions import TrialPruned
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

from etl.defs.resources import TaxonomyModel
from etl.models.taxonomy.ml.taxonomy import (
    TaxonomyConfig,
    TaxonomyInference,
    TaxonomyTrainer,
)
from etl.models.taxonomy.ml.taxonomy_classes import TaxonomyClassifier


class TaxonomyModelTests(unittest.TestCase):
    def test_tfidf_training_step_raises_on_non_finite_logits(self) -> None:
        model = TaxonomyClassifier(
            mode="tfidf",
            num_labels=2,
            id2label={0: "a", 1: "b"},
            input_dim=4,
            hidden_dim=8,
            dropout=0.0,
        )
        batch = (
            torch.zeros((1, 4), dtype=torch.float32),
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        )
        with patch.object(
            model,
            "forward",
            return_value=SimpleNamespace(
                logits=torch.tensor([[float("nan"), 0.0]], dtype=torch.float32)
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "Non-finite train_logits"):
                _ = model.training_step(batch, 0)

    def test_parse_labels_cell_supports_multilabel_inputs(self) -> None:
        self.assertEqual(TaxonomyTrainer._parse_labels_cell(["a", "b"]), ["a", "b"])
        self.assertEqual(TaxonomyTrainer._parse_labels_cell(("a", "b")), ["a", "b"])
        self.assertEqual(TaxonomyTrainer._parse_labels_cell("a"), ["a"])
        self.assertEqual(
            TaxonomyTrainer._parse_labels_cell('["a", "b"]'),
            ["a", "b"],
        )
        with self.assertRaises(ValueError):
            _ = TaxonomyTrainer._parse_labels_cell(123)

    def test_title_rules_learn_section_and_article_section_mappings(self) -> None:
        cfg = TaxonomyConfig(
            data_parquet="unused.parquet",
            model_name="unused",
            label_list=["x", "y"],
            mode="tfidf",
            num_trials=1,
            max_epochs=1,
            title_rule_min_support=2,
            title_rule_min_precision=1.0,
            title_rule_min_margin=1,
            title_rule_prefix_min_support=2,
        )
        trainer = TaxonomyTrainer(cfg)
        df = pd.DataFrame(
            {
                "article_title": [
                    "miscellaneous",
                    "miscellaneous",
                    "general provisions",
                    "general provisions",
                    "general provisions",
                    "other",
                ],
                "section_title": [
                    "governing law",
                    "governing law",
                    "governing law",
                    "governing law",
                    "notices",
                    "notices",
                ],
                "parsed_labels": [
                    ["law"],
                    ["law"],
                    ["law"],
                    ["law"],
                    ["notice"],
                    ["notice"],
                ],
            }
        )
        trainer._build_title_rules(train_df=df, label_list=["law", "notice"])
        self.assertEqual(
            trainer._resolve_title_rule_label("miscellaneous", "governing law"),
            "law",
        )
        self.assertEqual(
            trainer._resolve_title_rule_label("general provisions", "notices"),
            "notice",
        )
        self.assertEqual(
            trainer._resolve_title_rule_label(
                "miscellaneous",
                "governing law and venue",
            ),
            "law",
        )

    def test_tfidf_classifier_forward_multi_label_logits(self) -> None:
        model = TaxonomyClassifier(
            mode="tfidf",
            num_labels=3,
            id2label={0: "a", 1: "b", 2: "c"},
            input_dim=4,
            hidden_dim=8,
            dropout=0.0,
        )
        batch_features = torch.randn(2, 4)
        logits = torch.as_tensor(model(features=batch_features).logits)
        self.assertEqual(tuple(logits.shape), (2, 3))
        labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        loss = model.loss_fn(logits, labels)
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_title_rule_boost_updates_probability_matrix(self) -> None:
        cfg = TaxonomyConfig(
            data_parquet="unused.parquet",
            model_name="unused",
            label_list=["a", "b"],
            mode="tfidf",
            num_trials=1,
            max_epochs=1,
            title_rule_boost_probability=0.97,
        )
        trainer = TaxonomyTrainer(cfg)
        trainer.section_title_rules = {"governing law": "b"}
        probs = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
        boosted = trainer._apply_title_rule_boost(
            probs=probs,
            article_titles=["miscellaneous", "miscellaneous"],
            section_titles=["governing law", "notices"],
            label_list=["a", "b"],
        )
        self.assertAlmostEqual(float(boosted[0, 1]), 0.97)
        self.assertAlmostEqual(float(boosted[1, 1]), 0.4)

    def test_load_data_treats_empty_labels_as_explicit_class(self) -> None:
        df = pd.DataFrame(
            {
                "article_title": [
                    "misc",
                    "misc",
                    "closing",
                    "closing",
                    "tax",
                    "tax",
                ],
                "section_title": [
                    "notices",
                    "notices",
                    "conditions",
                    "conditions",
                    "governing law",
                    "governing law",
                ],
                "section_text": [
                    "<p>notice text</p>",
                    "<p>notice text</p>",
                    "<p>closing text</p>",
                    "<p>closing text</p>",
                    "<p>law text</p>",
                    "<p>law text</p>",
                ],
                "label": ["notice", "notice", "", "", "law", "law"],
            }
        )
        with TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "taxonomy.parquet"
            df.to_parquet(parquet_path)
            cfg = TaxonomyConfig(
                data_parquet=str(parquet_path),
                model_name="unused",
                label_list=None,
                mode="tfidf",
                num_trials=1,
                max_epochs=1,
                val_size=0.34,
                test_size=0.0,
                empty_label_name="__empty__",
            )
            trainer = TaxonomyTrainer(cfg)
            trainer._load_data()

        label_list = trainer.cfg.label_list
        self.assertIsNotNone(label_list)
        assert label_list is not None
        self.assertIn("__empty__", label_list)
        self.assertIsInstance(trainer.X_train, csr_matrix)
        self.assertIsInstance(trainer.X_val, csr_matrix)
        self.assertIsInstance(trainer.X_test, csr_matrix)

        train_vectors = cast(list[list[int]], trainer.train_rows["label_vectors"])
        val_vectors = cast(list[list[int]], trainer.val_rows["label_vectors"])
        test_vectors = cast(list[list[int]], trainer.test_rows["label_vectors"])
        all_vectors = [
            *train_vectors,
            *val_vectors,
            *test_vectors,
        ]
        self.assertTrue(all(sum(vec) >= 1 for vec in all_vectors))
        empty_label_idx = label_list.index("__empty__")
        self.assertTrue(any(vec[empty_label_idx] == 1 for vec in all_vectors))

    def test_grouped_split_keeps_duplicate_text_in_single_partition(self) -> None:
        df = pd.DataFrame(
            {
                "article_title": [
                    "a1",
                    "a1",
                    "a2",
                    "a2",
                    "a3",
                    "a3",
                    "a4",
                    "a4",
                    "a5",
                    "a5",
                ],
                "section_title": [
                    "s1",
                    "s1",
                    "s2",
                    "s2",
                    "s3",
                    "s3",
                    "s4",
                    "s4",
                    "s5",
                    "s5",
                ],
                "section_text": [
                    "<p>x1</p>",
                    "<p>x1</p>",
                    "<p>x2</p>",
                    "<p>x2</p>",
                    "<p>x3</p>",
                    "<p>x3</p>",
                    "<p>x4</p>",
                    "<p>x4</p>",
                    "<p>x5</p>",
                    "<p>x5</p>",
                ],
                "label": ["l1", "l1", "l2", "l2", "", "", "l3", "l3", "l1", "l1"],
            }
        )
        with TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "taxonomy.parquet"
            df.to_parquet(parquet_path)
            cfg = TaxonomyConfig(
                data_parquet=str(parquet_path),
                model_name="unused",
                label_list=None,
                mode="tfidf",
                num_trials=1,
                max_epochs=1,
                val_size=0.34,
                test_size=0.33,
            )
            trainer = TaxonomyTrainer(cfg)
            trainer._load_data()

        def _keys(rows: dict[str, list[str] | list[list[int]]]) -> set[tuple[str, str, str]]:
            article = [str(x) for x in rows["article_title"]]
            section = [str(x) for x in rows["section_title"]]
            text = [str(x) for x in rows["section_text"]]
            return {(a, s, t) for a, s, t in zip(article, section, text)}

        train_keys = _keys(trainer.train_rows)
        val_keys = _keys(trainer.val_rows)
        test_keys = _keys(trainer.test_rows)

        self.assertEqual(train_keys & val_keys, set())
        self.assertEqual(train_keys & test_keys, set())
        self.assertEqual(val_keys & test_keys, set())

    def test_threshold_tuning_skips_low_support_labels(self) -> None:
        cfg = TaxonomyConfig(
            data_parquet="unused.parquet",
            model_name="unused",
            label_list=["rare", "common"],
            mode="tfidf",
            num_trials=1,
            max_epochs=1,
            decision_threshold=0.5,
            threshold_tuning_min_support=2,
        )
        trainer = TaxonomyTrainer(cfg)
        trainer.val_rows = {
            "article_title": ["a", "a", "a", "a"],
            "section_title": ["s", "s", "s", "s"],
            "section_text": ["x", "x", "x", "x"],
            "label_vectors": [[1, 1], [0, 1], [0, 1], [0, 0]],
        }
        probs = np.asarray(
            [
                [0.92, 0.55],
                [0.40, 0.65],
                [0.30, 0.80],
                [0.10, 0.20],
            ],
            dtype=np.float64,
        )
        y_true = np.asarray(
            [
                [1, 1],
                [0, 1],
                [0, 1],
                [0, 0],
            ],
            dtype=np.int64,
        )

        with patch(
            "etl.models.taxonomy.ml.taxonomy.TaxonomyClassifier.load_from_checkpoint",
            return_value=object(),
        ):
            trainer._collect_val_probs_and_labels = lambda model, dm: (probs, y_true)
            payload = trainer._tune_per_class_thresholds(
                checkpoint_path="unused.ckpt",
                dm=pl.LightningDataModule(),
            )

        thresholds_by_label = cast(dict[str, float], payload["decision_thresholds_by_label"])
        per_label = cast(dict[str, dict[str, object]], payload["per_label"])
        self.assertEqual(float(thresholds_by_label["rare"]), 0.5)
        skipped_count = cast(int, payload["num_labels_skipped_low_support"])
        self.assertEqual(skipped_count, 1)
        self.assertFalse(bool(per_label["rare"]["tuned"]))
        self.assertEqual(str(per_label["rare"]["skip_reason"]), "low_support")
        self.assertTrue(bool(per_label["common"]["tuned"]))

    def test_model_score_components_apply_recall_penalty(self) -> None:
        cfg = TaxonomyConfig(
            data_parquet="unused.parquet",
            model_name="unused",
            label_list=["a"],
            mode="tfidf",
            num_trials=1,
            max_epochs=1,
            model_score_macro_weight=0.5,
            model_score_weighted_weight=0.2,
            model_score_micro_weight=0.2,
            model_score_subset_weight=0.1,
            model_score_recall_floor=0.8,
            model_score_recall_penalty=0.5,
        )
        trainer = TaxonomyTrainer(cfg)
        summary = {
            "f1_macro": 0.7,
            "f1_weighted": 0.6,
            "f1_micro": 0.65,
            "subset_accuracy": 0.5,
            "recall_micro": 0.4,
        }
        score, raw, penalty = trainer._model_score_components(summary=summary)
        self.assertAlmostEqual(raw, 0.65, places=6)
        self.assertAlmostEqual(penalty, 0.2, places=6)
        self.assertAlmostEqual(score, 0.45, places=6)

    def test_objective_returns_tuned_validation_score(self) -> None:
        cfg = TaxonomyConfig(
            data_parquet="unused.parquet",
            model_name="unused",
            label_list=["a"],
            mode="tfidf",
            num_trials=1,
            max_epochs=1,
        )
        trainer = TaxonomyTrainer(cfg)

        class _FakeTrial:
            def __init__(self) -> None:
                self.params: dict[str, float | int] = {}
                self.user_attrs: dict[str, float] = {}

            def suggest_categorical(self, name: str, choices: list[object]) -> object:
                value = choices[0]
                self.params[name] = cast(int, value)
                return value

            def suggest_float(
                self,
                name: str,
                low: float,
                _high: float,
                *,
                log: bool = False,
            ) -> float:
                _ = log
                self.params[name] = low
                return low

            def set_user_attr(self, name: str, value: float) -> None:
                self.user_attrs[name] = value

        fake_checkpoint = SimpleNamespace(best_model_path="best.ckpt")

        class _FakeTrainer:
            callback_metrics = {
                "val_model_score": torch.tensor(0.12),
            }

            def fit(self, model: object, datamodule: object) -> None:
                _ = model
                _ = datamodule
                return None

        fake_trial = _FakeTrial()
        fake_trainer = _FakeTrainer()

        with (
            patch.object(
                trainer,
                "_build",
                return_value=(pl.LightningDataModule(), object()),
            ),
            patch.object(
                trainer,
                "_get_callbacks",
                return_value=(fake_checkpoint, object(), object(), object(), []),
            ),
            patch("etl.models.taxonomy.ml.taxonomy.pl.Trainer", return_value=fake_trainer),
            patch.object(
                trainer,
                "_tune_per_class_thresholds",
                return_value={
                    "val_scores_at_selected_thresholds": {
                        "val_model_score": 0.91,
                    }
                },
            ),
        ):
            score = trainer._objective(cast(Trial, cast(object, fake_trial)))

        self.assertAlmostEqual(score, 0.91)
        self.assertAlmostEqual(
            fake_trial.user_attrs["val_model_score_fixed_threshold"],
            0.12,
        )
        self.assertAlmostEqual(fake_trial.user_attrs["val_model_score_tuned"], 0.91)

    def test_objective_prunes_non_finite_trials(self) -> None:
        cfg = TaxonomyConfig(
            data_parquet="unused.parquet",
            model_name="unused",
            label_list=["a"],
            mode="tfidf",
            num_trials=1,
            max_epochs=1,
        )
        trainer = TaxonomyTrainer(cfg)

        class _FakeTrial:
            def suggest_categorical(self, _name: str, choices: list[object]) -> object:
                return choices[0]

            def suggest_float(
                self,
                _name: str,
                low: float,
                _high: float,
                *,
                log: bool = False,
            ) -> float:
                _ = log
                return low

            def set_user_attr(self, _name: str, _value: float) -> None:
                return None

        fake_checkpoint = SimpleNamespace(best_model_path="")

        class _FakeTrainer:
            callback_metrics: dict[str, object] = {}

            def fit(self, model: object, datamodule: object) -> None:
                _ = model
                _ = datamodule
                raise RuntimeError("Non-finite train_loss detected (1/1 values).")

        fake_trainer = _FakeTrainer()

        with (
            patch.object(
                trainer,
                "_build",
                return_value=(pl.LightningDataModule(), object()),
            ),
            patch.object(
                trainer,
                "_get_callbacks",
                return_value=(fake_checkpoint, object(), object(), object(), []),
            ),
            patch("etl.models.taxonomy.ml.taxonomy.pl.Trainer", return_value=fake_trainer),
        ):
            with self.assertRaises(TrialPruned):
                _ = trainer._objective(cast(Trial, cast(object, _FakeTrial())))

    def test_inference_predict_returns_asset_compatible_fields(self) -> None:
        class _FakeVectorizer:
            def transform(self, texts: list[str]) -> csr_matrix:
                _ = texts
                return csr_matrix(
                    np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
                )

        class _FakeModel:
            def __call__(self, **_kwargs: object) -> SimpleNamespace:
                return SimpleNamespace(
                    logits=torch.tensor([[4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)
                )

        inference = TaxonomyInference.__new__(TaxonomyInference)
        inference.device = "cpu"
        inference.mode = "tfidf"
        inference.vectorizer = cast(
            TfidfVectorizer,
            cast(object, _FakeVectorizer()),
        )
        inference.model = cast(
            TaxonomyClassifier,
            cast(object, _FakeModel()),
        )
        inference.id2label = {0: "top", 1: "alt-a", 2: "alt-b", 3: "alt-c"}
        inference.label2id = {label: idx for idx, label in inference.id2label.items()}
        inference.decision_thresholds = [0.5, 0.5, 0.5, 0.5]
        inference.decision_threshold = 0.5
        inference.decision_threshold_tensor = torch.tensor(
            inference.decision_thresholds,
            dtype=torch.float32,
        )
        inference.section_title_rules = {}
        inference.article_section_title_rules = {}
        inference.section_title_prefix_rules = {}
        inference.section_title_prefix_rules_sorted = []
        inference.title_rule_boost_probability = 0.995

        def _combine(a: str, s: str, t: str) -> str:
            return f"{a} {s} {t}"

        inference._combine = _combine

        outputs = inference.predict(
            [
                {
                    "article_title": "misc",
                    "section_title": "notices",
                    "section_text": "body",
                }
            ]
        )

        self.assertEqual(outputs[0]["label"], "top")
        self.assertEqual(outputs[0]["alt_labels"], ["alt-a", "alt-b", "alt-c"])
        self.assertEqual(len(cast(list[float], outputs[0]["alt_probs"])), 3)

    def test_taxonomy_resource_loads_real_inference_once(self) -> None:
        resource = TaxonomyModel()
        fake_inference = object()
        with (
            patch.object(Path, "exists", return_value=True),
            patch("etl.defs.resources.TaxonomyInference", return_value=fake_inference) as inf_cls,
        ):
            first = resource.model()
            second = resource.model()

        self.assertIs(first, fake_inference)
        self.assertIs(second, fake_inference)
        inf_cls.assert_called_once_with(
            ckpt_path=ANY,
            label_list=None,
            mode=None,
            vectorizer_path=ANY,
            title_rules_path=ANY,
        )


if __name__ == "__main__":
    _ = unittest.main()
