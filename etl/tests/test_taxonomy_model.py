# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import cast
from unittest.mock import patch

import lightning.pytorch as pl
import pandas as pd
import torch
import numpy as np
from scipy.sparse import csr_matrix

from etl.models.taxonomy.taxonomy import TaxonomyConfig, TaxonomyTrainer
from etl.models.taxonomy.taxonomy_classes import TaxonomyClassifier


class TaxonomyModelTests(unittest.TestCase):
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
            "etl.models.taxonomy.taxonomy.TaxonomyClassifier.load_from_checkpoint",
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


if __name__ == "__main__":
    _ = unittest.main()
