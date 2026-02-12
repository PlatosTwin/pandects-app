# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false
import unittest

import pandas as pd
import torch

from etl.models.taxonomy.taxonomy import TaxonomyConfig, TaxonomyTrainer
from etl.models.taxonomy.taxonomy_classes import TaxonomyClassifier


class TaxonomyModelTests(unittest.TestCase):
    def test_parse_labels_cell_supports_multilabel_and_legacy(self) -> None:
        self.assertEqual(TaxonomyTrainer._parse_labels_cell(["a", "b"]), ["a", "b"])
        self.assertEqual(TaxonomyTrainer._parse_labels_cell(("a", "b")), ["a", "b"])
        self.assertEqual(TaxonomyTrainer._parse_labels_cell("a"), ["a"])
        self.assertEqual(
            TaxonomyTrainer._parse_labels_cell('["a", "b"]'),
            ["a", "b"],
        )
        with self.assertRaises(ValueError):
            _ = TaxonomyTrainer._parse_labels_cell(123)

    def test_group_split_avoids_agreement_leakage(self) -> None:
        cfg = TaxonomyConfig(
            data_parquet="unused.parquet",
            model_name="unused",
            label_list=["x", "y"],
            mode="tfidf",
            num_trials=1,
            max_epochs=1,
            split_group_column="agreement_uuid",
            val_size=0.5,
        )
        trainer = TaxonomyTrainer(cfg)
        df = pd.DataFrame(
            {
                "agreement_uuid": ["a", "a", "b", "b", "c", "c"],
                "article_title": ["A"] * 6,
                "section_title": ["S"] * 6,
                "section_text": ["T"] * 6,
                "label_vectors": [[1, 0]] * 6,
            }
        )
        tr, va = trainer._split_train_val(df)
        self.assertGreater(len(tr), 0)
        self.assertGreater(len(va), 0)
        self.assertTrue(
            set(tr["agreement_uuid"].astype(str)).isdisjoint(
                set(va["agreement_uuid"].astype(str))
            )
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


if __name__ == "__main__":
    _ = unittest.main()
