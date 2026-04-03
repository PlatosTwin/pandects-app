# pyright: reportPrivateUsage=false
import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Callable, cast

import yaml

from etl.models.ner.config import (
    DEFAULT_DATA_PATH,
    build_config,
    load_frozen_experiment_config,
)
from etl.models.ner.ner import (
    _append_experiment_row,
    parse_boolish,
    parse_train_docs,
    recover_experiment_row_from_run_dir,
)


class NerXpSetupTests(unittest.TestCase):
    def test_frozen_experiment_config_loads(self) -> None:
        frozen = load_frozen_experiment_config()
        self.assertEqual(frozen["model_name"], "answerdotai/ModernBERT-base")
        self.assertEqual(frozen["batch_size"], 8)
        self.assertEqual(frozen["train_subsample_window"], 512)

    def test_default_data_path_prefers_parquet(self) -> None:
        self.assertTrue(DEFAULT_DATA_PATH.endswith("ner-data.parquet"))

    def test_build_config_accepts_crf_boundary_recipe(self) -> None:
        config = build_config(
            xp_name="sampler_crf_boundary_02",
            train_docs=0,
            sampling_mode="boundary_mix",
            decoder_mode="crf",
            boundary_head=True,
            boundary_loss_weight=0.2,
            token_loss_mode="ce",
            token_loss_weight=0.15,
            crf_loss_weight=0.65,
            label_smoothing=0.02,
            split_version="default",
            seed=42,
        )
        self.assertEqual(config.decoder_mode, "crf")
        self.assertTrue(config.boundary_head)
        self.assertEqual(config.label_smoothing, 0.02)

    def test_build_config_accepts_preserve_case_variant(self) -> None:
        config = build_config(
            xp_name="sampler_only_clean_preserve_case",
            train_docs=0,
            sampling_mode="boundary_mix",
            decoder_mode="independent",
            preserve_case=True,
            split_version="default",
            seed=42,
        )
        self.assertTrue(config.preserve_case)
        self.assertEqual(config.sampling_mode, "boundary_mix")

    def test_build_config_accepts_learning_rate_override(self) -> None:
        config = build_config(
            xp_name="sampler_only_clean_preserve_case_boundary_010_lr2e-5",
            train_docs=0,
            sampling_mode="boundary_mix",
            decoder_mode="independent",
            boundary_head=True,
            boundary_loss_weight=0.10,
            preserve_case=True,
            learning_rate=2.0e-5,
            split_version="default",
            seed=42,
        )
        self.assertEqual(config.learning_rate, 2.0e-5)

    def test_build_config_rejects_invalid_boundary_combo(self) -> None:
        with self.assertRaisesRegex(ValueError, "boundary_loss_weight"):
            _ = build_config(
                xp_name="invalid",
                train_docs=0,
                boundary_head=False,
                boundary_loss_weight=0.2,
                split_version="default",
                seed=42,
            )

    def test_grid_rows_form_valid_configs(self) -> None:
        frozen = load_frozen_experiment_config()
        grid_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "etl"
            / "models"
            / "ner"
            / "configs"
            / "grid.csv"
        )
        with grid_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 8)
        for row in rows:
            config = build_config(
                xp_name=row["xp_name"],
                train_docs=parse_train_docs(row["train_docs"]),
                sampling_mode=row["sampling_mode"],
                decoder_mode=row["decoder_mode"],
                boundary_head=parse_boolish(row["boundary_head"]),
                boundary_loss_weight=float(row["boundary_loss_weight"]),
                token_loss_mode=row["token_loss_mode"],
                token_loss_weight=float(row["token_loss_weight"]),
                crf_loss_weight=float(row["crf_loss_weight"]),
                label_smoothing=float(row["label_smoothing"]),
                preserve_case=parse_boolish(row["preserve_case"]),
                split_version="default",
                seed=int(row.get("seed", "42")),
                model_name=frozen["model_name"],
                batch_size=frozen["batch_size"],
                train_subsample_window=int(
                    row.get(
                        "train_subsample_window",
                        str(frozen["train_subsample_window"]),
                    )
                ),
                val_window=int(row.get("val_window", str(frozen["val_window"]))),
                val_stride=int(row.get("val_stride", str(frozen["val_stride"]))),
                max_epochs=frozen["max_epochs"],
                learning_rate=float(row.get("learning_rate", frozen["learning_rate"])),
                weight_decay=frozen["weight_decay"],
                warmup_steps_pct=frozen["warmup_steps_pct"],
            )
            self.assertEqual(config.xp_name, row["xp_name"])

    def test_grid_window_overrides_are_respected(self) -> None:
        frozen = load_frozen_experiment_config()
        grid_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "etl"
            / "models"
            / "ner"
            / "configs"
            / "grid.csv"
        )
        with grid_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        long_window_row = next(row for row in rows if row["row_id"] == "5")
        config = build_config(
            xp_name=long_window_row["xp_name"],
            train_docs=parse_train_docs(long_window_row["train_docs"]),
            sampling_mode=long_window_row["sampling_mode"],
            decoder_mode=long_window_row["decoder_mode"],
            boundary_head=parse_boolish(long_window_row["boundary_head"]),
            boundary_loss_weight=float(long_window_row["boundary_loss_weight"]),
            token_loss_mode=long_window_row["token_loss_mode"],
            token_loss_weight=float(long_window_row["token_loss_weight"]),
            crf_loss_weight=float(long_window_row["crf_loss_weight"]),
            label_smoothing=float(long_window_row["label_smoothing"]),
            preserve_case=parse_boolish(long_window_row["preserve_case"]),
            split_version="default",
            seed=int(long_window_row.get("seed", "42")),
            model_name=frozen["model_name"],
            batch_size=frozen["batch_size"],
            train_subsample_window=int(long_window_row["train_subsample_window"]),
            val_window=int(long_window_row["val_window"]),
            val_stride=int(long_window_row["val_stride"]),
            max_epochs=frozen["max_epochs"],
            learning_rate=float(
                long_window_row.get("learning_rate", frozen["learning_rate"])
            ),
            weight_decay=frozen["weight_decay"],
            warmup_steps_pct=frozen["warmup_steps_pct"],
        )
        self.assertEqual(config.train_subsample_window, 768)
        self.assertEqual(config.val_window, 768)
        self.assertEqual(config.val_stride, 384)

    def test_append_experiment_row_expands_header_for_new_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "experiments_xp.csv"
            _append_experiment_row(
                str(csv_path),
                {
                    "run_id": "old_run",
                    "xp_name": "old_xp",
                },
            )
            _append_experiment_row(
                str(csv_path),
                {
                    "run_id": "new_run",
                    "xp_name": "new_xp",
                    "preserve_case": True,
                },
            )

            with csv_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 2)
            self.assertIn("preserve_case", rows[0])
            self.assertEqual(rows[0]["run_id"], "old_run")
            self.assertEqual(rows[0]["preserve_case"], "")
            self.assertEqual(rows[1]["preserve_case"], "True")

    def test_recover_experiment_row_from_run_dir_appends_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "run_123"
            run_dir.mkdir(parents=True, exist_ok=True)
            config_payload = {
                "run_id": "run_123",
                "git_commit": "abc123",
                "xp_name": "sampler_only_clean",
                "split_version": "default_xp",
                "train_docs": 0,
                "sampling_mode": "boundary_mix",
                "decoder_mode": "independent",
                "boundary_head": False,
                "boundary_loss_weight": 0.0,
                "token_loss_mode": "focal",
                "token_loss_weight": 1.0,
                "crf_loss_weight": 0.0,
                "label_smoothing": 0.0,
                "preserve_case": True,
                "seed": 42,
            }
            metrics_payload = {
                "variants": {
                    "raw": {
                        "entity_level": {
                            "micro": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
                            "macro": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
                            "per_type": {
                                "ARTICLE": {
                                    "precision": 0.7,
                                    "recall": 0.6,
                                    "f1": 0.646,
                                    "support": 10,
                                }
                            },
                            "total_entities": 10,
                        },
                        "token_level": {"accuracy": 0.99},
                    },
                }
            }
            with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
                safe_dump = cast(Callable[..., object], yaml.safe_dump)
                _ = safe_dump(config_payload, f, sort_keys=False)
            with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(metrics_payload, f)

            experiments_csv = Path(tmpdir) / "experiments_xp.csv"
            row = recover_experiment_row_from_run_dir(
                run_dir=str(run_dir),
                experiments_csv=str(experiments_csv),
            )

            self.assertEqual(row["run_id"], "run_123")
            with experiments_csv.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "run_123")
            self.assertEqual(rows[0]["preserve_case"], "True")


if __name__ == "__main__":
    _ = unittest.main()
