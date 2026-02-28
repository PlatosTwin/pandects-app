"""
Main classifier training and inference module.

This module provides the main entry points for training and testing the page classifier
using PyTorch Lightning with hyperparameter optimization via Optuna. Agreement splits
are deterministic, year-stratified, and can be shared via a manifest.
"""

# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportAny=false, reportPrivateUsage=false, reportUnusedCallResult=false

import logging
import os
import pprint
import time
import argparse
import sys
import csv
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, TypedDict, cast
import yaml

# Disable Tokenizers parallelism before loading any HF/Lightning modules
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress PyTorch Lightning logging and warnings
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.ERROR)

import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from optuna import Trial, create_study
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.integration import PyTorchLightningPruningCallback
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

CODE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.normpath(os.path.join(CODE_DIR, "./data"))
EVAL_METRICS_DIR = os.path.normpath(os.path.join(CODE_DIR, "./eval_metrics"))
LOG_DIR = os.path.normpath(os.path.join(CODE_DIR, "./logs"))
DEFAULT_SPLIT_PATH = os.path.join(DATA_DIR, "agreement-splits.json")

if TYPE_CHECKING:
    from .classifier_classes import PageClassifier, PageDataModule
    from .page_classifier_constants import (
        CLASSIFIER_CKPT_PATH,
        CLASSIFIER_XGB_PATH,
        CLASSIFIER_XGB_TRAIN_PATH,
        CLASSIFIER_LABEL_LIST,
    )
    from .split_utils import (
        build_agreement_split,
        load_split_manifest,
        write_split_manifest,
    )
else:
    try:
        from .classifier_classes import PageClassifier, PageDataModule
        from .page_classifier_constants import (
            CLASSIFIER_CKPT_PATH,
            CLASSIFIER_XGB_PATH,
            CLASSIFIER_XGB_TRAIN_PATH,
            CLASSIFIER_LABEL_LIST,
        )
        from .split_utils import (
            build_agreement_split,
            load_split_manifest,
            write_split_manifest,
        )
    except ImportError:  # pragma: no cover - supports running as a script
        from classifier_classes import PageClassifier, PageDataModule
        from page_classifier_constants import (
            CLASSIFIER_CKPT_PATH,
            CLASSIFIER_XGB_PATH,
            CLASSIFIER_XGB_TRAIN_PATH,
            CLASSIFIER_LABEL_LIST,
        )
        from split_utils import (
            build_agreement_split,
            load_split_manifest,
            write_split_manifest,
        )

# Reproducibility
_ = pl.seed_everything(2718, workers=True, verbose=False)


def _metrics_dir_for_job(base_dir: str) -> str:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        job_id = "##local##"
        # raise RuntimeError("SLURM_JOB_ID is required to write eval metrics on HPC.")
        print("Running local and using dummy value for SLURM_JOB_ID.")
    return os.path.join(base_dir, job_id)


class HyperParams(TypedDict):
    lr: float
    weight_decay: float
    batch_size: int
    dropout: float
    hidden_dim: int
    lstm_dropout: float
    lstm_hidden: int
    lstm_num_layers: int


class ModelOverrides(TypedDict, total=False):
    sig_label: str
    back_label: str
    enforce_single_sig_block: bool
    use_lstm: bool
    use_crf: bool
    prefer_earliest_sig: bool
    sig_late_penalty: float
    sig_penalty_center: float
    sig_penalty_sharpness: float
    back_late_bonus: float
    back_bonus_center: float
    back_bonus_sharpness: float
    aux_sig_start_loss_weight: float
    aux_back_start_loss_weight: float
    aux_sig_presence_loss_weight: float
    back_requires_sig: bool
    use_positional_prior: bool
    pos_prior_weight: float
    pos_prior_hidden: int
    use_positional_features: bool
    pos_feature_dim: int
    enable_first_sig_postprocessing: bool
    first_sig_threshold: float
    use_structured_moe: bool
    tail_expert_loss_weight: float
    router_loss_weight: float
    router_sig_threshold: float
    router_back_threshold: float


class ClassifierTrainer:
    """
    Orchestrates hyperparameter optimization and training of PageClassifier.

    Uses Optuna for hyperparameter search and PyTorch Lightning for training.
    """

    def __init__(
        self,
        data_csv: str,
        num_trials: int,
        max_epochs: int,
        batch_size: int = 32,
        val_split: float = 0.1,
        test_split: float = 0.1,
        num_workers: int = 0,
        split_path: str | None = None,
        file_mode: str = "version",
        xgb_path: str = CLASSIFIER_XGB_TRAIN_PATH,
        prod_val_split: float = 0.2,
        length_bucket_edges: list[float] | None = None,
        back_matter_bucket_edges: list[float] | None = None,
        use_lstm: bool = True,
        use_crf: bool = True,
    ):
        """
        Initialize the classifier trainer.

        Args:
            data_csv: Path to the data file
            num_trials: Number of Optuna trials for hyperparameter optimization
            max_epochs: Maximum training epochs per trial
            batch_size: Default batch size
            val_split: Fraction of data to use for validation
            test_split: Fraction of data to use for test
            num_workers: Number of data loading workers
            split_path: Optional path to agreement split manifest
            file_mode: Either 'version' or 'overwrite' - controls checkpoint file management
            length_bucket_edges: Fixed edges for agreement length buckets
            back_matter_bucket_edges: Fixed edges for back matter page buckets
        """
        self.data_file: str = data_csv
        self.num_trials: int = num_trials
        self.max_epochs: int = max_epochs
        self.default_batch_size: int = batch_size
        self.val_split: float = val_split
        self.test_split: float = test_split
        self.num_workers: int = self._cap_num_workers(num_workers)
        self.split_path: str | None = split_path
        self.file_mode: str = file_mode
        self.xgb_path: str = xgb_path
        self.prod_val_split: float = prod_val_split
        self.length_bucket_edges: list[float] | None = length_bucket_edges
        self.back_matter_bucket_edges: list[float] | None = back_matter_bucket_edges
        self.use_lstm: bool = bool(use_lstm)
        self.use_crf: bool = bool(use_crf)
        self.device: str = "mps"
        self.df: pd.DataFrame | None = None
        self.metrics_output_dir = _metrics_dir_for_job(EVAL_METRICS_DIR)

        # Device selection
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    @staticmethod
    def _cap_num_workers(requested_workers: int) -> int:
        if requested_workers < 0:
            raise ValueError("num_workers must be >= 0.")
        available: int | None = None
        if hasattr(os, "sched_getaffinity"):
            try:
                available = len(os.sched_getaffinity(0))
            except OSError:
                available = None
        if available is None:
            cpu_count = os.cpu_count()
            available = cpu_count if cpu_count is not None else 1
        if available < 1:
            available = 1
        capped = min(requested_workers, available)
        if capped != requested_workers:
            print(
                f"[data] capping num_workers from {requested_workers} to {capped} (available cpus: {available})."
            )
        return capped

    @staticmethod
    def _normalize_bucket_edges(raw_edges: object) -> list[float] | None:
        if not isinstance(raw_edges, list):
            return None
        return [float(edge) for edge in raw_edges]

    def _resolve_bucket_edges(self) -> tuple[list[float] | None, list[float] | None]:
        length_edges = self.length_bucket_edges
        back_edges = self.back_matter_bucket_edges
        if length_edges is not None and back_edges is not None:
            return length_edges, back_edges

        if self.split_path and os.path.exists(self.split_path):
            split = load_split_manifest(self.split_path)
            split_meta = split.get("meta")
            if split_meta:
                if length_edges is None:
                    length_edges = self._normalize_bucket_edges(
                        split_meta.get("length_bucket_edges")
                    )
                if back_edges is None:
                    back_edges = self._normalize_bucket_edges(
                        split_meta.get("back_matter_bucket_edges")
                    )
        return length_edges, back_edges

    def _merge_model_overrides(
        self, model_overrides: ModelOverrides | None
    ) -> ModelOverrides:
        merged: ModelOverrides = {}
        if model_overrides:
            merged.update(model_overrides)
        if "use_lstm" not in merged:
            merged["use_lstm"] = self.use_lstm
        if "use_crf" not in merged:
            merged["use_crf"] = self.use_crf
        if "use_structured_moe" not in merged:
            merged["use_structured_moe"] = True
        return merged

    def _load_data(self) -> None:
        """Load the classification DataFrame from parquet file."""
        df = pd.read_parquet(self.data_file)
        required_cols = {"html", "text", "label", "date_announcement"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(
                f"Data must contain columns: {sorted(required_cols)}. Missing: {sorted(missing)}"
            )
        self.df = df
        print(f"[data] loaded {df.shape[0]} rows from {self.data_file}")

    def _get_callbacks(
        self,
        trial: Trial | None = None,
        ckpt: str | None = None,
        overwrite: bool = False,
    ) -> tuple[
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
        TQDMProgressBar,
        list[PyTorchLightningPruningCallback],
    ]:
        """
        Instantiate Lightning callbacks.

        Args:
            trial: Optuna trial for pruning callback
            ckpt: Checkpoint path for resuming training
            overwrite: If True, overwrite existing checkpoint instead of versioning

        Returns:
            Tuple of (checkpoint, early_stop, lr_monitor, progress_bar, prune_callback)
        """
        if ckpt:
            dirpath = os.path.dirname(ckpt)
            filename = os.path.splitext(os.path.basename(ckpt))[0]
        else:
            dirpath = None
            filename = "best-{epoch:02d}-{val_f1:.4f}"

        checkpoint = ModelCheckpoint(
            monitor="val_f1",
            mode="max",
            save_top_k=1,
            filename=filename,
            dirpath=dirpath,
            enable_version_counter=(
                not overwrite
            ),  # Disable versioning when overwrite=True
        )

        early_stop = EarlyStopping(monitor="val_f1", patience=3, mode="max")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        prune_callback = (
            [PyTorchLightningPruningCallback(trial, monitor="val_f1")]
            if trial is not None
            else []
        )
        progress_bar = TQDMProgressBar(refresh_rate=15)

        return checkpoint, early_stop, lr_monitor, progress_bar, prune_callback

    def _trainer_precision(self) -> _PRECISION_INPUT:
        if self.device == "cuda":
            return "bf16-mixed"
        if self.device == "mps":
            return "16-mixed"
        return "32-true"

    @staticmethod
    def _get_sweep_callbacks() -> (
        tuple[EarlyStopping, LearningRateMonitor, TQDMProgressBar]
    ):
        early_stop = EarlyStopping(monitor="val_f1", patience=3, mode="max")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        progress_bar = TQDMProgressBar(refresh_rate=15)
        return early_stop, lr_monitor, progress_bar

    def _build(self, params: HyperParams) -> tuple[PageDataModule, PageClassifier]:
        """
        Instantiate DataModule and Model with given hyperparameters.

        Args:
            params: Dictionary of hyperparameters

        Returns:
            Tuple of (data_module, model)
        """
        if self.df is None:
            raise RuntimeError("Training data not loaded. Call _load_data first.")
        data_module = PageDataModule(
            df=self.df,
            batch_size=params["batch_size"],
            val_split=self.val_split,
            test_split=self.test_split,
            num_workers=self.num_workers,
            xgb_path=self.xgb_path,
            split_path=self.split_path,
            length_bucket_edges=self.length_bucket_edges,
            back_matter_bucket_edges=self.back_matter_bucket_edges,
        )

        data_module.setup()  # Populate num_classes and other attributes

        model = PageClassifier(
            num_classes=data_module.num_classes,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            hidden_dim=params["hidden_dim"],
            dropout=params["dropout"],
            lstm_dropout=params["lstm_dropout"],
            lstm_hidden=params["lstm_hidden"],
            lstm_num_layers=params["lstm_num_layers"],
            use_lstm=self.use_lstm,
            use_crf=self.use_crf,
        )
        return data_module, model

    def _build_with_overrides(
        self,
        params: HyperParams,
        *,
        model_overrides: ModelOverrides,
    ) -> tuple[PageDataModule, PageClassifier]:
        if self.df is None:
            raise RuntimeError("Training data not loaded. Call _load_data first.")
        data_module = PageDataModule(
            df=self.df,
            batch_size=params["batch_size"],
            val_split=self.val_split,
            test_split=self.test_split,
            num_workers=self.num_workers,
            xgb_path=self.xgb_path,
            split_path=self.split_path,
            length_bucket_edges=self.length_bucket_edges,
            back_matter_bucket_edges=self.back_matter_bucket_edges,
        )
        data_module.setup()

        label_names = self._label_names(data_module)
        merged_overrides = self._merge_model_overrides(model_overrides)
        model = PageClassifier(
            num_classes=data_module.num_classes,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            hidden_dim=params["hidden_dim"],
            dropout=params["dropout"],
            lstm_dropout=params["lstm_dropout"],
            lstm_hidden=params["lstm_hidden"],
            lstm_num_layers=params["lstm_num_layers"],
            label_names=label_names,
            **merged_overrides,
        )
        return data_module, model

    @staticmethod
    def _label_names(data_module: PageDataModule) -> list[str]:
        return [
            label
            for label, _ in sorted(data_module.label2idx.items(), key=lambda kv: kv[1])
        ]

    def _train_production_model(
        self,
        params: HyperParams,
        *,
        model_overrides: ModelOverrides | None = None,
    ) -> str | None:
        if self.df is None:
            raise RuntimeError("Training data not loaded. Call _load_data first.")
        if not os.path.exists(CLASSIFIER_XGB_PATH):
            raise FileNotFoundError(
                f"Production XGB model not found at {CLASSIFIER_XGB_PATH}."
            )
        length_edges, back_edges = self._resolve_bucket_edges()
        prod_val_split = self.prod_val_split
        if prod_val_split > 0 and (length_edges is None or back_edges is None):
            print(
                "[model] prod_val_split requested but bucket edges unavailable; falling back to prod_val_split=0.0"
            )
            prod_val_split = 0.0

        if prod_val_split > 0:
            data_module = PageDataModule(
                df=self.df,
                batch_size=params["batch_size"],
                val_split=prod_val_split,
                test_split=0.0,
                num_workers=self.num_workers,
                xgb_path=CLASSIFIER_XGB_PATH,
                split_path=None,
                length_bucket_edges=length_edges,
                back_matter_bucket_edges=back_edges,
            )
            data_module.setup(stage="fit")
        else:
            data_module = PageDataModule(
                df=self.df,
                batch_size=params["batch_size"],
                val_split=0.0,
                test_split=0.0,
                num_workers=self.num_workers,
                xgb_path=CLASSIFIER_XGB_PATH,
                split_path=None,
                length_bucket_edges=length_edges,
                back_matter_bucket_edges=back_edges,
            )
            data_module.setup(stage="fit")
        merged_overrides = self._merge_model_overrides(model_overrides)
        model = PageClassifier(
            num_classes=data_module.num_classes,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            hidden_dim=params["hidden_dim"],
            dropout=params["dropout"],
            lstm_dropout=params["lstm_dropout"],
            lstm_hidden=params["lstm_hidden"],
            lstm_num_layers=params["lstm_num_layers"],
            label_names=self._label_names(data_module),
            **merged_overrides,
        )
        ckpt_dir = os.path.dirname(CLASSIFIER_CKPT_PATH)
        ckpt_name = os.path.splitext(os.path.basename(CLASSIFIER_CKPT_PATH))[0]
        monitor_metric = "val_f1" if prod_val_split > 0 else "train_f1_epoch"
        checkpoint = ModelCheckpoint(
            monitor=monitor_metric,
            mode="max",
            save_top_k=1,
            filename=ckpt_name,
            dirpath=ckpt_dir,
            enable_version_counter=(self.file_mode != "overwrite"),
        )
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.device,
            devices=1,
            precision=self._trainer_precision(),
            logger=TensorBoardLogger(LOG_DIR, name="classifier/production"),
            callbacks=[checkpoint, TQDMProgressBar(refresh_rate=15)],
            log_every_n_steps=10,
            deterministic=True,
            num_sanity_val_steps=0 if prod_val_split == 0 else 2,
            limit_val_batches=0 if prod_val_split == 0 else 1.0,
        )
        if prod_val_split > 0:
            trainer.fit(model, datamodule=data_module)
        else:
            trainer.fit(model, train_dataloaders=data_module.train_dataloader())
        best_ckpt_path = checkpoint.best_model_path
        if best_ckpt_path:
            print(f"[model] production checkpoint: {best_ckpt_path}")
        return best_ckpt_path or None

    def _print_eval_diagnostics(
        self,
        model: PageClassifier,
        data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        label_names: list[str],
        *,
        tag: str,
    ) -> None:
        y_true, y_pred, y_pred_post = self._collect_eval_predictions(
            model,
            data_loader,
            label_names,
        )

        num_classes = len(label_names)
        if y_true.size == 0:
            raise RuntimeError("Validation set is empty; cannot compute diagnostics.")

        def _print_metrics(
            *,
            y_pred_local: np.ndarray,
            metrics_tag: str,
        ) -> None:
            accuracy = accuracy_score(y_true, y_pred_local)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred_local,
                average="macro",
                zero_division=cast(str, cast(object, 0)),
            )
            print(
                f"{metrics_tag}  Acc: {accuracy:.4f}  P/R/F1: {precision:.4f}/{recall:.4f}/{f1:.4f}"
            )

            cm = confusion_matrix(y_true, y_pred_local, labels=np.arange(num_classes))
            print(f"{metrics_tag} Confusion Matrix:")
            print(cm)

            class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
            per_prec_arr, per_rec_arr, per_f1_arr, _ = precision_recall_fscore_support(
                y_true,
                y_pred_local,
                labels=np.arange(num_classes),
                average=None,
                zero_division=cast(str, cast(object, 0)),
            )
            per_prec = np.asarray(per_prec_arr)
            per_rec = np.asarray(per_rec_arr)
            per_f1 = np.asarray(per_f1_arr)

            print(f"{metrics_tag} Per-class metrics:")
            for i, label in enumerate(label_names):
                print(
                    f"  {label}: Acc={class_acc[i]:.4f} P={per_prec[i]:.4f} R={per_rec[i]:.4f} F1={per_f1[i]:.4f}"
                )

        _print_metrics(y_pred_local=y_pred, metrics_tag=f"{tag} (raw)")
        if y_pred_post.size:
            _print_metrics(y_pred_local=y_pred_post, metrics_tag=f"{tag} (post)")

        sequences = self._collect_eval_sequences(model, data_loader, label_names)
        boundary = self._sequence_boundary_metrics(sequences, label_names)
        onset_sig = self._sequence_onset_metrics(sequences, label_names, label="sig")
        onset_back = self._sequence_onset_metrics(
            sequences, label_names, label="back_matter"
        )
        print(f"{tag} Boundary F1 (macro): {boundary['boundary_f1_macro']:.4f}")
        sig_count = int(cast(int, onset_sig.get("count", 0)))
        if sig_count > 0:
            print(
                f"{tag} Sig onset MAE: {float(cast(float, onset_sig['mae'])):.2f} (n={sig_count})"
            )
        back_count = int(cast(int, onset_back.get("count", 0)))
        if back_count > 0:
            print(
                f"{tag} Back onset MAE: {float(cast(float, onset_back['mae'])):.2f} (n={back_count})"
            )
        sig_presence = self._sig_presence_metrics(sequences, label_names)
        if int(cast(int, sig_presence.get("count", 0))) > 0:
            print(
                f"{tag} Sig presence F1: {float(cast(float, sig_presence['f1'])):.4f}"
            )

    def _metrics_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_post: np.ndarray,
        label_names: list[str],
    ) -> dict[str, object]:
        num_classes = len(label_names)
        if y_true.size == 0:
            raise RuntimeError("Test set is empty; cannot compute metrics.")

        def _metrics_for(y_eval: np.ndarray) -> dict[str, object]:
            overall_acc = accuracy_score(y_true, y_eval)
            overall_p, overall_r, overall_f1, _ = precision_recall_fscore_support(
                y_true,
                y_eval,
                average="macro",
                zero_division=cast(str, cast(object, 0)),
            )
            cm = confusion_matrix(y_true, y_eval, labels=np.arange(num_classes))
            class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
            per_prec_arr, per_rec_arr, per_f1_arr, _ = precision_recall_fscore_support(
                y_true,
                y_eval,
                labels=np.arange(num_classes),
                average=None,
                zero_division=cast(str, cast(object, 0)),
            )
            per_prec_arr = np.asarray(per_prec_arr)
            per_rec_arr = np.asarray(per_rec_arr)
            per_f1_arr = np.asarray(per_f1_arr)
            per_class = {
                label: {
                    "accuracy": float(class_acc[i]),
                    "precision": float(per_prec_arr[i]),
                    "recall": float(per_rec_arr[i]),
                    "f1": float(per_f1_arr[i]),
                }
                for i, label in enumerate(label_names)
            }
            return {
                "overall": {
                    "accuracy": float(overall_acc),
                    "precision": float(overall_p),
                    "recall": float(overall_r),
                    "f1": float(overall_f1),
                },
                "per_class": per_class,
                "confusion_matrix": cm.tolist(),
            }

        metrics: dict[str, object] = {
            "raw": _metrics_for(y_pred),
        }
        if y_pred_post.size:
            metrics["post"] = _metrics_for(y_pred_post)
        return metrics

    @staticmethod
    def _sequence_boundary_metrics(
        sequences: list[tuple[np.ndarray, np.ndarray]],
        label_names: list[str],
    ) -> dict[str, object]:
        def _boundaries(seq: np.ndarray) -> list[tuple[int, int, int]]:
            boundaries: list[tuple[int, int, int]] = []
            for i in range(1, seq.size):
                if seq[i] != seq[i - 1]:
                    boundaries.append((i, int(seq[i - 1]), int(seq[i])))
            return boundaries

        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []

        for y_true, y_pred in sequences:
            true_bounds = set(_boundaries(y_true))
            pred_bounds = set(_boundaries(y_pred))
            if not true_bounds and not pred_bounds:
                precisions.append(1.0)
                recalls.append(1.0)
                f1s.append(1.0)
                continue
            if not pred_bounds:
                precisions.append(0.0)
                recalls.append(0.0)
                f1s.append(0.0)
                continue
            intersection = true_bounds & pred_bounds
            prec = len(intersection) / len(pred_bounds) if pred_bounds else 0.0
            rec = len(intersection) / len(true_bounds) if true_bounds else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        return {
            "boundary_precision_macro": (
                float(np.mean(precisions)) if precisions else 0.0
            ),
            "boundary_recall_macro": float(np.mean(recalls)) if recalls else 0.0,
            "boundary_f1_macro": float(np.mean(f1s)) if f1s else 0.0,
            "label_names": list(label_names),
        }

    @staticmethod
    def _sequence_onset_metrics(
        sequences: list[tuple[np.ndarray, np.ndarray]],
        label_names: list[str],
        *,
        label: str,
    ) -> dict[str, object]:
        if label not in label_names:
            return {"label": label, "count": 0}
        label_idx = label_names.index(label)
        errors: list[int] = []
        for y_true, y_pred in sequences:
            true_positions = np.where(y_true == label_idx)[0]
            pred_positions = np.where(y_pred == label_idx)[0]
            if true_positions.size == 0:
                continue
            true_first = int(true_positions[0])
            pred_first = (
                int(pred_positions[0]) if pred_positions.size else int(y_pred.size)
            )
            errors.append(abs(pred_first - true_first))
        if not errors:
            return {"label": label, "count": 0}
        return {
            "label": label,
            "count": len(errors),
            "mae": float(np.mean(errors)),
            "median_abs_error": float(np.median(errors)),
        }

    @staticmethod
    def _sig_presence_metrics(
        sequences: list[tuple[np.ndarray, np.ndarray]],
        label_names: list[str],
    ) -> dict[str, object]:
        if "sig" not in label_names:
            return {"label": "sig", "count": 0}
        sig_idx = label_names.index("sig")
        tp = fp = tn = fn = 0
        for y_true, y_pred in sequences:
            true_has = bool(np.any(y_true == sig_idx))
            pred_has = bool(np.any(y_pred == sig_idx))
            if true_has and pred_has:
                tp += 1
            elif true_has and not pred_has:
                fn += 1
            elif not true_has and pred_has:
                fp += 1
            else:
                tn += 1
        total = tp + tn + fp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / total if total > 0 else 0.0
        return {
            "label": "sig",
            "count": total,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    def _write_test_metrics(
        self,
        model: PageClassifier,
        data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        label_names: list[str],
        *,
        output_name: str = "classifier_test_metrics.yaml",
    ) -> None:
        y_true, y_pred, y_pred_post = self._collect_eval_predictions(
            model,
            data_loader,
            label_names,
        )
        metrics = self._metrics_report(y_true, y_pred, y_pred_post, label_names)
        sequences = self._collect_eval_sequences(model, data_loader, label_names)
        metrics["sequence"] = {
            "boundary": self._sequence_boundary_metrics(sequences, label_names),
            "onset_sig": self._sequence_onset_metrics(
                sequences, label_names, label="sig"
            ),
            "onset_back_matter": self._sequence_onset_metrics(
                sequences, label_names, label="back_matter"
            ),
            "sig_presence": self._sig_presence_metrics(sequences, label_names),
        }
        os.makedirs(self.metrics_output_dir, exist_ok=True)
        path = os.path.join(self.metrics_output_dir, output_name)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(metrics, f, sort_keys=False)

    def _collect_eval_sequences(
        self,
        model: PageClassifier,
        data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        label_names: list[str],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if not label_names:
            return []
        model = model.to(self.device)
        model.eval()
        sequences: list[tuple[np.ndarray, np.ndarray]] = []
        with torch.no_grad():
            for emissions, labels in data_loader:
                emissions = emissions.to(self.device)
                labels = labels.to(self.device)

                mask = labels != -100
                labels = torch.where(mask, labels, torch.zeros_like(labels))

                emissions = model(emissions, mask=mask)
                emissions = model._apply_decoding_biases(emissions, mask)
                predictions = model._decode_predictions(emissions, mask)

                for b in range(predictions.shape[0]):
                    seq_len = int(mask[b].sum().item())
                    if seq_len <= 0:
                        continue
                    y_true_seq = labels[b, :seq_len].cpu().numpy()
                    y_pred_seq = predictions[b, :seq_len].cpu().numpy()
                    sequences.append((y_true_seq, y_pred_seq))
        return sequences

    def _collect_eval_predictions(
        self,
        model: PageClassifier,
        data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        label_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not label_names:
            return (
                np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=int),
            )

        num_classes = len(label_names)
        if num_classes <= 0:
            raise RuntimeError("No classes available for evaluation.")

        model = model.to(self.device)
        model.eval()

        y_true_parts: list[np.ndarray] = []
        y_pred_parts: list[np.ndarray] = []
        y_pred_post_parts: list[np.ndarray] = []
        label2idx = {name: i for i, name in enumerate(label_names)}

        with torch.no_grad():
            for emissions, labels in data_loader:
                emissions = emissions.to(self.device)
                labels = labels.to(self.device)

                mask = labels != -100
                labels = torch.where(mask, labels, torch.zeros_like(labels))

                emissions = model(emissions, mask=mask)
                emissions = model._apply_decoding_biases(emissions, mask)
                predictions = model._decode_predictions(emissions, mask)

                if model.enable_first_sig_postprocessing:
                    class_probs = model._class_probs_from_emissions(emissions)
                    predictions_post = torch.zeros_like(predictions)
                    for b in range(predictions.shape[0]):
                        seq_len = int(mask[b].sum().item())
                        seq_preds = []
                        for t in range(seq_len):
                            pred_idx = int(predictions[b, t].item())
                            class_name = str(label_names[pred_idx])
                            probs = {
                                str(name): float(class_probs[b, t, i].item())
                                for i, name in enumerate(label_names)
                            }
                            seq_preds.append(
                                {"pred_class": class_name, "pred_probs": probs}
                            )
                        seq_preds_fixed, _ = (
                            model._apply_first_sig_block_postprocessing(
                                seq_preds, model.first_sig_threshold
                            )
                        )
                        for t, pred in enumerate(seq_preds_fixed):
                            pred_class = str(pred["pred_class"])
                            if pred_class not in label2idx:
                                raise ValueError(
                                    f"Unknown predicted label: {pred_class}"
                                )
                            predictions_post[b, t] = label2idx[pred_class]
                    y_pred_post_parts.append(predictions_post[mask].cpu().numpy())

                y_true_parts.append(labels[mask].cpu().numpy())
                y_pred_parts.append(predictions[mask].cpu().numpy())

        y_true = (
            np.concatenate(y_true_parts, axis=0)
            if y_true_parts
            else np.array([], dtype=int)
        )
        y_pred = (
            np.concatenate(y_pred_parts, axis=0)
            if y_pred_parts
            else np.array([], dtype=int)
        )
        if y_pred_post_parts:
            y_pred_post = np.concatenate(y_pred_post_parts, axis=0)
        else:
            y_pred_post = np.array([], dtype=int)
        return y_true, y_pred, y_pred_post

    def _objective(self, trial: Trial) -> float:
        """
        Optuna objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Validation F1 score
        """
        # Define hyperparameter search space
        params: HyperParams = {
            "lr": trial.suggest_float("lr", 1e-6, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "lstm_dropout": trial.suggest_float("lstm_dropout", 0.0, 0.5),
            "lstm_hidden": trial.suggest_categorical("lstm_hidden", [32, 64, 128]),
            "lstm_num_layers": trial.suggest_categorical("lstm_num_layers", [1, 2]),
        }

        data_module, model = self._build(params)
        checkpoint, early_stop, lr_monitor, progress_bar, prune_callback = (
            self._get_callbacks(trial)
        )

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.device,
            devices=1,
            precision=self._trainer_precision(),
            logger=TensorBoardLogger(LOG_DIR, name="classifier/optuna"),
            callbacks=[
                checkpoint,
                early_stop,
                lr_monitor,
                progress_bar,
                *prune_callback,
            ],
            log_every_n_steps=10,
            deterministic=True,
        )
        trainer.fit(model, datamodule=data_module)

        val_f1 = float(trainer.callback_metrics["val_f1"])

        # Clean up to avoid memory leaks
        del (
            trainer,
            model,
            data_module,
            checkpoint,
            early_stop,
            lr_monitor,
            prune_callback,
        )

        return val_f1

    def _objective_with_overrides(self, trial: Trial) -> tuple[float, ModelOverrides]:
        """
        Optuna objective for hyperparameter optimization including model overrides.

        Returns:
            Tuple of (validation F1 score, model overrides)
        """
        params: HyperParams = {
            "lr": trial.suggest_float("lr", 1e-6, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "lstm_dropout": trial.suggest_float("lstm_dropout", 0.0, 0.5),
            "lstm_hidden": trial.suggest_categorical("lstm_hidden", [32, 64, 128]),
            "lstm_num_layers": trial.suggest_categorical("lstm_num_layers", [1, 2]),
        }

        model_overrides: ModelOverrides = {
            "aux_back_start_loss_weight": trial.suggest_categorical(
                "aux_back_start_loss_weight", [0.08, 0.1, 0.12]
            ),
            "back_late_bonus": trial.suggest_categorical(
                "back_late_bonus", [0.4, 0.5, 0.6]
            ),
            "tail_expert_loss_weight": trial.suggest_categorical(
                "tail_expert_loss_weight", [0.2, 0.4, 0.6]
            ),
            "router_loss_weight": trial.suggest_categorical(
                "router_loss_weight", [0.1, 0.2, 0.3]
            ),
            "router_sig_threshold": trial.suggest_categorical(
                "router_sig_threshold", [0.45, 0.55, 0.65]
            ),
            "router_back_threshold": trial.suggest_categorical(
                "router_back_threshold", [0.45, 0.55, 0.65]
            ),
        }

        data_module, model = self._build_with_overrides(
            params, model_overrides=model_overrides
        )
        checkpoint, early_stop, lr_monitor, progress_bar, prune_callback = (
            self._get_callbacks(trial)
        )

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.device,
            devices=1,
            precision=self._trainer_precision(),
            logger=TensorBoardLogger(LOG_DIR, name="classifier/optuna"),
            callbacks=[
                checkpoint,
                early_stop,
                lr_monitor,
                progress_bar,
                *prune_callback,
            ],
            log_every_n_steps=10,
            deterministic=True,
        )
        trainer.fit(model, datamodule=data_module)

        val_f1 = float(trainer.callback_metrics["val_f1"])

        # Clean up to avoid memory leaks
        del (
            trainer,
            model,
            data_module,
            checkpoint,
            early_stop,
            lr_monitor,
            prune_callback,
        )

        return val_f1, model_overrides

    def optimize_hparams(self) -> HyperParams:
        if self.num_trials <= 0:
            raise ValueError("num_trials must be > 0 for hyperparameter optimization.")
        no_improve_limit = 5
        best_value: float | None = None
        no_improve_count = 0

        def _stop_on_no_improve(study: Study, _trial: FrozenTrial) -> None:
            nonlocal best_value, no_improve_count
            current_best = study.best_value
            if best_value is None or current_best > best_value:
                best_value = current_best
                no_improve_count = 0
            else:
                no_improve_count += 1
            if no_improve_count >= no_improve_limit:
                study.stop()

        study = create_study(direction="maximize")
        study.optimize(
            self._objective,
            n_trials=self.num_trials,
            gc_after_trial=True,
            callbacks=[
                lambda study, trial: study.stop() if study.best_value >= 1.0 else None,
                _stop_on_no_improve,
            ],
        )
        trial_params = study.best_trial.params
        return {
            "lr": float(trial_params["lr"]),
            "weight_decay": float(trial_params["weight_decay"]),
            "batch_size": int(trial_params["batch_size"]),
            "dropout": float(trial_params["dropout"]),
            "hidden_dim": int(trial_params["hidden_dim"]),
            "lstm_dropout": float(trial_params["lstm_dropout"]),
            "lstm_hidden": int(trial_params["lstm_hidden"]),
            "lstm_num_layers": int(trial_params["lstm_num_layers"]),
        }

    def optimize_hparams_with_overrides(self) -> tuple[HyperParams, ModelOverrides]:
        if self.num_trials <= 0:
            raise ValueError("num_trials must be > 0 for hyperparameter optimization.")
        no_improve_limit = 5
        best_value: float | None = None
        no_improve_count = 0

        def _stop_on_no_improve(study: Study, _trial: FrozenTrial) -> None:
            nonlocal best_value, no_improve_count
            current_best = study.best_value
            if best_value is None or current_best > best_value:
                best_value = current_best
                no_improve_count = 0
            else:
                no_improve_count += 1
            if no_improve_count >= no_improve_limit:
                study.stop()

        def _objective(trial: Trial) -> float:
            score, _ = self._objective_with_overrides(trial)
            return score

        study = create_study(direction="maximize")
        study.optimize(
            _objective,
            n_trials=self.num_trials,
            gc_after_trial=True,
            callbacks=[
                lambda study, trial: study.stop() if study.best_value >= 1.0 else None,
                _stop_on_no_improve,
            ],
        )
        trial_params = study.best_trial.params
        best_params: HyperParams = {
            "lr": float(trial_params["lr"]),
            "weight_decay": float(trial_params["weight_decay"]),
            "batch_size": int(trial_params["batch_size"]),
            "dropout": float(trial_params["dropout"]),
            "hidden_dim": int(trial_params["hidden_dim"]),
            "lstm_dropout": float(trial_params["lstm_dropout"]),
            "lstm_hidden": int(trial_params["lstm_hidden"]),
            "lstm_num_layers": int(trial_params["lstm_num_layers"]),
        }
        best_overrides: ModelOverrides = {
            "aux_back_start_loss_weight": float(
                trial_params["aux_back_start_loss_weight"]
            ),
            "back_late_bonus": float(trial_params["back_late_bonus"]),
            "tail_expert_loss_weight": float(trial_params["tail_expert_loss_weight"]),
            "router_loss_weight": float(trial_params["router_loss_weight"]),
            "router_sig_threshold": float(trial_params["router_sig_threshold"]),
            "router_back_threshold": float(trial_params["router_back_threshold"]),
        }
        return best_params, best_overrides

    def run(self) -> None:
        """Execute hyperparameter optimization and final training."""
        self._load_data()
        if self.split_path and not os.path.exists(self.split_path):
            if self.df is None:
                raise RuntimeError("Training data not loaded. Call _load_data first.")
            split = build_agreement_split(
                self.df,
                val_split=self.val_split,
                test_split=self.test_split,
                length_bucket_edges=self.length_bucket_edges,
                back_matter_bucket_edges=self.back_matter_bucket_edges,
            )
            write_split_manifest(self.split_path, split)
            print(f"[split] wrote agreement split manifest to {self.split_path}")

        best_params, best_overrides = self.optimize_hparams_with_overrides()

        print(">> Hyperparameter optimization complete")
        print("   Best hyperparameters:")
        for key, value in best_params.items():
            print(f"     • {key}: {value}")
        print("   Best model overrides:")
        for key, value in best_overrides.items():
            print(f"     • {key}: {value}")
        data_module, model = self._build_with_overrides(
            best_params, model_overrides=best_overrides
        )

        checkpoint, early_stop, lr_monitor, progress_bar, _ = self._get_callbacks(
            ckpt=CLASSIFIER_CKPT_PATH, overwrite=(self.file_mode == "overwrite")
        )

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.device,
            devices=1,
            precision=self._trainer_precision(),
            logger=TensorBoardLogger(LOG_DIR, name="classifier/final"),
            callbacks=[checkpoint, early_stop, lr_monitor, progress_bar],
            log_every_n_steps=10,
        )
        trainer.fit(model, datamodule=data_module)

        best_ckpt_path = checkpoint.best_model_path
        if best_ckpt_path:
            print(f"[model] best checkpoint: {best_ckpt_path}")
            label_names = self._label_names(data_module)
            eval_model = PageClassifier.load_from_checkpoint(
                best_ckpt_path,
                label_names=label_names,
                strict=False,
            )
        else:
            eval_model = model

        label_names = self._label_names(data_module)
        self._print_eval_diagnostics(
            eval_model,
            data_module.test_dataloader(),
            label_names,
            tag="Model-Test (crf)",
        )
        self._write_test_metrics(
            eval_model,
            data_module.test_dataloader(),
            label_names,
        )
        if self.xgb_path == CLASSIFIER_XGB_TRAIN_PATH:
            print("[model] training production LSTM checkpoint using production XGB")
            prod_ckpt = self._train_production_model(
                best_params, model_overrides=best_overrides
            )
            if prod_ckpt:
                if not self.split_path or not os.path.exists(self.split_path):
                    raise FileNotFoundError(
                        "Split manifest required for production evaluation."
                    )
                if self.df is None:
                    raise RuntimeError(
                        "Training data not loaded. Call _load_data first."
                    )
                df_local = self.df
                prod_data_module = PageDataModule(
                    df=df_local,
                    batch_size=best_params["batch_size"],
                    val_split=self.val_split,
                    test_split=self.test_split,
                    num_workers=self.num_workers,
                    xgb_path=CLASSIFIER_XGB_PATH,
                    split_path=self.split_path,
                    length_bucket_edges=self.length_bucket_edges,
                    back_matter_bucket_edges=self.back_matter_bucket_edges,
                )
                prod_data_module.setup(stage="test")
                prod_label_names = self._label_names(prod_data_module)
                prod_model = PageClassifier.load_from_checkpoint(
                    prod_ckpt,
                    label_names=prod_label_names,
                    strict=False,
                )
                self._print_eval_diagnostics(
                    prod_model,
                    prod_data_module.test_dataloader(),
                    prod_label_names,
                    tag="Model-Test (crf) production",
                )
                self._write_test_metrics(
                    prod_model,
                    prod_data_module.test_dataloader(),
                    prod_label_names,
                    output_name="classifier_test_metrics_production.yaml",
                )


class ClassifierInference:
    """
    Wrapper for trained PageClassifier for easy batch inference.

    Usage:
        inf = ClassifierInference(
            ckpt_path="path/to/best.ckpt",
            batch_size=32,
            num_workers=4
        )
        df = pd.read_csv("new_pages.csv")  # must have 'html', 'text', and optional 'order'
        out = inf.classify(df)
        print(out[['pred_label','pred_proba']])
    """

    def __init__(
        self,
        ckpt_path: str = CLASSIFIER_CKPT_PATH,
        xgb_path: str = CLASSIFIER_XGB_PATH,
        batch_size: int = 32,
        num_workers: int = 0,
        enable_first_sig_postprocessing: bool = True,
        first_sig_threshold: float = 0.3,
    ):
        """
        Initialize the inference wrapper.

        Args:
            ckpt_path: Path to trained model checkpoint
            xgb_path: Path to XGBoost model
            batch_size: Batch size for inference
            num_workers: Number of data loading workers
            enable_first_sig_postprocessing: Whether to apply decode-time first signature block fix
            first_sig_threshold: Probability threshold for signature block detection in postprocessing
        """
        # Device selection
        self.device: str = "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Load model
        self.model: PageClassifier
        self.model = PageClassifier.load_from_checkpoint(
            ckpt_path,
            label_names=CLASSIFIER_LABEL_LIST,
            enable_first_sig_postprocessing=enable_first_sig_postprocessing,
            first_sig_threshold=first_sig_threshold,
            strict=False,
        )
        _ = self.model.to(self.device)
        _ = self.model.eval()

        self.xgb_path: str = xgb_path
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

        self.trainer: pl.Trainer
        self.trainer = pl.Trainer(
            accelerator=self.device,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
        )

    def classify(self, df: pd.DataFrame) -> list[dict[str, object]]:
        """
        Run inference on a DataFrame of pages.

        Args:
            df: DataFrame with 'html', 'text', and optional 'order' columns

        Returns:
            List of prediction dictionaries with probabilities and predicted class
        """
        # Set up DataModule
        data_module = PageDataModule(
            df=df,
            batch_size=self.batch_size,
            val_split=0.0,  # No validation split for inference
            test_split=0.0,
            num_workers=self.num_workers,
            xgb_path=self.xgb_path,
        )
        data_module.setup(stage="predict")

        # Run predictions with minimal logging
        outputs = self.trainer.predict(
            self.model, dataloaders=data_module.predict_dataloader()
        )
        if outputs is None:
            raise RuntimeError("Prediction returned no outputs.")

        flattened: list[dict[str, object]] = []
        for batch_out in outputs:
            if isinstance(batch_out, list):
                flattened.extend(batch_out)
            else:
                flattened.append(batch_out)

        return flattened


def main(
    mode: str,
    file: str = "version",
    *,
    use_lstm: bool = True,
    use_crf: bool = True,
) -> None:
    """
    Main entry point for classifier training and testing.

    Args:
        mode: Either 'train' or 'test'
        file: Either 'version' (default) or 'overwrite' - controls checkpoint file management
    """
    if mode == "train":
        classifier_trainer = ClassifierTrainer(
            data_csv=os.path.join(DATA_DIR, "page-data.parquet"),
            num_trials=40,
            max_epochs=25,
            num_workers=3,
            split_path=DEFAULT_SPLIT_PATH,
            file_mode=file,
            xgb_path=CLASSIFIER_XGB_TRAIN_PATH,
            use_lstm=use_lstm,
            use_crf=use_crf,
            # length_bucket_edges=[0.0, 68.0, float("inf")],
            # back_matter_bucket_edges=[0.0, 68.0, float("inf")],
        )
        classifier_trainer.run()

    elif mode == "sweep":
        print(">> Running sweep mode")
        classifier_trainer = ClassifierTrainer(
            data_csv=os.path.join(DATA_DIR, "page-data.parquet"),
            num_trials=20,
            max_epochs=25,
            num_workers=3,
            split_path=DEFAULT_SPLIT_PATH,
            file_mode=file,
            xgb_path=CLASSIFIER_XGB_TRAIN_PATH,
            use_lstm=use_lstm,
            use_crf=use_crf,
            length_bucket_edges=[0.0, 68.0, float("inf")],
            back_matter_bucket_edges=[0.0, 68.0, float("inf")],
        )
        classifier_trainer._load_data()

        if classifier_trainer.split_path and not os.path.exists(
            classifier_trainer.split_path
        ):
            raise FileNotFoundError(
                f"Split manifest not found at {classifier_trainer.split_path}. Create it during training so sweep matches the trained split."
            )

        sig_thresholds = [0.3, 0.4, 0.5]

        if not os.path.exists(CLASSIFIER_CKPT_PATH):
            raise FileNotFoundError(
                f"Checkpoint not found at {CLASSIFIER_CKPT_PATH}. Train the best model first."
            )

        if classifier_trainer.df is None:
            raise RuntimeError("Training data not loaded. Call _load_data first.")

        data_module = PageDataModule(
            df=classifier_trainer.df,
            batch_size=8,
            val_split=classifier_trainer.val_split,
            test_split=classifier_trainer.test_split,
            num_workers=classifier_trainer.num_workers,
            xgb_path=CLASSIFIER_XGB_TRAIN_PATH,
            split_path=classifier_trainer.split_path,
            length_bucket_edges=classifier_trainer.length_bucket_edges,
            back_matter_bucket_edges=classifier_trainer.back_matter_bucket_edges,
        )
        data_module.setup(stage="validate")
        label_names = classifier_trainer._label_names(data_module)
        eval_model = PageClassifier.load_from_checkpoint(
            CLASSIFIER_CKPT_PATH,
            strict=False,
        )
        eval_model.use_lstm = use_lstm
        eval_model.use_crf = use_crf
        ckpt_label_names = list(getattr(eval_model, "label_names", []))
        if ckpt_label_names and ckpt_label_names != label_names:
            raise ValueError(
                f"Checkpoint label order does not match sweep label order. Checkpoint: {ckpt_label_names} Sweep: {label_names}"
            )
        sweep_csv_path = os.path.join(EVAL_METRICS_DIR, "crf_sweep_metrics.csv")
        os.makedirs(os.path.dirname(sweep_csv_path), exist_ok=True)
        csv_file = open(sweep_csv_path, "w", newline="")
        csv_file.write(f"# checkpoint={CLASSIFIER_CKPT_PATH}\n")
        csv_file.write(f"# split_path={classifier_trainer.split_path}\n")
        csv_file.write(f"# data_path={classifier_trainer.data_file}\n")
        csv_file.write(
            f"# val_split={classifier_trainer.val_split} test_split={classifier_trainer.test_split}\n"
        )
        csv_writer: csv.DictWriter[object] | None = None
        best_sig_threshold: float | None = None
        best_post_f1: float | None = None

        for sig_thr in sig_thresholds:
            tag = f"Model-Val (crf) sweep sig_thr={sig_thr}"
            print(f">> Sweep run: sig_thr={sig_thr}")
            eval_model.enable_first_sig_postprocessing = True
            eval_model.first_sig_threshold = sig_thr
            classifier_trainer._print_eval_diagnostics(
                eval_model,
                data_module.val_dataloader(),
                label_names,
                tag=tag,
            )
            y_true, y_pred, y_post = classifier_trainer._collect_eval_predictions(
                eval_model,
                data_module.val_dataloader(),
                label_names,
            )
            y_eval_post = y_post if y_post.size else y_pred
            y_eval_raw = y_pred

            def _metrics_for(
                y_eval: np.ndarray,
            ) -> tuple[
                float,
                float,
                float,
                float,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ]:
                overall_acc = accuracy_score(y_true, y_eval)
                overall_p, overall_r, overall_f1, _ = precision_recall_fscore_support(
                    y_true,
                    y_eval,
                    average="macro",
                    zero_division=cast(str, cast(object, 0)),
                )
                cm = confusion_matrix(
                    y_true, y_eval, labels=np.arange(len(label_names))
                )
                class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
                per_prec_arr, per_rec_arr, per_f1_arr, _ = (
                    precision_recall_fscore_support(
                        y_true,
                        y_eval,
                        labels=np.arange(len(label_names)),
                        average=None,
                        zero_division=cast(str, cast(object, 0)),
                    )
                )
                per_prec = np.asarray(per_prec_arr)
                per_rec = np.asarray(per_rec_arr)
                per_f1 = np.asarray(per_f1_arr)
                return (
                    float(overall_acc),
                    float(overall_p),
                    float(overall_r),
                    float(overall_f1),
                    class_acc,
                    per_prec,
                    per_rec,
                    per_f1,
                )

            (
                raw_acc,
                raw_p,
                raw_r,
                raw_f1,
                raw_class_acc,
                raw_prec,
                raw_rec,
                raw_f1_arr,
            ) = _metrics_for(y_eval_raw)
            (
                post_acc,
                post_p,
                post_r,
                post_f1,
                post_class_acc,
                post_prec,
                post_rec,
                post_f1_arr,
            ) = _metrics_for(y_eval_post)
            row: dict[str, object] = {
                "run": f"sig_thr={sig_thr}",
                "raw_overall_acc": raw_acc,
                "raw_overall_p": raw_p,
                "raw_overall_r": raw_r,
                "raw_overall_f1": raw_f1,
                "post_overall_acc": post_acc,
                "post_overall_p": post_p,
                "post_overall_r": post_r,
                "post_overall_f1": post_f1,
            }
            if best_post_f1 is None or post_f1 > best_post_f1:
                best_post_f1 = post_f1
                best_sig_threshold = sig_thr
            for i, label in enumerate(label_names):
                row[f"raw_{label}_acc"] = float(raw_class_acc[i])
                row[f"raw_{label}_p"] = float(raw_prec[i])
                row[f"raw_{label}_r"] = float(raw_rec[i])
                row[f"raw_{label}_f1"] = float(raw_f1_arr[i])
                row[f"post_{label}_acc"] = float(post_class_acc[i])
                row[f"post_{label}_p"] = float(post_prec[i])
                row[f"post_{label}_r"] = float(post_rec[i])
                row[f"post_{label}_f1"] = float(post_f1_arr[i])
            if csv_writer is None:
                csv_writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                csv_writer.writeheader()
            csv_writer.writerow(cast(Mapping[object, object], row))
            csv_file.flush()

        if best_sig_threshold is not None and best_post_f1 is not None:
            print(
                f">> Sweep best first_sig_threshold: {best_sig_threshold} (post_overall_f1={best_post_f1:.4f})"
            )
        csv_file.close()

    elif mode == "test":
        # Load test data
        df = pd.read_parquet(os.path.join(DATA_DIR, "page-data-test.parquet"))
        print(df.shape)
        print(df["agreement_uuid"].nunique())
        # agreement = df["agreement_uuid"].unique().tolist()[-1]
        # df = df[df["agreement_uuid"] == agreement]

        # Initialize inference model
        inference_model = ClassifierInference(num_workers=3)

        # Run inference
        start = time.time()
        classified_result = inference_model.classify(df)
        inference_time = time.time() - start

        pprint.pprint(classified_result)
        print(f"Inference time: {inference_time:.2f} seconds")

    else:
        raise RuntimeError(f"Invalid mode: {mode}. Use 'train' or 'test'")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/test/evaluate the CRF page classifier."
    )
    parser.add_argument(
        "mode",
        choices=["train", "test", "eval", "sweep"],
        help="train: train model; test: run inference; eval: evaluate checkpoint on a deterministic split; sweep: sweep first_sig_threshold on validation",
    )
    parser.add_argument(
        "--file",
        default="version",
        choices=["version", "overwrite"],
        help="Checkpoint file mode (train only).",
    )
    parser.add_argument(
        "--data-path",
        default=os.path.join(DATA_DIR, "page-data.parquet"),
        help="Parquet data path for eval (must contain labels).",
    )
    parser.add_argument(
        "--ckpt-path",
        default=CLASSIFIER_CKPT_PATH,
        help="Checkpoint path for eval.",
    )
    parser.add_argument(
        "--split-path",
        default=DEFAULT_SPLIT_PATH,
        help="Agreement split manifest path.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split fraction for eval (split by agreement_uuid).",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Test split fraction for eval (split by agreement_uuid).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for eval dataloader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=3,
        help="Dataloader workers for eval.",
    )
    parser.add_argument(
        "--tag",
        default="Model-Test (crf)",
        help="Prefix label for printed eval diagnostics.",
    )
    parser.add_argument(
        "--no-lstm",
        dest="use_lstm",
        action="store_false",
        help="Disable the LSTM residual branch.",
    )
    parser.add_argument(
        "--no-crf",
        dest="use_crf",
        action="store_false",
        help="Disable CRF decoding and loss.",
    )
    parser.set_defaults(use_lstm=True, use_crf=True)
    return parser


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main(mode="train", file="overwrite")
    else:
        args = _build_arg_parser().parse_args()
        if args.mode in ("train", "test", "sweep"):
            main(
                mode=args.mode,
                file=args.file,
                use_lstm=args.use_lstm,
                use_crf=args.use_crf,
            )
        else:
            df = pd.read_parquet(args.data_path)
            data_module = PageDataModule(
                df=df,
                batch_size=args.batch_size,
                val_split=args.val_split,
                test_split=args.test_split,
                num_workers=args.num_workers,
                xgb_path=CLASSIFIER_XGB_TRAIN_PATH,
                split_path=args.split_path,
                length_bucket_edges=[0.0, 68.0, float("inf")],
                back_matter_bucket_edges=[0.0, 68.0, float("inf")],
            )
            data_module.setup(stage="test")

            label_names = ClassifierTrainer._label_names(data_module)
            eval_model = PageClassifier.load_from_checkpoint(
                args.ckpt_path,
                label_names=label_names,
                strict=False,
            )
            eval_model.use_lstm = args.use_lstm
            eval_model.use_crf = args.use_crf
            evaluator = ClassifierTrainer(
                data_csv=args.data_path,
                num_trials=0,
                max_epochs=0,
                val_split=args.val_split,
                test_split=args.test_split,
                num_workers=args.num_workers,
                split_path=args.split_path,
                length_bucket_edges=[0.0, 68.0, float("inf")],
                back_matter_bucket_edges=[0.0, 68.0, float("inf")],
                xgb_path=CLASSIFIER_XGB_TRAIN_PATH,
                use_lstm=args.use_lstm,
                use_crf=args.use_crf,
            )
            evaluator.df = df
            evaluator._print_eval_diagnostics(
                eval_model,
                data_module.test_dataloader(),
                label_names,
                tag=args.tag,
            )
