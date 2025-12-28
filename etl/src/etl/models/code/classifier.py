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
from typing import TypedDict, cast
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
DATA_DIR = os.path.normpath(os.path.join(CODE_DIR, "../data"))
DEFAULT_SPLIT_PATH = os.path.join(DATA_DIR, "agreement-splits.json")

try:
    from .classifier_classes import PageClassifier, PageDataModule
    from .shared_constants import (
        CLASSIFIER_CKPT_PATH,
        CLASSIFIER_XGB_PATH,
        CLASSIFIER_LABEL_LIST,
    )
    from .split_utils import build_agreement_split, write_split_manifest
except ImportError:  # pragma: no cover - supports running as a script
    from classifier_classes import PageClassifier, PageDataModule  # pyright: ignore[reportMissingImports]
    from shared_constants import (  # pyright: ignore[reportMissingImports]
        CLASSIFIER_CKPT_PATH,
        CLASSIFIER_XGB_PATH,
        CLASSIFIER_LABEL_LIST,
    )
    from split_utils import build_agreement_split, write_split_manifest  # pyright: ignore[reportMissingImports]

# Reproducibility
_ = pl.seed_everything(42, workers=True, verbose=False)


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
    use_positional_prior: bool
    pos_prior_weight: float
    pos_prior_hidden: int
    enable_first_sig_postprocessing: bool
    first_sig_threshold: float


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
        self.num_workers: int = num_workers
        self.split_path: str | None = split_path
        self.file_mode: str = file_mode
        self.length_bucket_edges: list[float] | None = length_bucket_edges
        self.back_matter_bucket_edges: list[float] | None = back_matter_bucket_edges
        self.use_lstm: bool = bool(use_lstm)
        self.use_crf: bool = bool(use_crf)
        self.device: str = "mps"
        self.df: pd.DataFrame | None = None
        self.metrics_output_dir = os.path.dirname(self.data_file) or "."

        # Device selection
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    def _load_data(self) -> None:
        """Load the classification DataFrame from parquet file."""
        df = pd.read_parquet(self.data_file)
        required_cols = {"html", "text", "label", "date_announcement"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Data must contain columns: {sorted(required_cols)}. Missing: {sorted(missing)}")
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
    def _get_sweep_callbacks() -> tuple[EarlyStopping, LearningRateMonitor, TQDMProgressBar]:
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
            xgb_path=CLASSIFIER_XGB_PATH,
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
            xgb_path=CLASSIFIER_XGB_PATH,
            split_path=self.split_path,
            length_bucket_edges=self.length_bucket_edges,
            back_matter_bucket_edges=self.back_matter_bucket_edges,
        )
        data_module.setup()

        label_names = self._label_names(data_module)
        merged_overrides = dict(model_overrides)
        merged_overrides.setdefault("use_lstm", self.use_lstm)
        merged_overrides.setdefault("use_crf", self.use_crf)
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

    def _write_test_metrics(
        self,
        model: PageClassifier,
        data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        label_names: list[str],
    ) -> None:
        y_true, y_pred, y_pred_post = self._collect_eval_predictions(
            model,
            data_loader,
            label_names,
        )
        metrics = self._metrics_report(y_true, y_pred, y_pred_post, label_names)
        os.makedirs(self.metrics_output_dir, exist_ok=True)
        path = os.path.join(self.metrics_output_dir, "classifier_test_metrics.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(metrics, f, sort_keys=False)

    def _collect_eval_predictions(
        self,
        model: PageClassifier,
        data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        label_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not label_names:
            return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

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
                emissions = model._apply_learned_positional_prior(emissions, mask)
                emissions = model._apply_sig_position_bias(emissions, mask)
                if model.use_crf:
                    predictions = model._viterbi_decode_ext(emissions, mask)
                else:
                    predictions = emissions.argmax(dim=-1)

                if model.enable_first_sig_postprocessing:
                    class_probs = torch.softmax(emissions, dim=-1)
                    if model.use_crf and model.enforce_single_sig_block:
                        class_probs = class_probs.view(
                            class_probs.shape[0], class_probs.shape[1], 2, -1
                        ).sum(2)
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
                            seq_preds.append({"pred_class": class_name, "pred_probs": probs})
                        seq_preds_fixed, _ = model._apply_first_sig_block_postprocessing(
                            seq_preds, model.first_sig_threshold
                        )
                        for t, pred in enumerate(seq_preds_fixed):
                            pred_class = str(pred["pred_class"])
                            if pred_class not in label2idx:
                                raise ValueError(f"Unknown predicted label: {pred_class}")
                            predictions_post[b, t] = label2idx[pred_class]
                    y_pred_post_parts.append(predictions_post[mask].cpu().numpy())

                y_true_parts.append(labels[mask].cpu().numpy())
                y_pred_parts.append(predictions[mask].cpu().numpy())

        y_true = np.concatenate(y_true_parts, axis=0) if y_true_parts else np.array([], dtype=int)
        y_pred = np.concatenate(y_pred_parts, axis=0) if y_pred_parts else np.array([], dtype=int)
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
            logger=TensorBoardLogger("tb_logs", name="classifier/optuna"),
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

    def _objective_with_overrides(
        self, trial: Trial
    ) -> tuple[float, ModelOverrides]:
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
            logger=TensorBoardLogger("tb_logs", name="classifier/optuna"),
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
            "aux_back_start_loss_weight": float(trial_params["aux_back_start_loss_weight"]),
            "back_late_bonus": float(trial_params["back_late_bonus"]),
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

        best_params = self.optimize_hparams()

        print(">> Hyperparameter optimization complete")
        print("   Best hyperparameters:")
        for key, value in best_params.items():
            print(f"     • {key}: {value}")
        data_module, model = self._build(best_params)

        checkpoint, early_stop, lr_monitor, progress_bar, _ = self._get_callbacks(
            ckpt=CLASSIFIER_CKPT_PATH, overwrite=(self.file_mode == "overwrite")
        )

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.device,
            devices=1,
            precision=self._trainer_precision(),
            logger=TensorBoardLogger("tb_logs", name="classifier/final"),
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

    def classify(
        self, df: pd.DataFrame
    ) -> list[dict[str, object]]:
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
            num_workers=7,
            split_path=DEFAULT_SPLIT_PATH,
            file_mode=file,
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
            num_workers=7,
            split_path=DEFAULT_SPLIT_PATH,
            file_mode=file,
            use_lstm=use_lstm,
            use_crf=use_crf,
            length_bucket_edges=[0.0, 68.0, float("inf")],
            back_matter_bucket_edges=[0.0, 68.0, float("inf")],
        )
        classifier_trainer._load_data()

        if classifier_trainer.split_path and not os.path.exists(classifier_trainer.split_path):
            raise FileNotFoundError(
                f"Split manifest not found at {classifier_trainer.split_path}. "
                "Create it during training so sweep matches the trained split."
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
            xgb_path=CLASSIFIER_XGB_PATH,
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
                "Checkpoint label order does not match sweep label order. "
                f"Checkpoint: {ckpt_label_names} "
                f"Sweep: {label_names}"
            )
        sweep_csv_path = os.path.join(DATA_DIR, "crf_sweep_metrics.csv")
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
            tag = (
                "Model-Val (crf) sweep "
                f"sig_thr={sig_thr}"
            )
            print(
                ">> Sweep run: "
                f"sig_thr={sig_thr}"
            )
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
                per_prec_arr, per_rec_arr, per_f1_arr, _ = precision_recall_fscore_support(
                    y_true,
                    y_eval,
                    labels=np.arange(len(label_names)),
                    average=None,
                    zero_division=cast(str, cast(object, 0)),
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
                ">> Sweep best first_sig_threshold: "
                f"{best_sig_threshold} (post_overall_f1={best_post_f1:.4f})"
            )
        csv_file.close()

    elif mode == "test":
        # Load test data
        df = pd.read_parquet(os.path.join(DATA_DIR, "page-data-test.parquet"))
        print(df.shape)
        print(df['agreement_uuid'].nunique())
        # agreement = df["agreement_uuid"].unique().tolist()[-1]
        # df = df[df["agreement_uuid"] == agreement]

        # Initialize inference model
        inference_model = ClassifierInference(num_workers=7)

        # Run inference
        start = time.time()
        classified_result = inference_model.classify(df)
        inference_time = time.time() - start

        pprint.pprint(classified_result)
        print(f"Inference time: {inference_time:.2f} seconds")

    else:
        raise RuntimeError(f"Invalid mode: {mode}. Use 'train' or 'test'")

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/test/evaluate the CRF page classifier.")
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
        default=0,
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
                xgb_path=CLASSIFIER_XGB_PATH,
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
