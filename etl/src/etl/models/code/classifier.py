"""
Main classifier training and inference module.

This module provides the main entry points for training and testing the page classifier
using PyTorch Lightning with hyperparameter optimization via Optuna.
"""

import logging
import os
import pprint
import time
import argparse
import sys
from typing import TypedDict, cast

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
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from optuna import Trial, create_study
from optuna.integration import PyTorchLightningPruningCallback
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

try:
    from .classifier_classes import PageClassifier, PageDataModule
    from .shared_constants import (
        CLASSIFIER_CKPT_PATH,
        CLASSIFIER_XGB_PATH,
        CLASSIFIER_LABEL_LIST,
    )
except ImportError:  # pragma: no cover - supports running as a script
    from classifier_classes import PageClassifier, PageDataModule  # pyright: ignore[reportMissingImports]
    from shared_constants import (  # pyright: ignore[reportMissingImports]
        CLASSIFIER_CKPT_PATH,
        CLASSIFIER_XGB_PATH,
        CLASSIFIER_LABEL_LIST,
    )

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
        val_split: float = 0.2,
        num_workers: int = 0,
        file_mode: str = "version",
    ):
        """
        Initialize the classifier trainer.

        Args:
            data_csv: Path to the data file
            num_trials: Number of Optuna trials for hyperparameter optimization
            max_epochs: Maximum training epochs per trial
            batch_size: Default batch size
            val_split: Fraction of data to use for validation
            num_workers: Number of data loading workers
            file_mode: Either 'version' or 'overwrite' - controls checkpoint file management
        """
        self.data_file: str = data_csv
        self.num_trials: int = num_trials
        self.max_epochs: int = max_epochs
        self.default_batch_size: int = batch_size
        self.val_split: float = val_split
        self.num_workers: int = num_workers
        self.file_mode: str = file_mode
        self.device: str = "mps"
        self.df: pd.DataFrame | None = None

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
        if not {"html", "text", "label"}.issubset(df.columns):
            raise ValueError("Data must contain 'html', 'text', and 'label' columns")
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
            num_workers=self.num_workers,
            xgb_path=CLASSIFIER_XGB_PATH,
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
        )
        return data_module, model

    @staticmethod
    def _label_names(data_module: PageDataModule) -> list[str]:
        return [
            label
            for label, _ in sorted(data_module.label2idx.items(), key=lambda kv: kv[1])
        ]

    def _print_val_diagnostics(
        self, model: PageClassifier, data_module: PageDataModule, *, tag: str
    ) -> None:
        if data_module.val_split <= 0:
            return

        label_names = self._label_names(data_module)
        num_classes = len(label_names)
        if num_classes <= 0:
            raise RuntimeError("No classes available for evaluation.")

        model = model.to(self.device)
        model.eval()

        y_true_parts: list[np.ndarray] = []
        y_pred_parts: list[np.ndarray] = []

        with torch.no_grad():
            for emissions, labels in data_module.val_dataloader():
                emissions = emissions.to(self.device)
                labels = labels.to(self.device)

                mask = labels != -100
                labels = torch.where(mask, labels, torch.zeros_like(labels))

                emissions = model(emissions)
                emissions = model._apply_learned_positional_prior(emissions, mask)
                emissions = model._apply_sig_position_bias(emissions, mask)
                predictions = model._viterbi_decode_ext(emissions, mask)

                y_true_parts.append(labels[mask].cpu().numpy())
                y_pred_parts.append(predictions[mask].cpu().numpy())

        y_true = np.concatenate(y_true_parts, axis=0) if y_true_parts else np.array([], dtype=int)
        y_pred = np.concatenate(y_pred_parts, axis=0) if y_pred_parts else np.array([], dtype=int)
        if y_true.size == 0:
            raise RuntimeError("Validation set is empty; cannot compute diagnostics.")

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=cast(str, cast(object, 0)),
        )
        print(f"{tag}  Acc: {accuracy:.4f}  P/R/F1: {precision:.4f}/{recall:.4f}/{f1:.4f}")

        cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
        print(f"{tag} Confusion Matrix:")
        print(cm)

        class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
        per_prec_arr, per_rec_arr, per_f1_arr, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=np.arange(num_classes),
            average=None,
            zero_division=cast(str, cast(object, 0)),
        )
        per_prec = np.asarray(per_prec_arr)
        per_rec = np.asarray(per_rec_arr)
        per_f1 = np.asarray(per_f1_arr)

        print(f"{tag} Per-class metrics:")
        for i, label in enumerate(label_names):
            print(
                f"  {label}: Acc={class_acc[i]:.4f} "
                f"P={per_prec[i]:.4f} R={per_rec[i]:.4f} F1={per_f1[i]:.4f}"
            )

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
            precision="bf16-mixed",
            logger=TensorBoardLogger("tb_logs", name="optuna"),
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

    def run(self) -> None:
        """Execute hyperparameter optimization and final training."""
        self._load_data()

        # Hyperparameter optimization
        study = create_study(direction="maximize")
        study.optimize(
            self._objective,
            n_trials=self.num_trials,
            gc_after_trial=True,
            callbacks=[
                lambda study, trial: study.stop() if study.best_value >= 1.0 else None
            ],
        )

        print(">> Hyperparameter optimization complete")
        print(f"   Best val_f1: {study.best_value:.4f}")
        print("   Best hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"     • {key}: {value}")

        # Final training with best hyperparameters
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
        data_module, model = self._build(best_params)

        checkpoint, early_stop, lr_monitor, progress_bar, _ = self._get_callbacks(
            ckpt=CLASSIFIER_CKPT_PATH, overwrite=(self.file_mode == "overwrite")
        )

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.device,
            devices=1,
            logger=TensorBoardLogger("tb_logs", name="final"),
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

        self._print_val_diagnostics(eval_model, data_module, tag="Model-Val (crf)")


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


def main(mode: str, file: str = "version") -> None:
    """
    Main entry point for classifier training and testing.

    Args:
        mode: Either 'train' or 'test'
        file: Either 'version' (default) or 'overwrite' - controls checkpoint file management
    """
    if mode == "train":
        classifier_trainer = ClassifierTrainer(
            data_csv="etl/src/etl/models/data/page-data.parquet",
            num_trials=40,
            max_epochs=25,
            num_workers=7,
            file_mode=file,
        )
        classifier_trainer.run()

    elif mode == "test":
        # Load test data
        df = pd.read_parquet("etl/src/etl/models/data/page-data-test.parquet")
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
        choices=["train", "test", "eval"],
        help="train: train model; test: run inference; eval: evaluate checkpoint on a deterministic split",
    )
    parser.add_argument(
        "--file",
        default="version",
        choices=["version", "overwrite"],
        help="Checkpoint file mode (train only).",
    )
    parser.add_argument(
        "--data-path",
        default="etl/src/etl/models/data/page-data.parquet",
        help="Parquet data path for eval (must contain labels).",
    )
    parser.add_argument(
        "--ckpt-path",
        default=CLASSIFIER_CKPT_PATH,
        help="Checkpoint path for eval.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split fraction for eval (split by agreement_uuid).",
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
        default="Model-Val (crf)",
        help="Prefix label for printed eval diagnostics.",
    )
    return parser


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main(mode="train", file="overwrite")
    else:
        args = _build_arg_parser().parse_args()
        if args.mode in ("train", "test"):
            main(mode=args.mode, file=args.file)
        else:
            df = pd.read_parquet(args.data_path)
            data_module = PageDataModule(
                df=df,
                batch_size=args.batch_size,
                val_split=args.val_split,
                num_workers=args.num_workers,
                xgb_path=CLASSIFIER_XGB_PATH,
            )
            data_module.setup(stage="validate")

            label_names = ClassifierTrainer._label_names(data_module)
            eval_model = PageClassifier.load_from_checkpoint(
                args.ckpt_path,
                label_names=label_names,
                strict=False,
            )
            evaluator = ClassifierTrainer(
                data_csv=args.data_path,
                num_trials=0,
                max_epochs=0,
                val_split=args.val_split,
                num_workers=args.num_workers,
            )
            evaluator.df = df
            evaluator._print_val_diagnostics(eval_model, data_module, tag=args.tag)
