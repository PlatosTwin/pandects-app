"""
Main classifier training and inference module.

This module provides the main entry points for training and testing the page classifier
using PyTorch Lightning with hyperparameter optimization via Optuna.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union
import pprint
import logging

# Disable Tokenizers parallelism before loading any HF/Lightning modules
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress PyTorch Lightning logging and warnings
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.ERROR)

import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from optuna import create_study
from optuna.integration import PyTorchLightningPruningCallback

from classifier_classes import PageClassifier, PageDataModule, load_xgb_model
from shared_constants import (
    CLASSIFIER_CKPT_PATH,
    CLASSIFIER_XGB_PATH,
    CLASSIFIER_LABEL_LIST,
)

# Reproducibility
pl.seed_everything(42, workers=True, verbose=False)


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
        self.data_file = data_csv
        self.num_trials = num_trials
        self.max_epochs = max_epochs
        self.default_batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.file_mode = file_mode

        # Device selection
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    def _load_data(self) -> None:
        """Load the classification DataFrame from parquet file."""
        self.df = pd.read_parquet(self.data_file)
        if not {"html", "text", "label"}.issubset(self.df.columns):
            raise ValueError("Data must contain 'html', 'text', and 'label' columns")
        print(f"[data] loaded {self.df.shape[0]} rows from {self.data_file}")

    def _get_callbacks(
        self, trial: Optional[object] = None, ckpt: Optional[str] = None, overwrite: bool = False
    ) -> tuple:
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
            enable_version_counter=(not overwrite),  # Disable versioning when overwrite=True
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

    def _build(
        self, params: Dict[str, Union[float, int]]
    ) -> Tuple[PageDataModule, PageClassifier]:
        """
        Instantiate DataModule and Model with given hyperparameters.

        Args:
            params: Dictionary of hyperparameters

        Returns:
            Tuple of (data_module, model)
        """
        data_module = PageDataModule(
            df=self.df,
            batch_size=params.get("batch_size", self.default_batch_size),
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

    def _objective(self, trial: object) -> float:
        """
        Optuna objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Validation F1 score
        """
        # Define hyperparameter search space
        params = {
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

        val_f1 = trainer.callback_metrics["val_f1"].item()

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
        best_params = study.best_trial.params
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
        device: Optional[str] = None,
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
            device: Device to use for inference
            enable_first_sig_postprocessing: Whether to apply decode-time first signature block fix
            first_sig_threshold: Probability threshold for signature block detection in postprocessing
        """
        # Device selection
        if device is not None:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Load model
        self.model = PageClassifier.load_from_checkpoint(
            ckpt_path, 
            label_names=CLASSIFIER_LABEL_LIST, 
            enable_first_sig_postprocessing=enable_first_sig_postprocessing,
            first_sig_threshold=first_sig_threshold,
            strict=False
        )
        self.model.to(self.device)
        self.model.eval()

        self.xgb_path = xgb_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def classify(
        self, df: pd.DataFrame
    ) -> List[List[Dict[str, Union[str, float, bool]]]]:
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
        trainer = pl.Trainer(
            logger=False,  # Disable logging
            enable_progress_bar=False,  # Disable progress bars
            enable_model_summary=False,  # Disable model summary
            enable_checkpointing=False,  # Disable checkpointing for inference
        )
        outputs = trainer.predict(
            self.model, dataloaders=data_module.predict_dataloader()
        )

        flattened: List[List[Dict[str, Union[str, float, bool]]]] = []
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
            num_trials=30,
            max_epochs=25,
            num_workers=7,
            file_mode=file,
        )
        classifier_trainer.run()

    elif mode == "test":
        # Load test data
        df = pd.read_parquet("etl/src/etl/models/data/page-data-test.parquet")
        agreement = df["agreement_uuid"].unique().tolist()[-1]
        df = df[df["agreement_uuid"] == agreement]

        # Initialize inference model
        inference_model = ClassifierInference(num_workers=7)

        # Run inference
        start = time.time()
        classified_result = inference_model.classify(df)
        inference_time = time.time() - start

        print(f"Inference time: {inference_time:.2f} seconds")
        # pprint.pprint(
        #     [
        #         {
        #             "postprocess_modified": c["postprocess_modified"],
        #             "class": c["pred_class"],
        #             "body": c["pred_probs"]["body"],
        #             "sig": c["pred_probs"]["sig"],
        #             "order": i,
        #         }
        #         for i, c in enumerate(classified_result[0])
        #     ]
        # )
        # pprint.pprint(classified_result)

    else:
        raise RuntimeError(f"Invalid mode: {mode}. Use 'train' or 'test'")


if __name__ == "__main__":
    main(mode="test", file="overwrite")
