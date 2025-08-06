import os
import time
from typing import Optional
import pprint

# Disable Tokenizers parallelism before loading any HF/Lightning modules
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
import torch
import torch.nn.functional as F

from classifier_classes import PageClassifier, PageDataModule
from shared_constants import CLASSIFIER_CKPT_PATH, CLASSIFIER_XGB_PATH

# Reproducibility
pl.seed_everything(42, workers=True, verbose=False)


class ClassifierTrainer:
    def __init__(
        self,
        data_csv: str,
        num_trials: int,
        max_epochs: int,
        batch_size: int = 32,
        val_split: float = 0.2,
        num_workers: int = 0,
    ):
        """
        Orchestrates data loading, hyperparameter search, and training
        of PageClassifier via PageDataModule.
        """
        self.data_file = data_csv
        self.NUM_TRIALS = num_trials
        self.MAX_EPOCHS = max_epochs
        self.DEFAULT_BATCH = batch_size
        self.VAL_SPLIT = val_split
        self.NUM_WORKERS = num_workers

        if torch.backends.mps.is_available():
            self.DEVICE = "mps"
        elif torch.cuda.is_available():
            self.DEVICE = "cuda"
        else:
            self.DEVICE = "cpu"

    def _load_data(self):
        """Load the classification DataFrame from CSV."""
        df = pd.read_parquet(self.data_file)
        if not {"html", "text", "label"}.issubset(df.columns):
            raise ValueError("CSV must contain 'html', 'text', and 'label' columns")
        print(f"[data] loaded {df.shape[0]} rows from {self.data_file}")
        self.df = df

    def _get_callbacks(self, trial=None, ckpt=None):
        """Instantiate checkpointing, early stopping, LR monitor, (and Optuna pruning)."""
        if ckpt:
            dirpath = os.path.dirname(ckpt)
            filename = os.path.splitext(os.path.basename(ckpt))[0]
        else:
            dirpath = None
            filename = "best-{epoch:02d}-{val_f1:.4f}"

        ckpt = ModelCheckpoint(
            monitor="val_f1",
            mode="max",
            save_top_k=1,
            filename=filename,
            dirpath=dirpath,
        )

        early_stop = EarlyStopping(monitor="val_f1", patience=3, mode="max")
        lr_mon = LearningRateMonitor(logging_interval="step")
        prune_cb = (
            [PyTorchLightningPruningCallback(trial, monitor="val_f1")]
            if trial is not None
            else []
        )
        progress_bar_cb = TQDMProgressBar(refresh_rate=15)

        return ckpt, early_stop, lr_mon, progress_bar_cb, prune_cb

    def _build(self, params: dict):
        """
        Instantiate DataModule and Model with a given hyperparameter set.
        Must call setup() to populate num_features, vocab_size, etc.
        """
        dm = PageDataModule(
            df=self.df,
            batch_size=params.get("batch_size", self.DEFAULT_BATCH),
            val_split=self.VAL_SPLIT,
            num_workers=self.NUM_WORKERS,
            xgb_path=CLASSIFIER_XGB_PATH,
        )

        dm.setup()  # usually not necessary, but we use vars from dm below

        model = PageClassifier(
            num_classes=dm.num_classes,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            hidden_dim=params["hidden_dim"],
            dropout=params["dropout"],
            lstm_dropout=params["lstm_dropout"],
            lstm_hidden=params["lstm_hidden"],
            lstm_num_layers=params["lstm_num_layers"],
        )
        return dm, model

    def _objective(self, trial):
        """Optuna objective: builds, trains, and returns validation loss."""
        # define hyperparameter search space
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

        dm, model = self._build(params)
        ckpt, early_stop, lr_mon, progress_bar_cb, prune_cb = self._get_callbacks(trial)

        trainer = pl.Trainer(
            max_epochs=self.MAX_EPOCHS,
            accelerator=self.DEVICE,
            devices=1,
            precision="bf16-mixed",
            logger=TensorBoardLogger("tb_logs", name="optuna"),
            callbacks=[ckpt, early_stop, lr_mon, progress_bar_cb, *prune_cb],
            log_every_n_steps=10,
            deterministic=True,
        )
        trainer.fit(model, datamodule=dm)
        # val_loss = trainer.callback_metrics["val_loss"].item()
        val_f1 = trainer.callback_metrics["val_f1"].item()

        # clean up to avoid memory leaks
        del trainer, model, dm, ckpt, early_stop, lr_mon, prune_cb

        return val_f1

    def run(self):
        """Execute HPO, then retrain final model with best hyperparameters."""
        self._load_data()

        study = create_study(direction="maximize")
        study.optimize(
            self._objective,
            n_trials=self.NUM_TRIALS,
            gc_after_trial=True,
            callbacks=[
                lambda study, trial: study.stop() if study.best_value >= 1.0 else None
            ],
        )

        print(">> HPO complete")
        print(f"   Best val_f1: {study.best_value:.4f}")
        print("   Best params:")
        for k, v in study.best_trial.params.items():
            print(f"     • {k}: {v}")

        # final training with best hyperparameters
        best_params = study.best_trial.params
        dm, model = self._build(best_params)

        ckpt, early_stop, lr_mon, progress_bar_cb, _ = self._get_callbacks(
            ckpt=CLASSIFIER_CKPT_PATH
        )

        trainer = pl.Trainer(
            max_epochs=self.MAX_EPOCHS,
            accelerator=self.DEVICE,
            devices=1,
            logger=TensorBoardLogger("tb_logs", name="final"),
            callbacks=[ckpt, early_stop, lr_mon, progress_bar_cb],
            log_every_n_steps=10,
        )
        trainer.fit(model, datamodule=dm)


class ClassifierInference:
    """
    Wraps a trained PageClassifier LightningModule for easy batch inference.

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
    ):
        # device selection
        if device is not None:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # load model (assumes you called self.save_hyperparameters() in __init__)
        self.model = PageClassifier.load_from_checkpoint(ckpt_path)
        self.model.to(self.device)
        self.model.eval()

        self.xgb_path = xgb_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def classify(self, df: pd.DataFrame):
        """
        Run inference on a DataFrame of pages and return a DataFrame with:
        - 'probabilities': dict[label → proba]
        - 'pred_class'   : most likely label
        """
        # set up DataModule
        dm = PageDataModule(
            df=df,
            batch_size=self.batch_size,
            val_split=0.0,
            num_workers=self.num_workers,
            xgb_path=self.xgb_path,
        )
        dm.setup(stage="predict")

        # predict: expect a list of tensors [batch_size, num_classes]
        trainer = pl.Trainer()
        outputs = trainer.predict(self.model, dataloaders=dm.predict_dataloader())

        return outputs


def main(mode):
    if mode == "train":
        classifier_trainer = ClassifierTrainer(
            data_csv="etl/src/etl/models/data/page-data.parquet",
            num_trials=30,
            max_epochs=25,
            num_workers=7,
        )
        classifier_trainer.run()

    elif mode == "test":
        df = pd.read_parquet("etl/src/etl/models/data/page-data.parquet")
        agreement = df["agreement_uuid"].unique().tolist()[-1]
        df = df[df["agreement_uuid"] == agreement]

        inference_model = ClassifierInference(num_workers=7)

        start = time.time()
        classified_result = inference_model.classify(df)
        print(time.time() - start)
        pprint.pprint(classified_result)

    else:
        raise RuntimeError(f"Invalid value for mode: {mode}")


if __name__ == "__main__":
    main(mode="test")
