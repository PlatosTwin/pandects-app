# Standard Library
import os

# Data classes
from etl.domain.classifier_model import PageClassifier, PageDataModule

# Environment config
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Third-Party Libraries
import pandas as pd

# PyTorch & Lightning
import torch

torch.set_float32_matmul_precision("high")

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything

seed_everything(42, workers=True, verbose=False)

from optuna import create_study
from optuna.integration import PyTorchLightningPruningCallback


class ClassifierTrainer:

    def __init__(
        self,
        num_trials=3,
        max_epochs=5,
    ):
        self.NUM_TRIALS = num_trials
        self.MAX_EPOCHS = max_epochs
        self.VOCAB_SIZE = 20_000
        self.EMBED_DIM = 256
        self.HIDDEN_DIM = 32

        if torch.backends.mps.is_available():
            self.DEVICE = "mps"
        elif torch.cuda.is_available():
            self.DEVICE = "cuda"
        else:
            self.DEVICE = "cpu"

        self.data = pd.DataFrame()

    def _load_data(self):
        """
        Load and split data, stratified by presence of tags.
        """
        df = pd.read_csv("../data/classified-data.csv")

        print(f"Loaded data shape: {df.shape}")
        print(df.head())

        self.data = df.copy()

    def _get_callbacks(self, trial=None):
        # single checkpoint callback for best val_loss
        checkpoint_cb = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_loss:.4f}",
        )
        early_stop_cb = EarlyStopping(monitor="val_loss", patience=3, mode="min")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        pruning_cb = (
            [PyTorchLightningPruningCallback(trial, monitor="val_loss")]
            if trial is not None
            else []
        )
        return checkpoint_cb, early_stop_cb, lr_monitor, pruning_cb

    def _build(self, params):
        """Instantiate DataModule and Model from a dict of hyperparams."""
        dm = PageDataModule(
            df=self.data,
            batch_size=params["batch_size"],
            num_workers=7,
        )
        model = PageClassifier(
            vocab_size=self.VOCAB_SIZE,
            embed_dim=self.EMBED_DIM,
            num_features=5,
            hidden_dim=self.HIDDEN_DIM,
            num_classes=4,
            learning_rate=params["lr"],
        )
        return dm, model

    def _objective(self, trial):
        params = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        }

        dm, model = self._build(params)
        checkpoint_cb, early_stop_cb, lr_monitor, pruning_cb = self._get_callbacks(
            trial
        )

        trainer = pl.Trainer(
            max_epochs=self.MAX_EPOCHS,
            accelerator=self.DEVICE,
            precision="16-mixed",
            devices=1,
            logger=TensorBoardLogger("tb_logs", name="optuna"),
            callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, *pruning_cb],
            log_every_n_steps=10,
            deterministic=True,
        )
        trainer.fit(model, datamodule=dm)

        val = trainer.callback_metrics["val_loss"].item()

        del model, dm, trainer, checkpoint_cb, early_stop_cb, lr_monitor
        if pruning_cb:
            del pruning_cb

        return val

    def run(self):
        self._load_data()

        study = create_study(direction="minimize")
        study.optimize(self._objective, n_trials=self.NUM_TRIALS, gc_after_trial=True)

        print("Finished HPO 👉")
        print(f"  Best val_loss: {study.best_value:.4f}")
        print("  Best hyperparams:")
        for k, v in study.best_trial.params.items():
            print(f"    • {k}: {v}")

        # Retrain best model to get its checkpoint on disk
        best_params = study.best_trial.params
        dm, model = self._build(best_params)
        checkpoint_cb, early_stop_cb, lr_monitor, _ = self._get_callbacks(trial=None)

        trainer = pl.Trainer(
            max_epochs=self.MAX_EPOCHS,
            accelerator=self.DEVICE,
            devices=1,
            logger=TensorBoardLogger("tb_logs", name="final"),
            callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
            log_every_n_steps=10,
        )
        trainer.fit(model, datamodule=dm)

        # # load best model from checkpoint
        # best_path = checkpoint_cb.best_model_path
        # best_model = NERTagger.load_from_checkpoint(best_path)

        # best_model = NERTagger.load_from_checkpoint(
        #     "/Users/nikitabogdanov/Downloads/best-epoch=24-val_loss=0.0000.ckpt"
        # )
        # trainer = pl.Trainer()
        # infer_loader = NERDataModule(
        #     train_data=self.train_data,
        #     val_data=self.val_data[:3],
        #     tokenizer_name=self.MODEL_NAME,
        #     label_list=self.LABEL_LIST,
        #     batch_size=1,
        #     max_length=1800,
        #     num_workers=0,
        # )

        # print("\n=== Inference on best model ===")
        # preds = trainer.predict(best_model, datamodule=infer_loader)

        # self._print_preds(preds, infer_loader)


def main():
    classifier_trainer = ClassifierTrainer(num_trials=5, max_epochs=5)
    classifier_trainer.run()


if __name__ == "__main__":
    main()
