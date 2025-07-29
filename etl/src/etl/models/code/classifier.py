# Standard Library
import os
import pickle

# ensure deterministic behavior
import torch
import pandas as pd
import lightning.pytorch as pl

# Training utilities
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from optuna import create_study
from optuna.integration import PyTorchLightningPruningCallback

# Custom
from etl.models.code.classifier_classes import (
    PageDataModule,
    PageClassifier,
    prepare_sample,
)
from etl.models.code.constants import CLASSIFIER_LABEL2IDX_PATH, CLASSIFIER_VOCAB_PATH, CLASSIFIER_CKPT_PATH

# Prevent tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        self.data_csv = data_csv
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
        df = pd.read_csv(self.data_csv)
        if not {"html", "text", "label"}.issubset(df.columns):
            raise ValueError("CSV must contain 'html', 'text', and 'label' columns")
        print(f"[data] loaded {df.shape[0]} rows from {self.data_csv}")
        self.df = df

    def _get_callbacks(self, trial=None):
        """Instantiate checkpointing, early stopping, LR monitor, (and Optuna pruning)."""
        ckpt = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_loss:.4f}",
        )
        early_stop = EarlyStopping(monitor="val_loss", patience=3, mode="min")
        lr_mon = LearningRateMonitor(logging_interval="step")
        prune_cb = (
            [PyTorchLightningPruningCallback(trial, monitor="val_loss")]
            if trial is not None
            else []
        )
        return ckpt, early_stop, lr_mon, prune_cb

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
            max_vocab_size=params.get("max_vocab_size", 20_000),
            max_seq_len=params.get("max_seq_len", 300),
        )

        dm.setup()  # usually not necessary, but we use vars from dm below

        model = PageClassifier(
            vocab_size=dm.vocab_size,
            embed_dim=params["embed_dim"],
            num_features=dm.num_features,
            hidden_dim=params["hidden_dim"],
            num_classes=dm.num_classes,
            learning_rate=params["lr"],
        )
        return dm, model

    def _objective(self, trial):
        """Optuna objective: builds, trains, and returns validation loss."""
        # define hyperparameter search space
        params = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "max_vocab_size": trial.suggest_categorical(
                "max_vocab_size", [5_000, 10_000, 20_000]
            ),
            "max_seq_len": trial.suggest_categorical("max_seq_len", [100, 200, 300]),
            "embed_dim": trial.suggest_categorical("embed_dim", [50, 100, 200]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        }

        dm, model = self._build(params)
        ckpt, early_stop, lr_mon, prune_cb = self._get_callbacks(trial)

        trainer = pl.Trainer(
            max_epochs=self.MAX_EPOCHS,
            accelerator=self.DEVICE,
            devices=1,
            precision="bf16" if self.DEVICE in ("cuda", "mps") else 32,
            logger=TensorBoardLogger("tb_logs", name="optuna"),
            callbacks=[ckpt, early_stop, lr_mon, *prune_cb],
            log_every_n_steps=10,
            deterministic=True,
        )
        trainer.fit(model, datamodule=dm)
        val_loss = trainer.callback_metrics["val_loss"].item()

        # clean up to avoid memory leaks
        del trainer, model, dm, ckpt, early_stop, lr_mon, prune_cb
        return val_loss

    def run(self):
        """Execute HPO, then retrain final model with best hyperparameters."""
        self._load_data()

        study = create_study(direction="minimize")
        study.optimize(self._objective, n_trials=self.NUM_TRIALS, gc_after_trial=True)

        print(">> HPO complete")
        print(f"   Best val_loss: {study.best_value:.4f}")
        print("   Best params:")
        for k, v in study.best_trial.params.items():
            print(f"     • {k}: {v}")

        # final training with best hyperparameters
        best_params = study.best_trial.params
        dm, model = self._build(best_params)

        with open(CLASSIFIER_VOCAB_PATH, "wb") as f:
            pickle.dump(dm.vocab, f)
        with open(CLASSIFIER_LABEL2IDX_PATH, "wb") as f:
            pickle.dump(dm.label2idx, f)
        print(f"Saved {CLASSIFIER_VOCAB_PATH} and {CLASSIFIER_LABEL2IDX_PATH}")

        ckpt, early_stop, lr_mon, _ = self._get_callbacks(trial=None)

        trainer = pl.Trainer(
            max_epochs=self.MAX_EPOCHS,
            accelerator=self.DEVICE,
            devices=1,
            logger=TensorBoardLogger("tb_logs", name="final"),
            callbacks=[ckpt, early_stop, lr_mon],
            log_every_n_steps=10,
        )
        trainer.fit(model, datamodule=dm)


class ClassifierInference:
    """
    Load a trained PageClassifier from checkpoint and run single-example or batch inference
    on raw HTML/text pairs.
    """

    def __init__(
        self,
        ckpt_path: str,
        vocab: dict,
        label2idx: dict,
        max_seq_len: int = 300,
        device: str | None = None,
    ):
        # load model
        self.model: PageClassifier = PageClassifier.load_from_checkpoint(ckpt_path)
        # mappings and padding/unk indices
        self.vocab = vocab
        self.label2idx = label2idx
        self.idx2label = {i: l for l, i in label2idx.items()}
        self.pad_idx = vocab.get("<PAD>", 0)
        self.unk_idx = vocab.get("<UNK>", 1)
        self.max_seq_len = max_seq_len

        # device
        if device:
            self.device = torch.device(device)
        else:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else (
                    torch.device("mps")
                    if torch.backends.mps.is_available()
                    else torch.device("cpu")
                )
            )
        self.model.to(self.device).eval()

    def _prepare(self, html: str, text: str):
        idxs, length, feats = prepare_sample(html, text, self.vocab, self.max_seq_len)
        idxs_t = torch.tensor([idxs], dtype=torch.long, device=self.device)
        lengths_t = torch.tensor([length], dtype=torch.long, device=self.device)
        feats_t = torch.tensor([feats], dtype=torch.float, device=self.device)
        return idxs_t, lengths_t, feats_t

    def classify(self, html: str, text: str) -> dict:
        """
        Returns a dict with:
          - predicted_class: the class label with highest probability
          - all_preds: mapping of each class to its probability
        """
        idxs_t, lengths_t, feats_t = self._prepare(html, text)
        with torch.no_grad():
            logits = self.model(idxs_t, lengths_t, feats_t)
            probs = torch.softmax(logits, dim=1)
            # turn into a flat list of floats
            probs_list = probs.squeeze(0).cpu().tolist()
            # build the per-class dict
            all_preds = {
                f"class_{self.idx2label[i]}": float(probs_list[i])
                for i in range(len(probs_list))
            }
            # pick the top class
            pred_idx = int(torch.argmax(probs, dim=1)[0])
            predicted_class = self.idx2label[pred_idx]
        return {"predicted_class": predicted_class, "all_preds": all_preds}


def main():
    classifier = ClassifierTrainer(
        data_csv="../data/page-data.csv",
        num_trials=10,
        max_epochs=10,
    )
    classifier.run()

    #####

    with open(CLASSIFIER_VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    with open(CLASSIFIER_LABEL2IDX_PATH, "rb") as f:
        label2idx = pickle.load(f)

    infer = ClassifierInference(
        ckpt_path=CLASSIFIER_CKPT_PATH,
        vocab=vocab,
        label2idx=label2idx,
        max_seq_len=300,
    )

    html = "<html><body><h1>Example</h1></body></html>"
    text = "This is the page text that you want to classify."
    out = infer.classify(html, text)
    print(out)


if __name__ == "__main__":
    main()
