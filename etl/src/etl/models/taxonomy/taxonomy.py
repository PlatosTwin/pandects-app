"""
Main taxonomy training and inference module.

This module mirrors ner.py in structure and provides the main entry points for
training and testing the taxonomy classifier using PyTorch Lightning with
optional hyperparameter optimization via Optuna. It supports two modes for text
representation: 'transformer' (HF model) and 'tfidf'.
"""
# pyright: reportUnknownVariableType=false

# Standard library
import os
import time
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast, Literal

# Environment config
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Third-party
import pandas as pd
import lightning.pytorch as pl
import torch
import numpy as np
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from optuna import Trial, create_study
from optuna.integration import PyTorchLightningPruningCallback

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# Local modules
from .taxonomy_constants import (
    TAXONOMY_LABEL_LIST,
    TAXONOMY_CKPT_PATH,
    TAXONOMY_VECTORIZER_PATH,
)
from .taxonomy_classes import (
    TransformerDataModule,
    TfidfDataModule,
    TaxonomyClassifier,
)

_ = seed_everything(42, workers=True, verbose=False)
torch.set_float32_matmul_precision("high")

PrecisionInput = Literal["bf16-mixed", "32-true"]
LOG_DIR = Path(__file__).resolve().parent / "logs"


class _LogitsOutput(Protocol):
    logits: torch.Tensor


def _combine_text(a: str, s: str, t: str) -> str:
    a = a or ""
    s = s or ""
    t = t or ""
    return f"[ARTICLE] {a}\n[SECTION] {s}\n[TEXT] {t}"


@dataclass
class TaxonomyConfig:
    data_parquet: str
    model_name: str
    label_list: list[str]
    mode: Literal["transformer", "tfidf"]
    num_trials: int
    max_epochs: int
    batch_size: int = 16
    max_length: int = 1024
    use_stratify: bool = True
    tfidf_max_features: int | None = 50_000


class _DenseConvertible(Protocol):
    def astype(self, dtype: str) -> "_DenseConvertible": ...

    def toarray(self) -> np.ndarray: ...


class TaxonomyTrainer:
    """
    Orchestrates hyperparameter optimization and training of TaxonomyClassifier.
    """

    def __init__(self, cfg: TaxonomyConfig) -> None:
        self.cfg: TaxonomyConfig = cfg
        if torch.backends.mps.is_available():
            self.device: str = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.train_rows: dict[str, list[str] | list[int]] = {}
        self.val_rows: dict[str, list[str] | list[int]] = {}
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.X_val: np.ndarray | None = None
        self.y_val: np.ndarray | None = None
        self.vectorizer: TfidfVectorizer | None = None

    def _load_data(self) -> None:
        df = pd.read_parquet(self.cfg.data_parquet)  # pyright: ignore[reportUnknownMemberType]
        assert set(["article_title", "section_title", "section_text", "label"]).issubset(df.columns)

        label2id = {l: i for i, l in enumerate(self.cfg.label_list)}
        df = df.dropna(subset=["section_text"]).fillna("")  # pyright: ignore[reportUnknownMemberType]
        df = df[df["label"].isin(self.cfg.label_list)].copy()  # pyright: ignore[reportUnknownMemberType]
        df["label_ids"] = df["label"].map(label2id)

        strat = df["label_ids"] if self.cfg.use_stratify else None
        tr, va = cast(
            tuple[pd.DataFrame, pd.DataFrame],
            cast(
                object,
                train_test_split(
                    df, test_size=0.2, random_state=42, stratify=strat
                ),
            ),
        )

        if self.cfg.mode == "transformer":
            article_title_tr = tr["article_title"].astype(str).tolist()
            section_title_tr = tr["section_title"].astype(str).tolist()
            section_text_tr = tr["section_text"].astype(str).tolist()
            label_ids_tr = tr["label_ids"].astype(int).tolist()
            article_title_va = va["article_title"].astype(str).tolist()
            section_title_va = va["section_title"].astype(str).tolist()
            section_text_va = va["section_text"].astype(str).tolist()
            label_ids_va = va["label_ids"].astype(int).tolist()
            self.train_rows = {
                "article_title": article_title_tr,
                "section_title": section_title_tr,
                "section_text": section_text_tr,
                "label_ids": label_ids_tr,
            }
            self.val_rows = {
                "article_title": article_title_va,
                "section_title": section_title_va,
                "section_text": section_text_va,
                "label_ids": label_ids_va,
            }
        else:
            article_title_tr = tr["article_title"].astype(str).tolist()
            section_title_tr = tr["section_title"].astype(str).tolist()
            section_text_tr = tr["section_text"].astype(str).tolist()
            article_title_va = va["article_title"].astype(str).tolist()
            section_title_va = va["section_title"].astype(str).tolist()
            section_text_va = va["section_text"].astype(str).tolist()
            texts_tr = [
                _combine_text(str(a), str(s), str(t))
                for a, s, t in zip(article_title_tr, section_title_tr, section_text_tr)
            ]
            texts_va = [
                _combine_text(str(a), str(s), str(t))
                for a, s, t in zip(article_title_va, section_title_va, section_text_va)
            ]
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9,
                strip_accents="unicode",
                lowercase=True,
                max_features=self.cfg.tfidf_max_features,
            )
            Xtr = cast(
                _DenseConvertible,
                self.vectorizer.fit_transform(texts_tr),  # pyright: ignore[reportUnknownMemberType]
            )
            Xva = cast(
                _DenseConvertible,
                self.vectorizer.transform(texts_va),  # pyright: ignore[reportUnknownMemberType]
            )

            # Convert to dense for a small MLP (could keep sparse for linear models)
            self.X_train = Xtr.astype("float32").toarray()
            self.X_val = Xva.astype("float32").toarray()
            self.y_train = tr["label_ids"].astype("int64").to_numpy()  # pyright: ignore[reportUnknownMemberType]
            self.y_val = va["label_ids"].astype("int64").to_numpy()  # pyright: ignore[reportUnknownMemberType]

    def _get_callbacks(
        self, trial: Trial | None = None
    ) -> tuple[
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
        TQDMProgressBar,
        list[PyTorchLightningPruningCallback],
    ]:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_f1_micro",
            mode="max",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_f1_micro:.4f}",
        )
        early_stop_callback = EarlyStopping(monitor="val_f1_micro", patience=3, mode="max")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        pruning_callback = (
            [PyTorchLightningPruningCallback(trial, monitor="val_f1_micro")] if trial is not None else []
        )
        progress_bar_callback = TQDMProgressBar(refresh_rate=100)
        return checkpoint_callback, early_stop_callback, lr_monitor, progress_bar_callback, pruning_callback

    def _build(
        self, params: dict[str, float | int]
    ) -> tuple[pl.LightningDataModule, pl.LightningModule]:
        label_list = self.cfg.label_list
        id2label = {i: l for i, l in enumerate(label_list)}
        if self.cfg.mode == "transformer":
            dm = TransformerDataModule(
                model_name=self.cfg.model_name,
                train_rows=self.train_rows,
                val_rows=self.val_rows,
                label_list=label_list,
                batch_size=int(params["batch_size"]),
                max_length=int(params.get("max_length", self.cfg.max_length)),
                num_workers=7,
            )
            model = TaxonomyClassifier(
                mode="transformer",
                num_labels=len(label_list),
                id2label=id2label,
                model_name=self.cfg.model_name,
                learning_rate=float(params["lr"]),
                weight_decay=float(params["weight_decay"]),
                warmup_steps_pct=float(params["warmup_steps_pct"]),
            )
        else:
            assert self.X_train is not None and self.y_train is not None
            assert self.X_val is not None and self.y_val is not None
            dm = TfidfDataModule(
                X_train=self.X_train,
                y_train=self.y_train,
                X_val=self.X_val,
                y_val=self.y_val,
                batch_size=int(params["batch_size"]),
                num_workers=3,
            )
            input_dim = int(cast(tuple[int, ...], self.X_train.shape)[1])
            model = TaxonomyClassifier(
                mode="tfidf",
                num_labels=len(label_list),
                id2label=id2label,
                input_dim=input_dim,
                hidden_dim=int(params.get("hidden_dim", 512)),
                dropout=float(params.get("dropout", 0.1)),
                learning_rate=float(params.get("lr", 1e-3)),
                weight_decay=float(params.get("weight_decay", 0.0)),
            )
        return dm, model

    def _objective(self, trial: Trial) -> float:
        params = {
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "lr": trial.suggest_float("lr", 1e-5 if self.cfg.mode == "transformer" else 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
            "warmup_steps_pct": trial.suggest_float("warmup_steps_pct", 0.0, 0.2) if self.cfg.mode == "transformer" else 0.0,
        }
        if self.cfg.mode == "tfidf":
            params.update(
                {
                    "hidden_dim": trial.suggest_categorical("hidden_dim", [256, 512, 768]),
                    "dropout": trial.suggest_float("dropout", 0.0, 0.3),
                }
            )

        dm, model = self._build(params)
        (
            checkpoint_callback,
            early_stop_callback,
            lr_monitor,
            progress_bar_callback,
            pruning_callback,
        ) = self._get_callbacks(trial)

        trainer = pl.Trainer(
            max_epochs=self.cfg.max_epochs,
            accelerator=self.device,
            precision=self._trainer_precision(),
            devices=1,
            logger=TensorBoardLogger(str(LOG_DIR), name="taxonomy-optuna"),
            callbacks=[
                checkpoint_callback,
                early_stop_callback,
                lr_monitor,
                progress_bar_callback,
                *pruning_callback,
            ],
            log_every_n_steps=10,
            deterministic=True,
        )
        trainer.fit(model, datamodule=dm)

        val_f1_micro = float(trainer.callback_metrics["val_f1_micro"])
        return val_f1_micro

    def run(self) -> None:
        self._load_data()

        study = create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.cfg.num_trials, gc_after_trial=True)

        print("Finished hyperparameter optimization 👉")
        print(f"  Best val_f1_micro: {study.best_value:.4f}")
        print("  Best hyperparameters:")
        best_params = cast(dict[str, float | int], study.best_trial.params)
        for key, value in best_params.items():
            print(f"    • {key}: {value}")

        dm, model = self._build(best_params)
        (
            checkpoint_callback,
            early_stop_callback,
            lr_monitor,
            progress_bar_callback,
            _,
        ) = self._get_callbacks()

        trainer = pl.Trainer(
            max_epochs=self.cfg.max_epochs,
            accelerator=self.device,
            precision=self._trainer_precision(),
            devices=1,
            logger=TensorBoardLogger(str(LOG_DIR), name="taxonomy-final"),
            callbacks=[
                checkpoint_callback,
                early_stop_callback,
                lr_monitor,
                progress_bar_callback,
            ],
            log_every_n_steps=10,
            deterministic=True,
        )
        trainer.fit(model, datamodule=dm)

        best_ckpt_path = cast(str, checkpoint_callback.best_model_path)
        if not best_ckpt_path:
            raise RuntimeError("No best checkpoint produced during training.")
        Path(TAXONOMY_CKPT_PATH).parent.mkdir(parents=True, exist_ok=True)
        _ = shutil.copyfile(best_ckpt_path, TAXONOMY_CKPT_PATH)

        # Save TF-IDF vectorizer if needed
        if self.cfg.mode == "tfidf" and self.vectorizer is not None:
            Path(TAXONOMY_VECTORIZER_PATH).parent.mkdir(parents=True, exist_ok=True)
            with open(TAXONOMY_VECTORIZER_PATH, "wb") as f:
                pickle.dump(self.vectorizer, f)

    def _trainer_precision(self) -> PrecisionInput:
        if self.device in ("mps", "cuda"):
            return "bf16-mixed"
        return "32-true"


class TaxonomyInference:
    """
    Inference for taxonomy classification. Supports transformer or tfidf modes.
    """

    def __init__(
        self,
        ckpt_path: str,
        label_list: list[str],
        mode: Literal["transformer", "tfidf"],
        model_name: str | None = None,
        vectorizer_path: str | None = None,
        max_length: int = 1024,
    ) -> None:
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model: TaxonomyClassifier
        self.mode: Literal["transformer", "tfidf"]
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.vectorizer: TfidfVectorizer | None = None

        if mode == "transformer":
            model = TaxonomyClassifier.load_from_checkpoint(  # pyright: ignore[reportUnknownMemberType]
                ckpt_path, map_location=self.device
            )
            if model.mode != "transformer":
                raise ValueError("Checkpoint is not a transformer taxonomy model.")
            _ = model.eval()
            _ = model.to(self.device)
            self.model = model
            self.mode = "transformer"
            self.max_length = max_length
            hparams = cast(object, model.hparams)
            tokenizer_name = model_name or cast(
                str | None, getattr(hparams, "model_name", None)
            )
            if tokenizer_name is None:
                raise ValueError("Transformer inference requires `model_name` (or a checkpoint that saved it).")
            self.tokenizer = cast(
                PreTrainedTokenizerBase,
                AutoTokenizer.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
                    tokenizer_name, use_fast=True
                ),
            )
        else:
            model = TaxonomyClassifier.load_from_checkpoint(  # pyright: ignore[reportUnknownMemberType]
                ckpt_path, map_location=self.device
            )
            if model.mode != "tfidf":
                raise ValueError("Checkpoint is not a TF-IDF taxonomy model.")
            _ = model.eval()
            _ = model.to(self.device)
            self.model = model
            self.mode = "tfidf"
            with open(vectorizer_path or TAXONOMY_VECTORIZER_PATH, "rb") as f:
                self.vectorizer = pickle.load(f)

        expected_id2label = {i: l for i, l in enumerate(label_list)}
        model_id2label = getattr(self.model, "id2label", None)
        if model_id2label is not None and model_id2label != expected_id2label:
            raise ValueError("`label_list` does not match the label mapping stored in the checkpoint.")
        self.id2label = expected_id2label

    def _combine(self, a: str, s: str, t: str) -> str:
        return _combine_text(a, s, t)

    def predict(self, rows: list[dict[str, str]]) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        batch_size = 32
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            texts = [
                self._combine(r.get("article_title", ""), r.get("section_title", ""), r.get("section_text", ""))
                for r in batch_rows
            ]

            if self.mode == "transformer":
                if self.tokenizer is None:
                    raise RuntimeError(
                        "Tokenizer not initialized for transformer inference."
                    )
                tokenizer = self.tokenizer
                assert tokenizer is not None
                enc = cast(
                    dict[str, torch.Tensor],
                    cast(
                        object,
                        tokenizer(
                            texts,
                            truncation=True,
                            max_length=self.max_length,
                            padding=True,
                            return_tensors="pt",
                        ),
                    ),
                )
                enc_tensors: dict[str, torch.Tensor] = {
                    k: v.to(self.device) for k, v in enc.items()
                }
                with torch.no_grad():
                    outputs = cast(
                        _LogitsOutput, self.model(**enc_tensors)
                    )
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    conf, pred = probs.max(dim=-1)
            else:
                if self.vectorizer is None:
                    raise RuntimeError("Vectorizer not initialized for TF-IDF inference.")
                features = cast(
                    _DenseConvertible,
                    self.vectorizer.transform(texts),
                ).astype("float32").toarray()
                features_tensor = torch.tensor(
                    features, dtype=torch.float32, device=self.device
                )
                with torch.no_grad():
                    outputs = cast(
                        _LogitsOutput, self.model(features=features_tensor)
                    )
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    conf, pred = probs.max(dim=-1)

            for i in range(len(texts)):
                pred_id = int(pred[i].item())
                out.append(
                    {
                        "pred_id": pred_id,
                        "pred_label": self.id2label[pred_id],
                        "confidence": float(conf[i].item()),
                    }
                )
        return out


def main(mode: str = "test") -> None:
    """
    Main entry point for training or test inference.
    """
    if mode == "train":
        cfg = TaxonomyConfig(
            data_parquet="etl/src/etl/models/taxonomy/data/taxonomy-data.parquet",
            model_name="answerdotai/ModernBERT-base",
            label_list=TAXONOMY_LABEL_LIST,
            mode="transformer",  # or "tfidf"
            num_trials=10,
            max_epochs=6,
            batch_size=16,
            max_length=1024,
        )
        trainer = TaxonomyTrainer(cfg)
        trainer.run()
    elif mode == "test":
        # Example quick test using TF-IDF or transformer
        inf = TaxonomyInference(
            ckpt_path=TAXONOMY_CKPT_PATH,
            label_list=TAXONOMY_LABEL_LIST,
            mode="transformer",
        )
        samples = [
            {
                "article_title": "Representations and Warranties",
                "section_title": "Buyer Representations",
                "section_text": "The Buyer represents that it is duly organized and in good standing...",
            },
            {
                "article_title": "Miscellaneous",
                "section_title": "Notices",
                "section_text": "Any notice shall be deemed given when delivered by certified mail...",
            },
        ]
        start = time.time()
        preds = inf.predict(samples)
        elapsed = time.time() - start
        print(preds)
        print(f"Inference time: {elapsed:.2f}s")
    else:
        raise RuntimeError(f"Invalid mode: {mode}. Use 'train' or 'test'")


if __name__ == "__main__":
    main(mode="test")
