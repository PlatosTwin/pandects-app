"""
Taxonomy (section type) classification models and datasets.

This mirrors the structure of the NER modules but focuses on strict
multi-class classification of sections. Supports two modes:
  - transformer: ModernBERT (or any HF encoder) via sequence classification
  - tfidf: TF-IDF vectorization with a small MLP classifier
"""
# pyright: reportUnknownMemberType=false

# Standard library
import os
from types import SimpleNamespace
from typing import Literal, Protocol, cast

# Environment config
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Third-party
import lightning.pytorch as pl
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Optimizer
from lightning.pytorch.utilities.types import LRSchedulerConfig
from torchmetrics.classification import Accuracy, F1Score as F1
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.optimization import get_linear_schedule_with_warmup  # pyright: ignore[reportUnknownVariableType]


class _LogitsOutput(Protocol):
    logits: torch.Tensor


class TextLabelDataset(Dataset[dict[str, object]]):
    """
    Dataset for text classification. Builds the combined text from
    article_title + section_title + section_text.
    """

    def __init__(
        self,
        article_title: list[str],
        section_title: list[str],
        section_text: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizerBase | None,
        max_length: int,
        mode: Literal["transformer", "tfidf"],
    ) -> None:
        self.article_title = article_title
        self.section_title = section_title
        self.section_text = section_text
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

    def __len__(self) -> int:
        return len(self.labels)

    def _combine(self, a: str, s: str, t: str) -> str:
        a = a or ""
        s = s or ""
        t = t or ""
        return f"[ARTICLE] {a}\n[SECTION] {s}\n[TEXT] {t}"

    def __getitem__(self, idx: int) -> dict[str, object]:
        txt = self._combine(self.article_title[idx], self.section_title[idx], self.section_text[idx])
        y = int(self.labels[idx])

        if self.mode == "transformer":
            assert self.tokenizer is not None
            enc = cast(
                dict[str, object],
                cast(
                    object,
                    self.tokenizer(
                        txt,
                        truncation=True,
                        max_length=self.max_length,
                        padding=False,
                        return_tensors=None,
                    ),
                ),
            )
            enc["labels"] = y
            return enc
        else:
            # For TF-IDF mode, tokenization done outside. Return raw text and label.
            return {"text": txt, "labels": y}


class TransformerDataModule(pl.LightningDataModule):
    """
    DataModule for transformer-based taxonomy classification.
    """

    def __init__(
        self,
        model_name: str,
        train_rows: dict[str, list[str] | list[int]],
        val_rows: dict[str, list[str] | list[int]],
        label_list: list[str],
        batch_size: int,
        max_length: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.label2id = {label: idx for idx, label in enumerate(label_list)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(model_name, use_fast=True),
        )
        self.collator = DataCollatorWithPadding(self.tokenizer, padding=True)
        self.train_rows = train_rows
        self.val_rows = val_rows
        self.train_dataset: TextLabelDataset | None = None
        self.val_dataset: TextLabelDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = TextLabelDataset(
            article_title=[str(x) for x in self.train_rows["article_title"]],
            section_title=[str(x) for x in self.train_rows["section_title"]],
            section_text=[str(x) for x in self.train_rows["section_text"]],
            labels=[int(x) for x in self.train_rows["label_ids"]],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            mode="transformer",
        )
        self.val_dataset = TextLabelDataset(
            article_title=[str(x) for x in self.val_rows["article_title"]],
            section_title=[str(x) for x in self.val_rows["section_title"]],
            section_text=[str(x) for x in self.val_rows["section_text"]],
            labels=[int(x) for x in self.val_rows["label_ids"]],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            mode="transformer",
        )

    def train_dataloader(self) -> DataLoader[dict[str, object]]:
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Call setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

    def val_dataloader(self) -> DataLoader[dict[str, object]]:
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )


class TfidfDataModule(pl.LightningDataModule):
    """
    DataModule for TF-IDF-based taxonomy classification. Expects pre-transformed
    dense float32 arrays for X (features) and int64 for y (labels).
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset: TensorDataset | None = None
        self.val_dataset: TensorDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        Xtr = torch.tensor(self.X_train, dtype=torch.float32)
        ytr = torch.tensor(self.y_train, dtype=torch.long)
        Xva = torch.tensor(self.X_val, dtype=torch.float32)
        yva = torch.tensor(self.y_val, dtype=torch.long)
        self.train_dataset = TensorDataset(Xtr, ytr)
        self.val_dataset = TensorDataset(Xva, yva)

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, ...]]:
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Call setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, ...]]:
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class TaxonomyClassifier(pl.LightningModule):
    """
    LightningModule for strict multi-class taxonomy classification.
    Supports two modes: transformer or tfidf.
    """

    def __init__(
        self,
        mode: Literal["transformer", "tfidf"],
        num_labels: int,
        id2label: dict[int, str],
        # transformer-specific
        model_name: str | None = None,
        learning_rate: float = 3e-5,
        weight_decay: float = 0.01,
        warmup_steps_pct: float = 0.1,
        # tfidf-specific
        input_dim: int | None = None,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        vectorizer_path: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.mode = mode
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.model: torch.nn.Module

        if self.mode == "transformer":
            assert model_name is not None, "model_name must be provided for transformer mode"
            self.tokenizer = cast(
                PreTrainedTokenizerBase,
                AutoTokenizer.from_pretrained(model_name, use_fast=True),
            )
            self.model = cast(
                torch.nn.Module,
                AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=self.num_labels,
                    id2label=self.id2label,
                    label2id=self.label2id,
                ),
            )
        else:
            assert input_dim is not None, "input_dim must be provided for tfidf mode"
            layers: list[torch.nn.Module] = [
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, self.num_labels),
            ]
            self.model = torch.nn.Sequential(*layers)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_labels)
        self.train_f1_micro = F1(task="multiclass", num_classes=self.num_labels, average="micro")
        self.train_f1_macro = F1(task="multiclass", num_classes=self.num_labels, average="macro")

        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_labels)
        self.val_f1_micro = F1(task="multiclass", num_classes=self.num_labels, average="micro")
        self.val_f1_macro = F1(task="multiclass", num_classes=self.num_labels, average="macro")

    def forward(self, **kwargs: object) -> _LogitsOutput:
        if self.mode == "transformer":
            return cast(_LogitsOutput, self.model(**kwargs))
        else:
            x = cast(torch.Tensor, kwargs["features"])
            logits = cast(torch.Tensor, self.model(x))
            return cast(_LogitsOutput, cast(object, SimpleNamespace(logits=logits)))

    # -------- TRAIN --------
    def training_step(
        self,
        batch: dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        if self.mode == "transformer":
            assert isinstance(batch, dict)
            outputs = self.forward(
                **{
                    k: v
                    for k, v in batch.items()
                    if k in ("input_ids", "attention_mask", "token_type_ids")
                }
            )
            logits = outputs.logits
            labels = batch["labels"]
        else:
            assert isinstance(batch, tuple)
            features, labels = batch
            outputs = self.forward(features=features)
            logits = outputs.logits

        loss = cast(torch.Tensor, self.loss_fn(logits, labels))
        preds = torch.argmax(logits, dim=-1)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=labels.size(0),
        )
        self.train_acc(preds, labels)
        self.train_f1_micro(preds, labels)
        self.train_f1_macro(preds, labels)
        return loss

    def on_train_epoch_end(self) -> None:
        train_acc = cast(torch.Tensor, self.train_acc.compute())
        train_f1_micro = cast(torch.Tensor, self.train_f1_micro.compute())
        train_f1_macro = cast(torch.Tensor, self.train_f1_macro.compute())
        self.log("train_acc", train_acc, prog_bar=True)
        self.log("train_f1_micro", train_f1_micro, prog_bar=True)
        self.log("train_f1_macro", train_f1_macro, prog_bar=True)
        self.train_acc.reset()
        self.train_f1_micro.reset()
        self.train_f1_macro.reset()

    # -------- VAL --------
    def validation_step(
        self,
        batch: dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        if self.mode == "transformer":
            assert isinstance(batch, dict)
            outputs = self.forward(
                **{
                    k: v
                    for k, v in batch.items()
                    if k in ("input_ids", "attention_mask", "token_type_ids")
                }
            )
            logits = outputs.logits
            labels = batch["labels"]
        else:
            assert isinstance(batch, tuple)
            features, labels = batch
            outputs = self.forward(features=features)
            logits = outputs.logits

        loss = cast(torch.Tensor, self.loss_fn(logits, labels))
        preds = torch.argmax(logits, dim=-1)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=labels.size(0),
        )
        self.val_acc(preds, labels)
        self.val_f1_micro(preds, labels)
        self.val_f1_macro(preds, labels)
        return loss

    def on_validation_epoch_end(self) -> None:
        val_acc = cast(torch.Tensor, self.val_acc.compute())
        val_f1_micro = cast(torch.Tensor, self.val_f1_micro.compute())
        val_f1_macro = cast(torch.Tensor, self.val_f1_macro.compute())
        self.log("val_acc", val_acc, prog_bar=True)
        self.log("val_f1_micro", val_f1_micro, prog_bar=True)
        self.log("val_f1_macro", val_f1_macro, prog_bar=True)
        self.val_acc.reset()
        self.val_f1_micro.reset()
        self.val_f1_macro.reset()

    # -------- OPTIM --------
    def configure_optimizers(self) -> tuple[list[Optimizer], list[LRSchedulerConfig]]:
        if self.mode == "transformer":
            hparams = cast(object, self.hparams)
            learning_rate = float(getattr(hparams, "learning_rate", 3e-5))
            weight_decay = float(getattr(hparams, "weight_decay", 0.01))
            warmup_steps_pct = float(getattr(hparams, "warmup_steps_pct", 0.1))
            no_decay = ["bias", "LayerNorm.weight"]
            params = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer: Optimizer = torch.optim.AdamW(params, lr=learning_rate)
            total_steps = self.trainer.estimated_stepping_batches
            warmup_steps = int(warmup_steps_pct * total_steps)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
            scheduler_dict = cast(
                LRSchedulerConfig, cast(object, {"scheduler": scheduler, "interval": "step"})
            )
            return [optimizer], [scheduler_dict]
        else:
            hparams = cast(object, self.hparams)
            learning_rate = float(getattr(hparams, "learning_rate", 1e-3))
            weight_decay = float(getattr(hparams, "weight_decay", 0.0))
            optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
            return [optimizer], []
