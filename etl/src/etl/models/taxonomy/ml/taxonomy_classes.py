"""
Taxonomy (section type) classification models and datasets.

This mirrors the structure of the NER modules but focuses on strict
multi-label classification of sections. Supports two modes:
  - transformer: ModernBERT (or any HF encoder) via sequence classification
  - tfidf: TF-IDF vectorization with a small MLP classifier
"""
# pyright: reportUnknownMemberType=false

# Standard library
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Protocol, cast

# Environment config
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Third-party
import lightning.pytorch as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from lightning.pytorch.utilities.types import LRSchedulerConfig
from torchmetrics.classification import F1Score as F1
from torchmetrics.classification import MultilabelExactMatch, MultilabelRecall
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.optimization import get_linear_schedule_with_warmup  # pyright: ignore[reportUnknownVariableType]

if __package__ in (None, ""):
    src_root = Path(__file__).resolve().parents[3]
    if str(src_root) not in sys.path:
        sys.path.append(str(src_root))
    from etl.models.taxonomy.taxonomy_text import (  # type: ignore[reportMissingImports]
        combine_taxonomy_text,
    )
else:
    from .taxonomy_text import combine_taxonomy_text


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
        labels: list[list[int]],
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
        return combine_taxonomy_text(a, s, t)

    def __getitem__(self, idx: int) -> dict[str, object]:
        txt = self._combine(self.article_title[idx], self.section_title[idx], self.section_text[idx])
        y = self.labels[idx]

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
        train_rows: dict[str, list[str] | list[list[int]]],
        val_rows: dict[str, list[str] | list[list[int]]],
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
            labels=[list(cast(list[int], x)) for x in self.train_rows["label_vectors"]],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            mode="transformer",
        )
        self.val_dataset = TextLabelDataset(
            article_title=[str(x) for x in self.val_rows["article_title"]],
            section_title=[str(x) for x in self.val_rows["section_title"]],
            section_text=[str(x) for x in self.val_rows["section_text"]],
            labels=[list(cast(list[int], x)) for x in self.val_rows["label_vectors"]],
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


class _SparseTfidfDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, X: csr_matrix, y: np.ndarray) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row_sparse = cast(csr_matrix, self.X.getrow(idx))
        row_dense = cast(
            NDArray[np.float32],
            row_sparse.toarray().astype(np.float32, copy=False).ravel(),
        )
        features = torch.tensor(row_dense, dtype=torch.float32)
        labels = torch.tensor(self.y[idx], dtype=torch.float32)
        return features, labels


class TfidfDataModule(pl.LightningDataModule):
    """
    DataModule for TF-IDF-based taxonomy classification. Expects pre-transformed
    sparse CSR features and multi-hot float32 labels.
    """

    def __init__(
        self,
        X_train: csr_matrix,
        y_train: np.ndarray,
        X_val: csr_matrix,
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
        self.train_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]] | None = None
        self.val_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]] | None = None

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = _SparseTfidfDataset(self.X_train, self.y_train)
        self.val_dataset = _SparseTfidfDataset(self.X_val, self.y_val)

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
    LightningModule for multi-label taxonomy classification.
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
        decision_threshold: float = 0.5,
        pos_weight: list[float] | None = None,
        model_score_macro_weight: float = 0.45,
        model_score_weighted_weight: float = 0.25,
        model_score_micro_weight: float = 0.20,
        model_score_subset_weight: float = 0.10,
        model_score_recall_floor: float = 0.70,
        model_score_recall_penalty: float = 0.20,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.mode = mode
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.model: torch.nn.Module
        self.tfidf_residual_head: torch.nn.Linear | None = None
        self.decision_threshold = decision_threshold
        self.model_score_macro_weight = float(model_score_macro_weight)
        self.model_score_weighted_weight = float(model_score_weighted_weight)
        self.model_score_micro_weight = float(model_score_micro_weight)
        self.model_score_subset_weight = float(model_score_subset_weight)
        self.model_score_recall_floor = float(model_score_recall_floor)
        self.model_score_recall_penalty = float(model_score_recall_penalty)
        self._validate_model_score_config()

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
                    problem_type="multi_label_classification",
                ),
            )
        else:
            assert input_dim is not None, "input_dim must be provided for tfidf mode"
            hidden_dim_inner = max(128, hidden_dim // 2)
            layers = [
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, hidden_dim_inner),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim_inner, self.num_labels),
            ]
            self.model = torch.nn.Sequential(*layers)
            self.tfidf_residual_head = torch.nn.Linear(input_dim, self.num_labels)

        loss_pos_weight = None
        if pos_weight is not None:
            loss_pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=loss_pos_weight)

        # Metrics
        self.train_f1_micro = F1(
            task="multilabel",
            num_labels=self.num_labels,
            average="micro",
            threshold=self.decision_threshold,
        )
        self.train_f1_macro = F1(
            task="multilabel",
            num_labels=self.num_labels,
            average="macro",
            threshold=self.decision_threshold,
        )
        self.train_f1_weighted = F1(
            task="multilabel",
            num_labels=self.num_labels,
            average="weighted",
            threshold=self.decision_threshold,
        )

        self.val_f1_micro = F1(
            task="multilabel",
            num_labels=self.num_labels,
            average="micro",
            threshold=self.decision_threshold,
        )
        self.val_f1_macro = F1(
            task="multilabel",
            num_labels=self.num_labels,
            average="macro",
            threshold=self.decision_threshold,
        )
        self.val_f1_weighted = F1(
            task="multilabel",
            num_labels=self.num_labels,
            average="weighted",
            threshold=self.decision_threshold,
        )
        self.val_recall_micro = MultilabelRecall(
            num_labels=self.num_labels,
            average="micro",
            threshold=self.decision_threshold,
        )
        self.val_subset_accuracy = MultilabelExactMatch(
            num_labels=self.num_labels,
            threshold=self.decision_threshold,
        )

    def _validate_model_score_config(self) -> None:
        weights = [
            self.model_score_macro_weight,
            self.model_score_weighted_weight,
            self.model_score_micro_weight,
            self.model_score_subset_weight,
        ]
        if any(weight < 0.0 for weight in weights):
            raise ValueError("Model-score weights must all be non-negative.")
        if float(sum(weights)) <= 0.0:
            raise ValueError("At least one model-score weight must be positive.")
        if not 0.0 <= self.model_score_recall_floor <= 1.0:
            raise ValueError("`model_score_recall_floor` must be in [0, 1].")
        if self.model_score_recall_penalty < 0.0:
            raise ValueError("`model_score_recall_penalty` must be >= 0.")

    def _compose_model_score(
        self,
        f1_macro: torch.Tensor,
        f1_weighted: torch.Tensor,
        f1_micro: torch.Tensor,
        subset_accuracy: torch.Tensor,
        recall_micro: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight_total = (
            self.model_score_macro_weight
            + self.model_score_weighted_weight
            + self.model_score_micro_weight
            + self.model_score_subset_weight
        )
        macro_weight = self.model_score_macro_weight / weight_total
        weighted_weight = self.model_score_weighted_weight / weight_total
        micro_weight = self.model_score_micro_weight / weight_total
        subset_weight = self.model_score_subset_weight / weight_total
        base_score = (
            macro_weight * f1_macro
            + weighted_weight * f1_weighted
            + micro_weight * f1_micro
            + subset_weight * subset_accuracy
        )
        recall_floor_tensor = torch.tensor(
            self.model_score_recall_floor,
            dtype=recall_micro.dtype,
            device=recall_micro.device,
        )
        penalty_scale_tensor = torch.tensor(
            self.model_score_recall_penalty,
            dtype=recall_micro.dtype,
            device=recall_micro.device,
        )
        recall_gap = torch.clamp(recall_floor_tensor - recall_micro, min=0.0)
        penalty = penalty_scale_tensor * recall_gap
        score = base_score - penalty
        return score, base_score, penalty

    @staticmethod
    def _ensure_finite_tensor(name: str, tensor: torch.Tensor) -> None:
        finite_mask = torch.isfinite(tensor)
        if bool(finite_mask.all()):
            return
        total = int(tensor.numel())
        finite = int(finite_mask.sum().item())
        raise RuntimeError(
            f"Non-finite {name} detected ({total - finite}/{total} values)."
        )

    def forward(self, **kwargs: object) -> _LogitsOutput:
        if self.mode == "transformer":
            return cast(_LogitsOutput, self.model(**kwargs))
        else:
            x = cast(torch.Tensor, kwargs["features"])
            logits = cast(torch.Tensor, self.model(x))
            if self.tfidf_residual_head is not None:
                residual_logits = cast(torch.Tensor, self.tfidf_residual_head(x))
                logits = logits + residual_logits
            return cast(_LogitsOutput, cast(object, SimpleNamespace(logits=logits)))

    # -------- TRAIN --------
    def training_step(
        self,
        batch: dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
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
            assert isinstance(batch, (tuple, list))
            if len(batch) != 2:
                raise ValueError(
                    f"Expected TF-IDF batch of length 2, got {len(batch)}."
                )
            features, labels = batch
            outputs = self.forward(features=features)
            logits = outputs.logits

        labels_float = labels.to(dtype=torch.float32)
        labels_int = labels_float.to(dtype=torch.int)
        self._ensure_finite_tensor("train_logits", logits)
        probs = torch.sigmoid(logits)
        self._ensure_finite_tensor("train_probs", probs)
        loss = cast(torch.Tensor, self.loss_fn(logits, labels_float))
        self._ensure_finite_tensor("train_loss", loss)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=labels.size(0),
        )
        self.train_f1_micro(probs, labels_int)
        self.train_f1_macro(probs, labels_int)
        self.train_f1_weighted(probs, labels_int)
        return loss

    def on_train_epoch_end(self) -> None:
        train_f1_micro = cast(torch.Tensor, self.train_f1_micro.compute())
        train_f1_macro = cast(torch.Tensor, self.train_f1_macro.compute())
        train_f1_weighted = cast(torch.Tensor, self.train_f1_weighted.compute())
        train_model_score, train_model_score_raw, train_model_score_penalty = (
            self._compose_model_score(
                f1_macro=train_f1_macro,
                f1_weighted=train_f1_weighted,
                f1_micro=train_f1_micro,
                subset_accuracy=train_f1_micro,
                recall_micro=train_f1_micro,
            )
        )
        self.log("train_f1_micro", train_f1_micro, prog_bar=True)
        self.log("train_f1_macro", train_f1_macro, prog_bar=True)
        self.log("train_f1_weighted", train_f1_weighted, prog_bar=False)
        self.log("train_model_score", train_model_score, prog_bar=True)
        self.log("train_model_score_raw", train_model_score_raw, prog_bar=False)
        self.log(
            "train_model_score_recall_penalty",
            train_model_score_penalty,
            prog_bar=False,
        )
        self.train_f1_micro.reset()
        self.train_f1_macro.reset()
        self.train_f1_weighted.reset()

    # -------- VAL --------
    def validation_step(
        self,
        batch: dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
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
            assert isinstance(batch, (tuple, list))
            if len(batch) != 2:
                raise ValueError(
                    f"Expected TF-IDF batch of length 2, got {len(batch)}."
                )
            features, labels = batch
            outputs = self.forward(features=features)
            logits = outputs.logits

        labels_float = labels.to(dtype=torch.float32)
        labels_int = labels_float.to(dtype=torch.int)
        self._ensure_finite_tensor("val_logits", logits)
        probs = torch.sigmoid(logits)
        self._ensure_finite_tensor("val_probs", probs)
        loss = cast(torch.Tensor, self.loss_fn(logits, labels_float))
        self._ensure_finite_tensor("val_loss", loss)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=labels.size(0),
        )
        self.val_f1_micro(probs, labels_int)
        self.val_f1_macro(probs, labels_int)
        self.val_f1_weighted(probs, labels_int)
        self.val_recall_micro(probs, labels_int)
        self.val_subset_accuracy(probs, labels_int)
        return loss

    def on_validation_epoch_end(self) -> None:
        val_f1_micro = cast(torch.Tensor, self.val_f1_micro.compute())
        val_f1_macro = cast(torch.Tensor, self.val_f1_macro.compute())
        val_f1_weighted = cast(torch.Tensor, self.val_f1_weighted.compute())
        val_recall_micro = self.val_recall_micro.compute()
        val_subset_accuracy = self.val_subset_accuracy.compute()
        val_model_score, val_model_score_raw, val_model_score_penalty = (
            self._compose_model_score(
                f1_macro=val_f1_macro,
                f1_weighted=val_f1_weighted,
                f1_micro=val_f1_micro,
                subset_accuracy=val_subset_accuracy,
                recall_micro=val_recall_micro,
            )
        )
        self.log("val_f1_micro", val_f1_micro, prog_bar=True)
        self.log("val_f1_macro", val_f1_macro, prog_bar=True)
        self.log("val_f1_weighted", val_f1_weighted, prog_bar=True)
        self.log("val_recall_micro", val_recall_micro, prog_bar=True)
        self.log("val_subset_accuracy", val_subset_accuracy, prog_bar=False)
        self.log("val_model_score_raw", val_model_score_raw, prog_bar=False)
        self.log(
            "val_model_score_recall_penalty",
            val_model_score_penalty,
            prog_bar=False,
        )
        self.log("val_model_score", val_model_score, prog_bar=True)
        self.val_f1_micro.reset()
        self.val_f1_macro.reset()
        self.val_f1_weighted.reset()
        self.val_recall_micro.reset()
        self.val_subset_accuracy.reset()

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
