"""
NER (Named Entity Recognition) models and datasets.

This module contains PyTorch Lightning modules and datasets for NER tasks
using transformer models with BIO tagging scheme.
"""

# Standard library
import os
import re
from typing import Any, Dict, List, Tuple, cast, Optional
from torch.optim import Optimizer
from lightning.pytorch.utilities.types import LRSchedulerConfig
import numpy as np

# Environment config
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ML frameworks & utilities
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics import F1Score as F1, Precision, Recall

torch.set_float32_matmul_precision("high")

# Transformers
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForTokenClassification
from transformers.optimization import get_linear_schedule_with_warmup

from shared_constants import SPECIAL_TOKENS_TO_ADD


def prep_data(raw: str) -> Tuple[List[Tuple[int, int, str]], str, List[str]]:
    """
    Prepare raw tagged text for NER training.

    Extracts spans, cleans text, and creates character-level BIOES labels.

    Args:
        raw: Raw text with XML-style tags

    Returns:
        Tuple of (mapped_spans, cleaned_text, char_labels)
    """
    tag_pattern = re.compile(r"<(section|article|page)>(.*?)</\1>", re.DOTALL)

    # Extract raw spans
    spans: List[Tuple[int, int, str]] = []
    for match in tag_pattern.finditer(raw):
        spans.append((match.start(2), match.end(2), match.group(1)))

    # Strip out tags, build cleaned_text & orig2clean map
    cleaned_chars, orig2clean = [], {}
    i_clean = i = 0
    while i < len(raw):
        if raw.startswith("<section>", i):
            i += len("<section>")
        elif raw.startswith("</section>", i):
            i += len("</section>")
        elif raw.startswith("<article>", i):
            i += len("<article>")
        elif raw.startswith("</article>", i):
            i += len("</article>")
        elif raw.startswith("<page>", i):
            i += len("<page>")
        elif raw.startswith("</page>", i):
            i += len("</page>")
        else:
            cleaned_chars.append(raw[i])
            orig2clean[i] = i_clean
            i_clean += 1
            i += 1

    cleaned_text = "".join(cleaned_chars)

    # Map spans into cleaned-text coordinates
    mapped_spans: List[Tuple[int, int, str]] = []
    for start, end, tag in spans:
        c_start = orig2clean.get(start)
        c_end = orig2clean.get(end - 1)
        if c_start is not None and c_end is not None:
            mapped_spans.append((c_start, c_end + 1, tag))

    # Build per-char labels using BIOES scheme
    char_labels = ["O"] * len(cleaned_text)
    for c_start, c_end, tag in mapped_spans:
        span_len = c_end - c_start
        if span_len == 1:
            # S- for single-token entities
            char_labels[c_start] = f"S-{tag.upper()}"
        else:
            # B- for begin, E- for end
            char_labels[c_start] = f"B-{tag.upper()}"
            char_labels[c_end - 1] = f"E-{tag.upper()}"
            # I- for tokens inside the entity
            for pos in range(c_start + 1, c_end - 1):
                char_labels[pos] = f"I-{tag.upper()}"

    return mapped_spans, cleaned_text, char_labels


class TrainDataset(Dataset):
    """
    Training dataset for NER with sub-sampling strategies.

    Implements entity-centered sub-sampling and negative sample sub-sampling
    to handle class imbalance and long sequences.
    """

    def __init__(
        self,
        data: List[str],
        tokenizer: PreTrainedTokenizerBase,
        label2id: Dict[str, int],
        subsample_window: int,
    ):
        """
        Initialize the training dataset.

        Args:
            data: List of raw tagged strings
            tokenizer: HuggingFace tokenizer
            label2id: Label to ID mapping
            subsample_window: Window size for sub-sampling
        """
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.subsample_window = subsample_window
        self.examples = []

        for raw in data:
            mapped_spans, cleaned_text, char_labels = prep_data(raw)

            # Subsample 1: Negative samples (max 3 windows)
            if not mapped_spans:
                window_size = self.subsample_window
                text_length = len(cleaned_text)
                for i, window_start in enumerate(range(0, text_length, window_size)):
                    if i > 2:  # Limit to 3 windows
                        break

                    window_end = min(window_start + window_size, text_length)
                    chunk_text = cleaned_text[window_start:window_end]
                    chunk_labels = char_labels[window_start:window_end]
                    self._tokenize_and_store(chunk_text, chunk_labels)

            else:
                # Subsample 2: Entity-centered sub-sampling (positive samples)
                window_size = self.subsample_window
                text_length = len(cleaned_text)
                for span_start, span_end, _ in mapped_spans:
                    span_length = span_end - span_start

                    if span_length >= window_size:
                        window_start = max(0, span_start)
                    else:
                        pad = (window_size - span_length) // 2
                        window_start = max(
                            0, min(span_start - pad, text_length - window_size)
                        )

                    window_end = window_start + window_size
                    sub_text = cleaned_text[window_start:window_end]
                    sub_char_labels = char_labels[window_start:window_end]
                    self._tokenize_and_store(sub_text, sub_char_labels)

                # Subsample 3: Broken-span sliding windows (boundary cases)
                half_window = self.subsample_window // 2
                # For each span, if it's safely away from edges, snip out the two halves
                for span_start, span_end, _ in mapped_spans:
                    if (
                        span_start >= half_window
                        and span_end <= len(cleaned_text) - half_window
                    ):
                        # Reduce frequency of broken-span samples
                        if np.random.rand() > 0.05:
                            continue

                        mid = (span_start + span_end) // 2

                        # Left half: [mid-half_window, mid)
                        window_start1, window_end1 = mid - half_window, mid
                        self._tokenize_and_store(
                            cleaned_text[window_start1:window_end1],
                            char_labels[window_start1:window_end1],
                        )

                        # Right half: [mid, mid+half_window)
                        window_start2, window_end2 = mid, mid + half_window
                        self._tokenize_and_store(
                            cleaned_text[window_start2:window_end2],
                            char_labels[window_start2:window_end2],
                        )

    def _tokenize_and_store(self, text: str, char_labels: List[str]) -> None:
        """
        Tokenize text and store example with proper label alignment.

        Args:
            text: Text to tokenize
            char_labels: Character-level labels
        """
        # Run the tokenizer
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.subsample_window + 2,  # Accounts for [CLS] + [SEP]
        )
        offsets = encoding.pop("offset_mapping")

        # Pull out the word_ids
        word_ids = encoding.word_ids()  # Gives you a list of ints or None

        labels: List[int] = []
        last_wid = None
        for (off_start, off_end), wid in zip(offsets, word_ids):
            # Special token (CLS, SEP, PAD, etc.) → ignore
            if wid is None:
                labels.append(-100)
                last_wid = None
                continue

            # Not the first sub-token of this word → ignore
            if wid == last_wid:
                labels.append(-100)
            else:
                # First sub-token → assign a real label
                span = char_labels[off_start:off_end]
                entity = next((label for label in span if label != "O"), None)
                labels.append(self.label2id[entity] if entity else self.label2id["O"])

            last_wid = wid

        # Store the example
        self.examples.append(
            {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": labels,
            }
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


class ValWindowedDataset(Dataset):
    """
    Sliding-window NER dataset for validation.

    Retains per-window metadata and delegates batching to an external DataCollator.
    """

    def __init__(
        self,
        data: List[str],
        tokenizer: PreTrainedTokenizerBase,
        label2id: Dict[str, int],
        data_collator: DataCollatorForTokenClassification,
        window: int = 512,
        stride: int = 256,
    ):
        """
        Initialize the validation dataset.

        Args:
            data: List of raw tagged strings
            tokenizer: HuggingFace tokenizer
            label2id: Label to ID mapping
            data_collator: Collator for padding/batching token features
            window: Window size in characters
            stride: Stride size in characters
        """
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.collator = data_collator
        self.window = window
        self.stride = stride
        self.examples: List[Dict[str, Any]] = []

        for doc_id, raw in enumerate(self.data):
            _, cleaned_text, char_labels = prep_data(raw)

            # Sliding windows
            text_length = len(cleaned_text)
            start = 0
            while start < text_length:
                end = min(start + self.window, text_length)
                sub_text = cleaned_text[start:end]
                sub_labels = char_labels[start:end]

                encoding = self.tokenizer(
                    sub_text,
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=self.window + 2,
                )
                offsets = encoding.pop("offset_mapping")
                word_ids = encoding.word_ids()

                # Token-to-label mapping
                labels: List[int] = []
                last_wid = None
                for (o0, o1), wid in zip(offsets, word_ids):
                    if wid is None:
                        labels.append(-100)
                        last_wid = None
                    elif wid == last_wid:
                        labels.append(-100)
                    else:
                        span_entity = next(
                            (label for label in sub_labels[o0:o1] if label != "O"), None
                        )
                        labels.append(
                            self.label2id[span_entity]
                            if span_entity
                            else self.label2id["O"]
                        )
                        last_wid = wid

                # Store example with metadata
                self.examples.append(
                    {
                        "doc_id": doc_id,
                        "window_start": start,
                        "input_ids": encoding["input_ids"],
                        "attention_mask": encoding["attention_mask"],
                        "labels": labels,
                        "offset_mapping": offsets,
                        "raw": cleaned_text,
                    }
                )

                if end == text_length:
                    break
                start += self.stride

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]

    def collate_fn(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of windowed examples.

        Args:
            features: List of feature dictionaries

        Returns:
            Batched features with metadata
        """
        # Extract metadata
        doc_ids = [f.pop("doc_id") for f in features]
        starts = [f.pop("window_start") for f in features]
        offsets = [f.pop("offset_mapping") for f in features]
        raw = [f.pop("raw") for f in features]

        # Batch token inputs
        batch = self.collator(features)

        # Attach metadata
        batch["doc_id"] = torch.tensor(doc_ids, dtype=torch.long)
        batch["window_start"] = torch.tensor(starts, dtype=torch.long)
        batch["offset_mapping"] = offsets
        batch["raw"] = raw

        return batch


class NERDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for NER task.

    Handles train/val dataset creation and dataloaders.
    """

    def __init__(
        self,
        train_data: List[str],
        val_data: List[str],
        tokenizer_name: str,
        label_list: List[str],
        batch_size: int,
        train_subsample_window: int,
        num_workers: int,
        val_window: int = 512,
        val_stride: int = 256,
    ):
        """
        Initialize the NER data module.

        Args:
            train_data: Training data strings
            val_data: Validation data strings
            tokenizer_name: HuggingFace tokenizer name
            label_list: List of label names
            batch_size: Batch size for training
            train_subsample_window: Window size for training sub-sampling
            num_workers: Number of data loading workers
            val_window: Window size for validation
            val_stride: Stride size for validation
        """
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS_TO_ADD}
        )

        self.train_subsample_window = train_subsample_window
        self.val_window = val_window
        self.val_stride = val_stride

        self.label2id = {label: idx for idx, label in enumerate(label_list)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer, label_pad_token_id=-100
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets for training and validation.

        Args:
            stage: Lightning stage
        """
        self.train_dataset = TrainDataset(
            data=self.train_data,
            tokenizer=self.tokenizer,
            label2id=self.label2id,
            subsample_window=self.train_subsample_window,
        )
        self.val_dataset = ValWindowedDataset(
            data=self.val_data,
            tokenizer=self.tokenizer,
            label2id=self.label2id,
            data_collator=self.data_collator,
            window=self.val_window,
            stride=self.val_stride,
        )

    def train_dataloader(self) -> DataLoader:
        """Returns DataLoader for training set."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns DataLoader for validation set."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate_fn,
        )


class FocalLoss(torch.nn.Module):
    """
    Focal loss for class imbalance in NER.

    Addresses class imbalance by down-weighting easy examples.
    """

    def __init__(self, gamma: float = 2.0, ignore_index: int = -100):
        """
        Initialize focal loss.

        Args:
            gamma: Focusing parameter
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        self.gamma = gamma
        self.ignore = ignore_index
        self.ce = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model logits (B, T, C) or (N, C)
            labels: Target labels (B, T) or (N,)

        Returns:
            Scalar loss value
        """
        if logits.ndim == 3:
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)

        loss = self.ce(logits, labels)
        pt = torch.exp(-loss)
        focal = (1 - pt) ** self.gamma * loss
        return focal.mean()


class NERTagger(pl.LightningModule):
    """
    PyTorch LightningModule for NER model.

    Handles training, validation, and metrics computation.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        id2label: Dict[int, str],
        learning_rate: float,
        weight_decay: float,
        warmup_steps_pct: float,
        window: int = 512,
        stride: int = 256,
    ):
        """
        Initialize the NER tagger.

        Args:
            model_name: HuggingFace model name
            num_labels: Number of label classes
            id2label: ID to label mapping
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps_pct: Percentage of steps for warmup
            window: Window size for validation
            stride: Stride size for validation
        """
        super().__init__()
        self.save_hyperparameters()
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = {v: k for k, v in self.id2label.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS_TO_ADD}
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.train()

        self.loss_fn = FocalLoss()

        # Validation buffers
        self._val_sum = {}
        self._val_count = {}
        self._val_raw_texts = {}

        self.ignore_index = -100

        # Metrics (micro-average to avoid O-class dominance)
        self.train_f1 = F1(task="multiclass", num_classes=num_labels, average="micro")
        self.train_precision = Precision(
            task="multiclass", num_classes=num_labels, average="micro"
        )
        self.train_recall = Recall(
            task="multiclass", num_classes=num_labels, average="micro"
        )

        self.val_f1 = F1(task="multiclass", num_classes=num_labels, average="micro")
        self.val_precision = Precision(
            task="multiclass", num_classes=num_labels, average="micro"
        )
        self.val_recall = Recall(
            task="multiclass", num_classes=num_labels, average="micro"
        )

        self.val_f1_doc = F1(task="multiclass", num_classes=num_labels, average="micro")
        self.val_precision_doc = Precision(
            task="multiclass", num_classes=num_labels, average="micro"
        )
        self.val_recall_doc = Recall(
            task="multiclass", num_classes=num_labels, average="micro"
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        """
        Forward pass for NER model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Model outputs
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step for NER model.

        Args:
            batch: Input batch
            batch_idx: Batch index

        Returns:
            Loss value
        """
        outputs = self(
            batch["input_ids"],
            batch["attention_mask"],
        )
        logits = outputs.logits
        loss = self.loss_fn(
            logits.view(-1, self.num_labels),
            batch["labels"].view(-1),
        )
        labels = batch["labels"]
        preds = torch.argmax(logits, dim=-1)
        mask = labels != -100
        valid_preds = preds[mask]
        valid_labels = labels[mask]

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch["input_ids"].shape[0],
        )

        self.train_f1(valid_preds, valid_labels)
        self.train_precision(valid_preds, valid_labels)
        self.train_recall(valid_preds, valid_labels)

        return loss

    def on_train_epoch_end(self) -> None:
        """Log and reset training metrics at epoch end."""
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        self.log("train_precision", self.train_precision.compute())
        self.log("train_recall", self.train_recall.compute())

        self.train_f1.reset()
        self.train_precision.reset()
        self.train_recall.reset()

    def on_validation_epoch_start(self) -> None:
        """Clear validation buffers at epoch start."""
        self._val_sum.clear()
        self._val_count.clear()
        self._val_raw_texts.clear()

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Compute window-level loss & metrics and buffer for document-level stitching.

        Args:
            batch: Input batch with metadata
            batch_idx: Batch index

        Returns:
            Loss value
        """
        outputs = self(
            batch["input_ids"],
            batch["attention_mask"],
        )
        logits = outputs.logits
        loss = self.loss_fn(
            logits.view(-1, self.num_labels),
            batch["labels"].view(-1),
        )
        labels = batch["labels"]

        # Window-level metrics
        preds = torch.argmax(logits, dim=-1)
        mask = labels != -100
        self.val_f1(preds[mask], labels[mask])
        self.val_precision(preds[mask], labels[mask])
        self.val_recall(preds[mask], labels[mask])
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch["input_ids"].shape[0],
        )

        # Buffer for document-level stitching
        for i in range(logits.size(0)):
            doc_id = int(batch["doc_id"][i])
            window_start = int(batch["window_start"][i])
            offsets = batch["offset_mapping"][i]  # List of (o0, o1)
            attention_mask = batch["attention_mask"][i]
            logits_window = logits[i]

            # Store raw text once
            if doc_id not in self._val_raw_texts:
                self._val_raw_texts[doc_id] = batch["raw"][i]

            # Figure how big we need the buffers
            max_offset = max(o1 for o0, o1 in offsets)
            required_length = window_start + max_offset

            # Lazy-init or pad _val_sum, _val_count
            if doc_id not in self._val_sum:
                device = logits_window.device
                self._val_sum[doc_id] = torch.zeros(
                    (required_length, self.num_labels), device=device
                )
                self._val_count[doc_id] = torch.zeros(required_length, device=device)
            elif required_length > self._val_sum[doc_id].size(0):
                old_size = self._val_sum[doc_id].size(0)
                pad_size = required_length - old_size
                device = logits_window.device
                self._val_sum[doc_id] = torch.cat(
                    [
                        self._val_sum[doc_id],
                        torch.zeros((pad_size, self.num_labels), device=device),
                    ],
                    dim=0,
                )
                self._val_count[doc_id] = torch.cat(
                    [self._val_count[doc_id], torch.zeros(pad_size, device=device)],
                    dim=0,
                )

            # Accumulate token logits into per-char buckets
            for token_idx, (offset_start, offset_end) in enumerate(offsets):
                if attention_mask[token_idx] == 0:
                    continue
                self._val_sum[doc_id][
                    window_start + offset_start : window_start + offset_end
                ] += logits_window[token_idx]
                self._val_count[doc_id][
                    window_start + offset_start : window_start + offset_end
                ] += 1

        return loss

    def _clean_and_label(self, raw: str) -> Tuple[str, List[int]]:
        """
        Replicate windowing-clean logic to get cleaned_text and char-level label-ids.

        Args:
            raw: Raw text with tags

        Returns:
            Tuple of (cleaned_text, label_ids)
        """
        _, cleaned_text, char_labels = prep_data(raw)

        # Map to IDs
        return cleaned_text, [
            (self.label2id.get(label, -100) if label != "O" else self.label2id["O"])
            for label in char_labels
        ]

    def on_validation_epoch_end(self) -> None:
        """Compute and log document-level metrics at epoch end."""
        # Window-level metrics
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.log("val_precision", self.val_precision.compute())
        self.log("val_recall", self.val_recall.compute())
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()

        all_preds, all_gold = [], []
        device = next(self.model.parameters()).device

        # For each doc, average logits→char preds, load gold, mask & collect
        for doc_id, sum_logits in self._val_sum.items():
            counts = self._val_count[doc_id].clamp(min=1.0).unsqueeze(-1)
            avg_logits = sum_logits / counts  # (T, C)
            char_preds = avg_logits.argmax(dim=-1).tolist()

            raw = self._val_raw_texts[doc_id]
            _, gold_ids = self._clean_and_label(raw)  # List[int], len T

            for pred, gold in zip(char_preds, gold_ids):
                if gold != self.ignore_index:
                    all_preds.append(pred)
                    all_gold.append(gold)

        # Mask and tensorize
        mask = [gold != self.ignore_index for gold in all_gold]
        filtered_preds = [pred for pred, mask_val in zip(all_preds, mask) if mask_val]
        filtered_gold = [gold for gold, mask_val in zip(all_gold, mask) if mask_val]
        filtered_preds_tensor = torch.tensor(
            filtered_preds, dtype=torch.long, device=device
        )
        filtered_gold_tensor = torch.tensor(
            filtered_gold, dtype=torch.long, device=device
        )

        # Document-level metrics
        self.val_f1_doc(filtered_preds_tensor, filtered_gold_tensor)
        self.val_precision_doc(filtered_preds_tensor, filtered_gold_tensor)
        self.val_recall_doc(filtered_preds_tensor, filtered_gold_tensor)

        self.log("val_f1_doc", self.val_f1_doc.compute(), prog_bar=True)
        self.log("val_precision_doc", self.val_precision_doc.compute())
        self.log("val_recall_doc", self.val_recall_doc.compute())
        self.val_f1_doc.reset()
        self.val_precision_doc.reset()
        self.val_recall_doc.reset()

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRSchedulerConfig]]:
        """
        Configure AdamW optimizer and linear warmup scheduler.

        Returns:
            Tuple for Lightning compatibility
        """
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": getattr(self, "weight_decay", 0.01),
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer: Optimizer = torch.optim.AdamW(
            params, lr=getattr(self, "learning_rate", 3e-5)
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(getattr(self, "warmup_steps_pct", 0.1) * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        scheduler_dict = cast(
            LRSchedulerConfig,
            {
                "scheduler": scheduler,
                "interval": "step",
            },
        )

        return [optimizer], [scheduler_dict]
