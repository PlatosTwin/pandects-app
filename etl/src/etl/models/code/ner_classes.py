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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import F1Score as F1, Precision, Recall

torch.set_float32_matmul_precision("high")

# Transformers
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForTokenClassification
from transformers.optimization import get_linear_schedule_with_warmup

try:
    from .shared_constants import SPECIAL_TOKENS_TO_ADD
except ImportError:  # pragma: no cover - supports running as a script
    from shared_constants import (  # pyright: ignore[reportMissingImports]
        SPECIAL_TOKENS_TO_ADD,
    )


# ASCII-only, length-preserving lowercase (A-Z -> a-z)
_ASCII_LOWER_TBL = str.maketrans({chr(i): chr(i + 32) for i in range(65, 91)})


def ascii_lower(s: str) -> str:
    return s.translate(_ASCII_LOWER_TBL)


def _upgrade_token_head(
    model, num_labels: int, p_drop: float = 0.1, hidden_mult: float = 1.0
):
    # Works for most HF token classifiers (including ModernBERT variants)
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        in_dim = model.classifier.in_features
        mid = int(in_dim * hidden_mult)
        model.classifier = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(in_dim, mid),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(mid, num_labels),
        )
    return model


def _process_document(
    raw: str, tokenizer: PreTrainedTokenizerBase, label2id: Dict[str, int]
) -> Dict[str, Any]:
    tag_pattern = re.compile(r"<(section|article|page)>(.*?)</\1>", re.DOTALL)

    # 1) Build cleaned_text while tracking spans in cleaned-text coordinates
    parts: List[str] = []
    spans: List[Tuple[int, int, str]] = []  # (start_in_cleaned, end_in_cleaned, tag)
    src_pos = 0
    out_len = 0

    for m in tag_pattern.finditer(raw):
        # text before tag
        pre = raw[src_pos : m.start()]
        parts.append(pre)
        out_len += len(pre)

        tag = m.group(1)
        content = m.group(2)

        # record span in cleaned-text coords
        span_start = out_len
        parts.append(content)
        out_len += len(content)
        span_end = out_len
        spans.append((span_start, span_end, tag))

        src_pos = m.end()

    # tail after last tag
    parts.append(raw[src_pos:])
    cleaned_text = "".join(parts)

    # 2) Tokenize without specials using ASCII-only lowercase mirror; get offsets & word_ids
    norm = ascii_lower(cleaned_text)
    encoding = tokenizer(
        norm,
        return_offsets_mapping=True,
        truncation=False,
        add_special_tokens=False,
    )
    offsets = encoding["offset_mapping"]
    word_ids = encoding.word_ids()

    # 3) Locate first token of each word (for word-level labeling & later span logic)
    first_token_idx: List[int] = []
    seen_words = set()
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid not in seen_words:
            first_token_idx.append(i)
            seen_words.add(wid)

    # 4) Initialize labels to 'O'
    o_id = label2id["O"]
    labels = [o_id] * len(offsets)

    # 5) Apply BIOES to the FIRST token of each word overlapping each char span
    for c0, c1, tag in spans:
        wid_to_first_tok: Dict[int, int] = {}
        for i, wid in enumerate(word_ids):
            if wid is None:
                continue
            t0, t1 = offsets[i]
            if max(t0, c0) < min(t1, c1):  # overlap
                if wid not in wid_to_first_tok:
                    wid_to_first_tok[wid] = i

        if not wid_to_first_tok:
            continue

        ordered_first_toks = sorted(wid_to_first_tok.values())
        if len(ordered_first_toks) == 1:
            labels[ordered_first_toks[0]] = label2id[f"S-{tag.upper()}"]
        else:
            labels[ordered_first_toks[0]] = label2id[f"B-{tag.upper()}"]
            labels[ordered_first_toks[-1]] = label2id[f"E-{tag.upper()}"]
            for idx in ordered_first_toks[1:-1]:
                labels[idx] = label2id[f"I-{tag.upper()}"]

    # 6) Mask non-first-subword tokens as -100; keep track of first-token indices
    final_labels: List[int] = []
    last_wid = None
    for i, wid in enumerate(word_ids):
        if wid is None:
            final_labels.append(-100)
        elif wid != last_wid:
            final_labels.append(labels[i])
        else:
            final_labels.append(-100)
        last_wid = wid

    encoding["labels"] = final_labels
    encoding["cleaned_text"] = cleaned_text
    encoding["first_token_idx"] = first_token_idx
    return encoding


def repair_bioes(tags: List[str]) -> List[str]:
    """
    Make a best-effort repair of illegal BIOES sequences.
    - Lone I-* without a preceding B-*: -> B-*
    - E-* without open entity: -> S-*
    - B-* followed directly by O or B-*: -> S-* (close immediately)
    - I-* followed by O/B-*: -> E-* (close)
    """
    repaired = tags[:]
    open_type = None
    for i, t in enumerate(repaired):
        if t == "O":
            if open_type is not None:
                # close previous entity
                repaired[i - 1] = f"E-{open_type}"
                open_type = None
            continue

        pref = t[0]  # B/I/E/S
        typ = t[2:] if len(t) > 2 else ""
        if pref == "B":
            if open_type is not None:
                repaired[i - 1] = f"E-{open_type}"
            # peek next
            nxt = repaired[i + 1] if i + 1 < len(repaired) else "O"
            if nxt.startswith(("O", "B", "S")):
                repaired[i] = f"S-{typ}"
                open_type = None
            else:
                open_type = typ

        elif pref == "I":
            if open_type is None or typ != open_type:
                # convert to B
                repaired[i] = f"B-{typ}"
                # same logic as B for immediate close
                nxt = repaired[i + 1] if i + 1 < len(repaired) else "O"
                if nxt.startswith(("O", "B", "S")):
                    repaired[i] = f"S-{typ}"
                    open_type = None
                else:
                    open_type = typ
            else:
                # potentially close if next breaks
                nxt = repaired[i + 1] if i + 1 < len(repaired) else "O"
                if nxt.startswith(("O", "B", "S")):
                    repaired[i] = f"E-{typ}"
                    open_type = None

        elif pref == "E":
            if open_type is None or typ != open_type:
                repaired[i] = f"S-{typ}"
                open_type = None
            else:
                open_type = None

        elif pref == "S":
            if open_type is not None:
                repaired[i - 1] = f"E-{open_type}"
                open_type = None
            # S-* is already a closed single, nothing to do
        else:
            repaired[i] = "O"

    if open_type is not None:
        # close trailing open entity
        repaired[-1] = f"E-{open_type}"
    return repaired


def tags_to_spans(tags: List[str]) -> List[Tuple[int, int, str]]:
    """
    Convert BIOES tags to spans as (start_idx, end_idx, type), inclusive.
    Assumes tags have been repaired (legal BIOES).
    """
    spans = []
    cur_start, cur_type = None, None
    for i, t in enumerate(tags):
        if t == "O":
            continue
        pref = t[0]
        typ = t[2:]
        if pref == "S":
            spans.append((i, i, typ))
        elif pref == "B":
            cur_start, cur_type = i, typ
        elif pref == "I":
            # continuing
            pass
        elif pref == "E":
            if cur_start is None or cur_type != typ:
                # fallback: treat as single
                spans.append((i, i, typ))
            else:
                spans.append((cur_start, i, typ))
            cur_start, cur_type = None, None
    return spans


def prf1_from_spans(
    pred_spans: List[Tuple[int, int, str]], gold_spans: List[Tuple[int, int, str]]
) -> Tuple[int, int, int]:
    """
    Exact-match micro: returns (tp, fp, fn).
    """
    pred_set = set(pred_spans)
    gold_set = set(gold_spans)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


def _parse_bioes(tag: str) -> tuple[str, str | None]:
    # "O" -> ("O", None), "B-FOO" -> ("B","FOO"), etc.
    if tag == "O":
        return "O", None
    if "-" not in tag:
        return "O", None
    p, t = tag.split("-", 1)
    return p, t


def _legal_bioes(
    prev_tag: str, prev_type: str | None, curr_tag: str, curr_type: str | None
) -> bool:
    # BIOES legality with type agreement for inside segments
    if prev_tag == "O":
        return curr_tag in ("O", "B", "S")
    if prev_tag == "B":
        return (curr_tag, curr_type) in (("I", prev_type), ("E", prev_type))
    if prev_tag == "I":
        return (curr_tag, curr_type) in (("I", prev_type), ("E", prev_type))
    if prev_tag == "E":
        return curr_tag in ("O", "B", "S")
    if prev_tag == "S":
        return curr_tag in ("O", "B", "S")
    return curr_tag in ("O", "B", "S")  # fallback


def build_bioes_constraints(
    id2label: dict[int, str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (trans, start, end):
      trans: [C,C] add to emissions[t] when transitioning prev->curr
      start: [C]   add to emissions[0]
      end:   [C]   add at the last step
    Legal = 0.0, illegal = large negative (not -inf to avoid NaNs).
    """
    C = len(id2label)
    NEG = -1e4
    trans = torch.full((C, C), NEG)
    start = torch.full((C,), NEG)
    end = torch.full((C,), NEG)

    # pre-parse
    parts = {i: _parse_bioes(lbl) for i, lbl in id2label.items()}

    # transitions
    for i in range(C):
        pi, ti = parts[i]
        for j in range(C):
            pj, tj = parts[j]
            if _legal_bioes(pi, ti, pj, tj):
                trans[i, j] = 0.0

    # start: cannot start with I/E
    for j in range(C):
        pj, _ = parts[j]
        if pj in ("O", "B", "S"):
            start[j] = 0.0

    # end: should not end on B/I (open entity)
    for i in range(C):
        pi, _ = parts[i]
        if pi in ("O", "E", "S"):
            end[i] = 0.0

    return trans, start, end


class TrainDataset(Dataset):
    """
    Training dataset for NER with token-based sub-sampling strategies.
    This version includes the fix for the off-by-one error in span collection.
    """

    def __init__(
        self,
        data: List[str],
        tokenizer: PreTrainedTokenizerBase,
        label2id: Dict[str, int],
        subsample_window: int,
    ):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.subsample_window = subsample_window
        self.examples = []

        for raw in data:
            processed_doc = _process_document(raw, self.tokenizer, self.label2id)
            self._create_samples_from_doc(processed_doc)

    def _create_samples_from_doc(self, processed_doc: Dict[str, Any]):
        input_ids = processed_doc["input_ids"]
        labels = processed_doc["labels"]
        num_tokens = len(input_ids)
        o_id = self.label2id["O"]

        # 1) Indices of FIRST tokens of words (labels != -100 by construction)
        first_idxs = [i for i, lab in enumerate(labels) if lab != -100]

        # 2) Mark which first-tokens are entity words
        is_ent_word = [labels[i] != o_id for i in first_idxs]

        # 3) Find contiguous runs in *word space* (k indexes into first_idxs)
        entity_spans = []
        open_k = None
        for k, flag in enumerate(is_ent_word):
            if flag and open_k is None:
                open_k = k
            elif (not flag) and (open_k is not None):
                last_k = k - 1
                start_tok = first_idxs[open_k]
                # end token = last token before the next word's first token
                end_tok = (
                    (first_idxs[last_k + 1] - 1)
                    if (last_k + 1 < len(first_idxs))
                    else (num_tokens - 1)
                )
                entity_spans.append((start_tok, end_tok))
                open_k = None
        if open_k is not None:
            start_tok = first_idxs[open_k]
            end_tok = num_tokens - 1
            entity_spans.append((start_tok, end_tok))

        # 4) Subsampling (unchanged)
        if not entity_spans:
            for i, start in enumerate(range(0, num_tokens, self.subsample_window)):
                if i > 2:
                    break
                end = min(start + self.subsample_window, num_tokens)
                self._store_chunk(input_ids[start:end], labels[start:end])
        else:
            for start, end in entity_spans:
                span_len = end - start + 1
                if span_len >= self.subsample_window:
                    window_start = start
                else:
                    pad = (self.subsample_window - span_len) // 2
                    window_start = max(
                        0, min(start - pad, num_tokens - self.subsample_window)
                    )
                window_end = min(window_start + self.subsample_window, num_tokens)
                self._store_chunk(
                    input_ids[window_start:window_end], labels[window_start:window_end]
                )

            half = self.subsample_window // 2
            for start, end in entity_spans:
                if start >= half and end <= num_tokens - half:
                    if np.random.rand() > 0.05:  # keep your 5% sampling
                        continue
                    mid = (start + end) // 2
                    self._store_chunk(
                        input_ids[mid - half : mid], labels[mid - half : mid]
                    )
                    self._store_chunk(
                        input_ids[mid : mid + half], labels[mid : mid + half]
                    )

    def _store_chunk(self, id_chunk: List[int], label_chunk: List[int]):
        """Adds special tokens and stores a training example."""
        input_ids = (
            [self.tokenizer.cls_token_id] + id_chunk + [self.tokenizer.sep_token_id]
        )
        labels = [-100] + label_chunk + [-100]
        attention_mask = [1] * len(input_ids)

        self.examples.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


class ValWindowedDataset(Dataset):
    """
    Sliding-window NER dataset for validation, operating on tokens.
    """

    def __init__(
        self,
        data: List[str],
        tokenizer: PreTrainedTokenizerBase,
        label2id: Dict[str, int],
        data_collator: DataCollatorForTokenClassification,
        window: int = 510,
        stride: int = 256,
    ):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.collator = data_collator
        self.window = window
        self.stride = stride
        self.examples: List[Dict[str, Any]] = []

        for doc_id, raw in enumerate(data):
            processed_doc = _process_document(raw, self.tokenizer, self.label2id)

            input_ids = processed_doc["input_ids"]
            labels = processed_doc["labels"]
            offset_mapping = processed_doc["offset_mapping"]
            cleaned_text = processed_doc["cleaned_text"]
            num_tokens = len(input_ids)

            for start in range(0, num_tokens, self.stride):
                end = min(start + self.window, num_tokens)

                id_chunk = input_ids[start:end]
                label_chunk = labels[start:end]
                offset_chunk = offset_mapping[start:end]

                final_ids = (
                    [self.tokenizer.cls_token_id]
                    + id_chunk
                    + [self.tokenizer.sep_token_id]
                )
                final_labels = [-100] + label_chunk + [-100]
                final_offsets = [(0, 0)] + offset_chunk + [(0, 0)]

                self.examples.append(
                    {
                        "doc_id": doc_id,
                        "window_start": start,
                        "input_ids": final_ids,
                        "attention_mask": [1] * len(final_ids),
                        "labels": final_labels,
                        "offset_mapping": final_offsets,
                        "raw": cleaned_text,
                    }
                )

                if end == num_tokens:
                    break

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]

    def collate_fn(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        doc_ids = [f.pop("doc_id") for f in features]
        starts = [f.pop("window_start") for f in features]
        offsets = [f.pop("offset_mapping") for f in features]
        raw = [f.pop("raw") for f in features]

        batch = self.collator(features)

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
        val_window: int = 510,
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
    LightningModule for NER with token-level validation stitching (BIOES-aware).
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        id2label: Dict[int, str],
        learning_rate: float,
        weight_decay: float,
        warmup_steps_pct: float,
    ):
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
        _upgrade_token_head(self.model, self.num_labels, p_drop=0.1, hidden_mult=1.0)

        self.loss_fn = FocalLoss()
        self.ignore_index = -100

        # CRF-style constraints (buffers, move with .to(device), saved in state_dict)
        _trans, _start, _end = build_bioes_constraints(self.id2label)
        self.register_buffer("_crf_trans", _trans)  # [C,C]
        self.register_buffer("_crf_start", _start)  # [C]
        self.register_buffer("_crf_end", _end)  # [C]

        # --- Metrics (micro) ---
        o_id = self.label2id["O"]

        # Training
        self.train_f1 = F1(
            task="multiclass",
            num_classes=num_labels,
            average="micro",
        )
        self.train_f1_no_o = F1(
            task="multiclass",
            num_classes=num_labels,
            average="micro",
            ignore_index=o_id,
        )

        # Sample-level validation
        self.val_f1_no_o = F1(
            task="multiclass",
            num_classes=num_labels,
            average="micro",
            ignore_index=o_id,
        )

        # >>> Token-level stitching buffers (per doc)
        self._tok_sum: Dict[int, torch.Tensor] = {}  # [T_doc, C]
        self._tok_cnt: Dict[int, torch.Tensor] = {}  # [T_doc]
        self._tok_gold: Dict[int, torch.Tensor] = (
            {}
        )  # [T_doc] (label ids, -100 for unknown)
        self._doc_raw: Dict[int, str] = {}  # Optional: original/clean text

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    # ----------------- TRAIN -----------------
    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        outputs = self(batch["input_ids"], batch["attention_mask"])
        logits = outputs.logits
        loss = self.loss_fn(logits.view(-1, self.num_labels), batch["labels"].view(-1))

        labels = batch["labels"]
        preds = torch.argmax(logits, dim=-1)
        mask = labels != self.ignore_index

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch["input_ids"].shape[0],
        )

        if mask.any():
            self.train_f1(preds[mask], labels[mask])
            self.train_f1_no_o(preds[mask], labels[mask])

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        self.log("train_f1_no_o", self.train_f1_no_o.compute(), prog_bar=True)

        self.train_f1.reset()
        self.train_f1_no_o.reset()

    def _viterbi_constrained_doc(self, emissions: torch.Tensor) -> torch.Tensor:
        """
        emissions: [T, C] (already stitched/averaged), full doc tokens only
        returns:  [T] best label path enforcing BIOES legality
        """
        T, C = emissions.shape
        trans = self._crf_trans
        start = self._crf_start
        end = self._crf_end

        score = start + emissions[0]  # [C]
        backp = torch.zeros((T, C), dtype=torch.long, device=emissions.device)

        for t in range(1, T):
            prev = score.unsqueeze(1) + trans  # [C_prev, C_curr]
            best_prev, best_idx = prev.max(dim=0)  # [C], [C]
            score = best_prev + emissions[t]  # [C]
            backp[t] = best_idx

        score = score + end
        last = int(torch.argmax(score))
        path = torch.empty((T,), dtype=torch.long, device=emissions.device)
        path[-1] = last
        for t in range(T - 1, 0, -1):
            path[t - 1] = backp[t, path[t]]
        return path

    # ----------------- VAL -----------------
    def on_validation_epoch_start(self) -> None:
        # >>> Clear token-level buffers
        self._tok_sum.clear()
        self._tok_cnt.clear()
        self._tok_gold.clear()
        self._doc_raw.clear()

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        outputs = self(batch["input_ids"], batch["attention_mask"])
        logits = outputs.logits
        loss = self.loss_fn(logits.view(-1, self.num_labels), batch["labels"].view(-1))

        labels = batch["labels"]
        preds = torch.argmax(logits, dim=-1)
        mask = labels != self.ignore_index

        # Window-level metrics (micro)
        if mask.any():
            self.val_f1_no_o(preds[mask], labels[mask])

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch["input_ids"].shape[0],
        )

        # >>> Stitch at token level across overlapping windows
        # batch contains: doc_id [B], window_start [B], offset_mapping (list of lists), raw (list[str])
        B = logits.size(0)
        for i in range(B):
            doc_id = int(batch["doc_id"][i])
            window_start = int(
                batch["window_start"][i]
            )  # token start index in full doc
            offsets = batch["offset_mapping"][i]  # len == T_win, (0,0) at CLS/SEP
            attn = batch["attention_mask"][i]  # [T_win]
            logits_win = logits[i]  # [T_win, C]
            labels_win = labels[i]  # [T_win]

            # Store raw once (optional)
            if doc_id not in self._doc_raw:
                # NOTE: ValWindowedDataset currently stores cleaned_text; if you want original, change it there.
                self._doc_raw[doc_id] = batch["raw"][i]

            # Count real tokens in this window (exclude specials with (0,0) offsets)
            non_special_tok_count = 0
            for (o0, o1), a in zip(offsets, attn):
                if a == 0:
                    continue
                if o0 == 0 and o1 == 0:
                    continue
                non_special_tok_count += 1

            # Grow buffers
            required_len = window_start + non_special_tok_count
            if doc_id not in self._tok_sum:
                device = logits_win.device
                self._tok_sum[doc_id] = torch.zeros(
                    (required_len, self.num_labels), device=device
                )
                self._tok_cnt[doc_id] = torch.zeros(required_len, device=device)
                self._tok_gold[doc_id] = torch.full(
                    (required_len,), self.ignore_index, dtype=torch.long, device=device
                )
            elif required_len > self._tok_sum[doc_id].size(0):
                device = logits_win.device
                old = self._tok_sum[doc_id].size(0)
                pad = required_len - old
                self._tok_sum[doc_id] = torch.cat(
                    [
                        self._tok_sum[doc_id],
                        torch.zeros((pad, self.num_labels), device=device),
                    ],
                    dim=0,
                )
                self._tok_cnt[doc_id] = torch.cat(
                    [self._tok_cnt[doc_id], torch.zeros(pad, device=device)], dim=0
                )
                self._tok_gold[doc_id] = torch.cat(
                    [
                        self._tok_gold[doc_id],
                        torch.full(
                            (pad,), self.ignore_index, dtype=torch.long, device=device
                        ),
                    ],
                    dim=0,
                )

            # Walk the window tokens and map to absolute token indices in doc
            abs_tok = window_start
            for t_idx, ((o0, o1), a) in enumerate(zip(offsets, attn)):
                if a == 0:
                    continue
                if o0 == 0 and o1 == 0:
                    # CLS/SEP we added in dataset
                    continue

                # accumulate logits/count
                self._tok_sum[doc_id][abs_tok] += logits_win[t_idx]
                self._tok_cnt[doc_id][abs_tok] += 1

                # set gold once (windows agree, but be safe)
                lab = int(labels_win[t_idx].item())
                if (
                    lab != self.ignore_index
                    and self._tok_gold[doc_id][abs_tok].item() == self.ignore_index
                ):
                    self._tok_gold[doc_id][abs_tok] = lab

                abs_tok += 1

        return loss

    def on_validation_epoch_end(self) -> None:
        # Window-level
        self.log("val_f1_no_o", self.val_f1_no_o.compute(), prog_bar=True)

        self.val_f1_no_o.reset()

        # ---- Entity-level (stitched) ----
        tp = fp = fn = 0
        for doc_id, sum_logits in self._tok_sum.items():
            cnt = self._tok_cnt[doc_id].clamp(min=1.0).unsqueeze(-1)
            avg_logits = sum_logits / cnt
            # pred_ids = avg_logits.argmax(dim=-1)  # [T]
            pred_ids = self._viterbi_constrained_doc(avg_logits)  # [T]
            gold_ids = self._tok_gold[doc_id]  # [T]
            mask = gold_ids != self.ignore_index

            if not mask.any():
                continue

            # keep only first-subword positions (gold defined there)
            pred_ids = pred_ids[mask].tolist()
            gold_ids = gold_ids[mask].tolist()

            pred_tags = [self.id2label[i] for i in pred_ids]
            gold_tags = [self.id2label[i] for i in gold_ids]

            # optional repair on predictions
            pred_tags = repair_bioes(pred_tags)

            pred_spans = tags_to_spans(pred_tags)
            gold_spans = tags_to_spans(gold_tags)

            tpi, fpi, fni = prf1_from_spans(pred_spans, gold_spans)
            tp += tpi
            fp += fpi
            fn += fni

        if tp + fp + fn > 0:
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            self.log("val_ent_precision", prec)
            self.log("val_ent_recall", rec)
            self.log("val_ent_f1", f1, prog_bar=True)

    # ----------------- OPTIM -----------------
    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRSchedulerConfig]]:
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
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        scheduler_dict = cast(
            LRSchedulerConfig, {"scheduler": scheduler, "interval": "step"}
        )
        return [optimizer], [scheduler_dict]
