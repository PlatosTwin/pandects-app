# Standard library
import os
import re
from typing import Any, Dict, List, Tuple, cast
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


def prep_data(raw):
    tag_pattern = re.compile(r"<(section|article|page)>(.*?)</\1>", re.DOTALL)

    # 1. extract raw spans
    spans: List[Tuple[int, int, str]] = []
    for m in tag_pattern.finditer(raw):
        spans.append((m.start(2), m.end(2), m.group(1)))

    # 2. strip out tags, build cleaned_text & orig2clean map
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

    # 3. map spans into cleaned-text coords
    mapped_spans: List[Tuple[int, int, str]] = []
    for start, end, tag in spans:
        c_start = orig2clean.get(start)
        c_end = orig2clean.get(end - 1)
        if c_start is not None and c_end is not None:
            mapped_spans.append((c_start, c_end + 1, tag))

    # 4. build per-char labels
    char_labels = ["O"] * len(cleaned_text)
    for c_start, c_end, tag in mapped_spans:
        char_labels[c_start] = f"B-{tag.upper()}"
        for pos in range(c_start + 1, c_end):
            char_labels[pos] = f"I-{tag.upper()}"

    return mapped_spans, cleaned_text, char_labels


class TrainDataset(Dataset):
    def __init__(
        self,
        data: List[str],
        tokenizer: PreTrainedTokenizerBase,
        label2id: Dict[str, int],
        subsample_window: int,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.subsample_window = subsample_window
        self.examples = []

        for raw in data:

            mapped_spans, cleaned_text, char_labels = prep_data(raw)

            # Subsample 1 -- sub-samples of negative samples, max three windows
            if not mapped_spans:
                L = self.subsample_window
                T = len(cleaned_text)
                for i, ws in enumerate(range(0, T, L)):
                    if i > 2:
                        break
                    
                    we = min(ws + L, T)
                    chunk_text = cleaned_text[ws:we]
                    chunk_labels = char_labels[ws:we]
                    self._tokenize_and_store(chunk_text, chunk_labels)

            else:
                # Subsample 2 -- entity-centered sub-sampling (positive samples)
                L = self.subsample_window
                T = len(cleaned_text)
                for c_start, c_end, _ in mapped_spans:
                    span_len = c_end - c_start

                    if span_len >= L:
                        ws = max(0, c_start)
                    else:
                        pad = (L - span_len) // 2
                        ws = max(0, min(c_start - pad, T - L))

                    we = ws + L
                    sub_text = cleaned_text[ws:we]
                    sub_char_labels = char_labels[ws:we]
                    self._tokenize_and_store(sub_text, sub_char_labels)

                # Subsample 3 -- broken‑span sliding windows (to teach boundary cases)
                half_L = self.subsample_window // 2
                # for each span, if it’s safely away from the edges,
                # snip out the two halves around its midpoint
                for c_start, c_end, _ in mapped_spans:
                    if c_start >= half_L and c_end <= len(cleaned_text) - half_L:

                        # we don't want as many broken-span sub-samples as full sub-samples
                        if np.random.rand() > 0.05:
                            continue

                        mid = (c_start + c_end) // 2

                        # left half: [mid-half_L, mid)
                        ws1, we1 = mid - half_L, mid
                        self._tokenize_and_store(
                            cleaned_text[ws1:we1],
                            char_labels[ws1:we1],
                        )

                        # right half: [mid, mid+half_L)
                        ws2, we2 = mid, mid + half_L
                        self._tokenize_and_store(
                            cleaned_text[ws2:we2],
                            char_labels[ws2:we2],
                        )

    def _tokenize_and_store(self, text: str, char_labels: List[str]):
        # 1. run the tokenizer
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.subsample_window + 2,  # accounts for [CLS] + [SEP]
        )
        offsets = encoding.pop("offset_mapping")

        # 2. pull out the word_ids
        word_ids = encoding.word_ids()  # gives you a list of ints or None

        labels: List[int] = []
        last_wid = None
        for (off_start, off_end), wid in zip(offsets, word_ids):
            # — special token (CLS, SEP, PAD, etc.) → ignore
            if wid is None:
                labels.append(-100)
                last_wid = None
                continue

            # — not the first sub‐token of this word → ignore
            if wid == last_wid:
                labels.append(-100)
            else:
                # first sub‐token → assign a real label
                span = char_labels[off_start:off_end]
                ent = next((l for l in span if l != "O"), None)
                labels.append(self.label2id[ent] if ent else self.label2id["O"])

            last_wid = wid

        # 3. stash the example
        self.examples.append(
            {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": labels,
            }
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class ValWindowedDataset(Dataset):
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
        Sliding-window NER dataset that retains per-window metadata
        and delegates batching to an external DataCollator.

        Args:
            data: List of raw tagged strings.
            tokenizer: Fast HuggingFace tokenizer.
            label2id: BIO tag-to-id map.
            data_collator: Collator for padding/batching token features.
            window: Window size in characters.
            stride: Stride size in characters.
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
            T = len(cleaned_text)
            start = 0
            while start < T:
                end = min(start + self.window, T)
                sub_text = cleaned_text[start:end]
                sub_labels = char_labels[start:end]

                enc = self.tokenizer(
                    sub_text,
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=self.window + 2,
                )
                offsets = enc.pop("offset_mapping")
                word_ids = enc.word_ids()

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
                        span_ent = next(
                            (l for l in sub_labels[o0:o1] if l != "O"), None
                        )
                        labels.append(
                            self.label2id[span_ent] if span_ent else self.label2id["O"]
                        )
                        last_wid = wid

                # Store example with metadata
                self.examples.append(
                    {
                        "doc_id": doc_id,
                        "window_start": start,
                        "input_ids": enc["input_ids"],
                        "attention_mask": enc["attention_mask"],
                        "labels": labels,
                        "offset_mapping": offsets,
                        "raw": cleaned_text
                    }
                )

                if end == T:
                    break
                start += self.stride

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def collate_fn(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of windowed examples:
        1) Pop metadata fields
        2) Batch remaining token features via self.collator
        3) Re-attach metadata tensors
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
        PyTorch Lightning DataModule for NER task.
        Handles train/val dataset creation and dataloaders.
        """
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.train_subsample_window = train_subsample_window
        self.val_window = val_window
        self.val_stride = val_stride

        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.id2label = {i: l for l, i in self.label2id.items()}

        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer, label_pad_token_id=-100
        )

    def setup(self, stage=None):
        """
        Setup datasets for training and validation.
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

    def train_dataloader(self):
        """
        Returns DataLoader for training set.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
            # persistent_workers=True,
        )

    def val_dataloader(self):
        """
        Returns DataLoader for validation set.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate_fn,
            # persistent_workers=True,
        )


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, ignore_index=-100):
        """
        Focal loss for class imbalance in NER.
        """
        super().__init__()
        self.gamma = gamma
        self.ignore = ignore_index
        self.ce = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)

    def forward(self, logits, labels):
        """
        Compute focal loss.
        Args:
            logits: (B, T, C) or (N, C)
            labels: (B, T) or (N,)
        Returns:
            Scalar loss.
        """
        if logits.ndim == 3:
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
        loss = self.ce(logits, labels)
        pt = torch.exp(-loss)
        focal = (1 - pt) ** self.gamma * loss
        return focal.mean()


class NERTagger(pl.LightningModule):
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
        PyTorch LightningModule for NER model.
        Handles training, validation, and metrics.
        """
        super().__init__()
        self.save_hyperparameters()
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.model.train()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.loss_fn = FocalLoss()

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

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for NER model.
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        """
        Training step for NER model.
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
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        self.train_f1(valid_preds, valid_labels)
        self.train_precision(valid_preds, valid_labels)
        self.train_recall(valid_preds, valid_labels)

        return loss

    def on_train_epoch_end(self):
        """
        Log and reset training metrics at epoch end.
        """
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        self.log("train_precision", self.train_precision.compute())
        self.log("train_recall", self.train_recall.compute())

        self.train_f1.reset()
        self.train_precision.reset()
        self.train_recall.reset()

    def on_validation_epoch_start(self):
        self._val_sum.clear()
        self._val_count.clear()
        self._val_raw_texts.clear()
        
    def validation_step(self, batch, batch_idx):
        """
        Compute window‐level loss & metrics as before.
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

        # window‐level metrics
        preds = torch.argmax(logits, dim=-1)
        mask = labels != -100
        self.val_f1(preds[mask], labels[mask])
        self.val_precision(preds[mask], labels[mask])
        self.val_recall(preds[mask], labels[mask])
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # 2) buffer for doc‐level stitching
        for i in range(logits.size(0)):
            doc_id = int(batch["doc_id"][i])
            w0 = int(batch["window_start"][i])
            offs = batch["offset_mapping"][i] # list of (o0, o1)
            m = batch["attention_mask"][i]
            lg = logits[i]

            # store raw text once
            if doc_id not in self._val_raw_texts:
                self._val_raw_texts[doc_id] = batch["raw"][i]

            # figure how big we need the buffers
            max_off = max(o1 for o0, o1 in offs)
            req_len = w0 + max_off

            # lazy‐init or pad _val_sum, _val_count
            if doc_id not in self._val_sum:
                device = lg.device
                self._val_sum[doc_id] = torch.zeros(
                    (req_len, self.num_labels), device=device
                )
                self._val_count[doc_id] = torch.zeros(req_len, device=device)
            elif req_len > self._val_sum[doc_id].size(0):
                old = self._val_sum[doc_id].size(0)
                pad = req_len - old
                device = lg.device
                self._val_sum[doc_id] = torch.cat(
                    [
                        self._val_sum[doc_id],
                        torch.zeros((pad, self.num_labels), device=device),
                    ],
                    dim=0,
                )
                self._val_count[doc_id] = torch.cat(
                    [self._val_count[doc_id], torch.zeros(pad, device=device)], dim=0
                )

            # accumulate token logits into per‐char buckets
            for ti, (o0, o1) in enumerate(offs):
                if m[ti] == 0:
                    continue
                self._val_sum[doc_id][w0 + o0 : w0 + o1] += lg[ti]
                self._val_count[doc_id][w0 + o0 : w0 + o1] += 1

        return loss

    def _clean_and_label(self, raw: str) -> tuple[str, list[int]]:
        """
        Replicate your windowing‐clean logic to get cleaned_text and a
        char‐level list of label‐ids.
        """
        _, cleaned_text, char_labels = prep_data(raw)

        # map to ids
        return cleaned_text, [
            (self.label2id.get(l, -100) if l != "O" else self.label2id["O"])
            for l in char_labels
        ]

    def on_validation_epoch_end(self):
        # 1) token-level window metrics
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.log("val_precision", self.val_precision.compute())
        self.log("val_recall", self.val_recall.compute())
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()

        all_preds, all_gold = [], []
        device = next(self.model.parameters()).device

        # 2) for each doc, average logits→char preds, load gold, mask & collect
        for doc_id, sum_logits in self._val_sum.items():
            counts = self._val_count[doc_id].clamp(min=1.0).unsqueeze(-1)
            avg_logits = sum_logits / counts  # (T, C)
            char_preds = avg_logits.argmax(dim=-1).tolist()

            raw = self._val_raw_texts[doc_id]
            _, gold_ids = self._clean_and_label(raw)  # list[int], len T

            for p, g in zip(char_preds, gold_ids):
                if g != self.ignore_index:
                    all_preds.append(p)
                    all_gold.append(g)

        # 2) mask and tensorize
        mask = [g != self.ignore_index for g in all_gold]
        fp = [p for p, m in zip(all_preds, mask) if m]
        fg = [g for g, m in zip(all_gold, mask) if m]
        fp_t = torch.tensor(fp, dtype=torch.long, device=device)
        fg_t = torch.tensor(fg, dtype=torch.long, device=device)

        # 3) doc-level metrics
        self.val_f1_doc(fp_t, fg_t)
        self.val_precision_doc(fp_t, fg_t)
        self.val_recall_doc(fp_t, fg_t)

        self.log("val_f1_doc", self.val_f1_doc.compute(), prog_bar=True)
        self.log("val_precision_doc", self.val_precision_doc.compute())
        self.log("val_recall_doc", self.val_recall_doc.compute())
        self.val_f1_doc.reset()
        self.val_precision_doc.reset()
        self.val_recall_doc.reset()

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRSchedulerConfig]]:
        """
        Configure AdamW optimizer and linear warmup scheduler.
        Returns tuple for Lightning compatibility.
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
