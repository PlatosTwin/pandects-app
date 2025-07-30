# Standard library
import os
import re
from typing import Any, Dict, List, Tuple, cast
from torch.optim import Optimizer
from lightning.pytorch.utilities.types import LRSchedulerConfig

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


class TrainDataset(Dataset):
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
        tag_pattern = re.compile(r"<(section|article|page)>(.*?)</\1>", re.DOTALL)

        for raw in data:
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

            # 5.a. sub-samples of negative samples
            if not mapped_spans:
                L = self.subsample_window
                T = len(cleaned_text)
                for ws in range(0, T, L):
                    we = min(ws + L, T)
                    chunk_text = cleaned_text[ws:we]
                    chunk_labels = char_labels[ws:we]
                    self._tokenize_and_store(chunk_text, chunk_labels)

            else:
                # 5.b. entity-centered sub-sampling (positive samples)
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
                    
                # 5.c. broken‑span sliding windows (to teach boundary cases)
                half_L = self.subsample_window // 2
                # for each span, if it’s safely away from the edges,
                # snip out the two halves around its midpoint
                for c_start, c_end, _ in mapped_spans:
                    if c_start >= half_L and c_end <= len(cleaned_text) - half_L:
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
        window: int = 512,
        stride: int = 256,
    ):
        self.examples = []
        tag_pattern = re.compile(r"<(section|article|page)>(.*?)</\1>", re.DOTALL)

        for raw in data:
            # 1) find + map spans
            spans = []
            for m in tag_pattern.finditer(raw):
                spans.append((m.start(2), m.end(2), m.group(1)))

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
            cleaned = "".join(cleaned_chars)

            mapped: List[Tuple[int, int, str]] = []
            for start, end, tag in spans:
                c0 = orig2clean.get(start)
                c1 = orig2clean.get(end - 1)
                if c0 is not None and c1 is not None:
                    mapped.append((c0, c1 + 1, tag))

            # 2) build char-level BIO labels
            chars_lbl = ["O"] * len(cleaned)
            for c0, c1, tag in mapped:
                chars_lbl[c0] = f"B-{tag.upper()}"
                for p in range(c0 + 1, c1):
                    chars_lbl[p] = f"I-{tag.upper()}"

            # 3) Sliding windows
            T = len(cleaned)
            L = window
            S = stride

            start = 0
            while start < T:
                end = min(start + L, T)
                sub_text = cleaned[start:end]
                sub_labels = chars_lbl[start:end]

                encoding = tokenizer(
                    sub_text,
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=L + 2,  # accounts for [CLS] + [SEP]
                )
                offsets = encoding["offset_mapping"]
                word_ids = encoding.word_ids()

                labels: List[int] = []
                last_wid = None
                for (off_start, off_end), wid in zip(offsets, word_ids):
                    if wid is None:
                        labels.append(-100)
                        last_wid = None
                        continue
                    if wid == last_wid:
                        labels.append(-100)
                    else:
                        span = sub_labels[off_start:off_end]
                        ent = next((l for l in span if l != "O"), None)
                        labels.append(label2id[ent] if ent else label2id["O"])
                    last_wid = wid

                self.examples.append(
                    {
                        "input_ids": encoding["input_ids"],
                        "attention_mask": encoding["attention_mask"],
                        "labels": labels,
                    }
                )

                if end == T:
                    break
                start += S

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


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
            self.train_data,
            self.tokenizer,
            self.label2id,
            subsample_window=self.train_subsample_window,
        )
        self.val_dataset = ValWindowedDataset(
            data=self.val_data,
            tokenizer=self.tokenizer,
            label2id=self.label2id,
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
            collate_fn=self.data_collator,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
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
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id={v: k for k, v in self.id2label.items()},
        )
        self.model.train()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.loss_fn = FocalLoss()

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

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass for NER model.
        """
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        """
        Training step for NER model.
        """
        outputs = self(**batch)
        logits = outputs.logits  # (B, T, C)
        labels = batch["labels"]  # (B, T)
        loss = self.loss_fn(
            logits.reshape(-1, self.num_labels),
            labels.view(-1),
        )
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

    def _get_window_labels(
        self,
        full_labels: torch.Tensor,
        windows: List[Tuple[int, int]],
        encoding,
    ) -> torch.Tensor:
        """
        Build token-level labels for each sliding window.
        Args:
            full_labels: 1D Tensor of labels for the full text (length T)
            windows: list of (w0, w1) char offsets
            encoding: BatchEncoding with offset_mapping (n_win, L, 2)
        Returns:
            Tensor of shape (n_win, L) with label_ids or -100.
        """
        offset_maps = encoding["offset_mapping"]  # (n_win, L, 2)
        n_win, L = offset_maps.shape[:2]
        win_labels = []
        for (w0, w1), offsets in zip(windows, offset_maps.tolist()):
            labels = []
            for off_start, off_end in offsets:
                if off_end == 0:
                    labels.append(-100)
                else:
                    span = full_labels[w0 + off_start : w0 + off_end]
                    non_ign = span[span != -100]
                    labels.append(int(non_ign[0]) if non_ign.numel() else -100)
            if len(labels) < L:
                labels += [-100] * (L - len(labels))
            else:
                labels = labels[:L]
            win_labels.append(labels)
        return torch.tensor(win_labels, dtype=torch.long, device=self.device)

    def validation_step(self, batch, batch_idx):
        """
        Validation step: just run model and aggregate metrics, since windowing and tokenization are precomputed.
        """
        outputs = self(**batch)
        logits = outputs.logits
        labels = batch["labels"]
        # Avoid torch.tensor(labels) if labels is already a tensor
        if not torch.is_tensor(labels):
            labels_tensor = torch.tensor(labels)
        else:
            labels_tensor = labels
        labels_tensor = labels_tensor.clone().detach()
        loss = self.loss_fn(
            logits.view(-1, self.num_labels),
            labels_tensor.view(-1),
        )
        preds = torch.argmax(logits, dim=-1)
        mask = labels_tensor != -100
        valid_preds = preds[mask]
        valid_labels = labels_tensor[mask]

        self.val_f1(valid_preds, valid_labels)
        self.val_precision(valid_preds, valid_labels)
        self.val_recall(valid_preds, valid_labels)
        self.log("val_loss", loss, prog_bar=True, batch_size=logits.size(0))

        return loss

    def on_validation_epoch_end(self):
        """
        Log and reset validation metrics at epoch end.
        """
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.log("val_precision", self.val_precision.compute())
        self.log("val_recall", self.val_recall.compute())

        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """
        Prediction step for inference.
        """
        outputs = self(**batch)
        logits = outputs.logits
        labels = batch["labels"]

        if not torch.is_tensor(labels):
            labels_tensor = torch.tensor(labels)
        else:
            labels_tensor = labels
        labels_tensor = labels_tensor.clone().detach()

        preds = torch.argmax(logits, dim=-1)

        return preds

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
