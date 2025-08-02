from typing import Dict, List, Optional, Tuple
from collections import Counter
import re

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import F1Score, Precision, Recall

from transformers import AutoTokenizer, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from torch.optim import AdamW


def extract_tag_features(html: str, top_k: int = 5) -> Tuple[List[int], List[int]]:
    """
    Extract fixed-length counts of HTML tag types and tag-ngrams from the input HTML.

    Args:
        html: Raw HTML string.
        top_k: Number of top tag-types and tag-bigrams to return.

    Returns:
        A tuple of two lists (type_counts, ngram_counts), each of length top_k.
    """
    tags = re.findall(r"<([^>\s/]+)", html)
    type_counts = [count for _, count in Counter(tags).most_common(top_k)]
    type_counts += [0] * max(0, top_k - len(type_counts))
    bigrams = [f"{tags[i]}_{tags[i+1]}" for i in range(len(tags) - 1)]
    ngram_counts = [count for _, count in Counter(bigrams).most_common(top_k)]
    ngram_counts += [0] * max(0, top_k - len(ngram_counts))
    return type_counts, ngram_counts


class PageDataset(Dataset):
    """
    PyTorch Dataset for page classification.
    Tokenizes combined HTML/text and computes features for each example.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label2idx: Dict[str, int],
        tokenizer: PreTrainedTokenizerBase,
        top_k: int = 5,
    ) -> None:
        df_filled = df.fillna({"html": "", "text": "", "order": 0})
        self.htmls = df_filled["html"].astype(str).tolist()
        self.texts = df_filled["text"].astype(str).tolist()
        self.orders = df_filled["order"].astype(float).tolist()
        self.labels = torch.tensor(
            df_filled["label"].map(label2idx).tolist(), dtype=torch.long
        )

        self.tokenizer = tokenizer
        self.top_k = top_k

        # #2: tokenize combined HTML + text
        combined_inputs = [
            "[HTML] " + h[:1024] + " [TEXT] " + t
            for h, t in zip(self.htmls, self.texts)
        ]
        encoding = tokenizer(
            combined_inputs,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        self.input_ids = encoding["input_ids"]
        self.attention_mask = encoding["attention_mask"]

        # Base statistics
        totals = self.attention_mask.sum(dim=1).clamp(min=1).float()
        html_lengths = torch.tensor([len(h) for h in self.htmls], dtype=torch.float)
        tag_counts = torch.tensor(
            [len(re.findall(r"<[^>]+>", h)) for h in self.htmls], dtype=torch.float
        )
        link_counts = torch.tensor(
            [h.lower().count("<a ") for h in self.htmls], dtype=torch.float
        )
        img_counts = torch.tensor(
            [h.lower().count("<img ") for h in self.htmls], dtype=torch.float
        )
        heading_counts = torch.tensor(
            [sum(h.lower().count(f"<h{i}") for i in range(1, 7)) for h in self.htmls],
            dtype=torch.float,
        )

        type_counts_list, ngram_counts_list = zip(
            *[extract_tag_features(h[:5000], top_k) for h in self.htmls]
        )
        type_counts = torch.tensor(type_counts_list, dtype=torch.float)
        ngram_counts = torch.tensor(ngram_counts_list, dtype=torch.float)

        token_lists = [
            tokenizer.convert_ids_to_tokens(ids.tolist()) for ids in self.input_ids
        ]
        num_caps = torch.tensor(
            [
                sum(1 for tok in toks if tok.isalpha() and tok.isupper())
                for toks in token_lists
            ],
            dtype=torch.float,
        )
        num_nums = torch.tensor(
            [
                sum(any(c.isdigit() for c in tok) for tok in toks)
                for toks in token_lists
            ],
            dtype=torch.float,
        )
        avg_token_lengths = torch.tensor(
            [
                sum(len(tok) for tok in toks) / tot
                for toks, tot in zip(token_lists, totals)
            ],
            dtype=torch.float,
        )
        orders_tensor = torch.tensor(self.orders, dtype=torch.float)

        # Combine all features
        self.features = torch.cat(
            [
                html_lengths.unsqueeze(1),
                tag_counts.unsqueeze(1),
                link_counts.unsqueeze(1),
                img_counts.unsqueeze(1),
                heading_counts.unsqueeze(1),
                totals.unsqueeze(1),
                (num_caps / totals).unsqueeze(1),
                (num_nums / totals).unsqueeze(1),
                avg_token_lengths.unsqueeze(1),
                orders_tensor.unsqueeze(1),
                type_counts,
                ngram_counts,
            ],
            dim=1,
        )

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.features[idx],
            self.labels[idx],
        )


class PageDataModule(pl.LightningDataModule):
    """
    LightningDataModule for loading PageDataset with train/validation splits.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        model_name: str = "distilbert/distilbert-base-uncased",
        batch_size: int = 32,
        val_split: float = 0.2,
        num_workers: int = 0,
        top_k: int = 5,
    ) -> None:
        """
        Args:
            df: Full dataset DataFrame.
            model_name: HuggingFace model name for tokenizer.
            batch_size: Mini-batch size.
            val_split: Fraction of data for validation.
            num_workers: Number of DataLoader workers.
            top_k: Number of tag-based features.
        """
        super().__init__()
        self.df = df
        self.model_name = model_name
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.top_k = top_k

    def setup(self, stage: Optional[str] = None) -> None:
        """Prepares train/validation splits and datasets."""
        df_filled = self.df.fillna({"html": "", "text": "", "order": -1})
        train_df, val_df = train_test_split(
            df_filled,
            test_size=self.val_split,
            stratify=df_filled["label"],
            random_state=42,
        )

        labels = sorted(train_df["label"].unique())
        self.label2idx: Dict[str, int] = {lab: i for i, lab in enumerate(labels)}
        self.num_classes: int = len(self.label2idx)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.train_ds = PageDataset(
            train_df, self.label2idx, self.tokenizer, self.top_k
        )
        self.val_ds = PageDataset(val_df, self.label2idx, self.tokenizer, self.top_k)

        # Determine number of HTML/text features
        _, _, sample_feats, _ = self.train_ds[0]
        self.num_features: int = sample_feats.numel()

    def train_dataloader(self) -> DataLoader:
        """Returns training DataLoader."""
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns validation DataLoader."""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
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


class PageClassifier(pl.LightningModule):
    """
    LightningModule for page classification.
    Implements: #3 feature projection, #4 AdamW+warmup, #5 focal loss, #6 stronger regularization.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_features: int,
        num_classes: int,
        learning_rate: float = 2e-5,
        model_name: str = "distilbert/distilbert-base-uncased",
        freeze_epochs: int = 1,
        unfreeze_last_n: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Backbone
        self.bert = AutoModel.from_pretrained(model_name)
        # Freeze backbone initially
        for p in self.bert.parameters():
            p.requires_grad = False

        # Feature projection
        self.feat_proj = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.LayerNorm(num_features),
            nn.ReLU(),
        )

        bert_dim = self.bert.config.hidden_size
        # MLP head
        self.fc1 = nn.Linear(bert_dim + num_features, hidden_dim)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # Focal loss for class imbalance
        self.loss_fn = FocalLoss(gamma=2.0)

        # Metrics
        self.train_precision = Precision(task="multiclass", num_classes=num_classes)
        self.train_recall = Recall(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.val_precision = Precision(task="multiclass", num_classes=num_classes)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, features: Tensor
    ) -> Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        feat_emb = self.feat_proj(features)
        x = torch.cat([pooled, feat_emb], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def training_step(self, batch, batch_idx: int):
        ids, mask, feats, labels = batch
        logits = self(ids, mask, feats)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.train_precision(preds, labels)
        self.train_recall(preds, labels)
        self.train_f1(preds, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        self.log("train_precision", self.train_precision.compute())
        self.log("train_recall", self.train_recall.compute())
        self.train_f1.reset()
        self.train_precision.reset()
        self.train_recall.reset()

    def validation_step(self, batch, batch_idx: int):
        ids, mask, feats, labels = batch
        logits = self(ids, mask, feats)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.log("val_precision", self.val_precision.compute())
        self.log("val_recall", self.val_recall.compute())
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01
        )
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
