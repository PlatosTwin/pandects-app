from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
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
        self.agreement_ids = df_filled["agreement_uuid"].tolist()  # new
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
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.features[idx],
            self.labels[idx],
        )


class DocumentDataset(Dataset):
    def __init__(self, page_ds: PageDataset):
        self.page_ds = page_ds
        # group indices by agreement
        groups = defaultdict(list)
        for idx, aid in enumerate(page_ds.agreement_ids):
            groups[aid].append(idx)
        # sort pages within each group by original order tensor
        self.batches: List[List[int]] = []
        for idx_list in groups.values():
            idx_list.sort(key=lambda i: page_ds.orders[i])
            self.batches.append(idx_list)

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int):
        ids = self.batches[idx]
        input_ids = self.page_ds.input_ids[ids]
        attention_mask = self.page_ds.attention_mask[ids]
        features = self.page_ds.features[ids]
        labels = self.page_ds.labels[ids]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "features": features,
            "labels": labels,
        }


def collate_pages(batch: List[Dict[str, Tensor]], pad_label: int = -100):
    B = len(batch)
    # find max sequence length
    max_S = max(item["labels"].shape[0] for item in batch)

    def pad_list(tensors, pad_value=0, dim=0):
        padded = []
        for t in tensors:
            pad_shape = list(t.shape)
            pad_shape[dim] = max_S - t.shape[dim]
            pad_tensor = torch.full(
                pad_shape, pad_value, dtype=t.dtype, device=t.device
            )
            padded.append(torch.cat([t, pad_tensor], dim=dim))
        return torch.stack(padded, dim=0)

    input_ids = pad_list([item["input_ids"] for item in batch], pad_value=0)
    attention_mask = pad_list([item["attention_mask"] for item in batch], pad_value=0)
    features = pad_list([item["features"] for item in batch], pad_value=0.0)
    labels = pad_list([item["labels"] for item in batch], pad_value=pad_label)

    return input_ids, attention_mask, features, labels


class PageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df: pd.DataFrame,
        model_name: str = "distilbert/distilbert-base-uncased",
        batch_size: int = 8,
        val_split: float = 0.2,
        num_workers: int = 0,
        top_k: int = 5,
    ) -> None:
        super().__init__()
        self.df = df
        self.model_name = model_name
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.top_k = top_k

    def setup(self, stage: Optional[str] = None) -> None:
        # split by agreement_uuid
        unique_ids = self.df["agreement_uuid"].unique().tolist()
        train_ids, val_ids = train_test_split(
            unique_ids, test_size=self.val_split, random_state=42
        )
        train_df = self.df[self.df["agreement_uuid"].isin(train_ids)]
        val_df = self.df[self.df["agreement_uuid"].isin(val_ids)]

        labels = sorted(train_df["label"].unique())
        self.label2idx = {lab: i for i, lab in enumerate(labels)}
        self.num_classes = len(labels)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        # page-level datasets
        train_page_ds = PageDataset(
            train_df, self.label2idx, self.tokenizer, self.top_k
        )
        val_page_ds = PageDataset(val_df, self.label2idx, self.tokenizer, self.top_k)
        # document-level datasets
        self.train_ds = DocumentDataset(train_page_ds)
        self.val_ds = DocumentDataset(val_page_ds)

        # feature dim (unchanged)
        _, _, sample_feats, _ = train_page_ds[0]
        self.num_features: int = sample_feats.numel()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_pages,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_pages,
        )


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

        # ─── CRF layer parameters ─────────────────────────────────────────
        # transition scores from tag_i → tag_j
        self.transitions = nn.Parameter(torch.empty(num_classes, num_classes))
        nn.init.xavier_uniform_(self.transitions)
        # start / end transition scores
        self.start_transitions = nn.Parameter(torch.empty(num_classes))
        nn.init.normal_(self.start_transitions, mean=0.0, std=0.1)
        self.end_transitions = nn.Parameter(torch.empty(num_classes))
        nn.init.normal_(self.end_transitions, mean=0.0, std=0.1)

        # Metrics
        self.train_precision = Precision(task="multiclass", num_classes=num_classes)
        self.train_recall = Recall(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.val_precision = Precision(task="multiclass", num_classes=num_classes)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)

    def _compute_log_likelihood(
        self, emissions: Tensor, tags: Tensor, mask: Tensor
    ) -> Tensor:
        """
        emissions: (B, S, C), tags: (B, S), mask: (B, S)  (bool tensor)
        returns mean log‐likelihood over batch
        """
        B, S, C = emissions.size()

        # 1) sanitize tags so that ignored positions never index out of bounds
        tags_s = tags.clone()
        # wherever mask is False (i.e. label == ignore_index), replace with class 0
        tags_s[~mask] = 0

        # 2) score of the provided path
        score = self.start_transitions[tags_s[:, 0]] + emissions[:, 0].gather(
            1, tags_s[:, 0:1]
        ).squeeze(1)

        for t in range(1, S):
            emit_score = emissions[:, t].gather(1, tags_s[:, t : t + 1]).squeeze(1)
            trans_score = self.transitions[tags_s[:, t - 1], tags_s[:, t]]
            # only add when mask[:,t] == True
            score = score + (emit_score + trans_score) * mask[:, t].float()

        # 3) add end transition for the last real tag in each sequence
        seq_lens = mask.sum(dim=1).long() - 1
        last_tags = tags_s.gather(1, seq_lens.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]

        # 4) compute partition function with forward algorithm
        alphas = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        for t in range(1, S):
            emit = emissions[:, t].unsqueeze(2)  # (B, C, 1)
            trans = self.transitions.unsqueeze(0)  # (1, C, C)
            scores = alphas.unsqueeze(2) + trans + emit  # (B, C, C)
            new_alphas = torch.logsumexp(scores, dim=1)  # (B, C)
            mask_t = mask[:, t].unsqueeze(1).float()
            alphas = new_alphas * mask_t + alphas * (1.0 - mask_t)

        alphas = alphas + self.end_transitions.unsqueeze(0)
        partition = torch.logsumexp(alphas, dim=1)  # (B,)

        # 5) return mean log‐likelihood
        return (score - partition).mean()

    def _viterbi_decode(self, emissions: Tensor, mask: Tensor) -> Tensor:
        """
        returns best tag path, shape (B, S)
        """
        B, S, C = emissions.size()
        seq_lens = mask.sum(dim=1).long()

        # init
        viterbi_score = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (B, C)
        backpointers = []

        for t in range(1, S):
            b_score = viterbi_score.unsqueeze(2)  # (B, C, 1)
            trans = self.transitions.unsqueeze(0)  # (1, C, C)
            scores = b_score + trans  # (B, C, C)
            max_score, idxs = scores.max(dim=1)  # (B, C)
            viterbi_score = max_score + emissions[:, t]
            backpointers.append(idxs)

        # add end transitions
        viterbi_score = viterbi_score + self.end_transitions.unsqueeze(0)
        best_last = viterbi_score.argmax(dim=1)  # (B,)

        # backtrack
        paths = []
        for b in range(B):
            length = seq_lens[b].item()
            best_tag = best_last[b].item()
            seq = [best_tag]
            for ptrs in reversed(backpointers[: length - 1]):
                best_tag = ptrs[b, best_tag].item()
                seq.append(best_tag)
            seq.reverse()
            # pad if needed
            if length < S:
                seq = seq + [0] * (S - length)
            paths.append(seq)

        return torch.tensor(paths, device=emissions.device)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, features: Tensor
    ) -> Tensor:
        # input_ids: (B, S, T), features: (B, S, F)
        B, S, T = input_ids.size()
        flat_ids = input_ids.view(B * S, T)
        flat_mask = attention_mask.view(B * S, T)
        flat_feats = features.view(B * S, -1)

        outputs = self.bert(input_ids=flat_ids, attention_mask=flat_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # (B·S, H)
        feat_emb = self.feat_proj(flat_feats)  # (B·S, F)

        x = torch.cat([pooled, feat_emb], dim=1)  # (B·S, H+F)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)  # (B·S, C)

        # reshape into a page‐sequence
        emissions = logits.view(B, S, -1)  # (B, S, C)
        return emissions

    def training_step(self, batch, batch_idx: int):
        ids, mask, feats, labels = batch  # labels: (B, S) with some ignore‐index
        emissions = self(ids, mask, feats)  # (B, S, C)

        seq_mask = labels != -100  # mask out padding if you used -100
        ll = self._compute_log_likelihood(emissions, labels, seq_mask)
        loss = -ll

        preds = self._viterbi_decode(emissions, seq_mask)  # (B, S)
        # flatten masked positions
        flat_preds = preds[seq_mask]
        flat_labels = labels[seq_mask]

        self.train_precision(flat_preds, flat_labels)
        self.train_recall(flat_preds, flat_labels)
        self.train_f1(flat_preds, flat_labels)
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
        emissions = self(ids, mask, feats)

        seq_mask = labels != -100
        ll = self._compute_log_likelihood(emissions, labels, seq_mask)
        loss = -ll

        preds = self._viterbi_decode(emissions, seq_mask)
        flat_preds = preds[seq_mask]
        flat_labels = labels[seq_mask]

        self.val_precision(flat_preds, flat_labels)
        self.val_recall(flat_preds, flat_labels)
        self.val_f1(flat_preds, flat_labels)
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
