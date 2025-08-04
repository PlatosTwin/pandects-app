from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import re
import string

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import lightning.pytorch as pl
import torch.nn as nn
from torchmetrics import F1Score, Precision, Recall

from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW


def extract_features(text: str, html: str, order: float) -> Tensor:
    """
    Compute engineered features for a single page.
    """

    # take last chars after the last newline; binary for if all digits, binary for if less than order

    # Text-based features
    num_chars = len(text)
    words = text.split()
    num_words = len(words)
    avg_chars_per_word = num_chars / num_words if num_words > 0 else 0.0
    prop_spaces = text.count(" ") / num_chars if num_chars > 0 else 0.0
    prop_digits = sum(c.isdigit() for c in text) / num_chars if num_chars > 0 else 0.0
    prop_newlines = text.count("\n") / num_chars if num_chars > 0 else 0.0
    prop_punct = (
        sum(c in string.punctuation for c in text) / num_chars if num_chars > 0 else 0.0
    )

    flag_is_all_digits = int(text.rsplit("\\n", 1)[-1].isdigit())
    s = text.rsplit("\\n", 1)[-1]
    flag_is_less_than_order = int(s.isdigit() and int(s) < order)
    
    flag_is_all_digits = int(text.rsplit("\n", 1)[-1].isdigit())
    flag_is_less_than_order = int(
        (s := text.rsplit("\n", 1)[-1]).isdigit() and int(s) < order
    )

    count_section = text.lower().count("section")
    count_article = text.lower().count("article")
    num_all_caps = sum(1 for w in words if w.isalpha() and w.isupper())
    prop_word_cap = (
        sum(1 for w in words if w[:1].isupper()) / num_words if num_words > 0 else 0.0
    )

    bigrams = [" ".join(bg) for bg in zip(words, words[1:])]
    num_bigrams = len(bigrams)
    unique_bigrams = len(set(bigrams))
    prop_unique_bigrams = unique_bigrams / num_bigrams if num_bigrams > 0 else 0.0
    # Trigrams
    trigrams = [" ".join(tg) for tg in zip(words, words[1:], words[2:])]
    num_trigrams = len(trigrams)
    unique_trigrams = len(set(trigrams))
    prop_unique_trigrams = unique_trigrams / num_trigrams if num_trigrams > 0 else 0.0

    # --- New HTML tag-based features ---
    num_tags = html.count("<")  # total HTML tags
    tag_to_text_ratio = num_tags / num_chars if num_chars > 0 else 0.0
    link_count = html.lower().count("<a ")
    img_count = html.lower().count("<img ")
    heading_tags = sum(html.lower().count(f"<h{i}") for i in range(1, 7))
    list_count = html.lower().count("<li")

    # --- Bullet/list detection ---
    bullet_count = sum(1 for line in text.split("\n") if line.strip().startswith("-"))

    # --- Legal boilerplate term counts ---
    terms = ["hereto", "herein", "hereby", "thereof", "wherein"]
    boilerplate_counts = [text.lower().count(term) for term in terms]

    # Keyword presence flags
    keywords = [
        "table of contents",
        "execution version",
        "in witness whereof",
        "exhibit",
        "signature",
        "list of exhibits",
        "schedule",
        "list of schedules",
        "index of",
        "recitals",
        "whereas",
        "now, therefore",
        "signed",
        "execution date",
        "effective",
        "dated as of"
        "entered into by and among"
        "[signature"
        "w i t n e s e t h"
        "/s/",
    ]
    flag_feats = [1.0 if kw in text.lower() else 0.0 for kw in keywords]

    # Punctuation breakdown
    num_colon = text.count(":")
    num_period = text.count(".")
    num_comma = text.count(",")
    total_punct = sum(c in string.punctuation for c in text)
    prop_colon = num_colon / total_punct if total_punct > 0 else 0.0
    prop_period = num_period / total_punct if total_punct > 0 else 0.0
    prop_comma = num_comma / total_punct if total_punct > 0 else 0.0

    # HTML tag features
    html_l = html.lower()
    has_table = 1.0 if re.search(r"</?(table|tr|td)", html_l) else 0.0
    count_p = html_l.count("<p")
    count_div = html_l.count("<div")

    # Aggregate feature vector
    feat_list = [
        num_words,
        num_chars,
        avg_chars_per_word,
        prop_spaces,
        prop_digits,
        prop_newlines,
        prop_punct,
        flag_is_all_digits,
        flag_is_less_than_order,
        count_section,
        count_article,
        num_all_caps,
        prop_word_cap,
        # n-gram stats
        num_bigrams,
        unique_bigrams,
        prop_unique_bigrams,
        num_trigrams,
        unique_trigrams,
        prop_unique_trigrams,
        # HTML tag-based
        num_tags,
        tag_to_text_ratio,
        link_count,
        img_count,
        heading_tags,
        list_count,
        bullet_count,
        # boilerplate term counts
        *boilerplate_counts,
        *flag_feats,
        prop_colon,
        prop_period,
        prop_comma,
        has_table,
        count_p,
        count_div,
        order,
    ]

    return torch.tensor(feat_list, dtype=torch.float)


class PageDataset(Dataset):
    """Dataset yielding (features, label) per page."""

    def __init__(self, df: pd.DataFrame, label2idx: Dict[str, int]) -> None:
        df = df.fillna({"text": "", "html": "", "order": 0})
        self.texts = df["text"].astype(str).tolist()
        self.htmls = df["html"].astype(str).tolist()
        self.orders = df["order"].astype(float).tolist()
        self.labels = torch.tensor(
            df["label"].map(label2idx).tolist(), dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        feats = extract_features(self.texts[idx], self.htmls[idx], self.orders[idx])
        return feats, self.labels[idx]


class DocumentDataset(Dataset):
    """Groups pages by agreement and sorts by order."""

    def __init__(self, df: pd.DataFrame, label2idx: Dict[str, int]):
        self.page_ds = PageDataset(df, label2idx)
        groups = defaultdict(list)
        for i, aid in enumerate(df["agreement_uuid"]):
            groups[aid].append(i)
        self.batches = []
        for idxs in groups.values():
            idxs.sort(key=lambda i: self.page_ds.orders[i])
            self.batches.append(idxs)

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        idxs = self.batches[idx]
        feats = []
        labels = []
        for i in idxs:
            f, l = self.page_ds[i]
            feats.append(f)
            labels.append(l)
        return {
            "features": torch.stack(feats),  # (S, F)
            "labels": torch.tensor(labels, dtype=torch.long),  # (S,)
        }


def collate_pages(batch: List[Dict[str, Tensor]], pad_label: int = -100):
    B = len(batch)
    max_S = max(item["labels"].size(0) for item in batch)
    F = batch[0]["features"].size(1)

    feats_padded = []
    labels_padded = []
    for item in batch:
        S = item["labels"].size(0)
        pad_feats = torch.cat([item["features"], torch.zeros((max_S - S, F))], dim=0)
        pad_labels = torch.cat(
            [item["labels"], torch.full((max_S - S,), pad_label, dtype=torch.long)]
        )
        feats_padded.append(pad_feats)
        labels_padded.append(pad_labels)

    features = torch.stack(feats_padded)  # (B, max_S, F)
    labels = torch.stack(labels_padded)  # (B, max_S)
    return features, labels


class PageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 8,
        val_split: float = 0.2,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        unique = self.df["agreement_uuid"].unique().tolist()
        train_ids, val_ids = train_test_split(
            unique, test_size=self.val_split, random_state=42
        )
        train_df = self.df[self.df["agreement_uuid"].isin(train_ids)]
        val_df = self.df[self.df["agreement_uuid"].isin(val_ids)]

        labels = sorted(train_df["label"].unique())
        self.label2idx = {lab: i for i, lab in enumerate(labels)}
        self.num_classes = len(labels)

        self.train_ds = DocumentDataset(train_df, self.label2idx)
        self.val_ds = DocumentDataset(val_df, self.label2idx)

        # number of features from a single page
        sample_feats, _ = PageDataset(train_df, self.label2idx)[0]
        self.num_features = sample_feats.numel()

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_pages,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_pages,
        )


class PageClassifier(pl.LightningModule):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # MLP head on engineered features
        self.fc1 = nn.Linear(num_features, hidden_dim)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # CRF parameters
        self.transitions = nn.Parameter(torch.empty(num_classes, num_classes))
        nn.init.xavier_uniform_(self.transitions)
        self.start_transitions = nn.Parameter(torch.empty(num_classes))
        nn.init.normal_(self.start_transitions, 0.0, 0.1)
        self.end_transitions = nn.Parameter(torch.empty(num_classes))
        nn.init.normal_(self.end_transitions, 0.0, 0.1)

        # Metrics
        self.train_precision = Precision(task="multiclass", num_classes=num_classes)
        self.train_recall = Recall(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.val_precision = Precision(task="multiclass", num_classes=num_classes)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, features: Tensor) -> Tensor:
        # features: (B, S, F)
        B, S, F = features.size()
        flat = features.view(B * S, F)
        x = torch.relu(self.fc1(flat))
        x = self.dropout(x)
        logits = self.fc2(x)  # (B·S, C)
        return logits.view(B, S, -1)

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

    def training_step(self, batch, batch_idx: int):
        feats, labels = batch  # now only two items
        emissions = self(feats)  # forward takes only features

        seq_mask = labels != -100
        ll = self._compute_log_likelihood(emissions, labels, seq_mask)
        loss = -ll

        preds = self._viterbi_decode(emissions, seq_mask)
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
        feats, labels = batch
        emissions = self(feats)

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
