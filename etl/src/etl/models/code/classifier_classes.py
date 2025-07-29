import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy
from collections import Counter


def build_vocab(texts: list[str], max_vocab_size: int = 20_000) -> dict:
    """Build vocabulary from a list of texts."""
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    most_common = counter.most_common(max_vocab_size)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, (tok, _) in enumerate(most_common, start=2):
        vocab[tok] = i
    return vocab


def prepare_sample(html: str, text: str, vocab: dict, max_seq_len: int):
    """Prepare token indices, sequence length, and extra features for a single example."""
    half = max_seq_len // 2
    part1 = html[:half]
    part2 = text[: max_seq_len - half]
    combined = f"{part1} {part2}"
    toks = combined.split()
    idxs = [vocab.get(tok, vocab.get("<UNK>", 1)) for tok in toks]
    length = len(idxs)
    if length < max_seq_len:
        idxs += [vocab.get("<PAD>", 0)] * (max_seq_len - length)
    else:
        idxs = idxs[:max_seq_len]
        length = max_seq_len

    total = length or 1
    num_caps = sum(1 for t in toks if t.isupper())
    num_nums = sum(1 for t in toks if t.isdigit())
    avg_len = sum(len(t) for t in toks) / total
    tags = re.findall(r"<[^>]+>", html)
    html_tag_ratio = len(tags) / total
    feats = [total, num_caps / total, num_nums / total, avg_len, html_tag_ratio]
    return idxs, length, feats


class PageDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, vocab: dict, label2idx: dict, max_seq_len: int = 300
    ):
        """Create a dataset for page classification."""
        self.vocab = vocab
        self.label2idx = label2idx
        self.max_seq_len = max_seq_len
        htmls = df["html"].fillna("").tolist()
        texts = df["text"].fillna("").tolist()
        self.samples = []
        for html, text, label in zip(htmls, texts, df["label"]):
            idxs, length, feats = prepare_sample(html, text, self.vocab, self.max_seq_len)
            self.samples.append(
                (
                    torch.tensor(idxs, dtype=torch.long),
                    torch.tensor(length, dtype=torch.long),
                    torch.tensor(feats, dtype=torch.float),
                    torch.tensor(label2idx[label], dtype=torch.long),
                )
            )

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, i: int):
        """Return the i-th sample."""
        return self.samples[i]


class PageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 32,
        val_split: float = 0.2,
        num_workers: int = 0,
        max_vocab_size: int = 20000,
        max_seq_len: int = 300,
    ):
        """Initialize the data module for page classification."""
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len

    def setup(self, stage=None) -> None:
        """Setup datasets and vocabulary."""
        train_df, val_df = train_test_split(
            self.df,
            test_size=self.val_split,
            stratify=self.df["label"],
            random_state=42,
        )

        # build vocab on training texts
        texts = (
            train_df["html"].fillna("") + " " + train_df["text"].fillna("")
        ).tolist()
        self.vocab = build_vocab(texts, self.max_vocab_size)

        # label mapping
        labels = sorted(train_df["label"].unique())
        self.label2idx = {lab: i for i, lab in enumerate(labels)}
        self.num_classes = len(self.label2idx)

        # datasets
        self.train_ds = PageDataset(
            train_df, self.vocab, self.label2idx, self.max_seq_len
        )
        self.val_ds = PageDataset(val_df, self.vocab, self.label2idx, self.max_seq_len)

        # number of extra features per sample
        self.num_features = len(self.train_ds[0][2])
        self.vocab_size = len(self.vocab)

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class PageClassifier(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_features: int,
        hidden_dim: int,
        num_classes: int,
        learning_rate: float = 1e-3,
    ):
        """Initialize the page classifier model."""
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(
            self.hparams.vocab_size, self.hparams.embed_dim, padding_idx=0
        )
        self.fc1 = nn.Linear(
            self.hparams.embed_dim + self.hparams.num_features, self.hparams.hidden_dim
        )
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(self.hparams.hidden_dim, self.hparams.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def forward(
        self, indices: torch.Tensor, lengths: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for page classification."""
        emb = self.embedding(indices)  # (B, L, D)
        mask = (indices != 0).unsqueeze(-1).float()  # (B, L, 1)
        summed = (emb * mask).sum(dim=1)  # (B, D)
        lengths = lengths.unsqueeze(1).float()  # (B, 1)
        avg_emb = summed / lengths  # (B, D)
        x = torch.cat([avg_emb, features], dim=1)  # (B, D+F)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # (B, C)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Training step for one batch."""
        idxs, lengths, feats, labels = batch
        logits = self(idxs, lengths, feats)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        """Validation step for one batch."""
        idxs, lengths, feats, labels = batch
        logits = self(idxs, lengths, feats)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizers for training."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
