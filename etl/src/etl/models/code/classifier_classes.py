from collections import defaultdict
from typing import Dict

import lightning.pytorch as pl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchmetrics import F1Score, Precision, Recall
import xgboost as xgb

from classifier_utils import extract_features
from constants import CLASSIFIER_LABEL_LIST


class PageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label2idx: Dict[str, int],
        model: xgb.Booster,
        inference: bool = False,
    ):
        df = df.fillna({"text": "", "html": "", "order": 0})
        feats = np.vstack(
            [
                extract_features(t, h, o)
                for t, h, o in zip(df["text"], df["html"], df["order"])
            ]
        )
        dmat = xgb.DMatrix(feats)
        probs = model.predict(dmat)  # (N, C)
        # avoid log(0) → -inf
        probs_tensor = torch.tensor(probs, dtype=torch.float)
        probs_tensor = probs_tensor.clamp(min=1e-8)
        self.emissions = torch.log(probs_tensor)

        if not inference:
            self.labels = torch.tensor(
                df["label"].map(label2idx).values, dtype=torch.long
            )
        else:
            self.labels = None

    def __len__(self):
        return self.emissions.size(0)

    def __getitem__(self, i):
        if self.labels is None:
            return {"emissions": self.emissions[i]}
        return {"emissions": self.emissions[i], "labels": self.labels[i]}


class DocumentDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label2idx: Dict[str, int],
        model: xgb.Booster,
        inference: bool = False,
    ):
        self.page_ds = PageDataset(df, label2idx, model, inference=inference)
        self.ids = []
        grp = defaultdict(list)
        for i, a in enumerate(df["agreement_uuid"]):
            grp[a].append(i)
        for idxs in grp.values():
            # order is preserved by df
            self.ids.append(sorted(idxs))

        self.inference = inference

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        idxs = self.ids[idx]
        emis = torch.stack([self.page_ds[i]["emissions"] for i in idxs])

        if self.inference:
            return {"emissions": emis}

        labs = torch.tensor([self.page_ds[i]["labels"] for i in idxs], dtype=torch.long)
        return {"emissions": emis, "labels": labs}


# pad emissions & labels to (B, S, C)/(B, S)
def collate_pages(batch, pad_label=-100):
    B = len(batch)
    maxS = max(b["labels"].size(0) for b in batch)
    C = batch[0]["emissions"].size(-1)
    Es, Ls = [], []
    NEG_INF = -1e4
    for b in batch:
        S = b["labels"].size(0)
        padE = torch.cat([b["emissions"], torch.full((maxS - S, C), NEG_INF)], dim=0)
        padL = torch.cat(
            [b["labels"], torch.full((maxS - S,), pad_label, dtype=torch.long)]
        )
        Es.append(padE)
        Ls.append(padL)
    return torch.stack(Es), torch.stack(Ls)


def collate_pages_predict(batch):
    # only pad emissions
    maxS = max(b["emissions"].size(0) for b in batch)
    C = batch[0]["emissions"].size(-1)
    NEG_INF = -1e4

    Es = []
    for b in batch:
        S = b["emissions"].size(0)
        padE = torch.cat([b["emissions"], torch.full((maxS - S, C), NEG_INF)], dim=0)
        Es.append(padE)
    return torch.stack(Es)


# --- DataModule loads XGB and emits emissions ---
class PageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df: pd.DataFrame,
        xgb_path,
        batch_size=8,
        val_split=0.2,
        num_workers=0,
    ):
        super().__init__()
        self.df, self.xgb_path = df, xgb_path
        self.batch_size, self.val_split, self.num_workers = (
            batch_size,
            val_split,
            num_workers,
        )

    def setup(self, stage=None):
        ids = self.df["agreement_uuid"].unique().tolist()
        if stage in ("fit", "validate", None) and self.val_split and self.val_split > 0:
            tr, vl = train_test_split(ids, test_size=self.val_split, random_state=42)
            df_tr = self.df[self.df["agreement_uuid"].isin(tr)]
            df_vl = self.df[self.df["agreement_uuid"].isin(vl)]

            present = set(df_tr["label"])
            labs = [l for l in CLASSIFIER_LABEL_LIST if l in present]
        else:
            # inference mode (or val_split==0): everything goes into “predict_ds”
            df_tr = self.df
            df_vl = self.df.iloc[0:0]

            labs = CLASSIFIER_LABEL_LIST

        self.label2idx = {l: i for i, l in enumerate(labs)}

        # load xgb
        self.xgb = xgb.Booster()
        self.xgb.load_model(self.xgb_path)

        if stage in ("fit", "validate", None) and self.val_split and self.val_split > 0:
            self.train_ds = DocumentDataset(
                df_tr, self.label2idx, self.xgb, inference=False
            )
            self.val_ds = DocumentDataset(
                df_vl, self.label2idx, self.xgb, inference=False
            )

        self.predict_ds = DocumentDataset(
            self.df, self.label2idx, self.xgb, inference=True
        )

        self.num_classes = len(labs)

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

    def predict_dataloader(self):
        return DataLoader(
            self.predict_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_pages_predict,
        )


# --- Classifier uses CRF on emissions ---
class PageClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        lstm_hidden: int = 64,
        lstm_num_layers: int = 1,
        lstm_dropout: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        C = num_classes
        hd = hidden_dim
        dp = dropout

        # transition params
        self.transitions = nn.Parameter(torch.empty(C, C))
        nn.init.xavier_uniform_(self.transitions)
        self.start_transitions = nn.Parameter(torch.empty(C))
        nn.init.normal_(self.start_transitions, 0.0, 0.1)
        self.end_transitions = nn.Parameter(torch.empty(C))
        nn.init.normal_(self.end_transitions, 0.0, 0.1)

        self.mlp = nn.Sequential(
            nn.Linear(C, hd),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(hd, C),
        )

        self.lstm = nn.LSTM(
            input_size=C,
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2proj = nn.Linear(2 * lstm_hidden, C)

        self.idx2label = {idx: label for idx, label in enumerate(CLASSIFIER_LABEL_LIST)}

        # metrics
        self.train_p = Precision(task="multiclass", num_classes=C)
        self.train_r = Recall(task="multiclass", num_classes=C)
        self.train_f = F1Score(task="multiclass", num_classes=C)
        self.val_p = Precision(task="multiclass", num_classes=C)
        self.val_r = Recall(task="multiclass", num_classes=C)
        self.val_f = F1Score(task="multiclass", num_classes=C)

    def forward(self, emissions: Tensor) -> Tensor:
        # # emissions: (B, S, C) coming from PageDataset
        # B, S, C = emissions.size()
        # # flatten to (B*S, C), run through MLP, then reshape
        # flat = emissions.view(-1, C)
        # trans = self.mlp(flat)  # (B*S, C)
        # return trans.view(B, S, C)

        # emissions: (B, S, C)
        B, S, C = emissions.size()
        x = self.mlp(emissions.view(-1, C)).view(B, S, C)  # your MLP
        lstm_out, _ = self.lstm(x)  # (B, S, 2*H)
        emissions = self.lstm2proj(lstm_out)  # back to (B, S, C)
        return emissions

    def _compute_log_likelihood(self, em: Tensor, tags: Tensor, mask: Tensor) -> Tensor:
        B, S, C = em.size()
        tags_s = tags.clone()
        tags_s[~mask] = 0
        # score of true path
        score = self.start_transitions[tags_s[:, 0]] + em[:, 0].gather(
            1, tags_s[:, 0:1]
        ).squeeze(1)
        for t in range(1, S):
            emit_sc = em[:, t].gather(1, tags_s[:, t : t + 1]).squeeze(1)
            trans_sc = self.transitions[tags_s[:, t - 1], tags_s[:, t]]
            score += (emit_sc + trans_sc) * mask[:, t].float()
        seq_lens = mask.sum(1).long() - 1
        last_tags = tags_s.gather(1, seq_lens.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        # partition via forward algorithm
        alphas = self.start_transitions.unsqueeze(0) + em[:, 0]
        for t in range(1, S):
            emit = em[:, t].unsqueeze(2)
            trans = self.transitions.unsqueeze(0)
            scores = alphas.unsqueeze(2) + trans + emit
            new_al = torch.logsumexp(scores, dim=1)
            m_t = mask[:, t].unsqueeze(1).float()
            alphas = new_al * m_t + alphas * (1 - m_t)
        alphas += self.end_transitions.unsqueeze(0)
        partition = torch.logsumexp(alphas, dim=1)
        return (score - partition).mean()

    def _viterbi_decode(self, em: Tensor, mask: Tensor) -> Tensor:
        B, S, C = em.size()
        seq_lens = mask.sum(1).long()
        v_score = self.start_transitions.unsqueeze(0) + em[:, 0]
        backptrs = []
        for t in range(1, S):
            b_sc = v_score.unsqueeze(2) + self.transitions.unsqueeze(0)
            max_sc, idxs = b_sc.max(dim=1)
            v_score = max_sc + em[:, t]
            backptrs.append(idxs)
        v_score += self.end_transitions.unsqueeze(0)
        last_tag = v_score.argmax(dim=1)
        paths = []
        for b in range(B):
            L = seq_lens[b].item()
            tag = last_tag[b].item()
            seq = [tag]
            for ptr in reversed(backptrs[: L - 1]):
                tag = ptr[b, tag].item()
                seq.append(tag)
            seq.reverse()
            if L < S:
                seq += [0] * (S - L)
            paths.append(seq)
        return torch.tensor(paths, device=em.device)

    def shared_step(self, batch, prefix: str):
        emissions, labels = batch
        mask = labels != -100
        ll = self._compute_log_likelihood(emissions, labels, mask)
        loss = -ll
        preds = self._viterbi_decode(emissions, mask)[mask]
        labs = labels[mask]
        if prefix == "train":
            self.train_p(preds, labs)
            self.train_r(preds, labs)
            self.train_f(preds, labs)
        else:
            self.val_p(preds, labs)
            self.val_r(preds, labs)
            self.val_f(preds, labs)
        self.log(f"{prefix}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def predict_step(self, batch, batch_idx):
        # batch is just the emissions tensor when you're in predict()
        # shape = (B, S, C)
        emissions = batch

        # rebuild the mask of “real” tokens/pages by spotting the padding value
        NEG_INF = -1e4
        # mask: (B, S) = True where at least one class prob ≠ NEG_INF
        mask = (emissions != NEG_INF).any(dim=-1)

        # 1) compute per‐class probs
        probs = F.softmax(emissions, dim=-1)
        # 2) flatten to (N, C) over all real positions
        probs_flat = probs[mask]

        # 3) viterbi decode → (B, S, C) to (B, S) → flatten to (N,)
        preds = self._viterbi_decode(emissions, mask)[mask]

        # 4) get label names
        n_classes = emissions.size(-1)
        if hasattr(self, "idx2label"):
            label_names = [self.idx2label[i] for i in range(n_classes)]
        else:
            label_names = list(range(n_classes))

        # 5) assemble list of dicts
        out = []
        for p_vec, p_idx in zip(probs_flat.tolist(), preds.tolist()):
            entry = {label_names[i]: p_vec[i] for i in range(n_classes)}
            entry["pred_class"] = label_names[p_idx]
            out.append(entry)

        return out

    def on_train_epoch_end(self):
        self.log("train_f1", self.train_f.compute(), prog_bar=True)
        self.log("train_precision", self.train_p.compute())
        self.log("train_recall", self.train_r.compute())
        self.train_f.reset()
        self.train_p.reset()
        self.train_r.reset()

    def on_validation_epoch_end(self):
        self.log("val_f1", self.val_f.compute(), prog_bar=True)
        self.log("val_precision", self.val_p.compute())
        self.log("val_recall", self.val_r.compute())
        self.val_f.reset()
        self.val_p.reset()
        self.val_r.reset()

    def configure_optimizers(self):
        return AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
