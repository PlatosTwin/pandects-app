"""
Page classification models and datasets.

This module contains PyTorch Lightning modules and datasets for page classification
using CRF (Conditional Random Fields) on top of XGBoost emissions.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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

from .classifier_utils import extract_features
from .shared_constants import CLASSIFIER_LABEL_LIST


class PageDataset(Dataset):
    """
    Dataset for individual page classification.
    
    Converts raw page data into XGBoost emissions for CRF training.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        label2idx: Dict[str, int],
        model: xgb.Booster,
        inference: bool = False,
    ):
        """
        Initialize the page dataset.
        
        Args:
            df: DataFrame with 'text', 'html', 'order', and optionally 'label' columns
            label2idx: Mapping from label strings to indices
            model: Pre-trained XGBoost model for feature extraction
            inference: If True, skip label loading for inference
        """
        # Handle missing values
        df = df.fillna({"text": "", "html": "", "order": 0})
        
        # Extract features and get XGBoost predictions
        features = np.vstack(
            [
                extract_features(text, html, order)
                for text, html, order in zip(df["text"], df["html"], df["order"])
            ]
        )
        dmatrix = xgb.DMatrix(features)
        probabilities = model.predict(dmatrix)  # Shape: (N, C)
        
        # Convert to log probabilities, avoiding log(0) → -inf
        probs_tensor = torch.tensor(probabilities, dtype=torch.float)
        probs_tensor = probs_tensor.clamp(min=1e-8)
        self.emissions = torch.log(probs_tensor)

        if not inference:
            self.labels = torch.tensor(
                df["label"].map(label2idx).values, dtype=torch.long
            )
        else:
            self.labels = None

    def __len__(self) -> int:
        return self.emissions.size(0)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        if self.labels is None:
            return {"emissions": self.emissions[idx]}
        return {"emissions": self.emissions[idx], "labels": self.labels[idx]}


class DocumentDataset(Dataset):
    """
    Dataset for document-level page classification.
    
    Groups pages by document and provides document-level sequences.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        label2idx: Dict[str, int],
        model: xgb.Booster,
        inference: bool = False,
    ):
        """
        Initialize the document dataset.
        
        Args:
            df: DataFrame with page data including 'agreement_uuid' column
            label2idx: Mapping from label strings to indices
            model: Pre-trained XGBoost model
            inference: If True, skip label loading for inference
        """
        self.page_dataset = PageDataset(df, label2idx, model, inference=inference)
        self.document_indices = []
        
        # Group pages by document
        document_groups = defaultdict(list)
        for i, agreement_uuid in enumerate(df["agreement_uuid"]):
            document_groups[agreement_uuid].append(i)
        
        # Store sorted page indices for each document
        for page_indices in document_groups.values():
            # Order is preserved by DataFrame
            self.document_indices.append(sorted(page_indices))

        self.inference = inference

    def __len__(self) -> int:
        return len(self.document_indices)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        page_indices = self.document_indices[idx]
        emissions = torch.stack([self.page_dataset[i]["emissions"] for i in page_indices])

        if self.inference:
            return {"emissions": emissions}

        labels = torch.tensor([self.page_dataset[i]["labels"] for i in page_indices], dtype=torch.long)
        return {"emissions": emissions, "labels": labels}


def collate_pages(batch: List[Dict[str, Tensor]], pad_label: int = -100) -> Tuple[Tensor, Tensor]:
    """
    Collate function for training/validation batches.
    
    Pads emissions and labels to the same sequence length.
    
    Args:
        batch: List of dictionaries with 'emissions' and 'labels' keys
        pad_label: Label value to use for padding
        
    Returns:
        Tuple of (padded_emissions, padded_labels)
    """
    batch_size = len(batch)
    max_seq_len = max(b["labels"].size(0) for b in batch)
    num_classes = batch[0]["emissions"].size(-1)
    
    padded_emissions, padded_labels = [], []
    NEG_INF = -1e4
    
    for b in batch:
        seq_len = b["labels"].size(0)
        # Pad emissions with negative infinity
        pad_emissions = torch.cat([b["emissions"], torch.full((max_seq_len - seq_len, num_classes), NEG_INF)], dim=0)
        # Pad labels with pad_label
        pad_labels = torch.cat(
            [b["labels"], torch.full((max_seq_len - seq_len,), pad_label, dtype=torch.long)]
        )
        padded_emissions.append(pad_emissions)
        padded_labels.append(pad_labels)
    
    return torch.stack(padded_emissions), torch.stack(padded_labels)


def collate_pages_predict(batch: List[Dict[str, Tensor]]) -> Tensor:
    """
    Collate function for prediction batches.
    
    Only pads emissions since labels are not available during inference.
    
    Args:
        batch: List of dictionaries with 'emissions' key
        
    Returns:
        Padded emissions tensor
    """
    max_seq_len = max(b["emissions"].size(0) for b in batch)
    num_classes = batch[0]["emissions"].size(-1)
    NEG_INF = -1e4

    padded_emissions = []
    for b in batch:
        seq_len = b["emissions"].size(0)
        pad_emissions = torch.cat([b["emissions"], torch.full((max_seq_len - seq_len, num_classes), NEG_INF)], dim=0)
        padded_emissions.append(pad_emissions)
    
    return torch.stack(padded_emissions)


class PageDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for page classification.
    
    Handles data loading, train/val splitting, and XGBoost model loading.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        xgb_path: str,
        batch_size: int = 8,
        val_split: float = 0.2,
        num_workers: int = 0,
    ):
        """
        Initialize the data module.
        
        Args:
            df: DataFrame with page data
            xgb_path: Path to pre-trained XGBoost model
            batch_size: Batch size for training
            val_split: Fraction of data to use for validation
            num_workers: Number of workers for data loading
        """
        super().__init__()
        self.df = df
        self.xgb_path = xgb_path
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets for training, validation, and prediction.
        
        Args:
            stage: Lightning stage ('fit', 'validate', 'predict', or None)
        """
        agreement_ids = self.df["agreement_uuid"].unique().tolist()
        
        if stage in ("fit", "validate", None) and self.val_split and self.val_split > 0:
            # Split by agreement to keep documents together
            train_ids, val_ids = train_test_split(agreement_ids, test_size=self.val_split, random_state=42)
            train_df = self.df[self.df["agreement_uuid"].isin(train_ids)]
            val_df = self.df[self.df["agreement_uuid"].isin(val_ids)]

            # Use only labels present in training data
            present_labels = set(train_df["label"])
            labels = [label for label in CLASSIFIER_LABEL_LIST if label in present_labels]
        else:
            # Inference mode: use all data for prediction
            train_df = self.df
            val_df = self.df.iloc[0:0]  # Empty DataFrame
            labels = CLASSIFIER_LABEL_LIST

        self.label2idx = {label: idx for idx, label in enumerate(labels)}

        # Load pre-trained XGBoost model
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(self.xgb_path)

        if stage in ("fit", "validate", None) and self.val_split and self.val_split > 0:
            self.train_dataset = DocumentDataset(
                train_df, self.label2idx, self.xgb_model, inference=False
            )
            self.val_dataset = DocumentDataset(
                val_df, self.label2idx, self.xgb_model, inference=False
            )

        self.predict_dataset = DocumentDataset(
            self.df, self.label2idx, self.xgb_model, inference=True
        )

        self.num_classes = len(labels)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_pages,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_pages,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_pages_predict,
        )


class PageClassifier(pl.LightningModule):
    """
    Page classifier using CRF (Conditional Random Fields) on XGBoost emissions.
    
    Combines XGBoost feature extraction with CRF for sequence modeling.
    """
    
    def __init__(
        self,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        lstm_hidden: int = 64,
        lstm_num_layers: int = 1,
        lstm_dropout: float = 0.0,
    ):
        """
        Initialize the page classifier.
        
        Args:
            num_classes: Number of classification classes
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            hidden_dim: Hidden dimension for MLP
            dropout: Dropout rate for MLP
            lstm_hidden: Hidden dimension for LSTM
            lstm_num_layers: Number of LSTM layers
            lstm_dropout: Dropout rate for LSTM
        """
        super().__init__()
        self.save_hyperparameters()
        
        num_classes = num_classes
        hidden_dim = hidden_dim
        dropout = dropout

        # CRF transition parameters
        self.transitions = nn.Parameter(torch.empty(num_classes, num_classes))
        nn.init.xavier_uniform_(self.transitions)
        self.start_transitions = nn.Parameter(torch.empty(num_classes))
        nn.init.normal_(self.start_transitions, 0.0, 0.1)
        self.end_transitions = nn.Parameter(torch.empty(num_classes))
        nn.init.normal_(self.end_transitions, 0.0, 0.1)

        # MLP for emission refinement
        self.mlp = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Bidirectional LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=num_classes,
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2proj = nn.Linear(2 * lstm_hidden, num_classes)

        # Label mappings
        self.idx2label = {idx: label for idx, label in enumerate(CLASSIFIER_LABEL_LIST)}

        # Illegal transition mask (prevents backward transitions)
        label2idx = {label: idx for idx, label in enumerate(CLASSIFIER_LABEL_LIST)}
        illegal_mask = torch.zeros(num_classes, num_classes, dtype=torch.bool)
        for _, idx in label2idx.items():
            for _, jdx in label2idx.items():
                if jdx < idx:
                    illegal_mask[idx, jdx] = True
        self.register_buffer("illegal_transitions", illegal_mask)

        # Metrics
        self.train_precision = Precision(task="multiclass", num_classes=num_classes)
        self.train_recall = Recall(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.val_precision = Precision(task="multiclass", num_classes=num_classes)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, emissions: Tensor) -> Tensor:
        """
        Forward pass through the model.
        
        Args:
            emissions: Input emissions tensor of shape (B, S, C)
            
        Returns:
            Refined emissions tensor of shape (B, S, C)
        """
        batch_size, seq_len, num_classes = emissions.size()
        
        # Apply MLP to refine emissions
        x = self.mlp(emissions.view(-1, num_classes)).view(batch_size, seq_len, num_classes)
        
        # Apply bidirectional LSTM
        lstm_out, _ = self.lstm(x)  # Shape: (B, S, 2*H)
        emissions = self.lstm2proj(lstm_out)  # Back to (B, S, C)
        
        return emissions

    def _compute_log_likelihood(self, emissions: Tensor, tags: Tensor, mask: Tensor) -> Tensor:
        """
        Compute log likelihood for CRF training.
        
        Args:
            emissions: Emission scores (B, S, C)
            tags: Gold tag sequences (B, S)
            mask: Sequence mask (B, S)
            
        Returns:
            Average log likelihood
        """
        batch_size, seq_len, num_classes = emissions.size()

        # Score the gold path
        base_transitions = self.transitions.masked_fill(self.illegal_transitions, -1e4)
        tags_safe = tags.clone()
        tags_safe[~mask] = 0
        
        # Initial score
        score = self.start_transitions[tags_safe[:, 0]] + emissions[:, 0].gather(
            1, tags_safe[:, 0:1]
        ).squeeze(1)
        
        # Transition scores
        for t in range(1, seq_len):
            emission_score = emissions[:, t].gather(1, tags_safe[:, t : t + 1]).squeeze(1)
            transition_score = base_transitions[tags_safe[:, t - 1], tags_safe[:, t]]
            score += (emission_score + transition_score) * mask[:, t].float()
        
        # Final transition
        last_indices = (mask.sum(1).long() - 1).unsqueeze(1)
        last_tags = tags_safe.gather(1, last_indices).squeeze(1)
        score += self.end_transitions[last_tags]

        # Compute partition function via forward algorithm
        alphas = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        trans_broadcast = base_transitions.unsqueeze(0)

        for t in range(1, seq_len):
            emission = emissions[:, t].unsqueeze(2)  # (B, C, 1)
            scores = alphas.unsqueeze(2) + trans_broadcast + emission
            new_alphas = torch.logsumexp(scores, dim=1)  # (B, C)
            mask_t = mask[:, t].unsqueeze(1)  # (B, 1)
            alphas = torch.where(mask_t, new_alphas, alphas)

        # Final partition
        alphas += self.end_transitions.unsqueeze(0)  # (B, C)
        partition = torch.logsumexp(alphas, dim=1)  # (B)

        return (score - partition).mean()

    def _viterbi_decode(self, emissions: Tensor, mask: Tensor) -> Tensor:
        """
        Viterbi decoding for inference.
        
        Args:
            emissions: Emission scores (B, S, C)
            mask: Sequence mask (B, S)
            
        Returns:
            Decoded tag sequences (B, S)
        """
        base_transitions = self.transitions.masked_fill(self.illegal_transitions, -1e4)
        trans_broadcast = base_transitions.unsqueeze(0)  # (1, C, C)

        batch_size, seq_len, num_classes = emissions.size()
        seq_lengths = mask.sum(1).long()  # (B)

        # Initial scores
        viterbi_scores = [self.start_transitions.unsqueeze(0) + emissions[:, 0]]
        backpointers = []

        # Forward pass
        for t in range(1, seq_len):
            prev_scores = viterbi_scores[-1]  # (B, C)
            jump_scores = prev_scores.unsqueeze(2) + trans_broadcast  # (B, C, C)
            max_scores, indices = jump_scores.max(dim=1)  # Both (B, C)
            candidate_scores = max_scores + emissions[:, t]  # (B, C)
            mask_t = mask[:, t].unsqueeze(1)  # (B, 1)
            viterbi_t = torch.where(mask_t, candidate_scores, prev_scores)
            backpointers.append(indices)
            viterbi_scores.append(viterbi_t)

        # Backtrack
        paths = []
        for b in range(batch_size):
            true_length = seq_lengths[b].item()
            last_state = viterbi_scores[true_length - 1][b].argmax().item()
            sequence = [last_state]
            
            # Walk backwards through backpointers
            for ptr in reversed(backpointers[: true_length - 1]):
                last_state = ptr[b, last_state].item()
                sequence.append(last_state)
            sequence.reverse()
            
            # Pad to sequence length
            if true_length < seq_len:
                sequence += [0] * (seq_len - true_length)
            paths.append(sequence)

        return torch.tensor(paths, device=emissions.device)

    def shared_step(self, batch: Tuple[Tensor, Tensor], prefix: str) -> Tensor:
        """
        Shared training/validation step.
        
        Args:
            batch: Tuple of (emissions, labels)
            prefix: Step prefix ('train' or 'val')
            
        Returns:
            Loss value
        """
        emissions, labels = batch
        mask = labels != -100
        
        # Compute loss
        log_likelihood = self._compute_log_likelihood(emissions, labels, mask)
        loss = -log_likelihood
        
        # Compute predictions and metrics
        predictions = self._viterbi_decode(emissions, mask)[mask]
        true_labels = labels[mask]
        
        if prefix == "train":
            self.train_precision(predictions, true_labels)
            self.train_recall(predictions, true_labels)
            self.train_f1(predictions, true_labels)
        else:
            self.val_precision(predictions, true_labels)
            self.val_recall(predictions, true_labels)
            self.val_f1(predictions, true_labels)
        
        self.log(f"{prefix}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self.shared_step(batch, "train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self.shared_step(batch, "val")

    def predict_step(self, batch: Tensor, batch_idx: int) -> List[Dict[str, float]]:
        """
        Prediction step for inference.
        
        Args:
            batch: Emissions tensor (B, S, C)
            batch_idx: Batch index
            
        Returns:
            List of prediction dictionaries with class probabilities
        """
        emissions = batch

        # Rebuild mask from padding values
        NEG_INF = -1e4
        mask = (emissions != NEG_INF).any(dim=-1)

        # Compute class probabilities
        probabilities = F.softmax(emissions, dim=-1)
        probabilities_flat = probabilities[mask]

        # Viterbi decode
        predictions = self._viterbi_decode(emissions, mask)[mask]

        # Get label names
        num_classes = emissions.size(-1)
        if hasattr(self, "idx2label"):
            label_names = [self.idx2label[i] for i in range(num_classes)]
        else:
            label_names = list(range(num_classes))

        # Assemble results
        results = []
        for prob_vector, pred_idx in zip(probabilities_flat.tolist(), predictions.tolist()):
            entry = {label_names[i]: prob_vector[i] for i in range(num_classes)}
            entry["pred_class"] = label_names[pred_idx]
            results.append(entry)

        return results

    def on_train_epoch_end(self) -> None:
        """Log training metrics at epoch end."""
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        self.log("train_precision", self.train_precision.compute())
        self.log("train_recall", self.train_recall.compute())
        self.train_f1.reset()
        self.train_precision.reset()
        self.train_recall.reset()

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics at epoch end."""
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.log("val_precision", self.val_precision.compute())
        self.log("val_recall", self.val_recall.compute())
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def configure_optimizers(self) -> AdamW:
        """Configure optimizer."""
        return AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
