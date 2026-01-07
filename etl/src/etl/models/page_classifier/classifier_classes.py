"""
Page classification models and datasets.

This module contains PyTorch Lightning modules and datasets for page classification
using CRF (Conditional Random Fields) on top of XGBoost emissions.
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false

from functools import lru_cache
from typing import cast

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchmetrics import F1Score, Precision, Recall, Metric
import xgboost as xgb
from joblib import Parallel, delayed

try:
    from .classifier_utils import extract_features
    from .page_classifier_constants import CLASSIFIER_LABEL_LIST
    from .split_utils import build_agreement_split, load_split_manifest
except ImportError:  # pragma: no cover - supports running as a script
    from classifier_utils import extract_features  # pyright: ignore[reportMissingImports]
    from page_classifier_constants import CLASSIFIER_LABEL_LIST  # pyright: ignore[reportMissingImports]
    from split_utils import build_agreement_split, load_split_manifest  # pyright: ignore[reportMissingImports]


Prediction = dict[str, str | dict[str, float] | bool]


@lru_cache(maxsize=2)
def load_xgb_model(xgb_path: str) -> xgb.Booster:
    """
    Load XGBoost model with caching to prevent redundant loading.
    
    Args:
        xgb_path: Path to the XGBoost model file
        
    Returns:
        Loaded XGBoost Booster model
        
    Note:
        Uses LRU cache to avoid reloading the same model multiple times.
        Cache size of 2 allows for train/test model variants.
    """
    model = xgb.Booster()
    model.load_model(xgb_path)
    return model


class PageDataset(Dataset[dict[str, Tensor]]):
    """
    Dataset for individual page classification.

    Converts raw page data into XGBoost emissions for CRF training.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label2idx: dict[str, int],
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
        # Handle missing values safely
        df = df.fillna({"text": "", "html": "", "order": 0})

        # Extract features using parallel processing for better performance
        features_list = cast(
            list[np.ndarray],
            list(
                Parallel(n_jobs=-1, prefer="threads")(
                    delayed(extract_features)(text, html, order)
                    for text, html, order in zip(df["text"], df["html"], df["order"])
                )
            ),
        )
        features = np.vstack(features_list)
        dmatrix = xgb.DMatrix(features)
        probabilities = model.predict(dmatrix)  # Shape: (N, C)

        # Convert to log probabilities, avoiding log(0) → -inf
        probs_tensor = torch.tensor(np.clip(probabilities, 1e-12, 1.0), dtype=torch.float)
        self.emissions: torch.Tensor = torch.log(probs_tensor)

        # Handle labels for training
        if not inference and "label" in df.columns:
            self.labels: torch.Tensor | None = torch.tensor(
                [label2idx.get(label, 0) for label in df["label"]], dtype=torch.long
            )
        else:
            self.labels = None

    def __len__(self) -> int:
        return self.emissions.size(0)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get emissions and optionally labels for a single page."""
        if self.labels is None:
            return {"emissions": self.emissions[idx]}
        return {"emissions": self.emissions[idx], "labels": self.labels[idx]}


class DocumentDataset(Dataset[dict[str, Tensor]]):
    """
    Dataset for document-level page classification.

    Groups pages by document and provides document-level sequences.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label2idx: dict[str, int],
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
        self.page_dataset: PageDataset = PageDataset(
            df, label2idx, model, inference=inference
        )
        self.document_indices: list[list[int]] = []

        # Group pages by document and sort by order
        df = df.copy()  # Make a copy to avoid modifying original
        df['idx'] = range(len(df))  # Add index column
        # Group by agreement and sort by order within each group, excluding grouping columns
        grouped = df.groupby('agreement_uuid', sort=False, group_keys=False).apply(  # pyright: ignore[reportCallIssue]
            lambda x: list(x.sort_values('order')['idx']),
            include_groups=False,  # pyright: ignore[reportCallIssue]
        )
        self.document_indices = grouped.tolist()
        self.inference: bool = inference

    def __len__(self) -> int:
        return len(self.document_indices)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get a document's worth of pages, properly ordered."""
        page_indices = self.document_indices[idx]
        emissions = torch.stack(
            [self.page_dataset[i]["emissions"] for i in page_indices]
        )

        if self.inference:
            return {"emissions": emissions}

        labels = torch.stack(
            [self.page_dataset[i]["labels"] for i in page_indices]
        ).long()
        return {"emissions": emissions, "labels": labels}


def collate_pages(
    batch: list[dict[str, Tensor]], pad_label: int = -100
) -> tuple[Tensor, Tensor]:
    """
    Collate function for training/validation batches.

    Pads emissions and labels to the same sequence length.

    Args:
        batch: List of dictionaries with 'emissions' and 'labels' keys
        pad_label: Label value to use for padding

    Returns:
        Tuple of (padded_emissions, padded_labels)
    """
    max_seq_len = max(b["labels"].size(0) for b in batch)
    num_classes = batch[0]["emissions"].size(-1)

    padded_emissions, padded_labels = [], []
    NEG_INF = -1e4

    for b in batch:
        seq_len = b["labels"].size(0)
        # Pad emissions with negative infinity
        pad_emissions = torch.cat(
            [b["emissions"], torch.full((max_seq_len - seq_len, num_classes), NEG_INF)],
            dim=0,
        )
        # Pad labels with pad_label
        pad_labels = torch.cat(
            [
                b["labels"],
                torch.full((max_seq_len - seq_len,), pad_label, dtype=torch.long),
            ]
        )
        padded_emissions.append(pad_emissions)
        padded_labels.append(pad_labels)

    return torch.stack(padded_emissions), torch.stack(padded_labels)


def collate_pages_predict(batch: list[dict[str, Tensor]]) -> tuple[Tensor, Tensor]:
    """
    Collate function for prediction batches.

    Only pads emissions since labels are not available during inference.

    Args:
        batch: List of dictionaries with 'emissions' key

    Returns:
        Tuple of (padded_emissions, mask)
    """
    max_seq_len = max(b["emissions"].size(0) for b in batch)
    num_classes = batch[0]["emissions"].size(-1)
    NEG_INF = -1e4

    padded_emissions = []
    masks = []
    for b in batch:
        seq_len = b["emissions"].size(0)
        pad_emissions = torch.cat(
            [b["emissions"], torch.full((max_seq_len - seq_len, num_classes), NEG_INF)],
            dim=0,
        )
        padded_emissions.append(pad_emissions)
        mask = torch.zeros((max_seq_len,), dtype=torch.bool)
        mask[:seq_len] = True
        masks.append(mask)

    return torch.stack(padded_emissions), torch.stack(masks)


class PageDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for page classification.

    Handles data loading, deterministic year-stratified splits, and XGBoost model loading.
    If split_path is provided, uses the shared agreement split manifest.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        xgb_path: str,
        batch_size: int = 8,
        val_split: float = 0.2,
        test_split: float = 0.0,
        num_workers: int = 0,
        split_path: str | None = None,
        length_bucket_edges: list[float] | None = None,
        back_matter_bucket_edges: list[float] | None = None,
    ):
        """
        Initialize the data module.

        Args:
            df: DataFrame with page data
            xgb_path: Path to pre-trained XGBoost model
            batch_size: Batch size for training
            val_split: Fraction of data to use for validation
            test_split: Fraction of data to use for test
            num_workers: Number of workers for data loading
            split_path: Optional path to agreement split manifest
            length_bucket_edges: Fixed edges for agreement length buckets
            back_matter_bucket_edges: Fixed edges for back matter page buckets
        """
        super().__init__()
        required_columns = {"text", "html", "order", "agreement_uuid"}
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self.df = df.copy()  # Defensive copy
        # Fill missing values appropriately
        self.df["text"] = self.df["text"].fillna("")
        self.df["html"] = self.df["html"].fillna("")
        self.df["order"] = self.df["order"].fillna(0)
        self.df["agreement_uuid"] = self.df["agreement_uuid"].fillna("UNKNOWN")
        
        self.xgb_path = xgb_path
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.split_path = split_path
        self.length_bucket_edges = length_bucket_edges
        self.back_matter_bucket_edges = back_matter_bucket_edges
        self.label2idx: dict[str, int] = {}
        self.xgb_model: xgb.Booster | None = None
        self.train_dataset: DocumentDataset | None = None
        self.val_dataset: DocumentDataset | None = None
        self.test_dataset: DocumentDataset | None = None
        self.predict_dataset: DocumentDataset | None = None
        self.num_classes: int = 0

    def setup(self, stage: str | None = None) -> None:
        """
        Set up datasets for training, validation, and prediction.

        Args:
            stage: Lightning stage ('fit', 'validate', 'test', 'predict', or None)
        """
        agreement_ids = self.df["agreement_uuid"].unique().tolist()

        needs_split = (
            stage in ("fit", "validate", "test", None)
            and (self.val_split > 0 or self.test_split > 0)
        )
        if needs_split:
            if self.split_path:
                split = load_split_manifest(self.split_path)
                train_ids = split.get("train", [])
                val_ids = split.get("val", [])
                test_ids = split.get("test", [])
            else:
                split = build_agreement_split(
                    self.df,
                    val_split=self.val_split,
                    test_split=self.test_split,
                    agreement_col="agreement_uuid",
                    date_col="date_announcement",
                    length_bucket_edges=self.length_bucket_edges,
                    back_matter_bucket_edges=self.back_matter_bucket_edges,
                )
                train_ids = split.get("train", [])
                val_ids = split.get("val", [])
                test_ids = split.get("test", [])

            df_agreement_ids = set(map(str, agreement_ids))
            split_ids = set(map(str, train_ids)) | set(map(str, val_ids)) | set(map(str, test_ids))
            missing_ids = split_ids - df_agreement_ids
            if missing_ids:
                raise ValueError("Split manifest contains unknown agreement_uuid values.")

            train_df = self.df[self.df["agreement_uuid"].isin(train_ids)]
            val_df = self.df[self.df["agreement_uuid"].isin(val_ids)]
            test_df = self.df[self.df["agreement_uuid"].isin(test_ids)]

            # Use only labels present in training data
            present_labels = set(train_df["label"])
            extra_labels: set[str] = set()
            if self.val_split > 0:
                extra_labels |= set(val_df["label"]) - present_labels
            if self.test_split > 0:
                extra_labels |= set(test_df["label"]) - present_labels
            if extra_labels:
                raise ValueError(
                    f"Validation/test labels missing from training split: {sorted(extra_labels)}"
                )
            labels = [
                label for label in CLASSIFIER_LABEL_LIST if label in present_labels
            ]
        else:
            # Inference mode: use all data for prediction
            train_df = self.df
            val_df = self.df.iloc[0:0]  # Empty DataFrame
            test_df = self.df.iloc[0:0]  # Empty DataFrame
            labels = CLASSIFIER_LABEL_LIST

        self.label2idx = {label: idx for idx, label in enumerate(labels)}

        # Load pre-trained XGBoost model using cached loader
        self.xgb_model = load_xgb_model(self.xgb_path)

        if stage in ("fit", "validate", "test", None) and (
            self.val_split > 0 or self.test_split > 0
        ):
            self.train_dataset = DocumentDataset(
                train_df, self.label2idx, self.xgb_model, inference=False
            )
            if self.val_split > 0:
                self.val_dataset = DocumentDataset(
                    val_df, self.label2idx, self.xgb_model, inference=False
                )
            if self.test_split > 0:
                self.test_dataset = DocumentDataset(
                    test_df, self.label2idx, self.xgb_model, inference=False
                )

        self.predict_dataset = DocumentDataset(
            self.df, self.label2idx, self.xgb_model, inference=True
        )

        self.num_classes = len(labels)
    

    def train_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Create training data loader with document-level batching."""
        if self.train_dataset is None:
            raise RuntimeError("Training dataset is not initialized. Call setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_pages,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Create validation data loader with document-level batching."""
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset is not initialized. Call setup() first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_pages,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Create test data loader with document-level batching."""
        if self.test_dataset is None:
            raise RuntimeError("Test dataset is not initialized. Call setup() first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_pages,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Create prediction data loader for inference."""
        if self.predict_dataset is None:
            raise RuntimeError("Prediction dataset is not initialized. Call setup() first.")
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_pages_predict,
            persistent_workers=self.num_workers > 0,
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
        use_lstm: bool = True,
        use_crf: bool = True,
        *,
        label_names: list[str] | None = None,
        sig_label: str = "sig",
        back_label: str = "back_matter",
        enforce_single_sig_block: bool = True,
        # Positional decoding preferences
        prefer_earliest_sig: bool = True,
        sig_late_penalty: float = 1.5,
        sig_penalty_center: float = 0.85,
        sig_penalty_sharpness: float = 12.0,
        back_late_bonus: float = 0.5,
        back_bonus_center: float = 0.9,
        back_bonus_sharpness: float = 10.0,
        # Auxiliary training signal to learn first signature onset
        aux_sig_start_loss_weight: float = 1.0,
        # Auxiliary training signal to learn back_matter onset (tail)
        aux_back_start_loss_weight: float = 0.1,
        # Learned positional prior (per-position per-class bias)
        use_positional_prior: bool = True,
        pos_prior_weight: float = 0.8,
        pos_prior_hidden: int = 32,
        # Decode-time signature block postprocessing
        enable_first_sig_postprocessing: bool = True,
        first_sig_threshold: float = 0.3,
    ):
        """
        Initialize the PageClassifier with CRF on top of LSTM.
        
        Args:
            num_classes: Number of page classification classes (must be > 0)
            lr: Learning rate for optimizer (must be in (0, 1))
            weight_decay: Weight decay for regularization (must be >= 0)
            hidden_dim: Hidden layer dimension for MLP (must be > 0)
            dropout: Dropout probability for MLP layers (must be in [0, 1])
            lstm_hidden: Hidden dimension for LSTM (must be > 0)
            lstm_num_layers: Number of LSTM layers (must be > 0)
            lstm_dropout: Dropout probability for LSTM (must be in [0, 1])
            use_lstm: Whether to apply the MLP+LSTM residual branch
            use_crf: Whether to use CRF decoding (otherwise per-page argmax)
            label_names: List of label names corresponding to class indices
            sig_label: Name of signature label for CRF constraints
            back_label: Name of back matter label for CRF constraints
            enforce_single_sig_block: Whether to enforce single signature block constraint
            prefer_earliest_sig: Whether to prefer earliest signature blocks via positional bias
            sig_late_penalty: Penalty factor for late signature blocks
            sig_penalty_center: Center point for signature penalty (fraction of sequence)
            sig_penalty_sharpness: Sharpness of signature penalty function
            back_late_bonus: Bonus factor for late back matter blocks
            back_bonus_center: Center point for back matter bonus (fraction of sequence)
            back_bonus_sharpness: Sharpness of back matter bonus function
            aux_sig_start_loss_weight: Auxiliary loss weight for signature onset detection
            aux_back_start_loss_weight: Auxiliary loss weight for back matter onset detection
            use_positional_prior: Whether to use learned positional bias
            pos_prior_weight: Weight for positional prior
            pos_prior_hidden: Hidden dimension for positional prior MLP
            enable_first_sig_postprocessing: Whether to apply decode-time first signature block fix
            first_sig_threshold: Probability threshold for signature block detection in postprocessing
            
        Raises:
            ValueError: If any parameter is outside its valid range
        """
        super().__init__()
        # Save hyperparameters with type validation
        self.save_hyperparameters()
        
        # Validate numerical parameters
        if not (0 <= dropout <= 1):
            raise ValueError(f"dropout must be in [0,1], got {dropout}")
        if not (0 <= lstm_dropout <= 1):
            raise ValueError(f"lstm_dropout must be in [0,1], got {lstm_dropout}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if lstm_hidden <= 0:
            raise ValueError(f"lstm_hidden must be positive, got {lstm_hidden}")
        if lstm_num_layers <= 0:
            raise ValueError(f"lstm_num_layers must be positive, got {lstm_num_layers}")
            
        # Initialize metrics
        metrics: dict[str, Metric] = {}
        for split in ["train", "val"]:
            metrics[f"{split}_precision"] = Precision(
                task="multiclass", num_classes=num_classes, average="macro"
            )
            metrics[f"{split}_recall"] = Recall(
                task="multiclass", num_classes=num_classes, average="macro"
            )
            metrics[f"{split}_f1"] = F1Score(
                task="multiclass", num_classes=num_classes, average="macro"
            )
        self.metrics: torch.nn.ModuleDict = torch.nn.ModuleDict(metrics)
        
        # Core model components
        self.C = num_classes  # number of classes
        self.hidden = torch.nn.Sequential(
            torch.nn.Linear(num_classes, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        
        self.lstm = torch.nn.LSTM(
            hidden_dim,
            lstm_hidden,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True,
        )
        
        lstm_out_dim = lstm_hidden * 2  # bidirectional
        self.proj = torch.nn.Linear(lstm_out_dim, num_classes)
        self.use_lstm = bool(use_lstm)
        self.use_crf = bool(use_crf)
        
        # Handle special labels and constraints
        self.label_names = label_names or CLASSIFIER_LABEL_LIST
        if len(self.label_names) != num_classes:
            raise ValueError(f"Label names ({len(self.label_names)}) != num_classes ({num_classes})")
            
        self.sig_idx = self.label_names.index(sig_label) if sig_label in self.label_names else None
        self.back_idx = self.label_names.index(back_label) if back_label in self.label_names else None
        
        self.enforce_single_sig_block = enforce_single_sig_block
        # Decoding bias knobs
        self.prefer_earliest_sig = prefer_earliest_sig
        self.sig_late_penalty = float(sig_late_penalty)
        self.sig_penalty_center = float(sig_penalty_center)
        self.sig_penalty_sharpness = float(sig_penalty_sharpness)
        self.back_late_bonus = float(back_late_bonus)
        self.back_bonus_center = float(back_bonus_center)
        self.back_bonus_sharpness = float(back_bonus_sharpness)
        # Aux onset loss
        if aux_sig_start_loss_weight < 0:
            raise ValueError("aux_sig_start_loss_weight must be >= 0")
        self.aux_sig_start_loss_weight = float(aux_sig_start_loss_weight)
        self.sig_start_head = torch.nn.Linear(lstm_out_dim, 1)
        if aux_back_start_loss_weight < 0:
            raise ValueError("aux_back_start_loss_weight must be >= 0")
        self.aux_back_start_loss_weight = float(aux_back_start_loss_weight)
        self.back_start_head = torch.nn.Linear(lstm_out_dim, 1)

        # Learned positional prior
        self.use_positional_prior = bool(use_positional_prior)
        self.pos_prior_weight = float(pos_prior_weight)
        if pos_prior_hidden <= 0:
            raise ValueError("pos_prior_hidden must be > 0")
        self.pos_prior_mlp = torch.nn.Sequential(
            torch.nn.Linear(6, pos_prior_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(pos_prior_hidden, num_classes),
        )
        
        # Store postprocessing parameters
        self.enable_first_sig_postprocessing = bool(enable_first_sig_postprocessing)
        self.first_sig_threshold = float(first_sig_threshold)
        
        # Extended state space for CRF
        self.ext_C = 2 * self.C if (enforce_single_sig_block and self.use_crf) else self.C
        trans_mask, start_mask = self._build_ext_masks()
        
        # Register buffers for CRF (important: these persist and move to the right device)
        self.register_buffer("trans_mask", trans_mask)
        self.register_buffer("start_mask", start_mask)
        self._last_lstm: torch.Tensor | None = None

        # Trainable transition and start scores (masked by constraints)
        self.trans_scores = torch.nn.Parameter(torch.zeros(self.ext_C, self.ext_C))
        self.start_scores = torch.nn.Parameter(torch.zeros(self.ext_C))

    def _metric(self, name: str) -> Metric:
        return cast(Metric, self.metrics[name])

    def _ext_index(self, y: int, f: int) -> int:
        # map (label y, flag f) to extended index
        return y + f * self.C

    def _build_ext_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build allowed transition mask over extended states and start-state mask.
        Rules:
          - monotone by label order: y_next >= y_prev
          - if there's a sig block, it must be single/contiguous
          - sig block is optional (can be skipped entirely)
          - flag evolution:
              if prev_flag==0 and y_prev==SIG and y_next!=SIG → next_flag=1
              else next_flag = prev_flag
        """
        C, ext_C = self.C, self.ext_C
        sig = self.sig_idx
        allowed = torch.zeros((ext_C, ext_C), dtype=torch.bool)
        start_allowed = torch.zeros((ext_C,), dtype=torch.bool)

        flag_values = (0, 1) if ext_C == 2 * C else (0,)
        for y0 in range(C):
            for f0 in flag_values:
                i0 = self._ext_index(y0, f0)
                for y1 in range(C):
                    # Basic monotonicity constraint
                    if y1 < y0:
                        continue
                        
                    # Special handling for sig transitions
                    if sig is not None:
                        # If we're post-sig (f0=1), can't go back to sig
                        if f0 == 1 and y1 == sig:
                            continue
                            
                        # Compute next flag - set to 1 if leaving sig section
                        if f0 == 0 and y0 == sig and y1 != sig:
                            next_flag = 1
                        else:
                            next_flag = f0
                            
                        # Allow transition only with correct flag
                        allowed[i0, self._ext_index(y1, next_flag)] = True
                    else:
                        # No sig label - just enforce monotonicity
                        allowed[i0, self._ext_index(y1, 0)] = True

        # Allow starting in any label with flag=0
        for y in range(C):
            start_allowed[self._ext_index(y, 0)] = True

        return allowed, start_allowed

    def _extend_emissions(self, emissions: torch.Tensor) -> torch.Tensor:
        """
        Duplicate emissions across flags: ext_emissions = [e | e] along last dim.
        emissions: (B,S,C) → (B,S,2C)
        """
        return torch.cat([emissions, emissions], dim=-1)

    def _apply_sig_position_bias(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply a small, position-dependent bias to encourage the earliest plausible
        signature block and to encourage back_matter toward the tail.

        - Subtract a penalty from SIG emissions that grows later in the sequence
          (only on the pre-flag half when using the extended state space).
        - Add a small bonus to BACK_MATTER emissions later in the sequence.

        Args:
            emissions: (B, S, C) or (B, S, 2C)
            mask: (B, S) boolean mask of valid positions

        Returns:
            Adjusted emissions tensor with same shape as input.
        """
        if self.sig_idx is None:
            return emissions

        B, S, D = emissions.shape
        device = emissions.device

        # Compute normalized position t in [0,1] per valid token
        seq_lens = mask.sum(dim=1).clamp(min=1)  # (B,)
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        # clamp denominator to avoid div by zero
        denom = (seq_lens - 1).unsqueeze(1).clamp(min=1)
        t = (positions / denom).masked_fill(~mask, 0.0)  # (B, S)

        def _sigmoid(x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(x)

        # Late penalty for SIG (increases after center)
        if self.prefer_earliest_sig and self.sig_late_penalty > 0:
            pen = self.sig_late_penalty * _sigmoid(
                self.sig_penalty_sharpness * (t - self.sig_penalty_center)
            )  # (B, S)
            if D == self.C:
                emissions[:, :, self.sig_idx] = emissions[:, :, self.sig_idx] - pen
            elif D == 2 * self.C:
                # Only penalize the pre-flag half (indices [0..C-1]) where sig block resides
                emissions[:, :, self.sig_idx] = emissions[:, :, self.sig_idx] - pen

        # Late bonus for BACK_MATTER (increases after center)
        if self.back_idx is not None and self.back_late_bonus != 0.0:
            bonus = self.back_late_bonus * _sigmoid(
                self.back_bonus_sharpness * (t - self.back_bonus_center)
            )
            if D == self.C:
                emissions[:, :, self.back_idx] = emissions[:, :, self.back_idx] + bonus
            elif D == 2 * self.C:
                # Apply to both halves so that post-flag states also benefit
                emissions[:, :, self.back_idx] = emissions[:, :, self.back_idx] + bonus
                emissions[:, :, self.back_idx + self.C] = emissions[:, :, self.back_idx + self.C] + bonus

        return emissions

    def _apply_learned_positional_prior(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply a learned per-class bias based on normalized position t in [0,1].
        Trained end-to-end to shape class locations along the document.
        """
        if not self.use_positional_prior or self.pos_prior_weight == 0.0:
            return emissions
        B, S, D = emissions.shape
        device = emissions.device
        seq_lens = mask.sum(dim=1).clamp(min=1)
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        denom = (seq_lens - 1).unsqueeze(1).clamp(min=1)
        t = (positions / denom).masked_fill(~mask, 0.0)  # (B, S)
        two_pi_t = 2 * torch.pi * t
        feats = torch.stack([
            t,
            t * t,
            t * t * t,
            torch.sin(two_pi_t),
            torch.cos(two_pi_t),
            1.0 - t,
        ], dim=-1)  # (B, S, 6)
        logits = self.pos_prior_mlp(feats.view(B * S, -1)).view(B, S, self.C)
        bias = self.pos_prior_weight * logits
        if D == self.C:
            return emissions + bias
        elif D == 2 * self.C:
            bias_ext = torch.cat([bias, bias], dim=-1)
            return emissions + bias_ext
        return emissions

    def _extend_tags(self, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Derive the post_sig flag from gold labels, then pack into extended indices.
        labels: (B,S) in [0..C-1], mask: (B,S) bool
        returns ext_labels: (B,S) in [0..2C-1]
        """
        B, S = labels.shape
        flags = torch.zeros_like(labels)
        if self.sig_idx is None:
            return labels  # no change if SIG label is absent

        for b in range(B):
            post = 0
            for s in range(S):
                if not mask[b, s]:
                    break
                if post == 0 and labels[b, s] == self.sig_idx:
                    flags[b, s] = 0  # still in sig
                elif post == 0 and s > 0 and labels[b, s-1] == self.sig_idx and labels[b, s] != self.sig_idx:
                    post = 1  # just left sig
                flags[b, s] = post

        # Convert to extended indices
        return labels + flags * self.C

    def forward(self, emissions: Tensor, mask: torch.Tensor | None = None) -> Tensor:
        """
        Forward pass through the neural networks and CRF.
        
        Args:
            emissions: XGBoost emission logits (B, S, C)
            mask: Optional boolean mask for valid positions (B, S)
            
        Returns:
            Modified emission logits (B, S, C) or (B, S, 2C) if using extended state space
        """
        batch_size, seq_len, _ = emissions.shape
        if not self.use_lstm:
            out = emissions
            self._last_lstm = None
        else:
            if mask is not None:
                emissions = emissions.masked_fill(~mask.unsqueeze(-1), 0.0)
            
            # Transform through MLP
            x = emissions.view(-1, self.C)  # (B*S, C)
            x = self.hidden(x)              # (B*S, H)
            x = x.view(batch_size, seq_len, -1)  # (B, S, H)
            
            # Process through LSTM
            if mask is not None:
                lengths = mask.sum(dim=1).to(torch.int64).clamp(min=1)
                packed = pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                packed_out, _ = self.lstm(packed)
                x, _ = pad_packed_sequence(
                    packed_out, batch_first=True, total_length=seq_len
                )
            else:
                x, _ = self.lstm(x)  # (B, S, 2H) due to bidirectional
            # Cache LSTM features for auxiliary losses
            self._last_lstm = x
            
            # Project back to class space
            x = self.proj(x)  # (B, S, C)
            
            # Add residual connection from original emissions
            out = x + emissions
        
        # Extend state space if using single-sig block
        if self.use_crf and self.enforce_single_sig_block:
            out = self._extend_emissions(out)
            
        return out

    def _aux_sig_start_loss(self, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary loss that encourages the model to place the first signature page
        at the correct earliest position. For each sequence that contains a sig
        label, compute a softmax over time from LSTM features and supervise the
        argmax to be the index of the first gold sig.

        Returns a scalar loss averaged over the batch (only sequences with a sig contribute).
        """
        if self.sig_idx is None or self.aux_sig_start_loss_weight <= 0:
            return torch.tensor(0.0, device=labels.device)

        feats = getattr(self, "_last_lstm", None)
        if feats is None:
            return torch.tensor(0.0, device=labels.device)

        B, _, _ = feats.shape
        scores = self.sig_start_head(feats).squeeze(-1)  # (B, S)

        losses = []
        for b in range(B):
            seq_len = int(mask[b].sum().item())
            if seq_len <= 0:
                continue
            labels_b = labels[b, :seq_len]
            # find first index where label == sig_idx
            sig_positions = (labels_b == self.sig_idx).nonzero(as_tuple=False)
            if sig_positions.numel() == 0:
                continue  # no sig in this document
            first_idx = int(sig_positions[0].item())
            logits_b = scores[b, :seq_len]
            log_probs = F.log_softmax(logits_b, dim=0)
            losses.append(-log_probs[first_idx])

        if not losses:
            return torch.tensor(0.0, device=labels.device)
        return torch.stack(losses).mean()

    def _aux_back_start_loss(self, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary loss that encourages the model to place back_matter onset
        toward the tail by supervising the first occurrence of back_matter.
        """
        if self.back_idx is None or self.aux_back_start_loss_weight <= 0:
            return torch.tensor(0.0, device=labels.device)
        feats = getattr(self, "_last_lstm", None)
        if feats is None:
            return torch.tensor(0.0, device=labels.device)
        B, _, _ = feats.shape
        scores = self.back_start_head(feats).squeeze(-1)  # (B, S)
        losses = []
        for b in range(B):
            seq_len = int(mask[b].sum().item())
            if seq_len <= 0:
                continue
            labels_b = labels[b, :seq_len]
            back_positions = (labels_b == self.back_idx).nonzero(as_tuple=False)
            if back_positions.numel() == 0:
                continue
            first_idx = int(back_positions[0].item())
            logits_b = scores[b, :seq_len]
            log_probs = F.log_softmax(logits_b, dim=0)
            losses.append(-log_probs[first_idx])
        if not losses:
            return torch.tensor(0.0, device=labels.device)
        return torch.stack(losses).mean()

    def _compute_log_likelihood(
        self, emissions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log-likelihood using forward algorithm for CRF.
        
        Implements the forward algorithm to compute:
        log P(y|x) = log(score(x,y)) - log(Z(x))
        where score(x,y) is the unnormalized score of sequence y given x,
        and Z(x) is the partition function (sum over all possible sequences).
        
        The algorithm uses dynamic programming to efficiently compute the
        log-partition function by maintaining forward variables alpha[t][s]
        representing the log-sum of scores of all partial sequences ending
        in state s at time t.
        
        Args:
            emissions: (B, S, C) or (B, S, 2C) emission scores from neural network
                where C is the number of base classes. If using extended state space
                for single signature block constraints, size is (B, S, 2C).
            labels: (B, S) true label sequence indices in range [0, C-1]
            mask: (B, S) boolean mask indicating valid positions (True = valid)
                
        Returns:
            Log-likelihood averaged over the batch
            
        Note:
            Uses extended state space when enforce_single_sig_block=True
            to maintain signature block constraints. The extended space
            tracks both the current label and whether we've already seen
            a signature block, preventing re-entry into signature pages.
        """
        batch_size, seq_len, _ = emissions.shape
        start_mask = cast(torch.Tensor, self.start_mask)
        trans_mask = cast(torch.Tensor, self.trans_mask)
        
        # Get extended labels and mask
        ext_labels = self._extend_tags(labels, mask)  # (B, S)
        
        # Forward recursion
        alphas = emissions.new_full((batch_size, seq_len, self.ext_C), -1e9)  # log space
        
        # Initialize with start constraints and learned start scores
        masked_start_scores = torch.where(
            start_mask,
            cast(torch.Tensor, self.start_scores),
            torch.tensor(-1e9, device=alphas.device),
        )
        alphas[:, 0] = emissions[:, 0] + masked_start_scores  # (B, C) or (B, 2C)
        
        # Iterate through sequence
        for t in range(1, seq_len):
            mask_t = mask[:, t].unsqueeze(1)  # (B, 1)
            emit_t = emissions[:, t]  # (B, C) or (B, 2C)
            
            # Previous alpha values: (B, C) or (B, 2C)
            prev_alpha = alphas[:, t-1].unsqueeze(2)  # (B, C/2C, 1)
            
            # Transition scores with constraints
            # trans_mask: (C, C) or (2C, 2C) boolean
            trans_scores = torch.where(
                trans_mask,
                cast(torch.Tensor, self.trans_scores),
                torch.tensor(-1e9, device=prev_alpha.device),
            )
            
            # Add everything up in log-space
            curr_alpha = (prev_alpha + trans_scores.unsqueeze(0)  # (B, C/2C, C/2C)
                         ).logsumexp(dim=1)  # (B, C/2C)
            curr_alpha = curr_alpha + emit_t  # (B, C/2C)
            
            # Mask out invalid positions
            alphas[:, t] = torch.where(
                mask_t,
                curr_alpha,
                alphas[:, t]
            )
        
        # Compute gold path score = start constraint + sum emissions on gold tags + transition constraints
        seq_lens = mask.sum(1).long()  # (B,)
        # Emissions along gold path
        gold_emit = torch.gather(emissions, 2, ext_labels.unsqueeze(-1)).squeeze(-1)  # (B, S)
        gold_emit = (gold_emit * mask.float()).sum(dim=1)  # (B,)
        
        # Start scores (masked by constraints)
        masked_start_scores = torch.where(
            start_mask,
            cast(torch.Tensor, self.start_scores),
            torch.tensor(-1e9, device=emissions.device),
        )
        start_score = masked_start_scores[ext_labels[:, 0]]
        start_score = start_score * mask[:, 0].float()
        
        # Transition penalties across gold path
        if emissions.size(1) > 1:
            prev_tags = ext_labels[:, :-1]
            curr_tags = ext_labels[:, 1:]
            masked_trans_scores = torch.where(
                trans_mask,
                cast(torch.Tensor, self.trans_scores),
                torch.tensor(-1e9, device=emissions.device),
            )
            trans_pen = masked_trans_scores[prev_tags, curr_tags]  # (B, S-1)
            valid_pairs = (mask[:, :-1] & mask[:, 1:]).float()
            trans_pen = (trans_pen * valid_pairs).sum(dim=1)
        else:
            trans_pen = torch.zeros_like(gold_emit)
        
        gold_score = gold_emit + trans_pen + start_score  # (B,)
        
        # Log-partition function from terminal alphas
        batch_idx = torch.arange(batch_size, device=emissions.device)
        seq_idx = seq_lens - 1  # last valid position in each sequence
        terminal_alphas = alphas[batch_idx, seq_idx]  # (B, C/2C)
        log_Z = terminal_alphas.logsumexp(dim=1)  # (B,)
        
        # Return average log-likelihood
        return (gold_score - log_Z).mean()
        
    def _viterbi_decode_ext(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Run Viterbi decoding to find best path through extended state space.
        
        Args:
            emissions: (B, S, C) or (B, S, 2C) emission scores
            mask: (B, S) boolean mask for valid positions
            
        Returns:
            Best path indices (B, S) in range [0, C-1] or [0, 2C-1]
        """
        batch_size, seq_len, num_tags = emissions.shape
        start_mask = cast(torch.Tensor, self.start_mask)
        trans_mask = cast(torch.Tensor, self.trans_mask)
        device = emissions.device
        
        # Initialize score and backpointers
        scores = emissions.new_full((batch_size, seq_len, num_tags), -1e9)
        backpointers = emissions.new_empty(
            (batch_size, seq_len, num_tags), dtype=torch.long
        )
        
        # First timestep uses start constraints and learned start scores
        masked_start_scores = torch.where(
            start_mask,
            cast(torch.Tensor, self.start_scores),
            torch.tensor(-1e9, device=device),
        )
        scores[:, 0] = emissions[:, 0] + masked_start_scores  # (B, C) or (B, 2C)
        
        # Iterate through sequence
        for t in range(1, seq_len):
            mask_t = mask[:, t].unsqueeze(1)  # (B, 1)
            emit_t = emissions[:, t]  # (B, C) or (B, 2C)
            
            # Previous scores: (B, C/2C)
            prev_scores = scores[:, t-1].unsqueeze(2)  # (B, C/2C, 1)
            
            # Transition scores with constraints
            trans_scores = torch.where(
                trans_mask,
                cast(torch.Tensor, self.trans_scores),
                torch.tensor(-1e9, device=device),
            )
            
            # Add up scores and track best previous tag
            curr_scores = prev_scores + trans_scores.unsqueeze(0)  # (B, C/2C, C/2C)
            best_prev, best_tag = curr_scores.max(dim=1)  # (B, C/2C)
            curr_scores = best_prev + emit_t
            
            # Mask out invalid positions
            scores[:, t] = torch.where(mask_t, curr_scores, scores[:, t])
            backpointers[:, t] = best_tag
        
        # Find best final tag and trace back
        seq_lens = mask.sum(1)  # (B,)
        batch_idx = torch.arange(batch_size, device=device)
        best_scores = torch.gather(scores, 1, (seq_lens - 1).unsqueeze(1).unsqueeze(2).expand(-1, -1, num_tags))[:, 0]
        best_last_tags = best_scores.argmax(dim=1)  # (B,)
        
        # Trace back through backpointers
        best_path = emissions.new_zeros((batch_size, seq_len), dtype=torch.long)
        best_path[batch_idx, seq_lens-1] = best_last_tags
        
        # Use basic Python loop for readability, since this is not a bottleneck
        for b in range(batch_size):
            for t in range(int(seq_lens[b])-2, -1, -1):
                best_path[b, t] = backpointers[b, t+1, best_path[b, t+1]]
        
        # For unmasked positions, fill with 0 (first class)
        best_path = torch.where(mask, best_path, torch.tensor(0, device=device))
        
        # Project back to original state space if using extended states
        if self.enforce_single_sig_block:
            best_path = best_path % self.C
            
        return best_path

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        _ = batch_idx
        emissions, labels = batch
        
        # Create mask for padding
        mask = (labels != -100)  # (B, S)
        labels = torch.where(mask, labels, torch.zeros_like(labels))  # replace pad with 0
        
        # Forward pass through neural networks
        emissions = self(emissions, mask=mask)  # (B, S, C) or (B, S, 2C)
        # Apply learned positional prior and small decoding bias
        emissions = self._apply_learned_positional_prior(emissions, mask)
        emissions = self._apply_sig_position_bias(emissions, mask)
        
        if self.use_crf:
            # Compute CRF loss
            log_likelihood = self._compute_log_likelihood(emissions, labels, mask)
            loss = -log_likelihood  # maximize likelihood = minimize negative log likelihood
        else:
            # Per-page softmax loss (no CRF)
            logits = emissions[mask]
            targets = labels[mask]
            loss = F.cross_entropy(logits, targets)

        # Auxiliary earliest-sig loss (training signal)
        aux_sig_loss = self._aux_sig_start_loss(labels, mask)
        if aux_sig_loss.requires_grad and self.aux_sig_start_loss_weight > 0:
            loss = loss + self.aux_sig_start_loss_weight * aux_sig_loss
        # Auxiliary back-matter onset loss (training signal)
        aux_back_loss = self._aux_back_start_loss(labels, mask)
        if aux_back_loss.requires_grad and self.aux_back_start_loss_weight > 0:
            loss = loss + self.aux_back_start_loss_weight * aux_back_loss
        
        # Compute predictions (in original label space)
        if self.use_crf:
            predictions = self._viterbi_decode_ext(emissions, mask)  # (B, S)
        else:
            predictions = emissions.argmax(dim=-1)
        
        # Compute metrics only on non-padded positions
        true_labels = labels[mask]
        pred_labels = predictions[mask]
        
        # Update metrics
        self._metric("train_precision").update(pred_labels, true_labels)
        self._metric("train_recall").update(pred_labels, true_labels)
        self._metric("train_f1").update(pred_labels, true_labels)
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        if self.aux_sig_start_loss_weight > 0:
            self.log("train_aux_sig_start_loss", aux_sig_loss.detach(), prog_bar=False)
        if hasattr(self, "aux_back_start_loss_weight") and self.aux_back_start_loss_weight > 0:
            self.log("train_aux_back_start_loss", aux_back_loss.detach(), prog_bar=False)
        
        # Note: Don't compute/log step-level metrics here, we'll do that in epoch_end
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Validation step."""
        _ = batch_idx
        emissions, labels = batch
        
        # Create mask for padding
        mask = (labels != -100)  # (B, S)
        labels = torch.where(mask, labels, torch.zeros_like(labels))  # replace pad with 0
        
        # Forward pass through neural networks
        emissions = self(emissions, mask=mask)  # (B, S, C) or (B, S, 2C)
        # Apply learned positional prior and small decoding bias
        emissions = self._apply_learned_positional_prior(emissions, mask)
        emissions = self._apply_sig_position_bias(emissions, mask)
        
        if self.use_crf:
            # Compute CRF loss
            log_likelihood = self._compute_log_likelihood(emissions, labels, mask)
            loss = -log_likelihood  # maximize likelihood = minimize negative log likelihood
        else:
            # Per-page softmax loss (no CRF)
            logits = emissions[mask]
            targets = labels[mask]
            loss = F.cross_entropy(logits, targets)

        # Auxiliary earliest-sig loss (report only)
        aux_sig_loss = self._aux_sig_start_loss(labels, mask)
        aux_back_loss = self._aux_back_start_loss(labels, mask)
        
        # Compute predictions (in original label space)
        if self.use_crf:
            predictions = self._viterbi_decode_ext(emissions, mask)  # (B, S)
        else:
            predictions = emissions.argmax(dim=-1)
        
        # Compute metrics only on non-padded positions
        true_labels = labels[mask]
        pred_labels = predictions[mask]
        
        # Update metrics
        self._metric("val_precision").update(pred_labels, true_labels)
        self._metric("val_recall").update(pred_labels, true_labels)
        self._metric("val_f1").update(pred_labels, true_labels)
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        if self.aux_sig_start_loss_weight > 0:
            self.log("val_aux_sig_start_loss", aux_sig_loss.detach(), prog_bar=False)
        if hasattr(self, "aux_back_start_loss_weight") and self.aux_back_start_loss_weight > 0:
            self.log("val_aux_back_start_loss", aux_back_loss.detach(), prog_bar=False)
        
        # Note: Don't compute/log step-level metrics here, we'll do that in epoch_end
        return loss

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        train_f1 = self._metric("train_f1").compute()
        train_precision = self._metric("train_precision").compute()
        train_recall = self._metric("train_recall").compute()
        
        self.log("train_f1_epoch", train_f1, prog_bar=True)
        self.log("train_precision_epoch", train_precision, prog_bar=True)
        self.log("train_recall_epoch", train_recall, prog_bar=True)
        
        # Reset metrics for next epoch
        self._metric("train_f1").reset()
        self._metric("train_precision").reset()
        self._metric("train_recall").reset()

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        val_f1 = self._metric("val_f1").compute()
        val_precision = self._metric("val_precision").compute()
        val_recall = self._metric("val_recall").compute()
        
        self.log("val_f1", val_f1, prog_bar=True)
        self.log("val_precision", val_precision, prog_bar=True)
        self.log("val_recall", val_recall, prog_bar=True)
        
        # Reset metrics for next epoch
        self._metric("val_f1").reset()
        self._metric("val_precision").reset()
        self._metric("val_recall").reset()

    def _apply_first_sig_block_postprocessing(
        self,
        seq_preds: list[Prediction],
        sig_threshold: float = 0.3,
    ) -> tuple[list[Prediction], bool]:
        """
        Apply decode-time fix: find the first continuous signature block and convert those
        pages to 'sig', then everything after to 'back_matter'. Uses predicted signature
        classes when present; otherwise falls back to a probability threshold.
        
        This addresses the case where real signature pages are incorrectly predicted as 'body' 
        or 'back_matter' because exhibit signature pages later in the document confuse the model.
        
        Args:
            seq_preds: List of prediction dictionaries for a single document
            sig_threshold: Minimum probability threshold for considering a page as signature
            
        Returns:
            Tuple of (modified_predictions, was_modified_flag)
        """
        if not seq_preds or self.sig_idx is None:
            # Add postprocess_modified flag to each prediction
            for pred in seq_preds:
                pred["postprocess_modified"] = False
            return seq_preds, False
            
        # Find the first continuous block of signature pages
        sig_label_name = self.label_names[self.sig_idx]
        back_label_name = (
            self.label_names[self.back_idx] if self.back_idx is not None else "back_matter"
        )
        first_sig_idx = None
        last_sig_idx = None

        # Prefer predicted signature block if present
        for i, pred in enumerate(seq_preds):
            if pred.get("pred_class") == sig_label_name:
                first_sig_idx = i
                break

        if first_sig_idx is not None:
            last_sig_idx = first_sig_idx
            for i in range(first_sig_idx + 1, len(seq_preds)):
                if seq_preds[i].get("pred_class") == sig_label_name:
                    last_sig_idx = i
                else:
                    break
        else:
            # Fall back to probability threshold if no signature predicted
            for i, pred in enumerate(seq_preds):
                pred_probs = cast(dict[str, float], pred["pred_probs"])
                sig_prob = pred_probs.get(sig_label_name, 0.0)
                if sig_prob >= sig_threshold:
                    first_sig_idx = i
                    break

            if first_sig_idx is None:
                for pred in seq_preds:
                    pred["postprocess_modified"] = False
                return seq_preds, False

            last_sig_idx = first_sig_idx
            for i in range(first_sig_idx + 1, len(seq_preds)):
                pred_probs = cast(dict[str, float], seq_preds[i]["pred_probs"])
                sig_prob = pred_probs.get(sig_label_name, 0.0)
                if sig_prob >= sig_threshold:
                    last_sig_idx = i
                else:
                    break  # End of continuous block
        
        # Create modified predictions
        modified_preds = []
        was_modified = False
        
        for i, pred in enumerate(seq_preds):
            new_pred = pred.copy()
            new_pred["pred_probs"] = cast(dict[str, float], pred["pred_probs"]).copy()
            
            if first_sig_idx <= i <= last_sig_idx:
                # Pages in the signature block: convert to 'sig'
                if pred["pred_class"] != sig_label_name:
                    new_pred["pred_class"] = sig_label_name
                    new_pred["postprocess_modified"] = True
                    was_modified = True
                else:
                    new_pred["postprocess_modified"] = False
            elif i > last_sig_idx:
                # Pages after signature block: convert to 'back_matter'
                if self.back_idx is not None and pred["pred_class"] != back_label_name:
                    new_pred["pred_class"] = back_label_name
                    new_pred["postprocess_modified"] = True
                    was_modified = True
                else:
                    new_pred["postprocess_modified"] = False
            else:
                # Pages before signature block: leave unchanged
                new_pred["postprocess_modified"] = False
                
            modified_preds.append(new_pred)
        
        return modified_preds, was_modified

    def predict_step(
        self, batch: Tensor | tuple[Tensor, Tensor], batch_idx: int
    ) -> list[list[Prediction]]:
        """
        Inference step.
        
        Args:
            batch: Emissions tensor
            batch_idx: Batch index
            
        Returns:
            List of dictionaries with predictions and probabilities, including postprocess_modified flag
        """
        _ = batch_idx
        # Unpack batch and build mask
        if isinstance(batch, tuple):
            emissions_in, mask = batch
        else:
            emissions_in, mask = batch, torch.ones_like(batch[:, :, 0], dtype=torch.bool)
        
        # Forward pass
        emissions = self(emissions_in, mask=mask)  # (B, S, C) or (B, S, 2C)
        emissions = self._apply_learned_positional_prior(emissions, mask)
        emissions = self._apply_sig_position_bias(emissions, mask)
        
        # Viterbi decoding
        if self.use_crf:
            predictions = self._viterbi_decode_ext(emissions, mask)  # (B, S)
        else:
            predictions = emissions.argmax(dim=-1)
        
        # Convert to probabilities (optional, since we're using hard Viterbi decoding)
        class_probs = torch.softmax(emissions, dim=-1)  # (B, S, C) or (B, S, 2C)
        if self.use_crf and self.enforce_single_sig_block:
            class_probs = class_probs.view(class_probs.shape[0], class_probs.shape[1], 2, -1).sum(2)
        
        # Convert to list of dictionaries (respect true sequence lengths)
        results: list[list[Prediction]] = []
        for b in range(predictions.size(0)):
            seq_preds: list[Prediction] = []
            seq_len = int(mask[b].sum().item())
            for t in range(seq_len):
                pred_class = int(predictions[b, t].item())  # ensure integer index
                class_name = str(self.label_names[pred_class])  # ensure string
                probs = {
                    str(name): float(class_probs[b, t, i])
                    for i, name in enumerate(self.label_names)
                }
                seq_preds.append(
                    {
                    "pred_class": class_name,
                    "pred_probs": probs
                    }
                )
            
            # Apply decode-time fix for signature block detection if enabled
            if self.enable_first_sig_postprocessing:
                seq_preds_fixed, _ = self._apply_first_sig_block_postprocessing(seq_preds, self.first_sig_threshold)
            else:
                # Add postprocess_modified flag as False for all predictions
                seq_preds_fixed = []
                for pred in seq_preds:
                    new_pred = pred.copy()
                    new_pred["postprocess_modified"] = False
                    seq_preds_fixed.append(new_pred)
            results.append(seq_preds_fixed)
        
        return results


    def configure_optimizers(self) -> AdamW:
        """Configure the optimizer with the correct learning rate and weight decay."""
        return AdamW(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"]
        )
