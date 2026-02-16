"""
Main taxonomy training and inference module.

This module mirrors ner.py in structure and provides the main entry points for
training and testing the taxonomy classifier using PyTorch Lightning with
optional hyperparameter optimization via Optuna. It supports two modes for text
representation: 'transformer' (HF model) and 'tfidf', and is configured for
multi-label prediction (one or more classes per section).
"""
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false

# Standard library
import argparse
import os
import time
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, cast, Literal
import sys

# Environment config
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Third-party
import pandas as pd
import lightning.pytorch as pl
import torch
import numpy as np
from numpy.typing import NDArray
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger

from sklearn.feature_extraction.text import TfidfVectorizer

from optuna import Trial, create_study
from optuna.integration import PyTorchLightningPruningCallback

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# Local modules
if __package__ in (None, ""):
    src_root = Path(__file__).resolve().parents[3]
    if str(src_root) not in sys.path:
        sys.path.append(str(src_root))
    from etl.models.taxonomy.taxonomy_constants import (  # type: ignore[reportMissingImports]
        TAXONOMY_CKPT_PATH,
        TAXONOMY_VECTORIZER_PATH,
        TAXONOMY_EVAL_METRICS_PATH,
    )
    from etl.models.taxonomy.taxonomy_classes import (  # type: ignore[reportMissingImports]
        TransformerDataModule,
        TfidfDataModule,
        TaxonomyClassifier,
    )
else:
    from .taxonomy_constants import (
        TAXONOMY_CKPT_PATH,
        TAXONOMY_VECTORIZER_PATH,
        TAXONOMY_EVAL_METRICS_PATH,
    )
    from .taxonomy_classes import (
        TransformerDataModule,
        TfidfDataModule,
        TaxonomyClassifier,
    )

_ = seed_everything(42, workers=True, verbose=False)
torch.set_float32_matmul_precision("high")

PrecisionInput = Literal["bf16-mixed", "32-true"]
LOG_DIR = Path(__file__).resolve().parent / "logs"


class _LogitsOutput(Protocol):
    logits: torch.Tensor


def _combine_text(a: str, s: str, t: str) -> str:
    a = a or ""
    s = s or ""
    t = t or ""
    return f"[ARTICLE] {a}\n[SECTION] {s}\n[TEXT] {t}"


@dataclass
class TaxonomyConfig:
    data_parquet: str
    model_name: str
    label_list: list[str] | None
    mode: Literal["transformer", "tfidf"]
    num_trials: int
    max_epochs: int
    batch_size: int = 16
    max_length: int = 1024
    labels_column: str = "labels"
    val_size: float = 0.2
    test_size: float = 0.1
    split_path: str | None = None
    split_seed: int = 42
    eval_metrics_path: str | None = None
    decision_threshold: float = 0.5
    # kept for backward compatibility with older config payloads; unused for multi-label
    use_stratify: bool = False
    tfidf_max_features: int | None = 50_000


class _DenseConvertible(Protocol):
    def astype(self, dtype: str) -> "_DenseConvertible": ...

    def toarray(self) -> np.ndarray: ...


IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


class TaxonomyTrainer:
    """
    Orchestrates hyperparameter optimization and training of TaxonomyClassifier.
    """

    def __init__(self, cfg: TaxonomyConfig) -> None:
        self.cfg: TaxonomyConfig = cfg
        if torch.backends.mps.is_available():
            self.device: str = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.train_rows: dict[str, list[str] | list[list[int]]] = {}
        self.val_rows: dict[str, list[str] | list[list[int]]] = {}
        self.test_rows: dict[str, list[str] | list[list[int]]] = {}
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.X_val: np.ndarray | None = None
        self.y_val: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_test: np.ndarray | None = None
        self.class_pos_weight: list[float] | None = None
        self.tuned_decision_thresholds: list[float] | None = None
        self.vectorizer: TfidfVectorizer | None = None

    @staticmethod
    def _parse_labels_cell(raw: object) -> list[str]:
        if raw is None or raw is pd.NA:
            return []
        if isinstance(raw, (float, np.floating)) and np.isnan(raw):
            return []
        if isinstance(raw, list):
            if not all(isinstance(item, str) for item in raw):
                raise ValueError("`labels` lists must contain only strings.")
            return [item for item in raw if item]
        if isinstance(raw, tuple):
            if not all(isinstance(item, str) for item in raw):
                raise ValueError("`labels` tuples must contain only strings.")
            return [item for item in raw if item]
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return []
            if text.startswith("["):
                parsed = cast(object, json.loads(text))
                if not isinstance(parsed, list) or not all(
                    isinstance(item, str) for item in parsed
                ):
                    raise ValueError(
                        "JSON-serialized `labels` must decode to list[str]."
                    )
                return [item for item in parsed if item]
            return [text]
        raise ValueError(
            f"`labels` entries must be list[str], tuple[str, ...], or string; got {type(raw)!r}."
        )

    def _to_multi_hot(self, label_ids: list[int], num_labels: int) -> list[int]:
        row = [0] * num_labels
        for label_id in label_ids:
            row[label_id] = 1
        return row

    def _compute_class_pos_weight(
        self,
        train_label_matrix: IntArray,
        label_list: list[str],
    ) -> list[float]:
        if train_label_matrix.ndim != 2:
            raise ValueError(
                f"Expected 2D training label matrix; got ndim={train_label_matrix.ndim}."
            )
        if train_label_matrix.shape[1] != len(label_list):
            raise ValueError(
                "Training label width does not match `label_list` length."
            )

        num_rows = int(train_label_matrix.shape[0])
        if num_rows == 0:
            raise ValueError("Training split is empty; cannot compute class weights.")

        pos_counts = train_label_matrix.sum(axis=0).astype(np.int64)
        neg_counts = num_rows - pos_counts
        safe_pos_counts = np.where(pos_counts > 0, pos_counts, 1)
        raw_ratio = neg_counts.astype(np.float64) / safe_pos_counts.astype(np.float64)
        # Temper class weights so very rare labels are up-weighted without dominating loss.
        pos_weight = np.sqrt(raw_ratio)
        pos_weight = np.clip(pos_weight, a_min=1.0, a_max=8.0)
        pos_weight = np.where(pos_counts > 0, pos_weight, 1.0)
        return [float(x) for x in pos_weight.tolist()]

    def _resolve_split_path(self) -> Path:
        if self.cfg.split_path:
            return Path(self.cfg.split_path)
        return Path(self.cfg.data_parquet).with_name("taxonomy-splits.json")

    def _iterative_stratified_binary_split(
        self,
        labels: IntArray,
        holdout_size: float,
        seed: int,
    ) -> tuple[IntArray, IntArray]:
        if labels.ndim != 2:
            raise ValueError("`labels` must be a 2D multi-hot array for stratified split.")

        num_rows = int(labels.shape[0])
        if num_rows < 2:
            raise ValueError("Need at least 2 rows to split data.")
        if not 0.0 < holdout_size < 1.0:
            raise ValueError(f"`holdout_size` must be in (0, 1); got {holdout_size!r}.")

        holdout_target = int(round(num_rows * holdout_size))
        holdout_target = max(1, min(num_rows - 1, holdout_target))
        rng = np.random.default_rng(seed)

        holdout_mask: BoolArray = np.zeros(num_rows, dtype=bool)
        unassigned_mask: BoolArray = np.ones(num_rows, dtype=bool)

        label_totals: IntArray = labels.sum(axis=0).astype(np.int64)
        desired_holdout: IntArray = np.rint(
            label_totals * holdout_target / num_rows
        ).astype(np.int64)
        current_holdout: IntArray = np.zeros(labels.shape[1], dtype=np.int64)

        while int(holdout_mask.sum()) < holdout_target:
            deficits = desired_holdout - current_holdout
            remaining: IntArray = labels[unassigned_mask].sum(axis=0).astype(np.int64)
            eligible_labels = np.where((deficits > 0) & (remaining > 0))[0]
            if eligible_labels.size == 0:
                break

            remaining_eligible: IntArray = remaining[eligible_labels]
            min_remaining = int(remaining_eligible.min())
            rare_label_candidates = eligible_labels[remaining_eligible == min_remaining]

            if rare_label_candidates.size > 1:
                rare_deficits = deficits[rare_label_candidates]
                max_deficit = int(rare_deficits.max())
                rare_label_candidates = rare_label_candidates[
                    rare_deficits == max_deficit
                ]

            chosen_label = int(rng.choice(rare_label_candidates))
            sample_candidates = np.where(
                unassigned_mask & (labels[:, chosen_label] == 1)
            )[0]
            if sample_candidates.size == 0:
                continue

            deficit_weights = np.clip(deficits, a_min=0, a_max=None)
            coverage_scores: IntArray = (
                labels[sample_candidates] * deficit_weights
            ).sum(axis=1).astype(np.int64)
            max_coverage = int(coverage_scores.max())
            best_candidates = sample_candidates[
                coverage_scores == max_coverage
            ]

            if best_candidates.size > 1:
                label_count_scores: IntArray = labels[best_candidates].sum(
                    axis=1
                ).astype(np.int64)
                max_label_count = int(label_count_scores.max())
                best_candidates = best_candidates[
                    label_count_scores == max_label_count
                ]

            selected_idx = int(rng.choice(best_candidates))
            holdout_mask[selected_idx] = True
            unassigned_mask[selected_idx] = False
            current_holdout += labels[selected_idx].astype(np.int64)

        assigned = int(holdout_mask.sum())
        if assigned < holdout_target:
            remainder = np.where(unassigned_mask)[0]
            needed = holdout_target - assigned
            chosen = rng.choice(remainder, size=needed, replace=False).astype(np.int64)
            holdout_mask[chosen] = True

        holdout_indices: IntArray = np.sort(np.where(holdout_mask)[0]).astype(np.int64)
        main_indices: IntArray = np.sort(np.where(~holdout_mask)[0]).astype(np.int64)
        return main_indices, holdout_indices

    def _split_train_val_test(
        self,
        labels: IntArray,
    ) -> tuple[IntArray, IntArray, IntArray]:
        val_size = self.cfg.val_size
        test_size = self.cfg.test_size
        if not 0.0 < val_size < 1.0:
            raise ValueError(f"`val_size` must be in (0, 1); got {val_size!r}.")
        if not 0.0 <= test_size < 1.0:
            raise ValueError(f"`test_size` must be in [0, 1); got {test_size!r}.")
        if val_size + test_size >= 1.0:
            raise ValueError(
                f"`val_size + test_size` must be < 1; got {val_size + test_size!r}."
            )

        seed = self.cfg.split_seed
        holdout_size = val_size + test_size
        train_indices, holdout_indices = self._iterative_stratified_binary_split(
            labels=labels, holdout_size=holdout_size, seed=seed
        )

        if test_size == 0.0:
            val_indices = holdout_indices
            test_indices: IntArray = np.array([], dtype=np.int64)
            return train_indices, val_indices, test_indices

        test_within_holdout = test_size / holdout_size
        holdout_labels = labels[holdout_indices]
        val_local, test_local = self._iterative_stratified_binary_split(
            labels=holdout_labels,
            holdout_size=test_within_holdout,
            seed=seed + 1,
        )
        val_indices = holdout_indices[val_local]
        test_indices = holdout_indices[test_local]
        return train_indices, val_indices, test_indices

    def _counts_and_prevalence(
        self,
        label_matrix: IntArray,
        indices: IntArray,
    ) -> tuple[IntArray, NDArray[np.float64]]:
        num_labels = int(label_matrix.shape[1])
        if indices.size == 0:
            return (
                np.zeros(num_labels, dtype=np.int64),
                np.zeros(num_labels, dtype=np.float64),
            )
        counts = label_matrix[indices].sum(axis=0).astype(np.int64)
        prevalence = counts.astype(np.float64) / float(indices.size)
        return counts, prevalence

    def _build_split_quality_report(
        self,
        label_matrix: IntArray,
        label_list: list[str],
        train_indices: IntArray,
        val_indices: IntArray,
        test_indices: IntArray,
    ) -> dict[str, object]:
        all_indices = np.arange(label_matrix.shape[0], dtype=np.int64)
        overall_counts, overall_prev = self._counts_and_prevalence(label_matrix, all_indices)
        train_counts, train_prev = self._counts_and_prevalence(label_matrix, train_indices)
        val_counts, val_prev = self._counts_and_prevalence(label_matrix, val_indices)
        test_counts, test_prev = self._counts_and_prevalence(label_matrix, test_indices)

        train_diff = np.abs(train_prev - overall_prev)
        val_diff = np.abs(val_prev - overall_prev)
        test_diff = np.abs(test_prev - overall_prev)

        per_label: dict[str, dict[str, object]] = {}
        for idx, label in enumerate(label_list):
            per_label[label] = {
                "counts": {
                    "overall": int(overall_counts[idx]),
                    "train": int(train_counts[idx]),
                    "val": int(val_counts[idx]),
                    "test": int(test_counts[idx]),
                },
                "prevalence": {
                    "overall": float(overall_prev[idx]),
                    "train": float(train_prev[idx]),
                    "val": float(val_prev[idx]),
                    "test": float(test_prev[idx]),
                },
                "abs_diff_vs_overall": {
                    "train": float(train_diff[idx]),
                    "val": float(val_diff[idx]),
                    "test": float(test_diff[idx]),
                },
            }

        empty_mask = (label_matrix.sum(axis=1) == 0).astype(np.int64)
        empty_total = int(empty_mask.sum())
        empty_train = int(empty_mask[train_indices].sum()) if train_indices.size > 0 else 0
        empty_val = int(empty_mask[val_indices].sum()) if val_indices.size > 0 else 0
        empty_test = int(empty_mask[test_indices].sum()) if test_indices.size > 0 else 0

        def _rate(count: int, size: int) -> float:
            if size == 0:
                return 0.0
            return float(count / size)

        return {
            "aggregate_abs_diff_vs_overall": {
                "train_mean": float(train_diff.mean()),
                "train_max": float(train_diff.max()),
                "val_mean": float(val_diff.mean()),
                "val_max": float(val_diff.max()),
                "test_mean": float(test_diff.mean()),
                "test_max": float(test_diff.max()),
            },
            "empty_label_rows": {
                "counts": {
                    "overall": empty_total,
                    "train": empty_train,
                    "val": empty_val,
                    "test": empty_test,
                },
                "rates": {
                    "overall": _rate(empty_total, int(label_matrix.shape[0])),
                    "train": _rate(empty_train, int(train_indices.size)),
                    "val": _rate(empty_val, int(val_indices.size)),
                    "test": _rate(empty_test, int(test_indices.size)),
                },
            },
            "per_label": per_label,
        }

    def _write_split_manifest(
        self,
        df: pd.DataFrame,
        label_matrix: IntArray,
        label_list: list[str],
        train_indices: IntArray,
        val_indices: IntArray,
        test_indices: IntArray,
    ) -> None:
        split_path = self._resolve_split_path()
        split_path.parent.mkdir(parents=True, exist_ok=True)

        if "section_uuid" in df.columns:
            section_uuids = cast(pd.Series, df["section_uuid"]).astype(str).tolist()
        else:
            section_uuids = [str(i) for i in range(len(df))]

        def _payload(indices: IntArray) -> dict[str, list[object]]:
            return {
                "row_indices": [int(i) for i in indices.tolist()],
                "section_uuids": [section_uuids[int(i)] for i in indices.tolist()],
            }

        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_parquet": self.cfg.data_parquet,
            "labels_column": self.cfg.labels_column,
            "label_list": label_list,
            "split_seed": self.cfg.split_seed,
            "val_size": self.cfg.val_size,
            "test_size": self.cfg.test_size,
            "counts": {
                "total": len(df),
                "train": int(train_indices.size),
                "val": int(val_indices.size),
                "test": int(test_indices.size),
            },
            "train": _payload(train_indices),
            "val": _payload(val_indices),
            "test": _payload(test_indices),
            "split_quality": self._build_split_quality_report(
                label_matrix=label_matrix,
                label_list=label_list,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
            ),
        }

        with split_path.open("w") as f:
            json.dump(manifest, f, indent=2)

    def _load_data(self) -> None:
        df: pd.DataFrame = pd.read_parquet(self.cfg.data_parquet)
        required_cols = {"article_title", "section_title", "section_text"}
        missing_required = sorted(required_cols - set(df.columns))
        if missing_required:
            raise ValueError(
                f"Training parquet missing required columns: {missing_required}."
            )

        label_col = self.cfg.labels_column if self.cfg.labels_column in df.columns else "label"
        if label_col not in df.columns:
            raise ValueError(
                f"Training parquet must contain `{self.cfg.labels_column}` or legacy `label` column."
            )

        df = df.dropna(subset=["section_text"]).copy()
        for col in ("article_title", "section_title", "section_text"):
            df[col] = cast(pd.Series, df[col]).fillna("").astype(str)

        parsed_labels: list[list[str]] = []
        raw_label_values = cast(list[object], cast(pd.Series, df[label_col]).tolist())
        for row_idx, raw in enumerate(raw_label_values):
            try:
                parsed = self._parse_labels_cell(raw)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid taxonomy labels at row index {row_idx}: {exc}"
                ) from exc
            parsed_labels.append(parsed)

        configured_label_list = self.cfg.label_list
        if configured_label_list is None:
            inferred_label_list = sorted(
                {
                    label
                    for row_labels in parsed_labels
                    for label in row_labels
                }
            )
            if not inferred_label_list:
                raise ValueError(
                    "Could not infer `label_list` from data: all rows have empty labels."
                )
            self.cfg.label_list = inferred_label_list
        else:
            if not configured_label_list:
                raise ValueError("`label_list` must be non-empty when provided.")
            if len(set(configured_label_list)) != len(configured_label_list):
                raise ValueError("`label_list` contains duplicate entries.")

        label_list = cast(list[str], self.cfg.label_list)
        label2id = {label: i for i, label in enumerate(label_list)}

        unknown_labels = sorted(
            {
                label
                for row_labels in parsed_labels
                for label in row_labels
                if label not in label2id
            }
        )
        if unknown_labels:
            preview = unknown_labels[:10]
            raise ValueError(
                f"Found taxonomy labels not present in `label_list`: {preview} (showing up to 10)."
            )

        label_id_rows: list[list[int]] = []
        for row_labels in parsed_labels:
            unique_ids: list[int] = []
            seen: set[int] = set()
            for label in row_labels:
                label_id = label2id[label]
                if label_id not in seen:
                    unique_ids.append(label_id)
                    seen.add(label_id)
            label_id_rows.append(unique_ids)

        df = df.reset_index(drop=True)
        df["label_vectors"] = [self._to_multi_hot(ids, len(label_list)) for ids in label_id_rows]
        label_matrix = np.asarray(df["label_vectors"].tolist(), dtype=np.int64)
        empty_label_column = (label_matrix.sum(axis=1) == 0).astype(np.int64).reshape(-1, 1)
        split_labels = np.concatenate((label_matrix, empty_label_column), axis=1)
        train_indices, val_indices, test_indices = self._split_train_val_test(
            labels=split_labels
        )
        tr = cast(pd.DataFrame, df.iloc[train_indices].copy()).reset_index(drop=True)
        va = cast(pd.DataFrame, df.iloc[val_indices].copy()).reset_index(drop=True)
        te = cast(pd.DataFrame, df.iloc[test_indices].copy()).reset_index(drop=True)
        train_label_matrix = np.asarray(
            cast(list[list[int]], cast(pd.Series, tr["label_vectors"]).tolist()),
            dtype=np.int64,
        )
        self.class_pos_weight = self._compute_class_pos_weight(
            train_label_matrix=train_label_matrix,
            label_list=label_list,
        )
        self._write_split_manifest(
            df=df,
            label_matrix=label_matrix,
            label_list=label_list,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        )

        if self.cfg.mode == "transformer":
            article_title_tr = cast(pd.Series, tr["article_title"]).astype(str).tolist()
            section_title_tr = cast(pd.Series, tr["section_title"]).astype(str).tolist()
            section_text_tr = cast(pd.Series, tr["section_text"]).astype(str).tolist()
            label_vectors_tr = cast(list[list[int]], cast(pd.Series, tr["label_vectors"]).tolist())
            article_title_va = cast(pd.Series, va["article_title"]).astype(str).tolist()
            section_title_va = cast(pd.Series, va["section_title"]).astype(str).tolist()
            section_text_va = cast(pd.Series, va["section_text"]).astype(str).tolist()
            label_vectors_va = cast(list[list[int]], cast(pd.Series, va["label_vectors"]).tolist())
            article_title_te = cast(pd.Series, te["article_title"]).astype(str).tolist()
            section_title_te = cast(pd.Series, te["section_title"]).astype(str).tolist()
            section_text_te = cast(pd.Series, te["section_text"]).astype(str).tolist()
            label_vectors_te = cast(list[list[int]], cast(pd.Series, te["label_vectors"]).tolist())
            self.train_rows = {
                "article_title": article_title_tr,
                "section_title": section_title_tr,
                "section_text": section_text_tr,
                "label_vectors": label_vectors_tr,
            }
            self.val_rows = {
                "article_title": article_title_va,
                "section_title": section_title_va,
                "section_text": section_text_va,
                "label_vectors": label_vectors_va,
            }
            self.test_rows = {
                "article_title": article_title_te,
                "section_title": section_title_te,
                "section_text": section_text_te,
                "label_vectors": label_vectors_te,
            }
        else:
            article_title_tr = cast(pd.Series, tr["article_title"]).astype(str).tolist()
            section_title_tr = cast(pd.Series, tr["section_title"]).astype(str).tolist()
            section_text_tr = cast(pd.Series, tr["section_text"]).astype(str).tolist()
            article_title_va = cast(pd.Series, va["article_title"]).astype(str).tolist()
            section_title_va = cast(pd.Series, va["section_title"]).astype(str).tolist()
            section_text_va = cast(pd.Series, va["section_text"]).astype(str).tolist()
            article_title_te = cast(pd.Series, te["article_title"]).astype(str).tolist()
            section_title_te = cast(pd.Series, te["section_title"]).astype(str).tolist()
            section_text_te = cast(pd.Series, te["section_text"]).astype(str).tolist()
            texts_tr = [
                _combine_text(str(a), str(s), str(t))
                for a, s, t in zip(article_title_tr, section_title_tr, section_text_tr)
            ]
            texts_va = [
                _combine_text(str(a), str(s), str(t))
                for a, s, t in zip(article_title_va, section_title_va, section_text_va)
            ]
            texts_te = [
                _combine_text(str(a), str(s), str(t))
                for a, s, t in zip(article_title_te, section_title_te, section_text_te)
            ]
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9,
                strip_accents="unicode",
                lowercase=True,
                max_features=self.cfg.tfidf_max_features,
            )
            Xtr = cast(
                _DenseConvertible,
                self.vectorizer.fit_transform(texts_tr),
            )
            Xva = cast(
                _DenseConvertible,
                self.vectorizer.transform(texts_va),
            )
            Xte = cast(
                _DenseConvertible,
                self.vectorizer.transform(texts_te),
            )

            # Convert to dense for a small MLP (could keep sparse for linear models)
            self.X_train = Xtr.astype("float32").toarray()
            self.X_val = Xva.astype("float32").toarray()
            self.X_test = Xte.astype("float32").toarray()
            self.y_train = np.asarray(
                cast(list[list[int]], cast(pd.Series, tr["label_vectors"]).tolist()),
                dtype="float32",
            )
            self.y_val = np.asarray(
                cast(list[list[int]], cast(pd.Series, va["label_vectors"]).tolist()),
                dtype="float32",
            )
            self.y_test = np.asarray(
                cast(list[list[int]], cast(pd.Series, te["label_vectors"]).tolist()),
                dtype="float32",
            )
            self.test_rows = {
                "article_title": article_title_te,
                "section_title": section_title_te,
                "section_text": section_text_te,
                "label_vectors": cast(
                    list[list[int]],
                    cast(pd.Series, te["label_vectors"]).tolist(),
                ),
            }

    def _get_callbacks(
        self, trial: Trial | None = None
    ) -> tuple[
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
        TQDMProgressBar,
        list[PyTorchLightningPruningCallback],
    ]:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_model_score",
            mode="max",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_model_score:.4f}",
        )
        early_stop_callback = EarlyStopping(
            monitor="val_model_score", patience=3, mode="max"
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        pruning_callback = (
            [PyTorchLightningPruningCallback(trial, monitor="val_model_score")]
            if trial is not None
            else []
        )
        progress_bar_callback = TQDMProgressBar(refresh_rate=100)
        return checkpoint_callback, early_stop_callback, lr_monitor, progress_bar_callback, pruning_callback

    def _build(
        self, params: dict[str, float | int]
    ) -> tuple[pl.LightningDataModule, pl.LightningModule]:
        label_list = self.cfg.label_list
        if not label_list:
            raise RuntimeError("`label_list` is not initialized. Call `_load_data()` first.")
        id2label = {i: l for i, l in enumerate(label_list)}
        if self.cfg.mode == "transformer":
            dm = TransformerDataModule(
                model_name=self.cfg.model_name,
                train_rows=self.train_rows,
                val_rows=self.val_rows,
                label_list=label_list,
                batch_size=int(params["batch_size"]),
                max_length=int(params.get("max_length", self.cfg.max_length)),
                num_workers=7,
            )
            model = TaxonomyClassifier(
                mode="transformer",
                num_labels=len(label_list),
                id2label=id2label,
                model_name=self.cfg.model_name,
                learning_rate=float(params["lr"]),
                weight_decay=float(params["weight_decay"]),
                warmup_steps_pct=float(params["warmup_steps_pct"]),
                decision_threshold=self.cfg.decision_threshold,
                pos_weight=self.class_pos_weight,
            )
        else:
            assert self.X_train is not None and self.y_train is not None
            assert self.X_val is not None and self.y_val is not None
            dm = TfidfDataModule(
                X_train=self.X_train,
                y_train=self.y_train,
                X_val=self.X_val,
                y_val=self.y_val,
                batch_size=int(params["batch_size"]),
                num_workers=3,
            )
            input_dim = int(cast(tuple[int, ...], self.X_train.shape)[1])
            model = TaxonomyClassifier(
                mode="tfidf",
                num_labels=len(label_list),
                id2label=id2label,
                input_dim=input_dim,
                hidden_dim=int(params.get("hidden_dim", 512)),
                dropout=float(params.get("dropout", 0.1)),
                learning_rate=float(params.get("lr", 1e-3)),
                weight_decay=float(params.get("weight_decay", 0.0)),
                decision_threshold=self.cfg.decision_threshold,
                pos_weight=self.class_pos_weight,
            )
        return dm, model

    def _objective(self, trial: Trial) -> float:
        params = {
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "lr": trial.suggest_float("lr", 1e-5 if self.cfg.mode == "transformer" else 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
            "warmup_steps_pct": trial.suggest_float("warmup_steps_pct", 0.0, 0.2) if self.cfg.mode == "transformer" else 0.0,
        }
        if self.cfg.mode == "tfidf":
            params.update(
                {
                    "hidden_dim": trial.suggest_categorical("hidden_dim", [256, 512, 768]),
                    "dropout": trial.suggest_float("dropout", 0.0, 0.3),
                }
            )

        dm, model = self._build(params)
        (
            checkpoint_callback,
            early_stop_callback,
            lr_monitor,
            progress_bar_callback,
            pruning_callback,
        ) = self._get_callbacks(trial)

        trainer = pl.Trainer(
            max_epochs=self.cfg.max_epochs,
            accelerator=self.device,
            precision=self._trainer_precision(),
            devices=1,
            logger=TensorBoardLogger(str(LOG_DIR), name="taxonomy-optuna"),
            callbacks=[
                checkpoint_callback,
                early_stop_callback,
                lr_monitor,
                progress_bar_callback,
                *pruning_callback,
            ],
            log_every_n_steps=10,
            deterministic=True,
        )
        trainer.fit(model, datamodule=dm)

        val_model_score = float(trainer.callback_metrics["val_model_score"])
        return val_model_score

    def _resolve_eval_metrics_path(self) -> Path:
        if self.cfg.eval_metrics_path:
            return Path(self.cfg.eval_metrics_path)
        return Path(TAXONOMY_EVAL_METRICS_PATH)

    def _copy_checkpoint_with_thresholds(
        self,
        source_checkpoint: str,
        destination_checkpoint: str,
        decision_threshold: float,
        decision_thresholds: list[float] | None = None,
    ) -> None:
        checkpoint_obj = cast(
            dict[str, object],
            torch.load(source_checkpoint, map_location="cpu"),
        )
        hyper_parameters = checkpoint_obj.get("hyper_parameters")
        if not isinstance(hyper_parameters, dict):
            raise ValueError(
                "Checkpoint is missing a mutable `hyper_parameters` mapping."
            )
        hyper_parameters["decision_threshold"] = float(decision_threshold)
        if decision_thresholds is not None:
            hyper_parameters["decision_thresholds"] = [
                float(t) for t in decision_thresholds
            ]
        torch.save(checkpoint_obj, destination_checkpoint)

    def _collect_val_probs_and_labels(
        self,
        model: TaxonomyClassifier,
        dm: pl.LightningDataModule,
    ) -> tuple[NDArray[np.float64], IntArray]:
        dm.setup("fit")
        val_loader = dm.val_dataloader()
        model = model.to(self.device)
        _ = model.eval()

        probs_batches: list[np.ndarray] = []
        label_batches: list[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    labels_tensor = cast(torch.Tensor, batch["labels"]).to(self.device)
                    model_inputs: dict[str, torch.Tensor] = {
                        key: value.to(self.device)
                        for key, value in batch.items()
                        if key in ("input_ids", "attention_mask", "token_type_ids")
                    }
                    outputs = cast(_LogitsOutput, model(**model_inputs))
                else:
                    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                        raise ValueError(
                            "Validation batch must be dict or length-2 tuple/list."
                        )
                    features_tensor = cast(torch.Tensor, batch[0]).to(self.device)
                    labels_tensor = cast(torch.Tensor, batch[1]).to(self.device)
                    outputs = cast(_LogitsOutput, model(features=features_tensor))
                probs_tensor = torch.sigmoid(outputs.logits)
                probs_batches.append(
                    probs_tensor.detach().cpu().numpy().astype(np.float64)
                )
                label_batches.append(
                    labels_tensor.detach().cpu().numpy().astype(np.int64)
                )

        if not probs_batches:
            num_labels = int(model.num_labels)
            empty_probs = np.zeros((0, num_labels), dtype=np.float64)
            empty_labels = np.zeros((0, num_labels), dtype=np.int64)
            return empty_probs, empty_labels

        probs = np.vstack(probs_batches).astype(np.float64)
        labels = np.vstack(label_batches).astype(np.int64)
        return probs, labels

    def _apply_thresholds_with_top1_fallback(
        self,
        probs: NDArray[np.float64],
        thresholds: NDArray[np.float64],
    ) -> IntArray:
        if probs.ndim != 2:
            raise ValueError(f"Expected 2D probability matrix; got ndim={probs.ndim}.")
        if thresholds.ndim != 1:
            raise ValueError(
                f"Expected 1D threshold vector; got ndim={thresholds.ndim}."
            )
        if probs.shape[1] != thresholds.shape[0]:
            raise ValueError(
                "Threshold count must equal number of probability columns."
            )
        if probs.shape[0] == 0:
            return np.zeros((0, probs.shape[1]), dtype=np.int64)

        pred_mask = probs >= thresholds[np.newaxis, :]
        no_label_rows = pred_mask.sum(axis=1) == 0
        if np.any(no_label_rows):
            top_ids = np.argmax(probs[no_label_rows], axis=1)
            rows = np.where(no_label_rows)[0]
            pred_mask[rows, top_ids] = True
        return pred_mask.astype(np.int64)

    def _binary_f1_score(
        self,
        y_true: IntArray | NDArray[np.int64],
        y_pred: IntArray | NDArray[np.int64],
    ) -> float:
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        denominator = (2 * tp) + fp + fn
        if denominator == 0:
            return 0.0
        return float((2 * tp) / denominator)

    def _compute_f1_scores(
        self,
        y_true: IntArray,
        y_pred: IntArray,
    ) -> dict[str, float]:
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Expected equal shapes; got y_true={y_true.shape}, y_pred={y_pred.shape}."
            )
        if y_true.ndim != 2:
            raise ValueError(f"Expected 2D label matrices; got ndim={y_true.ndim}.")

        f1_micro = self._binary_f1_score(y_true=y_true, y_pred=y_pred)
        per_label_f1: list[float] = []
        supports: list[int] = []
        for idx in range(y_true.shape[1]):
            y_true_col = y_true[:, idx]
            y_pred_col = y_pred[:, idx]
            per_label_f1.append(
                self._binary_f1_score(y_true=y_true_col, y_pred=y_pred_col)
            )
            supports.append(int(np.sum(y_true_col == 1)))
        f1_macro = float(np.mean(np.asarray(per_label_f1, dtype=np.float64)))
        support_sum = int(np.sum(np.asarray(supports, dtype=np.int64)))
        if support_sum == 0:
            f1_weighted = 0.0
        else:
            f1_weighted = float(
                np.average(
                    np.asarray(per_label_f1, dtype=np.float64),
                    weights=np.asarray(supports, dtype=np.float64),
                )
            )
        return {
            "f1_micro": float(f1_micro),
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }

    def _calibration_objective_score(self, f1_macro: float, f1_weighted: float) -> float:
        return float((0.6 * f1_macro) + (0.4 * f1_weighted))

    def _tune_per_class_thresholds(
        self,
        checkpoint_path: str,
        dm: pl.LightningDataModule,
    ) -> dict[str, object]:
        model = TaxonomyClassifier.load_from_checkpoint(
            checkpoint_path, map_location=self.device
        )
        probs, y_true = self._collect_val_probs_and_labels(model=model, dm=dm)
        label_list = self.cfg.label_list
        if not label_list:
            raise RuntimeError("`label_list` is not initialized. Call `_load_data()` first.")
        if probs.shape[1] != len(label_list):
            raise RuntimeError(
                "Validation probability width does not match `label_list` length."
            )

        thresholds = np.round(np.arange(0.05, 0.951, 0.01), 3)
        threshold_values = [float(t) for t in thresholds.tolist()]
        baseline_threshold = float(self.cfg.decision_threshold)
        if baseline_threshold not in threshold_values:
            threshold_values.append(baseline_threshold)
            threshold_values.sort()

        best_thresholds = np.full(probs.shape[1], baseline_threshold, dtype=np.float64)
        per_label: dict[str, dict[str, float | int]] = {}
        for idx, label in enumerate(label_list):
            y_true_col = y_true[:, idx]
            support = int(np.sum(y_true_col == 1))
            best_threshold = baseline_threshold
            best_f1 = -1.0
            for threshold in threshold_values:
                y_pred_col = (probs[:, idx] >= threshold).astype(np.int64)
                f1 = self._binary_f1_score(y_true=y_true_col, y_pred=y_pred_col)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = float(threshold)
                elif f1 == best_f1:
                    if abs(threshold - baseline_threshold) < abs(
                        best_threshold - baseline_threshold
                    ):
                        best_threshold = float(threshold)

            best_thresholds[idx] = best_threshold
            per_label[label] = {
                "support": support,
                "best_threshold": float(best_threshold),
                "best_f1": float(best_f1),
            }

        y_pred_tuned = self._apply_thresholds_with_top1_fallback(
            probs=probs,
            thresholds=best_thresholds,
        )
        val_scores = self._compute_f1_scores(y_true=y_true, y_pred=y_pred_tuned)
        val_model_score = self._calibration_objective_score(
            f1_macro=val_scores["f1_macro"],
            f1_weighted=val_scores["f1_weighted"],
        )
        return {
            "method": "per_class_f1_sweep",
            "objective_name": "val_model_score",
            "objective_formula": "0.6 * val_f1_macro + 0.4 * val_f1_weighted",
            "global_baseline_threshold": baseline_threshold,
            "decision_thresholds": [float(t) for t in best_thresholds.tolist()],
            "decision_thresholds_by_label": {
                label: float(best_thresholds[idx]) for idx, label in enumerate(label_list)
            },
            "val_scores_at_selected_thresholds": {
                "f1_micro": float(val_scores["f1_micro"]),
                "f1_macro": float(val_scores["f1_macro"]),
                "f1_weighted": float(val_scores["f1_weighted"]),
                "val_model_score": float(val_model_score),
            },
            # Backward-compatible key for existing analysis tooling.
            "best_val_f1_micro": float(val_scores["f1_micro"]),
            "num_val_rows": int(y_true.shape[0]),
            "per_label": per_label,
            "search_space": {
                "min_threshold": 0.05,
                "max_threshold": 0.95,
                "step": 0.01,
            },
        }

    def _build_test_metrics_payload(
        self,
        y_true: IntArray,
        y_pred: IntArray,
        label_list: list[str],
        checkpoint_path: str,
        decision_thresholds: list[float] | None = None,
        threshold_sweep: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch for test metrics: y_true={y_true.shape}, y_pred={y_pred.shape}."
            )
        if y_true.ndim != 2:
            raise ValueError(f"Expected 2D label matrices; got y_true.ndim={y_true.ndim}.")

        def _safe_div(numerator: int, denominator: int) -> float:
            if denominator == 0:
                return 0.0
            return float(numerator / denominator)

        num_rows = int(y_true.shape[0])
        num_labels = int(y_true.shape[1])

        per_class: dict[str, dict[str, object]] = {}
        class_precision_values: list[float] = []
        class_recall_values: list[float] = []
        class_f1_values: list[float] = []
        class_acc_values: list[float] = []
        class_support_values: list[int] = []

        tp_total = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn_total = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp_total = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn_total = int(np.sum((y_true == 1) & (y_pred == 0)))

        for idx, label in enumerate(label_list):
            y_true_col = y_true[:, idx]
            y_pred_col = y_pred[:, idx]
            tp = int(np.sum((y_true_col == 1) & (y_pred_col == 1)))
            tn = int(np.sum((y_true_col == 0) & (y_pred_col == 0)))
            fp = int(np.sum((y_true_col == 0) & (y_pred_col == 1)))
            fn = int(np.sum((y_true_col == 1) & (y_pred_col == 0)))
            support = int(np.sum(y_true_col == 1))

            precision = _safe_div(tp, tp + fp)
            recall = _safe_div(tp, tp + fn)
            f1 = _safe_div(2 * tp, (2 * tp) + fp + fn)
            accuracy = _safe_div(tp + tn, num_rows)

            class_precision_values.append(precision)
            class_recall_values.append(recall)
            class_f1_values.append(f1)
            class_acc_values.append(accuracy)
            class_support_values.append(support)

            per_class[label] = {
                "support": support,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": [[tn, fp], [fn, tp]],
            }

        if num_rows == 0:
            exact_match_accuracy = 0.0
            hamming_loss = 0.0
        else:
            exact_match_accuracy = float(np.mean(np.all(y_true == y_pred, axis=1)))
            hamming_loss = float(np.mean(y_true != y_pred))
        hamming_accuracy = 1.0 - hamming_loss
        support_sum = int(np.sum(np.asarray(class_support_values, dtype=np.int64)))

        payload: dict[str, object] = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": self.cfg.mode,
            "checkpoint_path": checkpoint_path,
            "decision_threshold": self.cfg.decision_threshold,
            "decision_thresholds": decision_thresholds,
            "counts": {
                "num_rows": num_rows,
                "num_labels": num_labels,
                "positive_truth": int(np.sum(y_true)),
                "positive_predictions": int(np.sum(y_pred)),
            },
            "overall": {
                "accuracy_micro": _safe_div(tp_total + tn_total, tp_total + tn_total + fp_total + fn_total),
                "precision_micro": _safe_div(tp_total, tp_total + fp_total),
                "recall_micro": _safe_div(tp_total, tp_total + fn_total),
                "f1_micro": _safe_div(2 * tp_total, (2 * tp_total) + fp_total + fn_total),
                "accuracy_macro": float(np.mean(np.asarray(class_acc_values, dtype=np.float64))),
                "precision_macro": float(np.mean(np.asarray(class_precision_values, dtype=np.float64))),
                "recall_macro": float(np.mean(np.asarray(class_recall_values, dtype=np.float64))),
                "f1_macro": float(np.mean(np.asarray(class_f1_values, dtype=np.float64))),
                "precision_weighted": float(
                    np.average(
                        np.asarray(class_precision_values, dtype=np.float64),
                        weights=np.asarray(class_support_values, dtype=np.float64),
                    )
                )
                if support_sum > 0
                else 0.0,
                "recall_weighted": float(
                    np.average(
                        np.asarray(class_recall_values, dtype=np.float64),
                        weights=np.asarray(class_support_values, dtype=np.float64),
                    )
                )
                if support_sum > 0
                else 0.0,
                "f1_weighted": float(
                    np.average(
                        np.asarray(class_f1_values, dtype=np.float64),
                        weights=np.asarray(class_support_values, dtype=np.float64),
                    )
                )
                if support_sum > 0
                else 0.0,
                "subset_accuracy": exact_match_accuracy,
                "hamming_accuracy": hamming_accuracy,
                "hamming_loss": hamming_loss,
            },
            "per_class": per_class,
        }
        if threshold_sweep is not None:
            payload["threshold_sweep"] = threshold_sweep
        return payload

    def _evaluate_test_and_save_metrics(
        self,
        checkpoint_path: str,
        decision_thresholds: list[float] | None = None,
        threshold_sweep: dict[str, object] | None = None,
    ) -> Path:
        label_list = self.cfg.label_list
        if not label_list:
            raise RuntimeError("`label_list` is not initialized. Call `_load_data()` first.")

        label_vectors_obj = self.test_rows.get("label_vectors")
        article_title_obj = self.test_rows.get("article_title")
        section_title_obj = self.test_rows.get("section_title")
        section_text_obj = self.test_rows.get("section_text")
        if not isinstance(label_vectors_obj, list):
            raise RuntimeError("Missing `test_rows['label_vectors']` for test evaluation.")
        if not isinstance(article_title_obj, list):
            raise RuntimeError("Missing `test_rows['article_title']` for test evaluation.")
        if not isinstance(section_title_obj, list):
            raise RuntimeError("Missing `test_rows['section_title']` for test evaluation.")
        if not isinstance(section_text_obj, list):
            raise RuntimeError("Missing `test_rows['section_text']` for test evaluation.")

        y_true = np.asarray(cast(list[list[int]], label_vectors_obj), dtype=np.int64)
        test_inputs = [
            {
                "article_title": str(a),
                "section_title": str(s),
                "section_text": str(t),
            }
            for a, s, t in zip(article_title_obj, section_title_obj, section_text_obj)
        ]

        inference = TaxonomyInference(
            ckpt_path=checkpoint_path,
            label_list=label_list,
            mode=self.cfg.mode,
            model_name=self.cfg.model_name if self.cfg.mode == "transformer" else None,
            vectorizer_path=TAXONOMY_VECTORIZER_PATH if self.cfg.mode == "tfidf" else None,
            max_length=self.cfg.max_length,
            decision_threshold=None,
            decision_thresholds=decision_thresholds,
        )
        pred_rows = inference.predict(test_inputs)
        if len(pred_rows) != len(test_inputs):
            raise RuntimeError(
                f"Prediction row count mismatch for test evaluation: {len(pred_rows)} != {len(test_inputs)}."
            )

        y_pred = np.zeros((len(test_inputs), len(label_list)), dtype=np.int64)
        for row_idx, pred_map in enumerate(pred_rows):
            pred_ids_obj = pred_map.get("pred_ids")
            if not isinstance(pred_ids_obj, list):
                raise RuntimeError(f"Prediction row {row_idx} missing `pred_ids` list.")
            for raw_label_id in pred_ids_obj:
                label_id = int(raw_label_id)
                if not 0 <= label_id < len(label_list):
                    raise RuntimeError(
                        f"Predicted label id {label_id} out of range at row {row_idx}."
                    )
                y_pred[row_idx, label_id] = 1

        payload = self._build_test_metrics_payload(
            y_true=y_true,
            y_pred=y_pred,
            label_list=label_list,
            checkpoint_path=checkpoint_path,
            decision_thresholds=decision_thresholds,
            threshold_sweep=threshold_sweep,
        )
        metrics_path = self._resolve_eval_metrics_path()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w") as f:
            json.dump(payload, f, indent=2)
        return metrics_path

    def run(self) -> None:
        self._load_data()

        study = create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.cfg.num_trials, gc_after_trial=True)

        print("Finished hyperparameter optimization 👉")
        print(f"  Best val_model_score: {study.best_value:.4f}")
        print("  Best hyperparameters:")
        best_params = cast(dict[str, float | int], study.best_trial.params)
        for key, value in best_params.items():
            print(f"    • {key}: {value}")

        dm, model = self._build(best_params)
        (
            checkpoint_callback,
            early_stop_callback,
            lr_monitor,
            progress_bar_callback,
            _,
        ) = self._get_callbacks()

        trainer = pl.Trainer(
            max_epochs=self.cfg.max_epochs,
            accelerator=self.device,
            precision=self._trainer_precision(),
            devices=1,
            logger=TensorBoardLogger(str(LOG_DIR), name="taxonomy-final"),
            callbacks=[
                checkpoint_callback,
                early_stop_callback,
                lr_monitor,
                progress_bar_callback,
            ],
            log_every_n_steps=10,
            deterministic=True,
        )
        trainer.fit(model, datamodule=dm)

        best_ckpt_path = cast(str, checkpoint_callback.best_model_path)
        if not best_ckpt_path:
            raise RuntimeError("No best checkpoint produced during training.")
        threshold_sweep = self._tune_per_class_thresholds(
            checkpoint_path=best_ckpt_path,
            dm=dm,
        )
        tuned_thresholds = cast(list[float], threshold_sweep["decision_thresholds"])
        self.tuned_decision_thresholds = [float(x) for x in tuned_thresholds]
        tuned_val_scores = cast(dict[str, float], threshold_sweep["val_scores_at_selected_thresholds"])
        tuned_threshold_mean = float(np.mean(np.asarray(self.tuned_decision_thresholds, dtype=np.float64)))
        self.cfg.decision_threshold = tuned_threshold_mean
        tuning_summary = (
            "Selected per-class decision thresholds "
            + "(val_model_score={val_model_score:.4f}, val_f1_macro={f1_macro:.4f}, "
            + "val_f1_weighted={f1_weighted:.4f}, val_f1_micro={f1_micro:.4f})"
        ).format(**tuned_val_scores)
        print(tuning_summary)

        Path(TAXONOMY_CKPT_PATH).parent.mkdir(parents=True, exist_ok=True)
        self._copy_checkpoint_with_thresholds(
            source_checkpoint=best_ckpt_path,
            destination_checkpoint=TAXONOMY_CKPT_PATH,
            decision_threshold=tuned_threshold_mean,
            decision_thresholds=self.tuned_decision_thresholds,
        )

        # Save TF-IDF vectorizer if needed
        if self.cfg.mode == "tfidf" and self.vectorizer is not None:
            Path(TAXONOMY_VECTORIZER_PATH).parent.mkdir(parents=True, exist_ok=True)
            with open(TAXONOMY_VECTORIZER_PATH, "wb") as f:
                pickle.dump(self.vectorizer, f)

        eval_metrics_path = self._evaluate_test_and_save_metrics(
            checkpoint_path=TAXONOMY_CKPT_PATH,
            decision_thresholds=self.tuned_decision_thresholds,
            threshold_sweep=threshold_sweep,
        )
        print(f"Saved test evaluation metrics to {eval_metrics_path}")

    def _trainer_precision(self) -> PrecisionInput:
        if self.device in ("mps", "cuda"):
            return "bf16-mixed"
        return "32-true"


class TaxonomyInference:
    """
    Inference for taxonomy classification. Supports transformer or tfidf modes.
    """

    def __init__(
        self,
        ckpt_path: str,
        label_list: list[str] | None,
        mode: Literal["transformer", "tfidf"],
        model_name: str | None = None,
        vectorizer_path: str | None = None,
        max_length: int = 1024,
        decision_threshold: float | None = None,
        decision_thresholds: list[float] | None = None,
    ) -> None:
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model: TaxonomyClassifier
        self.mode: Literal["transformer", "tfidf"]
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.vectorizer: TfidfVectorizer | None = None

        if mode == "transformer":
            model = TaxonomyClassifier.load_from_checkpoint(
                ckpt_path, map_location=self.device
            )
            if model.mode != "transformer":
                raise ValueError("Checkpoint is not a transformer taxonomy model.")
            _ = model.eval()
            _ = model.to(self.device)
            self.model = model
            self.mode = "transformer"
            self.max_length = max_length
            hparams = cast(object, model.hparams)
            tokenizer_name = model_name or cast(
                str | None, getattr(hparams, "model_name", None)
            )
            if tokenizer_name is None:
                raise ValueError("Transformer inference requires `model_name` (or a checkpoint that saved it).")
            self.tokenizer = cast(
                PreTrainedTokenizerBase,
                AutoTokenizer.from_pretrained(
                    tokenizer_name, use_fast=True
                ),
            )
        else:
            model = TaxonomyClassifier.load_from_checkpoint(
                ckpt_path, map_location=self.device
            )
            if model.mode != "tfidf":
                raise ValueError("Checkpoint is not a TF-IDF taxonomy model.")
            _ = model.eval()
            _ = model.to(self.device)
            self.model = model
            self.mode = "tfidf"
            with open(vectorizer_path or TAXONOMY_VECTORIZER_PATH, "rb") as f:
                self.vectorizer = pickle.load(f)

        model_id2label = getattr(self.model, "id2label", None)
        if label_list is not None:
            expected_id2label = {i: l for i, l in enumerate(label_list)}
            if model_id2label is not None and model_id2label != expected_id2label:
                raise ValueError("`label_list` does not match the label mapping stored in the checkpoint.")
            self.id2label = expected_id2label
        else:
            if not isinstance(model_id2label, dict):
                raise ValueError(
                    "Checkpoint does not expose a valid `id2label`; provide `label_list` explicitly."
                )
            normalized_id2label: dict[int, str] = {}
            for raw_idx, raw_label in model_id2label.items():
                try:
                    label_idx = int(raw_idx)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Checkpoint `id2label` key is not integer-like: {raw_idx!r}"
                    ) from exc
                if not isinstance(raw_label, str):
                    raise ValueError(
                        f"Checkpoint `id2label[{label_idx}]` must be a string; got {type(raw_label)!r}."
                    )
                normalized_id2label[label_idx] = raw_label

            if not normalized_id2label:
                raise ValueError("Checkpoint `id2label` is empty.")
            max_idx = max(normalized_id2label)
            expected_keys = list(range(max_idx + 1))
            if sorted(normalized_id2label) != expected_keys:
                raise ValueError(
                    "Checkpoint `id2label` keys must be contiguous and start at 0."
                )
            self.id2label = {idx: normalized_id2label[idx] for idx in expected_keys}
        hparams = cast(object, self.model.hparams)
        resolved_thresholds: list[float] | None = decision_thresholds
        if resolved_thresholds is None:
            raw_thresholds = getattr(hparams, "decision_thresholds", None)
            if isinstance(raw_thresholds, (list, tuple)):
                resolved_thresholds = [float(t) for t in raw_thresholds]

        num_labels = len(self.id2label)
        if resolved_thresholds is not None:
            if len(resolved_thresholds) != num_labels:
                raise ValueError(
                    f"`decision_thresholds` length ({len(resolved_thresholds)}) does not match number of labels ({num_labels})."
                )
            for threshold in resolved_thresholds:
                if not 0.0 < threshold < 1.0:
                    raise ValueError(
                        f"Each value in `decision_thresholds` must be in (0, 1); got {threshold!r}."
                    )
            self.decision_thresholds = [float(t) for t in resolved_thresholds]
        else:
            threshold = decision_threshold
            if threshold is None:
                threshold = float(getattr(hparams, "decision_threshold", 0.5))
            if not 0.0 < threshold < 1.0:
                raise ValueError(f"`decision_threshold` must be in (0, 1); got {threshold!r}.")
            self.decision_thresholds = [float(threshold)] * num_labels

        self.decision_threshold = float(
            np.mean(np.asarray(self.decision_thresholds, dtype=np.float64))
        )
        self.decision_threshold_tensor = torch.tensor(
            self.decision_thresholds,
            dtype=torch.float32,
            device=self.device,
        )

    def _combine(self, a: str, s: str, t: str) -> str:
        return _combine_text(a, s, t)

    def predict(self, rows: list[dict[str, str]]) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        batch_size = 32
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            texts = [
                self._combine(r.get("article_title", ""), r.get("section_title", ""), r.get("section_text", ""))
                for r in batch_rows
            ]

            if self.mode == "transformer":
                if self.tokenizer is None:
                    raise RuntimeError(
                        "Tokenizer not initialized for transformer inference."
                    )
                tokenizer = self.tokenizer
                assert tokenizer is not None
                enc = cast(
                    dict[str, torch.Tensor],
                    cast(
                        object,
                        tokenizer(
                            texts,
                            truncation=True,
                            max_length=self.max_length,
                            padding=True,
                            return_tensors="pt",
                        ),
                    ),
                )
                enc_tensors: dict[str, torch.Tensor] = {
                    k: v.to(self.device) for k, v in enc.items()
                }
                with torch.no_grad():
                    outputs = cast(
                        _LogitsOutput, self.model(**enc_tensors)
                    )
                    logits = outputs.logits
                    probs = torch.sigmoid(logits)
                    conf, pred = probs.max(dim=-1)
            else:
                if self.vectorizer is None:
                    raise RuntimeError("Vectorizer not initialized for TF-IDF inference.")
                features = cast(
                    _DenseConvertible,
                    self.vectorizer.transform(texts),
                ).astype("float32").toarray()
                features_tensor = torch.tensor(
                    features, dtype=torch.float32, device=self.device
                )
                with torch.no_grad():
                    outputs = cast(
                        _LogitsOutput, self.model(features=features_tensor)
                    )
                    logits = outputs.logits
                    probs = torch.sigmoid(logits)
                    conf, pred = probs.max(dim=-1)

            pred_mask = probs >= self.decision_threshold_tensor.unsqueeze(0)

            for i in range(len(texts)):
                top_pred_id = int(pred[i].item())
                selected_ids = [
                    idx
                    for idx, is_selected in enumerate(
                        cast(list[bool], pred_mask[i].tolist())
                    )
                    if is_selected
                ]
                if not selected_ids:
                    selected_ids = [top_pred_id]
                selected_labels = [self.id2label[label_id] for label_id in selected_ids]
                selected_probs = [
                    float(probs[i, label_id].item())
                    for label_id in selected_ids
                ]
                out.append(
                    {
                        "pred_ids": selected_ids,
                        "pred_labels": selected_labels,
                        "pred_probs": selected_probs,
                        "top_pred_id": top_pred_id,
                        "top_pred_label": self.id2label[top_pred_id],
                        "top_confidence": float(conf[i].item()),
                        # backward-compatible top-1 keys
                        "pred_id": top_pred_id,
                        "pred_label": self.id2label[top_pred_id],
                        "confidence": float(conf[i].item()),
                    }
                )
        return out


def _parse_cli_args() -> argparse.Namespace:
    data_dir = Path(__file__).resolve().parent / "data"
    default_data_parquet = str(data_dir / "taxonomy-data.parquet")

    parser = argparse.ArgumentParser(
        description="Train or run inference for the taxonomy model."
    )
    _ = parser.add_argument(
        "--mode",
        choices=("train", "test"),
        default="train",
        help="Run training or a local inference test.",
    )
    _ = parser.add_argument(
        "--model-mode",
        choices=("transformer", "tfidf"),
        default="tfidf",
        help="Model family for training/inference.",
    )
    _ = parser.add_argument(
        "--data-parquet",
        default=default_data_parquet,
        help="Training parquet path.",
    )
    _ = parser.add_argument(
        "--model-name",
        default="answerdotai/ModernBERT-base",
        help="Hugging Face model name (transformer mode).",
    )
    _ = parser.add_argument("--num-trials", type=int, default=10)
    _ = parser.add_argument("--max-epochs", type=int, default=6)
    _ = parser.add_argument("--batch-size", type=int, default=16)
    _ = parser.add_argument("--max-length", type=int, default=1024)
    _ = parser.add_argument("--labels-column", default="labels")
    _ = parser.add_argument("--val-size", type=float, default=0.2)
    _ = parser.add_argument("--test-size", type=float, default=0.1)
    _ = parser.add_argument("--split-path", default=None)
    _ = parser.add_argument("--split-seed", type=int, default=42)
    _ = parser.add_argument("--eval-metrics-path", default=None)
    _ = parser.add_argument("--decision-threshold", type=float, default=None)
    _ = parser.add_argument("--tfidf-max-features", type=int, default=50_000)

    _ = parser.add_argument("--ckpt-path", default=TAXONOMY_CKPT_PATH)
    _ = parser.add_argument("--vectorizer-path", default=None)
    _ = parser.add_argument(
        "--samples-path",
        default=None,
        help="JSON path with inference samples for --mode test.",
    )
    return parser.parse_args()


def _load_test_samples(samples_path: str | None) -> list[dict[str, str]]:
    if samples_path is None:
        return [
            {
                "article_title": "Representations and Warranties",
                "section_title": "Buyer Representations",
                "section_text": "The Buyer represents that it is duly organized and in good standing...",
            },
            {
                "article_title": "Miscellaneous",
                "section_title": "Notices",
                "section_text": "Any notice shall be deemed given when delivered by certified mail...",
            },
        ]

    path = Path(samples_path)
    payload = cast(object, json.loads(path.read_text()))
    if not isinstance(payload, list):
        raise ValueError("`samples_path` JSON must be a list of sample row objects.")

    samples: list[dict[str, str]] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"Sample row {idx} is not an object.")
        samples.append(
            {
                "article_title": str(row.get("article_title", "")),
                "section_title": str(row.get("section_title", "")),
                "section_text": str(row.get("section_text", "")),
            }
        )
    return samples


def main() -> None:
    """
    CLI entry point for taxonomy training or inference.
    """
    args = _parse_cli_args()
    if args.mode == "train":
        cfg = TaxonomyConfig(
            data_parquet=str(args.data_parquet),
            model_name=str(args.model_name),
            label_list=None,
            mode=cast(Literal["transformer", "tfidf"], args.model_mode),
            num_trials=int(args.num_trials),
            max_epochs=int(args.max_epochs),
            batch_size=int(args.batch_size),
            max_length=int(args.max_length),
            labels_column=str(args.labels_column),
            val_size=float(args.val_size),
            test_size=float(args.test_size),
            split_path=cast(str | None, args.split_path),
            split_seed=int(args.split_seed),
            eval_metrics_path=cast(str | None, args.eval_metrics_path),
            decision_threshold=(
                float(args.decision_threshold)
                if args.decision_threshold is not None
                else 0.5
            ),
            tfidf_max_features=int(args.tfidf_max_features),
        )
        trainer = TaxonomyTrainer(cfg)
        trainer.run()
        return

    inf = TaxonomyInference(
        ckpt_path=str(args.ckpt_path),
        label_list=None,
        mode=cast(Literal["transformer", "tfidf"], args.model_mode),
        model_name=str(args.model_name) if args.model_mode == "transformer" else None,
        vectorizer_path=cast(str | None, args.vectorizer_path),
        max_length=int(args.max_length),
        decision_threshold=cast(float | None, args.decision_threshold),
    )
    samples = _load_test_samples(cast(str | None, args.samples_path))
    start = time.time()
    preds = inf.predict(samples)
    elapsed = time.time() - start
    print(json.dumps(preds, indent=2))
    print(f"Inference time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
