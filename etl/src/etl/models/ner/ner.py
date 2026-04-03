"""
Main NER training and inference module.

This module provides the main entry points for training and testing the NER model
using PyTorch Lightning with hyperparameter optimization via Optuna.
"""
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false

# Standard library
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import yaml
from typing import Callable, Literal, TYPE_CHECKING, TypeAlias, cast

# Environment config
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Data manipulation
import pandas as pd
from sklearn.model_selection import train_test_split

# ML frameworks and utilities
import torch

torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from optuna import create_study, Trial
from optuna.integration import PyTorchLightningPruningCallback

# Local modules
if TYPE_CHECKING:
    from .entity_audit import ArticleFailureRecord, PageFailureRecord
    from .config import (
        NERExperimentConfig,
        FROZEN_EXPERIMENT_CONFIG_PATH,
        OPTUNA_BEST_CONFIG_PATH,
        build_config,
        config_to_dict,
        load_frozen_experiment_config,
        resolve_git_commit,
    )
    from .ner_constants import (
        NER_LABEL_LIST,
        NER_CKPT_PATH,
        SPECIAL_TOKENS_TO_ADD,
    )
    from .utils.paths import (
        CONFIG_NER_DIR,
        DATA_NER_DIR,
        EVAL_NER_DIR,
        LOG_NER_DIR,
        get_job_id,
    )
    from .ner_classes import NERTagger, NERDataModule, ascii_lower
else:
    try:
        from .config import (
        NERExperimentConfig,
        FROZEN_EXPERIMENT_CONFIG_PATH,
        OPTUNA_BEST_CONFIG_PATH,
        build_config,
        config_to_dict,
        load_frozen_experiment_config,
        resolve_git_commit,
        )
        from .ner_constants import (
            NER_LABEL_LIST,
            NER_CKPT_PATH,
            SPECIAL_TOKENS_TO_ADD,
        )
        from .utils.paths import (
            CONFIG_NER_DIR,
            DATA_NER_DIR,
            EVAL_NER_DIR,
            LOG_NER_DIR,
            get_job_id,
        )
        from .ner_classes import NERTagger, NERDataModule, ascii_lower
    except ImportError:  # pragma: no cover - supports running as a script
        from config import (
        NERExperimentConfig,
        FROZEN_EXPERIMENT_CONFIG_PATH,
        OPTUNA_BEST_CONFIG_PATH,
        build_config,
        config_to_dict,
        load_frozen_experiment_config,
        resolve_git_commit,
        )
        from ner_constants import (
            NER_LABEL_LIST,
            NER_CKPT_PATH,
            SPECIAL_TOKENS_TO_ADD,
        )
        from utils.paths import (
            CONFIG_NER_DIR,
            DATA_NER_DIR,
            EVAL_NER_DIR,
            LOG_NER_DIR,
            get_job_id,
        )
        from ner_classes import (
            NERTagger,
            NERDataModule,
            ascii_lower,
        )


def _load_entity_audit_builders() -> tuple[
    Callable[..., list["ArticleFailureRecord"]],
    Callable[..., list["PageFailureRecord"]],
]:
    try:
        from .entity_audit import (
            build_article_failure_records,
            build_page_failure_records,
        )
    except ImportError:  # pragma: no cover - supports running as a script
        from entity_audit import (
            build_article_failure_records,
            build_page_failure_records,
        )
    return build_article_failure_records, build_page_failure_records

PrecisionInput = Literal["bf16-mixed", "32-true"]
EvalSplit = Literal["val", "test"]
WindowItem: TypeAlias = dict[str, list[int]]

# Reproducibility
_ = seed_everything(42, workers=True, verbose=False)

SPLITS_DIR = CONFIG_NER_DIR / "splits"

NER_LABELS: list[str] = [str(label) for label in NER_LABEL_LIST]
NER_CKPT: str = str(NER_CKPT_PATH)
SPECIAL_TOKENS: list[str] = [str(token) for token in SPECIAL_TOKENS_TO_ADD]
YEAR_WINDOW = 5


def _split_path_for_version(split_version: str) -> str:
    version = split_version.strip() or "default"
    return str(SPLITS_DIR / f"ner-agreement-splits-{version}.json")


def _hash_sorted_ids(ids: list[str], seed: int) -> list[str]:
    keyed: list[tuple[str, str]] = []
    for pid in ids:
        digest = hashlib.sha1(f"{seed}:{pid}".encode("utf-8")).hexdigest()
        keyed.append((digest, pid))
    keyed.sort()
    return [pid for _, pid in keyed]


def _bucket_agreement_lengths(page_counts: pd.Series) -> pd.Series:
    buckets = pd.cut(
        page_counts,
        bins=[0, 10, 30, 80, float("inf")],
        labels=["xs", "sm", "md", "lg"],
        include_lowest=True,
    )
    if isinstance(buckets, tuple):
        raise TypeError("pd.cut returned unexpected tuple result.")
    return cast(pd.Series, buckets.astype(str))


def _bucket_agreement_tag_density(tag_rates: pd.Series) -> pd.Series:
    bucket_count = min(3, len(tag_rates))
    if bucket_count <= 1:
        return pd.Series(["mid"] * len(tag_rates), index=tag_rates.index, dtype="object")

    labels = ["low", "mid", "high"][:bucket_count]
    ranked = tag_rates.rank(method="first")
    buckets = pd.qcut(ranked, q=bucket_count, labels=labels)
    if isinstance(buckets, tuple):
        raise TypeError("pd.qcut returned unexpected tuple result.")
    return cast(pd.Series, buckets.astype(str))


def _build_strat_labels(
    frame: pd.DataFrame,
    strat_levels: list[list[str]],
    min_count: int,
) -> pd.Series | None:
    if len(frame) < min_count:
        return None

    assigned = pd.Series(index=frame.index, dtype="object")
    for level_idx, cols in enumerate(strat_levels):
        labels = cast(pd.Series, frame[cols].astype(str).agg("|".join, axis=1))
        counts = labels.value_counts()
        count_lookup = {str(key): int(value) for key, value in counts.items()}
        eligible = labels.map(count_lookup).fillna(0).astype(int) >= min_count
        fill_mask = assigned.isna() & eligible
        if not fill_mask.any():
            continue
        label_subset = labels.loc[fill_mask]
        assigned.loc[fill_mask] = pd.Series(
            [_format_strat_label(value, level_idx) for value in label_subset.tolist()],
            index=label_subset.index,
            dtype="object",
        )

    if assigned.isna().any():
        return None
    if cast(int, assigned.value_counts().min()) < min_count:
        return None
    return assigned


def _count_non_upsample_rows(values: pd.Series) -> int:
    bool_values = values.astype(bool)
    return int((~bool_values).sum())


def _format_strat_label(value: object, level_idx: int) -> str:
    return f"level{level_idx}:{value}"


def _requested_test_count(frame_size: int, test_size: float | int) -> int:
    if isinstance(test_size, int):
        return test_size
    return int(math.ceil(frame_size * test_size))


def _split_frame_with_stratification(
    frame: pd.DataFrame,
    test_size: float | int,
    strat_levels: list[list[str]],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(frame) == 0:
        empty_df = frame.iloc[0:0].copy()
        return empty_df, empty_df

    requested_test_count = _requested_test_count(len(frame), test_size)
    if requested_test_count <= 0 or requested_test_count >= len(frame):
        raise ValueError(
            f"Invalid split request: len(frame)={len(frame)} test_size={test_size!r}"
        )

    stratify_labels: pd.Series | None = None
    max_class_count = min(requested_test_count, len(frame) - requested_test_count)
    if max_class_count >= 2:
        candidate_labels = _build_strat_labels(
            frame, strat_levels=strat_levels, min_count=2
        )
        if (
            candidate_labels is not None
            and candidate_labels.nunique() <= max_class_count
        ):
            stratify_labels = candidate_labels

    return cast(
        tuple[pd.DataFrame, pd.DataFrame],
        cast(
            object,
            train_test_split(
                frame,
                test_size=test_size,
                stratify=stratify_labels,
                random_state=seed,
            ),
        ),
    )


def _target_split_counts(
    total_items: int,
    val_split: float,
    test_split: float,
) -> tuple[int, int]:
    total_holdout = int(round(total_items * (val_split + test_split)))
    if total_holdout <= 0:
        return 0, 0
    test_target = int(round(total_holdout * (test_split / (val_split + test_split))))
    val_target = total_holdout - test_target
    return val_target, test_target


def _eval_job_dir() -> str:
    return str(EVAL_NER_DIR / get_job_id())


def _eval_trials_dir() -> str:
    return os.path.join(_eval_job_dir(), "ner_trials")


def _eval_runs_dir() -> str:
    return os.path.join(_eval_job_dir(), "runs")


def _log_dir(stage: str) -> str:
    return str(LOG_NER_DIR / stage / get_job_id())


def _recommended_num_workers() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        try:
            cpus = int(slurm_cpus)
        except ValueError:
            cpus = 4
    else:
        detected = os.cpu_count()
        cpus = detected if detected is not None else 4
    return max(1, cpus - 1)


def _write_optuna_best_config(
    best_params: dict[str, float | int],
    model_name: str,
    max_epochs: int,
    val_window: int,
    val_stride: int,
) -> None:
    payload = {
        "batch_size": int(best_params["batch_size"]),
        "learning_rate": float(best_params["lr"]),
        "train_subsample_window": int(best_params["train_subsample_window"]),
        "val_window": int(val_window),
        "val_stride": int(val_stride),
        "weight_decay": float(best_params["weight_decay"]),
        "warmup_steps_pct": float(best_params["warmup_steps_pct"]),
        "max_epochs": int(max_epochs),
        "model_name": model_name,
    }
    path = str(OPTUNA_BEST_CONFIG_PATH)
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        _ = yaml.safe_dump(payload, f, sort_keys=False)
    print(f"[optuna] wrote best hyperparameters to {path}")


class NERTrainer:
    """
    Orchestrates hyperparameter optimization and training of NERTagger.

    Uses Optuna for hyperparameter search and PyTorch Lightning for training.
    """

    def __init__(
        self,
        data_path: str,
        model_name: str,
        label_list: list[str],
        num_trials: int,
        max_epochs: int,
        split_version: str = "default",
        train_docs: int | None = None,
        seed: int = 42,
        sampling_mode: str = "boundary_mix",
        decoder_mode: str = "independent",
        boundary_head: bool = False,
        boundary_loss_weight: float = 0.0,
        token_loss_mode: str = "focal",
        token_loss_weight: float = 1.0,
        crf_loss_weight: float = 0.0,
        label_smoothing: float = 0.0,
        preserve_case: bool = False,
        val_window: int = 510,
        val_stride: int = 256,
    ):
        """
        Initialize the NER trainer.

        Args:
            data_path: Path to the data file (.parquet or .csv)
            model_name: HuggingFace model name
            label_list: List of label names
            num_trials: Number of Optuna trials
            max_epochs: Maximum training epochs per trial
            split_version: Named split manifest to use
            train_docs: Optional number of training documents to include
            seed: Seed for split creation and training
            sampling_mode: Training sampler mode
            decoder_mode: Sequence decoder mode
            boundary_head: Whether to enable auxiliary boundary supervision
            boundary_loss_weight: Weight for boundary loss
            token_loss_mode: Token loss family
            token_loss_weight: Weight for token loss
            crf_loss_weight: Weight for CRF loss
            label_smoothing: CE label smoothing factor
            preserve_case: Whether to preserve original casing at tokenizer input
            val_window: Validation window size for evaluation
            val_stride: Validation stride size for evaluation
        """
        self.data_path = data_path
        self.model_name = model_name
        self.num_trials = num_trials
        self.max_epochs = max_epochs
        self.label_list = label_list
        self.split_version = split_version
        self.train_docs = train_docs
        self.seed = seed
        self.sampling_mode = sampling_mode
        self.decoder_mode = decoder_mode
        self.boundary_head = boundary_head
        self.boundary_loss_weight = boundary_loss_weight
        self.token_loss_mode = token_loss_mode
        self.token_loss_weight = token_loss_weight
        self.crf_loss_weight = crf_loss_weight
        self.label_smoothing = label_smoothing
        self.preserve_case = preserve_case
        self.val_window = val_window
        self.val_stride = val_stride
        self.split_path = _split_path_for_version(split_version)

        # Device selection
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.train_data: list[str] = []
        self.val_data: list[str] = []
        self.test_data: list[str] = []
        self.metrics_output_dir = _eval_trials_dir()

    def _read_data_frame(self) -> pd.DataFrame:
        path = self.data_path
        suffix = os.path.splitext(path)[1].lower()
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".csv":
            return pd.read_csv(path)
        raise ValueError(
            f"Unsupported data file extension for {path!r}. Expected .parquet or .csv."
        )

    def _validate_and_prepare_data_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = {
            "agreement_uuid",
            "tagged_text",
            "date_announcement",
            "page_uuid",
            "tagged_section",
            "tagged_article",
            "article_upsample",
        }
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(
                f"Data must contain columns: {sorted(required_cols)}. Missing: {sorted(missing)}"
            )

        def _has_tags(text: str) -> int:
            return 1 if "<section>" in text or "<article>" in text else 0

        df["tagged"] = df["tagged_text"].apply(_has_tags)
        df["agreement_uuid"] = df["agreement_uuid"].astype(str)
        df["page_uuid"] = df["page_uuid"].astype(str)
        announcement_dates = pd.to_datetime(df["date_announcement"], errors="raise")
        if announcement_dates.isna().any():
            raise ValueError("Found missing or invalid date_announcement values.")
        df["announcement_year"] = announcement_dates.dt.year.astype(int)
        df["announcement_window"] = (
            (df["announcement_year"] // YEAR_WINDOW) * YEAR_WINDOW
        ).astype(int)
        df["tagged_section"] = df["tagged_section"].astype(int)
        df["tagged_article"] = df["tagged_article"].astype(int)
        df["article_upsample"] = df["article_upsample"].astype(bool)

        print(f"Loaded data shape: {df.shape}")
        print(df.head(2))
        print(f"Tagged value counts:\n{df['tagged'].value_counts()}")
        print(f"Year value counts:\n{df['announcement_year'].value_counts()}")
        print(f"Year window value counts:\n{df['announcement_window'].value_counts()}")

        # For now, remove untagged pages
        # df = df[df["tagged"] == 1]
        return df

    def _load_data(self) -> None:
        """
        Load data and apply agreement-level train/val/test splits.
        """
        df = self._read_data_frame()
        df = self._validate_and_prepare_data_frame(df)

        split = self._load_or_build_split(df, self.split_path, self.seed)
        train_ids_list = [str(x) for x in cast(list[str], split["train"])]
        val_ids_list = [str(x) for x in cast(list[str], split["val"])]
        test_ids_list = [str(x) for x in cast(list[str], split["test"])]
        train_ids = set(train_ids_list)
        val_ids = set(val_ids_list)
        test_ids = set(test_ids_list)

        train_ids_sorted = _hash_sorted_ids(sorted(set(train_ids_list)), self.seed)
        if self.train_docs and self.train_docs > 0:
            if self.train_docs > len(train_ids_sorted):
                raise ValueError(
                    f"train_docs={self.train_docs} exceeds available train ids ({len(train_ids_sorted)})."
                )
            train_ids_sorted = train_ids_sorted[: self.train_docs]
        train_df = cast(
            pd.DataFrame, df[df["agreement_uuid"].isin(list(train_ids_sorted))].copy()
        )
        val_df = cast(pd.DataFrame, df[df["agreement_uuid"].isin(list(val_ids))].copy())
        test_df = cast(
            pd.DataFrame, df[df["agreement_uuid"].isin(list(test_ids))].copy()
        )

        if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
            raise ValueError("Split manifest has overlapping agreement_uuid values.")
        train_upsample_any = bool(train_df["article_upsample"].any())
        if not train_upsample_any:
            print("[split] warning: no article_upsample rows in train split.")
        val_upsample_any = bool(val_df["article_upsample"].any())
        test_upsample_any = bool(test_df["article_upsample"].any())
        if val_upsample_any or test_upsample_any:
            raise ValueError("Split manifest contains article_upsample rows in val/test.")

        print(
            "[split] agreements "
            + f"train={train_df['agreement_uuid'].nunique()}, "
            + f"val={val_df['agreement_uuid'].nunique()}, "
            + f"test={test_df['agreement_uuid'].nunique()}"
        )
        print(
            "[split] upsample rows "
            + f"train={int(train_df['article_upsample'].sum())}, "
            + f"val={int(val_df['article_upsample'].sum())}, "
            + f"test={int(test_df['article_upsample'].sum())}"
        )
        print(
            f"Splits -> train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}"
        )

        self.train_data = train_df["tagged_text"].tolist()
        self.val_data = val_df["tagged_text"].tolist()
        self.test_data = test_df["tagged_text"].tolist()

    def _load_or_build_split(
        self, df: pd.DataFrame, split_path: str, seed: int
    ) -> dict[str, object]:
        if os.path.exists(split_path):
            with open(split_path, "r", encoding="utf-8") as f:
                split = cast(dict[str, object], json.load(f))
            for key in ("train", "val", "test"):
                if key not in split:
                    raise ValueError("Split manifest missing required keys: train/val/test.")
            df_ids = set(df["agreement_uuid"].astype(str).tolist())
            split_ids = set(cast(list[str], split["train"])) | set(
                cast(list[str], split["val"])
            ) | set(cast(list[str], split["test"]))
            missing = split_ids - df_ids
            if missing:
                raise ValueError("Split manifest contains unknown agreement_uuid values.")
            return split

        val_split = 0.06
        test_split = 0.06
        total_split = val_split + test_split
        if total_split >= 1.0:
            raise ValueError("val_split + test_split must be < 1.0")

        base_df = cast(pd.DataFrame, df[~df["article_upsample"]].copy())
        if len(base_df) == 0:
            raise ValueError("No base rows (article_upsample=False) available for val/test.")
        date_counts = cast(
            pd.Series, df.groupby("agreement_uuid")["date_announcement"].nunique()
        )
        bad_dates = cast(pd.Series, date_counts[date_counts != 1])
        if not bad_dates.empty:
            raise ValueError(
                "Each agreement_uuid must map to exactly one date_announcement."
            )

        all_agreement_df = cast(
            pd.DataFrame,
            (
                df.groupby("agreement_uuid", as_index=False)
                .agg(
                    announcement_window=("announcement_window", "first"),
                    page_count=("page_uuid", "nunique"),
                    tagged_rate=("tagged", "mean"),
                    base_page_count=("article_upsample", _count_non_upsample_rows),
                )
                .copy()
            ),
        )
        all_agreement_df["upsample_only"] = cast(
            pd.Series, all_agreement_df["base_page_count"]
        ) == 0
        all_agreement_df["length_bucket"] = _bucket_agreement_lengths(
            cast(pd.Series, all_agreement_df["page_count"])
        )
        all_agreement_df["tag_density_bucket"] = _bucket_agreement_tag_density(
            cast(pd.Series, all_agreement_df["tagged_rate"])
        )
        base_agreement_df = cast(
            pd.DataFrame, all_agreement_df[~all_agreement_df["upsample_only"]].copy()
        )
        upsample_only_df = cast(
            pd.DataFrame, all_agreement_df[all_agreement_df["upsample_only"]].copy()
        )

        strat_levels = [
            ["announcement_window", "tag_density_bucket", "length_bucket"],
            ["announcement_window", "tag_density_bucket"],
            ["announcement_window"],
        ]

        target_val_count, target_test_count = _target_split_counts(
            total_items=len(all_agreement_df),
            val_split=val_split,
            test_split=test_split,
        )
        if target_val_count + target_test_count >= len(base_agreement_df):
            raise ValueError(
                "Not enough non-upsample agreements available to build val/test."
            )
        train_plus_test_df, val_agreement_df = _split_frame_with_stratification(
            base_agreement_df,
            test_size=target_val_count,
            strat_levels=strat_levels,
            seed=seed,
        )
        train_agreement_df, test_agreement_df = _split_frame_with_stratification(
            train_plus_test_df.reset_index(drop=True),
            test_size=target_test_count,
            strat_levels=strat_levels,
            seed=seed,
        )

        train_agreement_ids = set(
            train_agreement_df["agreement_uuid"].astype(str).tolist()
        ) | set(upsample_only_df["agreement_uuid"].astype(str).tolist())
        val_agreement_ids = set(val_agreement_df["agreement_uuid"].astype(str).tolist())
        test_agreement_ids = set(
            test_agreement_df["agreement_uuid"].astype(str).tolist()
        )

        split = {
            "train": sorted(train_agreement_ids),
            "val": sorted(val_agreement_ids),
            "test": sorted(test_agreement_ids),
            "meta": {
                "val_split": val_split,
                "test_split": test_split,
                "seed": seed,
                "stratify_cols": strat_levels[0],
                "stratify_fallback_levels": strat_levels[1:],
                "agreement_uuid_col": "agreement_uuid",
                "article_upsample_col": "article_upsample",
                "split_unit": "agreement_uuid",
                "upsample_holdout_fill": False,
                "upsample_only_train_only": True,
                "target_val_count": target_val_count,
                "target_test_count": target_test_count,
                "year_window": YEAR_WINDOW,
            },
        }
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=2, sort_keys=True)
        print(f"[split] wrote NER split manifest to {split_path}")
        return cast(dict[str, object], split)

    def _get_callbacks(
        self, trial: Trial | None = None, ckpt: str | None = None
    ) -> tuple[
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
        TQDMProgressBar,
        list[PyTorchLightningPruningCallback],
    ]:
        """
        Instantiate Lightning callbacks.

        Args:
            trial: Optuna trial for pruning callback

        Returns:
            Tuple of callbacks
        """
        # Single checkpoint callback for best val_loss
        if ckpt:
            dirpath = os.path.dirname(ckpt)
            filename = os.path.splitext(os.path.basename(ckpt))[0]
        else:
            dirpath = None
            filename = "best-{epoch:02d}-{val_ent_f1:.4f}"
        checkpoint_callback = ModelCheckpoint(
            monitor="val_ent_f1",
            mode="max",
            save_top_k=1,
            filename=filename,
            dirpath=dirpath,
        )
        early_stop_callback = EarlyStopping(
            monitor="val_ent_f1", patience=3, mode="max"
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        pruning_callback = (
            [PyTorchLightningPruningCallback(trial, monitor="val_ent_f1")]
            if trial is not None
            else []
        )
        progress_bar_callback = TQDMProgressBar(refresh_rate=200)

        return (
            checkpoint_callback,
            early_stop_callback,
            lr_monitor,
            progress_bar_callback,
            pruning_callback,
        )

    def load_data(self) -> None:
        """Public wrapper for data loading."""
        self._load_data()

    def build(
        self,
        params: dict[str, float | int],
        metrics_output_name: str | None = None,
    ) -> tuple["NERDataModule", "NERTagger"]:
        """Public wrapper for model/datamodule creation."""
        return self._build(params, metrics_output_name=metrics_output_name)

    def get_callbacks(
        self, trial: Trial | None = None, ckpt: str | None = None
    ) -> tuple[
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
        TQDMProgressBar,
        list[PyTorchLightningPruningCallback],
    ]:
        """Public wrapper for callback creation."""
        return self._get_callbacks(trial=trial, ckpt=ckpt)

    def _build(
        self,
        params: dict[str, float | int],
        metrics_output_name: str | None = None,
    ) -> tuple["NERDataModule", "NERTagger"]:
        """
        Instantiate DataModule and Model from hyperparameters.

        Args:
            params: Dictionary of hyperparameters

        Returns:
            Tuple of (data_module, model)
        """
        data_module = NERDataModule(
            train_data=self.train_data,
            val_data=self.val_data,
            test_data=self.test_data,
            tokenizer_name=self.model_name,
            label_list=self.label_list,
            batch_size=int(params["batch_size"]),
            train_subsample_window=int(params["train_subsample_window"]),
            num_workers=_recommended_num_workers(),
            sampling_mode=self.sampling_mode,
            seed=self.seed,
            val_window=int(params.get("val_window", self.val_window)),
            val_stride=int(params.get("val_stride", self.val_stride)),
            preserve_case=self.preserve_case,
        )
        model = NERTagger(
            model_name=self.model_name,
            num_labels=len(self.label_list),
            id2label={idx: label for idx, label in enumerate(self.label_list)},
            learning_rate=params["lr"],
            weight_decay=params["weight_decay"],
            warmup_steps_pct=params["warmup_steps_pct"],
            decoder_mode=self.decoder_mode,
            token_loss_mode=self.token_loss_mode,
            token_loss_weight=self.token_loss_weight,
            crf_loss_weight=self.crf_loss_weight,
            boundary_head=self.boundary_head,
            boundary_loss_weight=self.boundary_loss_weight,
            label_smoothing=self.label_smoothing,
            preserve_case=self.preserve_case,
            metrics_output_dir=self.metrics_output_dir,
            metrics_output_name=metrics_output_name or "ner_test_metrics.yaml",
        )
        return data_module, model

    def _objective(self, trial: Trial) -> float:
        """
        Optuna objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Validation loss
        """
        params = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "train_subsample_window": trial.suggest_categorical(
                "train_subsample_window", [128, 256, 512]
            ),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
            "warmup_steps_pct": trial.suggest_float("warmup_steps_pct", 0.0, 0.3),
        }

        data_module, model = self._build(params)
        (
            checkpoint_callback,
            early_stop_callback,
            lr_monitor,
            progress_bar_callback,
            pruning_callback,
        ) = self._get_callbacks(trial)

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.device,
            precision=self._trainer_precision(),
            devices=1,
            logger=TensorBoardLogger(_log_dir("optuna"), name="", version=""),
            callbacks=[
                checkpoint_callback,
                early_stop_callback,
                progress_bar_callback,
                *pruning_callback,
            ],
            log_every_n_steps=10,
            deterministic=True,
        )
        trainer.fit(model, datamodule=data_module)
        val_ent_f1 = float(trainer.callback_metrics["val_ent_f1"].item())

        trial_metrics = {
            "trial": trial.number,
            "val_ent_f1": val_ent_f1,
            "params": params,
        }
        os.makedirs(self.metrics_output_dir, exist_ok=True)
        trial_path = os.path.join(
            self.metrics_output_dir, f"trial_{trial.number:03d}.yaml"
        )
        with open(trial_path, "w", encoding="utf-8") as f:
            _ = yaml.safe_dump(trial_metrics, f, sort_keys=False)

        # Clean up to avoid memory leaks
        del (
            model,
            data_module,
            trainer,
            checkpoint_callback,
            early_stop_callback,
            lr_monitor,
        )
        if pruning_callback:
            del pruning_callback

        return val_ent_f1

    def objective(self, trial: Trial) -> float:
        """Public wrapper for Optuna objective."""
        return self._objective(trial)

    def run(self) -> None:
        """Execute hyperparameter optimization and final training."""
        _ = seed_everything(self.seed, workers=True, verbose=False)
        self._load_data()

        study = create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.num_trials, gc_after_trial=True)

        print("Finished hyperparameter optimization 👉")
        print(f"  Best val_ent_f1: {study.best_value:.4f}")
        print("  Best hyperparameters:")
        best_params = cast(dict[str, float | int], study.best_trial.params)
        for key, value in best_params.items():
            print(f"    • {key}: {value}")
        _write_optuna_best_config(
            best_params,
            self.model_name,
            self.max_epochs,
            self.val_window,
            self.val_stride,
        )

        # Retrain best model to get its checkpoint on disk
        data_module, model = self._build(
            best_params, metrics_output_name="ner_test_metrics_final.yaml"
        )
        (
            checkpoint_callback,
            early_stop_callback,
            _lr_monitor,
            progress_bar_callback,
            _,
        ) = self._get_callbacks(ckpt=NER_CKPT)

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.device,
            devices=1,
            precision=self._trainer_precision(),
            logger=TensorBoardLogger(_log_dir("final"), name="", version=""),
            callbacks=[
                checkpoint_callback,
                early_stop_callback,
                progress_bar_callback,
            ],
            log_every_n_steps=10,
        )
        trainer.fit(model, datamodule=data_module)
        _ = trainer.test(model, datamodule=data_module, ckpt_path=NER_CKPT)

    def _trainer_precision(self) -> PrecisionInput:
        if self.device in ("cuda", "mps"):
            return "bf16-mixed"
        return "32-true"

    def trainer_precision(self) -> PrecisionInput:
        """Public wrapper for trainer precision."""
        return self._trainer_precision()


class NERInference:
    """
    Token-window NER inference that returns:
      - tagged (str): XML-tagged text
      - low_count (int): number of low-confidence tokens
      - spans (list[dict]): contiguous low-confidence token spans
      - tokens (list[dict]): token-level records below threshold
    """

    def __init__(
        self,
        ckpt_path: str,
        label_list: list[str] | None,
        review_threshold: float = 0.5,
        window_batch_size: int = 32,
        window: int = 510,
        stride: int = 256,
    ) -> None:

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model: NERTagger = NERTagger.load_from_checkpoint(
            ckpt_path, map_location=self.device
        )
        _ = self.model.to(self.device)
        _ = self.model.eval()

        # tokenizer consistent with training (special tokens added, no resize needed at inference)
        hparams = cast(object, self.model.hparams)
        model_name = cast(str, getattr(hparams, "model_name"))
        self.tokenizer: PreTrainedTokenizerBase = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(model_name, use_fast=True),
        )
        added_tokens = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS}
        )
        if added_tokens > 0:
            _ = self.model.model.resize_token_embeddings(len(self.tokenizer))

        # Fallbacks for essential token IDs (safe-guards)
        pad_token_id = cast(int | None, self.tokenizer.pad_token_id)
        if pad_token_id is None:
            eos_token = cast(str | None, self.tokenizer.eos_token)
            unk_token = cast(str | None, self.tokenizer.unk_token)
            self.tokenizer.pad_token = eos_token or unk_token or "[PAD]"
        cls_token_id = cast(int | None, self.tokenizer.cls_token_id)
        if cls_token_id is None:
            bos_token = cast(str | None, self.tokenizer.bos_token)
            unk_token = cast(str | None, self.tokenizer.unk_token)
            self.tokenizer.cls_token = bos_token or unk_token or "[CLS]"
        sep_token_id = cast(int | None, self.tokenizer.sep_token_id)
        if sep_token_id is None:
            eos_token = cast(str | None, self.tokenizer.eos_token)
            unk_token = cast(str | None, self.tokenizer.unk_token)
            self.tokenizer.sep_token = eos_token or unk_token or "[SEP]"

        # >>> Use id2label/label2id from checkpoint to avoid order drift
        ckpt_id2label = cast(dict[int, str], getattr(hparams, "id2label"))
        self.id2label = {int(k): str(v) for k, v in ckpt_id2label.items()}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.label_list = [self.id2label[i] for i in range(len(self.id2label))]

        # Optional: validate if user passes label_list
        if label_list is not None:
            if list(label_list) != self.label_list:
                message = f"label_list provided to NERInference does not match the checkpoint label order.\nCheckpoint: {self.label_list}\nProvided:   {list(label_list)}"
                raise ValueError(message)

        self.C = len(self.label_list)
        self.review_threshold = review_threshold
        self.window_batch_size = window_batch_size
        self.window = window
        self.stride = stride
        self.preserve_case = bool(getattr(hparams, "preserve_case", False))

    # ---------------- Token aggregation over sliding windows (logit stitching) ----------------
    def _repair_bioes_ids(self, seq: list[int]) -> list[int]:
        out = seq[:]
        for t in range(len(out)):
            lab = self.id2label[out[t]]
            if lab == "O" or lab.startswith(("B-", "S-")):
                continue
            if t == 0:
                ent = lab.split("-", 1)[1]
                out[t] = self.label2id.get(f"B-{ent}", out[t])
                continue
            prev = self.id2label[out[t - 1]]
            if prev == "O" or prev.startswith(("E-", "S-")):
                ent = lab.split("-", 1)[1]
                out[t] = self.label2id.get(f"B-{ent}", out[t])
            elif prev.startswith(("B-", "I-")):
                prev_ent = prev.split("-", 1)[1]
                cur_ent = lab.split("-", 1)[1]
                if prev_ent != cur_ent:
                    out[t] = self.label2id.get(f"B-{cur_ent}", out[t])
        return out

    def _collapse_to_word_level(
        self,
        text: str,
        preds: list[int],
        confidences: list[float],
        offsets: list[tuple[int, int]],
        avg_logits: torch.Tensor,
        word_ids: list[int | None],
    ) -> tuple[list[int], list[float], list[tuple[int, int]], list[str], torch.Tensor]:
        first_token_indices: list[int] = []
        word_offsets: list[tuple[int, int]] = []
        word_index_by_id: dict[int, int] = {}

        for tok_idx, wid in enumerate(word_ids):
            if wid is None:
                continue
            start, end = offsets[tok_idx]
            if end <= start:
                continue
            word_idx = word_index_by_id.get(wid)
            if word_idx is None:
                word_index_by_id[wid] = len(word_offsets)
                first_token_indices.append(tok_idx)
                word_offsets.append((start, end))
                continue
            cur_start, cur_end = word_offsets[word_idx]
            word_offsets[word_idx] = (min(cur_start, start), max(cur_end, end))

        if not first_token_indices:
            empty_logits = torch.zeros((0, self.C), dtype=torch.float32)
            return [], [], [], [], empty_logits

        collapsed_offsets: list[tuple[int, int]] = []
        kept_first_token_indices: list[int] = []
        for first_token_idx, (start, end) in zip(first_token_indices, word_offsets):
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if end <= start:
                continue
            collapsed_offsets.append((start, end))
            kept_first_token_indices.append(first_token_idx)

        collapsed_preds = [preds[idx] for idx in kept_first_token_indices]
        collapsed_confidences = [confidences[idx] for idx in kept_first_token_indices]
        collapsed_tokens = [text[start:end] for start, end in collapsed_offsets]
        index_tensor = torch.tensor(kept_first_token_indices, dtype=torch.long)
        collapsed_logits = avg_logits.index_select(0, index_tensor)

        return (
            collapsed_preds,
            collapsed_confidences,
            collapsed_offsets,
            collapsed_tokens,
            collapsed_logits,
        )

    def _predict_tokens(
        self, text: str
    ) -> tuple[list[int], list[float], list[tuple[int, int]], list[str], torch.Tensor]:
        """
        Stitch per-token predictions across overlapping windows by averaging LOGITS
        (to match validation), then compute confidences from softmax(avg_logits) and
        apply a light BIOES repair on the predicted tag sequence.

        Returns:
            preds        : List[int]            # predicted label ids per token
            confidences  : List[float]          # max prob per token (from softmax(avg_logits))
            offsets      : List[tuple[int,int]] # original char offsets per token
            toks         : List[str]            # token strings
            avg_logits   : torch.Tensor         # [T, C] averaged logits on CPU
        """
        norm = text if self.preserve_case else ascii_lower(text)
        encoding = cast(
            object,
            self.tokenizer(
                norm,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=False,
                add_special_tokens=False,
            ),
        )
        enc_full = cast(dict[str, torch.Tensor], encoding)
        input_ids = enc_full["input_ids"][0]  # [T]
        offsets_tensor = enc_full["offset_mapping"][0]
        offsets_raw = cast(list[list[int]], offsets_tensor.tolist())
        offsets = [(int(s), int(e)) for s, e in offsets_raw]
        word_ids_fn = cast(Callable[[int], list[int | None]], getattr(encoding, "word_ids"))
        word_ids = word_ids_fn(0)
        convert_ids = cast(
            Callable[[list[int]], list[str]],
            getattr(self.tokenizer, "convert_ids_to_tokens"),
        )
        toks = convert_ids(cast(list[int], input_ids.tolist()))
        T = int(input_ids.numel())

        if T == 0:
            empty_logits = torch.zeros((0, self.C), dtype=torch.float32)
            return [], [], [], [], empty_logits

        # --- Accumulate LOGITS (not probabilities) across overlapping windows ---
        sum_logits = torch.zeros((T, self.C), device="cpu")
        counts = torch.zeros(T, device="cpu")

        # Build token windows; add CLS/SEP per chunk
        windows: list[WindowItem] = []
        bounds: list[tuple[int, int]] = []  # (start_tok, end_tok)
        i = 0
        while i < T:
            j = min(i + self.window, T)
            chunk_ids = cast(list[int], input_ids[i:j].tolist())
            cls_id = cast(int, self.tokenizer.cls_token_id)
            sep_id = cast(int, self.tokenizer.sep_token_id)
            windows.append(
                {
                    "input_ids": [cls_id] + chunk_ids + [sep_id],
                    "attention_mask": [1] * (len(chunk_ids) + 2),
                }
            )
            bounds.append((i, j))
            if j == T:
                break
            i += self.stride

        # Batched inference -> return LOGITS with CLS/SEP removed
        def _infer_logits(batch_items: list[WindowItem]) -> list[torch.Tensor]:
            max_len = max(len(x["input_ids"]) for x in batch_items)
            pad_id = cast(int, self.tokenizer.pad_token_id)

            ids: list[list[int]] = [
                x["input_ids"] + [pad_id] * (max_len - len(x["input_ids"]))
                for x in batch_items
            ]
            mask: list[list[int]] = [
                x["attention_mask"] + [0] * (max_len - len(x["attention_mask"]))
                for x in batch_items
            ]

            ids_t = torch.tensor(ids, device=self.device)
            mask_t = torch.tensor(mask, device=self.device)
            with torch.no_grad():
                logits = cast(
                    torch.Tensor,
                    self.model(input_ids=ids_t, attention_mask=mask_t).logits,
                )  # [B,L,C]

            outs: list[torch.Tensor] = []
            for lg, m in zip(logits, mask_t):
                true_len = int(m.sum().item())  # includes CLS/SEP; pads are 0
                if true_len <= 2:
                    outs.append(lg[0:0])  # empty
                else:
                    outs.append(lg[1 : true_len - 1])  # strip CLS/SEP
            return outs

        all_logits: list[torch.Tensor] = []
        for k in range(0, len(windows), self.window_batch_size):
            all_logits.extend(_infer_logits(windows[k : k + self.window_batch_size]))

        # Accumulate per-token LOGITS over window overlaps
        for (s, e), lg_tok in zip(bounds, all_logits):
            span = min(e - s, lg_tok.size(0))
            if span <= 0:
                continue
            sum_logits[s : s + span] += lg_tok[:span].to("cpu")
            counts[s : s + span] += 1

        # Average logits like validation; then derive probs/confidence/preds
        avg_logits = sum_logits / counts.unsqueeze(-1).clamp(min=1.0)
        probs = torch.softmax(avg_logits, dim=-1)
        confidences = cast(list[float], probs.max(dim=-1)[0].tolist())
        preds = cast(
            list[int],
            self.model.decode_constrained_doc(avg_logits.to(self.device))
            .detach()
            .cpu()
            .tolist(),
        )
        preds, confidences, offsets, toks, avg_logits = self._collapse_to_word_level(
            text,
            preds,
            confidences,
            offsets,
            avg_logits,
            word_ids,
        )
        preds = self._repair_bioes_ids(preds)
        return preds, confidences, offsets, toks, avg_logits

    # ---------------- Pretty print from token labels via offsets ----------------
    def _pretty_print_from_tokens(
        self, text: str, preds: list[int], offsets: list[tuple[int, int]]
    ) -> str:
        def ent(lid: int) -> str:
            tag = self.id2label.get(lid, "O")
            return "O" if tag == "O" else tag.split("-", 1)[1].lower()

        res = []
        cur_ent = "O"
        pos = 0  # last emitted char pos in source text

        for lid, (s, e) in zip(preds, offsets):
            if e == 0 or s >= e:
                continue
            tok_ent = ent(lid)
            if tok_ent != cur_ent:
                if cur_ent != "O":
                    res.append(f"</{cur_ent}>")
                if pos < s:
                    res.append(text[pos:s])
                    pos = s
                if tok_ent != "O":
                    res.append(f"<{tok_ent}>")
                cur_ent = tok_ent
            elif pos < s:
                res.append(text[pos:s])
                pos = s
            res.append(text[s:e])
            pos = e

        if pos < len(text):
            res.append(text[pos:])
        if cur_ent != "O":
            res.append(f"</{cur_ent}>")
        return "".join(res)

    # ---------------- Low-confidence spans over tokens ----------------
    def _token_spans(
        self, preds: list[int], confs: list[float], offsets: list[tuple[int, int]]
    ) -> tuple[int, list[dict[str, object]]]:
        low_idxs = [i for i, c in enumerate(confs) if c < self.review_threshold]
        low_count = len(low_idxs)
        spans: list[dict[str, object]] = []
        if not low_idxs:
            return low_count, spans

        def ent_name(lid: int) -> str:
            lab = self.id2label[lid]
            return lab.split("-", 1)[1].lower() if "-" in lab else lab.lower()

        start = low_idxs[0]
        cur_lab = preds[start]
        acc = [confs[start]]
        prev = start

        for i in low_idxs[1:]:
            if i == prev + 1 and preds[i] == cur_lab:
                acc.append(confs[i])
                prev = i
            else:
                spans.append(
                    {
                        "entity": ent_name(cur_lab),
                        "avg_confidence": sum(acc) / len(acc),
                        # character offsets relative to ORIGINAL untagged text
                        "start_char": (
                            int(offsets[start][0])
                            if offsets[start][1] > 0
                            else int(offsets[start][0])
                        ),
                        "end_char": (
                            int(offsets[prev][1])
                            if offsets[prev][1] > 0
                            else int(offsets[prev][0])
                        ),
                    }
                )
                start = i
                cur_lab = preds[i]
                acc = [confs[i]]
                prev = i

        spans.append(
            {
                "entity": ent_name(cur_lab),
                "avg_confidence": sum(acc) / len(acc),
                "start_char": (
                    int(offsets[start][0])
                    if offsets[start][1] > 0
                    else int(offsets[start][0])
                ),
                "end_char": (
                    int(offsets[prev][1])
                    if offsets[prev][1] > 0
                    else int(offsets[prev][0])
                ),
            }
        )
        return low_count, spans

    # ---------------- Public API ----------------
    # 2) label() gains a flag and optionally returns token_probs
    def label(
        self,
        texts: list[str],
        verbose: bool = False,
        return_token_probs: bool = False,  # <— new flag
    ) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for idx, text in enumerate(texts):
            preds, confs, offsets, _, avg_logits = self._predict_tokens(text)
            tagged = self._pretty_print_from_tokens(text, preds, offsets)
            low_count, spans = self._token_spans(preds, confs, offsets)

            tokens_below: list[dict[str, object]] = []
            for i, ((s, e), conf, lid) in enumerate(zip(offsets, confs, preds)):
                if e == 0 or s >= e:
                    continue
                if conf < self.review_threshold:
                    lab = self.id2label[lid]
                    tokens_below.append(
                        {
                            "i": i,
                            "token": text[s:e],
                            "start": s,
                            "end": e,
                            "entity": (
                                lab.split("-", 1)[1].lower()
                                if "-" in lab
                                else lab.lower()
                            ),
                            "confidence": conf,
                        }
                    )

            out: dict[str, object] = {
                "tagged": tagged,
                "low_count": low_count,
                "spans": spans,
                "tokens": tokens_below,
            }

            if return_token_probs:
                # Build full per-token probability vectors (from stitched avg_logits)
                probs_full = cast(
                    list[list[float]], torch.softmax(avg_logits, dim=-1).tolist()
                )
                token_probs = []
                for i, ((s, e), lid, pv) in enumerate(zip(offsets, preds, probs_full)):
                    if e == 0 or s >= e:
                        continue
                    token_probs.append(
                        {
                            "i": i,
                            "token": text[s:e],
                            "start": s,
                            "end": e,
                            "pred_class": self.id2label[lid],
                            "confidence": max(pv),
                            "probs": {
                                self.id2label[c]: float(pv[c]) for c in range(self.C)
                            },
                        }
                    )
                out["token_probs"] = token_probs

            if verbose:
                print(f"\n=== Text #{idx} ===")
                print(tagged)
                print(
                    f"\n[low-confidence tokens < {self.review_threshold}]: {low_count}"
                )
                if spans:
                    print("\nUncertain spans (token-level):")
                    for sp in spans:
                        print(f" - {sp}")
                else:
                    print("No spans below threshold.")

            results.append(out)
        return results


def _flatten_metrics(
    metrics: dict[str, object], prefix: str = ""
) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_metrics(value, f"{name}_"))
        elif isinstance(value, (int, float)):
            flat[name] = float(value)
    return flat


def _metrics_with_suffix(
    metrics: dict[str, object], suffix: str
) -> dict[str, float]:
    flat = _flatten_metrics(metrics)
    return {f"{key}_{suffix}": value for key, value in flat.items()}


def _append_experiment_row(csv_path: str, row: dict[str, object]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = list(row.keys())
    with open(csv_path, "a+", encoding="utf-8", newline="") as f:
        _ = fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            _ = f.seek(0)
            content = f.read()
            file_exists = bool(content)
            if file_exists:
                _ = f.seek(0)
                reader = csv.DictReader(f)
                header = reader.fieldnames or []
                existing_rows = list(reader)
                if header:
                    extra_fields = [key for key in row.keys() if key not in header]
                    missing_fields = [key for key in header if key not in row]
                    if extra_fields or missing_fields:
                        fieldnames = list(header) + extra_fields
                        rewritten_rows: list[dict[str, object]] = []
                        for existing_row in existing_rows:
                            normalized_row: dict[str, object] = {
                                name: existing_row.get(name, "") for name in fieldnames
                            }
                            rewritten_rows.append(normalized_row)
                        _ = f.seek(0)
                        _ = f.truncate()
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rewritten_rows)
                    else:
                        fieldnames = header
            _ = f.seek(0, os.SEEK_END)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        finally:
            _ = fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_grid_row(row_id: int, grid_path: str | None = None) -> dict[str, str]:
    """
    Load a single grid row by row_id.
    """
    path = grid_path or str(CONFIG_NER_DIR / "grid.csv")
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["row_id"]) == row_id:
                return cast(dict[str, str], row)
    raise ValueError(f"row_id {row_id} not found in {path}")


def parse_boolish(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Unable to parse boolean value from {value!r}.")


def run_hparam_tuning(
    *,
    data_path: str,
    model_name: str,
    num_trials: int,
    max_epochs: int,
    split_version: str,
    seed: int,
    val_window: int = 510,
    val_stride: int = 256,
    git_commit: str | None = None,
) -> dict[str, float | int]:
    """
    Run Optuna tuning on train/val, write optuna_best_config.yaml, and retrain final model.
    """
    _ = resolve_git_commit(git_commit)
    ner_trainer = NERTrainer(
        data_path=data_path,
        model_name=model_name,
        label_list=NER_LABELS,
        num_trials=num_trials,
        max_epochs=max_epochs,
        split_version=split_version,
        seed=seed,
        val_window=val_window,
        val_stride=val_stride,
    )
    ner_trainer.load_data()

    study = create_study(direction="maximize")
    study.optimize(ner_trainer.objective, n_trials=num_trials, gc_after_trial=True)

    print("Finished hyperparameter optimization 👉")
    print(f"  Best val_ent_f1: {study.best_value:.4f}")
    print("  Best hyperparameters:")
    best_params = cast(dict[str, float | int], study.best_trial.params)
    for key, value in best_params.items():
        print(f"    • {key}: {value}")
    _write_optuna_best_config(
        best_params, model_name, max_epochs, val_window, val_stride
    )

    # Retrain best model to get its checkpoint on disk
    print("\nRetraining final model with best hyperparameters...")
    data_module, model = ner_trainer.build(
        best_params, metrics_output_name="ner_test_metrics_final.yaml"
    )
    (
        checkpoint_callback,
        early_stop_callback,
        _lr_monitor,
        progress_bar_callback,
        _,
    ) = ner_trainer.get_callbacks(ckpt=NER_CKPT)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=ner_trainer.device,
        devices=1,
        precision=ner_trainer.trainer_precision(),
        logger=TensorBoardLogger(_log_dir("final"), name="", version=""),
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            progress_bar_callback,
        ],
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=data_module)
    _ = trainer.test(model, datamodule=data_module, ckpt_path=NER_CKPT)

    return best_params


def run_training_and_eval(
    config: "NERExperimentConfig",
    *,
    eval_split: EvalSplit = "val",
    run_test: bool = True,
    run_dir: str | None = None,
    ckpt_path: str | None = None,
    metrics_output_name: str = "metrics.yaml",
    log_experiment: bool = True,
    log_stage: str = "experiments",
) -> dict[str, object] | None:
    """
    Train a model and optionally evaluate on the requested split.
    """
    slurm_job_id = get_job_id()

    _ = seed_everything(config.seed, workers=True, verbose=False)

    ner_trainer = NERTrainer(
        data_path=config.data_path,
        model_name=config.model_name,
        label_list=NER_LABELS,
        num_trials=0,
        max_epochs=config.max_epochs,
        split_version=config.split_version,
        train_docs=config.train_docs,
        seed=config.seed,
        sampling_mode=config.sampling_mode,
        decoder_mode=config.decoder_mode,
        boundary_head=config.boundary_head,
        boundary_loss_weight=config.boundary_loss_weight,
        token_loss_mode=config.token_loss_mode,
        token_loss_weight=config.token_loss_weight,
        crf_loss_weight=config.crf_loss_weight,
        label_smoothing=config.label_smoothing,
        preserve_case=config.preserve_case,
        val_window=config.val_window,
        val_stride=config.val_stride,
    )
    ner_trainer.load_data()

    run_dir = run_dir or os.path.join(_eval_runs_dir(), config.run_id)
    os.makedirs(run_dir, exist_ok=True)
    ner_trainer.metrics_output_dir = run_dir

    ckpt_path = ckpt_path or os.path.join(run_dir, "best.ckpt")
    if ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    eval_data = ner_trainer.val_data if eval_split == "val" else ner_trainer.test_data
    if run_test and not eval_data:
        raise RuntimeError(f"No {eval_split} data available for evaluation.")
    original_test_data = ner_trainer.test_data
    if run_test:
        ner_trainer.test_data = eval_data

    params = {
        "lr": config.learning_rate,
        "batch_size": config.batch_size,
        "train_subsample_window": config.train_subsample_window,
        "val_window": config.val_window,
        "val_stride": config.val_stride,
        "weight_decay": config.weight_decay,
        "warmup_steps_pct": config.warmup_steps_pct,
    }
    data_module, model = ner_trainer.build(
        params, metrics_output_name=metrics_output_name
    )
    (
        checkpoint_callback,
        early_stop_callback,
        _lr_monitor,
        progress_bar_callback,
        _,
    ) = ner_trainer.get_callbacks(ckpt=ckpt_path)

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=ner_trainer.device,
        precision=ner_trainer.trainer_precision(),
        devices=1,
        logger=TensorBoardLogger(
            _log_dir(log_stage),
            name="",
            version=config.run_id,
        ),
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            progress_bar_callback,
        ],
        log_every_n_steps=10,
        deterministic=True,
    )
    trainer.fit(model, datamodule=data_module)

    config_payload = config_to_dict(config)
    config_payload["eval_split"] = eval_split if run_test else None
    if log_stage == "final":
        config_payload["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        _ = yaml.safe_dump(config_payload, f, sort_keys=False)
    if log_stage == "final":
        run_id_path = os.path.join(run_dir, "run_id.txt")
        with open(run_id_path, "w", encoding="utf-8") as f:
            _ = f.write(f"{config.run_id}\n")

    metrics: dict[str, object] | None = None
    if run_test:
        setattr(model, "eval_log_prefix", eval_split)
        _ = trainer.test(model, datamodule=data_module, ckpt_path="best")
        metrics = cast(dict[str, object], getattr(model, "test_metrics", None))
        if not metrics:
            raise RuntimeError("Test metrics were not captured from the model.")

        metrics_path = os.path.join(run_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=False)

        if log_experiment and trainer.logger is not None:
            summary_metrics = _extract_headline_metrics(
                metrics, prefix=f"{eval_split}"
            )
            hparams = {
                "xp_name": config.xp_name,
                "train_docs": config.train_docs,
                "sampling_mode": config.sampling_mode,
                "decoder_mode": config.decoder_mode,
                "boundary_head": config.boundary_head,
                "boundary_loss_weight": config.boundary_loss_weight,
                "token_loss_mode": config.token_loss_mode,
                "token_loss_weight": config.token_loss_weight,
                "crf_loss_weight": config.crf_loss_weight,
                "label_smoothing": config.label_smoothing,
                "preserve_case": config.preserve_case,
                "split_version": config.split_version,
                "seed": config.seed,
                "model_name": config.model_name,
                "batch_size": config.batch_size,
                "train_subsample_window": config.train_subsample_window,
                "val_window": config.val_window,
                "val_stride": config.val_stride,
                "max_epochs": config.max_epochs,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "warmup_steps_pct": config.warmup_steps_pct,
                "git_commit": config.git_commit,
            }
            trainer.logger.log_hyperparams(hparams, summary_metrics)
            trainer.logger.log_metrics(
                summary_metrics, step=int(config.train_docs)
            )

        if log_experiment:
            variants = cast(dict[str, dict[str, object]], metrics.get("variants", {}))
            flat_metrics: dict[str, float] = {}
            for variant_key in ("raw",):
                if variant_key in variants:
                    flat_metrics.update(
                        _metrics_with_suffix(variants[variant_key], variant_key)
                    )

            row: dict[str, object] = {
                "run_id": config.run_id,
                "slurm_job_id": slurm_job_id,
                "git_commit": config.git_commit,
                "xp_name": config.xp_name,
                "split_version": config.split_version,
                "train_docs": config.train_docs,
                "sampling_mode": config.sampling_mode,
                "decoder_mode": config.decoder_mode,
                "boundary_head": config.boundary_head,
                "boundary_loss_weight": config.boundary_loss_weight,
                "token_loss_mode": config.token_loss_mode,
                "token_loss_weight": config.token_loss_weight,
                "crf_loss_weight": config.crf_loss_weight,
                "label_smoothing": config.label_smoothing,
                "preserve_case": config.preserve_case,
                "seed": config.seed,
                "model_name": config.model_name,
                "batch_size": config.batch_size,
                "train_subsample_window": config.train_subsample_window,
                "val_window": config.val_window,
                "val_stride": config.val_stride,
                "max_epochs": config.max_epochs,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "warmup_steps_pct": config.warmup_steps_pct,
                "run_dir": run_dir,
                **flat_metrics,
            }
            experiments_csv = str(EVAL_NER_DIR / "experiments_xp.csv")
            _append_experiment_row(experiments_csv, row)

    ner_trainer.test_data = original_test_data
    return metrics


def _extract_headline_metrics(
    metrics: dict[str, object],
    *,
    prefix: str,
) -> dict[str, float]:
    variants = cast(dict[str, dict[str, object]], metrics["variants"])
    headline: dict[str, float] = {}
    for variant_key in ("raw",):
        variant = variants[variant_key]
        entity_level = cast(dict[str, object], variant["entity_level"])
        micro = cast(dict[str, float], entity_level["micro"])
        per_type = cast(dict[str, dict[str, float]], entity_level["per_type"])
        article_metrics = per_type.get("ARTICLE", {"f1": 0.0, "recall": 0.0})
        headline[f"{prefix}/entity_strict_f1_{variant_key}"] = float(micro["f1"])
        headline[f"{prefix}/article_strict_f1_{variant_key}"] = float(
            article_metrics["f1"]
        )
        headline[f"{prefix}/article_strict_recall_{variant_key}"] = float(
            article_metrics["recall"]
        )
    return headline


def recover_experiment_row_from_run_dir(
    *,
    run_dir: str,
    experiments_csv: str | None = None,
) -> dict[str, object]:
    config_path = os.path.join(run_dir, "config.yaml")
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(config_path):
        raise RuntimeError(f"Missing config.yaml in run dir: {run_dir}")
    if not os.path.exists(metrics_path):
        raise RuntimeError(f"Missing metrics.json in run dir: {run_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_payload = cast(dict[str, object], yaml.safe_load(f))
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = cast(dict[str, object], json.load(f))

    variants = cast(dict[str, dict[str, object]], metrics.get("variants", {}))
    flat_metrics: dict[str, float] = {}
    for variant_key in ("raw",):
        if variant_key in variants:
            flat_metrics.update(_metrics_with_suffix(variants[variant_key], variant_key))

    row: dict[str, object] = {
        "run_id": config_payload["run_id"],
        "slurm_job_id": os.path.basename(os.path.dirname(run_dir)),
        "git_commit": config_payload.get("git_commit", "unknown"),
        "xp_name": config_payload["xp_name"],
        "split_version": config_payload["split_version"],
        "train_docs": config_payload["train_docs"],
        "sampling_mode": config_payload["sampling_mode"],
        "decoder_mode": config_payload["decoder_mode"],
        "boundary_head": config_payload["boundary_head"],
        "boundary_loss_weight": config_payload["boundary_loss_weight"],
        "token_loss_mode": config_payload["token_loss_mode"],
        "token_loss_weight": config_payload["token_loss_weight"],
        "crf_loss_weight": config_payload["crf_loss_weight"],
        "label_smoothing": config_payload["label_smoothing"],
        "preserve_case": config_payload.get("preserve_case", False),
        "seed": config_payload["seed"],
        "run_dir": run_dir,
        **flat_metrics,
    }
    target_csv = experiments_csv or str(EVAL_NER_DIR / "experiments_xp.csv")
    _append_experiment_row(target_csv, row)
    return row


def run_grid_row(
    *,
    row_id: int,
    split_version: str,
    seed: int,
    git_commit: str | None,
    grid_path: str | None = None,
    frozen_config_path: str | None = None,
    data_path: str | None = None,
    eval_split: EvalSplit = "val",
) -> dict[str, object] | None:
    """
    Run a single grid row experiment using frozen hyperparameters.
    """
    row = load_grid_row(row_id, grid_path=grid_path)
    frozen = load_frozen_experiment_config(path=frozen_config_path)
    row_seed = int(row.get("seed", str(seed)))
    row_learning_rate = float(row.get("learning_rate", str(frozen["learning_rate"])))
    row_train_subsample_window = int(
        row.get("train_subsample_window", str(frozen["train_subsample_window"]))
    )
    row_val_window = int(row.get("val_window", str(frozen["val_window"])))
    row_val_stride = int(row.get("val_stride", str(frozen["val_stride"])))

    config = build_config(
        train_docs=parse_train_docs(row["train_docs"]),
        xp_name=row.get("xp_name", f"row_{row_id}"),
        sampling_mode=row.get("sampling_mode", "boundary_mix"),
        decoder_mode=row.get("decoder_mode", "independent"),
        boundary_head=parse_boolish(row.get("boundary_head", "0")),
        boundary_loss_weight=float(row.get("boundary_loss_weight", "0.0")),
        token_loss_mode=row.get("token_loss_mode", "focal"),
        token_loss_weight=float(row.get("token_loss_weight", "1.0")),
        crf_loss_weight=float(row.get("crf_loss_weight", "0.0")),
        label_smoothing=float(row.get("label_smoothing", "0.0")),
        preserve_case=parse_boolish(row.get("preserve_case", "0")),
        split_version=split_version,
        seed=row_seed,
        git_commit=git_commit,
        model_name=str(frozen["model_name"]),
        batch_size=int(frozen["batch_size"]),
        train_subsample_window=row_train_subsample_window,
        val_window=row_val_window,
        val_stride=row_val_stride,
        max_epochs=int(frozen["max_epochs"]),
        learning_rate=row_learning_rate,
        weight_decay=float(frozen["weight_decay"]),
        warmup_steps_pct=float(frozen["warmup_steps_pct"]),
        data_path=data_path or str(DATA_NER_DIR / "ner-data.parquet"),
    )
    return run_training_and_eval(config, eval_split=eval_split, run_test=True)


def run_article_audit(
    *,
    ckpt_path: str,
    output_path: str,
    eval_split: EvalSplit,
    split_version: str,
    data_path: str,
    batch_size: int,
    val_window: int,
    val_stride: int,
    context_chars: int,
    limit: int,
) -> dict[str, int]:
    build_article_failure_records, _ = _load_entity_audit_builders()
    inference = NERInference(ckpt_path=ckpt_path, label_list=None)
    model: NERTagger = inference.model
    model_name = cast(str, getattr(model.hparams, "model_name"))

    ner_trainer = NERTrainer(
        data_path=data_path,
        model_name=model_name,
        label_list=inference.label_list,
        num_trials=0,
        max_epochs=1,
        split_version=split_version,
        train_docs=0,
        seed=42,
    )
    ner_trainer.load_data()
    eval_data = ner_trainer.val_data if eval_split == "val" else ner_trainer.test_data
    if not eval_data:
        raise RuntimeError(f"No {eval_split} data available for article audit.")

    data_module = NERDataModule(
        train_data=[],
        val_data=[],
        test_data=eval_data,
        tokenizer_name=model_name,
        label_list=inference.label_list,
        batch_size=batch_size,
        train_subsample_window=val_window,
        num_workers=0,
        val_window=val_window,
        val_stride=val_stride,
        preserve_case=bool(getattr(model.hparams, "preserve_case", False)),
    )
    data_module.setup("test")

    if hasattr(model, "hparams"):
        _ = setattr(model.hparams, "metrics_output_dir", None)
    setattr(model, "eval_log_prefix", eval_split)

    pl_trainer = pl.Trainer(
        accelerator=ner_trainer.device,
        precision=ner_trainer.trainer_precision(),
        devices=1,
        logger=False,
    )
    _ = pl_trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)

    records: list[ArticleFailureRecord] = []
    for doc in model.collect_eval_docs():
        pred_tags = doc["pred_tags_raw"]
        gold_tags = doc["gold_tags"]
        doc_records = build_article_failure_records(
            doc_id=int(doc["doc_id"]),
            pred_tags=pred_tags,
            gold_tags=gold_tags,
            raw_text=doc["raw_text"],
            token_offsets=doc["token_offsets"],
            context_chars=context_chars,
        )
        records.extend(doc_records)

    if limit > 0:
        records = records[:limit]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            _ = f.write(json.dumps(record, ensure_ascii=True) + "\n")

    counts = {
        "total": len(records),
        "boundary_mismatch": sum(
            1 for record in records if record["failure_type"] == "boundary_mismatch"
        ),
        "false_negative": sum(
            1 for record in records if record["failure_type"] == "false_negative"
        ),
        "false_positive": sum(
            1 for record in records if record["failure_type"] == "false_positive"
        ),
    }
    print(f"Wrote {counts['total']} ARTICLE audit cases to {output_path}")
    print(json.dumps(counts, indent=2, sort_keys=True))
    return counts


def run_page_audit(
    *,
    ckpt_path: str,
    output_path: str,
    eval_split: EvalSplit,
    split_version: str,
    data_path: str,
    batch_size: int,
    val_window: int,
    val_stride: int,
    context_chars: int,
    limit: int,
) -> dict[str, int]:
    _, build_page_failure_records = _load_entity_audit_builders()
    inference = NERInference(ckpt_path=ckpt_path, label_list=None)
    model: NERTagger = inference.model
    model_name = cast(str, getattr(model.hparams, "model_name"))
    ner_trainer = NERTrainer(
        data_path=data_path,
        model_name=model_name,
        label_list=inference.label_list,
        num_trials=0,
        max_epochs=1,
        split_version=split_version,
        train_docs=0,
        seed=42,
    )
    ner_trainer.load_data()

    eval_data = ner_trainer.val_data if eval_split == "val" else ner_trainer.test_data
    if not eval_data:
        raise RuntimeError(f"No {eval_split} data available for page audit.")

    data_module = NERDataModule(
        train_data=[],
        val_data=[],
        test_data=eval_data,
        tokenizer_name=model_name,
        label_list=inference.label_list,
        batch_size=batch_size,
        train_subsample_window=val_window,
        num_workers=0,
        val_window=val_window,
        val_stride=val_stride,
        preserve_case=bool(getattr(model.hparams, "preserve_case", False)),
    )
    data_module.setup("test")

    if hasattr(model, "hparams"):
        _ = setattr(model.hparams, "metrics_output_dir", None)
    setattr(model, "eval_log_prefix", eval_split)

    pl_trainer = pl.Trainer(
        accelerator=ner_trainer.device,
        precision=ner_trainer.trainer_precision(),
        devices=1,
        logger=False,
    )
    _ = pl_trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)

    records: list[PageFailureRecord] = []
    for doc in model.collect_eval_docs():
        pred_tags = doc["pred_tags_raw"]
        gold_tags = doc["gold_tags"]
        doc_records = build_page_failure_records(
            doc_id=int(doc["doc_id"]),
            pred_tags=pred_tags,
            gold_tags=gold_tags,
            raw_text=doc["raw_text"],
            token_offsets=doc["token_offsets"],
            context_chars=context_chars,
        )
        records.extend(doc_records)

    if limit > 0:
        records = records[:limit]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            _ = f.write(json.dumps(record, ensure_ascii=True) + "\n")

    counts = {
        "total": len(records),
        "boundary_mismatch": sum(
            1 for record in records if record["failure_type"] == "boundary_mismatch"
        ),
        "false_negative": sum(
            1 for record in records if record["failure_type"] == "false_negative"
        ),
        "false_positive": sum(
            1 for record in records if record["failure_type"] == "false_positive"
        ),
    }
    print(f"Wrote {counts['total']} PAGE audit cases to {output_path}")
    print(json.dumps(counts, indent=2, sort_keys=True))
    return counts


def parse_train_docs(value: str) -> int:
    lowered = value.strip().lower()
    if lowered == "all":
        return 0
    try:
        parsed = int(lowered)
    except ValueError as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(
            f"train-docs must be an int or 'all', got {value!r}"
        ) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("train-docs must be >= 0")
    return parsed


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NER training commands")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tune_parser = subparsers.add_parser("tune", help="Run Optuna hyperparameter search")
    _ = tune_parser.add_argument("--num-trials", type=int, default=10)
    _ = tune_parser.add_argument("--max-epochs", type=int, default=10)
    _ = tune_parser.add_argument("--split-version", type=str, default="default")
    _ = tune_parser.add_argument("--seed", type=int, default=42)
    _ = tune_parser.add_argument(
        "--data-path", type=str, default=str(DATA_NER_DIR / "ner-data.parquet")
    )
    _ = tune_parser.add_argument(
        "--data-csv",
        dest="data_path",
        type=str,
        default=str(DATA_NER_DIR / "ner-data.parquet"),
        help=argparse.SUPPRESS,
    )
    _ = tune_parser.add_argument(
        "--model-name", type=str, default="answerdotai/ModernBERT-base"
    )
    _ = tune_parser.add_argument("--git-commit", type=str, default=None)

    grid_parser = subparsers.add_parser("grid-row", help="Run a single grid row")
    _ = grid_parser.add_argument("--row-id", type=int, required=True)
    _ = grid_parser.add_argument(
        "--split-version", type=str, default="default"
    )
    _ = grid_parser.add_argument("--seed", type=int, default=42)
    _ = grid_parser.add_argument("--git-commit", type=str, default=None)
    _ = grid_parser.add_argument(
        "--grid-path", type=str, default=str(CONFIG_NER_DIR / "grid.csv")
    )
    _ = grid_parser.add_argument(
        "--frozen-config-path",
        type=str,
        default=str(FROZEN_EXPERIMENT_CONFIG_PATH),
    )
    _ = grid_parser.add_argument(
        "--data-path", type=str, default=str(DATA_NER_DIR / "ner-data.parquet")
    )
    _ = grid_parser.add_argument(
        "--eval-split", type=str, choices=["val", "test"], default="val"
    )

    final_parser = subparsers.add_parser("final-train", help="Train a final model")
    _ = final_parser.add_argument("--train-docs", type=parse_train_docs, required=True)
    _ = final_parser.add_argument("--xp-name", type=str, default="final")
    _ = final_parser.add_argument(
        "--sampling-mode",
        type=str,
        choices=["boundary_mix"],
        default="boundary_mix",
    )
    _ = final_parser.add_argument(
        "--decoder-mode",
        type=str,
        choices=["independent", "crf"],
        default="independent",
    )
    _ = final_parser.add_argument("--boundary-head", action="store_true", default=False)
    _ = final_parser.add_argument("--boundary-loss-weight", type=float, default=0.0)
    _ = final_parser.add_argument(
        "--token-loss-mode", type=str, choices=["focal", "ce"], default="focal"
    )
    _ = final_parser.add_argument("--token-loss-weight", type=float, default=1.0)
    _ = final_parser.add_argument("--crf-loss-weight", type=float, default=0.0)
    _ = final_parser.add_argument("--label-smoothing", type=float, default=0.0)
    _ = final_parser.add_argument("--preserve-case", action="store_true", default=False)
    _ = final_parser.add_argument(
        "--frozen-config-path",
        type=str,
        default=str(FROZEN_EXPERIMENT_CONFIG_PATH),
    )
    _ = final_parser.add_argument("--split-version", type=str, default="default")
    _ = final_parser.add_argument("--seed", type=int, default=42)
    _ = final_parser.add_argument("--git-commit", type=str, default=None)
    _ = final_parser.add_argument("--train-subsample-window", type=int, default=None)
    _ = final_parser.add_argument("--val-window", type=int, default=None)
    _ = final_parser.add_argument("--val-stride", type=int, default=None)
    _ = final_parser.add_argument(
        "--data-path", type=str, default=str(DATA_NER_DIR / "ner-data.parquet")
    )

    recover_parser = subparsers.add_parser(
        "recover-run", help="Append a completed run back into experiments_xp.csv"
    )
    _ = recover_parser.add_argument("--run-dir", type=str, required=True)
    _ = recover_parser.add_argument("--experiments-csv", type=str, default=None)

    audit_parser = subparsers.add_parser(
        "audit-articles", help="Export problematic ARTICLE cases for a checkpoint"
    )
    _ = audit_parser.add_argument("--ckpt-path", type=str, required=True)
    _ = audit_parser.add_argument(
        "--output-path",
        type=str,
        default=str(LOG_NER_DIR / "article_audit_cases.jsonl"),
    )
    _ = audit_parser.add_argument(
        "--eval-split", type=str, choices=["val", "test"], default="val"
    )
    _ = audit_parser.add_argument("--split-version", type=str, default="default")
    _ = audit_parser.add_argument(
        "--data-path", type=str, default=str(DATA_NER_DIR / "ner-data.parquet")
    )
    _ = audit_parser.add_argument(
        "--data-csv",
        dest="data_path",
        type=str,
        default=str(DATA_NER_DIR / "ner-data.parquet"),
        help=argparse.SUPPRESS,
    )
    _ = audit_parser.add_argument("--batch-size", type=int, default=8)
    _ = audit_parser.add_argument("--val-window", type=int, default=510)
    _ = audit_parser.add_argument("--val-stride", type=int, default=256)
    _ = audit_parser.add_argument("--context-chars", type=int, default=160)
    _ = audit_parser.add_argument("--limit", type=int, default=500)

    page_audit_parser = subparsers.add_parser(
        "audit-pages", help="Export problematic PAGE cases for a checkpoint"
    )
    _ = page_audit_parser.add_argument("--ckpt-path", type=str, required=True)
    _ = page_audit_parser.add_argument(
        "--output-path",
        type=str,
        default=str(LOG_NER_DIR / "page_audit_cases.jsonl"),
    )
    _ = page_audit_parser.add_argument(
        "--eval-split", type=str, choices=["val", "test"], default="val"
    )
    _ = page_audit_parser.add_argument("--split-version", type=str, default="default")
    _ = page_audit_parser.add_argument(
        "--data-path", type=str, default=str(DATA_NER_DIR / "ner-data.parquet")
    )
    _ = page_audit_parser.add_argument(
        "--data-csv",
        dest="data_path",
        type=str,
        default=str(DATA_NER_DIR / "ner-data.parquet"),
        help=argparse.SUPPRESS,
    )
    _ = page_audit_parser.add_argument("--batch-size", type=int, default=8)
    _ = page_audit_parser.add_argument("--val-window", type=int, default=510)
    _ = page_audit_parser.add_argument("--val-stride", type=int, default=256)
    _ = page_audit_parser.add_argument("--context-chars", type=int, default=160)
    _ = page_audit_parser.add_argument("--limit", type=int, default=500)

    return parser


def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()

    if args.command == "tune":
        _ = run_hparam_tuning(
            data_path=args.data_path,
            model_name=args.model_name,
            num_trials=args.num_trials,
            max_epochs=args.max_epochs,
            split_version=args.split_version,
            seed=args.seed,
            git_commit=args.git_commit,
        )
        return

    if args.command == "grid-row":
        _ = run_grid_row(
            row_id=args.row_id,
            split_version=args.split_version,
            seed=args.seed,
            git_commit=args.git_commit,
            grid_path=args.grid_path,
            frozen_config_path=args.frozen_config_path,
            data_path=args.data_path,
            eval_split=cast(EvalSplit, args.eval_split),
        )
        return

    if args.command == "final-train":
        frozen = load_frozen_experiment_config(path=args.frozen_config_path)
        final_train_subsample_window = (
            args.train_subsample_window
            if args.train_subsample_window is not None
            else int(frozen["train_subsample_window"])
        )
        final_val_window = (
            args.val_window
            if args.val_window is not None
            else int(frozen["val_window"])
        )
        final_val_stride = (
            args.val_stride
            if args.val_stride is not None
            else int(frozen["val_stride"])
        )
        config = build_config(
            train_docs=args.train_docs,
            xp_name=args.xp_name,
            sampling_mode=args.sampling_mode,
            decoder_mode=args.decoder_mode,
            boundary_head=args.boundary_head,
            boundary_loss_weight=args.boundary_loss_weight,
            token_loss_mode=args.token_loss_mode,
            token_loss_weight=args.token_loss_weight,
            crf_loss_weight=args.crf_loss_weight,
            label_smoothing=args.label_smoothing,
            preserve_case=args.preserve_case,
            split_version=args.split_version,
            seed=args.seed,
            git_commit=args.git_commit,
            model_name=str(frozen["model_name"]),
            batch_size=int(frozen["batch_size"]),
            train_subsample_window=final_train_subsample_window,
            val_window=final_val_window,
            val_stride=final_val_stride,
            max_epochs=int(frozen["max_epochs"]),
            learning_rate=float(frozen["learning_rate"]),
            weight_decay=float(frozen["weight_decay"]),
            warmup_steps_pct=float(frozen["warmup_steps_pct"]),
            data_path=args.data_path,
        )
        final_run_dir = str(EVAL_NER_DIR / "final")
        _ = run_training_and_eval(
            config,
            eval_split="test",
            run_test=True,
            ckpt_path=NER_CKPT,
            run_dir=final_run_dir,
            metrics_output_name="metrics.yaml",
            log_experiment=False,
            log_stage="final",
        )
        return

    if args.command == "recover-run":
        _ = recover_experiment_row_from_run_dir(
            run_dir=args.run_dir,
            experiments_csv=args.experiments_csv,
        )
        return

    if args.command == "audit-articles":
        _ = run_article_audit(
            ckpt_path=args.ckpt_path,
            output_path=args.output_path,
            eval_split=cast(EvalSplit, args.eval_split),
            split_version=args.split_version,
            data_path=args.data_path,
            batch_size=args.batch_size,
            val_window=args.val_window,
            val_stride=args.val_stride,
            context_chars=args.context_chars,
            limit=args.limit,
        )
        return

    if args.command == "audit-pages":
        _ = run_page_audit(
            ckpt_path=args.ckpt_path,
            output_path=args.output_path,
            eval_split=cast(EvalSplit, args.eval_split),
            split_version=args.split_version,
            data_path=args.data_path,
            batch_size=args.batch_size,
            val_window=args.val_window,
            val_stride=args.val_stride,
            context_chars=args.context_chars,
            limit=args.limit,
        )
        return

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
