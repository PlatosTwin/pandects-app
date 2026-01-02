"""
Main NER training and inference module.

This module provides the main entry points for training and testing the NER model
using PyTorch Lightning with hyperparameter optimization via Optuna.
"""
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false

# Standard library
import csv
import hashlib
import json
import os
import time
import yaml
from typing import Callable, Literal, TYPE_CHECKING, TypeAlias, cast

# Environment config
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Data manipulation
import pandas as pd
from sklearn.model_selection import train_test_split

# ML frameworks and utilities
import torch

torch.set_float32_matmul_precision("high")

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
    from .config import (
        NERExperimentConfig,
        OPTUNA_BEST_CONFIG_PATH,
        config_to_dict,
    )
    from .shared_constants import (
        NER_LABEL_LIST,
        NER_CKPT_PATH,
        SPECIAL_TOKENS_TO_ADD,
    )
    from .ner_classes import NERTagger, NERDataModule, ascii_lower
else:
    try:
        from .config import (
            NERExperimentConfig,
            OPTUNA_BEST_CONFIG_PATH,
            config_to_dict,
        )
        from .shared_constants import (
            NER_LABEL_LIST,
            NER_CKPT_PATH,
            SPECIAL_TOKENS_TO_ADD,
        )
        from .ner_classes import NERTagger, NERDataModule, ascii_lower
    except ImportError:  # pragma: no cover - supports running as a script
        from config import (
            NERExperimentConfig,
            OPTUNA_BEST_CONFIG_PATH,
            config_to_dict,
        )
        from shared_constants import (
            NER_LABEL_LIST,
            NER_CKPT_PATH,
            SPECIAL_TOKENS_TO_ADD,
        )
        from ner_classes import (
            NERTagger,
            NERDataModule,
            ascii_lower,
        )

PrecisionInput = Literal["bf16-mixed", "32-true"]
WindowItem: TypeAlias = dict[str, list[int]]

# Reproducibility
_ = seed_everything(42, workers=True, verbose=False)

CODE_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.normpath(os.path.join(CODE_DIR, "../../../../.."))
EVAL_METRICS_DIR = os.path.normpath(os.path.join(CODE_DIR, "../eval_metrics"))
DATA_DIR = os.path.normpath(os.path.join(CODE_DIR, "../data"))
CONFIGS_DIR = os.path.normpath(os.path.join(CODE_DIR, "../configs"))
MODEL_FILES_DIR = os.path.normpath(os.path.join(CODE_DIR, "../model_files"))
NER_SPLIT_PATH = os.path.join(DATA_DIR, "ner-page-splits.json")

NER_LABELS: list[str] = [str(label) for label in NER_LABEL_LIST]
NER_CKPT: str = str(NER_CKPT_PATH)
SPECIAL_TOKENS: list[str] = [str(token) for token in SPECIAL_TOKENS_TO_ADD]
YEAR_WINDOW = 5


def _split_path_for_version(split_version: str) -> str:
    if split_version in ("", "default"):
        return NER_SPLIT_PATH
    return os.path.join(DATA_DIR, f"ner-page-splits-{split_version}.json")


def _hash_sorted_ids(ids: list[str], seed: int) -> list[str]:
    keyed: list[tuple[str, str]] = []
    for pid in ids:
        digest = hashlib.sha1(f"{seed}:{pid}".encode("utf-8")).hexdigest()
        keyed.append((digest, pid))
    keyed.sort()
    return [pid for _, pid in keyed]


def _metrics_dir_for_job(base_dir: str) -> str:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("SLURM_JOB_ID is required to write eval metrics on HPC.")
    return os.path.join(base_dir, job_id)


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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"[optuna] wrote best hyperparameters to {path}")


class NERTrainer:
    """
    Orchestrates hyperparameter optimization and training of NERTagger.

    Uses Optuna for hyperparameter search and PyTorch Lightning for training.
    """

    def __init__(
        self,
        data_csv: str,
        model_name: str,
        label_list: list[str],
        num_trials: int,
        max_epochs: int,
        split_version: str = "default",
        train_docs: int | None = None,
        seed: int = 42,
        article_class_weight: float = 3.0,
        gating_mode: str = "raw",
        val_window: int = 510,
        val_stride: int = 256,
    ):
        """
        Initialize the NER trainer.

        Args:
            data_csv: Path to the data file
            model_name: HuggingFace model name
            label_list: List of label names
            num_trials: Number of Optuna trials
            max_epochs: Maximum training epochs per trial
            split_version: Named split manifest to use
            train_docs: Optional number of training documents to include
            seed: Seed for split creation and training
            article_class_weight: Weight for ARTICLE class in loss
            gating_mode: Postprocessing mode for evaluation
            val_window: Validation window size for evaluation
            val_stride: Validation stride size for evaluation
        """
        self.data_csv = data_csv
        self.model_name = model_name
        self.num_trials = num_trials
        self.max_epochs = max_epochs
        self.label_list = label_list
        self.split_version = split_version
        self.train_docs = train_docs
        self.seed = seed
        self.article_class_weight = article_class_weight
        self.gating_mode = gating_mode
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
        self.metrics_output_dir = _metrics_dir_for_job(EVAL_METRICS_DIR)

    def _load_data(self) -> None:
        """
        Load and split data, stratified by announcement window.
        """
        df = pd.read_csv(self.data_csv)

        required_cols = {
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
        train_df = df[df["page_uuid"].astype(str).isin(set(train_ids_sorted))]
        val_df = df[df["page_uuid"].astype(str).isin(val_ids)]
        test_df = df[df["page_uuid"].astype(str).isin(test_ids)]

        if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
            raise ValueError("Split manifest has overlapping page_uuid values.")
        if not train_df["article_upsample"].any():
            print("[split] warning: no article_upsample rows in train split.")
        if val_df["article_upsample"].any() or test_df["article_upsample"].any():
            raise ValueError("Split manifest contains article_upsample rows in val/test.")

        print(
            f"Splits -> train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}"
        )

        self.train_data = train_df["tagged_text"].to_list()
        self.val_data = val_df["tagged_text"].to_list()
        self.test_data = test_df["tagged_text"].to_list()

    def _load_or_build_split(
        self, df: pd.DataFrame, split_path: str, seed: int
    ) -> dict[str, object]:
        if os.path.exists(split_path):
            with open(split_path, "r", encoding="utf-8") as f:
                split = cast(dict[str, object], json.load(f))
            for key in ("train", "val", "test"):
                if key not in split:
                    raise ValueError("Split manifest missing required keys: train/val/test.")
            df_ids = set(df["page_uuid"].astype(str).tolist())
            split_ids = set(cast(list[str], split["train"])) | set(
                cast(list[str], split["val"])
            ) | set(cast(list[str], split["test"]))
            missing = split_ids - df_ids
            if missing:
                raise ValueError("Split manifest contains unknown page_uuid values.")
            return split

        val_split = 0.1
        test_split = 0.1
        total_split = val_split + test_split
        if total_split >= 1.0:
            raise ValueError("val_split + test_split must be < 1.0")

        base_df = df[~df["article_upsample"]].copy()
        if base_df.empty:
            raise ValueError("No base rows (article_upsample=False) available for val/test.")

        strat_cols = ["announcement_window", "tagged_section", "tagged_article"]
        rare_mask = (base_df["tagged_article"] == 1) & (base_df["tagged_section"] == 0)
        rare_df = base_df[rare_mask].copy()
        main_df = base_df[~rare_mask].copy()

        empty_df = base_df.iloc[0:0].copy()
        if main_df.empty:
            train_main_df, holdout_main_df = empty_df, empty_df
        else:
            main_strat = main_df[strat_cols].astype(str).agg("_".join, axis=1)
            train_main_df, holdout_main_df = cast(
                tuple[pd.DataFrame, pd.DataFrame],
                cast(
                    object,
                    train_test_split(
                        main_df,
                        test_size=total_split,
                        stratify=main_strat,
                        random_state=seed,
                    ),
                ),
            )

        if rare_df.empty:
            rare_train_df, rare_holdout_df = empty_df, empty_df
        else:
            rare_train_df, rare_holdout_df = cast(
                tuple[pd.DataFrame, pd.DataFrame],
                cast(
                    object,
                    train_test_split(
                        rare_df,
                        test_size=total_split,
                        stratify=None,
                        random_state=seed,
                    ),
                ),
            )

        train_base_df = pd.concat(
            [train_main_df, rare_train_df], ignore_index=True
        )
        if holdout_main_df.empty:
            val_main_df, test_main_df = empty_df, empty_df
        else:
            holdout_strat = holdout_main_df[strat_cols].astype(str).agg("_".join, axis=1)
            val_main_df, test_main_df = cast(
                tuple[pd.DataFrame, pd.DataFrame],
                cast(
                    object,
                    train_test_split(
                        holdout_main_df,
                        test_size=test_split / total_split,
                        stratify=holdout_strat,
                        random_state=seed,
                    ),
                ),
            )

        if rare_holdout_df.empty:
            rare_val_df, rare_test_df = empty_df, empty_df
        else:
            rare_val_df, rare_test_df = cast(
                tuple[pd.DataFrame, pd.DataFrame],
                cast(
                    object,
                    train_test_split(
                        rare_holdout_df,
                        test_size=test_split / total_split,
                        stratify=None,
                        random_state=seed,
                    ),
                ),
            )

        val_df = pd.concat([val_main_df, rare_val_df], ignore_index=True)
        test_df = pd.concat([test_main_df, rare_test_df], ignore_index=True)

        upsample_df = df[df["article_upsample"]].copy()
        train_df = pd.concat([train_base_df, upsample_df], ignore_index=True)

        split = {
            "train": train_df["page_uuid"].astype(str).tolist(),
            "val": val_df["page_uuid"].astype(str).tolist(),
            "test": test_df["page_uuid"].astype(str).tolist(),
            "meta": {
                "val_split": val_split,
                "test_split": test_split,
                "seed": seed,
                "stratify_cols": strat_cols,
                "page_uuid_col": "page_uuid",
                "article_upsample_col": "article_upsample",
                "year_window": YEAR_WINDOW,
            },
        }
        os.makedirs(DATA_DIR, exist_ok=True)
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
        progress_bar_callback = TQDMProgressBar(refresh_rate=100)

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
            num_workers=7,
            val_window=int(params.get("val_window", self.val_window)),
            val_stride=int(params.get("val_stride", self.val_stride)),
        )
        model = NERTagger(
            model_name=self.model_name,
            num_labels=len(self.label_list),
            id2label={idx: label for idx, label in enumerate(self.label_list)},
            learning_rate=params["lr"],
            weight_decay=params["weight_decay"],
            warmup_steps_pct=params["warmup_steps_pct"],
            article_class_weight=self.article_class_weight,
            gating_mode=self.gating_mode,
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
            logger=TensorBoardLogger("tb_logs", name="ner/optuna"),
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
            yaml.safe_dump(trial_metrics, f, sort_keys=False)

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
            lr_monitor,
            progress_bar_callback,
            _,
        ) = self._get_callbacks(ckpt=NER_CKPT)

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.device,
            devices=1,
            logger=TensorBoardLogger("tb_logs", name="ner/final"),
            callbacks=[
                checkpoint_callback,
                early_stop_callback,
                lr_monitor,
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

    # ---------------- Token aggregation over sliding windows (logit stitching) ----------------
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
        norm = ascii_lower(text)
        enc_full = cast(
            dict[str, torch.Tensor],
            cast(
                object,
                self.tokenizer(
                    norm,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    truncation=False,
                    add_special_tokens=False,
                ),
            ),
        )
        input_ids = enc_full["input_ids"][0]  # [T]
        offsets_tensor = enc_full["offset_mapping"][0]
        offsets_raw = cast(list[list[int]], offsets_tensor.tolist())
        offsets = [(int(s), int(e)) for s, e in offsets_raw]
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
        preds = cast(list[int], avg_logits.argmax(dim=-1).tolist())

        # --- Light BIOES repair on the predicted sequence ---
        def _repair_bioes(seq: list[int]) -> list[int]:
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

        preds = _repair_bioes(preds)
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
            if pos < s:
                res.append(text[pos:s])
                pos = s
            tok_ent = ent(lid)
            if tok_ent != cur_ent:
                if cur_ent != "O":
                    res.append(f"</{cur_ent}>")
                if tok_ent != "O":
                    res.append(f"<{tok_ent}>")
                cur_ent = tok_ent
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
    file_exists = os.path.exists(csv_path)
    fieldnames = list(row.keys())
    if file_exists:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        if header:
            fieldnames = header
            if set(header) != set(row.keys()):
                raise RuntimeError("experiments.csv header does not match row keys.")

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_experiment(config: "NERExperimentConfig") -> dict[str, object]:
    """
    Train and evaluate a single NER experiment configuration.
    """
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if not slurm_job_id:
        raise RuntimeError("SLURM_JOB_ID is required for experiment logging.")

    _ = seed_everything(config.seed, workers=True, verbose=False)

    ner_trainer = NERTrainer(
        data_csv=config.data_csv,
        model_name=config.model_name,
        label_list=NER_LABELS,
        num_trials=0,
        max_epochs=config.max_epochs,
        split_version=config.split_version,
        train_docs=config.train_docs,
        seed=config.seed,
        article_class_weight=config.article_weight,
        gating_mode=config.gating_mode,
        val_window=config.val_window,
        val_stride=config.val_stride,
    )
    ner_trainer.load_data()

    run_dir = os.path.join(ner_trainer.metrics_output_dir, config.run_id)
    os.makedirs(run_dir, exist_ok=True)
    ner_trainer.metrics_output_dir = run_dir

    ckpt_dir = os.path.join(MODEL_FILES_DIR, "ner_experiments", config.run_id)
    os.makedirs(ckpt_dir, exist_ok=True)

    params = {
        "lr": config.learning_rate,
        "batch_size": config.batch_size,
        "train_subsample_window": config.train_subsample_window,
        "val_window": config.val_window,
        "val_stride": config.val_stride,
        "weight_decay": config.weight_decay,
        "warmup_steps_pct": config.warmup_steps_pct,
    }
    data_module, model = ner_trainer.build(params, metrics_output_name="metrics.yaml")
    ckpt_path = os.path.join(ckpt_dir, "best.ckpt")
    (
        checkpoint_callback,
        early_stop_callback,
        lr_monitor,
        progress_bar_callback,
        _,
    ) = ner_trainer.get_callbacks(ckpt=ckpt_path)

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=ner_trainer.device,
        precision=ner_trainer.trainer_precision(),
        devices=1,
        logger=TensorBoardLogger("tb_logs", name="ner/experiments", version=config.run_id),
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            lr_monitor,
            progress_bar_callback,
        ],
        log_every_n_steps=10,
        deterministic=True,
    )
    trainer.fit(model, datamodule=data_module)
    _ = trainer.test(model, datamodule=data_module, ckpt_path="best")

    metrics = cast(dict[str, object], getattr(model, "test_metrics", None))
    if not metrics:
        raise RuntimeError("Test metrics were not captured from the model.")

    config_dir = os.path.join(CONFIGS_DIR, "ner_experiments", config.run_id)
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_dict(config), f, sort_keys=False)

    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=False)

    variants = cast(dict[str, dict[str, object]], metrics.get("variants", {}))
    flat_metrics: dict[str, float] = {}
    for variant_key in ("raw", "regex", "snap"):
        if variant_key in variants:
            flat_metrics.update(_metrics_with_suffix(variants[variant_key], variant_key))

    row: dict[str, object] = {
        "run_id": config.run_id,
        "slurm_job_id": slurm_job_id,
        "git_commit": config.git_commit,
        "split_version": config.split_version,
        "train_docs": config.train_docs,
        "article_weight": config.article_weight,
        "gating_mode": config.gating_mode,
        "seed": config.seed,
        "run_dir": run_dir,
        **flat_metrics,
    }
    os.makedirs(EVAL_METRICS_DIR, exist_ok=True)
    experiments_csv = os.path.join(EVAL_METRICS_DIR, "experiments.csv")
    _append_experiment_row(experiments_csv, row)

    return metrics


def main(mode: str = "test") -> None:
    """
    Main entry point for NER training and testing.

    Args:
        mode: Either 'train' or 'test'
    """
    if mode == "train":
        ner_trainer = NERTrainer(
            data_csv="../data/ner-data.csv",
            model_name="answerdotai/ModernBERT-base",
            label_list=NER_LABELS,
            num_trials=10,
            max_epochs=10,
            article_class_weight=1.0,
            gating_mode="raw",
        )
        ner_trainer.run()

    elif mode == "test":
        # Load test samples
        with open(os.path.join(DATA_DIR, "ner_samples.yaml"), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        samples = data["samples"]

        # Initialize inference model
        inference_model = NERInference(
            ckpt_path=NER_CKPT, label_list=NER_LABELS, review_threshold=0.99
        )

        # Run inference
        start = time.time()
        tagged_result = inference_model.label(samples, verbose=False)
        inference_time = time.time() - start

        print(f"Inference time: {inference_time:.2f} seconds")
        print(tagged_result)

    else:
        raise RuntimeError(f"Invalid mode: {mode}. Use 'train' or 'test'")


if __name__ == "__main__":
    main(mode="train")
