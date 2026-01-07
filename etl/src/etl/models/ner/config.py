"""
NER experiment configuration helpers.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, TypedDict, cast

import yaml


if TYPE_CHECKING:
    from .utils.paths import CONFIG_NER_DIR, DATA_NER_DIR, get_git_root
else:
    try:
        from .utils.paths import CONFIG_NER_DIR, DATA_NER_DIR, get_git_root
    except ImportError:  # pragma: no cover - supports running as a script
        from utils.paths import (
            CONFIG_NER_DIR,
            DATA_NER_DIR,
            get_git_root,
        )


OPTUNA_BEST_CONFIG_PATH = CONFIG_NER_DIR / "optuna_best_config.yaml"

DEFAULT_DATA_CSV = str(DATA_NER_DIR / "ner-data.csv")
DEFAULT_MODEL_NAME = "answerdotai/ModernBERT-base"
DEFAULT_BATCH_SIZE = 16
DEFAULT_TRAIN_SUBSAMPLE_WINDOW = 256
DEFAULT_VAL_WINDOW = 510
DEFAULT_VAL_STRIDE = 256
DEFAULT_MAX_EPOCHS = 10
DEFAULT_LEARNING_RATE = 3e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_STEPS_PCT = 0.1
DEFAULT_ARTICLE_WEIGHT = 3.0
DEFAULT_SPLIT_VERSION = "default"
DEFAULT_SEED = 42
_GATING_MODE_ALIASES = {"regex+snap": "snap"}


@dataclass(frozen=True)
class NERExperimentConfig:
    """
    Configuration for a single NER experiment run.
    """

    run_id: str
    seed: int
    split_version: str
    train_docs: int
    article_weight: float
    gating_mode: str
    model_name: str
    batch_size: int
    train_subsample_window: int
    val_window: int
    val_stride: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_steps_pct: float
    data_csv: str
    git_commit: str


class OptunaBestConfig(TypedDict):
    batch_size: int
    learning_rate: float
    train_subsample_window: int
    val_window: int
    val_stride: int
    weight_decay: float
    warmup_steps_pct: float
    max_epochs: int
    model_name: str


def _config_fingerprint(payload: Mapping[str, object]) -> str:
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(stable.encode("utf-8")).hexdigest()[:8]


def _build_run_id(payload: Mapping[str, object]) -> str:
    return f"run_{int(time.time())}_{_config_fingerprint(payload)}"


def _normalize_gating_mode(gating_mode: str) -> str:
    normalized = gating_mode.strip().lower()
    return _GATING_MODE_ALIASES.get(normalized, normalized)


def _detect_git_commit() -> str | None:
    git_root = get_git_root()
    if git_root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(git_root),
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def resolve_git_commit(explicit: str | None) -> str:
    """
    Resolve git commit with optional override, falling back to 'unknown'.
    """
    if explicit:
        return explicit
    detected = _detect_git_commit()
    if detected:
        return detected
    print(
        "[git] warning: unable to detect git commit; using 'unknown'. "
        + "Pass --git-commit to record it."
    )
    return "unknown"


def build_config(
    *,
    train_docs: int,
    article_weight: float,
    gating_mode: str,
    split_version: str,
    seed: int,
    git_commit: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    train_subsample_window: int = DEFAULT_TRAIN_SUBSAMPLE_WINDOW,
    val_window: int = DEFAULT_VAL_WINDOW,
    val_stride: int = DEFAULT_VAL_STRIDE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    warmup_steps_pct: float = DEFAULT_WARMUP_STEPS_PCT,
    data_csv: str = DEFAULT_DATA_CSV,
) -> NERExperimentConfig:
    """
    Build a fully-populated experiment config with a generated run_id.
    """
    class _ConfigPayload(TypedDict):
        seed: int
        split_version: str
        train_docs: int
        article_weight: float
        gating_mode: str
        model_name: str
        batch_size: int
        train_subsample_window: int
        val_window: int
        val_stride: int
        max_epochs: int
        learning_rate: float
        weight_decay: float
        warmup_steps_pct: float
        data_csv: str
        git_commit: str

    normalized_gating_mode = _normalize_gating_mode(gating_mode)
    payload: _ConfigPayload = {
        "seed": seed,
        "split_version": split_version,
        "train_docs": train_docs,
        "article_weight": article_weight,
        "gating_mode": normalized_gating_mode,
        "model_name": model_name,
        "batch_size": batch_size,
        "train_subsample_window": train_subsample_window,
        "val_window": val_window,
        "val_stride": val_stride,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps_pct": warmup_steps_pct,
        "data_csv": data_csv,
        "git_commit": resolve_git_commit(git_commit),
    }
    run_id = _build_run_id(payload)
    return NERExperimentConfig(run_id=run_id, **payload)


def load_config_from_cli() -> NERExperimentConfig:
    """
    Parse CLI args and return a populated NERExperimentConfig.
    """
    parser = argparse.ArgumentParser(description="NER experiment config")
    _ = parser.add_argument("--train-docs", type=int, default=0)
    _ = parser.add_argument("--article-weight", type=float, default=DEFAULT_ARTICLE_WEIGHT)
    _ = parser.add_argument(
        "--gating-mode",
        type=str,
        choices=["raw", "regex", "snap", "regex+snap"],
        default="raw",
    )
    _ = parser.add_argument("--split-version", type=str, default=DEFAULT_SPLIT_VERSION)
    _ = parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    _ = parser.add_argument("--git-commit", type=str, default=None)
    _ = parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    _ = parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    _ = parser.add_argument(
        "--train-subsample-window", type=int, default=DEFAULT_TRAIN_SUBSAMPLE_WINDOW
    )
    _ = parser.add_argument("--val-window", type=int, default=DEFAULT_VAL_WINDOW)
    _ = parser.add_argument("--val-stride", type=int, default=DEFAULT_VAL_STRIDE)
    _ = parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    _ = parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    _ = parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    _ = parser.add_argument(
        "--warmup-steps-pct", type=float, default=DEFAULT_WARMUP_STEPS_PCT
    )
    _ = parser.add_argument("--data-csv", type=str, default=DEFAULT_DATA_CSV)
    args = parser.parse_args()

    return build_config(
        train_docs=cast(int, args.train_docs),
        article_weight=cast(float, args.article_weight),
        gating_mode=cast(str, args.gating_mode),
        split_version=cast(str, args.split_version),
        seed=cast(int, args.seed),
        git_commit=cast(str | None, args.git_commit),
        model_name=cast(str, args.model_name),
        batch_size=cast(int, args.batch_size),
        train_subsample_window=cast(int, args.train_subsample_window),
        val_window=cast(int, args.val_window),
        val_stride=cast(int, args.val_stride),
        max_epochs=cast(int, args.max_epochs),
        learning_rate=cast(float, args.learning_rate),
        weight_decay=cast(float, args.weight_decay),
        warmup_steps_pct=cast(float, args.warmup_steps_pct),
        data_csv=cast(str, args.data_csv),
    )


def config_to_dict(config: NERExperimentConfig) -> dict[str, object]:
    """
    Convert a config dataclass into a JSON-serializable dict.
    """
    return asdict(config)


def load_optuna_best_config(
    path: str | None = None,
) -> OptunaBestConfig:
    """
    Load frozen hyperparameters for experiment runs.
    """
    resolved = path or str(OPTUNA_BEST_CONFIG_PATH)
    if not os.path.exists(resolved):
        raise RuntimeError(
            f"optuna_best_config.yaml not found at {resolved}. Run normal training to generate it or create it manually."
        )
    with open(resolved, "r", encoding="utf-8") as f:
        raw: object = cast(object, yaml.safe_load(f))
    if not isinstance(raw, dict):
        raise RuntimeError("optuna_best_config.yaml must be a YAML mapping.")
    data: dict[str, object] = cast(dict[str, object], raw)

    required = {
        "batch_size",
        "learning_rate",
        "train_subsample_window",
        "val_window",
        "val_stride",
        "weight_decay",
        "warmup_steps_pct",
        "max_epochs",
    }
    missing = required - set(data)
    if missing:
        raise RuntimeError(
            f"optuna_best_config.yaml missing required keys: {sorted(missing)}"
        )

    def _require_int(value: object, name: str) -> int:
        if not isinstance(value, int):
            raise TypeError(f"{name} must be int, got {type(value).__name__}")
        return value

    def _require_float(value: object, name: str) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        raise TypeError(f"{name} must be float, got {type(value).__name__}")

    def _require_str(value: object, name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be str, got {type(value).__name__}")
        return value

    payload: OptunaBestConfig = {
        "batch_size": _require_int(data["batch_size"], "batch_size"),
        "learning_rate": _require_float(data["learning_rate"], "learning_rate"),
        "train_subsample_window": _require_int(
            data["train_subsample_window"], "train_subsample_window"
        ),
        "val_window": _require_int(data["val_window"], "val_window"),
        "val_stride": _require_int(data["val_stride"], "val_stride"),
        "weight_decay": _require_float(data["weight_decay"], "weight_decay"),
        "warmup_steps_pct": _require_float(
            data["warmup_steps_pct"], "warmup_steps_pct"
        ),
        "max_epochs": _require_int(data["max_epochs"], "max_epochs"),
        "model_name": (
            _require_str(data["model_name"], "model_name")
            if "model_name" in data
            else DEFAULT_MODEL_NAME
        ),
    }
    return payload
