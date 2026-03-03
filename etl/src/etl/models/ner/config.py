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
from collections.abc import Callable, Mapping
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
FROZEN_EXPERIMENT_CONFIG_PATH = CONFIG_NER_DIR / "frozen_experiment_config.yaml"

DEFAULT_DATA_PATH = str(DATA_NER_DIR / "ner-data.parquet")
DEFAULT_MODEL_NAME = "answerdotai/ModernBERT-base"
DEFAULT_BATCH_SIZE = 16
DEFAULT_TRAIN_SUBSAMPLE_WINDOW = 256
DEFAULT_VAL_WINDOW = 510
DEFAULT_VAL_STRIDE = 256
DEFAULT_MAX_EPOCHS = 10
DEFAULT_LEARNING_RATE = 3e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_STEPS_PCT = 0.1
DEFAULT_SPLIT_VERSION = "default"
DEFAULT_SEED = 42
DEFAULT_XP_NAME = "baseline"
DEFAULT_SAMPLING_MODE = "boundary_mix"
DEFAULT_DECODER_MODE = "independent"
DEFAULT_BOUNDARY_HEAD = False
DEFAULT_BOUNDARY_LOSS_WEIGHT = 0.0
DEFAULT_TOKEN_LOSS_MODE = "focal"
DEFAULT_TOKEN_LOSS_WEIGHT = 1.0
DEFAULT_CRF_LOSS_WEIGHT = 0.0
DEFAULT_LABEL_SMOOTHING = 0.0
DEFAULT_PRESERVE_CASE = False


def _yaml_safe_load(stream: object) -> object:
    safe_load = cast(Callable[[object], object], getattr(yaml, "safe_load"))
    return safe_load(stream)


@dataclass(frozen=True)
class NERExperimentConfig:
    """
    Configuration for a single NER experiment run.
    """

    run_id: str
    seed: int
    split_version: str
    xp_name: str
    train_docs: int
    sampling_mode: str
    decoder_mode: str
    boundary_head: bool
    boundary_loss_weight: float
    token_loss_mode: str
    token_loss_weight: float
    crf_loss_weight: float
    label_smoothing: float
    preserve_case: bool
    model_name: str
    batch_size: int
    train_subsample_window: int
    val_window: int
    val_stride: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_steps_pct: float
    data_path: str
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


class FrozenExperimentConfig(TypedDict):
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


def _normalize_sampling_mode(sampling_mode: str) -> str:
    normalized = sampling_mode.strip().lower()
    if normalized != "boundary_mix":
        raise ValueError("Unsupported sampling_mode. Expected 'boundary_mix'.")
    return normalized


def _normalize_decoder_mode(decoder_mode: str) -> str:
    normalized = decoder_mode.strip().lower()
    allowed = {"independent", "crf"}
    if normalized not in allowed:
        raise ValueError(
            f"Unsupported decoder_mode {decoder_mode!r}. Expected one of {sorted(allowed)}."
        )
    return normalized


def _normalize_token_loss_mode(token_loss_mode: str) -> str:
    normalized = token_loss_mode.strip().lower()
    allowed = {"focal", "ce"}
    if normalized not in allowed:
        raise ValueError(
            f"Unsupported token_loss_mode {token_loss_mode!r}. Expected one of {sorted(allowed)}."
        )
    return normalized


def _normalize_boundary_head(boundary_head: bool) -> bool:
    return bool(boundary_head)


def _validate_loss_configuration(
    *,
    decoder_mode: str,
    boundary_head: bool,
    boundary_loss_weight: float,
    token_loss_mode: str,
    token_loss_weight: float,
    crf_loss_weight: float,
    label_smoothing: float,
) -> None:
    if boundary_loss_weight < 0.0:
        raise ValueError("boundary_loss_weight must be >= 0.")
    if token_loss_weight < 0.0:
        raise ValueError("token_loss_weight must be >= 0.")
    if crf_loss_weight < 0.0:
        raise ValueError("crf_loss_weight must be >= 0.")
    if label_smoothing < 0.0 or label_smoothing >= 1.0:
        raise ValueError("label_smoothing must be in [0.0, 1.0).")
    if not boundary_head and boundary_loss_weight != 0.0:
        raise ValueError(
            "boundary_loss_weight must be 0 when boundary_head is disabled."
        )
    if decoder_mode == "independent" and crf_loss_weight != 0.0:
        raise ValueError(
            "crf_loss_weight must be 0 when decoder_mode is 'independent'."
        )
    if token_loss_mode == "focal" and label_smoothing != 0.0:
        raise ValueError(
            "label_smoothing is only supported with token_loss_mode='ce'."
        )


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
    split_version: str,
    seed: int,
    xp_name: str = DEFAULT_XP_NAME,
    sampling_mode: str = DEFAULT_SAMPLING_MODE,
    decoder_mode: str = DEFAULT_DECODER_MODE,
    boundary_head: bool = DEFAULT_BOUNDARY_HEAD,
    boundary_loss_weight: float = DEFAULT_BOUNDARY_LOSS_WEIGHT,
    token_loss_mode: str = DEFAULT_TOKEN_LOSS_MODE,
    token_loss_weight: float = DEFAULT_TOKEN_LOSS_WEIGHT,
    crf_loss_weight: float = DEFAULT_CRF_LOSS_WEIGHT,
    label_smoothing: float = DEFAULT_LABEL_SMOOTHING,
    preserve_case: bool = DEFAULT_PRESERVE_CASE,
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
    data_path: str = DEFAULT_DATA_PATH,
) -> NERExperimentConfig:
    """
    Build a fully-populated experiment config with a generated run_id.
    """
    class _ConfigPayload(TypedDict):
        seed: int
        split_version: str
        xp_name: str
        train_docs: int
        sampling_mode: str
        decoder_mode: str
        boundary_head: bool
        boundary_loss_weight: float
        token_loss_mode: str
        token_loss_weight: float
        crf_loss_weight: float
        label_smoothing: float
        preserve_case: bool
        model_name: str
        batch_size: int
        train_subsample_window: int
        val_window: int
        val_stride: int
        max_epochs: int
        learning_rate: float
        weight_decay: float
        warmup_steps_pct: float
        data_path: str
        git_commit: str

    normalized_sampling_mode = _normalize_sampling_mode(sampling_mode)
    normalized_decoder_mode = _normalize_decoder_mode(decoder_mode)
    normalized_token_loss_mode = _normalize_token_loss_mode(token_loss_mode)
    normalized_boundary_head = _normalize_boundary_head(boundary_head)
    _validate_loss_configuration(
        decoder_mode=normalized_decoder_mode,
        boundary_head=normalized_boundary_head,
        boundary_loss_weight=boundary_loss_weight,
        token_loss_mode=normalized_token_loss_mode,
        token_loss_weight=token_loss_weight,
        crf_loss_weight=crf_loss_weight,
        label_smoothing=label_smoothing,
    )
    payload: _ConfigPayload = {
        "seed": seed,
        "split_version": split_version,
        "xp_name": xp_name,
        "train_docs": train_docs,
        "sampling_mode": normalized_sampling_mode,
        "decoder_mode": normalized_decoder_mode,
        "boundary_head": normalized_boundary_head,
        "boundary_loss_weight": boundary_loss_weight,
        "token_loss_mode": normalized_token_loss_mode,
        "token_loss_weight": token_loss_weight,
        "crf_loss_weight": crf_loss_weight,
        "label_smoothing": label_smoothing,
        "preserve_case": bool(preserve_case),
        "model_name": model_name,
        "batch_size": batch_size,
        "train_subsample_window": train_subsample_window,
        "val_window": val_window,
        "val_stride": val_stride,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps_pct": warmup_steps_pct,
        "data_path": data_path,
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
    _ = parser.add_argument("--xp-name", type=str, default=DEFAULT_XP_NAME)
    _ = parser.add_argument(
        "--sampling-mode",
        type=str,
        choices=["boundary_mix"],
        default=DEFAULT_SAMPLING_MODE,
    )
    _ = parser.add_argument(
        "--decoder-mode",
        type=str,
        choices=["independent", "crf"],
        default=DEFAULT_DECODER_MODE,
    )
    _ = parser.add_argument(
        "--boundary-head",
        action="store_true",
        default=DEFAULT_BOUNDARY_HEAD,
    )
    _ = parser.add_argument(
        "--boundary-loss-weight",
        type=float,
        default=DEFAULT_BOUNDARY_LOSS_WEIGHT,
    )
    _ = parser.add_argument(
        "--token-loss-mode",
        type=str,
        choices=["focal", "ce"],
        default=DEFAULT_TOKEN_LOSS_MODE,
    )
    _ = parser.add_argument(
        "--token-loss-weight",
        type=float,
        default=DEFAULT_TOKEN_LOSS_WEIGHT,
    )
    _ = parser.add_argument(
        "--crf-loss-weight",
        type=float,
        default=DEFAULT_CRF_LOSS_WEIGHT,
    )
    _ = parser.add_argument(
        "--label-smoothing",
        type=float,
        default=DEFAULT_LABEL_SMOOTHING,
    )
    _ = parser.add_argument(
        "--preserve-case",
        action="store_true",
        default=DEFAULT_PRESERVE_CASE,
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
    _ = parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    _ = parser.add_argument(
        "--data-csv",
        dest="data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    return build_config(
        train_docs=cast(int, args.train_docs),
        xp_name=cast(str, args.xp_name),
        sampling_mode=cast(str, args.sampling_mode),
        decoder_mode=cast(str, args.decoder_mode),
        boundary_head=cast(bool, args.boundary_head),
        boundary_loss_weight=cast(float, args.boundary_loss_weight),
        token_loss_mode=cast(str, args.token_loss_mode),
        token_loss_weight=cast(float, args.token_loss_weight),
        crf_loss_weight=cast(float, args.crf_loss_weight),
        label_smoothing=cast(float, args.label_smoothing),
        preserve_case=cast(bool, args.preserve_case),
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
        data_path=cast(str, args.data_path),
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
        raw: object = _yaml_safe_load(f)
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


def load_frozen_experiment_config(
    path: str | None = None,
) -> FrozenExperimentConfig:
    """
    Load the fixed hyperparameter recipe used for architecture-grid experiments.
    """
    resolved = path or str(FROZEN_EXPERIMENT_CONFIG_PATH)
    if not os.path.exists(resolved):
        raise RuntimeError(
            f"frozen_experiment_config.yaml not found at {resolved}. "
            + "Create it before running the architecture grid."
        )
    with open(resolved, "r", encoding="utf-8") as f:
        raw: object = _yaml_safe_load(f)
    if not isinstance(raw, dict):
        raise RuntimeError("frozen_experiment_config.yaml must be a YAML mapping.")
    data = cast(dict[str, object], raw)

    required = {
        "batch_size",
        "learning_rate",
        "train_subsample_window",
        "val_window",
        "val_stride",
        "weight_decay",
        "warmup_steps_pct",
        "max_epochs",
        "model_name",
    }
    missing = required - set(data)
    if missing:
        raise RuntimeError(
            f"frozen_experiment_config.yaml missing required keys: {sorted(missing)}"
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

    payload: FrozenExperimentConfig = {
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
        "model_name": _require_str(data["model_name"], "model_name"),
    }
    return payload
