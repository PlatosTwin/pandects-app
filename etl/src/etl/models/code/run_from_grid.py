"""
Run a single NER experiment row from grid.csv.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Callable, NamedTuple, TypedDict, cast


CODE_DIR = Path(__file__).resolve().parent


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_config_mod = _load_module("config", CODE_DIR / "config.py")
_ner_mod = _load_module("ner", CODE_DIR / "ner.py")

REPO_ROOT = cast(Path, _config_mod.REPO_ROOT)
build_config = cast(Callable[..., object], _config_mod.build_config)
load_optuna_best_config = cast(Callable[..., dict[str, object]], _config_mod.load_optuna_best_config)
run_experiment = cast(Callable[..., object], _ner_mod.run_experiment)


DEFAULT_SPLIT_VERSION = "v1_article_stratified"
DEFAULT_SEED = 42


class _Args(NamedTuple):
    row_id: int | None
    split_version: str
    seed: int
    git_commit: str | None


class _FrozenConfig(TypedDict):
    model_name: str
    batch_size: int
    train_subsample_window: int
    val_window: int
    val_stride: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_steps_pct: float


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


def _load_grid_row(row_id: int, grid_path: Path) -> dict[str, str]:
    """
    Load a single row from grid.csv by row_id.
    """
    with grid_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["row_id"]) == row_id:
                return row
    raise ValueError(f"row_id {row_id} not found in {grid_path}")


def main() -> None:
    """
    CLI entrypoint for running a grid row.
    """
    parser = argparse.ArgumentParser(description="Run a grid row experiment")
    _ = parser.add_argument("--row-id", type=int, default=None)
    _ = parser.add_argument("--split-version", type=str, default=DEFAULT_SPLIT_VERSION)
    _ = parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    _ = parser.add_argument("--git-commit", type=str, default=None)
    parsed = parser.parse_args()
    args = _Args(
        row_id=cast(int | None, getattr(parsed, "row_id")),
        split_version=cast(str, getattr(parsed, "split_version")),
        seed=cast(int, getattr(parsed, "seed")),
        git_commit=cast(str | None, getattr(parsed, "git_commit")),
    )

    row_id: int | None = args.row_id
    if row_id is None:
        env_row = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_row is None:
            raise RuntimeError("row-id is required when SLURM_ARRAY_TASK_ID is not set.")
        row_id = int(env_row)

    grid_path: Path = CODE_DIR / ".." / "data" / "grid.csv"
    grid_path = grid_path.resolve()
    row = _load_grid_row(row_id, grid_path)
    frozen_raw = load_optuna_best_config()
    frozen: _FrozenConfig = {
        "model_name": _require_str(frozen_raw["model_name"], "model_name"),
        "batch_size": _require_int(frozen_raw["batch_size"], "batch_size"),
        "train_subsample_window": _require_int(
            frozen_raw["train_subsample_window"], "train_subsample_window"
        ),
        "val_window": _require_int(frozen_raw["val_window"], "val_window"),
        "val_stride": _require_int(frozen_raw["val_stride"], "val_stride"),
        "max_epochs": _require_int(frozen_raw["max_epochs"], "max_epochs"),
        "learning_rate": _require_float(frozen_raw["learning_rate"], "learning_rate"),
        "weight_decay": _require_float(frozen_raw["weight_decay"], "weight_decay"),
        "warmup_steps_pct": _require_float(
            frozen_raw["warmup_steps_pct"], "warmup_steps_pct"
        ),
    }

    config = build_config(
        train_docs=int(row["train_docs"]),
        article_weight=float(row["article_weight"]),
        gating_mode=row["gating_mode"],
        split_version=args.split_version,
        seed=args.seed,
        git_commit=args.git_commit,
        model_name=frozen["model_name"],
        batch_size=frozen["batch_size"],
        train_subsample_window=frozen["train_subsample_window"],
        val_window=frozen["val_window"],
        val_stride=frozen["val_stride"],
        max_epochs=frozen["max_epochs"],
        learning_rate=frozen["learning_rate"],
        weight_decay=frozen["weight_decay"],
        warmup_steps_pct=frozen["warmup_steps_pct"],
    )
    _ = run_experiment(config)


if __name__ == "__main__":
    main()
