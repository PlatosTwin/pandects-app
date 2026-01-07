"""
Run a single NER experiment row from grid.csv.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Callable, NamedTuple, cast


CODE_DIR = Path(__file__).resolve().parent


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ner_mod = _load_module("ner", CODE_DIR / "ner.py")

run_grid_row = cast(Callable[..., object], _ner_mod.run_grid_row)


DEFAULT_SPLIT_VERSION = "v1_article_stratified"
DEFAULT_SEED = 42


class _Args(NamedTuple):
    row_id: int | None
    split_version: str
    seed: int
    git_commit: str | None


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

    _ = run_grid_row(
        row_id=row_id,
        split_version=args.split_version,
        seed=args.seed,
        git_commit=args.git_commit,
        eval_split="val",
    )


if __name__ == "__main__":
    main()
