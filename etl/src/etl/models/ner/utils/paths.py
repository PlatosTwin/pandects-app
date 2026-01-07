"""
Shared path helpers for NER experiments.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_repo_root() -> Path:
    """
    Resolve the NER root that contains the configs/data/eval/log/model layout.
    """
    env_root = os.environ.get("PANDECTS_REPO_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "ner.py").is_file() and (parent / "utils").is_dir():
            return parent
        if (parent / "configs").is_dir() and (parent / "data").is_dir():
            return parent
    raise RuntimeError(
        "Unable to resolve NER root; set PANDECTS_REPO_ROOT to the NER root."
    )


def get_git_root() -> Path | None:
    """
    Resolve the git root, if available.
    """
    repo_root = get_repo_root()
    for parent in [repo_root, *repo_root.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def get_job_id() -> str:
    """
    Return the current job ID, or 'local' for non-SLURM runs.
    """
    return os.environ.get("SLURM_JOB_ID", "local")


REPO_ROOT = get_repo_root()
CONFIG_NER_DIR = REPO_ROOT / "configs"
DATA_NER_DIR = REPO_ROOT / "data"
EVAL_NER_DIR = REPO_ROOT / "eval_metrics"
LOG_NER_DIR = REPO_ROOT / "logs"
MODEL_DIR = REPO_ROOT / "model_files"
