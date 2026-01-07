"""
NER-specific constants.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .utils.paths import MODEL_DIR
else:
    try:
        from .utils.paths import MODEL_DIR
    except ImportError:  # pragma: no cover - supports running as a script
        from utils.paths import MODEL_DIR


MODEL_FILES_DIR = Path(MODEL_DIR)
NER_CKPT_PATH = str(MODEL_FILES_DIR / "ner-model-latest.ckpt")
NER_LABEL_LIST = [
    "O",
    "B-SECTION",
    "I-SECTION",
    "E-SECTION",
    "B-ARTICLE",
    "I-ARTICLE",
    "E-ARTICLE",
    "B-PAGE",
    "I-PAGE",
    "E-PAGE",
    "S-PAGE",
]
SPECIAL_TOKENS_TO_ADD: list[str] = []
