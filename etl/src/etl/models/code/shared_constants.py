"""
Shared constants for NER and Classifier models.

This module contains configuration constants used across the NER and Classifier
model training and inference pipelines.
"""

import os
from pathlib import Path

# Model file paths - using relative paths for better portability
MODEL_FILES_DIR = Path(__file__).parent.parent / "model_files"

# NER model configuration
NER_CKPT_PATH = str(MODEL_FILES_DIR / "dev-ner-model.ckpt")
NER_LABEL_LIST = [
    "O",  # outside any entity
    "B-SECTION",  # begin a SECTION span
    "I-SECTION",  # inside a SECTION span
    "E-SECTION",  # end of a SECTION span
    "S-SECTION",  # single SECTION span
    "B-ARTICLE",  # begin an ARTICLE span
    "I-ARTICLE",  # inside an ARTICLE span
    "E-ARTICLE",  # end of a ARTICLE span
    "S-ARTICLE",  # single ARTICLE span
    "B-PAGE",  # begin a PAGE span
    "I-PAGE",  # inside a PAGE span
    "E-PAGE",  # end of a PAGE span
    "S-PAGE",  # single PAGE span
]

# Classifier model configuration
CLASSIFIER_LABEL_LIST = ["front_matter", "toc", "body", "sig", "back_matter"]
CLASSIFIER_XGB_PATH = str(MODEL_FILES_DIR / "xgb_multi_class.json")
CLASSIFIER_CKPT_PATH = str(MODEL_FILES_DIR / "dev-classifier-model.ckpt")

SPECIAL_TOKENS_TO_ADD = ["SECTION", "ARTICLE"]