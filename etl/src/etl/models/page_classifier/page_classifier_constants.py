"""
Constants for the page classifier model.
"""

from pathlib import Path

MODEL_FILES_DIR = Path(__file__).parent / "model_files"

CLASSIFIER_LABEL_LIST = ["front_matter", "toc", "body", "sig", "back_matter"]
CLASSIFIER_XGB_PATH = str(MODEL_FILES_DIR / "xgb-classifier-latest.json")
CLASSIFIER_CKPT_PATH = str(MODEL_FILES_DIR / "bilstm-classifier-latest.ckpt")
