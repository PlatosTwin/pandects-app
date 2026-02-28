"""
Constants for the page classifier model.
"""

from pathlib import Path

MODEL_FILES_DIR = Path(__file__).parent / "model_files"

CLASSIFIER_LABEL_LIST = ["front_matter", "toc", "body", "sig", "back_matter"]
CLASSIFIER_XGB_PATH = str(MODEL_FILES_DIR / "xgb-classifier-latest.json")
CLASSIFIER_XGB_TRAIN_PATH = str(MODEL_FILES_DIR / "xgb-classifier-train.json")
CLASSIFIER_XGB_MOE_BASE_PATH = str(MODEL_FILES_DIR / "xgb-moe-base-latest.json")
CLASSIFIER_XGB_MOE_BASE_TRAIN_PATH = str(MODEL_FILES_DIR / "xgb-moe-base-train.json")
CLASSIFIER_XGB_MOE_TAIL_PATH = str(MODEL_FILES_DIR / "xgb-moe-tail-latest.json")
CLASSIFIER_XGB_MOE_TAIL_TRAIN_PATH = str(MODEL_FILES_DIR / "xgb-moe-tail-train.json")
CLASSIFIER_XGB_MOE_ROUTER_PATH = str(MODEL_FILES_DIR / "xgb-moe-router-latest.json")
CLASSIFIER_XGB_MOE_ROUTER_TRAIN_PATH = str(MODEL_FILES_DIR / "xgb-moe-router-train.json")
CLASSIFIER_CKPT_PATH = str(MODEL_FILES_DIR / "bilstm-classifier-latest.ckpt")
