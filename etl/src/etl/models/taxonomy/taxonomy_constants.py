"""
Constants for the taxonomy classifier.
"""

from pathlib import Path

MODEL_FILES_DIR = Path(__file__).parent / "model_files"

TAXONOMY_CKPT_PATH = str(MODEL_FILES_DIR / "dev-taxonomy-model.ckpt")
TAXONOMY_VECTORIZER_PATH = str(MODEL_FILES_DIR / "taxonomy-tfidf.pkl")
TAXONOMY_LABEL_LIST = [
    "other",
    "definitions",
    "representations_warranties",
    "covenants",
    "termination",
    "purchase_price_adjustments",
]
