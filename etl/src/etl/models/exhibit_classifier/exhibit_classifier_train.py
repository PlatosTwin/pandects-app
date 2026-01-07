"""Train the exhibit classifier on M&A agreement text data."""

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,  # pyright: ignore[reportUnknownVariableType]
    average_precision_score,  # pyright: ignore[reportUnknownVariableType]
    confusion_matrix,  # pyright: ignore[reportUnknownVariableType]
    precision_recall_fscore_support,  # pyright: ignore[reportUnknownVariableType]
    roc_auc_score,  # pyright: ignore[reportUnknownVariableType]
)
from sklearn.model_selection import train_test_split  # pyright: ignore[reportUnknownVariableType]

from etl.models.exhibit_classifier.exhibit_classifier import (
    ExhibitClassifier,
    load_training_data,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "exhibit-data.parquet"
DEFAULT_MODEL_PATH = BASE_DIR / "model_files" / "exhibit-classifier.joblib"
EVAL_METRICS_DIR = BASE_DIR / "eval_metrics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the exhibit classifier on agreement text data."
    )
    _ = parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help="Path to training data (CSV/Parquet/TXT with a 'text' column).",
    )
    _ = parser.add_argument(
        "--output-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to save the trained classifier.",
    )
    _ = parser.add_argument(
        "--method",
        choices=("isolation_forest", "one_class_svm"),
        default="isolation_forest",
        help="One-class classifier method to use.",
    )
    _ = parser.add_argument(
        "--contamination",
        type=float,
        default=0.1,
        help="Expected proportion of outliers in training data.",
    )
    _ = parser.add_argument(
        "--max-features",
        type=int,
        default=1000,
        help="Maximum number of TF-IDF features.",
    )
    _ = parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    _ = parser.add_argument(
        "--eval-split",
        type=float,
        default=0.2,
        help="Holdout fraction for supervised evaluation.",
    )
    _ = parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for supervised metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(cast(str, args.data_path))
    output_path = Path(cast(str, args.output_path))

    texts, labels = load_training_data(str(data_path))

    if labels and any(label == 0 for label in labels):
        y = np.array(labels, dtype=int)
        split = cast(
            tuple[list[str], list[str], np.ndarray, np.ndarray],
            cast(
                object,
                train_test_split(
                    texts,
                    y,
                    test_size=cast(float, args.eval_split),
                    random_state=cast(int, args.random_state),
                    stratify=y,
                ),
            ),
        )
        train_texts, test_texts, y_train, y_test = split
        classifier = ExhibitClassifier(
            method=cast(str, args.method),
            contamination=cast(float, args.contamination),
            max_features=cast(int, args.max_features),
            random_state=cast(int, args.random_state),
        )
        y_train_list = cast(list[int], y_train.tolist())
        _ = classifier.fit(train_texts, labels=y_train_list)

        probs = np.array([classifier.predict_proba(t) for t in test_texts])
        preds = (probs >= cast(float, args.threshold)).astype(int)
        
        # Calculate overall metrics
        accuracy = float(accuracy_score(y_test, preds))
        precision, recall, f1, _ = cast(
            tuple[float, float, float, object],
            precision_recall_fscore_support(
                y_test,
                preds,
                average="binary",
                zero_division=cast(str, cast(object, 0)),
            ),
        )
        try:
            roc_auc = float(cast(float, roc_auc_score(y_test, probs)))
        except ValueError:
            roc_auc = float("nan")
        avg_precision = float(cast(float, average_precision_score(y_test, probs)))
        pos_rate = float(np.mean(y_test))
        threshold = cast(float, args.threshold)
        
        # Calculate per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = cast(
            tuple[np.ndarray, np.ndarray, np.ndarray, object],
            precision_recall_fscore_support(
                y_test,
                preds,
                average=None,
                zero_division=cast(str, cast(object, 0)),
            ),
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, preds)
        
        # Calculate per-class accuracy from confusion matrix
        # For each class, accuracy = correct predictions for that class / total samples of that class
        # This is equivalent to recall, but we'll call it accuracy as requested
        class_0_total = int(cm[0, :].sum()) if cm.shape[0] > 0 else 0
        class_1_total = int(cm[1, :].sum()) if cm.shape[0] > 1 else 0
        accuracy_class_0 = float(cm[0, 0] / class_0_total) if class_0_total > 0 else 0.0
        accuracy_class_1 = float(cm[1, 1] / class_1_total) if class_1_total > 0 else 0.0
        
        # Print metrics
        message = f"Holdout metrics (pos_rate={pos_rate:.3f}, threshold={threshold:.2f}): accuracy={accuracy:.3f} precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} roc_auc={roc_auc:.3f} avg_precision={avg_precision:.3f}"
        print(message)
        
        # Save metrics to YAML file
        EVAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_file = EVAL_METRICS_DIR / "exhibit_classifier_test_metrics.yaml"
        
        metrics = {
            "overall": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "roc_auc": float(roc_auc),
            },
            "per_class": {
                "class_0": {
                    "accuracy": accuracy_class_0,
                    "precision": float(precision_per_class[0]),
                    "recall": float(recall_per_class[0]),
                    "f1": float(f1_per_class[0]),
                },
                "class_1": {
                    "accuracy": accuracy_class_1,
                    "precision": float(precision_per_class[1]),
                    "recall": float(recall_per_class[1]),
                    "f1": float(f1_per_class[1]),
                },
            },
            "confusion_matrix": cm.tolist(),
        }
        
        with open(metrics_file, "w") as f:
            yaml.dump(metrics, f, default_flow_style=False, sort_keys=False)
        
        print(f"Saved evaluation metrics to {metrics_file}")
    else:
        classifier = ExhibitClassifier(
            method=cast(str, args.method),
            contamination=cast(float, args.contamination),
            max_features=cast(int, args.max_features),
            random_state=cast(int, args.random_state),
        )
        _ = classifier.fit(texts, labels=labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ = classifier.save(output_path)


if __name__ == "__main__":
    main()
