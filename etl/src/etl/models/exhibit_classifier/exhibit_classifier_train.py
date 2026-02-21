"""Train the exhibit classifier on M&A agreement text data."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import argparse
import json
import statistics
from pathlib import Path
from typing import cast

import numpy as np
import yaml
from optuna import Trial, TrialPruned, create_study
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from etl.models.exhibit_classifier.exhibit_classifier import (
    ExhibitClassifier,
    load_training_data,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "exhibit-data.parquet"
DEFAULT_MODEL_PATH = BASE_DIR / "model_files" / "exhibit-classifier.joblib"
DEFAULT_SPLIT_PATH = BASE_DIR / "data" / "exhibit-splits.json"
EVAL_METRICS_DIR = BASE_DIR / "eval_metrics"
DEFAULT_OPTUNA_PATH = EVAL_METRICS_DIR / "exhibit_classifier_optuna_best.yaml"
DEFAULT_ERROR_URLS_PATH = EVAL_METRICS_DIR / "exhibit_classifier_test_error_urls.yaml"
TUNE_CV_FOLDS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the exhibit classifier on agreement text data."
    )
    _ = parser.add_argument(
        "--mode",
        choices=("train", "eval", "tune"),
        default="train",
        help="train: fit classifier; eval: evaluate existing checkpoint only.",
    )
    _ = parser.add_argument(
        "--min-recall",
        type=float,
        default=0.97,
        help="Minimum class_1 recall when tuning the decision threshold.",
    )
    _ = parser.add_argument(
        "--optuna-trials",
        type=int,
        default=10,
        help="Number of Optuna trials to run in tune mode.",
    )
    _ = parser.add_argument(
        "--tune-vectorizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to tune hashing vectorizer params in tune mode.",
    )
    return parser.parse_args()


def _require_supervised_labels(labels: list[int] | None) -> np.ndarray:
    if labels is None:
        raise RuntimeError("Evaluation requires labels; none found in dataset.")
    if not any(label == 0 for label in labels):
        raise RuntimeError("Evaluation requires both positive and negative labels.")
    return np.array(labels, dtype=int)


def _compute_and_log_metrics(
    *,
    y_true: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    metrics_file: Path,
) -> None:
    accuracy = float(accuracy_score(y_true, preds))
    precision, recall, f1, _ = cast(
        tuple[float, float, float, object],
        precision_recall_fscore_support(
            y_true,
            preds,
            average="binary",
            zero_division=cast(str, cast(object, 0)),
        ),
    )
    try:
        roc_auc = float(cast(float, roc_auc_score(y_true, probs)))
    except ValueError:
        roc_auc = float("nan")
    avg_precision = float(cast(float, average_precision_score(y_true, probs)))
    pos_rate = float(np.mean(y_true))

    precision_per_class, recall_per_class, f1_per_class, _ = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray, object],
        precision_recall_fscore_support(
            y_true,
            preds,
            average=None,
            zero_division=cast(str, cast(object, 0)),
        ),
    )
    cm = confusion_matrix(y_true, preds)

    class_0_total = int(cast(int, cm[0, :].sum())) if cm.shape[0] > 0 else 0
    class_1_total = int(cast(int, cm[1, :].sum())) if cm.shape[0] > 1 else 0
    accuracy_class_0 = float(cast(float, cm[0, 0] / class_0_total)) if class_0_total > 0 else 0.0
    accuracy_class_1 = float(cast(float, cm[1, 1] / class_1_total)) if class_1_total > 0 else 0.0

    message = (
        f"Holdout metrics (pos_rate={pos_rate:.3f}, threshold={threshold:.2f}): "
        f"accuracy={accuracy:.3f} precision={precision:.3f} recall={recall:.3f} "
        f"f1={f1:.3f} roc_auc={roc_auc:.3f} avg_precision={avg_precision:.3f}"
    )
    print(message)

    metrics = {
        "threshold": float(threshold),
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
                "precision": float(cast(float, precision_per_class[0])),
                "recall": float(cast(float, recall_per_class[0])),
                "f1": float(cast(float, f1_per_class[0])),
            },
            "class_1": {
                "accuracy": accuracy_class_1,
                "precision": float(cast(float, precision_per_class[1])),
                "recall": float(cast(float, recall_per_class[1])),
                "f1": float(cast(float, f1_per_class[1])),
            },
        },
        "confusion_matrix": cm.tolist(),
    }

    with open(metrics_file, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False, sort_keys=False)

    print(f"Saved evaluation metrics to {metrics_file}")


def _load_or_create_split(
    *,
    urls: list[str],
    labels: np.ndarray,
    split_path: Path,
    val_split: float,
    test_split: float,
    random_state: int,
) -> tuple[list[int], list[int], list[int]]:
    if val_split <= 0 or val_split >= 1:
        raise ValueError("val_split must be in (0, 1).")
    if test_split <= 0 or test_split >= 1:
        raise ValueError("test_split must be in (0, 1).")
    if val_split + test_split >= 1:
        raise ValueError("val_split + test_split must be less than 1.")

    url_to_indices: dict[str, list[int]] = {}
    for idx, url in enumerate(urls):
        url_to_indices.setdefault(url, []).append(idx)

    duplicate_urls = [url for url, idxs in url_to_indices.items() if len(idxs) > 1]
    if duplicate_urls:
        duplicate_urls_sorted = sorted(duplicate_urls)
        duplicate_list_text = "\n".join(duplicate_urls_sorted)
        raise RuntimeError(
            "Duplicate URLs found in exhibit dataset; remove duplicates before splitting.\n"
            + f"Found {len(duplicate_urls_sorted)} duplicate URLs:\n{duplicate_list_text}"
        )

    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as f:
            manifest_raw = cast(object, json.load(f))
        if not isinstance(manifest_raw, dict):
            raise ValueError("Split manifest must be a mapping.")
        manifest = cast(dict[str, object], manifest_raw)
        if "train" not in manifest or "val" not in manifest or "test" not in manifest:
            raise ValueError("Split manifest missing required keys: train/val/test.")
        train_raw = manifest["train"]
        val_raw = manifest["val"]
        test_raw = manifest["test"]
        if not isinstance(train_raw, list) or not isinstance(val_raw, list) or not isinstance(test_raw, list):
            raise ValueError("Split manifest train/val/test must be lists.")
        train_items = cast(list[object], train_raw)
        val_items = cast(list[object], val_raw)
        test_items = cast(list[object], test_raw)
        train_urls = [str(url) for url in train_items]
        val_urls = [str(url) for url in val_items]
        test_urls = [str(url) for url in test_items]
        train_url_set = set(train_urls)
        val_url_set = set(val_urls)
        test_url_set = set(test_urls)
        if train_url_set & val_url_set:
            raise RuntimeError("Split manifest has overlap between train and val URLs.")
        if train_url_set & test_url_set:
            raise RuntimeError("Split manifest has overlap between train and test URLs.")
        if val_url_set & test_url_set:
            raise RuntimeError("Split manifest has overlap between val and test URLs.")
    else:
        indices = list(range(len(urls)))
        holdout_size = val_split + test_split
        train_idx, holdout_idx = cast(
            tuple[list[int], list[int]],
            cast(
                object,
                train_test_split(
                    indices,
                    test_size=holdout_size,
                    random_state=random_state,
                    stratify=labels,
                ),
            ),
        )
        holdout_labels = labels[holdout_idx]
        test_fraction = test_split / holdout_size
        val_idx, test_idx = cast(
            tuple[list[int], list[int]],
            cast(
                object,
                train_test_split(
                    holdout_idx,
                    test_size=test_fraction,
                    random_state=random_state,
                    stratify=holdout_labels,
                ),
            ),
        )
        train_urls = [urls[i] for i in train_idx]
        val_urls = [urls[i] for i in val_idx]
        test_urls = [urls[i] for i in test_idx]
        split_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "train": train_urls,
            "val": val_urls,
            "test": test_urls,
            "meta": {
                "val_split": float(val_split),
                "test_split": float(test_split),
                "seed": int(random_state),
            },
        }
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=False)
        print(f"[split] wrote exhibit split manifest to {split_path}")

    def _indices_for(target_urls: list[str]) -> list[int]:
        missing = [url for url in target_urls if url not in url_to_indices]
        if missing:
            preview = ", ".join(missing[:10])
            extra = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
            raise RuntimeError(
                "Split manifest references URLs not present in dataset: "
                + f"{preview}{extra}"
            )
        return [url_to_indices[url][0] for url in target_urls]

    train_indices = _indices_for(train_urls)
    val_indices = _indices_for(val_urls)
    test_indices = _indices_for(test_urls)
    if not train_indices or not val_indices or not test_indices:
        raise RuntimeError("Split resulted in empty train, val, or test set.")
    train_idx_set = set(train_indices)
    val_idx_set = set(val_indices)
    test_idx_set = set(test_indices)
    if train_idx_set & val_idx_set:
        raise RuntimeError("Split mapping produced overlapping train and val indices.")
    if train_idx_set & test_idx_set:
        raise RuntimeError("Split mapping produced overlapping train and test indices.")
    if val_idx_set & test_idx_set:
        raise RuntimeError("Split mapping produced overlapping val and test indices.")

    return train_indices, val_indices, test_indices


def _select_threshold(
    *,
    y_true: np.ndarray,
    probs: np.ndarray,
    min_recall: float,
    threshold_step: float,
    forced_negative_mask: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    if threshold_step <= 0 or threshold_step > 1:
        raise ValueError("threshold_step must be in (0, 1].")
    thresholds = np.arange(0.0, 1.0 + threshold_step, threshold_step)
    best: tuple[float, dict[str, float]] | None = None

    for threshold in thresholds:
        preds = _preds_from_probs(
            probs=probs,
            threshold=float(threshold),
            forced_negative_mask=forced_negative_mask,
        )
        cm = np.asarray(confusion_matrix(y_true, preds), dtype=int)
        if cm.shape != (2, 2):
            continue
        tn = int(cast(int, cm[0, 0]))
        fp = int(cast(int, cm[0, 1]))
        fn = int(cast(int, cm[1, 0]))
        tp = int(cast(int, cm[1, 1]))
        recall = tp / max(tp + fn, 1)
        if recall < min_recall:
            continue
        precision = tp / max(tp + fp, 1)
        class_0_recall = tn / max(tn + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        metrics = {
            "recall": float(recall),
            "precision": float(precision),
            "class_0_recall": float(class_0_recall),
            "f1": float(f1),
            "false_positives": float(fp),
        }
        if best is None:
            best = (float(threshold), metrics)
            continue
        best_metrics = best[1]
        if class_0_recall > best_metrics["class_0_recall"]:
            best = (float(threshold), metrics)
            continue
        if class_0_recall == best_metrics["class_0_recall"]:
            if fp < best_metrics["false_positives"]:
                best = (float(threshold), metrics)
                continue
            if fp == best_metrics["false_positives"] and f1 > best_metrics["f1"]:
                best = (float(threshold), metrics)
                continue

    if best is None:
        raise RuntimeError(
            f"No threshold met min_recall={min_recall:.3f}. Consider lowering the recall floor."
        )
    return best


def _compute_threshold_metrics(
    *,
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    forced_negative_mask: np.ndarray | None = None,
) -> dict[str, float]:
    preds = _preds_from_probs(
        probs=probs,
        threshold=threshold,
        forced_negative_mask=forced_negative_mask,
    )
    cm = np.asarray(confusion_matrix(y_true, preds), dtype=int)
    if cm.shape != (2, 2):
        raise RuntimeError("Threshold metrics require binary labels with both classes present.")
    tn = int(cast(int, cm[0, 0]))
    fp = int(cast(int, cm[0, 1]))
    fn = int(cast(int, cm[1, 0]))
    tp = int(cast(int, cm[1, 1]))
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    class_0_recall = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": float(precision),
        "class_0_recall": float(class_0_recall),
        "recall": float(recall),
        "f1": float(f1),
        "false_positives": float(fp),
    }


def _hard_negative_mask(
    *,
    texts: list[str],
    min_chars_hard_negative: int,
) -> np.ndarray:
    return np.array(
        [len(str(text)) < min_chars_hard_negative for text in texts], dtype=bool
    )


def _preds_from_probs(
    *,
    probs: np.ndarray,
    threshold: float,
    forced_negative_mask: np.ndarray | None = None,
) -> np.ndarray:
    preds = (probs >= threshold).astype(int)
    if forced_negative_mask is None:
        return preds
    if forced_negative_mask.shape[0] != preds.shape[0]:
        raise ValueError("forced_negative_mask must align with probs length.")
    preds[forced_negative_mask] = 0
    return preds


def _drop_short_positive_examples(
    *,
    texts: list[str],
    labels: np.ndarray,
    urls: list[str] | None,
    min_chars_hard_negative: int,
) -> tuple[list[str], np.ndarray, list[str] | None, int]:
    if labels.shape[0] != len(texts):
        raise ValueError("labels length must match texts length.")
    if urls is not None and len(urls) != len(texts):
        raise ValueError("urls length must match texts length.")

    label_list = cast(list[int], labels.tolist())
    keep_mask_list = [
        not (label == 1 and len(str(text)) < min_chars_hard_negative)
        for text, label in zip(texts, label_list)
    ]
    keep_mask = np.array(keep_mask_list, dtype=bool)
    dropped_count = int(np.count_nonzero(~keep_mask))
    if dropped_count == 0:
        return texts, labels, urls, 0

    filtered_texts = [
        text for text, keep in zip(texts, keep_mask_list) if keep
    ]
    filtered_labels = labels[keep_mask]
    filtered_urls = (
        [url for url, keep in zip(urls, keep_mask_list) if keep]
        if urls is not None
        else None
    )
    return filtered_texts, filtered_labels, filtered_urls, dropped_count


def _ngram_from_key(key: str) -> tuple[int, int]:
    if key == "1_2":
        return (1, 2)
    if key == "1_3":
        return (1, 3)
    if key == "3_5":
        return (3, 5)
    if key == "4_6":
        return (4, 6)
    raise ValueError(f"Unknown ngram key: {key}")


def _class_weight_from_params(
    params: dict[str, object],
) -> str | dict[int, float] | None:
    mode = cast(str, params["class_weight_mode"])
    if mode == "none":
        return None
    if mode == "balanced":
        return "balanced"
    if mode == "negative_upweight":
        return {0: float(cast(float, params["negative_class_weight"])), 1: 1.0}
    raise ValueError(f"Unknown class_weight_mode: {mode}")


def _run_optuna_search(
    *,
    tune_texts: list[str],
    y_tune: np.ndarray,
    min_recall: float,
    threshold_step: float,
    random_state: int,
    min_chars_hard_negative: int,
    num_trials: int,
    tune_vectorizer: bool,
    cv_folds: int,
) -> tuple[dict[str, object], float, dict[str, float]]:
    fixed_logreg_max_iter = 2000
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def objective(trial: Trial) -> float:
        if tune_vectorizer:
            word_ngram_key = trial.suggest_categorical("word_ngram_range", ["1_2", "1_3"])
            char_ngram_key = trial.suggest_categorical("char_ngram_range", ["3_5", "4_6"])
            max_features = trial.suggest_int("max_features", 3000, 20000, log=True)
            char_max_features = trial.suggest_int("char_max_features", 1000, 8000, log=True)
        else:
            word_ngram_key = trial.suggest_categorical("word_ngram_range", ["1_3"])
            char_ngram_key = trial.suggest_categorical("char_ngram_range", ["3_5"])
            max_features = trial.suggest_int("max_features", 5000, 5000)
            char_max_features = trial.suggest_int("char_max_features", 3000, 3000)

        word_ngram_range = _ngram_from_key(word_ngram_key)
        char_ngram_range = _ngram_from_key(char_ngram_key)

        logreg_c = trial.suggest_float("logreg_c", 1e-2, 10.0, log=True)
        start_scan_chars = trial.suggest_categorical(
            "start_scan_chars", [1200, 2000, 3000]
        )
        class_weight_mode = trial.suggest_categorical(
            "class_weight_mode", ["none", "balanced", "negative_upweight"]
        )
        if class_weight_mode == "negative_upweight":
            negative_class_weight = trial.suggest_float(
                "negative_class_weight", 1.0, 30.0, log=True
            )
            class_weight: str | dict[int, float] | None = {
                0: float(negative_class_weight),
                1: 1.0,
            }
        elif class_weight_mode == "balanced":
            class_weight = "balanced"
        else:
            class_weight = None

        classifier = ExhibitClassifier(
            max_features=max_features,
            char_max_features=char_max_features,
            word_ngram_range=word_ngram_range,
            char_ngram_range=char_ngram_range,
            logreg_c=logreg_c,
            logreg_max_iter=fixed_logreg_max_iter,
            class_weight=class_weight,
            start_scan_chars=int(start_scan_chars),
            min_chars_hard_negative=min_chars_hard_negative,
            random_state=random_state,
        )
        fold_thresholds: list[float] = []
        fold_precisions: list[float] = []
        fold_class0_recalls: list[float] = []
        fold_recalls: list[float] = []
        fold_f1s: list[float] = []
        fold_false_positives: list[float] = []

        for fold_step, split_pair in enumerate(cv.split(tune_texts, y_tune)):
            fit_idx, fold_val_idx = cast(tuple[np.ndarray, np.ndarray], split_pair)
            fit_indices = cast(list[int], fit_idx.tolist())
            fold_val_indices = cast(list[int], fold_val_idx.tolist())
            fit_texts = [tune_texts[i] for i in fit_indices]
            fold_val_texts = [tune_texts[i] for i in fold_val_indices]
            y_fit = y_tune[fit_idx]
            y_fold_val = y_tune[fold_val_idx]
            _ = classifier.fit(fit_texts, labels=cast(list[int], y_fit.tolist()))
            probs_fold_val = np.array(classifier.predict_proba_batch(fold_val_texts))
            fold_val_forced_negative_mask = _hard_negative_mask(
                texts=fold_val_texts,
                min_chars_hard_negative=min_chars_hard_negative,
            )
            try:
                threshold, metrics = _select_threshold(
                    y_true=y_fold_val,
                    probs=probs_fold_val,
                    min_recall=min_recall,
                    threshold_step=threshold_step,
                    forced_negative_mask=fold_val_forced_negative_mask,
                )
            except RuntimeError as exc:
                raise TrialPruned(str(exc)) from exc

            fold_thresholds.append(float(threshold))
            fold_precisions.append(float(metrics["precision"]))
            fold_class0_recalls.append(float(metrics["class_0_recall"]))
            fold_recalls.append(float(metrics["recall"]))
            fold_f1s.append(float(metrics["f1"]))
            fold_false_positives.append(float(metrics["false_positives"]))

            running_mean_class0 = float(np.mean(np.array(fold_class0_recalls, dtype=float)))
            trial.report(running_mean_class0, step=fold_step)
            if trial.should_prune():
                raise TrialPruned()

        if not fold_thresholds:
            raise TrialPruned("No successful CV folds for this trial.")

        threshold_median = float(statistics.median(fold_thresholds))
        precision_mean = float(np.mean(np.array(fold_precisions, dtype=float)))
        class0_recall_mean = float(np.mean(np.array(fold_class0_recalls, dtype=float)))
        recall_mean = float(np.mean(np.array(fold_recalls, dtype=float)))
        f1_mean = float(np.mean(np.array(fold_f1s, dtype=float)))
        false_positives_mean = float(np.mean(np.array(fold_false_positives, dtype=float)))

        trial.set_user_attr("threshold", threshold_median)
        trial.set_user_attr("precision", precision_mean)
        trial.set_user_attr("class_0_recall", class0_recall_mean)
        trial.set_user_attr("recall", recall_mean)
        trial.set_user_attr("f1", f1_mean)
        trial.set_user_attr("false_positives", false_positives_mean)
        return class0_recall_mean

    no_improve_limit = 5
    best_value: float | None = None
    no_improve_count = 0

    def _stop_on_no_improve(study: Study, _trial: FrozenTrial) -> None:
        nonlocal best_value, no_improve_count
        current_best = study.best_value
        if best_value is None or current_best > best_value:
            best_value = current_best
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= no_improve_limit:
            study.stop()

    sampler = TPESampler(seed=random_state)
    pruner = MedianPruner(n_warmup_steps=5)
    study = create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(
        objective,
        n_trials=num_trials,
        gc_after_trial=True,
        callbacks=[
            lambda study, _trial: study.stop() if study.best_value >= 1.0 else None,
            _stop_on_no_improve,
        ],
    )

    best_params = cast(dict[str, object], study.best_trial.params)
    best_attrs = cast(dict[str, float], study.best_trial.user_attrs)
    best_threshold = float(best_attrs.get("threshold", 0.5))
    best_metrics = {
        "precision": float(best_attrs.get("precision", 0.0)),
        "class_0_recall": float(best_attrs.get("class_0_recall", 0.0)),
        "recall": float(best_attrs.get("recall", 0.0)),
        "f1": float(best_attrs.get("f1", 0.0)),
        "false_positives": float(best_attrs.get("false_positives", 0.0)),
    }
    return best_params, best_threshold, best_metrics


def _print_false_negatives(
    *,
    y_true: np.ndarray,
    preds: np.ndarray,
    urls: list[str],
) -> None:
    true_list = cast(list[int], y_true.tolist())
    pred_list = cast(list[int], preds.tolist())
    false_negative_urls = [
        url for url, y_val, p_val in zip(urls, true_list, pred_list) if y_val == 1 and p_val == 0
    ]
    print(f"False negatives (positive labeled as negative): {len(false_negative_urls)}")
    for url in false_negative_urls:
        print(url)


def _print_false_positives(
    *,
    y_true: np.ndarray,
    preds: np.ndarray,
    urls: list[str],
) -> None:
    true_list = cast(list[int], y_true.tolist())
    pred_list = cast(list[int], preds.tolist())
    false_positive_urls = [
        url for url, y_val, p_val in zip(urls, true_list, pred_list) if y_val == 0 and p_val == 1
    ]
    print(f"False positives (negative labeled as positive): {len(false_positive_urls)}")
    for url in false_positive_urls:
        print(url)


def _write_class_error_urls(
    *,
    y_true: np.ndarray,
    preds: np.ndarray,
    urls: list[str],
    output_file: Path,
) -> None:
    true_list = cast(list[int], y_true.tolist())
    pred_list = cast(list[int], preds.tolist())
    class_1_false_negatives = [
        url for url, y_val, p_val in zip(urls, true_list, pred_list) if y_val == 1 and p_val == 0
    ]
    class_0_false_positives = [
        url for url, y_val, p_val in zip(urls, true_list, pred_list) if y_val == 0 and p_val == 1
    ]
    payload = {
        "class_1_false_negatives": {
            "count": len(class_1_false_negatives),
            "urls": class_1_false_negatives,
        },
        "class_0_false_positives": {
            "count": len(class_0_false_positives),
            "urls": class_0_false_positives,
        },
    }
    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
    print(f"Saved class-specific error URLs to {output_file}")


def main() -> None:
    args = parse_args()
    min_recall = cast(float, args.min_recall)
    optuna_trials = cast(int, args.optuna_trials)
    data_path = DEFAULT_DATA_PATH
    output_path = DEFAULT_MODEL_PATH
    split_path = DEFAULT_SPLIT_PATH
    val_split = 0.1
    test_split = 0.2
    random_state = 42
    min_chars_hard_negative = 20000
    threshold_step = 0.01
    threshold = 0.5
    tune_threshold = True

    print(f"Loading training data from {data_path}...")
    texts, labels, urls = load_training_data(str(data_path))
    print(f"Loaded {len(texts)} examples.")

    mode = cast(str, args.mode)
    y = _require_supervised_labels(labels)
    texts, y, urls, dropped_short_positives = _drop_short_positive_examples(
        texts=texts,
        labels=y,
        urls=urls,
        min_chars_hard_negative=min_chars_hard_negative,
    )
    if dropped_short_positives > 0:
        print(
            f"Dropped {dropped_short_positives} positive examples shorter than {min_chars_hard_negative} chars to align training labels with hard-negative inference."
        )
        if split_path.exists():
            split_path.unlink()
            print(
                f"Removed stale split manifest at {split_path} after dataset filtering; a new split will be created."
            )
    y_list = cast(list[int], y.tolist())
    has_negative = 0 in y_list
    has_positive = 1 in y_list
    if not has_negative or not has_positive:
        raise RuntimeError(
            "Dataset has only one class after dropping short positive examples."
        )
    if mode == "eval":
        if urls is None:
            raise RuntimeError("Evaluation requires a 'url' column in the dataset.")
        train_idx, val_idx, test_idx = _load_or_create_split(
            urls=urls,
            labels=y,
            split_path=split_path,
            val_split=val_split,
            test_split=test_split,
            random_state=random_state,
        )
        print(f"Loading classifier from {output_path}...")
        classifier = ExhibitClassifier.load(output_path)
        if classifier.min_chars_hard_negative != min_chars_hard_negative:
            raise RuntimeError(
                f"Loaded model min_chars_hard_negative={classifier.min_chars_hard_negative} does not match hardcoded training/eval value {min_chars_hard_negative}."
            )
        print(f"Evaluating on {len(texts)} examples...")
        probs = np.array(classifier.predict_proba_batch(texts))
        forced_negative_mask = _hard_negative_mask(
            texts=texts,
            min_chars_hard_negative=min_chars_hard_negative,
        )
        val_probs = probs[val_idx]
        test_probs = probs[test_idx]
        val_forced_negative_mask = forced_negative_mask[val_idx]
        test_forced_negative_mask = forced_negative_mask[test_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
        if tune_threshold:
            threshold, tuning_metrics = _select_threshold(
                y_true=y_val,
                probs=val_probs,
                min_recall=min_recall,
                threshold_step=threshold_step,
                forced_negative_mask=val_forced_negative_mask,
            )
            msg = (
                f"Tuned threshold to {threshold:.2f} (min_recall={min_recall:.2f}, "
                f"false_positives={int(tuning_metrics['false_positives'])}, "
                f"class_0_recall={tuning_metrics['class_0_recall']:.3f}, "
                f"precision={tuning_metrics['precision']:.3f}, recall={tuning_metrics['recall']:.3f}, "
                f"f1={tuning_metrics['f1']:.3f})"
            )
            print(msg)
        preds = _preds_from_probs(
            probs=test_probs,
            threshold=threshold,
            forced_negative_mask=test_forced_negative_mask,
        )

        _compute_and_log_metrics(
            y_true=y_test,
            preds=preds,
            probs=test_probs,
            threshold=threshold,
            metrics_file=Path("/dev/null"),
        )
        test_urls = [urls[i] for i in test_idx]
        _print_false_negatives(y_true=y_test, preds=preds, urls=test_urls)
        _print_false_positives(y_true=y_test, preds=preds, urls=test_urls)
        print("Eval mode: metrics and feature importance are not written to disk.")
        return
    if mode == "tune":
        if urls is None:
            raise RuntimeError("Tuning requires a 'url' column in the dataset.")
        tune_vectorizer = cast(bool, args.tune_vectorizer)
        train_idx, val_idx, test_idx = _load_or_create_split(
            urls=urls,
            labels=y,
            split_path=split_path,
            val_split=val_split,
            test_split=test_split,
            random_state=random_state,
        )
        tune_idx = train_idx + val_idx
        tune_texts = [texts[i] for i in tune_idx]
        test_texts = [texts[i] for i in test_idx]
        test_urls = [urls[i] for i in test_idx]
        y_tune = y[tune_idx]
        y_test = y[test_idx]
        print(f"Tune pool (train+val): {len(tune_texts)} examples, Test set: {len(test_texts)} examples")
        print(f"Running Optuna with {optuna_trials} trials ({TUNE_CV_FOLDS}-fold stratified CV)...")
        best_params, best_threshold, best_metrics = _run_optuna_search(
            tune_texts=tune_texts,
            y_tune=y_tune,
            min_recall=min_recall,
            threshold_step=threshold_step,
            random_state=random_state,
            min_chars_hard_negative=min_chars_hard_negative,
            num_trials=optuna_trials,
            tune_vectorizer=tune_vectorizer,
            cv_folds=TUNE_CV_FOLDS,
        )
        msg = (
            f"Optuna best ({TUNE_CV_FOLDS}-fold CV): class_0_recall={best_metrics['class_0_recall']:.3f} "
            f"precision={best_metrics['precision']:.3f} recall={best_metrics['recall']:.3f} "
            f"f1={best_metrics['f1']:.3f} false_positives={int(best_metrics['false_positives'])} "
            f"threshold={best_threshold:.2f}"
        )
        print(msg)

        classifier = ExhibitClassifier(
            max_features=cast(int, best_params["max_features"]),
            char_max_features=cast(int, best_params["char_max_features"]),
            word_ngram_range=_ngram_from_key(cast(str, best_params["word_ngram_range"])),
            char_ngram_range=_ngram_from_key(cast(str, best_params["char_ngram_range"])),
            logreg_c=cast(float, best_params["logreg_c"]),
            logreg_max_iter=2000,
            class_weight=_class_weight_from_params(best_params),
            start_scan_chars=cast(int, best_params["start_scan_chars"]),
            min_chars_hard_negative=min_chars_hard_negative,
            random_state=random_state,
        )
        _ = classifier.fit(tune_texts, labels=cast(list[int], y_tune.tolist()))

        threshold = float(best_threshold)
        probs_tune = np.array(classifier.predict_proba_batch(tune_texts))
        tune_forced_negative_mask = _hard_negative_mask(
            texts=tune_texts,
            min_chars_hard_negative=min_chars_hard_negative,
        )
        final_threshold_metrics = _compute_threshold_metrics(
            y_true=y_tune,
            probs=probs_tune,
            threshold=threshold,
            forced_negative_mask=tune_forced_negative_mask,
        )
        probs_test = np.array(classifier.predict_proba_batch(test_texts))
        test_forced_negative_mask = _hard_negative_mask(
            texts=test_texts,
            min_chars_hard_negative=min_chars_hard_negative,
        )
        preds = _preds_from_probs(
            probs=probs_test,
            threshold=threshold,
            forced_negative_mask=test_forced_negative_mask,
        )
        print("Computing metrics...")
        EVAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_file = EVAL_METRICS_DIR / "exhibit_classifier_test_metrics.yaml"
        _compute_and_log_metrics(
            y_true=y_test,
            preds=preds,
            probs=probs_test,
            threshold=threshold,
            metrics_file=metrics_file,
        )
        error_urls_file = DEFAULT_ERROR_URLS_PATH
        _write_class_error_urls(
            y_true=y_test,
            preds=preds,
            urls=test_urls,
            output_file=error_urls_file,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        classifier.decision_threshold = float(threshold)
        print(f"Saving model to {output_path}...")
        _ = classifier.save(output_path)

        optuna_payload = {
            "best_params": best_params,
            "search_threshold": float(best_threshold),
            "search_metrics": best_metrics,
            "final_threshold": float(threshold),
            "final_threshold_source": "cv_median_fold_threshold",
            "final_threshold_metrics": final_threshold_metrics,
            "meta": {
                "min_recall": float(min_recall),
                "val_split": float(val_split),
                "test_split": float(test_split),
                "seed": int(random_state),
                "trials": int(optuna_trials),
                "cv_folds": int(TUNE_CV_FOLDS),
                "tune_pool_size": int(len(tune_texts)),
                "min_chars_hard_negative": int(min_chars_hard_negative),
            },
        }
        with open(DEFAULT_OPTUNA_PATH, "w", encoding="utf-8") as f:
            yaml.dump(optuna_payload, f, default_flow_style=False, sort_keys=False)
        print(f"[optuna] wrote best hyperparameters to {DEFAULT_OPTUNA_PATH}")
        print("Done.")
        return

    if urls is None:
        raise RuntimeError("False-negative reporting requires a 'url' column in the dataset.")
    pos_count = int(cast(int, np.sum(y == 1)))  # pyright: ignore[reportAny]
    neg_count = int(cast(int, np.sum(y == 0)))  # pyright: ignore[reportAny]
    print(f"Label distribution: {pos_count} positives, {neg_count} negatives")
    train_ratio = 1.0 - val_split - test_split
    print(f"Splitting data ({train_ratio:.0%} train, {val_split:.0%} val, {test_split:.0%} test)...")
    train_idx, val_idx, test_idx = _load_or_create_split(
        urls=urls,
        labels=y,
        split_path=split_path,
        val_split=val_split,
        test_split=test_split,
        random_state=random_state,
    )
    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    test_texts = [texts[i] for i in test_idx]
    test_urls = [urls[i] for i in test_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    print(f"Train set: {len(train_texts)} examples, Val set: {len(val_texts)} examples, Test set: {len(test_texts)} examples")
    print("Initializing classifier...")
    classifier = ExhibitClassifier(
        max_features=5000,
        min_chars_hard_negative=min_chars_hard_negative,
        random_state=random_state,
    )
    y_train_list = cast(list[int], y_train.tolist())
    print("Fitting classifier on training data...")
    _ = classifier.fit(train_texts, labels=y_train_list)

    print(f"Evaluating on {len(test_texts)} test examples...")
    probs_val = np.array(classifier.predict_proba_batch(val_texts))
    probs_test = np.array(classifier.predict_proba_batch(test_texts))
    val_forced_negative_mask = _hard_negative_mask(
        texts=val_texts,
        min_chars_hard_negative=min_chars_hard_negative,
    )
    test_forced_negative_mask = _hard_negative_mask(
        texts=test_texts,
        min_chars_hard_negative=min_chars_hard_negative,
    )
    if tune_threshold:
        threshold, tuning_metrics = _select_threshold(
            y_true=y_val,
            probs=probs_val,
            min_recall=cast(float, args.min_recall),
            threshold_step=threshold_step,
            forced_negative_mask=val_forced_negative_mask,
        )
        msg = (
            f"Tuned threshold to {threshold:.2f} (min_recall={cast(float, args.min_recall):.2f}, "
            f"false_positives={int(tuning_metrics['false_positives'])}, "
            f"class_0_recall={tuning_metrics['class_0_recall']:.3f}, "
            f"precision={tuning_metrics['precision']:.3f}, recall={tuning_metrics['recall']:.3f}, "
            f"f1={tuning_metrics['f1']:.3f})"
        )
        print(msg)
    preds = _preds_from_probs(
        probs=probs_test,
        threshold=threshold,
        forced_negative_mask=test_forced_negative_mask,
    )
    print("Computing metrics...")
    EVAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_file = EVAL_METRICS_DIR / "exhibit_classifier_test_metrics.yaml"
    _compute_and_log_metrics(
        y_true=y_test,
        preds=preds,
        probs=probs_test,
        threshold=threshold,
        metrics_file=metrics_file,
    )
    error_urls_file = DEFAULT_ERROR_URLS_PATH
    _write_class_error_urls(
        y_true=y_test,
        preds=preds,
        urls=test_urls,
        output_file=error_urls_file,
    )
    _print_false_negatives(y_true=y_test, preds=preds, urls=test_urls)
    _print_false_positives(y_true=y_test, preds=preds, urls=test_urls)
    
    coefs = classifier.get_model_coefficients()
    if coefs:
        importance_file = EVAL_METRICS_DIR / "exhibit_classifier_feature_importance.yaml"
        sorted_coefs = dict(sorted(coefs.items(), key=lambda item: abs(item[1]), reverse=True))
        
        with open(importance_file, "w") as f:
            yaml.dump(sorted_coefs, f, default_flow_style=False, sort_keys=False)
        print(f"Saved feature importance to {importance_file}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.decision_threshold = float(threshold)
    print(f"Saving model to {output_path}...")
    _ = classifier.save(output_path)
    print("Done.")


if __name__ == "__main__":
    main()
