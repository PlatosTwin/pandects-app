"""
Main NER training and inference module.

This module provides the main entry points for training and testing the NER model
using PyTorch Lightning with hyperparameter optimization via Optuna.
"""

# Standard library
import os
import time
from typing import Union
import yaml

# Environment config
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Data manipulation
import pandas as pd
from sklearn.model_selection import train_test_split

# ML frameworks and utilities
import torch

torch.set_float32_matmul_precision("high")

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger

from transformers import AutoTokenizer

from optuna import create_study
from optuna.integration import PyTorchLightningPruningCallback

# Local modules
from shared_constants import NER_LABEL_LIST, NER_CKPT_PATH, SPECIAL_TOKENS_TO_ADD
from ner_classes import NERTagger, NERDataModule

# Reproducibility
seed_everything(42, workers=True, verbose=False)


class NERTrainer:
    """
    Orchestrates hyperparameter optimization and training of NERTagger.

    Uses Optuna for hyperparameter search and PyTorch Lightning for training.
    """

    def __init__(
        self,
        data_csv: str,
        model_name: str,
        label_list: list,
        num_trials: int,
        max_epochs: int,
    ):
        """
        Initialize the NER trainer.

        Args:
            data_csv: Path to the data file
            model_name: HuggingFace model name
            label_list: List of label names
            num_trials: Number of Optuna trials
            max_epochs: Maximum training epochs per trial
        """
        self.data_csv = data_csv
        self.model_name = model_name
        self.num_trials = num_trials
        self.max_epochs = max_epochs
        self.label_list = label_list

        # Device selection
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.train_data = []
        self.val_data = []

    def _load_data(self) -> None:
        """
        Load and split data, stratified by presence of tags.
        """
        df = pd.read_csv(self.data_csv)
        df["tagged"] = df["llm_output"].apply(
            lambda x: 1 if "<section>" in x or "<article>" in x else 0
        )

        print(f"Loaded data shape: {df.shape}")
        print(df.head(2))
        print(f"Tagged value counts:\n{df['tagged'].value_counts()}")

        # For now, remove untagged pages
        # df = df[df["tagged"] == 1]

        train_data, val_data = train_test_split(
            df, test_size=0.2, stratify=df["tagged"], random_state=42
        )

        print(f"Train shape: {train_data.shape}, Validation shape: {val_data.shape}")

        self.train_data = train_data["llm_output"].to_list()
        self.val_data = val_data["llm_output"].to_list()

    def _get_callbacks(self, trial: object = None) -> tuple:
        """
        Instantiate Lightning callbacks.

        Args:
            trial: Optuna trial for pruning callback

        Returns:
            Tuple of callbacks
        """
        # Single checkpoint callback for best val_loss
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_loss:.4f}",
        )
        early_stop_callback = EarlyStopping(
            monitor="val_f1_doc", patience=3, mode="max"
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        pruning_callback = (
            [PyTorchLightningPruningCallback(trial, monitor="val_f1_doc")]
            if trial is not None
            else []
        )
        progress_bar_callback = TQDMProgressBar(refresh_rate=100)

        return (
            checkpoint_callback,
            early_stop_callback,
            lr_monitor,
            progress_bar_callback,
            pruning_callback,
        )

    def _build(self, params: dict) -> tuple[NERDataModule, NERTagger]:
        """
        Instantiate DataModule and Model from hyperparameters.

        Args:
            params: Dictionary of hyperparameters

        Returns:
            Tuple of (data_module, model)
        """
        data_module = NERDataModule(
            train_data=self.train_data,
            val_data=self.val_data,
            tokenizer_name=self.model_name,
            label_list=self.label_list,
            batch_size=params["batch_size"],
            train_subsample_window=params["train_subsample_window"],
            num_workers=7,
        )
        model = NERTagger(
            model_name=self.model_name,
            num_labels=len(self.label_list),
            id2label={idx: label for idx, label in enumerate(self.label_list)},
            learning_rate=params["lr"],
            weight_decay=params["weight_decay"],
            warmup_steps_pct=params["warmup_steps_pct"],
        )
        return data_module, model

    def _objective(self, trial: object) -> float:
        """
        Optuna objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Validation loss
        """
        params = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "train_subsample_window": trial.suggest_categorical(
                "train_subsample_window", [128, 256, 512]
            ),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
            "warmup_steps_pct": trial.suggest_float("warmup_steps_pct", 0.0, 0.3),
        }

        data_module, model = self._build(params)
        (
            checkpoint_callback,
            early_stop_callback,
            lr_monitor,
            progress_bar_callback,
            pruning_callback,
        ) = self._get_callbacks(trial)

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.device,
            precision="bf16-mixed",
            devices=1,
            logger=TensorBoardLogger("tb_logs", name="optuna"),
            callbacks=[
                checkpoint_callback,
                early_stop_callback,
                lr_monitor,
                progress_bar_callback,
                *pruning_callback,
            ],
            log_every_n_steps=10,
            deterministic=True,
        )
        trainer.fit(model, datamodule=data_module)

        val_f1_doc = trainer.callback_metrics["val_f1_doc"].item()

        # Clean up to avoid memory leaks
        del (
            model,
            data_module,
            trainer,
            checkpoint_callback,
            early_stop_callback,
            lr_monitor,
        )
        if pruning_callback:
            del pruning_callback

        return val_f1_doc

    def run(self) -> None:
        """Execute hyperparameter optimization and final training."""
        self._load_data()

        study = create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.num_trials, gc_after_trial=True)

        print("Finished hyperparameter optimization 👉")
        print(f"  Best val_f1_doc: {study.best_value:.4f}")
        print("  Best hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"    • {key}: {value}")

        # Retrain best model to get its checkpoint on disk
        best_params = study.best_trial.params
        data_module, model = self._build(best_params)
        (
            checkpoint_callback,
            early_stop_callback,
            lr_monitor,
            progress_bar_callback,
            _,
        ) = self._get_callbacks()

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.device,
            devices=1,
            logger=TensorBoardLogger("tb_logs", name="final"),
            callbacks=[
                checkpoint_callback,
                early_stop_callback,
                lr_monitor,
                progress_bar_callback,
            ],
            log_every_n_steps=10,
        )
        trainer.fit(model, datamodule=data_module)


class NERInference:
    """
    Wrapper for trained NERTagger for easy batch inference.

    Provides convenient interface for running NER predictions on new text data.
    """

    def __init__(
        self,
        ckpt_path: str,
        label_list: list[str],
        device: str = "cpu",
        review_threshold: float = 0.5,
        window_batch_size: int = 32,
    ) -> None:
        """
        Initialize the NER inference wrapper.

        Args:
            ckpt_path: Path to trained model checkpoint
            label_list: List of label names (MUST be the new BIOES list)
            device: Device to use for inference
            review_threshold: Confidence threshold for review
            window_batch_size: Batch size for window processing
        """
        # Load and prepare model
        self.model: NERTagger = NERTagger.load_from_checkpoint(ckpt_path)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # Tokenizer & label maps
        model_name = self.model.hparams.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS_TO_ADD}
        )
        self.label_list = label_list
        self.id2label = {idx: label for idx, label in enumerate(label_list)}
        self.label2id = {label: idx for idx, label in self.id2label.items()}

        self.review_threshold = review_threshold
        self.window_batch_size = window_batch_size

    def _batch_predict_full_texts(
        self,
        texts: list[str],
        window: int = 512,
        stride: int = 256,
    ) -> tuple[list[str], list[list[int]], list[list[float]]]:
        """
        Batch-window NER on multiple texts.

        Args:
            texts: List of texts to process
            window: Window size for sliding window
            stride: Stride size for sliding window

        Returns:
            Tuple of (cleaned_texts, raw_predictions_list, confidences_list)
        """
        # Flatten windows
        all_records: list[tuple[int, int, int]] = []
        all_slices: list[str] = []
        for text_idx, text in enumerate(texts):
            text_length = len(text)
            start = 0
            while start < text_length:
                end = min(start + window, text_length)
                all_records.append((text_idx, start, end))
                all_slices.append(text[start:end])
                if end == text_length:
                    break
                start += stride

        # Pre-allocate buffers per text
        num_labels = len(self.label_list)
        sum_probabilities = [
            torch.zeros((len(texts[i]), num_labels), device=self.device)
            for i in range(len(texts))
        ]
        counts = [
            torch.zeros(len(texts[i]), device=self.device) for i in range(len(texts))
        ]

        # Process windows in chunks
        for i in range(0, len(all_slices), self.window_batch_size):
            batch_slices = all_slices[i : i + self.window_batch_size]
            batch_records = all_records[i : i + self.window_batch_size]

            encoding = self.tokenizer(
                batch_slices,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=window,
                return_attention_mask=True,
                return_offsets_mapping=True,
            )
            for key in ("input_ids", "attention_mask", "offset_mapping"):
                encoding[key] = encoding[key].to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                )
                probabilities = torch.softmax(outputs.logits, dim=-1)

            for (text_idx, window_start, _), probs, offsets, attention in zip(
                batch_records,
                probabilities,
                encoding["offset_mapping"],
                encoding["attention_mask"],
            ):
                valid_tokens = attention.sum().item()
                for token_idx in range(valid_tokens):
                    offset_start, offset_end = offsets[token_idx].tolist()
                    if offset_end == 0:
                        continue
                    for char_idx in range(offset_start, offset_end):
                        char_position = window_start + char_idx
                        if char_position < sum_probabilities[text_idx].size(0):
                            sum_probabilities[text_idx][char_position] += probs[
                                token_idx
                            ]
                            counts[text_idx][char_position] += 1

        # Finalize predictions
        raw_predictions_list: list[list[int]] = []
        confidences_list: list[list[float]] = []

        for text_idx, text in enumerate(texts):
            avg_probabilities = sum_probabilities[text_idx] / counts[
                text_idx
            ].unsqueeze(-1).clamp(min=1)
            confidences = avg_probabilities.max(dim=-1)[0].cpu().tolist()
            raw_predictions = avg_probabilities.argmax(dim=-1).cpu().tolist()

            # BIOES fix: Correct invalid tag sequences.
            for j in range(len(raw_predictions)):
                curr_label = self.id2label[raw_predictions[j]]
                if (
                    curr_label.startswith("B-")
                    or curr_label.startswith("S-")
                    or curr_label == "O"
                ):
                    continue

                # An I- or E- tag is invalid at the start of a sequence, convert to B-
                if j == 0:
                    entity = curr_label.split("-", 1)[1]
                    raw_predictions[j] = self.label2id.get(
                        f"B-{entity}", raw_predictions[j]
                    )
                    continue

                # Check previous label
                prev_label = self.id2label[raw_predictions[j - 1]]

                # An I- or E- tag is invalid if the previous tag was O or a different entity
                if (
                    prev_label == "O"
                    or prev_label.startswith("E-")
                    or prev_label.startswith("S-")
                ):
                    entity = curr_label.split("-", 1)[1]
                    raw_predictions[j] = self.label2id.get(
                        f"B-{entity}", raw_predictions[j]
                    )
                elif prev_label.startswith("B-") or prev_label.startswith("I-"):
                    prev_entity = prev_label.split("-", 1)[1]
                    curr_entity = curr_label.split("-", 1)[1]
                    if prev_entity != curr_entity:
                        raw_predictions[j] = self.label2id.get(
                            f"B-{curr_entity}", raw_predictions[j]
                        )

            raw_predictions_list.append(raw_predictions)
            confidences_list.append(confidences)

        return texts, raw_predictions_list, confidences_list

    def _pretty_print_ner_text(
        self,
        cleaned_text: str,
        pred_labels: list[int],
    ) -> str:
        """
        Convert predictions to tagged text format by grouping consecutive
        characters with the same entity type. This is more robust to
        models that might not produce perfect BIOES sequences.

        Args:
            cleaned_text: The clean text without any tags.
            pred_labels: The list of predicted label IDs for each character.

        Returns:
            The text with XML-style tags for entities.
        """
        assert len(cleaned_text) == len(
            pred_labels
        ), "Text and predictions must have the same length."
        result: list[str] = []

        # Helper to extract the core entity name (e.g., 'section' from 'B-section')
        def get_entity_name(label_id: int) -> str:
            tag = self.id2label.get(label_id, "O")
            if tag == "O":
                return "O"
            return tag.split("-", 1)[1].lower()

        # Pad with a dummy label to handle closing the tag for the last character
        for i, (char, label_id) in enumerate(zip(cleaned_text, pred_labels)):
            current_entity = get_entity_name(label_id)
            prev_entity = get_entity_name(pred_labels[i - 1]) if i > 0 else "O"

            # If the entity type has changed, we need to adjust the tags
            if current_entity != prev_entity:
                # Close the previous tag if it was a real entity
                if prev_entity != "O":
                    result.append(f"</{prev_entity}>")
                # Open a new tag if the current one is a real entity
                if current_entity != "O":
                    result.append(f"<{current_entity}>")

            # Always append the current character
            result.append(char)

        # After the loop, check if the very last character was part of an entity
        if pred_labels:
            last_entity = get_entity_name(pred_labels[-1])
            if last_entity != "O":
                result.append(f"</{last_entity}>")

        return "".join(result)

    def _get_uncertain_spans(
        self,
        predictions: list[int],
        confidences: list[float],
    ) -> tuple[int, list[dict]]:
        """
        Group low-confidence characters into spans, emitting entity names.

        Args:
            predictions: Predicted label IDs
            confidences: Confidence scores

        Returns:
            Tuple of (low_count, spans)
        """
        low_positions = [
            i for i, conf in enumerate(confidences) if conf < self.review_threshold
        ]
        low_count = len(low_positions)
        spans: list[dict] = []

        if not low_positions:
            return low_count, spans

        span_start = low_positions[0]
        span_label_id = predictions[span_start]
        confidence_list = [confidences[span_start]]
        prev = span_start

        def _entity_name(label_id: int) -> str:
            label = self.id2label[label_id]
            if "-" in label:
                return label.split("-", 1)[1].lower()
            return label.lower()

        for pos in low_positions[1:]:
            if pos == prev + 1 and predictions[pos] == span_label_id:
                confidence_list.append(confidences[pos])
                prev = pos
            else:
                spans.append(
                    {
                        "entity": _entity_name(span_label_id),
                        "start": span_start,
                        "end": prev,
                        "avg_confidence": sum(confidence_list) / len(confidence_list),
                    }
                )
                span_start = pos
                span_label_id = predictions[pos]
                confidence_list = [confidences[pos]]
                prev = pos

        spans.append(
            {
                "entity": _entity_name(span_label_id),
                "start": span_start,
                "end": prev,
                "avg_confidence": sum(confidence_list) / len(confidence_list),
            }
        )
        return low_count, spans

    def label(
        self,
        texts: list[str],
        verbose: bool = False,
    ) -> list[dict[str, Union[str, int, list[dict], list[dict]]]]:
        """
        Batch-label a list of texts.

        Args:
            texts: List of texts to label
            verbose: Whether to print detailed output

        Returns:
            List of result dictionaries with tagged text, low count, spans, and chars
        """
        cleaned_texts, predictions_list, confidences_list = (
            self._batch_predict_full_texts(texts)
        )
        results: list[dict[str, Union[str, int, list[dict], list[dict]]]] = []

        for idx, (cleaned, predictions, confidences) in enumerate(
            zip(cleaned_texts, predictions_list, confidences_list)
        ):
            tagged = self._pretty_print_ner_text(cleaned, predictions)
            low_count, spans = self._get_uncertain_spans(predictions, confidences)

            chars = [
                {
                    "pos": i,
                    "char": ch,
                    "entity": (
                        self.id2label[pid].split("-", 1)[1].lower()
                        if "-" in self.id2label[pid]
                        else self.id2label[pid].lower()
                    ),
                    "confidence": conf,
                }
                for i, (ch, conf, pid) in enumerate(
                    zip(cleaned, confidences, predictions)
                )
                if conf < self.review_threshold
            ]

            if verbose:
                print(f"\n=== Text #{idx} ===")
                print(tagged)
                print(
                    f"\n[low-confidence chars < {self.review_threshold}]: {low_count}"
                )
                if spans:
                    print("\nUncertain spans:")
                    for span in spans:
                        snippet = cleaned[span["start"] : span["end"] + 1].replace(
                            "\n", "\\n"
                        )
                        print(
                            f" - chars {span['start']}-{span['end']} "
                            f"({snippet}) as {span['entity']} "
                            f"(avg_conf={span['avg_confidence']:.2f})"
                        )
                else:
                    print("No spans below threshold.")

            results.append(
                {
                    "tagged": tagged,
                    "low_count": low_count,
                    "spans": spans,
                    "chars": chars,
                }
            )

        return results


def main(mode: str = "test") -> None:
    """
    Main entry point for NER training and testing.

    Args:
        mode: Either 'train' or 'test'
    """
    if mode == "train":
        ner_trainer = NERTrainer(
            data_csv="../data/ner-data.csv",
            model_name="answerdotai/ModernBERT-base",
            label_list=NER_LABEL_LIST,
            num_trials=10,
            max_epochs=10,
        )
        ner_trainer.run()

    elif mode == "test":
        # Load test samples
        with open(
            "etl/src/etl/models/data/ner_samples.yaml", "r", encoding="utf-8"
        ) as f:
            data = yaml.safe_load(f)

        samples = data["samples"]

        # Initialize inference model
        inference_model = NERInference(
            ckpt_path=NER_CKPT_PATH, label_list=NER_LABEL_LIST, review_threshold=0.975
        )

        # Run inference
        start = time.time()
        tagged_result = inference_model.label(samples, verbose=False)
        inference_time = time.time() - start

        print(tagged_result)
        print(f"Inference time: {inference_time:.2f} seconds")

    else:
        raise RuntimeError(f"Invalid mode: {mode}. Use 'train' or 'test'")


if __name__ == "__main__":
    main(mode="test")
