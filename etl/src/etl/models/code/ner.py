"""
Main NER training and inference module.

This module provides the main entry points for training and testing the NER model
using PyTorch Lightning with hyperparameter optimization via Optuna.
"""

# Standard library
import os
import time
import yaml
import pprint

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
            monitor="val_ent_f1",
            mode="max",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_ent_f1:.4f}",
        )
        early_stop_callback = EarlyStopping(
            monitor="val_ent_f1", patience=3, mode="max"
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        pruning_callback = (
            [PyTorchLightningPruningCallback(trial, monitor="val_ent_f1")]
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

        val_ent_f1 = trainer.callback_metrics["val_ent_f1"].item()

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

        return val_ent_f1

    def run(self) -> None:
        """Execute hyperparameter optimization and final training."""
        self._load_data()

        study = create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.num_trials, gc_after_trial=True)

        print("Finished hyperparameter optimization 👉")
        print(f"  Best val_ent_f1: {study.best_value:.4f}")
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
    Token-window NER inference that returns:
      - tagged (str): XML-tagged text
      - low_count (int): number of low-confidence tokens
      - spans (list[dict]): contiguous low-confidence token spans
      - tokens (list[dict]): token-level records below threshold
    """

    def __init__(
        self,
        ckpt_path: str,
        label_list: list[str] | None,
        device: str = "cpu",
        review_threshold: float = 0.5,
        window_batch_size: int = 32,
        window: int = 510,
        stride: int = 256,
    ) -> None:
        self.device = torch.device(device)
        self.model: NERTagger = NERTagger.load_from_checkpoint(
            ckpt_path, map_location=self.device
        )
        self.model.to(self.device)
        self.model.eval()

        # tokenizer consistent with training (special tokens added, no resize needed at inference)
        model_name = self.model.hparams.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS_TO_ADD}
        )

        # Fallbacks for essential token IDs (safe-guards)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token or self.tokenizer.unk_token or "[PAD]"
            )
        if self.tokenizer.cls_token_id is None:
            self.tokenizer.cls_token = (
                self.tokenizer.bos_token or self.tokenizer.unk_token or "[CLS]"
            )
        if self.tokenizer.sep_token_id is None:
            self.tokenizer.sep_token = (
                self.tokenizer.eos_token or self.tokenizer.unk_token or "[SEP]"
            )

        # >>> Use id2label/label2id from checkpoint to avoid order drift
        ckpt_id2label = dict(self.model.hparams.id2label)  # keys are ints in training
        self.id2label = {int(k): v for k, v in ckpt_id2label.items()}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.label_list = [self.id2label[i] for i in range(len(self.id2label))]

        # Optional: validate if user passes label_list
        if label_list is not None:
            if list(label_list) != self.label_list:
                raise ValueError(
                    "label_list provided to NERInference does not match the checkpoint label order.\n"
                    f"Checkpoint: {self.label_list}\nProvided:   {list(label_list)}"
                )

        self.C = len(self.label_list)
        self.review_threshold = review_threshold
        self.window_batch_size = window_batch_size
        self.window = window
        self.stride = stride

    # ---------------- Token aggregation over sliding windows (logit stitching) ----------------
    def _predict_tokens(
        self, text: str
    ) -> tuple[list[int], list[float], list[tuple[int, int]], list[str], torch.Tensor]:
        """
        Stitch per-token predictions across overlapping windows by averaging LOGITS
        (to match validation), then compute confidences from softmax(avg_logits) and
        apply a light BIOES repair on the predicted tag sequence.

        Returns:
            preds        : List[int]            # predicted label ids per token
            confidences  : List[float]          # max prob per token (from softmax(avg_logits))
            offsets      : List[tuple[int,int]] # original char offsets per token
            toks         : List[str]            # token strings
            avg_logits   : torch.Tensor         # [T, C] averaged logits on CPU
        """
        enc_full = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=False,
            add_special_tokens=False,
        )
        input_ids = enc_full["input_ids"][0]                 # [T]
        offsets = enc_full["offset_mapping"][0].tolist()     # [(s,e)] * T
        toks = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())
        T = len(input_ids)

        if T == 0:
            empty_logits = torch.zeros((0, self.C), dtype=torch.float32)
            return [], [], [], [], empty_logits

        # --- Accumulate LOGITS (not probabilities) across overlapping windows ---
        sum_logits = torch.zeros((T, self.C), device="cpu")
        counts = torch.zeros(T, device="cpu")

        # Build token windows; add CLS/SEP per chunk
        windows: list[dict] = []
        bounds: list[tuple[int, int]] = []  # (start_tok, end_tok)
        i = 0
        while i < T:
            j = min(i + self.window, T)
            chunk_ids = input_ids[i:j].tolist()
            windows.append(
                {
                    "input_ids": [self.tokenizer.cls_token_id] + chunk_ids + [self.tokenizer.sep_token_id],
                    "attention_mask": [1] * (len(chunk_ids) + 2),
                }
            )
            bounds.append((i, j))
            if j == T:
                break
            i += self.stride

        # Batched inference -> return LOGITS with CLS/SEP removed
        def _infer_logits(batch_items: list[dict]) -> list[torch.Tensor]:
            max_len = max(len(x["input_ids"]) for x in batch_items)
            pad_id = self.tokenizer.pad_token_id

            ids = [x["input_ids"] + [pad_id] * (max_len - len(x["input_ids"])) for x in batch_items]
            mask = [x["attention_mask"] + [0] * (max_len - len(x["attention_mask"])) for x in batch_items]

            ids_t = torch.tensor(ids, device=self.device)
            mask_t = torch.tensor(mask, device=self.device)
            with torch.no_grad():
                logits = self.model(input_ids=ids_t, attention_mask=mask_t).logits  # [B,L,C]

            outs: list[torch.Tensor] = []
            for lg, m in zip(logits, mask_t):
                true_len = int(m.sum().item())  # includes CLS/SEP; pads are 0
                if true_len <= 2:
                    outs.append(lg[0:0])  # empty
                else:
                    outs.append(lg[1 : true_len - 1])  # strip CLS/SEP
            return outs

        all_logits: list[torch.Tensor] = []
        for k in range(0, len(windows), self.window_batch_size):
            all_logits.extend(_infer_logits(windows[k : k + self.window_batch_size]))

        # Accumulate per-token LOGITS over window overlaps
        for (s, e), lg_tok in zip(bounds, all_logits):
            span = min(e - s, lg_tok.size(0))
            if span <= 0:
                continue
            sum_logits[s : s + span] += lg_tok[:span].to("cpu")
            counts[s : s + span] += 1

        # Average logits like validation; then derive probs/confidence/preds
        avg_logits = sum_logits / counts.unsqueeze(-1).clamp(min=1.0)
        probs = torch.softmax(avg_logits, dim=-1)
        confidences = probs.max(dim=-1)[0].tolist()
        preds = avg_logits.argmax(dim=-1).tolist()

        # --- Light BIOES repair on the predicted sequence ---
        def _repair_bioes(seq: list[int]) -> list[int]:
            out = seq[:]
            for t in range(len(out)):
                lab = self.id2label[out[t]]
                if lab == "O" or lab.startswith(("B-", "S-")):
                    continue
                if t == 0:
                    ent = lab.split("-", 1)[1]
                    out[t] = self.label2id.get(f"B-{ent}", out[t])
                    continue
                prev = self.id2label[out[t - 1]]
                if prev == "O" or prev.startswith(("E-", "S-")):
                    ent = lab.split("-", 1)[1]
                    out[t] = self.label2id.get(f"B-{ent}", out[t])
                elif prev.startswith(("B-", "I-")):
                    prev_ent = prev.split("-", 1)[1]
                    cur_ent = lab.split("-", 1)[1]
                    if prev_ent != cur_ent:
                        out[t] = self.label2id.get(f"B-{cur_ent}", out[t])
            return out

        preds = _repair_bioes(preds)
        return preds, confidences, offsets, toks, avg_logits

    # ---------------- Pretty print from token labels via offsets ----------------
    def _pretty_print_from_tokens(
        self, text: str, preds: list[int], offsets: list[tuple[int, int]]
    ) -> str:
        def ent(lid: int) -> str:
            tag = self.id2label.get(lid, "O")
            return "O" if tag == "O" else tag.split("-", 1)[1].lower()

        res = []
        cur_ent = "O"
        pos = 0  # last emitted char pos in source text

        for lid, (s, e) in zip(preds, offsets):
            if e == 0 or s >= e:
                continue
            if pos < s:
                res.append(text[pos:s])
                pos = s
            tok_ent = ent(lid)
            if tok_ent != cur_ent:
                if cur_ent != "O":
                    res.append(f"</{cur_ent}>")
                if tok_ent != "O":
                    res.append(f"<{tok_ent}>")
                cur_ent = tok_ent
            res.append(text[s:e])
            pos = e

        if pos < len(text):
            res.append(text[pos:])
        if cur_ent != "O":
            res.append(f"</{cur_ent}>")
        return "".join(res)

    # ---------------- Low-confidence spans over tokens ----------------
    def _token_spans(
        self, preds: list[int], confs: list[float]
    ) -> tuple[int, list[dict]]:
        low_idxs = [i for i, c in enumerate(confs) if c < self.review_threshold]
        low_count = len(low_idxs)
        spans: list[dict] = []
        if not low_idxs:
            return low_count, spans

        def ent_name(lid: int) -> str:
            lab = self.id2label[lid]
            return lab.split("-", 1)[1].lower() if "-" in lab else lab.lower()

        start = low_idxs[0]
        cur_lab = preds[start]
        acc = [confs[start]]
        prev = start

        for i in low_idxs[1:]:
            if i == prev + 1 and preds[i] == cur_lab:
                acc.append(confs[i])
                prev = i
            else:
                spans.append(
                    {
                        "entity": ent_name(cur_lab),
                        "start_token": start,
                        "end_token": prev,
                        "avg_confidence": sum(acc) / len(acc),
                    }
                )
                start = i
                cur_lab = preds[i]
                acc = [confs[i]]
                prev = i

        spans.append(
            {
                "entity": ent_name(cur_lab),
                "start_token": start,
                "end_token": prev,
                "avg_confidence": sum(acc) / len(acc),
            }
        )
        return low_count, spans

    # ---------------- Public API ----------------
    # 2) label() gains a flag and optionally returns token_probs
    def label(
        self,
        texts: list[str],
        verbose: bool = False,
        return_token_probs: bool = False,   # <— new flag
    ) -> list[dict]:
        results = []
        for idx, text in enumerate(texts):
            preds, confs, offsets, _, avg_logits = self._predict_tokens(text)
            tagged = self._pretty_print_from_tokens(text, preds, offsets)
            low_count, spans = self._token_spans(preds, confs)

            tokens_below = []
            for i, ((s, e), conf, lid) in enumerate(zip(offsets, confs, preds)):
                if e == 0 or s >= e:
                    continue
                if conf < self.review_threshold:
                    lab = self.id2label[lid]
                    tokens_below.append(
                        {
                            "i": i,
                            "token": text[s:e],
                            "start": s,
                            "end": e,
                            "entity": lab.split("-", 1)[1].lower() if "-" in lab else lab.lower(),
                            "confidence": conf,
                        }
                    )

            out = {
                "tagged": tagged,
                "low_count": low_count,
                "spans": spans,
                "tokens": tokens_below,
            }

            if return_token_probs:
                # Build full per-token probability vectors (from stitched avg_logits)
                probs_full = torch.softmax(avg_logits, dim=-1).tolist()
                token_probs = []
                for i, ((s, e), lid, pv) in enumerate(zip(offsets, preds, probs_full)):
                    if e == 0 or s >= e:
                        continue
                    token_probs.append(
                        {
                            "i": i,
                            "token": text[s:e],
                            "start": s,
                            "end": e,
                            "pred_class": self.id2label[lid],
                            "confidence": max(pv),
                            "probs": {self.id2label[c]: float(pv[c]) for c in range(self.C)},
                        }
                    )
                out["token_probs"] = token_probs

            if verbose:
                print(f"\n=== Text #{idx} ===")
                print(tagged)
                print(f"\n[low-confidence tokens < {self.review_threshold}]: {low_count}")
                if spans:
                    print("\nUncertain spans (token-level):")
                    for sp in spans:
                        print(f" - {sp}")
                else:
                    print("No spans below threshold.")

            results.append(out)
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
        tagged_result = inference_model.label(samples, verbose=True)
        inference_time = time.time() - start

        print(f"Inference time: {inference_time:.2f} seconds")

    else:
        raise RuntimeError(f"Invalid mode: {mode}. Use 'train' or 'test'")


if __name__ == "__main__":
    main(mode="test")
