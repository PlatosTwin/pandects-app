# Standard library
import os
import time
from typing import Union

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
from .constants import NER_LABEL_LIST, NER_CKPT_PATH
from .ner_classes import NERTagger, NERDataModule

# Reproducibility
seed_everything(42, workers=True, verbose=False)


class NERTrainer:

    def __init__(
        self,
        data_csv: str,
        model_name: str,
        label_list: list,
        num_trials: int,
        max_epochs: int,
    ):
        self.data_csv = data_csv
        self.MODEL_NAME = model_name
        self.NUM_TRIALS = num_trials
        self.MAX_EPOCHS = max_epochs
        self.LABEL_LIST = label_list

        if torch.backends.mps.is_available():
            self.DEVICE = "mps"
        elif torch.cuda.is_available():
            self.DEVICE = "cuda"
        else:
            self.DEVICE = "cpu"

        self.train_data = []
        self.val_data = []

    def _load_data(self):
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

        # for now, remove untagged pages
        # df = df[df["tagged"] == 1]

        train_data, val_data = train_test_split(
            df, test_size=0.2, stratify=df["tagged"], random_state=42
        )

        print(f"Train shape: {train_data.shape}, Validation shape: {val_data.shape}")

        self.train_data = train_data["llm_output"].to_list()
        self.val_data = val_data["llm_output"].to_list()

    def _get_callbacks(self, trial=None):
        # single checkpoint callback for best val_loss
        checkpoint_cb = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_loss:.4f}",
        )
        early_stop_cb = EarlyStopping(monitor="val_loss", patience=3, mode="min")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        pruning_cb = (
            [PyTorchLightningPruningCallback(trial, monitor="val_loss")]
            if trial is not None
            else []
        )
        progress_bar_cb = TQDMProgressBar(refresh_rate=15)

        return checkpoint_cb, early_stop_cb, lr_monitor, progress_bar_cb, pruning_cb

    def _build(self, params):
        """Instantiate DataModule and Model from a dict of hyperparams."""
        dm = NERDataModule(
            train_data=self.train_data,
            val_data=self.val_data,
            tokenizer_name=self.MODEL_NAME,
            label_list=self.LABEL_LIST,
            batch_size=params["batch_size"],
            train_subsample_window=params["train_subsample_window"],
            num_workers=7,
        )
        model = NERTagger(
            model_name=self.MODEL_NAME,
            num_labels=len(self.LABEL_LIST),
            id2label={i: l for i, l in enumerate(self.LABEL_LIST)},
            learning_rate=params["lr"],
            weight_decay=params["weight_decay"],
            warmup_steps_pct=params["warmup_steps_pct"],
        )
        return dm, model

    def _objective(self, trial):
        params = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "train_subsample_window": trial.suggest_categorical(
                "train_subsample_window", [128, 256, 512]
            ),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
            "warmup_steps_pct": trial.suggest_float("warmup_steps_pct", 0.0, 0.3),
        }

        dm, model = self._build(params)
        checkpoint_cb, early_stop_cb, lr_monitor, progress_bar_cb, pruning_cb = (
            self._get_callbacks(trial)
        )

        trainer = pl.Trainer(
            max_epochs=self.MAX_EPOCHS,
            accelerator=self.DEVICE,
            precision="bf16-mixed",
            devices=1,
            logger=TensorBoardLogger("tb_logs", name="optuna"),
            callbacks=[
                checkpoint_cb,
                early_stop_cb,
                lr_monitor,
                progress_bar_cb,
                *pruning_cb,
            ],
            log_every_n_steps=10,
            deterministic=True,
        )
        trainer.fit(model, datamodule=dm)

        val = trainer.callback_metrics["val_loss"].item()

        del model, dm, trainer, checkpoint_cb, early_stop_cb, lr_monitor
        if pruning_cb:
            del pruning_cb

        return val

    def run(self):
        self._load_data()

        study = create_study(direction="minimize")
        study.optimize(self._objective, n_trials=self.NUM_TRIALS, gc_after_trial=True)

        print("Finished HPO 👉")
        print(f"  Best val_loss: {study.best_value:.4f}")
        print("  Best hyperparams:")
        for k, v in study.best_trial.params.items():
            print(f"    • {k}: {v}")

        # Retrain best model to get its checkpoint on disk
        best_params = study.best_trial.params
        dm, model = self._build(best_params)
        checkpoint_cb, early_stop_cb, lr_monitor, progress_bar_cb, _ = (
            self._get_callbacks()
        )

        trainer = pl.Trainer(
            max_epochs=self.MAX_EPOCHS,
            accelerator=self.DEVICE,
            devices=1,
            logger=TensorBoardLogger("tb_logs", name="final"),
            callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar_cb],
            log_every_n_steps=10,
        )
        trainer.fit(model, datamodule=dm)


class NERInference:
    def __init__(
        self,
        ckpt_path: str,
        label_list: list[str],
        device: str = "cpu",
        review_threshold: float = 0.5,
        window_batch_size: int = 32,
    ) -> None:
        # 1) Load & prep model
        self.model: NERTagger = NERTagger.load_from_checkpoint(ckpt_path)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # 2) Tokenizer & label maps
        model_name = self.model.hparams.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.label_list = label_list
        self.id2label = {i: l for i, l in enumerate(label_list)}
        self.label2id = {l: i for i, l in self.id2label.items()}

        self.review_threshold = review_threshold
        self.window_batch_size = window_batch_size

    def _batch_predict_full_texts(
        self,
        texts: list[str],
        window: int = 512,
        stride: int = 256,
    ) -> tuple[list[str], list[list[int]], list[list[float]]]:
        """
        Batch-window NER on multiple texts:
        - flatten windows across all texts,
        - tokenize in sub-batches of size self.window_batch_size,
        - aggregate token probs to chars,
        - average, argmax, BIO-fix.
        Returns (texts, raw_preds_list, confidences_list).
        """
        # 1) Flatten windows
        all_recs: list[tuple[int, int, int]] = []
        all_slices: list[str] = []
        for tidx, text in enumerate(texts):
            T = len(text)
            start = 0
            while start < T:
                end = min(start + window, T)
                all_recs.append((tidx, start, end))
                all_slices.append(text[start:end])
                if end == T:
                    break
                start += stride

        # 2) Pre-allocate buffers per text
        num_labels = len(self.label_list)
        sum_p = [
            torch.zeros((len(texts[i]), num_labels), device=self.device)
            for i in range(len(texts))
        ]
        counts = [
            torch.zeros(len(texts[i]), device=self.device) for i in range(len(texts))
        ]

        # 3) Process windows in chunks
        for i in range(0, len(all_slices), self.window_batch_size):
            batch_slices = all_slices[i : i + self.window_batch_size]
            batch_recs = all_recs[i : i + self.window_batch_size]

            enc = self.tokenizer(
                batch_slices,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=window,
                return_attention_mask=True,
                return_offsets_mapping=True,
            )
            for k in ("input_ids", "attention_mask", "offset_mapping"):
                enc[k] = enc[k].to(self.device)

            with torch.no_grad():
                out = self.model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                )
                probs = torch.softmax(out.logits, dim=-1)

            for (tidx, w0, _), p, offsets, attn in zip(
                batch_recs,
                probs,
                enc["offset_mapping"],
                enc["attention_mask"],
            ):
                valid = attn.sum().item()
                for tok_i in range(valid):
                    off_start, off_end = offsets[tok_i].tolist()
                    if off_end == 0:
                        continue
                    for c in range(off_start, off_end):
                        idx_char = w0 + c
                        if idx_char < sum_p[tidx].size(0):
                            sum_p[tidx][idx_char] += p[tok_i]
                            counts[tidx][idx_char] += 1

        # 4) Finalize predictions
        raw_preds_list: list[list[int]] = []
        confidences_list: list[list[float]] = []

        for tidx, text in enumerate(texts):
            avg_p = sum_p[tidx] / counts[tidx].unsqueeze(-1).clamp(min=1)
            confidences = avg_p.max(dim=-1)[0].cpu().tolist()
            raw_preds = avg_p.argmax(dim=-1).cpu().tolist()

            # BIO fix
            for j in range(1, len(raw_preds)):
                prev_lbl = self.id2label[raw_preds[j - 1]]
                curr_lbl = self.id2label[raw_preds[j]]
                if curr_lbl.startswith("B-"):
                    ent = curr_lbl.split("-", 1)[1]
                    if prev_lbl in (f"I-{ent}", f"B-{ent}"):
                        raw_preds[j] = self.label2id[f"I-{ent}"]

            raw_preds_list.append(raw_preds)
            confidences_list.append(confidences)

        return texts, raw_preds_list, confidences_list

    def _pretty_print_ner_text(
        self,
        cleaned_text: str,
        pred_labels: list[int],
    ) -> str:
        """
        Same as before — but now your pred_labels will only
        have a single B- at the start of each span.
        """
        assert len(cleaned_text) == len(pred_labels)
        result: list[str] = []
        open_tag = None

        for ch, lbl_id in zip(cleaned_text, pred_labels):
            tag = self.id2label[lbl_id]
            if tag.startswith("B-"):
                if open_tag:
                    result.append(f"</{open_tag}>")
                ent = tag.split("-", 1)[1].lower()
                open_tag = ent
                result.append(f"<{ent}>{ch}")
            elif tag.startswith("I-") and open_tag == tag.split("-", 1)[1].lower():
                result.append(ch)
            else:
                if open_tag:
                    result.append(f"</{open_tag}>")
                    open_tag = None
                result.append(ch)

        if open_tag:
            result.append(f"</{open_tag}>")
        return "".join(result)

    def _get_uncertain_spans(
        self,
        preds: list[int],
        confs: list[float],
    ) -> tuple[int, list[dict]]:
        """Group low‑confidence chars into spans, emitting entity names."""
        low_positions = [i for i, c in enumerate(confs) if c < self.review_threshold]
        low_count = len(low_positions)
        spans: list[dict] = []

        if not low_positions:
            return low_count, spans

        span_start = low_positions[0]
        span_label_id = preds[span_start]
        conf_list = [confs[span_start]]
        prev = span_start

        def _ent_name(lbl_id):
            lbl = self.id2label[lbl_id]
            if "-" in lbl:
                return lbl.split("-", 1)[1].lower()
            return lbl.lower()

        for pos in low_positions[1:]:
            if pos == prev + 1 and preds[pos] == span_label_id:
                conf_list.append(confs[pos])
                prev = pos
            else:
                spans.append(
                    {
                        "entity": _ent_name(span_label_id),
                        "start": span_start,
                        "end": prev,
                        "avg_confidence": sum(conf_list) / len(conf_list),
                    }
                )
                span_start = pos
                span_label_id = preds[pos]
                conf_list = [confs[pos]]
                prev = pos

        spans.append(
            {
                "entity": _ent_name(span_label_id),
                "start": span_start,
                "end": prev,
                "avg_confidence": sum(conf_list) / len(conf_list),
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
        Returns a list of (tagged, low_count, spans, chars) tuples.
        """
        cleaned_texts, preds_list, confs_list = self._batch_predict_full_texts(texts)
        results: list[dict[str, Union[str, int, list[dict], list[dict]]]] = []

        for idx, (cleaned, preds, confs) in enumerate(
            zip(cleaned_texts, preds_list, confs_list)
        ):
            tagged = self._pretty_print_ner_text(cleaned, preds)
            low_count, spans = self._get_uncertain_spans(preds, confs)

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
                for i, (ch, conf, pid) in enumerate(zip(cleaned, confs, preds))
                if conf < self.review_threshold
            ]

            if verbose:
                print(f"\n=== Text #{idx} ===")
                print(tagged)
                print(
                    f"\n[low‑confidence chars < {self.review_threshold}]: {low_count}"
                )
                if spans:
                    print("\nUncertain spans:")
                    for s in spans:
                        snippet = cleaned[s["start"] : s["end"] + 1].replace(
                            "\n", "\\n"
                        )
                        print(
                            f" - chars {s['start']}-{s['end']} "
                            f"({snippet}) as {s['entity']} "
                            f"(avg_conf={s['avg_confidence']:.2f})"
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


def main(mode="test"):

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
        text = [
            """\n\n(g) references herein to articles, sections, exhibits and schedules mean the articles and sections of, and the exhibits and schedules attached to, this Agreement; and\n\n(h) the words "hereof," "hereby," "herein," "hereunder" and similar terms in this Agreement refer to this Agreement as a whole and not only to a particular section in which such words appear.\n\nARTICLE II PURCHASE AND SALE\n\nSection 2.1 Purchase and Sale of Assets.\n\n(a) Generally. On the terms and subject to the conditions of this Agreement, Seller agrees to, and to cause the Companies to, assign, sell, transfer, convey and deliver to Buyer, and Buyer agrees to purchase from Seller and the Companies, all of Seller\'s and the Companies\' right, title and interest as of the Effective Time in the following property and assets (collectively, the "Assets"):\n\n(i) the real property listed on Exhibit E, together with all interests of Seller and the Companies in the buildings, structures, installations, fixtures, trade fixtures and other improvements situated thereon and all easements, rights of way and other rights, interests and appurtenances of Seller and the Companies therein or thereunto pertaining (collectively, "Owned Real Estate");\n\n(ii) the leasehold and subleasehold interests of Seller and the Companies in all real property listed on Exhibit F (collectively, "Leased Real Estate" and, together with the Owned Real Estate, the "Real Estate"), together with all interests of Seller and the Companies in the leases, subleases, licenses, occupancy agreements, and other documents or agreements related thereto and any and all interests of Seller and the Companies in the buildings, structures, installations, fixtures, trade fixtures and other improvements situated thereon and all easements, rights of way and other rights, interests and appurtenances of Seller and the Companies therein or thereunto pertaining (collectively with the Leased Real Estate, the "Leasehold Interests");\n\n(iii) the machinery, equipment, furniture, tools, computer hardware and network infrastructure and spare parts located on the Real Estate as of the Effective Time (exclusive of Inventory (which is defined in, and subject to, Section 2.1(a)(v)) (collectively, "Equipment") and all motor vehicles exclusively for use by Business Employees (excluding, for the avoidance of doubt, trucks, tractor-trailers and similar motor vehicles);\n\n(iv) all warranties or guarantees by any manufacturer, supplier or other vendor to the extent solely related to any of the Assets ("Warranties");\n\n(v) the inventory, packaging materials and supplies, in each case to the extent solely related to the Business and wherever located as of the Effective Time, and inventory, packaging materials and supplies on order or in transit as of the Effective\n\n11""",
            """but for the exception for contracts entered into in the ordinary course of business or (b) any non-competition agreement or any other agreement or obligation that materially limits or will materially limit Del Monte or any of its Subsidiaries from engaging in the business of Del Monte. Each of the "material contracts" (as defined above) of Del Monte and the Del Monte Subsidiaries is valid and in full force and effect and neither Del Monte nor any of its Subsidiaries has violated any provisions of, or committed or failed to perform any act that, with or without prejudice, lapse of time, or both, would constitute a default under the provisions of any such "material contract".

5.16 Brokers or Finders. No agent, broker, investment banker, financial advisor or other similar Person is or will be entitled, by reason of any agreement, act or statement by Del Monte or any of its Subsidiaries, directors, officers or employees, to any financial advisory, broker's, finder's or similar fee or commission from, to reimbursement of expenses by or to indemnification or contribution by, in each case, Del Monte or its Subsidiaries in connection with any of the transactions contemplated by this Agreement.

5.17 Board Approval. (a) The Boards of Directors of each of Del Monte and Merger Sub, in each case, at a meeting duly called and held, have unanimously approved this Agreement and declared it advisable and (b) the Board of Directors of Del Monte, at a meeting duly called and held, (i) has determined that this Agreement and the transactions contemplated hereby, including the Merger, taken together, are fair to, and in the best interests of, the Del Monte Stockholders, (ii) has resolved to recommend that the Del Monte Stockholders entitled to vote thereon adopt the Amended and Restated Certificate of Incorporation and approve the issuance of the Del Monte Common Stock in the Merger (the "Share Issuance"), subject to Section 6.4(c) (collectively, the "Del Monte Board Recommendation") and (iii) has determined that the Amended and Restated Certificate of Incorporation is advisable and fair to, and in the best interests of, the Del Monte Stockholders. The Special Committee has recommended that the Boards of Directors of Del Monte and Merger Sub approve this Agreement and the transactions contemplated hereby.

5.18 Vote Required. (a) The only vote of the Del Monte Stockholders required for (i) adoption of the Amended and Restated Certificate of Incorporation is the affirmative vote of a majority of the voting power of all outstanding shares of Del Monte Common Stock and (ii) the Share Issuance is, to the extent required by the applicable regulations of the NYSE, the affirmative vote of a majority of the voting power of the shares of Del Monte Common Stock present in person and voting on the issue or represented by proxy and voting on the issue at the Del Monte Stockholders Meeting (the "Share Issuance Approval") (together, sometimes referred to herein as the "Requisite Approval").

(b) The affirmative vote of Del Monte, as the sole stockholder of Merger Sub, is the only vote of the holders of the capital stock of Merger Sub necessary to adopt this Agreement.

5.19 Certain Payments. No Del Monte Benefit Plan and no other contractual arrangements between Del Monte and any third party exist that will, as a result of the transactions contemplated hereby and by the other Transaction Agreements, (a) result in the payment (or increase of any payment) by Del Monte or any of its Subsidiaries to any current,

44""",
        ]

        inference_model = NERInference(
            ckpt_path=NER_CKPT_PATH, label_list=NER_LABEL_LIST, review_threshold=0.99
        )

        start = time.time()
        tagged_result = inference_model.label(text, verbose=True)
        print(time.time() - start)

    else:
        raise RuntimeError(f"Invalid value for mode: {mode}")


if __name__ == "__main__":
    main()
