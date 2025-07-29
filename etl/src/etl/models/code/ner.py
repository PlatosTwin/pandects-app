# Standard Library
import os
import time

# Data classes
from etl.models.code.ner_classes import NERTagger, NERDataModule
from etl.models.code.constants import NER_LABEL_LIST, NER_CKPT_PATH

# Environment config
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Third-Party Libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# PyTorch & Lightning
import torch
from transformers import AutoTokenizer

torch.set_float32_matmul_precision("high")

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything

seed_everything(42, workers=True, verbose=False)

from optuna import create_study
from optuna.integration import PyTorchLightningPruningCallback

from transformers import AutoTokenizer


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
        print(df.head())
        print(f"Tagged value counts:\n{df['tagged'].value_counts()}")

        # for now, remove untagged pages
        df = df[df["tagged"] == 1]

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
        return checkpoint_cb, early_stop_cb, lr_monitor, pruning_cb

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
        checkpoint_cb, early_stop_cb, lr_monitor, pruning_cb = self._get_callbacks(
            trial
        )

        trainer = pl.Trainer(
            max_epochs=self.MAX_EPOCHS,
            accelerator=self.DEVICE,
            precision="bf16",
            devices=1,
            logger=TensorBoardLogger("tb_logs", name="optuna"),
            callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, *pruning_cb],
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
        checkpoint_cb, early_stop_cb, lr_monitor, _ = self._get_callbacks(trial=None)

        trainer = pl.Trainer(
            max_epochs=self.MAX_EPOCHS,
            accelerator=self.DEVICE,
            devices=1,
            logger=TensorBoardLogger("tb_logs", name="final"),
            callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
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

    def _predict_full_text(
        self,
        text: str,
        window: int = 512,
        stride: int = 256,
    ) -> tuple[str, list[int], list[float]]:
        """
        1) Break text into overlapping char-windows,
        2) Tokenize & run them through the model,
        3) Aggregate token-probs down to chars,
        4) Argmax → raw char-level B/I/O labels,
        5) Post-process BIO so only the *first* char of a span is B-,
           the rest become I-.
        """
        T = len(text)
        # build (start,end) windows in char-space
        windows: list[tuple[int, int]] = []
        slices: list[str] = []
        start = 0
        while start < T:
            end = min(start + window, T)
            windows.append((start, end))
            slices.append(text[start:end])
            if end == T:
                break
            start += stride

        # tokenize batch
        enc = self.tokenizer(
            slices,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=window,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
        # move inputs to device
        for k in ("input_ids", "attention_mask", "offset_mapping"):
            enc[k] = enc[k].to(self.device)

        # forward
        with torch.no_grad():
            out = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
            probs = torch.softmax(out.logits, dim=-1)  # (batch, seq, labels)

        # aggregate down to char-level
        sum_p = torch.zeros((T, probs.size(-1)), device=self.device)
        counts = torch.zeros(T, device=self.device)
        for (w0, w1), p, offsets, attn in zip(
            windows, probs, enc["offset_mapping"], enc["attention_mask"]
        ):
            valid = attn.sum().item()  # skip padding
            for tok_i in range(valid):
                off_start, off_end = offsets[tok_i].tolist()
                if off_end == 0:
                    continue
                # add this token's prob to each char in its span
                for char_pos in range(off_start, off_end):
                    idx = w0 + char_pos
                    if idx < T:
                        sum_p[idx] += p[tok_i]
                        counts[idx] += 1

        # average & argmax
        avg_p = sum_p / counts.unsqueeze(-1).clamp(min=1)
        confidences = avg_p.max(dim=-1)[0].cpu().tolist()
        raw_preds = avg_p.argmax(dim=-1).cpu().tolist()

        # BIO fix: convert B-X→I-X if the *previous* char was the same X
        for i in range(1, len(raw_preds)):
            prev_lbl = self.id2label[raw_preds[i - 1]]
            curr_lbl = self.id2label[raw_preds[i]]
            if curr_lbl.startswith("B-"):
                ent = curr_lbl.split("-", 1)[1]
                if prev_lbl == f"I-{ent}" or prev_lbl == f"B-{ent}":
                    raw_preds[i] = self.label2id[f"I-{ent}"]

        return text, raw_preds, confidences

    def _pretty_print_ner_text(
        self,
        cleaned_text: str,
        pred_labels: list[int],
    ) -> str:
        """
        Same as before — but now your pred_labels will only
        have a single B- at the start of each span, so you
        won't see per-character tagging noise.
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
        self, preds: list[int], confs: list[float]
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
            return lbl.lower()  # for "O" → "o" (or call it "outside")

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

        # flush last
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
        self, text: str, verbose: bool = False
    ) -> tuple[str, int, list[dict], list[dict]]:
        cleaned, preds, confs = self._predict_full_text(text)
        tagged = self._pretty_print_ner_text(cleaned, preds)
        low_count, spans = self._get_uncertain_spans(preds, confs)

        # we still build chars if you need them programmatically
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
            print(tagged)
            print(f"\n[low‑confidence chars < {self.review_threshold}]: {low_count}")

            if spans:
                print("\nUncertain spans:")
                for s in spans:
                    # grab the actual substring, escape newlines for readability
                    snippet = cleaned[s["start"] : s["end"] + 1].replace("\n", "\\n")
                    print(
                        f" - chars {s['start']}-{s['end']} "
                        f"({snippet}) as {s['entity']} "
                        f"(avg_conf={s['avg_confidence']:.2f})"
                    )
            else:
                print("No spans below threshold.")

        return tagged, low_count, spans, chars


def main():

    # ner_trainer = NERTrainer(
    #     data_csv="../data/ner-data.csv",
    #     model_name="answerdotai/ModernBERT-base",
    #     label_list=NER_LABEL_LIST,
    #     num_trials=10,
    #     max_epochs=10,
    # )
    # ner_trainer.run()

    # quick test
    text = """
Section 5.9 Securityholder Litigation. Each of ETP and ETE shall give the other the opportunity to participate in the defense or settlement of any securityholder litigation against such party and/or its officers and directors relating to the transactions contemplated hereby; provided that the party subject to the litigation shall in any event control such defense and/or settlement (subject to Section 5.2(a)(xii) and Section 5.2(b)(viii)) and shall not be required to provide information if doing so would be reasonably expected to threaten the loss of any attorney-client privilege or other applicable legal privilege. 

Section 5.10 Financing Matters. ETP hereby consents to ETE’s use of and reliance on any audited or unaudited financial statements relating to ETP and its consolidated Subsidiaries, any ETP Joint Ventures or entities or businesses acquired by ETP reasonably requested by ETE to be used in any financing or other activities of ETE, including any filings that ETE desires to make with the SEC. In addition, ETP will use commercially reasonable efforts, at ETE’s sole cost and expense, to obtain the consents of any auditor to the inclusion of the financial statements referenced above in appropriate filings with the SEC. Prior to the Closing, ETP will provide such assistance (and will cause its Subsidiaries and its and their respective personnel and advisors to provide such assistance), as ETE may reasonably request in order to assist ETE in connection with financing activities, including any public offerings to be registered under the Securities Act or private offerings. Such assistance shall include, but not be limited to, the following: (i) providing such information, and making available such personnel as ETE may reasonably request; (ii) participation in, and assistance with, any marketing activities related to such financing; (iii) participation by senior management of ETP in, and their assistance with, the preparation of rating agency presentations and meetings with rating agencies; (iv) taking such actions as are reasonably requested by ETE or its financing sources to facilitate the satisfaction of all conditions precedent to obtaining such financing; and (v) taking such actions as may be required to permit any cash and marketable securities of ETP or ETE to be made available to finance the transactions contemplated hereby at the Effective Time. 

Section 5.11 Fees and Expenses. All fees and expenses incurred in connection with the transactions contemplated hereby including all legal, accounting, financial advisory, consulting and all other fees and expenses of third parties incurred by a party in connection with the negotiation and effectuation of the terms and conditions of this Agreement and the transactions contemplated hereby, shall be the obligation of the respective party incurring such fees and expenses (other than the filing fee payable to the SEC in connection with the Registration Statement and the filing fee payable in connection with the filing of a Notification and Report Form pursuant to the HSR Act, which shall each be borne one half by ETP and one half by ETE). 

Section 5.12 Section 16 Matters. Prior to the Effective Time, ETE and ETP shall take all such steps as may be required (to the extent permitted under applicable Law) to cause any dispositions of Common Units (including derivative securities with respect to Common Units) or acquisitions of ETE Common Units (including derivative securities with respect to ETE Common Units) resulting from the transactions contemplated by this Agreement by each individual who is subject to the reporting requirements of Section 16(a) of the Exchange Act with respect to ETP, or will become subject to such reporting requirements with respect to ETE, to be exempt under Rule 16b-3 promulgated under the Exchange Act. 

52"""

    inference_model = NERInference(
        ckpt_path=NER_CKPT_PATH, label_list=NER_LABEL_LIST, review_threshold=0.99
    )

    start = time.time()
    inference_model.label(text, verbose=True)
    print(time.time() - start)


if __name__ == "__main__":
    main()
