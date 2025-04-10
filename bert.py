from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from utils_ner import NerDataset, Split, get_labels

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Argument groups
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "HF model name or path"})
    cache_dir: str | None = field(default=None)

@dataclass
class DataArguments:
    data_dir: str = field(metadata={"help": "Directory with train_dev.txt / devel.txt / test.txt"})
    labels: str | None = field(default=None)
    max_seq_length: int = field(default=128)
    overwrite_cache: bool = field(default=False)

# ---------------------------------------------------------------------------
# Helper: align predictions with words
# ---------------------------------------------------------------------------

def align_predictions(predictions: np.ndarray, label_ids: np.ndarray, valid_ids: np.ndarray, id2label: Dict[int, str]) -> Tuple[List[List[str]], List[List[str]]]:
    preds = predictions.argmax(-1)  # (bs, seq_len)

    batch_preds, batch_labels = [], []
    for p, l, v in zip(preds, label_ids, valid_ids):
        sent_preds, sent_labels = [], []
        for pi, li, vi in zip(p, l, v):
            if vi == 1:  # keep only first sub‑token of each word
                sent_preds.append(id2label[int(pi)])
                sent_labels.append(id2label[int(li)])
        batch_preds.append(sent_preds)
        batch_labels.append(sent_labels)
    return batch_preds, batch_labels

# ---------------------------------------------------------------------------
# Metrics callback for Trainer
# ---------------------------------------------------------------------------

def build_compute_metrics(id2label: Dict[int, str]):
    def compute_metrics(p: EvalPrediction):  # type: ignore
        preds_list, labels_list = align_predictions(
            p.predictions, p.label_ids, p.inputs["valid_ids"], id2label
        )
        return {
            "precision": precision_score(labels_list, preds_list),
            "recall": recall_score(labels_list, preds_list),
            "f1": f1_score(labels_list, preds_list),
        }
    return compute_metrics

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Safety check ----------------------------------------------------------------
    if (os.path.isdir(training_args.output_dir) and os.listdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir):
        raise ValueError(f"Output dir {training_args.output_dir} already exists. Use --overwrite_output_dir to proceed.")

    # Logging ----------------------------------------------------------------------
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if training_args.local_rank in (-1, 0) else logging.WARN,
    )

    set_seed(training_args.seed)

    # Labels -----------------------------------------------------------------------
    labels = get_labels(data_args.labels)
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}

    # Model / tokenizer ------------------------------------------------------------
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=len(labels), id2label=id2label, label2id=label2id, cache_dir=model_args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True, cache_dir=model_args.cache_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir)

    # Datasets ---------------------------------------------------------------------
    def load_dataset(split: Split | None):
        if split is None:
            return None
        return NerDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            max_seq_length=data_args.max_seq_length,
            mode=split,
            overwrite_cache=data_args.overwrite_cache,
        )

    train_ds = load_dataset(Split.train) if training_args.do_train else None
    dev_ds = load_dataset(Split.dev) if training_args.do_eval else None
    test_ds = load_dataset(Split.test) if training_args.do_predict else None

    # Trainer ----------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=build_compute_metrics(id2label),
    )

    # Training ---------------------------------------------------------------------
    if training_args.do_train:
        trainer.train()
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation -------------------------------------------------------------------
    if training_args.do_eval and dev_ds is not None:
        logger.info("***** Dev Evaluation *****")
        metrics = trainer.evaluate()
        for k, v in metrics.items():
            logger.info("%s = %.4f", k, v)

    # Prediction -------------------------------------------------------------------
    if training_args.do_predict and test_ds is not None:
        logger.info("***** Test Prediction *****")
        predictions, label_ids, _ = trainer.predict(test_ds)
        valid_ids = np.stack([f.valid_ids for f in test_ds.features])  # shape (N, seq_len)
        preds_list, _ = align_predictions(predictions, label_ids, valid_ids, id2label)

        out_path = os.path.join(training_args.output_dir, "test_predictions.txt")
        with open(out_path, "w", encoding="utf‑8") as w:
            for ex, pred in zip(test_ds.examples, preds_list):
                if len(ex.words) != len(pred):
                    logger.warning("Length mismatch for %s (words=%d, preds=%d)", ex.guid, len(ex.words), len(pred))
                for tok, lab in zip(ex.words, pred):
                    w.write(f"{tok}\t{lab}\n")
                w.write("\n")
        logger.info("Wrote predictions to %s", out_path)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
