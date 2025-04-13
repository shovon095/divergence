from typing import Tuple, Dict, List, Optional
import numpy as np
import torch
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
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from dataclasses import dataclass, field
import logging
import os
import sys

# -----------------------------
# Assuming your utilities exist:
#   get_labels, NerDataset, Split
# -----------------------------
from utils_ner import get_labels, NerDataset, Split

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class DataTrainingArguments:
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. "
                    "Sequences longer than this will be truncated, shorter ones padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

# ----------------------------
# Align predictions to tokens
# ----------------------------
def align_predictions(
    predictions: np.ndarray,
    label_ids: np.ndarray,
    label_map: Dict[int, str]
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Convert argmax predictions and label IDs to string labels,
    ignoring subword or special tokens with label_id == ignore_index.
    """
    preds = np.argmax(predictions, axis=2)  # shape: (batch_size, seq_len)
    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list, out_label_list

# ----------------------------
# Compute SeqEval metrics
# ----------------------------
def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    # We'll use the global label_map from main()
    preds_list, out_label_list = align_predictions(
        p.predictions, p.label_ids, label_map
    )
    # Overall metrics
    precision = precision_score(out_label_list, preds_list)
    recall = recall_score(out_label_list, preds_list)
    f1 = f1_score(out_label_list, preds_list)

    # Per-label detailed report
    per_label_report = classification_report(
        out_label_list, preds_list, output_dict=True
    )
    logger.info("\n***** Per-Label Report *****")
    for lbl, vals in per_label_report.items():
        if isinstance(vals, dict):
            logger.info(
                f"Label: {lbl} | "
                f"Precision={vals['precision']:.4f}, "
                f"Recall={vals['recall']:.4f}, "
                f"F1={vals['f1-score']:.4f}"
            )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    # ----------------------------------------------------
    # Load labels
    # ----------------------------------------------------
    global label_map
    labels = get_labels(data_args.labels)
    label_map = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    # ----------------------------------------------------
    # Load config, tokenizer, model
    # ----------------------------------------------------
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # ----------------------------------------------------
    # Load datasets
    # ----------------------------------------------------
    train_dataset, eval_dataset, test_dataset = None, None, None

    if training_args.do_train:
        train_dataset = NerDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
    if training_args.do_eval:
        eval_dataset = NerDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
    if training_args.do_predict:
        test_dataset = NerDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
        )

    # ----------------------------------------------------
    # Initialize Trainer
    # ----------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # ----------------------------------------------------
    # Training
    # ----------------------------------------------------
    if training_args.do_train:
        trainer.train(
            model_path=(
                model_args.model_name_or_path
                if os.path.isdir(model_args.model_name_or_path)
                else None
            )
        )
        # Save final model + tokenizer
        if trainer.is_world_process_zero():
            model.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)

    # ----------------------------------------------------
    # Evaluation
    # ----------------------------------------------------
    if training_args.do_eval and eval_dataset is not None:
        logger.info("*** Evaluate ***")
        result = trainer.evaluate()
        # Save and log results
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))
            logger.info("\nPrecision: %.4f, Recall: %.4f, F1: %.4f",
                        result["eval_precision"], result["eval_recall"], result["eval_f1"])

    # ----------------------------------------------------
    # Prediction
    # ----------------------------------------------------
    if training_args.do_predict and test_dataset is not None:
        logger.info("*** Test ***")
        predictions, label_ids, metrics = trainer.predict(test_dataset)
        logger.info("***** Test results *****")
        for key, value in metrics.items():
            logger.info("  %s = %s", key, value)

        # Save test metrics
        output_test_results_file = os.path.join(training_args.output_dir, "test_results.txt")
        if trainer.is_world_process_zero():
            with open(output_test_results_file, "w") as writer:
                for key, value in metrics.items():
                    writer.write(f"{key} = {value}\n")

            # Optionally align predictions and save them
            preds_list, _ = align_predictions(predictions, label_ids, label_map)
            output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
            with open(output_test_predictions_file, "w") as writer:
                # Example: each line is a list of predicted tags
                for pred_seq in preds_list:
                    writer.write(" ".join(pred_seq) + "\n")

def _mp_fn(index):
    main()

if __name__ == "__main__":
    main()



