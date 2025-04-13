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

# Local utility for NER tasks
from utils_ner import get_labels, NerDataset, Split

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    use_fast: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)


@dataclass
class DataTrainingArguments:
    data_dir: str = field(metadata={"help": "The input data dir."})
    labels: Optional[str] = field(default=None)
    max_seq_length: int = field(default=128)
    overwrite_cache: bool = field(default=False)


def align_predictions(
    predictions: np.ndarray,
    label_ids: np.ndarray,
    label_map: Dict[int, str]
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Converts raw model logits into label strings, ignoring '-100' (mask for padded tokens).
    Returns:
      preds_list: list of list of predicted labels
      out_label_list: list of list of true labels (for metric computation)
    """
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list, out_label_list


def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    """
    Reports standard token-level metrics plus an optional classification report.
    """
    preds_list, out_label_list = align_predictions(p.predictions, p.label_ids, label_map)
    precision = precision_score(out_label_list, preds_list)
    recall = recall_score(out_label_list, preds_list)
    f1 = f1_score(out_label_list, preds_list)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    # Parse script arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    # ------------------------------------------------------------------
    # Prepare labels
    # ------------------------------------------------------------------
    global label_map  # So compute_metrics can see it
    labels = get_labels(data_args.labels)  # Read from file if provided
    label_map = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    # ------------------------------------------------------------------
    # Load config, tokenizer, and model
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Initialize Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    if training_args.do_train:
        trainer.train(
            model_path=(model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None)
        )
        if trainer.is_world_process_zero():
            model.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    if training_args.do_eval and eval_dataset is not None:
        logger.info("*** Evaluate ***")
        result = trainer.evaluate()
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                for key, value in result.items():
                    writer.write(f"{key} = {value}\n")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    if training_args.do_predict and test_dataset is not None:
        logger.info("*** Predict (Test) ***")
        predictions, label_ids, metrics = trainer.predict(test_dataset)

        # Print and save metrics
        logger.info("***** Test results *****")
        for key, value in metrics.items():
            logger.info("  %s = %s", key, value)

        output_test_results_file = os.path.join(training_args.output_dir, "test_results.txt")
        if trainer.is_world_process_zero():
            with open(output_test_results_file, "w") as writer:
                for key, value in metrics.items():
                    writer.write(f"{key} = {value}\n")

            # ------------------------------------------
            # Align predictions -> list of label strings
            # ------------------------------------------
            preds_list, _ = align_predictions(predictions, label_ids, label_map)

            # -------------------------------------------------------------
            # Now write line-by-line predictions, mirroring your test.txt
            # -------------------------------------------------------------
            test_path = os.path.join(data_args.data_dir, "test.txt")
            output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions_ssjs.txt")

            if not os.path.exists(test_path):
                logger.warning(f"No test.txt file found in {data_args.data_dir}. Cannot write line-by-line predictions.")
            else:
                with open(test_path, "r", encoding="utf-8") as f_in, \
                     open(output_test_predictions_file, "w", encoding="utf-8") as writer:
                    logger.info(f"Writing predictions to {output_test_predictions_file} ...")

                    example_id = 0   # which 'sentence' or example from the dataset
                    token_id = 0     # index within that sentence's predictions

                    for line in f_in:
                        stripped = line.strip()
                        if not stripped:
                            # Blank line => new sentence / example
                            writer.write("\n")
                            example_id += 1
                            token_id = 0
                            continue

                        # For each non-blank line, take the first column as the "token"
                        # (if your file has more columns, adjust as needed)
                        token = stripped.split()[0]

                        # Grab the predicted label from preds_list
                        predicted_label = preds_list[example_id][token_id]
                        writer.write(f"{token} {predicted_label}\n")

                        token_id += 1

                logger.info(f"✅ Predictions saved to {output_test_predictions_file}")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()



