# run_ner.py – Python 3.8-compatible training / eval / predict script
# plus the NER utilities in the same file for simplicity.

import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from filelock import FileLock
from seqeval.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizer,
    set_seed,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. NER Utility Classes and Functions
# ---------------------------------------------------------------------------

@dataclass
class InputExample:
    """A single training/test example for token classification."""
    guid: str
    words: List[str]
    labels: Optional[List[str]] = None

@dataclass
class InputFeatures:
    """A single set of features (input_ids etc.) for an example."""
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    label_ids: List[int]
    valid_ids: List[int]

class Split(Enum):
    train = "train_dev"
    dev = "devel"
    test = "test"

def read_examples_from_file(data_dir: str, mode: Union[Split, str]) -> List[InputExample]:
    """Reads a CoNLL-style text file and returns a list of InputExample."""
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, f"{mode}.txt")
    logger.info("Reading %s data from %s", mode, file_path)

    examples: List[InputExample] = []
    words: List[str] = []
    labels: List[str] = []
    guid_idx = 1

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            # Blank line or -DOCSTART- signals start of new example
            if not line or line.startswith("-DOCSTART-"):
                if words:
                    examples.append(InputExample(
                        guid=f"{mode}-{guid_idx}",
                        words=words,
                        labels=labels
                    ))
                    guid_idx += 1
                    words, labels = [], []
            else:
                splits = line.split()
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1])
                else:
                    labels.append("O")
        # If the last row isn't blank, push the final example
        if words:
            examples.append(InputExample(
                guid=f"{mode}-{guid_idx}",
                words=words,
                labels=labels
            ))
    return examples

def convert_examples_to_features(
    examples: Sequence[InputExample],
    label_list: Sequence[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index,
) -> List[InputFeatures]:
    """
    Converts InputExamples into InputFeatures, tokenizing words.
    Subtokens get label_id = pad_token_label_id except the first subtoken (valid_ids=1).
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features: List[InputFeatures] = []

    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    cls_segment_id = 0

    for example in examples:
        tokens: List[str] = []
        label_ids: List[int] = []
        valid_ids: List[int] = []

        # Tokenize each word
        if example.labels is None:
            example_labels = ["O"] * len(example.words)
        else:
            example_labels = example.labels

        for word, label in zip(example.words, example_labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [tokenizer.unk_token]
            # first subtoken gets the real label; rest get pad_label
            tokens.extend(word_tokens)
            label_ids.append(label_map[label])
            label_ids.extend([pad_token_label_id] * (len(word_tokens) - 1))
            # valid_ids: 1 on first subtoken, 0 on subsequent
            valid_ids.append(1)
            valid_ids.extend([0] * (len(word_tokens) - 1))

        # Possibly truncate for special tokens
        special_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_count:
            tokens = tokens[: max_seq_length - special_count]
            label_ids = label_ids[: max_seq_length - special_count]
            valid_ids = valid_ids[: max_seq_length - special_count]

        # Insert [CLS], [SEP]
        tokens = [cls_token] + tokens + [sep_token]
        label_ids = [pad_token_label_id] + label_ids + [pad_token_label_id]
        valid_ids = [0] + valid_ids + [0]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [cls_segment_id] + [0] * (len(tokens) - 1)

        # Pad up to max_seq_length
        pad_length = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_length
        attention_mask += [0] * pad_length
        token_type_ids += [tokenizer.pad_token_type_id] * pad_length
        label_ids += [pad_token_label_id] * pad_length
        valid_ids += [0] * pad_length

        features.append(InputFeatures(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            label_ids=label_ids,
            valid_ids=valid_ids,
        ))

    return features

class NerDataset(Dataset):
    """
    PyTorch dataset for NER, caching the (features, examples) locally
    so we can do alignment for final predictions.
    """
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        labels: Sequence[str],
        max_seq_length: int,
        mode: Split = Split.train,
        overwrite_cache: bool = False,
    ) -> None:
        cache_file = os.path.join(
            data_dir,
            f"cached_{mode.value}_{tokenizer.__class__.__name__}_{max_seq_length}"
        )
        lock_file = cache_file + ".lock"
        with FileLock(lock_file):
            if os.path.exists(cache_file) and not overwrite_cache:
                logger.info("Loading dataset from %s", cache_file)
                bundle = torch.load(cache_file)
                self.features = bundle["features"]
                self.examples = bundle["examples"]
            else:
                logger.info("Reading and converting data for %s split", mode.value)
                self.examples = read_examples_from_file(data_dir, mode)
                self.features = convert_examples_to_features(
                    self.examples,
                    labels,
                    max_seq_length,
                    tokenizer,
                    pad_token_label_id=self.pad_token_label_id,
                )
                torch.save({"features": self.features, "examples": self.examples}, cache_file)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f = self.features[idx]
        return {
            "input_ids": torch.tensor(f.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(f.attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(f.token_type_ids, dtype=torch.long),
            "labels": torch.tensor(f.label_ids, dtype=torch.long),
            "valid_ids": torch.tensor(f.valid_ids, dtype=torch.long),
        }

def get_labels(path: Optional[str] = None) -> List[str]:
    """Load or default to some standard label set if no file is found."""
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if "O" not in lines:
            lines = ["O"] + lines
        return lines
    # Otherwise default:
    return [
        "O",
        "B-Disposition", "I-Disposition",
        "B-NoDisposition", "I-NoDisposition",
        "B-Undetermined", "I-Undetermined",
    ]

# ---------------------------------------------------------------------------
# 2. The main script: train/eval/predict
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "HF model name or path"})
    cache_dir: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    data_dir: str = field(metadata={"help": "Folder with train_dev.txt/devel.txt/test.txt"})
    labels: Optional[str] = field(default=None)
    max_seq_length: int = field(default=128)
    overwrite_cache: bool = field(default=False)

def align_predictions(
    predictions: np.ndarray,
    label_ids: np.ndarray,
    valid_ids: np.ndarray,
    id2label: Dict[int, str],
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Align predictions to tokens by ignoring subtoken positions (valid_ids=0).
    """
    preds = predictions.argmax(-1)
    batch_preds: List[List[str]] = []
    batch_labels: List[List[str]] = []

    for p_row, l_row, v_row in zip(preds, label_ids, valid_ids):
        sent_preds: List[str] = []
        sent_labels: List[str] = []
        for p_val, l_val, v_val in zip(p_row, l_row, v_row):
            if v_val == 1:
                sent_preds.append(id2label[int(p_val)])
                sent_labels.append(id2label[int(l_val)])
        batch_preds.append(sent_preds)
        batch_labels.append(sent_labels)
    return batch_preds, batch_labels

def build_compute_metrics(id2label: Dict[int, str]):
    """
    Returns a function that can be used as Trainer's compute_metrics argument.
    """
    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        # 'valid_ids' come from the model input
        valid_ids = eval_pred.inputs["valid_ids"]

        preds_list, labels_list = align_predictions(predictions, labels, valid_ids, id2label)
        return {
            "precision": precision_score(labels_list, preds_list),
            "recall": recall_score(labels_list, preds_list),
            "f1": f1_score(labels_list, preds_list)
        }
    return compute_metrics

def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # Possibly parse json file
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Check for existing output dir
    if (
        os.path.isdir(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError("Output directory already exists and is not empty. Use --overwrite_output_dir to proceed.")

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if training_args.local_rank in (-1, 0) else logging.WARN,
    )

    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(training_args.seed)

    # 1) Load labels
    labels = get_labels(data_args.labels)
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}

    # 2) Load model config & tokenizer
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        cache_dir=model_args.cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
        cache_dir=model_args.cache_dir
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir
    )

    # 3) Prepare datasets
    def maybe_load(split: Optional[Split]) -> Optional[NerDataset]:
        if split is None:
            return None
        return NerDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            max_seq_length=data_args.max_seq_length,
            mode=split,
            overwrite_cache=data_args.overwrite_cache
        )

    train_dataset = maybe_load(Split.train) if training_args.do_train else None
    eval_dataset = maybe_load(Split.dev) if training_args.do_eval else None
    test_dataset = maybe_load(Split.test) if training_args.do_predict else None

    # 4) Setup Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics(id2label),
    )

    # 5) Train
    if training_args.do_train:
        trainer.train()
        # Save final
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

    # 6) Evaluate
    if training_args.do_eval and eval_dataset is not None:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        logger.info("Eval results:")
        for k, v in metrics.items():
            logger.info("  %s = %s", k, v)

    # 7) Predict
    if training_args.do_predict and test_dataset is not None:
        logger.info("*** Test ***")
        preds, label_ids, metrics = trainer.predict(test_dataset)
        logger.info("Test metrics:")
        for k, v in metrics.items():
            logger.info("  %s = %s", k, v)

        valid_ids = np.stack([f.valid_ids for f in test_dataset.features], axis=0)
        preds_list, _ = align_predictions(preds, label_ids, valid_ids, id2label)

        # Write CoNLL-style predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        with open(output_test_predictions_file, "w", encoding="utf-8") as writer:
            for example, pred_seq in zip(test_dataset.examples, preds_list):
                # If there's a mismatch, handle carefully
                if len(example.words) != len(pred_seq):
                    logger.warning(
                        "Mismatch between example %s (words=%d) and prediction (labels=%d). "
                        "Truncation or subtoken mismatch may have occurred.",
                        example.guid,
                        len(example.words),
                        len(pred_seq)
                    )
                for word, label in zip(example.words, pred_seq):
                    writer.write(f"{word}\t{label}\n")
                writer.write("\n")
        logger.info("Wrote predictions to %s", output_test_predictions_file)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

