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
#from utils_ner import get_labels, NerDataset, Split

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



