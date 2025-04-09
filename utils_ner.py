# utils_ner.py
# ----------------------------------------
# A modernized utility for Named Entity Recognition,
# adapted to include a model_type parameter.
# ----------------------------------------

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from transformers import PreTrainedTokenizer
from filelock import FileLock

import torch
from torch import nn
from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------
# (A) Data classes
# --------------------------------------------------------------------
@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word.
    """
    guid: str
    words: List[str]
    labels: Optional[List[str]] = None


@dataclass
class InputFeatures:
    """
    A single set of features of data.

    The fields correspond to model inputs.
    """
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    valid_ids: Optional[List[int]] = None


class Split(Enum):
    train = "train_dev"
    dev = "devel"
    test = "test"


# --------------------------------------------------------------------
# (B) Reading raw data: CoNLL-style
# --------------------------------------------------------------------
def read_examples_from_file(data_dir: str, mode: Union[Split, str]) -> List[InputExample]:
    """
    Reads a file (e.g. "train_dev.txt", "devel.txt", "test.txt") in CoNLL format:
      Each line: "token [whitespace] label"
      Blank lines separate sentences.
    """
    if isinstance(mode, Split):
        mode = mode.value  # e.g. "train_dev", "devel", "test"

    file_path = os.path.join(data_dir, f"{mode}.txt")
    logger.info(f"Reading {mode} data from {file_path}")

    examples = []
    guid_index = 1
    words: List[str] = []
    labels: List[str] = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-DOCSTART-"):
                if words:
                    examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split()
                words.append(splits[0])
                labels.append(splits[-1] if len(splits) > 1 else "O")
        if words:
            examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
    return examples

def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    cls_token_at_end: bool = False,
    cls_token: str = "[CLS]",
    cls_token_segment_id: int = 0,
    sep_token: str = "[SEP]",
    sep_token_extra: bool = False,
    pad_on_left: bool = False,
    pad_token: int = 0,
    pad_token_segment_id: int = 0,
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index,
    sequence_a_segment_id: int = 0,
) -> List[InputFeatures]:
    """
    Converts a list of InputExamples into InputFeatures.
    For each original word, tokenizes it into sub-tokens.
    - The first sub-token gets the true label and is marked valid (valid_id 1).
    - Subsequent sub-tokens get the pad label (ignore_index) and are marked invalid (valid_id 0).

    Special tokens ([CLS] and [SEP]) are added according to the model requirements.
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for (ex_index, example) in enumerate(examples):
        tokens = []
        label_ids = []
        valid_ids = []

        # Iterate over each word and its label
        for word, label in zip(example.words, example.labels):
            # Tokenize the word into sub-tokens
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [tokenizer.unk_token]
            # Extend tokens with the tokenized word
            tokens.extend(word_tokens)
            # Assign the real label for the first token and pad token label for rest
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
            # Mark the first token as valid (1) and others as invalid (0)
            valid_ids.extend([1] + [0] * (len(word_tokens) - 1))

        # Truncate if needed: account for special tokens
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: max_seq_length - special_tokens_count]
            label_ids = label_ids[: max_seq_length - special_tokens_count]
            valid_ids = valid_ids[: max_seq_length - special_tokens_count]

        # Add special tokens and segment_ids accordingly
        if cls_token_at_end:
            # For models that put [CLS] at the end (e.g. XLNet)
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            valid_ids += [0]
            if sep_token_extra:
                tokens += [sep_token]
                label_ids += [pad_token_label_id]
                valid_ids += [0]
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            valid_ids += [0]
            segment_ids = [sequence_a_segment_id] * len(tokens)
        else:
            # For standard models (BERT, RoBERTa, etc.) with [CLS] at the beginning
            tokens = [cls_token] + tokens + [sep_token]
            label_ids = [pad_token_label_id] + label_ids + [pad_token_label_id]
            valid_ids = [0] + valid_ids + [0]
            if sep_token_extra:
                tokens = tokens[:-1] + [sep_token] + tokens[-1:]
                label_ids = label_ids[:-1] + [pad_token_label_id] + label_ids[-1:]
                valid_ids = valid_ids[:-1] + [0] + valid_ids[-1:]
            segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)

        # Convert tokens to input IDs and create attention mask
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Pad the inputs up to max_seq_length
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            valid_ids = ([0] * padding_length) + valid_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
            valid_ids += [0] * padding_length

        # Sanity checks
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid_ids) == max_seq_length

        # Log first few examples for debugging
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join(tokens))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            logger.info("valid_ids: %s", " ".join([str(x) for x in valid_ids]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                label_ids=label_ids,
                valid_ids=valid_ids
            )
        )
    return features


# --------------------------------------------------------------------
# (D) Final PyTorch dataset
# --------------------------------------------------------------------
class NerDataset(Dataset):
    """
    A dataset for NER that uses the above conversion logic.
    """

    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        labels: List[str],
        model_type: str,
        max_seq_length: int,
        overwrite_cache: bool = False,
        mode: Split = Split.train,
    ):
        # Cache file name includes model_type so that if you switch models (e.g., from bert to xlnet),
        # the cached features do not conflict.
        cached_features_file = os.path.join(
            data_dir,
            f"cached_{mode.value}_{tokenizer.__class__.__name__}_{model_type}_{max_seq_length}",
        )

        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                examples = read_examples_from_file(data_dir, mode)
                # Determine special token handling based on model_type
                cls_token_at_end = model_type in ["xlnet"]
                cls_token = tokenizer.cls_token
                sep_token = tokenizer.sep_token
                cls_token_segment_id = 2 if model_type in ["xlnet"] else 0

                self.features = convert_examples_to_features(
                    examples,
                    label_list=labels,
                    max_seq_length=max_seq_length,
                    tokenizer=tokenizer,
                    cls_token_at_end=cls_token_at_end,
                    cls_token=cls_token,
                    cls_token_segment_id=cls_token_segment_id,
                    sep_token=sep_token,
                    sep_token_extra=False,
                    pad_on_left=bool(tokenizer.padding_side == "left"),
                    pad_token=tokenizer.pad_token_id,
                    pad_token_segment_id=tokenizer.pad_token_type_id,
                    pad_token_label_id=self.pad_token_label_id,
                )
                logger.info(f"Saving features into cached file {cached_features_file}")
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        return {
        "input_ids": torch.tensor(feature.input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(feature.attention_mask, dtype=torch.long),
        "token_type_ids": torch.tensor(feature.token_type_ids, dtype=torch.long),
        "labels": torch.tensor(feature.label_ids, dtype=torch.long),
        "valid_ids": torch.tensor(feature.valid_ids, dtype=torch.long),  # ✅ Add this
    }

    


def get_labels(path: Optional[str] = None) -> List[str]:
    """
    Reads a file of labels (one per line) and returns a list.
    If no path is provided or the file doesn't exist, returns a default label set.
    """
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-Disposition", "I-Disposition", "B-NoDisposition", "I-NoDisposition", "B-Undetermined", "I-Undetermined"]


