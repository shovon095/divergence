# ============================================================
# utils_ner.py – clean, modern Hugging Face NER utilities
# ============================================================

"""Utility classes & functions for token‑classification datasets.

Key features
------------
* CoNLL‑style reader (token  label, blank line between sentences).
* Word‑piece tokenisation with a **valid_ids** mask (1 for the
  first sub‑token of each word, 0 otherwise) so that downstream
  alignment is trivial.
* Caching of features to disk (one cache file per model type /
  max‑sequence‑length).
* Dataset keeps a copy of the **examples** list so that the original
  tokens can be recovered later (e.g. when writing predictions).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Union

import torch
from filelock import FileLock
from torch import nn
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class InputExample:
    guid: str
    words: List[str]
    labels: Optional[List[str]] = None


@dataclass
class InputFeatures:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    label_ids: List[int]
    valid_ids: List[int]


class Split(Enum):
    train = "train_dev"
    dev = "devel"
    test = "test"

# ---------------------------------------------------------------------------
# Raw‑file reader
# ---------------------------------------------------------------------------

def read_examples_from_file(data_dir: str, mode: Union[Split, str]) -> List[InputExample]:
    """Read a CoNLL‑formatted file and return a list of InputExample."""
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, f"{mode}.tsv")
    logger.info("Reading %s data from %s", mode, file_path)

    examples: list[InputExample] = []
    words, labels = [], []
    guid = 1

    with open(file_path, encoding="utf‑8") as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("-DOCSTART-"):
                if words:
                    examples.append(InputExample(guid=f"{mode}-{guid}", words=words, labels=labels))
                    guid += 1
                    words, labels = [], []
            else:
                splits = line.split()
                words.append(splits[0])
                labels.append(splits[-1] if len(splits) > 1 else "O")
        if words:
            examples.append(InputExample(guid=f"{mode}-{guid}", words=words, labels=labels))
    return examples

# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def convert_examples_to_features(
    examples: Sequence[InputExample],
    label_list: Sequence[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    *,
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index,
) -> List[InputFeatures]:
    """Turn InputExample objects into BERT‑style InputFeatures."""
    label_map = {l: i for i, l in enumerate(label_list)}
    features: list[InputFeatures] = []

    cls_token, sep_token = tokenizer.cls_token, tokenizer.sep_token
    cls_token_segment_id = 0

    for ex_idx, example in enumerate(examples):
        tokens, label_ids, valid_ids = [], [], []

        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word) or [tokenizer.unk_token]
            tokens.extend(word_tokens)
            # first sub‑token gets the real label, the rest get pad
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
            valid_ids.extend([1] + [0] * (len(word_tokens) - 1))

        # Account for special tokens
        special = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special:
            tokens = tokens[: max_seq_length - special]
            label_ids = label_ids[: max_seq_length - special]
            valid_ids = valid_ids[: max_seq_length - special]

        # Add [CLS] and [SEP]
        tokens = [cls_token] + tokens + [sep_token]
        label_ids = [pad_token_label_id] + label_ids + [pad_token_label_id]
        valid_ids = [0] + valid_ids + [0]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [cls_token_segment_id] + [0] * (len(input_ids) - 1)

        # Padding
        padding_len = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_len
        attention_mask += [0] * padding_len
        token_type_ids += [tokenizer.pad_token_type_id] * padding_len
        label_ids += [pad_token_label_id] * padding_len
        valid_ids += [0] * padding_len

        assert all(len(x) == max_seq_length for x in (input_ids, attention_mask, token_type_ids, label_ids, valid_ids))

        if ex_idx < 3:  # brief logging
            logger.debug("*** Example %d ***", ex_idx)
            logger.debug("tokens: %s", " ".join(tokens))
            logger.debug("labels: %s", " ".join(map(str, label_ids)))

        features.append(InputFeatures(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            label_ids=label_ids,
            valid_ids=valid_ids,
        ))
    return features

# ---------------------------------------------------------------------------
# Torch dataset
# ---------------------------------------------------------------------------

class NerDataset(Dataset):
    """A torch‑style dataset that caches features and keeps raw examples."""

    pad_token_label_id = nn.CrossEntropyLoss().ignore_index

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        labels: Sequence[str],
        max_seq_length: int,
        mode: Split = Split.train,
        overwrite_cache: bool = False,
    ) -> None:
        cache_name = f"cached_{mode.value}_{tokenizer.__class__.__name__}_{max_seq_length}"
        cache_path = os.path.join(data_dir, cache_name)
        lock = FileLock(cache_path + ".lock")

        with lock:
            if os.path.exists(cache_path) and not overwrite_cache:
                logger.info("Loading features from %s", cache_path)
                bundle = torch.load(cache_path)
                self.features, self.examples = bundle["features"], bundle["examples"]
            else:
                logger.info("Creating features for %s", mode.value)
                self.examples = read_examples_from_file(data_dir, mode)
                self.features = convert_examples_to_features(
                    self.examples,
                    labels,
                    max_seq_length,
                    tokenizer,
                    pad_token_label_id=self.pad_token_label_id,
                )
                torch.save({"features": self.features, "examples": self.examples}, cache_path)

    # Required dataset methods ------------------------------------------------

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        f = self.features[idx]
        return {
            "input_ids": torch.tensor(f.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(f.attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(f.token_type_ids, dtype=torch.long),
            "labels": torch.tensor(f.label_ids, dtype=torch.long),
            "valid_ids": torch.tensor(f.valid_ids, dtype=torch.long),
        }

# ---------------------------------------------------------------------------
# Label helper
# ---------------------------------------------------------------------------

def get_labels(path: Optional[str] = None) -> List[str]:
    """Return the list of label strings, reading from *path* if given."""
    if path and os.path.isfile(path):
        with open(path, encoding="utf‑8") as f:
            labels = [l.strip() for l in f if l.strip()]
        return ["O"] + [l for l in labels if l != "O"]
    # Default label set for your disposition task
    return [
        "O",
        "B-Disposition", "I-Disposition",
        "B-NoDisposition", "I-NoDisposition",
        "B-Undetermined", "I-Undetermined",
    ]