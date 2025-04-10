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


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PretrainedConfig,
    set_seed,
)
from transformers.modeling_outputs import TokenClassifierOutput
from seqeval.metrics import classification_report


logger = logging.getLogger(__name__)
IGNORE_INDEX = -100

# ---------------------------------------------------------------------------
# CLI argument groups
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    teacher_model_name_or_path: str = field(
        metadata={"help": "HF model path/name for the teacher"}
    )
    student_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path/name for a StudentConfig JSON"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer path/name (defaults to teacher's)"},
    )
    use_fast: bool = field(default=True)
    cache_dir: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_dir: str = field(
        metadata={"help": "Folder with train_dev.txt / devel.txt / test.txt"}
    )
    labels: Optional[str] = field(default=None)
    max_seq_length: int = field(default=128)
    overwrite_cache: bool = field(default=False)


@dataclass
class DistillArguments:
    distillation_method: str = field(
        default="none",
        metadata={"help": "none | kl | sj | ssjs | kl+ssjs"},
    )
    temperature: float = 2.0
    alpha_ce: float = 0.5           # weight of CE vs distillation loss
    beta_mse: float = 0.1           # hidden‑state MSE weight
    lambda_ssjs: float = 1.0        # transition‑JS weight in SSJS
    gamma_kl_ssjs: float = 0.5      # weight between KL and SSJS in kl+ssjs


# ---------------------------------------------------------------------------
# Student model
# ---------------------------------------------------------------------------

class StudentConfig(PretrainedConfig):
    model_type = "student"

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 256,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        hidden_dropout_prob: float = 0.1,
        num_labels: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_labels = num_labels


class StudentModelForTokenClassification(PreTrainedModel):
    config_class = StudentConfig
    base_model_prefix = "student"

    def __init__(self, config: StudentConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dropout=config.hidden_dropout_prob,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_hidden_layers
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        output_hidden_states=False,
        teacher_logits=None,
    ):
        x = self.embeddings(input_ids)                # [B, L, H]
        x = x.transpose(0, 1)                         # [L, B, H] for transformer
        pad_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = x.transpose(0, 1)                         # [B, L, H]
        hidden = self.dropout(x)
        logits = self.classifier(hidden)

        loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
            if teacher_logits is not None:
                T = 2.0
                teacher_probs = F.softmax(teacher_logits / T, dim=-1)
                student_logp = F.log_softmax(logits / T, dim=-1)
                kl_loss = F.kl_div(student_logp, teacher_probs, reduction="batchmean") * (
                    T ** 2
                )
                loss = 0.5 * ce_loss + 0.5 * kl_loss
            else:
                loss = ce_loss

        if output_hidden_states:
            return TokenClassifierOutput(
                loss=loss, logits=logits, hidden_states=hidden
            )
        else:
            return TokenClassifierOutput(loss=loss, logits=logits)


# ---------------------------------------------------------------------------
# Distillation helpers
# ---------------------------------------------------------------------------

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p + eps
    q = q + eps
    m = 0.5 * (p + q)
    return 0.5 * (
        torch.sum(p * torch.log(p / m), dim=-1)
        + torch.sum(q * torch.log(q / m), dim=-1)
    )


def token_js_loss(t_logits: torch.Tensor, s_logits: torch.Tensor) -> torch.Tensor:
    return js_divergence(
        F.softmax(t_logits, dim=-1), F.softmax(s_logits, dim=-1)
    ).mean()


def transition_js_loss(t_logits: torch.Tensor, s_logits: torch.Tensor) -> torch.Tensor:
    t_probs = F.softmax(t_logits, dim=-1)
    s_probs = F.softmax(s_logits, dim=-1)
    if t_probs.size(1) < 2:  # seq_len < 2
        return torch.tensor(0.0, device=t_probs.device)
    t_joint = torch.einsum("bti,btj->btij", t_probs[:, :-1, :], t_probs[:, 1:, :])
    s_joint = torch.einsum("bti,btj->btij", s_probs[:, :-1, :], s_probs[:, 1:, :])
    return js_divergence(t_joint, s_joint).mean()


# ---------------------------------------------------------------------------
# Custom Trainer with distillation
# ---------------------------------------------------------------------------

class DistillationTrainer(Trainer):
    def __init__(
        self,
        teacher_model: PreTrainedModel,
        distill_args: DistillArguments,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        self.method = distill_args.distillation_method.lower()
        self.T = distill_args.temperature
        self.alpha_ce = distill_args.alpha_ce
        self.beta_mse = distill_args.beta_mse
        self.lambda_ssjs = distill_args.lambda_ssjs
        self.gamma_kl_ssjs = distill_args.gamma_kl_ssjs

        # projection if hidden sizes differ
        if (
            self.model.config.hidden_size != self.teacher.config.hidden_size
            and self.method != "none"
        ):
            self.proj = nn.Linear(
                self.model.config.hidden_size, self.teacher.config.hidden_size
            )
        else:
            self.proj = None

    # --------------------------------------------------
    # Training loss
    # --------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # 1) student forward
        s_out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            token_type_ids=inputs.get("token_type_ids"),
            labels=labels,
            output_hidden_states=True,
        )
        loss = s_out.loss

        if self.method == "none":
            return (loss, s_out) if return_outputs else loss

        # 2) teacher forward (no grad)
        with torch.no_grad():
            t_out = self.teacher(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                token_type_ids=inputs.get("token_type_ids"),
                output_hidden_states=True,
            )

        # 3) distillation losses
        distill_loss = torch.tensor(0.0, device=loss.device)

        if self.method in {"kl", "kl+ssjs"}:
            kl = F.kl_div(
                F.log_softmax(s_out.logits / self.T, dim=-1),
                F.softmax(t_out.logits / self.T, dim=-1),
                reduction="batchmean",
            ) * (self.T ** 2)
            if self.method == "kl":
                distill_loss = kl
            else:  # kl+ssjs
                token_js = token_js_loss(t_out.logits, s_out.logits)
                trans_js = transition_js_loss(t_out.logits, s_out.logits)
                ssjs = token_js + self.lambda_ssjs * trans_js
                distill_loss = self.gamma_kl_ssjs * kl + (1 - self.gamma_kl_ssjs) * ssjs

        elif self.method == "sj":
            distill_loss = token_js_loss(t_out.logits, s_out.logits)

        elif self.method == "ssjs":
            token_js = token_js_loss(t_out.logits, s_out.logits)
            trans_js = transition_js_loss(t_out.logits, s_out.logits)
            distill_loss = token_js + self.lambda_ssjs * trans_js

        # hidden‑state MSE
        student_hidden = s_out.hidden_states[-1]
        teacher_hidden = t_out.hidden_states[-1]
        if self.proj is not None:
            student_hidden = self.proj(student_hidden)
        mse = F.mse_loss(student_hidden, teacher_hidden)

        total_loss = (
            self.alpha_ce * loss
            + (1 - self.alpha_ce) * distill_loss
            + self.beta_mse * mse
        )

        return (total_loss, s_out) if return_outputs else total_loss


# ---------------------------------------------------------------------------
# Helper: align predictions (no valid_ids needed)
# ---------------------------------------------------------------------------

def align_predictions(
    predictions: np.ndarray, label_ids: np.ndarray, id2label: Dict[int, str]
) -> Tuple[List[List[str]], List[List[str]]]:
    preds = predictions.argmax(-1)
    preds_list, labels_list = [], []
    for p_row, l_row in zip(preds, label_ids):
        sent_pred, sent_gold = [], []
        for p, l in zip(p_row, l_row):
            if l == IGNORE_INDEX:
                continue
            sent_pred.append(id2label[int(p)])
            sent_gold.append(id2label[int(l)])
        preds_list.append(sent_pred)
        labels_list.append(sent_gold)
    return preds_list, labels_list


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def build_compute_metrics(id2label: Dict[int, str]):
    def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
        preds_list, gold_list = align_predictions(
            p.predictions, p.label_ids, id2label
        )
        report = classification_report(
            gold_list, preds_list, output_dict=True, zero_division=0
        )
        metrics: Dict[str, float] = {}
        # per‑label
        for lbl, stats in report.items():
            if isinstance(stats, dict) and lbl not in {
                "accuracy",
                "macro avg",
                "weighted avg",
                "micro avg",
            }:
                metrics[f"{lbl}_precision"] = stats["precision"]
                metrics[f"{lbl}_recall"] = stats["recall"]
                metrics[f"{lbl}_f1"] = stats["f1-score"]
        # macro average
        if "macro avg" in report:
            metrics["overall_precision"] = report["macro avg"]["precision"]
            metrics["overall_recall"] = report["macro avg"]["recall"]
            metrics["overall_f1"] = report["macro avg"]["f1-score"]
        return metrics

    return compute_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, DistillArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        m_args, d_args, dist_args, t_args = parser.parse_json_file(
            os.path.abspath(sys.argv[1])
        )
    else:
        m_args, d_args, dist_args, t_args = parser.parse_args_into_dataclasses()

    # safety on output dir
    if (
        os.path.exists(t_args.output_dir)
        and os.listdir(t_args.output_dir)
        and t_args.do_train
        and not t_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output dir {t_args.output_dir} exists and is not empty "
            "— use --overwrite_output_dir to continue."
        )

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if t_args.local_rank in (-1, 0) else logging.WARN,
    )
    set_seed(t_args.seed)

    # ------------------------------------------------------------------ labels
    labels = get_labels(d_args.labels)
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}
    num_labels = len(labels)

    # ---------------------------------------------------------- teacher model
    teacher_cfg = AutoConfig.from_pretrained(
        m_args.teacher_model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        cache_dir=m_args.cache_dir,
        output_hidden_states=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        m_args.tokenizer_name or m_args.teacher_model_name_or_path,
        use_fast=m_args.use_fast,
        cache_dir=m_args.cache_dir,
    )
    teacher = AutoModelForTokenClassification.from_pretrained(
        m_args.teacher_model_name_or_path,
        config=teacher_cfg,
        cache_dir=m_args.cache_dir,
    )

    # ---------------------------------------------------------- student model
    if m_args.student_config_name:
        s_cfg = StudentConfig.from_pretrained(
            m_args.student_config_name,
            num_labels=num_labels,
            vocab_size=tokenizer.vocab_size,
            cache_dir=m_args.cache_dir,
        )
    else:
        s_cfg = StudentConfig(
            num_labels=num_labels, vocab_size=tokenizer.vocab_size
        )
    student = StudentModelForTokenClassification(s_cfg)

    # ---------------------------------------------------------- datasets
    def load(split: Optional[Split]):
        if split is None:
            return None
        return NerDataset(
            data_dir=d_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=teacher_cfg.model_type,
            max_seq_length=d_args.max_seq_length,
            overwrite_cache=d_args.overwrite_cache,
            mode=split,
        )

    train_ds = load(Split.train) if t_args.do_train else None
    dev_ds = load(Split.dev) if t_args.do_eval else None
    test_ds = load(Split.test) if t_args.do_predict else None

    # ---------------------------------------------------------- trainer
    trainer = DistillationTrainer(
        teacher_model=teacher,
        distill_args=dist_args,
        model=student,
        args=t_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=build_compute_metrics(id2label),
    )

    # ---------------------------------------------------------- train
    if t_args.do_train:
        trainer.train()
        trainer.save_model()
        if trainer.args.local_rank in (-1, 0):
            tokenizer.save_pretrained(t_args.output_dir)

    # ---------------------------------------------------------- eval
    if t_args.do_eval and dev_ds is not None:
        logger.info("*** Evaluating student ***")
        stud_metrics = trainer.evaluate()
        for k, v in stud_metrics.items():
            logger.info("%s = %.4f", k, v)

        # teacher evaluation for comparison
        teacher_eval_trainer = Trainer(
            model=teacher,
            args=t_args,
            eval_dataset=dev_ds,
            compute_metrics=build_compute_metrics(id2label),
        )
        logger.info("*** Evaluating teacher ***")
        teach_metrics = teacher_eval_trainer.evaluate()
        for k, v in teach_metrics.items():
            logger.info("T_%s = %.4f", k, v)

    # ---------------------------------------------------------- predict
    if t_args.do_predict and test_ds is not None:
        logger.info("*** Predicting on test set ***")
        preds, label_ids, _ = trainer.predict(test_ds)
        preds_list, _ = align_predictions(preds, label_ids, id2label)

        raw_examples = read_examples_from_file(d_args.data_dir, Split.test)
        out_path = os.path.join(t_args.output_dir, "test_predictions.tsv")
        if trainer.is_world_process_zero():
            with open(out_path, "w", encoding="utf-8") as w:
                for ex, pred in zip(raw_examples, preds_list):
                    for tok, lab in zip(ex.words, pred):
                        w.write(f"{tok}\t{lab}\n")
                    w.write("\n")
            logger.info("Wrote predictions to %s", out_path)


if __name__ == "__main__":
    main()
