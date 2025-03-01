#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    PreTrainedModel,
    PretrainedConfig
)
from transformers.modeling_outputs import TokenClassifierOutput

from seqeval.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# local utility for NER tasks
from utils_ner import get_labels, NerDataset, Split

logger = logging.getLogger(__name__)


# --------------------------
# Argument DataClasses
# --------------------------
@dataclass
class ModelArguments:
    teacher_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained teacher model or identifier from huggingface.co/models"}
    )
    student_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path for the student model"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path (shared between teacher and student)"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-formatted NER task."}
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not specified, default labels are used."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer than this will be truncated, shorter ones padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


@dataclass
class DistillationArguments:
    distillation_method: str = field(
        default="none",
        metadata={
            "help": (
                "Distillation method to use: 'none', 'kl', 'sj', 'ssjs', 'kl+ssjs'. "
                "If none, no teacher pass is done and only the student is trained."
            )
        },
    )
    temperature: float = field(
        default=2.0,
        metadata={"help": "Temperature for the distillation (KL-based)."}
    )
    alpha_ce: float = field(
        default=0.5,
        metadata={"help": "Weight for CE loss when combining with distillation loss. (1 - alpha_ce) is for distillation."}
    )
    beta_mse: float = field(
        default=0.1,
        metadata={"help": "Weight for feature-level MSE (teacher vs student hidden states)."}
    )
    lambda_ssjs: float = field(
        default=1.0,
        metadata={"help": "Weight for the transition-level JS part in SSJS."}
    )
    gamma_kl_ssjs: float = field(
        default=0.5,
        metadata={
            "help": (
                "For 'kl+ssjs' distillation, weighting factor for how much KL vs SSJS to use. "
                "The combined distillation loss = gamma_kl_ssjs*(KL loss) + (1-gamma_kl_ssjs)*(SSJS loss)."
            )
        }
    )


# --------------------------
# Helper: Align predictions
# --------------------------
def align_predictions(
    predictions: np.ndarray,
    label_ids: np.ndarray,
    label_map: Dict[int, str]
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Align raw logits with actual labels for chunk-based scoring (seqeval).
    """
    preds = np.argmax(predictions, axis=2)  # shape: (batch_size, seq_len)
    batch_size, seq_len, _ = predictions.shape
    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]
    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])
    return preds_list, out_label_list


# --------------------------
# SSJS Helper Functions
# --------------------------
def js_divergence(p: torch.Tensor, q: torch.Tensor, epsilon=1e-6) -> torch.Tensor:
    """
    Computes the Jensen-Shannon divergence between two probability distributions.
    p, q: tensors of shape (..., num_classes)
    """
    p = p + epsilon  # avoid log(0)
    q = q + epsilon
    m = 0.5 * (p + q)
    kl_p_m = torch.sum(p * torch.log(p / m), dim=-1)
    kl_q_m = torch.sum(q * torch.log(q / m), dim=-1)
    return 0.5 * (kl_p_m + kl_q_m)

def compute_token_js_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """
    Computes token-level JS divergence loss.
    """
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_probs = F.softmax(student_logits, dim=-1)
    token_js = js_divergence(teacher_probs, student_probs)
    return token_js.mean()

def compute_transition_js_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """
    Computes transition-level JS divergence loss based on adjacent token pairs.
    """
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_probs = F.softmax(student_logits, dim=-1)
    batch_size, seq_len, num_labels = teacher_probs.shape
    if seq_len < 2:
        return torch.tensor(0.0, device=teacher_probs.device)
    # Joint distribution for consecutive tokens
    teacher_joint = torch.einsum("bti,btj->btij", teacher_probs[:, :-1, :], teacher_probs[:, 1:, :])
    student_joint = torch.einsum("bti,btj->btij", student_probs[:, :-1, :], student_probs[:, 1:, :])
    transition_js = js_divergence(teacher_joint, student_joint)
    return transition_js.mean()


# --------------------------
# Custom Student Config
# --------------------------
class StudentConfig(PretrainedConfig):
    """
    Minimal custom config for the student model.
    """
    model_type = "student_model"
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=256,
        num_attention_heads=4,
        num_hidden_layers=2,
        hidden_dropout_prob=0.1,
        num_labels=2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_labels = num_labels


# --------------------------
# Student Model (PreTrainedModel)
# --------------------------
class StudentModelForTokenClassification(PreTrainedModel):
    """
    Student model for token classification.
    """
    config_class = StudentConfig
    base_model_prefix = "student"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=config.num_attention_heads,
            dropout=config.hidden_dropout_prob
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=config.num_hidden_layers
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        teacher_logits=None,
        output_hidden_states=False,
        **kwargs
    ):
        """
        If teacher_logits is provided (and you're using a standard approach),
        a KL or some form of distillation can be computed at this level.
        """
        x = self.embedding(input_ids)  # [batch, seq_len, hidden_size]
        x = x.transpose(0, 1)  # [seq_len, batch, hidden_size]

        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = x.transpose(0, 1)  # [batch, seq_len, hidden_size]

        hidden_states = self.dropout(x)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100
            )
            # If teacher_logits is present, you could do a built-in or partial distillation here.
            # But typically we'll handle that in the Trainer's custom compute_loss.
            if teacher_logits is not None:
                temperature = 2.0
                alpha = 0.5
                teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
                student_log_probs = F.log_softmax(logits / temperature, dim=-1)
                kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
                loss = alpha * ce_loss + (1 - alpha) * kl_loss
            else:
                loss = ce_loss

        if output_hidden_states:
            return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=hidden_states)
        else:
            return TokenClassifierOutput(loss=loss, logits=logits)


# --------------------------
# Custom Distillation Trainer
# --------------------------
class DistillationTrainer(Trainer):
    """
    Trainer that supports multiple distillation methods:
      - none: no distillation, purely CE loss on the student
      - kl: KL-based distillation
      - sj: JS (token-level only)
      - ssjs: token-level + transition-level JS
      - kl+ssjs: combined KL and SSJS
    """
    def __init__(
        self,
        teacher_model,
        distillation_method="none",
        temperature=2.0,
        alpha_ce=0.5,
        beta_mse=0.1,
        lambda_ssjs=1.0,
        gamma_kl_ssjs=0.5,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distillation_method = distillation_method.lower()
        self.temperature = temperature
        self.alpha_ce = alpha_ce      # weight for CE
        # (1 - alpha_ce) is the weight for distillation
        self.beta_mse = beta_mse      # weight for hidden-state MSE
        self.lambda_ssjs = lambda_ssjs
        self.gamma_kl_ssjs = gamma_kl_ssjs  # for kl+ssjs

        # If the student hidden size != teacher hidden size, project to teacher's dimension
        if (self.model.config.hidden_size != self.teacher_model.config.hidden_size
                and self.distillation_method != "none"):
            self.proj_layer = nn.Linear(
                self.model.config.hidden_size, self.teacher_model.config.hidden_size
            )
        else:
            self.proj_layer = None

    def compute_loss(self, model, inputs, **kwargs):
        """
        Distillation pipeline:
          - If 'none', skip teacher entirely.
          - Otherwise, compute teacher forward pass, then combine CE + distillation + MSE.
        """
        if self.distillation_method == "none":
            # No teacher pass. Just student forward with CE.
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                token_type_ids=inputs.get("token_type_ids"),
                labels=inputs.get("labels"),
                output_hidden_states=True
            )
            loss = outputs.loss
            return (loss, outputs) if kwargs.get("return_outputs", False) else loss

        # If we are doing distillation, we do teacher forward pass:
        device = inputs["input_ids"].device
        self.teacher_model.to(device)
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                token_type_ids=inputs.get("token_type_ids"),
                labels=inputs.get("labels"),
                output_hidden_states=True
            )
            teacher_logits = teacher_outputs.logits
            teacher_hidden = teacher_outputs.hidden_states[-1]

        # Student forward pass with hidden states
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            token_type_ids=inputs.get("token_type_ids"),
            labels=inputs.get("labels"),
            output_hidden_states=True
        )

        # CE Loss (on ground-truth labels)
        ce_loss = student_outputs.loss if student_outputs.loss is not None else 0.0

        # Distillation Loss
        student_logits = student_outputs.logits

        # Different methods
        if self.distillation_method == "kl":
            # Standard KL
            kl_loss = F.kl_div(
                F.log_softmax(student_logits / self.temperature, dim=-1),
                F.softmax(teacher_logits / self.temperature, dim=-1),
                reduction='batchmean'
            ) * (self.temperature ** 2)
            distill_loss = kl_loss

        elif self.distillation_method == "sj":
            # Token-level JS
            distill_loss = compute_token_js_loss(teacher_logits, student_logits)

        elif self.distillation_method == "ssjs":
            # Token-level + transition-level
            token_js = compute_token_js_loss(teacher_logits, student_logits)
            transition_js = compute_transition_js_loss(teacher_logits, student_logits)
            distill_loss = token_js + self.lambda_ssjs * transition_js

        elif self.distillation_method == "kl+ssjs":
            # Combine KL and SSJS with weighting gamma_kl_ssjs
            #  distill_loss = gamma_kl_ssjs * (KL) + (1 - gamma_kl_ssjs)*(SSJS)
            kl_loss = F.kl_div(
                F.log_softmax(student_logits / self.temperature, dim=-1),
                F.softmax(teacher_logits / self.temperature, dim=-1),
                reduction='batchmean'
            ) * (self.temperature ** 2)

            token_js = compute_token_js_loss(teacher_logits, student_logits)
            transition_js = compute_transition_js_loss(teacher_logits, student_logits)
            ssjs_loss = token_js + self.lambda_ssjs * transition_js

            distill_loss = self.gamma_kl_ssjs * kl_loss + (1 - self.gamma_kl_ssjs) * ssjs_loss

        else:
            raise ValueError(f"Unknown distillation method: {self.distillation_method}")

        # Feature-level MSE
        student_hidden = student_outputs.hidden_states
        if self.proj_layer is not None:
            student_hidden = self.proj_layer(student_hidden)
        mse_loss = F.mse_loss(student_hidden, teacher_hidden)

        # Combine
        # Overall = alpha_ce * CE + (1-alpha_ce)*distill_loss + beta_mse * MSE
        total_loss = (
            self.alpha_ce * ce_loss
            + (1 - self.alpha_ce) * distill_loss
            + self.beta_mse * mse_loss
        )

        if kwargs.get("return_outputs", False):
            return total_loss, student_outputs
        else:
            return total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            loss, logits, labels = self._distillation_prediction_step_inner(model, inputs, prediction_loss_only)
        return (loss, logits, labels)

    def _distillation_prediction_step_inner(self, model, inputs, prediction_loss_only):
        labels = inputs.get("labels", None)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            token_type_ids=inputs.get("token_type_ids"),
            labels=labels,
            output_hidden_states=False
        )
        loss = outputs.loss
        logits = outputs.logits
        if prediction_loss_only:
            return (loss, None, None)
        else:
            return (loss, logits, labels)


# --------------------------
# Custom Trainer for Teacher's Eval
# --------------------------
class TeacherEvalTrainer(Trainer):
    """
    Just for evaluating the teacher.
    """
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            loss, logits, labels = self._teacher_prediction_step_inner(model, inputs, prediction_loss_only)
        return (loss, logits, labels)

    def _teacher_prediction_step_inner(self, model, inputs, prediction_loss_only):
        labels = inputs.get("labels", None)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            token_type_ids=inputs.get("token_type_ids"),
            labels=labels,
            output_hidden_states=False
        )
        loss = outputs.loss
        logits = outputs.logits
        if prediction_loss_only:
            return (loss, None, None)
        else:
            return (loss, logits, labels)


# --------------------------
# Main Function
# --------------------------
def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, DistillationArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, distill_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, distill_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            f"Use --overwrite_output_dir to overcome."
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(training_args.seed)

    # Setup labels
    if data_args.labels is None:
        default_labels = [
            "B-Disposition",
            "B-NoDisposition",
            "B-Undetermined",
            "I-Disposition",
            "I-NoDisposition",
            "I-Undetermined",
            "O"
        ]
        labels = default_labels
    else:
        labels = get_labels(data_args.labels)
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    # Teacher config
    teacher_config = AutoConfig.from_pretrained(
        model_args.teacher_model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
    )
    teacher_config.output_hidden_states = True

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.teacher_model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )

    # Teacher model
    teacher_model = AutoModelForTokenClassification.from_pretrained(
        model_args.teacher_model_name_or_path,
        config=teacher_config,
        cache_dir=model_args.cache_dir,
    )

    # Student config
    if model_args.student_config_name:
        student_config = StudentConfig.from_pretrained(
            model_args.student_config_name,
            num_labels=num_labels,
            vocab_size=tokenizer.vocab_size,
            cache_dir=model_args.cache_dir
        )
    else:
        student_config = StudentConfig(
            num_labels=num_labels,
            vocab_size=tokenizer.vocab_size,
            hidden_size=256,
            num_attention_heads=4,
            num_hidden_layers=2,
            hidden_dropout_prob=0.1
        )

    # Student model
    student_model = StudentModelForTokenClassification(student_config)

    # Load train/eval dataset
    train_dataset = (
        NerDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=teacher_config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train else None
    )
    eval_dataset = (
        NerDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=teacher_config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval else None
    )

    def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids, label_map)
        chunk_precision = precision_score(out_label_list, preds_list)
        chunk_recall = recall_score(out_label_list, preds_list)
        chunk_f1 = f1_score(out_label_list, preds_list)

        # Token-level metrics
        y_true, y_pred = [], []
        for pred_row, lab_row in zip(p.predictions, p.label_ids):
            for logits, lab_id in zip(pred_row, lab_row):
                if lab_id != -100:
                    y_true.append(lab_id)
                    y_pred.append(np.argmax(logits))
        token_accuracy = accuracy_score(y_true, y_pred)
        token_precision, token_recall, token_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro"
        )
        return {
            "chunk_precision": chunk_precision,
            "chunk_recall": chunk_recall,
            "chunk_f1": chunk_f1,
            "token_accuracy": token_accuracy,
            "token_precision": token_precision,
            "token_recall": token_recall,
            "token_f1": token_f1,
        }

    # Initialize Distillation Trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        distillation_method=distill_args.distillation_method,
        temperature=distill_args.temperature,
        alpha_ce=distill_args.alpha_ce,
        beta_mse=distill_args.beta_mse,
        lambda_ssjs=distill_args.lambda_ssjs,
        gamma_kl_ssjs=distill_args.gamma_kl_ssjs
    )

    # Training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        if trainer.args.local_rank in [-1, 0]:
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation (Student)
    if training_args.do_eval:
        logger.info("*** Evaluate Student Model ***")
        student_results = trainer.evaluate()
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.args.local_rank in [-1, 0]:
            with open(output_eval_file, "w") as writer:
                logger.info("***** Student Eval results *****")
                for key, value in student_results.items():
                    logger.info("  %s = %s", key, value)
                    writer.write(f"{key} = {value}\n")

        # Also Evaluate Teacher
        logger.info("*** Evaluate Teacher Model ***")
        teacher_trainer = TeacherEvalTrainer(
            model=teacher_model,
            args=training_args,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
        teacher_results = teacher_trainer.evaluate()
        output_teacher_eval_file = os.path.join(training_args.output_dir, "teacher_eval_results.txt")
        if trainer.args.local_rank in [-1, 0]:
            with open(output_teacher_eval_file, "w") as writer:
                logger.info("***** Teacher Eval results *****")
                for key, value in teacher_results.items():
                    logger.info("  %s = %s", key, value)
                    writer.write(f"{key} = {value}\n")

        print(
            f"Teacher Metrics -> Precision: {teacher_results.get('chunk_precision')}, "
            f"Recall: {teacher_results.get('chunk_recall')}, F1: {teacher_results.get('chunk_f1')}"
        )

    # Prediction
    if training_args.do_predict:
        test_dataset = NerDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=teacher_config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
        )
        predictions, label_ids, metrics = trainer.predict(test_dataset)
        preds_list, _ = align_predictions(predictions, label_ids, label_map)
        output_test_results_file = os.path.join(training_args.output_dir, "test_results.txt")
        if trainer.is_world_process_zero():
            with open(output_test_results_file, "w") as writer:
                logger.info("***** Test results *****")
                for key, value in metrics.items():
                    logger.info("  %s = %s", key, value)
                    writer.write(f"{key} = {value}\n")

        # Write predicted labels
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                test_path = os.path.join(data_args.data_dir, "test.txt")
                if not os.path.exists(test_path):
                    logger.warning(f"No test.txt file found in {data_args.data_dir}.")
                else:
                    with open(test_path, "r") as f:
                        example_id = 0
                        for line in f:
                            if line.startswith("-DOCSTART-") or line.strip() == "":
                                writer.write(line)
                                if not preds_list[example_id]:
                                    example_id += 1
                            elif preds_list[example_id]:
                                token = line.strip().split()[0]
                                entity_label = preds_list[example_id].pop(0)
                                writer.write(f"{token} {entity_label}\n")
                            else:
                                logger.warning("Max sequence length exceeded: No prediction for '%s'.", line.split()[0])

if __name__ == "__main__":
    main()
