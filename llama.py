#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
import sys

def get_schema_and_examples(db_path, example_rows=3):
    """
    Extracts the SQLite schema (CREATE TABLE statements) and a few example rows from each table.
    Returns two strings: schema_text and example_data_text.
    """
    schema_lines = []
    example_lines = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    except sqlite3.Error as e:
        print(f"Error connecting to {db_path}: {e}", file=sys.stderr)
        return "", ""
    try:
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Error reading sqlite_master for {db_path}: {e}", file=sys.stderr)
        tables = []
    for table_name, create_sql in tables:
        if create_sql is None:
            continue
        # Collect CREATE TABLE statements
        schema_lines.append(create_sql.strip() + ";")
        # Collect example rows if requested
        if example_rows > 0:
            try:
                # Get column names
                cursor.execute(f"PRAGMA table_info('{table_name}')")
                cols = [row[1] for row in cursor.fetchall()]
                if not cols:
                    continue
                # Fetch example rows
                cursor.execute(f"SELECT * FROM {table_name} LIMIT {example_rows}")
                rows = cursor.fetchall()
                if rows:
                    example_lines.append(f"Table: {table_name}")
                    example_lines.append(" | ".join(cols))
                    for row in rows:
                        example_lines.append(" | ".join(str(v) for v in row))
                    example_lines.append("")  # blank line after each table
            except Exception:
                continue
    schema_text = "\n".join(schema_lines)
    example_text = "\n".join(example_lines)
    return schema_text, example_text

def load_data(path):
    """
    Loads BirdSQL data from a JSON, JSONL, or CSV file.
    Returns a list of examples (dicts with keys like 'db_id', 'question', 'sql').
    """
    examples = []
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                # If 'data' key contains list of examples
                if "data" in data and isinstance(data["data"], list):
                    examples = data["data"]
                else:
                    # If top-level is a list or single object
                    if isinstance(data, list):
                        examples = data
                    else:
                        examples = [data]
            elif isinstance(data, list):
                examples = data
            else:
                print(f"Unrecognized JSON structure in {path}", file=sys.stderr)
    elif path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
    elif path.endswith(".csv"):
        import csv
        with open(path, newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                examples.append(row)
    else:
        raise ValueError(f"Unsupported file type for {path}. Use .json, .jsonl, or .csv")
    return examples

def prepare_prompt(question, schema_text, example_text, include_schema=True, include_example=True):
    """
    Constructs the prompt by combining schema, example rows (optional), and the question.
    """
    prompt_parts = []
    if include_schema and schema_text:
        prompt_parts.append(f"SQLite schema:\n{schema_text}")
    if include_example and example_text:
        prompt_parts.append(f"Example rows:\n{example_text}")
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append("SQL Query:")  # cue for model to generate SQL
    # join with double newline for readability
    prompt = "\n\n".join(prompt_parts) + " "
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Fine-tune and run inference with a LLaMA model on BirdSQL")
    parser.add_argument("--train_path", type=str, default=None, help="Path to BirdSQL training data (JSON/CSV)")
    parser.add_argument("--eval_path", type=str, default=None, help="Path to BirdSQL dev data (JSON/CSV)")
    parser.add_argument("--db_root_path", type=str, required=True, help="Directory with BirdSQL SQLite databases")
    parser.add_argument("--engine", type=str, default="meta-llama/Llama-2-7b-hf", help="HuggingFace model ID or path")
    parser.add_argument("--output_dir", type=str, default="./llama_finetuned", help="Directory to save/load the fine-tuned model")
    parser.add_argument("--data_output_path", type=str, default="./predictions.json", help="Path to output JSON with predictions")
    parser.add_argument("--do_train", action="store_true", help="Whether to fine-tune the model")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run inference on dev set")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for training")
    parser.add_argument("--max_length", type=int, default=2048, help="Max tokens (prompt+SQL)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not args.do_train and not args.do_eval:
        print("Specify --do_train and/or --do_eval", file=sys.stderr)
        return

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
        from transformers import DataCollatorForLanguageModeling
        import torch
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError:
        print("Please install transformers, peft, and torch", file=sys.stderr)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.engine, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    # Training phase
    if args.do_train:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.engine, load_in_8bit=True, device_map="auto"
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(args.engine, device_map="auto")
        # Configure LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        model.train()
        torch.manual_seed(args.seed)

        if not args.train_path:
            print("Provide --train_path for training", file=sys.stderr)
            return
        train_data = load_data(args.train_path)
        prompts, labels = [], []
        for ex in train_data:
            db_id = ex.get("db_id") or ex.get("database_id")
            question = ex.get("question") or ex.get("query") or ""
            answer_sql = ex.get("sql") or ex.get("SQL") or ex.get("target") or ""
            if not (db_id and question and answer_sql):
                continue
            # Locate DB file
            db_file = None
            for ext in [".sqlite", ".db", ".sqlite3"]:
                candidate = os.path.join(args.db_root_path, db_id + ext)
                if os.path.isfile(candidate):
                    db_file = candidate
                    break
            if db_file is None:
                continue
            schema_text, example_text = get_schema_and_examples(db_file)
            prompt_text = prepare_prompt(question, schema_text, example_text)
            # Tokenize prompt and answer
            full_text = prompt_text + answer_sql
            tokenized_full = tokenizer(full_text, truncation=True, max_length=args.max_length, padding="max_length")
            tokenized_prompt = tokenizer(prompt_text, truncation=True, max_length=args.max_length, padding="max_length")
            input_ids = tokenized_full["input_ids"]
            # Mask out prompt tokens in labels
            len_prompt = sum(1 for _ in tokenized_prompt["input_ids"] if _ != tokenizer.pad_token_id)
            label_ids = [-100] * len_prompt + tokenized_full["input_ids"][len_prompt:]
            # Ensure label_ids same length as input_ids
            label_ids = label_ids[:len(input_ids)] + [-100] * max(0, len(input_ids) - len(label_ids))
            prompts.append(input_ids)
            labels.append(label_ids)

        class BirdSQLDataset(torch.utils.data.Dataset):
            def __init__(self, inputs, labels):
                self.inputs = inputs
                self.labels = labels
            def __len__(self):
                return len(self.inputs)
            def __getitem__(self, idx):
                inp = torch.tensor(self.inputs[idx], dtype=torch.long)
                lab = torch.tensor(self.labels[idx], dtype=torch.long)
                mask = (inp != tokenizer.pad_token_id).long()
                return {"input_ids": inp, "attention_mask": mask, "labels": lab}

        train_dataset = BirdSQLDataset(prompts, labels)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            fp16=torch.cuda.is_available(),
            save_total_limit=1,
            save_steps=200,
            logging_steps=50,
            remove_unused_columns=False
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator
        )
        trainer.train()
        # Save fine-tuned model
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    # Inference phase
    if args.do_eval:
        if model is None:
            # Load fine-tuned model if available
            if os.path.isdir(args.output_dir):
                try:
                    model = AutoModelForCausalLM.from_pretrained(args.output_dir, device_map="auto")
                    tokenizer = AutoTokenizer.from_pretrained(args.output_dir, use_fast=False)
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                except Exception:
                    model = AutoModelForCausalLM.from_pretrained(args.engine, device_map="auto")
            else:
                model = AutoModelForCausalLM.from_pretrained(args.engine, device_map="auto")
        model.eval()
        if not args.eval_path:
            print("Provide --eval_path for evaluation", file=sys.stderr)
            return
        eval_data = load_data(args.eval_path)
        predictions = []
        for ex in eval_data:
            db_id = ex.get("db_id") or ex.get("database_id")
            question = ex.get("question") or ex.get("query") or ""
            if not (db_id and question):
                continue
            db_file = None
            for ext in [".sqlite", ".db", ".sqlite3"]:
                candidate = os.path.join(args.db_root_path, db_id + ext)
                if os.path.isfile(candidate):
                    db_file = candidate
                    break
            if db_file is None:
                continue
            schema_text, example_text = get_schema_and_examples(db_file)
            prompt_text = prepare_prompt(question, schema_text, example_text)
            inputs = tokenizer(prompt_text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256)
            output_ids = out[0][inputs["input_ids"].shape[1]:]  # exclude prompt tokens
            pred_sql = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            if not pred_sql:
                pred_sql = ""
            result_str = f"{pred_sql}\t----- bird -----\t{db_id}"
            predictions.append(result_str)
        with open(args.data_output_path, "w", encoding="utf-8") as fout:
            json.dump(predictions, fout, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
