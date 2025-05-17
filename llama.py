#!/usr/bin/env python3
"""
Fine-tune (LoRA) a HuggingFace LLaMA on BirdSQL and/or run inference.

Example:
  python llama.py \
      --do_train \
      --train_path  ./data/train/train.json \
      --db_root_path ./data/train/train_databases \
      --engine meta-llama/Llama-2-7b-hf \
      --output_dir checkpoints/llama_bird_ft

Author: you
"""
import argparse, json, os, sqlite3, sys, math, csv
from pathlib import Path

# -------------------------  HELPER FUNCTIONS  ------------------------- #
def find_sqlite_file(db_root: str, db_id: str):
    """
    Return path to the first *.sqlite / *.db / *.sqlite3* file for `db_id`.
    Handles three common layouts:
        1) db_root/db_id/db_id.sqlite
        2) db_root/db_id/<any_name>.{sqlite,db,sqlite3}
        3) db_root/db_id.{sqlite,db,sqlite3}
    """
    # 1 — canonical
    cand = Path(db_root) / db_id / f"{db_id}.sqlite"
    if cand.is_file():
        return str(cand)

    # 2 — any file under the folder
    folder = Path(db_root) / db_id
    if folder.is_dir():
        for f in folder.iterdir():
            if f.suffix.lower() in (".sqlite", ".db", ".sqlite3"):
                return str(f)

    # 3 — flat
    for ext in (".sqlite", ".db", ".sqlite3"):
        flat = Path(db_root) / f"{db_id}{ext}"
        if flat.is_file():
            return str(flat)

    return None

def get_schema_and_examples(db_path: str, example_rows: int = 3):
    """Extract CREATE TABLE DDL + a few example rows for each table."""
    schema_lines, example_lines = [], []
    try:
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()
        cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
        tables = cur.fetchall()
    except Exception as e:
        print(f"[WARN] SQLite error on {db_path}: {e}", file=sys.stderr)
        return "", ""

    for tbl, create_sql in tables:
        if not create_sql:
            continue
        schema_lines.append(create_sql.strip() + ";")
        if example_rows <= 0:
            continue
        try:
            cur.execute(f"PRAGMA table_info('{tbl}')")
            cols = [r[1] for r in cur.fetchall()]
            if not cols:
                continue
            cur.execute(f"SELECT * FROM '{tbl}' LIMIT {example_rows}")
            rows = cur.fetchall()
            if rows:
                example_lines.append(f"Table: {tbl}")
                example_lines.append(" | ".join(cols))
                for r in rows:
                    example_lines.append(" | ".join(map(str, r)))
                example_lines.append("")          # blank line
        except Exception:
            continue

    return "\n".join(schema_lines), "\n".join(example_lines)

def load_json_like(path: str):
    """Loads .json / .jsonl / .csv into a list[dict]."""
    if path.endswith(".json"):
        data = json.load(open(path))
        return data["data"] if isinstance(data, dict) and "data" in data else data
    if path.endswith(".jsonl"):
        return [json.loads(l) for l in open(path)]
    if path.endswith(".csv"):
        with open(path, newline='', encoding="utf-8") as f:
            return list(csv.DictReader(f))
    raise ValueError("file must be .json/.jsonl/.csv")

def prepare_prompt(question, schema_txt, example_txt, with_schema=True, with_example=True):
    parts = []
    if with_schema and schema_txt:
        parts.append("SQLite schema:\n" + schema_txt)
    if with_example and example_txt:
        parts.append("Example rows:\n" + example_txt)
    parts.append(f"Question: {question}")
    parts.append("SQL Query:")
    return "\n\n".join(parts) + " "

# ----------------------------  MAIN SCRIPT  --------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path")
    ap.add_argument("--eval_path")
    ap.add_argument("--db_root_path", required=True)
    ap.add_argument("--engine", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--output_dir", default="./llama_finetuned")
    ap.add_argument("--data_output_path", default="./predictions.json")
    ap.add_argument("--do_train", action="store_true")
    ap.add_argument("--do_eval",  action="store_true")
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.do_train and not args.do_eval:
        ap.error("Need at least --do_train or --do_eval")

    # lazy-import heavy libs only if needed
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              TrainingArguments, Trainer,
                              DataCollatorForLanguageModeling)
    import torch
    from peft import get_peft_model, LoraConfig, TaskType

    tokenizer = AutoTokenizer.from_pretrained(args.engine, use_fast=False, token=os.getenv("HF_TOKEN"))
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = None
    # ------------------------  TRAINING  ------------------------- #
    if args.do_train:
        print("→ Loading base model …")
        model = AutoModelForCausalLM.from_pretrained(
            args.engine, load_in_8bit=True, device_map="auto", token=os.getenv("HF_TOKEN")
        )
        lora_cfg = LoraConfig(r=8, lora_alpha=16,
                              target_modules=["q_proj","k_proj","v_proj","o_proj"],
                              task_type=TaskType.CAUSAL_LM)
        model = get_peft_model(model, lora_cfg).train()
        torch.manual_seed(args.seed)

        train_rows = load_json_like(args.train_path)
        prompts, labels = [], []
        missing, total = 0, len(train_rows)
        for i, ex in enumerate(train_rows):
            if i and i % max(1, total//10) == 0:
                print(f"  processed {i}/{total}")
            db_id      = ex.get("db_id")
            question   = ex.get("question")
            answer_sql = ex.get("SQL") or ex.get("sql")
            if not (db_id and question and answer_sql):
                continue
            db_path = find_sqlite_file(args.db_root_path, db_id)
            if db_path is None:
                missing += 1
                continue
            schema_txt, example_txt = get_schema_and_examples(db_path)
            prompt = prepare_prompt(question, schema_txt, example_txt)
            full   = prompt + answer_sql
            tok_full   = tokenizer(full,   max_length=args.max_length, truncation=True, padding="max_length")
            tok_prompt = tokenizer(prompt, max_length=args.max_length, truncation=True, padding="max_length")
            inp_ids    = tok_full["input_ids"]
            len_prompt = sum(1 for _ in tok_prompt["input_ids"] if _ != tokenizer.pad_token_id)
            lbl_ids    = [-100]*len_prompt + inp_ids[len_prompt:]
            lbl_ids    = lbl_ids[:len(inp_ids)] + [-100]*(len(inp_ids)-len(lbl_ids))
            prompts.append(inp_ids); labels.append(lbl_ids)

        print(f"✓ usable rows: {len(prompts)}   ✗ missing DB: {missing}")
        if len(prompts) == 0:
            raise RuntimeError("0 training examples after filtering – check --db_root_path and JSON keys")

        class BirdDS(torch.utils.data.Dataset):
            def __init__(s,x,y): s.x=x; s.y=y
            def __len__(s): return len(s.x)
            def __getitem__(s,i):
                ids = torch.tensor(s.x[i]); lab = torch.tensor(s.y[i])
                mask = (ids != tokenizer.pad_token_id).long()
                return {"input_ids":ids,"attention_mask":mask,"labels":lab}

        trainer = Trainer(
            model           = model,
            args            = TrainingArguments(
                output_dir   = args.output_dir,
                num_train_epochs=args.num_train_epochs,
                per_device_train_batch_size=args.per_device_train_batch_size,
                learning_rate=args.learning_rate,
                fp16          = torch.cuda.is_available(),
                save_total_limit=1, logging_steps=50, save_steps=200,
                remove_unused_columns=False),
            train_dataset   = BirdDS(prompts, labels),
            data_collator   = DataCollatorForLanguageModeling(tokenizer, mlm=False),
            tokenizer       = tokenizer
        )
        trainer.train()
        model.save_pretrained(args.output_dir); tokenizer.save_pretrained(args.output_dir)

    # ------------------------  INFERENCE  ------------------------ #
    if args.do_eval:
        if model is None:
            load_path = args.output_dir if Path(args.output_dir).is_dir() else args.engine
            model = AutoModelForCausalLM.from_pretrained(load_path, device_map="auto")
        model.eval()

        eval_rows = load_json_like(args.eval_path)
        preds = []
        for ex in eval_rows:
            db_id    = ex.get("db_id")
            question = ex.get("question")
            if not (db_id and question):
                continue
            db_path = find_sqlite_file(args.db_root_path, db_id)
            if db_path is None:
                print(f"[WARN] missing DB for {db_id}", file=sys.stderr)
                continue
            schema_txt, example_txt = get_schema_and_examples(db_path)
            prompt = prepare_prompt(question, schema_txt, example_txt)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256)
            sql_pred = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True).strip()
            preds.append(f"{sql_pred}\t----- bird -----\t{db_id}")

        json.dump(preds, open(args.data_output_path,"w"), indent=2, ensure_ascii=False)
        print("Predictions written to", args.data_output_path)

if __name__ == "__main__":
    main()
