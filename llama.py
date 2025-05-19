#!/usr/bin/env python3
"""
LoRA-fine-tune or inference for BirdSQL with a HuggingFace LLaMA.
* Single-GPU:     python llama.py --do_train ...
* Multi-GPU:      torchrun --nproc_per_node N llama.py --do_train ...
"""
import argparse, json, os, sqlite3, sys, csv
from pathlib import Path

# ---------------- utilities ---------------- #
def find_sqlite_file(db_root: str, db_id: str):
    p = Path(db_root) / db_id / f"{db_id}.sqlite"
    if p.is_file(): return str(p)
    folder = Path(db_root) / db_id
    if folder.is_dir():
        for f in folder.iterdir():
            if f.suffix.lower() in (".sqlite", ".db", ".sqlite3"): return str(f)
    for ext in (".sqlite", ".db", ".sqlite3"):
        p = Path(db_root) / f"{db_id}{ext}"
        if p.is_file(): return str(p)
    return None

def get_schema(db_path, rows=3):
    ddl, examples = [], []
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
        for tbl, create in cur.fetchall():
            if not create: continue
            ddl.append(create.strip() + ";")
            if rows > 0:
                try:
                    cur2 = conn.execute(f"SELECT * FROM '{tbl}' LIMIT {rows}")
                    data = cur2.fetchall()
                    if data:
                        cols = [c[0] for c in cur2.description]
                        examples.append(f"/* {tbl} first {rows} rows */")
                        examples.append(" | ".join(cols))
                        for r in data: examples.append(" | ".join(map(str, r)))
                        examples.append("")
                except Exception: pass
    return "\n".join(ddl + examples)

def prompt(schema, question):      # minimal
    return f"{schema}\n\n-- {question}\nSQL Query:\nSELECT "

def load_rows(path):
    if path.endswith(".jsonl"):
        return [json.loads(l) for l in open(path)]
    if path.endswith(".csv"):
        return list(csv.DictReader(open(path)))
    data = json.load(open(path))
    return data["data"] if isinstance(data, dict) and "data" in data else data

# ---------------- main ------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path");         ap.add_argument("--eval_path")
    ap.add_argument("--db_root_path", required=True)
    ap.add_argument("--engine", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--output_dir", default="llama_ft")
    ap.add_argument("--data_output_path", default="pred.json")
    ap.add_argument("--do_train", action="store_true")
    ap.add_argument("--do_eval",  action="store_true")
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--max_length", type=int, default=1024)
    args = ap.parse_args()
    if not args.do_train and not args.do_eval:
        ap.error("use --do_train and/or --do_eval")

    from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                              TrainingArguments, DataCollatorForLanguageModeling,
                              BitsAndBytesConfig)
    from peft import get_peft_model, LoraConfig, TaskType
    import torch, random

    # ----- tokenizer (shared) -----
    tok = AutoTokenizer.from_pretrained(args.engine, use_fast=False)
    tok.pad_token = tok.eos_token

    # ----- model loader helper -----
    def load_model(for_training=False):
        rank = int(os.environ.get("LOCAL_RANK", 0))
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        m = AutoModelForCausalLM.from_pretrained(
            args.engine, quantization_config=bnb, device_map={"": rank}
        )
        m.gradient_checkpointing_enable()
        if for_training:
            lora = LoraConfig(r=8, lora_alpha=16,
                              target_modules=["q_proj","k_proj","v_proj","o_proj"],
                              task_type=TaskType.CAUSAL_LM)
            m = get_peft_model(m, lora)
            m.train()
        else:
            m.eval()
        return m

    # ------------- train -------------
    if args.do_train:
        model = load_model(for_training=True)
        rows  = load_rows(args.train_path)
        inputs, labels = [], []
        for ex in rows:
            db = find_sqlite_file(args.db_root_path, ex["db_id"])
            if not db: continue
            ptxt = prompt(get_schema(db, 0), ex["question"])
            full = ptxt + ex["SQL"]
            tok_full   = tok(full,  max_length=args.max_length, truncation=True, padding="max_length")
            tok_prompt = tok(ptxt, max_length=args.max_length, truncation=True, padding="max_length")
            ids = tok_full["input_ids"]
            n   = sum(1 for t in tok_prompt["input_ids"] if t!=tok.pad_token_id)
            lbl = [-100]*n + ids[n:]; lbl += [-100]*(len(ids)-len(lbl))
            inputs.append(ids); labels.append(lbl)

        class DS(torch.utils.data.Dataset):
            def __init__(s,a,b): s.a=a; s.b=b
            def __len__(s): return len(s.a)
            def __getitem__(s,i):
                x=torch.tensor(s.a[i]); y=torch.tensor(s.b[i])
                m=(x!=tok.pad_token_id).long()
                return {"input_ids":x,"attention_mask":m,"labels":y}

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=args.output_dir,
                num_train_epochs=args.num_train_epochs,
                per_device_train_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.learning_rate,
                fp16=True,
                save_total_limit=1,
                logging_steps=50),
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
            train_dataset=DS(inputs, labels),
            tokenizer=tok)
        trainer.train()
        model.save_pretrained(args.output_dir); tok.save_pretrained(args.output_dir)

    # ------------- eval --------------
    if args.do_eval:
        model = load_model(for_training=False) if not args.do_train else model
        rows  = load_rows(args.eval_path)
        preds = {}
        for i, ex in enumerate(rows):
            db = find_sqlite_file(args.db_root_path, ex["db_id"])
            if not db: continue
            prompt_txt = prompt(get_schema(db, 0), ex["question"])
            inp = tok(prompt_txt, return_tensors="pt", truncation=True,
                      max_length=args.max_length).to(model.device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=256)
            sql = tok.decode(out[0][inp["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip()
            preds[i] = f"{sql}\t----- bird -----\t{ex['db_id']}"
        Path(args.data_output_path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(preds, open(args.data_output_path,"w"), indent=2)
        print("preds →", args.data_output_path)

if __name__ == "__main__":
    main()
