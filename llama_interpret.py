#!/usr/bin/env python
# llama_eval_interpret.py  (full version with --db_root support)

import os, sqlite3, argparse, json, numpy as np, torch, torch.nn.functional as F
import matplotlib.pyplot as plt, seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression

# ───────── helper: locate *.sqlite under a root ────────────
def find_sqlite_file(root: str, db_id: str):
    cand1 = os.path.join(root, db_id, f"{db_id}.sqlite")
    cand2 = os.path.join(root, f"{db_id}.sqlite")
    for c in (cand1, cand2):
        if os.path.isfile(c):
            return c
    for ext in (".db", ".sqlite3"):
        c1 = os.path.join(root, db_id, f"{db_id}{ext}")
        c2 = os.path.join(root, f"{db_id}{ext}")
        if os.path.isfile(c1):
            return c1
        if os.path.isfile(c2):
            return c2
    return None
# -----------------------------------------------------------

# ------- Utility Functions (unchanged) ---------------------

def tag_sql_clauses(sql_str):
    tokens = sql_str.replace(",", " ,").split()
    clause_tags = []
    current_clause = None
    for tok in tokens:
        up = tok.upper()
        if up == "SELECT":
            current_clause = "SELECT"
        elif up == "FROM":
            current_clause = "FROM"
        elif up in ["WHERE", "HAVING"]:
            current_clause = "WHERE"
        elif "JOIN" in up:
            current_clause = "JOIN"
        elif up in ["GROUP", "ORDER"]:
            current_clause = "GROUPBY" if up == "GROUP" else "ORDERBY"
        clause_tags.append(current_clause if current_clause else "OTHER")
    return tokens, clause_tags

def plot_attention_heatmap(input_tokens, clause_names, attn_scores, out_path):
    plt.figure(figsize=(len(input_tokens)*0.4+1, len(clause_names)*0.4+1))
    sns.heatmap(attn_scores.T, xticklabels=input_tokens,
                yticklabels=clause_names, cmap='Reds', cbar=True)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel("Input Tokens")
    plt.ylabel("SQL Clause")
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

# ---------------- Main pipeline ----------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device).eval()

    for subdir in ["preds", "viz", "probe"]:
        os.makedirs(subdir, exist_ok=True)

    # Load evaluation examples
    examples = [json.loads(l) for l in open(args.data_file)]
    print(f"Loaded {len(examples)} examples.")

    # Accumulators for concept probing
    concept_features = {"table": [], "column": []}
    concept_labels   = {"table": [], "column": []}

    for ex in examples:
        ex_id   = ex.get("id", ex.get("example_id", "0"))
        db_id   = ex.get("db_id", ex_id)          # NEW: for db_root lookup
        question= ex["question"]
        schema  = ex.get("schema", "")
        db_path = ex.get("db_path")

        # NEW: derive db_path if missing
        if not db_path and args.db_root:
            db_path = find_sqlite_file(args.db_root, db_id)

        # Prepare prompt
        prompt = f"-- SQL Generation. Question: {question}\n-- Schema: {schema}\nSELECT"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate SQL
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                output_attentions=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        gen_tokens = generation_output.sequences[0]
        pred_sql = tokenizer.decode(gen_tokens, skip_special_tokens=True)[len(prompt):].strip()

        with open(os.path.join("preds", f"{ex_id}_pred.sql"), "w") as f:
            f.write(pred_sql)

        # Clause tagging & attention flow
        sql_tokens, clause_tags = tag_sql_clauses(pred_sql)
        clause_names = sorted(set(ct for ct in clause_tags if ct))

        full_input = prompt + " " + pred_sql
        full_ids = tokenizer(full_input, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            outputs = model(full_ids, output_attentions=True)
        attn_tensor = torch.stack([a[0] for a in outputs.attentions])  # (L,H,S,S)
        seq_len = full_ids.size(1); input_len = inputs.input_ids.size(1); output_len = seq_len - input_len

        attn_scores = np.zeros((input_len, len(clause_names)))
        clause_to_idx = {c:i for i,c in enumerate(clause_names)}
        L, H = attn_tensor.shape[:2]
        for l in range(L):
            for h in range(H):
                mat = attn_tensor[l,h].cpu().numpy()
                for o in range(output_len):
                    clause = clause_tags[o]
                    if clause in clause_to_idx:
                        idx = clause_to_idx[clause]
                        attn_scores[:, idx] += mat[:input_len, input_len + o]
        attn_scores /= (L * H)

        input_tok_str = tokenizer.convert_ids_to_tokens(full_ids[0])[:input_len]
        plot_attention_heatmap(input_tok_str, clause_names, attn_scores,
                               os.path.join("viz", f"{ex_id}_attention_heatmap.png"))

        # Schema-Concept probe feature collection
        tbls = re.findall(r"CREATE TABLE (\\w+)", schema, re.I)
        col_names = []
        for cols in re.findall(r"\\(([^)]+)\\)", schema):
            col_names += [c.strip() for c in cols.split(",")]
        layer_idx = args.probe_layer if args.probe_layer < len(generation_output.hidden_states) else -1
        layer_states = generation_output.hidden_states[layer_idx][0].cpu().numpy()  # (seq, dim)
        for pos, tok_str in enumerate(sql_tokens):
            vec = layer_states[input_len + pos]
            concept_features["table"].append(vec)
            concept_labels["table"].append(int(tok_str in tbls))
            concept_features["column"].append(vec)
            concept_labels["column"].append(int(tok_str in col_names))

        # Counter-factual prompt
        cf_q, cf_s = question, schema
        for k,v in args.cf_map.items():
            cf_q = cf_q.replace(k, v); cf_s = cf_s.replace(k, v)
        if cf_q != question or cf_s != schema:
            cf_prompt = f"-- SQL Generation (counterfactual). Question: {cf_q}\n-- Schema: {cf_s}\nSELECT"
            with torch.no_grad():
                cf_out = model.generate(tokenizer(cf_prompt, return_tensors="pt").to(device),
                                        max_new_tokens=args.max_new_tokens)
            cf_sql = tokenizer.decode(cf_out[0], skip_special_tokens=True)[len(cf_prompt):].strip()
            with open(os.path.join("preds", f"{ex_id}_cf_pred.sql"), "w") as f:
                f.write(cf_sql)

        # Execution check
        exec_flag = False
        if db_path and os.path.exists(db_path):
            try:
                sqlite3.connect(db_path).execute(pred_sql)
                exec_flag = True
            except Exception:
                exec_flag = False
        with open(os.path.join("preds", f"{ex_id}_exec.txt"), "w") as f:
            f.write(f"Executable: {exec_flag}\n")

        # Δ-logP attribution
        target_ids = tokenizer(pred_sql, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            lp_full = model(input_ids=inputs.input_ids, labels=target_ids).loss.item() * target_ids.size(1)
        deltas = []
        for j in range(input_len):
            masked = torch.cat([inputs.input_ids[:,:j], inputs.input_ids[:,j+1:]], 1)
            with torch.no_grad():
                lp = model(input_ids=masked, labels=target_ids).loss.item() * target_ids.size(1)
            deltas.append(lp_full - lp)
        plt.figure(figsize=(input_len*0.3+1,2.5))
        sns.barplot(x=input_tok_str, y=deltas, color="skyblue")
        plt.xticks(rotation=90); plt.ylabel("Δ log-prob"); plt.tight_layout()
        plt.savefig(os.path.join("viz", f"{ex_id}_attr.png")); plt.close()

        print(f"Processed {ex_id}: exec={exec_flag}")

    # Train concept probes
    for concept in ("table", "column"):
        X = np.stack(concept_features[concept]); y = np.array(concept_labels[concept])
        if len(np.unique(y)) < 2:
            with open(os.path.join("probe", f"{concept}_probe.txt"), "w") as f:
                f.write("Not enough positive/negative examples.\n")
            continue
        clf = LogisticRegression(max_iter=1000).fit(X, y)
        acc = clf.score(X, y)
        with open(os.path.join("probe", f"{concept}_probe.txt"), "w") as f:
            f.write(f"Accuracy: {acc:.3f}\nNorm‖w‖: {np.linalg.norm(clf.coef_):.3f}\n")

    print("Interpretability evaluation completed.")

# ---------------- CLI -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned LLaMA model")
    parser.add_argument("--data_file", required=True, help="JSONL file of dev examples")
    parser.add_argument("--db_root", default="", help="Root folder containing <db_id>/<db_id>.sqlite")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--probe_layer", type=int, default=-1)
    parser.add_argument("--cf_map", type=json.loads, default="{}",
                        help='JSON string mapping originals→replacements for counter-factual (e.g. \'{"orders":"purchases"}\')')
    args = parser.parse_args()
    main(args)
