#!/usr/bin/env python
# llama_interpret.py  —  evaluate a fine-tuned LLaMA Text-to-SQL model
# with clause-level attention, Δ-logP attribution, schema-concept probes,
# counter-factual SQL, and execution tests.
# -------------------------------------------------------------

import os, sqlite3, argparse, json, re, numpy as np
import torch, torch.nn.functional as F
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

# ────────── utilities ──────────
def tag_sql_clauses(sql_str):
    tokens = sql_str.replace(",", " ,").split()
    clause_tags, cur = [], None
    for tok in tokens:
        up = tok.upper()
        if up == "SELECT": cur = "SELECT"
        elif up == "FROM": cur = "FROM"
        elif up in ("WHERE", "HAVING"): cur = "WHERE"
        elif "JOIN" in up: cur = "JOIN"
        elif up in ("GROUP", "ORDER"): cur = "GROUPBY" if up == "GROUP" else "ORDERBY"
        clause_tags.append(cur if cur else "OTHER")
    return tokens, clause_tags

def plot_attention_heatmap(tokens, clauses, matrix, path):
    plt.figure(figsize=(len(tokens)*0.4+1, len(clauses)*0.4+1))
    sns.heatmap(matrix.T, xticklabels=tokens, yticklabels=clauses,
                cmap="Reds", cbar=True)
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    plt.xlabel("Input tokens"); plt.ylabel("SQL clause")
    plt.tight_layout(h_pad=0.2); plt.savefig(path); plt.close()

# ────────── main ──────────
def main(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
                a.model_path,
                attn_implementation="eager"   # silence Sdpa warning
            ).to(device).eval()

    for d in ("preds","viz","probe"): os.makedirs(d, exist_ok=True)

    # --- load examples (JSONL or array) ---
    with open(a.data_file) as f:
        first = f.readline().lstrip(); f.seek(0)
        if first.startswith("{"):
            examples = [json.loads(l) for l in f if l.strip()]
        else:
            obj = json.load(f)
            examples = obj["data"] if isinstance(obj, dict) and "data" in obj else obj
    if a.limit > 0:
        examples = examples[: a.limit]
    print(f"Loaded {len(examples)} examples.")

    features, labels = {"table":[], "column":[]}, {"table":[], "column":[]}

    for idx, ex in enumerate(examples):
        ex_id = ex.get("id") or ex.get("example_id") \
                or ex.get("question_id") or ex.get("db_id") or str(idx)
        db_id = ex.get("db_id", ex_id)
        q     = ex["question"]
        schema= ex.get("schema", "")
        db_path = ex.get("db_path") or (find_sqlite_file(a.db_root, db_id) if a.db_root else None)

        # prompt
        prompt = f"-- Question: {q}\n-- Schema: {schema}\nSELECT"
        inp    = tok(prompt, return_tensors="pt").to(device)

        # generate SQL
        with torch.no_grad():
            gen = model.generate(**inp,
                                 max_new_tokens=a.max_new_tokens,
                                 output_attentions=True,
                                 output_hidden_states=True,
                                 return_dict_in_generate=True)
        sql_pred = tok.decode(gen.sequences[0], skip_special_tokens=True)[len(prompt):].strip()
        open(f"preds/{ex_id}_pred.sql","w").write(sql_pred)

        # forward pass on full sequence for attention+hidden
        full_ids = tok(prompt+" "+sql_pred, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model(full_ids,
                        output_attentions=True,
                        output_hidden_states=True)
        attn = torch.stack([a[0] for a in out.attentions])     # (L,H,S,S)
        hidden = out.hidden_states                             # list(L+1) of (1,S,D)

        S = full_ids.size(1); P = inp.input_ids.size(1); T = S-P
        sql_tokens, clause_tags = tag_sql_clauses(sql_pred)
        clause_names = sorted({c for c in clause_tags if c})

        # clause-level attention heat-map
        heat = np.zeros((P, len(clause_names))); c2i = {c:i for i,c in enumerate(clause_names)}
        L,H = attn.shape[:2]
        for l in range(L):
            for h in range(H):
                M = attn[l,h].cpu().numpy()
                for o, clause in enumerate(clause_tags):
                    if o >= T or clause not in c2i: continue
                    heat[:, c2i[clause]] += M[:P, P+o]
        heat /= (L*H)
        plot_attention_heatmap(tok.convert_ids_to_tokens(full_ids[0])[:P],
                               clause_names, heat,
                               f"viz/{ex_id}_attention_heatmap.png")

        # schema-concept probe feats
        tbls = re.findall(r"CREATE TABLE (\w+)", schema, re.I)
        cols = [c.strip() for grp in re.findall(r"\(([^)]+)\)", schema) for c in grp.split(",")]
        layer_idx = a.probe_layer if a.probe_layer < len(hidden) else -1
        hid_vecs  = hidden[layer_idx][0].cpu().numpy()          # (S,D)
        for pos, tok_str in enumerate(sql_tokens):
            if P+pos >= hid_vecs.shape[0]: break
            v = hid_vecs[P+pos]
            features["table"].append(v);  labels["table"].append(int(tok_str in tbls))
            features["column"].append(v); labels["column"].append(int(tok_str in cols))

        # counter-factual SQL (optional)
        cf_q, cf_schema = q, schema
        for k,v in a.cf_map.items():
            cf_q = cf_q.replace(k,v); cf_schema = cf_schema.replace(k,v)
        if cf_q != q or cf_schema != schema:
            cf_prompt = f"-- Question: {cf_q}\n-- Schema: {cf_schema}\nSELECT"
            cf_ids = tok(cf_prompt, return_tensors="pt").to(device)
            cf_sql = tok.decode(model.generate(**cf_ids, max_new_tokens=a.max_new_tokens)[0],
                                skip_special_tokens=True)[len(cf_prompt):].strip()
            open(f"preds/{ex_id}_cf_pred.sql","w").write(cf_sql)

        # execution test
        exec_ok = False
        if db_path and os.path.isfile(db_path):
            try: sqlite3.connect(db_path).execute(sql_pred); exec_ok=True
            except Exception: exec_ok=False
        open(f"preds/{ex_id}_exec.txt","w").write(f"Executable: {exec_ok}\n")

        # Δ-logP attribution for prompt tokens
        tgt_ids  = tok(sql_pred, return_tensors="pt").input_ids.to(device)
        full_ids = torch.cat([inp.input_ids, tgt_ids], dim=1)
        labels   = full_ids.clone(); labels[:, :P] = -100
        with torch.no_grad():
            base_lp = model(input_ids=full_ids, labels=labels).loss.item()*tgt_ids.size(1)
        deltas=[]
        for j in range(P):
            masked = full_ids.clone(); masked[0,j] = tok.pad_token_id
            with torch.no_grad():
                lp = model(input_ids=masked, labels=labels).loss.item()*tgt_ids.size(1)
            deltas.append(base_lp - lp)
        plt.figure(figsize=(P*0.3+1,2.5))
        sns.barplot(x=tok.convert_ids_to_tokens(full_ids[0])[:P], y=deltas, color="skyblue")
        plt.xticks(rotation=90); plt.ylabel("Δ log-prob"); plt.tight_layout(h_pad=0.2)
        plt.savefig(f"viz/{ex_id}_attr.png"); plt.close()

        print(f"Processed {ex_id}: exec={exec_ok}")

    # train simple logistic probes
    for c in ("table","column"):
        X=np.stack(features[c]); y=np.array(labels[c])
        if len(np.unique(y))<2:
            open(f"probe/{c}_probe.txt","w").write("Not enough positive/negative examples\n")
            continue
        clf=LogisticRegression(max_iter=1000).fit(X,y)
        acc=clf.score(X,y)
        open(f"probe/{c}_probe.txt","w").write(f"Accuracy: {acc:.3f}\n‖w‖: {np.linalg.norm(clf.coef_):.3f}\n")

    print("✓ interpretability evaluation completed")

# ───────── CLI ─────────
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_file", required=True)
    p.add_argument("--db_root", default="")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--probe_layer", type=int, default=-1)
    p.add_argument("--limit", type=int, default=-1,
                   help="Process only first N examples (-1 = all)")
    p.add_argument("--cf_map", type=json.loads, default="{}",
                   help='JSON str for counter-factual replacements, e.g. \'{"orders":"purchases"}\'')
    args=p.parse_args()
    main(args)
