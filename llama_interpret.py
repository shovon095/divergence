#!/usr/bin/env python3
"""
llama_interpret.py  – v2 (extended)
===================================
Batch Text-to-SQL interpretability for a fine-tuned LLaMA/Llama-2 model.

Original outputs
----------------
viz/<id>_attention_heatmap.png      clause-level avg attention
viz/<id>_attr.png                   Δ-log P bar chart
probe/{table,column}_probe.txt      linear-probe results
preds/<id>_pred.sql                 greedy SQL
preds/<id>_exec.txt                 SQLite execution flag
preds/<id>_cf_pred.sql              counter-factual SQL (if --cf_map)

New optional outputs
--------------------
viz/<id>_rollout.png                attention-rollout heat-map    (--rollout)
viz/<id>_attn_grad.png              Grad × Attn clause map        (--grad_attn)
viz/<id>_join_graph.svg             schema join graph             (always)

CLI changes
-----------
  --rollout      compute attention rollout         (cheap)
  --grad_attn    compute Grad × Attn heat-map      (≈2× slower)

Everything else is unchanged.
"""
from __future__ import annotations

import argparse, json, os, re, sqlite3, warnings
from pathlib import Path
from typing import List, Tuple

import graphviz                          # pip install graphviz
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer

# ════════════════════════════════════════════════════════════════════
# helpers
# ════════════════════════════════════════════════════════════════════
SQL_KWS = ("SELECT", "FROM", "WHERE", "GROUP", "ORDER", "JOIN", "HAVING")


def find_sqlite_file(root: str, db_id: str) -> str | None:
    """Return a .sqlite/.db path inside *root* (Spider-style layout)."""
    if not root:
        return None
    for base in (root, os.path.join(root, db_id)):
        for ext in (".sqlite", ".db", ".sqlite3"):
            cand = os.path.join(base, f"{db_id}{ext}")
            if os.path.isfile(cand):
                return cand
    return None


def tag_sql_clauses(sql: str) -> Tuple[List[str], List[str]]:
    toks = sql.replace(",", " ,").split()
    tags, cur = [], None
    for t in toks:
        up = t.upper()
        if up in SQL_KWS or "JOIN" in up:
            cur = (
                "JOIN"
                if "JOIN" in up
                else "GROUPBY"
                if up == "GROUP"
                else "ORDERBY"
                if up == "ORDER"
                else up
            )
        tags.append(cur or "OTHER")
    return toks, tags


def save_heat(tokens, clauses, mat, path):
    plt.figure(figsize=(len(tokens) * 0.4 + 1, len(clauses) * 0.4 + 1))
    sns.heatmap(
        mat.T,
        xticklabels=tokens,
        yticklabels=clauses,
        cmap="Reds",
        cbar=True,
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel("Input tokens")
    plt.ylabel("SQL clause")
    plt.tight_layout(h_pad=0.2)
    plt.savefig(path)
    plt.close()


def draw_join_graph(sql: str, out_path: str):
    tables = {m.group(1) for m in re.finditer(r"(?:FROM|JOIN)\s+(\w+)", sql, re.I)}
    if not tables:
        return
    dot = graphviz.Digraph()
    for t in tables:
        dot.node(t)
    for tbl, lhs, rhs in re.findall(
        r"JOIN\s+(\w+)\s+ON\s+(\w+\.\w+)\s*=\s*(\w+\.\w+)", sql, re.I
    ):
        dot.edge(lhs.split(".")[0], tbl)
        dot.edge(tbl, rhs.split(".")[0])
    try:
        dot.render(out_path, format="svg", cleanup=True)
    except Exception as e:
        warnings.warn(f"Graphviz render failed: {e}")


# ════════════════════════════════════════════════════════════════════
# advanced attribution
# ════════════════════════════════════════════════════════════════════
def attention_rollout(attn: torch.Tensor) -> torch.Tensor:
    """
    Average-head rollout (Eyeball et al. 2020).

    attn: (L,H,S,S) with softmax already applied.
    returns: (S,S) cumulative influence matrix.
    """
    L, H, S, _ = attn.shape
    eye = torch.eye(S, device=attn.device)
    R = eye.clone()
    for l in range(L):
        A = attn[l].mean(0) + eye           # add residual
        A /= A.sum(-1, keepdim=True)
        R = A @ R
    return R


def grad_times_attn(model, seq_ids: torch.Tensor, target_pos: int) -> torch.Tensor:
    """
    Grad × Attn on the **last layer** (avg heads).
    """
    model.zero_grad(set_to_none=True)
    seq_ids = seq_ids.clone().requires_grad_(True)
    out = model(seq_ids, output_attentions=True)
    loss = -F.log_softmax(out.logits, -1)[0, target_pos, seq_ids[0, target_pos]]
    loss.backward()

    last_attn = out.attentions[-1][0].detach()          # (H,S,S)
    grads = (
        out.attentions[-1].grad[0]
        if out.attentions[-1].grad is not None
        else torch.zeros_like(last_attn)
    )
    return (last_attn * grads).mean(0)                  # (S,S)


# ════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════
def main(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(argv.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        argv.model_path, attn_implementation="eager"
    ).to(device).eval()

    for d in ("preds", "viz", "probe"):
        Path(d).mkdir(exist_ok=True)

    # load evaluation set
    with open(argv.data_file) as f:
        first = f.readline().lstrip()
        f.seek(0)
        if first.startswith("{"):                         # JSON Lines
            examples = [json.loads(l) for l in f if l.strip()]
        else:                                             # JSON dict/array
            obj = json.load(f)
            examples = obj.get("data", obj)
    if argv.limit > 0:
        examples = examples[: argv.limit]
    print(f"Loaded {len(examples)} examples")

    feats, labs = {k: [] for k in ("table", "column")}, {k: [] for k in ("table", "column")}

    # ───────── process each example ─────────
    for idx, ex in enumerate(examples):
        ex_id = ex.get("id") or ex.get("db_id") or str(idx)
        db_id = ex.get("db_id", ex_id)
        q = ex["question"]
        schema = ex.get("schema", "")
        db_path = ex.get("db_path") or find_sqlite_file(argv.db_root, db_id)

        prompt = f"-- Question: {q}\n-- Schema: {schema}\nSELECT"
        inp = tok(prompt, return_tensors="pt").to(device)

        # greedy generation --------------------------------------------------
        with torch.no_grad():
            gen = model.generate(
                **inp,
                max_new_tokens=argv.max_new_tokens,
                output_attentions=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        sql_pred = tok.decode(gen.sequences[0], skip_special_tokens=True)[len(prompt) :].strip()
        Path(f"preds/{ex_id}_pred.sql").write_text(sql_pred)

        # forward pass on prompt + SQL ---------------------------------------
        full_ids = tok(prompt + " " + sql_pred, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model(full_ids, output_attentions=True, output_hidden_states=True)
        attn = torch.stack([a[0] for a in out.attentions])      # (L,H,S,S)
        S, P = full_ids.size(1), inp.input_ids.size(1)

        # clause tagging -----------------------------------------------------
        sql_toks, clause_tags = tag_sql_clauses(sql_pred)
        clause_names = sorted({c for c in clause_tags if c})
        c2i = {c: i for i, c in enumerate(clause_names)}

        # clause-level avg attention -----------------------------------------
        heat = np.zeros((P, len(clause_names)))
        L, H = attn.shape[:2]
        for l in range(L):
            for h in range(H):
                M = attn[l, h].cpu().numpy()
                for pos, cl in enumerate(clause_tags):
                    if cl in c2i and pos < S - P:
                        heat[:, c2i[cl]] += M[:P, P + pos]
        heat /= L * H
        save_heat(
            tok.convert_ids_to_tokens(full_ids[0])[:P],
            clause_names,
            heat,
            f"viz/{ex_id}_attention_heatmap.png",
        )

        # attention rollout (optional) ---------------------------------------
        if argv.rollout:
            R = attention_rollout(attn).cpu().numpy()
            roll = np.zeros_like(heat)
            for pos, cl in enumerate(clause_tags):
                if cl in c2i and pos < S - P:
                    roll[:, c2i[cl]] += R[:P, P + pos]
            save_heat(
                tok.convert_ids_to_tokens(full_ids[0])[:P],
                clause_names,
                roll,
                f"viz/{ex_id}_rollout.png",
            )

        # Grad × Attn (optional) ---------------------------------------------
        if argv.grad_attn:
            try:
                GA = grad_times_attn(model, full_ids, S - 1).cpu().numpy()
                gmat = np.zeros_like(heat)
                for pos, cl in enumerate(clause_tags):
                    if cl in c2i and pos < S - P:
                        gmat[:, c2i[cl]] += GA[:P, P + pos]
                save_heat(
                    tok.convert_ids_to_tokens(full_ids[0])[:P],
                    clause_names,
                    gmat,
                    f"viz/{ex_id}_attn_grad.png",
                )
            except RuntimeError as e:
                warnings.warn(f"Grad×Attn failed on {ex_id}: {e}")

        # join graph ----------------------------------------------------------
        draw_join_graph(sql_pred, f"viz/{ex_id}_join_graph")

        # concept probe vectors ----------------------------------------------
        tbls = re.findall(r"CREATE TABLE (\w+)", schema, re.I)
        cols = [
            c.strip()
            for grp in re.findall(r"\(([^)]+)\)", schema)
            for c in grp.split(",")
        ]
        lay_idx = argv.probe_layer if argv.probe_layer < len(out.hidden_states) else -1
        vecs = out.hidden_states[lay_idx][0].cpu().numpy()
        for pos, tok_str in enumerate(sql_toks[: vecs.shape[0] - P]):
            v = vecs[P + pos]
            feats["table"].append(v)
            labs["table"].append(int(tok_str in tbls))
            feats["column"].append(v)
            labs["column"].append(int(tok_str in cols))

        # counter-factual SQL -------------------------------------------------
        if argv.cf_map:
            cf_q, cf_schema = q, schema
            for k, v in argv.cf_map.items():
                cf_q, cf_schema = cf_q.replace(k, v), cf_schema.replace(k, v)
            if (cf_q, cf_schema) != (q, schema):
                cf_prompt = f"-- Question: {cf_q}\n-- Schema: {cf_schema}\nSELECT"
                cf_ids = tok(cf_prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    cf_seq = model.generate(**cf_ids, max_new_tokens=argv.max_new_tokens)[0]
                cf_sql = tok.decode(cf_seq, skip_special_tokens=True)[len(cf_prompt) :].strip()
                Path(f"preds/{ex_id}_cf_pred.sql").write_text(cf_sql)

        # execution check -----------------------------------------------------
        exec_ok = False
        if db_path and os.path.isfile(db_path):
            try:
                sqlite3.connect(db_path).execute(sql_pred)
                exec_ok = True
            except Exception:
                exec_ok = False
        Path(f"preds/{ex_id}_exec.txt").write_text(f"Executable: {exec_ok}\n")

        # Δ-log P attribution (prompt tokens) --------------------------------
        tgt_ids = tok(sql_pred, return_tensors="pt").input_ids.to(device)
        full_lbl = torch.cat([inp.input_ids, tgt_ids], 1)
        labels = full_lbl.clone()
        labels[:, :P] = -100
        with torch.no_grad():
            base_lp = (
                model(input_ids=full_lbl, labels=labels).loss.item()
                * tgt_ids.size(1)
            )
        deltas = []
        for j in range(P):
            masked = full_lbl.clone()
            masked[0, j] = tok.pad_token_id
            with torch.no_grad():
                lp = (
                    model(input_ids=masked, labels=labels).loss.item()
                    * tgt_ids.size(1)
                )
            deltas.append(base_lp - lp)
        plt.figure(figsize=(P * 0.3 + 1, 2.5))
        sns.barplot(
            x=tok.convert_ids_to_tokens(full_lbl[0])[:P],
            y=deltas,
            color="skyblue",
        )
        plt.xticks(rotation=90)
        plt.ylabel("Δ log-prob")
        plt.tight_layout(h_pad=0.2)
        plt.savefig(f"viz/{ex_id}_attr.png")
        plt.close()

        print(f"Processed {ex_id}: exec_ok={exec_ok}")

    # probe training ---------------------------------------------------------
    for concept in ("table", "column"):
        if not feats[concept]:
            Path(f"probe/{concept}_probe.txt").write_text("No vectors collected.\n")
            continue
        X = np.stack(feats[concept])
        y = np.array(labs[concept])
        if len(np.unique(y)) < 2:
            Path(f"probe/{concept}_probe.txt").write_text(
                "Positive/negative imbalance.\n"
            )
            continue
        clf = LogisticRegression(max_iter=1000).fit(X, y)
        acc = clf.score(X, y)
        norm = np.linalg.norm(clf.coef_)
        Path(f"probe/{concept}_probe.txt").write_text(
            f"Accuracy: {acc:.3f}\n||w||: {norm:.3f}\n"
        )

    print("✓ interpretability evaluation completed")


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="fine-tuned LLaMA checkpoint or HF repo")
    p.add_argument("--data_file", required=True, help="dev file (JSONL or JSON array)")
    p.add_argument("--db_root", default="", help="root dir with <db_id>/<db_id>.sqlite")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--probe_layer", type=int, default=-1, help="hidden layer to probe (-1 = last)")
    p.add_argument("--limit", type=int, default=-1, help="only first N examples (-1 = all)")
    p.add_argument(
        "--cf_map",
        type=json.loads,
        default="{}",
        help='JSON string of counter-factual replacements (e.g. \'{"orders":"purchases"}\')',
    )
    # new flags
    p.add_argument("--rollout", action="store_true", help="save attention-rollout heat-maps")
    p.add_argument("--grad_attn", action="store_true", help="save Grad×Attn heat-maps")
    args = p.parse_args()
    main(args)
