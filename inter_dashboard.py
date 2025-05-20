#!/usr/bin/env python3
"""
Interactive Text‑to‑SQL Interpretability Dashboard (v2)
------------------------------------------------------
Live decoder with:
  • Token‑level Δ‑log P attribution (leave‑one‑out)
  • Per‑step prompt‑token attention bars
  • **NEW** live clause‑level attention heat‑map
  • **NEW** optional gradient attribution (Integrated Gradients)
  • **NEW** join‑graph visualisation of SELECT‑FROM‑JOIN structure
Designed for research‑local use (single‑GPU).  Heavy ops (gradients) are
optional and off the main decode loop.
"""
from __future__ import annotations
import argparse, re, sys, time, typing as tp
import streamlit as st
import altair as alt
import graphviz
import torch, torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from captum.attr import LayerIntegratedGradients  # optional, slow
import sqlparse

# ───────── CLI & Streamlit config ─────────
cli = argparse.ArgumentParser(add_help=False)
cli.add_argument("--ckpt", default="", help="path to fine‑tuned model")
cli_args, _ = cli.parse_known_args()

st.set_page_config(page_title="Text‑to‑SQL Dashboard", layout="wide")
st.title("📡 Interactive Text‑to‑SQL Interpretability Dashboard (v2)")

# ───────── model cache ─────────
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        path, attn_implementation="eager", device_map="auto").eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, mdl

# ───────── sidebar controls ─────────
model_path = st.sidebar.text_input("Model path", cli_args.ckpt or "checkpoints/llama_bird_ft")
tok, model = load_model(model_path)

device = next(model.parameters()).device

question = st.sidebar.text_area("Natural‑language question", height=120)
schema   = st.sidebar.text_area("Schema (DDL subset)", height=120)
max_steps= st.sidebar.number_input("Max SQL tokens", 1, 256, 128)

st.sidebar.markdown("### Visualisations")
show_heatmap     = st.sidebar.checkbox("Live clause‑level heat‑map", value=True)
show_gradients   = st.sidebar.checkbox("Compute gradient attribution (slow)")
show_join_graph  = st.sidebar.checkbox("Show join graph", value=True)

if st.sidebar.button("Run 🚀"):
    prompt = f"-- Question: {question}\n-- Schema: {schema}\nSELECT"
    st.markdown("#### Prompt")
    st.code(prompt, language="sql")

    # placeholders
    sql_box   = st.empty()
    delta_box = st.empty()
    attn_box  = st.empty()
    heat_box  = st.empty() if show_heatmap else None
    graph_box = st.empty() if show_join_graph else None

    # encode prompt
    inputs  = tok(prompt, return_tensors="pt").to(device)
    seq_ids = inputs.input_ids.clone()
    P       = seq_ids.size(1)
    prompt_tokens = tok.convert_ids_to_tokens(seq_ids[0])

    # live clause‑map accumulator
    clause_names = ["SELECT", "FROM", "WHERE", "GROUP", "ORDER", "JOIN"]
    c2idx = {c: i for i, c in enumerate(clause_names)}
    clause_attn = torch.zeros((P, len(clause_names)), device=device)

    generated = ""

    for step in range(max_steps):
        with torch.no_grad():
            out = model(seq_ids, output_attentions=True)
        logits = out.logits[:, -1, :]
        next_id = torch.argmax(logits, -1, keepdim=True)  # greedy
        seq_ids = torch.cat([seq_ids, next_id], -1)

        # decode next token
        tok_str = tok.decode(next_id[0])
        generated += tok_str

        # ── colourised SQL view ──
        clause = None
        coloured_sql = ""
        for t in sqlparse.format(generated, strip_comments=True).split():
            up = t.upper()
            if up in ("SELECT", "FROM", "WHERE", "GROUP", "ORDER") or "JOIN" in up:
                clause = up.split()[0]
            colour = {"SELECT":"#A6CEE3", "FROM":"#B2DF8A", "WHERE":"#FB9A99",
                      "GROUP":"#FDBF6F", "ORDER":"#CAB2D6", "JOIN":"#FFCC99"}.get(clause, "#D9D9D9")
            coloured_sql += f"<span style='background-color:{colour}'>{t}</span> "
        sql_box.markdown(f"<div style='font-family:monospace;font-size:16px;'>{coloured_sql}</div>", unsafe_allow_html=True)

        # ── Δ‑logP bar (prompt tokens) ──
        base_logp = F.log_softmax(logits, -1)[0, next_id.item()].item()
        deltas = []
        for i in range(P):
            masked = seq_ids.clone()
            masked[0, i] = tok.pad_token_id
            with torch.no_grad():
                lp = F.log_softmax(model(masked).logits[:, -1, :], -1)[0, next_id.item()]
            deltas.append(base_logp - lp.item())
        fig, ax = plt.subplots(figsize=(max(4, P*0.25), 2.2))
        sns.barplot(x=prompt_tokens, y=deltas, palette="Blues_d", ax=ax)
        ax.set_ylabel("Δ log P"); ax.set_xticklabels(prompt_tokens, rotation=90)
        fig.tight_layout(); delta_box.pyplot(fig); plt.close(fig)

        # ── prompt‑token attention bar ──
        attn = torch.stack([a[0] for a in out.attentions])          # (L,H,S,S)
        attn_prompt = attn[:, :, -1, :P].mean((0,1))                # (P,)
        fig2, ax2 = plt.subplots(figsize=(max(4, P*0.25), 1.8))
        sns.barplot(x=prompt_tokens, y=attn_prompt.cpu(), palette="Reds_d", ax=ax2)
        ax2.set_ylabel("Attention"); ax2.set_xticklabels(prompt_tokens, rotation=90)
        fig2.tight_layout(); attn_box.pyplot(fig2); plt.close(fig2)

        # ── accumulate clause‑level attention ──
        if clause in c2idx:
            clause_attn[:, c2idx[clause]] += attn_prompt

        # ── live heat‑map ──
        if show_heatmap and (step == 0 or clause in c2idx):
            df = pd.DataFrame({
                "input_token": sum([[t]*len(clause_names) for t in prompt_tokens], []),
                "clause":      clause_names * P,
                "score":       clause_attn.flatten().cpu().tolist(),
            })
            chart = alt.Chart(df).mark_rect().encode(
                x=alt.X("input_token:N", sort=None, axis=alt.Axis(labelAngle=-90)),
                y="clause:N",
                color=alt.Color("score:Q", scale=alt.Scale(scheme="reds")),
                tooltip=["input_token","clause",alt.Tooltip("score:Q", format=".3f")]
            ).properties(height=180)
            heat_box.altair_chart(chart, use_container_width=True)

        # stop at semicolon or EOS
        if tok_str.strip().endswith(";") or next_id.item() == tok.eos_token_id:
            break

    st.success("Generation complete ✓")

    # ───────── post‑hoc gradient attribution (optional) ─────────
    if show_gradients:
        st.info("Computing Integrated Gradients … might take a minute ⏳")
        lig = LayerIntegratedGradients(model, model.get_input_embeddings())
        full_ids = tok(prompt + generated, return_tensors="pt").input_ids.to(device)
        target_index = full_ids.size(1) - 1  # last token log‑prob
        def fwd(inp):
            out = model(inp)
            return out.logits[:, target_index, :]
        baseline = torch.full_like(full_ids, tok.pad_token_id)
        attributions, _ = lig.attribute(full_ids, baselines=baseline, return_convergence_delta=True)
        attr = attributions.sum(-1)[0, :P].detach().cpu()
        fig3, ax3 = plt.subplots(figsize=(max(4, P*0.25), 2.2))
        sns.barplot(x=prompt_tokens, y=attr, palette="Purples_d", ax=ax3)
        ax3.set_ylabel("IG score"); ax3.set_xticklabels(prompt_tokens, rotation=90)
        fig3.tight_layout(); st.pyplot(fig3); plt.close(fig3)

    # ───────── join graph visualisation ─────────
    if show_join_graph:
        try:
            tables = re.findall(r"FROM\s+(\w+)|JOIN\s+(\w+)", generated, re.I)
            tables = {t for tup in tables for t in tup if t}
            dot = graphviz.Digraph()
            for t in tables: dot.node(t)
            joins = re.findall(r"JOIN\s+(\w+).*?ON\s+(\w+\.\w+)\s*=\s*(\w+\.\w+)", generated, re.I)
            for tbl, lhs, rhs in joins:
                left_tbl = lhs.split(".")[0]; right_tbl = rhs.split(".")[0]
                dot.edge(left_tbl, tbl)
                dot.edge(tbl, right_tbl)
            graph_box.graphviz_chart(dot, use_container_width=True)
        except Exception as e:
            st.warning(f"Graphviz error: {e}")
