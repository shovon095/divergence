#!/usr/bin/env python
# interactive_dashboard.py
#
# Live Text-to-SQL decoder with token-level attribution & attention tracing.

from __future__ import annotations
import argparse, sys, streamlit as st
import torch, torch.nn.functional as F
import matplotlib.pyplot as plt, seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
import sqlparse

# ───────── optional CLI flag (so you can pass --ckpt) ─────────
cli = argparse.ArgumentParser(add_help=False)
cli.add_argument("--ckpt", default="", help="path to fine-tuned model")
cli_args, _ = cli.parse_known_args()

# ───────── Streamlit basic config ─────────
st.set_page_config(page_title="Text-to-SQL Dashboard", layout="wide")
st.title("📡 Interactive Text-to-SQL Interpretability Dashboard")

# ───────── cached loader ─────────
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
            path, attn_implementation="eager", device_map="auto").eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, mdl

# ───────── sidebar inputs ─────────
model_path = st.sidebar.text_input("Model Path",
                                   value=cli_args.ckpt or "checkpoints/llama_bird_ft")
tok, model = load_model(model_path)
device = next(model.parameters()).device

question = st.sidebar.text_area("Natural-language question", height=120)
schema   = st.sidebar.text_area("Schema (paste DDL subset)",  height=120)
max_steps= st.sidebar.number_input("Max SQL tokens to generate",
                                   min_value=1, max_value=256, value=128)

if st.sidebar.button("Run Generation"):
    prompt = f"-- Question: {question}\n-- Schema: {schema}\nSELECT"
    st.markdown("#### Prompt")
    st.code(prompt, language="sql")

    # placeholders for live updates
    sql_box   = st.empty()
    delta_box = st.empty()
    attn_box  = st.empty()

    inputs  = tok(prompt, return_tensors="pt").to(device)
    seq_ids = inputs.input_ids
    generated = ""

    for step in range(max_steps):
        # --- forward, capture logits & attention for *current* last token
        with torch.no_grad():
            out = model(seq_ids, output_attentions=True)
        logits = out.logits[:, -1, :]                     # (1,V)
        next_id = torch.argmax(logits, -1, keepdim=True)  # greedy
        seq_ids = torch.cat([seq_ids, next_id], -1)

        # append decoded token
        tok_str = tok.decode(next_id[0])
        generated += tok_str
        # clause-aware colour coding  ────────────────────────────
        clause = None
        coloured_sql = ""
        for t in sqlparse.format(generated, strip_comments=True).split():
            up = t.upper()
            if up in ("SELECT", "FROM", "WHERE", "GROUP", "ORDER") or "JOIN" in up:
                clause = up.split()[0]
            colour = {"SELECT":"#A6CEE3", "FROM":"#B2DF8A",
                      "WHERE":"#FB9A99",  "GROUP":"#FDBF6F",
                      "ORDER":"#CAB2D6"}.get(clause, "#D9D9D9")
            coloured_sql += f"<span style='background-color:{colour}'>{t}</span> "
        sql_box.markdown(f"<div style='font-family: monospace; font-size:16px;'>{coloured_sql}</div>",
                         unsafe_allow_html=True)

        # --- Δ-logP bar chart for prompt tokens  ────────────────
        base_logp = F.log_softmax(logits, -1)[0, next_id.item()].item()
        prompt_tokens = tok.convert_ids_to_tokens(seq_ids[0][:inputs.input_ids.size(1)])
        deltas = []
        for i in range(inputs.input_ids.size(1)):
            masked = seq_ids.clone()
            masked[0, i] = tok.pad_token_id
            with torch.no_grad():
                lp = F.log_softmax(model(masked).logits[:, -1, :], -1)[0, next_id.item()]
            deltas.append(base_logp - lp.item())
        fig, ax = plt.subplots(figsize=(max(4, len(prompt_tokens)*0.25), 2.2))
        sns.barplot(x=prompt_tokens, y=deltas, palette="Blues_d", ax=ax)
        ax.set_ylabel("Δ log P"); ax.set_xticklabels(prompt_tokens, rotation=90)
        fig.tight_layout()
        delta_box.pyplot(fig); plt.close(fig)

        # --- Prompt-token attention heat-map for this step ──────
        attn = torch.stack([a[0] for a in out.attentions])       # (L,H,S,S)
        attn_prompt = attn[:, :, -1, :inputs.input_ids.size(1)].mean(0).mean(0)
        fig2, ax2 = plt.subplots(figsize=(max(4, len(prompt_tokens)*0.25), 1.8))
        sns.barplot(x=prompt_tokens, y=attn_prompt.cpu().numpy(),
                    palette="Reds_d", ax=ax2)
        ax2.set_ylabel("Attention"); ax2.set_xticklabels(prompt_tokens, rotation=90)
        fig2.tight_layout()
        attn_box.pyplot(fig2); plt.close(fig2)

        # stop at semicolon or EOS
        if tok_str.strip().endswith(";") or next_id.item() == tok.eos_token_id:
            break

    st.success("Generation complete ✓")
