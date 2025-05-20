#!/usr/bin/env python
# interactive_dashboard.py

import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer

st.set_page_config(page_title="Text-to-SQL Interpretability Dashboard", layout="wide")

@st.cache(allow_output_mutation=True)
def load_model(path):
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    mod = AutoModelForCausalLM.from_pretrained(path)
    return tok, mod

st.title("Interactive Text-to-SQL Interpretability Dashboard")

# Sidebar: Model selection and inputs
model_path = st.sidebar.text_input("Model Path", value="path/to/finetuned-llama")
tokenizer, model = load_model(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

question = st.sidebar.text_area("Natural Language Question", height=100)
schema = st.sidebar.text_area("Database Schema", height=100)
max_steps = st.sidebar.number_input("Max Tokens to Generate", min_value=1, max_value=200, value=50)

if st.sidebar.button("Run Generation"):
    prompt = f"-- Question: {question}\n-- Schema: {schema}\nSELECT"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    st.write("**Prompt:**", prompt)
    generated_tokens = []
    st.write("**Generated SQL:**", "")
    columns = st.columns(2)
    col0, col1 = columns

    for step in range(max_steps):
        # Perform one step of generation (greedy)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=1, return_dict_in_generate=True, output_attentions=True
            )
        next_token_id = outputs.sequences[0, -1].unsqueeze(0)
        generated_tokens.append(next_token_id.item())
        # Update inputs for next step
        inputs = tokenizer(prompt + tokenizer.decode(generated_tokens, skip_special_tokens=True),
                           return_tensors="pt").to(device)

        # Decode and display current output
        gen_sql = tokenizer.decode(torch.tensor(generated_tokens), skip_special_tokens=True)
        clause_tokens, clause_tags = gen_sql.split(), []
        # Clause tagging (as before)
        current = None
        for tok in gen_sql.split():
            if tok.upper() in ["SELECT","FROM","WHERE","GROUP","ORDER"] or "JOIN" in tok.upper():
                current = tok.upper().split()[0]
            clause_tags.append(current if current else "OTHER")
        # Color coding: highlight clause keywords
        colored_sql = ""
        for tok, tag in zip(gen_sql.split(), clause_tags):
            color = "#A6CEE3" if tag=="SELECT" else "#B2DF8A" if tag=="FROM" else "#FB9A99" if tag=="WHERE" else "#FDBF6F" if "JOIN" in (tag or "") else "#D9D9D9"
            colored_sql += f"<span style='background-color: {color};'>{tok}</span> "
        col0.markdown(f"<div style='font-family: monospace; font-size: 16px;'>{colored_sql}</div>", unsafe_allow_html=True)
        
        # Compute Δ-logP for latest token
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        target_id = torch.tensor([[next_token_id]])
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=target_id)
        log_probs = F.log_softmax(out.logits, dim=-1)
        base_logp = log_probs[0, -1, next_token_id].item()
        influences = []
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        for i in range(len(input_tokens)):
            mask_ids = torch.cat([input_ids[0,:i], input_ids[0,i+1:]]).unsqueeze(0)
            with torch.no_grad():
                out2 = model(input_ids=mask_ids, labels=target_id)
            logp2 = F.log_softmax(out2.logits, dim=-1)[0, -1, next_token_id].item()
            influences.append(base_logp - logp2)
        fig, ax = plt.subplots(figsize=(4,2))
        sns.barplot(x=input_tokens, y=influences, palette="Blues", ax=ax)
        ax.set_xlabel("Input Token"); ax.set_ylabel("Δ log-prob")
        ax.set_xticklabels(input_tokens, rotation=90)
        col1.pyplot(fig)

        # Compute and display attention heatmap for this step
        with torch.no_grad():
            attn_out = model(input_ids=inputs.input_ids, output_attentions=True)
        attn_tensor = torch.stack([a[0] for a in attn_out.attentions])  # (layers, heads, seq_len, seq_len)
        # Average attention across layers and heads for the last generated token
        avg_attn = attn_tensor[:, :, -1, :-1].mean(dim=0).mean(dim=0)  # shape (seq_len_input)
        fig2, ax2 = plt.subplots(figsize=(4,2))
        sns.barplot(x=input_tokens, y=avg_attn.cpu().numpy(), palette="Reds", ax=ax2)
        ax2.set_xlabel("Input Token"); ax2.set_ylabel("Attention Weight")
        ax2.set_xticklabels(input_tokens, rotation=90)
        col1.pyplot(fig2)

        if next_token_id == tokenizer.eos_token_id:
            break
    st.success("Generation complete.")
