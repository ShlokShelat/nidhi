"""
Isolate which row(s) of the poison batch break the model, and how.

Usage (GPU node):
  python src/isolate_poison.py runs/risk_v0_manual/poison_batch.csv

Per row, replicating training conditions exactly (user-context prefix,
truncation 512, padding to max_length):
  - token id range vs the embedding table size (out-of-range ids = the
    v5.9 tokenizer-regex bug -> environment fix, not code fix)
  - NaN in the bare encoder's hidden states, bf16 and fp32 separately
  - a preview of the offending post
"""

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")          # no hub calls; cache only (override: HF_HUB_OFFLINE=0)
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

csv_path = sys.argv[1] if len(sys.argv) > 1 else "runs/risk_v0_manual/poison_batch.csv"
model_name = sys.argv[2] if len(sys.argv) > 2 else "microsoft/deberta-v3-base"

df = pd.read_csv(csv_path)
tok = AutoTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(model_name).to(device).eval()
emb_rows = model.get_input_embeddings().weight.shape[0]
print(f"embedding table rows: {emb_rows} | device: {device}\n")

bad_rows = []
for _, row in df.iterrows():
    # replicate training text construction (prefix values don't matter here)
    text = f"[USER_POST 1/1] {row['post']}"
    enc = tok(text, truncation=True, max_length=512,
              padding="max_length", return_tensors="pt")
    mn = enc["input_ids"].min().item()
    mx = enc["input_ids"].max().item()
    oob = mx >= emb_rows or mn < 0

    flags = {"oob_ids": oob}
    for label, use_bf16 in [("bf16", True), ("fp32", False)]:
        if use_bf16 and device != "cuda":
            flags[label] = None
            continue
        with torch.no_grad(), torch.autocast(device_type=device,
                                             dtype=torch.bfloat16,
                                             enabled=use_bf16):
            h = model(input_ids=enc["input_ids"].to(device),
                      attention_mask=enc["attention_mask"].to(device)
                      ).last_hidden_state
        flags[label] = bool(torch.isnan(h).any() or torch.isinf(h).any())

    status = "BAD " if (oob or flags.get("bf16") or flags.get("fp32")) else "ok  "
    print(f"{status} {row['row_id']}: ids=[{mn},{mx}] oob={oob} "
          f"nan_bf16={flags.get('bf16')} nan_fp32={flags.get('fp32')} "
          f"chars={len(str(row['post']))}")
    if status == "BAD ":
        bad_rows.append(row)
        print(f"      post starts: {str(row['post'])[:150]!r}")
        toks = tok.convert_ids_to_tokens(enc["input_ids"][0][:30].tolist())
        print(f"      first tokens: {toks}")

if not bad_rows:
    print("\nNo single row reproduces it -> the failure needs the BATCH "
          "(padding interaction). Report this; next test is the full batch "
          "of 8 together.")
else:
    print(f"\n{len(bad_rows)} poison row(s) found. If oob=True: tokenizer "
          "bug in transformers 5.9 -> use the pinned stable env. If ids in "
          "range but nan=True: paste the post preview above.")
