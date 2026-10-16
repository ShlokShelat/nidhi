"""
Definition-grounded evidence extraction (NLI-style clause scoring).

Every prior evidence attempt used BIO token-tagging, which labels tokens
in/out of spans with no notion of WHY a span is evidence. This reformulates
evidence the way the NLI reformulation cracked factors: score each candidate
clause against the DEFINITION of risk evidence, and emit the clauses that
entail it.

Mechanism:
  premise    = a candidate clause from the post
  hypothesis = "This expresses the person's suicidal thoughts, distress,
                intent, plan, or attempt."
  label      = 1 if the clause overlaps a gold evidence span, else 0

Candidate clauses are produced by splitting the post on sentence and clause
delimiters (., !, ?, ;, newline, and comma-with-length). At inference every
candidate is scored and those above a tuned threshold are emitted, mapping
each kept clause back to its verbatim substring in the post (evidence must be
verbatim). Containment scoring rewards this: a kept clause that contains or
is contained by a gold span matches.

Why this can beat the tagger: it injects definitional grounding (the signal
that transferred cleanly for factors), and it changes evidence predictions
WHOLESALE, so val is measurable here (unlike the tiny-delta blends that
overfit val). It directly attacks the emit/don't-emit decision that error
analysis showed is where evidence bleeds (41 hard-zero posts).

Outputs model.pt, tok, thresholds via run_config.json (evidence_min_conf),
so predict.py-style consumers can use it, and a predict path writes an
evidence override CSV.

Usage:
  python src/train_evidence_nli.py \
      --train artifacts/train_split.csv --val artifacts/val_split.csv \
      --model microsoft/deberta-v3-large --outdir runs/evidence_nli --epochs 3
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(__file__))
from data_audit import load
from scoring import parse_evidence, phrase_f1

HYPOTHESIS = ("This expresses the person's suicidal thoughts, distress, "
              "intent, plan, or attempt.")

# split on sentence/clause boundaries, keep the delimiters out of the text
_SPLIT = re.compile(r"[.!?;\n]+|,(?=\s)")


def _norm(s):
    return " ".join(str(s).lower().split())


def clauses_of(post: str):
    """Return list of (clause_text, char_start, char_end) candidate clauses."""
    post = str(post)
    out, i = [], 0
    parts = _SPLIT.split(post)
    for p in parts:
        p_str = p.strip()
        if len(p_str.split()) >= 2:  # skip 1-word fragments
            idx = post.find(p_str, i)
            if idx >= 0:
                out.append((p_str, idx, idx + len(p_str)))
                i = idx + len(p_str)
    # fallback: whole post if nothing split
    if not out and len(post.split()) >= 2:
        out.append((post.strip(), 0, len(post)))
    return out


def clause_label(clause: str, gold_spans: list[str]) -> int:
    """1 if clause overlaps any gold span (containment either direction)."""
    c = _norm(clause)
    for g in gold_spans:
        g = _norm(g)
        if not g:
            continue
        if g in c or c in g:
            return 1
        # token overlap fallback (partial phrase match)
        cg, gg = set(c.split()), set(g.split())
        if gg and len(cg & gg) / len(gg) >= 0.6:
            return 1
    return 0


def build_rows(df):
    rows = []
    for _, r in df.iterrows():
        gold = parse_evidence(r["evidence"])
        for clause, s, e in clauses_of(r["post"]):
            rows.append((clause, clause_label(clause, gold)))
    return rows


class ClauseDS(Dataset):
    def __init__(self, rows, tok, max_len):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        clause, y = self.rows[i]
        enc = self.tok(clause, HYPOTHESIS, truncation="only_first",
                       max_length=self.max_len, padding="max_length",
                       return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(float(y))}


class ClauseScorer(nn.Module):
    def __init__(self, model_name, pos_weight=None):
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(h, 1)
        self.register_buffer("pos_weight",
            torch.tensor([pos_weight], dtype=torch.float) if pos_weight else None)

    def forward(self, input_ids=None, attention_mask=None, label=None, **kw):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        x = self.drop(out.last_hidden_state[:, 0]).to(self.head.weight.dtype)
        logit = self.head(x).squeeze(-1)
        loss = None
        if label is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(
                logit, label.to(logit.dtype), pos_weight=self.pos_weight)
        return {"loss": loss, "logit": logit}


@torch.no_grad()
def predict_spans(model, df, tok, max_len, device, use_bf16, min_conf, batch=64):
    """Return per-post list of kept verbatim clause strings at min_conf."""
    model.eval()
    per_post = []
    for _, r in df.iterrows():
        cands = clauses_of(r["post"])
        if not cands:
            per_post.append([])
            continue
        keep = []
        for s in range(0, len(cands), batch):
            chunk = cands[s:s + batch]
            enc = tok([c for c, _, _ in chunk], [HYPOTHESIS] * len(chunk),
                      truncation="only_first", max_length=max_len,
                      padding=True, return_tensors="pt").to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                logit = model(**enc)["logit"]
            probs = torch.sigmoid(logit.float()).cpu().numpy().reshape(-1)
            for (clause, _, _), p in zip(chunk, probs):
                if p >= min_conf:
                    keep.append(clause)
        per_post.append(keep)
    return per_post


@torch.no_grad()
def score_all_clause_probs(model, df, tok, max_len, device, use_bf16, batch=64):
    """Return per-post list of (clause, prob) for threshold sweeping."""
    model.eval()
    per_post = []
    for _, r in df.iterrows():
        cands = clauses_of(r["post"])
        scored = []
        for s in range(0, len(cands), batch):
            chunk = cands[s:s + batch]
            enc = tok([c for c, _, _ in chunk], [HYPOTHESIS] * len(chunk),
                      truncation="only_first", max_length=max_len,
                      padding=True, return_tensors="pt").to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                logit = model(**enc)["logit"]
            probs = torch.sigmoid(logit.float()).cpu().numpy().reshape(-1)
            for (clause, _, _), p in zip(chunk, probs):
                scored.append((clause, float(p)))
        per_post.append(scored)
    return per_post


def spans_at(scored_per_post, t):
    return [[c for c, p in sp if p >= t] for sp in scored_per_post]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[precision] {'bf16' if use_bf16 else 'fp32'} on {device}")
    if device == "cpu":
        print("WARNING: on CPU — use a GPU node to train.")

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    tr_df = load(args.train)
    va_df = load(args.val)
    tok = AutoTokenizer.from_pretrained(args.model)

    train_rows = build_rows(tr_df)
    npos = sum(y for _, y in train_rows)
    print(f"train clauses: {len(train_rows)} ({npos} pos, "
          f"{len(train_rows)-npos} neg, ratio {(len(train_rows)-npos)/max(npos,1):.1f}:1)")
    pw = float(np.clip((len(train_rows) - npos) / max(npos, 1), 1.0, 4.0))

    loader = DataLoader(ClauseDS(train_rows, tok, args.max_len),
                        batch_size=args.batch, shuffle=True, num_workers=2)
    model = ClauseScorer(args.model, pos_weight=pw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = (len(loader) // args.grad_accum + 1) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    va_gold = [parse_evidence(c) for c in va_df["evidence"]]
    grid = np.arange(0.30, 0.90, 0.05)
    best_f1, best_state, best_conf = -1.0, None, 0.5
    for epoch in range(args.epochs):
        model.train(); opt.zero_grad()
        run = []
        for b, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**batch)["loss"]
            (loss / args.grad_accum).backward()
            run.append(loss.item())
            if (b + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if (b + 1) % 200 == 0:
                print(f"epoch {epoch+1} step {b+1} loss {np.mean(run[-200:]):.4f}")
        scored = score_all_clause_probs(model, va_df, tok, args.max_len, device, use_bf16)
        ep_best_f1, ep_best_t = -1, 0.5
        for t in grid:
            f1 = phrase_f1(va_gold, spans_at(scored, t)).f1
            if f1 > ep_best_f1:
                ep_best_f1, ep_best_t = f1, t
        print(f"== epoch {epoch+1}: val phrase-F1 = {ep_best_f1:.4f} @min_conf={ep_best_t:.2f}")
        if ep_best_f1 > best_f1:
            best_f1, best_conf = ep_best_f1, ep_best_t
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        torch.save(best_state, os.path.join(args.outdir, "model.pt"))
    with open(os.path.join(args.outdir, "tok"), "w") as f:
        f.write(args.model)
    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump({**vars(args), "evidence_min_conf": float(best_conf),
                   "best_val_phrase_f1": float(best_f1)}, f, indent=2)
    print(f"\nbest val phrase-F1 = {best_f1:.4f} @min_conf={best_conf:.2f}")
    print("compare against BIO tagger evidence_v2 (repo-micro 0.6902, official-macro 0.7787)")


if __name__ == "__main__":
    main()
