"""
LLM evidence extraction — Qwen2.5-14B prompted as a psychiatrist to extract
VERBATIM risk-evidence spans, grounded in the risk-level definitions.

Motivation (from the literature): the Dual-Prompting work (arXiv 2402.14854)
found that for evidence extraction specifically, prompting an LLM with an
EXPERT (psychiatrist) IDENTITY plus suicide-specific grounding is the
strongest approach — beating discriminative taggers. The IEEE 2024 winner
(arXiv 2410.04501) fused big-LLM extraction with fine-tuned encoders. The
BIO tagger sits at official-macro phrase-F1 0.7787 and every encoder-side
tweak is exhausted; evidence is 0.3 of the composite and the most-neglected
subtask. This is a genuinely different mechanism (generative extraction).

The model is asked to copy EXACT substrings from the post that express the
person's suicidal thoughts, distress, intent, plan, or attempt. Only outputs
that are verbatim substrings of the post are kept (evidence must be
verbatim for containment scoring), deduped, and capped by count.

Validate on val_split gold first (official per-post macro), then, if it beats
or complements the tagger, write the test override CSV. Also saves per-post
spans so they can be FUSED (agreement-filtered) with the tagger later.

Usage (GPU node):
  python src/evidence_llm.py --split artifacts/val_split.csv \
      --qwen <QWEN_PATH> --out artifacts/evidence_llm_val.csv --score
  python src/evidence_llm.py --split data/test.csv \
      --qwen <QWEN_PATH> --out artifacts/evidence_llm_test.csv
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from data_audit import load
from scoring import parse_evidence, phrase_f1

SYSTEM = ("You are an experienced clinical psychiatrist specializing in "
          "suicide risk assessment.")

INSTRUCTION = """You are given a social media post. Extract the exact spans of text that provide evidence of the author's suicide risk — that is, phrases expressing suicidal thoughts, a wish to die, hopelessness tied to wanting to end life, self-harm, a suicide plan, the means or method, or a past/present attempt.

Rules:
- Copy phrases WORD-FOR-WORD from the post. Do not paraphrase, correct, or add words.
- Extract only the risk-relevant phrases, not whole sentences of unrelated context.
- If the post contains no suicide-risk evidence, output exactly: NONE
- Separate multiple phrases with a semicolon and a space.

Post:
\"\"\"{post}\"\"\"

Evidence phrases:"""


def _norm(s):
    return " ".join(str(s).lower().split())


def official_post_f1(gold, pred):
    g = [_norm(x) for x in gold if _norm(x)]
    p = [_norm(x) for x in pred if _norm(x)]
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    used, matched = set(), 0
    for pi in p:
        for gi, gv in enumerate(g):
            if gi in used:
                continue
            if (gv in pi or pi in gv) and len(pi.split()) <= 3 * len(gv.split()):
                used.add(gi); matched += 1; break
    if matched == 0:
        return 0.0
    prec, rec = matched / len(p), matched / len(g)
    return 2 * prec * rec / (prec + rec)


def official_f1(gl, pl):
    return float(np.mean([official_post_f1(g, p) for g, p in zip(gl, pl)]))


def extract_verbatim(raw, post, max_spans=8):
    """Keep only outputs that are verbatim substrings of the post."""
    if not raw or raw.strip().upper().startswith("NONE"):
        return []
    post_low = post.lower()
    out = []
    for piece in raw.split(";"):
        piece = piece.strip().strip('"').strip("'").strip()
        # trim a trailing period the model may add
        piece = piece.rstrip(".")
        if len(piece.split()) < 1:
            continue
        if piece.lower() in post_low and piece not in out:
            out.append(piece)
        else:
            # try to recover: find the longest verbatim window of the piece
            words = piece.split()
            for start in range(len(words)):
                for end in range(len(words), start, -1):
                    cand = " ".join(words[start:end])
                    if len(cand.split()) >= 2 and cand.lower() in post_low:
                        if cand not in out:
                            out.append(cand)
                        break
                else:
                    continue
                break
    return out[:max_spans]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--qwen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--score", action="store_true",
                    help="if set, compute official phrase-F1 vs gold in --split")
    ap.add_argument("--max-new", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.qwen)
    model = AutoModelForCausalLM.from_pretrained(args.qwen, torch_dtype=torch.bfloat16)
    model.to(device); model.eval()

    df = load(args.split)
    preds = []
    for i, (_, r) in enumerate(df.iterrows()):
        post = str(r["post"])
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": INSTRUCTION.format(post=post)}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new,
                                 do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                         skip_special_tokens=True)
        spans = extract_verbatim(gen, post)
        preds.append(spans)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)} processed")

    out_df = pd.DataFrame({
        "row_id": df["row_id"].tolist(),
        "evidence": ["; ".join(s) for s in preds],
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    tot = sum(len(s) for s in preds)
    empty = sum(1 for s in preds if not s)
    print(f"\nwrote {args.out}: {tot} spans, {empty}/{len(df)} posts with none, "
          f"mean {tot/len(df):.2f}/post")

    if args.score:
        gold = [parse_evidence(c) for c in df["evidence"]]
        f1 = official_f1(gold, preds)
        print(f"\n  LLM evidence official phrase-F1 (val): {f1:.4f}")
        print(f"  compare: BIO tagger evidence_v2 official-macro = 0.7787")


if __name__ == "__main__":
    main()
