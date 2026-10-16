"""
evidence_llm_extract.py

A GENERATIVE EVIDENCE EXTRACTOR. ALL EIGHT SHIPPED VOTERS ARE BIO TAGGERS.

The evidence vote combines 8 checkpoints that are the SAME architecture -- a
token classifier emitting B/I/O over the post. Diversity is where a vote pays,
and there is none in the reader. Meanwhile:

    oracle ceiling of the candidate pool   0.89-0.93 micro
    what is actually scored                ~0.776
    posts where NO candidate matches gold  92-153 of 1020 non-empty

That gap is span QUALITY, and a different kind of reader is the untried lever.
Past LLM evidence work here was RERANK and SCORE -- both selection over the
tagger's existing pool. Generation was never tried.

WHY EXTRACTION IS THE SAFE LLM TASK. `verify_spans` (train_evidence.py:114-127)
requires the output to be a VERBATIM substring of the post, located by exact
match, then case-insensitive, whitespace-flexible, punctuation-tolerant and
fuzzy fallbacks. Anything hallucinated fails grounding and is dropped
mechanically. The model is not being trusted to judge -- only to point.

CONFIDENCE COMES FROM SELF-CONSISTENCY. An LLM has no natural per-span score,
so each post is sampled K times at temperature and a span's `conf` is the
fraction of samples that produced it. That is exactly the quantity the
existing vote machinery consumes, so the output drops into the pipeline as
another voter with no special-casing.

RISK-CONDITIONED, because the champion's evidence column comes from a
risk-conditioned tagger and gold evidence is the justification FOR a specific
level -- a countdown for behavior, a completed act for attempt. Conditioning
uses PREDICTED risk (gold would leak).

Writes `artifacts/evidence_oof/{tag}_oof_candidates.csv` (or `_test_`) with
row_id, span, conf, min_conf -- the same schema every other tagger produces.

Usage:
  python src/evidence_llm_extract.py --split oof --limit 200      # pilot
  python src/evidence_llm_extract.py --split oof
  python src/evidence_llm_extract.py --split test
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_label, normalize_span  # noqa: E402
from train_evidence import verify_spans  # noqa: E402

PROMPT = """You are annotating a Reddit post from r/SuicideWatch for a \
clinical research study.

The post has been assessed at suicide risk level: {risk}.

Quote the exact phrase or short clause FROM THE POST that is the evidence for \
that risk level.

Rules:
- Copy the words EXACTLY as they appear in the post. Do not paraphrase, fix \
spelling, or add words.
- Keep each quote SHORT: usually 2 to 8 words, a single clause.
- If several separate clauses are evidence, put each on its own line.
- Output only the quotes, nothing else.
- If the post contains no explicit evidence of suicide risk, output exactly: \
NONE

Post:
\"\"\"{post}\"\"\"

Evidence:"""


def parse_out(txt, post):
    """Lines -> grounded verbatim spans. Everything ungrounded is dropped."""
    out = []
    for ln in str(txt).splitlines():
        s = ln.strip().strip("-*•").strip().strip('"').strip("'").strip()
        if not s or s.upper().startswith("NONE"):
            continue
        s = re.sub(r"^\d+[.)]\s*", "", s)
        if len(s.split()) > 20:
            continue
        acc, _ = verify_spans([s], post, max_tokens=12)
        if acc:
            out.append(acc[0])
    return out


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--model",
                    default="Qwen/Qwen2.5-14B-Instruct")
    a_.add_argument("--split", default="oof", choices=["oof", "test"])
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--test", default="data/test.csv")
    a_.add_argument("--risks-oof", default="artifacts/nligaur/oof_base.csv")
    a_.add_argument("--risks-test", default="submissions/NIDHI_factfit.csv")
    a_.add_argument("--tag", default="llmex")
    a_.add_argument("--outdir", default="artifacts/evidence_oof")
    a_.add_argument("--samples", type=int, default=5,
                    help="self-consistency samples; conf = fraction agreeing")
    a_.add_argument("--temp", type=float, default=0.7)
    a_.add_argument("--max-new", type=int, default=48)
    a_.add_argument("--batch", type=int, default=16)
    a_.add_argument("--limit", type=int, default=0)
    a_.add_argument("--load-4bit", action="store_true",
                    help="required for a 70B+ on one card")
    args = a_.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.split == "oof":
        d = pd.read_csv(args.labeled, keep_default_na=False)
        rp = args.risks_oof
    else:
        d = pd.read_csv(args.test, keep_default_na=False)
        rp = args.risks_test
    d["row_id"] = d["row_id"].astype(str)
    rs = pd.read_csv(rp, keep_default_na=False)
    rs["row_id"] = rs["row_id"].astype(str)
    rcol = next(c for c in ("pred_risk_level", "risk_level")
                if c in rs.columns)
    rmap = dict(zip(rs["row_id"], rs[rcol]))
    miss = [r for r in d["row_id"] if r not in rmap]
    if miss:
        raise SystemExit(f"{len(miss)} posts have no risk label to condition "
                         f"on (source {rp})")
    if args.limit:
        d = d.head(args.limit)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kw = {"torch_dtype": torch.bfloat16}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}
    m = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    if not args.load_4bit:
        m = m.cuda()
    m.eval()
    print(f"{args.model}   {len(d)} posts x {args.samples} samples", flush=True)

    prompts, rids, posts = [], [], []
    for r, p in zip(d["row_id"], d["post"].astype(str)):
        msg = [{"role": "user",
                "content": PROMPT.format(risk=normalize_label(rmap[r]),
                                         post=p[:4000])}]
        prompts.append(tok.apply_chat_template(msg, tokenize=False,
                                               add_generation_prompt=True))
        rids.append(r)
        posts.append(p)

    counts = {r: {} for r in rids}
    for s in range(args.samples):
        torch.manual_seed(1000 + s)
        for i in range(0, len(prompts), args.batch):
            chunk = prompts[i:i + args.batch]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(m.device)
            with torch.no_grad():
                g = m.generate(**enc, max_new_tokens=args.max_new,
                               do_sample=True, temperature=args.temp,
                               top_p=0.9, pad_token_id=tok.pad_token_id)
            for b in range(len(chunk)):
                txt = tok.decode(g[b][enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
                k = i + b
                for sp in parse_out(txt, posts[k]):
                    key = normalize_span(sp)
                    e = counts[rids[k]].setdefault(key, [0, sp])
                    e[0] += 1
            if i % (args.batch * 20) == 0:
                print(f"  sample {s+1}/{args.samples}  {i}/{len(prompts)}",
                      flush=True)
        print(f"  sample {s+1}/{args.samples} done", flush=True)

    rows = []
    for r in rids:
        for _, (c, sp) in counts[r].items():
            rows.append((r, sp, c / args.samples, 0.0))
    os.makedirs(args.outdir, exist_ok=True)
    dst = (f"{args.outdir}/{args.tag}_oof_candidates.csv"
           if args.split == "oof"
           else f"{args.outdir}/{args.tag}_test_candidates.csv")
    pd.DataFrame(rows, columns=["row_id", "span", "conf",
                                "min_conf"]).to_csv(dst, index=False)
    npost = len(set(x[0] for x in rows))
    print(f"\nwrote {dst}")
    print(f"  {len(rows)} grounded spans over {npost} of {len(rids)} posts")
    print(f"  {len(rows)/max(npost,1):.2f} spans per post that produced any")
    if rows:
        import numpy as np
        cf = np.array([x[2] for x in rows])
        print(f"  conf (self-consistency): median {np.median(cf):.2f}, "
              f"{(cf >= 0.8).mean():.0%} appear in >=80% of samples")
    print("\nNext -- judge it as a voter on the honest instrument:")
    print("  python src/evidence_arm_swap_check.py --rc {tag} "
          "--unc v2 ep12 lr3 bs16 ml256".replace("{tag}", args.tag))


if __name__ == "__main__":
    main()
