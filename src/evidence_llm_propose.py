"""
evidence_llm_propose.py

Does an LLM, told the risk level, propose gold spans the taggers NEVER propose?

WHY THIS IS NOT THE LANE THAT CLOSED. Attempts at evidence SELECTION have
failed, including with frontier LLMs -- DeepSeek graded candidates at AP
0.7007 against 0.7595 for simply counting how many taggers agreed. A strong
model ranked worse than vote-counting, because gold spans are annotator
CONVENTIONS about which exact words to quote, not "the sensible quote", and a
zero-shot model has no access to that convention.

But selection was never the binding constraint. GENERATION is:

    a PERFECT selector over the current candidate pool reaches only 0.8617
    because 24% of gold spans never appear in ANY candidate at ANY confidence

No ranking reaches a span nobody proposed. `train_evidence.py`'s own docstring
anticipated this -- "an LLM proposer can be ensembled later" -- and it was
deferred, never tested.

THE BAR IS NOT "BEATS THE EXISTING TAGGERS". An LLM proposer only has to be
DIFFERENTLY wrong. If it finds gold spans the current pool misses, and they survive the substring
check and the vote, that is headroom nothing else can reach. So the number that
decides this is POOL RECALL GAIN, not precision, not F1.

The risk level is given, because perfect levels moved a tagger from 0.7518 to
0.8287 and this proposer should get the same advantage.

EVERY SPAN IS SUBSTRING-VERIFIED against the post via train_evidence.verify_spans
-- the same gate the taggers pass. A generated span that is not literally in the
post scores zero under the official matcher, so it is dropped here rather than
counted as a proposal.

PRE-STATED BAR: of the gold spans the tagger pool MISSES, the LLM must recover
>= 20%. Below that the ensemble gain cannot exceed a small macro improvement
even if selection were perfect afterwards.

Usage:
  python src/evidence_llm_propose.py --model Qwen/Qwen2.5-72B-Instruct \
      --load-4bit
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (_span_pair_matches, normalize_label,  # noqa: E402
                     normalize_span, parse_evidence, phrase_f1)
from train_evidence import verify_spans  # noqa: E402

GUIDE = {
    "indicator": "distress with no explicit mention of suicide or self-harm",
    "ideation": "an explicit wish or thought of dying",
    "behavior": "self-harm, a named or available method, a stated time, or "
                "preparation for death",
    "attempt": "an act already carried out to end their life",
}

# Hand-written exemplars, not corpus rows. They exist to demonstrate the OUTPUT
# SHAPE -- a short verbatim quote, not a paraphrase or a summary -- and to show
# that the span carrying the level is often a fragment rather than a whole
# sentence. Writing them by hand keeps corpus text out of the shipped package.
SHOTS = [
    ("behavior",
     "another morning of the same thing and I am still counting the days "
     "until I do not have to keep going",
     '"counting the days"'),
    ("ideation",
     "I am done trying. thinking about it is the only thing that makes the "
     "week feel survivable.",
     '"thinking about it"'),
    ("attempt",
     "I need someone to talk to. last winter I swallowed a whole bottle and "
     "woke up in the emergency room and nothing has changed.",
     '"swallowed a whole bottle"'),
]


def build_prompt(post: str, level: str) -> str:
    lines = [
        "You are given a Reddit post and its assigned suicide risk level.",
        "Quote the EXACT words from the post that justify that level -- the "
        "shortest phrases a reader would point at as proof.",
        "",
        "Rules:",
        "  - copy the words EXACTLY as they appear, character for character",
        "  - quote the specific clause, not the whole sentence",
        "  - one quote per line, in double quotes",
        "  - if nothing in the post justifies the level, write NONE",
        "",
    ]
    for lv, p, a in SHOTS:
        lines += [f"Level: {lv} ({GUIDE[lv]})", f"Post: {p}", f"Quotes:\n{a}", ""]
    lines += [f"Level: {level} ({GUIDE[level]})", f"Post: {post}", "Quotes:"]
    return "\n".join(lines)


def parse_quotes(text: str) -> list[str]:
    text = text.split("Level:")[0].split("Post:")[0]
    out = re.findall(r'"([^"\n]{2,200})"', text)
    if not out:
        for ln in text.splitlines():
            ln = ln.strip(" -*\t")
            if ln and ln.upper() != "NONE" and len(ln) > 2:
                out.append(ln)
    return out[:8]


def load_pool(d: str, rows: list[str]) -> dict:
    """Every candidate any tagger proposed, at any confidence."""
    pool = {r: [] for r in rows}
    files = sorted(glob.glob(os.path.join(d, "*_val_candidates.csv")))
    if not files:
        raise SystemExit(f"no *_val_candidates.csv in {d}")
    for f in files:
        c = pd.read_csv(f, keep_default_na=False)
        for r, s in zip(c["row_id"].astype(str), c["span"]):
            if r in pool:
                pool[r].append(str(s))
    print(f"  pooled {len(files)} tagger candidate files")
    return pool


def covered(span: str, cands: list[str]) -> bool:
    return any(_span_pair_matches(c, span) for c in cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--cand-dir", default="artifacts/evidence_runs")
    ap.add_argument("--risk-from", default=None,
                    help="predicted levels; omit to use GOLD (the ceiling)")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=90)
    ap.add_argument("--max-post-words", type=int, default=250)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/evidence_llm_propose.csv")
    args = ap.parse_args()

    from data_audit import load
    va = load(args.val)
    va["row_id"] = va["row_id"].astype(str)
    va = va[[len(parse_evidence(e)) > 0 for e in va["evidence"]]].reset_index(
        drop=True)
    print(f"{len(va)} val posts WITH gold spans "
          f"(empty-gold posts are a suppression question, not a span one)")

    if args.n and args.n < len(va):
        rng = np.random.default_rng(args.seed)
        va = va.iloc[sorted(rng.choice(len(va), args.n, replace=False))]
        va = va.reset_index(drop=True)
    rows = list(va["row_id"])
    gold = [parse_evidence(e) for e in va["evidence"]]

    if args.risk_from:
        rk = pd.read_csv(args.risk_from, keep_default_na=False)
        col = ("pred_risk_level" if "pred_risk_level" in rk.columns
               else "risk_level")
        m = {str(a): normalize_label(b) for a, b in zip(rk["row_id"], rk[col])}
        absent = [r for r in rows if r not in m]
        if absent:
            raise SystemExit(f"--risk-from lacks {len(absent)} rows")
        levels = [m[r] for r in rows]
        print(f"  levels from {args.risk_from} (predicted)")
    else:
        levels = [normalize_label(x) for x in va["risk_level"]]
        print("  levels are GOLD -- this is the ceiling, not a shippable number")

    pool = load_pool(args.cand_dir, rows)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = {"device_map": "auto"}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
        print("[quant] NF4 4-bit")
    else:
        kw["torch_dtype"] = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    mdl.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    posts = [" ".join(str(p).split()[:args.max_post_words])
             for p in va["post"]]
    prompts = [build_prompt(p, lv) for p, lv in zip(posts, levels)]

    raw_spans, kept_spans, n_raw, n_verified = [], [], 0, 0
    B = max(1, args.batch)
    for s in range(0, len(prompts), B):
        enc = tok(prompts[s:s + B], return_tensors="pt",
                  padding=True).to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(**enc, max_new_tokens=args.max_new,
                               do_sample=False, pad_token_id=tok.pad_token_id)
        cut = enc["input_ids"].shape[1]
        for j, row in enumerate(out):
            gen = tok.decode(row[cut:], skip_special_tokens=True)
            q = parse_quotes(gen)
            n_raw += len(q)
            ok, _ = verify_spans(q, posts[s + j])
            n_verified += len(ok)
            raw_spans.append(q)
            kept_spans.append(ok)
        if (s // B) % 5 == 0:
            print(f"  {min(s + B, len(prompts))}/{len(prompts)}")

    print(f"\n{n_raw} quotes generated, {n_verified} survive the substring "
          f"check ({n_verified / max(n_raw, 1):.3f})")
    print("  (a quote that is not literally in the post scores zero under the")
    print("   official matcher, so the rest are dropped, not counted)")

    res = phrase_f1(gold, kept_spans, average="macro")
    resm = phrase_f1(gold, kept_spans)
    print(f"\nLLM alone   macro {res.f1:.4f}   micro {resm.f1:.4f} "
          f"(P={resm.precision:.3f} R={resm.recall:.3f})")

    # THE NUMBER THAT DECIDES IT
    miss_tot = miss_found = 0
    have_tot = 0
    for g, k, r in zip(gold, kept_spans, rows):
        for gs in g:
            if covered(gs, pool[r]):
                have_tot += 1
            else:
                miss_tot += 1
                if covered(gs, k):
                    miss_found += 1
    n_gold = have_tot + miss_tot
    print("\n" + "=" * 66)
    print("POOL RECALL -- can the LLM reach what the taggers never propose?")
    print("=" * 66)
    print(f"  gold spans total                     {n_gold}")
    print(f"  already in the tagger pool           {have_tot} "
          f"({have_tot / max(n_gold, 1):.3f})")
    print(f"  MISSED by every tagger               {miss_tot} "
          f"({miss_tot / max(n_gold, 1):.3f})")
    rate = miss_found / max(miss_tot, 1)
    print(f"  of those, the LLM proposes           {miss_found} ({rate:.3f})")
    new_pool = (have_tot + miss_found) / max(n_gold, 1)
    print(f"\n  pool recall {have_tot / max(n_gold, 1):.3f} -> "
          f"{new_pool:.3f} if the LLM joins the pool")
    print(f"\nBAR: >= 0.20 of missed spans recovered. "
          f"{'PASSES' if rate >= 0.20 else 'FAILS'}")
    if rate >= 0.20:
        print("The LLM is a DIFFERENTLY-WRONG proposer. Add it to the pool and")
        print("let cluster voting decide -- do NOT let it select.")
    else:
        print("It proposes the same spans the taggers already do. Generation")
        print("is not the fix either, and evidence has no lever left.")
    print("=" * 66)

    pd.DataFrame({
        "row_id": rows, "level": levels,
        "gold": ["; ".join(g) for g in gold],
        "llm_raw": ["; ".join(r) for r in raw_spans],
        "llm_verified": ["; ".join(k) for k in kept_spans],
    }).to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
