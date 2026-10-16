"""
gen_rare_factors.py

Generate synthetic training posts for the factor labels with the fewest real
examples.

WHY GENERATION IS TARGETED AT THE RARE LABELS. Subtask 2 is macro-F1 over 24
factors, so every label contributes 1/24 of the score no matter how often it
occurs. Thirteen labels have between 8 and 118 training examples while nine
have 200-745. A cross-encoder cannot learn a concept from eight examples, and
that label costs exactly as much as `hopelessness` with 745. Adding data to an
already-saturated label teaches the model nothing it has not already seen, so
--target-count tops up only the labels below it.

THE CRITICAL PROMPT CONSTRAINT. Factor labels are not topic mentions: posts
mentioning alcohol carry `substance use` only 26% of the time, and posts
mentioning pain carry `physical health` only 17% of the time. The annotators
mark a factor when it CONTRIBUTES TO THAT PERSON'S RISK, not when the subject
comes up. The prompt therefore demands the factor be causally load-bearing and
explicitly forbids passing mentions; a generator that ignored this would
manufacture exactly the false positives that make keyword matching useless on
this task.

RISKS
  - synthetic posts are not real posts; distribution shift can hurt
  - a post labelled with ONE factor teaches the model that the other 23 are
    absent, which is often false. Real posts carry 2-4 factors. This is why
    the output is fed via --external (training only) and why thresholds and
    every evaluation stay on REAL out-of-fold data.
  - generation runs on the local Qwen-14B; no API key leaves the cluster.

Usage (GPU):
  python src/gen_rare_factors.py \
      --model Qwen/Qwen2.5-14B-Instruct \
      --target-count 150 \
      --out artifacts/syn_rare_factors.csv
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import collections
import os
import re
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from scoring import FACTOR_TAXONOMY, parse_factors
from train_factors_nli import FACTOR_DEFINITIONS

# Rotated through so 140 posts for one factor are not 140 paraphrases of each
# other. Deliberately mundane and varied -- real r/SuicideWatch posts are
# specific about circumstances.
# FIVE OF THE 24 LABELS ARE PROTECTIVE, NOT RISK FACTORS. Their official
# definitions make this plain:
#     meaning in life         "a sense of purpose, goals, or reasons for living"
#     social support          (support the person actually receives)
#     coping strategy         (ways they manage)
#     psychological capital   (resilience, self-efficacy)
#     sense of responsibility (obligations that hold them here)
#
# Prompting for these as "a contributing cause of how bad they feel" would
# generate someone whose sense of purpose is making them suicidal -- incoherent
# text, mislabelled. Three of the thirteen starved labels are protective
# (meaning in life, sense of responsibility, social support), so the framing
# has to be inverted for them: the post still describes someone in crisis, but
# this element is present as something pulling the other way.
PROTECTIVE = {
    "social support",
    "coping strategy",
    "psychological capital",
    "sense of responsibility",
    "meaning in life",
}

SCENARIOS = [
    "a university student in their early twenties",
    "someone in their thirties who has been unemployed for months",
    "a teenager still living with their parents",
    "a parent of young children",
    "someone recently divorced",
    "a night-shift worker with no friends nearby",
    "someone who moved to a new city and knows nobody",
    "a person in their forties caring for a sick relative",
    "a recent graduate who cannot find work",
    "someone who has been in therapy for years without progress",
    "a person living in a shared flat with strangers",
    "someone whose long-term relationship just ended",
]
TONES = [
    "numb and matter-of-fact", "exhausted and rambling",
    "angry and bitter", "quiet and resigned",
    "panicked, jumping between thoughts", "flat, listing things",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--model", required=True)
    ap.add_argument("--target-count", type=int, default=150,
                    help="bring each starved factor up to roughly this many "
                         "training examples")
    ap.add_argument("--max-per-factor", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--out", default="artifacts/syn_rare_factors.csv")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("no GPU visible -- submit via sbatch")

    real = pd.read_csv(args.labeled)
    cnt = collections.Counter(
        f for c in real["factors"]
        for f in set(parse_factors(c)) & set(FACTOR_TAXONOMY))

    need = {}
    for f in FACTOR_TAXONOMY:
        deficit = args.target_count - cnt.get(f, 0)
        if deficit > 0:
            need[f] = min(deficit, args.max_per_factor)
    if not need:
        raise SystemExit("no factor below --target-count; nothing to generate")

    print(f"{'factor':<44}{'real':>6}{'to gen':>8}")
    for f, n in sorted(need.items(), key=lambda x: cnt.get(x[0], 0)):
        print(f"{f:<44}{cnt.get(f, 0):>6}{n:>8}")
    print(f"\ntotal to generate: {sum(need.values())}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=None).to(device)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    def prompt_for(factor, i):
        d = FACTOR_DEFINITIONS[factor]
        who = SCENARIOS[i % len(SCENARIOS)]
        tone = TONES[(i // len(SCENARIOS)) % len(TONES)]
        if factor in PROTECTIVE:
            body = (
                f"The post must clearly show this PROTECTIVE element:\n"
                f"  {factor}: {d}\n\n"
                f"Rules:\n"
                f"- The person is struggling and in distress, but this element "
                f"is genuinely PRESENT in their life and pulling the other "
                f"way. It should be doing real work in the post, not mentioned "
                f"in passing.\n"
                f"- Do NOT resolve the distress. They are still in a bad place; "
                f"this is one thing that is holding.\n"
            )
        else:
            body = (
                f"The post must show this RISK FACTOR:\n"
                f"  {factor}: {d}\n\n"
                f"Rules:\n"
                f"- The factor must be a REAL CONTRIBUTING CAUSE of how the "
                f"person feels, woven into their situation. A passing mention "
                f"of the topic does not count.\n"
            )
        return tok.apply_chat_template([{
            "role": "user",
            "content": (
                f"Write a single anonymous post for a suicide-support forum, "
                f"as if written by {who}. Voice: {tone}.\n\n"
                + body +
                f"- Do NOT name the factor or use clinical language.\n"
                f"- First person, lowercase-ish, informal, 50-110 words.\n"
                f"- Concrete specifics, not generalities.\n"
                f"- Output ONLY the post text, no title, no quotes, no "
                f"commentary."
            )}], tokenize=False, add_generation_prompt=True)

    rows, made = [], collections.Counter()
    for factor, n in need.items():
        print(f"\n=== {factor}  (target {n}) ===")
        i = 0
        while made[factor] < n:
            k = min(args.batch, n - made[factor])
            prompts = [prompt_for(factor, i + j) for j in range(k)]
            i += k
            enc = tok(prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=args.max_new, do_sample=True,
                    temperature=args.temperature, top_p=0.95,
                    pad_token_id=tok.pad_token_id)
            for j in range(k):
                txt = tok.decode(out[j][enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip()
                txt = re.sub(r'^["\'`]+|["\'`]+$', "", txt).strip()
                txt = re.sub(r"\s+", " ", txt)
                w = len(txt.split())
                if w < 25 or w > 200:      # reject degenerate generations
                    continue
                made[factor] += 1
                rows.append({
                    "row_id": f"SYN{len(rows):05d}",
                    "post": txt,
                    "factors": str([factor]),
                    "target_factor": factor,
                })
            print(f"   {made[factor]}/{n}")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {len(out)} posts to {args.out}")
    print(f"mean words: {out['post'].str.split().str.len().mean():.1f}"
          f"   (real corpus 68.8)")
    print("\nsample:")
    for t in out["post"].head(3):
        print(f"  - {t[:150]}")
    print("\nNEXT: retrain with --external, 5-fold, and fit thresholds on REAL")
    print("out-of-fold posts only. Synthetic data belongs in TRAINING; the")
    print("moment it touches thresholds or evaluation the numbers stop meaning")
    print("anything.")


if __name__ == "__main__":
    main()
