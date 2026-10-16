"""
External-data factor labeling — attack the r=0.806 scarcity wall.

The winning 2025 approach (arXiv 2510.20085) added ~3700 externally-sourced,
LLM-labeled examples and gained +44% relative Macro-F1. Our per-label
analysis shows factor F1 is almost entirely explained by example count
(Pearson r=0.806): the rare factors (sexual orientation n=8, exposure n=12,
substance use n=29, meaning-in-life n=35) simply lack training signal. No
architecture trick fixes 8 examples. More labeled examples does.

This script:
  1. Loads freely-available unlabeled mental-health Reddit posts (HF).
  2. Uses Qwen2.5-14B to label each post against our EXACT 24-factor
     clinical definitions, returning a strict-JSON multi-label verdict
     plus a per-factor confidence.
  3. Keeps ONLY high-confidence positives for RARE factors (configurable
     target list), so we inject clean signal where we're starved and avoid
     flooding common factors (which are already well-learned) or adding
     label noise.
  4. Writes an augmentation CSV in our training schema (row_id, post,
     factors as python-list-string, risk_level placeholder, evidence empty)
     for direct concatenation into train_factors_cv.py via --aug.

RULES: external data is allowed if declared in the report (confirmed from
official FAQ). This uses only publicly available unlabeled Reddit text +
our own LLM labeling — no DUA-restricted corpora.

Design guards learned from this campaign:
  * confidence gate (default 0.80) — noisy labels hurt more than they help,
    as 6x augmentation taught us (over-concentration degraded transfer)
  * rare-factor focus — only keep examples that add a rare positive; skip
    posts whose only positives are already-common factors
  * cap per factor — avoid over-concentrating on any single label's style
  * dedup against existing train posts (no leakage / near-duplicates)

Usage (login node for the HF download step, GPU node for Qwen inference):
  # step 1 (login node, internet): cache the raw posts locally
  PYTHONNOUSERSITE=1 python src/label_external.py --stage fetch \
      --hf-dataset solomonk/reddit_mental_health_posts \
      --text-col text --max-posts 4000 \
      --raw-out artifacts/external_raw.csv

  # step 2 (GPU node, offline): label with Qwen
  python src/label_external.py --stage label \
      --raw artifacts/external_raw.csv \
      --qwen Qwen/Qwen2.5-14B-Instruct \
      --conf-gate 0.80 --per-factor-cap 150 \
      --rare-factors "sexual orientation related issues,exposure to others' suicide,substance use,meaning in life,poor school performance,sense of responsibility,physical health/characteristic,cognitive deficits,low socio-economic status,traumatic experience" \
      --aug-out artifacts/external_factor_aug.csv
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from scoring import FACTOR_TAXONOMY

# Clinical definitions for each factor — given to the LLM so it labels
# against MEANING, not just surface words. These mirror the competition's
# taxonomy descriptions; refine against the official annotation guide before
# the real run.
FACTOR_DEFINITIONS = {
    # EXACT definitions from the organizers' paper (Li et al. 2025, arXiv
    # 2507.10008, Table III) — the criteria the human annotators used.
    "mental health issues": "Existing mental disorders (e.g., depression, personality disorders).",
    "physical health/characteristic": "Physical health issues (e.g., COVID-19, obesity/underweight).",
    "substance use": "The uncontrollable use of drugs/alcohol/tobacco.",
    "hopelessness": "Feelings of hopelessness, or feeling trapped/stuck.",
    "emotion dysregulation": "Emotion regulation difficulties (e.g., uncontrolled anxiety, anger).",
    "low self-esteem": "Self-negative feelings (e.g., worthlessness, being a burden, self-hate).",
    "poor school performance": "Low school performance (e.g., failing tests, bad grades).",
    "low socio-economic status": "Unemployment status, poverty, and homeless, etc.",
    "interpersonal violence": "Violence or assault, occurring outside home settings.",
    "prior self-harm or suicidal thought/attempt": "Past, resolved, previous self-harm and suicidal thought/plan/attempt.",
    "poor social support": "Lack friends, loneliness, being isolated/rejected/neglected/abandoned, etc.",
    "interpersonal difficulty": "Having difficulty making new friends, socialization, etc.",
    "dysfunctional family": "Family-related issues that leads to negative impacts on the individual.",
    "exposure to others' suicide": "Mentioning or describing others' suicidal thoughts, attempt or death.",
    "stressful life event": "Events that pose challenges but do not lead to traumatic responses.",
    "traumatic experience": "Event(s) that is overwhelming an individual's ability to cope.",
    "cognitive deficits": "Having difficulty in cognitive abilities.",
    "suicide means (with access)": "Description of potential means or access to that means.",
    "sexual orientation related issues": "Having gender/sexual disorder, same-sex relationship, etc.",
    "social support": "PROTECTIVE: Support from family, partners, friends, healthcare professionals.",
    "coping strategy": "PROTECTIVE: Activities individuals engage in to deal with stressful situations.",
    "psychological capital": "PROTECTIVE: Individual's positive psychological state of development.",
    "sense of responsibility": "PROTECTIVE: Awareness of responsibility to one's own health/survival and to others.",
    "meaning in life": "PROTECTIVE: Involves cognitive, motivational and affective components; a sense of purpose and reasons for living.",
}


def build_prompt(post: str) -> str:
    lines = ["You are a clinical annotator for suicide-risk research. "
             "Given a Reddit post, decide which of the following factors are "
             "PRESENT in the post. Judge by meaning, not just keywords.\n"]
    for i, f in enumerate(FACTOR_TAXONOMY):
        lines.append(f"{i+1}. {f}: {FACTOR_DEFINITIONS[f]}")
    lines.append(
        "\nReturn ONLY a JSON object mapping each PRESENT factor's exact name "
        "to a confidence 0.0-1.0. Omit absent factors. No prose, no markdown. "
        "Example: {\"hopelessness\": 0.9, \"substance use\": 0.7}\n")
    lines.append(f"POST:\n{post}\n\nJSON:")
    return "\n".join(lines)


def stage_fetch(args):
    _os.environ.pop("HF_HUB_OFFLINE", None)  # need internet here
    from datasets import load_dataset
    ds = load_dataset(args.hf_dataset, split="train")
    if args.text_col not in ds.column_names:
        raise SystemExit(f"--text-col '{args.text_col}' not in {ds.column_names}")
    # combine title + body if both present (richer context for labeling)
    def get_text(row_i):
        parts = []
        if args.title_col and args.title_col in ds.column_names:
            t = ds[row_i][args.title_col]
            if t: parts.append(str(t))
        b = ds[row_i][args.text_col]
        if b: parts.append(str(b))
        return "\n".join(parts)

    # cue terms for rare/crisis factors — pre-filter so Qwen inference is
    # spent on posts LIKELY to contain the factors we're starved on, not
    # random ADHD/OCD posts. Big speed + yield win.
    CUES = [
        "suicid", "kill myself", "end it", "end my life", "want to die",
        "self harm", "self-harm", "cut myself", "cutting", "overdose",
        "pills", "hang myself", "gun", "jump", "no reason to live",
        "better off dead", "attempt", "hopeless", "worthless",
        "can't go on", "give up", "meaningless", "no purpose",
    ]
    prefilter = not args.no_prefilter
    posts, seen = [], set()
    for i in range(len(ds)):
        txt = get_text(i)
        if len(txt.split()) < 15:
            continue
        low = txt.lower()
        if prefilter and not any(c in low for c in CUES):
            continue
        key = low[:120]
        if key in seen:
            continue
        seen.add(key)
        posts.append(txt)
        if len(posts) >= args.max_posts:
            break
    pd.DataFrame({"ext_id": [f"EXT{i:06d}" for i in range(len(posts))],
                  "post": posts}).to_csv(args.raw_out, index=False)
    print(f"wrote {args.raw_out}: {len(posts)} posts "
          f"(prefilter={'ON' if prefilter else 'OFF'}, >=15 words, deduped, "
          f"cap {args.max_posts})")


def parse_json_verdict(text: str) -> dict:
    text = text.strip()
    # strip markdown fences / prose around the JSON
    if "```" in text:
        text = text.split("```")[1].replace("json", "", 1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return {}
    try:
        d = json.loads(text[start:end + 1])
        return {k: float(v) for k, v in d.items() if k in FACTOR_TAXONOMY}
    except Exception:
        return {}


def stage_label(args):
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")  # models are local on GPU node
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    raw = pd.read_csv(args.raw)
    existing = set()
    if args.dedup_against and os.path.exists(args.dedup_against):
        ex = pd.read_csv(args.dedup_against)
        existing = {str(p).strip().lower()[:120] for p in ex["post"]}

    rare = set(s.strip() for s in args.rare_factors.split(",")) if args.rare_factors else set(FACTOR_TAXONOMY)

    tok = AutoTokenizer.from_pretrained(args.qwen)
    _dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.qwen, torch_dtype=torch.bfloat16)
    model.to(_dev)
    model.eval()

    kept_rows = []
    per_factor_count = {f: 0 for f in FACTOR_TAXONOMY}
    for i, row in raw.iterrows():
        post = str(row["post"])
        if post.strip().lower()[:120] in existing:
            continue
        prompt = build_prompt(post)
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                                 temperature=None, top_p=None,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        verdict = parse_json_verdict(gen)

        # keep only high-confidence factors
        pos = [f for f, c in verdict.items() if c >= args.conf_gate]
        # must contribute at least one RARE factor under its cap
        rare_pos = [f for f in pos if f in rare and per_factor_count[f] < args.per_factor_cap]
        if not rare_pos:
            continue
        # keep all confident positives (rare + any common that co-occur), but
        # the reason we keep the row is the rare contribution
        for f in pos:
            per_factor_count[f] += 1
        kept_rows.append({
            "row_id": row["ext_id"],
            "post": post,
            "risk_level": "Indicator",   # placeholder; factor trainer ignores risk
            "evidence": "",
            "factors": str(list(pos)),
        })
        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(raw)}, kept {len(kept_rows)}")

    pd.DataFrame(kept_rows).to_csv(args.aug_out, index=False)
    print(f"\nwrote {args.aug_out}: {len(kept_rows)} labeled external posts")
    print("per-factor kept counts (rare targets):")
    for f in sorted(rare, key=lambda x: per_factor_count[x], reverse=True):
        print(f"  {per_factor_count[f]:4d}  {f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fetch", "label"])
    # fetch
    ap.add_argument("--hf-dataset")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--title-col", default=None,
                    help="optional title column to prepend to body")
    ap.add_argument("--no-prefilter", action="store_true",
                    help="disable crisis-cue prefilter (label ALL posts)")
    ap.add_argument("--max-posts", type=int, default=4000)
    ap.add_argument("--raw-out", default="artifacts/external_raw.csv")
    # label
    ap.add_argument("--raw")
    ap.add_argument("--qwen", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--conf-gate", type=float, default=0.80)
    ap.add_argument("--per-factor-cap", type=int, default=150)
    ap.add_argument("--rare-factors", default="")
    ap.add_argument("--dedup-against", default="artifacts/train_split.csv")
    ap.add_argument("--aug-out", default="artifacts/external_factor_aug.csv")
    args = ap.parse_args()

    if args.stage == "fetch":
        stage_fetch(args)
    else:
        stage_label(args)


if __name__ == "__main__":
    main()
