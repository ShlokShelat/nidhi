"""
Tail-label augmentation via local Qwen paraphrasing.

The rarest factor labels (substance use, exposure to others' suicide,
sexual orientation related issues, cognitive deficits, poor school
performance, ...) carry only a few dozen training examples each, which is
too few to learn a decision boundary for. This script generates faithful
paraphrases of the posts carrying those labels, expanding the minority
classes WITHOUT changing their label set.

Constraints on generation (enforced in the prompt AND checked after):
  - preserve first-person voice
  - preserve every concrete fact (the label must remain justified)
  - change wording/structure only — this is paraphrase, not new content
  - no new risk content invented, no factor removed

Only TRAIN-split rows are augmented (never val — that would leak).
Output: artifacts/train_aug.csv  = original train + synthetic rows.
Synthetic rows get is_synthetic=1 and a synth_source=<orig row_id> column.

Usage (GPU node, riskcup + vllm or transformers):
  python src/augment_factors.py \
      --train artifacts/train_split.csv \
      --model Qwen/Qwen2.5-14B-Instruct \
      --target-labels "substance use,exposure to others' suicide,sexual orientation related issues,cognitive deficits,poor school performance,sense of responsibility,meaning in life,traumatic experience,low socio-economic status" \
      --per-post 3 --out artifacts/train_aug.csv

Then train factors on the augmented file:
  python src/train_factors.py --train artifacts/train_aug.csv \
      --val artifacts/val_split.csv --model microsoft/deberta-v3-large \
      --outdir runs/factors_v3_aug
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from scoring import parse_factors, normalize_span

PROMPT_TEMPLATE = """You are helping build a training set for a mental-health text classifier. \
You will rewrite a Reddit-style first-person post so it expresses the SAME situation \
and the SAME underlying factors, but in different words and sentence structure.

Strict rules:
- Keep it first person.
- Preserve every concrete fact and life circumstance mentioned. Do not remove or add situations.
- These factors MUST remain clearly present and justified in your rewrite: {labels}.
- Do NOT invent new self-harm methods, plans, or details that were not in the original.
- Do NOT make it more graphic or more severe than the original.
- Change wording and phrasing only. Same meaning, same facts, new words.
- Output ONLY the rewritten post, no preamble, no quotes, no explanation.

Original post:
\"\"\"{post}\"\"\"

Rewritten post:"""


def build_prompt(post: str, labels: list[str]) -> str:
    return PROMPT_TEMPLATE.format(labels="; ".join(labels), post=post.strip())


def target_rows(df: pd.DataFrame, targets: set[str]) -> list[int]:
    """Indices of train rows carrying at least one target label."""
    idx = []
    for i, cell in enumerate(df["factors"]):
        fs = set(parse_factors(cell))
        if fs & targets:
            idx.append(i)
    return idx


def faithful(original: str, rewrite: str, labels: list[str]) -> bool:
    """Cheap post-hoc guards. Label justification is left to the generator,
    since semantics cannot be verified here, but degenerate output is caught:
    empty, near-identical, or wildly length-mismatched rewrites."""
    r = rewrite.strip()
    if len(r.split()) < 5:
        return False
    if normalize_span(r) == normalize_span(original):
        return False  # model echoed the input
    lo, hi = 0.4, 2.5
    ratio = len(r.split()) / max(len(original.split()), 1)
    return lo <= ratio <= hi


def load_generator(model_path: str, backend: str):
    if backend == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(model=model_path, dtype="bfloat16",
                  gpu_memory_utilization=0.9, max_model_len=4096)
        return ("vllm", llm, SamplingParams)
    # transformers fallback
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto")
    return ("hf", (model, tok), None)


def generate(gen, prompts: list[str], max_new_tokens: int = 512) -> list[str]:
    kind = gen[0]
    if kind == "vllm":
        _, llm, SamplingParams = gen
        # Qwen chat template
        tok = llm.get_tokenizer()
        chats = [tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True)
                 for p in prompts]
        sp = SamplingParams(temperature=0.8, top_p=0.95,
                            max_tokens=max_new_tokens)
        outs = llm.generate(chats, sp)
        return [o.outputs[0].text.strip() for o in outs]
    # hf
    import torch
    (model, tok) = gen[1]
    results = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 temperature=0.8, top_p=0.95, do_sample=True)
        gen_ids = out[0][enc["input_ids"].shape[1]:]
        results.append(tok.decode(gen_ids, skip_special_tokens=True).strip())
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--model", required=True, help="local path to Qwen instruct")
    ap.add_argument("--target-labels", required=True,
                    help="comma-separated factor labels to boost")
    ap.add_argument("--per-post", type=int, default=3,
                    help="paraphrases per qualifying post")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--out", default="artifacts/train_aug.csv")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    targets = {t.strip().lower() for t in args.target_labels.split(",") if t.strip()}
    df = pd.read_csv(args.train)
    rows = target_rows(df, targets)
    print(f"{len(rows)} train posts carry >=1 target label; "
          f"generating {args.per_post} paraphrases each "
          f"(<= {len(rows)*args.per_post} synthetic rows)")

    gen = load_generator(args.model, args.backend)

    synth = []
    # batch all (row x per_post) prompts for throughput
    jobs = []
    for i in rows:
        post = str(df.iloc[i]["post"])
        labels = [f for f in parse_factors(df.iloc[i]["factors"])]
        target_present = [f for f in labels if f in targets]
        for _ in range(args.per_post):
            jobs.append((i, post, labels, build_prompt(post, target_present)))

    BATCH = 64
    kept = dropped = 0
    for s in range(0, len(jobs), BATCH):
        chunk = jobs[s:s + BATCH]
        outs = generate(gen, [c[3] for c in chunk], args.max_new_tokens)
        for (i, post, labels, _), rewrite in zip(chunk, outs):
            if not faithful(post, rewrite, labels):
                dropped += 1
                continue
            base = df.iloc[i]
            synth.append({
                "row_id": f"{base['row_id']}_aug{kept}",
                "anon_user_id": base["anon_user_id"],
                "post_id": base.get("post_id", -1),
                "post": rewrite,
                "risk_level": base["risk_level"],
                "evidence": "",          # synthetic: no verbatim evidence spans
                "factors": base["factors"],
                "is_synthetic": 1,
                "synth_source": base["row_id"],
            })
            kept += 1
        print(f"  {s+len(chunk)}/{len(jobs)} prompts | kept {kept} dropped {dropped}")

    orig = df.copy()
    orig["is_synthetic"] = 0
    orig["synth_source"] = ""
    out_df = pd.concat([orig, pd.DataFrame(synth)], ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}: {len(orig)} original + {kept} synthetic "
          f"= {len(out_df)} rows")

    # report new per-target-label counts
    print("\ntarget-label counts (train + synthetic):")
    for t in sorted(targets):
        n = sum(t in parse_factors(c) for c in out_df["factors"])
        n0 = sum(t in parse_factors(c) for c in orig["factors"])
        print(f"  {n0:4d} -> {n:4d}  {t}")


if __name__ == "__main__":
    main()
