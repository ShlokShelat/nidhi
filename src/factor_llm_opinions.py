"""
factor_llm_opinions.py

Get a reasoning LLM's opinion on all 24 factors, then ROUTE per label.

WHY ROUTING RATHER THAN A GLOBAL USE OF THE LLM. Forms that use the LLM's
scores globally -- replacing the incumbent, ensembling, few-shot, fine-tuning,
count-matched ranking substitution -- are all governed by average precision,
and the incumbent out-ranks the LLM on every label tested. But AP measures
ranking across ALL posts. Routing asks a different question:

    on the posts where the LLM says factor k is PRESENT and the bag says ABSENT,
    which one is right?

That is the disagreement-cell question. A model with worse global AP can still
own specific cells, because owning a cell requires only local precision inside
that cell, not a better ordering of every post.

THE ODDS FAVOUR FACTORS. Risk offers 4 classes and so a handful of cells;
factors give 24 labels. Routing pays if ANY label has a clean cell.

WHY ONE PASS PER POST, NOT 24. Asking 24 separate yes/no questions is 48k
generations for OOF+test. One pass listing which factors apply is 2k, and
routing only needs a handful of good cells -- per-label precision matters, not
calibrated per-label scores. If a label turns out to have a strong cell, THAT
label can be rescored individually later.

EXPECT OVER-LISTING. Gold averages 2.92 factors of 24; a model asked "which
apply" will name more. That is fine for routing -- a label where the LLM is
imprecise simply will not be routed. Precision per label is reported so the
choice is visible.

EVERYTHING LOCAL. No dataset post leaves the cluster.

Usage:
  python src/factor_llm_opinions.py --model Qwen/Qwen2.5-72B-Instruct \
      --load-4bit --oof artifacts/full_train.csv --extra-csv data/test.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402

# Short, concrete glosses. The published definitions of several labels are
# academic prose, and labels defined that way are scored far worse than labels
# whose definition is already plain English, so each label gets a gloss an
# instruction model can act on directly.
GLOSS = {
    "mental health issues": "a diagnosed or described mental disorder",
    "physical health/characteristic": "illness, pain, weight, appearance",
    "substance use": "drugs, alcohol or tobacco use",
    "hopelessness": "feeling hopeless, trapped or stuck",
    "emotion dysregulation": "uncontrolled anger, anxiety or panic",
    "low self-esteem": "worthlessness, self-hate, feeling a burden",
    "poor school performance": "failing, bad grades, dropping out",
    "low socio-economic status": "unemployment, poverty, homelessness",
    "interpersonal violence": "violence or assault outside the home",
    "prior self-harm or suicidal thought/attempt": "a PAST self-harm act, "
        "thought, plan or attempt",
    "poor social support": "loneliness, isolation, rejection, no friends",
    "interpersonal difficulty": "trouble making friends or socialising",
    "dysfunctional family": "family conflict, neglect or abuse at home",
    "exposure to others' suicide": "someone ELSE's suicide, attempt or "
        "suicidal talk",
    "stressful life event": "a specific difficult event: a move, a breakup, "
        "a job loss",
    "traumatic experience": "an overwhelming event the author cannot cope with",
    "cognitive deficits": "trouble thinking, concentrating or remembering",
    "suicide means (with access)": "a named or available method of dying",
    "sexual orientation related issues": "sexuality or gender identity",
    "social support": "PRESENT help from family, friends or professionals",
    "coping strategy": "something the author does to cope or get through",
    "psychological capital": "hope, confidence, resilience or optimism",
    "sense of responsibility": "duty to their own survival or to others",
    "meaning in life": "purpose, gratitude, pride, or something worth "
        "living for",
}


def build_prompt(post: str) -> str:
    lines = ["Below is a Reddit post and a list of 24 factors.",
             "List ONLY the factors clearly present in the post.",
             "Most posts contain 2 to 4. Do not list a factor you are unsure "
             "of.", "", "Factors:"]
    for i, f in enumerate(FACTOR_TAXONOMY, 1):
        lines.append(f"{i:>2}. {f} -- {GLOSS.get(f, '')}")
    lines += ["", f"Post: {post}", "",
              "Answer with the numbers only, comma separated, e.g. 4, 11, 22",
              "Present:"]
    return "\n".join(lines)


def parse_nums(text: str, n: int) -> list[int]:
    text = text.split("Post:")[0].split("Factors:")[0]
    out = []
    for m in re.findall(r"\d+", text[:200]):
        v = int(m)
        if 1 <= v <= n and (v - 1) not in out:
            out.append(v - 1)
    return out[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--oof", default="artifacts/full_train.csv")
    ap.add_argument("--extra-csv", default="data/test.csv")
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--max-post-words", type=int, default=250)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/factor_llm_oof.csv")
    args = ap.parse_args()

    labels = list(FACTOR_TAXONOMY)
    from data_audit import load
    df = load(args.oof)
    df["row_id"] = df["row_id"].astype(str)
    if args.n:
        df = df.head(args.n)
    print(f"{len(df)} posts, {len(labels)} labels")

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

    def run(frame, tag):
        posts = [" ".join(str(p).split()[:args.max_post_words])
                 for p in frame["post"]]
        prompts = [build_prompt(p) for p in posts]
        P = np.zeros((len(prompts), len(labels)), int)
        raw = []
        B = max(1, args.batch)
        for s in range(0, len(prompts), B):
            enc = tok(prompts[s:s + B], return_tensors="pt",
                      padding=True).to(mdl.device)
            with torch.no_grad():
                out = mdl.generate(**enc, max_new_tokens=args.max_new,
                                   do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            cut = enc["input_ids"].shape[1]
            for bi, row in enumerate(out):
                g = tok.decode(row[cut:], skip_special_tokens=True)
                raw.append(g.replace("\n", " ")[:120])
                for j in parse_nums(g, len(labels)):
                    P[s + bi, j] = 1
            if (s // B) % 10 == 0:
                print(f"  {tag} {min(s + B, len(prompts))}/{len(prompts)}")
        o = pd.DataFrame({"row_id": list(frame["row_id"]), "raw": raw})
        for j, f in enumerate(labels):
            o[f"llm_{j}"] = P[:, j]
        print(f"  {tag}: mean {P.sum(1).mean():.2f} factors/post named "
              f"(gold averages 2.92); {int((P.sum(1) == 0).sum())} posts empty")
        return o, P

    oo, PO = run(df, "oof")
    oo.to_csv(args.out, index=False)
    print(f"wrote {args.out}")

    # per-label precision/recall against gold, and the ROUTING cell
    if "factors" in df.columns:
        Y = np.array([[f in set(parse_factors(x)) for f in labels]
                      for x in df["factors"]], int)
        print(f"\n{'label':<42}{'gold':>6}{'llm':>6}{'P':>7}{'R':>7}")
        print("-" * 68)
        for j, f in enumerate(labels):
            g, q = Y[:, j], PO[:, j]
            tp = int((g & q).sum())
            pr = tp / max(int(q.sum()), 1)
            rc = tp / max(int(g.sum()), 1)
            print(f"{f[:41]:<42}{int(g.sum()):>6}{int(q.sum()):>6}"
                  f"{pr:>7.3f}{rc:>7.3f}")
        print("\nA label where the LLM's PRECISION is high is a routing")
        print("candidate: its YES calls can be trusted where the bag says NO.")
        print("Run src/factor_route_eval.py next for the cell analysis.")

    if args.extra_csv:
        ex = load(args.extra_csv)
        ex["row_id"] = ex["row_id"].astype(str)
        eo, _ = run(ex, "test")
        p = args.out + ".extra.csv"
        eo.to_csv(p, index=False)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
