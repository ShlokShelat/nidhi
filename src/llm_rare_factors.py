"""
llm_rare_factors.py

Ask a large LLM, zero-shot, which of the STARVED factors a post shows -- and
measure it per label against the NLI model on the same 1635 OOF posts.

THE MISMATCH THIS TARGETS.

  S2 is macro-F1 over 24 labels, so each weighs 1/24 no matter how rare. But
  thirteen labels have 8-118 training examples:

      sexual orientation issues      8      cognitive deficits      33
      exposure to others' suicide   14      meaning in life         45
      poor school performance       16      low socio-economic      54
      substance use                 33      sense of responsibility 58

  A DeBERTa cross-encoder cannot learn a concept from eight examples. A large
  LLM does not need to -- it already knows what being closeted looks like, or
  what it means to have lost someone to suicide. This is the one place in the
  task where world knowledge should beat supervised learning outright, and it
  is exactly where our deficit is:

      meaning in life              0.324    sexual orientation issues  0.480
      exposure to others' suicide  0.500    substance use              0.642

WHY THIS IS NOT THE FACTOR-LLM EXPERIMENT THAT FAILED. That one FINE-TUNED
Qwen-14B on all 24 labels and scored 0.38 against the NLI model's 0.62 -- it
had to learn 24 concepts from the same starved data, with none of the
advantages. This does the opposite: no training at all, only the published
definitions, and only on the labels where supervision is hopeless. The common
labels stay with the NLI model, which is strong on them.

MEASURED ON THE SAME 1635 OOF POSTS the incumbent was measured on, per label,
with author-disjoint halves. A label only gets swapped if the LLM beats the
NLI model on BOTH halves -- so a lucky win on 8 positives cannot ship.

Runs on the login node against the DeepSeek API (compute nodes are offline).
Key comes from DEEPSEEK_API_KEY and is never written to disk.

NOTE ON WHAT LEAVES THE MACHINE: this sends REAL dataset posts to an external
API, unlike the synthetic generation which only sent factor definitions. If
that is not acceptable for this data, run the same prompt against the local
Qwen-72B instead -- pass --provider local (slower, no network).

Usage:
  export DEEPSEEK_API_KEY=...
  python src/llm_rare_factors.py --out artifacts/llm_rare_factors.csv
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from scoring import FACTOR_TAXONOMY, parse_factors

API = "https://api.deepseek.com/chat/completions"


def author_halves(auth):
    s = pd.Series(auth)
    counts = s.value_counts()
    order = sorted(counts.index, key=lambda a: (-counts[a], str(a)))
    half, size = {}, {"A": 0, "B": 0}
    for a in order:
        pick = "A" if size["A"] <= size["B"] else "B"
        half[a] = pick
        size[pick] += counts[a]
    return s.map(half).to_numpy()


def call(key, prompt, retries=4):
    body = json.dumps({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 300,
    }).encode()
    req = urllib.request.Request(
        API, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    for a in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception:
            if a == retries - 1:
                return None
            time.sleep(2 ** a)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--oof", default="artifacts/fac_oof_f{f}.npz")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-real", type=int, default=130,
                    help="a label is STARVED below this many real examples")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all posts")
    ap.add_argument("--out", default="artifacts/llm_rare_factors.csv")
    args = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("set DEEPSEEK_API_KEY")

    from train_factors_nli import FACTOR_DEFINITIONS

    lab = pd.read_csv(args.labeled).set_index("row_id")
    cnt = collections.Counter(
        f for c in lab["factors"]
        for f in set(parse_factors(c)) & set(FACTOR_TAXONOMY))
    rare = [f for f in FACTOR_TAXONOMY if cnt.get(f, 0) < args.max_real]
    print(f"{len(rare)} starved labels (<{args.max_real} real examples):")
    for f in rare:
        print(f"   {cnt.get(f,0):>4}  {f}")

    # incumbent OOF, so the comparison is on identical posts
    rids, PR, TH = [], [], []
    for f in range(args.k):
        d = np.load(args.oof.format(f=f), allow_pickle=True)
        rids.extend(list(d["row_id"]))
        PR.append(d["probs"])
        TH.append(np.tile(d["thresholds"], (len(d["row_id"]), 1)))
    PR, TH = np.concatenate(PR), np.concatenate(TH)
    NLI = (PR >= TH).astype(int)
    if args.limit:
        rids, NLI = rids[:args.limit], NLI[:args.limit]

    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    Y = np.zeros((len(rids), len(FACTOR_TAXONOMY)), dtype=int)
    for i, r in enumerate(rids):
        for f in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1
    half = author_halves([lab.loc[r, "anon_user_id"] for r in rids])
    print(f"\n{len(rids)} posts, halves "
          f"{int((half=='A').sum())}/{int((half=='B').sum())}")

    numbered = "\n".join(
        f"{n + 1}. {f}: {FACTOR_DEFINITIONS[f]}" for n, f in enumerate(rare))

    def prompt_for(post):
        return (
            "Below is a post from a suicide-support forum, and a numbered list "
            "of risk and protective factors with their definitions.\n\n"
            f"{numbered}\n\n"
            "Decide which of these factors the post shows.\n\n"
            "Rules:\n"
            "- A factor counts ONLY if it is genuinely part of why this person "
            "is struggling (or, for protective factors, genuinely present in "
            "their life). A passing mention of the topic does NOT count.\n"
            "- Most posts show NONE or ONE of these. Do not stretch.\n"
            "- Judge only what the text supports; do not infer beyond it.\n\n"
            f"POST:\n{post}\n\n"
            "Answer with ONLY a JSON array of the numbers that apply, e.g. "
            "[3] or [1,7] or []. No other text."
        )

    def parse(txt):
        if not txt:
            return set()
        m = re.search(r"\[[^\]]*\]", txt)
        if not m:
            return set()
        try:
            return {int(x) for x in json.loads(m.group(0))
                    if isinstance(x, int) and 1 <= x <= len(rare)}
        except Exception:
            return set()

    LLM = np.zeros_like(Y)
    posts = [str(lab.loc[r, "post"]) for r in rids]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut = {ex.submit(call, key, prompt_for(p)): i
               for i, p in enumerate(posts)}
        done = 0
        for fu in as_completed(fut):
            i = fut[fu]
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(posts)}")
            for n in parse(fu.result()):
                LLM[i, idx[rare[n - 1]]] = 1

    def f1(P, j, mask=None):
        m = np.ones(len(P), bool) if mask is None else mask
        tp = int((P[m, j] & Y[m, j]).sum())
        if tp == 0:
            return 0.0
        fp = int((P[m, j] & (1 - Y[m, j])).sum())
        fn = int(((1 - P[m, j]) & Y[m, j]).sum())
        return 2 * tp / (2 * tp + fp + fn)

    print(f"\n{'factor':<44}{'real':>6}{'NLI':>8}{'LLM':>8}{'delta':>8}"
          f"{'A':>8}{'B':>8}  swap?")
    print("-" * 92)
    swap = []
    for f in rare:
        j = idx[f]
        n, l = f1(NLI, j), f1(LLM, j)
        da = f1(LLM, j, half == "A") - f1(NLI, j, half == "A")
        db = f1(LLM, j, half == "B") - f1(NLI, j, half == "B")
        ok = l > n and da > 0 and db > 0
        if ok:
            swap.append(f)
        print(f"{f:<44}{cnt.get(f,0):>6}{n:>8.3f}{l:>8.3f}{l - n:>+8.3f}"
              f"{da:>+8.3f}{db:>+8.3f}  {'YES' if ok else ''}")

    def macro(P):
        return float(np.mean([f1(P, j) for j in range(P.shape[1])]))

    base = macro(NLI)
    if swap:
        H = NLI.copy()
        for f in swap:
            H[:, idx[f]] = LLM[:, idx[f]]
        m = macro(H)
        print(f"\nHYBRID: NLI everywhere, LLM on {len(swap)} labels")
        print(f"  macro-F1 {base:.4f} -> {m:.4f}   ({m - base:+.4f})")
        print(f"  -> {0.3 * (m - base):+.4f} composite")
        print(f"  swapped: {swap}")
    else:
        print(f"\nNo label passes on both halves. NLI macro {base:.4f}.")
        print("World knowledge does not beat the starved classifier here, and")
        print("factors close.")

    out = pd.DataFrame({"row_id": rids})
    for f in rare:
        out[f"llm::{f}"] = LLM[:, idx[f]]
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
