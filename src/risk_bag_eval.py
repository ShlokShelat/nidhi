"""
risk_bag_eval.py

Evaluate a MULTI-FAMILY generative risk bag against the encoder chain.

WHY A MULTI-FAMILY BAG RATHER THAN AN INCREMENTAL CHANGE. Closing a
meaningful gap to the top of the held-out evaluation from one component alone
would need a large jump in that component's score, well beyond what incremental
tuning typically delivers. The one structural move with external validation
is a published recipe used by a prior top-placing entry on a related task:

    "ensemble of prompted Qwen2-72B (few-shot CoT) + LoRA fine-tuned Llama3-8B,
     Llama3.1-8B, Gemma2-9B, 5-fold CV checkpoints, majority voting"

A single LoRA'd Qwen2.5-7B, trained once for 3 epochs, already matches the
encoder ensemble (0.7694 vs 0.7683 on the same 372 posts) with a confidence
gap the encoders lack (0.892 right vs 0.703 wrong). `maxconf` between just
those two has a large oracle headroom. A bag of differently-wrong families is
what converts an oracle like that into score.

WHAT DECIDES IT. Not any single member's accuracy -- members at ~0.77 are fine
if they fail on DIFFERENT posts. So this reports, for every subset:
  - accuracy and weighted-F1
  - pairwise error overlap (low overlap is the whole point)
  - the ORACLE for that subset, bounding what any combiner could reach
and combines with rules that fit NOTHING, because 0-parameter rules
over-deliver here while a 1-fitted-parameter rule has historically transferred
poorly on this task:
    majority       plurality vote, ties broken by summed probability
    prob-avg       mean of the members' probability vectors
    logit-avg      mean in logit space, so a sharper member cannot dominate
    maxconf        believe whichever member is most confident

EVERY MEMBER MUST BE HONEST OOF ON THE SAME FOLDS. The fold models come from
artifacts/folds/, which is also what built artifacts/nligaur/oof_base.csv, so
the encoder baseline is paired by construction. Members trained with
--no-val-select, so no member's probabilities were tuned on the split they are
scored against.

Usage:
  python src/risk_bag_eval.py --members \
      qwen=artifacts/lora_lora7b_oof.csv \
      llama=artifacts/lora_loraL8_oof.csv \
      mistral=artifacts/lora_loraM7_oof.csv
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1  # noqa: E402

PC = [f"prob_{c}" for c in RISK_LEVELS]


def load(path: str, gold_col: str | None = None):
    d = pd.read_csv(path, keep_default_na=False)
    d["row_id"] = d["row_id"].astype(str)
    d = d.drop_duplicates("row_id")
    miss = [c for c in PC if c not in d.columns]
    if miss:
        raise SystemExit(f"{path} lacks {miss}")
    g = None
    for c in (gold_col, "gold", "risk_level"):
        if c and c in d.columns:
            g = [normalize_label(x) for x in d[c]]
            break
    return d.set_index("row_id"), g


def to_logit(P):
    P = np.clip(P, 1e-6, 1 - 1e-6)
    return np.log(P / (1 - P))


def combine(mats, rule):
    if rule == "prob-avg":
        return np.mean(mats, axis=0)
    if rule == "logit-avg":
        return np.mean([to_logit(M) for M in mats], axis=0)
    if rule == "maxconf":
        conf = np.stack([M.max(1) for M in mats])       # (k, N)
        pick = conf.argmax(0)
        return np.stack([mats[pick[i]][i] for i in range(mats[0].shape[0])])
    if rule == "majority":
        N, K = mats[0].shape
        votes = np.zeros((N, K))
        for M in mats:
            votes[np.arange(N), M.argmax(1)] += 1
        # ties broken by summed probability, so the rule stays deterministic
        votes = votes + np.mean(mats, axis=0) * 1e-3
        return votes
    raise ValueError(rule)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True,
                    help="name=path.csv, each with row_id + prob_* (+ gold)")
    ap.add_argument("--encoder", default="artifacts/nligaur/oof_base.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    enc_d, _ = load(args.encoder)
    mem = {}
    for spec in args.members:
        if "=" not in spec:
            raise SystemExit(f"--members wants name=path, got {spec!r}")
        n, p = spec.split("=", 1)
        if not os.path.exists(p):
            print(f"  skip {n}: {p} not found")
            continue
        mem[n] = load(p)[0]
    if not mem:
        raise SystemExit("no member files found")

    rows = sorted(set(enc_d.index).intersection(*[m.index for m in mem.values()]))
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    rows = [r for r in rows if r in lab.index]
    y = np.array([normalize_label(lab.loc[r, "risk_level"]) for r in rows])
    author = np.array([str(lab.loc[r, "anon_user_id"]) for r in rows])
    print(f"{len(rows)} posts common to all members / {len(set(author))} authors")

    M = {"encoder": enc_d.loc[rows, PC].to_numpy(float)}
    for n, d in mem.items():
        M[n] = d.loc[rows, PC].to_numpy(float)
    names = list(M)
    pred = {n: np.array([RISK_LEVELS[i] for i in M[n].argmax(1)]) for n in names}

    print(f"\n{'member':<12}{'wF1':>9}{'acc':>8}")
    print("-" * 29)
    for n in names:
        print(f"{n:<12}{risk_weighted_f1(list(y), list(pred[n])):>9.4f}"
              f"{(pred[n] == y).mean():>8.3f}")

    print("\nPAIRWISE ERROR OVERLAP -- low is the point of a bag")
    print(f"{'':<12}" + "".join(f"{n[:9]:>10}" for n in names))
    for a in names:
        line = f"{a[:11]:<12}"
        for b in names:
            wa, wb = pred[a] != y, pred[b] != y
            both = int((wa & wb).sum())
            line += f"{both / max(int((wa | wb).sum()), 1):>10.3f}"
        print(line)
    print("(Jaccard of error sets: 1.0 = identical failures, 0 = disjoint)")

    base = risk_weighted_f1(list(y), list(pred["encoder"]))
    rng = np.random.default_rng(args.seed)
    auths = sorted(set(author))
    aidx = {a: np.where(author == a)[0] for a in auths}

    def ci(p):
        draws = []
        for _ in range(args.boot):
            pick = rng.choice(len(auths), len(auths), replace=True)
            sel = np.concatenate([aidx[auths[i]] for i in pick])
            draws.append(risk_weighted_f1(list(y[sel]), list(p[sel]))
                         - risk_weighted_f1(list(y[sel]), list(pred["encoder"][sel])))
        return np.percentile(draws, [2.5, 97.5])

    print(f"\n{'subset':<34}{'rule':<11}{'wF1':>8}{'delta':>9}"
          f"{'95% CI':>22}{'oracle':>9}")
    print("-" * 93)
    subsets = []
    for k in range(2, len(names) + 1):
        subsets += [s for s in itertools.combinations(names, k)
                    if "encoder" in s]
    subsets += [tuple(n for n in names if n != "encoder")]  # generative only
    seen = set()
    for sub in subsets:
        if sub in seen or len(sub) < 2:
            continue
        seen.add(sub)
        mats = [M[n] for n in sub]
        # oracle: right if ANY member is right
        anyright = np.zeros(len(rows), bool)
        for n in sub:
            anyright |= (pred[n] == y)
        orc = np.where(anyright, y, pred[sub[0]])
        o = risk_weighted_f1(list(y), list(orc))
        tag = "+".join(n[:4] for n in sub)
        for rule in ("majority", "prob-avg", "logit-avg", "maxconf"):
            if rule == "majority" and len(sub) < 3:
                continue
            C = combine(mats, rule)
            p = np.array([RISK_LEVELS[i] for i in C.argmax(1)])
            f1 = risk_weighted_f1(list(y), list(p))
            lo, hi = ci(p)
            star = "  <-" if lo > 0 else ""
            print(f"{tag:<34}{rule:<11}{f1:>8.4f}{f1 - base:>+9.4f}"
                  f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{o - base:>+9.4f}{star}")

    print("\n" + "=" * 93)
    print(f"encoder baseline wF1 {base:.4f}. Composite impact = 0.4 x delta.")
    print("A row marked <- has its 95% paired-author CI clear of zero.")
    print("Read the ORACLE column too: it bounds what ANY combiner of that")
    print("subset could reach, so a small delta beside a large oracle means")
    print("the members are complementary and the COMBINER is the weak part.")


if __name__ == "__main__":
    main()
