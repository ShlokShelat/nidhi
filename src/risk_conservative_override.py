"""
risk_conservative_override.py

Overrule the champion only when the challengers are UNANIMOUS -- and validate
against an OOF analogue of the DEPLOYED champion, not a weaker surrogate.

WHY UNANIMITY, NOT MAJORITY. A plain majority vote over {encoder, cot72b,
llama} measured favorably offline (1635 posts / 153 authors) but underperforms
once deployed, because a majority vote counts heads and has no notion of
member quality:

    offline members  0.7985 / 0.7865 / 0.8006   roughly EQUAL -> voting helps
    deployed members 0.8203 / ~0.787  / ~0.800  champion FAR stronger
                                                -> two weaker models outvote it

TWO FIXES, BOTH APPLIED HERE.

1. A BETTER BASELINE. The deployed champion's risk IS route_attempt on top of
   the chain, and `route_attempt(oof_base)` scores 0.8055 -- much closer to the
   deployed 0.8203 than oof_base (0.7985) or champion_oof (0.7858). Validating
   against it removes most of the surrogate gap. All three are reported.

2. CONSERVATIVE OVERRIDE. "Weight the champion at 2" and "only overrule where
   the champion is weak" are the same family once you write them down: overrule
   ONLY when every challenger agrees against the champion. UNANIMITY is
   0-parameter, is strictly more conservative than majority, and degrades
   safely as the champion gets stronger -- the exact property majority lacks.

   With k challengers, unanimity is equivalent to giving the champion weight k
   and breaking ties in its favour. So the "weighted vote" idea IS the
   unanimity rule; no threshold to fit.

WHAT DECIDES IT. Not the delta alone -- the FIX/BREAK count on the moved
posts. A rule that moves many posts for a small delta is churn, not signal,
regardless of how favorable the delta looks in isolation.

Class-restricted variants are included because Llama-70B's per-class profile is
extreme (indicator 0.903 and behavior 0.801, both best of any member, against
ideation 0.582) -- a member can own a cell without being a good general voter.

Usage:
  python src/risk_conservative_override.py
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


def read(path, prefer=None):
    d = pd.read_csv(path, keep_default_na=False)
    d["row_id"] = d["row_id"].astype(str)
    return d.drop_duplicates("row_id").set_index("row_id")


def labels_of(d, rows):
    for c in ("cot", "pred_risk_level", "risk_level"):
        if c in d.columns:
            return np.array([normalize_label(x) for x in d.loc[rows, c]])
    M = d.loc[rows, PC].to_numpy(float)
    return np.array([RISK_LEVELS[i] for i in M.argmax(1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="artifacts/nligaur/oof_base.csv")
    ap.add_argument("--members", nargs="+", default=[
        "qwen72=artifacts/cot_soft_oof.csv",
        "llama70=artifacts/cot_soft_llama_oof.csv",
        "llama8=artifacts/lora_loraL8_oof.csv",
        "mistral=artifacts/lora_loraM7_oof.csv",
    ])
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")

    E = read(args.enc)
    M = {}
    for spec in args.members:
        n, p = spec.split("=", 1)
        if not os.path.exists(p):
            print(f"  skip {n}: {p} missing")
            continue
        M[n] = read(p)
    if not M:
        raise SystemExit("no member files found")

    rows = sorted(set(E.index).intersection(*[m.index for m in M.values()]))
    rows = [r for r in rows if r in lab.index]
    y = np.array([normalize_label(lab.loc[r, "risk_level"]) for r in rows])
    author = np.array([str(lab.loc[r, "anon_user_id"]) for r in rows])
    ep_raw = labels_of(E, rows)
    P = {n: labels_of(d, rows) for n, d in M.items()}
    print(f"{len(rows)} posts / {len(set(author))} authors")
    print(f"members: {list(P)}")

    # THE BASELINE THAT MATTERS: route_attempt on top of the encoder, which is
    # what the deployed champion's risk column actually is.
    ra = np.where(P.get("qwen72", ep_raw) == "attempt",
                  "attempt", ep_raw) if "qwen72" in P else ep_raw
    bases = {"oof_base (raw)": ep_raw,
             "route_attempt(oof_base)  <- closest to deployed": ra}

    print(f"\n{'member':<32}{'wF1':>9}")
    print("-" * 41)
    print(f"{'encoder (raw)':<32}{risk_weighted_f1(list(y), list(ep_raw)):>9.4f}")
    for n, p in P.items():
        print(f"{n:<32}{risk_weighted_f1(list(y), list(p)):>9.4f}")
    print(f"{'route_attempt(oof_base)':<32}{risk_weighted_f1(list(y), list(ra)):>9.4f}"
          f"   deployed champion is ~0.8203")

    rng = np.random.default_rng(args.seed)
    auths = sorted(set(author))
    aidx = {a: np.where(author == a)[0] for a in auths}

    def evaluate(name, pred, base, base_name):
        b = risk_weighted_f1(list(y), list(base))
        f = risk_weighted_f1(list(y), list(pred))
        mv = pred != base
        fix = int((mv & (pred == y)).sum())
        brk = int((mv & (base == y)).sum())
        draws = []
        for _ in range(args.boot):
            pick = rng.choice(len(auths), len(auths), replace=True)
            sel = np.concatenate([aidx[auths[i]] for i in pick])
            draws.append(risk_weighted_f1(list(y[sel]), list(pred[sel]))
                         - risk_weighted_f1(list(y[sel]), list(base[sel])))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        star = "  <-" if lo > 0 else ""
        print(f"  {name:<40}{f:>8.4f}{f - b:>+9.4f}"
              f"{int(mv.sum()):>7}{fix:>6}{brk:>6}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{star}")

    for base_name, base in bases.items():
        print(f"\n{'=' * 104}")
        print(f"BASELINE: {base_name}   wF1 {risk_weighted_f1(list(y), list(base)):.4f}")
        print("=" * 104)
        print(f"  {'rule':<40}{'wF1':>8}{'delta':>9}{'moved':>7}"
              f"{'fix':>6}{'brk':>6}{'95% CI':>22}")
        print("  " + "-" * 98)

        names = list(P)
        # UNANIMOUS override over every non-empty subset of challengers
        for k in range(2, len(names) + 1):
            for sub in itertools.combinations(names, k):
                agree = np.ones(len(rows), bool)
                for n in sub[1:]:
                    agree &= (P[n] == P[sub[0]])
                m = agree & (P[sub[0]] != base)
                pred = np.where(m, P[sub[0]], base)
                evaluate(f"unanimous({'+'.join(x[:5] for x in sub)})",
                         pred, base, base_name)

        # MAJORITY for contrast -- the vote-counting rule that underperforms
        # once the champion is much stronger than the other members
        for k in (2, 3, 4):
            if k > len(names):
                continue
            for sub in itertools.combinations(names, k):
                votes = {}
                allp = [base] + [P[n] for n in sub]
                pred = []
                for i in range(len(rows)):
                    c = {}
                    for a in allp:
                        c[a[i]] = c.get(a[i], 0) + 1
                    top = max(c.items(), key=lambda kv: kv[1])
                    pred.append(top[0] if top[1] > len(allp) / 2 else base[i])
                evaluate(f"MAJORITY(champ+{'+'.join(x[:5] for x in sub)})",
                         np.array(pred), base, base_name)
                break   # one example per size is enough for contrast

        # CLASS-RESTRICTED unanimity, for EVERY subset. A member can own a cell
        # without being a good general voter (Llama-70B: indicator 0.903 and
        # behavior 0.801, both best of any member, against ideation 0.582).
        # This is the shape route_attempt has -- narrow, one class, few posts,
        # high fix/brk -- which is the profile that has generalized best among
        # ensemble-style rules.
        print("  " + "-" * 98)
        print(f"  {'CLASS-RESTRICTED unanimity':<40}{'wF1':>8}{'delta':>9}"
              f"{'moved':>7}{'fix':>6}{'brk':>6}{'95% CI':>22}")
        print("  " + "-" * 98)
        for k in range(1, len(names) + 1):
            for sub in itertools.combinations(names, k):
                for c in RISK_LEVELS:
                    agree = np.ones(len(rows), bool)
                    for n in sub:
                        agree &= (P[n] == c)
                    m = agree & (base != c)
                    if m.sum() < 5:      # too few to measure
                        continue
                    pred = np.where(m, c, base)
                    tag = f"{'+'.join(x[:5] for x in sub)} -> {c}"
                    evaluate(tag, pred, base, base_name)

    print("\n" + "=" * 104)
    print("READ fix/brk, NOT the delta. Two rules can move the same NUMBER of")
    print("posts and be opposites: one concentrated on a class it gets right,")
    print("the other spread thin. A rule that moves many posts for a small")
    print("delta is churn, whatever the delta's sign.")
    print("\nUNANIMITY == the weighted vote: with k challengers it is exactly")
    print("'champion weight k, ties to the champion', so there is no threshold")
    print("to fit. It also degrades SAFELY as the champion gets stronger, which")
    print("is the property majority lacks and which sank bag3.")
    print("\n" + "=" * 104)
    print("REGIME MATCHING -- the second reason bag3 failed")
    print("=" * 104)
    print("  qwen72, llama70 : OOF and TEST both come from ONE prompted pass.")
    print("                    The event measured here is the event deployed.")
    print("  llama8, mistral : OOF is ONE fold model per post; TEST is the")
    print("                    AVERAGE OF FIVE. That is the documented paircm")
    print("                    mechanism -- averaging narrows the distribution")
    print("                    and moves the argmax -- so their agreement")
    print("                    pattern on test is NOT what was measured here.")
    print("                    mistral has no test predictions at all.")
    print("\nAnd OOF STRUCTURALLY UNDERSTATES the deployed champion: OOF uses")
    print("FOLD models (1308 posts), deployment uses FULL-DATA models (1635).")
    print("champion_oof 0.784 vs deployed 0.8203 -- a ~0.036 gap that cannot be")
    print("closed by picking a different baseline file. Every 'add a member'")
    print("rule exploits that gap, so a gain measured against the fold-model")
    print("baseline is absorbed by the deployed model and does not survive.")
    print("\nSHIP ONLY a rule that is ALL of:")
    print("  - class-restricted (the route_attempt shape)")
    print("  - built ONLY from regime-matched members (qwen72, llama70)")
    print("  - high fix/brk (>= 3:1), moving few posts (< 2% of the set)")
    print("  - CI clear of zero on the route_attempt(oof_base) baseline")
    print("All four together, not three of the four -- a rule that misses any")
    print("one of them is measuring the regime gap rather than a real effect.")


if __name__ == "__main__":
    main()
