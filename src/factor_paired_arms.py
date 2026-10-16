"""
factor_paired_arms.py

Judge factor arms by PAIRED author-bootstrap deltas instead of absolute macro-F1.

WHY THIS EXISTS. Every factor arm this campaign has been ranked by its absolute
number on the 372-post val split -- mnli 0.6218, dmnli 0.6177, mnlicm 0.6135,
mnli_s15 0.6011. Because a re-run of an IDENTICAL configuration landed 0.0207
away from its twin, that spread was declared a noise floor and every arm ranking
was written off as unresolvable.

That reasoning conflates two different quantities. 0.0207 is the noise of an
ABSOLUTE score, which carries the full difficulty of whichever 372 posts and
whichever rare-label positives happen to be in the split. But every arm is
scored on the SAME posts, so the interesting quantity is the PAIRED delta, where
post difficulty, author composition and rare-label scarcity all cancel. Paired
noise can be far smaller than absolute noise, and nothing has ever measured it.

If a real arm's paired delta clears the paired null, the existing 372-post
instrument is usable and the 5-fold OOF rebuild (about 50 GPU jobs) is not
needed. If it does not, we build the rebuild knowing why.

THE NULL IS MEASURED, NOT ASSUMED. Pass the identical-configuration replica pair
(mnli vs mnli_s15) as `--null`. Its paired delta is a draw from the
no-real-difference distribution: same recipe, same data, different
nondeterminism. Any candidate arm must beat THAT, not zero. This is the honest
bar and it is stricter than a CI excluding zero.

BOOTSTRAP OVER AUTHORS, NOT POSTS. Posts from one author share a label
distribution (gold factors correlate up to 6.2x within author), so resampling
posts would treat correlated observations as independent and produce a CI that
is too narrow. Authors are the independent unit.

WHAT IS OPTIMISTIC AND WHY IT IS STILL FAIR. Each run's thresholds were tuned on
this same val split, and the trainer also picked its best EPOCH on it, so the
absolute levels here are in-sample and flattered. That is not corrected, because
it is exactly what deploys -- and the flattery applies to both sides of every
comparison, so it largely cancels in the delta. Read the deltas, not the levels.

Usage:
  python src/factor_paired_arms.py \
      --arm mnli:factors_nli_mnli,factors_nli_mnli_s2 \
      --arm champ:factors_nli,factors_nli_s2 \
      --arm dmnli:factors_nli_dmnli,factors_nli_dmnli_s2 \
      --null mnli_s15:factors_nli_mnli_s15,factors_nli_mnli_s15_s2 \
      --vs mnli
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402

N = len(FACTOR_TAXONOMY)


def gold_matrix(df: pd.DataFrame) -> np.ndarray:
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    Y = np.zeros((len(df), N), dtype=np.int8)
    for i, cell in enumerate(df["factors"]):
        for f in parse_factors(cell):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1
    return Y


def macro_f1(P: np.ndarray, Y: np.ndarray) -> float:
    """Macro-F1 over the 24 labels, matching scoring.factors_macro_f1 exactly.

    That uses sklearn with zero_division=0, so a label with no gold positives
    AND no predictions scores 0.0 -- NOT 1.0. The distinction is not academic
    here: a bootstrap resample can easily drop every positive of a rare label,
    and the wrong convention would silently reward arms for predicting nothing.
    Verified equal to the sklearn path in tests/check_paired.py."""
    tp = (P & Y).sum(0).astype(float)
    fp = (P & (1 - Y)).sum(0).astype(float)
    fn = ((1 - P) & Y).sum(0).astype(float)
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)
    return float(f1.mean())


def load_arm(runs: list[str], d: str) -> tuple[np.ndarray, np.ndarray, list]:
    """The DEPLOYED operation: average the runs' probabilities, and separately
    average the runs' own val-fitted thresholds (predict_factors_nli.py:63-69)."""
    PR, TH, rid = [], [], None
    for r in runs:
        p = os.path.join(d, f"{r}_val.npz")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        z = np.load(p, allow_pickle=True)
        PR.append(np.asarray(z["probs"], dtype=float))
        TH.append(np.asarray(z["thresholds"], dtype=float))
        r_ids = list(z["row_id"])
        if rid is None:
            rid = r_ids
        elif rid != r_ids:
            raise SystemExit(f"{r}: row_id order differs from the first run")
    return np.mean(PR, axis=0), np.mean(TH, axis=0), rid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[],
                    help="NAME:run1,run2 -- repeatable")
    ap.add_argument("--null", default=None,
                    help="NAME:run1,run2 for an IDENTICAL-config replica of "
                         "--vs; its delta measures the true noise floor")
    ap.add_argument("--vs", default=None,
                    help="reference arm name (default: the first --arm)")
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    specs = list(args.arm) + ([args.null] if args.null else [])
    if len(specs) < 2:
        raise SystemExit("need at least two arms to compare")

    arms, null_name = {}, None
    rid_ref = None
    for s in specs:
        name, _, rl = s.partition(":")
        runs = [x for x in rl.split(",") if x]
        if not runs:
            raise SystemExit(f"bad --arm/--null spec: {s!r}")
        pr, th, rid = load_arm(runs, args.dir)
        if rid_ref is None:
            rid_ref = rid
        elif rid != rid_ref:
            raise SystemExit(f"{name}: row_id order differs from the first arm")
        arms[name] = (pr >= th).astype(np.int8)
        if args.null and s == args.null:
            null_name = name

    va = pd.read_csv(args.val, keep_default_na=False)
    va = va.set_index("row_id").loc[rid_ref].reset_index()
    Y = gold_matrix(va)
    auth = va["anon_user_id"].to_numpy() if "anon_user_id" in va.columns \
        else np.arange(len(va))
    uniq = pd.unique(auth)
    by_author = {a: np.flatnonzero(auth == a) for a in uniq}

    ref = args.vs or specs[0].split(":")[0]
    if ref not in arms:
        raise SystemExit(f"--vs {ref!r} is not one of {list(arms)}")

    print(f"{len(va)} val posts, {len(uniq)} authors, {len(arms)} arms")
    print(f"reference: {ref}")
    print(f"gold positives per label: min {int(Y.sum(0).min())}  "
          f"median {int(np.median(Y.sum(0)))}  max {int(Y.sum(0).max())}")
    print("  Labels with a handful of positives are why the ABSOLUTE macro is")
    print("  noisy; pairing removes that, because both arms face the same few.")

    print("\n" + "=" * 78)
    print("ABSOLUTE macro-F1 (the numbers we have been ranking by)")
    print("=" * 78)
    for nm, P in arms.items():
        dead = int(((P.sum(0) == 0) & (Y.sum(0) > 0)).sum())
        print(f"  {nm:<12}{macro_f1(P, Y):.4f}"
              + (f"   {dead} label(s) predicted zero times" if dead else ""))

    # ---- paired author bootstrap --------------------------------------------
    rng = np.random.default_rng(args.seed)
    idx_draws = []
    for _ in range(args.boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx_draws.append(np.concatenate([by_author[uniq[k]] for k in pick]))

    base = {nm: macro_f1(P, Y) for nm, P in arms.items()}
    boot = {nm: np.empty(args.boot) for nm in arms}
    for b, ix in enumerate(idx_draws):
        Yb = Y[ix]
        for nm, P in arms.items():
            boot[nm][b] = macro_f1(P[ix], Yb)

    print("\n" + "=" * 78)
    print(f"PAIRED delta vs {ref}   ({args.boot} author-bootstrap resamples)")
    print("=" * 78)
    print(f"{'arm':<12}{'delta':>9}{'95% CI':>20}{'win%':>8}{'verdict':>26}")
    print("-" * 78)

    null_hi = None
    order = [n for n in arms if n != ref]
    if null_name and null_name in order:      # evaluate the null first
        order = [null_name] + [n for n in order if n != null_name]

    for nm in order:
        d = boot[nm] - boot[ref]
        lo, hi = np.percentile(d, [2.5, 97.5])
        win = float((d > 0).mean()) * 100
        pt = base[nm] - base[ref]
        if nm == null_name:
            null_hi = max(abs(lo), abs(hi))
            verdict = "<- NULL, this is noise"
        elif null_hi is None:
            verdict = "real" if lo > 0 else ("worse" if hi < 0 else "unresolved")
        elif pt > null_hi:
            verdict = "REAL, beats the null band"
        elif pt < -null_hi:
            verdict = "WORSE, beyond the null band"
        else:
            verdict = "inside the null band"
        print(f"{nm:<12}{pt:>+9.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>20}"
              f"{win:>7.1f}%{verdict:>26}")

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    if null_hi is not None:
        print(f"The null pair is an IDENTICAL configuration re-run, so its true")
        print(f"delta is zero and everything it shows is noise. Its paired band")
        print(f"reaches +/-{null_hi:.4f}.")
        print(f"Compare that with the 0.0207 spread between the same two runs'")
        print(f"ABSOLUTE scores: pairing removes the shared post difficulty.")
        print(f"An arm is only real if its delta clears +/-{null_hi:.4f} AND its")
        print("win rate is at least 80%.")
    else:
        print("No --null given, so there is no measured noise floor and a CI")
        print("excluding zero is the weaker bar of the two. Pass the")
        print("identical-config replica pair as --null before trusting any")
        print("'real' verdict here.")
    print("\nAbsolute levels are in-sample (thresholds AND the epoch were picked")
    print("on these same posts) so they are flattered -- but both sides of every")
    print("delta are flattered equally. Read deltas, never levels.")


if __name__ == "__main__":
    main()
