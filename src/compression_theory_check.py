"""
compression_theory_check.py

DOES THE MEASURED THRESHOLD COMPRESSION MATCH WHAT AVERAGING PREDICTS?

Thresholds are fitted on cross-validated predictions, where each post is scored
by the one fold model that did not train on it, and applied to test
predictions, where each post is scored by the average of all k fold models.
Averaging contracts the score distribution, so the threshold no longer sits
where it was placed.

The size of that contraction is not mysterious. For k predictors each of
variance s^2 with mean pairwise correlation rho,

    Var(mean) = s^2 * [ rho + (1 - rho) / k ]

so the spread of the averaged scores relative to a single model is

    sigma_avg / sigma_single = sqrt( rho + (1 - rho) / k )                (1)

This script tests (1) against what we actually measure, and emits the table
that Figure 2 is drawn from.

TWO MODES, DEPENDING ON WHAT IS ON DISK.

  DIRECT   If per-fold test predictions exist, rho is computed straight from
           them and (1) becomes a genuine prediction that can be checked
           against the measured ratio. This is the strong version.

  IMPLIED  If only the averaged test dump exists, rho cannot be observed
           independently. We then invert (1) to report the rho each arm's
           measured compression implies, and the check becomes weaker: it
           asks whether those implied correlations are plausible for fold
           models of one architecture trained on overlapping data, rather
           than confirming the formula outright. The script says clearly
           which mode it ran in, because the two support different claims.

SECURITY: reads local probability dumps only, prints aggregate statistics.

Usage:
  python src/compression_theory_check.py
  python src/compression_theory_check.py --csv artifacts/compression_theory.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY  # noqa: E402

N = len(FACTOR_TAXONOMY)
SHIPPED = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7", "lnli",
           "ltail", "cbnsynapi", "friendfac"]


def logit(x):
    x = np.clip(np.asarray(x, float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def predicted_ratio(rho, k):
    return float(np.sqrt(rho + (1.0 - rho) / k))


def implied_rho(ratio, k):
    """Invert (1). ratio is sigma_avg / sigma_single, so at most 1."""
    r2 = float(ratio) ** 2
    return (r2 - 1.0 / k) / (1.0 - 1.0 / k)


def load_oof(d, arm, k):
    pr, ids = [], []
    for f in range(k):
        p = os.path.join(d, f"{arm}_cv_oof_f{f}.npz")
        if not os.path.exists(p):
            return None, None
        z = np.load(p, allow_pickle=True)
        pr.append(z["probs"].astype(float))
        ids.extend(str(x) for x in z["row_id"])
    return np.concatenate(pr), ids


def load_test(d, arm):
    for c in (f"{arm}_cv_test.npz", f"{arm}_test.npz",
              f"nli_ensemble_{arm}_test.npz"):
        p = os.path.join(d, c)
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            keys = list(z.keys())
            return z, keys
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/factor_runs")
    ap.add_argument("--arms", nargs="+", default=SHIPPED)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    k = args.folds
    rows = []
    mode_direct = 0

    for arm in args.arms:
        oof, _ = load_oof(args.dir, arm, k)
        z, keys = load_test(args.dir, arm)
        if oof is None or z is None:
            print(f"  skipping {arm}: dumps not found")
            continue
        test = z["probs"].astype(float)

        s_single = float(logit(oof).std())
        s_avg = float(logit(test).std())
        measured = s_avg / max(s_single, 1e-12)

        # DIRECT mode needs each fold model's own test predictions. Look for a
        # per-fold array under any plausible key before giving up.
        rho_obs = None
        per = None
        for cand in ("fold_probs", "probs_per_fold", "fold_test", "members"):
            if cand in keys:
                per = np.asarray(z[cand], dtype=float)
                break
        if per is not None and per.ndim == 3 and per.shape[0] == k:
            L = logit(per)                       # (k, n_posts, n_labels)
            cs = []
            for j in range(N):
                M = L[:, :, j]
                C = np.corrcoef(M)
                iu = np.triu_indices(k, 1)
                if np.all(np.isfinite(C[iu])):
                    cs.append(float(C[iu].mean()))
            if cs:
                rho_obs = float(np.mean(cs))
                mode_direct += 1

        rho_imp = implied_rho(measured, k)
        rows.append({
            "arm": arm,
            "sd_single_cv": s_single,
            "sd_averaged_test": s_avg,
            "measured_ratio": measured,
            "implied_rho": rho_imp,
            "observed_rho": rho_obs if rho_obs is not None else np.nan,
            "predicted_ratio": (predicted_ratio(rho_obs, k)
                                if rho_obs is not None else np.nan),
        })

    if not rows:
        raise SystemExit("no arms loaded; check --dir")
    df = pd.DataFrame(rows)

    direct = mode_direct == len(df)
    print("=" * 76)
    print("MODE: " + ("DIRECT (per-fold test predictions found)"
                      if direct else
                      "IMPLIED (only averaged test dumps on disk)"))
    print("=" * 76)
    if not direct:
        print("  Per-fold test predictions were not found, so rho cannot be")
        print("  observed independently. The table reports the correlation")
        print("  each arm's measured compression IMPLIES under the formula.")
        print("  Read it as a consistency check, not as a confirmation.\n")

    print(f"{'arm':<13}{'sd cv':>9}{'sd test':>9}{'ratio':>8}"
          f"{'implied rho':>13}" + (f"{'obs rho':>10}{'pred ratio':>12}"
                                    if direct else ""))
    for r in df.itertuples():
        line = (f"{r.arm:<13}{r.sd_single_cv:>9.3f}{r.sd_averaged_test:>9.3f}"
                f"{r.measured_ratio:>8.3f}{r.implied_rho:>13.3f}")
        if direct:
            line += f"{r.observed_rho:>10.3f}{r.predicted_ratio:>12.3f}"
        print(line)

    print(f"\n  k = {k} fold models averaged")
    print(f"  measured ratio spans {df.measured_ratio.min():.3f} to "
          f"{df.measured_ratio.max():.3f}")
    print(f"  implied rho  spans {df.implied_rho.min():.3f} to "
          f"{df.implied_rho.max():.3f}")
    lo, hi = df.implied_rho.min(), df.implied_rho.max()
    if 0.3 <= lo and hi <= 0.99:
        print("  Those correlations are in the range one would expect for")
        print("  fold models of one architecture trained on overlapping")
        print("  data, which is what the formula requires to be consistent.")
    else:
        print("  WARNING: implied correlations fall outside a plausible")
        print("  range, so the averaging account does not by itself explain")
        print("  the measured compression. Say so in the paper.")

    if direct:
        err = (df.predicted_ratio - df.measured_ratio).abs()
        print(f"\n  |predicted - measured| mean {err.mean():.4f}, "
              f"max {err.max():.4f}")
        print("  This is the number Figure 2 exists to show.")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}  <- Figure 2 is drawn from this")


if __name__ == "__main__":
    main()
