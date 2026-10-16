"""
factor_test_calibration.py

Tests whether factor thresholds transfer from the OOF pool to the test pool,
and optionally rescales the test logits so that they do.

Thresholds are estimated on the OOF set, where each post is predicted by the
SINGLE fold model that held it out. The test dump is built by AVERAGING the
five fold models. Averaging shrinks a probability distribution toward its
centre, so the test scores a threshold is applied to need not be drawn from the
distribution the threshold was fitted on.

THE PREDICTION THAT MAKES THIS FALSIFIABLE. If test logits are compressed
relative to OOF, then for any label

    a HIGH threshold (rare labels) fires on FEWER posts than intended
    a LOW  threshold (common labels) fires on MORE posts than intended

and macro-F1 punishes the first case hardest, because a rare label that
under-fires has its recall capped no matter how good the ranking is.

WHAT THIS MEASURES, BEFORE FIXING ANYTHING
  1. per-label mean and sd of the pooled LOGIT on OOF vs on TEST. If the test
     sd is systematically smaller, the compression is real and measured.
  2. per-label firing count under the deployed thresholds vs expected
     prevalence.
  3. THE CAUSAL TEST: does the per-label sd ratio PREDICT the per-label count
     deficit? If compression causes the miscalibration those two must
     correlate. If they do not, the mechanism is not compression and the fix
     is unmotivated.

THE FIX IF IT CONFIRMS -- zero gold parameters, fully transductive:

    z_test' = mu_oof + (z_test - mu_test) * (sd_oof / sd_test)

per label. It uses only the two probability distributions, never a gold label,
so nothing is fitted to the thing being scored. It restores the scale the
thresholds were calibrated on. It is NOT a quantile decode -- the ranking and
the shape of the test distribution are untouched, only its location and scale.

Usage:
  python src/factor_test_calibration.py
  python src/factor_test_calibration.py --build artifacts/factors_scaled_test.csv
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
IDX = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
BAG8 = ["mnli", "dmnli", "cmine", "prot", "d4", "qnli", "mnli7", "lnli"]
# The deployed factor column pools ELEVEN arms, not eight:
# factor_threshold_stabilise.py uses BAG8 + ltail + cbnsynapi + friendfac.
# Defaulting this script to BAG8 would measure a pool that is not deployed, so
# any conclusion drawn from it would not transfer to the shipped decode.
SHIPPED11 = BAG8 + ["ltail", "cbnsynapi", "friendfac"]
TGRID = np.round(np.arange(0.02, 0.985, 0.005), 3)


def logit(x):
    x = np.clip(np.asarray(x, float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def cells(col, n):
    M = np.zeros((n, N), dtype=np.int8)
    for i, c in enumerate(col):
        for x in set(parse_factors(str(c))):
            j = IDX.get(str(x).strip().lower())
            if j is not None:
                M[i, j] = 1
    return M


def f1_curve(p, y):
    npos = int(y.sum())
    if npos == 0:
        return np.zeros(len(TGRID))
    o = np.argsort(-p, kind="stable")
    ys, ps = y[o], p[o]
    cum = np.cumsum(ys)
    ks = np.searchsorted(-ps, -TGRID, side="right")
    tp = np.where(ks > 0, cum[np.clip(ks - 1, 0, len(cum) - 1)], 0)
    return 2.0 * tp / np.maximum(ks + npos, 1)


def load(d, arms, pat, k=5):
    O, rid = [], None
    for a in arms:
        pr, ids = [], []
        if pat == "oof":
            for f in range(k):
                p = f"{d}/{a}_cv_oof_f{f}.npz"
                z = np.load(p, allow_pickle=True)
                pr.append(z["probs"].astype(float))
                ids.extend(str(x) for x in z["row_id"])
            pr = np.concatenate(pr)
        else:
            p = next((c for c in (f"{d}/{a}_cv_test.npz", f"{d}/{a}_test.npz",
                                  f"{d}/nli_ensemble_{a}_test.npz")
                      if os.path.exists(c)), None)
            if p is None:
                raise SystemExit(f"no test dump for {a}")
            z = np.load(p, allow_pickle=True)
            pr = z["probs"].astype(float)
            ids = [str(x) for x in z["row_id"]]
        if rid is None:
            rid = ids
        elif ids != rid:
            o = {x: i for i, x in enumerate(ids)}
            pr = pr[[o[x] for x in rid]]
        O.append(pr)
    return rid, np.stack(O)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--dir", default="artifacts/factor_runs")
    a_.add_argument("--arms", nargs="+", default=SHIPPED11)
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--shipped", default="submissions/NIDHI_ff11.csv")
    a_.add_argument("--build", default=None)
    a_.add_argument("--firing-csv", default=None,
                    help="write per-label prevalence and the test/OOF firing "
                         "ratio at the SAME threshold. This is the table the "
                         "compression figure is drawn from, and it is the "
                         "honest form of the claim: the threshold does not "
                         "move, only the distribution beneath it does.")
    args = a_.parse_args()

    rid, SO = load(args.dir, args.arms, "oof")
    tid, ST = load(args.dir, args.arms, "test")
    ZO = logit(SO).mean(0)          # pooled logit, OOF
    ZT = logit(ST).mean(0)          # pooled logit, TEST
    n, nT = len(rid), len(tid)

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    Y = cells([lab.loc[r, "factors"] for r in rid], n)

    ship = pd.read_csv(args.shipped, keep_default_na=False)
    shipM = cells(ship["factors"], len(ship))

    # argmax-F1 thresholds on the full OOF -- the deployed decode
    th = np.array([float(TGRID[int(np.argmax(f1_curve(sig(ZO[:, j]),
                                                      Y[:, j])))])
                   for j in range(N)])

    print(f"OOF {n} posts (ONE fold model each)   "
          f"TEST {nT} posts (average of five)\n")
    print("=" * 104)
    print("1-2. DISTRIBUTION AND FIRING RATE, PER LABEL")
    print("=" * 104)
    print(f"{'label':<38}{'gold%':>7}{'sdrat':>7}{'thr':>6}"
          f"{'oof%@thr':>10}{'test%@thr':>10}{'t/o':>6}"
          f"{'fires':>6}{'expect':>7}{'f/e':>6}")
    sdr, cntr, rate, tor = [], [], [], []
    for j in range(N):
        so, st = ZO[:, j].std(), ZT[:, j].std()
        exp = Y[:, j].mean() * nT
        fires = int(shipM[:, j].sum())
        # THE DECISIVE COLUMNS: the SAME threshold applied to both pools.
        # If these two rates agree there is no transfer problem and the
        # deviation from prevalence is just what an F1-optimal threshold does
        # on a weak rare label -- legitimate, not a bug.
        ro = float((sig(ZO[:, j]) >= th[j]).mean())
        rt = float((sig(ZT[:, j]) >= th[j]).mean())
        r_sd = st / max(so, 1e-9)
        r_ct = fires / max(exp, 1e-9)
        sdr.append(r_sd)
        cntr.append(r_ct)
        rate.append(Y[:, j].mean())
        tor.append(rt / max(ro, 1e-9))
        print(f"{FACTOR_TAXONOMY[j]:<38}{100*Y[:, j].mean():>6.1f}%"
              f"{r_sd:>7.2f}{th[j]:>6.2f}"
              f"{100*ro:>9.1f}%{100*rt:>9.1f}%{rt/max(ro,1e-9):>6.2f}"
              f"{fires:>6}{exp:>7.1f}{r_ct:>6.2f}")
    sdr, cntr, rate = np.array(sdr), np.array(cntr), np.array(rate)
    tor = np.array(tor)

    if args.firing_csv:
        fr = pd.DataFrame({
            "factor_name": FACTOR_TAXONOMY,
            "positive_examples": [int(Y[:, j].sum()) for j in range(N)],
            "gold_prevalence_percent": [round(100 * float(rate[j]), 3)
                                        for j in range(N)],
            "oof_firing_percent": [round(100 * float(
                (sig(ZO[:, j]) >= th[j]).mean()), 3) for j in range(N)],
            "test_firing_percent": [round(100 * float(
                (sig(ZT[:, j]) >= th[j]).mean()), 3) for j in range(N)],
            "firing_ratio": [round(float(tor[j]), 4) for j in range(N)],
            "threshold": [round(float(th[j]), 3) for j in range(N)],
        }).sort_values("positive_examples")
        os.makedirs(os.path.dirname(args.firing_csv) or ".", exist_ok=True)
        fr.to_csv(args.firing_csv, index=False)
        print(f"\nwrote {args.firing_csv}  ({len(fr)} labels)")
        print("  columns for the figure: factor_name, "
              "gold_prevalence_percent, firing_ratio\n")
        print(fr[["factor_name", "positive_examples",
                  "gold_prevalence_percent", "firing_ratio"]]
              .to_string(index=False))
    print(f"\n  mean sd ratio (test/oof) {sdr.mean():.3f}")
    print(f"  mean test/oof FIRING RATE at the identical threshold "
          f"{tor.mean():.3f}")
    print(f"    labels firing <0.8x on test vs oof: "
          f"{int((tor < 0.8).sum())} of {N}")
    print("\n  't/o' near 1.0 everywhere => NO transfer problem. The gap to")
    print("  prevalence is then tfit's own choice of operating point, which is")
    print("  legitimate, and rescaling has nothing to fix.")
    print("  't/o' well below 1 on the under-firing labels => the SAME rule")
    print("  behaves differently on the two pools, and that IS fixable.")

    print("\n" + "=" * 104)
    print("3. THE CAUSAL TEST -- does compression predict the count deficit?")
    print("=" * 104)
    ok = np.isfinite(sdr) & np.isfinite(cntr) & (cntr > 0)
    r1 = float(np.corrcoef(sdr[ok], np.log(cntr[ok]))[0, 1])
    r2 = float(np.corrcoef(np.log(rate[ok]), np.log(cntr[ok]))[0, 1])
    print(f"  corr( sd_ratio , log fire/expected )  {r1:+.3f}")
    print(f"  corr( log prevalence , log fire/expected ) {r2:+.3f}")
    print("\n  The compression story predicts BOTH:")
    print("    sd_ratio positive  -- labels whose test scores are squashed")
    print("                          most should under-fire most")
    print("    prevalence positive -- rare labels (high thresholds) under-fire,")
    print("                          common labels (low thresholds) over-fire")
    print("  If the SECOND is clearly positive the miscalibration is real")
    print("  whatever causes it, and rescaling is worth building.")

    if args.build:
        mo, so = ZO.mean(0), ZO.std(0)
        mt, st = ZT.mean(0), ZT.std(0)
        ZT2 = mo[None, :] + (ZT - mt[None, :]) * (so / np.maximum(st, 1e-9))
        order = {r: i for i, r in enumerate(ship["row_id"].astype(str))}
        perm = [order[r] for r in tid]

        repro = int(((sig(ZT) >= th[None, :]).astype(np.int8)
                     != shipM[perm]).sum())
        print(f"\nPROVENANCE: tfit reconstruction vs {args.shipped} -- "
              f"{repro} of {nT*N} cells differ")
        if repro:
            raise SystemExit(
                "REFUSING to write -- the reconstruction does not match the "
                "shipped column, so the new file would mix a rebuild bug with "
                "the intended change.")

        on = (sig(ZT2) >= th[None, :]).astype(np.int8)
        out = ship.copy()
        col = list(out["factors"])
        moved = 0
        for i, r in enumerate(tid):
            labs = str([FACTOR_TAXONOMY[j] for j in range(N) if on[i, j]])
            k = order[r]
            if str(col[k]) != labs:
                moved += 1
            col[k] = labs
        out["factors"] = col
        os.makedirs(os.path.dirname(args.build) or ".", exist_ok=True)
        out.to_csv(args.build, index=False)
        print(f"\nrescaled: {on.sum()/nT:.2f} factors/post "
              f"(shipped {shipM.sum()/len(ship):.2f}, gold 2.92)")
        print(f"cells changed {int((on != shipM[perm]).sum())}   "
              f"posts changed {moved}")
        print(f"{'label':<40}{'was':>6}{'now':>6}{'expect':>8}")
        for j in range(N):
            print(f"{FACTOR_TAXONOMY[j]:<40}{int(shipM[:, j].sum()):>6}"
                  f"{int(on[:, j].sum()):>6}{Y[:, j].mean()*nT:>8.1f}")
        print(f"\nwrote {args.build}")
        print("Only the factors column changes -- the risk (S1) score must be "
              "unchanged.")


if __name__ == "__main__":
    main()
