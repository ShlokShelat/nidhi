"""
risk_from_evidence.py

THE RISK/EVIDENCE COUPLING, RUN BACKWARDS.

Gold evidence is empty EXACTLY when gold risk is indicator (0.986 agreement,
measured). The pipeline uses that identity in one direction only: a predicted
indicator suppresses spans. This script tests the other direction.

THE SIGNAL. If several independently-trained span taggers read a post and none
of them finds anything quotable, that is evidence the post IS indicator. The
signal comes from a different modality than any risk model -- span extraction
trained on span labels, not risk labels -- so its errors are not correlated
with the risk ensemble's errors, which is the condition under which an
override can add anything at all.

That condition is the whole point. An override built from other RISK models
cannot help where the risk ensemble is wrong, because those challengers tend
to be wrong on the same posts.

WHAT IT MEASURES
  1. how sharp the signal is -- gold risk distribution by how many taggers fire
  2. the rule `0 taggers fire -> indicator` as an override on the encoder OOF,
     scored as weighted F1 with a paired AUTHOR bootstrap
  3. the same at looser thresholds (<=1 tagger, <=2), because 0 may be too rare
  4. the reverse rule too: MANY taggers firing on a predicted-indicator post
     means the suppression is about to throw away real spans AND the risk label
     is probably wrong

ZERO FITTED PARAMETERS in form (1) and (2). The tagger-count cut is read off the
crosstab, not tuned against the score.

SCOPE CAVEAT: these are the SEVEN `artifacts/evidence_oof` taggers, not the
eight `_rc` taggers in the shipped configuration. Tagger-count rules are
sensitive to which pool produced the count, so a positive result here must be
re-checked against artifacts/evidence_runs/*_rc_test_candidates.csv before it
is used.

Usage:
  python src/risk_from_evidence.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1  # noqa: E402

TAGS = ["v2", "ep12", "lr3", "bs16", "ml256", "friend", "elec"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/evidence_oof")
    ap.add_argument("--tags", default=",".join(TAGS))
    ap.add_argument("--enc", default="artifacts/nligaur/oof_base.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")

    enc = pd.read_csv(args.enc, keep_default_na=False)
    enc["row_id"] = enc["row_id"].astype(str)
    enc = enc.drop_duplicates("row_id").set_index("row_id")
    col = next((c for c in ("pred_risk_level", "risk_level")
                if c in enc.columns), None)
    pc = [f"prob_{c}" for c in RISK_LEVELS]
    if col is not None:
        pred_all = {r: normalize_label(enc.loc[r, col]) for r in enc.index}
    elif all(c in enc.columns for c in pc):
        M = enc[pc].to_numpy(float)
        pred_all = {r: RISK_LEVELS[i] for r, i in zip(enc.index, M.argmax(1))}
    else:
        raise SystemExit(f"{args.enc}: no risk labels or prob_* columns")

    rows = [r for r in enc.index if r in lab.index]
    gold = np.array([normalize_label(lab.loc[r, "risk_level"]) for r in rows])
    pred = np.array([pred_all[r] for r in rows])
    auth = np.array([str(lab.loc[r, "anon_user_id"]) for r in rows])
    n = len(rows)
    pos = {r: i for i, r in enumerate(rows)}

    # how many taggers produce at least one candidate for each post
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    fired = np.zeros((n, len(tags)), dtype=np.int8)
    used = []
    for ti, t in enumerate(tags):
        p = f"{args.dir}/{t}_oof_candidates.csv"
        if not os.path.exists(p):
            print(f"  {t}: no candidates file, skipping")
            continue
        used.append(t)
        d = pd.read_csv(p, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        # honour each tagger's own confidence floor, as the vote does
        if "min_conf" in d.columns and "conf" in d.columns:
            mc = float(d["min_conf"].iloc[0]) if len(d) else 0.0
            d = d[d["conf"] >= mc]
        for r in d["row_id"].unique():
            if r in pos:
                fired[pos[r], ti] = 1
    k = fired.sum(1)
    T = len(used)
    print(f"{n} posts / {pd.unique(auth).size} authors   "
          f"{T} taggers usable: {used}\n")

    print("=" * 88)
    print("1. GOLD RISK BY HOW MANY TAGGERS FOUND ANYTHING")
    print("=" * 88)
    print(f"{'taggers firing':<16}{'posts':>7}" +
          "".join(f"{c:>12}" for c in RISK_LEVELS) + f"{'%indicator':>12}")
    for kk in range(T + 1):
        m = k == kk
        if not m.any():
            continue
        cnt = [int((gold[m] == c).sum()) for c in RISK_LEVELS]
        print(f"{kk:<16}{int(m.sum()):>7}" +
              "".join(f"{v:>12}" for v in cnt) +
              f"{100*cnt[0]/max(1,int(m.sum())):>11.1f}%")
    print("\n  If '0 taggers' is overwhelmingly indicator, the signal is real")
    print("  and it is independent of the risk models.")

    # ---------------- 2/3. the override ------------------------------------
    uauth = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uauth}
    rng = np.random.default_rng(args.seed)
    IXS = [np.concatenate([by[uauth[i]] for i in
                           rng.choice(len(uauth), len(uauth), replace=True)])
           for _ in range(args.boot)]
    base = risk_weighted_f1(list(gold), list(pred))

    def score(p, ix):
        return risk_weighted_f1(list(gold[ix]), list(np.asarray(p)[ix]))

    print("\n" + "=" * 88)
    print("2. RULE: few taggers fire -> call it INDICATOR")
    print("=" * 88)
    print(f"  encoder baseline wF1 {base:.4f}")
    print(f"\n{'rule':<22}{'moved':>7}{'fix':>6}{'brk':>6}{'wF1':>9}"
          f"{'delta':>9}{'95% CI':>20}{'win':>6}")
    for cut in range(0, min(3, T)):
        new = pred.copy()
        m = (k <= cut) & (pred != "indicator")
        new[m] = "indicator"
        fix = int(((gold == "indicator") & m).sum())
        brk = int(((gold != "indicator") & m).sum())
        v = risk_weighted_f1(list(gold), list(new))
        d = np.array([score(new, ix) - score(pred, ix) for ix in IXS])
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{'<= ' + str(cut) + ' taggers':<22}{int(m.sum()):>7}{fix:>6}"
              f"{brk:>6}{v:>9.4f}{v-base:>+9.4f}"
              f"{f'[{lo:+.4f},{hi:+.4f}]':>20}{(d > 0).mean():>6.0%}")

    print("\n" + "=" * 88)
    print("4. REVERSE RULE: many taggers fire on a predicted-indicator post")
    print("=" * 88)
    print(f"{'rule':<22}{'moved':>7}{'fix':>6}{'brk':>6}{'wF1':>9}"
          f"{'delta':>9}{'95% CI':>20}{'win':>6}")
    # move it to the encoder's best NON-indicator class
    if all(c in enc.columns for c in pc):
        M = enc.loc[rows, pc].to_numpy(float)
        M2 = M.copy()
        M2[:, 0] = -1.0
        alt = np.array([RISK_LEVELS[i] for i in M2.argmax(1)])
    else:
        alt = np.array(["ideation"] * n)
    for cut in range(T, max(T - 3, 0), -1):
        new = pred.copy()
        m = (k >= cut) & (pred == "indicator")
        new[m] = alt[m]
        fix = int(((gold != "indicator") & m).sum())
        brk = int(((gold == "indicator") & m).sum())
        v = risk_weighted_f1(list(gold), list(new))
        d = np.array([score(new, ix) - score(pred, ix) for ix in IXS])
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{'>= ' + str(cut) + ' taggers':<22}{int(m.sum()):>7}{fix:>6}"
              f"{brk:>6}{v:>9.4f}{v-base:>+9.4f}"
              f"{f'[{lo:+.4f},{hi:+.4f}]':>20}{(d > 0).mean():>6.0%}")

    print("\n" + "=" * 88)
    print("HOW TO DECIDE")
    print("=" * 88)
    print("composite = 0.4 x the wF1 delta, PLUS an evidence knock-on: every")
    print("post moved INTO indicator also has its spans suppressed, so a")
    print("correct move gains on both components and a wrong one loses twice.")
    print("A rule therefore has to clear a bar set by that doubling, not by")
    print("the risk term alone.")
    print("")
    print("Adopt only a rule whose fix:brk ratio is well above 2:1 and whose")
    print("advantage holds across the bootstrap. An offline ratio can invert")
    print("out of sample when the challenger signal shares the base model's")
    print("blind spots; the premise here is that span taggers do not.")
    print("")
    print("These are the SEVEN OOF taggers, not the eight `_rc` ones in the")
    print("shipped configuration. Tagger-count rules are sensitive to the pool")
    print("that produced the count, so re-check any winner against")
    print("artifacts/evidence_runs/*_rc_test_candidates.csv first.")


if __name__ == "__main__":
    main()
