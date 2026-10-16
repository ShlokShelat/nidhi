"""
Honest local validation of the cross-model ensemble.

Every blend weight submitted so far (risk 0.4/0.5, factors 0.7/0.5) was not
fitted offline and was confirmed only through the official score — the one
lever in this campaign that had never been validated locally first.

This script fixes that. The friend's oof.npy covers all 1635 labeled
posts via honest user-grouped 5-fold CV. val_split.csv (372 posts) is
a subset of those same posts, held out from the local models too. So
filtering the friend's OOF export to the val_split row_ids gives BOTH
models' honest, gold-backed predictions on an IDENTICAL set of posts —
no official submission needed to compare them.

What this does:
  1. Generate the LOCAL model's predictions (risk probs + factor probs) on
     val_split.
  2. Load their OOF export, filter to val_split row_ids.
  3. Compute each model's per-label F1 against real gold.
  4. Grid-search the blend weight (global, and per-label) that maximizes
     local weighted-F1 (risk) / macro-F1 (factors).
  5. Report the recommended weights — validated, not guessed.

Usage:
  python src/validate_ensemble.py \
      --val artifacts/val_split.csv \
      --our-risk-run runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15 \
      --our-factors-run runs/factors_cv_final \
      --friend-oof artifacts/friend_oof_export.csv
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from data_audit import load
from scoring import RISK_LEVELS, FACTOR_TAXONOMY, parse_factors, factors_macro_f1
from predict import predict_risk, _factor_probs

N_FAC = len(FACTOR_TAXONOMY)
RISK_COLS = list(RISK_LEVELS)  # internal names (lowercase, positional order)
# their export uses official capitalized names in the SAME clinical order
FRIEND_RISK_COLS = ["Indicator", "Ideation", "Behavior", "Attempt"]


def to_lists(binary):
    return [[FACTOR_TAXONOMY[j] for j in range(N_FAC) if row[j]] for row in binary]


def gold_factor_matrix(df):
    Y = np.zeros((len(df), N_FAC), dtype=int)
    for r, cell in enumerate(df["factors"]):
        for f in parse_factors(cell):
            if f in FACTOR_TAXONOMY:
                Y[r, FACTOR_TAXONOMY.index(f)] = 1
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--our-risk-run", required=True)
    ap.add_argument("--our-factors-run", required=True)
    ap.add_argument("--friend-oof", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load(args.val)
    ids = df["row_id"].tolist()
    gold_risk = df["risk_level"].str.strip().str.lower().tolist()
    gold_fac = gold_factor_matrix(df)

    print(f"val_split: {len(df)} posts\n")

    # ---- local predictions on val_split (honest: neither model trained on it) ----
    print("computing LOCAL risk probs on val_split ...")
    _, our_risk_p = predict_risk(args.our_risk_run, df, device)
    print("computing LOCAL factor probs on val_split ...")
    our_fac_p = _factor_probs(args.our_factors_run, df, device)
    fac_ths = np.load(os.path.join(args.our_factors_run, "thresholds.npy"))

    # ---- friend OOF, filtered to val_split row_ids ----
    foof = pd.read_csv(args.friend_oof).set_index("row_id")
    missing = [rid for rid in ids if rid not in foof.index]
    if missing:
        print(f"WARNING: {len(missing)} val_split posts missing from friend OOF "
              f"(likely a fold-count mismatch) — those rows fall back to the "
              f"LOCAL model's preds")
    friend_risk_p = np.zeros_like(our_risk_p)
    friend_fac_p = np.zeros_like(our_fac_p)
    for i, rid in enumerate(ids):
        if rid in foof.index:
            friend_risk_p[i] = foof.loc[rid, FRIEND_RISK_COLS].values.astype(float)
            friend_fac_p[i] = foof.loc[rid, list(FACTOR_TAXONOMY)].values.astype(float)
        else:
            friend_risk_p[i] = our_risk_p[i]
            friend_fac_p[i] = our_fac_p[i]

    # ================= RISK =================
    print("\n" + "=" * 50)
    print("RISK — weighted-F1 on val_split (honest, gold-backed)")
    print("=" * 50)
    our_pred = [RISK_LEVELS[k] for k in our_risk_p.argmax(1)]
    fr_pred = [RISK_LEVELS[k] for k in friend_risk_p.argmax(1)]
    our_f1 = f1_score(gold_risk, our_pred, average="weighted", zero_division=0)
    fr_f1 = f1_score(gold_risk, fr_pred, average="weighted", zero_division=0)
    print(f"  LOCAL model alone : {our_f1:.4f}")
    print(f"  FRIEND model alone: {fr_f1:.4f}")

    best_w, best_f1 = 1.0, our_f1
    for w in np.arange(0.0, 1.01, 0.05):
        blend = w * our_risk_p + (1 - w) * friend_risk_p
        pred = [RISK_LEVELS[k] for k in blend.argmax(1)]
        f1 = f1_score(gold_risk, pred, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1, best_w = f1, w
    print(f"  BEST BLEND: w_ours={best_w:.2f} -> weighted-F1={best_f1:.4f} "
          f"(vs ours-alone {our_f1:.4f}, delta {best_f1-our_f1:+.4f})")

    # ================= FACTORS =================
    print("\n" + "=" * 50)
    print("FACTORS — macro-F1 on val_split (honest, gold-backed)")
    print("=" * 50)
    our_on = our_fac_p >= fac_ths
    our_macro, our_solo_per = factors_macro_f1(
        to_lists(gold_fac), to_lists(our_on.astype(int)), return_per_label=True)
    fr_on = friend_fac_p >= 0.5  # they don't share the tuned thresholds; 0.5 baseline
    fr_macro, fr_solo_per = factors_macro_f1(
        to_lists(gold_fac), to_lists(fr_on.astype(int)), return_per_label=True)
    print(f"  LOCAL model alone (tuned ths) : {our_macro:.4f}")
    print(f"  FRIEND model alone (th=0.5)   : {fr_macro:.4f}")

    our_f1_vec = [our_solo_per[lab] for lab in FACTOR_TAXONOMY]
    fr_f1_vec = [fr_solo_per[lab] for lab in FACTOR_TAXONOMY]
    print("\n  --our-f1 (paste into ensemble_factors.py):")
    print("  " + ",".join(f"{v:.4f}" for v in our_f1_vec))
    print("  --friend-f1 (paste into ensemble_factors.py):")
    print("  " + ",".join(f"{v:.4f}" for v in fr_f1_vec))

    best_fw, best_fmacro = 1.0, our_macro
    sweep = {}
    for w in np.arange(0.0, 1.01, 0.05):
        blend = w * our_fac_p + (1 - w) * friend_fac_p
        on = blend >= fac_ths  # keep the LOCAL model's tuned thresholds as the operating point
        macro = factors_macro_f1(to_lists(gold_fac), to_lists(on.astype(int)))
        sweep[round(w, 2)] = macro
        if macro > best_fmacro:
            best_fmacro, best_fw = macro, w
    print(f"  BEST BLEND: w_ours={best_fw:.2f} -> macro-F1={best_fmacro:.4f} "
          f"(vs ours-alone {our_macro:.4f}, delta {best_fmacro-our_macro:+.4f})")
    print("\n  full sweep (w_ours -> macro-F1):")
    for w in sorted(sweep):
        marker = "  <-- best" if abs(w - best_fw) < 1e-6 else ""
        print(f"    {w:.2f}: {sweep[w]:.4f}{marker}")

    # per-label: where does friend actually help?
    print("\n  per-label F1 at BEST blend vs ours-alone (sorted by biggest gain):")
    blend = best_fw * our_fac_p + (1 - best_fw) * friend_fac_p
    on_best = blend >= fac_ths
    _, our_per = factors_macro_f1(to_lists(gold_fac), to_lists(our_on.astype(int)),
                                  return_per_label=True)
    _, best_per = factors_macro_f1(to_lists(gold_fac), to_lists(on_best.astype(int)),
                                   return_per_label=True)
    for lab in sorted(FACTOR_TAXONOMY, key=lambda L: best_per[L] - our_per[L], reverse=True)[:8]:
        print(f"    {our_per[lab]:.3f} -> {best_per[lab]:.3f} "
              f"({best_per[lab]-our_per[lab]:+.3f})  {lab}")

    print("\n" + "=" * 50)
    print("RECOMMENDATION FOR NEXT SUBMISSION (validated, not guessed):")
    print(f"  risk:    ensemble_risk.py --weight-ours {best_w:.2f}")
    print(f"  factors: ensemble_factors.py --weight-ours {best_fw:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
