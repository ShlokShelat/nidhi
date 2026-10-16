# Team NIDHI multi-task pipeline: one shared encoder with risk, evidence and
# factor heads. Vendored into this repository so the `friendfac` factor arm can
# be rebuilt from source without the original project tree.
#
# This stage also writes outputs/risk_probs_val.csv, the validation-side second
# risk member that src/ensemble_risk.py consumes as --friend-probs.
"""Stage 3 - tune decision thresholds on OUT-OF-FOLD predictions and print
the honest cross-validated composite score using the official metric rules.

Catches handled here:
  * evidence token threshold tuned against the real Phrase-F1 (containment,
    3x cap, one-to-one) - not against token accuracy
  * one threshold PER FACTOR, tuned for that factor's F1 -> directly optimizes
    Macro F1, which is where rare factors would otherwise bleed points
  * all tuning happens on posts the model never trained on (OOF), grouped by
    user, so this estimate is honest w.r.t. the unseen leaderboard users
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import config as C
import postprocess as PP
from scorer import composite, evidence_phrase_f1


def run():
    df = pd.read_csv(os.path.join(C.OUT_DIR, "train_clean.csv"))
    oof = np.load(os.path.join(C.OUT_DIR, "oof.npy"), allow_pickle=True).item()
    lex = PP.load_lexicon()

    # stitch OOF back together in dataframe order
    by_row = {}
    for fold, o in oof.items():
        for i, rid in enumerate(o["row_id"]):
            by_row[rid] = (o["risk"][i], o["fact"][i], o["tokp"][i],
                           o["offsets"][i], o["post"][i])
    df = df[df["row_id"].isin(by_row)].reset_index(drop=True)
    risk_p = np.stack([by_row[r][0] for r in df["row_id"]])
    prob_df = pd.DataFrame(risk_p, columns=C.RISK_LEVELS)
    prob_df.insert(0, "row_id", df["row_id"].values)
    prob_df.to_csv(os.path.join(C.OUT_DIR, "risk_probs_val.csv"), index=False)
    print("[stage3] wrote risk_probs_val.csv")
    fact_p = np.stack([by_row[r][1] for r in df["row_id"]])

    gold_lv = df["risk_level"].tolist()
    gold_ev = [json.loads(x) for x in df["evidence_clean"]]
    gold_fc = [json.loads(x) for x in df["factors_clean"]]
    pred_lv = [C.RISK_LEVELS[i] for i in risk_p.argmax(1)]

    # ---- tune evidence threshold on real Phrase F1 ----
    best_thr, best = None, -1
    for thr in C.EVID_THRESHOLD_GRID:
        pred_ev = []
        for r, rid in enumerate(df["row_id"]):
            _, _, tokp, offs, post = by_row[rid]
            spans = PP.decode_spans(post, offs, tokp, thr)
            _, spans = PP.consistency_rules(post, pred_lv[r], spans, lex)
            pred_ev.append(spans)
        f1 = evidence_phrase_f1(gold_ev, pred_ev)
        print(f"[stage3] evidence thr={thr:.2f} -> Phrase F1 {f1:.4f}")
        if f1 > best:
            best, best_thr, best_ev = f1, thr, pred_ev

    # ---- tune per-factor thresholds for Macro F1 ----
    Yg = np.zeros((len(df), len(C.FACTORS)), dtype=int)
    for r, lst in enumerate(gold_fc):
        for f in lst:
            Yg[r, C.FACTORS.index(f)] = 1
    fac_thr = {}
    for k, f in enumerate(C.FACTORS):
        bt, bf = 0.5, -1
        for t in C.FACTOR_THRESHOLD_GRID:
            s = f1_score(Yg[:, k], (fact_p[:, k] >= t).astype(int), zero_division=0)
            if s > bf:
                bf, bt = s, t
        fac_thr[f] = bt
    pred_fc = []
    for r in range(len(df)):
        fs = [f for k, f in enumerate(C.FACTORS) if fact_p[r, k] >= fac_thr[f]]
        fs = PP.apply_factor_rules(df["post"][r], fs)
        pred_fc.append(fs)

    print("\n[stage3] ===== HONEST CROSS-VALIDATED SCORE (official rules) =====")
    composite(gold_lv, pred_lv, gold_ev, best_ev, gold_fc, pred_fc)

    json.dump({"evid_threshold": best_thr, "factor_thresholds": fac_thr},
              open(os.path.join(C.OUT_DIR, "thresholds.json"), "w"), indent=1)
    print(f"[stage3] saved thresholds.json (evidence thr={best_thr})")


if __name__ == "__main__":
    run()
