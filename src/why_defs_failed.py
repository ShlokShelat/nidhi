"""
why_defs_failed.py

Diagnose why factor definitions derived from the training annotations score
worse than the organizers' published definitions.

WHAT IS BEING COMPARED. Definitions written by DeepSeek from 60 positive and 60
negative examples per label, per fold, cross-fitted, then scored against the
published-definition arm on the same OOF pool. The derived arm underperforms
overall, with the protective factors -- the intended target -- hit hardest,
and `meaning in life` -- the specific label the rewrite was built for -- also
worse. Run this script to print the exact per-label deltas on the current
artefacts (see also artifacts/why_defs_failed.csv).

THREE CANDIDATE MECHANISMS, all measurable from artefacts already on disk.

  H1 NEGATION. The derived definitions carry clauses like "not merely
     expressing hopelessness or suicidal ideation". Cross-encoders handle
     negation badly -- they respond to a concept being PRESENT in the
     hypothesis, not to its scope. Naming `hopelessness` inside the
     `meaning in life` hypothesis may make the model fire MORE on hopelessness
     posts, the exact opposite of the intent. Predicts: labels whose
     definitions gained a negation clause lose PRECISION.

  H2 TRUNCATION. Published definitions are ~10 words, derived ~40. The
     tokeniser uses truncation="only_first", so the POST absorbs the extra
     length inside max_len=320. The derived arm literally sees less of each
     post. Predicts: loss correlates with definition length, and concentrates
     on long posts.

  H3 CROSS-LABEL CONTAMINATION. A definition that names another factor makes
     two hypotheses lexically similar, blurring labels the CONFUSABLE hard
     negatives were specifically designed to separate. Predicts: loss
     correlates with how many other factor names a definition contains.

These make different predictions, so the data can separate them. If it is H1 or
H3, the fix is cheap -- strip the negative clauses and rerun. If it is H2, raise
max_len. If none of them correlate, the derived definitions are simply worse
descriptions and the direction closes.

Usage:
  python src/why_defs_failed.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors

PROT = {"social support", "coping strategy", "psychological capital",
        "sense of responsibility", "meaning in life"}
NEG_MARK = re.compile(r"\b(not merely|not mere|but not|; not|, not\b|rather than|"
                      r"excluding|does not count)", re.I)


def load_oof(pattern, k=5):
    r, P, T = [], [], []
    for f in range(k):
        p = pattern.format(f=f)
        if not os.path.exists(p):
            return None, None, None
        d = np.load(p, allow_pickle=True)
        r.extend(list(d["row_id"]))
        P.append(d["probs"])
        T.append(np.tile(d["thresholds"], (len(d["row_id"]), 1)))
    return r, np.concatenate(P), np.concatenate(T)


def prf(P, Y, j):
    tp = int((P[:, j] & Y[:, j]).sum())
    if tp == 0:
        return 0.0, 0.0, 0.0
    fp = int((P[:, j] & (1 - Y[:, j])).sum())
    fn = int(((1 - P[:, j]) & Y[:, j]).sum())
    pr, rc = tp / (tp + fp), tp / (tp + fn)
    return 2 * pr * rc / (pr + rc), pr, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--defs", default="artifacts/derived_defs")
    ap.add_argument("--derived-oof",
                    default="artifacts/factor_runs/defs_cv_oof_f{f}.npz")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    os.environ.pop("FACTOR_DEFS_JSON", None)
    from train_factors_nli import FACTOR_DEFINITIONS
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

    lab = pd.read_csv(args.labeled).set_index("row_id")
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}

    ir, IP, ITH = load_oof("artifacts/fac_oof_f{f}.npz", args.k)
    dr, DP, DTH = load_oof(args.derived_oof, args.k)
    if dr is None:
        raise SystemExit("derived OOF not found")
    o = {r: i for i, r in enumerate(dr)}
    DP = DP[[o[r] for r in ir]]
    DTH = DTH[[o[r] for r in ir]]

    Y = np.zeros(IP.shape, dtype=int)
    for i, r in enumerate(ir):
        for f in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1
    PI = (IP >= ITH).astype(int)
    PD = (DP >= DTH).astype(int)

    # definition properties, averaged over the five folds
    defs = [json.load(open(os.path.join(args.defs, f"defs_f{f}.json")))
            for f in range(args.k)]
    other_names = {t: [x for x in FACTOR_TAXONOMY if x != t and len(x) > 8]
                   for t in FACTOR_TAXONOMY}

    rows = []
    for j, t in enumerate(FACTOR_TAXONOMY):
        pub_tok = len(tok(FACTOR_DEFINITIONS[t])["input_ids"])
        dtoks = np.mean([len(tok(d[t])["input_ids"]) for d in defs])
        has_neg = np.mean([1.0 if NEG_MARK.search(d[t]) else 0.0 for d in defs])
        # how many OTHER factor names or their distinctive head words appear
        n_other = np.mean([
            sum(1 for x in other_names[t]
                if x.split()[0].lower() in d[t].lower()
                or x.lower() in d[t].lower())
            for d in defs])
        f_i, p_i, r_i = prf(PI, Y, j)
        f_d, p_d, r_d = prf(PD, Y, j)
        rows.append(dict(factor=t, n=int(Y[:, j].sum()), prot=t in PROT,
                         pub_tok=pub_tok, der_tok=dtoks,
                         extra_tok=dtoks - pub_tok,
                         has_neg=has_neg, n_other=n_other,
                         dF1=f_d - f_i, dP=p_d - p_i, dR=r_d - r_i))
    df = pd.DataFrame(rows)

    print(f"{'factor':<44}{'toks':>6}{'+tok':>6}{'neg':>5}{'oth':>5}"
          f"{'dF1':>8}{'dPrec':>8}{'dRec':>8}")
    print("-" * 92)
    for _, x in df.sort_values("dF1").iterrows():
        print(f"{x.factor:<44}{x.der_tok:>6.0f}{x.extra_tok:>6.0f}"
              f"{x.has_neg:>5.1f}{x.n_other:>5.1f}"
              f"{x.dF1:>+8.3f}{x.dP:>+8.3f}{x.dR:>+8.3f}"
              f"{'  prot' if x.prot else ''}")

    print("\n" + "=" * 78)
    print("H1  NEGATION -- does a 'not merely ...' clause predict the loss?")
    print("=" * 78)
    for lo, hi, name in [(0.0, 0.5, "no negation clause"),
                         (0.5, 1.01, "has negation clause")]:
        s = df[(df.has_neg >= lo) & (df.has_neg < hi)]
        if len(s):
            print(f"  {name:<24} n={len(s):>2}   dF1 {s.dF1.mean():+.4f}   "
                  f"dPrec {s.dP.mean():+.4f}   dRec {s.dR.mean():+.4f}")
    print("  H1 predicts: negation clause -> PRECISION falls (the model fires")
    print("  on the very concept the clause was excluding).")

    print("\n" + "=" * 78)
    print("H2  TRUNCATION -- does definition length predict the loss?")
    print("=" * 78)
    r = np.corrcoef(df.extra_tok, df.dF1)[0, 1]
    print(f"  correlation(extra tokens, dF1) = {r:+.3f}")
    for lo, hi, name in [(0, 25, "under +25 tokens"), (25, 40, "+25 to +40"),
                         (40, 999, "over +40 tokens")]:
        s = df[(df.extra_tok >= lo) & (df.extra_tok < hi)]
        if len(s):
            print(f"  {name:<24} n={len(s):>2}   dF1 {s.dF1.mean():+.4f}")
    print("  H2 predicts a clear negative correlation. Note every definition")
    print("  got longer, so length alone cannot explain WHICH labels suffered.")

    print("\n" + "=" * 78)
    print("H3  CROSS-LABEL CONTAMINATION -- does naming other factors hurt?")
    print("=" * 78)
    r = np.corrcoef(df.n_other, df.dF1)[0, 1]
    print(f"  correlation(other factor names, dF1) = {r:+.3f}")
    for lo, hi, name in [(0, 0.5, "names none"), (0.5, 1.5, "names ~1"),
                         (1.5, 99, "names 2+")]:
        s = df[(df.n_other >= lo) & (df.n_other < hi)]
        if len(s):
            print(f"  {name:<24} n={len(s):>2}   dF1 {s.dF1.mean():+.4f}   "
                  f"dPrec {s.dP.mean():+.4f}")

    print("\n" + "=" * 78)
    print("PRECISION vs RECALL OVERALL")
    print("=" * 78)
    print(f"  mean dPrecision {df.dP.mean():+.4f}    "
          f"mean dRecall {df.dR.mean():+.4f}")
    print("  Precision down, recall up  -> the hypotheses became broader /")
    print("                                negation backfired")
    print("  Both down                  -> the definitions are simply worse")
    print("  Recall down, precision up  -> they became too narrow")

    df.to_csv("artifacts/why_defs_failed.csv", index=False)
    print("\nwrote artifacts/why_defs_failed.csv")
    print("\nIf H1 or H3 carries it, the fix is cheap: regenerate with the")
    print("negative clauses forbidden, keeping everything else identical.")


if __name__ == "__main__":
    main()
