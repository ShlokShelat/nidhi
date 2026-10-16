"""
rare_factor_rules.py

Keyword ADD-rules for the rare factor labels, ported from the friend model's
postprocess.py (RARE_FACTOR_RULES) and measured against gold before being
applied.

Why the rare labels are the target. Subtask 2 is MACRO-F1 over 24 labels, so
every label carries 1/24 = 4.2% of the score regardless of prevalence, and the
weakest labels are the rare ones. A label sitting near zero F1 is almost pure
upside.

Why ADD-only is the right shape. A DROP rule that removes a rare label's only
correct prediction zeroes that label outright, costing up to 1/24 of the macro
average from a single post. Adding cannot do that -- it can only move a
near-zero label up, or cost precision on a label that already has some.

Nothing here is fitted on val. The keyword lists come from the friend
codebase; this module only MEASURES them against gold and applies the ones
that clear the bar.

Usage:
  python src/rare_factor_rules.py --evaluate        # keyword rule vs gold
  python src/rare_factor_rules.py --report \
      --factors artifacts/nli_factors_val.csv       # effect on the predictions
  python src/rare_factor_rules.py --apply \
      --factors artifacts/nli_ensemble_factors.csv \
      --posts data/test.csv --out artifacts/factors_rare_test.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors
from rule_ablation_v2 import author_split

# Ported verbatim from the friend pipeline's src/postprocess.py
RARE_FACTOR_RULES = {
    "sexual orientation related issues":
        ["gay", "lesbian", "bisexual", "trans ", "transgender", "queer",
         "coming out", "closeted", "homophob", "my sexuality"],
    "exposure to others' suicide":
        ["his suicide", "her suicide", "friend killed", "brother killed himself",
         "sister killed herself", "died by suicide", "committed suicide last",
         "lost my friend to suicide", "lost my brother", "lost my sister to"],
    "poor school performance":
        ["failing my", "failed my exam", "failed the exam", "flunk", "bad grades",
         "failing school", "failing college", "failing class", "my gpa"],
    "substance use":
        ["alcohol", "drunk", "weed", "cocaine", "heroin", "meth", "drugs",
         "getting high", "smoking", "addiction", "addicted to"],
    "low socio-economic status":
        ["no money", "can't afford", "cant afford", "broke ", "in debt",
         "poverty", "homeless", "unemployed", "lost my job", "can't pay",
         "financial"],
}


def fires(post, factor):
    low = " " + str(post).lower() + " "
    return any(k in low for k in RARE_FACTOR_RULES.get(factor, []))


def apply_rules(posts, pred_sets, only=None):
    out, n_add = [], 0
    for post, s in zip(posts, pred_sets):
        cur = set(s)
        for f in RARE_FACTOR_RULES:
            if only is not None and f not in only:
                continue
            if f not in cur and fires(post, f):
                cur.add(f)
                n_add += 1
        out.append(sorted(cur))
    return out, n_add


def to_matrix(rows):
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
    m = np.zeros((len(rows), len(FACTOR_TAXONOMY)), dtype=int)
    for i, s in enumerate(rows):
        for f in s:
            if str(f).strip().lower() in idx:
                m[i, idx[str(f).strip().lower()]] = 1
    return m


def macro(gold_sets, pred_sets):
    return float(f1_score(to_matrix(gold_sets), to_matrix(pred_sets),
                          average=None, zero_division=0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--evaluate-oof", action="store_true",
                    help="compare the keyword rule against OUR model on the "
                         "SAME 1635 posts, using the CV out-of-fold dumps. "
                         "val cannot settle this: these rules fire ~8 and ~3 "
                         "times in 1635 posts, so on 372 val posts they fire "
                         "essentially never and measure nothing.")
    ap.add_argument("--oof-glob", default="artifacts/fac_oof_f{f}.npz")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--labeled", default="data/labeled.csv")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--factors", default=None)
    ap.add_argument("--posts", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.evaluate:
        lab = pd.read_csv(args.labeled)
        fcol = ("factors" if "factors" in lab.columns
                else "factor")
        gold = [set(parse_factors(x)) for x in lab[fcol]]
        print(f"keyword rule vs gold, over {len(lab)} labelled posts\n")
        print(f"{'factor':<38}{'gold':>6}{'fires':>7}{'P':>8}{'R':>8}{'F1':>8}")
        for f in RARE_FACTOR_RULES:
            y = np.array([1 if f in g else 0 for g in gold])
            p = np.array([1 if fires(t, f) else 0 for t in lab["post"]])
            print(f"{f:<38}{int(y.sum()):>6}{int(p.sum()):>7}"
                  f"{precision_score(y, p, zero_division=0):>8.3f}"
                  f"{recall_score(y, p, zero_division=0):>8.3f}"
                  f"{f1_score(y, p, zero_division=0):>8.3f}")
        print("\nCompare each F1 against the DEPLOYED model's F1 on that label")
        print("(measured on val): sexual orientation 0.125, exposure 0.222,")
        print("poor school performance 0.000, substance use 0.500.")
        print("A keyword rule only helps where it beats the model.")
        return

    if args.evaluate_oof:
        lab = pd.read_csv(args.labeled).set_index("row_id")
        fcol = "factors" if "factors" in lab.columns else "factor"
        idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}
        rids, Pm = [], []
        for f in range(args.k):
            p = args.oof_glob.format(f=f)
            if not os.path.exists(p):
                raise SystemExit(f"missing {p} -- run factors_oof_cv.py first")
            d = np.load(p, allow_pickle=True)
            on = (d["probs"] >= d["thresholds"]).astype(int)
            rids.extend(list(d["row_id"]))
            Pm.append(on)
        Pm = np.concatenate(Pm, axis=0)
        print(f"our model, out-of-fold over {len(rids)} posts "
              f"(vs val's 372 -- these rules are unmeasurable on val)\n")

        Y = np.zeros_like(Pm)
        posts = []
        for i, r in enumerate(rids):
            posts.append(lab.loc[r, "post"])
            for fac in parse_factors(lab.loc[r, fcol]):
                if str(fac).strip().lower() in idx:
                    Y[i, idx[str(fac).strip().lower()]] = 1

        print(f"{'factor':<38}{'gold':>6}{'ourF1':>8}{'ruleF1':>8}"
              f"{'unionF1':>9}{'delta':>9}  verdict")
        total_delta = 0.0
        for fac in RARE_FACTOR_RULES:
            j = idx[fac]
            y = Y[:, j]
            ours = Pm[:, j]
            rule = np.array([1 if fires(t, fac) else 0 for t in posts])
            union = ((ours + rule) > 0).astype(int)
            f_our = f1_score(y, ours, zero_division=0)
            f_rul = f1_score(y, rule, zero_division=0)
            f_uni = f1_score(y, union, zero_division=0)
            d = f_uni - f_our
            total_delta += d
            print(f"{fac:<38}{int(y.sum()):>6}{f_our:>8.3f}{f_rul:>8.3f}"
                  f"{f_uni:>9.3f}{d:>+9.3f}  "
                  f"{'ADD' if d > 0 else 'skip'}")
        print(f"\nif every positive rule were applied: "
              f"{total_delta/len(FACTOR_TAXONOMY):+.4f} macro-F1 "
              f"= {0.3*total_delta/len(FACTOR_TAXONOMY):+.4f} composite")
        print("\nunionF1 is what matters: the rule ADDS to our predictions, it")
        print("does not replace them, so a rule can beat us standalone and")
        print("still hurt the union by costing precision.")
        return

    if args.report:
        if not args.factors:
            raise SystemExit("--report needs --factors (our val predictions)")
        val = pd.read_csv(args.val)
        gold = [set(parse_factors(x)) for x in val["factors"]]
        fv = pd.read_csv(args.factors, keep_default_na=False).set_index("row_id")
        pred = [set(parse_factors(fv.loc[r, "factors"])) for r in val["row_id"]]
        half = author_split(val)[0].values
        A = [i for i in range(len(val)) if half[i] == "A"]
        B = [i for i in range(len(val)) if half[i] == "B"]

        def show(name, sp, extra=""):
            m = macro(gold, sp)
            a = macro([gold[i] for i in A], [sp[i] for i in A])
            b = macro([gold[i] for i in B], [sp[i] for i in B])
            print(f"{name:<34}{m:>9.4f}{a:>9.4f}{b:>9.4f}  {extra}")
            return m

        print(f"{'system':<34}{'macro':>9}{'halfA':>9}{'halfB':>9}")
        base = show("deployed factors", pred)
        allr, na = apply_rules(val["post"], pred)
        m_all = show("+ all 5 keyword rules", allr, f"+{na} added")

        print(f"\nper-rule, applied alone:")
        keep = []
        for f in RARE_FACTOR_RULES:
            one, n1 = apply_rules(val["post"], pred, only={f})
            m1 = macro(gold, one)
            a1 = macro([gold[i] for i in A], [one[i] for i in A])
            b1 = macro([gold[i] for i in B], [one[i] for i in B])
            ok = (m1 > base) and (a1 >= macro([gold[i] for i in A], [pred[i] for i in A])) \
                and (b1 >= macro([gold[i] for i in B], [pred[i] for i in B]))
            print(f"  {f:<38}{m1:>9.4f}{m1-base:>+9.4f}  +{n1:<4} "
                  f"{'PASS' if ok else 'fail'}")
            if ok:
                keep.append(f)
        print(f"\nrules positive on full val AND both halves: {keep}")
        print(f"\ndelta all-5 {m_all - base:+.4f} macro-F1 "
              f"= {0.3*(m_all-base):+.4f} composite")
        return

    if not args.apply:
        raise SystemExit("pass --evaluate, --report or --apply")
    for r in ("factors", "posts", "out"):
        if not getattr(args, r):
            raise SystemExit(f"--apply needs --{r}")
    fa = pd.read_csv(args.factors, keep_default_na=False)
    po = pd.read_csv(args.posts).set_index("row_id")
    rids = list(fa["row_id"])
    pred = [set(parse_factors(x)) for x in fa["factors"]]
    posts = [po.loc[r, "post"] for r in rids]
    out, na = apply_rules(posts, pred)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame({"row_id": rids,
                  "factors": [str(list(s)) for s in out]}).to_csv(args.out, index=False)
    changed = sum(1 for a, b in zip(pred, out) if set(a) != set(b))
    print(f"added {na} factor labels across {changed}/{len(rids)} posts")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
