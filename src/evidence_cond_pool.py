"""
evidence_cond_pool.py

Evaluate risk-conditioned span tagging over pooled out-of-fold predictions.

Two measurement properties drive the design:

  1. MANY POSTS, NOT ONE FOLD. Single-fold evidence deltas on this data are
     within sampling noise, so all 5 folds are pooled into one honestly
     out-of-fold set before anything is compared.
  2. MACRO, NOT MICRO. train_evidence.py calls phrase_f1() at its default,
     which is MICRO. The official scorer is MACRO over posts, and the
     difference is not cosmetic: macro hands a free 1.0 to every post where
     gold and prediction are BOTH empty, which is precisely the indicator
     posts, which is precisely where the risk label acts. A micro win can be a
     macro loss.

Three numbers are reported per arm:

  micro                 comparable to the training logs
  macro RAW             the official average, single tagger, no rules
  macro + SUPPRESSION   the deployed consistency rule applied (predicted
                        indicator -> emit nothing), i.e. the shipping decode

The cond arm is scored twice -- on GOLD levels (the ceiling) and on PREDICTED
levels (shippable). The gap between those two is the part a better risk model
would hand back.

THRESHOLD HANDLING. Each run stores a min_conf chosen on its own val fold,
which is the fold being scored -- using it would be tuning on the data under
test. The headline uses one FIXED threshold applied identically to every arm;
the sweep is printed for information only, never to pick the winner.

Usage:
  python src/evidence_cond_pool.py
  python src/evidence_cond_pool.py --risk-from artifacts/nligaur/oof_fenced.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_label, parse_evidence, phrase_f1  # noqa: E402
from train_evidence import TaggerModel, collect_candidates, spans_at  # noqa: E402

SWEEP = [0.0, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]


def load_risk(path: str, rows: list[str], gold_lvl: dict) -> tuple[list, int]:
    if not os.path.exists(path):
        raise SystemExit(f"missing risk source {path}")
    d = pd.read_csv(path, keep_default_na=False)
    col = "pred_risk_level" if "pred_risk_level" in d.columns else "risk_level"
    m = {str(a): normalize_label(b) for a, b in zip(d["row_id"], d[col])}
    out, miss = [], 0
    for r in rows:
        if r in m:
            out.append(m[r])
        else:
            out.append(gold_lvl[r])
            miss += 1
    return out, miss


def score(preds: list[list[str]], gold: list[list[str]],
          risks: list[str] | None) -> tuple[float, float, float]:
    """(micro, macro raw, macro with the deployed suppression rule)."""
    mic = phrase_f1(gold, preds).f1
    mac = phrase_f1(gold, preds, average="macro").f1
    if risks is None:
        return mic, mac, float("nan")
    sup = [[] if lv == "indicator" else p for p, lv in zip(preds, risks)]
    return mic, mac, phrase_f1(gold, sup, average="macro").f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="base,cond")
    ap.add_argument("--prefix", default="runs/evcond5")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--risk-from", default="artifacts/nligaur/oof_base.csv",
                    help="OOF predicted risk levels; must cover the folds")
    ap.add_argument("--conf", type=float, default=0.70,
                    help="FIXED threshold for the headline, all arms alike")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    arms = [a.strip() for a in args.arms.split(",")]

    rows, posts, gold, gold_lvl = [], [], [], {}
    fold_of = []
    for f in range(args.folds):
        p = f"artifacts/folds/val_f{f}.csv"
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        va = pd.read_csv(p, keep_default_na=False)
        for rid, po, ev, rl in zip(va["row_id"].astype(str), va["post"],
                                   va["evidence"], va["risk_level"]):
            if rid in gold_lvl:
                continue
            rows.append(rid)
            posts.append(str(po))
            gold.append(parse_evidence(ev))
            gold_lvl[rid] = normalize_label(rl)
            fold_of.append(f)
    print(f"{len(rows)} pooled OOF posts over {args.folds} folds")
    n_empty = sum(1 for g in gold if not g)
    print(f"  gold-empty posts: {n_empty} ({100 * n_empty / len(rows):.1f}%) "
          f"-- these are free 1.0s under MACRO and invisible under micro")

    gl = [gold_lvl[r] for r in rows]
    pl, miss = load_risk(args.risk_from, rows, gold_lvl)
    print(f"  predicted risk from {args.risk_from}"
          + (f" ({miss} rows fell back to gold)" if miss else ""))

    # candidates per arm, decoded once, then thresholded many times
    cands = {}
    for tag in arms:
        for cond_name, risks in ((tag, None),) if tag == "base" else \
                (("cond@GOLD", gl), ("cond@PRED", pl)):
            key = cond_name
            allc = [None] * len(rows)
            for f in range(args.folds):
                d = f"{args.prefix}_{tag}_f{f}"
                mp = os.path.join(d, "model_final.pt")
                if not os.path.exists(mp):
                    raise SystemExit(f"missing {mp}")
                idx = [i for i, x in enumerate(fold_of) if x == f]
                mdl = TaggerModel(args.model)
                mdl.load_state_dict(torch.load(mp, map_location="cpu"))
                mdl.to(dev)
                sub = collect_candidates(
                    mdl, tok, [posts[i] for i in idx], args.max_len, dev,
                    risks=None if risks is None else [risks[i] for i in idx])
                for i, c in zip(idx, sub):
                    allc[i] = c
                del mdl
                if dev == "cuda":
                    torch.cuda.empty_cache()
            cands[key] = allc
            print(f"  decoded {key}")

    print(f"\n{'arm':<14}{'micro':>9}{'macro':>9}{'macro+supp':>12}"
          f"   (fixed conf {args.conf})")
    print("-" * 48)
    res = {}
    for key, c in cands.items():
        p = spans_at(c, args.conf)
        risks = pl if key != "cond@GOLD" else gl
        mic, mac, sup = score(p, gold, risks)
        res[key] = (mic, mac, sup)
        print(f"{key:<14}{mic:>9.4f}{mac:>9.4f}{sup:>12.4f}")

    if "base" in res:
        print("\n" + "=" * 64)
        bm, bM, bS = res["base"]
        for key in ("cond@PRED", "cond@GOLD"):
            if key not in res:
                continue
            m, M, S = res[key]
            tail = ("  <- SHIPPABLE" if key == "cond@PRED"
                    else "  <- ceiling if risk were perfect")
            print(f"{key} vs base   micro {m - bm:+.4f}   macro {M - bM:+.4f}"
                  f"   macro+supp {S - bS:+.4f}{tail}")
        print("\nMACRO+SUPP is the number that matters -- it is the official")
        print("average with the deployed suppression rule. A positive micro")
        print("with a flat macro+supp is a win in the wrong metric.")
        print("=" * 64)

    print("\nsweep (information only -- the headline above is the fixed conf)")
    print(f"{'conf':<7}" + "".join(f"{k:>14}" for k in cands))
    for t in SWEEP:
        line = f"{t:<7.2f}"
        for key, c in cands.items():
            risks = pl if key != "cond@GOLD" else gl
            line += f"{score(spans_at(c, t), gold, risks)[2]:>14.4f}"
        print(line)
    print("(values are macro+suppression)")


if __name__ == "__main__":
    main()
