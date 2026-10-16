"""
risk_nli_pool_oof.py

Pool the 5 fold models of a risk-NLI arm into an honest OOF prediction set.

WHY THIS EXISTS. scripts/risk_nli_gaur.sh ended its comparison by reading
runs/risknli_<tag>_f<k>/val_predictions.csv -- a file train_risk_nli.py has
never written. It saves model.pt and nothing else. Ten trainings finished and
the comparison died on a file that was never going to be there.

Recoverable by inference: each fold model scores its own held-out fold, the five
folds concatenate to a full out-of-fold set, and both arms are compared on
identical rows.

CRITICAL DETAIL. hypothesis() resolves the definition text through _defs(),
which reads the RISK_DEFS environment variable AT CALL TIME. Scoring a `gaur`
model with the default definitions would feed it hypotheses it never saw in
training and silently understate the arm. This sets RISK_DEFS per arm before
scoring, and prints the hypothesis it used so the pairing is visible.

WHAT IT REPORTS. Per-class precision/recall/F1 for each arm, the pre-stated
falsifier (behavior recall >= +0.03 AND weighted-F1 not worse), and the
INDICATOR row -- which matters independently, because gold evidence is empty
exactly on indicator posts, so indicator errors cost the risk term AND the
evidence term while ideation/behavior/attempt errors cost the risk term only.

Usage:
  python src/risk_nli_pool_oof.py --arms base,gaur
  python src/risk_nli_pool_oof.py --arms base,gaur --outdir artifacts/nligaur
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1  # noqa: E402

# RISK_DEFS is read at call time inside train_risk_nli.hypothesis(), so the
# import must happen AFTER it is set per arm. Import the module, not the names.
import train_risk_nli as T  # noqa: E402

ARM_DEFS = {"base": "", "gaur": "gaur", "ladder": "ladder",
            "fenced": "fenced"}


def score_arm(tag: str, folds: int, model_name: str, max_len: int,
              device: str, use_bf16: bool, batch: int) -> pd.DataFrame:
    if ARM_DEFS.get(tag, ""):
        os.environ["RISK_DEFS"] = ARM_DEFS[tag]
    else:
        os.environ.pop("RISK_DEFS", None)
    print(f"\n--- arm {tag}   RISK_DEFS={os.environ.get('RISK_DEFS', '(unset)')}")
    print(f"    behavior hypothesis: {T.hypothesis('behavior')[:120]}...")

    from transformers import AutoTokenizer
    out = []
    for f in range(folds):
        d = f"runs/risknli_{tag}_f{f}"
        mp = os.path.join(d, "model.pt")
        vp = f"artifacts/folds/val_f{f}.csv"
        if not os.path.exists(mp):
            raise SystemExit(f"missing {mp}")
        if not os.path.exists(vp):
            raise SystemExit(f"missing {vp}")
        cfgp = os.path.join(d, "run_config.json")
        mn, ml = model_name, max_len
        if os.path.exists(cfgp):
            cfg = json.load(open(cfgp))
            mn = cfg.get("model", model_name)
            # truncation must match training or the premise is cut differently
            ml = int(cfg.get("max_len", cfg.get("max-len", max_len)))
        tok = AutoTokenizer.from_pretrained(mn)
        va = pd.read_csv(vp)
        model = T.RiskNLIScorer(mn).to(device)
        model.load_state_dict(torch.load(mp, map_location="cpu"))
        probs = T.score_posts(model, va, tok, ml, device, use_bf16, batch)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        sub = pd.DataFrame({"row_id": va["row_id"].astype(str)})
        for j, c in enumerate(RISK_LEVELS):
            sub[f"prob_{c}"] = probs[:, j]
        sub["pred_risk_level"] = [RISK_LEVELS[i] for i in probs.argmax(1)]
        sub["gold"] = [normalize_label(x) for x in va["risk_level"]]
        sub["fold"] = f
        out.append(sub)
        print(f"    fold {f}: {len(sub)} posts scored")
    return pd.concat(out, ignore_index=True).drop_duplicates("row_id")


def per_class(y, p, tag):
    print(f"\n--- {tag} ---   weighted-F1 {risk_weighted_f1(list(y), list(p)):.4f}")
    print(f"{'class':<12}{'gold':>6}{'pred':>6}{'prec':>8}{'rec':>8}{'F1':>8}")
    for c in RISK_LEVELS:
        g, q = (y == c), (p == c)
        tp = int((g & q).sum())
        pr = tp / max(int(q.sum()), 1)
        rc = tp / max(int(g.sum()), 1)
        f1 = 0.0 if tp == 0 else 2 * pr * rc / (pr + rc)
        print(f"{c:<12}{int(g.sum()):>6}{int(q.sum()):>6}{pr:>8.3f}"
              f"{rc:>8.3f}{f1:>8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="base,gaur")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--outdir", default="artifacts/nligaur")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    os.makedirs(args.outdir, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",")]

    got = {}
    for tag in arms:
        df = score_arm(tag, args.folds, args.model, args.max_len, device,
                       use_bf16, args.batch)
        p = os.path.join(args.outdir, f"oof_{tag}.csv")
        df.to_csv(p, index=False)
        print(f"    wrote {p}")
        got[tag] = df.set_index("row_id")

    rows = sorted(set.intersection(*[set(v.index) for v in got.values()]))
    print(f"\n{len(rows)} pooled OOF posts common to all arms")
    y = got[arms[0]].loc[rows, "gold"].to_numpy()
    preds = {t: np.array([normalize_label(x)
                          for x in got[t].loc[rows, "pred_risk_level"]])
             for t in arms}
    for t in arms:
        per_class(y, preds[t], t)
        n_bi = int(((y == "behavior") & (preds[t] == "ideation")).sum())
        n_cross = int(((y == "indicator") != (preds[t] == "indicator")).sum())
        print(f"  behavior->ideation errors: {n_bi}")
        print(f"  errors crossing the INDICATOR boundary: {n_cross}  "
              f"(these cost evidence too)")

    # PER-CLASS recall deltas against base. An arm can improve the common
    # classes and still be a net loss because it collapses recall on `attempt`,
    # which is 7% of the data and carries the least margin. A scalar delta
    # averages that away completely, so print every class separately.
    if "base" in preds:
        b = preds["base"]
        wb = risk_weighted_f1(list(y), list(b))
        for t in arms:
            if t == "base":
                continue
            g = preds[t]
            wg = risk_weighted_f1(list(y), list(g))
            print("\n" + "=" * 66)
            print(f"{t} vs base")
            print("=" * 66)
            print(f"{'class':<12}{'base rec':>10}{'arm rec':>10}{'delta':>9}")
            print("-" * 41)
            for c in RISK_LEVELS:
                m = y == c
                rb = float((b[m] == c).mean())
                rg = float((g[m] == c).mean())
                print(f"{c:<12}{rb:>10.4f}{rg:>10.4f}{rg - rb:>+9.4f}")
            print(f"\nweighted-F1  base {wb:.4f} -> {t} {wg:.4f}   "
                  f"delta {wg - wb:+.4f}   composite {0.4 * (wg - wb):+.4f}")

            mb, ma = y == "behavior", y == "attempt"
            d_beh = float((g[mb] == "behavior").mean()) - float((b[mb] == "behavior").mean())
            r_att = float((g[ma] == "attempt").mean())
            if t in ("ladder", "fenced"):
                # pre-stated in train_risk_nli.py next to RISK_DEFINITIONS_LADDER
                ok = r_att >= 0.620 and wg > wb
                print(f"\n{t.upper()} BAR: attempt recall >= 0.620 "
                      f"(got {r_att:.4f}) AND wF1 > {wb:.4f} (got {wg:.4f})")
                print("PASSES -- the gaur content was right and only the "
                      "packaging was wrong."
                      if ok else
                      "FAILS -- definition engineering closes for risk.")
            else:
                print(f"\nbehavior recall {d_beh:+.4f}   attempt recall "
                      f"{r_att:.4f}")
        print("=" * 66)


if __name__ == "__main__":
    main()
