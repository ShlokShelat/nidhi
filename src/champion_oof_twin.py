"""
champion_oof_twin.py

BUILD THE VALIDATION-SIDE TWIN OF THE CHAMPION'S RISK VECTOR.

`champion_risk_inject.py --verify` now proves the TEST vector:

    M_test = 0.70 * ensembled_risk.csv + 0.30 * nli_risk_test_probs.csv
             -> rules -> route_attempt   = 378/378 champion labels

But every risk correction has historically been scored against
`artifacts/champion_oof.csv`, which is a DIFFERENT model: 0.786 where the
deployed one is ~0.826. Recalibrating risk on it showed the failure
mode of that substitution -- a stack learned to fix the fold models' errors
and was applied to a better model that does not make them.

THE FIX IS TO REBUILD THE SAME BLEND FROM THE SAME MEMBERS' VAL PREDICTIONS,
which `scripts/rebuild_and_apply_rules.sh` step 2 already names:

    ens_val = 0.5 * mean(risk_v1, risk_v1_s14, risk_v1_s15 val_predictions)
            + 0.5 * friend_risk_probs_val
    M_val   = 0.70 * ens_val + 0.30 * mean(nli val probs)
            -> rules -> route_attempt(cot_soft_oof.csv)

If `M_val` scores near 0.826 rather than 0.786, it is the matched instrument
and every shelved risk idea becomes honestly testable. If it scores 0.786 too,
then the 0.04 gap is NOT a fold/full-data artefact and the real explanation is
still missing -- which is worth knowing before another submission.

THE LIMIT, STATED UP FRONT. `val_split.csv` is 277 posts from the authors held
out of `train_split`. That is a fifth of what champion_oof covers, so per-split
noise is roughly 2.4x wider. It is small and honest, against large and wrong.
Judge by WIN RATE over author resamples, never the CI of the mean
(the CI of a mean over overlapping resamples is far too narrow).

Also re-measures the severity nudge from the evidence-tagger severity probe on
this matched base, since that nudge was previously sized on the mismatched
one.

Usage:
  python src/champion_oof_twin.py
  python src/champion_oof_twin.py --splits 400
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_rules_to_test import apply_rules  # noqa: E402
from scoring import normalize_label  # noqa: E402

LEVELS = ["indicator", "ideation", "behavior", "attempt"]


def wf1(gold, pred):
    gold, pred = np.asarray(gold), np.asarray(pred)
    tot = 0.0
    for c in LEVELS:
        g, p = (gold == c), (pred == c)
        tp = float((g & p).sum())
        pr = tp / max(p.sum(), 1e-9)
        rc = tp / max(g.sum(), 1e-9)
        tot += 2 * pr * rc / max(pr + rc, 1e-9) * g.sum()
    return tot / max(len(gold), 1)


def probs_of(path, rows):
    """Pull a [n,4] probability block for `rows`, tolerating the several
    column spellings these files use (p_Indicator / prob_indicator / ...)."""
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path, keep_default_na=False)
    low = {c.lower(): c for c in d.columns}
    rid = next((low[c] for c in ("row_id", "id", "post_id") if c in low), None)
    if rid is None:
        return None
    cols = None
    for pat in ("p_{}", "prob_{}", "{}", "prob{}"):
        c = [low.get(pat.format(x)) for x in LEVELS]
        if all(c):
            cols = c
            break
    if cols is None:
        return None
    d[rid] = d[rid].astype(str)
    d = d.drop_duplicates(subset=[rid]).set_index(rid)
    if any(r not in d.index for r in rows):
        n = sum(1 for r in rows if r not in d.index)
        print(f"    {os.path.basename(path)}: missing {n}/{len(rows)} rows")
        return None
    X = d.loc[rows, cols].to_numpy(float)
    return X / np.clip(X.sum(1, keepdims=True), 1e-9, None)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--ens-val", default=(
        "runs/risk_v1/val_predictions.csv,"
        "runs/risk_v1_s14/val_predictions.csv,"
        "runs/risk_v1_s15/val_predictions.csv"))
    a_.add_argument("--friend-val", default="artifacts/friend_risk_probs_val.csv")
    a_.add_argument("--nli-val", default=("artifacts/nli_risk_val_probs.csv,"
                                          "artifacts/nli_risk_val_probs2.csv"))
    a_.add_argument("--route", default="artifacts/cot_soft_oof.csv")
    a_.add_argument("--probe", default="artifacts/risk_probe.csv")
    a_.add_argument("--champ-oof", default="artifacts/champion_oof.csv")
    a_.add_argument("--weight-ours", type=float, default=0.5)
    a_.add_argument("--nli-w", type=float, default=0.30)
    a_.add_argument("--splits", type=int, default=400)
    a_.add_argument("--seed", type=int, default=17)
    a_.add_argument("--out", default="artifacts/champion_val_probs.csv")
    a_.add_argument("--no-friend-restrict", action="store_true",
                    help="keep all val rows even where the friend has no "
                         "prediction, which DROPS the friend and rebuilds an "
                         "encoder-only chain. Only for reproducing the old, "
                         "wrong twin.")
    args = a_.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)

    # RESTRICT TO FRIEND COVERAGE INSTEAD OF DROPPING THE FRIEND.
    #
    # `probs_of` returns None when a member does not cover every row, and the
    # friend covers 354 of 372. The old behaviour therefore fell through to
    # `E = O` -- an ENCODER-ONLY twin missing 35% of the deployed blend's mass
    # -- and merely printed a warning that is easy to lose in a long log. The
    # symptom was visible and went unread: the build table showed
    # "encoder seeds only 0.7498" and "+ friend 0.7498", identical, because no
    # friend was ever added. Every risk lane "closed on the twin" was closed on
    # an object that is not the champion.
    #
    # `risk_member_forward.py` already solves this the right way: intersect the
    # rows down to the friend's coverage, losing 18 posts but measuring the
    # actual deployed chain. Do the same here, by default.
    if not args.no_friend_restrict and os.path.exists(args.friend_val):
        fr = pd.read_csv(args.friend_val, keep_default_na=False)
        have = {str(x) for x in fr["row_id"]}
        keep = va["row_id"].isin(have)
        if 0 < (~keep).sum():
            print(f"friend coverage: restricting {len(va)} -> {int(keep.sum())} "
                  f"posts (dropping {int((~keep).sum())} the friend lacks) so "
                  f"the twin is the DEPLOYED blend and not encoder-only")
            va = va[keep].reset_index(drop=True)

    rows = va["row_id"].tolist()
    gold = np.array([normalize_label(v) for v in va["risk_level"]])
    posts = list(va["post"].astype(str))
    auth = np.array([str(x) for x in va["anon_user_id"]]) \
        if "anon_user_id" in va.columns else np.arange(len(rows)).astype(str)
    print(f"val_split: {len(rows)} posts, {len(pd.unique(auth))} authors")
    print("  gold: " + "  ".join(f"{c} {int((gold==c).sum())}" for c in LEVELS))

    print("\nmembers:")
    ours = []
    for p in [x.strip() for x in args.ens_val.split(",")]:
        X = probs_of(p, rows)
        print(f"  {'OK ' if X is not None else 'MISS'} {p}")
        if X is not None:
            ours.append(X)
    F = probs_of(args.friend_val, rows)
    print(f"  {'OK ' if F is not None else 'MISS'} {args.friend_val}")
    nlis = []
    for p in [x.strip() for x in args.nli_val.split(",")]:
        X = probs_of(p, rows)
        print(f"  {'OK ' if X is not None else 'MISS'} {p}")
        if X is not None:
            nlis.append(X)
    if not ours or not nlis:
        raise SystemExit("cannot rebuild: missing encoder or NLI val probs")

    O = np.mean(ours, axis=0)
    E = args.weight_ours * O + (1 - args.weight_ours) * F if F is not None else O
    if F is None:
        print("  WARNING: no friend val probs -- ens_val is encoder-only, so "
              "this is NOT the champion's blend")
    N = np.mean(nlis, axis=0)
    M = (1 - args.nli_w) * E + args.nli_w * N

    route = {}
    if os.path.exists(args.route):
        lm = pd.read_csv(args.route, keep_default_na=False)
        lm["row_id"] = lm["row_id"].astype(str)
        rc = "cot" if "cot" in lm.columns else "pred_risk_level"
        route = {r: normalize_label(v) for r, v in zip(lm["row_id"], lm[rc])}

    def finalize(P):
        lab = [LEVELS[i] for i in P.argmax(1)]
        lab = [normalize_label(apply_rules(p, l)[0])
               for p, l in zip(posts, lab)]
        return np.array([("attempt" if (route.get(r) == "attempt"
                                        and l != "indicator") else l)
                         for r, l in zip(rows, lab)])

    print("\n" + "=" * 78)
    print("STAGE SCORES ON val_split -- compare with the DEPLOYED ~0.826")
    print("=" * 78)
    stages = [("encoder seeds only", np.array([LEVELS[i]
                                               for i in O.argmax(1)])),
              ("+ friend (ens_val)", np.array([LEVELS[i]
                                               for i in E.argmax(1)])),
              ("+ NLI 0.30 (= M)", np.array([LEVELS[i] for i in M.argmax(1)])),
              ("+ rules + route", finalize(M))]
    for nm, lab in stages:
        print(f"  {nm:<26}{wf1(gold, lab):>8.4f}")
    final = stages[-1][1]

    co = probs_of(args.champ_oof, rows)
    if co is not None:
        print(f"  {'champion_oof on same rows':<26}"
              f"{wf1(gold, np.array([LEVELS[i] for i in co.argmax(1)])):>8.4f}"
              f"   <- the MISMATCHED instrument")
    print("\n  If '+ rules + route' lands near 0.826, this is the matched")
    print("  instrument and risk is honestly testable again. If it lands near")
    print("  0.786 like champion_oof, the 0.04 gap is NOT fold-vs-full-data")
    print("  and the real cause is still unidentified -- do not ship on it.")

    out = pd.DataFrame({"row_id": rows, "pred_risk_level": final})
    for j, c in enumerate(LEVELS):
        out[f"prob_{c}"] = M[:, j]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    # ---------------- the nudge, re-measured on the MATCHED base ------------
    sv = probe = None
    if os.path.exists(args.probe):
        pr = pd.read_csv(args.probe)
        pr["row_id"] = pr["row_id"].astype(str)
        pr = pr.drop_duplicates(subset=["row_id"]).set_index("row_id")
        have = [r for r in rows if r in pr.index]
        if len(have) == len(rows) and "conf_indicator" in pr.columns:
            sv = pr.loc[rows, "conf_indicator"].to_numpy(float)
        else:
            probe = f"covers {len(have)}/{len(rows)} val rows"
    if sv is None:
        print(f"\n(no severity for these rows{': ' + probe if probe else ''} "
              f"-- run risk_tagger_probe.py over val_split to re-measure the "
              f"nudge here)")
        return

    ua = pd.unique(auth)
    rng = np.random.default_rng(args.seed)
    by = {a: np.flatnonzero(auth == a) for a in ua}
    boots = [np.concatenate([by[a] for a in rng.choice(ua, len(ua), True)])
             for _ in range(args.splits)]
    print("\n" + "=" * 78)
    print("THE SEVERITY NUDGE, RE-MEASURED ON THE MATCHED BASE")
    print("=" * 78)
    print(f"  base {wf1(gold, final):.4f}   {len(rows)} posts, "
          f"{len(ua)} authors, {args.splits} author resamples")
    print(f"  {'w':>6}{'changed':>9}{'right':>7}{'wrong':>7}{'wF1':>9}"
          f"{'d comp':>9}{'win':>6}{'sd':>9}")
    lp = np.log(np.clip(M, 1e-9, None))
    for w_ in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
        nw = finalize(np.exp(lp + w_ * np.outer(sv, np.arange(4)) / 3.0))
        ch = nw != final
        if not ch.sum():
            print(f"  {w_:>6.2f}{0:>9}")
            continue
        d = np.array([wf1(gold[b], nw[b]) - wf1(gold[b], final[b])
                      for b in boots])
        print(f"  {w_:>6.2f}{int(ch.sum()):>9}"
              f"{int(((nw == gold) & (final != gold) & ch).sum()):>7}"
              f"{int(((final == gold) & (nw != gold) & ch).sum()):>7}"
              f"{wf1(gold, nw):>9.4f}{0.4*d.mean():>+9.4f}"
              f"{np.mean(d > 0):>5.0%}{0.4*d.std():>9.4f}")
    print("\n  THIS table supersedes the sizing from risk_probe_analyze.py.")
    print("  That one was measured on champion_oof, a model the champion is")
    print("  not. If the sign or size differs here, believe THIS one.")


if __name__ == "__main__":
    main()
