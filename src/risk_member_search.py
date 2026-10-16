"""
risk_member_search.py

Adds a genuinely different model to the risk blend, judged on a twin that
reproduces the deployed model.

WHY THIS LANE AND NOT ANOTHER DECODE. A different decode, a different
threshold, a different blend weight, or rules written over existing outputs
all re-process information the model chain already extracted from the post.
None of that adds new information about a post, and a decode cannot recover
signal a model's representation never captured in the first place. A new
member with a genuinely different architecture can.

The champion's risk column is 3 DeBERTa-v3 seeds + a friend model + an NLI
head -- one architecture family, all reading the post the same way. Trained and
sitting unused: `risk_lora7b`, `risk_loraL8`, `risk_q72`, `risk_ql70`, LoRA'd
LLMs, one of which is the first risk member observed with a real confidence
gap over the encoder ensemble. They were not previously added to the blend
because there was no instrument that could distinguish a real gain from
`champion_oof.csv` noise.

There is now: a twin that reproduces the deployed weighted-F1 (0.8294 against
the deployed 0.826), whose positive control passes (it re-identifies
route_attempt at 4 fix / 1 break).

WHAT IS MEASURED, per candidate member:

  1. STANDALONE  its own wF1 and per-class recall on the twin's val posts.
     A member does not need to beat the blend overall to be useful -- a much
     weaker member can still win on one class alone.

  2. COMPLEMENTARITY  the disagreement matrix: on posts where the blend and the
     member differ, who is right? **This is the number that decides.** A member
     that is worse overall but wins by a wide margin in one cell carries
     information the blend does not.

  3. AS A BLEND MEMBER  (1-w)*M + w*member, swept, then rules + route_attempt
     applied on top exactly as the champion does -- because a gain the
     symbolic layer would have produced anyway is not a real gain. Measuring
     a component's effect without that layer misrepresents what the deployed
     pipeline actually does.

  4. AS A CLASS ROUTE  force the member's call for ONE class only, the
     route_attempt mechanism, for each class in turn.

All scored with author resampling on the twin. Report WIN RATE and the share of
resamples that actually LOSE, not the confidence interval of the mean delta,
which understates the per-draw variance a single held-out evaluation draw
samples from, and read fix/break before any delta.

Usage:
  python src/risk_member_search.py
  python src/risk_member_search.py --members runs/risk_lora7b_f{f}/val_probs.npy
"""

from __future__ import annotations

import argparse
import glob
import os
import re
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


def load_member_partial(path, rows):
    """{row_id: probs} for whatever subset of `rows` this file covers.

    A fold run's val_probs.npy holds only the posts that fold held OUT, so no
    single one covers the twin -- but the five together cover it exactly and
    disjointly, and every post is then scored by a model that never saw it."""
    want = set(rows)
    if path.endswith(".npy"):
        sib = os.path.join(os.path.dirname(path), "val_predictions.csv")
        if not os.path.exists(sib):
            return None
        d = pd.read_csv(sib, keep_default_na=False)
        if "row_id" not in d.columns:
            return None
        X = np.load(path)
        if X.ndim != 2 or X.shape[0] != len(d) or X.shape[1] != 4:
            return None
        return {str(r): X[i] for i, r in enumerate(d["row_id"])
                if str(r) in want}
    got = load_member(path, rows)
    if got is None or isinstance(got, tuple):
        return None
    return {r: got[i] for i, r in enumerate(rows)}


def load_member(path, rows):
    """Accept .npy (aligned to a fold's val order, needs a row_id sidecar) or
    a csv carrying row_id + four probability columns. Returns [n,4] or None."""
    if path.endswith(".npy"):
        sib = os.path.join(os.path.dirname(path), "val_predictions.csv")
        if not os.path.exists(sib):
            return None
        d = pd.read_csv(sib, keep_default_na=False)
        if "row_id" not in d.columns:
            return None
        X = np.load(path)
        if X.ndim != 2 or X.shape[0] != len(d) or X.shape[1] != 4:
            return None
        m = {str(r): X[i] for i, r in enumerate(d["row_id"])}
    else:
        try:
            d = pd.read_csv(path, keep_default_na=False)
        except Exception:
            return None
        low = {c.lower(): c for c in d.columns}
        if "row_id" not in low:
            return None
        cols = None
        for pat in ("prob_{}", "p_{}", "{}"):
            c = [low.get(pat.format(x)) for x in LEVELS]
            if all(c):
                cols = c
                break
        if cols is None:
            return None
        d[low["row_id"]] = d[low["row_id"]].astype(str)
        d = d.drop_duplicates(subset=[low["row_id"]])
        m = {r: v for r, v in zip(d[low["row_id"]],
                                  d[cols].to_numpy(float))}
    miss = [r for r in rows if r not in m]
    if miss:
        return ("partial", len(rows) - len(miss))
    X = np.stack([m[r] for r in rows]).astype(float)
    return X / np.clip(X.sum(1, keepdims=True), 1e-9, None)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--probs", default="artifacts/champion_val_probs.csv")
    a_.add_argument("--route", default="artifacts/cot_soft_oof.csv")
    a_.add_argument("--members", default=(
        "runs/risk_lora7b*/val_probs.npy,runs/risk_loraL8*/val_probs.npy,"
        "runs/risk_q72*/val_probs.npy,runs/risk_ql70*/val_probs.npy,"
        "runs/risk_nli*/val_probs.npy,artifacts/qwen_ord_*probs.csv,"
        "artifacts/risk_llm14b*.csv,artifacts/cot_soft*.csv"))
    a_.add_argument("--splits", type=int, default=300)
    a_.add_argument("--seed", type=int, default=17)
    args = a_.parse_args()

    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    rows = va["row_id"].tolist()
    gold = np.array([normalize_label(v) for v in va["risk_level"]])
    posts = list(va["post"].astype(str))
    auth = np.asarray([str(x) for x in va["anon_user_id"]])

    pv = pd.read_csv(args.probs)
    pv["row_id"] = pv["row_id"].astype(str)
    pv = pv.set_index("row_id")
    M = pv.loc[rows, [f"prob_{c}" for c in LEVELS]].to_numpy(float)

    route = {}
    if os.path.exists(args.route):
        d = pd.read_csv(args.route, keep_default_na=False)
        d["row_id"] = d["row_id"].astype(str)
        c = "cot" if "cot" in d.columns else "pred_risk_level"
        route = {r: normalize_label(v) for r, v in zip(d["row_id"], d[c])}

    def finalize(X):
        lab = [LEVELS[i] for i in X.argmax(1)]
        lab = [normalize_label(apply_rules(p, l)[0]) for p, l in zip(posts, lab)]
        return np.array([("attempt" if (route.get(r) == "attempt"
                                        and x != "indicator") else x)
                         for r, x in zip(rows, lab)])

    base = finalize(M)
    ua = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in ua}
    rng = np.random.default_rng(args.seed)
    boots = [np.concatenate([by[a] for a in rng.choice(ua, len(ua), True)])
             for _ in range(args.splits)]
    print(f"twin: {len(rows)} posts / {len(ua)} authors   "
          f"base wF1 {wf1(gold, base):.4f}\n")

    cands = sorted({g for pat in args.members.split(",")
                    for g in glob.glob(pat.strip())})
    loaded = {}
    print("candidate members:")

    # ------------------------------------------------------------------
    # LEAKAGE GATE. A run whose name contains `full` was trained on ALL 1635
    # posts, val_split included, so scoring it here reads back its own training
    # labels. `risk_ql70full_full` scored **0.9415** standalone -- against a
    # deployed 0.826 and a human annotation ceiling near 0.80-0.83 -- and
    # produced 28 fix / 1 break as a blend member with almost no losing
    # resamples. That pattern is the signature of memorisation, not skill.
    # Refused by name, and by an implausible standalone score, because either
    # alone could be a coincidence.
    # ------------------------------------------------------------------
    LEAK_WF1 = 0.88

    def leaky(nm, X):
        s = wf1(gold, np.array(LEVELS)[X.argmax(1)])
        if "full" in nm.lower():
            return f"trained on ALL posts incl. val (name says `full`), " \
                   f"wF1 {s:.4f}"
        if s > LEAK_WF1:
            return f"wF1 {s:.4f} exceeds the annotation ceiling (~0.83) -- " \
                   f"almost certainly trained on these posts"
        return None

    # group fold families: runs/risk_ql70_f0..f4 each hold out a DISJOINT slice
    # of val, and together they cover it exactly. Stitching them gives a member
    # where every post is scored by a model that never saw it -- the honest
    # counterpart of the `full` run above.
    fams = {}
    singles = []
    for p in cands:
        m = re.match(r"^(.*?)_f\d+(/.*)?$", p)
        if m:
            fams.setdefault(m.group(1), []).append(p)
        else:
            singles.append(p)

    for p in singles:
        got = load_member(p, rows)
        if got is None:
            continue
        nm = p.replace("runs/", "").replace("/val_probs.npy", "") \
              .replace("artifacts/", "").replace(".csv", "")
        if isinstance(got, tuple):
            print(f"  skip {nm:<46} covers {got[1]}/{len(rows)}")
            continue
        why = leaky(nm, got)
        if why:
            print(f"  LEAK {nm:<46} REFUSED: {why}")
            continue
        loaded[nm] = got
        print(f"  OK   {nm:<46} wF1 "
              f"{wf1(gold, np.array(LEVELS)[got.argmax(1)]):.4f}")

    for fam, paths in sorted(fams.items()):
        acc, seen = {}, []
        for p in sorted(paths):
            part = load_member_partial(p, rows)
            if part:
                seen.append(os.path.basename(os.path.dirname(p)))
                for r, v in part.items():
                    acc.setdefault(r, v)
        nm = fam.replace("runs/", "") + "_xfit"
        miss = [r for r in rows if r not in acc]
        if miss:
            print(f"  skip {nm:<46} {len(paths)} folds cover "
                  f"{len(acc)}/{len(rows)}")
            continue
        X = np.stack([acc[r] for r in rows]).astype(float)
        X = X / np.clip(X.sum(1, keepdims=True), 1e-9, None)
        why = leaky(nm, X)
        if why:
            print(f"  LEAK {nm:<46} REFUSED: {why}")
            continue
        loaded[nm] = X
        print(f"  OK   {nm:<46} wF1 "
              f"{wf1(gold, np.array(LEVELS)[X.argmax(1)]):.4f}"
              f"   (stitched from {len(seen)} disjoint folds)")

    if not loaded:
        raise SystemExit("\nno leak-free member covers all twin rows")

    def judge(nm, new, ref=None):
        ref = base if ref is None else ref
        ch = new != ref
        if not ch.sum():
            return None
        fix = int(((new == gold) & (ref != gold) & ch).sum())
        brk = int(((ref == gold) & (new != gold) & ch).sum())
        d = np.array([wf1(gold[b], new[b]) - wf1(gold[b], ref[b])
                      for b in boots])
        return (int(ch.sum()), fix, brk, wf1(gold, new), 0.4 * d.mean(),
                float(np.mean(d > 0)), float(np.mean(d < 0)), 0.4 * d.std(), nm)

    print("\n" + "=" * 96)
    print("COMPLEMENTARITY -- on posts where the member and the blend DIFFER,")
    print("who is right? This decides the lane before any blending.")
    print("=" * 96)
    print(f"  {'member':<26}{'differ':>8}{'member right':>14}"
          f"{'blend right':>13}{'ratio':>8}   best cell (member wins)")
    for nm, X in loaded.items():
        mp = np.array(LEVELS)[X.argmax(1)]
        dif = mp != base
        if not dif.sum():
            continue
        mr = int(((mp == gold) & dif).sum())
        br = int(((base == gold) & dif).sum())
        cell, bestv = "-", 0
        for c in LEVELS:
            s = dif & (mp == c)
            if s.sum() >= 3:
                w = int(((mp == gold) & s).sum())
                l_ = int(((base == gold) & s).sum())
                if w - l_ > bestv:
                    bestv, cell = w - l_, f"->{c}  {w}W/{l_}L of {int(s.sum())}"
        print(f"  {nm:<26}{int(dif.sum()):>8}{mr:>14}{br:>13}"
              f"{mr/max(br,1):>8.2f}   {cell}")
    print("\n  A ratio above ~1.5 in ANY cell is the route_attempt shape and is")
    print("  worth routing even if the member is much worse overall.")

    print("\n" + "=" * 96)
    print("AS A BLEND MEMBER -- (1-w)*M + w*member, then rules + route")
    print("=" * 96)
    print(f"  {'member':<20}{'w':>6}{'moved':>7}{'fix':>5}{'brk':>5}"
          f"{'wF1':>9}{'d comp':>9}{'win':>6}{'LOSS':>6}{'sd':>8}")
    best = []
    for nm, X in loaded.items():
        for w_ in (0.10, 0.20, 0.30, 0.40, 0.50):
            r = judge(nm, finalize((1 - w_) * M + w_ * X))
            if r is None:
                continue
            if r[5] >= 0.60 or w_ == 0.30:
                print(f"  {nm:<20}{w_:>6.2f}{r[0]:>7}{r[1]:>5}{r[2]:>5}"
                      f"{r[3]:>9.4f}{r[4]:>+9.4f}{r[5]:>5.0%}{r[6]:>6.0%}"
                      f"{r[7]:>8.4f}")
            best.append((r[5], r[4], r[1], r[2], f"blend {nm} w={w_}"))

    print("\n" + "=" * 96)
    print("AS A CLASS ROUTE -- force the member's call for ONE class")
    print("=" * 96)
    print(f"  {'member -> class':<34}{'moved':>7}{'fix':>5}{'brk':>5}"
          f"{'wF1':>9}{'d comp':>9}{'win':>6}{'LOSS':>6}")
    for nm, X in loaded.items():
        mp = np.array(LEVELS)[X.argmax(1)]
        for c in LEVELS:
            new = np.array([(c if (mp[i] == c and base[i] != "indicator")
                             else base[i]) for i in range(len(rows))])
            r = judge(nm, new)
            if r is None or r[0] < 2:
                continue
            if r[5] >= 0.60:
                print(f"  {nm + ' -> ' + c:<34}{r[0]:>7}{r[1]:>5}{r[2]:>5}"
                      f"{r[3]:>9.4f}{r[4]:>+9.4f}{r[5]:>5.0%}{r[6]:>6.0%}")
            best.append((r[5], r[4], r[1], r[2], f"route {nm} -> {c}"))

    print("\n" + "=" * 96)
    if best:
        best.sort(reverse=True)
        for win, dc, fix, brk, nm in best[:5]:
            print(f"  {nm:<44}{dc:>+9.4f}  win {win:>4.0%}  "
                  f"{fix} fix / {brk} break")
        w, dc, fix, brk, nm = best[0]
        ok = w >= 0.90 and fix >= 2 * max(brk, 1)
        print(f"\n  BEST: {nm}")
        print(f"  {'CLEARS' if ok else 'does NOT clear'} the bar "
              f"(>=90% win AND >=2:1 fix/break).")
        if not ok:
            print("  An override that misses this bar is not a small win, it")
            print("  is an untested one: the class-restricted route_attempt")
            print("  shape is the only construction that clears it reliably.")
    print("\n  Only 32 authors on the twin, so resampling is coarse. Treat")
    print("  anything under the bar as untested rather than as a small win.")


if __name__ == "__main__":
    main()
