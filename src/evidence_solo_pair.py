"""
evidence_solo_pair.py

TWO TAGGERS, THE SAME POSTS, ONE FLAG APART.

`evidence_vote_oof.py` hardcodes `v2` as its solo baseline, so reordering
`--tags` does not swap which arm gets scored alone. Worse, its post set is the
UNION of candidate row_ids, so v2 was scored on 1102 posts and v2pp on 1110 --
different denominators, and 8 posts out of 1110 can move a macro mean by ~0.005,
which is exactly the size of the effect being claimed.

This scores both on the INTERSECTION and reports a paired author bootstrap, so
the comparison is like-for-like.

WHAT IS BEING COMPARED. `v2pp` is `v2` with one flag changed: `--per-post`,
which averages the token loss within each post before averaging across posts.
The scorer MACRO-averages phrase-F1 over POSTS, but the default token-mean loss
gives a 400-token post ~10x the gradient of a 40-token one. Same folds, same
seed 13, same max-len, same 6 epochs, same lr, same batch. One variable.

The claim to check: v2pp = 0.7590 vs v2 = 0.7540 on DIFFERENT post sets.

ALSO REPORTED, because a solo win only matters if it survives the vote: the
5-tagger cluster>=2 macro with v2 swapped for v2pp. Note the dilution -- one of
five taggers changing means any real gain arrives at about a fifth strength,
which is how six genuine member gains got buried earlier in this campaign.

Each tagger's own `min_conf` floor is applied, exactly as the vote does.

Usage:
  python src/evidence_solo_pair.py
  python src/evidence_solo_pair.py --a v2 --b v2pp --boot 4000
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import parse_evidence, phrase_f1, token_len  # noqa: E402


def norm(s):
    return " ".join(str(s).lower().split())


def load(path):
    d = pd.read_csv(path, keep_default_na=False)
    d["row_id"] = d["row_id"].astype(str)
    if "conf" in d.columns and "min_conf" in d.columns and len(d):
        mc = float(d["min_conf"].iloc[0])
        d = d[d["conf"] >= mc]
    g = {}
    for r, s in zip(d["row_id"], d["span"]):
        g.setdefault(r, []).append(str(s))
    return g


def dedupe(spans):
    out = []
    for s in spans:
        if not any(norm(s) == norm(t) for t in out):
            out.append(s)
    return out


def contains(a, b):
    na, nb = norm(a), norm(b)
    return na in nb or nb in na


def cluster_vote(per_run, r, k):
    items = [(run, s) for run in per_run for s in per_run[run].get(r, [])]
    if not items:
        return []
    items.sort(key=lambda x: (-token_len(x[1]), x[1]))
    groups = []
    for run, s in items:
        for g in groups:
            if any(contains(s, t) and
                   max(token_len(s), token_len(t)) <=
                   3 * max(min(token_len(s), token_len(t)), 1)
                   for _, t in g):
                g.append((run, s))
                break
        else:
            groups.append([(run, s)])
    out = []
    for g in groups:
        if len({run for run, _ in g}) < k:
            continue
        cnt = Counter(s for _, s in g)
        out.append(max(cnt.items(),
                       key=lambda kv: (kv[1], -token_len(kv[0])))[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/evidence_oof")
    ap.add_argument("--a", default="v2")
    ap.add_argument("--b", default="v2pp")
    ap.add_argument("--others", default="ep12,lr3,bs16,ml256")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    A = load(f"{args.dir}/{args.a}_oof_candidates.csv")
    B = load(f"{args.dir}/{args.b}_oof_candidates.csv")
    others = {t: load(f"{args.dir}/{t}_oof_candidates.csv")
              for t in args.others.split(",") if t.strip()}

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")

    # THE INTERSECTION -- the whole point of this script
    rows = sorted((set(A) | set(B)) & set(lab.index))
    both = sorted(set(A) & set(B) & set(lab.index))
    print(f"{args.a}: {len(A)} posts covered   {args.b}: {len(B)} posts")
    print(f"union with gold {len(rows)}   INTERSECTION {len(both)}")
    print(f"scoring on the intersection so the denominators match\n")

    gold = [parse_evidence(lab.loc[r, "evidence"]) for r in both]
    auth = np.array([str(lab.loc[r, "anon_user_id"]) for r in both])
    uauth = pd.unique(auth)
    by = {a: np.flatnonzero(auth == a) for a in uauth}

    pa = [dedupe(A.get(r, [])) for r in both]
    pb = [dedupe(B.get(r, [])) for r in both]

    def sc(pred, idx=None):
        if idx is None:
            return phrase_f1(gold, pred, average="macro").f1
        return phrase_f1([gold[i] for i in idx], [pred[i] for i in idx],
                         average="macro").f1

    rng = np.random.default_rng(args.seed)
    IXS = [np.concatenate([by[uauth[i]] for i in
                           rng.choice(len(uauth), len(uauth), replace=True)])
           for _ in range(args.boot)]
    half = set(uauth[: len(uauth) // 2])
    HA = np.concatenate([by[a] for a in uauth if a in half])
    HB = np.concatenate([by[a] for a in uauth if a not in half])

    print("=" * 88)
    print(f"SOLO, same posts   ({len(both)} posts / {len(uauth)} authors)")
    print("=" * 88)
    va, vb = sc(pa), sc(pb)
    d = np.array([sc(pb, ix) - sc(pa, ix) for ix in IXS])
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"  {args.a:<6} {va:.4f}   spans {sum(len(p) for p in pa)}")
    print(f"  {args.b:<6} {vb:.4f}   spans {sum(len(p) for p in pb)}")
    print(f"  delta {vb - va:+.4f}   [{lo:+.4f},{hi:+.4f}]   "
          f"win {(d > 0).mean():.0%}")
    print(f"  halves  A {sc(pb, HA) - sc(pa, HA):+.4f}   "
          f"B {sc(pb, HB) - sc(pa, HB):+.4f}")

    # ---- and inside the vote, where the dilution lives --------------------
    print("\n" + "=" * 88)
    print(f"IN THE 5-TAGGER VOTE (cluster>={args.k}), one tagger swapped")
    print("=" * 88)
    rowsv = sorted(set(lab.index) & (set(A) | set(B) |
                                     set().union(*[set(o) for o in
                                                   others.values()])))
    goldv = [parse_evidence(lab.loc[r, "evidence"]) for r in rowsv]
    authv = np.array([str(lab.loc[r, "anon_user_id"]) for r in rowsv])
    ua = pd.unique(authv)
    byv = {a: np.flatnonzero(authv == a) for a in ua}

    def vote(with_b):
        pr = dict(others)
        pr["solo"] = B if with_b else A
        return [dedupe(cluster_vote(pr, r, args.k)) for r in rowsv]

    wa, wb = vote(False), vote(True)

    def scv(pred, idx=None):
        if idx is None:
            return phrase_f1(goldv, pred, average="macro").f1
        return phrase_f1([goldv[i] for i in idx], [pred[i] for i in idx],
                         average="macro").f1

    rng2 = np.random.default_rng(args.seed + 1)
    IX2 = [np.concatenate([byv[ua[i]] for i in
                           rng2.choice(len(ua), len(ua), replace=True)])
           for _ in range(args.boot)]
    dv = np.array([scv(wb, ix) - scv(wa, ix) for ix in IX2])
    lo2, hi2 = np.percentile(dv, [2.5, 97.5])
    print(f"  with {args.a:<6} {scv(wa):.4f}")
    print(f"  with {args.b:<6} {scv(wb):.4f}")
    print(f"  delta {scv(wb) - scv(wa):+.4f}   [{lo2:+.4f},{hi2:+.4f}]   "
          f"win {(dv > 0).mean():.0%}   on {len(rowsv)} posts")

    print("\n" + "=" * 88)
    print("HOW TO DECIDE")
    print("=" * 88)
    print("The SOLO row is the honest test of the flag: same folds, same seed,")
    print("one variable. If it wins there with the CI clear of zero, the")
    print("per-post loss is real and retraining ep12/lr3/bs16/ml256 the same")
    print("way is ~16 GPU-hours for the whole evidence bag.")
    print("")
    print("The VOTE row will be about a fifth the size -- one of five taggers")
    print("changed. Do not read a small number there as a negative; read the")
    print("solo row for the mechanism and the vote row only for the sign.")
    print("")
    print("And note the champion ships EIGHT `_rc` taggers, not these five.")
    print("A solo win here justifies retraining the `_rc` family with")
    print("--per-post; it does not itself justify a submission.")


if __name__ == "__main__":
    main()
