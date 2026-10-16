"""
evidence_decouple.py

BREAK THE risk -> evidence DEPENDENCY WHERE RISK IS UNRELIABLE.

THE ARCHITECTURAL DEFECT, measured and confirmed
(`evidence_conditioning_oracle.py`, 372 val posts):

    decode        ALL     risk RIGHT  risk WRONG    (63 posts wrong, 16.9%)
    predicted   0.7864     0.8515       0.5789
    gold        0.8261     0.8515       0.7386

Identical on risk-RIGHT posts to four decimals. The conditioned tagger is not
weak -- it is being handed the wrong question on 17% of posts and answering it
faithfully. That is a property of the PIPELINE SHAPE, not of any model in it,
and it is the only structural defect left with a measured price:

  * risk is 0.826 against an annotation ceiling of ~0.80-0.83 -- at the
    ceiling, nothing to fix
  * the factor pool beats all 8 arms on 23 of 24 labels -- at its family's
    ceiling
  * evidence carries an oracle gain that no model change is needed to reach

WHY HEDGING FAILED AND THIS IS DIFFERENT. `evidence_gated_hedge.py` unioned the
predicted conditioning with the runner-up and lost everywhere, because both
labels come from the SAME wrong risk model -- the runner-up is only right some
of the time, giving ~0.30 expected gain against a 0.33 cost, so it needs
P(wrong) > 0.55 and the best gate reaches 0.387.

Conditioning on NOTHING has no such dependency. The unconditioned tagger family
(`artifacts/evidence_oof/{v2,ep12,lr3,bs16,ml256}_oof_candidates.csv`) was the
champion's evidence source before the conditioned family replaced it. It cannot
be mis-conditioned because it is never told anything. The trade should be:

    risk RIGHT   conditioned WINS  (the conditioning is correct and helps)
    risk WRONG   unconditioned WINS (no wrong answer to be led by)

If that shape holds, a gate that switches families is a REAL architectural fix,
and unlike the hedge it does not need to identify wrong posts precisely -- it
needs only the crossover point, because both families are competent everywhere.

WHAT IS REPORTED
  1. both families, split by risk correctness -- the shape test above. If
     unconditioned does NOT win on risk-wrong posts, the lane dies here.
  2. the crossover: at what risk confidence does unconditioned overtake?
  3. cross-fitted gated switch, threshold on FIT authors, scored on HELD-OUT.
  4. the UNION of both families, as a control -- more spans is not the fix,
     and the hedge already showed what that costs.

Usage:
  python src/evidence_conditioning_oracle.py     # writes the conditioned cache
  python src/evidence_decouple.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (normalize_label, normalize_span,  # noqa: E402
                     parse_evidence, phrase_f1, token_len)

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
UNCOND = ["v2", "ep12", "lr3", "bs16", "ml256"]


def pf1(gold, pred):
    """phrase_f1 takes GOLD FIRST and returns a PhraseF1Result, not a float."""
    return phrase_f1(gold, pred).f1


def dedupe(spans):
    out = []
    for s in sorted(spans, key=lambda x: (-token_len(x), -len(x))):
        n = normalize_span(s)
        if n and not any(n in normalize_span(k) for k in out):
            out.append(s)
    return out


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--val", default="artifacts/val_split.csv")
    a_.add_argument("--probs", default="artifacts/champion_val_probs.csv")
    a_.add_argument("--cond", default="artifacts/evcond_val_spans.json")
    a_.add_argument("--uncond-dir", default="artifacts/evidence_oof")
    a_.add_argument("--tags", default=",".join(UNCOND))
    a_.add_argument("--min-conf", type=float, default=0.35)
    a_.add_argument("--min-votes", type=int, default=2)
    a_.add_argument("--splits", type=int, default=200)
    a_.add_argument("--seed", type=int, default=17)
    args = a_.parse_args()

    S = json.load(open(args.cond))
    va = pd.read_csv(args.val, keep_default_na=False)
    va["row_id"] = va["row_id"].astype(str)
    va = va[va["row_id"].isin(S)].reset_index(drop=True)
    rows = va["row_id"].tolist()
    gold_risk = np.array([normalize_label(v) for v in va["risk_level"]])
    gold_ev = [parse_evidence(v) for v in va["evidence"]]
    auth = np.asarray([str(x) for x in va["anon_user_id"]])

    pv = pd.read_csv(args.probs)
    pv["row_id"] = pv["row_id"].astype(str)
    pv = pv.set_index("row_id")
    P = pv.loc[rows, [f"prob_{c}" for c in LEVELS]].to_numpy(float)
    pred_risk = np.array([normalize_label(v)
                          for v in pv.loc[rows, "pred_risk_level"]])
    wrong = pred_risk != gold_risk

    # unconditioned family: the same >=2-of-5 vote the champion used before the
    # conditioned taggers replaced it
    per = {}
    for t in [x.strip() for x in args.tags.split(",")]:
        p = os.path.join(args.uncond_dir, f"{t}_oof_candidates.csv")
        if not os.path.exists(p):
            print(f"  missing {p}")
            continue
        d = pd.read_csv(p)
        if "conf" in d.columns:
            mc = float(d["min_conf"].iloc[0]) if "min_conf" in d.columns else 0.0
            d = d[d["conf"] >= mc]
        m = {}
        for r, s in zip(d["row_id"].astype(str), d["span"]):
            m.setdefault(r, []).append(str(s))
        per[t] = m
    if len(per) < 2:
        raise SystemExit("need at least 2 unconditioned tagger dumps")
    cover = sum(1 for r in rows if any(r in per[t] for t in per))
    print(f"{len(rows)} val posts   unconditioned family covers {cover}   "
          f"({len(per)} taggers)")

    def uncond(r):
        c = {}
        for t in per:
            for s in per[t].get(r, []):
                c.setdefault(normalize_span(s), [set(), s])[0].add(t)
        keep = [v[1] for v in c.values() if len(v[0]) >= args.min_votes]
        if not keep:
            pool = [s for t in per for s in per[t].get(r, [])]
            keep = sorted(pool, key=token_len)[:1]
        return dedupe(keep)

    def cond(r, lv):
        return dedupe([s for s, c in S[r][lv] if c >= args.min_conf])

    C, U, UN = [], [], []
    for i, r in enumerate(rows):
        if pred_risk[i] == "indicator":          # suppression, both families
            C.append([]), U.append([]), UN.append([])
            continue
        c_, u_ = cond(r, pred_risk[i]), uncond(r)
        C.append(c_)
        U.append(u_)
        UN.append(dedupe(c_ + u_))

    iok, ibad = np.flatnonzero(~wrong), np.flatnonzero(wrong)
    print("\n" + "=" * 76)
    print("1. THE SHAPE TEST -- does unconditioned win where risk is WRONG?")
    print("=" * 76)
    print(f"  {'family':<18}{'ALL':>9}{'risk RIGHT':>13}{'risk WRONG':>13}")
    for nm, sp in (("conditioned", C), ("unconditioned", U), ("union", UN)):
        a = pf1(gold_ev, sp)
        b = pf1([gold_ev[i] for i in iok], [sp[i] for i in iok])
        c = pf1([gold_ev[i] for i in ibad], [sp[i] for i in ibad])
        star = "  <- shipped" if nm == "conditioned" else ""
        print(f"  {nm:<18}{a:>9.4f}{b:>13.4f}{c:>13.4f}{star}")
    print(f"\n  n right {len(iok)}   n wrong {len(ibad)}")
    print("  The lane LIVES only if unconditioned beats conditioned in the")
    print("  `risk WRONG` column. If it loses there too, the conditioned")
    print("  tagger is simply the better tagger and the dependency is not")
    print("  what costs the 0.16 -- those posts are just hard.")

    print("\n" + "=" * 76)
    print("2. CROSSOVER -- by risk confidence decile")
    print("=" * 76)
    top = P.max(1)
    q = np.quantile(top, np.linspace(0, 1, 6))
    print(f"  {'conf band':<18}{'n':>5}{'err':>7}{'cond':>9}{'uncond':>9}"
          f"{'diff':>9}")
    for k in range(5):
        m = (top >= q[k]) & (top <= q[k + 1] if k == 4 else top < q[k + 1])
        ix = np.flatnonzero(m)
        if len(ix) < 10:
            continue
        cc = pf1([gold_ev[i] for i in ix], [C[i] for i in ix])
        uu = pf1([gold_ev[i] for i in ix], [U[i] for i in ix])
        print(f"  {f'{q[k]:.2f}-{q[k+1]:.2f}':<18}{len(ix):>5}"
              f"{wrong[ix].mean():>7.0%}{cc:>9.4f}{uu:>9.4f}{uu-cc:>+9.4f}")
    print("\n  A positive `diff` in the low-confidence bands and negative in")
    print("  the high ones is the switch working. A uniformly negative diff")
    print("  means there is no crossover and no gate can help.")

    ua = pd.unique(auth)
    rng = np.random.default_rng(args.seed)
    QG = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    print("\n" + "=" * 76)
    print("3. CROSS-FITTED GATED SWITCH -- unconditioned below the threshold")
    print("=" * 76)
    ds, qs = [], []
    for _ in range(args.splits):
        o = list(ua)
        rng.shuffle(o)
        k = max(int(len(o) * 0.25), 2)
        ho = set(o[:k])
        hm = np.array([a in ho for a in auth])
        fi, hi = np.flatnonzero(~hm), np.flatnonzero(hm)
        if len(fi) < 40 or len(hi) < 15:
            continue

        def build(qq, idx):
            if qq <= 0:
                return [C[i] for i in idx]
            t = np.quantile(top[fi], qq)
            return [U[i] if top[i] <= t else C[i] for i in idx]

        bq = max(QG, key=lambda z: pf1([gold_ev[i] for i in fi],
                                       build(z, fi)))
        gh = [gold_ev[i] for i in hi]
        ds.append(pf1(gh, build(bq, hi)) - pf1(gh, [C[i] for i in hi]))
        qs.append(bq)
    d = np.array(ds)
    print(f"  d evidence {d.mean():+.4f}   d comp {0.3*d.mean():+.4f}   "
          f"win {np.mean(d > 0):.0%}   LOSS {np.mean(d < 0):.0%}   "
          f"sd {0.3*d.std():.4f}")
    print(f"  q chosen {np.mean(qs):.2f} +- {np.std(qs):.2f}")
    print("\n  A wandering q is the signature that killed the factor shrinkage")
    print("  (k = 65 +- 100) and the evidence hedge (q = 0.00 every split).")
    print("  Oracle for reference: converting the evidence gap through the")
    print("  0.30 evidence weight gives the composite ceiling.")


if __name__ == "__main__":
    main()
