"""
indicator_stack_ship.py

Apply the indicator stack fitted by `src/indicator_stack.py` to an existing
submission and write the resulting file.

The stack is a logistic model for P(gold evidence empty) over the risk logits
plus evidence-tagger features; see indicator_stack.py for why the tagger
features carry indicator information the risk logits do not. This script fits
it on ALL of the OOF data and applies it to test.

    decouple  keep the base file's risk_level, use the stack ONLY to decide
              evidence suppression. ONE column changes, S1's risk term is
              bit-identical, so the delta is attributable.
    couple    also set the risk label to `indicator` when the stack fires.
              Still ONE decision -- the scorer just charges it to two
              components.
    add-only  keep every span the base file emits and additionally empty the
              posts the stack flags.

THE PROVENANCE GATE. Which tagger arm and directory produced the base file's
evidence column is not recorded in the file, so decouple and couple RECONSTRUCT
that column from the test candidates plus the base file's risk labels and
compare. If no reconstruction matches exactly the script REFUSES to write: a
mismatch would ship a rebuild bug wearing the intended change's clothes, and the
resulting score would be uninterpretable.

Usage:
  python src/indicator_stack_ship.py                     # diagnose only
  python src/indicator_stack_ship.py --build-decouple submissions/NIDHI_indstk.csv
  python src/indicator_stack_ship.py --build-couple   submissions/NIDHI_indcpl.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (normalize_label, normalize_span, parse_evidence,  # noqa: E402
                     token_len)

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
PCOL = {c: f"prob_{c}" for c in LEVELS}
TAGS = ["v2", "ep12", "lr3", "bs16", "ml256"]
MINCONF = {"v2": 0.75, "ep12": 0.75, "lr3": 0.75, "bs16": 0.00, "ml256": 0.65}
DIRS = ["artifacts/evidence_runs_canon", "artifacts/evidence_runs"]


def dedupe(spans):
    out = []
    for s in sorted(spans, key=lambda x: (-token_len(x),
                                          -len(normalize_span(x)))):
        n = normalize_span(s)
        if not n or any(n in normalize_span(k) for k in out):
            continue
        out.append(s)
    return out


def logit(x):
    x = np.clip(np.asarray(x, float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def feats(per_tag, rows, posts):
    """(n, 5) tagger features + the per-post cluster records."""
    F, REC = [], []
    for r, po in zip(rows, posts):
        allc = [(t, s, c) for t in per_tag for s, c in per_tag[t].get(r, [])]
        cnt = {}
        for t, s, c in allc:
            k = normalize_span(s)
            cnt.setdefault(k, [set(), s])
            cnt[k][0].add(t)
        cs = [c for _, _, c in allc]
        F.append([len(cnt), 1.0 if len(cnt) == 0 else 0.0,
                  max(cs) if cs else 0.0,
                  float(np.mean(cs)) if cs else 0.0,
                  np.log1p(len(str(po).split()))])
        REC.append(cnt)
    return np.asarray(F, float), REC


def vote_spans(REC, per_tag, rows, k=2):
    out = []
    for i, r in enumerate(rows):
        keep = dedupe([v[1] for v in REC[i].values() if len(v[0]) >= k])
        if not keep:
            pool = [s for t in per_tag for s, _ in per_tag[t].get(r, [])]
            keep = sorted(pool, key=token_len)[:1]
        out.append(keep)
    return out


def load_tags(d, rows, split):
    per = {}
    for t in TAGS:
        cands = ([f"{d}/evidence_{t}_{split}_candidates.csv"] if split == "test"
                 else [f"{d}/{t}_oof_candidates.csv"])
        p = next((c for c in cands if os.path.exists(c)), None)
        if p is None:
            return None
        x = pd.read_csv(p)
        mc = (float(x["min_conf"].iloc[0]) if "min_conf" in x.columns
              else MINCONF[t])
        x = x[x["conf"] >= mc]
        m = {}
        for r, s, c in zip(x["row_id"].astype(str), x["span"], x["conf"]):
            m.setdefault(r, []).append((str(s), float(c)))
        per[t] = m
    return per


def as_sets(col):
    return [frozenset(normalize_span(s) for s in parse_evidence(str(c)))
            for c in col]


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--oof", default="artifacts/champion_oof.csv")
    a_.add_argument("--labeled", default="artifacts/full_train.csv")
    a_.add_argument("--oofdir", default="artifacts/evidence_oof")
    a_.add_argument("--test", default="data/test.csv")
    a_.add_argument("--testrisk", default="artifacts/risk_champion_qwen.csv")
    a_.add_argument("--champion", default="submissions/NIDHI_factfit.csv")
    a_.add_argument("--t", type=float, default=0.35)
    a_.add_argument("--build-decouple", default=None)
    a_.add_argument("--build-couple", default=None)
    a_.add_argument("--build-add", default=None,
                    help="ADD-ONLY: keep every span the base file emits, empty "
                         "the posts the stack flags. Needs no reconstruction, "
                         "so it works even when the base file's risk-"
                         "conditioned tagger cannot be re-queried.")
    a_.add_argument("--featdir", default="artifacts/evidence_runs",
                    help="test candidates for the STACK FEATURES. Must be the "
                         "same unconditioned taggers the stack was fitted on, "
                         "NOT the risk-conditioned arm the base file ships.")
    args = a_.parse_args()

    from sklearn.linear_model import LogisticRegression

    # ---------------- fit the stack on ALL of OOF --------------------------
    oof = pd.read_csv(args.oof, keep_default_na=False)
    oof["row_id"] = oof["row_id"].astype(str)
    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    lab = lab.set_index("row_id")
    rows = [r for r in oof["row_id"] if r in lab.index]
    ix = oof.set_index("row_id")
    perO = load_tags(args.oofdir, rows, "oof")
    if perO is None:
        raise SystemExit("missing OOF candidate files")
    PO = np.stack([ix.loc[rows, PCOL[c]].astype(float).to_numpy()
                   for c in LEVELS], axis=1)
    PO = PO / np.clip(PO.sum(1, keepdims=True), 1e-9, None)
    FO, _ = feats(perO, rows, [lab.loc[r, "post"] for r in rows])
    XO = np.column_stack([logit(PO), FO])
    yO = np.array([len(parse_evidence(lab.loc[r, "evidence"])) == 0
                   for r in rows])
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(XO, yO)
    print(f"stack fitted on {len(rows)} OOF posts, "
          f"{yO.mean():.1%} gold-empty")

    # ---------------- test side --------------------------------------------
    te = pd.read_csv(args.test, keep_default_na=False)
    te["row_id"] = te["row_id"].astype(str)
    ch = pd.read_csv(args.champion, keep_default_na=False)
    ch["row_id"] = ch["row_id"].astype(str)
    rc = "risk_level" if "risk_level" in ch.columns else "pred_risk_level"
    trows = list(ch["row_id"])
    pmap = dict(zip(te["row_id"], te["post"]))
    ship_risk = np.array([normalize_label(x) for x in ch[rc]])
    ship_ev = as_sets(ch["evidence"])

    # ---- ADD-ONLY needs no reconstruction -------------------------------
    # Emptying a post's evidence requires nothing from the tagger that wrote
    # it. The champion's spans come from a RISK-CONDITIONED tagger we cannot
    # re-query offline, which blocks decouple/couple -- but never add-only.
    # Measured: FITTED add-only +0.0009, 70% win, t=0.35 in 158/200 splits.
    if args.build_add:
        per = load_tags(args.featdir, trows, "test")
        if per is None:
            raise SystemExit(f"missing test candidate files in {args.featdir}")
        FT, _ = feats(per, trows, [pmap.get(r, "") for r in trows])
        tr = pd.read_csv(args.testrisk, keep_default_na=False)
        tr["row_id"] = tr["row_id"].astype(str)
        tr = tr.set_index("row_id")
        PT = np.stack([tr.loc[trows, PCOL[c]].astype(float).to_numpy()
                       for c in LEVELS], axis=1)
        PT = PT / np.clip(PT.sum(1, keepdims=True), 1e-9, None)
        agree = float((np.array(LEVELS)[PT.argmax(1)] == ship_risk).mean())
        print(f"\n{args.testrisk} argmax agrees with the champion's risk "
              f"labels on {agree:.1%} of posts")
        if agree < 0.90:
            raise SystemExit(
                "REFUSING: that is too low for the same model before rules. "
                "The stack's z_* features would come from a DIFFERENT risk "
                "model than it was fitted on.")
        q = lr.predict_proba(np.column_stack([logit(PT), FT]))[:, 1]
        cur = ship_risk == "indicator"
        new = cur | (q >= args.t)
        add = int((new & ~cur).sum())
        empt = np.array([len(s) == 0 for s in ship_ev])
        print(f"  champion suppresses {int(cur.sum())}, "
              f"add-only suppresses {int(new.sum())}  (+{add})")
        print(f"  of the {add} newly emptied, {int((new & ~cur & ~empt).sum())}"
              f" currently carry spans")
        out = ch.copy()
        out["evidence"] = ["" if s else e
                           for s, e in zip(new, ch["evidence"])]
        os.makedirs(os.path.dirname(args.build_add) or ".", exist_ok=True)
        out.to_csv(args.build_add, index=False)
        chg = int(sum(1 for a, b in zip(as_sets(out["evidence"]), ship_ev)
                      if a != b))
        print(f"\nwrote {args.build_add}   evidence changed on {chg} posts")
        print("  risk_level and factors are BYTE-IDENTICAL -- S2 must return")
        print("  exactly 0.6908 and only the evidence term of S1 moves.")
        print("  expected +0.0005..+0.0009 composite, 70% win, one-draw "
              "[-0.0024,+0.0042]")
        return

    print("\nPROVENANCE -- rebuild the champion's evidence column")
    best = None
    for d in DIRS:
        per = load_tags(d, trows, "test")
        if per is None:
            print(f"  {d:<34} missing a tagger file")
            continue
        _, REC = feats(per, trows, [pmap.get(r, "") for r in trows])
        for k in (2, 3):
            U = vote_spans(REC, per, trows, k)
            rebuilt = [frozenset() if ship_risk[i] == "indicator"
                       else frozenset(normalize_span(s) for s in U[i])
                       for i in range(len(trows))]
            same = sum(1 for a, b in zip(rebuilt, ship_ev) if a == b)
            print(f"  {d:<34} cluster>={k}  {same}/{len(trows)} posts "
                  f"identical")
            if best is None or same > best[0]:
                best = (same, d, k, per, REC, U)

    same, d, k, per, REC, U = best
    if same != len(trows):
        raise SystemExit(
            f"\nREFUSING: best reconstruction matches only {same}/{len(trows)} "
            f"posts. The champion's evidence column came from an arm this "
            f"script does not reproduce, so the new file would mix a rebuild "
            f"bug with the intended change. Pass the right --oofdir/dir or "
            f"identify the arm first.")
    print(f"  EXACT match with {d} cluster>={k}\n")

    tr = pd.read_csv(args.testrisk, keep_default_na=False)
    tr["row_id"] = tr["row_id"].astype(str)
    tr = tr.set_index("row_id")
    if not all(c in tr.columns for c in PCOL.values()):
        raise SystemExit(f"{args.testrisk} lacks prob_* columns")
    PT = np.stack([tr.loc[trows, PCOL[c]].astype(float).to_numpy()
                   for c in LEVELS], axis=1)
    PT = PT / np.clip(PT.sum(1, keepdims=True), 1e-9, None)
    FT, _ = feats(per, trows, [pmap.get(r, "") for r in trows])
    XT = np.column_stack([logit(PT), FT])
    q = lr.predict_proba(XT)[:, 1]
    sup = q >= args.t

    cur = ship_risk == "indicator"
    print(f"t = {args.t}")
    print(f"  champion suppresses {int(cur.sum())} posts, "
          f"stack suppresses {int(sup.sum())}")
    print(f"  newly suppressed {int((sup & ~cur).sum())}   "
          f"un-suppressed {int((~sup & cur).sum())}   "
          f"changed {int((sup != cur).sum())}")

    def write(path, risk, supmask):
        out = ch.copy()
        out["evidence"] = ["" if s else "; ".join(U[i])
                           for i, s in enumerate(supmask)]
        if risk is not None:
            seen = {normalize_label(x): x for x in ch[rc]}
            out[rc] = [seen.get(x, x) for x in risk]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        out.to_csv(path, index=False)
        ev_ch = int(sum(1 for a, b in zip(as_sets(out["evidence"]), ship_ev)
                        if a != b))
        rk_ch = int(sum(1 for a, b in zip(out[rc], ch[rc])
                        if normalize_label(a) != normalize_label(b)))
        print(f"\nwrote {path}")
        print(f"  evidence changed on {ev_ch} posts, risk on {rk_ch}")
        return out

    if args.build_decouple:
        write(args.build_decouple, None, sup)
        print("  risk_level UNCHANGED -> the risk term of S1 is bit-identical")
        print("  and S2 (factors) is untouched. One variable.")
        print("  expected +0.0011 composite, 73% win, one-draw "
              "[-0.0031,+0.0052]")

    if args.build_couple:
        other = np.array(LEVELS[1:])[PT[:, 1:].argmax(1)]
        newr = np.where(sup, "indicator", other)
        write(args.build_couple, newr, sup)
        print("  BOTH risk_level and evidence move, but from ONE decision.")
        print("  S2 (factors) is still untouched and must return 0.6908.")
        print("  expected +0.0046 composite, 78% win, one-draw "
              "[-0.0075,+0.0166]")
        agree = int((newr == ship_risk).sum())
        print(f"  risk labels agreeing with the champion: {agree}/{len(trows)}")
        if agree < len(trows) * 0.9:
            print("  WARNING: the coupled risk column diverges from the")
            print("  champion on >10% of posts. That is far more than the")
            print("  indicator decision alone should move -- check that "
                  f"{args.testrisk} is the champion's own risk model before "
                  "submitting.")


if __name__ == "__main__":
    main()
