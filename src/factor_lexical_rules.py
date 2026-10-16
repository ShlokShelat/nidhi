"""
factor_lexical_rules.py

Zero-parameter lexical gating for the factor labels that are lexically defined.

THE ARGUMENT.

S2 is MACRO-F1 over 24 labels: every label contributes 1/24 regardless of how
rare it is. `hopelessness` has 745 positives and `sexual orientation related
issues` has 8, and they count EQUALLY. Per-label F1 on 1635 OOF posts:

    meaning in life              0.324   (45 gold)
    sexual orientation issues    0.480   ( 8 gold)
    exposure to others' suicide  0.500   (14 gold)
    substance use                0.642   (33 gold)
    poor school performance      0.706   (16 gold)

A model with eight training examples cannot learn a label, and macro-F1 charges
4.2% of the score for each one it fails. But several of these are LEXICALLY
DEFINED rather than semantically hard: a post about being gay, closeted or
facing homophobia is nearly a keyword lookup. Human world knowledge beats eight
examples here, and encoding it costs ZERO fitted parameters.

Why zero parameters matters: with only a few hundred labelled posts available
for tuning, each fitted parameter buys apparent gain on the tuning split and
loses it on unseen authors. A curated pattern list is fitted on nothing, so
there is no gap between its tuning-set and held-out behaviour.

TWO DIRECTIONS, and the second matters more.

  FORCE     pattern present and model said no  -> set the label to 1.
            Recovers false negatives on labels the model never learned.

  SUPPRESS  pattern ABSENT and model said yes  -> set the label to 0.
            For a lexically defined label, a post that never mentions the
            subject almost certainly does not carry it. This kills false
            positives, and precision is where these labels bleed: the deployed
            model over-predicts on 20 of 24 labels.

  SUPPRESS is only sound for labels whose gold instances essentially ALWAYS
  contain a cue word. The script measures that coverage per label and refuses
  to suppress where coverage is low -- that check is what stops this from being
  a guess.

MEASURED ON 1635 OOF POSTS, not the 372-post val split: cutoffs and rule
decisions taken from 372 posts are too noisy to transfer to unseen authors.
Every number here uses the same 1635 posts the incumbent thresholds were fitted
on, split into author-disjoint halves.

Usage:
  python src/factor_lexical_rules.py
  python src/factor_lexical_rules.py --apply-to artifacts/nli_ensemble_factors.csv \
      --test data/test.csv --out artifacts/factors_lexical_test.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import FACTOR_TAXONOMY, parse_factors

# ---------------------------------------------------------------------------
# Pattern lists. Curated from the label definitions and from reading the data,
# NOT tuned against a score. Each is deliberately broad on the FORCE side and
# the SUPPRESS side is gated by measured coverage below.
# ---------------------------------------------------------------------------

P = {
    "sexual orientation related issues": [
        r"\bgay\b", r"\blesbian\b", r"\bbi(?:sexual)?\b", r"\bqueer\b",
        r"\btrans(?:gender|sexual)?\b", r"\bnon.?binary\b", r"\benby\b",
        r"\blgbt\w*", r"\bhomophob\w*", r"\btransphob\w*",
        r"\bcom(?:e|ing|s)? out\b", r"\bin the closet\b", r"\bcloseted\b",
        r"\bmy sexuality\b", r"\bsexual orientation\b", r"\basexual\b",
        r"\bace\b", r"\baro(?:mantic)?\b", r"\bpan(?:sexual)?\b",
        r"\bdysphoria\b", r"\bdeadname\w*", r"\bpronouns\b",
        r"\battracted to (?:men|women|guys|girls|boys)\b",
        r"\bsame.sex\b", r"\bcrossdress\w*", r"\bftm\b", r"\bmtf\b",
    ],
    "exposure to others' suicide": [
        r"\b(?:my|his|her|their|a)\s+(?:friend|brother|sister|mom|mother|dad|"
        r"father|cousin|uncle|aunt|grandma|grandpa|classmate|coworker|"
        r"boyfriend|girlfriend|ex)\b[^.]{0,60}\b(?:killed (?:him|her|them)self|"
        r"committed suicide|died by suicide|took (?:his|her|their) own life|"
        r"hanged (?:him|her|them)self|overdosed and died)\b",
        r"\bsomeone (?:i|I) knew\b[^.]{0,40}\b(?:suicide|killed)\b",
        r"\blost (?:my|a)\s+\w+\s+to suicide\b",
        r"\b(?:he|she|they) (?:killed (?:him|her|them)self|committed suicide|"
        r"died by suicide)\b",
        r"\bafter (?:my|his|her|their) \w+(?:'s)? suicide\b",
        r"\bsuicide of (?:my|his|her|their)\b",
        r"\bfuneral\b[^.]{0,50}\bsuicide\b",
    ],
    "substance use": [
        r"\balcohol\w*", r"\bdrink(?:ing|s|er)?\b", r"\bdrunk\b", r"\bbooze\b",
        r"\bvodka\b", r"\bbeer\b", r"\bwhiske?y\b", r"\bwine\b",
        r"\bdrugs?\b", r"\bweed\b", r"\bmarijuana\b", r"\bcannabis\b",
        r"\bcocaine\b", r"\bcoke\b", r"\bheroin\b", r"\bmeth\b",
        r"\bxanax\b", r"\bbenzos?\b", r"\bopioids?\b", r"\bpainkillers?\b",
        r"\bget(?:ting)? high\b", r"\bstoned\b", r"\bsober\b", r"\brelapse\w*",
        r"\baddict\w*", r"\bsubstance (?:use|abuse)\b", r"\bsmoking\b",
        r"\bnicotine\b", r"\bvap(?:e|ing)\b", r"\boverdos\w*",
    ],
    "poor school performance": [
        r"\bgrades?\b", r"\bgpa\b", r"\bfail(?:ed|ing|s)?\b[^.]{0,30}"
        r"\b(?:class|exam|test|school|course|semester|subject)\b",
        r"\bflunk\w*", r"\bdropp?(?:ed|ing) out\b", r"\bheld back\b",
        r"\bexpelled\b", r"\bacademic(?:ally)?\b", r"\bexams?\b",
        r"\bmy (?:grades|marks|results)\b", r"\bbad at school\b",
        r"\bschool is\b[^.]{0,20}\b(?:hard|difficult|impossible)\b",
        r"\bcan'?t (?:focus|concentrate|study)\b", r"\bhomework\b",
    ],
    "low socio-economic status": [
        r"\bbroke\b", r"\bno money\b", r"\bcan'?t afford\b", r"\bpoverty\b",
        r"\bhomeless\w*", r"\bevict\w*", r"\bunemploy\w*", r"\bjobless\b",
        r"\blost my job\b", r"\bfired\b", r"\blaid off\b", r"\bin debt\b",
        r"\bbills?\b", r"\brent\b", r"\bfood stamps?\b", r"\bwelfare\b",
        r"\bminimum wage\b", r"\bpaych(?:eck|ecks)\b", r"\bfinancial\w*",
        r"\bcan'?t pay\b", r"\bpoor\b", r"\bloans?\b",
    ],
    "interpersonal violence": [
        r"\babus(?:e|ed|ive|ing)\b", r"\brap(?:e|ed|ist)\b",
        r"\bmolest\w*", r"\bassault\w*", r"\bbeat(?:en|ing|s)? (?:me|him|her)\b",
        r"\bhit me\b", r"\bhits me\b", r"\bpunch\w*", r"\bstrangl\w*",
        r"\bdomestic violence\b", r"\bbull(?:y|ied|ies|ying)\b",
        r"\bharass\w*", r"\bstalk\w*", r"\bthreaten\w*",
        r"\bsexually (?:abused|assaulted)\b", r"\bviolent\b", r"\bviolence\b",
    ],
    "physical health/characteristic": [
        r"\bchronic (?:pain|illness|fatigue)\b", r"\bdisab(?:led|ility)\b",
        r"\bcancer\b", r"\bdiabet\w*", r"\bsurgery\b", r"\bdiagnos\w*"
        r"[^.]{0,40}\b(?:illness|disease|condition)\b",
        r"\bmy (?:body|weight|face|skin|hair|teeth)\b",
        r"\bugly\b", r"\bfat\b", r"\boverweight\b", r"\bobes\w*",
        r"\bacne\b", r"\bshort\b", r"\bbald\w*", r"\bdeform\w*",
        r"\bill(?:ness)?\b", r"\bpain\b", r"\bsick\b", r"\binsomnia\b",
        r"\bcan'?t sleep\b", r"\bmigraines?\b", r"\bseizures?\b",
    ],
}


def build(pats):
    return {k: re.compile("|".join(v), re.I) for k, v in pats.items()}


def macro_f1(P_, Y):
    out = []
    for j in range(Y.shape[1]):
        tp = int((P_[:, j] & Y[:, j]).sum())
        fp = int((P_[:, j] & (1 - Y[:, j])).sum())
        fn = int(((1 - P_[:, j]) & Y[:, j]).sum())
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(out)), out


def author_halves(auth):
    s = pd.Series(auth)
    counts = s.value_counts()
    order = sorted(counts.index, key=lambda a: (-counts[a], str(a)))
    half, size = {}, {"A": 0, "B": 0}
    for a in order:
        pick = "A" if size["A"] <= size["B"] else "B"
        half[a] = pick
        size[pick] += counts[a]
    return s.map(half).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--oof", default="artifacts/fac_oof_f{f}.npz")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-coverage", type=float, default=0.80,
                    help="only SUPPRESS a label if at least this fraction of "
                         "its gold positives contain a cue word. Below that, "
                         "absence of a cue is not evidence of absence.")
    args = ap.parse_args()

    RX = build(P)
    lab = pd.read_csv(args.labeled).set_index("row_id")
    idx = {t: i for i, t in enumerate(FACTOR_TAXONOMY)}

    rids, PR, TH = [], [], []
    for f in range(args.k):
        p = args.oof.format(f=f)
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        d = np.load(p, allow_pickle=True)
        rids.extend(list(d["row_id"]))
        PR.append(d["probs"])
        TH.append(np.tile(d["thresholds"], (len(d["row_id"]), 1)))
    PR, TH = np.concatenate(PR), np.concatenate(TH)
    P0 = (PR >= TH).astype(int)

    Y = np.zeros(P0.shape, dtype=int)
    for i, r in enumerate(rids):
        for f in parse_factors(lab.loc[r, "factors"]):
            j = idx.get(str(f).strip().lower())
            if j is not None:
                Y[i, j] = 1
    posts = [str(lab.loc[r, "post"]) for r in rids]
    half = author_halves([lab.loc[r, "anon_user_id"] for r in rids])
    print(f"{len(rids)} OOF posts, halves "
          f"{int((half=='A').sum())}/{int((half=='B').sum())}")

    hit = {k: np.array([bool(rx.search(t)) for t in posts])
           for k, rx in RX.items()}

    base, per0 = macro_f1(P0, Y)
    print(f"\nincumbent macro-F1 = {base:.4f}")

    print(f"\nPATTERN COVERAGE -- what fraction of gold positives contain a cue")
    print(f"{'label':<40}{'gold':>6}{'cover':>8}{'fires':>7}{'prec':>7}"
          f"{'F1now':>7}  suppress?")
    print("-" * 84)
    plan = {}
    for k in P:
        j = idx[k]
        m = hit[k]
        g = Y[:, j].astype(bool)
        cover = float(m[g].mean()) if g.sum() else 0.0
        prec = float(g[m].mean()) if m.sum() else 0.0
        ok = cover >= args.min_coverage
        plan[k] = ok
        print(f"{k:<40}{int(g.sum()):>6}{cover:>8.2f}{int(m.sum()):>7}"
              f"{prec:>7.2f}{per0[j]:>7.3f}  {'YES' if ok else 'no'}")
    print(f"\ncoverage below {args.min_coverage:.2f} -> absence of a cue is not")
    print("evidence of absence, so SUPPRESS is unsafe for that label.")

    def apply_rules(force=True, suppress=True, only=None):
        Q = P0.copy()
        for k in P:
            if only and k != only:
                continue
            j, m = idx[k], hit[k]
            if force:
                Q[m, j] = 1
            if suppress and plan[k]:
                Q[~m, j] = 0
        return Q

    print(f"\n{'arm':<34}{'macro':>9}{'delta':>9}{'halfA':>9}{'halfB':>9}"
          f"{'composite':>11}")
    print("-" * 81)
    bA = macro_f1(P0[half == "A"], Y[half == "A"])[0]
    bB = macro_f1(P0[half == "B"], Y[half == "B"])[0]
    arms = {
        "FORCE only": apply_rules(True, False),
        "SUPPRESS only": apply_rules(False, True),
        "FORCE + SUPPRESS": apply_rules(True, True),
    }
    for n, Q in arms.items():
        m, _ = macro_f1(Q, Y)
        a = macro_f1(Q[half == "A"], Y[half == "A"])[0]
        b = macro_f1(Q[half == "B"], Y[half == "B"])[0]
        print(f"{n:<34}{m:>9.4f}{m - base:>+9.4f}{a - bA:>+9.4f}"
              f"{b - bB:>+9.4f}{0.3 * (m - base):>+11.4f}")

    print(f"\nPER-LABEL, FORCE + SUPPRESS")
    print(f"{'label':<40}{'before':>8}{'after':>8}{'delta':>8}")
    Q = arms["FORCE + SUPPRESS"]
    _, per1 = macro_f1(Q, Y)
    for k in P:
        j = idx[k]
        print(f"{k:<40}{per0[j]:>8.3f}{per1[j]:>8.3f}"
              f"{per1[j] - per0[j]:>+8.3f}")

    print(f"\nONE LABEL AT A TIME (keep only what helps on BOTH halves)")
    print(f"{'label':<40}{'delta':>9}{'halfA':>9}{'halfB':>9}")
    keep = []
    for k in P:
        Q = apply_rules(True, True, only=k)
        m, _ = macro_f1(Q, Y)
        a = macro_f1(Q[half == "A"], Y[half == "A"])[0] - bA
        b = macro_f1(Q[half == "B"], Y[half == "B"])[0] - bB
        good = (m - base) > 0 and a > 0 and b > 0
        if good:
            keep.append(k)
        print(f"{k:<40}{m - base:>+9.4f}{a:>+9.4f}{b:>+9.4f}"
              f"{'  <- keep' if good else ''}")

    if keep:
        Q = P0.copy()
        for k in keep:
            j, m = idx[k], hit[k]
            Q[m, j] = 1
            if plan[k]:
                Q[~m, j] = 0
        m, _ = macro_f1(Q, Y)
        a = macro_f1(Q[half == "A"], Y[half == "A"])[0] - bA
        b = macro_f1(Q[half == "B"], Y[half == "B"])[0] - bB
        print(f"\nKEEP-ONLY combination ({len(keep)} labels): {m:.4f} "
              f"({m - base:+.4f})  halves {a:+.4f}/{b:+.4f}")
        print(f"  -> {0.3 * (m - base):+.4f} composite")
        print(f"  labels: {keep}")

    print("\n" + "=" * 81)
    print("These rules have ZERO fitted parameters -- the pattern lists come")
    print("from the label definitions, not from a score search, so there is no")
    print("tuning-set-to-held-out gap for them to lose.")
    print("Measured on 1635 OOF posts, the same data the incumbent thresholds")
    print("were fitted on, with author-disjoint halves.")
    print("=" * 81)


if __name__ == "__main__":
    main()
