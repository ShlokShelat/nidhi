"""Gold-only structural analysis of the 24 factor labels.

Question: is there exploitable STRUCTURE among the 24 labels that a rare label
could borrow ranking signal from?  Everything here is computed on GOLD
annotations only -- no model output is read, no post text leaves this machine.

Run:  python src/factor_label_structure.py
"""

from __future__ import annotations

import os
import sys
import math
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scoring import FACTOR_TAXONOMY, parse_factors  # noqa: E402

PROTECTIVE_FACTORS = {
    "social support",
    "coping strategy",
    "psychological capital",
    "sense of responsibility",
    "meaning in life",
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "artifacts", "full_train.csv")

RARE_CUT = 60          # "rare" label support threshold
N_TAX = len(FACTOR_TAXONOMY)
SEED = 0

BAR = "=" * 78
SUB = "-" * 78


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def section(title: str) -> None:
    print()
    print(BAR)
    print(title)
    print(BAR)


def short(name: str, width: int = 34) -> str:
    return name if len(name) <= width else name[: width - 1] + "~"


def load() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(TRAIN)
    idx = {t: k for k, t in enumerate(FACTOR_TAXONOMY)}
    Y = np.zeros((len(df), N_TAX), dtype=np.int8)
    unknown = defaultdict(int)
    for r, cell in enumerate(df["factors"].tolist()):
        for lab in parse_factors(cell):
            lab = lab.strip().lower()
            if lab in idx:
                Y[r, idx[lab]] = 1
            else:
                unknown[lab] += 1
    if unknown:
        print("[warn] labels outside the taxonomy (ignored):")
        for k, v in sorted(unknown.items(), key=lambda x: -x[1]):
            print("       %-45s %d" % (short(k, 45), v))
    return df, Y


def phi_matrix(Y: np.ndarray) -> np.ndarray:
    """Pairwise phi (= Pearson r on binaries). Zero-variance -> 0."""
    n, k = Y.shape
    X = Y.astype(np.float64)
    mu = X.mean(0)
    sd = X.std(0)
    sd_safe = np.where(sd == 0, 1.0, sd)
    Z = (X - mu) / sd_safe
    P = (Z.T @ Z) / n
    dead = sd == 0
    P[dead, :] = 0.0
    P[:, dead] = 0.0
    np.fill_diagonal(P, 1.0)
    return P


def best_f1_from_scores(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Oracle-threshold best F1 (sweeps every distinct score)."""
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    kk = np.arange(1, len(ys) + 1)
    npos = y.sum()
    if npos == 0:
        return 0.0, 1.0
    prec = tp / kk
    rec = tp / npos
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    b = int(np.argmax(f1))
    return float(f1[b]), float(s[order][b])


# ----------------------------------------------------------------------
# 1 + 2. co-occurrence, implications
# ----------------------------------------------------------------------

def analyse_cooccurrence(Y: np.ndarray) -> np.ndarray:
    n = Y.shape[0]
    sup = Y.sum(0)
    base = sup / n
    co = (Y.T.astype(np.float64) @ Y.astype(np.float64))   # co[i, j] = #(i and j)

    section("1. SUPPORT AND BASE RATES  (n = %d posts)" % n)
    print("%-38s %7s %8s %6s" % ("label", "support", "base", "prot"))
    print(SUB)
    for j in np.argsort(-sup):
        name = FACTOR_TAXONOMY[j]
        tag = "P" if name in PROTECTIVE_FACTORS else ""
        print("%-38s %7d %8.4f %6s" % (short(name, 38), sup[j], base[j], tag))

    # conditional matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = co / np.maximum(sup[:, None], 1)     # cond[i, j] = P(j | i)
        lift = cond / np.maximum(base[None, :], 1e-12)
    np.fill_diagonal(cond, np.nan)
    np.fill_diagonal(lift, np.nan)

    section("1b. STRONGEST IMPLICATIONS  P(j|i), ranked by P")
    rows = []
    for i in range(N_TAX):
        for j in range(N_TAX):
            if i == j or sup[i] < 8:
                continue
            rows.append((cond[i, j], lift[i, j], co[i, j], i, j))
    rows.sort(key=lambda t: -t[0])
    hdr = "%-32s -> %-32s %6s %6s %6s %6s" % (
        "IF label i (sup)", "THEN label j (sup)", "P(j|i)", "base", "lift", "n_co")
    print(hdr)
    print(SUB)
    for p, lf, nco, i, j in rows[:25]:
        print("%-26s(%4d) -> %-26s(%4d) %6.3f %6.3f %6.2f %6d" % (
            short(FACTOR_TAXONOMY[i], 26), sup[i],
            short(FACTOR_TAXONOMY[j], 26), sup[j],
            p, base[j], lf, int(nco)))

    section("1c. STRONGEST LIFTS  (both labels support >= 8, n_co >= 5)")
    rows2 = [r for r in rows if r[2] >= 5 and sup[r[4]] >= 8]
    rows2.sort(key=lambda t: -t[1])
    print(hdr)
    print(SUB)
    for p, lf, nco, i, j in rows2[:20]:
        print("%-26s(%4d) -> %-26s(%4d) %6.3f %6.3f %6.2f %6d" % (
            short(FACTOR_TAXONOMY[i], 26), sup[i],
            short(FACTOR_TAXONOMY[j], 26), sup[j],
            p, base[j], lf, int(nco)))

    section("1d. RARE <-> COMMON EDGES  (rare = support < %d)" % RARE_CUT)
    rare = set(int(j) for j in range(N_TAX) if sup[j] < RARE_CUT)
    print("rare labels: %d of %d" % (len(rare), N_TAX))
    print()
    print("[A] RARE implies COMMON   (does a rare label predict a common one?)")
    print(hdr)
    print(SUB)
    a = [r for r in rows if r[3] in rare and r[4] not in rare]
    a.sort(key=lambda t: -t[0])
    for p, lf, nco, i, j in a[:10]:
        print("%-26s(%4d) -> %-26s(%4d) %6.3f %6.3f %6.2f %6d" % (
            short(FACTOR_TAXONOMY[i], 26), sup[i],
            short(FACTOR_TAXONOMY[j], 26), sup[j], p, base[j], lf, int(nco)))
    print()
    print("[B] COMMON implies RARE   (THIS is the lane that would help ranking)")
    print(hdr)
    print(SUB)
    b = [r for r in rows if r[3] not in rare and r[4] in rare]
    b.sort(key=lambda t: -t[1])
    for p, lf, nco, i, j in b[:10]:
        print("%-26s(%4d) -> %-26s(%4d) %6.3f %6.3f %6.2f %6d" % (
            short(FACTOR_TAXONOMY[i], 26), sup[i],
            short(FACTOR_TAXONOMY[j], 26), sup[j], p, base[j], lf, int(nco)))

    section("2. NEAR-DETERMINISTIC RULES  P(j|i) > 0.8, support(i) >= 8")
    det = [r for r in rows if r[0] > 0.8 and sup[r[3]] >= 8]
    det.sort(key=lambda t: -t[0])
    if not det:
        print("NONE. No label is an almost-sure trigger for any other label.")
    else:
        print(hdr)
        print(SUB)
        for p, lf, nco, i, j in det:
            print("%-26s(%4d) -> %-26s(%4d) %6.3f %6.3f %6.2f %6d" % (
                short(FACTOR_TAXONOMY[i], 26), sup[i],
                short(FACTOR_TAXONOMY[j], 26), sup[j], p, base[j], lf, int(nco)))
    print()
    print("highest P(j|i) anywhere (sup(i)>=8): %.3f" % (rows[0][0] if rows else float("nan")))

    # "never alone" check
    section("2b. IS ANY LABEL NEVER PRESENT ALONE?")
    card = Y.sum(1)
    print("%-38s %7s %10s %10s" % ("label", "support", "P(alone)", "mean card"))
    print(SUB)
    for j in np.argsort(sup):
        m = Y[:, j] == 1
        if m.sum() == 0:
            continue
        alone = float((card[m] == 1).mean())
        print("%-38s %7d %10.3f %10.2f" % (
            short(FACTOR_TAXONOMY[j], 38), sup[j], alone, float(card[m].mean())))
    return cond


# ----------------------------------------------------------------------
# 3. cardinality
# ----------------------------------------------------------------------

def analyse_cardinality(Y: np.ndarray) -> None:
    section("3. CARDINALITY  (distinct labels per post)")
    card = Y.sum(1)
    n = len(card)
    print("mean   %.3f" % card.mean())
    print("median %.1f" % np.median(card))
    print("sd     %.3f" % card.std())
    print("min    %d" % card.min())
    print("max    %d" % card.max())
    print("sum of base rates (expected card if independent) = %.3f" % Y.mean(0).sum())
    print()
    print("%5s %7s %8s %8s" % ("k", "posts", "frac", "cum"))
    print(SUB)
    cum = 0.0
    for k in range(0, int(card.max()) + 1):
        c = int((card == k).sum())
        if c == 0:
            continue
        cum += c / n
        print("%5d %7d %8.4f %8.4f" % (k, c, c / n, cum))

    # how well does a fixed-k or prevalence-matched decode fit?
    section("3b. FIXED-k vs PREVALENCE-MATCHED DECODE, GOLD CALIBRATION")
    for k in range(1, 8):
        frac = float((card == k).mean())
        # macro recall ceiling for a fixed-k decode is bounded by k/card for
        # posts with card > k; report the label-mass a fixed-k decode can reach
        reachable = np.minimum(card, k).sum() / max(card.sum(), 1)
        print("fixed k=%d : %5.1f%% of posts have exactly k, "
              "recall ceiling on gold label mass = %.3f" % (k, 100 * frac, reachable))
    print()
    print("A prevalence-matched decode fires sum(base) = %.2f labels/post on average," % Y.mean(0).sum())
    print("gold mean is %.2f -- these are the same number BY CONSTRUCTION" % card.mean())
    print("(sum of column means == mean of row sums), so 'prevalence-matched'")
    print("is calibrated in AGGREGATE by definition and says nothing per-post.")
    print("Per-post spread is what matters: sd = %.2f, IQR = [%d, %d]." % (
        card.std(), int(np.percentile(card, 25)), int(np.percentile(card, 75))))


# ----------------------------------------------------------------------
# 4. protective vs risk group structure
# ----------------------------------------------------------------------

def analyse_groups(Y: np.ndarray) -> np.ndarray:
    section("4. GROUP STRUCTURE: protective vs risk (phi coefficients)")
    P = phi_matrix(Y)
    prot = [i for i, t in enumerate(FACTOR_TAXONOMY) if t in PROTECTIVE_FACTORS]
    risk = [i for i, t in enumerate(FACTOR_TAXONOMY) if t not in PROTECTIVE_FACTORS]
    print("protective: %d labels, risk: %d labels" % (len(prot), len(risk)))

    def mean_off(idx_a, idx_b, same: bool) -> tuple[float, float, int]:
        vals = []
        for i in idx_a:
            for j in idx_b:
                if same and j <= i:
                    continue
                if not same and i == j:
                    continue
                vals.append(P[i, j])
        v = np.array(vals)
        return float(v.mean()), float(np.abs(v).mean()), len(v)

    pp, pp_abs, n_pp = mean_off(prot, prot, True)
    rr, rr_abs, n_rr = mean_off(risk, risk, True)
    pr, pr_abs, n_pr = mean_off(prot, risk, False)
    print()
    print("%-28s %8s %8s %6s" % ("block", "mean phi", "mean|phi|", "pairs"))
    print(SUB)
    print("%-28s %8.4f %8.4f %6d" % ("protective x protective", pp, pp_abs, n_pp))
    print("%-28s %8.4f %8.4f %6d" % ("risk x risk", rr, rr_abs, n_rr))
    print("%-28s %8.4f %8.4f %6d" % ("protective x risk (between)", pr, pr_abs, n_pr))
    within_n = n_pp + n_rr
    within = (pp * n_pp + rr * n_rr) / within_n
    print("%-28s %8.4f %8s %6d" % ("WITHIN (pooled)", within, "", within_n))
    print("%-28s %8.4f %8s %6d" % ("BETWEEN", pr, "", n_pr))
    print()
    print("within - between = %+.4f" % (within - pr))

    # permutation test on the group assignment
    rng = np.random.default_rng(SEED)
    obs = within - pr
    allidx = list(range(N_TAX))
    null = []
    for _ in range(20000):
        perm = rng.permutation(allidx)
        pa, pb = list(perm[:len(prot)]), list(perm[len(prot):])
        w1, _, na = mean_off(pa, pa, True)
        w2, _, nb = mean_off(pb, pb, True)
        bt, _, _ = mean_off(pa, pb, False)
        null.append((w1 * na + w2 * nb) / (na + nb) - bt)
    null = np.array(null)
    pval = float((np.abs(null) >= abs(obs)).mean())
    print("permutation test over random 5/19 splits of the 24 labels:")
    print("  observed %+.4f, null mean %+.4f, null sd %.4f, two-sided p = %.4f" % (
        obs, null.mean(), null.std(), pval))

    print()
    print("protective-protective pairs in full:")
    print("%-26s %-26s %8s" % ("label i", "label j", "phi"))
    print(SUB)
    for a in range(len(prot)):
        for b in range(a + 1, len(prot)):
            i, j = prot[a], prot[b]
            print("%-26s %-26s %8.4f" % (
                short(FACTOR_TAXONOMY[i], 26), short(FACTOR_TAXONOMY[j], 26), P[i, j]))

    print()
    print("top 12 |phi| pairs overall:")
    print("%-26s %-26s %8s" % ("label i", "label j", "phi"))
    print(SUB)
    pairs = []
    for i in range(N_TAX):
        for j in range(i + 1, N_TAX):
            pairs.append((abs(P[i, j]), P[i, j], i, j))
    pairs.sort(key=lambda t: -t[0])
    for _, v, i, j in pairs[:12]:
        print("%-26s %-26s %8.4f" % (
            short(FACTOR_TAXONOMY[i], 26), short(FACTOR_TAXONOMY[j], 26), v))
    return P


# ----------------------------------------------------------------------
# 5. per-author structure
# ----------------------------------------------------------------------

def analyse_authors(df: pd.DataFrame, Y: np.ndarray) -> None:
    section("5. PER-AUTHOR STRUCTURE  (leave-one-post-out)")
    aid = df["anon_user_id"].astype(str).values
    uniq = sorted(set(aid))
    print("authors: %d, posts: %d, posts/author mean %.1f median %.1f max %d" % (
        len(uniq), len(aid), len(aid) / len(uniq),
        np.median(pd.Series(aid).value_counts().values),
        pd.Series(aid).value_counts().max()))

    groups = defaultdict(list)
    for r, a in enumerate(aid):
        groups[a].append(r)

    # For each label: P(y=1 | author has it in >=1 OTHER post) vs base rate
    print()
    hdrline = ("%-34s %6s %8s %10s %10s %7s %7s" % (
        "label", "sup", "base", "P|auth+", "P|auth-", "lift", "ICC"))
    print(hdrline)
    print(SUB)
    lifts, iccs, sups = [], [], []
    rows_out = []
    for j in range(N_TAX):
        y = Y[:, j].astype(np.float64)
        pos_n = pos_d = neg_n = neg_d = 0
        for a, rs in groups.items():
            if len(rs) < 2:
                continue
            tot = y[rs].sum()
            for r in rs:
                other = tot - y[r]
                if other > 0:
                    pos_d += 1
                    pos_n += y[r]
                else:
                    neg_d += 1
                    neg_n += y[r]
        base = y.mean()
        p_pos = pos_n / pos_d if pos_d else float("nan")
        p_neg = neg_n / neg_d if neg_d else float("nan")
        lift = p_pos / base if base > 0 else float("nan")

        # one-way ANOVA ICC over authors with >= 2 posts
        gm = y.mean()
        ssb = sst = 0.0
        nn = 0
        kbar = []
        for a, rs in groups.items():
            if len(rs) < 2:
                continue
            v = y[rs]
            ssb += len(rs) * (v.mean() - gm) ** 2
            sst += ((v - gm) ** 2).sum()
            nn += len(rs)
            kbar.append(len(rs))
        ssw = sst - ssb
        kgrp = len(kbar)
        if kgrp > 1 and nn - kgrp > 0:
            msb = ssb / (kgrp - 1)
            msw = ssw / (nn - kgrp)
            k0 = (nn - sum(x * x for x in kbar) / nn) / (kgrp - 1)
            icc = (msb - msw) / (msb + (k0 - 1) * msw) if (msb + (k0 - 1) * msw) > 0 else 0.0
        else:
            icc = float("nan")
        sup = int(y.sum())
        print("%-34s %6d %8.4f %10.4f %10.4f %7.2f %7.3f" % (
            short(FACTOR_TAXONOMY[j], 34), sup, base, p_pos, p_neg, lift, icc))
        lifts.append(lift)
        iccs.append(icc)
        sups.append(sup)
        rows_out.append((FACTOR_TAXONOMY[j], sup, base, p_pos, lift, icc))

    lifts = np.array(lifts, dtype=float)
    iccs = np.array(iccs, dtype=float)
    print(SUB)
    print("mean lift  %.2f   median lift  %.2f   (1.00 = no author signal)" % (
        np.nanmean(lifts), np.nanmedian(lifts)))
    print("mean ICC   %.3f   median ICC   %.3f" % (np.nanmean(iccs), np.nanmedian(iccs)))
    print("labels with ICC > 0.10: %d of %d" % (int((iccs > 0.10).sum()), N_TAX))
    print("labels with lift > 1.5: %d of %d" % (int((lifts > 1.5).sum()), N_TAX))

    # what the author prior is WORTH: oracle best-F1 from the leave-one-out
    # author feature alone, macro-averaged, vs the trivial all-positive floor
    section("5b. WHAT AN AUTHOR PRIOR IS WORTH (oracle threshold, gold-only)")
    f_auth, f_base = [], []
    for j in range(N_TAX):
        y = Y[:, j].astype(np.float64)
        feat = np.zeros(len(y))
        for a, rs in groups.items():
            tot = y[rs].sum()
            for r in rs:
                nother = len(rs) - 1
                feat[r] = (tot - y[r]) / nother if nother > 0 else 0.0
        f1a, _ = best_f1_from_scores(y, feat + 1e-9 * np.random.default_rng(j).random(len(y)))
        p = y.mean()
        f1b = 2 * p / (1 + p)          # predict-all-positive F1
        f_auth.append(f1a)
        f_base.append(f1b)
    f_auth = np.array(f_auth)
    f_base = np.array(f_base)
    print("macro best-F1 from AUTHOR-PRIOR alone (oracle threshold): %.4f" % f_auth.mean())
    print("macro F1 from PREDICT-ALL-POSITIVE (no information)     : %.4f" % f_base.mean())
    print("delta                                                    : %+.4f" % (f_auth.mean() - f_base.mean()))
    print("NB both use an oracle threshold, so these are UPPER bounds.")


# ----------------------------------------------------------------------
# 6. rarest labels
# ----------------------------------------------------------------------

def analyse_rarest(df: pd.DataFrame, Y: np.ndarray, P: np.ndarray, k: int = 8) -> None:
    section("6. RAREST %d LABELS" % k)
    aid = df["anon_user_id"].astype(str).values
    sup = Y.sum(0)
    order = np.argsort(sup)[:k]
    print("%-34s %5s %8s %9s %-26s %7s" % (
        "label", "sup", "n_auth", "top-auth%", "most-correlated label", "phi"))
    print(SUB)
    for j in order:
        m = Y[:, j] == 1
        auths = pd.Series(aid[m])
        na = auths.nunique()
        topfrac = auths.value_counts().iloc[0] / max(m.sum(), 1) if m.sum() else 0.0
        row = P[j].copy()
        row[j] = 0.0
        b = int(np.argmax(np.abs(row)))
        print("%-34s %5d %8d %9.2f %-26s %7.3f" % (
            short(FACTOR_TAXONOMY[j], 34), int(sup[j]), na, topfrac,
            short(FACTOR_TAXONOMY[b], 26), row[b]))

    print()
    print("author concentration detail for the rarest %d:" % k)
    for j in order:
        m = Y[:, j] == 1
        vc = pd.Series(aid[m]).value_counts()
        n_auth = int(len(vc))
        cum2 = float(vc.iloc[:2].sum()) / max(int(m.sum()), 1)
        note = ""
        if n_auth <= 3 or cum2 >= 0.6:
            note = "   <-- CONCENTRATED: effectively unlearnable AND unmeasurable"
        print("  %-34s sup=%-4d n_auth=%-4d top2 share=%.2f%s" % (
            short(FACTOR_TAXONOMY[j], 34), int(sup[j]), n_auth, cum2, note))

    # 5-fold author-disjoint: how many positives land in a held-out fold?
    print()
    print("author-disjoint 5-fold: positives per held-out fold for the rarest %d" % k)
    rng = np.random.default_rng(SEED)
    uniq = np.array(sorted(set(aid)))
    perm = rng.permutation(len(uniq))
    fold_of = {uniq[perm[i]]: i % 5 for i in range(len(uniq))}
    fa = np.array([fold_of[a] for a in aid])
    print("%-34s %s" % ("label", "  f0   f1   f2   f3   f4"))
    print(SUB)
    for j in order:
        counts = [int(Y[fa == f, j].sum()) for f in range(5)]
        print("%-34s %s" % (short(FACTOR_TAXONOMY[j], 34),
                            "".join("%5d" % c for c in counts)))


# ----------------------------------------------------------------------
# 7. THE ORACLE BOUND -- can the other 23 GOLD labels predict label j?
# ----------------------------------------------------------------------

def analyse_oracle(df: pd.DataFrame, Y: np.ndarray) -> None:
    section("7. ORACLE BOUND FOR STRUCTURED PREDICTION")
    print("If you had PERFECT knowledge of the other 23 gold labels, how well")
    print("could you rank label j?  This is the hard ceiling for classifier")
    print("chains / label embedding / hierarchical decode: no real system can")
    print("beat a system that already knows all the other labels exactly.")
    print()
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score
    except Exception as exc:                                  # pragma: no cover
        print("[skip] scikit-learn unavailable: %s" % exc)
        return

    aid = df["anon_user_id"].astype(str).values
    rng = np.random.default_rng(SEED)
    uniq = np.array(sorted(set(aid)))
    perm = rng.permutation(len(uniq))
    fold_of = {uniq[perm[i]]: i % 5 for i in range(len(uniq))}
    fa = np.array([fold_of[a] for a in aid])

    sup = Y.sum(0)
    print("%-34s %5s %8s %9s %9s %8s" % (
        "label", "sup", "base AP", "oracle AP", "oracleF1", "allposF1"))
    print(SUB)
    aps, f1s, base_f1s, base_aps = [], [], [], []
    for j in range(N_TAX):
        y = Y[:, j].astype(int)
        X = np.delete(Y, j, axis=1).astype(np.float64)
        s = np.zeros(len(y), dtype=np.float64)
        for f in range(5):
            tr, te = fa != f, fa == f
            if y[tr].sum() < 2 or y[tr].sum() == tr.sum():
                s[te] = y[tr].mean()
                continue
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(X[tr], y[tr])
            s[te] = clf.predict_proba(X[te])[:, 1]
        base = y.mean()
        ap = average_precision_score(y, s) if 0 < y.sum() < len(y) else float("nan")
        f1o, _ = best_f1_from_scores(y.astype(float), s)
        f1all = 2 * base / (1 + base)
        aps.append(ap)
        base_aps.append(base)
        f1s.append(f1o)
        base_f1s.append(f1all)
        print("%-34s %5d %8.4f %9.4f %9.4f %8.4f" % (
            short(FACTOR_TAXONOMY[j], 34), int(sup[j]), base, ap, f1o, f1all))
    aps = np.array(aps, dtype=float)
    f1s = np.array(f1s, dtype=float)
    base_f1s = np.array(base_f1s, dtype=float)
    base_aps = np.array(base_aps, dtype=float)
    print(SUB)
    print("macro base AP (prevalence)            %.4f" % np.nanmean(base_aps))
    print("macro ORACLE AP from other 23 labels  %.4f" % np.nanmean(aps))
    print("macro ORACLE best-F1                  %.4f  (oracle threshold)" % f1s.mean())
    print("macro ALL-POSITIVE F1                 %.4f" % base_f1s.mean())
    print()
    print("Our shipped model is ~0.677 macro-F1 with a REAL threshold.")
    print("If the oracle-label ceiling is below that, structured prediction has")
    print("no headroom to give: the labels do not know about each other.")

    # rare-only view
    rare = sup < RARE_CUT
    print()
    print("rare labels only (support < %d, n=%d):" % (RARE_CUT, int(rare.sum())))
    print("  macro base AP %.4f -> macro oracle AP %.4f" % (
        float(np.nanmean(base_aps[rare])), float(np.nanmean(aps[rare]))))
    print("  macro oracle best-F1 %.4f vs all-positive %.4f" % (
        float(f1s[rare].mean()), float(base_f1s[rare].mean())))

    # ---- 7b: joint ceiling, other-23 gold labels + author leave-one-out prior
    section("7b. JOINT CEILING: other 23 GOLD labels + AUTHOR prior")
    groups = defaultdict(list)
    for r, a in enumerate(aid):
        groups[a].append(r)
    aps2, f1s2 = [], []
    print("%-34s %5s %9s %10s %10s" % (
        "label", "sup", "oracleAP", "jointAP", "jointF1"))
    print(SUB)
    for j in range(N_TAX):
        y = Y[:, j].astype(int)
        prior = np.zeros(len(y))
        for a, rs in groups.items():
            tot = y[rs].sum()
            for r in rs:
                nother = len(rs) - 1
                prior[r] = (tot - y[r]) / nother if nother > 0 else 0.0
        X = np.hstack([np.delete(Y, j, axis=1).astype(np.float64), prior[:, None]])
        s = np.zeros(len(y), dtype=np.float64)
        for f in range(5):
            tr, te = fa != f, fa == f
            if y[tr].sum() < 2 or y[tr].sum() == tr.sum():
                s[te] = y[tr].mean()
                continue
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(X[tr], y[tr])
            s[te] = clf.predict_proba(X[te])[:, 1]
        ap = average_precision_score(y, s) if 0 < y.sum() < len(y) else float("nan")
        f1o, _ = best_f1_from_scores(y.astype(float), s)
        aps2.append(ap)
        f1s2.append(f1o)
        print("%-34s %5d %9.4f %10.4f %10.4f" % (
            short(FACTOR_TAXONOMY[j], 34), int(sup[j]), aps[j], ap, f1o))
    aps2 = np.array(aps2, dtype=float)
    f1s2 = np.array(f1s2, dtype=float)
    print(SUB)
    print("macro JOINT AP      %.4f  (labels-only %.4f, prevalence %.4f)" % (
        float(np.nanmean(aps2)), float(np.nanmean(aps)), float(np.nanmean(base_aps))))
    print("macro JOINT best-F1 %.4f  (labels-only %.4f, all-positive %.4f)" % (
        float(f1s2.mean()), float(f1s.mean()), float(base_f1s.mean())))
    print()
    print("Reference: the shipped text model is ~0.677 macro-F1 and its per-label")
    print("AP runs 0.23 (meaning in life) to 0.92 (hopelessness).")
    print("meaning-in-life:  text-model AP 0.23  vs  gold-label oracle AP %.3f" % aps[
        FACTOR_TAXONOMY.index("meaning in life")])
    print("hopelessness   :  text-model AP 0.92  vs  gold-label oracle AP %.3f" % aps[
        FACTOR_TAXONOMY.index("hopelessness")])

    # ---- 7c: is the "structure" just post verbosity / annotation depth?
    section("7c. CONTROL -- is the label structure just CARDINALITY?")
    print("Feature = the COUNT of the other 23 gold labels, one scalar.")
    print("If a scalar recovers most of the 23-dim oracle AP, the 'structure'")
    print("is post verbosity / annotation depth, which the text model already")
    print("sees (post length, number of evidence spans) -- not label semantics.")
    print()
    print("%-34s %5s %9s %9s %9s" % ("label", "sup", "baseAP", "countAP", "23dimAP"))
    print(SUB)
    aps3 = []
    for j in range(N_TAX):
        y = Y[:, j].astype(int)
        cnt = (Y.sum(1) - Y[:, j]).astype(np.float64)
        ap = average_precision_score(y, cnt) if 0 < y.sum() < len(y) else float("nan")
        aps3.append(ap)
        print("%-34s %5d %9.4f %9.4f %9.4f" % (
            short(FACTOR_TAXONOMY[j], 34), int(sup[j]), float(y.mean()), ap, aps[j]))
    aps3 = np.array(aps3, dtype=float)
    print(SUB)
    m_base = float(np.nanmean(base_aps))
    m_cnt = float(np.nanmean(aps3))
    m_full = float(np.nanmean(aps))
    print("macro AP: prevalence %.4f -> CARDINALITY-ONLY %.4f -> full 23-dim %.4f" % (
        m_base, m_cnt, m_full))
    denom = m_full - m_base
    frac = (m_cnt - m_base) / denom if denom > 0 else float("nan")
    print("cardinality alone recovers %.1f%% of the oracle's lift over prevalence." % (100 * frac))
    print("residual lift attributable to LABEL IDENTITY (not depth): %+.4f macro AP" % (
        m_full - m_cnt))


def main() -> None:
    print(BAR)
    print("FACTOR LABEL STRUCTURE -- GOLD ONLY")
    print("source: artifacts/full_train.csv")
    print(BAR)
    df, Y = load()
    analyse_cooccurrence(Y)
    analyse_cardinality(Y)
    P = analyse_groups(Y)
    analyse_authors(df, Y)
    analyse_rarest(df, Y, P, k=8)
    analyse_oracle(df, Y)
    print()
    print(BAR)
    print("done")
    print(BAR)


if __name__ == "__main__":
    main()
