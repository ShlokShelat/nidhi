"""
risk_lora_member.py

Is the LoRA'd Qwen2.5-7B a useful RISK BLEND MEMBER? Free, CPU, on disk already.

WHY THIS SUDDENLY MATTERS. `runs/risk_lora7b/run_config.json` records
best_val_weighted_f1 = **0.7694** on artifacts/val_split.csv. The deployed risk
ensemble scores **~0.7646** on that same split (the figure train_risk_nli.py
prints for comparison). A LoRA'd 7B with no reasoning traces, 3 epochs, matches
or beats the whole encoder chain -- and it is architecturally unlike every
current member, which is precisely the diversity the risk audit found missing
(38 of 39 confident errors had NEITHER existing head right).

That run also saved `val_probs.npy` (372 x 4), so the member test costs nothing.

WHAT THIS CHECKS, in order of what it would change:
 1. ORDERING SANITY. val_probs.npy carries no row_id, so it is assumed to be in
    val_split.csv order. If the weighted-F1 we recompute does not land near the
    recorded 0.7694, the assumption is wrong and every number below is garbage.
    This is checked FIRST and the script stops if it fails.
 2. COMPLEMENTARITY. Of the posts the encoder gets wrong, how many does the LoRA
    model get right -- and how many of the encoder's correct answers does it
    break? The net matters, not the rescue rate: on the real distribution the
    encoder is right ~79% of the time, so breakage is multiplied by ~3.8x more
    posts than rescue is. A good-looking rescue rate can still be a net loss
    once that multiplier is applied.
 3. CONFIDENCE GATE. Mean confidence when right vs when wrong. A gap means the
    encoder can be overruled only where the LoRA model is sure, which converts
    a break-even swap into a real gain. No gap means it is confidently wrong
    the same way the encoder is, and no gate exists.
 4. THE BLEND ITSELF, at several weights, with a paired AUTHOR bootstrap.

CAVEAT STATED UP FRONT: 372 posts / ~32 authors. The confidence interval will
be wide -- a prior test on this same split could not resolve a modest true
effect from noise. Treat a positive result as grounds for a full 5-fold
rebuild, not as a shippable number.

Usage:
  python src/risk_lora_member.py
  python src/risk_lora_member.py --weights 0.1,0.2,0.3,0.4,0.5
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="runs/risk_lora7b/val_probs.npy")
    ap.add_argument("--lora-csv", default=None,
                    help="pooled OOF CSV (row_id, prob_*, gold) instead of the "
                         "bare .npy. Preferred: row_ids remove the ordering "
                         "assumption entirely, and a 5-fold pool removes both "
                         "the 32-author CI and the epoch-selection bias.")
    ap.add_argument("--enc", default="artifacts/nligaur/oof_base.csv",
                    help="encoder OOF probs; its rows on val_split are honest "
                         "out-of-fold, and the LoRA model trained on "
                         "train_split only, so both are clean on these posts")
    ap.add_argument("--val", default="artifacts/val_split.csv")
    ap.add_argument("--labeled", default="artifacts/full_train.csv")
    ap.add_argument("--recorded", type=float, default=0.7694,
                    help="best_val_weighted_f1 from run_config.json, used to "
                         "verify the row ordering assumption")
    ap.add_argument("--weights", default="0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pc = [f"prob_{c}" for c in RISK_LEVELS]
    if args.lora_csv:
        lc = pd.read_csv(args.lora_csv, keep_default_na=False)
        lc["row_id"] = lc["row_id"].astype(str)
        need = [c for c in pc + ["gold"] if c not in lc.columns]
        if need:
            raise SystemExit(f"{args.lora_csv} lacks {need}")
        lc = lc.drop_duplicates("row_id")
        rows = list(lc["row_id"])
        y = np.array([normalize_label(x) for x in lc["gold"]])
        L = lc[pc].to_numpy(float)
        print(f"{args.lora_csv}: {L.shape[0]} posts, keyed by row_id")
        print("\n1. ORDERING SANITY -- not applicable, rows are row_id-keyed")
        got = risk_weighted_f1(
            list(y), [RISK_LEVELS[i] for i in L.argmax(1)])
        print(f"   LoRA pooled OOF weighted-F1 {got:.4f}")
    else:
        from data_audit import load
        va = load(args.val)
        va["row_id"] = va["row_id"].astype(str)
        rows = list(va["row_id"])
        y = np.array([normalize_label(x) for x in va["risk_level"]])

        L = np.load(args.lora)
        print(f"{args.lora}: {L.shape}   val_split: {len(rows)} posts")
        if L.shape[0] != len(rows):
            raise SystemExit(f"row count mismatch {L.shape[0]} vs {len(rows)} "
                             f"-- cannot align without row_id")
        if L.shape[1] != len(RISK_LEVELS):
            raise SystemExit(f"expected {len(RISK_LEVELS)} columns, "
                             f"got {L.shape[1]}")
        print(f"\n1. ORDERING SANITY")
        got = risk_weighted_f1(
            list(y), [RISK_LEVELS[i] for i in L.argmax(1)])
        print(f"   recomputed weighted-F1 {got:.4f}   "
              f"recorded {args.recorded:.4f}")
        if abs(got - args.recorded) > 0.03:
            raise SystemExit(
                "   MISMATCH > 0.03 -- val_probs.npy is probably NOT in "
                "val_split.csv order.\n   Every number below would be "
                "meaningless, so stopping here.")
        print("   within tolerance; the ordering assumption holds")

    if L.min() < 0 or L.max() > 1.0001:
        print("  note: values outside [0,1]; treating as logits and softmaxing")
        e = np.exp(L - L.max(1, keepdims=True))
        L = e / e.sum(1, keepdims=True)
    lora_p = np.array([RISK_LEVELS[i] for i in L.argmax(1)])

    # encoder probs restricted to these rows
    d = pd.read_csv(args.enc, keep_default_na=False)
    d["row_id"] = d["row_id"].astype(str)
    d = d.set_index("row_id")
    absent = [r for r in rows if r not in d.index]
    if absent:
        # champion_oof.csv covers 1590 of 1635 posts, so a few val rows can be
        # missing. Restrict rather than refuse -- but restrict the LoRA matrix
        # and gold in lockstep, or every alignment below silently shifts.
        keep = np.array([r in d.index for r in rows])
        print(f"   note: {len(absent)} of {len(rows)} val rows absent from "
              f"{args.enc}; restricting to {int(keep.sum())}")
        rows = [r for r, k in zip(rows, keep) if k]
        y = y[keep]
        L = L[keep]
        lora_p = lora_p[keep]
    missing_cols = [c for c in pc if c not in d.columns]
    if missing_cols:
        raise SystemExit(f"{args.enc} lacks {missing_cols}")
    E = d.loc[rows, pc].to_numpy(float)
    enc_p = np.array([RISK_LEVELS[i] for i in E.argmax(1)])
    print(f"\n   encoder on these posts: wF1 "
          f"{risk_weighted_f1(list(y), list(enc_p)):.4f}")

    lab = pd.read_csv(args.labeled, keep_default_na=False)
    lab["row_id"] = lab["row_id"].astype(str)
    amap = dict(zip(lab["row_id"], lab["anon_user_id"].astype(str)))
    author = np.array([amap.get(r, r) for r in rows])
    print(f"   {len(set(author))} authors -- expect a wide CI")

    print("\n2. COMPLEMENTARITY")
    wrong = enc_p != y
    right = ~wrong
    resc = float((lora_p[wrong] == y[wrong]).mean()) if wrong.any() else 0.0
    brk = float((lora_p[right] != y[right]).mean()) if right.any() else 0.0
    print(f"   encoder wrong on {int(wrong.sum())}, LoRA right on {resc:.3f}")
    print(f"   encoder right on {int(right.sum())}, LoRA breaks {brk:.3f}")
    print(f"   same error repeated: "
          f"{float((lora_p[wrong] == enc_p[wrong]).mean()):.3f}")
    # net on the REAL distribution, not this sample
    p_wrong = float(wrong.mean())
    net = p_wrong * resc - (1 - p_wrong) * brk
    print(f"   net if we simply SWAPPED: {net:+.4f} per post "
          f"(encoder wrong {p_wrong:.1%} of the time)")
    print("   A swap near zero is expected and is not the point -- a blend can")
    print("   beat a swap only if confidence separates its hits from misses.")

    print("\n3. CONFIDENCE GATE")
    conf = L.max(1)
    ok = lora_p == y
    print(f"   mean confidence when RIGHT {conf[ok].mean():.3f}   "
          f"when WRONG {conf[~ok].mean():.3f}   "
          f"gap {conf[ok].mean() - conf[~ok].mean():+.3f}")
    for t in (0.5, 0.7, 0.9):
        m = conf >= t
        if m.sum():
            print(f"   conf >= {t}: {int(m.sum()):>4} posts, "
                  f"LoRA accuracy {float((lora_p[m] == y[m]).mean()):.3f}, "
                  f"encoder accuracy {float((enc_p[m] == y[m]).mean()):.3f}")

    print("\n4. BLEND (probability average, weight on the LoRA member)")
    ws = [float(x) for x in args.weights.split(",")]
    rng = np.random.default_rng(args.seed)
    auths = sorted(set(author))
    aidx = {a: np.where(author == a)[0] for a in auths}
    base_f1 = risk_weighted_f1(list(y), list(enc_p))
    print(f"   {'weight':>7}{'wF1':>9}{'delta':>9}{'95% CI':>22}")
    print("   " + "-" * 47)
    for w in ws:
        B = (1 - w) * E + w * L
        p = np.array([RISK_LEVELS[i] for i in B.argmax(1)])
        f1 = risk_weighted_f1(list(y), list(p))
        draws = []
        for _ in range(args.boot):
            pick = rng.choice(len(auths), len(auths), replace=True)
            sel = np.concatenate([aidx[auths[i]] for i in pick])
            draws.append(risk_weighted_f1(list(y[sel]), list(p[sel]))
                         - risk_weighted_f1(list(y[sel]), list(enc_p[sel])))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        print(f"   {w:>7.2f}{f1:>9.4f}{f1 - base_f1:>+9.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}")

    # ------------------------------------------------------------------
    # 5. SELECTION RULES, which is the right shape here.
    #
    # Probability averaging failed above for a scale reason, not a signal one:
    # at weight 0.10 the blend was bit-identical to the encoder (CI [0,0])
    # because the encoder's distribution is sharper and dominates any mean.
    # But confidence SEPARATES this member's hits from its misses (+0.188 gap),
    # which averaging cannot exploit and selection can.
    # ------------------------------------------------------------------
    print("\n5. SELECTION RULES")
    econf = E.max(1)

    def report(name, pred, n_params):
        f1 = risk_weighted_f1(list(y), list(pred))
        draws = []
        for _ in range(args.boot):
            pick = rng.choice(len(auths), len(auths), replace=True)
            sel = np.concatenate([aidx[auths[i]] for i in pick])
            draws.append(risk_weighted_f1(list(y[sel]), list(pred[sel]))
                         - risk_weighted_f1(list(y[sel]), list(enc_p[sel])))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        moved = int((pred != enc_p).sum())
        print(f"   {name:<26}{f1:>8.4f}{f1 - base_f1:>+9.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{moved:>7}{n_params:>4}")

    print(f"   {'rule':<26}{'wF1':>8}{'delta':>9}{'95% CI':>22}"
          f"{'moved':>7}{'par':>4}")
    print("   " + "-" * 68)

    # ZERO PARAMETERS: believe whichever model is more confident.
    p = np.where(L.max(1) > econf, lora_p, enc_p)
    report("maxconf (0 params)", p, 0)

    # ZERO PARAMETERS: only overrule when the encoder is BELOW its own median
    # confidence and the LoRA member is above its own. Both cutoffs are the
    # data's own medians, so nothing is fitted.
    m = (econf < np.median(econf)) & (L.max(1) > np.median(L.max(1)))
    p = np.where(m, lora_p, enc_p)
    report("median-swap (0 params)", p, 0)

    # ONE PARAMETER, CROSS-FITTED over author halves so the threshold is never
    # chosen on the posts it is scored on. One fitted parameter has historically
    # transferred poorly to held-out evaluation; zero-parameter rules over-deliver.
    perm = rng.permutation(len(auths))
    hA = {auths[i] for i in perm[:len(auths) // 2]}
    inA = np.array([a in hA for a in author])
    grid = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    p = enc_p.copy()
    picked = []
    for tune, appl in ((inA, ~inA), (~inA, inA)):
        best_t, best_f = grid[0], -1.0
        for t in grid:
            q = np.where(L.max(1) >= t, lora_p, enc_p)
            f = risk_weighted_f1(list(y[tune]), list(q[tune]))
            if f > best_f:
                best_f, best_t = f, t
        picked.append(best_t)
        q = np.where(L.max(1) >= best_t, lora_p, enc_p)
        p[appl] = q[appl]
    report(f"gate xfit (t={picked})", p, 1)

    # logit-space average: less dominated by whichever model is sharper
    def lg(P):
        P = np.clip(P, 1e-6, 1 - 1e-6)
        return np.log(P / (1 - P))
    for w in (0.3, 0.5):
        B = (1 - w) * lg(E) + w * lg(L)
        p = np.array([RISK_LEVELS[i] for i in B.argmax(1)])
        report(f"logit-avg w={w}", p, 0)

    # THE CEILING for any selection rule: pick the right model every time.
    orc = np.where(enc_p == y, enc_p, np.where(lora_p == y, lora_p, enc_p))
    orc_f1 = risk_weighted_f1(list(y), list(orc))
    print(f"\n   ORACLE selector (unreachable): {orc_f1:.4f} "
          f"({orc_f1 - base_f1:+.4f})")
    print("   That is the most ANY rule choosing between these two can buy.")

    print("\n" + "=" * 68)
    print("372 posts / ~32 authors. This split could not resolve a +0.013")
    print("effect in the evidence-conditioning test, so read the SIGN and the")
    print("confidence gap, not the magnitude. A positive result justifies a")
    print("5-fold rebuild of this member; it is not itself shippable.")
    print("=" * 68)


if __name__ == "__main__":
    main()
