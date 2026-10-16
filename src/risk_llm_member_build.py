"""
risk_llm_member_build.py

Add the Qwen risk model as a LOW-WEIGHT, COUNT-MATCHED MEMBER of the risk
chain -- the one risk-LLM configuration that was never scored.

WHY THIS IS DIFFERENT FROM REPLACING THE RISK COLUMN OUTRIGHT. Fully replacing
the champion's risk column with the Qwen bag was measured and found harmful.
A forensic on that result shows why it failed and why a member is different:
Qwen's class MARGINALS were
already fine (154/118/94/12, all inside the probe-measured 90% intervals
112-170 / 99-142 / 69-113 / 13-41), but it disagreed with the champion on 75 of
378 posts and was wrong on nearly all of them. Its per-post ranking is worse;
its marginals are not the problem. So no decision rule rescues a replacement --
but a small weight on its probabilities can still nudge the chain's ordering
without letting its own argmax through.

MEASURED, on 1590 out-of-fold posts against the FULL chain (champion_oof.csv,
which carries pre-rule probabilities for the deployed ensemble+friend+NLI):

    chain wF1 0.7858 -> count-matched blend at w=0.2  0.7942

measured across 50 author partitions, passing the pre-stated bar of a clear
positive mean gain with a strong majority of splits winning. Note the raw
complementarity is NEGATIVE (122 fixes vs 143 breaks): Qwen loses head-to-head
on argmax and still helps as a weighted member. That is the signature of a
genuine ensemble effect rather than a better model.

WHAT THIS REBUILDS. The champion's test-side risk chain, from saved artifacts:

    chain = 0.7 * ensembled_risk.csv + 0.3 * risk_nli_ens_probs.npy

`ensembled_risk.csv` ALREADY contains the 0.5/0.5 friend blend (ensemble_risk.py
does that), and blend_risk_saved.py adds the two NLI members at a forced 0.30.
The reconstruction is VERIFIED against the champion's own risk column before
anything is blended -- if it does not reproduce it to within the handful of
posts the two symbolic rules move, the stage order is wrong and the script
refuses to continue.

COUNT-MATCHING. The final labels reproduce the champion's own class counts
exactly, assigning by margin. So density is held fixed and the ONLY thing that
changes is which posts sit where -- the same device behind the one factor
change that transferred at 80%, and the reason this degenerates toward a no-op
rather than a directional loss if Qwen's ranking is not actually better.

Usage:
  python src/risk_llm_member_build.py --weight 0.2
  python src/risk_llm_member_build.py --weight 0.2 --out artifacts/risk_qmem.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import RISK_LEVELS, normalize_label  # noqa: E402

PC = ["prob_indicator", "prob_ideation", "prob_behavior", "prob_attempt"]
EC = ["p_Indicator", "p_Ideation", "p_Behavior", "p_Attempt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", default="submissions/NIDHI_bag6q.csv")
    ap.add_argument("--ens", default="artifacts/ensembled_risk.csv")
    ap.add_argument("--nli", default="artifacts/risk_nli_ens_probs.npy")
    ap.add_argument("--llm", default="artifacts/risk_llm14b_test.csv")
    ap.add_argument("--w-nli", type=float, default=0.30)
    ap.add_argument("--weight", type=float, default=0.20)
    ap.add_argument("--out", default="artifacts/risk_qwen_member.csv")
    ap.add_argument("--max-rule-posts", type=int, default=12,
                    help="how many posts the two symbolic rules may account "
                         "for; more than this means the reconstruction is wrong")
    args = ap.parse_args()

    ch = pd.read_csv(args.champion, keep_default_na=False)
    ch["row_id"] = ch["row_id"].astype(str)
    ens = pd.read_csv(args.ens)
    ens["row_id"] = ens["row_id"].astype(str)
    llm = pd.read_csv(args.llm)
    llm["row_id"] = llm["row_id"].astype(str)
    nli = np.load(args.nli)

    if not (list(ch.row_id) == list(ens.row_id) == list(llm.row_id)):
        raise SystemExit("row order differs across the three sources")
    if nli.shape != (len(ch), 4):
        raise SystemExit(f"{args.nli} is {nli.shape}, expected ({len(ch)}, 4)")

    E = ens[EC].to_numpy(float)
    L = llm[PC].to_numpy(float)
    chain = (1 - args.w_nli) * E + args.w_nli * nli

    # ---- verify the reconstruction before trusting it -------------------
    champ = np.array([normalize_label(x) for x in ch["risk_level"]])
    recon = np.array([RISK_LEVELS[i] for i in chain.argmax(1)])
    diff = int((recon != champ).sum())
    print(f"reconstruction check: argmax(chain) differs from the champion's "
          f"risk column on {diff} of {len(ch)} posts")
    print("  (the two symbolic rules move a handful; a large number means the "
          "stage order or weights are wrong)")
    if diff > args.max_rule_posts:
        raise SystemExit(
            f"REFUSING: {diff} posts differ, more than the {args.max_rule_posts} "
            f"the rules can explain. The chain reconstruction is not faithful, "
            f"so a blend built on it would not be the thing that was measured.")

    # ---- blend, then count-match to the champion's own class counts ------
    B = (1 - args.weight) * chain + args.weight * L
    tgt = {c: int((champ == c).sum()) for c in RISK_LEVELS}
    print(f"\nchampion counts {tgt}")

    margin = B.max(1) - np.sort(B, axis=1)[:, -2]
    order = np.argsort(-margin)          # most confident assigned first
    left = dict(tgt)
    pred = np.array([""] * len(ch), dtype=object)
    for i in order:
        for j in np.argsort(-B[i]):
            c = RISK_LEVELS[j]
            if left[c] > 0:
                pred[i] = c
                left[c] -= 1
                break
        if not pred[i]:
            pred[i] = RISK_LEVELS[int(B[i].argmax())]
    pred = np.array([str(x) for x in pred])

    got = {c: int((pred == c).sum()) for c in RISK_LEVELS}
    if got != tgt:
        raise SystemExit(f"count-matching failed: {got} vs {tgt}")

    changed = int((pred != champ).sum())
    print(f"counts reproduced exactly")
    print(f"risk labels changed vs champion: {changed} of {len(ch)}")
    for a in RISK_LEVELS:
        for b in RISK_LEVELS:
            n = int(((champ == a) & (pred == b)).sum())
            if a != b and n:
                print(f"    {a} -> {b}: {n}")

    # keep the champion's own casing convention
    case = {normalize_label(x): x for x in ch["risk_level"]}
    out = pd.DataFrame({"row_id": ch["row_id"],
                        "pred_risk_level": [case.get(p, p) for p in pred]})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print("\nPatch it on as a single-variable change:")
    print(f"  python src/patch_submission_risk.py --base {args.champion} \\")
    print(f"      --risk {args.out} --out submissions/NIDHI_riskqmem.csv")
    print("\nS2 must come back UNCHANGED; only the risk column moved.")
    if changed == 0:
        print("\nNOTHING CHANGED -- the blend does not reorder any post. "
              "Do not spend a slot.")


if __name__ == "__main__":
    main()
