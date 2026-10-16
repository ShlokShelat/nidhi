"""
dump_disagreement_cases_v5.py

Writes two review files for manual error analysis on the val split.

  1. disagreement_cases — posts bucketed by the SHAPE of the models'
     disagreement, so the most informative cases are read first.
  2. ensemble_errors — EVERY post where the deployed ensemble is wrong,
     regardless of whether the individual models agreed. A
     disagreement-only view is blind to the case where all models
     confidently agree on the SAME wrong answer: no disagreement exists,
     so nothing flags it, yet a systematic blind spot is exactly the
     kind of error that repeats.

     Split into:
       NEAR MISS  — gold label was the ensemble's SECOND-best guess
                    (the model was in the right neighborhood)
       FAR MISS   — gold wasn't even second-best (the model was lost)
     These are different failure signatures and are worth reading
     separately.

  3. Per-case: post length (words), evidence length (words), and the
     gold / predicted / second-best labels explicitly.

Usage — writes TWO files (disagreement + errors).
  python src/dump_disagreement_cases_v5.py \
      --val artifacts/val_split.csv \
      --val-probs runs/risk_v1/val_predictions.csv,runs/risk_v1_s14/val_predictions.csv,runs/risk_v1_s15/val_predictions.csv \
      --names risk_v1,risk_v1_s14,risk_v1_s15 \
      --nli-run runs/risk_nli,runs/risk_nli_s2 \
      --out-disagreement artifacts/disagreement_cases_v4.txt \
      --out-errors artifacts/ensemble_errors.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from collections import Counter

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from scoring import RISK_LEVELS, normalize_label
from data_audit import load

PROBCOLS = ["prob_indicator", "prob_ideation", "prob_behavior", "prob_attempt"]


def wrap(text, width=95, indent="    "):
    return ("\n" + indent).join(textwrap.wrap(str(text), width=width))


def load_val_probs(path, row_ids):
    df = pd.read_csv(path).set_index("row_id")
    return df.loc[row_ids, PROBCOLS].values.astype(float)


def top1_top2(probs_row):
    order = np.argsort(probs_row)[::-1]
    return RISK_LEVELS[order[0]], RISK_LEVELS[order[1]], probs_row[order[0]], probs_row[order[1]]


def nli_probs_for(df, run_dirs, max_len, device, use_bf16):
    from train_risk_nli import RiskNLIScorer, score_posts
    acc = []
    for rd in run_dirs:
        mn = open(os.path.join(rd, "tok")).read().strip()
        tok = AutoTokenizer.from_pretrained(mn)
        m = RiskNLIScorer(mn).to(device)
        m.load_state_dict(torch.load(os.path.join(rd, "model.pt"),
                                     map_location=device), strict=False)
        m.eval()
        acc.append(score_posts(m, df, tok, max_len, device, use_bf16))
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.mean(acc, axis=0)


def word_count(text):
    return len(str(text).split())


def format_disagreement_case(n, priority_note, rid, gold, row, names, probs_list, ensemble_probs):
    lines = []
    lines.append("=" * 100)
    lines.append(f"CASE {n}  row_id={rid}  GOLD={gold}  [{priority_note}]")
    lines.append("=" * 100)
    lines.append(f"POST ({word_count(row['post'])} words):\n    {wrap(row['post'])}")
    lines.append("")
    lines.append("MODEL VOTES (top1 / margin = top1-top2):")
    votes = []
    rows = []
    for name, probs_row in zip(names, probs_list):
        p1, p2, c1, c2 = top1_top2(probs_row)
        votes.append(p1)
        rows.append((name, p1, c1, c1 - c2))
    vote_counts = Counter(votes)
    majority_label = vote_counts.most_common(1)[0][0]
    for name, pred, top1, margin in sorted(rows, key=lambda x: -x[3]):
        correct = "correct" if pred == gold else "WRONG  "
        maj = "majority" if pred == majority_label else "MINORITY"
        lines.append(f"    {name:<18} {pred:<12} top1={top1:.2f}  margin={margin:.2f}  "
                     f"[{correct}] [{maj}]")

    ens_p1, ens_p2, ens_c1, ens_c2 = top1_top2(ensemble_probs)
    ens_correct = "correct" if ens_p1 == gold else "WRONG"
    lines.append("")
    lines.append("DEPLOYED ENSEMBLE (0.30*risk-NLI + 0.70*(0.5*equal-3-model "
                 "+ 0.5*friend)):")
    lines.append(f"    predicted={ens_p1}  second-best={ens_p2}  "
                 f"margin={ens_c1-ens_c2:.2f}  [{ens_correct}]")

    ev = str(row.get("evidence", "")).strip()
    ev_wc = word_count(ev) if ev and ev.lower() != "none" else 0
    lines.append("")
    lines.append(f"GOLD EVIDENCE ({ev_wc} words):\n    "
                 f"{wrap(ev) if ev_wc else '(none)'}")
    lines.append("")
    lines.append("YOUR NOTES: _______________________________________________")
    lines.append("")
    return "\n".join(lines)


def format_error_case(n, kind, rid, gold, row, ensemble_probs):
    p1, p2, c1, c2 = top1_top2(ensemble_probs)
    ev = str(row.get("evidence", "")).strip()
    ev_wc = word_count(ev) if ev and ev.lower() != "none" else 0
    lines = []
    lines.append("=" * 100)
    lines.append(f"{kind} {n}  row_id={rid}")
    lines.append("=" * 100)
    lines.append(f"POST ({word_count(row['post'])} words):\n    {wrap(row['post'])}")
    lines.append("")
    lines.append(f"GOLD:      {gold}")
    lines.append(f"PREDICTED: {p1}  (conf={c1:.2f})")
    lines.append(f"2nd BEST:  {p2}  (conf={c2:.2f})")
    lines.append(f"GOLD EVIDENCE ({ev_wc} words):\n    "
                 f"{wrap(ev) if ev_wc else '(none)'}")
    lines.append("")
    lines.append("YOUR NOTES: _______________________________________________")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--val-probs", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--nli-run", required=True)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--max-disagreement-cases", type=int, default=40)
    ap.add_argument("--friend-probs-val", required=True,
                    help="friend model's OOF val predictions (row_id,Indicator,"
                         "Ideation,Behavior,Attempt)")
    ap.add_argument("--out-disagreement", default="artifacts/disagreement_cases_v5.txt")
    ap.add_argument("--out-errors", default="artifacts/ensemble_errors.txt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[precision] {'bf16' if use_bf16 else 'fp32'} on {device}")

    val_df = load(args.val)
    row_ids = val_df["row_id"].tolist()
    gold = [normalize_label(x) for x in val_df["risk_level"]]

    prob_paths = [p.strip() for p in args.val_probs.split(",")]
    names = [n.strip() for n in args.names.split(",")]
    per_model_probs = [load_val_probs(p, row_ids) for p in prob_paths]
    equal_avg = np.mean(per_model_probs, axis=0)

    print("scoring risk-NLI on val...")
    nli_runs = [r.strip() for r in args.nli_run.split(",")]
    nli_probs = nli_probs_for(val_df, nli_runs, args.max_len, device, use_bf16)

    # Friend model's own OOF val predictions, so the ensemble reproduced here
    # is the deployed one rather than an approximation of it.
    FRIEND_COLS = ["Indicator", "Ideation", "Behavior", "Attempt"]
    friend_df = pd.read_csv(args.friend_probs_val).set_index("row_id")
    missing = [r for r in row_ids if r not in friend_df.index]
    if missing:
        print(f"WARNING: {len(missing)} val row_ids missing from friend file "
              f"(e.g. {missing[:3]}) — falling back to the 3-seed average "
              f"for these rows only")
    friend_probs = np.zeros((len(row_ids), 4))
    for i, rid in enumerate(row_ids):
        if rid in friend_df.index:
            p = friend_df.loc[rid, FRIEND_COLS].values.astype(float)
        else:
            p = equal_avg[i]  # explicit, logged fallback — not silent
        friend_probs[i] = p / p.sum()

    base_ensemble = 0.5 * equal_avg + 0.5 * friend_probs  # matches ensemble_risk.py's weight-ours=0.5
    deployed_probs = 0.30 * nli_probs + 0.70 * base_ensemble  # matches blend_risk_saved.py

    # --- File 1: priority-sorted disagreement cases ---
    priority1, priority2, priority3, priority4 = [], [], [], []
    # --- File 2: exhaustive ensemble errors, split near/far miss ---
    near_miss, far_miss = [], []

    all_names = names + ["friend_model"]
    for i, rid in enumerate(row_ids):
        row = val_df.iloc[i]
        probs_list = [per_model_probs[m][i] for m in range(len(names))] + [friend_probs[i]]
        votes = [RISK_LEVELS[int(np.argmax(p))] for p in probs_list]
        ens_p1, ens_p2, _, _ = top1_top2(deployed_probs[i])
        ens_correct = ens_p1 == gold[i]
        any_model_correct = any(v == gold[i] for v in votes)
        all_agree = len(set(votes)) == 1

        dcase = (rid, gold[i], row, all_names, probs_list, deployed_probs[i])
        if all_agree and votes[0] != gold[i]:
            priority1.append(dcase)
        elif not ens_correct and any_model_correct:
            priority2.append(dcase)
        elif ens_correct and not all_agree:
            priority3.append(dcase)
        else:
            priority4.append(dcase)

        if not ens_correct:
            ecase = (rid, gold[i], row, deployed_probs[i])
            if ens_p2 == gold[i]:
                near_miss.append(ecase)
            else:
                far_miss.append(ecase)

    # write disagreement file
    out_lines = []
    sections = [
        ("PRIORITY 1: ALL MODELS AGREE BUT WRONG (systematic blind spot)", priority1, "all-agree-wrong"),
        ("PRIORITY 2: ENSEMBLE WRONG, BUT >=1 MODEL WAS RIGHT (signal discarded)", priority2, "ensemble-discards-signal"),
        ("PRIORITY 3: ENSEMBLE RIGHT DESPITE DISAGREEMENT (worked out — why?)", priority3, "ensemble-recovered"),
        ("PRIORITY 4: OTHER", priority4, "other"),
    ]
    per_section_cap = max(5, args.max_disagreement_cases // 4)
    for title, cases, tag in sections:
        out_lines.append("#" * 100)
        out_lines.append(f"# {title}")
        out_lines.append(f"# {len(cases)} total, showing up to {per_section_cap}")
        out_lines.append("#" * 100 + "\n")
        for n, case in enumerate(cases[:per_section_cap], start=1):
            out_lines.append(format_disagreement_case(n, tag, *case))
    os.makedirs(os.path.dirname(args.out_disagreement) or ".", exist_ok=True)
    with open(args.out_disagreement, "w") as f:
        f.write("\n".join(out_lines))

    # write exhaustive errors file
    err_lines = []
    err_lines.append("#" * 100)
    err_lines.append(f"# NEAR MISS — gold was the ensemble's 2nd-best guess ({len(near_miss)} total)")
    err_lines.append("#" * 100 + "\n")
    for n, case in enumerate(near_miss, start=1):
        err_lines.append(format_error_case(n, "NEAR MISS", *case))
    err_lines.append("#" * 100)
    err_lines.append(f"# FAR MISS — gold wasn't even 2nd-best ({len(far_miss)} total)")
    err_lines.append("#" * 100 + "\n")
    for n, case in enumerate(far_miss, start=1):
        err_lines.append(format_error_case(n, "FAR MISS", *case))
    os.makedirs(os.path.dirname(args.out_errors) or ".", exist_ok=True)
    with open(args.out_errors, "w") as f:
        f.write("\n".join(err_lines))

    total_errors = len(near_miss) + len(far_miss)
    print(f"\nwrote {args.out_disagreement}")
    print(f"  P1 (all-agree-wrong): {len(priority1)}  P2 (discards signal): {len(priority2)}  "
          f"P3 (recovered): {len(priority3)}  P4 (other): {len(priority4)}")
    print(f"\nwrote {args.out_errors}")
    print(f"  Total ensemble errors: {total_errors}/{len(row_ids)} "
          f"(val weighted-accuracy context)")
    print(f"  Near-miss (gold was 2nd-best): {len(near_miss)}")
    print(f"  Far-miss (gold not even 2nd-best): {len(far_miss)}")
    print("\nThe friend model enters as a 4th voice from its own OOF val "
          "predictions, so the ensemble scored here is the deployed formula "
          "rather than an approximation of it.")
    if missing:
        print(f"  (except {len(missing)} val rows where friend had no OOF pred — "
              "fallback used there, see WARNING above)")


if __name__ == "__main__":
    main()
