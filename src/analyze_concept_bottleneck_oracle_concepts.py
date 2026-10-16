"""
analyze_concept_bottleneck_oracle_concepts.py

Runs the concept-bottleneck diagnostic suite on a trained ConceptRiskModel
checkpoint. Steps 2-4 (ablation, fusion weights, fusion activation)
are BOTTLENECK-ONLY -- meaningless on the auxiliary/Option-1
architecture, where concepts never reach the risk head at inference.

Step 1: concept accuracy/precision/recall/F1 (needs --val-concept-labels)
Step 2: concept ablation -- zero one concept at a time, measure delta val F1
Step 3: fusion weight norms (embedding vs concept_probs vs concept_logits)
Step 4: fusion activation contribution (per-example decomposition, not just weights)
Step 5: error analysis -- 20 correct vs 20 incorrect, compare concept predictions
Step 6: agreement analysis -- concept accuracy split by whether risk prediction was correct
Step 7: per-class ablation -- same ablation as Step 2, broken down by risk class
Step 8: zero ALL concepts at once -- embedding-only vs embedding+concepts, single biggest signal
Step 9: concept correlation matrix -- are concepts redundant with each other?
Step 10: concept probability calibration -- histogram per concept (near-binary vs graded)
Step 11: concept confidence vs risk correctness -- do errors cluster on low-confidence concepts?
Step 12: manual audit -- 20 wrong predictions with full concept trace, for qualitative reading

--oracle-concepts: evaluation-only mode. Feeds ground-truth concept
  labels into the fusion layer instead of the concept head's own
  predictions (no retraining, checkpoint/architecture unchanged), to
  test whether the fusion layer COULD use concepts if they were perfect.

Every run writes a full copy of its printed output to a timestamped log
file under --log-dir (default: analysis_logs/), named after the
--checkpoint directory and the run mode, so repeated runs (e.g. a
confidence sweep, or comparing two checkpoints) never overwrite each
other's results and nothing is ever written into the checkpoint
directory itself.

Usage:
  python src/analyze_concept_bottleneck_oracle_concepts.py \
      --val artifacts/val_split.csv \
      --checkpoint runs/risk_concepts_bottleneck \
      --val-concept-labels artifacts/val_concept_labels.csv
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class Tee:
    """Mirrors everything written to stdout into a log file as well,
    so a full transcript of each run is saved automatically -- without
    ever touching the --checkpoint directory (that's the model's, not
    the analysis's) and without ever reusing a filename across runs
    (timestamped, so a sweep or repeated runs can't clobber each other)."""
    def __init__(self, filepath, real_stdout):
        self.file = open(filepath, "w")
        self.real_stdout = real_stdout

    def write(self, data):
        self.real_stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.real_stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def make_log_path(log_dir, checkpoint_dir, oracle_mode):
    os.makedirs(log_dir, exist_ok=True)
    ckpt_name = os.path.basename(os.path.normpath(checkpoint_dir))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "oracle" if oracle_mode else "diagnostics"
    filename = f"{ckpt_name}__{suffix}__{timestamp}.log"
    return os.path.join(log_dir, filename)

sys.path.insert(0, os.path.dirname(__file__))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1
from train_risk_concepts_bottleneck import ConceptRiskModel, PostDataset, CONCEPTS


def build_oracle_concept_tensors(val_df, val_concept_labels_path, confidence=0.98):
    """Build (probs, logits) tensors, aligned to val_df's row order, from
    ground-truth concept labels instead of the concept head's predictions.

    The fusion layer was trained on [sigmoid(logit), logit] pairs, so a raw
    binary label alone isn't a compatible input -- it's missing the logit
    half entirely, and feeding a bare {0,1} as "probability" with no partner
    logit would put an activation into the fusion Linear layer that it never
    saw during training. Instead a logit is synthesized consistent with a
    binary label at a fixed confidence level: prob = confidence for label=1,
    (1-confidence) for label=0, then logit = logit(prob). This keeps both
    halves of the fusion input internally consistent (probs = sigmoid(logits),
    exactly as at train/inference time) while making them maximally
    informative -- as close to "certain" as the fusion layer's own input
    format allows. confidence is a tunable flag (--oracle-confidence) rather
    than a hardcoded extreme (e.g. 0.9999) so the magnitude of the synthetic
    logits stays in a plausible range relative to what the concept head
    itself produces (see Step 10's calibration histograms) instead of
    injecting activations far outside anything the fusion layer's weights
    were ever trained against.
    """
    assert 0.5 < confidence < 1.0, "--oracle-confidence must be in (0.5, 1.0)"
    concept_df = pd.read_csv(val_concept_labels_path)

    # ROW-ALIGNMENT SAFETY CHECK: a duplicate row_id in val_concept_labels.csv
    # silently inflates the merge past len(val_df), which shifts every
    # downstream positional index and invalidates the whole experiment without
    # raising any error on its own. Catch that here, hard, before it can happen.
    dup_ids = concept_df["row_id"][concept_df["row_id"].duplicated()].unique()
    assert len(dup_ids) == 0, (
        f"val_concept_labels.csv has duplicate row_id values: {list(dup_ids)[:5]}"
        f"{'...' if len(dup_ids) > 5 else ''} -- a duplicate would silently "
        f"inflate the merge and misalign every oracle row with the wrong "
        f"validation example. Deduplicate the CSV before running --oracle-concepts."
    )

    merged = val_df[["row_id"]].merge(concept_df, on="row_id", how="left")
    assert len(merged) == len(val_df), (
        f"oracle merge produced {len(merged)} rows but val_df has {len(val_df)} -- "
        f"row alignment would be broken. Aborting rather than silently mis-scoring."
    )
    assert (merged["row_id"].values == val_df["row_id"].values).all(), (
        "oracle merge did not preserve val_df's row order -- this should be "
        "structurally impossible with how='left', but refusing to proceed on "
        "an unverified alignment rather than risk silently wrong results."
    )
    n_missing = merged[CONCEPTS[0]].isna().sum()
    if n_missing > 0:
        print(f"  [oracle] WARNING: {n_missing}/{len(merged)} validation rows have "
             f"no entry in val_concept_labels.csv -- missing concept labels for "
             f"those rows default to False (0), which is a real gap in oracle "
             f"coverage, not a labeling signal. Check --val-concept-labels covers "
             f"the full validation set if this number is large.")
    for c in CONCEPTS:
        merged[c] = merged[c].fillna(False).astype(float)
    labels = merged[CONCEPTS].values  # (N, 9), verified aligned to val_df row order
    probs = labels * confidence + (1 - labels) * (1 - confidence)
    logits = np.log(probs / (1 - probs))
    return (torch.tensor(probs, dtype=torch.float32),
            torch.tensor(logits, dtype=torch.float32))


@torch.no_grad()
def get_concept_and_risk_outputs(model, loader, device, use_bf16, ablate_idx=None, oracle=None):
    """Runs the model's forward pass MANUALLY so that one concept's
    contribution to the fusion layer can be intercepted and ablated (only
    meaningful for the bottleneck architecture -- for auxiliary models
    this is a no-op since fusion doesn't exist).

    oracle: optional (probs_full, logits_full) tensors covering the WHOLE
    val set, aligned to the loader's row order (shuffle=False), from
    build_oracle_concept_tensors. When set, the fusion layer receives
    these ground-truth-derived values instead of the concept head's own
    predictions -- everything else (encoder, fusion weights, risk heads)
    is untouched. Mutually exclusive with ablate_idx in practice (oracle
    mode isn't used for ablation runs), but not enforced since nothing
    breaks if both are passed -- ablate_idx would just zero out oracle
    values for that index."""
    model.eval()
    all_concept_probs, all_cat_logits, all_ord_logits, all_labels = [], [], [], []
    has_fusion = hasattr(model, "fusion")

    for batch in loader:
        idx = batch.pop("idx")
        batch.pop("concept_labels", None)
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            out = model.encoder(**batch)
            pooled = model.dropout(out.last_hidden_state[:, 0])
            pooled = pooled.to(model.concept_head.weight.dtype)
            concept_logits = model.concept_head(pooled)
            concept_probs = torch.sigmoid(concept_logits)

            if has_fusion:
                if oracle is not None:
                    oracle_probs_full, oracle_logits_full = oracle
                    cp = oracle_probs_full[idx].to(device).to(concept_probs.dtype)
                    cl = oracle_logits_full[idx].to(device).to(concept_logits.dtype)
                else:
                    cp = concept_probs.clone()
                    cl = concept_logits.clone()
                if ablate_idx is not None:
                    if ablate_idx == "all":
                        cp[:, :] = 0.0
                        cl[:, :] = 0.0
                    else:
                        cp[:, ablate_idx] = 0.0
                        cl[:, ablate_idx] = 0.0
                fusion_input = torch.cat([pooled, cp, cl], dim=-1)
                fused = model.fusion(fusion_input)
                cat_logits = model.cat_head(fused)
                ord_logits = model.ord_head(fused)
            else:
                # auxiliary architecture — risk heads read raw pooled embedding,
                # ablation here is a structural no-op (kept for completeness)
                cat_logits = model.cat_head(pooled)
                ord_logits = model.ord_head(pooled)

        all_concept_probs.append(concept_probs.float().cpu())
        all_cat_logits.append(cat_logits.float().cpu())
        all_ord_logits.append(ord_logits.float().cpu())
        all_labels.append(labels)

    return (torch.cat(all_concept_probs), torch.cat(all_cat_logits),
            torch.cat(all_ord_logits), torch.cat(all_labels))


def combine_logits(cat_logits, ord_logits):
    cat_p = torch.softmax(cat_logits, dim=-1)
    cond = torch.sigmoid(ord_logits)
    cum = torch.cumprod(cond, dim=-1)
    ones = torch.ones_like(cum[:, :1])
    upper = torch.cat([ones, cum], dim=-1)
    lower = torch.cat([cum, torch.zeros_like(cum[:, :1])], dim=-1)
    ord_p = (upper - lower).clamp_min(1e-8)
    return 0.5 * cat_p + 0.5 * ord_p


def f1_from_outputs(cat_logits, ord_logits, labels):
    probs = combine_logits(cat_logits, ord_logits)
    gold = [RISK_LEVELS[i] for i in labels.tolist()]
    pred = [RISK_LEVELS[i] for i in probs.argmax(-1).tolist()]
    return risk_weighted_f1(gold, pred), pred, gold


def per_class_f1(gold, pred):
    """One-vs-rest precision/recall/F1 for each RISK_LEVELS class.
    No sklearn dependency -- plain counting."""
    out = {}
    for cls in RISK_LEVELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(gold, pred) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(gold, pred) if g == cls and p != cls)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        out[cls] = f1
    return out


def step1_concept_metrics(concept_probs, val_concept_labels_path, row_ids):
    if val_concept_labels_path is None:
        print("\n[STEP 1] SKIPPED — no --val-concept-labels provided")
        return
    print("\n" + "=" * 90)
    print("STEP 1: CONCEPT HEAD ACCURACY / PRECISION / RECALL / F1")
    print("=" * 90)
    print("CAVEAT: val_concept_labels.csv is DeepSeek-generated, not human-annotated.")
    print("These numbers measure how well the concept head imitates DeepSeek's")
    print("pseudo-labels, NOT whether the underlying concepts are objectively")
    print("correct. High agreement = learned the pseudo-labels well; it does not")
    print("independently confirm the pseudo-labels themselves are right.\n")
    gold_df = pd.read_csv(val_concept_labels_path).set_index("row_id")
    preds = (concept_probs >= 0.5).int().numpy()
    print(f"{'concept':<25}{'acc':>8}{'prec':>8}{'rec':>8}{'f1':>8}")
    for j, c in enumerate(CONCEPTS):
        gold = gold_df.loc[row_ids, c].astype(int).values
        pred = preds[:, j]
        tp = int(((pred == 1) & (gold == 1)).sum())
        fp = int(((pred == 1) & (gold == 0)).sum())
        fn = int(((pred == 0) & (gold == 1)).sum())
        tn = int(((pred == 0) & (gold == 0)).sum())
        acc = (tp + tn) / max(len(gold), 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        print(f"{c:<25}{acc:>8.1%}{prec:>8.1%}{rec:>8.1%}{f1:>8.1%}")


def step2_ablation(model, val_loader, device, use_bf16, baseline_f1):
    if not hasattr(model, "fusion"):
        print("\n[STEP 2] SKIPPED — no fusion layer (auxiliary architecture, "
              "concepts don't reach the risk head, ablation is structurally meaningless)")
        return
    print("\n" + "=" * 90)
    print("STEP 2: CONCEPT ABLATION (bottleneck-only)")
    print("=" * 90)
    print(f"baseline val weighted-F1 (no ablation): {baseline_f1:.4f}\n")
    results = []
    for j, c in enumerate(CONCEPTS):
        _, cat_l, ord_l, labels = get_concept_and_risk_outputs(
            model, val_loader, device, use_bf16, ablate_idx=j)
        f1, _, _ = f1_from_outputs(cat_l, ord_l, labels)
        results.append((c, f1, f1 - baseline_f1))
    results.sort(key=lambda r: r[2])  # most damaging removal first
    print(f"{'concept':<25}{'F1 w/o it':>12}{'delta':>10}")
    for c, f1, delta in results:
        tag = " <-- IMPORTANT" if delta < -0.005 else ("  (ignored)" if abs(delta) < 0.001 else "")
        print(f"{c:<25}{f1:>12.4f}{delta:>+10.4f}{tag}")


def step3_fusion_weights(model):
    if not hasattr(model, "fusion"):
        print("\n[STEP 3] SKIPPED — no fusion layer (auxiliary architecture)")
        return
    print("\n" + "=" * 90)
    print("STEP 3: FUSION WEIGHT NORMS")
    print("=" * 90)
    W = model.fusion[0].weight.detach().float().cpu()  # (h, h + 2*n_concepts)
    h = model.encoder.config.hidden_size
    n = len(CONCEPTS)
    w_emb = W[:, :h]
    w_probs = W[:, h:h + n]
    w_logits = W[:, h + n:h + 2 * n]
    norm_emb = w_emb.norm().item()
    norm_probs = w_probs.norm().item()
    norm_logits = w_logits.norm().item()
    total = norm_emb + norm_probs + norm_logits
    print(f"  embedding block norm:      {norm_emb:>10.2f}  ({100*norm_emb/total:.1f}%)")
    print(f"  concept_probs block norm:  {norm_probs:>10.2f}  ({100*norm_probs/total:.1f}%)")
    print(f"  concept_logits block norm: {norm_logits:>10.2f}  ({100*norm_logits/total:.1f}%)")
    print("\n  CAVEAT: weight norm alone can mislead — a small weight times a large")
    print("  activation can contribute more than a large weight times a near-zero")
    print("  activation. Treat this as a CHEAP SANITY CHECK only. STEP 4 (actual")
    print("  per-example activation contribution) is the trustworthy answer to")
    print("  whether concepts are really being used — defer to that, not this.")


@torch.no_grad()
def step4_activation_contribution(model, val_loader, device, use_bf16):
    if not hasattr(model, "fusion"):
        print("\n[STEP 4] SKIPPED — no fusion layer (auxiliary architecture)")
        return
    print("\n" + "=" * 90)
    print("STEP 4: FUSION ACTIVATION CONTRIBUTION (per-example, not just weights)")
    print("=" * 90)
    model.eval()
    h = model.encoder.config.hidden_size
    n = len(CONCEPTS)
    W = model.fusion[0].weight.detach()  # (h_out, h + 2n)
    W_emb, W_probs, W_logits = W[:, :h], W[:, h:h + n], W[:, h + n:h + 2 * n]

    emb_contrib_total, probs_contrib_total, logits_contrib_total, n_examples = 0.0, 0.0, 0.0, 0
    for batch in val_loader:
        batch.pop("idx")
        batch.pop("concept_labels", None)
        batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            out = model.encoder(**batch)
            pooled = model.dropout(out.last_hidden_state[:, 0]).to(model.concept_head.weight.dtype)
            concept_logits = model.concept_head(pooled)
            concept_probs = torch.sigmoid(concept_logits)

        pooled_f = pooled.float()
        emb_term = pooled_f @ W_emb.float().T
        probs_term = concept_probs.float() @ W_probs.float().T
        logits_term = concept_logits.float() @ W_logits.float().T

        emb_contrib_total += emb_term.norm(dim=-1).sum().item()
        probs_contrib_total += probs_term.norm(dim=-1).sum().item()
        logits_contrib_total += logits_term.norm(dim=-1).sum().item()
        n_examples += pooled.shape[0]

    total = emb_contrib_total + probs_contrib_total + logits_contrib_total
    print(f"  embedding contribution:      {100*emb_contrib_total/total:>6.1f}%")
    print(f"  concept_probs contribution:  {100*probs_contrib_total/total:>6.1f}%")
    print(f"  concept_logits contribution: {100*logits_contrib_total/total:>6.1f}%")


def step5_error_analysis(concept_probs, cat_logits, ord_logits, labels, val_df, n_each=20):
    print("\n" + "=" * 90)
    print("STEP 5: ERROR ANALYSIS — correct vs incorrect predictions")
    print("=" * 90)
    f1, pred, gold = f1_from_outputs(cat_logits, ord_logits, labels)
    correct_idx = [i for i in range(len(gold)) if pred[i] == gold[i]]
    wrong_idx = [i for i in range(len(gold)) if pred[i] != gold[i]]

    def summarize(idxs, label):
        if not idxs:
            print(f"  {label}: none")
            return
        sub = concept_probs[idxs]
        mean_conf = sub.mean(dim=0)
        print(f"  {label} (n={len(idxs)}) — mean predicted concept probability:")
        for j, c in enumerate(CONCEPTS):
            print(f"    {c:<25} {mean_conf[j].item():.2f}")

    import random
    random.seed(1)
    c_sample = random.sample(correct_idx, min(n_each, len(correct_idx)))
    w_sample = random.sample(wrong_idx, min(n_each, len(wrong_idx)))
    summarize(c_sample, "CORRECT predictions")
    print()
    summarize(w_sample, "WRONG predictions")
    print(f"\n  Total: {len(correct_idx)} correct, {len(wrong_idx)} wrong "
         f"(F1={f1:.4f})")
    print("  If concept probabilities look similar between the two groups, the "
         "risk head likely isn't discriminating based on concepts at all.")


def step6_agreement_analysis(concept_probs, cat_logits, ord_logits, labels,
                             val_concept_labels_path, row_ids):
    if val_concept_labels_path is None:
        print("\n[STEP 6] SKIPPED — needs --val-concept-labels")
        return
    print("\n" + "=" * 90)
    print("STEP 6: AGREEMENT ANALYSIS — are risk errors caused by bad concepts,")
    print("        or by bad reasoning over otherwise-fine concepts?")
    print("=" * 90)
    f1, pred, gold = f1_from_outputs(cat_logits, ord_logits, labels)
    correct_idx = [i for i in range(len(gold)) if pred[i] == gold[i]]
    wrong_idx = [i for i in range(len(gold)) if pred[i] != gold[i]]

    gold_df = pd.read_csv(val_concept_labels_path).set_index("row_id")
    preds_bin = (concept_probs >= 0.5).int().numpy()

    def concept_acc_for(idxs):
        accs = []
        sub_row_ids = [row_ids[i] for i in idxs]
        for j, c in enumerate(CONCEPTS):
            gold_vals = gold_df.loc[sub_row_ids, c].astype(int).values
            pred_vals = preds_bin[idxs, j]
            accs.append((gold_vals == pred_vals).mean())
        return accs

    acc_correct = concept_acc_for(correct_idx)
    acc_wrong = concept_acc_for(wrong_idx)

    print(f"{'concept':<25}{'acc|risk-correct':>18}{'acc|risk-wrong':>18}{'gap':>8}")
    for j, c in enumerate(CONCEPTS):
        gap = acc_correct[j] - acc_wrong[j]
        print(f"{c:<25}{acc_correct[j]:>18.1%}{acc_wrong[j]:>18.1%}{gap:>+8.1%}")

    mean_correct, mean_wrong = np.mean(acc_correct), np.mean(acc_wrong)
    print(f"\n  OVERALL concept accuracy | risk prediction CORRECT: {mean_correct:.1%}")
    print(f"  OVERALL concept accuracy | risk prediction WRONG:   {mean_wrong:.1%}")
    print("\n  CAVEAT: do not read the overall average alone. A large gap on one")
    print("  important concept can hide behind a flat overall average if every")
    print("  other concept stays near-perfect. Check the per-concept table above")
    print("  first — the max single-concept gap is the number that matters most:")
    max_gap_idx = int(np.argmax([ac - aw for ac, aw in zip(acc_correct, acc_wrong)]))
    print(f"    largest single-concept gap: {CONCEPTS[max_gap_idx]} "
         f"({acc_correct[max_gap_idx]:.1%} vs {acc_wrong[max_gap_idx]:.1%}, "
         f"gap {acc_correct[max_gap_idx]-acc_wrong[max_gap_idx]:+.1%})")
    if mean_correct - mean_wrong > 0.05:
        print("\n  --> concept quality drops meaningfully when risk is wrong: bad "
             "concepts are likely CAUSING risk errors (garbage in, garbage out).")
    else:
        print("\n  --> average concept quality is similar regardless of risk "
             "correctness, but check the per-concept gap above before concluding "
             "fusion reasoning is the problem — a single collapsed concept can "
             "still be the real cause even when the average looks fine.")


def step7_per_class_ablation(model, val_loader, device, use_bf16, baseline_per_class):
    if not hasattr(model, "fusion"):
        print("\n[STEP 7] SKIPPED — no fusion layer (auxiliary architecture)")
        return
    print("\n" + "=" * 90)
    print("STEP 7: PER-CLASS CONCEPT ABLATION")
    print("STEP 2 measures overall F1 -- this breaks the same ablation down by")
    print("risk class, since a concept can matter a lot for one class and not")
    print("at all for another, which overall F1 would hide.")
    print("=" * 90)
    header = f"{'concept':<25}" + "".join(f"{c+' dF1':>16}" for c in RISK_LEVELS)
    print(header)
    for j, c in enumerate(CONCEPTS):
        _, cat_l, ord_l, labels = get_concept_and_risk_outputs(
            model, val_loader, device, use_bf16, ablate_idx=j)
        _, pred, gold = f1_from_outputs(cat_l, ord_l, labels)
        pc = per_class_f1(gold, pred)
        row = f"{c:<25}"
        for cls in RISK_LEVELS:
            delta = pc[cls] - baseline_per_class[cls]
            row += f"{delta:>+16.4f}"
        print(row)
    print("\n  Read each column independently -- a concept can be near-zero for")
    print("  the overall/majority class but essential for a rare, high-stakes one.")


def step8_zero_all_concepts(model, val_loader, device, use_bf16, baseline_f1):
    if not hasattr(model, "fusion"):
        print("\n[STEP 8] SKIPPED — no fusion layer (auxiliary architecture)")
        return
    print("\n" + "=" * 90)
    print("STEP 8: ZERO ALL CONCEPTS AT ONCE (embedding-only vs embedding+concepts)")
    print("=" * 90)
    _, cat_l, ord_l, labels = get_concept_and_risk_outputs(
        model, val_loader, device, use_bf16, ablate_idx="all")
    f1_no_concepts, _, _ = f1_from_outputs(cat_l, ord_l, labels)
    delta = f1_no_concepts - baseline_f1
    print(f"  F1 with embedding + concepts (normal):  {baseline_f1:.4f}")
    print(f"  F1 with embedding only (all zeroed):    {f1_no_concepts:.4f}")
    print(f"  delta:                                  {delta:+.4f}")
    if abs(delta) < 0.005:
        print("\n  --> performance barely changes with concepts fully removed: the "
             "concept bottleneck is contributing ~nothing at inference. This is the "
             "single strongest signal that the bottleneck, as trained, failed.")
    else:
        print("\n  --> removing all concepts hurts performance: concepts are "
             "carrying real signal for the fusion layer.")


def step9_concept_correlation(concept_probs):
    print("\n" + "=" * 90)
    print("STEP 9: CONCEPT PREDICTION CORRELATION MATRIX")
    print("If concepts are near-perfectly correlated, the network doesn't need")
    print("all of them -- one carries the redundant information for the rest.")
    print("=" * 90)
    mat = concept_probs.numpy()
    corr = np.corrcoef(mat, rowvar=False)
    short = [c[:10] for c in CONCEPTS]
    print(" " * 13 + "".join(f"{s:>12}" for s in short))
    for i, c in enumerate(short):
        row = f"{c:<13}" + "".join(f"{corr[i, j]:>12.2f}" for j in range(len(CONCEPTS)))
        print(row)
    n = len(CONCEPTS)
    pairs = [(CONCEPTS[i], CONCEPTS[j], corr[i, j])
             for i in range(n) for j in range(i + 1, n)]
    pairs.sort(key=lambda p: -abs(p[2]))
    print("\n  Most correlated pairs:")
    for a, b, r in pairs[:5]:
        flag = "  <-- possibly redundant" if abs(r) > 0.85 else ""
        print(f"    {a:<25} vs {b:<25} r={r:+.2f}{flag}")


def step10_concept_calibration(concept_probs):
    print("\n" + "=" * 90)
    print("STEP 10: CONCEPT PROBABILITY CALIBRATION (histogram per concept)")
    print("If a concept is always near 0 or 1 and almost never in between, the")
    print("fusion layer is effectively receiving a binary signal for it, which")
    print("changes how much nuance it can learn to use.")
    print("=" * 90)
    bins = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001]
    labels_bins = ["<0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", ">0.9"]
    mat = concept_probs.numpy()
    print(f"{'concept':<25}" + "".join(f"{b:>10}" for b in labels_bins))
    for j, c in enumerate(CONCEPTS):
        counts, _ = np.histogram(mat[:, j], bins=bins)
        pct = 100 * counts / max(counts.sum(), 1)
        print(f"{c:<25}" + "".join(f"{p:>9.1f}%" for p in pct))
    print("\n  High mass only in the '<0.1' and '>0.9' columns for a concept means")
    print("  it is behaving almost binary; heavy mass in the middle columns means")
    print("  it is carrying genuine graded/uncertain information.")


def step11_confidence_vs_correctness(concept_probs, cat_logits, ord_logits, labels):
    print("\n" + "=" * 90)
    print("STEP 11: CONCEPT CONFIDENCE vs RISK CORRECTNESS")
    print("Tests whether the fusion layer's errors cluster on examples where the")
    print("concept head itself was unsure -- if so, a confidence-gated fusion")
    print("(only trust concepts when the concept head is confident) is motivated.")
    print("=" * 90)
    f1, pred, gold = f1_from_outputs(cat_logits, ord_logits, labels)
    correct_idx = [i for i in range(len(gold)) if pred[i] == gold[i]]
    wrong_idx = [i for i in range(len(gold)) if pred[i] != gold[i]]
    conf = (2 * (concept_probs - 0.5).abs()).mean(dim=-1)  # 0 = maximally unsure, 1 = maximally sure
    conf_correct = conf[correct_idx].mean().item() if correct_idx else float("nan")
    conf_wrong = conf[wrong_idx].mean().item() if wrong_idx else float("nan")
    print(f"  mean concept confidence | risk CORRECT: {conf_correct:.3f}")
    print(f"  mean concept confidence | risk WRONG:   {conf_wrong:.3f}")
    if conf_correct - conf_wrong > 0.03:
        print("\n  --> risk errors cluster on low-concept-confidence examples. "
             "Gating the fusion layer's reliance on concepts by confidence is "
             "directly motivated by this result.")
    else:
        print("\n  --> concept confidence looks similar for correct vs wrong risk "
             "predictions; low confidence alone doesn't explain the errors, so "
             "confidence gating is less likely to help by itself.")


def step12_manual_audit(concept_probs, cat_logits, ord_logits, labels, val_df,
                        val_concept_labels_path, row_ids, n=20):
    print("\n" + "=" * 90)
    print("STEP 12: MANUAL MISTAKE AUDIT (20 wrong predictions, full trace)")
    print("The other steps tell you THAT something is off. This is the one that")
    print("tells you WHAT the fusion layer failed to reason about.")
    print("=" * 90)
    f1, pred, gold = f1_from_outputs(cat_logits, ord_logits, labels)
    wrong_idx = [i for i in range(len(gold)) if pred[i] != gold[i]]
    import random
    random.seed(1)
    sample = random.sample(wrong_idx, min(n, len(wrong_idx)))
    gold_concept_df = None
    if val_concept_labels_path is not None:
        gold_concept_df = pd.read_csv(val_concept_labels_path).set_index("row_id")
    for i in sample:
        post = str(val_df.iloc[i]["post"])
        post_short = post[:200] + ("..." if len(post) > 200 else "")
        print(f"\n  --- row_id={row_ids[i]} ---")
        print(f"  post: {post_short}")
        print(f"  gold={gold[i]}   pred={pred[i]}")
        print(f"  {'concept':<25}{'pred_prob':>10}{'pseudo_label':>14}")
        for j, c in enumerate(CONCEPTS):
            p = concept_probs[i, j].item()
            if gold_concept_df is not None and row_ids[i] in gold_concept_df.index:
                gl = int(gold_concept_df.loc[row_ids[i], c])
            else:
                gl = "n/a"
            print(f"  {c:<25}{p:>10.2f}{str(gl):>14}")
    print("\n  Look for cases where the concept prediction matches the pseudo-label")
    print("  (so the concept head 'got it right') but the risk prediction is still")
    print("  wrong -- that is direct evidence of a fusion/reasoning failure rather")
    print("  than a concept-quality failure, for that specific example.")


def macro_concept_f1(concept_probs, val_concept_labels_path, row_ids):
    """Same per-concept F1 computation as step1_concept_metrics, but returns
    just the macro-average number (no printing) so step0 can fold it into
    its own summary table instead of duplicating step1's full printout."""
    gold_df = pd.read_csv(val_concept_labels_path).set_index("row_id")
    preds = (concept_probs >= 0.5).int().numpy()
    f1s = []
    for j, c in enumerate(CONCEPTS):
        gold = gold_df.loc[row_ids, c].astype(int).values
        pred = preds[:, j]
        tp = int(((pred == 1) & (gold == 1)).sum())
        fp = int(((pred == 1) & (gold == 0)).sum())
        fn = int(((pred == 0) & (gold == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-8))
    return float(np.mean(f1s))


def step0_oracle_concepts(model, val_loader, device, use_bf16, baseline_f1,
                          baseline_per_class, val_df, val_concept_labels_path,
                          concept_probs, row_ids, confidences):
    """ORACLE CONCEPTS EVALUATION -- not a training experiment. Same
    checkpoint, same encoder, same fusion weights, same risk heads.
    The only thing that changes is what the fusion layer receives in
    place of the concept head's own predictions: ground-truth-derived
    values instead. This isolates the question Step 8 leaves open --
    Step 8 shows the fusion layer isn't using the concepts it currently
    gets; this asks whether it COULD use concepts at all, if they were
    perfect. Case A (oracle F1 >> baseline): concept quality is still
    the bottleneck, fusion is fine. Case B (oracle F1 ~= baseline):
    fusion itself can't make use of concept information even when it's
    perfect -- the architecture, not the concept predictor, is the
    problem.

    confidences: list of oracle-confidence values to sweep. A single-value
    list still prints the full table; more than one adds a stability check
    across confidence levels at the end."""
    if not hasattr(model, "fusion"):
        print("\n[ORACLE] SKIPPED — no fusion layer (auxiliary architecture, "
              "concepts never reach the risk head, oracle substitution is moot)")
        return
    if val_concept_labels_path is None:
        print("\n[ORACLE] SKIPPED — needs --val-concept-labels")
        return
    print("\n" + "=" * 90)
    print("ORACLE CONCEPTS EVALUATION (evaluation only -- no retraining, no")
    print("checkpoint changes; fusion layer, risk heads, encoder all identical)")
    print("=" * 90)

    concept_f1 = macro_concept_f1(concept_probs, val_concept_labels_path, row_ids)

    sweep_results = []
    for confidence in confidences:
        logit_mag = np.log(confidence / (1 - confidence))
        print(f"\n  --- oracle confidence = {confidence} "
             f"(prob={confidence:.2f}/{1-confidence:.2f}, logit=±{logit_mag:.2f}) ---")
        oracle_probs, oracle_logits = build_oracle_concept_tensors(
            val_df, val_concept_labels_path, confidence=confidence)
        _, cat_l, ord_l, labels = get_concept_and_risk_outputs(
            model, val_loader, device, use_bf16, oracle=(oracle_probs, oracle_logits))
        oracle_f1, oracle_pred, oracle_gold = f1_from_outputs(cat_l, ord_l, labels)
        oracle_per_class = per_class_f1(oracle_gold, oracle_pred)
        sweep_results.append((confidence, oracle_f1, oracle_per_class))

        print(f"\n  {'stage':<32}{'value':>10}")
        print(f"  {'predicted-concept macro-F1':<32}{concept_f1:>10.4f}")
        print(f"  {'oracle-concept macro-F1':<32}{1.0:>10.4f}  (ground truth, by construction)")
        print(f"  {'risk F1, predicted concepts':<32}{baseline_f1:>10.4f}")
        print(f"  {'risk F1, oracle concepts':<32}{oracle_f1:>10.4f}")
        print(f"  {'delta':<32}{oracle_f1 - baseline_f1:>+10.4f}")
        print(f"\n  {'class':<12}{'predicted F1':>14}{'oracle F1':>14}{'delta':>10}")
        for cls in RISK_LEVELS:
            d = oracle_per_class[cls] - baseline_per_class[cls]
            print(f"  {cls:<12}{baseline_per_class[cls]:>14.4f}{oracle_per_class[cls]:>14.4f}{d:>+10.4f}")

    if len(sweep_results) > 1:
        print("\n  --- confidence sweep stability ---")
        print(f"  {'confidence':<12}{'oracle F1':>12}{'delta vs baseline':>20}")
        for conf, f1, _ in sweep_results:
            print(f"  {conf:<12}{f1:>12.4f}{f1 - baseline_f1:>+20.4f}")
        spread = max(f1 for _, f1, _ in sweep_results) - min(f1 for _, f1, _ in sweep_results)
        print(f"\n  spread across confidence levels: {spread:.4f}")
        if spread < 0.01:
            print("  --> oracle result is stable across confidence levels -- the "
                 "conclusion doesn't hinge on the specific synthetic-logit choice.")
        else:
            print("  --> oracle result VARIES meaningfully with confidence -- treat "
                 "the single-confidence verdict with caution and report the range, "
                 "not a single number.")

    final_f1 = sweep_results[-1][1]
    delta = final_f1 - baseline_f1
    if delta > 0.02:
        print("\n  --> CASE A: perfect concepts meaningfully improve risk prediction. "
             "The fusion layer CAN use concept information when it's reliable -- "
             "the concept predictor's remaining quality gap is the bottleneck, "
             "not the fusion mechanism. Prioritize concept-head quality further "
             "before redesigning fusion.")
    else:
        print("\n  --> CASE B: even ground-truth concepts don't move risk prediction. "
             "The fusion layer is structurally unable (or untrained) to make use "
             "of concept information at all, independent of concept quality. This "
             "is strong evidence the FUSION MECHANISM itself is the bottleneck -- "
             "concat+MLP fusion is not learning to route this information into "
             "the risk heads, and a different fusion mechanism (gating, "
             "cross-attention) is now justified rather than premature.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-concept-labels", default=None)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--quick", action="store_true",
                    help="Run only Steps 1, 4, and 8 -- for checking mid-Stage-2 "
                         "whether fusion has started using a frozen concept head, "
                         "without paying for the full 12-step suite.")
    ap.add_argument("--oracle-concepts", action="store_true",
                    help="Evaluation-only: feed val_concept_labels.csv into the "
                         "fusion layer instead of the concept head's own "
                         "predictions, to test whether fusion COULD use perfect "
                         "concepts. No retraining, checkpoint/architecture "
                         "unchanged. Requires --val-concept-labels.")
    ap.add_argument("--oracle-confidence", type=float, default=0.98,
                    help="Probability magnitude used to synthesize oracle "
                         "probs/logits from binary labels (see "
                         "build_oracle_concept_tensors docstring). Default 0.98. "
                         "Ignored if --oracle-confidence-sweep is passed.")
    ap.add_argument("--oracle-confidence-sweep", type=str, default=None,
                    help="Comma-separated list of confidence values to sweep, "
                         "e.g. '0.9,0.95,0.98', to check whether the oracle "
                         "result is stable or an artifact of one specific "
                         "synthetic-logit choice. Overrides --oracle-confidence.")
    ap.add_argument("--log-dir", default="analysis_logs",
                    help="Directory for the timestamped run transcript. Never "
                         "the checkpoint directory, and never overwritten -- "
                         "each run gets its own timestamped file here.")
    ap.add_argument("--no-log", action="store_true",
                    help="Disable log-file capture, print to stdout only.")
    args = ap.parse_args()

    tee = None
    real_stdout = sys.stdout
    if not args.no_log:
        log_path = make_log_path(args.log_dir, args.checkpoint, args.oracle_concepts)
        tee = Tee(log_path, real_stdout)
        sys.stdout = tee
        print(f"[log] writing full transcript to {log_path}\n")

    try:
        with open(os.path.join(args.checkpoint, "run_config.json")) as f:
            cfg = json.load(f)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
        print(f"[precision] {'bf16' if use_bf16 else 'fp32'} on {device}")

        tok = AutoTokenizer.from_pretrained(cfg["model"])
        va_df = pd.read_csv(args.val)
        row_ids = va_df["row_id"].tolist()

        va_loader = DataLoader(
            PostDataset(va_df, tok, cfg["max_len"], cfg["user_context"], concept_df=None),
            batch_size=args.batch, shuffle=False, num_workers=0)

        model = ConceptRiskModel(cfg["model"], alpha_ordinal=cfg["alpha_ordinal"],
                                 focal_gamma=cfg["focal_gamma"],
                                 concept_loss_weight=cfg["concept_loss_weight"]).to(device)
        state = torch.load(os.path.join(args.checkpoint, "model_final.pt"), map_location=device)
        model.load_state_dict(state, strict=False)

        concept_probs, cat_logits, ord_logits, labels = get_concept_and_risk_outputs(
            model, va_loader, device, use_bf16, ablate_idx=None)
        baseline_f1, baseline_pred, baseline_gold = f1_from_outputs(cat_logits, ord_logits, labels)
        baseline_per_class = per_class_f1(baseline_gold, baseline_pred)
        print(f"\nBaseline val weighted-F1 for this checkpoint: {baseline_f1:.4f}")
        print("Baseline per-class F1: " +
             ", ".join(f"{cls}={f:.3f}" for cls, f in baseline_per_class.items()))

        if args.oracle_concepts:
            if args.oracle_confidence_sweep:
                confidences = [float(x) for x in args.oracle_confidence_sweep.split(",")]
            else:
                confidences = [args.oracle_confidence]
            step0_oracle_concepts(model, va_loader, device, use_bf16, baseline_f1,
                                  baseline_per_class, va_df, args.val_concept_labels,
                                  concept_probs, row_ids, confidences)

        if args.quick:
            print("\n[--quick] running only Steps 1, 4, 8\n")
            step1_concept_metrics(concept_probs, args.val_concept_labels, row_ids)
            step4_activation_contribution(model, va_loader, device, use_bf16)
            step8_zero_all_concepts(model, va_loader, device, use_bf16, baseline_f1)
            return

        step1_concept_metrics(concept_probs, args.val_concept_labels, row_ids)
        step2_ablation(model, va_loader, device, use_bf16, baseline_f1)
        step3_fusion_weights(model)
        step4_activation_contribution(model, va_loader, device, use_bf16)
        step5_error_analysis(concept_probs, cat_logits, ord_logits, labels, va_df)
        step6_agreement_analysis(concept_probs, cat_logits, ord_logits, labels,
                                 args.val_concept_labels, row_ids)
        step7_per_class_ablation(model, va_loader, device, use_bf16, baseline_per_class)
        step8_zero_all_concepts(model, va_loader, device, use_bf16, baseline_f1)
        step9_concept_correlation(concept_probs)
        step10_concept_calibration(concept_probs)
        step11_confidence_vs_correctness(concept_probs, cat_logits, ord_logits, labels)
        step12_manual_audit(concept_probs, cat_logits, ord_logits, labels, va_df,
                            args.val_concept_labels, row_ids)
    finally:
        if tee is not None:
            sys.stdout = real_stdout
            tee.close()


if __name__ == "__main__":
    main()
