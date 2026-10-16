"""
train_concept_only_mlp.py

DIAGNOSTIC EXPERIMENT -- NOT A DEPLOYABLE MODEL.

Estimates the CEILING of what the 9 concept labels alone can explain
about risk level, with no text, no encoder, no fusion layer:

    9 binary concept pseudo-labels -> small MLP -> risk prediction

This answers a different question than the Oracle Concepts experiment
did. Oracle Concepts asked "does the TRAINED fusion layer make use of
perfect concepts?" (answer: no, F1 was flat at ~0.7358). This script
asks "are the 9 concepts THEMSELVES sufficient to predict risk, under
the best possible mapping from concepts to risk, with no fusion
architecture in the way at all?"

IMPORTANT CAVEATS (read before trusting the number this script prints):

1. This measures the ceiling of the PSEUDO-labels (DeepSeek-generated),
   not the true underlying concepts. A low score could mean the concept
   taxonomy is insufficient, OR that DeepSeek's labeling is too noisy to
   be a reliable ceiling estimate -- this script cannot distinguish
   those two explanations on its own.

2. The comparison target matters. Comparing this ceiling to the
   BOTTLENECK MODEL's F1 (via --bottleneck-run-config) tells you
   whether the fusion architecture is already close to the ceiling --
   it does NOT tell you how much information the concept taxonomy
   discards relative to the raw text, since there is no text-only
   (no-concept-branch) classifier's F1 on record to compare against.
   If you want that second comparison, it needs a separate text-only
   baseline run; this script only measures the concept-only ceiling.

3. Two independent ceiling estimates are reported, not one: the MLP
   result, AND a deterministic lookup-table baseline (majority risk
   label per unique 9-bit concept pattern seen in training, applied to
   val). The lookup table has zero training variance. If the MLP and
   lookup-table numbers disagree substantially, trust the disagreement
   as a sign the MLP result may be under/over-fit, not as evidence
   about the concepts themselves -- re-run the MLP with a different
   seed or more epochs before concluding anything from it alone.

Usage:
  python src/train_concept_only_mlp.py \
      --train artifacts/train_split.csv \
      --val artifacts/val_split.csv \
      --train-concept-labels artifacts/train_concept_labels.csv \
      --val-concept-labels artifacts/val_concept_labels.csv \
      --outdir runs/concept_only_mlp_diagnostic \
      --class-weighting \
      --bottleneck-run-config runs/risk_concepts_bottleneck_v2/run_config.json \
      --seed 13
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from scoring import RISK_LEVELS, normalize_label, risk_weighted_f1

LABEL2ID = {l: i for i, l in enumerate(RISK_LEVELS)}
CONCEPTS = [
    "method_mechanism", "plan_present", "completed_act", "urgency_imminence",
    "speaker_is_author", "farewell_framing", "protective_factor", "negation",
    "past_attempt_history",
]


def load_concept_features(split_df, concept_labels_path, split_name):
    """Merge split_df with concept labels on row_id, with the same hard
    row-alignment guarantees used in the oracle-concepts diagnostic:
    no duplicate row_id in the labels file, merge doesn't change row
    count, merge preserves split_df's row order exactly."""
    concept_df = pd.read_csv(concept_labels_path)

    dup_ids = concept_df["row_id"][concept_df["row_id"].duplicated()].unique()
    assert len(dup_ids) == 0, (
        f"{concept_labels_path} has duplicate row_id values: {list(dup_ids)[:5]} -- "
        f"a duplicate would silently inflate the merge and misalign features with "
        f"the wrong labels. Deduplicate before running this script."
    )

    merged = split_df[["row_id", "risk_level"]].merge(concept_df, on="row_id", how="left")
    assert len(merged) == len(split_df), (
        f"[{split_name}] merge produced {len(merged)} rows but split has "
        f"{len(split_df)} -- row alignment broken, aborting rather than risk "
        f"silently wrong features."
    )
    assert (merged["row_id"].values == split_df["row_id"].values).all(), (
        f"[{split_name}] merge did not preserve row order -- refusing to proceed "
        f"on an unverified alignment."
    )
    n_missing = merged[CONCEPTS[0]].isna().sum()
    if n_missing > 0:
        print(f"  [{split_name}] WARNING: {n_missing}/{len(merged)} rows have no "
             f"entry in {concept_labels_path} -- defaulting to all-False concepts "
             f"for those rows. This is a coverage gap, not a labeling signal.")
    for c in CONCEPTS:
        merged[c] = merged[c].fillna(False).astype(float)

    features = merged[CONCEPTS].values.astype(np.float32)  # (N, 9)
    labels = np.array([LABEL2ID[normalize_label(x)] for x in merged["risk_level"]])
    return features, labels


class ConceptOnlyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"features": self.features[i], "labels": self.labels[i]}


class ConceptOnlyMLP(nn.Module):
    """9 binary concepts -> risk level. Deliberately tiny -- the whole
    point is to measure what the concepts alone carry, not to add model
    capacity that could inflate the ceiling estimate beyond what the
    concept information actually supports."""
    def __init__(self, n_concepts=9, hidden=64, n_classes=4, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_concepts, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def per_class_f1(gold, pred):
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


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        x = batch["features"].to(device)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_labels.append(batch["labels"])
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    gold = [RISK_LEVELS[i] for i in labels.tolist()]
    pred = [RISK_LEVELS[i] for i in logits.argmax(-1).tolist()]
    f1 = risk_weighted_f1(gold, pred)
    return f1, per_class_f1(gold, pred), pred, gold


def lookup_table_baseline(train_features, train_labels, val_features, val_labels):
    """Deterministic, zero-training-variance ceiling estimate: majority
    risk label per unique 9-bit concept pattern seen in training,
    applied to val. Patterns in val never seen in training fall back to
    the global training majority class. This is the theoretically best
    achievable F1 given ONLY these 9 binary features and this training
    distribution -- useful as a sanity check against the MLP's number,
    since it has no optimizer, init, or early-stopping variance at all."""
    train_patterns = {}
    for feat, lab in zip(train_features, train_labels):
        key = tuple(int(v) for v in feat)
        train_patterns.setdefault(key, []).append(lab)
    pattern_majority = {k: max(set(v), key=v.count) for k, v in train_patterns.items()}
    global_majority = max(set(train_labels.tolist()), key=train_labels.tolist().count)

    n_unseen = 0
    preds = []
    for feat in val_features:
        key = tuple(int(v) for v in feat)
        if key in pattern_majority:
            preds.append(pattern_majority[key])
        else:
            preds.append(global_majority)
            n_unseen += 1

    gold = [RISK_LEVELS[i] for i in val_labels.tolist()]
    pred = [RISK_LEVELS[i] for i in preds]
    f1 = risk_weighted_f1(gold, pred)
    n_unique_train_patterns = len(pattern_majority)
    return f1, per_class_f1(gold, pred), n_unique_train_patterns, n_unseen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--train-concept-labels", required=True)
    ap.add_argument("--val-concept-labels", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200,
                    help="Max epochs -- early stopping will typically end far sooner "
                         "given the tiny input space.")
    ap.add_argument("--patience", type=int, default=15,
                    help="Stop if val weighted-F1 doesn't improve for this many epochs.")
    ap.add_argument("--class-weighting", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--bottleneck-run-config", default=None,
                    help="Path to a bottleneck model's run_config.json, to print an "
                         "explicit comparison against its best_val_weighted_f1. "
                         "Optional -- without it, only the raw ceiling numbers print.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 90)
    print("CONCEPT-ONLY MLP -- DIAGNOSTIC EXPERIMENT, NOT A DEPLOYABLE MODEL")
    print("Measures the ceiling of what the 9 PSEUDO-LABEL concepts alone explain")
    print("about risk level. See this file's module docstring for methodology")
    print("caveats before trusting the number below.")
    print("=" * 90)

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr_df = pd.read_csv(args.train)
    va_df = pd.read_csv(args.val)
    tr_features, tr_labels = load_concept_features(tr_df, args.train_concept_labels, "train")
    va_features, va_labels = load_concept_features(va_df, args.val_concept_labels, "val")

    n_unique_patterns = len(set(tuple(int(v) for v in f) for f in tr_features))
    print(f"\n[data] train n={len(tr_labels)}  val n={len(va_labels)}")
    print(f"[data] unique 9-bit concept patterns in train: {n_unique_patterns} "
         f"(of at most 512 possible)")

    weights = None
    if args.class_weighting:
        counts = pd.Series(tr_labels).value_counts()
        weights = [len(tr_labels) / (len(RISK_LEVELS) * counts.get(i, 1))
                   for i in range(len(RISK_LEVELS))]
        print(f"[class_weighting] weights = " +
             ", ".join(f"{RISK_LEVELS[i]}={w:.2f}" for i, w in enumerate(weights)))

    g = torch.Generator()
    g.manual_seed(args.seed)
    tr_loader = DataLoader(ConceptOnlyDataset(tr_features, tr_labels),
                           batch_size=args.batch, shuffle=True, generator=g)
    va_loader = DataLoader(ConceptOnlyDataset(va_features, va_labels),
                           batch_size=args.batch * 2, shuffle=False)

    model = ConceptOnlyMLP(hidden=args.hidden, dropout=args.dropout).to(device)
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device) if weights else None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1, best_state, best_epoch, epochs_no_improve = -1.0, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []
        for batch in tr_loader:
            x = batch["features"].to(device)
            y = batch["labels"].to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            running.append(loss.item())

        f1, _, _, _ = evaluate(model, va_loader, device)
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:>4}  train_loss {np.mean(running):.4f}  val_f1 {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[early stop] no improvement for {args.patience} epochs, "
                     f"stopping at epoch {epoch} (best was epoch {best_epoch})")
                break

    model.load_state_dict(best_state)
    model.to(device)
    mlp_f1, mlp_per_class, mlp_pred, mlp_gold = evaluate(model, va_loader, device)

    lut_f1, lut_per_class, n_unique_train, n_unseen_val = lookup_table_baseline(
        tr_features, tr_labels, va_features, va_labels)

    print("\n" + "=" * 90)
    print("RESULTS")
    print("=" * 90)
    print(f"\nBest validation weighted F1: {mlp_f1:.4f}   (MLP, epoch {best_epoch})")
    print(f"Lookup-table ceiling F1:     {lut_f1:.4f}   (deterministic, zero training variance)")
    print(f"  lookup table built from {n_unique_train} unique training patterns; "
         f"{n_unseen_val}/{len(va_labels)} val rows had a pattern unseen in training "
         f"(fell back to global majority class)")
    agreement_gap = abs(mlp_f1 - lut_f1)
    if agreement_gap > 0.03:
        print(f"\n  WARNING: MLP and lookup-table ceilings disagree by {agreement_gap:.4f} -- "
             f"treat the MLP number with caution; this gap is more likely a training-"
             f"variance artifact than a real difference in what's achievable from these "
             f"9 features. Consider it inconclusive until you re-run the MLP (different "
             f"seed / more epochs) and see whether it converges toward the lookup-table "
             f"number.")
    else:
        print(f"\n  MLP and lookup-table ceilings agree within {agreement_gap:.4f} -- "
             f"the MLP result is not an artifact of training variance; both estimates "
             f"point to the same ceiling.")

    print(f"\n{'class':<12}{'MLP F1':>10}{'lookup F1':>12}")
    for cls in RISK_LEVELS:
        print(f"{cls:<12}{mlp_per_class[cls]:>10.4f}{lut_per_class[cls]:>12.4f}")

    if args.bottleneck_run_config:
        with open(args.bottleneck_run_config) as f:
            bottleneck_cfg = json.load(f)
        bottleneck_f1 = bottleneck_cfg.get("best_val_weighted_f1")
        print(f"\n--- comparison to bottleneck model ({args.bottleneck_run_config}) ---")
        print(f"  bottleneck model risk F1:     {bottleneck_f1:.4f}")
        print(f"  concept-only ceiling (MLP):   {mlp_f1:.4f}")
        print(f"  gap:                          {mlp_f1 - bottleneck_f1:+.4f}")
        print(f"\n  NOTE: this comparison tells you whether the fusion architecture is")
        print(f"  near the concept-only ceiling -- it does NOT tell you how much text")
        print(f"  information the 9-concept taxonomy discards, since there is no")
        print(f"  text-only (no-concept) classifier F1 on record to compare against.")
        if mlp_f1 > bottleneck_f1 + 0.02:
            print(f"\n  --> the concept-only ceiling EXCEEDS the trained bottleneck's F1. "
                 f"Combined with the Oracle Concepts result (perfect concepts didn't move "
                 f"the trained fusion layer's F1), this means: the concepts carry more "
                 f"signal than the current architecture extracts, AND the current fusion "
                 f"mechanism is provably not extracting it. The fusion layer, not the "
                 f"concept predictor, is the clear next thing to fix.")
        elif mlp_f1 < bottleneck_f1 - 0.02:
            print(f"\n  --> the concept-only ceiling is BELOW the bottleneck's F1. This "
                 f"means the bottleneck model is doing better than concepts alone could "
                 f"ever achieve -- i.e. the encoder/fusion path is using information from "
                 f"the raw text that isn't captured in these 9 concepts at all. Expecting "
                 f"the concept branch alone to match or exceed the full model was never "
                 f"realistic; the concepts are a genuinely lossy summary of the text.")
        else:
            print(f"\n  --> concept-only ceiling and bottleneck F1 are close. The fusion "
                 f"architecture is already near what these concepts alone could support -- "
                 f"combined with the Oracle result, this suggests the concepts are simply "
                 f"not very informative for this task at the level this taxonomy captures "
                 f"them, rather than the fusion mechanism failing to use available signal.")
    else:
        print(f"\n  (pass --bottleneck-run-config <path> to compare directly against the "
             f"trained bottleneck model's F1)")

    va_out = va_df.copy()
    va_out["pred_risk_level"] = mlp_pred
    va_out.to_csv(os.path.join(args.outdir, "val_predictions.csv"), index=False)
    torch.save(best_state, os.path.join(args.outdir, "model_final.pt"))
    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump({
            **vars(args),
            "best_val_weighted_f1": float(mlp_f1),
            "best_epoch": best_epoch,
            "lookup_table_ceiling_f1": float(lut_f1),
            "n_unique_train_patterns": n_unique_train,
            "n_val_patterns_unseen_in_train": n_unseen_val,
            "note": "DIAGNOSTIC ONLY -- not a deployable model. Trained and evaluated "
                    "on DeepSeek pseudo-concept-labels, not human-annotated concepts.",
        }, f, indent=2)
    print(f"\nSaved to {args.outdir}/ (model_final.pt, val_predictions.csv, run_config.json)")


if __name__ == "__main__":
    main()
