"""
Generate a competition submission CSV from the three trained checkpoints.

Usage (on a GPU node):
  python src/predict.py \
      --test data/test.csv \
      --risk-run runs/risk_v1 \
      --evidence-run runs/evidence_v1 \
      --factors-run runs/factors_v1 \
      --team YourTeamName

Reads each run's run_config.json to reconstruct the exact model (name,
max_len, flags) that produced the checkpoint, so a checkpoint can never be
silently paired with the wrong architecture or tokenizer settings.

Output: submissions/YourTeamName.csv, refused if validation fails.
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")          # no hub calls; cache only (override: HF_HUB_OFFLINE=0)
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_audit import load  # canonicalization + column aliases
from scoring import RISK_LEVELS, FACTOR_TAXONOMY
from train_risk import DualHeadModel, build_text
from train_factors import MultiLabelModel, N as N_FACTORS
from train_evidence import TaggerModel, predict_spans
from make_submission import build_submission, validate_submission
from transformers import AutoTokenizer


def read_cfg(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "run_config.json")) as f:
        return json.load(f)


TRAINING_ONLY_BUFFERS = {"pos_weight", "class_weights"}


def load_state(model: torch.nn.Module, run_dir: str, device: str):
    sd = torch.load(os.path.join(run_dir, "model_final.pt"),
                    map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # loss-only buffers may exist in the checkpoint but not the inference
    # model (or vice versa); anything else mismatching is a real error
    bad = ([k for k in missing if k not in TRAINING_ONLY_BUFFERS]
           + [k for k in unexpected if k not in TRAINING_ONLY_BUFFERS])
    if bad:
        raise RuntimeError(f"checkpoint/model mismatch beyond loss buffers: {bad}")
    return model.to(device).eval()


@torch.no_grad()
def _risk_probs_one(run_dir: str, df: pd.DataFrame, device: str) -> np.ndarray:
    cfg = read_cfg(run_dir)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = load_state(DualHeadModel(cfg["model"],
                                     alpha_ordinal=cfg.get("alpha_ordinal", 0.5)),
                       run_dir, device)
    d = df.copy()
    d["user_post_count"] = d.groupby("anon_user_id")["post"].transform("size")

    # THE INDEX MUST COME FROM post_id, NOT FILE ORDER.
    #
    # train_risk.build_text prepends "[USER_POST k/n] " to every post when a
    # run was trained with user_context, so `k` is part of the model's INPUT
    # TEXT. Taking it from `cumcount()` makes that text depend on the row order
    # of whatever CSV was handed in: the same posts in a different order then
    # produce different predictions for most rows. The evaluation harness runs
    # this code on unseen posts in an order this code does not control, and
    # risk is 40% of the composite score.
    #
    # `post_id` IS the true within-user index, so use it and fall back to
    # cumcount only when it is absent -- warning loudly, because that fallback
    # is order-dependent by construction.
    if "post_id" in d.columns:
        pid = pd.to_numeric(d["post_id"], errors="coerce")
        if pid.notna().all():
            base = pid.groupby(d["anon_user_id"]).transform("min")
            d["post_index_in_user"] = (pid - base + 1).astype(int)
        else:
            print("  WARNING: post_id present but not numeric; falling back "
                  "to row order for [USER_POST k/n] -- predictions become "
                  "order-dependent")
            d["post_index_in_user"] = d.groupby("anon_user_id").cumcount() + 1
    else:
        print("  WARNING: no post_id column; [USER_POST k/n] falls back to "
              "ROW ORDER and predictions depend on how the CSV is sorted")
        d["post_index_in_user"] = d.groupby("anon_user_id").cumcount() + 1

    # user_context is part of the trained input format, not a runtime choice:
    # a config missing the key must not silently flip the text format.
    if "user_context" not in cfg:
        raise SystemExit(
            f"{run_dir}/run_config.json has no `user_context` key. It decides "
            f"whether '[USER_POST k/n] ' is prepended to the post, i.e. the "
            f"model's input format. Defaulting it would silently score a "
            f"model on text it was not trained on.")
    use_ctx = cfg["user_context"]

    all_probs = []
    for _, row in d.iterrows():
        enc = tok(build_text(row, use_ctx), truncation=True,
                  max_length=cfg.get("max_len", 512), return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()
               if k in ("input_ids", "attention_mask")}
        logits = model(**enc)["logits"].float().cpu()
        all_probs.append(torch.softmax(logits, -1).numpy()[0])
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.stack(all_probs)


def predict_risk(run_dirs: str, df: pd.DataFrame, device: str,
                  weights: list[float] | None = None) -> tuple[list[str], np.ndarray]:
    """run_dirs: one path, or comma-separated paths for a probability-
    averaging ensemble (e.g. runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15).
    weights: optional list of per-run weights, in the same order as run_dirs,
    for a weighted average. None means equal weights."""
    dirs = [d.strip() for d in run_dirs.split(",") if d.strip()]
    all_probs = [_risk_probs_one(d, df, device) for d in dirs]
    if weights is not None:
        if len(weights) != len(dirs):
            raise ValueError(f"weights has {len(weights)} entries, expected {len(dirs)}")
        if any(w < 0 for w in weights):
            raise ValueError(f"weights must be non-negative, got {weights}")
        if sum(weights) == 0:
            raise ValueError("weights cannot all be zero")
        w = np.array(weights, dtype=float)
        w = w / w.sum()
        probs = np.average(all_probs, axis=0, weights=w)
        print(f"  (risk ensemble of {len(dirs)} runs) ensemble weights: {w.tolist()}")
    else:
        probs = np.mean(all_probs, axis=0)
        if len(dirs) > 1:
            print(f"  (risk ensemble of {len(dirs)} runs)")
    preds = [RISK_LEVELS[i] for i in probs.argmax(-1)]
    return preds, probs


@torch.no_grad()
def _factor_probs(run_dir: str, df: pd.DataFrame, device: str) -> np.ndarray:
    """Stage-1: raw 24 factor probabilities from the DeBERTa model."""
    cfg = read_cfg(run_dir)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = load_state(MultiLabelModel(cfg["model"]), run_dir, device)
    rows = []
    for post in df["post"].astype(str):
        enc = tok(post, truncation=True, max_length=cfg.get("max_len", 512),
                  return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()
               if k in ("input_ids", "attention_mask")}
        rows.append(torch.sigmoid(model(**enc)["logits"].float()).cpu().numpy()[0])
    return np.stack(rows)


def predict_factors(run_dir: str, df: pd.DataFrame, device: str,
                    stack_dir: str = None) -> list[list[str]]:
    """Factors with optional two-stage co-occurrence refinement.
    If stack_dir has stage-2 GBMs + gate, apply best-of-both; else stage-1."""
    probs = _factor_probs(run_dir, df, device)

    if stack_dir and os.path.exists(os.path.join(stack_dir, "stage2_models.pkl")):
        import pickle
        with open(os.path.join(stack_dir, "stage2_models.pkl"), "rb") as f:
            models = pickle.load(f)
        use_stage2 = np.load(os.path.join(stack_dir, "use_stage2_mask.npy"))
        gated_ths = np.load(os.path.join(stack_dir, "gated_thresholds.npy"))
        refined = probs.copy()
        for j in range(N_FACTORS):
            if use_stage2[j] and models[j]:
                refined[:, j] = models[j][0].predict_proba(probs)[:, 1]
        final = np.where(use_stage2[None, :], refined, probs)
        on = final >= gated_ths
        print(f"  (two-stage factors: stage-2 for {int(use_stage2.sum())}/{N_FACTORS} labels)")
    else:
        ths = np.load(os.path.join(run_dir, "thresholds.npy"))
        on = probs >= ths

    return [[FACTOR_TAXONOMY[j] for j in range(N_FACTORS) if on[i, j]]
            for i in range(len(probs))]


def predict_evidence(run_dir: str, df: pd.DataFrame, device: str) -> list[list[str]]:
    cfg = read_cfg(run_dir)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = load_state(TaggerModel(cfg["model"]), run_dir, device)
    return predict_spans(model, tok, df["post"].astype(str).tolist(),
                         cfg.get("max_len", 512), device,
                         min_conf=cfg.get("evidence_min_conf", 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--risk-run", required=True)
    ap.add_argument("--evidence-run", required=True)
    ap.add_argument("--factors-run", required=True)
    ap.add_argument("--factors-stack", default=None,
                    help="dir with stage-2 co-occurrence models (optional)")
    ap.add_argument("--risk-override", default=None,
                    help="CSV with row_id,pred_risk_level to use instead of computing risk")
    ap.add_argument("--factors-override", default=None,
                    help="CSV with row_id,factors (list-string) to use instead of computing factors")
    ap.add_argument("--evidence-override", default=None,
                    help="CSV with row_id,evidence (semicolon spans) to use instead of computing evidence")
    ap.add_argument("--team", required=True)
    ap.add_argument("--outdir", default="submissions")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: running inference on CPU; fine for 378 posts, "
              "but do not train here.")

    df = load(args.test)  # canonicalized, same text space as training

    print("risk level ...")
    risk_preds, risk_logits = predict_risk(args.risk_run, df, device)
    if args.risk_override:
        ov = pd.read_csv(args.risk_override).set_index("row_id")["pred_risk_level"]
        risk_preds = [str(ov.get(rid, risk_preds[i])).strip().lower()
                      for i, rid in enumerate(df["row_id"].tolist())]
        print(f"  (risk overridden from {args.risk_override})")
    print("evidence spans ...")
    if args.evidence_override:
        from scoring import parse_evidence
        ovv = pd.read_csv(args.evidence_override, keep_default_na=False).set_index("row_id")["evidence"]
        ev_preds = [parse_evidence(ovv.get(rid, "")) for rid in df["row_id"].tolist()]
        print(f"  (evidence overridden from {args.evidence_override})")
    else:
        ev_preds = predict_evidence(args.evidence_run, df, device)
    print("factors ...")
    if args.factors_override:
        from scoring import parse_factors
        ov = pd.read_csv(args.factors_override, keep_default_na=False).set_index("row_id")["factors"]
        fac_preds = [parse_factors(ov.get(rid, "[]")) for rid in df["row_id"].tolist()]
        print(f"  (factors overridden from {args.factors_override})")
    else:
        fac_preds = predict_factors(args.factors_run, df, device, args.factors_stack)

    # sanity summary before writing anything
    from collections import Counter
    print("\npredicted risk distribution:", dict(Counter(risk_preds)))
    n_ev = sum(len(e) for e in ev_preds)
    print(f"evidence spans: {n_ev} total, "
          f"{sum(1 for e in ev_preds if not e)} posts with none")
    print(f"factors per post: mean {np.mean([len(f) for f in fac_preds]):.1f}")
    # indicator posts should mostly have no explicit suicide evidence;
    # a large mismatch here means the two heads disagree systematically
    ind_with_ev = sum(1 for r, e in zip(risk_preds, ev_preds)
                      if r == "indicator" and e)
    print(f"posts predicted indicator BUT with evidence spans: {ind_with_ev}")

    path = build_submission(df, risk_preds, ev_preds, fac_preds,
                            args.team, args.outdir)
    problems = validate_submission(path, df)
    if problems:
        os.rename(path, path + ".INVALID")
        print("\nBLOCKED — submission failed validation:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    # keep per-class probabilities next to the CSV for later ensembling
    np.save(os.path.join(args.outdir, f"{args.team}_risk_logits.npy"), risk_logits)
    print(f"\nwrote {path} — validated, ready to upload")


if __name__ == "__main__":
    main()
