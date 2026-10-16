# Team NIDHI multi-task pipeline: one shared encoder with risk, evidence and
# factor heads. Vendored into this repository so the `friendfac` factor arm can
# be rebuilt from source without the original project tree.
"""Stage 4 - ensemble the 5 fold-models on the leaderboard set, apply the
symbolic guards + tuned thresholds, and write the official submission CSV.

Catches handled here:
  * ensemble = average of probabilities across all fold models (voting)
  * evidence emitted verbatim, short, split into separate spans
  * risk_level written with exact official capitalization
  * factors written as a python-style list string, matching the train format
  * output file named <TeamName>.csv as required
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

import config as C
import postprocess as PP
from model import TriHeadModel
from stage2_train import predict


def load_fold_model(fold, hw):
    from transformers import AutoModel, AutoTokenizer
    ck = torch.load(os.path.join(C.OUT_DIR, f"model_fold{fold}.pt"),
                    map_location=hw["device"])
    tok = AutoTokenizer.from_pretrained(ck["tok"])
    enc = AutoModel.from_pretrained(ck["tok"])
    model = TriHeadModel(enc, ck["hidden"]).to(hw["device"])
    model.load_state_dict(ck["state"])
    model.eval()
    return model, tok


def run(team_name):
    hw = C.hardware_profile()
    lb = pd.read_csv(os.path.join(C.OUT_DIR, "leaderboard_clean.csv"))
    thr = json.load(open(os.path.join(C.OUT_DIR, "thresholds.json")))
    lex = PP.load_lexicon()

    risk_acc, fact_acc, tok_acc = None, None, None
    offs_ref, posts = None, None
    n_models = 0
    for fold in range(C.N_FOLDS):
        path = os.path.join(C.OUT_DIR, f"model_fold{fold}.pt")
        if not os.path.exists(path):
            print(f"[stage4] missing {path}, skipping fold {fold}")
            continue
        model, tok = load_fold_model(fold, hw)
        o = predict(model, tok, lb, hw)
        risk_acc = o["risk"] if risk_acc is None else risk_acc + o["risk"]
        fact_acc = o["fact"] if fact_acc is None else fact_acc + o["fact"]
        if tok_acc is None:
            tok_acc = [np.array(t, dtype=np.float64) for t in o["tokp"]]
            offs_ref, posts = o["offsets"], o["post"]
        else:
            for i, t in enumerate(o["tokp"]):
                tok_acc[i] += t
        n_models += 1
        del model
        torch.cuda.empty_cache()
    assert n_models > 0, "no trained fold models found - run stage 2 first"
    risk_p = risk_acc / n_models
    fact_p = fact_acc / n_models
    tok_p = [t / n_models for t in tok_acc]
    print(f"[stage4] ensembled {n_models} fold models")

    # dump per-post risk probabilities for cross-team ensembling
    prob_df = pd.DataFrame(risk_p, columns=C.RISK_LEVELS)
    prob_df.insert(0, "row_id", lb["row_id"].values)
    prob_df.to_csv(os.path.join(C.OUT_DIR, "risk_probs.csv"), index=False)
    print(f"[stage4] wrote risk_probs.csv")

    # dump per-post factor probabilities for cross-team ensembling
    fact_df = pd.DataFrame(fact_p, columns=C.FACTORS)
    fact_df.insert(0, "row_id", lb["row_id"].values)
    fact_df.to_csv(os.path.join(C.OUT_DIR, "factor_probs.csv"), index=False)
    print(f"[stage4] wrote factor_probs.csv")

    rows = []
    for i in range(len(lb)):
        post = posts[i]
        level = C.RISK_LEVELS[int(risk_p[i].argmax())]
        spans = PP.decode_spans(post, offs_ref[i], tok_p[i], thr["evid_threshold"])
        level, spans = PP.consistency_rules(post, level, spans, lex)
        facs = [f for k, f in enumerate(C.FACTORS)
                if fact_p[i, k] >= thr["factor_thresholds"][f]]
        facs = PP.apply_factor_rules(post, facs)
        rows.append({
            "row_id": lb["row_id"].iloc[i],
            "risk_level": level,
            "evidence": "; ".join(spans) if spans else "none",
            "factors": json.dumps(facs).replace('"', "'"),
        })

    out = os.path.join(C.OUT_DIR, f"{team_name}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[stage4] wrote submission: {out}")
    print(pd.DataFrame(rows)["risk_level"].value_counts())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--team_name", required=True)
    run(ap.parse_args().team_name)
