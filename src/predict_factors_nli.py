"""
Predict factors on the test set with the trained definition-grounded NLI
model. Produces a CSV (row_id, factors) consumable by
predict.py --factors-override.
Usage:
  python src/predict_factors_nli.py \
      --test data/test.csv \
      --nli-run runs/factors_nli \
      --out artifacts/nli_factors.csv
"""
from __future__ import annotations
import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("PYTHONUNBUFFERED", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(__file__))
from data_audit import load
from scoring import FACTOR_TAXONOMY
from train_factors_nli import NLIScorer, score_all, N
from transformers import AutoTokenizer
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--nli-run", required=True,
                    help="one run dir, OR comma-separated dirs to seed-ensemble (avg probs)")
    ap.add_argument("--out", default="artifacts/nli_factors.csv")
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--thresholds", default=None,
                    help="optional .npy threshold array to use INSTEAD of "
                         "each run's own saved thresholds.npy; omit it to "
                         "decode with the thresholds the runs were tuned with")
    ap.add_argument("--dump-probs", default=None,
                    help="also save the seed-averaged per-label probability "
                         "matrix (.npz with probs/row_id/thresholds) so "
                         "threshold analysis can run on CPU without "
                         "re-scoring the models")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    run_dirs = [r.strip() for r in args.nli_run.split(",")]
    df = load(args.test)
    all_probs = []
    ths_list = []
    _hyp_checked = 0
    for rd in run_dirs:
        # TRAIN/SERVE SKEW GUARD. The NLI hypothesis is assembled from three
        # env vars (FACTOR_DEFS_JSON, FACTOR_ASPECTS_JSON, PROTECTIVE_TEMPLATE)
        # in BOTH the training and the scoring path. If they differ here from
        # what the run was trained with, the model is asked a different
        # question than the one it learned to answer. Nothing downstream can
        # detect that: the probabilities stay in range and the decoded factors
        # stay plausible, so the dump is wrong in a way no later check sees.
        # An unset env var in the caller is enough to cause it, which is why
        # the trained hypotheses are recorded per run and compared here.
        _hp = os.path.join(rd, "hypotheses.json")
        if os.path.exists(_hp):
            import json as _json
            from train_factors_nli import hypotheses_for as _hf
            want = _json.load(open(_hp))
            now = {t: _hf(t) for t in want}
            bad = [t for t in want if list(want[t]) != list(now[t])]
            if bad:
                raise SystemExit(
                    f"REFUSING to score {rd}: the hypotheses in this "
                    f"environment differ from the ones it was TRAINED on, for "
                    f"{len(bad)} factor(s): {bad[:4]}...\n"
                    f"  trained: {want[bad[0]]}\n"
                    f"  now:     {now[bad[0]]}\n"
                    f"Set FACTOR_DEFS_JSON / FACTOR_ASPECTS_JSON / "
                    f"PROTECTIVE_TEMPLATE to match {rd}/hypotheses.json, or "
                    f"the dump is train/serve skew.")
            _hyp_checked += 1
        model_name = open(os.path.join(rd, "tok")).read().strip()
        tok = AutoTokenizer.from_pretrained(model_name)
        model = NLIScorer(model_name).to(device)
        state = torch.load(os.path.join(rd, "model.pt"), map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()
        all_probs.append(score_all(model, df, tok, args.max_len, device, use_bf16))
        ths_list.append(np.load(os.path.join(rd, "thresholds.npy")))
        print(f"  scored {rd}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    if _hyp_checked:
        print(f"  hypothesis check: {_hyp_checked}/{len(run_dirs)} runs "
              f"verified against their trained hypotheses")
    else:
        print(f"  hypothesis check: SKIPPED -- none of these {len(run_dirs)} "
              f"runs has hypotheses.json (trained before the guard existed). "
              f"If this arm uses aspects, defs or PROTECTIVE_TEMPLATE, confirm "
              f"the env matches its training by hand.")
    probs = np.mean(all_probs, axis=0)
    if args.thresholds:
        ths = np.load(args.thresholds)
        print(f"using threshold override: {args.thresholds}")
    else:
        # average thresholds across seeds (they were each tuned on same val)
        ths = np.mean(ths_list, axis=0)
    if len(run_dirs) > 1:
        print(f"seed-ensembled {len(run_dirs)} NLI models (avg probs + avg thresholds)")
    if args.dump_probs:
        os.makedirs(os.path.dirname(args.dump_probs) or ".", exist_ok=True)
        np.savez(args.dump_probs, probs=probs, thresholds=ths,
                 row_id=np.array(df["row_id"].tolist(), dtype=object))
        print(f"dumped probabilities -> {args.dump_probs}  {probs.shape}")
    on = (probs >= ths).astype(int)
    fac_lists = [[FACTOR_TAXONOMY[j] for j in range(N) if on[i, j]]
                 for i in range(len(df))]
    out = pd.DataFrame({
        "row_id": df["row_id"].tolist(),
        "factors": [str(list(f)) for f in fac_lists],
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}: mean {on.sum(1).mean():.2f} factors/post")
    print(f"posts with 0 factors: {(on.sum(1)==0).sum()}/{len(df)}")
if __name__ == "__main__":
    main()
