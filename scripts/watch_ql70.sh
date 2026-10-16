#!/bin/bash
# LOGIN-NODE watcher. Do NOT sbatch it.
#
#   nohup bash scripts/watch_ql70.sh > logs/watch_ql70.log 2>&1 &
#   tail -f logs/watch_ql70.log
#   stop:   pkill -f watch_ql70.sh
#
# WHY. Folds can only be submitted in the moments the job count drops below the
# 5-submitted cap, and those moments land at unpredictable hours. A slot that
# frees at 3am and is claimed at 9am costs six hours on the only lane that still
# decides anything. This reruns the (idempotent) submitters every 30 minutes.
#
# TWO LANES, IN PRIORITY ORDER. ql70 (Llama-3.1-70B) first, then q72
# (Qwen2.5-72B). Llama already has a fold-3 number (0.8658) so it is the lane we
# are betting on; Qwen is the second independent shot at the same bar.
#
# THE QWEN LANE IS GATED, AND THE GATE OPENS ITSELF. Its four folds cost ~48
# GPU-hours, so they must not be committed before its pilot returns a number.
# Once runs/risk_q72_f3/val_predictions.csv appears, this script scores fold 3
# itself and opens the gate only if the number justifies the spend:
#
#   Q72_BAR = 0.8600. Fold 3 flatters by ~0.049 (the 8B scored 0.8496 there
#   against a 0.8006 pooled OOF), and the pooled bar in pool_ql70.sh is 0.8150.
#   So a lane needs roughly 0.864 on fold 3 to project over that bar; 0.8600
#   leaves a little room for noise. Below it, Qwen's four folds project UNDER
#   the bar and are 48 GPU-hours spent on a member we could not ship.
#
# Override either way by hand:
#   touch artifacts/q72_go          # force the lane open
#   rm    artifacts/q72_go          # force it shut (also kill queued q72f* jobs)
#
# Costs one squeue call, a few stats, and one short python per half hour.
# Exits by itself when every ENABLED lane has all five folds.

set -u
cd "$(dirname "$0")/.." || exit 1

INTERVAL="${INTERVAL:-1800}"
LANES="${LANES:-ql70 q72}"
Q72_BAR="${Q72_BAR:-0.8600}"

model_for() {
    case "$1" in
        ql70) echo "meta-llama/Llama-3.1-70B-Instruct" ;;
        q72)  echo "Qwen/Qwen2.5-72B-Instruct" ;;
        *)    echo "" ;;
    esac
}

gate_for() {
    case "$1" in
        q72) echo "artifacts/q72_go" ;;
        *)   echo "" ;;
    esac
}

# Score a lane's fold-3 pilot from its saved predictions and open the gate if it
# clears the bar. Scoring from val_predictions.csv rather than grepping the log
# means this does not care what the job was named or whether the log was rotated.
try_open_gate() {
    local tag="$1" gate="$2"
    local pred="runs/risk_${tag}_f3/val_predictions.csv"
    [ -f "$pred" ] || { echo "[${tag}] gated -- fold-3 pilot still running"; return; }

    TAG="$tag" BAR="$Q72_BAR" GATE="$gate" python - <<'PY'
import os, sys
import pandas as pd
from sklearn.metrics import f1_score

tag, bar, gate = os.environ["TAG"], float(os.environ["BAR"]), os.environ["GATE"]
pred = pd.read_csv(f"runs/risk_{tag}_f3/val_predictions.csv")
gold = pd.read_csv("artifacts/folds/val_f3.csv")

pc = next((c for c in ("pred_risk_level", "pred", "risk_level") if c in pred.columns), None)
gc = next((c for c in ("risk_level", "label", "gold") if c in gold.columns), None)
if pc is None or gc is None:
    print(f"[{tag}] gate check FAILED: pred cols {list(pred.columns)} "
          f"gold cols {list(gold.columns)}")
    sys.exit(0)

pred["row_id"] = pred["row_id"].astype(str)
gold["row_id"] = gold["row_id"].astype(str)
m = gold[["row_id", gc]].merge(pred[["row_id", pc]], on="row_id")
if len(m) == 0:
    print(f"[{tag}] gate check FAILED: no row_id overlap")
    sys.exit(0)

norm = lambda s: s.astype(str).str.strip().str.lower()
w = f1_score(norm(m[gc]), norm(m[pc]), average="weighted")
print(f"[{tag}] fold-3 pilot: val weighted-F1 = {w:.4f} on {len(m)} posts "
      f"(bar {bar:.4f}, Llama-70B got 0.8658, 8B got 0.8496)")

if w >= bar:
    open(gate, "w").write(f"fold3={w:.4f} bar={bar:.4f}\n")
    print(f"[{tag}] GATE OPENED -- projects to ~{w - 0.049:.4f} pooled, over the "
          f"0.8150 bar. Submitting its four folds.")
else:
    print(f"[{tag}] gate STAYS SHUT -- projects to ~{w - 0.049:.4f} pooled, under "
          f"the 0.8150 bar. Not worth 48 GPU-hours.")
    print(f"[{tag}] override with: touch {gate}")
PY
}

while true; do
    echo ""
    echo "[$(date)] ------------------------------------------------"
    ALLDONE=1

    for TAG in $LANES; do
        MODEL="$(model_for "$TAG")"
        if [ -z "$MODEL" ]; then
            echo "[${TAG}] no model path known for this tag -- skipping"
            continue
        fi

        GATE="$(gate_for "$TAG")"
        if [ -n "$GATE" ] && [ ! -f "$GATE" ]; then
            try_open_gate "$TAG" "$GATE"
            [ -f "$GATE" ] || { ALLDONE=0; continue; }
        fi

        COMPLETE=1
        for f in 0 1 2 3 4; do
            [ -f "runs/risk_${TAG}_f${f}/val_predictions.csv" ] || COMPLETE=0
        done
        if [ "$COMPLETE" -eq 1 ]; then
            echo "[${TAG}] all five folds trained -- run: TAG=${TAG} bash scripts/pool_ql70.sh"
            continue
        fi

        ALLDONE=0
        TAG="$TAG" MODEL="$MODEL" bash scripts/submit_ql70_folds.sh || true
    done

    if [ "$ALLDONE" -eq 1 ]; then
        echo ""
        echo "[$(date)] every enabled lane is complete -- watcher stopping."
        break
    fi

    sleep "$INTERVAL"
done
