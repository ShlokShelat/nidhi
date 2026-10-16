#!/bin/bash
# Preflight + run. Refuses to start in any context that has burned us:
# login node, CPU-only torch, wrong env. Usage:
#   bash scripts/run_gpu.sh python src/train_risk.py --train ... --val ... --outdir ...
set -e

host=$(hostname)
case "$host" in
  mn*) echo "REFUSED: you are on login node '$host'. Get a GPU shell first:"
       echo "  srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=2:00:00 --pty bash"
       exit 1;;
esac

# Expected env: riskcup, or whatever NIDHI_CONDA_ENV names (none = skip check).
want="${NIDHI_CONDA_ENV:-riskcup}"
if [ "$want" != "none" ] && [ "$CONDA_DEFAULT_ENV" != "$want" ]; then
  echo "REFUSED: conda env is '$CONDA_DEFAULT_ENV', expected '$want'."
  echo "  conda activate $want     (or set NIDHI_CONDA_ENV)"
  exit 1
fi

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "REFUSED: torch.cuda.is_available() is False in this env on $host."
  echo "  (wrong torch build or not a GPU node)"
  exit 1
fi

export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
echo "[preflight ok] host=$host env=$CONDA_DEFAULT_ENV gpu=$(python -c 'import torch;print(torch.cuda.get_device_name(0))')"
exec "$@"
