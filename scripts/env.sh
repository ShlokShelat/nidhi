# Sourced by every job script in scripts/, from the repository root:
#
#     source scripts/env.sh [ENV]     # ENV defaults to riskcup2
#
# Activates the conda environment a job needs without assuming where conda is
# installed. Override with environment variables, e.g. at submission time:
#
#     NIDHI_CONDA_ENV=myenv sbatch scripts/risk_nli.sh
#     NIDHI_CONDA_ENV=none  sbatch scripts/risk_nli.sh   # use the python already on PATH (venv, module, ...)

if [ ! -f scripts/env.sh ]; then
  echo "env.sh: run jobs from the repository root (cd to it before sbatch/bash)" >&2
  exit 1
fi

_nidhi_env="${NIDHI_CONDA_ENV:-${1:-riskcup2}}"
if [ "$_nidhi_env" != "none" ]; then
  _nidhi_base=""
  if command -v conda >/dev/null 2>&1; then
    _nidhi_base="$(conda info --base 2>/dev/null)"
  elif [ -n "$CONDA_EXE" ]; then
    _nidhi_base="$(dirname "$(dirname "$CONDA_EXE")")"
  else
    for _d in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3" "$HOME/mambaforge" /opt/conda; do
      [ -f "$_d/etc/profile.d/conda.sh" ] && { _nidhi_base="$_d"; break; }
    done
  fi
  if [ -z "$_nidhi_base" ] || [ ! -f "$_nidhi_base/etc/profile.d/conda.sh" ]; then
    echo "env.sh: conda not found; put it on PATH, or set NIDHI_CONDA_ENV=none to use the current python" >&2
    exit 1
  fi
  . "$_nidhi_base/etc/profile.d/conda.sh"
  conda activate "$_nidhi_env" || {
    echo "env.sh: cannot activate conda env '$_nidhi_env' (create it, or set NIDHI_CONDA_ENV)" >&2
    exit 1
  }
fi
unset _nidhi_env _nidhi_base _d
