#!/bin/bash
#SBATCH --job-name=evdom
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# ---------------------------------------------------------------------------
# DOMAIN-PRETRAINED EVIDENCE TAGGERS -- the one architecture family never
# tried for evidence, and the family that just broke the factor plateau
# (mental/mental-roberta-base, 125M) on a factor pool that nine
# DeBERTa-v3-large arms had closed in both directions.
#
# The only evidence member that ever helped the RC7 vote, s33_rc
# (roberta-large-mnli), is also the only one with a different pretraining
# objective. Same-family additions s31_rc/s32_rc DILUTED it.
# This job trains up to three taggers whose PRETRAINING CORPUS is the
# domain itself (Reddit mental-health text), not just a different objective:
#
#   s34_rc  mental/mental-roberta-base       125M  RoBERTa,   REQUIRED
#   s35_rc  mental/mental-bert-base-uncased  110M  BERT,      REQUIRED
#   s36_rc  AIMH/mental-roberta-large        355M  RoBERTa-L, OPTIONAL
#
# GATING / OFFLINE AVAILABILITY (verified via the HF API; compute nodes run
# HF_HUB_OFFLINE=1, so only the local cache matters at runtime):
#   all three repos are gated:"auto" (accept terms with an HF account ->
#   immediate access; no manual approval).
#   s35: CACHED on this cluster -- snapshot 24809aa822c76639760d0d934742d1b4
#        2f89942f, already used by scripts/factors_nli_mbert.sh and
#        scripts/risk_nli_mbert.sh, and it passes src/check_encoders.py.
#   s34: may not be fully cached on the training cluster. The repo is gated on
#        the Hub, so the download needs an account that has accepted its terms.
#        Where a complete snapshot (pytorch_model.bin ~499MB + tokenizer)
#        already exists in a local HF cache elsewhere, copying that directory
#        across avoids the token entirely. The preflight below fails LOUDLY
#        with instructions when the cache here still lacks it.
#   s36: gated and NOT cached (the 403 + 0.0GB dir is documented). Optional:
#        skipped with a loud banner and download instructions when absent.
#
# NO SILENT FALLBACK, EVER. train_evidence.py has no fallback logic and this
# script resolves each backbone with a preflight (config+tokenizer+AutoModel,
# offline, same checks as src/check_encoders.py). A model that cannot load
# is reported with its fix command; it is never substituted.
#
# PROTOCOL -- identical to the shipped RC7 and the s31-33 diverse taggers,
# so the LOO read is honest:
#   * SINGLE SPLIT: train on artifacts/train_split.csv, validate on
#     artifacts/val_split.csv. NO FOLDS. The matched val/test construction
#     (both dumped from ONE checkpoint in ONE pass) is why evidence produced
#     the only submission that won this month.
#   * --risk-cond with --val-risk-from artifacts/nligaur/oof_base.csv
#     (out-of-fold PREDICTED risk), exactly like evidence_diverse.sh.
#   * dump val AND test TOGETHER via src/evidence_dump_all.py with
#     --risk-cond --test-risk --val-risk. Omitting those flags silently
#     disables conditioning -- the exact mistake that once collapsed three
#     GPU-trained taggers' candidate pools from ~800 to ~240 and made them
#     look useless. Here the three flags are HARDCODED next to each other
#     and a spans/post sanity check runs after every dump.
#   * TEST conditioning uses submissions/NIDHI_cluster.csv -- NOT the
#     current champion (they differ on 5 risk labels) -- because that is
#     what the stored RC7 test candidates were conditioned on. Matching it
#     keeps a new member directly comparable to the incumbents. (s31-33
#     used NIDHI_ev7.csv on test, which is why their test side needed a
#     separate rebuild; this job avoids that.)
#   * LR: 2e-5 for the base-size models (the train_evidence.py default of
#     1e-5 was tuned for deberta-v3-large; the friend's mental-roberta-base
#     -- the model this lane copies -- fine-tuned at 2e-5). s36 keeps 1e-5.
#   * max_len 512: RoBERTa/BERT absolute position embeddings cap there.
#     Effective batch 16 = the RC7/diverse protocol.
#
# JUDGE (runs at the end of this job, CPU): src/evidence_member_loo.py on
# RC7 + the new member. A new member HELPS when the "drop <member>" row shows
# d < 0 with LOSS >= 90% (the 90% win bar, read from the drop side).
#
# PRE-REGISTERED FALSIFIER. Evidence carries 0.3 composite weight, so a
# meaningful composite gain requires a much larger VAL LOO delta once the
# documented val->test attenuation is applied. A same-family, single-objective
# addition like s33_rc does NOT clear that translated bar.
#
# Usage:
#   sbatch scripts/evidence_domain_backbones.sh
# Resumable: rerunning skips finished training and finished dumps.
# ---------------------------------------------------------------------------

set -u
source scripts/env.sh
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

TR=artifacts/train_split.csv
VA=artifacts/val_split.csv
VALRISK=artifacts/nligaur/oof_base.csv          # OOF PREDICTED risk for val
TESTRISK=submissions/NIDHI_cluster.csv          # matches the stored RC7 dumps
DUMPDIR=artifacts/evidence_runs                 # where RC7 + s31-33 val dumps live
RC7=evidence_ep9_rc,evidence_lr2_rc,evidence_s14_rc,evidence_s15_rc,evidence_s16_rc,evidence_s17_rc,evidence_v1_rc

for f in "$TR" "$VA" "$VALRISK" "$TESTRISK" data/test.csv; do
    [ -f "$f" ] || { echo "FATAL: required file $f is missing"; exit 1; }
done
mkdir -p logs "$DUMPDIR"

echo "=== environment ==="
nvidia-smi -L
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"

echo ""
echo "=== risk-conditioning alignment guard ==="
python tests/test_risk_cond_alignment.py || {
    echo "ALIGNMENT TEST FAILED -- not training anything"; exit 1; }

# ---------------------------------------------------------------------------
# resolve_model REPO -> prints a loadable model path/id on stdout, or fails.
# Tries the repo id offline first, then every complete local snapshot dir.
# Falling back to a SNAPSHOT PATH OF THE SAME REPO is fine; a different
# model never is.
# ---------------------------------------------------------------------------
resolve_model() {
    python - "$1" <<'PY'
import glob, os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
repo = sys.argv[1]

def try_load(target):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    AutoConfig.from_pretrained(target)
    tok = AutoTokenizer.from_pretrained(target)
    assert tok.is_fast, "need a fast tokenizer for offset mapping"
    m = AutoModel.from_pretrained(target)
    return sum(p.numel() for p in m.parameters())

cands = [repo]
hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
    os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"),
    "hub")
snapdir = os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots")
for s in sorted(glob.glob(os.path.join(snapdir, "*"))):
    if os.path.isfile(os.path.join(s, "config.json")):
        cands.append(s)
for c in cands:
    try:
        n = try_load(c)
    except Exception as e:
        msg = str(e).splitlines()[0][:110] if str(e) else ""
        print("  [preflight] cannot load %s: %s: %s"
              % (c, type(e).__name__, msg), file=sys.stderr)
        continue
    print("  [preflight] OK %s via %s (%.0fM params)" % (repo, c, n / 1e6),
          file=sys.stderr)
    print(c)
    sys.exit(0)
sys.exit(1)
PY
}

print_gate_help() {
    local repo="$1"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!! MODEL NOT LOADABLE OFFLINE: $repo"
    echo "!! It is a GATED repo (gated:auto) and the compute nodes are"
    echo "!! offline. NOTHING was substituted for it. To make it available:"
    if [ "$repo" = "mental/mental-roberta-base" ]; then
        echo "!!   Fastest (no HF token): copy a COMPLETE snapshot from any"
        echo "!!   machine whose HF cache already holds it. cd into the cache"
        echo "!!   dir first -- a drive-letter path breaks scp:"
        echo "!!     cd <local HF cache>/hub"
        echo "!!     scp -r models--mental--mental-roberta-base <user>@<host>:~/.cache/huggingface/hub/"
        echo "!!   Or on the LOGIN node with an HF account that accepted the"
        echo "!!   terms at https://huggingface.co/$repo :"
    else
        echo "!!   On the LOGIN node (compute nodes have no network), with an"
        echo "!!   HF account that accepted the terms at"
        echo "!!   https://huggingface.co/$repo :"
    fi
    echo "!!     HF_HUB_OFFLINE=0 huggingface-cli download $repo --token hf_YOUR_TOKEN"
    echo "!!   (or: HF_HUB_OFFLINE=0 python -c \"from huggingface_hub import snapshot_download; snapshot_download('$repo', token='hf_YOUR_TOKEN')\")"
    echo "!!   Then resubmit this job -- finished parts are skipped."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
}

# ---------------------------------------------------------------------------
# run_entry TAG REPO MAXLEN BATCH ACCUM LR
# returns nonzero on any failure; never falls back to another model.
# ---------------------------------------------------------------------------
run_entry() {
    local tag="$1" repo="$2" maxlen="$3" batch="$4" accum="$5" lr="$6"
    local key="${repo##*/}"
    local outdir="runs/evidence_${tag}"
    local valc="${DUMPDIR}/evidence_${tag}_val_candidates.csv"
    local tstc="${DUMPDIR}/evidence_${tag}_test_candidates.csv"

    echo ""
    echo "=============================================================="
    echo "ENTRY ${tag}  <-  ${repo}"
    echo "  max_len ${maxlen}   batch ${batch} x accum ${accum}   lr ${lr}   seed 31"
    echo "=============================================================="
    df -h .

    local resolved
    resolved="$(resolve_model "$repo")" || { print_gate_help "$repo"; return 1; }
    echo "  resolved model: ${resolved}"

    # ---- TAG-collision guards -------------------------------------------
    if [ -f "${outdir}/run_config.json" ]; then
        python - "${outdir}/run_config.json" "$key" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
rec = str(cfg.get("model", ""))
if sys.argv[2] not in rec:
    print("!! TAG COLLISION: %s already exists and was trained on"
          % sys.argv[1])
    print("!!   %r, which is not %r." % (rec, sys.argv[2]))
    print("!! Refusing to touch it. Pick a fresh TAG or delete the run dir.")
    sys.exit(1)
print("  existing run dir matches this backbone -- resumable")
PY
        [ $? -eq 0 ] || return 1
    elif [ -f "$valc" ] || [ -f "$tstc" ]; then
        echo "!! TAG COLLISION: no ${outdir}, but candidate dumps named"
        echo "!! evidence_${tag}_* already exist in ${DUMPDIR}. They belong to"
        echo "!! some other lane. Refusing to overwrite them."
        return 1
    fi

    # ---- train (skip when the checkpoint already exists) ----------------
    if [ -f "${outdir}/model_final.pt" ] || [ -f "${outdir}/model.pt" ]; then
        echo "  checkpoint already in ${outdir} -- skipping training"
    else
        python src/train_evidence.py \
            --train "$TR" --val "$VA" \
            --model "$resolved" --outdir "$outdir" \
            --risk-cond --val-risk-from "$VALRISK" \
            --max-len "$maxlen" --batch "$batch" --grad-accum "$accum" \
            --lr "$lr" --seed 31 --epochs 6 \
            || { echo "!! TRAINING FAILED for ${tag}"; return 1; }
    fi

    # ---- dump val + test TOGETHER, from the ONE checkpoint --------------
    # --risk-cond --test-risk --val-risk are deliberately hardcoded as a
    # unit: dropping them silently disables conditioning (the ~800 -> ~240
    # candidate collapse). --test-risk is NIDHI_cluster.csv to match the
    # stored RC7 test candidates (NOT the current champion; 5 labels apart).
    if [ -f "$valc" ] && [ -f "$tstc" ]; then
        echo "  candidate dumps already exist -- skipping dump"
    else
        python src/evidence_dump_all.py --runs "evidence_${tag}" \
            --test data/test.csv --val "$VA" \
            --outdir "$DUMPDIR" \
            --risk-cond --test-risk "$TESTRISK" --val-risk "$VALRISK" \
            || { echo "!! DUMP FAILED for ${tag}"; return 1; }
    fi

    # ---- conditioning sanity: kept spans/post on BOTH splits -------------
    # Conditioned RC members sit at ~1.1-1.2 kept spans/post; the historic
    # unconditioned-dump bug collapsed that to 0.33-0.48. Anything under 0.7
    # is treated as that bug (or a broken tagger) and FAILS the entry.
    python - "$DUMPDIR" "evidence_${tag}" <<'PY'
import os, sys
import pandas as pd
d, run = sys.argv[1], sys.argv[2]
bad = False
for split in ("val", "test"):
    p = os.path.join(d, "%s_%s_pred.csv" % (run, split))
    if not os.path.exists(p):
        print("!! sanity: missing %s" % p)
        bad = True
        continue
    df = pd.read_csv(p, keep_default_na=False)
    n_sp = sum(len([s for s in str(v).split(";") if s.strip()])
               for v in df["pred_evidence"])
    per = n_sp / max(len(df), 1)
    cp = os.path.join(d, "%s_%s_candidates.csv" % (run, split))
    nc = len(pd.read_csv(cp)) if os.path.exists(cp) else -1
    print("  sanity %s %s: %d raw candidates, %.2f kept spans/post"
          % (run, split, nc, per))
    if per < 0.7:
        print("!! %.2f kept spans/post on %s is the UNCONDITIONED-DUMP"
              " signature (healthy conditioned members: ~1.1-1.2;"
              " the historic bug: 0.33-0.48). Delete the dumps for this"
              " tag and investigate before trusting any LOO read." % (per, split))
        bad = True
sys.exit(1 if bad else 0)
PY
    [ $? -eq 0 ] || return 1

    echo "  ENTRY ${tag} COMPLETE"
    return 0
}

# ---------------------------------------------------------------------------
# the three entries. s34/s35 are REQUIRED; s36 is OPTIONAL (gated + uncached
# -- it runs only when someone has staged it, and is loudly skipped otherwise).
# ---------------------------------------------------------------------------
FAILED=""
OKTAGS=""

if run_entry s34_rc mental/mental-roberta-base 512 16 1 2e-5; then
    OKTAGS="${OKTAGS}${OKTAGS:+,}evidence_s34_rc"
else
    FAILED="${FAILED} s34_rc"
fi

if run_entry s35_rc mental/mental-bert-base-uncased 512 16 1 2e-5; then
    OKTAGS="${OKTAGS}${OKTAGS:+,}evidence_s35_rc"
else
    FAILED="${FAILED} s35_rc"
fi

if run_entry s36_rc AIMH/mental-roberta-large 512 8 2 1e-5; then
    OKTAGS="${OKTAGS}${OKTAGS:+,}evidence_s36_rc"
else
    echo ""
    echo ">> s36_rc (AIMH/mental-roberta-large) NOT trained -- OPTIONAL entry."
    echo ">> The job is not failed by this; see the instructions above to"
    echo ">> stage it, then resubmit (s34/s35 will be skipped as finished)."
fi

echo ""
df -h .

# ---------------------------------------------------------------------------
# JUDGE -- LOO on the val twin, one added member at a time against the
# shipped RC7. Read the "drop evidence_s3X_rc" row: the member HELPS when
# d < 0, and it clears the bar when LOSS >= 90 percent AND the implied val
# delta, once translated through the composite weight and attenuation
# factor, clears the required composite gain.
# ---------------------------------------------------------------------------
if [ -n "$OKTAGS" ]; then
    echo ""
    echo "=============================================================="
    echo "LOO judging on the val twin (CPU)"
    echo "=============================================================="
    for t in $(echo "$OKTAGS" | tr ',' ' '); do
        echo ""
        echo "----- RC7 + ${t}  (k=2 and the proportional k) -----"
        python src/evidence_member_loo.py --runs "${RC7},${t}" --k 2 \
            || echo "!! LOO failed for ${t}"
    done
    # all survivors together, plus s33_rc when its dump exists -- the k sweep
    # in section 3 shows whether a larger pool wants k=3 (k is a FRACTION).
    COMBO="${RC7},${OKTAGS}"
    if [ -f "${DUMPDIR}/evidence_s33_rc_val_candidates.csv" ]; then
        COMBO="${COMBO},evidence_s33_rc"
    fi
    echo ""
    echo "----- combined pool: ${COMBO} -----"
    python src/evidence_member_loo.py --runs "$COMBO" --k 2 \
        || echo "!! combined LOO failed"
fi

echo ""
echo "=============================================================="
echo "SUMMARY"
echo "=============================================================="
echo "trained + dumped OK: ${OKTAGS:-none}"
if [ -n "$FAILED" ]; then
    echo "REQUIRED ENTRIES FAILED:${FAILED} -- scroll up for the exact fix"
    echo "commands. Nothing was silently substituted."
    exit 1
fi
echo ""
echo "Decision rule (pre-registered):"
echo "  A meaningful composite gain requires a VAL LOO delta -- translated"
echo "  through the composite weight and the val->test attenuation factor --"
echo "  that clears the required threshold, at >= 90 percent win."
echo "  Members that help but miss that bar are still worth stacking with"
echo "  s33_rc via the combined-pool k sweep above before shipping."
echo "Ship path (unchanged from evidence_cond_deploy.sh stage 3):"
echo "  src/evidence_vote_test.py --runs <winning pool> --dir ${DUMPDIR} \\"
echo "    --k <winning k> --cluster --rep modal --risk-from <champion> \\"
echo "    --test data/test.csv, then merge into the champion columns."
