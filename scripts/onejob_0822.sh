#!/bin/bash
#SBATCH --job-name=onejob
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out

# ===========================================================================
# ONE JOB, THREE PARTS, ORDERED CHEAPEST-AND-SUREST FIRST.
#
# Only one submit slot is free given the current queue occupancy (QOS cap is
# 5 submitted / 2 running), so everything worth GPU time this round has to
# fit in a single script. It is ordered so that a job killed at any
# point still leaves the certain value on disk:
#
#   PART 1  ~20 min   evidence soft-decode sweep + token posteriors, RC7
#   PART 2  ~20 min   arm inventory, then the ONE real arm's test dump (dv2xl)
#   PART 3  ~8 h      two-stage long-tail factor training, CB-NTR vs control
#
# Timing is from the repo's own measured rate (factors_d6_5fold.sh:49, ~40 min
# per fold for this exact stage-1 recipe), which puts Part 3 near 8h, not the
# 5h first estimated. Hence --time=20:00:00: the earlier 12h left too little
# headroom against filesystem contention from concurrent 70B jobs.
#
# Every part writes NEW files under new names. Nothing the champion depends on
# is overwritten, and no part decides anything by itself -- each ends in a CPU
# measurement that can still say no.
#
# ---------------------------------------------------------------------------
# PART 1 -- WHY
#
# The evidence candidate pool holds 1.02-1.30 spans/post against 1.12 gold
# spans, and 42% of missed golds are ABSENT from it. Rearranging that pool
# cannot fix this on its own, because selection cannot choose what generation
# never proposed: `_decode_candidates` walks probs.argmax(-1), so a span the
# model scores at 45% has argmax O on every token and is invisible at any
# min_conf.
#
# `evidence_redump.py` already measures the soft decode -- but on the
# `ev5f_*` fold runs against fold vals, which are not what ships. This runs
# the identical decode on the RC7 taggers that build the evidence column, on
# the val twin AND on test, in one pass. That matters: evidence is the ONE
# component whose val and test artifacts are matched by construction, which
# is why its offline measurement is trustworthy in a way the risk
# component's is not.
#
# PART 2 -- WHY
#
# Non-BAG8 arms are best-in-pool on 5 of the 8 rarest labels and have no
# `_cv_test.npz`, so they were excluded from every audit and cannot ship
# however good their OOF is. An audit found that three of the
# four -- paircm, mnli3, pairprot -- are CPU-averaged BAGS of other arms, not
# trained models: there is no checkpoint to re-dump, and factors_bag_test.sh
# writes bag test dumps as `{TAG}_test.npz` while factor_perlabel_pool.py globs
# `{TAG}_cv_test.npz`. So a bag may already BE shippable and be invisible on a
# filename. Part 2 now prints that inventory first and only re-dumps `dv2xl`,
# which is a genuine single-arm run. Inference only.
#
# PART 3 -- WHY
#
# Factors are MACRO-F1 over 24 labels, each worth 1/24 whatever its support.
# `--neg-ratio 3` gives pairs ~ 4x positives, so `meaning in life` (45) trains
# on 180 pairs and `hopelessness` (745) on 2980 -- 16.6x, against a metric that
# weights them 1:1. `factor_loss_metric_gap.py` located the damage precisely:
# AP 0.23 vs 0.92. It is an ORDERING failure, and per-label thresholds cannot
# fix ordering, which is why two threshold lanes already died.
#
# Two literatures converge on the fix and both were checked before writing this:
#
#   Kang et al. ICLR 2020 (decoupling) and LDAM-DRW: learn the representation
#   on the IMBALANCED data, re-balance LATE. That predicts two results already
#   observed here -- reweighting from scratch BACKFIRED, and `fenum` (balanced
#   from scratch) is FIRST on the rarest label and LAST on every common one.
#
#   Wu et al. ECCV 2020 (Distribution-Balanced Loss) and Huang et al. EMNLP
#   2021 (CB-NTR, the text replication): the ONLY long-tail method validated at
#   this scale in both modalities -- VOC-MLT is 1142 images / 20 classes /
#   support 4-775 against this project's 1635 posts / 24 labels / support
#   8-745, and on Reuters with BERT + sigmoid the tail went BCE 0.00 -> NTR
#   22.3 -> CB-NTR 32.3. The mechanism it fixes is tail logits buried by
#   negative gradient so the RANKING collapses. That matches the measured AP
#   of 0.23 exactly.
#
# THE CONTROL IS THE POINT. Two stage-2 arms branch off the SAME stage-1
# checkpoint and differ in one variable:
#
#   cbntr    --full-enum --loss cbntr           the hypothesis
#   cbctl   --full-enum --loss bce_perlabel    two-stage + plain re-weighting
#
# Without `cbctl`, a win by `cbntr` cannot be attributed: it could be the
# decoupling rather than the loss. With it, the comparison is clean and cheap,
# because stage 2 is one epoch on a warm start.
#
# NOT INCLUDED, AND WHY. Soft-macro-F1 was proposed and killed by arithmetic:
# the rarest label has 8 positives in 1635, so a 256-pair batch holds ~1.25
# expected positives and the per-label F1 gradient degenerates exactly where it
# is needed. tau-norm and LWS are per-label monotone logit rescalings, i.e.
# mathematically inside the per-label threshold family already exhausted.
# SWA/EMA is the other strong small-data finding but it changes the CHAMPION's
# training recipe rather than adding an arm, so it is a separate submission.
#
# HOW TO JUDGE IT AFTERWARDS -- the falsifier is per label, not in aggregate:
# `cbntr` must beat `fenum` on the COMMON labels AND beat BAG8's best on the
# RARE ones. If it only does one, the two-stage did not work and the arm is
# just `fenum` under another name.
# ===========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# CLEAR THE HYPOTHESIS ENVIRONMENT. FACTOR_DEFS_JSON, FACTOR_ASPECTS_JSON,
# PROTECTIVE_TEMPLATE, CONFUSABLE_JSON and FACTOR_SUBSET are read at BOTH train
# and inference time, and `sbatch` defaults to --export=ALL, so whatever is set
# in the submitting login shell silently propagates into every part of this
# job. Every other script on this path clears them (factors_bag_test.sh:59-65,
# factor_arm_test_dumps.sh:84-87); this one did not, which meant a dirty launch
# shell could train and score models on hypotheses they never saw, with no
# error anywhere. Anything that genuinely needs one sets it locally below.
unset FACTOR_DEFS_JSON FACTOR_ASPECTS_JSON PROTECTIVE_TEMPLATE \
      CONFUSABLE_JSON FACTOR_SUBSET 2>/dev/null || true

set -u
OUT=artifacts/factor_runs
mkdir -p "$OUT" logs

FAIL=0
echo "node $(hostname)  $(date)"
df -h . | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---------------------------------------------------------------------------
echo "################################################################"
echo "PART 1 -- evidence soft-decode sweep on the shipped RC7"
echo "################################################################"
# --test-risk is NIDHI_cluster.csv on purpose. The stored
# artifacts/evidence_runs_rc/*_candidates.csv were dumped by
# evidence_cond_deploy.sh with CHAMP=NIDHI_cluster.csv, and the current
# champion NIDHI_ev7.csv differs from it on 5 risk labels. Conditioning on ev7
# would move those 5 posts for a reason unrelated to the soft decode and make
# the comparison unattributable. The script's reproduction check enforces this
# and will refuse if the argmax half fails to reproduce the shipped pool.
CLUSTER=submissions/NIDHI_cluster.csv
VALRISK=artifacts/nligaur/oof_base.csv
if [ -f "$CLUSTER" ] && [ -f "$VALRISK" ]; then
    python src/evidence_sweep_rc.py \
        --test-risk "$CLUSTER" --val-risk "$VALRISK" \
        --taus 0.20,0.35,0.50 --max-tokens 64 \
        || { echo "PART 1 FAILED (continuing to Part 2)"; FAIL=1; }
else
    echo "SKIP: missing $CLUSTER or $VALRISK"; FAIL=1
fi

# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "PART 2 -- test dumps for the rare-label winners (env-verified)"
echo "################################################################"
# ENVIRONMENT-MISMATCH GUARD. These arms have no run_config.json, so the
# hypothesis text they were TRAINED with is unrecorded -- FACTOR_DEFS_JSON,
# FACTOR_ASPECTS_JSON and PROTECTIVE_TEMPLATE are read from the environment at
# both train and inference time. Dumping test predictions under default env
# when the arm was trained under a different one is exactly the bug that made
# three GPU-trained evidence taggers look useless (trained conditioned, dumped
# unconditioned). So: re-dump fold-0 OOF under DEFAULT env and compare to the
# stored .npz. Identical -> defaults were used and the test dump is valid.
# Different -> refuse, rather than ship a silent mismatch.
#
# NOTE ON NAMING. Three arm names sometimes referenced elsewhere --
# paircm, mnli3, pairprot -- are NOT trained arms. They are CPU-averaged BAGS
# of other arms (arm_robustness.py:280-285; factors_bag_test.sh:14 launches
# paircm as ARMS="mnli dmnli cmine"). There is no runs/factors_paircm_cv_f0 to
# re-dump, so a naive loop over those names would print "skip" for each of
# them. Also, factors_bag_test.sh writes its test dump to `{TAG}_test.npz`
# while factor_perlabel_pool.py globs `{TAG}_cv_test.npz` -- so a bag can
# ALREADY have a test dump that nothing looks for, and be reported "not
# shippable" purely on a filename mismatch. The inventory below settles that
# in seconds; only real single-arm runs are re-dumped.
#
# pairprot additionally contains `prot`, which needs PROTECTIVE_TEMPLATE=1
# while its MNLI members need it OFF. factors_bag_test.sh REFUSES that mix on
# purpose, because the flag is global at inference. Probing pairprot under
# cleared defaults would have reported ENV MISMATCH for a knowable reason and
# shelved it for the wrong one.
echo ""
echo "--- inventory: which factor arms exist, and under which filename? ---"
python - <<'PY'
import glob, os
d = "artifacts/factor_runs"
BAGS = {"paircm": "mnli dmnli cmine (factors_bag_test.sh:14)",
        "mnli3": "mnli dmnli mnlipair", "pairprot": "mnli dmnli prot mnlipair"}
arms = sorted({os.path.basename(p).replace("_cv_oof_f0.npz", "")
               for p in glob.glob(f"{d}/*_cv_oof_f0.npz")})
print(f"  {'arm':<14}{'run dir':>9}{'_cv_test':>10}{'_test':>8}   note")
for a in arms:
    run = os.path.isdir(f"runs/factors_{a}_cv_f0")
    cvt = os.path.exists(f"{d}/{a}_cv_test.npz")
    bt = os.path.exists(f"{d}/{a}_test.npz")
    note = ""
    if a in BAGS:
        note = "BAG of " + BAGS[a]
    elif not run:
        note = "no checkpoints -- bag or deleted"
    if bt and not cvt:
        note += "  <-- HAS a test dump under the OTHER name"
    print(f"  {a:<14}{str(run):>9}{str(cvt):>10}{str(bt):>8}   {note}")
print("  An arm with _test.npz but no _cv_test.npz is SHIPPABLE TODAY --")
print("  factor_perlabel_pool.py just globs the wrong filename for it.")
PY

for a in dv2xl; do
    DST="${OUT}/${a}_cv_test.npz"
    if [ -f "${DST}" ]; then echo "--- ${a}: test dump exists ---"; continue; fi
    d0="runs/factors_${a}_cv_f0"
    if [ ! -d "$d0" ]; then echo "  ${a}: no ${d0}, skip"; continue; fi

    echo ""
    echo "=== ${a}: verifying env by re-dumping fold-0 OOF ==="
    # Two candidate inference lengths, because predict_factors_nli.py defaults
    # to --max-len 320 while the training recipes use 384 and nothing records
    # which was used. Probing both separates "trained under a different
    # hypothesis env" (unusable) from "dumped at the other length" (usable,
    # just dump at the matching one). Without the probe the second would be
    # misreported as the first and four shippable arms would stay shelved.
    GOODML=""
    TMP="${OUT}/_envcheck_${a}.npz"
    for ML in 320 384; do
        rm -f "${TMP}"
        python src/predict_factors_nli.py \
            --test "artifacts/folds/val_f0.csv" --nli-run "${d0}" \
            --max-len "${ML}" \
            --dump-probs "${TMP}" --out "/dev/null" \
            || { echo "  ${a}: re-dump at ${ML} failed"; continue; }
        python - "$a" "$TMP" "${OUT}/${a}_cv_oof_f0.npz" "$ML" <<'PY'
import sys, numpy as np
arm, tmp, ref, ml = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
# Exit 4 = the reference is missing/unreadable, which is NOT an environment
# mismatch and must not be reported as one.
if not __import__("os").path.exists(ref):
    print(f"  {arm} @{ml}: no reference {ref}"); sys.exit(4)
try:
    A = np.load(tmp, allow_pickle=True)["probs"].astype(float)
    B = np.load(ref, allow_pickle=True)["probs"].astype(float)
except Exception as e:
    print(f"  {arm} @{ml}: cannot read ({e})"); sys.exit(4)
if A.shape != B.shape:
    print(f"  {arm} @{ml}: shape {A.shape} vs stored {B.shape}"); sys.exit(3)
d = float(np.abs(A - B).max())
# 1e-4 was too tight. score_all runs under bf16 autocast, whose ~3-decimal
# mantissa makes 1e-3..1e-2 differences normal across GPU models or library
# builds -- so an identical model on a different node would be branded an ENV
# MISMATCH and a good arm shelved. What actually matters for AP is the
# ORDERING, so check rank agreement too and report both.
rho = float(np.mean([np.corrcoef(np.argsort(np.argsort(A[:, j])),
                                 np.argsort(np.argsort(B[:, j])))[0, 1]
                     for j in range(A.shape[1])]))
print(f"  {arm} @{ml}: max |re-dump - stored| = {d:.6f}   "
      f"mean per-label rank corr = {rho:.5f}")
sys.exit(0 if (d < 1e-2 and rho > 0.999) else 3)
PY
        if [ $? -eq 0 ]; then GOODML="$ML"; break; fi
    done
    rm -f "${TMP}"

    if [ -n "$GOODML" ]; then
        echo "  ${a}: VERIFIED at --max-len ${GOODML} (reproduces the stored OOF)"
        R=""; MISS=0
        for f in 0 1 2 3 4; do
            dd="runs/factors_${a}_cv_f${f}"
            [ -d "$dd" ] || { echo "  ${a}: missing ${dd}"; MISS=1; break; }
            R="${R}${R:+,}${dd}"
        done
        if [ $MISS -eq 0 ]; then
            python src/predict_factors_nli.py --test data/test.csv \
                --nli-run "${R}" --max-len "${GOODML}" --dump-probs "${DST}" \
                --out "${OUT}/${a}_cv_test_labels.csv" \
                || echo "  ${a}: test dump failed (continuing)"
        fi
    else
        echo "  ${a}: MISMATCH AT BOTH LENGTHS -- the stored OOF came from a"
        echo "        different hypothesis configuration, not just a different"
        echo "        --max-len. NOT dumping test; shipping it would repeat the"
        echo "        diverse-tagger bug (trained conditioned, dumped"
        echo "        unconditioned, three taggers wasted)."
    fi
done

# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "PART 3 -- two-stage long-tail factors: cbntr vs its control"
echo "################################################################"
if [ ! -f artifacts/folds/train_f0.csv ]; then
    echo "ERROR: artifacts/folds/ missing -- skipping Part 3"
    echo "DONE (FAIL=${FAIL})"; exit 1
fi

# TAG-COLLISION PRECONDITION. These arms were originally called `ltail` and
# `ltailb` -- but scripts/factors_longtail.sh ALREADY defines TAG=ltail with
# `--full-enum --loss bce_perlabel`, i.e. exactly what this job calls the
# CONTROL, and it uses the identical stage-1 path. Had that script ever run,
# the skip-if-exists guards below would have silently reused its checkpoints
# and the analysis would have compared bce_perlabel against bce_perlabel while
# reporting it as CB-NTR versus its control -- a null result indistinguishable
# from a real one. The arms are renamed to cbntr/cbctl, and this refuses to
# start if anything already owns those names. [[hpc-cluster-and-queue]] records
# TAG collision as a known hazard on this cluster; this is what checking for it
# looks like.
# It must still allow a RESUME of this same script, so it does not simply test
# for existence: it checks whether each existing run dir carries the
# run_config.json this script writes. Ours -> resume. Anyone else's -> stop.
for a in "${A_HYP:-cbntr}" "${A_CTL:-cbctl}" "${A_S1:-cbs1}"; do
    for d in "runs/factors_${a}_cv_f"*; do
        [ -d "$d" ] || continue
        if ! grep -q '"stage"' "${d}/run_config.json" 2>/dev/null; then
            echo "ERROR: ${d} exists but was NOT written by this script (no"
            echo "run_config.json with a stage marker). Another experiment owns"
            echo "the name '${a}' -- scripts/factors_longtail.sh used 'ltail'"
            echo "for a bce_perlabel arm on this exact path. Reusing it would"
            echo "compare the control against itself and report it as CB-NTR."
            echo "Rename the arms or clear the directory. Refusing to start."
            echo "DONE (FAIL=1)"; exit 1
        fi
    done
done

# Stage 1 is the known-good d6 recipe, TRAINED HERE rather than initialised
# from an existing arm: factor runs carry no run_config.json, so no existing
# checkpoint's training environment can be verified. Part 2 above exists
# entirely because of that gap, so this run closes it for its own outputs.
write_cfg () {   # $1 outdir  $2 stage  $3 extra args
    python - "$1" "$2" "$3" <<'PY'
import json, os, sys
d, stage, extra = sys.argv[1], sys.argv[2], sys.argv[3]
if os.path.isdir(d):
    json.dump({"stage": stage, "extra_args": extra,
               "model": "microsoft/deberta-v3-large",
               "max_len": 384, "dump_max_len": 320,
               "seed": 21,
               "FACTOR_DEFS_JSON": os.environ.get("FACTOR_DEFS_JSON", ""),
               "FACTOR_ASPECTS_JSON": os.environ.get("FACTOR_ASPECTS_JSON", ""),
               "PROTECTIVE_TEMPLATE": os.environ.get("PROTECTIVE_TEMPLATE", ""),
               "CONFUSABLE_JSON": os.environ.get("CONFUSABLE_JSON", ""),
               "FACTOR_SUBSET": os.environ.get("FACTOR_SUBSET", "")},
              open(os.path.join(d, "run_config.json"), "w"), indent=2)
PY
}

# ARM NAMES ARE OVERRIDABLE, for one specific recovery case. An earlier
# submission of this script used ltail/ltailb/ltails1 before being renamed to
# avoid a collision with factors_longtail.sh (verified not to exist on disk).
# Slurm snapshots the batch script at submit time, so if that earlier job dies
# part-way through Part 3, its completed folds are on disk under the OLD names
# and this script would otherwise ignore them and retrain from scratch.
# Resume it with:
#   sbatch --export=ALL,A_HYP=ltail,A_CTL=ltailb,A_S1=ltails1 scripts/onejob_0822.sh
A_HYP="${A_HYP:-cbntr}"
A_CTL="${A_CTL:-cbctl}"
A_S1="${A_S1:-cbs1}"
echo "arms: hypothesis=${A_HYP}  control=${A_CTL}  stage1=${A_S1}"

for f in 0 1 2 3 4; do
    S1="runs/factors_${A_S1}_cv_f${f}"

    # RESUME GUARD, and it needs BOTH conditions. The stage-1 checkpoint is
    # deleted further down once both stage-2 arms have branched off it, so on a
    # resubmission `[ ! -f "${S1}/model.pt" ]` is true for every COMPLETED fold
    # and stage 1 would be retrained -- ~40 min of DeBERTa-v3-large per fold --
    # only to be deleted again with nothing using it. A job resumed at fold 3
    # would burn ~2h producing nothing, which is exactly the scenario the
    # deletion was meant to protect against.
    if [ ! -f "${S1}/model.pt" ] \
       && { [ ! -f "runs/factors_${A_HYP}_cv_f${f}/model.pt" ] \
            || [ ! -f "runs/factors_${A_CTL}_cv_f${f}/model.pt" ]; }; then
        echo ""
        echo "=== fold ${f} STAGE 1 (standard, imbalanced representation) ==="
        python src/train_factors_nli.py \
            --train "artifacts/folds/train_f${f}.csv" \
            --val   "artifacts/folds/val_f${f}.csv" \
            --model microsoft/deberta-v3-large \
            --outdir "${S1}" \
            --seed 21 --lr 2e-5 --epochs 3 --max-len 384 \
            || { echo "STAGE1 FAILED fold ${f}"; FAIL=1; continue; }
        write_cfg "${S1}" stage1 ""
    else
        echo "--- fold ${f} stage 1: exists ---"
    fi

    # Both stage-2 arms branch off the SAME S1, so `cbntr` minus `cbctl` is
    # the loss and nothing else. lr 5e-6 for 1 epoch keeps the representation
    # intact -- the decoupling result is that the representation must not be
    # retrained under balancing, only the decision surface.
    #
    # --full-enum gives every label all 1635 posts instead of 4x its positives,
    # which restores the TRUE negative:positive ratio. That is the regime NTR
    # is designed for; under --neg-ratio 3 the ocean of negatives it tolerates
    # has already been sampled away and the loss would have little to do.
    for arm in "${A_HYP}" "${A_CTL}"; do
        if [ "$arm" = "${A_HYP}" ]; then
            LOSSARGS="--loss cbntr --ntr-kappa 0.05 --ntr-lambda 2.0 --cb-beta 0.99 --db-gamma 2.0"
        else
            LOSSARGS="--loss bce_perlabel"
        fi
        S2="runs/factors_${arm}_cv_f${f}"
        if [ ! -f "${S2}/model.pt" ]; then
            echo ""
            echo "=== fold ${f} STAGE 2 ${arm}: ${LOSSARGS} ==="
            python src/train_factors_nli.py \
                --train "artifacts/folds/train_f${f}.csv" \
                --val   "artifacts/folds/val_f${f}.csv" \
                --model microsoft/deberta-v3-large \
                --outdir "${S2}" --init-from "${S1}" \
                --full-enum ${LOSSARGS} \
                --seed 21 --lr 5e-6 --epochs 1 --max-len 384 \
                || { echo "STAGE2 ${arm} FAILED fold ${f}"; FAIL=1; continue; }
            write_cfg "${S2}" "stage2:${arm}" "--full-enum ${LOSSARGS}"
        else
            echo "--- fold ${f} stage 2 ${arm}: exists ---"
        fi

        # --max-len 320 DELIBERATELY, even though training used 384.
        # predict_factors_nli.py's default is 320 and Part 2 above establishes
        # what the existing pool was dumped at. Matching the pool keeps the
        # only new variable the LOSS; giving these two arms a longer context
        # than every arm they are compared against would hand them an
        # advantage that looks like the hypothesis working.
        DST="${OUT}/${arm}_cv_oof_f${f}.npz"
        if [ ! -f "${DST}" ]; then
            python src/predict_factors_nli.py \
                --test "artifacts/folds/val_f${f}.csv" \
                --nli-run "${S2}" --max-len 320 --dump-probs "${DST}" \
                --out "${OUT}/${arm}_cv_oof_f${f}_labels.csv" \
                || { echo "OOF dump failed ${arm} fold ${f}"; FAIL=1; }
        fi
    done

    # The stage-1 checkpoint is a pure intermediate: 15 DeBERTa-v3-large state
    # dicts at ~1.7GB each is ~25GB, enough to fill the quota mid-job and lose
    # the folds already trained. Drop S1 once BOTH stage-2 arms have branched
    # off it -- and only then, so a failed second arm can still be resumed.
    if [ -f "runs/factors_${A_HYP}_cv_f${f}/model.pt" ] \
       && [ -f "runs/factors_${A_CTL}_cv_f${f}/model.pt" ] \
       && [ -f "${S1}/model.pt" ]; then
        rm -f "${S1}/model.pt"
        echo "  freed ${S1}/model.pt (both stage-2 arms exist; config kept)"
    fi
    df -h . | tail -1
done

# 5-fold bagged test dumps, matching how BAG8's test column is built. Without
# these the arms cannot ship however good the OOF looks -- the mistake that
# left four rare-label winners unusable and made Part 2 necessary.
for arm in "${A_HYP}" "${A_CTL}"; do
    DST="${OUT}/${arm}_cv_test.npz"
    if [ ! -f "${DST}" ]; then
        R=""
        OK=1
        for f in 0 1 2 3 4; do
            dd="runs/factors_${arm}_cv_f${f}"
            [ -f "${dd}/model.pt" ] || { echo "  ${arm}: ${dd} missing, no test dump"; OK=0; break; }
            R="${R}${R:+,}${dd}"
        done
        if [ $OK -eq 1 ]; then
            echo ""
            echo "=== ${arm} test dump (5-fold bagged) ==="
            python src/predict_factors_nli.py --test data/test.csv \
                --nli-run "${R}" --max-len 320 --dump-probs "${DST}" \
                --out "${OUT}/${arm}_cv_test_labels.csv" \
                || { echo "test dump failed ${arm}"; FAIL=1; }
        fi
    fi
done

echo ""
echo "################################################################"
echo "DONE (FAIL=${FAIL}).  Everything below is CPU."
echo "################################################################"
cat <<'NEXT'

PART 1 -- evidence. Run these two IN ORDER and stop at the first no.
  # a. is the pool still starved? oracle recall on the val twin at each
  #    (tau, cap). If the 42% absent golds are STILL absent at tau 0.20, no
  #    decoding trick can ever reach them and the entire selection lane closes
  #    on one measurement instead of a tenth failed experiment.
  # b. only if the pool grew: MBR subset selection -- pick the subset that
  #    maximises EXPECTED phrase-F1 with the seven taggers as samples. vote>=2
  #    is a crude approximation of exactly that, so the metric-matched version
  #    is the right first try, needs no training, and cannot overfit 372 posts.
  # The shipped decode is INSIDE this pool (argmax at <=12 tokens), so any
  # rebuild must be able to return the incumbent exactly.

PART 2 -- factors, membership.
  python src/factor_perlabel_pool.py --shippable-only
  # `all29` has ONE bit of freedom and cannot overfit. If only `perlabel` wins,
  # the per-label search is fitting the sample, which is what killed threshold
  # shrinkage and the greedy membership sweeps.

PART 3 -- factors, the long-tail arms. THE FALSIFIER IS PER LABEL.
  python src/factor_arm_ap_protective.py \
      --arms mnli dmnli cmine prot d4 qnli mnli7 lnli fenum cbntr cbctl \
      --all-labels
  # cbntr must beat fenum on hopelessness / mental-health AND beat BAG8's best
  # on substance-use / meaning-in-life. One without the other means the
  # two-stage did not work.
  # cbntr vs cbctl isolates CB-NTR from the decoupling. If cbctl matches
  # cbntr, the loss did nothing and the gain (if any) is the two-stage.
  python src/factor_member_forward.py

SHIPPING BAR, unchanged and non-negotiable:
  * judge by WIN RATE across independent author splits and per-split sd, never
    the CI of the mean, which can exclude zero while still failing to
    generalize to held-out authors
  * one single-variable change per submission
  * S1 must return EXACTLY 0.8061 on any factors-only change; only S2 may move
NEXT
