#!/bin/bash
#SBATCH --job-name=cbndiv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out

# ===========================================================================
# CB-NTR ON ARCHITECTURALLY DIVERSE BASES + THE MISSING PARENT TEST DUMPS.
#
# WHAT THIS JOB IS FOR. An added member helps a bag through DIVERSITY rather
# than through individual strength. A CB-NTR stage-2 child warm-started off an
# existing arm therefore contributes only when its PARENT is decorrelated from
# the rest of the pool: two children of the same synthetic-data parent are
# near-duplicates of each other and pool worse than either one alone. The
# backbone is the coarsest axis of decorrelation available, and the pool is
# dominated by a single one (deberta-v3-large), so this job applies CB-NTR to
# the non-v3 backbones already sitting in runs/ with five fold checkpoints
# each: mrob/mbert/rmnli/bmnli/dv1mnli/dv2xl by their names -- but names lie,
# so the ACTUAL backbone is read from runs/<dir>/tok at runtime and anything
# that turns out to be deberta-v3 is filtered out, one arm per unique
# backbone is kept, and the rest is reported for the log.
#
# PART 1 IS FREE VALUE. Several old arms (bmnli, dv1mnli, ml512, rmnli, nli,
# mnlip, mnlipl, ...) have OOF dumps but NO _cv_test.npz. An arm without a
# test dump cannot ship however good its OOF is. Test dumps are inference-only
# and cheap, so this job produces every missing one it can VERIFY first: re-dump
# fold-0 OOF under the default env at both --max-len 320 and 384 and require it
# to reproduce the stored OOF. Refusing on mismatch is what catches an arm whose
# inference environment no longer matches the one it was trained under -- a
# silent failure that otherwise looks like a merely mediocre arm. The parent
# test dump is made at the max-len that REPRODUCES the stored OOF, because
# matched OOF/test construction is what lets an offline measurement transfer;
# new arms are dumped at 320 like the whole existing pool.
#
# PROTECTIVE_TEMPLATE ARMS ARE HANDLED, NOT GUESSED. prot, pairprot, mnlip,
# dmnlip, cminep, mnlifep were trained with PROTECTIVE_TEMPLATE=1, which is
# read at BOTH train and inference time. They are excluded from CB-NTR
# training (the flag is global at inference and a stage-2 child would inherit
# the requirement silently), but their missing test dumps ARE produced -- with
# PROTECTIVE_TEMPLATE=1 set for exactly those commands, gated by the same
# fold-0 reproduction probe, which fails loudly if the flag story is wrong.
# prot5/prot5a/prot5d are FACTOR_SUBSET specialists whose probability matrices
# are not 24-wide, and are excluded from everything.
#
# ORDERED CHEAPEST-AND-SUREST FIRST, resumable at every stage:
#   PART 0   ~1 min    inventory: every arm, its ACTUAL backbone, its dumps
#   PART 1   ~1-2 h    env-verified test dumps for every parent missing one
#   PART 2   ~6-14 h   CB-NTR stage 2 on up to MAXARMS diverse bases
#                      (1 epoch x 5 folds each + OOF + bagged test dump)
#
# NOTHING SHIPS FROM HERE. Judge afterwards with factor_member_forward.py /
# arm_robustness.py over 50 author splits. The falsifier: each new arm is an
# ADD on top of the shipped set, not a standalone arm, and must beat the fixed
# per-arm dilution tax that any additional member charges. Judge on win rate
# and per-split sd, never on the CI of the mean -- that CI shrinks with the
# number of resamples and can exclude zero at a win rate barely above chance,
# which says nothing about a single held-out draw.
# ===========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
NT="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${NT}" MKL_NUM_THREADS="${NT}"

# sbatch defaults to --export=ALL, so a dirty login shell would otherwise
# train and score on hypotheses these arms never saw, silently.
unset FACTOR_DEFS_JSON FACTOR_ASPECTS_JSON PROTECTIVE_TEMPLATE \
      CONFUSABLE_JSON FACTOR_SUBSET 2>/dev/null || true

set -u
OUT=artifacts/factor_runs
mkdir -p "$OUT" logs

LOSSARGS="--loss cbntr --ntr-kappa 0.05 --ntr-lambda 2.0 --cb-beta 0.99 --db-gamma 2.0"
MAXARMS="${MAXARMS:-4}"
export MAXARMS

# Trained with PROTECTIVE_TEMPLATE=1 (read at train AND inference time).
PROT_ARMS="prot pairprot mnlip dmnlip cminep mnlifep"
# FACTOR_SUBSET specialists: dead lane, and not 24-label-shaped. Never touch.
SPEC_ARMS="prot5 prot5a prot5d"

needs_pt () { case " ${PROT_ARMS} " in *" $1 "*) return 0;; *) return 1;; esac; }
is_spec ()  { case " ${SPEC_ARMS} " in *" $1 "*) return 0;; *) return 1;; esac; }

# Wrapper so PROTECTIVE_TEMPLATE=1 is set for EXACTLY the commands that need
# it and nothing else. $1 = arm, rest = the python command line.
run_pred () {
    local _arm="$1"; shift
    if needs_pt "${_arm}"; then
        PROTECTIVE_TEMPLATE=1 python "$@"
    else
        python "$@"
    fi
}

free_gb () { df -Pk . | awk 'NR==2 {print int($4/1024/1024)}'; }

FAIL=0
echo "node $(hostname)  $(date)"
df -h . | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

[ -f artifacts/folds/train_f0.csv ] || { echo "ERROR: artifacts/folds/ missing"; exit 1; }
[ -f data/test.csv ]               || { echo "ERROR: data/test.csv missing"; exit 1; }

# ---------------------------------------------------------------------------
# ENV/CONSTRUCTION PROBE, shared by Parts 1 and 2. Re-dumps fold-0 OOF under
# the arm's declared env at 320 then 384 and compares to the stored npz.
# Result is cached in a marker file so a resumed job never re-probes.
#   return 0 -> verified; the good max-len is echoed on stdout
#   return 1 -> PROVEN mismatch at both lengths (different hypothesis env)
#   return 2 -> unverifiable (no stored fold-0 OOF to compare against)
# 1e-2 / rank-corr 0.999 tolerances, not 1e-4: score_all runs under bf16
# autocast, whose ~3-decimal mantissa makes 1e-3..1e-2 differences normal
# across GPU models -- what matters for AP is the ORDERING.
# ---------------------------------------------------------------------------
verify_arm () {
    local arm="$1"
    local okm="${OUT}/_envok_${arm}"
    local badm="${OUT}/_envbad_${arm}"
    if [ -f "${okm}" ]; then cat "${okm}"; return 0; fi
    [ ! -f "${badm}" ] || return 1
    local ref="${OUT}/${arm}_cv_oof_f0.npz"
    if [ ! -f "${ref}" ]; then
        echo "  ${arm}: no stored fold-0 OOF -- construction unverifiable" >&2
        return 2
    fi
    local tmp="${OUT}/_envcheck_${arm}.npz"
    local ml
    for ml in 320 384; do
        rm -f "${tmp}"
        run_pred "${arm}" src/predict_factors_nli.py \
            --test artifacts/folds/val_f0.csv \
            --nli-run "runs/factors_${arm}_cv_f0" \
            --max-len "${ml}" --dump-probs "${tmp}" --out /dev/null >&2 \
            || { echo "  ${arm}: re-dump at ${ml} failed" >&2; continue; }
        python - "${arm}" "${tmp}" "${ref}" "${ml}" >&2 <<'PY'
import sys
import numpy as np
arm, tmp, ref, ml = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    A = np.load(tmp, allow_pickle=True)["probs"].astype(float)
    B = np.load(ref, allow_pickle=True)["probs"].astype(float)
except Exception as e:
    print(f"  {arm} @{ml}: cannot read ({e})"); sys.exit(3)
if A.shape != B.shape:
    print(f"  {arm} @{ml}: shape {A.shape} vs stored {B.shape}"); sys.exit(3)
d = float(np.abs(A - B).max())
rho = float(np.mean([np.corrcoef(np.argsort(np.argsort(A[:, j])),
                                 np.argsort(np.argsort(B[:, j])))[0, 1]
                     for j in range(A.shape[1])]))
print(f"  {arm} @{ml}: max |re-dump - stored| = {d:.6f}   "
      f"mean per-label rank corr = {rho:.5f}")
sys.exit(0 if (d < 1e-2 and rho > 0.999) else 3)
PY
        if [ $? -eq 0 ]; then
            rm -f "${tmp}"
            echo "${ml}" > "${okm}"
            cat "${okm}"
            return 0
        fi
    done
    rm -f "${tmp}"
    : > "${badm}"
    return 1
}

# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "PART 0 -- inventory: every arm, its ACTUAL backbone, its dumps"
echo "################################################################"
python - <<'PY'
import glob, os
d = "artifacts/factor_runs"
dirs = sorted(glob.glob("runs/factors_*_cv_f0"))
print(f"  {'arm':<12}{'folds':>6}{'oof_f0':>8}{'_cv_test':>10}{'_test':>7}   base model (runs/<dir>/tok)")
for r in dirs:
    a = os.path.basename(r)[len("factors_"):-len("_cv_f0")]
    folds = sum(os.path.isfile(f"runs/factors_{a}_cv_f{f}/model.pt") for f in range(5))
    tokp = os.path.join(r, "tok")
    tok = open(tokp).read().strip() if os.path.isfile(tokp) else "<no tok file>"
    oof = os.path.exists(f"{d}/{a}_cv_oof_f0.npz")
    cvt = os.path.exists(f"{d}/{a}_cv_test.npz")
    bt = os.path.exists(f"{d}/{a}_test.npz")
    print(f"  {a:<12}{folds:>4}/5{str(oof):>8}{str(cvt):>10}{str(bt):>7}   {tok}")
print("  (an arm with _test.npz but no _cv_test.npz is a BAG dump under the")
print("   other filename -- factor_perlabel_pool.py globs {TAG}_cv_test.npz)")
PY

# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "PART 1 -- env-verified test dumps for every parent missing one"
echo "################################################################"
for d0 in runs/factors_*_cv_f0; do
    [ -d "${d0}" ] || continue
    arm="${d0#runs/factors_}"; arm="${arm%_cv_f0}"

    is_spec "${arm}" && { echo "  ${arm}: FACTOR_SUBSET specialist, excluded"; continue; }
    # Children of the CB-NTR scripts get their dumps from their own jobs.
    case "${arm}" in cbn*|ltail*|cbntr|cbctl|cbs1) continue;; esac
    if [ -f "${OUT}/${arm}_cv_test.npz" ] || [ -f "${OUT}/${arm}_test.npz" ]; then
        continue
    fi

    R=""; OK=1
    for f in 0 1 2 3 4; do
        dd="runs/factors_${arm}_cv_f${f}"
        if [ ! -f "${dd}/model.pt" ] || [ ! -f "${dd}/tok" ] \
           || [ ! -f "${dd}/thresholds.npy" ]; then OK=0; break; fi
        R="${R}${R:+,}${dd}"
    done
    [ ${OK} -eq 1 ] || { echo "  ${arm}: incomplete fold checkpoints (bag or deleted), skip"; continue; }

    echo ""
    echo "=== ${arm}: missing test dump -- verifying construction first ==="
    GOODML="$(verify_arm "${arm}")"; rc=$?
    if [ ${rc} -eq 0 ]; then
        echo "  ${arm}: VERIFIED at --max-len ${GOODML} -- dumping test (5-fold bagged)"
        run_pred "${arm}" src/predict_factors_nli.py --test data/test.csv \
            --nli-run "${R}" --max-len "${GOODML}" \
            --dump-probs "${OUT}/${arm}_cv_test.npz" \
            --out "${OUT}/${arm}_cv_test_labels.csv" \
            || { echo "  ${arm}: test dump FAILED"; rm -f "${OUT}/${arm}_cv_test.npz"; FAIL=1; }
    elif [ ${rc} -eq 1 ]; then
        echo "  ${arm}: MISMATCH AT BOTH LENGTHS -- the stored OOF came from a"
        echo "        different hypothesis configuration. NOT dumping test;"
        echo "        shipping it would repeat the diverse-tagger bug (trained"
        echo "        conditioned, dumped unconditioned, three taggers wasted)."
        FAIL=1
    else
        echo "  ${arm}: skipped -- nothing to verify a test dump against."
    fi
done
df -h . | tail -1

# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "PART 2 -- CB-NTR stage 2 on architecturally diverse bases"
echo "################################################################"
# BASES ARE DISCOVERED, NOT HARDCODED, because this repo's mirror of runs/ is
# not readable from the machine that wrote this script and names lie (ml512
# is probably a v3 max-len variant; mnli could be anything). Rules:
#   * all five folds must have model.pt AND tok, and the five toks must agree
#   * anything whose tok contains "deberta-v3" is OUT -- the pool is already
#     saturated with that backbone, which is the axis this job is widening
#   * PROTECTIVE_TEMPLATE and FACTOR_SUBSET arms are OUT (see header)
#   * one arm per unique backbone -- near-duplicate parents make near-duplicate
#     children, so a second child off the same backbone adds no diversity
#   * preference order puts the most architecturally distinct first, then
#     alphabetical; capped at MAXARMS (default 4)
# Override by submitting with --export=ALL,BASES="mrob mbert",MAXARMS=2
if [ -n "${BASES:-}" ]; then
    echo "BASES overridden by environment: ${BASES}"
else
    BASES="$(python - <<'PY'
import glob, os, sys
pref = ["mrob", "mbert", "bmnli", "rmnli", "dv1mnli", "dv2xl", "mnli", "nli", "ml512"]
excl = set("prot pairprot mnlip dmnlip cminep mnlifep prot5 prot5a prot5d".split())
arms = sorted({os.path.basename(p)[len("factors_"):-len("_cv_f0")]
               for p in glob.glob("runs/factors_*_cv_f0")})
cands = []
for a in arms:
    if a in excl or a.startswith("cbn") or a.startswith("ltail") \
       or a in {"cbntr", "cbctl", "cbs1"}:
        continue
    toks = set()
    ok = True
    for f in range(5):
        d = f"runs/factors_{a}_cv_f{f}"
        if not (os.path.isfile(f"{d}/model.pt") and os.path.isfile(f"{d}/tok")):
            ok = False; break
        toks.add(open(f"{d}/tok").read().strip())
    if not ok:
        print(f"  {a}: incomplete folds, out", file=sys.stderr); continue
    if len(toks) != 1:
        print(f"  {a}: folds disagree on base {sorted(toks)}, out", file=sys.stderr); continue
    tok = toks.pop()
    if "deberta-v3" in tok.lower():
        print(f"  {a}: {tok} -- v3, already mined, out", file=sys.stderr); continue
    cands.append((a, tok))
order = {n: i for i, n in enumerate(pref)}
cands.sort(key=lambda x: (order.get(x[0], 999), x[0]))
seen, picks = set(), []
for a, tok in cands:
    if tok in seen:
        print(f"  {a}: duplicate backbone {tok}, out", file=sys.stderr); continue
    seen.add(tok)
    if len(picks) < int(os.environ.get("MAXARMS", "4")):
        picks.append(a)
        print(f"  {a}: PICKED  ({tok})", file=sys.stderr)
    else:
        print(f"  {a}: eligible but over MAXARMS cap  ({tok})", file=sys.stderr)
print(" ".join(picks))
PY
)"
fi
echo "bases: ${BASES:-<none>}"
echo "loss : ${LOSSARGS}"
[ -n "${BASES}" ] || echo "NO eligible diverse bases found -- Part 2 is a no-op."

for BASE in ${BASES}; do
    TAG="cbn${BASE}"
    echo ""
    echo "################################################################"
    echo "ARM ${TAG}  (CB-NTR stage 2 from ${BASE})"
    echo "################################################################"

    # TAG-collision guard: a dir written by the cbntr scripts carries this
    # marker in run_config.json. Ours or factors_cbntr_arms.sh's -> resume
    # (identical recipe). Anyone else's -> the TAG is owned, skip the arm.
    SKIP=0
    for d in "runs/factors_${TAG}_cv_f"*; do
        [ -d "${d}" ] || continue
        grep -q '"cbntr_stage2"' "${d}/run_config.json" 2>/dev/null || {
            echo "ERROR: ${d} exists and was not written by a cbntr script."
            echo "Another experiment owns TAG=${TAG}. Skipping this arm."
            SKIP=1; break; }
    done
    [ ${SKIP} -eq 0 ] || { FAIL=1; continue; }

    # Refuse to warm-start off a parent PROVEN to have been trained under a
    # different hypothesis env -- one epoch at lr 5e-6 will not un-teach that,
    # and the child would silently be a mismatch too. Unverifiable (rc=2) is
    # allowed: the child is trained AND judged under the default env, so its
    # own OOF/test stay honest either way.
    verify_arm "${BASE}" > /dev/null; rc=$?
    if [ ${rc} -eq 1 ]; then
        echo "  ${BASE}: parent fails the fold-0 reproduction probe -- skipping arm"
        FAIL=1; continue
    elif [ ${rc} -eq 2 ]; then
        echo "  ${BASE}: parent construction unverifiable (no stored OOF); proceeding --"
        echo "          the child's own OOF and test dumps are matched by construction."
    fi

    for f in 0 1 2 3 4; do
        INIT="runs/factors_${BASE}_cv_f${f}"
        DST="runs/factors_${TAG}_cv_f${f}"
        if [ ! -f "${INIT}/model.pt" ] || [ ! -f "${INIT}/tok" ]; then
            echo "  ${BASE} fold ${f}: missing model.pt or tok, skipping arm"
            FAIL=1; SKIP=1; break
        fi
        # The model name comes from the BASE, not from a flag: --init-from
        # checks the state dict against the architecture, and a mismatch there
        # is a hard failure five folds deep.
        MODEL=$(cat "${INIT}/tok")

        if [ ! -f "${DST}/model.pt" ]; then
            # DeBERTa-class checkpoints are ~1.7GB (dv2xl bigger) and a full
            # disk mid-job has already cost a run once. Stop before, not after.
            FREE=$(free_gb)
            if [ "${FREE}" -lt 15 ]; then
                echo "ABORT: only ${FREE}G free -- refusing to train into a full disk."
                echo "DONE (FAIL=1)"; exit 1
            fi
            echo ""
            echo "=== ${TAG} fold ${f}: from ${INIT}  (${MODEL}) ==="
            python src/train_factors_nli.py \
                --train "artifacts/folds/train_f${f}.csv" \
                --val   "artifacts/folds/val_f${f}.csv" \
                --model "${MODEL}" \
                --outdir "${DST}" --init-from "${INIT}" \
                --full-enum ${LOSSARGS} \
                --seed 21 --lr 5e-6 --epochs 1 --max-len 384 \
                || { echo "FAILED ${TAG} fold ${f}"; FAIL=1; continue; }
            MODEL="${MODEL}" BASE="${BASE}" python - "${DST}" <<'PY'
import json, os, sys
d = sys.argv[1]
if os.path.isdir(d):
    json.dump({"cbntr_stage2": True, "script": "factors_cbntr_diverse.sh",
               "init_from_arm": os.environ.get("BASE"),
               "model": os.environ.get("MODEL"), "max_len": 384,
               "dump_max_len": 320, "seed": 21, "lr": 5e-6, "epochs": 1,
               "loss": "cbntr", "cb_beta": 0.99, "ntr_kappa": 0.05,
               "ntr_lambda": 2.0, "db_gamma": 2.0,
               "FACTOR_DEFS_JSON": os.environ.get("FACTOR_DEFS_JSON", ""),
               "FACTOR_ASPECTS_JSON": os.environ.get("FACTOR_ASPECTS_JSON", ""),
               "PROTECTIVE_TEMPLATE": os.environ.get("PROTECTIVE_TEMPLATE", "")},
              open(os.path.join(d, "run_config.json"), "w"), indent=2)
PY
        else
            echo "--- ${TAG} fold ${f}: exists ---"
        fi

        # --max-len 320 matches every other arm's dump, so the only new
        # variable is the arm itself.
        O="${OUT}/${TAG}_cv_oof_f${f}.npz"
        if [ ! -f "${O}" ]; then
            python src/predict_factors_nli.py \
                --test "artifacts/folds/val_f${f}.csv" --nli-run "${DST}" \
                --max-len 320 --dump-probs "${O}" \
                --out "${OUT}/${TAG}_cv_oof_f${f}_labels.csv" \
                || { echo "OOF dump failed ${TAG} fold ${f}"; FAIL=1; }
        fi
    done
    [ ${SKIP} -eq 0 ] || continue

    # OOF = per-fold held-out, test = 5-fold bagged: the matched construction
    # shared by the two prior changes that generalized best from offline
    # measurement.
    DST="${OUT}/${TAG}_cv_test.npz"
    if [ ! -f "${DST}" ]; then
        R=""; OK=1
        for f in 0 1 2 3 4; do
            d="runs/factors_${TAG}_cv_f${f}"
            [ -f "${d}/model.pt" ] || { echo "  ${d} missing, no test dump"; OK=0; break; }
            R="${R}${R:+,}${d}"
        done
        if [ ${OK} -eq 1 ]; then
            echo ""
            echo "=== ${TAG} test dump (5-fold bagged) ==="
            python src/predict_factors_nli.py --test data/test.csv \
                --nli-run "${R}" --max-len 320 --dump-probs "${DST}" \
                --out "${OUT}/${TAG}_cv_test_labels.csv" \
                || { echo "test dump failed ${TAG}"; FAIL=1; }
        fi
    fi
    df -h . | tail -1
done

echo ""
echo "################################################################"
echo "DONE (FAIL=${FAIL}).  Everything below is CPU."
echo "################################################################"
cat <<'NEXT'
  python src/factor_member_forward.py 2>&1 | tail -40

  # THE BAR. The shipped set is BAG8 + ltail + friendfac + cbnsynapi.
  # Judge every new arm as an ADD on top of THAT set -- never on top of
  # BAG8, or the incumbents' contributions double-count.
  # Each add must clear the fixed per-arm dilution tax; judge by WIN RATE
  # across independent author splits and per-split sd via arm_robustness.py
  # (50 splits), never by the CI of the mean, which can exclude zero while
  # still failing to generalize to held-out authors.

  # THE PREDICTION THIS JOB TESTS: cbn<diverse> children should decorrelate
  # MORE than cbnsynsty did (parents share no backbone with the pool), so if
  # cbnmrob/cbnmbert/etc. also add less than cbnsynapi did, the diversity
  # story is wrong and the CB-NTR lane closes for good.

  # Parents that gained a _cv_test.npz in Part 1 are shippable candidates in
  # factor_perlabel_pool.py / the membership audits IMMEDIATELY -- rerun
  #   python src/factor_perlabel_pool.py --shippable-only
  # and remember the ensemble-membership audit needs the 5th check:
  # same-model provenance PER MEMBER.

  # One single-variable change per submission; S1 must return EXACTLY 0.8061
  # on any factors-only change; only S2 may move.
NEXT
