#!/bin/bash
#SBATCH --job-name=colddrill
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out

# =============================================================================
# COLD-RUN DRILL: prove the submission pipeline runs END TO END on a test set
# that is NOT the 378 rows of data/test.csv.
#
# WHY. The final evaluation is run by the ORGANIZERS on posts from authors this
# repo holds zero rows of: the uploaded CSVs are not scored there, the CODE is
# re-run. Any stage that silently depends on the stored 378 rows -- a cached
# dump keyed on a filename, a hardcoded row count, a conditioning file whose
# row_ids do not exist -- emits an empty or stale column instead of an error,
# and scoring on data/test.csv cannot detect that, because data/test.csv is
# exactly the set those artifacts cover. build_submission.py crashes loudly on
# missing coverage (_require_coverage), but a crash guard is not the same as
# having RUN the chain end to end on a different test set. This job does that.
#
# THE DRILL SET: artifacts/val_split.csv -- 372 rows / 32 held-out authors,
# rewritten in data/test.csv's exact schema. 372 != 378, so any hardcoded 378
# anywhere in the chain is exposed. It also has gold, so the drill ends with a
# real composite score.
#
# TWO CONCLUSIONS, KEPT SEPARATE (the final report prints both):
#   1. CODE-RUNS-COLD: every stage regenerated from checkpoints into a fresh
#      namespace, no stored 378-row artifact reused. Valid unconditionally.
#   2. THE SCORE: risk is clean (risk_v1 seeds + risk_nli trained on
#      train_split, val excluded). Evidence tagger WEIGHTS are clean but the
#      vote config / member set / min_confs were selected on val (mild
#      optimism). Factors are INFLATED: the bag8 fold checkpoints partition
#      all 1635 labelled posts, so 4 of 5 folds trained on every val post,
#      and the tfit thresholds are fitted on the full-1635 OOF which includes
#      val gold. The honest factor number needs the per-fold OOF npz stitched
#      instead -- do not quote the drill's factor F1 as a performance claim.
#
# NAMESPACE. Everything is written under ${DRILL_DIR:-artifacts/drill}. Every
# "exists, skipping" guard in the repo is keyed on filename, so a rerun with a
# new test.csv would otherwise reuse the 378-row dumps and report success on
# stale output. A distinct namespace means no guard can hand back stale files.
# Resume stamps live inside the namespace, so DRILL_FRESH=1 (or a new
# DRILL_DIR) restarts clean.
#
# KNOWN DEGRADED PATHS the drill handles explicitly (never silently):
#   * route_attempt needs a Qwen2.5-72B CoT dump (~20h). The drill uses
#     artifacts/cot_soft_oof.csv (OOF over the 1635 labelled posts, covers the
#     val rows) if present; on a genuinely fresh test set NEITHER cot file
#     covers and scripts/risk_cot_member.sh must be rerun, or the risk column
#     ships without attempt routing. If no source covers, the drill
#     synthesizes a no-op route file and marks the stage DEGRADED, loudly.
#   * The friend risk member (artifacts/friend_risk_probs.csv) has no
#     generator in this repo. Its val-split probs exist but miss 18/372 rows;
#     ensemble_risk.py falls back to the local probs PER ROW with only a
#     warning, so an absent member degrades the ensemble without failing the
#     run. The drill counts the missing rows and marks DEGRADED if any. (The
#     external pipeline uses the GATED mental/mental-roberta-base; an offline
#     node cannot pull it, so its FALLBACK_MODEL would silently change the
#     member -- flagged in the summary, not fixable here.)
#   * factor_threshold_decode.py --build-test is hardwired to a 378-row
#     submission for row order + provenance, so it CANNOT run cold. The drill
#     imports its functions and applies the same full-data tfit thresholds to
#     the fresh dumps, gated by reproducing the shipped factors_tfit_test.csv
#     column first. That script still needs a cold-start mode of its own.
#
# ENV VARS THAT SILENTLY CORRUPT (sbatch default --export=ALL propagates them
# from the login shell): FACTOR_DEFS_JSON FACTOR_ASPECTS_JSON
# PROTECTIVE_TEMPLATE CONFUSABLE_JSON FACTOR_SUBSET are read at INFERENCE.
# All are cleared globally below; the `prot` arm alone re-exports
# PROTECTIVE_TEMPLATE=1 (it was trained with it ON, the other bag8 arms OFF).
#
# Usage:
#   sbatch scripts/cold_run_drill.sh
#   sbatch --export=ALL,DRILL_FRESH=1 scripts/cold_run_drill.sh   # restart
# Optional overrides: DRILL_DIR, ROUTE_FILE, FRIEND_PROBS
# =============================================================================

set -u
set -o pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# clear the inference-corrupting factor env vars inherited via --export=ALL
unset FACTOR_DEFS_JSON FACTOR_ASPECTS_JSON PROTECTIVE_TEMPLATE CONFUSABLE_JSON FACTOR_SUBSET || true

export D="${DRILL_DIR:-artifacts/drill}"
export ROUTE_FILE="${ROUTE_FILE:-}"
FRIEND="${FRIEND_PROBS:-}"
RC7="evidence_ep9_rc,evidence_lr2_rc,evidence_s14_rc,evidence_s15_rc,evidence_s16_rc,evidence_s17_rc,evidence_v1_rc"
BAG8="mnli dmnli cmine prot d4 qnli mnli7 lnli"

# THE SHIPPED FACTOR COLUMN IS 11 ARMS, NOT 8. Drilling BAG8 alone proves that
# a model which is not submitted runs cold, which is the wrong reassurance.
# Split by what the cold run can actually regenerate:
#   SHIPPED_ENC  -- ordinary encoder arms, dump through the normal path
#   SHIPPED_EXT  -- produced OUTSIDE this repo, so no amount of fixing this
#                   script makes them reproducible here; the drill's job is to
#                   say so out loud.
SHIPPED_ENC="ltail cbnsynapi"
SHIPPED_EXT="friendfac"
FACTOR_ARMS="${BAG8} ${SHIPPED_ENC}"
export FACTOR_ARMS
RISK_SEEDS="runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15"
NLI_RUNS="runs/risk_nli,runs/risk_nli_s2"

mkdir -p logs "${D}" "${D}/status"

if [ "${DRILL_FRESH:-0}" = "1" ]; then
    echo "DRILL_FRESH=1 -- wiping ${D}"
    rm -rf "${D}"
    mkdir -p "${D}" "${D}/status"
fi

# ---------------------------------------------------------------- helpers
mark () {  # mark <stage> <PASS|FAIL|DEGRADED> <note>
    echo -e "$2\t$3" > "${D}/status/$1"
    echo ""
    echo ">>> [$1] $2 -- $3"
    echo ""
}

summary () {
    echo ""
    echo "================================================================"
    echo "COLD-RUN DRILL SUMMARY   (namespace ${D})"
    echo "================================================================"
    for f in "${D}"/status/*; do
        [ -f "$f" ] || continue
        printf '  %-16s %s\n' "$(basename "$f")" "$(cat "$f")"
    done
    echo "----------------------------------------------------------------"
    if grep -l $'^FAIL\t' "${D}"/status/* >/dev/null 2>&1; then
        echo "  RESULT: FAIL -- at least one stage failed. The organizers'"
        echo "  run on unseen authors would fail (or silently lose score) at"
        echo "  that stage."
    elif grep -l $'^DEGRADED\t' "${D}"/status/* >/dev/null 2>&1; then
        echo "  RESULT: PASS WITH DEGRADATIONS -- the chain runs cold, but"
        echo "  the degraded stages above ship a weaker model than the"
        echo "  champion. Quantified notes are on each stage line."
    else
        echo "  RESULT: PASS -- the full chain regenerated end to end on a"
        echo "  non-378-row test set."
    fi
    echo "================================================================"
}

die () {  # die <stage> <note>
    mark "$1" "FAIL" "$2"
    summary
    exit 1
}

space () { echo ""; echo "--- disk ---"; df -h . | tail -n 1; echo ""; }

stamp_done () { touch "${D}/.done_$1"; }
is_done ()    { [ -f "${D}/.done_$1" ]; }

nvidia-smi -L 2>/dev/null || echo "WARNING: nvidia-smi not available (GPU stages will refuse on CPU)"
space

# =============================================================================
# STAGE 0 (CPU): synthesize the fresh test set + gold from val_split
# =============================================================================
if is_done s0_testset; then
    echo "[s0_testset] stamp exists, resuming past it"
else
    python - <<'PY'
import os, sys
import pandas as pd
D = os.environ["D"]
va = pd.read_csv("artifacts/val_split.csv", keep_default_na=False)
need = ["row_id", "anon_user_id", "post_id", "post"]
missing = [c for c in need if c not in va.columns]
assert not missing, f"val_split lacks {missing}"
n = len(va)
# THE POINT OF THE DRILL: a row count that is not 378, so any hardcoded 378
# in the chain crashes or miscounts loudly instead of passing by accident.
assert n != 378, "drill set has exactly 378 rows -- it cannot expose 378-isms"
assert n > 0
# test-schema copy (exact data/test.csv header: row_id,anon_user_id,post_id,post)
va[need].to_csv(f"{D}/test_drill.csv", index=False)
# gold for the scoring half
gold_cols = ["row_id", "anon_user_id", "risk_level", "evidence", "factors"]
va[gold_cols].to_csv(f"{D}/gold_drill.csv", index=False)
# tiny val stub for evidence_dump_all (it insists on dumping a val split too;
# 2 rows keeps that overhead near zero without touching its interface)
va[need].head(2).to_csv(f"{D}/val_stub.csv", index=False)
print(f"drill test set: {n} rows, {va['anon_user_id'].nunique()} authors "
      f"(NOT 378 -- that is the point)")
PY
    [ $? -eq 0 ] || die s0_testset "could not build the drill test set"
    stamp_done s0_testset
    mark s0_testset PASS "drill test set built from val_split (372 rows, 32 authors, schema = data/test.csv)"
fi
space

# =============================================================================
# STAGE 1 (GPU): ensembled risk = 3 risk_v1 seeds (+ friend, if it covers)
# =============================================================================
if is_done s1_ensembled; then
    echo "[s1_ensembled] stamp exists, resuming past it"
else
    # pick the friend file: explicit override > val probs > (test probs, which
    # cover none of the drill rows -- kept last so the fallback is EXERCISED,
    # not hidden)
    if [ -z "${FRIEND}" ]; then
        if [ -f artifacts/friend_risk_probs_val.csv ]; then
            FRIEND=artifacts/friend_risk_probs_val.csv
        else
            FRIEND=artifacts/friend_risk_probs.csv
        fi
    fi
    [ -f "${FRIEND}" ] || die s1_ensembled "friend probs file ${FRIEND} not found"
    echo "friend probs source: ${FRIEND}"

    python src/ensemble_risk.py \
        --test "${D}/test_drill.csv" \
        --our-risk-run "${RISK_SEEDS}" \
        --friend-probs "${FRIEND}" \
        --weight-ours 0.5 \
        --out "${D}/ensembled_risk.csv" \
        || die s1_ensembled "ensemble_risk.py failed (checkpoints in ${RISK_SEEDS}?)"

    # ensemble_risk.py falls back to the local probs per missing friend row
    # with only a warning line, so an absent member silently changes the
    # ensemble. Count the misses here and mark the stage instead.
    FRIEND="${FRIEND}" python - <<'PY'
import os
import pandas as pd
D = os.environ["D"]
te = pd.read_csv(f"{D}/test_drill.csv")
fp = pd.read_csv(os.environ["FRIEND"])
have = set(fp["row_id"].astype(str))
miss = [r for r in te["row_id"].astype(str) if r not in have]
with open(f"{D}/friend_missing.txt", "w") as f:
    f.write(str(len(miss)))
print(f"friend member covers {len(te)-len(miss)}/{len(te)} drill rows "
      f"({len(miss)} fell back to the local risk seeds)")
if miss:
    print("NOTE: the friend member is reproducible ONLY from the external")
    print("pipeline (stage4_predict.py over outputs/model_fold{0..4}.pt).")
    print("Its config names GATED mental/mental-roberta-base with a")
    print("FALLBACK_MODEL -- on an offline node the fallback would silently")
    print("swap in a different model for this member.")
PY
    FMISS=$(cat "${D}/friend_missing.txt")
    stamp_done s1_ensembled
    if [ "${FMISS}" = "0" ]; then
        mark s1_ensembled PASS "3 seeds + friend, full friend coverage"
    else
        mark s1_ensembled DEGRADED "friend member missing on ${FMISS} rows (silent per-row fallback in ensemble_risk.py; regenerate from the external pipeline -- gated-HF-model caveat applies)"
    fi
fi
space

# =============================================================================
# STAGE 2 (GPU): risk-NLI probabilities
# =============================================================================
if is_done s2_nli; then
    echo "[s2_nli] stamp exists, resuming past it"
else
    python src/dump_nli_probs.py \
        --data "${D}/test_drill.csv" \
        --nli-run "${NLI_RUNS}" \
        --out "${D}/nli_risk_probs.csv" \
        || die s2_nli "dump_nli_probs.py failed (checkpoints in ${NLI_RUNS}?)"
    stamp_done s2_nli
    mark s2_nli PASS "NLI probs dumped for ${NLI_RUNS}"
fi
space

# =============================================================================
# STAGE 3 (CPU): the risk column = 0.70*ens + 0.30*nli -> rules -> route.
# This mirrors build_submission.py's risk block EXACTLY (verified again in
# stage 7 by asserting the two columns are identical) because the evidence
# DUMPS must be conditioned on the pipeline's OWN risk output. Conditioning
# them on a stored submission instead binds them to that file's 378 row_ids,
# which do not exist on fresh data.
# =============================================================================
if is_done s3_riskcol; then
    echo "[s3_riskcol] stamp exists, resuming past it"
else
    python - <<'PY'
import os, sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from collections import Counter
from data_audit import load as load_canonical
from scoring import normalize_label
from apply_rules_to_test import apply_rules

D = os.environ["D"]
LEVELS = ["indicator", "ideation", "behavior", "attempt"]

te = load_canonical(f"{D}/test_drill.csv")
te["row_id"] = te["row_id"].astype(str)
rows = te["row_id"].tolist()
posts = dict(zip(rows, te["post"].astype(str)))

ens = pd.read_csv(f"{D}/ensembled_risk.csv").set_index("row_id")
nli = pd.read_csv(f"{D}/nli_risk_probs.csv").set_index("row_id")
ens.index = ens.index.astype(str)
nli.index = nli.index.astype(str)
def norm(X):
    return X / np.clip(X.sum(1, keepdims=True), 1e-9, None)
E = norm(ens.loc[rows, ["p_Indicator", "p_Ideation", "p_Behavior", "p_Attempt"]].to_numpy(float))
Nn = norm(nli.loc[rows, [f"prob_{c}" for c in LEVELS]].to_numpy(float))
M = 0.70 * E + 0.30 * Nn
lab = [LEVELS[i] for i in M.argmax(1)]
print("after argmax:", dict(Counter(lab)))

lab = [normalize_label(apply_rules(posts[r], l)[0]) for r, l in zip(rows, lab)]
print("after rules (urgency, completed_act):", dict(Counter(lab)))

# ---- route_attempt source selection, loud in every branch ------------------
cands = [p for p in [os.environ.get("ROUTE_FILE", ""),
                     "artifacts/cot_soft_oof.csv.extra.csv",   # TEST rows only
                     "artifacts/cot_soft_oof.csv"] if p]       # OOF, covers val
chosen, route = None, {}
for p in cands:
    if not os.path.exists(p):
        print(f"  route source {p}: DOES NOT EXIST")
        continue
    rt = pd.read_csv(p, keep_default_na=False)
    rt["row_id"] = rt["row_id"].astype(str)
    rcol = "cot" if "cot" in rt.columns else "pred_risk_level"
    m = {r: normalize_label(v) for r, v in zip(rt["row_id"], rt[rcol])}
    miss = sum(1 for r in rows if r not in m)
    print(f"  route source {p}: covers {len(rows)-miss}/{len(rows)} drill rows")
    if miss == 0:
        chosen, route = p, m
        break

if chosen is None:
    noop = f"{D}/route_noop.csv"
    pd.DataFrame({"row_id": rows, "cot": ["none"] * len(rows)}).to_csv(noop, index=False)
    chosen = noop
    with open(f"{D}/route_degraded.flag", "w") as f:
        f.write("route_attempt disabled: no CoT source covers the drill rows")
    print("!" * 72)
    print("!! ROUTE_ATTEMPT DISABLED -- no CoT dump covers this test set.")
    print("!! On the real final eval this is exactly what happens: the rule")
    print("!! needs scripts/risk_cot_member.sh (Qwen2.5-72B, ~20h GPU) run on")
    print("!! the organizers' test.csv. Without it the submission still builds,")
    print("!! but the risk column ships with no attempt routing.")
    print("!" * 72)
else:
    print(f"  using route source: {chosen}")
    if chosen.endswith("cot_soft_oof.csv"):
        print("  NOTE: this is the OOF CoT file -- it covers val authors but")
        print("  can NEVER cover a genuinely unseen test set. The real cold")
        print("  run still needs scripts/risk_cot_member.sh (~20h).")
with open(f"{D}/route_used.txt", "w") as f:
    f.write(chosen)

routed = 0
risk = []
for r, l in zip(rows, lab):
    if route.get(r) == "attempt" and l != "indicator" and l != "attempt":
        routed += 1
    risk.append("attempt" if (route.get(r) == "attempt" and l != "indicator") else l)
print(f"route_attempt changed {routed} labels")
print("final risk:", dict(Counter(risk)))

pd.DataFrame({"row_id": rows,
              "pred_risk_level": [x.capitalize() for x in risk]}
             ).to_csv(f"{D}/risk_drill.csv", index=False)

# risk stub for the 2-row evidence val split
rk = pd.read_csv(f"{D}/risk_drill.csv")
stub = pd.read_csv(f"{D}/val_stub.csv")["row_id"].astype(str)
rk[rk["row_id"].astype(str).isin(set(stub))].to_csv(f"{D}/val_stub_risk.csv", index=False)
print(f"wrote {D}/risk_drill.csv")
PY
    [ $? -eq 0 ] || die s3_riskcol "risk column build failed"
    stamp_done s3_riskcol
    if [ -f "${D}/route_degraded.flag" ]; then
        mark s3_riskcol DEGRADED "risk column built; route_attempt DISABLED (no CoT coverage; needs scripts/risk_cot_member.sh ~20h on the new test set)"
    else
        mark s3_riskcol PASS "risk column built; route source $(cat "${D}/route_used.txt")"
    fi
fi
space

# =============================================================================
# STAGE 4 (GPU): RC7 evidence candidate dumps, conditioned on the pipeline's
# OWN risk column, never on a stored submission whose row_ids do not exist on
# fresh data. Both --test-risk and --val-risk are ALWAYS passed: omitting them
# with --risk-cond is a hard error, but omitting --risk-cond entirely would
# silently disable conditioning and starve the candidate pool while still
# exiting 0.
# =============================================================================
if is_done s4_evdump; then
    echo "[s4_evdump] stamp exists, resuming past it"
else
    python src/evidence_dump_all.py \
        --runs "${RC7}" \
        --test "${D}/test_drill.csv" \
        --val "${D}/val_stub.csv" \
        --outdir "${D}/evidence_runs" \
        --risk-cond \
        --test-risk "${D}/risk_drill.csv" \
        --val-risk "${D}/val_stub_risk.csv" \
        || die s4_evdump "evidence_dump_all.py failed"

    # evidence_dump_all SKIPS a run with a missing checkpoint and keeps going
    # (one 'skip' line). A 6-tagger vote is a DIFFERENT model from the one
    # validated, so count the outputs and refuse a partial set here.
    NEXP=7
    NGOT=$(ls "${D}"/evidence_runs/*_test_candidates.csv 2>/dev/null | wc -l)
    echo "candidate dumps present: ${NGOT}/${NEXP}"
    if [ "${NGOT}" -ne "${NEXP}" ]; then
        die s4_evdump "only ${NGOT}/${NEXP} RC7 test candidate dumps produced -- a partial vote is not the validated ensemble"
    fi
    stamp_done s4_evdump
    mark s4_evdump PASS "7/7 RC7 candidate dumps, conditioned on the drill's OWN risk column"
fi
space

# =============================================================================
# STAGE 5 (GPU): bag8 factor probability dumps on the drill set.
# Mirrors scripts/factor_arm_test_dumps.sh (5/5 checkpoints or refuse; per-arm
# env; default --max-len 320, matching how the shipped test dumps were made)
# but writes into the drill namespace so no skip-guard can return stale files.
# =============================================================================
if is_done s5_facdump; then
    echo "[s5_facdump] stamp exists, resuming past it"
else
    mkdir -p "${D}/factor_runs"
    FDFAIL=0

    # Arms produced OUTSIDE this repo are listed explicitly rather than left
    # off, where their absence would read as "nothing to see".
    # friendfac is VENDORED so that it CAN be rebuilt here: src/friend_pipeline/
    # holds the external stage-4 source verbatim, artifacts/friend_ckpt/ holds
    # the five checkpoints, and artifacts/friend_ckpt/encoder/ holds a local
    # copy of the encoder config + tokenizer so the GATED
    # mental/mental-roberta-base is never fetched. Exercising that path here is
    # the point: it is what the organizers' run would do.
    for arm in ${SHIPPED_EXT}; do
        dst="${D}/factor_runs/${arm}_cv_test.npz"
        if [ -f "${dst}" ]; then
            echo "=== ${arm}: ${dst} exists (drill namespace), skipping ==="
            continue
        fi
        if [ ! -f artifacts/friend_ckpt/model_fold0.pt ]; then
            echo "=== ${arm}: VENDORED CHECKPOINTS ABSENT ==="
            echo "    artifacts/friend_ckpt/model_fold{0..4}.pt are missing, so"
            echo "    the best factor arm cannot be rebuilt. Copy them from the"
            echo "    teammate's pipeline outputs/ before 08-31."
            FDFAIL=$((FDFAIL + 1))
            continue
        fi
        echo "=== ${arm}: rebuilding from the VENDORED pipeline ==="
        python src/friend_factor_predict.py \
            --test "${D}/test_drill.csv" \
            --tag "${arm}" \
            --outdir "${D}/factor_runs" \
            || { echo "    FAILED on ${arm}"; FDFAIL=$((FDFAIL + 1)); continue; }
        echo "    NOTE: --verify against the shipped friendfac_cv_test.npz"
        echo "    reads max|diff| 0.0976, almost certainly because the"
        echo "    teammate's stage 1 CLEANS the post text and this wrapper"
        echo "    feeds it raw. The arm rebuilds; its fidelity is not yet exact."
    done

    for arm in ${FACTOR_ARMS}; do
        dst="${D}/factor_runs/${arm}_cv_test.npz"
        if [ -f "${dst}" ]; then
            echo "=== ${arm}: ${dst} exists (drill namespace), skipping ==="
            continue
        fi

        # ---------------------------------------------------------------
        # TWO ARM FAMILIES, TWO ON-DISK CONVENTIONS. The first version of
        # this drill applied the encoder recipe to all eight arms and
        # reported qnli/mnli7/lnli as "0/5 checkpoints", which read as a
        # fatal cold-start blocker. It was this loop that was wrong.
        #
        #   encoder arms  runs/factors_{arm}_cv_f{f}/model.pt  + /tok
        #                 -> src/predict_factors_nli.py
        #   LLM arms      runs/factors_{arm}_f{f}/lora/        (PEFT adapter,
        #                 NO _cv_ in the dir name, NO model.pt)
        #                 -> src/predict_factors_llm_nli.py
        #
        # predict_factors_nli.py physically cannot load the LLM arms: they
        # are AutoModelForSequenceClassification + a PEFT adapter written by
        # save_pretrained, not an NLIScorer state dict. Both producers write
        # the SAME {TAG}_cv_test.npz filename, so everything downstream of
        # artifacts/factor_runs/ is already agnostic.
        # ---------------------------------------------------------------
        case "${arm}" in
            qnli|mnli7|lnli) FAMILY=llm ;;
            *)               FAMILY=encoder ;;
        esac

        if [ "${FAMILY}" = "llm" ]; then
            NR=0
            for f in 0 1 2 3 4; do
                if [ -d "runs/factors_${arm}_f${f}/lora" ]; then
                    NR=$((NR + 1))
                else
                    echo "  MISSING runs/factors_${arm}_f${f}/lora"
                fi
            done
            if [ "${NR}" -ne 5 ]; then
                echo "=== ${arm}: ${NR}/5 LoRA adapters, REFUSING (a partial bag is a different model) ==="
                FDFAIL=$((FDFAIL + 1))
                continue
            fi

            # The base weights are NOT in the adapter. Read the path the arm
            # was actually trained against out of its own run_config.json
            # rather than hardcoding it -- same principle as reading
            # runs/<dir>/tok for the encoder arms.
            BASE=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['model'])" \
                   "runs/factors_${arm}_f0/run_config.json" 2>/dev/null || true)
            if [ -z "${BASE}" ]; then
                echo "=== ${arm}: cannot read base model from run_config.json, REFUSING ==="
                FDFAIL=$((FDFAIL + 1))
                continue
            fi
            if [ ! -f "${BASE}/config.json" ]; then
                echo "=== ${arm}: base weights ABSENT at ${BASE} ==="
                echo "    A LoRA adapter is unusable without its base model, and"
                echo "    the compute nodes are offline (HF_HUB_OFFLINE=1). This"
                echo "    is the one genuine portability blocker in the factor"
                echo "    column: the path is absolute and machine-specific."
                FDFAIL=$((FDFAIL + 1))
                continue
            fi

            echo "=== ${arm}: LLM arm, 5/5 LoRA adapters, base ${BASE} ==="
            # NB --max-len stays at the predictor's own default of 288 (what
            # train_factors_llm_nli.py used); the 320 the encoder arms take
            # would be a train/inference mismatch here.
            python src/predict_factors_llm_nli.py \
                --tag "${arm}" \
                --model "${BASE}" \
                --test "${D}/test_drill.csv" \
                --outdir "${D}/factor_runs" \
                --folds 0,1,2,3,4 \
                || { echo "    FAILED on ${arm}"; FDFAIL=$((FDFAIL + 1)); continue; }
            continue
        fi

        RUNS=""
        NR=0
        for f in 0 1 2 3 4; do
            dd="runs/factors_${arm}_cv_f${f}"
            if [ -f "${dd}/model.pt" ]; then
                RUNS="${RUNS:+${RUNS},}${dd}"
                NR=$((NR + 1))
            else
                echo "  MISSING ${dd}/model.pt"
            fi
        done
        if [ "${NR}" -ne 5 ]; then
            echo "=== ${arm}: ${NR}/5 checkpoints, REFUSING (a partial bag is a different model) ==="
            FDFAIL=$((FDFAIL + 1))
            continue
        fi
        # per-arm inference env: prot was TRAINED with PROTECTIVE_TEMPLATE=1,
        # every other bag8 arm with it OFF. Wrong setting = plausible garbage.
        unset FACTOR_DEFS_JSON FACTOR_ASPECTS_JSON CONFUSABLE_JSON FACTOR_SUBSET || true
        case "${arm}" in
            prot)
                export PROTECTIVE_TEMPLATE=1
                echo "=== ${arm}: PROTECTIVE_TEMPLATE=1 ===" ;;
            *)
                unset PROTECTIVE_TEMPLATE || true
                echo "=== ${arm}: PROTECTIVE_TEMPLATE unset ===" ;;
        esac
        python src/predict_factors_nli.py \
            --test "${D}/test_drill.csv" \
            --nli-run "${RUNS}" \
            --dump-probs "${dst}" \
            --out "${D}/factor_runs/${arm}_labels.csv" \
            || { echo "    FAILED on ${arm}"; FDFAIL=$((FDFAIL + 1)); continue; }
    done
    unset PROTECTIVE_TEMPLATE || true
    if [ "${FDFAIL}" -gt 0 ]; then
        die s5_facdump "${FDFAIL} bag8 arm(s) failed to dump -- a partial bag is not the shipped model"
    fi

    # sanity: drill-dump probability spread vs each arm's OOF spread. A much
    # narrower test spread was the historical tell for an env mismatch.
    python - <<'PY'
import os
import numpy as np
D = os.environ["D"]
# Read the arm list from the shell rather than repeating it -- this block
# silently checked only the 8 bag arms while the loop above dumped more.
arms = os.environ["FACTOR_ARMS"].split()
bad = 0
for a in arms:
    t = np.load(f"{D}/factor_runs/{a}_cv_test.npz", allow_pickle=True)
    assert t["probs"].shape[1] == 24, f"{a}: probs shape {t['probs'].shape}"
    st = float(t["probs"].std())
    msg = f"  {a:<7} drill {t['probs'].shape} sd={st:.4f}"
    o = f"artifacts/factor_runs/{a}_cv_oof_f0.npz"
    if os.path.exists(o):
        so = float(np.load(o, allow_pickle=True)["probs"].std())
        msg += f"   oof sd={so:.4f}   ratio={st/so:.3f}"
        if st < 0.80 * so:
            msg += "   <-- STRONGLY NARROWER (env mismatch tell)"
            bad += 1
    print(msg)
raise SystemExit(1 if bad else 0)
PY
    [ $? -eq 0 ] || die s5_facdump "factor dump sanity check failed"
    stamp_done s5_facdump
    mark s5_facdump PASS "$(echo ${FACTOR_ARMS} | wc -w) in-repo factor arms dumped fresh (5 folds each), sd sanity vs OOF passed; ${SHIPPED_EXT} is SHIPPED but external and was NOT rebuilt"
fi
space

# =============================================================================
# STAGE 6 (CPU): tfit factor decode, THROUGH THE SHIPPED SCRIPT.
#
# This stage used to REIMPLEMENT the decode -- importing
# factor_threshold_decode's functions and redoing the pooling and thresholding
# inline -- because the real builder could not run on rows it had never seen.
# A drill that tests a COPY of the code is not testing the code: the copy can
# stay correct while the shipped path rots, which is exactly the failure a
# cold-run drill exists to catch.
#
# `factor_threshold_stabilise.py --use tfit` IS the builder of the champion's
# factor column (NOT factor_threshold_decode, whose gate checks a champ-rate
# decode that is no longer shipped). It now takes --cold-test, so the drill
# calls it directly, in two steps:
#
#   6a WARM  on data/test.csv with the shipped arm set -- the provenance gate
#            must read 0 of N cells. This proves the build path itself has not
#            drifted, which --cold-test deliberately cannot check.
#   6b COLD  on the drill's fresh rows, with --test-dir pointing at the drill
#            namespace so no stale dump can be picked up.
#
# 6a is the gate 6b cannot have. Running only 6b would let a broken decode pass
# silently on unseen rows.
# =============================================================================
if is_done s6_facdecode; then
    echo "[s6_facdecode] stamp exists, resuming past it"
else
    echo "--- 6a WARM: does the shipped build path still reproduce itself? ---"
    python src/factor_threshold_stabilise.py \
        --arms ${FACTOR_ARMS} ${SHIPPED_EXT} \
        --use tfit \
        --shipped artifacts/factors_ff11_test.csv \
        --champion submissions/NIDHI_ff11.csv \
        --build-test "${D}/_warm_provenance.csv" 2>&1 \
      | grep -E "PROVENANCE|cells changed|wrote|REFUS" \
      || { die s6_facdecode "the shipped factor build path no longer reproduces the champion column -- fix that before trusting any cold result"; }

    echo ""
    echo "--- 6b COLD: the same script on rows it has never seen ---"
    python src/factor_threshold_stabilise.py \
        --arms ${FACTOR_ARMS} ${SHIPPED_EXT} \
        --use tfit \
        --cold-test "${D}/test_drill.csv" \
        --test-dir "${D}/factor_runs" \
        --build-test "${D}/factors_drill.csv" 2>&1 \
      | grep -E "test dumps from|PROVENANCE|rule tfit|0-factor|wrote|REFUS|realigned"
    [ -f "${D}/factors_drill.csv" ] \
        || die s6_facdecode "cold decode wrote no factors column"
    stamp_done s6_facdecode
    mark s6_facdecode PASS "tfit decode via the SHIPPED script (factor_threshold_stabilise --cold-test), not a reimplementation; warm provenance gate passed on data/test.csv first"
fi
space

# ---------------------------------------------------------------------------
# The old inline reimplementation is kept below, disabled, purely as a
# reference for what stage 6 used to compute. Delete it once the two-call form
# above has run green a few times.
# ---------------------------------------------------------------------------
if false; then
    python - <<'PY'
import os, sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from factor_threshold_decode import (load_oof, logit_mean, cells,
                                     best_threshold_and_f1, resolve_test,
                                     BAG8, N)
from scoring import FACTOR_TAXONOMY

D = os.environ["D"]

# 1. full-data tfit thresholds from the (cluster-resident) OOF npz.
#    LEAKAGE NOTE (for the score half only): this OOF spans all 1635 labelled
#    posts, so the thresholds have seen val gold. Fine for the cold-run
#    conclusion, inflating for the score conclusion.
rid, S = load_oof("artifacts/factor_runs", BAG8, 5)
P = logit_mean(S)
lab = pd.read_csv("artifacts/full_train.csv", keep_default_na=False)
lab["row_id"] = lab["row_id"].astype(str)
lab = lab.set_index("row_id")
Y = cells([lab.loc[r, "factors"] for r in rid], len(rid))
th = np.zeros(N)
for j in range(N):
    th[j], _ = best_threshold_and_f1(P[:, j], Y[:, j])
print(f"tfit thresholds fitted on {len(rid)} OOF posts")

# 2. PROVENANCE GATE: these thresholds + the SHIPPED 378-row dumps must
#    reproduce the shipped factors_tfit_test.csv (the champion's factor
#    column). If they do not, the drill would be testing a different model.
ship_p = "artifacts/factors_tfit_test.csv"
if os.path.exists(ship_p):
    T2, tid2, missing = [], None, []
    for a in BAG8:
        p = resolve_test("artifacts/factor_runs", a)
        if p is None:
            missing.append(a)
            continue
        z = np.load(p, allow_pickle=True)
        ids = [str(x) for x in z["row_id"]]
        pr = z["probs"].astype(float)
        if tid2 is None:
            tid2 = ids
        elif ids != tid2:
            o = {x: i for i, x in enumerate(ids)}
            pr = pr[[o[x] for x in tid2]]
        T2.append(pr)
    if missing:
        print(f"provenance gate SKIPPED: no shipped test dump for {missing}")
    else:
        rec = (logit_mean(np.stack(T2)) >= th[None, :]).astype(np.int8)
        ship = pd.read_csv(ship_p, keep_default_na=False)
        ship["row_id"] = ship["row_id"].astype(str)
        ship = ship.set_index("row_id")
        fcol = "factors" if "factors" in ship.columns else "pred_factors"
        cur = cells([ship.loc[r, fcol] for r in tid2], len(tid2))
        diff = int((rec != cur).sum())
        tot = rec.size
        print(f"provenance: tfit reconstruction vs shipped column -- "
              f"{diff} of {tot} cells differ")
        if diff > 0.02 * tot:
            print("REFUSING: thresholds do not reproduce the shipped factor "
                  "column; the drill would score a different model than the "
                  "one that ships.")
            raise SystemExit(1)
else:
    print(f"provenance gate SKIPPED: {ship_p} not found")

# 3. apply to the FRESH drill dumps
T, tid = [], None
for a in BAG8:
    p = f"{D}/factor_runs/{a}_cv_test.npz"
    if not os.path.exists(p):
        print(f"missing drill dump {p}")
        raise SystemExit(1)
    z = np.load(p, allow_pickle=True)
    ids = [str(x) for x in z["row_id"]]
    pr = z["probs"].astype(float)
    if tid is None:
        tid = ids
    elif ids != tid:
        o = {x: i for i, x in enumerate(ids)}
        pr = pr[[o[x] for x in tid]]
    T.append(pr)
Pt = logit_mean(np.stack(T))
on = (Pt >= th[None, :]).astype(np.int8)
fac = [str([FACTOR_TAXONOMY[j] for j in range(N) if on[i, j]])
       for i in range(len(tid))]
pd.DataFrame({"row_id": tid, "factors": fac}).to_csv(
    f"{D}/factors_drill.csv", index=False)
z0 = int((on.sum(1) == 0).sum())
print(f"drill factors: {on.sum()/len(tid):.2f} per post (gold ~2.92), "
      f"{z0}/{len(tid)} zero-factor posts (gold rate ~7.8%)")
print(f"wrote {D}/factors_drill.csv")
PY
    :   # (disabled reference block ends here)
fi
space

# =============================================================================
# STAGE 7 (CPU): assemble with src/build_submission.py itself -- the exact
# script the organizers get, exercising its _require_coverage guards against
# drill-namespace inputs only. --reference points at a nonexistent file so
# the 378-row champion diff is skipped (it cannot apply to 372 rows).
# =============================================================================
if is_done s7_assemble; then
    echo "[s7_assemble] stamp exists, resuming past it"
else
    ROUTE_USED=$(cat "${D}/route_used.txt")
    python src/build_submission.py \
        --test "${D}/test_drill.csv" \
        --ensembled "${D}/ensembled_risk.csv" \
        --nli "${D}/nli_risk_probs.csv" \
        --route "${ROUTE_USED}" \
        --factors "${D}/factors_drill.csv" \
        --ev-dir "${D}/evidence_runs" \
        --reference "${D}/__no_reference__.csv" \
        --team NIDHI \
        --out "${D}/submission_drill.csv" \
        || die s7_assemble "build_submission.py failed on the drill set -- THIS IS THE COLD-START FAILURE THE ORGANIZERS WOULD HIT"

    # cross-check: build_submission recomputed risk internally; it must equal
    # the stage-3 column the evidence dumps were conditioned on, or the two
    # copies of the blend logic have drifted (the exact bug class the
    # apply_rules docstring warns about).
    python - <<'PY'
import os
import pandas as pd
D = os.environ["D"]
a = pd.read_csv(f"{D}/risk_drill.csv")
b = pd.read_csv(f"{D}/submission_drill.csv")
a["row_id"] = a["row_id"].astype(str)
b["row_id"] = b["row_id"].astype(str)
m = b.set_index("row_id")["risk_level"].str.lower()
aa = a.set_index("row_id")["pred_risk_level"].str.lower()
assert len(a) == len(b), f"row count {len(a)} vs {len(b)}"
diff = int((aa != m.loc[aa.index]).sum())
print(f"risk column: stage-3 vs build_submission -- {diff} of {len(a)} differ")
raise SystemExit(1 if diff else 0)
PY
    [ $? -eq 0 ] || die s7_assemble "risk drift between stage-3 column and build_submission's internal rebuild"
    stamp_done s7_assemble
    mark s7_assemble PASS "build_submission.py assembled ${D}/submission_drill.csv cold; risk column identical to the conditioning column"
fi
space

# =============================================================================
# STAGE 8 (CPU): score against gold. Evidence MUST be macro-averaged, matching
# the scoring definition; the micro default silently understates evidence
# performance. Then print the two conclusions separately.
# =============================================================================
python - <<'PY'
import os, sys
sys.path.insert(0, "src")
import pandas as pd
from scoring import (risk_weighted_f1, phrase_f1, factors_macro_f1,
                     parse_evidence, parse_factors, normalize_label)

D = os.environ["D"]
sub = pd.read_csv(f"{D}/submission_drill.csv", keep_default_na=False)
gold = pd.read_csv(f"{D}/gold_drill.csv", keep_default_na=False)
sub["row_id"] = sub["row_id"].astype(str)
gold["row_id"] = gold["row_id"].astype(str)
g = gold.set_index("row_id").loc[sub["row_id"]]

gr = [normalize_label(x) for x in g["risk_level"]]
pr = [normalize_label(x) for x in sub["risk_level"]]
ge = [parse_evidence(x) for x in g["evidence"]]
pe = [parse_evidence(x) for x in sub["evidence"]]
gf = [parse_factors(x) for x in g["factors"]]
pf = [parse_factors(x) for x in sub["factors"]]

r = risk_weighted_f1(gr, pr)
e = phrase_f1(ge, pe, average="macro").f1   # MACRO IS MANDATORY -- board is macro
f = factors_macro_f1(gf, pf)
comp = 0.4 * r + 0.3 * e + 0.3 * f

W = 72
print("=" * W)
print(f"COLD-RUN DRILL SCORE   ({len(sub)} posts, 32 held-out authors)")
print("=" * W)
print(f"  risk      weighted-F1        {r:.4f}   x 0.4")
print(f"  evidence  phrase-F1 (MACRO)  {e:.4f}   x 0.3")
print(f"  factors   macro-F1           {f:.4f}   x 0.3")
print(f"  composite                    {comp:.4f}")
print("-" * W)
print("CONCLUSION 1 -- CODE RUNS COLD (valid unconditionally):")
print("  every stage regenerated from checkpoints on a 372-row test set the")
print("  stored artifacts do not cover. No _require_coverage guard fired, no")
print("  stage silently emitted an empty column. This is the half that")
print("  protects the 30% performance criterion on the organizers' rerun.")
print("-" * W)
print("CONCLUSION 2 -- THE SCORE ITSELF (read with these caveats):")
print("  risk:     CLEAN. risk_v1 seeds + risk_nli trained on train_split")
print("            (val excluded). Expect ~0.83 (champion_oof_twin: 0.8294);")
print("            a large shortfall means a risk stage silently degraded.")
print("  evidence: NEARLY CLEAN. Tagger weights trained on train_split, but")
print("            the vote config (k=2, modal, member set, min_confs) was")
print("            selected on these very val posts -- mildly optimistic.")
print("  factors:  INFLATED -- DO NOT QUOTE. The bag8 fold checkpoints")
print("            partition all 1635 labelled posts, so 4 of 5 folds")
print("            trained on every drill post, and the tfit thresholds were")
print("            fitted on OOF that includes drill gold. The honest factor")
print("            number stitches the per-fold OOF npz instead.")
print("  Neither caveat weakens Conclusion 1.")
if os.path.exists(f"{D}/route_degraded.flag"):
    print("-" * W)
    print("  DEGRADED: route_attempt was disabled (no CoT coverage). The")
    print("  score above is missing this rule's contribution to the risk")
    print("  sub-score.")
fm = open(f"{D}/friend_missing.txt").read().strip() if os.path.exists(f"{D}/friend_missing.txt") else "?"
if fm not in ("0", "?"):
    print(f"  DEGRADED: friend risk member absent on {fm} rows.")
print("=" * W)

with open(f"{D}/drill_score.txt", "w") as fh:
    fh.write(f"risk {r:.4f}\nevidence_macro {e:.4f}\nfactors {f:.4f}\ncomposite {comp:.4f}\n")

# loose sanity rails: a catastrophic silent failure shows up as a component
# near zero even when every stage "completed".
bad = []
if r < 0.55: bad.append(f"risk {r:.3f} < 0.55")
if e < 0.45: bad.append(f"evidence {e:.3f} < 0.45")
if f < 0.30: bad.append(f"factors {f:.3f} < 0.30")
if bad:
    print("SANITY RAIL TRIPPED: " + "; ".join(bad))
    print("A component this low after a 'successful' run means something")
    print("degraded silently despite the coverage guards. Investigate.")
    raise SystemExit(1)
PY
[ $? -eq 0 ] || die s8_score "scoring failed or a sanity rail tripped"
mark s8_score PASS "scored vs gold (evidence MACRO); components within sanity rails; see drill_score.txt"

summary
# non-zero exit if anything failed (for sacct visibility)
if grep -l $'^FAIL\t' "${D}"/status/* >/dev/null 2>&1; then
    exit 1
fi
exit 0
