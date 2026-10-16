#!/bin/bash
#SBATCH --job-name=wsplit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

# ===========================================================================
# w_ours=0.40 SPLIT INTO ITS TWO SEPARABLE HALVES.
#
# Changing the friend's ensemble weight to 0.40 moves 12 risk labels, and those
# 12 changes are not one hypothesis but two. Evidence is suppressed exactly on
# `indicator`, so a label that crosses that boundary also empties or refills an
# evidence cell. A plain risk-only patch would therefore change two columns
# while looking like a one-column change, and the resulting score could not be
# attributed to the weight.
#
#   A (non-crossing)  ideation<->behavior<->attempt moves. Evidence
#                     conditioning shifts but no cell is emptied, so patching
#                     risk alone is genuinely single-variable and reads the
#                     weight direction on its own.
#
#   B (crossing)      ideation -> indicator moves. Their evidence must be
#                     REBUILT, because suppression empties it. Two-sided: if
#                     those posts really are indicator, gold evidence is empty
#                     and empty-vs-empty scores 1.0 under the macro-averaged
#                     evidence metric; if they are not, the same mechanism
#                     costs a full post each.
#
# Running them together confounds a cheap risk test with an expensive evidence
# change, which is why the script builds them as two separate artefacts.
#
# WARNING ON B's SIZING. The crossing posts are CONSECUTIVE row_ids moving
# identically. Independent posts do not do that. If they are near-duplicates or
# one author's thread, they are ONE piece of evidence with several times the
# score leverage, not several independent ones. Stage 0 measures this before B
# is built.
#
# NOTHING IS SUBMITTED HERE.
# ===========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate riskcup2
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
NT="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${NT}" MKL_NUM_THREADS="${NT}"

set -u
CHAMP="${CHAMP:-submissions/NIDHI_ff11.csv}"
ENS="${ENS:-artifacts/ensembled_risk_w040.csv}"
NEWDIR="${NEWDIR:-artifacts/evidence_runs_w040}"
RC7="evidence_ep9_rc,evidence_lr2_rc,evidence_s14_rc,evidence_s15_rc,evidence_s16_rc,evidence_s17_rc,evidence_v1_rc"
mkdir -p "${NEWDIR}" logs

echo "node $(hostname)  $(date)"
for f in "${CHAMP}" "${ENS}" data/test.csv; do
    [ -f "$f" ] || { echo "ERROR: $f missing -- run scripts/risk_friend_weight.sh first"; exit 1; }
done

# ---------------------------------------------------------------------------
# STAGE 1 -- dump the w=0.40 risk labels for all 378 rows.
# --out-risk writes even when posts cross the boundary; that is its purpose.
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "STAGE 1: dump w=0.40 risk labels"
echo "################################################################"
python src/champion_risk_inject.py \
    --champion "${CHAMP}" \
    --alt-ensembled "${ENS}" \
    --out-risk artifacts/risk_w040.csv \
    || { echo "risk dump failed"; exit 1; }

# ---------------------------------------------------------------------------
# STAGE 0 -- are the five crossers really five independent posts?
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "STAGE 0: are P00008..P00012 independent, or one thing five times?"
echo "################################################################"
CHAMP="${CHAMP}" python - <<'PY'
import sys
import pandas as pd
sys.path.insert(0, "src")
from data_audit import load
from scoring import normalize_span, parse_evidence, normalize_label
t = load("data/test.csv")
tc = next(c for c in ("clean_text", "post", "text", "body") if c in t.columns)
t["row_id"] = t["row_id"].astype(str)
ids = ["P00008", "P00009", "P00010", "P00011", "P00012"]
sub = t[t.row_id.isin(ids)]
print(f"found {len(sub)} of 5 in test.csv")
txt = {}
for r in sub.itertuples():
    s = normalize_span(str(getattr(r, tc)))
    txt[r.row_id] = s
    print(f"\n  {r.row_id}  {len(s.split())} tokens")
    print(f"    {s[:200]}")
# pairwise containment / jaccard -- cheap duplicate detector
print("\n  pairwise word-set Jaccard:")
ks = sorted(txt)
dup = 0
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        A, B = set(txt[ks[i]].split()), set(txt[ks[j]].split())
        if not A or not B:
            continue
        jac = len(A & B) / len(A | B)
        flag = ""
        if jac > 0.8:
            flag = "  <-- NEAR-DUPLICATE"
            dup += 1
        print(f"    {ks[i]} vs {ks[j]}  {jac:.3f}{flag}")
print(f"\n  near-duplicate pairs: {dup}")
print("  If >0, B is ONE hypothesis with amplified leverage, not five votes.")
for c in ("author", "author_id", "user", "user_id"):
    if c in t.columns:
        print(f"\n  {c} of the five: {sub[c].tolist()}")
        break
else:
    print("\n  (no author column in test.csv -- cannot check thread identity)")
ch = pd.read_csv("submissions/NIDHI_ff11.csv", keep_default_na=False)
ch["row_id"] = ch["row_id"].astype(str)
print("\n  what the champion currently ships on these five:")
for r in ch[ch.row_id.isin(ids)].itertuples():
    ev = parse_evidence(r.evidence)
    print(f"    {r.row_id}  risk={normalize_label(r.risk_level):<10} "
          f"{len(ev)} span(s)  {str(ev)[:110]}")
print("\n  Those spans are what B DELETES. If the five are truly indicator,")
print("  deleting them is the whole gain. If not, it is the whole loss.")
PY

# ---------------------------------------------------------------------------
# STAGE 2 -- SUBMISSION A: apply w=0.40 but HOLD the boundary crossers.
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "STAGE 2: build A -- w=0.40 with indicator crossers held at champion"
echo "################################################################"
CHAMP="${CHAMP}" python - <<'PY'
import os
import sys
import pandas as pd
sys.path.insert(0, "src")
from scoring import normalize_label
ch = pd.read_csv(os.environ["CHAMP"], keep_default_na=False)
ch["row_id"] = ch["row_id"].astype(str)
nw = pd.read_csv("artifacts/risk_w040.csv")
nw["row_id"] = nw["row_id"].astype(str)
base = dict(zip(ch.row_id, (normalize_label(x) for x in ch.risk_level)))
new = dict(zip(nw.row_id, (normalize_label(x) for x in nw.pred_risk_level)))
held, kept, rev = [], 0, 0
for r in ch.row_id:
    b, n = base[r], new.get(r, base[r])
    if b == n:
        held.append(b)
    elif (b == "indicator") != (n == "indicator"):
        held.append(b)          # HOLD: would change the suppression set
        rev += 1
    else:
        held.append(n)
        kept += 1
pd.DataFrame({"row_id": ch.row_id,
              "pred_risk_level": [x.capitalize() for x in held]}
             ).to_csv("artifacts/risk_w040_held.csv", index=False)
print(f"  kept {kept} non-crossing changes, reverted {rev} crossers")
print(f"  wrote artifacts/risk_w040_held.csv")
assert kept > 0, "nothing left to test after holding the crossers"
PY

python src/patch_submission_risk.py \
    --base "${CHAMP}" \
    --risk artifacts/risk_w040_held.csv \
    --out submissions/NIDHI_w040hold.csv \
    || { echo "patch A failed"; exit 1; }

CHAMP="${CHAMP}" OUT=submissions/NIDHI_w040hold.csv python - <<'PY'
import os
import sys
import pandas as pd
sys.path.insert(0, "src")
from scoring import parse_evidence, normalize_label
a = pd.read_csv(os.environ["CHAMP"], keep_default_na=False)
b = pd.read_csv(os.environ["OUT"], keep_default_na=False)
assert list(a.row_id.astype(str)) == list(b.row_id.astype(str)), "ROW ORDER"
ok = True
for c in ("evidence", "factors"):
    n = int((a[c].astype(str) != b[c].astype(str)).sum())
    print(f"  {c:<11} differs {n}   {'OK' if n == 0 else 'FAIL'}")
    ok &= n == 0
la = [normalize_label(x) for x in a.risk_level]
lb = [normalize_label(x) for x in b.risk_level]
print(f"  risk_level  differs {sum(x != y for x, y in zip(la, lb))}")
cross = sum(1 for x, y in zip(la, lb) if (x == "indicator") != (y == "indicator"))
print(f"  crossing indicator: {cross}   {'OK' if cross == 0 else 'FAIL'}")
ea = sum(1 for x in a.evidence if not parse_evidence(x))
eb = sum(1 for x in b.evidence if not parse_evidence(x))
print(f"  empty evidence {ea} -> {eb}   {'OK' if ea == eb else 'FAIL'}")
print(f"  A is {'SHIPPABLE (single-variable)' if (ok and cross == 0 and ea == eb) else 'BROKEN'}")
PY

# ---------------------------------------------------------------------------
# STAGE 3 -- SUBMISSION B: full w=0.40, evidence rebuilt on the new risk.
# ---------------------------------------------------------------------------
echo ""
echo "################################################################"
echo "STAGE 3: build B -- full w=0.40 with evidence RECONDITIONED"
echo "################################################################"
python src/evidence_dump_all.py \
    --runs "${RC7}" \
    --test data/test.csv \
    --val artifacts/val_split.csv \
    --outdir "${NEWDIR}" \
    --risk-cond \
    --test-risk artifacts/risk_w040.csv \
    --val-risk artifacts/nligaur/oof_base.csv \
    || { echo "REDUMP FAILED -- A is still shippable on its own"; exit 0; }

python src/evidence_vote_test.py \
    --runs "${RC7}" \
    --dir "${NEWDIR}" \
    --split test \
    --test data/test.csv \
    --risk-from artifacts/risk_w040.csv \
    --k 2 --cluster --rep modal \
    --out artifacts/evidence_w040_test.csv \
    || { echo "VOTE FAILED -- A is still shippable"; exit 0; }

python src/patch_submission_risk.py \
    --base "${CHAMP}" \
    --risk artifacts/risk_w040_held.csv \
    --out artifacts/_stage_b_risk_only.csv \
    --max-changes 60 >/dev/null 2>&1 || true

# both columns together: risk from the FULL w040 dump, evidence rebuilt on it
python src/patch_submission_risk.py \
    --base "${CHAMP}" \
    --risk artifacts/risk_w040.csv \
    --evidence artifacts/evidence_w040_test.csv \
    --out submissions/NIDHI_w040full.csv \
    --max-changes 60 \
    || { echo "patch B failed (it may refuse two columns at once --"; \
         echo " if so, apply the risk and evidence patches in sequence)"; exit 0; }

CHAMP="${CHAMP}" OUT=submissions/NIDHI_w040full.csv python - <<'PY'
import os
import sys
import pandas as pd
sys.path.insert(0, "src")
from scoring import parse_evidence, normalize_label
a = pd.read_csv(os.environ["CHAMP"], keep_default_na=False)
b = pd.read_csv(os.environ["OUT"], keep_default_na=False)
assert list(a.row_id.astype(str)) == list(b.row_id.astype(str)), "ROW ORDER"
n = int((a.factors.astype(str) != b.factors.astype(str)).sum())
print(f"  factors     differs {n}   {'OK' if n == 0 else 'FAIL - factors must not move'}")
la = [normalize_label(x) for x in a.risk_level]
lb = [normalize_label(x) for x in b.risk_level]
print(f"  risk_level  differs {sum(x != y for x, y in zip(la, lb))}")
print(f"  evidence    differs {int((a.evidence.astype(str) != b.evidence.astype(str)).sum())}")
ea = sum(1 for x in a.evidence if not parse_evidence(x))
eb = sum(1 for x in b.evidence if not parse_evidence(x))
print(f"  empty evidence {ea} -> {eb}  (expect +5 from the crossers)")
print(f"  spans {sum(len(parse_evidence(x)) for x in a.evidence)} -> "
      f"{sum(len(parse_evidence(x)) for x in b.evidence)}")
PY

cat <<'NEXT'

################################################################
DONE. Nothing submitted.
################################################################
A -- submissions/NIDHI_w040hold.csv
  Non-crossing risk labels only; evidence and factors byte-identical and the
  empty-evidence count unchanged, so the factor sub-score must not move. This
  is the clean single-variable read on the weight direction.

B -- submissions/NIDHI_w040full.csv
  Meaningful only once A has been read, or once stage 0 has shown the crossing
  posts are near-duplicates and therefore one change rather than several. B
  moves risk AND evidence, so it is not single-variable -- but the coupling is
  forced by the suppression rule, not by a second free parameter.

If A does not gain, w=0.5 is an interior optimum and the weight is settled.
NEXT
