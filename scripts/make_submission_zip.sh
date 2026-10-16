#!/bin/bash
# Build NIDHI.zip for the IEEE BigData Cup 2026 final submission.
#
# REQUIREMENTS set by the organizers:
#   * ONE .zip named after the team          -> NIDHI.zip
#   * source code AND the report inside it
#   * a README with instructions to run on Linux or Windows
#   * report in the IEEE conference template
#   * deadline 2026-08-31 23:59:59 AoE, via the Google Form
#   * anonymity NOT required -- names and affiliations may be included
#
# WHAT IS DELIBERATELY EXCLUDED, and why:
#   data/         the organizers' own clinical Reddit corpus. The organizers
#                 already hold it, and crisis posts must not be redistributed.
#   runs/         134 GB of checkpoints. Impossible to ship; the README says
#                 how to retrain, and which artifacts a rerun needs.
#   artifacts/    model outputs derived from the data (probabilities, spans).
#                 Same reasoning, plus size.
#   logs/         job output containing post-level diagnostics.
#   .git/ .venv/ __pycache__/ .pytest_cache/
#
# WHAT IS INCLUDED: src/, scripts/, tests/, README.md, requirements.txt,
# LICENSE, and the final submission CSV, so a grader can see exactly which
# predictions the reported results describe.
#
# Usage:  bash scripts/make_submission_zip.sh [path/to/report.pdf]

set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TEAM="NIDHI"
STAGE="$(mktemp -d)"
OUT="$ROOT/${TEAM}.zip"
REPORT="${1:-}"
# The final submission CSV. It MUST be the same file as build_submission.py's
# --reference default: if the reference is absent from the package, `--verify`
# has nothing to compare against, skips silently and prints a false pass.
CHAMPION="submissions/NIDHI_scaled.csv"

echo "staging in ${STAGE}"
mkdir -p "${STAGE}/${TEAM}"

for d in src scripts tests; do
    [ -d "$d" ] || { echo "ERROR: $d/ missing"; exit 1; }
    rsync -a --exclude='__pycache__' --exclude='*.pyc' "$d" "${STAGE}/${TEAM}/"
done

# The report SOURCE, when present. The compiled PDF is passed as $1 and lands
# at the package root; shipping the .tex and .bib alongside it costs a few KB
# and lets a reader see exactly how every number in the report was produced.
# LaTeX build products are excluded -- they are regenerable and noisy.
if [ -d report ]; then
    rsync -a --exclude='*.aux' --exclude='*.log' --exclude='*.out' \
          --exclude='*.bbl' --exclude='*.blg' --exclude='*.synctex.gz' \
          --exclude='*.fdb_latexmk' --exclude='*.fls' \
          report "${STAGE}/${TEAM}/"
    echo "  report source included"
fi
for f in README.md requirements.txt LICENSE; do
    [ -f "$f" ] || { echo "ERROR: $f missing"; exit 1; }
    cp "$f" "${STAGE}/${TEAM}/"
done

# the final predictions file
mkdir -p "${STAGE}/${TEAM}/submissions"
[ -f "$CHAMPION" ] || { echo "ERROR: ${CHAMPION} missing"; exit 1; }
cp "$CHAMPION" "${STAGE}/${TEAM}/submissions/"

# the report
if [ -n "$REPORT" ]; then
    [ -f "$REPORT" ] || { echo "ERROR: report ${REPORT} not found"; exit 1; }
    cp "$REPORT" "${STAGE}/${TEAM}/"
    echo "  report: $(basename "$REPORT")"
else
    echo ""
    echo "  !! NO REPORT PASSED. The organizers require code AND report in"
    echo "  !! the same zip. Re-run as:"
    echo "  !!     bash scripts/make_submission_zip.sh path/to/report.pdf"
    echo ""
fi

# ---- refuse to ship anything derived from the corpus ----------------------
echo "checking for dataset leakage into the package..."
LEAK=0
INC="--include=*.py --include=*.sh --include=*.md --include=*.txt --include=*.json"

# HARD CHECK -- a post ID sitting immediately before a quoted sentence.
#
# A bare row id is harmless: they are the organizers' own identifiers and carry
# no content. An id followed by QUOTED PROSE is different -- that is a labelled
# corpus row reproduced verbatim, which is exactly what must not be
# redistributed. Requiring 20+ characters inside the quote is what separates it
# from an id LIST like "cases": "P00876 P00878 P00879", where the character
# after the final id is the closing quote and no prose follows.
# shellcheck disable=SC2086
HARD=$(grep -rlnE 'P[0-9]{5}[^"]{0,10}"[^"]{20,}"' $INC \
       "${STAGE}/${TEAM}" 2>/dev/null || true)
if [ -n "$HARD" ]; then
    echo "  CORPUS ROW REPRODUCED VERBATIM in:"
    echo "$HARD" | sed 's/^/    /'
    LEAK=$((LEAK+1))
fi

# SOFT CHECK -- first-person crisis prose inside a long quoted string.
#
# This one WARNS instead of refusing, because it cannot distinguish a copied
# post from a hand-written exemplar, and both exist in this tree on purpose
# (src/risk_cot_probe.py and src/evidence_llm_propose.py carry invented
# examples that teach a prompt the output shape). A check that blocks the build
# on legitimate content gets switched off, so this one reports and lets a human
# judge. Regex vocabulary lists are excluded by requiring surrounding prose
# rather than the bare phrase.
# shellcheck disable=SC2086
SOFT=$(grep -rlnE '"[^"]{25,}\b(i tried|i swallowed|i woke up in|i keep hurting|i want to die|i give up)\b[^"]{10,}"' \
       $INC "${STAGE}/${TEAM}" 2>/dev/null \
       | grep -v 'make_submission_zip.sh' || true)
if [ -n "$SOFT" ]; then
    echo ""
    echo "  REVIEW (not blocking): first-person prose in a quoted string here."
    echo "  Confirm each is hand-written and not copied from the corpus:"
    echo "$SOFT" | sed 's/^/    /'
    echo ""
fi
for bad in data artifacts runs logs; do
    if [ -e "${STAGE}/${TEAM}/${bad}" ]; then
        echo "  ERROR: ${bad}/ made it into the package"; LEAK=$((LEAK+1))
    fi
done
CSVN=$(find "${STAGE}/${TEAM}" -name '*.csv' | wc -l)
if [ "$CSVN" -ne 1 ]; then
    echo "  ERROR: expected exactly 1 csv (the submission), found ${CSVN}"
    find "${STAGE}/${TEAM}" -name '*.csv'
    LEAK=$((LEAK+1))
fi
[ "$LEAK" -eq 0 ] || { echo "REFUSING to build the zip"; exit 1; }
echo "  clean"

rm -f "$OUT"
( cd "$STAGE" && zip -qr "$OUT" "$TEAM" )
rm -rf "$STAGE"

echo ""
echo "wrote ${OUT}   $(du -h "$OUT" | cut -f1)"
unzip -l "$OUT" | tail -5
echo ""
echo "FINAL CHECKLIST before the Google Form:"
echo "  [ ] report PDF is inside the zip (IEEE conference template)"
echo "  [ ] README run instructions work from a clean checkout"
echo "  [ ] python -m pytest tests/ -q      passes"
echo "  [ ] python src/build_submission.py --verify   prints FULL MATCH"
echo "  [ ] zip is named NIDHI.zip"
echo "  [ ] submitted via the Google Form, NOT emailed"
echo "  [ ] before 2026-08-31 23:59:59 AoE"
