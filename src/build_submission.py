"""
build_submission.py

Rebuild all three submission columns from their component artefacts in one
process, and check the result against the shipped submission.

The organisers re-run this pipeline on their own held-out set for part of the
rank, so the chain has to exist as executable code rather than as a sequence of
shell invocations, and any drift in a step has to surface as an error rather
than as an unexplained score.

THE CHAIN:

  RISK   0.70 * artifacts/ensembled_risk.csv          (3 risk_v1 seeds + friend)
       + 0.30 * artifacts/nli_risk_test_probs.csv
       -> argmax
       -> apply_rules(urgency, completed_act)   on CANONICAL text
       -> route_attempt: CoT-72B says attempt AND label != indicator

       The seeds are the FOLD-TRAINED checkpoints, not full-data retrains. The
       blend weight, the NLI mix and the rule set were all selected against
       these checkpoints; swapping in retrained members changes the models the
       weights were chosen for, so the tuned operating point no longer applies.

  EVID   vote k>=2 over SEVEN risk-conditioned taggers, clustered, rep=modal,
       **cap 3 spans per post**, **uniform confidence gate 0.65** replacing
       each run's own evidence_min_conf, suppressed wherever the risk column
       says indicator. The cap and the uniform gate are part of the shipped
       decode; dropping either produces a different, unvalidated evidence
       column. 0.65/k2/cap3 is the global maximum of a 48-cell
       gate x k x cap grid.
       **evidence_v2_rc is deliberately EXCLUDED**: it is the least precise of
       the candidate taggers, and nested author-level selection drops it
       essentially always, so it contributes votes that push the pool past the
       k>=2 threshold on spans the other members do not support.

  FACT   the factors column of --factors as written -- the arm pool combined in
       logit space and decoded with one threshold per label (`tfit`).

WHAT THIS REFUSES TO DO
  * write if any component fails to reproduce its reference column, unless
    --allow-drift is passed. A pipeline that cannot rebuild its own output is
    not a pipeline anybody should run on unseen data.
  * write if the indicator<->empty-evidence correspondence breaks. Gold
    evidence is empty exactly on indicator posts, so suppression must follow
    the risk column it was generated against.

Usage:
  python src/build_submission.py --verify
  python src/build_submission.py --out submissions/NIDHI_rebuild.csv
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_rules_to_test import apply_rules  # noqa: E402
from data_audit import load as load_canonical  # noqa: E402
from scoring import normalize_label, parse_evidence, parse_factors  # noqa

LEVELS = ["indicator", "ideation", "behavior", "attempt"]
ENSCOLS = ["p_Indicator", "p_Ideation", "p_Behavior", "p_Attempt"]
NLICOLS = [f"prob_{c}" for c in LEVELS]
RC7 = ["evidence_ep9_rc", "evidence_lr2_rc", "evidence_s14_rc",
       "evidence_s15_rc", "evidence_s16_rc", "evidence_s17_rc",
       "evidence_v1_rc"]


def norm(X):
    return X / np.clip(X.sum(1, keepdims=True), 1e-9, None)


def _require_coverage(mapping, rows, source, what, remedy):
    """Refuse to build a column from an artifact that does not cover the test.

    THE FAILURE THIS EXISTS TO STOP IS SILENT. Every precomputed artifact this
    script reads was dumped for one specific set of row_ids. Run the pipeline
    against a different test set -- which is exactly what the final evaluation
    does, on authors none of these artifacts were built for -- and every lookup
    misses. The build code reads those maps with `.get(r, "")` / `.get(r)`
    defaults, so a total artifact mismatch becomes an empty factors column, an
    empty evidence column and a route rule that fires zero times: no exception,
    no warning, and a well-formed submission that scores near zero on the two
    components those artifacts feed. Scoring against a set built from the same
    row_ids as the artifacts cannot reveal the problem either.

    A crash here is always better than a silent empty column.
    """
    missing = [r for r in rows if r not in mapping]
    if missing:
        raise SystemExit(
            f"\nCOLD-START FAILURE: {what}\n"
            f"  {source}\n"
            f"  covers {len(rows) - len(missing)} of {len(rows)} test rows; "
            f"{len(missing)} missing (e.g. {missing[:5]}).\n"
            f"  This artifact was built for a DIFFERENT test set.\n"
            f"  {remedy}\n"
            f"  Refusing to emit a silently-empty column.")


def _check_risk_head(ensembled_path):
    """Refuse an ensembled-risk file built from the wrong model head.

    DualHeadModel returns the categorical logits under RISK_CAT_ONLY=1 and a
    fixed 50/50 categorical+ordinal mixture otherwise. Those are two different
    models producing two different probability files, and the CSV itself
    carries no record of which one it came from -- both have identical columns
    and a plausible distribution. Rebuilding without the flag therefore yields
    a complete, well-formed submission with a silently different risk column,
    which is 40% of the composite.

    ensemble_risk.py writes the choice to <out>.meta.json. Where that sidecar
    exists this refuses on a mismatch; where it does not (a file predating the
    sidecar) it warns, because the alternative -- refusing outright -- would
    reject artifacts that are known good.
    """
    meta_path = ensembled_path + ".meta.json"
    if not os.path.exists(meta_path):
        print(f"  NOTE  no {os.path.basename(meta_path)} beside the ensembled "
              f"risk file, so the model head it used cannot be verified. "
              f"Regenerate with src/ensemble_risk.py to record it.")
        return
    with open(meta_path) as fh:
        meta = json.load(fh)
    if not meta.get("risk_cat_only", False):
        raise SystemExit(
            f"\nWRONG RISK HEAD: {ensembled_path}\n"
            f"  {meta_path} records risk_cat_only=false, so this file holds "
            f"the 50/50 categorical+ordinal mixture.\n"
            f"  The deployed risk column reads the CATEGORICAL head only.\n"
            f"  Rebuild with:  python src/ensemble_risk.py --cat-only ...\n"
            f"  Refusing to ship a risk column from a different model.")
    print("  risk head verified: categorical only")


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--test", default="data/test.csv")
    # THE SHIPPED ENSEMBLE IS THE CAT-ONLY ONE. The deployed risk probabilities
    # are produced with the ordinal head disabled at inference
    # (RISK_CAT_ONLY=1), which is a different probability file from the default
    # 50/50 head mixture. Pointing this at artifacts/ensembled_risk.csv rebuilds
    # the mixture column instead, and because that file also reproduces an
    # older submission, --verify can pass against a superseded reference while
    # the organisers' re-run ships the wrong risk column.
    #
    # ON FRESH DATA this file must be regenerated WITH the env var:
    #   RISK_CAT_ONLY=1 python src/ensemble_risk.py --test <new.csv> \
    #       --our-risk-run runs/risk_v1,runs/risk_v1_s14,runs/risk_v1_s15 \
    #       --friend-probs artifacts/friend_risk_probs.csv \
    #       --out artifacts/ensembled_risk_catonly2.csv
    # RISK_CAT_ONLY is read inside DualHeadModel.forward (train_risk.py:153),
    # so it only affects that step; nothing here loads a model.
    a_.add_argument("--ensembled",
                    default="artifacts/ensembled_risk_catonly2.csv")
    a_.add_argument("--nli", default="artifacts/nli_risk_test_probs.csv")
    a_.add_argument("--route", default="artifacts/cot_soft_oof.csv.extra.csv")
    # THE FACTOR DEFAULT MUST TRACK THE REFERENCE SUBMISSION. Each factor CSV
    # in artifacts/ is named for the submission it belongs to; a default left
    # pointing at an older one produces a factor column that differs from the
    # reference on many rows, which --verify reports as drift. Clearing that
    # drift with --allow-drift silently ships the older column, so the fix is
    # always to repoint this default, never to suppress the check.
    #
    # THE SHIPPED FACTOR COLUMN IS THE RESCALED ONE. The pre-rescale column
    # (`factors_ff11_test.csv`) reproduces an earlier submission, not the
    # current reference. Per-arm scale matching is applied by:
    #
    #   python src/factor_threshold_stabilise.py --use tfit --match-oof-scale \
    #       --arms mnli dmnli cmine prot d4 qnli mnli7 lnli ltail cbnsynapi \
    #             friendfac \
    #       --shipped artifacts/factors_ff11_test.csv \
    #       --champion submissions/NIDHI_scaled.csv --expect-change 0.30 \
    #       --build-test artifacts/factors_scaled_test.csv
    #
    # ON FRESH DATA that step must be re-run FIRST -- it is transductive and
    # recomputes each arm's scale from whatever rows it is given, so it
    # self-adapts, but the CSV below will not exist until it has.
    a_.add_argument("--factors", default="artifacts/factors_scaled_test.csv")
    # The taggers are RISK-CONDITIONED, so a candidate dump is only valid for
    # the risk column it was generated against. `evidence_runs_rc` was dumped
    # against an earlier risk column; the shipped candidates live in
    # `evidence_runs_ff11`, re-dumped against the risk column this script
    # builds. Mixing the two conditions the taggers on labels the pipeline does
    # not emit.
    a_.add_argument("--ev-dir", default="artifacts/evidence_runs_ff11")
    a_.add_argument("--ev-runs", default=",".join(RC7))
    a_.add_argument("--ev-cap", type=int, default=3,
                    help="per-post span cap; 3 is the shipped value")
    a_.add_argument("--ev-min-conf", type=float, default=0.65,
                    help="UNIFORM confidence gate replacing each run's own "
                         "evidence_min_conf; 0.65 is the shipped value")
    # The submission this rebuild is checked against. It must name the file
    # whose columns the defaults above reproduce -- an older submission here
    # turns --verify into a check of the wrong artefact.
    a_.add_argument("--reference", default="submissions/NIDHI_scaled.csv")
    a_.add_argument("--nli-w", type=float, default=0.30)
    a_.add_argument("--team", default="NIDHI")
    a_.add_argument("--out", default=None)
    a_.add_argument("--verify", action="store_true")
    a_.add_argument("--allow-drift", action="store_true")
    args = a_.parse_args()

    te = load_canonical(args.test)
    te["row_id"] = te["row_id"].astype(str)
    rows = te["row_id"].tolist()
    posts = dict(zip(rows, te["post"].astype(str)))
    print(f"test {len(rows)} posts (canonicalised via data_audit.load)")

    # ---------------------------------------------------------------- RISK
    _check_risk_head(args.ensembled)
    ens = pd.read_csv(args.ensembled).set_index("row_id")
    nli = pd.read_csv(args.nli).set_index("row_id")
    ens.index = ens.index.astype(str)
    nli.index = nli.index.astype(str)
    E = norm(ens.loc[rows, ENSCOLS].to_numpy(float))
    N = norm(nli.loc[rows, NLICOLS].to_numpy(float))
    M = (1 - args.nli_w) * E + args.nli_w * N
    lab = [LEVELS[i] for i in M.argmax(1)]
    lab = [normalize_label(apply_rules(posts[r], l)[0])
           for r, l in zip(rows, lab)]
    rt = pd.read_csv(args.route, keep_default_na=False)
    rt["row_id"] = rt["row_id"].astype(str)
    rcol = "cot" if "cot" in rt.columns else "pred_risk_level"
    route = {r: normalize_label(v) for r, v in zip(rt["row_id"], rt[rcol])}
    _require_coverage(route, rows, args.route, "route_attempt",
                      "regenerate with scripts/risk_cot_member.sh "
                      "(TEST=<your test csv>) -- it needs Qwen2.5-72B and "
                      "~20h of GPU. Uncaught, this rule no-ops silently and "
                      "the attempt class is never promoted.")
    risk = [("attempt" if (route.get(r) == "attempt" and l != "indicator")
             else l) for r, l in zip(rows, lab)]
    print(f"  RISK  built   " + "  ".join(
        f"{c} {risk.count(c)}" for c in LEVELS))

    # ------------------------------------------------------------ EVIDENCE
    rk = pd.DataFrame({"row_id": rows,
                       "pred_risk_level": [x.capitalize() for x in risk]})
    tmp_risk = "artifacts/_build_risk.csv"
    tmp_ev = "artifacts/_build_evidence.csv"
    # artifacts/ is gitignored and excluded from the submission zip, so on a
    # clean unzip this directory does not exist and writing the intermediates
    # below would fail with OSError before any model is loaded.
    os.makedirs("artifacts", exist_ok=True)
    rk.to_csv(tmp_risk, index=False)
    # --cap 3 and --min-conf-override 0.65 are part of the SHIPPED decode;
    # omitting either builds a different evidence column from the reference.
    # The cap bounds precision loss on posts where the pool proposes many
    # overlapping spans, since evidence F1 is averaged per post and extra
    # unmatched spans only reduce that post's precision.
    # The uniform gate REPLACES each run's own evidence_min_conf. Those
    # per-run values were each chosen while training that tagger in isolation,
    # so as a set they impose an arbitrary mix of operating points on a pool
    # whose members must agree k>=2 to emit anything; one gate applied to all
    # members makes the vote a comparison between like quantities.
    # 0.65/k2/cap3 is the global maximum of a 48-cell (gate x k x cap) grid.
    cmd = [sys.executable, "src/evidence_vote_test.py",
           "--runs", args.ev_runs, "--dir", args.ev_dir,
           "--split", "test", "--test", args.test,
           "--k", "2", "--cluster", "--rep", "modal",
           "--cap", str(args.ev_cap),
           "--min-conf-override", str(args.ev_min_conf),
           "--risk-from", tmp_risk, "--out", tmp_ev]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit("evidence vote failed")
    ev = pd.read_csv(tmp_ev, keep_default_na=False)
    ev["row_id"] = ev["row_id"].astype(str)
    evm = dict(zip(ev["row_id"], ev["evidence"]))
    _require_coverage(evm, rows, tmp_ev, "evidence",
                      "the RC7 candidate dumps in --ev-dir were produced for "
                      "a DIFFERENT test set. Re-dump with "
                      "src/evidence_dump_all.py --risk-cond --test-risk "
                      "<this run's own risk column>, into a fresh --outdir so "
                      "no skip-if-exists guard hands back the stale files.")
    evidence = [evm.get(r, "") for r in rows]
    n_emp = sum(1 for x in evidence if not parse_evidence(str(x)))
    n_ind = sum(1 for x in risk if x == "indicator")
    print(f"  EVID  built   {n_emp} empty of {len(rows)}  "
          f"(indicator {n_ind})   {len(RC7)} taggers, v2_rc excluded")
    bad = [rows[i] for i in range(len(rows))
           if (risk[i] == "indicator")
           != (not parse_evidence(str(evidence[i])))]
    if bad:
        raise SystemExit(f"indicator<->empty broken on {len(bad)} rows "
                         f"(e.g. {bad[:3]})")

    # ------------------------------------------------------------- FACTORS
    fa = pd.read_csv(args.factors, keep_default_na=False)
    fa["row_id"] = fa["row_id"].astype(str)
    fcol = next((c for c in ("factors", "pred_factors") if c in fa.columns),
                None)
    if fcol is None:
        raise SystemExit(f"{args.factors} has no factors column")
    fm = dict(zip(fa["row_id"], fa[fcol]))
    _require_coverage(fm, rows, args.factors, "factors",
                      "rebuild with src/factor_threshold_decode.py "
                      "--build-test (or factor_threshold_stabilise.py "
                      "--use tfit) against THIS test set. Factors are 30% of "
                      "the composite and this failure writes an empty cell "
                      "for every post with no error at all.")
    factors = [fm.get(r, "") for r in rows]
    print(f"  FACT  built   "
          f"{np.mean([len(parse_factors(str(x))) for x in factors]):.2f}"
          f" per post")

    # LOWERCASE, matching the serialisation of every file in submissions/.
    # Case carries no points: the gold column itself mixes `Ideation `,
    # `ideation ` and `ideation`, so the official scorer has to normalise it.
    # But the verifier below also compares through `normalize_label` and is
    # therefore blind to casing, so nothing downstream would flag a change
    # here -- and the final evaluation runs THIS script, so its output must
    # serialise identically to the file that was validated, not merely score
    # the same. The byte-exact check further down exists for that reason.
    out = pd.DataFrame({"row_id": rows,
                        "risk_level": [str(x).strip().lower() for x in risk],
                        "evidence": evidence, "factors": factors})

    # -------------------------------------------------------------- VERIFY
    if os.path.exists(args.reference):
        ref = pd.read_csv(args.reference, keep_default_na=False)
        ref["row_id"] = ref["row_id"].astype(str)
        rc = "risk_level" if "risk_level" in ref.columns else "pred_risk_level"
        ok = list(ref["row_id"]) == rows
        print(f"\nverify against {args.reference}")
        print(f"  row order {'OK' if ok else 'DIFFERS'}")
        drift = 0
        for c, mine in (("risk", [normalize_label(x) for x in out.risk_level]),
                        ("evidence", list(out.evidence)),
                        ("factors", list(out.factors))):
            theirs = ([normalize_label(x) for x in ref[rc]] if c == "risk"
                      else list(ref[c]))
            if c == "risk":
                n = sum(1 for a, b in zip(mine, theirs) if a != b)
            else:
                n = sum(1 for a, b in zip(mine, theirs)
                        if str(a).strip() != str(b).strip())
            drift += n
            print(f"  {c:<9}{n:>4} of {len(rows)} differ"
                  + ("   <== MATCH" if n == 0 else ""))

        # BYTE-EXACT CHECK, on top of the normalised one above. The normalised
        # comparison is the one that corresponds to the score, but it is blind
        # to serialisation -- casing and whitespace differences pass it
        # silently. The final evaluation runs THIS script, so a serialisation
        # drift means the judges' artifact is not the artifact that was
        # validated, even when both would score the same. Warn rather than
        # refuse: only the normalised check gates.
        exact = {c: sum(1 for a, b in zip(out[c].astype(str),
                                          ref[rc if c == "risk_level" else c]
                                          .astype(str)) if a != b)
                 for c in ("risk_level", "evidence", "factors")
                 if (rc if c == "risk_level" else c) in ref.columns}
        if any(exact.values()):
            print("  byte-exact drift (scores the same, serialises "
                  f"differently): {dict((k, v) for k, v in exact.items() if v)}")
        else:
            print("  byte-exact: identical serialisation too")
        if drift and not args.allow_drift:
            raise SystemExit(
                "\nREFUSING: the rebuild does not reproduce the reference. "
                "A pipeline that cannot rebuild its own output should not be "
                "run on unseen data. Investigate, or pass --allow-drift if "
                "the difference is intended.")
        if drift == 0:
            print("\n  FULL MATCH -- the chain is reproducible end to end.")
    elif args.verify:
        # A MISSING REFERENCE MUST NOT PASS. `--verify` exists to prove the
        # chain rebuilds the shipped column, so skipping the comparison when
        # the reference file is absent would report success without having
        # compared anything -- the one situation in which the flag is most
        # likely to be trusted, such as a fresh unzip of the submission where
        # only some files under submissions/ are present.
        raise SystemExit(
            f"REFUSING: --verify was requested but {args.reference} does not "
            f"exist, so there is nothing to verify against. Point "
            f"--reference at the shipped submission, or drop --verify if you "
            f"are deliberately building for a NEW test set (its rows will "
            f"not match any reference we hold).")
    else:
        print(f"\n(no reference at {args.reference}; nothing to verify "
              f"against -- this is a build, not a verification)")

    for p in (tmp_risk, tmp_ev):
        try:
            os.remove(p)
        except OSError:
            pass

    if args.verify or not args.out:
        print("\n(--verify: nothing written)")
        return
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # lineterminator is explicit so the file is byte-identical on Windows
    # and Linux; pandas would otherwise emit CRLF on Windows.
    out.to_csv(args.out, index=False, lineterminator=chr(10))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
