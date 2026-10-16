"""Small stages of the new-data run that are not scripts of their own.

Invoked by src/run_new_data.py; not intended to be run by hand. Each
subcommand is a separate process so a failure stops the pipeline at exactly
the stage that failed, the same as every other stage.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LEVELS = ["indicator", "ideation", "behavior", "attempt"]


def noop_route(a):
    """A route file that routes nothing, for --skip-route."""
    import pandas as pd
    te = pd.read_csv(a.test)
    pd.DataFrame({"row_id": te["row_id"],
                  "cot": ["none"] * len(te)}).to_csv(a.out, index=False)
    print(f"  wrote {a.out}: attempt routing DISABLED for {len(te)} rows")


def risk_column(a):
    """0.70*ensemble + 0.30*entailment -> symbolic rules -> attempt routing.

    This mirrors build_submission.py's risk block exactly. It has to be
    computed here as well because the evidence taggers are conditioned on the
    risk column, and conditioning them on a stored submission would bind them
    to row_ids that do not exist in the new data.
    """
    import numpy as np
    import pandas as pd
    from data_audit import load as load_canonical
    from scoring import normalize_label
    from apply_rules_to_test import apply_rules

    D = a.workdir
    te = load_canonical(a.test)
    te["row_id"] = te["row_id"].astype(str)
    rows = te["row_id"].tolist()
    posts = dict(zip(rows, te["post"].astype(str)))

    def norm(X):
        return X / np.clip(X.sum(1, keepdims=True), 1e-9, None)

    ens = pd.read_csv(f"{D}/ensembled_risk.csv").set_index("row_id")
    nli = pd.read_csv(f"{D}/nli_risk_probs.csv").set_index("row_id")
    ens.index = ens.index.astype(str)
    nli.index = nli.index.astype(str)
    for name, frame in (("ensembled_risk", ens), ("nli_risk_probs", nli)):
        miss = [r for r in rows if r not in frame.index]
        if miss:
            sys.exit(f"{name}.csv covers {len(rows)-len(miss)}/{len(rows)} "
                     f"rows; first missing: {miss[:3]}")

    E = norm(ens.loc[rows, ["p_Indicator", "p_Ideation", "p_Behavior",
                            "p_Attempt"]].to_numpy(float))
    Nn = norm(nli.loc[rows, [f"prob_{c}" for c in LEVELS]].to_numpy(float))
    lab = [LEVELS[i] for i in (0.70 * E + 0.30 * Nn).argmax(1)]
    lab = [normalize_label(apply_rules(posts[r], lv)[0])
           for r, lv in zip(rows, lab)]

    rt = pd.read_csv(a.route, keep_default_na=False)
    rt["row_id"] = rt["row_id"].astype(str)
    rcol = "cot" if "cot" in rt.columns else "pred_risk_level"
    route = {r: normalize_label(v) for r, v in zip(rt["row_id"], rt[rcol])}
    nmiss = sum(1 for r in rows if r not in route)
    if nmiss:
        print(f"  WARNING: the route file covers {len(rows)-nmiss}/{len(rows)}"
              f" rows; the rest keep their un-routed label")

    risk = ["attempt" if (route.get(r) == "attempt" and lv != "indicator")
            else lv for r, lv in zip(rows, lab)]
    pd.DataFrame({"row_id": rows,
                  "pred_risk_level": [x.capitalize() for x in risk]}
                 ).to_csv(f"{D}/risk_col.csv", index=False)

    # evidence_dump_all always dumps a val split too; a 2-row stub keeps that
    # overhead near zero without changing its interface.
    te.head(2)[["row_id", "anon_user_id", "post_id", "post"]].to_csv(
        f"{D}/val_stub.csv", index=False)
    pd.read_csv(f"{D}/risk_col.csv").head(2).to_csv(
        f"{D}/val_stub_risk.csv", index=False)

    from collections import Counter
    print(f"  risk column for {len(rows)} rows: {dict(Counter(risk))}")


def check_evidence(a):
    """A partial vote is a different model from the one that was measured."""
    got = glob.glob(f"{a.workdir}/evidence_runs/*_test_candidates.csv")
    print(f"  evidence candidate dumps: {len(got)}/7")
    if len(got) != 7:
        sys.exit("a partial evidence vote is not the validated ensemble; "
                 "check that all 7 evidence_*_rc checkpoints are present")


def lora_arm(a):
    """One LoRA factor arm.

    The adapter is useless without its base model. The path recorded at
    training time is absolute and specific to the machine that trained it, so
    src/fetch_weights.py rewrites it to a Hub id; this reads whatever is
    recorded rather than hardcoding either form.
    """
    for f in range(5):
        d = os.path.join(ROOT, "runs", f"factors_{a.arm}_f{f}", "lora")
        if not os.path.isdir(d):
            sys.exit(f"{a.arm}: missing runs/factors_{a.arm}_f{f}/lora "
                     f"(a partial bag is a different model)")
    cfg = os.path.join(ROOT, "runs", f"factors_{a.arm}_f0", "run_config.json")
    try:
        base = json.load(open(cfg))["model"]
    except Exception as e:
        sys.exit(f"{a.arm}: cannot read the base model from {cfg}: {e}")
    if os.path.isabs(base) and not os.path.exists(base):
        sys.exit(f"{a.arm}: its base model is recorded as {base}, which does "
                 f"not exist here. Re-run src/fetch_weights.py, which "
                 f"rewrites this to a Hugging Face id.")
    print(f"  {a.arm}: 5/5 adapters, base {base}")
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "src",
                                      "predict_factors_llm_nli.py"),
         "--tag", a.arm, "--model", base, "--test", a.test,
         "--outdir", a.outdir, "--folds", "0,1,2,3,4"], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"{a.arm}: predict_factors_llm_nli.py failed")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("noop-route")
    p.add_argument("--test", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=noop_route)

    p = sub.add_parser("risk-column")
    p.add_argument("--test", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--route", required=True)
    p.set_defaults(fn=risk_column)

    p = sub.add_parser("check-evidence")
    p.add_argument("--workdir", required=True)
    p.set_defaults(fn=check_evidence)

    p = sub.add_parser("lora-arm")
    p.add_argument("--arm", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--outdir", required=True)
    p.set_defaults(fn=lora_arm)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
