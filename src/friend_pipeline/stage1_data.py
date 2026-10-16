# Team NIDHI multi-task pipeline: one shared encoder with risk, evidence and
# factor heads. Vendored into this repository so the `friendfac` factor arm can
# be rebuilt from source without the original project tree.
"""Stage 1 - Data cleaning, user-grouped folds, evidence lexicon.

Catches handled here:
  * dirty risk labels (mixed case, trailing spaces) -> 4 canonical labels
  * factors parsed from python-list strings, deduplicated, validated vs taxonomy
  * evidence split on ';', snapped to the exact verbatim text of the post
  * 'none'/empty evidence -> empty list (predicting nothing is a valid answer)
  * duplicate (post, label) rows dropped
  * folds are grouped by anon_user_id -> no user appears in two folds
"""
import ast
import json
import os
from collections import Counter

import pandas as pd
from sklearn.model_selection import GroupKFold

import config as C

CANON = {"indicator": "Indicator", "ideation": "Ideation",
         "behavior": "Behavior", "attempt": "Attempt"}


def parse_evidence(ev, post):
    s = str(ev).strip()
    if s.lower() in ("none", "nan", ""):
        return []
    out = []
    for raw in s.split(";"):
        raw = raw.strip().strip('"').strip()
        if not raw:
            continue
        low = post.lower()
        if raw.lower() in low:
            i = low.index(raw.lower())
            out.append(post[i:i + len(raw)])  # snap to post's exact casing
        else:
            out.append(raw)  # keep non-verbatim gold (train only; ~9%)
    return out


def run():
    os.makedirs(C.OUT_DIR, exist_ok=True)
    tr = pd.read_excel(C.TRAIN_XLSX)
    lb = pd.read_excel(C.LEADERBOARD_XLSX)

    # --- labels ---
    tr["risk_level"] = tr["suicide risk"].astype(str).str.strip().str.lower().map(CANON)
    if tr["risk_level"].isna().any():
        bad = tr.loc[tr["risk_level"].isna(), "suicide risk"].unique()
        raise ValueError(f"Unmapped risk labels: {bad}")

    # --- factors ---
    valid = set(C.FACTORS)
    fac, unknown = [], Counter()
    for s in tr["factors"]:
        lst = sorted(set(x.strip() for x in ast.literal_eval(str(s)) if x.strip()))
        unknown.update([x for x in lst if x not in valid])
        fac.append(json.dumps([x for x in lst if x in valid]))
    tr["factors_clean"] = fac
    if unknown:
        print(f"[stage1] WARNING unknown factor strings dropped: {dict(unknown)}")

    # --- evidence ---
    tr["evidence_clean"] = [json.dumps(parse_evidence(e, p)) for e, p in
                            zip(tr["evidence for suicide risk level"], tr["post"])]

    # --- dedupe ---
    before = len(tr)
    tr = tr.drop_duplicates(subset=["post", "risk_level"]).reset_index(drop=True)
    print(f"[stage1] dropped {before - len(tr)} duplicate rows -> {len(tr)} rows")

    # --- grouped folds ---
    gkf = GroupKFold(n_splits=C.N_FOLDS)
    tr["fold"] = -1
    for f, (_, va) in enumerate(gkf.split(tr, tr["risk_level"], groups=tr["anon_user_id"])):
        tr.loc[va, "fold"] = f
    assert tr.groupby("anon_user_id")["fold"].nunique().max() == 1, "user leaked across folds"
    print("[stage1] fold x class counts:")
    print(pd.crosstab(tr["fold"], tr["risk_level"]))

    # --- lexicon (per-level evidence phrases, used by the rule layer) ---
    lex = {}
    for lvl in ["Ideation", "Behavior", "Attempt"]:
        cnt = Counter()
        for ev in tr.loc[tr.risk_level == lvl, "evidence_clean"]:
            for s in json.loads(ev):
                s = " ".join(s.lower().split())
                if len(s) >= 4:
                    cnt[s] += 1
        lex[lvl] = [p for p, n in cnt.most_common() if n >= 2]
    with open(os.path.join(C.OUT_DIR, "lexicon.json"), "w") as f:
        json.dump(lex, f, indent=1)

    keep = ["row_id", "anon_user_id", "post_id", "post",
            "risk_level", "evidence_clean", "factors_clean", "fold"]
    tr[keep].to_csv(os.path.join(C.OUT_DIR, "train_clean.csv"), index=False)
    lb.to_csv(os.path.join(C.OUT_DIR, "leaderboard_clean.csv"), index=False)
    print("[stage1] wrote train_clean.csv, leaderboard_clean.csv, lexicon.json")


if __name__ == "__main__":
    run()
