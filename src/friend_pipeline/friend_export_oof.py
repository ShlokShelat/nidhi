# Team NIDHI multi-task pipeline: one shared encoder with risk, evidence and
# factor heads. Vendored into this repository so the `friendfac` factor arm can
# be rebuilt from source without the original project tree.
"""
Export stage2's OOF predictions to a flat CSV.

Reads outputs/oof.npy (written by stage 2) and flattens every fold's held-out
risk and factor probabilities into one row per post, so the other half of the
system can consume this model as an ensemble member without importing it.
"""
import os
import numpy as np
import pandas as pd
import config as C

all_oof = np.load(os.path.join(C.OUT_DIR, "oof.npy"), allow_pickle=True).item()

rows = []
for fold, oof in all_oof.items():
    n = len(oof["row_id"])
    for i in range(n):
        row = {"row_id": oof["row_id"][i]}
        for j, lvl in enumerate(C.RISK_LEVELS):
            row[lvl] = float(oof["risk"][i, j])
        for j, fac in enumerate(C.FACTORS):
            row[fac] = float(oof["fact"][i, j])
        rows.append(row)

df = pd.DataFrame(rows)
out = os.path.join(C.OUT_DIR, "oof_export.csv")
df.to_csv(out, index=False)
print(f"wrote {out}: {len(df)} rows (all folds concatenated, OOF-honest)")
print(df[["row_id"] + C.RISK_LEVELS].head(3))