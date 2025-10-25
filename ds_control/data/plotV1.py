#!/usr/bin/env python3

import pandas as pd

# === File paths ===
input_csv  = "/home/iitgn-robotics/bimanual_ws/src/ds_control/data/joint_log.csv"
output_csv = "/home/iitgn-robotics/bimanual_ws/src/ds_control/data/joint_log_merged.csv"

# === Load the raw log ===
df = pd.read_csv(input_csv)

# === Identify column groups ===
heal_cols = [c for c in df.columns if c.startswith("heal_")]
fr3_cols  = [c for c in df.columns if c.startswith("fr3_")]

# === Separate HEAL and FR3 rows ===
heal_df = df[["time"] + heal_cols].dropna(how="all", subset=heal_cols).copy()
fr3_df  = df[["time"] + fr3_cols].dropna(how="all", subset=fr3_cols).copy()

# === Merge by nearest timestamp (within 3ms tolerance) ===
TOL_SECONDS = 0.003
heal_df = heal_df.sort_values("time").reset_index(drop=True)
fr3_df  = fr3_df.sort_values("time").reset_index(drop=True)

merged = pd.merge_asof(
    heal_df,
    fr3_df,
    on="time",
    direction="nearest",
    tolerance=TOL_SECONDS
)

# === Drop rows where FR3 or HEAL is still fully missing ===
missing_heal = merged[heal_cols].isnull().all(axis=1)
missing_fr3 = merged[fr3_cols].isnull().all(axis=1)
merged_clean = merged[~(missing_heal | missing_fr3)].reset_index(drop=True)

# === Save to CSV ===
merged_clean.to_csv(output_csv, index=False)
print(f"✔ Merged log saved to:\n{output_csv}")
