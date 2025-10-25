import pandas as pd
import matplotlib.pyplot as plt

# ===== PATHS =====
FRANKA_CSV = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/dynamic_box_lifting/important_csvs/Lifting_with_weight_T1/fr3_force_log.csv"
HEAL_CSV   = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/dynamic_box_lifting/important_csvs/Lifting_with_weight_T1/heal_force_log.csv"

# take the FIRST N samples from each CSV
N_SAMPLES = 500

def pick(colnames, *candidates):
    """Return the first column name that exists from candidates."""
    for c in candidates:
        if c in colnames:
            return c
    raise KeyError(f"None of {candidates} found. Available: {list(colnames)}")

# ---------- Load & prep FRANKA ----------
franka = pd.read_csv(FRANKA_CSV, skipinitialspace=True)
franka["time"] = pd.to_numeric(franka["time"], errors="coerce")
franka.dropna(subset=["time"], inplace=True)
franka.sort_values("time", inplace=True)

F_star_col = pick(franka.columns, "lambda_star_Fy", "lam_star_Fy")
F_meas_col = pick(franka.columns, "meas_Fy")

franka_head = franka.head(N_SAMPLES)
tF  = franka_head["time"].to_numpy()
FsF = franka_head[F_star_col].to_numpy()
FmF = franka_head[F_meas_col].to_numpy()

# ---------- Load & prep HEAL ----------
heal = pd.read_csv(HEAL_CSV, skipinitialspace=True)
heal["time"] = pd.to_numeric(heal["time"], errors="coerce")
heal.dropna(subset=["time"], inplace=True)
heal.sort_values("time", inplace=True)

H_star_col = pick(heal.columns, "lambda_star_Fx", "lam_star_Fx")
H_meas_col = pick(heal.columns, "meas_Fx")

heal_head = heal.head(N_SAMPLES)
tH  = heal_head["time"].to_numpy()
FsH = heal_head[H_star_col].to_numpy()
FmH = heal_head[H_meas_col].to_numpy()

# ---------- Plot FRANKA (desired vs measured) ----------
plt.figure(figsize=(12, 4))
plt.plot(tF, FsF, label="Franka target (Fy)", linewidth=1.6)
plt.plot(tF, FmF, label="Franka measured (Fy)", linewidth=1.6)
plt.title(f"Franka — Desired vs Measured Force (first {N_SAMPLES} samples)")
plt.xlabel("Time [s]")
plt.ylabel("Force [N]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# ---------- Plot HEAL (desired vs measured) ----------
plt.figure(figsize=(12, 4))
plt.plot(tH, FsH, label="HEAL target (Fx)", linewidth=1.6)
plt.plot(tH, FmH, label="HEAL measured (Fx)", linewidth=1.6)
plt.title(f"HEAL — Desired vs Measured Force (first {N_SAMPLES} samples)")
plt.xlabel("Time [s]")
plt.ylabel("Force [N]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()
