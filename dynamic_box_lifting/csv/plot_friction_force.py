import pandas as pd
import matplotlib.pyplot as plt

# Load CSVs (no filtering; just read exactly what's there)
heal_df = pd.read_csv(
    "/home/iitgn-robotics/ds_yash/bimanual_ws/src/dynamic_box_lifting/csv/heal_force_log.csv",
    skipinitialspace=True,
)
franka_df = pd.read_csv(
    "/home/iitgn-robotics/ds_yash/bimanual_ws/src/dynamic_box_lifting/csv/fr3_force_log.csv",
    skipinitialspace=True,
)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# --------------- HEAL ---------------
axes[0].plot(
    heal_df["time"].to_numpy(), heal_df["lam_star_Fx"].to_numpy(), label="lam_star_Fx"
)
axes[0].plot(
    heal_df["time"].to_numpy(), heal_df["meas_Fx"].to_numpy(), label="meas_Fx"
)
axes[0].plot(
    heal_df["time"].to_numpy(), heal_df["meas_Fy"].to_numpy(), label="meas_Fy"
)
axes[0].plot(
    heal_df["time"].to_numpy(), heal_df["meas_Fz"].to_numpy(), label="meas_Fz"
)
axes[0].set_title("Heal Robot Forces")
axes[0].set_ylabel("Force [N]")
axes[0].set_ylim(-30, 5)
axes[0].legend()

# --------------- FRANKA ---------------
axes[1].plot(
    franka_df["time"].to_numpy(), franka_df["lam_star_Fy"].to_numpy(), label="lam_star_Fy"
)
axes[1].plot(
    franka_df["time"].to_numpy(), franka_df["meas_Fx"].to_numpy(), label="meas_Fx"
)
axes[1].plot(
    franka_df["time"].to_numpy(), franka_df["meas_Fy"].to_numpy(), label="meas_Fy"
)
axes[1].plot(
    franka_df["time"].to_numpy(), franka_df["meas_Fz"].to_numpy(), label="meas_Fz"
)
axes[1].set_title("Franka Robot Forces")
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Force [N]")
axes[1].set_ylim(-30, 5)
axes[1].legend()
# For both subplots sharing x-axis
axes[1].set_xlim(1756159468.3953454, 1756159455.505622,)

plt.tight_layout()
plt.show()
