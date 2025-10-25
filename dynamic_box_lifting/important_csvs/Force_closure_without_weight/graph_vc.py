import pandas as pd
import matplotlib.pyplot as plt

# ===== PATH =====
CSV_PATH = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/dynamic_box_lifting/important_csvs/Lifting_with_weight_T1/v_cmd.csv"

# take the FIRST N samples
N_SAMPLES = 500

# ---------- Load & prep ----------
df = pd.read_csv(CSV_PATH, skipinitialspace=True)

# force numeric (avoids weird dtype issues)
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# keep valid times, sort, and slice first N
df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
head = df.head(min(N_SAMPLES, len(df)))

# convenience: NumPy arrays (prevents Matplotlib 1-D errors)
t = head["time"].to_numpy()

cur_vx = head["cur_vx"].to_numpy(); cur_vy = head["cur_vy"].to_numpy(); cur_vz = head["cur_vz"].to_numpy()
cur_wx = head["cur_wx"].to_numpy(); cur_wy = head["cur_wy"].to_numpy(); cur_wz = head["cur_wz"].to_numpy()

pose_x = head["pose_x"].to_numpy(); pose_y = head["pose_y"].to_numpy(); pose_z = head["pose_z"].to_numpy()
pose_rx = head["pose_rx"].to_numpy(); pose_ry = head["pose_ry"].to_numpy(); pose_rz = head["pose_rz"].to_numpy()

cmd_vx = head["cmd_vx"].to_numpy(); cmd_vy = head["cmd_vy"].to_numpy(); cmd_vz = head["cmd_vz"].to_numpy()
cmd_wx = head["cmd_wx"].to_numpy(); cmd_wy = head["cmd_wy"].to_numpy(); cmd_wz = head["cmd_wz"].to_numpy()

# ---------- Linear velocities (current vs command) ----------
plt.figure(figsize=(12, 5))
plt.plot(t, cur_vx, label="cur_vx", linewidth=1.6)
plt.plot(t, cmd_vx, label="cmd_vx", linewidth=1.6)
plt.plot(t, cur_vy, label="cur_vy", linewidth=1.6)
plt.plot(t, cmd_vy, label="cmd_vy", linewidth=1.6)
plt.plot(t, cur_vz, label="cur_vz", linewidth=1.6)
plt.plot(t, cmd_vz, label="cmd_vz", linewidth=1.6)
plt.title(f"Linear Velocities — current vs command (first {len(head)} samples)")
plt.xlabel("Time [s]")
plt.ylabel("Linear velocity [m/s]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# ---------- Angular velocities (current vs command) ----------
plt.figure(figsize=(12, 5))
plt.plot(t, cur_wx, label="cur_wx", linewidth=1.6)
plt.plot(t, cmd_wx, label="cmd_wx", linewidth=1.6)
plt.plot(t, cur_wy, label="cur_wy", linewidth=1.6)
plt.plot(t, cmd_wy, label="cmd_wy", linewidth=1.6)
plt.plot(t, cur_wz, label="cur_wz", linewidth=1.6)
plt.plot(t, cmd_wz, label="cmd_wz", linewidth=1.6)
plt.title(f"Angular Velocities — current vs command (first {len(head)} samples)")
plt.xlabel("Time [s]")
plt.ylabel("Angular velocity [rad/s]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# ---------- Pose (position) ----------
plt.figure(figsize=(12, 5))
plt.plot(t, pose_x, label="pose_x", linewidth=1.6)
plt.plot(t, pose_y, label="pose_y", linewidth=1.6)
plt.plot(t, pose_z, label="pose_z", linewidth=1.6)
plt.title(f"Pose — position (first {len(head)} samples)")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# ---------- Pose (orientation) ----------
plt.figure(figsize=(12, 5))
plt.plot(t, pose_rx, label="pose_rx", linewidth=1.6)
plt.plot(t, pose_ry, label="pose_ry", linewidth=1.6)
plt.plot(t, pose_rz, label="pose_rz", linewidth=1.6)
plt.title(f"Pose — orientation (first {len(head)} samples)")
plt.xlabel("Time [s]")
plt.ylabel("Orientation [rad]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()
