'''
This code publishes the force on franka to maintatin the force closure 

'''


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import numpy as np
import rospy
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from franka_msgs.msg import FrankaState
import sys
sys.path.insert(0, '/home/iitgn-robotics/ds_yash/bimanual_ws/src')

from dynamic_box_lifting.utils.jacobian_franka import FrankaKDL

# ======================= Parameters ==========================
K_VEC = np.array([1.0, 0.2, 1.0, 0.1, 0.1, 0.1], dtype=float)
MAX_JOINT_VEL = 0.1  # rad/s

# Desired external wrench (BASE/EE content is irrelevant here; we compare with measured base-frame)
#     [Fx, Fy, Fz, Tx, Ty, Tz]
LAMBDA_STAR = np.array([0.0, 0.0, -6.0, 0.0, 0.0, 0.0], dtype=float)

# === Output directory: graph/<|Fz|>N_des ===
FORCE_Z   = float(LAMBDA_STAR[2])
FORCE_TAG = f"{abs(int(round(FORCE_Z)))}N_des"
GRAPH_BASE = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/fr3_controllers/dynamic_box_lifting/data_for_ee_force/graph"
CSV_DIR = os.path.join(GRAPH_BASE, FORCE_TAG)
os.makedirs(CSV_DIR, exist_ok=True)

# FR3 specifics
URDF_FILE = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/addverb_heal_description/urdf/fr3.urdf"
BASE_LINK = "fr3_link0"
TIP_LINK  = "fr3_link8"
FR3_JOINTS = [
    "fr3_joint1","fr3_joint2","fr3_joint3",
    "fr3_joint4","fr3_joint5","fr3_joint6","fr3_joint7"
]

# ROS topics (namespaced for fr3)
TOPIC_FRANKA_STATE = "/fr3/franka_state_controller/franka_states"
TOPIC_JOINT_STATES = "/fr3/joint_states"
TOPIC_VEL_COMMAND  = "/fr3/joint_velocity_controller/joint_velocity_command"

# ======================= State ===============================
f_left = np.zeros(6, dtype=float)   # measured external wrench in BASE frame (we store O_F_ext_hat_K here)
joint_positions = None              # np.array shape (7,)
n_joints = 7

# Buffers for plotting at shutdown
_tbuf, _qdbuf, _vcbuf, _fbuf = [], [], [], []

# ======================= CSV setup ===========================
joint_f = open(os.path.join(CSV_DIR, "joint_velocities.csv"), "w", newline="")
cart_f  = open(os.path.join(CSV_DIR, "cartesian_velocities.csv"), "w", newline="")
force_f = open(os.path.join(CSV_DIR, "forces.csv"), "w", newline="")

joint_w = csv.writer(joint_f)
cart_w  = csv.writer(cart_f)
force_w = csv.writer(force_f)

joint_w.writerow(["time"] + [f"qdot_{i}" for i in range(n_joints)])
cart_w.writerow( ["time"] + [f"v_c_{i}"  for i in range(6)] )
force_w.writerow(["time"] + [f"lambda_meas_{i}" for i in range(6)] +
                          [f"lambda_star_{i}" for i in range(6)])

def close_files_and_plots():
    """Flush + close CSVs and save a single combined figure + qdot plot."""
    try:
        joint_f.close(); cart_f.close(); force_f.close()
    except Exception:
        pass

    if len(_tbuf) > 2:
        t  = np.array(_tbuf)
        vc = np.array(_vcbuf)      # (N,6) cartesian velocities
        f  = np.array(_fbuf)       # (N,6) measured wrench in base frame
        qd = np.array(_qdbuf)      # (N,7) joint velocities

        # -------- Combined figure (top: v_c; bottom: Fz Measured vs Desired) --------
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8), sharex=True)

        # Top: Cartesian velocities
        for i, label in enumerate(["vx","vy","vz","wx","wy","wz"]):
            ax1.plot(t, vc[:, i], label=label)
        ax1.set_ylabel("Cartesian velocity")
        ax1.set_title("Cartesian velocities over time")
        ax1.grid(True)
        ax1.legend()

        # Bottom: Fz measured vs desired
        ax2.plot(t, f[:, 2], label="Measured Fz (base frame)")
        ax2.plot(t, np.ones_like(t) * LAMBDA_STAR[2], "--", label=f"Desired Fz = {LAMBDA_STAR[2]:.1f} N")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Force Z [N]")
        ax2.set_title("Fz (Measured vs Desired)")
        ax2.grid(True)
        ax2.legend()

        fig.tight_layout()
        plt.savefig(os.path.join(CSV_DIR, "combined_plot.png"), dpi=150)
        plt.close()

        # -------- Separate qdot time series (optional, unchanged) --------
        plt.figure(figsize=(10,4))
        for i in range(qd.shape[1]):
            plt.plot(t, qd[:, i], label=f"qdot_{i+1}")
        plt.xlabel("Time [s]"); plt.ylabel("Joint velocity [rad/s]")
        plt.title("qdot vs time")
        plt.grid(True); plt.legend(ncol=2); plt.tight_layout()
        plt.savefig(os.path.join(CSV_DIR, "qdot_timeseries.png"), dpi=150)
        plt.close()

    rospy.loginfo(f"CSV & plots saved in {CSV_DIR}")

# ======================= Callbacks ===========================
def franka_state_cb(msg: FrankaState):
    """Use external wrench in BASE frame: O_F_ext_hat_K."""
    global f_left
    try:
        meas = np.array(list(msg.O_F_ext_hat_K), dtype=float)  # [Fx,Fy,Fz,Tx,Ty,Tz] in base frame
        f_left = meas  # no filtering
    except Exception as e:
        rospy.logwarn_throttle(2.0, f"Failed to parse O_F_ext_hat_K: {e}")

def joint_cb(msg: JointState):
    """Keep joint_positions in FR3 order."""
    global joint_positions
    name_to_pos = {n: p for n, p in zip(msg.name, msg.position)}
    try:
        joint_positions = np.array([name_to_pos[j] for j in FR3_JOINTS], dtype=float)
    except KeyError:
        pass

# ======================= Main ================================
def main():
    rospy.init_node("force_vel_mapping_franka")
    rospy.on_shutdown(close_files_and_plots)

    # --- KDL init (URDF → chain/solver) ---
    franka_kdl = FrankaKDL(
        urdf_file=URDF_FILE,
        base_link=BASE_LINK,
        tip_link=TIP_LINK,
        joint_names=FR3_JOINTS
    )

    # --- ROS I/O ---
    rospy.Subscriber(TOPIC_FRANKA_STATE, FrankaState, franka_state_cb, queue_size=1)
    rospy.Subscriber(TOPIC_JOINT_STATES, JointState,  joint_cb,       queue_size=1)
    pub = rospy.Publisher(TOPIC_VEL_COMMAND, Float64MultiArray, queue_size=1)

    rate = rospy.Rate(100)
    t0 = rospy.Time.now().to_sec()

    while not rospy.is_shutdown():
        t_now = rospy.Time.now().to_sec() - t0

        lam_meas = f_left.copy()       # measured wrench (BASE frame)
        lam_star = LAMBDA_STAR.copy()  # desired wrench (we control only Fz component)

        # Default outputs
        q_dot = np.zeros(n_joints, dtype=float)
        v_c   = np.zeros(6, dtype=float)

        if joint_positions is not None and len(joint_positions) == n_joints:
            # Z-only force control
            force_error = lam_star - lam_meas
            force_error[0:2] = 0.0  # no Fx, Fy
            force_error[3:6] = 0.0  # no torques

            if abs(force_error[2]) >= 0.01:  # deadband on Fz
                v_c = (0.01 * force_error) / K_VEC
                try:
                    J = franka_kdl.compute_jacobian(joint_positions)  # 6x7
                    q_dot = np.clip(np.linalg.pinv(J) @ v_c, -MAX_JOINT_VEL, MAX_JOINT_VEL)
                except Exception as e:
                    rospy.logwarn_throttle(1.0, f"Jacobian/command error: {e}")
                    q_dot = np.zeros(n_joints, dtype=float)

        # Publish to C++ controller
        pub.publish(Float64MultiArray(data=q_dot.tolist()))

        # CSV log + buffers
        _tbuf.append(t_now)
        _qdbuf.append(q_dot.tolist())
        _vcbuf.append(v_c.tolist())
        _fbuf.append(lam_meas.tolist())

        joint_w.writerow([t_now] + q_dot.tolist()); joint_f.flush()
        cart_w.writerow( [t_now] + v_c.tolist() );  cart_f.flush()
        force_w.writerow([t_now] + lam_meas.tolist() + lam_star.tolist()); force_f.flush()

        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
