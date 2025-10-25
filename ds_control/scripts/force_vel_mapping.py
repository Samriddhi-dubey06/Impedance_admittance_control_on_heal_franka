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
import os

# Add your workspace 'src' folder to PYTHONPATH
sys.path.append(os.path.join(os.path.expanduser("~"), "ds_yash", "bimanual_ws", "src"))

# Import your KDL helper (FrankaKDL) from utils
# Package path mirrors your folder layout
from fr3_controllers.dynamic_box_lifting.data_for_ee_force.utils.jacobian_franka import JacobianLogger


# ======================= Parameters ==========================
# Gains / limits
K_VEC = np.array([1.0, 0.2, 1.0, 0.1, 0.1, 0.1], dtype=float)
MAX_JOINT_VEL = 0.1  # rad/s
FILTER_ALPHA = 0.01  # low-pass on forces (0..1), smaller = more smoothing

# Desired external wrench (EE frame K)  [Fx, Fy, Fz, Tx, Ty, Tz]
LAMBDA_STAR = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=float)

# File paths
CSV_DIR = os.path.expanduser("~/csv")

# FR3 specifics
URDF_FILE = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/addverb_heal_description/urdf/fr3.urdf"
BASE_LINK = "fr3_link0"
TIP_LINK  = "fr3_link8"
FR3_JOINTS = [
    "fr3_joint1","fr3_joint2","fr3_joint3",
    "fr3_joint4","fr3_joint5","fr3_joint6","fr3_joint7"
]

# ROS topics (namespaced for fr3)
TOPIC_FRANKA_STATE   = "/fr3/franka_state_controller/franka_states"
TOPIC_JOINT_STATES   = "/fr3/joint_states"
TOPIC_VEL_COMMAND    = "/fr3/joint_velocity_controller/joint_velocity_command" # your C++ controller subscribes to "joint_velocity_command" in its NS

# ======================= State ===============================
f_left = np.zeros(6, dtype=float)         # filtered external wrench (EE frame K)
joint_positions = None                    # np.array shape (7,)
n_joints = 7

# Buffers for plotting at shutdown
_tbuf   = []
_qdbuf  = []   # list of lists (7)
_vcbuf  = []   # list of lists (6)
_fbuf   = []   # list of lists (6)

# ======================= CSV setup ===========================
os.makedirs(CSV_DIR, exist_ok=True)

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
    """Flush + close CSVs and save plots."""
    try:
        joint_f.close(); cart_f.close(); force_f.close()
    except Exception:
        pass

    # Quick plots if we have data
    if len(_tbuf) > 2:
        t = np.array(_tbuf)

        # Plot forces (Fx,Fy,Fz) and torques (Tx,Ty,Tz)
        f = np.array(_fbuf)  # shape (N,6)
        plt.figure()
        for i, label in enumerate(["Fx","Fy","Fz","Tx","Ty","Tz"]):
            plt.plot(t, f[:, i], label=label)
        plt.xlabel("Time [s]"); plt.ylabel("External wrench (EE frame) [N,Nm]")
        plt.title("K_F_ext_hat_K vs time")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(CSV_DIR, "K_F_ext_hat_K_timeseries.png"), dpi=150)
        plt.close()

        # Plot v_c
        vc = np.array(_vcbuf)  # (N,6)
        plt.figure()
        for i, label in enumerate(["vx","vy","vz","wx","wy","wz"]):
            plt.plot(t, vc[:, i], label=label)
        plt.xlabel("Time [s]"); plt.ylabel("Cartesian twist command")
        plt.title("v_c vs time")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(CSV_DIR, "v_c_timeseries.png"), dpi=150)
        plt.close()

        # Plot qdot
        qd = np.array(_qdbuf)  # (N,7)
        plt.figure()
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
    """Read external wrench in EE frame: K_F_ext_hat_K."""
    global f_left
    try:
        meas = np.array(list(msg.K_F_ext_hat_K), dtype=float)  # [Fx,Fy,Fz,Tx,Ty,Tz]
        # low-pass filter
        # f_left = FILTER_ALPHA * meas + (1.0 - FILTER_ALPHA) * f_left
        f_left = meas 
    except Exception as e:
        rospy.logwarn_throttle(2.0, f"Failed to parse K_F_ext_hat_K: {e}")

def joint_cb(msg: JointState):
    """Keep joint_positions in FR3 order."""
    global joint_positions
    name_to_pos = {n: p for n, p in zip(msg.name, msg.position)}
    try:
        joint_positions = np.array([name_to_pos[j] for j in FR3_JOINTS], dtype=float)
    except KeyError:
        # Ignore partial/early messages until all names are present
        pass

# ======================= Main ================================

def main():
    rospy.init_node("force_vel_mapping_franka")
    rospy.on_shutdown(close_files_and_plots)

    # --- KDL init (URDF → chain/solver) ---
    franka_kdl = JacobianLogger(
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

    # Example desired wrench: only -Z force (e.g., pushing down with 15 N)
    LAMBDA_STAR = np.array([0.0, 0.0, -15.0, 0.0, 0.0, 0.0], dtype=float)

    while not rospy.is_shutdown():
        t_now = rospy.Time.now().to_sec() - t0

        lam_meas = f_left.copy()       # measured external wrench (EE frame)
        lam_star = LAMBDA_STAR.copy()  # desired wrench

        # Default outputs
        q_dot = np.zeros(n_joints, dtype=float)
        v_c   = np.zeros(6, dtype=float)

        if joint_positions is not None and len(joint_positions) == n_joints:
            # --- Z-only force control ---
            force_error = lam_star - lam_meas

            # Keep only the Z-component of force, zero out all others
            force_error[0:2] = 0.0  # no X/Y forces
            force_error[3:6] = 0.0  # no torques

            # Apply a small deadband for noise in Fz
            if abs(force_error[2]) >= 0.01:
                # Cartesian twist command from force error
                v_c = (0.01 * force_error) / K_VEC

                try:
                    # Get Jacobian from your FrankaKDL class
                    J = franka_kdl.compute_jacobian(joint_positions)  # 6x7 matrix

                    # Convert Cartesian velocity to joint velocities
                    q_dot = np.clip(
                        np.linalg.pinv(J) @ v_c,
                        -MAX_JOINT_VEL,
                        MAX_JOINT_VEL
                    )
                except Exception as e:
                    rospy.logwarn_throttle(1.0, f"Jacobian/command error: {e}")
                    q_dot = np.zeros(n_joints, dtype=float)

        # Publish to your C++ velocity controller
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
