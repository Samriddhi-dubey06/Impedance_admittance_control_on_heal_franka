#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import time
import threading
from datetime import datetime

import rospy
from franka_msgs.msg import FrankaState
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# --- User paths (as requested) ---
BASE_DIR = "/home/iitgn-robotics/ds_yash/bimanual_ws/src//dynamic_box_lifting"
CSV_DIR = os.path.join(BASE_DIR, "csv")
GRAPH_DIR = os.path.join(BASE_DIR, "graph")

TOPIC = "/franka_state_controller/franka_states"
TARGET_RATE_HZ = 100.0

# Thread-safe buffer for latest message
_latest_msg = None
_msg_lock = threading.Lock()

# Buffers for plotting K_F_ext_hat_K (EE-frame external wrench)
_time_buf = []
_fx_buf = []
_fy_buf = []
_fz_buf = []
_tx_buf = []
_ty_buf = []
_tz_buf = []

def ensure_dirs():
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)

def flatten_ros_message(msg, prefix="", out=None):
    """
    Recursively flattens a ROS message (including lists) into a flat dict.
    Keys will be 'prefix.field', lists become 'prefix.field[i]'.
    This dynamically handles "each and every value" without hardcoding fields.
    """
    if out is None:
        out = {}

    # Primitive types
    if isinstance(msg, (int, float, bool, str)):
        out[prefix.rstrip(".")] = msg
        return out

    # Bytes/bytearray
    if isinstance(msg, (bytes, bytearray)):
        out[prefix.rstrip(".")] = msg.decode("utf-8", errors="ignore")
        return out

    # Lists / tuples
    if isinstance(msg, (list, tuple)):
        for i, v in enumerate(msg):
            flatten_ros_message(v, f"{prefix}[{i}].", out)
        return out

    # ROS messages (duck-typing: has __slots__ and _slot_types)
    if hasattr(msg, "__slots__"):
        for slot in msg.__slots__:
            try:
                val = getattr(msg, slot)
            except Exception:
                continue
            flatten_ros_message(val, f"{prefix}{slot}.", out)
        return out

    # Fallback: store string repr
    out[prefix.rstrip(".")] = str(msg)
    return out

def state_cb(msg: FrankaState):
    # Update latest message
    global _latest_msg
    with _msg_lock:
        _latest_msg = msg

    # Append K_F_ext_hat_K to plotting buffers (if present)
    try:
        t = rospy.get_time()
        fx, fy, fz, tx, ty, tz = msg.K_F_ext_hat_K
        _time_buf.append(t)
        _fx_buf.append(fx)
        _fy_buf.append(fy)
        _fz_buf.append(fz)
        _tx_buf.append(tx)
        _ty_buf.append(ty)
        _tz_buf.append(tz)
    except Exception:
        # If field missing/unexpected, just skip plotting buffers
        pass

def write_plots():
    """Save force and torque plots vs time into GRAPH_DIR."""
    if len(_time_buf) < 2:
        rospy.logwarn("Not enough samples to plot K_F_ext_hat_K.")
        return

    # Forces
    plt.figure()
    plt.plot(_time_buf, _fx_buf, label="Fx (N)")
    plt.plot(_time_buf, _fy_buf, label="Fy (N)")
    plt.plot(_time_buf, _fz_buf, label="Fz (N)")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title("K_F_ext_hat_K - Forces vs Time (EE frame)")
    plt.legend()
    plt.tight_layout()
    force_path = os.path.join(GRAPH_DIR, "K_F_ext_hat_K_forces.png")
    plt.savefig(force_path, dpi=150)
    plt.close()

    # Torques
    plt.figure()
    plt.plot(_time_buf, _tx_buf, label="Tx (N·m)")
    plt.plot(_time_buf, _ty_buf, label="Ty (N·m)")
    plt.plot(_time_buf, _tz_buf, label="Tz (N·m)")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (N·m)")
    plt.title("K_F_ext_hat_K - Torques vs Time (EE frame)")
    plt.legend()
    plt.tight_layout()
    torque_path = os.path.join(GRAPH_DIR, "K_F_ext_hat_K_torques.png")
    plt.savefig(torque_path, dpi=150)
    plt.close()

    rospy.loginfo(f"Saved plots:\n  {force_path}\n  {torque_path}")

def main():
    rospy.init_node("franka_states_csv_logger", anonymous=True)
    ensure_dirs()

    # CSV filename with timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(CSV_DIR, f"franka_states_{ts}.csv")

    sub = rospy.Subscriber(TOPIC, FrankaState, state_cb, queue_size=1)

    rate = rospy.Rate(TARGET_RATE_HZ)

    csv_file = open(csv_path, "w", newline="")
    writer = None
    header_written = False

    rospy.loginfo(f"Logging {TOPIC} at {TARGET_RATE_HZ} Hz")
    rospy.loginfo(f"CSV: {csv_path}")
    rospy.loginfo(f"Graphs will be saved in: {GRAPH_DIR} on shutdown (Ctrl+C)")

    try:
        while not rospy.is_shutdown():
            # Snapshot the latest message
            with _msg_lock:
                msg = _latest_msg

            if msg is not None:
                # Flatten the entire message to a row dict
                flat = flatten_ros_message(msg)
                # Add logger time (seconds) for a clean x-axis
                flat["logger_time"] = rospy.get_time()

                if not header_written:
                    # Write header from keys (stable order)
                    fieldnames = list(flat.keys())
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
                    header_written = True

                # Write the row
                writer.writerow(flat)
                csv_file.flush()

            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        try:
            csv_file.close()
        except Exception:
            pass
        # Save plots for K_F_ext_hat_K
        write_plots()
        rospy.loginfo("Shut down cleanly.")

if __name__ == "__main__":
    main()
