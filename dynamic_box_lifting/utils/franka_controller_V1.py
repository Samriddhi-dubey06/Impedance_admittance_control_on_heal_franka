#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import rospy

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import WrenchStamped 
from geometry_msgs.msg import TwistStamped
import math   

import os
import csv


import sys
sys.path.insert(0, '/home/iitgn-robotics/ds_yash/bimanual_ws/src')
# Import your existing KDL wrapper that lives in the same utils/ folder
from .jacobian_franka import FrankaKDL


# ======================= Default Config (override via kwargs) =======================
DEFAULTS = dict(
    urdf="/home/iitgn-robotics/ds_yash/bimanual_ws/src/addverb_heal_description/urdf/fr3.urdf",
    base="fr3_link0",
    tip="fr3_link8",
    joint_names=[
        "fr3_joint1","fr3_joint2","fr3_joint3",
        "fr3_joint4","fr3_joint5","fr3_joint6","fr3_joint7"
    ],
    topic_state="/fr3/franka_state_controller/franka_states",
    topic_wrench="/fr3/wrench_filtered",
    topic_joints="/fr3/joint_states",
    topic_command="/fr3/joint_velocity_controller/joint_velocity_command",
    lambda_star=np.array([0.0, -12.0, 0.0, 0.0, 0.0, 0.0], dtype=float),  # Fy target
    k_vec=np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1], dtype=float),
    max_joint_vel=0.1,  # rad/s
    deadband=0.01,       # |error| threshold
    mu=0.4,          # ADD: friction coefficient
    nmax=20.0,       # ADD: max allowed normal force magnitude (N)
)


class FrankaController:
    """
    Minimal Franka velocity controller using FrankaKDL and a force error on Fy.
    Construct with defaults or override any field in DEFAULTS via kwargs.
    """

    def __init__(self, **kwargs):
        cfg = {**DEFAULTS, **kwargs}
        self.urdf = cfg['urdf']
        self.base = cfg['base']
        self.tip = cfg['tip']
        self.joint_names = cfg['joint_names']
        # self.topic_state = cfg['topic_state']
        self.topic_joints = cfg['topic_joints']
        self.topic_wrench = cfg['topic_wrench']
        self.topic_command = cfg['topic_command']
        self.lambda_star = cfg['lambda_star'].astype(float)
        self.k_vec = cfg['k_vec'].astype(float)
        self.max_joint_vel = float(cfg['max_joint_vel'])
        self.deadband = float(cfg['deadband'])
        self.lambda_star = cfg['lambda_star'].astype(float)

        # ADD: friction and cap
        self.mu = float(cfg['mu'])
        self.nmax = float(cfg['nmax'])

        # ADD: preload N0 from your initial Fy target (|−12| = 12)
        self.N0 = abs(self.lambda_star[1])

        # Guardrail: ensure preload ≤ cap
        if self.N0 > self.nmax:
            rospy.logwarn(f"[FR3] N0 ({self.N0:.2f}) > nmax ({self.nmax:.2f}); clamping preload to nmax.")
            self.N0 = self.nmax

        # State
        self.f_left = np.zeros(6, dtype=float)  # O_F_ext_hat_K (base-frame external wrench)
        self.joint_positions = None             # np.array, size = 7
        self.n_joints = 7
        self.obj_vel_x = 0.0
        
        # === CSV setup (lambda_star and measured wrench) ===
        csv_dir = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/dynamic_box_lifting/csv/"  # your path
        os.makedirs(csv_dir, exist_ok=True)
        self.csv_file = os.path.join(csv_dir, "fr3_force_log.csv")

        # Create file with header if it doesn't exist
        if not os.path.isfile(self.csv_file):
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "time",
                    "lam_star_Fx","lam_star_Fy","lam_star_Fz","lam_star_Tx","lam_star_Ty","lam_star_Tz",
                    "meas_Fx","meas_Fy","meas_Fz","meas_Tx","meas_Ty","meas_Tz"
                ])

        # Log rate control (20 Hz)
        self.log_hz = 20.0
        self.log_dt = 1.0 / self.log_hz
        self.last_log_t = None


        # KDL init via your wrapper
        self.franka_kdl = FrankaKDL(
            urdf_file=self.urdf,
            base_link=self.base,
            tip_link=self.tip,
            joint_names=self.joint_names
        )

        # ROS I/O
        # rospy.Subscriber(self.topic_state,  FrankaState, self._franka_state_cb, queue_size=1)
        rospy.Subscriber(self.topic_wrench, WrenchStamped, self._wrench_cb, queue_size=1)
        rospy.Subscriber(self.topic_joints, JointState,   self._joint_cb,       queue_size=1)
        rospy.Subscriber("/object_velocity_cmd", TwistStamped, self._obj_vel_cb, queue_size=1)

        self.pub = rospy.Publisher(self.topic_command, Float64MultiArray, queue_size=1)

    # -------------------- Callbacks --------------------
    def _franka_state_cb(self, msg: FrankaState):
        try:
            meas = np.array(list(msg.O_F_ext_hat_K), dtype=float)  # [Fx,Fy,Fz,Tx,Ty,Tz]
            self.f_left = meas
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"[FR3] Failed to parse O_F_ext_hat_K: {e}")
    def _wrench_cb(self, msg: WrenchStamped):
        # msg.wrench.force.{x,y,z}, msg.wrench.torque.{x,y,z}
        fx = msg.wrench.force.x
        fy = msg.wrench.force.y
        fz = msg.wrench.force.z
        tx = msg.wrench.torque.x
        ty = msg.wrench.torque.y
        tz = msg.wrench.torque.z
        self.f_left = np.array([fx, fy, fz, tx, ty, tz], dtype=float)


    def _joint_cb(self, msg: JointState):
        name_to_pos = {n: p for n, p in zip(msg.name, msg.position)}
        name_to_vel = {n: v for n, v in zip(msg.name, msg.velocity)}
        try:
            self.joint_positions = np.array([name_to_pos[j] for j in self.joint_names], dtype=float)
            self.joint_velocities = np.array([name_to_vel[j] for j in self.joint_names], dtype=float)

        except KeyError:
            pass
        
    def _obj_vel_cb(self, msg: TwistStamped):
        """Store only the linear.x component from /object_velocity_cmd"""
        self.obj_vel_x = msg.twist.linear.x
        rospy.logdebug_throttle(1.0, f"[FR3] Received object vx = {self.obj_vel_x:.3f}")

    # -------------------- Control Step -----------------
    def step(self):
        """One control tick; Fy-only control with damping."""
        lam_meas = self.f_left.copy()

        # === Compute required normal from tangential load (friction cone) ===
        # Frame assumption: y = contact normal (compression = negative Fy).
        Fx = float(lam_meas[0])
        Fy = float(lam_meas[1])
        Fz = float(lam_meas[2])

        # Tangential magnitude
        Ft = math.sqrt(Fx*Fx + Fz*Fz)

        # Required normal (positive scalar) from friction, with preload and cap
        if self.mu <= 0.0:
            rospy.logwarn_throttle(2.0, "[FR3] mu <= 0; using preload only.")
            N_req_ideal = 0.0
        else:
            N_req_ideal = Ft / self.mu

        N_req = max(self.N0, N_req_ideal)          # keep contact
        N_req_clamped = min(N_req, self.nmax)      # obey upper bound

        # Integer-only command: round magnitude UP (ceiling)
        # (13.2 -> 14, 13.6 -> 14, 17.4 -> 18)
        N_int = int(math.ceil(N_req_clamped))

        # Build desired wrench lambda_star: only Fy active (negative for compression)
        lam_star = np.zeros(6, dtype=float)
        lam_star[1] = -float(N_int)

        # Optionally store it so CSV logs the updated command too
        self.lambda_star = lam_star.copy()
        
                # === Log at ~20 Hz: lambda_star (command) and measured wrench from topic ===
        t_now = rospy.Time.now().to_sec()
        if (self.last_log_t is None) or (t_now - self.last_log_t >= self.log_dt):
            try:
                with open(self.csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        t_now,
                        self.lambda_star[0], self.lambda_star[1], self.lambda_star[2],
                        self.lambda_star[3], self.lambda_star[4], self.lambda_star[5],
                        lam_meas[0], lam_meas[1], lam_meas[2],
                        lam_meas[3], lam_meas[4], lam_meas[5],
                    ])
                self.last_log_t = t_now
            except Exception as e:
                rospy.logwarn_throttle(2.0, f"[FR3] CSV log failed: {e}")


        q_dot = np.zeros(self.n_joints, dtype=float)

        if self.joint_positions is not None and len(self.joint_positions) == self.n_joints:
            try:
                # Compute Jacobian and end-effector velocity
                J = self.franka_kdl.compute_jacobian(self.joint_positions)  # 6x7
                v_ee = J @ self.joint_velocities   # 6x1

                # Force error (only Fy active)
                force_error = lam_star - lam_meas
                force_error[0] = 0.0   # no Fx
                force_error[2] = 0.0   # no Fz
                force_error[3:6] = 0.0 # no torques

                # if abs(force_error[1]) >= self.deadband:
                v_force = (0.001 * force_error) / self.k_vec 
                    # NEW: object velocity contribution
                v_obj = np.zeros(6)
                v_obj[2] = self.obj_vel_x   # linear.x only 
                    # Final control twist
                    # copy so you don’t overwrite the raw v_obj
                v_obj_scaled = v_obj.copy()

                    # scale only the lifting direction (z, index 2)
                lift_scale = 0.12   # e.g. 20% of original speed
                v_obj_scaled[2] *= lift_scale

                    # combine
                v_c =  v_obj_scaled
                v_final = v_force + v_c

                    
                q_dot = np.clip(np.linalg.pinv(J) @ v_final,
                                    -self.max_joint_vel, self.max_joint_vel)
            except Exception as e:
                rospy.logwarn_throttle(1.0, f"[FR3] Jacobian/command error: {e}")
                q_dot = np.zeros(self.n_joints, dtype=float)

        self.pub.publish(Float64MultiArray(data=q_dot.tolist()))
