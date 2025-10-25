#!/usr/bin/env python3
"""
trajectory_planner.py

Utility class for generating quintic polynomial and SLERP trajectories
for joints, Cartesian position, and orientation.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from typing import Tuple

class TrajectoryPlanner:
    """
    Generates time parameterized trajectories with smooth boundary
    conditions (zero velocity/acceleration at endpoints).
    """

    def __init__(self):
        pass

    def quintic_joint_trajectory(
        self,
        start_joints: np.ndarray,
        goal_joints: np.ndarray,
        T: float,
        dt: float = 0.001
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Quintic polynomial trajectory for n DOF joint space.

        Parameters
        ----------
        start_joints : (n,) array
            Initial joint angles.
        goal_joints  : (n,) array
            Target joint angles.
        T            : float
            Total duration [s].
        dt           : float
            Time step [s].

        Returns
        -------
        positions    : (N,n) array
            Joint angles over time.
        velocities   : (N,n) array
            Joint velocities over time.
        accelerations: (N,n) array
            Joint accelerations over time.

        N = int(T/dt) + 1
        """
        # time vector
        t = np.arange(0.0, T + dt, dt)
        N = t.size

        # BCs: position, velocity, acceleration at start/end
        J0, Jf = start_joints, goal_joints
        V0 = np.zeros_like(J0)
        Vf = np.zeros_like(J0)
        A0 = np.zeros_like(J0)
        Af = np.zeros_like(J0)

        # Quintic coefficient matrix (6×6)
        M = np.array([
            [1,    0,     0,      0,       0,        0],
            [1,    T,   T**2,    T**3,    T**4,     T**5],
            [0,    1,     0,      0,       0,        0],
            [0,    1,   2*T,    3*T**2,  4*T**3,   5*T**4],
            [0,    0,     2,      0,       0,        0],
            [0,    0,     2,    6*T,    12*T**2,  20*T**3]
        ])

        n = J0.size
        coeffs = np.zeros((n, 6))  # one row per joint

        # Solve for each joint's quintic coefficients
        for i in range(n):
            b = np.array([J0[i], Jf[i], V0[i], Vf[i], A0[i], Af[i]])
            coeffs[i, :] = np.linalg.solve(M, b)

        # allocate output
        positions     = np.zeros((N, n))
        velocities    = np.zeros((N, n))
        accelerations = np.zeros((N, n))

        # evaluate polynomial and its derivatives at each time
        for idx, ti in enumerate(t):
            Tvec   = np.array([1, ti, ti**2, ti**3, ti**4, ti**5])
            dTvec  = np.array([0, 1,  2*ti, 3*ti**2, 4*ti**3, 5*ti**4])
            ddTvec = np.array([0, 0,      2,   6*ti,  12*ti**2, 20*ti**3])

            positions[idx, :]     = coeffs @ Tvec
            velocities[idx, :]    = coeffs @ dTvec
            accelerations[idx, :] = coeffs @ ddTvec

        return positions, velocities, accelerations

    def quintic_position_trajectory(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        T: float,
        dt: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Quintic trajectory in Cartesian space (x,y,z).

        Same form as joint version but fixed 3‐dimensional.
        """
        # reuse joint‐trajectory code with dim=3
        return self.quintic_joint_trajectory(
            start_pos, goal_pos, T, dt
        )

    def slerp_orientation_trajectory(
        self,
        start_quat: np.ndarray,
        goal_quat: np.ndarray,
        T: float,
        dt: float = 0.01
    ) -> np.ndarray:
        """
        SLERP trajectory for quaternions.

        Parameters
        ----------
        start_quat : (4,) array
            Initial [x,y,z,w].
        goal_quat  : (4,) array
            Final [x,y,z,w].
        T          : float
            Duration [s].
        dt         : float
            Timestep [s].

        Returns
        -------
        orientations : (N,4) array
            Interpolated quaternions [x,y,z,w] over time.
        """
        # time samples
        t = np.arange(0.0, T + dt, dt)
        # key frames at t=0 and t=T
        key_times = [0.0, T]
        key_rots = R.from_quat([start_quat, goal_quat])
        slerp    = Slerp(key_times, key_rots)
        interp   = slerp(t)
        return interp.as_quat()
