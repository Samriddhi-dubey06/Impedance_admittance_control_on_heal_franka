import numpy as np
from typing import List
import rospy

class CoupledDSSynchronizer:
    """
    Synchronizes multiple end-effector positions so each reaches its
    own goal at the same time by adapting each phase variable z_i.

    Based on:
      Khadivar & Billard, "Adaptive Fingers Coordination for Robust Grasp and
      In-Hand Manipulation Under Disturbances and Unknown Dynamics," IEEE TRO, 2023.

    Attributes:
        x1_list       List of start positions (each np.ndarray of shape (d,))
        x2_list       List of goal  positions (each np.ndarray of shape (d,))
        zd            Desired common relative phase (scalar in [0,1])
        beta1_list    DS convergence rates (list of positive floats)
        beta2_list    DS sigmoid slopes      (list of positive floats)
        A_x           Array of shape (n,d,d): A_x[i] = A_gain[i] * I_d
        z_list        List of per-finger phase variables, initialized on first call
        dt            Integration time step in seconds
    """

    def __init__(
        self,
        x1_list: List[np.ndarray],
        x2_list: List[np.ndarray],
        zd: float,
        beta1_list: List[float],
        beta2_list: List[float],
        A_gain: List[float],
        dt: float = None,
    ):
        # Basic consistency checks
        self.n = len(x1_list)
        if not (len(x2_list) == self.n == len(beta1_list) == len(beta2_list) == len(A_gain)):
            raise ValueError("All input lists (x1_list, x2_list, beta1_list, beta2_list, A_gain) must have the same length n")

        self.d = x1_list[0].shape[0]
        self.x1_list = x1_list
        self.x2_list = x2_list
        self.zd = float(zd)
        self.beta1_list = beta1_list
        self.beta2_list = beta2_list

        # Build A_x blocks: shape (n, d, d)
        self.A_x = np.array([gain * np.eye(self.d) for gain in A_gain])

        # Phase variables will be initialized to the current α on first compute_velocity call
        self.z_list = None

        # Integration timestep
        self.dt = dt if dt is not None else 1.0 / 100.0

    def compute_alpha(self, x: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Eq. (7): relative progress alpha in [0,1].
        alpha = 1 when x == x1, alpha = 0 when x == x2.
        Returns 0.0 if x1 == x2 to avoid division by zero.
        """
        diff = x2 - x1
        denom = np.dot(diff, diff)
        if denom < 1e-8:
            return 0.0

        numer = np.dot(x2 - x, diff)
        alpha = numer / denom
        # clamp for safety
        return float(np.clip(alpha, 0.0, 1.0))

    def compute_velocity(
        self,
        x_list: List[np.ndarray],
        dt: float = None
    ) -> List[np.ndarray]:
        """
        Given current positions x_list = [x_i], compute the list of
        desired velocities [dx_i] for each end-effector.

        Args:
            x_list: Current positions for each finger, length must be n.
            dt:     Optional override for the integration timestep.

        Returns:
            v_list: List of np.ndarray velocities for each finger.
        """
        if len(x_list) != self.n:
            raise ValueError(f"x_list must have length {self.n}")

        dt = dt if dt is not None else self.dt

        # 1) Measure progress alpha_i
        alphas = [
            self.compute_alpha(xi, x1, x2)
            for xi, x1, x2 in zip(x_list, self.x1_list, self.x2_list)
        ]

        # 2) Compute the shared coupling variable z_c (Eq. 10)
        z_c = (self.zd + sum(alphas)) / (self.n + 1)

        v_list: List[np.ndarray] = []
        for i in range(self.n):
            # DS update (Eq. 5) with alpha_i and common z_c
            delta = alphas[i] - z_c
            exp_term = np.exp(-self.beta2_list[i] * delta)
            dz = -self.beta1_list[i] * (1 - exp_term) / (1 + exp_term)

            # integrate z_i and clamp to [0,1]
            self.z_list[i] = np.clip(self.z_list[i] + dz * dt, 0.0, 1.0)

            # compute task-space velocity (Eq. 6)
            x, x1, x2 = x_list[i], self.x1_list[i], self.x2_list[i]
            dx = (
                -(self.z_list[i] * self.A_x[i] + dz * np.eye(self.d)) @ (x2 - x1)
                - self.A_x[i] @ (x - x2)
            )

            rospy.logdebug(
                f"[CoupledDS] idx={i} alpha={alphas[i]:.3f} z={self.z_list[i]:.3f} "
                f"z_c={z_c:.3f} dz={dz:.3f} dx={dx.tolist()}"
            )
            v_list.append(dx)

        return v_list