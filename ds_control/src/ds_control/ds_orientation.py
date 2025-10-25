# ============================================================================
#  ds_orientation.py
#
#  Coupled orientation DS synchronizer (quaternion version).
# ============================================================================

import numpy as np
from typing import List

class CoupledDSOrientationSynchronizer:
    # ------------------------------------------------------------------ #
    #  Constructor                                                       #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        q1_list: List[np.ndarray],                 # start orientations
        q2_list: List[np.ndarray],                 # goal  orientations
        beta1 : float = 1.0,                       # Eq. 5 slope
        beta2 : float = 0.5,                       # Eq. 5 slope
        A_gain: float = 1.0                        # a  in  A_R = a I₃
    ):
        self.n  = len(q1_list)
        self.q1_list = [self._normalize(q) for q in q1_list]
        self.q2_list = [self._normalize(q) for q in q2_list]

        # Coupling progress scalars (start at 1 → still at q1)
        self.z_rot_list = [1.0 for _ in range(self.n)]

        # Sigmoid parameters for z‑dynamics  (Eq. 5)
        self.beta1 = beta1
        self.beta2 = beta2

        # Orientation gain  A_R  (taken identical for every arm)
        self.A_R = A_gain * np.eye(3)


    # ------------------------------------------------------------------ #
    #  Basic quaternion helpers                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(q: np.ndarray) -> np.ndarray:
        return q / np.linalg.norm(q)

    @staticmethod
    def _quat_conj(q: np.ndarray) -> np.ndarray:
        return np.array([-q[0], -q[1], -q[2],  q[3]])

    @staticmethod
    def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])

    @staticmethod
    def _log_quat(q: np.ndarray) -> np.ndarray:

        q = q / np.linalg.norm(q)
        v, w = q[:3], q[3]
        w = np.clip(w, -1.0, 1.0)
        theta = 2.0 * np.arccos(w)
        if theta < 1e-8:                    # near‑identity → no rotation
            return np.zeros(3)
        return theta * v / np.sin(theta/2.0)

    # ------------------------------------------------------------------ #
    #  Progress computation  (α_i  from Eq. 7 adapted to rotations)      #
    # ------------------------------------------------------------------ #
    def _alpha_rot(self, q, q1, q2) -> float:

        q, q1, q2 = map(self._normalize, (q, q1, q2))

        # Flip to common hemisphere  (shortest‑arc)
        if np.dot(q2, q1) < 0: q1 = -q1
        if np.dot(q2, q)  < 0: q  = -q

        theta_tot = 2*np.arccos(np.clip(abs(self._quat_mul(q2, self._quat_conj(q1))[3]), 0, 1))
        theta_rem = 2*np.arccos(np.clip(abs(self._quat_mul(q2, self._quat_conj(q ))[3]), 0, 1))

        if theta_tot < 1e-6:
            return 0.0
        return float(np.clip(theta_rem/theta_tot, 0.0, 1.0))

    def _all_alpha_rot(self, q_list: List[np.ndarray]) -> List[float]:
        """Compute α_i for every manipulator."""
        return [
            self._alpha_rot(q, q1, q2)
            for q, q1, q2 in zip(q_list, self.q1_list, self.q2_list)
        ]

    # ------------------------------------------------------------------ #
    #  z‑dynamics (Eq. 5) + coupling (Eq. 10)                            #
    # ------------------------------------------------------------------ #
    def _update_z_rot(self, alpha_rot_list: List[float], z_d: float = 0.0):
        z_c = (z_d + sum(alpha_rot_list)) / (self.n + 1)   # Eq. 10
        new_z = []
        for z_i in self.z_rot_list:
            dz = -self.beta1 * (1 - np.exp(-self.beta2*(z_i - z_c))) \
                             / (1 + np.exp(-self.beta2*(z_i - z_c)))    # Eq. 5
            new_z.append(z_i + dz)
        self.z_rot_list = new_z

    # ------------------------------------------------------------------ #
    #  Public:  compute desired body‑frame angular velocities ωᵈ         #
    # ------------------------------------------------------------------ #
    def compute_angular_velocity(self, q_list: List[np.ndarray]) -> List[np.ndarray]:

        # 1.  Progress scalars α_i  (Eq. 7 for rotations)
        alpha_rot = self._all_alpha_rot(q_list)

        # 2.  Update coupled z_i  (Eqs. 5 & 10)
        self._update_z_rot(alpha_rot)

        # 3.  Build ωᵈ using orientation Eq. 6
        omega_list = []
        for i in range(self.n):
            # ------------------------------------------------------------------
            #  Retrieve current / reference quaternions and progress scalar
            # ------------------------------------------------------------------
            q   = self._normalize(q_list[i])   # current
            q1  = self.q1_list[i]              # start
            q2  = self.q2_list[i]              # goal
            z   = self.z_rot_list[i]           # coupled progress

            # ------------------------------------------------------------------
            #  Hemisphere alignment  (*** essential to avoid 180° flips ***)
            # ------------------------------------------------------------------
            if np.dot(q2, q)  < 0.0:  # keep current quaternion on same side as q₂
                q = -q
            if np.dot(q2, q1) < 0.0:  # ensure Δ is the short arc
                q1 = -q1

            # ------------------------------------------------------------------
            #  Fixed start → goal rotation vector  Δ  (Eq. 4 for orientations)
            # ------------------------------------------------------------------
            Delta = self._log_quat(self._quat_mul(q2, self._quat_conj(q1)))

            # ------------------------------------------------------------------
            #  Instantaneous orientation error  e  (body‑frame, axis–angle)
            # ------------------------------------------------------------------
            e = self._log_quat(self._quat_mul(q2, self._quat_conj(q)))

            # ------------------------------------------------------------------
            #  ż_i   from sigmoid (Eq. 5)            — same form as position DS
            # ------------------------------------------------------------------
            dz = -self.beta1 * (1 - np.exp(-self.beta2 * (z - alpha_rot[i]))) \
                           / (1 + np.exp(-self.beta2 * (z - alpha_rot[i])))

            # Optional clamp: once z ≈ 0 lock it at zero to prevent rebound
            if z + dz < 1e-3:
                z, dz = 0.0, 0.0
                self.z_rot_list[i] = 0.0

            # ------------------------------------------------------------------
            #  Desired angular velocity   ωᵈ  (orientation Eq. 6)
            # ------------------------------------------------------------------
            term1 = -(z * self.A_R + dz * np.eye(3)) @ Delta
            term2 = self.A_R @ e
            omega_list.append(term1 + term2)

        return omega_list

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                       #
    # ------------------------------------------------------------------ #
    def get_z_list_rot(self) -> List[float]:
        """Return current rotational progress scalars  z_i  (useful for logs)."""
        return self.z_rot_list