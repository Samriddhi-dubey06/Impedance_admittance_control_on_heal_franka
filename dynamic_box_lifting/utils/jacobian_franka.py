#!/usr/bin/env python3
import os
import re
import numpy as np
import PyKDL as kdl
import xacro
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromString

class FrankaKDL:
    """
    Minimal KDL helper:
      - loads URDF/XACRO from disk (once)
      - builds KDL chain (base_link -> tip_link)
      - compute_jacobian(q) returns 6xN Jacobian at joint vector q
    No ROS init, no subscribers.
    """
    def __init__(self, urdf_file: str, base_link: str, tip_link: str, joint_names):
        if not os.path.exists(urdf_file):
            raise FileNotFoundError(f"URDF file not found: {urdf_file}")

        # Load URDF text (support .xacro)
        if urdf_file.endswith(".xacro"):
            urdf_str = xacro.process_file(urdf_file).toxml()
        else:
            with open(urdf_file, "r") as f:
                urdf_str = f.read()

        # Strip non-standard dynamics attributes to silence parser warnings
        urdf_str = re.sub(r'\s+(D|K|mu_coulomb|mu_viscous)="[^"]*"', '', urdf_str)

        ok, tree = treeFromString(urdf_str)
        if not ok:
            raise RuntimeError("Failed to build KDL tree from URDF")

        self.chain = tree.getChain(base_link, tip_link)
        self.n = self.chain.getNrOfJoints()

        # Keep joint names (ordering must match the chain)
        self.joint_names = list(joint_names)

        # Optional: sanity check length
        if len(self.joint_names) != self.n:
            # Not fatal, but warn in logs if you like
            pass

        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)

    def compute_jacobian(self, q_np: np.ndarray) -> np.ndarray:
        """Return 6xN Jacobian for joint positions q_np (len N)."""
        if len(q_np) != self.n:
            raise ValueError(f"Expected {self.n} joint positions, got {len(q_np)}")

        q = kdl.JntArray(self.n)
        for i, qi in enumerate(q_np):
            q[i] = float(qi)

        J_kdl = kdl.Jacobian(self.n)
        self.jac_solver.JntToJac(q, J_kdl)

        J = np.zeros((6, self.n))
        for r in range(6):
            for c in range(self.n):
                J[r, c] = J_kdl[r, c]
        return J
