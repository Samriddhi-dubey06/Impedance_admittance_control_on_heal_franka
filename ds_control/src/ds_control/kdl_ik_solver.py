# ============================================================================
#  kdl_ik_solver.py
#
#  Damped Least‐Squares Inverse Kinematics using PyKDL.
#  Maps a 6D Cartesian twist to joint velocity vector.
# ============================================================================

import rospy
import numpy as np
import PyKDL as kdl
from typing import List
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromParam

class DLSIKSolver:
    """
    Wraps PyKDL ChainFkSolver & ChainJntToJacSolver to provide
    DLS-IK: dq = J^T (J J^T + λ² I)^-1 v_cartesian.
    """

    def __init__(
        self,
        urdf_param:  str,
        base_link:   str,
        tip_link:    str,
        joint_names: List[str],
        damping:     float = 0.01
    ):
        """
        :param urdf_param:  ROS param name for robot_description
        :param base_link:   root link of kinematic chain
        :param tip_link:    end-effector link
        :param joint_names: ordered joint names matching the chain
        :param damping:     lambda for DLS regularization (smaller: aggressive)
        """
        # --------------------------------------------------------------------
        #  Load URDF and build KDL tree
        # --------------------------------------------------------------------
        if not rospy.has_param(urdf_param):
            raise RuntimeError(f"URDF param '{urdf_param}' not found on parameter server")

        self.robot = URDF.from_parameter_server(urdf_param)
        success, tree = treeFromParam(urdf_param)
        if not success:
            raise RuntimeError("Failed to construct KDL tree from URDF")

        # --------------------------------------------------------------------
        #  Extract chain & solvers
        # --------------------------------------------------------------------
        self.chain     = tree.getChain(base_link, tip_link)
        self.n_joints  = self.chain.getNrOfJoints()
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.jac_solver= kdl.ChainJntToJacSolver(self.chain)

        # --------------------------------------------------------------------
        #  Validate joint_names length
        # --------------------------------------------------------------------
        if len(joint_names) != self.n_joints:
            raise ValueError("joint_names length != chain joint count")
        self.joint_names = joint_names

        # --------------------------------------------------------------------
        #  Damping coefficient
        # --------------------------------------------------------------------
        self.damping = damping

    def compute_joint_velocity(
        self,
        q_current: np.ndarray,  # shape: (n_joints,)
        twist:     np.ndarray   # shape: (6,) = [vx,vy,vz, wx,wy,wz]
    ) -> np.ndarray:
        """
        :returns: dq (shape n_joints) such that end-effector moves with given twist.
        """
        # --------------------------------------------------------------------
        #  1) Build KDL JntArray from numpy angles
        # --------------------------------------------------------------------
        jnt = kdl.JntArray(self.n_joints)
        for i, qi in enumerate(q_current):
            jnt[i] = float(qi)

        # --------------------------------------------------------------------
        #  2) Compute Jacobian J (6×n)
        # --------------------------------------------------------------------
        jac_kdl = kdl.Jacobian(self.n_joints)
        self.jac_solver.JntToJac(jnt, jac_kdl)

        J = np.zeros((6, self.n_joints))
        for r in range(6):
            for c in range(self.n_joints):
                J[r, c] = jac_kdl[r, c]

        # --------------------------------------------------------------------
        #  3) DLS pseudo‐inverse: J^T (J J^T + λ² I)^-1
        # --------------------------------------------------------------------
        λ2      = self.damping ** 2
        JJt     = J @ J.T
        inv_term= np.linalg.inv(JJt + λ2 * np.eye(6))
        J_dls   = J.T @ inv_term

        # --------------------------------------------------------------------
        #  4) Multiply by twist to get joint velocities
        # --------------------------------------------------------------------
        dq = J_dls @ twist
        return dq
