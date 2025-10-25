# ============================================================================
#  robot_state.py
#
#  Encapsulates joint‐state and end‐effector pose for a single robot.
#  Provides clean ROS state update methods.
# ============================================================================

import numpy as np
from typing import List, Union
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped

class RobotState:
    """
    Holds current joint angles and EE pose for one manipulator,
    and provides methods to update from ROS messages.
    """

    def __init__(self, name: str, joint_names: List[str], logger=None):
        """
        :param name:         Identifier for logging (e.g. "franka", "heal").
        :param joint_names:  Ordered list of joint names in the kinematic chain.
        :param logger:       Optional logging object (e.g. rospy).
        """
        self.name        = name
        self.joint_names = joint_names
        self.logger      = logger

        # --------------------------------------------------------------------
        #  Current joint positions (shape: n_joints), initialized to zeros.
        # --------------------------------------------------------------------
        self.joint_positions = np.zeros(len(joint_names), dtype=float)

        # --------------------------------------------------------------------
        #  Current end‐effector pose in world frame:
        #    - ee_pos:  np.array([x, y, z])
        #    - ee_ori:  np.array([qx, qy, qz, qw])
        #  Remain None until first Pose/PoseStamped arrives.
        # --------------------------------------------------------------------
        self.ee_pos = None
        self.ee_ori = None

    def update_from_joint_state(self, msg: JointState) -> None:
        """
        Extract and store joint positions from a JointState message.
        Missing joints are left at their previous values.
        """
        # build map: joint name -- index in msg.position
        name_to_idx = {n: i for i, n in enumerate(msg.name)}

        for idx, jn in enumerate(self.joint_names):
            if jn in name_to_idx:
                self.joint_positions[idx] = msg.position[name_to_idx[jn]]
            else:
                warn = f"[{self.name}] warning: joint '{jn}' not in JointState"
                if self.logger:
                    self.logger.logwarn(warn)
                else:
                    print(warn)

    def update_from_pose(self, msg: Union[Pose, PoseStamped]) -> None:
        """
        Extract and store EE pose from a Pose or PoseStamped message.
        Always sets:
          - self.ee_pos = np.array([x,y,z])
          - self.ee_ori = np.array([qx,qy,qz,qw])
        """
        # unwrap PoseStamped - Pose
        pose = msg.pose if isinstance(msg, PoseStamped) else msg

        p = pose.position
        self.ee_pos = np.array([p.x, p.y, p.z], dtype=float)

        o = pose.orientation
        self.ee_ori = np.array([o.x, o.y, o.z, o.w], dtype=float)

    def has_full_state(self) -> bool:
        """
        :returns: True once we've received both JointState and EE pose.
        """
        return (self.ee_pos is not None
                and self.ee_ori is not None
                and self.joint_positions is not None)
