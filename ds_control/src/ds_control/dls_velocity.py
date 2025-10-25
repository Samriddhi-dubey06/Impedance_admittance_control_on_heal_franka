# =============================================================================
#  dls_velocity_commander.py
#
#  A ROS‐compatible controller node that:
#    - Subscribes to JointState and end‐effector Pose/PoseStamped messages
#      to keep its RobotState up to date.
#    - Queries a user‐provided “DS” function to obtain a desired 6D Cartesian
#      twist [vx,vy,vz, wx,wy,wz].
#    - Uses a DLSIKSolver to convert that twist into joint velocities (dq).
#    - Publishes dq on a Float64MultiArray to drive a joint‐velocity controller.
# =============================================================================

import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped

from .robot_state    import RobotState
from .kdl_ik_solver  import DLSIKSolver

class DLSVelocityCommander:
    """
    Inverse-kinematics commander using Damped Least Squares (DLS) + a
    custom dynamical system (DS) for Cartesian twist generation.

    Attributes
    ----------
    state : RobotState
        Holds current joint angles and end-effector pose.
    ik_solver : DLSIKSolver
        Computes joint velocities from a 6D twist.
    ds : callable[[], np.ndarray]
        Zero-arg function returning a 6-vector twist.
    pub : rospy.Publisher
        Publishes Float64MultiArray of joint velocities.
    """

    def __init__(self,
                 robot_state: RobotState,
                 ik_solver:   DLSIKSolver,
                 custom_ds:   callable,
                 joint_state_topic:      str,
                 ee_pose_topic:          str,
                 ee_pose_msg_type,
                 velocity_command_topic: str,
                 logger=None,
                 max_cartesian_vel: float = None,
                 max_angular_vel: float = None,):
        """
        Initialize the commander.

        :param robot_state:            RobotState instance for this robot.
        :param ik_solver:              DLSIKSolver instance for this robot.
        :param custom_ds:              Function returning desired twist (shape (6,)).
        :param joint_state_topic:      ROS topic for sensor_msgs/JointState.
        :param ee_pose_topic:          ROS topic for Pose or PoseStamped.
        :param velocity_command_topic: ROS topic for Float64MultiArray commands.
        :param logger:                 Optional logger (defaults to rospy).
        """
        self.state     = robot_state
        self.ik_solver = ik_solver
        self.custom_ds = custom_ds
        self.logger    = logger or rospy

        # Publisher: joint velocity commands
        self.pub = rospy.Publisher(
            velocity_command_topic,
            Float64MultiArray,
            queue_size=1
        )
        
        self.max_cartesian_vel = max_cartesian_vel
        self.max_angular_vel   = max_angular_vel

        # Subscriber: JointState: update RobotState.joint_positions
        rospy.Subscriber(
            joint_state_topic,
            JointState,
            self.state.update_from_joint_state
        )

        # Subscriber: Pose or PoseStamped: update RobotState.ee_pos & ee_ori
        rospy.Subscriber(
            ee_pose_topic,
            ee_pose_msg_type,
            self.state.update_from_pose
        )

    def run(self):
        """
        Main control loop:
          1) Wait until RobotState has both joint angles and EE pose.
          2) At 100 Hz:
             a) Query custom_ds() for [vx,vy,vz, wx,wy,wz].
             b) Call ik_solver.compute_joint_velocity(...), dq.
             c) Publish dq as Float64MultiArray.
        """
        rate = rospy.Rate(100)  # 100 Hz

        # Wait for initial state
        while not rospy.is_shutdown() and not self.state.has_full_state():
            self.logger.loginfo_throttle(
                1.0,
                f"[{self.state.name}] waiting for joint and pose messages..."
            )
            rate.sleep()

        self.logger.loginfo(f"[{self.state.name}] state ready, starting control loop")

        while not rospy.is_shutdown():
            # 1) raw twist from DS
            twist = self.custom_ds()

            # 1a) clamp linear
            lin = twist[:3].copy()
            if self.max_cartesian_vel is not None:
                norm_lin = np.linalg.norm(lin)
                if norm_lin > self.max_cartesian_vel:
                    lin = lin / norm_lin * self.max_cartesian_vel

            # 1b) clamp angular
            ang = twist[3:].copy()
            if self.max_angular_vel is not None:
                norm_ang = np.linalg.norm(ang)
                if norm_ang > self.max_angular_vel:
                    ang = ang / norm_ang * self.max_angular_vel

            # Desired Cartesian twist from DS
            # Expected shape: (6,) = [vx, vy, vz, wx, wy, wz]
            twist = np.hstack([lin, ang])

            # 2) DLS‐IK: map twist to joint velocities dq (shape: n_joints)
            dq = self.ik_solver.compute_joint_velocity(
                self.state.joint_positions,
                twist
            )

            # 3) publish dq
            msg = Float64MultiArray()
            msg.data = dq.tolist()
            self.pub.publish(msg)

            rate.sleep()