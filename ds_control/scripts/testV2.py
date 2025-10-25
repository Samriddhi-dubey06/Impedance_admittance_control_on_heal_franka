#!/usr/bin/env python3

import threading
import rospy
import numpy as np
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose, PoseStamped
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest, LoadController

from ds_control.robot_state   import RobotState
from ds_control.kdl_ik_solver import DLSIKSolver
from ds_control.ds_coordination import DSCoordinationFramework
from ds_control.dls_velocity  import DLSVelocityCommander

class TestController:
    def __init__(self):
        
        rospy.init_node('test', anonymous=True)
        
        # Prepare twist log file
        log_path = '/home/iitgn-robotics/bimanual_ws/src/ds_control/data/twist_log.csv'
        self._twist_file = open(log_path, 'w')
        # CSV header: time, heal_vx, heal_vy, heal_vz, heal_wx, heal_wy, heal_wz, fr3_vx, ...
        header = ['time']
        for agent in ['heal','fr3']:
            header += [f'{agent}_v{x}' for x in ['x','y','z']]
            header += [f'{agent}_w{x}' for x in ['x','y','z']]
        self._twist_file.write(','.join(header) + '\n')
        self._twist_file.flush()
        self._twist_lock = threading.Lock()
        rospy.on_shutdown(self._close_twist_log)
        
        franka_joints = [f"fr3_joint{i}" for i in range(1, 8)]
        heal_joints   = ["Joint_1", "Joint_2", "Joint_3",
                        "Joint_4", "Joint_5", "Joint_6"]
        
        self.fr3_state  = RobotState('franka', franka_joints, logger=rospy)
        self.heal_state = RobotState('heal',   heal_joints,   logger=rospy)
        
        self.fr3_ik     = DLSIKSolver(
            urdf_param  = "/fr3/robot_description",
            base_link   = "fr3_link0",
            tip_link    = "fr3_link8",
            joint_names = franka_joints,
            damping     = 0.01
        )
        self.heal_ik    = DLSIKSolver(
            urdf_param  = "/heal/robot_description",
            base_link   = "base_link",
            tip_link    = "tool_ff",
            joint_names = heal_joints,
            damping     = 0.10
        )
        
        rospy.loginfo("Waiting for initial EE poses.")
        fr3_msg  = rospy.wait_for_message("/fr3/ee_pose",  Pose)
        heal_msg = rospy.wait_for_message("/heal/ee_pose", PoseStamped)
        self.fr3_state.update_from_pose(fr3_msg)
        self.heal_state.update_from_pose(heal_msg)
        rospy.loginfo("Received initial EE poses.")
        
        x_start_fr3 = self.fr3_state.ee_pos.copy()
        q_start_fr3 = self.fr3_state.ee_ori.copy()
        x_goal_fr3 = np.array([0.21054845253302076, -0.5925971092302784, 0.3101385754541805])
        # x_goal_fr3 = x_start_fr3
        q_goal_fr3 = np.array([-0.4848103881155475, -0.5052214683928531, 0.5119012469523467, -0.49766180164731777])
        # q_goal_fr3 = q_start_fr3
        
        x_start_heal= self.heal_state.ee_pos.copy()
        q_start_heal= self.heal_state.ee_ori.copy()
        # x_goal_heal = np.array([-0.4, 0.21264464830453816, 0.5064464265917316])
        x_goal_heal= np.array([-0.4794363103357136, 0.21264464830453816, 0.5064464265917316])
        q_goal_heal = np.array([0.0053085034719848835, 0.0002831146616524242, 0.27632553233987667, 0.9610493950958298]) 
        
        # instantiate DSCoordinationFramework
        self.coord_ds = DSCoordinationFramework(
            x1_list   = [x_start_heal, x_start_fr3],
            x2_list   = [x_goal_heal,  x_goal_fr3],
            q1_list   = [q_start_heal, q_start_fr3],
            q2_list   = [q_goal_heal,  q_goal_fr3],
            k1        = 5.0,
            k2        = 10.0,
            kappa     = 1.0,
            k_rot     = 2.0,
            alpha_min = 1.0,
            alpha_max = 5.0,
        )

        self.heal_comm = DLSVelocityCommander(
            robot_state            = self.heal_state,
            ik_solver              = self.heal_ik,
            custom_ds              = self.twist_fn(index=0),
            joint_state_topic      = "/heal/joint_states",
            ee_pose_topic          = "/heal/ee_pose",
            ee_pose_msg_type       = PoseStamped,
            velocity_command_topic = "/heal/velocity_controller/command",
            max_cartesian_vel      = 0.05,
            max_angular_vel        = 0.05,
        )

        self.fr3_comm = DLSVelocityCommander(
            robot_state            = self.fr3_state,
            ik_solver              = self.fr3_ik,
            custom_ds              = self.twist_fn(index=1),
            joint_state_topic      = "/fr3/joint_states",
            ee_pose_topic          = "/fr3/ee_pose",
            ee_pose_msg_type       = Pose,
            velocity_command_topic = "/fr3/joint_velocity_controller/joint_velocity_command",
            max_cartesian_vel      = 0.05,
            max_angular_vel        = 0.2,
        )
        
    def twist_fn(self, index: int):
        """
        Returns a zero arg function for DLSVelocityCommander that:
          1) pulls both current EE positions,
          2) calls coord_ds.compute_velocity(...),
          3) picks out the linear part for arm 'index',
          4) appends a zero angular velocity.
        """
        def twist():
            x_heal = self.heal_state.ee_pos
            x_fr3  = self.fr3_state.ee_pos
            q_heal = self.heal_state.ee_ori
            q_fr3  = self.fr3_state.ee_ori

            twists = self.coord_ds.compute_twists(
                [x_heal, x_fr3],
                [q_heal, q_fr3],
                self.coord_ds.dt
            )

            with self._twist_lock:
                try:
                    if not self._twist_file.closed:
                        t = rospy.get_time()
                        row = [f"{t:.6f}"] + [f"{v:.6f}" for v in np.hstack(twists)]
                        self._twist_file.write(','.join(row) + '\n')
                        self._twist_file.flush()
                except Exception:
                    pass

            return twists[index]
        return twist
    
    def _close_twist_log(self):
        try:
            self._twist_file.close()
        except Exception:
            pass
        
    def start(self):
        threading.Thread(target=self.heal_comm.run, name="heal_loop").start()
        threading.Thread(target=self.fr3_comm.run,  name="fr3_loop").start()

if __name__ == "__main__":
    ctrl = TestController()
    ctrl.start()
    rospy.spin()