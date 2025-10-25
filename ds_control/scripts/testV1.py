#!/usr/bin/env python3

import threading
import rospy
import numpy as np
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose, PoseStamped
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest, LoadController

from ds_control.robot_state   import RobotState
from ds_control.kdl_ik_solver import DLSIKSolver
from ds_control.ds_position   import CoupledDSSynchronizer
from ds_control.dls_velocity  import DLSVelocityCommander
from ds_control.sync_ds       import DualArmController

class TestController:
    def __init__(self):
        
        rospy.init_node('test', anonymous=True)
        
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
        
        self.async_ds = DualArmController(2.0, 5.0, 2.0, 2.0)
        self.async_ds.set_mode("asynchronous")
        

        # self.x_final_heal = np.array([-0.4, 0.21264464830453816, 0.5064464265917316])
        self.x_final_heal = np.array([-0.4794363103357136, 0.21264464830453816, 0.5064464265917316])
        self.quat_final_heal = np.array([0.0053085034719848835, 0.0002831146616524242, 0.27632553233987667, 0.9610493950958298]) 
        
        self.x_final_fr3 = np.array([0.2029472579377401, -0.6856056884065369, 0.27533198430149775])
        self.quat_final_fr3 = np.array([-0.4842684342986501, -0.5004146650146468, 0.5206964142174111, -0.49390737066918033])      
        # self.quat_final_fr3 = np.array([-0.4842684342986501, -0.5004146650146468, 0.5206964142174111, -0.49390737066918033])      


        self.heal_arrived = False
        self.fr3_arrived = False
        
        self.pos_thresh = 0.005
        
        self.fr3_comm = DLSVelocityCommander(
            robot_state            = self.fr3_state,
            ik_solver              = self.fr3_ik,
            custom_ds              = self.fr3_twist,
            joint_state_topic      = "/fr3/joint_states",
            ee_pose_topic          = "/fr3/ee_pose",
            ee_pose_msg_type       = Pose,
            velocity_command_topic = "/fr3/joint_velocity_controller/joint_velocity_command",
            max_cartesian_vel      = 0.05,
            max_angular_vel        = 0.1
        )
        
        self.heal_comm = DLSVelocityCommander(
            robot_state            = self.heal_state,
            ik_solver              = self.heal_ik,
            custom_ds              = self.heal_twist,
            joint_state_topic      = "/heal/joint_states",
            ee_pose_topic          = "/heal/ee_pose",
            ee_pose_msg_type       = PoseStamped,
            velocity_command_topic = "/heal/velocity_controller/command",
            max_cartesian_vel      = 0.05,
            max_angular_vel        = 0.05
        )
        
    def heal_twist(self):
        if not self.heal_arrived:
            v_heal, omega_heal, _, _ = self.async_ds.compute_commands(
                self.heal_state.ee_pos, self.fr3_state.ee_pos,  # current position
                self.x_final_heal,      self.x_final_fr3,       # async goal position
                self.x_final_heal,      self.x_final_fr3,       # sync goal position
                self.heal_state.ee_ori, self.fr3_state.ee_ori,  # current orientation
                self.quat_final_heal, [0,0,0,1],                # async goal orientation
                [0,0,0,1], [0,0,0,1],                           # sync goal orientation
                rospy.get_time()
            )
            if np.linalg.norm(self.heal_state.ee_pos - self.x_final_heal) < self.pos_thresh:
                self.heal_arrived = True
            return np.hstack((v_heal, omega_heal))
        return np.zeros(6)
    
    
    def fr3_twist(self):
        if not self.fr3_arrived:
            _, _, v_fr3, omega_fr3 = self.async_ds.compute_commands(
                self.heal_state.ee_pos, self.fr3_state.ee_pos,  # current position
                self.x_final_heal,      self.x_final_fr3,       # async goal position
                self.x_final_heal,      self.x_final_fr3,       # sync goal position
                self.heal_state.ee_ori, self.fr3_state.ee_ori,  # current orientation
                self.quat_final_heal, self.quat_final_fr3,      # async goal orientation
                [0,0,0,1], [0,0,0,1],                           # sync goal orientation
                rospy.get_time()
            )
            if np.linalg.norm(self.fr3_state.ee_pos - self.x_final_fr3) < self.pos_thresh:
                self.fr3_arrived = True
            return np.hstack((v_fr3, omega_fr3))
        return np.zeros(6)
    
    def start(self):
        # threading.Thread(target=self.fr3_comm.run,  name="fr3_loop").start()
        threading.Thread(target=self.heal_comm.run, name="heal_loop").start()

if __name__ == "__main__":
    ctrl = TestController()
    ctrl.start()
    rospy.spin()