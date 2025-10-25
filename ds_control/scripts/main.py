#!/usr/bin/env python3
"""
main.py — Two-Phase Coordinated DS + Snap-Triggered VIC with Two Intermediates
"""

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

class MainController:
    def __init__(self):
        # 1) ROS setup
        rospy.init_node('main_controller', anonymous=True)
        
        franka_joints = [f"fr3_joint{i}" for i in range(1, 8)]
        heal_joints   = ["Joint_1", "Joint_2", "Joint_3",
                        "Joint_4", "Joint_5", "Joint_6"]
        
        # 2) RobotState & IK for each arm
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
        
        # 3) Wait for initial EE poses
        rospy.loginfo("Waiting for initial EE poses.")
        fr3_msg  = rospy.wait_for_message("/fr3/ee_pose",  Pose)
        heal_msg = rospy.wait_for_message("/heal/ee_pose", PoseStamped)
        self.fr3_state.update_from_pose(fr3_msg)
        self.heal_state.update_from_pose(heal_msg)
        rospy.loginfo("Received initial EE poses.")
        
        # Two intermediate targets configured by user
        self.intermediate1 = [
            np.array([0.4, 0.1, 0.1]),
            np.array([-0.3, 0.3, 0.3])
        ]
        self.intermediate2 = [
            np.array([0.4, -0.4, 0.3]),
            np.array([-0.3,  0.3, 0.5])
        ]

        # 4) Phase 1 Coupled DS to intermediate1
        self.syncer = CoupledDSSynchronizer(
            x1_list    = [self.fr3_state.ee_pos, self.heal_state.ee_pos],
            x2_list    = self.intermediate1,
            zd         = 0.0,
            beta1_list = [0.1, 0.1],
            beta2_list = [8.0, 4.0],
            A_gain     = [5.0, 5.0],
            dt         = 0.01
        )
        # initialize DS blending variable z_list
        x_list = [self.fr3_state.ee_pos, self.heal_state.ee_pos]
        self.syncer.z_list = [
            self.syncer.compute_alpha(x_list[i], self.syncer.x1_list[i], self.syncer.x2_list[i])
            for i in range(self.syncer.n)
        ]

        # phase = 1 (to intermediate1), 2 (to intermediate2), 3 (handoff)
        self.phase = 1
        self.pos_thresh = 0.005

        # 5) Async DS controller for HEAL post-handoff
        self.async_ds = DualArmController(2.0, 2.0, 2.0, 2.0)
        self.async_ds.set_mode("asynchronous")
        self.x_final_heal = np.array([-0.4, 0.4, 0.5])

        # Flags and snap trigger
        self.heal_arrived = False
        self.snap = False
        rospy.Subscriber("/snap", Bool, self._snap_callback)
        
        # 6) Load & switch VIC controller
        rospy.loginfo("Waiting for controller services...")
        rospy.wait_for_service('/fr3/controller_manager/load_controller')
        loader = rospy.ServiceProxy('/fr3/controller_manager/load_controller', LoadController)
        try:
            loader('vic_controller')
            rospy.loginfo("Loaded vic_controller")
        except rospy.ServiceException as e:
            rospy.logwarn(f"Could not load vic_controller: {e}")
        rospy.wait_for_service('/fr3/controller_manager/switch_controller')
        self._switch = rospy.ServiceProxy('/fr3/controller_manager/switch_controller', SwitchController)
        rospy.loginfo("Controller services ready.")

        # 7) Velocity commander threads
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
            max_angular_vel        = 0.075
        )

    def _snap_callback(self, msg: Bool):
        if msg.data and not self.snap:
            rospy.loginfo("Snap detected: Freezing Heal")
        self.snap = msg.data

    def fr3_twist(self):
        # coupled DS during phases 1 & 2
        if self.phase in (1, 2):
            v_list = self.syncer.compute_velocity([
                self.fr3_state.ee_pos,
                self.heal_state.ee_pos
            ])
            err_fr3  = np.linalg.norm(self.fr3_state.ee_pos  - self.syncer.x2_list[0])
            err_heal = np.linalg.norm(self.heal_state.ee_pos - self.syncer.x2_list[1])
            if err_fr3 < self.pos_thresh and err_heal < self.pos_thresh:
                if self.phase == 1:
                    rospy.loginfo("Reached intermediate 1, moving to intermediate 2")
                    self.phase = 2
                    # re-init DS for intermediate2
                    self.syncer.x1_list = [self.fr3_state.ee_pos.copy(), self.heal_state.ee_pos.copy()]
                    self.syncer.x2_list = self.intermediate2
                    self.syncer.z_list = [
                        self.syncer.compute_alpha(
                            [self.fr3_state.ee_pos, self.heal_state.ee_pos][i],
                            self.syncer.x1_list[i], self.syncer.x2_list[i]
                        ) for i in range(self.syncer.n)
                    ]
                else:
                    rospy.loginfo("Reached intermediate 2: handoff to VIC")
                    self.phase = 3
                    self._hand_off()
            return np.hstack((v_list[0], np.zeros(3)))
        # phase 3: FR3 under VIC control
        return np.zeros(6)

    def heal_twist(self):
        # coupled DS during phases 1 & 2
        if self.phase in (1, 2):
            v_list = self.syncer.compute_velocity([
                self.fr3_state.ee_pos,
                self.heal_state.ee_pos
            ])
            return np.hstack((v_list[1], np.zeros(3)))
        # asynchronous DS for HEAL in phase 3
        if not self.snap and not self.heal_arrived:
            v, _, _, _ = self.async_ds.compute_commands(
                self.heal_state.ee_pos, self.fr3_state.ee_pos,
                self.x_final_heal,      self.x_final_heal,
                self.x_final_heal,      self.x_final_heal,
                [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1],
                rospy.get_time()
            )
            if np.linalg.norm(self.heal_state.ee_pos - self.x_final_heal) < self.pos_thresh:
                self.heal_arrived = True
            return np.hstack((v, np.zeros(3)))
        return np.zeros(6)

    def _hand_off(self):
        req = SwitchControllerRequest(
            stop_controllers=['joint_velocity_controller'],
            start_controllers=['vic_controller'],
            strictness=SwitchControllerRequest.STRICT
        )
        try:
            self._switch(req)
            rospy.loginfo("VIC started on FR3")
        except rospy.ServiceException as e:
            rospy.logerr(f"SwitchController failed: {e}")

    def start(self):
        threading.Thread(target=self.fr3_comm.run,  name="fr3_loop").start()
        threading.Thread(target=self.heal_comm.run, name="heal_loop").start()

if __name__ == "__main__":
    ctrl = MainController()
    ctrl.start()
    rospy.spin()




# #!/usr/bin/env python3
# """
# main.py — Two Phase Coordinated DS + Snap Triggered VIC
# """

# import threading
# import rospy 
# import numpy as np
# from std_msgs.msg import Bool
# from geometry_msgs.msg import Pose, PoseStamped
# from controller_manager_msgs.srv import (
#     SwitchController,
#     SwitchControllerRequest,
#     LoadController,
# )

# from ds_control.robot_state   import RobotState
# from ds_control.kdl_ik_solver import DLSIKSolver
# from ds_control.ds_position   import CoupledDSSynchronizer
# from ds_control.dls_velocity  import DLSVelocityCommander
# from ds_control.sync_ds       import DualArmController

# class MainController:
#     def __init__(self):
#         # 1) ROS setup
#         rospy.init_node('main_controller', anonymous=True)
        
#         franka_joints = [f"fr3_joint{i}" for i in range(1, 8)]
#         heal_joints   = ["Joint_1", "Joint_2", "Joint_3",
#                         "Joint_4", "Joint_5", "Joint_6"]
        
#         # 2) RobotState & IK for each arm
#         self.fr3_state  = RobotState('franka', franka_joints, logger=rospy)
#         self.heal_state = RobotState('heal',   heal_joints,   logger=rospy)
        
#         self.fr3_ik     = DLSIKSolver(
#             urdf_param  = "/fr3/robot_description",
#             base_link   = "fr3_link0",
#             tip_link    = "fr3_link8",
#             joint_names = franka_joints,
#             damping     = 0.01
#         )
#         self.heal_ik    = DLSIKSolver(
#             urdf_param  = "/heal/robot_description",
#             base_link   = "base_link",
#             tip_link    = "tool_ff",
#             joint_names = heal_joints,
#     r        damping     = 0.10
#         )
        
#         # 3) Wait for initial EE poses
#         rospy.loginfo("Waiting for initial EE poses.")
#         fr3_msg     = rospy.wait_for_message("/fr3/ee_pose",  Pose)
#         heal_msg    = rospy.wait_for_message("/heal/ee_pose", PoseStamped)
#         self.fr3_state.update_from_pose(fr3_msg)
#         self.heal_state.update_from_pose(heal_msg)
#         rospy.loginfo("Recieved initial EE poses.")
        
#         # 4) Phase 1: Coupled DS 
#         x1 = [self.fr3_state.ee_pos.copy(), self.heal_state.ee_pos.copy()]
#         x2 = [
#             np.array([0.4086877711053955, -0.20118787964395773, 0.23837679661108473]),
#             np.array([-0.3164535866927198, 0.32937048323951307, 0.6598576180226702])
#         ]
        
#         # heal intermediate position
#         # position: [-0.3164535866927198, 0.32937048323951307, 0.6598576180226702]
#         # orientation: [0.025824623175741912, 0.7152904272059226, 0.08118310730104568, 0.6936151646802039] (x, y, z, w)

#         self.syncer = CoupledDSSynchronizer(
#             x1_list     = x1, 
#             x2_list     = x2,
#             zd          = 0.0,
#             beta1_list  = [0.1,0.1],
#             beta2_list  = [8.0,4.0],
#             A_gain      = [5.0,5.0],
#             dt          = 0.01
#         )
#         x_list = [self.fr3_state.ee_pos, self.heal_state.ee_pos]
#         self.syncer.z_list = [self.syncer.compute_alpha(x_list[i], self.syncer.x1_list[i], self.syncer.x2_list[i]) for i in range(self.syncer.n)]

#         # 5) Phase 2: Asynchronous DS 
#         self.async_ds = DualArmController(2.0, 2.0, 2.0, 2.0)
#         self.async_ds.set_mode("asynchronous")
#         self.x_final_heal = np.array([-0.3, 0.3, 0.5])  
        
#         # 6) Flags
#         self.phase1_done    = False
#         self.heal_arrived   = False
#         self.pos_thresh     = 0.005  
#         self.snap           = False
#         rospy.Subscriber("/snap", Bool, self._snap_callback)
        
#         # 7) Controller switch srv
#         rospy.loginfo("Waiting for /controller_manager/switch_controller service.")
#         rospy.wait_for_service('/fr3/controller_manager/switch_controller')
#         rospy.loginfo("Found switch_controller service")
#         self._loader = rospy.ServiceProxy(
#             '/fr3/controller_manager/load_controller', LoadController
#         )
#         try:
#             self._loader('vic_controller')
#             rospy.loginfo("Loaded vic_controller")
#         except rospy.ServiceException as e:
#             rospy.logwarn(f"Could not load vic_controller: {e}")

#         rospy.loginfo("Waiting for /fr3/controller_manager/switch_controller…")
#         rospy.wait_for_service('/fr3/controller_manager/switch_controller')
#         self._switch = rospy.ServiceProxy(
#             '/fr3/controller_manager/switch_controller', SwitchController
#         )
#         rospy.loginfo("Controller services ready.")
        
#         # 8) DLSVelocityCommissioner threads
#         self.fr3_comm               = DLSVelocityCommander(
#             robot_state             = self.fr3_state,
#             ik_solver               = self.fr3_ik,
#             custom_ds               = self.fr3_twist,
#             joint_state_topic       = "/fr3/joint_states",
#             ee_pose_topic           = "/fr3/ee_pose",
#             ee_pose_msg_type        = Pose,
#             velocity_command_topic  = "/fr3/joint_velocity_controller/joint_velocity_command",
#             max_cartesian_vel       = 0.5,
#             max_angular_vel         = 0.5
#         )
#         self.heal_comm              = DLSVelocityCommander(
#             robot_state             = self.heal_state,
#             ik_solver               = self.heal_ik,
#             custom_ds               = self.heal_twist,
#             joint_state_topic       = "/heal/joint_states",
#             ee_pose_topic           = "/heal/ee_pose",
#             ee_pose_msg_type        = PoseStamped,
#             velocity_command_topic  = "/heal/velocity_controller/command",
#             max_cartesian_vel       = 0.05,
#             max_angular_vel         = 0.075
#         )
    
#     def _snap_callback(self, msg: Bool):
#         if msg.data and not self.snap:
#             rospy.loginfo("Snap detected: Freezing Heal")
#         self.snap = msg.data
        
#     def fr3_twist(self):
#         """Zero arg twist for FR3."""
#         # Phase 1: coupled DS
#         if not self.phase1_done:
#             v_list = self.syncer.compute_velocity([
#                 self.fr3_state.ee_pos,
#                 self.heal_state.ee_pos
#             ])
#             v_fr3 = v_list[0]
#             # check stop condition
#             err_fr3 = np.linalg.norm(self.fr3_state.ee_pos - self.syncer.x2_list[0])
#             err_heal= np.linalg.norm(self.heal_state.ee_pos - self.syncer.x2_list[1])
#             if err_fr3<self.pos_thresh and err_heal<self.pos_thresh:
#                 self.phase1_done = True
#                 self._hand_off()
#             return np.hstack((v_fr3, np.zeros(3)))
#         # Phase 2: FR3 is now VIC‐controlled → always zero twist
#         return np.zeros(6)

#     def heal_twist(self):
#         """Zero arg twist for HEAL."""
#         # Phase 1: coupled DS
#         if not self.phase1_done:
#             v_list = self.syncer.compute_velocity([
#                 self.fr3_state.ee_pos,
#                 self.heal_state.ee_pos
#             ])
#             return np.hstack((v_list[1], np.zeros(3)))
#         # Phase 2: async DS until snap or arrival
#         if not self.snap and not self.heal_arrived:
#             v, omega, _, _ = self.async_ds.compute_commands(
#                 # current positions
#                 self.heal_state.ee_pos,
#                 self.fr3_state.ee_pos,    # dummy
#                 # async goals
#                 self.x_final_heal,
#                 self.x_final_heal,        # dummy
#                 # sync goals (unused here)
#                 self.x_final_heal,
#                 self.x_final_heal,        # dummy
#                 # current orientations
#                 [0,0,0,1],
#                 [0,0,0,1],
#                 # async orientation goals
#                 [0,0,0,1],
#                 [0,0,0,1],
#                 # sync orientation goals
#                 [0,0,0,1],
#                 [0,0,0,1],
#                 # time
#                 rospy.get_time()
#             )
#             err = np.linalg.norm(self.heal_state.ee_pos - self.x_final_heal)
#             if err < self.pos_thresh:
#                 self.heal_arrived = True
#             return np.hstack((v, np.zeros(3)))
#         return np.zeros(6)
    
#     def _hand_off(self):
#         """Stop FR3 joint vel, start VIC; HEAL stays in Python async mode."""
#         rospy.loginfo("Phase 1 done: switching FR3 to VIC")
#         req = SwitchControllerRequest(
#             stop_controllers=['joint_velocity_controller'],
#             start_controllers=['vic_controller'],
#             strictness=SwitchControllerRequest.STRICT
#         )
#         try:
#             self._switch(req)
#             rospy.loginfo("VIC started on FR3")
#         except rospy.ServiceException as e:
#             rospy.logerr(f"SwitchController failed: {e}")
            
#     def start(self):
#         # launch both commander loops
#         threading.Thread(target = self.fr3_comm.run,  name = "fr3_loop").start()
#         threading.Thread(target = self.heal_comm.run, name = "heal_loop").start()

# if __name__ == "__main__":
#     ctrl = MainController()
#     ctrl.start()
#     rospy.spin()