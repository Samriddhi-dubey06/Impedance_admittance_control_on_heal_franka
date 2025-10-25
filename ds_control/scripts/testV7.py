#!/usr/bin/env python3

import threading
import rospy
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped
from controller_manager_msgs.srv import LoadController, SwitchController, SwitchControllerRequest

from ds_control.robot_state   import RobotState
from ds_control.kdl_ik_solver import DLSIKSolver
from ds_control.ds_coordination import DSCoordinationFramework
from ds_control.dls_velocity  import DLSVelocityCommander
from ds_control.gripper_actions import HealGripperSimpleController, FrankaGripperSimpleController

class MainController:
    def __init__(self):
        rospy.init_node('main_controller', anonymous=True)

        fr3_joints = [f"fr3_joint{i}" for i in range(1,8)]
        heal_joints= [f"Joint_{i}"   for i in range(1,7)]
        self.fr3_state  = RobotState('franka', fr3_joints, logger=rospy)
        self.heal_state = RobotState('heal',   heal_joints, logger=rospy)

        self.fr3_ik  = DLSIKSolver(
            urdf_param="/fr3/robot_description",
            base_link ="fr3_link0",
            tip_link  ="fr3_link8",
            joint_names=fr3_joints,
            damping=0.01
        )
        self.heal_ik = DLSIKSolver(
            urdf_param="/heal/robot_description",
            base_link ="base_link",
            tip_link  ="tool_ff",
            joint_names=heal_joints,
            damping=0.10
        )
       
        self.fr3_gripper = FrankaGripperSimpleController()
        self.heal_gripper= HealGripperSimpleController()

        # wait for initial EE poses
        rospy.loginfo("Waiting for initial EE poses...")
        fr3_msg  = rospy.wait_for_message("/fr3/ee_pose", Pose)
        heal_msg = rospy.wait_for_message("/heal/ee_pose", PoseStamped)
        self.fr3_state.update_from_pose(fr3_msg)
        self.heal_state.update_from_pose(heal_msg)
        rospy.loginfo("EE poses received.")

        # record “rest” poses
        self.x_rest_fr3,  self.q_rest_fr3  = self.fr3_state.ee_pos.copy(),  self.fr3_state.ee_ori.copy()
        self.x_rest_heal, self.q_rest_heal = self.heal_state.ee_pos.copy(), self.heal_state.ee_ori.copy()

        # 1) Pick poses:
        self.x_pick_fr3,    self.q_pick_fr3     = np.array([0.518871485875372, -0.01929494859305955, 0.08539020620845625]), np.array([1.0, 0.0, 0.0, 0.0])
        self.x_pick_heal,   self.q_pick_heal    = np.array([-0.11913892083802295, 0.5507251308925488, 0.28001017064576156]), np.array([0.0185024957031101, 0.0016630467873368553, 0.13432478086842833, 0.9907632134737283])
        
        # 2) Assembly poses:
        self.x_assembly_fr3,    self.q_assembly_fr3     = np.array([ 0.426688, -0.694639 , 0.422938]), np.array([ 0.488407 , 0.48911,  -0.506527,  0.515423])
        self.x_assembly_heal,   self.q_assembly_heal    = np.array([-0.3,0.44775823462306896,0.41869831600162616]), np.array([-0.10123182765431753 ,0.7080480116925151, -0.030799968402897986 ,0.6981915869977401])
        
        # 3) Final heal-only pose:
        self.x_final_fr3,   self.q_final_fr3   = self.x_assembly_fr3.copy(), self.q_assembly_fr3.copy()
        self.x_final_heal, self.q_final_heal = np.array([-0.4513943275488582,0.44775823462306896,0.41869831600162616]), np.array([-0.10123182765431753 ,0.7080480116925151, -0.030799968402897986 ,0.6981915869977401])

        # thresholds
        self.pos_thresh = 0.001
        self.ori_thresh = 0.01

        # phase: 1=pick, 2=assembly, 3=final‐heal
        self.phase = 1
        self.fr3_arrived  = False
        self.heal_arrived = False

        # preload VIC controller for later
        rospy.wait_for_service('/fr3/controller_manager/load_controller')
        load = rospy.ServiceProxy('/fr3/controller_manager/load_controller', LoadController)
        load('vic_controller')
        rospy.wait_for_service('/fr3/controller_manager/switch_controller')
        self._switch = rospy.ServiceProxy('/fr3/controller_manager/switch_controller', SwitchController)

        # DSCoordination for phase 1
        self._setup_phase()

        # velocity commanders
        self.heal_comm = DLSVelocityCommander(
            robot_state            = self.heal_state,
            ik_solver              = self.heal_ik,
            custom_ds              = self._twist_fn(0),
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
            custom_ds              = self._twist_fn(1),
            joint_state_topic      = "/fr3/joint_states",
            ee_pose_topic          = "/fr3/ee_pose",
            ee_pose_msg_type       = Pose,
            velocity_command_topic = "/fr3/joint_velocity_controller/joint_velocity_command",
            max_cartesian_vel      = 0.05,
            max_angular_vel        = 0.20,
        )

        # check arrival / phase change at 10 Hz
        rospy.Timer(rospy.Duration(0.1), self._phase_callback)


    def _setup_phase(self):
        """(Re)build DSCoordinationFramework for current phase."""
        if self.phase == 1:
            x1, x2 = [self.x_rest_heal,  self.x_rest_fr3],   [self.x_pick_heal,  self.x_pick_fr3]
            q1, q2 = [self.q_rest_heal,  self.q_rest_fr3],   [self.q_pick_heal,  self.q_pick_fr3]
            κ      = 1.0
        elif self.phase == 2:
            x1, x2 = [self.x_pick_heal,  self.x_pick_fr3],   [self.x_assembly_heal,   self.x_assembly_fr3]
            q1, q2 = [self.q_pick_heal,  self.q_pick_fr3],   [self.q_assembly_heal,   self.q_assembly_fr3]
            κ      = 1.0
        elif self.phase == 3:
            x1, x2 = [self.x_assembly_heal,   self.x_assembly_fr3],    [self.x_final_heal, self.x_final_fr3]
            q1, q2 = [self.q_assembly_heal,   self.q_assembly_fr3],    [self.q_final_heal, self.q_final_fr3]
            κ      = 0.0
        else:
            return

        rospy.loginfo(f"→ Entering phase {self.phase} (κ={κ})")
        self.fr3_arrived  = False
        self.heal_arrived = False
        self.coord_ds = DSCoordinationFramework(
            x1_list   = x1,
            x2_list   = x2,
            q1_list   = q1,
            q2_list   = q2,
            k1        = 10.0,
            k2        = 10.0,
            kappa     = κ,
            k_rot     = 2.0,
            alpha_min = 1.0,
            alpha_max = 5.0,
        )

    def _phase_callback(self, _):
        """Check arrival; switch VIC at end of assembly (phase 2) immediately."""
        from scipy.spatial.transform import Rotation

        # pick correct targets for this phase
        if   self.phase == 1:
            xt = [self.x_pick_heal,  self.x_pick_fr3]
            qt = [self.q_pick_heal,  self.q_pick_fr3]
        elif self.phase == 2:
            xt = [self.x_assembly_heal,   self.x_assembly_fr3]
            qt = [self.q_assembly_heal,   self.q_assembly_fr3]
        elif self.phase == 3:
            xt = [self.x_final_heal, self.x_final_fr3]
            qt = [self.q_final_heal, self.q_final_fr3]
        else:
            return

        # check each arm’s pos+ori error
        for idx, state in enumerate((self.heal_state, self.fr3_state)):
            arrived_flag = self.heal_arrived if idx==0 else self.fr3_arrived
            if not arrived_flag:
                p_err = np.linalg.norm(state.ee_pos - xt[idx])
                q_err = (Rotation.from_quat(state.ee_ori)
                         * Rotation.from_quat(qt[idx]).inv()
                        ).as_rotvec()
                o_err = np.linalg.norm(q_err)
                if p_err<self.pos_thresh and o_err<self.ori_thresh:
                    if idx==0:
                        self.heal_arrived = True
                        rospy.loginfo("→ Heal done phase %d", self.phase)
                    else:
                        self.fr3_arrived = True
                        rospy.loginfo("→ Franka done phase %d", self.phase)

        # once both have arrived:
        if self.heal_arrived and self.fr3_arrived:
            if self.phase == 2:
                # *** immediately hand off Franka to VIC ***
                req = SwitchControllerRequest(
                    stop_controllers = ['joint_velocity_controller'],
                    start_controllers= ['vic_controller'],
                    strictness       = SwitchControllerRequest.STRICT
                )
                self._switch(req)
                rospy.loginfo("→ VIC started on Franka at end of assembly")
                # advance to phase 3 for Heal only
                self.phase = 3
                self._setup_phase()
            elif self.phase < 2:
                # pick → assembly
                self.phase += 1
                self._setup_phase()

    def _twist_fn(self, idx):
        """zero-arg twist for heal (idx=0) or fr3 (idx=1)."""
        def twist():
            xs = [self.heal_state.ee_pos, self.fr3_state.ee_pos]
            qs = [self.heal_state.ee_ori, self.fr3_state.ee_ori]
            tws = self.coord_ds.compute_twists(xs, qs, self.coord_ds.dt)
            return tws[idx]
        return twist

    def start(self):
        threading.Thread(target=self.heal_comm.run, name="heal_loop").start()
        threading.Thread(target=self.fr3_comm.run,  name="fr3_loop").start()

if __name__ == '__main__':
    mc = MainController()
    mc.start()
    rospy.spin()
