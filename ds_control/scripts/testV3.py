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
        
        log_path = '/home/iitgn-robotics/ds_yash/bimanual_ws/src/ds_control/data/twist_log.csv'
        self._twist_file = open(log_path, 'w')
        header = ['time']
        for agent in ['heal','fr3']:
            header += [f'{agent}_v{x}' for x in ['x','y','z']]
            header += [f'{agent}_w{x}' for x in ['x','y','z']]
        self._twist_file.write(','.join(header) + '\n')
        self._twist_file.flush()
        self._twist_lock = threading.Lock()
        rospy.on_shutdown(self._close_twist_log)
        
        # Joint log setup
        joint_log_path = '/home/iitgn-robotics/ds_yash/bimanual_ws/src/ds_control/data/joint_log.csv'
        self._joint_log_file = open(joint_log_path, 'w')
        self._joint_log_lock = threading.Lock()
        joint_header = ['time']
        for robot in ['heal', 'fr3']:
            for typ in ['pos', 'vel', 'effort']:
                for i in range(7):
                    joint_header.append(f"{robot}_{typ}{i}")
        self._joint_log_file.write(','.join(joint_header) + '\n')
        self._joint_log_file.flush()
        rospy.on_shutdown(self._close_joint_log)

        franka_joints = [f"fr3_joint{i}" for i in range(1, 8)]
        heal_joints = ["Joint_1", "Joint_2", "Joint_3", "Joint_4", "Joint_5", "Joint_6"]
        
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
        
        
        self.x_rest_fr3,    self.q_rest_fr3    = self.fr3_state.ee_pos.copy(),  self.fr3_state.ee_ori.copy()
        self.x_rest_heal,   self.q_rest_heal   = self.heal_state.ee_pos.copy(), self.heal_state.ee_ori.copy()
        
        # 0) Before Pick poses:
        self.x_pick_fr3_b,    self.q_pick_fr3_b     = np.array([0.514521,-0.252241, 0.21]), np.array([ 0.999622,  0.024782,  0.011864, 0.000831])
        self.x_pick_heal_b,   self.q_pick_heal_b    = np.array([-0.26432285244070264, 0.5719909757730203 ,0.25]), np.array([-0.03025622214135188, 0.030867830557301655, 0.25859643496817486, 0.9650179386312825])
        
        # 1) Pick poses:
        self.x_pick_fr3,    self.q_pick_fr3     = np.array([0.514521,-0.252241, 0.165841]), np.array([ 0.999622,  0.024782,  0.011864, 0.000831])
        self.x_pick_heal,   self.q_pick_heal    = np.array([-0.26432285244070264, 0.5719909757730203 ,0.20508361686007825]), np.array([-0.03025622214135188, 0.030867830557301655, 0.25859643496817486, 0.9650179386312825])
        
        # 2) Assembly poses:
        self.x_assembly_fr3,    self.q_assembly_fr3     = np.array([0.426361, -0.721674, 0.420954]), np.array([0.480514, 0.548639, -0.493223, 0.474166])
        self.x_assembly_heal,   self.q_assembly_heal    = np.array([-0.3, 0.4484379455390496, 0.41449499125848754]), np.array([-0.10345417287658103, 0.7095098836010515, -0.032842406583131464 ,0.6962861017690021])
        
        # 3) Final heal-only pose:
        self.x_final_fr3,   self.q_final_fr3   = self.x_assembly_fr3.copy(), self.q_assembly_fr3.copy()
        self.x_final_heal, self.q_final_heal = np.array([-0.4235810557710056, 0.4484379455390496, 0.41449499125848754]), np.array([-0.10345417287658103, 0.7095098836010515, -0.032842406583131464 ,0.6962861017690021])
        
        self.pos_thresh = 0.00075   
        self.ori_thresh = 0.01   

        # phase bookkeeping
        self.phase = 1
        self.heal_arrived = False
        self.fr3_arrived  = False
        
        self._setup_phase()
        
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
        
        rospy.Timer(rospy.Duration(0.1), self._phase_callback)
        
    def _setup_phase(self):
        """
        (Re)initialize self.coord_ds for the current self.phase.
        """
        if self.phase == 1:
            x1 = [self.x_rest_heal,    self.x_rest_fr3]
            x2 = [self.x_pick_heal_b,    self.x_pick_fr3_b]
            q1 = [self.q_rest_heal,    self.q_rest_fr3]
            q2 = [self.q_pick_heal_b,    self.q_pick_fr3_b]
            kappa = 1.0
            
        elif self.phase == 2:
            x1 = [self.x_pick_heal_b,    self.x_pick_fr3_b]
            x2 = [self.x_pick_heal,    self.x_pick_fr3]
            q1 = [self.q_pick_heal_b,    self.q_pick_fr3_b]
            q2 = [self.q_pick_heal,    self.q_pick_fr3]
            kappa = 1.0

        elif self.phase == 3:
            x1 = [self.x_pick_heal,    self.x_pick_fr3]
            x2 = [self.x_assembly_heal,self.x_assembly_fr3]
            q1 = [self.q_pick_heal,    self.q_pick_fr3]
            q2 = [self.q_assembly_heal,self.q_assembly_fr3]
            kappa = 1.0

        elif self.phase == 4:
            x1 = [self.x_assembly_heal,self.x_assembly_fr3]
            x2 = [self.x_final_heal,   self.x_final_fr3]
            q1 = [self.q_assembly_heal,self.q_assembly_fr3]
            q2 = [self.q_final_heal,   self.q_final_fr3]
            kappa = 0.0

        else:
            rospy.logwarn("All phases done.")
            return

        rospy.loginfo(f"Switching to phase {self.phase}, κ={kappa}")
        self.heal_arrived = False
        self.fr3_arrived  = False

        self.coord_ds = DSCoordinationFramework(
            x1_list   = x1,
            x2_list   = x2,
            q1_list   = q1,
            q2_list   = q2,
            k1        = 15.0,
            k2        = 15.0,
            kappa     = kappa,
            k_rot     = 2.0,
            alpha_min = 1.0,
            alpha_max = 5.0,
        )

    def _phase_callback(self, event):
        """
        Called at 10 Hz: check each arm against its current phase target.
        """
        if self.phase == 1:
            xt, qt = [self.x_pick_heal_b, self.x_pick_fr3_b], [self.q_pick_heal_b, self.q_pick_fr3_b]
        elif self.phase == 2:
            xt, qt = [self.x_pick_heal, self.x_pick_fr3], [self.q_pick_heal, self.q_pick_fr3]
        elif self.phase == 3:
            xt, qt = [self.x_assembly_heal, self.x_assembly_fr3], [self.q_assembly_heal, self.q_assembly_fr3]
        elif self.phase == 4:
            xt, qt = [self.x_final_heal,    self.x_final_fr3],   [self.q_final_heal,    self.q_final_fr3]
        else:
            return

        for idx, (state, arrived_flag) in enumerate([
            (self.heal_state, 'heal_arrived'),
            (self.fr3_state,  'fr3_arrived' )
        ]):
            if not getattr(self, arrived_flag):
                ep = np.linalg.norm(state.ee_pos - xt[idx])
                from scipy.spatial.transform import Rotation
                qr = Rotation.from_quat(state.ee_ori)
                qtgt = Rotation.from_quat(qt[idx])
                eo = np.linalg.norm((qr * qtgt.inv()).as_rotvec())
                if ep < self.pos_thresh and eo < self.ori_thresh:
                    setattr(self, arrived_flag, True)
                    rospy.loginfo(f"  Phase {self.phase} arm {idx} arrived (ep={ep:.4f}, eo={eo:.4f})")

        if self.heal_arrived and self.fr3_arrived:
            self.phase += 1
            if self.phase <= 4:
                self._setup_phase()

    def twist_fn(self, index):
        """
        Closure for DLSVelocityCommander, logs and returns the current 6D twist.
        """
        def twist():
            xs = [self.heal_state.ee_pos, self.fr3_state.ee_pos]
            qs = [self.heal_state.ee_ori, self.fr3_state.ee_ori]
            tws = self.coord_ds.compute_twists(xs, qs, self.coord_ds.dt)

            with self._twist_lock:
                if not self._twist_file.closed:
                    t = rospy.get_time()
                    row = [f"{t:.6f}"] + [f"{v:.6f}" for v in np.hstack(tws)]
                    self._twist_file.write(','.join(row)+'\n')
                    self._twist_file.flush()

            return tws[index]
        return twist

    def _close_twist_log(self):
        try:
            self._twist_file.close()
        except Exception:
            pass
        
    def _close_joint_log(self):
        try:
            self._joint_log_file.close()
        except Exception:
            pass

    def start(self):
        threading.Thread(target=self.heal_comm.run, name="heal_loop").start()
        threading.Thread(target=self.fr3_comm.run,  name="fr3_loop").start()

if __name__ == "__main__":
    ctrl = TestController()
    ctrl.start()
    rospy.spin()
