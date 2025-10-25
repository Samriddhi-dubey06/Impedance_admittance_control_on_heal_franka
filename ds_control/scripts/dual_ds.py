#!/usr/bin/env python3
"""
dual_ds.py

Simplified dual robot controller: runs only the positional DS (CoupledDSSynchronizer)
for two arms (FR3 and HEAL), no orientation synchronization.
Computes 6D twists [vx,vy,vz, 0,0,0] and sends them through DLS IK to joint velocity controllers.
"""

import threading

import rospy
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped

from ds_control.robot_state     import RobotState
from ds_control.kdl_ik_solver   import DLSIKSolver
from ds_control.ds_position     import CoupledDSSynchronizer
from ds_control.dls_velocity    import DLSVelocityCommander

def main():
    rospy.init_node('dual_ds_pos_only', anonymous=True)

    # 1) Define joint name lists
    franka_joints = [f"fr3_joint{i}" for i in range(1, 8)]
    heal_joints   = ["Joint_1", "Joint_2", "Joint_3",
                     "Joint_4", "Joint_5", "Joint_6"]

    # 2) Create RobotState and IK solver for each arm
    franka_state = RobotState('franka', franka_joints, logger=rospy)
    heal_state   = RobotState('heal',   heal_joints,   logger=rospy)

    franka_ik = DLSIKSolver(
        urdf_param="/fr3/robot_description",
        base_link="fr3_link0",
        tip_link="fr3_link8",
        joint_names=franka_joints,
        damping=0.01
    )
    heal_ik = DLSIKSolver(
        urdf_param="/heal/robot_description",
        base_link="base_link",
        tip_link="tool_ff",
        joint_names=heal_joints,
        damping=0.10
    )

    # 3) Wait for initial end‐effector poses
    rospy.loginfo("Waiting for initial EE poses...")
    franka_msg = rospy.wait_for_message("/fr3/ee_pose", Pose)
    heal_msg   = rospy.wait_for_message("/heal/ee_pose", PoseStamped)

    franka_state.update_from_pose(franka_msg)
    heal_state.update_from_pose(heal_msg)

    x1_franka = np.array([0.3073032327909314,
                          0.00019789930399554886,
                           0.48633868793270196])
    
    x1_heal   = np.array([-0.0008534687695265184, 
                          0.36794519040992163, 
                          0.5538129942450314])

    x2_franka = np.array([0.4086877711053955,
                          -0.20118787964395773,
                           0.23837679661108473])
    
    x2_heal   = np.array([-0.332073421791668, 
                          0.48192858307270514, 
                          0.2630344124776479])

    # 4) Build positional DS with start & goal waypoints
    x1_list = [x1_franka, x1_heal]
    x2_list = [x2_franka, x2_heal]

    beta1 = [0.1, 0.1]
    beta2 = [8.0, 4.0]
    A_gain = [5.0, 5.0]
    zd = 0.0
    
    pos_ds = CoupledDSSynchronizer(
        x1_list, 
        x2_list,
        zd,
        beta1, 
        beta2, 
        A_gain
    )

    x_list = [
            franka_state.ee_pos,
            heal_state.ee_pos
        ]
    
    # seed the DS phase z_i to reflect initial positions
    pos_ds.z_list = [
        pos_ds.compute_alpha(x_list[i], x1_list[i], x2_list[i])
        for i in range(len(x1_list))
    ]

    # 5) Define zero‐arg DS functions returning 6D twist
    def franka_ds():
        v_list = pos_ds.compute_velocity([
            franka_state.ee_pos,
            heal_state.ee_pos
        ])
        # no orientation DS → angular velocity = zero
        return np.hstack((v_list[0], np.zeros(3)))

    def heal_ds():
        v_list = pos_ds.compute_velocity([
            franka_state.ee_pos,
            heal_state.ee_pos
        ])
        return np.hstack((v_list[1], np.zeros(3)))

    # 6) Instantiate and start two DLSVelocityCommander threads
    franka_commander = DLSVelocityCommander(
        robot_state=franka_state,
        ik_solver=franka_ik,
        custom_ds=franka_ds,
        joint_state_topic="/fr3/joint_states",
        ee_pose_topic="/fr3/ee_pose",
        ee_pose_msg_type=Pose,
        velocity_command_topic="/fr3/joint_velocity_controller/joint_velocity_command",
        logger=rospy,
        max_cartesian_vel=0.5,
        max_angular_vel=0.5
    )

    heal_commander = DLSVelocityCommander(
        robot_state=heal_state,
        ik_solver=heal_ik,
        custom_ds=heal_ds,
        joint_state_topic="/heal/joint_states",
        ee_pose_topic="/heal/ee_pose",
        ee_pose_msg_type=PoseStamped,
        velocity_command_topic="/heal/velocity_controller/command",
        logger=rospy,
        max_cartesian_vel=0.05,
        max_angular_vel=0.075
    )

    threads = [
        threading.Thread(target=franka_commander.run, name="franka_loop"),
        threading.Thread(target=heal_commander.run,   name="heal_loop")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

if __name__ == "__main__":
    main()




# #!/usr/bin/env python3
# """
# dual_ds.py

# Bimanual controller: runs positional DS (CoupledDSSynchronizer)
# and orientation DS (CoupledDSOrientationSynchronizer) for FR3 & HEAL.
# Computes 6D twists [vx,vy,vz, wx,wy,wz] and sends them through DLS IK.
# """

# import threading
# import rospy
# import numpy as np
# from geometry_msgs.msg import Pose, PoseStamped

# from ds_control.robot_state       import RobotState
# from ds_control.kdl_ik_solver     import DLSIKSolver
# from ds_control.ds_position       import CoupledDSSynchronizer
# from ds_control.ds_orientation    import CoupledDSOrientationSynchronizer
# from ds_control.dls_velocity      import DLSVelocityCommander

# def main():
#     rospy.init_node('dual_ds_full', anonymous=True)

#     # 1) Joint names
#     franka_joints = [f"fr3_joint{i}" for i in range(1, 8)]
#     heal_joints   = [f"Joint_{i}"  for i in range(1, 7)]

#     # 2) RobotState & IK
#     fr3_state = RobotState('fr3', franka_joints, logger=rospy)
#     heal_state= RobotState('heal', heal_joints,   logger=rospy)

#     fr3_ik = DLSIKSolver(
#         urdf_param="/fr3/robot_description",
#         base_link ="fr3_link0",
#         tip_link  ="fr3_link8",
#         joint_names=franka_joints,
#         damping   =0.01
#     )
#     heal_ik = DLSIKSolver(
#         urdf_param="/heal/robot_description",
#         base_link ="base_link",
#         tip_link  ="tool_ff",
#         joint_names=heal_joints,
#         damping   =0.10
#     )

#     # 3) Wait for initial EE poses
#     rospy.loginfo("Waiting for initial EE poses...")
#     fr3_msg  = rospy.wait_for_message("/fr3/ee_pose",  Pose)
#     heal_msg = rospy.wait_for_message("/heal/ee_pose", PoseStamped)
#     fr3_state .update_from_pose(fr3_msg)
#     heal_state.update_from_pose(heal_msg)

#     # 4) Define start & goal pos+quat for both arms
#     x1_fr = fr3_state.ee_pos.copy()
#     x1_he = heal_state.ee_pos.copy()
#     q1_fr = fr3_state.ee_ori.copy()
#     q1_he = heal_state.ee_ori.copy()

#     x2_fr = np.array([0.3157663755966209, -0.6641038105595556, 0.45864489784781526])
#     x2_he = np.array([-0.33207342,  0.48192858, 0.26303441])
#     # q2_fr = np.array([0.5, 0.5, -0.5, 0.5])    
#     q2_fr = np.array([0.0, 0.0, 0.0, 1.0])    
#     q2_he = np.array([0.0, 0.0, 0.0, 1.0])
    

#     # 5) Positional DS
#     pos_ds = CoupledDSSynchronizer(
#         x1_list     = [x1_fr, x1_he],
#         x2_list     = [x2_fr, x2_he],
#         zd          = 0.0,
#         beta1_list  = [0.1, 0.1],
#         beta2_list  = [8.0, 4.0],
#         A_gain      = [5.0, 5.0],
#         dt          = 0.01
#     )
#     # seed z_list from current positions
#     pos_ds.z_list = [
#         pos_ds.compute_alpha(state, start, goal)
#         for state, start, goal in zip(
#             [fr3_state.ee_pos, heal_state.ee_pos],
#             pos_ds.x1_list, pos_ds.x2_list
#         )
#     ]

#     # 6) Orientation DS (UNCHANGED from your working copy)
#     ori_ds = CoupledDSOrientationSynchronizer(
#         q1_list=[q1_fr, q1_he],
#         q2_list=[q2_fr, q2_he],
#         beta1=0.2,
#         beta2=0.5,
#         A_gain=3.0
#     )
#     # seed z_rot_list using the same α‐rot that the class uses internally:
#     ori_ds.z_rot_list = ori_ds._all_alpha_rot([q1_fr, q1_he])

#     # 7) DS‐based twist functions
#     def fr3_ds():
#         v_list = pos_ds.compute_velocity([fr3_state.ee_pos, heal_state.ee_pos])
#         w_list = ori_ds.compute_angular_velocity([fr3_state.ee_ori, heal_state.ee_ori])
#         return np.hstack((v_list[0], w_list[0]))

#     def heal_ds():
#         v_list = pos_ds.compute_velocity([fr3_state.ee_pos, heal_state.ee_pos])
#         w_list = ori_ds.compute_angular_velocity([fr3_state.ee_ori, heal_state.ee_ori])
#         return np.hstack((v_list[1], w_list[1]))

#     # 8) Launch DLSVelocityCommander threads
#     fr3_comm = DLSVelocityCommander(
#         robot_state             = fr3_state,
#         ik_solver               = fr3_ik,
#         custom_ds               = fr3_ds,
#         joint_state_topic       = "/fr3/joint_states",
#         ee_pose_topic           = "/fr3/ee_pose",
#         ee_pose_msg_type        = Pose,
#         velocity_command_topic  = "/fr3/joint_velocity_controller/joint_velocity_command",
#         max_cartesian_vel       = 0.5,
#         max_angular_vel         = 0.5
#     )
#     heal_comm = DLSVelocityCommander(
#         robot_state             = heal_state,
#         ik_solver               = heal_ik,
#         custom_ds               = heal_ds,
#         joint_state_topic       = "/heal/joint_states",
#         ee_pose_topic           = "/heal/ee_pose",
#         ee_pose_msg_type        = PoseStamped,
#         velocity_command_topic  = "/heal/velocity_controller/command",
#         max_cartesian_vel       = 0.05,
#         max_angular_vel         = 0.075
#     )

#     threads = [
#         threading.Thread(target=fr3_comm.run,  name="fr3_loop"),
#         threading.Thread(target=heal_comm.run, name="heal_loop")
#     ]
#     for t in threads:
#         t.start()
#     for t in threads:
#         t.join()

# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python3
# """
# dual_ds.py

# Bimanual controller: runs positional DS (CoupledDSSynchronizer)
# and orientation DS (CoupledDSOrientationSynchronizer) for FR3 & HEAL.
# Computes 6D twists [vx,vy,vz, wx,wy,wz] and sends them through DLS IK.
# """

# import threading

# import rospy
# import numpy as np
# from geometry_msgs.msg import Pose, PoseStamped

# from ds_control.robot_state       import RobotState
# from ds_control.kdl_ik_solver     import DLSIKSolver
# from ds_control.ds_position       import CoupledDSSynchronizer
# from ds_control.ds_orientation    import CoupledDSOrientationSynchronizer
# from ds_control.dls_velocity      import DLSVelocityCommander


# def main():
#     rospy.init_node('dual_ds_full', anonymous=True)

#     # 1) Joint names
#     franka_joints = [f"fr3_joint{i}" for i in range(1, 8)]
#     heal_joints   = [f"Joint_{i}"  for i in range(1, 7)]

#     # 2) RobotState & IK
#     fr3_state = RobotState('fr3', franka_joints, logger=rospy)
#     heal_state= RobotState('heal', heal_joints,   logger=rospy)

#     fr3_ik = DLSIKSolver(
#         urdf_param="/fr3/robot_description",
#         base_link ="fr3_link0",
#         tip_link  ="fr3_link8",
#         joint_names=franka_joints,
#         damping   =0.01
#     )
#     heal_ik = DLSIKSolver(
#         urdf_param="/heal/robot_description",
#         base_link ="base_link",
#         tip_link  ="tool_ff",
#         joint_names=heal_joints,
#         damping   =0.10
#     )

#     # 3) Wait for initial EE poses
#     rospy.loginfo("Waiting for initial EE poses...")
#     fr3_msg  = rospy.wait_for_message("/fr3/ee_pose",  Pose)
#     heal_msg = rospy.wait_for_message("/heal/ee_pose", PoseStamped)
#     fr3_state .update_from_pose(fr3_msg)
#     heal_state.update_from_pose(heal_msg)

#     # 4) Define start & goal pos+quat for both arms
#     x1_fr = fr3_state.ee_pos.copy()
#     x1_he = heal_state.ee_pos.copy()
#     q1_fr = fr3_state.ee_ori.copy()
#     q1_he = heal_state.ee_ori.copy()

#     x2_fr = np.array([0.3157663755966209, -0.6641038105595556, 0.45864489784781526])
#     x2_he = np.array([-0.33207342,  0.48192858, 0.26303441])
#     # example goal orientations:
#     q2_fr = np.array([-0.5, -0.5, 0.5, 0.5])
#     q2_he = np.array([0.0, 0.0, 0.0, 1.0])

#     # 5) Positional DS
#     pos_ds = CoupledDSSynchronizer(
#         x1_list=[x1_fr, x1_he],
#         x2_list=[x2_fr, x2_he],
#         zd       =0.0,
#         beta1_list=[0.1, 0.1],
#         beta2_list=[8.0, 4.0],
#         A_gain   =[5.0, 5.0],
#         dt       =0.01
#     )
#     # seed z
#     pos_ds.z_list = [
#         pos_ds.compute_alpha(state, start, goal)
#         for state, start, goal in zip(
#             [fr3_state.ee_pos, heal_state.ee_pos],
#             pos_ds.x1_list, pos_ds.x2_list
#         )
#     ]

#     # 6) Orientation DS
#     ori_ds = CoupledDSOrientationSynchronizer(
#         q1_list=[q1_fr, q1_he],
#         q2_list=[q2_fr, q2_he],
#         beta1   =1.0,
#         beta2   =0.5,
#         A_gain  =1.0
#     )

#     # 7) DS‐based twist functions
#     def fr3_ds():
#         v_list = pos_ds.compute_velocity([fr3_state.ee_pos, heal_state.ee_pos])
#         w_list = ori_ds.compute_angular_velocity([fr3_state.ee_ori, heal_state.ee_ori])
#         return np.hstack((v_list[0], w_list[0]))

#     def heal_ds():
#         v_list = pos_ds.compute_velocity([fr3_state.ee_pos, heal_state.ee_pos])
#         w_list = ori_ds.compute_angular_velocity([fr3_state.ee_ori, heal_state.ee_ori])
#         return np.hstack((v_list[1], w_list[1]))

#     # 8) Launch DLSVelocityCommander threads
#     fr3_comm = DLSVelocityCommander(
#         robot_state             =fr3_state,
#         ik_solver               =fr3_ik,
#         custom_ds               =fr3_ds,
#         joint_state_topic       ="/fr3/joint_states",
#         ee_pose_topic           ="/fr3/ee_pose",
#         ee_pose_msg_type        =Pose,
#         velocity_command_topic  ="/fr3/joint_velocity_controller/joint_velocity_command",
#         logger                  =rospy,
#         max_cartesian_vel       =0.5,
#         max_angular_vel         =0.5
#     )
#     heal_comm = DLSVelocityCommander(
#         robot_state             =heal_state,
#         ik_solver               =heal_ik,
#         custom_ds               =heal_ds,
#         joint_state_topic       ="/heal/joint_states",
#         ee_pose_topic           ="/heal/ee_pose",
#         ee_pose_msg_type        =PoseStamped,
#         velocity_command_topic  ="/heal/velocity_controller/command",
#         logger                  =rospy,
#         max_cartesian_vel       =0.05,
#         max_angular_vel         =0.075
#     )
#     for thr in (
#         threading.Thread(target=fr3_comm.run,  name="fr3_loop"),
#         threading.Thread(target=heal_comm.run, name="heal_loop")
#     ):
#         thr.start()
#     for thr in threading.enumerate():
#         if thr.name in ("fr3_loop", "heal_loop"):
#             thr.join()


# if __name__ == "__main__":
#     main()




# #!/usr/bin/env python3
# import threading

# import rospy
# import numpy as np
# from geometry_msgs.msg import Pose, PoseStamped

# from ds_control.robot_state      import RobotState
# from ds_control.kdl_ik_solver    import DLSIKSolver
# from ds_control.ds_position      import CoupledDSSynchronizer
# from ds_control.ds_orientation   import CoupledDSOrientationSynchronizer
# from ds_control.dls_velocity     import DLSVelocityCommander

# def main():
#     rospy.init_node('dual_ds', anonymous=True)

#     # 1) -- Define joint name lists --
#     franka_joints = [f"fr3_joint{i}" for i in range(1, 8)]
#     heal_joints   = ["Joint_1", "Joint_2", "Joint_3",
#                      "Joint_4", "Joint_5", "Joint_6"]

#     # 2) -- Instantiate RobotState and IK solver for each robot --
#     franka_state = RobotState('franka', franka_joints, logger=rospy)
#     heal_state   = RobotState('heal',   heal_joints,   logger=rospy)

#     franka_ik = DLSIKSolver(
#         urdf_param="/fr3/robot_description",
#         base_link="fr3_link0",
#         tip_link="fr3_link8",
#         joint_names=franka_joints,
#         damping=0.01
#     )
#     heal_ik   = DLSIKSolver(
#         urdf_param="/heal/robot_description",
#         base_link="base_link",
#         tip_link="tool_ff",
#         joint_names=heal_joints,
#         damping=0.1
#     )

#     # 3) -- Wait for initial EE poses and seed RobotState --
#     rospy.loginfo("Waiting for initial EE poses...")
#     franka_msg = rospy.wait_for_message("/fr3/ee_pose", Pose)
#     heal_msg   = rospy.wait_for_message("/heal/ee_pose", PoseStamped)

#     franka_state.update_from_pose(franka_msg)
#     heal_state.  update_from_pose(heal_msg)

#     # 4) -- Build DS synchronizers with start & goal waypoints --
#     x1_list = [franka_state.ee_pos.copy(), heal_state.ee_pos.copy()]
#     q1_list = [franka_state.ee_ori.copy(), heal_state.ee_ori.copy()]

#     # Fill in your goal waypoints here:
#     x2_franka = np.array([0.4086877711053955, -0.20118787964395773, 0.13837679661108473])
#     x2_heal   = np.array([-0.2576, 0.46032, 0.30512])
#     q2_franka = np.array([0.9998145599783635, -0.0015483369991307035, 0.0028005495347031696, -0.018989608477055286])
#     q2_heal   = np.array([-0.061123483842736734, 0.05443661634760124, 0.2802209229396051, 0.9564396524979643])

#     x2_list = [x2_franka, x2_heal]
#     q2_list = [q2_franka, q2_heal]

#     # Normalize & hemisphere align quaternions
#     for i in range(2):
#         q1_list[i] /= np.linalg.norm(q1_list[i])
#         q2_list[i] /= np.linalg.norm(q2_list[i])
#         if np.dot(q2_list[i], q1_list[i]) < 0:
#             q1_list[i] = -q1_list[i]

#     pos_ds = CoupledDSSynchronizer(x1_list, x2_list,
#                                    beta1=4.0, beta2=0.1, A_gain=5.0)
#     ori_ds = CoupledDSOrientationSynchronizer(q1_list, q2_list,
#                                               beta1=0.2, beta2=0.5, A_gain=3.0)

#     # Seed their internal phase variables so z_i starts at 1.0
#     pos_ds.z_list     = [pos_ds.compute_alpha(x1_list[i], x1_list[i], x2_list[i])
#                          for i in range(2)]


#     # 5) -- Define zero‐arg DS functions to feed into DLSVelocityCommander --
#     def franka_ds():
#         v_list = pos_ds.compute_velocity([franka_state.ee_pos,
#                                           heal_state.ee_pos])
#         w_list = ori_ds.compute_angular_velocity([franka_state.ee_ori,
#                                                   heal_state.ee_ori])
#         return np.hstack((v_list[0], w_list[0]))

#     def heal_ds():
#         v_list = pos_ds.compute_velocity([franka_state.ee_pos,
#                                           heal_state.ee_pos])
#         w_list = ori_ds.compute_angular_velocity([franka_state.ee_ori,
#                                                   heal_state.ee_ori])
#         return np.hstack((v_list[1], w_list[1]))

#     # 6) -- Instantiate and start two DLSVelocityCommander threads --
#     franka_commander = DLSVelocityCommander(
#         robot_state=franka_state,
#         ik_solver=franka_ik,
#         custom_ds=franka_ds,
#         joint_state_topic="/fr3/joint_states",
#         ee_pose_topic="/fr3/ee_pose",
#         ee_pose_msg_type=Pose,
#         velocity_command_topic="/fr3/joint_velocity_controller/joint_velocity_command",
#         logger=rospy,
#         max_cartesian_vel=0.05,             
#         max_angular_vel=0.100
#     )
#     heal_commander = DLSVelocityCommander(
#         robot_state=heal_state,
#         ik_solver=heal_ik,
#         custom_ds=heal_ds,
#         joint_state_topic="heal/joint_states",
#         ee_pose_topic="/heal/ee_pose",
#         ee_pose_msg_type=PoseStamped,
#         velocity_command_topic="/heal/velocity_controller/command",
#         logger=rospy,
#         max_cartesian_vel=0.05,             
#         max_angular_vel=0.075              
#     )

#     threads = [
#         threading.Thread(target=franka_commander.run, name="franka_loop"),
#         threading.Thread(target=heal_commander.  run, name="heal_loop")
#     ]
#     for t in threads:
#         t.start()
#     for t in threads:
#         t.join()

# if __name__ == "__main__":
#     main()
