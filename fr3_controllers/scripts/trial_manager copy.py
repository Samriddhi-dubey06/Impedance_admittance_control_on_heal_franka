#!/usr/bin/env python3

import os
import subprocess
import signal
import time
import rospy
from controller_manager_msgs.srv import (
    SwitchController,
    SwitchControllerRequest,
    ListControllers,
    UnloadController,
    LoadController,
)
from franka_msgs.msg import FrankaState

KY_VALUES = [0, 20, 40, 60, 80]
TRIALS_PER_KY = 20
MAX_TRIAL_SEC = 50.0
MAX_Y_ERROR = 0.05
HOME_SETTLE_TIME = 10.0
Y_DESIRED = 0.03231082225011088

class TrialManager:
    def __init__(self):
        rospy.wait_for_service('/controller_manager/switch_controller')
        rospy.wait_for_service('/controller_manager/list_controllers')
        rospy.wait_for_service('/controller_manager/unload_controller')
        rospy.wait_for_service('/controller_manager/load_controller')

        self.switch_srv = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
        self.list_srv = rospy.ServiceProxy('/controller_manager/list_controllers', ListControllers)
        self.unload_srv = rospy.ServiceProxy('/controller_manager/unload_controller', UnloadController)
        self.load_srv = rospy.ServiceProxy('/controller_manager/load_controller', LoadController)

        self.y_error = 0.0
        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.franka_cb)

    def franka_cb(self, msg):
        current_y = msg.O_T_EE[13]
        self.y_error = abs(current_y - Y_DESIRED)

    def unload_all_controllers(self):
        try:
            resp = self.list_srv()
            for ctrl in resp.controller:
                if ctrl.state == 'stopped':
                    rospy.loginfo(f"Unloading stopped controller: {ctrl.name}")
                    self.unload_srv(ctrl.name)
        except rospy.ServiceException as e:
            rospy.logwarn(f"Failed to unload controllers: {e}")

    def switch(self, start, stop):
        self.unload_all_controllers()  # Clean up first

        # Check if controllers in `start` list are loaded; load if not
        try:
            loaded = [ctrl.name for ctrl in self.list_srv().controller]
            for ctrl_name in start:
                if ctrl_name not in loaded:
                    rospy.loginfo(f"Loading controller: {ctrl_name}")
                    self.load_srv(ctrl_name)
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed during controller loading: {e}")
            return

        req = SwitchControllerRequest()
        req.start_controllers = start
        req.stop_controllers = stop
        req.strictness = SwitchControllerRequest.BEST_EFFORT

        try:
            self.switch_srv(req)
            rospy.loginfo(f"Switched controllers: start={start}, stop={stop}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to switch controllers: {e}")

    def run(self):
        rospy.wait_for_service('/controller_manager/switch_controller', timeout=5.0)
        rospy.sleep(1.0)

        for Ky in KY_VALUES:
            rospy.set_param('/eoi_controller/K_y', float(Ky))
            for trial in range(1, TRIALS_PER_KY + 1):
                rospy.set_param('/eoi_controller/trial_id', trial)
                rospy.loginfo(f"\n=== K_y={Ky} | Trial {trial}/{TRIALS_PER_KY} ===")

                log_proc = subprocess.Popen(['rosrun', 'fr3_controllers', 'data_collection.py'])
                self.switch(['eoi_controller'], ['cartesian_pose_controller'])

                t0 = time.time()
                self.y_error = 0.0

                while not rospy.is_shutdown():
                    if (time.time() - t0) > MAX_TRIAL_SEC or self.y_error > MAX_Y_ERROR:
                        rospy.loginfo("Trial condition met — ending this trial.")
                        break
                    time.sleep(0.02)

                self.switch(['cartesian_pose_controller'], ['eoi_controller'])
                rospy.loginfo("Returning to home pose...")
                time.sleep(HOME_SETTLE_TIME)

                log_proc.send_signal(signal.SIGINT)
                try:
                    log_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    log_proc.kill()

                if rospy.is_shutdown():
                    return

if __name__ == '__main__':
    rospy.init_node('trial_manager')
    TrialManager().run()
