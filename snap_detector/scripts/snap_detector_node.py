#!/usr/bin/env python3

import rospy
import numpy as np
import torch
from sensor_msgs.msg import JointState
import sys
sys.path.append("/home/iitgn-robotics/ds_yash/bimanual_ws/src/snap_detector/src")

from std_msgs.msg import Bool
from snap_detector.snapnet import SnapDetectorNet

# === Config ===
WINDOW_SIZE = 14
THRESHOLD = 0.2
JOINT_DIM = 7

MODEL_PATH = "/home/iitgn-robotics/bimanual_ws/src/snap_detector/models/attention_model_newer.pt"
MIN_PATH = MODEL_PATH.replace(".pt", "_min.npy")
MAX_PATH = MODEL_PATH.replace(".pt", "_max.npy")

class SnapDetector:
    def __init__(self):
        self.model = SnapDetectorNet(window_size=WINDOW_SIZE)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        self.model.eval()

        self.vmin = np.load(MIN_PATH)
        self.vmax = np.load(MAX_PATH)

        self.buffer = []

        rospy.Subscriber("fr3/franka_state_controller/joint_states", JointState, self.joint_callback)
        self.pub = rospy.Publisher("/snap", Bool, queue_size=1)
        # rospy.loginfo("Snap Detector Node Initialized")
        
        print(f"Snap Detection Threshold: {THRESHOLD}")

    def normalize(self, v):
        return (v - self.vmin) / (self.vmax - self.vmin + 1e-8)

    def joint_callback(self, msg):
        if len(msg.velocity) != JOINT_DIM:
            rospy.logwarn("Unexpected joint dimension")
            return

        velocity = np.array(msg.velocity)
        normed = self.normalize(velocity)
        self.buffer.append(normed)

        if len(self.buffer) >= WINDOW_SIZE:
            window = np.array(self.buffer[-WINDOW_SIZE:])  # Shape (T, J)
            window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, T, J)
            with torch.no_grad():
                prob = torch.sigmoid(self.model(window_tensor)).item()
                is_snap = prob >= THRESHOLD
                self.pub.publish(Bool(data=is_snap))

                #Debug:
                rospy.loginfo(f"Snap Prob: {prob:.3f} → {'SNAP' if is_snap else 'No Snap'}")
     
                
def main():
    rospy.init_node("snap_detector_node")
    SnapDetector()
    rospy.spin()

if __name__ == "__main__":
    main()