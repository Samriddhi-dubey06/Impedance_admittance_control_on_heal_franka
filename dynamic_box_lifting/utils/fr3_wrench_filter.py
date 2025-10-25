'''

This code publish the filtered wrench of the franka robot on the topic /fr3/wrench_filtered

'''


#!/usr/bin/env python
import rospy
import numpy as np
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import WrenchStamped
import csv
import os


# ---- Defaults (you can override via rosparam) ----
DEFAULT_OFFSET_FORCE  = [0.270, 0.076, 0.768]   # N  (Fx,Fy,Fz)
DEFAULT_OFFSET_TORQUE = [0.315,-0.042, 0.283]   # Nm (Tx,Ty,Tz)
DEFAULT_FC_HZ = 8.0                              # low-pass cutoff frequency
DEFAULT_FRAME = "fr3_link0"                      # frame_id for output message

class WrenchFilterNode:
    def __init__(self):
        self.offset_f = np.array(rospy.get_param("~offset_force",  DEFAULT_OFFSET_FORCE), dtype=float)
        self.offset_t = np.array(rospy.get_param("~offset_torque", DEFAULT_OFFSET_TORQUE), dtype=float)
        self.fc_hz    = float(rospy.get_param("~cutoff_hz", DEFAULT_FC_HZ))
        self.frame_id = rospy.get_param("~frame_id", DEFAULT_FRAME)
        
            # === CSV setup ===
        # === CSV setup ===
        self.csv_dir = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/dynamic_box_lifting/csv"
        os.makedirs(self.csv_dir, exist_ok=True)
        self.csv_file = os.path.join(self.csv_dir, "franka_EE_forces.csv")  # <- file name

        # Create file with header if it doesn't exist
        if not os.path.isfile(self.csv_file):
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])


        # === 20 Hz logger setup ===
        self.log_hz = 20.0
        self.log_dt = 1.0 / self.log_hz
        self.last_log_t = None


        # State for EMA filter
        self.has_prev = False
        self.prev     = np.zeros(6)
        self.prev_t   = None

        self.pub = rospy.Publisher("/fr3/wrench_filtered", WrenchStamped, queue_size=10)
        rospy.Subscriber("/fr3/franka_state_controller/franka_states", FrankaState, self.cb, queue_size=50)

        rospy.loginfo("fr3_wrench_filter: offsets F=%s, T=%s, fc=%.2f Hz, frame_id=%s",
                      self.offset_f, self.offset_t, self.fc_hz, self.frame_id)

    def cb(self, msg: FrankaState):
        # Raw 6D wrench from Franka (base frame): [Fx,Fy,Fz,Tx,Ty,Tz]
        raw = np.array(list(msg.O_F_ext_hat_K), dtype=float)

        # Offset removal
        off = np.hstack([self.offset_f, self.offset_t])
        corr = raw - off

        # Time step
        t_now = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
        if self.prev_t is None:
            dt = None
        else:
            dt = max(t_now - self.prev_t, 1e-6)


        # One-pole low-pass: y = (1-a)*y_prev + a*x
        if (dt is None) or (dt > 0.5):  # if first sample or big gap, reset filter state
            y = corr.copy()
            self.has_prev = True
        else:
            # Compute alpha from cutoff and dt: a = 1 - exp(-2*pi*fc*dt)
            alpha = 1.0 - np.exp(-2.0*np.pi*self.fc_hz*dt)
            alpha = np.clip(alpha, 0.0, 1.0)
            y = (1.0 - alpha) * self.prev + alpha * corr
            

        # Publish WrenchStamped (filtered, bias-removed)
        out = WrenchStamped()
        out.header.stamp = msg.header.stamp if msg.header.stamp else rospy.Time.now()
        out.header.frame_id = self.frame_id
        out.wrench.force.x,  out.wrench.force.y,  out.wrench.force.z  = y[0], y[1], y[2]
        out.wrench.torque.x, out.wrench.torque.y, out.wrench.torque.z = y[3], y[4], y[5]
        self.pub.publish(out)
        
        # === Log at 20 Hz ===
        if (self.last_log_t is None) or (t_now - self.last_log_t >= self.log_dt):
            with open(self.csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([t_now, y[0], y[1], y[2], y[3], y[4], y[5]])
            self.last_log_t = t_now

        # Save state
        self.prev   = y
        self.prev_t = t_now

def main():
    rospy.init_node("fr3_wrench_filter")
    WrenchFilterNode()
    rospy.spin()

if __name__ == "__main__":
    main()
