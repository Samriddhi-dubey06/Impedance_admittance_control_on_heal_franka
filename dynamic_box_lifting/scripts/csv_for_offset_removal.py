'''
 Uncomment the below script to remove the offset from the FT sensor attached on heal and comment the below written code of franka 
'''

#!/usr/bin/env python
# import rospy
# import csv
# from geometry_msgs.msg import WrenchStamped

# # Parameters
# OUTPUT_FILE = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/fr3_controllers/dynamic_box_lifting/data_for_ee_force/csv/ft_samples.csv"   # change path if needed
# NUM_SAMPLES = 700

# samples = []

# def ft_callback(msg):
#     global samples
#     if len(samples) < NUM_SAMPLES:
#         samples.append([
#             msg.header.stamp.to_sec(),
#             msg.wrench.force.x,
#             msg.wrench.force.y,
#             msg.wrench.force.z,
#             msg.wrench.torque.x,
#             msg.wrench.torque.y,
#             msg.wrench.torque.z
#         ])

#     if len(samples) == NUM_SAMPLES:
#         rospy.loginfo("Collected 700 samples. Writing to CSV...")
#         with open(OUTPUT_FILE, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(["time", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])
#             writer.writerows(samples)
#         rospy.loginfo("Saved to {}".format(OUTPUT_FILE))
#         rospy.signal_shutdown("Done collecting samples")

# def main():
#     rospy.init_node("ft_data_collector")
#     rospy.Subscriber("/ft_sensor", WrenchStamped, ft_callback)
#     rospy.spin()

# if __name__ == "__main__":
#     main()


'''This script is used to remove the offset from the Franka EE_wrench '''

import rospy
import csv
from franka_msgs.msg import FrankaState

NUM_SAMPLES = 500
OUTPUT_FILE = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/dynamic_box_lifting/csv/franka_wrench_samples.csv"

def main():
    rospy.init_node("franka_wrench_logger")

    samples = []

    def callback(msg):
        nonlocal samples

        # msg.O_F_ext_hat_K is a 6D vector: [Fx, Fy, Fz, Tx, Ty, Tz]
        wrench = list(msg.O_F_ext_hat_K)

        samples.append(wrench)
        rospy.loginfo("Sample %d/%d: %s", len(samples), NUM_SAMPLES, wrench)

        if len(samples) >= NUM_SAMPLES:
            # Save to CSV
            with open(OUTPUT_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])
                writer.writerows(samples)
            rospy.loginfo("Saved %d samples to %s", NUM_SAMPLES, OUTPUT_FILE)
            rospy.signal_shutdown("Data collection complete")

    rospy.Subscriber("/fr3/franka_state_controller/franka_states", FrankaState, callback)
    rospy.loginfo("Collecting %d samples from Franka wrench...", NUM_SAMPLES)

    rospy.spin()

if __name__ == "__main__":
    main()
