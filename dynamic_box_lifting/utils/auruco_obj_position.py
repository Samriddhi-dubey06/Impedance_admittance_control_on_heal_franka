#!/usr/bin/env python3
import os
import csv
import math
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from tf.transformations import euler_from_quaternion

# Fixed quaternion for 180° about (1/√2, -1/√2, 0)
Q_R = (1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0.0, 0.0)  # (x, y, z, w)

# ---------- quaternion helpers ----------
def qmul(q1, q2):
    """Quaternion multiply q = q1 ⊗ q2, each as (x, y, z, w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return (x, y, z, w)

def normalize(q):
    x, y, z, w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    inv = 1.0 / n
    return (x*inv, y*inv, z*inv, w*inv)

# ---------- callback ----------
def pose_callback(msg):
    # Transform position: x' = -y, y' = -x, z' = -z
    px, py, pz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
    tx, ty, tz = -py, -px, -pz

    # Transform orientation: q_out = q_R ⊗ q_in
    qi = (msg.pose.orientation.x,
          msg.pose.orientation.y,
          msg.pose.orientation.z,
          msg.pose.orientation.w)
    qo = normalize(qmul(Q_R, qi))

    # ---- Publisher 1: PoseStamped on /transformed_pos
    out = PoseStamped()
    out.header = msg.header
    out.header.frame_id = rospy.get_param("~output_frame", "marker_rotated")
    out.pose.position.x = tx
    out.pose.position.y = ty
    out.pose.position.z = tz
    out.pose.orientation.x = qo[0]
    out.pose.orientation.y = qo[1]
    out.pose.orientation.z = qo[2]
    out.pose.orientation.w = qo[3]
    pub_pose.publish(out)

    # ---- Publisher 2: Float64MultiArray [x, y, z, roll, pitch, yaw] on /transformed_pos_euler
    roll, pitch, yaw = euler_from_quaternion([qo[0], qo[1], qo[2], qo[3]])  # radians
    if publish_deg:
        roll  = math.degrees(roll)
        pitch = math.degrees(pitch)
        yaw   = math.degrees(yaw)

    arr = Float64MultiArray()
    arr.layout.dim = [MultiArrayDimension(label="fields", size=6, stride=6)]
    arr.data = [tx, ty, tz, roll, pitch, yaw]
    pub_euler.publish(arr)

    # ---- CSV logging (transformed pose + RPY) ----
    # Prefer message time; fall back to wall time
    if msg.header.stamp and (msg.header.stamp.secs != 0 or msg.header.stamp.nsecs != 0):
        t = msg.header.stamp.to_sec()
    else:
        t = rospy.Time.now().to_sec()
    try:
        csv_writer.writerow([t, tx, ty, tz, roll, pitch, yaw])
        csv_file.flush()
    except Exception as e:
        rospy.logwarn_throttle(5.0, "CSV write failed: %s", str(e))

def _on_shutdown():
    try:
        csv_file.flush()
        csv_file.close()
    except Exception:
        pass
    rospy.loginfo("CSV closed and node shutting down.")

# ---------- main ----------
if __name__ == "__main__":
    rospy.init_node("pose_transformer", anonymous=True)

    # Params (you can override via rosparam)
    pose_topic      = rospy.get_param("~input_topic", "/aruco/id6/pose")
    out_pose_topic  = rospy.get_param("~output_topic", "/transformed_pos")
    out_euler_topic = rospy.get_param("~output_euler_topic", "/transformed_pos_euler")
    publish_deg     = rospy.get_param("~euler_in_degrees", False)

    # CSV settings:
    # Default path: your utils folder + filename 'aruco_box_positions.csv'
    default_csv_path = os.path.expanduser(
        "~/ds_yash/bimanual_ws/src/fr3_controllers/dynamic_box_lifting/data_for_ee_force/utils/aruco_box_positions.csv"
    )
    csv_path   = rospy.get_param("~csv_path", default_csv_path)
    csv_append = rospy.get_param("~csv_append", False)  # True=append, False=overwrite
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not (csv_append and os.path.exists(csv_path) and os.path.getsize(csv_path) > 0)
    csv_mode = "a" if csv_append else "w"
    csv_file = open(csv_path, csv_mode, newline="")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(["time_sec", "x", "y", "z", "roll", "pitch", "yaw"])
        csv_file.flush()

    # Publishers & subscriber
    pub_pose  = rospy.Publisher(out_pose_topic, PoseStamped, queue_size=10)
    pub_euler = rospy.Publisher(out_euler_topic, Float64MultiArray, queue_size=10)
    rospy.Subscriber(pose_topic, PoseStamped, pose_callback)

    rospy.on_shutdown(_on_shutdown)
    rospy.loginfo("Pose transformer running: %s -> %s (PoseStamped) and %s ([x y z r p y])",
                  pose_topic, out_pose_topic, out_euler_topic)
    rospy.loginfo("Logging transformed pose to CSV: %s (append=%s)", csv_path, str(csv_append))
    rospy.spin()
