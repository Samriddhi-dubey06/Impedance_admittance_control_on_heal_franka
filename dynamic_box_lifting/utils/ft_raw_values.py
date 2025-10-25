'''
This code publishes the filtered value of the ft sensor attached on the heal robot on the topic /ft_sensor

'''


#!/usr/bin/env python
import rospy
import serial
import struct
from geometry_msgs.msg import WrenchStamped
import csv
import os


# ---------- FT sensor config ----------
PORT = '/dev/ttyUSB0'
BAUD = 115200
DF = 50.0     # Force divisor
DT = 1000.0   # Torque divisor

# ---------- Offsets (from your 700-sample CSV means) ----------
# You can override these with ROS params:
#   ~offset_force:  [Fx_off, Fy_off, Fz_off]
#   ~offset_torque: [Tx_off, Ty_off, Tz_off]
DEFAULT_OFFSET_FORCE  = [-1.289, 10.856, -6.630]   # N
DEFAULT_OFFSET_TORQUE = [ 0.197,  0.716, -0.018]   # Nm

def compute_checksum(data_bytes):
    return sum(data_bytes) & 0xFF

def parse_force_torque(data):
    # Extract 6 values (3 force, 3 torque) from data bytes
    ft_raw = [struct.unpack('>h', bytes(data[i:i+2]))[0] for i in range(1, 13, 2)]
    force = [ft_raw[i] / DF for i in range(3)]
    torque = [ft_raw[i] / DT for i in range(3, 6)]
    return force, torque

def main():
    rospy.init_node('robotous_ft_publisher')

    # Load offsets (allow overriding via ROS params)
    offset_force  = rospy.get_param('~offset_force',  DEFAULT_OFFSET_FORCE)
    offset_torque = rospy.get_param('~offset_torque', DEFAULT_OFFSET_TORQUE)

    if len(offset_force) != 3 or len(offset_torque) != 3:
        rospy.logwarn("Offset vectors must be length 3 each; falling back to defaults.")
        offset_force, offset_torque = DEFAULT_OFFSET_FORCE, DEFAULT_OFFSET_TORQUE

    pub = rospy.Publisher('/ft_sensor', WrenchStamped, queue_size=None)
    
        # === CSV setup ===
    csv_dir = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/dynamic_box_lifting/csv"
    os.makedirs(csv_dir, exist_ok=True)
    csv_file = os.path.join(csv_dir, "heal_ft_sensor.csv")  # <- new file name

    # Create file with header if it doesn't exist
    if not os.path.isfile(csv_file):
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])


    # === 20 Hz logger setup ===
    log_hz = 20.0
    log_dt = 1.0 / log_hz
    last_log_t = None


    try:
        ser = serial.Serial(PORT, BAUD, timeout=None)
        ser.reset_input_buffer()

        # Send start command
        command = [0x0B] + [0x00] * 7
        checksum = compute_checksum(command)
        packet = bytes([0x55] + command + [checksum, 0xAA])
        ser.write(packet)
        rospy.loginfo("Sent start command to FT sensor.")
        rospy.loginfo("Using offsets: force=%s, torque=%s", offset_force, offset_torque)

        while not rospy.is_shutdown():
            if ser.read() != b'\x55':
                continue

            data = ser.read(16)
            if len(data) != 16:
                continue

            received_checksum = ser.read(1)
            if len(received_checksum) != 1 or ser.read() != b'\xAA':
                continue

            if compute_checksum(data) != ord(received_checksum):
                rospy.logwarn("Checksum mismatch")
                continue

            # Parse raw values
            force, torque = parse_force_torque(data)

            # --------- Offset removal (per-channel subtraction) ---------
            force_corr  = [f - o for f, o in zip(force,  offset_force)]
            torque_corr = [t - o for t, o in zip(torque, offset_torque)]
            # ------------------------------------------------------------

            # Publish corrected wrench
            msg = WrenchStamped()
            msg.header.stamp = rospy.Time.now()
            msg.wrench.force.x,  msg.wrench.force.y,  msg.wrench.force.z  = force_corr
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = torque_corr
            pub.publish(msg)
            
            # === Log at 20 Hz ===
            t_now = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
            if (last_log_t is None) or (t_now - last_log_t >= log_dt):
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        t_now,
                        force_corr[0], force_corr[1], force_corr[2],
                        torque_corr[0], torque_corr[1], torque_corr[2]
                    ])
                last_log_t = t_now


            # Optional debug (corrected values)
            rospy.loginfo("Force(corr): %.2f %.2f %.2f | Torque(corr): %.3f %.3f %.3f",
                          *force_corr, *torque_corr)

    except serial.SerialException as e:
        rospy.logerr("Serial exception: {}".format(e))
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
