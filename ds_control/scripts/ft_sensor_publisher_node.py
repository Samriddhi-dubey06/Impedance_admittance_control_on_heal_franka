#!/usr/bin/env python
import rospy
import serial
import struct
from geometry_msgs.msg import WrenchStamped

# FT sensor configuration
PORT = '/dev/ttyUSB0'
BAUD = 115200
DF = 50.0     # Force divisor
DT = 1000.0   # Torque divisor

# Filtering
FILTER_ALPHA = 0.01
filtered_force = [0.0, 0.0, 0.0]
filtered_torque = [0.0, 0.0, 0.0]
DEADBAND = 0.2  # Optional: threshold below which values are zeroed

# Offset storage
force_offset = [0.0, 0.0, 0.0]
torque_offset = [0.0, 0.0, 0.0]
zeroed = False  # Flag to ensure offset is captured once

def compute_checksum(data_bytes):
    return sum(data_bytes) & 0xFF

def parse_force_torque(data):
    ft_raw = [struct.unpack('>h', bytes(data[i:i+2]))[0] for i in range(1, 13, 2)]
    force = [ft_raw[i] / DF for i in range(3)]
    torque = [ft_raw[i] / DT for i in range(3, 6)]
    return force, torque

def zero_sensor(force, torque):
    global force_offset, torque_offset, zeroed
    force_offset = force
    torque_offset = torque
    zeroed = True
    rospy.loginfo("FT sensor zeroed (offsets captured).")

def apply_deadband(val, threshold):
    return [v if abs(v) > threshold else 0.0 for v in val]

def main():
    global zeroed, force_offset, torque_offset, filtered_force, filtered_torque

    rospy.init_node('robotous_ft_publisher')
    pub = rospy.Publisher('/ft_sensor', WrenchStamped, queue_size=10)

    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        ser.reset_input_buffer()

        command = [0x0B] + [0x00] * 7
        checksum = compute_checksum(command)
        packet = bytes([0x55] + command + [checksum, 0xAA])
        ser.write(packet)
        rospy.loginfo("Sent start command to FT sensor.")

        rate = rospy.Rate(100)  # 100 Hz

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

            force, torque = parse_force_torque(data)

            if not zeroed:
                zero_sensor(force, torque)

            # Subtract offset
            corrected_force = [f - f0 for f, f0 in zip(force, force_offset)]
            corrected_torque = [t - t0 for t, t0 in zip(torque, torque_offset)]

            # Apply low-pass filter
            for i in range(3):
                filtered_force[i] = FILTER_ALPHA * corrected_force[i] + (1 - FILTER_ALPHA) * filtered_force[i]
                filtered_torque[i] = FILTER_ALPHA * corrected_torque[i] + (1 - FILTER_ALPHA) * filtered_torque[i]

            # Apply deadband (optional)
            clean_force = apply_deadband(filtered_force, DEADBAND)
            clean_torque = apply_deadband(filtered_torque, DEADBAND)

            # Publish
            msg = WrenchStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "ft_sensor_frame"
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = clean_force
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = clean_torque
            pub.publish(msg)

            rospy.loginfo_throttle(1, "Force: {:.2f} {:.2f} {:.2f} | Torque: {:.3f} {:.3f} {:.3f}".format(
                *clean_force, *clean_torque))

            rate.sleep()

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
