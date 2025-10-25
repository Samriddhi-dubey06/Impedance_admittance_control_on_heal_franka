#!/usr/bin/env python3

import rospy
import serial
import threading
import csv
import os
from franka_msgs.msg import FrankaState

class FrankaSensorLogger:
    def __init__(self):
        self.serial_port = serial.Serial('/dev/ttyUSB0', 9600, timeout=0.05)

        self.latest_encoder = None
        self.latest_force = None
        self.lock = threading.Lock()

        self.reader_thread = threading.Thread(target=self.read_serial_loop)
        self.reader_thread.daemon = True
        self.reader_thread.start()

        self.csv_file = open(self.get_log_path(), mode='w')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['time', 'encoder_deg', 'force',
                                  'ee_x', 'ee_y', 'ee_z',
                                  'K_y', 'delay_jacobian', 'delay_cartesian',
                                  'noise_jacobian', 'noise_cartesian'])

        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.franka_callback)

        self.msg_count = 0
        self.last_rate_time = rospy.Time.now()

    def get_log_path(self):
        base_path = '/home/iitgn-robotics/franka_lab/franka_ros_ws/src/fr3_controllers/data'

        Ky = rospy.get_param('/eoi_controller/K_y', 0.0)
        dj = rospy.get_param('/eoi_controller/delay_jacobian', 0)
        dc = rospy.get_param('/eoi_controller/delay_cartesian', 0)
        nj = rospy.get_param('/eoi_controller/noise_jacobian', 0.0)
        nc = rospy.get_param('/eoi_controller/noise_cartesian', 0.0)
        trial_id = rospy.get_param('/eoi_controller/trial_id', 0)

        filename = f'Ky_{int(Ky)}_delayJ{dj}_delayC{dc}_noiseJ{int(nj)}_noiseC{int(nc)}_trial{trial_id}.csv'
        return os.path.join(base_path, filename)

    def read_serial_loop(self):
        while not rospy.is_shutdown():
            try:
                line = self.serial_port.readline().decode('utf-8').strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 2:
                        encoder = float(parts[0])
                        force = float(parts[1])
                        with self.lock:
                            self.latest_encoder = encoder
                            self.latest_force = force
            except Exception as e:
                rospy.logwarn_throttle(2.0, f"Serial read error: {e}")

    def franka_callback(self, msg):
        with self.lock:
            encoder = self.latest_encoder
            force = self.latest_force

        if encoder is None or force is None:
            return

        o_t_ee = msg.O_T_EE
        position = (o_t_ee[12], o_t_ee[13], o_t_ee[14])
        now = rospy.Time.now().to_sec()

        Ky = rospy.get_param('/eoi_controller/K_y', 0.0)
        dj = rospy.get_param('/eoi_controller/delay_jacobian', 0)
        dc = rospy.get_param('/eoi_controller/delay_cartesian', 0)
        nj = rospy.get_param('/eoi_controller/noise_jacobian', 0.0)
        nc = rospy.get_param('/eoi_controller/noise_cartesian', 0.0)

        self.csv_writer.writerow([now, encoder, force, *position, Ky, dj, dc, nj, nc])
        self.csv_file.flush()

        self.msg_count += 1
        if (now - self.last_rate_time.to_sec()) >= 1.0:
            #print("Logging rate: {} Hz".format(self.msg_count))
            self.msg_count = 0
            self.last_rate_time = rospy.Time.now()

    def __del__(self):
        self.csv_file.close()

if __name__ == '__main__':
    rospy.init_node('franka_sensor_logger')
    FrankaSensorLogger()
    rospy.spin()