import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV file
file_path = '/home/iitgn-robotics/bimanual_ws/src/ds_control/data/twist_log.csv'

# Load data
df = pd.read_csv(file_path)

# Replace time with timestep (assuming 100 Hz)
df['time'] = df.index / 100.0

# Heal Linear Velocity
plt.figure()
plt.plot(df['time'], df['heal_vx'], label='vx')
plt.plot(df['time'], df['heal_vy'], label='vy')
plt.plot(df['time'], df['heal_vz'], label='vz')
plt.title('Heal Linear Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)

# Heal Angular Velocity
plt.figure()
plt.plot(df['time'], df['heal_wx'], label='wx')
plt.plot(df['time'], df['heal_wy'], label='wy')
plt.plot(df['time'], df['heal_wz'], label='wz')
plt.title('Heal Angular Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)

# Franka Linear Velocity
plt.figure()
plt.plot(df['time'], df['fr3_vx'], label='vx')
plt.plot(df['time'], df['fr3_vy'], label='vy')
plt.plot(df['time'], df['fr3_vz'], label='vz')
plt.title('Franka Linear Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)

# Franka Angular Velocity
plt.figure()
plt.plot(df['time'], df['fr3_wx'], label='wx')
plt.plot(df['time'], df['fr3_wy'], label='wy')
plt.plot(df['time'], df['fr3_wz'], label='wz')
plt.title('Franka Angular Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)

plt.show()
