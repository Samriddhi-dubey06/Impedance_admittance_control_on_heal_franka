import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("~/ft_velocity_log.csv")

plt.figure()
plt.plot(df["time"], df["lambda_meas_0"], label="Measured Force Z")
plt.plot(df["time"], [1.0]*len(df), '--', label="Desired Force Z")
plt.xlabel("Time [s]")
plt.ylabel("Force [N]")
plt.legend()
plt.grid()

plt.figure()
for i in range(6):
    plt.plot(df["time"], df[f"v_c_{i}"], label=f"v_c[{i}]")
plt.xlabel("Time [s]")
plt.ylabel("Cartesian Velocity")
plt.legend()
plt.grid()

plt.figure()
for i in range(len([col for col in df.columns if "qdot_" in col])):
    plt.plot(df["time"], df[f"qdot_{i}"], label=f"qdot[{i}]")
plt.xlabel("Time [s]")
plt.ylabel("Joint Velocity [rad/s]")
plt.legend()
plt.grid()

plt.show()
