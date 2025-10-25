#!/usr/bin/env python3
import csv
import matplotlib.pyplot as plt

CSV_PATH = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/dynamic_box_lifting/csv/franka_EE_forces.csv"

t, Fx, Fy, Fz = [], [], [], []

with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            t.append(float(row["time"]))
            Fx.append(float(row["Fx"]))
            Fy.append(float(row["Fy"]))
            Fz.append(float(row["Fz"]))
        except (KeyError, ValueError):
            continue

plt.figure()
plt.plot(t, Fx, label="Fx")
plt.plot(t, Fy, label="Fy")
plt.plot(t, Fz, label="Fz")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.title("HEAL FT Sensor Forces")
plt.legend()
plt.grid(True)
plt.show()
