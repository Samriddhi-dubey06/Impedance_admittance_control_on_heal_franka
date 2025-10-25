#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# ========= Paths =========
SAVE_DIR = "/home/iitgn-robotics/ds_yash/bimanual_ws/src/fr3_controllers/dynamic_box_lifting/data_for_ee_force/graph/fft_plots"
os.makedirs(SAVE_DIR, exist_ok=True)  # Create the folder if it doesn't exist

# ========= Load the data =========
times = []
fz    = []

with open('/home/iitgn-robotics/ds_yash/bimanual_ws/src/fr3_controllers/dynamic_box_lifting/data_for_ee_force/graph/10N_des/forces.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        times.append(float(row['time']))
        fz.append(float(row['lambda_meas_2']))

times = np.array(times)
fz    = np.array(fz)

# Sampling information
dt    = np.mean(np.diff(times))
fs    = 1.0 / dt
N     = len(fz)

# ========= Remove DC offset =========
fz_zero_mean = fz - np.mean(fz)

# ========= FFT =========
X     = np.fft.fft(fz_zero_mean)
x_mag = np.abs(X)[:N // 2] * (2.0 / N)
freqs = np.fft.fftfreq(N, d=dt)[:N // 2]

# ========= Limit to 0–10 Hz =========
mask = freqs <= 10
freqs = freqs[mask]
x_mag = x_mag[mask]

# ========= Plot and Save =========
plt.figure()
plt.plot(freqs, x_mag)
plt.xlabel('Frequency [Hz]')
plt.ylabel('FFT Magnitude of Force (10 N)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.yscale('log')  # Logarithmic Y-axis
plt.title('FFT of Measured Force Z (0–10 Hz, log scale)')

save_path = os.path.join(SAVE_DIR, "fft_force_z_0_10Hz_log.png")
plt.savefig(save_path, dpi=150)
plt.close()

print(f"FFT plot saved at: {save_path}")
