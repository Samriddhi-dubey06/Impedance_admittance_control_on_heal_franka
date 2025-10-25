#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  

class DSReacher:
    def __init__(self, x1, x2, beta1=1.0, beta2=0.5, A_gain=5.0):
        self.x1 = np.array(x1, dtype=float)
        self.x2 = np.array(x2, dtype=float)
        self.d = len(x1)
        self.I = np.eye(self.d)
        self.A_gain = A_gain * self.I
        self.beta1 = beta1
        self.beta2 = beta2
        self.reset()

    def reset(self):
        self.x = self.x1.copy()
        self.z = 1.0
        self.z_d = 0.0
        self.x_traj = [self.x.copy()]
        self.z_traj = [self.z]
        self.t_list = [0.0]

    def step(self, dt):
        delta = self.x2 - self.x1
        dz = -self.beta1 * (1 - np.exp(-self.beta2 * (self.z - self.z_d))) \
             / (1 + np.exp(-self.beta2 * (self.z - self.z_d)))
        self.z += dz * dt
        dx = -(self.z * self.A_gain + dz * self.I) @ delta - self.A_gain @ (self.x - self.x2)
        self.x += dx * dt
        return self.x.copy(), self.z, dz

    def simulate(self, T=3.0, dt=0.01):
        self.reset()
        t = 0.0
        while t < T:
            x, z, _ = self.step(dt)
            self.x_traj.append(x)
            self.z_traj.append(z)
            t += dt
            self.t_list.append(t)
        return np.array(self.x_traj), np.array(self.z_traj), np.array(self.t_list)


if __name__ == "__main__":
    ds = DSReacher(x1=[0, 0, 0], x2=[2, 2, 1], beta1=5.0, beta2=1, A_gain=10.0)
    ds.simulate(T=10.0, dt=0.02)
    x_traj = np.array(ds.x_traj)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("DS Reacher")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True)

    ax.plot(x_traj[:, 0], x_traj[:, 1], x_traj[:, 2], '--', color='gray', lw=1, alpha=0.5)
    ax.scatter(*ds.x1, color='green', s=80, label='Start', marker='s')
    ax.scatter(*ds.x2, color='red', s=80, label='Goal', marker='X')

    pt, = ax.plot([x_traj[0, 0]], [x_traj[0, 1]], [x_traj[0, 2]], 'o', color='dodgerblue', ms=10)
    ax.legend()

    def update(i):
        pt.set_data([x_traj[i, 0]], [x_traj[i, 1]])
        pt.set_3d_properties([x_traj[i, 2]])
        return pt,

    ani = FuncAnimation(fig, update, frames=len(x_traj), interval=30, blit=True)
    plt.tight_layout()
    plt.show()
