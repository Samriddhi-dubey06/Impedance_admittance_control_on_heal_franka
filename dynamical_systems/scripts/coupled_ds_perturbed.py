#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

class CoupledReacher:
    def __init__(self, x1_list, x2_list, beta1=1.0, beta2=5.0, A_gain=5.0):
        self.x1_list = [np.array(x, dtype=float) for x in x1_list]
        self.x2_list = [np.array(x, dtype=float) for x in x2_list]
        self.x_list = [x.copy() for x in self.x1_list]
        self.z_list = [1.0 for _ in x1_list]
        self.beta1 = beta1
        self.beta2 = beta2
        self.A = A_gain * np.eye(len(x1_list[0]))
        self.trajs = [[x.copy()] for x in self.x1_list]

    def compute_alpha(self, x, x1, x2):
        delta = x2 - x1
        return float(np.dot((x2 - x), delta) / np.dot(delta, delta))

    def step(self, dt):
        alphas = [self.compute_alpha(x, x1, x2)
                  for x, x1, x2 in zip(self.x_list, self.x1_list, self.x2_list)]
        z_c = sum(alphas) / (len(self.x_list) + 1)

        for i in range(len(self.x_list)):
            z = self.z_list[i]
            x, x1, x2 = self.x_list[i], self.x1_list[i], self.x2_list[i]
            delta = x2 - x1

            dz = -self.beta1 * (1 - np.exp(-self.beta2 * (z - z_c))) / (1 + np.exp(-self.beta2 * (z - z_c)))
            z += dz * dt
            self.z_list[i] = z

            dx = -(z * self.A + dz * np.eye(3)) @ delta - self.A @ (x - x2)
            x += dx * dt
            self.x_list[i] = x
            self.trajs[i].append(x.copy())

    def simulate(self, T=5.0, dt=0.02, perturb_step=None, perturb_agent=None):
        steps = int(T / dt)
        for step in range(steps):
            if perturb_step is not None and step == perturb_step:
                perturb = np.random.uniform(-2.5, 2.5, size=3)
                self.x_list[perturb_agent] += perturb
                print(f"Perturbed agent {perturb_agent} at step {step}: {perturb}")
            self.step(dt)
        return [np.array(traj) for traj in self.trajs]



if __name__ == "__main__":
    x1_list = [[0, 0, 0], [-2, 1, 0]]
    x2_list = [[2, 2, 1], [2, -1, 2]]

    sim = CoupledReacher(x1_list, x2_list, beta1=1.0, beta2=5.0, A_gain=2.0)
    trajs = sim.simulate(T=6.0, dt=0.02, perturb_step=60, perturb_agent=0)

    fig = plt.figure(figsize=(7, 6))
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.set_title("Coupled Reacher (3D)")
    ax3d.set_xlim(-3, 3); ax3d.set_ylim(-3, 3); ax3d.set_zlim(-1, 3)
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.grid(True)

    colors = ['dodgerblue', 'darkorange']
    dots = []

    for i, traj in enumerate(trajs):
        ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], '--', color=colors[i], alpha=0.4)
        ax3d.scatter(*x1_list[i], color=colors[i], marker='s', s=70)
        ax3d.scatter(*x2_list[i], color='red', marker='X', s=70)
        dot, = ax3d.plot([traj[0, 0]], [traj[0, 1]], [traj[0, 2]], 'o', color=colors[i], ms=10)
        dots.append(dot)

    def update(frame):
        for i, dot in enumerate(dots):
            dot.set_data([trajs[i][frame, 0]], [trajs[i][frame, 1]])
            dot.set_3d_properties([trajs[i][frame, 2]])
        return dots

    ani = FuncAnimation(fig, update, frames=len(trajs[0]), interval=30, blit=True)
    plt.tight_layout()
    plt.show()
