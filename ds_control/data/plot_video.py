import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load your data
df = pd.read_csv("/home/iitgn-robotics/bimanual_ws/src/ds_control/data/joint_log.csv")  # Make sure this file is in the same directory
y = df["fr3_vel3"].values
x = list(range(len(y)))

# Set up the plot
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, len(x))
ax.set_ylim(min(y) * 1.1, max(y) * 1.1)
ax.set_title("Animated Plot of fr3_joint4_vel")
ax.set_xlabel("Data Point Index")
ax.set_ylabel("Velocity")

# Animation update function
def update(frame):
    line.set_data(x[:frame], y[:frame])
    return line,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(x), blit=True, interval=10)

# Save the animation
ani.save("fr3_joint4_vel_animation.mp4", writer="ffmpeg", fps=60)
