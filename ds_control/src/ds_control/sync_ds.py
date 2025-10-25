# ============================================================================
#  sync_ds.py
# ============================================================================
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def logistic_tau(t, t0, k):
    """
    Compute the logistic blending parameter tau and its derivative, given:
      - t:   current time
      - t0:  midpoint of the transition
      - k:   steepness factor
    Returns:
      tau(t)   = 1 / (1 + exp(-k*(t - t0)))
      d_tau(t) = derivative of tau(t) w.r.t. time
    """
    tau = 1.0 / (1.0 + np.exp(-k * (t - t0)))
    d_tau = k * np.exp(-k * (t - t0)) / ((1.0 + np.exp(-k * (t - t0))) ** 2)
    return tau, d_tau


class DSController:
    """
    DS Controller for controlling both position and orientation of a single arm.
    Uses:
     - Asynchronous references (x_d, q_d)
     - Synchronous references (x_V, q_V)
    Blends them via a parameter (tau) and uses P-control for both translation and rotation.
    """

    def __init__(self, K_pos, K_rot):
        """
        Initialize DSController with:
          K_pos: position gain
          K_rot: orientation (rotation) gain
        """
        self.K_pos = K_pos
        self.K_rot = K_rot

    def compute_command(
        self,
        pos_current,    pos_des_async,      pos_des_sync,
        quat_current,   quat_des_async,     quat_des_sync,
        tau, d_tau,
        dot_x_V = None,
        dot_q_V = None
    ):
        """
        Compute velocity (v_command) and angular velocity (omega_command)
        given the current state and references.

        Parameters:
          pos_current    (np.array):     current position (3D)
          pos_des_async  (np.array):     asynchronous (position) reference
          pos_des_sync   (np.array):     synchronous (position) reference
          quat_current   (np.array):     current orientation quaternion [x, y, z, w]
          quat_des_async (np.array):     asynchronous orientation quaternion [x, y, z, w]
          quat_des_sync  (np.array):     synchronous orientation quaternion [x, y, z, w]
          tau            (float):        blending parameter, 0 <= tau <= 1
          d_tau          (float):        time derivative of the blending parameter
          dot_x_V        (np.array):     optional linear velocity of x_V (default zero)
          dot_q_V        (np.array):     optional angular velocity of q_V in axis-angle form (default zero)

        Returns:
          (v_command, omega_command):
             v_command (3D)        -> linear velocity
             omega_command (3D)    -> angular velocity
        """
        if dot_x_V is None:
            dot_x_V = np.zeros_like(pos_des_sync)
        if dot_q_V is None:
            dot_q_V = np.zeros(3)

        #-----------------------
        # Position Blending
        #-----------------------
        # Weighted interpolation between the asynchronous (x_d) and synchronous (x_V) references
        x_target = tau * pos_des_sync + (1.0 - tau) * pos_des_async
        
        # v_command includes feedforward from dot_x_V, the partial derivative from tau, 
        # and a proportional error term.
        v_command = (
            tau * dot_x_V
            + d_tau * (pos_des_sync - pos_des_async)
            - self.K_pos * (pos_current - x_target)
        )

        #-----------------------
        # Orientation Blending
        #-----------------------
        # Use SLERP to get the blended orientation between q_d and q_V at fraction tau.
        key_times = [0, 1]
        key_rots = Rotation.from_quat([quat_des_async, quat_des_sync])  # note: [x, y, z, w] format
        slerp = Slerp(key_times, key_rots)
        q_target = slerp([tau]).as_quat()[0]

        # Compute the rotation error from q_R to q_target
        q_error = Rotation.from_quat(quat_current) * Rotation.from_quat(q_target).inv()
        e_rot = q_error.as_rotvec()  # rotation vector error

        # Angular velocity command
        omega_command = -self.K_rot * e_rot + tau * dot_q_V

        return v_command, omega_command


class DualArmController:
    """
    DualArmController manages two DSControllers (one for each arm)
    and orchestrates the blending (tau) among modes:
      - 'asynchronous' -> tau=0
      - 'synchronous'  -> tau=1
      - 'blending'     -> logistic transition from 0 to 1
    """

    def __init__(self, K_pos_left, K_rot_left, K_pos_right, K_rot_right):
        """
        Create two DSControllers, one for each arm (left/right).
        """
        self.left_controller = DSController(K_pos_left, K_rot_left)
        self.right_controller = DSController(K_pos_right, K_rot_right)
        self.mode = "asynchronous"
        self.blend_start_time = None
        self.blend_duration = 1.0
        # If blend_k is large, the logistic transition is abrupt; if smaller, it's smoother.
        self.blend_k = 10000.0

    def set_mode(self, mode, current_time=None, blend_duration=None):
        """
        Set the control mode:
          - 'asynchronous' -> tau=0
          - 'synchronous'  -> tau=1
          - 'blending'     -> logistic transition from 0 -> 1 around blend_start_time
        """
        self.mode = mode
        if mode == "blending" and current_time is not None:
            self.blend_start_time = current_time
            if blend_duration is not None:
                self.blend_duration = blend_duration

    def get_tau(self, current_time):
        """
        Determine tau and its derivative (d_tau) based on the current mode:
          'synchronous'  => tau=1
          'asynchronous' => tau=0
          'blending'     => logistic function from 0 to 1
        """
        if self.mode == "synchronous":
            return 1.0, 0.0
        elif self.mode == "asynchronous":
            return 0.0, 0.0
        elif self.mode == "blending":
            if self.blend_start_time is None:
                self.blend_start_time = current_time
            t0 = self.blend_start_time + self.blend_duration / 2.0
            tau, d_tau = logistic_tau(current_time, t0, self.blend_k)
            return tau, d_tau
        else:
            return 0.0, 0.0

    def compute_commands(
        self,
        pos_current_left, pos_current_right,   
        pos_des_async_left, pos_des_async_right,     
        pos_des_sync_left, pos_des_sync_right,
        quat_current_left, quat_current_right,
        quat_des_async_left, quat_des_async_right,
        quat_des_sync_left, quat_des_sync_right,
        current_time,
        dot_x_V_left = None, dot_x_V_right = None,
        dot_q_V_left = None, dot_q_V_right = None
    ):
        """
        Compute velocity & orientation commands for both arms.

        Parameters (per arm):
          pos_current,   pos_des_async,  pos_des_sync:  current pos, async pos, sync pos
          quat_current, quat_des_async, quat_des_sync:  current quat, async quat, sync quat
          dot_x_V_*, dot_q_V_*: optional velocity references for the synchronous target
          current_time:         used to compute tau in 'blending' mode

        Returns:
          (v_left, omega_left, v_right, omega_right)
        """
        tau, d_tau = self.get_tau(current_time)

        v_left, omega_left = self.left_controller.compute_command(
            pos_current_left, pos_des_async_left, pos_des_sync_left,
            quat_current_left, quat_des_async_left, quat_des_sync_left,
            tau, d_tau,
            dot_x_V_left, dot_q_V_left
        )

        v_right, omega_right = self.right_controller.compute_command(
            pos_current_right, pos_des_async_right, pos_des_sync_right,
            quat_current_right, quat_des_async_right, quat_des_sync_right,
            tau, d_tau,
            dot_x_V_right, dot_q_V_right
        )

        return v_left, omega_left, v_right, omega_right


# -----------------------------------------------------------------------------
# Example usage (if run as a script)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    # Create a dual arm controller
    dual_arm = DualArmController(
        K_pos_left=2.0, K_rot_left=2.0,
        K_pos_right=2.0, K_rot_right=2.0
    )

    # Start in asynchronous mode
    dual_arm.set_mode("asynchronous")

    # Initial positions, orientations
    x_R_left = np.array([0.0, 0.0, 0.0])
    x_R_right = np.array([0.5, 0.0, 0.0])
    q_R_left = np.array([1, 0, 0, 0])   # Identity quaternion
    q_R_right = np.array([1, 0, 0, 0]) # Identity quaternion

    # Asynchronous targets
    x_d_left = np.array([0.1, 0.1, 0.3])
    x_d_right = np.array([0.1, 0.1, 0.3])
    q_d_left = np.array([0.707, 0.0, 0.707, 0.0])  # 90 deg about X
    q_d_right = np.array([0.707, 0.0, 0.707, 0.0])

    # Synchronous (virtual) targets
    x_V_left = np.array([0.1, 0.1, 0.4])
    x_V_right = np.array([0.1, 0.1, 0.4])
    q_V_left = np.array([0.0, 0.0, 1.0, 0.0]) # 180 deg about Z
    q_V_right = np.array([0.0, 0.0, 1.0, 0.0])

    dt = 0.01
    start_time = time.time()

    for step in range(500):
        current_time = time.time() - start_time

        # Example: switch to synchronous mode after 2 seconds
        if current_time > 2.0 and dual_arm.mode != "synchronous":
            dual_arm.set_mode("synchronous")

        # Compute velocities/ang. velocities
        v_left, omega_left, v_right, omega_right = dual_arm.compute_commands(
            x_R_left, x_R_right,
            x_d_left, x_d_right,
            x_V_left, x_V_right,
            q_R_left, q_R_right,
            q_d_left, q_d_right,
            q_V_left, q_V_right,
            current_time
        )

        # Integrate positions (Euler method)
        x_R_left += dt * v_left
        x_R_right += dt * v_right

        # Integrate orientations
        # rotation vector is half dt * omega (exponential map)
        q_R_left = (
            Rotation.from_quat(q_R_left)
            * Rotation.from_rotvec(0.5 * dt * omega_left)
        ).as_quat()
        q_R_right = (
            Rotation.from_quat(q_R_right)
            * Rotation.from_rotvec(0.5 * dt * omega_right)
        ).as_quat()

        # Print every 50 steps as an example
        if step % 50 == 0:
            print(
                f"Time {current_time:.2f}s | "
                f"Left pos: {x_R_left}, Right pos: {x_R_right}, "
                f"Left quat: {q_R_left}, Right quat: {q_R_right}"
            )

        time.sleep(dt)