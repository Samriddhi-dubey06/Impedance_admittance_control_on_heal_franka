import numpy as np
from typing import List
import rospy
from scipy.spatial.transform import Rotation, Slerp

class DSCoordinationFramework:
    """
    DS coordination framework for multi-agent manipulation.

    Attributes:
        x1_list     : List[np.ndarray], start positions for each agent
        x2_list     : List[np.ndarray], target positions for each agent
        q1_list     : List[np.ndarray], start orientations (quaternions)
        q2_list     : List[np.ndarray], target orientations (quaternions)
        k1, k2      : float, DS stiffness gains
        kappa       : float, coupling gain
        k_rot       : float, orientation gain
        alpha_min   : float, minimum proportional gain
        alpha_max   : float, maximum proportional gain
        dt          : float, time step for internal integration
        z_list      : List[float], current phase variable for each agent
    """

    def __init__(self,
                x1_list: List[np.ndarray],
                x2_list: List[np.ndarray],
                q1_list: List[np.ndarray],
                q2_list: List[np.ndarray],
                k1: float = 5.0,
                k2: float = 10.0,
                kappa: float = 1.0,
                k_rot: float = 2.0,
                alpha_min: float = 1.0,
                alpha_max: float = 5.0,
                dt: float = None):
        
        assert len(x1_list) == len(x2_list), "start/target lists must match length"
        self.x1_list    = x1_list
        self.x2_list    = x2_list
        self.q1_list    = q1_list
        self.q2_list    = q2_list
        self.n          = len(x1_list)
        self.d          = x1_list[0].shape[0]
        
        # DS gains
        self.k1         = k1
        self.k2         = k2
        self.kappa      = kappa
        
        # Orientation gain
        self.k_rot      = k_rot
        
        # Adaptive gain range
        self.alpha_min  = alpha_min
        self.alpha_max  = alpha_max
        
        # Integration timestep
        self.dt         = dt if dt is not None else 1.0 / 100.0

        # Phase variables initialized at 0
        self.z_list     = [0.0 for _ in range(self.n)]   
    
    def compute_z(self, x, x1, x2) -> float:
        """ 
        Compute relative progress scalar z 
        """
        if np.linalg.norm(x2 - x1) < 1e-6:
            return 1.0
        return np.clip((np.linalg.norm(x - x1)/np.linalg.norm(x2 - x1)), 0.0, 1.0)

    def gamma(self, z, x1, x2) -> np.ndarray:
        """ 
        Interpolated intermediate target 
        """
        return x1 + z * (x2 - x1)
    
    
    def compute_twists(
        self,
        x_list: List[np.ndarray],
        q_list: List[np.ndarray],
        dt: float = None
    ) -> List[np.ndarray]:
        """
        Compute full 6D twist [vx,vy,vz, wx,wy,wz] for each agent.

        :param x_list: List of current positions
        :param q_list: List of current orientations (quaternions [x,y,z,w])
        :param dt:     Optional timestep override
        :return:       List of 6D twist vectors
        """
        
        dt = dt if dt is not None else self.dt

        for i in range(self.n):
            self.z_list[i] = self.compute_z(x_list[i], self.x1_list[i], self.x2_list[i])
        
        z_d = 1.0
        z_c = sum(self.z_list) / self.n
        
        twists: List[np.ndarray] = []
        
        for i in range(self.n):
            z    = self.z_list[i]
            x    = x_list[i]
            x1   = self.x1_list[i]
            x2   = self.x2_list[i]

            # Finite-time coupling exponents
            gamma_ft = 0.5   
            eps      = 1e-3
            
            # Individual and coupling dynamics (dz)
            dz_ind = -self.k1 * (z - z_d) / ((abs(z - z_d) + eps)**(1 - gamma_ft))
            dz_cpl = -self.k2 * (z - z_c) / ((abs(z - z_c) + eps)**(1 - gamma_ft))
            dz     = dz_ind + self.kappa * dz_cpl
            
            z_new           = np.clip(z + dz * dt, 0.0, 1.0)
            self.z_list[i]  = z_new
            
            # Compute intermediate target on the line
            x_target = self.gamma(z_new, x1, x2)
            
            # Adaptive proportional gain based on current error
            err_pos  = np.linalg.norm(x - x_target)
            delta    = 1e-2
            alpha_i = self.alpha_min + (self.alpha_max - self.alpha_min) * (err_pos / (err_pos + delta))
            v_lin    = -alpha_i * (x - x_target)
            
            # Orientation target via SLERP
            key_times = [0.0, 1.0]
            key_rots  = Rotation.from_quat([
                self.q1_list[i], self.q2_list[i]
            ])
            slerp     = Slerp(key_times, key_rots)
            q_target  = slerp([z_new]).as_quat()[0]

            # Rotation error and angular velocity
            q_err  = Rotation.from_quat(q_list[i]) * Rotation.from_quat(q_target).inv()
            e_rot  = q_err.as_rotvec()
            v_ang  = -self.k_rot * e_rot

            # Combine into 6D twist
            twist = np.hstack((v_lin, v_ang))
            twists.append(twist)

        return twists
        
        