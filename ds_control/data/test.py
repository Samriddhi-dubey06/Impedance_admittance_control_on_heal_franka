import numpy as np
from scipy.spatial.transform import Rotation as R

# Your column-major 4×4 transform as a flat list
col_major = [
    0.875291, -0.483038,  0.023226,  0.0,
   -0.483547, -0.874878,  0.027782,  0.0,
    0.006901, -0.035548, -0.999344,  0.0,
    0.405869, -0.315065,  0.326436,  1.0
]

# 1) Reshape into a 4×4 matrix (column-major = Fortran order)
M = np.array(col_major).reshape((4, 4), order='F')

# 2) Extract position (translation)
position = M[:3, 3]

# 3) Extract rotation matrix
rot_mat = M[:3, :3]

# 4) Convert to quaternion [x, y, z, w]
quat = R.from_matrix(rot_mat).as_quat()

print("Position:", position)
print("Quaternion [x, y, z, w]:", quat)
