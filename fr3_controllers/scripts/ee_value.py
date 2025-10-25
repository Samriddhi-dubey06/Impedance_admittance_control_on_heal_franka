import numpy as np
from scipy.spatial.transform import Rotation as R

# Sample O_T_EE matrix as a flat list (row-major order)
O_T_EE =  [0.7933566036548426, -0.05344099875296326, -0.6064069253311171, 0.0, 0.5963848361246308, -0.13158079045312163, 0.7918406549453806, 0.0,
           -0.12210825802479719, -0.9898639074317324, -0.07251908773648563, 0.0, 0.3934835976760441, -0.6166060869545575, 0.42452071219483967, 1.0]


# Step 1: Convert to 4x4 matrix (column-major order)
M = np.array(O_T_EE).reshape((4, 4), order='F')

# Step 2: Extract translation
position = M[:3, 3]

# Step 3: Extract rotation matrix
R_mat = M[:3, :3]

# Step 4: Convert to quaternion [x, y, z, w]
quat = R.from_matrix(R_mat).as_quat()

# Step 5: Output as numpy arrays in specified format
position_array = np.array([round(p, 6) for p in position])
quat_array = np.array([round(q, 6) for q in quat])

print("Position array:")
print(position_array)
print("Quaternion array:")
print(quat_array)
