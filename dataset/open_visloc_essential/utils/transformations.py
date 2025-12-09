import numpy as np

def rotation_matrix_from_angles(roll, pitch, yaw, order="ZYX"):
    """
    Compute a rotation matrix from roll, pitch, and yaw angles.
    
    Parameters:
        roll (float): Rotation angle around the X-axis in radians.
        pitch (float): Rotation angle around the Y-axis in radians.
        yaw (float): Rotation angle around the Z-axis in radians.
        order (str): Rotation order, default is "ZYX" (yaw, pitch, roll).
    
    Returns:
        numpy.ndarray: A 3x3 rotation matrix.
    """
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    order_dict = {
        "ZYX": R_z @ R_y @ R_x,
        "XYZ": R_x @ R_y @ R_z,
        "XZY": R_x @ R_z @ R_y,
        "YXZ": R_y @ R_x @ R_z,
        "YZX": R_y @ R_z @ R_x,
        "ZXY": R_z @ R_x @ R_y,
    }
    
    if order not in order_dict:
        raise ValueError("Invalid rotation order. Choose from: 'ZYX', 'XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY'.")
    
    return order_dict[order]

def get_yaw_pitch_roll(matrix):
    """
    Extract yaw, pitch, and roll (ZYX Euler angles) from a 4x4 transformation matrix.
    
    Parameters:
        matrix (numpy.ndarray): A 4x4 transformation matrix.
    
    Returns:
        tuple: (yaw, pitch, roll) in radians.
    """
    if matrix.shape != (4, 4):
        raise ValueError("Input must be a 4x4 matrix.")
    
    rotation_matrix = matrix[:3, :3]
    
    if not np.allclose(np.dot(rotation_matrix.T, rotation_matrix), np.eye(3), atol=1e-6):
        raise ValueError("The upper-left 3x3 matrix is not a valid rotation matrix.")
    if not np.isclose(np.linalg.det(rotation_matrix), 1.0, atol=1e-6):
        raise ValueError("The determinant of the rotation matrix is not 1.")
    
    if np.isclose(rotation_matrix[2, 0], -1.0):
        yaw, pitch, roll = 0, -np.pi / 2, np.arctan2(-rotation_matrix[0, 1], rotation_matrix[0, 2])
    elif np.isclose(rotation_matrix[2, 0], 1.0):
        yaw, pitch, roll = 0, np.pi / 2, np.arctan2(rotation_matrix[0, 1], rotation_matrix[0, 2])
    else:
        pitch = np.arcsin(-rotation_matrix[2, 0])
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    
    return yaw, pitch, roll

def voxel_grid_downsample(points, voxel_size=0.1):
    """
    Downsample a LiDAR point cloud using voxel grid filtering.
    
    :param points: (N, 3) numpy array of LiDAR points (x, y, z)
    :param voxel_size: Size of the voxel grid (higher = more downsampling)
    :return: (M, 3) numpy array of downsampled points
    """
    # Compute voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    # Use np.unique to keep only one point per voxel
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)

    return points[unique_indices]

def filter_above_ground(lidar_points, ground_threshold=0.2):
    """
    Filters LiDAR points to keep only those above a ground threshold.

    :param lidar_points: (N, 3) numpy array of LiDAR points (x, y, z).
    :param ground_threshold: Minimum height above ground to keep (meters).
    :return: (M, 3) numpy array of filtered LiDAR points.
    """
    # Keep only points where Z > ground_threshold
    above_ground_points = lidar_points[lidar_points[:, 2] > ground_threshold]
    return above_ground_points

def compute_heading(utm_e1, utm_n1, utm_e2, utm_n2):
    """
    Compute heading angle from two UTM coordinate points.
    
    Args:
        utm_e1, utm_n1: UTM coordinates of the first point
        utm_e2, utm_n2: UTM coordinates of the second point
    
    Returns:
        Heading angle in degrees (0° = North, 90° = East)
    """
    # Compute differences
    delta_e = utm_e2 - utm_e1  # Change in Easting
    delta_n = utm_n2 - utm_n1  # Change in Northing

    # Compute heading using atan2
    heading_rad = np.arctan2(delta_e, delta_n)

    # Convert to degrees
    heading_deg = np.degrees(heading_rad)

    # Ensure positive heading in range [0, 360]
    if heading_deg < 0:
        heading_deg += 360

    return heading_deg