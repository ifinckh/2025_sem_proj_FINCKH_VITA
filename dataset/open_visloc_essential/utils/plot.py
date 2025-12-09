import matplotlib.pyplot as plt
import numpy as np

def plot_lidar_camera_oxts(T_cam2oxts, T_lidar2oxts):
    """
    Plots the LiDAR and Camera frames relative to the OXTS frame using given transformation matrices.
    
    Parameters:
    T_cam2oxts (numpy.ndarray): 4x4 transformation matrix from the Camera to OXTS frame.
    T_lidar2oxts (numpy.ndarray): 4x4 transformation matrix from the LiDAR to OXTS frame.
    """
    # Extract rotation matrices and translation vectors
    rotation_cam = T_cam2oxts[:3, :3]
    translation_cam = T_cam2oxts[:3, 3]

    rotation_lidar = T_lidar2oxts[:3, :3]
    translation_lidar = T_lidar2oxts[:3, 3]

    # Extract coordinate axes
    cam_axes = np.array([rotation_cam[:, 0], rotation_cam[:, 1], rotation_cam[:, 2]])
    lidar_axes = np.array([rotation_lidar[:, 0], rotation_lidar[:, 1], rotation_lidar[:, 2]])

    # Colors and labels
    colors = ['r', 'b', 'g']  # X = Red, Y = Blue, Z = Green
    cam_labels = ['X_cam', 'Y_cam', 'Z_cam']
    lidar_labels = ['X_lidar', 'Y_lidar', 'Z_lidar']

    # Create 3D plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Adjust plot limits dynamically based on the largest translation value
    max_translation = max(np.max(abs(translation_cam)), np.max(abs(translation_lidar))) + 1

    # Plot OXTS frame axes (dashed)
    ax.quiver(0, 0, 0, 1, 0, 0, color='red', alpha=0.5, label="X_oxts")
    ax.quiver(0, 0, 0, 0, 1, 0, color='blue', alpha=0.5, label="Y_oxts")
    ax.quiver(0, 0, 0, 0, 0, 1, color='green', alpha=0.5, label="Z_oxts")

    # Plot Camera frame axes with translation
    for i in range(3):
        ax.quiver(*translation_cam, *cam_axes[i], color=colors[i], linewidth=2.5, linestyle='dashed', label=cam_labels[i])

    # Mark camera origin
    ax.scatter(*translation_cam, color="black", marker="o", s=80, label="Camera Origin")

    # Plot LiDAR frame axes with translation
    for i in range(3):
        ax.quiver(*translation_lidar, *lidar_axes[i], color=colors[i], linewidth=2.5, linestyle='dotted', label=lidar_labels[i])

    # Mark LiDAR origin
    ax.scatter(*translation_lidar, color="purple", marker="o", s=80, label="LiDAR Origin")

    # Labels and limits
    ax.set_xlim([-max_translation, max_translation])
    ax.set_ylim([-max_translation, max_translation])
    ax.set_zlim([-max_translation, max_translation])
    ax.set_xlabel('X (OXTS)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y (OXTS)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z (OXTS)', fontsize=12, labelpad=10)
    ax.set_title('LiDAR & Camera Frames Relative to OXTS', fontsize=14, pad=15)
    ax.legend()

    # Improve visualization with grid and viewing angle
    ax.grid(True)
    ax.view_init(elev=20, azim=30)  # Adjust the view angle

    # Show plot
    plt.show()