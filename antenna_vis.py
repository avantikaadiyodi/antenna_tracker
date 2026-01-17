import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
from datetime import datetime

def generate_trajectory(t):
    """
    Generates a dummy 3D trajectory (figure-8 on a sphere surface roughly).
    """
    # Parameters for the trajectory
    radius = 10
    omega = 0.5
    
    x = radius * np.sin(omega * t)
    y = radius * np.sin(omega * t) * np.cos(omega * t)
    z = radius * np.cos(omega * t) + 10 # Offset z so it's largely above the "ground"
    
    return x, y, z

def create_parabolic_dish(diameter=2, depth=0.5, resolution=20): # Create a mesh for a parabolic dish antenna model
    """
    Creates a mesh for a parabolic dish.
    Returns x, y, z arrays of the dish surface pointing up (positive z).
    """
    r = np.linspace(0, diameter/2, resolution)
    theta = np.linspace(0, 2*np.pi, resolution)
    r, theta = np.meshgrid(r, theta)
    
    # Parabola equation: z = a * r^2
    # at r = diameter/2, z = depth => depth = a * (d/2)^2 => a = depth / (d/2)^2
    a = depth / ((diameter/2)**2)
    
    x_dish = r * np.cos(theta)
    y_dish = r * np.sin(theta)
    z_dish = a * r**2
    
    return x_dish, y_dish, z_dish

def rotation_matrix_from_vectors(vec1, vec2): # Find the rotation matrix that aligns vec1 to vec2
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3) # Normalize vectors
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v) # Magnitude of cross product
    
    # Handle parallel and anti-parallel cases
    if s < 1e-6:
        if c > 0:
            # Parallel, return identity
            return np.eye(3) # Identity matrix
        else:
            # Anti-parallel, rotate 180 degrees around any orthogonal axis
            # Find an orthogonal vector
            if np.abs(a[0]) > np.abs(a[2]): # If x component is largest
                ortho = np.array([-a[1], a[0], 0]) # Orthogonal to a
            else: # If y component is largest
                ortho = np.array([0, -a[2], a[1]]) # Orthogonal to a
            ortho = ortho / np.linalg.norm(ortho) # Normalize
            # Rodriques formula for 180 degree rotation around ortho
            K = np.array([[0, -ortho[2], ortho[1]], [ortho[2], 0, -ortho[0]], [-ortho[1], ortho[0], 0]]) # Skew-symmetric matrix
            return np.eye(3) + 2 * (K @ K) # 180 rotation matrix simplified

    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) # Skew-symmetric matrix
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2)) # Rodrigues formula
    return rotation_matrix

def update(frame, trajectory_time, ax_3d, target_dot, tracking_lines, data_lines, az_hist, el_hist, range_hist): # Update function for animation
    # frame: index of the current frame
    # trajectory_time: array of time values for the trajectory
    # ax_3d: 3D axes for the dish and target
    # target_dot: Dot representing the target
    # tracking_lines: Lines representing the tracking path
    # data_lines: Lines representing the data history
    # az_hist, el_hist, range_hist: Pre-calculated history arrays
    t = trajectory_time[frame] # Current time in trajectory
    
    # 1. Update Target Position
    tx, ty, tz = generate_trajectory(t)
    target_pos = np.array([tx, ty, tz])
    
    # 2. Update Antenna Orientation
    # Antenna is at origin (0,0,0)
    antenna_pos = np.array([0, 0, 0])
    pointing_vec = target_pos - antenna_pos
    
    # Calculate Range, Azimuth, Elevation
    dist = np.linalg.norm(pointing_vec) 
    azimuth = np.degrees(np.arctan2(ty, tx))
    elevation = np.degrees(np.arcsin(tz / dist))
    
    # Generate dish geometry (base, pointing up)
    dx, dy, dz = create_parabolic_dish(diameter=3, depth=1)
    
    # Rotate dish to point at target
    # Default dish points in +Z direction
    default_dir = np.array([0, 0, 1])
    # Handle singularity when target is exactly above or exactly same direction
    try:
        rot_mat = rotation_matrix_from_vectors(default_dir, pointing_vec)
    except:
        rot_mat = np.eye(3) # Identity matrix

    # Apply rotation to surface points
    # Reshape for matrix multiplication
    original_shape = dx.shape # Shape of the dish
    pts = np.vstack([dx.flatten(), dy.flatten(), dz.flatten()]) # Flatten the dish points
    rot_pts = rot_mat @ pts # Rotate the dish points
    
    # Reshape the rotated points
    rdx = rot_pts[0, :].reshape(original_shape)
    rdy = rot_pts[1, :].reshape(original_shape)
    rdz = rot_pts[2, :].reshape(original_shape)
    
    # Update 3D plot
    # Remove old surface
    if ax_3d.collections:
        for coll in ax_3d.collections:
            coll.remove()
    
    # Replot Dish
    # Using plot_surface with color/shading
    ax_3d.plot_surface(rdx, rdy, rdz, color='cyan', alpha=0.6, edgecolors='b', linewidth=0.5)
    
    # Target
    target_dot.set_data([tx], [ty])
    target_dot.set_3d_properties([tz])
    
    # Line from antenna to target
    tracking_lines[0].set_data([0, tx], [0, ty])
    tracking_lines[0].set_3d_properties([0, tz])
    
    # 3. Update 2D Plots
    # We append data to lists (using a global or simple list in closure)
    # For this simple script, we'll re-slice from global arrays if pre-calculated, or just append
    # Let's assume we pre-calculate the whole trajectory for plotting 'history' trails easier
    
    # Update current time markers on 2D plots
    data_lines['az'].set_data(trajectory_time[:frame], az_hist[:frame])
    data_lines['el'].set_data(trajectory_time[:frame], el_hist[:frame])
    data_lines['range'].set_data(trajectory_time[:frame], range_hist[:frame])
    
    return

def save_animation(ani, output_dir="./track_plots"):
    """
    Saves the animation as a video file.
    Creates directory structure: ./track_plots/YYYYMMDD/track_plot_HHMMSS.mp4
    """
    # Get current timestamp
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    
    # Create directory structure
    save_dir = os.path.join(output_dir, date_str)
    os.makedirs(save_dir, exist_ok=True)
    
    # Create filename
    filename = f"track_plot_{time_str}.mp4"
    filepath = os.path.join(save_dir, filename)
    
    print(f"Saving animation to: {filepath}")
    print("This may take a moment...")
    
    # Save the animation
    # Using FFMpegWriter with reasonable settings
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='Antenna Tracker'), bitrate=1800)
    ani.save(filepath, writer=writer)
    
    print(f"Animation saved successfully to: {filepath}")
    return filepath

def main():
    """Main function to run the antenna tracking visualization."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Antenna Tracking Visualization')
    parser.add_argument('--save', action='store_true', 
                        help='Save the animation as a video file')
    parser.add_argument('--output-dir', type=str, default='./track_plots',
                        help='Output directory for saved videos (default: ./track_plots)')
    args = parser.parse_args()

    # --- Main Setup ---

    # Pre-calculate data for smoother 2D plots limits
    steps = 200
    time_array = np.linspace(0, 20, steps) # 20 seconds
    pos_history = np.array([generate_trajectory(ti) for ti in time_array]) # shape (steps, 3)
    az_history = []
    el_history = []
    range_history = []


    for p in pos_history: # Calculate range, azimuth, and elevation for each point in the trajectory
        r = np.linalg.norm(p)
        az = np.degrees(np.arctan2(p[1], p[0]))
        el = np.degrees(np.arcsin(p[2] / r))
        range_history.append(r)
        az_history.append(az)
        el_history.append(el)

    # Setup Figures
    fig = plt.figure(figsize=(14, 8))

    # 3D Plot (Left side)
    ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax_3d.set_xlim(-15, 15)
    ax_3d.set_ylim(-15, 15)
    ax_3d.set_zlim(0, 25)
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title("3D Antenna Tracking")

    # Initial objects
    target_dot, = ax_3d.plot([], [], [], 'ro', markersize=8, label='Target')
    track_line, = ax_3d.plot([], [], [], 'k--', linewidth=1, label='Tracking Vector')
    ax_3d.legend()

    # 2D Plots (Right side, stacked)
    ax_az = fig.add_subplot(3, 2, 2)
    ax_az.set_title("Azimuth (deg)")
    ax_az.set_xlim(0, 20)
    ax_az.set_ylim(min(az_history)-10, max(az_history)+10)
    line_az, = ax_az.plot([], [], 'r-')

    ax_el = fig.add_subplot(3, 2, 4)
    ax_el.set_title("Elevation (deg)")
    ax_el.set_xlim(0, 20)
    ax_el.set_ylim(min(el_history)-10, max(el_history)+10)
    line_el, = ax_el.plot([], [], 'g-')

    ax_range = fig.add_subplot(3, 2, 6)
    ax_range.set_title("Range (m)")
    ax_range.set_xlim(0, 20)
    ax_range.set_ylim(min(range_history)-1, max(range_history)+1)
    line_range, = ax_range.plot([], [], 'b-')

    plt.tight_layout()

    data_lines = {'az': line_az, 'el': line_el, 'range': line_range}
    tracking_lines = [track_line]

    ani = animation.FuncAnimation(fig, update, frames=steps, fargs=(time_array, ax_3d, target_dot, tracking_lines, data_lines, az_history, el_history, range_history), interval=50, blit=False)

    # Save animation if requested
    if args.save:
        save_animation(ani, args.output_dir)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
