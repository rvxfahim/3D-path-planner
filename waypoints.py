import numpy as np
import networkx as nx
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import time
import pandas as pd
import pyvista as pv

# Start timing the overall execution
start_time_total = time.time()

# Load point cloud from PCD file instead of generating random points
start_time = time.time()
def load_point_cloud(file_path):
    try:
        # Try to load with PyVista
        point_cloud = pv.read(file_path)
        points = np.array(point_cloud.points)
        print(f"Loaded {len(points)} points from {file_path}")
        return points
    except Exception as e:
        print(f"Error loading PCD file with PyVista: {e}")
        try:
            # Alternative: load with Open3D if PyVista fails
            import open3d as o3d
            point_cloud = o3d.io.read_point_cloud(file_path)
            points = np.asarray(point_cloud.points)
            print(f"Loaded {len(points)} points from {file_path}")
            return points
        except Exception as e2:
            print(f"Error loading PCD file with Open3D: {e2}")
            # Fallback: Generate random points
            print("Falling back to random point generation")
            return np.random.rand(1000, 3)

# Specify the path to your PCD file
pcd_file_path = "Scene.pcd"  # Change this to your file path
obstacle_points = load_point_cloud(pcd_file_path)

# Normalize point cloud to [0,1] range if needed
def normalize_point_cloud(points):
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    range_vals = max_vals - min_vals
    # Avoid division by zero
    range_vals[range_vals == 0] = 1
    normalized_points = (points - min_vals) / range_vals
    return normalized_points

obstacle_points = normalize_point_cloud(obstacle_points)
point_cloud_time = time.time() - start_time

# Create visualization-only downsampled point cloud
visualization_max_points = 5000  # Adjust based on your computer's performance
if len(obstacle_points) > visualization_max_points:
    # Random downsampling for visualization
    vis_indices = np.random.choice(len(obstacle_points), visualization_max_points, replace=False)
    obstacle_points_vis = obstacle_points[vis_indices]
    print(f"Downsampled visualization to {len(obstacle_points_vis)} points")
else:
    obstacle_points_vis = obstacle_points

# Create empty space points for navigation
# Generate a grid of points in the empty space (avoiding obstacles)
start_time = time.time()
grid_resolution = 0.05  # Adjust based on your environment size
x = np.arange(0, 1, grid_resolution)
y = np.arange(0, 1, grid_resolution)
z = np.arange(0, 1, grid_resolution)
X, Y, Z = np.meshgrid(x, y, z)
grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

# Filter grid points to keep only those sufficiently far from obstacles
min_obstacle_distance = 0.03  # Minimum clearance from obstacles
tree_obstacles = KDTree(obstacle_points)  # Use full resolution for accurate distances
distances, _ = tree_obstacles.query(grid_points)
empty_space_points = grid_points[distances > min_obstacle_distance]

# Downsample if needed (too many points will make graph creation slow)
if len(empty_space_points) > 3000:
    indices = np.random.choice(len(empty_space_points), 3000, replace=False)
    empty_space_points = empty_space_points[indices]
empty_space_creation_time = time.time() - start_time

# Timing: KDTree creation for empty space points
start_time = time.time()
tree = KDTree(empty_space_points)
kdtree_time = time.time() - start_time

# Timing: Graph creation of navigable empty space
start_time = time.time()
G = nx.Graph()
k_neighbors = 10  # Connect each point to its k nearest neighbors
for i, p in enumerate(empty_space_points):
    neighbors = tree.query(p, k=k_neighbors+1)[1][1:]  # Skip self
    for n in neighbors:
        # Check if line between points intersects obstacles
        line_points = 10  # Number of check points along the line
        t = np.linspace(0, 1, line_points).reshape(-1, 1)
        check_points = p.reshape(1, 3) * (1-t) + empty_space_points[n].reshape(1, 3) * t
        
        # Query distances to nearest obstacles
        check_distances, _ = tree_obstacles.query(check_points)
        if np.all(check_distances > min_obstacle_distance):
            # Path is clear, add edge
            G.add_edge(i, n, weight=np.linalg.norm(empty_space_points[i] - empty_space_points[n]))
graph_creation_time = time.time() - start_time

# Initialize start and goal
start_point = np.array([0.1, 0.2, 0.3])
goal_point = np.array([0.8, 0.8, 0.8])

# Find nearest empty space points
start_idx = tree.query(start_point)[1]
goal_idx = tree.query(goal_point)[1]

# Path calculation
path_calculation_time = 0
path = []
waypoints = []

def find_path():
    global path, waypoints, path_calculation_time
    path_start_time = time.time()
    try:
        # Path finding with A*
        path = nx.astar_path(G, start_idx, goal_idx, weight='weight')
        waypoints = empty_space_points[path]
        path_calculation_time = time.time() - path_start_time
        print(f"Found path with {len(waypoints)} waypoints in {path_calculation_time:.4f} seconds")
        return True
    except nx.NetworkXNoPath:
        path_calculation_time = time.time() - path_start_time
        print(f"No path found between start and goal (took {path_calculation_time:.4f} seconds)")
        return False

# Create a better structured visualization setup
def create_visualization():
    global fig, ax, colorbar, s_start_x, s_start_y, s_start_z, s_goal_x, s_goal_y, s_goal_z
    
    # Create figure with proper size and layout
    fig = plt.figure(figsize=(12, 10))
    
    # Add adjustable sliders at the bottom of the figure
    plt.subplots_adjust(bottom=0.35)
    
    # Create the main 3D axes for visualization
    ax = fig.add_subplot(111, projection='3d')
    
    # Create slider axes
    ax_start_x = plt.axes([0.2, 0.25, 0.65, 0.02])
    ax_start_y = plt.axes([0.2, 0.20, 0.65, 0.02])
    ax_start_z = plt.axes([0.2, 0.15, 0.65, 0.02])
    ax_goal_x = plt.axes([0.2, 0.10, 0.65, 0.02])
    ax_goal_y = plt.axes([0.2, 0.05, 0.65, 0.02])
    ax_goal_z = plt.axes([0.2, 0.00, 0.65, 0.02])
    
    # Create the sliders
    s_start_x = Slider(ax_start_x, 'Start X', 0, 1, valinit=start_point[0])
    s_start_y = Slider(ax_start_y, 'Start Y', 0, 1, valinit=start_point[1])
    s_start_z = Slider(ax_start_z, 'Start Z', 0, 1, valinit=start_point[2])
    s_goal_x = Slider(ax_goal_x, 'Goal X', 0, 1, valinit=goal_point[0])
    s_goal_y = Slider(ax_goal_y, 'Goal Y', 0, 1, valinit=goal_point[1])
    s_goal_z = Slider(ax_goal_z, 'Goal Z', 0, 1, valinit=goal_point[2])
    
    # Connect the update function to the sliders
    s_start_x.on_changed(update)
    s_start_y.on_changed(update)
    s_start_z.on_changed(update)
    s_goal_x.on_changed(update)
    s_goal_y.on_changed(update)
    s_goal_z.on_changed(update)
    
    # Initialize colorbar as None
    colorbar = None
    
    return fig, ax

# Function to update the plot
def update_plot():
    global colorbar
    plot_start_time = time.time()
    
    # First, remove the existing colorbar if it exists and is still valid
    if colorbar is not None and getattr(colorbar, 'ax', None) is not None:
        colorbar.remove()
        colorbar = None

    # Clear the axis (or selectively clear only plots without affecting the colorbar axes)
    ax.clear()
    
    # Now create the scatter plot and a new colorbar
    scatter = ax.scatter(
        obstacle_points_vis[:, 0], obstacle_points_vis[:, 1], obstacle_points_vis[:, 2],
        c=obstacle_points_vis[:, 2], cmap='viridis', alpha=0.6, s=10
    )
    
    colorbar = plt.colorbar(scatter, ax=ax, pad=0.1, fraction=0.046, aspect=20)
    colorbar.set_label('Height (Z)')
    
    # Plot the start, goal, and navigation points, etc.
    ax.scatter(start_point[0], start_point[1], start_point[2], c='green', s=100, label='Start')
    ax.scatter(goal_point[0], goal_point[1], goal_point[2], c='blue', s=100, label='Goal')
    ax.scatter(
        empty_space_points[start_idx][0], empty_space_points[start_idx][1], empty_space_points[start_idx][2],
        c='limegreen', s=80
    )
    ax.scatter(
        empty_space_points[goal_idx][0], empty_space_points[goal_idx][1], empty_space_points[goal_idx][2],
        c='royalblue', s=80
    )
    
    if path_found:
        ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c='red', s=30, label='Waypoints')
        ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 'r-', linewidth=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Navigation Through Height-Colored Space')
    ax.legend()
    
    plot_time = time.time() - plot_start_time
    print(f"Plot update took {plot_time:.4f} seconds")
    fig.canvas.draw_idle()

# Update function for sliders
def update(_):
    global start_point, goal_point, start_idx, goal_idx, path_found
    
    # Update start and goal from sliders
    start_point = np.array([s_start_x.val, s_start_y.val, s_start_z.val])
    goal_point = np.array([s_goal_x.val, s_goal_y.val, s_goal_z.val])
    
    # Find nearest navigable points
    start_idx = tree.query(start_point)[1]
    goal_idx = tree.query(goal_point)[1]
    
    # Recompute path
    path_found = find_path()
    
    # Update plot
    update_plot()

# Initialize path finding
path_found = find_path()

# Create the visualization
fig, ax = create_visualization()

# Initial plot
update_plot()

# Function to visualize the full graph (optional)
def visualize_graph():
    fig_graph = plt.figure(figsize=(10, 8))
    ax_graph = fig_graph.add_subplot(111, projection='3d')
    
    # Plot obstacles with height-based coloring
    scatter = ax_graph.scatter(obstacle_points_vis[:, 0], obstacle_points_vis[:, 1], obstacle_points_vis[:, 2], 
                   c=obstacle_points_vis[:, 2], cmap='viridis', alpha=0.6, s=10)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax_graph, pad=0.1)
    cbar.set_label('Height (Z)')
    
    # Plot navigation points
    ax_graph.scatter(empty_space_points[:, 0], empty_space_points[:, 1], empty_space_points[:, 2], 
                   c='lightblue', alpha=0.3, s=5, label='Navigation Points')
    
    # Plot edges (use a subset if there are too many)
    edge_count = 0
    max_edges = 1000  # Adjust based on your computer's capability
    edge_sample = list(G.edges())
    if len(edge_sample) > max_edges:
        edge_sample = np.random.choice(edge_sample, max_edges, replace=False)
    
    for u, v in edge_sample:
        ax_graph.plot([empty_space_points[u][0], empty_space_points[v][0]], 
                    [empty_space_points[u][1], empty_space_points[v][1]], 
                    [empty_space_points[u][2], empty_space_points[v][2]], 
                    'b-', alpha=0.1, linewidth=0.5)
        edge_count += 1
    
    # Plot path
    if path_found:
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            ax_graph.plot([empty_space_points[u][0], empty_space_points[v][0]], 
                        [empty_space_points[u][1], empty_space_points[v][1]], 
                        [empty_space_points[u][2], empty_space_points[v][2]], 
                        'r-', linewidth=2.0, label='Path' if i == 0 else "")
    
    # Plot start and goal
    ax_graph.scatter(start_point[0], start_point[1], start_point[2], 
                   c='green', s=100, label='Start')
    ax_graph.scatter(goal_point[0], goal_point[1], goal_point[2], 
                   c='blue', s=100, label='Goal')
    
    ax_graph.set_xlabel('X')
    ax_graph.set_ylabel('Y')
    ax_graph.set_zlabel('Z')
    ax_graph.set_title(f'Navigation Graph (showing {edge_count} of {len(G.edges())} edges)')
    ax_graph.legend()
    
    return fig_graph

# Uncomment to show the full graph visualization
# graph_fig = visualize_graph()

plt.show()