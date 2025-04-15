import faiss
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

res_obstacles = faiss.StandardGpuResources()
index_flat_obs = faiss.IndexFlatL2(3)
gpu_index_obstacles = faiss.index_cpu_to_gpu(res_obstacles, 0, index_flat_obs)
gpu_index_obstacles.add(obstacle_points.astype(np.float32))

# Normalize point cloud to [0,1] range if needed
def normalize_point_cloud(points):
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    range_vals = max_vals - min_vals
    # Avoid division by zero
    range_vals[range_vals == 0] = 1
    normalized_points = (points - min_vals) / range_vals
    return normalized_points

# obstacle_points = normalize_point_cloud(obstacle_points)
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
min_vals = np.min(obstacle_points, axis=0)
max_vals = np.max(obstacle_points, axis=0)
grid_resolution = 0.5  # Adjust resolution for grid points
x_vals = np.arange(min_vals[0], max_vals[0], grid_resolution)
y_vals = np.arange(min_vals[1], max_vals[1], grid_resolution)
z_vals = np.arange(min_vals[2], max_vals[2], grid_resolution)
X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)
grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

# Filter grid points to keep only those sufficiently far from obstacles
min_obstacle_distance = .2  # Minimum clearance from obstacles
# tree_obstacles = KDTree(obstacle_points)  # Use full resolution for accurate distances
# distances, _ = tree_obstacles.query(grid_points)
# empty_space_points = grid_points[distances > min_obstacle_distance]

grid_points_f32 = grid_points.astype(np.float32)
D_g, I_g = gpu_index_obstacles.search(grid_points_f32, 1)
distances = np.sqrt(D_g[:, 0])
empty_space_points = grid_points[distances > min_obstacle_distance]

def is_potential_edge(x, y, z, cell_size, gpu_index_obstacles):
    """Check if a cell might be at an edge by sampling its neighborhood"""
    # Sample points around the cell at different distances
    check_points = []
    for radius in [cell_size*0.8, cell_size*1.2, cell_size*1.6]:
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            dx = radius * np.cos(angle)
            dy = radius * np.sin(angle)
            check_points.append([x + dx, y + dy, z - cell_size/2])
    
    check_points = np.array(check_points).astype(np.float32)
    D_check, _ = gpu_index_obstacles.search(check_points, 1)
    distances = np.sqrt(D_check[:, 0])
    
    # If some points have support and others don't, it might be an edge
    has_support = distances < 0.2
    if np.any(has_support) and np.any(~has_support):
        return True
    return False

def check_cell_support(x, y, z, cell_size, gpu_index_obstacles):
    """Check if a cell has sufficient support underneath it"""
    # Generate a grid of points on the bottom face of the cell
    support_check_points = []
    check_resolution = cell_size / 4  # Check 5x5 grid on bottom face
    
    for dx in np.linspace(-cell_size/2, cell_size/2, 5):
        for dy in np.linspace(-cell_size/2, cell_size/2, 5):
            support_check_points.append([x + dx, y + dy, z - cell_size/2 - 0.05])  # Slightly below cell
    
    support_check_points = np.array(support_check_points).astype(np.float32)
    
    # Check distances to obstacles
    D_support, _ = gpu_index_obstacles.search(support_check_points, 1)
    distances = np.sqrt(D_support[:, 0])
    
    # Calculate support ratio - percentage of points that have support
    supported_points = np.sum(distances < 0.2)  # Points close to obstacle = supported
    support_ratio = supported_points / len(support_check_points)
    
    return support_ratio > 0.6  # Require at least 60% support

def improved_look_down_algorithm(obstacle_points, grid_min, grid_max, cell_size=0.1, height_intervals=3):
    """
    Improved 'look down' algorithm that can detect surfaces under overhangs.
    
    Args:
        obstacle_points: The obstacle point cloud
        grid_min: Minimum coordinates of the grid
        grid_max: Maximum coordinates of the grid
        cell_size: Size of each cubic cell (default: 0.1 = 10cm)
        height_intervals: Number of different heights to test from (default: 3)
    
    Returns:
        Array of points representing the centers of supported cells
    """
    print("Applying improved look down algorithm...")
    start_time = time.time()
    
    # Prepare grid for cell centers
    x_cells = np.arange(grid_min[0] + cell_size/2, grid_max[0], cell_size)
    y_cells = np.arange(grid_min[1] + cell_size/2, grid_max[1], cell_size)
    
    # Calculate height intervals
    z_range = grid_max[2] - grid_min[2]
    z_starts = np.linspace(grid_max[2], grid_min[2] + cell_size, height_intervals)
    
    # Keep track of placed cells
    placed_cells = []
    placed_cells_dict = {}  # Dictionary to track cells by xy position
    
    # Prepare FAISS index for fast collision detection
    res_placed = faiss.StandardGpuResources()
    placed_cells_gpu_index = None
    
    # Process cells in XY horizontal slices
    total_cells = len(x_cells) * len(y_cells) * height_intervals
    cells_processed = 0
    
    print(f"Processing {total_cells} potential cell positions...")
    
    # For each starting height
    for z_start in z_starts:
        for x_idx, x in enumerate(x_cells):
            for y_idx, y in enumerate(y_cells):
                # Display progress periodically
                cells_processed += 1
                if cells_processed % 5000 == 0:
                    print(f"Processing cell {cells_processed}/{total_cells} ({cells_processed/total_cells*100:.1f}%)")
                
                # Create a key for this xy position
                xy_key = f"{int(x*100)}_{int(y*100)}"
                
                # Check if we already have cells at this xy position at different heights
                if xy_key in placed_cells_dict:
                    # Get the heights of previously placed cells at this position
                    placed_heights = placed_cells_dict[xy_key]
                    # Skip if we already have too many cells at this xy position
                    if len(placed_heights) >= 5:  # Allow at most 2 cells in same xy position (ground level + one shelf)
                        continue
                
                # Start dropping the cell from the current height level
                z = z_start
                cell_center = np.array([x, y, z])
                
                # Start dropping the cell
                cell_placed = False
                edge_safety_margin = cell_size / 4  # Safety margin for edge detection
                while z >= grid_min[2] + cell_size/2 and not cell_placed:
                    # Define cell corners for collision check
                    half_size = cell_size / 2
                    safety_check_size = half_size + edge_safety_margin  # Safety margin for edge detection
                    
                    # Use additional points around the cell perimeter to check for edges
                    cell_check_points = np.array([
                        # Original corners
                        [x-half_size, y-half_size, z-half_size],
                        [x+half_size, y-half_size, z-half_size],
                        [x-half_size, y+half_size, z-half_size],
                        [x+half_size, y+half_size, z-half_size],
                        [x-half_size, y-half_size, z+half_size],
                        [x+half_size, y-half_size, z+half_size],
                        [x-half_size, y+half_size, z+half_size],
                        [x+half_size, y+half_size, z+half_size],
                        
                        # Additional edge detection points with safety margin
                        [x-safety_check_size, y, z-half_size],  # Left edge
                        [x+safety_check_size, y, z-half_size],  # Right edge
                        [x, y-safety_check_size, z-half_size],  # Front edge
                        [x, y+safety_check_size, z-half_size],  # Back edge
                        [x-safety_check_size, y-safety_check_size, z-half_size],  # Diagonals
                        [x+safety_check_size, y-safety_check_size, z-half_size],
                        [x-safety_check_size, y+safety_check_size, z-half_size],
                        [x+safety_check_size, y+safety_check_size, z-half_size],
                    ]).astype(np.float32)
                    
                    # Check if any of these points are too close to obstacles
                    D_check, _ = gpu_index_obstacles.search(cell_check_points, 1)
                    min_dist = np.min(np.sqrt(D_check))
                    
                    if min_dist < cell_size/2:  # Collision with obstacles
                        # Move up a bit to avoid intersection
                        z += cell_size/2
                        if is_potential_edge(x, y, z, cell_size, gpu_index_obstacles):
                            # This might be an edge cell, do stricter checking
                            if not check_cell_support(x, y, z, cell_size/2, gpu_index_obstacles):  # Use smaller cell size for edge check
                                # Skip this cell - it's likely on an edge
                                continue
                        # Check for clearance above for robot height
                        clearance_check = np.array([[x, y, z + 0.5]]).astype(np.float32)
                        D_clearance, _ = gpu_index_obstacles.search(clearance_check, 1)

                        # Add support check to prevent edge cells without sufficient support
                        if np.sqrt(D_clearance[0, 0]) > 0.5 and check_cell_support(x, y, z, cell_size, gpu_index_obstacles):
                            cell_center = np.array([x, y, z])
                            
                            # Check if we already have a cell at this xy position
                            if xy_key in placed_cells_dict:
                                # If there's already a cell at this position, only add if at a different height
                                existing_heights = placed_cells_dict[xy_key]
                                # Make sure the new cell is not too close to existing ones
                                too_close = False
                                for existing_z in existing_heights:
                                    if abs(existing_z - z) < cell_size*2:
                                        too_close = True
                                        break
                                
                                if not too_close:
                                    placed_cells.append(cell_center)
                                    placed_cells_dict.setdefault(xy_key, []).append(z)
                                    if placed_cells_gpu_index is not None:
                                        placed_cells_gpu_index.add(np.array([cell_center]).astype(np.float32))
                            else:
                                placed_cells.append(cell_center)
                                placed_cells_dict[xy_key] = [z]
                                # Initialize GPU index for placed cells if needed
                                if len(placed_cells) % 1000 == 1:
                                    if placed_cells_gpu_index is None and len(placed_cells) > 1:
                                        placed_cells_array = np.array(placed_cells).astype(np.float32)
                                        placed_cells_gpu_index = faiss.IndexFlatL2(3)
                                        placed_cells_gpu_index = faiss.index_cpu_to_gpu(res_placed, 0, placed_cells_gpu_index)
                                        placed_cells_gpu_index.add(placed_cells_array)
                                    elif placed_cells_gpu_index is not None:
                                        placed_cells_gpu_index.add(np.array([cell_center]).astype(np.float32))
                        
                        cell_placed = True
                    else:
                        # Drop the cell further down
                        z -= cell_size/4  # Use smaller steps for more precision
                
                # If we reached the bottom with no collision, place at ground level
                if not cell_placed:
                    if xy_key not in placed_cells_dict:  # Only place if we don't already have a cell here
                        cell_center = np.array([x, y, grid_min[2] + cell_size/2])
                        placed_cells.append(cell_center)
                        placed_cells_dict[xy_key] = [grid_min[2] + cell_size/2]
                        if placed_cells_gpu_index is not None:
                            placed_cells_gpu_index.add(np.array([cell_center]).astype(np.float32))
    
    # Convert to numpy array
    supported_cells = np.array(placed_cells) if placed_cells else np.empty((0, 3))
    
    # Log timing information
    elapsed_time = time.time() - start_time
    print(f"Improved look down algorithm completed in {elapsed_time:.2f} seconds")
    print(f"Generated {len(supported_cells)} supported cells")
    
    # Add connectivity information
    print("Analyzing connectivity between cells...")
    
    # Create a graph of connected cells to identify isolated cells
    if len(supported_cells) > 0:
        connect_supported_cells(supported_cells, cell_size)
    
    return supported_cells

def connect_supported_cells(supported_cells, cell_size):
    """
    Build a connectivity graph between supported cells and identify isolated regions
    """
    # Create FAISS index for supported cells
    supported_cells_f32 = supported_cells.astype(np.float32)
    res_support = faiss.StandardGpuResources()
    support_index = faiss.IndexFlatL2(3)
    gpu_support_index = faiss.index_cpu_to_gpu(res_support, 0, support_index)
    gpu_support_index.add(supported_cells_f32)
    
    # Find connections between nearby cells
    G = nx.Graph()
    
    # Add all cells as nodes
    for i in range(len(supported_cells)):
        G.add_node(i)
    
    # Connect nearby cells (within 1.5 * cell_size)
    connection_threshold = cell_size * 1.5
    
    # Batch processing for large cell arrays
    batch_size = 1000
    for i in range(0, len(supported_cells), batch_size):
        batch_end = min(i + batch_size, len(supported_cells))
        batch = supported_cells_f32[i:batch_end]
        
        # Search for nearby cells
        D, I = gpu_support_index.search(batch, 10)  # Find 10 nearest neighbors
        
        # Process results
        for j in range(len(batch)):
            cell_idx = i + j
            for k in range(10):
                neighbor_idx = I[j, k]
                if neighbor_idx != cell_idx and neighbor_idx < len(supported_cells):
                    dist = np.sqrt(D[j, k])
                    if dist <= connection_threshold:
                        # Cells are close enough to be connected
                        G.add_edge(cell_idx, neighbor_idx, weight=dist)
    
    # Find connected components
    components = list(nx.connected_components(G))
    print(f"Found {len(components)} connected regions")
    
    # Identify the largest component
    largest_component = max(components, key=len)
    print(f"Largest connected region has {len(largest_component)} cells")
    
    # Count isolated cells (not in the largest component)
    isolated_cells = sum(len(c) for c in components if c != largest_component)
    print(f"Found {isolated_cells} isolated cells in {len(components)-1} smaller regions")
    
    return G

# Apply the improved algorithm
cell_size = 0.5  # (can adjust based on your robot size)
supported_cells = improved_look_down_algorithm(obstacle_points, min_vals, max_vals, cell_size, height_intervals=5)

def filter_unsafe_cells(cells, gpu_index_obstacles, cell_size):
    """Remove cells that appear to be on edges or have insufficient support"""
    safe_cells = []
    
    for cell in cells:
        x, y, z = cell
        
        # Check for minimum support underneath
        if check_cell_support(x, y, z, cell_size, gpu_index_obstacles):
            # Check if not too close to edge
            if not is_potential_edge(x, y, z, cell_size, gpu_index_obstacles):
                safe_cells.append(cell)
    
    return np.array(safe_cells)

# Use this after generating all cells
supported_cells = filter_unsafe_cells(supported_cells, gpu_index_obstacles, cell_size)

# Visualize the supported cells
def visualize_supported_cells_3d():
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot obstacles (downsampled)
    max_vis_obstacles = min(5000, len(obstacle_points))
    if len(obstacle_points) > max_vis_obstacles:
        indices = np.random.choice(len(obstacle_points), max_vis_obstacles, replace=False)
        vis_obstacles = obstacle_points[indices]
    else:
        vis_obstacles = obstacle_points
        
    ax.scatter(vis_obstacles[:, 0], vis_obstacles[:, 1], vis_obstacles[:, 2], 
              c='gray', alpha=0.2, s=1, label='Obstacles')
    
    # Plot supported cells
    if len(supported_cells) > 0:
        # Color by height
        norm_heights = (supported_cells[:, 2] - min_vals[2]) / (max_vals[2] - min_vals[2])
        sc = ax.scatter(supported_cells[:, 0], supported_cells[:, 1], supported_cells[:, 2],
                  c=norm_heights, cmap='viridis', alpha=0.8, s=30, label='Supported Cells')
        plt.colorbar(sc, ax=ax, label='Height')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Supported Cells from Improved Look Down Algorithm')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([max_vals[0]-min_vals[0], max_vals[1]-min_vals[1], max_vals[2]-min_vals[2]]).max() / 2.0
    mid_x = (max_vals[0]+min_vals[0]) * 0.5
    mid_y = (max_vals[1]+min_vals[1]) * 0.5
    mid_z = (max_vals[2]+min_vals[2]) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()

# Visualize the results
visualize_supported_cells_3d()

# Downsample if needed (too many points will make graph creation slow)
# if len(empty_space_points) > 3000:
#     indices = np.random.choice(len(empty_space_points), 3000, replace=False)
#     empty_space_points = empty_space_points[indices]
empty_space_creation_time = time.time() - start_time

empty_space_points = np.array(supported_cells)  # Use the supported cells as empty space points

def visualize_empty_space_points():
    fig_empty = plt.figure(figsize=(8,6))
    ax_empty = fig_empty.add_subplot(111, projection='3d')
    ax_empty.scatter(empty_space_points[:, 0], empty_space_points[:, 1], empty_space_points[:, 2],
                     c='cyan', s=5, alpha=0.7)
    ax_empty.set_xlabel("X")
    ax_empty.set_ylabel("Y")
    ax_empty.set_zlabel("Z")
    ax_empty.set_title("Empty Space Points")
    plt.show()

# Call the new function to visualize empty space
visualize_empty_space_points()

# Timing: KDTree creation for empty space points
start_time = time.time()
# tree = KDTree(empty_space_points)
empty_space_points_f32 = empty_space_points.astype(np.float32)
res_empty = faiss.StandardGpuResources()
index_flat_empty = faiss.IndexFlatL2(3)
gpu_index_empty = faiss.index_cpu_to_gpu(res_empty, 0, index_flat_empty)
gpu_index_empty.add(empty_space_points_f32)
kdtree_time = time.time() - start_time

# Timing: Graph creation of navigable empty space
start_time = time.time()
G = nx.Graph()
# k_neighbors = 30  # Connect each point to its k nearest neighbors
# for i, p in enumerate(empty_space_points):
#     neighbors = tree.query(p, k=k_neighbors+1)[1][1:]  # Skip self
#     for n in neighbors:
#         # Check if line between points intersects obstacles
#         line_points = 10  # Number of check points along the line
#         t = np.linspace(0, 1, line_points).reshape(-1, 1)
#         check_points = p.reshape(1, 3) * (1-t) + empty_space_points[n].reshape(1, 3) * t
        
#         # Query distances to nearest obstacles
#         check_distances, _ = tree_obstacles.query(check_points)
#         if np.all(check_distances > min_obstacle_distance):
#             # Path is clear, add edge
#             G.add_edge(i, n, weight=np.linalg.norm(empty_space_points[i] - empty_space_points[n]))
start_time = time.time()
G = nx.Graph()
k_neighbors = 30
D_es, I_es = gpu_index_empty.search(empty_space_points_f32, k_neighbors + 1)
for i in range(len(empty_space_points)):
    p = empty_space_points[i]
    neighbors = I_es[i, 1:]  # skip self
    for n in neighbors:
        # Only connect points at similar heights (prevent flying)
        height_diff = abs(p[2] - empty_space_points[n][2])
        if height_diff > cell_size * 0.5:  # Allow only small vertical changes
            continue  # Skip this connection
            
        # Check horizontal distance (optional additional constraint)
        horizontal_dist = np.sqrt((p[0] - empty_space_points[n][0])**2 + 
                                 (p[1] - empty_space_points[n][1])**2)
        if horizontal_dist > cell_size * 2.0:  # Too far horizontally
            continue
            
        # Check small line segments for obstacle clearance
        line_points = 10
        t = np.linspace(0, 1, line_points).reshape(-1, 1)
        check_points = p.reshape(1, 3)*(1-t) + empty_space_points[n].reshape(1, 3)*t
        check_points_f32 = check_points.astype(np.float32)

        D_chk, I_chk = gpu_index_obstacles.search(check_points_f32, 1)
        if np.all(np.sqrt(D_chk[:, 0]) > min_obstacle_distance):
            G.add_edge(i, n, weight=np.linalg.norm(p - empty_space_points[n]))

graph_creation_time = time.time() - start_time

# Initialize start and goal
start_point = np.array([0.1, 0.2, 0.3])
goal_point = np.array([0.8, 0.8, 0.8])

# Find nearest empty space points
# start_idx = tree.query(start_point)[1]
# goal_idx = tree.query(goal_point)[1]
start_idx = gpu_index_empty.search(start_point.reshape(1, -1).astype(np.float32), 1)[1][0][0]
goal_idx = gpu_index_empty.search(goal_point.reshape(1, -1).astype(np.float32), 1)[1][0][0]

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
    s_start_x = Slider(ax_start_x, 'Start X', min_vals[0], max_vals[0], valinit=start_point[0])
    s_start_y = Slider(ax_start_y, 'Start Y', min_vals[1], max_vals[1], valinit=start_point[1])
    s_start_z = Slider(ax_start_z, 'Start Z', min_vals[2], max_vals[2], valinit=start_point[2])
    s_goal_x = Slider(ax_goal_x, 'Goal X', min_vals[0], max_vals[0], valinit=goal_point[0])
    s_goal_y = Slider(ax_goal_y, 'Goal Y', min_vals[1], max_vals[1], valinit=goal_point[1])
    s_goal_z = Slider(ax_goal_z, 'Goal Z', min_vals[2], max_vals[2], valinit=goal_point[2])
    
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
    # start_idx = tree.query(start_point)[1]
    # goal_idx = tree.query(goal_point)[1]

    start_idx = gpu_index_empty.search(start_point.reshape(1, -1).astype(np.float32), 1)[1][0][0]
    goal_idx = gpu_index_empty.search(goal_point.reshape(1, -1).astype(np.float32), 1)[1][0][0]
    
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