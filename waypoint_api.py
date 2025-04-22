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
faiss_res = faiss.StandardGpuResources()
print("Using CuPy and FAISS for GPU acceleration")
gpu_accelerated = True
import cupy as cp
def cpu_raycasting_fallback(obstacle_points, grid_min, grid_max, cell_size):
    print("Warning: CPU fallback not fully implemented.")
    return np.empty((0, 3))

def check_clearance_faiss_gpu(potential_hits, obstacle_points_f32, min_clearance, cell_size, gpu_index_obstacles):
    """
    Checks clearance above potential hit points using FAISS GPU index with k=1 NN search.
    Ensures no obstacle point is within min_clearance distance vertically above the hit point.
    """
    if len(potential_hits) == 0:
        return np.array([], dtype=bool)

    # Define clearance check points slightly ABOVE the required clearance zone
    # If this point is clear, the space below it down to the hit point should be clear.
    check_points = potential_hits.copy()
    # Check at the very top of the required clearance volume
    check_points[:, 2] += min_clearance
    check_points_f32 = check_points.astype(np.float32)

    # Find the single nearest neighbor to the check point
    k = 1
    # D contains SQUARE distances
    D, I = gpu_index_obstacles.search(check_points_f32, k)

    # The clearance is okay if the distance to the NEAREST obstacle point
    # from the check_point (at height z + min_clearance) is greater than min_clearance.
    # This implies the sphere of radius min_clearance centered at the check point is empty.
    # We compare squared distances for efficiency.
    min_clearance_sq = min_clearance**2

    # D has shape (N, 1), so we access the first column D[:, 0]
    clearance_ok = D[:, 0] > min_clearance_sq

    # Alternative stricter check (closer to original intent?):
    # Ensure nothing is within half the clearance distance of a point halfway up the clearance zone.
    # check_points = potential_hits.copy()
    # check_points[:, 2] += min_clearance / 2.0
    # check_points_f32 = check_points.astype(np.float32)
    # k=1
    # D, I = gpu_index_obstacles.search(check_points_f32, k)
    # clearance_radius_sq = (min_clearance / 2.0)**2
    # clearance_ok = D[:, 0] > clearance_radius_sq

    return clearance_ok


def improved_look_down_algorithm_multi_start(
    obstacle_points, grid_min, grid_max,
    grid_cell_size=0.1,               # Grid resolution for ray casting
    point_influence_radius=None,      # Radius for point detection (default: 0.4 * grid_cell_size)
    num_z_levels=5,                   # Number of ray starting heights
    min_clearance=0.5,                # Minimum vertical clearance required
    vertical_duplicate_threshold=None, # Distance for merging cells vertically (default: grid_cell_size)
    batch_size=10000                  # GPU batch processing size
):
    """
    Ray casting algorithm with multiple start heights to find surfaces under overhangs.

    Args:
        obstacle_points: The obstacle point cloud (N x 3 numpy array)
        grid_min: Minimum coordinates of the grid (3-element array/list)
        grid_max: Maximum coordinates of the grid (3-element array/list)
        grid_cell_size: Size of each grid cell for ray casting (default: 0.1)
        point_influence_radius: Radius of influence for obstacle points (default: 0.4 * grid_cell_size)
        num_z_levels: Number of different Z heights to start rays from (default: 5)
        min_clearance: Minimum vertical clearance required above a supported cell (default: 0.5)
        vertical_duplicate_threshold: Distance threshold for merging vertical duplicates (default: grid_cell_size)
        batch_size: Number of rays to process per GPU batch (default: 10000)

    Returns:
        Array of points representing the centers of supported cells
    """
    if not gpu_accelerated:
        print("GPU acceleration not available. Aborting.")
        # Or call a CPU fallback if implemented
        return cpu_raycasting_fallback(obstacle_points, grid_min, grid_max, grid_cell_size)

    # Set default values for parameters if not provided
    if point_influence_radius is None:
        point_influence_radius = 0.4 * grid_cell_size
        
    if vertical_duplicate_threshold is None:
        vertical_duplicate_threshold = grid_cell_size

    print(f"Applying multi-start GPU ray casting (Z levels={num_z_levels})...")
    start_time = time.time()

    # --- 1. Prepare Grid and Ray Origins ---
    x_cells = np.arange(grid_min[0] + grid_cell_size / 2, grid_max[0], grid_cell_size)
    y_cells = np.arange(grid_min[1] + grid_cell_size / 2, grid_max[1], grid_cell_size)
    # Define multiple Z starting heights
    z_starts = np.linspace(grid_max[2] + grid_cell_size, grid_min[2] + min_clearance, num_z_levels) # Ensure lowest start is above min clearance needs

    xx, yy, zz = np.meshgrid(x_cells, y_cells, z_starts)
    ray_origins_x = xx.flatten()
    ray_origins_y = yy.flatten()
    ray_origins_z = zz.flatten()

    num_rays = len(ray_origins_x)
    print(f"Total rays to cast: {num_rays} ({len(x_cells)}x{len(y_cells)}x{num_z_levels})")

    ray_origins = np.column_stack([ray_origins_x, ray_origins_y, ray_origins_z]).astype(np.float32)
    ray_directions = np.zeros((num_rays, 3), dtype=np.float32)
    ray_directions[:, 2] = -1.0 # Pointing down

    # --- 2. Prepare GPU Data ---
    cp_ray_origins = cp.asarray(ray_origins)
    cp_ray_directions = cp.asarray(ray_directions)
    cp_obstacle_points = cp.asarray(obstacle_points.astype(np.float32))
    obstacle_points_f32 = obstacle_points.astype(np.float32) # Keep a CPU copy for FAISS index building

    # Build FAISS index for obstacles (used for clearance check later)
    print("Building FAISS GPU index for obstacles...")
    index_obstacles = faiss.IndexFlatL2(3) # Using L2 distance
    gpu_index_obstacles = faiss.index_cpu_to_gpu(faiss_res, 0, index_obstacles)
    gpu_index_obstacles.add(obstacle_points_f32)
    print(f"FAISS index built with {gpu_index_obstacles.ntotal} points.")

    # --- 3. Define Simplified Ray Intersection Kernel ---
    # Calculate radius squared once - using the configurable point_influence_radius
    point_radius_sq = point_influence_radius * point_influence_radius
    
    # Kernel now accepts a separate point_radius_sq parameter
    simple_ray_intersection_kernel = cp.ElementwiseKernel(
        'float32 ox, float32 oy, float32 oz, float32 dx, float32 dy, float32 dz, ' +
        'float32 grid_min_z, float32 cell_sz, raw float32 obstacles, float32 point_radius_sq',
        'float32 hit_x, float32 hit_y, float32 hit_z, int32 is_hit',
        '''
        const float ray_epsilon = 0.0001f;
        const float max_dist = oz - grid_min_z + cell_sz; // Max ray distance down to below grid floor

        is_hit = 0;
        hit_x = ox; // Default to origin X
        hit_y = oy; // Default to origin Y
        hit_z = 0.0f; // Default Z
        float closest_t = max_dist;

        int num_obstacles = obstacles.size() / 3;

        for (int i = 0; i < num_obstacles; i++) {
            float px = obstacles[i*3 + 0];
            float py = obstacles[i*3 + 1];
            float pz = obstacles[i*3 + 2];

            // Simplified check: Distance from point to ray origin projected onto ray path
            // This is a faster approximation than sphere intersection for dense points
            float oc_x = px - ox;
            float oc_y = py - oy;
            float oc_z = pz - oz;

            float proj_t = oc_x * dx + oc_y * dy + oc_z * dz; // Ray direction is normalized (length 1 in Z)

            // Check if point is generally 'in front' of the ray origin along its path
            if (proj_t > ray_epsilon) {
                // Calculate closest point on ray to obstacle point p
                float closest_pt_x = ox + proj_t * dx;
                float closest_pt_y = oy + proj_t * dy;
                float closest_pt_z = oz + proj_t * dz;

                // Calculate squared distance from obstacle point to the ray line
                float dist_sq = (px - closest_pt_x)*(px - closest_pt_x) +
                                (py - closest_pt_y)*(py - closest_pt_y) +
                                (pz - closest_pt_z)*(pz - closest_pt_z);

                if (dist_sq < point_radius_sq) {
                    // Ray passes close enough to the point
                    // Use the projection distance 'proj_t' as the hit distance 't'
                    if (proj_t < closest_t) {
                         closest_t = proj_t;
                         is_hit = 1;
                         // We only store 't', hit point calculated later if needed
                    }
                }
            }
        } // End loop through obstacles

        // If hit, calculate precise hit point
        if (is_hit == 1) {
             hit_x = ox + closest_t * dx;
             hit_y = oy + closest_t * dy;
             hit_z = oz + closest_t * dz;

             // Ensure hit is not below the grid floor (can happen with approximations)
             if (hit_z < grid_min_z) {
                 is_hit = 0; // Invalidate hit if it ends up below floor
             }
        } else {
            // Option 1: If no obstacle hit, the ray reaches the ground implicitly.
            // We don't mark this as a hit here, handle ground plane later if needed.
            // Option 2: Mark ground as hit (less flexible if ground isn't flat)
            // hit_x = ox; hit_y = oy; hit_z = grid_min_z; is_hit = 1; // If desired
             is_hit = 0; // Prefer finding actual obstacle hits
        }
        ''',
        'simple_ray_intersection_kernel'
    )

    # --- 4. Execute Kernel in Batches ---
    print("Running ray intersection kernel...")
    all_potential_hits = []
    all_is_hit = []

    for i in range(0, num_rays, batch_size):
        end_idx = min(i + batch_size, num_rays)
        batch_origins = cp_ray_origins[i:end_idx]
        batch_directions = cp_ray_directions[i:end_idx]
        batch_size_actual = end_idx - i

        # Prepare output arrays for the batch
        hit_x = cp.zeros(batch_size_actual, dtype=cp.float32)
        hit_y = cp.zeros(batch_size_actual, dtype=cp.float32)
        hit_z = cp.zeros(batch_size_actual, dtype=cp.float32)
        is_hit = cp.zeros(batch_size_actual, dtype=cp.int32)

        # Execute kernel with the separate point_radius_sq parameter
        simple_ray_intersection_kernel(
            batch_origins[:, 0], batch_origins[:, 1], batch_origins[:, 2],
            batch_directions[:, 0], batch_directions[:, 1], batch_directions[:, 2],
            grid_min[2], grid_cell_size, cp_obstacle_points, point_radius_sq,
            hit_x, hit_y, hit_z, is_hit
        )

        # Get results back to CPU
        batch_hits_xyz = cp.asnumpy(cp.stack([hit_x, hit_y, hit_z], axis=-1))
        batch_is_hit = cp.asnumpy(is_hit)

        # Store results
        all_potential_hits.append(batch_hits_xyz)
        all_is_hit.append(batch_is_hit)

        # Optional: print progress
        # print(f"Processed batch {i // batch_size + 1}/{(num_rays + batch_size - 1) // batch_size}")

    # Concatenate results from all batches
    potential_hits_np = np.concatenate(all_potential_hits, axis=0)
    is_hit_np = np.concatenate(all_is_hit, axis=0)

    # Filter out rays that didn't hit anything
    valid_hit_indices = np.where(is_hit_np == 1)[0]
    actual_hits = potential_hits_np[valid_hit_indices]
    print(f"Found {len(actual_hits)} potential hit points.")

    # --- 5. Check Clearance using FAISS ---
    print(f"Checking clearance ({min_clearance}m) for potential hits...")
    if len(actual_hits) > 0:
        clearance_ok = check_clearance_faiss_gpu(
            actual_hits, obstacle_points_f32, min_clearance, grid_cell_size, gpu_index_obstacles
        )
        supported_hits = actual_hits[clearance_ok]
        print(f"Found {len(supported_hits)} hits with sufficient clearance.")
    else:
        supported_hits = np.empty((0, 3))
        print("No potential hits found, skipping clearance check.")

    # --- 6. Post-process: Consolidate and Remove Duplicates ---
    # Now using the vertical_duplicate_threshold parameter instead of cell_size
    supported_cells = []
    if len(supported_hits) > 0:
        print("Consolidating results and removing close duplicates...")
        placed_cells_dict = {}
        # Sort by Z descending, so we prioritize higher surfaces if they are too close
        sorted_indices = np.argsort(supported_hits[:, 2])[::-1]
        sorted_supported_hits = supported_hits[sorted_indices]

        # Use grid_cell_size for XY grid but vertical_duplicate_threshold for Z comparison
        for cell in sorted_supported_hits:
            x, y, z = cell
            # Replace integer division with proper floor division
            x_cell = int(np.floor(x / grid_cell_size))
            y_cell = int(np.floor(y / grid_cell_size))
            xy_key = f"{x_cell}_{y_cell}"

            if xy_key not in placed_cells_dict:
                placed_cells_dict[xy_key] = [z]
                supported_cells.append(cell)
            else:
                # Check if this hit is too close to existing hits in this column
                # Using vertical_duplicate_threshold for the Z comparison
                is_too_close = False
                for existing_z in placed_cells_dict[xy_key]:
                    if abs(existing_z - z) < vertical_duplicate_threshold:
                        is_too_close = True
                        break
                if not is_too_close:
                    placed_cells_dict[xy_key].append(z)
                    supported_cells.append(cell)
                    # Optional limit on number of surfaces per column:
                    # if len(placed_cells_dict[xy_key]) >= max_surfaces_per_column:
                    #     break # Or handle differently

        supported_cells = np.array(supported_cells) if supported_cells else np.empty((0, 3))

    elapsed_time = time.time() - start_time
    print(f"Multi-start ray casting completed in {elapsed_time:.2f} seconds")
    print(f"Generated {len(supported_cells)} final supported cells")

    # --- 7. Connectivity Analysis (Optional but Recommended) ---
    if len(supported_cells) > 0:
        print("Analyzing connectivity...")
        # Use a separate connectivity parameter set instead of reusing cell_size
        G = connect_supported_cells(
            supported_cells, 
            connection_radius=grid_cell_size * 1.5,  # Default connection radius 
            max_vertical_step=grid_cell_size * 1.1,  # Default vertical step constraint
            faiss_res=faiss_res
        )
    else:
        G = None # Or nx.Graph()

    # Cleanup FAISS GPU resources if they are not needed elsewhere
    # del gpu_index_obstacles # Or manage resource lifetime appropriately

    return supported_cells, G # Return graph too

def connect_supported_cells(supported_cells, connection_radius=0.15, max_vertical_step=0.11, k_neighbors=30, faiss_res=None):
    """
    Build a connectivity graph between supported cells using FAISS GPU.
    Identifies connected components and isolated regions.
    Args:
        supported_cells (np.array): Nx3 array of cell centers.
        connection_radius (float): Maximum distance to consider cells connected (default: 0.15)
        max_vertical_step (float): Maximum allowed vertical distance between connected cells (default: 0.11)
        k_neighbors (int): Number of nearest neighbors to search for each cell (default: 30)
        faiss_res (faiss.GpuResources, optional): Pre-initialized FAISS GPU resources. Defaults to None.
    Returns:
        networkx.Graph: Graph where nodes are cell indices and edges connect nearby cells.
    """
    if len(supported_cells) < 2:
        print("Not enough cells to build connectivity graph.")
        return nx.Graph()
        
    if faiss_res is None:
         # Initialize resources if not provided (less efficient if called multiple times)
         print("Initializing temporary FAISS GPU resources for connectivity.")
         faiss_res = faiss.StandardGpuResources()

    print("Building FAISS index for supported cells...")
    supported_cells_f32 = supported_cells.astype(np.float32)
    support_index = faiss.IndexFlatL2(3)
    gpu_support_index = faiss.index_cpu_to_gpu(faiss_res, 0, support_index)
    gpu_support_index.add(supported_cells_f32)
    G = nx.Graph()
    G.add_nodes_from(range(len(supported_cells)))

    # Square the connection radius for more efficient distance comparisons
    connection_threshold_sq = connection_radius**2

    print(f"Finding connections within radius {connection_radius:.3f}...")
    # Batch processing for potentially large number of supported cells
    batch_size = 5000
    num_cells = len(supported_cells)

    for i in range(0, num_cells, batch_size):
        batch_end = min(i + batch_size, num_cells)
        # Query points directly on GPU
        batch_query_cp = cp.asarray(supported_cells_f32[i:batch_end])
        batch_query_np = batch_query_cp.get()
        # Use knn search and filter by distance
        D, I = gpu_support_index.search(batch_query_np, k_neighbors)

        # Process results (can potentially stay on GPU longer if needed)
        D_cpu = cp.asnumpy(D)
        I_cpu = cp.asnumpy(I)

        for j in range(len(batch_query_cp)): # Loop through query points in the batch
            cell_idx = i + j
            for k in range(1, k_neighbors): # Start from 1 to skip self-comparison
                neighbor_idx = I_cpu[j, k]
                dist_sq = D_cpu[j, k]

                # Check if neighbor is valid and within threshold
                if neighbor_idx >= 0 and neighbor_idx < num_cells and neighbor_idx != cell_idx:
                    if dist_sq <= connection_threshold_sq:
                        # Check vertical distance constraint using the max_vertical_step parameter
                        z_diff = abs(supported_cells[cell_idx, 2] - supported_cells[neighbor_idx, 2])
                        if z_diff < max_vertical_step:
                            G.add_edge(cell_idx, neighbor_idx, weight=dist_sq**0.5)
                    else:
                        # Since neighbors are sorted by distance, we can stop early for this query point
                        break
                        
        # print(f"Processed connectivity batch {i // batch_size + 1}/{(num_cells + batch_size - 1) // batch_size}")

    # Find connected components
    components = list(nx.connected_components(G))
    print(f"Found {len(components)} connected regions.")

    if components:
        largest_component = max(components, key=len)
        print(f"Largest connected region has {len(largest_component)} cells.")
        isolated_cells = num_cells - len(largest_component)
        num_isolated_regions = len(components) - 1
        if num_isolated_regions > 0:
             print(f"Found {isolated_cells} isolated cells in {num_isolated_regions} smaller regions.")
    else:
        print("Graph has no components.")
        
    # Cleanup index (optional, depends on resource management)
    # del gpu_support_index

    return G

# Apply the improved algorithm
cell_size = 0.2  # (can adjust based on your robot size)
supported_cells, G = improved_look_down_algorithm_multi_start(
    obstacle_points, grid_min=min_vals, grid_max=max_vals,
    grid_cell_size=0.1,               # Grid resolution for ray casting
    point_influence_radius=0.2,      # Radius for point detection (default: 0.4 * grid_cell_size) # can influence ramp connection
    num_z_levels=5,                   # Number of ray starting heights
    min_clearance=0.5,                # Minimum vertical clearance required
    vertical_duplicate_threshold=0.25, # Distance for merging cells vertically (default: grid_cell_size)
    batch_size=10000                  # GPU batch processing size
)

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

k_neighbors = 6
D_es, I_es = gpu_index_empty.search(empty_space_points_f32, k_neighbors + 1)
batch_size = 1000
num_points = len(empty_space_points)
for batch_start in range(0, num_points, batch_size):
    batch_end = min(batch_start + batch_size, num_points)
    batch_indices = range(batch_start, batch_end)
    
    # Create all potential connections at once
    connections = []
    for i in batch_indices:
        p = empty_space_points[i]
        neighbors = I_es[i, 1:k_neighbors+1]  # k nearest neighbors
        
        for n in neighbors:
            # Quick checks for height and horizontal distance
            height_diff = abs(p[2] - empty_space_points[n][2])
            if height_diff > 0.25:  # 0.5 * 0.5
                continue
                
            horizontal_dist = np.sqrt((p[0] - empty_space_points[n][0])**2 + 
                                     (p[1] - empty_space_points[n][1])**2)
            if horizontal_dist > 0.25:  # 4 * 2.0
                continue
                
            connections.append((i, n))
    
    # Batch process all collision checks at once
    if connections:
        all_check_points = []
        for i, n in connections:
            line_points = np.linspace(0, 1, 10).reshape(-1, 1)
            points = empty_space_points[i].reshape(1, 3) * (1-line_points) + \
                    empty_space_points[n].reshape(1, 3) * line_points
            all_check_points.append(points)
        
        # Single FAISS query for all connections
        all_points = np.vstack(all_check_points).astype(np.float32)
        D_all, _ = gpu_index_obstacles.search(all_points, 1)
        
        # Process results in batches of 10 points per connection
        point_idx = 0
        for idx, (i, n) in enumerate(connections):
            collision_free = True
            for _ in range(10):  # 10 points per line
                if np.sqrt(D_all[point_idx][0]) <= min_obstacle_distance:
                    collision_free = False
                    break
                point_idx += 1
            
            if collision_free:
                G.add_edge(i, n, weight=np.linalg.norm(empty_space_points[i] - empty_space_points[n]))

graph_creation_time = time.time() - start_time

# Initialize start and goal
start_point = np.array([0.1, 0.2, 0.3])
goal_point = np.array([0.8, 0.8, 0.8])

# Find nearest empty space points
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

# Initialize path finding
path_found = find_path()

# Open3D visualization instead of matplotlib
import open3d as o3d
import tkinter as tk
from tkinter import ttk

class Open3DVisualizer:
    def __init__(self):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("3D Navigation Visualization", width=1280, height=720)
        
        # Configure rendering options
        render_options = self.vis.get_render_option()
        render_options.background_color = np.array([0.1, 0.1, 0.1])
        render_options.point_size = 3.0
        render_options.line_width = 2.0
        
        # Create UI window for sliders
        self.ui_window = tk.Tk()
        self.ui_window.title("Navigation Controls")
        self.ui_window.geometry("600x350")
        
        # Create objects dictionary to manage visualization objects
        self.objects = {}
        
        # Initialize the geometric objects
        self.init_geometries()
        
        # Setup the UI controls
        self.setup_ui()
    
    def init_geometries(self):
        # Point cloud for obstacles
        self.objects["obstacles"] = o3d.geometry.PointCloud()
        self.objects["obstacles"].points = o3d.utility.Vector3dVector(obstacle_points_vis)
        # Use height-based coloring
        colors = np.zeros((len(obstacle_points_vis), 3))
        normalized_height = (obstacle_points_vis[:, 2] - min_vals[2]) / (max_vals[2] - min_vals[2])
        viridis = plt.cm.viridis
        for i, h in enumerate(normalized_height):
            colors[i] = viridis(h)[:3]  # Get RGB components only
        self.objects["obstacles"].colors = o3d.utility.Vector3dVector(colors)
        self.vis.add_geometry(self.objects["obstacles"])
        
        # Empty space points
        self.objects["empty_space"] = o3d.geometry.PointCloud()
        self.objects["empty_space"].points = o3d.utility.Vector3dVector(empty_space_points)
        self.objects["empty_space"].paint_uniform_color([0.4, 0.4, 0.9])  # Light blue
        self.vis.add_geometry(self.objects["empty_space"])
        
        # Start point sphere
        self.objects["start"] = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        self.objects["start"].translate(start_point)
        self.objects["start"].paint_uniform_color([0, 1, 0])  # Green
        self.objects["start"].compute_vertex_normals()
        self.vis.add_geometry(self.objects["start"])
        
        # Goal point sphere
        self.objects["goal"] = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        self.objects["goal"].translate(goal_point)
        self.objects["goal"].paint_uniform_color([0, 0, 1])  # Blue
        self.objects["goal"].compute_vertex_normals()
        self.vis.add_geometry(self.objects["goal"])
        
        # Start nearest point sphere
        self.objects["start_nearest"] = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
        self.objects["start_nearest"].translate(empty_space_points[start_idx])
        self.objects["start_nearest"].paint_uniform_color([0.5, 1, 0.5])  # Light green
        self.objects["start_nearest"].compute_vertex_normals()
        self.vis.add_geometry(self.objects["start_nearest"])
        
        # Goal nearest point sphere
        self.objects["goal_nearest"] = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
        self.objects["goal_nearest"].translate(empty_space_points[goal_idx])
        self.objects["goal_nearest"].paint_uniform_color([0.5, 0.5, 1])  # Light blue
        self.objects["goal_nearest"].compute_vertex_normals()
        self.vis.add_geometry(self.objects["goal_nearest"])
        
        # Path line set (initially empty)
        self.objects["path"] = o3d.geometry.LineSet()
        if path_found and len(waypoints) > 1:
            self.update_path_visualization()
        self.vis.add_geometry(self.objects["path"])
        
        # Bounding box for context
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_vals, max_bound=max_vals)
        self.objects["bbox"] = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
        self.objects["bbox"].paint_uniform_color([0.7, 0.7, 0.7])
        self.vis.add_geometry(self.objects["bbox"])
    
    def update_path_visualization(self):
        if not path_found or len(waypoints) <= 1:
            self.objects["path"].points = o3d.utility.Vector3dVector([])
            self.objects["path"].lines = o3d.utility.Vector2iVector([])
            self.objects["path"].colors = o3d.utility.Vector3dVector([])
            return
            
        # Create line set for the path
        points = o3d.utility.Vector3dVector(waypoints)
        lines = []
        for i in range(len(waypoints) - 1):
            lines.append([i, i + 1])
        line_colors = [[1, 0, 0] for _ in range(len(lines))]  # Red lines
        
        self.objects["path"].points = points
        self.objects["path"].lines = o3d.utility.Vector2iVector(lines)
        self.objects["path"].colors = o3d.utility.Vector3dVector(line_colors)
    
    def setup_ui(self):
        # Create frames for sliders
        start_frame = ttk.LabelFrame(self.ui_window, text="Start Position")
        start_frame.pack(fill="x", padx=10, pady=5)
        
        goal_frame = ttk.LabelFrame(self.ui_window, text="Goal Position")
        goal_frame.pack(fill="x", padx=10, pady=5)
        
        button_frame = ttk.Frame(self.ui_window)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        # Status frame
        status_frame = ttk.LabelFrame(self.ui_window, text="Status")
        status_frame.pack(fill="x", padx=10, pady=5)
        
        # Create sliders for start position
        self.start_x = self.create_slider(start_frame, "X:", min_vals[0], max_vals[0], start_point[0])
        self.start_y = self.create_slider(start_frame, "Y:", min_vals[1], max_vals[1], start_point[1])
        self.start_z = self.create_slider(start_frame, "Z:", min_vals[2], max_vals[2], start_point[2])
        
        # Create sliders for goal position
        self.goal_x = self.create_slider(goal_frame, "X:", min_vals[0], max_vals[0], goal_point[0])
        self.goal_y = self.create_slider(goal_frame, "Y:", min_vals[1], max_vals[1], goal_point[1])
        self.goal_z = self.create_slider(goal_frame, "Z:", min_vals[2], max_vals[2], goal_point[2])
        
        # Create update button
        self.update_button = ttk.Button(button_frame, text="Update Path", command=self.update_positions)
        self.update_button.pack(side=tk.LEFT, padx=5)
        
        # Create graph visualization button
        self.graph_button = ttk.Button(button_frame, text="Show Graph", command=self.show_graph_view)
        self.graph_button.pack(side=tk.LEFT, padx=5)
        
        # Create reset view button
        self.reset_button = ttk.Button(button_frame, text="Reset View", command=self.reset_view)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        # Create hide/show obstacle points button
        self.toggle_obstacles_button = ttk.Button(button_frame, text="Toggle Obstacles", command=self.toggle_obstacles)
        self.toggle_obstacles_button.pack(side=tk.LEFT, padx=5)
        
        # Create status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack(fill="x", padx=5, pady=5)
        
        # Show initial status
        self.update_status()
    
    def create_slider(self, parent, label_text, min_val, max_val, initial_val):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=5, pady=2)
        
        label = ttk.Label(frame, text=label_text, width=3)
        label.pack(side=tk.LEFT)
        
        slider = ttk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL)
        slider.set(initial_val)
        slider.pack(side=tk.LEFT, fill="x", expand=True)
        
        value_var = tk.StringVar(value=f"{initial_val:.2f}")
        value_label = ttk.Label(frame, textvariable=value_var, width=6)
        value_label.pack(side=tk.LEFT)
        
        # Update value label when slider changes
        def on_change(event):
            value_var.set(f"{slider.get():.2f}")
        
        slider.bind("<Motion>", on_change)
        slider.bind("<ButtonRelease-1>", on_change)
        
        return slider
    
    def update_positions(self):
        global start_point, goal_point, start_idx, goal_idx, path_found
        
        # Get current values from sliders
        start_point = np.array([
            self.start_x.get(),
            self.start_y.get(),
            self.start_z.get()
        ])
        
        goal_point = np.array([
            self.goal_x.get(),
            self.goal_y.get(),
            self.goal_z.get()
        ])
        
        # Update sphere positions
        self.objects["start"].translate(start_point - np.asarray(self.objects["start"].get_center()))
        self.objects["goal"].translate(goal_point - np.asarray(self.objects["goal"].get_center()))
        
        # Find nearest navigable points
        start_idx = gpu_index_empty.search(start_point.reshape(1, -1).astype(np.float32), 1)[1][0][0]
        goal_idx = gpu_index_empty.search(goal_point.reshape(1, -1).astype(np.float32), 1)[1][0][0]
        
        # Update nearest points
        self.objects["start_nearest"].translate(
            empty_space_points[start_idx] - np.asarray(self.objects["start_nearest"].get_center())
        )
        self.objects["goal_nearest"].translate(
            empty_space_points[goal_idx] - np.asarray(self.objects["goal_nearest"].get_center())
        )
        
        # Recompute path
        path_found = find_path()
        
        # Update path visualization
        self.update_path_visualization()
        
        # Update status
        self.update_status()
        
        # Update all geometries in the visualizer
        self.update_geometries()
    
    def update_status(self):
        if path_found:
            self.status_var.set(f"Found path with {len(waypoints)} waypoints in {path_calculation_time:.4f} seconds")
        else:
            self.status_var.set(f"No path found between start and goal (took {path_calculation_time:.4f} seconds)")
    
    def update_geometries(self):
        self.vis.update_geometry(self.objects["start"])
        self.vis.update_geometry(self.objects["goal"])
        self.vis.update_geometry(self.objects["start_nearest"])
        self.vis.update_geometry(self.objects["goal_nearest"])
        self.vis.update_geometry(self.objects["path"])
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def toggle_obstacles(self):
        """Toggle visibility of obstacle point cloud"""
        if "obstacles" in self.objects:
            # Instead of checking if the geometry is in the visualizer,
            # we'll keep track of visibility state
            if not hasattr(self, '_obstacles_visible'):
                self._obstacles_visible = True
                
            if self._obstacles_visible:
                # Currently visible, so remove it
                self.vis.remove_geometry(self.objects["obstacles"])
                self._obstacles_visible = False
            else:
                # Currently hidden, so add it back
                self.vis.add_geometry(self.objects["obstacles"])
                self._obstacles_visible = True
                
            self.vis.poll_events()
            self.vis.update_renderer()
    
    def show_graph_view(self):
        """Show navigation graph in a new window"""
        # Create a new visualizer for graph view with a unique name
        graph_vis = o3d.visualization.Visualizer()
        graph_vis.create_window("Navigation Graph View", width=1280, height=720)

        # Add obstacles
        obstacles_pc = o3d.geometry.PointCloud()
        obstacles_pc.points = o3d.utility.Vector3dVector(obstacle_points_vis)
        colors = np.zeros((len(obstacle_points_vis), 3))
        normalized_height = (obstacle_points_vis[:, 2] - min_vals[2]) / (max_vals[2] - min_vals[2])
        viridis = plt.cm.viridis
        for i, h in enumerate(normalized_height):
            colors[i] = viridis(h)[:3]
        obstacles_pc.colors = o3d.utility.Vector3dVector(colors)
        graph_vis.add_geometry(obstacles_pc)

        # Add empty space points
        empty_space_pc = o3d.geometry.PointCloud()
        empty_space_pc.points = o3d.utility.Vector3dVector(empty_space_points)
        empty_space_pc.paint_uniform_color([0.4, 0.4, 0.9])
        graph_vis.add_geometry(empty_space_pc)

        # Add graph edges (limited to avoid too many edges)
        edge_count = 0
        max_edges = 1000  # Maximum number of edges to display

        edge_points = []
        edge_indices = []

        # Safely get edges from the graph
        try:
            edge_sample = list(G.edges())
            if len(edge_sample) > max_edges:
                # Randomly sample edges if there are too many
                np.random.seed(42)  # For reproducibility
                edge_sample = np.random.choice(len(edge_sample), max_edges, replace=False)
                edge_sample = [edge_sample[i] for i in edge_sample]

            for i, (u, v) in enumerate(edge_sample):
                if u < len(empty_space_points) and v < len(empty_space_points):
                    edge_points.append(empty_space_points[u])
                    edge_points.append(empty_space_points[v])
                    edge_indices.append([2*i, 2*i+1])
                    edge_count += 1
        except Exception as e:
            print(f"Error processing graph edges: {e}")
            # Continue with what we have

        if edge_points:
            edge_lines = o3d.geometry.LineSet()
            edge_lines.points = o3d.utility.Vector3dVector(edge_points)
            edge_lines.lines = o3d.utility.Vector2iVector(edge_indices)
            edge_lines.paint_uniform_color([0.7, 0.7, 0.7])  # Gray
            graph_vis.add_geometry(edge_lines)

        # Add path if found
        if path_found and len(waypoints) > 1:
            path_line_set = o3d.geometry.LineSet()
            path_points = waypoints
            path_lines = [[i, i+1] for i in range(len(path_points)-1)]
            path_line_set.points = o3d.utility.Vector3dVector(path_points)
            path_line_set.lines = o3d.utility.Vector2iVector(path_lines)
            path_line_set.paint_uniform_color([1, 0, 0])  # Red
            graph_vis.add_geometry(path_line_set)

        # Add start and goal markers
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        start_sphere.translate(start_point)
        start_sphere.paint_uniform_color([0, 1, 0])
        start_sphere.compute_vertex_normals()
        graph_vis.add_geometry(start_sphere)

        goal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        goal_sphere.translate(goal_point)
        goal_sphere.paint_uniform_color([0, 0, 1])
        goal_sphere.compute_vertex_normals()
        graph_vis.add_geometry(goal_sphere)

        # Set render options
        render_opt = graph_vis.get_render_option()
        render_opt.background_color = np.array([0.1, 0.1, 0.1])
        render_opt.point_size = 3.0
        render_opt.line_width = 2.0

        # Set title in status bar instead of recreating the window
        graph_vis.get_window_control().set_title(
            f"Navigation Graph (showing {edge_count} of {len(G.edges() if hasattr(G, 'edges') else [])} edges)"
        )

        # Set a default camera position
        graph_view_control = graph_vis.get_view_control()
        graph_view_control.set_zoom(0.8)

        # Run the visualization loop
        graph_vis.run()
        graph_vis.destroy_window()
    
    def reset_view(self):
        # Reset the camera view
        self.vis.reset_view_point(True)
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def run(self):
        # Setup callback for closing the UI window properly
        def on_closing():
            self.ui_window.destroy()
            self.vis.destroy_window()
        
        self.ui_window.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Start UI update loop
        def update_loop():
            self.vis.poll_events()
            self.vis.update_renderer()
            self.ui_window.after(33, update_loop)  # 30 FPS
        
        update_loop()
        self.ui_window.mainloop()

# Create and run the Open3D visualizer
visualizer = Open3DVisualizer()
visualizer.run()