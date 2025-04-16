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
    """Checks clearance above potential hit points using FAISS GPU index with kNN search."""
    if len(potential_hits) == 0:
        return np.array([], dtype=bool)

    # Define clearance check points above the hit surface
    check_points = potential_hits.copy()
    check_points[:, 2] += min_clearance / 2.0
    check_points_f32 = check_points.astype(np.float32)
    
    # Search radius squared (used for filtering distances)
    threshold_sq = ((min_clearance / 2.0) + (cell_size / 2.0))**2
    
    # Use k-nearest neighbor search instead of range search
    k = 10  # Check several closest points to ensure we don't miss anything
    D, I = gpu_index_obstacles.search(check_points_f32, k)
    
    # For each query point, check if any of the k nearest neighbors are too close
    clearance_ok = np.all(D > threshold_sq, axis=1)
    
    return clearance_ok


def improved_look_down_algorithm_multi_start(
    obstacle_points, grid_min, grid_max, cell_size=0.1,
    num_z_levels=5, min_clearance=0.5, batch_size=10000
):
    """
    Ray casting algorithm with multiple start heights to find surfaces under overhangs.

    Args:
        obstacle_points: The obstacle point cloud (N x 3 numpy array)
        grid_min: Minimum coordinates of the grid (3-element array/list)
        grid_max: Maximum coordinates of the grid (3-element array/list)
        cell_size: Size of each cell (default: 0.1)
        num_z_levels: Number of different Z heights to start rays from (default: 5)
        min_clearance: Minimum vertical clearance required above a supported cell (default: 0.5)
        batch_size: Number of rays to process per GPU batch (default: 10000)

    Returns:
        Array of points representing the centers of supported cells
    """
    if not gpu_accelerated:
        print("GPU acceleration not available. Aborting.")
        # Or call a CPU fallback if implemented
        return cpu_raycasting_fallback(obstacle_points, grid_min, grid_max, cell_size)

    print(f"Applying multi-start GPU ray casting (Z levels={num_z_levels})...")
    start_time = time.time()

    # --- 1. Prepare Grid and Ray Origins ---
    x_cells = np.arange(grid_min[0] + cell_size / 2, grid_max[0], cell_size)
    y_cells = np.arange(grid_min[1] + cell_size / 2, grid_max[1], cell_size)
    # Define multiple Z starting heights
    z_starts = np.linspace(grid_max[2] + cell_size, grid_min[2] + min_clearance, num_z_levels) # Ensure lowest start is above min clearance needs

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
    # Kernel now *only* finds the first hit point. Clearance check is done later.
    simple_ray_intersection_kernel = cp.ElementwiseKernel(
        'float32 ox, float32 oy, float32 oz, float32 dx, float32 dy, float32 dz, ' +
        'float32 grid_min_z, float32 cell_sz, raw float32 obstacles', # Renamed cell_size to cell_sz
        'float32 hit_x, float32 hit_y, float32 hit_z, int32 is_hit', # Output only hit info
        '''
        const float ray_epsilon = 0.0001f;
        const float max_dist = oz - grid_min_z + cell_sz; // Max ray distance down to below grid floor
        const float point_radius_sq = (cell_sz * 0.4f) * (cell_sz * 0.4f); // Approx point influence (squared), slightly smaller than half cell

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

        # Execute kernel
        simple_ray_intersection_kernel(
            batch_origins[:, 0], batch_origins[:, 1], batch_origins[:, 2],
            batch_directions[:, 0], batch_directions[:, 1], batch_directions[:, 2],
            grid_min[2], cell_size, cp_obstacle_points, # Pass grid_min_z and cell_size
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
            actual_hits, obstacle_points_f32, min_clearance, cell_size, gpu_index_obstacles
        )
        supported_hits = actual_hits[clearance_ok]
        print(f"Found {len(supported_hits)} hits with sufficient clearance.")
    else:
        supported_hits = np.empty((0, 3))
        print("No potential hits found, skipping clearance check.")

    # --- Optional: Add ground plane cells ---
    # If desired, add ground cells where no obstacles were hit *and* clearance is guaranteed
    # This requires tracking which XY cells never had a valid hit + clearance check for ground points.
    # For simplicity, we focus on hits on obstacles first.

    # --- 6. Post-process: Consolidate and Remove Duplicates ---
    # Use the dictionary approach to handle multiple Z hits in the same XY column
    # and remove hits that are too close vertically.
    supported_cells = []
    if len(supported_hits) > 0:
        print("Consolidating results and removing close duplicates...")
        placed_cells_dict = {}
        # Sort by Z descending, so we prioritize higher surfaces if they are too close
        sorted_indices = np.argsort(supported_hits[:, 2])[::-1]
        sorted_supported_hits = supported_hits[sorted_indices]

        for cell in sorted_supported_hits:
            x, y, z = cell
            # Create a key based on discretized XY coordinates
            xy_key = f"{int(x / cell_size)}_{int(y / cell_size)}"

            if xy_key not in placed_cells_dict:
                placed_cells_dict[xy_key] = [z]
                supported_cells.append(cell)
            else:
                # Check if this hit is too close to existing hits in this column
                is_too_close = False
                for existing_z in placed_cells_dict[xy_key]:
                    if abs(existing_z - z) < cell_size: # Check if vertically too close
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
        # Assuming connect_supported_cells uses FAISS GPU index internally
        G = connect_supported_cells(supported_cells, cell_size, gpu_index_obstacles.res) # Pass faiss resources
    else:
        G = None # Or nx.Graph()

    # Cleanup FAISS GPU resources if they are not needed elsewhere
    # del gpu_index_obstacles # Or manage resource lifetime appropriately

    return supported_cells, G # Return graph too

def connect_supported_cells(supported_cells, cell_size, faiss_res=None):
    """
    Build a connectivity graph between supported cells using FAISS GPU.
    Identifies connected components and isolated regions.
    Args:
        supported_cells (np.array): Nx3 array of cell centers.
        cell_size (float): The size of the cells.
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

    # Connect nearby cells (within sqrt(2) * cell_size for adjacent, maybe slightly more for diagonal/vertical)
    connection_threshold_sq = (cell_size * 1.5)**2 # Squared distance threshold
    k_neighbors = 10 # Search for up to 10 nearest neighbors

    print(f"Finding connections within radius {connection_threshold_sq**0.5:.3f}...")
    # Batch processing for potentially large number of supported cells
    batch_size = 5000
    num_cells = len(supported_cells)

    for i in range(0, num_cells, batch_size):
        batch_end = min(i + batch_size, num_cells)
        # Query points directly on GPU
        batch_query_cp = cp.asarray(supported_cells_f32[i:batch_end])

        # Use range search for fixed radius or knn search and filter by distance
        # Let's use knn search and filter, might be more robust than guessing radius
        D, I = gpu_support_index.search(batch_query_cp, k_neighbors) # D is squared distances

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
                        # Check vertical distance constraint if needed (e.g., robot can't step too high)
                        z_diff = abs(supported_cells[cell_idx, 2] - supported_cells[neighbor_idx, 2])
                        if z_diff < cell_size * 1.1: # Allow stepping up/down one cell height
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
cell_size = 0.1  # (can adjust based on your robot size)
supported_cells, G = improved_look_down_algorithm_multi_start(obstacle_points=obstacle_points, 
    grid_min=min_vals, grid_max=max_vals, cell_size=cell_size, num_z_levels=5, min_clearance=0.001)

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
# G = nx.Graph()
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
# G = nx.Graph()
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