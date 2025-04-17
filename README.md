# Waypoint Generation & Navigation Planner

This repository implements a GPU‑accelerated ray‑casting algorithm to identify navigable “supported cells” beneath overhangs in a 3D point cloud, build a connectivity graph, and compute collision‑free paths. Key components leverage [FAISS](https://github.com/facebookresearch/faiss), [CuPy](https://cupy.dev/), [Open3D](http://www.open3d.org/), and [PyVista](https://docs.pyvista.org/).

---

## Contents

- [Overview](#overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Algorithm Details](#algorithm-details)  
- [Functions & API](#functions--api)  
- [Visualization](#visualization)  
- [License](#license)  
- [Contact](#contact)  

---

## Overview

The core script is [waypoints.py](waypoints.py), which:

1. Loads a point cloud (`Scene.pcd`).  
2. Builds a FAISS GPU index for obstacle proximity queries.  
3. Generates a 3D grid of candidate points.  
4. Casts vertical rays from multiple heights to detect surfaces under overhangs.  
5. Checks vertical clearance using a nearest‑neighbor search.  
6. Removes duplicates and clusters supported cells.  
7. Builds a connectivity graph among cells.  
8. Finds a shortest path (A⋆) between start and goal points.  
9. Provides interactive 3D visualizations (Open3D/PyVista/Matplotlib).

![Architecture Diagram](docs/overview.png) _\<-- placeholder for your diagram_

---

## Features

- GPU‑accelerated nearest‑neighbor queries via FAISS  
- Multi‑start ray casting with CuPy kernels  
- Clearance checks using \(d > r\) where  
  \[
    d = \sqrt{(x_o - x_p)^2 + (y_o - y_p)^2 + (z_o - z_p)^2}\,, 
    \quad r = \text{clearance radius}
  \]  
- Duplicate removal with vertical threshold  
- Connectivity graph built with [`networkx`](https://networkx.org/)  
- A* path planning on the graph  
- Interactive sliders & camera controls  

---

## Installation

```sh
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

# Install dependencies
pip install faiss-cpu cupy-cuda11x numpy scipy networkx matplotlib pandas pyvista open3d
```

---

## Usage

1. Place your point cloud as `Scene.pcd` in the workspace root.  
2. Run the script:
   ```sh
   python waypoints.py
   ```
3. Adjust parameters (grid size, clearance, batch size) directly in `waypoints.py` or via the interactive GUI.

---

## Algorithm Details

### 1. Grid & Ray Origins

- Grid cell centers \( (x_i, y_j, z_k) \) are defined by
  $$
    x_i = x_{\min} + \frac{\Delta x}{2} + i\,\Delta x,\quad
    y_j = y_{\min} + \frac{\Delta y}{2} + j\,\Delta y
  $$
- Multiple start heights \( z_k \) linearly spaced above obstacles.

### 2. Ray‑Casting Kernel

- For each ray origin \(\mathbf{o}\) and direction \(\mathbf{d}=(0,0,-1)\), we approximate intersection by checking:
  $$
    \mathrm{proj}_t = (\mathbf{p}-\mathbf{o})\cdot\mathbf{d},\quad
    \|\mathbf{p}- (\mathbf{o}+t\mathbf{d})\|^2 < r^2
  $$
- Implemented as a [CuPy ElementwiseKernel](waypoints.py).

### 3. Clearance Check

- For each hit point \(\mathbf{h}\), verify no obstacle lies within a sphere of radius \(r\) above:
  $$
    \|\mathbf{q} - (\mathbf{h} + r\,\mathbf{e}_z)\| > r\;\;\forall\,\mathbf{q}\in\text{obstacles}
  $$

### 4. Connectivity Graph

- Nodes: supported cell centers.  
- Edges: connect neighbors within horizontal radius \(R\) and vertical step \(\Delta z\).  
- Path planning via A* (`networkx.astar_path`).

---

## Functions & API

- **`improved_look_down_algorithm_multi_start(...)`**  
  Computes supported cells & graph.  
- **`connect_supported_cells(...)`**  
  Builds a `networkx.Graph` of cell connectivity.  
- **`animate_ray_casting_open3d(...)`**  
  Interactive Open3D animation of ray casting stages.  
- **Visualization helpers**:  
  - `visualize_supported_cells_3d()`  
  - `visualize_empty_space_points()`

Refer to [waypoints.py](waypoints.py) for full signatures and docstrings.

---

## Visualization

![Ray Casting Animation](docs/ray_casting.png) _\<-- placeholder_  
![Supported Cells](docs/supported_cells.png) _\<-- placeholder_  

- **Matplotlib 3D**: final supported cells & obstacles.  
- **Open3D GUI**: real‑time ray casting with sliders.  
- **Network graph view**: connectivity edges & computed path.

---

## License

This project is licensed under **GNU GPL v2**. See [LICENSE.GPL](LICENSE.GPL) for details.

---