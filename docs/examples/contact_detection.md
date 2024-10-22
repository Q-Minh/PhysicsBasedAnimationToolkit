# 3D Vertex-Triangle Contact Detection

This script performs vertex-triangle contact detection using GPU acceleration and visualizes the results in real-time. It uses libraries like `pbatoolkit`, `meshio`, `numpy`, `polyscope`, and `igl` to handle geometry processing, visualization, and interaction.

## Prerequisites

Before running the script, make sure the following libraries are installed:
- `pbatoolkit`
- `meshio`
- `numpy`
- `polyscope`
- `igl`

You can install them using pip:
```bash
pip numpy==1.26 scipy==1.14 meshio==5.3.5 libigl==v2.5.1 polyscope==2.2.1 ilupp==1.0.2 ipctk==1.2.0 networkx==3.3
```

## Command-line Arguments

The script expects the following argument:

- `--input`: Path to the input mesh file (required).

```bash
python script.py --input /path/to/your/meshfile
```

## Workflow Overview

### 1. Loading the Mesh
The input mesh is read using `meshio`, which extracts the points (vertices) and tetrahedral cells of the mesh. The boundary facets are calculated using `igl.boundary_facets`.

```python
imesh = meshio.read(args.input)
V, T = imesh.points.astype(np.float32), imesh.cells_dict["tetra"].astype(np.int32)
F = igl.boundary_facets(T)
```

### 2. Duplicating the Mesh
The input mesh is duplicated to simulate a multi-layered structure. Both the vertices and the cells are duplicated and updated accordingly.

```python
V = np.vstack((V.copy(), V.copy()))
T = np.vstack((T.copy(), T + nverts))
F = np.vstack((F.copy(), F + nverts))
```

### 3. BVH (Bounding Volume Hierarchy)
To optimize the detection of contact between vertices and triangles, a BVH structure is built for the tetrahedra and facets using the `pbatoolkit`.

```python
Vquery = pbat.gpu.geometry.BvhQuery(V.shape[0], 24*T.shape[0], 8*F.shape[0])
Tbvh = pbat.gpu.geometry.Bvh(T.shape[0], 0)
Fbvh = pbat.gpu.geometry.Bvh(F.shape[0], 0)
```

### 4. Animation and Visualization
The code sets up an interactive visualization using `polyscope`, where you can animate and visualize the movement of vertices and the detection of active contact points.

#### Key Variables:
- `animate`: Whether the mesh should be animated.
- `dhat`: Radius for nearest neighbor search.
- `show_nn_pairs`: Whether to display the nearest neighbors.

#### Polyscope User Interface:
A user-defined callback handles the animation, interaction, and display of results. Users can control animation speed, search radius, and enable or disable the display of nearest neighbor pairs.

```python
def callback():
    global dhat, t, speed, animate, N, n, export, show_nn_pairs, show_all_nn_pairs
    # (Code to handle user input and update visualization)
```

### 5. Exporting Frames
You can export the current frame as a screenshot if the `export` option is enabled. Screenshots are saved as `.png` files with a frame index.

```python
if export:
    ps.screenshot(f"{args.output}/{t}.png")
```

### 6. Contact Detection
Contact detection is performed using BVH, and results are displayed by marking active vertices and triangles.

```python
O = Vquery.detect_overlaps(P, SV, ST, Tbvh)
N = Vquery.detect_contact_pairs(P, SV, SF, BV, BF, Fbvh, dhat)
```

## Code Sample

```python
import pbatoolkit as pbat
import argparse
import meshio
import numpy as np
import polyscope as ps
import igl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="3D vertex-triangle contact detection")
    parser.add_argument("-i", "--input", help="Path to input mesh", required=True)
    args = parser.parse_args()

    imesh = meshio.read(args.input)
    V, T = imesh.points.astype(np.float32), imesh.cells_dict["tetra"].astype(np.int32)
    F = igl.boundary_facets(T)

    # Duplicate mesh, BVH setup
    V = np.vstack((V.copy(), V.copy()))
    T = np.vstack((T.copy(), T + V.shape[0] // 2))
    F = np.vstack((F.copy(), F + V.shape[0] // 2))

    profiler = pbat.profiling.Profiler()
    ps.init()

    # Initialize visualization
    sm = ps.register_surface_mesh("Boundary Mesh", V, F)
    vm = ps.register_volume_mesh("Mesh", V, T)

    def callback():
        # Update positions and visualization
        sm.update_vertex_positions(V)
        vm.update_vertex_positions(V)
        
    ps.set_user_callback(callback)
    ps.show()
```

### Running the Script

Once the script is ready, you can run it by passing the path to the mesh file as an argument:

```bash
python script.py --input /path/to/your/meshfile
```