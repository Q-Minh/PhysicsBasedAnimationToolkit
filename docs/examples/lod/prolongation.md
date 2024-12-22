# Shape Function Interpolation Prolongation

This script demonstrates shape function interpolation prolongation for multilevel tetrahedral meshes. It applies a signal to deform the coarsest level of a mesh hierarchy and propagates the deformation to finer levels using prolongation operators. The simulation showcases dynamic deformation modes based on wave functions and allows real-time visualization.

## Prerequisites

Ensure the following Python libraries are installed before running the script:

- `pbatoolkit`
- `meshio`
- `numpy`
- `polyscope`
- `argparse`

You can install the required packages using `pip`:

```bash
pip install pbatoolkit meshio numpy polyscope argparse
```

*Note*: If `pbatoolkit` is not available via `pip`, refer to its official documentation for installation instructions.

## Command-line Arguments

The script accepts several command-line arguments:

- `-i`, `--input`: **Paths to input tetrahedral meshes** (required). The meshes must be ordered from finest to coarsest for proper interpolation.

### Example Usage

```bash
python interpolation_prolongation.py -i mesh1.vtk mesh2.vtk mesh3.vtk
```

- **Explanation**:
  - Loads `mesh1.vtk`, `mesh2.vtk`, and `mesh3.vtk` as the input tetrahedral meshes.
  - Constructs a multilevel hierarchy for shape function interpolation.

## Workflow Overview

### 1. Loading and Preparing Meshes

The input tetrahedral meshes are loaded and converted into FEM meshes.

```python
# Load input meshes
imeshes = [meshio.read(input) for input in args.input]
meshes = [
    pbat.fem.Mesh(
        imesh.points.T,
        imesh.cells_dict["tetra"].T, element=pbat.fem.Element.Tetrahedron)
    for imesh in imeshes
]
```

### 2. Constructing Levels and Prolongation Operators

Each input mesh is treated as a level in the hierarchy. Prolongation operators are computed to interpolate between levels.

```python
# Compute levels and prolongation operators
levels = [
    pbat.sim.vbd.lod.Level(mesh) for mesh in meshes
]
prolongators = [
    pbat.sim.vbd.lod.Prolongation(fine_mesh, coarse_mesh) for
    (fine_mesh, coarse_mesh) in zip(meshes[:-1], meshes[1:])
]
prolongators.reverse()
```

### 3. Computing Modes on the Coarsest Mesh

The eigenmodes of the rest pose are computed on the coarsest mesh to generate the dynamic deformation signal.

```python
# Compute modes on coarsest mesh
w, U = pbat.fem.rest_pose_hyper_elastic_modes(meshes[-1])
```

### 4. Visualizing the Mesh Hierarchy

Polyscope is used to visualize the hierarchy. Transparency is adjusted for each level to distinguish between finer and coarser meshes.

```python
# Create visual meshes
V, F = zip(*[
    pbat.geometry.simplex_mesh_boundary(mesh.E, n=mesh.X.shape[1])
    for mesh in meshes
])
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_ground_plane_mode("shadow_only")
ps.init()

min_transparency, max_transparency = 0.25, 1.
n_levels = len(levels)
lsms = [None]*n_levels
for l, level in enumerate(levels):
    lsms[l] = ps.register_surface_mesh(f"Level {l}", level.x.T, F[l].T)
    transparency = min_transparency + (n_levels-l-1) * \
        (max_transparency - min_transparency) / (n_levels-1)
    lsms[l].set_transparency(transparency)
```

### 5. Applying Dynamic Deformation Signal

A sinusoidal signal is applied to the coarsest mesh and prolonged to finer meshes in the hierarchy.

```python
# Signal function
def signal(w: float, v: np.ndarray, t: float, c: float, k: float):
    u = c * np.sin(k * w * t) * v
    return u

# Simulation parameters
mode = 6
t0 = time.time()
t = 0
c = 10
k = 0.1

def callback():
    global mode, c, k
    changed, mode = imgui.InputInt("Mode", mode)
    changed, c = imgui.InputFloat("Wave amplitude", c)
    changed, k = imgui.InputFloat("Wave frequency", k)

    t = time.time() - t0
    X = levels[-1].X
    u = signal(w[mode], U[:, mode], t, c, k).reshape(X.shape, order="F")
    levels[-1].x = X + u
    for l, prolongator in enumerate(prolongators):
        prolongator.apply(levels[-1-l], levels[-2-l])
    for l, level in enumerate(levels):
        lsms[l].update_vertex_positions(level.x.T)

# Register callback and start visualization
ps.set_user_callback(callback)
ps.show()
```
