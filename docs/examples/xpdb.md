# XPBD Elastic Simulation Using Linear FEM Tetrahedra

This script performs an **Extended Position Based Dynamics (XPBD)** elastic simulation using **linear Finite Element Method (FEM) tetrahedra**. It simulates the elastic behavior of 3D meshes under gravity, handling collisions and constraints to produce realistic deformations over time.

## Prerequisites

Ensure the following Python libraries are installed before running the script:

- `pbatoolkit`
- `meshio`
- `polyscope`
- `numpy`
- `scipy`
- `libigl`
- `networkx`
- `qpsolvers`

You can install the required packages using `pip`:

```bash
pip install pbatoolkit meshio polyscope numpy scipy libigl networkx qpsolvers
```

*Note*: If `pbatoolkit` is not available via `pip`, refer to its official documentation for installation instructions.

## Command-line Arguments

The script accepts several command-line arguments to customize the simulation:

- `-i`, `--input`: **Paths to input mesh files** (required). Supports multiple mesh files.
- `-o`, `--output`: **Path to output directory** for saving results (default: current directory).
- `-m`, `--mass-density`: **Mass density** of the material (default: `1000.0`).
- `-Y`, `--young-modulus`: **Young's modulus** of the material (default: `1e6`).
- `-n`, `--poisson-ratio`: **Poisson's ratio** of the material (default: `0.45`).
- `-t`, `--translation`: **Translation multiplier** in the z-axis between input meshes (default: `0.1`).
- `--percent-fixed`: **Percentage of the mesh along the z-axis to fix** (default: `0.01` or 1%).

### Example Usage

```bash
python xpbd_simulation.py -i mesh1.vtk mesh2.vtk -o output_directory -m 1200 -Y 2e6 -n 0.4 -t 0.05 --percent-fixed 0.02
```

- **Explanation**:
  - Loads `mesh1.vtk` and `mesh2.vtk`.
  - Outputs results to `output_directory`.
  - Sets mass density to `1200`.
  - Sets Young's modulus to `2e6`.
  - Sets Poisson's ratio to `0.4`.
  - Applies a translation multiplier of `0.05` between meshes.
  - Fixes `2%` of the mesh along the z-axis.

## Workflow Overview

### 1. Loading and Combining Meshes

Multiple input meshes are loaded and combined into a single FEM mesh. Each mesh's position is adjusted along the z-axis to prevent overlap based on the translation multiplier.

```python
# Load input meshes
imeshes = [meshio.read(input) for input in args.inputs]
V, C = [imesh.points / (imesh.points.max() - imesh.points.min()) for imesh in imeshes], [
    imesh.cells_dict["tetra"] for imesh in imeshes]

# Translate meshes along z-axis
for i in range(len(V) - 1):
    extent = V[i][:, -1].max() - V[i][:, -1].min()
    offset = V[i][:, -1].max() - V[i+1][:, -1].min()
    V[i+1][:, -1] += offset + extent * args.translation

# Combine vertices and cells
V, Vsizes, C, Coffsets, Csizes = combine(V, C)
```

### 2. Constructing the FEM Mesh

A FEM mesh is created using the combined vertices and cells. Boundary facets are extracted for collision handling.

```python
# Create FEM mesh
mesh = pbat.fem.Mesh(
    V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=1)
V = mesh.X.T
C = mesh.E.T

# Extract boundary facets
F, Fsizes = boundary_triangles(C, Coffsets, Csizes)
BV, BF = bodies(Vsizes, Fsizes)
```

### 3. Setting Up Mass Matrix and Load Vector

The mass matrix and load vector are computed based on the material properties and external forces like gravity.

```python
# Mass matrix
rho = args.rho
M, detJeM = pbat.fem.mass_matrix(mesh, rho=rho, dims=1, lump=True)
m = np.array(M.diagonal()).squeeze()

# Load vector (gravity)
g = np.zeros(mesh.dims)
g[-1] = -9.81
f, detJeF = pbat.fem.load_vector(mesh, rho * g, flatten=False)
```

### 4. Defining Material Properties

Material properties such as Young's modulus and Poisson's ratio are used to compute Lame parameters for the simulation.

```python
# Material properties
Y = np.full(mesh.E.shape[1], args.Y)
nu = np.full(mesh.E.shape[1], args.nu)
mue = Y / (2 * (1 + nu))
lambdae = (Y * nu) / ((1 + nu) * (1 - 2 * nu))
```

### 5. Applying Boundary Conditions

Dirichlet boundary conditions are applied to a specified percentage of the mesh along the z-axis to fix certain vertices.

```python
# Define fixed vertices
Xmin = mesh.X.min(axis=1)
Xmax = mesh.X.max(axis=1)
extent = Xmax - Xmin
Xmax[-1] = Xmin[-1] + args.percent_fixed * extent[-1]
aabb = pbat.geometry.aabb(np.vstack((Xmin, Xmax)).T)
vdbc = aabb.contained(mesh.X)

# Inverse mass vector
minv = 1 / m
minv[vdbc] = 0.  # Fix vertices by setting inverse mass to zero
```

### 6. Initializing the XPBD Solver

The XPBD solver is set up with the mesh, constraints, and material properties. Constraint partitions are created to optimize the simulation.

```python
# Setup XPBD
Vcollision = np.unique(F)
VC = Vcollision[:, np.newaxis].T
BV = BV[Vcollision]
max_overlaps = 20 * mesh.X.shape[1]
max_contacts = 8 * max_overlaps

xpbd = pbat.gpu.xpbd.Xpbd(mesh.X, VC, F.T, mesh.E,
                          BV, BF, max_overlaps, max_contacts)
xpbd.f = f
xpbd.minv = minv
xpbd.lame = np.vstack((mue, lambdae))

# Partition constraints
partitions, GC = partition_constraints(mesh.E.T)
xpbd.partitions = partitions
alphac = 0
xpbd.set_compliance(
    alphac * np.ones(VC.shape[1]), pbat.gpu.xpbd.ConstraintType.Collision)
xpbd.prepare()
```

### 7. Setting Up Visualization with Polyscope

Polyscope is initialized for real-time visualization of the simulation mesh and fixed vertices.

```python
# Initialize Polyscope
ps.set_verbosity(0)
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_ground_plane_mode("shadow_only")
ps.set_ground_plane_height_factor(0.5)
ps.set_program_name("eXtended Position Based Dynamics")
ps.init()

# Register meshes for visualization
vm = ps.register_volume_mesh("Simulation mesh", mesh.X.T, mesh.E.T)
vm.add_scalar_quantity("Coloring", GC, defined_on="cells", cmap="jet")
pc = ps.register_point_cloud("Dirichlet", mesh.X[:, vdbc].T)
```

### 8. Running the Simulation Loop with User Interaction

A simulation loop is created with interactive controls for adjusting parameters, stepping through the simulation, and resetting.

```python
# Simulation parameters
dt = 0.01
iterations = 1
substeps = 50
animate = False
export = False
t = 0

profiler = pbat.profiling.Profiler()

def callback():
    global dt, iterations, substeps, alphac
    global animate, export, t
    global profiler

    # GUI controls
    changed, dt = imgui.InputFloat("dt", dt)
    changed, iterations = imgui.InputInt("Iterations", iterations)
    changed, substeps = imgui.InputInt("Substeps", substeps)
    alphac_changed, alphac = imgui.InputFloat(
        "Collision compliance", alphac, format="%.10f")
    changed, animate = imgui.Checkbox("Animate", animate)
    changed, export = imgui.Checkbox("Export", export)
    step = imgui.Button("Step")
    reset = imgui.Button("Reset")

    # Reset simulation
    if reset:
        xpbd.x = mesh.X
        xpbd.v = np.zeros(mesh.X.shape)
        vm.update_vertex_positions(mesh.X.T)
        t = 0

    # Update compliance if changed
    if alphac_changed:
        xpbd.set_compliance(
            alphac * np.ones(VC.shape[1]), pbat.gpu.xpbd.ConstraintType.Collision)

    # Advance simulation
    if animate or step:
        profiler.begin_frame("Physics")
        xpbd.step(dt, iterations, substeps)
        profiler.end_frame("Physics")

        # Update visualization
        V = xpbd.x.T
        min_v, max_v = np.min(V, axis=0), np.max(V, axis=0)
        xpbd.scene_bounding_box = min_v, max_v
        if export:
            ps.screenshot(f"{args.output}/{t}.png")

        vm.update_vertex_positions(V)
        t += 1

    imgui.Text(f"Frame={t}")

# Register callback and start visualization
ps.set_user_callback(callback)
ps.show()
```