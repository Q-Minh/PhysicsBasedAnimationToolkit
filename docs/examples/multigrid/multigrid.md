# Multigrid VBD Elastic Simulation Using Linear FEM Tetrahedra

This script performs a **Multigrid Vertex Block Descent (VBD)** elastic simulation using **linear Finite Element Method (FEM) tetrahedra**. It simulates the elastic behavior of 3D meshes under gravity, handling multigrid hierarchies, collisions, and constraints to produce realistic deformations over time. The simulation employs a multigrid approach to optimize performance and convergence, leveraging adaptive refinement and coarse-graining techniques.

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

The script accepts several command-line arguments to customize the simulation:

- `-i`, `--input`: **Path to input mesh file** (required).
- `-c`, `--cages`: **Paths to cage tetrahedral mesh files** (required).
- `--cycle`: **Multigrid cycle** to perform at each level (required).
- `--smoothing-iters`: **Number of smoothing iterations** at each level of the cycle (required).
- `-o`, `--output`: **Path to output directory** for saving results (default: current directory).
- `-m`, `--mass-density`: **Mass density** of the material (default: `1000.0`).
- `-Y`, `--young-modulus`: **Young's modulus** of the material (default: `1e6`).
- `-n`, `--poisson-ratio`: **Poisson's ratio** of the material (default: `0.45`).
- `--percent-fixed`: **Percentage of the mesh along the fixed axis to fix** (default: `0.01` or 1%).
- `--fixed-axis`: **Axis to fix** (`0` for x, `1` for y, `2` for z) (default: `2`).
- `--fixed-end`: **Which end to fix** (`min` or `max`) of the bounding box along the fixed axis (default: `min`).

### Example Usage

```bash
python multigrid_vbd_simulation.py -i scene.vtk -c cage1.vtk cage2.vtk --cycle 1 2 1 --smoothing-iters 10 5 5 -o output_directory -m 1200 -Y 2e6 -n 0.4 --percent-fixed 0.02 --fixed-axis 2 --fixed-end max
```

- **Explanation**:
  - Loads `scene.vtk` as the main mesh.
  - Loads `cage1.vtk` and `cage2.vtk` as cage meshes for multigrid refinement.
  - Performs a multigrid cycle with levels `1 -> 2 -> 1` and smoothing iterations of `10`, `5`, and `5` at each level.
  - Outputs results to `output_directory`.
  - Sets mass density to `1200`.
  - Sets Young's modulus to `2e6`.
  - Sets Poisson's ratio to `0.4`.
  - Fixes `2%` of the mesh at the maximum end along the z-axis.

## Workflow Overview

### 1. Loading and Preprocessing Meshes

The input mesh and cage meshes are loaded, normalized, and prepared for multigrid processing.

```python
# Load input mesh
imesh = meshio.read(args.input)
V, C = imesh.points.astype(np.float64), imesh.cells_dict["tetra"].astype(np.int64)
center = V.mean(axis=0)
scale = V.max() - V.min()
V = (V - center) / scale

# Load and normalize cage meshes
icmeshes = [meshio.read(cage) for cage in args.cages]
VC, CC = [icmesh.points.astype(np.float64) for icmesh in icmeshes], [
    icmesh.cells_dict["tetra"].astype(np.int64) for icmesh in icmeshes
]
for c in range(len(VC)):
    VC[c] = (VC[c] - center) / scale
```

### 2. Setting Up the FEM Mesh and Cages

Finite Element Method (FEM) meshes are constructed for the input and cage meshes.

```python
# Create FEM meshes
mesh = pbat.fem.Mesh(V.T, C.T, element=pbat.fem.Element.Tetrahedron)
cmeshes = [
    pbat.fem.Mesh(VCc.T, CCc.T, element=pbat.fem.Element.Tetrahedron)
    for (VCc, CCc) in zip(VC, CC)
]
```

### 3. Defining Material Properties

Material properties are defined using Lame constants.

```python
# Define material properties
Y = np.full(mesh.E.shape[1], args.Y)
nu = np.full(mesh.E.shape[1], args.nu)
mue = Y / (2 * (1 + nu))
lambdae = (Y * nu) / ((1 + nu) * (1 - 2 * nu))
rhoe = np.full(mesh.E.shape[1], args.rho)
```

### 4. Applying Boundary Conditions

Boundary conditions are applied to fix a specified percentage of the mesh along a chosen axis.

```python
# Set Dirichlet boundary conditions
Xmin = mesh.X.min(axis=1)
Xmax = mesh.X.max(axis=1)
extent = Xmax - Xmin
if args.fixed_end == "min":
    Xmax[args.fixed_axis] = (
        Xmin[args.fixed_axis] + args.percent_fixed * extent[args.fixed_axis]
    )
    Xmin[args.fixed_axis] -= args.percent_fixed * extent[args.fixed_axis]
elif args.fixed_end == "max":
    Xmin[args.fixed_axis] = (
        Xmax[args.fixed_axis] - args.percent_fixed * extent[args.fixed_axis]
    )
    Xmax[args.fixed_axis] += args.percent_fixed * extent[args.fixed_axis]
aabb = pbat.geometry.aabb(np.vstack((Xmin, Xmax)).T)
vdbc = aabb.contained(mesh.X)
```

### 5. Initializing Multigrid VBD Solver

The multigrid hierarchy is created, and the VBD solver is initialized with the input mesh, cage meshes, and material properties.

```python
# Setup multigrid hierarchy
cycle = [int(l) for l in args.cycle]
siters = [int(iters) for iters in args.siters]
hierarchy = pbat.sim.vbd.multigrid.Hierarchy(
    data, cmeshes, cycle=cycle, siters=siters
)

# Initialize VBD solver
vbd = pbat.sim.vbd.multigrid.Integrator()
```

### 6. Setting Up Visualization

Polyscope is initialized for interactive visualization of the simulation mesh and results.

```python
# Initialize Polyscope
ps.set_verbosity(0)
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_ground_plane_mode("shadow_only")
ps.set_ground_plane_height_factor(0.5)
ps.set_program_name("Vertex Block Descent")
ps.init()

# Register volume mesh for visualization
vm = ps.register_volume_mesh("Simulation mesh", V, C)
vm.add_scalar_quantity(
    "Coloring", hierarchy.data.colors, defined_on="vertices", cmap="jet"
)
```

### 7. Running the Simulation Loop

An interactive simulation loop allows users to step through the simulation, adjust parameters, and visualize results.

```python
# Simulation loop parameters
dt = 0.01
substeps = 1
RdetH = 1e-10
animate = False
export = False
t = 0

# Callback function for Polyscope GUI
def callback():
    global dt, substeps, RdetH, animate, export, t

    # GUI controls
    changed, dt = imgui.InputFloat("dt", dt)
    changed, substeps = imgui.InputInt("Substeps", substeps)
    changed, animate = imgui.Checkbox("Animate", animate)
    changed, export = imgui.Checkbox("Export", export)
    step = imgui.Button("Step")
    reset = imgui.Button("Reset")

    if animate or step:
        vbd.step(dt, substeps, hierarchy)

        # Update visuals
        V = hierarchy.data.x.T
        vm.update_vertex_positions(V)
        t += 1

        if export:
            ps.screenshot()

    imgui.Text(f"Frame={t}")
    imgui.Text("Using CPU Multi-Scale VBD integrator")

ps.set_user_callback(callback)
ps.show()
