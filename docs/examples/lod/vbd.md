# VBD Elastic Simulation Using Linear FEM Tetrahedra

This script demonstrates a **Vertex Block Descent (VBD)** elastic simulation using **linear Finite Element Method (FEM) tetrahedra**. It simulates elastic deformations in a 3D mesh under external forces, leveraging multiscale hierarchies for efficiency. The setup allows for interactive visualization and dynamic parameter tuning.

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

- `-i`, `--input`: **Path to input tetrahedral mesh** (required).
- `-c`, `--cages`: **Paths to cage tetrahedral meshes** (required).
- `-o`, `--output`: **Path to output directory** for saving results (default: current directory).
- `-m`, `--mass-density`: **Mass density** of the material (default: `1000.0`).
- `-Y`, `--young-modulus`: **Young's modulus** of the material (default: `1e6`).
- `-n`, `--poisson-ratio`: **Poisson's ratio** of the material (default: `0.45`).
- `--percent-fixed`: **Percentage of the mesh along the fixed axis to constrain** (default: `0.01`).
- `--fixed-axis`: **Axis to constrain** (`0` for x, `1` for y, `2` for z) (default: `2`).
- `--fixed-end`: **End of the axis to constrain** (`min` or `max`) (default: `min`).

### Example Usage

```bash
python vbd_simulation.py -i input_mesh.vtk -c cage1.vtk cage2.vtk -o results/ -m 1200 -Y 2e6 -n 0.4 --percent-fixed 0.02 --fixed-axis 2 --fixed-end max
```

- **Explanation**:
  - Loads `input_mesh.vtk` as the main simulation mesh.
  - Loads `cage1.vtk` and `cage2.vtk` as cage meshes.
  - Outputs results to the `results/` directory.
  - Sets material properties: mass density to `1200`, Young's modulus to `2e6`, and Poisson's ratio to `0.4`.
  - Constrains 2% of the mesh at the maximum end along the z-axis.

## Workflow Overview

### 1. Loading and Preparing Meshes

The input tetrahedral mesh and cage meshes are loaded and normalized for simulation.

```python
# Load input and cage meshes
imesh = meshio.read(args.input)
V, C = imesh.points.astype(np.float64), imesh.cells_dict["tetra"].astype(np.int64)
center = V.mean(axis=0)
scale = V.max() - V.min()
V = (V - center) / scale

icmeshes = [meshio.read(cage) for cage in args.cages]
VC, CC = [icmesh.points.astype(np.float64) for icmesh in icmeshes], [
    icmesh.cells_dict["tetra"].astype(np.int64) for icmesh in icmeshes
]
for c in range(len(VC)):
    VC[c] = (VC[c] - center) / scale

# Construct FEM meshes
mesh = pbat.fem.Mesh(V.T, C.T, element=pbat.fem.Element.Tetrahedron)
cmeshes = [
    pbat.fem.Mesh(VCc.T, CCc.T, element=pbat.fem.Element.Tetrahedron)
    for (VCc, CCc) in zip(VC, CC)
]
```

### 2. Defining Material Properties

Material properties such as Young's modulus and Poisson's ratio are converted to Lame coefficients.

```python
# Define material properties
Y = np.full(mesh.E.shape[1], args.Y)
nu = np.full(mesh.E.shape[1], args.nu)
mue = Y / (2 * (1 + nu))
lambdae = (Y * nu) / ((1 + nu) * (1 - 2 * nu))
rhoe = np.full(mesh.E.shape[1], args.rho)
```

### 3. Applying Boundary Conditions

Dirichlet boundary conditions are applied to constrain specific parts of the mesh along a chosen axis.

```python
# Apply boundary conditions
Xmin = mesh.X.min(axis=1)
Xmax = mesh.X.max(axis=1)
extent = Xmax - Xmin
if args.fixed_end == "min":
    Xmax[args.fixed_axis] = (
        Xmin[args.fixed_axis] + args.percent_fixed * extent[args.fixed_axis]
    )
elif args.fixed_end == "max":
    Xmin[args.fixed_axis] = (
        Xmax[args.fixed_axis] - args.percent_fixed * extent[args.fixed_axis]
    )
aabb = pbat.geometry.aabb(np.vstack((Xmin, Xmax)).T)
vdbc = aabb.contained(mesh.X)
```

### 4. Setting Up the VBD Solver

The VBD solver is initialized using the input and cage meshes, along with multiscale integration parameters.

```python
# Setup VBD solver
VF, FF = pbat.geometry.simplex_mesh_boundary(C.T, V.shape[0])
data = (
    pbat.sim.vbd.Data()
    .with_volume_mesh(V.T, C.T)
    .with_surface_mesh(VF, FF)
    .with_material(rhoe, mue, lambdae)
    .with_dirichlet_vertices(vdbc, muD=args.Y)
    .with_initialization_strategy(pbat.sim.vbd.InitializationStrategy.AdaptivePbat)
    .construct(validate=True)
)

# Setup multiscale VBD
cycle = [-1, 0, -1]
siters = [1, 0, 0]
riters = [0, 0]
hierarchy = pbat.sim.vbd.lod.Hierarchy(data, cmeshes, cycle, siters, riters)
vbd = pbat.sim.vbd.lod.Integrator()
```

### 5. Visualizing the Simulation

Polyscope is used to visualize the simulation results in real time, including volume meshes and constrained vertices.

```python
# Initialize Polyscope
ps.set_verbosity(0)
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_ground_plane_mode("shadow_only")
ps.set_ground_plane_height_factor(0.5)
ps.set_program_name("Vertex Block Descent")
ps.init()

# Register meshes for visualization
vm = ps.register_volume_mesh("Simulation mesh", V, C)
vm.add_scalar_quantity("Coloring", hierarchy.root.colors, defined_on="vertices", cmap="jet")
pc = ps.register_point_cloud("Dirichlet", V[vdbc, :])
```

### 6. Interactive Simulation

A simulation loop with GUI controls allows users to adjust parameters, step through the simulation, and visualize results.

```python
# Simulation parameters
dt = 0.01
substeps = 1
RdetH = 1e-10
animate = False
export = False
t = 0

profiler = pbat.profiling.Profiler()

def callback():
    global dt, substeps, RdetH, animate, export, t, profiler

    # GUI controls
    changed, dt = imgui.InputFloat("dt", dt)
    changed, substeps = imgui.InputInt("Substeps", substeps)
    changed, animate = imgui.Checkbox("Animate", animate)
    changed, export = imgui.Checkbox("Export", export)
    step = imgui.Button("Step")
    reset = imgui.Button("Reset")

    # Simulation logic
    if animate or step:
        profiler.begin_frame("Physics")
        vbd.step(dt, substeps, hierarchy)
        profiler.end_frame("Physics")

        # Update visuals
        V = hierarchy.root.x.T
        if export:
            ps.screenshot(f"{args.output}/{t}.png")
        vm.update_vertex_positions(V)
        t += 1

    # Display frame information
    imgui.Text(f"Frame={t}")
    imgui.Text("Using CPU Multi-Scale VBD integrator")

ps.set_user_callback(callback)
ps.show()
```
