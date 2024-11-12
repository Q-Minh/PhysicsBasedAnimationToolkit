# VBD Elastic Simulation Using Linear FEM Tetrahedra

This script performs a **Vertex Block Descent (VBD)** elastic simulation using **linear Finite Element Method (FEM) tetrahedra**. It simulates the elastic behavior of 3D meshes under gravity, handling collisions and constraints to produce realistic deformations over time. The simulation can leverage GPU acceleration for enhanced performance.

## Prerequisites

Ensure the following Python libraries are installed before running the script:

- `pbatoolkit`
- `meshio`
- `polyscope`
- `numpy`
- `scipy`
- `libigl`
- `networkx`
- `argparse`
- `math`

You can install the required packages using `pip`:

```bash
pip install pbatoolkit meshio polyscope numpy scipy libigl networkx argparse math
```

*Note*: If `pbatoolkit` is not available via `pip`, refer to its official documentation for installation instructions.

## Command-line Arguments

The script accepts several command-line arguments to customize the simulation:

- `-i`, `--input`: **Paths to input mesh files** (required). Supports multiple mesh files.
- `-o`, `--output`: **Path to output directory** for saving results (default: current directory).
- `-m`, `--mass-density`: **Mass density** of the material (default: `1000.0`).
- `-Y`, `--young-modulus`: **Young's modulus** of the material (default: `1e6`).
- `-n`, `--poisson-ratio`: **Poisson's ratio** of the material (default: `0.45`).
- `-t`, `--translation`: **Translation multiplier** in the specified axis between input meshes (default: `0.1`).
- `--percent-fixed`: **Percentage of the mesh along the fixed axis to fix** (default: `0.01` or 1%).
- `--fixed-axis`: **Axis to fix** (`0` for x, `1` for y, `2` for z) (default: `2`).
- `--fixed-end`: **Which end to fix** (`min` or `max`) of the bounding box along the fixed axis (default: `min`).
- `--use-gpu`: **Flag to run GPU implementation** of VBD (default: `False`).

### Example Usage

```bash
python vbd_simulation.py -i mesh1.vtk mesh2.vtk -o output_directory -m 1200 -Y 2e6 -n 0.4 -t 0.05 --percent-fixed 0.02 --fixed-axis 2 --fixed-end max --use-gpu
```

- **Explanation**:
  - Loads `mesh1.vtk` and `mesh2.vtk`.
  - Outputs results to `output_directory`.
  - Sets mass density to `1200`.
  - Sets Young's modulus to `2e6`.
  - Sets Poisson's ratio to `0.4`.
  - Applies a translation multiplier of `0.05` between meshes along the z-axis.
  - Fixes `2%` of the mesh at the maximum end along the z-axis.
  - Utilizes the GPU implementation for the VBD integrator.

## Workflow Overview

### 1. Loading and Combining Meshes

Multiple input meshes are loaded and combined into a single FEM mesh. Each mesh's position is adjusted along the specified axis to prevent overlap based on the translation multiplier.

```python
# Load input meshes
imeshes = [meshio.read(input) for input in args.inputs]
V, C = [imesh.points / (imesh.points.max() - imesh.points.min()) for imesh in imeshes], [
    imesh.cells_dict["tetra"] for imesh in imeshes]

# Translate meshes along the specified axis
for i in range(len(V) - 1):
    extent = V[i][:, args.fixed_axis].max() - V[i][:, args.fixed_axis].min()
    offset = V[i][:, args.fixed_axis].max() - V[i+1][:, args.fixed_axis].min()
    V[i+1][:, args.fixed_axis] += offset + extent * args.translation

# Combine vertices and cells
V, Vsizes, C, Coffsets, Csizes = combine(V, C)
```

### 2. Constructing the FEM Mesh

A FEM mesh is created using the combined vertices and cells. Boundary facets are extracted for collision handling.

```python
# Create FEM mesh
mesh = pbat.fem.Mesh(
    V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=1)
F, Fsizes = boundary_triangles(C, Coffsets, Csizes)
```

### 3. Setting Up Mass Matrix and Load Vector

The mass matrix and load vector are computed based on the material properties and external forces like gravity.

```python
# Compute mass matrix
detJeM = pbat.fem.jacobian_determinants(mesh, quadrature_order=2)
rho = args.rho
M, detJeM = pbat.fem.mass_matrix(mesh, rho=rho, dims=1, lump=True)
m = np.array(M.diagonal()).squeeze()

# Construct load vector from gravity field
detJeU = pbat.fem.jacobian_determinants(mesh, quadrature_order=1)
GNeU = pbat.fem.shape_function_gradients(mesh, quadrature_order=1)
g = np.zeros(mesh.dims)
g[-1] = -9.81
f, detJeF = pbat.fem.load_vector(mesh, rho * g, detJe=detJeU, flatten=False)
a = f / m
```

### 4. Defining Material Properties

Material properties such as Young's modulus and Poisson's ratio are used to compute Lame parameters for the simulation.

```python
# Define material (Lame) constants
Y = np.full(mesh.E.shape[1], args.Y)
nu = np.full(mesh.E.shape[1], args.nu)
mue = Y / (2 * (1 + nu))
lambdae = (Y * nu) / ((1 + nu) * (1 - 2 * nu))
```

### 5. Applying Boundary Conditions

Dirichlet boundary conditions are applied to a specified percentage of the mesh along the chosen axis to fix certain vertices.

```python
# Set Dirichlet boundary conditions
Xmin = mesh.X.min(axis=1)
Xmax = mesh.X.max(axis=1)
extent = Xmax - Xmin

if args.fixed_end == "min":
    Xmax[args.fixed_axis] = Xmin[args.fixed_axis] + \
        args.percent_fixed * extent[args.fixed_axis]
    Xmin[args.fixed_axis] -= args.percent_fixed * extent[args.fixed_axis]
elif args.fixed_end == "max":
    Xmin[args.fixed_axis] = Xmax[args.fixed_axis] - \
        args.percent_fixed * extent[args.fixed_axis]
    Xmax[args.fixed_axis] += args.percent_fixed * extent[args.fixed_axis]

aabb = pbat.geometry.aabb(np.vstack((Xmin, Xmax)).T)
vdbc = aabb.contained(mesh.X)
```

### 6. Constructing the Vertex-Tetrahedron Adjacency Graph

An adjacency graph is constructed to manage the relationships between vertices and tetrahedrons, which is essential for constraint partitioning.

```python
# Construct vertex-tetrahedron adjacency graph
GVT = vertex_tetrahedron_adjacency_graph(V, C)
```

### 7. Partitioning Vertices for Optimization

Constraints are partitioned to optimize the simulation process, ensuring that internal forces are not applied to constrained vertices.

```python
# Partition vertices for optimization
GVTtopology = GVT.copy()
GVTtopology.data[:] = 1  # Set all edge weights to 1
partitions, GC = partition_vertices(GVTtopology, vdbc)
```

### 8. Setting Up the VBD Solver

The VBD solver is set up with the mesh, constraints, and material properties. It can utilize GPU acceleration if specified.

```python
# Setup VBD
VC = np.unique(F)
data = pbat.sim.vbd.Data().with_volume_mesh(
    V.T, C.T
).with_surface_mesh(
    VC, F.T
).with_acceleration(
    a
).with_mass(
    m
).with_quadrature(
    detJeU[0, :] / 6, GNeU, np.vstack((mue, lambdae))
).with_vertex_adjacency(
    GVT.indptr, GVT.indices, GVT.indices, GVT.data
).with_partitions(
    partitions
).with_dirichlet_vertices(
    vdbc
).with_initialization_strategy(
    pbat.sim.vbd.InitializationStrategy.KineticEnergyMinimum
).construct(validate=False)

# Initialize the integrator
thread_block_size = 64
if args.gpu:
    vbd = pbat.gpu.vbd.Integrator(data)
    vbd.set_gpu_block_size(thread_block_size)
else:
    vbd = pbat.sim.vbd.Integrator(data)

# Set initialization strategy
initialization_strategies = [
    pbat.sim.vbd.InitializationStrategy.Position,
    pbat.sim.vbd.InitializationStrategy.Inertia,
    pbat.sim.vbd.InitializationStrategy.KineticEnergyMinimum,
    pbat.sim.vbd.InitializationStrategy.AdaptiveVbd,
    pbat.sim.vbd.InitializationStrategy.AdaptivePbat
]
initialization_strategy = initialization_strategies[2]  # KineticEnergyMinimum
vbd.strategy = initialization_strategy
```

### 9. Setting Up Visualization with Polyscope

Polyscope is initialized for real-time visualization of the simulation mesh and fixed vertices.

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
vm.add_scalar_quantity("Coloring", GC, defined_on="vertices", cmap="jet")
pc = ps.register_point_cloud("Dirichlet", V[vdbc, :])
```

### 10. Running the Simulation Loop with User Interaction

A simulation loop is created with interactive controls for adjusting parameters, stepping through the simulation, and resetting.

```python
# Simulation parameters
dt = 0.01
iterations = 20
substeps = 1
rho_chebyshev = 1.0
RdetH = 1e-10
kD = 0.0
animate = False
export = False
t = 0

profiler = pbat.profiling.Profiler()

def callback():
    global dt, iterations, substeps, rho_chebyshev, thread_block_size, initialization_strategy, RdetH, kD
    global animate, export, t
    global profiler

    # GUI controls
    changed, dt = imgui.InputFloat("dt", dt)
    changed, iterations = imgui.InputInt("Iterations", iterations)
    changed, substeps = imgui.InputInt("Substeps", substeps)
    changed, rho_chebyshev = imgui.InputFloat(
        "Chebyshev rho", rho_chebyshev)
    changed, kD = imgui.InputFloat(
        "Damping", kD, format="%.8f")
    changed, RdetH = imgui.InputFloat(
        "Residual det(H)", RdetH, format="%.15f")
    changed, thread_block_size = imgui.InputInt(
        "Thread block size", thread_block_size)
    
    # Initialization strategy selection
    changed = imgui.BeginCombo(
        "Initialization strategy", str(initialization_strategy).split('.')[-1])
    if changed:
        for i in range(len(initialization_strategies)):
            _, selected = imgui.Selectable(
                str(initialization_strategies[i]).split('.')[-1], initialization_strategy == initialization_strategies[i])
            if selected:
                initialization_strategy = initialization_strategies[i]
        imgui.EndCombo()
    vbd.strategy = initialization_strategy
    vbd.kD = kD
    vbd.detH_residual = RdetH

    # Animation and export controls
    changed, animate = imgui.Checkbox("Animate", animate)
    changed, export = imgui.Checkbox("Export", export)
    step = imgui.Button("Step")
    reset = imgui.Button("Reset")

    # Reset simulation
    if reset:
        vbd.x = mesh.X
        vbd.v = np.zeros(mesh.X.shape)
        vm.update_vertex_positions(mesh.X)
        t = 0

    # Update GPU block size if using GPU
    if args.gpu:
        vbd.set_gpu_block_size(thread_block_size)

    # Advance simulation
    if animate or step:
        profiler.begin_frame("Physics")
        vbd.step(dt, iterations, substeps, rho_chebyshev)
        profiler.end_frame("Physics")

        # Update visualization
        V = vbd.x.T
        if export:
            ps.screenshot(f"{args.output}/{t}.png")
            # Uncomment below to export mesh
            # omesh = meshio.Mesh(V, {"tetra": mesh.E.T})
            # meshio.write(f"{args.output}/{t}.mesh", omesh)

        vm.update_vertex_positions(V)
        t += 1

    # Display frame information
    imgui.Text(f"Frame={t}")
    if args.gpu:
        imgui.Text("Using GPU VBD integrator")
    else:
        imgui.Text("Using CPU VBD integrator")

# Register callback and start visualization
ps.set_user_callback(callback)
ps.show()
```