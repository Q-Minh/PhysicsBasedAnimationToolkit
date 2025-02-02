# 3D Elastic Simulation of Linear FEM Tetrahedra Using Incremental Potential Contact

This script performs a **3D elastic simulation** based on the **linear Finite Element Method (FEM) tetrahedra**. It integrates **Incremental Potential Contact (IPC)** for handling collisions and friction constraints, enabling realistic deformation of complex 3D meshes under external forces. The simulation leverages **Newton-Raphson** methods for solving nonlinear systems and provides real-time visualization using **Polyscope**.

## Prerequisites

Ensure the following Python libraries are installed before running the script:

- `pbatoolkit`
- `igl`
- `ipctk`
- `meshio`
- `polyscope`
- `numpy`
- `scipy`
- `argparse`
- `itertools`

You can install the required packages using `pip`:

```bash
pip install pbatoolkit igl ipctk meshio polyscope numpy scipy argparse
```

*Note*: If `pbatoolkit` or `ipctk` are not available via `pip`, refer to their official documentation for installation instructions.

## Command-line Arguments

The script accepts several command-line arguments to customize the simulation:

- `-i`, `--input`: **Path to input mesh file** (required). The mesh should contain tetrahedral elements.
- `-t`, `--translation`: **Vertical translation** applied to copies of the input mesh along the z-axis (default: `0.1`).
- `--percent-fixed`: **Percentage of the input mesh's bottom** to fix as Dirichlet boundary conditions (default: `0.1` or 10%).
- `-m`, `--mass-density`: **Mass density** of the material (default: `1000.0`).
- `-Y`, `--young-modulus`: **Young's modulus** of the material (default: `1e6`).
- `-n`, `--poisson-ratio`: **Poisson's ratio** of the material (default: `0.45`).
- `-c`, `--copy`: **Number of copies** of the input model to create and translate (default: `1`).

### Example Usage

```bash
python elastic_simulation_ipc.py -i input_mesh.vtk -t 0.2 --percent-fixed 0.15 -m 1200 -Y 2e6 -n 0.4 -c 2
```

- **Explanation**:
  - Loads `input_mesh.vtk` as the primary tetrahedral mesh.
  - Creates 2 additional copies of the mesh, each translated vertically by `0.2` units.
  - Fixes `15%` of the mesh's bottom along the z-axis as Dirichlet boundary conditions.
  - Sets mass density to `1200`.
  - Sets Young's modulus to `2e6`.
  - Sets Poisson's ratio to `0.4`.

## Workflow Overview

### 1. Loading and Combining Meshes

The script begins by loading the input mesh using `meshio`. It supports creating multiple copies of the mesh, each translated vertically to prevent overlap. The `combine` function merges multiple meshes into a single FEM mesh for simulation.

```python
# Load input meshes and combine them into one mesh
V, C = [], []
imesh = meshio.read(args.input)
V1 = imesh.points.astype(np.float64, order='C')
C1 = imesh.cells_dict["tetra"].astype(np.int64, order='C')
V.append(V1)
C.append(C1)
for c in range(args.ncopy):
    R = sp.spatial.transform.Rotation.from_quat(
        [0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]).as_matrix()
    V2 = (V[-1] - V[-1].mean(axis=0)) @ R.T + V[-1].mean(axis=0)
    V2[:, 2] += (V2[:, 2].max() - V2[:, 2].min()) + args.translation
    C2 = C[-1]
    V.append(V2)
    C.append(C2)

V, C = combine(V, C)
mesh = pbat.fem.Mesh(
    V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=1)
```

### 2. Constructing FEM Quantities for Simulation

The script initializes the FEM mesh, computes the lumped mass matrix, and constructs the load vector based on gravity. These are fundamental for simulating the dynamic behavior of the mesh.

```python
# Initialize FEM quantities
x = mesh.X.reshape(math.prod(mesh.X.shape), order='F')
n = x.shape[0]
v = np.zeros(n)

# Lumped mass matrix
rho = args.rho
M, detJeM = pbat.fem.mass_matrix(mesh, rho=rho, lump=True)
Minv = sp.sparse.diags(1./M.diagonal())

# Construct load vector from gravity
g = np.zeros(mesh.dims)
g[-1] = -9.81
f, detJeF = pbat.fem.load_vector(mesh, rho*g)
a = Minv @ f
```

### 3. Defining Material Properties

Material properties, including Young's modulus and Poisson's ratio, are defined to create the hyperelastic potential, which models the elastic behavior of the material.

```python
# Create hyperelastic potential
Y, nu, psi = args.Y, args.nu, pbat.fem.HyperElasticEnergy.StableNeoHookean
hep, egU, wgU, GNeU = pbat.fem.hyper_elastic_potential(
    mesh, Y=Y, nu=nu, energy=psi)
```

### 4. Setting Up IPC (Incremental Potential Contact) Constraints

IPC is utilized for handling collisions and friction between different parts of the mesh or with external objects. The script sets up the collision mesh, defines contact and friction constraints, and applies Dirichlet boundary conditions to fix a portion of the mesh.

```python
# Setup IPC contact handling
F = igl.boundary_facets(C)
F[:, :2] = np.roll(F[:, :2], shift=1, axis=1)
E = ipctk.edges(F)
cmesh = ipctk.CollisionMesh.build_from_full_mesh(V, E, F)
dhat = 1e-3
cconstraints = ipctk.CollisionConstraints()
fconstraints = ipctk.FrictionConstraints()
mu = 0.3
epsv = 1e-4
dmin = 1e-4

# Fix a percentage of the bottom of the input models as Dirichlet boundary conditions
Xmin = mesh.X.min(axis=1)
Xmax = mesh.X.max(axis=1)
dX = Xmax - Xmin
Xmax[-1] = Xmin[-1] + args.percent_fixed * dX[-1]
Xmin[-1] = Xmin[-1] - 1e-4
aabb = pbat.geometry.aabb(np.vstack((Xmin, Xmax)).T)
vdbc = aabb.contained(mesh.X)
dbcs = np.array(vdbc)[:, np.newaxis]
dbcs = np.repeat(dbcs, mesh.dims, axis=1)
for d in range(mesh.dims):
    dbcs[:, d] = mesh.dims * dbcs[:, d] + d
dbcs = dbcs.reshape(math.prod(dbcs.shape))
dofs = np.setdiff1d(list(range(n)), dbcs)
```

### 5. Setting Up the Simulation Loop and Solver

The simulation loop utilizes a **Newton-Raphson** method to solve the nonlinear system arising from the FEM formulation. It integrates the incremental potential contact constraints and updates the mesh's position and velocity over time. A custom callback function manages user interactions and updates the visualization accordingly.

```python
# Setup GUI
ps.set_verbosity(0)
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_ground_plane_mode("shadow_only")
ps.set_ground_plane_height_factor(0.5)
ps.set_program_name("Incremental Potential Contact")
ps.init()
vm = ps.register_surface_mesh(
    "Visual mesh", cmesh.rest_positions, cmesh.faces)
pc = ps.register_point_cloud("Dirichlet", mesh.X[:, vdbc].T)
dt = 0.01
animate = False
newton_maxiter = 10
newton_rtol = 1e-5

profiler = pbat.profiling.Profiler()
# ipctk.set_logger_level(ipctk.LoggerLevel.trace)

def callback():
    global x, v, dt
    global dhat, dmin, mu
    global newton_maxiter, newton_rtol
    global animate, step

    # GUI controls
    changed, dt = imgui.InputFloat("dt", dt)
    changed, dhat = imgui.InputFloat(
        "IPC activation distance", dhat, format="%.6f")
    changed, dmin = imgui.InputFloat(
        "IPC minimum distance", dmin, format="%.6f")
    changed, mu = imgui.InputFloat(
        "Coulomb friction coeff", mu, format="%.2f")
    changed, newton_maxiter = imgui.InputInt(
        "Newton max iterations", newton_maxiter)
    changed, newton_rtol = imgui.InputFloat(
        "Newton convergence residual", newton_rtol, format="%.8f")
    changed, animate = imgui.Checkbox("animate", animate)
    step = imgui.Button("step")

    if animate or step:
        ps.screenshot()
        profiler.begin_frame("Physics")
        params = Parameters(mesh, x, v, a, M, hep, dt, cmesh,
                            cconstraints, fconstraints, dhat, dmin, mu, epsv)
        f = Potential(params)
        g = Gradient(params)
        H = Hessian(params)
        solver = LinearSolver(dofs)
        ccd = CCD(params)
        updater = BarrierUpdater(params)
        xtp1 = newton(x, f, g, H, solver, ccd,
                      newton_maxiter, newton_rtol, updater)
        v = (xtp1 - x) / dt
        x = xtp1
        BX = to_surface(x, mesh, cmesh)
        profiler.end_frame("Physics")

        # Update visuals
        vm.update_vertex_positions(BX)

if __name__ == "__main__":
    ps.set_user_callback(callback)
    ps.show()
```

### 6. Visualization with Polyscope

**Polyscope** is used for real-time visualization of the simulation mesh and the applied constraints. The mesh is displayed along with the fixed vertices, allowing users to observe the deformation and interaction dynamics as the simulation progresses.

```python
# Setup GUI
ps.set_verbosity(0)
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_ground_plane_mode("shadow_only")
ps.set_ground_plane_height_factor(0.5)
ps.set_program_name("Incremental Potential Contact")
ps.init()
vm = ps.register_surface_mesh(
    "Visual mesh", cmesh.rest_positions, cmesh.faces)
pc = ps.register_point_cloud("Dirichlet", mesh.X[:, vdbc].T)
```

## Usage Instructions

1. **Prepare the Input Mesh**: Ensure your input mesh file contains tetrahedral elements and is formatted correctly (e.g., `.vtk` format).

2. **Run the Simulation**: Execute the script with the desired command-line arguments. For example:

    ```bash
    python elastic_simulation_ipc.py -i input_mesh.vtk -t 0.2 --percent-fixed 0.15 -m 1200 -Y 2e6 -n 0.4 -c 2
    ```

3. **Interact with the Simulation**:
    - **Adjust Parameters**: Use the GUI controls to modify simulation parameters such as time step (`dt`), IPC activation distance (`dhat`), IPC minimum distance (`dmin`), Coulomb friction coefficient (`mu`), Newton-Raphson solver settings (`newton_maxiter`, `newton_rtol`), and toggle animation.
    - **Control Simulation**: Click the "step" button to advance the simulation by one timestep or enable animation to run the simulation continuously.
    - **Export Results**: Enable the export option to save screenshots of the simulation at each timestep.

4. **Visualize the Results**: The Polyscope window will display the simulation mesh along with the fixed vertices. Observe the deformation and interaction dynamics in real-time as the simulation progresses.

---

*For further customization and advanced features, refer to the `pbatoolkit`, `ipctk`, and `polyscope` documentation.*