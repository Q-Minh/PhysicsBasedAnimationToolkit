# Simple 3D Elastic Simulation Using Linear FEM Tetrahedra

This script performs a **3D elastic simulation** based on the **linear Finite Element Method (FEM) tetrahedra**. It simulates the elastic behavior of a 3D mesh under gravity, solving for the deformation of the mesh over time using a Newton-Raphson method. The simulation provides real-time visualization and interactive controls for adjusting simulation parameters.

## Prerequisites

Ensure the following Python libraries are installed before running the script:

- `pbatoolkit`
- `ilupp`
- `meshio`
- `polyscope`
- `numpy`
- `scipy`
- `argparse`
- `math`

You can install the required packages using `pip`:

```bash
pip install pbatoolkit ilupp meshio polyscope numpy scipy argparse math
```

*Note*: If `pbatoolkit` is not available via `pip`, refer to its official documentation for installation instructions.

## Command-line Arguments

The script accepts several command-line arguments to customize the simulation:

- `-i`, `--input`: **Path to input mesh file** (required). The mesh should contain tetrahedral elements.
- `-o`, `--output`: **Path to output directory** for saving results (default: current directory).
- `-m`, `--mass-density`: **Mass density** of the material (default: `1000.0`).
- `-Y`, `--young-modulus`: **Young's modulus** of the material (default: `1e6`).
- `-n`, `--poisson-ratio`: **Poisson's ratio** of the material (default: `0.45`).

### Example Usage

```bash
python simple_elastic_simulation.py -i input_mesh.vtk -o output_directory -m 1200 -Y 2e6 -n 0.4
```

- **Explanation**:
  - Loads `input_mesh.vtk`.
  - Outputs results to `output_directory`.
  - Sets mass density to `1200`.
  - Sets Young's modulus to `2e6`.
  - Sets Poisson's ratio to `0.4`.

## Workflow Overview

### 1. Loading and Combining Meshes

The script begins by loading the input mesh using `meshio`. It normalizes the vertex positions and ensures that the mesh is properly scaled for simulation.

```python
# Load input mesh
imesh = meshio.read(args.input)
V, C = imesh.points, imesh.cells_dict["tetra"]
```

### 2. Constructing the FEM Mesh

A FEM mesh is created using the loaded vertices and tetrahedral cells. This mesh serves as the foundation for the simulation.

```python
# Create FEM mesh
mesh = pbat.fem.Mesh(
    V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=1)
```

### 3. Setting Up Mass Matrix and Load Vector

The mass matrix is computed based on the material's mass density. The load vector is constructed to account for external forces, such as gravity.

```python
# Mass matrix
rho = args.rho
M, detJeM = pbat.fem.mass_matrix(mesh, rho=rho)
Minv = pbat.math.linalg.ldlt(M)
Minv.compute(M)

# Construct load vector from gravity field
g = np.zeros(mesh.dims)
g[-1] = -9.81
fe = rho * g
f, detJeF = pbat.fem.load_vector(mesh, fe)
a = Minv.solve(f).squeeze()
```

### 4. Defining Material Properties

Material properties, including Young's modulus and Poisson's ratio, are used to define the hyperelastic potential of the material.

```python
# Create hyper elastic potential
Y, nu, energy = args.Y, args.nu, pbat.fem.HyperElasticEnergy.StableNeoHookean
hep, egU, wgU, GNeU = pbat.fem.hyper_elastic_potential(
    mesh, Y=Y, nu=nu, energy=energy)
```

### 5. Applying Boundary Conditions

Dirichlet boundary conditions are applied to a specified portion of the mesh to fix certain vertices, preventing them from moving during the simulation.

```python
# Set Dirichlet boundary conditions
Xmin = mesh.X.min(axis=1)
Xmax = mesh.X.max(axis=1)
Xmax[0] = Xmin[0] + 1e-4
Xmin[0] = Xmin[0] - 1e-4
aabb = pbat.geometry.aabb(np.vstack((Xmin, Xmax)).T)
vdbc = aabb.contained(mesh.X)
```

### 6. Setting Up Linear Solver

A linear solver is configured to solve the system of equations arising from the FEM formulation. The solver can utilize either a direct method or the Conjugate Gradient (CG) method with preconditioning.

```python
# Setup linear solver
Hdd = hep.to_matrix()[:, dofs].tocsr()[dofs, :]
Mdd = M[:, dofs].tocsr()[dofs, :]
Addinv = pbat.math.linalg.ldlt(
    Hdd, solver=pbat.math.linalg.SolverBackend.Eigen)
Addinv.analyze(Hdd)
```

### 7. Setting Up Visualization with Polyscope

Polyscope is initialized for real-time visualization of the simulation mesh and the fixed vertices. It provides an interactive interface to observe the deformation of the mesh over time.

```python
# Initialize Polyscope
ps.set_verbosity(0)
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_ground_plane_mode("shadow_only")
ps.set_ground_plane_height_factor(0.5)
ps.set_program_name("Elasticity")
ps.init()
vm = ps.register_volume_mesh("world model", mesh.X.T, mesh.E.T)
pc = ps.register_point_cloud("Dirichlet", mesh.X[vdbc, :].T)
```

### 8. Running the Simulation Loop with User Interaction

The simulation loop is managed through a callback function that handles user interactions via the Polyscope GUI. Users can adjust simulation parameters, step through the simulation, reset the simulation, and export snapshots.

```python
# Simulation parameters
dt = 0.01
animate = False
use_direct_solver = False
export = False
t = 0
newton_maxiter = 1
cg_fill_in = 0.01
cg_drop_tolerance = 1e-4
cg_residual = 1e-5
cg_maxiter = 100
dx = np.zeros(n)

profiler = pbat.profiling.Profiler()

def callback():
    global x, v, dx, hep, dt, M, Minv, f
    global cg_fill_in, cg_drop_tolerance, cg_residual, cg_maxiter
    global animate, step, use_direct_solver, export, t
    global newton_maxiter
    global profiler

    # GUI controls
    changed, dt = imgui.InputFloat("dt", dt)
    changed, newton_maxiter = imgui.InputInt(
        "Newton max iterations", newton_maxiter)
    changed, cg_fill_in = imgui.InputFloat(
        "IC column fill in", cg_fill_in, format="%.4f")
    changed, cg_drop_tolerance = imgui.InputFloat(
        "IC drop tolerance", cg_drop_tolerance, format="%.8f")
    changed, cg_residual = imgui.InputFloat(
        "PCG residual", cg_residual, format="%.8f")
    changed, cg_maxiter = imgui.InputInt(
        "PCG max iterations", cg_maxiter)
    changed, animate = imgui.Checkbox("animate", animate)
    changed, use_direct_solver = imgui.Checkbox(
        "Use direct solver", use_direct_solver)
    changed, export = imgui.Checkbox("Export", export)
    step = imgui.Button("step")

    if animate or step:
        profiler.begin_frame("Physics")
        # Newton solve
        dt2 = dt**2
        xtilde = x + dt * v + dt2 * a
        xk = x
        for k in range(newton_maxiter):
            hep.compute_element_elasticity(xk, grad=True, hessian=True)
            gradU, HU = hep.gradient(), hep.hessian()

            global bd, Add

            def setup():
                global bd, Add
                A = M + dt2 * HU
                b = -(M @ (xk - xtilde) + dt2 * gradU)
                Add = A.tocsc()[:, dofs].tocsr()[dofs, :]
                bd = b[dofs]

            profiler.profile("Setup Linear System", setup)

            if k > 0:
                gradnorm = np.linalg.norm(bd, 1)
                if gradnorm < 1e-3:
                    break

            def solve():
                global dx, Add, bd
                global cg_fill_in, cg_drop_tolerance, cg_maxiter, cg_residual
                global use_direct_solver
                if use_direct_solver:
                    Addinv.factorize(Add)
                    dx[dofs] = Addinv.solve(bd).squeeze()
                else:
                    P = ilupp.ICholTPreconditioner(
                        Add, add_fill_in=int(Add.shape[0] * cg_fill_in), threshold=cg_drop_tolerance)
                    dx[dofs], cginfo = sp.sparse.linalg.cg(
                        Add, bd, rtol=cg_residual, maxiter=cg_maxiter, M=P)

            profiler.profile("Solve Linear System", solve)
            xk = xk + dx

        v = (xk - x) / dt
        x = xk
        profiler.end_frame("Physics")

        if export:
            ps.screenshot(f"{args.output}/{t}.png")

        # Update visuals
        X = x.reshape(mesh.X.shape[0], mesh.X.shape[1], order='f')
        vm.update_vertex_positions(X.T)

        t = t + 1

    # Display frame information
    imgui.Text(f"Frame={t}")
```

## Usage Instructions

1. **Prepare the Input Mesh**: Ensure your input mesh file contains tetrahedral elements and is formatted correctly (e.g., `.vtk` format).

2. **Run the Simulation**: Execute the script with the desired command-line arguments. For example:

    ```bash
    python simple_elastic_simulation.py -i input_mesh.vtk -o output_directory -m 1200 -Y 2e6 -n 0.4
    ```

3. **Interact with the Simulation**:
    - **Adjust Parameters**: Use the GUI controls to modify simulation parameters such as time step (`dt`), Newton iterations, CG solver settings, and solver type (direct or iterative).
    - **Control Animation**: Toggle the animation or step through the simulation frame by frame.
    - **Export Results**: Enable the export option to save screenshots of the simulation at each timestep.

4. **Visualize the Results**: The Polyscope window will display the simulation mesh and the fixed vertices. Observe the deformation of the mesh in real-time as the simulation progresses.