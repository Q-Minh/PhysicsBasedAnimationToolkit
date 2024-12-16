# Linear FEM Shape Transfer

This script performs **Linear Finite Element Method (FEM) shape transfer** between fine and coarse tetrahedral meshes. It computes dynamic deformations using restriction operators and wave-based signals, enabling efficient transfer of shape changes from a high-resolution mesh to a coarser representation.

## Prerequisites

Ensure the following Python libraries are installed before running the script:

- `pbatoolkit`
- `numpy`
- `scipy`
- `polyscope`
- `meshio`
- `argparse`

You can install the required packages using `pip`:

```bash
pip install pbatoolkit numpy scipy polyscope meshio argparse
```

*Note*: If `pbatoolkit` is not available via `pip`, refer to its official documentation for installation instructions.

## Command-line Arguments

The script accepts several command-line arguments:

- `-i`, `--input`: **Path to input tetrahedral mesh** (required).
- `-c`, `--cage`: **Path to cage tetrahedral mesh** (required).
- `-m`, `--mass-density`: **Mass density** (default: `1000.0`).
- `-Y`, `--young-modulus`: **Young's modulus** (default: `1e6`).
- `-n`, `--poisson-ratio`: **Poisson's ratio** (default: `0.45`).
- `-k`, `--num-modes`: **Number of modes to compute** (default: `30`).

### Example Usage

```bash
python linear_fem_shape_transfer.py -i fine_mesh.vtk -c coarse_mesh.vtk -m 1200 -Y 2e6 -n 0.4 -k 20
```

- **Explanation**:
  - Loads `fine_mesh.vtk` as the high-resolution mesh.
  - Loads `coarse_mesh.vtk` as the coarse representation.
  - Sets mass density to `1200`.
  - Sets Young's modulus to `2e6`.
  - Sets Poisson's ratio to `0.4`.
  - Computes 20 deformation modes.

## Workflow Overview

### 1. Loading and Preparing Meshes

The fine and coarse tetrahedral meshes are loaded and converted into FEM meshes.

```python
# Load input meshes
imesh, icmesh = meshio.read(args.input), meshio.read(args.cage)
V, C = imesh.points.astype(np.float64, order='c'), imesh.cells_dict["tetra"].astype(np.int64, order='c')
CV, CC = icmesh.points.astype(np.float64, order='c'), icmesh.cells_dict["tetra"].astype(np.int64, order='c')

# Construct FEM meshes
mesh = pbat.fem.Mesh(V.T, C.T, element=pbat.fem.Element.Tetrahedron)
cmesh = pbat.fem.Mesh(CV.T, CC.T, element=pbat.fem.Element.Tetrahedron)
```

### 2. Precomputing Quantities

The eigenmodes of the rest pose are computed on the fine mesh.

```python
# Compute eigenmodes
w, L = pbat.fem.rest_pose_hyper_elastic_modes(
    mesh, rho=args.rho, Y=args.Y, nu=args.nu, modes=args.modes)
```

### 3. Constructing the Restriction Operator

A restriction operator is defined to transfer shape deformations between the fine and coarse meshes using predefined quadrature parameters.

```python
# Define quadrature parameters
cage_quad_params = pbat.sim.vbd.lod.CageQuadratureParameters(
).with_strategy(
    pbat.sim.vbd.lod.CageQuadratureStrategy.PolynomialSubCellIntegration
).with_cage_mesh_pts(4
).with_patch_cell_pts(4
).with_patch_error(1e-4)

# Initialize restriction operator
Fvbd = VbdRestrictionOperator(
    mesh,
    cmesh,
    cage_quad_params=cage_quad_params,
    iters=10
)
```

### 4. Visualization Setup

Polyscope is used to visualize the meshes and their deformations in real time.

```python
# Setup Polyscope
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_ground_plane_mode("shadow_only")
ps.init()

vm = ps.register_surface_mesh("model", mesh.X.T, F.T)
vbdvm = ps.register_surface_mesh("VBD cage", cmesh.X.T, CF.T)
vbdvm.set_transparency(0.5)
vbdvm.set_edge_width(1)
```

### 5. Dynamic Deformation and Interaction

A sinusoidal signal is applied to generate dynamic deformations on the fine mesh, which are then transferred to the coarse mesh.

```python
# Signal function
def signal(w: float, v: np.ndarray, t: float, c: float, k: float):
    u = c * np.sin(k * w * t) * v
    return u

# Simulation parameters
mode = 6
t0 = time.time()
c = 3.0
k = 0.1
theta = 0
dtheta = np.pi / 120
animate = False
step = False
screenshot = False

def callback():
    global mode, c, k, theta, dtheta
    global animate, step, screenshot

    # GUI controls
    changed, mode = imgui.InputInt("Mode", mode)
    changed, c = imgui.InputFloat("Wave amplitude", c)
    changed, k = imgui.InputFloat("Wave frequency", k)
    changed, animate = imgui.Checkbox("animate", animate)
    changed, screenshot = imgui.Checkbox("screenshot", screenshot)
    step = imgui.Button("step")

    if animate or step:
        t = time.time() - t0

        R = sp.spatial.transform.Rotation.from_quat(
            [0, np.sin(theta / 2), 0, np.cos(theta / 4)]).as_matrix()
        X = (V - V.mean(axis=0)) @ R.T + V.mean(axis=0)
        uf = signal(w[mode], L[:, mode], t, c, k)
        ur = (X - V).flatten(order="C")
        ut = 1e-1 * np.ones(math.prod(X.shape))
        u = ut + ur + uf
        XCvbd = CV + (Fvbd @ u).reshape(CV.shape)

        vm.update_vertex_positions(V + u.reshape(V.shape))
        vbdvm.update_vertex_positions(XCvbd)

        theta += dtheta
        if theta > 2 * np.pi:
            theta = 0

    if screenshot:
        ps.screenshot()

ps.set_user_callback(callback)
ps.show()
```
