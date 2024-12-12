# Heat Geodesics Demo

This script demonstrates the computation of **heat geodesics** on a 3D mesh using the **Finite Element Method (FEM)**. Heat geodesics are useful for various applications in geometry processing, including shape analysis, segmentation, and visualization. The simulation utilizes heat diffusion and the computation of geodesic distances to generate contour lines representing equal geodesic distances from a source point.

## Prerequisites

Ensure the following Python libraries are installed before running the script:

- `pbatoolkit`
- `igl`
- `polyscope`
- `numpy`
- `scipy`
- `argparse`
- `meshio`

You can install the required packages using `pip`:

```bash
pip install pbatoolkit igl polyscope numpy scipy argparse meshio
```

*Note*: If `pbatoolkit` is not available via `pip`, refer to its official documentation for installation instructions.

## Command-line Arguments

The script accepts several command-line arguments to customize the simulation:

- `-i`, `--input`: **Path to input mesh file** (required). Supports both tetrahedral and triangle meshes.
- `-o`, `--output`: **Path to output directory** for saving results (default: current directory).

### Example Usage

```bash
python heat_geodesics_demo.py -i input_mesh.vtk -o output_directory
```

- **Explanation**:
  - Loads `input_mesh.vtk`.
  - Outputs results to `output_directory`.

## Workflow Overview

### 1. Loading the Input Mesh

The script begins by loading the input mesh using `meshio`. It supports both tetrahedral (`"tetra"`) and triangle (`"triangle"`) meshes. Depending on the mesh type, it constructs the appropriate FEM mesh.

```python
# Load input mesh
imesh = meshio.read(args.input)
mesh = None
if "tetra" in imesh.cells_dict.keys():
    V, C = imesh.points, imesh.cells_dict["tetra"]
    mesh = pbat.fem.Mesh(
        V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=1)
if "triangle" in imesh.cells_dict.keys():
    V, C = imesh.points, imesh.cells_dict["triangle"]
    mesh = pbat.fem.Mesh(
        V.T, C.T, element=pbat.fem.Element.Triangle, order=1)
```

### 2. Extracting Boundary Facets

For tetrahedral meshes, boundary facets are extracted to identify the surface of the mesh. This is essential for visualizing the geodesic contours.

```python
F = C
if mesh.element == pbat.fem.Element.Tetrahedron:
    F = igl.boundary_facets(C)
    F[:, :2] = np.roll(F[:, :2], shift=1, axis=1)
```

### 3. Constructing FEM Quantities

The mass matrix and load vector are computed based on the mesh and material properties. These matrices are fundamental for simulating heat diffusion and solving the associated linear systems.

```python
# Construct Galerkin laplacian, mass and gradient operators
n = V.shape[0]
M, detJeM = pbat.fem.mass_matrix(mesh, dims=1)
G, egG, GNegG = pbat.fem.gradient(mesh)
wgD = pbat.fem.inner_product_weights(mesh)
D, wgD, egD, GNegD = pbat.fem.divergence(mesh, eg=GNegG, wg=wgD, GNeg=GNegG)
L = D @ G
```

### 4. Setting Up Heat Diffusion Parameters

Heat diffusion is simulated by solving the heat equation. The script sets up the necessary parameters, including the time step (`dt`), diffusion coefficient (`k`), and the linear system to be solved at each step.

```python
# Setup 1-step heat diffusion
h = igl.avg_edge_length(V, C)
dt = h**2
k = 2
A = M - k*dt*L
```

### 5. Precomputing Linear Solvers

To efficiently solve the linear systems arising from the heat diffusion simulation, the script precomputes linear solvers using LDLT decomposition. This optimization accelerates the simulation, especially for large meshes.

```python
# Precompute linear solvers
Ainv = pbat.math.linalg.ldlt(A)
Ainv.compute(A)
Linv = pbat.math.linalg.ldlt(L)
Linv.compute(L)
```

### 6. Setting Up Visualization with Polyscope

Polyscope is initialized for real-time visualization of the mesh and the computed heat geodesics. The mesh is registered for visualization, and scalar and vector quantities are added to enhance the visual output.

```python
# Setup isoline visuals
niso = 10

ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_ground_plane_mode("shadow_only")
ps.init()

if mesh.element == pbat.fem.Element.Tetrahedron:
    ps.register_volume_mesh("model", V, C)
if mesh.element == pbat.fem.Element.Triangle:
    ps.register_surface_mesh("model", V, C)
```

### 7. Running the Simulation Loop with User Interaction

The simulation loop is managed through a callback function that handles user interactions via the Polyscope GUI. Users can adjust parameters such as the diffusion coefficient (`k`), the number of isocontours (`niso`), and the source point (`gamma`). Upon triggering the "Compute" button, the script performs heat diffusion, computes geodesic distances, and generates contour lines representing the geodesics.

```python
def callback():
    global k, dt, Ainv, Linv, G, M, L, gamma, niso
    kchanged, k = imgui.InputFloat("k", k)
    if kchanged:
        A = M - k*dt*L
        Ainv.factorize(A)

    _, niso = imgui.InputInt("# iso", niso)
    _, gamma[0] = imgui.InputInt("source", gamma[0])
    if imgui.Button("Compute"):
        # Compute heat and its gradient
        u0 = np.zeros(n)
        u0[gamma] = 1
        b = M @ u0
        u = Ainv.solve(b).squeeze()
        gradu = (G @ u).reshape(int(G.shape[0]/3), 3)
        # Stable normalize gradient
        gradnorm = sp.linalg.norm(gradu, axis=1, keepdims=True)
        gnnz = gradnorm[:, 0] > 0
        gradu[gnnz, :] = gradu[gnnz, :] / gradnorm[gnnz, :]
        # Solve Poisson problem to reconstruct geodesic distance field, knowing that phi[0] = 0
        divGu = D @ gradu.reshape(G.shape[0])
        phi = Linv.solve(divGu).squeeze()
        # Handle reflection and shifting
        if phi[gamma].mean() > phi.mean():
            phi = -phi
        phi -= phi.min()

        # Compute isocontours
        diso = (phi.max() - phi.min()) / niso
        isovalues = np.array([(i+0.5)*diso for i in range(niso)])
        Viso, Eiso, Iiso = igl.isolines(V, F, phi, isovalues)
        # Register contour lines
        cn = ps.register_curve_network("distance contours", Viso, Eiso)
        cn.set_color((0, 0, 0))
        cn.set_radius(0.002)
        # Update mesh visualization with scalar and vector quantities
        vm = ps.get_volume_mesh(
            "model") if mesh.element == pbat.fem.Element.Tetrahedron else ps.get_surface_mesh("model")
        vm.add_scalar_quantity("heat", u, cmap="reds")
        vm.add_scalar_quantity("distance", phi, cmap="reds", enabled=True)
        grad_defined_on = "cells" if mesh.element == pbat.fem.Element.Tetrahedron else "faces"
        vm.add_vector_quantity("normalized heat grad",
                               gradu, defined_on=grad_defined_on)
        vm.add_scalar_quantity("div unit gradient", divGu)
```

## Usage Instructions

1. **Prepare the Input Mesh**: Ensure your input mesh file contains either tetrahedral (`"tetra"`) or triangle (`"triangle"`) elements and is formatted correctly (e.g., `.vtk` format).

2. **Run the Simulation**: Execute the script with the desired command-line arguments. For example:

    ```bash
    python heat_geodesics_demo.py -i input_mesh.vtk -o output_directory
    ```

3. **Interact with the Simulation**:
    - **Adjust Parameters**: Use the GUI controls to modify simulation parameters such as the diffusion coefficient (`k`), the number of isocontours (`niso`), and the source point (`gamma`).
    - **Compute Geodesics**: Click the "Compute" button to perform heat diffusion and compute the geodesic distances. Contour lines representing geodesics will be visualized on the mesh.
    - **Export Results**: Enable the export option to save screenshots of the simulation results.

4. **Visualize the Results**: The Polyscope window will display the simulation mesh, heat distribution, geodesic distances, and contour lines. Observe the progression of heat diffusion and the resulting geodesic contours in real-time.

---

*For further customization and advanced features, refer to the `pbatoolkit` and `polyscope` documentation.*