# Diffusion Mesh Smoothing Demo

This script demonstrates how to perform **diffusion-based mesh smoothing** using the Finite Element Method (FEM). It constructs the necessary operators for a given triangle mesh and applies a heat diffusion process to smooth the mesh. The visualization is handled through **Polyscope**, providing real-time feedback with controls for adjusting smoothing parameters.

## Prerequisites

Make sure you have the required libraries installed:

- `pbatoolkit`
- `igl`
- `polyscope`
- `numpy`
- `scipy`
- `meshio`

Install them with `pip`:

```bash
pip install pbatoolkit igl polyscope numpy scipy meshio
```

## Command-line Arguments

The script expects the following argument:

- `--input`: Path to the input triangle mesh file (required).

### Example usage:

```bash
python script.py --input /path/to/triangle/mesh.obj
```

## Workflow Overview

### 1. Reading and Preparing the Mesh

The input mesh is loaded using `meshio`, which reads the vertices (`V`) and faces (`F`) from the file. The data is converted to suitable types for further processing.

```python
imesh = meshio.read(args.input)
V, F = imesh.points, imesh.cells_dict["triangle"]
V, F = V.astype(np.float64, order='c'), F.astype(np.int64, order='c')
```

### 2. Finite Element Method (FEM) Setup

The FEM framework from `pbatoolkit` is used to set up the following:

- **Laplacian Operator (`L`)**: Used to model the heat diffusion.
- **Mass Matrix (`M`)**: Helps in discretizing the system.
  
The `detJeL` and `GNeL` are Jacobian determinants and shape function gradients, respectively, which are necessary for constructing the FEM Laplacian.

```python
mesh = pbat.fem.Mesh(V.T, F.T, element=pbat.fem.Element.Triangle, order=1)
detJeL = pbat.fem.jacobian_determinants(mesh, quadrature_order=1)
GNeL = pbat.fem.shape_function_gradients(mesh, quadrature_order=1)
L = pbat.fem.Laplacian(mesh, detJeL, GNeL, quadrature_order=1).to_matrix()
detJeM = pbat.fem.jacobian_determinants(mesh, quadrature_order=2)
M = pbat.fem.MassMatrix(mesh, detJeM, dims=1, quadrature_order=2).to_matrix()
```

### 3. Diffusion Process

The diffusion is modeled using a simple heat equation. A timestep (`dt`) and diffusion coefficient (`c`) are defined, and a linear system is constructed. The matrix `A` is precomputed to enable efficient updates during smoothing.

```python
dt = 0.016
c = 1
A = M - c*dt*L
Ainv = pbat.math.linalg.ldlt(A)
Ainv.compute(A)
```

### 4. Visualization

The script uses **Polyscope** to visualize the input mesh and the smoothed result. Two meshes are registered: one for the original model and one for the smoothed version. A user interface is provided to adjust the timestep (`dt`) and the diffusion coefficient (`c`), as well as toggle the smoothing process.

```python
ps.init()
vmm = ps.register_surface_mesh("model", V, F)
vms = ps.register_surface_mesh("smoothed", V, F)
```

### 5. User Interaction

The user can adjust the following parameters:

- **Timestep (`dt`)**: Controls the smoothing speed.
- **Diffusion Coefficient (`c`)**: Affects the intensity of the smoothing.
- **Smooth Toggle**: Enables or disables the smoothing process in real-time.

```python
def callback():
    global dt, Ainv, M, L, smooth, V, c
    dtchanged, dt = imgui.InputFloat("dt", dt)
    cchanged, c = imgui.SliderFloat("c", c, v_min=0, v_max=100)
    if dtchanged or cchanged:
        A = M - c*dt*L
        Ainv.factorize(A)
    _, smooth = imgui.Checkbox("smooth", smooth)
    if smooth:
        V = Ainv.solve(M @ V)
        vms.update_vertex_positions(V)
```

### Running the Script

Run the script and pass the path to a triangle mesh file to apply diffusion-based smoothing:

```bash
python script.py --input /path/to/triangle/mesh.obj
```