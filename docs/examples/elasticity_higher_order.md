# Simple 3D Elastic Simulation using Quadratic FEM Tetrahedra

This script demonstrates a **3D elastic simulation** based on **quadratic Finite Element Method (FEM) tetrahedra**. It simulates the elastic behavior of a 3D mesh under gravity, solving for the deformation of the mesh over time using a Newton-Raphson method.

## Prerequisites

Ensure the following libraries are installed before running the script:

- `pbatoolkit`
- `meshio`
- `polyscope`
- `numpy`
- `scipy`

You can install the required packages using pip:

```bash
pip install pbatoolkit meshio polyscope numpy scipy
```

## Command-line Arguments

The script accepts several command-line arguments for customization:

- `--input`: Path to the input tetrahedral mesh file (required).
- `--refined-input`: Path to a refined surface mesh for visualization (required).
- `--mass-density`: Mass density of the material (default: 1000).
- `--young-modulus`: Young's modulus of the material (default: 1e6).
- `--poisson-ratio`: Poisson's ratio of the material (default: 0.45).

### Example usage:

```bash
python script.py --input /path/to/input_mesh.vtk --refined-input /path/to/refined_mesh.obj --mass-density 1000 --young-modulus 1e6 --poisson-ratio 0.45
```

## Workflow Overview

### 1. Loading the Meshes

Two meshes are loaded using `meshio`:
- A tetrahedral domain mesh for the FEM simulation (`V`, `C`).
- A refined surface mesh for visualization purposes (`VR`, `FR`).

```python
imesh = meshio.read(args.input)
V, C = imesh.points, imesh.cells_dict["tetra"]
rimesh = meshio.read(args.rinput)
VR, FR = rimesh.points, rimesh.cells_dict["triangle"]
```

### 2. Setting up FEM Simulation

The **FEM Mesh** is constructed using quadratic tetrahedral elements. We calculate the **mass matrix** and precompute its inverse for efficient simulations. The load vector due to gravity is also constructed.

```python
mesh = pbat.fem.Mesh(V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=2)
detJeM = pbat.fem.jacobian_determinants(mesh, quadrature_order=4)
M = pbat.fem.MassMatrix(mesh, detJeM, rho=args.rho, dims=3, quadrature_order=4).to_matrix()
Minv = pbat.math.linalg.ldlt(M)
Minv.compute(M)
```

### 3. Hyperelastic Potential and Boundary Conditions

A **Stable Neo-Hookean Hyperelastic Potential** is used to model the elastic energy of the material, governed by the **Young’s modulus** and **Poisson’s ratio**. Dirichlet boundary conditions are set at the extremities of the mesh.

```python
Y = np.full(mesh.E.shape[1], args.Y)
nu = np.full(mesh.E.shape[1], args.nu)
psi = pbat.fem.HyperElasticEnergy.StableNeoHookean
hep = pbat.fem.HyperElasticPotential(mesh, detJeU, GNeU, Y, nu, energy=psi, quadrature_order=4)
hep.precompute_hessian_sparsity()
```

### 4. Solving the Elasticity Problem

The simulation is performed using Newton’s method. At each timestep, we solve the system for displacement, update the velocity and position of the nodes, and compute the forces acting on the mesh.

```python
Hdd = hep.to_matrix()[:, dofs].tocsr()[dofs, :]
Hddinv = pbat.math.linalg.ldlt(Hdd)
Hddinv.analyze(Hdd)
```

### 5. Visualization

**Polyscope** is used for visualizing the simulation in real time. The mesh deformation is shown dynamically as the simulation runs, and users can control the simulation speed and toggle the animation.

```python
ps.set_up_dir("z_up")
vm = ps.register_volume_mesh("Domain", V, C)
sm = ps.register_surface_mesh("Visual", VR, FR)
```

### 6. User Interaction

The user interface provides controls to adjust the time step (`dt`) and toggle the animation. Clicking the "step" button advances the simulation by one step.

```python
def callback():
    global dt, animate, step
    changed, dt = imgui.InputFloat("dt", dt)
    changed, animate = imgui.Checkbox("animate", animate)
    step = imgui.Button("step")
```

## Code Sample

```python
import pbatoolkit as pbat
import meshio
import numpy as np
import polyscope as ps
import polyscope.imgui as imgui
import math
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Simple 3D elastic simulation using quadratic FEM tetrahedra")
    parser.add_argument("-i", "--input", help="Path to input mesh", type=str, required=True)
    parser.add_argument("-r", "--refined-input", help="Path to refined input mesh", type=str, required=True)
    parser.add_argument("-m", "--mass-density", help="Mass density", type=float, default=1000.)
    parser.add_argument("-Y", "--young-modulus", help="Young's modulus", type=float, default=1e6)
    parser.add_argument("-n", "--poisson-ratio", help="Poisson's ratio", type=float, default=0.45)
    args = parser.parse_args()

    # Load meshes
    imesh = meshio.read(args.input)
    V, C = imesh.points, imesh.cells_dict["tetra"]
    V, C = V.astype(np.float64, order='c'), C.astype(np.int64, order='c')
    rimesh = meshio.read(args.rinput)
    VR, FR = rimesh.points, rimesh.cells_dict["triangle"]
    VR, FR = VR.astype(np.float64, order='c'), FR.astype(np.int64, order='c')

    # FEM setup
    mesh = pbat.fem.Mesh(V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=2)
    detJeM = pbat.fem.jacobian_determinants(mesh, quadrature_order=4)
    rho = args.rho
    M = pbat.fem.MassMatrix(mesh, detJeM, rho=rho, dims=3, quadrature_order=4).to_matrix()
    Minv = pbat.math.linalg.ldlt(M)
    Minv.compute(M)

    # Gravity load vector
    g = np.zeros(mesh.dims)
    g[-1] = -9.81
    detJef = pbat.fem.jacobian_determinants(mesh, quadrature_order=2)
    fe = np.tile(rho * g[:, np.newaxis], mesh.quadrature_weights(2).shape[0] * mesh.E.shape[1])
    qgf = pbat.fem.inner_product_weights(mesh, quadrature_order=2).flatten(order="F")
    Qf = sp.sparse.diags_array([qgf], offsets=[0])
    Nf = pbat.fem.shape_function_matrix(mesh, quadrature_order=2)
    f = fe @ Qf @ Nf
    f = f.reshape(math.prod(f.shape), order="F")

    # Hyperelastic potential
    Y = np.full(mesh.E.shape[1], args.Y)
    nu = np.full(mesh.E.shape[1], args.nu)
    psi = pbat.fem.HyperElasticEnergy.StableNeoHookean
    hep = pbat.fem.HyperElasticPotential(mesh, detJeU, GNeU, Y, nu, energy=psi, quadrature_order=4)
    hep.precompute_hessian_sparsity()

    # Visualization setup
    ps.init()
    vm = ps.register_volume_mesh("Domain", V, C)
    sm = ps.register_surface_mesh("Visual", VR, FR)
    ps.show()
```

### Running the Script

```bash
python script.py --input /path/to/input_mesh.vtk --refined-input /path/to/refined_mesh.obj
```
