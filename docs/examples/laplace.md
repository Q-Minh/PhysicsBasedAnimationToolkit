# Higher Order FEM Demo

This script demonstrates the computation of **harmonic fields** on a 3D mesh using the **Higher Order Finite Element Method (FEM)**. It calculates harmonic solutions of different orders (linear and quadratic) and visualizes the results using **Polyscope**. The demo showcases how higher-order elements can provide more accurate representations of physical phenomena on complex geometries.

## Prerequisites

Ensure the following Python libraries are installed before running the script:

- `pbatoolkit`
- `igl`
- `meshio`
- `polyscope`
- `numpy`
- `scipy`
- `argparse`

You can install the required packages using `pip`:

```bash
pip install pbatoolkit igl meshio polyscope numpy scipy argparse
```

*Note*: If `pbatoolkit` is not available via `pip`, refer to its official documentation for installation instructions.

## Command-line Arguments

The script accepts several command-line arguments to customize the simulation:

- `-i`, `--input`: **Path to input tetrahedral mesh file** (required). The mesh should contain tetrahedral elements.
- `-r`, `--refined-input`: **Path to refined input tetrahedral mesh file** (required). This mesh is used for detailed visualization of the harmonic fields.

### Example Usage

```bash
python higher_order_fem_demo.py -i input_mesh.vtk -r refined_mesh.vtk
```

- **Explanation**:
  - Loads `input_mesh.vtk` as the primary tetrahedral mesh.
  - Loads `refined_mesh.vtk` as the refined mesh for visualization.

## Workflow Overview

### 1. Loading the Input Meshes

The script begins by loading the input mesh and the refined mesh using `meshio`. It supports both linear (`order=1`) and higher-order (`order=2`) tetrahedral elements for the FEM simulation.

```python
# Load input mesh
imesh = meshio.read(args.input)
V, C = imesh.points, imesh.cells_dict["tetra"]
V = np.copy(V, order='c')
C = C.astype(np.int64, order='c')
```

### 2. Computing Harmonic Fields

Two harmonic fields of different orders are computed on the mesh. The `harmonic_field` function constructs the FEM mesh, applies Dirichlet boundary conditions, and solves the boundary value problem to obtain the harmonic solutions.

```python
u1, mesh1 = harmonic_field(V, C, order=1)
u2, mesh2 = harmonic_field(V, C, order=2)
```

### 3. Mapping to Refined Mesh

A Bounding Volume Hierarchy (BVH) is constructed to efficiently map the harmonic solutions from the original mesh to the refined mesh. This ensures that the visualizations on the refined mesh accurately reflect the computed harmonic fields.

```python
bvh = pbat.geometry.bvh(V.T, C.T, cell=pbat.geometry.Cell.Tetrahedron)
rmesh = meshio.read(args.rinput)
Vrefined, Crefined = rmesh.points.astype(
    np.float64, order='c'), rmesh.cells_dict["tetra"].astype(np.int64, order='c')
Frefined = igl.boundary_facets(Crefined)
Frefined[:, :2] = np.roll(Frefined[:, :2], shift=1, axis=1)

e, d = bvh.nearest_primitives_to_points(Vrefined.T)
Xi1 = pbat.fem.reference_positions(mesh1, e, Vrefined.T)
Xi2 = pbat.fem.reference_positions(mesh2, e, Vrefined.T)
phi1 = pbat.fem.shape_functions_at(mesh1, Xi1)
phi2 = pbat.fem.shape_functions_at(mesh2, Xi2)
u1ref = (u1[mesh1.E[:, e]] * phi1).sum(axis=0)
u2ref = (u2[mesh2.E[:, e]] * phi2).sum(axis=0)
```

### 4. Visualizing the Results

Polyscope is initialized for real-time visualization of the refined mesh and the computed harmonic fields. Scalar quantities representing the harmonic solutions are added to the mesh for visualization. Additionally, contour lines (isolines) are generated to represent levels of equal harmonic values.

```python
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_ground_plane_mode("shadow_only")
ps.init()
vm = ps.register_volume_mesh("domain refined", Vrefined, Crefined)
vm.add_scalar_quantity("Order 1 harmonic solution",
                       u1ref, enabled=True, cmap="turbo")
vm.add_scalar_quantity("Order 2 harmonic solution", u2ref, cmap="turbo")
niso = 15
```

### 5. Generating Isocontours

The script computes isocontours (isolines) for both harmonic solutions. These contours represent lines of equal harmonic value on the refined mesh, providing a clear visualization of the harmonic fields.

```python
def isolines(V, F, u, niso):
    # Code for libigl 2.5.1
    diso = (u.max() - u.min()) / (niso+2)
    isovalues = np.array([(i+1)*diso for i in range(niso)])
    Viso, Eiso, Iiso = igl.isolines(V, F, u, isovalues)
    return Viso, Eiso

Viso1, Eiso1 = isolines(Vrefined, Frefined, u1ref, niso)
Viso2, Eiso2 = isolines(Vrefined, Frefined, u2ref, niso)
cn1 = ps.register_curve_network("Order 1 contours", Viso1, Eiso1)
cn1.set_radius(0.002)
cn1.set_color((0, 0, 0))
cn2 = ps.register_curve_network("Order 2 contours", Viso2, Eiso2)
cn2.set_radius(0.002)
cn2.set_color((0, 0, 0))
cn2.set_enabled(False)
ps.show()
```

## Usage Instructions

1. **Prepare the Input Meshes**: Ensure your input mesh files contain tetrahedral elements and are formatted correctly (e.g., `.vtk` format).

2. **Run the Simulation**: Execute the script with the desired command-line arguments. For example:

    ```bash
    python higher_order_fem_demo.py -i input_mesh.vtk -r refined_mesh.vtk
    ```

3. **Visualize the Results**: The Polyscope window will display the refined mesh along with the computed harmonic fields. Toggle between different harmonic orders and observe the corresponding isocontours.

---

*For further customization and advanced features, refer to the `pbatoolkit` and `polyscope` documentation.*