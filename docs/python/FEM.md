# `pbatoolkit._pbat.fem` Module Documentation

The **FEM (Finite Element Method)** module in `pbatoolkit` provides a collection of functions for dealing with mesh elements, computing shape functions, jacobian determinants, and handling various FEM operators like mass matrices, stiffness matrices, and gradients.

## **Functions**

### 1. **`shape_functions_at`**
```python
shape_functions_at(mesh: pbatoolkit._pbat.fem.Mesh, Xi: numpy.ndarray) -> numpy.ndarray
```
This function computes the **shape functions** at specified reference points `Xi`.

- **Parameters:**
  - `mesh`: The input FEM mesh of type `pbatoolkit._pbat.fem.Mesh`.
  - `Xi`: A 2D NumPy array of shape `(m, n)`, representing reference points.
  
- **Returns:** A NumPy array of shape `(m, n)` that holds the shape functions evaluated at the reference points `Xi`.

---

### 2. **`inner_product_weights`**
```python
inner_product_weights(mesh: pbatoolkit._pbat.fem.Mesh, quadrature_order: int = 1) -> numpy.ndarray
inner_product_weights(mesh: pbatoolkit._pbat.fem.Mesh, detJe: numpy.ndarray, quadrature_order: int = 1) -> numpy.ndarray
```
Computes the **inner product weights** as a product of quadrature weights and the Jacobian determinants at element quadrature points.

- **Overloaded versions:**
  - Version 1: Takes the `mesh` and `quadrature_order`.
  - Version 2: Takes the `mesh`, the precomputed Jacobian determinants (`detJe`), and `quadrature_order`.

- **Parameters:**
  - `mesh`: The input FEM mesh.
  - `detJe` (optional): A NumPy array of Jacobian determinants.
  - `quadrature_order`: The order of quadrature to use (default is 1).

- **Returns:** A NumPy array of shape `(m, n)` containing the inner product weights.

---

### 3. **`jacobian_determinants`**
```python
jacobian_determinants(mesh: pbatoolkit._pbat.fem.Mesh, quadrature_order: int = 1) -> numpy.ndarray
```
Computes the **Jacobian determinants** for each element at quadrature points.

- **Parameters:**
  - `mesh`: The input FEM mesh.
  - `quadrature_order`: The order of quadrature for which Jacobian determinants are computed.

- **Returns:** A NumPy array containing the Jacobian determinants at each quadrature point.

---

### 4. **`reference_positions`**
```python
reference_positions(mesh: pbatoolkit._pbat.fem.Mesh, E: numpy.ndarray, X: numpy.ndarray, max_iters: int = 5, eps: float = 1e-10) -> numpy.ndarray
```
Computes the **reference positions** for domain points `X` within elements `E`.

- **Parameters:**
  - `mesh`: The input FEM mesh.
  - `E`: A NumPy array of element indices.
  - `X`: A NumPy array of domain points.
  - `max_iters`: The maximum number of iterations for the computation (default: 5).
  - `eps`: The tolerance level for convergence (default: `1e-10`).

- **Returns:** A NumPy array of reference positions associated with the domain points `X`.

---

### 5. **`shape_function_gradients`**
```python
shape_function_gradients(mesh: pbatoolkit._pbat.fem.Mesh, quadrature_order: int = 1) -> numpy.ndarray
```
Computes the **shape function gradients** for each element at quadrature points.

- **Parameters:**
  - `mesh`: The input FEM mesh.
  - `quadrature_order`: The quadrature order for computing the gradients (default: 1).

- **Returns:** A NumPy array representing the gradients of shape functions at the quadrature points.

---

### 6. **`shape_function_gradients_at`**
```python
shape_function_gradients_at(mesh: pbatoolkit._pbat.fem.Mesh, E: numpy.ndarray, Xi: numpy.ndarray) -> numpy.ndarray
```
Computes the **nodal shape function gradients** at specified reference points `Xi`.

- **Parameters:**
  - `mesh`: The input FEM mesh.
  - `E`: A NumPy array of element indices.
  - `Xi`: A NumPy array of reference points.

- **Returns:** A NumPy array of nodal shape function gradients at the reference points `Xi`.

---

### 7. **`shape_function_matrix`**
```python
shape_function_matrix(mesh: pbatoolkit._pbat.fem.Mesh, quadrature_order: int = 1) -> scipy.sparse.csr_matrix
```
Constructs the **shape function matrix** for the FEM mesh.

- **Parameters:**
  - `mesh`: The input FEM mesh.
  - `quadrature_order`: The quadrature order for the shape function matrix (default: 1).

- **Returns:** A sparse matrix (CSR format) representing the shape functions.

---

## **Data Types**

### Elements

Several element types are predefined in the module to be used with the mesh construction:

- **`Line`**: Represents line elements (1D).
- **`Triangle`**: Represents triangular elements (2D).
- **`Quadrilateral`**: Represents quadrilateral elements (2D).
- **`Tetrahedron`**: Represents tetrahedral elements (3D).
- **`Hexahedron`**: Represents hexahedral elements (3D).

---

### Hyperelastic Energy Types

The module also provides common **hyperelastic energy** models:

- **`SaintVenantKirchhoff`**: Saint Venant-Kirchhoff hyperelastic model.
- **`StableNeoHookean`**: Stable Neo-Hookean hyperelastic model.

---

## **Usage Example**

Below is a brief example showing how to use some functions from this module:

```python
import pbatoolkit as pbat
import numpy as np

# Create a mesh object (example)
V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
C = np.array([[0, 1, 2, 3]])
mesh = pbat.fem.Mesh(V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=1)

# Compute the shape functions at reference points
Xi = np.array([[0.25, 0.25, 0.25]])
shape_funcs = pbat.fem.shape_functions_at(mesh, Xi)

# Compute Jacobian determinants for the mesh
jacobian_dets = pbat.fem.jacobian_determinants(mesh, quadrature_order=2)

# Compute the inner product weights
weights = pbat.fem.inner_product_weights(mesh, quadrature_order=2)

print("Shape Functions:", shape_funcs)
print("Jacobian Determinants:", jacobian_dets)
print("Inner Product Weights:", weights)
```