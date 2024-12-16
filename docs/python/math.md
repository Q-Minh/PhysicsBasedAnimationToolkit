# Math Module

This module provides mathematical operations and tools for solving various numerical problems, including linear algebra, moment fitting, and quadrature rule transfers. It is essential for supporting computational methods in the Physics-Based Animation Toolkit (PBAT).

## Functions

### **block_diagonalize_moment_fitting**

Assembles the block diagonal row sparse matrix `GM`, such that `GM @ w = B.flatten(order='F')` contains all the reference moment fitting systems in `(M, B, P)`.

- **Args**:
  - `M (np.ndarray)`: A matrix of moment fitting systems. Shape: (m, n).
  - `P (np.ndarray)`: Array of indices defining the quadrature points. Shape: (m, 1).

- **Returns**:
  - **scipy.sparse.csr_matrix**: A sparse matrix representing the block-diagonal system.

---

### **reference_moment_fitting_systems**

Obtains a collection of reference moment fitting systems `(M, B, P)` for moment fitting tasks in numerical simulations.

- **Args**:
  - `S1 (np.ndarray)`: Indices for the simplices in the first quadrature set. Shape: (m, 1).
  - `X1 (np.ndarray)`: Arrays of quadrature points for the first set. Shape: (dims, num_points).
  - `S2 (np.ndarray)`: Indices for the simplices in the second quadrature set. Shape: (m, 1).
  - `X2 (np.ndarray)`: Arrays of quadrature points for the second set. Shape: (dims, num_points).
  - `w2 (np.ndarray)`: Quadrature weights for existing quadrature points. Shape: (num_points, 1).
  - `order (int)`: Polynomial order of integration.

- **Returns**:
  - **tuple**: A tuple `(M, B, P)` containing the reference moment fitting matrices and vectors.

---

### **transfer_quadrature**

Transfers quadrature weights and points from one quadrature rule to another. This is typically used for optimizing numerical integration across different discretization schemes.

- **Args**:
  - `S1 (np.ndarray)`: Indices for quadrature points in the first rule. Shape: (m, 1).
  - `X1 (np.ndarray)`: Arrays of quadrature points in the first rule. Shape: (dims, num_points).
  - `S2 (np.ndarray)`: Indices for quadrature points in the second rule. Shape: (m, 1).
  - `X2 (np.ndarray)`: Arrays of quadrature points in the second rule. Shape: (dims, num_points).
  - `w2 (np.ndarray)`: Existing quadrature weights in the second rule. Shape: (num_points, 1).
  - `order (int, optional)`: Desired polynomial order for the quadrature rule transfer. Default is `1`.
  - `with_error (bool, optional)`: Whether to return the integration error. Default is `False`.
  - `max_iters (int, optional)`: Maximum iterations for convergence. Default is `20`.
  - `precision (float, optional)`: Desired precision for the transfer. Default is `2.220446049250313e-16`.

- **Returns**:
  - **tuple**: A tuple `(w1, error)` containing the transferred quadrature weights and optional error estimate.