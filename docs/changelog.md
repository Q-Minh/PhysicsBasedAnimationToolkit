# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.0.8] - 2024-04-22

### Added
- New VBD (Vertex Block Descent) implementation on the GPU. Currently does not support contacts.
- Alternative adaptive initialization strategy for VBD, differing from the strategy proposed in the VBD paper.
- Nested cage generation tool.
- Compilation of GPU code included in the CI servers within `pyinstall.yml`.
- Split PyPI package into `pbatoolkit` and `pbatoolkit-gpu` to facilitate easier usage of GPU features.

### Changed
- Upgraded the repository's README for improved aesthetics and better documentation on using GPU features.

### Fixed
- (No fixes in this release)

### Contributors
- **@pranavAL** made their first contribution in [#3](https://github.com/Q-Minh/PhysicsBasedAnimationToolkit/pull/3).

**Full Changelog:** [v0.0.7...v0.0.8](https://github.com/Q-Minh/PhysicsBasedAnimationToolkit/compare/v0.0.7...v0.0.8)

---

## [v0.0.7] - 2024-09-18

### Added
- Incremental Potential Contact (IPC) Python example using `pbatoolkit` as the elastodynamics engine.
- Support for optional elastic element Hessian SPD (Symmetric Positive Definite) projections.
- Parallelized Sweep and Prune GPU implementation for broad phase collision detection.
- Various C++ utilities for simpler GPU programming (e.g., device buffers, queues, stacks, lists).
- XPBD (eXtended Position Based Dynamics) implementation on the GPU using stable neo-Hookean constraints and vertex-triangle contact constraints.
- Parallelized linear BVH (Bounding Volume Hierarchy) data structure on the GPU for broad phase overlap and nearest neighbor queries.

### Changed
- Allow Windows system search paths to be examined for DLL resolution when importing `pbatoolkit`.

### Fixed
- (No fixes in this release)

### Contributors
- **@pranavAL**

**Full Changelog:** [v0.0.6...v0.0.7](https://github.com/Q-Minh/PhysicsBasedAnimationToolkit/compare/v0.0.6...v0.0.7)

---

## [v0.0.6] - 2024-07-18

### Added
- Support and tests for optional integration with fast linear solvers (SuiteSparse's Cholmod and Intel MKL's Pardiso). *Note:* MKL Pardiso is untested and has known errors.
- Versioned NumPy and SciPy Python dependencies specified in `pyproject.toml`.
- Refactored CMake sources to facilitate shared builds, bundle transitive dependencies in Python bindings installation, and expose better CMake configure presets.
- Enabled on-demand profiling for `pbatoolkit`.
- Simplified Python FEM bindings using type erasure to minimize binary size.

### Changed
- (No changes in this release)

### Fixed
- (No fixes in this release)

**Full Changelog:** [v0.0.5...v0.0.6](https://github.com/Q-Minh/PhysicsBasedAnimationToolkit/compare/v0.0.5...v0.0.6)

---

## [v0.0.5] - 2024-07-12

### Added
- Tutorials on FEM using `pbatoolkit`.

### Changed
- Exposed more library internals to make Python scripting more flexible (e.g., quadrature points and weights on mesh, shape function gradients).

### Fixed
- (No fixes in this release)

**Full Changelog:** [v0.0.4...v0.0.5](https://github.com/Q-Minh/PhysicsBasedAnimationToolkit/compare/v0.0.4...v0.0.5)

---

## [v0.0.4] - 2024-07-09

### Added
- Implemented and tested optional integration with fast linear solvers (SuiteSparse's Cholmod and Intel MKL's Pardiso). *Note:* MKL Pardiso is untested and has known errors.
- Specified versioned NumPy and SciPy Python dependencies in `pyproject.toml`.
- Refactored CMake sources to facilitate shared builds, bundle transitive dependencies in Python bindings installation, and expose better CMake configure presets.
- Enabled on-demand profiling for `pbatoolkit`.
- Simplified Python FEM bindings using type erasure to minimize binary size.

### Changed
- (No changes in this release)

### Fixed
- (No fixes in this release)

**Full Changelog:** [v0.0.3...v0.0.4](https://github.com/Q-Minh/PhysicsBasedAnimationToolkit/compare/v0.0.3...v0.0.4)

---

## [v0.0.3] - 2024-06-26

### Added
- Initial release of `pbatoolkit` with the following features:
  - FEM utilities (mass matrix, Laplacian, gradient, hyperelastic potential, load) for meshes in dimensions 1, 2, and 3. Supports Lagrange shape functions up to order 3 for line, triangle, quadrilateral, tetrahedral, and hexahedral elements. *Note:* For non-linear elements such as quads and hexes, shape function gradients and Jacobians may be inaccurate (untested).
  - Efficient sparse matrix construction.
  - Saint-Venant Kirchhoff and Stable Neo-Hookean material models.
  - Axis-aligned Bounding Volume Hierarchies (BVH) for triangles and tetrahedra.
  - Automatic profiling data generation compatible with Tracy.
- Mostly tested on tetrahedral element meshes; designed to be element-agnostic and dimension-agnostic.
- Sample code demonstrating the use of `pbatoolkit` available in the `examples` folder.

### Changed
- (No changes in this release)

### Fixed
- (No fixes in this release)

**Full Changelog:** [Initial Release](https://github.com/Q-Minh/PhysicsBasedAnimationToolkit/releases/tag/v0.0.3)

---

# Contributors

- **@pranavAL** - Made their first contribution in [#3](https://github.com/Q-Minh/PhysicsBasedAnimationToolkit/pull/3).
