Changelog
=========

v0.0.7 (Sep 14, 2024)
---------------------

- Added Incremental Potential Contact (IPC) Python example using ``pbatoolkit`` as the elastodynamics engine.
- Supported optional elastic element Hessian SPD projections.
- Implemented a parallelized Sweep and Prune GPU implementation for broad-phase collision detection.
- Implemented various C++ tools for simpler GPU programming (i.e., device buffers, queues, stacks, lists, etc.).
- Allowed Windows system search paths to be examined for DLL resolution when importing ``pbatoolkit``.
- Implemented XPBD (Extended Position-Based Dynamics) on the GPU using stable Neo-Hookean constraints and vertex-triangle contact constraints.
- Added a parallelized linear BVH data structure on the GPU for broad-phase overlap and nearest neighbor queries.

Contributors
~~~~~~~~~~~~
- @pranavAL made their first contribution in `#3 <https://github.com/Q-Minh/PhysicsBasedAnimationToolkit/pull/3>`__.


v0.0.6 (Jul 18, 2024)
---------------------

- Implemented and tested optional integration to fast linear solvers (SuiteSparse's Cholmod and Intel MKL's Pardiso). MKL Pardiso is untested and has errors.
- Specified versioned ``numpy`` and ``scipy`` Python dependencies in ``pyproject.toml``.
- Refactored CMake sources to facilitate shared builds, bundle transitive dependencies in Python bindings installation, and expose better CMake configure presets.
- Enabled on-demand profiling for ``pbatoolkit``.
- Simplified Python FEM bindings using type erasure to minimize binary size.


v0.0.5 (Jul 12, 2024)
---------------------

- Ensured dependencies have version requirements (numpy 2 broke the Python bindings).
- Eliminated newlines in Python's f-strings to avoid parsing issues in older Python versions.
- Created cross-platform local pip installation GitHub workflow.
- Included macOS images in wheels workflow (PyPI package release).
- Fixed compiler errors on macOS images targeting AppleClang.


v0.0.4 (Jul 9, 2024)
---------------------

- Wrote tutorials on FEM using ``pbatoolkit``.
- Added and modified FEM operators to facilitate use in matrix operations (e.g., shape function matrix, quadrature matrix, gradient).
- Exposed more library internals to make Python scripting more flexible (quadrature points and weights on mesh, shape function gradients).


v0.0.3 (Jun 26, 2024)
---------------------

- First release of ``pbatoolkit``.
- Included FEM tools for meshes in dimensions 1, 2, and 3.
- Efficient sparse matrix construction.
- Support for Saint-Venant Kirchhoff and Stable Neo-Hookean material models.
- Axis-aligned bounding volume hierarchies for triangles and tetrahedra.
- Automatic profiling data generation using Tracy.
