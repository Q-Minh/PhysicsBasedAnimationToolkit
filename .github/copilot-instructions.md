# PBAT — Copilot Instructions

## Project Overview
Physics Based Animation Toolkit (PBAT) is a C++20 library exposing modular building blocks for physics-based simulation (FEM, elasticity, contact, GPU solvers). It ships as:
- **C++ library** (`PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit`) built with CMake
- **Python package** (`pbatoolkit` / `pbatoolkit-gpu`) via nanobind bindings

## Architecture

### Source Layout (`source/pbat/`)
All C++ sources live under `source/pbat/`. Each subdirectory is a self-contained module added via `add_subdirectory` in CMake. Key modules:
| Module | Purpose |
|--------|---------|
| `fem/` | Finite element meshes, shape functions, quadrature, operators (Mass, Laplacian, Gradient, etc.) |
| `geometry/` | BVH, SDF, spatial queries, CCD, mesh boundary, hash grids |
| `physics/` | Hyper-elastic energy models (StableNeoHookean, SaintVenantKirchhoff) |
| `sim/` | Simulation algorithms — split into `sim/dynamics/`, `sim/contact/`, `sim/algorithm/` |
| `gpu/` | CUDA implementations mirroring CPU modules (`gpu/vbd/`, `gpu/xpbd/`, `gpu/contact/`, `gpu/geometry/`) |
| `math/linalg/mini/` | Header-only small fixed-size linear algebra library used in kernels (host+device) |
| `graph/` | Adjacency, coloring, partitioning, BFS/DFS, connected components |
| `common/` | Utilities — sorting, indexing, concepts, Eigen helpers |
| `io/` | HDF5 archive serialization via HighFive |
| `profiling/` | Tracy profiler integration (`PBAT_PROFILE_SCOPE`, `PBAT_PROFILE_NAMED_SCOPE`) |

### Migration (`sim/` vs `sim/algorithm/`)
The previous library architecture had `sim/vbd/` and `sim/xpbd/` contain higher-level `Data`/`Integrator` types, but I want to deprecate them. The `sim/algorithm/` layer is the new API that directly operates on `*Dynamics` types, parameterized by `*Params` types, and attempts as best as possible to code against reusable kernels `Kernels.h` files with `PBAT_HOST_DEVICE` functions shared between CPU and GPU code paths. When adding simulation logic, put reusable math in `sim/algorithm/*/Kernels.h`.
We favor free functions and generally use class/struct types for storage purposes.

### Python Bindings (`bindings/pypbat/`)
Bindings use **nanobind** (not pybind11). Each module under `bindings/pypbat/` mirrors `source/pbat/` (e.g., `bindings/pypbat/fem/`, `bindings/pypbat/sim/`). The entry point is `bindings/pypbat/PythonBindings.cpp` registering submodules. The Python package itself is at `python/pbatoolkit/`.

## Key Types & Conventions

### Aliases (`source/pbat/Aliases.h`)
- `pbat::Scalar` = `float`, `pbat::Index` = `std::ptrdiff_t`
- `pbat::VectorX`, `pbat::MatrixX` — dynamic Eigen types using `Scalar`
- `pbat::IndexVectorX`, `pbat::IndexMatrixX` — dynamic Eigen types using `Index`
- `pbat::CSCMatrix`, `pbat::CSRMatrix` — sparse matrix types
- Matrices are **column-major** by convention; mesh data stored as `|dims| x |count|` (e.g., positions are `3 x N`, elements are `|nodes_per_elem| x |num_elems|`)

### C++20 Concepts
The project uses C++20 concepts extensively. Check `Concepts.h` files per module:
- `fem::CElement`, `fem::CMesh` — FEM element/mesh concepts
- `common::CArithmetic`, `common::CIndex`, `common::CFloatingPoint` — type constraints
- `physics::CHyperElasticEnergy` — energy model concept
- `math::linalg::mini::CMatrix` — mini-lib matrix concept

### API Export
Public API functions use `PBAT_API` macro (generated `PhysicsBasedAnimationToolkitExport.h`). Headers in `FILE_SET api` are public; those in `FILE_SET implementation` are private.

### Host/Device Code
`PBAT_HOST_DEVICE` marks functions compilable by both C++ and CUDA compilers. The `math/linalg/mini/` library is designed for this. Avoid Eigen in `PBAT_HOST_DEVICE` functions; use `mini::` types instead.
While Eigen has advanced indexing features, `mini::` also provides a lightweight alternative suitable for GPU code, i.e. via `mini::ToBuffers`, `mini::FromBuffers`, etc. in `math/linalg/mini/Matrix.h`.

## Build System

### CMake Presets (use these, don't guess flags)
```powershell
cmake --preset=default              # basic build (requires VCPKG_ROOT env var)
cmake --preset=dev                  # tests + profiler + docs + benchmarks + cppcheck
cmake --preset=user                 # Generally always use the user preset for local development, as this is exactly the config that the developer wants to use.
```
Key presets: `default`, `dev`, `pip`, `pip-cuda`, `local-pip-cuda`, `user`. See `CMakePresets.json` and `CMakeUserPresets.json`.

### CMake Targets
| Target | Description |
|--------|-------------|
| `PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit` | Main C++ library |
| `PhysicsBasedAnimationToolkit_Tests` | Test executable (doctest) |
| `PhysicsBasedAnimationToolkit_Python` | Python extension module |

### Dependencies
Managed via **vcpkg** (`vcpkg.json`) + **FetchContent** (`cmake/dependencies.cmake`):
- vcpkg: fmt, range-v3, TBB, HDF5 (+ optional: SuiteSparse, MKL, METIS)
- FetchContent: Eigen, Embree, Spectra, doctest, HighFive, nanobind, Tracy, cuda-api-wrappers

### Python Build
```bash
pip install . --config-settings cmake.args="--preset=pip" -v
```

## Testing

### In-Source Tests
Tests are **co-located in `.cpp` files** alongside the code they test, using **doctest**. Example from `source/pbat/fem/Mesh.cpp`:
```cpp
#include <doctest/doctest.h>
TEST_CASE("[fem] Mesh") { /* ... */ }
```
Tests are compiled into the library itself (guarded by `PBAT_HAS_DOCTEST`) and linked into `PhysicsBasedAnimationToolkit_Tests` via `WHOLE_ARCHIVE`. Convention: prefix test names with module in brackets, e.g., `[fem]`, `[geometry]`, `[sim][vbd]`.

### Running Tests
```powershell
cmake --build build --target PhysicsBasedAnimationToolkit_Tests --config Release
.\build\bin\Release\PhysicsBasedAnimationToolkit_Tests.exe
```

## Code Style
- **clang-format** config at `.clang-format` — Allman-like brace style, 100-column limit, 4-space indent, `Standard: c++20`
- Use `// clang-format off` / `// clang-format on` around matrix literal blocks
- Namespace convention: `pbat::<module>::<submodule>` (e.g., `pbat::sim::algorithm::vbd::kernels`)
- Header guards: `#ifndef PBAT_<MODULE>_<FILE>_H`
- Doxygen (javadoc style) comments on all public API with `@brief`, `@param`, `@return`, `@tparam`

## Demos

Demo scripts are generally located in the `python/examples/` directory. They required `pbatoolkit`, i.e. our bindings, to be installed, generally along with other dependencies stated in `requirements.txt` files located in the same directory.

## Tools

Tool scripts are located in the `python/tools` directory. Some tools required `pbatoolkit` to be installed. Most of my current work happens in the `python/tools/vbd` directory, mainly rooted in the `editor.py` and `simulate.py` scripts.

## Adding New Code Checklist
1. Place headers/sources under `source/pbat/<module>/`
2. Register in the module's `CMakeLists.txt` (`FILE_SET api` for public headers, `PRIVATE` for `.cpp`)
3. Write doctest `TEST_CASE` in the `.cpp` file (not in `tests/`)
4. For GPU-shared math, put kernels in `sim/algorithm/*/Kernels.h` with `PBAT_HOST_DEVICE`
5. For Python exposure, add bindings under `bindings/pypbat/<module>/`
6. Ensure documentation is always up-to-date with API changes (or updates APIs that were parsed and un-modified, but contain stale documentation).
7. Run clang-format before committing
8. Ensure all demo scripts are tested and working correctly, or reflect new API changes if execution is disallowed.

