# Physics Based Animation Toolkit

[![Wheels](https://github.com/Q-Minh/PhysicsBasedAnimationToolkit/actions/workflows/wheels.yml/badge.svg?event=release)](https://github.com/Q-Minh/PhysicsBasedAnimationToolkit/actions/workflows/wheels.yml)

> *We recommend exploring the official [CMake documentation](https://cmake.org/cmake/help/latest/) to beginner CMake users*.

## Overview

The Physics Based Animation Toolkit (PBAT) is a (mostly templated) cross-platform C++20 library of algorithms and data structures commonly used in computer graphics research on physically-based simulation in dimensions `1,2,3`. For most use cases, we recommend using our library via its Python interface, enabling seamless integration into Python's ecosystem of powerful scientific computing packages. 

### Features

- Finite Element Method (FEM) meshes and operators
  - Dimensions `1,2,3`
  - Lagrange shape functions of order `1,2,3`
  - Line, triangle, quadrilateral, tetrahedron and hexahedron elements
- Hyper elastic material models
  - Saint-Venant Kirchhoff
  - Stable Neo-Hookean
- Polynomial quadrature rules
  - Simplices in dimensions `1,2,3`
  - Gauss-Legendre quadrature
- Spatial query acceleration data structures
  - Bounding volume hierarchy for triangles (2D+3D) and tetrahedra (3D)
    - Nearest neighbours
    - Overlapping primitive pairs
    - Point containment
- Seamless profiling integration via [Tracy](https://github.com/wolfpld/tracy)

## Dependencies

See [`vcpkg.json`](./vcpkg.json) for a versioned list of our dependencies, available via [vcpkg](https://github.com/microsoft/vcpkg) (use of [vcpkg](https://github.com/microsoft/vcpkg) is not mandatory, as long as dependencies have compatible versions and are discoverable by CMake's [`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html) mechanism).

## Configuration

| Option | Values | Default | Description |
|---|---|---|---|
| `PBAT_BUILD_PYTHON_BINDINGS` | `ON,OFF` | `OFF` | Enable `PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit` Python bindings. Generates the CMake target `PhysicsBasedAnimationToolkit_Python`, an extension module for Python, built by this project. |
| `PBAT_BUILD_TESTS` | `ON,OFF` | `OFF` | Enable `PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit` unit tests. Generates the CMake target executable `PhysicsBasedAnimationToolkit_Tests`, built by this project. |
| `PBAT_ENABLE_PROFILER` | `ON,OFF` | `OFF` | Enable [`Tracy`](https://github.com/wolfpld/tracy) instrumentation profiling in built `PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit`. |
| `PBAT_PROFILE_ON_DEMAND` | `ON,OFF` | `OFF` | Activate Tracy's on-demand profiling when `PBAT_ENABLE_PROFILER` is `ON`. |
| `PBAT_USE_INTEL_MKL` | `ON,OFF` | `OFF` | Link to user-provided [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) installation via CMake's [`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html). |
| `PBAT_USE_SUITESPARSE` | `ON,OFF` | `OFF` | Link to user-provided [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) installation via CMake's [`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html). |
| `PBAT_BUILD_SHARED_LIBS` | `ON,OFF` | `OFF` | Build project's library targets as shared/dynamic. |

## Build

Build transparently across platforms using the [cmake build CLI](https://cmake.org/cmake/help/latest/manual/cmake.1.html#build-a-project). 

CMake build targets:
| Target | Description |
|---|---|
| `PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit` | The PBA Toolkit library. |
| `PhysicsBasedAnimationToolkit_Tests` | The test executable, using [doctest](https://github.com/doctest/doctest). |
| `PhysicsBasedAnimationToolkit_Python` | PBAT's Python extension module, using [pybind11](https://github.com/pybind/pybind11). |

## Install

From command line:
```bash
cd path/to/PhysicsBasedAnimationToolkit
cmake -S . -B build -A x64 -DPBAT_ENABLE_PROFILER:BOOL=ON -DPBAT_BUILD_TESTS:BOOL=ON
cmake --install build --config Release
```

## Quick start

> *We recommend downloading the [Tracy](https://github.com/wolfpld/tracy) profiler server to analyze execution of PBAT algorithms, available as [precompiled executable](https://github.com/wolfpld/tracy/releases)*.

### C++

Take a look at the unit tests, found in the library's source (`.cpp`) files.

### Python

To download and install from PyPI, run in command line:
```bash
pip install pbatoolkit
```

For a local installation, which builds from source, our Python bindings build relies on [Scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/index.html), which relies on CMake's [`install`](https://cmake.org/cmake/help/latest/command/install.html) mechanism. As such, you can configure the installation as you typically would when using the CMake CLI directly, by now passing the corresponding CMake arguments in the `pip`'s `config-settings` parameter, or via the `SKBUILD_*` environment variables. See our [pyinstall workflow](.github/workflows/pyinstall.yml) for working examples of building from source on Linux, MacOS and Windows. Then, assuming that external dependencies are found via CMake's [`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html), you can build and install our Python package `pbatoolkit` locally and get the most up to date features. Consider using a [Python virtual environment](https://docs.python.org/3/library/venv.html) for this step. 

As an example, using [`vcpkg`](https://github.com/microsoft/vcpkg) for external dependencies, run in command line:
```bash
pip install . --config-settings=cmake.args="-DPBAT_BUILD_PYTHON_BINDINGS:BOOL=ON;-DPBAT_BUILD_TESTS:BOOL=OFF;-DPBAT_ENABLE_PROFILER:BOOL=ON;-DPBAT_PROFILE_ON_DEMAND:BOOL=ON;-DPBAT_BUILD_SHARED_LIBS:BOOL=OFF;-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON;-DCMAKE_TOOLCHAIN_FILE='path/to/vcpkg/scripts/buildsystems/vcpkg.cmake';-DVCPKG_MANIFEST_FEATURES=python;-DPBAT_PROFILE_ON_DEMAND:BOOL=ON" -v
```

For parallel builds on Unix systems, you can add the `build.tool-args=-j<# threads>` to the `config-settings` parameter, where `<# threads>` is the number of compilation jobs to run simultaneously. On Windows, using `MSBuild`, you may specify `build.tool-args=/p:CL_MPCount=<# threads>` instead. This assumes that parallel builds were enabled, meaning `/MP` may need to be appended to `CMAKE_CXX_FLAGS` through `cmake.args`. Otherwise, `CMAKE_BUILD_PARALLEL_LEVEL=<# threads>`may be usable, again through the `cmake.args` parameter of `config-settings`.

Verify `pbatoolkit`'s contents in Python shell:
```python
import pbatoolkit as pbat
import pbatoolkit.fem, pbatoolkit.geometry, pbatoolkit.profiling
import pbatoolkit.math.linalg
help(pbat.fem)
help(pbat.geometry)
help(pbat.profiling)
help(pbat.math.linalg)
```

To profile relevant calls to `pbatoolkit` functions/methods, connect to `python.exe` in the `Tracy` profiler server GUI.
All calls to pbat will be profiled on a per-frame basis in the Tracy profiler server GUI.

> *Use method `profile` of `pbatoolkit.profiling.Profiler` to profile code external to PBAT, allowing for an integrated profiling experience while using various scientific computing packages*.
> ```python
> def expensive_external_computation():
>     # Some expensive computation
> profiler.profile("My expensive external computation", expensive_external_computation)
> ```

### Tutorial

Head over to our hands-on [tutorials section](./doc/tutorial/) to learn more about physics based animation in both theory and practice!

## Gallery

Below, we show a few examples of what can be done in just a few lines of code using `pbatoolkit` and Python. Code can be found [here](./python/examples/).

##### Harmonic interpolation
A smooth (harmonic) function is constructed on [Entei](https://bulbapedia.bulbagarden.net/wiki/Entei_(Pok%C3%A9mon)), required to evaluate to `1` on its paws, and `0` at the top of its tail, using piece-wise linear (left) and quadratic (right) shape functions. Its isolines are displayed as black curves.
<p float="left">
  <img src="doc/imgs/entei.harmonic.interpolation.order.1.png" width="250" alt="Harmonic interpolation on Entei model using linear shape functions" />
  <img src="doc/imgs/entei.harmonic.interpolation.order.2.png" width="250" alt="Harmonic interpolation on Entei model using quadratic shape functions" /> 
</p>

##### Heat method for geodesic distance computation
Approximate geodesic distances are computed from the top center vertex of [Metagross](https://bulbapedia.bulbagarden.net/wiki/Metagross_(Pok%C3%A9mon)) by diffusing heat from it (left), and recovering a function whose gradient matches the normalized heat's negative gradient. Its isolines are displayed as black curves.
<p float="left">
  <img src="doc/imgs/metagross.heat.source.png" width="250" alt="Heat source on top center of metagross model" />
  <img src="doc/imgs/metagross.heat.geodesics.png" width="250" alt="Reconstructed single source geodesic distance" /> 
</p>

##### Mesh smoothing via diffusion
Fine details of Godzilla's skin are smoothed out by diffusing `x,y,z` coordinates in time.
<p float="left">
    <img src="doc/imgs/godzilla.diffusion.smoothing.gif" width="250" alt="Godzilla model with fine details being smoothed out via diffusion" />
</p>

##### Hyper elastic simulation
Linear (left) and quadratic (right) shape functions are compared on a hyper elastic simulation of the beam model, whose left side is fixed. Quadratic shape functions result in visually smoother and softer bending.
<p float="left">
  <img src="doc/imgs/beam.bending.order.1.png" width="250" alt="Bending beam FEM elastic simulation using linear shape functions" />
  <img src="doc/imgs/beam.bending.order.2.png" width="250" alt="Bending beam FEM elastic simulation using quadratic shape functions" /> 
</p>

##### Modal analysis
The hyper elastic beam's representative deformation modes, i.e. its low frequency eigen vectors, 
are animated as time continuous signals.
<p float="left">
    <img src="doc/imgs/beam.modes.gif" width="250" alt="Unconstrained hyper elastic beam's eigen frequencies" />
</p>

##### Profiling statistics
Computation details are gathered when using `pbatoolkit` and consulted in the [Tracy](https://github.com/wolfpld/tracy) profiling server GUI.
<p float="left">
    <img src="doc/imgs/profiling.statistics.png" alt="Profiling statistics widget in Tracy server" />
</p>

## Contributing

### Coding style

A `.clang-format` description file is provided in the repository root which should be used to enforce a uniform coding style throughout the code base using the [clang-format tool](https://releases.llvm.org/12.0.0/tools/clang/docs/ClangFormatStyleOptions.html). Recent versions of Visual Studio Code and Visual Studio should come bundled with a `clang-format` installation. On Unix-like systems, `clang-format` can be installed using your favorite package manager.
