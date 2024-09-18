# Physics Based Animation Toolkit

[![Wheels](https://github.com/Q-Minh/PhysicsBasedAnimationToolkit/actions/workflows/wheels.yml/badge.svg?event=release)](https://github.com/Q-Minh/PhysicsBasedAnimationToolkit/actions/workflows/wheels.yml)

> _We recommend exploring the official [CMake documentation](https://cmake.org/cmake/help/latest/) to beginner CMake users_.

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
- GPU algorithms
  - eXtended Position Based Dynamics (XPBD)
  - Broad phase collision detection
    - Sweep and Prune
    - Linear Bounding Volume Hierarchy
  - Fixed-size matrix operations in kernel code
- Seamless profiling integration via [Tracy](https://github.com/wolfpld/tracy)

## Dependencies

See [`vcpkg.json`](./vcpkg.json) for a versioned list of our dependencies, available via [vcpkg](https://github.com/microsoft/vcpkg) (use of [vcpkg](https://github.com/microsoft/vcpkg) is not mandatory, as long as dependencies have compatible versions and are discoverable by CMake's [`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html) mechanism).

## Configuration

| Option                       | Values   | Default | Description                                                                                                                                                                                                             |
| ---------------------------- | -------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PBAT_BUILD_PYTHON_BINDINGS` | `ON,OFF` | `OFF`   | Enable `PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit` Python bindings. Generates the CMake target `PhysicsBasedAnimationToolkit_Python`, an extension module for Python, built by this project.            |
| `PBAT_BUILD_TESTS`           | `ON,OFF` | `OFF`   | Enable `PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit` unit tests. Generates the CMake target executable `PhysicsBasedAnimationToolkit_Tests`, built by this project.                                       |
| `PBAT_ENABLE_PROFILER`       | `ON,OFF` | `OFF`   | Enable [`Tracy`](https://github.com/wolfpld/tracy) instrumentation profiling in built `PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit`.                                                                      |
| `PBAT_PROFILE_ON_DEMAND`     | `ON,OFF` | `OFF`   | Activate Tracy's on-demand profiling when `PBAT_ENABLE_PROFILER` is `ON`.                                                                                                                                               |
| `PBAT_USE_INTEL_MKL`         | `ON,OFF` | `OFF`   | Link to user-provided [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) installation via CMake's [`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html). |
| `PBAT_USE_SUITESPARSE`       | `ON,OFF` | `OFF`   | Link to user-provided [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) installation via CMake's [`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html).                       |
| `PBAT_BUILD_SHARED_LIBS`     | `ON,OFF` | `OFF`   | Build project's library targets as shared/dynamic.                                                                                                                                                                      |

Our project provides [configuration presets](./CMakePresets.json) that capture typical use configurations. Refer to the [CMake presets documentation](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) for more information.

```bash
cmake -S <path/to/PhysicsBasedAnimationToolkit> -B <path/to/build> #-D<option>=<value>
# or, alternatively
cmake --preset=<my-favorite-user-preset>
```

## Build

Build transparently across platforms using the [cmake build CLI](https://cmake.org/cmake/help/latest/manual/cmake.1.html#build-a-project).

CMake build targets:
| Target | Description |
|---|---|
| `PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit` | The PBA Toolkit library. |
| `PhysicsBasedAnimationToolkit_Tests` | The test executable, using [doctest](https://github.com/doctest/doctest). |
| `PhysicsBasedAnimationToolkit_Python` | PBAT's Python extension module, using [pybind11](https://github.com/pybind/pybind11). |

For example, to build tests, run:

```bash
cmake --build <path/to/build/folder> --target PhysicsBasedAnimationToolkit_Tests --config Release
```

## Install

From command line:

```bash
cd path/to/PhysicsBasedAnimationToolkit
cmake -S . -B build # -D<option>=<value> ...
cmake --install build --config Release
```

Alternatively, if [`vcpkg`](https://github.com/microsoft/vcpkg) is installed and `VCPKG_ROOT=path/to/vcpkg` is set as an environment variable, you can select one of our available presets, for example `cmake --preset=default` and then install.

## Quick start

> _We recommend downloading the [Tracy](https://github.com/wolfpld/tracy) profiler server to analyze execution of PBAT algorithms, available as [precompiled executable](https://github.com/wolfpld/tracy/releases). PBAT currently supports [Tracy 0.10](https://github.com/wolfpld/tracy/releases/tag/v0.10)._

### C++

Take a look at the unit tests, found in the library's source (`.cpp` or `.cu`) files.

### Python

To download and install from PyPI, run in command line:

```bash
pip install pbatoolkit
```

> _To use [`pbatoolkit`](https://pypi.org/project/pbatoolkit/)'s GPU algorithms, you must build from source, i.e. the prebuilt [`pbatoolkit`](https://pypi.org/project/pbatoolkit/) package hosted from PyPI does not include GPU code._

For a local installation, which builds from source, our Python bindings build relies on [Scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/index.html), which relies on CMake's [`install`](https://cmake.org/cmake/help/latest/command/install.html) mechanism. As such, you can configure the installation as you typically would when using the CMake CLI directly, by now passing the corresponding CMake arguments in `pip`'s `config-settings` parameter (refer to the [Scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/index.html) documentation for the relevant parameters). See our [pyinstall workflow](.github/workflows/pyinstall.yml) for working examples of building from source on Linux, MacOS and Windows. Then, assuming that external dependencies are found via CMake's [`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html), you can build and install our Python package [`pbatoolkit`](https://pypi.org/project/pbatoolkit/) locally and get the most up to date features. Consider using a [Python virtual environment](https://docs.python.org/3/library/venv.html) for this step.

As an example, assuming use of [`vcpkg`](https://github.com/microsoft/vcpkg) for external dependency management, with `VCPKG_ROOT` set as an environment variable, run

```bash
pip install . --config-settings=cmake.args="--preset=pip-local" -v
```

on the command line to build [`pbatoolkit`](https://pypi.org/project/pbatoolkit/) from source. To build with GPU algorithms included, refer to the Configuration section. Additional CMake variables (i.e. [CMAKE_CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html#variable:CMAKE_CUDA_ARCHITECTURES), [CMAKE_CUDA_COMPILER](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html#variable:CMAKE_%3CLANG%3E_COMPILER)) may be required to be set in order for CMake to discover your local CUDA installation.

Verify [`pbatoolkit`](https://pypi.org/project/pbatoolkit/)'s contents in Python shell:

```python
import pbatoolkit as pbat
help(pbat.fem)
help(pbat.geometry)
help(pbat.profiling)
help(pbat.math.linalg)
```

To profile relevant calls to [`pbatoolkit`](https://pypi.org/project/pbatoolkit/) functions/methods, connect to `python.exe` in the `Tracy` profiler server GUI.
All calls to pbat will be profiled on a per-frame basis in the Tracy profiler server GUI.

> _Use method `profile` of `pbatoolkit.profiling.Profiler` to profile code external to PBAT, allowing for an integrated profiling experience while using various scientific computing packages_.
>
> ```python
> def expensive_external_computation():
>     # Some expensive computation
> profiler.profile("My expensive external computation", expensive_external_computation)
> ```

### Tutorial

Head over to our hands-on [tutorials section](./doc/tutorial/) to learn more about physics based animation in both theory and practice!

## Gallery

Below, we show a few examples of what can be done in just a few lines of code using [`pbatoolkit`](https://pypi.org/project/pbatoolkit/) and Python. Code can be found [here](./python/examples/).

##### Harmonic interpolation

A smooth (harmonic) function is constructed on [Entei](<https://bulbapedia.bulbagarden.net/wiki/Entei_(Pok%C3%A9mon)>), required to evaluate to `1` on its paws, and `0` at the top of its tail, using piece-wise linear (left) and quadratic (right) shape functions. Its isolines are displayed as black curves.

<p float="left">
  <img src="doc/imgs/entei.harmonic.interpolation.order.1.png" width="250" alt="Harmonic interpolation on Entei model using linear shape functions" />
  <img src="doc/imgs/entei.harmonic.interpolation.order.2.png" width="250" alt="Harmonic interpolation on Entei model using quadratic shape functions" /> 
</p>

##### Heat method for geodesic distance computation

Approximate geodesic distances are computed from the top center vertex of [Metagross](<https://bulbapedia.bulbagarden.net/wiki/Metagross_(Pok%C3%A9mon)>) by diffusing heat from it (left), and recovering a function whose gradient matches the normalized heat's negative gradient. Its isolines are displayed as black curves.

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

##### Inter-penetration free elastodynamic contact

Combining [`pbatoolkit`](https://pypi.org/project/pbatoolkit/)'s FEM+elasticity features and the [`IPC Toolkit`](https://ipctk.xyz/) results in guaranteed inter-penetration free contact dynamics between deformable bodies.

<p float="left">
    <img src="doc/imgs/ipc.bar.stacks.gif" width="250" alt="A stack of bending beams fall on top of each other, simulated via Incremental Potential Contact (IPC)." />
</p>

##### Real-time elastodynamics

Our GPU implementation of the eXtended Position Based Dynamics (XPBD) algorithm simulates a ~324k element FEM elastic mesh interactively with contact.

<p float="left">
    <img src="doc/imgs/gpu.xpbd.bvh.gif" width="250" alt="A 162k element armadillo mesh is dropped on top of another duplicate, but fixed, armadillo mesh on the bottom." />
</p>

##### Modal analysis

The hyper elastic beam's representative deformation modes, i.e. its low frequency eigen vectors,
are animated as time continuous signals.

<p float="left">
    <img src="doc/imgs/beam.modes.gif" width="250" alt="Unconstrained hyper elastic beam's eigen frequencies" />
</p>

##### GPU broad phase collision detection

Real-time collision detection between 2 large scale meshes (~324k tetrahedra) is accelerated by highly parallel implementations of the [sweep and prune](https://en.wikipedia.org/wiki/Sweep_and_prune) algorithm, or [linear bounding volume hierarchies](https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf).

<p float="left">
    <img src="doc/imgs/gpu.broadphase.gif" width="250" alt="Broad phase collision detection on the GPU between 2 moving tetrahedral meshes" />
</p>

##### Profiling statistics

Computation details are gathered when using [`pbatoolkit`](https://pypi.org/project/pbatoolkit/) and consulted in the [Tracy](https://github.com/wolfpld/tracy) profiling server GUI.

<p float="left">
    <img src="doc/imgs/profiling.statistics.png" alt="Profiling statistics widget in Tracy server" />
</p>

## Contributing

### Coding style

A `.clang-format` description file is provided in the repository root which should be used to enforce a uniform coding style throughout the code base using the [clang-format tool](https://releases.llvm.org/12.0.0/tools/clang/docs/ClangFormatStyleOptions.html). Recent versions of Visual Studio Code and Visual Studio should come bundled with a `clang-format` installation. On Unix-like systems, `clang-format` can be installed using your favorite package manager.
