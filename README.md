# Physics Based Animation Toolkit

[![Wheels](https://github.com/Q-Minh/PhysicsBasedAnimationToolkit/actions/workflows/wheels.yml/badge.svg?event=release)](https://github.com/Q-Minh/PhysicsBasedAnimationToolkit/actions/workflows/wheels.yml)

> *We recommend exploring the official [CMake documentation](https://cmake.org/cmake/help/latest/) to beginner CMake users*.

## Overview

The Physics Based Animation Toolkit (PBAT) is a (mostly templated) cross-platform C++20 library of algorithms and data structures commonly used in computer graphics research on physically-based simulation in dimensions `1,2,3`. For most use cases, we recommend using our library via its Python interface, enabling seamless integration into Python's ecosystem of powerful scientific computing packages. 

> *Our Python bindings are currently *not* available on MacOS* via PyPI.

## Dependencies

See [`vcpkg.json`](./vcpkg.json) for a versioned list of our dependencies, available via [vcpkg](https://github.com/microsoft/vcpkg) (use of [vcpkg](https://github.com/microsoft/vcpkg) is not mandatory, as long as dependencies have compatible versions and are discoverable by CMake's [`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html) mechanism).

## Configuration

| Option | Values | Default | Description |
|---|---|---|---|
| `PBAT_BUILD_PYTHON_BINDINGS` | `ON,OFF` | `OFF` | Enable `PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit` Python bindings. Generates the CMake target `PhysicsBasedAnimationToolkit_Python`, an extension module for Python, built by this project. |
| `PBAT_BUILD_TESTS` | `ON,OFF` | `OFF` | Enable `PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit` unit tests. Generates the CMake target executable `PhysicsBasedAnimationToolkit_Tests`, built by this project. |
| `PBAT_ENABLE_PROFILER` | `ON,OFF` | `OFF` | Enable [`Tracy`](https://github.com/wolfpld/tracy) instrumentation profiling in built `PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit`. |
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

To install locally and get the most up to date features, run in command line:
```bash
pip install .
```

Example code can be found [here](./python/examples/).

To profile relevant calls to `pbatoolkit` functions/methods, connect to `python.exe` in the `Tracy` profiler server GUI.
All calls to pbat will be profiled on a per-frame basis in the Tracy profiler server GUI.

> *Use method `profile` of `pbatoolkit.profiling.Profiler` to profile code external to PBAT, allowing for an integrated profiling experience while using various scientific computing packages*.
> ```python
> def expensive_external_computation():
>     # Some expensive computation
> profiler.profile("My expensive external computation", expensive_external_computation)
> ```

## Contributing

### Coding style

A `.clang-format` description file is provided in the repository root which should be used to enforce a uniform coding style throughout the code base using the [clang-format tool](https://releases.llvm.org/12.0.0/tools/clang/docs/ClangFormatStyleOptions.html). Recent versions of Visual Studio Code and Visual Studio should come bundled with a `clang-format` installation. On Unix-like systems, `clang-format` can be installed using your favorite package manager.
