## Build & Install

### C++

Build and install transparently across platforms using the [cmake build CLI](https://cmake.org/cmake/help/latest/manual/cmake.1.html#build-a-project) and [cmake install CLI](https://cmake.org/cmake/help/latest/guide/tutorial/Installing%20and%20Testing.html), respectively.

Our CMake project exposes the following build targets
| Target | Description |
|---|---|
| `PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit` | The PBA Toolkit library. |
| `PhysicsBasedAnimationToolkit_Tests` | The test executable, using [doctest](https://github.com/doctest/doctest). |
| `PhysicsBasedAnimationToolkit_Python` | PBAT's Python extension module, using [pybind11](https://github.com/pybind/pybind11). |

For example, to build tests, run
```bash
cmake --build <path/to/build/folder> --target PhysicsBasedAnimationToolkit_Tests --config Release
```

To install *PhysicsBasedAnimationToolkit* locally, run
```bash
cd path/to/PhysicsBasedAnimationToolkit
cmake -S . -B build -D<option>=<value> ...
cmake --install build --config Release
```