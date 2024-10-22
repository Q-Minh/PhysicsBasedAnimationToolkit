## Configuration



| Option                       | Values   | Default | Description |
| ---------------------------- | -------- | ------- | ----------- |
| `PBAT_BUILD_PYTHON_BINDINGS` | `ON,OFF` | `OFF`   | Enable `PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit` Python bindings. Generates the CMake target `PhysicsBasedAnimationToolkit_Python`, an extension module for Python, built by this project. |
| `PBAT_BUILD_TESTS`           | `ON,OFF` | `OFF`   | Enable `PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit` unit tests. Generates the CMake target executable `PhysicsBasedAnimationToolkit_Tests`, built by this project. |
| `PBAT_ENABLE_PROFILER`       | `ON,OFF` | `OFF`   | Enable [`Tracy`](https://github.com/wolfpld/tracy) instrumentation profiling in built `PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit`. |
| `PBAT_PROFILE_ON_DEMAND`     | `ON,OFF` | `OFF`   | Activate Tracy's on-demand profiling when `PBAT_ENABLE_PROFILER` is `ON`. |
| `PBAT_USE_INTEL_MKL`         | `ON,OFF` | `OFF`   | Link to user-provided [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) installation via CMake's [`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html). |
| `PBAT_USE_SUITESPARSE`       | `ON,OFF` | `OFF`   | Link to user-provided [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) installation via CMake's [`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html). |
| `PBAT_BUILD_SHARED_LIBS`     | `ON,OFF` | `OFF`   | Build project's library targets as shared/dynamic. |



