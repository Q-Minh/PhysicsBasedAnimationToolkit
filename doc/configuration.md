\page configuration Configuration

| Option                       | Values   | Default | Description |
| ---------------------------- | -------- | ------- | ----------- |
| `PBAT_BUILD_PYTHON_BINDINGS` | `ON,OFF` | `OFF`   | Enable `PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit` Python bindings. Generates the CMake target `PhysicsBasedAnimationToolkit_Python`, an extension module for Python, built by this project. |
| `PBAT_BUILD_TESTS`           | `ON,OFF` | `OFF`   | Enable `PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit` unit tests. Generates the CMake target executable `PhysicsBasedAnimationToolkit_Tests`, built by this project. |
| `PBAT_ENABLE_PROFILER`       | `ON,OFF` | `OFF`   | Enable [`Tracy`](https://github.com/wolfpld/tracy) instrumentation profiling in built `PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit`. |
| `PBAT_PROFILE_ON_DEMAND`     | `ON,OFF` | `OFF`   | Activate Tracy's on-demand profiling when `PBAT_ENABLE_PROFILER` is `ON`. |
| `PBAT_USE_INTEL_MKL`         | `ON,OFF` | `OFF`   | Link to user-provided [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) installation via CMake's [`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html). |
| `PBAT_USE_SUITESPARSE`       | `ON,OFF` | `OFF`   | Link to user-provided [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) installation via CMake's [`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html). |
| `PBAT_BUILD_SHARED_LIBS`     | `ON,OFF` | `OFF`   | Build project's library targets as shared/dynamic. |

Either run CMake's configure step manually
```bash
cmake -S <path/to/PhysicsBasedAnimationToolkit> -B <path/to/build> -D<option>=<value> ...
```
or, alternatively (and preferably)
```bash
cmake --preset=<my-favorite-user-preset>
```

Our project provides [configuration presets](https://github.com/Q-Minh/PhysicsBasedAnimationToolkit/blob/master/CMakePresets.json) that capture typical use configurations. For the best experience, install [`vcpkg`](https://github.com/microsoft/vcpkg) and set `VCPKG_ROOT=path/to/vcpkg` as an environment variable. Then, you can select one of our available presets, for example `cmake --preset=default`. Refer to the [CMake presets documentation](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) for more information.

<div class="section_buttons">

| Previous          |         Next |
|:------------------|-------------:|
| \ref dependencies | \ref install |

</div>