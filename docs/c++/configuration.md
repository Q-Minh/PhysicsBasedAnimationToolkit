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


## vcpkg Configuration

The `vcpkg.json` file defines the dependencies and optional features for the **PBAT Toolkit**:

- **Core Dependencies**:
  - `range-v3` for range algorithms.
  - `fmt` for string formatting.
  - `eigen3` for linear algebra.
  - `tbb` for parallel programming.
  - `doctest` for testing.

- **Optional Features**:
  - **`suitesparse`**: For sparse matrix operations.
  - **`mkl`**: For Intel MKL optimized math functions.
  - **`python`**: To enable Python bindings.
  - **`cuda`**: To enable CUDA support for GPU acceleration.

---

## Installing vcpkg


Open a terminal or command prompt and run:

```bash
git clone https://github.com/microsoft/vcpkg.git
```

After cloning, navigate into the **vcpkg** directory and run the bootstrap script to build the tool.

#### On Linux and macOS:

```bash
cd vcpkg
./bootstrap-vcpkg.sh
```

#### On Windows:

```bash
cd vcpkg
.\bootstrap-vcpkg.bat
```

## Setting up CMake Presets

CMake presets simplify the configuration and build process. Follow these steps to set them up.

In the root of your project, create a `CMakePresets.json` file:

```json
{
    "version": 6,
    "configurePresets": [
        {
            "name": "dev-msvc-cuda",
            "inherits": ["dev", "x64", "msvc", "pic"],
            "cacheVariables": {
                "VCPKG_MANIFEST_FEATURES": "python;cuda",
                "PBAT_BUILD_PYTHON_BINDINGS": {
                    "type": "BOOL",
                    "value": "ON"
                },
                "PBAT_USE_CUDA": {
                    "type": "BOOL",
                    "value": "ON"
                }
            }
        }
    ]
}
```

You can adjust the settings in `CMakePresets.json` based on your needs, such as enabling or disabling CUDA or Python bindings.