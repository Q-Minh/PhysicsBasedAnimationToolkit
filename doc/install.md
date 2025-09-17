\page install Install

<div class="tabbed">
- <b class="tab-title">C++</b>
  Build and install transparently across platforms using the [cmake build CLI](https://cmake.org/cmake/help/latest/manual/cmake.1.html#build-a-project) and [cmake install CLI](https://cmake.org/cmake/help/latest/guide/tutorial/Installing%20and%20Testing.html), respectively.

  Our CMake project exposes the following build targets
  | Target | Description |
  |---|---|
  | `PhysicsBasedAnimationToolkit_PhysicsBasedAnimationToolkit` | The PBA Toolkit library. |
  | `PhysicsBasedAnimationToolkit_Tests` | The test executable, using [doctest](https://github.com/doctest/doctest). |
  | `PhysicsBasedAnimationToolkit_Python` | PBAT's Python extension module, using [nanobind](https://github.com/wjakob/nanobind). |
  
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
- <b class="tab-title">Python</b>

  For a local installation, which builds from source, our Python bindings build relies on [Scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/index.html), which relies on CMake's [`install`](https://cmake.org/cmake/help/latest/command/install.html) mechanism. As such, you can configure the installation as you typically would when using the CMake CLI directly, by now passing the corresponding CMake arguments in `pip`'s `config-settings` parameter (refer to the [Scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/index.html) documentation for the relevant parameters). See our [pyinstall workflow](.github/workflows/pyinstall.yml) for working examples of building from source on Linux, MacOS and Windows. Then, assuming that external dependencies are found via CMake's [`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html), you can build and install our Python package [`pbatoolkit`](https://pypi.org/project/pbatoolkit/) locally and get the most up to date features. 
  
  \note Consider using a [Python virtual environment](https://docs.python.org/3/library/venv.html) for this step.
  
  As an example, assuming use of [`vcpkg`](https://github.com/microsoft/vcpkg) for external dependency management with `VCPKG_ROOT=path/to/vcpkg` set as an environment variable, run
  
  ```bash
  pip install . --config-settings=cmake.args="--preset=pip-cuda" -v
  ```
  
  on the command line to build [`pbatoolkit`](https://pypi.org/project/pbatoolkit/) from source with GPU algorithms included. Additional environment variables (i.e. [`CUDA_PATH`](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html)) and/or CMake variables (i.e. [`CMAKE_CUDA_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html#variable:CMAKE_%3CLANG%3E_COMPILER)) may be required to be set in order for CMake to correctly discover and compile against your targeted local CUDA installation. Refer to [the CMake documentation](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html) for more details.
  
<div class="section_buttons">

| Previous           |           Next |
|:-------------------|---------------:|
| \ref configuration | \ref userguide |

</div>