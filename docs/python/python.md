### Python

For a local installation, which builds from source, our Python bindings build relies on [Scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/index.html), which relies on CMake's [`install`](https://cmake.org/cmake/help/latest/command/install.html) mechanism. As such, you can configure the installation as you typically would when using the CMake CLI directly, by now passing the corresponding CMake arguments in `pip`'s `config-settings` parameter (refer to the [Scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/index.html) documentation for the relevant parameters). See our [pyinstall workflow](.github/workflows/pyinstall.yml) for working examples of building from source on Linux, MacOS and Windows. Then, assuming that external dependencies are found via CMake's [`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html), you can build and install our Python package [`pbatoolkit`](https://pypi.org/project/pbatoolkit/) locally and get the most up to date features. 

> Consider using a [Python virtual environment](https://docs.python.org/3/library/venv.html) for this step.

As an example, assuming use of [`vcpkg`](https://github.com/microsoft/vcpkg) for external dependency management with `VCPKG_ROOT=path/to/vcpkg` set as an environment variable, run

```bash
pip install . --config-settings=cmake.args="--preset=pip-cuda" -v
```

on the command line to build [`pbatoolkit`](https://pypi.org/project/pbatoolkit/) from source with GPU algorithms included. Additional environment variables (i.e. [`CUDA_PATH`](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html)) and/or CMake variables (i.e. [`CMAKE_CUDA_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html#variable:CMAKE_%3CLANG%3E_COMPILER)) may be required to be set in order for CMake to correctly discover and compile against your targeted local CUDA installation. Refer to [the CMake documentation](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html) for more details.


### CUDA

#### PyPI

[`pbatoolkit-gpu`](https://pypi.org/project/pbatoolkit-gpu/) (downloaded from PyPI) requires dynamically linking to an instance of the
- [CUDA 12 Runtime library](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-runtime), and your
- [CUDA Driver](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#driver-api). 

> Recall that the CUDA Runtime is [ABI compatible](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-driver-api/version-mixing-rules.html) up to major version.

On 64-bit Windows, these are `cudart64_12.dll` and `nvcuda.dll`. Ensure that they are discoverable via Windows' [DLL search order](https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order). We recommend adding `<drive>:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.<minor>\bin` (i.e. the binary folder of your CUDA Toolkit installation) to the `PATH` environment variable. The driver should already be on the search path by default after installation.

On Linux, they are `libcudart.so.12` and `libcuda.so.1`. Ensure that they are discoverable via Linux's [dynamic linker/loader](https://man7.org/linux/man-pages/man8/ld.so.8.html). If they are not already in a default search path, we recommend simply updating the library search path, i.e. `export LD_LIBRARY_PATH="path/to/driver/folder;path/to/runtime/folder;$LD_LIBRARY_PATH"`.

> MacOS does not support CUDA GPUs.

Our [`pbatoolkit-gpu`](https://pypi.org/project/pbatoolkit/) prebuilt binaries include [PTX](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#virtual-architectures), such that program load times will be delayed by [JIT](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#just-in-time-compilation) compilation on first use. [Verify](https://developer.nvidia.com/cuda-gpus) that your NVIDIA GPU supports [compute capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) at least 7.0. For example, only RTX 2060 up to 4090 chips are supported in the GeForce series. Runtime GPU performance may be constrained by the targeted compute capability.
