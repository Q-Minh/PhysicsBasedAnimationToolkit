\page dependencies Dependencies

See [`vcpkg.json`](https://github.com/Q-Minh/PhysicsBasedAnimationToolkit/blob/master/vcpkg.json) for a versioned list of our external dependencies, available via [vcpkg](https://github.com/microsoft/vcpkg).

\note Use of [vcpkg](https://github.com/microsoft/vcpkg) is not mandatory, as long as external dependencies have compatible versions and are discoverable by CMake's [`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html) mechanism.

### CUDA

#### PyPI

[`pbatoolkit-gpu`](https://pypi.org/project/pbatoolkit-gpu/) (downloaded from PyPI) requires dynamically linking to an instance of the
- [CUDA 12 Runtime library](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-runtime), and your
- [CUDA Driver](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#driver-api). 

\note Recall that the CUDA Runtime is [ABI compatible](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-driver-api/version-mixing-rules.html) up to major version.

On 64-bit Windows, these are `cudart64_12.dll` and `nvcuda.dll`. Ensure that they are discoverable via Windows' [DLL search order](https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order). We recommend adding `<drive>:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.<minor>\bin` (i.e. the binary folder of your CUDA Toolkit installation) to the `PATH` environment variable. The driver should already be on the search path by default after installation.

On Linux, they are `libcudart.so.12` and `libcuda.so.1`. Ensure that they are discoverable via Linux's [dynamic linker/loader](https://man7.org/linux/man-pages/man8/ld.so.8.html). If they are not already in a default search path, we recommend simply updating the library search path, i.e. `export LD_LIBRARY_PATH="path/to/driver/folder;path/to/runtime/folder;$LD_LIBRARY_PATH"`.

\note MacOS does not support CUDA GPUs.

Our [`pbatoolkit-gpu`](https://pypi.org/project/pbatoolkit/) prebuilt binaries include [PTX](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#virtual-architectures), such that program load times will be delayed by [JIT](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#just-in-time-compilation) compilation on first use. [Verify](https://developer.nvidia.com/cuda-gpus) that your NVIDIA GPU supports [compute capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) at least 7.0. For example, only RTX 2060 up to 4090 chips are supported in the GeForce series. Runtime GPU performance may be constrained by the targeted compute capability.

<div class="section_buttons">

| Previous   |               Next |
|:-----------|-------------------:|
| \ref build | \ref configuration |

</div>