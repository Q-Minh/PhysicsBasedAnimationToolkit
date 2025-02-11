/**
 * @file DisableWarnings.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Disables irrelevant warnings in GPU sources.
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 * @details
 * Disables the following warnings.
 * - `Unknown pragmas`: [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) uses the old
 * nvcc `diag_suppress` directive, which has now been changed to `nv_diag_suppress`. See
 * [here](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-command-options-diagnostic-options).
 * - `Structure padded`: [thrust::cub](https://nvidia.github.io/cccl/cub/) always emits `structure
 * padded due to alignment specifier` warning.
 * - `Conditional expression is constant`:
 * [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) has many of these, which would be
 * resolved with `if constexpr`, but [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
 * needs to be backwards compatible with previous c++ standards.
 * - `was declared deprecated`: [cuda-api-wrappers](https://github.com/eyalroz/cuda-api-wrappers)
 * includes deprecated cuda features.
 */

#ifndef PBAT_GPU_DISABLEWARNINGS_H
#define PBAT_GPU_DISABLEWARNINGS_H

#if defined(__clang__)
    #pragma clang diagnostic ignored "-Wunknown-pragmas"
    #pragma clang diagnostic ignored "-Wpadded"
    #pragma clang diagnostic ignored "-Wdeprecated"
#elif defined(__GNUC__) || defined(__GNUG__)
    #pragma GCC diagnostic ignored "-Wunknown-pragmas"
    #pragma GCC diagnostic ignored "-Wpadded"
    #pragma GCC diagnostic ignored "-Wdeprecated"
#elif defined(_MSC_VER)
    #pragma warning(disable : 4068)
    #pragma warning(disable : 4324)
    #pragma warning(disable : 4127)
    #pragma warning(disable : 4996)
#endif

#if defined(CUDART_VERSION)
    #pragma nv_diag_suppress 3189
#endif

#endif // PBAT_GPU_DISABLEWARNINGS_H
