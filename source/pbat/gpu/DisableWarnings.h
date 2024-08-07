#ifndef PBAT_GPU_DISABLE_WARNINGS_H
#define PBAT_GPU_DISABLE_WARNINGS_H

/**
 * @brief Disables the following warnings.
 *
 * 1. Unknown pragmas: Eigen uses the old nvcc "diag_suppress" directive, which has now been changed
 * to "nv_diag_suppress".
 * 2. Structure padded: thrust::cub always emits structure padded due to alignment specifier warning
 * 3. Conditional expression is constant: Eigen has many of these, which would be resolved with if constexpr, but Eigen needs to be backwards compatible with previous c++ standards.
 * 4. was declared deprecated: cuda-api-wrappers includes deprecated cuda features.
 *
 */
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

#endif // PBAT_GPU_DISABLE_WARNINGS_H