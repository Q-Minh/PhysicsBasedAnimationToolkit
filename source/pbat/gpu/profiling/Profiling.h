/**
 * @file Profiling.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Profiling utilities for host-side GPU code
 * @date 2025-02-19
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GPU_PROFILING_PROFILING_H
#define PBAT_GPU_PROFILING_PROFILING_H

#include "PhysicsBasedAnimationToolkitExport.h"

/**
 * @namespace pbat::gpu::profiling
 * @brief Namespace for host-side GPU profiling utilities
 */
namespace pbat::gpu::profiling {

/**
 * @brief Profiler for CUDA execution
 */
class CudaProfiler
{
  public:
    /**
     * @brief Construct a new CUDA Profiler object
     */
    PBAT_API CudaProfiler();
    /**
     * @brief Deleted copy constructor
     */
    PBAT_API CudaProfiler(const CudaProfiler&) = delete;
    /**
     * @brief Deleted copy assignment operator
     */
    PBAT_API CudaProfiler& operator=(const CudaProfiler&) = delete;
    /**
     * @brief Defaulted move constructor
     */
    [[maybe_unused]] PBAT_API CudaProfiler(CudaProfiler&&) noexcept = default;
    /**
     * @brief Defaulted move assignment operator
     */
    [[maybe_unused]] PBAT_API CudaProfiler& operator=(CudaProfiler&&) noexcept = default;
    /**
     * @brief Start profiling CUDA calls
     */
    PBAT_API void Start();
    /**
     * @brief Stop profiling CUDA calls
     */
    PBAT_API void Stop();
    /**
     * @brief Destroy the CUDA profiling context
     */
    PBAT_API ~CudaProfiler();

  private:
    void* mContext; ///< Tracy CUDA context
};

} // namespace pbat::gpu::profiling

#endif // PBAT_GPU_PROFILING_PROFILING_H
