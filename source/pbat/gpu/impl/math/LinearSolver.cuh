#ifndef PBAT_GPU_IMPL_MATH_LINEARSOLVER_H
#define PBAT_GPU_IMPL_MATH_LINEARSOLVER_H

#include "HouseholderQR.cuh"

#include <algorithm>
#include <cstdio>
#include <cuda/api.hpp>
#include <cusolverDn.h>

#define CUSOLVER_CHECK(err)                                                        \
    {                                                                              \
        cusolverStatus_t err_ = (err);                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS)                                       \
        {                                                                          \
            std::printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("cusolver error");                            \
        }                                                                          \
    }

namespace pbat::gpu::impl::math {

class LinearSolver
{
  public:
    LinearSolver();

    LinearSolver(LinearSolver const&)            = delete;
    LinearSolver(LinearSolver&&)                 = delete;
    LinearSolver& operator=(LinearSolver const&) = delete;
    LinearSolver& operator=(LinearSolver&&)      = delete;

    bool IsInitialized() const;

    ~LinearSolver();

  private:
    cusolverDnHandle_t mCusolverHandle;
    bool mIsInitialized;
};

} // namespace pbat::gpu::impl::math

#endif // PBAT_GPU_IMPL_MATH_LINEARSOLVER_H
