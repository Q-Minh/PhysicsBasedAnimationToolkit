#ifndef PBAT_GPU_IMPL_MATH_HOUSEHOLDERQR_CUH
#define PBAT_GPU_IMPL_MATH_HOUSEHOLDERQR_CUH

#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/impl/common/Buffer.cuh"
#include "pbat/gpu/impl/common/Var.cuh"

#include <cublas_v2.h>
#include <cuda/api/stream.hpp>
#include <cusolverDn.h>

namespace pbat::gpu::impl::math {

class HouseholderQR
{
  public:
    HouseholderQR(cuda::stream_t stream = cuda::device::current::get().default_stream());
    
    ~HouseholderQR();

  private:
    cusolverDnHandle_t mCusolverHandle;
    cublasHandle_t mCublasHandle;
    cuda::stream_t mStream;
    common::Buffer<GpuScalar> mWorkspace;

    common::Buffer<GpuScalar> mQR;
    common::Buffer<GpuScalar> mTau;
    common::Var<int> mInfo;
};

} // namespace pbat::gpu::impl::math

#endif // PBAT_GPU_IMPL_MATH_HOUSEHOLDERQR_CUH
