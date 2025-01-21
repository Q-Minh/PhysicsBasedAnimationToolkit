#ifndef PBAT_GPU_GEOMETRY_IMPL_BVH_KERNELS_CUH
#define PBAT_GPU_GEOMETRY_IMPL_BVH_KERNELS_CUH

#include "Bvh.cuh"
#include "pbat/HostDevice.h"
#include "pbat/common/Stack.h"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/common/SynchronizedList.cuh"

#include <array>
#include <assert.h>
#include <cuda/atomic>

namespace pbat {
namespace gpu {
namespace geometry {
namespace BvhImplKernels {

struct FLeafBoundingBoxes
{
    PBAT_DEVICE void operator()(auto s)
    {
        for (auto d = 0; d < 3; ++d)
        {
            auto bs  = leafBegin + s;
            b[d][bs] = x[d][inds[0][s]];
            e[d][bs] = x[d][inds[0][s]];
            for (auto m = 1; m < nSimplexVertices; ++m)
            {
                b[d][bs] = min(b[d][bs], x[d][inds[m][s]]);
                e[d][bs] = max(e[d][bs], x[d][inds[m][s]]);
            }
            b[d][bs] -= r;
            e[d][bs] += r;
        }
    }

    std::array<GpuScalar const*, 3> x;
    std::array<GpuIndex const*, 4> inds;
    int nSimplexVertices;
    std::array<GpuScalar*, 3> b;
    std::array<GpuScalar*, 3> e;
    GpuIndex leafBegin;
    GpuScalar r;
};

} // namespace BvhImplKernels
} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_IMPL_BVH_KERNELS_CUH