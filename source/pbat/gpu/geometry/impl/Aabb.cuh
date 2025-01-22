#ifndef PBAT_GPU_GEOMETRY_IMPL_AABB_H
#define PBAT_GPU_GEOMETRY_IMPL_AABB_H

#include "pbat/common/ConstexprFor.h"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/common/Buffer.cuh"
#include "pbat/math/linalg/mini/Matrix.h"

#include <cstddef>
#include <limits>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace pbat {
namespace gpu {
namespace geometry {
namespace impl {

template <auto kDims>
struct Aabb
{
    Aabb() = default;
    Aabb(GpuIndex nBoxes) : b(nBoxes), e(nBoxes) {}
    void Resize(GpuIndex nBoxes);

    template <class FLowerUpper>
    void Construct(FLowerUpper&& fLowerUpper);

    auto Size() const { return b.Size(); }

    common::Buffer<GpuScalar, kDims> b, e;
};

template <auto kDims>
template <class FLowerUpper>
inline void Aabb<kDims>::Construct(FLowerUpper&& fLowerUpper)
{
    using namespace pbat::math::linalg;
    auto const nBoxes = static_cast<GpuIndex>(b.Size());
    thrust::for_each(
        thrust::device,
        thrust::counting_iterator<GpuIndex>(0),
        thrust::counting_iterator<GpuIndex>(nBoxes),
        [b           = b.Raw(),
         e           = e.Raw(),
         fLowerUpper = std::forward<FLowerUpper>(fLowerUpper)] PBAT_DEVICE(GpuIndex p) {
            pbat::common::ForRange<0, kDims>([b, e, p, LU = fLowerUpper(p)] PBAT_DEVICE<auto d>() {
                b[d][p] = LU(d, 0);
                e[d][p] = LU(d, 1);
            });
        });
}

template <auto kDims>
inline void Aabb<kDims>::Resize(GpuIndex nBoxes)
{
    b.Resize(nBoxes);
    e.Resize(nBoxes);
}

} // namespace impl
} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_IMPL_AABB_H
