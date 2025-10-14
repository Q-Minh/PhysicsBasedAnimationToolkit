#ifndef PBAT_GPU_IMPL_GEOMETRY_AABB_H
#define PBAT_GPU_IMPL_GEOMETRY_AABB_H

#include "pbat/common/ConstexprFor.h"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/impl/common/Buffer.cuh"
#include "pbat/math/linalg/mini/Matrix.h"
#include "pbat/math/linalg/mini/UnaryOperations.h"
#include "pbat/profiling/Profiling.h"

#include <algorithm>
#include <cstddef>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace pbat {
namespace gpu {
namespace impl {
namespace geometry {

template <auto kDims>
struct Aabb
{
    Aabb() = default;
    Aabb(GpuIndex nBoxes) : b(nBoxes), e(nBoxes) {}

    template <auto kSimplexVerts>
    Aabb(
        common::Buffer<GpuScalar, kDims> const& V,
        common::Buffer<GpuIndex, kSimplexVerts> const& S);

    void Resize(GpuIndex nBoxes);

    template <class FLowerUpper>
    void Construct(FLowerUpper&& fLowerUpper, GpuIndex begin = 0, GpuIndex end = -1);

    template <auto kSimplexVerts>
    void Construct(
        common::Buffer<GpuScalar, kDims> const& V,
        common::Buffer<GpuIndex, kSimplexVerts> const& S,
        GpuIndex begin = 0);

    GpuIndex Size() const { return static_cast<GpuIndex>(b.Size()); }

    common::Buffer<GpuScalar, kDims> b, e;
};

template <auto kDims>
template <auto kSimplexVerts>
inline Aabb<kDims>::Aabb(
    common::Buffer<GpuScalar, kDims> const& V,
    common::Buffer<GpuIndex, kSimplexVerts> const& S)
{
    Construct(V, S);
}

template <auto kDims>
template <class FLowerUpper>
inline void Aabb<kDims>::Construct(FLowerUpper&& fLowerUpper, GpuIndex begin, GpuIndex end)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.gpu.impl.geometry.Aabb.Construct");
    using namespace pbat::math::linalg;
    auto const nBoxes = static_cast<GpuIndex>(b.Size());
    end               = end < 0 ? nBoxes : end;
    thrust::for_each(
        thrust::device,
        thrust::counting_iterator<GpuIndex>(begin),
        thrust::counting_iterator<GpuIndex>(end),
        [begin,
         b           = b.Raw(),
         e           = e.Raw(),
         fLowerUpper = std::forward<FLowerUpper>(fLowerUpper)] PBAT_DEVICE(GpuIndex i) {
            mini::SMatrix<GpuScalar, kDims, 2> LU = fLowerUpper(i - begin);
            mini::ToBuffers(LU.Col(0), b, i);
            mini::ToBuffers(LU.Col(1), e, i);
        });
}

template <auto kDims>
template <auto kSimplexVerts>
inline void Aabb<kDims>::Construct(
    common::Buffer<GpuScalar, kDims> const& V,
    common::Buffer<GpuIndex, kSimplexVerts> const& S,
    GpuIndex begin)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.gpu.impl.geometry.Aabb.Construct");
    using namespace pbat::math::linalg;
    auto const nSimplices = static_cast<GpuIndex>(S.Size());
    if (Size() < nSimplices)
        Resize(nSimplices);
    auto const end = std::min(begin + nSimplices, Size());
    thrust::for_each(
        thrust::device,
        thrust::counting_iterator(begin),
        thrust::counting_iterator(end),
        [begin, b = b.Raw(), e = e.Raw(), V = V.Raw(), S = S.Raw()] PBAT_DEVICE(GpuIndex i) {
            auto s    = i - begin;
            auto inds = mini::FromBuffers<kSimplexVerts, 1>(S, s);
            auto P    = mini::FromBuffers(V, inds.Transpose());
            pbat::common::ForRange<0, kDims>([i, b, e, &P] PBAT_DEVICE<auto d>() {
                b[d][i] = Min(P.Row(d));
                e[d][i] = Max(P.Row(d));
            });
        });
}

template <auto kDims>
inline void Aabb<kDims>::Resize(GpuIndex nBoxes)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.gpu.impl.geometry.Aabb.Resize");
    b.Resize(nBoxes);
    e.Resize(nBoxes);
}

} // namespace geometry
} // namespace impl
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_IMPL_GEOMETRY_AABB_H
