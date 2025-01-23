#ifndef PBAT_GPU_IMPL_GEOMETRY_AABB_H
#define PBAT_GPU_IMPL_GEOMETRY_AABB_H

#include "pbat/common/ConstexprFor.h"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/impl/common/Buffer.cuh"
#include "pbat/math/linalg/mini/Matrix.h"
#include "pbat/math/linalg/mini/UnaryOperations.h"

#include <cstddef>
#include <limits>
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
    void Construct(FLowerUpper&& fLowerUpper);

    template <auto kSimplexVerts>
    void Construct(
        common::Buffer<GpuScalar, kDims> const& V,
        common::Buffer<GpuIndex, kSimplexVerts> const& S);

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
template <auto kSimplexVerts>
inline void Aabb<kDims>::Construct(
    common::Buffer<GpuScalar, kDims> const& V,
    common::Buffer<GpuIndex, kSimplexVerts> const& S)
{
    using namespace pbat::math::linalg;
    auto const nBoxes = static_cast<GpuIndex>(S.Size());
    Resize(nBoxes);
    thrust::for_each(
        thrust::device,
        thrust::counting_iterator(0),
        thrust::counting_iterator(nBoxes),
        [b = b.Raw(), e = e.Raw(), V = V.Raw(), S = S.Raw()] PBAT_DEVICE(GpuIndex s) {
            auto inds = mini::FromBuffers<kSimplexVerts, 1>(S, s);
            auto P    = mini::FromBuffers(V, inds.Transpose());
            pbat::common::ForRange<0, kDims>([b, e, s, &P] PBAT_DEVICE<auto d>() {
                b[d][s] = Min(P.Row(d));
                e[d][s] = Max(P.Row(d));
            });
        });
}

template <auto kDims>
inline void Aabb<kDims>::Resize(GpuIndex nBoxes)
{
    b.Resize(nBoxes);
    e.Resize(nBoxes);
}

} // namespace geometry
} // namespace impl
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_IMPL_GEOMETRY_AABB_H
