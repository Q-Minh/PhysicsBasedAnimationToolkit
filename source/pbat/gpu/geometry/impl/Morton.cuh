#ifndef PBAT_GPU_GEOMETRY_IMPL_MORTON_H
#define PBAT_GPU_GEOMETRY_IMPL_MORTON_H

#include "Aabb.cuh"
#include "pbat/geometry/Morton.h"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/common/Buffer.cuh"
#include "pbat/math/linalg/mini/Matrix.h"

namespace pbat {
namespace gpu {
namespace geometry {
namespace impl {

class Morton
{
  public:
    using Bound = pbat::math::linalg::mini::SVector<GpuScalar, 3>;
    using Code  = pbat::geometry::MortonCodeType;

    static void Encode(
        Aabb<3> const& aabbs,
        Bound const& wmin,
        Bound const& wmax,
        common::Buffer<Code>& morton);
};

} // namespace impl
} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_IMPL_MORTON_H
