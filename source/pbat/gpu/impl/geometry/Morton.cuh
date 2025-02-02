#ifndef PBAT_GPU_IMPL_GEOMETRY_MORTON_H
#define PBAT_GPU_IMPL_GEOMETRY_MORTON_H

#include "Aabb.cuh"
#include "pbat/geometry/Morton.h"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/impl/common/Buffer.cuh"
#include "pbat/math/linalg/mini/Matrix.h"

namespace pbat {
namespace gpu {
namespace impl {
namespace geometry {

class Morton
{
  public:
    using Bound = pbat::math::linalg::mini::SVector<GpuScalar, 3>;
    using Code  = pbat::geometry::MortonCodeType;

    Morton(std::size_t n);

    void Encode(Aabb<3> const& aabbs, Bound const& wmin, Bound const& wmax);

    common::Buffer<Code> codes;
};

} // namespace geometry
} // namespace impl
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_IMPL_GEOMETRY_MORTON_H
