// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Morton.cuh"
#include "pbat/HostDevice.h"
#include "pbat/common/ConstexprFor.h"
#include "pbat/math/linalg/mini/Mini.h"

#include <array>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace pbat {
namespace gpu {
namespace geometry {
namespace impl {

void Morton::Encode(
    Aabb<3> const& aabbs,
    Bound const& wmin,
    Bound const& wmax,
    common::Buffer<Code>& morton)
{
    auto const nBoxes = aabbs.Size();
    if (morton.Size() < nBoxes)
        morton.Resize(nBoxes);

    using namespace pbat::math::linalg;
    mini::SVector<GpuScalar, 3> sb  = wmin;
    mini::SVector<GpuScalar, 3> sbe = wmax - wmin;
    thrust::for_each(
        thrust::counting_iterator<GpuIndex>(0),
        thrust::counting_iterator<GpuIndex>(nBoxes),
        [sb, sbe, b = aabbs.b.Raw(), e = aabbs.e.Raw(), m = morton.Raw()] PBAT_DEVICE(GpuIndex i) {
            // Compute Morton code of the centroid of the bounding box of simplex s
            std::array<GpuScalar, 3> c{};
            pbat::common::ForRange<0, 3>(
                [&]<auto d>() { c[d] = (GpuScalar(0.5) * (b[d][i] + e[d][i]) - sb[d]) / sbe[d]; });
            using pbat::geometry::Morton3D;
            m[i] = Morton3D(c);
        });
}

} // namespace impl
} // namespace geometry
} // namespace gpu
} // namespace pbat