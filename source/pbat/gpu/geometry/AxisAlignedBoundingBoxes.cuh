#ifndef PBAT_GPU_GEOMETRY_AXIS_ALIGNED_BOUNDING_BOXES_CUH
#define PBAT_GPU_GEOMETRY_AXIS_ALIGNED_BOUNDING_BOXES_CUH

#include "pbat/gpu/Aliases.h"

#include <thrust/device_vector.h>

namespace pbat {
namespace gpu {
namespace geometry {

class AxisAlignedBoundingBoxes
{
  public:
    thrust::device_vector<GpuScalar> bx, by, bz; ///< Box beginnings
    thrust::device_vector<GpuScalar> ex, ey, ez; ///< Box endings
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_AXIS_ALIGNED_BOUNDING_BOXES_CUH