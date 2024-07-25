#ifndef PBAT_GPU_GEOMETRY_AXIS_ALIGNED_BOUNDING_BOXES_CUH
#define PBAT_GPU_GEOMETRY_AXIS_ALIGNED_BOUNDING_BOXES_CUH

#include <thrust/device_vector.h>

namespace pbat {
namespace gpu {
namespace geometry {

class AxisAlignedBoundingBoxes
{
  public:
    using ScalarType = float;

    thrust::device_vector<ScalarType> bx, by, bz; ///< Box beginnings
    thrust::device_vector<ScalarType> ex, ey, ez; ///< Box endings
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_AXIS_ALIGNED_BOUNDING_BOXES_CUH