
#ifndef PBAT_GEOMETRY_AXIS_ALIGNED_BOUNDING_BOX_H
#define PBAT_GEOMETRY_AXIS_ALIGNED_BOUNDING_BOX_H

#include <Eigen/Geometry>
#include <array>
#include <pbat/aliases.h>

namespace pbat {
namespace geometry {

template <int Dims>
class AxisAlignedBoundingBox : public Eigen::AlignedBox<Scalar, Dims>
{
  public:
    using BaseType = Eigen::AlignedBox<Scalar, Dims>;
    using SelfType = AxisAlignedBoundingBox;

    AxisAlignedBoundingBox()                = default;
    AxisAlignedBoundingBox(SelfType const&) = default;
    AxisAlignedBoundingBox(SelfType&&)      = default;
    SelfType& operator=(SelfType const&)    = default;
    SelfType& operator=(SelfType&&)         = default;

    AxisAlignedBoundingBox(BaseType const& box);
    AxisAlignedBoundingBox(BaseType&& box);
    AxisAlignedBoundingBox& operator=(BaseType const& box);
    AxisAlignedBoundingBox& operator=(BaseType&& box);
    AxisAlignedBoundingBox(Vector<Dims> const& min, Vector<Dims> const& max);

    template <class TDerived>
    AxisAlignedBoundingBox(Eigen::DenseBase<TDerived> const& P);
};

/**
 * @brief
 * @tparam Vector3Iterator
 * @param begin
 * @param end
 * @return
 */
template <class Vector3Iterator>
AxisAlignedBoundingBox aabb_of(Vector3Iterator begin, Vector3Iterator end)
{
    AxisAlignedBoundingBox aabb{};
    for (auto it = begin; it != end; ++it)
    {
        aabb.extend(*it);
    }
    Vector3 const e = Vector3::Ones() * eps();
    aabb.extend(aabb.min() - e);
    aabb.extend(aabb.max() + e);
    return aabb;
}

} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_AXIS_ALIGNED_BOUNDING_BOX_H
