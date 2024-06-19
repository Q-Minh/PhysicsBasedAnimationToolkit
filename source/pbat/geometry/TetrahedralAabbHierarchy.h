#ifndef PBAT_GEOMETRY_TETRAHEDRAL_AABB_HIERARCHY_H
#define PBAT_GEOMETRY_TETRAHEDRAL_AABB_HIERARCHY_H

#include "AxisAlignedBoundingBox.h"
#include "BoundingVolumeHierarchy.h"

#include <pbat/Aliases.h>

namespace pbat {
namespace geometry {

class TetrahedralAabbHierarchy : public BoundingVolumeHierarchy<
                                     TetrahedralAabbHierarchy,
                                     AxisAlignedBoundingBox<3>,
                                     IndexVector<4>,
                                     3>
{
  public:
    TetrahedralAabbHierarchy() = default;
    TetrahedralAabbHierarchy(
        Eigen::Ref<MatrixX const> const& V,
        Eigen::Ref<IndexMatrixX const> const& T,
        std::size_t maxPointsInLeaf = 10ULL);

    PrimitiveType const& Primitive(Index p) const;

    Vector<3> PrimitiveLocation(PrimitiveType const& primitive) const;

    template <class RPrimitiveIndices>
    BoundingVolumeType BoundingVolumeOf(RPrimitiveIndices&& pinds) const;

    IndexMatrixX OverlappingPrimitives(TetrahedralAabbHierarchy const& tetbbh) const;

    Eigen::Ref<MatrixX const> V;
};

} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_TETRAHEDRAL_AABB_HIERARCHY_H
