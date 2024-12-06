#ifndef PBAT_GEOMETRY_MESH_BOUNDARY_H
#define PBAT_GEOMETRY_MESH_BOUNDARY_H

#include "pbat/Aliases.h"

namespace pbat {
namespace geometry {

IndexMatrixX SimplexMeshBoundary(IndexMatrixX const& C, Index n = Index(-1));

} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_MESH_BOUNDARY_H
