/**
 * @file MeshBoundary.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file contains functions to compute the boundary of a mesh.
 * @date 2025-02-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GEOMETRY_MESH_BOUNDARY_H
#define PBAT_GEOMETRY_MESH_BOUNDARY_H

#include "pbat/Aliases.h"

#include <tuple>

namespace pbat {
namespace geometry {

/**
 * @brief Obtains the boundary mesh of a simplex mesh.
 *
 * @note Only works for triangle (`C.rows()==3`) and tetrahedral (`C.rows()==4`) meshes.
 *
 * @param C The connectivity matrix of the mesh (i.e. the simplices)
 * @param n The number of vertices in the mesh. If -1, the number of vertices is computed from C.
 * @return A tuple containing the boundary vertices and the boundary facets
 */
std::tuple<IndexVectorX, IndexMatrixX>
SimplexMeshBoundary(IndexMatrixX const& C, Index n = Index(-1));

} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_MESH_BOUNDARY_H
