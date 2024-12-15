#ifndef PBAT_GEOMETRY_MODEL_CUBE_H
#define PBAT_GEOMETRY_MODEL_CUBE_H

#include "Enums.h"
#include "pbat/Aliases.h"

#include <utility>

namespace pbat {
namespace geometry {
namespace model {

std::pair<MatrixX, IndexMatrixX> Cube(EMesh mesh = EMesh::Tetrahedral, int layer = 0);

} // namespace model
} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_MODEL_CUBE_H