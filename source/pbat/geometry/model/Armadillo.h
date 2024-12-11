#ifndef PBAT_GEOMETRY_MODEL_ARMADILLO_H
#define PBAT_GEOMETRY_MODEL_ARMADILLO_H

#include "Enums.h"
#include "pbat/Aliases.h"

#include <utility>

namespace pbat {
namespace geometry {
namespace model {

std::pair<MatrixX, IndexMatrixX> Armadillo(EMesh mesh = EMesh::Tetrahedral, Index layer = Index(0));

} // namespace model
} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_MODEL_ARMADILLO_H