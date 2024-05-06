#ifndef PBA_CORE_FEM_MESH_H
#define PBA_CORE_FEM_MESH_H

#include "pba/aliases.h"
#include "pba/math/Concepts.h"

namespace pba {
namespace fem {

template <math::PolynomialBasis P>
struct Mesh
{
    using PolynomialBasis = P;
    using Dims = P::Dims;
    using Order = P::Order;

    MatrixX X; ///< Dims x |Nodes| nodal positions
    IndexMatrixX E; ///< |#nodes per element| x |Elements| element nodal indices
    MatrixX S; ///< Dims x |Elements|*|#vertices per simplex - 1| linear mappings from mesh domain to reference simplex
    MatrixX W; ///< |P::Size| x |Elements|*|P::Size| Polynomial weights of each shape function
};

} // namespace fem
} // namespace pba

#endif // PBA_CORE_FEM_MESH_H