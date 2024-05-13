#ifndef PBA_CORE_FEM_MASS_MATRIX_H
#define PBA_CORE_FEM_MASS_MATRIX_H

#include "Concepts.h"

namespace pba {
namespace fem {

template <CMesh TMesh, int Dims>
struct MassMatrix
{
  public:
    using MeshType              = TMesh;
    static int constexpr kDims  = Dims;
    using ElementType           = typename TMesh::ElementType;
    static int constexpr kOrder = 2 * ElementType::kOrder;

    MassMatrix(MeshType const& mesh, Scalar rho = 1.);

    template <class Derived>
    MassMatrix(MeshType const& mesh, Eigen::DenseBase<Derived> const& rho);

    VectorX rho; ///< |#elements| x 1 piecewise constant mass density
    MatrixX Me;  ///< |ElementType::Nodes|x|ElementType::Nodes * |#elements| element mass matrices
};

template <CMesh TMesh, int Dims>
inline MassMatrix<TMesh, Dims>::MassMatrix(MeshType const& mesh, Scalar rho)
    : rho(VectorX::Constant(rho, mesh.E.cols())),
      Me(ElementType::kNodes, ElementType::kNodes * mesh.E.cols())
{
}

template <CMesh TMesh, int Dims>
template <class Derived>
inline MassMatrix<TMesh, Dims>::MassMatrix(
    MeshType const& mesh,
    Eigen::DenseBase<Derived> const& rho)
    : rho(rho), Me(ElementType::kNodes, ElementType::kNodes * mesh.E.cols())
{
}

} // namespace fem
} // namespace pba
#endif // PBA_CORE_FEM_MASS_MATRIX_H